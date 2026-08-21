"""Protocol adapters for BACRA V14 atlas and supervision stages.

The numerical modules deliberately operate on immutable in-memory objects.
This module owns the deeper protocol seam between persisted workspace tables
and those objects.  It keeps three distinctions explicit:

* a complete inventory of cell probes is different from all probes being
  labelable;
* low margin and normalized-Jacobian sensitivity are risks, not physical
  invalidity; and
* primary point labels and chart-expert labels are separate supervision
  contracts, even when they originate at the same physical XYZ.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .canonical_atlas import (
    AtlasCandidate,
    AtlasTaskNode,
    ContinuationOutcome,
    make_predictor_corrector_continuation,
)
from .workspace_atlas import normalized_jacobian_metrics
from .workspace_dataset import (
    SupervisionKind,
    SupervisionPriority,
    SupervisionRecord,
)
from .workspace_reach import CellKey


BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
XYZ_COLUMNS = ("x_m", "y_m", "z_m")


@dataclass(frozen=True)
class WorkspaceAtlasFramePolicy:
    maximum_residual_mm: float = 3.0
    gold_margin_deg: float = 1.5
    low_margin_deg: float = 0.25
    ill_conditioned_normalized_sigma3_m: float = 0.05
    ill_conditioned_normalized_kappa: float = 100.0
    candidate_cluster_deg: float = 0.5
    maximum_measure_parent_candidates: int = 4
    minimum_measure_probes_per_cell: int = 2

    def __post_init__(self) -> None:
        positive = (
            self.maximum_residual_mm,
            self.gold_margin_deg,
            self.low_margin_deg,
            self.ill_conditioned_normalized_sigma3_m,
            self.ill_conditioned_normalized_kappa,
            self.candidate_cluster_deg,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("workspace atlas frame thresholds must be finite and positive")
        if self.low_margin_deg >= self.gold_margin_deg:
            raise ValueError("low_margin_deg must be below gold_margin_deg")
        if int(self.maximum_measure_parent_candidates) < 1:
            raise ValueError("maximum_measure_parent_candidates must be positive")
        if int(self.minimum_measure_probes_per_cell) < 1:
            raise ValueError("minimum_measure_probes_per_cell must be positive")


@dataclass(frozen=True)
class WorkspaceAtlasFrames:
    task_nodes: pd.DataFrame
    candidates: pd.DataFrame
    task_edges: pd.DataFrame
    candidate_diagnostics: pd.DataFrame


@dataclass(frozen=True)
class DenseCorrectionResult:
    records: tuple[SupervisionRecord, ...]
    audit: pd.DataFrame


def _bounds(environment: Any) -> np.ndarray:
    values = np.asarray(environment.bounds, dtype=float)
    if values.shape != (6, 2) or not np.isfinite(values).all():
        raise ValueError("environment bounds must be finite shape (6, 2)")
    if np.any(values[:, 0] >= values[:, 1]):
        raise ValueError("environment bounds must be strictly ordered")
    return values


def _inside(beta: np.ndarray, bounds: np.ndarray) -> bool:
    return bool(
        np.isfinite(beta).all()
        and np.all(beta >= bounds[:, 0] - 1.0e-12)
        and np.all(beta <= bounds[:, 1] + 1.0e-12)
    )


def _margin(beta: np.ndarray, bounds: np.ndarray) -> tuple[float, float]:
    distance = np.minimum(beta - bounds[:, 0], bounds[:, 1] - beta)
    span = bounds[:, 1] - bounds[:, 0]
    return float(np.min(np.rad2deg(distance))), float(np.min(distance / span))


def _candidate_metrics(
    environment: Any,
    *,
    beta: np.ndarray,
    target_xyz: np.ndarray,
    policy: WorkspaceAtlasFramePolicy,
) -> Mapping[str, Any]:
    bounds = _bounds(environment)
    actual_bounds = _inside(beta, bounds)
    achieved = np.asarray(environment.fk(beta.reshape(1, 6)), dtype=float).reshape(-1, 3)[0]
    residual_mm = float(np.linalg.norm(achieved - target_xyz) * 1000.0)
    margin_deg, normalized_margin = _margin(beta, bounds)
    jacobian = np.asarray(environment.jacobian(beta), dtype=float)
    sigma3, kappa = normalized_jacobian_metrics(jacobian, bounds)
    risk_flags: list[str] = []
    if margin_deg < policy.low_margin_deg:
        risk_flags.append("low_margin")
    if (
        sigma3 < policy.ill_conditioned_normalized_sigma3_m
        or kappa > policy.ill_conditioned_normalized_kappa
    ):
        risk_flags.append("ill_conditioned")
    if not actual_bounds or residual_mm > policy.maximum_residual_mm:
        quality = "Reject"
    elif margin_deg >= policy.gold_margin_deg:
        quality = "Gold"
    else:
        # Low margin is a sensitivity class, not a claim of physical invalidity.
        quality = "Silver"
    return {
        "achieved_xyz_m": achieved,
        "residual_mm": residual_mm,
        "min_margin_deg": margin_deg,
        "normalized_min_margin": normalized_margin,
        "normalized_sigma3_m": sigma3,
        "normalized_kappa": kappa,
        "risk_flags": tuple(sorted(risk_flags)),
        "actual_bounds": actual_bounds,
        "quality": quality,
    }


def prepare_workspace_atlas_frames(
    probes: pd.DataFrame,
    representative_candidates: pd.DataFrame,
    representative_edges: pd.DataFrame,
    environment: Any,
    *,
    policy: WorkspaceAtlasFramePolicy | None = None,
) -> WorkspaceAtlasFrames:
    """Expand representative cells into an audited multi-probe task graph.

    Every registered probe becomes a distinct task node.  Representative
    candidates remain on the representative node, while each measure probe
    receives its exact capability solution plus bounded corrections from the
    representative candidate families.  A cell is marked measure-complete
    only from the probe inventory; atlas success is still required separately.
    """

    active = WorkspaceAtlasFramePolicy() if policy is None else policy
    required_probe = {
        "probe_id", "physical_point_id", "node_id", "cell_id",
        "cell_level_mm", "cell_ix", "cell_iy", "cell_iz",
        "is_representative", "is_measure_probe", *XYZ_COLUMNS, *BETA_COLUMNS,
    }
    missing = sorted(required_probe - set(probes.columns))
    if missing:
        raise ValueError(f"workspace probe table missing columns: {missing}")
    if probes["probe_id"].astype(str).duplicated().any():
        raise ValueError("workspace probe IDs must be unique")
    if len(probes) == 0:
        raise ValueError("workspace probe table must be non-empty")

    ordered = probes.sort_values(
        ["node_id", "is_representative", "probe_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    ordered["task_node_id"] = np.arange(len(ordered), dtype=np.int64)
    task_id_by_probe = dict(
        zip(ordered["probe_id"].astype(str), ordered["task_node_id"].astype(int), strict=True)
    )
    rep_task_by_cell_node: dict[int, int] = {}
    complete_by_cell: dict[str, bool] = {}
    for cell_id, group in ordered.groupby("cell_id", sort=False):
        representatives = group[group["is_representative"].astype(bool)]
        if len(representatives) != 1:
            raise ValueError(f"cell {cell_id!r} must have exactly one representative")
        parent_ids = set(group["node_id"].astype(int))
        if len(parent_ids) != 1:
            raise ValueError(f"cell {cell_id!r} mixes representative node identities")
        parent_id = next(iter(parent_ids))
        rep_task_by_cell_node[parent_id] = int(representatives.iloc[0]["task_node_id"])
        measure_count = int(group["is_measure_probe"].astype(bool).sum())
        complete_by_cell[str(cell_id)] = measure_count >= int(
            active.minimum_measure_probes_per_cell
        )

    task_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    exact_metrics_by_task: dict[int, Mapping[str, Any]] = {}
    for row in ordered.itertuples(index=False):
        beta = np.asarray([getattr(row, name) for name in BETA_COLUMNS], dtype=float)
        xyz = np.asarray([getattr(row, name) for name in XYZ_COLUMNS], dtype=float)
        metrics = _candidate_metrics(
            environment, beta=beta, target_xyz=xyz, policy=active
        )
        task_node_id = int(row.task_node_id)
        exact_metrics_by_task[task_node_id] = metrics
        task_rows.append(
            {
                "task_node_id": task_node_id,
                "task_id": str(row.probe_id),
                "physical_point_id": str(row.physical_point_id),
                "x_m": float(xyz[0]),
                "y_m": float(xyz[1]),
                "z_m": float(xyz[2]),
                "cell_level_mm": int(row.cell_level_mm),
                "cell_ix": int(row.cell_ix),
                "cell_iy": int(row.cell_iy),
                "cell_iz": int(row.cell_iz),
                "is_representative": bool(row.is_representative),
                "is_measure_probe": bool(row.is_measure_probe),
                "cell_measure_complete": complete_by_cell[str(row.cell_id)],
                "physical_status": (
                    "valid" if bool(metrics["actual_bounds"]) else "physically_invalid"
                ),
                "risk_flags": "|".join(metrics["risk_flags"]),
                "core_safe": not bool(metrics["risk_flags"]),
                "source_parent_node_id": int(row.node_id),
            }
        )
        diagnostic_rows.append(
            {
                "task_node_id": task_node_id,
                "candidate_id": "capability_exact",
                "source": "registered_capability_exact",
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "achieved_xyz_m"
                },
            }
        )

    edge_pairs: set[tuple[int, int]] = set()
    for _cell_id, group in ordered.groupby("cell_id", sort=False):
        # Every pair of registered probes is checked.  A representative-star
        # would be a tree and therefore provide no cell-internal fundamental
        # cycle; with three or more probes this clique creates the independent
        # paths required to test section consistency instead of merely
        # asserting it from center-to-probe corrections.
        task_node_ids = sorted(group["task_node_id"].astype(int))
        for left_index, left in enumerate(task_node_ids):
            for right in task_node_ids[left_index + 1 :]:
                edge_pairs.add((left, right))
    required_edge_columns = {"left_node_id", "right_node_id"}
    missing_edges = sorted(required_edge_columns - set(representative_edges.columns))
    if missing_edges:
        raise ValueError(f"representative edge table missing columns: {missing_edges}")
    for row in representative_edges.itertuples(index=False):
        left_parent, right_parent = int(row.left_node_id), int(row.right_node_id)
        if left_parent not in rep_task_by_cell_node or right_parent not in rep_task_by_cell_node:
            raise ValueError("representative edge references an unknown selected cell")
        edge_pairs.add(
            tuple(
                sorted(
                    (
                        rep_task_by_cell_node[left_parent],
                        rep_task_by_cell_node[right_parent],
                    )
                )
            )
        )
    edge_frame = pd.DataFrame.from_records(
        [
            {"left_node_id": left, "right_node_id": right, "adjacency": "registered"}
            for left, right in sorted(edge_pairs)
        ],
        columns=["left_node_id", "right_node_id", "adjacency"],
    )

    candidate_rows: list[dict[str, Any]] = []

    def append_candidate(
        *, task_node_id: int, candidate_id: str, beta: np.ndarray,
        target_xyz: np.ndarray, source: str,
    ) -> None:
        metrics = _candidate_metrics(
            environment, beta=np.asarray(beta, dtype=float), target_xyz=target_xyz, policy=active
        )
        candidate_rows.append(
            {
                "task_node_id": int(task_node_id),
                "candidate_id": str(candidate_id),
                "source": source,
                "residual_mm": float(metrics["residual_mm"]),
                "min_margin_deg": float(metrics["min_margin_deg"]),
                "normalized_min_margin": float(metrics["normalized_min_margin"]),
                "posture_cost": float(np.linalg.norm(beta)),
                "condition_number": float(metrics["normalized_kappa"]),
                "normalized_sigma3_m": float(metrics["normalized_sigma3_m"]),
                "quality": str(metrics["quality"]),
                "solver_success": str(metrics["quality"]) != "Reject",
                "actual_bounds": bool(metrics["actual_bounds"]),
                "risk_flags": "|".join(metrics["risk_flags"]),
                **{
                    name: float(np.asarray(beta, dtype=float)[index])
                    for index, name in enumerate(BETA_COLUMNS)
                },
            }
        )

    # Every actual FK probe is legitimate reach evidence.  It is admitted as
    # Silver when close to a joint limit, with the risk retained explicitly.
    for row in ordered.itertuples(index=False):
        append_candidate(
            task_node_id=int(row.task_node_id),
            candidate_id="capability_exact",
            beta=np.asarray([getattr(row, name) for name in BETA_COLUMNS], dtype=float),
            target_xyz=np.asarray([getattr(row, name) for name in XYZ_COLUMNS], dtype=float),
            source="registered_capability_exact",
        )

    accepted = representative_candidates[
        representative_candidates["quality_class"].astype(str).isin(("Gold", "Silver"))
        & representative_candidates["solver_success"].astype(bool)
    ].copy()
    rep_candidates_by_parent: dict[int, list[dict[str, Any]]] = {}
    for parent_node_id, group in accepted.groupby("node_id", sort=True):
        representative_task_id = rep_task_by_cell_node.get(int(parent_node_id))
        if representative_task_id is None:
            raise ValueError("representative candidate references an unknown selected cell")
        target_row = ordered[ordered["task_node_id"].eq(representative_task_id)].iloc[0]
        target_xyz = target_row.loc[list(XYZ_COLUMNS)].to_numpy(dtype=float)
        parent_rows: list[dict[str, Any]] = []
        for source_row in group.sort_values(
            ["quality_class", "minimum_margin_deg", "residual_mm", "candidate_id"],
            ascending=[True, False, True, True],
            kind="stable",
        ).itertuples(index=False):
            beta = np.asarray([getattr(source_row, name) for name in BETA_COLUMNS], dtype=float)
            candidate_id = f"teacher_{source_row.candidate_id}"
            append_candidate(
                task_node_id=representative_task_id,
                candidate_id=candidate_id,
                beta=beta,
                target_xyz=target_xyz,
                source="representative_diversity_search",
            )
            parent_rows.append({"candidate_id": candidate_id, "beta": beta})
        rep_candidates_by_parent[int(parent_node_id)] = parent_rows

    continuation = make_predictor_corrector_continuation(
        environment, residual_tolerance_mm=active.maximum_residual_mm
    )
    task_row_by_id = {int(row["task_node_id"]): row for row in task_rows}
    for row in ordered.itertuples(index=False):
        if bool(row.is_representative):
            continue
        task_node_id = int(row.task_node_id)
        target_xyz = np.asarray([getattr(row, name) for name in XYZ_COLUMNS], dtype=float)
        target = AtlasTaskNode(task_node_id, target_xyz)
        parent_rows = rep_candidates_by_parent.get(int(row.node_id), ())
        for parent_index, parent in enumerate(
            tuple(parent_rows)[: int(active.maximum_measure_parent_candidates)]
        ):
            source = AtlasCandidate(
                node_id=rep_task_by_cell_node[int(row.node_id)],
                candidate_id=str(parent["candidate_id"]),
                beta_rad=np.asarray(parent["beta"], dtype=float),
                residual_mm=0.0,
                min_margin_deg=1.0,
                normalized_min_margin=0.1,
                quality="Silver",
            )
            outcome = continuation(source, target)
            if not isinstance(outcome, ContinuationOutcome):
                raise TypeError("predictor-corrector continuation returned an invalid result")
            if not outcome.success or not outcome.actual_bounds:
                continue
            append_candidate(
                task_node_id=task_node_id,
                candidate_id=f"parent_{parent_index:02d}_{parent['candidate_id']}",
                beta=outcome.beta_rad,
                target_xyz=target_xyz,
                source="representative_parent_correction",
            )

    candidates = pd.DataFrame.from_records(candidate_rows)
    if candidates.duplicated(["task_node_id", "candidate_id"]).any():
        raise RuntimeError("atlas candidate identities must be unique within each task node")
    # Keep Reject rows in diagnostics, but do not let them inflate graph
    # candidates or candidate-family counts.
    diagnostics = candidates.copy()
    candidates = candidates[
        candidates["quality"].isin(("Gold", "Silver"))
        & candidates["solver_success"].astype(bool)
        & candidates["actual_bounds"].astype(bool)
    ].reset_index(drop=True)
    if set(candidates["task_node_id"].astype(int)) != set(ordered["task_node_id"].astype(int)):
        raise RuntimeError("every registered task probe must retain a valid exact candidate")
    return WorkspaceAtlasFrames(
        task_nodes=pd.DataFrame.from_records(task_rows),
        candidates=candidates,
        task_edges=edge_frame,
        candidate_diagnostics=diagnostics,
    )


def _cell_from_row(row: Any) -> CellKey:
    return CellKey(
        int(row.cell_level_mm), int(row.cell_ix), int(row.cell_iy), int(row.cell_iz)
    )


def supervision_records_from_atlas_frames(
    *,
    task_probes: pd.DataFrame,
    candidates: pd.DataFrame,
    primary_partition: pd.DataFrame,
    product_edges: pd.DataFrame,
) -> tuple[SupervisionRecord, ...]:
    """Build primary, expert and stateful contracts from persisted atlas rows.

    The output intentionally contains both static contracts.  The downstream
    materializer selects the representation-specific view and counts every row
    it actually supervises.  No FK-residual tie-break is used to assign a
    primary branch.
    """

    required_probe = {
        "task_probe_id", "task_node_id", "physical_point_id", "chart_id",
        "selected_candidate_id", "labelable", "cell_level_mm", "cell_ix",
        "cell_iy", "cell_iz", *XYZ_COLUMNS,
    }
    missing = sorted(required_probe - set(task_probes.columns))
    if missing:
        raise ValueError(f"atlas task-probe table missing columns: {missing}")
    required_candidate = {"task_node_id", "candidate_id", *BETA_COLUMNS}
    missing = sorted(required_candidate - set(candidates.columns))
    if missing:
        raise ValueError(f"atlas candidate table missing columns: {missing}")
    candidate_by_key = {
        (int(row.task_node_id), str(row.candidate_id)): row
        for row in candidates.itertuples(index=False)
    }
    partition_by_node = {
        int(row.task_node_id): row for row in primary_partition.itertuples(index=False)
    }
    selected_by_chart_node: dict[tuple[str, int], tuple[Any, Any]] = {}
    records: list[SupervisionRecord] = []
    primary_points: set[str] = set()
    for probe in task_probes[task_probes["labelable"].astype(bool)].itertuples(index=False):
        if pd.isna(probe.task_node_id) or not probe.chart_id or not probe.selected_candidate_id:
            continue
        node_id = int(probe.task_node_id)
        chart_id = str(probe.chart_id)
        key = (node_id, str(probe.selected_candidate_id))
        candidate = candidate_by_key.get(key)
        if candidate is None:
            raise ValueError(f"task probe selection has no candidate row: {key}")
        partition = partition_by_node.get(node_id)
        assigned = None if partition is None else partition.assigned_section_id
        is_primary = bool(assigned is not None and str(assigned) == chart_id)
        point_id = str(probe.physical_point_id)
        if is_primary:
            if point_id in primary_points:
                raise ValueError("primary atlas contract must be single-valued per physical point")
            primary_points.add(point_id)
        beta = np.asarray([getattr(candidate, name) for name in BETA_COLUMNS], dtype=float)
        xyz = np.asarray([getattr(probe, name) for name in XYZ_COLUMNS], dtype=float)
        quality = str(getattr(candidate, "quality", "Gold"))
        residual = float(getattr(candidate, "residual_mm", 0.0))
        actual_bounds = bool(getattr(candidate, "actual_bounds", True))
        record = SupervisionRecord(
            record_id=f"static:{probe.task_probe_id}",
            kind=SupervisionKind.STATIC,
            physical_point_id=point_id,
            chart_id=chart_id,
            cell=_cell_from_row(probe),
            xyz_m=xyz,
            beta_rad=beta,
            is_primary=is_primary,
            required=is_primary,
            priority=(
                SupervisionPriority.BASE_COVERAGE
                if is_primary
                else SupervisionPriority.CHART_EXPERT
            ),
            source_family="atlas_task_probe",
            sample_weight=1.0 if quality == "Gold" else 0.5,
            quality_class=quality,
            residual_mm=residual,
            actual_bounds=actual_bounds,
        )
        records.append(record)
        selected_by_chart_node[(chart_id, node_id)] = (probe, candidate)

    for edge in product_edges.itertuples(index=False):
        left_node = int(edge.left_task_node_id)
        right_node = int(edge.right_task_node_id)
        left_candidate_id = str(edge.left_candidate_id)
        right_candidate_id = str(edge.right_candidate_id)
        for chart_id in sorted(
            chart
            for chart, node_id in selected_by_chart_node
            if node_id == left_node and (chart, right_node) in selected_by_chart_node
        ):
            left_probe, left_candidate = selected_by_chart_node[(chart_id, left_node)]
            right_probe, right_candidate = selected_by_chart_node[(chart_id, right_node)]
            if (
                str(left_candidate.candidate_id) != left_candidate_id
                or str(right_candidate.candidate_id) != right_candidate_id
            ):
                continue
            for source_probe, source_candidate, target_probe, target_candidate in (
                (left_probe, left_candidate, right_probe, right_candidate),
                (right_probe, right_candidate, left_probe, left_candidate),
            ):
                target_beta = np.asarray(
                    [getattr(target_candidate, name) for name in BETA_COLUMNS], dtype=float
                )
                previous_beta = np.asarray(
                    [getattr(source_candidate, name) for name in BETA_COLUMNS], dtype=float
                )
                target_xyz = np.asarray(
                    [getattr(target_probe, name) for name in XYZ_COLUMNS], dtype=float
                )
                records.append(
                    SupervisionRecord(
                        record_id=(
                            f"stateful:{chart_id}:{source_probe.task_probe_id}"
                            f"->{target_probe.task_probe_id}"
                        ),
                        kind=SupervisionKind.STATEFUL,
                        physical_point_id=str(target_probe.physical_point_id),
                        chart_id=chart_id,
                        cell=_cell_from_row(target_probe),
                        xyz_m=target_xyz,
                        beta_rad=target_beta,
                        previous_beta_rad=previous_beta,
                        is_primary=False,
                        required=True,
                        priority=SupervisionPriority.BASE_COVERAGE,
                        source_family="atlas_robust_transition",
                        source_probe_id=str(source_probe.task_probe_id),
                        target_probe_id=str(target_probe.task_probe_id),
                        branch_id=chart_id,
                        sample_weight=1.0,
                    )
                )
    return tuple(records)


def supervision_records_frame(records: Sequence[SupervisionRecord]) -> pd.DataFrame:
    """Primitive, lossless persisted projection of supervision records."""

    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "record_id": record.record_id,
            "kind": record.kind.value,
            "physical_point_id": record.physical_point_id,
            "chart_id": record.chart_id,
            "cell_level_mm": record.cell.level_mm,
            "cell_ix": record.cell.ix,
            "cell_iy": record.cell.iy,
            "cell_iz": record.cell.iz,
            "is_primary": record.is_primary,
            "required": record.required,
            "priority": int(record.priority),
            "source_family": record.source_family,
            "source_probe_id": record.source_probe_id,
            "target_probe_id": record.target_probe_id,
            "branch_id": record.branch_id,
            "sample_weight": record.sample_weight,
            "quality_class": record.quality_class,
            "residual_mm": record.residual_mm,
            "actual_bounds": record.actual_bounds,
            "macroblock_id": record.macroblock_id,
            "split_role": None if record.split_role is None else record.split_role.value,
            **dict(zip(XYZ_COLUMNS, record.xyz_m, strict=True)),
            **dict(zip(BETA_COLUMNS, record.beta_rad, strict=True)),
        }
        if record.previous_beta_rad is not None:
            row.update(
                {
                    f"previous_beta{index}_rad": float(value)
                    for index, value in enumerate(record.previous_beta_rad, start=1)
                }
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def correct_static_targets_from_primary_sections(
    targets: pd.DataFrame,
    *,
    task_probes: pd.DataFrame,
    candidates: pd.DataFrame,
    primary_partition: pd.DataFrame,
    environment: Any,
    source_family: str,
    priority: SupervisionPriority,
    maximum_rows: int,
    parent_gap_max_deg: float = 1.0,
    minimum_successful_parents: int = 2,
    policy: WorkspaceAtlasFramePolicy | None = None,
) -> DenseCorrectionResult:
    """Correct new XYZ targets from two independently registered chart parents.

    The spatial primary partition fixes the chart before any correction is
    attempted.  FK residual is used only as a label-quality metric.  Targets
    whose parents land on non-stitchable beta values are retained in the audit
    as chart-boundary evidence and never enter the static regression rows.
    """

    active = WorkspaceAtlasFramePolicy() if policy is None else policy
    limit = int(maximum_rows)
    if limit < 0:
        raise ValueError("maximum_rows must be non-negative")
    if int(minimum_successful_parents) < 2:
        raise ValueError("minimum_successful_parents must be at least two")
    required_target = {"physical_point_id", *XYZ_COLUMNS}
    missing = sorted(required_target - set(targets.columns))
    if missing:
        raise ValueError(f"dense correction targets missing columns: {missing}")
    if targets["physical_point_id"].astype(str).duplicated().any():
        raise ValueError("dense correction physical_point_id must be unique")

    candidate_by_key = {
        (int(row.task_node_id), str(row.candidate_id)): row
        for row in candidates.itertuples(index=False)
    }
    partition_by_node = {
        int(row.task_node_id): row for row in primary_partition.itertuples(index=False)
    }
    anchors_by_cell: dict[CellKey, list[dict[str, Any]]] = {}
    for probe in task_probes[task_probes["labelable"].astype(bool)].itertuples(index=False):
        if pd.isna(probe.task_node_id) or not probe.chart_id or not probe.selected_candidate_id:
            continue
        node_id = int(probe.task_node_id)
        partition = partition_by_node.get(node_id)
        if partition is None or pd.isna(partition.assigned_section_id):
            continue
        chart_id = str(partition.assigned_section_id)
        if chart_id != str(probe.chart_id):
            continue
        candidate = candidate_by_key.get((node_id, str(probe.selected_candidate_id)))
        if candidate is None:
            raise ValueError("primary task probe references a missing atlas candidate")
        cell = _cell_from_row(probe)
        anchors_by_cell.setdefault(cell, []).append(
            {
                "task_node_id": node_id,
                "task_probe_id": str(probe.task_probe_id),
                "chart_id": chart_id,
                "xyz": np.asarray([getattr(probe, name) for name in XYZ_COLUMNS], dtype=float),
                "beta": np.asarray([getattr(candidate, name) for name in BETA_COLUMNS], dtype=float),
                "candidate_id": str(candidate.candidate_id),
            }
        )
    continuation = make_predictor_corrector_continuation(
        environment, residual_tolerance_mm=active.maximum_residual_mm
    )
    bounds = _bounds(environment)
    accepted_records: list[SupervisionRecord] = []
    audit_rows: list[dict[str, Any]] = []
    ordered = targets.sort_values("physical_point_id", kind="stable")
    for target in ordered.itertuples(index=False):
        if len(accepted_records) >= limit:
            break
        xyz = np.asarray([getattr(target, name) for name in XYZ_COLUMNS], dtype=float)
        indices = np.floor(xyz / 0.010).astype(int)
        cell = CellKey(10, int(indices[0]), int(indices[1]), int(indices[2]))
        anchors = anchors_by_cell.get(cell, ())
        chart_ids = {str(anchor["chart_id"]) for anchor in anchors}
        if len(chart_ids) != 1:
            audit_rows.append(
                {
                    "physical_point_id": str(target.physical_point_id),
                    "status": "missing_or_indeterminate_primary_section",
                    "successful_parent_count": 0,
                    "parent_gap_max_deg": None,
                }
            )
            continue
        ordered_anchors = sorted(
            anchors,
            key=lambda anchor: (
                float(np.linalg.norm(anchor["xyz"] - xyz)),
                anchor["task_probe_id"],
            ),
        )
        # Use distinct task probes.  More than two are attempted so one
        # numerical failure does not turn a well-supported cell into a false
        # negative.
        attempted = ordered_anchors[: max(int(minimum_successful_parents), 4)]
        outcomes: list[tuple[dict[str, Any], ContinuationOutcome]] = []
        destination = AtlasTaskNode(-1, xyz)
        for anchor in attempted:
            source = AtlasCandidate(
                node_id=int(anchor["task_node_id"]),
                candidate_id=str(anchor["candidate_id"]),
                beta_rad=anchor["beta"],
                residual_mm=0.0,
                min_margin_deg=1.0,
                normalized_min_margin=0.1,
                quality="Silver",
            )
            outcome = continuation(source, destination)
            if (
                isinstance(outcome, ContinuationOutcome)
                and outcome.success
                and outcome.actual_bounds
                and outcome.residual_mm <= active.maximum_residual_mm + 1.0e-12
                and _inside(outcome.beta_rad, bounds)
            ):
                outcomes.append((anchor, outcome))
        gap = 0.0
        if len(outcomes) >= 2:
            gap = max(
                float(
                    np.sqrt(
                        np.mean(
                            np.square(np.rad2deg(left[1].beta_rad - right[1].beta_rad))
                        )
                    )
                )
                for left_index, left in enumerate(outcomes)
                for right in outcomes[left_index + 1 :]
            )
        if len(outcomes) < int(minimum_successful_parents):
            status = "insufficient_independent_parents"
        elif gap > float(parent_gap_max_deg) + 1.0e-12:
            status = "nonstitchable_parent_disagreement"
        else:
            status = "accepted"
        audit_row = {
            "physical_point_id": str(target.physical_point_id),
            "status": status,
            "successful_parent_count": len(outcomes),
            "parent_gap_max_deg": gap if len(outcomes) >= 2 else None,
            "chart_id": next(iter(chart_ids)),
            "cell_level_mm": cell.level_mm,
            "cell_ix": cell.ix,
            "cell_iy": cell.iy,
            "cell_iz": cell.iz,
        }
        if status != "accepted":
            audit_rows.append(audit_row)
            continue
        chosen_anchor, chosen = min(
            outcomes,
            key=lambda item: (
                float(item[1].residual_mm),
                item[0]["task_probe_id"],
            ),
        )
        metrics = _candidate_metrics(
            environment, beta=chosen.beta_rad, target_xyz=xyz, policy=active
        )
        quality = str(metrics["quality"])
        if quality == "Reject":
            audit_row["status"] = "corrected_candidate_rejected"
            audit_rows.append(audit_row)
            continue
        chart_id = next(iter(chart_ids))
        accepted_records.append(
            SupervisionRecord(
                record_id=f"dense:{source_family}:{target.physical_point_id}",
                kind=SupervisionKind.STATIC,
                physical_point_id=str(target.physical_point_id),
                chart_id=chart_id,
                cell=cell,
                xyz_m=xyz,
                beta_rad=chosen.beta_rad,
                is_primary=True,
                required=False,
                priority=priority,
                source_family=str(source_family),
                sample_weight=1.0 if quality == "Gold" else 0.5,
                quality_class=quality,
                residual_mm=float(metrics["residual_mm"]),
                actual_bounds=bool(metrics["actual_bounds"]),
            )
        )
        audit_row.update(
            {
                "selected_parent_probe_id": chosen_anchor["task_probe_id"],
                "selected_residual_mm": float(metrics["residual_mm"]),
                "quality_class": quality,
                "normalized_sigma3_m": float(metrics["normalized_sigma3_m"]),
                "normalized_kappa": float(metrics["normalized_kappa"]),
                "risk_flags": "|".join(metrics["risk_flags"]),
            }
        )
        audit_rows.append(audit_row)
    return DenseCorrectionResult(
        records=tuple(accepted_records), audit=pd.DataFrame.from_records(audit_rows)
    )
