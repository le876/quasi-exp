"""Deterministic scientific contracts for BACRA V14.3R retry10.

The module owns decisions shared by the retry10 runner and its tests.  It does
not own numerical continuation or file I/O: those remain explicit in the
runner so stage order, cost, provenance, and recovery are visible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .exploration_qualification import DEFAULT_BETA_WEIGHTS, weighted_beta_rms_deg


BETA_COLUMNS = tuple(f"beta{index}_rad" for index in range(1, 7))
XYZ_COLUMNS = ("x_m", "y_m", "z_m")
METRIC_VERSION = "normalized_weighted_v1"


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class GateDecision:
    status: GateStatus
    authorized: bool
    reasons: tuple[str, ...] = ()
    evidence_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        status = GateStatus(self.status)
        if bool(self.authorized) != (status is GateStatus.PASS):
            raise ValueError("authorized is true exactly when status=pass")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reasons", tuple(map(str, self.reasons)))
        object.__setattr__(self, "evidence_paths", tuple(map(str, self.evidence_paths)))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["reasons"] = list(self.reasons)
        value["evidence_paths"] = list(self.evidence_paths)
        return value


def normalized_weighted_beta_deg(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] = DEFAULT_BETA_WEIGHTS,
) -> float:
    """The retry10 Teacher/Student beta distance, normalized by sum(w^2)."""

    return weighted_beta_rms_deg(left, right, weights=weights)


def raw_beta_max_deg(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float:
    delta = np.asarray(left, dtype=float).reshape(6) - np.asarray(
        right, dtype=float
    ).reshape(6)
    return float(np.max(np.abs(np.degrees(delta))))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def compatible_candidate_medoid(
    candidates: pd.DataFrame,
    *,
    weighted_max_deg: float = 2.0,
    raw_max_deg: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Complete-link cluster candidate labels and select only safe medoids.

    A node is admitted only when every candidate pair is mutually compatible.
    This prevents a chain-connected cluster from hiding a distant branch.
    """

    _require_columns(
        candidates,
        ("task_node_id", "chart_id", "candidate_id", *BETA_COLUMNS),
        "candidate labels",
    )
    selected: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for node_id, group in candidates.groupby("task_node_id", sort=True):
        ordered = group.sort_values(
            ["chart_id", "candidate_id"], kind="stable"
        ).reset_index(drop=True)
        beta = ordered.loc[:, BETA_COLUMNS].to_numpy(dtype=float)
        weighted = np.zeros((len(beta), len(beta)), dtype=float)
        raw = np.zeros_like(weighted)
        for left in range(len(beta)):
            for right in range(left + 1, len(beta)):
                weighted[left, right] = weighted[right, left] = (
                    normalized_weighted_beta_deg(beta[left], beta[right])
                )
                raw[left, right] = raw[right, left] = raw_beta_max_deg(
                    beta[left], beta[right]
                )
        branch_conflict = bool(np.any(raw > float(raw_max_deg) + 1.0e-12))
        local_incompatible = bool(
            np.any(weighted > float(weighted_max_deg) + 1.0e-12)
        )
        all_compatible = not branch_conflict and not local_incompatible
        chosen_index: int | None = None
        if all_compatible:
            chosen_index = min(
                range(len(ordered)),
                key=lambda index: (
                    float(np.sum(weighted[index])),
                    str(ordered.iloc[index]["chart_id"]),
                    str(ordered.iloc[index]["candidate_id"]),
                ),
            )
            row = ordered.iloc[chosen_index].to_dict()
            row.update(
                {
                    "new_node_candidate_cluster_id": f"node_{int(node_id)}_cluster_0",
                    "new_node_branch_conflict": False,
                    "new_node_local_incompatible": False,
                    "supporting_chart_count": int(len(ordered)),
                    "selection_method": "normalized_weighted_complete_link_medoid",
                }
            )
            selected.append(row)
        audits.append(
            {
                "task_node_id": int(node_id),
                "candidate_count": int(len(ordered)),
                "weighted_gap_max_deg": float(np.max(weighted)) if len(beta) else 0.0,
                "raw_gap_max_deg": float(np.max(raw)) if len(beta) else 0.0,
                "new_node_candidate_cluster_id": (
                    f"node_{int(node_id)}_cluster_0" if all_compatible else None
                ),
                "new_node_branch_conflict": branch_conflict,
                "new_node_local_incompatible": local_incompatible,
                "selected_candidate_id": (
                    None
                    if chosen_index is None
                    else str(ordered.iloc[chosen_index]["candidate_id"])
                ),
            }
        )
    return pd.DataFrame.from_records(selected), pd.DataFrame.from_records(audits)


def _adjacency(task_edges: pd.DataFrame, node_ids: set[int]) -> dict[int, set[int]]:
    _require_columns(task_edges, ("left_node_id", "right_node_id"), "task edges")
    adjacency = {int(node): set() for node in node_ids}
    for row in task_edges.itertuples(index=False):
        left, right = int(row.left_node_id), int(row.right_node_id)
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency


def _within_hops(adjacency: Mapping[int, set[int]], seed: int, hops: int) -> set[int]:
    visited = {int(seed)}
    frontier = {int(seed)}
    for _ in range(max(0, int(hops))):
        frontier = {
            neighbor
            for node in frontier
            for neighbor in adjacency.get(node, set())
            if neighbor not in visited
        }
        visited.update(frontier)
    return visited


def dynamic_frontier_root_registry(
    task_nodes: pd.DataFrame,
    task_edges: pd.DataFrame,
    atlas_labels: pd.DataFrame,
    *,
    prior_root_node_ids: Sequence[int] = (),
    historical_support: Mapping[int, int] | None = None,
    root_count: int = 4,
    hops: int = 2,
) -> pd.DataFrame:
    """Greedily select retry10 roots from current, not historical, support."""

    _require_columns(
        task_nodes,
        ("task_node_id", "source_parent_node_id", *XYZ_COLUMNS),
        "task nodes",
    )
    _require_columns(atlas_labels, ("task_node_id", *XYZ_COLUMNS), "atlas labels")
    if int(root_count) < 1:
        raise ValueError("root_count must be positive")
    nodes = task_nodes.drop_duplicates("task_node_id").copy()
    node_ids = set(nodes["task_node_id"].astype(int))
    adjacency = _adjacency(task_edges, node_ids)
    node_by_id = nodes.set_index("task_node_id")
    covered = set(atlas_labels["task_node_id"].astype(int))
    uncovered = node_ids - covered
    current_node_support = (
        atlas_labels.groupby("task_node_id")["chart_id"].nunique().astype(int)
        if "chart_id" in atlas_labels
        else atlas_labels.groupby("task_node_id").size().astype(int)
    )
    parent_by_node = nodes.set_index("task_node_id")["source_parent_node_id"].astype(int)
    current_parent_support = (
        nodes[nodes["task_node_id"].astype(int).isin(covered)]
        .groupby("source_parent_node_id")["task_node_id"]
        .nunique()
        .astype(int)
    )
    label_by_node = (
        atlas_labels.sort_values(["task_node_id", "chart_id"], kind="stable")
        .drop_duplicates("task_node_id")
        .set_index("task_node_id")
    )
    historical = {} if historical_support is None else {
        int(key): int(value) for key, value in historical_support.items()
    }
    fixed_roots = [int(value) for value in prior_root_node_ids if int(value) in node_ids]
    selected: list[int] = []
    records: list[dict[str, Any]] = []
    candidates = sorted(covered)
    for selection_rank in range(int(root_count)):
        scored: list[tuple[tuple[Any, ...], int, dict[str, Any]]] = []
        separation_roots = fixed_roots + selected
        separation_xyz = (
            node_by_id.loc[separation_roots, list(XYZ_COLUMNS)].to_numpy(float)
            if separation_roots
            else np.empty((0, 3), dtype=float)
        )
        for node_id in candidates:
            if node_id in selected:
                continue
            within = _within_hops(adjacency, node_id, hops)
            uncovered_nodes = within & uncovered
            uncovered_cells = {
                int(parent_by_node.loc[target]) for target in uncovered_nodes
            }
            if not uncovered_cells:
                continue
            parent_id = int(parent_by_node.loc[node_id])
            xyz = node_by_id.loc[node_id, list(XYZ_COLUMNS)].to_numpy(float)
            minimum_distance_mm = (
                float(np.min(np.linalg.norm(separation_xyz - xyz, axis=1)) * 1000.0)
                if len(separation_xyz)
                else math.inf
            )
            label = label_by_node.loc[node_id]
            margin = float(label.get("normalized_min_margin", label.get("min_margin_deg", 0.0)))
            condition = float(label.get("condition_number", math.inf))
            if not math.isfinite(condition):
                condition = math.inf
            row = {
                "selection_rank": int(selection_rank),
                "task_node_id": int(node_id),
                "source_parent_node_id": parent_id,
                "current_frontier_uncovered_cell_count": int(len(uncovered_cells)),
                "current_frontier_uncovered_node_count": int(len(uncovered_nodes)),
                "current_node_support_count": int(current_node_support.get(node_id, 0)),
                "current_parent_cell_support_count": int(current_parent_support.get(parent_id, 0)),
                "minimum_prior_root_distance_mm": minimum_distance_mm,
                "normalized_margin": margin,
                "condition_number": condition,
                "retry9_historical_support_count": int(historical.get(node_id, 0)),
                **{name: float(node_by_id.loc[node_id, name]) for name in XYZ_COLUMNS},
            }
            score = (
                -row["current_frontier_uncovered_cell_count"],
                row["current_parent_cell_support_count"],
                row["current_node_support_count"],
                -row["minimum_prior_root_distance_mm"],
                -row["normalized_margin"],
                row["condition_number"],
                row["retry9_historical_support_count"],
                node_id,
            )
            scored.append((score, node_id, row))
        if not scored:
            break
        _score, chosen, record = min(scored, key=lambda item: item[0])
        selected.append(chosen)
        records.append(record)
    return pd.DataFrame.from_records(records)


def k2_retention_decision(
    *, recovered_node_count: int, denominator_task_nodes: int = 25_000
) -> dict[str, Any]:
    if int(recovered_node_count) < 0 or int(denominator_task_nodes) < 1:
        raise ValueError("K2 counts must be non-negative with a positive denominator")
    gain = float(recovered_node_count) / float(denominator_task_nodes)
    return {
        "keep_valid_k2_labels": True,
        "incremental_coverage_gain": gain,
        "promote_k2_policy": bool(gain >= 0.005 - 1.0e-12),
    }


TAXONOMY_ORDER = (
    "branch_conflict",
    "beam_limited",
    "frontier_reachable",
    "graph_limited",
    "disconnected_but_feasible",
    "no_feasible_solution_observed",
)


def classify_frontier_taxonomy(evidence: pd.DataFrame) -> pd.DataFrame:
    """Apply the registered retry10 taxonomy predicates in strict priority."""

    required = (
        "task_node_id",
        "feasible_label_count",
        "raw_candidate_gap_max_deg",
        "catastrophic_neighbor_jump",
        "cap_hit",
        "pruned_hypothesis_reachable",
        "registered_retained_neighbor",
        "local_continuation_succeeded",
        "local_continuation_exhausted_or_pending",
        "edge_predicate_satisfied_missing_edge",
        "finite_in_bounds_fk_valid_label",
        "certified_primary_path",
    )
    _require_columns(evidence, required, "frontier evidence")
    rows: list[dict[str, Any]] = []
    for row in evidence.itertuples(index=False):
        branch = bool(
            (
                int(row.feasible_label_count) >= 2
                and float(row.raw_candidate_gap_max_deg) > 5.0 + 1.0e-12
            )
            or bool(row.catastrophic_neighbor_jump)
        )
        beam = bool(row.cap_hit and row.pruned_hypothesis_reachable)
        frontier = bool(
            row.registered_retained_neighbor
            and not row.local_continuation_succeeded
            and row.local_continuation_exhausted_or_pending
        )
        graph = bool(row.edge_predicate_satisfied_missing_edge)
        disconnected = bool(
            row.finite_in_bounds_fk_valid_label and not row.certified_primary_path
        )
        flags = (branch, beam, frontier, graph, disconnected, True)
        category = next(
            name for name, active in zip(TAXONOMY_ORDER, flags, strict=True) if active
        )
        record = row._asdict()
        record["taxonomy"] = category
        record["taxonomy_priority"] = TAXONOMY_ORDER.index(category)
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def component_boundary_audit(
    labels: pd.DataFrame,
    task_edges: pd.DataFrame,
    *,
    weighted_p95_max_deg: float = 2.0,
    raw_catastrophic_deg: float = 5.0,
    raw_catastrophic_rate_max: float = 0.01,
) -> pd.DataFrame:
    """Decide whether components may share the retry10 global Student."""

    _require_columns(labels, ("task_node_id", "component_id", *BETA_COLUMNS), "component labels")
    _require_columns(task_edges, ("left_node_id", "right_node_id"), "task edges")
    indexed = labels.sort_values("task_node_id", kind="stable").drop_duplicates(
        "task_node_id"
    ).set_index("task_node_id")
    component_ids = sorted(map(str, labels["component_id"].astype(str).unique()))
    records: list[dict[str, Any]] = []
    for left_index, left_component in enumerate(component_ids):
        for right_component in component_ids[left_index + 1 :]:
            pairs: list[tuple[int, int]] = []
            weighted: list[float] = []
            raw: list[float] = []
            for edge in task_edges.itertuples(index=False):
                left, right = int(edge.left_node_id), int(edge.right_node_id)
                if left not in indexed.index or right not in indexed.index:
                    continue
                pair_components = {
                    str(indexed.loc[left, "component_id"]),
                    str(indexed.loc[right, "component_id"]),
                }
                if pair_components != {left_component, right_component}:
                    continue
                pairs.append((left, right))
                left_beta = indexed.loc[left, list(BETA_COLUMNS)].to_numpy(float)
                right_beta = indexed.loc[right, list(BETA_COLUMNS)].to_numpy(float)
                weighted.append(normalized_weighted_beta_deg(left_beta, right_beta))
                raw.append(raw_beta_max_deg(left_beta, right_beta))
            if not pairs:
                relation = "spatially_separated"
                compatible = True
                weighted_p95 = 0.0
                catastrophic_rate = 0.0
            else:
                weighted_p95 = float(np.percentile(weighted, 95))
                catastrophic_rate = float(np.mean(np.asarray(raw) > raw_catastrophic_deg))
                compatible = bool(
                    weighted_p95 <= weighted_p95_max_deg + 1.0e-12
                    and catastrophic_rate <= raw_catastrophic_rate_max + 1.0e-12
                )
                relation = "boundary_compatible" if compatible else "incompatible"
            records.append(
                {
                    "component_a": left_component,
                    "component_b": right_component,
                    "registered_boundary_pair_count": int(len(pairs)),
                    "weighted_gap_p95_deg": weighted_p95,
                    "raw_catastrophic_rate": catastrophic_rate,
                    "relation": relation,
                    "component_global_student_compatible": compatible,
                }
            )
    return pd.DataFrame.from_records(records)


def select_global_student_components(
    labels: pd.DataFrame,
    pair_audit: pd.DataFrame,
    *,
    primary_component_id: str,
) -> tuple[str, ...]:
    counts = labels.groupby(labels["component_id"].astype(str)).size().to_dict()
    components = sorted(counts, key=lambda value: (-int(counts[value]), value))
    if str(primary_component_id) not in components:
        raise ValueError("primary component is absent from labels")
    relation: dict[frozenset[str], bool] = {}
    for row in pair_audit.itertuples(index=False):
        relation[frozenset((str(row.component_a), str(row.component_b)))] = bool(
            row.component_global_student_compatible
        )
    selected = [str(primary_component_id)]
    for component in components:
        if component in selected:
            continue
        if all(relation.get(frozenset((component, other)), True) for other in selected):
            selected.append(component)
    return tuple(selected)


def wilson_lower_bound(successes: int, trials: int, *, z: float = 1.959963984540054) -> float:
    if int(trials) <= 0 or int(successes) < 0 or int(successes) > int(trials):
        raise ValueError("Wilson counts must satisfy 0 <= successes <= trials")
    n = float(trials)
    p = float(successes) / n
    denominator = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, float((center - radius) / denominator))


def preflight_generation_decision(
    strata: pd.DataFrame,
    *,
    required_new_rows: int,
    allowed_qualities: Sequence[str],
    attempt_multiplier: int = 5,
) -> tuple[GateDecision, dict[str, Any]]:
    """Authorize exact generation before materialization, never after it."""

    _require_columns(
        strata,
        ("stratum_id", "target_weight", "trial_count", "gold_count", "silver_count"),
        "preflight strata",
    )
    allowed = set(map(str, allowed_qualities))
    if not allowed or not allowed <= {"Gold", "Silver"}:
        raise ValueError("allowed qualities must be Gold and/or Silver")
    weights = pd.to_numeric(strata["target_weight"], errors="raise").to_numpy(float)
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("preflight target weights must be non-negative and sum to one")
    lower: list[float] = []
    observed_any: list[bool] = []
    for row in strata.itertuples(index=False):
        successes = 0
        if "Gold" in allowed:
            successes += int(row.gold_count)
        if "Silver" in allowed:
            successes += int(row.silver_count)
        lower.append(wilson_lower_bound(successes, int(row.trial_count)))
        observed_any.append(successes > 0)
    p_lower = float(np.dot(weights, np.asarray(lower, dtype=float)))
    estimated_attempts = (
        math.inf if p_lower <= 0.0 else int(math.ceil(int(required_new_rows) / p_lower))
    )
    within_budget = bool(
        math.isfinite(estimated_attempts)
        and estimated_attempts <= int(attempt_multiplier) * int(required_new_rows)
    )
    nonempty = bool(all(value or weight <= 0.0 for value, weight in zip(observed_any, weights, strict=True)))
    authorized = bool(int(required_new_rows) >= 0 and within_budget and nonempty)
    reasons = []
    if not within_budget:
        reasons.append("estimated_attempts_exceed_registered_budget")
    if not nonempty:
        reasons.append("mandatory_stratum_has_no_accepted_label")
    decision = GateDecision(
        GateStatus.PASS if authorized else GateStatus.FAIL,
        authorized,
        tuple(reasons),
    )
    report = {
        "allowed_qualities": sorted(allowed),
        "required_new_rows": int(required_new_rows),
        "acceptance_lower_bound": p_lower,
        "estimated_attempts": None if not math.isfinite(estimated_attempts) else int(estimated_attempts),
        "attempt_budget": int(attempt_multiplier) * int(required_new_rows),
        "within_attempt_budget": within_budget,
        "mandatory_strata_nonempty": nonempty,
    }
    return decision, report


def dataset_generation_decision(
    *,
    target_rows: int,
    atlas_frozen: bool,
    exploration_hard_guards_pass: bool,
    coverage_fraction: float,
    plateau_or_exhausted: bool,
    seed_rows: int,
    seed_learning_curve_artifact_exists: bool,
    prior_dataset_complete: bool,
    taxonomy_complete: bool,
    preflight_pass: bool,
    x_tertile_minimum_coverage: float = 0.0,
) -> tuple[GateDecision, str | None]:
    """Dataset authorization with no Student-quality or tension dependency."""

    target = int(target_rows)
    common = bool(atlas_frozen and exploration_hard_guards_pass and preflight_pass)
    reasons: list[str] = []
    dataset_id: str | None = None
    authorized = False
    if target == 50_000:
        common = bool(
            common
            and int(seed_rows) >= 10_000
            and seed_learning_curve_artifact_exists
        )
        if common and float(coverage_fraction) >= 0.30:
            authorized = True
            dataset_id = "pilot_domain_expanded_candidate_dataset_50k"
        elif common and 0.20 <= float(coverage_fraction) < 0.30 and plateau_or_exhausted:
            authorized = True
            dataset_id = "pilot_domain_limited_candidate_dataset_50k"
    elif target == 100_000:
        common = bool(common and prior_dataset_complete and taxonomy_complete)
        if common and float(coverage_fraction) >= 0.50:
            authorized = True
            dataset_id = "pilot_domain_expanded_candidate_dataset_100k"
        elif common and float(coverage_fraction) >= 0.20 and plateau_or_exhausted:
            authorized = True
            dataset_id = "pilot_domain_limited_candidate_dataset_100k"
    elif target == 200_000:
        common = bool(common and prior_dataset_complete and taxonomy_complete)
        if (
            common
            and float(coverage_fraction) >= 0.60
            and plateau_or_exhausted
            and float(x_tertile_minimum_coverage) >= 0.40
        ):
            authorized = True
            dataset_id = "pilot_domain_expanded_exploratory_dataset_200k"
        elif common and 0.40 <= float(coverage_fraction) < 0.60 and plateau_or_exhausted:
            authorized = True
            dataset_id = "pilot_domain_limited_exploratory_dataset_200k"
    else:
        raise ValueError("retry10 staged dataset target must be 50k, 100k, or 200k")
    if not atlas_frozen:
        reasons.append("atlas_not_frozen")
    if not exploration_hard_guards_pass:
        reasons.append("exploration_hard_guards_failed")
    if not preflight_pass:
        reasons.append("preflight_failed")
    if target == 50_000 and int(seed_rows) < 10_000:
        reasons.append("seed_rows_below_10k")
    if target == 50_000 and not seed_learning_curve_artifact_exists:
        reasons.append("seed_learning_curve_not_evaluated")
    if target >= 100_000 and not prior_dataset_complete:
        reasons.append("prior_dataset_incomplete")
    if target >= 100_000 and not taxonomy_complete:
        reasons.append("taxonomy_incomplete")
    if target == 200_000 and float(coverage_fraction) < 0.40:
        reasons.append("coverage_below_40_percent")
    if not authorized and not reasons:
        reasons.append("coverage_path_not_admitted")
    return (
        GateDecision(
            GateStatus.PASS if authorized else GateStatus.FAIL,
            authorized,
            tuple(reasons),
        ),
        dataset_id,
    )


def student_quality_decision(
    *,
    evaluated: bool,
    no_nan: bool,
    bounds_violation_count: int,
    raw_fk_p95_mm: float,
    dls_two_step_fk_p95_mm: float,
) -> GateDecision:
    if not evaluated:
        return GateDecision(
            GateStatus.NOT_EVALUATED,
            False,
            ("evaluation_panel_unavailable",),
        )
    passed = bool(
        no_nan
        and int(bounds_violation_count) == 0
        and (
            float(raw_fk_p95_mm) <= 30.0
            or float(dls_two_step_fk_p95_mm) <= 15.0
        )
    )
    return GateDecision(
        GateStatus.PASS if passed else GateStatus.FAIL,
        passed,
        () if passed else ("registered_student_quality_threshold_failed",),
    )


def macroblock_ids(frame: pd.DataFrame, *, block_size_mm: int = 40) -> pd.Series:
    _require_columns(frame, XYZ_COLUMNS, "macroblock frame")
    block = int(block_size_mm)
    if block < 1:
        raise ValueError("macroblock size must be positive")
    coordinates = np.floor(
        frame.loc[:, XYZ_COLUMNS].to_numpy(float) / (block / 1000.0)
    ).astype(np.int64)
    return pd.Series(
        [f"{int(x)}:{int(y)}:{int(z)}" for x, y, z in coordinates],
        index=frame.index,
        dtype="object",
    )


def stable_hash_rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{int(seed)}:{value}".encode()).digest()


def deterministic_paired_bootstrap_delta(
    smaller: Sequence[float] | np.ndarray,
    larger: Sequence[float] | np.ndarray,
    *,
    statistic: str = "p95",
    repetitions: int = 10_000,
    seed: int = 20260881,
) -> dict[str, float | bool]:
    left = np.asarray(smaller, dtype=float).reshape(-1)
    right = np.asarray(larger, dtype=float).reshape(-1)
    if left.shape != right.shape or len(left) < 2:
        raise ValueError("paired bootstrap requires equal arrays with at least two rows")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired bootstrap inputs must be finite")
    if statistic == "p95":
        function = lambda values: float(np.percentile(values, 95))
    elif statistic == "mean":
        function = lambda values: float(np.mean(values))
    else:
        raise ValueError("unsupported paired bootstrap statistic")
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(repetitions), dtype=float)
    for index in range(int(repetitions)):
        sample = rng.integers(0, len(left), size=len(left))
        deltas[index] = function(right[sample]) - function(left[sample])
    observed = function(right) - function(left)
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    return {
        "observed_delta": float(observed),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "systematic_degradation": bool(lower > 0.0),
    }
