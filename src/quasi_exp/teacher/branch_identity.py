"""Cut-independent branch identity tools for the V11.3 canonical teacher repair.

The V11 teacher deliberately treats traversal variants as independent solves.
This module introduces the next protocol boundary: variants are evidence about
one frozen canonical inverse branch, never competing label sources.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .canonical import (
    CanonicalTeacher,
    TeacherPolicy,
    TeacherTrajectory,
    _correct_target,
    _environment_jacobian,
    _trajectory_metrics,
    weighted_damped_pinv,
)


BETA_COLUMNS = tuple(f"teacher_beta{index}_rad" for index in range(1, 7))


def cyclic_traversal_order(
    phase_count: int, *, direction: str, cut: int
) -> np.ndarray:
    """Return a cyclic order whose first element is always the requested cut."""

    count = int(phase_count)
    if count < 3:
        raise ValueError("phase_count must be at least three")
    normalized = str(direction).lower()
    if normalized not in {"forward", "reverse"}:
        raise ValueError("direction must be forward or reverse")
    start = int(cut) % count
    step = 1 if normalized == "forward" else -1
    return (start + step * np.arange(count, dtype=np.int64)) % count


def _beta_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = [name for name in ("phase_idx", *BETA_COLUMNS) if name not in frame]
    if missing:
        raise ValueError(f"branch frame is missing columns: {missing}")
    ordered = frame.sort_values("phase_idx", kind="stable").reset_index(drop=True)
    phase = ordered["phase_idx"].to_numpy(dtype=np.int64)
    if len(np.unique(phase)) != len(phase):
        raise ValueError("phase_idx must be unique")
    beta = ordered[list(BETA_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(beta).all():
        raise ValueError("beta values must be finite")
    return phase, beta


def _rms_deg(delta_rad: np.ndarray, *, axis: int = -1) -> np.ndarray:
    return np.rad2deg(np.sqrt(np.mean(np.square(delta_rad), axis=axis)))


def _gap_summary(name: str, gap_deg: np.ndarray) -> dict[str, Any]:
    gap = np.asarray(gap_deg, dtype=float).reshape(-1)
    return {
        "variant": str(name),
        "gap_p50_deg": float(np.percentile(gap, 50)),
        "gap_p90_deg": float(np.percentile(gap, 90)),
        "gap_p95_deg": float(np.percentile(gap, 95)),
        "gap_max_deg": float(np.max(gap)),
        "ratio_gap_gt_0p5deg": float(np.mean(gap > 0.5)),
        "ratio_gap_gt_1deg": float(np.mean(gap > 1.0)),
        "ratio_gap_gt_2deg": float(np.mean(gap > 2.0)),
        "ratio_gap_gt_5deg": float(np.mean(gap > 5.0)),
    }


def _connected_labels(values: np.ndarray, threshold_deg: float) -> np.ndarray:
    beta = np.asarray(values, dtype=float).reshape(-1, 6)
    count = len(beta)
    labels = np.full(count, -1, dtype=np.int64)
    cluster = 0
    for start in range(count):
        if labels[start] >= 0:
            continue
        labels[start] = cluster
        pending = [start]
        while pending:
            left = pending.pop()
            gap = _rms_deg(beta - beta[left])
            for right in np.flatnonzero(gap <= float(threshold_deg) + 1.0e-12):
                if labels[right] < 0:
                    labels[right] = cluster
                    pending.append(int(right))
        cluster += 1
    return labels


def _cluster_summary(
    phase: np.ndarray,
    paths: Mapping[str, np.ndarray],
    *,
    threshold_deg: float,
) -> pd.DataFrame:
    names = tuple(paths)
    rows: list[dict[str, Any]] = []
    for row_index, phase_idx in enumerate(phase):
        values = np.vstack([paths[name][row_index] for name in names])
        labels = _connected_labels(values, threshold_deg)
        cluster_count = int(labels.max() + 1)
        separation = 0.0
        if cluster_count > 1:
            cross = []
            for left in range(len(values)):
                for right in range(left + 1, len(values)):
                    if labels[left] != labels[right]:
                        cross.append(float(_rms_deg(values[left] - values[right])))
            separation = min(cross)
        rows.append(
            {
                "phase_idx": int(phase_idx),
                "cluster_count": cluster_count,
                "min_cluster_separation_deg": float(separation),
                "cluster_membership": ";".join(
                    f"{name}:{int(label)}" for name, label in zip(names, labels)
                ),
            }
        )
    return pd.DataFrame(rows)


def _transition_intervals(
    phase: np.ndarray,
    variant: str,
    gap: np.ndarray,
    *,
    threshold_deg: float = 1.0,
) -> list[dict[str, Any]]:
    selected = np.asarray(gap > float(threshold_deg), dtype=bool)
    if not np.any(selected):
        return []
    intervals: list[dict[str, Any]] = []
    start: int | None = None
    for index, active in enumerate(np.r_[selected, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            stop = index - 1
            intervals.append(
                {
                    "variant": str(variant),
                    "threshold_deg": float(threshold_deg),
                    "start_phase_idx": int(phase[start]),
                    "end_phase_idx": int(phase[stop]),
                    "phase_count": int(stop - start + 1),
                    "gap_max_deg": float(np.max(gap[start : stop + 1])),
                }
            )
            start = None
    return intervals


@dataclass(frozen=True)
class BranchForensicsReport:
    per_phase_variant_gap: pd.DataFrame
    per_joint_gap: pd.DataFrame
    branch_clusters: pd.DataFrame
    nullspace_gap_report: pd.DataFrame
    transition_intervals: pd.DataFrame
    variant_summary: pd.DataFrame


def analyze_branch_variants(
    primary: pd.DataFrame,
    variants: Mapping[str, pd.DataFrame],
    *,
    jacobian: Callable[[np.ndarray], np.ndarray],
    cluster_threshold_deg: float = 0.5,
) -> BranchForensicsReport:
    """Compare frozen variants at identical phase indices and decompose gaps."""

    if not variants:
        raise ValueError("at least one branch variant is required")
    phase, primary_beta = _beta_from_frame(primary)
    paths: dict[str, np.ndarray] = {"primary": primary_beta}
    per_phase_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    for name, frame in variants.items():
        variant_phase, beta = _beta_from_frame(frame)
        if not np.array_equal(variant_phase, phase):
            raise ValueError(f"variant {name!r} phase inventory does not match primary")
        paths[str(name)] = beta
        delta = beta - primary_beta
        gap = _rms_deg(delta)
        summary_rows.append(_gap_summary(str(name), gap))
        interval_rows.extend(_transition_intervals(phase, str(name), gap))
        for joint in range(6):
            joint_gap = np.abs(np.rad2deg(delta[:, joint]))
            joint_rows.append(
                {
                    "variant": str(name),
                    "joint": f"beta{joint + 1}",
                    "abs_gap_p50_deg": float(np.percentile(joint_gap, 50)),
                    "abs_gap_p95_deg": float(np.percentile(joint_gap, 95)),
                    "abs_gap_max_deg": float(np.max(joint_gap)),
                }
            )
        for row_index, phase_idx in enumerate(phase):
            jac = np.asarray(jacobian(primary_beta[row_index]), dtype=float).reshape(3, 6)
            null_projector = np.eye(6) - np.linalg.pinv(jac, rcond=1.0e-12) @ jac
            null_delta = null_projector @ delta[row_index]
            task_delta = delta[row_index] - null_delta
            row = {
                "variant": str(name),
                "phase_idx": int(phase_idx),
                "gap_rms_deg": float(gap[row_index]),
                "nullspace_gap_rms_deg": float(_rms_deg(null_delta)),
                "task_gap_rms_deg": float(_rms_deg(task_delta)),
            }
            per_phase_rows.append(row)
            null_rows.append(row.copy())
    clusters = _cluster_summary(
        phase, paths, threshold_deg=float(cluster_threshold_deg)
    )
    return BranchForensicsReport(
        per_phase_variant_gap=pd.DataFrame(per_phase_rows),
        per_joint_gap=pd.DataFrame(joint_rows),
        branch_clusters=clusters,
        nullspace_gap_report=pd.DataFrame(null_rows),
        transition_intervals=pd.DataFrame(interval_rows),
        variant_summary=pd.DataFrame(summary_rows),
    )


def _normalized_preference(values: np.ndarray, *, larger_is_better: bool) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    median = float(np.median(data))
    scale = float(np.percentile(data, 75) - np.percentile(data, 25))
    if scale <= 1.0e-12:
        scale = max(float(np.std(data)), 1.0)
    score = (data - median) / scale
    return score if larger_is_better else -score


def select_canonical_root_phase(
    primary: pd.DataFrame,
    *,
    branch_clusters: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Select the frozen root using margin, conditioning, residual and separation."""

    required = {
        "phase_idx",
        "joint_margin_min_deg",
        "kappa",
        "teacher_fk_residual_mm",
    }
    if not required.issubset(primary.columns):
        raise ValueError(f"primary frame is missing root metrics: {sorted(required - set(primary))}")
    frame = primary.sort_values("phase_idx", kind="stable").reset_index(drop=True)
    separation = np.zeros(len(frame), dtype=float)
    cluster_count = np.ones(len(frame), dtype=np.int64)
    if branch_clusters is not None:
        merged = frame[["phase_idx"]].merge(
            branch_clusters[
                ["phase_idx", "cluster_count", "min_cluster_separation_deg"]
            ],
            on="phase_idx",
            how="left",
            validate="one_to_one",
        )
        separation = merged["min_cluster_separation_deg"].fillna(0.0).to_numpy(float)
        cluster_count = merged["cluster_count"].fillna(1).to_numpy(np.int64)
    score = (
        1.0
        * _normalized_preference(
            frame["joint_margin_min_deg"].to_numpy(float), larger_is_better=True
        )
        + 0.5
        * _normalized_preference(
            np.log1p(frame["kappa"].to_numpy(float)), larger_is_better=False
        )
        + 0.5
        * _normalized_preference(
            frame["teacher_fk_residual_mm"].to_numpy(float), larger_is_better=False
        )
        + 0.25 * _normalized_preference(separation, larger_is_better=True)
    )
    selected = int(np.argmax(score))
    return {
        "phase_idx": int(frame.loc[selected, "phase_idx"]),
        "root_score": float(score[selected]),
        "score_by_phase": score.astype(float).tolist(),
        "joint_margin_min_deg": float(frame.loc[selected, "joint_margin_min_deg"]),
        "kappa": float(frame.loc[selected, "kappa"]),
        "teacher_fk_residual_mm": float(
            frame.loc[selected, "teacher_fk_residual_mm"]
        ),
        "cluster_count": int(cluster_count[selected]),
        "min_cluster_separation_deg": float(separation[selected]),
    }


def _weighted_pairwise_gap_deg(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    a = np.asarray(left, dtype=float).reshape(-1, 6)
    b = np.asarray(right, dtype=float).reshape(-1, 6)
    weight = np.asarray(weights, dtype=float).reshape(6)
    delta = a[:, None, :] - b[None, :, :]
    return np.rad2deg(
        np.sqrt(np.sum(weight[None, None, :] * np.square(delta), axis=2) / np.sum(weight))
    )


def link_reference_cyclic_candidates(
    candidate_layers: Sequence[np.ndarray],
    residual_mm: Sequence[np.ndarray],
    *,
    reference_beta: np.ndarray,
    root_phase_idx: int,
    canonical_root_beta: np.ndarray,
    beta_weights: np.ndarray | Sequence[float],
    lambda_velocity: float,
    lambda_reference: float,
    closure_weight: float,
    root_cluster_threshold_deg: float,
    lambda_acceleration: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose one closed candidate path while freezing the canonical root cluster."""

    if not candidate_layers or len(candidate_layers) != len(residual_mm):
        raise ValueError("candidate layers and residuals must be non-empty and aligned")
    count = len(candidate_layers)
    reference = np.asarray(reference_beta, dtype=float).reshape(count, 6)
    layers = [np.asarray(layer, dtype=float).reshape(-1, 6) for layer in candidate_layers]
    errors = [np.asarray(value, dtype=float).reshape(-1) for value in residual_mm]
    if any(len(layer) == 0 for layer in layers):
        return np.zeros((0, 6)), {"success": False, "reason": "empty_layer"}
    if any(len(layer) != len(error) for layer, error in zip(layers, errors)):
        raise ValueError("each residual vector must match its candidate layer")
    weights = np.asarray(beta_weights, dtype=float).reshape(6)
    if np.any(weights <= 0.0):
        raise ValueError("beta_weights must be positive")
    root = int(root_phase_idx) % count
    order = cyclic_traversal_order(count, direction="forward", cut=root)
    ordered_layers = [layers[int(index)] for index in order]
    ordered_errors = [errors[int(index)] for index in order]
    ordered_reference = reference[order]
    root_gap = _weighted_pairwise_gap_deg(
        ordered_layers[0], np.asarray(canonical_root_beta).reshape(1, 6), weights
    )[:, 0]
    allowed_starts = np.flatnonzero(
        root_gap <= float(root_cluster_threshold_deg) + 1.0e-12
    )
    if len(allowed_starts) == 0:
        return np.zeros((0, 6)), {
            "success": False,
            "reason": "canonical_root_cluster_missing",
            "root_cluster_candidate_count": 0,
        }
    unary = []
    for layer, error, anchor in zip(
        ordered_layers, ordered_errors, ordered_reference
    ):
        reference_gap = _weighted_pairwise_gap_deg(
            layer, anchor.reshape(1, 6), weights
        )[:, 0]
        unary.append(
            np.square(error) + float(lambda_reference) * np.square(reference_gap)
        )
    transitions = [
        float(lambda_velocity)
        * np.square(
            _weighted_pairwise_gap_deg(ordered_layers[i - 1], ordered_layers[i], weights)
        )
        for i in range(1, count)
    ]
    best_cost = math.inf
    best_indices: list[int] | None = None
    for start in allowed_starts:
        if float(lambda_acceleration) > 0.0:
            cost = np.full(
                (len(ordered_layers[0]), len(ordered_layers[1])), np.inf
            )
            cost[int(start), :] = (
                unary[0][int(start)]
                + unary[1]
                + transitions[0][int(start), :]
            )
            second_order_parents: list[np.ndarray] = []
            for index in range(2, count):
                acceleration = (
                    ordered_layers[index][None, None, :, :]
                    - 2.0 * ordered_layers[index - 1][None, :, None, :]
                    + ordered_layers[index - 2][:, None, None, :]
                )
                acceleration_deg = np.rad2deg(
                    np.sqrt(
                        np.sum(weights[None, None, None, :] * np.square(acceleration), axis=3)
                        / np.sum(weights)
                    )
                )
                values = (
                    cost[:, :, None]
                    + transitions[index - 1][None, :, :]
                    + float(lambda_acceleration) * np.square(acceleration_deg)
                )
                parent = np.argmin(values, axis=0).astype(np.int64)
                cost = np.min(values, axis=0) + unary[index][None, :]
                second_order_parents.append(parent)
            seam = _weighted_pairwise_gap_deg(
                ordered_layers[-1],
                ordered_layers[0][int(start) : int(start) + 1],
                weights,
            )[:, 0]
            seam_acceleration = (
                ordered_layers[0][int(start)][None, None, :]
                - 2.0 * ordered_layers[-1][None, :, :]
                + ordered_layers[-2][:, None, :]
            )
            seam_acceleration_deg = np.rad2deg(
                np.sqrt(
                    np.sum(weights[None, None, :] * np.square(seam_acceleration), axis=2)
                    / np.sum(weights)
                )
            )
            total = (
                cost
                + float(closure_weight) * np.square(seam)[None, :]
                + float(lambda_acceleration) * np.square(seam_acceleration_deg)
            )
            penultimate, end = np.unravel_index(int(np.argmin(total)), total.shape)
            candidate_cost = float(total[penultimate, end])
            if candidate_cost >= best_cost:
                continue
            indices = [0] * count
            indices[-2] = int(penultimate)
            indices[-1] = int(end)
            for phase_index in range(count - 1, 1, -1):
                parent = second_order_parents[phase_index - 2]
                indices[phase_index - 2] = int(
                    parent[indices[phase_index - 1], indices[phase_index]]
                )
            best_indices = indices
            best_cost = candidate_cost
            continue
        cost = np.full(len(ordered_layers[0]), np.inf)
        cost[int(start)] = unary[0][int(start)]
        parents: list[np.ndarray] = []
        for index in range(1, count):
            values = cost[:, None] + transitions[index - 1]
            parent = np.argmin(values, axis=0).astype(np.int64)
            cost = values[parent, np.arange(values.shape[1])] + unary[index]
            parents.append(parent)
        seam = _weighted_pairwise_gap_deg(
            ordered_layers[-1], ordered_layers[0][int(start) : int(start) + 1], weights
        )[:, 0]
        total = cost + float(closure_weight) * np.square(seam)
        end = int(np.argmin(total))
        if float(total[end]) >= best_cost:
            continue
        indices = [end]
        current = end
        for parent in reversed(parents):
            current = int(parent[current])
            indices.append(current)
        best_indices = list(reversed(indices))
        best_cost = float(total[end])
    if best_indices is None:
        return np.zeros((0, 6)), {
            "success": False,
            "reason": "no_closed_path",
            "root_cluster_candidate_count": int(len(allowed_starts)),
        }
    ordered_selected = np.vstack(
        [ordered_layers[index][best_indices[index]] for index in range(count)]
    )
    selected = np.empty_like(ordered_selected)
    selected[order] = ordered_selected
    velocity = _rms_deg(np.roll(selected, -1, axis=0) - selected)
    return selected, {
        "success": True,
        "cost": best_cost,
        "root_cluster_candidate_count": int(len(allowed_starts)),
        "delta_beta_rms_p95_deg": float(np.percentile(velocity, 95)),
        "delta_beta_rms_max_deg": float(np.max(velocity)),
        "seam_beta_rms_deg": float(velocity[-1]),
        "second_order_acceleration_enabled": bool(float(lambda_acceleration) > 0.0),
    }


@dataclass(frozen=True)
class ConsensusVariantAudit:
    per_phase_gap: pd.DataFrame
    summary: pd.DataFrame
    branch_clusters: pd.DataFrame


def audit_consensus_variants(
    consensus_beta: np.ndarray,
    variants: Mapping[str, np.ndarray],
    *,
    cluster_threshold_deg: float,
) -> ConsensusVariantAudit:
    """Audit all traversal variants against one immutable consensus path."""

    consensus = np.asarray(consensus_beta, dtype=float).reshape(-1, 6)
    if not variants:
        raise ValueError("at least one audit variant is required")
    phase = np.arange(len(consensus), dtype=np.int64)
    paths: dict[str, np.ndarray] = {"consensus": consensus}
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    primary = (
        np.asarray(variants["primary"], dtype=float).reshape(-1, 6)
        if "primary" in variants
        else None
    )
    if primary is not None and primary.shape != consensus.shape:
        raise ValueError("variant 'primary' does not match consensus shape")
    for name, value in variants.items():
        beta = np.asarray(value, dtype=float).reshape(-1, 6)
        if beta.shape != consensus.shape:
            raise ValueError(f"variant {name!r} does not match consensus shape")
        paths[str(name)] = beta
        comparison = primary if str(name) == "repeat" and primary is not None else consensus
        gap = _rms_deg(beta - comparison)
        rows.extend(
            {
                "variant": str(name),
                "phase_idx": int(index),
                "gap_rms_deg": float(item),
            }
            for index, item in enumerate(gap)
        )
        summaries.append(_gap_summary(str(name), gap))
    return ConsensusVariantAudit(
        per_phase_gap=pd.DataFrame(rows),
        summary=pd.DataFrame(summaries),
        branch_clusters=_cluster_summary(
            phase, paths, threshold_deg=float(cluster_threshold_deg)
        ),
    )


@dataclass(frozen=True)
class BranchIdentityGate:
    residual_p95_mm: float = 1.0
    residual_max_mm: float = 3.0
    joint_margin_min_deg: float = 1.5
    delta_beta_rms_p95_deg: float = 1.0
    delta_beta_rms_max_deg: float = 2.0
    acceleration_beta_rms_p95_deg: float = 0.5
    seam_beta_rms_deg: float = 1.0
    repeat_beta_rms_p95_deg: float = 0.2
    variant_beta_rms_p95_deg: float = 1.0
    variant_beta_rms_max_deg: float = 2.0
    variant_ratio_gt_1deg: float = 0.01

    def evaluate(
        self,
        *,
        numerical_metrics: Mapping[str, float],
        variant_audit: ConsensusVariantAudit,
        variant_solver_success: Mapping[str, bool] | None = None,
        variant_trust_violation_count: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        required = {
            "residual_p95_mm",
            "residual_max_mm",
            "joint_margin_min_deg",
            "delta_beta_rms_p95_deg",
            "delta_beta_rms_max_deg",
            "acceleration_beta_rms_p95_deg",
            "seam_beta_rms_deg",
        }
        missing = required - set(numerical_metrics)
        if missing:
            raise ValueError(f"numerical metrics missing: {sorted(missing)}")
        summary = variant_audit.summary
        repeat = summary[summary["variant"] == "repeat"]
        traversal = summary[summary["variant"] != "repeat"]
        checks = {
            "primary_residual_p95": float(numerical_metrics["residual_p95_mm"])
            <= self.residual_p95_mm,
            "primary_residual_max": float(numerical_metrics["residual_max_mm"])
            <= self.residual_max_mm,
            "strict_joint_margin": float(numerical_metrics["joint_margin_min_deg"])
            >= self.joint_margin_min_deg,
            "trajectory_velocity_p95": float(
                numerical_metrics["delta_beta_rms_p95_deg"]
            )
            <= self.delta_beta_rms_p95_deg,
            "trajectory_velocity_max": float(
                numerical_metrics["delta_beta_rms_max_deg"]
            )
            <= self.delta_beta_rms_max_deg,
            "trajectory_acceleration_p95": float(
                numerical_metrics["acceleration_beta_rms_p95_deg"]
            )
            <= self.acceleration_beta_rms_p95_deg,
            "trajectory_closure": float(numerical_metrics["seam_beta_rms_deg"])
            <= self.seam_beta_rms_deg,
            "repeatability": bool(
                len(repeat) == 1
                and float(repeat.iloc[0]["gap_p95_deg"])
                <= self.repeat_beta_rms_p95_deg
            ),
            "traversal_cut_p95": bool(
                not traversal.empty
                and float(traversal["gap_p95_deg"].max())
                <= self.variant_beta_rms_p95_deg
            ),
            "traversal_cut_max": bool(
                not traversal.empty
                and float(traversal["gap_max_deg"].max())
                <= self.variant_beta_rms_max_deg
            ),
            "traversal_cut_ratio": bool(
                not traversal.empty
                and float(traversal["ratio_gap_gt_1deg"].max())
                <= self.variant_ratio_gt_1deg
            ),
            "single_branch_cluster": bool(
                not variant_audit.branch_clusters.empty
                and int(variant_audit.branch_clusters["cluster_count"].max()) == 1
            ),
            "audit_variant_solver_success": bool(
                variant_solver_success is None
                or (
                    len(variant_solver_success) > 0
                    and all(bool(value) for value in variant_solver_success.values())
                )
            ),
            "audit_variant_trust_preserved": bool(
                variant_trust_violation_count is None
                or (
                    len(variant_trust_violation_count) > 0
                    and all(
                        float(value) <= 0.0
                        for value in variant_trust_violation_count.values()
                    )
                )
            ),
        }
        normalized = {key: bool(value) for key, value in checks.items()}
        return {"checks": normalized, "gate_pass": bool(all(normalized.values()))}


@dataclass(frozen=True)
class BranchRepairPolicy:
    """Frozen local continuation policy around one reference branch."""

    beta_weights: tuple[float, ...] = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0)
    trust_radius_deg: float = 0.5
    candidate_budget: int = 4
    lambda_reference: float = 2.0
    lambda_previous: float = 1.0
    local_seed_std_deg: float = 0.1
    root_cluster_threshold_deg: float = 0.5
    reference_seed_mode: str = "all"
    enforce_reference_trust: bool = True

    def __post_init__(self) -> None:
        if len(self.beta_weights) != 6 or min(self.beta_weights) <= 0.0:
            raise ValueError("beta_weights must contain six positive values")
        if self.trust_radius_deg <= 0.0:
            raise ValueError("trust_radius_deg must be positive")
        if self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        if min(self.lambda_reference, self.lambda_previous) < 0.0:
            raise ValueError("reference and previous weights must be non-negative")
        if self.local_seed_std_deg < 0.0:
            raise ValueError("local_seed_std_deg must be non-negative")
        if self.root_cluster_threshold_deg <= 0.0:
            raise ValueError("root_cluster_threshold_deg must be positive")
        if self.reference_seed_mode not in {"all", "first", "none"}:
            raise ValueError("reference_seed_mode must be all, first or none")


class ReferenceBranchTeacher:
    """Solve every traversal as a projection onto one frozen reference branch."""

    def __init__(self, environment: Any):
        bounds = np.asarray(environment.bounds, dtype=float)
        if bounds.shape != (6, 2):
            raise ValueError("forward environment bounds must have shape (6, 2)")
        self._environment = environment

    def solve(
        self,
        target_xyz_m: np.ndarray,
        *,
        reference_beta: np.ndarray,
        root_phase_idx: int,
        canonical_root_beta: np.ndarray,
        direction: str,
        cut: int,
        corrector_policy: TeacherPolicy,
        repair_policy: BranchRepairPolicy,
        solver_seed: int,
        trajectory_id: str = "canonical-consensus",
        family_id: str = "canonical-family",
        radius_mm: float = 0.0,
    ) -> TeacherTrajectory:
        target = np.asarray(target_xyz_m, dtype=float)
        if target.ndim != 2 or target.shape[1] != 3 or len(target) < 3:
            raise ValueError("target_xyz_m must have shape (N, 3), N >= 3")
        reference = np.asarray(reference_beta, dtype=float)
        if reference.shape != (len(target), 6) or not np.isfinite(reference).all():
            raise ValueError(f"reference_beta must have shape ({len(target)}, 6)")
        root_index = int(root_phase_idx) % len(target)
        root_beta = np.asarray(canonical_root_beta, dtype=float).reshape(6)
        order = cyclic_traversal_order(
            len(target), direction=str(direction), cut=int(cut)
        )
        rng = np.random.default_rng(int(solver_seed))
        selected = np.empty_like(reference)
        solved_indices: list[int] = []
        trust_violations = 0
        total_iterations = 0
        candidate_counts: list[int] = []
        weights = np.asarray(repair_policy.beta_weights, dtype=float)
        for traversal_index, raw_phase in enumerate(order):
            phase = int(raw_phase)
            include_reference = bool(
                repair_policy.reference_seed_mode == "all"
                or (
                    repair_policy.reference_seed_mode == "first"
                    and traversal_index == 0
                )
            )
            seeds: list[np.ndarray] = [reference[phase]] if include_reference else []
            if phase == root_index:
                seeds.insert(0, root_beta)
            if solved_indices:
                previous_phase = solved_indices[-1]
                previous = selected[previous_phase]
                seeds.append(previous)
                if len(solved_indices) >= 2:
                    before = selected[solved_indices[-2]]
                    seeds.append(previous + (previous - before))
                previous_target = target[previous_phase]
                jacobian = _environment_jacobian(self._environment, previous)
                seeds.append(
                    previous
                    + weighted_damped_pinv(
                        jacobian,
                        damping=float(corrector_policy.damping),
                        weights=weights,
                    )
                    @ (target[phase] - previous_target)
                )
            while len(seeds) < int(repair_policy.candidate_budget):
                local_anchor = (
                    selected[solved_indices[-1]]
                    if solved_indices
                    else (root_beta if phase == root_index else reference[phase])
                )
                seeds.append(
                    local_anchor
                    + rng.normal(
                        0.0,
                        math.radians(float(repair_policy.local_seed_std_deg)),
                        size=6,
                    )
                )
            candidates: list[tuple[np.ndarray, float, int]] = []
            for seed in seeds[: int(repair_policy.candidate_budget)]:
                beta, residual, iterations, _success = _correct_target(
                    self._environment,
                    target[phase],
                    seed,
                    corrector_policy,
                )
                total_iterations += int(iterations)
                candidates.append((beta, float(residual), int(iterations)))
            candidate_counts.append(len(candidates))
            reference_gap = np.asarray(
                [
                    _weighted_pairwise_gap_deg(
                        beta.reshape(1, 6), reference[phase].reshape(1, 6), weights
                    )[0, 0]
                    for beta, _residual, _iterations in candidates
                ]
            )
            allowed = (
                reference_gap
                <= float(repair_policy.trust_radius_deg) + 1.0e-12
                if repair_policy.enforce_reference_trust
                else np.ones(len(candidates), dtype=bool)
            )
            if phase == root_index:
                root_gap = np.asarray(
                    [
                        _weighted_pairwise_gap_deg(
                            beta.reshape(1, 6), root_beta.reshape(1, 6), weights
                        )[0, 0]
                        for beta, _residual, _iterations in candidates
                    ]
                )
                allowed &= root_gap <= (
                    float(repair_policy.root_cluster_threshold_deg) + 1.0e-12
                )
            if not np.any(allowed):
                trust_violations += 1
                # Fail closed on branch identity.  Preserve the frozen chart for
                # evidence generation, but never promote an out-of-trust solve.
                selected[phase] = root_beta if phase == root_index else reference[phase]
                solved_indices.append(phase)
                continue
            scores = np.full(len(candidates), np.inf)
            for candidate_index, (beta, residual, _iterations) in enumerate(candidates):
                if not allowed[candidate_index]:
                    continue
                previous_cost = 0.0
                if solved_indices:
                    previous_cost = float(
                        _weighted_pairwise_gap_deg(
                            beta.reshape(1, 6),
                            selected[solved_indices[-1]].reshape(1, 6),
                            weights,
                        )[0, 0]
                    )
                scores[candidate_index] = (
                    residual * residual
                    + float(repair_policy.lambda_reference)
                    * float(reference_gap[candidate_index]) ** 2
                    + float(repair_policy.lambda_previous) * previous_cost**2
                )
            chosen = int(np.argmin(scores))
            selected[phase] = candidates[chosen][0]
            solved_indices.append(phase)
        achieved = np.asarray(self._environment.fk(selected), dtype=float).reshape(-1, 3)
        theta = np.asarray(self._environment.theta(selected), dtype=float)
        metrics = _trajectory_metrics(selected, target, achieved)
        bounds = np.asarray(self._environment.bounds, dtype=float)
        margin = np.minimum(
            selected - bounds[:, 0][None, :], bounds[:, 1][None, :] - selected
        )
        metrics["joint_margin_min_deg"] = float(np.rad2deg(np.min(margin)))
        metrics["trust_violation_count"] = float(trust_violations)
        metrics["corrector_iteration_count"] = float(total_iterations)
        metrics["reference_gap_rms_p95_deg"] = float(
            np.percentile(_rms_deg(selected - reference), 95)
        )
        success = bool(
            metrics["residual_max_mm"]
            <= float(corrector_policy.tracking_tolerance_mm)
            and np.all(margin >= -1.0e-12)
            and trust_violations == 0
        )
        return TeacherTrajectory(
            beta_rad=selected,
            theta_rad=theta,
            achieved_xyz_m=achieved,
            target_xyz_m=target.copy(),
            chart_id=np.zeros(len(target), dtype=np.int64),
            branch_id=np.zeros(len(target), dtype=np.int64),
            metrics=metrics,
            provenance={
                "teacher_mode": "reference_branch_continuation",
                "trajectory_id": str(trajectory_id),
                "family_id": str(family_id),
                "radius_mm": float(radius_mm),
                "teacher_policy_id": corrector_policy.fingerprint,
                "solver_seed": int(solver_seed),
                "traversal_direction": str(direction),
                "cyclic_cut": int(cut) % len(target),
                "canonical_root_phase_idx": root_index,
                "trust_radius_deg": float(repair_policy.trust_radius_deg),
            },
            success=success,
            candidate_diagnostics={
                "candidate_counts": candidate_counts,
                "trust_violation_count": int(trust_violations),
            },
        )


@dataclass(frozen=True)
class CanonicalRootSelection:
    selected_beta: np.ndarray
    selected_cluster_id: int
    candidate_table: pd.DataFrame
    cluster_table: pd.DataFrame


def select_canonical_root_solution(
    beta_candidates: np.ndarray,
    *,
    residual_mm: np.ndarray,
    environment: Any,
    safe_margin_deg: float,
    cluster_threshold_deg: float,
) -> CanonicalRootSelection:
    """Cluster feasible root IK solutions and freeze one canonical cluster."""

    beta = np.asarray(beta_candidates, dtype=float).reshape(-1, 6)
    residual = np.asarray(residual_mm, dtype=float).reshape(-1)
    if len(beta) == 0 or len(beta) != len(residual):
        raise ValueError("root candidates and residuals must be non-empty and aligned")
    if not np.isfinite(beta).all() or not np.isfinite(residual).all():
        raise ValueError("root candidates and residuals must be finite")
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    within = np.all(
        (beta >= bounds[:, 0][None, :] - 1.0e-12)
        & (beta <= bounds[:, 1][None, :] + 1.0e-12),
        axis=1,
    )
    labels = _connected_labels(beta, float(cluster_threshold_deg))
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
    midpoint = 0.5 * (bounds[:, 0] + bounds[:, 1])
    weights = np.asarray([4.0, 4.0, 2.0, 2.0, 1.0, 1.0])
    posture = np.sum(
        weights[None, :] * np.square((beta - midpoint[None, :]) / span[None, :]),
        axis=1,
    ) / np.sum(weights)
    margin_deg = np.rad2deg(
        np.min(
            np.minimum(
                beta - bounds[:, 0][None, :],
                bounds[:, 1][None, :] - beta,
            ),
            axis=1,
        )
    )
    kappa = []
    for value in beta:
        singular = np.linalg.svd(
            _environment_jacobian(environment, value), compute_uv=False
        )
        kappa.append(
            float(singular[0] / max(float(singular[-1]), 1.0e-12))
        )
    kappa_array = np.asarray(kappa, dtype=float)
    margin_deficit = np.maximum(float(safe_margin_deg) - margin_deg, 0.0)
    canonical_cost = (
        np.square(residual)
        + 10.0 * posture
        + 100.0 * np.square(margin_deficit)
        + 0.001 * np.log1p(kappa_array)
    )
    canonical_cost[~within] = np.inf
    candidate_table = pd.DataFrame(
        {
            "candidate_idx": np.arange(len(beta), dtype=np.int64),
            "cluster_id": labels,
            "residual_mm": residual,
            "joint_margin_min_deg": margin_deg,
            "kappa": kappa_array,
            "posture_cost": posture,
            "canonical_cost": canonical_cost,
            "within_joint_bounds": within,
        }
    )
    for joint in range(6):
        candidate_table[f"beta{joint + 1}_rad"] = beta[:, joint]
    cluster_rows: list[dict[str, Any]] = []
    for cluster_id in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == cluster_id)
        feasible = members[np.isfinite(canonical_cost[members])]
        if len(feasible) == 0:
            representative = int(members[0])
            score = math.inf
        else:
            representative = int(feasible[np.argmin(canonical_cost[feasible])])
            score = float(canonical_cost[representative])
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "candidate_count": int(len(members)),
                "representative_candidate_idx": representative,
                "cluster_score": score,
                "residual_min_mm": float(np.min(residual[members])),
                "joint_margin_max_deg": float(np.max(margin_deg[members])),
                "kappa_min": float(np.min(kappa_array[members])),
            }
        )
    cluster_table = pd.DataFrame(cluster_rows).sort_values(
        ["cluster_score", "cluster_id"], kind="stable"
    )
    if not np.isfinite(cluster_table.iloc[0]["cluster_score"]):
        raise ValueError("no root candidate lies within joint bounds")
    selected_cluster_id = int(cluster_table.iloc[0]["cluster_id"])
    selected_candidate = int(cluster_table.iloc[0]["representative_candidate_idx"])
    return CanonicalRootSelection(
        selected_beta=beta[selected_candidate].copy(),
        selected_cluster_id=selected_cluster_id,
        candidate_table=candidate_table,
        cluster_table=cluster_table.reset_index(drop=True),
    )


def optimize_consensus_branch(
    environment: Any,
    target_xyz_m: np.ndarray,
    initial_beta: np.ndarray,
    *,
    reference_beta: np.ndarray,
    root_phase_idx: int,
    canonical_root_beta: np.ndarray,
    policy: TeacherPolicy,
    reference_weight: float,
    reference_scale_deg: float = 1.0,
    root_cluster_threshold_deg: float = 0.5,
    trajectory_id: str = "canonical-consensus-optimized",
    family_id: str = "canonical-family",
    radius_mm: float = 0.0,
) -> TeacherTrajectory:
    """Jointly optimize tracking/smoothness while remaining on the reference chart."""

    target = np.asarray(target_xyz_m, dtype=float).reshape(-1, 3)
    initial = np.asarray(initial_beta, dtype=float).reshape(len(target), 6)
    reference = np.asarray(reference_beta, dtype=float).reshape(len(target), 6)
    root_index = int(root_phase_idx) % len(target)
    root_beta = np.asarray(canonical_root_beta, dtype=float).reshape(6)
    anchored_reference = reference.copy()
    anchored_reference[root_index] = root_beta
    optimized_policy = replace(
        policy,
        surface_neighbor_weight=float(reference_weight),
        surface_neighbor_scale_deg=float(reference_scale_deg),
    )
    teacher = CanonicalTeacher(environment)
    optimized = teacher._optimize_trajectory(  # noqa: SLF001 - same package seam
        target,
        initial,
        optimized_policy,
        neighbor_anchor=anchored_reference,
    )
    corrected = []
    total_iterations = 0
    for phase, (point, seed) in enumerate(zip(target, optimized)):
        if phase == root_index:
            seed = root_beta
        beta, _residual, iterations, _success = _correct_target(
            environment, point, seed, policy
        )
        total_iterations += int(iterations)
        corrected.append(beta)
    selected = np.vstack(corrected)
    achieved = np.asarray(environment.fk(selected), dtype=float).reshape(-1, 3)
    theta = np.asarray(environment.theta(selected), dtype=float)
    metrics = _trajectory_metrics(selected, target, achieved)
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    margin = np.minimum(
        selected - bounds[:, 0][None, :], bounds[:, 1][None, :] - selected
    )
    metrics["joint_margin_min_deg"] = float(np.rad2deg(np.min(margin)))
    metrics["corrector_iteration_count"] = float(total_iterations)
    metrics["reference_gap_rms_p95_deg"] = float(
        np.percentile(_rms_deg(selected - anchored_reference), 95)
    )
    root_gap = float(_rms_deg(selected[root_index] - root_beta))
    metrics["canonical_root_gap_rms_deg"] = root_gap
    success = bool(
        metrics["residual_max_mm"] <= float(policy.tracking_tolerance_mm)
        and np.all(margin >= -1.0e-12)
        and root_gap <= float(root_cluster_threshold_deg)
    )
    return TeacherTrajectory(
        beta_rad=selected,
        theta_rad=theta,
        achieved_xyz_m=achieved,
        target_xyz_m=target.copy(),
        chart_id=np.zeros(len(target), dtype=np.int64),
        branch_id=np.zeros(len(target), dtype=np.int64),
        metrics=metrics,
        provenance={
            "teacher_mode": "consensus_whole_curve_optimization",
            "trajectory_id": str(trajectory_id),
            "family_id": str(family_id),
            "radius_mm": float(radius_mm),
            "teacher_policy_id": optimized_policy.fingerprint,
            "solver_seed": int(policy.solver_seed),
            "traversal_direction": "consensus",
            "cyclic_cut": root_index,
            "canonical_root_phase_idx": root_index,
            "reference_weight": float(reference_weight),
        },
        success=success,
        candidate_diagnostics={"whole_curve_optimization": True},
    )
