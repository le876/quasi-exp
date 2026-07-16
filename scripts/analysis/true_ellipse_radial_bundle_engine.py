"""Domain-aware primitives shared by registered true-ellipse experiments.

The module intentionally keeps experiment registration separate from the V6
and V7 runners.  A runner selects one immutable protocol; the numerical code
then receives the joint domain and margin policy explicitly instead of
silently falling back to the historical V6 chart.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _stable_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class JointDomainSpec:
    """Registered six-axis beta domain used by generation and evaluation."""

    domain_id: str
    bounds_deg: tuple[tuple[float, float], ...]
    source: str

    def __post_init__(self) -> None:
        bounds = np.asarray(self.bounds_deg, dtype=float)
        if bounds.shape != (6, 2):
            raise ValueError("joint domain bounds must have shape (6, 2)")
        if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("joint domain bounds must be finite increasing pairs")

    @property
    def bounds_rad(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self.bounds_deg, dtype=float))

    @property
    def fingerprint(self) -> str:
        return _stable_fingerprint(asdict(self))


_REGISTERED_JOINT_DOMAINS: dict[str, JointDomainSpec] = {
    "current_v6": JointDomainSpec(
        domain_id="current_v6",
        bounds_deg=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-15.0, 15.0), (-15.0, 15.0)),
        source="V2-V6 current true-ellipse chart",
    ),
    "standard_beta34_10deg_v1": JointDomainSpec(
        domain_id="standard_beta34_10deg_v1",
        bounds_deg=((-5.0, 5.0), (-5.0, 5.0), (-10.0, 10.0), (-10.0, 10.0), (-15.0, 15.0), (-15.0, 15.0)),
        source="configs/robot_rods_only_standard_100k.yaml",
    ),
}


def registered_joint_domain(domain_id: str) -> JointDomainSpec:
    try:
        return _REGISTERED_JOINT_DOMAINS[str(domain_id)]
    except KeyError as exc:
        raise ValueError(f"unregistered joint domain: {domain_id}") from exc


@dataclass(frozen=True)
class JointMarginPolicy:
    policy_id: str
    minimum_deg: float
    p01_deg: float
    p05_deg: float
    at_bound_tolerance_deg: float = 1.0e-9
    soft_barrier_margin_deg: float = 0.25
    barrier_weights: tuple[float, ...] = (1.0e-3, 1.0e-2, 1.0e-1)

    @property
    def fingerprint(self) -> str:
        return _stable_fingerprint(asdict(self))


def balanced_joint_margin_policy() -> JointMarginPolicy:
    return JointMarginPolicy(
        policy_id="balanced_min0p05_p010p10_p050p25_v1",
        minimum_deg=0.05,
        p01_deg=0.10,
        p05_deg=0.25,
    )


def joint_margin_matrix_deg(
    beta_rad: np.ndarray,
    *,
    domain: JointDomainSpec,
) -> np.ndarray:
    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    if not np.isfinite(beta).all():
        raise ValueError("joint labels must be finite")
    bounds = np.asarray(domain.bounds_deg, dtype=float)
    beta_deg = np.rad2deg(beta)
    return np.minimum(
        beta_deg - bounds[:, 0][None, :],
        bounds[:, 1][None, :] - beta_deg,
    )


def joint_margin_report(
    beta_rad: np.ndarray,
    *,
    domain: JointDomainSpec,
    at_bound_tolerance_deg: float = 1.0e-9,
) -> dict[str, Any]:
    margins = joint_margin_matrix_deg(beta_rad, domain=domain)
    if len(margins) == 0:
        raise ValueError("joint margin report requires at least one label")
    row_margin = np.min(margins, axis=1)
    tol = float(at_bound_tolerance_deg)
    violations = margins < -tol
    at_bound = (~violations) & (margins <= tol)
    return {
        "joint_domain_id": domain.domain_id,
        "joint_domain_fingerprint": domain.fingerprint,
        "rows": int(len(margins)),
        "min_joint_margin_deg": float(np.min(row_margin)),
        "joint_margin_p01_deg": float(np.percentile(row_margin, 1)),
        "joint_margin_p05_deg": float(np.percentile(row_margin, 5)),
        "out_of_bounds_count": int(np.count_nonzero(violations)),
        "at_bound_count": int(np.count_nonzero(at_bound)),
        "joint_min_margin_deg": {
            f"beta{index + 1}": float(np.min(margins[:, index]))
            for index in range(6)
        },
        "joint_p01_margin_deg": {
            f"beta{index + 1}": float(np.percentile(margins[:, index], 1))
            for index in range(6)
        },
        "joint_p05_margin_deg": {
            f"beta{index + 1}": float(np.percentile(margins[:, index], 5))
            for index in range(6)
        },
    }


def annotate_joint_margins(
    frame: pd.DataFrame,
    *,
    domain: JointDomainSpec,
    beta_columns: Sequence[str] = tuple(f"beta{index}_rad" for index in range(1, 7)),
) -> pd.DataFrame:
    missing = sorted(set(beta_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"joint margin annotation missing columns: {missing}")
    margins = joint_margin_matrix_deg(
        frame[list(beta_columns)].to_numpy(dtype=float),
        domain=domain,
    )
    output = frame.copy()
    for index in range(6):
        output[f"beta{index + 1}_margin_deg"] = margins[:, index]
    output["min_joint_margin_deg"] = np.min(margins, axis=1)
    output["joint_domain_id"] = domain.domain_id
    output["joint_domain_fingerprint"] = domain.fingerprint
    return output


def evaluate_joint_margin_gate(
    report: Mapping[str, Any],
    *,
    policy: JointMarginPolicy,
) -> dict[str, Any]:
    checks = {
        "rows_positive": int(report.get("rows", 0)) > 0,
        "minimum_margin": float(report.get("min_joint_margin_deg", -np.inf)) >= float(policy.minimum_deg),
        "p01_margin": float(report.get("joint_margin_p01_deg", -np.inf)) >= float(policy.p01_deg),
        "p05_margin": float(report.get("joint_margin_p05_deg", -np.inf)) >= float(policy.p05_deg),
        "zero_out_of_bounds": int(report.get("out_of_bounds_count", 1)) == 0,
        "zero_at_bound": int(report.get("at_bound_count", 1)) == 0,
    }
    return {
        "joint_margin_policy_id": policy.policy_id,
        "joint_margin_policy_fingerprint": policy.fingerprint,
        "checks": checks,
        "joint_margin_gate_pass": bool(all(checks.values())),
    }


def joint_margin_barrier_residual(
    beta_rad: np.ndarray,
    *,
    domain: JointDomainSpec | None = None,
    bounds_rad: np.ndarray | None = None,
    soft_margin_deg: float,
    transition_deg: float | None = None,
) -> np.ndarray:
    """Return a smooth one-sided penalty for entering the soft limit shell."""

    if (domain is None) == (bounds_rad is None):
        raise ValueError("provide exactly one of domain or bounds_rad")
    if domain is not None:
        margins = joint_margin_matrix_deg(beta_rad, domain=domain)
    else:
        beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
        bounds = np.asarray(bounds_rad, dtype=float).reshape(6, 2)
        if not np.isfinite(beta).all() or not np.isfinite(bounds).all():
            raise ValueError("joint barrier inputs must be finite")
        beta_deg = np.rad2deg(beta)
        bounds_deg = np.rad2deg(bounds)
        margins = np.minimum(
            beta_deg - bounds_deg[:, 0][None, :],
            bounds_deg[:, 1][None, :] - beta_deg,
        )
    transition = (
        max(float(soft_margin_deg) / 10.0, 1.0e-6)
        if transition_deg is None
        else max(float(transition_deg), 1.0e-6)
    )
    scaled = (float(soft_margin_deg) - margins) / transition
    # logaddexp is a stable softplus. Multiplying by the transition keeps the
    # residual in degrees and makes it essentially zero deep in the domain.
    return (transition * np.logaddexp(0.0, scaled)).reshape(-1)


@dataclass(frozen=True)
class OutputLink:
    """Parameterization between physical beta labels and model targets."""

    link_id: str
    version: int = 1
    epsilon: float = 1.0e-6

    @property
    def fingerprint(self) -> str:
        return _stable_fingerprint(asdict(self))

    def encode(
        self,
        beta_rad: np.ndarray,
        *,
        domain: JointDomainSpec,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
        if not np.isfinite(beta).all():
            raise ValueError("output-link training targets must be finite")
        if self.link_id == "identity":
            return beta.copy(), {
                "output_link_id": self.link_id,
                "output_link_fingerprint": self.fingerprint,
                "target_link_clip_count": 0,
            }
        if self.link_id != "tanh_bounds":
            raise ValueError(f"unsupported output link: {self.link_id}")
        bounds = domain.bounds_rad
        midpoint = np.mean(bounds, axis=1)
        half_range = 0.5 * (bounds[:, 1] - bounds[:, 0])
        normalized = (beta - midpoint[None, :]) / half_range[None, :]
        clipped = np.clip(
            normalized,
            -1.0 + float(self.epsilon),
            1.0 - float(self.epsilon),
        )
        clip_count = int(np.count_nonzero(clipped != normalized))
        latent = np.arctanh(clipped)
        return latent, {
            "output_link_id": self.link_id,
            "output_link_fingerprint": self.fingerprint,
            "target_link_clip_count": clip_count,
        }

    def decode(
        self,
        latent: np.ndarray,
        *,
        domain: JointDomainSpec,
    ) -> np.ndarray:
        values = np.asarray(latent, dtype=float).reshape(-1, 6)
        if self.link_id == "identity":
            return values.copy()
        if self.link_id != "tanh_bounds":
            raise ValueError(f"unsupported output link: {self.link_id}")
        if np.isnan(values).any():
            raise ValueError("output-link predictions must not contain NaN")
        bounds = domain.bounds_rad
        midpoint = np.mean(bounds, axis=1)
        half_range = 0.5 * (bounds[:, 1] - bounds[:, 0])
        normalized = np.clip(
            np.tanh(values),
            -1.0 + float(self.epsilon),
            1.0 - float(self.epsilon),
        )
        return midpoint[None, :] + half_range[None, :] * normalized


_REGISTERED_OUTPUT_LINKS: dict[str, OutputLink] = {
    "identity": OutputLink(link_id="identity"),
    "tanh_bounds": OutputLink(link_id="tanh_bounds"),
}


def registered_output_link(link_id: str) -> OutputLink:
    try:
        return _REGISTERED_OUTPUT_LINKS[str(link_id)]
    except KeyError as exc:
        raise ValueError(f"unregistered output link: {link_id}") from exc


@dataclass(frozen=True)
class RadialBundleProtocol:
    protocol_id: str
    strategy_version: int
    family_id: str
    joint_domain_id: str
    joint_margin_policy_id: str
    start_radius_mm: float
    target_radius_mm: float
    formal_checkpoints_mm: tuple[float, ...]
    cut_indices: tuple[int, ...]
    predictors: tuple[str, ...]
    base_step_mm: float
    retry_steps_mm: tuple[float, ...]
    strict_candidate_limit: int
    rescue_candidate_limit: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _stable_fingerprint(self.as_dict())


def v7_standard_domain_protocol() -> RadialBundleProtocol:
    return RadialBundleProtocol(
        protocol_id="true-ellipse-standard-domain-v7.1",
        strategy_version=1,
        family_id="c0273_a100_py210_pz330_s0243",
        joint_domain_id="standard_beta34_10deg_v1",
        joint_margin_policy_id=balanced_joint_margin_policy().policy_id,
        start_radius_mm=75.0,
        target_radius_mm=120.0,
        formal_checkpoints_mm=(
            75.0,
            80.0,
            82.5,
            85.0,
            87.5,
            90.0,
            92.5,
            95.0,
            97.5,
            100.0,
            102.5,
            105.0,
            107.5,
            110.0,
            112.5,
            115.0,
            117.5,
            120.0,
        ),
        cut_indices=(0, 90, 180, 270),
        predictors=("parent_copy", "radial_secant"),
        base_step_mm=1.0,
        retry_steps_mm=(0.5, 0.25),
        strict_candidate_limit=8,
        rescue_candidate_limit=24,
    )


def compute_radius_frontiers(
    status: pd.DataFrame,
    *,
    anchor_mm: float,
) -> dict[str, float | None]:
    required = {"radius_mm", "strict_gate_pass", "rescue_admission_pass"}
    missing = sorted(required - set(status.columns))
    if missing:
        raise ValueError(f"radius frontier status missing columns: {missing}")
    rows = status.sort_values("radius_mm", kind="stable")
    rows = rows[rows["radius_mm"].to_numpy(dtype=float) >= float(anchor_mm) - 1.0e-9]
    if rows.empty or not np.isclose(float(rows.iloc[0]["radius_mm"]), float(anchor_mm), atol=1.0e-8):
        raise ValueError("radius frontier is missing its registered anchor")
    strict_max: float | None = None
    first_failure: float | None = None
    for _idx, row in rows.iterrows():
        if not bool(row["strict_gate_pass"]):
            first_failure = float(row["radius_mm"])
            break
        strict_max = float(row["radius_mm"])
    admitted = rows[rows["rescue_admission_pass"].fillna(False).astype(bool)]
    exploratory_max = float(admitted["radius_mm"].max()) if not admitted.empty else None
    return {
        "strict_geometry_rmax_mm": strict_max,
        "exploratory_rescue_rmax_mm": exploratory_max,
        "first_strict_failure_mm": first_failure,
    }


def training_anchor_radii(strict_geometry_rmax_mm: float) -> tuple[float, ...]:
    frontier = float(strict_geometry_rmax_mm)
    if frontier <= 100.0 + 1.0e-9:
        return ()
    anchors: list[float] = []
    lower = 100.0
    while lower + 2.5 <= frontier + 1.0e-9:
        anchors.append(round(lower + 1.25, 8))
        lower += 2.5
    return tuple(anchors)


def select_registered_holdouts(
    *,
    strict_geometry_rmax_mm: float,
    support_by_radius: pd.DataFrame,
    minimum_test_radius_mm: float = 105.0,
    validation_gap_mm: float = 7.5,
) -> dict[str, Any]:
    required = {"radius_mm", "strict_support_gate_pass"}
    missing = sorted(required - set(support_by_radius.columns))
    if missing:
        raise ValueError(f"support table missing columns: {missing}")
    support = support_by_radius.copy()
    support["radius_mm"] = support["radius_mm"].astype(float)

    def supported(radius_mm: float) -> bool:
        mask = np.isclose(support["radius_mm"].to_numpy(), float(radius_mm), atol=1.0e-8)
        return bool(np.any(mask) and support.loc[mask, "strict_support_gate_pass"].fillna(False).astype(bool).all())

    ceiling = np.floor(float(strict_geometry_rmax_mm) / 2.5 + 1.0e-9) * 2.5
    candidate = float(ceiling)
    while candidate >= float(minimum_test_radius_mm) - 1.0e-9:
        validation = candidate - float(validation_gap_mm)
        if supported(candidate) and supported(validation):
            return {
                "test_radius_mm": float(candidate),
                "validation_radius_mm": float(validation),
                "minimum_test_radius_mm": float(minimum_test_radius_mm),
                "validation_gap_mm": float(validation_gap_mm),
                "selection_gate_pass": True,
            }
        candidate -= 2.5
    raise ValueError("no support-backed test radius at or above the registered minimum")


def _normalized_offsets(offsets_mm: Sequence[float]) -> tuple[float, ...]:
    offsets = tuple(sorted(float(value) for value in offsets_mm))
    if len(offsets) < 3 or len(offsets) != len(set(offsets)):
        raise ValueError("tube surface requires at least three unique offsets")
    if not any(np.isclose(value, 0.0, atol=1.0e-12) for value in offsets):
        raise ValueError("tube surface offsets must include zero")
    return offsets


def tube_surface_adjacency(
    offsets_mm: Sequence[float],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Return undirected four-neighbor edges for a square normal grid."""

    offsets = _normalized_offsets(offsets_mm)
    edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for left_index, left in enumerate(offsets):
        for right_index, right in enumerate(offsets):
            node = (left, right)
            if left_index + 1 < len(offsets):
                edges.append((node, (offsets[left_index + 1], right)))
            if right_index + 1 < len(offsets):
                edges.append((node, (left, offsets[right_index + 1])))
    return tuple(edges)


def tube_surface_sweep_order(
    offsets_mm: Sequence[float],
    *,
    direction: str,
) -> tuple[tuple[float, float], ...]:
    offsets = _normalized_offsets(offsets_mm)
    center_index = min(range(len(offsets)), key=lambda index: abs(offsets[index]))
    nodes = [(left, right) for left in offsets for right in offsets]

    def key(node: tuple[float, float]) -> tuple[float, float, float, float]:
        left_index = offsets.index(node[0])
        right_index = offsets.index(node[1])
        shell = max(abs(left_index - center_index), abs(right_index - center_index))
        radial = math.hypot(node[0], node[1])
        return float(shell), radial, node[0], node[1]

    outward = tuple(sorted(nodes, key=key))
    normalized = str(direction).strip().lower()
    if normalized == "outward":
        return outward
    if normalized == "inward":
        return tuple(reversed(outward))
    raise ValueError(f"unsupported tube surface sweep direction: {direction}")


def _surface_neighbors(
    pair: tuple[float, float],
    edges: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[tuple[float, float], ...]:
    neighbors: list[tuple[float, float]] = []
    for left, right in edges:
        if pair == left:
            neighbors.append(right)
        elif pair == right:
            neighbors.append(left)
    return tuple(neighbors)


def _surface_hash(frame: pd.DataFrame, beta_columns: Sequence[str]) -> str:
    ordered = frame.sort_values(
        ["delta_n1_mm", "delta_n2_mm", "angle_idx"],
        kind="stable",
    )
    values = np.round(ordered[list(beta_columns)].to_numpy(dtype=float), 12)
    return hashlib.sha256(values.tobytes()).hexdigest()


def optimize_tube_surface(
    *,
    tube_targets: pd.DataFrame,
    initial_surface: pd.DataFrame,
    offsets_mm: Sequence[float],
    domain: JointDomainSpec,
    margin_policy: JointMarginPolicy,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray,
    theta_sign: float,
    stage_specs: Sequence[Mapping[str, float | str]],
    cut_indices: Sequence[int],
    sweep_directions: Sequence[str] = ("outward", "inward"),
    max_nfev: int = 40,
    compute_conditioning: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Jointly relabel a phase×normal-grid surface by block-coordinate sweeps.

    Each curve is solved cyclically.  Its initializer/anchor is the mean of
    the current four-neighbor curves, which couples adjacent normal offsets
    without constructing one prohibitively large 54k-variable least-squares
    problem.  Every requested cyclic cut starts from the same input surface so
    cut invariance remains auditable.
    """

    import true_ellipse_atlas_utils as atlas

    offsets = _normalized_offsets(offsets_mm)
    pairs = tuple((left, right) for left in offsets for right in offsets)
    edges = tube_surface_adjacency(offsets)
    beta_columns = tuple(atlas.BETA_COLS)
    key_columns = ["angle_idx", "delta_n1_mm", "delta_n2_mm"]
    target_required = {*key_columns, *atlas.TARGET_XYZ_COLS}
    initial_required = {*key_columns, *beta_columns}
    missing_targets = sorted(target_required - set(tube_targets.columns))
    missing_initial = sorted(initial_required - set(initial_surface.columns))
    if missing_targets:
        raise ValueError(f"tube surface targets missing columns: {missing_targets}")
    if missing_initial:
        raise ValueError(f"tube surface initializer missing columns: {missing_initial}")
    if tube_targets.duplicated(key_columns).any() or initial_surface.duplicated(key_columns).any():
        raise ValueError("tube surface keys must be unique")
    angle_indices = sorted(int(value) for value in tube_targets["angle_idx"].unique())
    expected_rows = len(angle_indices) * len(pairs)
    if len(tube_targets) != expected_rows or len(initial_surface) != expected_rows:
        raise ValueError("tube surface must contain every angle×offset row")

    target = tube_targets.drop(columns=[column for column in beta_columns if column in tube_targets]).copy()
    initial_beta = initial_surface[key_columns + list(beta_columns)].copy()
    merged = target.merge(
        initial_beta,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )
    if merged[list(beta_columns)].isna().any().any():
        raise ValueError("tube surface initializer does not cover all targets")

    initial_curves: dict[tuple[float, float], pd.DataFrame] = {}
    for pair in pairs:
        mask = np.isclose(merged["delta_n1_mm"].to_numpy(dtype=float), pair[0])
        mask &= np.isclose(merged["delta_n2_mm"].to_numpy(dtype=float), pair[1])
        curve = merged.loc[mask].sort_values("angle_idx", kind="stable").reset_index(drop=True)
        if curve["angle_idx"].astype(int).tolist() != angle_indices:
            raise ValueError(f"tube surface curve is incomplete at offset {pair}")
        initial_curves[pair] = curve

    normalized_directions = tuple(str(value).strip().lower() for value in sweep_directions)
    if not normalized_directions:
        raise ValueError("tube surface requires at least one sweep direction")
    normalized_cuts = tuple(int(value) for value in cut_indices)
    if not normalized_cuts:
        raise ValueError("tube surface requires at least one cyclic cut")

    cut_outputs: list[tuple[int, pd.DataFrame, dict[str, Any]]] = []
    for cut_idx in normalized_cuts:
        if cut_idx not in angle_indices:
            raise ValueError(f"cyclic cut is absent from tube angles: {cut_idx}")
        curves = {pair: frame.copy() for pair, frame in initial_curves.items()}
        curve_reports: list[dict[str, Any]] = []
        for sweep_index, direction in enumerate(normalized_directions):
            for pair in tube_surface_sweep_order(offsets, direction=direction):
                curve = curves[pair].sort_values("angle_idx", kind="stable").reset_index(drop=True)
                neighbor_pairs = _surface_neighbors(pair, edges)
                neighbor_beta = np.mean(
                    [
                        curves[neighbor]
                        .sort_values("angle_idx", kind="stable")[list(beta_columns)]
                        .to_numpy(dtype=float)
                        for neighbor in neighbor_pairs
                    ],
                    axis=0,
                )
                current_beta = curve[list(beta_columns)].to_numpy(dtype=float)
                initializer = 0.5 * current_beta + 0.5 * neighbor_beta
                cut_position = angle_indices.index(cut_idx)
                rotation = np.roll(np.arange(len(angle_indices)), -cut_position)
                rotated_targets = curve.iloc[rotation].reset_index(drop=True)
                rotated_initializer = initializer[rotation]
                corrected, curve_report = atlas.optimize_cyclic_trajectory(
                    targets=rotated_targets,
                    initial_beta=rotated_initializer,
                    bounds=domain.bounds_rad,
                    lengths_m=np.asarray(lengths_m, dtype=float),
                    p_end_local_m=np.asarray(p_end_local_m, dtype=float),
                    theta_sign=float(theta_sign),
                    stages=stage_specs,
                    max_nfev=int(max_nfev),
                    compute_conditioning=bool(compute_conditioning),
                    stop_on_centerline_gate=False,
                )
                corrected = corrected.sort_values("angle_idx", kind="stable").reset_index(drop=True)
                corrected["surface_cut_idx"] = int(cut_idx)
                corrected["surface_sweep_index"] = int(sweep_index)
                corrected["surface_sweep_direction"] = direction
                corrected["surface_neighbor_count"] = int(len(neighbor_pairs))
                corrected["surface_strategy"] = "neighbor_anchor_block_coordinate_v1"
                corrected["tube_success"] = corrected["xyz_residual_mm"].le(1.5)
                curves[pair] = corrected
                curve_reports.append(
                    {
                        "cut_idx": int(cut_idx),
                        "sweep_index": int(sweep_index),
                        "sweep_direction": direction,
                        "pair": [float(pair[0]), float(pair[1])],
                        "neighbor_count": int(len(neighbor_pairs)),
                        "residual_p95_mm": float(curve_report.get("residual_p95_mm", np.inf)),
                        "residual_max_mm": float(curve_report.get("residual_max_mm", np.inf)),
                        "selected_stage": curve_report.get("selected_stage"),
                    }
                )
        surface = pd.concat(
            [curves[pair] for pair in pairs],
            ignore_index=True,
            sort=False,
        ).sort_values(["angle_idx", "delta_n1_mm", "delta_n2_mm"], kind="stable").reset_index(drop=True)
        margin = joint_margin_report(
            surface[list(beta_columns)].to_numpy(dtype=float),
            domain=domain,
            at_bound_tolerance_deg=margin_policy.at_bound_tolerance_deg,
        )
        margin_gate = evaluate_joint_margin_gate(margin, policy=margin_policy)
        cut_report = {
            "cut_idx": int(cut_idx),
            "rows": int(len(surface)),
            "residual_p95_mm": float(np.percentile(surface["xyz_residual_mm"], 95)),
            "residual_max_mm": float(surface["xyz_residual_mm"].max()),
            "label_success_ratio": float(surface["tube_success"].mean()),
            "surface_hash": _surface_hash(surface, beta_columns),
            "curve_reports": curve_reports,
            **margin,
            **margin_gate,
        }
        cut_outputs.append((int(cut_idx), surface, cut_report))

    pair_differences: list[dict[str, Any]] = []
    for (left_cut, left, _left_report), (right_cut, right, _right_report) in combinations(cut_outputs, 2):
        left_beta = left.sort_values(key_columns, kind="stable")[list(beta_columns)].to_numpy(dtype=float)
        right_beta = right.sort_values(key_columns, kind="stable")[list(beta_columns)].to_numpy(dtype=float)
        rms_deg = np.sqrt(np.mean(np.square(left_beta - right_beta), axis=1)) * 180.0 / np.pi
        pair_differences.append(
            {
                "left_cut_idx": int(left_cut),
                "right_cut_idx": int(right_cut),
                "beta_rms_p95_deg": float(np.percentile(rms_deg, 95)),
                "beta_rms_max_deg": float(np.max(rms_deg)),
            }
        )
    cut_pair_p95 = max(
        (float(row["beta_rms_p95_deg"]) for row in pair_differences),
        default=0.0,
    )
    selected_cut, selected_surface, selected_cut_report = min(
        cut_outputs,
        key=lambda item: (
            not bool(item[2]["joint_margin_gate_pass"]),
            float(item[2]["residual_p95_mm"]),
            float(item[2]["residual_max_mm"]),
            int(item[0]),
        ),
    )
    selected_surface = annotate_joint_margins(
        selected_surface,
        domain=domain,
        beta_columns=beta_columns,
    )
    complete = bool(
        len(selected_surface) == expected_rows
        and not selected_surface.duplicated(key_columns).any()
        and int(selected_surface["tube_offset_id"].nunique()) == len(pairs)
    )
    report = {
        "strategy_version": "tube-surface-v1",
        "joint_domain_id": domain.domain_id,
        "joint_domain_fingerprint": domain.fingerprint,
        "joint_margin_policy_id": margin_policy.policy_id,
        "joint_margin_policy_fingerprint": margin_policy.fingerprint,
        "rows": int(len(selected_surface)),
        "expected_rows": int(expected_rows),
        "offset_count": int(len(pairs)),
        "angle_count": int(len(angle_indices)),
        "surface_complete": complete,
        "cut_count": int(len(normalized_cuts)),
        "cut_indices": list(normalized_cuts),
        "sweep_directions": list(normalized_directions),
        "selected_cut_idx": int(selected_cut),
        "selected_surface_hash": _surface_hash(selected_surface, beta_columns),
        "cut_pair_reports": pair_differences,
        "cut_pair_beta_p95_max_deg": float(cut_pair_p95),
        "cut_invariance_gate_pass": bool(cut_pair_p95 <= 1.0),
        "cut_reports": [dict(item[2]) for item in cut_outputs],
        "surface_gate_pass": bool(
            complete
            and cut_pair_p95 <= 1.0
            and selected_cut_report["joint_margin_gate_pass"]
        ),
        **{
            key: selected_cut_report[key]
            for key in (
                "residual_p95_mm",
                "residual_max_mm",
                "label_success_ratio",
                "min_joint_margin_deg",
                "joint_margin_p01_deg",
                "joint_margin_p05_deg",
                "out_of_bounds_count",
                "at_bound_count",
                "joint_margin_gate_pass",
            )
        },
    }
    return selected_surface, report
