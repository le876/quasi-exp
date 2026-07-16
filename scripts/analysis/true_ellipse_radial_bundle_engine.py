"""Domain-aware primitives shared by registered true-ellipse experiments.

The module intentionally keeps experiment registration separate from the V6
and V7 runners.  A runner selects one immutable protocol; the numerical code
then receives the joint domain and margin policy explicitly instead of
silently falling back to the historical V6 chart.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

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
    domain: JointDomainSpec,
    soft_margin_deg: float,
    transition_deg: float | None = None,
) -> np.ndarray:
    """Return a smooth one-sided penalty for entering the soft limit shell."""

    margins = joint_margin_matrix_deg(beta_rad, domain=domain)
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
