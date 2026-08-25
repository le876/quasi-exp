"""Canonical identity and null-space gauge policies for BACRA V14.2R retry7.

The canonical anchor is a registered policy choice.  It is not inferred from
which inverse chart happens to cover the most task nodes after a larger search
budget.  This module intentionally owns that identity decision independently
from chart growth and audit execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Protocol, Sequence

import numpy as np
from scipy.optimize import minimize

from .canonical import weighted_damped_pinv
from .canonical_atlas import (
    AtlasCandidate,
    AtlasTaskNode,
    CandidateKey,
    ContinuationAdapter,
    ContinuationOutcome,
)
from .section_first_atlas import RootedSectionChart


@dataclass(frozen=True)
class CanonicalAnchorPolicy:
    ordered_anchor_root_keys: tuple[CandidateKey, ...]
    minimum_component_coverage: float = 0.9
    minimum_coherent_measure: float = 0.6
    allow_stitchable_extensions: bool = True
    require_anchor_selected: bool = True

    def __post_init__(self) -> None:
        roots = tuple(
            dict.fromkeys(
                (int(node_id), str(candidate_id))
                for node_id, candidate_id in self.ordered_anchor_root_keys
            )
        )
        if not roots and self.require_anchor_selected:
            raise ValueError("a required canonical anchor policy needs anchor roots")
        if not 0.0 <= float(self.minimum_component_coverage) <= 1.0:
            raise ValueError("minimum_component_coverage must be a fraction")
        if not 0.0 <= float(self.minimum_coherent_measure) <= 1.0:
            raise ValueError("minimum_coherent_measure must be a fraction")
        object.__setattr__(self, "ordered_anchor_root_keys", roots)


@dataclass(frozen=True)
class GaugeCorrectorPolicy:
    mode: Literal["predictor_proximal", "anchor_potential", "proximal_slsqp"]
    gauge_gain: float
    maximum_gauge_step_deg: float
    anchor_weight: float
    cartesian_step_mm: float
    maximum_iterations: int
    damping: float = 1.0e-3
    beta_weights: tuple[float, ...] = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0)
    residual_tolerance_mm: float = 3.0

    def __post_init__(self) -> None:
        if self.mode not in (
            "predictor_proximal",
            "anchor_potential",
            "proximal_slsqp",
        ):
            raise ValueError(f"unsupported gauge mode: {self.mode}")
        if len(self.beta_weights) != 6 or min(self.beta_weights) <= 0.0:
            raise ValueError("beta_weights must contain six positive values")
        positive = (
            self.gauge_gain,
            self.maximum_gauge_step_deg,
            self.cartesian_step_mm,
            self.residual_tolerance_mm,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("gauge budgets must be finite and positive")
        if self.anchor_weight < 0.0 or self.maximum_iterations < 1 or self.damping < 0.0:
            raise ValueError("gauge iteration and damping budgets are invalid")


class GaugeKinematics(Protocol):
    bounds: np.ndarray

    def fk(self, beta_rad: np.ndarray) -> np.ndarray: ...

    def fk_and_jacobian(
        self, beta_rad: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


def _wrapped_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def gauge_locked_predictor_corrector(
    environment: GaugeKinematics,
    *,
    policy: GaugeCorrectorPolicy,
    anchor_beta_rad: np.ndarray | None = None,
) -> ContinuationAdapter:
    """Build a target-blind gauge-locked analytic predictor/corrector."""

    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    spans = bounds[:, 1] - bounds[:, 0]
    if np.any(spans <= 0.0):
        raise ValueError("gauge corrector requires positive beta ranges")
    weights = np.asarray(policy.beta_weights, dtype=float).reshape(6)
    anchor = (
        None
        if anchor_beta_rad is None
        else np.asarray(anchor_beta_rad, dtype=float).reshape(6).copy()
    )
    if policy.mode in ("anchor_potential", "proximal_slsqp") and anchor is None:
        raise ValueError(f"{policy.mode} requires a frozen anchor posture")
    maximum_gauge_step_rad = math.radians(policy.maximum_gauge_step_deg)

    def continuation(
        source: AtlasCandidate, target: AtlasTaskNode
    ) -> ContinuationOutcome:
        try:
            source_xyz, source_jac = environment.fk_and_jacobian(source.beta_rad)
            predictor = source.beta_rad + weighted_damped_pinv(
                source_jac[0], damping=policy.damping, weights=weights
            ) @ (target.xyz_m - source_xyz[0])
            if not np.all(
                (predictor >= bounds[:, 0] - 1.0e-12)
                & (predictor <= bounds[:, 1] + 1.0e-12)
            ):
                predictor = source.beta_rad.copy()
            if policy.mode == "proximal_slsqp":
                cache_beta: np.ndarray | None = None
                cache_xyz: np.ndarray | None = None
                cache_jacobian: np.ndarray | None = None

                def evaluate(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                    nonlocal cache_beta, cache_xyz, cache_jacobian
                    beta_value = np.asarray(value, dtype=float).reshape(6)
                    if cache_beta is None or not np.array_equal(
                        beta_value, cache_beta
                    ):
                        xyz_value, jacobian_value = environment.fk_and_jacobian(
                            beta_value
                        )
                        cache_beta = beta_value.copy()
                        cache_xyz = xyz_value[0].copy()
                        cache_jacobian = jacobian_value[0].copy()
                    assert cache_xyz is not None and cache_jacobian is not None
                    return cache_xyz, cache_jacobian

                tolerance_m = policy.residual_tolerance_mm / 1000.0
                constraint_tolerance_m = 0.99 * tolerance_m

                # C4 begins with an explicit weighted-DLS correction, then
                # solves the proximal objective under a Cartesian residual
                # constraint.  The only posture references are the source
                # predictor and the frozen anchor; target canonical beta is
                # deliberately unavailable to this kernel.
                initial = predictor.copy()
                dls_iterations = 0
                for dls_iterations in range(
                    1, min(32, max(1, policy.maximum_iterations // 4)) + 1
                ):
                    xyz_value, jacobian_value = environment.fk_and_jacobian(initial)
                    task_error = target.xyz_m - xyz_value[0]
                    if float(np.linalg.norm(task_error)) <= tolerance_m:
                        break
                    task_step = weighted_damped_pinv(
                        jacobian_value[0],
                        damping=policy.damping,
                        weights=weights,
                    ) @ task_error
                    accepted = False
                    current_norm = float(np.linalg.norm(task_error))
                    for scale in (1.0, 0.5, 0.25, 0.125):
                        candidate = initial + scale * task_step
                        if not np.all(
                            (candidate >= bounds[:, 0] - 1.0e-12)
                            & (candidate <= bounds[:, 1] + 1.0e-12)
                        ):
                            continue
                        candidate_norm = float(
                            np.linalg.norm(environment.fk(candidate)[0] - target.xyz_m)
                        )
                        if candidate_norm < current_norm - 1.0e-15:
                            initial = candidate
                            accepted = True
                            break
                    if not accepted:
                        break

                def objective(value: np.ndarray) -> float:
                    beta_value = np.asarray(value, dtype=float).reshape(6)
                    predictor_delta = _wrapped_delta(beta_value, predictor) / spans
                    anchor_delta = _wrapped_delta(beta_value, anchor) / spans
                    return float(
                        policy.gauge_gain * (predictor_delta @ predictor_delta)
                        + policy.anchor_weight * (anchor_delta @ anchor_delta)
                    )

                def objective_jacobian(value: np.ndarray) -> np.ndarray:
                    beta_value = np.asarray(value, dtype=float).reshape(6)
                    predictor_gradient = (
                        2.0
                        * policy.gauge_gain
                        * _wrapped_delta(beta_value, predictor)
                        / np.square(spans)
                    )
                    anchor_gradient = (
                        2.0
                        * policy.anchor_weight
                        * _wrapped_delta(beta_value, anchor)
                        / np.square(spans)
                    )
                    return predictor_gradient + anchor_gradient

                def task_constraint(value: np.ndarray) -> float:
                    xyz_value, _jacobian = evaluate(value)
                    normalized = (
                        xyz_value - target.xyz_m
                    ) / constraint_tolerance_m
                    return float(1.0 - normalized @ normalized)

                def task_constraint_jacobian(value: np.ndarray) -> np.ndarray:
                    xyz_value, jacobian_value = evaluate(value)
                    return (
                        -2.0
                        * jacobian_value.T
                        @ (xyz_value - target.xyz_m)
                        / (constraint_tolerance_m * constraint_tolerance_m)
                    )

                solved = minimize(
                    objective,
                    initial,
                    method="SLSQP",
                    jac=objective_jacobian,
                    bounds=[tuple(map(float, row)) for row in bounds],
                    constraints=(
                        {
                            "type": "ineq",
                            "fun": task_constraint,
                            "jac": task_constraint_jacobian,
                        },
                    ),
                    options={
                        "maxiter": int(policy.maximum_iterations),
                        "ftol": 1.0e-12,
                        "disp": False,
                    },
                )
                beta = np.asarray(solved.x, dtype=float).reshape(6)
                residual_mm = float(
                    np.linalg.norm(environment.fk(beta)[0] - target.xyz_m) * 1000.0
                )
                actual_bounds = bool(
                    np.all(beta >= bounds[:, 0] - 1.0e-12)
                    and np.all(beta <= bounds[:, 1] + 1.0e-12)
                )
                return ContinuationOutcome(
                    beta,
                    residual_mm,
                    bool(
                        solved.success
                        and residual_mm <= policy.residual_tolerance_mm
                        and actual_bounds
                    ),
                    actual_bounds,
                    dls_iterations + int(getattr(solved, "nit", 0)),
                    (
                        "gauge[weighted_dls>proximal_slsqp;"
                        f"status={solved.status};step_mm={policy.cartesian_step_mm}]"
                    ),
                )
            beta = predictor.copy()
            iterations = 0
            for iterations in range(1, policy.maximum_iterations + 1):
                xyz, jacobian = environment.fk_and_jacobian(beta)
                task_error = target.xyz_m - xyz[0]
                pinv = weighted_damped_pinv(
                    jacobian[0], damping=policy.damping, weights=weights
                )
                task_step = pinv @ task_error
                projector = np.eye(6) - pinv @ jacobian[0]
                gradient = _wrapped_delta(beta, predictor) / np.square(spans)
                if anchor is not None:
                    gradient = gradient + policy.anchor_weight * (
                        _wrapped_delta(beta, anchor) / np.square(spans)
                    )
                gauge_step = -policy.gauge_gain * (projector @ gradient)
                gauge_norm = float(np.linalg.norm(gauge_step))
                if gauge_norm > maximum_gauge_step_rad:
                    gauge_step *= maximum_gauge_step_rad / gauge_norm
                    gauge_norm = maximum_gauge_step_rad
                current_residual = float(np.linalg.norm(task_error) * 1000.0)
                current_gauge_objective = float(
                    np.sum(np.square(_wrapped_delta(beta, predictor) / spans))
                    + (
                        policy.anchor_weight
                        * np.sum(np.square(_wrapped_delta(beta, anchor) / spans))
                        if anchor is not None
                        else 0.0
                    )
                )
                if (
                    current_residual <= policy.residual_tolerance_mm
                    and gauge_norm <= 1.0e-12
                ):
                    break
                accepted = False
                for scale in (1.0, 0.5, 0.25, 0.125):
                    candidate = beta + scale * (task_step + gauge_step)
                    if not np.all(
                        (candidate >= bounds[:, 0] - 1.0e-12)
                        & (candidate <= bounds[:, 1] + 1.0e-12)
                    ):
                        continue
                    candidate_xyz = environment.fk(candidate)[0]
                    candidate_residual = float(
                        np.linalg.norm(candidate_xyz - target.xyz_m) * 1000.0
                    )
                    candidate_objective = float(
                        np.sum(
                            np.square(_wrapped_delta(candidate, predictor) / spans)
                        )
                        + (
                            policy.anchor_weight
                            * np.sum(
                                np.square(_wrapped_delta(candidate, anchor) / spans)
                            )
                            if anchor is not None
                            else 0.0
                        )
                    )
                    residual_ok = candidate_residual <= max(
                        current_residual - 1.0e-12,
                        policy.residual_tolerance_mm,
                    )
                    gauge_ok = candidate_objective < current_gauge_objective - 1.0e-15
                    if residual_ok or (
                        current_residual <= policy.residual_tolerance_mm
                        and candidate_residual <= policy.residual_tolerance_mm
                        and gauge_ok
                    ):
                        beta = candidate
                        accepted = True
                        break
                if not accepted:
                    break
            residual_mm = float(
                np.linalg.norm(environment.fk(beta)[0] - target.xyz_m) * 1000.0
            )
            actual_bounds = bool(
                np.all(beta >= bounds[:, 0] - 1.0e-12)
                and np.all(beta <= bounds[:, 1] + 1.0e-12)
            )
            return ContinuationOutcome(
                beta,
                residual_mm,
                residual_mm <= policy.residual_tolerance_mm and actual_bounds,
                actual_bounds,
                iterations,
                (
                    f"gauge[{policy.mode};gain={policy.gauge_gain};"
                    f"step_mm={policy.cartesian_step_mm}]"
                ),
            )
        except Exception as error:
            return ContinuationOutcome(
                source.beta_rad,
                1.0e300,
                False,
                False,
                0,
                f"gauge_exception:{type(error).__name__}",
            )

    return continuation


@dataclass(frozen=True)
class AnchorComponentSelection:
    component: tuple[str, ...]
    anchor_root_key: CandidateKey | None
    anchor_rank: int | None
    anchor_qualified: bool
    anchor_component_selected: bool
    component_coverage: float
    coherent_measure: float
    fallback_reason: str | None


def _component_nodes(
    component: Sequence[str], charts: Mapping[str, RootedSectionChart]
) -> set[int]:
    return set().union(
        *(set(charts[chart_id].selected_by_node) for chart_id in component)
    ) if component else set()


def _coherent_measure(
    node_ids: set[int], task_nodes: Sequence[AtlasTaskNode]
) -> float:
    if not node_ids:
        return 0.0
    node_by_id = {int(node.node_id): node for node in task_nodes}
    remaining = set(node_ids)
    largest = 0
    while remaining:
        seed = min(remaining)
        queue = [seed]
        component: set[int] = set()
        while queue:
            current = queue.pop()
            if current in component:
                continue
            component.add(current)
            node = node_by_id.get(current)
            if node is not None:
                queue.extend(
                    int(neighbor)
                    for neighbor in node.neighbor_node_ids
                    if int(neighbor) in node_ids and int(neighbor) not in component
                )
        remaining -= component
        largest = max(largest, len(component))
    return largest / max(1, len(task_nodes))


def select_anchor_locked_component(
    components: Sequence[tuple[str, ...]],
    charts: Mapping[str, RootedSectionChart],
    task_nodes: Sequence[AtlasTaskNode],
    policy: CanonicalAnchorPolicy,
) -> AnchorComponentSelection:
    """Select the first qualified registered anchor component.

    Coverage orders components only *within the same anchor rank*.  A larger
    incompatible component can never replace an already qualified higher
    priority anchor.
    """

    total_nodes = max(1, len(task_nodes))
    normalized_components = tuple(tuple(sorted(component)) for component in components)
    for anchor_rank, anchor_root in enumerate(policy.ordered_anchor_root_keys):
        candidates: list[tuple[str, ...]] = []
        for component in normalized_components:
            anchor_charts = tuple(
                chart_id
                for chart_id in component
                if charts[chart_id].root_key == anchor_root
            )
            if not anchor_charts:
                continue
            if policy.allow_stitchable_extensions:
                candidates.append(component)
            else:
                candidates.extend((chart_id,) for chart_id in anchor_charts)
        ranked = sorted(
            candidates,
            key=lambda component: (
                -len(_component_nodes(component, charts)),
                len(component),
                tuple(charts[chart_id].root_key for chart_id in component),
                component,
            ),
        )
        for component in ranked:
            nodes = _component_nodes(component, charts)
            coverage = len(nodes) / total_nodes
            coherent = _coherent_measure(nodes, task_nodes)
            if (
                coverage + 1.0e-12 >= policy.minimum_component_coverage
                and coherent + 1.0e-12 >= policy.minimum_coherent_measure
            ):
                return AnchorComponentSelection(
                    component=component,
                    anchor_root_key=anchor_root,
                    anchor_rank=anchor_rank,
                    anchor_qualified=True,
                    anchor_component_selected=True,
                    component_coverage=float(coverage),
                    coherent_measure=float(coherent),
                    fallback_reason=(
                        None
                        if anchor_rank == 0
                        else "higher_priority_anchor_not_qualified"
                    ),
                )

    if policy.require_anchor_selected:
        return AnchorComponentSelection(
            component=(),
            anchor_root_key=None,
            anchor_rank=None,
            anchor_qualified=False,
            anchor_component_selected=False,
            component_coverage=0.0,
            coherent_measure=0.0,
            fallback_reason="no_registered_anchor_component_qualified",
        )

    ranked_fallback = sorted(
        normalized_components,
        key=lambda component: (
            -len(_component_nodes(component, charts)),
            len(component),
            component,
        ),
    )
    component = ranked_fallback[0] if ranked_fallback else ()
    nodes = _component_nodes(component, charts)
    coverage = len(nodes) / total_nodes
    coherent = _coherent_measure(nodes, task_nodes)
    return AnchorComponentSelection(
        component=component,
        anchor_root_key=None,
        anchor_rank=None,
        anchor_qualified=False,
        anchor_component_selected=False,
        component_coverage=float(coverage),
        coherent_measure=float(coherent),
        fallback_reason="unanchored_coverage_fallback" if component else "no_component",
    )


__all__ = [
    "AnchorComponentSelection",
    "CanonicalAnchorPolicy",
    "GaugeCorrectorPolicy",
    "gauge_locked_predictor_corrector",
    "select_anchor_locked_component",
]
