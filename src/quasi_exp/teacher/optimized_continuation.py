"""Predictor-corrector using the V14.3 analytic FK/Jacobian seam."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from quasi_exp.teacher.canonical import weighted_damped_pinv
from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasTaskNode,
    ContinuationAdapter,
    ContinuationOutcome,
)
from quasi_exp.teacher.optimized_forward import OptimizedForwardEnvironment


def make_optimized_predictor_corrector_continuation(
    environment: OptimizedForwardEnvironment,
    *,
    damping: float = 1.0e-3,
    beta_weights: Sequence[float] = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0),
    max_corrector_iterations: int = 100,
    residual_tolerance_mm: float = 3.0,
) -> ContinuationAdapter:
    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    weights = np.asarray(beta_weights, dtype=float).reshape(6)

    def continuation(source: AtlasCandidate, target: AtlasTaskNode) -> ContinuationOutcome:
        try:
            source_xyz, source_jacobian = environment.fk_and_jacobian(source.beta_rad)
            predictor = source.beta_rad + weighted_damped_pinv(
                source_jacobian[0], damping=float(damping), weights=weights
            ) @ (target.xyz_m - source_xyz[0])
            predictor_in_bounds = bool(
                np.all(predictor >= bounds[:, 0] - 1e-12)
                and np.all(predictor <= bounds[:, 1] + 1e-12)
            )
            initial = predictor if predictor_in_bounds else source.beta_rad.copy()
            cache_beta: np.ndarray | None = None
            cache_xyz: np.ndarray | None = None
            cache_jac: np.ndarray | None = None

            def evaluate(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                nonlocal cache_beta, cache_xyz, cache_jac
                row = np.asarray(beta, dtype=float).reshape(6)
                if cache_beta is None or not np.array_equal(row, cache_beta):
                    xyz, jac = environment.fk_and_jacobian(row)
                    cache_beta, cache_xyz, cache_jac = row.copy(), xyz[0], jac[0]
                assert cache_xyz is not None and cache_jac is not None
                return cache_xyz, cache_jac

            solved = least_squares(
                lambda beta: (evaluate(beta)[0] - target.xyz_m) / 0.001,
                initial,
                jac=lambda beta: evaluate(beta)[1] / 0.001,
                bounds=(bounds[:, 0], bounds[:, 1]),
                max_nfev=int(max_corrector_iterations),
            )
            beta = np.asarray(solved.x, dtype=float).reshape(6)
            residual_mm = float(np.linalg.norm(environment.fk(beta)[0] - target.xyz_m) * 1000)
            actual_bounds = bool(
                np.all(beta >= bounds[:, 0] - 1e-12)
                and np.all(beta <= bounds[:, 1] + 1e-12)
            )
            return ContinuationOutcome(
                beta,
                residual_mm,
                bool(solved.success and residual_mm <= residual_tolerance_mm and actual_bounds),
                actual_bounds,
                int(solved.nfev),
                str(solved.status) if predictor_in_bounds else f"predictor_out_of_bounds_fallback:{solved.status}",
            )
        except Exception as error:
            return ContinuationOutcome(
                source.beta_rad, 1.0e300, False, False, 0,
                f"exception:{type(error).__name__}",
            )

    return continuation


def make_iterative_weighted_dls_continuation(
    environment: OptimizedForwardEnvironment,
    *,
    damping: float = 2.0e-3,
    beta_weights: Sequence[float] = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0),
    max_corrector_iterations: int = 200,
    residual_tolerance_mm: float = 3.0,
) -> ContinuationAdapter:
    """Target-blind bounded iterative weighted-DLS corrector.

    This is intentionally a distinct numerical kernel from SciPy bounded
    least-squares so the registered R1/R2 chain is executable rather than a
    descriptive label.
    """

    bounds = np.asarray(environment.bounds, dtype=float).reshape(6, 2)
    weights = np.asarray(beta_weights, dtype=float).reshape(6)

    def continuation(
        source: AtlasCandidate, target: AtlasTaskNode
    ) -> ContinuationOutcome:
        beta = source.beta_rad.copy()
        try:
            for iteration in range(1, int(max_corrector_iterations) + 1):
                xyz, jacobian = environment.fk_and_jacobian(beta)
                error = target.xyz_m - xyz[0]
                residual_mm = float(np.linalg.norm(error) * 1000.0)
                if residual_mm <= float(residual_tolerance_mm):
                    return ContinuationOutcome(
                        beta,
                        residual_mm,
                        True,
                        True,
                        iteration - 1,
                        "weighted_dls_converged",
                    )
                step = weighted_damped_pinv(
                    jacobian[0], damping=float(damping), weights=weights
                ) @ error
                accepted = False
                for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    candidate = beta + scale * step
                    if not np.all(
                        (candidate >= bounds[:, 0] - 1.0e-12)
                        & (candidate <= bounds[:, 1] + 1.0e-12)
                    ):
                        continue
                    candidate_residual = float(
                        np.linalg.norm(environment.fk(candidate)[0] - target.xyz_m)
                        * 1000.0
                    )
                    if candidate_residual < residual_mm - 1.0e-12:
                        beta = candidate
                        accepted = True
                        break
                if not accepted:
                    return ContinuationOutcome(
                        beta,
                        residual_mm,
                        False,
                        True,
                        iteration,
                        "weighted_dls_line_search_stalled",
                    )
            residual_mm = float(
                np.linalg.norm(environment.fk(beta)[0] - target.xyz_m) * 1000.0
            )
            return ContinuationOutcome(
                beta,
                residual_mm,
                residual_mm <= float(residual_tolerance_mm),
                True,
                int(max_corrector_iterations),
                "weighted_dls_iteration_limit",
            )
        except Exception as error:
            return ContinuationOutcome(
                source.beta_rad,
                1.0e300,
                False,
                False,
                0,
                f"weighted_dls_exception:{type(error).__name__}",
            )

    return continuation


__all__ = [
    "make_iterative_weighted_dls_continuation",
    "make_optimized_predictor_corrector_continuation",
]
