from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from quasi_exp.model.kinematics import forward_kinematics
from quasi_exp.model.sampling import beta_to_theta


@dataclass(frozen=True)
class ForwardValidationResult:
    """Summary returned by authoritative forward-model validation routines."""

    passed: bool
    sample_count: int
    valid_count: int
    metrics: Mapping[str, float]
    per_scale: tuple[Mapping[str, float], ...] = ()


@dataclass(frozen=True)
class ForwardEnvironment:
    """Exact CPU/float64 beta-space forward environment."""

    lengths_m: np.ndarray
    p_end_local_m: np.ndarray
    theta_sign: float = -1.0
    beta_bounds_rad: np.ndarray | None = None

    def __post_init__(self) -> None:
        lengths = np.asarray(self.lengths_m, dtype=float)
        end = np.asarray(self.p_end_local_m, dtype=float)
        if lengths.shape != (31,):
            raise ValueError(f"lengths_m must have shape (31,), got {lengths.shape}")
        if end.shape != (4,):
            raise ValueError(f"p_end_local_m must have shape (4,), got {end.shape}")
        if not np.isfinite(lengths).all() or not np.isfinite(end).all():
            raise ValueError("forward-environment inputs must be finite")
        if not np.isfinite(self.theta_sign):
            raise ValueError("theta_sign must be finite")
        bounds = None
        if self.beta_bounds_rad is not None:
            bounds = np.asarray(self.beta_bounds_rad, dtype=float)
            if bounds.shape != (6, 2):
                raise ValueError(f"beta_bounds_rad must have shape (6, 2), got {bounds.shape}")
            if not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
                raise ValueError("beta_bounds_rad must contain finite ordered bounds")
            bounds = bounds.copy()
            bounds.setflags(write=False)
        lengths = lengths.copy()
        end = end.copy()
        lengths.setflags(write=False)
        end.setflags(write=False)
        object.__setattr__(self, "lengths_m", lengths)
        object.__setattr__(self, "p_end_local_m", end)
        object.__setattr__(self, "beta_bounds_rad", bounds)

    def fk(self, beta_rad: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta_rad, dtype=float)
        if beta.ndim == 1 and beta.shape == (6,):
            beta = beta.reshape(1, 6)
        if beta.ndim != 2 or beta.shape[1] != 6:
            raise ValueError(f"beta_rad must have shape (N, 6), got {beta.shape}")
        if not np.isfinite(beta).all():
            raise ValueError("beta_rad must contain only finite values")

        xyz = np.empty((len(beta), 3), dtype=float)
        for index, row in enumerate(beta):
            theta = beta_to_theta(row)
            xyz[index] = forward_kinematics(
                theta,
                self.lengths_m,
                self.p_end_local_m,
                theta_sign=self.theta_sign,
            )[0]
        return xyz

    @property
    def bounds(self) -> np.ndarray:
        """Registered beta bounds consumed by the canonical teacher seam."""

        if self.beta_bounds_rad is None:
            raise ValueError("beta_bounds_rad is required for inverse teacher solving")
        return self.beta_bounds_rad

    def theta(self, beta_rad: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta_rad, dtype=float)
        if beta.ndim == 1 and beta.shape == (6,):
            beta = beta.reshape(1, 6)
        if beta.ndim != 2 or beta.shape[1] != 6:
            raise ValueError(f"beta_rad must have shape (N, 6), got {beta.shape}")
        if not np.isfinite(beta).all():
            raise ValueError("beta_rad must contain only finite values")
        return np.vstack([beta_to_theta(row) for row in beta])

    def jacobian(self, beta_rad: np.ndarray) -> np.ndarray:
        return self.numerical_jacobian(beta_rad)

    def numerical_jacobian(self, beta_rad: np.ndarray, *, eps_rad: float = 1.0e-4) -> np.ndarray:
        beta = np.asarray(beta_rad, dtype=float)
        if beta.shape != (6,):
            raise ValueError(f"beta_rad must have shape (6,), got {beta.shape}")
        if not np.isfinite(beta).all():
            raise ValueError("beta_rad must contain only finite values")
        if not np.isfinite(eps_rad) or eps_rad <= 0.0:
            raise ValueError("eps_rad must be finite and positive")

        offsets = np.eye(6, dtype=float) * float(eps_rad)
        plus = self.fk(beta[None, :] + offsets)
        minus = self.fk(beta[None, :] - offsets)
        return ((plus - minus) / (2.0 * float(eps_rad))).T

    @staticmethod
    def jacobian_metrics(jacobian: np.ndarray) -> dict[str, float]:
        jac = np.asarray(jacobian, dtype=float)
        if jac.shape != (3, 6):
            raise ValueError(f"jacobian must have shape (3, 6), got {jac.shape}")
        if not np.isfinite(jac).all():
            raise ValueError("jacobian must contain only finite values")
        singular_values = np.linalg.svd(jac, compute_uv=False)
        sigma1, sigma2, sigma3 = (float(value) for value in singular_values)
        kappa = float(np.inf) if sigma3 <= 0.0 else float(sigma1 / sigma3)
        return {
            "sigma1_m": sigma1,
            "sigma2_m": sigma2,
            "sigma3_m": sigma3,
            "kappa": kappa,
        }

    def validate_jacobian(
        self,
        beta_rad: np.ndarray,
        *,
        perturbation_scales_rad: tuple[float, ...] = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2),
        validation_scale_rad: float = 1.0e-4,
        jacobian_eps_rad: float = 1.0e-5,
        direction_seed: int = 0,
        relative_error_p95_limit: float = 0.01,
        absolute_error_p95_m_limit: float = 5.0e-5,
    ) -> ForwardValidationResult:
        beta = np.asarray(beta_rad, dtype=float)
        if beta.ndim == 1 and beta.shape == (6,):
            beta = beta.reshape(1, 6)
        if beta.ndim != 2 or beta.shape[1] != 6:
            raise ValueError(f"beta_rad must have shape (N, 6), got {beta.shape}")
        if len(beta) == 0:
            raise ValueError("beta_rad must contain at least one sample")
        if not np.isfinite(beta).all():
            raise ValueError("beta_rad must contain only finite values")

        scales = tuple(float(scale) for scale in perturbation_scales_rad)
        if not scales or not np.isfinite(scales).all() or any(scale <= 0.0 for scale in scales):
            raise ValueError("perturbation_scales_rad must contain finite positive values")
        validation_scale = float(validation_scale_rad)
        if validation_scale not in scales:
            raise ValueError("validation_scale_rad must be present in perturbation_scales_rad")
        limits = (float(relative_error_p95_limit), float(absolute_error_p95_m_limit))
        if not np.isfinite(limits).all() or any(limit < 0.0 for limit in limits):
            raise ValueError("Jacobian validation limits must be finite and non-negative")

        jacobians = np.stack(
            [self.numerical_jacobian(row, eps_rad=jacobian_eps_rad) for row in beta],
            axis=0,
        )
        base_xyz = self.fk(beta)
        rng = np.random.default_rng(int(direction_seed))
        rows: list[dict[str, float]] = []
        selected_absolute = np.empty(0, dtype=float)
        selected_relative = np.empty(0, dtype=float)
        selected_valid = np.empty(0, dtype=bool)

        for scale in scales:
            directions = rng.normal(size=(len(beta), 6))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            delta = directions * scale
            predicted = np.einsum("nij,nj->ni", jacobians, delta)
            exact = self.fk(beta + delta) - base_xyz
            absolute = np.linalg.norm(predicted - exact, axis=1)
            denominator = np.maximum(np.linalg.norm(exact, axis=1), np.finfo(float).eps)
            relative = absolute / denominator
            valid = np.isfinite(absolute) & np.isfinite(relative)
            finite_absolute = absolute[valid]
            finite_relative = relative[valid]
            row = {
                "perturbation_scale_rad": scale,
                "valid_count": float(np.count_nonzero(valid)),
                "nonfinite_count": float(len(beta) - np.count_nonzero(valid)),
                "absolute_error_p95_m": self._percentile_or_inf(finite_absolute, 95.0),
                "absolute_error_max_m": self._max_or_inf(finite_absolute),
                "relative_error_p95": self._percentile_or_inf(finite_relative, 95.0),
                "relative_error_max": self._max_or_inf(finite_relative),
            }
            rows.append(row)
            if scale == validation_scale:
                selected_absolute = finite_absolute
                selected_relative = finite_relative
                selected_valid = valid

        valid_count = int(np.count_nonzero(selected_valid))
        metrics = {
            "validation_scale_rad": validation_scale,
            "absolute_error_p95_m": self._percentile_or_inf(selected_absolute, 95.0),
            "absolute_error_max_m": self._max_or_inf(selected_absolute),
            "relative_error_p95": self._percentile_or_inf(selected_relative, 95.0),
            "relative_error_max": self._max_or_inf(selected_relative),
            "nonfinite_count": float(len(beta) - valid_count),
        }
        passed = bool(
            valid_count == len(beta)
            and metrics["absolute_error_p95_m"] <= absolute_error_p95_m_limit
            and metrics["relative_error_p95"] <= relative_error_p95_limit
        )
        return ForwardValidationResult(
            passed=passed,
            sample_count=len(beta),
            valid_count=valid_count,
            metrics=metrics,
            per_scale=tuple(rows),
        )

    def evaluate_synthetic_recovery(
        self,
        source_beta_rad: np.ndarray,
        recovered_beta_rad: np.ndarray,
        *,
        solver_success: np.ndarray | None = None,
        beta_bounds_rad: np.ndarray | None = None,
        residual_p95_limit_m: float = 5.0e-4,
        residual_max_limit_m: float = 3.0e-3,
        success_rate_limit: float = 0.99,
    ) -> ForwardValidationResult:
        source = np.asarray(source_beta_rad, dtype=float)
        recovered = np.asarray(recovered_beta_rad, dtype=float)
        if source.ndim == 1 and source.shape == (6,):
            source = source.reshape(1, 6)
        if recovered.ndim == 1 and recovered.shape == (6,):
            recovered = recovered.reshape(1, 6)
        if source.ndim != 2 or source.shape[1] != 6:
            raise ValueError(f"source_beta_rad must have shape (N, 6), got {source.shape}")
        if recovered.shape != source.shape:
            raise ValueError(f"recovered_beta_rad must have shape {source.shape}, got {recovered.shape}")
        if len(source) == 0:
            raise ValueError("source_beta_rad must contain at least one sample")
        if not np.isfinite(source).all():
            raise ValueError("source_beta_rad must contain only finite values")

        limits = (
            float(residual_p95_limit_m),
            float(residual_max_limit_m),
            float(success_rate_limit),
        )
        if (
            not np.isfinite(limits).all()
            or limits[0] < 0.0
            or limits[1] < 0.0
            or not 0.0 <= limits[2] <= 1.0
        ):
            raise ValueError("synthetic recovery limits must be finite and valid")

        finite_recovery = np.isfinite(recovered).all(axis=1)
        if solver_success is None:
            reported_success = np.ones(len(source), dtype=bool)
        else:
            reported_success = np.asarray(solver_success, dtype=bool)
            if reported_success.shape != (len(source),):
                raise ValueError(f"solver_success must have shape ({len(source)},), got {reported_success.shape}")
        successful = reported_success & finite_recovery

        target_xyz = self.fk(source)
        achieved_xyz = np.full((len(source), 3), np.nan, dtype=float)
        if np.any(finite_recovery):
            achieved_xyz[finite_recovery] = self.fk(recovered[finite_recovery])
        residual_m = np.linalg.norm(achieved_xyz[successful] - target_xyz[successful], axis=1)

        bounds = self.beta_bounds_rad if beta_bounds_rad is None else np.asarray(beta_bounds_rad, dtype=float)
        bounds_violation = np.zeros(len(source), dtype=bool)
        if bounds is not None:
            if bounds.shape != (6, 2):
                raise ValueError(f"beta_bounds_rad must have shape (6, 2), got {bounds.shape}")
            if not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
                raise ValueError("beta_bounds_rad must contain finite ordered bounds")
            bounds_violation[finite_recovery] = np.any(
                (recovered[finite_recovery] < bounds[:, 0]) | (recovered[finite_recovery] > bounds[:, 1]),
                axis=1,
            )

        valid_count = int(np.count_nonzero(successful))
        metrics = {
            "success_rate": float(valid_count / len(source)),
            "residual_p95_m": self._percentile_or_inf(residual_m, 95.0),
            "residual_max_m": self._max_or_inf(residual_m),
            "bounds_violation_count": float(np.count_nonzero(bounds_violation)),
            "nonfinite_count": float(len(source) - np.count_nonzero(finite_recovery)),
        }
        passed = bool(
            metrics["success_rate"] >= success_rate_limit
            and metrics["residual_p95_m"] <= residual_p95_limit_m
            and metrics["residual_max_m"] <= residual_max_limit_m
            and metrics["bounds_violation_count"] == 0.0
            and metrics["nonfinite_count"] == 0.0
        )
        return ForwardValidationResult(
            passed=passed,
            sample_count=len(source),
            valid_count=valid_count,
            metrics=metrics,
        )

    @staticmethod
    def _percentile_or_inf(values: np.ndarray, percentile: float) -> float:
        return float(np.percentile(values, percentile)) if len(values) else float(np.inf)

    @staticmethod
    def _max_or_inf(values: np.ndarray) -> float:
        return float(np.max(values)) if len(values) else float(np.inf)
