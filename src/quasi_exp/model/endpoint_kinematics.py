"""Fast, authoritative endpoint kinematics in the six-dimensional beta space.

The public interface deliberately hides the 30-joint expansion and the DH
derivative propagation.  Callers can evaluate one or many beta rows and ask
for the analytic 3x6 endpoint Jacobian without performing finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _beta_expansion_matrix() -> np.ndarray:
    expansion = np.zeros((30, 6), dtype=np.float64)
    for joint in range(30):
        section = joint // 10
        expansion[joint, 2 * section + (joint % 2)] = 1.0
    expansion.setflags(write=False)
    return expansion


_BETA_EXPANSION = _beta_expansion_matrix()


@dataclass(frozen=True)
class EndpointEvaluation:
    """Endpoint values returned by :class:`EndpointKinematics`."""

    xyz_m: np.ndarray
    jacobian_m: np.ndarray | None = None


class EndpointKinematics:
    """Batch float64 endpoint FK and analytic beta-space Jacobian.

    The implementation preserves the exact DH convention used by
    :func:`quasi_exp.model.kinematics.forward_kinematics`.  It executes only 30
    Python joint iterations for a whole batch; all row and beta-derivative
    operations inside a joint are NumPy batch operations.
    """

    def __init__(
        self,
        lengths_m: np.ndarray,
        p_end_local_m: np.ndarray,
        *,
        theta_sign: float = -1.0,
    ) -> None:
        lengths = np.asarray(lengths_m, dtype=np.float64).reshape(-1)
        end = np.asarray(p_end_local_m, dtype=np.float64).reshape(-1)
        if lengths.shape != (31,):
            raise ValueError(f"lengths_m must have shape (31,), got {lengths.shape}")
        if end.shape != (4,):
            raise ValueError(f"p_end_local_m must have shape (4,), got {end.shape}")
        if not np.isfinite(lengths).all() or not np.isfinite(end).all():
            raise ValueError("kinematics inputs must be finite")
        if not np.isfinite(theta_sign):
            raise ValueError("theta_sign must be finite")
        self._lengths_m = lengths.copy()
        self._p_end_local_m = end.copy()
        self._theta_sign = float(theta_sign)
        self._alpha_cos = np.ones(30, dtype=np.float64)
        self._alpha_sin = np.zeros(30, dtype=np.float64)
        for joint_index in range(30):
            joint_number = joint_index + 1
            if joint_number == 1:
                alpha = 0.0
            else:
                alpha = np.pi / 2.0 if joint_number % 2 == 1 else -np.pi / 2.0
            self._alpha_cos[joint_index] = np.cos(alpha)
            self._alpha_sin[joint_index] = np.sin(alpha)

    @staticmethod
    def _coerce_beta(beta_rad: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta_rad, dtype=np.float64)
        if beta.ndim == 1 and beta.shape == (6,):
            beta = beta.reshape(1, 6)
        if beta.ndim != 2 or beta.shape[1] != 6:
            raise ValueError(f"beta_rad must have shape (N, 6), got {beta.shape}")
        if not np.isfinite(beta).all():
            raise ValueError("beta_rad must contain only finite values")
        return beta

    def evaluate(self, beta_rad: np.ndarray, *, jacobian: bool = False) -> EndpointEvaluation:
        beta = self._coerce_beta(beta_rad)
        count = len(beta)
        theta = (beta @ _BETA_EXPANSION.T) * self._theta_sign
        transform = np.broadcast_to(np.eye(4, dtype=np.float64), (count, 4, 4)).copy()
        derivative = (
            np.zeros((count, 6, 4, 4), dtype=np.float64) if jacobian else None
        )

        for joint in range(30):
            angle = theta[:, joint]
            cosine = np.cos(angle)
            sine = np.sin(angle)
            ca = self._alpha_cos[joint]
            sa = self._alpha_sin[joint]
            local = np.zeros((count, 4, 4), dtype=np.float64)
            local[:, 0, 0] = cosine
            local[:, 0, 1] = -sine
            local[:, 0, 3] = self._lengths_m[joint]
            local[:, 1, 0] = sine * ca
            local[:, 1, 1] = cosine * ca
            local[:, 1, 2] = -sa
            local[:, 2, 0] = sine * sa
            local[:, 2, 1] = cosine * sa
            local[:, 2, 2] = ca
            local[:, 3, 3] = 1.0

            if derivative is not None:
                dlocal = np.zeros((count, 4, 4), dtype=np.float64)
                dlocal[:, 0, 0] = -sine
                dlocal[:, 0, 1] = -cosine
                dlocal[:, 1, 0] = cosine * ca
                dlocal[:, 1, 1] = -sine * ca
                dlocal[:, 2, 0] = cosine * sa
                dlocal[:, 2, 1] = -sine * sa
                propagated = np.einsum("nbij,njk->nbik", derivative, local, optimize=True)
                active_beta = 2 * (joint // 10) + (joint % 2)
                propagated[:, active_beta] += (
                    self._theta_sign
                    * np.einsum("nij,njk->nik", transform, dlocal, optimize=True)
                )
                derivative = propagated
            transform = np.einsum("nij,njk->nik", transform, local, optimize=True)

        endpoint = np.einsum("nij,j->ni", transform, self._p_end_local_m, optimize=True)
        if derivative is None:
            return EndpointEvaluation(xyz_m=endpoint[:, :3].copy())
        endpoint_derivative = np.einsum(
            "nbij,j->nbi", derivative, self._p_end_local_m, optimize=True
        )
        jacobian_m = np.transpose(endpoint_derivative[:, :, :3], (0, 2, 1)).copy()
        return EndpointEvaluation(xyz_m=endpoint[:, :3].copy(), jacobian_m=jacobian_m)

    def fk(self, beta_rad: np.ndarray) -> np.ndarray:
        return self.evaluate(beta_rad, jacobian=False).xyz_m

    def fk_and_jacobian(self, beta_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        evaluation = self.evaluate(beta_rad, jacobian=True)
        assert evaluation.jacobian_m is not None
        return evaluation.xyz_m, evaluation.jacobian_m


__all__ = ["EndpointEvaluation", "EndpointKinematics"]
