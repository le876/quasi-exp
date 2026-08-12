"""V14.3-only optimized forward seam.

Historical formal runners continue to use :class:`ForwardEnvironment` and its
bit-exact scalar FK.  New exploratory runners opt into this adapter explicitly,
so sealed V12/V14 artifacts are not silently reinterpreted by a common-module
change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quasi_exp.model.endpoint_kinematics import EndpointEvaluation, EndpointKinematics
from quasi_exp.teacher.forward import ForwardEnvironment


@dataclass(frozen=True)
class OptimizedForwardEnvironment:
    reference: ForwardEnvironment
    _endpoint: EndpointKinematics = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_endpoint",
            EndpointKinematics(
                self.reference.lengths_m,
                self.reference.p_end_local_m,
                theta_sign=float(self.reference.theta_sign),
            ),
        )

    def __getattr__(self, name: str):
        return getattr(self.reference, name)

    @property
    def bounds(self) -> np.ndarray:
        return self.reference.bounds

    @property
    def lengths_m(self) -> np.ndarray:
        return self.reference.lengths_m

    @property
    def p_end_local_m(self) -> np.ndarray:
        return self.reference.p_end_local_m

    @property
    def theta_sign(self) -> float:
        return self.reference.theta_sign

    def evaluate(self, beta_rad: np.ndarray, *, jacobian: bool = False) -> EndpointEvaluation:
        return self._endpoint.evaluate(beta_rad, jacobian=jacobian)

    def fk(self, beta_rad: np.ndarray) -> np.ndarray:
        return self._endpoint.fk(beta_rad)

    def fk_and_jacobian(self, beta_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._endpoint.fk_and_jacobian(beta_rad)

    def jacobian(self, beta_rad: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta_rad, dtype=float)
        if beta.shape != (6,):
            raise ValueError(f"beta_rad must have shape (6,), got {beta.shape}")
        return self.fk_and_jacobian(beta)[1][0]


def optimized_forward(reference: ForwardEnvironment) -> OptimizedForwardEnvironment:
    return OptimizedForwardEnvironment(reference)


__all__ = ["OptimizedForwardEnvironment", "optimized_forward"]
