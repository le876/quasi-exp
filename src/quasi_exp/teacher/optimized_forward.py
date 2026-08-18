"""V14.3-only optimized forward seam.

Historical formal runners continue to use :class:`ForwardEnvironment` and its
bit-exact scalar FK.  New exploratory runners opt into this adapter explicitly,
so sealed V12/V14 artifacts are not silently reinterpreted by a common-module
change.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from quasi_exp.model.endpoint_kinematics import EndpointEvaluation, EndpointKinematics
from quasi_exp.teacher.forward import ForwardEnvironment


@dataclass(frozen=True)
class OptimizedForwardEnvironment:
    reference: ForwardEnvironment
    _endpoint: EndpointKinematics = field(init=False, repr=False, compare=False)
    _fk_row_count: int = field(init=False, repr=False, compare=False, default=0)
    _jacobian_row_count: int = field(init=False, repr=False, compare=False, default=0)
    _cache_hit_count: int = field(init=False, repr=False, compare=False, default=0)
    _cache_miss_count: int = field(init=False, repr=False, compare=False, default=0)
    _single_row_cache: OrderedDict[bytes, tuple[np.ndarray, np.ndarray | None]] = field(
        init=False, repr=False, compare=False, default_factory=OrderedDict
    )
    cache_capacity: int = 4096

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
        beta = np.asarray(beta_rad, dtype=float)
        object.__setattr__(
            self,
            "_fk_row_count",
            self._fk_row_count + (1 if beta.ndim == 1 else int(len(beta))),
        )
        if beta.shape != (6,) or self.cache_capacity <= 0:
            return self._endpoint.fk(beta_rad)
        key = beta.tobytes(order="C")
        cached = self._single_row_cache.get(key)
        if cached is not None:
            self._single_row_cache.move_to_end(key)
            object.__setattr__(self, "_cache_hit_count", self._cache_hit_count + 1)
            return cached[0].reshape(1, 3).copy()
        xyz = self._endpoint.fk(beta)
        object.__setattr__(self, "_cache_miss_count", self._cache_miss_count + 1)
        self._single_row_cache[key] = (xyz[0].copy(), None)
        self._trim_cache()
        return xyz

    def fk_and_jacobian(self, beta_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        beta = np.asarray(beta_rad, dtype=float)
        rows = 1 if beta.ndim == 1 else int(len(beta))
        object.__setattr__(self, "_fk_row_count", self._fk_row_count + rows)
        object.__setattr__(
            self, "_jacobian_row_count", self._jacobian_row_count + rows
        )
        if beta.shape != (6,) or self.cache_capacity <= 0:
            return self._endpoint.fk_and_jacobian(beta_rad)
        key = beta.tobytes(order="C")
        cached = self._single_row_cache.get(key)
        if cached is not None and cached[1] is not None:
            self._single_row_cache.move_to_end(key)
            object.__setattr__(self, "_cache_hit_count", self._cache_hit_count + 1)
            return cached[0].reshape(1, 3).copy(), cached[1].reshape(1, 3, 6).copy()
        xyz, jacobian = self._endpoint.fk_and_jacobian(beta)
        object.__setattr__(self, "_cache_miss_count", self._cache_miss_count + 1)
        self._single_row_cache[key] = (xyz[0].copy(), jacobian[0].copy())
        self._trim_cache()
        return xyz, jacobian

    def _trim_cache(self) -> None:
        while len(self._single_row_cache) > int(self.cache_capacity):
            self._single_row_cache.popitem(last=False)

    def performance_counters(self) -> dict[str, int]:
        return {
            "fk_row_count": int(self._fk_row_count),
            "jacobian_row_count": int(self._jacobian_row_count),
            "kinematics_cache_hit_count": int(self._cache_hit_count),
            "kinematics_cache_miss_count": int(self._cache_miss_count),
            "kinematics_cache_entry_count": len(self._single_row_cache),
        }

    def jacobian(self, beta_rad: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta_rad, dtype=float)
        if beta.shape != (6,):
            raise ValueError(f"beta_rad must have shape (6,), got {beta.shape}")
        return self.fk_and_jacobian(beta)[1][0]


def optimized_forward(reference: ForwardEnvironment) -> OptimizedForwardEnvironment:
    return OptimizedForwardEnvironment(reference)


__all__ = ["OptimizedForwardEnvironment", "optimized_forward"]
