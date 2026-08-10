"""Trajectory-level canonical teacher pipeline (V10)."""

from .canonical import (
    CanonicalTeacher,
    TeacherPolicy,
    TeacherTrajectory,
    TeacherVariant,
    TrajectorySpec,
)
from .forward import ForwardEnvironment, ForwardValidationResult
from .large_scale import EllipseChallenge
from .region import (
    CanonicalRegionTeacher,
    EllipseFamilySpec,
    TeacherSurface,
    TubeCrossSection,
)

__all__ = [
    "CanonicalTeacher",
    "ForwardEnvironment",
    "ForwardValidationResult",
    "EllipseChallenge",
    "TeacherPolicy",
    "TeacherTrajectory",
    "TeacherVariant",
    "TrajectorySpec",
    "CanonicalRegionTeacher",
    "EllipseFamilySpec",
    "TeacherSurface",
    "TubeCrossSection",
]
