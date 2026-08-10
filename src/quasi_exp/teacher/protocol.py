"""Pure protocol helpers for trajectory-level teacher experiments."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class RadiusFrontierResult:
    """A radius frontier backed by the exact evaluations used to locate it."""

    highest_pass_mm: float | None
    first_fail_mm: float | None
    resolution_mm: float
    evaluations: tuple[tuple[float, bool], ...]
    non_monotonic: bool
    post_failure_passes_mm: tuple[float, ...]

    @property
    def evaluated_radii_mm(self) -> tuple[float, ...]:
        return tuple(radius_mm for radius_mm, _ in self.evaluations)


@dataclass(frozen=True)
class FamilySplitSeal:
    """Immutable family-role assignment and its canonical fingerprint."""

    assignments: tuple[tuple[str, str], ...]
    protocol_sha256: str

    def _ids_for_role(self, role: str) -> tuple[str, ...]:
        return tuple(family_id for family_id, assigned_role in self.assignments if assigned_role == role)

    @property
    def training_family_ids(self) -> tuple[str, ...]:
        return self._ids_for_role("train")

    @property
    def model_selection_family_ids(self) -> tuple[str, ...]:
        return self._ids_for_role("validation")

    @property
    def virgin_family_ids(self) -> tuple[str, ...]:
        return self._ids_for_role("virgin_test")

    def role_for(self, family_id: str) -> str:
        for assigned_family_id, role in self.assignments:
            if assigned_family_id == family_id:
                return role
        raise KeyError(f"family is not present in split seal: {family_id}")


def locate_radius_frontier(
    evaluator: Callable[[float], bool],
    *,
    coarse_anchors_mm: Sequence[float],
    resolution_mm: float = 0.5,
) -> RadiusFrontierResult:
    """Locate the first pass-to-fail radius transition on a fixed grid.

    Coarse anchors are evaluated first.  The bracket ending at the first
    failed anchor is then filled at ``resolution_mm``.  Results after the
    first failure are retained as evidence, but never extend the claimable
    frontier.
    """

    if resolution_mm <= 0.0:
        raise ValueError("resolution_mm must be positive")
    anchors = tuple(sorted({float(radius_mm) for radius_mm in coarse_anchors_mm}))
    if not anchors:
        raise ValueError("coarse_anchors_mm must not be empty")

    evaluated: dict[float, bool] = {}

    def evaluate(radius_mm: float) -> bool:
        radius_mm = round(float(radius_mm), 10)
        if radius_mm not in evaluated:
            outcome = evaluator(radius_mm)
            if not isinstance(outcome, bool):
                raise TypeError("evaluator must return bool")
            evaluated[radius_mm] = outcome
        return evaluated[radius_mm]

    for anchor_mm in anchors:
        evaluate(anchor_mm)

    seen_failed_anchor = False
    anchor_non_monotonic = False
    for anchor_mm in anchors:
        if not evaluated[anchor_mm]:
            seen_failed_anchor = True
        elif seen_failed_anchor:
            anchor_non_monotonic = True

    if anchor_non_monotonic:
        radius_mm = anchors[0]
        while radius_mm <= anchors[-1] + resolution_mm * 1e-9:
            evaluate(radius_mm)
            radius_mm += resolution_mm

    failed_anchors = [radius_mm for radius_mm in anchors if not evaluated[radius_mm]]
    if failed_anchors and not anchor_non_monotonic:
        first_failed_anchor_mm = failed_anchors[0]
        lower_anchors = [radius_mm for radius_mm in anchors if radius_mm < first_failed_anchor_mm]
        if lower_anchors:
            bracket_start_mm = lower_anchors[-1]
            step = 1
            while bracket_start_mm + step * resolution_mm < first_failed_anchor_mm:
                evaluate(bracket_start_mm + step * resolution_mm)
                step += 1

    ordered = tuple(sorted(evaluated.items()))
    first_fail_mm = next((radius_mm for radius_mm, passed in ordered if not passed), None)
    if first_fail_mm is None:
        highest_pass_mm = max(radius_mm for radius_mm, passed in ordered if passed)
        post_failure_passes_mm: tuple[float, ...] = ()
    else:
        pre_failure_passes = [
            radius_mm for radius_mm, passed in ordered if passed and radius_mm < first_fail_mm
        ]
        highest_pass_mm = max(pre_failure_passes) if pre_failure_passes else None
        post_failure_passes_mm = tuple(
            radius_mm for radius_mm, passed in ordered if passed and radius_mm > first_fail_mm
        )

    return RadiusFrontierResult(
        highest_pass_mm=highest_pass_mm,
        first_fail_mm=first_fail_mm,
        resolution_mm=float(resolution_mm),
        evaluations=ordered,
        non_monotonic=bool(post_failure_passes_mm),
        post_failure_passes_mm=post_failure_passes_mm,
    )


def select_maximin_families(
    family_features: Mapping[str, Sequence[float]],
    *,
    count: int,
    seed_family_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    """Greedily select families farthest from the current selected set.

    Features are standardised across the supplied candidate pool.  At each
    step the candidate with the largest minimum Euclidean distance to any
    already-selected family is chosen; family id is the deterministic
    tie-breaker.  Only newly selected ids are returned.
    """

    if count < 0:
        raise ValueError("count must be non-negative")
    if not family_features:
        if count:
            raise ValueError("family_features must not be empty")
        return ()

    ids = tuple(sorted(str(family_id) for family_id in family_features))
    if len(ids) != len(family_features):
        raise ValueError("family ids must remain unique when converted to strings")
    vectors = {family_id: tuple(float(v) for v in family_features[family_id]) for family_id in ids}
    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) == 0:
        raise ValueError("all family feature vectors must have one shared non-zero dimension")
    if any(not math.isfinite(value) for vector in vectors.values() for value in vector):
        raise ValueError("family features must be finite")

    seed_ids = tuple(str(family_id) for family_id in seed_family_ids)
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("seed_family_ids must be unique")
    unknown_seed_ids = sorted(set(seed_ids) - set(ids))
    if unknown_seed_ids:
        raise ValueError(f"unknown seed family ids: {unknown_seed_ids}")

    available = set(ids) - set(seed_ids)
    if count > len(available):
        raise ValueError("count exceeds the number of unselected families")

    dimension_count = next(iter(dimensions))
    means = tuple(
        sum(vectors[family_id][index] for family_id in ids) / len(ids)
        for index in range(dimension_count)
    )
    scales = tuple(
        math.sqrt(
            sum((vectors[family_id][index] - means[index]) ** 2 for family_id in ids)
            / len(ids)
        )
        for index in range(dimension_count)
    )
    standardised = {
        family_id: tuple(
            0.0 if scales[index] == 0.0 else (value - means[index]) / scales[index]
            for index, value in enumerate(vectors[family_id])
        )
        for family_id in ids
    }

    selected = list(seed_ids)
    additions: list[str] = []
    while len(additions) < count:
        if not selected:
            chosen_id = min(available)
        else:
            scores: dict[str, float] = {}
            for candidate_id in available:
                scores[candidate_id] = min(
                    math.dist(standardised[candidate_id], standardised[selected_id])
                    for selected_id in selected
                )
            best_score = max(scores.values())
            chosen_id = min(
                family_id
                for family_id, score in scores.items()
                if math.isclose(score, best_score, rel_tol=1e-12, abs_tol=1e-12)
            )
        selected.append(chosen_id)
        additions.append(chosen_id)
        available.remove(chosen_id)

    return tuple(additions)


def adaptive_family_radii(
    frontier_mm: float,
    *,
    base_radius_mm: float = 75.0,
    minimum_frontier_mm: float = 92.5,
    resolution_mm: float = 0.5,
) -> tuple[float, float, float, float, float]:
    """Return the five formal radii spanning base to a family's frontier.

    Intermediate quartile points are rounded downward to the protocol grid;
    the endpoints remain the exact base and frontier values.
    """

    frontier = Decimal(str(frontier_mm))
    base = Decimal(str(base_radius_mm))
    minimum = Decimal(str(minimum_frontier_mm))
    resolution = Decimal(str(resolution_mm))
    if resolution <= 0:
        raise ValueError("resolution_mm must be positive")
    if frontier < minimum:
        raise ValueError(f"frontier_mm must be at least {float(minimum):g}")
    if base >= frontier:
        raise ValueError("base_radius_mm must be smaller than frontier_mm")
    if frontier % resolution != 0 or base % resolution != 0:
        raise ValueError("base and frontier radii must lie on the resolution grid")

    span = frontier - base
    radii = []
    for index in range(5):
        unrounded = base + Decimal(index) * span / Decimal(4)
        grid_units = (unrounded / resolution).to_integral_value(rounding=ROUND_FLOOR)
        radii.append(float(grid_units * resolution))
    radii[0] = float(base)
    radii[-1] = float(frontier)
    if len(set(radii)) != 5:
        raise ValueError("frontier does not yield five distinct grid radii")
    return tuple(radii)  # type: ignore[return-value]


def _family_split_digest(assignments: tuple[tuple[str, str], ...]) -> str:
    canonical = json.dumps(
        {
            "protocol": "trajectory-canonical-teacher-v10",
            "assignments": assignments,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def seal_family_splits(
    *,
    train_family_ids: Sequence[str],
    validation_family_ids: Sequence[str],
    virgin_test_family_ids: Sequence[str],
) -> FamilySplitSeal:
    """Create a deterministic, mutually-exclusive family split seal."""

    role_groups = {
        "train": tuple(str(family_id) for family_id in train_family_ids),
        "validation": tuple(str(family_id) for family_id in validation_family_ids),
        "virgin_test": tuple(str(family_id) for family_id in virgin_test_family_ids),
    }
    for role, family_ids in role_groups.items():
        if not family_ids:
            raise ValueError(f"{role} family ids must not be empty")
        if any(not family_id.strip() for family_id in family_ids):
            raise ValueError(f"{role} family ids must be non-empty strings")
        if len(family_ids) != len(set(family_ids)):
            raise ValueError(f"{role} family ids must be unique")

    all_ids = [family_id for family_ids in role_groups.values() for family_id in family_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("family ids must be disjoint across train, validation, and virgin_test")

    assignments = tuple(
        sorted(
            (family_id, role)
            for role, family_ids in role_groups.items()
            for family_id in family_ids
        )
    )
    return FamilySplitSeal(
        assignments=assignments,
        protocol_sha256=_family_split_digest(assignments),
    )


def verify_family_split_seal(seal: FamilySplitSeal) -> bool:
    """Return whether the stored fingerprint matches the role assignments."""

    return seal.protocol_sha256 == _family_split_digest(seal.assignments)
