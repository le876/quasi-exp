"""Frozen protocol helpers for the V11 generalized ellipse region."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from .region import EllipseFamilySpec


ROLE_QUOTAS: Mapping[str, tuple[int, int]] = {
    "train": (14, 28),
    "validation": (5, 10),
    "virgin_test": (5, 10),
}


def _canonical_frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.sort_values("family_id", kind="stable").to_dict(orient="records")
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FamilyCatalog:
    """Immutable family parameters and pre-label role assignments."""

    frame: pd.DataFrame
    seed: int

    def __post_init__(self) -> None:
        frame = self.frame.copy().reset_index(drop=True)
        required = {
            "family_id",
            "role",
            "is_primary",
            "reserve_rank",
            "major_semiaxis_m",
            "axis_ratio",
            "center_x_m",
            "center_y_m",
            "center_z_m",
            "major_x",
            "major_y",
            "major_z",
            "minor_x",
            "minor_y",
            "minor_z",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"family catalog missing columns: {missing}")
        if len(frame) != 72 or not frame["family_id"].is_unique:
            raise ValueError("family catalog must contain 72 unique families")
        actual = frame.groupby(["role", "is_primary"]).size().to_dict()
        expected = {
            (role, True): primary
            for role, (primary, _reserve) in ROLE_QUOTAS.items()
        } | {
            (role, False): reserve
            for role, (_primary, reserve) in ROLE_QUOTAS.items()
        }
        if actual != expected:
            raise ValueError(f"family catalog role quotas differ: {actual}")
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "seed", int(self.seed))

    @property
    def fingerprint(self) -> str:
        return _canonical_frame_hash(self.frame)

    @property
    def seal_token(self) -> str:
        encoded = f"{self.fingerprint}:sealed_evaluation:v11".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def primary_ids(self, role: str) -> tuple[str, ...]:
        selected = self.frame[
            self.frame["role"].eq(str(role)) & self.frame["is_primary"].eq(True)
        ]
        return tuple(selected.sort_values("catalog_order")["family_id"].astype(str))

    def authorized_family_ids(
        self, purpose: str, *, unseal_token: str | None = None
    ) -> tuple[str, ...]:
        normalized = str(purpose)
        if normalized == "training":
            return self.primary_ids("train")
        if normalized == "model_selection":
            return self.primary_ids("train") + self.primary_ids("validation")
        if normalized == "sealed_evaluation":
            if unseal_token != self.seal_token:
                raise PermissionError("virgin-test families are sealed until model lock")
            return self.primary_ids("virgin_test")
        raise ValueError(f"unknown family access purpose: {purpose}")

    def family(self, family_id: str) -> EllipseFamilySpec:
        rows = self.frame[self.frame["family_id"].astype(str).eq(str(family_id))]
        if len(rows) != 1:
            raise KeyError(f"unknown family id: {family_id}")
        row = rows.iloc[0]
        return EllipseFamilySpec(
            family_id=str(row["family_id"]),
            center_m=row[["center_x_m", "center_y_m", "center_z_m"]].to_numpy(dtype=float),
            major_direction=row[["major_x", "major_y", "major_z"]].to_numpy(dtype=float),
            minor_direction=row[["minor_x", "minor_y", "minor_z"]].to_numpy(dtype=float),
            major_semiaxis_m=float(row["major_semiaxis_m"]),
            minor_semiaxis_m=float(row["major_semiaxis_m"] * row["axis_ratio"]),
            metadata={
                "role": str(row["role"]),
                "is_primary": bool(row["is_primary"]),
                "catalog_fingerprint": self.fingerprint,
            },
        )

    def reanchor(self, anchor: EllipseFamilySpec) -> "FamilyCatalog":
        """Apply the frozen relative offsets/tilts to a selected anchor pose."""

        required = {
            "center_q1_offset_mm",
            "center_q2_offset_mm",
            "center_normal_offset_mm",
            "tilt_q1_deg",
            "tilt_q2_deg",
            "tilt_normal_deg",
        }
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise ValueError(f"family catalog lacks relative design fields: {missing}")
        frame = self.frame.copy()
        axes = np.column_stack(
            [anchor.major_direction, anchor.minor_direction, anchor.plane_normal]
        )
        for index, row in frame.iterrows():
            center = (
                anchor.center_m
                + float(row["center_q1_offset_mm"]) / 1000.0 * anchor.major_direction
                + float(row["center_q2_offset_mm"]) / 1000.0 * anchor.minor_direction
                + float(row["center_normal_offset_mm"]) / 1000.0 * anchor.plane_normal
            )
            local_angles = np.asarray(
                [
                    row["tilt_q1_deg"],
                    row["tilt_q2_deg"],
                    row["tilt_normal_deg"],
                ],
                dtype=float,
            )
            rotation = Rotation.from_rotvec(axes @ np.deg2rad(local_angles))
            major = rotation.apply(anchor.major_direction)
            minor = rotation.apply(anchor.minor_direction)
            frame.loc[index, ["center_x_m", "center_y_m", "center_z_m"]] = center
            frame.loc[index, ["major_x", "major_y", "major_z"]] = major
            frame.loc[index, ["minor_x", "minor_y", "minor_z"]] = minor
        return FamilyCatalog(frame, seed=self.seed)


def _scaled(unit: np.ndarray, low: float, high: float) -> np.ndarray:
    return float(low) + np.asarray(unit, dtype=float) * (float(high) - float(low))


def generate_family_catalog(
    baseline: EllipseFamilySpec, *, seed: int = 20260721
) -> FamilyCatalog:
    """Generate and seal the 24 primary plus 48 reserve family catalog."""

    sample = qmc.LatinHypercube(d=8, seed=int(seed)).random(72)
    major_radius = _scaled(sample[:, 0], 0.46, 0.50)
    ratio = _scaled(sample[:, 1], 0.31, 0.36)
    center_q1 = _scaled(sample[:, 2], -0.010, 0.010)
    center_q2 = _scaled(sample[:, 3], -0.010, 0.010)
    center_normal = _scaled(sample[:, 4], -0.005, 0.005)
    tilt_deg = _scaled(sample[:, 5:8], -3.0, 3.0)
    axes = np.column_stack(
        [baseline.major_direction, baseline.minor_direction, baseline.plane_normal]
    )

    role_slots: list[tuple[str, bool, int]] = []
    for role, (primary_count, _reserve_count) in ROLE_QUOTAS.items():
        role_slots.extend((role, True, -1) for _ in range(primary_count))
    for role, (_primary_count, reserve_count) in ROLE_QUOTAS.items():
        role_slots.extend((role, False, rank) for rank in range(reserve_count))
    if len(role_slots) != len(sample):
        raise AssertionError("family role slots do not cover the catalog")

    rows: list[dict[str, Any]] = []
    for index, (role, is_primary, reserve_rank) in enumerate(role_slots):
        rotation_vector = axes @ np.deg2rad(tilt_deg[index])
        rotation = Rotation.from_rotvec(rotation_vector)
        major = rotation.apply(baseline.major_direction)
        minor = rotation.apply(baseline.minor_direction)
        center = (
            baseline.center_m
            + center_q1[index] * baseline.major_direction
            + center_q2[index] * baseline.minor_direction
            + center_normal[index] * baseline.plane_normal
        )
        rows.append(
            {
                "family_id": f"F{index:03d}",
                "catalog_order": index,
                "role": role,
                "is_primary": bool(is_primary),
                "reserve_rank": int(reserve_rank),
                "major_semiaxis_m": float(major_radius[index]),
                "axis_ratio": float(ratio[index]),
                "center_x_m": float(center[0]),
                "center_y_m": float(center[1]),
                "center_z_m": float(center[2]),
                "major_x": float(major[0]),
                "major_y": float(major[1]),
                "major_z": float(major[2]),
                "minor_x": float(minor[0]),
                "minor_y": float(minor[1]),
                "minor_z": float(minor[2]),
                "center_q1_offset_mm": float(center_q1[index] * 1000.0),
                "center_q2_offset_mm": float(center_q2[index] * 1000.0),
                "center_normal_offset_mm": float(center_normal[index] * 1000.0),
                "tilt_q1_deg": float(tilt_deg[index, 0]),
                "tilt_q2_deg": float(tilt_deg[index, 1]),
                "tilt_normal_deg": float(tilt_deg[index, 2]),
            }
        )
    return FamilyCatalog(pd.DataFrame(rows), seed=int(seed))


def _strict_bool(value: Any, *, path: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{path} must be bool, not {type(value).__name__}")
    return bool(value)


def require_boolean_gate_tree(report: Mapping[str, Any]) -> bool:
    """Require a top-level bool gate and the raw bool checks behind it."""

    if "gate_pass" not in report:
        raise ValueError("gate report is missing gate_pass")
    gate_pass = _strict_bool(report["gate_pass"], path="gate_pass")
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not checks:
        raise ValueError("gate report must include non-empty raw checks")
    raw = [_strict_bool(value, path=f"checks.{key}") for key, value in checks.items()]
    if gate_pass != all(raw):
        raise ValueError("gate_pass does not equal the conjunction of raw checks")
    return gate_pass


def promote_complete_families(
    catalog: FamilyCatalog, quality: pd.DataFrame
) -> pd.DataFrame:
    """Fill every role quota using only whole-family pass/fail outcomes."""

    required = {"family_id", "complete_gate_pass"}
    missing = sorted(required - set(quality.columns))
    if missing:
        raise ValueError(f"family quality table missing columns: {missing}")
    if quality["family_id"].duplicated().any():
        raise ValueError("family quality table must have exactly one row per family")
    aligned = catalog.frame.merge(
        quality[["family_id", "complete_gate_pass"]],
        on="family_id",
        how="left",
        validate="one_to_one",
    )
    if aligned["complete_gate_pass"].isna().any():
        raise ValueError("family quality table does not cover the frozen catalog")
    for value in aligned["complete_gate_pass"]:
        _strict_bool(value, path="complete_gate_pass")

    selected: list[pd.DataFrame] = []
    for role, (quota, _reserve_count) in ROLE_QUOTAS.items():
        candidates = aligned[
            aligned["role"].eq(role) & aligned["complete_gate_pass"].eq(True)
        ].copy()
        candidates["selection_priority"] = np.where(
            candidates["is_primary"], candidates["catalog_order"], 10_000 + candidates["reserve_rank"]
        )
        candidates = candidates.sort_values("selection_priority", kind="stable")
        if len(candidates) < quota:
            raise RuntimeError(
                f"insufficient complete families for {role}: {len(candidates)} < {quota}"
            )
        selected.append(candidates.iloc[:quota])
    output = pd.concat(selected, ignore_index=True)
    output["selected_for_formal"] = True
    return output.drop(columns=["selection_priority", "complete_gate_pass"])


def nested_family_sample_indices(
    frame: pd.DataFrame,
    *,
    sizes: Sequence[int],
    seed: int,
) -> dict[int, np.ndarray]:
    """Create one boundary-prioritised master order and return nested prefixes."""

    required = {
        "family_id",
        "normalized_radial_offset",
        "normalized_plane_offset",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"training frame missing columns: {missing}")
    requested = tuple(int(value) for value in sizes)
    if not requested or any(value <= 0 for value in requested):
        raise ValueError("nested sample sizes must be positive")
    if tuple(sorted(set(requested))) != requested:
        raise ValueError("nested sample sizes must be unique and increasing")
    if requested[-1] > len(frame):
        raise ValueError("largest nested sample exceeds the available rows")
    families = tuple(sorted(frame["family_id"].astype(str).unique()))
    if requested[0] < len(families):
        raise ValueError("smallest subset cannot include every family")

    rng = np.random.default_rng(int(seed))
    queues: dict[str, list[int]] = {}
    for family_id in families:
        group = frame.index[frame["family_id"].astype(str).eq(family_id)].to_numpy(dtype=int)
        radius = np.sqrt(
            np.square(frame.loc[group, "normalized_radial_offset"].to_numpy(dtype=float))
            + np.square(frame.loc[group, "normalized_plane_offset"].to_numpy(dtype=float))
        )
        jitter = rng.random(len(group))
        order = np.lexsort((jitter, -radius))
        queues[family_id] = group[order].tolist()

    family_order = list(families)
    rng.shuffle(family_order)
    master: list[int] = []
    cursor = 0
    while len(master) < len(frame):
        family_id = family_order[cursor % len(family_order)]
        cursor += 1
        if queues[family_id]:
            master.append(queues[family_id].pop(0))
    values = np.asarray(master, dtype=int)
    return {size: values[:size].copy() for size in requested}
