#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc


TARGET_XYZ_COLS = ["x_target_m", "y_target_m", "z_target_m"]


@dataclass(frozen=True)
class FamilyParameters:
    family_id: str
    center_x_m: float
    center_y_m: float
    center_z_m: float
    phase_y_rad: float
    phase_z_rad: float


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    if table.empty:
        return "_empty_"

    def cell(value: Any) -> str:
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            return ""
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    headers = [cell(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _column in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def parse_float_csv(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    return [float(item) for item in value]


def family_from_mapping(row: Mapping[str, Any]) -> FamilyParameters:
    return FamilyParameters(
        family_id=str(row["family_id"]),
        center_x_m=float(row["center_x_m"]),
        center_y_m=float(row["center_y_m"]),
        center_z_m=float(row["center_z_m"]),
        phase_y_rad=float(row["phase_y_rad"]),
        phase_z_rad=float(row["phase_z_rad"]),
    )


def generate_family_targets(
    family: FamilyParameters | Mapping[str, Any],
    *,
    radius_mm: float,
    n_points: int,
) -> pd.DataFrame:
    params = family if isinstance(family, FamilyParameters) else family_from_mapping(family)
    angle = np.linspace(0.0, 2.0 * math.pi, int(n_points), endpoint=False)
    amp_xy_m = float(radius_mm) / 1000.0
    amp_z_m = 1.5 * amp_xy_m
    frame = pd.DataFrame(
        {
            "family_id": params.family_id,
            "candidate_id": params.family_id,
            "ellipse_id": params.family_id,
            "angle_idx": np.arange(int(n_points), dtype=np.int64),
            "angle_rad": angle,
            "center_x_m": params.center_x_m,
            "center_y_m": params.center_y_m,
            "center_z_m": params.center_z_m,
            "amp_xy_mm": float(radius_mm),
            "amp_z_mm": 1.5 * float(radius_mm),
            "phase_y_rad": params.phase_y_rad,
            "phase_z_rad": params.phase_z_rad,
        }
    )
    frame["x_target_m"] = params.center_x_m + amp_xy_m * np.sin(angle)
    frame["y_target_m"] = params.center_y_m + amp_xy_m * np.sin(angle + params.phase_y_rad)
    frame["z_target_m"] = params.center_z_m + amp_z_m * np.sin(angle + params.phase_z_rad)
    return frame


def _sobol_points(*, dimensions: int, count: int, seed: int) -> np.ndarray:
    if int(count) <= 0:
        raise ValueError("Sobol count must be positive")
    sampler = qmc.Sobol(d=int(dimensions), scramble=True, seed=int(seed))
    exponent = math.log2(int(count))
    if exponent.is_integer():
        return sampler.random_base2(m=int(exponent))
    return sampler.random(n=int(count))


def sobol_family_perturbations(
    seeds: pd.DataFrame,
    *,
    samples_per_seed: int,
    seed: int,
    center_x_offset_mm: tuple[float, float] = (-20.0, 0.0),
    center_yz_offset_mm: tuple[float, float] = (-15.0, 15.0),
    phase_offset_deg: tuple[float, float] = (-15.0, 15.0),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed_rank, (_idx, base) in enumerate(seeds.reset_index(drop=True).iterrows()):
        points = _sobol_points(dimensions=5, count=int(samples_per_seed), seed=int(seed) + seed_rank)
        low = np.asarray(
            [
                center_x_offset_mm[0],
                center_yz_offset_mm[0],
                center_yz_offset_mm[0],
                phase_offset_deg[0],
                phase_offset_deg[0],
            ],
            dtype=float,
        )
        high = np.asarray(
            [
                center_x_offset_mm[1],
                center_yz_offset_mm[1],
                center_yz_offset_mm[1],
                phase_offset_deg[1],
                phase_offset_deg[1],
            ],
            dtype=float,
        )
        offsets = low[None, :] + (high - low)[None, :] * points
        base_id = str(base["family_id"])
        for sample_rank, offset in enumerate(offsets):
            family_id = f"{base_id}_s{sample_rank:04d}"
            rows.append(
                {
                    "candidate_id": family_id,
                    "family_id": family_id,
                    "source_family_id": base_id,
                    "center_x_m": float(base["center_x_m"]) + float(offset[0]) / 1000.0,
                    "center_y_m": float(base["center_y_m"]) + float(offset[1]) / 1000.0,
                    "center_z_m": float(base["center_z_m"]) + float(offset[2]) / 1000.0,
                    "phase_y_rad": float(base["phase_y_rad"]) + math.radians(float(offset[3])),
                    "phase_z_rad": float(base["phase_z_rad"]) + math.radians(float(offset[4])),
                    "delta_center_x_mm": float(offset[0]),
                    "delta_center_y_mm": float(offset[1]),
                    "delta_center_z_mm": float(offset[2]),
                    "delta_phase_y_deg": float(offset[3]),
                    "delta_phase_z_deg": float(offset[4]),
                }
            )
    return pd.DataFrame(rows)


def strict_support_gate(metrics: Mapping[str, Any]) -> bool:
    return bool(
        float(metrics.get("nn_p95_mm", np.inf)) <= 5.0
        and float(metrics.get("nn_max_mm", np.inf)) <= 8.0
        and float(metrics.get("tube_count_p10", -np.inf)) >= 32.0
    )


def score_family_candidates(
    candidates: pd.DataFrame,
    *,
    pool_xyz: np.ndarray,
    radii_mm: Iterable[float],
    n_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = np.asarray(pool_xyz, dtype=float).reshape(-1, 3)
    if len(pool) == 0:
        raise ValueError("family search requires a non-empty support pool")
    radii = parse_float_csv(radii_mm)
    if not radii:
        raise ValueError("family search requires at least one radius")
    tree = cKDTree(pool)
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _idx, candidate in candidates.iterrows():
        params = family_from_mapping(candidate)
        metrics: list[dict[str, float]] = []
        for radius_mm in radii:
            targets = generate_family_targets(params, radius_mm=float(radius_mm), n_points=int(n_points))
            nearest = np.asarray(tree.query(targets[TARGET_XYZ_COLS].to_numpy(dtype=float), k=1)[0], dtype=float) * 1000.0
            row = {
                "candidate_id": str(candidate["candidate_id"]),
                "family_id": params.family_id,
                "radius_mm": float(radius_mm),
                "nn_mean_mm": float(np.mean(nearest)),
                "nn_p95_mm": float(np.percentile(nearest, 95)),
                "nn_max_mm": float(np.max(nearest)),
            }
            detail_rows.append(row)
            metrics.append(row)
        worst_p95 = float(max(row["nn_p95_mm"] for row in metrics))
        worst_max = float(max(row["nn_max_mm"] for row in metrics))
        summary = {
            "candidate_id": str(candidate["candidate_id"]),
            "family_id": params.family_id,
            "source_family_id": str(candidate.get("source_family_id", params.family_id)),
            "center_x_m": params.center_x_m,
            "center_y_m": params.center_y_m,
            "center_z_m": params.center_z_m,
            "phase_y_rad": params.phase_y_rad,
            "phase_z_rad": params.phase_z_rad,
            "worst_nn_p95_mm": worst_p95,
            "worst_nn_max_mm": worst_max,
            "search_score": worst_p95 + 0.25 * worst_max,
        }
        summary_rows.append(summary)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["search_score", "worst_nn_p95_mm", "worst_nn_max_mm", "candidate_id"]
    ).reset_index(drop=True)
    detail_df = pd.DataFrame(detail_rows).sort_values(["candidate_id", "radius_mm"]).reset_index(drop=True)
    return summary_df, detail_df


def connected_radius_max(
    status: pd.DataFrame,
    *,
    gate_col: str,
    anchor_mm: float,
) -> float | None:
    rows = status.sort_values("radius_mm").reset_index(drop=True)
    rows = rows[rows["radius_mm"].to_numpy(dtype=float) >= float(anchor_mm) - 1.0e-9]
    if rows.empty or not np.isclose(float(rows.iloc[0]["radius_mm"]), float(anchor_mm), atol=1.0e-8):
        return None
    maximum: float | None = None
    for _idx, row in rows.iterrows():
        if not bool(row[gate_col]):
            break
        maximum = float(row["radius_mm"])
    return maximum


def _circular_run_count(indices: Iterable[int], *, size: int) -> int:
    mask = np.zeros(int(size), dtype=bool)
    values = np.asarray(sorted(set(int(value) for value in indices)), dtype=int)
    if values.size == 0:
        return 0
    if np.any(values < 0) or np.any(values >= int(size)):
        raise ValueError("failed angle index outside target range")
    mask[values] = True
    if bool(mask.all()):
        return 1
    return int(np.count_nonzero(mask & ~np.roll(mask, 1)))


def should_retry_pointwise(
    *,
    target_count: int,
    failed_angle_indices: Iterable[int],
    residual_max_mm: float,
) -> bool:
    failed = sorted(set(int(value) for value in failed_angle_indices))
    if not failed:
        return False
    failure_ratio = len(failed) / max(int(target_count), 1)
    return bool(
        failure_ratio <= 0.10
        and float(residual_max_mm) <= 5.0
        and _circular_run_count(failed, size=int(target_count)) <= 2
    )


def family_record(family: FamilyParameters) -> dict[str, Any]:
    return asdict(family)
