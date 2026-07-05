#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / ".mplconfig"))

from fk_dh_numpy import fk_dh_batch  # noqa: E402
from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402


BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
THETA_COLS = [f"theta_{i}_rad" for i in range(1, 31)]
XYZ_COLS = ["x_m", "y_m", "z_m"]
POLICY_VERSION = "hierarchical_beta_fk_v1"


@dataclass(frozen=True)
class HierarchicalGridSpec:
    beta12_deg: tuple[float, float, float] = (-5.0, 5.0, 2.5)
    beta34_deg: tuple[float, float, float] = (-5.0, 5.0, 1.25)
    beta56_deg: tuple[float, float, float] = (-15.0, 15.0, 0.5)

    def step_deg(self, joint: str) -> float:
        if joint == "joint1":
            return float(self.beta12_deg[2])
        if joint == "joint2":
            return float(self.beta34_deg[2])
        if joint == "joint3":
            return float(self.beta56_deg[2])
        raise ValueError(f"unknown joint group: {joint}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _parse_range_step(raw: str) -> tuple[float, float, float]:
    parts = [float(v.strip()) for v in str(raw).split(",") if v.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected lo,hi,step")
    lo, hi, step = parts
    if not (lo < hi and step > 0.0):
        raise argparse.ArgumentTypeError("range must satisfy lo < hi and step > 0")
    return float(lo), float(hi), float(step)


def _levels_from_range_deg(range_step: tuple[float, float, float]) -> np.ndarray:
    lo, hi, step = (float(v) for v in range_step)
    n = int(round((hi - lo) / step)) + 1
    vals = lo + np.arange(n, dtype=float) * step
    vals[-1] = hi
    if np.any(vals < lo - 1.0e-9) or np.any(vals > hi + 1.0e-9):
        raise ValueError(f"invalid range/step produced out-of-bounds levels: {range_step}")
    return vals


def grid_levels_deg(spec: HierarchicalGridSpec) -> dict[str, np.ndarray]:
    beta12 = _levels_from_range_deg(spec.beta12_deg)
    beta34 = _levels_from_range_deg(spec.beta34_deg)
    beta56 = _levels_from_range_deg(spec.beta56_deg)
    return {
        "beta1": beta12,
        "beta2": beta12,
        "beta3": beta34,
        "beta4": beta34,
        "beta5": beta56,
        "beta6": beta56,
    }


def expected_grid_rows(spec: HierarchicalGridSpec) -> int:
    levels = grid_levels_deg(spec)
    total = 1
    for axis in ("beta1", "beta2", "beta3", "beta4", "beta5", "beta6"):
        total *= int(len(levels[axis]))
    return int(total)


def generate_beta_chunk_rad(spec: HierarchicalGridSpec, *, start: int, stop: int) -> np.ndarray:
    if int(start) < 0 or int(stop) < int(start):
        raise ValueError("chunk bounds must satisfy 0 <= start <= stop")
    total = expected_grid_rows(spec)
    if int(stop) > total:
        raise ValueError(f"stop={stop} exceeds grid rows={total}")
    levels = grid_levels_deg(spec)
    deg_levels = [levels[f"beta{i}"] for i in range(1, 7)]
    sizes = [len(v) for v in deg_levels]

    idx = np.arange(int(start), int(stop), dtype=np.int64)
    work = idx.copy()
    indices: list[np.ndarray] = []
    for size in reversed(sizes):
        indices.append(work % int(size))
        work //= int(size)
    indices = list(reversed(indices))
    beta_deg = np.column_stack([deg_levels[i][indices[i]] for i in range(6)])
    return np.deg2rad(beta_deg.astype(float, copy=False))


def theta_from_beta_batch(beta_rad: np.ndarray, *, theta_sign: float) -> np.ndarray:
    beta = np.asarray(beta_rad, dtype=float).reshape(-1, 6)
    theta = np.empty((len(beta), 30), dtype=float)
    for section, (odd_col, even_col) in enumerate(((0, 1), (2, 3), (4, 5))):
        start = section * 10
        theta[:, start : start + 10 : 2] = beta[:, odd_col : odd_col + 1]
        theta[:, start + 1 : start + 10 : 2] = beta[:, even_col : even_col + 1]
    return theta * float(theta_sign)


def _priority_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    j1 = np.sqrt(np.square(df["beta1_rad"].to_numpy(float)) + np.square(df["beta2_rad"].to_numpy(float)))
    j2 = np.sqrt(np.square(df["beta3_rad"].to_numpy(float)) + np.square(df["beta4_rad"].to_numpy(float)))
    j3 = np.sqrt(np.square(df["beta5_rad"].to_numpy(float)) + np.square(df["beta6_rad"].to_numpy(float)))
    return j1, j2, j3


def add_priority_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    j1, j2, j3 = _priority_arrays(out)
    out["proximal_j1_norm_rad"] = j1
    out["proximal_j2_norm_rad"] = j2
    out["distal_j3_norm_rad"] = j3
    out["hierarchical_priority_score"] = j1 + 0.1 * j2 - 0.01 * j3
    out["source_component"] = POLICY_VERSION
    return out


def _format_voxel_ids(voxels: np.ndarray) -> np.ndarray:
    return np.asarray([f"{int(a)}:{int(b)}:{int(c)}" for a, b, c in voxels], dtype=object)


def add_voxel_columns(df: pd.DataFrame, *, voxel_mm: float) -> pd.DataFrame:
    out = df.copy()
    voxel_m = float(voxel_mm) / 1000.0
    if voxel_m <= 0.0:
        raise ValueError("voxel_mm must be > 0")
    voxels = np.floor(out[XYZ_COLS].to_numpy(dtype=float) / voxel_m).astype(np.int64)
    out["workspace_voxel_id"] = _format_voxel_ids(voxels)
    return out


def _x_bin_indices(x: np.ndarray, *, x_range: tuple[float, float], x_bin_mm: float) -> tuple[np.ndarray, np.ndarray]:
    width = float(x_bin_mm) / 1000.0
    if width <= 0.0:
        raise ValueError("x_bin_mm must be > 0")
    lo, hi = float(x_range[0]), float(x_range[1])
    n_bins = max(1, int(math.ceil((hi - lo) / width)))
    idx = np.floor((np.asarray(x, dtype=float) - lo) / width).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    return idx, np.arange(n_bins, dtype=int)


def workspace_metrics(
    xyz: np.ndarray,
    *,
    x_range: tuple[float, float] = (1.0, 1.2),
    x_bin_mm: float = 5.0,
    yz_cell_mm: float = 20.0,
    voxel_mm: float = 10.0,
    nn_sample: int = 20000,
    nn_fit_limit: int = 600000,
    seed: int = 20260705,
) -> dict[str, Any]:
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if len(xyz) == 0:
        return {
            "rows": 0,
            "x_bin_nonempty_ratio": 0.0,
            "x_bin_count_cv": float("inf"),
            "yz_cell_x_range_p95_mm": 0.0,
            "voxel_count_3d": 0,
            "voxel_entropy_ratio": 0.0,
            "nn_dist_p95_mm": float("inf"),
        }
    x = xyz[:, 0]
    x_idx, x_bins = _x_bin_indices(x, x_range=x_range, x_bin_mm=x_bin_mm)
    x_counts = np.bincount(x_idx, minlength=len(x_bins)).astype(float)
    mean_count = float(np.mean(x_counts))
    x_cv = float(np.std(x_counts) / mean_count) if mean_count > 0.0 else float("inf")
    x_nonempty = float(np.mean(x_counts > 0.0))

    yz_cell_m = float(yz_cell_mm) / 1000.0
    yz_cells = np.floor(xyz[:, 1:3] / yz_cell_m).astype(np.int64)
    cell_df = pd.DataFrame({"cell_y": yz_cells[:, 0], "cell_z": yz_cells[:, 1], "x": x})
    x_ranges = cell_df.groupby(["cell_y", "cell_z"], sort=False)["x"].agg(lambda s: float(s.max() - s.min()))
    x_range_p95_mm = float(np.percentile(x_ranges.to_numpy(dtype=float), 95) * 1000.0) if len(x_ranges) else 0.0

    voxel_m = float(voxel_mm) / 1000.0
    voxels = np.floor(xyz / voxel_m).astype(np.int64)
    _unique, counts = np.unique(voxels, axis=0, return_counts=True)
    probs = counts.astype(float) / max(float(np.sum(counts)), 1.0)
    entropy = -float(np.sum(probs * np.log(probs + 1.0e-12)))
    entropy_ratio = float(entropy / np.log(len(counts))) if len(counts) > 1 else 0.0

    nn_p95 = float("inf")
    if len(xyz) >= 2:
        try:
            from sklearn.neighbors import NearestNeighbors

            rng = np.random.default_rng(int(seed))
            if len(xyz) > int(nn_sample):
                pick = rng.choice(len(xyz), size=int(nn_sample), replace=False)
                nn_xyz = xyz[pick]
            else:
                nn_xyz = xyz
            fit_xyz = xyz if len(xyz) <= int(nn_fit_limit) else nn_xyz
            nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(fit_xyz)
            dist, _idx = nn.kneighbors(nn_xyz, return_distance=True)
            nn_p95 = float(np.percentile(dist[:, 1], 95) * 1000.0)
        except Exception:
            nn_p95 = float("inf")

    return {
        "rows": int(len(xyz)),
        "x_range_m": [float(np.min(x)), float(np.max(x))],
        "y_range_m": [float(np.min(xyz[:, 1])), float(np.max(xyz[:, 1]))],
        "z_range_m": [float(np.min(xyz[:, 2])), float(np.max(xyz[:, 2]))],
        "x_bin_mm": float(x_bin_mm),
        "x_bin_count_min": int(np.min(x_counts)) if len(x_counts) else 0,
        "x_bin_count_max": int(np.max(x_counts)) if len(x_counts) else 0,
        "x_bin_nonempty_ratio": x_nonempty,
        "x_bin_count_cv": x_cv,
        "yz_cell_mm": float(yz_cell_mm),
        "yz_cell_count": int(len(x_ranges)),
        "yz_cell_x_range_p95_mm": x_range_p95_mm,
        "voxel_mm": float(voxel_mm),
        "voxel_count_3d": int(len(counts)),
        "voxel_entropy_ratio": entropy_ratio,
        "nn_dist_p95_mm": nn_p95,
    }


def select_workspace_balanced_subset(
    df: pd.DataFrame,
    *,
    max_rows: int,
    x_range: tuple[float, float] = (1.0, 1.2),
    x_bin_mm: float = 5.0,
    voxel_mm: float = 10.0,
    max_per_voxel: int = 1,
    seed: int = 20260705,
) -> pd.DataFrame:
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be > 0")
    work = df.copy()
    work = work[(work["x_m"] >= float(x_range[0])) & (work["x_m"] <= float(x_range[1]))].copy()
    if work.empty:
        return work
    work = add_priority_columns(work)
    work = add_voxel_columns(work, voxel_mm=voxel_mm)
    j1, j2, j3 = _priority_arrays(work)
    work["_j1"] = j1
    work["_j2"] = j2
    work["_j3_neg"] = -j3
    work["_sample_id"] = work["sample_id"].to_numpy(dtype=np.int64)
    work = work.sort_values(["workspace_voxel_id", "_j1", "_j2", "_j3_neg", "_sample_id"], kind="mergesort")
    per_voxel = max(1, int(max_per_voxel))
    canonical = work.groupby("workspace_voxel_id", sort=False).head(per_voxel).copy()
    x_bin, _bins = _x_bin_indices(canonical["x_m"].to_numpy(dtype=float), x_range=x_range, x_bin_mm=x_bin_mm)
    canonical["_x_bin"] = x_bin

    if len(canonical) <= int(max_rows):
        selected = canonical.sort_values(["_x_bin", "workspace_voxel_id"], kind="mergesort")
    else:
        rng = np.random.default_rng(int(seed))
        bins = sorted(int(v) for v in canonical["_x_bin"].unique())
        quota_base = int(max_rows) // max(len(bins), 1)
        remainder = int(max_rows) - quota_base * max(len(bins), 1)
        selected_parts: list[pd.DataFrame] = []
        leftovers: list[pd.DataFrame] = []
        for pos, bin_id in enumerate(bins):
            group = canonical[canonical["_x_bin"] == bin_id].copy()
            group = group.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
            quota = quota_base + (1 if pos < remainder else 0)
            selected_parts.append(group.iloc[:quota])
            if len(group) > quota:
                leftovers.append(group.iloc[quota:])
        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else canonical.iloc[:0].copy()
        if len(selected) < int(max_rows) and leftovers:
            need = int(max_rows) - len(selected)
            extra_pool = pd.concat(leftovers, ignore_index=True)
            extra = extra_pool.sample(n=min(need, len(extra_pool)), random_state=int(rng.integers(0, 2**31 - 1)))
            selected = pd.concat([selected, extra], ignore_index=True)
        selected = selected.sort_values(["_x_bin", "workspace_voxel_id"], kind="mergesort")

    drop_cols = [c for c in selected.columns if c.startswith("_")]
    return selected.drop(columns=drop_cols).reset_index(drop=True)


def _chunk_to_frame(
    *,
    sample_ids: np.ndarray,
    beta: np.ndarray,
    theta: np.ndarray,
    xyz: np.ndarray,
    spec: HierarchicalGridSpec,
) -> pd.DataFrame:
    data: dict[str, Any] = {
        "sample_id": sample_ids.astype(np.int64, copy=False),
        "x_m": xyz[:, 0],
        "y_m": xyz[:, 1],
        "z_m": xyz[:, 2],
    }
    for i, col in enumerate(BETA_COLS):
        data[col] = beta[:, i]
    for i, col in enumerate(THETA_COLS):
        data[col] = theta[:, i]
    df = pd.DataFrame(data)
    df["joint1_step_deg"] = float(spec.beta12_deg[2])
    df["joint2_step_deg"] = float(spec.beta34_deg[2])
    df["joint3_step_deg"] = float(spec.beta56_deg[2])
    return add_priority_columns(df)


def _empty_output_frame() -> pd.DataFrame:
    cols = [
        "sample_id",
        *XYZ_COLS,
        *BETA_COLS,
        *THETA_COLS,
        "joint1_step_deg",
        "joint2_step_deg",
        "joint3_step_deg",
        "proximal_j1_norm_rad",
        "proximal_j2_norm_rad",
        "distal_j3_norm_rad",
        "hierarchical_priority_score",
        "source_component",
    ]
    return pd.DataFrame({col: pd.Series(dtype="float64") for col in cols})


class _ParquetAppender:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.writer = None

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(str(self.path), table.schema, compression="zstd")
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def _read_parquet_columns(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def _read_rows_by_sample_id(path: Path, sample_ids: set[int], *, batch_size: int = 200000) -> pd.DataFrame:
    if not sample_ids:
        return _empty_output_frame()
    import pyarrow.parquet as pq

    parts: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(str(path))
    for batch in parquet.iter_batches(batch_size=int(batch_size)):
        frame = batch.to_pandas()
        mask = frame["sample_id"].isin(sample_ids)
        if bool(mask.any()):
            parts.append(frame.loc[mask].copy())
    if not parts:
        return _empty_output_frame()
    return pd.concat(parts, ignore_index=True)


def _sample_xyz_for_plot(xyz: np.ndarray, *, max_points: int, seed: int) -> np.ndarray:
    if len(xyz) <= int(max_points):
        return xyz
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(xyz), size=int(max_points), replace=False)
    return xyz[np.sort(idx)]


def _plot_workspace(xyz: np.ndarray, out_path: Path, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if len(xyz) == 0:
        return
    xyz = _sample_xyz_for_plot(xyz, max_points=160000, seed=20260705)
    views = [
        ("isometric front", 24, -58),
        ("isometric back", 24, 122),
        ("top xy", 90, -90),
        ("side xz", 0, -90),
        ("side yz", 0, 0),
        ("low oblique", 12, -35),
    ]
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    norm = Normalize(vmin=float(xyz[:, 0].min()), vmax=float(xyz[:, 0].max()))
    fig = plt.figure(figsize=(18, 12))
    for idx, (name, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 0], cmap="viridis", norm=norm, s=0.8, alpha=0.18, linewidths=0, rasterized=True)
        ax.set_xlim(float(mins[0]), float(maxs[0]))
        ax.set_ylim(float(mins[1]), float(maxs[1]))
        ax.set_zlim(float(mins[2]), float(maxs[2]))
        try:
            ax.set_box_aspect(tuple(np.maximum(maxs - mins, 1.0e-9).tolist()))
        except Exception:
            pass
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
    cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.64, pad=0.02)
    cbar.set_label("x position (m)")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_x_hist(xyz: np.ndarray, out_path: Path, *, x_range: tuple[float, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(xyz[:, 0], bins=80, color="#2f6f91", alpha=0.85)
    ax.axvline(float(x_range[0]), color="#c7522a", linewidth=1.5)
    ax.axvline(float(x_range[1]), color="#c7522a", linewidth=1.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("count")
    ax.set_title("x distribution")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(Path(args.config))
    inputs = load_robot_inputs(cfg)
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", -1.0))
    spec = HierarchicalGridSpec(
        beta12_deg=tuple(args.beta12_deg),
        beta34_deg=tuple(args.beta34_deg),
        beta56_deg=tuple(args.beta56_deg),
    )
    if not (spec.step_deg("joint3") < spec.step_deg("joint2") < spec.step_deg("joint1")):
        raise SystemExit("expected hierarchical steps: joint3 < joint2 < joint1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    x_range = (float(args.x_min), float(args.x_max))
    total_rows = expected_grid_rows(spec)
    limit_rows = int(args.limit_rows) if int(args.limit_rows) > 0 else total_rows
    limit_rows = min(limit_rows, total_rows)
    full_path = out_dir / "full_fk_pool.parquet"
    x_path = out_dir / "x1p0_1p2_pool.parquet"

    full_writer = None if bool(args.skip_full_pool) else _ParquetAppender(full_path)
    x_writer = _ParquetAppender(x_path)
    xyz_sample_parts: list[np.ndarray] = []
    x_xyz_sample_parts: list[np.ndarray] = []
    generated = 0
    x_rows = 0

    try:
        for start in range(0, limit_rows, int(args.chunk_rows)):
            stop = min(start + int(args.chunk_rows), limit_rows)
            beta = generate_beta_chunk_rad(spec, start=start, stop=stop)
            theta = theta_from_beta_batch(beta, theta_sign=theta_sign)
            xyz = fk_dh_batch(theta, lengths_m=inputs.lengths_m, p_end_local_m=inputs.p_end_local_m)
            sample_ids = np.arange(start, stop, dtype=np.int64)
            frame = _chunk_to_frame(sample_ids=sample_ids, beta=beta, theta=theta, xyz=xyz, spec=spec)
            if full_writer is not None:
                full_writer.write(frame)
            mask = (frame["x_m"].to_numpy(dtype=float) >= x_range[0]) & (frame["x_m"].to_numpy(dtype=float) <= x_range[1])
            x_frame = frame.loc[mask].copy()
            x_writer.write(x_frame)
            generated += len(frame)
            x_rows += len(x_frame)
            if len(xyz_sample_parts) < 40:
                xyz_sample_parts.append(_sample_xyz_for_plot(xyz, max_points=max(1, int(args.plot_sample_rows) // 40), seed=start + 17))
            if len(x_frame) and len(x_xyz_sample_parts) < 40:
                x_xyz_sample_parts.append(
                    _sample_xyz_for_plot(x_frame[XYZ_COLS].to_numpy(dtype=float), max_points=max(1, int(args.plot_sample_rows) // 40), seed=start + 31)
                )
            if bool(args.progress):
                print(json.dumps({"generated": int(generated), "x_rows": int(x_rows), "total": int(limit_rows)}, ensure_ascii=False), flush=True)
    finally:
        if full_writer is not None:
            full_writer.close()
        x_writer.close()

    if not x_path.exists():
        _write_parquet(_empty_output_frame(), x_path)

    selection_columns = ["sample_id", *XYZ_COLS, *BETA_COLS]
    x_select = _read_parquet_columns(x_path, selection_columns)
    balanced_reports: dict[str, Any] = {}
    for rows in [int(v) for v in args.balanced_rows if int(v) > 0]:
        selected = select_workspace_balanced_subset(
            x_select,
            max_rows=rows,
            x_range=x_range,
            x_bin_mm=float(args.x_bin_mm),
            voxel_mm=float(args.voxel_mm),
            max_per_voxel=int(args.max_per_voxel),
            seed=int(args.seed) + rows,
        )
        selected_ids = set(int(v) for v in selected["sample_id"].to_numpy(dtype=np.int64))
        if selected_ids:
            selected_full = _read_rows_by_sample_id(x_path, selected_ids)
            selected_full = selected_full.merge(
                selected[["sample_id", "workspace_voxel_id"]],
                on="sample_id",
                how="left",
                suffixes=("", "_sel"),
            )
            if "workspace_voxel_id_sel" in selected_full.columns:
                selected_full["workspace_voxel_id"] = selected_full["workspace_voxel_id"].fillna(selected_full["workspace_voxel_id_sel"])
                selected_full = selected_full.drop(columns=["workspace_voxel_id_sel"])
        else:
            selected_full = _empty_output_frame()
        out_path = out_dir / f"balanced_{rows // 1000}k.parquet" if rows % 1000 == 0 else out_dir / f"balanced_{rows}.parquet"
        _write_parquet(selected_full.reset_index(drop=True), out_path)
        balanced_reports[str(rows)] = {
            "path": str(out_path),
            "rows": int(len(selected_full)),
            "metrics": workspace_metrics(
                selected_full[XYZ_COLS].to_numpy(dtype=float),
                x_range=x_range,
                x_bin_mm=float(args.x_bin_mm),
                yz_cell_mm=float(args.yz_cell_mm),
                voxel_mm=float(args.voxel_mm),
                seed=int(args.seed),
            )
            if len(selected_full)
            else {},
        }

    full_sample = np.vstack(xyz_sample_parts) if xyz_sample_parts else np.zeros((0, 3), dtype=float)
    x_sample = np.vstack(x_xyz_sample_parts) if x_xyz_sample_parts else np.zeros((0, 3), dtype=float)
    full_metrics = workspace_metrics(full_sample, x_range=x_range, x_bin_mm=float(args.x_bin_mm), yz_cell_mm=float(args.yz_cell_mm), voxel_mm=float(args.voxel_mm), seed=int(args.seed))
    x_metrics = workspace_metrics(
        x_select[XYZ_COLS].to_numpy(dtype=float),
        x_range=x_range,
        x_bin_mm=float(args.x_bin_mm),
        yz_cell_mm=float(args.yz_cell_mm),
        voxel_mm=float(args.voxel_mm),
        seed=int(args.seed),
    )

    viz_dir = report_dir / "visualizations"
    _plot_workspace(full_sample, viz_dir / "full_pool_workspace_3d_multi_view.png", title="Hierarchical beta FK full pool sample")
    if len(x_sample):
        _plot_workspace(x_sample, viz_dir / "x1p0_1p2_workspace_3d_multi_view.png", title="Hierarchical beta FK x=1.0..1.2 sample")
    if len(x_select):
        _plot_x_hist(x_select[XYZ_COLS].to_numpy(dtype=float), viz_dir / "x1p0_1p2_x_hist.png", x_range=x_range)

    report = {
        "mode": "hierarchical_beta_fk",
        "policy_version": POLICY_VERSION,
        "config": str(args.config),
        "out_dir": str(out_dir),
        "report_dir": str(report_dir),
        "spec": {
            "beta12_deg": list(spec.beta12_deg),
            "beta34_deg": list(spec.beta34_deg),
            "beta56_deg": list(spec.beta56_deg),
            "expected_grid_rows": int(total_rows),
            "generated_rows": int(generated),
            "limit_rows": int(limit_rows),
        },
        "paths": {
            "full_fk_pool": None if bool(args.skip_full_pool) else str(full_path),
            "x_pool": str(x_path),
        },
        "x_filter": {
            "x_min": float(x_range[0]),
            "x_max": float(x_range[1]),
            "rows": int(x_rows),
            "ratio": float(x_rows / max(generated, 1)),
        },
        "metrics": {
            "full_sample": full_metrics,
            "x_pool": x_metrics,
            "balanced": balanced_reports,
        },
        "visualizations": {
            "full_pool_workspace_3d_multi_view": str(viz_dir / "full_pool_workspace_3d_multi_view.png"),
            "x1p0_1p2_workspace_3d_multi_view": str(viz_dir / "x1p0_1p2_workspace_3d_multi_view.png"),
            "x1p0_1p2_x_hist": str(viz_dir / "x1p0_1p2_x_hist.png"),
        },
    }
    report_path = report_dir / "hierarchical_beta_fk_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    write_readme(report_dir, report)
    return report


def write_readme(report_dir: Path, report: dict[str, Any]) -> None:
    x_pool = report.get("metrics", {}).get("x_pool", {})
    lines = [
        "# Hierarchical Beta FK Dataset",
        "",
        "This dataset is generated directly in joint/beta space and uses FK to obtain Cartesian poses.",
        "No tension PSO or tension feasibility gate is applied in this phase.",
        "",
        "## Grid",
        "",
        f"- beta1/beta2: `{report['spec']['beta12_deg']}` deg",
        f"- beta3/beta4: `{report['spec']['beta34_deg']}` deg",
        f"- beta5/beta6: `{report['spec']['beta56_deg']}` deg",
        f"- expected full rows: `{report['spec']['expected_grid_rows']}`",
        f"- generated rows: `{report['spec']['generated_rows']}`",
        "",
        "## X Filter",
        "",
        f"- range: `{report['x_filter']['x_min']:.3f} .. {report['x_filter']['x_max']:.3f} m`",
        f"- rows: `{report['x_filter']['rows']}`",
        f"- ratio: `{report['x_filter']['ratio']:.4f}`",
        "",
        "## Workspace Metrics",
        "",
        f"- x-bin nonempty ratio: `{x_pool.get('x_bin_nonempty_ratio', float('nan')):.4f}`",
        f"- x-bin count CV: `{x_pool.get('x_bin_count_cv', float('nan')):.4f}`",
        f"- yz-cell x-range p95: `{x_pool.get('yz_cell_x_range_p95_mm', float('nan')):.2f} mm`",
        f"- 3D voxel count: `{x_pool.get('voxel_count_3d', 0)}`",
        f"- NN p95: `{x_pool.get('nn_dist_p95_mm', float('nan')):.2f} mm`",
        "",
        "## Outputs",
        "",
        f"- full FK pool: `{report['paths']['full_fk_pool']}`",
        f"- x-filtered pool: `{report['paths']['x_pool']}`",
    ]
    for rows, payload in report.get("metrics", {}).get("balanced", {}).items():
        lines.append(f"- balanced {rows}: `{payload.get('path')}`")
    lines.extend(["", "## Visualizations", ""])
    for path in report.get("visualizations", {}).values():
        lines.append(f"- `{path}`")
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate a hierarchical beta-grid FK-only pose dataset.")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "hierarchical_beta_fk_x1p0_1p2_v1")
    ap.add_argument("--report-dir", type=Path, default=REPO_ROOT / "runs" / "diagnostics" / "hierarchical_beta_fk_x1p0_1p2_v1")
    ap.add_argument("--beta12-deg", type=_parse_range_step, default=(-5.0, 5.0, 2.5))
    ap.add_argument("--beta34-deg", type=_parse_range_step, default=(-5.0, 5.0, 1.25))
    ap.add_argument("--beta56-deg", type=_parse_range_step, default=(-15.0, 15.0, 0.5))
    ap.add_argument("--x-min", type=float, default=1.0)
    ap.add_argument("--x-max", type=float, default=1.2)
    ap.add_argument("--x-bin-mm", type=float, default=5.0)
    ap.add_argument("--yz-cell-mm", type=float, default=20.0)
    ap.add_argument("--voxel-mm", type=float, default=10.0)
    ap.add_argument("--max-per-voxel", type=int, default=8)
    ap.add_argument("--chunk-rows", type=int, default=50000)
    ap.add_argument("--limit-rows", type=int, default=0, help="0 means full grid")
    ap.add_argument("--balanced-rows", type=int, nargs="*", default=[100000, 500000])
    ap.add_argument("--plot-sample-rows", type=int, default=160000)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--skip-full-pool", action="store_true")
    ap.add_argument("--progress", action="store_true")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    report = generate_dataset(parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
