#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


DEFAULT_DATASET = REPO_ROOT / "data" / "priority_grid_fixed_layer_s1_0125_s2_0250_100k_relabel_t1_k32_w40_huber_mean_iter1" / "dataset.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "visualizations" / "cartesian_workspace_100k"


VIEWS = [
    ("isometric front", 24, -58),
    ("isometric back", 24, 122),
    ("top xy", 90, -90),
    ("side xz", 0, -90),
    ("side yz", 0, 0),
    ("low oblique", 12, -35),
]


def _set_axes(ax, mins: np.ndarray, maxs: np.ndarray) -> None:
    ax.set_xlim(float(mins[0]), float(maxs[0]))
    ax.set_ylim(float(mins[1]), float(maxs[1]))
    ax.set_zlim(float(mins[2]), float(maxs[2]))
    ax.set_xlabel("x (m)", labelpad=5)
    ax.set_ylabel("y (m)", labelpad=5)
    ax.set_zlabel("z (m)", labelpad=5)
    ranges = np.maximum(maxs - mins, 1e-9)
    try:
        ax.set_box_aspect(tuple(ranges.tolist()))
    except Exception:
        pass
    ax.grid(True, alpha=0.25)


def _load_xyz(dataset: Path) -> np.ndarray:
    df = pd.read_parquet(dataset, columns=["x_m", "y_m", "z_m"])
    return df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)


def plot_multi_view(xyz: np.ndarray, out_path: Path) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    color_values = xyz[:, 0]
    norm = Normalize(vmin=float(color_values.min()), vmax=float(color_values.max()))

    fig = plt.figure(figsize=(18.0, 12.0))
    for idx, (title, elev, azim) in enumerate(VIEWS, start=1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        sc = ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=color_values,
            cmap="viridis",
            norm=norm,
            s=0.8,
            alpha=0.17,
            linewidths=0,
            rasterized=True,
        )
        _set_axes(ax, mins, maxs)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title}\nelev={elev}, azim={azim}", fontsize=11)
    cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.64, pad=0.02)
    cbar.set_label("x position (m)")
    fig.suptitle("Cartesian workspace distribution, fixed-layer relabel 100k", fontsize=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_single_angle_grid(xyz: np.ndarray, out_dir: Path) -> list[str]:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    names: list[str] = []
    color_values = xyz[:, 0]
    norm = Normalize(vmin=float(color_values.min()), vmax=float(color_values.max()))
    for title, elev, azim in VIEWS:
        name = f"workspace_3d_{title.replace(' ', '_')}.png"
        fig = plt.figure(figsize=(8.2, 7.0))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=color_values,
            cmap="viridis",
            norm=norm,
            s=0.9,
            alpha=0.18,
            linewidths=0,
            rasterized=True,
        )
        _set_axes(ax, mins, maxs)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Cartesian workspace 100k: {title}")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.08)
        cbar.set_label("x position (m)")
        path = out_dir / name
        fig.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(fig)
        names.append(name)
    return names


def write_readme(out_dir: Path, dataset: Path, xyz: np.ndarray, images: list[str]) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    q = np.quantile(xyz, [0.01, 0.05, 0.5, 0.95, 0.99], axis=0)
    lines = [
        "# Cartesian Workspace 100k",
        "",
        f"- dataset: `{dataset}`",
        f"- rows: `{len(xyz)}`",
        f"- x range: `{mins[0]:.6f} .. {maxs[0]:.6f} m`",
        f"- y range: `{mins[1]:.6f} .. {maxs[1]:.6f} m`",
        f"- z range: `{mins[2]:.6f} .. {maxs[2]:.6f} m`",
        "",
        "## Quantiles",
        "",
        "| axis | p1 | p5 | p50 | p95 | p99 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for axis_idx, axis in enumerate(("x", "y", "z")):
        lines.append(
            f"| {axis} | {q[0, axis_idx]:.6f} | {q[1, axis_idx]:.6f} | {q[2, axis_idx]:.6f} | {q[3, axis_idx]:.6f} | {q[4, axis_idx]:.6f} |"
        )
    lines.extend(["", "## Images", ""])
    for image in images:
        lines.append(f"- `{image}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset)
    xyz = _load_xyz(dataset)
    multi_name = "workspace_3d_multi_view.png"
    plot_multi_view(xyz, out_dir / multi_name)
    images = [multi_name]
    if bool(args.single_views):
        images.extend(plot_single_angle_grid(xyz, out_dir))
    write_readme(out_dir, dataset, xyz, images)
    return {
        "out_dir": str(out_dir),
        "dataset": str(dataset),
        "rows": int(len(xyz)),
        "images": images,
        "x_range_m": [float(xyz[:, 0].min()), float(xyz[:, 0].max())],
        "y_range_m": [float(xyz[:, 1].min()), float(xyz[:, 1].max())],
        "z_range_m": [float(xyz[:, 2].min()), float(xyz[:, 2].max())],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot multi-view 3D Cartesian workspace distribution for the 100k dataset.")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--single-views", action="store_true")
    return ap.parse_args()


def main() -> None:
    import json

    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
