from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "runs" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DEFAULT_DATASET = REPO_ROOT / "data" / "standard_beta_sweep_100k_segmented_canonical" / "dataset.parquet"
DEFAULT_META = REPO_ROOT / "data" / "standard_beta_sweep_100k_segmented_canonical" / "dataset_meta.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "visualizations" / "segmented_canonical_100k"
DEFAULT_DIAG_DIR = REPO_ROOT / "runs" / "diagnostics"
DEFAULT_CLASSIC_RUN = REPO_ROOT / "runs" / "baselines_standard_beta_sweep_100k_segmented_canonical_classic_iid"
DEFAULT_TF_RUN = REPO_ROOT / "runs" / "baselines_standard_beta_sweep_100k_segmented_canonical_tf_iid"

SCALE_RUNS = {
    "2k": {
        "Classic MLP": REPO_ROOT / "runs" / "baselines_standard_beta_sweep_2k_segmented_canonical_classic_iid",
        "TF MLP": REPO_ROOT / "runs" / "baselines_standard_beta_sweep_2k_segmented_canonical_tf_iid",
    },
    "10k": {
        "Classic MLP": REPO_ROOT / "runs" / "baselines_standard_beta_sweep_10k_segmented_canonical_classic_iid",
        "TF MLP": REPO_ROOT / "runs" / "baselines_standard_beta_sweep_10k_segmented_canonical_tf_iid",
    },
    "100k": {
        "Classic MLP": DEFAULT_CLASSIC_RUN,
        "TF MLP": DEFAULT_TF_RUN,
    },
}

PLOT_SPECS = [
    ("metrics_scale_comparison.png", "2k/10k/100k model metrics comparison"),
    ("workspace_xy_max_tension.png", "Workspace x-y scatter colored by max tension"),
    ("workspace_xz_rms_rnorm.png", "Workspace x-z scatter colored by force residual RMS"),
    ("workspace_3d_quality_views.png", "3D workspace scatter colored by max tension and force residual RMS"),
    ("workspace_3d_orbit_views.png", "3D workspace orbit views colored by max tension"),
    ("tension_boxplot_12cables.png", "12-cable tension distribution boxplot"),
    ("max_tension_distribution.png", "Max tension histogram and ECDF"),
    ("rms_rnorm_distribution.png", "Force residual RMS histogram and ECDF"),
    ("local_distance_vs_tension_delta.png", "Local distance vs. tension delta quantiles"),
    ("model_error_ecdf_classic_vs_tf.png", "Classic MLP vs TF MLP error ECDF"),
]


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")
    return path


def _read_json(path: Path) -> dict:
    _require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_cols(df: pd.DataFrame, prefix: str, suffix: str) -> list[str]:
    cols: list[tuple[int, str]] = []
    for c in df.columns:
        if not (c.startswith(prefix) and c.endswith(suffix)):
            continue
        raw = c[len(prefix) : -len(suffix)]
        if raw.isdigit():
            cols.append((int(raw), c))
    return [c for _, c in sorted(cols)]


def _metric_block(metrics_path: Path) -> dict:
    payload = _read_json(metrics_path)
    if not payload:
        raise ValueError(f"empty metrics file: {metrics_path}")
    if len(payload) != 1:
        # Baseline files in this project usually contain one model for these runs.
        return next(iter(payload.values()))
    return next(iter(payload.values()))


def _model_preds(run_dir: Path, model_dir_name: str) -> pd.DataFrame:
    return pd.read_parquet(_require_file(run_dir / model_dir_name / "preds_test.parquet"))


def _sample_frame(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).sort_index()


def _savefig(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=180)
    plt.close(fig)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float).reshape(-1))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
    return x, y


def plot_metrics_scale_comparison(out_path: Path) -> dict:
    rows = []
    for scale, runs in SCALE_RUNS.items():
        for model_name, run_dir in runs.items():
            m = _metric_block(run_dir / "all_metrics.json")
            rows.append(
                {
                    "scale": scale,
                    "model": model_name,
                    "tension_mae_n": float(m["tension_mae_n"]),
                    "theta_mae_deg": float(m["theta_mae_deg"]),
                    "ee_pos_p95_mm": float(m["ee_pos_p95_mm"]),
                    "fit_time_s": float(m["fit_time_s"]),
                }
            )
    df = pd.DataFrame(rows)

    metrics = [
        ("tension_mae_n", "Tension MAE (N)"),
        ("theta_mae_deg", "Theta MAE (deg)"),
        ("ee_pos_p95_mm", "EE position p95 (mm)"),
        ("fit_time_s", "Fit time (s)"),
    ]
    scales = ["2k", "10k", "100k"]
    models = ["Classic MLP", "TF MLP"]
    colors = {"Classic MLP": "#3B82F6", "TF MLP": "#F97316"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    x = np.arange(len(scales), dtype=float)
    width = 0.36
    for ax, (key, ylabel) in zip(axes.ravel(), metrics):
        for offset, model in [(-width / 2, "Classic MLP"), (width / 2, "TF MLP")]:
            vals = []
            for scale in scales:
                v = df[(df["scale"] == scale) & (df["model"] == model)][key].iloc[0]
                vals.append(v)
            ax.bar(x + offset, vals, width=width, label=model, color=colors[model], alpha=0.88)
        ax.set_xticks(x, scales)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Segmented canonical dataset scale comparison", fontsize=14)
    _savefig(fig, out_path)
    return {"rows": rows}


def plot_workspace_xy_max_tension(dataset: pd.DataFrame, meta: pd.DataFrame, out_path: Path, sample_n: int) -> dict:
    merged = pd.concat(
        [
            dataset[["x_m", "y_m"]].reset_index(drop=True),
            meta[["max_tension"]].reset_index(drop=True),
        ],
        axis=1,
    )
    sdf = _sample_frame(merged, sample_n, seed=20260530)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sc = ax.scatter(sdf["x_m"], sdf["y_m"], c=sdf["max_tension"], s=5, cmap="viridis", alpha=0.75, linewidths=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Max tension (N)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Workspace x-y colored by max cable tension")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    _savefig(fig, out_path)
    return {"sampled_points": int(len(sdf))}


def plot_workspace_xz_rms_rnorm(dataset: pd.DataFrame, meta: pd.DataFrame, out_path: Path, sample_n: int) -> dict:
    merged = pd.concat(
        [
            dataset[["x_m", "z_m"]].reset_index(drop=True),
            meta[["rms_rnorm"]].reset_index(drop=True),
        ],
        axis=1,
    )
    sdf = _sample_frame(merged, sample_n, seed=20260531)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sc = ax.scatter(sdf["x_m"], sdf["z_m"], c=sdf["rms_rnorm"], s=5, cmap="magma", alpha=0.75, linewidths=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("RMS residual norm")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Workspace x-z colored by force residual RMS")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    _savefig(fig, out_path)
    return {"sampled_points": int(len(sdf))}


def plot_workspace_3d_quality_views(dataset: pd.DataFrame, meta: pd.DataFrame, out_path: Path, sample_n: int) -> dict:
    merged = pd.concat(
        [
            dataset[["x_m", "y_m", "z_m"]].reset_index(drop=True),
            meta[["max_tension", "rms_rnorm"]].reset_index(drop=True),
        ],
        axis=1,
    )
    sdf = _sample_frame(merged, sample_n, seed=20260602)

    fig = plt.figure(figsize=(14, 6.5))
    specs = [
        ("max_tension", "Max tension (N)", "viridis", "Workspace 3D colored by max tension"),
        ("rms_rnorm", "RMS residual norm", "magma", "Workspace 3D colored by force residual RMS"),
    ]
    axes = []
    for i, (col, cbar_label, cmap, title) in enumerate(specs, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        axes.append(ax)
        sc = ax.scatter(
            sdf["x_m"],
            sdf["y_m"],
            sdf["z_m"],
            c=sdf[col],
            s=4,
            cmap=cmap,
            alpha=0.72,
            linewidths=0,
            depthshade=False,
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.08)
        cb.set_label(cbar_label)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(title)
        ax.set_xticks([1.0, 1.1, 1.2])
        ax.view_init(elev=22, azim=-54)
        ax.grid(alpha=0.25)

    spans = [
        float(dataset["x_m"].max() - dataset["x_m"].min()),
        float(dataset["y_m"].max() - dataset["y_m"].min()),
        float(dataset["z_m"].max() - dataset["z_m"].min()),
    ]
    for ax in axes:
        ax.set_box_aspect(spans)
    fig.suptitle("Segmented canonical 100k workspace in 3D", fontsize=14)
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.15, top=0.86, wspace=0.20)
    _savefig(fig, out_path, tight=False)
    return {"sampled_points": int(len(sdf)), "view_elev_deg": 22, "view_azim_deg": -54}


def plot_workspace_3d_orbit_views(dataset: pd.DataFrame, meta: pd.DataFrame, out_path: Path, sample_n: int) -> dict:
    merged = pd.concat(
        [
            dataset[["x_m", "y_m", "z_m"]].reset_index(drop=True),
            meta[["max_tension"]].reset_index(drop=True),
        ],
        axis=1,
    )
    sdf = _sample_frame(merged, sample_n, seed=20260603)
    azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
    elev = 24
    spans = [
        float(dataset["x_m"].max() - dataset["x_m"].min()),
        float(dataset["y_m"].max() - dataset["y_m"].min()),
        float(dataset["z_m"].max() - dataset["z_m"].min()),
    ]
    color_min = float(np.percentile(sdf["max_tension"], 1))
    color_max = float(np.percentile(sdf["max_tension"], 99))

    fig = plt.figure(figsize=(18, 9.5))
    scatter = None
    for i, azim in enumerate(azimuths, start=1):
        ax = fig.add_subplot(2, 4, i, projection="3d")
        scatter = ax.scatter(
            sdf["x_m"],
            sdf["y_m"],
            sdf["z_m"],
            c=sdf["max_tension"],
            s=3,
            cmap="viridis",
            alpha=0.76,
            linewidths=0,
            depthshade=False,
            vmin=color_min,
            vmax=color_max,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"azim {azim} deg", fontsize=10)
        ax.set_axis_off()
        ax.set_box_aspect(spans)

    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.04, top=0.90, wspace=0.02, hspace=0.10)
    if scatter is not None:
        cbar_ax = fig.add_axes([0.91, 0.16, 0.015, 0.66])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label("Max tension (N)")
    fig.suptitle("Segmented canonical 100k workspace orbit views", fontsize=15)
    _savefig(fig, out_path, tight=False)
    return {
        "sampled_points": int(len(sdf)),
        "view_elev_deg": int(elev),
        "azimuths_deg": azimuths,
        "color": "max_tension",
    }


def plot_tension_boxplot(dataset: pd.DataFrame, out_path: Path) -> dict:
    tension_cols = _ordered_cols(dataset, "tension_", "_n")
    if len(tension_cols) != 12:
        raise ValueError(f"expected 12 tension columns, got {len(tension_cols)}")
    data = [dataset[c].to_numpy(dtype=float) for c in tension_cols]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.62)
    for patch in bp["boxes"]:
        patch.set(facecolor="#93C5FD", alpha=0.72, edgecolor="#1D4ED8")
    for median in bp["medians"]:
        median.set(color="#111827", linewidth=1.5)
    ax.set_xticks(np.arange(1, 13), [f"T{i}" for i in range(1, 13)])
    ax.set_ylabel("Tension (N)")
    ax.set_title("Canonical tension distribution across 12 cables")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _savefig(fig, out_path)
    return {
        "median_by_cable_n": {f"T{i + 1}": float(np.median(data[i])) for i in range(12)},
        "p95_by_cable_n": {f"T{i + 1}": float(np.percentile(data[i], 95)) for i in range(12)},
    }


def plot_distribution(values: np.ndarray, out_path: Path, title: str, xlabel: str, color: str) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    x_ecdf, y_ecdf = _ecdf(values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(values, bins=55, color=color, alpha=0.82, edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Histogram")
    axes[0].grid(axis="y", alpha=0.22)

    axes[1].plot(x_ecdf, y_ecdf, color=color, linewidth=2)
    for q in [50, 90, 95, 99]:
        v = float(np.percentile(values, q))
        axes[1].axvline(v, color="#6B7280", linestyle="--", linewidth=0.9, alpha=0.7)
        axes[1].text(v, 0.04 + q / 115.0, f"p{q}", rotation=90, va="bottom", ha="right", fontsize=8)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("ECDF")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Empirical CDF")
    axes[1].grid(alpha=0.22)

    fig.suptitle(title, fontsize=14)
    _savefig(fig, out_path)
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def plot_local_distance_vs_tension_delta(
    dataset: pd.DataFrame,
    out_path: Path,
    sample_n: int,
    k_neighbors: int,
) -> dict:
    tension_cols = _ordered_cols(dataset, "tension_", "_n")
    if len(tension_cols) != 12:
        raise ValueError(f"expected 12 tension columns, got {len(tension_cols)}")
    sdf = _sample_frame(dataset[["x_m", "y_m", "z_m"] + tension_cols], sample_n, seed=20260601)
    xyz = sdf[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    tension = sdf[tension_cols].to_numpy(dtype=float)

    k = min(max(2, int(k_neighbors)), len(sdf))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xyz)
    distances_m, indices = nn.kneighbors(xyz)
    src = np.repeat(np.arange(len(sdf))[:, None], k - 1, axis=1).ravel()
    dst = indices[:, 1:].ravel()
    dist_mm = distances_m[:, 1:].ravel() * 1000.0
    delta_mae_n = np.mean(np.abs(tension[src] - tension[dst]), axis=1)

    finite = np.isfinite(dist_mm) & np.isfinite(delta_mae_n)
    dist_mm = dist_mm[finite]
    delta_mae_n = delta_mae_n[finite]
    edges = np.array([0.0, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist_mm > lo) & (dist_mm <= hi)
        if not np.any(m):
            continue
        vals = delta_mae_n[m]
        rows.append(
            {
                "lo_mm": float(lo),
                "hi_mm": float(hi),
                "mid_mm": float((lo + hi) / 2.0),
                "pairs": int(np.sum(m)),
                "p50": float(np.percentile(vals, 50)),
                "p90": float(np.percentile(vals, 90)),
                "p95": float(np.percentile(vals, 95)),
            }
        )
    bdf = pd.DataFrame(rows)
    if bdf.empty:
        raise ValueError("no local neighbor pairs available for plotting")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(bdf["mid_mm"], bdf["p50"], marker="o", label="p50", color="#2563EB")
    ax.plot(bdf["mid_mm"], bdf["p90"], marker="o", label="p90", color="#F97316")
    ax.plot(bdf["mid_mm"], bdf["p95"], marker="o", label="p95", color="#DC2626")
    ax.set_xscale("log")
    ax.set_xlabel("Neighbor distance (mm)")
    ax.set_ylabel("Mean absolute tension delta across 12 cables (N)")
    ax.set_title("Local continuity: distance vs tension delta")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    _savefig(fig, out_path)
    return {
        "sampled_points": int(len(sdf)),
        "k_neighbors": int(k),
        "pairs": int(len(dist_mm)),
        "bins": rows,
    }


def plot_model_error_ecdf(classic_preds: pd.DataFrame, tf_preds: pd.DataFrame, out_path: Path) -> dict:
    def errors(df: pd.DataFrame) -> dict[str, np.ndarray]:
        tension_true_cols = _ordered_cols(df, "tension_true_", "_n")
        tension_pred_cols = _ordered_cols(df, "tension_pred_", "_n")
        theta_true_cols = _ordered_cols(df, "theta_true_", "_rad")
        theta_pred_cols = _ordered_cols(df, "theta_pred_", "_rad")
        if len(tension_true_cols) != 12 or len(tension_pred_cols) != 12:
            raise ValueError("predictions file must contain 12 true/pred tension columns")
        if len(theta_true_cols) != 30 or len(theta_pred_cols) != 30:
            raise ValueError("predictions file must contain 30 true/pred theta columns")
        t_true = df[tension_true_cols].to_numpy(dtype=float)
        t_pred = df[tension_pred_cols].to_numpy(dtype=float)
        th_true = df[theta_true_cols].to_numpy(dtype=float)
        th_pred = df[theta_pred_cols].to_numpy(dtype=float)
        return {
            "tension_abs_n": np.abs(t_pred - t_true).reshape(-1),
            "theta_abs_deg": (np.abs(th_pred - th_true) * 180.0 / np.pi).reshape(-1),
            "ee_err_mm": df["ee_err_mm"].to_numpy(dtype=float),
        }

    series = {
        "Classic MLP": errors(classic_preds),
        "TF MLP": errors(tf_preds),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    plot_keys = [
        ("tension_abs_n", "Tension absolute error (N)"),
        ("theta_abs_deg", "Theta absolute error (deg)"),
        ("ee_err_mm", "EE position error (mm)"),
    ]
    colors = {"Classic MLP": "#2563EB", "TF MLP": "#EA580C"}
    summary: dict[str, dict[str, float]] = {}
    for ax, (key, xlabel) in zip(axes, plot_keys):
        for model_name, vals_by_key in series.items():
            vals = vals_by_key[key]
            x, y = _ecdf(vals)
            ax.plot(x, y, label=model_name, color=colors[model_name], linewidth=2)
            summary.setdefault(model_name, {})[f"{key}_p50"] = float(np.percentile(vals, 50))
            summary[model_name][f"{key}_p95"] = float(np.percentile(vals, 95))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ECDF")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    fig.suptitle("Prediction error ECDF on saved test samples", fontsize=14)
    _savefig(fig, out_path)
    return summary


def write_readme(out_dir: Path, stats: dict, source_paths: dict[str, Path]) -> None:
    lines = [
        "# Segmented Canonical 100k Visualizations",
        "",
        "本目录由 `scripts/analysis/plot_segmented_canonical_100k.py` 生成，集中展示 100k segmented/canonical 数据集、张力解算质量、局部连续性和 MLP/TF MLP 预测误差。",
        "",
        "## Source Files",
        "",
    ]
    for label, path in source_paths.items():
        lines.append(f"- `{label}`: `{path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}`")
    lines.extend(["", "## Figures", ""])
    for filename, description in PLOT_SPECS:
        lines.append(f"- [{filename}]({filename}) - {description}")

    lines.extend(["", "## Key Numbers", ""])
    dataset = stats.get("dataset", {})
    local = stats.get("local_continuity", {})
    classic = stats.get("classic_metrics", {})
    tf = stats.get("tf_metrics", {})
    if dataset:
        lines.append(f"- Rows: `{dataset.get('rows')}`")
        lines.append(f"- Tension max p95: `{dataset.get('tension_max_p95_n'):.4f} N`")
        lines.append(f"- RMS residual p95: `{dataset.get('rms_rnorm_p95'):.6f}`")
    if local:
        lines.append(f"- Local continuity passed: `{local.get('passed')}`")
        lines.append(f"- 10mm tension MAE p95: `{local.get('tension_mae_10mm_p95_n'):.4f} N`")
    if classic:
        lines.append(f"- Classic MLP tension MAE: `{classic.get('tension_mae_n'):.4f} N`; EE p95: `{classic.get('ee_pos_p95_mm'):.4f} mm`")
    if tf:
        lines.append(f"- TF MLP tension MAE: `{tf.get('tension_mae_n'):.4f} N`; EE p95: `{tf.get('ee_pos_p95_mm'):.4f} mm`")

    lines.extend(["", "## Generation Stats", ""])
    for filename, description in PLOT_SPECS:
        detail = stats.get(filename, {})
        if detail:
            lines.append(f"- `{filename}`: `{json.dumps(detail, ensure_ascii=False)}`")

    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate segmented canonical 100k diagnostic visualizations.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scatter-sample", type=int, default=30000)
    parser.add_argument("--local-sample", type=int, default=20000)
    parser.add_argument("--local-k", type=int, default=31)
    args = parser.parse_args()

    dataset_path = _require_file(args.dataset)
    meta_path = _require_file(args.meta)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_parquet(dataset_path)
    meta = pd.read_parquet(meta_path)
    if len(dataset) != len(meta):
        raise ValueError(f"dataset/meta row mismatch: {len(dataset)} vs {len(meta)}")

    classic_metrics = _metric_block(DEFAULT_CLASSIC_RUN / "all_metrics.json")
    tf_metrics = _metric_block(DEFAULT_TF_RUN / "all_metrics.json")
    dataset_analysis = _read_json(DEFAULT_DIAG_DIR / "standard_beta_sweep_100k_segmented_canonical_dataset_analysis.json")
    local_continuity = _read_json(DEFAULT_DIAG_DIR / "standard_beta_sweep_100k_segmented_canonical_local_continuity.json")
    classic_preds = _model_preds(DEFAULT_CLASSIC_RUN, "mlp")
    tf_preds = _model_preds(DEFAULT_TF_RUN, "tf_mlp")

    stats: dict[str, object] = {
        "dataset": {
            "rows": int(len(dataset)),
            "tension_max_p95_n": float(dataset_analysis["tension_max"]["q95"]),
            "rms_rnorm_p95": float(np.percentile(meta["rms_rnorm"].to_numpy(dtype=float), 95)),
        },
        "local_continuity": {
            "passed": bool(local_continuity["passed"]),
            "tension_mae_10mm_p95_n": float(
                local_continuity["metrics"]["local"]["<= 10mm"]["tension_mae_n_p95"]
            ),
        },
        "classic_metrics": classic_metrics,
        "tf_metrics": tf_metrics,
    }

    stats["metrics_scale_comparison.png"] = plot_metrics_scale_comparison(out_dir / "metrics_scale_comparison.png")
    stats["workspace_xy_max_tension.png"] = plot_workspace_xy_max_tension(
        dataset, meta, out_dir / "workspace_xy_max_tension.png", sample_n=args.scatter_sample
    )
    stats["workspace_xz_rms_rnorm.png"] = plot_workspace_xz_rms_rnorm(
        dataset, meta, out_dir / "workspace_xz_rms_rnorm.png", sample_n=args.scatter_sample
    )
    stats["workspace_3d_quality_views.png"] = plot_workspace_3d_quality_views(
        dataset, meta, out_dir / "workspace_3d_quality_views.png", sample_n=args.scatter_sample
    )
    stats["workspace_3d_orbit_views.png"] = plot_workspace_3d_orbit_views(
        dataset, meta, out_dir / "workspace_3d_orbit_views.png", sample_n=args.scatter_sample
    )
    stats["tension_boxplot_12cables.png"] = plot_tension_boxplot(dataset, out_dir / "tension_boxplot_12cables.png")
    stats["max_tension_distribution.png"] = plot_distribution(
        meta["max_tension"].to_numpy(dtype=float),
        out_dir / "max_tension_distribution.png",
        title="Max tension distribution",
        xlabel="Max tension per sample (N)",
        color="#059669",
    )
    stats["rms_rnorm_distribution.png"] = plot_distribution(
        meta["rms_rnorm"].to_numpy(dtype=float),
        out_dir / "rms_rnorm_distribution.png",
        title="Force residual RMS distribution",
        xlabel="RMS residual norm",
        color="#7C3AED",
    )
    stats["local_distance_vs_tension_delta.png"] = plot_local_distance_vs_tension_delta(
        dataset,
        out_dir / "local_distance_vs_tension_delta.png",
        sample_n=args.local_sample,
        k_neighbors=args.local_k,
    )
    stats["model_error_ecdf_classic_vs_tf.png"] = plot_model_error_ecdf(
        classic_preds,
        tf_preds,
        out_dir / "model_error_ecdf_classic_vs_tf.png",
    )

    source_paths = {
        "dataset": dataset_path,
        "dataset_meta": meta_path,
        "dataset_analysis": DEFAULT_DIAG_DIR / "standard_beta_sweep_100k_segmented_canonical_dataset_analysis.json",
        "local_continuity": DEFAULT_DIAG_DIR / "standard_beta_sweep_100k_segmented_canonical_local_continuity.json",
        "classic_metrics": DEFAULT_CLASSIC_RUN / "all_metrics.json",
        "tf_metrics": DEFAULT_TF_RUN / "all_metrics.json",
        "classic_preds": DEFAULT_CLASSIC_RUN / "mlp" / "preds_test.parquet",
        "tf_preds": DEFAULT_TF_RUN / "tf_mlp" / "preds_test.parquet",
    }
    write_readme(out_dir, stats, source_paths)
    print(f"WROTE {out_dir}")
    for filename, _ in PLOT_SPECS:
        print(f"- {out_dir / filename}")
    print(f"- {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
