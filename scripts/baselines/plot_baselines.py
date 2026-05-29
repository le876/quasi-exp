#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _read_parquet(path):
    import pyarrow.parquet as pq

    return pq.read_table(str(path)).to_pandas()


def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def _pearsonr(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size < 2:
        return float("nan")
    try:
        from scipy.stats import pearsonr  # type: ignore

        return float(pearsonr(a, b)[0])
    except Exception:
        C = np.corrcoef(a, b)
        return float(C[0, 1])


def _bar(values_by_model, title, ylabel, out_path):
    models = list(values_by_model.keys())
    vals = [values_by_model[m] for m in models]
    x = np.arange(len(models))

    plt.figure(figsize=(10, 6), dpi=200)
    bars = plt.bar(x, vals, color="#4C72B0")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)

    for rect, v in zip(bars, vals):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        plt.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height(),
            ("%g" % v),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


def _scatter_pred_true(y_true, y_pred, title, out_path):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    lim0 = np.nanmin([y_true.min(), y_pred.min()])
    lim1 = np.nanmax([y_true.max(), y_pred.max()])

    plt.figure(figsize=(6, 6), dpi=200)
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    plt.plot([lim0, lim1], [lim0, lim1], "r--", linewidth=1)
    plt.title(title)
    plt.xlabel("Truth")
    plt.ylabel("Pred")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


def _hist(resid, title, out_path):
    resid = np.asarray(resid, dtype=float).reshape(-1)
    resid = resid[np.isfinite(resid)]
    plt.figure(figsize=(10, 6), dpi=200)
    plt.hist(resid, bins=60, alpha=0.8, color="#55A868")
    plt.title(title)
    plt.xlabel("Residual (pred - true)")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


def _pearson_bar(r_vals, title, out_path, xlabel):
    r_vals = np.asarray(r_vals, dtype=float).reshape(-1)
    x = np.arange(r_vals.shape[0])
    plt.figure(figsize=(12, 4), dpi=200)
    plt.bar(x, r_vals, color="#C44E52")
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.ylim(-1.05, 1.05)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Pearson r")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


def _plot_mlp_curves(curves, title_prefix, out_dir):
    out_dir = Path(out_dir)
    if not isinstance(curves, dict):
        return
    if "loss_curve" in curves and curves["loss_curve"]:
        lc = np.asarray(curves["loss_curve"], dtype=float)
        plt.figure(figsize=(10, 6), dpi=200)
        plt.plot(np.arange(1, lc.shape[0] + 1), lc, marker="o", linewidth=1.5)
        plt.title("%s Loss Curve" % title_prefix)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(out_dir / ("%s_loss_curve.png" % title_prefix)))
        plt.close()

    if "validation_scores" in curves and curves["validation_scores"]:
        vs = np.asarray(curves["validation_scores"], dtype=float)
        plt.figure(figsize=(10, 6), dpi=200)
        plt.plot(np.arange(1, vs.shape[0] + 1), vs, marker="o", linewidth=1.5, color="#4C72B0")
        plt.title("%s Validation Score Curve" % title_prefix)
        plt.xlabel("Iteration")
        plt.ylabel("Validation score")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(out_dir / ("%s_val_score_curve.png" % title_prefix)))
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path, help="run_baselines.py output dir")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    all_metrics_p = out_dir / "all_metrics.json"
    if not all_metrics_p.exists():
        raise SystemExit("missing: %s" % all_metrics_p)
    all_metrics = _read_json(all_metrics_p)
    models = sorted(all_metrics.keys())

    plots_dir = out_dir / "plots"
    _ensure_dir(plots_dir)

    # metric bars
    def bar_metric(metric_key, title, ylabel, scale=1.0, exists_required=False):
        vals = {}
        for m in models:
            v = all_metrics[m].get(metric_key, None)
            if v is None:
                if exists_required:
                    return
                vals[m] = np.nan
            else:
                vals[m] = float(v) * float(scale)
        _bar(vals, title=title, ylabel=ylabel, out_path=plots_dir / ("%s_bar.png" % metric_key))

    bar_metric("theta_rmse_deg", "Theta RMSE (deg) by model", "RMSE (deg)")
    bar_metric("theta_mae_deg", "Theta MAE (deg) by model", "MAE (deg)")
    bar_metric("tension_rmse_n", "Tension RMSE (N) by model", "RMSE (N)")
    bar_metric("tension_mae_n", "Tension MAE (N) by model", "MAE (N)")
    bar_metric("pred_time_ms_per_sample", "Inference time by model", "ms / sample")
    bar_metric("tension_lt0_ratio", "%(T<0) by model", "%", scale=100.0)
    bar_metric("tension_gt_tmax_ratio", "%(T>tmax) by model", "%", scale=100.0)
    # optional
    bar_metric("ee_pos_p95_mm", "End-effector p95 error (mm) by model", "mm", exists_required=False)
    bar_metric("ee_pos_rmse_mm", "End-effector RMSE error (mm) by model", "mm", exists_required=False)

    # per-model plots
    overall_pearson = {}
    for m in models:
        mdir = out_dir / m
        preds_p = mdir / "preds_test.parquet"
        if not preds_p.exists():
            continue
        df = _read_parquet(preds_p)

        # extract arrays
        th_true = np.stack([df["theta_true_%d_rad" % (i + 1)].to_numpy() for i in range(30)], axis=1)
        th_pred = np.stack([df["theta_pred_%d_rad" % (i + 1)].to_numpy() for i in range(30)], axis=1)
        t_true = np.stack([df["tension_true_%d_n" % (j + 1)].to_numpy() for j in range(12)], axis=1)
        t_pred = np.stack([df["tension_pred_%d_n" % (j + 1)].to_numpy() for j in range(12)], axis=1)

        # pred vs true scatter (select dims)
        for i in [1, 15, 30]:
            _scatter_pred_true(
                th_true[:, i - 1],
                th_pred[:, i - 1],
                title="%s: theta_%d (rad) pred vs true" % (m, i),
                out_path=plots_dir / ("%s_pred_vs_true_theta_%d.png" % (m, i)),
            )
        for j in [1, 6, 12]:
            _scatter_pred_true(
                t_true[:, j - 1],
                t_pred[:, j - 1],
                title="%s: tension_%d (N) pred vs true" % (m, j),
                out_path=plots_dir / ("%s_pred_vs_true_tension_%d.png" % (m, j)),
            )

        # residual hists
        _hist((th_pred - th_true).reshape(-1), "%s: theta residuals (rad)" % m, plots_dir / ("%s_residual_hist_theta.png" % m))
        _hist((t_pred - t_true).reshape(-1), "%s: tension residuals (N)" % m, plots_dir / ("%s_residual_hist_tension.png" % m))

        # pearson per-dim + overall
        r_th = [ _pearsonr(th_true[:, i], th_pred[:, i]) for i in range(30) ]
        r_t = [ _pearsonr(t_true[:, j], t_pred[:, j]) for j in range(12) ]
        _pearson_bar(r_th, "%s: Pearson r per theta dim" % m, plots_dir / ("%s_pearson_theta_per_dim.png" % m), xlabel="theta dim (1..30)")
        _pearson_bar(r_t, "%s: Pearson r per tension dim" % m, plots_dir / ("%s_pearson_tension_per_dim.png" % m), xlabel="tension dim (1..12)")

        overall_pearson[m] = _pearsonr(
            np.concatenate([th_true.reshape(-1), t_true.reshape(-1)]),
            np.concatenate([th_pred.reshape(-1), t_pred.reshape(-1)]),
        )

        # mlp curves
        curves_p = mdir / "curves.json"
        if curves_p.exists():
            curves = _read_json(curves_p)
            if "loss_curve" in curves or "validation_scores" in curves:
                _plot_mlp_curves(curves, title_prefix=m, out_dir=plots_dir)

    if overall_pearson:
        _bar(
            overall_pearson,
            title="Overall Pearson r (theta+tension) by model",
            ylabel="Pearson r",
            out_path=plots_dir / "pearson_overall_bar.png",
        )

    print("OK: wrote plots to", plots_dir)


if __name__ == "__main__":
    main()
