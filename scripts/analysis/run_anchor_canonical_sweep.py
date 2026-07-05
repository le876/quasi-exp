#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PY_DEFAULT = sys.executable
BASE_CONFIG = Path("configs/robot_rods_only_mixed_20k_distal_preferred_anchor_v1.yaml")
SOURCE_DATASET = Path("data/mixed_beta_100k_distal_preferred_segmented_canonical/dataset.parquet")
SOURCE_META = Path("data/mixed_beta_100k_distal_preferred_segmented_canonical/dataset_meta.parquet")
V1_DATASET_DIR = Path("data/mixed_beta_20k_distal_preferred_anchor_v1")
DIAG_DIR = Path("runs/diagnostics/anchor_sweep_v2")
BASELINE_PREFIX = Path("runs")
TENSION_COLS = [f"tension_{i}_n" for i in range(1, 13)]
BETA_COLS = [f"beta{i}_rad" for i in range(1, 7)]
SPLITS = ["iid", "radius", "beta_block", "angular_sector"]
MODELS = "mlp,mlp_large,knn,rf"
V1_CONTINUITY_P95_N = 190.59667035158145
V1_THETA_RMS_P95_DEG = 8.341146050703324
FIRST_STAGE_VARIANTS = [
    {"variant": "anchor_v2_k16_w20", "anchor_k": 16, "w_anchor": 20.0},
    {"variant": "anchor_v2_k16_w30", "anchor_k": 16, "w_anchor": 30.0},
    {"variant": "anchor_v2_k32_w10", "anchor_k": 32, "w_anchor": 10.0},
    {"variant": "anchor_v2_k32_w20", "anchor_k": 32, "w_anchor": 20.0},
    {"variant": "anchor_v2_k32_w30", "anchor_k": 32, "w_anchor": 30.0},
]


def expected_component_counts(rows: int) -> dict[str, int]:
    if int(rows) == 20000:
        return {
            "sobol_full": 8000,
            "lhs_full": 4000,
            "workspace_balanced": 4000,
            "distal_biased": 4000,
        }
    raw = {
        "sobol_full": 0.40 * int(rows),
        "lhs_full": 0.20 * int(rows),
        "workspace_balanced": 0.20 * int(rows),
        "distal_biased": 0.20 * int(rows),
    }
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = int(rows) - sum(base.values())
    for key in sorted(base, key=lambda k: (-(raw[k] - base[k]), k))[:remainder]:
        base[key] += 1
    return base


def dataset_dir_for_variant(variant: str) -> Path:
    return Path(f"data/mixed_beta_20k_distal_preferred_{variant}")


def baseline_dir_for_variant(variant: str, split: str) -> Path:
    return BASELINE_PREFIX / f"baselines_mixed_beta_20k_distal_preferred_{variant}_{split}_fast4"


def analysis_path_for_variant(variant: str) -> Path:
    return DIAG_DIR / f"{variant}_dataset_analysis.json"


def continuity_path_for_variant(variant: str) -> Path:
    return DIAG_DIR / f"{variant}_local_continuity.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str], *, allow_exit_codes: set[int] | None = None) -> int:
    allow = {0} if allow_exit_codes is None else set(allow_exit_codes)
    print("[cmd]", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode
    if rc not in allow:
        raise SystemExit(f"command failed rc={rc}: {' '.join(cmd)}")
    return int(rc)


def rank_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [r for r in rows if bool(r.get("hard_passed", False))]
    return sorted(
        eligible,
        key=lambda r: (
            float(r.get("continuity_tension_p95_10mm", float("inf"))),
            float(r.get("rms_rnorm_q95", float("inf"))),
            float(r.get("max_tension", float("inf"))),
            str(r.get("variant", "")),
        ),
    )


def second_stage_weight(top_variant: dict[str, Any]) -> float:
    return max(20.0, float(top_variant["w_anchor"]))


def hard_gate_summary(
    *,
    dataset_path: Path,
    meta_path: Path,
    analysis_path: Path,
    expected_rows: int,
    expected_counts: dict[str, int],
) -> dict[str, Any]:
    dataset = pd.read_parquet(dataset_path)
    meta = pd.read_parquet(meta_path)
    analysis = _read_json(analysis_path)

    tension = dataset[TENSION_COLS].to_numpy(dtype=float)
    beta_max_abs = meta[BETA_COLS].abs().max()
    component_counts = meta["source_component"].astype(str).value_counts().to_dict()
    checks = {
        "rows": int(len(dataset)) == int(expected_rows),
        "meta_rows": int(len(meta)) == int(expected_rows),
        "component_counts": component_counts == expected_counts,
        "beta_1_4_le_5deg": bool((beta_max_abs[[f"beta{i}_rad" for i in range(1, 5)]] <= np.deg2rad(5) + 1e-12).all()),
        "beta_5_6_le_10deg": bool((beta_max_abs[[f"beta{i}_rad" for i in range(5, 7)]] <= np.deg2rad(10) + 1e-12).all()),
        "max_tension_le_2000": bool(float(np.max(tension)) <= 2000.0),
        "sat_ratio_zero": bool(float(np.mean(tension >= 2000.0 - 1e-9)) == 0.0),
        "rms_rnorm_q95_le_0p06": bool(float(analysis["rms_rnorm"]["q95"]) <= 0.06),
    }
    if "segmented_success" in meta.columns:
        checks["segmented_success_all"] = bool(meta["segmented_success"].astype(bool).all())
    return {
        "hard_passed": bool(all(checks.values())),
        "checks": checks,
        "component_counts": {str(k): int(v) for k, v in component_counts.items()},
        "beta_max_abs_deg": {str(k): float(np.rad2deg(v)) for k, v in beta_max_abs.items()},
        "rms_rnorm_q95": float(analysis["rms_rnorm"]["q95"]),
        "max_tension": float(analysis["tension_max"]["max"]),
        "sat_ratio_max_t": float(analysis["sat_ratio_max_t"]),
    }


def continuity_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    local_10 = payload["metrics"]["local"]["<= 10mm"]
    return {
        "continuity_passed_builtin": bool(payload["passed"]),
        "continuity_pairs_10mm": int(local_10["pairs"]),
        "continuity_tension_p50_10mm": float(local_10["tension_mae_n_p50"]),
        "continuity_tension_p90_10mm": float(local_10["tension_mae_n_p90"]),
        "continuity_tension_p95_10mm": float(local_10["tension_mae_n_p95"]),
        "continuity_theta_rms_p95_10mm": float(local_10["theta_rms_deg_p95"]),
    }


def baseline_best_metrics(variant: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = baseline_dir_for_variant(variant, split) / "all_metrics.json"
        if not path.exists():
            continue
        metrics = _read_json(path)
        model, row = min(metrics.items(), key=lambda kv: float(kv[1]["tension_mae_n"]))
        out[split] = {
            "model": model,
            "tension_mae_n": float(row["tension_mae_n"]),
            "tension_rmse_n": float(row["tension_rmse_n"]),
            "ee_pos_p95_mm": float(row["ee_pos_p95_mm"]),
            "theta_mae_deg": float(row["theta_mae_deg"]),
        }
    return out


def model_acceptance(top_variant: str) -> dict[str, Any]:
    ref = baseline_best_metrics("anchor_v1")
    top = baseline_best_metrics(top_variant)
    split_rows = {}
    for split in SPLITS:
        if split not in ref or split not in top:
            continue
        ref_mae = float(ref[split]["tension_mae_n"])
        top_mae = float(top[split]["tension_mae_n"])
        split_rows[split] = {
            "reference_tension_mae_n": ref_mae,
            "candidate_tension_mae_n": top_mae,
            "delta_pct": (top_mae - ref_mae) / max(ref_mae, 1.0e-12) * 100.0,
        }
    no_worse = bool(len(split_rows) == len(SPLITS) and all(r["delta_pct"] <= 5.0 for r in split_rows.values()))
    improved_two = bool(sum(1 for r in split_rows.values() if r["delta_pct"] <= -5.0) >= 2)
    return {
        "available": bool(len(split_rows) == len(SPLITS)),
        "no_split_worse_gt_5pct": no_worse,
        "at_least_two_splits_improve_ge_5pct": improved_two,
        "passed": bool(no_worse and improved_two),
        "splits": split_rows,
    }


def run_relabel(
    *,
    env_py: str,
    source_dataset: Path,
    source_meta: Path,
    out_dir: Path,
    anchor_k: int,
    w_anchor: float,
    num_samples: int,
    seed: int,
    workers: int,
    force: bool,
) -> None:
    dataset_path = out_dir / "dataset.parquet"
    meta_path = out_dir / "dataset_meta.parquet"
    report_path = out_dir / "dataset_report.json"
    if not force and dataset_path.exists() and meta_path.exists() and report_path.exists():
        print(f"[skip] relabel exists: {out_dir}", flush=True)
        return
    _run(
        [
            env_py,
            "scripts/analysis/relabel_anchor_canonical.py",
            "--dataset",
            str(source_dataset),
            "--meta",
            str(source_meta),
            "--config",
            str(BASE_CONFIG),
            "--out-dir",
            str(out_dir),
            "--num-samples",
            str(num_samples),
            "--seed",
            str(seed),
            "--component-scope",
            "same_component",
            "--distance-space",
            "beta",
            "--anchor-k",
            str(anchor_k),
            "--w-anchor",
            str(w_anchor),
            "--workers",
            str(workers),
            "--accept-infeasible",
        ]
    )


def run_analysis(*, env_py: str, dataset_dir: Path, variant: str, force: bool) -> None:
    out_path = analysis_path_for_variant(variant)
    if not force and out_path.exists():
        print(f"[skip] analysis exists: {out_path}", flush=True)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            env_py,
            "scripts/analyze_dataset.py",
            "--dataset",
            str(dataset_dir / "dataset.parquet"),
            "--meta",
            str(dataset_dir / "dataset_meta.parquet"),
            "--tension-cap",
            "2000",
            "--x-bins",
            "12",
            "--r-bins",
            "12",
            "--out-json",
            str(out_path),
        ]
    )


def run_continuity(*, env_py: str, dataset_dir: Path, variant: str, force: bool) -> None:
    out_path = continuity_path_for_variant(variant)
    if not force and out_path.exists():
        print(f"[skip] continuity exists: {out_path}", flush=True)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            env_py,
            "scripts/analysis/eval_local_continuity.py",
            "--dataset",
            str(dataset_dir / "dataset.parquet"),
            "--out-json",
            str(out_path),
            "--k-neighbors",
            "80",
        ],
        allow_exit_codes={0, 2},
    )


def summarize_variant(variant_cfg: dict[str, Any], expected_rows: int) -> dict[str, Any]:
    variant = str(variant_cfg["variant"])
    dataset_dir = dataset_dir_for_variant(variant)
    row: dict[str, Any] = dict(variant_cfg)
    row.update(
        hard_gate_summary(
            dataset_path=dataset_dir / "dataset.parquet",
            meta_path=dataset_dir / "dataset_meta.parquet",
            analysis_path=analysis_path_for_variant(variant),
            expected_rows=expected_rows,
            expected_counts=expected_component_counts(expected_rows),
        )
    )
    row.update(continuity_summary(continuity_path_for_variant(variant)))
    return row


def run_baseline_for_variant(*, env_py: str, variant: str, force: bool) -> None:
    dataset = dataset_dir_for_variant(variant) / "dataset.parquet"
    for split in SPLITS:
        out_dir = baseline_dir_for_variant(variant, split)
        if not force and (out_dir / "all_metrics.json").exists() and (out_dir / "SUMMARY.md").exists():
            print(f"[skip] baseline exists: {out_dir}", flush=True)
            continue
        _run(
            [
                env_py,
                "scripts/baselines/run_baselines.py",
                "--dataset",
                str(dataset),
                "--robot-config",
                str(BASE_CONFIG),
                "--out-dir",
                str(out_dir),
                "--backend",
                "classic",
                "--models",
                MODELS,
                "--feature-set",
                "poly_heavy",
                "--seed",
                "20260605",
                "--val-size",
                "0.1",
                "--test-size",
                "0.1",
                "--split",
                split,
                "--save-split-file",
                str(out_dir / f"split_{split}_seed20260605.npz"),
                "--eval-splits",
                "train,val,test",
                "--summary-split",
                "test",
                "--save-preds",
                "2000",
                "--save-preds-splits",
                "val,test",
                "--save-curves",
                "--n-jobs",
                "-1",
                "--wrapper-n-jobs",
                "1",
            ]
        )
        _run([env_py, "scripts/baselines/plot_baselines.py", "--out-dir", str(out_dir)])
        summary = subprocess.run(
            [env_py, "scripts/baselines/summarize_results.py", "--all-metrics", str(out_dir / "all_metrics.json")],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        (out_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")


def v1_reference_summary() -> dict[str, Any] | None:
    variant = "anchor_v1"
    dataset_dir = V1_DATASET_DIR
    analysis = Path("runs/diagnostics/mixed_beta_20k_distal_preferred_anchor_v1_dataset_analysis.json")
    continuity = Path("runs/diagnostics/mixed_beta_20k_distal_preferred_anchor_v1_local_continuity.json")
    if not ((dataset_dir / "dataset.parquet").exists() and analysis.exists() and continuity.exists()):
        return None
    row = {
        "variant": variant,
        "anchor_k": 16,
        "w_anchor": 10.0,
        "stage": "reference",
    }
    row.update(
        hard_gate_summary(
            dataset_path=dataset_dir / "dataset.parquet",
            meta_path=dataset_dir / "dataset_meta.parquet",
            analysis_path=analysis,
            expected_rows=20000,
            expected_counts=expected_component_counts(20000),
        )
    )
    row.update(continuity_summary(continuity))
    return row


def write_report(
    *,
    rows: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    baseline_variants: list[str],
    elapsed_s: float,
) -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "elapsed_s": float(elapsed_s),
        "rows": rows,
        "ranked_variants": [r["variant"] for r in ranked],
        "baseline": {variant: baseline_best_metrics(variant) for variant in baseline_variants},
    }
    (DIAG_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Anchor Canonical v2 Sweep Report",
        "",
        f"- Elapsed: `{elapsed_s:.2f}s`",
        f"- First-stage variants: `{len(FIRST_STAGE_VARIANTS)}`",
        "- Hard gates: rows/meta/component counts, beta range, max tension, zero saturation, rms_rnorm q95.",
        "",
        "## Continuity Ranking",
        "",
        "| rank | variant | hard | k | w | <=10mm T p95 N | <=10mm theta p95 deg | rms q95 | max T N |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            f"{i} | {row['variant']} | {row['hard_passed']} | {int(row['anchor_k'])} | "
            f"{float(row['w_anchor']):.1f} | {float(row['continuity_tension_p95_10mm']):.2f} | "
            f"{float(row['continuity_theta_rms_p95_10mm']):.2f} | {float(row['rms_rnorm_q95']):.5f} | "
            f"{float(row['max_tension']):.2f} |"
        )

    lines += [
        "",
        "## All Variant Quality",
        "",
        "| variant | stage | hard | <=10mm T p50 | p90 | p95 | theta p95 | rms q95 | max T |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['variant']} | {row.get('stage', 'first')} | {row['hard_passed']} | "
            f"{float(row['continuity_tension_p50_10mm']):.2f} | "
            f"{float(row['continuity_tension_p90_10mm']):.2f} | "
            f"{float(row['continuity_tension_p95_10mm']):.2f} | "
            f"{float(row['continuity_theta_rms_p95_10mm']):.2f} | "
            f"{float(row['rms_rnorm_q95']):.5f} | {float(row['max_tension']):.2f} |"
        )

    lines += [
        "",
        "## Baseline Best Metrics",
        "",
        "| variant | split | model | T MAE N | T RMSE N | EE p95 mm | theta MAE deg |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for variant in baseline_variants:
        for split, row in baseline_best_metrics(variant).items():
            lines.append(
                "| "
                f"{variant} | {split} | {row['model']} | {row['tension_mae_n']:.2f} | "
                f"{row['tension_rmse_n']:.2f} | {row['ee_pos_p95_mm']:.2f} | {row['theta_mae_deg']:.3f} |"
            )

    top = ranked[0] if ranked else None
    lines += ["", "## Acceptance", ""]
    if top is None:
        lines.append("- No hard-gated first-stage variant was available; do not run 100k.")
    else:
        continuity_ok = (
            float(top["continuity_tension_p95_10mm"]) <= 180.0
            or float(top["continuity_tension_p95_10mm"]) <= V1_CONTINUITY_P95_N * 0.90
        )
        theta_ok = float(top["continuity_theta_rms_p95_10mm"]) <= V1_THETA_RMS_P95_DEG + 0.3
        model_gate = model_acceptance(str(top["variant"]))
        lines.append(f"- Best continuity candidate: `{top['variant']}`.")
        lines.append(f"- Continuity gate: `{bool(continuity_ok and theta_ok)}`.")
        lines.append(f"- Model gate: `{bool(model_gate['passed'])}`.")
        for split, row in model_gate["splits"].items():
            lines.append(
                f"- `{split}` T MAE delta vs v1: `{row['delta_pct']:+.1f}%` "
                f"({row['reference_tension_mae_n']:.2f}N -> {row['candidate_tension_mae_n']:.2f}N)."
            )
        recommend_100k = bool(continuity_ok and theta_ok and model_gate["passed"])
        lines.append("- 100k recommendation: " + ("scale this second-stage configuration" if recommend_100k else "do not scale yet") + ".")

    (DIAG_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> None:
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    expected_rows = int(args.num_samples)

    v1 = v1_reference_summary()
    if v1 is not None:
        rows.append(v1)

    for cfg in FIRST_STAGE_VARIANTS:
        variant = str(cfg["variant"])
        out_dir = dataset_dir_for_variant(variant)
        run_relabel(
            env_py=args.env_py,
            source_dataset=args.source_dataset,
            source_meta=args.source_meta,
            out_dir=out_dir,
            anchor_k=int(cfg["anchor_k"]),
            w_anchor=float(cfg["w_anchor"]),
            num_samples=expected_rows,
            seed=int(args.seed),
            workers=int(args.workers),
            force=bool(args.force_relabel),
        )
        run_analysis(env_py=args.env_py, dataset_dir=out_dir, variant=variant, force=bool(args.force_analysis))
        run_continuity(env_py=args.env_py, dataset_dir=out_dir, variant=variant, force=bool(args.force_analysis))
        row = summarize_variant({**cfg, "stage": "first"}, expected_rows)
        rows.append(row)
        write_report(rows=rows, ranked=rank_variants([r for r in rows if r.get("stage") == "first"]), baseline_variants=["anchor_v1"], elapsed_s=time.time() - t0)

    first_rows = [r for r in rows if r.get("stage") == "first"]
    ranked = rank_variants(first_rows)
    baseline_variants = ["anchor_v1"]
    if ranked:
        top = ranked[0]
        second_variant = f"anchor_v2_second_stage_from_{top['variant']}"
        second_cfg = {
            "variant": second_variant,
            "anchor_k": 32,
            "w_anchor": second_stage_weight(top),
            "stage": "second",
            "parent_variant": top["variant"],
        }
        source_dir = dataset_dir_for_variant(str(top["variant"]))
        second_dir = dataset_dir_for_variant(second_variant)
        run_relabel(
            env_py=args.env_py,
            source_dataset=source_dir / "dataset.parquet",
            source_meta=source_dir / "dataset_meta.parquet",
            out_dir=second_dir,
            anchor_k=32,
            w_anchor=float(second_cfg["w_anchor"]),
            num_samples=expected_rows,
            seed=int(args.seed),
            workers=int(args.workers),
            force=bool(args.force_relabel),
        )
        run_analysis(env_py=args.env_py, dataset_dir=second_dir, variant=second_variant, force=bool(args.force_analysis))
        run_continuity(env_py=args.env_py, dataset_dir=second_dir, variant=second_variant, force=bool(args.force_analysis))
        rows.append(summarize_variant(second_cfg, expected_rows))

        candidate_variants = [str(r["variant"]) for r in ranked[:2]] + [second_variant]
        for variant in candidate_variants:
            run_baseline_for_variant(env_py=args.env_py, variant=variant, force=bool(args.force_baseline))
        baseline_variants += candidate_variants

    write_report(rows=rows, ranked=rank_variants([r for r in rows if r.get("stage") in {"first", "second"}]), baseline_variants=baseline_variants, elapsed_s=time.time() - t0)
    print(json.dumps(_read_json(DIAG_DIR / "summary.json"), ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--env-py", default=ENV_PY_DEFAULT)
    ap.add_argument("--source-dataset", type=Path, default=SOURCE_DATASET)
    ap.add_argument("--source-meta", type=Path, default=SOURCE_META)
    ap.add_argument("--num-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260607)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force-relabel", action="store_true")
    ap.add_argument("--force-analysis", action="store_true")
    ap.add_argument("--force-baseline", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_all:
        print(json.dumps({"first_stage_variants": FIRST_STAGE_VARIANTS}, indent=2), flush=True)
        return
    run_all(args)


if __name__ == "__main__":
    main()
