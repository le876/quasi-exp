#!/usr/bin/env python3
"""Run 100k fixed-layer model comparison without overwriting existing MLP runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


DANTE_PYTHON = "/mnt/ML_projects/conda_envs/dante_env/bin/python"
LAYER_LABEL = "s1_0125_s2_0250"
SPLITS = ("iid", "radius", "beta_block", "angular_sector")
OOD_SPLITS = ("radius", "beta_block", "angular_sector")
ROBOT_CONFIG = Path("configs/robot_rods_only_priority_grid_third_joint_first_v1.yaml")
DIAG_ROOT = Path("runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1")

DATASETS: dict[str, dict[str, str]] = {
    "raw": {
        "dataset": "data/priority_grid_fixed_layer_s1_0125_s2_0250_100k/dataset.parquet",
        "meta": "data/priority_grid_fixed_layer_s1_0125_s2_0250_100k/dataset_meta.parquet",
        "existing_root": "runs/baselines_fixed_layer_s1_0125_s2_0250_100k_raw_v1",
        "direct_root": "runs/baselines_fixed_layer_s1_0125_s2_0250_100k_raw_compare_v2",
        "beta_prefix": "runs/baselines_fixed_layer_s1_0125_s2_0250_100k_raw_beta_first_v1",
    },
    "relabel": {
        "dataset": "data/priority_grid_fixed_layer_s1_0125_s2_0250_100k_relabel_t1_k32_w40_huber_mean_iter1/dataset.parquet",
        "meta": "data/priority_grid_fixed_layer_s1_0125_s2_0250_100k_relabel_t1_k32_w40_huber_mean_iter1/dataset_meta.parquet",
        "existing_root": "runs/baselines_fixed_layer_s1_0125_s2_0250_100k_relabel_v1",
        "direct_root": "runs/baselines_fixed_layer_s1_0125_s2_0250_100k_relabel_compare_v2",
        "beta_prefix": "runs/baselines_fixed_layer_s1_0125_s2_0250_100k_relabel_beta_first_v1",
    },
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_direct_command(
    *,
    dataset: Path,
    split_file: Path,
    out_dir: Path,
    robot_config: Path,
    models: str,
    seed: int,
    n_jobs: int = 4,
    params_file: Path | None = None,
    max_train_rows: int = 0,
) -> list[str]:
    cmd = [
        DANTE_PYTHON,
        "scripts/baselines/run_baselines.py",
        "--dataset",
        str(dataset),
        "--out-dir",
        str(out_dir),
        "--robot-config",
        str(robot_config),
        "--split-file",
        str(split_file),
        "--models",
        str(models),
        "--backend",
        "classic",
        "--seed",
        str(int(seed)),
        "--eval-splits",
        "val,test",
        "--summary-split",
        "test",
        "--feature-set",
        "poly_heavy",
        "--save-preds",
        "2000",
        "--save-preds-splits",
        "val,test",
        "--save-curves",
        "--n-jobs",
        str(int(n_jobs)),
        "--wrapper-n-jobs",
        "1",
    ]
    if params_file is not None:
        cmd.extend(["--params-file", str(params_file)])
    if int(max_train_rows) > 0:
        cmd.extend(["--max-train-rows", str(int(max_train_rows))])
    return cmd


def build_beta_first_command(
    *,
    dataset: Path,
    meta: Path,
    out_prefix: Path,
    robot_config: Path,
    splits: tuple[str, ...],
    models: str,
    tension_modes: str,
    seed: int,
) -> list[str]:
    return [
        DANTE_PYTHON,
        "scripts/baselines/run_beta_first_baselines.py",
        "--dataset",
        str(dataset),
        "--meta",
        str(meta),
        "--out-prefix",
        str(out_prefix),
        "--splits",
        ",".join(str(s) for s in splits),
        "--models",
        str(models),
        "--tension-modes",
        str(tension_modes),
        "--seed",
        str(int(seed)),
        "--feature-set",
        "poly_heavy",
        "--robot-config",
        str(robot_config),
    ]


def _metrics_has_models(path: Path, models: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    try:
        payload = _read_json(path)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    return all(model in payload for model in models)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(payload, sort_keys=False)
    except Exception:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def direct_model_profile(model: str, *, n_jobs: int) -> dict[str, Any]:
    model = str(model)
    if model == "rf":
        return {
            "max_train_rows": 30000,
            "params": {
                "rf": {
                    "n_estimators": 80,
                    "max_features": "sqrt",
                    "min_samples_leaf": 2,
                    "n_jobs": int(n_jobs),
                }
            },
        }
    if model == "lgbm":
        return {
            "max_train_rows": 30000,
            "params": {
                "lgbm": {
                    "estimator__n_estimators": 300,
                    "estimator__learning_rate": 0.05,
                    "estimator__num_leaves": 31,
                    "estimator__subsample": 0.9,
                    "estimator__colsample_bytree": 0.9,
                    "estimator__reg_lambda": 1.0,
                    "estimator__n_jobs": int(n_jobs),
                }
            },
        }
    return {"max_train_rows": 0, "params": None}


def _beta_summary_complete(prefix: Path, splits: tuple[str, ...]) -> bool:
    summary_path = Path(f"{prefix}_summary.json")
    if not summary_path.exists():
        return False
    try:
        payload = _read_json(summary_path)
    except Exception:
        return False
    return isinstance(payload, dict) and all(split in payload for split in splits)


def collect_direct_rows(
    *,
    dataset_key: str,
    split: str,
    existing_metrics: Path,
    comparison_metrics: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, path in (("existing", existing_metrics), ("comparison", comparison_metrics)):
        if not path.exists():
            continue
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        for model, metrics in payload.items():
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "family": "direct",
                    "source": source,
                    "dataset": str(dataset_key),
                    "split": str(split),
                    "model": str(model),
                    "theta_mae_deg": _float_or_none(metrics.get("theta_mae_deg")),
                    "tension_mae_n": _float_or_none(metrics.get("tension_mae_n")),
                    "tension_rmse_n": _float_or_none(metrics.get("tension_rmse_n")),
                    "ee_pos_p95_mm": _float_or_none(metrics.get("ee_pos_p95_mm")),
                    "fit_time_s": _float_or_none(metrics.get("fit_time_s")),
                    "pred_time_ms_per_sample": _float_or_none(metrics.get("pred_time_ms_per_sample")),
                    "train_rows_used": _int_or_none(metrics.get("train_rows_used")),
                }
            )
    return rows


def collect_direct_model_dir_rows(*, dataset_key: str, split: str, comparison_split_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not comparison_split_dir.exists():
        return rows
    for metrics_path in sorted(comparison_split_dir.glob("*/all_metrics.json")):
        rows.extend(
            collect_direct_rows(
                dataset_key=dataset_key,
                split=split,
                existing_metrics=Path("__missing_existing_metrics__.json"),
                comparison_metrics=metrics_path,
            )
        )
    return rows


def collect_beta_first_rows(*, dataset_key: str, prefix: Path, splits: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        metrics_path = Path(f"{prefix}_{split}_fast4") / "all_metrics.json"
        if not metrics_path.exists():
            continue
        payload = _read_json(metrics_path)
        models = payload.get("models", {}) if isinstance(payload, dict) else {}
        if not isinstance(models, dict):
            continue
        for model, row in models.items():
            if not isinstance(row, dict):
                continue
            test = row.get("metrics", {}).get("test", {})
            if not isinstance(test, dict):
                continue
            rows.append(
                {
                    "family": "beta_first",
                    "source": "comparison",
                    "dataset": str(dataset_key),
                    "split": str(split),
                    "model": str(model),
                    "theta_mae_deg": _float_or_none(test.get("theta_mae_deg")),
                    "beta_mae_deg": _float_or_none(test.get("beta_mae_deg")),
                    "tension_mae_n": _float_or_none(test.get("tension_mae_n")),
                    "tension_rmse_n": _float_or_none(test.get("tension_rmse_n")),
                    "tension_p95_n": _float_or_none(test.get("tension_p95_n")),
                    "ee_pos_p95_mm": _float_or_none(test.get("ee_pos_p95_mm")),
                    "fit_time_s": _float_or_none(row.get("fit_time_s")),
                    "train_rows_used": _int_or_none(row.get("train_rows_used")),
                }
            )
    return rows


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _run_jobs(jobs: list[dict[str, Any]], *, max_parallel: int, summary_path: Path) -> list[dict[str, Any]]:
    running: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    pending = list(jobs)
    max_parallel = max(1, int(max_parallel))

    def launch(job: dict[str, Any]) -> None:
        log_path = Path(job["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8")
        log_fh.write("RUN " + " ".join(str(x) for x in job["cmd"]) + "\n")
        log_fh.flush()
        print(f"RUN {job['name']} log={log_path}", flush=True)
        job["started_at"] = time.strftime("%F %T")
        job["t0"] = time.perf_counter()
        job["log_fh"] = log_fh
        job["proc"] = subprocess.Popen(job["cmd"], stdout=log_fh, stderr=subprocess.STDOUT)
        _write_json(summary_path, {"status": "running", "running": [j["name"] for j in running] + [job["name"]], "pending": [j["name"] for j in pending]})

    failed = False
    while pending or running:
        while pending and len(running) < max_parallel and not failed:
            job = pending.pop(0)
            launch(job)
            running.append(job)

        time.sleep(5.0)
        still: list[dict[str, Any]] = []
        for job in running:
            proc = job["proc"]
            rc = proc.poll()
            if rc is None:
                still.append(job)
                continue
            job["log_fh"].close()
            elapsed_s = float(time.perf_counter() - float(job["t0"]))
            status = "success" if int(rc) == 0 else "failed"
            result = {
                "name": job["name"],
                "kind": job["kind"],
                "status": status,
                "returncode": int(rc),
                "started_at": job["started_at"],
                "finished_at": time.strftime("%F %T"),
                "elapsed_s": elapsed_s,
                "log": str(job["log"]),
            }
            completed.append(result)
            print(f"{status.upper()} {job['name']} rc={rc} elapsed={elapsed_s:.1f}s", flush=True)
            if int(rc) != 0:
                failed = True
        running = still
        _write_json(
            summary_path,
            {
                "status": "failed" if failed else "running",
                "completed": completed,
                "running": [j["name"] for j in running],
                "pending": [j["name"] for j in pending],
            },
        )
        if failed and not running:
            break

    if failed:
        raise RuntimeError("at least one comparison job failed")
    return completed


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["dataset"]), str(row["model"]))
        grouped.setdefault(key, []).append(row)
    best_by_group: dict[str, Any] = {}
    for key, items in grouped.items():
        family, dataset, model = key
        ood = [r for r in items if r.get("split") in OOD_SPLITS and r.get("tension_mae_n") is not None]
        all_rows = [r for r in items if r.get("tension_mae_n") is not None]
        best_by_group[f"{family}/{dataset}/{model}"] = {
            "family": family,
            "dataset": dataset,
            "model": model,
            "avg_tension_mae_n": _mean([r["tension_mae_n"] for r in all_rows]),
            "ood_avg_tension_mae_n": _mean([r["tension_mae_n"] for r in ood]),
            "avg_theta_mae_deg": _mean([r["theta_mae_deg"] for r in all_rows if r.get("theta_mae_deg") is not None]),
            "avg_ee_pos_p95_mm": _mean([r["ee_pos_p95_mm"] for r in all_rows if r.get("ee_pos_p95_mm") is not None]),
            "splits": len(all_rows),
        }
    candidates = [v for v in best_by_group.values() if v.get("ood_avg_tension_mae_n") is not None]
    best_ood = min(candidates, key=lambda item: float(item["ood_avg_tension_mae_n"])) if candidates else None
    return {"best_by_group": best_by_group, "best_ood": best_ood}


def _mean(values: list[Any]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return float(sum(cleaned) / len(cleaned))


def _write_markdown(path: Path, *, rows: list[dict[str, Any]], summary: dict[str, Any], jobs: list[dict[str, Any]]) -> None:
    lines = [
        "# Fixed-Layer 100k Model Comparison",
        "",
        f"- layer: `{LAYER_LABEL}`",
        f"- generated_at: `{time.strftime('%F %T')}`",
        f"- jobs: {len(jobs)}",
        "",
        "## Best OOD",
        "",
    ]
    best = summary.get("best_ood")
    if best:
        lines.append(
            "- `{family}/{dataset}/{model}` OOD T MAE = `{tension:.2f} N`, avg theta = `{theta:.4f} deg`, avg EE p95 = `{ee:.2f} mm`".format(
                family=best["family"],
                dataset=best["dataset"],
                model=best["model"],
                tension=float(best["ood_avg_tension_mae_n"]),
                theta=float(best["avg_theta_mae_deg"] or 0.0),
                ee=float(best["avg_ee_pos_p95_mm"] or 0.0),
            )
        )
    else:
        lines.append("- No complete OOD candidate.")
    lines.extend(
        [
            "",
            "## All Test Metrics",
            "",
            "| family | dataset | split | model | theta MAE deg | T MAE N | T RMSE N | EE p95 mm | fit s |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda r: (str(r["family"]), str(r["dataset"]), str(r["split"]), str(r["model"]))):
        lines.append(
            "| {family} | {dataset} | {split} | {model} | {theta} | {tmae} | {trmse} | {ee} | {fit} |".format(
                family=row["family"],
                dataset=row["dataset"],
                split=row["split"],
                model=row["model"],
                theta=_fmt(row.get("theta_mae_deg"), 4),
                tmae=_fmt(row.get("tension_mae_n"), 2),
                trmse=_fmt(row.get("tension_rmse_n"), 2),
                ee=_fmt(row.get("ee_pos_p95_mm"), 2),
                fit=_fmt(row.get("fit_time_s"), 1),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    direct_models = tuple(s.strip() for s in str(args.direct_models).split(",") if s.strip())
    jobs: list[dict[str, Any]] = []
    for dataset_key, spec in DATASETS.items():
        dataset = Path(spec["dataset"])
        meta = Path(spec["meta"])
        existing_root = Path(spec["existing_root"])
        direct_root = Path(spec["direct_root"])
        beta_prefix = Path(spec["beta_prefix"])
        if str(args.stages) in {"all", "direct"}:
            for split in SPLITS:
                split_file = existing_root / split / "split.npz"
                for model in direct_models:
                    out_dir = direct_root / split / model
                    metrics_path = out_dir / "all_metrics.json"
                    if _metrics_has_models(metrics_path, (model,)):
                        print(f"SKIP direct existing {dataset_key}/{split}/{model}: {metrics_path}", flush=True)
                        continue
                    profile = direct_model_profile(model, n_jobs=int(args.n_jobs))
                    params_file = None
                    if profile.get("params") is not None:
                        params_file = DIAG_ROOT / "model_comparison_params" / f"{dataset_key}_{split}_{model}.yaml"
                        _write_yaml(params_file, profile["params"])
                    jobs.append(
                        {
                            "name": f"direct_{dataset_key}_{split}_{model}",
                            "kind": "direct",
                            "cmd": build_direct_command(
                                dataset=dataset,
                                split_file=split_file,
                                out_dir=out_dir,
                                robot_config=Path(args.robot_config),
                                models=str(model),
                                seed=int(args.seed),
                                n_jobs=int(args.n_jobs),
                                params_file=params_file,
                                max_train_rows=int(profile.get("max_train_rows") or 0),
                            ),
                            "log": Path(args.log_dir) / f"direct_{dataset_key}_{split}_{model}.log",
                        }
                    )
        if str(args.stages) in {"all", "beta_first"}:
            if _beta_summary_complete(beta_prefix, SPLITS):
                print(f"SKIP beta_first existing {dataset_key}: {beta_prefix}_summary.json", flush=True)
            else:
                jobs.append(
                    {
                        "name": f"beta_first_{dataset_key}",
                        "kind": "beta_first",
                        "cmd": build_beta_first_command(
                            dataset=dataset,
                            meta=meta,
                            out_prefix=beta_prefix,
                            robot_config=Path(args.robot_config),
                            splits=SPLITS,
                            models=str(args.beta_models),
                            tension_modes=str(args.tension_modes),
                            seed=int(args.seed),
                        ),
                        "log": Path(args.log_dir) / f"beta_first_{dataset_key}.log",
                    }
                )
    return jobs


def collect_all_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_key, spec in DATASETS.items():
        existing_root = Path(spec["existing_root"])
        direct_root = Path(spec["direct_root"])
        for split in SPLITS:
            rows.extend(
                collect_direct_rows(
                    dataset_key=dataset_key,
                    split=split,
                    existing_metrics=existing_root / split / "all_metrics.json",
                    comparison_metrics=direct_root / split / "all_metrics.json",
                )
            )
            rows.extend(collect_direct_model_dir_rows(dataset_key=dataset_key, split=split, comparison_split_dir=direct_root / split))
        rows.extend(collect_beta_first_rows(dataset_key=dataset_key, prefix=Path(spec["beta_prefix"]), splits=SPLITS))
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260207)
    ap.add_argument("--robot-config", default=str(ROBOT_CONFIG))
    ap.add_argument("--direct-models", default="knn,rf,lgbm")
    ap.add_argument("--beta-models", default="mlp,rf,knn")
    ap.add_argument("--tension-modes", default="direct,beta_knn")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--stages", default="all", choices=["all", "direct", "beta_first", "collect"])
    ap.add_argument("--log-dir", default="runs/logs/fixed_layer_100k_model_comparison_jobs")
    ap.add_argument("--summary", default=str(DIAG_ROOT / "model_comparison_summary.json"))
    ap.add_argument("--summary-md", default=str(DIAG_ROOT / "model_comparison_summary.md"))
    ap.add_argument("--progress", default=str(DIAG_ROOT / "model_comparison_progress.json"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    jobs: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    if str(args.stages) != "collect":
        jobs = build_jobs(args)
        if jobs:
            completed = _run_jobs(jobs, max_parallel=int(args.max_parallel), summary_path=Path(args.progress))
        else:
            print("No comparison jobs to run; collecting existing metrics.", flush=True)

    rows = collect_all_rows()
    rollup = _summarize(rows)
    payload = {
        "mode": "fixed_layer_100k_model_comparison",
        "layer_label": LAYER_LABEL,
        "generated_at": time.strftime("%F %T"),
        "jobs_requested": len(jobs),
        "jobs": completed,
        "rows": rows,
        "summary": rollup,
    }
    _write_json(Path(args.summary), payload)
    _write_markdown(Path(args.summary_md), rows=rows, summary=rollup, jobs=completed)
    print(json.dumps({"summary": str(args.summary), "summary_md": str(args.summary_md), "rows": len(rows)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
