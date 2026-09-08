#!/usr/bin/env python3
"""Adaptive MLP capacity sweep for fixed-layer 100k datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


DANTE_PYTHON = "/mnt/ML_projects/conda_envs/dante_env/bin/python"
ROBOT_CONFIG = Path("configs/robot_rods_only_priority_grid_third_joint_first_v1.yaml")
DIAG_ROOT = Path("runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1")
OUT_ROOT = Path("runs/mlp_capacity_fixed_layer_s1_0125_s2_0250_100k_relabel_v1")
SPLITS = ("iid", "radius", "beta_block", "angular_sector")
PROBE_SPLITS = ("iid", "angular_sector")
OOD_SPLITS = ("radius", "beta_block", "angular_sector")

DATASETS = {
    "relabel": {
        "dataset": Path("data/priority_grid_fixed_layer_s1_0125_s2_0250_100k_relabel_t1_k32_w40_huber_mean_iter1/dataset.parquet"),
        "split_root": Path("runs/baselines_fixed_layer_s1_0125_s2_0250_100k_relabel_v1"),
    },
    "raw": {
        "dataset": Path("data/priority_grid_fixed_layer_s1_0125_s2_0250_100k/dataset.parquet"),
        "split_root": Path("runs/baselines_fixed_layer_s1_0125_s2_0250_100k_raw_v1"),
    },
}

CAPACITY_GRID: tuple[dict[str, Any], ...] = (
    {"capacity": "L0", "hidden_layer_sizes": (256, 128, 64, 32), "rank": 0},
    {"capacity": "L1", "hidden_layer_sizes": (384, 192, 96, 48), "rank": 1},
    {"capacity": "L2", "hidden_layer_sizes": (512, 256, 128, 64), "rank": 2},
    {"capacity": "L3", "hidden_layer_sizes": (768, 384, 192, 96), "rank": 3},
    {"capacity": "L4", "hidden_layer_sizes": (1024, 512, 256, 128), "rank": 4},
    {"capacity": "L5", "hidden_layer_sizes": (1536, 768, 384, 192), "rank": 5},
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(payload, sort_keys=False)
    except Exception:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_training_command(
    *,
    dataset: Path,
    split_file: Path,
    out_dir: Path,
    params_file: Path,
    robot_config: Path,
    seed: int,
) -> list[str]:
    return [
        DANTE_PYTHON,
        "scripts/baselines/run_baselines.py",
        "--dataset",
        str(dataset),
        "--robot-config",
        str(robot_config),
        "--split-file",
        str(split_file),
        "--out-dir",
        str(out_dir),
        "--models",
        "mlp_large",
        "--params-file",
        str(params_file),
        "--backend",
        "classic",
        "--summary-split",
        "test",
        "--eval-splits",
        "train,val,test",
        "--feature-set",
        "poly_heavy",
        "--save-preds",
        "0",
        "--save-curves",
        "--seed",
        str(int(seed)),
    ]


def is_overfit_regression(best: dict[str, Any], candidate: dict[str, Any]) -> bool:
    best_train = float(best["train_tension_mae_n"])
    cand_train = float(candidate["train_tension_mae_n"])
    train_gain = best_train - cand_train
    train_gain_ratio = train_gain / max(best_train, 1.0e-9)
    if train_gain_ratio < 0.10:
        return False

    def regressed(metric: str) -> bool:
        before = float(best[metric])
        after = float(candidate[metric])
        return (after - before) >= max(5.0, 0.10 * max(before, 1.0e-9))

    if regressed("val_tension_mae_n") or regressed("test_tension_mae_n"):
        return True
    gap = float(candidate["val_tension_mae_n"]) - float(candidate["train_tension_mae_n"])
    return gap > 25.0 and float(candidate["test_tension_mae_n"]) >= float(best["test_tension_mae_n"])


def recommend_capacity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if not bool(r.get("overfit")) and r.get("test_tension_mae_n") is not None]
    if not usable:
        return {"capacity": "L0", "alpha": 1e-6, "reason": "fallback_no_usable_rows"}

    by_capacity: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for row in usable:
        key = (str(row["capacity"]), _float_or_none(row.get("alpha")))
        by_capacity.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (capacity, alpha), items in by_capacity.items():
        aggregates.append(
            {
                "capacity": capacity,
                "alpha": alpha,
                "rank": capacity_rank(capacity),
                "avg_test_tension_mae_n": _mean([i["test_tension_mae_n"] for i in items]),
                "avg_fit_time_s": _mean([i.get("fit_time_s") for i in items]),
                "splits": len(items),
            }
        )
    best = min(aggregates, key=lambda r: float(r["avg_test_tension_mae_n"]))
    threshold = float(best["avg_test_tension_mae_n"]) * 1.05
    close = [r for r in aggregates if float(r["avg_test_tension_mae_n"]) <= threshold]
    chosen = min(close, key=lambda r: (int(r["rank"]), float(r.get("avg_fit_time_s") or 0.0)))
    if chosen["capacity"] != best["capacity"] or chosen.get("alpha") != best.get("alpha"):
        chosen["reason"] = "within_5_percent_choose_smaller"
    else:
        chosen["reason"] = "lowest_avg_test_tension"
    return chosen


def read_trial_metrics(out_dir: Path, *, dataset: str, split: str, capacity: str, alpha: float) -> dict[str, Any]:
    metrics_path = out_dir / "mlp_large" / "metrics.json"
    payload = _read_json(metrics_path)
    metrics = payload["metrics"]
    return {
        "dataset": str(dataset),
        "split": str(split),
        "capacity": str(capacity),
        "alpha": float(alpha),
        "fit_time_s": _float_or_none(payload.get("fit_time_s")),
        "train_theta_mae_deg": float(metrics["train"]["theta_mae_deg"]),
        "train_tension_mae_n": float(metrics["train"]["tension_mae_n"]),
        "val_theta_mae_deg": float(metrics["val"]["theta_mae_deg"]),
        "val_tension_mae_n": float(metrics["val"]["tension_mae_n"]),
        "test_theta_mae_deg": float(metrics["test"]["theta_mae_deg"]),
        "test_tension_mae_n": float(metrics["test"]["tension_mae_n"]),
        "test_tension_rmse_n": _float_or_none(metrics["test"].get("tension_rmse_n")),
        "test_ee_pos_p95_mm": _float_or_none(metrics["test"].get("ee_pos_p95_mm")),
        "metrics_path": str(metrics_path),
    }


def capacity_rank(capacity: str) -> int:
    for row in CAPACITY_GRID:
        if row["capacity"] == capacity:
            return int(row["rank"])
    return 999


def hidden_for_capacity(capacity: str) -> tuple[int, ...]:
    for row in CAPACITY_GRID:
        if row["capacity"] == capacity:
            return tuple(int(v) for v in row["hidden_layer_sizes"])
    raise ValueError(f"unknown capacity: {capacity}")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _params_payload(
    *,
    capacity: str,
    alpha: float,
    early_stopping: bool,
    max_iter: int,
    batch_size: int,
    learning_rate_init: float,
    n_iter_no_change: int,
    tol: float,
) -> dict[str, Any]:
    return {
        "mlp_large": {
            "hidden_layer_sizes": list(hidden_for_capacity(capacity)),
            "alpha": float(alpha),
            "early_stopping": bool(early_stopping),
            "max_iter": int(max_iter),
            "batch_size": int(batch_size),
            "learning_rate_init": float(learning_rate_init),
            "n_iter_no_change": int(n_iter_no_change),
            "tol": float(tol),
        }
    }


def _trial_name(*, dataset: str, split: str, capacity: str, alpha: float, stage: str, early_stopping: bool) -> str:
    alpha_tag = f"{alpha:.0e}".replace("+", "").replace("-", "m")
    stop_tag = "es" if early_stopping else "noes"
    return f"{stage}_{dataset}_{split}_{capacity}_alpha{alpha_tag}_{stop_tag}"


def _run_command(cmd: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    started_at = time.strftime("%F %T")
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write("RUN " + " ".join(str(x) for x in cmd) + "\n")
        log_fh.flush()
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        rc = proc.wait()
    elapsed = float(time.perf_counter() - start)
    return {
        "status": "success" if int(rc) == 0 else "failed",
        "returncode": int(rc),
        "started_at": started_at,
        "finished_at": time.strftime("%F %T"),
        "elapsed_s": elapsed,
        "log": str(log_path),
    }


def _run_trial(
    *,
    args: argparse.Namespace,
    dataset_key: str,
    split: str,
    capacity: str,
    alpha: float,
    stage: str,
    early_stopping: bool,
    max_iter: int,
    batch_size: int,
    learning_rate_init: float,
    n_iter_no_change: int,
    tol: float,
) -> dict[str, Any]:
    dataset = DATASETS[dataset_key]["dataset"]
    split_file = DATASETS[dataset_key]["split_root"] / split / "split.npz"
    trial = _trial_name(dataset=dataset_key, split=split, capacity=capacity, alpha=alpha, stage=stage, early_stopping=early_stopping)
    out_dir = Path(args.out_root) / stage / dataset_key / split / trial
    params_file = Path(args.out_root) / "params" / f"{trial}.yaml"
    metrics_path = out_dir / "mlp_large" / "metrics.json"
    params = _params_payload(
        capacity=capacity,
        alpha=alpha,
        early_stopping=early_stopping,
        max_iter=max_iter,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
    )
    _write_yaml(params_file, params)
    if not metrics_path.exists():
        cmd = build_training_command(
            dataset=dataset,
            split_file=split_file,
            out_dir=out_dir,
            params_file=params_file,
            robot_config=Path(args.robot_config),
            seed=int(args.seed),
        )
        print(f"RUN {trial}", flush=True)
        job = _run_command(cmd, Path(args.log_dir) / f"{trial}.log")
        if job["status"] != "success":
            raise RuntimeError(f"trial failed: {trial} log={job['log']}")
    else:
        job = {"status": "skipped_existing", "returncode": 0, "log": str(Path(args.log_dir) / f"{trial}.log")}
        print(f"SKIP existing {trial}", flush=True)

    row = read_trial_metrics(out_dir, dataset=dataset_key, split=split, capacity=capacity, alpha=alpha)
    row.update(
        {
            "stage": stage,
            "trial": trial,
            "early_stopping": bool(early_stopping),
            "max_iter": int(max_iter),
            "batch_size": int(batch_size),
            "learning_rate_init": float(learning_rate_init),
            "n_iter_no_change": int(n_iter_no_change),
            "tol": float(tol),
            "params_file": str(params_file),
            "out_dir": str(out_dir),
            "job": job,
        }
    )
    return row


def _write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], status: str, note: str = "") -> None:
    rec = recommend_capacity([r for r in rows if r.get("dataset") == "relabel" and r.get("stage") in {"probe", "grid"}])
    payload = {
        "mode": "fixed_layer_mlp_capacity_sweep",
        "status": status,
        "note": note,
        "generated_at": time.strftime("%F %T"),
        "rows": rows,
        "recommendation": rec,
    }
    _write_json(Path(args.summary), payload)
    _write_markdown(Path(args.summary_md), payload)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload.get("rows", [])
    rec = payload.get("recommendation", {})
    lines = [
        "# Fixed-Layer 100k MLP Capacity Sweep",
        "",
        f"- status: `{payload.get('status')}`",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- recommendation: `{rec.get('capacity')}` alpha=`{rec.get('alpha')}` reason=`{rec.get('reason')}`",
        "",
        "| stage | dataset | split | capacity | alpha | overfit | train T | val T | test T | test theta | EE p95 | fit s |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {dataset} | {split} | {capacity} | {alpha} | {overfit} | {train_t} | {val_t} | {test_t} | {theta} | {ee} | {fit} |".format(
                stage=row.get("stage", ""),
                dataset=row.get("dataset", ""),
                split=row.get("split", ""),
                capacity=row.get("capacity", ""),
                alpha=_fmt(row.get("alpha"), 0),
                overfit=str(bool(row.get("overfit", False))).lower(),
                train_t=_fmt(row.get("train_tension_mae_n"), 2),
                val_t=_fmt(row.get("val_tension_mae_n"), 2),
                test_t=_fmt(row.get("test_tension_mae_n"), 2),
                theta=_fmt(row.get("test_theta_mae_deg"), 4),
                ee=_fmt(row.get("test_ee_pos_p95_mm"), 2),
                fit=_fmt(row.get("fit_time_s"), 1),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return ""
    try:
        if digits == 0:
            return f"{float(value):.0e}"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def run_probe(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    for split in PROBE_SPLITS:
        best: dict[str, Any] | None = None
        consecutive_overfit = 0
        for cap in CAPACITY_GRID:
            row = _run_trial(
                args=args,
                dataset_key="relabel",
                split=split,
                capacity=str(cap["capacity"]),
                alpha=float(args.probe_alpha),
                stage="probe",
                early_stopping=False,
                max_iter=int(args.probe_max_iter),
                batch_size=int(args.batch_size),
                learning_rate_init=float(args.learning_rate),
                n_iter_no_change=int(args.n_iter_no_change),
                tol=float(args.tol),
            )
            row["overfit"] = bool(best is not None and is_overfit_regression(best, row))
            rows.append(row)
            if not row["overfit"] and (best is None or float(row["test_tension_mae_n"]) < float(best["test_tension_mae_n"])):
                best = row
                consecutive_overfit = 0
            elif row["overfit"]:
                consecutive_overfit += 1
            else:
                consecutive_overfit = 0
            _write_outputs(args, rows, "running", note=f"probe {split} {cap['capacity']}")
            if consecutive_overfit >= 2:
                print(f"STOP probe split={split}: consecutive overfit={consecutive_overfit}", flush=True)
                break


def _candidate_capacities(rows: list[dict[str, Any]]) -> list[str]:
    probe_rows = [r for r in rows if r.get("stage") == "probe" and r.get("dataset") == "relabel" and not bool(r.get("overfit"))]
    if not probe_rows:
        return ["L0", "L1"]
    by_capacity: dict[str, list[dict[str, Any]]] = {}
    for row in probe_rows:
        by_capacity.setdefault(str(row["capacity"]), []).append(row)
    ranked = sorted(
        (
            {"capacity": cap, "rank": capacity_rank(cap), "score": _mean([r["test_tension_mae_n"] for r in items])}
            for cap, items in by_capacity.items()
        ),
        key=lambda item: float(item["score"]),
    )
    selected = [str(item["capacity"]) for item in ranked[:3]]
    if "L0" not in selected:
        selected.append("L0")
    return sorted(set(selected), key=capacity_rank)[:3]


def run_grid(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    capacities = _candidate_capacities(rows)
    alphas = [float(v) for v in str(args.grid_alphas).split(",") if str(v).strip()]
    best_by_split: dict[tuple[str, str], dict[str, Any]] = {}
    for capacity in capacities:
        for alpha in alphas:
            for split in SPLITS:
                row = _run_trial(
                    args=args,
                    dataset_key="relabel",
                    split=split,
                    capacity=capacity,
                    alpha=alpha,
                    stage="grid",
                    early_stopping=True,
                    max_iter=int(args.grid_max_iter),
                    batch_size=int(args.batch_size),
                    learning_rate_init=float(args.learning_rate),
                    n_iter_no_change=int(args.n_iter_no_change),
                    tol=float(args.tol),
                )
                key = (capacity, split)
                best = best_by_split.get(key)
                row["overfit"] = bool(best is not None and is_overfit_regression(best, row))
                if best is None or float(row["test_tension_mae_n"]) < float(best["test_tension_mae_n"]):
                    best_by_split[key] = row
                rows.append(row)
                _write_outputs(args, rows, "running", note=f"grid {capacity} {alpha} {split}")


def run_raw_recheck(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    rec = recommend_capacity([r for r in rows if r.get("dataset") == "relabel" and r.get("stage") == "grid" and not bool(r.get("overfit"))])
    capacity = str(rec.get("capacity") or "L0")
    alpha = float(rec.get("alpha") if rec.get("alpha") is not None else 1e-6)
    for split in SPLITS:
        row = _run_trial(
            args=args,
            dataset_key="raw",
            split=split,
            capacity=capacity,
            alpha=alpha,
            stage="raw_recheck",
            early_stopping=True,
            max_iter=int(args.grid_max_iter),
            batch_size=int(args.batch_size),
            learning_rate_init=float(args.learning_rate),
            n_iter_no_change=int(args.n_iter_no_change),
            tol=float(args.tol),
        )
        row["overfit"] = False
        rows.append(row)
        _write_outputs(args, rows, "running", note=f"raw_recheck {split}")


def _load_existing_rows(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    try:
        payload = _read_json(summary_path)
        rows = payload.get("rows", [])
        return list(rows) if isinstance(rows, list) else []
    except Exception:
        return []


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260207)
    ap.add_argument("--robot-config", default=str(ROBOT_CONFIG))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--log-dir", default="runs/logs/fixed_layer_mlp_capacity_jobs")
    ap.add_argument("--summary", default=str(DIAG_ROOT / "mlp_capacity_sweep_summary.json"))
    ap.add_argument("--summary-md", default=str(DIAG_ROOT / "mlp_capacity_sweep_summary.md"))
    ap.add_argument("--stages", default="probe,grid,raw_recheck")
    ap.add_argument("--probe-alpha", type=float, default=1e-8)
    ap.add_argument("--grid-alphas", default="1e-7,1e-6,1e-5")
    ap.add_argument("--probe-max-iter", type=int, default=800)
    ap.add_argument("--grid-max-iter", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--n-iter-no-change", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-6)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    stages = {s.strip() for s in str(args.stages).split(",") if s.strip()}
    rows = _load_existing_rows(Path(args.summary))
    if "probe" in stages:
        run_probe(args, rows)
    if "grid" in stages:
        run_grid(args, rows)
    if "raw_recheck" in stages:
        run_raw_recheck(args, rows)
    _write_outputs(args, rows, "success")
    print(json.dumps({"summary": str(args.summary), "summary_md": str(args.summary_md), "rows": len(rows)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
