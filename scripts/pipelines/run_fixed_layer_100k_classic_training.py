#!/usr/bin/env python3
"""Run classic MLP baselines for the fixed-layer 100k experiment.

This runner intentionally avoids TensorFlow because the current dante_env
TensorFlow install may be a namespace package without tf.random/tf.config.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


DANTE_PYTHON = "/mnt/ML_projects/conda_envs/dante_env/bin/python"
DATASETS = {
    "100k_raw": {
        "dataset": "data/priority_grid_fixed_layer_s1_0125_s2_0250_100k/dataset.parquet",
        "out_root": "runs/baselines_fixed_layer_s1_0125_s2_0250_100k_raw_v1",
    },
    "100k_relabel": {
        "dataset": "data/priority_grid_fixed_layer_s1_0125_s2_0250_100k_relabel_t1_k32_w40_huber_mean_iter1/dataset.parquet",
        "out_root": "runs/baselines_fixed_layer_s1_0125_s2_0250_100k_relabel_v1",
    },
}
SPLITS = ("iid", "radius", "beta_block", "angular_sector")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_command(
    *,
    dataset: Path,
    out_dir: Path,
    robot_config: Path,
    split: str,
    models: str,
    seed: int,
) -> list[str]:
    return [
        DANTE_PYTHON,
        "scripts/baselines/run_baselines.py",
        "--dataset",
        str(dataset),
        "--out-dir",
        str(out_dir),
        "--robot-config",
        str(robot_config),
        "--split",
        str(split),
        "--save-split-file",
        str(out_dir / "split.npz"),
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
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-config", default="configs/robot_rods_only_priority_grid_third_joint_first_v1.yaml")
    ap.add_argument("--seed", type=int, default=20260207)
    ap.add_argument("--models", default="mlp,mlp_large")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--summary", default="runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1/classic_100k_training_summary.json")
    ap.add_argument("--log-dir", default="runs/logs/fixed_layer_100k_classic_train_jobs")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "backend": "classic",
        "models": args.models,
        "seed": int(args.seed),
        "max_parallel": int(args.max_parallel),
        "tensorflow_skipped_reason": "current dante_env tensorflow has no __version__/tf.random",
        "runs": {},
    }
    if summary_path.exists():
        old = _read_json(summary_path)
        if isinstance(old, dict):
            summary.update(old)
            summary.setdefault("runs", {})

    pending: list[dict[str, Any]] = []
    for dataset_key, spec in DATASETS.items():
        dataset_path = Path(spec["dataset"])
        out_root = Path(spec["out_root"])
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        summary["runs"].setdefault(dataset_key, {})
        for split in SPLITS:
            out_dir = out_root / split
            metrics_path = out_dir / "all_metrics.json"
            if metrics_path.exists():
                summary["runs"][dataset_key][split] = {
                    "status": "skipped_existing",
                    "out_dir": str(out_dir),
                    "metrics": str(metrics_path),
                }
                _write_json(summary_path, summary)
                print(f"SKIP existing {dataset_key}/{split}: {metrics_path}", flush=True)
                continue

            cmd = build_command(
                dataset=dataset_path,
                out_dir=out_dir,
                robot_config=Path(args.robot_config),
                split=split,
                models=args.models,
                seed=int(args.seed),
            )
            pending.append(
                {
                    "dataset_key": dataset_key,
                    "split": split,
                    "cmd": cmd,
                    "out_dir": out_dir,
                    "metrics_path": metrics_path,
                    "log_path": log_dir / f"{dataset_key}_{split}.log",
                }
            )

    running: list[dict[str, Any]] = []
    failed_rc = 0
    max_parallel = max(1, int(args.max_parallel))

    def launch(job: dict[str, Any]) -> None:
        log_path = Path(job["log_path"])
        log_fh = log_path.open("w", encoding="utf-8")
        log_fh.write("RUN " + " ".join(job["cmd"]) + "\n")
        log_fh.flush()
        print(f"RUN {job['dataset_key']}/{job['split']} log={log_path}", flush=True)
        job["started_at"] = time.strftime("%F %T")
        job["t0"] = time.perf_counter()
        job["log_fh"] = log_fh
        job["proc"] = subprocess.Popen(job["cmd"], stdout=log_fh, stderr=subprocess.STDOUT)
        summary["runs"].setdefault(str(job["dataset_key"]), {})[str(job["split"])] = {
            "status": "running",
            "started_at": job["started_at"],
            "out_dir": str(job["out_dir"]),
            "metrics": None,
            "log": str(log_path),
        }
        _write_json(summary_path, summary)

    while pending or running:
        while pending and len(running) < max_parallel and failed_rc == 0:
            job = pending.pop(0)
            launch(job)
            running.append(job)

        time.sleep(5.0)
        still_running: list[dict[str, Any]] = []
        for job in running:
            proc = job["proc"]
            rc = proc.poll()
            if rc is None:
                still_running.append(job)
                continue

            job["log_fh"].close()
            elapsed_s = time.perf_counter() - float(job["t0"])
            metrics_path = Path(job["metrics_path"])
            status = "success" if int(rc) == 0 and metrics_path.exists() else "failed"
            print(f"DONE {job['dataset_key']}/{job['split']} status={status} rc={rc} elapsed_s={elapsed_s:.1f}", flush=True)
            summary["runs"].setdefault(str(job["dataset_key"]), {})[str(job["split"])] = {
                "status": status,
                "rc": int(rc),
                "elapsed_s": elapsed_s,
                "started_at": job.get("started_at"),
                "finished_at": time.strftime("%F %T"),
                "out_dir": str(job["out_dir"]),
                "metrics": str(metrics_path) if metrics_path.exists() else None,
                "log": str(job["log_path"]),
            }
            _write_json(summary_path, summary)
            if status != "success" and failed_rc == 0:
                failed_rc = int(rc) if int(rc) != 0 else 1

        running = still_running
        if failed_rc != 0:
            pending.clear()
            for job in running:
                job["proc"].terminate()
            for job in running:
                try:
                    job["proc"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    job["proc"].kill()
                job["log_fh"].close()
            return failed_rc

    print(json.dumps({"summary": str(summary_path)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
