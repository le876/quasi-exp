#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DANTE_PYTHON = "/mnt/ML_projects/conda_envs/dante_env/bin/python"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def layer_label_from_values(s1: float, s2: float) -> str:
    return f"s1_{int(round(float(s1) * 1000)):04d}_s2_{int(round(float(s2) * 1000)):04d}"


def _reset_ids(dataset: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = dataset.sort_values("source_sample_id", kind="mergesort").reset_index(drop=True)
    meta = meta.sort_values("source_sample_id", kind="mergesort").reset_index(drop=True)
    dataset["sample_id"] = np.arange(len(dataset), dtype=int)
    meta["sample_id"] = np.arange(len(meta), dtype=int)
    return dataset, meta


def export_fixed_layers(dataset: pd.DataFrame, meta: pd.DataFrame, out_root: Path) -> dict[str, Any]:
    if "sample_id" not in dataset.columns or "sample_id" not in meta.columns:
        raise ValueError("dataset and meta must contain sample_id")
    for col in ["s1", "s2"]:
        if col not in meta.columns:
            raise ValueError(f"meta must contain {col}")
    out_root.mkdir(parents=True, exist_ok=True)
    joined = dataset.merge(meta, on="sample_id", how="left", validate="one_to_one", suffixes=("", "__meta"))
    summaries: list[dict[str, Any]] = []
    for (s1, s2), block in joined.groupby(["s1", "s2"], sort=True):
        label = layer_label_from_values(float(s1), float(s2))
        layer_dir = out_root / f"layer_{label}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        block = block.copy()
        block["source_sample_id"] = block["sample_id"].astype(int)
        block["layer_label"] = label
        block["source_component"] = label
        block["layer_field_component_id"] = 0
        dataset_cols = [c for c in dataset.columns if c in block.columns] + ["source_sample_id"]
        meta_cols = [c for c in meta.columns if c in block.columns]
        for col in ["source_sample_id", "layer_label", "source_component", "layer_field_component_id"]:
            if col not in meta_cols:
                meta_cols.append(col)
        out_dataset = block[dataset_cols].copy()
        out_meta = block[meta_cols].copy()
        out_dataset, out_meta = _reset_ids(out_dataset, out_meta)
        out_dataset.to_parquet(layer_dir / "dataset.parquet", index=False)
        out_meta.to_parquet(layer_dir / "dataset_meta.parquet", index=False)
        summaries.append(
            {
                "layer_label": label,
                "s1": float(s1),
                "s2": float(s2),
                "rows": int(len(out_dataset)),
                "dataset": str(layer_dir / "dataset.parquet"),
                "meta": str(layer_dir / "dataset_meta.parquet"),
            }
        )
    summary = {"mode": "export_fixed_layers", "layer_count": int(len(summaries)), "layers": summaries}
    (out_root / "fixed_layers_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def _load_analysis_script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_diagnostics(dataset_path: Path, meta_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(dataset_path)
    meta = pd.read_parquet(meta_path)
    continuity_mod = _load_analysis_script("eval_branch_aware_continuity")
    clustering_mod = _load_analysis_script("eval_branch_clustering")
    oracle_mod = _load_analysis_script("eval_oracle_floor")
    branch_column = "layer_label" if "layer_label" in meta.columns else "source_component"
    payload = {
        "continuity": continuity_mod.evaluate_frames(dataset, meta),
        "branch_clustering": {k: v for k, v in clustering_mod.evaluate_frames(dataset, meta).items() if k != "per_ball"},
        "oracle_floor": oracle_mod.evaluate_frames(dataset, meta, branch_column=branch_column),
    }
    (out_dir / "diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return payload


def build_training_commands(
    *,
    dataset_path: Path,
    out_root: Path,
    robot_config: Path,
    splits: Iterable[str],
    models: Iterable[str],
    backend: str,
    seed: int,
    dry_run: bool,
    eval_splits: str = "val,test",
    params_file: Path | None = None,
) -> list[list[str]]:
    del dry_run
    model_arg = ",".join(str(m) for m in models)
    commands: list[list[str]] = []
    py = DANTE_PYTHON if Path(DANTE_PYTHON).exists() else sys.executable
    for split in splits:
        split = str(split)
        split_dir = out_root / split
        commands.append(
            [
                py,
                "scripts/baselines/run_baselines.py",
                "--dataset",
                str(dataset_path),
                "--out-dir",
                str(split_dir),
                "--robot-config",
                str(robot_config),
                "--split",
                split,
                "--save-split-file",
                str(split_dir / "split.npz"),
                "--models",
                model_arg,
                "--backend",
                str(backend),
                "--seed",
                str(int(seed)),
                "--eval-splits",
                str(eval_splits),
                "--summary-split",
                "test",
                "--feature-set",
                "poly_heavy",
            ]
        )
        if params_file is not None:
            commands[-1].extend(["--params-file", str(params_file)])
    return commands


def run_training_commands(commands: list[list[str]], *, dry_run: bool) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for cmd in commands:
        if dry_run:
            results.append({"cmd": cmd, "returncode": None, "dry_run": True})
            continue
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)  # noqa: S603
        results.append({"cmd": cmd, "returncode": int(completed.returncode), "dry_run": False})
        if completed.returncode != 0:
            raise RuntimeError(f"training command failed with exit {completed.returncode}: {' '.join(cmd)}")
    return results


def _metric_pick(diagnostics: dict[str, Any]) -> dict[str, Any]:
    cont = diagnostics.get("continuity", {}).get("groups", {})
    cl = diagnostics.get("branch_clustering", {})
    oracle = diagnostics.get("oracle_floor", {})
    all10 = cont.get("all_xyz_<=10mm", {})
    beta_close = cont.get("xyz_<=10mm_beta_close", {})
    return {
        "all10_theta_p95_deg": all10.get("theta_rms_deg_p95"),
        "all10_tension_p95_n": all10.get("tension_mae_n_p95"),
        "beta_close_tension_p95_n": beta_close.get("tension_mae_n_p95"),
        "multi_branch_ball_ratio": cl.get("multi_branch_ball_ratio"),
        "within_branch_tension_p95_n": cl.get("within_branch_tension_mae_n_p95"),
        "between_branch_tension_p95_n": cl.get("between_branch_tension_mae_n_p95"),
        "xyz_nn_tension_mae_n": oracle.get("xyz_nn_oracle", {}).get("tension_mae_n"),
        "branch_aware_xyz_nn_tension_mae_n": oracle.get("branch_aware_xyz_nn_oracle", {}).get("tension_mae_n"),
        "beta_nn_tension_mae_n": oracle.get("beta_nn_oracle", {}).get("tension_mae_n"),
    }


def write_summary_markdown(summary: dict[str, Any], out_path: Path) -> None:
    def fmt(value: Any, digits: int) -> str:
        if value is None:
            return "nan"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "nan"

    lines = [
        "# Priority Grid Layer Experiments",
        "",
        "## Datasets",
        "",
        "| dataset | rows | all10 theta p95 deg | all10 T p95 N | multi-branch ratio | xyz-NN T MAE N |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in summary.get("diagnostics", {}).items():
        metrics = row.get("metrics", {})
        lines.append(
            f"| {name} | {row.get('rows', 0)} | "
            f"{fmt(metrics.get('all10_theta_p95_deg'), 3)} | "
            f"{fmt(metrics.get('all10_tension_p95_n'), 2)} | "
            f"{fmt(metrics.get('multi_branch_ball_ratio'), 4)} | "
            f"{fmt(metrics.get('xyz_nn_tension_mae_n'), 2)} |"
        )
    lines.extend(["", "## Training Commands", ""])
    for result in summary.get("training", []):
        lines.append("- `" + " ".join(str(v) for v in result.get("cmd", [])) + "`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("data/priority_grid_third_joint_first_v1/dataset.parquet"))
    ap.add_argument("--meta", type=Path, default=Path("data/priority_grid_third_joint_first_v1/dataset_meta.parquet"))
    ap.add_argument("--layer-field-dir", type=Path, default=Path("data/priority_grid_third_joint_first_v1_layer_field_v1"))
    ap.add_argument("--fixed-layers-dir", type=Path, default=Path("data/priority_grid_third_joint_first_v1_layers"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/diagnostics/priority_grid_layer_experiments_v1"))
    ap.add_argument("--robot-config", type=Path, default=Path("configs/robot_rods_only_priority_grid_third_joint_first_v1.yaml"))
    ap.add_argument("--splits", default="iid,radius,angular_sector")
    ap.add_argument("--models", default="mlp,tf_mlp")
    ap.add_argument("--backend", default="both", choices=["classic", "tf", "both"])
    ap.add_argument("--params-file", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=20260207)
    ap.add_argument("--skip-layer-field", action="store_true")
    ap.add_argument("--skip-fixed-layers", action="store_true")
    ap.add_argument("--skip-diagnostics", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(args.dataset)
    meta = pd.read_parquet(args.meta)
    summary: dict[str, Any] = {
        "dataset": str(args.dataset),
        "meta": str(args.meta),
        "layer_field_dir": str(args.layer_field_dir),
        "fixed_layers_dir": str(args.fixed_layers_dir),
        "diagnostics": {},
        "training": [],
    }

    if not args.skip_layer_field:
        selector = _load_analysis_script("select_priority_grid_layer_field")
        payload = selector.select_dataset(args.dataset, args.meta, args.layer_field_dir)
        summary["layer_field_selection"] = payload["summary"]

    if not args.skip_fixed_layers:
        summary["fixed_layers"] = export_fixed_layers(dataset, meta, args.fixed_layers_dir)

    if not args.skip_diagnostics:
        datasets = {
            "priority_grid_original": (args.dataset, args.meta),
            "priority_grid_layer_field_v1": (args.layer_field_dir / "dataset.parquet", args.layer_field_dir / "dataset_meta.parquet"),
        }
        if args.fixed_layers_dir.exists():
            for layer_dir in sorted(args.fixed_layers_dir.glob("layer_*")):
                if (layer_dir / "dataset.parquet").exists():
                    datasets[layer_dir.name] = (layer_dir / "dataset.parquet", layer_dir / "dataset_meta.parquet")
        for name, (ds_path, meta_path) in datasets.items():
            if not ds_path.exists() or not meta_path.exists():
                continue
            diagnostics = run_diagnostics(ds_path, meta_path, args.runs_dir / name)
            summary["diagnostics"][name] = {
                "rows": int(len(pd.read_parquet(ds_path, columns=["sample_id"]))),
                "dataset": str(ds_path),
                "meta": str(meta_path),
                "metrics": _metric_pick(diagnostics),
            }

    split_names = tuple(s.strip() for s in str(args.splits).split(",") if s.strip())
    model_names = tuple(m.strip() for m in str(args.models).split(",") if m.strip())
    train_targets = {
        "priority_grid_original": args.dataset,
        "priority_grid_layer_field_v1": args.layer_field_dir / "dataset.parquet",
    }
    if args.fixed_layers_dir.exists():
        for layer_dir in sorted(args.fixed_layers_dir.glob("layer_*")):
            if (layer_dir / "dataset.parquet").exists():
                train_targets[layer_dir.name] = layer_dir / "dataset.parquet"
    for name, ds_path in train_targets.items():
        if not ds_path.exists():
            continue
        commands = build_training_commands(
            dataset_path=ds_path,
            out_root=Path("runs") / f"baselines_{name}",
            robot_config=args.robot_config,
            splits=split_names,
            models=model_names,
            backend=args.backend,
            seed=int(args.seed),
            dry_run=bool(args.dry_run or not args.train),
            params_file=args.params_file,
        )
        if args.train:
            summary["training"].extend(run_training_commands(commands, dry_run=bool(args.dry_run)))
        else:
            summary["training"].extend({"cmd": cmd, "returncode": None, "dry_run": True} for cmd in commands)

    (args.runs_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    write_summary_markdown(summary, args.runs_dir / "summary.md")
    print(json.dumps({"summary_path": str(args.runs_dir / "summary.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
