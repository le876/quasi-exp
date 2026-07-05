#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pipelines"))

from generate_fixed_layer_manifold_dataset import layer_label  # noqa: E402
from run_phase5_fixed_layer_experiments import dataset_quality  # noqa: E402


DANTE_PYTHON = "/mnt/ML_projects/conda_envs/dante_env/bin/python"
DEFAULT_CONFIG = Path("configs/robot_rods_only_priority_grid_third_joint_first_v1.yaml")
DEFAULT_DOC = Path("docs/FixedLayer20k100k扩展实验记录.md")
DEFAULT_RUNS_DIR = Path("runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1")
MODEL_KEYS = ("mlp", "mlp_large", "tf_mlp", "tf_mlp_large")
SPLITS = ("iid", "radius", "beta_block", "angular_sector")
BASELINE_MIXED_T_MAE_N = 50.94


class ScalePaths(NamedTuple):
    label: str
    raw_dir: Path
    relabel_dir: Path


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


def _sample_tag(samples: int) -> str:
    n = int(samples)
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def scale_paths(
    *,
    data_root: Path,
    s1: float,
    s2: float,
    samples: int,
    anchor_k: int = 32,
    w_anchor: float = 40.0,
    anchor_stat: str = "huber_mean",
) -> ScalePaths:
    label = layer_label(float(s1), float(s2))
    raw_dir = Path(data_root) / f"priority_grid_fixed_layer_{label}_{_sample_tag(int(samples))}"
    relabel_dir = Path(data_root) / (
        f"priority_grid_fixed_layer_{label}_{_sample_tag(int(samples))}"
        f"_relabel_t1_k{int(anchor_k)}_w{int(float(w_anchor))}_{anchor_stat}_iter1"
    )
    return ScalePaths(label=label, raw_dir=raw_dir, relabel_dir=relabel_dir)


def _metric(gate: dict[str, Any], key: str, default: float = float("inf")) -> float:
    try:
        value = float(gate.get(key, default))
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _rows(payload: dict[str, Any]) -> int:
    return int(payload.get("rows", payload.get("gate", {}).get("rows", 0)) or 0)


def raw_quality_passes(payload: dict[str, Any]) -> bool:
    gate = dict(payload.get("gate", {}))
    return bool(
        _rows(payload) >= 19900
        and gate.get("hard_gate_passed", False)
        and _metric(gate, "rms_rnorm_q95") <= 0.06
        and _metric(gate, "max_tension_n") <= 2000.0
        and _metric(gate, "all10_theta_p95_deg") <= 0.5
        and _metric(gate, "multi_branch_ball_ratio") <= 0.02
        and _metric(gate, "all10_tension_p95_n") <= 80.0
        and _metric(gate, "beta_close_tension_p95_n") <= 80.0
        and _metric(gate, "xyz_nn_tension_mae_n") <= 25.0
        and _metric(gate, "beta_nn_tension_mae_n") <= 25.0
    )


def relabel_quality_passes(raw: dict[str, Any], relabel: dict[str, Any]) -> bool:
    gate = dict(relabel.get("gate", {}))
    return bool(
        _rows(relabel) == _rows(raw)
        and _rows(relabel) >= 19900
        and gate.get("hard_gate_passed", False)
        and _metric(gate, "rms_rnorm_q95") <= 0.06
        and _metric(gate, "max_tension_n") <= 2000.0
        and _metric(gate, "all10_theta_p95_deg") <= 0.5
        and _metric(gate, "multi_branch_ball_ratio") <= 0.02
        and _metric(gate, "all10_tension_p95_n") <= 50.0
        and _metric(gate, "beta_close_tension_p95_n") <= 50.0
        and _metric(gate, "same_beta_tension_p95_n") <= 90.0
        and _metric(gate, "xyz_nn_tension_mae_n") <= 20.0
        and _metric(gate, "beta_nn_tension_mae_n") <= 20.0
    )


def quality_promotes_to_100k(raw: dict[str, Any], relabel: dict[str, Any]) -> bool:
    return raw_quality_passes(raw) and relabel_quality_passes(raw, relabel)


def assert_ready_for_generation(out_dir: Path, *, report_name: str, force: bool) -> None:
    out_dir = Path(out_dir)
    report = out_dir / report_name
    if report.exists():
        return
    if not out_dir.exists():
        return
    if force:
        return
    if any(out_dir.iterdir()):
        raise RuntimeError(f"incomplete output directory exists without {report_name}: {out_dir}")


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> int:
    print("RUN", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(cmd)}")
    return int(completed.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_generate_command(*, config: Path, out_dir: Path, s1: float, s2: float, samples: int, seed: int) -> list[str]:
    return [
        DANTE_PYTHON,
        "scripts/generate_fixed_layer_manifold_dataset.py",
        "--config",
        str(config),
        "--out-dir",
        str(out_dir),
        "--s1",
        str(float(s1)),
        "--s2",
        str(float(s2)),
        "--num-samples",
        str(int(samples)),
        "--seed",
        str(int(seed)),
    ]


def build_relabel_command(
    *,
    config: Path,
    raw_dir: Path,
    out_dir: Path,
    anchor_k: int,
    w_anchor: float,
    anchor_stat: str,
    workers: int,
    accept_infeasible: bool = False,
) -> list[str]:
    cmd = [
        DANTE_PYTHON,
        "scripts/analysis/relabel_tension_graph_canonical.py",
        "--config",
        str(config),
        "--dataset",
        str(raw_dir / "dataset.parquet"),
        "--meta",
        str(raw_dir / "dataset_meta.parquet"),
        "--out-dir",
        str(out_dir),
        "--anchor-k",
        str(int(anchor_k)),
        "--w-anchor",
        str(float(w_anchor)),
        "--anchor-stat",
        str(anchor_stat),
        "--workers",
        str(int(workers)),
        "--distance-space",
        "effective_beta_xyz",
    ]
    if bool(accept_infeasible):
        cmd.append("--accept-infeasible")
    return cmd


def build_training_commands(
    *,
    dataset_path: Path,
    out_root: Path,
    robot_config: Path,
    splits: Iterable[str] = SPLITS,
    models: Iterable[str] = MODEL_KEYS,
    seed: int = 20260207,
) -> list[list[str]]:
    model_arg = ",".join(str(m) for m in models)
    commands: list[list[str]] = []
    for split in tuple(str(s) for s in splits):
        split_dir = Path(out_root) / split
        commands.append(
            [
                DANTE_PYTHON,
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
                "both",
                "--tf-device",
                "gpu",
                "--require-gpu",
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
        )
    return commands


def load_or_compute_quality(data_dir: Path, *, out_dir: Path, sample_rows: int = 0, seed: int = 20260614) -> dict[str, Any]:
    quality_path = out_dir / "quality_summary.json"
    if quality_path.exists():
        return _read_json(quality_path)

    dataset_path = Path(data_dir) / "dataset.parquet"
    meta_path = Path(data_dir) / "dataset_meta.parquet"
    if not dataset_path.exists() or not meta_path.exists():
        raise RuntimeError(f"missing dataset/meta for quality: {data_dir}")

    if int(sample_rows) > 0:
        import pandas as pd

        dataset = pd.read_parquet(dataset_path)
        meta = pd.read_parquet(meta_path)
        if len(dataset) > int(sample_rows):
            pick = np.sort(np.random.default_rng(int(seed)).choice(np.arange(len(dataset)), size=int(sample_rows), replace=False))
            dataset = dataset.iloc[pick].reset_index(drop=True).copy()
            meta = meta.iloc[pick].reset_index(drop=True).copy()
            dataset["sample_id"] = np.arange(len(dataset), dtype=int)
            meta["sample_id"] = np.arange(len(meta), dtype=int)
            tmp = out_dir / "_quality_sample"
            tmp.mkdir(parents=True, exist_ok=True)
            dataset.to_parquet(tmp / "dataset.parquet", index=False)
            meta.to_parquet(tmp / "dataset_meta.parquet", index=False)
            payload = dataset_quality(tmp / "dataset.parquet", tmp / "dataset_meta.parquet")
            payload["diagnostic_sample_rows"] = int(sample_rows)
            payload["source_rows"] = int(len(pd.read_parquet(dataset_path, columns=["sample_id"])))
        else:
            payload = dataset_quality(dataset_path, meta_path)
    else:
        payload = dataset_quality(dataset_path, meta_path)

    _write_json(quality_path, payload)
    return payload


def _metrics_from_all_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    if not payload:
        return None
    best_key = None
    best_metrics = None
    for key, value in payload.items():
        metrics = value.get("metrics", value) if isinstance(value, dict) else {}
        test = metrics.get("test", metrics) if isinstance(metrics, dict) else {}
        if "tension_mae_n" not in test:
            continue
        if best_metrics is None or float(test["tension_mae_n"]) < float(best_metrics["tension_mae_n"]):
            best_key = key
            best_metrics = test
    if best_metrics is None:
        return None
    return {"best_model": best_key, "best_test_metrics": best_metrics}


def collect_training_result(*, dataset_key: str, split: str, out_dir: Path, returncode: int | None) -> dict[str, Any]:
    metrics = _metrics_from_all_metrics(Path(out_dir) / "all_metrics.json")
    row: dict[str, Any] = {
        "dataset_key": dataset_key,
        "split": split,
        "out_dir": str(out_dir),
        "returncode": returncode,
        "status": "ok" if metrics is not None else "missing",
    }
    if metrics is not None:
        row.update(metrics)
    return row


def run_training_set(
    *,
    dataset_key: str,
    dataset_path: Path,
    out_root: Path,
    robot_config: Path,
    seed: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for cmd in build_training_commands(dataset_path=dataset_path, out_root=out_root, robot_config=robot_config, seed=seed):
        split = cmd[cmd.index("--split") + 1]
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        if dry_run:
            results.append({"dataset_key": dataset_key, "split": split, "out_dir": str(out_dir), "returncode": None, "status": "dry_run", "cmd": cmd})
            continue
        if (out_dir / "all_metrics.json").exists():
            results.append(collect_training_result(dataset_key=dataset_key, split=split, out_dir=out_dir, returncode=0))
            continue
        rc = _run(cmd)
        results.append(collect_training_result(dataset_key=dataset_key, split=split, out_dir=out_dir, returncode=rc))
    return results


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        if value is None:
            return "nan"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "nan"


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Fixed-Layer 20k/100k Scale Experiment",
        "",
        f"- layer: `{summary.get('layer_label', 's1_0125_s2_0250')}`",
        f"- mixed baseline avg best T MAE: `{BASELINE_MIXED_T_MAE_N:.2f} N`",
        "",
        "## Dataset Quality",
        "",
        "| dataset | rows | hard | theta p95 deg | T p95 N | same-beta T p95 N | multi-branch | xyz-NN T | beta-NN T |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in summary.get("datasets", {}).items():
        quality = row.get("quality", {})
        gate = quality.get("gate", {})
        lines.append(
            f"| {key} | {quality.get('rows', 0)} | {gate.get('hard_gate_passed', False)} | "
            f"{_fmt(gate.get('all10_theta_p95_deg'))} | {_fmt(gate.get('all10_tension_p95_n'))} | "
            f"{_fmt(gate.get('same_beta_tension_p95_n'))} | {_fmt(gate.get('multi_branch_ball_ratio'), 4)} | "
            f"{_fmt(gate.get('xyz_nn_tension_mae_n'))} | {_fmt(gate.get('beta_nn_tension_mae_n'))} |"
        )

    lines.extend(["", "## Promotion", ""])
    promotion = summary.get("promotion", {})
    lines.append(f"- promote_to_100k: `{promotion.get('promote_to_100k', False)}`")
    if promotion.get("reason"):
        lines.append(f"- reason: {promotion.get('reason')}")

    lines.extend(
        [
            "",
            "## Training",
            "",
            "| dataset | split | status | best model | theta MAE deg | T MAE N | T p95 N | EE p95 mm |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("training", []):
        metrics = row.get("best_test_metrics", {})
        lines.append(
            f"| {row.get('dataset_key')} | {row.get('split')} | {row.get('status')} | {row.get('best_model', '')} | "
            f"{_fmt(metrics.get('theta_mae_deg'))} | {_fmt(metrics.get('tension_mae_n'))} | "
            f"{_fmt(metrics.get('tension_p95_n'))} | {_fmt(metrics.get('ee_pos_p95_mm'))} |"
        )
    return "\n".join(lines) + "\n"


def current_progress_markdown() -> str:
    return """# Fixed-Layer 20k/100k 扩展实验记录

## 当前已知结论

- fixed-layer 单支路方法已经把 2k pilot 的 `multi_branch_ball_ratio` 压到 0。
- 当前最佳固定层是 `s1_0125_s2_0250`。
- 2k raw 指标：`all10_theta_p95_deg=0.099`，`all10_tension_p95_n=36.317N`，`xyz_nn_tension_mae_n=11.821N`。
- 2k relabel 最佳配置是 `k=32,w_anchor=40,anchor_stat=huber_mean`。
- 2k relabel 最佳指标：`all10_tension_p95_n=13.012N`，`same_beta_tension_p95_n=61.851N`。
- 张力断点审计显示，剩余张力跳变主要来自 friction case flip，active-set flip 比例为 0。

## 本轮计划

- 先生成 `s1_0125_s2_0250` 的 20k raw fixed-layer 数据。
- 对 20k raw 做 `k=32,w=40,huber_mean` graph-anchored relabel。
- raw 和 relabel 都跑质量诊断，并训练 `mlp,mlp_large,tf_mlp,tf_mlp_large`。
- 只要 20k relabel 质量门槛通过，就生成 100k raw + relabel，并继续做同口径诊断与训练。

## 输出位置

- 运行汇总：`runs/diagnostics/fixed_layer_scale_s1_0125_s2_0250_v1/summary.md`
- 20k raw：`data/priority_grid_fixed_layer_s1_0125_s2_0250_20k`
- 20k relabel：`data/priority_grid_fixed_layer_s1_0125_s2_0250_20k_relabel_t1_k32_w40_huber_mean_iter1`
- 100k raw：`data/priority_grid_fixed_layer_s1_0125_s2_0250_100k`
- 100k relabel：`data/priority_grid_fixed_layer_s1_0125_s2_0250_100k_relabel_t1_k32_w40_huber_mean_iter1`
"""


def append_run_result_to_doc(doc_path: Path, summary: dict[str, Any]) -> None:
    base = current_progress_markdown()
    if Path(doc_path).exists():
        base = Path(doc_path).read_text(encoding="utf-8").split("\n## 最近运行结果\n")[0].rstrip() + "\n"
    text = base + "\n## 最近运行结果\n\n" + summary_markdown(summary)
    _write_text(Path(doc_path), text)


def _init_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": "fixed_layer_scale_experiment",
        "started_at": time.strftime("%F %T"),
        "config": str(args.config),
        "layer_label": layer_label(float(args.s1), float(args.s2)),
        "s1": float(args.s1),
        "s2": float(args.s2),
        "anchor": {"k": int(args.anchor_k), "w": float(args.w_anchor), "stat": str(args.anchor_stat)},
        "relabel": {"accept_infeasible": bool(args.accept_infeasible_relabel)},
        "datasets": {},
        "training": [],
        "promotion": {"promote_to_100k": False, "reason": "not_evaluated"},
    }


def _load_summary(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    return _read_json(path) if path.exists() else _init_summary(args)


def _save_summary(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    summary["updated_at"] = time.strftime("%F %T")
    _write_json(args.runs_dir / "summary.json", summary)
    _write_text(args.runs_dir / "summary.md", summary_markdown(summary))
    append_run_result_to_doc(args.doc_path, summary)


def _register_quality(summary: dict[str, Any], key: str, data_dir: Path, quality: dict[str, Any]) -> None:
    summary.setdefault("datasets", {})[key] = {"path": str(data_dir), "quality": quality}


def generate_raw(args: argparse.Namespace, summary: dict[str, Any], *, samples: int) -> Path:
    paths = scale_paths(data_root=args.data_root, s1=args.s1, s2=args.s2, samples=samples, anchor_k=args.anchor_k, w_anchor=args.w_anchor, anchor_stat=args.anchor_stat)
    assert_ready_for_generation(paths.raw_dir, report_name="fixed_layer_generation_report.json", force=bool(args.force))
    if not (paths.raw_dir / "fixed_layer_generation_report.json").exists():
        _run(build_generate_command(config=args.config, out_dir=paths.raw_dir, s1=args.s1, s2=args.s2, samples=samples, seed=args.seed))
    summary.setdefault("datasets", {}).setdefault(f"{_sample_tag(samples)}_raw", {"path": str(paths.raw_dir)})
    return paths.raw_dir


def relabel_dataset(args: argparse.Namespace, summary: dict[str, Any], *, samples: int) -> Path:
    paths = scale_paths(data_root=args.data_root, s1=args.s1, s2=args.s2, samples=samples, anchor_k=args.anchor_k, w_anchor=args.w_anchor, anchor_stat=args.anchor_stat)
    if not (paths.raw_dir / "dataset.parquet").exists():
        raise RuntimeError(f"raw dataset missing for relabel: {paths.raw_dir}")
    assert_ready_for_generation(paths.relabel_dir, report_name="dataset_report.json", force=bool(args.force))
    if not (paths.relabel_dir / "dataset_report.json").exists():
        _run(
            build_relabel_command(
                config=args.config,
                raw_dir=paths.raw_dir,
                out_dir=paths.relabel_dir,
                anchor_k=args.anchor_k,
                w_anchor=args.w_anchor,
                anchor_stat=args.anchor_stat,
                workers=args.workers,
                accept_infeasible=bool(args.accept_infeasible_relabel),
            )
        )
    summary.setdefault("datasets", {}).setdefault(f"{_sample_tag(samples)}_relabel", {"path": str(paths.relabel_dir)})
    return paths.relabel_dir


def quality_pair(args: argparse.Namespace, summary: dict[str, Any], *, samples: int) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = scale_paths(data_root=args.data_root, s1=args.s1, s2=args.s2, samples=samples, anchor_k=args.anchor_k, w_anchor=args.w_anchor, anchor_stat=args.anchor_stat)
    tag = _sample_tag(samples)
    sample_rows = int(args.quality_sample_rows_100k) if int(samples) >= 100000 else 0
    raw_quality = load_or_compute_quality(paths.raw_dir, out_dir=args.runs_dir / f"{tag}_raw_quality", sample_rows=sample_rows, seed=args.seed)
    relabel_quality = load_or_compute_quality(paths.relabel_dir, out_dir=args.runs_dir / f"{tag}_relabel_quality", sample_rows=sample_rows, seed=args.seed)
    _register_quality(summary, f"{tag}_raw", paths.raw_dir, raw_quality)
    _register_quality(summary, f"{tag}_relabel", paths.relabel_dir, relabel_quality)
    if int(samples) == 20000:
        promote = quality_promotes_to_100k(raw_quality, relabel_quality)
        summary["promotion"] = {
            "promote_to_100k": bool(promote),
            "reason": "20k_quality_passed" if promote else "20k_quality_failed",
            "raw_passed": raw_quality_passes(raw_quality),
            "relabel_passed": relabel_quality_passes(raw_quality, relabel_quality),
        }
    return raw_quality, relabel_quality


def train_pair(args: argparse.Namespace, summary: dict[str, Any], *, samples: int) -> None:
    paths = scale_paths(data_root=args.data_root, s1=args.s1, s2=args.s2, samples=samples, anchor_k=args.anchor_k, w_anchor=args.w_anchor, anchor_stat=args.anchor_stat)
    tag = _sample_tag(samples)
    targets = {
        f"{tag}_raw": (paths.raw_dir / "dataset.parquet", args.runs_root / f"baselines_fixed_layer_{paths.label}_{tag}_raw_v1"),
        f"{tag}_relabel": (paths.relabel_dir / "dataset.parquet", args.runs_root / f"baselines_fixed_layer_{paths.label}_{tag}_relabel_v1"),
    }
    existing = {(row.get("dataset_key"), row.get("split")) for row in summary.get("training", []) if row.get("status") == "ok"}
    for key, (dataset_path, out_root) in targets.items():
        if not dataset_path.exists():
            raise RuntimeError(f"training dataset missing: {dataset_path}")
        rows = run_training_set(dataset_key=key, dataset_path=dataset_path, out_root=out_root, robot_config=args.config, seed=args.train_seed, dry_run=bool(args.dry_run_train))
        for row in rows:
            ident = (row.get("dataset_key"), row.get("split"))
            if ident in existing:
                continue
            summary.setdefault("training", []).append(row)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--s1", type=float, default=0.125)
    ap.add_argument("--s2", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20260614)
    ap.add_argument("--train-seed", type=int, default=20260207)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--anchor-k", type=int, default=32)
    ap.add_argument("--w-anchor", type=float, default=40.0)
    ap.add_argument("--anchor-stat", default="huber_mean")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--doc-path", type=Path, default=DEFAULT_DOC)
    ap.add_argument("--stages", default="record_current,generate20k,relabel20k,quality20k,train20k,maybe100k,relabel100k,quality100k,train100k")
    ap.add_argument("--quality-sample-rows-100k", type=int, default=0)
    ap.add_argument("--accept-infeasible-relabel", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run-train", action="store_true")
    return ap.parse_args()


def _stage_set(raw: str) -> set[str]:
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def main() -> int:
    args = parse_args()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.runs_dir / "summary.json"
    summary = _load_summary(summary_path, args)
    stages = _stage_set(args.stages)

    if "record_current" in stages:
        _write_text(args.doc_path, current_progress_markdown())
        _save_summary(args, summary)

    if "generate20k" in stages:
        generate_raw(args, summary, samples=20000)
        _save_summary(args, summary)
    if "relabel20k" in stages:
        relabel_dataset(args, summary, samples=20000)
        _save_summary(args, summary)
    if "quality20k" in stages:
        quality_pair(args, summary, samples=20000)
        _save_summary(args, summary)
    if "train20k" in stages:
        train_pair(args, summary, samples=20000)
        _save_summary(args, summary)

    if "maybe100k" in stages:
        if "20k_raw" not in summary.get("datasets", {}) or "20k_relabel" not in summary.get("datasets", {}):
            quality_pair(args, summary, samples=20000)
        promote = bool(summary.get("promotion", {}).get("promote_to_100k", False))
        if promote:
            generate_raw(args, summary, samples=100000)
        else:
            print("[fixed-layer-scale] 20k quality gate did not pass; skipping 100k generation", flush=True)
        _save_summary(args, summary)

    promote = bool(summary.get("promotion", {}).get("promote_to_100k", False))
    if promote and "relabel100k" in stages:
        relabel_dataset(args, summary, samples=100000)
        _save_summary(args, summary)
    if promote and "quality100k" in stages:
        quality_pair(args, summary, samples=100000)
        _save_summary(args, summary)
    if promote and "train100k" in stages:
        train_pair(args, summary, samples=100000)
        _save_summary(args, summary)

    print(json.dumps({"summary": str(summary_path), "promotion": summary.get("promotion", {})}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
