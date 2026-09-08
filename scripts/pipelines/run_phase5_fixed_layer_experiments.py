#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_fixed_layer_manifold_dataset import LAYER_PAIRS, layer_label  # noqa: E402


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _inf(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("inf")
    return out if np.isfinite(out) else float("inf")


def _metric(gate: dict[str, Any], key: str) -> float:
    return _inf(gate.get(key, float("inf")))


def phase5_score(summary: dict[str, Any]) -> float:
    gate = dict(summary.get("gate", {}))
    score = (
        _metric(gate, "all10_tension_p95_n") / 110.0
        + _metric(gate, "beta_close_tension_p95_n") / 90.0
        + _metric(gate, "beta_nn_tension_mae_n") / 20.0
        + _metric(gate, "multi_branch_ball_ratio") / 0.15
        + _metric(gate, "all10_theta_p95_deg") / 1.5
    )
    if not bool(gate.get("hard_gate_passed", False)):
        score += 100.0
    return float(score)


def rank_fixed_layers(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in summaries:
        row = dict(item)
        row["phase5_score"] = phase5_score(row)
        out.append(row)
    return sorted(out, key=lambda row: (float(row["phase5_score"]), str(row.get("layer_label", ""))))


def fixed_layer_passes_initial(summary: dict[str, Any]) -> bool:
    gate = dict(summary.get("gate", {}))
    return bool(
        gate.get("hard_gate_passed", False)
        and _metric(gate, "all10_theta_p95_deg") <= 1.5
        and _metric(gate, "all10_tension_p95_n") <= 110.0
        and _metric(gate, "beta_close_tension_p95_n") <= 90.0
        and _metric(gate, "multi_branch_ball_ratio") <= 0.15
        and _metric(gate, "xyz_nn_tension_mae_n") <= 35.0
        and _metric(gate, "beta_nn_tension_mae_n") <= 20.0
    )


def relabel_passes_silver(summary: dict[str, Any]) -> bool:
    gate = dict(summary.get("gate", {}))
    return bool(
        gate.get("hard_gate_passed", False)
        and _metric(gate, "rms_rnorm_q95") <= 0.06
        and _metric(gate, "max_tension_n") <= 2000.0
        and _metric(gate, "all10_tension_p95_n") <= 90.0
        and _metric(gate, "beta_close_tension_p95_n") <= 70.0
        and _metric(gate, "same_beta_tension_p95_n") <= 60.0
        and _metric(gate, "xyz_nn_tension_mae_n") <= 28.0
        and _metric(gate, "beta_nn_tension_mae_n") <= 18.0
    )


def _run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_quality(dataset_path: Path, meta_path: Path) -> dict[str, Any]:
    continuity_mod = _load_script("eval_branch_aware_continuity")
    clustering_mod = _load_script("eval_branch_clustering")
    oracle_mod = _load_script("eval_oracle_floor")
    dataset = pd.read_parquet(dataset_path)
    meta = pd.read_parquet(meta_path)
    continuity = continuity_mod.evaluate_frames(dataset, meta)
    clustering = clustering_mod.evaluate_frames(dataset, meta)
    branch_col = "layer_label" if "layer_label" in meta.columns else None
    oracle = oracle_mod.evaluate_frames(dataset, meta, branch_column=branch_col)
    all10 = continuity.get("groups", {}).get("all_xyz_<=10mm", {})
    beta_close = continuity.get("groups", {}).get("xyz_<=10mm_beta_close", {})
    same_beta = continuity.get("groups", {}).get("same_beta_knn_k16", {})
    tension_cols = [f"tension_{i}_n" for i in range(1, 13)]
    tension = dataset[tension_cols].to_numpy(dtype=float)
    rms = meta["rms_rnorm"].to_numpy(dtype=float) if "rms_rnorm" in meta.columns else np.full(len(meta), np.inf)
    gate = {
        "hard_gate_passed": bool(np.isfinite(tension).all() and (tension >= 0.0).all() and (tension <= 2000.0).all() and np.percentile(rms, 95) <= 0.06),
        "rms_rnorm_q95": float(np.percentile(rms, 95)) if len(rms) else float("inf"),
        "max_tension_n": float(np.max(tension)) if tension.size else float("inf"),
        "all10_theta_p95_deg": _inf(all10.get("theta_rms_deg_p95", float("inf"))),
        "all10_tension_p95_n": _inf(all10.get("tension_mae_n_p95", float("inf"))),
        "beta_close_tension_p95_n": _inf(beta_close.get("tension_mae_n_p95", float("inf"))),
        "same_beta_tension_p95_n": _inf(same_beta.get("tension_mae_n_p95", float("inf"))),
        "multi_branch_ball_ratio": _inf(clustering.get("multi_branch_ball_ratio", float("inf"))),
        "xyz_nn_tension_mae_n": _inf(oracle.get("xyz_nn_oracle", {}).get("tension_mae_n", float("inf"))),
        "beta_nn_tension_mae_n": _inf(oracle.get("beta_nn_oracle", {}).get("tension_mae_n", float("inf"))),
    }
    return {
        "rows": int(len(dataset)),
        "gate": gate,
        "continuity": continuity,
        "branch_clustering": {k: v for k, v in clustering.items() if k != "per_ball"},
        "oracle_floor": oracle,
    }


def _write_quality(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _layer_dir(data_root: Path, label: str, samples: int) -> Path:
    return data_root / f"priority_grid_fixed_layer_{label}_{samples // 1000 if samples >= 1000 else samples}{'k' if samples >= 1000 else ''}"


def run_fixed2k(args: argparse.Namespace) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for s1, s2 in LAYER_PAIRS:
        label = layer_label(s1, s2)
        out_dir = _layer_dir(args.data_root, label, int(args.num_samples_2k))
        report_path = out_dir / "fixed_layer_generation_report.json"
        if bool(args.force) or not report_path.exists():
            _run(
                [
                    sys.executable,
                    "scripts/generate_fixed_layer_manifold_dataset.py",
                    "--config",
                    str(args.config),
                    "--out-dir",
                    str(out_dir),
                    "--s1",
                    str(s1),
                    "--s2",
                    str(s2),
                    "--num-samples",
                    str(args.num_samples_2k),
                    "--seed",
                    str(args.seed),
                ]
            )
        report = _read_report(report_path)
        quality = dataset_quality(out_dir / "dataset.parquet", out_dir / "dataset_meta.parquet")
        _write_quality(out_dir / "diagnostics", quality)
        summaries.append({"layer_label": label, "s1": float(s1), "s2": float(s2), "data_dir": str(out_dir), "gate": quality["gate"], "report": report})
    ranked = rank_fixed_layers(summaries)
    out = args.runs_root / "fixed_layer_2k_sweep_v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"ranked": ranked}, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (out / "summary.md").write_text(_fixed_summary_md(ranked), encoding="utf-8")
    return ranked


def _fixed_summary_md(ranked: list[dict[str, Any]]) -> str:
    lines = [
        "# Fixed-Layer 2k Sweep",
        "",
        "| rank | layer | score | pass | all10 theta deg | all10 T N | beta-close T N | multi-branch | xyz-NN T | beta-NN T |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(ranked, 1):
        gate = row["gate"]
        lines.append(
            f"| {idx} | {row['layer_label']} | {row['phase5_score']:.3f} | {fixed_layer_passes_initial(row)} | "
            f"{_metric(gate, 'all10_theta_p95_deg'):.3f} | {_metric(gate, 'all10_tension_p95_n'):.3f} | "
            f"{_metric(gate, 'beta_close_tension_p95_n'):.3f} | {_metric(gate, 'multi_branch_ball_ratio'):.3f} | "
            f"{_metric(gate, 'xyz_nn_tension_mae_n'):.3f} | {_metric(gate, 'beta_nn_tension_mae_n'):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _load_fixed_ranked(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.runs_root / "fixed_layer_2k_sweep_v1" / "summary.json"
    if not path.exists():
        return run_fixed2k(args)
    return json.loads(path.read_text(encoding="utf-8"))["ranked"]


def select_top_layers(ranked: list[dict[str, Any]], *, max_layers: int = 3) -> list[dict[str, Any]]:
    passing = [row for row in ranked if fixed_layer_passes_initial(row)]
    return (passing or ranked)[: int(max_layers)]


def run_audit(args: argparse.Namespace) -> list[dict[str, Any]]:
    ranked = _load_fixed_ranked(args)
    selected = select_top_layers(ranked, max_layers=int(args.max_layers))
    audit_mod = _load_script("audit_tension_discontinuity_sources")
    outputs: list[dict[str, Any]] = []
    root = args.runs_root / "tension_discontinuity_audit_fixed_layer_v1"
    root.mkdir(parents=True, exist_ok=True)
    for row in selected:
        data_dir = Path(row["data_dir"])
        payload = audit_mod.audit(data_dir / "dataset.parquet", data_dir / "dataset_meta.parquet")
        layer_out = root / row["layer_label"]
        layer_out.mkdir(parents=True, exist_ok=True)
        (layer_out / "audit_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        (layer_out / "REPORT.md").write_text(audit_mod._markdown(payload), encoding="utf-8")
        outputs.append({"layer_label": row["layer_label"], "data_dir": row["data_dir"], "audit": payload})
    (root / "summary.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return outputs


def _parse_ints(raw: str) -> list[int]:
    return [int(v.strip()) for v in str(raw).split(",") if v.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(v.strip()) for v in str(raw).split(",") if v.strip()]


def _parse_strings(raw: str) -> list[str]:
    return [str(v.strip()) for v in str(raw).split(",") if v.strip()]


def run_relabel(args: argparse.Namespace) -> list[dict[str, Any]]:
    ranked = _load_fixed_ranked(args)
    selected = select_top_layers(ranked, max_layers=int(args.max_layers))
    outputs: list[dict[str, Any]] = []
    root = args.runs_root / "tension_relabel_fixed_layer_ablation_v1"
    root.mkdir(parents=True, exist_ok=True)
    for row in selected:
        data_dir = Path(row["data_dir"])
        for k in _parse_ints(args.anchor_k_values):
            for w in _parse_floats(args.w_anchor_values):
                for stat in _parse_strings(args.anchor_stats):
                    out_dir = args.data_root / f"{data_dir.name}_relabel_t1_k{k}_w{int(w)}_{stat}_iter1"
                    report_path = out_dir / "dataset_report.json"
                    if bool(args.force) or not report_path.exists():
                        _run(
                            [
                                sys.executable,
                                "scripts/analysis/relabel_tension_graph_canonical.py",
                                "--config",
                                str(args.config),
                                "--dataset",
                                str(data_dir / "dataset.parquet"),
                                "--meta",
                                str(data_dir / "dataset_meta.parquet"),
                                "--out-dir",
                                str(out_dir),
                                "--anchor-k",
                                str(k),
                                "--w-anchor",
                                str(w),
                                "--anchor-stat",
                                stat,
                                "--workers",
                                str(args.workers),
                                "--distance-space",
                                "effective_beta_xyz",
                            ]
                        )
                    quality = dataset_quality(out_dir / "dataset.parquet", out_dir / "dataset_meta.parquet")
                    _write_quality(out_dir / "diagnostics", quality)
                    outputs.append(
                        {
                            "layer_label": row["layer_label"],
                            "source_data_dir": str(data_dir),
                            "data_dir": str(out_dir),
                            "anchor_k": int(k),
                            "w_anchor": float(w),
                            "anchor_stat": stat,
                            "gate": quality["gate"],
                            "silver_passed": relabel_passes_silver({"gate": quality["gate"]}),
                        }
                    )
    outputs = sorted(outputs, key=lambda item: (not bool(item["silver_passed"]), phase5_score(item), str(item["data_dir"])))
    (root / "summary.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    (root / "summary.md").write_text(_relabel_summary_md(outputs), encoding="utf-8")
    return outputs


def _relabel_summary_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Tension Relabel Fixed-Layer Ablation",
        "",
        "| rank | layer | k | w | stat | silver | all10 T N | beta-close T N | same-beta T N | xyz-NN T | beta-NN T |",
        "| ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(rows, 1):
        gate = row["gate"]
        lines.append(
            f"| {idx} | {row['layer_label']} | {row['anchor_k']} | {row['w_anchor']:.1f} | {row['anchor_stat']} | {row['silver_passed']} | "
            f"{_metric(gate, 'all10_tension_p95_n'):.3f} | {_metric(gate, 'beta_close_tension_p95_n'):.3f} | "
            f"{_metric(gate, 'same_beta_tension_p95_n'):.3f} | {_metric(gate, 'xyz_nn_tension_mae_n'):.3f} | "
            f"{_metric(gate, 'beta_nn_tension_mae_n'):.3f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/robot_rods_only_priority_grid_third_joint_first_v1.yaml"))
    ap.add_argument("--stage", default="fixed2k,audit,relabel")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--runs-root", type=Path, default=Path("runs/diagnostics"))
    ap.add_argument("--num-samples-2k", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260614)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-layers", type=int, default=3)
    ap.add_argument("--anchor-k-values", default="16,32")
    ap.add_argument("--w-anchor-values", default="20,40")
    ap.add_argument("--anchor-stats", default="median,huber_mean")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    stages = set(_parse_strings(args.stage))
    t0 = time.time()
    if "fixed2k" in stages:
        run_fixed2k(args)
    if "audit" in stages:
        run_audit(args)
    if "relabel" in stages:
        run_relabel(args)
    print(json.dumps({"elapsed_s": time.time() - t0, "stages": sorted(stages)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
