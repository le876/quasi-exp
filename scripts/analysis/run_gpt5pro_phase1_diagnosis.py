#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data/mixed_beta_20k_distal_preferred_segmented_canonical/dataset.parquet"
DEFAULT_META = REPO_ROOT / "data/mixed_beta_20k_distal_preferred_segmented_canonical/dataset_meta.parquet"
DEFAULT_ANCHOR_DATASET = (
    REPO_ROOT / "data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20/dataset.parquet"
)
DEFAULT_ANCHOR_META = (
    REPO_ROOT / "data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20/dataset_meta.parquet"
)
DEFAULT_STANDARD_DATASET = REPO_ROOT / "data/standard_beta_sweep_100k_segmented_canonical/dataset.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "runs/diagnostics/gpt5pro_plan_phase1_20k"
DEFAULT_SEED_CONFIG = REPO_ROOT / "configs/robot_rods_only_mixed_20k_distal_preferred_segmented_canonical.yaml"


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _numbered_cols(columns: list[str], prefix: str, suffix: str) -> list[str]:
    out: list[tuple[int, str]] = []
    for col in columns:
        if not (col.startswith(prefix) and col.endswith(suffix)):
            continue
        raw = col[len(prefix) : -len(suffix)]
        if raw.isdigit():
            out.append((int(raw), col))
    return [col for _, col in sorted(out)]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def _safe_q(values: np.ndarray, q: float, default: float = float("nan")) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, q))


def _quality_gates(dataset: pd.DataFrame, meta: pd.DataFrame) -> dict[str, Any]:
    tension_cols = _numbered_cols(list(dataset.columns), "tension_", "_n")
    tension = dataset[tension_cols].to_numpy(dtype=float) if len(tension_cols) == 12 else np.empty((len(dataset), 0))
    rms = meta["rms_rnorm"].to_numpy(dtype=float) if "rms_rnorm" in meta.columns else np.asarray([])
    max_tension_meta = meta["max_tension"].to_numpy(dtype=float) if "max_tension" in meta.columns else np.max(tension, axis=1)
    if "segmented_success" in meta.columns:
        segmented_success_all = bool(meta["segmented_success"].astype(bool).all())
    else:
        segmented_success_all = True
    checks = {
        "segmented_success_all": segmented_success_all,
        "rms_rnorm_q95_le_0_06": bool(_safe_q(rms, 95, 0.0) <= 0.06),
        "max_tension_le_2000n": bool(float(np.nanmax(max_tension_meta)) <= 2000.0),
        "saturation_ratio_eq_0": bool(float(np.mean(max_tension_meta >= 2000.0)) == 0.0),
        "tension_lt0_ratio_eq_0": bool(float(np.mean(tension < 0.0)) == 0.0) if tension.size else False,
        "tension_gt_tmax_ratio_eq_0": bool(float(np.mean(tension > 2000.0)) == 0.0) if tension.size else False,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "metrics": {
            "rows": int(len(dataset)),
            "rms_rnorm_q95": _safe_q(rms, 95, 0.0),
            "rms_rnorm_max": _safe_q(rms, 100, 0.0),
            "max_tension_n": float(np.nanmax(max_tension_meta)),
            "max_tension_q95_n": _safe_q(max_tension_meta, 95, 0.0),
            "saturation_ratio": float(np.mean(max_tension_meta >= 2000.0)),
            "tension_lt0_ratio": float(np.mean(tension < 0.0)) if tension.size else float("nan"),
            "tension_gt_tmax_ratio": float(np.mean(tension > 2000.0)) if tension.size else float("nan"),
        },
    }


def _extract_tension_p95(continuity: dict[str, Any], key: str) -> float:
    return float(continuity.get("groups", {}).get(key, {}).get("tension_mae_n_p95", float("inf")))


def _extract_theta_p95(continuity: dict[str, Any], key: str) -> float:
    return float(continuity.get("groups", {}).get(key, {}).get("theta_rms_deg_p95", float("inf")))


def _decide_bottleneck(
    *,
    quality: dict[str, Any],
    continuity: dict[str, Any],
    clustering: dict[str, Any],
    oracle: dict[str, Any],
    seed_stability: dict[str, Any] | None,
) -> dict[str, Any]:
    if seed_stability and not bool(seed_stability.get("skipped", False)) and not bool(seed_stability.get("gate_pass", False)):
        return {
            "primary_bottleneck": "allocator_bottleneck",
            "reason": "same-theta seed stability gate failed",
        }
    beta_close_p95 = _extract_tension_p95(continuity, "xyz_<=10mm_beta_close")
    same_component_p95 = _extract_tension_p95(continuity, "same_component_xyz_<=10mm")
    same_branch_p95 = min(beta_close_p95, same_component_p95)
    all_xyz_p95 = _extract_tension_p95(continuity, "all_xyz_<=10mm")
    all_theta_p95 = _extract_theta_p95(continuity, "all_xyz_<=10mm")
    between_ratio = float(clustering.get("between_branch_variance_ratio_p50", 0.0))
    xyz_nn = float(oracle.get("xyz_nn_oracle", {}).get("tension_mae_n", float("inf")))
    branch_nn = float(oracle.get("branch_aware_xyz_nn_oracle", {}).get("tension_mae_n", float("inf")))
    beta_nn = float(oracle.get("beta_nn_oracle", {}).get("tension_mae_n", float("inf")))

    if not bool(quality.get("passed", False)):
        return {"primary_bottleneck": "allocator_bottleneck", "reason": "data quality hard gate failed"}
    if same_branch_p95 <= 90.0 and (all_xyz_p95 > 120.0 or all_theta_p95 > 3.0) and (
        between_ratio > 0.6 or beta_nn < xyz_nn
    ):
        return {
            "primary_bottleneck": "workspace_multibranch_bottleneck",
            "reason": "beta-close/same-branch continuity is acceptable while all-workspace continuity is high",
        }
    if same_branch_p95 > 90.0 and between_ratio > 0.6 and branch_nn < xyz_nn:
        return {
            "primary_bottleneck": "mixed_bottleneck",
            "reason": "same-branch tension is still high and branch-aware oracle is better than xyz-only oracle",
        }
    if same_branch_p95 > 90.0:
        return {
            "primary_bottleneck": "allocator_bottleneck",
            "reason": "beta-close and same-component 10mm tension p95 are above silver threshold",
        }
    return {
        "primary_bottleneck": "labels_already_phase1_silver",
        "reason": "hard gate and same-branch continuity meet phase-1 silver thresholds",
    }


def _run_seed_stability(
    *,
    dataset_path: Path,
    config_path: Path,
    out_dir: Path,
    num_theta: int,
    seeds: str,
) -> dict[str, Any]:
    seed_dir = out_dir / "seed_stability"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/analysis/eval_pso_tension_seed_stability.py"),
        "--config",
        str(config_path),
        "--dataset",
        str(dataset_path),
        "--num-theta",
        str(num_theta),
        "--seeds",
        seeds,
        "--out-dir",
        str(seed_dir),
        "--median-threshold-n",
        "1.0",
        "--p95-threshold-n",
        "5.0",
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    payload_path = seed_dir / "summary.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8")) if payload_path.exists() else {}
    payload["command"] = cmd
    payload["returncode"] = int(proc.returncode)
    payload["stdout_tail"] = proc.stdout[-4000:]
    payload["stderr_tail"] = proc.stderr[-4000:]
    return payload


def _write_report(summary: dict[str, Any], out_dir: Path) -> None:
    quality = summary["quality_gates"]["metrics"]
    cont = summary["continuity_current"]["groups"]
    anchor_cont = summary["continuity_anchor"]["groups"]
    clustering = summary["branch_clustering"]
    oracle = summary["oracle_floor"]
    decision = summary["decision"]

    def fmt(value: Any, digits: int = 3) -> str:
        try:
            val = float(value)
            if not np.isfinite(val):
                return "nan"
            return f"{val:.{digits}f}"
        except Exception:
            return str(value)

    lines = [
        "# GPT-5 Pro Phase 1 20k Diagnosis",
        "",
        f"- primary_bottleneck: `{summary['primary_bottleneck']}`",
        f"- decision_reason: {decision['reason']}",
        f"- dataset: `{summary['dataset_path']}`",
        f"- anchor_dataset: `{summary['anchor_dataset_path']}`",
        "",
        "## Hard Gate",
        "",
        f"- passed: `{summary['quality_gates']['passed']}`",
        f"- rows: {quality['rows']}",
        f"- rms_rnorm q95: {fmt(quality['rms_rnorm_q95'], 6)}",
        f"- max tension: {fmt(quality['max_tension_n'], 2)} N",
        f"- saturation ratio: {fmt(quality['saturation_ratio'], 6)}",
        f"- tension <0 ratio: {fmt(quality['tension_lt0_ratio'], 6)}",
        f"- tension >2000 ratio: {fmt(quality['tension_gt_tmax_ratio'], 6)}",
        "",
        "## Local Continuity",
        "",
        "| dataset | group | pairs | T p50 N | T p90 N | T p95 N | theta p95 deg |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, groups in [("current", cont), ("anchor_v2_second_stage", anchor_cont)]:
        for key in ["same_component_xyz_<=10mm", "cross_component_xyz_<=10mm", "all_xyz_<=10mm", "xyz_<=10mm_beta_close", "xyz_<=10mm_beta_far"]:
            row = groups.get(key, {})
            lines.append(
                "| {label} | {key} | {pairs} | {p50} | {p90} | {p95} | {theta} |".format(
                    label=label,
                    key=key,
                    pairs=row.get("pairs", 0),
                    p50=fmt(row.get("tension_mae_n_p50", float("nan")), 2),
                    p90=fmt(row.get("tension_mae_n_p90", float("nan")), 2),
                    p95=fmt(row.get("tension_mae_n_p95", float("nan")), 2),
                    theta=fmt(row.get("theta_rms_deg_p95", float("nan")), 3),
                )
            )
    lines.extend(
        [
            "",
            "## Branch Clustering",
            "",
            f"- balls evaluated: {clustering.get('balls_evaluated', 0)}",
            f"- multi-branch ball ratio: {fmt(clustering.get('multi_branch_ball_ratio', 0.0), 3)}",
            f"- branch count p50/p90: {fmt(clustering.get('branch_count_p50', 0.0), 2)} / {fmt(clustering.get('branch_count_p90', 0.0), 2)}",
            f"- within-branch T p50/p95: {fmt(clustering.get('within_branch_tension_mae_n_p50', 0.0), 2)} / {fmt(clustering.get('within_branch_tension_mae_n_p95', 0.0), 2)} N",
            f"- between-branch T p50/p95: {fmt(clustering.get('between_branch_tension_mae_n_p50', 0.0), 2)} / {fmt(clustering.get('between_branch_tension_mae_n_p95', 0.0), 2)} N",
            f"- between-branch variance ratio p50/p95: {fmt(clustering.get('between_branch_variance_ratio_p50', 0.0), 3)} / {fmt(clustering.get('between_branch_variance_ratio_p95', 0.0), 3)}",
            "",
            "## Oracle Floor",
            "",
            "| oracle | T MAE N | T p95 N | theta MAE deg | EE p95 mm |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in ["xyz_nn_oracle", "branch_aware_xyz_nn_oracle", "beta_nn_oracle"]:
        row = oracle.get(key, {})
        lines.append(
            f"| {key} | {fmt(row.get('tension_mae_n', float('nan')), 2)} | {fmt(row.get('tension_p95_n', float('nan')), 2)} | {fmt(row.get('theta_mae_deg', float('nan')), 3)} | {fmt(row.get('ee_pos_p95_mm', float('nan')), 2)} |"
        )
    seed = summary.get("seed_stability")
    lines.extend(["", "## Seed Stability", ""])
    if seed is None or bool(seed.get("skipped", False)):
        lines.append("- skipped: true")
    else:
        lines.extend(
            [
                f"- gate_pass: `{seed.get('gate_pass')}`",
                f"- median pairwise T MAE: {fmt(seed.get('median_pairwise_mae_n', float('nan')), 2)} N",
                f"- p95 pairwise T MAE: {fmt(seed.get('p95_pairwise_mae_n', float('nan')), 2)} N",
                f"- feasible rate: {fmt(seed.get('feasible_rate', float('nan')), 3)}",
            ]
        )
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            "- If `workspace_multibranch_bottleneck`: prioritize branch-consistent sampling/modeling.",
            "- If `allocator_bottleneck`: prioritize integrated anchor / LS-QP allocator.",
            "- If `mixed_bottleneck`: run source ablation before 100k expansion.",
        ]
    )
    (out_dir / "diagnosis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnosis(
    *,
    dataset_path: Path,
    meta_path: Path,
    anchor_dataset_path: Path,
    anchor_meta_path: Path,
    standard_dataset_path: Path | None,
    out_dir: Path,
    skip_seed_stability: bool,
    skip_existing_fast4: bool,
    config_path: Path = DEFAULT_SEED_CONFIG,
    seed_num_theta: int = 20,
    seed_seeds: str = "1001,1002,1003,1004,1005,1006,1007,1008",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    continuity_mod = _load_script("eval_branch_aware_continuity")
    clustering_mod = _load_script("eval_branch_clustering")
    oracle_mod = _load_script("eval_oracle_floor")

    dataset = pd.read_parquet(dataset_path)
    meta = pd.read_parquet(meta_path)
    anchor_dataset = pd.read_parquet(anchor_dataset_path)
    anchor_meta = pd.read_parquet(anchor_meta_path)

    quality = _quality_gates(dataset, meta)
    continuity_current = continuity_mod.evaluate_frames(dataset, meta)
    continuity_anchor = continuity_mod.evaluate_frames(anchor_dataset, anchor_meta)
    clustering = clustering_mod.evaluate_frames(anchor_dataset, anchor_meta)
    oracle = oracle_mod.evaluate_frames(anchor_dataset, anchor_meta)
    seed_stability = None
    if not skip_seed_stability:
        seed_stability = _run_seed_stability(
            dataset_path=anchor_dataset_path,
            config_path=config_path,
            out_dir=out_dir,
            num_theta=seed_num_theta,
            seeds=seed_seeds,
        )
    else:
        seed_stability = {"skipped": True, "reason": "run with --run-seed-stability to execute PSO multi-seed diagnostic"}

    decision = _decide_bottleneck(
        quality=quality,
        continuity=continuity_anchor,
        clustering=clustering,
        oracle=oracle,
        seed_stability=seed_stability,
    )
    summary: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "meta_path": str(meta_path),
        "anchor_dataset_path": str(anchor_dataset_path),
        "anchor_meta_path": str(anchor_meta_path),
        "standard_dataset_path": str(standard_dataset_path) if standard_dataset_path else None,
        "skip_existing_fast4": bool(skip_existing_fast4),
        "quality_gates": quality,
        "continuity_current": continuity_current,
        "continuity_anchor": continuity_anchor,
        "branch_clustering": {k: v for k, v in clustering.items() if k != "per_ball"},
        "branch_clustering_per_ball_sample": clustering.get("per_ball", [])[:200],
        "oracle_floor": oracle,
        "seed_stability": seed_stability,
        "decision": decision,
        "primary_bottleneck": decision["primary_bottleneck"],
    }
    (out_dir / "quality_gates.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "local_continuity_branch_aware.json").write_text(
        json.dumps({"current": continuity_current, "anchor": continuity_anchor}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "branch_clustering.json").write_text(json.dumps(clustering, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "oracle_floor.json").write_text(json.dumps(oracle, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "seed_stability.json").write_text(
        json.dumps(seed_stability, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    (out_dir / "diagnosis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    _write_report(summary, out_dir)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--anchor-dataset", type=Path, default=DEFAULT_ANCHOR_DATASET)
    ap.add_argument("--anchor-meta", type=Path, default=DEFAULT_ANCHOR_META)
    ap.add_argument("--standard-dataset", type=Path, default=DEFAULT_STANDARD_DATASET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--config", type=Path, default=DEFAULT_SEED_CONFIG)
    ap.add_argument("--run-seed-stability", action="store_true")
    ap.add_argument("--seed-num-theta", type=int, default=20)
    ap.add_argument("--seed-seeds", default="1001,1002,1003,1004,1005,1006,1007,1008")
    ap.add_argument("--skip-existing-fast4", action="store_true")
    args = ap.parse_args()

    standard_dataset = args.standard_dataset if args.standard_dataset and args.standard_dataset.exists() else None
    summary = run_diagnosis(
        dataset_path=args.dataset,
        meta_path=args.meta,
        anchor_dataset_path=args.anchor_dataset,
        anchor_meta_path=args.anchor_meta,
        standard_dataset_path=standard_dataset,
        out_dir=args.out_dir,
        skip_seed_stability=not bool(args.run_seed_stability),
        skip_existing_fast4=bool(args.skip_existing_fast4),
        config_path=args.config,
        seed_num_theta=args.seed_num_theta,
        seed_seeds=args.seed_seeds,
    )
    print(json.dumps({"out_dir": str(args.out_dir), "primary_bottleneck": summary["primary_bottleneck"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
