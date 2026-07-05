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

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path("/mnt/ML_projects/conda_envs/dante_env/bin/python")
DEFAULT_CONFIG = REPO_ROOT / "configs/robot_rods_only_mixed_20k_single_branch_distal_v2.yaml"
DEFAULT_REFERENCE = REPO_ROOT / "data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20"
DEFAULT_POOL = [
    REPO_ROOT / "data/mixed_beta_100k_distal_preferred_segmented_canonical",
    REPO_ROOT / "data/mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20",
]


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write("\n$ " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _phase_payload(summary_path: Path) -> dict[str, Any]:
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "phases": {}}


def _quality_gates(dataset: pd.DataFrame, meta: pd.DataFrame) -> dict[str, Any]:
    phase1 = _load_script("run_gpt5pro_phase1_diagnosis")
    return phase1._quality_gates(dataset, meta)  # reuse project diagnostic gate


def _diagnose_dataset(dataset_path: Path, meta_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(dataset_path)
    meta = pd.read_parquet(meta_path)
    continuity_mod = _load_script("eval_branch_aware_continuity")
    clustering_mod = _load_script("eval_branch_clustering")
    oracle_mod = _load_script("eval_oracle_floor")
    quality = _quality_gates(dataset, meta)
    continuity = continuity_mod.evaluate_frames(dataset, meta)
    clustering = clustering_mod.evaluate_frames(dataset, meta)
    oracle = oracle_mod.evaluate_frames(
        dataset,
        meta,
        branch_column="graph_component_id" if "graph_component_id" in meta.columns else None,
    )
    payload = {
        "quality_gates": quality,
        "continuity": continuity,
        "branch_clustering": {k: v for k, v in clustering.items() if k != "per_ball"},
        "oracle_floor": oracle,
    }
    _write_json(out_dir / "diagnostics.json", payload)
    return payload


def _extract_gate(diagnostics: dict[str, Any]) -> dict[str, Any]:
    quality_passed = bool(diagnostics.get("quality_gates", {}).get("passed", False))
    groups = diagnostics.get("continuity", {}).get("groups", {})
    all10 = groups.get("all_xyz_<=10mm", {})
    beta_close = groups.get("xyz_<=10mm_beta_close", {})
    oracle = diagnostics.get("oracle_floor", {})
    return {
        "hard_gate_passed": quality_passed,
        "all10_theta_p95_deg": float(all10.get("theta_rms_deg_p95", float("inf"))),
        "all10_tension_p95_n": float(all10.get("tension_mae_n_p95", float("inf"))),
        "beta_close_theta_p95_deg": float(beta_close.get("theta_rms_deg_p95", float("inf"))),
        "beta_close_tension_p95_n": float(beta_close.get("tension_mae_n_p95", float("inf"))),
        "xyz_nn_tension_mae_n": float(oracle.get("xyz_nn_oracle", {}).get("tension_mae_n", float("inf"))),
        "branch_aware_xyz_nn_tension_mae_n": float(
            oracle.get("branch_aware_xyz_nn_oracle", {}).get("tension_mae_n", float("inf"))
        ),
        "beta_nn_tension_mae_n": float(oracle.get("beta_nn_oracle", {}).get("tension_mae_n", float("inf"))),
    }


def _theta_continuity_stop_failed(gate: dict[str, Any], *, stop_deg: float) -> bool:
    return float(gate.get("all10_theta_p95_deg", float("inf"))) > float(stop_deg)


def _training_acceptance_passed(
    gate: dict[str, Any],
    *,
    theta_target_deg: float,
    tension_target_n: float,
    beta_close_tension_target_n: float,
) -> bool:
    return bool(gate.get("hard_gate_passed", False)) and (
        float(gate.get("all10_theta_p95_deg", float("inf"))) <= float(theta_target_deg)
        and float(gate.get("all10_tension_p95_n", float("inf"))) <= float(tension_target_n)
        and float(gate.get("beta_close_tension_p95_n", float("inf"))) <= float(beta_close_tension_target_n)
    )


def _write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    phases = payload.get("phases", {})
    lines = [
        "# Phase 4 Single-Branch Distal V2 Summary",
        "",
        f"- started_at: {payload.get('started_at', '')}",
        f"- updated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| phase | status | output |",
        "| --- | --- | --- |",
    ]
    for key, row in phases.items():
        lines.append(f"| {key} | {row.get('status', 'unknown')} | `{row.get('output', '')}` |")
    gate = phases.get("diagnostics_selected", {}).get("gate", {})
    if gate:
        lines.extend(
            [
                "",
                "## Selected Dataset Gate",
                "",
                f"- hard gate passed: `{gate.get('hard_gate_passed')}`",
                f"- all-workspace 10mm theta p95: {gate.get('all10_theta_p95_deg'):.3f} deg",
                f"- all-workspace 10mm T p95: {gate.get('all10_tension_p95_n'):.2f} N",
                f"- beta-close 10mm theta p95: {gate.get('beta_close_theta_p95_deg'):.3f} deg",
                f"- beta-close 10mm T p95: {gate.get('beta_close_tension_p95_n'):.2f} N",
                f"- xyz-NN T MAE: {gate.get('xyz_nn_tension_mae_n'):.2f} N",
                f"- branch-aware xyz-NN T MAE: {gate.get('branch_aware_xyz_nn_tension_mae_n'):.2f} N",
                f"- beta-NN T MAE: {gate.get('beta_nn_tension_mae_n'):.2f} N",
                f"- acceptance gate passed: `{gate.get('acceptance_gate_passed')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mark(summary_path: Path, md_path: Path, phase: str, status: str, **kwargs: Any) -> dict[str, Any]:
    payload = _phase_payload(summary_path)
    payload.setdefault("phases", {})[phase] = {"status": status, **kwargs}
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(summary_path, payload)
    _write_summary_md(md_path, payload)
    return payload


def _run_direct_baselines(
    *,
    python_bin: Path,
    dataset: Path,
    config: Path,
    out_root: Path,
    splits: list[str],
    models: str,
    log_path: Path,
) -> None:
    for split in splits:
        _run(
            [
                str(python_bin),
                "scripts/baselines/run_baselines.py",
                "--dataset",
                str(dataset),
                "--out-dir",
                str(out_root / split),
                "--split",
                split,
                "--backend",
                "both",
                "--models",
                models,
                "--robot-config",
                str(config),
                "--feature-set",
                "poly_heavy",
                "--tf-verbose",
                "0",
            ],
            log_path,
        )


def _run_relabel_sweep(
    *,
    python_bin: Path,
    config: Path,
    dataset_dir: Path,
    out_parent: Path,
    w_values: list[float],
    workers: int,
    log_path: Path,
) -> Path:
    rows = int(pd.read_parquet(dataset_dir / "dataset.parquet").shape[0])
    diagnostics_by_w: dict[str, Any] = {}
    best_dir = dataset_dir
    best_t = float("inf")
    for w in w_values:
        out_dir = out_parent / f"anchor_w{int(w) if float(w).is_integer() else str(w).replace('.', 'p')}"
        _run(
            [
                str(python_bin),
                "scripts/analysis/relabel_anchor_canonical.py",
                "--config",
                str(config),
                "--dataset",
                str(dataset_dir / "dataset.parquet"),
                "--meta",
                str(dataset_dir / "dataset_meta.parquet"),
                "--out-dir",
                str(out_dir),
                "--num-samples",
                str(rows),
                "--component-scope",
                "selected_branch_graph",
                "--distance-space",
                "effective_beta",
                "--no-cross-branch-anchor",
                "--anchor-k",
                "32",
                "--w-anchor",
                str(w),
                "--workers",
                str(workers),
            ],
            log_path,
        )
        diag = _diagnose_dataset(out_dir / "dataset.parquet", out_dir / "dataset_meta.parquet", out_dir / "diagnostics")
        gate = _extract_gate(diag)
        diagnostics_by_w[str(w)] = gate
        if gate["hard_gate_passed"] and gate["all10_tension_p95_n"] < best_t:
            best_t = gate["all10_tension_p95_n"]
            best_dir = out_dir
    _write_json(out_parent / "relabel_sweep_summary.json", {"diagnostics_by_w": diagnostics_by_w, "best_dir": str(best_dir)})
    return best_dir


def run(args: argparse.Namespace) -> int:
    python_bin = Path(args.python_bin)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "phase4_summary.json"
    md_path = run_root / "phase4_summary.md"
    log_path = run_root / "phase4.log"
    config = Path(args.config)
    reference_dataset = Path(args.reference_dataset)
    reference_meta = Path(args.reference_meta)
    candidate_pools = [Path(v.strip()) for v in str(args.candidate_pool).split(",") if v.strip()]
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    graph_out = Path(args.graph_out)
    single_out = Path(args.single_out)
    selected_dir = single_out

    _mark(summary_path, md_path, "phase4a_graph_filter", "running", output=str(graph_out))
    _run(
        [
            str(python_bin),
            "scripts/analysis/select_canonical_branch_graph.py",
            "--dataset",
            str(reference_dataset),
            "--meta",
            str(reference_meta),
            "--out",
            str(graph_out),
            "--voxel-mm",
            str(args.voxel_mm),
            "--ball-mm",
            str(args.ball_mm),
            "--beta-dbscan-eps",
            str(args.beta_dbscan_eps),
            "--policy",
            str(args.policy),
        ],
        log_path,
    )
    _mark(summary_path, md_path, "phase4a_graph_filter", "complete", output=str(graph_out))

    _mark(summary_path, md_path, "phase4b_single_branch_pool", "running", output=str(single_out))
    _run(
        [
            str(python_bin),
            "scripts/generate_single_branch_dataset.py",
            "--candidate-pool",
            ",".join(str(p) for p in candidate_pools),
            "--out-dir",
            str(single_out),
            "--num-samples",
            str(args.num_samples),
            "--voxel-mm",
            str(args.voxel_mm),
            "--ball-mm",
            str(args.ball_mm),
            "--beta-dbscan-eps",
            str(args.beta_dbscan_eps),
            "--seed",
            str(args.seed),
        ],
        log_path,
    )
    _mark(summary_path, md_path, "phase4b_single_branch_pool", "complete", output=str(single_out))

    diag = _diagnose_dataset(single_out / "dataset.parquet", single_out / "dataset_meta.parquet", run_root / "diagnostics_single_branch")
    gate = _extract_gate(diag)
    gate["acceptance_gate_passed"] = _training_acceptance_passed(
        gate,
        theta_target_deg=float(args.theta_continuity_target_deg),
        tension_target_n=float(args.tension_continuity_target_n),
        beta_close_tension_target_n=float(args.beta_close_tension_target_n),
    )
    _mark(summary_path, md_path, "diagnostics_selected", "complete", output=str(run_root / "diagnostics_single_branch"), gate=gate)

    if not bool(gate["hard_gate_passed"]):
        _mark(summary_path, md_path, "stopped", "hard_gate_failed", output=str(single_out))
        return 2

    if _theta_continuity_stop_failed(gate, stop_deg=float(args.theta_continuity_stop_deg)) and not bool(args.force_baselines):
        _mark(summary_path, md_path, "stopped", "theta_continuity_gate_failed_before_relabel", output=str(single_out))
        return 3

    if not bool(args.skip_relabel):
        _mark(summary_path, md_path, "phase4c_relabel_sweep", "running", output=str(args.relabel_out))
        selected_dir = _run_relabel_sweep(
            python_bin=python_bin,
            config=config,
            dataset_dir=single_out,
            out_parent=Path(args.relabel_out),
            w_values=[float(v) for v in str(args.w_anchor_sweep).split(",") if v.strip()],
            workers=int(args.relabel_workers),
            log_path=log_path,
        )
        _mark(summary_path, md_path, "phase4c_relabel_sweep", "complete", output=str(selected_dir))
        diag = _diagnose_dataset(selected_dir / "dataset.parquet", selected_dir / "dataset_meta.parquet", run_root / "diagnostics_anchor_selected")
        gate = _extract_gate(diag)
        gate["acceptance_gate_passed"] = _training_acceptance_passed(
            gate,
            theta_target_deg=float(args.theta_continuity_target_deg),
            tension_target_n=float(args.tension_continuity_target_n),
            beta_close_tension_target_n=float(args.beta_close_tension_target_n),
        )
        _mark(summary_path, md_path, "diagnostics_selected", "complete", output=str(run_root / "diagnostics_anchor_selected"), gate=gate)

    if _theta_continuity_stop_failed(gate, stop_deg=float(args.theta_continuity_stop_deg)) and not bool(args.force_baselines):
        _mark(summary_path, md_path, "stopped", "theta_continuity_gate_failed", output=str(selected_dir))
        return 3

    if not _training_acceptance_passed(
        gate,
        theta_target_deg=float(args.theta_continuity_target_deg),
        tension_target_n=float(args.tension_continuity_target_n),
        beta_close_tension_target_n=float(args.beta_close_tension_target_n),
    ) and not bool(args.force_baselines):
        _mark(summary_path, md_path, "stopped", "selected_dataset_acceptance_gate_failed", output=str(selected_dir))
        return 4

    if not bool(args.skip_baselines):
        _mark(summary_path, md_path, "phase4d_direct_baselines", "running", output=str(args.baseline_out))
        _run_direct_baselines(
            python_bin=python_bin,
            dataset=selected_dir / "dataset.parquet",
            config=config,
            out_root=Path(args.baseline_out),
            splits=splits,
            models=str(args.models),
            log_path=log_path,
        )
        _mark(summary_path, md_path, "phase4d_direct_baselines", "complete", output=str(args.baseline_out))

    if not bool(args.skip_beta_aux):
        _mark(summary_path, md_path, "phase4d_beta_aux", "running", output=str(args.beta_aux_out))
        _run(
            [
                str(python_bin),
                "scripts/baselines/run_beta_aux_baselines.py",
                "--dataset",
                str(selected_dir / "dataset.parquet"),
                "--out-dir",
                str(args.beta_aux_out),
                "--splits",
                ",".join(splits),
                "--robot-config",
                str(config),
                "--epochs",
                str(args.beta_aux_epochs),
            ],
            log_path,
        )
        _mark(summary_path, md_path, "phase4d_beta_aux", "complete", output=str(args.beta_aux_out))

    _mark(summary_path, md_path, "complete", "complete", output=str(selected_dir))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--reference-dataset", default=str(DEFAULT_REFERENCE / "dataset.parquet"))
    ap.add_argument("--reference-meta", default=str(DEFAULT_REFERENCE / "dataset_meta.parquet"))
    ap.add_argument("--candidate-pool", default=",".join(str(p) for p in DEFAULT_POOL))
    ap.add_argument("--num-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--policy", default="distal_v2_graph")
    ap.add_argument("--voxel-mm", type=float, default=10.0)
    ap.add_argument("--ball-mm", type=float, default=15.0)
    ap.add_argument("--beta-dbscan-eps", type=float, default=0.15)
    ap.add_argument("--w-anchor-sweep", default="5,10,20,40")
    ap.add_argument("--relabel-workers", type=int, default=16)
    ap.add_argument("--splits", default="iid,radius,beta_block,angular_sector")
    ap.add_argument("--models", default="mlp,mlp_large,rf,lgbm,knn")
    ap.add_argument("--run-root", default="runs/diagnostics/phase4_single_branch_distal_v2")
    ap.add_argument("--graph-out", default="data/mixed_beta_20k_anchor_v2_graph_branch_filtered_s2")
    ap.add_argument("--single-out", default="data/mixed_beta_20k_single_branch_distal_v2")
    ap.add_argument("--relabel-out", default="data/mixed_beta_20k_single_branch_distal_v2_anchor_sweep")
    ap.add_argument("--baseline-out", default="runs/baselines_single_branch_distal_v2_fast4")
    ap.add_argument("--beta-aux-out", default="runs/baselines_single_branch_distal_v2_beta_aux")
    ap.add_argument("--beta-aux-epochs", type=int, default=200)
    ap.add_argument("--theta-continuity-stop-deg", type=float, default=5.0)
    ap.add_argument("--theta-continuity-target-deg", type=float, default=3.0)
    ap.add_argument("--tension-continuity-target-n", type=float, default=100.0)
    ap.add_argument("--beta-close-tension-target-n", type=float, default=50.0)
    ap.add_argument("--skip-relabel", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-beta-aux", action="store_true")
    ap.add_argument("--force-baselines", action="store_true")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
