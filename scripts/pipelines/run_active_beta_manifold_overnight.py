#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts" / "generate_active_beta_manifold_dataset.py"
BASELINES = REPO_ROOT / "scripts" / "baselines" / "run_baselines.py"
VARIANTS = ("manifold_3d_full_angle", "filtered_6d_distal_cloud", "manifold_3d_quadrant")


def _json_default(obj: Any) -> Any:
    return str(obj)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def all10_pairs(report: dict[str, Any]) -> int:
    return int(
        report.get("diagnostics", {})
        .get("continuity", {})
        .get("groups", {})
        .get("all_xyz_<=10mm", {})
        .get("pairs", 0)
    )


def strict_gate_passed(gate: dict[str, Any], *, all10_pairs: int, min_all10_pairs: int) -> bool:
    return bool(gate.get("hard_gate_passed", False)) and (
        int(all10_pairs) >= int(min_all10_pairs)
        and float(gate.get("all10_theta_p95_deg", float("inf"))) <= 3.0
        and float(gate.get("all10_tension_p95_n", float("inf"))) <= 100.0
        and float(gate.get("beta_close_tension_p95_n", float("inf"))) <= 50.0
        and float(gate.get("multi_branch_ball_ratio", float("inf"))) <= 0.25
    )


def gate_score(gate: dict[str, Any], *, pairs: int) -> float:
    return float(
        (float(gate.get("all10_theta_p95_deg", float("inf"))) / 3.0) ** 2
        + (float(gate.get("all10_tension_p95_n", float("inf"))) / 100.0) ** 2
        + (float(gate.get("beta_close_tension_p95_n", float("inf"))) / 50.0) ** 2
        + (float(gate.get("multi_branch_ball_ratio", float("inf"))) / 0.25) ** 2
        + 1.0 / max(int(pairs), 1)
    )


def select_best_passing_variant(reports: list[dict[str, Any]], *, min_all10_pairs: int) -> dict[str, Any] | None:
    passing: list[tuple[float, dict[str, Any]]] = []
    for report in reports:
        gate = dict(report.get("gate", {}))
        pairs = all10_pairs(report)
        if strict_gate_passed(gate, all10_pairs=pairs, min_all10_pairs=int(min_all10_pairs)):
            passing.append((gate_score(gate, pairs=pairs), report))
    if not passing:
        return None
    return sorted(passing, key=lambda item: (item[0], str(item[1].get("variant", ""))))[0][1]


def _run(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write("$ " + " ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code {proc.returncode}; see {log_path}")


def _load_generation_payload(out_dir: Path) -> dict[str, Any]:
    report = _read_json(out_dir / "active_generation_report.json")
    diagnostics = _read_json(out_dir / "diagnostics" / "active_diagnostics.json")
    report["diagnostics"] = diagnostics
    report["gate"] = diagnostics.get("gate", report.get("gate", {}))
    return report


def run_generation(
    *,
    config: Path,
    variant: str,
    num_samples: int,
    pool_size: int,
    out_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--config",
        str(config),
        "--variant",
        str(variant),
        "--num-samples",
        str(int(num_samples)),
        "--pool-size",
        str(int(pool_size)),
        "--out-dir",
        str(out_dir),
    ]
    _run(cmd, cwd=REPO_ROOT, log_path=log_path)
    return _load_generation_payload(out_dir)


def run_baselines(*, dataset: Path, config: Path, runs_root: Path, log_dir: Path) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for split in ["iid", "radius", "beta_block", "angular_sector"]:
        out_dir = runs_root / "baselines" / split
        cmd = [
            sys.executable,
            str(BASELINES),
            "--dataset",
            str(dataset),
            "--out-dir",
            str(out_dir),
            "--robot-config",
            str(config),
            "--backend",
            "both",
            "--tf-device",
            "gpu",
            "--models",
            "mlp_large,tf_mlp_large",
            "--split",
            split,
            "--feature-set",
            "poly_heavy",
            "--save-curves",
        ]
        _run(cmd, cwd=REPO_ROOT, log_path=log_dir / f"baseline_{split}.log")
        summaries[split] = _read_json(out_dir / "all_metrics.json")
    return summaries


def run_pipeline(
    *,
    config: Path,
    num_samples_2k: int,
    pool_size_2k: int,
    num_samples_20k: int,
    pool_size_20k: int,
    out_root: Path,
    runs_root: Path,
    variants: tuple[str, ...] = VARIANTS,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    pilot_reports: list[dict[str, Any]] = []
    for variant in variants:
        out_dir = out_root / "pilot_2k" / variant
        report = run_generation(
            config=config,
            variant=variant,
            num_samples=int(num_samples_2k),
            pool_size=int(pool_size_2k),
            out_dir=out_dir,
            log_path=runs_root / "logs" / f"generate_2k_{variant}.log",
        )
        report["variant"] = variant
        report["stage"] = "pilot_2k"
        report["out_dir"] = str(out_dir)
        report["strict_gate_passed"] = strict_gate_passed(report["gate"], all10_pairs=all10_pairs(report), min_all10_pairs=100)
        report["gate_score"] = gate_score(report["gate"], pairs=all10_pairs(report))
        pilot_reports.append(report)
        _write_json(runs_root / "pilot_2k_summary.json", {"pilots": pilot_reports})

    best = select_best_passing_variant(pilot_reports, min_all10_pairs=100)
    payload: dict[str, Any] = {
        "config": str(config),
        "pilots": pilot_reports,
        "best_variant": None if best is None else str(best.get("variant")),
        "twenty_k": None,
        "baselines": None,
        "status": "stopped_no_2k_variant_passed",
    }
    if best is None:
        _write_json(runs_root / "summary.json", payload)
        return payload

    variant = str(best["variant"])
    out_dir_20k = out_root / "twenty_k" / variant
    report_20k = run_generation(
        config=config,
        variant=variant,
        num_samples=int(num_samples_20k),
        pool_size=int(pool_size_20k),
        out_dir=out_dir_20k,
        log_path=runs_root / "logs" / f"generate_20k_{variant}.log",
    )
    report_20k["variant"] = variant
    report_20k["stage"] = "twenty_k"
    report_20k["out_dir"] = str(out_dir_20k)
    report_20k["strict_gate_passed"] = strict_gate_passed(report_20k["gate"], all10_pairs=all10_pairs(report_20k), min_all10_pairs=1000)
    report_20k["gate_score"] = gate_score(report_20k["gate"], pairs=all10_pairs(report_20k))
    payload["twenty_k"] = report_20k
    payload["status"] = "stopped_20k_gate_failed"
    _write_json(runs_root / "summary.json", payload)

    if not bool(report_20k["strict_gate_passed"]):
        return payload

    baselines = run_baselines(
        dataset=out_dir_20k / "dataset.parquet",
        config=config,
        runs_root=runs_root,
        log_dir=runs_root / "logs",
    )
    payload["baselines"] = baselines
    payload["status"] = "complete"
    _write_json(runs_root / "summary.json", payload)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--num-samples-2k", type=int, default=2000)
    ap.add_argument("--pool-size-2k", type=int, default=20000)
    ap.add_argument("--num-samples-20k", type=int, default=20000)
    ap.add_argument("--pool-size-20k", type=int, default=200000)
    ap.add_argument("--out-root", type=Path, default=Path("data/active_beta_manifold_distal_v1"))
    ap.add_argument("--runs-root", type=Path, default=Path("runs/active_beta_manifold_distal_v1"))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    variants = tuple(s.strip() for s in str(args.variants).split(",") if s.strip())
    payload = run_pipeline(
        config=args.config,
        num_samples_2k=int(args.num_samples_2k),
        pool_size_2k=int(args.pool_size_2k),
        num_samples_20k=int(args.num_samples_20k),
        pool_size_20k=int(args.pool_size_20k),
        out_root=args.out_root,
        runs_root=args.runs_root,
        variants=variants,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
