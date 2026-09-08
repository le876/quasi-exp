from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AttemptResult:
    attempt_id: int
    tag: str
    config_path: Path
    dataset_dir: Path
    gate_json: Path
    passed: bool
    p95_10mm: float
    p95_20mm: float
    pairs_10mm: int
    pairs_20mm: int
    elapsed_s: float


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n$ {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def _deep_update(cfg: dict[str, Any], patch: dict[str, Any]) -> None:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            _deep_update(cfg[k], v)
        else:
            cfg[k] = v


def _attempt_patch(attempt_id: int) -> tuple[str, dict[str, Any]]:
    # 渐进策略：先强化连续性与规范解筛选，再逐步增加搜索空间
    schedule: list[tuple[str, dict[str, Any]]] = [
        (
            "cont_up_mild",
            {
                "sampling": {"xyz_neighbor_buffer": 1024},
                "inverse_pso": {
                    "w_continuity": 12000.0,
                    "continuity_max_ratio": 0.8,
                    "continuity_relax_if_err_ratio": 1.6,
                    "continuity_relax_scale": 0.8,
                    "select_w_delta_theta": 60.0,
                    "canonical_w_delta_theta": 60.0,
                    "canonical_top_k": 4,
                },
                "dataset": {"xyz_err_threshold_m": 0.012},
            },
        ),
        (
            "cont_up_plus_canonical",
            {
                "sampling": {"xyz_neighbor_buffer": 2048},
                "inverse_pso": {
                    "w_continuity": 16000.0,
                    "continuity_max_ratio": 0.9,
                    "continuity_relax_if_err_ratio": 1.8,
                    "continuity_relax_scale": 0.9,
                    "select_w_delta_theta": 80.0,
                    "canonical_w_delta_theta": 80.0,
                    "canonical_top_k": 6,
                    "canonical_xyz_tol_m": 0.006,
                },
                "dataset": {"xyz_err_threshold_m": 0.010},
            },
        ),
        (
            "restart2",
            {
                "inverse_pso": {
                    "n_restarts": 2,
                    "warm_start_all_restarts": True,
                    "restart_seed_stride": 997,
                    "canonical_top_k": 6,
                    "select_w_delta_theta": 90.0,
                    "canonical_w_delta_theta": 90.0,
                }
            },
        ),
        (
            "search_expand_light",
            {
                "inverse_pso": {
                    "n_particles": 28,
                    "iters": 50,
                    "w_continuity": 20000.0,
                    "select_w_delta_theta": 100.0,
                    "canonical_w_delta_theta": 100.0,
                    "canonical_top_k": 8,
                }
            },
        ),
        (
            "restart3",
            {
                "inverse_pso": {
                    "n_restarts": 3,
                    "warm_start_all_restarts": True,
                    "restart_seed_stride": 701,
                    "canonical_top_k": 8,
                    "canonical_xyz_tol_m": 0.008,
                }
            },
        ),
        (
            "search_expand_mid",
            {
                "inverse_pso": {
                    "n_particles": 32,
                    "iters": 60,
                    "w_continuity": 25000.0,
                    "continuity_max_ratio": 1.0,
                    "select_w_delta_theta": 120.0,
                    "canonical_w_delta_theta": 120.0,
                }
            },
        ),
        (
            "canonical_strong",
            {
                "inverse_pso": {
                    "n_restarts": 4,
                    "canonical_top_k": 10,
                    "canonical_xyz_tol_m": 0.010,
                    "select_w_delta_theta": 140.0,
                    "canonical_w_delta_theta": 140.0,
                }
            },
        ),
        (
            "final_strong",
            {
                "inverse_pso": {
                    "n_particles": 36,
                    "iters": 70,
                    "w_continuity": 30000.0,
                    "n_restarts": 4,
                    "canonical_top_k": 12,
                    "select_w_delta_theta": 160.0,
                    "canonical_w_delta_theta": 160.0,
                },
                "dataset": {"xyz_err_threshold_m": 0.008},
            },
        ),
    ]
    idx = max(1, min(attempt_id, len(schedule))) - 1
    return schedule[idx]


def _read_gate(gate_path: Path) -> tuple[bool, float, float, int, int]:
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    passed = bool(payload.get("passed", False))
    local = payload.get("metrics", {}).get("local", {})
    m10 = local.get("<= 10mm", {})
    m20 = local.get("<= 20mm", {})
    p95_10 = float(m10.get("theta_rms_deg_p95", float("nan")))
    p95_20 = float(m20.get("theta_rms_deg_p95", float("nan")))
    n10 = int(m10.get("pairs", 0))
    n20 = int(m20.get("pairs", 0))
    return passed, p95_10, p95_20, n10, n20


def _append_md_row(md_path: Path, r: AttemptResult) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    exists = md_path.exists()
    with md_path.open("a", encoding="utf-8") as f:
        if not exists:
            f.write(
                "| attempt | tag | passed | p95<=10mm(deg) | p95<=20mm(deg) | pairs<=10mm | pairs<=20mm | elapsed(min) | cfg |\n"
            )
            f.write("|---:|---|:---:|---:|---:|---:|---:|---:|---|\n")
        f.write(
            f"| {r.attempt_id} | {r.tag} | {'Y' if r.passed else 'N'} | "
            f"{r.p95_10mm:.4f} | {r.p95_20mm:.4f} | {r.pairs_10mm} | {r.pairs_20mm} | "
            f"{r.elapsed_s/60.0:.2f} | {r.config_path} |\n"
        )


def run(
    base_config: Path,
    env_name: str,
    python_bin: str | None,
    max_attempts: int,
    start_attempt: int,
    num_samples: int,
    max_tried: int,
) -> int:
    base = _load_yaml(base_config)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path("runs/acceptance") / f"inverse_5deg_tune_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    summary_md = run_root / "summary.md"
    meta_json = run_root / "meta.json"

    meta = {
        "base_config": str(base_config),
        "env": env_name,
        "python_bin": python_bin,
        "max_attempts": max_attempts,
        "start_attempt": start_attempt,
        "num_samples": num_samples,
        "max_tried": max_tried,
        "attempts": [],
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def py_runner() -> list[str]:
        if python_bin:
            return [python_bin]
        return ["conda", "run", "-n", env_name, "python"]

    for aid in range(start_attempt, max_attempts + 1):
        tag, patch = _attempt_patch(aid)
        cfg = deepcopy(base)
        _deep_update(cfg, patch)

        # 每轮独立输出目录，避免污染
        out_dir = Path(f"data/tune_inverse_5deg_attempt{aid:02d}")
        cfg["dataset"]["out_dir"] = str(out_dir)
        cfg["sampling"]["rng_seed"] = int(base["sampling"].get("rng_seed", 0)) + aid * 37
        cfg["pso"]["rng_seed"] = int(base["pso"].get("rng_seed", 0)) + aid * 101

        cfg_path = Path("configs/tuning") / f"robot_rods_only_5deg_inverse_2k_attempt{aid:02d}.yaml"
        _dump_yaml(cfg_path, cfg)

        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        gate_json = run_root / f"attempt_{aid:02d}_gate.json"
        log_path = run_root / f"attempt_{aid:02d}.log"
        t0 = time.time()

        try:
            _run_cmd(
                py_runner()
                + [
                    "scripts/generate_dataset.py",
                    "--config",
                    str(cfg_path),
                    "--num-samples",
                    str(num_samples),
                    "--max-tried",
                    str(max_tried),
                ],
                log_path,
            )
            _run_cmd(
                py_runner()
                + [
                    "scripts/analysis/eval_local_continuity.py",
                    "--dataset",
                    str(out_dir / "dataset.parquet"),
                    "--out-json",
                    str(gate_json),
                ],
                log_path,
            )
            passed, p95_10, p95_20, n10, n20 = _read_gate(gate_json)
        except subprocess.CalledProcessError:
            # eval脚本失败时也尽量读已生成gate（若有）
            if gate_json.exists():
                passed, p95_10, p95_20, n10, n20 = _read_gate(gate_json)
            else:
                passed, p95_10, p95_20, n10, n20 = False, float("nan"), float("nan"), 0, 0

        elapsed_s = time.time() - t0
        ar = AttemptResult(
            attempt_id=aid,
            tag=tag,
            config_path=cfg_path,
            dataset_dir=out_dir,
            gate_json=gate_json,
            passed=passed,
            p95_10mm=p95_10,
            p95_20mm=p95_20,
            pairs_10mm=n10,
            pairs_20mm=n20,
            elapsed_s=elapsed_s,
        )
        _append_md_row(summary_md, ar)

        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        meta["attempts"].append(
            {
                "attempt": aid,
                "tag": tag,
                "passed": passed,
                "p95_10mm": p95_10,
                "p95_20mm": p95_20,
                "pairs_10mm": n10,
                "pairs_20mm": n20,
                "elapsed_s": elapsed_s,
                "config": str(cfg_path),
                "dataset_dir": str(out_dir),
                "gate_json": str(gate_json),
            }
        )
        meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if passed:
            return 0

    return 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-config",
        default="configs/robot_rods_only_5deg_inverse_2k_paper_tf_cpu.yaml",
    )
    ap.add_argument("--env-name", default="quasi_exp")
    ap.add_argument("--python-bin", default="")
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--start-attempt", type=int, default=1)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--max-tried", type=int, default=15000)
    args = ap.parse_args()

    rc = run(
        base_config=Path(args.base_config),
        env_name=args.env_name,
        python_bin=(args.python_bin.strip() or None),
        max_attempts=int(args.max_attempts),
        start_attempt=int(args.start_attempt),
        num_samples=int(args.num_samples),
        max_tried=int(args.max_tried),
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
