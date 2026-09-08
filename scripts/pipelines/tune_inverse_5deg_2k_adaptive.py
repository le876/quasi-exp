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
    config_path: Path
    dataset_dir: Path
    gate_json: Path
    passed: bool
    checks: dict[str, bool]
    p95_10mm: float
    p95_20mm: float
    pairs_10mm: int
    pairs_20mm: int
    ratio_gt10_10mm: float
    ratio_gt10_20mm: float
    accept_rate: float
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
            logf.flush()
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def _deep_update(cfg: dict[str, Any], patch: dict[str, Any]) -> None:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            _deep_update(cfg[k], v)
        else:
            cfg[k] = v


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _read_gate(gate_path: Path) -> tuple[bool, dict[str, bool], float, float, int, int, float, float]:
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    passed = bool(payload.get("passed", False))
    checks = payload.get("checks", {})
    local = payload.get("metrics", {}).get("local", {})
    m10 = local.get("<= 10mm", {})
    m20 = local.get("<= 20mm", {})
    p95_10 = float(m10.get("theta_rms_deg_p95", float("nan")))
    p95_20 = float(m20.get("theta_rms_deg_p95", float("nan")))
    n10 = int(m10.get("pairs", 0))
    n20 = int(m20.get("pairs", 0))
    gt10_10 = float(m10.get("ratio_rms_gt10deg", float("nan")))
    gt10_20 = float(m20.get("ratio_rms_gt10deg", float("nan")))
    return passed, checks, p95_10, p95_20, n10, n20, gt10_10, gt10_20


def _read_accept_rate(dataset_dir: Path) -> float:
    report = dataset_dir / "dataset_report.json"
    if not report.exists():
        return float("nan")
    payload = json.loads(report.read_text(encoding="utf-8"))
    return float(payload.get("accept_rate", float("nan")))


def _state_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    inv = cfg["inverse_pso"]
    sampling = cfg.get("sampling", {})
    dataset = cfg.get("dataset", {})
    parallel = cfg.get("parallel", {})
    state = {
        "inverse_pso": {
            "w_continuity": float(inv.get("w_continuity", 6000.0)),
            "select_w_delta_theta": float(inv.get("select_w_delta_theta", 20.0)),
            "canonical_w_delta_theta": float(inv.get("canonical_w_delta_theta", 20.0)),
            "canonical_top_k": int(inv.get("canonical_top_k", 3)),
            "canonical_xyz_tol_m": float(inv.get("canonical_xyz_tol_m", 0.004)),
            "n_restarts": int(inv.get("n_restarts", 1)),
            "n_particles": int(inv.get("n_particles", 24)),
            "iters": int(inv.get("iters", 40)),
            "use_continuity_chain": bool(inv.get("use_continuity_chain", True)),
            "warm_start_particles": int(inv.get("warm_start_particles", 8)),
            "warm_start_sigma": float(inv.get("warm_start_sigma", 0.08)),
            "continuity_max_ratio": float(inv.get("continuity_max_ratio", 0.3)),
            "continuity_relax_if_err_ratio": float(inv.get("continuity_relax_if_err_ratio", 1.2)),
            "continuity_relax_scale": float(inv.get("continuity_relax_scale", 0.3)),
        },
        "sampling": {
            "xyz_neighbor_buffer": int(sampling.get("xyz_neighbor_buffer", 256)),
        },
        "dataset": {
            "xyz_err_threshold_m": float(dataset.get("xyz_err_threshold_m", 0.02)),
        },
        "parallel": {
            "workers": int(parallel.get("workers", 1)),
            "allow_tf_multi_worker": bool(parallel.get("allow_tf_multi_worker", False)),
        },
    }
    return state


def _patch_from_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "inverse_pso": deepcopy(state["inverse_pso"]),
        "sampling": deepcopy(state["sampling"]),
        "dataset": deepcopy(state["dataset"]),
        "parallel": deepcopy(state["parallel"]),
    }


def _append_summary(summary_md: Path, r: AttemptResult, decision_text: str) -> None:
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_md.exists()
    with summary_md.open("a", encoding="utf-8") as f:
        if not exists:
            f.write(
                "| attempt | passed | p95<=10mm | p95<=20mm | pairs10 | pairs20 | gt10@10mm | gt10@20mm | accept_rate | elapsed(min) | cfg |\n"
            )
            f.write("|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        f.write(
            f"| {r.attempt_id} | {'Y' if r.passed else 'N'} | {r.p95_10mm:.4f} | {r.p95_20mm:.4f} | "
            f"{r.pairs_10mm} | {r.pairs_20mm} | {r.ratio_gt10_10mm:.4f} | {r.ratio_gt10_20mm:.4f} | "
            f"{r.accept_rate:.4f} | {r.elapsed_s/60.0:.2f} | {r.config_path.name} |\n"
        )
        f.write(f"\n- decision: {decision_text}\n\n")


def _append_decision_log(decision_md: Path, aid: int, before_state: dict[str, Any], after_state: dict[str, Any], notes: list[str]) -> None:
    decision_md.parent.mkdir(parents=True, exist_ok=True)
    with decision_md.open("a", encoding="utf-8") as f:
        f.write(f"## attempt {aid}\n")
        for n in notes:
            f.write(f"- {n}\n")
        f.write("- before:\n")
        f.write("```json\n")
        f.write(json.dumps(before_state, ensure_ascii=False, indent=2))
        f.write("\n```\n")
        f.write("- after:\n")
        f.write("```json\n")
        f.write(json.dumps(after_state, ensure_ascii=False, indent=2))
        f.write("\n```\n\n")


def _adapt_state(
    state: dict[str, Any],
    result: AttemptResult,
    history: list[AttemptResult],
) -> tuple[dict[str, Any], str, list[str]]:
    new_state = deepcopy(state)
    inv = new_state["inverse_pso"]
    sampling = new_state["sampling"]
    dataset = new_state["dataset"]
    parallel = new_state["parallel"]

    notes: list[str] = []

    if not bool(inv.get("use_continuity_chain", False)):
        inv["use_continuity_chain"] = True
        parallel["workers"] = 1
        notes.append("启用use_continuity_chain，并切到workers=1保证链式连续性生效")

    if int(parallel.get("workers", 1)) != 1 and bool(inv.get("use_continuity_chain", False)):
        parallel["workers"] = 1
        notes.append("连续性链模式下固定workers=1")

    if result.pairs_10mm < 100 or result.pairs_20mm < 500:
        sampling["xyz_neighbor_buffer"] = int(min(4096, int(sampling["xyz_neighbor_buffer"] * 1.5)))
        dataset["xyz_err_threshold_m"] = _clip(dataset["xyz_err_threshold_m"] + 0.002, 0.01, 0.03)
        notes.append("邻域样本对不足，提升xyz_neighbor_buffer并适度放宽xyz_err_threshold")

    if result.p95_20mm > 6.0:
        inv["w_continuity"] = _clip(inv["w_continuity"] + 2500.0, 4000.0, 35000.0)
        inv["select_w_delta_theta"] = _clip(inv["select_w_delta_theta"] + 15.0, 20.0, 220.0)
        inv["canonical_w_delta_theta"] = _clip(inv["canonical_w_delta_theta"] + 15.0, 20.0, 220.0)
        inv["canonical_top_k"] = int(_clip(inv["canonical_top_k"] + 1, 3, 14))
        inv["continuity_max_ratio"] = _clip(inv["continuity_max_ratio"] + 0.05, 0.2, 0.7)
        notes.append("20mm局部连续性未过，提升连续性惩罚和canonical筛选强度")

    if result.p95_10mm > 6.0:
        inv["n_restarts"] = int(_clip(inv["n_restarts"] + 1, 1, 4))
        inv["canonical_xyz_tol_m"] = _clip(inv["canonical_xyz_tol_m"] + 0.001, 0.003, 0.012)
        inv["warm_start_particles"] = int(_clip(inv["warm_start_particles"] + 2, 4, 20))
        inv["warm_start_sigma"] = _clip(inv["warm_start_sigma"] - 0.01, 0.03, 0.12)
        notes.append("10mm局部连续性未过，增加重启次数并扩大canonical候选容差")

    if result.ratio_gt10_10mm > 0.01 or result.ratio_gt10_20mm > 0.02:
        inv["n_particles"] = int(_clip(inv["n_particles"] + 4, 16, 44))
        inv["iters"] = int(_clip(inv["iters"] + 8, 30, 90))
        notes.append("大跳变比例偏高，提升搜索粒度（particles/iters）")

    if result.accept_rate == result.accept_rate and result.accept_rate < 0.25:
        dataset["xyz_err_threshold_m"] = _clip(dataset["xyz_err_threshold_m"] + 0.0015, 0.01, 0.03)
        notes.append("接受率偏低，轻微放宽xyz_err_threshold防止采样效率过低")

    if len(history) >= 2:
        prev = history[-1]
        prev2 = history[-2]
        no_improve = (
            (prev.p95_20mm - result.p95_20mm) < 0.05
            and (prev2.p95_20mm - prev.p95_20mm) < 0.05
            and result.p95_20mm > 6.0
        )
        if no_improve:
            inv["n_particles"] = int(_clip(inv["n_particles"] + 4, 16, 44))
            inv["iters"] = int(_clip(inv["iters"] + 8, 30, 90))
            inv["n_restarts"] = int(_clip(inv["n_restarts"] + 1, 1, 4))
            inv["warm_start_particles"] = int(_clip(inv["warm_start_particles"] + 2, 4, 20))
            notes.append("连续两轮20mm p95改善不足，增强搜索容量")

    if not notes:
        notes.append("指标已改善但未过门限，保持参数小步前进")
        inv["select_w_delta_theta"] = _clip(inv["select_w_delta_theta"] + 5.0, 20.0, 220.0)
        inv["canonical_w_delta_theta"] = _clip(inv["canonical_w_delta_theta"] + 5.0, 20.0, 220.0)

    decision = "；".join(notes)
    return new_state, decision, notes


def run(
    base_config: Path,
    env_name: str,
    python_bin: str | None,
    max_attempts: int,
    num_samples: int,
    max_tried: int,
) -> int:
    base = _load_yaml(base_config)
    state = _state_from_cfg(base)
    if not bool(state["inverse_pso"].get("use_continuity_chain", False)):
        state["inverse_pso"]["use_continuity_chain"] = True
    state["parallel"]["workers"] = 1
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path("runs/acceptance") / f"inverse_5deg_tune_adaptive_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    summary_md = run_root / "summary.md"
    meta_json = run_root / "meta.json"
    decision_md = run_root / "decisions.md"

    meta = {
        "base_config": str(base_config),
        "env": env_name,
        "python_bin": python_bin,
        "max_attempts": max_attempts,
        "num_samples": num_samples,
        "max_tried": max_tried,
        "adaptive": True,
        "attempts": [],
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def py_runner() -> list[str]:
        if python_bin:
            return [python_bin]
        return ["conda", "run", "-n", env_name, "python"]

    history: list[AttemptResult] = []
    for aid in range(1, max_attempts + 1):
        cfg = deepcopy(base)
        patch = _patch_from_state(state)
        _deep_update(cfg, patch)

        out_dir = Path(f"data/tune_inverse_5deg_adaptive_attempt{aid:02d}")
        cfg["dataset"]["out_dir"] = str(out_dir)
        cfg["sampling"]["rng_seed"] = int(base["sampling"].get("rng_seed", 0)) + aid * 37
        cfg["pso"]["rng_seed"] = int(base["pso"].get("rng_seed", 0)) + aid * 101

        cfg_path = Path("configs/tuning") / f"robot_rods_only_5deg_inverse_2k_adaptive_attempt{aid:02d}.yaml"
        _dump_yaml(cfg_path, cfg)

        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        gate_json = run_root / f"attempt_{aid:02d}_gate.json"
        log_path = run_root / f"attempt_{aid:02d}.log"
        t0 = time.time()
        passed = False
        checks: dict[str, bool] = {}
        p95_10 = float("nan")
        p95_20 = float("nan")
        n10 = 0
        n20 = 0
        gt10_10 = float("nan")
        gt10_20 = float("nan")

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
            passed, checks, p95_10, p95_20, n10, n20, gt10_10, gt10_20 = _read_gate(gate_json)
        except subprocess.CalledProcessError:
            if gate_json.exists():
                passed, checks, p95_10, p95_20, n10, n20, gt10_10, gt10_20 = _read_gate(gate_json)

        elapsed_s = time.time() - t0
        accept_rate = _read_accept_rate(out_dir)
        result = AttemptResult(
            attempt_id=aid,
            config_path=cfg_path,
            dataset_dir=out_dir,
            gate_json=gate_json,
            passed=passed,
            checks=checks,
            p95_10mm=p95_10,
            p95_20mm=p95_20,
            pairs_10mm=n10,
            pairs_20mm=n20,
            ratio_gt10_10mm=gt10_10,
            ratio_gt10_20mm=gt10_20,
            accept_rate=accept_rate,
            elapsed_s=elapsed_s,
        )
        history.append(result)

        before_state = deepcopy(state)
        state, decision_text, notes = _adapt_state(state, result, history[:-1])
        _append_summary(summary_md, result, decision_text)
        _append_decision_log(decision_md, aid, before_state, state, notes)

        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        meta["attempts"].append(
            {
                "attempt": aid,
                "passed": passed,
                "checks": checks,
                "p95_10mm": p95_10,
                "p95_20mm": p95_20,
                "pairs_10mm": n10,
                "pairs_20mm": n20,
                "gt10_10mm": gt10_10,
                "gt10_20mm": gt10_20,
                "accept_rate": accept_rate,
                "elapsed_s": elapsed_s,
                "config": str(cfg_path),
                "dataset_dir": str(out_dir),
                "gate_json": str(gate_json),
                "decision": decision_text,
                "state_after": deepcopy(state),
            }
        )
        meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            f"[attempt {aid}] passed={passed} p95_10={p95_10:.4f} p95_20={p95_20:.4f} "
            f"pairs10={n10} pairs20={n20} accept_rate={accept_rate:.4f} decision={decision_text}"
        )

        if passed:
            return 0

    return 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-config",
        default="configs/robot_rods_only_5deg_inverse_2k_paper_tf_cpu_mt.yaml",
    )
    ap.add_argument("--env-name", default="quasi_exp")
    ap.add_argument("--python-bin", default="")
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--max-tried", type=int, default=15000)
    args = ap.parse_args()

    rc = run(
        base_config=Path(args.base_config),
        env_name=args.env_name,
        python_bin=(args.python_bin.strip() or None),
        max_attempts=int(args.max_attempts),
        num_samples=int(args.num_samples),
        max_tried=int(args.max_tried),
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
