from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import numpy as np

from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from quasi_exp.io import load_config, load_robot_inputs
from quasi_exp.model.sampling import beta_to_theta, validate_generation_strategy
from quasi_exp.model.kinematics import forward_kinematics
from quasi_exp.model.quasi_static import QuasiStaticModel
from quasi_exp.opt.pso_inverse import solve_inverse_joint_pso
from quasi_exp.opt.tension_labeler import solve_tension_label


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean(np.square(x))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    validate_generation_strategy(cfg)
    inputs = load_robot_inputs(cfg)
    model = QuasiStaticModel(cfg, inputs)

    mode = str(cfg.get("dataset", {}).get("mode", "forward")).strip().lower()
    if mode not in {"forward", "inverse_joint"}:
        raise SystemExit(f"Unsupported dataset.mode={mode}, expected forward|inverse_joint")

    rms_thresh = float(cfg["dataset"]["rms_rnorm_threshold"])
    xyz_thresh = float(cfg["dataset"].get("xyz_err_threshold_m", float("inf")))
    pso_cfg = cfg["pso"]
    base_pso_seed = int(pso_cfg.get("rng_seed", 0))
    inverse_pso_cfg = cfg.get("inverse_pso", {})
    beta_ranges = cfg["sampling"]["beta_ranges_rad"]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        t0 = time.time()
        task: dict | None = None
        try:
            task = json.loads(line)
            sample_id = int(task.get("sample_id", -1))
            sample_seed = int(task.get("seed", 0))
            pso_seed = int(task.get("pso_seed", base_pso_seed + sample_seed))
            extra_meta = {}
            for key in ["scan_axis", "scan_axis_idx", "scan_sign", "scan_level", "scan_angle_rad", "scan_angle_deg", "retry_count"]:
                if key in task:
                    extra_meta[key] = task[key]
            if mode == "forward":
                beta = np.asarray(task["beta6_rad"], dtype=float).reshape(6)
                theta_raw = beta_to_theta(beta)
                theta = theta_raw * model.theta_sign
                p_xyz, _T = forward_kinematics(
                    theta_raw,
                    inputs.lengths_m,
                    inputs.p_end_local_m,
                    theta_sign=model.theta_sign,
                )

                cache = model.build_cache(theta_raw)
                label = solve_tension_label(
                    model=model,
                    cache=cache,
                    pso_cfg=pso_cfg,
                    pso_seed=pso_seed,
                    rms_thresh=rms_thresh,
                )
                T_base_12 = label.T_base_12
                ok = bool(np.isfinite(p_xyz).all() and np.isfinite(T_base_12).all() and label.ok)

                out = {
                    "ok": ok,
                    "sample_id": sample_id,
                    "seed": sample_seed,
                    "beta6_rad": beta.tolist(),
                    "x_m": float(p_xyz[0]),
                    "y_m": float(p_xyz[1]),
                    "z_m": float(p_xyz[2]),
                    "theta_rad": theta.tolist(),
                    "tension_n": T_base_12.tolist(),
                    "meta": {
                        **label.meta,
                        "elapsed_s": float(time.time() - t0),
                        **extra_meta,
                    },
                }
            else:
                target_xyz = np.asarray(task["target_xyz_m"], dtype=float).reshape(3)
                res = solve_inverse_joint_pso(
                    model=model,
                    inputs=inputs,
                    xyz_target_m=target_xyz,
                    beta_ranges_rad=beta_ranges,
                    inverse_pso_cfg=inverse_pso_cfg,
                    tension_pso_cfg=pso_cfg,
                    rng_seed=pso_seed,
                )
                ok = bool(
                    np.isfinite(res.p_xyz_m).all()
                    and np.isfinite(res.theta_rad).all()
                    and np.isfinite(res.T_base_12).all()
                    and (res.rms_rnorm < rms_thresh)
                    and (res.xyz_err_m < xyz_thresh)
                )
                out = {
                    "ok": ok,
                    "sample_id": sample_id,
                    "seed": sample_seed,
                    "beta6_rad": res.beta6_rad.tolist(),
                    "x_m": float(res.p_xyz_m[0]),
                    "y_m": float(res.p_xyz_m[1]),
                    "z_m": float(res.p_xyz_m[2]),
                    "theta_rad": res.theta_rad.tolist(),
                    "tension_n": res.T_base_12.tolist(),
                    "meta": {
                        "rms_rnorm": float(res.rms_rnorm),
                        "mean_rnorm2": float(res.mean_rnorm2),
                        "max_tension": float(res.max_tension),
                        "best_cost": float(res.best_cost),
                        "iters_used": int(res.iters_used),
                        "evals": int(res.evals),
                        "pso_seed": int(pso_seed),
                        "case_flag_12": res.case_flag_12.astype(int).tolist(),
                        "target_x_m": float(res.target_xyz_m[0]),
                        "target_y_m": float(res.target_xyz_m[1]),
                        "target_z_m": float(res.target_xyz_m[2]),
                        "xyz_err_m": float(res.xyz_err_m),
                        "elapsed_s": float(time.time() - t0),
                    },
                }
            sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except Exception as e:  # noqa: BLE001
            err = {
                "ok": False,
                "sample_id": int(task.get("sample_id", -1)) if isinstance(task, dict) else -1,
                "seed": int(task.get("seed", 0)) if isinstance(task, dict) else 0,
                "retry_count": int(task.get("retry_count", 0)) if isinstance(task, dict) else 0,
                "scan_axis": str(task.get("scan_axis", "")) if isinstance(task, dict) and ("scan_axis" in task) else "",
                "scan_level": int(task.get("scan_level", 0)) if isinstance(task, dict) and ("scan_level" in task) else 0,
                "error": repr(e),
                "traceback": traceback.format_exc(limit=5),
            }
            sys.stdout.write(json.dumps(err, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
