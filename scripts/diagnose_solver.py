from __future__ import annotations

import argparse
import json
import time

import numpy as np

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from quasi_exp.model.sampling import beta_to_theta  # noqa: E402
from quasi_exp.model.kinematics import forward_kinematics  # noqa: E402
from quasi_exp.model.quasi_static import QuasiStaticModel  # noqa: E402
from quasi_exp.opt.pso import solve_tensions_pso  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robot.yaml")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--scale", type=float, default=1.0, help="按 sampling.beta_ranges_rad 采样；scale<1 会缩小范围")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pso-iters", type=int, default=None)
    ap.add_argument("--pso-particles", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    inputs = load_robot_inputs(cfg)
    model = QuasiStaticModel(cfg, inputs)

    beta_ranges = cfg["sampling"]["beta_ranges_rad"]
    rng = np.random.default_rng(args.seed)
    pso_cfg = cfg["pso"]
    if args.pso_iters is not None:
        pso_cfg = dict(pso_cfg)
        pso_cfg["iters"] = int(args.pso_iters)
    if args.pso_particles is not None:
        pso_cfg = dict(pso_cfg)
        pso_cfg["n_particles"] = int(args.pso_particles)
    base_pso_seed = int(pso_cfg.get("rng_seed", 0))

    rms = []
    max_t = []
    iters = []
    costs = []
    t0 = time.time()

    for k in range(args.samples):
        beta = np.zeros(6, dtype=float)
        for i in range(6):
            lo, hi = beta_ranges[f"beta{i+1}"]
            mid = 0.5 * (float(lo) + float(hi))
            span = 0.5 * (float(hi) - float(lo)) * float(args.scale)
            beta[i] = rng.uniform(mid - span, mid + span)

        theta_raw = beta_to_theta(beta)
        cache = model.build_cache(theta_raw)
        res = solve_tensions_pso(model, cache, pso_cfg=pso_cfg, rng_seed=base_pso_seed + k + 1)
        rnorm, _ = model.residual_norm(cache, res.T_base_12)
        rms_rnorm = float(np.sqrt(np.mean(np.square(rnorm))))

        rms.append(rms_rnorm)
        max_t.append(float(res.max_tension))
        iters.append(int(res.iters_used))
        costs.append(float(res.best_cost))

    def q(arr, p):
        return float(np.quantile(np.asarray(arr, dtype=float), p))

    report = {
        "samples": args.samples,
        "elapsed_s": float(time.time() - t0),
        "rms_rnorm": {"min": min(rms), "p50": q(rms, 0.5), "p90": q(rms, 0.9), "max": max(rms)},
        "max_tension": {"p50": q(max_t, 0.5), "p90": q(max_t, 0.9), "max": max(max_t)},
        "iters_used": {"p50": q(iters, 0.5), "p90": q(iters, 0.9), "max": max(iters)},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
