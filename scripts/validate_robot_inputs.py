from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from quasi_exp.io import load_config, load_robot_inputs
from quasi_exp.model.kinematics import forward_kinematics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robot.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    inputs = load_robot_inputs(cfg)

    kD = int(cfg["robot"]["kD"])
    theta0 = np.zeros(kD, dtype=float)
    p_xyz, T_0_from_i = forward_kinematics(theta0, inputs.lengths_m, inputs.p_end_local_m)

    # 直线姿态：主要检查 x 方向长度是否合理
    expected_x = float(np.sum(inputs.lengths_m[:kD]))
    print("OK: inputs loaded")
    print(f"- kD={kD}")
    print(f"- total_length_chain_m={expected_x:.6f}")
    print(f"- p_xyz(theta=0)={p_xyz.tolist()}")
    print(f"- p_end_local={inputs.p_end_local_m.tolist()}")

    R = T_0_from_i[kD][:3, :3]
    ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    print(f"- R_orthonormal_max_err={ortho_err:.3e}")

    if not np.isfinite(p_xyz).all():
        raise SystemExit("FAIL: p_xyz contains NaN/Inf")
    if abs(p_xyz[0] - expected_x - float(inputs.p_end_local_m[0])) > 1e-6:
        raise SystemExit(
            "FAIL: straight pose x mismatch. "
            "Check DH convention, lengths.csv units, and end_effector.csv."
        )

    print("PASS")


if __name__ == "__main__":
    main()
