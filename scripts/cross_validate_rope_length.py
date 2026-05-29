from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from quasi_exp.io import load_config, load_robot_inputs  # noqa: E402
from quasi_exp.model.cable_geometry import cable_lengths_all, holes_in_base  # noqa: E402
from quasi_exp.model.kinematics import forward_kinematics  # noqa: E402
from quasi_exp.model.sampling import beta_to_theta  # noqa: E402


def _theta2rope_mm(theta6: np.ndarray) -> np.ndarray:
    """
    复刻 auto_controller_1.cpp 中的 theta2rope（单位：mm）。
    仅用于和 Eq.(5) 绳长变化做趋势/量级交叉验证。
    """
    theta = np.asarray(theta6, dtype=float).reshape(6)

    Gap_1 = np.zeros(12, dtype=float)
    Gap_2 = np.array([29.63, 29.63, 29.09, 29.09, 29.63, 29.63, 29.63, 29.63, 29.09, 29.09, 29.63, 29.63], dtype=float)
    Gap_3 = np.zeros(12, dtype=float)
    Gap_4 = np.array([29.63, 29.63, 29.09, 29.09, 29.63, 29.63, 29.63, 29.63, 29.09, 29.09, 29.63, 29.63], dtype=float)
    Gap_base = np.array([24.57, 24.57, 26.11, 26.11, 28.12, 28.12, 28.12, 28.12, 26.11, 26.11, 24.57, 24.57], dtype=float)

    Rope_length = np.zeros(12, dtype=float)

    beta = np.pi * 50.0 / 180.0
    R_Gap_1 = np.array([23.03, 23.03, 18.82, 18.82, 13.33, 13.33, 13.33, 13.33, 18.82, 18.82, 23.03, 23.03], dtype=float)
    R_Gap_3 = np.array([13.33, 13.33, 18.82, 18.82, 23.03, 23.03, 23.03, 23.03, 18.82, 18.82, 13.33, 13.33], dtype=float)

    for _N in range(5):
        for i in range(12):
            # 第一节（所有 12 根绳都经过）
            if i > 5:
                Gap_1[i] = np.sin((beta - theta[0] + np.pi / 10.0) / 2.0) * 2.0 * R_Gap_1[i]
            else:
                Gap_1[i] = np.sin((beta + theta[0] - np.pi / 10.0) / 2.0) * 2.0 * R_Gap_1[i]

            if (i % 2) == 0:
                Gap_3[i] = np.sin((beta + theta[1]) / 2.0) * 2.0 * R_Gap_3[i]
            else:
                Gap_3[i] = np.sin((beta - theta[1]) / 2.0) * 2.0 * R_Gap_3[i]

            Rope_length[i] += Gap_1[i] + Gap_2[i] + Gap_3[i] + Gap_4[i] + Gap_base[i]

            # 第二节（排除 1,2,11,12 => i!=0,1,10,11）
            if i not in (0, 1, 10, 11):
                if i > 5:
                    Gap_1[i] = np.sin((beta - theta[2]) / 2.0) * 2.0 * R_Gap_1[i]
                else:
                    Gap_1[i] = np.sin((beta + theta[2]) / 2.0) * 2.0 * R_Gap_1[i]

                if (i % 2) == 0:
                    Gap_3[i] = np.sin((beta + theta[3]) / 2.0) * 2.0 * R_Gap_3[i]
                else:
                    Gap_3[i] = np.sin((beta - theta[3]) / 2.0) * 2.0 * R_Gap_3[i]

                Rope_length[i] += Gap_1[i] + Gap_2[i] + Gap_3[i] + Gap_4[i]

            # 第三节（再排除 3,4,9,10 => i!=2,3,8,9）
            if i not in (0, 1, 10, 11, 2, 3, 8, 9):
                if i > 5:
                    Gap_1[i] = np.sin((beta - theta[4]) / 2.0) * 2.0 * R_Gap_1[i]
                else:
                    Gap_1[i] = np.sin((beta + theta[4]) / 2.0) * 2.0 * R_Gap_1[i]

                if (i % 2) == 0:
                    Gap_3[i] = np.sin((beta + theta[5]) / 2.0) * 2.0 * R_Gap_3[i]
                else:
                    Gap_3[i] = np.sin((beta - theta[5]) / 2.0) * 2.0 * R_Gap_3[i]

                Rope_length[i] += Gap_1[i] + Gap_2[i] + Gap_3[i] + Gap_4[i]

    Rope_length = Rope_length - Gap_4
    return Rope_length


def _rope_length_eq5_mm(cfg: dict, inputs, theta30: np.ndarray) -> np.ndarray:
    end_disk_by_j = {int(k): int(v) for k, v in cfg["cables"]["end_disk_by_j"].items()}
    theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", 1.0))
    _p, T_0_from_i = forward_kinematics(theta30, inputs.lengths_m, inputs.p_end_local_m, theta_sign=theta_sign)
    holes_0 = holes_in_base(T_0_from_i, inputs.holes_local_m)
    L_m = cable_lengths_all(holes_0, end_disk_by_j)
    return L_m * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robot.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--scale", type=float, default=0.2, help="随机 theta6 的幅度缩放（rad）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    inputs = load_robot_inputs(cfg)

    rng = np.random.default_rng(args.seed)
    theta6_zero = np.zeros(6, dtype=float)

    beta0 = theta6_zero
    theta30_0 = beta_to_theta(beta0)
    L0_eq5 = _rope_length_eq5_mm(cfg, inputs, theta30_0)
    R0 = _theta2rope_mm(theta6_zero)

    maes = []
    corrs = []
    for _ in range(args.trials):
        theta6 = rng.uniform(-1.0, 1.0, size=(6,)) * float(args.scale)
        theta30 = beta_to_theta(theta6)

        L_eq5 = _rope_length_eq5_mm(cfg, inputs, theta30)
        R = _theta2rope_mm(theta6)

        dL = L_eq5 - L0_eq5
        dR = R - R0

        mae = float(np.mean(np.abs(dL - dR)))
        maes.append(mae)
        corr = float(np.corrcoef(dL, dR)[0, 1]) if np.std(dL) > 1e-9 and np.std(dR) > 1e-9 else float("nan")
        corrs.append(corr)

    print("Cross validation (delta rope length, mm):")
    print(f"- trials={args.trials}, scale={args.scale} rad")
    print(f"- mean(MAE_mm)={float(np.nanmean(maes)):.3f}")
    print(f"- mean(corr)={float(np.nanmean(corrs)):.3f}")
    print("If corr is low (<0.8), check hole index mapping and odd/even templates.")


if __name__ == "__main__":
    main()
