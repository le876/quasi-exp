from __future__ import annotations

import numpy as np


def _ratio_case1(theta_i: float, theta_ip1: float, mu_cable: float) -> float:
    num = np.cos(theta_i / 2.0) + mu_cable * np.sin(theta_i / 2.0)
    den = np.cos(theta_ip1 / 2.0) - mu_cable * np.sin(theta_ip1 / 2.0)
    return float(num / den)


def _ratio_case2(theta_i: float, theta_ip1: float, mu_cable: float) -> float:
    num = np.cos(theta_i / 2.0) - mu_cable * np.sin(theta_i / 2.0)
    den = np.cos(theta_ip1 / 2.0) + mu_cable * np.sin(theta_ip1 / 2.0)
    return float(num / den)


def transmit_tensions(
    theta_rad: np.ndarray,
    T_base_12: np.ndarray,
    mu_cable: float,
    end_disk_by_j: dict[int, int],
    L0_12: np.ndarray,
    L_12: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    论文 Eq.(42) 张力传递。

    返回：
      - F_ct: shape (kD+1,12) 其中 F_ct[i,j] 表示 F_CT_{i,j}（i=0..kD）
      - case_flag: shape (12,) 1=Case1, 2=Case2（按 ΔL_j 选择）
    """
    theta = np.asarray(theta_rad, dtype=float).reshape(-1)
    kD = theta.shape[0]
    T_base = np.asarray(T_base_12, dtype=float).reshape(12)
    L0 = np.asarray(L0_12, dtype=float).reshape(12)
    L = np.asarray(L_12, dtype=float).reshape(12)

    delta = L - L0
    case_flag = np.where(delta > 0, 1, 2).astype(int)

    F = np.zeros((kD + 1, 12), dtype=float)
    F[0] = T_base
    theta_ext = np.zeros(kD + 1, dtype=float)
    theta_ext[1:] = theta

    for j1 in range(1, 13):
        end_disk = int(end_disk_by_j[j1])
        cflag = int(case_flag[j1 - 1])
        for i in range(0, min(end_disk, kD)):
            th_i = float(theta_ext[i])
            th_ip1 = float(theta_ext[i + 1])
            if cflag == 1:
                ratio = _ratio_case1(th_i, th_ip1, mu_cable)
            else:
                ratio = _ratio_case2(th_i, th_ip1, mu_cable)
            if not np.isfinite(ratio):
                F[i + 1, j1 - 1] = np.inf
            else:
                F[i + 1, j1 - 1] = F[i, j1 - 1] * ratio
        if end_disk < kD:
            F[end_disk + 1 :, j1 - 1] = 0.0

    return F, case_flag

