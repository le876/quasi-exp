from __future__ import annotations

import numpy as np


def _alpha_rad(i: int) -> float:
    if i == 1:
        return 0.0
    return (np.pi / 2.0) if (i % 2 == 1) else (-np.pi / 2.0)


def dh_transform_i_to_im1(alpha_rad: float, a_m: float, theta_rad: float, d_m: float) -> np.ndarray:
    """
    D-H 变换矩阵（与论文 Eq.(1) 形式一致）。

    约定：返回 T_i^{i-1}，将 {i} 中的点/向量（齐次点）映射到 {i-1}。
    """
    ca = float(np.cos(alpha_rad))
    sa = float(np.sin(alpha_rad))
    ct = float(np.cos(theta_rad))
    st = float(np.sin(theta_rad))

    T = np.array(
        [
            [ct, -st, 0.0, a_m],
            [st * ca, ct * ca, -sa, -d_m * sa],
            [st * sa, ct * sa, ca, d_m * ca],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti


def forward_kinematics(
    theta_rad: np.ndarray,
    lengths_m: np.ndarray,
    p_end_local_m: np.ndarray | None = None,
    theta_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    输入：
      - theta_rad: shape (kD,)  (θ1..θkD)
      - lengths_m: shape (kD+1,) (l0..lkD)
      - p_end_local_m: shape (4,), 默认为 [0,0,0,1]

    输出：
      - p_xyz: shape (3,) 末端点在 {0} 的坐标
      - T_0_from_i: shape (kD+1,4,4) 其中 T_0_from_i[i] = Π_i^0 T (i→0)
    """
    theta_rad = np.asarray(theta_rad, dtype=float).reshape(-1) * float(theta_sign)
    lengths_m = np.asarray(lengths_m, dtype=float).reshape(-1)

    kD = theta_rad.shape[0]
    if lengths_m.shape[0] != kD + 1:
        raise ValueError(f"lengths_m must have shape ({kD+1},), got {lengths_m.shape}")

    if p_end_local_m is None:
        p_end_local_m = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    p_end_local_m = np.asarray(p_end_local_m, dtype=float).reshape(4)

    T_0_from_i = np.zeros((kD + 1, 4, 4), dtype=float)
    T_0_from_i[0] = np.eye(4, dtype=float)

    for i in range(1, kD + 1):
        alpha = _alpha_rad(i)
        a = float(lengths_m[i - 1])
        theta = float(theta_rad[i - 1])
        d = 0.0
        T_i_to_im1 = dh_transform_i_to_im1(alpha, a, theta, d)
        T_0_from_i[i] = T_0_from_i[i - 1] @ T_i_to_im1

    p0 = T_0_from_i[kD] @ p_end_local_m
    return p0[:3].copy(), T_0_from_i
