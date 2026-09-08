from __future__ import annotations

import numpy as np


def holes_in_base(T_0_from_i: np.ndarray, holes_local_m: np.ndarray) -> np.ndarray:
    """
    输入：
      - T_0_from_i: shape (kD+1,4,4) i→0
      - holes_local_m: shape (kD+1,12,2,4) (disk0..kD)

    输出：
      - holes_0_m: shape (kD+1,12,2,4) 齐次点在 {0}
    """
    T_0_from_i = np.asarray(T_0_from_i, dtype=float)
    holes_local_m = np.asarray(holes_local_m, dtype=float)
    kD = T_0_from_i.shape[0] - 1

    holes_0 = np.empty_like(holes_local_m, dtype=float)
    for i in range(kD + 1):
        Ti = T_0_from_i[i]
        pts = holes_local_m[i].reshape(-1, 4).T  # (4, 24)
        holes_0[i] = (Ti @ pts).T.reshape(12, 2, 4)
    return holes_0


def cable_length_j(holes_0_m: np.ndarray, j_1based: int, end_disk: int) -> float:
    j = int(j_1based) - 1
    if j < 0 or j >= 12:
        raise ValueError("j must be 1..12")
    if end_disk < 1:
        return 0.0

    L = 0.0
    for i in range(1, end_disk + 1):
        h_i_prox = holes_0_m[i, j, 0, :3]
        h_i_dist = holes_0_m[i, j, 1, :3]
        h_im1_dist = holes_0_m[i - 1, j, 1, :3]
        L += float(np.linalg.norm(h_i_dist - h_i_prox))
        L += float(np.linalg.norm(h_i_prox - h_im1_dist))
    return L


def cable_lengths_all(holes_0_m: np.ndarray, end_disk_by_j: dict[int, int]) -> np.ndarray:
    L = np.zeros(12, dtype=float)
    for j in range(1, 13):
        end_disk = int(end_disk_by_j[j])
        L[j - 1] = cable_length_j(holes_0_m, j, end_disk=end_disk)
    return L

