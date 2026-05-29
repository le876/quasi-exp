from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobotInputs:
    lengths_m: np.ndarray  # shape (31,) for l0..l30
    holes_local_m: np.ndarray  # shape (31, 12, 2, 4) disks 0..30, side 0=prox 1=dist, homogeneous
    masses_kg: np.ndarray  # shape (31,) disks 0..30 (0 unused -> 0)
    com_local_m: np.ndarray  # shape (31,4) homogeneous, in each disk frame
    E_pa: np.ndarray  # shape (31,) disks 0..30 (0 unused -> 0)
    Iz_m4: np.ndarray  # shape (31,) disks 0..30 (0 unused -> 0)
    p_end_local_m: np.ndarray  # shape (4,)


_SIDE_TO_IDX = {"prox": 0, "dist": 1}


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def load_robot_inputs(cfg: dict) -> RobotInputs:
    paths = cfg["paths"]
    kD = int(cfg["robot"]["kD"])

    lengths_df = _read_csv(paths["lengths_csv"])
    if set(lengths_df.columns) != {"name", "value_m"}:
        raise ValueError("lengths.csv must have columns: name,value_m")
    lengths_map = dict(zip(lengths_df["name"].astype(str), lengths_df["value_m"].astype(float)))
    lengths = np.zeros(kD + 1, dtype=float)
    for i in range(kD + 1):
        key = f"l{i}"
        if key not in lengths_map:
            raise ValueError(f"Missing {key} in lengths.csv")
        lengths[i] = float(lengths_map[key])

    holes_df = _read_csv(paths["holes_csv"])
    req_cols = {"disk_idx", "hole_idx", "side", "x_m", "y_m", "z_m"}
    if set(holes_df.columns) != req_cols:
        raise ValueError(f"holes.csv must have columns: {sorted(req_cols)}")
    holes = np.full((kD + 1, 12, 2, 4), np.nan, dtype=float)
    holes[:, :, :, 3] = 1.0
    for row in holes_df.itertuples(index=False):
        disk_idx = int(row.disk_idx)
        hole_idx = int(row.hole_idx)
        side = str(row.side)
        if disk_idx < 0 or disk_idx > kD:
            raise ValueError(f"holes.csv disk_idx out of range: {disk_idx}")
        if hole_idx < 1 or hole_idx > 12:
            raise ValueError(f"holes.csv hole_idx out of range: {hole_idx}")
        if side not in _SIDE_TO_IDX:
            raise ValueError(f"holes.csv side must be prox/dist, got: {side}")
        sidx = _SIDE_TO_IDX[side]
        holes[disk_idx, hole_idx - 1, sidx, 0] = float(row.x_m)
        holes[disk_idx, hole_idx - 1, sidx, 1] = float(row.y_m)
        holes[disk_idx, hole_idx - 1, sidx, 2] = float(row.z_m)

    if np.isnan(holes[0, :, 1, :3]).any():
        raise ValueError("holes.csv must include disk_idx=0, side=dist for all 12 holes (h'_0,j)")

    missing_disks: list[int] = []
    for disk_idx in range(1, kD + 1):
        prox_any_nan = bool(np.isnan(holes[disk_idx, :, 0, :3]).any())
        dist_any_nan = bool(np.isnan(holes[disk_idx, :, 1, :3]).any())
        prox_all_nan = bool(np.isnan(holes[disk_idx, :, 0, :3]).all())
        dist_all_nan = bool(np.isnan(holes[disk_idx, :, 1, :3]).all())

        # 允许“整盘缺失”时用模板盘复制（论文中盘设计参数相同）
        if prox_all_nan and dist_all_nan:
            missing_disks.append(disk_idx)
            continue

        if prox_any_nan:
            raise ValueError(f"holes.csv missing prox holes for disk {disk_idx} (partial)")
        if dist_any_nan:
            raise ValueError(f"holes.csv missing dist holes for disk {disk_idx} (partial)")

    if missing_disks:
        # 选择一个完整盘作为模板（优先 disk 1）
        template = None
        for cand in range(1, kD + 1):
            if not np.isnan(holes[cand, :, :, :3]).any():
                template = cand
                break
        if template is None:
            raise ValueError("holes.csv: no fully-defined disk to use as template")
        for disk_idx in missing_disks:
            holes[disk_idx] = holes[template]

    disks_df = _read_csv(paths["disks_csv"])
    req_cols = {"disk_idx", "mass_kg", "com_x_m", "com_y_m", "com_z_m"}
    if set(disks_df.columns) != req_cols:
        raise ValueError(f"disks.csv must have columns: {sorted(req_cols)}")
    masses = np.zeros(kD + 1, dtype=float)
    com = np.zeros((kD + 1, 4), dtype=float)
    com[:, 3] = 1.0
    for row in disks_df.itertuples(index=False):
        disk_idx = int(row.disk_idx)
        if disk_idx < 1 or disk_idx > kD:
            raise ValueError(f"disks.csv disk_idx out of range: {disk_idx}")
        masses[disk_idx] = float(row.mass_kg)
        com[disk_idx, 0] = float(row.com_x_m)
        com[disk_idx, 1] = float(row.com_y_m)
        com[disk_idx, 2] = float(row.com_z_m)
    if (masses[1:] == 0).any():
        missing = (np.where(masses[1:] == 0)[0] + 1).tolist()
        # 允许“整盘缺失”时用模板盘复制（论文中盘设计参数相同）
        template = None
        for cand in range(1, kD + 1):
            if masses[cand] > 0:
                template = cand
                break
        if template is None:
            raise ValueError("disks.csv: no disk with non-zero mass to use as template")
        for disk_idx in missing:
            masses[disk_idx] = masses[template]
            com[disk_idx] = com[template]

    material_df = _read_csv(paths["material_csv"])
    req_cols = {"disk_idx", "E_pa", "Iz_m4"}
    if set(material_df.columns) != req_cols:
        raise ValueError(f"material.csv must have columns: {sorted(req_cols)}")
    E = np.zeros(kD + 1, dtype=float)
    Iz = np.zeros(kD + 1, dtype=float)
    for row in material_df.itertuples(index=False):
        disk_idx = int(row.disk_idx)
        if disk_idx < 1 or disk_idx > kD:
            raise ValueError(f"material.csv disk_idx out of range: {disk_idx}")
        E[disk_idx] = float(row.E_pa)
        Iz[disk_idx] = float(row.Iz_m4)
    if (E[1:] == 0).any() or (Iz[1:] == 0).any():
        missing = sorted(set((np.where(E[1:] == 0)[0] + 1).tolist() + (np.where(Iz[1:] == 0)[0] + 1).tolist()))
        template = None
        for cand in range(1, kD + 1):
            if E[cand] > 0 and Iz[cand] > 0:
                template = cand
                break
        if template is None:
            raise ValueError("material.csv: no disk with non-zero E and Iz to use as template")
        for disk_idx in missing:
            E[disk_idx] = E[template]
            Iz[disk_idx] = Iz[template]

    p_end = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    end_path = Path(paths.get("end_effector_csv", ""))
    if end_path and end_path.exists():
        end_df = _read_csv(end_path)
        req_cols = {"p_end_x_m", "p_end_y_m", "p_end_z_m"}
        if set(end_df.columns) != req_cols or len(end_df) != 1:
            raise ValueError("end_effector.csv must have 1 row with columns: p_end_x_m,p_end_y_m,p_end_z_m")
        p_end[0] = float(end_df.iloc[0]["p_end_x_m"])
        p_end[1] = float(end_df.iloc[0]["p_end_y_m"])
        p_end[2] = float(end_df.iloc[0]["p_end_z_m"])

    return RobotInputs(
        lengths_m=lengths,
        holes_local_m=holes,
        masses_kg=masses,
        com_local_m=com,
        E_pa=E,
        Iz_m4=Iz,
        p_end_local_m=p_end,
    )
