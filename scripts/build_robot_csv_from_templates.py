from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _mm(x: float) -> float:
    return float(x) / 1000.0


def _parse_xyz(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("expected x,y,z")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


def _com_sw_to_dh_mm(com_sw_mm: np.ndarray) -> np.ndarray:
    """
    将 SolidWorks（你当前报告使用的坐标系 S）下的 COM 转成 DH 坐标系 D 的分量。

    你给出的轴关系：
      z_S = x_DH
      y_S = -z_DH
      x_S = -y_DH

    因此对坐标分量：
      x_DH = z_S
      y_DH = -x_S
      z_DH = -y_S
    """
    x_s, y_s, z_s = float(com_sw_mm[0]), float(com_sw_mm[1]), float(com_sw_mm[2])
    return np.array([z_s, -x_s, -y_s], dtype=float)


def _parse_yz_list(s: str) -> list[tuple[float, float]]:
    """
    解析形如 "y1,z1;y2,z2;..." 的 mm 列表（逗号分隔 y/z）。

    注意：在某些命令执行器/沙箱里，字符 ';' / '|' 可能被当成“命令分隔符”而被提前拆分。
    因此这里也支持用 '@' 分隔点： "y1,z1@y2,z2@..."
    """
    pts: list[tuple[float, float]] = []
    for item in re.split(r"[;|@]", s):
        item = item.strip()
        if not item:
            continue
        yz = [p.strip() for p in item.split(",")]
        if len(yz) != 2:
            raise ValueError(f"bad yz item: {item!r}")
        pts.append((float(yz[0]), float(yz[1])))
    if not pts:
        raise ValueError("empty rod yz list")
    return pts


def _circle_I_centroid_m4(d_m: float) -> float:
    return float(np.pi * d_m**4 / 64.0)


def _circle_area_m2(d_m: float) -> float:
    return float(np.pi * d_m**2 / 4.0)


def _equiv_Iz_for_rods_m4(d_m: float, yz_m: list[tuple[float, float]], about: str) -> tuple[float, float]:
    """
    计算多根平行圆杆组成的等效二次矩（仅 rods 贡献），返回 (Iz, Iy)。

    - about="origin": 以盘坐标系原点为参考轴（z 轴用 y 距离，y 轴用 z 距离）
    - about="centroid": 以 rods 截面形心为参考（对称性差时更符合梁理论的中性轴）
    """
    I0 = _circle_I_centroid_m4(d_m)
    A = _circle_area_m2(d_m)
    ys = np.array([p[0] for p in yz_m], dtype=float)
    zs = np.array([p[1] for p in yz_m], dtype=float)
    if about == "centroid":
        ys = ys - float(np.mean(ys))
        zs = zs - float(np.mean(zs))
    Iz = float(np.sum(I0 + A * ys**2))
    Iy = float(np.sum(I0 + A * zs**2))
    return Iz, Iy


def _four_holes_from_left_upper_odd(x: float, y: float, z: float) -> dict[str, np.ndarray]:
    """
    以“左上”点为基准生成 4 个孔位（同一 x），按奇数圆盘定义：
      - right_upper: z 变号
      - left_lower: y 变号
      - right_lower: y/z 都变号
    """
    yy = float(y)
    zz = float(z)
    return {
        "left_upper": np.array([x, yy, zz], dtype=float),
        "right_upper": np.array([x, yy, -zz], dtype=float),
        "left_lower": np.array([x, -yy, zz], dtype=float),
        "right_lower": np.array([x, -yy, -zz], dtype=float),
    }


def _four_holes_from_left_upper_even(x: float, y: float, z: float) -> dict[str, np.ndarray]:
    """
    以“左上”点为基准生成 4 个孔位（同一 x），按偶数圆盘定义（来自 DH-参数.md / one-segment 模型）：
      - right_upper: y 变号
      - left_lower: z 变号
      - right_lower: y/z 都变号
    """
    yy = float(y)
    zz = float(z)
    return {
        "left_upper": np.array([x, yy, zz], dtype=float),
        "right_upper": np.array([x, -yy, zz], dtype=float),
        "left_lower": np.array([x, yy, -zz], dtype=float),
        "right_lower": np.array([x, -yy, -zz], dtype=float),
    }


def _swap_yz(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).copy()
    p[1], p[2] = p[2], p[1]
    return p


def _odd_to_even_from_group1(p_odd: np.ndarray, delta_x_axis_mm: float) -> np.ndarray:
    """
    从 group1(第一关节) odd/even 模板观察到的近似变换：
      x_even ≈ x_odd - delta
      y_even ≈ -z_odd
      z_even ≈  y_odd

    用于在缺少 even 模板时派生（比如第二/第三关节孔位）。
    """
    p = np.asarray(p_odd, dtype=float)
    return np.array([p[0] - delta_x_axis_mm, -p[2], p[1]], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/robot/generated")
    ap.add_argument(
        "--disk-mass-kg",
        type=float,
        default=0.125,
        help="普通关节圆盘质量（kg），用于生成 disks.csv 模板",
    )
    ap.add_argument(
        "--disk-com-sw-mm",
        type=str,
        default="-0.601,-0.754,11.261",
        help="普通关节圆盘 COM 在 SolidWorks 报告坐标系下的 (x,y,z) mm，逗号分隔",
    )
    ap.add_argument(
        "--end-disk-mass-kg",
        type=float,
        default=None,
        help="末端圆盘质量（kg）。不填则默认与普通盘相同。",
    )
    ap.add_argument(
        "--end-disk-com-sw-mm",
        type=str,
        default=None,
        help="末端圆盘 COM 在 SolidWorks 报告坐标系下的 (x,y,z) mm。只在提供 end-disk-mass-kg 或本项时生效。",
    )
    ap.add_argument(
        "--E-pa",
        type=float,
        default=None,
        help="Nitinol 杆弹性模量 E（Pa），用于生成 material.csv。缺省则写 0（需要你后续补齐）。",
    )
    ap.add_argument(
        "--Iz-m4",
        type=float,
        default=None,
        help="Nitinol 杆截面二次矩 Iz（m^4），用于生成 material.csv。缺省则写 0（需要你后续补齐）。",
    )
    ap.add_argument(
        "--rod-d-mm",
        type=float,
        default=1.6,
        help="Nitinol 杆直径（mm）。用于在未显式指定 --Iz-m4 时估算 Iz。",
    )
    ap.add_argument(
        "--rod-yz-mm",
        type=str,
        default="-10.08,-13.08@16.08,-13.08@10.08,13.08@16.08,13.08",
        help='4 根杆孔在盘坐标系下的 (y,z) mm 列表：用 ";" / "|" / "@" 分隔点（推荐 "@"），例如 "y1,z1@y2,z2@..."，用于估算 Iz/Iy。',
    )
    ap.add_argument(
        "--Iz-about",
        choices=["origin", "centroid"],
        default="origin",
        help="计算等效 Iz/Iy 时以盘原点还是 rods 截面形心为参考。",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== User-provided / derived templates (mm) =====
    # l0=0, l1..l29=40mm, l30=55.498mm (轴向末端长度)
    l0 = 0.0
    l_regular = 40.0
    l_last = 55.498

    # 末端点：^{30}p_end = [l_last, 0, 0]
    p_end = np.array([_mm(l_last), 0.0, 0.0], dtype=float)

    # 由第一关节 odd/even 的 x 差得到“两个转轴中心沿 x 的偏置”（mm）
    # odd input x=7.275, even input x=4.211 => delta=3.064
    delta_x_axis = 7.275 - 4.211

    # Segment group templates (odd disk) in disk local frame {i} (mm)
    # group1 (第一关节4根绳)：
    # - odd/even 都有模板（来源：DH-参数.md / 你提供的 ports.odd/ports.even）
    g1_odd_in = np.array([7.275, 19.987, -11.500], dtype=float)
    g1_odd_out_x = 35.789

    # group2: from provided "第二关节第一圆盘左上孔"
    g2_odd_in = np.array([5.944311, 16.263456, -16.263456], dtype=float)
    g2_odd_out_x = 34.080586

    # group3 (第三关节4根绳)：你提供的是第一圆盘（disk_idx=1, odd）上的孔位
    g3_odd_in = np.array([4.210555, 11.500000, -19.918584], dtype=float)
    g3_odd_out_x = 32.725331

    # group1 even: from ports.even (keep input yz; output yz will be corrected to equal input yz)
    g1_even_in = np.array([4.211, 11.568, 19.919], dtype=float)
    g1_even_out_x = 32.725

    # group2 even: not provided; derive using group1 odd->even transform
    g2_even_in = _odd_to_even_from_group1(g2_odd_in, delta_x_axis_mm=delta_x_axis)
    g2_even_out_x = g2_odd_out_x - delta_x_axis

    # group3 even: not provided; derive via odd->even transform
    g3_even_in = _odd_to_even_from_group1(g3_odd_in, delta_x_axis_mm=delta_x_axis)
    g3_even_out_x = g3_odd_out_x - delta_x_axis
    # ===== Base disk (disk_idx=0) dist holes for each group (mm) =====
    base_g1 = np.array([-9.28188, 19.918584, -11.500000], dtype=float)
    base_g2 = np.array([-7.583774, 16.263456, -16.263456], dtype=float)
    base_g3 = np.array([-5.362538, 11.500000, 19.918584], dtype=float)

    # Cable index mapping (1..12) to hole quadrant names per group
    # group1 cables: 1,2,11,12 ; group2: 3,4,9,10 ; group3: 5,6,7,8
    group_to_cables = {
        1: [(1, "left_upper"), (2, "right_upper"), (11, "left_lower"), (12, "right_lower")],
        2: [(3, "left_upper"), (4, "right_upper"), (9, "left_lower"), (10, "right_lower")],
        3: [(5, "left_upper"), (6, "right_upper"), (7, "left_lower"), (8, "right_lower")],
    }

    # Build per-parity hole sets (prox=input, dist=output), with correction: output yz == input yz.
    def build_group_holes(
        in_left_upper: np.ndarray, out_x: float, parity: str
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if parity == "odd":
            prox = _four_holes_from_left_upper_odd(in_left_upper[0], in_left_upper[1], in_left_upper[2])
        elif parity == "even":
            prox = _four_holes_from_left_upper_even(in_left_upper[0], in_left_upper[1], in_left_upper[2])
        else:
            raise ValueError("parity must be odd/even")
        dist = {k: v.copy() for k, v in prox.items()}
        for k in dist:
            dist[k][0] = float(out_x)
        return prox, dist

    g1_odd_prox, g1_odd_dist = build_group_holes(g1_odd_in, g1_odd_out_x, parity="odd")
    g1_even_prox, g1_even_dist = build_group_holes(g1_even_in, g1_even_out_x, parity="even")
    g2_odd_prox, g2_odd_dist = build_group_holes(g2_odd_in, g2_odd_out_x, parity="odd")
    g2_even_prox, g2_even_dist = build_group_holes(g2_even_in, g2_even_out_x, parity="even")
    g3_odd_prox, g3_odd_dist = build_group_holes(g3_odd_in, g3_odd_out_x, parity="odd")
    g3_even_prox, g3_even_dist = build_group_holes(g3_even_in, g3_even_out_x, parity="even")

    # 基座盘按 odd 的“左右/上下”定义生成 4 个孔
    base1 = _four_holes_from_left_upper_odd(base_g1[0], base_g1[1], base_g1[2])
    base2 = _four_holes_from_left_upper_odd(base_g2[0], base_g2[1], base_g2[2])
    base3 = _four_holes_from_left_upper_odd(base_g3[0], base_g3[1], base_g3[2])

    # ===== Write lengths.csv =====
    lengths = []
    lengths.append(("l0", _mm(l0)))
    for i in range(1, 30):
        lengths.append((f"l{i}", _mm(l_regular)))
    lengths.append(("l30", _mm(l_last)))
    pd.DataFrame(lengths, columns=["name", "value_m"]).to_csv(out_dir / "lengths.csv", index=False)

    # ===== Write end_effector.csv =====
    pd.DataFrame(
        [{"p_end_x_m": float(p_end[0]), "p_end_y_m": float(p_end[1]), "p_end_z_m": float(p_end[2])}]
    ).to_csv(out_dir / "end_effector.csv", index=False)

    # ===== Write holes.csv =====
    rows = []

    def emit(disk_idx: int, hole_idx: int, side: str, p_m: np.ndarray) -> None:
        rows.append(
            {
                "disk_idx": int(disk_idx),
                "hole_idx": int(hole_idx),
                "side": side,
                "x_m": float(_mm(p_m[0])),
                "y_m": float(_mm(p_m[1])),
                "z_m": float(_mm(p_m[2])),
            }
        )

    # disk 0: only dist holes h'_{0,j} in {0}; {0} 与 {1} 重合时，这也是 disk1 的零姿态坐标口径
    for group_id, base in [(1, base1), (2, base2), (3, base3)]:
        for cable_j, quad in group_to_cables[group_id]:
            emit(0, cable_j, "dist", base[quad])

    # disks 1..30: prox+dist for all 12 holes, by parity (odd/even disk index)
    for disk_idx in range(1, 31):
        is_odd = (disk_idx % 2 == 1)
        if is_odd:
            maps = {1: (g1_odd_prox, g1_odd_dist), 2: (g2_odd_prox, g2_odd_dist), 3: (g3_odd_prox, g3_odd_dist)}
        else:
            maps = {1: (g1_even_prox, g1_even_dist), 2: (g2_even_prox, g2_even_dist), 3: (g3_even_prox, g3_even_dist)}
        for group_id, (prox_map, dist_map) in maps.items():
            for cable_j, quad in group_to_cables[group_id]:
                emit(disk_idx, cable_j, "prox", prox_map[quad])
                emit(disk_idx, cable_j, "dist", dist_map[quad])

    pd.DataFrame(rows).to_csv(out_dir / "holes.csv", index=False)

    # ===== Write disks.csv (template) =====
    disk_mass = float(args.disk_mass_kg)
    disk_com_sw_mm = _parse_xyz(args.disk_com_sw_mm)
    disk_com_dh_m = _com_sw_to_dh_mm(disk_com_sw_mm) / 1000.0

    end_mass = float(args.end_disk_mass_kg) if args.end_disk_mass_kg is not None else disk_mass
    if args.end_disk_com_sw_mm is not None:
        end_com_dh_m = _com_sw_to_dh_mm(_parse_xyz(args.end_disk_com_sw_mm)) / 1000.0
    else:
        end_com_dh_m = disk_com_dh_m.copy()

    disk_rows = []
    for disk_idx in range(1, 31):
        if disk_idx == 30:
            m = end_mass
            c = end_com_dh_m
        else:
            m = disk_mass
            c = disk_com_dh_m
        disk_rows.append(
            {
                "disk_idx": disk_idx,
                "mass_kg": float(m),
                "com_x_m": float(c[0]),
                "com_y_m": float(c[1]),
                "com_z_m": float(c[2]),
            }
        )
    pd.DataFrame(disk_rows).to_csv(out_dir / "disks.csv", index=False)

    # ===== Write material.csv (template) =====
    E = float(args.E_pa) if args.E_pa is not None else 0.0
    if args.Iz_m4 is not None:
        Iz = float(args.Iz_m4)
        Iy = None
    else:
        rod_d_m = float(args.rod_d_mm) / 1000.0
        yz_mm = _parse_yz_list(args.rod_yz_mm)
        yz_m = [(y / 1000.0, z / 1000.0) for y, z in yz_mm]
        Iz, Iy = _equiv_Iz_for_rods_m4(rod_d_m, yz_m, about=str(args.Iz_about))
    mat_rows = [{"disk_idx": i, "E_pa": E, "Iz_m4": Iz} for i in range(1, 31)]
    pd.DataFrame(mat_rows).to_csv(out_dir / "material.csv", index=False)

    print("Wrote:")
    print(f"- {out_dir/'lengths.csv'}")
    print(f"- {out_dir/'holes.csv'}")
    print(f"- {out_dir/'end_effector.csv'}")
    print(f"- {out_dir/'disks.csv'}")
    print(f"- {out_dir/'material.csv'}")
    print()
    print("NOTE:")
    print("- group2/group3 even/odd templates include derived values (swap yz and/or x shift).")
    print("- verify with: python3 scripts/validate_robot_inputs.py --config <your_config>")
    if E == 0.0 or Iz == 0.0:
        print("- material.csv has E/Iz=0 placeholders; set real values before generating dataset.")
    if Iy is not None:
        print(f"- derived (rods-only) Iz={Iz:.6e} m^4, Iy={Iy:.6e} m^4, about={args.Iz_about}")


if __name__ == "__main__":
    main()
