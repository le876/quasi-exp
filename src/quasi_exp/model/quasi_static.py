from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cable_geometry import cable_lengths_all, holes_in_base
from .kinematics import forward_kinematics, invert_T
from .tension_transmission import transmit_tensions


def compute_tau0(masses_kg: np.ndarray, lengths_m: np.ndarray, g: float = 9.81) -> float:
    masses_kg = np.asarray(masses_kg, dtype=float).reshape(-1)
    lengths_m = np.asarray(lengths_m, dtype=float).reshape(-1)
    total_mass = float(np.sum(masses_kg))
    total_length = float(np.sum(lengths_m))
    tau0 = total_mass * g * total_length
    return float(max(tau0, 1e-9))


@dataclass(frozen=True)
class QuasiStaticThetaCache:
    theta_rad: np.ndarray  # (kD,)
    T_0_from_i: np.ndarray  # (kD+1,4,4)
    T_i_from_0: np.ndarray  # (kD+1,4,4)
    holes_local_m: np.ndarray  # (kD+1,12,2,4)
    holes_0_m: np.ndarray  # (kD+1,12,2,4)
    L_12: np.ndarray  # (12,)
    case_flag_12: np.ndarray  # (12,)
    active_ij: np.ndarray  # (kD+1,12) bool, active at joint i

    # Cable geometry in frame {i} for residual computations
    h_i_prox_m: np.ndarray  # (kD+1,12,3) in {i}; i=0 unused
    dir_i_j: np.ndarray  # (kD+1,12,3) in {i}; i=0 unused
    mcoef_z: np.ndarray  # (kD+1,12)  Mz = mcoef_z * F_CT_{i,j}

    # Gravity precomputes
    grav_mz_sum: np.ndarray  # (kD+1,) sum_{p>=i} M_G_p,z in {i}
    grav_force_sum: np.ndarray  # (kD+1,3) sum_{p>=i} G_p in {i}

    # Bending term (z)
    bnd_mz: np.ndarray  # (kD+1,)


class QuasiStaticModel:
    def __init__(self, cfg: dict[str, Any], inputs) -> None:
        self.cfg = cfg
        self.inputs = inputs
        self.kD = int(cfg["robot"]["kD"])
        self.theta_sign = float(cfg.get("kinematics", {}).get("theta_sign", 1.0))
        self.g = float(cfg["physics"]["g"])
        self.mu_shaft = float(cfg["friction"]["mu_shaft"])
        self.r_shaft_m = float(cfg["friction"]["r_shaft_m"])
        self.mu_cable = float(cfg["friction"]["mu_cable"])
        self.t_min = float(cfg["tension"]["bounds_n"][0])
        self.t_max = float(cfg["tension"]["bounds_n"][1])

        self.end_disk_by_j = {int(k): int(v) for k, v in cfg["cables"]["end_disk_by_j"].items()}
        self.tau0 = compute_tau0(inputs.masses_kg, inputs.lengths_m, g=self.g)

        theta0 = np.zeros(self.kD, dtype=float)
        _, T_0_from_i = forward_kinematics(theta0, inputs.lengths_m, inputs.p_end_local_m, theta_sign=self.theta_sign)
        holes0 = holes_in_base(T_0_from_i, inputs.holes_local_m)
        self.L0_12 = cable_lengths_all(holes0, self.end_disk_by_j)

    def build_cache(self, theta_rad: np.ndarray) -> QuasiStaticThetaCache:
        theta_raw = np.asarray(theta_rad, dtype=float).reshape(self.kD)
        theta = theta_raw * self.theta_sign
        _, T_0_from_i = forward_kinematics(theta_raw, self.inputs.lengths_m, self.inputs.p_end_local_m, theta_sign=self.theta_sign)
        T_i_from_0 = np.zeros_like(T_0_from_i)
        for i in range(self.kD + 1):
            T_i_from_0[i] = invert_T(T_0_from_i[i])

        holes_0 = holes_in_base(T_0_from_i, self.inputs.holes_local_m)
        L_12 = cable_lengths_all(holes_0, self.end_disk_by_j)

        # case_flag 只依赖 ΔL
        dummy_T_base = np.ones(12, dtype=float)
        _, case_flag = transmit_tensions(
            theta_rad=theta,
            T_base_12=dummy_T_base,
            mu_cable=self.mu_cable,
            end_disk_by_j=self.end_disk_by_j,
            L0_12=self.L0_12,
            L_12=L_12,
        )

        active = np.zeros((self.kD + 1, 12), dtype=bool)
        for j in range(1, 13):
            end_disk = int(self.end_disk_by_j[j])
            active[1 : end_disk + 1, j - 1] = True

        h_i_prox = np.zeros((self.kD + 1, 12, 3), dtype=float)
        dir_i_j = np.zeros((self.kD + 1, 12, 3), dtype=float)
        mcoef_z = np.zeros((self.kD + 1, 12), dtype=float)

        for i in range(1, self.kD + 1):
            h_i_prox[i] = self.inputs.holes_local_m[i, :, 0, :3]
            for j in range(1, 13):
                if not active[i, j - 1]:
                    continue
                # h'_{i-1,j} in {i}
                p0 = T_0_from_i[i - 1] @ self.inputs.holes_local_m[i - 1, j - 1, 1]
                pi = T_i_from_0[i] @ p0
                v = pi[:3] - h_i_prox[i, j - 1]
                n = float(np.linalg.norm(v))
                if n < 1e-12 or not np.isfinite(n):
                    dir_vec = np.array([0.0, 0.0, 0.0], dtype=float)
                else:
                    dir_vec = v / n
                dir_i_j[i, j - 1] = dir_vec
                # Mz coefficient: (r x dir)_z, r = h_i_prox (Oi=0)
                cross = np.cross(h_i_prox[i, j - 1], dir_vec)
                mcoef_z[i, j - 1] = float(cross[2])

        # Gravity precompute for each i: sum over p>=i
        grav_mz_sum = np.zeros(self.kD + 1, dtype=float)
        grav_force_sum = np.zeros((self.kD + 1, 3), dtype=float)
        g0_dir = np.array([0.0, -1.0, 0.0], dtype=float)

        com_local = self.inputs.com_local_m
        masses = self.inputs.masses_kg

        for i in range(1, self.kD + 1):
            R_i_from_0 = T_i_from_0[i, :3, :3]
            sum_mz = 0.0
            sum_F = np.zeros(3, dtype=float)
            for p in range(i, self.kD + 1):
                F0 = (masses[p] * self.g) * g0_dir
                Fi = R_i_from_0 @ F0
                p0 = T_0_from_i[p] @ com_local[p]
                pi = T_i_from_0[i] @ p0
                r = pi[:3]
                M = np.cross(r, Fi)
                sum_mz += float(M[2])
                sum_F += Fi
            grav_mz_sum[i] = sum_mz
            grav_force_sum[i] = sum_F

        # Bending term
        bnd_mz = np.zeros(self.kD + 1, dtype=float)
        lengths = self.inputs.lengths_m
        E = self.inputs.E_pa
        Iz = self.inputs.Iz_m4
        for i in range(1, self.kD + 1):
            li = float(lengths[i])
            if li <= 0:
                bnd_mz[i] = np.nan
            else:
                bnd_mz[i] = float((theta[i - 1] / li) * E[i] * Iz[i])

        return QuasiStaticThetaCache(
            theta_rad=theta,
            T_0_from_i=T_0_from_i,
            T_i_from_0=T_i_from_0,
            holes_local_m=self.inputs.holes_local_m,
            holes_0_m=holes_0,
            L_12=L_12,
            case_flag_12=case_flag,
            active_ij=active,
            h_i_prox_m=h_i_prox,
            dir_i_j=dir_i_j,
            mcoef_z=mcoef_z,
            grav_mz_sum=grav_mz_sum,
            grav_force_sum=grav_force_sum,
            bnd_mz=bnd_mz,
        )

    def residual_z(
        self,
        cache: QuasiStaticThetaCache,
        T_base_12: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        T_base = np.asarray(T_base_12, dtype=float).reshape(12)
        if (T_base < self.t_min).any() or (T_base > self.t_max).any():
            # 仍然返回，让上层惩罚/裁剪
            pass

        F_ct, _ = transmit_tensions(
            theta_rad=cache.theta_rad,
            T_base_12=T_base,
            mu_cable=self.mu_cable,
            end_disk_by_j=self.end_disk_by_j,
            L0_12=self.L0_12,
            L_12=cache.L_12,
        )

        r_z = np.zeros(self.kD + 1, dtype=float)
        fric_z = np.zeros(self.kD + 1, dtype=float)
        support_norm = np.zeros(self.kD + 1, dtype=float)

        for i in range(1, self.kD + 1):
            # Cable moment z
            F_i = F_ct[i]  # tensions at segment i
            cable_mz = float(np.nansum(cache.mcoef_z[i] * F_i))
            Mz_raw = cache.grav_mz_sum[i] + cable_mz + cache.bnd_mz[i]

            # Support force in {i}
            cable_force_sum = np.zeros(3, dtype=float)
            for j in range(12):
                if not cache.active_ij[i, j]:
                    continue
                if not np.isfinite(cache.dir_i_j[i, j]).all():
                    continue
                cable_force_sum += cache.dir_i_j[i, j] * F_i[j]
            Fs = -(cache.grav_force_sum[i] + cable_force_sum)
            Fs_norm = float(np.linalg.norm(Fs))
            support_norm[i] = Fs_norm

            if Mz_raw == 0.0 or not np.isfinite(Mz_raw) or not np.isfinite(Fs_norm):
                Mz_fric = 0.0
            else:
                Mz_fric = -float(np.sign(Mz_raw)) * self.mu_shaft * Fs_norm * self.r_shaft_m

            fric_z[i] = Mz_fric
            r_z[i] = Mz_raw + Mz_fric

        debug = {
            "support_norm": support_norm[1:].copy(),
            "fric_z": fric_z[1:].copy(),
            "F_ct": F_ct.copy(),
        }
        return r_z[1:].copy(), debug

    def residual_norm(
        self,
        cache: QuasiStaticThetaCache,
        T_base_12: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        r_z, debug = self.residual_z(cache, T_base_12)
        return r_z / self.tau0, debug
