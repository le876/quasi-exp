"""Target-only retry20 relocation. No inverse solver or proposal beta is consumed."""
from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .retry12_symmetry import XYZ_COLUMNS, BETA_COLUMNS
from .retry19_fullspace import (
    BOX_MIN_COLUMNS, BOX_MAX_COLUMNS, mixed_resolution_coverage_metrics,
    axial_slice_coverage,
)


def hash_order(ids, seed):
    return sorted(ids, key=lambda value: (hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(), value))


def single_macroblock(lower, upper, size=40.0):
    """Check all eight interior corners of a half-open box in millimetres.

    Move the quotient below its upper integer boundary, avoiding division
    rounding nextafter(upper) back onto that boundary.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if not np.isfinite([lower, upper]).all() or np.any(upper <= lower):
        return False
    return bool(np.array_equal(np.floor(lower / size), np.floor(np.nextafter(upper / size, -np.inf))))


def cell_ids(xyz_m, zero_x_m):
    mm = np.asarray(xyz_m) * 1000
    parent = np.floor(mm / 10) * 10
    zero = np.array([zero_x_m * 1000, 0, 0])
    delta = np.maximum(np.maximum(parent - zero, zero - parent - 10), 0)
    size = np.where(np.linalg.norm(delta, axis=1) <= 40 + 1e-9, 5, 10)
    idx = np.floor(mm / size[:, None]).astype(int)
    return np.array([f"box{s}:{x}:{y}:{z}" for s, (x, y, z) in zip(size, idx)]), size


def recover_rows(labels, targets, unified, anchors, root_edges, zero_x_m):
    for name, table in (("labels", labels), ("targets", targets), ("unified", unified)):
        if table.target_id.isna().any() or table.target_id.duplicated().any():
            raise ValueError(f"{name}: nonunique target identity")
    metadata = targets[["target_id", "target_role", "cell_id", *BOX_MIN_COLUMNS, *BOX_MAX_COLUMNS, "cell_size_mm"]]
    rows = labels[["target_id", *XYZ_COLUMNS, *BETA_COLUMNS]].merge(metadata, on="target_id", validate="one_to_one", how="left")
    rows = rows.merge(unified[["target_id", "split_role", "sample_weight", "macroblock_id"]], on="target_id", how="left", validate="one_to_one")
    if rows.target_role.isna().any() or rows.split_role.isna().any():
        raise ValueError("unresolved target role or final split")
    rows["row_ordinal"] = np.arange(len(rows))
    zero = rows[rows.target_role.eq("exact_zero")]
    if len(zero) != 1 or not np.array_equal(zero[list(BETA_COLUMNS)].to_numpy(), np.zeros((1, 6))):
        raise ValueError("exact zero beta identity missing")
    if not np.array_equal(zero[list(XYZ_COLUMNS)].to_numpy()[0], [zero_x_m, 0, 0]):
        raise ValueError("exact zero XYZ identity mismatch")
    zero_id = zero.iloc[0].target_id
    roots = root_edges[root_edges.teacher_edge_legal & (root_edges.left_target_id.eq(zero_id) | root_edges.right_target_id.eq(zero_id))]
    if roots.empty:
        raise ValueError("no certified zero attachment")
    root_ids = set(roots.left_target_id) | set(roots.right_target_id)
    hard_ids = set(anchors.loc[anchors.anchor_class.eq("hard"), "target_id"])
    protected_roles = {"exact_zero", "hard_teacher_zero_attachment", "hard_root_connector", "registered_component_anchor"}
    reasons = []
    for row in rows.itertuples():
        if row.target_role in protected_roles:
            reason = row.target_role
        elif row.target_id in root_ids:
            reason = "hard_teacher_zero_attachment"
        elif row.target_id in hard_ids:
            reason = "registered_component_anchor"
        elif row.target_role != "primary_cell_service":
            reason = "retained_historical_label"
        elif pd.isna(row.cell_id):
            raise ValueError("cell-service target lacks source cell")
        elif not single_macroblock([row.x_min_mm, row.y_min_mm, row.z_min_mm], [row.x_max_mm, row.y_max_mm, row.z_max_mm]):
            reason = "split_boundary_cell"
        else:
            reason = ""
        reasons.append(reason)
    rows["relocation_exclusion_reason"] = reasons
    rows["relocation_eligible"] = rows.relocation_exclusion_reason.eq("")
    rows["source_cell_id"] = rows.cell_id
    return rows


def prepare_proposals(a, b, cells, split_registry, zero_x_m):
    required = {"proposal_id", "proposal_pool", *XYZ_COLUMNS}
    for pool in (a, b):
        if not required <= set(pool):
            raise ValueError("missing proposal XYZ provenance")
        for col in pool:
            if col.startswith("beta") or (col.startswith("proposal_beta_") and pool[col].fillna(False).any()):
                raise ValueError("proposal beta must not enter relocation")
    p = pd.concat([a[list(required)], b[list(required)]], ignore_index=True)
    if p.proposal_id.duplicated().any():
        raise ValueError("duplicate proposal ID")
    xyz = p[list(XYZ_COLUMNS)].to_numpy(float)
    p["source_cell_id"], size = cell_ids(xyz, zero_x_m)
    phase = xyz * 1000 / size[:, None] - np.floor(xyz * 1000 / size[:, None])
    for d, axis in enumerate("xyz"):
        p[f"phase_{axis}"] = phase[:, d]
    block = np.floor(xyz * 1000 / 40).astype(int)
    p["macroblock_id"] = [f"smacro:{x}:{y}:{z}" for x, y, z in block]
    lookup = split_registry.set_index("macroblock_id").split_role
    if not lookup.index.is_unique:
        raise ValueError("macroblock split is ambiguous")
    p["split_role"] = p.macroblock_id.map(lookup)
    p["u_mm"] = (zero_x_m - p.x_m) * 1000
    observed = cells.set_index("cell_id")
    p["same_cell_cross_pool_supported"] = p.source_cell_id.map(observed.pool_a_observed & observed.pool_b_observed).fillna(False)
    return p


def reserve_panels(proposals, unified, seed):
    p = proposals[proposals.u_mm.between(0, 200)].copy()
    existing = set(map(tuple, unified[list(XYZ_COLUMNS)].to_numpy(float)))
    p = p[[tuple(x) not in existing for x in p[list(XYZ_COLUMNS)].to_numpy(float)]]
    p = p.drop_duplicates(list(XYZ_COLUMNS)).copy()
    # Evaluation coordinates are actual off-centre proposals, not teacher successes.
    p = p[~np.isclose(p.phase_x, .5, atol=1e-7, rtol=0)].copy()
    p["phase_bin"] = np.minimum((p.phase_x * 5).astype(int), 4)
    p["rank"] = [hashlib.sha256(f"{seed}:panel:{x}".encode()).hexdigest() for x in p.proposal_id]
    panels = []
    for split, count, name in (("test", 600, "geometry_test"), ("validation", 200, "geometry_validation")):
        for phase in range(5):
            pool = p[p.split_role.eq(split) & p.phase_bin.eq(phase)].sort_values("rank")
            if len(pool) < count:
                raise ValueError(f"insufficient {split} phase {phase}: {len(pool)} < {count}")
            selected = pool.head(count).copy()
            selected["panel"] = name
            # Round-robin spatial strata for the canonical subset are frozen here.
            selected["canonical_reference"] = False
            if split == "test":
                spatial = selected.assign(u_bin=np.floor(selected.u_mm / 20).astype(int),
                                          y_bin=np.floor(selected.y_m * 1000 / 80).astype(int),
                                          z_bin=np.floor(selected.z_m * 1000 / 80).astype(int))
                spatial["stratum_rank"] = spatial.groupby(["u_bin", "y_bin", "z_bin"]).cumcount()
                chosen = spatial.sort_values(["stratum_rank", "rank"]).head(200).index
                selected.loc[chosen, "canonical_reference"] = True
            panels.append(selected)
    return pd.concat(panels, ignore_index=True).drop(columns="rank")


class SpacingIndex:
    def __init__(self, xyz, radius):
        self.radius = radius
        self.xyz = np.asarray(xyz, float).copy()
        self.buckets = defaultdict(set)
        for i, point in enumerate(self.xyz):
            self.buckets[self.key(point)].add(i)

    def key(self, p):
        return tuple(np.floor(np.asarray(p) / self.radius).astype(int))

    def allowed(self, p, own):
        key = self.key(p)
        for delta in itertools.product((-1, 0, 1), repeat=3):
            for j in self.buckets.get(tuple(np.array(key) + delta), ()):
                if j != own and np.linalg.norm(self.xyz[j] - p) < self.radius:
                    return False
        return True

    def replace(self, i, p):
        self.buckets[self.key(self.xyz[i])].remove(i)
        self.xyz[i] = p
        self.buckets[self.key(p)].add(i)


def select_replacements(rows, proposals, panels, unified, *, seed, epsilon=.05, spacing_mm=1.0):
    rows = rows.copy()
    p = proposals.copy()
    keep = p[[f"phase_{a}" for a in "xyz"]].ge(epsilon).all(axis=1) & p[[f"phase_{a}" for a in "xyz"]].le(1-epsilon).all(axis=1)
    reserved = set(map(tuple, panels[list(XYZ_COLUMNS)].to_numpy(float)))
    keep &= p.u_mm.between(0, 200)
    p = p[keep].copy()
    p = p[[tuple(x) not in reserved for x in p[list(XYZ_COLUMNS)].to_numpy(float)]]
    missing = ~p.same_cell_cross_pool_supported
    # Native service cells normally have both pools; the radius rule covers other cells.
    for pool, other in (("a", "b"), ("b", "a")):
        local = missing & p.proposal_pool.eq(pool)
        if local.any():
            distance = cKDTree(proposals.loc[proposals.proposal_pool.eq(other), list(XYZ_COLUMNS)].to_numpy()).query(p.loc[local, list(XYZ_COLUMNS)].to_numpy())[0] * 1000
            p.loc[local, "same_cell_cross_pool_supported"] = distance <= 5 + 1e-6
    p = p[p.same_cell_cross_pool_supported].reset_index(drop=True)
    groups = p.groupby("source_cell_id").indices
    eligible = rows[rows.relocation_eligible]
    if eligible.source_cell_id.duplicated().any():
        raise ValueError("more than one relocation row per source cell")
    ordered = hash_order(eligible.source_cell_id.tolist(), seed)
    power = int(np.ceil(np.log2(max(1, len(ordered)))))
    offsets = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power)[:len(ordered)]
    row_lookup = eligible.set_index("source_cell_id").row_ordinal.to_dict()
    unified_index = dict(zip(unified.target_id, range(len(unified))))
    spatial = SpacingIndex(unified[list(XYZ_COLUMNS)].to_numpy(float)*1000, spacing_mm)
    xyz = p[list(XYZ_COLUMNS)].to_numpy(float)
    selected = []
    for ordinal, (cell_id, offset) in enumerate(zip(ordered, offsets)):
        i = row_lookup[cell_id]
        old = rows.iloc[i]
        candidates = groups.get(cell_id, [])
        lower = old[list(BOX_MIN_COLUMNS)].to_numpy(float)
        ideal = lower + float(old.cell_size_mm) * (.1 + .8 * offset)
        candidates = sorted(candidates, key=lambda j: (float(np.linalg.norm(xyz[j]*1000-ideal)), str(p.iloc[j].proposal_id)))
        own = unified_index[old.target_id]
        chosen = None
        for j in candidates:
            candidate = p.iloc[j]
            if candidate.macroblock_id != old.macroblock_id or candidate.split_role != old.split_role:
                continue
            if not spatial.allowed(xyz[j]*1000, own):
                continue
            chosen = j
            break
        if chosen is None:
            rows.loc[i, "relocation_exclusion_reason"] = "no_admissible_proposal"
            continue
        spatial.replace(own, xyz[chosen]*1000)
        candidate = p.iloc[chosen]
        service = np.linalg.norm(xyz[chosen]*1000 - (lower + float(old.cell_size_mm)/2))
        if service > 10+1e-6:
            raise AssertionError("own source-cell probe lost")
        selected.append({"target_id": old.target_id, "row_ordinal": i, "selection_ordinal": ordinal,
                         "source_cell_id": cell_id, "cell_size_mm": old.cell_size_mm,
                         "proposal_id": candidate.proposal_id, "proposal_pool": candidate.proposal_pool,
                         "macroblock_id": candidate.macroblock_id, "split_role": candidate.split_role,
                         "source_cell_probe_served": True, "source_cell_distance_mm": service,
                         **{c: float(candidate[c]) for c in XYZ_COLUMNS},
                         **{f"phase_{a}": float(candidate[f"phase_{a}"]) for a in "xyz"}})
    return rows, pd.DataFrame(selected)


def phase_audit(xyz_m, sizes=None):
    xyz = np.asarray(xyz_m, float)*1000
    size = np.full(len(xyz), 10.0) if sizes is None else np.asarray(sizes, float)
    phase = np.mod(xyz / size[:, None], 1)
    def entropy(count):
        p = count / count.sum()
        nonzero = p[p > 0]
        return float(-(nonzero*np.log(nonzero)).sum()/np.log(len(p)))
    result = {}
    for d, axis in enumerate("xyz"):
        count = np.histogram(phase[:, d], bins=np.linspace(0, 1, 11))[0]
        result[axis] = {"entropy": entropy(count), "max_bin_fraction": float(count.max()/len(phase)), "counts": count.tolist()}
    joint = np.histogramdd(phase, bins=[np.linspace(0, 1, 5)]*3)[0].ravel()
    result["joint_entropy_4cubed"] = entropy(joint)
    result["phase_correlation"] = np.corrcoef(phase.T).tolist()
    result["center_plane_fraction"] = float(np.mean(np.abs(np.mod(xyz[:, 0], 10)-5) <= 1e-6))
    return result


def coverage_audit(cells, xyz_m, zero_x_m):
    cells = cells[cells.domain_class.eq("primary")].copy()
    probes = cells[["probe_x_m", "probe_y_m", "probe_z_m"]].to_numpy(float)
    distance = cKDTree(np.asarray(xyz_m, float)).query(probes)[0]*1000
    cells["service_distance_mm"] = distance
    cells["served"] = distance <= 10+1e-6
    metrics = dict(mixed_resolution_coverage_metrics(cells))
    slices = axial_slice_coverage(cells, zero_x_mm=zero_x_m*1000, bin_edges_u_mm=np.arange(0, 201, 10))
    metrics["minimum_axial_slice_coverage"] = float(slices.volume_coverage.min())
    return metrics, cells, slices


def geometry_gate(baseline, proposed, phase, candidate_count, requested=24000):
    checks = {
        "coarse_replacement_count": candidate_count >= requested,
        "phase_entropy": phase["x"]["entropy"] >= .90,
        "maximum_phase_bin": phase["x"]["max_bin_fraction"] <= .20,
        "center_plane_fraction": phase["center_plane_fraction"] <= .20,
        "volume_coverage": proposed["volume_coverage"] >= max(.97, baseline["volume_coverage"]-.01),
        "service_p95": proposed["service_distance_volume_weighted_p95_mm"] <= 10+1e-6,
        "axial_nonregression": proposed["minimum_axial_slice_coverage"] >= baseline["minimum_axial_slice_coverage"]-1e-12,
        "axial_minimum": proposed["minimum_axial_slice_coverage"] >= .90,
    }
    return {"checks": checks, "passed": all(checks.values()), "failed_checks": [k for k,v in checks.items() if not v]}
