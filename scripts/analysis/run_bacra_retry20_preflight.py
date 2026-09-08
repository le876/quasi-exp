#!/usr/bin/env python3
"""Sealed target-only preflight; never launches an inverse solver or training."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from quasi_exp.teacher.retry20_relocation import (
    recover_rows, prepare_proposals, reserve_panels, select_replacements,
    phase_audit, coverage_audit, geometry_gate,
)
from quasi_exp.teacher.retry12_symmetry import XYZ_COLUMNS

EXPERIMENT = "bacra_retry20_continuous_target_preflight"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    def normalize(v):
        if isinstance(v, dict):
            return {str(k): normalize(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [normalize(x) for x in v]
        if isinstance(v, np.generic):
            return normalize(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize(value), indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def identity(config_path, binding):
    if git("rev-parse", "HEAD") != binding or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("requires clean exact binding checkout")
    entry = yaml.safe_load(git("show", f"{binding}:spec/registry.yaml"))["experiments"][EXPERIMENT]
    source = entry["scientific_source_fixed_point"]
    subprocess.run(["git", "merge-base", "--is-ancestor", source, binding], cwd=ROOT, check=True)
    closure = ["src", "scripts/analysis", "scripts/pipelines", "configs", "tests", *entry["protocol_sources"]]
    subprocess.run(["git", "diff", "--exit-code", source, binding, "--", *closure], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    if str(config_path.relative_to(ROOT)) != entry["config"]:
        raise ValueError("config not bound by registry")
    return {"experiment_id": EXPERIMENT, "scientific_source_fixed_point": source,
            "binding_fixed_point": binding, "config_sha256": digest(config_path),
            "execution_scope": "target_only_preflight", "proposal_beta_used_as_label_or_hint": False}


def verify_upstream(config):
    root = Path(config["upstream_root"])
    inventory = []
    for rel, expected in config["upstream_manifests"].items():
        manifest = root / rel
        if digest(manifest) != expected:
            raise ValueError(f"changed upstream manifest: {rel}")
        content = json.loads(manifest.read_text())
        for item in content["artifacts"]:
            path = manifest.parent / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError(f"changed upstream artifact: {path}")
            inventory.append({"path": str(path), "sha256": item["sha256"]})
    return root, inventory


def budget_gate(base, candidate_count, requested, geometry_passed):
    """Shared feasibility authorization is only one necessary condition.

    Geometry and this target-only binding independently forbid inverse solving.
    Do not encode a scientific layout failure as a resource shortfall.
    """
    feasible = candidate_count >= requested
    return {**base, "inputs_valid": True, "required_objectives_budget_feasible": feasible,
            "status": "feasible" if feasible else "diagnostic_only",
            "claim_bearing_run_authorized": feasible, "diagnostic_pilot_authorized": False,
            "failed_objective_ids": [] if feasible else ["coarse_relocation"],
            "reason_codes": [] if feasible else ["INSUFFICIENT_ADMISSIBLE_COARSE_TARGETS"],
            "geometry_gate_passed": bool(geometry_passed), "full_experiment_authorized": False,
            "authorization_scope": "budget_lower_bound_only; solver also requires geometry pass and executable solver binding"}


def run(config, config_path, output, binding):
    started = time.monotonic()
    ident = identity(config_path, binding)
    upstream, inventory = verify_upstream(config)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_identity.json", ident)
    stage = output / "00_objective_feasibility"
    stage.mkdir()
    base = {"schema_version": 1, "experiment_id": EXPERIMENT}
    def progress(phase):
        write_json(output / "progress.json", {"status": "running", "phase": phase, "elapsed_seconds": time.monotonic()-started})
        print(phase, flush=True)
    def table(name, frame):
        frame.to_parquet(stage / f"{name}.parquet", index=False)
        return {"path": f"00_objective_feasibility/{name}.parquet", "sha256": digest(stage/f"{name}.parquet"), "row_count": len(frame)}
    def read(path):
        return pd.read_parquet(upstream/path)
    progress("recover_protected_rows")
    labels = read("05_fullspace_graph_teacher/primary_direct_teacher_labels.parquet")
    targets = read("04_adaptive_fullspace_fill/final_primary_target_registry.parquet")
    unified = read("06_unified_dataset/unified_signed_supervision.parquet")
    cells = read("00_fullspace_discovery/mixed_resolution_cell_registry.parquet")
    anchors = read("05_fullspace_graph_teacher/old_anchor_registry.parquet")
    roots = read("05_fullspace_graph_teacher/teacher_root_attachment.parquet")
    rows = recover_rows(labels, targets, unified, anchors, roots, config["zero_x_m"])
    if len(rows) != 29970 or len(unified) != 74696:
        raise ValueError("upstream row denominator changed")
    primary_xyz = rows[list(XYZ_COLUMNS)].to_numpy(float)
    baseline, baseline_cells, baseline_slices = coverage_audit(cells, primary_xyz, config["zero_x_m"])
    table("baseline_served_cells", baseline_cells)
    table("baseline_axial_slices", baseline_slices)
    progress("reserve_independent_panels")
    proposals = prepare_proposals(read("00_fullspace_discovery/proposal_pool_a_xyz_only.parquet"),
                                 read("00_fullspace_discovery/proposal_pool_b_xyz_only.parquet"), cells,
                                 read("06_unified_dataset/unified_split_registry.parquet"), config["zero_x_m"])
    panels = reserve_panels(proposals, unified, config["seed"])
    panel_ref = table("evaluation_panel_registry", panels)
    progress("select_cell_internal_targets")
    rows, replacement = select_replacements(rows, proposals, panels, unified, seed=config["seed"],
                                            epsilon=config["actual_phase_epsilon"], spacing_mm=config["minimum_spacing_mm"])
    row_ref = table("relocation_eligibility", rows)
    replacement_ref = table("candidate_replacement_registry", replacement)
    coarse = replacement[replacement.cell_size_mm.eq(10)]
    requested = config["successful_coarse_replacements"]
    # A concrete 24k layout, with all eligible fine relocations, before any IK.
    planned = pd.concat([replacement[replacement.cell_size_mm.eq(5)], coarse.head(requested)]).sort_values("selection_ordinal")
    plan_ref = table("planned_replacement_registry", planned)
    target_xyz = primary_xyz.copy()
    target_xyz[planned.row_ordinal.to_numpy(int)] = planned[list(XYZ_COLUMNS)].to_numpy(float)
    final_unified_xyz = unified[list(XYZ_COLUMNS)].to_numpy(float).copy()
    final_index = unified.set_index("target_id").index.get_indexer(planned.target_id)
    final_unified_xyz[final_index] = planned[list(XYZ_COLUMNS)].to_numpy(float)
    distances = cKDTree(final_unified_xyz).query(final_unified_xyz[final_index], k=2)[0][:, 1]*1000
    prospective = rows[["target_id", "target_role", "source_cell_id", "cell_size_mm", "split_role"]].copy()
    prospective[list(XYZ_COLUMNS)] = target_xyz
    prospective["evidence_role"] = "geometry_only_not_supervision"
    table("prospective_primary_targets", prospective)
    progress("global_coverage_and_phase_audit")
    proposed, proposed_cells, proposed_slices = coverage_audit(cells, target_xyz, config["zero_x_m"])
    table("prospective_served_cells", proposed_cells)
    table("prospective_axial_slices", proposed_slices)
    phase = phase_audit(target_xyz)
    phase_results = {"baseline_primary_10mm": phase_audit(primary_xyz), "prospective_primary_10mm": phase}
    for size in (5., 10.):
        mask = rows.cell_size_mm.eq(size).to_numpy()
        phase_results[f"native_{int(size)}mm_baseline"] = phase_audit(primary_xyz[mask], np.full(mask.sum(), size))
        phase_results[f"native_{int(size)}mm_prospective"] = phase_audit(target_xyz[mask], np.full(mask.sum(), size))
    gate = geometry_gate(baseline, proposed, phase, len(coarse), requested)
    gate["checks"]["final_replacement_spacing"] = bool((distances >= config["minimum_spacing_mm"]-1e-9).all())
    gate["passed"] = all(gate["checks"].values())
    gate["failed_checks"] = [key for key, value in gate["checks"].items() if not value]
    audit = {"baseline": baseline, "prospective": proposed, "phase": phase_results,
             "eligible_relocation_cell_count": int(rows.relocation_eligible.sum()),
             "admissible_coarse_replacement_count": len(coarse), "planned_coarse_replacements": min(len(coarse), requested),
             "exclusion_counts": rows.relocation_exclusion_reason.value_counts().to_dict(),
             "cross_pool_supported_fraction": 1.0 if len(replacement) else 0.,
             "replacement_minimum_spacing_mm": float(distances.min()) if len(distances) else None,
             "replacement_pairs_below_1mm": int((distances < config["minimum_spacing_mm"]-1e-9).sum()),
             "layout_is_global_optimum": False, "teacher_labels_generated": 0, "solver_attempts": 0,
             "geometry_gate": gate, "upstream_verified_artifact_count": len(inventory)}
    write_json(stage/"geometry_audit.json", audit)
    required_cells = cells[cells.required & cells.domain_class.eq("primary")].copy()
    cell_ref = table("required_cell_registry", required_cells)
    coarse_rows = rows[rows.cell_size_mm.eq(10)]
    coarse_ref = table("coarse_center_denominator", coarse_rows[["target_id", "source_cell_id"]])
    # Exact volume-unit denominator preserves the 5/10 mm volume weighting.
    volumes = np.rint(required_cells.volume_mm3.to_numpy()/125).astype(int)
    volume_rows = required_cells.loc[required_cells.index.repeat(volumes), ["cell_id"]].reset_index(drop=True)
    volume_rows["volume_unit_id"] = volume_rows.cell_id + ":" + volume_rows.groupby("cell_id").cumcount().astype(str)
    volume_ref = table("required_volume_units", volume_rows)
    served_ids = set(baseline_cells.loc[baseline_cells.served, "cell_id"])
    reusable_volume = int(volume_rows.cell_id.isin(served_ids).sum())
    contract = {**base, "scope": "coverage_trajectory", "scientific_source_sha": ident["scientific_source_fixed_point"],
                "config_sha256": ident["config_sha256"], "diagnostic_pilot_allowed": False,
                "primary_objectives": [
                    {"id": "coarse_relocation", "kind": "dataset_size", "metric": "successful_coarse_replacements", "required_for_claim": True,
                     "denominator_id": "coarse_centers", "target": {"operator": ">=", "value": requested, "unit": "count"}},
                    {"id": "coverage", "kind": "coverage", "metric": "volume_coverage", "required_for_claim": True,
                     "denominator_id": "volume_units", "target": {"operator": ">=", "value": max(.97, baseline["volume_coverage"]-.01), "unit": "fraction"}},
                ], "execution_scope": "target_only_preflight", "downstream_requires_separate_executable_binding": True}
    write_json(stage/"objective_contract.json", contract)
    write_json(stage/"denominator_size.json", {**base, "frozen_before_launch": True,
        "denominators": [{"id": "coarse_centers", "kind": "dataset_size", "unit": "row", "required_count": len(coarse_rows), "registry": {**coarse_ref, "id_column": "target_id"}},
                         {"id": "volume_units", "kind": "coverage", "unit": "125_mm3", "required_count": len(volume_rows), "registry": {**volume_ref, "id_column": "volume_unit_id"}}],
        "required_cells": cell_ref,
        "resource_denominators": {"required_supervision_vertex_count": len(rows), "required_logical_edge_count": len(roots), "required_second_parent_certification_count": requested}})
    write_json(stage/"reusable_evidence.json", {**base, "source_artifacts": inventory,
        "eligible_counts": {"supervision_vertices": len(rows), "connector_only_vertices": 0, "served_coverage_units": reusable_volume, "complete_trajectories": 0, "verified_edges": int(roots.teacher_edge_legal.sum()), "second_parent_certifications": 0},
        "ineligible_counts": {"proposal_only": len(proposals), "branch_conflicts": 0, "unused": 0},
        "credit_registry": row_ref, "proposal_beta_used_as_label_or_hint": False})
    target_volume = int(np.ceil(len(volume_rows)*contract["primary_objectives"][1]["target"]["value"]))
    write_json(stage/"budget_lower_bound.json", {**base, "objective_lower_bounds": [
        {"objective_id": "coarse_relocation", "method": "count_credit", "required_units": len(coarse_rows), "target_units": requested, "reusable_eligible_units": 0,
         "maximum_credit_per_new_supervision_vertex": 1, "minimum_resources": {"new_supervision_vertices": requested}, "basis_registry": plan_ref},
        {"objective_id": "coverage", "method": "count_credit", "required_units": len(volume_rows), "target_units": target_volume, "reusable_eligible_units": reusable_volume,
         "maximum_credit_per_new_supervision_vertex": 1, "minimum_resources": {"new_supervision_vertices": 0}, "basis_registry": volume_ref}],
        "resources": {"new_supervision_vertices": {"optimistic_minimum": requested, "registered_budget": len(coarse), "basis_registry": replacement_ref}},
        "all_required_objectives_bounded": True, "all_required_resources_feasible": len(coarse) >= requested,
        "registered_wall_time_seconds": config["deadline_hours"]*3600,
        "runtime_projection_available": False, "pilot_success_assumption": "optimistic 100 percent only; no numerical success measured"})
    write_json(stage/"atomic_objective_schedule.json", {**base, "row_count_is_stop_condition": False,
        "scheduled_objective_ids": ["coarse_relocation", "coverage"], "entries": [
            {"id": "paired_replacement", "objective_id": "coarse_relocation", "kind": "dataset", "priority": 0, "required_for_claim": True, "requirement_registry": plan_ref,
             "reserved_resources": {"new_supervision_vertices": requested, "edge_certificates": 0}},
            {"id": "retain_global_coverage", "objective_id": "coverage", "kind": "coverage", "priority": 0, "required_for_claim": True, "requirement_registry": volume_ref,
             "reserved_resources": {"new_supervision_vertices": 0, "edge_certificates": 0}}],
        "evaluation_panel_registry": panel_ref, "evaluation_reserve_seconds": 7200})
    write_json(stage/"gate.json", budget_gate(base, len(coarse), requested, gate["passed"]))
    elapsed = time.monotonic()-started
    summary = {**ident, "operational_status": "complete", "scientific_status": "geometry_preflight_pass" if gate["passed"] else "geometry_preflight_failed",
               "teacher_status": "not_evaluated", "student_status": "not_evaluated", "formal_authorized": False,
               "elapsed_seconds": elapsed, "geometry_gate": gate, "generated_supervision_rows": 0,
               "next_stage": "implement_bound_solver_only_after_pass" if gate["passed"] else "stop_before_ik"}
    write_json(output/"summary.json", summary)
    write_json(output/"progress.json", {"status": "complete", "phase": summary["scientific_status"], "elapsed_seconds": elapsed})
    manifest = {"experiment_id": EXPERIMENT, "scientific_source_fixed_point": ident["scientific_source_fixed_point"],
                "artifacts": [{"path": str(p.relative_to(output)), "sha256": digest(p)} for p in sorted(output.rglob("*")) if p.is_file()]}
    write_json(output/"completion_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--binding-sha", required=True)
    args = parser.parse_args()
    run(yaml.safe_load(args.config.read_text()), args.config.resolve(), args.output_root.resolve(), args.binding_sha)
