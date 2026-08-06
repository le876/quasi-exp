"""Budget stability and patch-level admission for selected inverse sections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .canonical import beta_rms_deg
from .section_first_atlas import SectionGrowthResult


@dataclass(frozen=True)
class SectionStabilityPolicy:
    coverage_jaccard_min: float = 0.95
    exploratory_beta_p95_max_deg: float = 1.0
    exploratory_beta_max_deg: float = 2.0
    chart_assignment_change_max: float = 0.05
    boundary_change_max: float = 0.05
    selected_edge_change_max: float = 0.05

    def __post_init__(self) -> None:
        fractions = (
            self.coverage_jaccard_min,
            self.chart_assignment_change_max,
            self.boundary_change_max,
            self.selected_edge_change_max,
        )
        if any(not 0 <= float(value) <= 1 for value in fractions):
            raise ValueError("section stability ratios must be in [0, 1]")
        if self.exploratory_beta_p95_max_deg <= 0 or self.exploratory_beta_max_deg <= 0:
            raise ValueError("section stability beta limits must be positive")


@dataclass(frozen=True)
class SectionBudgetStabilityReport:
    common_node_count: int
    coverage_jaccard: float
    selected_beta_p95_deg: float
    selected_beta_max_deg: float
    chart_assignment_change_ratio: float
    boundary_change_ratio: float
    selected_edge_change_ratio: float
    raw_cap_hit: bool
    selected_section_affected_by_cap: bool
    gate_pass: bool
    checks: Mapping[str, bool]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))
        object.__setattr__(self, "reasons", tuple(self.reasons))

    def as_dict(self) -> dict[str, Any]:
        return {
            "common_node_count": self.common_node_count,
            "coverage_jaccard": self.coverage_jaccard,
            "selected_beta_p95_deg": self.selected_beta_p95_deg,
            "selected_beta_max_deg": self.selected_beta_max_deg,
            "chart_assignment_change_ratio": self.chart_assignment_change_ratio,
            "boundary_change_ratio": self.boundary_change_ratio,
            "selected_edge_change_ratio": self.selected_edge_change_ratio,
            "raw_cap_hit": self.raw_cap_hit,
            "selected_section_affected_by_cap": self.selected_section_affected_by_cap,
            "gate_pass": self.gate_pass,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
        }


def compare_section_growth(
    lower_budget: SectionGrowthResult,
    higher_budget: SectionGrowthResult,
    *,
    policy: SectionStabilityPolicy | None = None,
) -> SectionBudgetStabilityReport:
    """Compare two complete selected sections, normally K=4 versus K=8."""

    active = SectionStabilityPolicy() if policy is None else policy
    low_coverage = set(lower_budget.covered_node_ids)
    high_coverage = set(higher_budget.covered_node_ids)
    union = low_coverage | high_coverage
    common = low_coverage & high_coverage
    coverage_jaccard = len(common) / max(1, len(union))

    gaps = []
    chart_changes = 0
    for node_id in sorted(common):
        low_chart_id = lower_budget.primary_chart_by_node[node_id]
        high_chart_id = higher_budget.primary_chart_by_node[node_id]
        if low_chart_id != high_chart_id:
            chart_changes += 1
        low_candidate = lower_budget.chart_by_id[str(low_chart_id)].selected_by_node[node_id].candidate
        high_candidate = higher_budget.chart_by_id[str(high_chart_id)].selected_by_node[node_id].candidate
        gaps.append(beta_rms_deg(low_candidate.beta_rad, high_candidate.beta_rad))
    beta_p95 = float(np.percentile(gaps, 95)) if gaps else math.inf
    beta_max = float(np.max(gaps)) if gaps else math.inf
    chart_change = chart_changes / max(1, len(common))

    low_boundary = _boundary_nodes(lower_budget)
    high_boundary = _boundary_nodes(higher_budget)
    boundary_change = len(low_boundary ^ high_boundary) / max(1, len(low_boundary | high_boundary))
    low_edges = _selected_edges(lower_budget)
    high_edges = _selected_edges(higher_budget)
    edge_change = len(low_edges ^ high_edges) / max(1, len(low_edges | high_edges))

    checks = {
        "coverage_jaccard": coverage_jaccard >= active.coverage_jaccard_min - 1e-12,
        "selected_beta_p95": beta_p95 <= active.exploratory_beta_p95_max_deg + 1e-12,
        "selected_beta_max": beta_max <= active.exploratory_beta_max_deg + 1e-12,
        "chart_assignment": chart_change <= active.chart_assignment_change_max + 1e-12,
        "boundary": boundary_change <= active.boundary_change_max + 1e-12,
        "selected_edges": edge_change <= active.selected_edge_change_max + 1e-12,
    }
    raw_cap_hit = bool(lower_budget.raw_cap_hit or higher_budget.raw_cap_hit)
    affected = bool(raw_cap_hit and not all(checks.values()))
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return SectionBudgetStabilityReport(
        common_node_count=len(common),
        coverage_jaccard=float(coverage_jaccard),
        selected_beta_p95_deg=beta_p95,
        selected_beta_max_deg=beta_max,
        chart_assignment_change_ratio=float(chart_change),
        boundary_change_ratio=float(boundary_change),
        selected_edge_change_ratio=float(edge_change),
        raw_cap_hit=raw_cap_hit,
        selected_section_affected_by_cap=affected,
        gate_pass=bool(all(checks.values()) and not affected),
        checks=checks,
        reasons=reasons,
    )


def evaluate_section_patch_gate(
    patch_reports: Sequence[Mapping[str, Any]],
    *,
    development_pass_min: int = 6,
    confirmation_pass_min: int = 3,
    largest_component_ratio_min: float = 0.60,
    single_cell_chart_ratio_max: float = 0.50,
) -> Mapping[str, Any]:
    """Apply the Q18 exploratory 12-patch method-freeze contract."""

    rows = tuple(dict(row) for row in patch_reports)
    development = [row for row in rows if row.get("patch_split") == "development"]
    confirmation = [row for row in rows if row.get("patch_split") == "confirmation"]
    development_pass = sum(bool(row.get("gate_pass", False)) for row in development)
    confirmation_pass = sum(bool(row.get("gate_pass", False)) for row in confirmation)
    passing = [row for row in rows if bool(row.get("gate_pass", False))]
    component_values = [float(row.get("largest_selected_component_ratio", 0.0)) for row in passing]
    median_component = float(np.median(component_values)) if component_values else 0.0
    single_cell_values = [float(row.get("single_cell_chart_ratio", 1.0)) for row in passing]
    median_single_cell = float(np.median(single_cell_values)) if single_cell_values else 1.0
    checks = {
        "all_registered_patches": len(rows) == 12 and len(development) == 8 and len(confirmation) == 4,
        "development_pass_count": development_pass >= int(development_pass_min),
        "confirmation_pass_count": confirmation_pass >= int(confirmation_pass_min),
        "largest_selected_component_ratio": median_component >= float(largest_component_ratio_min) - 1e-12,
        "single_cell_charts_not_dominant": median_single_cell <= float(single_cell_chart_ratio_max) + 1e-12,
    }
    total_pass = development_pass + confirmation_pass
    return MappingProxyType(
        {
            "gate_pass": bool(all(checks.values())),
            "checks": checks,
            "development_pass_count": development_pass,
            "confirmation_pass_count": confirmation_pass,
            "median_largest_selected_component_ratio": median_component,
            "median_single_cell_chart_ratio": median_single_cell,
            "unresolved_patch_ids": sorted(
                str(row.get("patch_id")) for row in rows if not bool(row.get("gate_pass", False))
            ),
            "representation_diagnostic_required": bool(total_pass <= 1),
        }
    )


def _boundary_nodes(result: SectionGrowthResult) -> set[int]:
    return set(result.abstained_node_ids) | {
        node_id
        for chart in result.charts
        for node_id in chart.boundary_risk_node_ids | chart.frontier_node_ids
    }


def _selected_edges(result: SectionGrowthResult) -> set[tuple[str, int, int]]:
    return {
        (chart.chart_id, left, right)
        for chart in result.charts
        for left, right in chart.selected_edges
    }
