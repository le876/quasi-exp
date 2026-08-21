from __future__ import annotations

import math

import numpy as np

from quasi_exp.teacher.atlas_audit import (
    AuditMetric,
    AtlasAuditPlan,
    AtlasAuditPolicy,
    PathTrace,
    audit_atlas,
    plan_atlas_audit,
    reduce_planned_atlas_audit,
    replay_product_graph_path,
)
from quasi_exp.teacher.canonical_atlas import (
    AtlasCandidate,
    AtlasPolicy,
    AtlasTaskNode,
    ContinuationOutcome,
    FROZEN_EDGE_COST_WEIGHTS,
    build_canonical_atlas,
    build_product_graph,
)


def _beta(deg: float) -> np.ndarray:
    values = np.zeros(6, dtype=float)
    values[0] = math.radians(deg)
    return values


def _candidate(node_id: int, candidate_id: str, deg: float, *, quality: str = "Gold") -> AtlasCandidate:
    return AtlasCandidate(
        node_id=node_id,
        candidate_id=candidate_id,
        beta_rad=_beta(deg),
        residual_mm=0.1,
        min_margin_deg=2.0,
        normalized_min_margin=0.5,
        posture_cost=0.2,
        condition_number=2.0,
        quality=quality,
        solver_success=True,
        actual_bounds=True,
    )


def _diamond_nodes() -> tuple[AtlasTaskNode, ...]:
    return (
        AtlasTaskNode(0, [0.0, 0.0, 0.0], (1, 2)),
        AtlasTaskNode(1, [0.001, 0.0, 0.0], (0, 3)),
        AtlasTaskNode(2, [0.0, 0.001, 0.0], (0, 3)),
        AtlasTaskNode(3, [0.001, 0.001, 0.0], (1, 2)),
    )


def _exact_continuation(by_node: dict[int, list[AtlasCandidate]]):
    def solve(source: AtlasCandidate, target: AtlasTaskNode) -> ContinuationOutcome:
        target_candidate = min(
            by_node[target.node_id],
            key=lambda candidate: float(np.linalg.norm(candidate.beta_rad - source.beta_rad)),
        )
        return ContinuationOutcome(
            beta_rad=target_candidate.beta_rad,
            residual_mm=0.1,
            success=True,
            actual_bounds=True,
            status="exact-test-corrector",
        )

    return solve


def test_product_graph_requires_both_directed_continuations_and_uses_frozen_cost() -> None:
    nodes = _diamond_nodes()
    candidates = tuple(_candidate(node, f"n{node}_c0", 0.1 * node) for node in range(4))
    by_node = {node: [candidate] for node, candidate in enumerate(candidates)}
    graph = build_product_graph(nodes, candidates, _exact_continuation(by_node))

    assert len(graph.directed_edges) == 8
    assert len(graph.robust_edges) == 4
    assert dict(FROZEN_EDGE_COST_WEIGHTS) == {
        "transition": 1.0,
        "posture": 0.05,
        "condition": 0.01,
        "margin": 0.10,
    }
    assert graph.robust_edges[0].cost > graph.robust_edges[0].transition_deg

    def one_way(source: AtlasCandidate, target: AtlasTaskNode) -> ContinuationOutcome:
        outcome = _exact_continuation(by_node)(source, target)
        if source.node_id > target.node_id:
            return ContinuationOutcome(
                beta_rad=outcome.beta_rad,
                residual_mm=outcome.residual_mm,
                success=False,
                actual_bounds=True,
                status="reverse-failed",
            )
        return outcome

    one_way_graph = build_product_graph(nodes, candidates, one_way)
    assert len(one_way_graph.directed_edges) == 4
    assert one_way_graph.robust_edges == ()


def test_missing_audit_metric_is_strict_json_safe_and_fail_closed() -> None:
    metric = AuditMetric(
        sample_count=0,
        p95_deg=math.inf,
        max_deg=math.inf,
        missing_count=3,
        gate_pass=False,
    )

    assert metric.as_dict() == {
        "sample_count": 0,
        "p95_deg": None,
        "max_deg": None,
        "missing_count": 3,
        "gate_pass": False,
    }


def test_chart_ids_are_deterministic_and_distinct_branches_do_not_merge() -> None:
    nodes = (
        AtlasTaskNode(0, [0.0, 0.0, 0.0], (1,)),
        AtlasTaskNode(1, [0.001, 0.0, 0.0], (0,)),
    )
    candidates = (
        _candidate(0, "n0_a", 0.0),
        _candidate(0, "n0_b", 3.0),
        _candidate(1, "n1_a", 0.1),
        _candidate(1, "n1_b", 3.1),
    )
    by_node = {
        0: [candidates[0], candidates[1]],
        1: [candidates[2], candidates[3]],
    }
    atlas = build_canonical_atlas(
        nodes,
        candidates,
        _exact_continuation(by_node),
        policy=AtlasPolicy(root_count=1, top_section_count=8),
    )

    rerun = build_canonical_atlas(
        nodes,
        candidates,
        _exact_continuation(by_node),
        policy=AtlasPolicy(root_count=1, top_section_count=8),
    )
    assert [chart.chart_id for chart in atlas.charts] == [0, 1]
    assert [chart.root_key for chart in atlas.charts] == [chart.root_key for chart in rerun.charts]
    assert {chart.root_key for chart in atlas.charts} == {(0, "n0_a"), (0, "n0_b")}
    assert len(atlas.overlap_reports) == 1
    assert atlas.overlap_reports[0].resolution == "separate_distinct_branch"
    assert atlas.overlap_reports[0].gap_max_deg > 1.0


def test_audit_accepts_path_independent_static_chart_with_independent_repeat() -> None:
    nodes = _diamond_nodes()
    candidates = tuple(_candidate(node, f"n{node}_c0", 0.1 * node) for node in range(4))
    by_node = {node: [candidate] for node, candidate in enumerate(candidates)}
    atlas = build_canonical_atlas(
        nodes,
        candidates,
        _exact_continuation(by_node),
        policy=AtlasPolicy(root_count=1),
    )
    assert len(atlas.charts) == 1

    def path_executor(chart, path):
        return replay_product_graph_path(atlas, chart, path)

    def repeat_executor(chart, path):
        trace = replay_product_graph_path(atlas, chart, path)
        return (trace, trace)

    report = audit_atlas(
        atlas,
        policy=AtlasAuditPolicy(endpoint_count=1, paths_per_endpoint=2, loop_count=1),
        path_executor=path_executor,
        repeat_executor=repeat_executor,
    )
    assert report.gate_pass is True
    assert report.representation_decision == "static_xyz_to_beta6"
    assert report.path.gate_pass and report.loop.gate_pass
    assert report.direction.gate_pass and report.repeat.gate_pass


def test_planned_audit_reduction_matches_legacy_serial_audit() -> None:
    nodes = _diamond_nodes()
    candidates = tuple(
        _candidate(node, f"n{node}_c0", 0.1 * node)
        for node in range(4)
    )
    by_node = {
        node: [candidate] for node, candidate in enumerate(candidates)
    }
    atlas = build_canonical_atlas(
        nodes,
        candidates,
        _exact_continuation(by_node),
        policy=AtlasPolicy(root_count=1),
    )
    policy = AtlasAuditPolicy(
        endpoint_count=1, paths_per_endpoint=2, loop_count=1
    )

    def execute(chart, path):
        return replay_product_graph_path(atlas, chart, path)

    legacy = audit_atlas(
        atlas,
        policy=policy,
        path_executor=execute,
        repeat_executor=lambda chart, path: (
            execute(chart, path),
            execute(chart, path),
        ),
    )
    plan = plan_atlas_audit(atlas, policy=policy)
    roundtrip = AtlasAuditPlan.from_dict(plan.as_dict())
    charts = {chart.chart_id: chart for chart in atlas.charts}
    traces = {
        task.task_id: execute(charts[task.chart_id], task.path)
        for task in roundtrip.tasks
    }
    parallel = reduce_planned_atlas_audit(
        atlas,
        roundtrip,
        traces,
        evidence_limitations=("planned_test_execution",),
    )

    assert parallel.gate_pass == legacy.gate_pass
    assert parallel.representation_decision == legacy.representation_decision
    for name in ("path", "loop", "direction", "repeat", "overlap"):
        assert getattr(parallel, name).as_dict() == getattr(
            legacy, name
        ).as_dict()


def test_audit_does_not_call_missing_repeat_evidence_stable_and_detects_path_dependence() -> None:
    nodes = _diamond_nodes()
    candidates = tuple(_candidate(node, f"n{node}_c0", 0.1 * node) for node in range(4))
    by_node = {node: [candidate] for node, candidate in enumerate(candidates)}
    atlas = build_canonical_atlas(nodes, candidates, _exact_continuation(by_node), policy=AtlasPolicy(root_count=1))

    no_repeat = audit_atlas(
        atlas,
        policy=AtlasAuditPolicy(endpoint_count=1, paths_per_endpoint=2, loop_count=1),
    )
    assert no_repeat.repeat.gate_pass is False
    assert "independent_repeat_executor_not_supplied" in no_repeat.evidence_limitations

    def divergent_executor(chart, path):
        trace = replay_product_graph_path(atlas, chart, path)
        beta = trace.beta_rad_by_node.copy()
        if 2 in path:
            beta[-1, 0] += math.radians(2.0)
        return PathTrace(trace.node_ids, beta, trace.success, "path-dependent-test")

    def stable_repeat(chart, path):
        trace = divergent_executor(chart, path)
        return (trace, trace)

    report = audit_atlas(
        atlas,
        policy=AtlasAuditPolicy(endpoint_count=1, paths_per_endpoint=2, loop_count=1),
        path_executor=divergent_executor,
        repeat_executor=stable_repeat,
    )
    assert report.path.gate_pass is False
    assert report.representation_decision == "stateful_xyz_beta_prev_to_delta_beta"
