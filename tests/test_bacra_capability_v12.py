from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quasi_exp.teacher.capability_map import (
    CapabilityPolicy,
    batch_fk,
    batch_jacobian_metrics,
    load_beta_bounds_from_config,
    materialize_capability_region,
    materialize_capability_region_from_samples,
    nested_sobol_beta,
)


class LinearEnvironment:
    def __init__(self, bounds: np.ndarray) -> None:
        self.bounds = np.asarray(bounds, dtype=float)

    def fk(self, beta: np.ndarray) -> np.ndarray:
        values = np.asarray(beta, dtype=float)
        return values[:, :3]

    def jacobian(self, _beta: np.ndarray) -> np.ndarray:
        return np.concatenate([np.eye(3), np.zeros((3, 3))], axis=1)


def _config(tmp_path: Path, *, bound: float = 1.0) -> Path:
    path = tmp_path / "robot.yaml"
    body = "\n".join(
        [
            "sampling:",
            "  beta_ranges_rad:",
            *[f"    beta{index}: [-{bound}, {bound}]" for index in range(1, 7)],
        ]
    )
    path.write_text(body, encoding="utf-8")
    return path


def _policy(config: Path, *, count: int = 64, support: float = 0.0) -> CapabilityPolicy:
    return CapabilityPolicy(
        robot_config_path=config,
        sobol_power=8,
        seed=17,
        task_seed=23,
        voxel_size_m=2.00,
        chunk_rows=31,
        roi_radii_mm_desc=(2000.0, 1000.0),
        centerline_core_support_min=support,
        task_node_count=count,
        knn_k=12,
        max_edge_to_median_nn=20.0,
    )


def test_config_bounds_and_nested_sobol_prefix_are_authoritative(tmp_path: Path) -> None:
    config = _config(tmp_path)
    bounds = load_beta_bounds_from_config(config)
    short = nested_sobol_beta(bounds, power=6, seed=20260730)
    long = nested_sobol_beta(bounds, power=8, seed=20260730)
    assert bounds.shape == (6, 2)
    assert np.array_equal(short, long[: len(short)])
    assert np.all(long >= bounds[:, 0])
    assert np.all(long <= bounds[:, 1])


def test_chunked_fk_and_jacobian_metrics_preserve_scalar_environment_semantics(tmp_path: Path) -> None:
    bounds = load_beta_bounds_from_config(_config(tmp_path))
    environment = LinearEnvironment(bounds)
    beta = nested_sobol_beta(bounds, power=5, seed=11)
    assert np.array_equal(batch_fk(environment, beta, chunk_rows=3), environment.fk(beta))
    metrics = batch_jacobian_metrics(environment, beta)
    assert np.allclose(metrics["sigma1_m"], 1.0)
    assert np.allclose(metrics["sigma2_m"], 1.0)
    assert np.allclose(metrics["sigma3_m"], 1.0)
    assert np.allclose(metrics["kappa"], 1.0)


def test_materialization_builds_deterministic_tiers_roi_and_connected_task_graph(tmp_path: Path) -> None:
    config = _config(tmp_path)
    bounds = load_beta_bounds_from_config(config)
    environment = LinearEnvironment(bounds)
    centerline = np.asarray([[-0.5, -0.5, -0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    policy = _policy(config)
    first = materialize_capability_region(environment, centerline, policy, tmp_path / "one")
    second = materialize_capability_region(environment, centerline, policy, tmp_path / "two")

    assert first.gate.gate_pass
    assert first.task_region.radius_mm == 2000.0
    assert len(first.task_region.nodes) == 64
    assert first.task_region.connected
    assert {"Core-safe", "Feasible-boundary"}.issuperset(set(first.capability_rows["capability_tier"]))
    assert np.array_equal(first.task_region.nodes["capability_sample_id"], second.task_region.nodes["capability_sample_id"])
    assert first.task_region.edges.equals(second.task_region.edges)
    assert (tmp_path / "one" / "capability_map.parquet").exists()
    assert (tmp_path / "one" / "gate.json").exists()


def test_margin_and_support_fail_closed_without_sufficient_core_component(tmp_path: Path) -> None:
    config = _config(tmp_path, bound=0.2)
    bounds = load_beta_bounds_from_config(config)
    environment = LinearEnvironment(bounds)
    centerline = np.asarray([[5.0, 5.0, 5.0], [5.1, 5.1, 5.1]])
    manifest = materialize_capability_region(environment, centerline, _policy(config, support=0.95))
    assert not manifest.gate.gate_pass
    assert manifest.task_region.radius_mm is None
    assert not manifest.gate.checks["centerline_core_support"]


def test_environment_bounds_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    environment = LinearEnvironment(np.tile(np.asarray([[-0.5, 0.5]]), (6, 1)))
    with pytest.raises(ValueError, match="exactly match"):
        materialize_capability_region(environment, np.zeros((2, 3)), _policy(config))


def test_augmented_deterministic_pool_is_explicit_and_legacy_default_stays_strict(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    bounds = load_beta_bounds_from_config(config)
    environment = LinearEnvironment(bounds)
    policy = _policy(config, count=16)
    beta = nested_sobol_beta(bounds, power=5, seed=17)
    xyz = batch_fk(environment, beta, chunk_rows=7)
    metrics = batch_jacobian_metrics(environment, beta)
    centerline = np.asarray([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])

    with pytest.raises(ValueError, match="complete finite frozen Sobol prefix"):
        materialize_capability_region_from_samples(
            environment,
            centerline,
            policy,
            beta_rad=beta,
            xyz_m=xyz,
            jacobian_metrics=metrics,
        )

    manifest = materialize_capability_region_from_samples(
        environment,
        centerline,
        policy,
        beta_rad=beta,
        xyz_m=xyz,
        jacobian_metrics=metrics,
        require_complete_sobol_prefix=False,
    )
    assert manifest.gate.metrics["capability_sample_count"] == len(beta)
