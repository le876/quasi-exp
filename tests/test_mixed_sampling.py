from __future__ import annotations

import numpy as np

from quasi_exp.model import sampling


def _ranges() -> dict[str, tuple[float, float]]:
    return {
        "beta1": (np.deg2rad(-5.0), np.deg2rad(5.0)),
        "beta2": (np.deg2rad(-5.0), np.deg2rad(5.0)),
        "beta3": (np.deg2rad(-5.0), np.deg2rad(5.0)),
        "beta4": (np.deg2rad(-5.0), np.deg2rad(5.0)),
        "beta5": (np.deg2rad(-10.0), np.deg2rad(10.0)),
        "beta6": (np.deg2rad(-10.0), np.deg2rad(10.0)),
    }


def _group_rms(beta: np.ndarray, idx: list[int], limits_deg: list[float]) -> np.ndarray:
    limits = np.deg2rad(np.asarray(limits_deg, dtype=float))
    return np.sqrt(np.mean(np.square(beta[:, idx] / limits[None, :]), axis=1))


def test_build_mixed_beta_tasks_respects_counts_ranges_and_components() -> None:
    tasks, plan = sampling.build_mixed_beta_tasks(
        beta_ranges_rad=_ranges(),
        num_samples=100,
        mixed_cfg={
            "components": {
                "sobol_full": 0.40,
                "lhs_full": 0.20,
                "workspace_balanced": 0.20,
                "distal_biased": 0.20,
            },
            "distal_biased": {
                "proximal_abs_ratio_max": 0.45,
                "distal_abs_ratio_min": 0.55,
            },
        },
        rng_seed=123,
    )

    assert len(tasks) == 100
    assert plan.component_counts == {
        "sobol_full": 40,
        "lhs_full": 20,
        "workspace_balanced": 20,
        "distal_biased": 20,
    }
    assert [task["sample_id"] for task in tasks] == list(range(100))

    beta = np.asarray([task["beta6_rad"] for task in tasks], dtype=float)
    bounds = np.asarray([_ranges()[f"beta{i+1}"] for i in range(6)], dtype=float)
    assert np.all(beta >= bounds[:, 0][None, :] - 1e-12)
    assert np.all(beta <= bounds[:, 1][None, :] + 1e-12)

    components = [task["source_component"] for task in tasks]
    assert components.count("sobol_full") == 40
    assert components.count("lhs_full") == 20
    assert components.count("workspace_balanced") == 20
    assert components.count("distal_biased") == 20


def test_distal_biased_component_uses_third_joint_more_than_first_two() -> None:
    tasks, _ = sampling.build_mixed_beta_tasks(
        beta_ranges_rad=_ranges(),
        num_samples=200,
        mixed_cfg={
            "components": {"distal_biased": 1.0},
            "distal_biased": {
                "proximal_abs_ratio_max": 0.35,
                "distal_abs_ratio_min": 0.65,
            },
        },
        rng_seed=456,
    )

    beta = np.asarray([task["beta6_rad"] for task in tasks], dtype=float)
    g1 = _group_rms(beta, [0, 1], [5.0, 5.0])
    g2 = _group_rms(beta, [2, 3], [5.0, 5.0])
    g3 = _group_rms(beta, [4, 5], [10.0, 10.0])

    assert float(np.median(g3)) > float(np.median(np.maximum(g1, g2)))
    assert float(np.min(g3)) >= 0.65 - 1e-12
    assert float(np.max(np.maximum(g1, g2))) <= 0.35 + 1e-12


def test_workspace_balanced_component_uses_candidate_pool_workspace_function() -> None:
    calls: list[int] = []

    def fake_workspace(beta_rows: np.ndarray) -> np.ndarray:
        calls.append(int(beta_rows.shape[0]))
        return beta_rows[:, [0, 1, 4]]

    tasks, plan = sampling.build_mixed_beta_tasks(
        beta_ranges_rad=_ranges(),
        num_samples=12,
        mixed_cfg={
            "components": {"workspace_balanced": 1.0},
            "workspace_balanced": {
                "candidate_multiplier": 4,
                "radius_bins": 3,
                "z_bins": 3,
                "angle_bins": 4,
            },
        },
        rng_seed=789,
        workspace_xyz_fn=fake_workspace,
    )

    assert len(tasks) == 12
    assert plan.component_counts == {"workspace_balanced": 12}
    assert calls == [48]
    assert {task["source_component"] for task in tasks} == {"workspace_balanced"}


def test_validate_generation_strategy_accepts_mixed_beta_forward_only() -> None:
    cfg = {
        "dataset": {"mode": "forward"},
        "sampling": {
            "strategy": "mixed_beta",
            "beta_ranges_rad": {key: (-0.1, 0.1) for key in [f"beta{i}" for i in range(1, 7)]},
        },
    }
    assert sampling.validate_generation_strategy(cfg) == "mixed_beta"

    cfg["dataset"]["mode"] = "inverse_joint"
    with np.testing.assert_raises_regex(ValueError, "mixed_beta"):
        sampling.validate_generation_strategy(cfg)
