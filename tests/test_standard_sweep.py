from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from quasi_exp.io.config import load_config
from quasi_exp.model import sampling


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_generate_dataset_module():
    mod_path = REPO_ROOT / "scripts" / "generate_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_dataset", mod_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deg_ranges() -> dict[str, tuple[float, float]]:
    return {
        "beta1": (np.deg2rad(-5.0), np.deg2rad(5.0)),
        "beta2": (np.deg2rad(-5.0), np.deg2rad(5.0)),
        "beta3": (np.deg2rad(-10.0), np.deg2rad(10.0)),
        "beta4": (np.deg2rad(-10.0), np.deg2rad(10.0)),
        "beta5": (np.deg2rad(-15.0), np.deg2rad(15.0)),
        "beta6": (np.deg2rad(-15.0), np.deg2rad(15.0)),
    }


def test_plan_standard_sweep_counts_match_targets() -> None:
    plan_10k = sampling.plan_standard_sweep_counts(_deg_ranges(), num_samples=10_000)
    assert plan_10k.total_samples == 10_000
    assert plan_10k.per_axis_positive_levels == {
        "beta1": 417,
        "beta2": 416,
        "beta3": 833,
        "beta4": 833,
        "beta5": 1249,
        "beta6": 1249,
    }

    plan_100k = sampling.plan_standard_sweep_counts(_deg_ranges(), num_samples=100_000)
    assert plan_100k.total_samples == 100_000
    assert plan_100k.per_axis_positive_levels == {
        "beta1": 4167,
        "beta2": 4166,
        "beta3": 8333,
        "beta4": 8333,
        "beta5": 12499,
        "beta6": 12499,
    }


def test_build_standard_sweep_tasks_center_out_axis_major() -> None:
    ranges = {
        "beta1": (np.deg2rad(-1.0), np.deg2rad(1.0)),
        "beta2": (np.deg2rad(-1.0), np.deg2rad(1.0)),
        "beta3": (np.deg2rad(-2.0), np.deg2rad(2.0)),
        "beta4": (np.deg2rad(-2.0), np.deg2rad(2.0)),
        "beta5": (np.deg2rad(-3.0), np.deg2rad(3.0)),
        "beta6": (np.deg2rad(-3.0), np.deg2rad(3.0)),
    }
    tasks, plan = sampling.build_standard_sweep_tasks(ranges, num_samples=20)

    assert len(tasks) == 20
    assert [task["sample_id"] for task in tasks] == list(range(20))
    assert plan.total_samples == 20

    beta1 = [task for task in tasks if task["scan_axis"] == "beta1"]
    assert len(beta1) == 3
    assert beta1[0]["scan_level"] == 0
    assert beta1[0]["scan_sign"] == 0
    assert np.isclose(beta1[0]["beta6_rad"][0], 0.0)
    assert beta1[1]["scan_level"] == 1
    assert beta1[1]["scan_sign"] == 1
    assert beta1[2]["scan_level"] == 1
    assert beta1[2]["scan_sign"] == -1
    assert np.isclose(beta1[1]["beta6_rad"][0], -beta1[2]["beta6_rad"][0])

    beta5 = [task for task in tasks if task["scan_axis"] == "beta5"]
    assert len(beta5) == 5
    assert np.isclose(beta5[-1]["scan_angle_deg"], -3.0, atol=1e-9)
    assert np.isclose(beta5[-2]["scan_angle_deg"], 3.0, atol=1e-9)


def test_validate_generation_strategy_rejects_inverse_joint_standard_sweep() -> None:
    cfg = {
        "dataset": {"mode": "inverse_joint"},
        "sampling": {
            "strategy": "standard_sweep",
            "beta_ranges_rad": {key: (-0.1, 0.1) for key in [f"beta{i}" for i in range(1, 7)]},
        },
    }
    with np.testing.assert_raises_regex(ValueError, "standard_sweep"):
        sampling.validate_generation_strategy(cfg)


def test_standard_configs_exist_and_use_standard_sweep() -> None:
    for rel_path, expected_out_dir in [
        ("configs/robot_rods_only_standard_10k.yaml", "data/standard_beta_sweep_10k"),
        ("configs/robot_rods_only_standard_100k.yaml", "data/standard_beta_sweep_100k"),
        ("configs/robot_rods_only_standard_2k_paper_strict.yaml", "data/standard_beta_sweep_2k_paper_strict"),
        (
            "configs/robot_rods_only_standard_2k_canonical.yaml",
            "data/standard_beta_sweep_2k_canonical",
        ),
        (
            "configs/robot_rods_only_standard_10k_canonical.yaml",
            "data/standard_beta_sweep_10k_canonical",
        ),
        (
            "configs/robot_rods_only_standard_100k_canonical.yaml",
            "data/standard_beta_sweep_100k_canonical",
        ),
        (
            "configs/robot_rods_only_standard_2k_segmented_canonical.yaml",
            "data/standard_beta_sweep_2k_segmented_canonical",
        ),
        (
            "configs/robot_rods_only_standard_10k_segmented_canonical.yaml",
            "data/standard_beta_sweep_10k_segmented_canonical",
        ),
    ]:
        cfg = load_config(REPO_ROOT / rel_path)
        assert cfg["dataset"]["mode"] == "forward"
        assert cfg["dataset"]["out_dir"] == expected_out_dir
        assert cfg["sampling"]["strategy"] == "standard_sweep"
        assert cfg["sampling"]["standard_sweep"]["step_policy"] == "per_axis_exact"

    paper_cfg = load_config(REPO_ROOT / "configs/robot_rods_only_standard_2k_paper_strict.yaml")
    assert paper_cfg["pso"]["objective"] == "paper_constraint"
    assert paper_cfg["pso"]["w_tension_mean"] == 0.0
    assert paper_cfg["pso"]["w_tension_soft_cap"] == 0.0

    for rel_path in [
        "configs/robot_rods_only_standard_2k_canonical.yaml",
        "configs/robot_rods_only_standard_10k_canonical.yaml",
        "configs/robot_rods_only_standard_100k_canonical.yaml",
    ]:
        canonical_cfg = load_config(REPO_ROOT / rel_path)
        assert canonical_cfg["pso"]["objective"] == "paper_constraint"
        assert canonical_cfg["canonical_tension"]["enabled"] is True
        assert canonical_cfg["canonical_tension"]["method"] == "slsqp_penalty_ref"
        assert canonical_cfg["canonical_tension"]["t_ref_n"] == 800.0
        assert canonical_cfg["canonical_tension"]["maxiter"] == 40

    for rel_path in [
        "configs/robot_rods_only_standard_2k_segmented_canonical.yaml",
        "configs/robot_rods_only_standard_10k_segmented_canonical.yaml",
    ]:
        segmented_cfg = load_config(REPO_ROOT / rel_path)
        assert segmented_cfg["tension_labeler"]["method"] == "segmented_canonical"
        assert segmented_cfg["segmented_tension"]["t_ref_n"] == 800.0
        assert segmented_cfg["segmented_tension"]["max_nfev"] == 80
        assert segmented_cfg["segmented_tension"]["feasible_rms_rnorm"] == 6.0e-2


def test_mixed_distal_preferred_configs_use_reduced_joint_ranges() -> None:
    for rel_path, expected_out_dir in [
        (
            "configs/robot_rods_only_mixed_20k_distal_preferred_segmented_canonical.yaml",
            "data/mixed_beta_20k_distal_preferred_segmented_canonical",
        ),
        (
            "configs/robot_rods_only_mixed_100k_distal_preferred_segmented_canonical.yaml",
            "data/mixed_beta_100k_distal_preferred_segmented_canonical",
        ),
        (
            "configs/robot_rods_only_mixed_20k_distal_preferred_anchor_v1.yaml",
            "data/mixed_beta_20k_distal_preferred_anchor_v1",
        ),
    ]:
        cfg = load_config(REPO_ROOT / rel_path)
        assert cfg["dataset"]["mode"] == "forward"
        assert cfg["dataset"]["out_dir"] == expected_out_dir
        assert cfg["sampling"]["strategy"] == "mixed_beta"
        assert cfg["sampling"]["mixed_beta"]["components"] == {
            "sobol_full": 0.40,
            "lhs_full": 0.20,
            "workspace_balanced": 0.20,
            "distal_biased": 0.20,
        }
        ranges = cfg["sampling"]["beta_ranges_rad"]
        for key in ["beta1", "beta2", "beta3", "beta4"]:
            assert np.allclose(ranges[key], [np.deg2rad(-5.0), np.deg2rad(5.0)])
        for key in ["beta5", "beta6"]:
            assert np.allclose(ranges[key], [np.deg2rad(-10.0), np.deg2rad(10.0)])
        assert cfg["tension_labeler"]["method"] == "segmented_canonical"
        assert cfg["canonical_tension"]["method"] == "segmented_canonical"
        if "anchor_canonical" in cfg:
            assert cfg["anchor_canonical"]["component_scope"] == "same_component"
            assert cfg["anchor_canonical"]["distance_space"] == "beta"
            assert cfg["anchor_canonical"]["k_neighbors"] == 16
            assert cfg["anchor_canonical"]["w_anchor"] == 10.0


def test_sort_output_tables_by_sample_id(tmp_path: Path) -> None:
    module = _load_generate_dataset_module()

    dataset_path = tmp_path / "dataset.parquet"
    meta_path = tmp_path / "dataset_meta.parquet"

    pq.write_table(
        pa.Table.from_pydict(
            {
                "sample_id": [2, 0, 1],
                "x_m": [2.0, 0.0, 1.0],
            }
        ),
        dataset_path,
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pydict(
            {
                "sample_id": [2, 0, 1],
                "scan_axis": ["beta3", "beta1", "beta2"],
            }
        ),
        meta_path,
        compression="zstd",
    )

    module.sort_output_tables_by_sample_id(dataset_path, meta_path)

    dataset_sorted = pq.read_table(dataset_path).to_pandas()
    meta_sorted = pq.read_table(meta_path).to_pandas()

    assert dataset_sorted["sample_id"].tolist() == [0, 1, 2]
    assert meta_sorted["sample_id"].tolist() == [0, 1, 2]
    assert dataset_sorted["x_m"].tolist() == [0.0, 1.0, 2.0]
    assert meta_sorted["scan_axis"].tolist() == ["beta1", "beta2", "beta3"]
