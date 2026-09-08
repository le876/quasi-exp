from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_3r_retry12.py"
CONFIG = ROOT / "configs/bacra_v14_3r_retry12_symmetry_quotient_zero_centered.yaml"
PROTOCOL = ROOT / "docs/protocols/29-BACRA-V14.3R-retry12-symmetry-quotient-zero-centered执行协议.md"
LAUNCHER = ROOT / "scripts/pipelines/run_bacra_v14_3r_retry12.sh"


def _module():
    spec = importlib.util.spec_from_file_location("retry12_runner_for_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retry12_stage_order_is_fixed() -> None:
    module = _module()
    assert module.STAGE_ORDER == (
        "symmetry_contract",
        "quotient_target_registry",
        "zero_rooted_outward_atlas",
        "symmetry_fixed_seams",
        "orbit_selection_expansion_split",
        "full_graph_connectivity_audit",
        "freeze_kinematic_dataset",
        "quotient_student",
        "zero_to_target_trajectory_audit",
        "summary",
    )
    assert set(module.STAGE_RUNNERS) == set(module.STAGE_ORDER)


def test_config_freezes_retry12_scientific_contract() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["dataset"]["target_expanded_rows"] == 19999
    assert config["dataset"]["required_size2_orbit_parity"] == "odd"
    assert config["dataset"]["theta_storage"] == "beta_to_theta_without_theta_sign_multiplication"
    assert config["quotient_registry"]["angle_bin_count"] == 8
    assert config["quotient_registry"]["remote_yellow_angle_bins_min"] == 6
    assert config["seams"]["maximum_consecutive_empty_shells"] == 1
    assert config["trajectory"]["corrected_beta_step_catastrophic_rate_max"] == 0.02
    assert not any(config["claims"].values())


def test_protocol_locks_all_six_must_fix_items() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "不能由普通 6D Sobol FK 后再投影坐标" in text
    assert "不得额外乘 `theta_sign`" in text
    assert "quotient macroblock > orbit > row" in text
    assert "6--7/8：YELLOW" in text
    assert "maximum_consecutive_empty_eligible_radial_shells <= 1" in text
    assert "DLS2 corrected" in text
    assert "double_seam_nonzero_abstain" in text


def test_launcher_uses_binding_and_fixed_dag() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    positions = [text.index(f"  {stage}\n") for stage in _module().STAGE_ORDER]
    assert positions == sorted(positions)
    assert "--binding-sha \"$BINDING_SHA\"" in text
    assert "BACRA_RETRY12_BINDING_SHA" in text


def test_runner_never_uses_proposal_beta_as_solver_input() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    solve_section = text[text.index("def _solve_outward_targets"):text.index("def _annotate_new_labels")]
    assert "proposal_beta" not in solve_section
    assert "proposal_beta_scientifically_discarded" in text
    assert "theta_sign_applied_to_storage\": False" in text


def test_constrained_dls_accepts_three_free_coordinates() -> None:
    module = _module()

    class Environment:
        bounds = np.asarray([[-1.0, 1.0]] * 6)

        @staticmethod
        def fk(beta):
            values = np.asarray(beta, dtype=float).reshape(-1, 6)
            return values[:, [1, 3, 5]]

        @staticmethod
        def jacobian(beta):
            del beta
            jacobian = np.zeros((3, 6))
            jacobian[:, [1, 3, 5]] = np.eye(3)
            return jacobian

    target = np.asarray([[0.1, -0.2, 0.3]])
    corrected = module._two_step_dls(
        Environment(), np.zeros((1, 6)), target, frozen_indices=[(0, 2, 4)]
    )

    assert corrected.shape == (1, 6)
    assert np.array_equal(corrected[:, [0, 2, 4]], np.zeros((1, 3)))
    assert np.allclose(corrected[:, [1, 3, 5]], target, atol=1.0e-6)


def test_global_artifact_manifest_excludes_mutable_progress(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "progress.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "run_identity.json").write_text("{}\n", encoding="utf-8")

    manifest = module._artifact_manifest(tmp_path)

    paths = {row["path"] for row in manifest["artifacts"]}
    assert "run_identity.json" in paths
    assert "progress.json" not in paths


def test_summary_stage_marks_progress_terminal_complete() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    branch = text[text.index('if stage_name == "summary"'):text.index("return result", text.index('if stage_name == "summary"'))]
    assert '"status": "complete"' in branch
    assert '"completed": len(STAGE_ORDER)' in branch
