from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_bacra_v14_2_section_first_atlas.py"


def _module():
    spec = importlib.util.spec_from_file_location("bacra_v14_2_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v14_2_config_registers_section_methods_and_twelve_workers() -> None:
    module = _module()
    config = module.load_config(ROOT / "configs/bacra_v14_2_section_first_atlas.yaml")

    assert config["parallel"]["patch_workers"] == 12
    assert config["parallel"]["numerical_threads_per_worker"] == 1
    assert config["diagnostics"]["methods"] == ["F0", "F1", "S2", "S4", "S8", "S4C"]
    assert config["section_first"]["beam_widths"] == [2, 4, 8]
    assert config["exploratory_dataset"]["row_padding"] is False


def test_v14_2_runner_has_fail_closed_conditional_stage_order() -> None:
    module = _module()

    assert module.STAGE_ORDER == (
        "inventory",
        "diagnostics",
        "mechanisms",
        "confirmation",
        "reach_round6",
        "repaired_pilot",
        "exploratory_student",
        "formal_gate",
        "summary",
    )
    assert set(module.STAGE_RUNNERS) == set(module.STAGE_ORDER)
