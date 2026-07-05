from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "analysis" / "run_gpt5pro_phase1_diagnosis.py"
    spec = importlib.util.spec_from_file_location("run_gpt5pro_phase1_diagnosis", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_dataset(root: Path, *, tension_jump: float) -> tuple[Path, Path]:
    rows = []
    meta = []
    for i in range(6):
        beta = np.ones(6) * (0.01 * i)
        tension = 100.0 + tension_jump * (i % 2)
        row = {"sample_id": i, "x_m": 0.001 * i, "y_m": 0.0, "z_m": 0.0}
        for j in range(30):
            row[f"theta_{j + 1}_rad"] = float(beta[j // 5])
        for j in range(12):
            row[f"tension_{j + 1}_n"] = tension
        rows.append(row)
        m = {
            "sample_id": i,
            "source_component": "sobol_full",
            "rms_rnorm": 0.01,
            "max_tension": tension,
            "segmented_success": True,
        }
        for j in range(6):
            m[f"beta{j + 1}_rad"] = float(beta[j])
        meta.append(m)
    dataset_path = root / "dataset.parquet"
    meta_path = root / "dataset_meta.parquet"
    pd.DataFrame(rows).to_parquet(dataset_path, index=False)
    pd.DataFrame(meta).to_parquet(meta_path, index=False)
    return dataset_path, meta_path


def test_phase1_diagnosis_writes_summary_and_report(tmp_path: Path) -> None:
    mod = _load_module()
    dataset_path, meta_path = _write_dataset(tmp_path, tension_jump=5.0)
    out_dir = tmp_path / "diag"

    summary = mod.run_diagnosis(
        dataset_path=dataset_path,
        meta_path=meta_path,
        anchor_dataset_path=dataset_path,
        anchor_meta_path=meta_path,
        standard_dataset_path=None,
        out_dir=out_dir,
        skip_seed_stability=True,
        skip_existing_fast4=True,
    )

    assert summary["primary_bottleneck"] in {
        "allocator_bottleneck",
        "workspace_multibranch_bottleneck",
        "mixed_bottleneck",
        "labels_already_phase1_silver",
    }
    assert (out_dir / "diagnosis_summary.json").exists()
    assert (out_dir / "diagnosis_report.md").exists()
    assert "oracle_floor" in summary
    assert "branch_clustering" in summary

