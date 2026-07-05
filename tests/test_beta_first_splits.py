from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    mod_path = REPO_ROOT / "scripts" / "baselines" / "run_beta_first_baselines.py"
    spec = importlib.util.spec_from_file_location("run_beta_first_baselines", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_beta_first_uses_existing_split_strategies() -> None:
    mod = _load_module()
    xyz = np.column_stack([np.linspace(0, 1, 20), np.zeros(20), np.linspace(0, 0.2, 20)])
    beta = np.column_stack([np.linspace(0, 1, 20) for _ in range(6)])

    for split in ["iid", "radius", "beta_block", "angular_sector"]:
        train_idx, val_idx, test_idx = mod.make_split(split, xyz, beta, seed=1, val_size=0.1, test_size=0.2)
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(train_idx) + len(val_idx) + len(test_idx) == 20
        assert len(test_idx) == 4


def test_load_beta_columns_from_meta_are_ordered() -> None:
    mod = _load_module()
    meta = pd.DataFrame({f"beta{i}_rad": [float(i)] for i in range(1, 7)})
    beta = mod.beta_matrix(meta)
    assert beta.shape == (1, 6)
    assert beta.tolist()[0] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

