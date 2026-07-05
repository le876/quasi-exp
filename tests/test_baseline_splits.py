from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "baselines"))

from splits import split_angular_sector_holdout, split_beta_block_holdout  # noqa: E402


def _assert_partition(train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, n: int) -> None:
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert len(all_idx) == n
    assert len(np.unique(all_idx)) == n
    assert sorted(all_idx.tolist()) == list(range(n))


def test_split_beta_block_holdout_partitions_by_largest_beta_norm() -> None:
    beta = np.zeros((10, 6), dtype=float)
    beta[:, 4] = np.linspace(0.0, 1.0, 10)

    train_idx, val_idx, test_idx = split_beta_block_holdout(beta, val_size=0.2, test_size=0.2)

    _assert_partition(train_idx, val_idx, test_idx, n=10)
    assert set(test_idx.tolist()) == {8, 9}
    assert set(val_idx.tolist()) == {6, 7}


def test_split_angular_sector_holdout_uses_contiguous_workspace_sectors() -> None:
    angles = np.linspace(-np.pi, np.pi, 12, endpoint=False)
    xyz = np.column_stack([np.ones(12), np.cos(angles), np.sin(angles)])

    train_idx, val_idx, test_idx = split_angular_sector_holdout(xyz, val_size=0.25, test_size=0.25)

    _assert_partition(train_idx, val_idx, test_idx, n=12)
    assert len(test_idx) == 3
    assert len(val_idx) == 3
    assert set(test_idx.tolist()).isdisjoint(set(val_idx.tolist()))
