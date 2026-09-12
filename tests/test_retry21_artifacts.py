"""Artifact-bound checks; no training and no modification of source artifacts."""
from pathlib import Path
import importlib.util
import yaml
import pytest


def test_bundle_freeze_contract(tmp_path):
    root=Path(__file__).resolve().parents[1]
    path=root/'scripts/analysis/run_bacra_retry21_q33.py'
    spec=importlib.util.spec_from_file_location('retry21_artifact_runner',path)
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    config=yaml.safe_load((root/'configs/bacra_retry21_q33.yaml').read_text())
    if not Path(config['upstream_root']).exists():pytest.skip('requires frozen retry20 artifacts')
    left,right,targets=r.freeze(config,tmp_path,{'scientific_source_fixed_point':'a'*40,'config_sha256':'b'*64})
    assert len(targets[targets.case.eq('off_grid_test')])==3000
    assert targets[~targets.case.eq('off_grid_test')].case.nunique()==35
    assert len(right[right.split_role.eq('train')])==58629
    from validate_objective_feasibility import validate_contract_dir
    assert validate_contract_dir(tmp_path/'00_objective_feasibility')==[]
    previous=Path('/mnt/ML_projects/quasi_exp/runs/bacra_retry21_q33_attempt1')
    if previous.exists():
        summary,proof=r.reuse_completed(previous,tmp_path,root/'configs/bacra_retry21_q33.yaml')
        assert summary['q33_local_result_reproduced'] is True
        assert proof['new_optimizer_updates']==0 and len(proof['reused_artifacts'])>60
        assert r.digest(tmp_path/'model_lock.json')==r.digest(previous/'model_lock.json')


def test_failed_attempt1_preflight_is_preserved_and_detected():
    import json
    root=Path('/mnt/ML_projects/quasi_exp/runs/bacra_retry21_q33_attempt1')
    if not root.exists():pytest.skip('requires recorded attempt1 metadata defect')
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/spec'))
    from validate_objective_feasibility import validate_contract_dir
    errors=validate_contract_dir(root/'00_objective_feasibility')
    assert any('resources must be non-empty' in e for e in errors)
    assert any('full-cycle' in e for e in errors)
    assert json.loads((root/'summary.json').read_text())['q33_local_result_reproduced'] is True
