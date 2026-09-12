from pathlib import Path
import importlib.util
import json
import pandas as pd
import pytest


def test_artifact_bound_preflight(tmp_path):
    root=Path(__file__).resolve().parents[1];path=root/'scripts/analysis/run_bacra_retry22_junction.py'
    spec=importlib.util.spec_from_file_location('retry22_artifact_runner',path);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    import yaml
    config=yaml.safe_load((root/'configs/bacra_retry22_junction.yaml').read_text())
    if not Path(config['upstream_root']).exists():pytest.skip('requires frozen retry21 artifacts')
    left,right,targets=runner.freeze(config,tmp_path,{'scientific_source_fixed_point':'a'*40,'config_sha256':'b'*64})
    assert len(left)==len(right)==74696 and len(targets)==9193
    budget=json.loads((tmp_path/'00_objective_feasibility/budget_lower_bound.json').read_text())
    assert budget['diagnostic_budget']['optimizer_updates']==77000
    assert budget['experiment_id']==runner.EXPERIMENT
    assert not runner.validate_contract_dir(tmp_path/'00_objective_feasibility')
    masks=pd.read_parquet(tmp_path/'01_inputs/test_regions.parquet')
    assert len(masks)==8272 and masks.mixed_k32.sum()==1111
    assert json.loads((tmp_path/'00_objective_feasibility/gate.json').read_text())['claim_bearing_run_authorized'] is False
