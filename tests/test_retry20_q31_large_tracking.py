import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('large_tracking',ROOT/'scripts/analysis/run_retry20_q31_large_tracking.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
SUITE = ROOT/'.agents/skills/quasi-exp-dataset-visualization-report/assets/q31_large_shapes_v1.json'


def test_fixed_full_geometry_and_open_topology():
    suite, frame = m.load_suite(SUITE)
    assert len(frame)==1504 and not frame.target_id.duplicated().any()
    assert frame.groupby('trajectory_id').size().to_dict()=={'large_star':407,'large_helix':523,'large_tapered_helix':574}
    for t in suite['tracks']:
        beta=np.tile(np.arange(len(t['waypoints_m']))[:,None]/10000,(1,6))
        stats=m.step_metrics(beta,t['closed'])
        assert stats['edge_count']==len(beta)-1+int(t['closed'])


def test_shrunk_or_closed_helix_rejected(tmp_path):
    s=json.loads(SUITE.read_text());s['tracks'][1]['closed']=True
    p=tmp_path/'suite.json';p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='topology'):m.load_suite(p)
    s=json.loads(SUITE.read_text());s['tracks'][0]['waypoints_m'][0][1]*=.1;p.write_text(json.dumps(s))
    # Interior edits are protected by the config SHA at runtime; extent edits also fail here.
    s['tracks'][0]['waypoints_m']=(np.asarray(s['tracks'][0]['waypoints_m'])*.1).tolist();p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='scale'):m.load_suite(p)


def test_missing_edges_do_not_become_zero_jumps():
    beta=np.zeros((3,6));beta[1]=np.nan
    stats=m.step_metrics(beta,False)
    assert stats['edge_count']==2 and stats['missing_edge_count']==2 and stats['max_beta_step_deg'] is None


def test_model_lock_hash_mismatch_rejected(tmp_path):
    (tmp_path/'03_student').mkdir();(tmp_path/'03_student/L0.keras').write_text('changed')
    (tmp_path/'03_student/model_lock.json').write_text(json.dumps({'selection':'fixed_final_step','models':{'L0':'0'*64},'datasets':{'L0':'0'*64}}))
    with pytest.raises(ValueError,match='locked input changed'):m.locked_inputs(tmp_path)


def test_preflight_full_denominator_remains_diagnostic(tmp_path):
    _,targets=m.load_suite(SUITE)
    m.preflight(tmp_path,{'teacher_seconds':3600,'deadline_seconds':7200},
        {'scientific_source_fixed_point':'a'*40,'config_sha256':'b'*64},targets,[{'path':str(SUITE),'sha256':m.digest(SUITE)}])
    gate=json.loads((tmp_path/'00_objective_feasibility/gate.json').read_text())
    assert gate['diagnostic_pilot_authorized'] and not gate['claim_bearing_run_authorized']
    assert m.validate_contract_dir(tmp_path/'00_objective_feasibility')==[]
