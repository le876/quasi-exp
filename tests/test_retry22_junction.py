from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import pytest

tf=pytest.importorskip('tensorflow')
from quasi_exp.teacher.retry21_raw_student import TrainSampler, build_student, make_update
from quasi_exp.teacher.retry22_junction import JunctionSampler, make_intervention_update
from quasi_exp.teacher.forward import ForwardEnvironment
from quasi_exp.teacher.optimized_forward import optimized_forward
from quasi_exp.teacher.retry12_symmetry import XYZ_COLUMNS,BETA_COLUMNS


@pytest.fixture
def env():
    lengths=np.full(31,.04);lengths[0]=0
    return optimized_forward(ForwardEnvironment(lengths,np.array([.055498,0,0,1]),beta_bounds_rad=np.deg2rad([[-5,5],[-5,5],[-10,10],[-10,10],[-15,15],[-15,15]])))


@pytest.fixture
def data():
    d=pd.DataFrame({'x_m':np.r_[np.full(500,1.055498),np.full(600,1.055498),np.full(500,1.195498)],
        'y_m':np.r_[np.full(500,.03),np.full(600,.15),np.full(500,.03)],'z_m':0.,'split_role':'train'})
    return d


def test_control_sampling_identical_and_intervention_changes_only_registered_rows(data):
    original=TrainSampler(data,1.215498,12);control=JunctionSampler(data,1.215498,12);alt=JunctionSampler(data,1.215498,12)
    for _ in range(3):
        a=original.draw('regions');b=control.draw('regions');c=alt.draw('junction')
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(b[:896],c[:896])
        assert len(c)==1024 and len(set(c[896:]))==128
        assert set(c[896:]).issubset(set(alt.junction))
    state=alt.state();expected=alt.draw('junction');alt.restore(state)
    np.testing.assert_array_equal(expected,alt.draw('junction'))


def test_test_rows_and_missing_junction_rejected(data):
    bad=data.copy();bad.loc[0,'split_role']='test'
    with pytest.raises(ValueError,match='train-only'):JunctionSampler(bad,1.215498,12)
    with pytest.raises(ValueError,match='insufficient'):JunctionSampler(data,1.215498,12,halfwidth_mm=.1)


def test_control_update_matches_retry21(env,data):
    x=data[list(XYZ_COLUMNS)].to_numpy(np.float32);y=np.full((8,6),.01,np.float32);w=np.ones(8,np.float32)
    a=build_student(x,env,12);b=build_student(x,env,12)
    for model in [a,b]:model.get_layer('anchored_beta').anchor_enabled.assign(1.)
    oa=tf.keras.optimizers.Adam(.001,epsilon=1e-7);ob=tf.keras.optimizers.Adam(.001,epsilon=1e-7)
    old=make_update(a,oa,env,True);new=make_intervention_update(b,ob,env,.01)
    for _ in range(2):
        old(x[:8],y,w);new(x[:8],y,w)
    for av,bv in zip(a.get_weights(),b.get_weights()):np.testing.assert_array_equal(av,bv)


def test_zero_coefficient_removes_label_gradient(env,data):
    x=data[list(XYZ_COLUMNS)].to_numpy(np.float32);w=np.ones(8,np.float32)
    a=build_student(x,env,12);b=build_student(x,env,12)
    oa=tf.keras.optimizers.Adam(.001);ob=tf.keras.optimizers.Adam(.001)
    make_intervention_update(a,oa,env,0.)(x[:8],np.zeros((8,6),np.float32),w)
    make_intervention_update(b,ob,env,0.)(x[:8],np.full((8,6),.15,np.float32),w)
    for av,bv in zip(a.get_weights(),b.get_weights()):np.testing.assert_array_equal(av,bv)
    with pytest.raises(ValueError,match='unregistered'):make_intervention_update(a,oa,env,.1)


def test_frozen_input_and_preflight(tmp_path):
    root=Path(__file__).resolve().parents[1];path=root/'scripts/analysis/run_bacra_retry22_junction.py'
    spec=importlib.util.spec_from_file_location('retry22_runner',path);runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    import yaml
    config=yaml.safe_load((root/'configs/bacra_retry22_junction.yaml').read_text())
    (tmp_path/'completion_manifest.json').write_text('{}')
    config['upstream_root']=str(tmp_path)
    config['upstream_completion_sha256']='0'*64
    with pytest.raises(ValueError,match='completion identity'):runner.freeze(config,tmp_path,{})
