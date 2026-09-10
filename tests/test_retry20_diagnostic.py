import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from quasi_exp.teacher.retry20_diagnostic import bounded_dls, solve_anchored, paired_datasets, prediction_metrics
from quasi_exp.teacher.retry12_symmetry import XYZ_COLUMNS, BETA_COLUMNS

class Linear:
    bounds=np.tile([-1.,1.],(6,1))
    def fk(self,b): return np.asarray(b).reshape(-1,6)[:,:3]
    def fk_and_jacobian(self,b): return self.fk(b),np.tile(np.eye(3,6),(len(b),1,1))
    def jacobian(self,b): return np.eye(3,6)


def test_dls_reduces_residual_and_respects_bounds():
    e=Linear();x=np.array([[.1,-.2,.3]])
    b=bounded_dls(e,x,np.zeros((1,6)))
    assert np.linalg.norm(e.fk(b)-x)<1e-6
    b=bounded_dls(e,np.array([[2.,0.,0.]]),np.zeros((1,6)))
    assert np.max(np.abs(b))<=1


def test_teacher_solves_new_target_not_old_label():
    e=Linear();x=np.array([[.1,.1,.1]])
    bank=np.array([[0.,0.,0.,0.,0.,0.],[.01,.01,.01,0.,0.,0.]])
    b,a=solve_anchored(e,x,bank[:1],e.fk(bank),bank,fallback_seeds=2)
    assert a.accepted.all()
    assert np.linalg.norm(e.fk(b)-x)<1e-6
    assert not np.array_equal(b,bank[:1])


def fixtures():
    original=pd.DataFrame({'target_id':['a','b'],'x_m':[1.01,1.02],'y_m':[.01,.02],'z_m':[.01,.02],
        'split_role':['train','train'],'sample_weight':[1.,1.],'domain_class':['primary','primary']})
    for c in BETA_COLUMNS: original[c]=0.
    repl=original.iloc[:1][['target_id',*XYZ_COLUMNS,'split_role']].copy();repl['x_m']=1.015
    return original,repl


def test_failed_relocation_removed_from_both_not_backfilled():
    old,repl=fixtures()
    a,b,keep=paired_datasets(old,repl,repl,np.full((1,6),np.nan),[False])
    assert a.target_id.tolist()==b.target_id.tolist()==['b']
    assert keep.tolist()==[False,True]


def test_success_replaces_xyz_and_beta_only_in_l1():
    old,repl=fixtures()
    a,b,keep=paired_datasets(old,repl,repl,np.ones((1,6))*.1,[True])
    assert a.x_m.iloc[0]==1.01 and b.x_m.iloc[0]==1.015
    assert b.beta1_rad.iloc[0]==.1 and a.beta1_rad.iloc[0]==0


def test_split_and_macroblock_corruption_rejected():
    old,repl=fixtures();repl['split_role']='test'
    with pytest.raises(ValueError,match='split'):paired_datasets(old,repl,repl,np.zeros((1,6)),[True])
    repl['split_role']='train';repl['x_m']=1.08
    with pytest.raises(ValueError,match='macroblock'):paired_datasets(old,repl,repl,np.zeros((1,6)),[True])


def test_teacher_failure_stays_in_success_denominator():
    b=np.zeros((2,6));b[1]=np.nan
    m,_,_=prediction_metrics(Linear(),np.zeros((2,3)),b)
    assert m['within_3mm_rate']==.5 and m['failed_count']==1


def test_real_forward_batch_scalar_parity_and_solver():
    root=Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0,str(root/'scripts/analysis'))
    from run_bacra_retry20_candidate_diagnostic import environment
    env=environment({'project_root':'/mnt/ML_projects/quasi_exp','sources':{'robot_config':'configs/robot_rods_only_standard_100k.yaml'}})
    rng=np.random.default_rng(12)
    beta=rng.uniform(-.03,.03,(8,6))
    xyz=env.reference.fk(beta)
    assert np.max(np.abs(env.fk(beta)-xyz))<1e-10
    fitted=bounded_dls(env,xyz,beta+.001)
    assert np.max(np.linalg.norm(env.reference.fk(fitted)-xyz,axis=1))*1000<3


def test_pair_training_same_inputs_gives_same_models(tmp_path):
    from quasi_exp.teacher.retry20_diagnostic import train_pair
    from run_bacra_retry20_preflight import write_json
    rng=np.random.default_rng(23)
    d=pd.DataFrame(rng.uniform(-.1,.1,(32,3)),columns=XYZ_COLUMNS)
    d['target_id']=[str(i) for i in range(32)]
    for i,c in enumerate(BETA_COLUMNS):d[c]=d.iloc[:,i%3].to_numpy() if i<3 else 0.
    d['split_role']=['train']*24+['validation']*8;d['sample_weight']=1.
    models=train_pair(d,d,Linear(),d[list(XYZ_COLUMNS)].to_numpy(),tmp_path,write_json,steps=2)
    for a,b in zip(models['L0'].get_weights(),models['L1'].get_weights()):np.testing.assert_allclose(a,b,rtol=0,atol=0)
    history=pd.read_parquet(tmp_path/'training_history.parquet')
    assert history.step.tolist()==[2,2]


def test_process_batch_roundtrip(tmp_path):
    import sys,time,yaml
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts/analysis'))
    from run_bacra_retry20_candidate_diagnostic import environment,solve_stage
    config_path=root/'configs/bacra_retry20_candidate_diagnostic.yaml'
    cfg=yaml.safe_load(config_path.read_text());cfg['runtime']['workers']=1
    env=environment(cfg)
    beta=np.array([[.01,.01,.01,.01,.01,.01],[.012,.012,.012,.012,.012,.012]])
    d=pd.DataFrame(env.fk(beta),columns=XYZ_COLUMNS);d['target_id']=['a','b']
    for i,c in enumerate(BETA_COLUMNS):d[c]=beta[:,i]
    target=d[['target_id',*XYZ_COLUMNS]].copy()
    for i,c in enumerate(BETA_COLUMNS):target['anchor_'+c]=beta[:,i]+.0001
    result=solve_stage(config_path,cfg,tmp_path/'stage',target,d,time.time()+60,lambda d,t:None)
    assert len(result)==2 and result.accepted.all()
    assert np.max(result.scalar_fk_residual_mm)<3
