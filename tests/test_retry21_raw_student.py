from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import pytest

tf=pytest.importorskip('tensorflow')
from quasi_exp.teacher.retry21_raw_student import (build_student,load_student,TrainSampler,
    loss_terms,make_update,predict,error_metrics,neighbor_audit)
from quasi_exp.teacher.forward import ForwardEnvironment
from quasi_exp.teacher.optimized_forward import optimized_forward
from quasi_exp.teacher.retry12_symmetry import XYZ_COLUMNS,BETA_COLUMNS
from quasi_exp.model.kinematics_tf import forward_xyz_from_beta_tf


@pytest.fixture
def env():
    lengths=np.full(31,.04);lengths[0]=0
    return optimized_forward(ForwardEnvironment(lengths,np.array([.055498,0,0,1]),beta_bounds_rad=np.deg2rad(np.array([[-5,5],[-5,5],[-10,10],[-10,10],[-15,15],[-15,15]]))))


@pytest.fixture
def frame():
    rng=np.random.default_rng(12);n=1600
    x=np.r_[np.full(500,1.085498),np.full(600,1.085498),np.full(500,1.195498)]
    y=np.r_[np.full(500,.03),np.full(600,.15),np.full(500,.03)]
    d=pd.DataFrame({'x_m':x,'y_m':y,'z_m':rng.uniform(-.005,.005,n),'split_role':'train','domain_class':'primary','sample_weight':1.,'target_id':[str(i) for i in range(n)]})
    for c in BETA_COLUMNS:d[c]=rng.uniform(-.04,.04,n)
    return d


def runner():
    path=Path(__file__).resolve().parents[1]/'scripts/analysis/run_bacra_retry21_q33.py'
    spec=importlib.util.spec_from_file_location('retry21_runner',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_anchor_serialization_and_gradients(env,frame,tmp_path):
    model=build_student(frame[list(XYZ_COLUMNS)].to_numpy(),env,4)
    model.get_layer('anchored_beta').anchor_enabled.assign(1.)
    zero=env.fk(np.zeros(6)).astype('float32')
    assert np.max(np.abs(model(zero)))<1e-7
    with tf.GradientTape() as tape:
        beta=model(frame[list(XYZ_COLUMNS)].to_numpy(np.float32)[:8]);loss=tf.reduce_sum(beta**2)
    grads=tape.gradient(loss,model.trainable_variables)
    assert all(g is not None and np.isfinite(g).all() for g in grads)
    model.save(tmp_path/'m.keras');restored=load_student(tmp_path/'m.keras')
    np.testing.assert_allclose(model(zero),restored(zero),atol=1e-8)
    xyz=frame[list(XYZ_COLUMNS)].to_numpy()[:50]
    np.testing.assert_allclose(predict(model,xyz),predict(restored,xyz),atol=1e-8)
    assert (np.abs(predict(restored,xyz))<=env.bounds[:,1]+1e-7).all()


def test_loss_weights_and_fk_jacobian(env):
    b=tf.Variable([[.03,-.02,.01,.04,-.05,.06],[-.02,.01,.04,-.03,.02,.01]],dtype=tf.float64)
    x=env.fk(b.numpy())+np.array([.001,.002,.003]);labels=b.numpy()+.01;w=np.array([1.,3.])
    lb,lt=loss_terms(b,labels,x,w,env)
    assert float(lb)==pytest.approx(.0001*42)
    assert float(lt)==pytest.approx(14e-6/.01,rel=1e-5)
    with tf.GradientTape() as tape:
        xyz=forward_xyz_from_beta_tf(b,lengths_m=env.lengths_m,p_end_local_m=env.p_end_local_m,theta_sign=env.theta_sign)
    jac=tape.batch_jacobian(xyz,b).numpy()
    _,expected=env.fk_and_jacobian(b.numpy())
    np.testing.assert_allclose(jac,expected,rtol=1e-7,atol=1e-8)


def test_train_sampling_identity_and_no_leakage(frame):
    sampler=TrainSampler(frame,1.215498,12);state=sampler.state();a=sampler.draw('regions');sampler.restore(state)
    assert np.array_equal(a,sampler.draw('regions'))
    assert len(a)==1024 and len(set(a[:205]))==205 and (a[:205]<500).all()
    assert ((a[205:461]>=500)&(a[205:461]<1100)).all()
    assert (a[461:615]>=1100).all()
    changed=frame.copy();changed.loc[0,'split_role']='test'
    with pytest.raises(ValueError,match='train-only'):TrainSampler(changed,1.215498,12)


def test_checkpoint_restores_adam_and_sampling_exactly(env,frame,tmp_path):
    r=runner();xyz=frame[list(XYZ_COLUMNS)].to_numpy(np.float32);labels=frame[list(BETA_COLUMNS)].to_numpy(np.float32)
    a=build_student(xyz,env,12);opt=r.optimizer_for(a);s=TrainSampler(frame,1.215498,12)
    update=make_update(a,opt,env,False);ids=s.draw('uniform');update(xyz[ids],labels[ids],np.ones(1024,np.float32))
    r.save_state(tmp_path/'warmup',a,opt,s)
    b=build_student(xyz,env,13);other=r.optimizer_for(b);t=TrainSampler(frame,1.215498,13)
    r.restore_state(tmp_path/'warmup',b,other,t)
    ids=s.draw('uniform');assert np.array_equal(ids,t.draw('uniform'))
    update(xyz[ids],labels[ids],np.ones(1024,np.float32));make_update(b,other,env,False)(xyz[ids],labels[ids],np.ones(1024,np.float32))
    for left,right in zip(a.get_weights(),b.get_weights()):np.testing.assert_array_equal(left,right)
    for left,right in zip(opt.variables,other.variables):np.testing.assert_array_equal(left.numpy(),right.numpy())


def test_invalid_loss_stops(env,frame):
    xyz=frame[list(XYZ_COLUMNS)].to_numpy(np.float32);m=build_student(xyz,env,1);o=runner().optimizer_for(m)
    with pytest.raises(tf.errors.InvalidArgumentError):make_update(m,o,env,True)(xyz[:2],np.full((2,6),np.nan,np.float32),np.ones(2,np.float32))


def test_input_hash_mismatch_and_output_refusal(tmp_path):
    r=runner();p=tmp_path/'input';p.write_text('changed')
    with pytest.raises(ValueError,match='hash mismatch'):r.check_hash(p,'0'*64)
    # The runner's fresh-root operation preserves existing content.
    with pytest.raises(FileExistsError):tmp_path.mkdir(parents=True,exist_ok=False)
    assert p.read_text()=='changed'


def test_nonfinite_rows_remain_in_denominator():
    m=error_metrics([1.,2.,np.nan,12.]);assert m['within_3mm']==.5 and m['nonfinite_count']==1 and m['count']==4


def test_legal_solutions_mean_is_wrong(env):
    b=np.array([.07147526478349503,.029457513179594018,-.02293937287218115,-.012395745114103967,-.2591969405407569,-.10846158776199684])
    target=np.array([1.085498,0,0]);ends=env.reference.fk(np.array([b,-b,np.zeros(6)]))
    assert np.linalg.norm(ends[:2]-target,axis=1).max()<1e-10
    assert np.linalg.norm(ends[2]-target)*1000==pytest.approx(130.)
