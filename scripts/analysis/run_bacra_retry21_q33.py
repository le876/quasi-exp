#!/usr/bin/env python3
"""Bounded TensorFlow reproduction of Q33 and post-lock outer diagnostics."""
from __future__ import annotations
import argparse
import ast
import shutil
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis'), str(ROOT/'scripts/spec')]
from run_bacra_retry20_candidate_diagnostic import environment, verify_manifest
from quasi_exp.teacher.retry12_symmetry import XYZ_COLUMNS, BETA_COLUMNS
from quasi_exp.teacher.retry19_direct_student import load_direct_student
from quasi_exp.teacher.retry21_raw_student import (build_student, load_student, TrainSampler,
    make_update, predict, regions, error_metrics, neighbor_audit)
import tensorflow as tf
from validate_objective_feasibility import validate_contract_dir

EXPERIMENT = 'bacra_retry21_q33_raw_student_diagnostic'
SMALL = [f'retry19_{stem}_{i}' for stem in ('exact_seam_rectangle','seam_between_rectangle') for i in range(3)] + ['retry19_slanted_ellipse', 'retry19_spiral_3d']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')


def verify_identity(config_path, binding):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    if git('rev-parse', 'HEAD') != binding or git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('requires clean exact binding checkout')
    entry = yaml.safe_load(git('show', binding+':spec/registry.yaml'))['experiments'][EXPERIMENT]
    source = entry['scientific_source_fixed_point']
    subprocess.run(['git','merge-base','--is-ancestor', source, binding], cwd=ROOT, check=True)
    subprocess.run(['git','diff','--exit-code',source,binding,'--','src','scripts/analysis','scripts/pipelines','configs','tests',*entry['protocol_sources']], cwd=ROOT, check=True)
    if str(config_path.relative_to(ROOT)) != entry['config']:
        raise ValueError('unbound config')
    return {'experiment_id': EXPERIMENT, 'scientific_source_fixed_point': source, 'binding_fixed_point': binding,
            'config_sha256': digest(config_path), 'diagnostic_only': True, 'formal_authorized': False}


def check_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f'input hash mismatch: {path}')


def freeze(config, output, ident):
    upstream = Path(config['upstream_root']); q31 = Path(config['q31_root'])
    verify_manifest(upstream, upstream/'completion_manifest.json')
    verify_manifest(q31, q31/'completion_manifest.json')
    source_files = {}
    left = right = None
    for name in ('L0','L1'):
        p = upstream/'02_dataset'/f'{name}_supervision.parquet'
        check_hash(p, config['input_hashes'][name])
        source_files[str(p)] = digest(p)
        df = pd.read_parquet(p)
        if len(df) != 74696 or df.split_role.value_counts().to_dict() != {'train':58629,'test':8272,'validation':7795}:
            raise ValueError('changed supervision denominator')
        if name == 'L0': left = df
        else: right = df
        p = upstream/'03_student'/f'{name}.keras'
        check_hash(p, config['input_hashes'][name+'_model']);source_files[str(p)] = digest(p)
    if not left.target_id.equals(right.target_id) or not left.split_role.equals(right.split_role):
        raise ValueError('changed paired row identity')
    original_path = upstream/'05_evaluation/waypoint_evaluation.parquet'
    q31_path = q31/'05_evaluation/waypoint_evaluation.parquet'
    for path in (original_path, q31_path):
        check_hash(path, config['input_hashes']['original_waypoints' if path == original_path else 'q31_waypoints'])
        source_files[str(path)] = digest(path)
    targets = pd.concat([pd.read_parquet(original_path), pd.read_parquet(q31_path)], ignore_index=True)
    archive = Path(config['q33_bundle']);check_hash(archive, config['input_hashes']['q33_bundle'])
    source_files[str(archive)] = digest(archive)
    with zipfile.ZipFile(archive) as z:
        prefix = 'Q33_raw_student_next_round/'
        manifest = json.loads(z.read(prefix+'manifest.json'))
        for n, entry in manifest['files'].items():
            if hashlib.sha256(z.read(prefix+n)).hexdigest() != entry['sha256']:
                raise ValueError('Q33 bundle member changed: '+n)
        names = ['audit/axis_mirror_counterexample.json','phase2_protocol.json','protocol.json',
                 'matched_controls/frozen_targets.csv','matched_controls/feasibility.csv',
                 'final_audit/anchored_seed1_test_top40.csv','final_audit/anchored_seed2_test_top40.csv']
        for n in names:
            p = output/'01_inputs/q33'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(prefix+n))
        matched = pd.read_csv(io.BytesIO(z.read(prefix+'matched_controls/frozen_targets.csv')))
    matched['target_id'] = matched.case + ':' + matched['index'].astype(str)
    matched['waypoint_index'] = matched['index']
    matched['diagnostic_suite'] = 'q33_matched'
    targets['diagnostic_suite'] = np.where(targets.case.eq('off_grid_test'), 'off_grid', 'original')
    targets = pd.concat([targets, matched], ignore_index=True)
    if targets.target_id.isna().any():
        targets['target_id'] = targets.target_id.fillna(targets.case + ':' + targets.index.astype(str))
    targets['evaluation_id'] = np.arange(len(targets)).astype(str)
    targets.to_parquet(output/'01_inputs/targets.parquet', index=False)
    write_json(output/'01_inputs/source_artifacts.json', source_files)
    pd.DataFrame({'target_id':right.target_id,'split_role':right.split_role}).to_parquet(output/'01_inputs/split_registry.parquet',index=False)
    freeze_feasibility(output, targets, ident, config, source_files)
    return left, right, targets


def freeze_feasibility(output, targets, ident, config, source_files):
    folder = output/'00_objective_feasibility';folder.mkdir()
    paths = targets[~targets.case.eq('off_grid_test')].groupby('case',sort=False).size().rename('point_count').reset_index()
    diameters={case:float(pdist(part[list(XYZ_COLUMNS)].to_numpy(float)).max()*1000) for case,part in targets[~targets.case.eq('off_grid_test')].groupby('case')}
    paths['diameter_mm']=paths.case.map(diameters)
    paths.to_parquet(folder/'trajectory_registry.parquet',index=False)
    registry = {'path':'00_objective_feasibility/trajectory_registry.parquet','sha256':digest(folder/'trajectory_registry.parquet'),'id_column':'case','row_count':len(paths)}
    base={'schema_version':1,'experiment_id':EXPERIMENT}
    write_json(folder/'objective_contract.json',{**base,'scope':'coverage_trajectory','scientific_source_sha':ident['scientific_source_fixed_point'],
      'config_sha256':ident['config_sha256'],'diagnostic_pilot_allowed':True,'primary_objectives':[{'id':'raw_tracking','kind':'complete_trajectory','metric':'all_waypoints_within_3mm',
      'required_for_claim':True,'denominator_id':'fixed_paths','target':{'operator':'>=','value':len(paths),'unit':'count'}}]})
    write_json(folder/'denominator_size.json',{**base,'frozen_before_launch':True,'denominators':[{'id':'fixed_paths','kind':'complete_trajectory','unit':'trajectory','required_count':len(paths),'registry':registry}],
      'resource_denominators':{'required_supervision_vertex_count':74696,'required_logical_edge_count':0,'required_second_parent_certification_count':0},
      'evaluation_points':len(targets),'on_grid_test_points':8272,'off_grid_test_points':3000})
    write_json(folder/'reusable_evidence.json',{**base,'source_artifacts':[{'path':p,'sha256':h} for p,h in source_files.items()],
      'eligible_counts':{'supervision_vertices':58629,'connector_only_vertices':0,'served_coverage_units':0,'complete_trajectories':0,'verified_edges':0,'second_parent_certifications':0},
      'ineligible_counts':{'proposal_only':0,'branch_conflicts':0,'unused':16067},'credit_registry':{'path':'01_inputs/split_registry.parquet','sha256':digest(output/'01_inputs/split_registry.parquet')},
      'proposal_beta_used_as_label_or_hint':False})
    write_json(folder/'budget_lower_bound.json',{**base,'objective_lower_bounds':[{'objective_id':'raw_tracking','method':'not_bounded','required_units':len(paths),'target_units':len(paths),
      'reusable_eligible_units':0,'minimum_resources':{'new_supervision_vertices':0},'basis_registry':registry}],
      'resources':{'new_supervision_vertices':{'optimistic_minimum':0,'registered_budget':0,'basis_registry':registry}},'all_required_objectives_bounded':False,'all_required_resources_feasible':False,
      'diagnostic_budget':{'optimizer_updates':30500,'new_supervision_vertices':0,'deadline_seconds':config['deadline_seconds']},
      'reason':'Training steps cannot certify trajectory success; fixed-budget diagnostic only.'})
    write_json(folder/'atomic_objective_schedule.json',{**base,'row_count_is_stop_condition':False,'scheduled_objective_ids':['raw_tracking'],
      'entries':[{'id':row.case,'objective_id':'raw_tracking','kind':'trajectory','priority':i,'required_for_claim':True,'minimum_diameter_mm':row.diameter_mm,
       'full_cycle_required': not (row.case in ('retry19_spiral_3d','large_helix','large_tapered_helix') or row.case.startswith('retry19_zero_to_boundary_')),'held_out':True,'requirement_registry':registry,'reserved_resources':{'new_supervision_vertices':0,'edge_certificates':0}}
       for i,row in paths.iterrows()]})
    write_json(folder/'gate.json',{**base,'inputs_valid':True,'required_objectives_budget_feasible':False,'status':'diagnostic_only',
      'claim_bearing_run_authorized':False,'diagnostic_pilot_authorized':True,'failed_objective_ids':['raw_tracking'],'reason_codes':['TRAINING_SUCCESS_NOT_BOUNDED']})
    errors=validate_contract_dir(folder)
    if errors:raise ValueError('invalid objective feasibility: '+'; '.join(errors))


def structural_audit(output, right, targets, env):
    folder=output/'02_audit';folder.mkdir()
    train=right[right.split_role.eq('train')].reset_index(drop=True)
    counter=json.loads((output/'01_inputs/q33/audit/axis_mirror_counterexample.json').read_text())
    b=np.asarray([counter['branch_a_beta_rad'],counter['branch_b_beta_rad'],counter['mean_beta_rad']])
    errors=np.linalg.norm(env.reference.fk(b)-counter['target_xyz_m'],axis=1)*1000
    write_json(folder/'axis_counterexample.json',{'reference_fk_residual_mm':errors.tolist(),'within_bounds':bool(((b>=env.bounds[:,0])&(b<=env.bounds[:,1])).all()),'beta_rad':b.tolist(),
      'target_xyz_m':counter['target_xyz_m'],'interpretation':'Legal endpoints do not imply a legal beta average; not proof of the unique cause of outer regression.'})
    chosen=targets[~targets.case.eq('off_grid_test')].reset_index(drop=True)
    audit=pd.DataFrame(neighbor_audit(train,chosen,env))
    audit.to_parquet(folder/'path_neighbor_means.parquet',index=False)
    # Spatial partition audit uses every held-out on-grid test target.
    held=right[right.split_role.eq('test')].reset_index(drop=True)
    pd.DataFrame(neighbor_audit(train,held,env)).to_parquet(folder/'test_neighbor_means.parquet',index=False)
    tx=train[list(XYZ_COLUMNS)].to_numpy(float);tb=train[list(BETA_COLUMNS)].to_numpy(float)
    _,nn=cKDTree(tx).query(tx,k=17)
    a=np.repeat(np.arange(len(tx)),16);bidx=nn[:,1:].ravel()
    domain=train.domain_class.to_numpy()
    cross=domain[a]!=domain[bidx]
    edges=np.unique(np.sort(np.column_stack([a[cross],bidx[cross]]),axis=1),axis=0)
    records=[]
    for start in range(0,len(edges),4096):
        e=edges[start:start+4096];i,j=e.T
        fa=env.fk(tb[i]);fb=env.fk(tb[j]);mid=env.fk((tb[i]+tb[j])*.5)
        gap=np.rad2deg(np.abs(tb[i]-tb[j]))
        for k in range(len(e)):
            records.append({'a':str(train.iloc[i[k]].target_id),'b':str(train.iloc[j[k]].target_id),
              'distance_mm':float(np.linalg.norm(tx[i[k]]-tx[j[k]])*1000),'max_beta_gap_deg':float(gap[k].max()),
              'a_residual_mm':float(np.linalg.norm(fa[k]-tx[i[k]])*1000),'b_residual_mm':float(np.linalg.norm(fb[k]-tx[j[k]])*1000),
              'mean_fk_chord_error_mm':float(np.linalg.norm(mid[k]-(tx[i[k]]+tx[j[k]])*.5)*1000)})
    pd.DataFrame(records).to_parquet(folder/'primary_outer_edges.parquet',index=False)
    write_json(folder/'region_counts.json',{s:{k:int(m.sum()) for k,m in regions(d,env.fk(np.zeros(6))[0,0]).items()}
       for s,d in right.groupby('split_role')})


def optimizer_for(model):
    opt=tf.keras.optimizers.Adam(.001,epsilon=1e-7);opt.build(model.trainable_variables)
    return opt


def save_state(folder,model,opt,sampler):
    folder.mkdir(parents=True,exist_ok=True)
    model.save(folder/'model.keras')
    np.savez(folder/'adam.npz',**{f'v{i}':v.numpy() for i,v in enumerate(opt.variables)})
    write_json(folder/'sampling.json',sampler.state())


def restore_state(folder,model,opt,sampler):
    restored=load_student(folder/'model.keras');model.set_weights(restored.get_weights())
    with np.load(folder/'adam.npz') as a:
        if len(a.files)!=len(opt.variables):raise ValueError('optimizer state mismatch')
        for i,v in enumerate(opt.variables):v.assign(a[f'v{i}'])
    sampler.restore(json.loads((folder/'sampling.json').read_text()))


def training(config,output,left,right,env,deadline,progress):
    folder=output/'03_training';folder.mkdir()
    train=right[right.split_role.eq('train')].reset_index(drop=True)
    valid=right[right.split_role.eq('validation')].reset_index(drop=True)
    norm=left[left.split_role.eq('train')][list(XYZ_COLUMNS)].to_numpy(float)
    x=train[list(XYZ_COLUMNS)].to_numpy(np.float32);y=train[list(BETA_COLUMNS)].to_numpy(np.float32);w=train.sample_weight.fillna(1.).to_numpy(np.float32)
    if not np.isfinite(w).all() or (w<0).any():raise ValueError('invalid sample weights')
    vx=valid[list(XYZ_COLUMNS)].to_numpy(float);zero=float(env.fk(np.zeros(6))[0,0]);history=[];models={};total_updates=0
    def run(name,model,opt,sampler,start,end,phase):
        nonlocal total_updates
        beta_update=make_update(model,opt,env,False);task_update=make_update(model,opt,env,True)
        for step in range(start,end+1):
            if time.monotonic()>deadline:
                save_state(folder/(name+'_interrupted'),model,opt,sampler)
                raise TimeoutError('registered task deadline exceeded; partial checkpoint is not final')
            if phase=='phase2':
                anchored=step>500;model.get_layer('anchored_beta').anchor_enabled.assign(float(anchored))
                mode='regions' if anchored else 'uniform';use_task=anchored
                lr=.001 if step<=6000 else (.0003 if step<=9000 else .0001)
            else:
                mode='core' if name.endswith('balanced') else 'uniform';use_task=name.startswith('fk_')
                lr=.001 if step<=2500 else .0003
            opt.learning_rate.assign(lr)
            ids=sampler.draw(mode)
            loss,lb,lt=(task_update if use_task else beta_update)(x[ids],y[ids],w[ids])
            total_updates+=1
            if step%100==0 or step==end:
                progress('training:'+name,total_updates,30500)
            if step%500==0 or step==end:
                vb=predict(model,vx);err=np.linalg.norm(env.fk(vb)-vx,axis=1)*1000
                row={'arm':name,'step':step,'loss':float(loss),'beta_loss':float(lb),'task_loss':float(lt),'learning_rate':lr}
                for reg,mask in regions(valid,zero).items():
                    row.update({reg+'_'+k:v for k,v in error_metrics(err[mask]).items()})
                history.append(row);pd.DataFrame(history).to_csv(folder/'validation_history.csv',index=False)
                print(json.dumps({'arm':name,'step':step,'validation_p95_mm':row['all_p95_mm'],'core_p95_mm':row['core_p95_mm']}),flush=True)
        save_state(folder/name,model,opt,sampler);models[name]=model
    seed=config['seeds'][0]
    m=build_student(norm,env,seed);o=optimizer_for(m);s=TrainSampler(train,zero,seed)
    run('warmup500',m,o,s,1,500,'phase1')
    branches=[]
    for name in ('beta_uniform','beta_balanced','fk_uniform','fk_balanced'):
        m=build_student(norm,env,seed);o=optimizer_for(m);s=TrainSampler(train,zero,seed)
        restore_state(folder/'warmup500',m,o,s)
        state_hash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in o.variables)+b''.join(v.tobytes() for v in m.get_weights())+json.dumps(s.state(),sort_keys=True).encode()).hexdigest()
        branches.append({'arm':name,'initial_state_sha256':state_hash})
        run(name,m,o,s,501,3000,'phase1')
    if len({b['initial_state_sha256'] for b in branches})!=1:raise ValueError('unfair ablation fork')
    write_json(folder/'branch_state_proof.json',branches)
    for seed in config['seeds']:
        m=build_student(norm,env,seed);o=optimizer_for(m);s=TrainSampler(train,zero,seed)
        run(f'anchored_{seed}',m,o,s,1,10000,'phase2')
    lock={'selection':'fixed_final_step','optimizer_updates':total_updates,'models':{n:{'path':f'03_training/{n}/model.keras','sha256':digest(folder/n/'model.keras')} for n in models},
          'raw_inference':'bounded feedforward; no IK, DLS, trajectory registration or beta expert averaging'}
    write_json(output/'model_lock.json',lock)
    # Evaluate serialized models, not in-memory training objects.
    return {name:load_student(folder/name/'model.keras') for name in models}


def evaluate(output,right,targets,models,env,progress):
    folder=output/'04_evaluation';folder.mkdir();metrics=[];all_predictions={}
    frames={'on_grid':right.reset_index(drop=True),'targets':targets.reset_index(drop=True),
            'zero':pd.DataFrame([{'target_id':'exact_zero','case':'exact_zero','x_m':float(env.fk(np.zeros(6))[0,0]),'y_m':0.,'z_m':0.}])}
    for number,(name,model) in enumerate(models.items()):
        progress('evaluation:'+name,number,len(models))
        model_predictions={}
        for group,frame in frames.items():
            xyz=frame[list(XYZ_COLUMNS)].to_numpy(float);b=predict(model,xyz)
            finite=np.isfinite(b).all(axis=1);actual=np.full_like(xyz,np.nan)
            # Final position metrics are recomputed by the reference scalar FK.
            for start in range(0,len(xyz),2048):
                ids=np.arange(start,min(start+2048,len(xyz)));ids=ids[finite[ids]]
                actual[ids]=env.reference.fk(b[ids])
            err=np.linalg.norm(actual-xyz,axis=1)*1000
            keep=[c for c in ('target_id','evaluation_id','case','waypoint_index','split_role','domain_class',*XYZ_COLUMNS) if c in frame]
            table=frame[keep].copy()
            for j,c in enumerate(BETA_COLUMNS):table['pred_'+c]=b[:,j]
            for j,c in enumerate(XYZ_COLUMNS):table['raw_'+c]=actual[:,j]
            table['error_mm']=err
            table['bounds_violation']=((b<env.bounds[:,0]-1e-7)|(b>env.bounds[:,1]+1e-7)).any(axis=1)
            table.to_parquet(folder/f'{name}_{group}.parquet',index=False);model_predictions[group]=table
            subsets={}
            if group=='on_grid':
                for split,idx in frame.groupby('split_role').indices.items():
                    for region,mask in regions(frame.iloc[idx],env.fk(np.zeros(6))[0,0]).items():subsets[(split,region)]=np.asarray(idx)[mask]
            else:
                for case,idx in frame.groupby('case',sort=False).indices.items():subsets[('diagnostic',case)]=idx
            for (split,case),idx in subsets.items():
                row={'arm':name,'group':group,'split':split,'case':case,**error_metrics(err[idx]),'bounds_violations':int(table.iloc[idx].bounds_violation.sum())}
                if group=='targets' and case!='off_grid_test':
                    delta=actual[idx]-xyz[idx];row.update({f'centroid_d{c}_mm':float(delta[:,j].mean()*1000) for j,c in enumerate('xyz')})
                    radius=lambda q:float(np.sqrt(np.mean(np.sum((q-q.mean(axis=0))**2,axis=1))))
                    row['centered_rms_scale_ratio']=radius(actual[idx])/max(radius(xyz[idx]),1e-15)
                    centered=xyz[idx]-xyz[idx].mean(axis=0);normal=np.linalg.svd(centered,full_matrices=False)[2][-1]
                    planar=np.max(np.abs(centered@normal))*1000<1e-6
                    row['plane_p95_mm']=float(np.percentile(np.abs((actual[idx]-xyz[idx].mean(axis=0))@normal)*1000,95)) if planar else None
                    steps=np.rad2deg(np.abs(np.diff(b[idx],axis=0))).max(axis=1)
                    row['beta_step_max_deg']=float(steps.max());row['beta_step_gt7deg_rate']=float(np.mean(steps>7))
                    row['endpoint_beta_gap_deg']=float(np.rad2deg(np.abs(b[idx[-1]]-b[idx[0]])).max())
                metrics.append(row)
        all_predictions[name]=model_predictions
        pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
    return pd.DataFrame(metrics),all_predictions


def outer_audit(output,right,predictions,env):
    folder=output/'05_outer';folder.mkdir();right=right.reset_index(drop=True)
    baseline=predictions['original_L1']['on_grid'];warm=predictions['warmup500']['on_grid'];selected=set()
    for p in (output/'01_inputs/q33/final_audit').glob('*.csv'):
        selected.update(pd.read_csv(p).target_id.astype(str))
    regressions=[]
    for name,pred in predictions.items():
        if not name.startswith('anchored_'):continue
        p=pred['on_grid'].copy();p['delta_original_L1_mm']=p.error_mm.to_numpy()-baseline.error_mm.to_numpy();p['delta_warmup_mm']=p.error_mm.to_numpy()-warm.error_mm.to_numpy()
        p.to_parquet(folder/f'{name}_pointwise_delta.parquet',index=False)
        tail=p[p.domain_class.eq('retry18_outer')].nlargest(40,'delta_original_L1_mm')
        selected.update(tail.target_id.astype(str));tail.to_csv(folder/f'{name}_outer_top40.csv',index=False)
        for split,part in p.groupby('split_role'):
            for reg,mask in regions(part,env.fk(np.zeros(6))[0,0]).items():
                d=part.iloc[np.flatnonzero(mask)]
                regressions.append({'arm':name,'split':split,'region':reg,'count':len(d),'worse_gt1mm':int((d.delta_original_L1_mm>1).sum()),
                   'worse_gt10mm':int((d.delta_original_L1_mm>10).sum()),'improved_gt1mm':int((d.delta_original_L1_mm < -1).sum()),
                   'max_regression_mm':float(d.delta_original_L1_mm.max()) if len(d) else None})
    pd.DataFrame(regressions).to_csv(folder/'regression_counts.csv',index=False)
    selected_frame=right[right.target_id.astype(str).isin(selected)].reset_index(drop=True)
    selected_frame.to_parquet(folder/'selected_points.parquet',index=False)
    train=right[right.split_role.eq('train')].reset_index(drop=True)
    pd.DataFrame(neighbor_audit(train,selected_frame,env)).to_parquet(folder/'selected_neighbor_means.parquet',index=False)
    tx=train[list(XYZ_COLUMNS)].to_numpy(float);tb=train[list(BETA_COLUMNS)].to_numpy(float)
    dist,idx=cKDTree(tx).query(selected_frame[list(XYZ_COLUMNS)].to_numpy(float),k=32)
    rows=[];interpolation=[]
    for i,target in selected_frame.iterrows():
        ids=idx[i];xyz=target[list(XYZ_COLUMNS)].to_numpy(float)
        for rank,j in enumerate(ids):
            rows.append({'target_id':str(target.target_id),'neighbor_id':str(train.iloc[j].target_id),'neighbor_domain':str(train.iloc[j].domain_class),
                'rank':rank+1,'distance_mm':float(dist[i,rank]*1000),'neighbor_beta_rad':tb[j].tolist()})
        pairs=np.column_stack([np.repeat(ids[0],31),ids[1:]])
        for alpha in (.25,.5,.75):
            beta=(1-alpha)*tb[pairs[:,0]]+alpha*tb[pairs[:,1]]
            chord=(1-alpha)*tx[pairs[:,0]]+alpha*tx[pairs[:,1]]
            actual=env.fk(beta)
            for k in range(len(pairs)):
                interpolation.append({'target_id':str(target.target_id),'neighbor_rank':k+2,'alpha':alpha,
                    'fk_chord_error_mm':float(np.linalg.norm(actual[k]-chord[k])*1000),'fk_target_error_mm':float(np.linalg.norm(actual[k]-xyz)*1000)})
    pd.DataFrame(rows).to_parquet(folder/'neighbor_labels.parquet',index=False)
    pd.DataFrame(interpolation).to_parquet(folder/'neighbor_pair_interpolation.parquet',index=False)
    comparisons=[]
    positions={str(v):i for i,v in enumerate(right.target_id)}
    for _,target in selected_frame.iterrows():
        i=positions[str(target.target_id)];teacher=right.iloc[i][list(BETA_COLUMNS)].to_numpy(float)
        for name,parts in predictions.items():
            p=parts['on_grid'].iloc[i];b=p[['pred_'+c for c in BETA_COLUMNS]].to_numpy(float)
            _,jac=env.fk_and_jacobian(np.asarray([teacher,b]));sv=np.linalg.svd(jac,compute_uv=False)
            comparisons.append({'target_id':str(target.target_id),'arm':name,'split_role':str(target.split_role),'raw_error_mm':float(p.error_mm),
              'teacher_error_mm':float(np.linalg.norm(env.fk(teacher)[0]-target[list(XYZ_COLUMNS)].to_numpy(float))*1000),
              'beta_gap_max_deg':float(np.rad2deg(np.abs(b-teacher)).max()),'beta_norm_ratio':float(np.linalg.norm(b)/max(np.linalg.norm(teacher),1e-15)),
              'teacher_condition':float(sv[0,0]/max(sv[0,-1],1e-15)),'raw_condition':float(sv[1,0]/max(sv[1,-1],1e-15))})
    pd.DataFrame(comparisons).to_csv(folder/'selected_model_comparison.csv',index=False)


def reuse_completed(root, output, config_path):
    """Reuse only byte-identical completed numerics after a preflight metadata repair."""
    manifest=json.loads((root/'completion_manifest.json').read_text())
    if manifest.get('status')!='completed':raise ValueError('reuse requires completed evidence')
    verify_manifest(root,root/'completion_manifest.json')
    if manifest['config_sha256']!=digest(config_path):raise ValueError('reuse requires identical scientific configuration')
    old_source=manifest['scientific_source_fixed_point']
    subprocess.run(['git','diff','--exit-code',old_source,'HEAD','--','src','configs',
                    'docs/protocols/40-BACRA-retry21-Q33-raw-Student执行协议.md'],cwd=ROOT,check=True)
    old_text=subprocess.check_output(['git','show',old_source+':scripts/analysis/run_bacra_retry21_q33.py'],cwd=ROOT,text=True)
    funcs=lambda text:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    before=funcs(old_text);after=funcs(Path(__file__).read_text())
    for name in ('structural_audit','training','evaluate','outer_audit','optimizer_for','save_state','restore_state'):
        if before[name]!=after[name]:raise ValueError('cannot reuse changed numerical stage: '+name)
    # New input freezing must select exactly the same numerical targets and split.
    for relative in ('01_inputs/targets.parquet','01_inputs/split_registry.parquet'):
        if digest(root/relative)!=digest(output/relative):raise ValueError('reused input differs: '+relative)
    reused=[]
    for directory in ('02_audit','03_training','04_evaluation','05_outer'):
        shutil.copytree(root/directory,output/directory)
        for path in sorted((output/directory).rglob('*')):
            if path.is_file():reused.append({'path':str(path.relative_to(output)),'sha256':digest(path)})
    shutil.copy2(root/'model_lock.json',output/'model_lock.json')
    reused.append({'path':'model_lock.json','sha256':digest(output/'model_lock.json')})
    for e in reused:
        if digest(root/e['path'])!=e['sha256']:raise ValueError('reuse copy mismatch')
    provenance={'reason':'Repair objective-feasibility resource and closed-path metadata; training/evaluation code and inputs unchanged.',
      'reused_root':str(root),'training_source_fixed_point':old_source,'training_binding_fixed_point':manifest['binding_fixed_point'],
      'reused_completion_manifest_sha256':digest(root/'completion_manifest.json'),'new_optimizer_updates':0,'reused_artifacts':reused}
    write_json(output/'reuse_provenance.json',provenance)
    return json.loads((root/'summary.json').read_text()),provenance


def main(args):
    config_path=Path(args.config).resolve();config=yaml.safe_load(config_path.read_text())
    ident=verify_identity(config_path,args.binding);output=Path(args.output_root).resolve()
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=start+config['deadline_seconds']
    write_json(output/'run_identity.json',ident)
    def progress(phase,completed=0,total=0):
        write_json(output/'progress.json',{'status':'running','phase':phase,'completed':completed,'total':total,'elapsed_seconds':time.monotonic()-start,'output_root':str(output)})
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.experimental.enable_op_determinism()
        env=environment({'project_root':str(ROOT),'sources':config['sources']})
        progress('freeze');left,right,targets=freeze(config,output,ident)
        if args.reuse_completed_root:
            summary,reuse=reuse_completed(Path(args.reuse_completed_root).resolve(),output,config_path)
            summary.update(ident)
            summary.update(elapsed_seconds=time.monotonic()-start,new_optimizer_updates=0,
                reused_training_source_fixed_point=reuse['training_source_fixed_point'],
                reused_training_binding_fixed_point=reuse['training_binding_fixed_point'],
                metadata_repair_only=True)
        else:
            progress('structural_audit');structural_audit(output,right,targets,env)
            models=training(config,output,left,right,env,deadline,progress)
            models={'original_L0':load_direct_student(Path(config['upstream_root'])/'03_student/L0.keras'),
                    'original_L1':load_direct_student(Path(config['upstream_root'])/'03_student/L1.keras'),**models}
            metrics,predictions=evaluate(output,right,targets,models,env,progress)
            progress('outer_audit');outer_audit(output,right,predictions,env)
            checks={name: bool(len(metrics[(metrics.arm==name)&metrics.case.isin(SMALL)])==8 and
                        (metrics[(metrics.arm==name)&metrics.case.isin(SMALL)].within_3mm==1.).all()) for name in models if name.startswith('anchored_')}
            summary={**ident,'status':'completed','elapsed_seconds':time.monotonic()-start,'model_count':len(models),
                     'small_paths_all_points_3mm_by_seed':checks,'q33_local_result_reproduced':all(checks.values()),
                     'global_replacement_authorized':False,'outer_cause':'diagnostic evidence only; multiple branches not established as necessary',
                     'on_grid_test_count':8272,'off_grid_test_count':3000,'trajectory_count':35,'training_updates':30500}
        write_json(output/'summary.json',summary)
        write_json(output/'progress.json',{'status':'completed','phase':'terminal','completed':30500,'total':30500,'output_root':str(output)})
        write_json(output/'completion_manifest.json',{'status':'completed',**ident,'artifacts':[{'path':str(p.relative_to(output)),'sha256':digest(p)} for p in sorted(output.rglob('*')) if p.is_file()]})
        print(json.dumps(summary),flush=True)
    except Exception as exc:
        write_json(output/'failure.json',{'status':'failed','exception':type(exc).__name__,'message':str(exc),'elapsed_seconds':time.monotonic()-start})
        write_json(output/'progress.json',{'status':'failed','phase':'terminal','message':str(exc),'output_root':str(output)})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--binding',required=True);parser.add_argument('--output-root',required=True);parser.add_argument('--reuse-completed-root')
    main(parser.parse_args())
