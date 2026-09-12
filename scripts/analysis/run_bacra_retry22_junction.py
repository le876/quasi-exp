#!/usr/bin/env python3
"""Fixed-data paired factorial diagnosis of junction sampling and label loss."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis'), str(ROOT/'scripts/spec')]
from run_bacra_retry21_q33 import (digest, write_json, optimizer_for, save_state,
    restore_state, evaluate, SMALL)
from run_bacra_retry20_candidate_diagnostic import environment, verify_manifest
from quasi_exp.teacher.retry12_symmetry import XYZ_COLUMNS, BETA_COLUMNS
from quasi_exp.teacher.retry21_raw_student import build_student, load_student, make_update, predict, error_metrics, regions
from quasi_exp.teacher.retry22_junction import JunctionSampler, make_intervention_update
from validate_objective_feasibility import validate_contract_dir
import tensorflow as tf

EXPERIMENT = 'bacra_retry22_junction_factorial_diagnostic'


def identity(config_path, binding):
    def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    if git('rev-parse', 'HEAD') != binding or git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('requires clean exact binding checkout')
    entry = yaml.safe_load(git('show', binding+':spec/registry.yaml'))['experiments'][EXPERIMENT]
    source = entry['scientific_source_fixed_point']
    subprocess.run(['git','merge-base','--is-ancestor',source,binding],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--exit-code',source,binding,'--','src','scripts/analysis','configs','tests',*entry['protocol_sources']],cwd=ROOT,check=True)
    if str(config_path.relative_to(ROOT)) != entry['config']:
        raise ValueError('unbound config')
    return {'experiment_id':EXPERIMENT,'scientific_source_fixed_point':source,'binding_fixed_point':binding,
            'config_sha256':digest(config_path),'diagnostic_only':True,'formal_authorized':False}


def feasibility(upstream, output, config, ident, sources):
    """Reuse unchanged path geometry, reconstruct the new attempt's contract."""
    dest=output/'00_objective_feasibility';dest.mkdir()
    old=upstream/'00_objective_feasibility'
    shutil.copyfile(old/'trajectory_registry.parquet',dest/'trajectory_registry.parquet')
    for name in ['objective_contract','denominator_size','reusable_evidence','budget_lower_bound','atomic_objective_schedule','gate']:
        value=json.loads((old/f'{name}.json').read_text())
        value['experiment_id']=EXPERIMENT
        if name=='objective_contract':
            value['scientific_source_sha']=ident['scientific_source_fixed_point'];value['config_sha256']=ident['config_sha256']
        if name=='reusable_evidence':
            value['source_artifacts']=[{'path':p,'sha256':h} for p,h in sources.items()]
        if name=='budget_lower_bound':
            value['diagnostic_budget']={'optimizer_updates':config['optimizer_updates'],'new_supervision_vertices':0,'deadline_seconds':config['deadline_seconds']}
        write_json(dest/f'{name}.json',value)
    errors=validate_contract_dir(dest)
    if errors: raise ValueError('invalid preflight: '+'; '.join(errors))


def freeze(config, output, ident):
    upstream=Path(config['upstream_root']);dataset=Path(config['dataset_root'])
    if digest(upstream/'completion_manifest.json') != config['upstream_completion_sha256']:
        raise ValueError('upstream completion identity changed')
    verify_manifest(upstream,upstream/'completion_manifest.json')
    sourcefiles={str(upstream/'completion_manifest.json'):digest(upstream/'completion_manifest.json')}
    frames=[]
    for name in ['L0','L1']:
        path=dataset/'02_dataset'/f'{name}_supervision.parquet'
        if digest(path)!=config['input_hashes'][name]: raise ValueError('changed dataset')
        sourcefiles[str(path)]=digest(path);frames.append(pd.read_parquet(path))
    left,right=frames
    if len(right)!=74696 or right.target_id.duplicated().any(): raise ValueError('invalid data denominator')
    if right.groupby('split_role').size().to_dict()!={'train':58629,'validation':7795,'test':8272}: raise ValueError('changed split')
    folder=output/'01_inputs';folder.mkdir()
    for name in ['targets.parquet','split_registry.parquet']:
        path=upstream/'01_inputs'/name;shutil.copyfile(path,folder/name);sourcefiles[str(path)]=digest(path)
    targets=pd.read_parquet(folder/'targets.parquet')
    if len(targets)!=9193 or targets.loc[targets.case.ne('off_grid_test'),'case'].nunique()!=35: raise ValueError('changed trajectory denominator')
    train=right[right.split_role.eq('train')];test=right[right.split_role.eq('test')]
    _,nn=cKDTree(train[list(XYZ_COLUMNS)]).query(test[list(XYZ_COLUMNS)],k=32)
    dom=train.domain_class.to_numpy()[nn]
    mask=test[['target_id']].copy();mask['mixed_k32']=(dom!=dom[:,0,None]).any(axis=1)
    for width in [5,10,20]:mask[f'junction_{width}mm']=(test.x_m*1000-config['junction']['center_x_mm']).abs()<=width
    mask.to_parquet(folder/'test_regions.parquet',index=False)
    pool=train.loc[(train.x_m*1000-config['junction']['center_x_mm']).abs()<=config['junction']['halfwidth_mm'],['target_id']]
    pool.to_parquet(folder/'junction_train_pool.parquet',index=False)
    write_json(folder/'source_artifacts.json',sourcefiles)
    write_json(folder/'analysis_plan.json',{'test_count':len(test),'mixed_k32_count':int(mask.mixed_k32.sum()),'junction_train_count':len(pool),
        'severe_error_mm':50,'candidate_rule':'both seeds test >50mm count at most half control, positive control count, all eight small paths <=3mm',
        'posthoc_reused_test':True,'new_supervision':0,'arms':config['arms']})
    feasibility(upstream,output,config,ident,sourcefiles)
    return left,right,targets


def state_hash(model,opt,sampler):
    return hashlib.sha256(b''.join(v.tobytes() for v in model.get_weights())+
        b''.join(v.numpy().tobytes() for v in opt.variables)+json.dumps(sampler.state(),sort_keys=True).encode()).hexdigest()


def training(config,output,left,right,env,deadline,progress):
    folder=output/'03_training';folder.mkdir()
    tr=right[right.split_role.eq('train')].reset_index(drop=True);va=right[right.split_role.eq('validation')]
    norm=left.loc[left.split_role.eq('train'),list(XYZ_COLUMNS)].to_numpy(float)
    x=tr[list(XYZ_COLUMNS)].to_numpy(np.float32);y=tr[list(BETA_COLUMNS)].to_numpy(np.float32);w=tr.sample_weight.fillna(1).to_numpy(np.float32)
    if not np.isfinite(w).all() or (w<0).any(): raise ValueError('invalid row weights')
    vx=va[list(XYZ_COLUMNS)].to_numpy(float);zero=float(env.fk(np.zeros(6))[0,0])
    if abs(zero*1000-200-config['junction']['center_x_mm'])>1e-6: raise ValueError('junction coordinate mismatch')
    updates=0;history=[];proof=[];models={}

    def setup(seed):
        model=build_student(norm,env,seed);opt=optimizer_for(model)
        sampler=JunctionSampler(tr,zero,seed,config['junction']['halfwidth_mm'],config['junction']['batch_count'])
        return model,opt,sampler

    def run(name,model,opt,sampler,start,end,sampling,beta_coefficient=None):
        nonlocal updates
        update=make_update(model,opt,env,False) if beta_coefficient is None else make_intervention_update(model,opt,env,beta_coefficient)
        batches=hashlib.sha256()
        for step in range(start,end+1):
            if time.monotonic()>deadline:
                save_state(folder/(name+'_interrupted'),model,opt,sampler)
                raise TimeoutError('fixed task budget exhausted')
            opt.learning_rate.assign(.001 if step<=6000 else (.0003 if step<=9000 else .0001))
            ids=sampler.draw(sampling);batches.update(np.asarray(ids,dtype='<i8').tobytes())
            loss,lb,lt=update(x[ids],y[ids],w[ids]);updates+=1
            if step%250==0 or step==end: progress('training:'+name,updates,config['optimizer_updates'])
            if step%500==0 or step==end:
                error=np.linalg.norm(env.fk(predict(model,vx))-vx,axis=1)*1000
                row={'arm':name,'step':step,'loss':float(loss),'beta_loss':float(lb),'task_loss':float(lt)}
                for region,mask in regions(va,zero).items(): row.update({region+'_'+k:v for k,v in error_metrics(error[mask]).items()})
                history.append(row);pd.DataFrame(history).to_csv(folder/'validation_history.csv',index=False)
                print(json.dumps({'arm':name,'step':step,'validation_p95_mm':row['all_p95_mm']}),flush=True)
        save_state(folder/name,model,opt,sampler)
        return batches.hexdigest()

    for seed in config['seeds']:
        model,opt,sampler=setup(seed);warm=f'warmup_{seed}'
        run(warm,model,opt,sampler,1,config['warmup_steps'],'uniform')
        for arm,settings in config['arms'].items():
            model,opt,sampler=setup(seed);restore_state(folder/warm,model,opt,sampler)
            h=state_hash(model,opt,sampler)
            model.get_layer('anchored_beta').anchor_enabled.assign(1.0)
            name=f'{arm}_{seed}'
            batchhash=run(name,model,opt,sampler,config['warmup_steps']+1,config['final_step'],settings['sampling'],settings['beta_coefficient'])
            proof.append({'arm':arm,'seed':seed,'initial_state_sha256':h,'batch_sha256':batchhash,'sampling':settings['sampling']})
            models[name]={'path':f'03_training/{name}/model.keras','sha256':digest(folder/name/'model.keras')}
        block=[row for row in proof if row['seed']==seed]
        if len({row['initial_state_sha256'] for row in block})!=1: raise ValueError('unfair initial states')
        for mode in ['regions','junction']:
            if len({row['batch_sha256'] for row in block if row['sampling']==mode})!=1: raise ValueError('unpaired batch sequences')
    if updates!=config['optimizer_updates']: raise ValueError('update budget mismatch')
    write_json(output/'model_lock.json',{'selection':'fixed_final_step','optimizer_updates':updates,'models':models})
    write_json(folder/'paired_state_proof.json',proof)
    return {name:load_student(output/item['path']) for name,item in models.items()}


def diagnose(config,output,metrics,predictions):
    folder=output/'05_diagnosis';folder.mkdir()
    masks=pd.read_parquet(output/'01_inputs/test_regions.parquet').set_index('target_id')
    rows=[];repro=[];candidate={}
    for seed in config['seeds']:
        control=predictions[f'control_{seed}']['on_grid'].set_index('target_id')
        old=pd.read_parquet(Path(config['upstream_root'])/f'04_evaluation/anchored_{seed}_on_grid.parquet').set_index('target_id').loc[control.index]
        repro.append({'seed':seed,'max_beta_difference_rad':float(np.abs(control[['pred_'+c for c in BETA_COLUMNS]].to_numpy()-old[['pred_'+c for c in BETA_COLUMNS]].to_numpy()).max()),
            'max_error_difference_mm':float(np.abs(control.error_mm-old.error_mm).max())})
        candidate[str(seed)]={}
        for arm in config['arms']:
            name=f'{arm}_{seed}';p=predictions[name]['on_grid'].set_index('target_id')
            test=p.loc[masks.index];base=control.loc[masks.index]
            small=metrics[(metrics.arm==name)&metrics.case.isin(SMALL)]
            retained=bool(len(small)==8 and small.within_3mm.eq(1).all())
            oldbad=base.error_mm>50;newbad=test.error_mm>50
            candidate[str(seed)][arm]=bool(oldbad.sum()>0 and newbad.sum()<=oldbad.sum()/2 and retained)
            point=test[['error_mm']].copy();point['control_error_mm']=base.error_mm;point['delta_mm']=point.error_mm-point.control_error_mm
            point.join(masks).reset_index().to_parquet(folder/f'{name}_test_delta.parquet',index=False)
            subsets={'all':np.ones(len(test),bool),'mixed_k32':masks.mixed_k32.to_numpy(),'single_domain':~masks.mixed_k32.to_numpy(),
                     **{f'junction_{width}mm':masks[f'junction_{width}mm'].to_numpy() for width in [5,10,20]}}
            for subset,mask in subsets.items():
                g=test.iloc[np.flatnonzero(mask)];b=base.iloc[np.flatnonzero(mask)]
                rows.append({'arm':arm,'seed':seed,'subset':subset,**error_metrics(g.error_mm),'gt50':int((g.error_mm>50).sum()),'gt100':int((g.error_mm>100).sum()),
                    'fixed_control_gt50':int(((b.error_mm>50)&(g.error_mm<=50)).sum()),'new_gt50':int(((b.error_mm<=50)&(g.error_mm>50)).sum()),
                    'regression_gt10':int((g.error_mm-b.error_mm>10).sum()),'small_paths_retained':retained})
    pd.DataFrame(rows).to_csv(folder/'paired_metrics.csv',index=False)
    write_json(folder/'control_reproduction.json',repro)
    return {'candidate_by_seed':candidate,'candidate_both_seeds':{arm:all(candidate[str(seed)][arm] for seed in config['seeds']) for arm in config['arms']},'control_reproduction':repro}


def main(args):
    path=Path(args.config).resolve();config=yaml.safe_load(path.read_text());ident=identity(path,args.binding)
    output=Path(args.output_root).resolve();output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=start+config['deadline_seconds'];write_json(output/'run_identity.json',ident)
    def progress(phase,completed=0,total=0):
        if time.monotonic()>deadline: raise TimeoutError('fixed task budget exhausted')
        write_json(output/'progress.json',{'status':'running','phase':phase,'completed':completed,'total':total,'elapsed_seconds':time.monotonic()-start,'output_root':str(output)})
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_op_determinism()
        env=environment(config);progress('freeze');left,right,targets=freeze(config,output,ident)
        models=training(config,output,left,right,env,deadline,progress)
        metrics,predictions=evaluate(output,right,targets,models,env,progress)
        result=diagnose(config,output,metrics,predictions)
        summary={**ident,**result,'status':'completed','elapsed_seconds':time.monotonic()-start,'training_updates':config['optimizer_updates'],
                 'model_count':len(models),'on_grid_test_count':8272,'off_grid_test_count':3000,'trajectory_count':35,'new_supervision':0,'global_replacement_authorized':False}
        write_json(output/'summary.json',summary)
        write_json(output/'progress.json',{'status':'completed','phase':'terminal','completed':config['optimizer_updates'],'total':config['optimizer_updates'],'output_root':str(output)})
        write_json(output/'completion_manifest.json',{'status':'completed',**ident,'artifacts':[{'path':str(p.relative_to(output)),'sha256':digest(p)} for p in sorted(output.rglob('*')) if p.is_file()]})
        print(json.dumps(summary),flush=True)
    except Exception as exc:
        write_json(output/'failure.json',{'status':'failed','exception':type(exc).__name__,'message':str(exc),'elapsed_seconds':time.monotonic()-start})
        write_json(output/'progress.json',{'status':'failed','phase':'terminal','message':str(exc),'output_root':str(output)})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--binding',required=True);parser.add_argument('--output-root',required=True)
    main(parser.parse_args())
