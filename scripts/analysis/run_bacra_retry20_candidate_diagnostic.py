#!/usr/bin/env python3
"""Execute the explicitly authorized failed-geometry candidate diagnostic."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis')]
from run_bacra_retry20_preflight import digest, write_json, verify_upstream
from run_trajectory_canonical_teacher_v10 import load_environment
from quasi_exp.teacher.optimized_forward import optimized_forward
from quasi_exp.teacher.retry12_symmetry import XYZ_COLUMNS, BETA_COLUMNS
from quasi_exp.teacher.retry20_diagnostic import solve_anchored, paired_datasets, prediction_metrics, train_pair, bounded_dls
from quasi_exp.teacher.retry20_relocation import coverage_audit
from quasi_exp.teacher.retry19_fullspace import target_knn_edges
EXPERIMENT = 'bacra_retry20_candidate_teacher_student_diagnostic'


def environment(config):
    return optimized_forward(load_environment(Path(config['project_root']), ROOT/config['sources']['robot_config']))


def verify_manifest(root, manifest):
    entries = json.loads(manifest.read_text())['artifacts']
    for entry in entries:
        if digest(root/entry['path']) != entry['sha256']:
            raise ValueError(f'changed artifact: {root/entry["path"]}')


def identity(config_path, binding):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    if git('rev-parse', 'HEAD') != binding or git('status','--porcelain','--untracked-files=no'):
        raise ValueError('requires clean exact binding checkout')
    entry = yaml.safe_load(git('show',binding+':spec/registry.yaml'))['experiments'][EXPERIMENT]
    source = entry['scientific_source_fixed_point']
    subprocess.run(['git','merge-base','--is-ancestor',source,binding],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--exit-code',source,binding,'--','src','scripts/analysis','scripts/pipelines','configs','tests',*entry['protocol_sources']],cwd=ROOT,check=True)
    if str(config_path.relative_to(ROOT)) != entry['config']:
        raise ValueError('unbound config')
    return dict(experiment_id=EXPERIMENT, scientific_source_fixed_point=source, binding_fixed_point=binding,
        config_sha256=digest(config_path), diagnostic_only=True, formal_authorized=False,
        geometry_failure_override='explicit user request 2026-09-10; separate diagnostic protocol')


def worker(args, config):
    stage = Path(args.output_root)
    targets = pd.read_parquet(stage/'targets.parquet').iloc[args.start:args.stop]
    bank = pd.read_parquet(stage/'seed_bank.parquet')
    xyz = targets[list(XYZ_COLUMNS)].to_numpy()
    anchors = targets[['anchor_'+c for c in BETA_COLUMNS]].to_numpy()
    env = environment(config)
    beta, audit = solve_anchored(env, xyz, anchors, bank[list(XYZ_COLUMNS)].to_numpy(), bank[list(BETA_COLUMNS)].to_numpy(),
        fallback_seeds=config['teacher']['fallback_seeds'], deadline=time.monotonic()+max(0., args.deadline-time.time()))
    finite = np.isfinite(beta).all(axis=1)
    residual = np.full(len(beta), np.nan)
    if finite.any():
        residual[finite] = np.linalg.norm(env.reference.fk(beta[finite])-xyz[finite],axis=1)*1000
    accepted = finite & (residual<=3) & ((beta>=env.bounds[:,0]-1e-10)&(beta<=env.bounds[:,1]+1e-10)).all(axis=1)
    audit['accepted'] = accepted
    audit['scalar_fk_residual_mm'] = residual
    frame = pd.concat([targets.reset_index(drop=True), audit],axis=1)
    for i,c in enumerate(BETA_COLUMNS): frame[c]=beta[:,i]
    frame.to_parquet(stage/f'batch_{args.start:06d}.parquet',index=False)


def solve_stage(config_path, config, stage, targets, bank, deadline, progress):
    stage.mkdir(parents=True)
    targets.to_parquet(stage/'targets.parquet',index=False)
    bank.to_parquet(stage/'seed_bank.parquet',index=False)
    pending = list(range(0,len(targets),config['teacher']['batch_size']))
    active=[]
    completed=0
    started=time.time()
    while pending or active:
        while pending and len(active)<config['runtime']['workers'] and time.time()<deadline:
            start=pending.pop(0)
            stop=min(len(targets),start+config['teacher']['batch_size'])
            log=(stage/f'batch_{start:06d}.log').open('w')
            proc=subprocess.Popen([sys.executable,__file__,'--config',str(config_path),'--output-root',str(stage),
                '--worker','--start',str(start),'--stop',str(stop),'--deadline',str(deadline)],stdout=log,stderr=subprocess.STDOUT)
            active.append((proc,log,start,stop))
        for item in active[:]:
            proc,log,start,stop=item
            code=proc.poll()
            if code is not None:
                log.close(); active.remove(item)
                if code:
                    # Drain running children before returning an operational error.
                    for child, stream, _, _ in active: child.wait(); stream.close()
                    raise RuntimeError(f'Teacher batch {start} failed: {stage}/batch_{start:06d}.log')
                completed+=stop-start
                progress(completed,len(targets))
                elapsed=time.time()-started
                write_json(stage/'timing.json',dict(completed=completed,total=len(targets),elapsed_seconds=elapsed,
                    projected_total_seconds=elapsed*len(targets)/completed*1.25,projection_margin=1.25))
        if pending and not active and time.time()>=deadline:
            break
        if active: time.sleep(.5)
    parts=[pd.read_parquet(p) for p in sorted(stage.glob('batch_*.parquet'))]
    actual=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['target_id'])
    result=targets.merge(actual.drop(columns=[c for c in targets if c!='target_id']),on='target_id',how='left',validate='one_to_one')
    if 'accepted' not in result: result['accepted']=False
    result['accepted']=result['accepted'].fillna(False).astype(bool)
    for c in BETA_COLUMNS:
        if c not in result: result[c]=np.nan
    result.to_parquet(stage/'teacher.parquet',index=False)
    return result


def freeze_inputs(config, output, ident):
    old=Path(config['upstream_root']); pre=Path(config['preflight_root'])
    preconfig=yaml.safe_load((ROOT/'configs/bacra_retry20_preflight.yaml').read_text())
    verify_upstream(preconfig)
    if digest(pre/'completion_manifest.json')!=config['preflight_manifest_sha256']:
        raise ValueError('changed preflight manifest')
    verify_manifest(pre,pre/'completion_manifest.json')
    if digest(ROOT/config['sources']['robot_config']) != '21794d9beb6b841dea1c42e57a44ca91439cfe76be4b04930a66fd57fbbb59bc':
        raise ValueError('robot config changed')
    geom=json.loads((pre/'summary.json').read_text())
    if not config['claims']['failed_geometry_diagnostic_authorized']:
        raise ValueError('diagnostic override not authorized')
    original=pd.read_parquet(old/'06_unified_dataset/unified_signed_supervision.parquet')
    planned=pd.read_parquet(pre/'00_objective_feasibility/planned_replacement_registry.parquet')
    prospective=pd.read_parquet(pre/'00_objective_feasibility/prospective_primary_targets.parquet')
    panels=pd.read_parquet(pre/'00_objective_feasibility/evaluation_panel_registry.parquet')
    paired_datasets(original,prospective,planned,np.zeros((len(planned),6)),np.ones(len(planned),bool))
    trajectories=pd.read_parquet(old/'09_postlock_trajectories/postlock_15_trajectory_waypoints.parquet')
    # An explicit fixed-radius benchmark, frozen before solving/training; support is audited, never used to shrink it.
    phase=np.linspace(0,2*np.pi,121)
    zero=float(config['zero_x_m'])
    circle=pd.DataFrame({'trajectory_id':'fixed_circle_r40_u100','waypoint_index':np.arange(121),
        'shape_class':'circle','trajectory_role':'fixed_40mm_diagnostic','x_m':zero-.1,
        'y_m':.04*np.cos(phase),'z_m':.04*np.sin(phase)})
    trajectories=pd.concat([trajectories,circle],ignore_index=True)
    trajectories['target_id']=[f'track:{t}:{i}' for t,i in zip(trajectories.trajectory_id,trajectories.waypoint_index)]
    stage=output/'00_objective_feasibility';stage.mkdir()
    trajectories.to_parquet(stage/'trajectory_registry.parquet',index=False)
    planned.to_parquet(stage/'replacement_registry.parquet',index=False)
    panels.to_parquet(stage/'evaluation_panel_registry.parquet',index=False)
    common={'schema_version':1,'experiment_id':EXPERIMENT}
    path_registry=trajectories.groupby('trajectory_id',as_index=False).agg(waypoint_count=('target_id','size'))
    path_registry.to_parquet(stage/'path_registry.parquet',index=False)
    unchanged=original[~original.target_id.isin(planned.target_id)][['target_id']]
    unchanged.to_parquet(stage/'credit_registry.parquet',index=False)
    def reg(name,id_column,count):
        p=stage/name
        return dict(path=str(p.relative_to(output)),sha256=digest(p),id_column=id_column,row_count=int(count))
    rreg=reg('replacement_registry.parquet','target_id',len(planned))
    treg=reg('path_registry.parquet','trajectory_id',len(path_registry))
    wreg=reg('trajectory_registry.parquet','target_id',len(trajectories))
    write_json(stage/'objective_contract.json',{**common,'scope':'coverage_trajectory',
        'scientific_source_sha':ident['scientific_source_fixed_point'],'config_sha256':ident['config_sha256'],
        'diagnostic_pilot_allowed':True,'primary_objectives':[
            dict(id='new_labels',kind='dataset_size',metric='accepted_new_labels',required_for_claim=True,
                denominator_id='replacements',target=dict(operator='>=',value=len(planned),unit='count')),
            dict(id='tracking',kind='complete_trajectory',metric='complete_reference_and_raw_tracking',required_for_claim=True,
                denominator_id='paths',target=dict(operator='>=',value=len(path_registry),unit='count'))],
        'description':'No full-path claim without complete Teacher and raw evaluation; current run remains diagnostic'})
    write_json(stage/'denominator_size.json',{**common,'frozen_before_launch':True,
        'replacement_count':len(planned),'primary_count':len(prospective),'unified_count':len(original),
        'trajectory_count':len(path_registry),'trajectory_waypoints':len(trajectories),
        'off_grid_test_count':int(panels.panel.eq('geometry_test').sum()),
        'frozen_required_cells':88168,'frozen_required_volume_mm3':87104000,
        'denominators':[dict(id='replacements',kind='dataset_size',unit='target',required_count=len(planned),registry=rreg),
            dict(id='paths',kind='complete_trajectory',unit='path',required_count=len(path_registry),registry=treg)],
        'resource_denominators':dict(required_supervision_vertex_count=len(planned)+len(trajectories),
            required_logical_edge_count=sum(len(p)-1 for _,p in trajectories.groupby('trajectory_id')),
            required_second_parent_certification_count=0)})
    write_json(stage/'reusable_evidence.json',{**common,'unchanged_supervision_vertices':len(unchanged),
        'new_required_labels':len(planned),'old_teacher_seeds_allowed':True,'proposal_beta_used_as_label_or_hint':False,
        'source_artifacts':[dict(path=str(old/'06_unified_dataset/unified_signed_supervision.parquet'),
            sha256=digest(old/'06_unified_dataset/unified_signed_supervision.parquet'))],
        'credit_registry':reg('credit_registry.parquet','target_id',len(unchanged)),
        'eligible_counts':dict(supervision_vertices=len(unchanged),connector_only_vertices=0,served_coverage_units=0,
            complete_trajectories=0,verified_edges=0,second_parent_certifications=0),
        'ineligible_counts':dict(proposal_only=len(planned),branch_conflicts=0,unused=0),
        'upstream_identity':json.loads((old/'run_identity.json').read_text()),'preflight_identity':json.loads((pre/'run_identity.json').read_text())})
    write_json(stage/'budget_lower_bound.json',{**common,'registered_budget_seconds':43200,'solver_budget_seconds':36000,
        'objective_lower_bounds':[
            dict(objective_id='new_labels',method='count_credit',required_units=len(planned),target_units=len(planned),
                reusable_eligible_units=0,maximum_credit_per_new_supervision_vertex=1,
                minimum_resources=dict(new_supervision_vertices=len(planned)),basis_registry=rreg),
            dict(objective_id='tracking',method='not_bounded',minimum_resources={},
                reason='No bound for complete legal continuous Teacher/reference paths; scalar IK success is insufficient')],
        'resources':{'new_supervision_vertices':dict(optimistic_minimum=len(planned),registered_budget=len(planned)+len(trajectories),basis_registry=rreg)},
        'all_required_objectives_bounded':False,'all_required_resources_feasible':False})
    write_json(stage/'atomic_objective_schedule.json',{**common,'row_count_is_stop_condition':False,
        'scheduled_objective_ids':['new_labels','tracking'],
        'entries':[dict(id='all_replacements',objective_id='new_labels',kind='dataset',requirement_registry=rreg)]+
            [dict(id=str(t),objective_id='tracking',kind='trajectory',waypoints=len(p),
                full_cycle_required=not p.shape_class.eq('zero_to_boundary').all(),held_out=True,
                minimum_diameter_mm=80 if t=='fixed_circle_r40_u100' else float(np.max(np.ptp(p[list(XYZ_COLUMNS)].to_numpy(),axis=0))*1000),
                requirement_registry=wreg) for t,p in trajectories.groupby('trajectory_id')]})
    write_json(stage/'gate.json',{**common,'inputs_valid':True,'status':'diagnostic_only',
        'required_objectives_budget_feasible':False,'claim_bearing_run_authorized':False,
        'diagnostic_pilot_authorized':True,'original_geometry_gate':geom['geometry_gate'],
        'failed_objective_ids':['tracking'],'reason_codes':['COMPLETE_REFERENCE_PATH_NOT_BOUNDED','USER_AUTHORIZED_LAYOUT_DIAGNOSTIC']})
    sys.path.insert(0,str(ROOT/'scripts/spec'))
    from validate_objective_feasibility import validate_contract_dir
    errors=validate_contract_dir(stage)
    if errors: raise ValueError(errors)
    write_json(output/'input_manifest.json',{'inputs':[{'path':str(p),'sha256':digest(p)} for p in [
        old/'06_unified_dataset/unified_signed_supervision.parquet',pre/'completion_manifest.json',
        pre/'00_objective_feasibility/planned_replacement_registry.parquet',pre/'00_objective_feasibility/prospective_primary_targets.parquet',
        pre/'00_objective_feasibility/evaluation_panel_registry.parquet',old/'09_postlock_trajectories/postlock_15_trajectory_waypoints.parquet',
        ROOT/config['sources']['robot_config']]]})
    return original,planned,prospective,panels,trajectories


def main(args):
    config_path=Path(args.config).resolve();config=yaml.safe_load(config_path.read_text())
    if args.worker: return worker(args,config)
    ident=identity(config_path,args.binding_sha)
    output=Path(args.output_root).resolve();output.mkdir(parents=True,exist_ok=False)
    write_json(output/'run_identity.json',ident)
    started=time.time()
    def progress(phase,done=0,total=0):
        write_json(output/'progress.json',dict(status='running',phase=phase,completed=done,total=total,elapsed_seconds=time.time()-started))
    try:
        progress('verify_inputs')
        original,planned,prospective,panels,trajectories=freeze_inputs(config,output,ident)
        env=environment(config)
        bank=original[['target_id',*XYZ_COLUMNS,*BETA_COLUMNS]].copy()
        anchors=original.set_index('target_id').loc[planned.target_id,list(BETA_COLUMNS)].to_numpy()
        targets=planned.copy()
        for i,c in enumerate(BETA_COLUMNS):targets['anchor_'+c]=anchors[:,i]
        progress('relocation_teacher',0,len(targets))
        teacher=solve_stage(config_path,config,output/'01_relocation_teacher',targets,bank,
            started+config['runtime']['solver_hours']*3600,lambda d,t:progress('relocation_teacher',d,t))
        accepted=teacher.accepted.to_numpy()
        left,right,keep=paired_datasets(original,prospective,planned,teacher[list(BETA_COLUMNS)].to_numpy(),accepted)
        stage=output/'02_dataset';stage.mkdir()
        left.to_parquet(stage/'L0_supervision.parquet',index=False);right.to_parquet(stage/'L1_supervision.parquet',index=False)
        pd.DataFrame({'target_id':original.target_id,'included_in_pair':keep}).to_parquet(stage/'paired_row_mask.parquet',index=False)
        primary_ids=set(prospective.target_id)
        primary=right[right.target_id.isin(primary_ids)].copy()
        cells=pd.read_parquet(Path(config['preflight_root'])/'00_objective_feasibility/required_cell_registry.parquet')
        audit,served,slices=coverage_audit(cells,primary[list(XYZ_COLUMNS)].to_numpy(),zero_x_m=config['zero_x_m'])
        write_json(stage/'coverage_audit.json',audit)
        served.to_parquet(stage/'served_cells.parquet',index=False);slices.to_parquet(stage/'axial_slices.parquet',index=False)
        edges=target_knn_edges(primary,k=16)
        beta_by_id=primary.set_index('target_id')[list(BETA_COLUMNS)]
        gaps=beta_by_id.loc[edges.left_target_id].to_numpy()-beta_by_id.loc[edges.right_target_id].to_numpy()
        edges['weighted_gap_deg']=np.rad2deg(np.sqrt(np.mean((gaps*np.array([4,4,2,2,1,1]))**2,axis=1)))
        edges['raw_gap_deg']=np.rad2deg(np.max(np.abs(gaps),axis=1))
        edges.to_parquet(stage/'teacher_edge_audit.parquet',index=False)
        write_json(stage/'dataset_summary.json',dict(planned_replacements=len(planned),accepted_replacements=int(accepted.sum()),
            failed_replacements=int((~accepted).sum()),paired_rows=len(right),primary_rows=len(primary),
            teacher_method='two-seed anchored continuation plus bounded least-squares; no global graph relabeling',
            root_connectivity='not_certified_for_relocated_rows',edge_weighted_p95_deg=float(np.percentile(edges.weighted_gap_deg,95)),
            edge_raw_gt7_rate=float(np.mean(edges.raw_gap_deg>7))))
        if not accepted.any(): raise RuntimeError('no valid relocated Teacher labels')
        progress('student_training')
        stage=output/'03_student';stage.mkdir()
        pv=panels[panels.panel.ne('geometry_test')][list(XYZ_COLUMNS)].to_numpy()
        models=train_pair(left,right,env,pv,stage,write_json,steps=config['student']['steps'])
        write_json(stage/'model_lock.json',{'selection':'fixed_final_step','models':{n:digest(stage/(n+'.keras')) for n in models},
            'datasets':{n:digest(output/'02_dataset'/(n+'_supervision.parquet')) for n in models}})
        progress('trajectory_teacher',0,len(trajectories))
        _,near=cKDTree(right[list(XYZ_COLUMNS)]).query(trajectories[list(XYZ_COLUMNS)])
        anchors=right.iloc[near][list(BETA_COLUMNS)].to_numpy()
        for i,c in enumerate(BETA_COLUMNS):trajectories['anchor_'+c]=anchors[:,i]
        tracked=solve_stage(config_path,config,output/'04_trajectory_teacher',trajectories,
            right[['target_id',*XYZ_COLUMNS,*BETA_COLUMNS]],started+11*3600,
            lambda d,t:progress('trajectory_teacher',d,t))
        tracked.loc[~tracked.accepted,list(BETA_COLUMNS)]=np.nan
        progress('evaluation')
        stage=output/'05_evaluation';stage.mkdir()
        records=[];point_records=[]
        cases=[(str(t),p.sort_values('waypoint_index')) for t,p in tracked.groupby('trajectory_id')]
        test=panels[panels.panel.eq('geometry_test')].copy();test['waypoint_index']=np.arange(len(test))
        cases.append(('off_grid_test',test))
        for case,part in cases:
            xyz=part[list(XYZ_COLUMNS)].to_numpy()
            frame=part.copy()
            for layout,data in [('L0',left),('L1',right)]:
                distance,near=cKDTree(data[list(XYZ_COLUMNS)]).query(xyz)
                frame[layout+'_nearest_support_mm']=distance*1000
            for name in ['Teacher','L0_raw','L0_DLS2','L1_raw','L1_DLS2']:
                if name=='Teacher':
                    if case=='off_grid_test':continue
                    beta=part[list(BETA_COLUMNS)].to_numpy()
                else:
                    model=models[name[:2]]
                    beta=np.asarray(model(xyz.astype(np.float32),training=False),float)
                    if name.endswith('DLS2'): beta=bounded_dls(env,xyz,beta,steps=2)
                metric,actual,error=prediction_metrics(env.reference,xyz,beta)
                metric.update(case=case,output=name)
                if case!='off_grid_test':
                    closed=not part.shape_class.isin(['zero_to_boundary','spiral_3d']).all()
                    sequence=np.vstack([beta,beta[:1]]) if closed else beta
                    delta=np.diff(sequence,axis=0)
                    metric['closed_path']=closed
                    metric['missing_edge_count']=int((~np.isfinite(delta).all(axis=1)).sum())
                    metric['raw_step_gt7_rate']=float(np.mean(np.rad2deg(np.max(np.abs(delta),axis=1))>7))
                    metric['max_beta_step_deg']=float(np.nanmax(np.rad2deg(np.max(np.abs(delta),axis=1)))) if np.isfinite(delta).any() else None
                records.append(metric)
                frame[name+'_error_mm']=error
                for i,c in enumerate(XYZ_COLUMNS):frame[name+'_'+c]=actual[:,i]
                for i,c in enumerate(BETA_COLUMNS):frame[name+'_'+c]=beta[:,i]
            frame['case']=case;point_records.append(frame)
        pd.DataFrame(records).to_parquet(stage/'metrics.parquet',index=False)
        pd.concat(point_records,ignore_index=True).to_parquet(stage/'waypoint_evaluation.parquet',index=False)
        write_json(output/'summary.json',{**ident,'operational_status':'complete','geometry_gate_passed':False,
            'teacher_accepted_replacements':int(accepted.sum()),'teacher_attempted_replacements':len(planned),
            'student_status':'trained','trajectory_count':len(cases)-1,'off_grid_test_count':len(test),
            'elapsed_seconds':time.time()-started,'formal_authorized':False,
            'claim_boundary':'anchored diagnostic Teacher, paired fixed-step Student, same-model pointwise correction; no canonical-field or closed-loop claim'})
        write_json(output/'completion_manifest.json',{'artifacts':[{'path':str(p.relative_to(output)),'sha256':digest(p)} for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ['progress.json','training_progress.json'] and p.suffix!='.log']})
        write_json(output/'progress.json',dict(status='complete',phase='complete',completed=len(cases),total=len(cases)))
    except Exception as error:
        write_json(output/'failure.json',dict(status='operational_failure',error_type=type(error).__name__,error=str(error)))
        write_json(output/'progress.json',dict(status='failed',phase='operational_failure',message=str(error)))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--binding-sha');p.add_argument('--worker',action='store_true');p.add_argument('--start',type=int);p.add_argument('--stop',type=int);p.add_argument('--deadline',type=float)
    main(p.parse_args())
