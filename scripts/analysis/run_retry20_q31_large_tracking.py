#!/usr/bin/env python3
"""Evaluate fixed Q31 large paths with the sealed retry20 model pair; no training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis'), str(ROOT/'scripts/spec')]
from run_bacra_retry20_candidate_diagnostic import environment, solve_stage, verify_manifest
from run_bacra_retry20_preflight import digest, write_json
from quasi_exp.teacher.retry12_symmetry import XYZ_COLUMNS, BETA_COLUMNS
from quasi_exp.teacher.retry19_direct_student import load_direct_student
from quasi_exp.teacher.retry20_diagnostic import bounded_dls, prediction_metrics
from validate_objective_feasibility import validate_contract_dir

EXPERIMENT = 'retry20_q31_large_tracking_diagnostic'


def load_suite(path):
    suite = json.loads(Path(path).read_text())
    expected = {'large_star': (407, True), 'large_helix': (523, False), 'large_tapered_helix': (574, False)}
    if suite['frame'] != 'robot_base' or suite['coordinate_unit'] != 'm':
        raise ValueError('wrong coordinate frame or unit')
    if {t['id'] for t in suite['tracks']} != set(expected) or len(suite['tracks']) != 3:
        raise ValueError('wrong fixed trajectory inventory')
    frames = []
    for t in suite['tracks']:
        count, closed = expected[t['id']]
        xyz = np.asarray(t['waypoints_m'], float)
        if xyz.shape != (count, 3) or not np.isfinite(xyz).all() or t['closed'] is not closed:
            raise ValueError('changed trajectory count, coordinates or topology')
        if not np.allclose(np.ptp(xyz, axis=0)*1000, t['geometry']['span_mm'], atol=1e-9, rtol=0):
            raise ValueError('changed trajectory scale')
        frame = pd.DataFrame(xyz, columns=XYZ_COLUMNS)
        frame['trajectory_id'] = t['id']; frame['closed'] = closed
        frame['shape_class'] = 'star' if closed else 'spiral_3d'
        frame['trajectory_role'] = 'q31_large_fixed_independent_diagnostic'
        frame['waypoint_index'] = np.arange(count)
        frame['target_id'] = [f"q31:{t['id']}:{i}" for i in range(count)]
        frames.append(frame)
    return suite, pd.concat(frames, ignore_index=True)


def locked_inputs(run):
    lock = json.loads((run/'03_student/model_lock.json').read_text())
    if lock['selection'] != 'fixed_final_step':
        raise ValueError('wrong model selection')
    for name in ['L0', 'L1']:
        for path, expected in [(run/f'03_student/{name}.keras', lock['models'][name]),
                               (run/f'02_dataset/{name}_supervision.parquet', lock['datasets'][name])]:
            if digest(path) != expected:
                raise ValueError(f'locked input changed: {path}')
    return lock


def step_metrics(beta, closed):
    sequence = np.vstack([beta, beta[:1]]) if closed else beta
    delta = np.diff(sequence, axis=0)
    valid = np.isfinite(delta).all(axis=1)
    steps = np.rad2deg(np.max(np.abs(delta), axis=1))
    return dict(closed_path=bool(closed), edge_count=len(delta), missing_edge_count=int((~valid).sum()),
                raw_step_gt7_rate=float(np.mean(valid & (steps > 7))),
                max_beta_step_deg=float(np.max(steps[valid])) if valid.any() else None,
                step_rate_denominator='all registered edges; missing edges reported separately')


def preflight(output, cfg, ident, targets, inputs):
    stage = output/'00_objective_feasibility'; stage.mkdir()
    common = dict(schema_version=1, experiment_id=EXPERIMENT)
    targets.to_parquet(stage/'trajectory_registry.parquet', index=False)
    paths = targets.groupby('trajectory_id', sort=False, as_index=False).agg(waypoint_count=('target_id','size'), closed=('closed','first'))
    paths.to_parquet(stage/'path_registry.parquet', index=False)
    pd.DataFrame({'target_id': pd.Series(dtype=str)}).to_parquet(stage/'credit_registry.parquet', index=False)
    def reg(name, key, count):
        p = stage/name
        return dict(path=str(p.relative_to(output)), sha256=digest(p), id_column=key, row_count=count)
    tr = reg('trajectory_registry.parquet','target_id',len(targets)); pr = reg('path_registry.parquet','trajectory_id',len(paths))
    write_json(stage/'objective_contract.json', {**common, 'scope':'coverage_trajectory',
        'scientific_source_sha':ident['scientific_source_fixed_point'], 'config_sha256':ident['config_sha256'],
        'diagnostic_pilot_allowed':True, 'primary_objectives':[dict(id='tracking', kind='complete_trajectory',
        metric='complete_teacher_and_raw_tracking',required_for_claim=True,denominator_id='paths',target=dict(operator='>=',value=3,unit='count'))]})
    write_json(stage/'denominator_size.json', {**common,'frozen_before_launch':True,
        'denominators':[dict(id='paths',kind='complete_trajectory',unit='path',required_count=3,registry=pr)],
        'trajectory_waypoints':len(targets), 'prediction_rows':len(targets)*5,
        'resource_denominators':dict(required_supervision_vertex_count=len(targets),
        required_logical_edge_count=sum(len(p)-1+int(p.closed.iloc[0]) for _,p in targets.groupby('trajectory_id')),required_second_parent_certification_count=0)})
    write_json(stage/'reusable_evidence.json', {**common,'source_artifacts':inputs,
        'credit_registry':reg('credit_registry.parquet','target_id',0), 'proposal_beta_used_as_label_or_hint':False,
        'eligible_counts':dict(supervision_vertices=0,connector_only_vertices=0,served_coverage_units=0,complete_trajectories=0,verified_edges=0,second_parent_certifications=0),
        'ineligible_counts':dict(proposal_only=0,branch_conflicts=0,unused=0),
        'reuse_scope':'locked models, normalization, datasets as seed/support banks and exact Q31 target geometry; no prior tracking result credit'})
    write_json(stage/'budget_lower_bound.json', {**common,'registered_budget_seconds':cfg['deadline_seconds'],
        'solver_budget_seconds':cfg['teacher_seconds'], 'maximum_fallback_seed_solves':len(targets)*16,
        'objective_lower_bounds':[dict(objective_id='tracking',method='not_bounded',minimum_resources={},reason='No guaranteed complete-path Teacher or raw success bound')],
        'resources':{'new_supervision_vertices':dict(optimistic_minimum=0,registered_budget=len(targets),basis_registry=tr)},
        'all_required_objectives_bounded':False,'all_required_resources_feasible':False})
    write_json(stage/'atomic_objective_schedule.json', {**common,'row_count_is_stop_condition':False,'scheduled_objective_ids':['tracking'],
        'entries':[dict(id=str(t),objective_id='tracking',kind='trajectory',priority=i,required_for_claim=True,held_out=True,
        waypoints=len(p),full_cycle_required=bool(p.closed.iloc[0]),minimum_diameter_mm=float(np.max(np.ptp(p[list(XYZ_COLUMNS)].to_numpy(),axis=0))*1000),
        requirement_registry=tr,reserved_resources=dict(new_supervision_vertices=len(p),edge_certificates=0)) for i,(t,p) in enumerate(targets.groupby('trajectory_id',sort=False))]})
    write_json(stage/'gate.json', {**common,'inputs_valid':True,'status':'diagnostic_only','required_objectives_budget_feasible':False,
        'claim_bearing_run_authorized':False,'diagnostic_pilot_authorized':True,'failed_objective_ids':['tracking'],
        'reason_codes':['COMPLETE_TRACKING_SUCCESS_NOT_BOUNDED','USER_AUTHORIZED_2026_09_12']})
    errors = validate_contract_dir(stage)
    if errors: raise ValueError(errors)


def run(config_path, output, binding, prepare_only=False):
    cfg = yaml.safe_load(config_path.read_text())
    git = lambda *a: subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
    if git('rev-parse','HEAD') != binding or git('status','--porcelain','--untracked-files=no'):
        raise ValueError('requires clean exact diagnostic binding')
    entry = yaml.safe_load(git('show',binding+':spec/registry.yaml'))['experiments'][EXPERIMENT]
    upstream = Path(cfg['upstream_root'])
    if output.exists(): raise FileExistsError(output)
    verify_manifest(upstream, upstream/'completion_manifest.json')
    lock = locked_inputs(upstream)
    old_identity = json.loads((upstream/'run_identity.json').read_text())
    if old_identity['binding_fixed_point'] != cfg['upstream_binding']: raise ValueError('upstream identity changed')
    # Existing numerical code/config is byte-identical to the model's original binding.
    for directory in ['src', 'configs/robot_rods_only_standard_100k.yaml', 'scripts/analysis/run_bacra_retry20_candidate_diagnostic.py', 'scripts/analysis/run_trajectory_canonical_teacher_v10.py']:
        subprocess.run(['git','diff','--exit-code',cfg['upstream_binding'],binding,'--',directory],cwd=ROOT,check=True)
    suite_path = ROOT/cfg['suite']
    if digest(suite_path) != cfg['suite_sha256']: raise ValueError('fixed suite hash changed')
    suite, targets = load_suite(suite_path)
    old_cfg_path = ROOT/'configs/bacra_retry20_candidate_diagnostic.yaml'
    old_cfg = yaml.safe_load(old_cfg_path.read_text())
    data = {name:pd.read_parquet(upstream/f'02_dataset/{name}_supervision.parquet') for name in ['L0','L1']}
    env = environment(old_cfg)
    models = {name:load_direct_student(upstream/f'03_student/{name}.keras') for name in ['L0','L1']}
    replay = pd.read_parquet(upstream/'05_evaluation/waypoint_evaluation.parquet').iloc[:8]
    replay_check = {}
    for name,model in models.items():
        pred = np.asarray(model(replay[list(XYZ_COLUMNS)].to_numpy(np.float32),training=False),float)
        stored = replay[[name+'_raw_'+c for c in BETA_COLUMNS]].to_numpy()
        difference = float(np.max(np.abs(pred-stored)))
        if difference > 1e-6: raise ValueError(f'locked model replay differs: {name} {difference}')
        replay_check[name] = difference
    inputs = [dict(path=str(p),sha256=digest(p)) for p in [config_path,suite_path,old_cfg_path,
        upstream/'completion_manifest.json',upstream/'03_student/model_lock.json',upstream/'03_student/training_contract.json',
        *[upstream/f'03_student/{n}.keras' for n in ['L0','L1']], *[upstream/f'02_dataset/{n}_supervision.parquet' for n in ['L0','L1']]]]
    ident = dict(experiment_id=EXPERIMENT,scientific_source_fixed_point=entry['scientific_source_fixed_point'],binding_fixed_point=binding,
        upstream_binding=cfg['upstream_binding'],config_sha256=digest(config_path),suite_sha256=digest(suite_path),model_lock=lock,
        locked_model_replay_max_abs_rad=replay_check,diagnostic_only=True,formal_authorized=False)
    output.mkdir(parents=True)
    write_json(output/'run_identity.json',ident);write_json(output/'input_manifest.json',{'inputs':inputs})
    preflight(output,cfg,ident,targets,inputs)
    if prepare_only:
        write_json(output/'preparation.json',dict(status='prepared_only',teacher_launched=False,model_replay=replay_check));return
    started=time.time()
    def progress(phase,completed=0,total=len(targets)):
        write_json(output/'progress.json',dict(status='running',phase=phase,completed=completed,total=total,elapsed_seconds=time.time()-started))
    try:
        bank=data['L1'];xyz=targets[list(XYZ_COLUMNS)].to_numpy()
        _,near=cKDTree(bank[list(XYZ_COLUMNS)]).query(xyz)
        for i,c in enumerate(BETA_COLUMNS):targets['anchor_'+c]=bank.iloc[near][c].to_numpy()
        effective=json.loads(json.dumps(old_cfg));effective['teacher']['batch_size']=cfg['teacher_batch_size'];effective['runtime']['workers']=cfg['workers']
        progress('trajectory_teacher')
        tracked=solve_stage(old_cfg_path,effective,output/'04_trajectory_teacher',targets,
            bank[['target_id',*XYZ_COLUMNS,*BETA_COLUMNS]],started+cfg['teacher_seconds'],lambda n,total:progress('trajectory_teacher',n,total))
        tracked.loc[~tracked.accepted,list(BETA_COLUMNS)]=np.nan
        records=[];frames=[];stage=output/'05_evaluation';stage.mkdir()
        for case,part in tracked.groupby('trajectory_id',sort=False):
            part=part.sort_values('waypoint_index').copy();xyz=part[list(XYZ_COLUMNS)].to_numpy();closed=bool(part.closed.iloc[0])
            progress('evaluation',len(frames),3)
            for name,df in data.items():part[name+'_nearest_support_mm']=cKDTree(df[list(XYZ_COLUMNS)]).query(xyz)[0]*1000
            for name in ['Teacher','L0_raw','L0_DLS2','L1_raw','L1_DLS2']:
                if name=='Teacher':beta=part[list(BETA_COLUMNS)].to_numpy()
                else:
                    beta=np.asarray(models[name[:2]](xyz.astype(np.float32),training=False),float)
                    if name.endswith('DLS2'):beta=bounded_dls(env,xyz,beta,steps=2)
                metric,actual,error=prediction_metrics(env.reference,xyz,beta)
                metric.update(case=case,output=name,**step_metrics(beta,closed));records.append(metric)
                part[name+'_error_mm']=error
                for i,c in enumerate(XYZ_COLUMNS):part[name+'_'+c]=actual[:,i]
                for i,c in enumerate(BETA_COLUMNS):part[name+'_'+c]=beta[:,i]
            part['case']=case;frames.append(part)
        pd.DataFrame(records).to_parquet(stage/'metrics.parquet',index=False)
        pd.concat(frames,ignore_index=True).to_parquet(stage/'waypoint_evaluation.parquet',index=False)
        locked_inputs(upstream)
        for item in inputs:
            if digest(Path(item['path'])) != item['sha256']: raise ValueError('input mutated during evaluation')
        write_json(output/'summary.json',{**ident,'operational_status':'complete','trajectory_count':3,'waypoint_count':len(targets),
            'teacher_accepted':int(tracked.accepted.sum()),'teacher_missing':int((~tracked.accepted).sum()),
            'elapsed_seconds':time.time()-started,'claim_boundary':'independent locked-model large-path diagnostic; no training, original Gate unchanged; DLS2 is same-model pointwise correction'})
        write_json(output/'completion_manifest.json',{'artifacts':[dict(path=str(p.relative_to(output)),sha256=digest(p)) for p in sorted(output.rglob('*')) if p.is_file() and p.suffix!='.log' and p.name!='progress.json']})
        write_json(output/'progress.json',dict(status='complete',phase='complete',completed=3,total=3))
    except Exception as exc:
        write_json(output/'failure.json',dict(error_type=type(exc).__name__,message=str(exc)))
        write_json(output/'progress.json',dict(status='failed',phase='evaluation',message=str(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--binding-sha',required=True);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    run(args.config.resolve(),args.output_root.resolve(),args.binding_sha,args.prepare_only)
