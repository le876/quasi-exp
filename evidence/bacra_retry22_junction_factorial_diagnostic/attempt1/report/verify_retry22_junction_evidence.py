#!/usr/bin/env python3
"""Read-only independent recheck of sealed retry22 numerical tables and model lock."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from run_bacra_retry20_candidate_diagnostic import environment
from quasi_exp.teacher.retry12_symmetry import BETA_COLUMNS,XYZ_COLUMNS


def verify(root,config_path):
    config=yaml.safe_load(config_path.read_text())
    env=environment({'project_root':str(ROOT),'sources':config['sources']})
    manifest=json.loads((root/'completion_manifest.json').read_text())
    for e in manifest['artifacts']:
        assert hashlib.sha256((root/e['path']).read_bytes()).hexdigest()==e['sha256'],e['path']
    metrics=pd.read_csv(root/'04_evaluation/metrics.csv');cases=[];rows_checked=0;max_fk_delta=0.;max_metric_delta=0.
    datasets={}
    for path in sorted((root/'04_evaluation').glob('*.parquet')):
        d=pd.read_parquet(path);b=d[['pred_'+c for c in BETA_COLUMNS]].to_numpy(float)
        xyz=d[list(XYZ_COLUMNS)].to_numpy(float);actual=env.fk(b)
        recorded=d[['raw_'+c for c in XYZ_COLUMNS]].to_numpy(float)
        max_fk_delta=max(max_fk_delta,float(np.max(np.abs(actual-recorded))))
        np.testing.assert_allclose(actual,recorded,rtol=0,atol=1e-11)
        err=np.linalg.norm(actual-xyz,axis=1)*1000
        np.testing.assert_allclose(err,d.error_mm,rtol=0,atol=1e-8)
        rows_checked+=len(d);datasets[path.stem]=d
    # Independently select every metric's denominator and recompute percentiles.
    zero=float(env.fk(np.zeros(6))[0,0])
    for row in metrics.itertuples():
        d=datasets[f'{row.arm}_{row.group}']
        if row.group=='on_grid':
            d=d[d.split_role.eq(row.split)]
            u=(zero-d.x_m)*1000;rho=np.hypot(d.y_m,d.z_m)*1000
            if row.case=='core':d=d[(u>=60)&(u<=200)&(rho<=80)]
            elif row.case=='transition':d=d[(u>=60)&(u<=200)&(rho>80)&(rho<=280)]
            elif row.case=='tip':d=d[(u>=-1e-6)&(u<60)&(rho<=160)]
            elif row.case=='outer':d=d[d.domain_class.eq('retry18_outer')]
            else:assert row.case=='all'
        else:d=d[d.case.eq(row.case)]
        assert row.count==len(d),(row.arm,row.case,'denominator')
        assert row.nonfinite_count==int((~np.isfinite(d.error_mm)).sum())
        for percentile in (50,95,99):
            expected=np.percentile(d.error_mm,percentile);reported=getattr(row,f'p{percentile}_mm')
            max_metric_delta=max(max_metric_delta,abs(expected-reported));assert abs(expected-reported)<1e-8
        assert abs(float(d.error_mm.max())-row.max_mm)<1e-8
        assert abs(float((d.error_mm<=3).mean())-row.within_3mm)<1e-12
        assert abs(float((d.error_mm<=10).mean())-row.within_10mm)<1e-12
        cases.append((row.arm,row.group,row.split,row.case))
    assert len(cases)==len(set(cases))
    lock=json.loads((root/'model_lock.json').read_text());assert lock['optimizer_updates']==77000 and lock['selection']=='fixed_final_step'
    for e in lock['models'].values():assert hashlib.sha256((root/e['path']).read_bytes()).hexdigest()==e['sha256']
    from scipy.spatial import cKDTree
    source=pd.read_parquet(Path(config['dataset_root'])/'02_dataset/L1_supervision.parquet')
    tr=source[source.split_role.eq('train')];te=source[source.split_role.eq('test')]
    _,nn=cKDTree(tr[list(XYZ_COLUMNS)]).query(te[list(XYZ_COLUMNS)],k=32)
    nouter=(tr.domain_class.to_numpy()[nn]=='retry18_outer').sum(axis=1)
    masks={'all':np.ones(len(te),bool),'mixed_k32':(nouter>0)&(nouter<32),'single_domain':(nouter==0)|(nouter==32)}
    for width in [5,10,20]:masks[f'junction_{width}mm']=(abs(te.x_m*1000-1015.498)<=width).to_numpy()
    paired=pd.read_csv(root/'05_diagnosis/paired_metrics.csv')
    for row in paired.itertuples():
        d=datasets[f'{row.arm}_{row.seed}_on_grid'];d=d[d.split_role.eq('test')]
        c=datasets[f'control_{row.seed}_on_grid'];c=c[c.split_role.eq('test')]
        assert np.array_equal(d.target_id,te.target_id)
        mask=masks[row.subset];err=d.error_mm.to_numpy()[mask];base=c.error_mm.to_numpy()[mask]
        assert row.count==len(err) and row.gt50==int((err>50).sum()) and row.gt100==int((err>100).sum())
        assert row.fixed_control_gt50==int(((base>50)&(err<=50)).sum())
        assert row.new_gt50==int(((base<=50)&(err>50)).sum())
        assert row.regression_gt10==int((err-base>10).sum())
        assert abs(np.percentile(err,95)-row.p95_mm)<1e-8
    fork=json.loads((root/'03_training/paired_state_proof.json').read_text())
    assert len(fork)==8
    for seed in config['seeds']:
        block=[r for r in fork if r['seed']==seed]
        assert len(block)==4 and len({r['initial_state_sha256'] for r in block})==1
        for mode in ['regions','junction']:
            assert len({r['batch_sha256'] for r in block if r['sampling']==mode})==1
        old=pd.read_parquet(Path(config['upstream_root'])/f'04_evaluation/anchored_{seed}_on_grid.parquet')
        control=datasets[f'control_{seed}_on_grid']
        np.testing.assert_array_equal(old[['pred_'+c for c in BETA_COLUMNS]],control[['pred_'+c for c in BETA_COLUMNS]])
        np.testing.assert_array_equal(old.error_mm,control.error_mm)
    return {'artifact_hashes_checked':len(manifest['artifacts']),'prediction_rows_checked':rows_checked,'metric_rows_checked':len(cases),
            'max_batch_vs_recorded_scalar_fk_delta_m':max_fk_delta,'max_percentile_recompute_delta_mm':max_metric_delta,
            'paired_metric_rows_checked':len(paired),'control_vs_retry21_max_beta_delta_rad':0.,'optimizer_updates':77000,'fork_state_identity':True,
            'verification':'pass','scientific_source':manifest['scientific_source_fixed_point'],'binding':manifest['binding_fixed_point'],
            'claim_boundary':'Numerical and provenance verification; no formal or global replacement authorization.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.root.resolve() in args.output.resolve().parents:raise ValueError('verification output must remain outside sealed root')
    result=verify(args.root,args.config);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
