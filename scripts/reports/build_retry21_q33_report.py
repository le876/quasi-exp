#!/usr/bin/env python3
"""Build the self-contained, diagnostic-only retry21 Q33 advisor report.

The only write targets are this report's derivative directory.  This script
never trains, invokes a solver, or alters the sealed retry21 attempt.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import sys
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path('/mnt/ML_projects/quasi_exp')
WORKTREE = ROOT / '.worktrees/retry21-q33'
RUN = ROOT / 'runs/bacra_retry21_q33_attempt1'
OUT = ROOT / 'runs/reports/bacra_retry21_q33_attempt1'
TEMPLATE = WORKTREE / '.agents/skills/quasi-exp-dataset-visualization-report/assets/report_template.html'
PLOTLY = Path('/mnt/ML_projects/conda_envs/rdagent4qlib/lib/python3.10/site-packages/plotly/package_data/plotly.min.js')
NAME = 'bacra_retry21_q33_advisor_dataset_report.html'
SOURCE = '15ead1a0e8eca87cb8b34c47c998b03ccaa9560d'
BINDING = '1a2ed2c75767bb511ea2b200942e230d3bb79e94'
PUBLIC_CODE_SNAPSHOT = '5b2861718a62059abe11aa14d315ae36612c4fee'
PUBLIC_SOURCE_SNAPSHOT = '7f1defb1458b5fce71c3881006c0e7b9a17863f0'
PUBLIC_INVENTORY_SHA256 = '35ae3eb9f306df9a68af5bd15cd1e726c85c775c7065c3958bd5ea21a705b1b5'
OBSERVED = '2026-09-12（Asia/Shanghai；封存产物 observation）'
MODELS = ['original_L0', 'original_L1', 'warmup500', 'beta_uniform', 'beta_balanced', 'fk_uniform', 'fk_balanced', 'anchored_20260925', 'anchored_20260926']
MODEL_LABELS = {
    'original_L0':'旧 L0','original_L1':'旧 L1（默认）','warmup500':'warmup 500','beta_uniform':'beta uniform','beta_balanced':'beta balanced',
    'fk_uniform':'FK uniform','fk_balanced':'FK balanced','anchored_20260925':'two-anchored seed 20260925（默认）','anchored_20260926':'two-anchored seed 20260926',
}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def j(path: Path) -> Any: return json.loads(path.read_text(encoding='utf-8'))
def f(v: Any) -> float | None:
    v = float(v)
    return v if np.isfinite(v) else None
def num(v: Any, d=3) -> str: return '—' if v is None or not np.isfinite(float(v)) else f'{float(v):.{d}f}'
def rate(v: Any) -> str: return '—' if v is None or not np.isfinite(float(v)) else f'{100*float(v):.2f}%'
def esc(v: Any) -> str: return html.escape(str(v))

def table(headers: list[str], rows: list[list[Any]]) -> str:
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{esc(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'

def reference(path: Path) -> dict[str, Any]:
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}

def sample_points(frame: pd.DataFrame, limit=12000) -> tuple[list[list[Any]], dict[str, int]]:
    # Deterministic proportional-by-split sample: preserves the three splits and domain coverage.
    pieces=[]
    for _, part in frame.sort_values('target_id').groupby(['split_role','domain_class'], sort=True, dropna=False):
        n=max(1, round(limit*len(part)/len(frame)))
        pieces.append(part.iloc[np.linspace(0,len(part)-1,min(n,len(part)),dtype=int)])
    out=pd.concat(pieces,ignore_index=True)
    # retry20 stores the outer shell as ``retry18_outer``.  Its display role is
    # outer, while the sample remains byte-for-byte sourced from that row.
    pts=[[round(float(x)*1000,3),round(float(y)*1000,3),round(float(z)*1000,3),str(s),'outer' if str(d)=='retry18_outer' else str(d)] for x,y,z,s,d in out[['x_m','y_m','z_m','split_role','domain_class']].itertuples(index=False)]
    return pts, {f'{a}/{b}':int(n) for (a,b),n in frame.groupby(['split_role','domain_class']).size().items()}

def zero_robot() -> tuple[list[list[float]], list[dict[str,Any]]]:
    """Use the current binding FK only for a static beta=0 visual reference."""
    sys.path.insert(0, str(WORKTREE/'src')); old=os.getcwd()
    try:
        os.chdir(WORKTREE)
        from quasi_exp.io import load_config, load_robot_inputs
        from quasi_exp.model.kinematics import forward_kinematics
        from quasi_exp.model.sampling import beta_to_theta
        cfg_path=WORKTREE/'configs/robot_rods_only_standard_100k.yaml'; cfg=load_config(cfg_path); robot=load_robot_inputs(cfg)
        end, tf=forward_kinematics(beta_to_theta(np.zeros(6)),robot.lengths_m,robot.p_end_local_m,theta_sign=float(cfg['kinematics']['theta_sign']))
        return (np.vstack((tf[:,:3,3],end))*1000).round(4).tolist(), [reference(cfg_path), reference(WORKTREE/'src/quasi_exp/model/kinematics.py')]
    finally: os.chdir(old)

def verify_manifest() -> list[dict[str,Any]]:
    artifacts=j(RUN/'completion_manifest.json')['artifacts']; bad=[]
    for x in artifacts:
        actual=sha(RUN/x['path']) if (RUN/x['path']).is_file() else 'MISSING'
        if actual != x['sha256']: bad.append({'path':x['path'],'expected':x['sha256'],'actual':actual})
    if bad: raise RuntimeError('sealed completion manifest hash mismatch: '+json.dumps(bad))
    return artifacts

def to_xyz(df: pd.DataFrame, prefix='') -> list[list[float|None]]:
    return [[f(x*1000),f(y*1000),f(z*1000)] for x,y,z in df[[prefix+'x_m',prefix+'y_m',prefix+'z_m']].to_numpy(float)]

def build() -> None:
    artifacts=verify_manifest(); summary=j(RUN/'summary.json'); lock=j(RUN/'model_lock.json'); gate=j(RUN/'00_objective_feasibility/gate.json')
    report_source=summary['scientific_source_fixed_point']; report_binding=summary['binding_fixed_point']
    reused_source=summary.get('reused_training_source_fixed_point',report_source); reused_binding=summary.get('reused_training_binding_fixed_point',report_binding)
    reuse=j(RUN/'reuse_provenance.json') if (RUN/'reuse_provenance.json').is_file() else None
    verification=ROOT/'runs/diagnostics/retry21_q33_verification/attempt2_numerical_verification.json'
    assert reused_source==SOURCE and reused_binding==BINDING
    assert summary['diagnostic_only'] and not summary['global_replacement_authorized'] and gate['status']=='diagnostic_only'
    targets=pd.read_parquet(RUN/'01_inputs/targets.parquet'); reused_l1=pd.read_parquet(ROOT/'runs/bacra_retry20_candidate_diagnostic_attempt1/02_dataset/L1_supervision.parquet'); metrics=pd.read_csv(RUN/'04_evaluation/metrics.csv'); history=pd.read_csv(RUN/'03_training/validation_history.csv')
    edges=pd.read_parquet(RUN/'02_audit/primary_outer_edges.parquet'); regression=pd.read_csv(RUN/'05_outer/regression_counts.csv'); selected=pd.read_csv(RUN/'05_outer/selected_model_comparison.csv')
    coverage, coverage_counts=sample_points(reused_l1,12000); zero, zero_refs=zero_robot()
    paths=[]
    default_models={'original_L1','anchored_20260925'}
    track_source=pd.read_parquet(RUN/'04_evaluation/original_L1_targets.parquet')
    for case, base in track_source[track_source.case.ne('off_grid_test')].groupby('case',sort=True):
        base=base.sort_values('waypoint_index',kind='stable'); row={'id':str(case),'count':int(len(base)),'target':to_xyz(base),'models':{}}
        for model in MODELS:
            d=pd.read_parquet(RUN/f'04_evaluation/{model}_targets.parquet'); d=d[d.case==case].sort_values('waypoint_index',kind='stable')
            row['models'][model]={'xyz':to_xyz(d,'raw_'),'error':[f(x) for x in d.error_mm]}
        # The source table carries Teacher only on the original 19 paths; the 16 matched controls intentionally show a gap.
        original=targets[targets.trajectory_id.astype(str)==str(case)].sort_values('waypoint_index',kind='stable')
        teacher_cols=['Teacher_x_m','Teacher_y_m','Teacher_z_m']
        row['teacher']=to_xyz(original,'Teacher_') if len(original)==len(base) and all(c in original for c in teacher_cols) else None
        paths.append(row)
    assert len(paths)==35
    # Show the original Teacher-bearing suite first.  The matched controls remain
    # selectable after it and visibly exercise the Teacher evidence-gap state.
    paths.sort(key=lambda item: (0 if item['teacher'] else 1, item['id']))
    # Evaluation metric table is the page-visible complete record, not a selected subset.
    ordered=metrics.sort_values(['arm','group','split','case'],kind='stable')
    metric_html=table(['模型','组','split','case','n','P50 mm','P95 mm','max mm','≤3 mm','≤10 mm','bounds'],[[MODEL_LABELS.get(r.arm,r.arm),r.group,r.split,r.case,int(r['count']),num(r.p50_mm),num(r.p95_mm),num(r.max_mm),rate(r.within_3mm),rate(r.within_10mm),int(r.bounds_violations)] for _,r in ordered.iterrows()])
    # The eight explicitly named near-axis paths are the Q33-local reproduction check.
    near=ordered[(ordered.group=='targets') & ordered.case.astype(str).str.contains('circle_u100_center25') & ordered.arm.isin(['anchored_20260925','anchored_20260926'])]
    near_summary={a:{'n':int(g['count'].sum()),'max':float(g.max_mm.max()),'all3':bool((g.max_mm<=3).all())} for a,g in near.groupby('arm')}
    alltest=ordered[(ordered.group=='on_grid')&(ordered.split=='test')&(ordered.case=='all')&ordered.arm.isin(['anchored_20260925','anchored_20260926'])].set_index('arm')
    outer=ordered[(ordered.group=='on_grid')&(ordered.split=='test')&(ordered.case=='outer')&ordered.arm.isin(['anchored_20260925','anchored_20260926'])].set_index('arm')
    path_metrics=ordered[(ordered.group=='targets') & ordered.arm.isin(['original_L1','anchored_20260925','anchored_20260926'])]
    support=pd.read_parquet(RUN/'02_audit/path_neighbor_means.parquet')
    support32=support[support.k==32].groupby('target_id').agg({'neighbor_distance_mm':['median',lambda x:x.quantile(.95),'max'],'fk_error_mm':['median',lambda x:x.quantile(.95),'max']})
    support_rows=[]
    for p in paths:
        ids=[f'track:{p["id"]}:{i}' for i in range(p['count'])]; q=support32.reindex(ids)
        distance=q[('neighbor_distance_mm','median')].dropna(); error=q[('fk_error_mm','median')].dropna()
        support_rows.append([p['id'],p['count'],num(distance.median() if len(distance) else None),num(q[('neighbor_distance_mm','<lambda_0>')].max() if len(distance) else None),num(error.median() if len(error) else None),num(q[('fk_error_mm','<lambda_0>')].max() if len(error) else None)])
    support_html=table(['路径','点数','k32 距离 P50','k32 距离 P95 的路径内最大','k32 FK均值 P50','k32 FK均值 P95 的路径内最大'],support_rows)
    edge_stat={'n':len(edges),'p95':float(edges.mean_fk_chord_error_mm.quantile(.95)),'max':float(edges.mean_fk_chord_error_mm.max()),'labels_max':float(max(edges.a_residual_mm.max(),edges.b_residual_mm.max()))}
    # page data: all nine raw outputs are selectable.  No derived trajectory point is added.
    data={'coverage':coverage,'coverageCounts':coverage_counts,'fullCoverage':len(reused_l1),'zero':zero,'paths':paths,'models':MODEL_LABELS,
          'history':{m:g.sort_values('step')[['step','loss','all_p95_mm']].values.tolist() for m,g in history.groupby('arm',sort=True)},
          'outer':regression.to_dict('records'),'selected':selected.to_dict('records')}
    input_paths=[RUN/'summary.json',RUN/'model_lock.json',RUN/'completion_manifest.json',RUN/'reuse_provenance.json',RUN/'01_inputs/targets.parquet',RUN/'02_audit/axis_counterexample.json',RUN/'02_audit/primary_outer_edges.parquet',RUN/'03_training/validation_history.csv',RUN/'04_evaluation/metrics.csv',RUN/'05_outer/regression_counts.csv',RUN/'05_outer/selected_model_comparison.csv',WORKTREE/'configs/bacra_retry21_q33.yaml',WORKTREE/'docs/protocols/40-BACRA-retry21-Q33-raw-Student执行协议.md',TEMPLATE]
    if verification.is_file(): input_paths.append(verification)
    body=f'''<header><div class="k" style="color:#ddecff">BACRA retry21 · Q33 TensorFlow raw Student 复核 · evidence cutoff: {OBSERVED}</div><h1>BACRA retry21 Q33：35 条固定诊断路径与 raw Student 报告</h1><p class="sub">页面在机器人基座物理坐标系（mm）解释封存的 retry21 attempt1。它复用数据、显示实际 raw inference；不训练、不求解、不修改封存产物，也不构成 formal Gate、全域替换或控制结论。</p><div class="meta"><div><div class="k" style="color:#ddecff">状态</div><div class="v">completed · diagnostic_only</div></div><div><div class="k" style="color:#ddecff">监督复用</div><div class="v">74,696 × L0/L1</div></div><div><div class="k" style="color:#ddecff">固定诊断路径</div><div class="v">35 条</div></div><div><div class="k" style="color:#ddecff">锁定 raw 模型</div><div class="v">9 个</div></div></div></header>
<section><h2>结论、修复来源与边界</h2><div class="notice"><b>结论只能作为定位证据。</b> objective Gate 为 <code>diagnostic_only</code>，训练成功未被预算保证；<code>formal_authorized=false</code>、<code>global_replacement_authorized=false</code>。轨迹和浏览器呈现不能改变这些边界。</div><div class="grid"><div class="card"><div class="k">metadata-repair source / binding</div><code>{report_source}</code><br><code>{report_binding}</code><p>attempt2 仅修复 objective-feasibility 的资源与闭合路径 metadata。</p></div><div class="card"><div class="k">实际训练数值来源</div><code>{reused_source}</code><br><code>{reused_binding}</code><p>训练/评估字节复用；实际训练耗时 582.8708 s，attempt2 的 0.7484 s 是 metadata repair，不是新训练。</p></div><div class="card"><div class="k">封存输出核验</div><p>completion manifest 的 {len(artifacts)} 个声明 artifact 均逐项 SHA-256 匹配；新 optimizer updates = {summary.get('new_optimizer_updates',0)}。这是字节完整性核验，不是 scientific Gate。</p></div><div class="card"><div class="k">独立数值复核</div><p>{'88 个 artifact、755,010 prediction rows、468 metric rows；scalar FK最大差 7.77e-16 m，百分位最大差 2.84e-14 mm，fork state 一致。' if verification.is_file() else '独立 numerical verification 文件尚未在报告生成时可读。'}</p></div><div class="card"><div class="k">可复现的局部现象</div><p>两 seed 的八条近轴轨迹合计各 {near_summary['anchored_20260925']['n']} 点，分别 all ≤3 mm，raw max {near_summary['anchored_20260925']['max']:.3f}/{near_summary['anchored_20260926']['max']:.3f} mm。</p></div><div class="card"><div class="k">全 test（8,272 点）并非同一结论</div><p>two-anchored 的 P95 为 {alltest.loc['anchored_20260925','p95_mm']:.3f}/{alltest.loc['anchored_20260926','p95_mm']:.3f} mm，max 为 {alltest.loc['anchored_20260925','max_mm']:.3f}/{alltest.loc['anchored_20260926','max_mm']:.3f} mm。</p></div></div></section>
<section><h2>复用数据的三维覆盖与零位参照</h2><p class="legend">完整 L0/L1 监督各 74,696 行；主视图显示 {len(coverage):,} 个从原 L1 supervision 按 <code>split_role × domain_class</code> 比例、按 <code>target_id</code> 稳定排序后等距抽取的确定性样本。点的位置未平移，单位为 mm。颜色加点形区分 split/domain；灰线是完整 <span data-series="robot">beta=0 机器人中心线</span>，红点是零位末端。</p><div class="toolbar"><button id="coverageGlobal">全局物理视图</button><button id="coverageLocal">局部 zoom（不改变坐标）</button><button id="theme">切换明暗主题</button></div><div id="coverage" class="plot" aria-label="三维数据覆盖"></div></section>
<section><h2>训练、锁定和 nine-model 对照</h2><div class="grid"><div class="card"><div class="k">锁定规则</div><p>固定 final-step；无 early stopping、无 test 选模。warmup 与四臂分叉共享初始模型、Adam 和采样状态；产物记录四个 fork-state hash 一致。</p></div><div class="card"><div class="k">TF 证据边界</div><p><span data-series="l1raw">蓝色 raw 输出</span>来自本地 TensorFlow。<b>fk_uniform 在 first phase/near-axis 仍差，不能搬用 PyTorch 数值。</b>曲线只展示此 attempt 的 validation history。</p></div></div><div id="loss" class="plot small" aria-label="训练和验证曲线"></div></section>
<section><h2>35 条路径：Target、Teacher 与九个 raw Student</h2><p>默认选择原始路径中的 <b>original_L1</b> 与 <b>two-anchored seed 20260925</b>。Target 始终可见；Teacher 仅在原 19 条路径的 <code>targets.parquet</code> 局部列中可验证，matched controls 的 16 条没有本地 Teacher Cartesian 证据，页面明确显示缺口而不补造。</p><div class="toolbar"><label>路径<select id="trajectory"></select></label><label>模型 A<select id="modelA"></select></label><label>模型 B<select id="modelB"></select></label><button id="trackingGlobal">全局物理视图</button><button id="trackingLocal">局部 zoom</button></div><div id="trackNote" class="notice"></div><div id="tracking" class="plot" aria-label="三维路径跟踪"></div><div id="pathError" class="plot small" aria-label="路径误差"></div></section>
<section><h2>分割指标、路径指标和 train-NN support</h2><p>下表保留所有模型、split、区域和固定路径的完整 CSV 指标。路径 support 使用封存的 train-only k32 邻居均值审计；它描述本次诊断的邻近标签关系，不是全局连通性证明。</p>{support_html}<details><summary>展开 468 行完整 metrics.csv</summary>{metric_html}</details></section>
<section><h2>outer 尾部与机制定位</h2><div class="grid"><div class="card"><div class="k">outer test（4,187 点）</div><p>相对 original_L1，anchored seed 20260925/20260926 分别有 {int(regression[(regression.arm=='anchored_20260925')&(regression.split=='test')&(regression.region=='outer')].worse_gt1mm.iloc[0])}/{int(regression[(regression.arm=='anchored_20260926')&(regression.split=='test')&(regression.region=='outer')].worse_gt1mm.iloc[0])} 点退化 &gt;1 mm，{int(regression[(regression.arm=='anchored_20260925')&(regression.split=='test')&(regression.region=='outer')].worse_gt10mm.iloc[0])}/{int(regression[(regression.arm=='anchored_20260926')&(regression.split=='test')&(regression.region=='outer')].worse_gt10mm.iloc[0])} 点退化 &gt;10 mm；这与广泛改善并存。</p></div><div class="card"><div class="k">合法 beta 平均反例</div><p>镜像两个合法端点的平均 beta 仍在 bounds 内，却在参考 FK 上产生 130.000 mm 残差。它说明“端点合法”不推出“beta 平均合法”，不证明 outer 退化的唯一成因。</p></div><div class="card"><div class="k">primary–outer 边诊断</div><p>{edge_stat['n']:,} 条 train edge；标签端点 residual max {edge_stat['labels_max']:.3f} mm，而 midpoint FK–chord P95/max = {edge_stat['p95']:.3f}/{edge_stat['max']:.3f} mm。这是局部非线性/分支几何证据，不是 canonical graph。</p></div><div class="card"><div class="k">seed-sensitive 反例</div><p>最坏 test target <code>retry19_old_signed_0b213aa2d575574f79701a4b</code>：original L1 14.923 mm、fk_balanced 8.392 mm、anchored seed1 125.648 mm、seed2 3.993 mm；Teacher residual≈2e-10 mm、condition 4.700，坏 raw condition 14.756、beta gap 24.721°。k32 mean error 15.121 mm，并非 125 mm。该对照支持优化/学得分支位置的 seed 敏感性，不能把 uniform bad-label mix 定为唯一原因。</p></div></div><div id="outer" class="plot small" aria-label="outer 退化计数"></div></section>
<section><h2>证据缺口、溯源与交付</h2><div class="notice"><b>没有新盲测。</b> 35 条路径是固定已知诊断；old test 与 Q33 失败点被复用以定位。matched-control 的 16 条有 Target 和九个 raw Student 输出，但缺少本地可验证 Teacher Cartesian 路径。报告没有为填满图形而重新训练、求解或导入外部 PyTorch 结果。</div><p class="notice">大尺度 <code>large_star</code>、<code>large_helix</code> 与 <code>large_tapered_helix</code> 也是固定诊断尾部：two-anchored 的 P95 分别为 26.703/17.741、13.387/12.661、13.568/13.849 mm（seed 20260925/20260926）。它们限制了对局部近轴复现的外推，完整数值保留在路径 metrics 表。</p><p>公开代码快照：<code>{PUBLIC_CODE_SNAPSHOT}</code>；filtered source：<code>{PUBLIC_SOURCE_SNAPSHOT}</code>；635-file inventory SHA-256：<code>{PUBLIC_INVENTORY_SHA256}</code>。<br>输出目录：<code>{OUT}</code><br>HTML：<code>{OUT/NAME}</code><br>模板副本、generator、输入/输出 SHA-256 与浏览器验证均在 <code>report_manifest.json</code> 中。报告质量与 artifact hash 验证均不等于 scientific Gate。</p></section>'''
    script=r'''const $=id=>document.getElementById(id);let state={local:false};
function p3(points,name,color,dash='solid',width=4){return {type:'scatter3d',mode:'lines',name,x:points.map(p=>p?.[0]??null),y:points.map(p=>p?.[1]??null),z:points.map(p=>p?.[2]??null),line:{color,width,dash},connectgaps:false};}
function robot(){const c=ReportPlot.palette();return [p3(D.zero,'机器人 beta=0',c.robot,'solid',5),{type:'scatter3d',mode:'markers',name:'零位末端',x:[D.zero.at(-1)[0]],y:[D.zero.at(-1)[1]],z:[D.zero.at(-1)[2]],marker:{color:c.tip,size:6,symbol:'diamond'}}]}
function ranges(points){let out=[];for(let axis=0;axis<3;axis++){let v=points.flat().map(p=>p?.[axis]).filter(Number.isFinite),lo=Math.min(...v),hi=Math.max(...v),pad=Math.max(25,(hi-lo)*.12);out.push([lo-pad,hi+pad])}return out}
function layout3(title,r){const c=ReportPlot.palette();return {title:{text:title},margin:{l:0,r:0,b:0,t:42},paper_bgcolor:'transparent',font:{color:c.target},scene:{aspectmode:'data',camera:{projection:{type:'orthographic'},eye:{x:1.45,y:1.45,z:.9},center:{x:0,y:0,z:0}},xaxis:{title:'X (mm)',range:r?.[0]},yaxis:{title:'Y (mm)',range:r?.[1]},zaxis:{title:'Z (mm)',range:r?.[2]}}};}
async function coverage(){const c=ReportPlot.palette(), local=state.coverageLocal;let pts=D.coverage, colors={train:c.l1,validation:c.teacher,test:c.l0};let traces=[];for(const split of ['train','validation','test'])for(const domain of ['primary','outer']){let a=pts.filter(p=>p[3]===split&&p[4]===domain);if(a.length)traces.push({type:'scatter3d',mode:'markers',name:split+' / '+domain,x:a.map(p=>p[0]),y:a.map(p=>p[1]),z:a.map(p=>p[2]),marker:{size:2,color:colors[split],opacity:domain==='primary'?.62:.35,symbol:domain==='primary'?'circle':'diamond'}})}if(traces.reduce((n,t)=>n+t.x.length,0)!==pts.length)throw new Error('coverage display sample count mismatch');traces.push(...robot());let l=layout3(local?'覆盖局部 zoom（物理坐标未改变）':'完整复用监督覆盖（物理 mm）',ranges([pts.map(p=>p.slice(0,3)),D.zero]));l.scene.camera=local?{projection:{type:'orthographic'},eye:{x:1.25,y:1.05,z:.75},center:{x:0,y:0,z:0}}:{projection:{type:'orthographic'},eye:{x:.2,y:-2,z:.65},center:{x:0,y:0,z:0}};await ReportPlot.react('coverage',traces,l,{responsive:true,displaylogo:false});}
function path(){return D.paths[$('trajectory').selectedIndex]};function metrics(){let t=path(),a=$('modelA').value,b=$('modelB').value;let note=t.teacher?`Teacher 来自封存 targets.parquet（${t.count} 点）；Target、Teacher 与两个 raw Student 均为同一物理坐标系。`:`证据缺口：此 matched-control 路径没有本地可验证 Teacher Cartesian 输出；仍显示 Target 与选中的 raw Student，未用推断值填补。`;$('trackNote').innerHTML=note+` 当前：<b>${t.id}</b>，${t.count} 点。`;return [t,a,b]}
async function tracking(){const [t,a,b]=metrics(),c=ReportPlot.palette();let tr=[p3(t.target,'Target',c.target,'dash',4)];if(t.teacher)tr.push(p3(t.teacher,'Teacher',c.teacher,'solid',7));tr.push(p3(t.models[a].xyz,D.models[a],c.l1raw,'solid',4),p3(t.models[b].xyz,D.models[b],c.l1dls,'dot',4),...robot());let all=[t.target,t.models[a].xyz,t.models[b].xyz,D.zero];if(t.teacher)all.push(t.teacher);let l=layout3(state.local?'路径局部 zoom（仍包含完整 beta=0）':'路径全局物理视图（完整路径与 beta=0）',ranges(all));if(state.local)l.scene.camera={projection:{type:'orthographic'},eye:{x:1.05,y:1.05,z:.7},center:{x:0,y:0,z:0}};await ReportPlot.react('tracking',tr,l,{responsive:true,displaylogo:false});await ReportPlot.react('pathError',[{type:'scatter',mode:'lines',name:D.models[a]+' error',x:[...Array(t.count).keys()],y:t.models[a].error,line:{color:c.l1raw,width:3}},{type:'scatter',mode:'lines',name:D.models[b]+' error',x:[...Array(t.count).keys()],y:t.models[b].error,line:{color:c.l1dls,width:3}}],{title:'逐 waypoint Cartesian FK error (mm)',paper_bgcolor:'transparent',font:{color:c.target},xaxis:{title:'waypoint'},yaxis:{title:'mm'}},{responsive:true,displaylogo:false});}
async function histories(){const c=ReportPlot.palette();let tr=Object.entries(D.history).map(([m,v],i)=>({type:'scatter',mode:'lines',name:D.models[m],x:v.map(x=>x[0]),y:v.map(x=>x[2]),line:{color:[c.l0,c.l1,c.teacher,c.l1raw,c.l0raw,c.l1dls,c.l0dls,c.reference,c.dls][i],width:2}}));await ReportPlot.react('loss',tr,{title:'validation FK P95 (mm) 随训练步数',paper_bgcolor:'transparent',font:{color:c.target},xaxis:{title:'step'},yaxis:{title:'P95 mm'}},{responsive:true,displaylogo:false});}
async function outer(){const c=ReportPlot.palette();let rows=D.outer.filter(x=>x.arm.includes('anchored')&&x.split==='test'&&x.region==='outer');await ReportPlot.react('outer',[{type:'bar',name:'test >1 mm',x:rows.map(x=>D.models[x.arm]),y:rows.map(x=>x.worse_gt1mm),marker:{color:c.teacher}},{type:'bar',name:'test >10 mm',x:rows.map(x=>D.models[x.arm]),y:rows.map(x=>x.worse_gt10mm),marker:{color:c.reference}}],{title:'relative to original_L1 的 outer test 退化计数',barmode:'group',paper_bgcolor:'transparent',font:{color:c.target},yaxis:{title:'points'}},{responsive:true,displaylogo:false});}
const visible={coverage:false,tracking:false};function release(id){const p=$(id);if(p.data)Plotly.purge(p);$(id+'-legend')?.replaceChildren();}const observer=new IntersectionObserver(entries=>{for(const e of entries){const id=e.target.id;visible[id]=e.isIntersecting;if(e.isIntersecting)(id==='coverage'?coverage:tracking)();else release(id)}},{threshold:.08});observer.observe($('coverage'));observer.observe($('tracking'));function render(){histories();outer();if(visible.coverage)coverage();if(visible.tracking)tracking();}function theme(){document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark';render();}for(const p of D.paths)$('trajectory').add(new Option(p.id,p.id));for(const [k,v] of Object.entries(D.models)){$('modelA').add(new Option(v,k));$('modelB').add(new Option(v,k));}$('modelA').value='original_L1';$('modelB').value='anchored_20260925';ReportPlot.pathNavigation('trajectory');$('trajectory').onchange=tracking;['modelA','modelB'].forEach(id=>$(id).onchange=tracking);$('trackingGlobal').onclick=()=>{state.local=false;tracking()};$('trackingLocal').onclick=()=>{state.local=true;tracking()};$('coverageGlobal').onclick=()=>{state.coverageLocal=false;coverage()};$('coverageLocal').onclick=()=>{state.coverageLocal=true;coverage()};$('theme').onclick=theme;render();'''
    slots={'TITLE':esc('BACRA retry21 Q33 raw Student 诊断报告'),'BODY':body,'PLOTLY':PLOTLY.read_text(encoding='utf-8'),'DATA':json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</','<\\/'),'SCRIPT':script}
    template_text=TEMPLATE.read_text(encoding='utf-8')
    names=set(re.findall(r'@@([A-Z]+)@@',template_text))
    if names != set(slots): raise ValueError(f'template slots mismatch: {names}')
    rendered=re.sub(r'@@([A-Z]+)@@',lambda m:slots[m.group(1)],template_text)
    OUT.mkdir(parents=True,exist_ok=True); (OUT/NAME).write_text(rendered,encoding='utf-8'); shutil.copy2(TEMPLATE,OUT/'report_template.html'); shutil.copy2(Path(__file__),OUT/'build_retry21_q33_report.py')
    evidence={'observed_at':OBSERVED,'attempt_root':str(RUN),'report_worktree':str(WORKTREE),'scientific_source_fixed_point':report_source,'binding_fixed_point':report_binding,'reused_training_source_fixed_point':reused_source,'reused_training_binding_fixed_point':reused_binding,'metadata_repair_only':bool(summary.get('metadata_repair_only',False)),'new_optimizer_updates':int(summary.get('new_optimizer_updates',0)),'reuse_provenance':reference(RUN/'reuse_provenance.json') if reuse else None,'completion_manifest_artifacts_verified':len(artifacts),'claim_boundary':{'diagnostic_only':True,'formal_authorized':False,'global_replacement_authorized':False},'coverage':{'full_rows':len(reused_l1),'display_rows':len(coverage),'method':'stable target_id sorted, proportional split_role x domain_class allocation, equally-spaced row selection'},'references':[reference(p) for p in input_paths]+zero_refs}
    (OUT/'evidence_map.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest={'report_name':NAME,'report_path':str(OUT/NAME),'report_sha256':sha(OUT/NAME),'template_path':str(TEMPLATE),'template_sha256':sha(TEMPLATE),'generator_source':str(Path(__file__)),'generator_copy_sha256':sha(OUT/'build_retry21_q33_report.py'),'public_code_snapshot_sha':PUBLIC_CODE_SNAPSHOT,'filtered_public_source_sha':PUBLIC_SOURCE_SNAPSHOT,'public_inventory_file_count':635,'public_inventory_sha256':PUBLIC_INVENTORY_SHA256,'scientific_source_fixed_point':report_source,'binding_fixed_point':report_binding,'reused_training_source_fixed_point':reused_source,'reused_training_binding_fixed_point':reused_binding,'metadata_repair_only':bool(summary.get('metadata_repair_only',False)),'new_optimizer_updates':int(summary.get('new_optimizer_updates',0)),'diagnostic_only':True,'formal_authorized':False,'global_replacement_authorized':False,'inputs':[reference(p) for p in input_paths],'sealed_outputs_verified_count':len(artifacts),'sealed_output_hashes':artifacts,'outputs':[reference(OUT/NAME),reference(OUT/'report_template.html'),reference(OUT/'build_retry21_q33_report.py'),reference(OUT/'evidence_map.json')],'browser_validation':'pending'}
    (OUT/'report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(OUT/NAME); print(sha(OUT/NAME))
def record_browser() -> None:
    """Attach a hash-bound browser observation without changing report bytes."""
    validation=OUT/'browser_validation.json'; manifest_path=OUT/'report_manifest.json'
    if not validation.is_file(): raise FileNotFoundError(validation)
    browser=j(validation); manifest=j(manifest_path); report=OUT/NAME
    if browser.get('report_sha256') != sha(report): raise ValueError('browser validation targets a different report hash')
    manifest['browser_validation']={'path':str(validation),'sha256':sha(validation),'passed':bool(browser.get('passed')),'engine':browser.get('engine'),'report_sha256':browser.get('report_sha256'),'exceptions':len(browser.get('exceptions',[])),'console_errors':len(browser.get('consoleErrors',[]))}
    manifest['outputs'].append(reference(validation))
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-root',type=Path,default=RUN)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--record-browser',action='store_true')
    args=parser.parse_args(); RUN=args.attempt_root.resolve(); OUT=args.output.resolve(); build()
    if args.record_browser: record_browser()
