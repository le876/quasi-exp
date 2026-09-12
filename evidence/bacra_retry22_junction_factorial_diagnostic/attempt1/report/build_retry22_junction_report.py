#!/usr/bin/env python3
"""Build the self-contained advisor-facing retry22 junction factorial report.

This only reads the frozen attempt and writes derivative report files.  It does
not train, solve, or modify sealed experiment artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path('/mnt/ML_projects/quasi_exp')
WORKTREE = ROOT / '.worktrees/retry22-junction'
RUN = ROOT / 'runs/bacra_retry22_junction_attempt1'
OUT = ROOT / 'runs/reports/bacra_retry22_junction_attempt1'
TEMPLATE = WORKTREE / '.agents/skills/quasi-exp-dataset-visualization-report/assets/report_template.html'
PLOTLY = Path('/mnt/ML_projects/conda_envs/rdagent4qlib/lib/python3.10/site-packages/plotly/package_data/plotly.min.js')
VERIFY = ROOT / 'runs/diagnostics/retry22_junction_verification/numerical_verification.json'
NAME = 'bacra_retry22_junction_advisor_dataset_report.html'
OBSERVED = '2026-09-13（Asia/Shanghai；冻结 attempt 完成后读取）'
ARMS = ['control_20260925', 'junction_20260925', 'task_only_20260925', 'junction_task_only_20260925',
        'control_20260926', 'junction_20260926', 'task_only_20260926', 'junction_task_only_20260926']
LABELS = {'control_20260925':'control · seed 20260925','junction_20260925':'junction · seed 20260925',
          'task_only_20260925':'task only · seed 20260925','junction_task_only_20260925':'junction + task only · seed 20260925',
          'control_20260926':'control · seed 20260926','junction_20260926':'junction · seed 20260926',
          'task_only_20260926':'task only · seed 20260926','junction_task_only_20260926':'junction + task only · seed 20260926'}

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def j(p: Path) -> Any: return json.loads(p.read_text(encoding='utf-8'))
def esc(x: Any) -> str: return html.escape(str(x))
def n(x: Any, d: int = 3) -> str:
    return '—' if x is None or not np.isfinite(float(x)) else f'{float(x):.{d}f}'
def pct(x: Any) -> str: return '—' if x is None or not np.isfinite(float(x)) else f'{100*float(x):.2f}%'
def ref(p: Path) -> dict[str, Any]: return {'path':str(p), 'bytes':p.stat().st_size, 'sha256':sha(p)}
def table(head: list[str], rows: list[list[Any]]) -> str:
    return '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{esc(x)}</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'

def verify_manifest() -> list[dict[str, Any]]:
    artifacts = j(RUN/'completion_manifest.json')['artifacts']; bad=[]
    for x in artifacts:
        actual = sha(RUN/x['path']) if (RUN/x['path']).is_file() else 'MISSING'
        if actual != x['sha256']: bad.append({'path':x['path'],'expected':x['sha256'],'actual':actual})
    if bad: raise RuntimeError('completion manifest hash mismatch: '+json.dumps(bad))
    return artifacts

def robot_zero() -> tuple[list[list[float]], list[dict[str, Any]]]:
    sys.path.insert(0, str(WORKTREE/'src')); old=os.getcwd()
    try:
        os.chdir(WORKTREE)
        from quasi_exp.io import load_config, load_robot_inputs
        from quasi_exp.model.kinematics import forward_kinematics
        from quasi_exp.model.sampling import beta_to_theta
        cfgp=WORKTREE/'configs/robot_rods_only_standard_100k.yaml'; cfg=load_config(cfgp); robot=load_robot_inputs(cfg)
        end, tf=forward_kinematics(beta_to_theta(np.zeros(6)),robot.lengths_m,robot.p_end_local_m,theta_sign=float(cfg['kinematics']['theta_sign']))
        return (np.vstack((tf[:,:3,3],end))*1000).round(4).tolist(), [ref(cfgp), ref(WORKTREE/'src/quasi_exp/model/kinematics.py')]
    finally: os.chdir(old)

def sample_coverage(frame: pd.DataFrame, limit: int=12000) -> tuple[list[list[Any]], dict[str,int]]:
    parts=[]
    for _, g in frame.sort_values('target_id').groupby(['split_role','domain_class'], sort=True, dropna=False):
        parts.append(g.iloc[np.linspace(0,len(g)-1,min(len(g),max(1,round(limit*len(g)/len(frame)))),dtype=int)])
    out=pd.concat(parts,ignore_index=True)
    pts=[]
    for x,y,z,split,domain in out[['x_m','y_m','z_m','split_role','domain_class']].itertuples(index=False):
        # The stored retry18_outer value is a provenance domain label; its display role is outer.
        pts.append([round(float(x)*1000,3),round(float(y)*1000,3),round(float(z)*1000,3),str(split),'outer' if str(domain)=='retry18_outer' else str(domain)])
    return pts, {f'{a}/{b}':int(v) for (a,b),v in frame.groupby(['split_role','domain_class']).size().items()}

def xyz(d: pd.DataFrame, prefix: str='') -> list[list[float|None]]:
    return [[float(x*1000) if np.isfinite(x) else None,float(y*1000) if np.isfinite(y) else None,float(z*1000) if np.isfinite(z) else None] for x,y,z in d[[prefix+'x_m',prefix+'y_m',prefix+'z_m']].to_numpy(float)]

def build() -> None:
    artifacts=verify_manifest(); summary=j(RUN/'summary.json'); lock=j(RUN/'model_lock.json'); gate=j(RUN/'00_objective_feasibility/gate.json'); verification=j(VERIFY)
    assert summary['status']=='completed' and summary['diagnostic_only'] and not summary['global_replacement_authorized']
    assert gate['status']=='diagnostic_only' and len(lock['models'])==8
    source,binding=summary['scientific_source_fixed_point'],summary['binding_fixed_point']
    assert source=='e63318d68016d8a60f35c832318c03a729d9fdd3' and binding=='f94d8743ee9cf35f25f6c6f26d2c8e9bed1e1190'
    assert verification['verification']=='pass' and verification['artifact_hashes_checked']==84 and verification['metric_rows_checked']==416
    l1=pd.read_parquet(ROOT/'runs/bacra_retry20_candidate_diagnostic_attempt1/02_dataset/L1_supervision.parquet')
    targets=pd.read_parquet(RUN/'01_inputs/targets.parquet'); metrics=pd.read_csv(RUN/'04_evaluation/metrics.csv'); paired=pd.read_csv(RUN/'05_diagnosis/paired_metrics.csv'); hist=pd.read_csv(RUN/'03_training/validation_history.csv')
    coverage,cover_counts=sample_coverage(l1); zero,zero_refs=robot_zero()
    paths=[]
    base=pd.read_parquet(RUN/'04_evaluation/control_20260925_targets.parquet')
    for case,g in base[base.case.ne('off_grid_test')].groupby('case',sort=True):
        g=g.sort_values('waypoint_index',kind='stable'); src=targets[targets.trajectory_id.astype(str)==str(case)].sort_values('waypoint_index',kind='stable')
        teacher = xyz(src,'Teacher_') if len(src)==len(g) and src.Teacher_x_m.notna().all() else None
        row={'id':str(case),'count':len(g),'target':xyz(g),'teacher':teacher,'models':{}}
        for arm in ARMS:
            d=pd.read_parquet(RUN/f'04_evaluation/{arm}_targets.parquet'); d=d[d.case==case].sort_values('waypoint_index',kind='stable')
            if len(d)!=len(g): raise ValueError(f'{arm}/{case} length mismatch')
            row['models'][arm]={'xyz':xyz(d,'raw_'),'error':[float(x) if np.isfinite(x) else None for x in d.error_mm]}
        paths.append(row)
    assert len(paths)==35
    paths.sort(key=lambda p:(0 if p['teacher'] else 1,p['id']))
    alltest=metrics[(metrics.group=='on_grid')&(metrics.split=='test')&(metrics.case=='all')].set_index('arm')
    paired['arm_key']=paired.arm.astype(str)+'_'+paired.seed.astype(str)
    paired_all=paired[paired.subset=='all'].set_index('arm_key')
    paired_rows=[]
    for arm in ARMS:
        r=paired_all.loc[arm]
        paired_rows.append([LABELS[arm],int(r.gt50),int(r.fixed_control_gt50),int(r.new_gt50),bool(r.small_paths_retained)])
    near_cases=['retry19_exact_seam_rectangle_0','retry19_exact_seam_rectangle_1','retry19_exact_seam_rectangle_2',
                'retry19_seam_between_rectangle_0','retry19_seam_between_rectangle_1','retry19_seam_between_rectangle_2',
                'retry19_slanted_ellipse','retry19_spiral_3d']
    near_max=metrics[(metrics.group=='targets') & metrics.case.isin(near_cases)].groupby('arm').max_mm.max().to_dict()
    if set(near_max)!=set(ARMS): raise ValueError('eight near-axis path metrics are incomplete')
    metric_rows=[]
    for arm in ARMS:
        r=alltest.loc[arm]; metric_rows.append([LABELS[arm],int(r['count']),n(r.p50_mm),n(r.p95_mm),n(r.p99_mm),n(r.max_mm),pct(r.within_3mm),pct(r.within_10mm)])
    ordered=metrics.sort_values(['arm','group','split','case'],kind='stable')
    metrics_html=table(['arm','group','split','case','n','P50 mm','P95 mm','max mm','≤3 mm','≤10 mm','bounds'],[[LABELS.get(r.arm,r.arm),r.group,r.split,r.case,int(r['count']),n(r.p50_mm),n(r.p95_mm),n(r.max_mm),pct(r.within_3mm),pct(r.within_10mm),int(r.bounds_violations)] for _,r in ordered.iterrows()])
    # Exact protocol result: severe count and the eight small-path condition must both hold in both seeds.
    candidate={a:bool(all(summary['candidate_by_seed'][str(seed)][a] for seed in [20260925,20260926])) for a in ['control','junction','task_only','junction_task_only']}
    pdata={'coverage':coverage,'coverageCounts':cover_counts,'fullCoverage':len(l1),'zero':zero,'paths':paths,'labels':LABELS,
           'history':{a:g.sort_values('step')[['step','all_p95_mm','loss']].values.tolist() for a,g in hist[hist.arm.isin(ARMS)].groupby('arm',sort=True)},
           'paired':paired[paired.subset=='all'][['arm_key','gt50','fixed_control_gt50','new_gt50']].to_dict('records'),
           'near':paired[paired.subset=='all'][['arm_key','small_paths_retained']].to_dict('records'),'nearMax':near_max}
    inputs=[RUN/'summary.json',RUN/'model_lock.json',RUN/'completion_manifest.json',RUN/'01_inputs/targets.parquet',RUN/'01_inputs/junction_train_pool.parquet',RUN/'03_training/validation_history.csv',RUN/'04_evaluation/metrics.csv',RUN/'05_diagnosis/paired_metrics.csv',VERIFY,WORKTREE/'configs/bacra_retry22_junction.yaml',WORKTREE/'docs/protocols/41-BACRA-retry22-交界采样与标签牵制对照协议.md',TEMPLATE]
    body=f'''<header><div class="k" style="color:#ddecff">BACRA retry22 · 交界采样 × β 标签牵制 · evidence cutoff: {OBSERVED}</div><h1>BACRA retry22：交界采样与 β 标签牵制的八模型配对诊断</h1><p class="sub">在同一 L1、split、35 条路径与 3000 off-grid 分母上，比较 4 个 arm × 2 seed。页面只解释完成的冻结产物；不重新训练、不增加监督，也不把可视化当作 formal Gate。</p><div class="meta"><div><div class="k" style="color:#ddecff">状态</div><div class="v">completed · diagnostic_only</div></div><div><div class="k" style="color:#ddecff">监督 / 新增</div><div class="v">74,696 / 0</div></div><div><div class="k" style="color:#ddecff">固定路径</div><div class="v">35 + zero reference</div></div><div><div class="k" style="color:#ddecff">优化更新</div><div class="v">77,000</div></div></div></header>
<section><h2>结论与证据边界</h2><div class="notice"><b>没有 arm 通过两个 seed 的预登记联合候选判据。</b> 该判据要求每个 seed 的完整 test <code>&gt;50 mm</code> 严重点至少减半，并且八条近轴小路径全部 ≤3 mm。control、junction、task only、junction + task only 的 <code>candidate_both_seeds</code> 均为 <code>false</code>；因此本页不支持“全域修复”、数据扩充成功或部署结论。</div><div class="grid"><div class="card"><div class="k">source / binding fixed point</div><code>{source}</code><br><code>{binding}</code><p>冻结 run elapsed {summary['elapsed_seconds']:.4f} s，8 个锁定 .keras 模型。</p></div><div class="card"><div class="k">完备性与完整性</div><p>completion manifest 声明的 {len(artifacts)} 个 artifact 已逐项 SHA-256 匹配。这核验字节完整性，不改变 scientific Gate。</p></div><div class="card"><div class="k">唯一干预</div><p>junction arm 用 train 内 ±60 mm 接缝 pool 替换 128 个 uniform batch 位；task-only 移除 β 项。没有新几何厚度或新 Teacher 监督。</p></div><div class="card"><div class="k">严格限制</div><p>两 seed 是配对复现，不构成统计显著性样本；已知失败位置和固定 test 被用来定位，不能称新盲测。</p></div></div></section>
<section><h2>三维数据覆盖、物理零位与 lineage</h2><p class="legend">L1 完整分母 74,696 行（train 58,629 / validation 7,795 / test 8,272），显示 {len(coverage):,} 个按 <code>split_role × domain_class</code> 分层、<code>target_id</code> 稳定排序后等距抽样的真实 XYZ 点。坐标是机器人基座物理 frame、单位 mm，未排版平移。存储域 <code>retry18_outer</code> 仅在显示中映射为 outer。灰色中心线与红色端点是完整 beta=0 机器人参照。</p><div class="toolbar"><button id="coverageGlobal">全局物理视图</button><button id="coverageLocal">接缝局部投影（不改坐标）</button><button id="theme">切换明暗主题</button></div><div id="coverage" class="plot" aria-label="三维 L1 覆盖与零位"></div></section>
<section><h2>训练、锁定与八模型比较</h2><div class="grid"><div class="card"><div class="k">锁定规则</div><p>每 seed beta-only warmup 500 步后，从同一模型、Adam 和采样随机状态分叉；固定 final step，无 early stop、无 test 选模。</p></div><div class="card"><div class="k">模型呈现</div><p>本次 artifact 只有 raw Student FK 输出；页面不以 DLS 或任何后处理曲线替代 raw 结果。默认并列 control / junction 的 seed 20260925。</p></div></div><div id="history" class="plot small" aria-label="验证 P95 训练轨迹"></div></section>
<section><h2>35 条路径：Target、Teacher 与 raw Student</h2><p>所有场景在同一物理 mm frame 显示 Target、可验证 Teacher、两个 raw Student 和完整 beta=0 机器人。Teacher 仅来自 retry20/Q31 原始路径保存的几何 target 输出；16 条 q33 matched-control 路径没有本地 Teacher Cartesian 证据，选择后会清楚显示缺口，绝不推断或移植旧模型结果。</p><div class="toolbar"><label>路径<select id="trajectory"></select></label><label>模型 A<select id="modelA"></select></label><label>模型 B<select id="modelB"></select></label><button id="trackingGlobal">全局物理视图</button><button id="trackingLocal">路径局部投影（仍含完整 zero）</button></div><div id="trackNote" class="notice"></div><div id="tracking" class="plot" aria-label="路径三维 target teacher raw student"></div><div id="pathError" class="plot small" aria-label="路径逐点 FK 误差"></div></section>
<section><h2>完整 test 指标与配对严重失败迁移</h2><p>以下是 8,272 点完整 on-grid test；阈值和分母来自冻结 <code>metrics.csv</code>。配对 “fixed/new” 均相对同 seed control 的逐点 >50 mm 关系，不是只看汇总百分位的推断。</p>{table(['模型','n','P50 mm','P95 mm','P99 mm','max mm','≤3 mm','≤10 mm'],metric_rows)}<br>{table(['模型','test >50 mm','原 control 严重点已修复','新增 >50 mm','8 小路径全 ≤3 mm'],paired_rows)}<div id="migration" class="plot small" aria-label="严重失败迁移"></div></section>
<section><h2>因果 factorial 读法与边界空间诊断</h2><div class="grid"><div class="card"><div class="k">junction 采样效应</div><p>seed 20260925/20260926：control 的 >50 mm 为 {int(paired_all.loc['control_20260925'].gt50)}/{int(paired_all.loc['control_20260926'].gt50)}，junction 为 {int(paired_all.loc['junction_20260925'].gt50)}/{int(paired_all.loc['junction_20260926'].gt50)}。junction 修复各 seed 原 control 的全部严重失败，但同时分别引入 {int(paired_all.loc['junction_20260925'].new_gt50)}/{int(paired_all.loc['junction_20260926'].new_gt50)} 个新的严重失败。</p></div><div class="card"><div class="k">去 β 项与交互</div><p>task only 为 {int(paired_all.loc['task_only_20260925'].gt50)}/{int(paired_all.loc['task_only_20260926'].gt50)}，junction + task only 为 {int(paired_all.loc['junction_task_only_20260925'].gt50)}/{int(paired_all.loc['junction_task_only_20260926'].gt50)}。四臂与两个 seed 的差异应按采样效应、去 β 项效应及交互阅读；它不能单独证明 Teacher 标签冲突或网络容量的唯一机制。</p></div><div class="card"><div class="k">近轴路径限制</div><p>八条小路径的最大 error（seed 20260925/20260926）为 control {n(near_max['control_20260925'])}/{n(near_max['control_20260926'])}、junction {n(near_max['junction_20260925'])}/{n(near_max['junction_20260926'])}、task only {n(near_max['task_only_20260925'])}/{n(near_max['task_only_20260926'])}、combined {n(near_max['junction_task_only_20260925'])}/{n(near_max['junction_task_only_20260926'])} mm。该条件阻止任一 arm 成为两 seed 候选。</p></div><div class="card"><div class="k">空间诊断含义</div><p>接缝带是从既有诊断确定的 ±60 mm 干预区，混合邻域和接缝带的结果可定位失败迁移，不是完整连续空间的覆盖证明。</p></div></div><div id="near" class="plot small" aria-label="近轴路径最大误差"></div></section>
<section><h2>全部指标、溯源与交付</h2><details><summary>展开 metrics.csv 的 416 行完整记录</summary>{metrics_html}</details><p>输出目录：<code>{OUT}</code><br>HTML、模板快照、生成器、输入 hash、输出 hash 和浏览器验证记录均由 <code>report_manifest.json</code> 关联。报告的 hash/结构/浏览器检查不等于 formal Gate、下游授权、全域替换或真实闭环控制证明。</p></section>'''
    script=r'''const $=id=>document.getElementById(id);let state={local:false,coverageLocal:false};
const colors=()=>{const p=ReportPlot.palette();return {target:p.target,teacher:p.teacher,control:p.l1raw,junction:p.l1dls,task:p.l0,jtask:p.l0raw,robot:p.robot,tip:p.tip}};
function p3(v,name,color,dash='solid',width=4){return {type:'scatter3d',mode:'lines',name,x:v.map(p=>p?.[0]??null),y:v.map(p=>p?.[1]??null),z:v.map(p=>p?.[2]??null),line:{color,width,dash},connectgaps:false}}
function robot(){let c=colors();return [p3(D.zero,'机器人 beta=0',c.robot,'solid',5),{type:'scatter3d',mode:'markers',name:'零位末端',x:[D.zero.at(-1)[0]],y:[D.zero.at(-1)[1]],z:[D.zero.at(-1)[2]],marker:{color:c.tip,size:7,symbol:'diamond'}}]}
function range(groups){return [0,1,2].map(k=>{let a=groups.flat().map(p=>p?.[k]).filter(Number.isFinite),lo=Math.min(...a),top=Math.max(...a),pad=Math.max(100,(top-lo)*.30);return[lo-pad,top+pad]})}
function layout3(title,r){let c=colors();return {title:{text:title},margin:{l:0,r:0,b:0,t:42},paper_bgcolor:'transparent',font:{color:c.target},scene:{aspectmode:'data',camera:{projection:{type:'orthographic'},eye:{x:1.9,y:1.9,z:1.25}},xaxis:{title:'X (mm)',range:r[0]},yaxis:{title:'Y (mm)',range:r[1]},zaxis:{title:'Z (mm)',range:r[2]}}}}
async function coverage(){let c=colors(), pts=D.coverage, tr=[];for(let split of ['train','validation','test'])for(let domain of ['primary','outer']){let a=pts.filter(p=>p[3]===split&&p[4]===domain);if(a.length)tr.push({type:'scatter3d',mode:'markers',name:split+' / '+domain,x:a.map(p=>p[0]),y:a.map(p=>p[1]),z:a.map(p=>p[2]),marker:{color:split==='train'?c.control:split==='validation'?c.teacher:c.task,size:2,opacity:domain==='primary'?.62:.36,symbol:domain==='primary'?'circle':'diamond'}})}tr.push(...robot());let r=range([pts.map(p=>p.slice(0,3)),D.zero]);let l=layout3(state.coverageLocal?'接缝带局部投影（物理 mm，未改坐标）':'完整 L1 覆盖（物理 mm）',r);l.scene.camera=state.coverageLocal?{projection:{type:'orthographic'},eye:{x:1.3,y:1,z:.7}}:{projection:{type:'orthographic'},eye:{x:.2,y:-2,z:.65}};await ReportPlot.react('coverage',tr,l,{responsive:true,displaylogo:false})}
function current(){return D.paths[$('trajectory').selectedIndex]}
async function tracking(){let t=current(),a=$('modelA').value,b=$('modelB').value,c=colors();$('trackNote').innerHTML=(t.teacher?`Teacher 来自冻结 retry20/Q31 原始路径输出（${t.count} 点）。`:`证据缺口：这条 q33 matched-control 路径没有本地可验证 Teacher Cartesian 输出；页面只显示 Target 与 raw Student。`)+` 当前 <b>${t.id}</b>，${t.count} 点。`;let tr=[p3(t.target,'Target',c.target,'dash',4)];if(t.teacher)tr.push(p3(t.teacher,'Teacher',c.teacher,'solid',7));tr.push(p3(t.models[a].xyz,D.labels[a],c.control,'solid',4),p3(t.models[b].xyz,D.labels[b],c.junction,'dot',4),...robot());let all=[t.target,t.models[a].xyz,t.models[b].xyz,D.zero];if(t.teacher)all.push(t.teacher);let l=layout3(state.local?'路径局部投影（完整 zero 保留）':'路径全局物理视图（完整路径与 zero）',range(all));if(state.local)l.scene.camera={projection:{type:'orthographic'},eye:{x:1.15,y:1.1,z:.72}};await ReportPlot.react('tracking',tr,l,{responsive:true,displaylogo:false});await ReportPlot.react('pathError',[{type:'scatter',mode:'lines',name:D.labels[a]+' raw error',x:[...Array(t.count).keys()],y:t.models[a].error,line:{color:c.control,width:3}},{type:'scatter',mode:'lines',name:D.labels[b]+' raw error',x:[...Array(t.count).keys()],y:t.models[b].error,line:{color:c.junction,width:3}}],{title:'逐 waypoint Cartesian FK error (mm)',paper_bgcolor:'transparent',font:{color:c.target},xaxis:{title:'waypoint'},yaxis:{title:'mm'}},{responsive:true,displaylogo:false})}
async function history(){let c=colors(),cs=[c.control,c.junction,c.task,c.jtask];let tr=Object.entries(D.history).map(([a,v],i)=>({type:'scatter',mode:'lines',name:D.labels[a],x:v.map(x=>x[0]),y:v.map(x=>x[1]),line:{color:cs[i%4],width:a.includes('20260925')?3:2,dash:a.includes('20260925')?'solid':'dot'}}));await ReportPlot.react('history',tr,{title:'validation FK P95 (mm) 随训练步骤',paper_bgcolor:'transparent',font:{color:c.target},xaxis:{title:'step'},yaxis:{title:'P95 mm'}},{responsive:true,displaylogo:false})}
async function migration(){let c=colors(),r=D.paired;await ReportPlot.react('migration',[{type:'bar',name:'已修复原 control >50',x:r.map(x=>D.labels[x.arm_key]),y:r.map(x=>x.fixed_control_gt50),marker:{color:c.junction}},{type:'bar',name:'新增 >50',x:r.map(x=>D.labels[x.arm_key]),y:r.map(x=>x.new_gt50),marker:{color:c.target}}],{title:'相对同 seed control 的严重失败迁移（完整 test）',barmode:'group',paper_bgcolor:'transparent',font:{color:c.target},yaxis:{title:'points'}},{responsive:true,displaylogo:false})}
async function near(){let c=colors(), keys=['control','junction','task_only','junction_task_only'], labels=['control','junction','task only','junction + task only'];let rows=keys.map((k,i)=>[labels[i],D.nearMax[k+'_20260925'],D.nearMax[k+'_20260926']]);await ReportPlot.react('near',[{type:'bar',name:'seed 20260925',x:rows.map(x=>x[0]),y:rows.map(x=>x[1]),marker:{color:c.control}},{type:'bar',name:'seed 20260926',x:rows.map(x=>x[0]),y:rows.map(x=>x[2]),marker:{color:c.junction}}],{title:'八条近轴小路径的路径内最大 FK error (mm)',barmode:'group',paper_bgcolor:'transparent',font:{color:c.target},yaxis:{title:'max mm',rangemode:'tozero'},shapes:[{type:'line',x0:-.5,x1:3.5,y0:3,y1:3,line:{color:c.target,dash:'dash'}}]},{responsive:true,displaylogo:false})}
const vis={coverage:false,tracking:false};function release(id){let p=$(id);if(p.data)Plotly.purge(p);$(id+'-legend')?.replaceChildren()}const ob=new IntersectionObserver(es=>es.forEach(e=>{vis[e.target.id]=e.isIntersecting;if(e.isIntersecting)(e.target.id==='coverage'?coverage:tracking)();else release(e.target.id)}),{threshold:.08});ob.observe($('coverage'));ob.observe($('tracking'));function render(){history();migration();near();if(vis.coverage)coverage();if(vis.tracking)tracking()}function theme(){document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark';render()}for(let p of D.paths)$('trajectory').add(new Option(p.id,p.id));for(let [k,v] of Object.entries(D.labels)){$('modelA').add(new Option(v,k));$('modelB').add(new Option(v,k))}$('modelA').value='control_20260925';$('modelB').value='junction_20260925';ReportPlot.pathNavigation('trajectory');$('trajectory').onchange=tracking;['modelA','modelB'].forEach(id=>$(id).onchange=tracking);$('trackingGlobal').onclick=()=>{state.local=false;tracking()};$('trackingLocal').onclick=()=>{state.local=true;tracking()};$('coverageGlobal').onclick=()=>{state.coverageLocal=false;coverage()};$('coverageLocal').onclick=()=>{state.coverageLocal=true;coverage()};$('theme').onclick=theme;render();'''
    slots={'TITLE':esc('BACRA retry22 junction factorial 诊断报告'),'BODY':body,'PLOTLY':PLOTLY.read_text(encoding='utf-8'),'DATA':json.dumps(pdata,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</','<\\/'),'SCRIPT':script}
    template=TEMPLATE.read_text(encoding='utf-8'); names=set(re.findall(r'@@([A-Z]+)@@',template))
    if names!=set(slots): raise ValueError(f'template slots mismatch: {names}')
    OUT.mkdir(parents=True,exist_ok=True); report=OUT/NAME; report.write_text(re.sub(r'@@([A-Z]+)@@',lambda m:slots[m.group(1)],template),encoding='utf-8')
    shutil.copy2(TEMPLATE,OUT/'report_template.html'); shutil.copy2(Path(__file__),OUT/Path(__file__).name)
    evidence={'observed_at':OBSERVED,'attempt_root':str(RUN),'report_worktree':str(WORKTREE),'scientific_source_fixed_point':source,'binding_fixed_point':binding,'completion_manifest_artifacts_verified':len(artifacts),'claim_boundary':{'diagnostic_only':True,'formal_authorized':False,'global_replacement_authorized':False,'new_supervision_authorized':False},'coverage':{'full_rows':len(l1),'display_rows':len(coverage),'method':'stable target_id sorted, proportional split_role x domain_class allocation, equally-spaced row selection'},'candidate_both_seeds':candidate,'references':[ref(p) for p in inputs]+zero_refs}
    (OUT/'evidence_map.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest={'report_name':NAME,'report_path':str(report),'report_sha256':sha(report),'template_path':str(TEMPLATE),'template_sha256':sha(TEMPLATE),'generator_source':str(Path(__file__)),'generator_copy_sha256':sha(OUT/Path(__file__).name),'scientific_source_fixed_point':source,'binding_fixed_point':binding,'isolated_public_code_snapshot_sha':'696f3f70e16e897f036da921ab391166aa24d85c','isolated_public_inventory_file_count':642,'isolated_public_inventory_sha256':'db9a9f55e9e4042d82dda6fb77931ea0e7ece1b96788d6afc463fa88ec3cc5f7','isolated_public_snapshot_published':False,'diagnostic_only':True,'formal_authorized':False,'global_replacement_authorized':False,'inputs':[ref(p) for p in inputs],'sealed_outputs_verified_count':len(artifacts),'sealed_output_hashes':artifacts,'outputs':[ref(report),ref(OUT/'report_template.html'),ref(OUT/Path(__file__).name),ref(OUT/'evidence_map.json')],'browser_validation':'pending'}
    (OUT/'report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(report); print(sha(report))

def record_browser() -> None:
    v=OUT/'browser_validation.json'; m=OUT/'report_manifest.json'; report=OUT/NAME
    b=j(v); manifest=j(m)
    if b.get('report_sha256')!=sha(report): raise ValueError('browser validation targets different HTML')
    manifest['browser_validation']={'path':str(v),'sha256':sha(v),'passed':bool(b.get('passed')),'engine':b.get('engine'),'report_sha256':b.get('report_sha256'),'exceptions':len(b.get('exceptions',[])),'console_errors':len(b.get('consoleErrors',[]))}
    manifest['outputs'].append(ref(v)); m.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt-root',type=Path,default=RUN);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--record-browser',action='store_true');a=p.parse_args();RUN=a.attempt_root.resolve();OUT=a.output.resolve();build()
    if a.record_browser: record_browser()
