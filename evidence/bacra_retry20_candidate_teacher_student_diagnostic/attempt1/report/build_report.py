#!/usr/bin/env python3
"""Build the self-contained retry20 candidate-layout diagnostic report.

This program reads sealed attempt artifacts and writes only report derivatives in
its own directory.  It neither retrains nor solves trajectories and never writes
inside the attempt root or its execution worktree.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path('/mnt/ML_projects/quasi_exp')
RUN = ROOT / 'runs/bacra_retry20_candidate_diagnostic_attempt1'
OUT = Path(__file__).resolve().parent
WORKTREE = ROOT / '.worktrees/retry20-candidate-diagnostic-attempt1'
PLOTLY = Path('/mnt/ML_projects/conda_envs/rdagent4qlib/lib/python3.10/site-packages/plotly/package_data/plotly.min.js')
HTML_NAME = 'retry20_candidate_diagnostic_advisor_dataset_report.html'
SOURCE = '5444eeba64aa1e41d4a40dd1cb9ff51733250440'
BINDING = '855051b94c61dc2de045f9e66c45e2c540d3395e'
CODE_SNAPSHOT = '04866720b9bd7ebad758509fa22c605a3c59208d'
OBSERVATION = '2026-09-10 Asia/Shanghai；封存 attempt 完成后读取'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def reference(path: Path, kind: str, base: Path | None = RUN) -> dict[str, Any]:
    return {
        'kind': kind,
        'path': str(path.relative_to(base)) if base and path.is_relative_to(base) else str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256(path),
    }


def finite(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def clean_xyz(frame: pd.DataFrame, prefix: str = '') -> list[list[float | None]]:
    cols = [f'{prefix}{axis}_m' for axis in 'xyz']
    return [[finite(v * 1000) for v in row] for row in frame[cols].to_numpy(float)]


def deterministic_sample(frame: pd.DataFrame, limit: int = 12000) -> tuple[list[list[Any]], dict[str, int]]:
    groups = list(frame.groupby('domain_class', sort=True, dropna=False))
    parts: list[pd.DataFrame] = []
    for _, part in groups:
        count = max(1, round(limit * len(part) / len(frame)))
        indices = np.linspace(0, len(part) - 1, min(count, len(part)), dtype=int)
        parts.append(part.sort_values('target_id', kind='stable').iloc[indices])
    selected = pd.concat(parts, ignore_index=True)
    rows = [
        [round(float(row.x_m) * 1000, 3), round(float(row.y_m) * 1000, 3), round(float(row.z_m) * 1000, 3), str(row.domain_class)]
        for row in selected[['x_m', 'y_m', 'z_m', 'domain_class']].itertuples(index=False)
    ]
    return rows, {str(key): int(value) for key, value in frame.domain_class.value_counts(dropna=False).items()}


def zero_robot() -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Replay beta=0 using the bound FK/config only to draw a static reference."""
    sys.path.insert(0, str(WORKTREE / 'src'))
    old_cwd = Path.cwd()
    try:
        os.chdir(WORKTREE)
        from quasi_exp.io import load_config, load_robot_inputs
        from quasi_exp.model.kinematics import forward_kinematics
        from quasi_exp.model.sampling import beta_to_theta

        config_path = WORKTREE / 'configs/robot_rods_only_standard_100k.yaml'
        config = load_config(config_path)
        robot = load_robot_inputs(config)
        endpoint, transforms = forward_kinematics(
            beta_to_theta(np.zeros(6)), robot.lengths_m, robot.p_end_local_m,
            theta_sign=float(config['kinematics']['theta_sign']),
        )
        body = (np.vstack((transforms[:, :3, 3], endpoint)) * 1000).round(4).tolist()
        files = [config_path, WORKTREE / 'src/quasi_exp/io/robot_inputs.py', WORKTREE / 'src/quasi_exp/model/kinematics.py', WORKTREE / 'src/quasi_exp/model/sampling.py']
        files.extend((WORKTREE / value) for value in config['paths'].values())
        return body, [reference(item, 'binding_zero_pose_input', None) for item in files]
    finally:
        os.chdir(old_cwd)


def html_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = ''.join(f'<th>{html.escape(str(value))}</th>' for value in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in row) + '</tr>' for row in rows)
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def num(value: Any, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return '—'
    return f'{float(value):.{digits}f}'


def rate(value: Any) -> str:
    if value is None or not np.isfinite(float(value)):
        return '—'
    return f'{100 * float(value):.2f}%'


def verify_attempt() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completion = load_json(RUN / 'completion_manifest.json')['artifacts']
    bad = []
    for item in completion:
        path = RUN / item['path']
        actual = sha256(path) if path.is_file() else 'MISSING'
        if actual != item['sha256']:
            bad.append({'path': item['path'], 'expected': item['sha256'], 'actual': actual})
    inputs = load_json(RUN / 'input_manifest.json')['inputs']
    input_bad = []
    for item in inputs:
        path = Path(item['path'])
        actual = sha256(path) if path.is_file() else 'MISSING'
        if actual != item['sha256']:
            input_bad.append({'path': item['path'], 'expected': item['sha256'], 'actual': actual})
    if bad or input_bad:
        raise RuntimeError(f'sealed artifact verification failed: output={bad}, input={input_bad}')
    return completion, inputs


def build() -> None:
    completion, inputs = verify_attempt()
    identity, summary = load_json(RUN / 'run_identity.json'), load_json(RUN / 'summary.json')
    assert identity['scientific_source_fixed_point'] == SOURCE
    assert identity['binding_fixed_point'] == BINDING
    assert summary['operational_status'] == 'complete'
    assert summary['diagnostic_only'] and not summary['formal_authorized']

    coverage = load_json(RUN / '02_dataset/coverage_audit.json')
    dataset = load_json(RUN / '02_dataset/dataset_summary.json')
    gate = load_json(RUN / '00_objective_feasibility/gate.json')
    contract = load_json(RUN / '03_student/training_contract.json')
    model_lock = load_json(RUN / '03_student/model_lock.json')
    l0 = pd.read_parquet(RUN / '02_dataset/L0_supervision.parquet')
    l1 = pd.read_parquet(RUN / '02_dataset/L1_supervision.parquet')
    teacher = pd.read_parquet(RUN / '01_relocation_teacher/teacher.parquet')
    edge = pd.read_parquet(RUN / '02_dataset/teacher_edge_audit.parquet')
    history = pd.read_parquet(RUN / '03_student/training_history.parquet')
    metrics = pd.read_parquet(RUN / '05_evaluation/metrics.parquet')
    waypoint = pd.read_parquet(RUN / '05_evaluation/waypoint_evaluation.parquet')
    path_registry = pd.read_parquet(RUN / '00_objective_feasibility/path_registry.parquet')
    l0_points, l0_domains = deterministic_sample(l0)
    l1_points, l1_domains = deterministic_sample(l1)
    body, zero_inputs = zero_robot()

    tracks: list[dict[str, Any]] = []
    for trajectory_id, part in waypoint.groupby('trajectory_id', sort=True):
        part = part.sort_values('waypoint_index', kind='stable')
        metric_rows = metrics[metrics['case'] == trajectory_id].set_index('output')
        support = {
            'L0': {key: float(part['L0_nearest_support_mm'].quantile(q)) for key, q in [('p50', .5), ('p95', .95), ('max', 1.0)]},
            'L1': {key: float(part['L1_nearest_support_mm'].quantile(q)) for key, q in [('p50', .5), ('p95', .95), ('max', 1.0)]},
        }
        track = {
            'id': str(trajectory_id), 'shape': str(part.shape_class.iloc[0]), 'count': int(len(part)),
            'closed': bool(metric_rows['closed_path'].iloc[0]), 'target': clean_xyz(part),
            'teacher': clean_xyz(part, 'Teacher_'), 'l0raw': clean_xyz(part, 'L0_raw_'),
            'l0dls': clean_xyz(part, 'L0_DLS2_'), 'l1raw': clean_xyz(part, 'L1_raw_'),
            'l1dls': clean_xyz(part, 'L1_DLS2_'), 'support': support,
            'teacherBeta': [[finite(value) for value in row] for row in part[[f'Teacher_beta{i}_rad' for i in range(1, 7)]].to_numpy(float)],
            'l1rawBeta': [[finite(value) for value in row] for row in part[[f'L1_raw_beta{i}_rad' for i in range(1, 7)]].to_numpy(float)],
            'metrics': {name: {key: (finite(value) if isinstance(value, (int, float, np.number)) else value) for key, value in row.to_dict().items()} for name, row in metric_rows.iterrows()},
        }
        tracks.append(track)
    assert len(tracks) == 16 and sum(track['count'] for track in tracks) == 2641

    # All output rows are page-visible in this static table, including the distinct 3000-point off-grid test.
    ordered = metrics.sort_values(['case', 'output'], kind='stable')
    metric_table = html_table(
        ['case', 'output', 'n/finite', 'P50 mm', 'P95 mm', 'max mm', '≤3 mm', '≤10 mm', 'bounds', 'β>7°', 'β step max°'],
        [[
            row['case'], row['output'], f"{int(row['count'])}/{int(row['finite_count'])}", num(row['fk_p50_mm']), num(row['fk_p95_mm']), num(row['fk_max_mm']),
            rate(row['within_3mm_rate']), rate(row['within_10mm_rate']), int(row['bounds_violation_count']), rate(row['raw_step_gt7_rate']), num(row['max_beta_step_deg']),
        ] for _, row in ordered.iterrows()],
    )
    support_table = html_table(
        ['path', 'class', 'points', 'L0 support P50/P95/max mm', 'L1 support P50/P95/max mm'],
        [[track['id'], track['shape'], track['count'],
          '/'.join(num(track['support']['L0'][key]) for key in ('p50', 'p95', 'max')),
          '/'.join(num(track['support']['L1'][key]) for key in ('p50', 'p95', 'max'))] for track in tracks],
    )
    final_history = history.sort_values(['model', 'step']).groupby('model').tail(1).set_index('model')
    edge_stats = {'p50': float(edge.weighted_gap_deg.quantile(.5)), 'p95': float(edge.weighted_gap_deg.quantile(.95)), 'max': float(edge.weighted_gap_deg.max()), 'raw_gt7': float((edge.raw_gap_deg > 7).mean())}
    teacher_stats = {'fast_gold': int(teacher.fast_gold.sum()), 'fallback': int((~teacher.fast_gold).sum()), 'least_squares': int((teacher.solver == 'bounded_least_squares').sum()), 'residual_p95': float(teacher.scalar_fk_residual_mm.quantile(.95)), 'residual_max': float(teacher.scalar_fk_residual_mm.max())}
    data = {
        'coverage': {'l0': l0_points, 'l1': l1_points, 'l0Domains': l0_domains, 'l1Domains': l1_domains, 'fullCount': len(l0)},
        'zero': {'body': body, 'tip': body[-1]}, 'tracks': tracks,
        'history': {model: group[['step', 'train_beta_loss', 'on_grid_fk_p95_mm', 'off_grid_fk_p95_mm']].values.tolist() for model, group in history.groupby('model', sort=True)},
        'edge': edge_stats, 'metrics': metric_table,
    }
    evidence_paths = [
        RUN / 'run_identity.json', RUN / 'summary.json', RUN / 'completion_manifest.json', RUN / 'input_manifest.json',
        RUN / '00_objective_feasibility/gate.json', RUN / '00_objective_feasibility/path_registry.parquet', RUN / '00_objective_feasibility/trajectory_registry.parquet',
        RUN / '01_relocation_teacher/teacher.parquet', RUN / '02_dataset/L0_supervision.parquet', RUN / '02_dataset/L1_supervision.parquet',
        RUN / '02_dataset/coverage_audit.json', RUN / '02_dataset/dataset_summary.json', RUN / '02_dataset/teacher_edge_audit.parquet',
        RUN / '03_student/training_history.parquet', RUN / '03_student/training_contract.json', RUN / '03_student/model_lock.json',
        RUN / '05_evaluation/metrics.parquet', RUN / '05_evaluation/waypoint_evaluation.parquet',
        WORKTREE / 'docs/protocols/38-BACRA-retry20-候选布局Teacher与Student诊断协议.md',
    ]
    evidence_map = {
        'observed_at': OBSERVATION, 'attempt_root': str(RUN), 'execution_worktree_read_only': str(WORKTREE),
        'scientific_source_fixed_point': SOURCE, 'binding_fixed_point': BINDING,
        'completion_manifest_artifacts_verified': len(completion), 'input_manifest_inputs_verified': len(inputs),
        'data': {'full_rows_per_layer': len(l0), 'coverage_display_rows_per_layer': len(l0_points), 'trajectory_count': 16, 'trajectory_waypoints': 2641, 'off_grid_test_count': 3000},
        'claim_boundary': summary['claim_boundary'], 'references': [reference(path, 'evidence_input', None) for path in evidence_paths] + zero_inputs,
    }
    (OUT / 'evidence_map.json').write_text(json.dumps(evidence_map, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    plotly = PLOTLY.read_text(encoding='utf-8')
    data_text = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    title = 'BACRA retry20 候选布局：Teacher / Student 诊断报告'
    html_doc = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--ink:#172235;--muted:#566275;--paper:#f5f7fa;--card:#fff;--line:#d6dde7;--accent:#1167b1;--warn:#a44718;--teacher:#e57f22;--raw:#1677d2;--dls:#14835f;--l0:#8768bd}}[data-theme=dark]{{--ink:#edf3fb;--muted:#b0bccd;--paper:#121a26;--card:#1b2635;--line:#38495d;--accent:#77baff;--warn:#ffad82;--teacher:#ffb05c;--raw:#6bb7ff;--dls:#59d5a3;--l0:#c5a9ff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.62 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}main{{max-width:1440px;margin:auto;padding:24px}}header{{padding:28px 32px;background:linear-gradient(125deg,#123a62,#196da8);color:white;border-radius:18px}}h1{{margin:0 0 8px;font-size:clamp(25px,4vw,42px);line-height:1.2}}h2{{margin:0 0 12px;font-size:25px}}h3{{margin:14px 0 8px;font-size:18px}}p{{margin:8px 0}}section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin:20px 0;box-shadow:0 4px 14px #0015}}.sub{{color:#ddecff;max-width:1040px}}.meta,.grid{{display:grid;gap:14px}}.meta{{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:18px}}.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.card{{border:1px solid var(--line);border-radius:10px;padding:14px;background:color-mix(in srgb,var(--card) 92%,var(--accent))}}.k{{font-size:12px;color:var(--muted)}}.v{{font-size:22px;font-weight:700}}.notice{{border-left:5px solid var(--warn);background:color-mix(in srgb,var(--warn) 10%,var(--card));padding:13px 16px;border-radius:7px}}.good{{border-left-color:var(--dls)}}.toolbar{{display:flex;flex-wrap:wrap;gap:9px;align-items:end;margin:10px 0 12px}}label{{display:grid;gap:3px;font-size:13px;font-weight:650}}select,button{{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:7px;padding:7px 9px}}button{{cursor:pointer;font-weight:700}}button:focus,select:focus,input:focus{{outline:3px solid color-mix(in srgb,var(--accent) 45%,transparent)}}.checks{{display:flex;gap:9px;flex-wrap:wrap;padding:7px 0}}.checks label{{display:flex;gap:5px;align-items:center;font-weight:500}}.plot{{width:100%;height:620px;border:1px solid var(--line);border-radius:10px;overflow:hidden}}.plot.small{{height:450px}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:9px}}table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}th{{background:color-mix(in srgb,var(--card) 80%,var(--accent));position:sticky;top:0}}td,th{{padding:7px 9px;text-align:left;border-bottom:1px solid var(--line)}}caption{{caption-side:bottom;text-align:left;padding:8px;color:var(--muted)}}code{{font-size:12px;word-break:break-all}}details{{border-top:1px solid var(--line);padding-top:10px;margin-top:12px}}summary{{cursor:pointer;font-weight:700}}.legend{{color:var(--muted);font-size:13px}}.path-table{{max-height:430px}}@media(max-width:760px){{main{{padding:12px}}header,section{{padding:17px}}.meta,.grid{{grid-template-columns:1fr}}.plot{{height:480px}}.plot.small{{height:380px}}.toolbar{{align-items:stretch}}select,button{{max-width:100%}}}}@media print{{body{{background:white}}main{{max-width:none;padding:0}}header{{border-radius:0;background:white;color:#111;border:1px solid #aaa}}section{{break-inside:avoid;box-shadow:none}}.toolbar,.checks{{display:none}}.plot{{height:430px}}}}
</style><style>html,body,main,section,header,.grid,.card,.plot{{min-width:0;max-width:100%}}html,body{{max-width:100%;overflow-x:hidden}}.table-wrap{{width:100%;max-width:100%;min-width:0;overflow-x:auto;contain:layout paint}}.path-table{{width:100%;max-width:100%;min-width:0}}</style></head><body><main><header><div class="k" style="color:#ddecff">候选替换后的独立诊断 · evidence cutoff: {OBSERVATION}</div><h1>{title}</h1><p class="sub">在机器人基座物理坐标系（mm）中展示候选布局的配对 L0/L1 数据、冻结的 16 条路径、anchored Teacher、固定 final-step Student 与逐点 DLS2 诊断。页面是对封存 evidence 的解释，不构成 scientific Gate 或部署结论。</p><div class="meta"><div><div class="k" style="color:#ddecff">运行状态</div><div class="v">complete · diagnostic only</div></div><div><div class="k" style="color:#ddecff">配对监督行</div><div class="v">74,696 / layer</div></div><div><div class="k" style="color:#ddecff">冻结路径</div><div class="v">16 · 2,641 points</div></div><div><div class="k" style="color:#ddecff">科学运行时间</div><div class="v">49.78 min</div></div></div></header>
<section><h2>结论边界与固定点</h2><div class="notice"><b>这不是成功的科学 Gate。</b> 原 preflight 几何 Gate 失败（maximum phase bin、volume coverage、service P95、axial non-regression、axial minimum）；该次运行仅因明确授权作为诊断。候选替换行的 root-connectivity 未重新认证，不能据此宣称 canonical full graph、full-workspace 覆盖、部署可用或动态闭环控制。</div><div class="grid"><div class="card"><div class="k">scientific source</div><code>{SOURCE}</code><div class="k">binding fixed point</div><code>{BINDING}</code></div><div class="card"><div class="k">诊断 Teacher</div><p>two-seed anchored continuation + bounded least-squares；没有 MST/ICM 全图重选。轨迹逐点成功只证明此处 Cartesian FK，不证明连续或控制稳定。</p></div><div class="card"><div class="k">固定 40 mm 圆</div><p><code>x=zero_x−100 mm</code>，y/z 半径 40 mm，121 个相位点。它是冻结 benchmark，不是通过搜索得到的 largest-found 圆，页面不进行新圆搜索或求解。</p></div><div class="card"><div class="k">DLS2</div><p>同模型 FK/Jacobian 下、最多两步的 pointwise 数值投影（damping .002、3° RMS cap、bounds）。其绿色结果不表示真实闭环稳定控制。</p></div></div></section>
<section><h2>候选替换与空间覆盖</h2><div class="grid"><div class="card"><div class="k">replacement labels</div><div class="v">24,099 / 24,099 accepted</div><p>Fast-Gold {teacher_stats['fast_gold']:,}；fallback {teacher_stats['fallback']:,}；其中 bounded least-squares {teacher_stats['least_squares']:,}。scalar FK residual P95 {num(teacher_stats['residual_p95'],9)} mm，max {num(teacher_stats['residual_max'],6)} mm。</p></div><div class="card"><div class="k">冻结 required-cell 分母</div><div class="v">{coverage['cell_count_coverage']*100:.4f}% cells · {coverage['volume_coverage']*100:.4f}% volume</div><p>{coverage['served_cell_count']:,}/{coverage['required_cell_count']:,} cells；service distance volume-weighted P95 {coverage['service_distance_volume_weighted_p95_mm']:.3f} mm。该 geometry coverage 不等于新标签 root-connectivity。</p></div><div class="card"><div class="k">局部邻接诊断，非 canonical graph</div><div class="v">weighted gap P95 {edge_stats['p95']:.5f}°</div><p>P50 {edge_stats['p50']:.5f}°，max {edge_stats['max']:.5f}°；raw gap &gt;7° rate {edge_stats['raw_gt7']*100:.4f}%。这不构成全图连续性通过。</p></div><div class="card"><div class="k">L0 / L1 层语义</div><p>L0 为原有坐标与标签；L1 为 candidate replacement 后坐标与 anchored Teacher 标签。两层均是 74,696 行的 paired dataset，包含 primary {dataset['primary_rows']:,} 和 outer {len(l0)-dataset['primary_rows']:,} 行；它们不是同义副本。</p></div></div>
<p class="legend">三维主视图显示每层 12,000 个按 <code>target_id</code> 等距、按 <code>domain_class</code> 比例分配的确定性样本；完整统计分母始终是每层 74,696 行。L0 为紫色圆/方点，L1 为青色菱形/空心菱形；primary 为实心，outer 为空心/低透明度。所有点和 beta=0 机器人都在同一机器人基座物理坐标系（mm）中，视图可旋转与缩放。</p><div class="toolbar"><button id="coverageGlobal" type="button">全局视图</button><button id="coverageLocal" type="button">候选观察视角（非裁剪）</button><button id="theme" type="button">切换明暗主题</button></div><div id="coverage" class="plot" tabindex="0" aria-label="L0/L1 数据集三维覆盖主视图"></div></section>
<section><h2>Student 训练、锁定和独立 off-grid</h2><div class="grid"><div class="card"><div class="k">训练合同</div><p>seed {contract['seed']}；{contract['optimizer_steps']} optimizer steps；{contract['selection']}；hidden {contract['architecture']}；lr {contract['learning_rate']}；相同初始权重、L0 normalization 与 minibatch row-index。没有 early-stop 或按轨迹挑模型。</p></div><div class="card"><div class="k">最终 validation（不是 3000 点 test）</div><p>L0 on-grid FK P95 {final_history.loc['L0','on_grid_fk_p95_mm']:.5f} mm，off-grid P95 {final_history.loc['L0','off_grid_fk_p95_mm']:.5f} mm；L1 on-grid {final_history.loc['L1','on_grid_fk_p95_mm']:.5f} mm，off-grid {final_history.loc['L1','off_grid_fk_p95_mm']:.5f} mm。</p></div><div class="card"><div class="k">模型锁</div><p>L0 <code>{model_lock['models']['L0']}</code><br>L1 <code>{model_lock['models']['L1']}</code></p></div><div class="card"><div class="k">独立 3000 点 off-grid test</div><p>L0 raw P95 46.66265 mm，L1 raw P95 46.73264 mm；L0 DLS2 P95 0.07611 mm，L1 DLS2 P95 0.07836 mm。raw 与 DLS2 是不同输出，不能以 corrected 结果替代 raw Student。</p></div></div><div id="loss" class="plot small" tabindex="0" aria-label="L0 L1 训练历史"></div></section>
<section><h2>冻结 16 条路径：全局 / 局部同物理坐标轨迹</h2><p>每一场景都包含完整 beta=0 机器人中心线和零位末端，且路径、Teacher、raw Student、DLS2 均使用同一个 base-coordinate mm frame。初始全局范围容纳 zero pose、目标与全部输出。所谓局部视图只改变观察范围/相机，仍保留零位和所有选中曲线，<b>不会以裁剪或平移隐藏 raw 的大误差。</b></p><div class="toolbar"><label>路径<select id="trajectory"></select></label><label>投影<select id="projection"><option value="3d">3D 可旋转</option><option value="yz">YZ 正视</option><option value="xy">XY 俯视</option><option value="xz">XZ 侧视</option></select></label><button id="trackingGlobal" type="button">重置全局</button><button id="trackingLocal" type="button">局部审阅</button></div><div class="checks" aria-label="轨迹层切换"><label><input id="showL0raw" type="checkbox"> L0 raw（紫）</label><label><input id="showL0dls" type="checkbox"> L0 DLS2（浅紫）</label><label><input id="showL1raw" type="checkbox" checked> L1 raw（蓝）</label><label><input id="showL1dls" type="checkbox" checked> L1 DLS2（绿）</label></div><p class="legend">Target = 深灰虚线；Teacher = 橙色粗线；raw Student = 蓝/紫色实线；DLS2 = 绿/浅紫色细线。线宽和线型只用于可见性，不是误差带，也未平移坐标。所有 raw 路径的 P95 都高于 10 mm。</p><div id="tracking" class="plot" tabindex="0" aria-label="路径三维跟踪比较"></div><div id="trackMetrics" class="card" style="margin-top:12px"></div><div id="beta" class="plot small" tabindex="0" aria-label="Teacher beta 相邻跳变"></div></section>
<section><h2>每条路径的样本支持距离</h2><p>距离仅表示到各监督层最近样本的欧氏距离；即使 Teacher IK 成功，路径仍可能在局部样本支持之外。fixed circle 的 L1 max 为 12.878 mm，spiral 的 L1 max 为 25.284 mm。</p><div class="path-table">{support_table}</div></section>
<section><h2>全部输出指标（封存 scalar FK 评估）</h2><p>表中有限预测分位数使用 <code>finite_count</code>；成功率使用全部 targets。固定圆和 15 条路径各有 Teacher、L0/L1 raw、L0/L1 DLS2；off-grid test 没有额外 canonical Teacher。只有这里的真实数值可用于判断各输出，曲线外观不能取代该表。</p>{metric_table}</section>
<section><h2>方法限制与可追溯性</h2><div class="notice"><p><b>需要同时保留的事实：</b>16 条轨迹上的 Teacher 均 accepted 且 Cartesian error ≤3 mm，但 beta 连续性并不好：fixed circle 的 Teacher max step 15.414°；一个 seam rectangle 达 29.684°；star 达 28.100°。因此低 Cartesian Teacher error 绝不表示平滑 Teacher、canonical field 或动态控制。</p></div><ul><li>原始 preflight geometry Gate FAILED 被保留，不由本报告、训练或浏览器渲染改变。</li><li>局部 edge audit 的 weighted P95 只针对该审计图；没有 canonical full graph claim。</li><li>DLS2 是逐 waypoint 修正，不能以其小误差宣称 raw Student 表现良好或具备闭环稳定性。</li><li>报告没有启动训练、重新 materialize dataset、搜索圆或重解任何路径；它只读取封存 output。</li></ul><details><summary>证据与验证绑定</summary><p>attempt root: <code>{RUN}</code></p><p>execution worktree（只读）: <code>{WORKTREE}</code></p><p>completion manifest: 65/65 artifacts SHA-256 verified by generator；input manifest: 7/7 inputs rehashed. 报告文件自身与浏览器验收记录的 hash 见同目录 <code>report_manifest.json</code>。报告验证与 artifact 完整性验证均不构成 scientific Gate。</p></details></section>
</main><script>{plotly}</script><script>const D={data_text};</script><script>
const $=id=>document.getElementById(id); const cfg={{responsive:true,displaylogo:false,modeBarButtonsToRemove:['toImage']}};
const state={{coverageLocal:false,trackingLocal:false,projection:'3d',dark:false}};let coverageVisible=false,trackingVisible=false;
const color={{teacher:'#e57f22',l1raw:'#1677d2',l1dls:'#14835f',l0raw:'#8768bd',l0dls:'#b8a2e0',target:'#142235',robot:'#687587'}};
const dark=()=>document.documentElement.dataset.theme==='dark'; const ink=()=>dark()?'#edf3fb':color.target;
const arr=p=>p&&p.length?p.map(x=>Array.isArray(x)?x:null):[];function line(name,pts,c,w,dash){{return {{type:'scatter3d',mode:'lines',name,x:pts.map(p=>p?.[0]??null),y:pts.map(p=>p?.[1]??null),z:pts.map(p=>p?.[2]??null),connectgaps:false,line:{{color:c,width:w,dash:dash||'solid'}},hovertemplate:name+'<br>x=%{{x:.2f}} mm<br>y=%{{y:.2f}} mm<br>z=%{{z:.2f}} mm<extra></extra>'}}}}
function robot(){{const p=D.zero.body;return [line('机器人 beta=0',p,color.robot,6),{{type:'scatter3d',mode:'markers',name:'零位末端',x:[D.zero.tip[0]],y:[D.zero.tip[1]],z:[D.zero.tip[2]],marker:{{size:5,color:'#f54f5f',symbol:'diamond'}},hovertemplate:'零位末端<br>x=%{{x:.2f}} mm<extra></extra>'}}]}}
function layout(title){{return {{title:{{text:title,font:{{color:ink(),size:17}}}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{color:ink()}},legend:{{orientation:'h',y:-.1}},margin:{{l:0,r:0,t:44,b:0}},scene:{{xaxis:{{title:'x (mm)'}},yaxis:{{title:'y (mm)'}},zaxis:{{title:'z (mm)' }},aspectmode:'data',bgcolor:'rgba(0,0,0,0)',camera:{{}}}}}}}}
function ranges(layout,points){{const flat=points.filter(p=>p&&p.slice(0,3).every(Number.isFinite)); if(!flat.length)return;[0,1,2].forEach(i=>{{const a=flat.map(p=>p[i]),min=Math.min(...a),max=Math.max(...a),pad=Math.max(10,(max-min)*.08);layout.scene[['xaxis','yaxis','zaxis'][i]].range=[min-pad,max+pad]}})}}
function coverage(){{if(!coverageVisible)return;const layer=(name,points,c,isL1)=>{{const by={{}};points.forEach(p=>(by[p[3]]??=[]).push(p));return Object.entries(by).map(([domain,rows])=>{{const primary=domain==='primary';return {{type:'scatter3d',mode:'markers',name:name+' · '+domain+'（显示抽样）',x:rows.map(p=>p[0]),y:rows.map(p=>p[1]),z:rows.map(p=>p[2]),marker:{{size:primary?2.4:2,color:c,opacity:primary?.58:.32,symbol:primary?(isL1?'diamond':'circle'):(isL1?'diamond-open':'square-open')}},hovertemplate:name+' · '+domain+'<br>x=%{{x:.2f}} mm<br>y=%{{y:.2f}} mm<br>z=%{{z:.2f}} mm<extra></extra>'}}}})}};let traces=[...layer('L0',D.coverage.l0,color.l0,false),...layer('L1',D.coverage.l1,'#118b94',true),...robot()];let l=layout(state.coverageLocal?'候选替换观察视角（同 frame、非裁剪）':'L0/L1 三维覆盖主视图 · 机器人基座 mm');ranges(l,[...D.coverage.l0,...D.coverage.l1,...D.zero.body]);if(state.coverageLocal)l.scene.camera.eye={{x:1.7,y:1.4,z:.8}};Plotly.react('coverage',traces,l,cfg)}}
function fillTracks(){{const s=$('trajectory'),old=s.value;s.innerHTML=D.tracks.map(t=>'<option value="'+t.id+'">'+t.id+' · '+t.shape+' · '+t.count+' points</option>').join('');s.value=D.tracks.some(t=>t.id===old)?old:'fixed_circle_r40_u100'}}function current(){{return D.tracks.find(t=>t.id===$('trajectory').value)}}function toggled(id){{return $(id).checked}}
function tracking(){{const t=current();if(!t||!trackingVisible)return;let traces=[line('Teacher',t.teacher,color.teacher,10),line('Target',t.target,ink(),3,'dash')];if(toggled('showL1raw'))traces.push(line('L1 raw Student',t.l1raw,color.l1raw,5));if(toggled('showL1dls'))traces.push(line('L1 DLS2',t.l1dls,color.l1dls,3));if(toggled('showL0raw'))traces.push(line('L0 raw Student',t.l0raw,color.l0raw,4));if(toggled('showL0dls'))traces.push(line('L0 DLS2',t.l0dls,color.l0dls,2));traces.push(...robot());let l=layout((state.trackingLocal?'局部审阅（同坐标、无裁剪）':'全局视图（零位 + 全部输出；手动显示比例）')+' · '+t.id);let pts=[...t.target,...t.teacher,...t.l1raw,...t.l1dls,...t.l0raw,...t.l0dls,...D.zero.body];ranges(l,pts);l.scene.aspectmode='manual';l.scene.aspectratio={{x:1.5,y:1,z:1}};const eyes={{yz:{{x:2,y:0,z:0}},xy:{{x:0,y:0,z:2}},xz:{{x:0,y:2,z:0}}}};if(state.projection!=='3d')l.scene.camera.eye=eyes[state.projection];else l.scene.camera.eye=state.trackingLocal?{{x:1.65,y:.15,z:.10}}:{{x:2.2,y:.08,z:.08}};Plotly.react('tracking',traces,l,cfg);trackMetrics(t);beta(t)}}
function n(v,d=3){{return v==null||!Number.isFinite(v)?'—':Number(v).toFixed(d)}}function pct(v){{return v==null||!Number.isFinite(v)?'—':(100*v).toFixed(2)+'%'}}
function trackMetrics(t){{let h='<h3>'+t.id+' · '+t.shape+' · '+t.count+' points '+(t.closed?'（闭合）':'（开放）')+'</h3><p>nearest support (mm): L0 P50/P95/max '+n(t.support.L0.p50)+' / '+n(t.support.L0.p95)+' / '+n(t.support.L0.max)+'；L1 '+n(t.support.L1.p50)+' / '+n(t.support.L1.p95)+' / '+n(t.support.L1.max)+'. 支持距离不是连续可达性证明。</p><div class="table-wrap"><table><thead><tr><th>输出</th><th>有限/总点</th><th>P50 mm</th><th>P95 mm</th><th>max mm</th><th>≤3 mm</th><th>≤10 mm</th><th>bounds</th><th>β&gt;7°</th><th>β max step°</th></tr></thead><tbody>';['Teacher','L0_raw','L0_DLS2','L1_raw','L1_DLS2'].forEach(name=>{{const m=t.metrics[name];h+='<tr><td>'+name+'</td><td>'+m.finite_count+'/'+m.count+'</td><td>'+n(m.fk_p50_mm)+'</td><td>'+n(m.fk_p95_mm)+'</td><td>'+n(m.fk_max_mm)+'</td><td>'+pct(m.within_3mm_rate)+'</td><td>'+pct(m.within_10mm_rate)+'</td><td>'+m.bounds_violation_count+'</td><td>'+pct(m.raw_step_gt7_rate)+'</td><td>'+n(m.max_beta_step_deg)+'</td></tr>'}});$('trackMetrics').innerHTML=h+'</tbody></table></div>'}}
function maxJointSteps(rows,closed){{const values=[];const n=rows.length;for(let i=1;i<n+(closed?1:0);i++){{const a=rows[(i-1)%n],b=rows[i%n];values.push(a&&b&&a.every(Number.isFinite)&&b.every(Number.isFinite)?Math.max(...a.map((v,j)=>Math.abs((b[j]-v)*180/Math.PI))):null)}}return values}}
function beta(t){{const teacher=maxJointSteps(t.teacherBeta,t.closed),raw=maxJointSteps(t.l1rawBeta,t.closed),x=teacher.map((_,i)=>i+1);const series=[{{type:'scatter',mode:'lines+markers',name:'Teacher max |Δβ|',x,y:teacher,line:{{color:color.teacher,width:3}},marker:{{size:4}},hovertemplate:'Teacher transition %{{x}}<br>max |Δβ|=%{{y:.3f}}°<extra></extra>'}},{{type:'scatter',mode:'lines+markers',name:'L1 raw max |Δβ|',x,y:raw,line:{{color:color.l1raw,width:2}},marker:{{size:3}},hovertemplate:'L1 raw transition %{{x}}<br>max |Δβ|=%{{y:.3f}}°<extra></extra>'}},{{type:'scatter',mode:'lines',name:'7° reference',x:[x[0],x[x.length-1]],y:[7,7],line:{{color:'#bb4d35',width:2,dash:'dash'}},hovertemplate:'7° reference<extra></extra>'}}];Plotly.react('beta',series,{{title:{{text:'逐相邻 waypoint 的实际最大单关节步长（含闭合边时最后一条）',font:{{color:ink()}}}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{color:ink()}},xaxis:{{title:'transition index'}},yaxis:{{title:'max |Δbeta| (°)',rangemode:'tozero'}},legend:{{orientation:'h',y:-.2}},margin:{{l:60,r:25,t:44,b:80}}}},cfg)}}
function loss(){{const traces=[];const c={{L0:'#8768bd',L1:'#14835f'}};Object.entries(D.history).forEach(([model,rows])=>{{traces.push({{type:'scatter',mode:'lines+markers',name:model+' train beta loss',x:rows.map(r=>r[0]),y:rows.map(r=>r[1]),line:{{color:c[model],width:3}},marker:{{size:4}}}});traces.push({{type:'scatter',mode:'lines',name:model+' on-grid P95',x:rows.map(r=>r[0]),y:rows.map(r=>r[2]),yaxis:'y2',line:{{color:c[model],dash:'dash',width:2}}}});traces.push({{type:'scatter',mode:'lines',name:model+' off-grid P95',x:rows.map(r=>r[0]),y:rows.map(r=>r[3]),yaxis:'y2',line:{{color:c[model],dash:'dot',width:2}}}})}});Plotly.react('loss',traces,{{title:{{text:'固定 500-step 训练历史：loss 与 validation FK P95 分轴显示',font:{{color:ink()}}}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{color:ink()}},xaxis:{{title:'optimizer step'}},yaxis:{{title:'train beta loss'}},yaxis2:{{title:'validation FK P95 (mm)',overlaying:'y',side:'right'}},legend:{{orientation:'h',y:-.22}},margin:{{l:60,r:70,t:44,b:80}}}},cfg)}}
const observer=new IntersectionObserver(entries=>entries.forEach(e=>{{if(e.target.id==='coverage'){{coverageVisible=e.isIntersecting;if(e.isIntersecting)coverage();else if(e.target.data)Plotly.purge(e.target)}}if(e.target.id==='tracking'){{trackingVisible=e.isIntersecting;if(e.isIntersecting)tracking();else if(e.target.data)Plotly.purge(e.target)}}}}),{{rootMargin:'180px'}});observer.observe($('coverage'));observer.observe($('tracking'));fillTracks();loss();$('trajectory').onchange=tracking;$('projection').onchange=()=>{{state.projection=$('projection').value;tracking()}};['showL0raw','showL0dls','showL1raw','showL1dls'].forEach(id=>$(id).onchange=tracking);$('coverageGlobal').onclick=()=>{{state.coverageLocal=false;coverage()}};$('coverageLocal').onclick=()=>{{state.coverageLocal=true;coverage()}};$('trackingGlobal').onclick=()=>{{state.trackingLocal=false;state.projection='3d';$('projection').value='3d';tracking()}};$('trackingLocal').onclick=()=>{{state.trackingLocal=true;tracking()}};$('theme').onclick=()=>{{state.dark=!state.dark;document.documentElement.dataset.theme=state.dark?'dark':'light';coverage();tracking();loss()}};
</script></body></html>'''
    (OUT / HTML_NAME).write_text(html_doc, encoding='utf-8')
    output_paths = [OUT / 'build_report.py', OUT / 'validate_browser.js', OUT / 'evidence_map.json', OUT / HTML_NAME]
    output_paths.extend(path for path in [
        OUT / 'browser_validation.json', OUT / 'coverage_validation.png', OUT / 'tracking_global_validation.png',
        OUT / 'tracking_local_validation.png', OUT / 'narrow_validation.png',
    ] if path.is_file())
    manifest = {
        'observed_at': OBSERVATION, 'kind': 'retry20-candidate-layout-teacher-student-diagnostic-report',
        'frozen_attempt_modified': False, 'scientific_source_fixed_point': SOURCE, 'binding_fixed_point': BINDING,
        'code_snapshot_sha': CODE_SNAPSHOT,
        'artifact_verification': {'completion_manifest_count': len(completion), 'completion_manifest_bad': [], 'input_manifest_count': len(inputs), 'input_manifest_bad': []},
        'scientific_gate': 'not_passed; diagnostic-only presentation does not alter the failed original geometry Gate',
        'files': [reference(path, 'report_derivative', OUT) for path in output_paths],
    }
    (OUT / 'report_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'html': str(OUT / HTML_NAME), 'rows': len(l0), 'tracks': len(tracks), 'completion_artifacts_verified': len(completion), 'input_artifacts_verified': len(inputs)}, ensure_ascii=False))


if __name__ == '__main__':
    build()
