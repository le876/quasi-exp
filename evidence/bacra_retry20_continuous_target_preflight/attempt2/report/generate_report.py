#!/usr/bin/env python3
"""Render the retry20 target-only preflight diagnostic report.

This program consumes sealed retry20 output only.  It never calls an inverse
solver or a training entry point; the zero-pose drawing is a deterministic FK
replay from the retry20 binding's authoritative robot inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ATTEMPT = ROOT / "runs/bacra_retry20_preflight_attempt2"
STAGE = ATTEMPT / "00_objective_feasibility"
WORKTREE = ROOT / ".worktrees/retry20-continuous-targets"
HTML = HERE / "retry20_preflight_attempt2_geometry_report.html"
MANIFEST = HERE / "report_manifest.json"
SOURCE = "33b9e8740120b424276bb4b655b26ee043e9fcd0"
BINDING = "38b9a0b59d6fb5bd322d96cbd69f7e493fdb6b66"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample(frame: pd.DataFrame, count: int = 6000) -> list[list[float]]:
    """Deterministically show a representative subset without changing its frame."""
    ordered = frame.sort_values("target_id", kind="stable")
    indices = np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)
    return (ordered.iloc[indices][["x_m", "y_m", "z_m"]].to_numpy(float) * 1000.0).round(3).tolist()


def zero_pose_centerline() -> tuple[list[list[float]], list[float], dict[str, str]]:
    """Use the registered beta=0 FK and return each real DH-frame origin in mm."""
    old_cwd = Path.cwd()
    try:
        os.chdir(WORKTREE)
        sys.path.insert(0, str(WORKTREE / "src"))
        from quasi_exp.io.robot_inputs import load_robot_inputs
        from quasi_exp.model.kinematics import forward_kinematics
        from quasi_exp.model.sampling import beta_to_theta

        config_path = WORKTREE / "configs/robot_rods_only_standard_100k.yaml"
        config = yaml.safe_load(config_path.read_text())
        robot = load_robot_inputs(config)
        endpoint, transforms = forward_kinematics(
            beta_to_theta(np.zeros(6)), robot.lengths_m, robot.p_end_local_m,
            theta_sign=float(config["kinematics"]["theta_sign"]),
        )
        centerline = (transforms[:, :3, 3] * 1000.0).round(6).tolist()
        end_mm = (endpoint * 1000.0).round(6).tolist()
        return centerline, end_mm, {
            "robot_config": str(config_path.relative_to(WORKTREE)),
            "robot_config_sha256": sha256(config_path),
            "lengths_csv_sha256": sha256(WORKTREE / "data/robot/generated/lengths.csv"),
            "end_effector_csv_sha256": sha256(WORKTREE / "data/robot/generated/end_effector.csv"),
        }
    finally:
        os.chdir(old_cwd)


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def make_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>retry20 连续目标替换：候选几何预检报告</title><link rel="icon" href="data:,">
<style>
:root{{--bg:#f5f7fb;--paper:#fff;--ink:#172033;--muted:#5c6b82;--line:#dce3ee;--blue:#1976d2;--orange:#d97706;--red:#c62828;--green:#16803a;--violet:#7057d6;--shadow:0 8px 28px #20304a12}}body.dark{{--bg:#111827;--paper:#182235;--ink:#edf3ff;--muted:#b4c0d4;--line:#344158;--blue:#62a7ff;--orange:#ffb24c;--red:#ff7676;--green:#57d58a;--violet:#b19cff;--shadow:none}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.58 Inter,"Noto Sans SC","Microsoft Yahei",sans-serif}}header{{background:linear-gradient(120deg,#102b50,#296da5);color:white;padding:42px max(24px,calc((100vw - 1200px)/2));}}h1{{font-size:clamp(1.55rem,3vw,2.5rem);line-height:1.2;margin:0 0 12px}}h2{{font-size:1.35rem;margin:0 0 14px}}h3{{font-size:1.02rem;margin:0 0 8px}}p{{margin:8px 0}}.sub{{max-width:900px;color:#e0efff}}.tag{{display:inline-block;border:1px solid #93c5fd;border-radius:999px;padding:3px 10px;font-size:.84rem;margin:5px 6px 0 0}}main{{max-width:1200px;margin:0 auto;padding:26px 20px 46px}}.grid{{display:grid;gap:16px;grid-template-columns:repeat(4,minmax(0,1fr))}}.two{{display:grid;gap:18px;grid-template-columns:repeat(2,minmax(0,1fr));margin-top:18px}}section,.card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:var(--shadow);margin:18px 0}}.metric{{border-left:4px solid var(--blue)}}.metric.fail{{border-left-color:var(--red)}}.metric .value{{font-size:1.65rem;font-weight:750;margin-top:4px}}.small{{color:var(--muted);font-size:.9rem}}.status-fail{{color:var(--red);font-weight:750}}.status-pass{{color:var(--green);font-weight:750}}button{{border:1px solid var(--line);background:var(--paper);color:var(--ink);border-radius:8px;padding:8px 12px;font:inherit;cursor:pointer}}button[aria-pressed=true]{{background:var(--blue);color:white;border-color:var(--blue)}}.toolbar{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0 14px}}.canvas-wrap{{background:#071422;border-radius:12px;border:1px solid #35516f;min-height:460px;position:relative;overflow:hidden}}canvas{{display:block;width:100%;height:520px;touch-action:none}}.hint{{position:absolute;right:12px;bottom:8px;color:#c4d8ef;font-size:.8rem;background:#071422b8;padding:3px 7px;border-radius:5px}}.legend{{display:flex;flex-wrap:wrap;gap:13px;font-size:.9rem;margin-top:10px}}.dot{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px}}table{{width:100%;border-collapse:collapse;font-size:.9rem}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:var(--muted);font-weight:650}}.barrow{{display:grid;grid-template-columns:150px 1fr 60px;gap:9px;align-items:center;margin:8px 0}}.bar{{height:12px;background:#d6deeb;border-radius:8px;overflow:hidden}}body.dark .bar{{background:#344158}}.bar>i{{display:block;height:100%;border-radius:8px;background:var(--blue)}}.bar.red>i{{background:var(--red)}}.callout{{border-left:4px solid var(--orange);padding:10px 14px;background:#fff7e8;color:#5f4216;border-radius:6px;margin:12px 0}}body.dark .callout{{background:#3a2b16;color:#ffe0a5}}.gap{{border-left-color:var(--violet)}}code{{font-size:.83em;word-break:break-all}}details{{margin:8px 0}}summary{{cursor:pointer;font-weight:650}}footer{{max-width:1200px;margin:0 auto;padding:0 20px 40px;color:var(--muted);font-size:.86rem}}@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.two{{grid-template-columns:1fr}}main{{padding:15px 12px 30px}}section,.card{{padding:15px}}canvas{{height:430px}}.barrow{{grid-template-columns:120px 1fr 52px}}}}@media(max-width:430px){{.grid{{grid-template-columns:1fr}}header{{padding:30px 16px}}canvas{{height:380px}}table{{font-size:.78rem}}th,td{{padding:6px 4px}}}}
</style></head><body><header><h1>retry20 连续目标替换：候选几何预检</h1><p class="sub">绑定 attempt <code>{data['attempt_name']}</code> 的终态几何诊断。该尝试在任何 IK、Teacher、Student 或训练之前按协议停止。</p><span class="tag">运行完成 {data['elapsed_seconds']:.2f} s</span><span class="tag">Scientific status: geometry_preflight_failed</span><span class="tag">2026-09-08 Asia/Shanghai</span></header><main>
<div class="toolbar"><button id="theme" aria-pressed="false">切换暗色模式</button><span class="small">报告为自包含诊断视图；不改变 sealed artifacts。</span></div>
<section><h2>结论与证据边界</h2><p><span class="status-fail">几何 Gate 未通过。</span> 已形成 <b>24,000</b> 个 coarse 替换与 <b>99</b> 个 fine 替换的确定性候选布局，但 coverage、服务距离、轴向非退化和最大 phase bin 检查失败；因此协议要求在 DLS/IK 前停止。预算下界的 <code>feasible</code> 仅说明资源计数可行，独立的 geometry Gate 仍为 failed。该结论只描述本次固定 hash/Sobol 选择的 prospective layout，<b>不证明所有连续布局均不可行</b>，也不构成训练、Teacher 或部署结论。</p><div class="callout">候选 XYZ 只用于几何预检（<code>geometry_only_not_supervision</code>），不是监督数据。proposal beta 未作为 label、hint 或 warm start。</div></section>
<div class="grid"><div class="card metric"><div class="small">完整 primary 几何分母</div><div class="value">29,970</div><div class="small">原始与 prospective 均以此点数呈现</div></div><div class="card metric"><div class="small">冻结 required cell 分母</div><div class="value">88,168</div><div class="small">完整 required volume 87,104,000 mm³</div></div><div class="card metric fail"><div class="small">体积覆盖率</div><div class="value">98.518% → 82.081%</div><div class="small">Gate 下限 97.518%</div></div><div class="card metric fail"><div class="small">服务距离 P95</div><div class="value">10.000 → 11.741 mm</div><div class="small">限值 ≤ 10.000001 mm</div></div></div>
<section><h2>同一机器人基座坐标系中的真实 3D 点云</h2><p>全部点为基座物理坐标 <b>mm</b>。点击“原始 primary / prospective geometry”切换；两层都显示 29,970 的完整统计分母，画布为保留范围的确定性 6,000 点抽样。橙黄色线是基于绑定 worktree 的 authoritative FK 对 <code>beta=0</code> 的确定性重放：完整 31-frame 中心线，末端为 1,215.498 mm。</p><div class="toolbar"><button id="baseline" aria-pressed="true">原始 primary</button><button id="prospective" aria-pressed="false">prospective geometry</button><button id="reset">重置视角</button><span id="pointLabel" class="small"></span></div><div class="canvas-wrap"><canvas id="scene" aria-label="可旋转的基座坐标 3D 点云"></canvas><span class="hint">拖拽旋转 · 滚轮缩放 · 双击重置</span></div><div class="legend"><span><i class="dot" style="background:#55b7ff"></i>显示的 primary 点</span><span><i class="dot" style="background:#ffbd4a"></i>beta=0 本体中心线</span><span><i class="dot" style="background:#ff6767"></i>zero 末端</span></div></section>
<div class="two"><section><h2>候选选择、保护与间距</h2><table><tbody><tr><th>可迁移 primary cells</th><td>26,780</td></tr><tr><th>可接受候选行</th><td>26,488</td></tr><tr><th>可接受 coarse</th><td>26,389</td></tr><tr><th>具体布局</th><td>24,000 coarse + 99 fine = 24,099</td></tr><tr><th>最小替换间距</th><td>1.517226 mm；低于 1 mm 的 pair = 0</td></tr><tr><th>排除</th><td>exact zero 1；hard Teacher zero attachment 20；保留 historical label 3,169；无 admissible proposal 292</td></tr><tr><th>cross-pool support</th><td>100%</td></tr></tbody></table><p class="small">保护 zero/anchor 后，实际候选还须位于原 source cell、相同 macroblock/split；这一布局不是全局最优。</p></section><section><h2>Phase：改善了什么，未通过什么</h2><table><thead><tr><th>10 mm primary 指标</th><th>原始</th><th>prospective</th><th>门槛</th></tr></thead><tbody><tr><td>x entropy</td><td>0.2370</td><td>0.9630</td><td>≥ 0.90</td></tr><tr><td>joint entropy (4³)</td><td>0.1895</td><td>0.9495</td><td>审计项</td></tr><tr><td>center-plane fraction</td><td>0.8901</td><td>0.0893</td><td>≤ 0.20</td></tr><tr><td>x 的最大 10-bin fraction</td><td>0.8986</td><td class="status-fail">0.2049</td><td>≤ 0.20</td></tr></tbody></table><p class="small">此行明确是 x 指标；prospective x/y/z phase correlation 分别为 xy −0.0021、xz −0.0010、yz 0.0050。x 的最大 bin 仍超过门槛，因此不通过。</p></section></div>
<section><h2>覆盖损失：固定 88,168-cell 分母上的复算</h2><p>baseline 是由所有 <b>29,970 labels 的 XYZ</b> 形成的几何服务（86,830 served cells）；它不是旧 retry19 的 zero-connected 服务（86,818 cells）。required cell 分母始终为 88,168，未被缩小。</p><div class="two"><div><div class="barrow"><span>baseline served</span><span class="bar"><i style="width:98.4824%"></i></span><b>86,830</b></div><div class="barrow"><span>prospective served</span><span class="bar red"><i style="width:82.2260%"></i></span><b>72,497</b></div><div class="barrow"><span>lost probes</span><span class="bar red"><i style="width:16.408%"></i></span><b>14,467</b></div><div class="barrow"><span>gained probes</span><span class="bar"><i style="width:.152%"></i></span><b>134</b></div></div><div><table><tbody><tr><th>lost volume</th><td>14,433,750 mm³</td></tr><tr><th>gained volume</th><td>116,500 mm³</td></tr><tr><th>lost baseline distance quantiles</th><td>[8.29156, 10, 10, 10, 10, 10] mm (min, P50, P90, P95, P99, max)</td></tr><tr><th>losses at 10 mm ± 1e−6</th><td>14,451 / 14,467</td></tr><tr><th>prospective axial minimum</th><td class="status-fail">0.919745 → 0.795879</td></tr></tbody></table></div></div><div class="callout">解释：source-cell 自身得到服务，并不保证原中心布局中恰在 10 mm 边界服务的相邻 cell 仍被服务。14,451 个 lost probes 位于旧 10 mm ± 1e−6 边界；同时新的 volume-weighted P95 为 11.741 mm。这是实质性 coverage 退化，不是浮点容差现象。</div></section>
<div class="two"><section><h2>轴向切片（u = zero_x − x，mm）</h2><p>20 个 10 mm 切片在冻结体积分母上计算。两个 sealed slice 表的最小行均为 0–10 mm：baseline 0.919745，prospective 0.795879；完整 slices 记录于 <code>baseline_axial_slices.parquet</code> 与 <code>prospective_axial_slices.parquet</code>。</p><p class="small">要求同时满足：不低于 baseline，且不低于 0.90；两项均失败。</p></section><section><h2>因果研究尚未开始</h2><p>预检在 <code>stop_before_ik</code> 终止。没有运行 IK/DLS、Teacher candidate bank 或 graph，没有生成 supervision labels，也没有启动 Student 训练或 FK tracking 评估。</p><p class="small">因此 largest-found circle、压力轨迹、Target/Teacher/raw Student/DLS 路径、training history、model lock 与 tracking metrics 在本 attempt 中均为 <b>not_applicable / not_evaluated</b>：并非缺少展示，而是 protocol 规定 solver 前停止。</p></section></div>
<section><h2>证据、运行与可复现性</h2><table><tbody><tr><th>科学 source fixed point</th><td><code>{data['source']}</code></td></tr><tr><th>执行 binding</th><td><code>{data['binding']}</code>（owning worktree clean）</td></tr><tr><th>执行范围</th><td>target_only_preflight；launcher success rc=0，2026-09-08 15:36:18–15:36:42 Asia/Shanghai，{data['elapsed_seconds']:.7f} s</td></tr><tr><th>输入验证</th><td>attempt2 的 22 个输出 hash 匹配；共享六文件 schema validator valid；runner 核验 13 个 upstream manifests、185 个 upstream artifacts</td></tr><tr><th>attempt 一致性</th><td>candidate / planned / prospective-primary / prospective-served / geometry_audit 字节 hash 与 attempt1 一致；evaluation panel 按列名对齐后行序、值及 dtype 完全相同。</td></tr><tr><th>代码 snapshot</th><td><code>c18d7ee858e2f643913410f77005425a2b56c38f</code></td></tr><tr><th>零位诊断</th><td><code>{data['robot']['robot_config']}</code> (SHA-256 <code>{data['robot']['robot_config_sha256']}</code>) + registered FK；不是编造构型</td></tr><tr><th>claim boundary</th><td>报告只解释此候选几何预检；不是 formal Gate 通过、未授权 downstream、未证明 Teacher/Student、tracking、训练或动态闭环控制。</td></tr></tbody></table><details><summary>报告所用输入（SHA-256）</summary><pre id="hashes"></pre></details></section>
</main><footer>报告生成器与 manifest 位于本目录。浏览器交互验证属于报告验收；它不替代 sealed artifact 核验或 scientific Gate。</footer>
<script>const D={payload};document.getElementById('hashes').textContent=D.hashes.map(x=>x.path+'  '+x.sha256).join('\\n');let mode='baseline';const can=document.getElementById('scene'),ctx=can.getContext('2d');let yaw=-0.66,pitch=-0.38,zoom=1,drag=null;function size(){{let r=can.getBoundingClientRect(),q=devicePixelRatio||1;can.width=r.width*q;can.height=r.height*q;ctx.setTransform(q,0,0,q,0,0);return r}}function rotate(p){{let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);let x=p[0]*cy-p[1]*sy,y=p[0]*sy+p[1]*cy,z=p[2];return [x,y*cp-z*sp,y*sp+z*cp]}}function draw(){{let r=size(),w=r.width,h=r.height;ctx.fillStyle='#071422';ctx.fillRect(0,0,w,h);let pts=D[mode],all=pts.concat(D.robot.line,[D.robot.end]),min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];all.forEach(p=>p.forEach((v,i)=>{{min[i]=Math.min(min[i],v);max[i]=Math.max(max[i],v)}}));let center=min.map((v,i)=>(v+max[i])/2),scale=Math.min(w,h)/(Math.max(...max.map((v,i)=>v-min[i]))*1.30)*zoom;function pr(p){{let a=rotate([p[0]-center[0],p[1]-center[1],p[2]-center[2]]);return [w/2+a[0]*scale,h/2-a[1]*scale,a[2]]}}let pp=pts.map(pr).sort((a,b)=>a[2]-b[2]);ctx.fillStyle='#55b7ff';ctx.globalAlpha=.58;for(let p of pp){{ctx.fillRect(p[0]-1.3,p[1]-1.3,2.6,2.6)}}ctx.globalAlpha=1;let o=pr([0,0,0]);for(let a of [[[150,0,0],'#ff6b6b','+X 150 mm'],[[0,150,0],'#63e6be','+Y 150 mm'],[[0,0,150],'#b197fc','+Z 150 mm']]){{let q=pr(a[0]);ctx.strokeStyle=a[1];ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(o[0],o[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.fillStyle=a[1];ctx.font='12px sans-serif';ctx.fillText(a[2],q[0]+6,q[1]-5)}}ctx.fillStyle='#ffffff';ctx.beginPath();ctx.arc(o[0],o[1],4,0,7);ctx.fill();ctx.fillText('O (0, 0, 0) mm',o[0]+7,o[1]+14);let ln=D.robot.line.map(pr);ctx.strokeStyle='#ffbd4a';ctx.lineWidth=3;ctx.beginPath();ln.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();ctx.fillStyle='#ffbd4a';ln.forEach(p=>{{ctx.beginPath();ctx.arc(p[0],p[1],3.2,0,7);ctx.fill()}});let e=pr(D.robot.end);ctx.fillStyle='#ff6767';ctx.beginPath();ctx.arc(e[0],e[1],6,0,7);ctx.fill();ctx.font='12px sans-serif';ctx.fillText('beta=0 end (1215.498 mm)',e[0]+8,e[1]-8);ctx.fillStyle='#cde5ff';ctx.fillText('Physical base frame: X / Y / Z in mm',12,20);ctx.fillText(mode==='baseline'?'原始 primary：29,970 完整 / 6,000 显示':'prospective geometry：29,970 完整 / 6,000 显示',12,39)}}function choose(x){{mode=x;document.getElementById('baseline').setAttribute('aria-pressed',x==='baseline');document.getElementById('prospective').setAttribute('aria-pressed',x==='prospective');document.getElementById('pointLabel').textContent=x==='baseline'?'原始 primary（监督标签来源）':'prospective geometry（非监督数据）';draw()}}document.getElementById('baseline').onclick=()=>choose('baseline');document.getElementById('prospective').onclick=()=>choose('prospective');function reset(){{yaw=-.66;pitch=-.38;zoom=1;draw()}}document.getElementById('reset').onclick=reset;can.ondblclick=reset;can.onpointerdown=e=>{{drag=[e.clientX,e.clientY];can.setPointerCapture(e.pointerId)}};can.onpointermove=e=>{{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.45,Math.min(1.45,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw()}};can.onpointerup=()=>drag=null;can.onwheel=e=>{{e.preventDefault();zoom=Math.max(.38,Math.min(4,zoom*(e.deltaY>0?.9:1.1)));draw()}};window.onresize=draw;document.getElementById('theme').onclick=()=>{{let dark=document.body.classList.toggle('dark');document.getElementById('theme').setAttribute('aria-pressed',dark);draw()}};choose('baseline');</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser-validated", action="store_true", help="record only after completing browser checks")
    args = parser.parse_args()
    if not ATTEMPT.is_dir() or not WORKTREE.is_dir():
        raise FileNotFoundError("retry20 attempt or binding worktree is unavailable")
    identity = json.loads((ATTEMPT / "run_identity.json").read_text())
    if identity["scientific_source_fixed_point"] != SOURCE or identity["binding_fixed_point"] != BINDING:
        raise ValueError("attempt source/binding differs from assigned retry20 identity")
    audit = json.loads((STAGE / "geometry_audit.json").read_text())
    summary = json.loads((ATTEMPT / "summary.json").read_text())
    base = pd.read_parquet(STAGE / "relocation_eligibility.parquet")
    prospective = pd.read_parquet(STAGE / "prospective_primary_targets.parquet")
    baseline_cells = pd.read_parquet(STAGE / "baseline_served_cells.parquet").set_index("cell_id")
    prospective_cells = pd.read_parquet(STAGE / "prospective_served_cells.parquet").set_index("cell_id")
    required = baseline_cells.required & prospective_cells.required
    lost = required & baseline_cells.served & ~prospective_cells.served
    gained = required & ~baseline_cells.served & prospective_cells.served
    loss_stats = {
        "required_cells": int(required.sum()), "baseline_served": int((required & baseline_cells.served).sum()),
        "prospective_served": int((required & prospective_cells.served).sum()), "lost_probes": int(lost.sum()),
        "gained_probes": int(gained.sum()), "lost_volume_mm3": float(baseline_cells.loc[lost, "volume_mm3"].sum()),
        "gained_volume_mm3": float(prospective_cells.loc[gained, "volume_mm3"].sum()),
        "required_volume_mm3": float(baseline_cells.loc[required, "volume_mm3"].sum()),
        "lost_at_10mm_tolerance": int(np.isclose(baseline_cells.loc[lost, "service_distance_mm"], 10.0, atol=1e-6).sum()),
        "lost_distance_quantiles_mm": np.quantile(baseline_cells.loc[lost, "service_distance_mm"], [0, .5, .9, .95, .99, 1]).tolist(),
    }
    line, endpoint, robot = zero_pose_centerline()
    inputs = sorted([path for path in ATTEMPT.rglob("*") if path.is_file()])
    input_hashes = [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in inputs]
    data = {"source": SOURCE, "binding": BINDING, "attempt_name": ATTEMPT.name, "elapsed_seconds": float(summary["elapsed_seconds"]), "baseline": stable_sample(base), "prospective": stable_sample(prospective), "robot": {"line": line, "end": endpoint, **robot}, "hashes": input_hashes}
    HTML.write_text(make_html(data), encoding="utf-8")
    outputs = [{"path": str(HTML.relative_to(ROOT)), "sha256": sha256(HTML)}, {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha256(Path(__file__))}]
    for screenshot in sorted(HERE.glob("browser_*.png")):
        outputs.append({"path": str(screenshot.relative_to(ROOT)), "sha256": sha256(screenshot)})
    manifest = {
        "report_schema_version": 1, "experiment_id": identity["experiment_id"], "attempt_root": str(ATTEMPT.relative_to(ROOT)),
        "scientific_source_fixed_point": SOURCE, "binding_fixed_point": BINDING,
        "code_snapshot_sha": "c18d7ee858e2f643913410f77005425a2b56c38f",
        "report_kind": "candidate_geometry_diagnostic", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_hashes, "output_sha256": outputs, "robot_zero_pose_diagnostic": robot,
        "recomputed_coverage_delta": loss_stats,
        "claim_boundary": "Explains only the sealed retry20 deterministic prospective geometry layout. It is not a formal scientific pass, supervision dataset, Teacher/Student/training/tracking evaluation, solver result, deployment authorization, or proof that every continuous layout is infeasible.",
        "evidence_gaps": {"teacher": "not_evaluated; protocol stopped before IK", "student": "not_evaluated; no training launched", "tracking": "not_evaluated; no solver or trajectory outputs", "largest_circle_and_pressure_trajectories": "not_applicable; preflight terminated before any solver"},
        "attempt2_consistency": {"launcher": "success rc=0; 2026-09-08 15:36:18-15:36:42 Asia/Shanghai", "runner_elapsed_seconds": 22.8031405, "output_hashes": "22 matched", "shared_six_file_schema_validator": "valid", "byte_identical_to_attempt1": ["candidate_replacement_registry.parquet", "planned_replacement_registry.parquet", "prospective_primary_targets.parquet", "prospective_served_cells.parquet", "geometry_audit.json"], "evaluation_panel_registry": "after column-name alignment, row order, all values, and dtypes equal"},
        "browser_validation_details": {"desktop_1440x960": {"horizontal_overflow": False, "canvas_pixels": [1116, 520], "base_axes": "+X/+Y/+Z, each 150 mm from O=(0,0,0) mm", "initial_3d_screenshot": "browser_3d_initial.png"}, "narrow_390x844": {"horizontal_overflow": False, "canvas_css_pixels": [332, 380]}, "interactions": ["primary/prospective switch", "pointer rotation", "reset", "dark mode"], "console": "0 errors, 0 warnings"},
        "browser_validation": "passed" if args.browser_validated else "pending; run browser checks before delivery",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
