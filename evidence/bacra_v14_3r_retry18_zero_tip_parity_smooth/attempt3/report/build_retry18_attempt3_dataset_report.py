#!/usr/bin/env python3
"""Build the isolated Retry18 attempt3 advisor-facing diagnostic report.

This is deliberately a report-local, read-only replay.  It verifies every
sealed input listed by the summary and stage manifests before it reads the
Parquet files; it then writes only derived report material in this directory.
It does not re-run Student training, IK solving, data materialization, or any
formal Gate.
"""
from __future__ import annotations

import hashlib
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCE = Path("/mnt/ML_projects/quasi_exp/.worktrees/bacra-v14-3r-retry18-full-exec-6b3ba06")
SOURCE_SHA = "6b3ba0636c1a620cafaf425c7c235754dea379f6"
BINDING_SHA = "a028916517419d43ffcc1a0d8d0439536dac6bf2"
HTML_NAME = "retry18_attempt3_advisor_dataset_report.html"


def get_plotlyjs() -> str:
    """Keep the disabled historical placeholder empty; this report uses Canvas."""
    return ""

if str(SOURCE / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE / "src"))
from quasi_exp.model.kinematics import forward_kinematics  # noqa: E402
from quasi_exp.model.sampling import beta_to_theta  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((RUN / rel).read_text(encoding="utf-8"))


def record(rel: str, kind: str = "sealed_attempt_input") -> dict[str, Any]:
    path = RUN / rel
    return {"kind": kind, "path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)}


def source_record(rel: str, kind: str) -> dict[str, Any]:
    path = SOURCE / rel
    return {"kind": kind, "path": rel, "source_fixed_point": SOURCE_SHA,
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_seals() -> dict[str, Any]:
    summary = load_json("13_summary/artifact_manifest.json")
    summary_bad: list[dict[str, Any]] = []
    for item in summary["artifacts"]:
        path = RUN / item["path"]
        actual = sha256(path) if path.is_file() else None
        if actual != item["sha256"]:
            summary_bad.append({"path": item["path"], "expected": item["sha256"], "actual": actual})
    stage_bad: list[dict[str, Any]] = []
    manifests = sorted(RUN.glob("*/completion_manifest.json"))
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["artifacts"]:
            path = manifest_path.parent / item["path"]
            actual = sha256(path) if path.is_file() else None
            if actual != item["sha256"]:
                stage_bad.append({"stage": manifest_path.parent.name, "path": item["path"],
                                  "expected": item["sha256"], "actual": actual})
    if summary_bad or stage_bad:
        raise RuntimeError(f"sealed input verification failed: {summary_bad=}, {stage_bad=}")
    return {"summary_artifact_entries": len(summary["artifacts"]),
            "summary_sha256_mismatches": summary_bad,
            "completion_manifest_count": len(manifests),
            "completion_sha256_mismatches": stage_bad}


def sample(frame: pd.DataFrame, group: str, maximum: int) -> pd.DataFrame:
    """Stable stratified visual sample; statistics always retain full rows."""
    pieces = []
    total = max(len(frame), 1)
    for _value, part in frame.groupby(group, sort=True, dropna=False):
        take = max(1, round(maximum * len(part) / total))
        ordered = part.sort_values("target_id", kind="stable")
        pieces.append(ordered if len(ordered) <= take else ordered.iloc[np.linspace(0, len(ordered) - 1, take, dtype=int)])
    return pd.concat(pieces, ignore_index=True)


def pts(frame: pd.DataFrame, role: str) -> list[list[float]]:
    return [[round(float(x) * 1000, 3), round(float(y) * 1000, 3), round(float(z) * 1000, 3), str(label)]
            for x, y, z, label in frame[["x_m", "y_m", "z_m", role]].itertuples(index=False, name=None)]


def zero_geometry() -> tuple[list[list[float]], list[float], dict[str, Any], np.ndarray, np.ndarray]:
    lengths_path = SOURCE / "data/robot/generated/lengths.csv"
    end_path = SOURCE / "data/robot/generated/end_effector.csv"
    lengths = pd.read_csv(lengths_path)["value_m"].to_numpy(float)
    end = pd.read_csv(end_path).iloc[0]
    p_end = np.array([end["p_end_x_m"], end["p_end_y_m"], end["p_end_z_m"], 1.0], dtype=float)
    endpoint, frames = forward_kinematics(beta_to_theta(np.zeros(6)), lengths, p_end, theta_sign=-1.0)
    centreline = (np.vstack([frames[:, :3, 3], endpoint]) * 1000).round(5).tolist()
    provenance = {"kind": "derived_diagnostic_fk_replay",
                  "method": "sealed beta -> beta_to_theta -> forward_kinematics; no training, solving, repair, or Gate mutation",
                  "source_fixed_point": SOURCE_SHA, "theta_sign": -1.0,
                  "kinematics": source_record("src/quasi_exp/model/kinematics.py", "authoritative_fk"),
                  "beta_mapping": source_record("src/quasi_exp/model/sampling.py", "authoritative_beta_mapping"),
                  "lengths": source_record("data/robot/generated/lengths.csv", "robot_geometry"),
                  "end_effector": source_record("data/robot/generated/end_effector.csv", "robot_geometry")}
    return centreline, (endpoint * 1000).round(5).tolist(), provenance, lengths, p_end


def fk_replay(beta: np.ndarray, lengths: np.ndarray, p_end: np.ndarray) -> np.ndarray:
    out = np.empty((len(beta), 3), dtype=float)
    for index, row in enumerate(np.asarray(beta, dtype=float)):
        endpoint, _frames = forward_kinematics(beta_to_theta(row), lengths, p_end, theta_sign=-1.0)
        out[index] = endpoint * 1000
    return out


def table_rows(frame: pd.DataFrame, columns: list[str], formatters: dict[str, str] | None = None) -> str:
    formatters = formatters or {}
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    rows = []
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        cells = []
        for column, value in zip(columns, values, strict=True):
            if isinstance(value, (float, np.floating)):
                pattern = formatters.get(column, ".3f")
                text = format(float(value), pattern)
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-scroll'><table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    identity = load_json("run_identity.json")
    gate = load_json("13_summary/gate.json")
    completion = load_json("13_summary/completion_manifest.json")
    lock = load_json("10_dataset_model_lock/dataset_model_lock.json")
    selected = load_json("09_student_selection/selected_model.json")
    if identity["scientific_source_fixed_point"] != SOURCE_SHA or identity["binding_fixed_point"] != BINDING_SHA:
        raise RuntimeError("run identity does not match the assigned fixed points")
    if not gate["operational_completion"] or gate["status"] != "complete":
        raise RuntimeError("attempt3 summary is not operationally complete")
    seals = verify_seals()

    full = pd.read_parquet(RUN / "05_unified_dataset/unified_full_g4_dataset.parquet")
    quotient = pd.read_parquet(RUN / "05_unified_dataset/unified_quotient_supervision.parquet")
    support = pd.read_parquet(RUN / "01_zero_tip_discovery/common_axis_aware_support.parquet")
    profile = pd.read_parquet(RUN / "02_profile_preflight/frozen_tip_profile.parquet")
    compatibility = pd.read_parquet(RUN / "05_unified_dataset/parity_label_compatibility.parquet")
    sweeps = pd.read_parquet(RUN / "06_seam_causal_audit/teacher_seam_sweeps.parquet")
    causal = pd.read_parquet(RUN / "06_seam_causal_audit/student_teacher_seam_causal_panel.parquet")
    exponents = pd.read_parquet(RUN / "07_parity_regularity/teacher_odd_vanish_exponent.parquet")
    old_rectangle = pd.read_parquet(RUN / "00_objective_baseline/rectangle_raw_spike_report.parquet")
    trajectories = pd.read_parquet(RUN / "12_trajectory_evaluation/trajectory_report.parquet")
    waypoints = pd.read_parquet(RUN / "12_trajectory_evaluation/trajectory_waypoint_evaluation.parquet")
    shapes = pd.read_parquet(RUN / "11_heldout_trajectories/maximal_shape_registry.parquet")
    selection_metrics = pd.read_parquet(RUN / "09_student_selection/model_selection_metrics.parquet")
    history = pd.read_parquet(RUN / "08_student_ablation/models/S1_seam_root/training_history.parquet")

    display_full = sample(full, "domain_class", 12000)
    display_q = sample(quotient, "target_role", 6000)
    centreline, zero_end, fk_provenance, lengths, p_end = zero_geometry()
    trajectory_data: dict[str, Any] = {}
    for trajectory_id, frame in waypoints.groupby("trajectory_id", sort=True):
        ordered = frame.sort_values("waypoint_index", kind="stable")
        target = ordered[["x_m", "y_m", "z_m"]].to_numpy(float) * 1000
        rendered: dict[str, list[list[float]]] = {"target": np.round(target, 4).tolist()}
        for key, prefix in (("teacher", "teacher"), ("raw", "raw_student"), ("dls2", "dls2")):
            beta = ordered[[f"{prefix}_beta{i}_rad" for i in range(1, 7)]].to_numpy(float)
            rendered[key] = np.round(fk_replay(beta, lengths, p_end), 4).tolist()
        report = trajectories.loc[trajectories["trajectory_id"].eq(trajectory_id)].iloc[0]
        trajectory_data[str(trajectory_id)] = {"shape": str(ordered["shape_class"].iloc[0]),
                                                "kind": str(ordered["heldout_kind"].iloc[0]),
                                                "data": rendered,
                                                "metrics": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v) if isinstance(v, (float, np.floating, int, np.integer)) else str(v)) for k, v in report.to_dict().items()}}

    seam_summary = causal.groupby(["seam", "normal_offset_mm"], as_index=False).agg(
        student_fk_residual_mm=("student_fk_residual_mm", "median"),
        student_teacher_weighted_deg=("student_teacher_weighted_deg", "median"),
        teacher_jacobian_spectral_norm_m_per_rad=("teacher_jacobian_spectral_norm_m_per_rad", "median"))
    exponent_summary = exponents.groupby("seam", as_index=False).agg(
        components=("component", "count"), alpha_min=("alpha", "min"), alpha_median=("alpha", "median"), alpha_max=("alpha", "max"))
    profile_rows = profile[["u_center_mm", "inner_radius_mm", "outer_radius_mm"]].to_dict(orient="records")
    profile_draw = []
    for row in profile_rows:
        for phi in np.linspace(0, 2 * np.pi, 49):
            profile_draw.append([round(1215.498 - float(row["u_center_mm"]), 3),
                                 round(float(row["outer_radius_mm"]) * np.cos(phi), 3),
                                 round(float(row["outer_radius_mm"]) * np.sin(phi), 3)])
    inputs = ["run_identity.json", "13_summary/gate.json", "13_summary/artifact_manifest.json", "13_summary/completion_manifest.json",
              "10_dataset_model_lock/dataset_model_lock.json", "09_student_selection/selected_model.json",
              "01_zero_tip_discovery/common_axis_aware_support.parquet", "02_profile_preflight/frozen_tip_profile.parquet",
              "05_unified_dataset/unified_full_g4_dataset.parquet", "05_unified_dataset/unified_quotient_supervision.parquet",
              "05_unified_dataset/parity_label_compatibility.parquet", "06_seam_causal_audit/teacher_seam_sweeps.parquet",
              "06_seam_causal_audit/student_teacher_seam_causal_panel.parquet", "07_parity_regularity/teacher_odd_vanish_exponent.parquet",
              "00_objective_baseline/rectangle_raw_spike_report.parquet", "08_student_ablation/models/S1_seam_root/training_history.parquet",
              "09_student_selection/model_selection_metrics.parquet", "11_heldout_trajectories/maximal_shape_registry.parquet",
              "12_trajectory_evaluation/trajectory_report.parquet", "12_trajectory_evaluation/trajectory_waypoint_evaluation.parquet"]
    input_manifest = {"schema_version": 1, "kind": "derived_diagnostic_report_inputs", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                      "attempt_root": str(RUN), "source_fixed_point": SOURCE_SHA, "binding_fixed_point": BINDING_SHA,
                      "inputs": [record(rel) for rel in inputs], "source_inputs": list(fk_provenance.values())[3:]}
    (OUT / "input_manifest.json").write_text(json.dumps(input_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    data_axis = gate["data_axis"]
    rectangle_new = trajectories[trajectories["shape_class"].str.contains("rectangle")].copy()
    rectangle_new["small_pimple_flag"] = rectangle_new["raw_student_path_step_excess_maximum_mm"].gt(5.0)
    metric_cols = ["trajectory_id", "shape_class", "teacher_complete", "teacher_fk_p95_mm", "teacher_fk_maximum_mm",
                   "raw_student_fk_p95_mm", "raw_student_fk_maximum_mm", "raw_student_path_step_excess_p99_mm",
                   "raw_student_path_step_excess_maximum_mm", "raw_student_raw_step_gt7_rate", "dls2_fk_p95_mm", "dls2_fk_maximum_mm",
                   "dls2_path_step_excess_p99_mm", "dls2_path_step_excess_maximum_mm", "dls2_raw_step_gt7_rate", "raw_student_green", "dls2_green"]
    evidence_map = {"report_kind": "derived_diagnostic_visualization", "attempt": identity["experiment_id"], "attempt_root": str(RUN),
                    "scientific_source_fixed_point": SOURCE_SHA, "binding_fixed_point": BINDING_SHA,
                    "observation_date": datetime.now(timezone.utc).date().isoformat(), "sealed_verification": seals,
                    "coverage": {"full_g4_rows": len(full), "full_g4_display_rows": len(display_full), "quotient_rows": len(quotient), "quotient_display_rows": len(display_q),
                                 "sampling": "stable target_id-ordered stratified sample; visible points are real XYZ coordinates in mm; all denominators use sealed full rows"},
                    "fk_replay": fk_provenance, "trajectory_count": len(trajectory_data),
                    "claim_boundary": {"diagnostic_only": True, "formal_authorized": False, "deployment_authorized": False,
                                       "full_workspace_authorized": False, "continuous_workspace_authorized": False, "dls2_is_real_closed_loop": False}}
    (OUT / "evidence_map.json").write_text(json.dumps(evidence_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    payload = {"full": pts(display_full, "domain_class"), "quotient": pts(display_q, "target_role"), "zero_line": centreline, "zero_end": zero_end,
               "profile": profile_draw, "trajectory": trajectory_data,
               "seam": {"rows": seam_summary.to_dict(orient="records"), "teacher_sweep_rows": int(len(sweeps)), "causal_rows": int(len(causal))},
               "history": {k: [float(x) for x in history[k].to_numpy()] for k in history.select_dtypes(include=[np.number]).columns}}
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data_color = "#ce3f3f" if not data_axis["green"] else "#1f9d6a"
    parity_ok = int(compatibility["compatible"].sum())
    alpha_in_band = int(exponents["alpha"].between(.75, 1.25).sum())
    alpha_total = int(len(exponents))
    key_input_rows = "".join(f"<tr><td>{html.escape(item['path'])}</td><td>{item['bytes']:,}</td><td><code>{item['sha256']}</code></td></tr>" for item in input_manifest["inputs"])
    html_text = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>retry18 attempt3：zero-tip 与 parity-smooth 诊断报告</title><style>
:root{{--bg:#f5f7fb;--card:#fff;--text:#14212d;--muted:#506170;--line:#d9e1e8;--accent:#2456a6;--target:#20242a;--teacher:#e2772b;--raw:#1775c1;--dls:#168b62;--red:#b73636;--green:#147454}}body.dark{{--bg:#101720;--card:#17222e;--text:#e8eff7;--muted:#b5c2cf;--line:#304253;--accent:#8fbcff;--target:#fff;--teacher:#ffc477;--raw:#73baff;--dls:#6ce0b0;--red:#ff9898;--green:#79dcad}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.65 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif}}main{{max-width:1480px;margin:auto;padding:20px}}header{{background:linear-gradient(120deg,#163560,#3069ab);color:#fff;border-radius:16px;padding:30px;box-shadow:0 8px 24px #122a4530}}h1{{margin:0;font-size:clamp(25px,4vw,40px);line-height:1.2}}h2{{margin:0 0 10px;font-size:24px}}h3{{margin:14px 0 6px;font-size:17px}}section{{margin-top:18px;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;box-shadow:0 3px 12px #1231}}.lead{{font-size:17px;max-width:1000px}}.meta,.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.card{{background:color-mix(in srgb,var(--card) 85%,var(--accent));border:1px solid var(--line);padding:14px;border-radius:10px}}.status{{font-weight:800;font-size:22px}}.red{{color:var(--red)}}.green{{color:var(--green)}}.small{{color:var(--muted);font-size:13px}}.plot{{width:100%;height:570px;border:1px solid var(--line);border-radius:10px}}.plot.short{{height:360px}}.controls{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:10px 0}}button,select,label{{font:inherit}}button,select{{padding:7px 10px;border:1px solid var(--line);border-radius:7px;background:var(--card);color:var(--text)}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border-bottom:1px solid var(--line);padding:7px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{position:sticky;top:0;background:var(--card);z-index:1}}.table-scroll{{max-height:500px;overflow:auto;border:1px solid var(--line);border-radius:8px}}code{{font-size:10px;word-break:break-all;white-space:normal}}.callout{{border-left:5px solid var(--accent);padding:12px 15px;background:color-mix(in srgb,var(--card) 90%,var(--accent));border-radius:0 8px 8px 0}}.warn{{border-left-color:var(--red)}}.legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:13px}}.swatch{{display:inline-block;width:13px;height:3px;vertical-align:middle;margin-right:4px}}@media(max-width:600px){{main{{padding:10px}}section{{padding:14px}}header{{padding:20px}}.plot{{height:420px}}th,td{{padding:5px}}}}
</style></head><body><main><header><div class='small' style='color:#dce9ff'>BACRA V14.3R · execution attempt3 · evidence cutoff {datetime.now(timezone.utc).date().isoformat()} UTC</div><h1>Zero-tip 补域与 parity-smooth Student：分轴诊断报告</h1><p class='lead'>统一呈现真实封存数据集、Teacher、冻结 Raw Student 及 DLS2 的空间与轨迹证据。该 attempt 已 operationally complete，但结论严格停留在 diagnostic-only 边界。</p><div class='meta'><div>实验：<b>{identity['experiment_id']}</b></div><div>执行 source：<code>{SOURCE_SHA}</code></div><div>binding：<code>{BINDING_SHA}</code></div><div>锁定 Student：<b>{selected['model_id']}</b></div></div></header>
<section><div class='controls'><h2 style='flex:1'>摘要与边界</h2><button id='theme' aria-label='切换深色模式'>切换深色模式</button></div><div class='cards'><div class='card'><div class='small'>Data axis</div><div class='status red'>RED</div><div>最近 zero 非零标签 {data_axis['nearest_zero_nonzero_mm']:.3f} mm（阈值 10）<br>tip fill P95 {data_axis['tip_fill']['p95_mm']:.3f} mm<br>LCC {data_axis['lcc']:.6f}（要求 1）</div></div><div class='card'><div class='small'>Raw Student axis</div><div class='status red'>RED</div><div>锁定模型 {selected['model_id']}；raw-first 选择，DLS2 未参与主选择。</div></div><div class='card'><div class='small'>DLS2 axis</div><div class='status green'>GREEN</div><div>仅逐 waypoint 数值诊断，<b>不等同真实闭环控制</b>。</div></div><div class='card'><div class='small'>Formal claim</div><div class='status red'>未授权</div><div>formal / deployment / downstream / full & continuous workspace 均为 false。</div></div></div><p class='callout warn'>本页的曲线、DLS2 可视化和浏览器检查不重算 sealed Gate，也不证明数据集 formal-ready、Student 可部署或机器人获得动态闭环能力。</p></section>
<section><h2>1. 统一 quotient / full-G4 数据集与零位</h2><p>三维主视图使用机器人基座物理坐标（mm），未为排版平移。显示 full-G4 <b>{len(display_full):,}/{len(full):,}</b> 点与 quotient <b>{len(display_q):,}/{len(quotient):,}</b> 点；按 sealed 的 domain / role 分层稳定抽样，完整分母保留在统计中。紫色环列为 zero-tip frozen profile 的真实外边界；红线与标记为 beta=0 机器人中心线和零位末端。</p><div id='coverage' class='plot' aria-label='可旋转三维 unified dataset 覆盖图'></div><div class='legend'><span><i class='swatch' style='background:#6554c0'></i>full-G4</span><span><i class='swatch' style='background:#45a2a2'></i>quotient</span><span><i class='swatch' style='background:#9c5dc2'></i>zero-tip profile</span><span><i class='swatch' style='background:#d63e3e'></i>beta=0 robot</span></div><h3>axis-aware support 与 Data RED</h3><p>axis core 使用独立 pool A/B 的共同支持，避免柱坐标奇异性制造假 zero hole；support artifact 共 {len(support):,} 个 (u, rho, sector) 记录，frozen profile 含 {len(profile)} 个 axial bin。尽管 root-to-annulus 图连通度高，Gate 同时要求 LCC=1、最近非零点不超过 10 mm 与 tip fill P95 不超过 10 mm，因此 {data_axis['lcc']:.6f}、{data_axis['nearest_zero_nonzero_mm']:.3f} mm、{data_axis['tip_fill']['p95_mm']:.3f} mm 的组合仍是 Data RED。</p></section>
<section><h2>2. Parity label、Teacher seam 与因果面板</h2><p>exact y=0 / z=0 标签兼容性为 <b>{parity_ok}/{len(compatibility)}</b>；但 Teacher odd-vanish exponent 只有 <b>{alpha_in_band}/{alpha_total}</b> 个有效 component 落在授权区间 [0.75, 1.25]，所以 parity architecture 未获授权。本 run 仅训练 S0/S1/S4，锁定的 S1_seam_root 仍为标准模型。</p><div id='seam' class='plot short' aria-label='Teacher seam sweep 与 Student Teacher 因果诊断'></div><p class='small'>seam teacher sweep：{len(sweeps):,} 行；Student–Teacher causal panel：{len(causal):,} 行。图中为每个 normal offset 的跨 anchor 中位数：左轴 Raw Student FK residual，右轴 Student–Teacher weighted difference；它比较同一 evaluation-only seam 点上的 frozen Student 与 ordered canonical Teacher。</p>{table_rows(exponent_summary, ['seam','components','alpha_min','alpha_median','alpha_max'])}<h3>锁定 Student 的训练/选择</h3><p>{selected['model_id']} 的选择优先级：<code>{selected['selection_priority']}</code>。它的 selection score 为 {selected['selection_score']:.3f}，但 raw_green=false；下表显示并没有以 DLS2 的较优数值投影覆盖 raw 指标。</p>{table_rows(selection_metrics, list(selection_metrics.columns), {c:'.3f' for c in selection_metrics.select_dtypes(include=[np.number]).columns})}</section>
<section><h2>3. 长方形回归与 post-lock held-out 轨迹</h2><p>旧 rounded-rectangle 是 retry17 regression panel；新 trajectory 在 dataset/model lock 后才生成，且未回流训练。下方三维场景可以切换 15 条 post-lock trajectory，并分别开关 Target、Teacher、Raw Student、DLS2。所有轨迹图均在同一物理 frame 中保留 beta=0 robot 与零位末端，不进行可见性坐标偏移。</p><div class='controls'><label>轨迹 <select id='trajectory'>{''.join(f"<option value='{html.escape(k)}'>{html.escape(k)} · {html.escape(v['shape'])}</option>" for k,v in trajectory_data.items())}</select></label><label><input type='checkbox' data-method='target' checked> Target</label><label><input type='checkbox' data-method='teacher' checked> Teacher</label><label><input type='checkbox' data-method='raw' checked> Raw Student</label><label><input type='checkbox' data-method='dls2' checked> DLS2</label><button id='reset'>重置 3D 视角</button></div><div id='trajectory-plot' class='plot' aria-label='可交互三维 Target Teacher Raw Student DLS2 跟踪图'></div><div class='legend'><span><i class='swatch' style='background:#20242a'></i>Target：虚线</span><span><i class='swatch' style='background:#e2772b'></i>Teacher：橙色宽线</span><span><i class='swatch' style='background:#1775c1'></i>Raw Student：蓝线</span><span><i class='swatch' style='background:#168b62'></i>DLS2：绿色线</span></div><h3>旧 rectangle regression（非 held-out）</h3>{table_rows(old_rectangle, list(old_rectangle.columns), {c:'.3f' for c in old_rectangle.select_dtypes(include=[np.number]).columns})}<h3>小揪揪是否仍存在？</h3><p class='callout warn'>存在：post-lock <b>retry17_rounded_rectangle_1</b> 的 Raw Student path-step excess maximum 为 <b>{float(rectangle_new.loc[rectangle_new['trajectory_id'].eq('retry17_rounded_rectangle_1'),'raw_student_path_step_excess_maximum_mm'].iloc[0]):.3f} mm</b>，仍超过 5 mm 阻断线；该路径 Raw FK P95 为 {float(rectangle_new.loc[rectangle_new['trajectory_id'].eq('retry17_rounded_rectangle_1'),'raw_student_fk_p95_mm'].iloc[0]):.3f} mm。另一个明显异常为 rounded_star_0 的 Raw path-step excess maximum 18.714 mm。DLS2 GREEN 不能覆盖 Raw Student RED。</p><h3>逐轨迹三方法指标</h3><p class='small'>每行含 Teacher completeness；Teacher / Raw / DLS2 的 FK P95/max、path-step excess P99/max 及 raw beta step&gt;7° rate。数值直接来自 sealed trajectory_report.parquet。</p>{table_rows(trajectories, metric_cols, {c:'.4f' for c in trajectories.select_dtypes(include=[np.number]).columns})}</section>
<section><h2>4. Dataset / model lock、生成顺序与可追溯性</h2><p>lock 明确记录 <b>locked_before_heldout_generation={lock['locked_before_heldout_generation']}</b>。先锁定 unified quotient、full-G4、S1_seam_root model tree 与 manifest，后生成 held-out trajectories；报告仅以这些封存输入进行确定性 FK replay。所有输入封印检查均通过：summary manifest {seals['summary_artifact_entries']} 项、{seals['completion_manifest_count']} 个 completion manifest，mismatch 均为 0。</p><div class='callout'>生成顺序：<b>05 unified dataset → 09 selected Student → 10 dataset/model lock → 11 held-out generation → 12 evaluation → 13 summary → report（本目录）</b>。report 是派生诊断物，不是 stage artifact，未写入 summary manifest。</div><h3>关键输入 hash</h3><div class='table-scroll'><table><thead><tr><th>路径</th><th>bytes</th><th>SHA-256</th></tr></thead><tbody>{key_input_rows}</tbody></table></div><p class='small'>完整 input manifest、evidence map、浏览器验证和本报告自身 hash 位于同一 report/ 目录。</p></section>
</main><script>{get_plotlyjs()}</script><script>const D={payload_json}; const C={{target:'#20242a',teacher:'#e2772b',raw:'#1775c1',dls2:'#168b62',robot:'#d63e3e'}}; const base={{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{color:getComputedStyle(document.body).getPropertyValue('--text')}},margin:{{l:8,r:8,t:45,b:8}}}};
function lines(points,name,color,dash,width){{return {{type:'scatter3d',mode:'lines',name,x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),line:{{color,width,dash}},hovertemplate:name+'<br>x=%{{x:.1f}} mm<br>y=%{{y:.1f}} mm<br>z=%{{z:.1f}} mm<extra></extra>'}}}}
function robot(){{return lines(D.zero_line,'beta=0 robot',C.robot,'solid',7)}}
function coverage(){{const g=(arr,name,color)=>({{type:'scatter3d',mode:'markers',name,x:arr.map(p=>p[0]),y:arr.map(p=>p[1]),z:arr.map(p=>p[2]),text:arr.map(p=>p[3]),marker:{{size:2,color,opacity:.45}},hovertemplate:name+'<br>%{{text}}<br>x=%{{x:.1f}} mm<br>y=%{{y:.1f}} mm<br>z=%{{z:.1f}} mm<extra></extra>'}});const p=D.profile;const traces=[g(D.full,'full-G4 (sampled)','#6554c0'),g(D.quotient,'quotient (sampled)','#45a2a2'),lines(p,'zero-tip profile outer boundary','#9c5dc2','solid',2),robot(),{{type:'scatter3d',mode:'markers',name:'zero end-effector',x:[D.zero_end[0]],y:[D.zero_end[1]],z:[D.zero_end[2]],marker:{{size:7,color:C.robot,symbol:'diamond'}}}}];Plotly.newPlot('coverage',traces,{{...base,title:'Unified quotient / full-G4 coverage（真实 XYZ，mm）',scene:{{xaxis:{{title:'X (mm)'}},yaxis:{{title:'Y (mm)'}},zaxis:{{title:'Z (mm)' }},aspectmode:'data',camera:{{eye:{{x:1.55,y:1.35,z:.95}}}}}},legend:{{orientation:'h'}}}},{{responsive:true,displaylogo:false}})}}
function seam(){{const ys=D.seam.rows.filter(r=>r.seam==='y'),zs=D.seam.rows.filter(r=>r.seam==='z');const t=[];for(const [name,a,col] of [['y seam',ys,'#e2772b'],['z seam',zs,'#1775c1']]){{t.push({{type:'scatter',mode:'lines+markers',name:name+' Raw FK',x:a.map(r=>r.normal_offset_mm),y:a.map(r=>r.student_fk_residual_mm),line:{{color:col}}}});t.push({{type:'scatter',mode:'lines',name:name+' Student–Teacher',x:a.map(r=>r.normal_offset_mm),y:a.map(r=>r.student_teacher_weighted_deg),yaxis:'y2',line:{{color:col,dash:'dot'}}}})}}Plotly.newPlot('seam',t,{{...base,title:'seam normal sweep：Raw FK residual 与 Student–Teacher difference',xaxis:{{title:'normal offset (mm)'}},yaxis:{{title:'median Raw FK residual (mm)'}},yaxis2:{{title:'median weighted difference (deg)',overlaying:'y',side:'right'}},legend:{{orientation:'h'}}}},{{responsive:true,displaylogo:false}})}}
function trajectory(reset=false){{const id=document.getElementById('trajectory').value,q=D.trajectory[id],visible=m=>document.querySelector('[data-method="'+m+'"]').checked;const t=[];if(visible('target'))t.push(lines(q.data.target,'Target',C.target,'dash',5));if(visible('teacher'))t.push(lines(q.data.teacher,'Teacher',C.teacher,'solid',8));if(visible('raw'))t.push(lines(q.data.raw,'Raw Student',C.raw,'solid',4));if(visible('dls2'))t.push(lines(q.data.dls2,'DLS2 diagnostic',C.dls2,'solid',3));t.push(robot());t.push({{type:'scatter3d',mode:'markers',name:'zero end-effector',x:[D.zero_end[0]],y:[D.zero_end[1]],z:[D.zero_end[2]],marker:{{size:7,color:C.robot,symbol:'diamond'}}}});Plotly.react('trajectory-plot',t,{{...base,title:id+' · '+q.shape+' · '+q.kind,scene:{{xaxis:{{title:'X (mm)'}},yaxis:{{title:'Y (mm)'}},zaxis:{{title:'Z (mm)'}},aspectmode:'data',camera:{{eye:reset?{{x:1.55,y:1.35,z:.95}}:{{x:1.55,y:1.35,z:.95}}}}}},legend:{{orientation:'h'}}}},{{responsive:true,displaylogo:false}})}}
coverage();seam();trajectory();document.getElementById('trajectory').onchange=()=>trajectory();document.querySelectorAll('[data-method]').forEach(x=>x.onchange=()=>trajectory());document.getElementById('reset').onclick=()=>trajectory(true);document.getElementById('theme').onclick=()=>{{document.body.classList.toggle('dark');coverage();seam();trajectory()}};</script></body></html>"""
    # The page is intentionally dependency-free: use the small Canvas renderer
    # below instead of an external Plotly/WebGL CDN.  Disable the provisional
    # Plotly script retained in the readable HTML template, then replace its
    # chart hosts with keyboard-focusable canvases and append the renderer.
    html_text = html_text.replace("<script></script><script>", "<script type='application/x-disabled'>")
    html_text = html_text.replace("</title>", "</title><link rel='icon' href='data:,'>", 1)
    html_text = html_text.replace("<div id='coverage' class='plot' aria-label='可旋转三维 unified dataset 覆盖图'></div>", "<div class='controls'><label><input id='show-full' type='checkbox' checked> full-G4</label><label><input id='show-quotient' type='checkbox' checked> quotient</label><button id='coverage-reset'>重置 3D 视角</button></div><canvas id='coverage' class='plot' tabindex='0' aria-label='可旋转三维 unified dataset 覆盖图'></canvas>")
    html_text = html_text.replace("<div id='seam' class='plot short' aria-label='Teacher seam sweep 与 Student Teacher 因果诊断'></div>", "<canvas id='seam' class='plot short' tabindex='0' aria-label='Teacher seam sweep 与 Student Teacher 因果诊断'></canvas>")
    html_text = html_text.replace("<div id='trajectory-plot' class='plot' aria-label='可交互三维 Target Teacher Raw Student DLS2 跟踪图'></div>", "<canvas id='trajectory-plot' class='plot' tabindex='0' aria-label='可交互三维 Target Teacher Raw Student DLS2 跟踪图'></canvas>")
    canvas_js = r"""
<style>body{overflow-x:hidden}section,.meta>*{min-width:0}.meta>*{overflow-wrap:anywhere}.meta code{font-size:10px}.meta b{overflow-wrap:anywhere}.table-scroll{width:100%;max-width:100%;min-width:0;overflow-x:auto}table{min-width:max-content}.controls label{max-width:100%;min-width:0}.controls select{max-width:100%;width:min(520px,100%);overflow:hidden;text-overflow:ellipsis}code{max-width:100%;overflow-wrap:anywhere}</style><script>
const D = __REPORT_DATA__;
const COL = {target:'#20242a',teacher:'#e2772b',raw:'#1775c1',dls2:'#168b62',robot:'#d63e3e',profile:'#9c5dc2'};
function colour(name){ return getComputedStyle(document.body).getPropertyValue(name).trim(); }
function draw3d(canvas, provider) {
  let yaw=-0.78, pitch=0.36, zoom=1, down=null;
  const ctx=canvas.getContext('2d');
  function allPoints(data) { return data.points || data.target || []; }
  function bounds(points) { const mn=[Infinity,Infinity,Infinity], mx=[-Infinity,-Infinity,-Infinity]; points.forEach(p=>{for(let i=0;i<3;i++){mn[i]=Math.min(mn[i],p[i]);mx[i]=Math.max(mx[i],p[i]);}}); return {c:mn.map((v,i)=>(v+mx[i])/2),e:Math.max(...mx.map((v,i)=>v-mn[i]),1)}; }
  function project(p,w,h,c,e) { const x=p[0]-c[0],y=p[1]-c[1],z=p[2]-c[2],cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),a=cy*x-sy*y,b=sy*x+cy*y,d=cp*z-sp*b,depth=cp*b+sp*z,s=Math.min(w,h)*.44/e*zoom/(1+depth/(e*5)); return [w/2+a*s,h/2-d*s,depth]; }
  function line(points,col,width,dash,c,e,w,h,closed,alpha=1) { if(!points.length)return;ctx.save();ctx.strokeStyle=col;ctx.lineWidth=width;ctx.globalAlpha=alpha;ctx.lineCap='round';ctx.lineJoin='round';ctx.setLineDash(dash||[]);ctx.beginPath();points.forEach((p,i)=>{const q=project(p,w,h,c,e);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])});if(closed){const q=project(points[0],w,h,c,e);ctx.lineTo(q[0],q[1])}ctx.stroke();ctx.restore(); }
  function draw() { const r=canvas.getBoundingClientRect(), dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(r.width*dpr)),h=Math.max(1,Math.round(r.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}ctx.clearRect(0,0,w,h);ctx.fillStyle=colour('--card');ctx.fillRect(0,0,w,h);const data=provider(),base=allPoints(data),b=bounds(base),c=b.c,e=b.e;const axes=[[c[0]+e*.25,c[1],c[2]],[c[0],c[1]+e*.25,c[2]],[c[0],c[1],c[2]+e*.25]];['#d95555','#2b9d70','#3679d6'].forEach((x,i)=>line([c,axes[i]],x,2*dpr,[],c,e,w,h,false));ctx.font=`${12*dpr}px sans-serif`;ctx.fillStyle=colour('--text');['X','Y','Z'].forEach((x,i)=>{const q=project(axes[i],w,h,c,e);ctx.fillText(x,q[0]+5*dpr,q[1]-5*dpr)});
    if(data.points){const source=[];if(document.getElementById('show-full').checked) source.push(...D.full.map(p=>[p,'#6554c0']));if(document.getElementById('show-quotient').checked)source.push(...D.quotient.map(p=>[p,'#45a2a2']));source.map(a=>[a,project(a[0],w,h,c,e)]).sort((a,b)=>a[1][2]-b[1][2]).forEach(([a,q])=>{ctx.fillStyle=a[1];ctx.globalAlpha=.48;ctx.beginPath();ctx.arc(q[0],q[1],1.25*dpr,0,Math.PI*2);ctx.fill()});ctx.globalAlpha=1;line(D.profile,COL.profile,1.3*dpr,[],c,e,w,h,false,.7);line(D.zero_line,COL.robot,4*dpr,[],c,e,w,h,false);const q=project(D.zero_end,w,h,c,e);ctx.fillStyle=COL.robot;ctx.beginPath();ctx.arc(q[0],q[1],4*dpr,0,Math.PI*2);ctx.fill();ctx.fillText('zero end-effector',q[0]+7*dpr,q[1]-5*dpr);
    } else { const vis=m=>document.querySelector('[data-method="'+m+'"]').checked;if(vis('teacher'))line(data.teacher,COL.teacher,7*dpr,[],c,e,w,h,data.closed,.96);if(vis('raw'))line(data.raw,COL.raw,4*dpr,[],c,e,w,h,data.closed);if(vis('dls2'))line(data.dls2,COL.dls2,2.5*dpr,[],c,e,w,h,data.closed);if(vis('target'))line(data.target,COL.target,2*dpr,[8*dpr,5*dpr],c,e,w,h,data.closed);line(D.zero_line,COL.robot,4*dpr,[],c,e,w,h,false);const q=project(D.zero_end,w,h,c,e);ctx.fillStyle=COL.robot;ctx.beginPath();ctx.arc(q[0],q[1],4*dpr,0,Math.PI*2);ctx.fill(); }
    ctx.fillStyle=colour('--muted');ctx.fillText('robot-base physical coordinates (mm) · drag rotate · wheel zoom',12*dpr,20*dpr);canvas.dataset.sceneReady='true'; }
  canvas.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!down)return;yaw+=(e.clientX-down[0])*.008;pitch=Math.max(-1.35,Math.min(1.35,pitch+(e.clientY-down[1])*.008));down=[e.clientX,e.clientY];draw()});canvas.addEventListener('pointerup',()=>down=null);canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.45,Math.min(3.5,zoom*(e.deltaY>0?.9:1.1)));draw()},{passive:false});canvas.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')yaw-=.12;if(e.key==='ArrowRight')yaw+=.12;if(e.key==='ArrowUp')pitch-=.12;if(e.key==='ArrowDown')pitch+=.12;draw()}); return {draw,reset:()=>{yaw=-.78;pitch=.36;zoom=1;draw()}};
}
function seamChart() { const c=document.getElementById('seam'),ctx=c.getContext('2d'),r=c.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2),w=Math.round(r.width*dpr),h=Math.round(r.height*dpr);c.width=w;c.height=h;ctx.fillStyle=colour('--card');ctx.fillRect(0,0,w,h);const rows=D.seam.rows, x0=48*dpr,x1=w-48*dpr,y0=35*dpr,y1=h-36*dpr,x=v=>x0+(v+30)/60*(x1-x0),fy=v=>y1-v/70*(y1-y0),fd=v=>y1-v/.7*(y1-y0);ctx.strokeStyle=colour('--line');for(let n=-30;n<=30;n+=15){ctx.beginPath();ctx.moveTo(x(n),y0);ctx.lineTo(x(n),y1);ctx.stroke();ctx.fillStyle=colour('--muted');ctx.fillText(String(n),x(n)-7*dpr,y1+17*dpr)}ctx.fillText('normal offset (mm)',w/2-45*dpr,h-8*dpr);[['y','#e2772b'],['z','#1775c1']].forEach(([s,col])=>{const a=rows.filter(r=>r.seam===s);ctx.strokeStyle=col;ctx.lineWidth=2*dpr;ctx.beginPath();a.forEach((r,i)=>i?ctx.lineTo(x(r.normal_offset_mm),fy(r.student_fk_residual_mm)):ctx.moveTo(x(r.normal_offset_mm),fy(r.student_fk_residual_mm)));ctx.stroke();ctx.setLineDash([5*dpr,3*dpr]);ctx.beginPath();a.forEach((r,i)=>i?ctx.lineTo(x(r.normal_offset_mm),fd(r.student_teacher_weighted_deg)):ctx.moveTo(x(r.normal_offset_mm),fd(r.student_teacher_weighted_deg)));ctx.stroke();ctx.setLineDash([])});ctx.fillStyle=colour('--muted');ctx.fillText('solid: median Raw FK residual (0–70 mm)',x0,y0-16*dpr);ctx.fillText('dotted: Student–Teacher weighted difference (0–0.7 deg)',x0+300*dpr,y0-16*dpr);c.dataset.sceneReady='true'; }
let trajectoryId=document.getElementById('trajectory').value;const coverage=draw3d(document.getElementById('coverage'),()=>({points:[...D.full,...D.quotient]}));const trajectory=draw3d(document.getElementById('trajectory-plot'),()=>{const q=D.trajectory[trajectoryId];return {...q.data,closed:q.shape!=='profile_ray'}});document.getElementById('coverage-reset').onclick=coverage.reset;document.getElementById('show-full').onchange=coverage.draw;document.getElementById('show-quotient').onchange=coverage.draw;document.getElementById('trajectory').onchange=e=>{trajectoryId=e.target.value;trajectory.reset()};document.querySelectorAll('[data-method]').forEach(x=>x.onchange=trajectory.draw);document.getElementById('reset').onclick=trajectory.reset;document.getElementById('theme').onclick=()=>{document.body.classList.toggle('dark');coverage.draw();trajectory.draw();seamChart()};window.addEventListener('resize',()=>{coverage.draw();trajectory.draw();seamChart()});coverage.draw();trajectory.draw();seamChart();window.__retry18Attempt3={coverage,trajectory};
</script>"""
    canvas_js = canvas_js.replace("__REPORT_DATA__", payload_json)
    html_text = html_text.replace("</body></html>", canvas_js + "</body></html>")
    html_path = OUT / HTML_NAME
    html_path.write_text(html_text, encoding="utf-8")
    report_manifest = {"schema_version": 1, "kind": "derived_diagnostic_report", "report": HTML_NAME,
                       "source_fixed_point": SOURCE_SHA, "binding_fixed_point": BINDING_SHA,
                       "summary_gate": {"path": "../13_summary/gate.json", "sha256": sha256(RUN / "13_summary/gate.json")},
                       "input_manifest": {"path": "input_manifest.json", "sha256": sha256(OUT / "input_manifest.json")},
                       "evidence_map": {"path": "evidence_map.json", "sha256": sha256(OUT / "evidence_map.json")},
                       "html": {"path": HTML_NAME, "bytes": html_path.stat().st_size, "sha256": sha256(html_path)},
                       "claim_boundary": evidence_map["claim_boundary"]}
    browser_validation = OUT / "browser_validation.json"
    if browser_validation.is_file():
        report_manifest["browser_validation"] = {"path": browser_validation.name, "sha256": sha256(browser_validation)}
    (OUT / "report_manifest.json").write_text(json.dumps(report_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
