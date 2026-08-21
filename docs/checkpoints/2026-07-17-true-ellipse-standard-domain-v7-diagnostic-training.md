---
title: True Ellipse Standard-Domain V7-D1 Diagnostic Training Checkpoint
date: 2026-07-17
status: frozen
tags:
  - quasi-exp
  - checkpoint
  - true-ellipse
  - diagnostic-training
  - beta6
  - visualization
---

# True Ellipse Standard-Domain V7-D1 Diagnostic Training Checkpoint

> [!warning] 非正式边界
> 本 checkpoint 只固化 centerline-only 诊断实验。`diagnostic_only=true`、`formal_claims_allowed=false`、`changes_v7_formal_gate=false`、`static_inverse_claim_radius_mm=null`。它不修改正式 V7 checkpoint。

## Git 与文档边界

- 分支：`codex/true-ellipse-v7-standard-domain-120mm`；
- 实现固定点起点：`85730c92ba38cf1c0a1e4edeb2ac55b457633dac`；
- 详细记录：[[../TrueEllipseStandardDomainV7诊断中心线模型实验|V7-D1 诊断中心线模型实验]]；
- 忽略产物：`runs/true_ellipse_standard_domain_training_v7_diagnostic_centerline/`。

## 数据与协议

```text
registered continuous centerlines  35 trajectories / 12,600 rows
screen train                       30 radii <=100 / 10,800 rows
screen validation                  102.5 mm / 360 integer + 360 half phase
post-selection only                101, 102, 103.5, 104 mm
screen grid                        24 base configs × 2 links = 48
screen seed                        20260711
stability seeds                    20260711..20260715
max_iter                           800
dense evaluation                   720 points/radius
```

101/102/103.5/104 mm 在 selection report 冻结后才打开，未进入 screen parquet。

## 结果

```text
selected config
  mlp_beta6_wide_poly_medium_relu_a1em04__identity

screen dual-validation eligible    21/48
  identity                          9/24
  tanh_bounds                      12/24

diagnostic stable accepted radii
  100, 101, 102, 102.5 mm

diagnostic failed stability
  103.5 mm  2/5 seeds pass
  104.0 mm  1/5 seeds pass

raw bound violations               0
minimum predicted margin            about 1.96 deg
formal model gate                   false
formal claim radius                 null
```

103.5/104 mm 的失败主因是 `axiserr_max_p95_abs_mm>3`，部分 seed 同时触发 EE P95/max；不是 beta3/beta4 越界。

## 图像

- 48 张 `individual_models/<config_id>.png`：每张包含 100/102.5/104 mm 正视 3D、主平面投影与逐轴误差；
- `contact_sheet_identity.png` 与 `contact_sheet_tanh_bounds.png`：各 24 个模型；
- `screen_model_metric_bars.png`；
- `screen_model_error_heatmap.png`；
- `identity_vs_tanh_bounds.png`；
- `selected_config_seed_comparison.png`；
- `selected_config_unsupported_extrapolation.png`。

相机策略为目标轨迹 SVD 平面法向 + 12° 方位偏移，并同时显示主平面投影；人工检查未发现椭圆被侧视压成直线。

## 关键哈希

| 产物 | SHA-256 |
|---|---|
| screen dataset | `b2e5e2a63b272d556bde6919d37d13ab41b1b0f48550be1e846b35e535601427` |
| selection report | `9bcf2654a485f7c03c800909de96707fe153e82857305be2dea32a34cfed284a` |
| stability report | `c251c612c350de76cae8b4d486e66010c910f5b718228cc3bf50b2d1d57a14b2` |
| evaluation report | `601dc655bf6cf950954ba2f8e97357bb2582332dc1fd3bcf87d47fa3fa1073c7` |
| visualization report | `783735c3d1df19521b71399fc1cf09a5236876a2de0bbbdd170e03de4c8975a0` |
| final report | `3f82911888dd4cbbe9770ad5c58e0ada425829f31a76c884a4d4ecb1be94c4cb` |
| accepted-radius summary | `8193e648e53e1bbe04baf7e2945bf2a5113fc85fca671689833f16f0c1cfb4c7` |

## 验证边界

- smoke：2 configs / 30 iter / 1 seed，端到端通过；
- pilot：8 configs / 300 iter / 3 seeds，端到端通过；
- full：48 configs / 800 iter / 5 seeds，端到端通过；
- visualization integrity：`true`；
- V7/V7-D1 定向回归：32 项通过；
- 全仓回归：391 项通过；
- 4 个相关脚本 `py_compile` 通过，`git diff --check` 通过；
- 产物审计通过：48 个 screen 模型、5 个稳定性模型、179 份预测轨迹、55 张 PNG 及 9 个报告附属产物的内容哈希均与权威报告一致；
- Standards consolidated re-review 已通过；Spec 限定复核已关闭 F1/F2/F4，最终结论为 `Pass`，无开放 finding；
- 当前 worktree 尚未提交，等待分支交付方式选择。
