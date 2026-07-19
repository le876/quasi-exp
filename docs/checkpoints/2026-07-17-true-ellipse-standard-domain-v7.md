---
title: True Ellipse Standard-Domain V7 Checkpoint
date: 2026-07-17
status: frozen
tags:
  - quasi-exp
  - checkpoint
  - true-ellipse
  - standard-domain
  - radial-bundle
  - beta6
---

# True Ellipse Standard-Domain V7 Checkpoint

> [!abstract] 固化目标
> V7 在固定 family `c0273_a100_py210_pz330_s0243` 上使用项目标准关节域，把 joint-margin、联合 tube surface、动态 support-backed holdout、half-phase challenge 与 identity/tanh-bounds 48 配置模型协议纳入同一条 fail-closed 链。探索上限为 120 mm；只有 strict radial/tube/dataset checkpoint 至少达到 105 mm 才能授权正式训练。

## Git 边界

- 分支：`codex/true-ellipse-v7-standard-domain-120mm`；
- V7 起点：`14bce8995e8478f5cbd5a410de33fc4b70e79982`；
- 结果文档：[[../TrueEllipseStandardDomainV7实验记录|True Ellipse Standard-Domain V7 实验记录]]；
- 大体积产物：`runs/true_ellipse_standard_domain_v7/`，继续由 `.gitignore` 排除。

## 注册协议

```text
family               c0273_a100_py210_pz330_s0243
joint domain          beta1/2 ±5°, beta3/4 ±10°, beta5/6 ±15°
strict minimum        105 mm
exploration target    120 mm
radial predictors     parent_copy + radial_secant
cyclic cuts           0, 90, 180, 270
margin gate           min/P01/P05 = 0.05/0.10/0.25 deg
tube grid             360 × 5 × 5
holdout               Rtest=max support-backed strict checkpoint >=105
validation            Rtest - 7.5 mm
challenge             integer + (k+0.5)° half phase
model screen          24 base configs × 2 output links = 48
formal seeds          20260711..20260715
```

## 正式结果

| 阶段 | 结果 | Gate |
|---|---|---|
| audit | fixed family、标准关节域、V6 source manifest、robot config 与声明 runtime 全部通过；protocol fingerprint=`10bcc17f…29c3a` | `formal_audit_gate_pass=true` |
| radial strict | 注册 checkpoint 通过到 102.5 mm；连续中间路径通过到 104.0 mm；104.25/104.5/105 mm direct+candidate rescue 失败；首个 strict failure=105 mm | `formal_radial_gate_pass=false` |
| radial exploratory | 105 mm 72/72 residual≤2 mm；120 mm success ratio=`0.847222`、P95/max=`15.232731/17.015809 mm`，仍过 admission | `exploratory_rescue_rmax_mm=120`，不构成 strict 证书 |
| tube | radial strict minimum 未达到 105 mm，未物化 tube | `formal_tube_gate_pass=false`，`strict_tube_rmax_mm=null` |
| dataset/challenge | 未注册动态 holdout，未生成正式数据集或 half-phase challenge | `formal_dataset_gate_pass=false` |
| model | 48-config screen 与五 seed final 未启动 | `model_training_authorized=false` |
| visualization | 按现有 Matplotlib/Agg 方式生成 strict/continuous、joint、conditioning、exploratory 四张 180-dpi PNG；3D 视角由轨迹平面法向自动确定 | `visualization_integrity_gate_pass=true`，不改变 formal gate |

关键边界：

```text
100 mm margin min/P01/P05    1.979084 / 1.980657 / 2.011967 deg
102.5 mm kappa P95           91.8995
104.0 mm kappa P95           142.5808   pass
104.25 mm best direct kappa  183.5939   fail
105.0 mm direct kappa        361.9766..441.9038 fail
```

标准域已把 V6 beta3/beta4 的贴界标签改造成约 2° 的内点标签；新的 strict 瓶颈是 104.0–104.25 mm 间的 Jacobian 近奇异性。V7 没有获得可授权训练的 `Rtest≥105 mm`，所以静态逆模型半径不作新声明。

## 轨迹可视化

输出目录：`runs/true_ellipse_standard_domain_v7/05_visualization/`。

- `strict_tracking_overview.png`：100/102.5/104 mm 的 3D、x-y、y-z 和逐轴误差；
- `strict_joint_profiles.png`：六个 beta 随相位变化及注册关节域；
- `conditioning_frontier.png`：104 pass 对比 104.25/105 conditioning failure；
- `exploratory_pointwise_overview.png`：105/110/115/120 mm 独立 pointwise FK，绿/红表示 residual 是否 ≤2 mm；
- `visualization_report.json`：全部输入/输出 SHA、证据类别和非 gating 声明。

3D 相机由目标轨迹 SVD 平面法向计算，并偏转 12°，因此展示的是近正视椭圆而不是侧视直线。探索点不连线，且不能解释为连续 branch。

## 审查边界

- Standards 与 Spec 双轴初审已完成；
- 一次 consolidated re-review 已完成；
- 高优先级 finding 已批量修复：运动学 cache fingerprint、四阶段上游/产物 SHA、non-formal test 物理隔离、动态 test 外层截断、V6 per-cut rescue 冻结语义、低于 105 mm 的下游短路、正式 runtime fingerprint；
- V7 runner 按 phase 拆分作为非阻断架构债保留。

## 声明环境

```text
Python 3.11.5
numpy 1.26.4
scipy 1.11.4
pandas 2.2.2
pyarrow 16.1.0
scikit-learn 1.3.0
joblib 1.2.0
matplotlib 3.10.0
pytest 8.2.2
```

正式 protocol fingerprint 会校验前五项；环境不匹配时 `formal_protocol_gate_pass=false`。

## 验证状态

- 正式 Python 3.11 radial：exit 0，`ELAPSED=4445.96 s`，`formal_radial_gate_pass=false`；
- fail-closed tube/dataset/summary：exit 0，未物化下游数据；
- Python 3.11 全仓回归：`368 passed, 314 warnings`；新增 warning 均来自 Matplotlib/pyparsing 的弃用提示；
- V7 下游 blocked-frontier 定向回归：`19 passed`；
- Standards 初审：无文档化 hard violation；S2/S3 已解决，runner 拆分保留为低优先级债；
- Spec consolidated re-review 的 P1/P3 遗留已在后续提交中修复，P2/P4 已由 reviewer 确认 resolved。

## 关键忽略产物

| 相对路径（以 `runs/true_ellipse_standard_domain_v7/` 为根） | SHA-256 |
|---|---|
| `00_audit/audit_report.json` | `0ddbcb51c847fd7e3d64bda352703aca7e497b1c386369e6d0fd0314920ad1e9` |
| `01_radial/radial_report.json` | `d695e8548044b82998ede890c203d0bb3aaaa08f2461d7211df9f2126882c8dc` |
| `01_radial/radius_status.csv` | `75ade74eb4feb236da970f9c4521cdbbeb8f60cfe395830965e11bc02cb6061d` |
| `01_radial/strict_path_manifest.json` | `36c1f7db30bbe3f021e1db5483c56b78bdabc2f455e481c12f62b5d4e51d5aec` |
| `02_tube/tube_report.json` | `27fe7cf27b0db59fd72c9b802ed5f0f1f3ab281d53eeb449752a1589dda60a91` |
| `03_dataset/dataset_report.json` | `96922aa498e7b22e9460cf1bc604b6a48c60339bd20be51ad67c199a663790d0` |
| `04_summary/summary_report.json` | `ba015b29891bf0c5609ccc1a1c5a4442a7a14e0a3ee83fe5bdc7d0283f38e651` |
| `05_visualization/strict_tracking_overview.png` | `55d12ddcd3bee6f7b53ba7021a13d5b4c28075fc8ffcd7772f9ab1d2bb5f5506` |
| `05_visualization/strict_joint_profiles.png` | `c61b86b82a6810b20864609d125cb3b912e18089e6e96f9a0b26edc8235aaa5d` |
| `05_visualization/conditioning_frontier.png` | `449743a6eaf1dd49217871de32f32214046e61176ba55038f91a2f6e0e9bf846` |
| `05_visualization/exploratory_pointwise_overview.png` | `7ca570a0f07ed56e7a830a7840bf293cd0e41e285d3fd37ebd5fefa9a594eacf` |
| `05_visualization/visualization_report.json` | `07949287a5b66f6d78e38fb5082989f8d8ed438c2a392293e6fec38ed05bc1c6` |

> [!warning] 冻结结论边界
> `exploratory_rescue_rmax_mm=120` 只是 pointwise admission，不能当作 120 mm strict branch、tube、dataset 或 model 阳性。正式结论是：标签余量问题已解决；最后注册 strict radial checkpoint 为 102.5 mm；105 mm minimum 未通过；模型训练未获授权。

## 后续非正式诊断

正式 checkpoint 冻结后另行完成了 [[2026-07-17-true-ellipse-standard-domain-v7-diagnostic-training|V7-D1 centerline-only 诊断训练]]。该实验训练了 48 个模型并生成每个模型的轨迹图，但所有产物都固定为 `diagnostic_only=true`，不修改本 checkpoint 的任何正式 gate 或 claim。
