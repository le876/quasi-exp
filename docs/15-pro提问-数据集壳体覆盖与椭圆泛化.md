---
question_id: Q15
question_number: 15
question_confirmed_by_user: true
question_confirmation_summary: "仅讨论数据集泛化性：审计当前 V12.16C 的覆盖边界，并设计下一轮具有足够厚度的三维壳体与多尺度椭圆泛化实验；暂不讨论 Teacher 实际跟踪外观问题。"
date: "2026-07-31"
status: ready-to-send
project: "quasi_exp / BACRA"
evidence_cutoff: "2026-07-31T17:39:55+08:00"
source_snapshot: "实现 fixed point 为 clean worktree codex/bacra-v12-16-multichart@8afde48d8642defff7738de9aa31cc75826ca49c；交接文档和派生审计位于另一个 dirty 主工作树，不属于实现 commit。"
---

# GPT-5 Pro 第 15 次交接：数据集壳体覆盖与椭圆泛化

## 0. 经用户确认的 GPT-5 Pro 任务

用户已经确认的任务原意：

> 先不要管第二个核心问题，着重在第一个问题上。这次是 Q15 交接。当前数据集从可视化看主要分布在椭圆周围，两个 chart 仍然缺乏泛化性。需要审计当前结果，并设计下一轮实验，使数据至少覆盖一个具有明确且足够厚度的壳体；在这个壳体的可行域内，至少能够系统跟踪不同半径和长短轴比例的椭圆，而不是继续只在当前环状走廊附近加密。

确认范围：本次只讨论“数据集空间覆盖和椭圆族泛化”。不把用户暂缓的第二个核心问题加入提问，也不要求 GPT-5 Pro 在本轮分析 Teacher 跟踪曲线的外观或偏置问题。

## 1. 当前有效事实与最短摘要

### 1.1 证据截止时间

本文只使用截至 `2026-07-31T17:39:55+08:00` 已存在并重新核验的代码、配置和 artifact。当前正式实现 fixed point 是 clean worktree：

- worktree：`/mnt/ML_projects/quasi_exp/.worktrees/bacra-v12-16-multichart`
- branch：`codex/bacra-v12-16-multichart`
- HEAD：`8afde48d8642defff7738de9aa31cc75826ca49c`
- commit subject：`Add V12.16C chart-gated residual Student`

项目主工作树 `/mnt/ML_projects/quasi_exp` 位于另一个 dirty 分支 `canonical-layer-field-u3@740d8c0`。本次 Q15 文档、派生审计脚本和派生图位于该主工作树；它们只整理现有证据，没有改变 V12.16C 实现或正式结果。因此：

1. V12.16C 的实现和正式 artifact 绑定到上面的 clean fixed point；
2. Q15 交接材料本身不是 `8afde48...` 的一部分；
3. 仅 checkout `8afde48...` 可以取得当前实现，但不能取得本次新增的 Q15 派生画像与交接文档。

### 1.2 当前有效事实

1. 当前提交给 GPT-5 Pro 的完整训练/验证数据集是 `v12_16c_train_validation.parquet`。它有 `88,976` 行、`119` 列、`1` 个 row group，字节数 `40,579,423`，SHA256 为 `86f7bd2b2fd4c02e94b4e0f013473bc271080902059012c9828492e128f593ad`。
2. 数据由两个已知 chart 合并而成。Chart A 为 `76,398` 行，其中 train `59,178`、validation `17,220`；Chart B 为 `12,578` 行，其中 train `9,420`、validation `3,158`。
3. `q15_chart_ab_endpoint_distribution.png` 使用全部 `88,976` 行、无抽样、等比例二维投影。它显示的是沿现有环状/弯曲走廊及若干局部块分布的样本，不是已经验证的均匀厚壳体。
4. 当前正式 V12.16C gate 通过，但其 claim scope 明确是 `simulation_known_chart_multichart_spatial_generalization`：Chart A sealed paths `20/20`、Chart B sealed paths `20/20`、Chart A sealed random seeds `2/3`、Chart B sealed point seeds `3/3`、legacy final8 families `8/8`。`deployment_claim_gate_pass=false`。
5. 当前 `40` 条 sealed path 分别是 Chart A `20` 条和 Chart B `20` 条；每个 chart 都是 `circle / ellipse / lissajous / open_bspline / closed_bspline` 五类、每类 `4` 条、每条 `360` 点。
6. `q15_holdout_geometry_scale_audit.csv` 对这 `40` 条路径按 family 对目标 XYZ 居中后做 SVD/PCA。当前 `8` 条 sealed ellipse 的两个面内半尺度都固定在约 `3 mm` 和 `2 mm`；它们没有覆盖“任意半径、任意轴比”的椭圆族。
7. 当前 Student 输入是 `(x, y, z, known_chart_id)`，输出 `beta1..beta6`。V12.16C 冻结 V12.15 的 Chart A 主干，对 Chart B 学习 `128/128/64/6` latent residual，并把 residual 乘以 `chart_feature`；Chart A 为 `0`，因此保持原输出，Chart B 为 `1`。它不是自动 chart classifier。

### 1.3 关键边界

1. 三个轴向的 min/max 或二维投影只能证明“出现过这些坐标”，不能证明包围盒内部被填满，也不能证明任一点都有安全、连续且单值的 canonical label。
2. “whole spatial block holdout”证明的是已注册 block 划分下的局部空间泛化；它不等价于在一个预先定义的三维壳体域内具有覆盖率保证。
3. 当前 sealed ellipse 的 PCA 尺度审计只描述已注册测试轨迹，不描述完整数据集的壳厚，更不定义可外推的 ellipse 参数域。
4. 两个 chart 的 identity 是外部已知输入；当前证据没有自动 chart 选择、chart overlap 一致性或 deployment 结论。
5. 现有材料没有定义用户所说壳体的中心面、法向坐标、内外边界、厚度、可行域测度或覆盖率指标，也尚未执行多半径、多轴比、多中心、多姿态的独立封存实验。

## 2. 项目目标与本轮边界

### 2.1 用户给定的项目目标

用户希望数据不再只是围绕少数椭圆或局部 chart 走廊取样，而是至少覆盖一个“具有足够厚度的壳体”。在该壳体的 Teacher 可行域内，应能够系统构造和跟踪不同：

- 半长轴/总体尺度；
- 长短轴比例；
- 椭圆中心位置；
- 椭圆所在平面或空间姿态；

的椭圆，并使所得实验结论真正对应空间区域泛化，而不是对少量局部轨迹的邻域插值。

这里“足够厚度”是用户给定的实验目标，但目前没有由现有 artifact 定义成具体数值；这正是 Q15 希望 GPT-5 Pro 帮助形成可实施协议的部分。

### 2.2 当前实际覆盖范围

当前已实现的是：

1. 以 V12.14 region-first canonical Teacher 为基础，在 capability voxels 内从已知 canonical anchors 向邻域做 support-aware predictor-corrector continuation；
2. 使用多个父 anchor 的候选一致性、残差和 joint margin 对标签分级；
3. 在 `15 mm` macro spatial blocks 上做 train / validation / sealed 的 whole-block 隔离；
4. 从 Chart A 扩到第二个局部 Chart B；
5. 用一个需要已知 chart identity 的 gated-residual Student 同时覆盖两个已注册 chart；
6. 在两个 chart 的 sealed points 和 40 条已注册局部路径上验证 locked Student。

### 2.3 当前证据未覆盖的内容

- 没有一个显式、坐标化、可计算体积或面积的壳体定义。
- 没有证明当前 `88,976` 个点在壳体内均匀、分层或按最坏空洞半径充分覆盖。
- 没有预注册 ellipse 参数空间，也没有把 radius、axis ratio、center、plane orientation 与 train / development / sealed test 正交拆分。
- 没有对壳体内部任意椭圆的可行性概率、完整环闭合率或跨 chart 连续性做统计保证。
- 没有自动 chart routing，也没有 deployment 证据。

## 3. 相对上次材料新增的事实

上次材料为 `q14_region_first_plan_and_context.md`。其中 GPT-5 Pro 的内容只作为历史建议，不作为已执行事实。Q14 之后实际实施了 V12.14、V12.15、V12.16A、V12.16B 和 V12.16C。

| 阶段 | 已发生的实现/实验 | 实际结果 | 遇到的问题与当时处理 | 证据等级 |
|---|---|---|---|---|
| Q14 历史计划 | GPT-5 Pro 提出 region-first、capability lattice、局部 continuation、consistency audit、whole-block holdout 的方向 | 这是后续实现的历史输入，不是实验结论 | 当时只有建议，没有当前 V12.16C 数据和结果 | historical |
| V12.14 region growth | 从 D3 canonical seed 出发，做 capability-region 选择、sparse frontier growth、dense sampling、consistency audit、Student 训练和独立随机点/路径测试 | 正式 gate 通过；random seeds `3/3`、unseen paths `20/20`、final8 families `8/8`，但 final8 seed-family 为 `21/24`；claim scope 仍是 exploratory region generalization | 一次性稀疏随机 target 对 support 不敏感，改成 batched frontier expansion；triangle audit 代理不是真闭环，改成 `a→b→c→a` corrector return；一致性边包含最大连通分量外节点，改为先过滤；一 voxel 一点导致空间离散，改为 voxel 内连续平衡采样；enrichment 出现重复 voxel index，改为稳定按 voxel 聚合。纯 region Student 对旧 final8 出现约 `120–190 mm` 灾难性遗忘，最后用 `alpha=0.94` output-residual composite 修复保留性 | formal（限本协议）；问题处理为同 lineage 实现事实 |
| V12.15 single Student | 把 V12.14 composite 蒸馏成单个 bounded MLP，并把 `15 mm` whole blocks 预先封存 | margin-tail retry1 正式 gate 通过；sealed paths `20/20`、sealed random `2/3` seeds、final8 `8/8` families、`23/24` seed-family | smoke 先遇到 GPU/sandbox 与 `major_semiaxis` merge 缺列；修复入口后，第一次 formal 在 sealed holdout 打开前的 selection lock 停止：minimum-margin 尾部约 `1.23–1.38°`。同 block 的 composite 约 `1.656–1.777°`，因此没有放宽 gate，而是增加 margin-tail sample weight 后重跑 retry1 | formal |
| V12.16A Chart B | 在第二个局部区域生成 Chart B，并独立做 spatial seal | sparse accepted `2,987/3,000`；最大连通分量 `2,486`；triangle P95 `0.010°`、two-path P95 `0.333°`；dense region `20,000`；region macro blocks `255`，其中 train `179`、validation `38`、sealed `38` | 该阶段只证明 Chart B region readiness；没有把 Chart B 直接等同于全局第二张 chart 或厚壳体 | formal（Chart B readiness） |
| V12.16B shared-trunk Student | 直接训练一个 `(xyz, chart_feature)→beta` shared-trunk bounded MLP | Chart B points `3/3`、B paths `20/20`、A paths `20/20`，但 legacy final8 只有 `7/8` families；`final_core_01` 只有 `1/3` seeds 通过，因此最终 gate 失败 | 没有放宽 gate；把 failure 保留下来，改用冻结 Chart A 的架构 | formal negative result |
| V12.16C gated residual | 冻结 V12.15 Chart A trunk，只训练 Chart B latent residual；训练抽样权重 A/B=`0.10/0.90` | 最终 gate 通过；A paths `20/20`、B paths `20/20`、A random `2/3`、B sealed points `3/3`、final8 `8/8`，final8 seed-family `23/24` | 解决了 V12.16B 的旧域回退，但仍依赖已知 chart identity；未解决壳体覆盖和 ellipse 参数域问题 | formal |
| Q15 派生审计 | 对完整 V12.16C 数据做全行统计与投影；对两 chart 的 40 条 sealed path 做 PCA 尺度审计 | 图像显示局部环状/走廊分布；8 条 sealed ellipse 只有约 `3 mm × 2 mm` 半尺度 | 这是对已有数据的无抽样描述，不是新的正式实验，也不能从 bbox 推断壳体 | diagnostic-only |

对应附件分别是 `q14_region_first_plan_and_context.md`、`v12_14_final_gate.json`、`v12_14_final_recommendation.md`、`v12_15_final_gate.json`、`v12_15_experiment_record.md`、`v12_16a_spatial_seal_gate.json`、`v12_16b_failure_gate.json`、`v12_16c_final_gate.json`、`q15_current_dataset_profile.json`、`q15_holdout_geometry_scale_audit.csv` 和 `q15_chart_ab_endpoint_distribution.png`。

## 4. 实验方法与复现口径

### 4.1 代码与工作树 fixed point

| 项目 | 值 |
|---|---|
| 项目根 | `/mnt/ML_projects/quasi_exp` |
| 实现分支 | `codex/bacra-v12-16-multichart` |
| 实现 HEAD | `8afde48d8642defff7738de9aa31cc75826ca49c` |
| 实现 worktree | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v12-16-multichart` |
| 实现 dirty 状态 | clean |
| V12.16C artifact root | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731` |
| Q15 派生证据 root | `/mnt/ML_projects/quasi_exp/runs/gpt5pro_handoff_evidence/Q15` |
| 主工作树 dirty 状态 | dirty，包含并发项目文件与本次未跟踪 Q15 文件；不是实现 fixed point |
| 仅 checkout HEAD 是否足以复现 | 足以取得 V12.16C 实现；不含已有 `runs/` artifact、完整数据集和本次 Q15 派生审计 |

`v12_16c_source_manifest.json` 锁定了正式运行使用的源文件哈希，`v12_16c_model_lock.json` 锁定了选择结果、三个模型及其 SHA256，并记录 implementation commit `8afde48...`。

### 4.2 Teacher 实现方法

这里的 Teacher 是数值 canonical labeling pipeline，不是另一个神经网络：

1. `canonical.py` 通过 forward environment、Jacobian、weighted damped pseudoinverse 和 bounded corrector，把给定 XYZ target 修正到六维 `beta`；
2. 对路径任务，它会从多个 seed 构造候选层，按 candidate continuity 连接并进行全路径 least-squares refinement；
3. `region_growth.py` 对空间 region 采用 parent-anchor predictor-corrector continuation。每个 target 从多个邻近 canonical parents 独立生成候选，再由 `reduce_parent_candidates` 按 candidate gap、residual 和共识父节点裁决；
4. 默认质量边界在代码中为：Gold residual `≤1 mm` 且 candidate gap `≤0.5°`；Silver residual `≤3 mm` 且 candidate gap `≤1°`。最终正式协议还叠加 joint margin、连通性、loop audit、spatial split 和 artifact hash gate；
5. dense target 只在有足够 anchor support 的 capability region 中产生，不能把任意 bbox 点直接承认为 Teacher 可行 label。

实际代码是 `canonical.py`、`region_growth.py` 和入口 `run_bacra_v12_14_region_growth.py`。这些附件用于让 GPT-5 Pro 判断如何把现有 region-first 机制改造成壳体覆盖协议。

### 4.3 Student 实现方法

当前 V12.16C Student：

1. 输入：`x_m, y_m, z_m, chart_feature`；
2. 输出：`beta1_rad..beta6_rad`；
3. `chart_feature=0` 表示已知 Chart A，`chart_feature=1` 表示已知 Chart B；
4. 从已锁定的 V12.15 single bounded MLP 载入 Chart A trunk，并冻结其 layer；
5. 用 `xyz + chart_feature` 训练 `128 GELU → 128 GELU → 64 GELU → 6 latent residual`；
6. residual 与 `chart_feature` 相乘后加到冻结 trunk 的 latent output，再通过原 bounded output transform 得到 beta；因此 Chart A 路径在结构上保持原 trunk；
7. 三个 training seeds 是 `20260738, 20260739, 20260740`；V12.16C chart sampling weights 为 Chart A `0.10`、Chart B `0.90`；
8. 它仍需要外部已知 chart id，没有自动 classifier。

实际配置、模型构造和正式 runner 分别是 `bacra_v12_16c_chart_gated_residual.yaml`、`multichart_distillation.py` 和 `run_bacra_v12_16b_chart_conditioned_student.py`。

### 4.4 数据生成与划分

| 项目 | 实际口径 |
|---|---|
| 完整数据 | `v12_16c_train_validation.parquet`，全部纳入本 ZIP |
| manifest | `v12_16c_dataset_manifest.json` |
| 样本数 | `88,976` |
| schema | `119` 列；完整列名在 `q15_current_dataset_profile.json` |
| Chart A | train `59,178`；validation `17,220` |
| Chart B | train `9,420`；validation `3,158` |
| Chart A quality | Gold `870`；RegionGold `48,947`；RegionSilver `2,823`；Silver `23,758` |
| Chart B quality | Gold `677`；RegionGold `10,957`；RegionSilver `944` |
| validation 隔离 | profile 中 Chart A train/validation spatial block overlap `0`；Chart B overlap `0` |
| sealed test | 不包含在训练 Parquet；Chart A 和 Chart B 分别使用预注册 sealed block/registry |
| split seed | Chart B macro partition seed `20260808`；其他 source seed 和哈希由各 protocol/model lock 记录 |
| trajectory generation | 每个 chart `5` 类 × `4` family × `360` points |
| 关键限制 | 当前 split 是既有局部 region 的 block 隔离，不是 ellipse 参数空间或显式壳体坐标的隔离 |

Chart A profile 的轴向范围：

- `x: 0.974402..1.138563 m`
- `y: -0.502575..0.405978 m`
- `z: -0.312971..0.274986 m`

Chart B profile 的轴向范围：

- `x: 1.005309..1.094764 m`
- `y: -0.484996..0.370947 m`
- `z: -0.297352..0.226756 m`

这些范围只作描述，不代表包围盒内覆盖。完整 quantiles、block counts、列名和生成方法在 `q15_current_dataset_profile.json`。

### 4.5 baseline、评价和 gate

| 项目 | 实际定义 | 是否正式 |
|---|---|---|
| V12.14 baseline | V12.13 base Student + V12.14 region Student 的 output-residual composite，`alpha=0.94` | V12.14 formal，限其 claim scope |
| V12.15 baseline | 单个 bounded MLP 对 V12.14 composite/Teacher 进行蒸馏 | formal |
| V12.16B 对照 | shared-trunk known-chart single bounded MLP | formal negative result |
| V12.16C 当前模型 | frozen Chart A trunk + chart-gated B residual | formal |
| point/path metric | FK P50/P95/max mm、minimum joint margin deg、actual bounds | formal |
| legacy ellipse retention | P95/major semiaxis `≤1%`、max/major semiaxis `≤2%` | formal |
| 当前总 gate | known-chart multi-chart point/path + final8 retention | formal |
| 壳体覆盖 metric | 未定义、未执行 | unknown |
| 多尺度 ellipse 参数域 gate | 未定义、未执行 | unknown |

### 4.6 已执行入口与产物

当前实现入口：

```text
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python \
  scripts/analysis/run_bacra_v12_16b_chart_conditioned_student.py \
  --config configs/bacra_v12_16c_chart_gated_residual.yaml \
  --formal
```

该命令应在 clean V12.16C worktree 语境理解；正式长任务已完成，本交接没有重新启动实验。

Q15 派生画像已经执行：

```text
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python \
  scripts/analysis/build_q15_handoff_evidence.py
```

派生规则：

- dataset profile：全部行，无抽样；
- distribution figure：全部行，无抽样，二维投影等比例；
- path scale audit：每个 family 的 target XYZ 居中后做 SVD/PCA，报告局部轴上 half peak-to-peak extent；
- random seed：无。

主要产物：

- `q15_current_dataset_profile.json`
- `q15_holdout_geometry_scale_audit.csv`
- `q15_chart_ab_endpoint_distribution.png`
- 生成器 `build_q15_handoff_evidence.py`

## 5. 实验结果

| Fact ID | 已观察结果 | 数值 | 证据等级 | 实际附件/字段 | 适用边界 |
|---|---|---:|---|---|---|
| F1 | 当前完整数据集规模 | `88,976 × 119`；`40,579,423 B` | formal artifact | `v12_16c_dataset_manifest.json`、`v12_16c_train_validation.parquet` | 训练/验证数据，不含 sealed test |
| F2 | Chart A/B 行数 | A `76,398`；B `12,578` | diagnostic-only profile of formal artifact | `q15_current_dataset_profile.json → dataset.chart_profiles` | 描述现有 mixture |
| F3 | train/validation block overlap | A `0`；B `0` | diagnostic-only profile | `q15_current_dataset_profile.json → spatial_block_audit` | 不代表显式壳体 holdout |
| F4 | 当前数据投影形态 | 局部环状/弯曲走廊和局部块 | diagnostic-only | `q15_chart_ab_endpoint_distribution.png` | 视觉证据，不是拓扑或体积证明 |
| F5 | sealed ellipse 尺度 | 8 条；PCA half extents 约 `3 mm × 2 mm` | diagnostic-only | `q15_holdout_geometry_scale_audit.csv`、profile `sealed_path_scale_audit` | 只描述已注册 holdouts |
| F6 | V12.14 region generalization | random `3/3`；paths `20/20`；final8 `8/8`，seed-family `21/24` | formal | `v12_14_final_gate.json` | exploratory region claim；deployment false |
| F7 | V12.15 single Student | random `2/3`；paths `20/20`；final8 `8/8`，seed-family `23/24` | formal | `v12_15_final_gate.json` | single-chart spatial block claim |
| F8 | Chart B spatial seal | region blocks `255`；train/val/sealed=`179/38/38`；train/val rows=`9,420/3,158`；sealed rows `2,721` | formal | `v12_16a_spatial_seal_gate.json` | Chart B readiness |
| F9 | V12.16B shared-trunk failure | final8 `7/8` | formal negative | `v12_16b_failure_gate.json` | 其他 multi-chart 指标通过但总 gate 失败 |
| F10 | V12.16C 总 gate | gate pass；A paths `20/20`、B paths `20/20`、A random `2/3`、B points `3/3`、final8 `8/8` | formal | `v12_16c_final_gate.json`、`v12_16c_generalization_summary.csv` | known-chart simulation；deployment false |
| F11 | Chart A path worst observed | worst P95 `1.5169 mm`；max `1.5280 mm`；minimum margin `1.6759°` | formal | `v12_16c_chart_a_path_metrics.csv` | 20 registered local paths × 3 seeds |
| F12 | Chart B path worst observed | worst P95 `4.4307 mm`；max `4.4401 mm`；minimum margin `2.7615°` | formal | `v12_16c_chart_b_path_metrics.csv` | 20 registered local paths × 3 seeds |
| F13 | Chart A random | `2/3` seeds；P95 max `1.3430 mm`；max error `3.3514 mm`；minimum margin `1.3853°` | formal | `v12_16c_chart_a_random_metrics.csv` | sealed local blocks |
| F14 | Chart B sealed points | `3/3` seeds；P95 max `3.0895 mm`；max error `5.3754 mm`；minimum margin `2.4568°` | formal | `v12_16c_chart_b_sealed_metrics.csv` | `2,721` sealed rows |
| F15 | final8 retention | families `8/8`；seed-family `23/24`；worst P95 `5.0065 mm`；max `6.2254 mm`；minimum margin `1.1270°` | formal | `v12_16c_final8_retention_metrics.csv` | legacy 大椭圆 retention，不代表壳体覆盖 |

### 5.1 通过项

- 现有 pipeline 已经从单一路径附近扩到一个 region-first dataset，并从单 Chart A 扩到已知 Chart A + Chart B。
- whole-block train/validation/sealed 隔离、model lock、holdout-before-open 和 artifact hash closure 已实际实施。
- V12.16C 在其注册范围内通过两个 chart 的 local path/point gate，并保留 legacy final8 family gate。

### 5.2 失败与反例

- V12.16B 是直接反例：多 chart 的 point/path 指标可以通过，但 shared-trunk Student 仍可破坏旧域 final8，使总 gate 失败。
- V12.14 的纯 region Student 出现旧 final8 `120–190 mm` 量级遗忘，说明“扩大局部 region 数据”本身不保证旧几何 family 保留。
- V12.15/V12.16C 的 Chart A random 都只有 `2/3` seed 通过；当前 gate 允许该 formal 结果，但它不是任意空间点稳定性的证明。
- 当前 8 条 sealed ellipse 尺度都约为 `3 mm × 2 mm`，与“系统覆盖多半径、多轴比”目标不一致。
- 当前分布图和轴向范围不能证明存在一个已经覆盖的厚壳体。

## 6. 证据等级、失败结果与未知事项

### 6.1 正式事实

- V12.14、V12.15、V12.16A、V12.16B、V12.16C 的 gate 只在各自 `claim_scope` 内作为 formal。
- V12.16C 的最终 claim 是 `simulation_known_chart_multichart_spatial_generalization`，而且 `deployment_claim_gate_pass=false`。
- 当前模型在 sealed evaluation 之前已经锁定；三个 seed model、selection artifact 和 registry 均有 hash。
- V12.16B 的失败没有通过放宽 gate 改写；V12.16C 通过架构隔离修复旧域 retention。

### 6.2 诊断、探索、evidence-only、post-hoc 或历史事实

| 内容 | 等级 | 能支持什么 | 不能支持什么 |
|---|---|---|---|
| Q14 GPT-5 Pro region-first 回答 | historical | 解释实现路线的来源 | 不能证明任何阶段已经实施或通过 |
| Q15 全行 profile | diagnostic-only | 精确描述当前数据 schema、行数、分组、quantiles 和 bbox | 不能证明 bbox 内覆盖或壳体厚度 |
| Q15 全行投影图 | diagnostic-only | 显示当前样本空间形态集中在局部走廊 | 不能单独定义三维拓扑、体积或可行率 |
| sealed path PCA audit | diagnostic-only | 量化已注册 40 条路径、尤其 8 条 ellipse 的局部尺度 | 不能推出未注册 ellipse 的可行参数域 |
| V12.14 claim scope | formal but exploratory scope | 对应其预注册 region generalization gate | 不能升级为全局 workspace/deployment |

### 6.3 已发现的实现或证据缺口

- 数据 schema 没有 shell id、shell normal coordinate、signed thickness coordinate、local tangent frame 或 shell stratum。
- 没有用 coverage distance、empty-ball radius、voxel occupancy、connected shell component、normal-depth bins 等方式定义“足够厚”和“覆盖完成”。
- ellipse catalogs 没有显式记录/封存半长轴、轴比、中心、平面法向/姿态的组合域；现有 8 条 ellipse 实际尺度高度固定。
- 两 chart 是已知离散 id；没有说明它们是否共同参数化同一壳体、互相重叠还是两个分离局部 chart。
- 现有 region growth 只允许 support-aware continuation；如果目标壳体超出现有 capability lattice，如何发现新 canonical branch 仍未解决。

### 6.4 尚未知或尚未执行

- 用户目标中“足够厚度”的物理/几何数值和允许的壳体范围尚未定义。
- 壳体内 Teacher 可行域的体积分数、连通分量数量和 branch multiplicity 尚未知。
- 任意给定 ellipse 参数组合能否完整闭合、是否跨 chart、是否保持 canonical branch identity 尚未知。
- 对半径、轴比、中心和姿态联合 OOD 的 Student 泛化尚未执行。
- 新壳体数据量、Teacher 求解成本、失败率和需要的 chart 数量尚未测量。

### 6.5 当前不能声称

- 不能声称当前数据已经覆盖 axis-aligned bbox 或一个均匀厚壳体。
- 不能声称两个 chart 已经覆盖任意半径/比例/中心/姿态的椭圆。
- 不能把 40 条局部 path 的通过等同于 ellipse-family 参数域泛化。
- 不能把 legacy final8 大椭圆 retention 等同于在新壳体中训练或封存了大尺度 ellipse。
- 不能声称自动 chart selection、跨 chart 连续切换或 deployment readiness。

## 7. 决策所需的客观对照

本节只比较已经执行的 lineage，不给出路线推荐。

| 对照对象 | 已执行输入/方法 | 已观察结果 | 证据等级 | 已知限制 |
|---|---|---|---|---|
| V12.14 composite | region Student + V12.13 base Student，output residual `alpha=0.94` | random `3/3`、paths `20/20`、final8 `8/8` | formal | 两个模型组合；exploratory region scope |
| V12.15 single Student | 单 Chart A bounded MLP 蒸馏，15 mm whole-block seal | random `2/3`、paths `20/20`、final8 `8/8` | formal | 单 chart；仍是局部 region |
| V12.16B shared trunk | `(xyz, known chart)→beta` 单 shared-trunk bounded MLP | 新旧 chart paths/points 多数通过，但 final8 `7/8`，总 gate 失败 | formal negative | 旧域干扰 |
| V12.16C gated residual | frozen A trunk + B residual，known chart hard gate | 全部 family-level gate 通过 | formal | 需要 chart id；仍无壳体定义 |
| 当前 sealed path set | 每 chart 20 条，5 类 ×4，360 点 | 40/40 family 通过 | formal | ellipse 只有约 `3×2 mm` 半尺度 |
| 用户目标 | 显式厚壳体 + 壳内多半径/轴比/中心/姿态 ellipse | 尚未执行 | user-stated target | 需要新协议、数据结构和验证设计 |

## 8. 经用户确认的问题

1. 请审计当前 V12.16C 是否足以支持“数据集区域泛化”这一表述，并设计下一轮实验：把数据扩展为具有明确且足够厚度的三维壳体，在该壳体的 Teacher 可行域内系统覆盖多半径、长短轴比例、中心位置和平面姿态的椭圆，而不是继续只在当前环状走廊附近加密。请基于附件中的完整数据集、可视化、几何尺度审计、历史失败和实际 Teacher/Student 实现，给出边界判断和一份可直接实施的下一轮实验设计/协议。

## 9. 经用户确认的回答要求

返回一份可直接实施的下一轮实验设计/协议，着重回答如何实现并验证具有足够厚度的壳体覆盖，以及如何在该壳体可行域内实现并验证多半径、多轴比、多中心和多姿态的椭圆泛化。回答同时需要先明确当前 V12.16C 证据实际支持到哪里、不能支持什么。

## 10. 附件清单与复现信息

GPT-5 Pro 实际可见的是“上传文件名”列；原始 SSH 路径只用于 provenance。

唯一上传 ZIP 的成员数（`QUESTION.md + 实际证据附件`）：`31`。无论成员数多少，网页端和 Windows 传输都只使用打包器生成的 `Q15_GPT5Pro_upload.zip`。

| 上传文件名 | 原始 SSH 路径 | 大小 | 证据等级 | 支持事实 | 上传理由 |
|---|---|---:|---|---|---|
| `v12_16c_train_validation.parquet` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731/01_multichart_dataset/train_validation.parquet` | 40,579,423 B | formal artifact | F1–F5 | 用户明确要求提交当前完整数据集；允许 GPT 独立检查 schema 和分布 |
| `v12_16c_dataset_manifest.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731/01_multichart_dataset/dataset_manifest.json` | 382 B | formal | F1 | 记录数据行数和 SHA256 |
| `q15_current_dataset_profile.json` | `/mnt/ML_projects/quasi_exp/runs/gpt5pro_handoff_evidence/Q15/q15_current_dataset_profile.json` | 9,729 B | diagnostic-only | F1–F5 | 全行、无抽样的数据画像和派生方法 |
| `q15_holdout_geometry_scale_audit.csv` | `/mnt/ML_projects/quasi_exp/runs/gpt5pro_handoff_evidence/Q15/q15_holdout_geometry_scale_audit.csv` | 8,671 B | diagnostic-only | F5 | 独立列出 40 条 sealed path 的 PCA 尺度 |
| `q15_chart_ab_endpoint_distribution.png` | `/mnt/ML_projects/quasi_exp/runs/gpt5pro_handoff_evidence/Q15/q15_chart_ab_endpoint_distribution.png` | 1,128,210 B | diagnostic-only | F4 | 全数据三种等比例二维投影，直观看到局部走廊形态 |
| `build_q15_handoff_evidence.py` | `/mnt/ML_projects/quasi_exp/scripts/analysis/build_q15_handoff_evidence.py` | 12,773 B | diagnostic provenance | F1–F5 | 公开画像、图和 PCA 审计的生成方法 |
| `q14_region_first_plan_and_context.md` | `/mnt/ML_projects/quasi_exp/docs/14-pro提问-从当前椭圆走廊扩展到多几何泛化数据集.md` | 58,169 B | historical | 实现 lineage | 保留最近一次 GPT-5 Pro 计划；文中明确不能作为已执行证据 |
| `v12_14_final_gate.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_14_region_grown_dataset_retry6/11_summary/gate.json` | 10,276 B | formal | F6 | V12.14 最终 gate、claim scope、model lock 和通过数 |
| `v12_14_final_recommendation.md` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_14_region_grown_dataset_retry6/11_summary/final_recommendation.md` | 273 B | formal summary | F6 | V12.14 的短结论与边界 |
| `v12_15_final_gate.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_15_single_student_formal_margin_tail_retry1_20260730/08_summary/gate.json` | 9,404 B | formal | F7 | V12.15 single Student 最终 gate |
| `v12_15_experiment_record.md` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v12-16-multichart/docs/BACRAV12_15单模型蒸馏与空间块封存验证实验记录.md` | 10,646 B | formal record | F7、问题处理 | 记录 V12.15 first formal 停止、margin-tail 修复和 retry1 |
| `v12_16a_spatial_seal_gate.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16_chart_b_region_formal_20260731/02_chart_b_spatial_seal/gate.json` | 2,767 B | formal | F8 | Chart B train/validation/sealed block 闭合 |
| `v12_16b_failure_gate.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16b_known_chart_single_student_formal_20260731/06_summary/gate.json` | 5,931 B | formal negative | F9 | shared-trunk Student 的 final8 失败反例 |
| `v12_16c_final_gate.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731/06_summary/gate.json` | 5,924 B | formal | F10 | 当前最终裁决和 claim boundary |
| `v12_16c_generalization_summary.csv` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731/05_locked_evaluation/generalization_summary.csv` | 193 B | formal | F10 | 当前通过数的紧凑汇总 |
| `v12_16c_chart_a_path_metrics.csv` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731/05_locked_evaluation/chart_a_sealed_path_metrics.csv` | 7,261 B | formal | F11 | Chart A 20 path ×3 seed 逐项指标 |
| `v12_16c_chart_b_path_metrics.csv` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731/05_locked_evaluation/chart_b_path_metrics.csv` | 7,203 B | formal | F12 | Chart B 20 path ×3 seed 逐项指标 |
| `v12_16c_chart_a_random_metrics.csv` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731/05_locked_evaluation/chart_a_sealed_random_metrics.csv` | 399 B | formal | F13 | Chart A sealed random 逐 seed 结果 |
| `v12_16c_chart_b_sealed_metrics.csv` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731/05_locked_evaluation/chart_b_sealed_metrics.csv` | 394 B | formal | F14 | Chart B sealed point 逐 seed 结果 |
| `v12_16c_final8_retention_metrics.csv` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731/05_locked_evaluation/final8_retention_metrics.csv` | 4,345 B | formal | F15 | legacy final8 retention 的尺度归一化指标 |
| `v12_15_chart_a_path_catalog.csv` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_15_single_student_formal_margin_tail_retry1_20260730/07_sealed_trajectory_test/trajectory_catalog.csv` | 2,015 B | formal | F5 | Chart A 20 条 sealed path 注册表 |
| `v12_16c_chart_b_path_catalog.csv` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731/05_locked_evaluation/chart_b_path_catalog.csv` | 2,008 B | formal | F5 | Chart B 20 条 sealed path 注册表 |
| `v12_16c_source_manifest.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731/00_protocol/source_manifest.json` | 4,616 B | formal provenance | fixed point | 锁定正式运行使用的代码/config hash |
| `v12_16c_model_lock.json` | `/mnt/ML_projects/quasi_exp/runs/bacra_v12_16c_known_chart_gated_residual_formal_20260731/04_model_lock/final_model_lock.json` | 1,192 B | formal provenance | F10 | 锁定 selection、三个模型及 implementation commit |
| `bacra_v12_16c_chart_gated_residual.yaml` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v12-16-multichart/configs/bacra_v12_16c_chart_gated_residual.yaml` | 512 B | implementation | Student 方法 | 当前 V12.16C 覆盖配置与 A/B 采样权重 |
| `multichart_distillation.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v12-16-multichart/src/quasi_exp/teacher/multichart_distillation.py` | 14,540 B | implementation | Student 方法 | shared-trunk 与 gated-residual 的实际构造和 loss |
| `run_bacra_v12_16b_chart_conditioned_student.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v12-16-multichart/scripts/analysis/run_bacra_v12_16b_chart_conditioned_student.py` | 41,251 B | implementation | Student/正式流程 | 合并数据、训练、锁模和 sealed evaluation 的实际入口 |
| `run_bacra_v12_14_region_growth.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v12-16-multichart/scripts/analysis/run_bacra_v12_14_region_growth.py` | 95,766 B | implementation | Teacher/region 流程 | Q14 region-first 计划的完整执行入口 |
| `region_growth.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v12-16-multichart/src/quasi_exp/teacher/region_growth.py` | 43,854 B | implementation | Teacher 方法 | support-aware growth、parent consensus、consistency audit 和 dense target |
| `canonical.py` | `/mnt/ML_projects/quasi_exp/.worktrees/bacra-v12-16-multichart/src/quasi_exp/teacher/canonical.py` | 29,241 B | implementation | Teacher 方法 | canonical corrector、candidate linking 和 path refinement |

### 未上传的大文件或派生数据说明

1. 本轮没有用摘要替代当前完整训练/验证数据：`v12_16c_train_validation.parquet` 已完整进入 ZIP。
2. Chart A/Chart B 的 `teacher_reference.parquet` 和 tracking Parquet 没有上传，因为本轮问题只聚焦数据覆盖；它们的源路径、字节数、SHA256 和行数已记录在 `q15_current_dataset_profile.json`，而 path catalog、尺度审计和正式 metrics 已上传。
3. 没有上传 model checkpoint、整棵 `runs/`、重复轨迹图、worker logs 或中间 cache。
4. `q15_current_dataset_profile.json`、`q15_holdout_geometry_scale_audit.csv` 和 `q15_chart_ab_endpoint_distribution.png` 是派生证据；生成器、源文件 SHA、行数、无抽样规则、无随机 seed 和限制均随包提供。
5. 派生投影图不能证明三维壳体拓扑或覆盖完整性；PCA audit 不能推出未注册 ellipse 参数域。

## 11. 给下一位讨论者的最短事实摘要

1. 当前完整数据集是 `88,976 × 119`、约 `40.6 MB` 的 Chart A/B train+validation Parquet；它已随包提交。
2. V12.16C 在 `simulation_known_chart_multichart_spatial_generalization` 范围内正式通过，但依赖已知 chart id，且 deployment gate 为 false。
3. 全行投影显示当前数据仍是局部环状/弯曲走廊和局部块；轴向 bbox 不能证明厚壳体覆盖。
4. 当前两个 chart 的 8 条 sealed ellipse 都只有约 `3 mm × 2 mm` 的 PCA 半尺度，不能支持多半径、多轴比 ellipse-family 泛化。
5. Q14 到 V12.16C 已完成 region growth、whole-block seal、single Student、第二 chart、shared-trunk 失败和 gated-residual 修复；相关 gate、配置和关键实现均随包提供。
6. 用户确认的问题只有一个：审计当前 claim 边界，并给出可直接实施的“显式厚壳体覆盖 + 壳内多尺度、多轴比、多中心、多姿态椭圆泛化”下一轮实验协议。

---
# 核心判断

当前主要矛盾已经发生了阶段性转移。

过去的主要问题是：

$$
\text{Teacher能否沿一条轨迹稳定选择同一逆解分支}
$$

现在这部分在局部区域已经基本解决，Student 也能够较好地拟合 Teacher。当前主要矛盾变成了：

$$
\boxed{
\text{扩大任务空间覆盖}
\quad\text{与}\quad
\text{保持canonical标签一致}
}
$$

现有方法为了保证标签一致，要求新点必须靠近已有 anchor、由多个父节点短距离 continuation 得到近似相同的 (\beta)。这种策略确实能抑制 branch 混叠，但它天然只会在已有轨迹附近缓慢膨胀。

从实现上看，当前 `select_capability_region` 是从 seed voxel 出发，按固定层数做 26 邻域扩张；`select_sparse_targets` 又要求新节点必须能被已有父节点在有限距离内支持。因此，当 seed 本身主要来自椭圆曲线或局部块时，最终区域在拓扑上就是这些 seed 的“形态学膨胀”，很难自发形成完整二维曲面和三维厚壳。[当前 region-growth 实现](sandbox:/mnt/data/Q15_extracted/region_growth.py)

这不是参数没调好，而是**生成坐标系和区域生长拓扑本身限制了结果**。

当前材料给出的直接事实也支持这一判断：

* 数据已经有 (88{,}976) 行，但投影仍主要表现为环状、弯曲走廊和少量局部块；
* Chart A、Chart B 都只是局部区域；
* 当前注册的 8 条局部 sealed ellipse，PCA 半尺度仍固定在约 (3\text{ mm}\times2\text{ mm})；
* Student 虽然在已注册 Chart A/B 范围内表现良好，但这不能证明一个显式厚壳体已被覆盖。[Q15实验说明](sandbox:/mnt/data/Q15_extracted/QUESTION.md) [当前数据画像](sandbox:/mnt/data/Q15_extracted/q15_current_dataset_profile.json) [完整分布图](sandbox:/mnt/data/Q15_extracted/q15_chart_ab_endpoint_distribution.png)

此前从 D2 的 (25{,}920) 行增加到 D3 的 (93{,}600) 行，主要增加的是同一走廊附近的 tube 样本，而 stress00 的 FK P95 只从约 (3.123)mm 改善到 (3.085)mm。真正产生主要提升的是增加不同 family，而不是继续局部加密。

所以，下一步不能再采用：

$$
\text{已有椭圆}
\rightarrow
\text{局部tube}
\rightarrow
\text{继续向外扩几层voxel}
$$

需要改成：

$$
\boxed{
\text{先设计一个显式三维壳体}
\rightarrow
\text{在整个壳体表面建立canonical atlas}
\rightarrow
\text{沿壳体厚度扩展}
\rightarrow
\text{在壳体中均匀生成数据}
}
$$

这是方法层面的质变。

---

# 一、把当前问题拆成六个本质问题

## 问题一：目标区域没有被数学定义

当前只有：

* XYZ 的 min/max；
* voxel；
* spatial block；
* distance-to-seed；
* region wave；
* chart id。

但没有：

* 壳体中心面；
* 壳体内外边界；
* 法向；
* 厚度坐标；
* 表面面积；
* 壳体体积；
* 空洞半径；
* 面内覆盖率；
* 法向层覆盖率。

因此当前无法回答：

> 数据究竟覆盖了多大的壳体？

只能回答：

> 某些坐标上出现过数据。

学术界在处理这类问题时，通常先显式定义 task-space region，再设计采样和投影方法。Task Space Regions 的核心就是把任务允许区域显式表示为可采样、可计算距离的集合，而不是依赖若干历史轨迹隐式定义区域。([Sage Journals][1])

### 对本项目的解决方法

必须先定义：

$$
\Omega_{\text{shell}}
=====================

\left{
x=s(u,v)+\rho n(u,v)
\right}
$$

其中：

* (s(u,v)) 是二维中心面；
* (n(u,v)) 是单位法向；
* (\rho\in[-h_-,h_+]) 是厚度坐标。

只有这样才能定义“壳体覆盖率”和“足够厚度”。

---

## 问题二：现有 seed 的维度和目标壳体不匹配

当前 seed 主要来自：

* 椭圆中心线；
* 椭圆附近的小 tube；
* 由这些数据生成的局部区域。

这些 seed 的基本结构仍接近：

$$
\text{一维曲线}
+
\text{少量局部三维块}
$$

而目标壳体是：

$$
\text{二维中心面}
\times
\text{一维厚度}
$$

也就是三维区域。

从一维曲线做局部 flood-fill，不可能自然获得质量均匀的二维中心面。它通常只会得到一条加粗的走廊。

这与 AtlasRRT 一类方法的区别在于：AtlasRRT 会显式建立局部 chart，并通过 chart 的切空间向尚未覆盖的流形方向扩展，而不是只围绕历史轨迹点做欧氏邻域膨胀。([Sage Journals][2])

### 对本项目的解决方法

下一轮必须先生成一张**全局二维表面网格**，再在这张网格上求逆解标签。

不能再让历史椭圆决定任务点的位置。

---

## 问题三：壳体上的 IK 是多值的

对每个目标点：

$$
x\in\mathbb R^3
$$

存在多组：

$$
\beta\in\mathbb R^6
$$

满足：

$$
F(\beta)=x
$$

所以即使壳体点采得很均匀，若每个点独立求一个 IK，仍然可能得到：

$$
x_i\approx x_j
\qquad\text{但}\qquad
\beta_i\not\approx\beta_j
$$

工业 IK 求解器通常使用 seed、consistency limit、多起点和不同数值求解机制来扩大成功域，但这些方法本身不会自动产生一个全局连续逆映射。连续轨迹通常依赖前一时刻构型作为 seed。

Fang 等针对软机器人选择学习 FK 和 Jacobian，再使用 Jacobian continuation 求 IK，正是为了避免直接学习多值逆映射。([arXiv][3])

IKLink 则为每个 waypoint 保留多组 IK 候选，再通过图和动态规划选择整体连续路径。([arXiv][4])

### 对本项目的解决方法

在壳体表面每个节点保留多组 IK 候选：

$$
\mathcal C_i=
{\beta_{i,1},\ldots,\beta_{i,K}}
$$

然后在整张二维表面网格上联合选择 branch，而不是逐点独立选 residual 最小解。

---

## 问题四：一个全局 chart 很可能覆盖不了完整壳体

一个闭合壳体的拓扑通常接近：

$$
S^2\times[-h,h]
$$

而机器人逆解 branch 可能在某些位置：

* 接近关节边界；
* 穿过奇异区域；
* 与另一 branch 交换；
* 无法连续延伸；
* 产生不同的闭环返回构型。

因此，不能预设：

$$
xyz\rightarrow\beta_6
$$

在整个壳体上必然是一个全局连续函数。

机器人学中使用 atlas 的原因正是：复杂流形往往无法被一张无奇异坐标图覆盖，需要多个相互重叠的 chart。

局部数据驱动逆模型在软机器人中也被证明可能优于单一高阶全局模型。([Datalogisk Institut][5])

### 对本项目的解决方法

目标不应是强迫一个 chart 覆盖全部壳体，而应允许：

$$
\Omega_{\text{shell}}
=====================

\bigcup_{k=1}^{K}\Omega_k
$$

每个 chart 对应：

$$
c_k:\Omega_k\rightarrow\beta_6
$$

当前 V12.16C 已经支持 known chart id，因此下一轮可以继续使用已知 chart routing，不需要把自动 chart classifier 作为当前阻塞项。

---

## 问题五：样本数不等于覆盖质量

在已有走廊上增加几十万点，只会降低局部采样间距，不会扩大几何支持。

真正的覆盖指标应该包括：

### 表面覆盖率

$$
R_A=
\frac{
\text{已标注表面面积}
}{
\text{目标中心面总面积}
}
$$

### 壳体体积覆盖率

$$
R_V=
\frac{
\text{已标注壳体体积}
}{
\text{目标壳体总体积}
}
$$

### Fill distance

$$
h(D,\Omega)
===========

\max_{x\in\Omega}
\min_{x_i\in D}
|x-x_i|
$$

### 厚度分布

$$
h_{\min},\quad h_{10},\quad h_{50}
$$

### 最大空洞半径

$$
r_{\text{hole,max}}
$$

均匀曲面采样通常采用 Poisson-disk、farthest-point、centroidal Voronoi tessellation 等方法，而不是按历史轨迹密度采样。([SIAM Ebooks][6])

---

## 问题六：当前 ellipse 测试没有覆盖 ellipse 参数空间

当前局部 sealed ellipse 的尺度都约为：

$$
3\text{ mm}\times2\text{ mm}
$$

这只能检查：

> 模型在一个局部 block 中能否跟踪一条小闭环。

不能检查：

* 不同半长轴；
* 不同轴比；
* 不同中心；
* 不同平面姿态；
* 不同 chart；
* 不同壳体厚度层。

因此，下一轮不仅要扩大数据，还要重新设计 ellipse test family。

---

# 二、推荐的质变方案：Ellipsoidal Shell Canonical Atlas

建议将下一阶段命名为：

```text
BACRA V13 — Ellipsoidal Shell Canonical Atlas
```

核心思想是：

> 不再从椭圆轨迹生成壳体，而是先构造一个椭球壳体；然后利用椭球与平面的交线天然生成各种椭圆。

这是本方案最关键的几何设计。

---

# 三、为什么选择椭球壳体

定义中心椭球：

$$
\Sigma_\psi
===========

\left{
x:
\left[
Q^\top(x-c)
\right]^\top
D^{-2}
\left[
Q^\top(x-c)
\right]
=1
\right}
$$

其中：

$$
D=\operatorname{diag}(a_s,b_s,c_s)
$$

* (c) 是壳体中心；
* (Q\in SO(3)) 是壳体姿态；
* (a_s,b_s,c_s) 是三个半轴。

中心面上的点可以写为：

$$
s(u)=c+QDu,
\qquad
u\in S^2
$$

单位法向为：

$$
n(u)
====

\frac{
QD^{-1}u
}{
|D^{-1}u|
}
$$

厚壳体定义为：

$$
\boxed{
\Omega_{\psi,h}
===============

\left{
s(u)+\rho n(u):
u\in S^2,\
\rho\in[-h,h]
\right}
}
$$

这一表示有三个优势。

## 1. 它是显式三维壳体

坐标是：

$$
(u_1,u_2,\rho)
$$

两个面内自由度加一个厚度自由度。

## 2. 平面截取椭球天然得到椭圆

对任意平面：

$$
P(n_p,d)
========

{x:n_p^\top x=d}
$$

只要平面与椭球相交且不相切：

$$
\Sigma_\psi\cap P(n_p,d)
$$

就是一个椭圆或圆。

因此改变：

* 平面法向 (n_p)；
* 平面偏移 (d)；

就能系统改变：

* 椭圆平面姿态；
* 椭圆中心；
* 半长轴；
* 半短轴；
* 轴比。

不需要再人为设计大量彼此无关的 sin/sin/cos family。

## 3. 它使数据域和测试轨迹域具有统一几何关系

训练数据来自整个椭球壳体。

测试椭圆是壳体中心面的平面截线。

因此可以清楚声称：

> Student 学习了壳体区域中的 canonical inverse mapping，并在该区域内跟踪未见过的椭圆截线。

这比“用一组椭圆训练，再用相似椭圆测试”更有说服力。

---

# 四、完整实验流程

# Phase 0：从 capability map 搜索可行椭球壳体

## 0.1 输入

复用或重新生成完整 (\beta_6) Sobol FK pool：

$$
N_{\text{FK}}\approx1\text{M}
$$

每个样本保存：

* (x,y,z)；
* (\beta_1,\ldots,\beta_6)；
* joint margin；
* (\sigma_3)；
* (\kappa)。

Capability map 方法本身就是先对工作空间做离线离散和随机采样，再提取不同区域中的可达结构和方向性。([researchgate.net][7])

## 0.2 不搜索工作空间最外边界

本轮目标不是最大 workspace，而是：

> 找到一个较大、较厚、容易标注的内部壳体。

定义 capability score：

$$
q(x)
====

w_s\log(1+N_{\text{support}})
+
w_m m_{\max}(x)
---------------

## w_\kappa\log(1+\kappa_{\min}(x))

w_d d_{\text{NN}}(x)
$$

困难边界区可以不纳入本轮。

## 0.3 椭球搜索变量

搜索：

$$
\psi=
(c_x,c_y,c_z,Q,a_s,b_s,c_s,h)
$$

建议使用：

* Differential Evolution；
* CMA-ES；
* Sobol initial design + local refinement。

初始化可以来自 safe capability component 的 PCA，但最终评价必须使用完整 capability pool。

## 0.4 快速评价

每个候选椭球使用：

* (2048) 个等面积表面点；
* (\rho\in{-h,0,+h}) 三层；

进行 capability support 检查。

候选优先级采用字典序：

1. 中心面支持率；
2. (\pm h) 层支持率；
3. 最小/中位厚度；
4. 已支持表面面积；
5. (\kappa)；
6. joint margin。

## 0.5 Pilot 目标

先搜索：

$$
h\in{5,10,15,20}\text{ mm}
$$

本轮最低质变目标建议为：

$$
\boxed{
h_{10}\ge10\text{ mm}
}
$$

也就是至少 (80%\sim90%) 的有效表面位置，都能支持总厚度约 (20)mm 的壳体。

当前局部测试只有约 (3)mm、(2)mm 尺度；(20)mm 总厚度已经是明显的量级变化。

---

# Phase 1：建立二维中心面网格

不要使用经纬网格，因为两极会严重不均匀。

建议使用：

* icosphere；
* ellipsoid surface CVT；
* surface Poisson-disk。

## Pilot

$$
N_s=642
$$

个表面顶点。

## Formal

$$
N_s=2562
$$

个表面顶点。

每个顶点保存：

```text
shell_id
surface_vertex_id
surface_triangle_id
surface_u1
surface_u2
surface_xyz
surface_normal
surface_area_weight
```

当前 Student 数据缺少这些字段，下一版必须显式加入。

---

# Phase 2：在整个中心面生成多解 IK 候选

对每个表面顶点 (x_i) 生成：

$$
K=16\sim32
$$

组 distinct IK candidates。

候选来源：

1. 当前 Chart A 最近 anchor；
2. 当前 Chart B 最近 anchor；
3. full-(\beta) FK pool 的多个 (\beta)-cluster；
4. bounded SQP；
5. weighted DLS；
6. null-space perturbation；
7. 已经标注的相邻 surface vertex。

候选只要求：

* actual bounds 内；
* residual (\le3)mm；
* solver 成功。

探索阶段 joint margin 可以继续作为 Gold/Silver 权重，而不是统一阻断。

---

# Phase 3：构造 surface product graph

图节点：

$$
v_{i,k}=(x_i,\beta_{i,k})
$$

对于中心面网格中的相邻顶点 (i,j)，若：

1. 从 (\beta_{i,k}) 沿 (x_i\to x_j) 做 predictor-corrector 成功；
2. 结果落入 (x_j) 的候选 cluster (l)；
3. 构型变化不超过阈值；

则连接：

$$
v_{i,k}\leftrightarrow v_{j,l}
$$

这相当于把 IKLink 的“每 waypoint 多候选 + 图连接”从一维轨迹推广到二维表面网格。([arXiv][4])

---

# Phase 4：提取 canonical charts

在表面上均匀选择：

$$
N_{\text{root}}=24\sim32
$$

个 root vertices。

从每个 root candidate 向外做 atlas expansion。

当两个 expansion fronts 相遇时：

### 标签一致

若：

$$
d_\beta\le0.5^\circ
$$

则合并 chart。

### 标签冲突

若：

$$
d_\beta>1^\circ
$$

则保持为不同 chart，不强制平均。

### 无法继续

若局部没有合格 candidate，则形成 chart boundary。

输出：

```text
chart_id
surface_vertex_id
canonical_beta
candidate_count
parent_chart
overlap_chart
overlap_beta_gap
```

目标不是强制一张 chart 覆盖整个壳体，而是找到能够覆盖大面积表面的少数 charts。

## Pilot 通过条件

* 最大 chart 覆盖中心面面积 (\ge60%)；
* 所有 charts 合计覆盖 (\ge80%)；
* chart overlap P95 (\le0.5^\circ)；
* 未解决冲突区域可以裁掉。

AtlasRRT 和 higher-dimensional continuation 的核心思想，就是通过局部 chart 协同覆盖不能被单一参数化表示的流形。([DOI][8])

---

# Phase 5：沿法向扩展壳体厚度

在每个已接受中心面节点上，生成：

$$
\rho
\in
{-20,-15,-10,-5,0,5,10,15,20}\text{ mm}
$$

先从 (\pm10)mm Pilot 开始，通过后再扩到 (\pm20)mm。

每个 radial point 同时使用：

1. 同一 surface vertex 的前一 radial layer；
2. 相邻 surface vertices 的同层节点；
3. 中心面的 canonical beta；

作为三个独立父节点。

只有多个父节点给出的 corrected (\beta) 近似一致时才接受。

这一步得到真正的：

$$
\beta(u_1,u_2,\rho)
$$

而不是椭圆走廊。

## 厚度评价

对每个 surface vertex 记录：

$$
h_i^-,h_i^+
$$

报告：

$$
h_{\min},\quad h_{10},\quad h_{50},\quad h_{90}
$$

困难区域可以剔除，然后保留最大连通壳体组件。

---

# Phase 6：稠密、均匀地生成壳体数据

稀疏 atlas 通过后，再生成：

## Pilot

$$
50{,}000
$$

行。

## Formal

$$
150{,}000\sim250{,}000
$$

行。

采样不能再按历史 trajectory density。

采用：

* 等面积 surface cell；
* 分层 thickness bin；
* cell 内 Poisson-disk 或 CVT；
* 每个 surface cell × radial bin 相近样本数。

建议训练数据构成：

* (80%) 新 shell-uniform 数据；
* (20%) 现有 Chart A/B 和 legacy final8 retention 数据。

每行保存：

```text
shell_id
chart_id
surface_vertex_id
surface_triangle_id
surface_u1
surface_u2
rho_mm
normal_x/y/z
x/y/z
beta1...beta6
teacher_residual
candidate_gap
joint_margin
kappa
quality_class
```

---

# Phase 7：系统生成多尺度椭圆测试集

对中心椭球随机生成平面：

$$
P(n,d)
$$

并计算截线：

$$
E(n,d)=\Sigma_\psi\cap P(n,d)
$$

对每条截线重新计算真实：

* semi-major (a_e)；
* semi-minor (b_e)；
* axis ratio (r_e=b_e/a_e)；
* ellipse center；
* plane normal；
* plane orientation；
* chart sequence。

## 参数分层

### 半长轴

按最大可行截线半长轴 (a_{\max}) 的比例分五层：

$$
a_e/a_{\max}
\in
[0.2,0.35],
[0.35,0.5],
[0.5,0.65],
[0.65,0.8],
[0.8,0.95]
$$

要求最大/最小测试尺度至少相差：

$$
\boxed{
3\times
}
$$

### 轴比

$$
b_e/a_e
\in
[0.3,0.45],
[0.45,0.6],
[0.6,0.8],
[0.8,1.0]
$$

### 平面姿态

使用 (12) 个近似等面积法向方向 bin。

### 中心位置

使用平面 offset 的四个等级：

$$
|d|/d_{\max}
\in
[0,0.25],
[0.25,0.5],
[0.5,0.75],
[0.75,0.9]
$$

从这些组合中用 maximin/LHS 选取：

$$
120
$$

条完整椭圆。

其中：

* (40) 条用于开发；
* (80) 条在模型锁定后评估。

每条使用 (360) 或 (720) 个点。

只有整条椭圆都位于已接受 shell/chart 中时才进入测试。

---

# Phase 8：Student 训练

本轮不要同时大改网络。

继续使用当前已经成功的：

* bounded output；
* FK loss；
* row-space loss；
* known chart id；
* frozen-trunk + chart residual 或 per-chart experts。

这样可以把提升明确归因于数据集几何，而不是模型变化。

训练三个对照：

## D0：当前 V12.16C 数据

$$
88{,}976\text{ rows}
$$

## D1：相同样本量的 shell-uniform 数据

$$
88{,}976\text{ rows}
$$

这是最重要的对照，能够判断提升是否来自数据分布而不是样本数。

## D2：完整 shell 数据

$$
150k\sim250k
$$

---

# 五、实验成功的“质变”标准

本轮不需要覆盖最困难工作空间，但至少应同时满足以下四项。

## 1. 壳体几何成立

* 显式中心面、法向和厚度坐标；
* 中心面总覆盖率：

$$
R_A\ge80%
$$

* 有效壳体体积覆盖率：

$$
R_V\ge60%
$$

* 厚度：

$$
h_{10}\ge10\text{ mm}
$$

$$
h_{50}\ge15\sim20\text{ mm}
$$

## 2. 数据不再是轨迹走廊

* shell cell × radial bin 的采样密度 CV 明显下降；
* Cartesian fill-distance：

$$
h_{95}\le5\text{ mm}
$$

$$
h_{\max}\le10\text{ mm}
$$

## 3. 椭圆参数域具有跨度

* semi-major 最大/最小比 (\ge3)；
* axis ratio 覆盖 (0.3\sim1.0)；
* 至少 8 个平面姿态 bin；
* 至少 3 个中心 offset bin。

## 4. 一个 Student 能在该参数域内工作

* 随机 shell point：

$$
FK_{\mathrm{P95}}\le5\text{ mm}
$$

$$
FK_{\max}\le10\text{ mm}
$$

* 至少：

$$
80%
$$

的未见椭圆通过；

* 每个半径、轴比和姿态 bin 的通过率不低于 (60%)；
* legacy final8 不出现明显退化。

这才可以称为从“局部走廊泛化”到“壳体区域泛化”的质变。

---

# 六、必须保留的消融实验

| 方法 | 几何采样  | 标签方法                | 目的              |
| -- | ----- | ------------------- | --------------- |
| A0 | 当前走廊  | 当前 Teacher          | 现有基线            |
| A1 | 壳体均匀点 | 每点独立 IK             | 展示 branch 冲突    |
| A2 | 壳体均匀点 | 单 root continuation | 展示单 branch 覆盖限制 |
| A3 | 壳体均匀点 | 多候选 + atlas         | 推荐方法            |

四种数据使用相同样本量和相同 Student。

若 A3 明显优于 A0，而 A1 存在高 local-beta conflict，就能说明改进来自**壳体 atlas 数据生成方法**，不是简单增加数据量。

---

# 七、如果完整椭球壳体搜索失败

不要退回椭圆走廊。

依次采用：

1. 两个重叠的半椭球 shell patches；
2. 2–4 个局部椭球壳片；
3. 每个壳片独立 chart；
4. 只保留能够形成完整平面截线的区域。

当前 V12.16C 已经证明 known-chart gated residual Student 可行，因此多 chart 不是当前核心障碍。

---

# 八、给 Codex 的直接任务定义

> Implement BACRA V13 as an explicit ellipsoidal-shell canonical-atlas experiment. Do not derive the new task domain by dilating existing ellipse or trajectory seeds. Use the branch-agnostic full-beta capability pool to search for a large, well-conditioned interior ellipsoid and a physical normal thickness of at least 10 mm half-thickness over a large connected surface fraction. Mesh the ellipsoid with an approximately equal-area triangular surface grid. At every surface vertex generate multiple distinct IK candidates using existing Chart A/B anchors, FK-pool clusters, bounded SQP, weighted DLS, null-space perturbations, and neighboring surface vertices. Build a task/configuration product graph whose edges are verified by predictor-corrector continuation. Grow multiple surface charts from spatially distributed roots, merging consistent fronts and keeping inconsistent fronts as separate chart IDs. After labeling the center surface, extend every chart along the ellipsoid normal through stratified radial layers and accept radial samples only when multiple independent parents agree. Densify the accepted shell using equal-area surface cells and stratified thickness bins rather than trajectory density. Generate plane sections of the center ellipsoid to obtain systematic ellipse families spanning multiple semi-major-axis, axis-ratio, center-offset, and plane-orientation bins. Train the current chart-conditioned bounded Student without architecture changes and compare the existing corridor dataset against an equal-size shell-uniform dataset and a larger shell dataset. Report surface-area coverage, shell-volume coverage, thickness quantiles, fill distance, local beta consistency, chart overlap, random shell-point FK error, and pass rates over the stratified unseen ellipse family.

---

# 最终结论

目前泛化提升小，不是因为：

* Student 太小；
* Teacher 不会跟踪；
* 数据还不够多。

根本原因是：

$$
\boxed{
\text{当前数据生成器始终继承了历史椭圆seed的几何拓扑}
}
$$

它所生成的是：

$$
\text{一条越来越密的走廊}
$$

而不是：

$$
\text{一张二维中心面加一维厚度的壳体}
$$

下一轮真正需要的质变是：

$$
\boxed{
\text{轨迹驱动采样}
\rightarrow
\text{显式壳体驱动采样}
}
$$

并且：

$$
\boxed{
\text{逐点或单路径branch选择}
\rightarrow
\text{二维表面product graph与多chart atlas}
}
$$

椭球壳体尤其适合当前目标，因为训练区域可以独立于椭圆轨迹建立，而各种半径、轴比、中心和姿态的椭圆又可以作为该壳体的平面截线系统生成。这样，数据集几何、Teacher 一致性和 Student 泛化测试才能第一次形成统一、可计算、可扩展的方法体系。

[1]: https://journals.sagepub.com/doi/10.1177/0278364910396389 "https://journals.sagepub.com/doi/10.1177/0278364910396389"
[2]: https://journals.sagepub.com/doi/10.1177/0278364911432324 "https://journals.sagepub.com/doi/10.1177/0278364911432324"
[3]: https://arxiv.org/abs/2012.13965 "https://arxiv.org/abs/2012.13965"
[4]: https://arxiv.org/abs/2402.16154 "https://arxiv.org/abs/2402.16154"
[5]: https://di.ku.dk/english/research/groups/simulation/research-results/local-models-for-data-driven-inverse-kinematics-of-soft-robots/ "https://di.ku.dk/english/research/groups/simulation/research-results/local-models-for-data-driven-inverse-kinematics-of-soft-robots/"
[6]: https://epubs.siam.org/doi/10.1137/S0036144599352836 "https://epubs.siam.org/doi/10.1137/S0036144599352836"
[7]: https://www.researchgate.net/publication/224296369_Capturing_Robot_Workspace_Structure_Representing_Robot_Capabilities "https://www.researchgate.net/publication/224296369_Capturing_Robot_Workspace_Structure_Representing_Robot_Capabilities"
[8]: https://doi.org/10.1007/978-3-642-17452-0_20 "https://doi.org/10.1007/978-3-642-17452-0_20"
/