# True Ellipse Standard-Domain V7 实验记录

## 1. 目标与结论口径

V7 接续 V6 已经物化的固定 family `c0273_a100_py210_pz330_s0243`，处理的首要问题不再是 100 mm 的点可达性，而是 V6 标签在 beta3/beta4 硬边界上几乎没有余量，导致无约束静态 MLP 的微小误差触发 raw-bound gate。

本轮预注册以下边界：

- 使用项目标准关节域：beta1/beta2=`±5°`、beta3/beta4=`±10°`、beta5/beta6=`±15°`；
- family、中心、相位与形状比例保持固定，不通过换 family 绕过困难角度；
- 从 75 mm 连续延拓，严格 checkpoint 至少需要达到 105 mm，探索上限为 120 mm；
- strict path bundle 与 exploratory pointwise admission 分开报告，后者不得改写前者；
- 标签余量 gate 固定为 min≥`0.05°`、P01≥`0.10°`、P05≥`0.25°`，out-of-bounds/at-bound 均为 0；
- 只有 support-backed strict checkpoint `Rtest≥105 mm` 才能注册模型 test，`Rval=Rtest−7.5 mm`；
- 100 mm 以上训练 anchor 位于每个 2.5 mm checkpoint 区间中点，即 `+1.25 mm`；
- integer centerline 与 `(k+0.5)°` half-phase challenge 是正式模型 gate，tube 只作模型诊断；
- 24 个 V5 base config 与 identity/tanh-bounds 两种 output link 做笛卡尔积，共 48 个配置；tanh 是训练前注册的输出参数化，不是查看 test 后的裁剪；
- 上游任何正式 gate 失败时，不生成正式数据集、不读取注册 test、不运行正式训练，也不提高静态逆模型半径。

因此，V7 的“实现完成”和“105 mm strict 阳性”是两个不同命题。代码、审计和训练协议可以完整实现，而实验仍可能在几何条件数门上给出严格阴性。

## 2. 标准关节域与标签内点化

V7 在共享引擎中注册命名关节域及 SHA-256 fingerprint，所有 FK/IK、margin、模型评估和输出 link 都通过该领域对象解释 beta。V6 默认仍使用原 `current_v6` 域，不因 V7 扩大 beta3/beta4 而改变冻结语义。

标签内点化包含三层：

1. pointwise/trajectory 求解使用标准域 bounds；
2. cyclic radial correction 与 5×5 tube surface optimization 加入平滑 joint-limit barrier，正式权重 `lambda_margin=0.01`、soft margin=`0.25°`；
3. 每条 centerline、每个 tube surface 和最终数据集都记录总体及逐关节 min/P01/P05 margin，并按 fail-closed 策略验收。

这使“机械上在 bounds 内”与“为模型误差留出可复现余量”成为两个显式审计项。

## 3. 径向路径面

径向生成保留 V6 的完整 parent surface 思路：

- predictor=`parent_copy, radial_secant`；
- cyclic cut=`0,90,180,270`；
- anchor schedule=`conservative, balanced, loose`；
- 每一 stage 均保留正 radial anchor；
- 直接联合修正失败后才运行 candidate graph rescue；
- 普通角度最多 8 个 seed；predictor Jacobian kappa≥100 的角度最多 24 个 seed；
- candidate graph 在相同 predictor 的四个 cut 间可复用，但缓存 fingerprint 同时绑定 targets、完整 predictor、pointwise pool、关节域、候选预算、IK 参数、随机 seed、`lengths_m`、`p_end_local_m` 与 `theta_sign`；
- V7 在首个必需 job 失败时短路该半径；V6 默认仍完成全部 predictor/cut 诊断，并恢复 per-cut `seed+cut_idx` 与 cut-local cache。

严格 checkpoint 为：

```text
75, 80, 82.5, 85, 87.5, 90, 92.5, 95, 97.5,
100, 102.5, 105, 107.5, 110, 112.5, 115, 117.5, 120 mm
```

checkpoint 之间以 1 mm 走步，失败后仅允许预注册的 0.5/0.25 mm 回退。中间半径可用于定位连续边界，但不能冒充注册 checkpoint。

## 4. 联合 tube surface 与数据集

对每个获准数据半径，V7 把 25 条独立 tube curve 升级为 `phase × n1 × n2` 联合 surface：

- 法向 offsets 固定为 `[-5,-2.5,0,2.5,5] mm²`；
- 初值由 centerline Jacobian 对整张网格一次性预测，不先做 9,000 次独立 pointwise IK；
- 优化同时约束角向周期平滑、n1/n2 四邻接平滑、centerline/parent anchor、Cartesian residual 与 joint margin；
- 四个 cyclic cut 独立计算并验证 cut invariance，正式 360 点任务可并行执行；
- stage schedule 在首个完整 gate pass 后停止，不继续改变已通过标签。

若 strict geometry frontier 超过 100 mm，数据半径为所有不超过 frontier 的注册 checkpoint，加上 100 mm 以上的 1.25 mm 中点 anchor。候选动态 test 的 support 只使用该候选以下的非中心线训练样本，同时排除 validation/test 整半径；若 test 从更大 checkpoint 回退，所有 `radius>Rtest` 的样本既不参与 support，也不进入最终训练集。

正式数据集另外物化一个不含 test 行的 non-formal parquet。smoke/pilot 的 audit、split、worker task 与 final fingerprint 只解析该安全视图及 validation challenge；只有 formal 配置锁定后才允许打开完整数据集和 test challenge。

## 5. 模型协议

训练输入固定为 `x_target_m,y_target_m,z_target_m`，不读取 radius、angle、family、branch 或 split 作为特征。模型网格为：

```text
24 V5 base configs × {identity, tanh_bounds} = 48 configs
```

`tanh_bounds` 将物理 beta 映射到无界 latent target，并在模型输出端通过注册的 tanh link 解码到关节域内点。正式训练要求 target encode clip count 为 0；identity 与 tanh 都按原始解码输出评估，禁止额外 post-hoc clipping。

screen 只能读取 validation half-phase challenge。锁定配置后，五个 seed 才评估：

- validation integer centerline；
- validation half-phase centerline；
- test integer centerline；
- test half-phase centerline；
- validation/test tube diagnostic（不进入 formal model gate）。

正式阳性要求 validation/test 的 integer 与 half-phase 四组 gate 均达到 5 seeds 中至少 4 个通过。

## 6. 正式实验结果

最终 Python 3.11 无缓存重跑中，已通过注册 checkpoint 的 canonical 指标如下。margin 一列为 `min/P01/P05`：

| radius (mm) | residual P95/max (mm) | delta beta P95/max (deg) | kappa P95 | sigma3 P05 (m) | margin min/P01/P05 (deg) | cut pair P95 max (deg) |
|---:|---:|---:|---:|---:|---:|---:|
| 82.5 | 0.000294/0.000334 | 0.035460/0.190729 | 17.0712 | 0.361066 | 2.072547/2.073716/2.095612 | 0.001147 |
| 85.0 | 0.000382/0.000441 | 0.037560/0.154224 | 18.3179 | 0.337619 | 2.059006/2.060252/2.083614 | 0.001709 |
| 87.5 | 0.000512/0.000608 | 0.039841/0.136799 | 19.8730 | 0.312245 | 2.045533/2.046868/2.071738 | 0.002469 |
| 90.0 | 0.000727/0.000894 | 0.042349/0.124991 | 21.8857 | 0.284483 | 2.032097/2.033478/2.059984 | 0.003618 |
| 92.5 | 0.001121/0.001454 | 0.045164/0.116174 | 24.6285 | 0.253651 | 2.018734/2.020163/2.048360 | 0.005402 |
| 95.0 | 0.001960/0.002779 | 0.048563/0.109311 | 28.6640 | 0.218668 | 2.005445/2.006921/2.036309 | 0.008261 |
| 97.5 | 0.004246/0.007131 | 0.053056/0.103831 | 35.3922 | 0.177683 | 1.992228/1.993752/2.024096 | 0.013938 |
| 100.0 | 0.014027/0.033987 | 0.060138/0.099383 | 49.5257 | 0.127382 | 1.979084/1.980657/2.011967 | 0.024386 |
| 102.5 | 0.102481/0.293968 | 0.084511/0.095710 | 91.8995 | 0.068846 | 1.966012/1.967634/1.999922 | 0.070295 |

100 mm 的逐关节最小余量为：

```text
beta1  3.543535°
beta2  1.979084°
beta3  4.764884°
beta4  4.678082°
beta5 13.779495°
beta6  2.713108°
```

这与 V6 的 beta3=`0.009656806°`、beta4≈`5.73×10⁻⁹°` 形成直接对照：标准域和 margin-aware relabel 已消除模型 raw-bound 失败的标签根因。

### 6.1 严格径向边界

最终 radial walk 共记录 39 次尝试：

```text
last continuous passing radius      104.0 mm
last registered strict checkpoint   102.5 mm
first failed strict checkpoint      105.0 mm
formal_radial_gate_pass              false
```

104.0 mm 中间路径通过全部 8 个 predictor/cut job，其 canonical 指标为：

```text
residual P95/max       0.301300 / 0.911929 mm
kappa P95              142.5808
sigma3 P05             0.0444465 m
margin min/P01/P05     1.958202 / 1.959854 / 1.992735 deg
```

从 104.0 mm 继续时，104.25、104.5 和 105.0 mm 都在第一个必需 `parent_copy/cut=0` job 失败。三者的 direct schedule 代表值如下：

| radius (mm) | schedule | residual P95/max (mm) | kappa P95 | sigma3 P05 (m) |
|---:|---|---:|---:|---:|
| 104.25 | conservative | 0.222208/1.136071 | 216.9825 | 0.029232 |
| 104.25 | balanced | 0.176841/1.135122 | 204.8652 | 0.030966 |
| 104.25 | loose | 0.154569/1.134781 | 183.5939 | 0.034556 |
| 104.50 | conservative | 0.370704/1.382333 | 256.1301 | 0.024770 |
| 105.00 | conservative | 0.733902/1.876723 | 361.9766 | 0.017533 |
| 105.00 | balanced | 0.687103/1.876530 | 429.4605 | 0.014788 |
| 105.00 | loose | 0.666246/1.876505 | 441.9038 | 0.014379 |

这些失败路径的 joint margin 仍约为 1.96° 或更大，Cartesian residual 也没有发生不连续爆炸；条件数从 104.0 mm 的 142.6 在 0.25 mm 内跃升到至少 183.6，成为唯一决定性瓶颈。若把 kappa gate 从 150 放到 160，仍无法通过 104.25 mm；要直接接纳 105 mm 至少需要 2.4 倍以上的 materially different 放宽，不能称为“适当微调”。

三次困难半径都启用了高条件数 24-seed rescue。以 105 mm 为例，2,962 个 raw solutions 经 0.25° 去重后只剩 360 个，即每角仍恰好一个解簇；candidate graph 没有找到另一条低 kappa 连续支路。

### 6.2 探索性点可达边界

strict walk 失败后，使用最后通过的 104 mm path 作为 seed，对未通过 checkpoint 做 72 点 exploratory scan：

| radius (mm) | success ratio ≤2 mm | residual P95 (mm) | residual max (mm) | admission |
|---:|---:|---:|---:|---|
| 105.0 | 1.000000 | 0.361108 | 1.876471 | pass |
| 107.5 | 0.930556 | 2.801548 | 4.360621 | pass |
| 110.0 | 0.902778 | 5.257461 | 6.860541 | pass |
| 112.5 | 0.875000 | 7.728743 | 9.376126 | pass |
| 115.0 | 0.875000 | 10.215293 | 11.907269 | pass |
| 117.5 | 0.847222 | 12.717009 | 14.453865 | pass |
| 120.0 | 0.847222 | 15.232731 | 17.015809 | pass |

所以 `exploratory_rescue_rmax_mm=120`，但这只证明预注册 admission 条件下仍存在大量域内 pointwise 解，不证明 360 点、双 predictor、四 cut、低条件数且可重复的严格路径。102.5 mm 只能记为径向阶段的 `radial_strict_rmax_mm`；tube 未获授权，因此整体 `strict_geometry_rmax_mm=null`，pointwise 结果不能改写这两个边界。

### 6.3 下游结论

由于 strict radial checkpoint 没有达到 105 mm：

- tube 阶段 fail-closed，`strict_tube_rmax_mm=null`，没有物化 V7 tube；
- dataset 阶段 fail-closed，没有注册 validation/test、没有生成 V7 正式训练 parquet；
- half-phase challenge 未生成；
- 48-config screen 与五 seed training 没有启动；
- `model_training_authorized=false`，没有新的静态逆模型半径声明。

V7 因而证明了两点：一是标准域与 margin-aware 标签已经彻底解决 V6 的贴界根因；二是同一 fixed family 的下一个真实瓶颈位于 104.0–104.25 mm 的 Jacobian 近奇异边界。最终“可拟合椭圆半径”仍不能从 V4 的 81.25 mm 正式上调，因为本轮没有获得获准训练的新 holdout。

### 6.4 椭圆跟踪轨迹展示

可视化沿用项目现有的 Matplotlib/Agg 生成方式：PNG、180 dpi、`bbox_inches="tight"`，目标轨迹使用深蓝虚线，实际 IK+FK 使用橙色实线。3D 视角不再使用固定方位角，而是从目标轨迹的 SVD 平面法向自动计算近正视方向，并偏转 12° 保留立体感，避免椭圆被侧视成一条直线。

100、102.5 和 104 mm 的严格/连续通过轨迹如下。100 与 102.5 mm 是注册 strict checkpoint；104 mm 是完整 360 点、双 predictor、四 cut 均通过的连续中间路径，但不是注册 checkpoint。右列同时给出逐轴误差，因此目标与实际曲线重叠时仍能读出跟踪差异。

![V7 strict and continuous true-ellipse tracking](../runs/true_ellipse_standard_domain_v7/05_visualization/strict_tracking_overview.png)

同三条路径的六关节轨迹如下。阴影与虚线表示注册关节域；beta3/beta4 在扩大到 ±10° 后保持明显内点余量。

![V7 strict joint profiles](../runs/true_ellipse_standard_domain_v7/05_visualization/strict_joint_profiles.png)

104、104.25 和 105 mm 的前沿对比如下。Cartesian residual 仍处于原分支 gate 内，但 kappa P95 从 142.6 上升到 183.6 和 362.0；图中对数坐标同时标注 P95/P05 门统计与逐点极值，避免把逐点越过参考线误读为 percentile gate 结论。

![V7 conditioning frontier](../runs/true_ellipse_standard_domain_v7/05_visualization/conditioning_frontier.png)

探索性 105、110、115、120 mm 只显示独立 pointwise IK：深蓝虚线是目标椭圆，绿点为重算 residual≤2 mm 的 FK 点，红点为 residual>2 mm 的 FK 点。点之间不连线，标题也明确标为 `EXPLORATORY pointwise IK`，不能把这张图解释成连续 branch。

![V7 exploratory pointwise IK](../runs/true_ellipse_standard_domain_v7/05_visualization/exploratory_pointwise_overview.png)

#### 为什么正式 V7 没有模型训练表现图

这里没有遗漏模型曲线。V7 的连续几何路径止于 104.0 mm，最后注册 strict checkpoint 为 102.5 mm，低于协议要求的 105 mm 最低训练门槛；因此 tube、正式 dataset、holdout、48-config screen 与五 seed final 全部 fail-closed，`model_training_authorized=false`。105–120 mm 仅为 72 点独立 pointwise IK，不是可用于正式拟合的连续轨迹数据集。

V6 曾完成模型拟合：24 个配置中选出一个 MLP 后运行五个 seed，92.5 mm validation 为 1/5、100 mm test 为 0/5，唯一失败项是 beta3/beta4 raw-bound violation。该结果属于 V6，不能作为 V7 的正式模型表现图。

后续已经按这里定义的边界另建 `V7-D1` non-formal centerline-only 诊断训练：使用 75–100 mm 的 30 条完整中心线训练 48 个配置，以 102.5 mm 锁定配置，再对最佳配置运行五个 seed。诊断稳定通过 100/101/102/102.5 mm，103.5 mm 为 2/5、104 mm 为 1/5；失败来自 Cartesian 逐轴误差，不再来自关节越界。48 个模型各自的 100/102.5/104 mm 轨迹图、contact sheet、误差热图、五 seed 对比和 105–120 mm 目标外推图均已生成。

详细协议、指标与证据边界见：

```text
docs/TrueEllipseStandardDomainV7诊断中心线模型实验.md
runs/true_ellipse_standard_domain_training_v7_diagnostic_centerline/
```

V7-D1 全部产物都标记 `diagnostic_only=true`、`formal_claims_allowed=false`，因此不修改本文件的正式 V7 阴性结论。

图片及输入证据 SHA 记录于：

```text
runs/true_ellipse_standard_domain_v7/05_visualization/visualization_report.json
visualization_integrity_gate_pass=true
visualization_only=true
changes_formal_gate=false
```

## 7. Fail-closed 与提交边界审查

V7 的四阶段缓存不再只比较 protocol 字符串：

- audit 绑定 V6 四份报告、V6 checkpoint 文档和 robot config 的 SHA；
- radial 绑定 audit task fingerprint、全部 strict path artifact 和 radius-status SHA；
- tube 绑定 radial task fingerprint、summary SHA 和每个通过 tube artifact SHA；
- dataset 绑定 tube task fingerprint、attempt/manifest/support、两条 challenge 及 formal/non-formal dataset SHA。

相同协议下只要上游报告或 parquet 被替换，ensure 链就会重算而不是沿用旧阳性。V7 还将正式 runtime 写入 protocol fingerprint，并要求：

```text
Python 3.11.*
numpy 1.26.4
scipy 1.11.4
pandas 2.2.2
pyarrow 16.1.0
```

提交边界做了 Standards 与 Spec 双轴审查及一次 consolidated re-review。已关闭的高优先级 finding 包括：运动学未进入 rescue cache、上游 cache 非 fail-closed、non-formal 仍触碰 test、回退 Rtest 使用更大半径、V7 性能优化改变 V6 失败诊断、以及错误 runtime 不被协议识别。保留的非阻断债务是 V7 runner 仍偏大，后续可按 phase 拆模块；本轮不为架构美化改动已冻结数值协议。

## 8. 运行命令

正式运行使用：

```bash
env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/tmp/quasi-exp-py311-packages \
  LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib \
  /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 \
  scripts/analysis/run_true_ellipse_standard_domain_v7.py \
  --preset formal --phases radial --workers 12
```

下游阶段只在 radial formal gate 通过时物化；若低于 105 mm，tube/dataset/summary 写出明确的阻断报告，training runner 不启动。

轨迹图使用相同的声明 Python 3.11 环境生成：

```bash
env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH=/tmp/quasi-exp-py311-packages \
  LD_LIBRARY_PATH=/tmp/quasi-exp-libs:/mnt/ML_projects/conda_envs/quasi_exp/lib \
  /mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11 \
  scripts/analysis/plot_true_ellipse_standard_domain_v7.py \
  --run-dir runs/true_ellipse_standard_domain_v7 \
  --out-dir runs/true_ellipse_standard_domain_v7/05_visualization \
  --robot-config configs/robot_rods_only_standard_100k.yaml
```

## 9. Gate-v2 后续实验：解耦 KPI、训练授权与候选策略

本节是独立的 gate-v2 后续实验，不追溯改写第 6 节已经冻结的 V7 结论。运行目录分别为：

```text
runs/true_ellipse_standard_domain_v7_gate_v2/
runs/true_ellipse_standard_domain_v7_gate_v2_99pct/
runs/true_ellipse_standard_domain_training_v7_gate_v2_99pct/
runs/true_ellipse_standard_domain_gate_sweep_v7/
```

gate-v2 保留 `kappa P95 <= 150` 作为 legacy strict conditioning gate，同时把下列语义拆开：

- `target_105_achieved`：只回答连续、注册的几何/证据前沿是否达到 105 mm；
- `downstream_radial_admission_gate_pass`：当前存在完整、可审计的 strict frontier，可继续生成 tube/data 证据；
- `formal_dataset_gate_pass`：当前正式协议、tube、support、split、challenge 与 artifact 审计全部通过；
- `model_training_authorized`：只依赖当前已注册数据链，不再把 105 mm KPI 当作训练短路条件；
- `formal_model_gate_pass`：正式五 seed 模型结论；候选 κ policy 的训练固定为 `evidence_only`，不得授予正式 claim。

因此，在 105 mm KPI 仍为阴性时，只要当前 strict frontier 的 tube/data/support 完整，仍可注册该前沿上的动态 holdout 并运行正式模型。test 从当前最高 support-backed strict checkpoint 向下选择，validation 固定为 `Rtest - 7.5 mm`；更大半径既不参与 support，也不进入训练集。smoke/pilot 仍只能读取物理上不含 test 行的 non-formal parquet。

radial 与 tube 前沿不要求数值相等：tube 会一直尝试到 radial strict frontier，但正式数据前沿取“不超过 radial 前沿、且其以内所有要求半径连续通过”的最高注册 checkpoint。第一个失败半径以及更外层结果只作为诊断保留，不进入 dataset。这样 102.5 mm tube 若失败，已经完整通过的 100 mm 链仍可注册 100/92.5 mm 动态 holdout；若 101.25 training anchor 与 102.5 mm 都通过，则自动注册 102.5/95.0 mm。该回退规则在读取两个外层最终结果前固定，不按事后模型表现选择半径。

候选 conditioning policy 只扫描预注册阈值：

```text
150, 200, 250, 300, 400
```

每个阈值必须回放两个 predictor × 四个 cyclic cut、cut invariance、deterministic repeatability 与 joint margin。若某个较早 optimizer stage 通过候选 κ，而 legacy 最终 stage 没有通过，候选 policy 必须实际物化那个较早 stage；只有历史指标而没有对应路径时，`candidate_paths_materialized=false`，禁止注册。

候选 policy 的注册还要求径向候选前沿、tube 前沿、dataset 前沿和最终 test radius 四者完全一致。较小的完整 tube 前沿仍可用于训练一个有效模型，但不能反过来证明更大的 κ 候选径向前沿已经获得完整下游证据；此时 `frontier_consistency_gate_pass=false`，policy recommendation 必须 fail-closed。

candidate graph rescue 采用完整角层 fail-closed 语义：输入和输出都必须与预期 360 个 `angle_idx` 完全一致。缺失任一角层时返回空 graph，并记录 `incomplete_candidate_angle_layers`、缺失角度和 coverage；不再允许不完整的 338/341 点图路径进入 360 点联合修正。

sweep 允许只读复用 gate-v2 证据，但必须同时满足 frontier 相同、径向 manifest 的每个 Parquet SHA 完全相同、tube summary 及每个 tube artifact SHA 当前、正式/证据 dataset 与两条 challenge SHA 当前。任一字节不一致即退回独立物化；复用时仍为每个候选 policy 写独立 task fingerprint，并强制 `formal_claims_allowed=false`。

正式 tube 数值链运行到 100 mm 后又暴露出一处与项目既有协议不一致的重复硬门：V3/V5 的注册 tube gate 本来允许 `target_success_ratio >= 0.99`，同时继续要求 coverage、P95/max residual、局部 beta 平滑性和 multi-branch 全部合格；V7 却在这些门之后额外要求 `tube_success_ratio == 1.0`，数据集完整性阶段还再次要求每一行 `tube_success=true`。这两个条件把同一个成功率门无依据地收紧成了 100%。

因此在读取 101.25/102.5 mm 最终结果之前，gate-v2 后续实验预注册以下修正：

- tube label 与 dataset trajectory completeness 统一复用 V3/V5 的 `>=0.99` 成功率下限；
- residual P95/max、25-point normal grid、coverage、局部 beta RMS、multi-branch、joint margin、四 cut、cut invariance 均保持不变；
- 旧的 `==1.0` 报告原样保留在 `true_ellipse_standard_domain_v7_gate_v2`，不原地覆盖；
- 新门只通过独立 gate-only replay 应用，逐一绑定 source report、centerline 与 tube artifact SHA，并从 Parquet 重新计算 geometry/margin 指标；
- 任一 hash、四 cut、任务指纹或重算指标不一致即 fail-closed；新结果写入 `true_ellipse_standard_domain_v7_gate_v2_99pct`。

这项修改不是放松 Cartesian 误差或几何连续性，而是删除 V7 新增的重复 100% 成功率条件，使正式 label gate 与项目原有 tube gate 恢复同一语义。gate policy、tube strategy 和 dataset strategy 均已升版，新的 0.99 下限进入 protocol/task fingerprint。

### 9.1 0.99 gate-only replay 与完整数据前沿

旧 V7 数值产物保持只读。独立 replay 对 source report、centerline、tube surface 和
四个 cyclic cut 的 hash 逐一校验，并从 Parquet 重算 Cartesian residual、joint
margin、coverage 与 tube success。100 mm 在统一的 0.99 成功率语义下通过：

```text
success ratio             0.9931111111
coverage                  1.0
residual P95 / max        0.0532366 / 2.4795384 mm
beta local RMS P95        0.955817 deg
multi-branch ratio        0
```

101.25 mm training anchor 是完整前缀的第一个失败点。其最好/最终 success ratio 为
0.989333/0.988667，最终 residual max=3.685917 mm、beta local RMS
P95=1.175044°，分别失败 `success>=0.99`、`residual max<=3 mm` 和
`beta local RMS P95<=1°`。这三项均是未修改的项目原门，因此没有继续以单一 gate
放宽解释该失败。正式 tube/data 前沿按预注册规则回退到 100 mm；102.5 mm 不能绕过
失败的 101.25 mm anchor 形成完整训练前缀。

为完整回答外层是否仍有提升空间，102.5 mm 随后也完成了四 cut、双向 surface
sweep 和三阶段 whole-surface joint correction。初始 surface 的 success ratio
为 0.949667、residual P95/max 为 1.515852/10.748114 mm；最终
`balanced_3` 将其改善为 0.978667、0.140529/4.896392 mm，beta local RMS P95
为 1.307521°。coverage=1、multi-branch=0、cut invariance 和 joint margin 均通过，
但它同样失败未修改的 success、residual max 与 local-RMS 三门，不能进入数据集。

失败点具有高度局部的结构。101.25 mm 的 102 个失败样本中 98 个位于
`delta_n2=-5 mm` 外壳，主要覆盖 78°–101°，另有 55°–56°；最坏点为 84°、
`delta_n1=+5 mm, delta_n2=-5 mm`，residual=3.685917 mm。102.5 mm 的 192 个
失败样本覆盖 55°–56° 与 76°–103°，负 `delta_n2` 外壳仍占主导，最坏点仍是
84°、`(+5,-5) mm`，residual=4.896392 mm。因此下一轮最有价值的改动是对该相位段
的负 n2 outer shell 做定向整曲线修正，而不是全局放松三个质量门。

最终数据集为 90,000 行、10 个完整半径、单 family。validation=92.5 mm、
test=100 mm，训练使用其余 8 个半径的 69,120 行非中心线样本；trajectory/radius
leakage 均为 0，sample ID 全部唯一，conflict voxel=0，最大 voxel beta
RMS=0.852693°。全数据 joint margin min/P01/P05 为
1.800186°/1.941607°/2.066736°，越界和贴边样本均为 0。validation/test 的
half-phase challenge 均通过四 cut 审计。数据集 SHA-256 为：

```text
997260c4b9798a3f492a4ba994e6afcd8fca1a68c6a884ad22253221dfb655dc
```

因此 `formal_dataset_gate_pass=true`、`model_training_authorized=true`，同时
`target_105_achieved=false`。这正是本次 KPI 与当前前沿训练授权解耦的预期语义。

### 9.2 正式 48-config 模型与五 seed 结果

48 个预注册配置中 46 个通过 screen validation gate；锁定配置为：

```text
mlp_beta6_large_poly_medium_relu_a1em06__identity
[512,256,128,64], poly_medium, ReLU, alpha=1e-6
```

五个 seed 在 92.5 mm validation 和 100 mm test 的 integer/half-phase 四组正式
centerline evaluation 全部 5/5 通过。100 mm test 的五 seed 范围为：

| evaluation | EE P95 (mm) | EE max (mm) | beta P95 (deg) | min joint margin (deg) |
|---|---:|---:|---:|---:|
| integer centerline | 1.649803–2.515722 | 2.448309–3.302809 | 0.259335–0.274777 | 1.980315–1.994847 |
| half-phase centerline | 1.631334–2.514905 | 2.514953–3.281050 | 0.259324–0.275467 | 1.980334–1.994865 |

所有 seed 的 raw joint bound violation 和 output-link target clip 均为 0。因此：

```text
formal_model_gate_pass       true
static_inverse_claim_radius  100.0 mm
```

100 mm tube 模型诊断为 1/5 seeds 通过，但它在训练前即标记为
`formal_gate_role=false`。四个失败 seed 只失败 `EE max<=10 mm`，最大误差为
11.5966–15.4900 mm；它们的 EE P95、beta、局部平滑、seam、joint bound 与 margin
均通过。这说明静态 MLP 对中心线很稳定，但 tube 外壳仍存在少数高最大误差点，后续
若目标升级为整 tube 部署，需要单独优化尾部误差，而不是否定本次中心线 claim。

五个 seed 的模型、20 份正式中心线 prediction 和五张 3D 复合图均已保存于：

```text
runs/true_ellipse_standard_domain_training_v7_gate_v2_99pct/
```

轨迹图沿用项目现有 Matplotlib/Agg 风格，每张同时展示 validation/test 的
integer/half-phase 倾斜 3D、主平面投影和逐轴误差。相机采用目标轨迹 SVD 平面法向
再偏转 12°，目视审计中椭圆均清楚可辨，没有退化成直线。

### 9.3 κ 候选 sweep 的径向结论

预注册的 κ=150、200、250、300、400 五个候选全部停在同一个注册径向前沿
102.5 mm，首个失败 checkpoint 均为 105 mm。κ=400 时 105 mm 的四个
`parent_copy` 较早 stage 已降至约 338.24–338.26，但四个 `radial_secant` 仍为
471.37–471.38；106.25/106.5 mm 的 radial-secant candidate graph 又分别只覆盖
343/341 个角层。完整角层 fail-closed 正确阻止这些不完整路径进入联合修正。

因此 κ 放宽到 400 仍不能把注册径向半径推过 102.5 mm。代表性 κ=150 路径随后
完成了 101.25/102.5 mm tube、完整数据和候选模型下游：径向前沿为 102.5 mm，
但连续通过的 tube/data/test 前沿均为 100 mm。候选数据仍为 90,000 行、十个完整
半径，conflict voxel=0，validation/test support 与 half-phase challenge 全部通过；
它只作为 `evidence_only` 产物，不取得正式 claim。候选模型和 policy 的最终数值见
checkpoint；由于四个前沿不一致，候选 policy 必须 fail-closed，本次不修改 legacy
κ=150 policy。

### 9.4 候选下游模型、3D 轨迹与 policy 决定

κ=150 的代表性 102.5 mm 径向前沿完成独立下游后，实际可连续通过的 tube/data/test
前沿仍为 100 mm。候选数据集有 90,000 行、十个完整半径，SHA-256 为
`a68832956048f3c7ea9791c1a09eefea8b3d901c50cdc635421f4a8886563bac`；
conflict voxel=0，最大 voxel beta RMS=0.712889°，joint margin min/P01/P05
为 1.787149°/1.933261°/2.059515°，validation/test support 与两条 half-phase
challenge 均通过。由于它来自未注册的候选 policy，全链固定
`evidence_only=true`、`formal_claims_allowed=false`。

候选数据与正式 Gate-v2 数据的 90,000 个目标 XYZ 逐行完全相同，81,000 行 beta
标签也完全相同；只有 100 mm 的 9,000 行 tube 标签因独立 surface 链发生变化，
beta RMS 差异 P95/max 为 0.121382°/0.813683°。因此训练半径与 92.5 mm
validation 完全相同，完整 48-config screen 如预期复现 46/48 通过，并再次锁定
`mlp_beta6_large_poly_medium_relu_a1em06__identity`。

五个候选 seed 的四组 centerline evaluation 全部 5/5 通过；100 mm 结果为：

| evaluation | EE P95 (mm) | EE max (mm) | beta P95 (deg) | min joint margin (deg) |
|---|---:|---:|---:|---:|
| integer centerline | 1.649803–2.515722 | 2.448309–3.302809 | 0.088070–0.135202 | 1.980315–1.994847 |
| half-phase centerline | 1.631334–2.514905 | 2.514953–3.281050 | 0.088345–0.139516 | 1.980334–1.994865 |
| tube diagnostic | 1.852970–2.555658 | 7.631825–15.490014 | 0.115680–0.137985 | 1.786286–1.797444 |

所有 seed 的 target clip 与 raw joint bound violation 都为 0。候选的 100 mm beta
P95 比正式 Gate-v2 标签上的 0.259–0.275° 更低，说明独立径向分支与内半径训练面
更加一致；但这只改善模型证据，不会替失败的 101.25/102.5 mm tube gate 补证。
`model_evidence_gate_pass=true`，而 `formal_model_gate_pass=false` 是因为候选运行被
预先限定为 evidence-only，不是模型质量失败。100 mm tube diagnostic 仍为 1/5，
且只由少数 `EE max>10 mm` 尾部点触发，继续不参与 centerline formal gate。

五张候选 3D 复合图保存于
`runs/true_ellipse_standard_domain_gate_sweep_v7/03_models/r102p50_k150/04_visualization/seed_trajectories/`。
相机仍采用 SVD 主平面法向加 12°，原始分辨率目视审计中四条椭圆均清楚可辨；
`visualization_integrity_gate_pass=true`。

最终 policy 汇总中，radial、tube、dataset、support、五 seed 模型、prediction 与
provenance 的 current/hash 检查全部通过。但五个 κ 候选在 105 mm 都没有完整
8-job 阳性：κ≤300 的 parent-copy 与 radial-secant 都超过阈值；κ=400 仅四条
parent-copy 较早 stage 通过，四条 radial-secant 仍失败且候选路径未完整物化。
此外，radial=102.5 mm 与 tube/data/test=100 mm 使
`frontier_consistency_gate_pass=false`。因此：

```text
policy_registration_allowed = false   # all five candidates
recommendation.available    = false
registered legacy kappa     = 150
formal fitted radius        = 100.0 mm
target_105_achieved         = false
```
