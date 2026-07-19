# 2026-07-17 True Ellipse Standard-Domain V7 Gate-v2

## Git 与实验边界

- 分支：`codex/true-ellipse-v7-standard-domain-120mm`
- gate-v2 fixed point：`85730c92ba38cf1c0a1e4edeb2ac55b457633dac`
- worktree：`/mnt/ML_projects/quasi_exp/.worktrees/true-ellipse-v7-standard-domain-120mm`
- legacy strict 数值目录：`runs/true_ellipse_standard_domain_v7_gate_v2/`
- 0.99 gate-only replay：`runs/true_ellipse_standard_domain_v7_gate_v2_99pct/`
- 正式训练目录：`runs/true_ellipse_standard_domain_training_v7_gate_v2_99pct/`
- κ sweep：`runs/true_ellipse_standard_domain_gate_sweep_v7/`

本 checkpoint 不追溯改写冻结的 V7 与 V7-D1 结论。`target_105_achieved`、当前前沿训练授权、候选 policy 证据和正式模型 claim 保持分离。

## Gate-v2 契约

1. legacy strict conditioning gate 仍为 `kappa P95 <= 150`；
2. 105 mm 只作为 KPI，不再阻断当前完整 strict frontier 的 tube/data/model 证据链；
3. test/validation 从最高 support-backed 当前前沿动态选择，gap 固定 7.5 mm；
4. smoke/pilot 物理上不能读取 test，只有 formal final evaluation 能打开注册 test；
5. 候选 κ policy 固定扫描 150、200、250、300、400，且不能授予正式 claim；
6. candidate graph 缺少任一预期角层即 fail-closed；
7. optimizer 较早 stage 只有在对应路径实际物化时才能成为候选 policy 证据。

Tube 尝试仍覆盖 radial strict frontier，但正式数据前沿采用预注册的“最高完整通过前缀”：候选 checkpoint 以内的正式半径和 training anchor 必须全部通过；第一个失败点及其外层只保留诊断。因而 102.5 mm 外层失败时可以正式回退到完整的 100 mm 前沿，而不是把整个数据/模型阶段短路。

该回退只授权较小前沿上的数据/模型，不替较大的 κ 候选 policy 补证。候选阈值要进入注册建议，radial candidate frontier、tube frontier、dataset frontier 与 test radius 必须完全一致；否则 `frontier_consistency_gate_pass=false`。

## κ sweep 结果

五个预注册候选的注册径向前沿完全相同：

| κ threshold | registered frontier | first failed checkpoint | 结论 |
|---:|---:|---:|---|
| 150 | 102.5 mm | 105 mm | 不提升 |
| 200 | 102.5 mm | 105 mm | 不提升 |
| 250 | 102.5 mm | 105 mm | 不提升 |
| 300 | 102.5 mm | 105 mm | 不提升 |
| 400 | 102.5 mm | 105 mm | 不提升 |

在 κ=400 时，105 mm 四个 `parent_copy` job 的较早 optimizer stage 为约 338.24–338.26，满足候选 κ；但四个 `radial_secant` job 仍为 471.37–471.38，全部失败。106.25/106.5 mm 的 radial-secant candidate graph 分别只覆盖 343/341 个角层，集中缺失约 82°–98°/81°–99°；完整角层门正确阻止了不完整路径进入 360 点联合修正。因此单纯把 κ 放宽到 400 不会增加半径，下一步若继续扩张应优先处理该相位区间的 radial-secant/branch 几何，而不是继续小幅上调 κ。

## Tube 成功率门修正

旧 V7 在既有 V3/V5 `target_success_ratio >= 0.99` tube gate 之后，又额外要求 `tube_success_ratio == 1.0`，并在 dataset completeness 中再次要求每行成功。该重复门已在读取 101.25/102.5 mm 最终结果之前修正为统一的 `>=0.99`：

- 保留 coverage、P95/max residual、局部 beta RMS、multi-branch、joint margin、四 cut、cut invariance 全部原门；
- gate policy、tube strategy、dataset strategy 升版并进入 fingerprint；
- 旧 `==1.0` 报告不覆盖；
- 新 runner 对 source report、centerline 和 tube artifact 做 SHA-256 绑定，从 Parquet 重算 geometry/margin 后才重放 label gate。

已验证到 100 mm 的真实重放结果：

| radius | success ratio | residual P95/max (mm) | beta local RMS P95 | replay gate |
|---:|---:|---:|---:|---|
| 75–97.5 mm | 1.000000 | 各半径均通过原门 | <1° | pass |
| 100 mm | 0.993111 | 0.053237 / 2.479538 | 0.955817° | pass |

100 mm 同时满足 coverage=1、multi-branch=0、surface/cut/margin/tube gate 全部为真；旧 V7 唯一失败项是新增的 `==1.0` label 条件。

## 完整 tube 前沿与正式数据集

101.25 mm 是 100 mm 之后的必需 training anchor，也是完整前缀的第一个失败点：

| radius | best/final success ratio | final residual P95/max (mm) | final beta local RMS P95 | 结果 |
|---:|---:|---:|---:|---|
| 100 mm | 0.993111 / 0.993111 | 0.053237 / 2.479538 | 0.955817° | pass |
| 101.25 mm | 0.989333 / 0.988667 | 0.091696 / 3.685917 | 1.175044° | fail |
| 102.5 mm | 0.979222 / 0.978667 | 0.140529 / 4.896392 | 1.307521° | fail |

101.25 mm 同时失败 `success ratio >= 0.99`、`residual max <= 3 mm` 和 `beta local RMS P95 <= 1°`；这不是单个频繁 gate 的误杀。因此完整 tube/data 前沿正式回退为 100 mm。为排除更外层存在反常改善的可能，102.5 mm 也完成了四 cut、双向 surface sweep 和三阶段 joint correction：相对初始 success=0.949667、residual P95/max=1.515852/10.748114 mm 已显著改善，但最终仍失败同样三门。其 coverage=1、multi-branch=0、cut invariance、surface 与 joint-margin gate 均通过。

失败定位显示，101.25 mm 的 102 个失败样本中 98 个位于 `delta_n2=-5 mm` 外壳，主要集中在 78°–101°（另有 55°–56°）；102.5 mm 的 192 个失败样本集中在 76°–103°（另有 55°–56°），负 `delta_n2` 外壳仍占主导。两个半径的最坏点都是 84°、`delta_n1=+5 mm, delta_n2=-5 mm`。这把下一轮优化目标收敛到局部 outer-shell 整曲线修正，而不是全局改 gate。

gate-only replay 生成的正式数据为：

- 90,000 行、10 个完整半径、单 family；
- validation=92.5 mm，test=100 mm，训练半径 8 个、训练行 69,120；
- trajectory leakage=0、radius leakage=0、sample ID 全部唯一；
- conflict voxel=0，最大 voxel beta RMS=0.852693°；
- 全数据最小/P01/P05 joint margin=1.800186°/1.941607°/2.066736°，越界和贴边样本均为 0；
- validation/test half-phase challenge 均由四个 cyclic cut 验证通过；
- `formal_dataset_gate_pass=true`、`model_training_authorized=true`、`target_105_achieved=false`。

正式 dataset SHA-256 为 `997260c4b9798a3f492a4ba994e6afcd8fca1a68c6a884ad22253221dfb655dc`。

## 正式模型与 3D 轨迹

正式训练读取上述 SHA 锁定的 90,000 行数据集，输入仍仅为无约束
`x_target_m,y_target_m,z_target_m`，没有把 radius、angle、family 或 split
作为特征。48 个预注册配置中 46 个通过 screen validation gate，最终按注册排序
锁定：

```text
mlp_beta6_large_poly_medium_relu_a1em06__identity
hidden layers = [512, 256, 128, 64]
feature set   = poly_medium
activation    = relu
alpha         = 1e-6
output link   = identity
```

锁定配置随后独立训练五个 seed。下表中的误差范围均为五个 seed 的最小值至最大值：

| evaluation | EE P95 (mm) | EE max (mm) | beta P95 (deg) | prediction min margin (deg) | pass |
|---|---:|---:|---:|---:|---:|
| validation integer, 92.5 mm | 1.190645–2.154362 | 1.836616–2.432956 | 0.032519–0.048064 | 2.015311–2.030400 | 5/5 |
| validation half-phase, 92.5 mm | 1.197571–2.149343 | 1.814104–2.444700 | 0.032016–0.047478 | 2.015274–2.030403 | 5/5 |
| test integer, 100 mm | 1.649803–2.515722 | 2.448309–3.302809 | 0.259335–0.274777 | 1.980315–1.994847 | 5/5 |
| test half-phase, 100 mm | 1.631334–2.514905 | 2.514953–3.281050 | 0.259324–0.275467 | 1.980334–1.994865 | 5/5 |
| validation tube diagnostic, 92.5 mm | 1.236399–2.127865 | 2.998446–3.884314 | 0.057612–0.065407 | 1.825609–1.837235 | 5/5 |
| test tube diagnostic, 100 mm | 1.852970–2.555658 | 7.631825–15.490014 | 0.338230–0.351290 | 1.786286–1.797444 | 1/5 diagnostic |

四组正式 centerline evaluation 全部为 5/5 seeds 通过，超过预注册的至少 4/5
稳定性要求；所有 seed 的 `target_link_clip_count=0`、
`beta_bound_violation_count=0`。因此：

```text
formal_model_gate_pass       = true
static_inverse_claim_radius  = 100.0 mm
target_105_achieved          = false
```

100 mm tube 模型诊断的 4 个失败 seed 只失败 `EE max <= 10 mm`：对应最大值为
11.5966–15.4900 mm；其 EE P95、beta P95、局部/二阶 beta、seam、逐轴偏置、
joint bound 与 margin gate 均通过。该 tube evaluation 在训练前已注册为
`formal_gate_role=false`，所以它揭示的是 tube 外壳的稀疏局部最大误差风险，不会
事后改变 centerline 静态逆模型的正式结论。

五个 seed 均保存独立模型、四组正式 holdout prediction 和一张复合轨迹图。每张图
依次展示 validation/test 的 integer/half-phase 轨迹，并同时给出倾斜 3D、轨迹
主平面投影和逐轴 Cartesian error。3D 相机由每条目标轨迹的 SVD 平面法向自动计算，
再偏转 12°，因此不会把椭圆侧视成直线：

- `runs/true_ellipse_standard_domain_training_v7_gate_v2_99pct/04_visualization/seed_trajectories/seed_20260711.png`
- `runs/true_ellipse_standard_domain_training_v7_gate_v2_99pct/04_visualization/seed_trajectories/seed_20260712.png`
- `runs/true_ellipse_standard_domain_training_v7_gate_v2_99pct/04_visualization/seed_trajectories/seed_20260713.png`
- `runs/true_ellipse_standard_domain_training_v7_gate_v2_99pct/04_visualization/seed_trajectories/seed_20260714.png`
- `runs/true_ellipse_standard_domain_training_v7_gate_v2_99pct/04_visualization/seed_trajectories/seed_20260715.png`

`visualization_integrity_gate_pass=true`；可视化报告绑定正式训练报告 SHA-256
`946f47750e3c112f0bb0db913c356d08f654d619eda7197c3dccdbefb9264c0e`、
20 份 prediction SHA 和 5 份 PNG SHA，且明确标记 `visualization_only=true`，
不修改模型 gate。

最终模型包的独立 SHA-256 审计为：

| seed | model.joblib SHA-256 |
|---:|---|
| 20260711 | `83db60a49fb6e4d5a793e30f361d108e1342c01c52470f263283209959228de8` |
| 20260712 | `cecc081969d7947957b86ff065a1bc4e28e866dd84cc5e17323b4e0990410b0d` |
| 20260713 | `108ff92c45f5210154b94e089cd3ae7257338edb37d530e6b3f0a2f0b11f690e` |
| 20260714 | `15407c7399880cec247c47f43991c6aff5027d1fcf13a46df848eca8d654fbb6` |
| 20260715 | `343a317ad7461ae2290d63d4f13fc1c9a12f81068b9b7c1fc1fbfcbff486b605` |

`visualization_report.json` 本身的 SHA-256 为
`d99b5782032166718b9ed926bf61afb3dcf62bec7988312aabff9117b4a74378`。

正式重跑还把缓存审计从“文件存在”收紧为“路径、预期集合和 SHA-256 全部一致”：
五个模型包、每个 seed 的六份 prediction 均进入训练报告 manifest；任一文件被替换即
拒绝缓存并重新训练。候选 policy 汇总同样逐层复核 radial manifest、选中 tube
完整前缀、dataset/两条 challenge、五 seed 模型与 prediction，并要求
policies→downstream→models 及 radial→tube→dataset 的 task fingerprint 连续绑定。
smoke/pilot 即使带 `evidence_only` 也只能打开物理上不含 test 的 non-formal parquet；
只有 formal final evaluation 可以读取 test。

## 候选 sweep 的完整下游闭合

代表性 κ=150、径向前沿 102.5 mm 的独立下游最终得到：

| 层级 | 前沿/结果 | gate |
|---|---:|---|
| radial candidate | 102.5 mm；105 mm 首次失败 | evidence pass to 102.5 |
| tube complete prefix | 100 mm；101.25/102.5 mm 均失败 | evidence pass to 100 |
| dataset/test | 100 mm；90,000 行、十个半径 | evidence pass |
| model | 100 mm centerline，四组 5/5 | evidence pass |
| policy registration | 102.5 与 100 mm 前沿不一致 | fail-closed |

候选数据集 SHA-256 为
`a68832956048f3c7ea9791c1a09eefea8b3d901c50cdc635421f4a8886563bac`。
它有 conflict voxel=0、最大 voxel beta RMS=0.712889°、joint margin
min/P01/P05=1.787149°/1.933261°/2.059515°，support、split 和两条 challenge
全部通过。正式与候选数据的目标 XYZ 全部相同，81,000/90,000 行 beta 标签也完全
相同；只有 100 mm 的 9,000 行因独立 surface 链变化，beta RMS 差异
P95/max=0.121382°/0.813683°。

候选 48-config screen 再次得到 46/48 通过，并锁定同一配置
`mlp_beta6_large_poly_medium_relu_a1em06__identity`。五 seed 结果为：

| evaluation | EE P95 (mm) | EE max (mm) | beta P95 (deg) | prediction min margin (deg) | pass |
|---|---:|---:|---:|---:|---:|
| validation integer, 92.5 mm | 1.190645–2.154362 | 1.836616–2.432956 | 0.032519–0.048064 | 2.015311–2.030400 | 5/5 |
| validation half-phase, 92.5 mm | 1.197571–2.149343 | 1.814104–2.444700 | 0.032016–0.047478 | 2.015274–2.030403 | 5/5 |
| test integer, 100 mm | 1.649803–2.515722 | 2.448309–3.302809 | 0.088070–0.135202 | 1.980315–1.994847 | 5/5 |
| test half-phase, 100 mm | 1.631334–2.514905 | 2.514953–3.281050 | 0.088345–0.139516 | 1.980334–1.994865 | 5/5 |
| validation tube diagnostic | 1.236399–2.127865 | 2.998446–3.884314 | 0.057612–0.065407 | 1.825609–1.837235 | 5/5 |
| test tube diagnostic | 1.852970–2.555658 | 7.631825–15.490014 | 0.115680–0.137985 | 1.786286–1.797444 | 1/5 diagnostic |

所有 target clip 与 beta bound violation 均为 0。候选 100 mm 标签的 beta P95
明显低于正式标签上的 0.259–0.275°，但模型运行预注册为 evidence-only，所以正确
报告 `model_evidence_gate_pass=true`、`formal_model_gate_pass=false`；后者不是拟合
失败，不能与正式 100 mm claim 混淆。

候选五模型包 SHA-256：

| seed | model.joblib SHA-256 |
|---:|---|
| 20260711 | `da063c29d10bdc50475d0da1d818705c070d2ee44cf4176d1055ae2983287061` |
| 20260712 | `9260382ab45771d7a6c0e5abcfd837218219ffe1658e880a123e34e9f6a9940d` |
| 20260713 | `77540a437df45ed9221ab7e60ecc003f21fc33672eb0e506a1a9f5c4aaff5ba5` |
| 20260714 | `415145c00bc248474aae7a3ef788c311dfb8dbf0c49990629d31fca6be534665` |
| 20260715 | `d298e11d61ba92a0d6f3d33ab05ff473150d3b78cb306d0ca1d4461c4b0b543f` |

五张倾斜 3D 复合图均已生成并目视确认不是直线视角：

- `runs/true_ellipse_standard_domain_gate_sweep_v7/03_models/r102p50_k150/04_visualization/seed_trajectories/seed_20260711.png`
- `runs/true_ellipse_standard_domain_gate_sweep_v7/03_models/r102p50_k150/04_visualization/seed_trajectories/seed_20260712.png`
- `runs/true_ellipse_standard_domain_gate_sweep_v7/03_models/r102p50_k150/04_visualization/seed_trajectories/seed_20260713.png`
- `runs/true_ellipse_standard_domain_gate_sweep_v7/03_models/r102p50_k150/04_visualization/seed_trajectories/seed_20260714.png`
- `runs/true_ellipse_standard_domain_gate_sweep_v7/03_models/r102p50_k150/04_visualization/seed_trajectories/seed_20260715.png`

`visualization_integrity_gate_pass=true`。关键报告 SHA-256：

| report | SHA-256 |
|---|---|
| candidate final training | `13578efc0d1d3c1590844d9f81f47534b6cbfba558ab036276dd7795a92ee832` |
| candidate visualization | `f38e2f785dda94907ba1df6852f2781f727df6ccc18498e39dc41e48fb0abe88` |
| gate sweep summary | `0dc8a2af08ba96802dd21c0843b4c26cf3ee620e5ef58cecabfc69130cdb10ba` |

最终 summary 对五个候选均给出 `policy_registration_allowed=false`、
`recommendation.available=false`。所有 artifact-current、support、五 seed model 与
provenance 检查均为真；失败原因是 105 mm candidate radius 8-job gate 未完整通过，
并且 radial=102.5 mm 与 tube/dataset/test=100 mm 导致
`frontier_consistency_gate_pass=false`。因此 legacy κ=150 保持不变，正式可拟合
半径保持 100 mm，`target_105_achieved=false`。
