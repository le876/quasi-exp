# True Ellipse Radial Bundle V6 实验记录

## 1. 目标与严格结论口径

V6 针对 V5 已经证明“100 mm 逐点可达、但 360 点 canonical branch 无法稳定延拓”的矛盾，改造的是数据生成路径，而不是先扩大模型或放宽验收门槛。固定目标为同一条 family：

- family：`c0273_a100_py210_pz330_s0243`；
- `amp_xy=100 mm`，`amp_z=150 mm`；
- 360 个相位点；
- 5×5 法向 tube，偏移严格为 `[-5,-2.5,0,2.5,5] mm²`；
- 只训练静态 `xyz -> beta6`，半径、角度、family 和 branch 元数据均不得作为模型输入。

严格的 100 mm 模型结论必须按顺序同时满足：

1. V2–V5 输入与 100 mm pointwise 证书审计通过；
2. 同一 fixed family 从 75/80 mm parent 连续延拓到 100 mm；
3. 每个新半径的两种 radial predictor × 四个 cyclic cut 全部通过，且切口不变性与 bitwise exact rerun 通过；
4. 十个正式半径的 360×25 tube 全部通过原 V3/V5 gate；
5. 单 family 90,000 行数据集无 branch conflict，92.5/100 mm 整半径分别作为 validation/test；
6. support 只从训练半径的非中心线样本计算，100 mm strict support 通过；
7. 完整 24 配置只使用 92.5 mm validation 选型，锁定后五个 seed 在未触碰的 100 mm test 上至少 4/5 通过。

上游任何 gate 失败时，模型结果不能产生 strict 阳性。

## 2. V2–V5 实验效果总结

V2–V5 已把失败边界逐层收窄：

| 版本 | 得到的证据 | 仍然存在的限制 |
|---|---|---|
| V2 | 1M full-`beta6` atlas 与 100 mm 可达候选 | 最近邻/逐点可达不保证闭合、平滑且唯一的逆解分支 |
| V3 | 建立 canonical branch、tube 与严格 gate | 角向 continuation 对起点、方向和 branch 选择敏感 |
| V4 | 获得 `81.25 mm strict support-backed` 静态模型 checkpoint | 训练数据的半径覆盖仍受上游 branch materialization 限制 |
| V5 | 固定 family 搜索 6,147 条；87.5 mm 为 5/5 family pointwise 通过，100 mm 为 2/5；所有已物化 tube 均通过 | 最远 robust trajectory 只有 85 mm；`s0243@82.5` forward/reverse P95=`1.020413°`，`v3_selected_s1008@87.5` 为 `1.313585°`；因此正式训练被 gate 阻断 |

其中最关键的反证是：V5 的 `s0243@100 mm` 72 点 pointwise residual P95/最大值只有 `0.001683/0.016823 mm`，但沿角度独立 continuation 仍不能形成通过正反一致性 gate 的 360 点 canonical branch。因此瓶颈不是简单的“机械臂够不到”，也不是“MLP 不够大”，而是逆解标签生成没有利用完整的二维结构 `beta(phase, radius)`。

## 3. 根因判断

V5 在扩半径时主要沿 angle 方向重新追踪分支。这样会丢失前一半径上每个相位点已经确定的 canonical posture：

- 新半径只从少量角向 seed 重新开始，容易在冗余逆解流形上换支；
- forward/reverse 依赖遍历方向，失败项集中在分支一致性，而不是 Cartesian residual、平滑度或条件数；
- 不同半径之间没有显式 radial anchor，导致可达点没有组成稳定的 `phase × radius` 解面；
- tube 的 25 条 offset curve 若逐条独立 IK，也可能在相邻法向位置选择不同冗余姿态，制造监督标签梯度。

V6 的核心假设因此是：应把“生成更多随机点”改成“沿完整 parent path 做半径同伦，并联合约束相位环与法向外层”。

## 4. V6 数据生成方法

### 4.1 全路径 radial predictor

从已审计的 75、80 mm V5 centerline 开始，每次扩半径都保留全部 360 个相位点：

- `parent_copy`：把同 angle 的完整 parent beta 作为新半径初值；
- `radial_secant`：利用相邻两个已通过半径，对每个 angle 独立做 radial secant 外推；
- predictor 始终按 `family_id + angle_idx` 对齐，不允许只复制 angle 0 或重新搜索 center/phase。

### 4.2 周期 joint correction 与多切口审计

每个目标半径同时运行：

- predictor：`parent_copy`、`radial_secant`；
- cyclic cut：`0,90,180,270`；
- anchor schedule：`conservative -> balanced -> loose`，所有 stage 的 `lambda_anchor > 0`；
- 若直接 joint corrector 失败，才启用每角最多 8 个、2 mm residual、0.25° 去重的 candidate graph rescue。

一个半径只有在全部 8 个 predictor/cut 任务通过、任意两条恢复后路径的 beta P95 差异不超过 1°、canonical 路径 bitwise exact rerun 通过时才被接纳。半径 walk 使用 1 mm 基础步长，并预注册 0.5/0.25 mm bounded retry；正式 checkpoint 为 `82.5,85,87.5,90,92.5,95,97.5,100 mm`。

### 4.3 tube 外层 joint fallback

100 mm 初始逐曲线 tube 的唯一失败项是局部一致性：

- success ratio=`0.993556`，通过；
- residual P95/最大值=`0.000561/2.479300 mm`，通过；
- multi-branch ratio=`0`，通过；
- tube10 beta RMS P95=`1.019249°`，超过 1°。

诊断显示在约 70°–110° 相位，`n2=-5` 与 `n2=+5` 两侧虽然 Cartesian 距离不超过 10 mm，独立 IK 却选择了不同冗余姿态。V6 保留初始失败产物，并对两侧共 10 条外层曲线使用“同 n1、内一层 n2=±2.5”的完整 360 点 parent-anchored cyclic joint correction。该 fallback 不删除困难样本、不收窄 tube，也不改变 success、residual 或 beta gate。

## 5. 审计与 100 mm radial 结果

- V2–V5 关键输入：22 个，缺失 0；
- 旧产物在 radial run 前后 hash 不变；
- V5 100 mm pointwise 证书分辨率独立保持 72 点，不伪装成 V6 的 360 点；
- pointwise covered/success=`72/72`，residual P95/最大值=`0.001683/0.016823 mm`；
- V6 目标几何与注册 family 最大差异=`0 mm`；
- fixed family 一次性走到 100 mm，未触发 8,192 Sobol family fallback；
- 24 个径向尝试全部按同一 family 执行，最后通过半径=`100 mm`。

正式 checkpoint 的 canonical `radial_secant/cut=0` 指标如下：

| radius (mm) | parent (mm) | residual P95/max (mm) | delta beta P95/max (deg) | delta2 P95 (deg) | seam RMS (deg) | cut pair P95 max (deg) | sigma3 P05 (m) | kappa P95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 82.5 | 82.0 | 0.000297/0.000336 | 0.035390/0.440289 | 0.001362 | 0.211852 | 0.006184 | 0.360214 | 17.1107 |
| 85.0 | 84.5 | 0.000399/0.000462 | 0.037599/0.278566 | 0.001555 | 0.273846 | 0.006233 | 0.336409 | 18.3827 |
| 87.5 | 87.0 | 0.000533/0.000639 | 0.040042/0.221198 | 0.001781 | 0.047899 | 0.006286 | 0.310624 | 19.9754 |
| 90.0 | 89.5 | 0.000771/0.000943 | 0.042783/0.189914 | 0.002156 | 0.061053 | 0.007330 | 0.282384 | 22.0469 |
| 92.5 | 92.0 | 0.001205/0.001570 | 0.045613/0.169452 | 0.002909 | 0.031372 | 0.009030 | 0.250985 | 24.8882 |
| 95.0 | 94.5 | 0.002143/0.003062 | 0.049197/0.154731 | 0.009803 | 0.034999 | 0.012016 | 0.215322 | 29.1072 |
| 97.5 | 97.0 | 0.004781/0.008114 | 0.053928/0.143596 | 0.010641 | 0.033115 | 0.018369 | 0.173531 | 36.2359 |
| 100.0 | 99.5 | 0.016643/0.040152 | 0.061909/0.134820 | 0.007397 | 0.033791 | 0.030487 | 0.122628 | 51.4401 |

100 mm 的 8 个 predictor/cut job 全部通过，28 个两两比较的最坏 P95 只有 `0.030487°`；exact rerun 的最大 beta bit difference 为 `0`。

## 6. 十半径 tube 结果

每个半径均为 9,000 行，coverage=1.0，multi-branch ratio=0：

| radius (mm) | success ratio | residual P95 (mm) | residual max (mm) | tube10 beta RMS P95 (deg) | 结果 |
|---:|---:|---:|---:|---:|---|
| 75.0 | 1.000000 | 0.000027 | 0.000049 | 0.337281 | pass（V5 hash-bound reuse） |
| 80.0 | 1.000000 | 0.000031 | 0.000062 | 0.363054 | pass（V5 hash-bound reuse） |
| 82.5 | 1.000000 | 0.000033 | 0.000336 | 0.379462 | pass |
| 85.0 | 1.000000 | 0.000035 | 0.000462 | 0.398625 | pass |
| 87.5 | 1.000000 | 0.000038 | 0.000639 | 0.421209 | pass |
| 90.0 | 1.000000 | 0.000045 | 0.000943 | 0.451316 | pass |
| 92.5 | 1.000000 | 0.000064 | 0.001570 | 0.495633 | pass |
| 95.0 | 1.000000 | 0.000104 | 0.003062 | 0.562497 | pass |
| 97.5 | 1.000000 | 0.000203 | 0.094545 | 0.695936 | pass |
| 100.0 | 0.991222 | 0.129934 | 2.483489 | 0.937835 | pass（outer-shell joint fallback） |

100 mm 最终结果仍满足原 gate：success≥0.99、coverage≥0.95、P95≤1.5 mm、max≤3 mm、beta P95≤1°、multi-branch ratio=0。

## 7. 正式数据集

最终 parquet：`runs/true_ellipse_radial_bundle_v6/03_dataset/true_ellipse_radial_bundle_tubes_v6.parquet`。

- 行数：`90,000`；
- trajectories/radii：`10/10`；
- family：`1`；
- 每轨迹：精确 `360×25=9,000` 行；
- sample ID：全部唯一；
- 2 mm voxel：40,831 个 occupied，25,806 个 multi-sample；
- conflict voxels/points：`0/0`；
- 最大 voxel beta RMS：`0.972208° < 3°`；
- validation：完整 92.5 mm，9,000 行；
- test：完整 100 mm，9,000 行；
- train：其余 8 个完整半径；只使用非中心线 tube 行时为 `69,120` 行；
- trajectory/radius leakage：`0/0`；
- 模型输入：仅 `x_target_m,y_target_m,z_target_m`。

training-only support 在全部十个审计半径均通过。100 mm 的 NN P95/最大值为 `1.488390/1.626673 mm`，tube count P10=`871.8`；该结果完全不使用 92.5 或 100 mm 样本，也不使用任何训练半径中心线。

## 8. 模型训练协议与结果

smoke 端到端已通过执行验收：audit、split、2-config screen、1-seed final 与 summary 均 exit 0。30 iterations 的 smoke 模型不用于质量结论。

正式训练按以下不可变协议完成：

- validation/test=`92.5/100 mm`；
- screen：24 个配置、seed=`20260711`、angle stride=2、max_iter=800；
- 配置网格：large/wide × raw/poly_medium/poly_heavy × relu/tanh × alpha `1e-6/1e-4`，包含 V4 baseline；
- final seeds：`20260711..20260715`；
- final 使用全部 69,120 个训练样本，100 mm 只在配置锁定后评估；
- 正式阳性要求 validation 与 test 都是恰好 5 seeds 且至少 4/5 通过。

24 配置 screen 只读取 92.5 mm validation，最终锁定：

```text
config       mlp_beta6_large_poly_heavy_relu_a1em04
layers       512, 256, 128, 64
features     poly_heavy
activation   relu
alpha        1e-4
screen EE    P95/max = 2.671148/3.168446 mm
screen beta  P95 = 0.079875 deg
screen bound violations = 0
```

配置锁定后才首次读取 100 mm test。五个 final seed 的正式结果为：

| seed | validation EE P95/max (mm) | validation beta P95 (deg) | validation bound violations | validation gate | test EE P95/max (mm) | test beta P95 (deg) | test bound violations | test gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260711 | 2.348832/2.882250 | 0.087330 | 56 | fail | 2.739755/3.145635 | 0.157179 | 84 | fail |
| 20260712 | 1.919570/3.841996 | 0.074259 | 3 | fail | 2.246900/3.454550 | 0.214044 | 17 | fail |
| 20260713 | 1.889722/2.535510 | 0.076400 | 20 | fail | 2.045124/3.062135 | 0.179673 | 38 | fail |
| 20260714 | 2.725490/3.221669 | 0.073557 | 0 | pass | 2.836196/3.221054 | 0.156759 | 2 | fail |
| 20260715 | 2.170068/3.386659 | 0.076574 | 59 | fail | 3.340709/4.002521 | 0.153931 | 123 | fail |

因此 validation=`1/5`、test=`0/5`，`formal_model_gate_pass=false`。当前 strict static inverse 半径不能从 V4 的 `81.25 mm` 上调。

### 8.1 失败项的后验定位

五个 seed 在 100 mm 的 Cartesian、逐轴、beta 误差、一/二阶平滑度和 seam 门槛其实全部通过；唯一失败字段是“原始模型输出存在 beta bound violation”。越界只发生在 beta3/beta4：

| seed | validation beta3/beta4 violations | test beta3/beta4 violations | test 最大越界 (deg) |
|---:|---:|---:|---:|
| 20260711 | 46/10 | 67/17 | 0.023117 |
| 20260712 | 0/3 | 0/17 | 0.010171 |
| 20260713 | 0/20 | 12/26 | 0.024439 |
| 20260714 | 0/0 | 0/2 | 0.000650 |
| 20260715 | 12/47 | 51/72 | 0.066509 |

这不只是模型随机性。真值标签本身就紧贴硬边界：

- 100 mm centerline 上 beta3/beta4 的最小真值余量分别只有 `0.009656806°` 和 `5.728769e-9°`；
- 92.5 mm 的对应余量也只有 `0.004228190°` 和 `4.719493e-7°`；
- 十个半径的 tube 都包含数值上贴界的标签，每个半径约 `26.2%–30.6%` 的 tube 行距某个关节边界不超过 `0.05°`。

两个非正式反事实仅用于定位，不改写正式结论：

1. 如果只在 gate 计算中忽略 bound violation，validation/test 的其他原始指标均为 `5/5` 通过；
2. 如果事后将预测 beta 裁到物理 bounds 内并重算 FK，100 mm 也会变成 `5/5` 通过。但这是查看 test 后才做的协议外变换，不能追溯性地当作 V6 strict 阳性。

所以 V6 的真实结论是：数据的空间覆盖和静态回归精度已足够到达 100 mm，但当前生成器选出的冗余逆解分支没有为回归误差留出任何可稳定复现的关节边界余量。

## 9. 当前结论与下一轮优先级

已经成立的结论是：

1. 100 mm 并非被 pointwise reachability 或 single fixed family 的连续 canonical branch 否决；
2. full-parent radial secant + cyclic joint correction 把 robust centerline 上限从 V5 的 85 mm 推到 100 mm；
3. 法向 tube 也能在不放宽 gate 的情况下物化到 100 mm；
4. 单 family、十半径、无冲突、training-only-support-backed 的正式数据集已经生成；
5. 正式模型 validation/test 分别只有 `1/5` 和 `0/5` 通过，故 V6 不产生 100 mm strict static-model 阳性；
6. 每个失败 seed 的唯一失败条件是 beta3/beta4 微小越界，而标签本身几乎没有 joint-limit margin。

下一轮应把“更多数据”改成“可学且有物理余量的标签”，优先级如下：

1. **P0：把 joint-limit margin 变成数据生成的一等指标。** 对每个 centerline/tube 标签记录 `min(beta-lower, upper-beta)`，在模型训练前输出按关节、半径和 angle×offset 的 min/P01/P05 报告。先做 `0.05/0.1/0.25°` 余量可行性扫描，再预注册正式 margin gate；不应在看到新 test 后改阈值。
2. **P0：在候选分支和 joint corrector 中直接优化内点余量。** pointwise/candidate graph 除 residual、周期速度和条件数外，加入 joint-limit barrier；启用并调度现有 `lambda_limit`，对贴界候选施加非线性惩罚。族搜索也不再只排“能到 100 mm”，而应先排“在全相位和 tube 上仍有最差关节余量”。
3. **P0：将 tube 从 25 条独立 IK 升级为 `phase×n1×n2` 联合标注。** V6 只在 100 mm 失败后联合修正了 10 条外层曲线；V7 应在所有半径、所有 shell 上同时约束相位平滑、跨 offset 平滑和 joint margin，避免局部 IK 直接裁在 bounds 上。
4. **P1：余量问题解决后再增密 95–100 mm。** 增加 `98,98.5,99,99.5 mm` 等训练半径，并用 validation-only active error map 按 angle×offset 补点；单纯复制当前贴界标签只会增加同一种噪声，不会解决零越界 gate。
5. **P1：预注册有界输出作为模型侧对照。** 使用 sigmoid/tanh 物理区间参数化，而不是事后对 test 裁剪；它不替代标签余量修复，但能区分“数据分支失败”和“无约束输出失败”。
6. **P2：只有 single-chart 内点化失败才升级 multi-chart。** 如果所有候选 family 在 100 mm 都必须贴界，或强制余量后出现 >3° conflict voxel，再引入多图、gating 或分支分类；不把 branch ID 直接泄漏给当前静态回归器。

V6 的 100 mm test 已经用于上述诊断，下一轮修复不能再把同一份结果称为 untouched test。V7 应在改代码之前锁定新的 100 mm challenge：最少使用同几何的半相位网格 `angle=(k+0.5)°`；更强的验证是再锁定一条未参与 V7 family/margin 选型的第二个 100 mm family。

## 10. 运行环境

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

解释器为 `/mnt/ML_projects/conda_envs/quasi_exp/bin/python3.11`；使用 `/tmp/quasi-exp-py311-packages` 与 `/tmp/quasi-exp-libs` 隔离声明依赖，没有修改 Conda 环境。
