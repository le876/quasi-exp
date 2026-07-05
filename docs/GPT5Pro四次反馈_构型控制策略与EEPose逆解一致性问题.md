# GPT5Pro 四次反馈背景：构型控制策略与 EE pose 逆解一致性问题

日期：2026-07-05

## 1. 这次希望 GPT5Pro 帮助判断的问题

我们当前的最终目标仍然是提高模型拟合性能，尤其是让模型能够稳定学习：

```text
EE pose / xyz -> theta
```

后续再扩展到：

```text
EE pose / xyz -> theta + tension
```

现在新的问题是：当前数据集中，空间中非常接近的末端位姿可能对应多个差异很大的 `theta` 解。也就是说，当前监督学习标签很可能不是一个确定的单值函数：

```text
相近 xyz -> 多个不同 theta 分支
```

这会导致 MLP / LGBM 等单值回归模型被迫在多个分支之间取平均，从而出现：

- theta 拟合误差下降有限；
- FK 回代 EE 误差偏大；
- 圆形/椭圆轨迹跟踪出现固定偏移或形状变形；
- 增加数据量或模型容量后收益很小。

我们希望 GPT5Pro 判断：**应该添加什么类型的构型控制策略，让连续体机器人以一种确定、连续、可学习的方式抵达同一个 EE pose，从而提升 `xyz -> theta` 以及后续 `xyz -> theta/T` 的一致性。**

换句话说，我们现在不是单纯问“换什么模型”，而是问：

```text
如何定义一个 deterministic canonical inverse policy？
```

使得数据集从生成阶段就满足局部单值、连续、可学习。

## 2. 机器人与数据背景

机器人是三段连续体结构，共 30 个圆盘，12 根绳索。

关节/圆盘划分：

- 第 1 关节：第 1-10 盘；
- 第 2 关节：第 11-20 盘；
- 第 3 关节：第 21-30 盘。

绳索终止盘：

- `{1,2,11,12}` 终止于第 10 盘，主要决定第一关节角；
- `{3,4,9,10}` 终止于第 20 盘，主要决定第二关节角；
- `{5,6,7,8}` 终止于第 30 盘，主要决定第三关节角。

同时：

- 第一关节仍会受到其余 8 根绳索的摩擦影响；
- 第二关节仍会受到第 3 关节 4 根绳索的摩擦影响；
- 第三关节由 `{5,6,7,8}` 直接控制。

此前根据论文力学建模，我们已经实现过分段 canonical 张力求解。这个方向解决了同一个 theta 下 PSO 多 seed 张力不稳定的问题，但当 workspace 扩展、theta branch 混入后，模型拟合性能仍然受限。

## 3. 已经验证过的关键现象

### 3.1 论文式 PSO 张力本身存在多解

早期复现论文 Eq.(52)+(53) 约束式目标时，对同一个 theta 做多 seed 复验，发现张力差异可以达到几百牛到上千牛量级。这说明论文描述的 PSO 目标只给出 feasible tension，不自动唯一化 canonical tension。

后续我们通过 segmented/canonical 张力 allocator 解决了同 theta 多 seed 不稳定问题。

### 3.2 分段 canonical 小数据集上张力一致性曾经明显改善

在较规则、较小范围的数据上，分段求解 + canonical 张力分配可以把相邻样本张力一致性改善到约 `50N` 量级。这个阶段说明：

- PSO 随机性可以通过 canonical allocator 大幅降低；
- 张力求解不是完全不可控；
- 但这并不保证 workspace 扩展后的 `xyz -> theta/T` 单值性。

### 3.3 扩展 workspace 后，问题从张力随机性转移到 inverse branch 混叠

后续我们尝试了多种更复杂采样：

- mixed beta；
- distal-preferred；
- active beta-manifold；
- priority grid；
- fixed-layer 单支流形；
- hierarchical beta FK-only 数据集。

反复出现的现象是：

```text
beta-close / same-branch 内部 theta 很连续；
all-workspace 近邻里 theta 跳变很大。
```

这说明主要问题不是 FK 本身，也不是同一 branch 内部不平滑，而是同一个 workspace 局部邻域中混入多个 beta/theta branch。

## 4. 当前主数据集：hierarchical beta FK-only

为了解决 fixed-layer 100k 只像薄曲面、缺少 x 方向厚度的问题，我们生成了新的 hierarchical beta FK-only 数据集。

脚本：

```text
scripts/generate_hierarchical_beta_fk_dataset.py
```

核心原则：

```text
优先第三关节，其次第二关节，最后第一关节。
```

角度网格：

| beta | 范围 | 步长 | 档位 |
|---|---:|---:|---:|
| `beta1,beta2` | `[-5,5] deg` | `2.5 deg` | `5 x 5` |
| `beta3,beta4` | `[-5,5] deg` | `1.25 deg` | `9 x 9` |
| `beta5,beta6` | `[-15,15] deg` | `0.5 deg` | `61 x 61` |

总 FK 候选数：

```text
5 * 5 * 9 * 9 * 61 * 61 = 7,535,025
```

输出数据：

- full FK pool: `data/hierarchical_beta_fk_x1p0_1p2_v1/full_fk_pool.parquet`
- x-filtered pool: `data/hierarchical_beta_fk_x1p0_1p2_v1/x1p0_1p2_pool.parquet`
- balanced 100k: `data/hierarchical_beta_fk_x1p0_1p2_v1/balanced_100k.parquet`
- balanced 500k: `data/hierarchical_beta_fk_x1p0_1p2_v1/balanced_500k.parquet`

注意：这批数据是 FK-only 位姿数据，目前没有 `tension_*_n` 标签。它只用于先判断几何逆解：

```text
xyz -> theta
```

不用于判断 PSO 张力一致性。

## 5. hierarchical beta FK-only 数据的空间覆盖

这批数据确实改善了 workspace 覆盖，尤其是 x 方向厚度。

`x=1.0..1.2m` 过滤结果：

| metric | value |
|---|---:|
| generated rows | `7,535,025` |
| x-filtered rows | `4,587,630` |
| x-filter ratio | `0.6088` |
| x-bin nonempty ratio | `1.0000` |
| x-bin count CV | `0.3323` |
| 20mm yz-cell x-range p95 | `198.74mm` |
| 10mm 3D voxel count | `150,490` |

balanced 子集：

| dataset | rows | x-bin CV | yz-cell x-range p95 | 10mm voxels | NN p95 |
|---|---:|---:|---:|---:|---:|
| balanced 100k | `100,000` | `0.0000` | `190.65mm` | `69,870` | `10.78mm` |
| balanced 500k | `500,000` | `0.1599` | `197.18mm` | `143,978` | `6.13mm` |

这说明新数据不是原来那种薄曲面，空间覆盖和稠密度都显著改善。

但问题是：**空间覆盖变好了，逆解一致性并没有自动变好。**

## 6. 当前模型训练结果

### 6.1 100k MLP

输出目录：

```text
runs/baselines_hierarchical_beta_fk_x1p0_1p2_100k_theta_mlp_v1
```

| split | theta MAE deg | theta p95 deg | EE p95 mm | fit s |
|---|---:|---:|---:|---:|
| iid | `2.405` | `4.722` | `82.929` | `103.4` |
| x_slab | `1.708` | `2.884` | `42.975` | `68.9` |
| radius | `1.615` | `2.672` | `39.160` | `57.0` |

### 6.2 100k LGBM, 50 trees

输出目录：

```text
runs/baselines_hierarchical_beta_fk_x1p0_1p2_100k_theta_lgbm_v1
```

| split | theta MAE deg | theta p95 deg | EE p95 mm | fit s |
|---|---:|---:|---:|---:|
| iid | `2.446` | `4.655` | `90.881` | `2.43` |
| x_slab | `1.994` | `3.293` | `35.166` | `2.22` |
| radius | `1.741` | `2.689` | `46.502` | `2.27` |

### 6.3 500k LGBM, 50 trees

输出目录：

```text
runs/baselines_hierarchical_beta_fk_x1p0_1p2_500k_theta_lgbm_v1
```

| split | theta MAE deg | theta p95 deg | EE p95 mm | fit s |
|---|---:|---:|---:|---:|
| iid | `2.488` | `4.808` | `91.949` | `7.73` |
| x_slab | `2.268` | `3.876` | `34.935` | `7.14` |
| radius | `1.822` | `2.750` | `43.807` | `7.59` |

### 6.4 500k LGBM, 300 trees

输出目录：

```text
runs/baselines_hierarchical_beta_fk_x1p0_1p2_500k_theta_lgbm300_v1
```

| split | theta MAE deg | theta p95 deg | EE p95 mm | fit s |
|---|---:|---:|---:|---:|
| iid | `2.453` | `4.809` | `89.899` | `24.72` |
| x_slab | `2.257` | `3.903` | `49.151` | `23.24` |
| radius | `1.802` | `2.800` | `65.101` | `24.25` |

结论：

1. 从 100k 增加到 500k，没有明显改善 `xyz -> theta` 拟合。
2. LGBM 从 50 trees 增加到 300 trees，也没有稳定改善。
3. 这不像单纯数据量不足或模型容量不足，更像标签映射本身存在多解混入。

## 7. 轨迹实验现象

我们尝试搜索固定比例椭圆轨迹：

```text
X = center_x + a * sin(t)
Y = center_y + a * sin(t)
Z = center_z + 1.5a * cos(t)
```

对应用户设想的 `200:200:300` 比例，只允许整体缩放。

结果：

- 100k MLP/LGBM：没有找到合格展示椭圆；
- 500k LGBM：workspace 最近邻支持更好，但模型轨迹没有改善；
- 主要失败模式是固定 x 轴偏移；
- 即使候选椭圆的 `nn_dist_p95_mm` 可以到 `3.77mm`，模型 FK 回代仍有约 `33-43mm` 的轴向 p95 误差。

这个现象进一步提示：

```text
目标点附近有数据，不代表 xyz -> theta 标签在该区域是单值且可学习的。
```

## 8. 最新逆解一致性诊断

为了验证“空间相邻 EE pose 多解”假设，我们按之前的方法分析了当前数据集的局部逆解一致性。

方法：

1. 在笛卡尔空间 `x_m,y_m,z_m` 上寻找相邻 EE pose；
2. 对每对相邻点计算 6 个独立关节角维度的 `theta RMS` 差值；
3. 统计 5mm、10mm、20mm 邻域内的 theta 差值分布；
4. 对 100k 数据额外拆分：
   - `beta-close`: normalized beta RMS 距离 `<= 0.15`
   - `beta-far`: normalized beta RMS 距离 `> 0.15`

这里的 6 个独立 theta 维度为：

```text
theta_1, theta_2, theta_11, theta_12, theta_21, theta_22
```

输出目录：

```text
runs/diagnostics/hierarchical_beta_fk_x1p0_1p2_inverse_consistency_100k_v1/
runs/diagnostics/hierarchical_beta_fk_x1p0_1p2_inverse_consistency_500k_v1/
```

### 8.1 100k 全局近邻结果

| 邻域 | pairs | theta RMS mean | theta RMS p90 | theta RMS p95 | RMS > 5deg | RMS > 10deg |
|---|---:|---:|---:|---:|---:|---:|
| `<=10mm` | `367,256` | `4.106 deg` | `7.893 deg` | `9.666 deg` | `28.41%` | `4.27%` |
| `<=20mm` | `2,842,813` | `4.253 deg` | `8.080 deg` | `9.773 deg` | `29.93%` | `4.52%` |

最近邻距离：

| metric | value |
|---|---:|
| nearest dxyz mean | `6.168mm` |
| nearest dxyz p50 | `5.983mm` |
| nearest dxyz p90 | `9.696mm` |
| nearest dxyz p95 | `10.748mm` |

这说明 100k 数据已经有较多 10mm 邻居，但这些邻居对应的 theta 并不连续。10mm 内 theta p95 接近 `9.7 deg`，明显超过我们希望单值逆解数据达到的 `1-3 deg` 范围。

### 8.2 beta-close / beta-far 拆分

| 邻域 | group | pairs | theta RMS p50 | theta RMS p90 | theta RMS p95 |
|---|---|---:|---:|---:|---:|
| `<=5mm` | all | `25,378` | `3.223 deg` | `7.600 deg` | `9.360 deg` |
| `<=5mm` | beta-close | `4,516` | `1.517 deg` | `1.517 deg` | `1.517 deg` |
| `<=5mm` | beta-far | `20,862` | `3.705 deg` | `8.171 deg` | `9.721 deg` |
| `<=10mm` | all | `183,628` | `3.393 deg` | `7.893 deg` | `9.666 deg` |
| `<=10mm` | beta-close | `30,389` | `1.327 deg` | `1.517 deg` | `1.711 deg` |
| `<=10mm` | beta-far | `153,239` | `3.982 deg` | `8.414 deg` | `10.060 deg` |
| `<=20mm` | all | `1,427,498` | `3.549 deg` | `8.073 deg` | `9.760 deg` |
| `<=20mm` | beta-close | `196,463` | `1.342 deg` | `1.711 deg` | `1.723 deg` |
| `<=20mm` | beta-far | `1,231,035` | `3.987 deg` | `8.464 deg` | `10.091 deg` |

关键比例：

| 邻域 | beta-far pair ratio |
|---|---:|
| `<=5mm` | `82.20%` |
| `<=10mm` | `83.45%` |
| `<=20mm` | `86.24%` |

最关键结论：

```text
beta-close 内部 theta 是连续的；
beta-far 近邻造成了 all-workspace theta p95 接近 10 deg。
```

这说明问题主要不是同一构型分支内部不平滑，而是多个逆解分支在同一个局部 workspace 邻域内混合。

### 8.3 500k 复核

| 邻域 | pairs | theta RMS mean | theta RMS p90 | theta RMS p95 | RMS > 5deg | RMS > 10deg |
|---|---:|---:|---:|---:|---:|---:|
| `<=10mm` | `7,811,368` | `4.335 deg` | `8.513 deg` | `10.259 deg` | `31.46%` | `5.51%` |
| `<=20mm` | `15,484,983` | `4.457 deg` | `8.917 deg` | `10.718 deg` | `32.42%` | `6.77%` |

最近邻距离：

| metric | value |
|---|---:|
| nearest dxyz mean | `3.474mm` |
| nearest dxyz p50 | `3.373mm` |
| nearest dxyz p90 | `5.475mm` |
| nearest dxyz p95 | `6.130mm` |

500k 数据更密，但 10mm theta p95 没有下降，反而从 100k 的 `9.666 deg` 到 `10.259 deg`。这说明：

```text
扩大数据量本身没有消除局部多解混入；
更多数据只是更密地采到了多个分支。
```

## 9. 与 fixed-layer 单支流形结果的对比

此前 fixed-layer 单支流形实验提供了一个重要对照。

固定层定义：

```text
beta = [s1*beta5, s1*beta6, s2*beta5, s2*beta6, beta5, beta6]
```

也就是固定 `(s1,s2)`，主要移动第三关节，第一、二关节只跟随第三关节按比例变化。

在 2k pilot 中，7 个 fixed-layer 全部通过初筛。其中最好的 `s1_0125_s2_0250` 达到：

- `all10 theta p95 = 0.099 deg`
- `all10 tension p95 = 36.317 N`
- `multi_branch_ball_ratio = 0.000`
- `xyz-NN tension MAE = 11.821 N`
- `beta-NN tension MAE = 11.890 N`

graph-anchored relabel 后，最优配置：

```text
k=32, w_anchor=40, anchor_stat=huber_mean
```

把 `all10 tension p95` 从 `36.317N` 降到 `13.012N`。

这个结果说明：

```text
只要构型分支被固定，theta 和 tension 都可以非常连续；
问题出现在为了扩展 workspace 而混入多个构型分支时。
```

但是 fixed-layer 单支流形的缺点是：

- 单个 fixed-layer 的 workspace 覆盖有限；
- 多个 fixed-layer 混合后又可能重新引入分支混叠；
- 如果每个 fixed-layer 单独训练，可能需要 branch-aware/expert 模型。

## 10. 当前判断

我们现在认为当前瓶颈是：

```text
缺少一个全局或局部连续的构型控制策略。
```

更具体地说，当前 hierarchical beta FK-only 数据集虽然满足“第三关节步长最密、第二关节次之、第一关节最粗”的采样原则，但它并没有定义：

```text
同一个 EE pose 或同一个局部 workspace 邻域中，应该选择哪一个 theta branch。
```

所以它仍然是多分支混合数据。

当前训练表现和逆解一致性诊断共同指向：

1. 数据空间覆盖已经改善；
2. 模型容量和数据量不是当前第一瓶颈；
3. 单值 `xyz -> theta` 模型不能直接学习 mixed inverse branches；
4. 必须先定义 canonical branch policy，或者显式训练 branch-aware 模型。

## 11. 我们希望 GPT5Pro 重点回答的问题

### 11.1 应该如何定义 canonical 构型控制策略？

我们直觉上希望满足：

```text
优先移动第三关节；
其次移动第二关节；
最后移动第一关节。
```

但这需要从一个口头原则变成数学策略。希望 GPT5Pro 判断以下形式是否合理：

1. 对每个 target xyz，在可达解集合中最小化：

```text
J = w_1 ||beta_1,beta_2||^2
  + w_2 ||beta_3,beta_4||^2
  - w_3 ||beta_5,beta_6||^2
  + w_s smoothness_to_neighbors
  + w_c conditioning_or_jacobian_penalty
```

其中 `w_1 > w_2`，鼓励第一关节最小、第二关节次小、第三关节承担主要运动。

2. 或者固定比例流形：

```text
beta1,beta2 = s1 * beta5,beta6
beta3,beta4 = s2 * beta5,beta6
```

但允许 `s1,s2` 在 workspace 中缓慢连续变化，而不是离散混合多个 layer。

3. 或者构造一个连续层场：

```text
s1 = f1(x,y,z)
s2 = f2(x,y,z)
```

使得局部 workspace 邻域内 layer 不跳变。

希望 GPT5Pro 判断哪种策略更符合连续体机器人控制和监督学习数据生成。

### 11.2 是否应该主动从 target xyz 反解，而不是从 beta 网格 FK 后筛选？

当前 hierarchical beta 数据是：

```text
beta grid -> FK -> xyz -> balanced sampling
```

这种方式覆盖空间很好，但不会保证某个 xyz 附近只有一个 canonical beta/theta。

另一种方式是：

```text
target xyz grid / trajectory / workspace samples
-> inverse solver 生成多个候选 theta
-> canonical policy 选一个
-> graph smoothing 保证邻域连续
```

希望 GPT5Pro 判断：为了得到可训练的 `xyz -> theta` 数据，是否应该转向 active inverse generation。

### 11.3 canonical policy 是否应该加入局部连续性项？

仅对单个 target xyz 定义“第三关节优先”可能仍然不够，因为逐点最优可能在相邻 xyz 之间跳 branch。

是否应该在 kNN graph 上做全局或局部优化：

```text
min sum_i J_single(i, candidate_i)
  + lambda * sum_(i,j) ||theta_i - theta_j||^2
```

也就是把 branch selection 明确变成图优化问题。

希望 GPT5Pro 判断：

- 图优化是否必要；
- kNN 半径/邻域大小如何选；
- 是否需要 continuation/path-following，从一个初始 pose 沿路径逐步展开。

### 11.4 是否应该加入机器人控制上的路径历史/上一时刻 theta？

如果物理机器人实际控制时不会从任意 pose 瞬移到目标，而是沿连续路径运动，那么 inverse map 也许天然不是：

```text
xyz -> theta
```

而是：

```text
(previous theta, target xyz) -> next theta
```

希望 GPT5Pro 判断：如果目标是轨迹跟踪和真实控制，是否应将数据集和模型改成带状态/路径历史的形式。

这可能从根本上避免“同一个 xyz 有多个 theta 解”的问题，因为上一时刻 theta 决定了当前应该留在哪个 branch。

### 11.5 如果保持单值 `xyz -> theta`，验收指标应该怎么设？

我们目前使用的候选 gate：

- all-workspace 10mm theta p95 `<= 2-3 deg`
- beta-close 10mm theta p95 `<= 1.5 deg`
- beta-far pair ratio 显著下降
- multi_branch_ball_ratio 接近 0
- NN p95 保持在 `5-8mm` 以内
- 后续加入张力后，10mm tension p95 应低于 `50-70N`

希望 GPT5Pro 判断这些门槛是否合理，是否应该补充：

- Jacobian condition number；
- local injectivity；
- workspace fold detection；
- branch margin；
- FK residual refinement gate；
- path-following success rate。

## 12. 当前可选路线

### 路线 A：固定层 / 连续层场

优点：

- 已经证明 fixed-layer 单支流形可以让 theta 连续到 `0.1 deg` 量级；
- 张力 relabel 后也能达到很强一致性；
- 工程实现相对可控。

缺点：

- 单个 fixed-layer 覆盖有限；
- 多 fixed-layer 混合需要 branch-aware 模型；
- 如何定义连续层场还不明确。

### 路线 B：active inverse generation + canonical policy

优点：

- 直接从目标 workspace 出发；
- 可以对每个 target 生成多个候选解后按 policy 选一个；
- 更接近“同一个 EE pose 选择一个 canonical 解”的目标。

缺点：

- 反解速度和稳定性可能较差；
- 需要定义候选生成、多 seed、多初值、失败重试；
- 如果没有图平滑，逐点最优仍可能跳 branch。

### 路线 C：branch-aware / beta-first 模型

优点：

- 不强迫全局 `xyz -> theta` 单值；
- 可以保留更大的 workspace 覆盖；
- 与当前证据一致，因为 beta-close 内部已经很连续。

缺点：

- 原论文形式是直接 MLP 输出 theta/T，branch-aware 会偏离论文表述；
- 需要设计 branch 标签或 beta latent；
- 推理时仍需决定选择哪个 branch。

### 路线 D：stateful inverse model

模型输入从：

```text
xyz
```

改成：

```text
previous theta + target xyz
```

或者：

```text
current state + delta xyz
```

优点：

- 更符合真实连续控制；
- 多解问题由上一时刻状态消解；
- 适合轨迹跟踪。

缺点：

- 与原论文静态 MLP 数据集不同；
- 需要生成轨迹数据而不是独立散点；
- 评价指标也需要从点预测变成轨迹跟踪。

## 13. 希望 GPT5Pro 给出的实验计划

希望 GPT5Pro 不只是给概念建议，而是给出一个可执行的实验方案，包括：

1. 推荐的 canonical 构型控制目标函数；
2. 推荐的数据生成方式，是 FK grid filtering、active inverse generation、fixed-layer field，还是混合；
3. pilot 数据规模，例如 2k/5k/20k；
4. 必须通过的 inverse consistency gate；
5. 是否需要 graph smoothing 或 continuation；
6. 如果使用 branch-aware 模型，branch 标签如何定义；
7. 与原论文直接 `xyz -> theta/T` MLP 方案如何对齐或解释；
8. 失败时如何判别原因：
   - workspace 覆盖不足；
   - branch 混叠；
   - 模型容量不足；
   - 张力 allocator 不连续；
   - FK 几何本身局部不可逆。

## 14. 我们当前倾向

基于目前证据，我们的倾向是：

1. 不再单纯扩大 hierarchical beta FK-only mixed 数据。
2. 先定义 canonical 构型控制策略，让 `xyz -> theta` 在局部成为单值。
3. 优先测试两条路线：
   - fixed-layer 的连续层场版本；
   - active inverse generation + graph-smoothed canonical branch selection。
4. 只有当 all-workspace 10mm theta p95 降到 `2-3 deg` 以内，再进行 MLP/LGBM/TF MLP 对比。
5. 张力 PSO/relabel 放在第二阶段；否则张力标签会被 theta 多解问题继续污染。

## 15. 关键文件索引

数据生成与空间覆盖：

- `docs/HierarchicalBetaFK位姿数据集实验记录.md`
- `scripts/generate_hierarchical_beta_fk_dataset.py`
- `data/hierarchical_beta_fk_x1p0_1p2_v1/balanced_100k.parquet`
- `data/hierarchical_beta_fk_x1p0_1p2_v1/balanced_500k.parquet`
- `runs/diagnostics/hierarchical_beta_fk_x1p0_1p2_v1/README.md`

模型与轨迹结果：

- `docs/ThetaOnlyEllipse_MLP_LGBM_HierarchicalBetaFK实验记录.md`
- `scripts/baselines/run_theta_fk_baseline.py`
- `scripts/analysis/run_theta_only_ellipse_trajectory_benchmark.py`
- `runs/baselines_hierarchical_beta_fk_x1p0_1p2_100k_theta_mlp_v1`
- `runs/baselines_hierarchical_beta_fk_x1p0_1p2_100k_theta_lgbm_v1`
- `runs/baselines_hierarchical_beta_fk_x1p0_1p2_500k_theta_lgbm_v1`
- `runs/baselines_hierarchical_beta_fk_x1p0_1p2_500k_theta_lgbm300_v1`

逆解一致性诊断：

- `runs/diagnostics/hierarchical_beta_fk_x1p0_1p2_inverse_consistency_100k_v1/README.md`
- `runs/diagnostics/hierarchical_beta_fk_x1p0_1p2_inverse_consistency_100k_v1/local_continuity.json`
- `runs/diagnostics/hierarchical_beta_fk_x1p0_1p2_inverse_consistency_100k_v1/branch_aware_continuity.json`
- `runs/diagnostics/hierarchical_beta_fk_x1p0_1p2_inverse_consistency_500k_v1/local_continuity_k32.json`

fixed-layer 单支流形对照：

- `docs/Phase5固定层单支流形实验结果.md`
- `scripts/generate_fixed_layer_manifold_dataset.py`
- `scripts/analysis/relabel_tension_graph_canonical.py`
- `runs/diagnostics/fixed_layer_2k_sweep_v1/summary.md`
- `runs/diagnostics/tension_relabel_fixed_layer_ablation_v1/summary.md`

此前 GPT5Pro 反馈与阶段记录：

- `docs/GPT5Pro二次反馈_模型拟合性能瓶颈与Phase1-3实验结果.md`
- `docs/GPT5pro 三次回答.md`

---
实验计划
可以。这里要把两个概念分开：

**(u) 是你人为设计的“canonical 构型流形坐标”，不是机器人已有变量，也不是最终 MLP 的输入。** 最终 MLP 仍然学 (xyz\to\theta)。(u) 只在数据生成阶段使用，用来规定“我只允许机器人沿哪一族连续姿态去覆盖 workspace”。

你之前 fixed-layer 已经证明：固定

$$
\beta=[s_1\beta_5,s_1\beta_6,s_2\beta_5,s_2\beta_6,\beta_5,\beta_6]
$$

时，(\theta) 可以极其连续，最优 fixed-layer 的 all10 theta p95 达到 (0.099^\circ)，但单个 fixed-layer 覆盖有限；混合多个 layer 又会重新引入 branch 混叠。 所以我的路线一就是把 fixed-layer 从“离散固定层”改成“连续层场”。

---

# 实验计划
可以。这里要把两个概念分开：

**(u) 是你人为设计的“canonical 构型流形坐标”，不是机器人已有变量，也不是最终 MLP 的输入。** 最终 MLP 仍然学 (xyz\to\theta)。(u) 只在数据生成阶段使用，用来规定“我只允许机器人沿哪一族连续姿态去覆盖 workspace”。

你之前 fixed-layer 已经证明：固定

$$
\beta=[s_1\beta_5,s_1\beta_6,s_2\beta_5,s_2\beta_6,\beta_5,\beta_6]
$$

时，(\theta) 可以极其连续，最优 fixed-layer 的 all10 theta p95 达到 (0.099^\circ)，但单个 fixed-layer 覆盖有限；混合多个 layer 又会重新引入 branch 混叠。 所以我的路线一就是把 fixed-layer 从“离散固定层”改成“连续层场”。

---

## 1. (u) 具体怎么定义

最推荐第一版定义为：

$$
u=(a,b,\eta)
$$

其中：

* (a)：第三段在第一个弯曲平面内的 bending command，也就是 (\beta_5)；
* (b)：第三段在第二个弯曲平面内的 bending command，也就是 (\beta_6)；
* (\eta)：连续 layer / shape-distribution 变量，用来控制第一段、第二段跟随第三段的比例。

注意，(\eta) 不应该让 (s_1,s_2) 变成两个独立自由变量。否则你会变成：

$$
(a,b,s_1,s_2)\in\mathbb{R}^4
$$

再通过 FK 映射到 (xyz\in\mathbb{R}^3)，这又会产生冗余和多解。所以 (s_1,s_2) 应该被限制在一条一维连续曲线上：

$$
(s_1,s_2)=\gamma(\eta)
$$

最简单的形式是线性层场：

$$
s_1(\eta)=s_{1,0}+\Delta s_1\eta
$$

$$
s_2(\eta)=s_{2,0}+\Delta s_2\eta
$$

其中：

$$
\eta\in[-1,1]
$$

然后定义 canonical manifold：

$$
g(a,b,\eta)=
\begin{bmatrix}
s_1(\eta)a\
s_1(\eta)b\
s_2(\eta)a\
s_2(\eta)b\
a\
b
\end{bmatrix}
=============

\begin{bmatrix}
\beta_1\
\beta_2\
\beta_3\
\beta_4\
\beta_5\
\beta_6
\end{bmatrix}
$$

这就是你要生成数据的 3 维构型流形：

$$
u\in\mathbb{R}^3
\overset{g}{\longrightarrow}
\beta\in\mathbb{R}^6
\overset{FK}{\longrightarrow}
x\in\mathbb{R}^3
$$

最终数据集保存的是：

$$
x=FK(g(u)),\quad \theta=\beta
$$

或者按你代码里的命名保存为：

$$
(\theta_1,\theta_2,\theta_{11},\theta_{12},\theta_{21},\theta_{22})
===================================================================

(\beta_1,\beta_2,\beta_3,\beta_4,\beta_5,\beta_6)
$$

---

## 2. 第一版参数怎么取

从你之前最好的 fixed-layer 开始：

$$
s_{1,0}=0.125,\quad s_{2,0}=0.250
$$

因为这个中心点已经被实验证明是连续性非常好的 branch。

然后给 (\eta) 一个小范围扰动。建议先测试 4 条 layer path：

### Path A：同步增加 proximal participation

$$
s_1(\eta)=0.125+0.075\eta
$$

$$
s_2(\eta)=0.250+0.100\eta
$$

也就是：

$$
s_1\in[0.050,0.200],\quad s_2\in[0.150,0.350]
$$

### Path B：第一段增加，第二段减少

$$
s_1(\eta)=0.125+0.075\eta
$$

$$
s_2(\eta)=0.250-0.100\eta
$$

这条很重要，因为它不是单纯增加整体弯曲，而是在第一段和第二段之间重新分配 bending。它更可能带来独立的 (x) 方向厚度。

### Path C：第一段减少，第二段增加

$$
s_1(\eta)=0.125-0.075\eta
$$

$$
s_2(\eta)=0.250+0.100\eta
$$

### Path D：更宽范围

$$
s_1(\eta)=0.150+0.150\eta
$$

$$
s_2(\eta)=0.300+0.200\eta
$$

再把超出安全范围的样本裁掉，例如要求：

$$
0\le s_1\le0.35
$$

$$
0.05\le s_2\le0.60
$$

第一轮不要追求最优。目标是看哪条 layer path 能在 (x=1.0\sim1.2)m 目标 slab 内形成足够厚的 3D 覆盖，并且不重新引入 branch 混叠。

---

## 3. 为什么 (u=(a,b,\eta)) 比直接采 (\beta_1\ldots\beta_6) 好

你当前 hierarchical beta 是完整 6D 网格的一种稀疏版本：

$$
\beta\ \text{grid}\rightarrow FK\rightarrow xyz
$$

它覆盖确实变好了，balanced 500k 的 NN p95 到 (6.13)mm，但逆解一致性没有改善，10mm 内 all-workspace theta p95 反而到了 (10.259^\circ)。 

原因是 6D (\beta) 到 3D (xyz) 本来就是降维投影。你从 6D 体积里采样，必然会把多个构型 branch 压到同一个局部 (xyz) 邻域里。

而 (u=(a,b,\eta)) 的做法是先人为规定一个 3D 子流形：

$$
\mathcal{M}={g(a,b,\eta)\mid a,b,\eta}
$$

再看这个 3D 子流形通过 FK 后能不能稳定覆盖目标 workspace。这样维度匹配：

$$
u\in\mathbb{R}^3\rightarrow x\in\mathbb{R}^3
$$

如果这个映射局部可逆，就可以得到单值数据：

$$
x\rightarrow u\rightarrow\beta
$$

---

## 4. (\operatorname{rank}\left(\frac{\partial(f\circ g)}{\partial(a,b,\eta)}\right)=3) 是什么意思

令：

$$
h(u)=f(g(u))
$$

其中 (f) 是 FK：

$$
f:\beta\mapsto x
$$

那么：

$$
h:(a,b,\eta)\mapsto(x,y,z)
$$

你要检查的是：

$$
J_h(u)=\frac{\partial h}{\partial u}
====================================

\frac{\partial(f\circ g)}{\partial(a,b,\eta)}
\in\mathbb{R}^{3\times3}
$$

要求：

$$
\operatorname{rank}(J_h)=3
$$

直观含义是：在当前点附近，(a,b,\eta) 这三个方向分别能造成 3 个线性独立的末端位移方向。

更具体地说：

$$
\frac{\partial h}{\partial a}
$$

主要对应第三段在一个方向弯曲导致的末端位移；

$$
\frac{\partial h}{\partial b}
$$

主要对应第三段在另一个正交方向弯曲导致的末端位移；

$$
\frac{\partial h}{\partial \eta}
$$

对应在固定 distal bending ((a,b)) 的情况下，改变第一段、第二段跟随比例，改变整条机器人弯曲分布，从而改变 reach / x-thickness / 姿态分布。

如果这三个方向线性独立，则 (h) 在局部可逆。也就是说，在这个局部区域内，可以近似认为：

$$
x\leftrightarrow u\leftrightarrow\beta
$$

这就是你想要的局部单值性。

---

## 5. 这个 rank 要怎么“实现”

它不是靠训练实现的，而是靠三步实现：

第一，设计 (g(u)) 时让三个变量有不同物理作用；第二，用数值雅可比检查；第三，拒绝 rank 不好的样本或调 layer path。

### 第一步：让 (\eta) 真的产生独立形变

如果 (\eta) 只是让 (s_1,s_2) 同时按相同比例增加，它可能只是在模仿 (a,b) 变大，导致第三列不独立。更好的做法是让 (\eta) 控制 curvature redistribution，例如：

$$
s_1(\eta)=0.125+0.075\eta
$$

$$
s_2(\eta)=0.250-0.100\eta
$$

这样 (\eta) 增大时，第一段参与更多，第二段参与更少；(\eta) 减小时，第二段参与更多，第一段参与更少。这更像“弯曲位置前后移动”，而不是简单“弯曲幅值变大”。

这类 (\eta) 才更可能产生独立的 (x) 方向厚度。

---

### 第二步：用数值差分计算 (J_h)

建议不要用解析雅可比，直接数值差分就够了。

先把变量归一化：

$$
\bar{a}=\frac{a}{A},\quad \bar{b}=\frac{b}{A},\quad \bar{\eta}=\eta
$$

其中：

$$
A=15^\circ
$$

内部计算时用弧度：

$$
A=\frac{15\pi}{180}
$$

于是：

$$
\bar{u}=(\bar{a},\bar{b},\bar{\eta})\in[-1,1]^3
$$

然后对归一化变量做中心差分：

$$
J_h[:,j]
========

\frac{h(\bar{u}+\delta e_j)-h(\bar{u}-\delta e_j)}{2\delta}
$$

建议：

$$
\delta=10^{-3}\sim10^{-2}
$$

因为 (\bar{u}) 都是无量纲变量，所以 (J_h) 的三个列向量单位一致，都是“每单位归一化变量引起的末端位移”，单位为 m。

伪代码类似这样：

```python
import numpy as np

A = np.deg2rad(15.0)

def beta_from_u_bar(u_bar, s10=0.125, s20=0.250, ds1=0.075, ds2=-0.100):
    a_bar, b_bar, eta = u_bar
    a = A * a_bar
    b = A * b_bar

    s1 = s10 + ds1 * eta
    s2 = s20 + ds2 * eta

    beta = np.array([
        s1 * a,
        s1 * b,
        s2 * a,
        s2 * b,
        a,
        b,
    ])
    return beta

def h(u_bar):
    beta = beta_from_u_bar(u_bar)
    xyz = fk_xyz_from_beta(beta)
    return xyz

def numerical_jacobian(u_bar, delta=1e-3):
    J = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1.0
        xp = h(u_bar + delta * e)
        xm = h(u_bar - delta * e)
        J[:, j] = (xp - xm) / (2.0 * delta)
    return J

def jacobian_gate(u_bar):
    J = numerical_jacobian(u_bar)
    s = np.linalg.svd(J, compute_uv=False)
    sigma_max = s[0]
    sigma_min = s[-1]
    cond = sigma_max / max(sigma_min, 1e-12)

    pass_gate = (sigma_min > 0.002) and (cond < 50.0)
    return pass_gate, sigma_min, cond, s
```

这里：

$$
\sigma_{\min}>0.002
$$

表示最弱方向改变一个归一化单位时，至少能带来约 2mm 级别的末端变化。这个阈值只是 pilot 初值，可以根据实际分布调整。

---

### 第三步：用 SVD 判断 rank

不要在浮点计算里判断“rank 是否等于 3”。应该判断奇异值：

$$
\sigma_1\ge\sigma_2\ge\sigma_3
$$

实际 gate 用：

$$
\sigma_3>\tau_\sigma
$$

并且：

$$
\kappa(J_h)=\frac{\sigma_1}{\sigma_3}<\kappa_{\max}
$$

建议第一轮：

$$
\tau_\sigma=0.002\text{m}
$$

$$
\kappa_{\max}=50
$$

如果太严格，可以放宽到：

$$
\tau_\sigma=0.001\text{m},\quad \kappa_{\max}=100
$$

但不要完全取消这个 gate。因为 rank 或 conditioning 不好时，(xyz\to\beta) 会对噪声非常敏感，MLP/LGBM 学到的逆映射也会不稳定。

---

## 6. rank 不满足时怎么调

如果大量样本不满足：

$$
\operatorname{rank}(J_h)=3
$$

或者 (\sigma_3) 很小，说明当前 (u) 的第三个变量 (\eta) 没有带来独立 workspace 厚度。按下面顺序处理。

### 6.1 改 layer path 的方向

不要只测：

$$
(\Delta s_1,\Delta s_2)=(+,+)
$$

还要测：

$$
(\Delta s_1,\Delta s_2)=(+,-)
$$

和：

$$
(\Delta s_1,\Delta s_2)=(-,+)
$$

因为你的目标不是简单让第一、二段都多动，而是制造一个“沿 backbone 分配曲率”的自由度。

我预计更可能有效的是：

$$
(\Delta s_1,\Delta s_2)=(+,-)
$$

或者：

$$
(\Delta s_1,\Delta s_2)=(-,+)
$$

它们更像在改变弯曲中心位置。

---

### 6.2 增大 (\eta) 对 (s_1,s_2) 的影响

如果 (\partial h/\partial\eta) 太小，说明 (\eta) 改变 layer 后末端几乎不变。可以把：

$$
\Delta s_1=0.075,\quad \Delta s_2=0.100
$$

扩大到：

$$
\Delta s_1=0.125,\quad \Delta s_2=0.150
$$

但要保持 (s_1,s_2) 在安全范围内。

---

### 6.3 去掉近直线区域

注意，当：

$$
a\approx0,\quad b\approx0
$$

时：

$$
\frac{\partial g}{\partial\eta}
===============================

\begin{bmatrix}
s_1'(\eta)a\
s_1'(\eta)b\
s_2'(\eta)a\
s_2'(\eta)b\
0\
0
\end{bmatrix}
\approx0
$$

所以 (J_h) 不可能 rank 3。这不是 bug，而是几何事实：机器人几乎直的时候，改变 layer 比例没有意义，因为每段弯曲都接近 0。

因此可以设置：

$$
\rho=\sqrt{a^2+b^2}
$$

拒绝：

$$
\rho<2^\circ
$$

或：

$$
\rho<3^\circ
$$

如果你的目标 workspace 包含近直线最大伸长边界，那一块本身就是 workspace boundary，天然不适合要求局部 3D 可逆。可以单独用 active inverse / trajectory continuation 处理，不要让它污染主训练集。

---

### 6.4 必要时拆成多个 chart

如果一条 layer path 不能覆盖整个 (x=1.0\sim1.2)m slab，不要硬做一个全局 (g(u))。可以定义两个或三个 canonical charts：

$$
g^{(1)}(u),\quad g^{(2)}(u),\quad g^{(3)}(u)
$$

每个 chart 覆盖一个局部 workspace。然后有两种处理方式：

* 只选其中一个 chart 作为论文主实验区域；
* 或者给不同 chart 训练不同 expert，但这会偏离你现在想保持的单一 MLP 叙事。

第一阶段建议只做一个 chart，先把机制跑通。

---

## 7. rank=3 不是充分条件，还要做全局 branch 检查

即使每个局部点都有：

$$
\operatorname{rank}(J_h)=3
$$

也只能说明 (h) 局部可逆。它不能保证整个数据集全局不自交。

可能出现：

$$
u_i\neq u_j
$$

但：

$$
|h(u_i)-h(u_j)|<10\text{mm}
$$

且：

$$
|g(u_i)-g(u_j)|\ \text{很大}
$$

这就是 global fold。你当前 mixed dataset 的问题正是局部 (xyz) 邻域混入多个 beta/theta branch；100k 数据里 10mm 邻域 all-workspace theta p95 接近 (9.7^\circ)，beta-far 近邻比例超过 80%，而 beta-close 内部明显连续。

所以最终 gate 应该是两层：

第一层，Jacobian gate：

$$
\sigma_3>\tau_\sigma,\quad \kappa<\kappa_{\max}
$$

第二层，neighbor consistency gate：

$$
\text{10mm theta RMS p95}\le2^\circ\sim3^\circ
$$

以及：

$$
\text{multi-branch ball ratio}\approx0
$$

你自己的计划里也已经把 Jacobian condition number、local injectivity、workspace fold detection、branch margin、path-following success rate 列为可能补充 gate，这是合理的。

---

## 8. 我建议你马上做的 pilot

第一轮不要训练模型，先只生成和诊断。

### Step 1：生成 (u)-grid

取：

$$
a,b\in[-15^\circ,15^\circ]
$$

步长：

$$
0.5^\circ
$$

即 (61\times61)。

取：

$$
\eta\in[-1,1]
$$

先用 21 层。

总数：

$$
61\times61\times21=78{,}141
$$

这个规模很适合作为 pilot。

---

### Step 2：测试 4 条 layer path

分别测试：

$$
(\Delta s_1,\Delta s_2)\in
{(0.075,0.100),(0.075,-0.100),(-0.075,0.100),(0.125,-0.150)}
$$

每条 path 输出：

* (x,y,z) range；
* (x=1.0\sim1.2)m slab 里保留多少点；
* (J_h) pass ratio；
* (\sigma_3) p5 / p50 / p95；
* (\kappa) p50 / p90 / p95；
* 10mm voxel count；
* 10mm theta RMS p95；
* multi-branch ball ratio。

---

### Step 3：筛掉 rank 差的点

保留：

$$
\sigma_3>0.002\text{m}
$$

且：

$$
\kappa<50
$$

同时保留目标 slab：

$$
1.0\le x\le1.2
$$

---

### Step 4：做 xyz balanced sampling

先不要像之前那样从 6D mixed beta pool 里 balance。要在 canonical manifold + Jacobian gate 之后再 balance。

---

### Step 5：跑 inverse consistency 诊断

目标是：

$$
\text{10mm theta RMS p95}\le2^\circ\sim3^\circ
$$

如果达不到，但 beta-close 内部仍然很好，说明 (h(u)) 出现全局 fold；需要缩小 ((a,b,\eta)) 范围或换 layer path。

如果达到了，再训练 MLP/LGBM。

---

## 9. 一句话回答你的两个问题

第一，(u) 就是你人为定义的 3 维 canonical 构型坐标，第一版建议：

$$
u=(a,b,\eta)
$$

其中 (a,b) 是第三段两个弯曲方向，(\eta) 是第一、二段跟随第三段的连续 layer 分配变量。

第二，(\operatorname{rank}(J_h)=3) 不是手工指定出来的，而是通过设计 (g(u))、数值计算 (J_h)、用 SVD gate 筛选和调参实现的。核心检查是：

$$
J_h=\frac{\partial FK(g(u))}{\partial u}
$$

$$
\sigma_3(J_h)>\tau_\sigma,\quad \frac{\sigma_1(J_h)}{\sigma_3(J_h)}<\kappa_{\max}
$$

它的作用是确保你生成的 3D canonical manifold 在局部真的能形成一个可逆的 (xyz\leftrightarrow\beta) 映射，而不是又退化成薄曲面或折叠多解。
