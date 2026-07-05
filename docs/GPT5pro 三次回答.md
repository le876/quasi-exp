根据 2026-06-13 的最新进展文档，我建议下一阶段路线从 **“继续构造 single-branch 数据集”** 精确调整为：

[
\boxed{
\text{以 active beta-manifold / 第三关节优先基础网格作为几何骨架，先做全局连续层场选择与自交叠剪枝，再做 manifold-aware 张力重标注。}
}
]

也就是说，当前不再是单纯的 workspace 多分支问题，也不应该直接进入模型训练。最新结果显示：active beta-manifold 已经把 all-workspace 10mm theta p95 从 active xyz generation 的 8.917° 降到 0.727–1.092°，multi-branch ratio 也降到 0.0–0.123；此时严格 gate 主要失败在张力连续性，all10 T p95 仍为 109.2–118.7 N，beta-close T p95 仍为 75.4–113.0 N。 第三关节优先基础网格也证明几何方向是对的：11767 行、hard gate 通过、yz hull area 达到 1,233,167.91 mm²，theta p95 只有 1.096°，但 T p95 仍为 117.44 N，multi-branch ball ratio 仍为 0.4696。

---

# 1. 当前实现的主要矛盾判断

## 1.1 事实：主要矛盾已经发生阶段性转移

之前的主要矛盾是：

[
\text{同一 } \theta \text{ 下张力 allocator 多解}
]

论文原始方法用 Eq.(52) 的准静态平衡方程和 Eq.(53) 的最小化最大张力目标，再用 PSO 求 12 路张力。 但已有诊断显示，接近论文式 PSO 的同 theta 多 seed 可行率为 1.0，median pairwise tension MAE 为 731.85 N，p95 为 1108.30 N；这说明问题不是无解，而是可行张力解太多且缺少唯一化 tie-break。

segmented canonical 之后，这个问题基本被压住。同一 theta 的随机性不再是当前主要矛盾。

随后主要矛盾转为：

[
xyz \rightarrow \theta/T \text{ 的多分支监督目标不单值}
]

Phase 4A graph filtering 仍然 all-workspace 10mm theta p95 = 8.16°，T p95 = 131.63 N；Phase 4B candidate-pool single-branch 20k 虽然 hard gate 通过，但 all-workspace 10mm theta p95 仍为 7.137°，T p95 为 194.19 N，multi-branch ball ratio = 1.0。

然后 Phase 4F / 4G 之后，主要矛盾再次转移：

[
\boxed{
\text{几何分支基本可控后，张力标签沿连续 manifold 仍不够连续}
}
]

最关键证据是 active beta-manifold 的 theta 已经达标，但张力 gate 未过。`manifold_3d_quadrant` 的 all10 theta p95 = 0.750°，same-beta theta p95 = 0.460°，multi-branch ratio = 0.123，beta-NN T MAE = 12.9 N；但 all10 T p95 = 109.2 N，beta-close T p95 = 109.2 N。

## 1.2 当前主要矛盾

当前主矛盾应写成：

[
\boxed{
\text{如何构造一个在 } xyz \text{ 中局部单值、在 } \theta/\beta \text{ 中连续、在 } T \text{ 中平滑的 canonical label manifold}
}
]

它的两个方面是：

1. **几何侧**：(\beta/\theta \rightarrow xyz) 的 selected manifold 是否局部嵌入、无自交、无 10mm ball 内跨层混叠。
2. **力学侧**：在 selected manifold 上，(T^*(\theta)) 是否是连续的 canonical 张力场，而不是逐样本独立求解得到的跳变标签。

当前更突出的一面已经从几何侧转到力学侧，但 priority grid 仍有层重叠，所以不能说几何侧完全解决。

---

# 2. 从论文原理看，当前实现哪里是对的，哪里还缺关键条件

论文的完整链条是：

[
\beta \rightarrow \theta \rightarrow p \rightarrow r(\theta,T)=0 \rightarrow T
]

论文 Eq.(50) 用 6 个 (\beta) 参数生成 30 个关节角：(\beta_1,\beta_2) 对应第一段奇偶盘，(\beta_3,\beta_4) 对应第二段，(\beta_5,\beta_6) 对应第三段。 Eq.(51) 再用运动学得到末端点 (p_{target})。 Eq.(52) 是准静态力矩平衡，Eq.(53) 是最小化最大张力。

论文还明确说，仿真数据输入是末端盘中心的齐次笛卡尔坐标 ({}^0p_{target}=[x,y,z,1]^T)，输出是 12 组驱动张力和 30 个关节角，最大驱动力约束为 2000 N。

这里缺了一个监督学习前提：

[
xyz \mapsto \theta,T
]

必须是 **人为选定的 canonical 单值函数**。否则 MLP 不是在学习物理规律，而是在拟合多个逆解 branch 的平均值。

## 2.1 论文公式层面的关键补全

从公式上，应把 inverse model 改写为：

[
p = FK(\theta)
]

[
r(\theta,T)=0,\quad 0\leq T_j \leq T_{\max}
]

但这只定义可行集合：

[
\mathcal{S}(p)=
\left{
(\theta,T)
\mid
FK(\theta)=p,\ r(\theta,T)=0,\ 0\leq T\leq T_{\max}
\right}
]

监督学习需要的是一个选择算子：

[
C(p)=
\operatorname{SelectCanonical}
\left(
\mathcal{S}(p)
\right)
=======

(\theta^*(p),T^*(p))
]

当前所有实验本质上都是在补这个 (C(p))。

---

# 3. 对当前实现的审查

## 3.1 做对的部分

第一，`effective_beta` 修正是必须的。之前 meta beta 与 theta 存在符号不一致，所以后续用 `effective_beta_from_theta(theta)` 是正确的；当前实现已经把它纳入工具和测试。

第二，Phase 4 runner 设置 “theta continuity gate failed 就停止 relabel/model training” 是正确的。文档也明确指出 relabel 只能改张力，不会修复 theta branch 混叠。

第三，从 active xyz generation 转到 active beta-manifold 是正确方向。active xyz target 仍然 multi_branch_ball_ratio = 1.0，10mm beta-far pairs 占 340/379；文档判断原因是 target pool 来自 mixed reachable beta 空间，只要允许所有 reachable xyz，就天然有多个 beta branch。

第四，第三关节优先基础网格把“优先第三关节、前两段小”数学化为：

[
\beta_1=s_1\beta_5,\quad
\beta_2=s_1\beta_6,\quad
\beta_3=s_2\beta_5,\quad
\beta_4=s_2\beta_6,\quad
\beta_5=\beta_5,\quad
\beta_6=\beta_6
]

并约束 (s_1\leq s_2)，这比之前的口号式 distal-preferred 更可复现。

## 3.2 当前最主要的实现缺陷

### 缺陷 A：voxel 内单支选择不是 10mm ball 连续性

Phase 4H 已经证明，“每个 10mm voxel 内保留一层”没有解决问题。原因是 continuity 诊断使用的是 10mm ball，而不是互不重叠 voxel。即使每个 voxel 内单支，10mm ball 仍会跨越相邻 voxel；相邻 voxel 仍可能选不同 ((s_1,s_2)) 层。结果 all10 T p95 只从 117.44 N 降到 115.28 N，multi_branch_ball_ratio 反而从 0.4696 升到 0.5882。

所以不能再做独立 voxel 策略。必须做：

[
\boxed{
\text{workspace kNN graph / beta-plane graph 上的全局连续层场选择}
}
]

### 缺陷 B：张力 allocator 仍是逐样本 canonical，不是 manifold-aware canonical

当前 segmented/canonical 对同一 theta 是确定的，但它没有保证相邻 beta / 相邻 theta 的张力平滑。active beta-manifold 已经说明：theta p95 可以低到 0.727–1.092°，但 T p95 仍在 109–119 N。

这意味着张力分配器应从：

[
T_i^*=A(\theta_i)
]

改成：

[
{T_i^*}_{i=1}^{N}
=================

\operatorname{argmin}*{{T_i}}
\sum_i
\mathcal{L}*{eq}(\theta_i,T_i)
+
\lambda
\sum_{(i,k)\in E}
w_{ik}|T_i-T_k|^2
+
\text{tie-break terms}
]

也就是说，张力 canonical 不应只在单点定义，而应在 selected manifold 上定义为一个连续场。

### 缺陷 C：摩擦 Case 1/2 的历史状态可能被独立样本化破坏

论文在张力传递 Eq.(42) 附近说明，绳索摩擦 Case 1/2 的方向和作用可由 historical state 推断。 但现在数据集是独立采样点。如果当前实现按单点的几何符号、绳长变化或局部规则判断摩擦方向，那么在 manifold 上可能出现离散 case flip，直接造成 100 N 级张力跳变。

这点必须专项审查。当前 beta-close theta 很平滑但 T p95 高，正像是 “几何连续、摩擦/active set/canonical 解分支不连续”。

---

# 4. 下一阶段总体实验路线

我建议下一阶段命名为：

```text
Phase 5: Continuity-first canonical manifold and tension-field relabel
```

核心顺序：

```text
5A 固定层 / 层场几何审查
5B 全局连续层场选择
5C 10mm ball conflict pruning
5D friction-case / active-set discontinuity audit
5E manifold-aware tension relabel
5F 20k expansion + minimal model verification
```

不要直接 100k，不要先 MoE，不要先复杂神经网络。

---

# 5. Phase 5A：固定层 ablation，判断“层混叠”到底占多少

当前 priority grid 有 7 个 ((s_1,s_2)) 层。Phase 4H 失败说明局部 voxel 去重不够，但还没有回答一个关键问题：

[
\text{如果只取固定一层，张力和几何连续性能否过 gate？}
]

Codex 先做最便宜的固定层 ablation。

## 5A.1 数据

输入：

```text
data/priority_grid_third_joint_first_v1/dataset.parquet
data/priority_grid_third_joint_first_v1/dataset_meta.parquet
```

按 7 个层分别过滤：

```text
(s1,s2) =
(0,0),
(0,0.25),
(0,0.5),
(0.125,0.25),
(0.125,0.5),
(0.25,0.25),
(0.25,0.5)
```

每层约 1681 行。

## 5A.2 输出指标

每一层报告：

```text
hard gate
workspace yz hull area
all xyz <= 5mm / 10mm theta p95
all xyz <= 5mm / 10mm T p95
same beta kNN theta/T p95
multi_branch_ball_ratio
xyz-NN T MAE
beta-NN T MAE
friction_case_flip_rate
active_set_flip_rate
```

## 5A.3 判断

如果某个固定层满足：

```text
all10 theta p95 <= 1.5 deg
all10 T p95 <= 100 N
multi_branch_ball_ratio <= 0.10
xyz-NN T MAE <= 30 N
```

则说明 full grid 的主要问题确实是层混叠。下一步用这个固定层扩 20k 或做连续层场。

如果所有固定层 T p95 都在 100 N 以上，则说明张力 allocator 本身是主要瓶颈，直接进入 Phase 5D/5E。

---

# 6. Phase 5B：从 voxel 单支升级为全局连续层场选择

Phase 4H 的正确下一步不是“更小 voxel”，而是 “连续层场”。

## 6.1 层场定义

令：

[
u=(\beta_5,\beta_6)
]

[
L(u)\in \mathcal{L}
===================

{
(0,0),
(0,0.25),
(0,0.5),
(0.125,0.25),
(0.125,0.5),
(0.25,0.25),
(0.25,0.5)
}
]

然后：

[
\beta(u,L)=
[
s_1\beta_5,\ s_1\beta_6,\
s_2\beta_5,\ s_2\beta_6,\
\beta_5,\ \beta_6
]
]

目标不是每个 voxel 选一层，而是在 ((\beta_5,\beta_6)) 平面上选一个连续层函数 (L(u))，使映射：

[
u \mapsto xyz = FK(\beta(u,L(u)))
]

尽量无自交、局部平滑、张力平滑。

## 6.2 能量函数

Codex 实现一个 graph MRF / ICM 即可：

[
E(L)
====

\sum_i U_i(L_i)
+
\lambda_{\beta}
\sum_{(i,k)\in E_{\beta}}
\mathbf{1}[L_i\neq L_k]
+
\lambda_{\theta}
\sum_{(i,k)\in E_{\beta}}
|\theta_i^{L_i}-\theta_k^{L_k}|^2
+
\lambda_p
\sum_{(i,k)\in E_{\beta}}
|p_i^{L_i}-p_k^{L_k}|^2
+
\lambda_T
\sum_{(i,k)\in E_{\beta}}
|T_i^{L_i}-T_k^{L_k}|^2
+
\lambda_C C_{\text{self-overlap}}
]

其中 (C_{\text{self-overlap}}) 是非局部冲突惩罚：

[
C_{\text{self-overlap}}
=======================

\sum_{(i,k)\in E_{\text{conflict}}}
\mathbf{1}
[
|p_i-p_k|\le 10mm
\land
|u_i-u_k|>\epsilon_u
]
]

这项直接针对 10mm ball 多分支，而不是 voxel。

## 6.3 unary score

[
U_i(L)=
-w_3 \tilde B_3
+w_2 \tilde B_2
+w_1 \tilde B_1
+w_T \widetilde{T}*{max}
+w_R \widetilde r
+w_J \kappa(J_i)
-w_A A*{\text{local coverage}}
]

但在 priority grid 里 (B_3) 对所有层基本相同，所以更重要的是：

```text
低 s1
低 s2
低 max tension
低 residual
低 local Jacobian condition number
低 predicted tension jump
高 workspace coverage contribution
```

建议先 sweep：

```text
lambda_layer = [0.1, 1, 5, 10]
lambda_theta = [1, 5, 10]
lambda_T_pre = [0, 0.1, 0.3]
lambda_conflict = [10, 30, 100]
```

注意：`lambda_T_pre` 不要太大，因为当前 T 本身还没 relabel，不能让有噪声的 T 主导几何层选择。

## 6.4 Codex 任务

实现：

```text
scripts/analysis/select_global_layer_field.py
```

输入：

```text
data/priority_grid_third_joint_first_v1/priority_grid_pool.parquet
data/priority_grid_third_joint_first_v1/dataset.parquet
data/priority_grid_third_joint_first_v1/dataset_meta.parquet
```

输出：

```text
data/priority_grid_third_joint_first_v2_layerfield/
```

报告：

```text
layerfield_report.md
layerfield_decisions.csv
diagnostics/layerfield_diagnostics.json
```

硬性要求：

```text
不能只按 voxel 独立决策；
必须有 beta-plane graph 或 workspace kNN graph；
必须报告 conflict edges before/after；
必须报告 retained rows 和 coverage shrinkage；
必须报告 5mm 与 10mm continuity。
```

---

# 7. Phase 5C：10mm ball conflict pruning，而不是保留 95% 样本

Phase 4H 保留了 94.87% 样本，但几乎没改善。下一步要接受一个事实：

[
\boxed{
\text{为了得到 } xyz\rightarrow\theta/T \text{ 单值训练集，必须主动删除一部分自交叠区域。}
}
]

## 7.1 冲突图定义

建立 conflict graph：

[
(i,k)\in E_c
\quad\text{if}\quad
|p_i-p_k|\le r_p
\ \text{and}
d_{\beta}(i,k)>\epsilon_{\beta}
]

或：

[
|p_i-p_k|\le 10mm
\quad\text{and}\quad
L_i\neq L_k
]

这些点不能同时保留。

## 7.2 选择目标

[
\max_{z_i\in{0,1}}
\sum_i q_i z_i
--------------

\lambda
\sum_{(i,k)\in E_c} z_i z_k
---------------------------

\gamma
\operatorname{CoverageLoss}(z)
]

其中：

[
q_i=
-w_T\widetilde T_{max}
-w_R\widetilde r
-w_1\tilde B_1
-w_2\tilde B_2
+w_3\tilde B_3
+w_{\rho}\rho_{\text{local density}}
]

工程上不必精确求解最大独立集，Codex 可先用 greedy：

```text
按 conflict degree / quality score 排序；
每次删除 quality 低且 conflict degree 高的点；
直到 multi_branch_ball_ratio <= target。
```

## 7.3 三档剪枝强度

```text
P0: no pruning
P1: mild, max deletion 15%
P2: medium, max deletion 30%
P3: aggressive, max deletion 45%
```

如果 P2 才过 gate，接受 P2。连续性比保留率重要。

## 7.4 通过标准

2k / 12k pilot：

```text
all10 theta p95 <= 1.5 deg
all10 T p95 <= 105 N before relabel
multi_branch_ball_ratio <= 0.15
xyz-NN T MAE <= 30 N
coverage yz hull area >= 0.8 * priority_grid_v1
```

---

# 8. Phase 5D：张力跳变来源审计

在做 relabel 前，必须知道 100 N 张力跳变来自哪里。

## 8.1 必须新增日志字段

Codex 在 tension labeler 中记录：

```text
segment_1_success, segment_2_success, segment_3_success
segment_1_objective, segment_2_objective, segment_3_objective
active_cable_set_signature
friction_case_signature_per_segment
friction_case_signature_per_cable
constraint_active_signature
rms_rnorm_per_segment
tension_group_sum_1, tension_group_sum_2, tension_group_sum_3
max_tension_group
```

尤其是：

```text
friction_case_signature
```

论文说明摩擦方向依赖 historical state。 如果相邻 beta 点的 theta 很近，但 friction case signature 跳变，那么 T 跳变很可能不是物理连续张力场，而是独立样本的摩擦状态选择不连续。

## 8.2 诊断表

对所有 10mm beta-close / same-beta kNN pair 统计：

| pair type            | T p95 | case flip ratio | active-set flip ratio | segment causing jump |
| -------------------- | ----: | --------------: | --------------------: | -------------------- |
| no case flip         |       |                 |                       |                      |
| case flip only       |       |                 |                       |                      |
| active-set flip only |       |                 |                       |                      |
| both flip            |       |                 |                       |                      |

判断标准：

```text
如果 case flip pair 的 T p95 是 no flip 的 2 倍以上：
    优先修 friction history / case smoothing。
如果 active-set flip pair 主导：
    优先修 canonical allocator 的 tie-break / bounds / group balance。
如果两者都不解释：
    检查 quasi-static residual Jacobian conditioning。
```

## 8.3 Codex 任务

实现：

```text
scripts/analysis/audit_tension_discontinuity_sources.py
```

输出：

```text
runs/diagnostics/tension_discontinuity_audit_priority_grid_v2/REPORT.md
```

---

# 9. Phase 5E：manifold-aware 张力 relabel

这是下一阶段最关键的力学实验。

## 9.1 不要再用 all-workspace anchor

只能在 selected manifold 的 graph 上做 anchor：

```text
graph space:
    beta-plane coordinates u=(beta5,beta6)
    + same selected layer / nearby selected layer
    + xyz distance <= 10 or 15mm
forbidden:
    cross conflict edge
    cross branch edge
    deleted sample edge
```

## 9.2 两种 relabel 路线

### 路线 E1：iterative anchor relabel，低风险

对当前 segmented solver 加入：

[
T_{ref,i}^{(m)}
===============

\operatorname{weighted\ median}
\left(
T_k^{(m-1)}
\mid
k\in N(i)
\right)
]

然后每个样本重新求：

[
T_i^{(m)}
=========

\arg\min_T
\left[
|r(\theta_i,T)|^2
+
\lambda_{ref}|T-T_{ref,i}^{(m)}|^2
+
\lambda_0|T-T_{nom}|^2
+
\lambda_{\max}\operatorname{smoothmax}(T)
\right]
]

约束：

[
0\le T_j\le 2000,\quad rms_rnorm\le0.06
]

sweep：

```text
k = [8, 16, 32]
lambda_ref = [5, 10, 20, 40, 80]
iterations = [1, 2, 3]
anchor_stat = median / huber_mean
```

输出变体：

```text
priority_grid_layerfield_v2_relabel_k16_w20_iter2
priority_grid_layerfield_v2_relabel_k32_w40_iter2
...
```

### 路线 E2：linearized global QP，收益更大

在每个样本附近线性化：

[
r_i(T_i)
\approx
A_iT_i+b_i
]

整体求解：

[
\min_{{T_i}}
\sum_i
|A_iT_i+b_i|^2_{W_r}
+
\lambda_L
\sum_{(i,k)\in E}
w_{ik}
|T_i-T_k|^2
+
\lambda_0
\sum_i
|T_i-T_{seg,i}|^2
+
\lambda_g
\sum_i
\mathcal{B}(T_i)
]

约束：

[
0\le T_i\le2000
]

其中 (\mathcal{B}(T_i)) 是 group balance / smoothmax penalty。

这条路线可先只在 2k pilot 做，不要直接全量。

## 9.3 张力 relabel 的通过标准

2k / 12k pilot 必须达到：

```text
hard gate true
rms_rnorm q95 <= 0.045   # 比原 0.06 稍严格
max_tension <= 1600 N
all10 T p95 <= 90 N
same-beta T p95 <= 60 N
beta-close T p95 <= 70 N
xyz-NN T MAE <= 25 N
beta-NN T MAE <= 15-20 N
```

这里我建议把之前的 `beta-close T p95 <= 50 N` 暂时改成：

```text
2k pilot: <= 70 N
20k final: <= 60 N
paper-grade: <= 50 N
```

原因是当前 active beta-manifold 的 best beta-NN T MAE 已经到 12.9 N，但 beta-close T p95 仍 109.2 N；如果直接用 50 N gate，可能会阻止进入有价值的 relabel sweep。 但最终若要让四 split T MAE 稳定到 30–40 N，beta-close T p95 仍应压到 60 N 以下。

---

# 10. Phase 5F：20k 扩展与最小模型验证

只有当 2k/12k pilot 过连续性 gate 后，才生成 20k。

## 10.1 20k 数据生成目标

推荐数据集名：

```text
data/priority_grid_layerfield_v2_pruned_relabel_20k
```

或者：

```text
data/active_beta_manifold_quadrant_v2_relabel_20k
```

当前我更倾向于前者，因为 priority grid 的 workspace coverage 更大，且几何规则更可解释；但如果固定层 ablation 显示某个 active beta-manifold variant 的张力更容易 relabel，则可以并行保留后者。

## 10.2 20k 通过标准

数据质量：

```text
rows >= 18000
hard gate true
rms_rnorm q95 <= 0.045
max tension <= 1600 N
tension_lt0_ratio = 0
tension_gt_tmax_ratio = 0
```

几何连续性：

```text
all10 theta p95 <= 1.5 deg
all5 theta p95 <= 1.0 deg
multi_branch_ball_ratio <= 0.15
branch_count p90 <= 2
xyz-NN theta MAE <= 0.7 deg
```

张力连续性：

```text
all10 T p95 <= 90 N
all5 T p95 <= 75 N
same-beta T p95 <= 60 N
within-branch T p95 <= 50-60 N
xyz-NN T MAE <= 25 N
beta-NN T MAE <= 15-20 N
```

coverage：

```text
yz hull area >= 1.0e6 mm^2
x range 不低于 priority_grid_v1 的 80%
radius range 不低于 priority_grid_v1 的 80%
```

## 10.3 模型验证只跑最小集合

先跑：

```text
direct MLP
MLP_large
KNN
RF / LGBM
beta_aux MLP
```

另外必须新增一个 **physics-tension 两阶段 baseline**：

```text
xyz -> theta_hat
theta_hat -> tension by canonical allocator
```

因为如果张力标签仍有小量不可学跳变，直接监督 (xyz\rightarrow T) 可能不如预测 (\theta) 后由物理 allocator 计算 (T)。

模型 gate：

| split            |  T MAE | theta MAE |  EE p95 |
| ---------------- | -----: | --------: | ------: |
| iid              | ≤ 30 N |    ≤ 1.3° | ≤ 35 mm |
| radius           | ≤ 35 N |    ≤ 1.6° | ≤ 45 mm |
| beta/layer block | ≤ 42 N |    ≤ 2.2° | ≤ 65 mm |
| angular_sector   | ≤ 38 N |    ≤ 2.0° | ≤ 70 mm |

20k 晋级 100k 的硬条件：

```text
四 split 平均 T MAE <= 36-38 N
四 split 平均 theta MAE <= 1.8 deg
EE p95 不比 anchor v2 对照恶化
```

当前 anchor v2 对照 EE p95 是 iid 49.75 mm、radius 55.64 mm、beta_block 77.37 mm、angular_sector 89.38 mm。

---

# 11. 给 Codex 的直接任务清单

## Task 1：固定层 ablation

```text
scripts/analysis/run_priority_grid_fixed_layer_ablation.py
```

命令示例：

```bash
python scripts/analysis/run_priority_grid_fixed_layer_ablation.py \
  --dataset data/priority_grid_third_joint_first_v1/dataset.parquet \
  --meta data/priority_grid_third_joint_first_v1/dataset_meta.parquet \
  --out runs/diagnostics/priority_grid_fixed_layer_ablation_v1
```

必须输出：

```text
fixed_layer_summary.csv
fixed_layer_report.md
best_layer_by_theta.json
best_layer_by_tension.json
best_layer_by_coverage.json
```

## Task 2：全局连续层场选择

```text
scripts/analysis/select_global_layer_field.py
```

命令示例：

```bash
python scripts/analysis/select_global_layer_field.py \
  --pool data/priority_grid_third_joint_first_v1/priority_grid_pool.parquet \
  --dataset data/priority_grid_third_joint_first_v1/dataset.parquet \
  --meta data/priority_grid_third_joint_first_v1/dataset_meta.parquet \
  --graph-space beta5_beta6 \
  --knn 8 \
  --conflict-radius-mm 10 \
  --lambda-conflict-sweep 10,30,100 \
  --lambda-layer-sweep 1,5,10 \
  --out-root data/priority_grid_layerfield_v2_sweep
```

## Task 3：10mm ball conflict pruning

```text
scripts/analysis/prune_workspace_conflicts.py
```

命令示例：

```bash
python scripts/analysis/prune_workspace_conflicts.py \
  --dataset data/priority_grid_layerfield_v2_best/dataset.parquet \
  --meta data/priority_grid_layerfield_v2_best/dataset_meta.parquet \
  --radius-mm 10 \
  --beta-far-threshold 0.10 \
  --max-delete-frac-sweep 0.15,0.30,0.45 \
  --out-root data/priority_grid_layerfield_v2_pruned_sweep
```

## Task 4：张力跳变审计

```text
scripts/analysis/audit_tension_discontinuity_sources.py
```

命令示例：

```bash
python scripts/analysis/audit_tension_discontinuity_sources.py \
  --dataset data/priority_grid_layerfield_v2_pruned_best/dataset.parquet \
  --meta data/priority_grid_layerfield_v2_pruned_best/dataset_meta.parquet \
  --graph-space beta_manifold \
  --knn 8 \
  --xyz-radius-mm 10 \
  --out runs/diagnostics/tension_discontinuity_audit_layerfield_v2
```

## Task 5：manifold-aware anchor relabel

```text
scripts/analysis/relabel_manifold_anchor_canonical.py
```

命令示例：

```bash
python scripts/analysis/relabel_manifold_anchor_canonical.py \
  --dataset data/priority_grid_layerfield_v2_pruned_best/dataset.parquet \
  --meta data/priority_grid_layerfield_v2_pruned_best/dataset_meta.parquet \
  --graph-space beta_manifold \
  --no-cross-conflict-edge \
  --k-sweep 8,16,32 \
  --w-anchor-sweep 10,20,40,80 \
  --iterations-sweep 1,2,3 \
  --out-root data/priority_grid_layerfield_v2_pruned_relabel_sweep
```

## Task 6：20k pipeline

```text
scripts/pipelines/run_phase5_continuity_pipeline.py
```

要求：

```text
先 fixed-layer / layerfield / pruning 2k or 12k pilot
通过后 relabel
通过后生成 20k
通过后跑 minimal baselines
任一 hard/theta/T/oracle gate 失败则停止
```

---

# 12. 论文中应同步修改的 model inverse solution 逻辑

论文原文可以保留 “virtual prototype + Monte Carlo/PSO + MLP” 的框架，但必须补上 canonical inverse map。建议将 inverse model 逻辑写成：

[
\mathcal{M}_{c}
===============

\left{
\beta(u)\mid u\in\Omega,\ \beta(u)\text{ satisfies the distal-preferred canonical policy}
\right}
]

[
p = FK(\theta(\beta(u)))
]

[
\theta^*(p)
===========

\theta(\beta(u^*)),
\quad
u^*
===

\operatorname{argmin}*{u\in\Omega}
|FK(\theta(\beta(u)))-p|^2
+
\lambda_c \Phi*{\text{canonical}}(\beta(u))
]

[
T^*(\theta)
===========

\operatorname{argmin}*{T}
\left[
|r(\theta,T)|^2
+
\lambda_L
\sum*{k\in N(\theta)}
w_k|T-T_k^*|^2
+
\lambda_0|T-T_{nom}|^2
+
\lambda_{\max}\operatorname{smoothmax}(T)
\right]
]

这样论文的学习目标不再是：

[
xyz\rightarrow \text{任意可行 }(\theta,T)
]

而是：

[
\boxed{
xyz\rightarrow
\text{canonical branch 上的 }
(\theta^*,T^*)
}
]

这与原论文低速准静态、分段变曲率、力矩平衡和 MLP 实时控制的思想一致；不同点是你补上了冗余逆解下监督标签必须单值化的必要条件。论文已经强调低速场景下可忽略惯性和速度项，并用准静态力矩平衡建模。 当前工作要补的不是推翻论文，而是把 Eq.(52)+(53) 从“可行解求解”升级为“可学习 canonical 标签生成”。

---

# 13. 最终建议

下一步不要继续：

```text
active xyz generation
voxel 内单支过滤
直接训练 MLP / MoE
直接扩 100k
```

应该立即做：

```text
Phase 5A 固定层 ablation
Phase 5B 全局连续层场选择
Phase 5C 10mm ball conflict pruning
Phase 5D friction / active-set 张力跳变审计
Phase 5E manifold-aware tension relabel
Phase 5F 20k minimal model verification
```

最重要的判断标准是：

```text
几何先过：
all10 theta p95 <= 1.5 deg
multi_branch_ball_ratio <= 0.15

张力再过：
all10 T p95 <= 90 N
same-beta T p95 <= 60 N
xyz-NN T MAE <= 25 N

模型最后过：
四 split 平均 T MAE <= 36-38 N
EE p95 不比 anchor v2 恶化
```

当前最有希望的基础不是 mixed，也不是 active xyz，而是：

[
\boxed{
\text{priority grid / active beta-manifold}
+
\text{global layer field}
+
\text{conflict pruning}
+
\text{manifold-aware tension relabel}
}
]

这条路线同时对应论文原理、当前实验事实和最终模型泛化目标。

根据 2026-06-13 的进度文档和原论文，我的判断是：**下一阶段不要再围绕 mixed workspace 做事后筛选，也不要直接上模型训练；应以“连续几何流形 + 图一致张力 canonical relabel + 路径跟踪验证”为主线。** 现在几何 branch 已经有可行方向，真正阻碍角度/张力拟合精度继续提升的主要矛盾正在从 `xyz -> theta` 多分支，转移到 **单值几何流形上的张力标签连续性不足**。

---

## 1. 先给结论：下一步主线

建议交给 Codex 的下一阶段实验总目标是：

[
\boxed{
\text{构造局部单值、路径连续、张力可重标定的 canonical inverse dataset}
}
]

具体路线是：

```text
Phase 5A：固定/连续比例层的 beta-manifold 数据集，而不是 mixed workspace 数据集
Phase 5B：全局连续 layer-field 选择，而不是 voxel 内独立单支选择
Phase 5C：在选定几何流形上做 graph-anchored canonical tension relabel
Phase 5D：先训练 theta-only / theta+EE baseline，再训练 theta+T / physics allocator hybrid
Phase 5E：以路径跟踪作为最终验收，而不是只看随机 split MAE
```

当前最接近正确方向的是两个分支：

1. **active beta-manifold / manifold_3d_quadrant**：几何 branch 已基本解决，但张力 p95 仍偏高。
2. **第三关节优先 priority grid**：workspace 覆盖更大，theta 连续性好，但 7 层比例网格仍有局部层重叠，张力连续性没有达标。

下一轮不要直接扩 100k。应先在 2k/12k/20k 级别把“几何单值 + 张力连续 + 路径可跟踪”跑通。

---

## 2. 原论文原理审查：原文缺了一个关键前提

原论文的训练数据输入是末端盘中心的齐次笛卡尔坐标：

[
{}^0p_{target}=[x,y,z,1]^T
]

输出是 12 路驱动绳张力和 30 个关节角，并设置最大驱动力 2000 N 约束。原文还用 6 维 (\beta) 生成 30 个关节角，即 (\beta_1,\beta_2) 对应第一段 odd/even 角，(\beta_3,\beta_4) 对应第二段，(\beta_5,\beta_6) 对应第三段。

随后原文用运动学得到末端点：

[
{}^0p_{target}={}^0_{30}T\cdot{}^{30}p_{end}
]

再用 30 组准静态力矩平衡方程：

[
f(\theta_i,Tension_j)=0
]

以及“最小化最大绳张力”的目标：

[
\min_{Tension}\max_{j=1,\dots,12}Tension_j
]

最后用 PSO 求得 12 路张力。 原文 MLP 则从 3D Cartesian coordinates 预测 30 个 joint angles 和 12 个 cable tensions，输出层共 42 个神经元。

这里的理论缺口是：**Eq.(52)+(53) 定义的是可行平衡解和张力安全目标，不定义唯一、连续、可监督学习的逆解分支。**

对冗余连续体机器人而言，真实问题不是单值函数：

[
p\mapsto(\theta,T)
]

而是一个可行解集合：

[
\mathcal{S}(p)=
\left{
(\theta,T)
\mid
FK(\theta)=p,\
r(\theta,T)=0,\
0\le T_j\le T_{\max}
\right}
]

要训练 MLP，就必须额外定义一个 canonical selector：

[
c(p)=
\left(
\theta^*(p),T^*(p)
\right)
\in \mathcal{S}(p)
]

并且要求 (c(p)) 在路径邻域内尽量连续。否则 MLP 学到的是多个 branch 的平均，路径跟踪时就会出现角度跳变、张力跳变和 FK 回代误差。

因此，论文逆运动学部分应从“PSO 求一个可行解”改成：

[
\boxed{
\text{PSO / quasi-static solver 生成候选解；canonical branch policy 选择唯一连续解；MLP 学习该 canonical inverse map。}
}
]

---

## 3. 当前实现审查：哪些方向已经被证伪，哪些方向是对的

### 3.1 mixed / candidate-pool filtering 已经基本证伪

Phase 4B 的 candidate-pool single-branch 20k 在物理上可行，但不是单值流形：all-workspace 10mm theta p95 仍为 7.137°，all-workspace 10mm T p95 为 194.19 N，multi_branch_ball_ratio 仍为 1.0；因此没有继续执行 relabel 和模型训练。

这说明：**从已有 mixed pool 里事后过滤，不能把一个多分支 workspace 变成真正单分支训练集。**

### 3.2 active xyz target 也被证伪

Phase 4E 从 target xyz 出发生成 candidates，但 target pool 本身来自 mixed reachable beta 空间，因此仍然保留 workspace 多 branch。结果 all-workspace 10mm theta p95 为 8.917°，T p95 为 237.01 N，multi_branch_ball_ratio = 1.0；10mm 近邻中 beta-far pairs 占 340/379，仍主导不连续。

这个实验很关键：它证明问题不只是“candidate 不够多”，而是 **目标集合本身必须来自一个局部 injective 的 beta manifold**。否则 global selection 只能在混乱目标上做局部修补。

### 3.3 active beta-manifold 是正确方向，但张力还没解决

Phase 4F 的 active beta-manifold 明显改善了 theta/branch：三个 2k variant 的 all-workspace 10mm theta p95 降到 0.727–1.092°，manifold_3d_quadrant 的 same-beta theta p95 为 0.460°；multi-branch ratio 也降到 0.0、0.049、0.123，说明 branch 混叠已不再是最主要失败点。

但这三组都没过严格 gate，原因是张力连续性：all10 T p95 仍在 109.2–118.7 N，beta-close T p95 仍在 75.4–113.0 N；manifold_3d_quadrant 虽然 beta-NN T MAE 只有 12.9 N，但 beta-close T p95 仍为 109.2 N。

这说明下一步主矛盾已经变为：

[
\boxed{
\theta/\beta\text{ 几何流形基本连续，但 }T(\theta)\text{ 的 canonical label 仍有局部跳变。}
}
]

### 3.4 priority grid 方向有价值，但不能直接训练

Phase 4G 的第三关节优先基础网格把规则写成：

[
\beta_1=s_1\beta_5,\quad
\beta_2=s_1\beta_6,\quad
\beta_3=s_2\beta_5,\quad
\beta_4=s_2\beta_6,\quad
\beta_5=\beta_5,\quad
\beta_6=\beta_6
]

其中 (\beta_5,\beta_6\in[-10^\circ,10^\circ])，(s_2\in{0,0.25,0.5})，(s_1\in{0,0.125,0.25})，并保留 (s_1\le s_2)。这等价于“第三关节完整参与，第二关节较小比例参与，第一关节更小比例参与”。完整网格为 (41\times41\times7=11767) 行。

这个方向的优点是 workspace 覆盖明显扩大，yz hull area 达到 1,233,167.91 mm²，约为 standard sweep 的 2.5 倍；同时 all-workspace 10mm theta p95 = 1.096°，角度连续性已达标。

但张力没有改善：all-workspace 10mm T p95 = 117.44 N，multi_branch_ball_ratio = 0.4696，within_branch T p95 = 60.51 N。文档也明确判断它不能直接作为 20k/100k 主训练数据。

### 3.5 voxel 内单支选择也被证伪

Phase 4H 在 10mm voxel 内做 canonical 单支选择，rows 从 11767 降到 11163，保留率 94.87%，hard gate 仍通过；但 all xyz <=10mm T p95 只从 117.44 N 小幅降到 115.28 N，multi_branch_ball_ratio 反而从 0.4696 升到 0.5882，within_branch T p95 从 60.51 N 变差到 82.07 N。

失败原因很明确：当前诊断是 10mm ball，而不是互不重叠 voxel。即使每个 voxel 内只保留一层，相邻 voxel 之间仍可能选到不同 ((s_1,s_2)) 层，所以 10mm ball 仍跨层。文档结论也指出，下一步应从“每个 voxel 独立选层”升级为“全局连续的层场选择”，或者把 ((s_1,s_2)) 固定成少数连续子流形分别训练。

---

## 4. 矛盾结构判断

### 主要矛盾

当前主要矛盾是：

[
\boxed{
\text{为了路径跟踪，必须让 } xyz\rightarrow\theta/T
\text{ 在训练子域内成为局部单值且连续的监督目标。}
}
]

### 主要矛盾的主要方面

目前主要方面已经从 `xyz -> theta` 多分支转为：

[
\boxed{
T(\theta,\beta)
\text{ 的 canonical allocator 在局部连续 manifold 上仍不够平滑。}
}
]

理由是 active beta-manifold 和 priority grid 已把 theta p95 压到约 1°以内；但张力 p95 仍在 100 N 量级。

### 次要矛盾

次要矛盾包括：

1. workspace 覆盖面积与单值性之间的矛盾。覆盖越大，越容易跨 branch 或跨比例层。
2. 张力平滑与准静态残差之间的矛盾。不能为了平滑 T 破坏 Eq.(52) 的力矩平衡。
3. 论文复现口径与工程可学性之间的矛盾。原论文默认 (xyz\rightarrow42) 维输出，但工程上必须补 canonical selector。
4. 路径跟踪连续性与随机 split 指标之间的矛盾。随机测试 MAE 低，不代表路径上 (\Delta\theta,\Delta T) 平滑。

---

## 5. 下一轮核心理论公式

### 5.1 canonical 几何 branch

用 (\beta) 表示角度生成规则：

[
\theta=B(\beta)
]

运动学为：

[
p=F(\beta)=FK(B(\beta))
]

为了让 (p\rightarrow\beta\rightarrow\theta) 可学，需要选一个子流形 (\mathcal{M}\subset\mathbb{R}^6)，使得：

[
F|_{\mathcal{M}}:\mathcal{M}\rightarrow\mathcal{W}
]

在局部近似单射。也就是对相近的 (p_i,p_j)，不能出现很远的 (\beta_i,\beta_j)：

[
|p_i-p_j|\le r_p
\quad\Rightarrow\quad
|\beta_i-\beta_j|\le r_\beta
]

否则监督目标不连续。

第三关节优先可以定义为：

[
\beta=
[
s_1 b_5,\ s_1 b_6,\ s_2 b_5,\ s_2 b_6,\ b_5,\ b_6
]^T
]

[
0\le s_1\le s_2\le 1
]

这比之前“distal preferred”口号更好，因为它把规则写成了严格可复现的生成流形。

### 5.2 全局连续 layer-field

priority grid 的问题不是每个点的规则，而是相邻点可能选择不同 layer。因此应定义一个连续 layer field：

[
\ell(p)\in\mathcal{L}
]

其中：

[
\mathcal{L}=
{(s_1,s_2)\mid s_1\in{0,0.125,0.25},\
s_2\in{0,0.25,0.5},\
s_1\le s_2}
]

在 workspace kNN graph 上求：

[
\min_{\ell_i,m_i}
\sum_i m_i U_i(\ell_i)
+
\lambda_L\sum_{(i,j)\in E}w_{ij}m_im_j d_L(\ell_i,\ell_j)^2
+
\lambda_\theta\sum_{(i,j)\in E}w_{ij}m_im_j
|\theta_i^{\ell_i}-\theta_j^{\ell_j}|^2
+
\lambda_T\sum_{(i,j)\in E}w_{ij}m_im_j
\rho(|T_i^{\ell_i}-T_j^{\ell_j}|)
+
\lambda_D\sum_i(1-m_i)
]

其中 (m_i\in{0,1}) 表示是否保留样本。
这一步的关键不是“尽量保留 95% 样本”，而是 **允许删除边界/层切换/局部非单射区域，形成真正可学习的 workspace patch**。

### 5.3 graph-anchored tension relabel

在几何 branch 固定后，张力应重新 canonical 化：

[
T_i^*=
\arg\min_{0\le T_i\le T_{\max}}
\left[
w_r|r(\theta_i,T_i)|^2
+
w_a|T_i-\bar T_i^{graph}|^2
+
w_0|T_i-T_{nom}|^2
+
w_m \operatorname{smoothmax}(T_i)
\right]
]

其中 (\bar T_i^{graph}) 不是 all-workspace 近邻，而是同一 selected manifold 上的 kNN robust anchor：

[
\bar T_i^{graph}
================

\operatorname{RobustMedian}
\left(
T_j\mid j\in\mathcal{N}_{\mathcal{M}}(i)
\right)
]

更进一步，可以做全局图优化：

[
\min_{{T_i}}
\sum_i
\left[
w_r|r(\theta_i,T_i)|^2
+
w_0|T_i-T_{nom}|^2
+
w_m \operatorname{smoothmax}(T_i)
\right]
+
\lambda_G\sum_{(i,j)\in E_\mathcal{M}}
w_{ij}\rho(|T_i-T_j|)
]

约束：

[
0\le T_{ij}\le2000N,\quad
rms_rnorm_i\le0.06
]

注意：**不能只对 T 做后处理平滑。** 必须平滑后重新投影回准静态可行集，否则张力标签可能变得连续但不满足 Eq.(52)。

---

## 6. 给 Codex 的下一阶段实验计划

### Phase 5A：固定比例层子流形实验

目的：先验证“固定 ((s_1,s_2)) 层”是否能比 7 层混合更适合直接 MLP。

不要再把 7 层强行压成一个单值模型。先分别生成 7 个 fixed-layer dataset：

```text
data/priority_grid_fixed_layer_s1_000_s2_000_20k
data/priority_grid_fixed_layer_s1_000_s2_025_20k
data/priority_grid_fixed_layer_s1_000_s2_050_20k
data/priority_grid_fixed_layer_s1_0125_s2_025_20k
data/priority_grid_fixed_layer_s1_0125_s2_050_20k
data/priority_grid_fixed_layer_s1_025_s2_025_20k
data/priority_grid_fixed_layer_s1_025_s2_050_20k
```

生成方式：

```text
beta5,beta6 用 Sobol / grid / Latin hypercube 混合采样
固定 s1,s2
由 beta -> theta -> FK 得到 xyz
用当前 segmented/canonical solver 标注 T
计算 local injectivity 和 continuity
```

每层先跑 2k，再选 2–3 个最优层扩 20k。

验收指标：

| 指标                      |              2k 通过 | 20k 通过 |
| ----------------------- | -----------------: | -----: |
| hard gate               |               true |   true |
| all10 theta p95         |              ≤1.5° |  ≤1.5° |
| all10 T p95             | ≤110 N 初筛，≤90 N 理想 |  ≤90 N |
| beta-close T p95        |  ≤90 N 初筛，≤70 N 理想 |  ≤70 N |
| multi_branch_ball_ratio |              ≤0.15 |  ≤0.10 |
| xyz-NN T MAE            |              ≤35 N |  ≤30 N |
| beta-NN T MAE           |              ≤20 N |  ≤18 N |

这个实验很重要，因为它能回答：**张力 p95 主要来自 layer 混合，还是来自 tension allocator 本身。**

如果固定层后 T p95 明显下降，说明 layer 切换是主因。
如果固定层后 T p95 仍在 100 N 左右，说明 allocator continuity 是主因。

---

### Phase 5B：全局 layer-field 选择

若固定层覆盖不足，再做全局连续 layer-field，而不是 voxel 内单支。

Codex 实现：

```text
scripts/analysis/select_priority_grid_global_layer_field.py
```

输入：

```text
data/priority_grid_third_joint_first_v1/priority_grid_pool.parquet
```

输出：

```text
data/priority_grid_third_joint_first_v1_global_layerfield_v2/
```

核心参数：

```text
--graph-space xyz
--graph-k 16
--graph-radius-mm 10,15,20
--layer-smooth-weight 1,3,10
--theta-smooth-weight 1
--tension-smooth-weight 0.1,0.3,1
--drop-penalty 0.2,0.5,1.0
--min-component-size 200
--allow-drop-boundary true
```

候选能量：

[
U_i(\ell)
=========

w_d D_{distal}(\ell)
+
w_r \widetilde{rms_rnorm}*i
+
w_T \widetilde{T}*{max,i}
+
w_c C_{coverage,i}
]

但 pairwise 项应主导局部一致性：

[
V_{ij}(\ell_i,\ell_j)
=====================

\alpha_\ell|\ell_i-\ell_j|^2
+
\alpha_\theta|\theta_i^{\ell_i}-\theta_j^{\ell_j}|^2
+
\alpha_T\rho(|T_i^{\ell_i}-T_j^{\ell_j}|)
]

输出必须报告：

```text
retention ratio
connected components
workspace hull area
all5/all10 theta/T p95
multi_branch_ball_ratio
within/between branch T p95
xyz-NN / beta-NN oracle
path-neighbor Δtheta / ΔT
```

通过条件：

```text
retention ratio >= 0.65
multi_branch_ball_ratio <= 0.20
all10 theta p95 <= 1.5 deg
all10 T p95 <= 95 N
within_branch T p95 <= 50 N
xyz-NN T MAE <= 30 N
```

若它比固定层更好，再作为主 20k 候选。

---

### Phase 5C：张力 graph canonical relabel

几何流形确定后，再做张力重标定。建议三种 allocator ablation：

#### T0：当前 segmented/canonical baseline

作为对照，不改。

#### T1：local robust anchor relabel

先在 selected manifold 上构图：

```text
graph space = effective_beta + xyz
same layer only
k = 8,16,32
radius = 10/20mm
```

对每个样本计算：

[
T_{ref,i}=
\operatorname{HuberMean}
\left(T_j,\ j\in N(i)\right)
]

然后调用 segmented solver 重新求解：

```text
--anchor-scope selected_manifold
--distance-space effective_beta_xyz
--no-cross-layer-anchor
--w-anchor 5,10,20,40,80
```

#### T2：graph Laplacian target + feasibility projection

第一步先做全局平滑目标：

[
\tilde T=
\arg\min_{\tilde T}
\sum_i|\tilde T_i-T_i^{old}|^2
+
\lambda\sum_{(i,j)}w_{ij}\rho(|\tilde T_i-\tilde T_j|)
]

第二步对每个样本重新投影到准静态可行集：

[
T_i^*=
\arg\min_{T_i}
|r(\theta_i,T_i)|^2
+
\eta|T_i-\tilde T_i|^2
]

#### T3：integrated graph-anchored allocator

把 graph anchor 直接加入分段求解，而不是先 smooth 再投影。

优先级：

```text
residual feasibility
> tension bound
> graph continuity
> nominal pretension
> min max tension
```

张力 relabel 的验收指标：

| 指标                  |  silver |    gold |
| ------------------- | ------: | ------: |
| rms_rnorm q95       |   ≤0.06 |  ≤0.045 |
| max tension         | ≤2000 N | ≤1600 N |
| all10 T p95         |   ≤90 N |   ≤70 N |
| beta-close T p95    |   ≤70 N |   ≤50 N |
| same beta kNN T p95 |   ≤60 N |   ≤40 N |
| xyz-NN T MAE        |   ≤28 N |   ≤22 N |
| beta-NN T MAE       |   ≤18 N |   ≤12 N |
| (\Delta T) path p95 |   ≤80 N |   ≤50 N |

---

### Phase 5D：模型训练分三条，不要只做 direct 42 输出

原论文用 MLP 直接学：

[
xyz\rightarrow(\theta,T)
]

但下一阶段应比较三条路线。

#### M0：paper-style direct MLP

保留作为论文复现 baseline：

```text
input: x,y,z
output: theta_1..30 + T_1..12
```

#### M1：theta-only / beta-aux MLP

先证明几何逆解可学：

```text
input: x,y,z
output: theta_1..30
aux output: effective_beta_1..6
loss = theta loss + FK loss + beta consistency loss
```

loss：

[
\mathcal{L}_{geo}
=================

|\hat\theta-\theta|*1
+
\lambda*{FK}|FK(\hat\theta)-p|*2
+
\lambda*\beta|\beta(\hat\theta)-\beta|_1
]

这条路线用于验证路径跟踪几何精度。

#### M2：theta network + physics tension allocator

模型只预测 (\theta) 或 (\beta)，张力由 fast canonical allocator 根据预测 (\theta) 求解：

[
\hat T=A(\hat\theta)
]

这条路线更符合物理一致性。缺点是实时性可能弱于纯 MLP，所以要报告 allocator 时间。

#### M3：hybrid tension residual network

如果 allocator 稳定但稍慢，可以训练：

[
T=A(\theta)+\Delta T_{NN}(x,y,z)
]

或者：

[
T=T_{graph_canonical}+\Delta T_{NN}
]

其中 (\Delta T) 要小，并用 bound / residual loss 限制。

模型验收：

| 指标                      | 20k silver | 20k gold |
| ----------------------- | ---------: | -------: |
| theta MAE               |      ≤1.2° |    ≤0.8° |
| EE p95                  |     ≤35 mm |   ≤20 mm |
| T MAE                   |      ≤35 N |    ≤25 N |
| T RMSE                  |      ≤55 N |    ≤40 N |
| tension bound violation |          0 |        0 |
| predicted residual q95  |      ≤0.08 |    ≤0.06 |
| path (\Delta T) p95     |     ≤100 N |    ≤60 N |

---

### Phase 5E：路径跟踪验证必须提前加入

原论文路径跟踪不是随机点预测，而是把预定义路径分成多个 waypoints，依次计算每个 waypoint 所需的关节角和绳张力；原文也说明增加 waypoints 可以让路径更平滑，降低相邻点之间的绳长和构型变化。

因此，模型最终验收必须加入 path metrics：

```text
scripts/eval/evaluate_path_tracking.py
```

测试路径建议三类：

1. **paper curve**：复现原文 Eq.(54) 风格路径。
2. **workspace patch 内圆/螺旋/8 字路径**：保证完全落在 selected manifold 的 workspace hull 内。
3. **边界路径**：沿 selected manifold 边缘测试 extrapolation。

路径指标：

[
e_{EE,k}=|FK(\hat\theta_k)-p_k|
]

[
\Delta\theta_k=|\hat\theta_{k+1}-\hat\theta_k|
]

[
\Delta T_k=|\hat T_{k+1}-\hat T_k|
]

[
J_T=|\hat T_{k+1}-2\hat T_k+\hat T_{k-1}|
]

报告：

```text
EE mean / p95 / max
theta step p95
T step p95
T jerk p95
tension bound violation
quasi-static residual q95
inference time
path success ratio
```

路径跟踪通过标准：

| 指标                |                                 目标 |
| ----------------- | ---------------------------------: |
| EE p95            |                          ≤20–30 mm |
| EE max            |                             ≤50 mm |
| theta step p95    |                              ≤1.5° |
| T step p95        |                              ≤80 N |
| T jerk p95        |                             ≤120 N |
| residual q95      |                         ≤0.06–0.08 |
| tension violation |                                  0 |
| inference time    | MLP ≤1 ms；hybrid allocator 单点需单独报告 |

---

## 7. 下一轮最小可执行实验矩阵

建议 Codex 按这个顺序做，不要并行乱跑。

### Step 1：fixed-layer 2k sweep

```text
generate_fixed_layer_manifold_dataset.py
```

跑 7 个 ((s_1,s_2)) 层，每层 2k。

选择条件：

```text
all10 theta p95 <= 1.5deg
all10 T p95 <= 110N
multi_branch <= 0.15
xyz-NN T MAE <= 35N
beta-NN T MAE <= 20N
```

输出：

```text
runs/diagnostics/fixed_layer_2k_sweep_summary.md
```

### Step 2：最佳 2–3 层做 tension relabel ablation

对每个最佳层跑：

```text
T0 current segmented
T1 local robust anchor
T2 graph smooth + projection
T3 integrated graph anchor
```

输出：

```text
runs/diagnostics/tension_relabel_fixed_layer_ablation.md
```

选择标准：

```text
all10 T p95 <= 90N
same-beta T p95 <= 60N
rms_rnorm q95 <= 0.06
max tension <= 2000N
```

### Step 3：扩最佳 fixed-layer 到 20k

只扩 1–2 个。

训练：

```text
theta-only MLP
theta+beta_aux MLP
paper-style theta+T MLP
theta + physics allocator
```

输出：

```text
runs/baselines/fixed_layer_20k_model_comparison/
```

### Step 4：再做 global layer-field

如果 fixed-layer 覆盖不足，再做：

```text
select_priority_grid_global_layer_field.py
```

目标不是保留最多数据，而是构造更大 workspace patch。

### Step 5：路径跟踪验证

在最优数据集上跑：

```text
evaluate_path_tracking.py
```

至少报告：

```text
paper curve
circle / spiral / figure-eight
boundary path
```

---

## 8. 对 gate 的修正建议

之前 `beta-close T p95 <= 50 N` 对 2k pilot 可能过严。因为当前最优 active beta-manifold 的 beta-NN T MAE 已经能到 12.9 N，但 beta-close T p95 仍在 109.2 N，这说明 p95 可能被局部 outlier、layer 切换或 allocator 跳变主导。

建议把 gate 分成三层：

### 8.1 几何 gate，不放松

```text
all10 theta p95 <= 1.5-2.0 deg
multi_branch_ball_ratio <= 0.15-0.20
xyz-NN theta MAE <= 0.8 deg
```

### 8.2 张力 label gate，分 silver/gold

```text
silver: all10 T p95 <= 100N, beta-close T p95 <= 90N, beta-NN T MAE <= 20N
gold:   all10 T p95 <= 70N,  beta-close T p95 <= 50N, beta-NN T MAE <= 12N
```

### 8.3 模型/路径 gate，最终决定是否可写入论文

```text
T MAE <= 30-35N
EE p95 <= 20-30mm
path ΔT p95 <= 80N
tension violation = 0
residual q95 <= 0.06-0.08
```

也就是说，2k pilot 不应因为 T p95 稍高就完全停止；但必须进入 tension relabel ablation，而不是直接训练最终模型。

---

## 9. 论文逆运动学部分应如何改写实验逻辑

建议在文章中把 inverse model 的实验分成三层：

### 第一层：canonical branch generation

说明原始逆解集合多值：

[
\mathcal{S}(p)=
{(\theta,T)\mid FK(\theta)=p,\ r(\theta,T)=0,\ 0\le T\le T_{\max}}
]

然后定义第三关节优先 canonical branch：

[
\beta=
[s_1b_5,s_1b_6,s_2b_5,s_2b_6,b_5,b_6]^T,\quad
0\le s_1\le s_2\le1
]

这保证低速路径跟踪时优先使用远端弯曲，减少近端大幅摆动。

### 第二层：canonical tension allocation

说明 Eq.(52)+(53) 只保证可行和平衡，不保证唯一张力：

[
T^*(\theta)=
\arg\min_{T}
\left[
|r(\theta,T)|^2
+
\lambda_a|T-T_{ref}|^2
+
\lambda_g\sum_{j\in N(i)}w_{ij}|T_i-T_j|^2
+
\lambda_m\operatorname{smoothmax}(T)
\right]
]

### 第三层：path-following evaluation

不要只报 test MAE。要报：

```text
path EE error
theta continuity
tension continuity
quasi-static residual
tension bound
inference time
```

这样才能和原文“path-following control”目标对应起来，而不是只做静态点回归。

---

## 10. 最终建议给 Codex 的任务清单

```text
Task 1: implement generate_fixed_layer_manifold_dataset.py
Task 2: run fixed-layer 2k sweep over 7 (s1,s2) layers
Task 3: implement diagnose_local_injectivity.py
Task 4: implement relabel_tension_graph_canonical.py
Task 5: implement select_priority_grid_global_layer_field.py
Task 6: train theta-only / theta+beta_aux / paper-style theta+T / theta+allocator baselines
Task 7: implement evaluate_path_tracking.py
Task 8: produce paper-ready ablation tables
```

下一步优先级：

[
\boxed{
\text{fixed-layer 2k sweep}
\rightarrow
\text{graph-anchored tension relabel}
\rightarrow
\text{20k geometry + tension model}
\rightarrow
\text{path tracking}
}
]

不要直接做：

```text
mixed 100k 扩展
Phase 4B/4E 再训练
voxel 内单支选择
MoE / branch classifier
只调 MLP 结构
不投影可行集的张力平滑
```

当前最有希望的论文路线是：**把原论文的“Monte Carlo + PSO + MLP”升级为“第三关节优先 canonical branch + graph canonical tension + MLP inverse map + path-following verification”。** 这样既能解释为什么原始 PSO/MLP 不稳定，也能形成一套完整的逆运动学求解实验闭环。
