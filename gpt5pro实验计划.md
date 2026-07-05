## 一、总判断：当前主要矛盾已经从“同一 theta 下 PSO 多解”转为“复杂采样下的 canonical 不足 + workspace 多分支混叠”

论文原文中的 inverse model 思路是：先用虚拟样机生成数据，输入为末端点齐次笛卡尔坐标 ({}^{0}p_{target}=[x,y,z,1]^T)，输出为 12 路驱动张力和 30 个关节角，并约束最大驱动力 2000 N；随后用 Eq.(52) 的 30 个准静态力矩平衡方程和 Eq.(53) 的“最小化最大张力”目标，通过 PSO 获得 12 路张力标签。 论文还把最终学习问题写成从 3D 末端坐标到 30 个 joint angles + 12 cable tensions 的 MLP 映射，输出维度为 42。

但是从现有实验看，Eq.(52)+(53) 的 PSO 形式缺少唯一化 tie-break：同一 theta 下 8 个 seed 都可行，但张力解的 median pairwise MAE 达到 731.85 N，p95 达到 1108.30 N；这说明问题不是“找不到可行解”，而是可行张力集合太大、优化目标没有选出唯一解。 segmented deterministic/canonical 已经把同 theta 的随机性基本消掉：repeat/pairwise 张力差为 0 N，并且比 PSO 快约 7.7 倍。 在标准 100k 简单角度网格上，segmented canonical 的 10mm tension MAE p95 只有 5.8468 N，Classic/TF MLP 张力 MAE 可到 4.20/3.36 N，这已经证明“张力本身不可学”不是根因。

所以我会把下一阶段主线定义为：

[
\textbf{先判别 }p\rightarrow(\theta,T)\textbf{ 是否单值，再在同一 branch 内优化 }T\textbf{ 的 canonical allocator。}
]

不能先盲目扩 100k，也不能先堆模型结构。mixed/distal-preferred 20k 和 100k 都通过硬质量门槛，但 10mm workspace 近邻 tension p95 约 250 N，100k 相对 20k 的直接回归提升只有 0–4%，说明简单扩数据不是主瓶颈。  anchor v2 second-stage 已把 20k 的 10mm T p95 降到 134.73 N、四 split 平均 T MAE 降到约 50.94 N，但 10mm theta p95 仍为 8.34°，这强烈指向 workspace 近邻跨构型分支配对的问题。

---

## 二、阶段 0：固定对照组与不可变验收口径

先不要新增大规模数据。先把所有实验都绑定到同一套 20k 诊断集、同一套 split、同一套 hard gate。

**固定数据集：**

1. `mixed_beta_20k_distal_preferred_segmented_canonical`
2. `mixed_beta_20k_distal_preferred_anchor_v2_second_stage_from_anchor_v2_k32_w20`
3. `standard_beta_sweep_100k_segmented_canonical` 的 20k stratified 子集，作为“标签可学上界参照”

**固定 hard gate：**

[
rms_rnorm\ q95 \le 0.06,\quad T_{max}\le 2000N,\quad saturation=0,\quad segmented_success_all=true
]

这些 gate 与现有实验包要求一致。

**固定报告表：**

每个实验都必须报告：

[
\text{data quality} + \text{seed stability} + \text{local continuity} + \text{oracle floor} + \text{model metrics}
]

其中 model metrics 只作为最后一层，不允许用模型误差掩盖标签不连续。现有决策背景明确指出，核心目标是张力标签“唯一、连续、可学习”，同时 EE/theta/残差/上限/饱和率只是约束或副指标。

---

## 三、阶段 1：诊断实验，区分 allocator 不唯一与 workspace 多分支

### 1.1 Same-theta seed stability：判断张力求解器是否仍然多解

抽样 300 个 theta，分层覆盖：

* source：`sobol_full/lhs_full/workspace_balanced/distal_biased`
* tension max 分位数：p50/p75/p90/p95
* residual 分位数：p50/p90/p95
* workspace 边界/内部
* 近邻 theta 混叠高/低区域

对每个 theta 跑：

1. paper-style PSO，8 seeds
2. current segmented canonical，重复 2 次
3. anchor v2 integrated 版本，重复 2 次
4. proposed convex/least-squares canonical，重复 2 次

**判断标准：**

| 指标                               | allocator 通过 | allocator 失败 |
| -------------------------------- | -----------: | -----------: |
| same-theta pairwise T MAE median |        ≤ 1 N |       > 10 N |
| same-theta pairwise T MAE p95    |        ≤ 5 N |       > 30 N |
| repeat max abs delta             |  0 或 <1e-6 N |      非零且不可解释 |
| feasible rate                    |          1.0 |        <0.99 |
| rms_rnorm q95                    |        ≤0.06 |        >0.06 |

若 same-theta 仍不稳定，说明主矛盾仍在 allocator；若 same-theta 稳定而 same-xyz 不稳定，说明主矛盾已经转为 branch/mapping。

### 1.2 三类 continuity 对比：same-beta、same-component、same-xyz

对同一 20k 数据构图，分别计算：

**A. same-beta-neighbor continuity**

在 normalized beta 空间找 kNN，k = 8/16/32。只在 (|\Delta \beta|) 小的点对上评价：

[
\Delta T,\quad \Delta \theta,\quad \Delta xyz,\quad \Delta residual
]

这是“同一构型流形上张力场是否连续”的主要指标。

**B. same-component continuity**

在同一 source component 内找 beta-kNN 与 xyz-kNN。比较：

[
\text{same-component xyz-neighbor} \quad vs \quad \text{cross-component xyz-neighbor}
]

如果 same-component 好而 cross-component 差，说明 mixed source 混合制造了局部混叠。

**C. same-xyz-neighbor continuity**

在 workspace 中找半径 5/10/20 mm 的点对，同时记录 beta/theta 距离。分成三档：

| 档位                          | 定义                                    | 用途      |
| --------------------------- | ------------------------------------- | ------- |
| xyz-neighbor + beta-close   | (\Delta xyz \le 10mm,\ \Delta\beta) 小 | 正常局部连续性 |
| xyz-neighbor + beta-far     | (\Delta xyz \le 10mm,\ \Delta\beta) 大 | 多分支证据   |
| xyz-neighbor + cross-source | (\Delta xyz \le 10mm,\ source不同)      | 采样混叠证据  |

**核心判据：**

[
\operatorname{Var}(T|xyz)=\operatorname{Var}*{within\ branch}(T)+\operatorname{Var}*{between\ branch}(T)
]

若 between-branch 占比 > 60%，当前主要问题不是张力 allocator，而是 (xyz\rightarrow(\theta,T)) 不单值。此时直接训练单值 MLP 必然学到“平均 branch”，theta/T 都会差。

### 1.3 Branch clustering：给每个 workspace 小球打 branch 标签

对每个 10mm workspace ball：

1. 用 normalized beta 或 theta 做 DBSCAN/HDBSCAN 聚类。
2. 统计 branch 数 (B(p))。
3. 记录每个 branch 的中心 (\bar{\beta}, \bar{\theta}, \bar{T})。
4. 计算 branch purity：

[
purity(p)=\frac{\max_b n_b}{\sum_b n_b}
]

**解释规则：**

| 现象                                 | 结论                     |
| ---------------------------------- | ---------------------- |
| (B(p)=1)，但 T p95 大                 | allocator/canonical 不足 |
| (B(p)>1)，branch 内 T 小、branch 间 T 大 | workspace 多分支          |
| source 分开后 (B(p)) 下降               | 采样混叠                   |
| branch 内 theta/T 仍大                | 物理求解或摩擦路径依赖问题          |

### 1.4 Oracle floor：估计单值回归理论下限

跑三个非训练型 oracle：

1. **xyz-NN oracle**：只用 xyz 最近邻预测 T/theta。
2. **branch-aware xyz-NN oracle**：xyz 最近邻，但限制 same branch。
3. **beta-NN oracle**：用 beta 最近邻预测 T，只评估 label smoothness。

如果 branch-aware oracle T MAE 很低，而 xyz-NN oracle 很高，则模型结构不是首要矛盾，输入信息不足才是首要矛盾。若 beta-NN oracle 仍高，则说明同一构型附近的张力 canonical 仍不够连续。

---

## 四、阶段 2：张力 canonical allocator 优化实验

### 2.1 保留 anchor v2 second-stage，但只作为 baseline，不作为终点

anchor v2 已经有明显收益：second-stage from k32/w20 是当前 20k 上最好的结果，10mm T p50/p90/p95 为 67.38/118.66/134.73 N，hard gate 通过。 但它没有降低 workspace 近邻 theta p95，因此不能继续只调 `k` 和 `w_anchor`。

下一步 anchor v2 只用于两件事：

1. 作为所有新 allocator 的对照组。
2. 为 integrated canonical solver 提供初始 anchor/reference。

### 2.2 把 canonical 目标集成进 solver，而不是只做 relabel 后处理

论文 Eq.(52) 给出力矩平衡约束，Eq.(53) 只最小化最大张力。 这不足以唯一化。建议改成分层/字典序 canonical 目标：

[
\begin{aligned}
T^*(q)=\arg\min_T\quad
& w_r|r(q,T)|*2^2
+w*{ref}|T-T_{ref}|*2^2\
&+w*{nom}|T-T_{nom}|*2^2
+w*{\Delta}\sum_{k\in\mathcal{N}(i)} a_{ik}|T_i-T_k|*2^2\
&+w*{\max}\operatorname{smoothmax}(T)
+w_{bal}\sum_g|T_g-\bar T_g\mathbf{1}|_2^2
\end{aligned}
]

约束：

[
0\le T_j\le T_{max},\quad |r(q,T)|*2\le \epsilon,\quad T*{max}\le2000N
]

其中 (r(q,T)) 是 Eq.(43)/(46) 的准静态力矩残差，论文原文说明了从第 (i) 个 disk 到末端整体做力矩平衡，并对 (z_i) 轴分量形成 (kD) 个方程。

**实现优先级：**

1. **Segmented lexicographic allocator**
   先满足 residual，再最小 maxT，再最小 (|T-T_{nom}|^2)，最后最小邻域变化。
   成本低，最适合 20k sweep。

2. **Segmented integrated anchor allocator**
   在当前 distal-to-proximal 分段求解中加入 (T_{ref})，不是 relabel 后处理。
   当前分段顺序已经明确为第三段 ({5,6,7,8})、第二段 ({3,4,9,10})、第一段 ({1,2,11,12})，且已被证明同 theta 稳定。

3. **Graph Laplacian allocator**
   在 same-branch beta graph 上做全局平滑：

   [
   \min_{{T_i}}\sum_i|r_i(T_i)|^2+\lambda\sum_{(i,k)}w_{ik}|T_i-T_k|^2+\lambda_0\sum_i|T_i-T_{nom}|^2
   ]

   只允许 same branch/same component 边，不允许跨 branch 强行平滑。

4. **Branch-aware allocator**
   对每个 branch 单独建立 (T_{ref}) 与图，不把 workspace 近邻当作张力近邻。

### 2.3 allocator 实验矩阵

| 编号 | 方法                                       |                20k 成本 | 风险              | 通过条件                                       |
| -- | ---------------------------------------- | --------------------: | --------------- | ------------------------------------------ |
| A0 | current segmented canonical              |                    已有 | baseline        | 复现实验                                       |
| A1 | anchor v2 second-stage k32/w20           | 已有，约 323s/20k relabel | 只降 T，不降 theta   | 作为 baseline                                |
| A2 | integrated (T_{ref}) in segmented solver |       预计 10–20min/20k | 权重过大会伤 residual | hard gate + T p95 < A1                     |
| A3 | lexicographic LS/QP allocator            |       预计 20–40min/20k | 残差线性化需验证        | same-theta p95 <5N, same-branch T p95 <50N |
| A4 | graph Laplacian global smoothing         |       预计 30–90min/20k | 跨 branch 误平滑    | branch 内改善，branch 间不强行变小                   |
| A5 | friction Case per-segment/path-dependent |           预计 1–2h/20k | 历史状态定义复杂        | 闭环路径张力跳变下降                                 |

**20k 晋级 100k 的最低条件：**

* hard gate 全过。
* same-theta pairwise T p95 ≤ 5 N。
* same-branch 10mm T p50/p90/p95 至少达到 20/60/90 N；理想目标 10/30/50 N。
* 四 split 平均 T MAE 相对当前 anchor v2 的 50.94 N 至少再降 25%，即 ≤38 N。
* EE p95 不允许比 anchor v2 平均恶化超过 10%。

---

## 五、阶段 3：采样策略实验，解决 workspace 混叠

当前 mixed 采样由 `sobol_full/lhs_full/workspace_balanced/distal_biased` 组成。 这个设计能增加复杂度，但也可能把不同构型 branch 放进同一 workspace 小区域。下一步不是删除所有复杂采样，而是给采样增加 branch consistency 约束。

### 3.1 source ablation

生成或重用四个 20k 子数据：

1. sobol-only
2. lhs-only
3. workspace-balanced-only
4. distal-biased-only

每个单独跑：

* same-source local continuity
* cross-source local continuity
* source-train/source-test
* branch count distribution

**判断：**

| 结果                        | 动作                                                |
| ------------------------- | ------------------------------------------------- |
| 单 source 连续，混合后不连续        | 保留 source，但训练/采样需 branch gating                   |
| workspace_balanced 引入最多混叠 | 改 workspace_balanced：每 voxel 只保留 canonical branch |
| distal_biased 连续性最好       | 把 distal preference 变成全局 branch selection policy  |
| sobol/lhs 都差              | beta 采样本身跨 branch 太强，需要分区采样                       |

### 3.2 branch-consistent dataset

在 workspace voxel 内执行 canonical branch selection：

[
b^*(p)=\arg\max_b \left[
w_3|\beta_{5:6}|
-w_{12}|\beta_{1:4}|
-w_T\max(T)
-w_R|r|
\right]
]

直观上，这延续“优先第三关节、前两段小”的 distal-preferred 原则；现有 mixed 设计已经包含这种倾向。

生成三版 20k：

| 数据集 | 规则                                                            |
| --- | ------------------------------------------------------------- |
| S0  | 原 mixed                                                       |
| S1  | branch-consistent filtered，只保留每 voxel 一个 branch               |
| S2  | branch-consistent regenerated，主动补齐低密度 voxel                   |
| S3  | source-conditioned，source 作为 branch hint，但测试时预测 source/branch |

**通过条件：**

* (B(p)>1) 的 10mm ball 比例下降 50% 以上。
* all-workspace 10mm theta p95 从 8.3° 降到 <3°。
* all-workspace 10mm T p95 至少从 250 N 降到 <120 N。
* 模型平均 T MAE ≤40 N。

---

## 六、阶段 4：模型结构实验，只在标签诊断通过后进行

模型实验要避免掩盖标签问题。只有当 branch-aware oracle 和 same-branch continuity 达标后，才比较复杂模型。

### 4.1 两阶段逆解模型：先预测 canonical beta/branch，再预测 theta/T

论文原数据生成里先用 6 维 (\beta) 参数定义 30 个角度的结构，再通过运动学得到末端点。 因此模型不应只直接从 xyz 回归 42 维输出。建议改成：

[
xyz \rightarrow \hat{\beta}*{canonical} \rightarrow \hat{\theta}*{1:30},\hat{T}_{1:12}
]

优点：

* beta 是 branch 的低维表达。
* 更容易做 branch selection。
* 可以把 distal-preferred policy 显式写进 inverse model。
* theta head 不再承担全部多分支歧义。

### 4.2 结构化多头网络

模型结构：

[
h=f_{\phi}(x,y,z)
]

[
\hat{\theta}=g_{\theta}(h),\quad
\hat{T}^{(3)}=g_3(h,\hat{\theta}*{21:30}),\quad
\hat{T}^{(2)}=g_2(h,\hat{\theta}*{11:20},\hat{T}^{(3)}),\quad
\hat{T}^{(1)}=g_1(h,\hat{\theta}_{1:10},\hat{T}^{(2)},\hat{T}^{(3)})
]

张力头按物理分段：

* 第三段：({5,6,7,8})
* 第二段：({3,4,9,10})
* 第一段：({1,2,11,12})

这和当前分段 canonical 求解顺序一致。

### 4.3 Mixture-of-experts / classifier + expert

若阶段 1 证明 workspace 多分支显著，则训练：

[
P(b|xyz),\quad y_b=f_b(xyz)
]

控制时不能让模型随机选 branch，必须有 deterministic branch policy：

[
b^*(xyz)=\arg\max_b score(b; xyz)
]

score 可以来自 distal preference、最小 max tension、最小 residual、路径连续性。

### 4.4 loss 设计

[
\mathcal{L}
===========

w_{\theta}|\hat{\theta}-\theta|*1
+w_T|\hat{T}-T|*1
+w*{FK}|FK(\hat{\theta})-xyz|*2
+w*{bound}\operatorname{ReLU}(\hat{T}-T*{max})
+w_{smooth}\sum_{(i,k)\in E}|\hat{T}_i-\hat{T}_k|^2
]

但 (w_{smooth}) 只能用于 same-branch graph，不能跨 workspace 多分支强行平滑。

---

## 七、阶段 5：20k 到 100k 的扩展决策

### 5.1 20k 必须先回答三个问题

| 问题                | 判断方式                                       | 决策                          |
| ----------------- | ------------------------------------------ | --------------------------- |
| 同 theta 是否稳定？     | seed stability                             | 不稳定则先修 allocator            |
| 同 xyz 是否多 branch？ | branch clustering + variance decomposition | 多 branch 则修采样/branch policy |
| 同 branch 张力是否连续？  | beta-kNN/same-branch continuity            | 不连续则修 canonical allocator   |

只有三个问题都有明确答案，才扩展 100k。

### 5.2 100k 扩展条件

满足以下任意一组即可扩展：

**方案 A：single-branch dataset 成立**

* 10mm all-workspace theta p95 < 3°
* 10mm all-workspace T p95 < 100 N
* 20k 四 split 平均 T MAE ≤35 N

**方案 B：multi-branch 但 branch-aware 模型成立**

* branch-aware oracle T MAE ≤20 N
* branch classifier accuracy ≥95%
* classifier + expert 四 split 平均 T MAE ≤35–40 N
* wrong-branch 样本有明确 fallback 策略

**方案 C：allocator 是主瓶颈且已改善**

* same-theta T p95 ≤5 N
* same-branch T p50/p90/p95 ≤10/30/50 N
* hard gate 全过
* T MAE 相对 anchor v2 second-stage 再降 ≥25%

### 5.3 成本门槛

现有 segmented 100k 生成约 2446 s，即约 40.8 min；mixed 100k 约 2426 s，速度约 0.024 s/sample。  anchor v2 second-stage 20k relabel 约 322.84 s，线性估算 100k 约 27 min。

建议成本上限：

| 阶段                    | 可接受成本 |
| --------------------- | ----: |
| 20k 诊断全套              | ≤2 小时 |
| 20k allocator sweep   | ≤4 小时 |
| 20k 模型全 split         | ≤4 小时 |
| 100k 生成 + relabel     | ≤3 小时 |
| 100k 全 split baseline | ≤8 小时 |

超过这个成本，必须证明 T MAE 或 continuity 有显著收益，否则不继续。

---

## 八、具体可执行实验序列

### Week 1：诊断，不改模型

1. 跑 300 theta seed stability。
2. 跑 same-beta/same-component/same-xyz 三类 continuity。
3. 跑 branch clustering。
4. 跑 oracle floor。
5. 输出 `diagnosis_report.md`。

**预期结论：**

* 若 same-theta 稳定、same-branch 低、same-xyz 高：主矛盾是 workspace 多分支。
* 若 same-theta 稳定、same-branch 仍高：主矛盾是 cross-sample canonical。
* 若 same-theta 不稳定：主矛盾回到 allocator。

### Week 2：allocator ablation

1. current segmented
2. anchor v2 second-stage
3. integrated anchor
4. lexicographic allocator
5. branch-aware graph smoothing

每个只跑 20k，输出：

* hard gate
* same-theta
* same-branch
* all-workspace
* oracle floor
* fast4 model

### Week 3：采样 ablation

1. source-only continuity
2. source-cross continuity
3. branch-consistent filtering
4. branch-consistent regeneration
5. distal-preferred canonical branch policy

判断是否保留四个 source：

* `sobol_full/lhs_full`：保留用于 beta 覆盖，但需要 branch 标签。
* `workspace_balanced`：只有在 voxel 内选 canonical branch 后保留。
* `distal_biased`：保留，并提升为 branch selection policy 的重要先验。

### Week 4：模型结构

只在标签通过后跑：

1. beta-first MLP
2. structured theta/tension multi-head
3. segmented tension heads
4. branch classifier + expert
5. mixture-of-experts

输出按 split 对比，并同时报告 oracle floor。模型不能只报 T MAE，必须报 FK 后 EE p95 与 tension bounds。

---

## 九、验收标准建议

### 9.1 数据质量硬门槛

| 指标                    |      门槛 |
| --------------------- | ------: |
| segmented_success_all |    true |
| rms_rnorm q95         |   ≤0.06 |
| max tension           | ≤2000 N |
| saturation ratio      |       0 |
| tension_lt0_ratio     |       0 |
| tension_gt_tmax_ratio |       0 |

### 9.2 张力一致性门槛

| 场景                                       |                    p50 |   p90 |    p95 |
| ---------------------------------------- | ---------------------: | ----: | -----: |
| same-theta repeat                        |                    0 N |   0 N |   ≤1 N |
| same-theta multi-seed canonical          |                   ≤1 N |  ≤3 N |   ≤5 N |
| same-branch 10mm，silver                  |                  ≤20 N | ≤60 N |  ≤90 N |
| same-branch 10mm，gold                    |                  ≤10 N | ≤30 N |  ≤50 N |
| all-workspace 10mm，single-branch dataset |                  ≤25 N | ≤70 N | ≤100 N |
| all-workspace 10mm，multi-branch dataset  | 不作为失败门槛，但必须分 branch 报告 |       |        |

### 9.3 模型门槛

20k silver：

| split          | T MAE | EE p95 | theta MAE |
| -------------- | ----: | -----: | --------: |
| iid            | ≤35 N | ≤50 mm |     ≤2.2° |
| radius         | ≤40 N | ≤60 mm |     ≤2.2° |
| beta_block     | ≤50 N | ≤80 mm |     ≤3.2° |
| angular_sector | ≤45 N | ≤85 mm |     ≤2.7° |

100k paper-grade：

| split          | T MAE | EE p95 | theta MAE |
| -------------- | ----: | -----: | --------: |
| iid            | ≤25 N | ≤35 mm |     ≤1.5° |
| radius         | ≤35 N | ≤50 mm |     ≤2.0° |
| beta_block     | ≤45 N | ≤70 mm |     ≤3.0° |
| angular_sector | ≤40 N | ≤80 mm |     ≤2.5° |

最终论文级目标可以更严格，但当前阶段先用这些作为“是否值得扩 100k”的工程验收线。

---

## 十、论文中 “model 求逆解” 部分应如何补全

建议把原论文第 4 节中 Eq.(52)+(53) 后的文字改成下面这种结构。核心是承认 inverse mapping 和 tension allocation 的非唯一性，并给出 canonical 化方案，而不是只说“PSO 求得张力”。

### 可替换/新增段落草稿

**Canonical inverse solution of the quasi-static model.**
For a desired end-effector position (p), the inverse control problem is to determine a physically feasible configuration (\theta\in\mathbb{R}^{30}) and a set of base cable tensions (T\in[0,T_{\max}]^{12}) satisfying the kinematic relation and the quasi-static moment equilibrium constraints. The kinematic relation is

[
p = FK(\theta),
]

and the quasi-static equilibrium residual is written as

[
r(\theta,T)=0,
]

where (r(\cdot)) is obtained from the disk-wise moment balance equations. In practice, the feasible tension set

[
\mathcal{T}(\theta)={T\mid |r(\theta,T)|\le \epsilon,\ 0\le T_j\le T_{\max}}
]

may contain multiple feasible solutions because the cable-driven robot is redundantly actuated and the equilibrium equations do not by themselves impose a unique tension distribution. Therefore, minimizing only the maximum cable tension is insufficient to guarantee a deterministic and smooth inverse label.

To obtain a single-valued inverse model suitable for supervised learning, we define a canonical tension allocator:

[
T^*(\theta)=
\arg\min_{T\in\mathcal{T}(\theta)}
\left[
\lambda_r|r(\theta,T)|*2^2
+\lambda*{\max}\operatorname{smoothmax}(T)
+\lambda_0|T-T_{nom}|*2^2
+\lambda_c|T-T*{ref}|_2^2
\right].
]

Here, (T_{nom}) is a nominal pretension vector, and (T_{ref}) is a local reference tension obtained from branch-consistent neighboring samples. The first term enforces quasi-static equilibrium, while the remaining terms select a unique, bounded, and smooth tension distribution among multiple feasible solutions.

For the three-section robot, the allocator is solved in a distal-to-proximal order. First, the distal cable group ({5,6,7,8}) is solved for Disk 21–30. Then the middle cable group ({3,4,9,10}) is solved for Disk 11–20 while accounting for the transmitted effects of distal cables. Finally, the proximal group ({1,2,11,12}) is solved for Disk 1–10, producing the final 12-dimensional base tension vector. This order follows the physical routing of the cables and avoids assigning distal segment loads to proximal actuation variables ambiguously.

For samples that are close in task space but belong to different configuration branches, tension continuity is not enforced across branches. Instead, a branch-consistent graph is constructed in the reduced configuration space, and the smoothness regularization is only applied between neighboring samples on the same branch. If multiple branches are available for the same task-space point, a deterministic branch-selection policy is used, for example selecting the branch with larger distal bending and smaller proximal bending, lower maximum tension, and lower residual. This makes the inverse training target single-valued:

[
c(p)=\left(\theta^*(p),T^*(\theta^*(p))\right).
]

The neural network then learns this canonical inverse map rather than an arbitrary feasible inverse solution. During evaluation, we report not only prediction errors of (\theta), (T), and end-effector position, but also seed stability of the tension allocator, local tension continuity within the same branch, and tension-bound violations.

### 还需要在论文里明确的一句话

原文现在容易让读者以为 Eq.(52)+(53)+PSO 自然会给出唯一张力。建议明确改成：

> Eq.(52) defines the feasible equilibrium manifold, while Eq.(53) is only a safety-oriented tension reduction criterion. A canonical tie-break is required to obtain deterministic and learnable labels for supervised inverse modeling.

这句话能直接解释为什么需要你的 segmented canonical / anchor / graph smoothing 实验，也能把“提升张力求解一致性以提升模型拟合性能”的逻辑补完整。
