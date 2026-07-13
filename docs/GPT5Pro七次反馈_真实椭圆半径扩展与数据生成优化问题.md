根据 2026-07-09 的实验记录，下一轮实验的核心任务应明确改为：**针对真实 `sin_sin_cos` 闭合椭圆，生成 trajectory-conditioned、single-branch、tube-aware 的 canonical 数据集**。当前旧的 (100/150)mm 成功结果来自同相位轨迹，SVD 只有一个非零奇异值，本质是空间线段；修正为真实 (x/y=\sin,z=\cos) 后，当前最佳 (87.5/131.25)mm 椭圆的 EE p95 仍为 (24.23)mm，说明真实闭合椭圆 tube 尚未被正确补密。

下面这份计划可以直接交给 Codex 执行。

---

# 真实椭圆 tube-aware canonical 数据生成实验计划

## 0. 实验总目标

本次实验目标不是继续扩大全局 (xyz) 点云，而是围绕真实闭合椭圆轨迹族生成高质量数据，使模型能够稳定学习：

$$
xyz\to\beta_6\to\theta_{30}\to FK
$$

优先目标：

$$
\text{amp}_{xy}=87.5\text{mm},\quad \text{amp}*z=131.25\text{mm},\quad EE*{95}\le5\text{mm}
$$

阶段性目标：

$$
\text{amp}_{xy}=75\text{mm},\quad \text{amp}*z=112.5\text{mm},\quad EE*{95}\le5\text{mm}
$$

挑战目标：

$$
\text{amp}_{xy}=100\text{mm},\quad \text{amp}*z=150\text{mm},\quad EE*{95}\le5\text{mm}
$$

当前不考虑 tension，只解决几何逆解和数据一致性问题。

---

# 1. 实验原理

## 1.1 为什么不能继续直接扩大 hierarchical FK 数据

hierarchical beta FK-only 500k 的 (xyz) 覆盖范围很大，500k 数据的 (x) 范围达到 (1.000001\sim1.199996)m，(y,z) 范围也接近 (\pm0.62)m；但 10mm 邻域内 (\theta) RMS p95 仍达到 (10.26^\circ)，模型 FK EE p95 高达几十毫米。这说明它的问题不是空间点不够，而是局部 (xyz) 邻域中混入多个 (\beta/\theta) branch。

因此，本实验不能把数据目标定义为：

$$
\text{maximize } xyz\text{ coverage}
$$

而应定义为：

$$
\text{maximize trajectory/tube coverage subject to single-branch consistency}
$$

也就是说，目标不只是“点在附近”，而是：

$$
|x_i-x_j|\le r\Rightarrow |\beta_i-\beta_j|\text{ small}
$$

---

## 1.2 为什么 path_a 成功但椭圆小

canonical layer-field U3 path_a 的优势是单支一致性好：10mm (\theta) RMS p95 只有 (0.418^\circ)，multi-branch ratio 为 0；在当前小椭圆上，MLP 已经能达到 (25/37.5)mm 椭圆 EE p95 (1.47)mm。

但 path_a 是由 (u=(a,b,\eta)) 投影到 (xyz) 的三维 canonical shell，不是完整实心 workspace。真实椭圆半径变大后，会有弧段离开当前 shell 或进入 tube 密度很薄的区域。你的最新记录也指出，真实 `sin_sin_cos` (87.5/131.25)mm 椭圆虽然 benchmark 里记录 `nn_p95=7.89mm`，但 `tube_count_p10=5`，EE p95 仍为 (24.23)mm。

所以当前真实瓶颈是：

$$
\text{真实闭合椭圆 tube 没有被 single-branch 数据充分覆盖}
$$

而不是：

$$
\text{MLP 容量不足}
$$

---

## 1.3 本实验的核心策略

本实验采用三层策略：

1. **轨迹几何纠错**：确保目标确实是二维闭合椭圆，而不是同相位线段。
2. **centerline active inverse**：先验证真实椭圆中心线能否沿同一个 canonical branch 连续求解。
3. **normal tube active generation + graph branch selection**：围绕中心线生成真实法向 tube，并用图优化选出连续 branch。

最终训练：

$$
xyz\to\beta_6
$$

再由确定映射得到：

$$
\theta_{30}=\Gamma(\beta_6)
$$

不再优先直接训练：

$$
xyz\to\theta_{30}
$$

因为你的现有真实椭圆结果已经显示，(\beta_6) 模型比 (\theta_{30}) direct 模型更好：最佳 (\beta_6) 模型 EE p95 为 (24.23)mm，而 direct (\theta_{30}) 模型在同一真实椭圆上为 (32.84\sim37.47)mm。

---

# 2. 总输出目录

Codex 新建：

```text
runs/true_sinsincos_tube_canonical_v1/
```

目录结构：

```text
runs/true_sinsincos_tube_canonical_v1/
  00_input_audit/
  01_true_trajectory_generation/
  02_support_recompute/
  03_centerline_active_inverse/
  04_centerline_branch_selection/
  05_tube_target_generation/
  06_tube_candidate_generation/
  07_tube_graph_branch_selection/
  08_dataset_assembly/
  09_model_training_beta6_u/
  10_true_ellipse_benchmark/
  11_failure_diagnostics/
  12_paper_summary/
```

---

# 3. Phase 0：输入数据审计

## 3.1 目的

避免再次出现 support 指标与真实轨迹不一致的问题。旧 `tube_aware_ellipse_enrichment.py` 的轨迹点生成需要复查，因为它可能按 `phase_x/phase_y/phase_z` 生成目标点，从而再次加密同相位线段。

## 3.2 输入

使用以下数据源：

```text
runs/dataset_optimization_large_ellipse_v1/03_improved_u_manifold_sweep_medium_top/A1/jacobian_pass_pool.parquet
data/hierarchical_beta_fk_x1p0_1p2_v1/x1p0_1p2_pool.parquet
runs/dataset_optimization_large_ellipse_v1/11_tube_aware_enrichment_100mm_v1/enriched_pool.parquet
runs/dataset_optimization_large_ellipse_v1/13b_final_ellipse_benchmark_tube100_true_sinsincos_v1/trajectory_points.parquet
runs/dataset_optimization_large_ellipse_v1/13b_final_ellipse_benchmark_tube100_true_sinsincos_v1/global_showcase_scoreboard.csv
```

## 3.3 审计内容

Codex 需要检查：

1. 每个数据源是否包含：

```text
x_m, y_m, z_m
beta1_rad ... beta6_rad
theta_1_rad ... theta_30_rad
```

1. A1 pool 是否包含：

```text
u_a_rad, u_b_rad, u_eta
s1, s2
sigma1_m, sigma2_m, sigma3_m, kappa
```

1. benchmark 文件中的 `trajectory_family` 是否全部为：

```text
sin_sin_cos
```

1. `trajectory_points.parquet` 中是否存在重复角度。若同一：

```text
model_id, candidate_id, angle_deg
```

出现多行，必须增加 `repeat_id` 或去重生成 `trajectory_points_unique.parquet`。

## 3.4 输出

```text
00_input_audit/input_manifest.csv
00_input_audit/schema_check.md
00_input_audit/trajectory_duplicate_report.csv
```

---

# 4. Phase 1：真实椭圆轨迹生成与几何验收

## 4.1 目的

彻底避免再次把同相位线段误判为真实椭圆。

## 4.2 轨迹定义

本实验只使用真实闭合椭圆：

$$
x(t)=c_x+a\sin t
$$

$$
y(t)=c_y+a\sin t
$$

$$
z(t)=c_z+1.5a\cos t
$$

也就是：

```text
trajectory_family = sin_sin_cos
```

第一轮只测试三个半径：

| ellipse_id | amp_xy |    amp_z |
| ---------- | -----: | -------: |
| E75        |   75mm |  112.5mm |
| E87p5      | 87.5mm | 131.25mm |
| E100       |  100mm |    150mm |

中心优先使用当前已知候选：

```text
path_a_cx1p0854_cym0p2951_czm0p1754_a87p5_px270_py270_pz270
path_a_cx1p0896_cym0p2071_czm0p2873_a75p0_px90_py90_pz90
path_a_cx1p0904_cym0p0507_czm0p3322_a100p0_px90_py90_pz90
```

同时可增加 3–5 个从 A1 shell 高支撑区域采样的候选中心。

## 4.3 轨迹几何 gate

对每条轨迹点集 (P\in\mathbb{R}^{N\times3}) 做 SVD：

$$
P-\bar{P}=U\Sigma V^\top
$$

要求：

$$
\frac{\sigma_2}{\sigma_1}>0.3
$$

$$
\frac{\sigma_3}{\sigma_1}<10^{-3}
$$

闭环 gap：

$$
|p_0-p_N|<10^{-6}\text{m}
$$

轴向幅值比例：

$$
A_x:A_y:A_z=1:1:1.5\pm5%
$$

每条轨迹还要计算 polygon area。若面积接近 0，直接判为线段伪椭圆。

## 4.4 输出

```text
01_true_trajectory_generation/true_ellipse_targets.parquet
01_true_trajectory_generation/trajectory_geometry_report.csv
01_true_trajectory_generation/trajectory_svd_report.md
```

字段包括：

```text
ellipse_id
angle_idx
angle_rad
x_target_m
y_target_m
z_target_m
amp_xy_mm
amp_z_mm
center_x
center_y
center_z
svd1_mm
svd2_mm
svd3_mm
rank2_gate_pass
closed_loop_gap_m
```

---

# 5. Phase 2：针对真实轨迹重新计算 support

## 5.1 目的

所有 support 指标必须针对 Phase 1 生成的真实 `sin_sin_cos` target points 重新计算，不能继承旧 candidate CSV 中的值。

## 5.2 支撑池

分别对以下池计算 support：

1. A1 base pool：

```text
A1/jacobian_pass_pool.parquet
```

1. old enriched pool：

```text
11_tube_aware_enrichment_100mm_v1/enriched_pool.parquet
```

1. hierarchical candidate pool：

```text
hierarchical_beta_fk_x1p0_1p2_v1/x1p0_1p2_pool.parquet
```

## 5.3 指标

对每条真实椭圆，计算：

### NN support

$$
d_i=\min_j|x_i-x_j^{pool}|
$$

输出：

```text
nn_mean_mm
nn_p50_mm
nn_p95_mm
nn_max_mm
```

### tube count

对半径 (r=15)mm：

$$
n_i=#{x_j^{pool}\mid |x_j^{pool}-x_i|\le15\text{mm}}
$$

输出：

```text
tube_count_min
tube_count_p10
tube_count_p50
tube_count_p95
```

### per-angle support

必须输出每个 angle 的：

```text
angle_deg
nn_mm
tube_count_15mm
nearest_beta_dist_to_A1_deg
nearest_x_diff_mm
nearest_y_diff_mm
nearest_z_diff_mm
```

## 5.4 输出

```text
02_support_recompute/support_summary_by_pool.csv
02_support_recompute/per_angle_support.parquet
02_support_recompute/support_by_angle_plots/
```

## 5.5 判定

若对某条椭圆：

$$
\text{nn}_{95}>8\text{mm}
$$

或：

$$
\text{tube_count}_{p10}<16
$$

则该椭圆不能直接训练，必须进入 active inverse 补密。

---

# 6. Phase 3：centerline active inverse

## 6.1 目的

先验证真实椭圆中心线是否能沿同一个 canonical branch 连续求解。若中心线不能连续求解，生成 tube 没有意义。

## 6.2 优先在 (u)-space 求解

定义：

$$
u_i=(a_i,b_i,\eta_i)
$$

映射：

$$
\beta_i=g(u_i)
$$

$$
\theta_i=\Gamma(\beta_i)
$$

$$
x_i^{FK}=FK(\theta_i)
$$

对目标点 (x_i^\star)，求解：

$$
\min_{u_i}
\frac{|FK(\Gamma(g(u_i)))-x_i^\star|^2}{\sigma_x^2}
+
\lambda_pP_{\text{prox}}(u_i)
+
\lambda_\kappa P_\kappa(u_i)
$$

其中：

$$
P_{\text{prox}}(u_i)=w_1|\beta_{1:2}|^2+w_2|\beta_{3:4}|^2
$$

并且：

$$
w_1>w_2>0
$$

这保留“优先第三关节，其次第二关节，再次第一关节”的软偏好。canonical layer-field U3 当前正是为了避免混合所有可达 (xyz)，而构造更接近单支连续层场的样本。

## 6.3 初值策略

对每个目标点，生成多个初值：

1. A1 pool 最近邻的 (u)；
2. 上一个角度点的解 (u_{i-1})；
3. 前向 continuation 的预测：

$$
u_i^{init}=u_{i-1}+(u_{i-1}-u_{i-2})
$$

1. hierarchical pool 最近邻 (\beta) 映射到近似 (u) 的结果；
2. 少量随机扰动：

$$
u^{init}+\epsilon,\quad \epsilon_a,\epsilon_b\in[-1^\circ,1^\circ],\quad \epsilon_\eta\in[-0.05,0.05]
$$

## 6.4 连续性正则

先逐点求候选，再做 centerline graph selection。对中心线候选 (c_i\in\mathcal{C}_i)，选择：

$$
\min_{c_i}
\sum_i
\left[
\alpha_x\frac{|FK(\beta(c_i))-x_i^\star|^2}{\sigma_x^2}
+
\alpha_pP_{\text{prox}}(\beta(c_i))
+
\alpha_\kappa P_\kappa(c_i)
\right]
+
\lambda_s\sum_i|\beta(c_i)-\beta(c_{i-1})|^2
+
\lambda_a\sum_i|\beta(c_{i+1})-2\beta(c_i)+\beta(c_{i-1})|^2
$$

闭环边必须包括：

$$
(N-1,0)
$$

确保 (\beta_0) 和 (\beta_N) 不在接缝处跳变。

## 6.5 centerline 验收

对每条椭圆：

| 指标                       |     pilot 门槛 |            强门槛 |
| ------------------------ | -----------: | -------------: |
| FK residual p95          |     (\le2)mm |       (\le1)mm |
| FK residual max          |     (\le5)mm |       (\le3)mm |
| (|\Delta\beta|) p95      |          无尖峰 |            无尖峰 |
| (|\Delta^2\beta|) p95    |          无尖峰 |            无尖峰 |
| seam (|\beta_0-\beta_N|) | (\le1^\circ) | (\le0.5^\circ) |
| (\kappa) p95             |     (\le120) |        (\le80) |

## 6.6 输出

```text
03_centerline_active_inverse/centerline_candidates.parquet
03_centerline_active_inverse/centerline_selected.parquet
03_centerline_active_inverse/centerline_inverse_report.csv
03_centerline_active_inverse/per_angle_residual_plots/
03_centerline_active_inverse/beta_smoothness_plots/
```

---

# 7. Phase 4：centerline strategy 对比

## 7.1 目的

比较不同中心线生成策略，确定后续 tube 的 branch 来源。

## 7.2 策略

### S1：A1-nearest-only

直接使用 A1 pool 最近邻。

用途：baseline。

### S2：pointwise active inverse

每个目标点独立 inverse。

用途：检查可达性，但可能跳 branch。

### S3：continuation active inverse

从角度 (0) 开始沿轨迹逐点 warm-start。

用途：检查沿路径连续性。

### S4：cyclic graph-smoothed inverse

在 S3 候选基础上加入闭环图优化。

用途：正式方案。

## 7.3 选择标准

优先选择：

1. residual p95 低；
2. seam 小；
3. (\Delta\beta)、(\Delta^2\beta) 无尖峰；
4. proximal bending 代价低；
5. (\kappa) 不爆炸。

输出：

```text
04_centerline_branch_selection/strategy_comparison.csv
04_centerline_branch_selection/best_centerline_branch.parquet
```

---

# 8. Phase 5：真实法向 tube target 生成

## 8.1 目的

同相位线段加密成功但真实椭圆失败的关键在于：真实椭圆需要覆盖两个独立方向。你的最新记录中也明确提出，真实椭圆需要覆盖两个主方向，而当前 tube 可能太像一条窄线。

因此，不能只补中心线，必须生成真实法向 tube。

## 8.2 构造轨迹局部坐标系

对中心线点 (p_i)，计算切向：

$$
\tau_i=\frac{p_{i+1}-p_{i-1}}{|p_{i+1}-p_{i-1}|}
$$

选择两个法向 (n_{1,i},n_{2,i})，满足：

$$
n_{1,i}\perp\tau_i,\quad n_{2,i}\perp\tau_i,\quad n_{1,i}\perp n_{2,i}
$$

并沿轨迹平滑传递法向，避免法向框架突然翻转。

## 8.3 Tube 扰动

pilot tube：

$$
\delta_t\in{-5,0,5}\text{mm}
$$

$$
\delta_1,\delta_2\in{-5,0,5}\text{mm}
$$

每个中心线点 (27) 个目标。

正式 tube：

$$
\delta_t\in{-10,-5,0,5,10}\text{mm}
$$

$$
\delta_1,\delta_2\in{-10,-5,0,5,10}\text{mm}
$$

每个中心线点 (125) 个目标，可以后续下采样。

目标点：

$$
x_{i,j}^{tube}=p_i+\delta_t\tau_i+\delta_1n_{1,i}+\delta_2n_{2,i}
$$

## 8.4 输出

```text
05_tube_target_generation/tube_targets_pilot.parquet
05_tube_target_generation/tube_targets_full.parquet
05_tube_target_generation/tube_frame_report.csv
```

字段：

```text
ellipse_id
angle_idx
tube_idx
x_target_m
y_target_m
z_target_m
delta_t_mm
delta_n1_mm
delta_n2_mm
tau_x, tau_y, tau_z
n1_x, n1_y, n1_z
n2_x, n2_y, n2_z
is_centerline
```

---

# 9. Phase 6：tube active inverse 候选生成

## 9.1 目的

对每个真实 tube target 生成多个 (\beta/u) 候选，为后续图优化选择连续 branch。

## 9.2 候选来源

### C1：centerline branch warm-start

对 tube 点 (x_{i,j})，使用同 angle 的 centerline 解 (u_i) 作为主初值。

### C2：局部 (u)-lattice

在 (u_i) 周围生成小 lattice：

$$
\Delta a,\Delta b\in{-1^\circ,0,1^\circ}
$$

$$
\Delta\eta\in{-0.05,0,0.05}
$$

对这些 (u) 做 FK，作为候选或 active inverse 初值。

### C3：A1/improved canonical pool 最近邻

检索半径：

$$
r=10\text{mm}
$$

保留最近 (K=8) 个。

### C4：hierarchical x1p0_1p2 pool 最近邻

作为可达性扩展，不直接作为训练标签。每个 target 保留最近 (K=16) 个，并按 (\beta)-space 聚类选代表。

### C5：active inverse multi-start

如果某点候选 residual 不达标，从 C1–C4 初值出发做局部 inverse refinement。

## 9.3 单点 inverse objective

对目标 (x^\star)：

$$
\min_u
\frac{|FK(\Gamma(g(u)))-x^\star|^2}{\sigma_x^2}
+
\lambda_r|u-u_{\text{centerline}}|^2
+
\lambda_pP_{\text{prox}}(u)
+
\lambda_\kappa P_\kappa(u)
$$

其中：

$$
\sigma_x=2\text{mm}
$$

建议：

$$
\lambda_r=0.5,\quad \lambda_p=0.1,\quad \lambda_\kappa=0.1
$$

可在 pilot 中 sweep。

## 9.4 候选保留规则

每个 target 至少保留：

$$
K_i\ge3
$$

强通过：

$$
K_i\ge8
$$

每个候选保存：

```text
target_id
candidate_id
source_type
x_target_m,y_target_m,z_target_m
x_fk_m,y_fk_m,z_fk_m
xyz_residual_mm
u_a_rad,u_b_rad,u_eta
beta1_rad...beta6_rad
theta_1_rad...theta_30_rad
sigma3_m
kappa
proximal_cost
dist_to_centerline_beta_deg
```

输出：

```text
06_tube_candidate_generation/tube_candidates.parquet
06_tube_candidate_generation/candidate_generation_report.csv
```

---

# 10. Phase 7：tube 图优化选择 single branch

## 10.1 图结构

节点是 tube target：

$$
V={i}
$$

每个节点有候选集合：

$$
\mathcal{C}_i
$$

边包括：

1. 同一 tube offset 下相邻 angle；
2. 同一 angle 的 tube 截面邻居；
3. (xyz) kNN，(k=8)；
4. 闭环 seam 边；
5. centerline-to-tube anchoring edge。

## 10.2 目标函数

选择候选 (c_i\in\mathcal{C}_i)：

$$
\min_{{c_i}}
\sum_iJ_i(c_i)
+
\lambda_s\sum_{(i,j)\in E}w_{ij}\rho(|\beta(c_i)-\beta(c_j)|^2)
$$

单点项：

$$
J_i(c)=
\alpha_x\frac{|FK(\beta_c)-x_i^\star|^2}{\sigma_x^2}
+
\alpha_pP_{\text{prox}}(\beta_c)
+
\alpha_r|\beta_c-\beta_{\text{centerline}(i)}|^2
+
\alpha_\kappa P_\kappa(c)
$$

推荐初值：

$$
\alpha_x=10,\quad \alpha_p=0.5,\quad \alpha_r=1.0,\quad \alpha_\kappa=0.1
$$

平滑项：

$$
w_{ij}=\exp\left(-\frac{|x_i-x_j|^2}{2\sigma_g^2}\right)
$$

$$
\sigma_g=15\text{mm}
$$

(\rho) 使用 Huber，防止少量异常点拖垮整图。

## 10.3 求解流程

1. MST / continuation 初始化；
2. ICM 局部优化 10–20 轮；
3. seam 修复；
4. 高 residual 节点重新生成候选；
5. 输出最终 selected branch。

## 10.4 验收指标

| 指标                    |     pilot 门槛 |            强门槛 |
| --------------------- | -----------: | -------------: |
| selected residual p95 |     (\le2)mm |       (\le1)mm |
| selected residual max |     (\le5)mm |       (\le3)mm |
| tube10 beta RMS p95   | (\le1^\circ) | (\le0.5^\circ) |
| tube10 theta RMS p95  | (\le1^\circ) | (\le0.5^\circ) |
| multi-branch ratio    |      (<0.5%) |            (0) |
| seam beta gap         | (\le1^\circ) | (\le0.5^\circ) |
| (\kappa) p95          |     (\le120) |        (\le80) |

输出：

```text
07_tube_graph_branch_selection/selected_tube_dataset.parquet
07_tube_graph_branch_selection/selection_report.json
07_tube_graph_branch_selection/per_angle_branch_smoothness.csv
07_tube_graph_branch_selection/branch_smoothness_plots/
```

---

# 11. Phase 8：数据集组装

## 11.1 数据集类型

生成三类数据。

### D-center

仅中心线 active inverse selected samples。

用途：验证模型能否拟合中心线。

### D-tube-small

真实椭圆小 tube：

$$
\delta_t,\delta_1,\delta_2\in{-5,0,5}\text{mm}
$$

用途：核心训练集。

### D-tube-full

更厚 tube：

$$
\delta_t,\delta_1,\delta_2\in{-10,-5,0,5,10}\text{mm}
$$

用途：验证泛化与鲁棒性。

### D-mix

A1 global pool + tube dataset 混合：

$$
D_{\text{mix}}=0.3D_{\text{A1}}+0.7D_{\text{tube}}
$$

因为当前目标优先是展示真实椭圆，而不是全局泛化。

## 11.2 输出

```text
08_dataset_assembly/D_center.parquet
08_dataset_assembly/D_tube_small.parquet
08_dataset_assembly/D_tube_full.parquet
08_dataset_assembly/D_mix_g30_t70.parquet
08_dataset_assembly/dataset_quality_report.md
```

## 11.3 数据质量 gate

| 指标                              |                       要求 |
| ------------------------------- | -----------------------: |
| true trajectory rank            |                        2 |
| target closed-loop gap          |              (<10^{-6})m |
| selected FK residual p95        |                 (\le2)mm |
| tube10 beta RMS p95             |             (\le1^\circ) |
| tube_count_p10 after enrichment |                  (\ge32) |
| normal thickness p10            | (n_1\ge5)mm, (n_2\ge3)mm |
| (\kappa) p95                    |                 (\le120) |

若 D-tube-small 不通过，不允许进入模型训练。

---

# 12. Phase 9：模型训练

## 12.1 模型形式

优先比较三种：

### M1：(xyz\to\beta_6)

$$
(x,y,z)\mapsto(\beta_1,\ldots,\beta_6)
$$

再：

$$
\theta_{30}=\Gamma(\beta_6)
$$

### M2：(xyz\to u)

$$
(x,y,z)\mapsto(a,b,\eta)
$$

再：

$$
u\to\beta_6\to\theta_{30}
$$

该模型用于验证 (u) 是否是更可学习的 canonical coordinate。

### M3：direct (xyz\to\theta_{30})

只作为对照。

暂不做 LGBM 主模型，因为已有 canonical 数据结果表明 MLP 明显优于 LGBM，尤其在 OOD 和轨迹平滑性上。

## 12.2 Split

必须包含：

1. iid；
2. angle_holdout：留出连续角度区间；
3. offset_holdout：留出部分 tube offset；
4. radius_holdout：例如训练 (75,87.5)，测试 (100)；
5. center_holdout：多中心阶段使用。

## 12.3 训练损失

基础：

$$
\mathcal{L}_\beta=|\hat{\beta}_6-\beta_6|^2
$$

可选 FK 辅助：

$$
\mathcal{L}_{FK}=|FK(\Gamma(\hat{\beta}_6))-x|^2
$$

总损失：

$$
\mathcal{L}=\mathcal{L}*\beta+\lambda*{FK}\mathcal{L}_{FK}
$$

sweep：

$$
\lambda_{FK}\in{0,0.1,0.5,1.0}
$$

## 12.4 输出指标

每个模型输出：

```text
beta MAE / p95
theta MAE / p95
FK EE mean / p95 / max
axis p95 max
fixed axis bias
per-angle EE curve
per-angle x/y/z error
Delta beta p95
Delta2 beta p95
```

输出：

```text
09_model_training_beta6_u/metrics_flat.csv
09_model_training_beta6_u/model_packages/
09_model_training_beta6_u/predictions/
```

---

# 13. Phase 10：真实椭圆最终 benchmark

## 13.1 Benchmark 目标

对以下轨迹测试：

1. train ellipse centerline；
2. train ellipse tube；
3. holdout radius；
4. holdout phase；
5. holdout center，若有多中心数据。

## 13.2 成功标准

### Pilot 成功

$$
a=75\text{mm},\quad EE_{95}\le5\text{mm}
$$

### 展示成功

$$
a=87.5\text{mm},\quad EE_{95}\le5\text{mm}
$$

### 强成功

$$
a=100\text{mm},\quad EE_{95}\le5\text{mm}
$$

同时要求：

$$
\text{axis p95 max}\le3\sim5\text{mm}
$$

$$
\text{fixed axis bias=false}
$$

$$
|\Delta^2\beta|\text{ 无尖峰}
$$

## 13.3 输出

```text
10_true_ellipse_benchmark/global_showcase_scoreboard.csv
10_true_ellipse_benchmark/best_largest_qualified_3d.png
10_true_ellipse_benchmark/best_largest_qualified_axis_errors.png
10_true_ellipse_benchmark/per_angle_error.csv
10_true_ellipse_benchmark/per_angle_error_plots/
```

---

# 14. Phase 11：失败诊断

## 14.1 如果 centerline inverse 失败

含义：该真实椭圆不在 A1 canonical branch 的连续可达区域。

处理：

1. 换中心；
2. 降半径；
3. 放宽 (u)-policy；
4. 进入 multi-chart 方案。

## 14.2 如果 centerline 成功但 tube 失败

含义：中心线可达，但法向厚度不足。

处理：

1. 增大 active inverse 候选数量；
2. 使用 normal tube in xyz，而不是只做 u-lattice；
3. 增加 hierarchical candidate source；
4. 使用 graph smoothing。

## 14.3 如果 tube 数据通过，但模型失败

含义：数据可达但模型表达不足或训练 split 太难。

处理：

1. 比较 (xyz\to\beta_6) 与 (xyz\to u)；
2. 加 FK auxiliary loss；
3. 增加 tube thickness；
4. 加 angle/context 输入作为实验对照；
5. 最后再考虑 stateful model。

## 14.4 如果 (xyz\to u) 明显优于 (xyz\to\beta_6)

说明 (u) 是更好的 canonical inverse coordinate。后续论文可描述为：

$$
xyz\to u\to\beta_6\to\theta_{30}
$$

## 14.5 如果 (xyz\to\beta_6) 优于 (xyz\to u)

说明 (u) 适合数据生成，但不一定适合作为学习输出。论文仍可保持：

$$
xyz\to\beta_6\to\theta_{30}
$$

---

# 15. 实验矩阵总结

## Pilot A：真实 centerline inverse

| 项  | 设置                                                   |
| -- | ---------------------------------------------------- |
| 椭圆 | sin_sin_cos                                          |
| 半径 | 75, 87.5, 100mm                                      |
| 点数 | 360                                                  |
| 方法 | A1 nearest + active inverse + cyclic graph smoothing |
| 通过 | residual p95 (\le2)mm, seam (\le1^\circ)             |

## Pilot B：小 tube active inverse

| 项    | 设置                                                  |
| ---- | --------------------------------------------------- |
| 半径   | 75, 87.5mm                                          |
| tube | (\pm5)mm                                            |
| 候选   | centerline warm-start + A1 pool + hierarchical pool |
| 通过   | tube residual p95 (\le2)mm, beta p95 (\le1^\circ)   |

## Pilot C：生成策略对比

| strategy        | 内容                                          |
| --------------- | ------------------------------------------- |
| centerline_only | 只中心线                                        |
| u_lattice       | 中心线 (u) 周围 lattice                          |
| normal_xyz_tube | 真实法向 xyz tube + active inverse              |
| hybrid          | xyz tube + (u) warm-start + graph smoothing |

预期优先选择 `hybrid`。

## Pilot D：模型对比

| 模型          | 输入  | 输出      |
| ----------- | --- | ------- |
| MLP-beta6   | xyz | beta6   |
| MLP-u       | xyz | u       |
| MLP-theta30 | xyz | theta30 |

通过：

$$
a=87.5\text{mm},\quad EE_{95}\le5\text{mm}
$$

---

# 16. 不再优先投入的旧路线

1. **不要继续直接扩大 hierarchical beta FK-only 训练集。**
   它已经证明空间覆盖大但局部 branch 混叠严重；目标附近有点不代表 (xyz\to\theta) 单值。

2. **不要继续只靠全局 (s_1/s_2/q) sweep。**
   simple improved (u)-manifold sweep 没有超过 A1，下一步应以真实轨迹为条件生成数据。

3. **不要优先调大模型容量。**
   同相位线段 tube 内 MLP 已经能低误差拟合，而真实椭圆失败主要来自目标 tube 没有被正确覆盖。

4. **不要先做 FK residual refinement。**
   refinement 可以作为后续上限验证，但会掩盖数据生成问题。当前目标是先证明真实椭圆 tube 数据可以让 MLP 学到正确 branch。

---

# 17. 最终交付物

Codex 完成后，必须给出：

```text
12_paper_summary/experiment_summary.md
12_paper_summary/final_dataset_quality_table.csv
12_paper_summary/final_model_quality_table.csv
12_paper_summary/final_true_ellipse_tracking.png
12_paper_summary/final_axis_error.png
12_paper_summary/failure_diagnosis.md
```

`experiment_summary.md` 必须回答：

1. 旧 same-phase 结果为什么无效；
2. 真实 sin_sin_cos 当前为什么失败；
3. centerline active inverse 是否成功；
4. normal tube active inverse 是否成功；
5. (xyz\to\beta_6) 和 (xyz\to u) 哪个更适合；
6. 是否达到 (75/112.5)mm 或 (87.5/131.25)mm 展示目标；
7. 当前方案是否可扩展为 multi-tube inspection dataset。

---

# 18. 最重要的一句话

这次实验的成败不取决于全局点云有多大，而取决于：

$$
\text{真实 sin/sin/cos 椭圆及其法向 tube 是否被同一条 canonical branch 连续覆盖}
$$

只有这个条件满足，MLP 才能从 (xyz) 稳定预测 (\beta_6/u)，再通过 FK 实现大半径闭合椭圆的毫米级跟踪。

# 下一个实验计划

# 更新后的核心判断

这轮 Codex 的 gate 停止是正确的，而且它改变了下一轮实验的重点。

已经确定的事实是：

1. E75、E87.5、E100 都是真正的二维闭合 `sin_sin_cos` 椭圆，几何 rank、闭环和幅值比例均通过。
2. 这三条椭圆在 A1 和旧 enriched pool 中几乎没有真实支撑，NN p95 达 (86\sim118)mm，15mm tube 内样本数 p10 为 0。
3. hierarchical pool 更接近这些轨迹，但 NN p95 仍有 (30\sim46)mm，而且局部 (\beta) RMS p95 约为 (12.6^\circ)，不能作为单值监督标签。
4. E75 在 A1 的 (u=(a,b,\eta)) 流形内做 active inverse 时，centerline residual p95 仍为 (61.64)mm。

由此可以确定：

> **E75/E87.5/E100 的现有中心和轨迹方向不属于 A1 canonical chart 的连续可达区域。**

但还不能确定：

> **这些椭圆对完整 6 维 (\beta) 构型空间也不可达。**

A1 active inverse 失败，只能证明单一 A1 参数化失败；hierarchical pool 距离较大也只说明已有离散采样没有充分覆盖。下一轮必须先区分四种情况：

$$
\text{全局不可达}
$$

$$
\text{全局可达，但 A1 chart 不可达}
$$

$$
\text{逐点可达，但不存在连续闭合构型分支}
$$

$$
\text{存在连续闭合分支，可进一步生成 single-branch tube}
$$

因此，下一轮不应直接围绕原 E75/E87.5/E100 继续补 tube，也不应马上训练模型。应改为：

# **全局可达性搜索 → 连续闭环 branch 搜索 → 局部多 chart tube 生成 → 模型训练**

这与冗余机器人轨迹求解中的成熟做法一致。IKLink 对每个轨迹 waypoint 生成多组 IK 解，再用动态图算法连接出全局连续运动；TORM 则直接优化整条关节轨迹，同时考虑末端误差和平滑性。([arXiv][1]) 对连续体机器人，已有工作也使用数值 Jacobian、伪逆和逐点迭代来求解空间轨迹，而不是依赖一个固定的全局闭式逆映射。([Ian Walker][2])

---

# 下一轮实验名称

```text
True Ellipse Global Reachability and Canonical Atlas Experiment V2
```

总输出目录：

```text
runs/true_ellipse_reachability_atlas_v2/
```

建议结构：

```text
00_baseline_audit/
01_reachability_pool/
02_ellipse_family_search/
03_fullbeta_pointwise_ik/
04_cyclic_branch_linking/
05_centerline_trajectory_optimization/
06_local_atlas_tube_generation/
07_multitrajectory_dataset/
08_beta6_model_training/
09_final_benchmark/
10_failure_diagnostics/
11_paper_summary/
```

---

# 一、实验总目标

## 1. 第一目标：找到真正可达的较大真实椭圆

优先寻找：

$$
a_{xy}\ge75\text{mm},\qquad a_z=1.5a_{xy}
$$

争取达到：

$$
a_{xy}=87.5\text{mm},\qquad a_z=131.25\text{mm}
$$

挑战：

$$
a_{xy}=100\text{mm},\qquad a_z=150\text{mm}
$$

但不再固定使用原来的三个中心。中心、两个相位差和半径都需要重新搜索。

## 2. 第二目标：找到连续闭合的构型 lift

对于轨迹：

$$
x_i^\star\in\mathbb R^3,\qquad i=0,\ldots,N-1
$$

需要找到：

$$
\beta_i\in\mathbb R^6
$$

满足：

$$
FK(\beta_i)\approx x_i^\star
$$

并且：

$$
\beta_{i+1}\approx\beta_i
$$

以及闭环：

$$
\beta_N\approx\beta_0
$$

## 3. 第三目标：围绕该闭合 branch 生成有厚度的 canonical tube

只有 centerline branch 通过后，才生成法向 tube 数据，并训练：

$$
xyz\to\beta_6\to\theta_{30}
$$

当前阶段仍不加入 tension，也不使用模型输出后的 FK residual refinement。

---

# 二、实验原理

## 1. 从固定 A1 流形切换到完整 (\beta_6) 可达性

当前 A1 的映射是：

$$
u=(a,b,\eta)\to\beta_6\to xyz
$$

它只覆盖完整 6 维构型空间中的一个 3 维子流形。E75 在 A1 中 residual 达 (61.6)mm，说明 A1 这张 chart 不包含该轨迹，但完整机器人仍有三个冗余自由度可能提供其他构型解。

所以首先应求：

$$
\min_{\beta\in\mathcal B}|FK(\beta)-x^\star|^2
$$

这里 (\mathcal B) 是完整有效 (\beta_6) 范围，而不是 A1 的 (g(u))。

## 2. 从逐点 IK 切换到 trajectory-level branch selection

即使每个点都存在 IK 解，也可能在相邻角度间跳 branch。学术界通常通过“每点多候选 + 全局连接”或“整条轨迹联合优化”解决。IKLink 明确采用每个 waypoint 的多组 IK 解和动态规划连接；TORM 将 task-space tracking 与 joint-space smoothness 联合优化。([arXiv][1])

## 3. 从单一全局参数化切换到局部 atlas

A1 失败也提示：一个全局 (u)-chart 未必能覆盖所需路径。流形规划方法通常使用多个局部 chart，并将其组织成 atlas，因为隐式流形往往不存在单一稳定的全局参数化。

这里的 multi-chart 不是混合多个不一致 branch，而是：

> 用多张重叠局部坐标图表达同一条连续构型 branch。

只有 chart overlap 中的 (\beta) 一致，才能合并到同一个训练集。

---

# 三、Phase 0：固定当前失败基线

## 目的

保存当前事实，作为下一轮的 negative control，防止后续错误地把 A1 失败解释成模型失败。

## 输入

```text
runs/true_sinsincos_tube_canonical_v1/
```

## 输出

```text
00_baseline_audit/current_failure_summary.json
00_baseline_audit/current_failure_summary.md
```

必须记录：

| 椭圆    | A1 NN p95 | hierarchical NN p95 | A1 centerline inverse p95 |
| ----- | --------: | ------------------: | ------------------------: |
| E75   |   86.04mm |             30.26mm |                   61.64mm |
| E87.5 |  117.98mm |             37.88mm |                   未进入正式求解 |
| E100  |  115.87mm |             45.81mm |                   未进入正式求解 |

这些旧中心保留为负例，不再作为下一轮唯一目标。

---

# 四、Phase 1：构建全局 reachability-only pool

## 1. 目的

建立更均匀的完整 (\beta_6\to xyz) 候选库，只用于：

* 判断空间点是否可能可达；
* 提供 full-(\beta) IK 初值；
* 搜索新的椭圆中心和方向。

它不直接用于训练，因此允许包含多 branch。

## 2. 基础范围

首先使用与现有 hierarchical 数据一致的范围：

$$
\beta_{1,2,3,4}\in[-5^\circ,5^\circ]
$$

$$
\beta_{5,6}\in[-15^\circ,15^\circ]
$$

使用 Sobol 或其他 low-discrepancy 采样，而不是笛卡尔全网格。

Pilot：

$$
N=2^{20}=1{,}048{,}576
$$

正式：

$$
N=2^{22}=4{,}194{,}304
$$

## 3. 可选 relaxed domain

由于 distal-priority 是软偏好，可以额外测试一个更宽但仍受模型有效范围约束的域：

$$
\beta_{1,2}\in[-7.5^\circ,7.5^\circ]
$$

$$
\beta_{3,4}\in[-10^\circ,10^\circ]
$$

$$
\beta_{5,6}\in[-15^\circ,15^\circ]
$$

只有在这些范围被当前运动学模型认定有效时才启用。该 pool 需要标记：

```text
source_domain = relaxed_proximal
```

不能和正式 canonical 数据混淆。

## 4. 保存字段

```text
sample_id
source_domain
beta1_rad ... beta6_rad
theta_1_rad ... theta_30_rad
x_m
y_m
z_m
```

不需要 balance。

## 5. 输出

```text
01_reachability_pool/sobol_current_bounds_1m.parquet
01_reachability_pool/sobol_relaxed_bounds_1m.parquet
01_reachability_pool/reachability_pool_merged.parquet
01_reachability_pool/pool_convergence_report.csv
```

## 6. 收敛检查

比较 1M 与 4M pool 对随机 workspace probe 的 NN 距离。如果从 1M 到 4M，NN p95 改善不足 (10%)，说明该范围内的几何覆盖已经接近饱和。

---

# 五、Phase 2：重新搜索真实椭圆族

## 1. 轨迹参数化

不再固定原来的中心和相位。令：

$$
p(t;\xi)=
\begin{bmatrix}
c_x+a\sin t\
c_y+a\sin(t+\phi_y)\
c_z+1.5a\sin(t+\phi_z)
\end{bmatrix}
$$

其中：

$$
\xi=(c_x,c_y,c_z,a,\phi_y,\phi_z)
$$

全局相位不影响几何形状，因此固定 (\phi_x=0)。

这始终保持三个轴的幅值比例：

$$
1:1:1.5
$$

但允许改变椭圆平面方向。

## 2. 几何 gate

每个候选必须满足：

$$
\frac{\sigma_2}{\sigma_1}>0.3
$$

$$
\frac{\sigma_3}{\sigma_1}<10^{-3}
$$

闭环 gap：

$$
|p(0)-p(2\pi)|<10^{-6}\text{m}
$$

并验证三个轴的幅值比误差不超过 (5%)。

## 3. Coarse search

### 中心候选

从 reachability pool 的 20mm voxel 中选择局部密度高、远离明显边界的 512 个中心。

中心应覆盖不同：

* (x)-bin；
* (\rho=\sqrt{y^2+z^2})-bin；
* (\psi=\operatorname{atan2}(z,y))-bin。

### 相位

$$
\phi_y,\phi_z\in{0^\circ,30^\circ,\ldots,330^\circ}
$$

剔除 rank 1 退化轨迹。

### 半径

$$
a\in{50,75,100}\text{mm}
$$

每条轨迹先采 36 点。

## 4. Reachability pool 支撑评分

计算：

$$
d_{95}=\operatorname{p95}_i\min_j|p_i-x_j^{pool}|
$$

$$
d_{\max}=\max_i\min_j|p_i-x_j^{pool}|
$$

以及 15mm 邻域空点比例：

$$
r_{\text{empty}}
================

\frac{
# {i:\text{15mm 内没有 pool sample}}
}{N}
$$

评分：

$$
S_{\text{geom}}
===============

d_{95}+0.5d_{\max}+20\text{mm}\cdot r_{\text{empty}}
$$

每个半径保留 top 100。

## 5. Refined search

对 top 100：

* 中心每轴扰动 ({-20,-10,0,10,20})mm；
* 相位在 coarse 值附近以 (7.5^\circ) 细化；
* 半径以 (12.5)mm 步长细化；
* 每条轨迹用 72 点。

保留每个半径 top 20。

## 6. 几何初筛 gate

Pilot：

$$
d_{95}\le10\text{mm}
$$

$$
d_{\max}\le20\text{mm}
$$

强通过：

$$
d_{95}\le5\sim8\text{mm}
$$

注意：这里只做几何可达性粗筛，不检查 branch consistency。

## 7. 输出

```text
02_ellipse_family_search/coarse_candidates.csv
02_ellipse_family_search/refined_candidates.csv
02_ellipse_family_search/top_candidates_by_radius.csv
02_ellipse_family_search/candidate_geometry_plots/
```

---

# 六、Phase 3：完整 (\beta_6) 逐点 IK 可达性验证

## 1. 目的

回答：

> 候选椭圆的每个点在完整 6 维 (\beta) 范围中是否存在高精度解？

这一阶段暂不要求解连续，只验证 pointwise reachability。

## 2. 每个 target 的初值

对每个轨迹点，生成：

1. reachability pool 最近邻中按 (\beta) 聚类后的 12 个候选；
2. A1/path_d 最近的 4 个候选；
3. 对最近邻 (\beta) 加 8 个局部扰动；
4. 对失败点增加 2–4 个全局 PSO 或随机初值。

## 3. 逐点目标函数

$$
J_{\text{point}}(\beta;x^\star)
===============================

\left(
\frac{|FK(\beta)-x^\star|}{2\text{mm}}
\right)^2
+
\lambda_{\text{lim}}P_{\text{limit}}(\beta)
$$

第一轮：

$$
\lambda_{\text{lim}}=0.01
$$

这里不加入 distal-priority，也不加入上一点连续性，因为当前只测试是否可达。

## 4. 求解器

建议按以下层次：

1. bounded nonlinear least squares；
2. numerical Jacobian / damped Gauss–Newton；
3. 对失败点使用 PSO；
4. 最终用 least squares 精修。

保留 residual 不超过 5mm 的所有解，并在 (\beta)-space 聚类。

聚类阈值：

$$
d_{\beta,\mathrm{RMS}}\ge1^\circ
$$

才认为是不同 branch。

每个 target 最多保留 16 个 distinct candidates。

## 5. Coarse gate

先对每条候选轨迹 72 个点运行。

通过条件：

$$
\text{success ratio}=100%
$$

$$
\text{residual p95}\le2\text{mm}
$$

$$
\text{residual max}\le5\text{mm}
$$

若 success ratio 为 (98%\sim100%)，允许对失败点增加初值重试一次。

## 6. Fine gate

通过 coarse gate 的候选改为 360 点重新求解，门槛不变。

## 7. 输出

```text
03_fullbeta_pointwise_ik/<candidate_id>/pointwise_candidates.parquet
03_fullbeta_pointwise_ik/<candidate_id>/pointwise_report.json
03_fullbeta_pointwise_ik/candidate_feasibility_summary.csv
```

## 8. 决策

### 如果所有 (a\ge75)mm 候选都失败

说明在测试的 (\beta) 范围和轨迹比例下，尚未找到全局逐点可达的大椭圆。此时：

1. 扩大中心/相位搜索；
2. 尝试 relaxed proximal domain；
3. 将半径降为 62.5mm；
4. 不进入 branch linking。

### 如果存在逐点可达候选

进入 Phase 4。

---

# 七、Phase 4：闭环连续 branch 搜索

## 1. 目的

逐点可达不等于存在一条连续关节路径。这里需要找到：

$$
\beta_0,\beta_1,\ldots,\beta_{N-1}
$$

并满足闭环。

## 2. 构造 cyclic layered graph

第 (i) 个轨迹点对应一个 layer，每个 IK candidate 是一个节点：

$$
v_{i,k}=\beta_{i,k}
$$

## 3. 单点代价

$$
C_i(k)
======

\left(
\frac{e_{i,k}}{2\text{mm}}
\right)^2
+
\lambda_HH(\beta_{i,k})
+
\lambda_\kappa P_\kappa(\beta_{i,k})
$$

其中 canonical posture penalty：

$$
H(\beta)
========

\frac{
4|\beta_{1:2}|^2
+
2|\beta_{3:4}|^2
+
|\beta_{5:6}|^2
}{
\sum_j\beta_{j,\max}^2
}
$$

这表达了：

$$
\text{第一段代价}>\text{第二段代价}>\text{第三段代价}
$$

但它只是软偏好。

## 4. 相邻边代价

$$
C_{i\to i+1}(k,l)
=================

\left(
\frac{
|\beta_{i+1,l}-\beta_{i,k}|_{\mathrm{RMS}}
}{
1^\circ
}
\right)^2
$$

coarse graph 的硬阈值：

$$
|\Delta\beta|_{\mathrm{RMS}}\le3^\circ
$$

fine 360 点 graph 的硬阈值：

$$
|\Delta\beta|_{\mathrm{RMS}}\le1.5^\circ
$$

加入最后一点到第一点的闭环边。

## 5. 求解

对每个可能起点候选枚举一次，使用 Viterbi/动态规划找到最低成本路径，并加入 final-to-start closure cost。

这一步与 IKLink 的“每 waypoint 多解 + 动态规划连接”思路相同。([arXiv][1])

## 6. 初步 branch gate

要求：

$$
\text{reconfiguration count}=0
$$

$$
\text{seam beta RMS}\le1^\circ
$$

$$
\Delta\beta_{\mathrm{RMS,p95}}\le1^\circ
$$

$$
\Delta\beta_{\mathrm{RMS,max}}\le2^\circ
$$

如果不存在零 reconfiguration 的闭环路径，该椭圆不适合作为静态 single-branch 数据集，应换候选。

## 7. 输出

```text
04_cyclic_branch_linking/<candidate_id>/linked_branch.parquet
04_cyclic_branch_linking/<candidate_id>/linking_report.json
04_cyclic_branch_linking/branch_candidate_summary.csv
```

---

# 八、Phase 5：整条 centerline 联合优化

## 1. 目的

图连接给出离散初值后，用 trajectory-level optimization 同时降低 FK residual、速度和加速度。

## 2. 总目标函数

采用 cyclic index：

$$
J_{\text{traj}}
===============

\sum_i
\left(
\frac{|FK(\beta_i)-x_i^\star|}{2\text{mm}}
\right)^2
+
\lambda_v
\sum_i
\left(
\frac{|\beta_{i+1}-\beta_i|_{\mathrm{RMS}}}{1^\circ}
\right)^2
$$

$$
+
\lambda_a
\sum_i
\left(
\frac{|\beta_{i+1}-2\beta_i+\beta_{i-1}|*{\mathrm{RMS}}}{0.5^\circ}
\right)^2
+
\lambda_H\sum_iH(\beta_i)
+
\lambda*\kappa\sum_iP_\kappa(\beta_i)
$$

候选权重：

$$
\lambda_v\in{0.5,2,10}
$$

$$
\lambda_a\in{0,0.5,2}
$$

$$
\lambda_H\in{0.05,0.2,1}
$$

## 3. 分阶段优化

### Stage A

高 tracking 权重，先将残差降到门槛。

### Stage B

固定较低残差，增加速度和平滑项。

### Stage C

加入闭环 seam、canonical posture 和 conditioning 代价。

TORM 的核心思想也是针对给定 task-space path 联合优化整条 redundant joint trajectory，而不是独立求解每个点。([arXiv][3])

## 4. centerline 最终 gate

Pilot：

$$
\text{residual p95}\le2\text{mm}
$$

$$
\text{residual max}\le5\text{mm}
$$

$$
\Delta\beta_{\mathrm{p95}}\le0.75^\circ
$$

$$
\Delta^2\beta_{\mathrm{p95}}\le0.25^\circ
$$

$$
\text{seam}\le0.75^\circ
$$

conditioning：

$$
\sigma_{3,\mathrm{p05}}\ge0.0015\text{m}
$$

$$
\kappa_{\mathrm{p95}}\le150
$$

强通过：

$$
\text{residual p95}\le1\text{mm}
$$

$$
\kappa_{\mathrm{p95}}\le100
$$

## 5. 输出

```text
05_centerline_trajectory_optimization/<candidate_id>/optimized_centerline.parquet
05_centerline_trajectory_optimization/<candidate_id>/optimization_report.json
05_centerline_trajectory_optimization/centerline_final_summary.csv
```

只有通过本阶段的轨迹才能进入 tube generation。

---

# 九、Phase 6：局部 atlas 与 tube 数据生成

## 1. 原理

一个固定全局 (u)-chart 失败，不代表不能沿成功 centerline 建立多个局部 chart。

每张 chart 保存：

$$
(x_c,\beta_c,J_c,J_{W,c}^{#})
$$

其中 (J_c) 是：

$$
J_c=\frac{\partial FK}{\partial\beta}\in\mathbb R^{3\times6}
$$

## 2. Weighted damped pseudoinverse

设置 proximal penalty：

$$
W=\operatorname{diag}(4,4,2,2,1,1)
$$

$$
J_W^#
=====

W^{-1}J^\top
\left(
JW^{-1}J^\top+\mu^2I
\right)^{-1}
$$

这使小任务位移优先由 distal variables 承担。

## 3. Tube 几何

对 centerline 计算切向和两个平滑法向：

$$
\tau_i,n_{1,i},n_{2,i}
$$

先生成小 tube：

$$
\delta_1,\delta_2
\in
{-5,-2.5,0,2.5,5}\text{mm}
$$

目标：

$$
x_{i,j}^\star
=============

x_i
+
\delta_1n_{1,i}
+
\delta_2n_{2,i}
$$

360 个角度，每角度 25 个点：

$$
N_{\text{tube}}=9000
$$

不需要额外 tangential offset，因为 360 个角度已经覆盖切向。

## 4. Predictor

$$
\beta_{\text{pred}}
===================

\beta_i
+
J_W^#
\left(
x_{i,j}^\star-x_i
\right)
$$

## 5. Corrector

$$
J_{\text{correct}}
==================

\left(
\frac{|FK(\beta)-x_{i,j}^\star|}{1\text{mm}}
\right)^2
+
\lambda_b|\beta-\beta_{\text{pred}}|_W^2
+
\lambda_HH(\beta)
$$

使用相邻 angle 和相邻 tube grid 点进行 continuation，禁止每个 tube 点完全独立随机求解。

## 6. Chart 创建规则

以下任一发生时新建 chart：

1. 距离上一个 chart anchor 超过 (10^\circ) 轨迹角；
2. (\kappa>80)；
3. predictor residual 超过 2mm；
4. corrector 迭代次数显著增加。

相邻 chart 至少重叠 (10^\circ\sim20^\circ)。

## 7. Chart overlap gate

在 overlap 中要求：

$$
|\beta^{(A)}-\beta^{(B)}|_{\mathrm{RMS}}
\le0.5^\circ
$$

否则两张 chart 属于不同 branch，不能混入同一静态数据集。

## 8. Tube gate

| 指标                     |          Pilot |
| ---------------------- | -------------: |
| target success ratio   |       (\ge99%) |
| residual p95           |     (\le1.5)mm |
| residual max           |       (\le3)mm |
| tube10 beta RMS p95    |   (\le1^\circ) |
| chart overlap beta gap | (\le0.5^\circ) |
| multi-branch ratio     |              0 |
| normal grid coverage   |       (\ge95%) |

小 tube 通过后，测试：

$$
\delta_1,\delta_2\in[-10,10]\text{mm}
$$

## 9. 输出

```text
06_local_atlas_tube_generation/atlas_manifest.csv
06_local_atlas_tube_generation/chart_parameters.parquet
06_local_atlas_tube_generation/tube_small.parquet
06_local_atlas_tube_generation/tube_full.parquet
06_local_atlas_tube_generation/tube_quality_report.json
```

---

# 十、Phase 7：扩展成多轨迹 canonical 数据集

单条椭圆 tube 只适合作为展示专用数据。为了提高泛化，应从成功轨迹周围生成 trajectory family。

## 1. 轨迹扰动

对于主轨迹 ((c,a,\phi_y,\phi_z))，测试：

### 半径

$$
a'\in{0.9a,a,1.1a}
$$

### 中心

沿三个正交方向偏移：

$$
\Delta c\in{-10,0,10}\text{mm}
$$

### 相位

$$
\Delta\phi_y,\Delta\phi_z
\in
{-7.5^\circ,0,7.5^\circ}
$$

不是所有组合都保留。每条扰动轨迹必须重新通过：

1. full-(\beta) pointwise gate；
2. cyclic branch gate；
3. centerline optimization gate。

## 2. 目标规模

Pilot：

* 3 条成功轨迹；
* 每条 9000 tube samples；
* 总计约 27k。

正式：

* 10–20 条成功轨迹；
* 总计 90k–180k。

## 3. 混合 global canonical 数据

建议最终训练集：

$$
D_{\text{train}}
================

0.7D_{\text{atlas-tube}}
+
0.3D_{\text{A1-global}}
$$

A1 数据用于保持一定的通用性，但 A1 和 atlas tube 在同一 (xyz) 邻域若出现不同 (\beta) cluster，只保留当前 atlas branch。

## 4. 去重和冲突处理

在 2mm xyz voxel 内聚类 (\beta)。

若同一 voxel 内：

$$
d_{\beta,\mathrm{RMS}}>1^\circ
$$

只保留与 atlas reference branch 最近的 cluster，其他 cluster 不能进入静态回归训练集。

## 5. 输出

```text
07_multitrajectory_dataset/atlas_tube_30k.parquet
07_multitrajectory_dataset/atlas_tube_100k.parquet
07_multitrajectory_dataset/mix_a1_30_tube_70.parquet
07_multitrajectory_dataset/dataset_quality_report.md
```

---

# 十一、Phase 8：模型训练

## 1. 主模型

优先：

$$
xyz\to\beta_6
$$

然后：

$$
\theta_{30}=\Gamma(\beta_6)
$$

模型：

```text
mlp_beta6
mlp_beta6_large
resmlp_beta6
```

## 2. 对照

```text
mlp_theta30_direct
```

## 3. 关于 (xyz\to u)

本轮不应强制使用旧 A1 的 (u=(a,b,\eta))，因为新 branch 可能已脱离 A1 参数化。

只有在所有成功轨迹都能被统一拟合为一个新 (u)-mapping 时，才增加：

```text
mlp_u
```

如果使用多个局部 chart，更合理的备选是：

$$
xyz\to\text{chart id}\to\beta_6
$$

即 chart classifier + local expert。

## 4. 损失

$$
\mathcal L
==========

\mathcal L_\beta
+
\lambda_{FK}\mathcal L_{FK}
$$

$$
\mathcal L_\beta
================

|\hat\beta-\beta|^2
$$

$$
\mathcal L_{FK}
===============

|FK(\Gamma(\hat\beta))-x|^2
$$

测试：

$$
\lambda_{FK}\in{0,0.1,0.5,1}
$$

## 5. Split

必须按完整轨迹划分，不能随机拆同一轨迹的点：

1. `trajectory_iid`：训练/测试使用不同轨迹；
2. `radius_holdout`；
3. `center_holdout`；
4. `phase_holdout`；
5. `tube_offset_holdout`。

## 6. 模型 gate

展示轨迹：

$$
EE_{95}\le5\text{mm}
$$

$$
EE_{\max}\le10\text{mm}
$$

$$
\text{axis p95 max}\le5\text{mm}
$$

$$
\Delta^2\beta\text{ 无尖峰}
$$

---

# 十二、最终决策树

## 情况 A：没有任何 (a\ge75)mm 椭圆通过 full-(\beta) pointwise gate

结论：

> 在当前角度范围和 1:1:1.5 轨迹定义下，没有找到该尺度的逐点可达椭圆。

下一步：

1. 扩展中心/相位搜索；
2. 测试 relaxed proximal bounds；
3. 降到 (a=62.5)mm；
4. 不生成 tube。

## 情况 B：逐点可达，但不存在连续闭环 branch

结论：

> 椭圆逐点可达，但需要构型重配置，不能由单值静态模型连续跟踪。

处理：

1. 换另一候选中心；
2. 换相位/方向；
3. 允许 IKLink-style 最少 reconfiguration 仅作为诊断；
4. 不作为论文展示主轨迹。

## 情况 C：centerline 连续，但 tube 失败

结论：

> 轨迹线可达，但周围局部逆映射条件差或 branch 太薄。

处理：

1. 缩小 tube；
2. 换更好-conditioned centerline；
3. 增加局部 chart；
4. 不进入模型训练。

## 情况 D：tube 通过，但单一 MLP 失败

结论：

> 数据已可学，但跨 chart 的函数复杂度过高。

处理：

1. chart classifier + expert；
2. 增加 FK loss；
3. stateful ((\beta_{t-1},x_t)\to\Delta\beta_t) 作为轨迹模型对照。

## 情况 E：单条轨迹成功，holdout 轨迹失败

处理：

1. 增加 center/radius/phase family；
2. 从单 tube 扩展到 atlas-tube dataset；
3. 暂不声称全工作空间 IK。

---

# 十三、下一轮最小 Pilot

为了避免一次性投入过大，Codex 先执行三个 pilot。

## Pilot 1：全局可达椭圆搜索

* reachability Sobol pool：1M；
* centers：512；
* phases：30° 网格；
* radii：50、75、100mm；
* top 20 每半径进入 full-(\beta) IK。

成功标准：

至少找到一条：

$$
a\ge75\text{mm}
$$

且：

$$
\text{pointwise residual p95}\le2\text{mm}
$$

## Pilot 2：连续闭环 branch

对 Pilot 1 top 5：

* 72 点候选生成；
* graph linking；
* 360 点 refine；
* cyclic trajectory optimization。

成功标准：

$$
\text{residual p95}\le2\text{mm}
$$

$$
\text{seam}\le1^\circ
$$

$$
\text{zero reconfiguration}
$$

## Pilot 3：(\pm5)mm local atlas tube

对最优 1–2 条 centerline：

* 360 angles；
* 5×5 normal grid；
* 9000 points/trajectory。

成功标准：

$$
\text{tube success}\ge99%
$$

$$
\text{residual p95}\le1.5\text{mm}
$$

$$
\text{tube10 beta p95}\le1^\circ
$$

只有 Pilot 3 通过后，才启动模型训练。

---

# 十四、最终研究方向的变化

上一轮的核心问题是：

> 如何把指定椭圆塞进 A1 canonical shell？

现在事实表明，这三个指定椭圆并不属于 A1 shell。下一轮应改成：

> **在完整构型空间中主动搜索一条较大、全局逐点可达、存在连续闭环 branch、并具有可扩展局部 tube 的真实椭圆。**

这比继续修改 A1 的 (s_1,s_2) 更合理，也比先训练更大模型更有效。

最终的数据策略将不再是单一固定公式：

$$
u\to\beta
$$

而是：

$$
\text{global reachability search}
\to
\text{continuous branch lift}
\to
\text{local chart atlas}
\to
\text{canonical tube dataset}
$$

这套流程既保留 deterministic canonical policy，又不再要求一张 A1 chart 覆盖所有目标。

[1]: https://arxiv.org/abs/2402.16154 "[2402.16154] IKLink: End-Effector Trajectory Tracking with Minimal Reconfigurations"
[2]: https://iwalker.people.clemson.edu/ICRA20_tendons_proceedings.pdf "Mechanics for Tendon Actuated Multisection Continuum Arms"
[3]: https://arxiv.org/abs/1909.12517 "[1909.12517] TORM: Fast and Accurate Trajectory Optimization of Redundant Manipulator given an End-Effector Path"

根据你本轮实验结果，**按 gate 停止是完全正确的，而且这是一条非常有价值的“负结果”**：它已经把“模型拟合失败”和“目标轨迹本身没有位于当前 canonical branch 的连续可达区域”明确区分开了。

但有一个必须谨慎的结论边界：

> 本轮证明的是 E75/E87.5/E100 不在 **A1 这个特定 (u)-chart** 的连续可达像中；它还没有证明这些椭圆在完整 (6) 维 (\beta) 构型空间中不可达。

下一步不应该直接生成 tube，也不应该训练模型，而应转向：

$$
\boxed{\text{全构型可达性判定}\rightarrow\text{真实椭圆搜索}\rightarrow\text{连续 branch lifting}\rightarrow\text{multi-chart canonical atlas}}
$$

---

# 一、本轮结果究竟证明了什么

建议把问题分成五个逐层递进的 gate：

| 层级    | 问题                                     | 本轮状态     |
| ----- | -------------------------------------- | -------- |
| (G_0) | 目标是不是几何上真正的二维闭合椭圆？                     | **通过**   |
| (G_1) | 椭圆每个点在完整 (\beta_6) 空间中是否可达？            | **尚未确定** |
| (G_2) | 是否存在一条连续、闭合的 (\beta(t)) branch 覆盖整条椭圆？ | **尚未确定** |
| (G_3) | 该 branch 周围是否存在有法向厚度的可达 tube？          | **尚未确定** |
| (G_4) | 模型能否从数据学习该 tube 上的 (xyz\to\beta)？      | **不应启动** |

你现在已经确认：

* (G_0) 通过；
* E75 在 A1 chart 内的 centerline residual p95 为 (61.64)mm，说明它不属于 A1 chart 的有效像；
* A1 和 old enriched 对三个椭圆都没有真实 tube support；
* hierarchical pool 虽然更大，但局部 (\beta) RMS p95 约 (12.6^\circ)，不能直接用作单值训练标签。

这与前面实验的核心认识完全一致：canonical layer-field 能把局部 (\theta) 连续性从 hierarchical mixed 数据的约 (9^\circ\sim10^\circ) p95 降到 (0.23^\circ\sim0.42^\circ)，但它只覆盖一张受限壳面，而不是完整工作空间。

---

# 二、目前还存在三个潜在实验问题

## 1. A1 active inverse 失败，不等于完整机器人不可达

A1 规定了：

$$
\beta_1=s_1(\eta)a,\quad
\beta_2=s_1(\eta)b
$$

$$
\beta_3=s_2(\eta)a,\quad
\beta_4=s_2(\eta)b
$$

$$
\beta_5=a,\quad
\beta_6=b
$$

所以 A1 只是完整 (\mathbb{R}^6) 构型空间中的一个三维子流形：

$$
\mathcal{M}_{A1}\subset\mathbb{R}^6
$$

本轮 residual p95 (61.64)mm 表示：

$$
\operatorname{dist}\left(x_{\mathrm{E75}},FK(\mathcal{M}_{A1})\right)
$$

很大，而不是：

$$
\operatorname{dist}\left(x_{\mathrm{E75}},FK(\mathcal{B}_{\beta})\right)
$$

一定很大。

连续体机器人的冗余自运动本身可能形成多个不相交的 self-motion manifolds；同一个末端点可能属于不同 IK branch。相关研究也明确指出，连续体机器人冗余空间中的自运动结构与刚性机器人不同，不能只用一条预设参数化代表全部可达构型。([iwalker.people.clemson.edu][1]) 对冗余机械臂，一个末端位姿甚至可能对应多个彼此分离的 self-motion manifold，每个 manifold 属于不同 IK branch。([arXiv][2])

所以当前最重要的下一步，是用完整 (\beta_6) 空间建立一个**可达性 oracle**。

---

## 2. E87.5 和 E100 还部分越过了当前 (x)-slab 边界

你当前 A1 `jacobian_pass_pool` 的 (x) 范围约为：

$$
x\in[1.000009,\ 1.181194]\text{m}
$$

而三个椭圆的 (x) 范围分别是：

### E75

$$
x\in[1.0896208-0.075,\ 1.0896208+0.075]
$$

即：

$$
x\in[1.01462,\ 1.16462]\text{m}
$$

完全在 slab 内。

### E87.5

$$
x\in[1.0854381-0.0875,\ 1.0854381+0.0875]
$$

即：

$$
x\in[0.99794,\ 1.17294]\text{m}
$$

已经低于 (1.0)m。

### E100

$$
x\in[1.090426-0.1,\ 1.090426+0.1]
$$

即：

$$
x\in[0.99043,\ 1.19043]\text{m}
$$

同样越过 slab 下边界。

如果还需要一个 (\pm5)mm tube，那么 E87.5 和 E100 越界更明显。因此：

* 如果实验仍限制在 (x=1.0\sim1.2)m，搜索候选时必须加入 slab margin；
* 如果不再限制 slab，就必须重新使用 A1/full (\beta) 的非过滤 pool，并在全范围重新计算 Jacobian gate。

这个因素不能解释 E75 的失败，但会系统性恶化 E87.5 和 E100 的 support。

---

## 3. 当前 E75/E87.5/E100 的中心来自旧数据搜索，不一定适合真实椭圆

这些中心最早是依据旧 pool support 或 same-phase 路径筛出来的。same-phase 路径只探索一条空间线，适合线段的中心未必适合二维椭圆。

因此，下一步不能继续固定这些中心，再强行寻找新 branch。正确顺序应该是：

$$
\text{先建立 branch-aware reachability map}
$$

再在这个 map 中搜索：

$$
\text{center}+\text{radius}+\text{phase/orientation}
$$

---

# 三、学术界给出的最重要启示

## 1. 用 Reachability / Capability Map，而不是只看点云范围

Reachability map 通常将 task space 离散成 voxel，并记录每个 voxel 是否可达；Capability map 则进一步记录局部 dexterity、条件数或其他质量指标。研究中常采用 FK 大规模采样建立初始 map，再用 IK 检查或补齐未充分覆盖区域，即 hybrid FK/IK generation。

这与你的问题高度一致：

* hierarchical pool 可以作为低成本 FK 初始 map；
* active inverse 用于验证和补齐；
* 但你的 map 不能只有“reachable/unreachable”，还必须带有 branch 信息和 canonical quality。

## 2. 全局逆映射需要划分为多个局部可逆区域

早期的 global direct IK 学习工作已经指出：直接对多值逆映射做均方误差训练，会得到多个解的平均，而平均值通常不是有效解。其解决思路是根据 configuration-space pre-image 的拓扑结构，将数据划分成若干局部可逆区域，再分别学习逆映射。([NeurIPS Papers][3])

这意味着你的长期方案不应再期待“一条 A1 chart 覆盖整个工作空间”，而应构造：

$$
\mathcal{W}
===========

\bigcup_{k=1}^{K}\mathcal{W}_k
$$

每个 (\mathcal{W}_k) 对应一个局部 canonical chart：

$$
c_k:\mathcal{W}_k\to\mathcal{B}_k
$$

## 3. 轨迹 IK 必须与路径连续性一起求解

冗余机器人的 IK configuration 选择与 path planning 分开处理，可能会选到与当前构型不连通的目标 branch。已有工作因此把 IK configuration selection 与 path search 联合起来，而不是先独立选一个 IK 解再规划。([Robotics Institute Publications][4])

对你的椭圆问题，对应的是：

> 不能只检查每个椭圆点是否存在某个解；必须检查这些解能否组成一条连续、闭合的 (\beta(t)) 曲线。

---

# 四、下一阶段的核心方案：Canonical Capability Atlas

建议不再直接围绕 E75 生成 tube，而先建立一个 **branch-aware canonical capability atlas**。

它包含四层信息：

1. 一个 task-space voxel 是否可达；
2. 这个 voxel 内有多少个 (\beta) branch；
3. 每个 branch 的 canonical cost、Jacobian condition 和 branch margin；
4. 相邻 voxel 中的 branch 是否连续相连。

---

## 1. 定义完整 (\beta_6) 可达性 oracle

对任意目标点 (x)，先求纯可达性：

$$
r^*(x)
======

\min_{\beta\in\mathcal{B}}
|FK(\beta)-x|
$$

这个阶段**不能加入强 distal-priority 惩罚**，否则又会把完整可达性误判为 A1 可达性。

只允许加入：

* 关节范围；
* 硬安全约束；
* 非常轻的数值正则。

得到 pointwise reachable 后，再计算 canonical quality：

$$
J_{\mathrm{can}}(\beta)
=======================

w_1|\beta_{1:2}|^2
+
w_2|\beta_{3:4}|^2
+
w_\kappa P_\kappa(\beta)
+
w_{\mathrm{lim}}P_{\mathrm{lim}}(\beta)
$$

其中：

$$
w_1>w_2>0
$$

也就是说，先问“能否到达”，再问“哪一种姿态最 canonical”。

---

## 2. 候选生成方式

每个 target 使用以下初值：

* hierarchical full pool 中最近的 32–64 个 (\beta)；
* A1/path_a 中最近的 4–8 个 (\beta)；
* 16–32 个 Latin hypercube / Sobol 随机初值；
* 相邻 target 的已知解；
* 必要时 PSO/global search 作为 fallback。

然后在完整 (\beta_6) 空间做 local refinement。

每个目标点保留：

$$
K=8\sim32
$$

个满足：

$$
|FK(\beta)-x|\le2\text{mm}
$$

的候选，并在 (\beta)-space 聚类。

---

## 3. 建立 branch-aware voxel graph

将 task space 按 (5)mm 或 (10)mm voxel 划分。

在每个 voxel (v) 内：

1. 收集所有可行 (\beta)；
2. 在归一化 (\beta_6) 空间聚类；
3. 每个 cluster 形成一个 branch node：

$$
n=(v,b)
$$

相邻 voxel 的两个 branch node 只有满足以下条件才连边：

$$
|\bar{\beta}*{v,b}-\bar{\beta}*{v',b'}|*{\mathrm{RMS}}
\le d*{\beta,\max}
$$

并且：

$$
\sigma_{\min}(J)> \tau_\sigma
$$

$$
\kappa(J)<\kappa_{\max}
$$

推荐初始值：

$$
d_{\beta,\max}=1.5^\circ\sim2^\circ
$$

$$
\tau_\sigma=0.0015\sim0.002\text{m}
$$

$$
\kappa_{\max}=100
$$

这个 graph 的 connected components 就是候选 canonical charts。

---

## 4. 每个 branch node 的 capability score

对 branch node 定义：

$$
Q(v,b)=
\alpha_r r_{\mathrm{FK}}
+
\alpha_\kappa\log(1+\kappa)
+
\alpha_pJ_{\mathrm{prox}}
+
\alpha_mP_{\mathrm{margin}}
+
\alpha_eP_{\mathrm{edge}}
$$

其中：

* (r_{\mathrm{FK}})：FK residual；
* (\kappa)：条件数；
* (J_{\mathrm{prox}})：第一、二段参与代价；
* (P_{\mathrm{margin}})：与其他 branch 距离过近时的惩罚；
* (P_{\mathrm{edge}})：接近 workspace/chart 边界的惩罚。

最终得到的不是普通 reachability map，而是：

$$
\boxed{\text{reachable}+\text{branch label}+\text{canonical quality}}
$$

---

# 五、下一轮可执行实验

## Phase A：先验证 full-(\beta_6) inverse solver

在搜索新椭圆前，必须确认 full-(\beta_6) solver 本身有效。

### A1. Synthetic recovery

从 A1、hierarchical 和随机 (\beta) 中各取 500 个样本：

$$
x_i=FK(\beta_i)
$$

然后只给 (x_i)，运行 full-(\beta_6) inverse。

要求：

$$
\text{residual p95}\le0.5\text{mm}
$$

$$
\text{residual max}\le2\text{mm}
$$

不要求恢复原 (\beta_i)，只要求恢复某个正确解。

### A2. Improvement check

对每个 target：

$$
r_{\mathrm{optimized}}
\le r_{\mathrm{nearest\ pool}}
$$

否则说明优化器、bounds、单位或 FK 接口有问题。

### A3. A1 self-consistency

对 A1 生成的目标，在 A1 (u)-solver 中必须恢复到：

$$
\le0.5\text{mm}
$$

否则当前 (61.6)mm 可能混有 solver scaling 或 bounds 问题。

---

## Phase B：重新诊断当前 E75/E87.5/E100

每条椭圆先取 (72) 个点。

对每个点运行 full-(\beta_6) oracle。

输出：

* best residual；
* feasible candidate count；
* (\beta)-cluster count；
* best canonical cost；
* (\sigma_3,\kappa)。

### Gate B1：pointwise reachable

要求：

$$
r_{\mathrm{p95}}\le2\text{mm}
$$

$$
r_{\max}\le5\text{mm}
$$

### 三种结果

#### 结果 1：full-(\beta_6) residual 仍很大

说明这些中心/半径确实不适合，直接弃用，不做 multi-chart。

#### 结果 2：full-(\beta_6) pointwise 可达，但 A1 不可达

说明问题是 chart mismatch，应寻找新 chart。

#### 结果 3：每点可达，但候选 branch 不连续

说明 pointwise reachable，但不存在单一连续 lifting，需要调整轨迹或 multi-chart。

---

## Phase C：真实椭圆搜索，而不是固定旧中心

### 轨迹参数化

为了保持各轴幅值比例 (1:1:1.5)，同时允许轨迹平面改变，使用：

$$
x(t)=c_x+a\sin t
$$

$$
y(t)=c_y+a\sin(t+\phi_y)
$$

$$
z(t)=c_z+1.5a\sin(t+\phi_z)
$$

其中固定 (\phi_x=0)，搜索：

$$
\phi_y,\phi_z
$$

只保留 rank-2 轨迹。

这样每个轴幅值仍严格是：

$$
a:a:1.5a
$$

但不再强制轨迹平面为：

$$
x-y=\mathrm{const}
$$

当前 (x=y=\sin t) 的轨迹固定在法向量 ((1,-1,0)) 的平面中，这个平面可能恰好与机器人高质量 workspace 不匹配。

### 搜索变量

$$
(c_x,c_y,c_z,a,\phi_y,\phi_z)
$$

### Coarse search

* 半径：(25,37.5,50,62.5,75,87.5,100)mm；
* phase：每 (30^\circ)；
* center：从 capability map 高质量 voxel 中取 500–2000 个；
* 每条轨迹先取 36–72 点。

### Slab 约束

如果仍坚持 (x\in[1.0,1.2])：

$$
c_x-a-r_{\mathrm{tube}}\ge1.0
$$

$$
c_x+a+r_{\mathrm{tube}}\le1.2
$$

其中：

$$
r_{\mathrm{tube}}=5\sim10\text{mm}
$$

### 搜索目标

$$
J_{\mathrm{ellipse}}
====================

-r_a a
+
w_r r_{\mathrm{p95}}
+
w_{\max}r_{\max}
+
w_\kappa\kappa_{\mathrm{p95}}
+
w_bP_{\mathrm{branch}}
+
w_eP_{\mathrm{edge}}
$$

先选出 20–50 条 pointwise reachable 候选，再做连续 branch 检查。

---

## Phase D：循环 branch lifting

对每条候选椭圆的每个角度 (i)，有候选集合：

$$
\mathcal{C}*i=
{\beta*{i,1},\ldots,\beta_{i,K}}
$$

建立分层图，边代价：

$$
E_{i}(k,l)
==========

\lambda_s
|\beta_{i,k}-\beta_{i+1,l}|^2
+
\lambda_c
\left(
J_{\mathrm{can}}(\beta_{i,k})
+
J_{\mathrm{can}}(\beta_{i+1,l})
\right)
$$

必须包括闭环边：

$$
\beta_{N-1}\leftrightarrow\beta_0
$$

求最小代价闭合 cycle。

### Gate D1

$$
\text{FK residual p95}\le2\text{mm}
$$

$$
\text{adjacent beta RMS p95}\le0.5^\circ\sim1^\circ
$$

$$
\text{adjacent beta RMS max}\le2^\circ
$$

$$
\text{seam gap}\le0.5^\circ\sim1^\circ
$$

$$
\kappa_{\mathrm{p95}}\le100
$$

通过这一步，才算找到真正适合生成数据的椭圆。

---

# 六、何时采用 multi-chart

如果一条椭圆：

* 所有点 full-(\beta_6) 可达；
* 但不存在单一闭合 branch；
* 某些角度区间分别存在平滑局部 branch；

则进行 multi-chart 分解。

例如：

$$
[0^\circ,120^\circ]\rightarrow\mathcal{M}_1
$$

$$
[100^\circ,260^\circ]\rightarrow\mathcal{M}_2
$$

$$
[240^\circ,360^\circ]\rightarrow\mathcal{M}_3
$$

chart 必须有重叠区。重叠区要求：

$$
|\beta^{(k)}(x)-\beta^{(k+1)}(x)|
\le1^\circ\sim2^\circ
$$

否则不是平滑 chart transition，而是 branch jump。

## 模型处理

### chart 区域在 (xyz) 中可分

使用：

$$
xyz\to\text{chart id}
$$

再：

$$
(xyz,\text{chart id})\to\beta_6
$$

即 classifier + expert。

### chart 在相同 (xyz) 区域重叠

静态 (xyz\to\beta) 仍然多值，需要：

$$
(\beta_{t-1},x_t)\to\beta_t
$$

或：

$$
(\text{previous chart},x_t)\to\beta_t
$$

不能把多个 chart 数据直接混在一个 MLP 中。

---

# 七、通过 branch lifting 后再生成 tube

找到连续 centerline (\beta_i) 后，对法向 target 扰动：

$$
\delta x
========

\delta_1n_1+\delta_2n_2+\delta_t\tau
$$

初值不应只靠随机 active inverse，而应使用局部 Jacobian：

$$
\delta\beta_0
=============

J^\dagger\delta x
$$

对于冗余系统：

$$
\delta\beta
===========

J^\dagger\delta x
+
(I-J^\dagger J)z
$$

其中 (z) 用来最小化：

$$
J_{\mathrm{prox}}+
J_{\mathrm{limit}}+
J_{\mathrm{continuity}}
$$

这正是冗余 resolution 的标准思想：主任务满足末端位移，null space 用于次级构型目标。任务优先级和不等式约束通常可以写成分层 QP。

然后再用 nonlinear FK residual refinement 修正。

---

# 八、建议的下一轮最小实验矩阵

| Pilot | 目的                                            |                   数据量 | 决策                                |
| ----- | --------------------------------------------- | --------------------: | --------------------------------- |
| P0    | inverse solver synthetic validation           |          1500 targets | 验证工具正确性                           |
| P1    | 当前 E75/E87.5/E100 full-(\beta_6) reachability |   (3\times72) targets | 区分 chart failure / global failure |
| P2    | center + phase + radius coarse search         | 5k–20k ellipses，36点/条 | 找 pointwise reachable 椭圆          |
| P3    | top-20 cyclic branch lifting                  |       360点/条，8–32候选/点 | 找连续闭合 branch                      |
| P4    | 单-chart (\pm5)mm tube                         |        5k–20k targets | 验证 tube 可达性                       |
| P5    | multi-chart atlas pilot                       |            2–4 charts | 判断是否需要 expert/state               |

---

# 九、最关键的决策树

## 情况 A

$$
\min_{\beta_6}|FK(\beta)-x|
$$

对 E75 仍有 (>10)mm residual：

**结论：** 当前中心/相位/半径物理上不合适。
**动作：** 搜索新中心和新 phase，不做 multi-chart。

## 情况 B

完整 (\beta_6) residual (<2)mm，但 A1 residual (61)mm：

**结论：** A1 chart 不覆盖目标。
**动作：** 从 full-(\beta_6) 候选发现新 chart。

## 情况 C

所有点可达，但没有闭合连续 cycle：

**结论：** 轨迹跨越 branch barrier。
**动作：** 调整 center/phase/radius；其次尝试有重叠的 multi-chart。

## 情况 D

有连续 centerline，但 tube 无法生成：

**结论：** 轨迹位于 workspace/chart 边界。
**动作：** 换中心、缩小半径，或搜索 tube-margin 更大的椭圆。

## 情况 E

centerline 和 tube 均通过，但 MLP 失败：

**结论：** 才进入模型问题。
**动作：** 比较 (xyz\to\beta_6)、(xyz\to u)、chart expert 或 stateful model。

---

# 十、对论文“全工作空间求解 IK”的现实修正

当前证据表明，一条全局 continuous canonical inverse 很可能无法覆盖所有 branch 和奇异区域。冗余 IK 的全局逆映射通常需要将 configuration space 划分为多个局部可逆区域，再分别学习。([NeurIPS Papers][3])

因此论文中更稳妥的长期表述是：

> A branch-consistent canonical inverse is constructed over the task-relevant reachable workspace.

而不是立刻声称：

> A single globally continuous inverse exists over the entire workspace.

若最终构建了多个 chart，可以写成：

$$
\mathcal{W}_{\mathrm{reachable}}
================================

\bigcup_k\mathcal{W}_k
$$

并在各 chart 上学习：

$$
c_k:\mathcal{W}_k\to\beta_6
$$

这既符合机器人冗余逆解的拓扑结构，也能保留机器学习实时推理的优势。

---

# 最终判断

本轮实验没有失败；它准确证明了：

1. 当前 E75/E87.5/E100 的选择方式不适用于真实二维椭圆；
2. A1 是高一致性 chart，但不是足够大的全局 chart；
3. 下一步不能继续把任意目标硬塞进 A1；
4. 必须先建立 full-(\beta_6) reachability oracle；
5. 然后在 branch-aware capability atlas 中搜索真正存在连续闭合 lifting 的椭圆；
6. 只有找到这样的 centerline，才值得生成 tube 和训练模型。

最应该交给 Codex 的下一条主任务是：

> **Build a branch-aware reachability/capability atlas from the full (\beta_6) space, then search for the largest true rank-2 ellipse that admits a continuous closed canonical branch with tube margin.**

[4]: https://publications.ri.cmu.edu/storage/publications/pub_files/pub4/bertram_dominik_2006_1/bertram_dominik_2006_1.pdf "An Integrated Approach to Inverse Kinematics and Path Planning for Redundant Manipulators"
