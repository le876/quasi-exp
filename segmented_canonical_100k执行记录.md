# Segmented Canonical 100k 执行记录

日期：2026-05-29

本文记录 `segmented_canonical` 张力标注方案从 2k/10k 验证扩展到 100k 数据集的最终执行结果。后续检索关键词：

- `segmented_canonical`
- `tension_labeler`
- `100k canonical`
- `standard_beta_sweep_100k_segmented_canonical`
- `张力 MAE`
- `local continuity`

## 1. 背景结论

原始 PSO 张力求解在同一或近似 `theta` 下可能落到不同可行张力解，导致张力标签不唯一、不连续，MLP 学到的是冲突监督信号。为提升一致性，本轮采用分段 deterministic/canonical 张力标注：

- 按绳索终止盘和关节层级远端到近端求解；
- 每段只优化本段主控绳；
- 以 `t_ref_n = 800.0` 作为 canonical tie-break；
- 数据生成统一走 `tension_labeler.method = segmented_canonical`。

核心入口：

- `src/quasi_exp/opt/tension_labeler.py`
- `src/quasi_exp/opt/segmented_tension.py`
- `scripts/generate_dataset.py`
- `scripts/worker_generate_sample.py`

## 2. 正式配置与提交

代码提交：

- commit：`8ca4781`
- message：`Add segmented canonical tension dataset pipeline`

100k 配置：

- `configs/robot_rods_only_standard_100k_segmented_canonical.yaml`

关键配置：

```yaml
tension_labeler:
  method: segmented_canonical

segmented_tension:
  max_nfev: 80
  t_ref_n: 800.0
  feasible_rms_rnorm: 6.0e-2

canonical_tension:
  enabled: true
  method: segmented_canonical
  rms_rnorm_threshold: 6.0e-2
  t_ref_n: 800.0

dataset:
  mode: forward
  rms_rnorm_threshold: 6.0e-2
  shard_rows: 2000
  out_dir: data/standard_beta_sweep_100k_segmented_canonical

parallel:
  workers: 16
```

## 3. 100k 数据集生成结果

生成命令：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python scripts/generate_dataset.py \
  --config configs/robot_rods_only_standard_100k_segmented_canonical.yaml \
  --num-samples 100000 \
  --workers 16 \
  --max-tried 120000
```

输出：

- `data/standard_beta_sweep_100k_segmented_canonical/dataset.parquet`
- `data/standard_beta_sweep_100k_segmented_canonical/dataset_meta.parquet`
- `data/standard_beta_sweep_100k_segmented_canonical/dataset_report.json`

生成统计：

- 行数：`100000`
- `tried = accepted = 100000`
- 总耗时：`2446.62s`，约 `40 分 47 秒`
- 平均速度：`0.02447s/sample`，约 `40.87 samples/s`
- `dataset.parquet`：约 `16M`
- `dataset_meta.parquet`：约 `19M`

## 4. 数据质量验收

诊断输出：

- `runs/diagnostics/standard_beta_sweep_100k_segmented_canonical_dataset_analysis.json`

关键指标：

- `tension_solver_method = segmented_canonical`：`100000/100000`
- `segmented_success_rate = 1.0`
- `rms_rnorm p95 = 0.035676`
- `rms_rnorm max = 0.039803`
- `rms_rnorm mean = 0.029048`
- 张力最大值 p95：`929.976 N`
- 张力最大值 max：`1010.519 N`
- 张力饱和比例：`0`
- 每根绳 `per_cable_sat_ratio = 0`

验收结论：通过。

## 5. 局部连续性验收

诊断输出：

- `runs/diagnostics/standard_beta_sweep_100k_segmented_canonical_local_continuity.json`

结果：`passed = true`

10mm 邻域关键指标：

- pairs：`7,900,000`
- `theta_rms_deg_p95 = 0.25335 deg`
- `tension_mae_n_p50 = 0.1652 N`
- `tension_mae_n_p95 = 5.8468 N`
- `tension_max_abs_n_p95 = 12.9831 N`
- `ratio_rms_gt10deg = 0`

对比 10k 数据集，100k 的 10mm 邻域张力 MAE p95 从约 `44.32 N` 降至约 `5.85 N`，说明密集标准扫描网格 + segmented/canonical 标签显著改善局部平滑性。

## 6. MLP 验收

Classic MLP 输出：

- `runs/baselines_standard_beta_sweep_100k_segmented_canonical_classic_iid/all_metrics.json`

Classic MLP 指标：

- `theta_mae_deg = 0.03205`
- `tension_mae_n = 4.1979 N`
- `ee_pos_p95_mm = 4.9772 mm`
- `ee_pos_rmse_mm = 2.6958 mm`
- `tension_lt0_ratio = 0.000625`
- `tension_gt_tmax_ratio = 0.0`

TF MLP 输出：

- `runs/baselines_standard_beta_sweep_100k_segmented_canonical_tf_iid/all_metrics.json`

TF MLP 指标：

- `theta_mae_deg = 0.03768`
- `tension_mae_n = 3.3610 N`
- `ee_pos_p95_mm = 5.8180 mm`
- `ee_pos_rmse_mm = 3.3054 mm`
- `tension_lt0_ratio = 0.0009667`
- `tension_gt_tmax_ratio = 0.0`

验收结论：Classic MLP 与 TF MLP 均通过。两个模型都有极小比例负张力预测，后续若用于物理控制或仿真输入，应在推理后执行 `clip(tension, 0, Tmax)` 或加入非负输出约束。

## 7. 可复用结论

本轮效果好的主要原因不是网络结构变复杂，而是标签问题被修正：

```text
旧路径：theta -> PSO 随机可行张力解，多解、不连续、标签冲突
新路径：theta -> segmented canonical 张力，确定、连续、物理分段明确
```

因此 MLP 面对的是单值、低噪声、局部连续的监督目标，张力 MAE 降到 `3-4 N` 量级，EE p95 降到 `5-6 mm` 量级。

## 8. 后续可视化建议

建议优先做以下图表：

1. `2k/10k/100k` 指标对比柱状图：张力 MAE、theta MAE、EE p95。
2. 100k 工作空间散点图：`x-y`、`x-z`、`y-z`，按 `max_tension` 或 `rms_rnorm` 着色。
3. 张力分布图：12 根绳的箱线图/小提琴图，检查是否有偏载。
4. 局部连续性图：邻域距离 vs 张力差，突出 10mm 内 p95。
5. 预测误差分布图：Classic/TF 的张力误差、EE 误差直方图或 ECDF。
6. 负张力预测定位图：把预测负张力样本映射回工作空间，检查是否集中在边界。
