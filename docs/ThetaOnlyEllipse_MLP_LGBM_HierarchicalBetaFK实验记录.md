# Theta-only MLP/LGBM 椭圆轨迹实验记录

日期：2026-07-05

## 实验目的

在 `data/hierarchical_beta_fk_x1p0_1p2_v1` 这批层级 beta FK-only 数据上训练纯几何逆解模型：

```text
xyz -> theta -> FK -> achieved xyz
```

然后搜索固定比例椭圆：

```text
X = center_x + a * sin(t)
Y = center_y + a * sin(t)
Z = center_z + 1.5a * cos(t)
```

这里 `a:a:1.5a` 对应用户给定的 `200:200:300` 比例，只允许整体缩放。

## 代码改动

- `scripts/baselines/run_theta_fk_baseline.py`
  - 新增 theta-only `lgbm` 模型。
  - 新增统一 `theta_only` model package 保存格式。
  - 支持 `--n-jobs`、`--lgbm-n-estimators`、`--lgbm-learning-rate`、`--lgbm-num-leaves`。
  - 继续不依赖张力列，适配 FK-only 数据。
- `scripts/analysis/run_theta_only_ellipse_trajectory_benchmark.py`
  - 新增 theta-only 椭圆轨迹 benchmark。
  - 先用数据集最近邻检查 workspace 支撑，再评估模型轨迹。
  - 输出 3D 图、XY/XZ/YZ 投影、轴向误差曲线、CSV、JSON、README。
  - 不再依赖 `pandas.to_markdown()` 的 `tabulate` 可选包。
- 测试：
  - `tests/test_theta_fk_baseline.py`
  - `tests/test_theta_only_ellipse_trajectory_benchmark.py`

## 训练结果

### 100k MLP

输出目录：

```text
runs/baselines_hierarchical_beta_fk_x1p0_1p2_100k_theta_mlp_v1
```

| split | theta MAE deg | theta p95 deg | EE p95 mm | fit s |
| --- | ---: | ---: | ---: | ---: |
| iid | 2.405 | 4.722 | 82.929 | 103.4 |
| x_slab | 1.708 | 2.884 | 42.975 | 68.9 |
| radius | 1.615 | 2.672 | 39.160 | 57.0 |

### 100k LGBM, 50 trees

输出目录：

```text
runs/baselines_hierarchical_beta_fk_x1p0_1p2_100k_theta_lgbm_v1
```

| split | theta MAE deg | theta p95 deg | EE p95 mm | fit s |
| --- | ---: | ---: | ---: | ---: |
| iid | 2.446 | 4.655 | 90.881 | 2.43 |
| x_slab | 1.994 | 3.293 | 35.166 | 2.22 |
| radius | 1.741 | 2.689 | 46.502 | 2.27 |

重要性能结论：

- LGBM 使用 `--n-jobs -1` 会非常慢，原因是多输出 head 与 LightGBM 内部线程竞争。
- 固定 `--n-jobs 4` 后，100k 三个 split 训练只需要十几秒。

### 500k LGBM, 50 trees

输出目录：

```text
runs/baselines_hierarchical_beta_fk_x1p0_1p2_500k_theta_lgbm_v1
```

| split | theta MAE deg | theta p95 deg | EE p95 mm | fit s |
| --- | ---: | ---: | ---: | ---: |
| iid | 2.488 | 4.808 | 91.949 | 7.73 |
| x_slab | 2.268 | 3.876 | 34.935 | 7.14 |
| radius | 1.822 | 2.750 | 43.807 | 7.59 |

### 500k LGBM, 300 trees

输出目录：

```text
runs/baselines_hierarchical_beta_fk_x1p0_1p2_500k_theta_lgbm300_v1
```

| split | theta MAE deg | theta p95 deg | EE p95 mm | fit s |
| --- | ---: | ---: | ---: | ---: |
| iid | 2.453 | 4.809 | 89.899 | 24.72 |
| x_slab | 2.257 | 3.903 | 49.151 | 23.24 |
| radius | 1.802 | 2.800 | 65.101 | 24.25 |

300 trees 没有带来稳定提升；部分 split 的 FK EE 误差反而变差。

## 椭圆轨迹结果

### 100k MLP + LGBM

输出目录：

```text
runs/visualizations/theta_only_ellipse_hierarchical_beta_fk_100k_v1
```

搜索：

- 生成候选：11025
- 通过 workspace 支撑过滤并评估：69
- 评估模型：6 个，包含 `mlp/lgbm x iid/x_slab/radius`

最佳结果：

| selection | model | candidate | amp_xy m | amp_z m | nn p95 mm | EE p95 mm | axis p95 mm | fixed axis bias |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fallback_no_qualified | radius_mlp | ell_cx1p160_cym0p120_cz0p120_a0p010_z0p015 | 0.010 | 0.015 | 6.363 | 34.674 | 33.255 | True |
| lowest_error | radius_mlp | 同上 | 0.010 | 0.015 | 6.363 | 34.674 | 33.255 | True |

结论：100k 里没有找到合格展示椭圆。即使最小 `a=10mm`，仍有约 33mm 的固定 x 轴偏移。

### 500k LGBM, 50 trees

输出目录：

```text
runs/visualizations/theta_only_ellipse_hierarchical_beta_fk_500k_lgbm_v1
```

搜索：

- 生成候选：11025
- 通过 workspace 支撑过滤并评估：108
- 评估模型：3 个，包含 `lgbm x iid/x_slab/radius`

最佳结果：

| selection | model | candidate | amp_xy m | amp_z m | nn p95 mm | EE p95 mm | axis p95 mm | fixed axis bias |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fallback_no_qualified | x_slab_lgbm | ell_cx1p155_cy0p080_czm0p120_a0p010_z0p015 | 0.010 | 0.015 | 3.765 | 43.383 | 43.093 | True |
| lowest_error | x_slab_lgbm | 同上 | 0.010 | 0.015 | 3.765 | 43.383 | 43.093 | True |

结论：500k 的 workspace 支撑更好，最近邻 p95 从 6.36mm 降到 3.77mm，但模型轨迹没有改善，固定 x 偏移仍然存在。

### 500k LGBM, 300 trees

输出目录：

```text
runs/visualizations/theta_only_ellipse_hierarchical_beta_fk_500k_lgbm300_v1
```

最佳结果：

| selection | model | candidate | amp_xy m | amp_z m | nn p95 mm | EE p95 mm | axis p95 mm | fixed axis bias |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fallback_no_qualified | iid_lgbm | ell_cx1p155_cy0p080_czm0p120_a0p010_z0p015 | 0.010 | 0.015 | 3.765 | 43.327 | 41.482 | True |
| lowest_error | iid_lgbm | 同上 | 0.010 | 0.015 | 3.765 | 43.327 | 41.482 | True |

结论：提高 LGBM 容量没有解决椭圆轨迹的固定 x 偏移。

## 当前结论

1. 新数据覆盖确实改善了 workspace 支撑。
   - 500k 椭圆候选的 `nn_dist_p95_mm` 可以到 3.77mm。
   - 这说明目标椭圆点附近确实有数据，不是单纯 workspace 最近邻缺失。

2. 但 `xyz -> theta` 的单值模型仍然没有学出足够好的逆解。
   - 训练集规模从 100k 到 500k 并没有显著降低椭圆跟踪误差。
   - LGBM 容量从 50 trees 到 300 trees 也没有解决。

3. 当前最主要的轨迹失败模式是固定 x 轴偏移。
   - 最佳椭圆均被标记为 `fixed_any_axis_bias_gt2mm=True`。
   - 轴向最大 p95 基本由 x 误差主导，约 33-43mm。

4. 这批结果不支持把当前椭圆作为最终展示轨迹。
   - 当前脚本已输出 fallback 图，但 README 和 JSON 都明确标记 `fallback_no_qualified`。
   - 不应把这些图当作成功展示，只能作为诊断图。

## 建议下一步

优先不要继续只增加样本量或 LGBM 树数。更有价值的方向是：

1. 训练 `xyz -> beta6 -> theta` 的 beta-first 模型，减少直接 `xyz -> theta` 的多解平均。
2. 增加 branch/sector-aware expert，把不同 workspace 区域或 beta family 分开拟合。
3. 对展示轨迹先做 inverse consistency 校正：用模型输出 theta 后，在局部 theta/beta 邻域做一小步 FK residual refinement。
4. 若只需要展示，应优先回到圆形或更贴合数据流形的轨迹；当前 `x/y` 同相、`z` 正交的 200:200:300 椭圆对单值逆模型仍然困难。
