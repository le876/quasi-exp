# BACRA V13 三维路径 Student 泛化评测记录

日期：2026-08-02  
评测实现 fixed point：`413d322c75bd6f97851b0352289e9e49156b367a`  
标准解释器：`/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python`

## 1. 目的与证据边界

本实验不再只测试椭球与平面的二维椭圆截线，而是在 V13 已接受的三维壳体内预注册非共面闭环路径，检查同一个锁定 Student 是否能同时跟踪不同倾角、长短轴设计比例、形状和厚度振幅。

本实验是 simulation-only、known-chart、locked-Student 泛化评测，不构成 deployment claim。模型没有重新训练，路径没有依据预测结果调参或筛选。

## 2. 锁定输入

- Shell 数据集：`runs/bacra_v13_ellipsoidal_shell_atlas_formal_dense_replay1_20260801/04_dense_dataset/A3_shell_dataset.parquet`
- Shell 数据 SHA-256：`528c8a6dd902c6d665a1d28378c30eb19c6969f314a44399c18872e1a1a6f1a8`
- Shell Gate：PASS；中心面覆盖率与壳体体积覆盖率均为 1.0，全表面 half-thickness 为 20 mm。
- Student model lock：`runs/bacra_v13_formal_shell_student_20260801/02_model_lock/model_lock.json`
- 模型 seed：`20260738`、`20260739`、`20260740`
- 自动 chart classifier：关闭；输入显式使用已知的 `chart_00`。

## 3. 三维路径族

预注册 16 条闭环，每条 720 点，共 11,520 个唯一目标点。路径族包含：

| 维度 | 覆盖 |
| --- | --- |
| 形状 | `tilted_ellipse`、`rounded_superellipse`、`peanut`、`spherical_lissajous` |
| 倾角 | 5°–82°，四个 bin |
| 设计轴比 | 0.35–0.90，四个 bin |
| 实际 PCA 轴比 | 0.341–0.826 |
| 法向厚度振幅 | 4、8、12、16 mm |
| 非共面度 `sigma3/sigma1` | 0.060–0.301 |

路径先在单位球面上生成不同形状和姿态，再通过正式椭球的线性映射落到中心面，并沿物理外法向加入周期性 `rho(t)`。因此路径同时使用二维中心面和一维厚度，不是倾斜平面中的二维曲线。所有 `|rho(t)|` 都不超过 16 mm，位于已接受的 ±20 mm 壳体内。

路径目录和 11,520 个目标点在加载 TensorFlow 与模型之前落盘并哈希锁定：

- `path_catalog.parquet` SHA-256：`133e5c13df56c4fdef332ebdd4b476a5a9d8b47eb605aa902dbbfcf2c7e69cb7`
- `target_paths.parquet` SHA-256：`da304e35de77e7439173d64ac723b1f3751362f3eeac269a52948944ec4e0aab`

## 4. Gate

单个 seed × path 需要同时满足：

- FK P95 ≤ 5 mm；
- FK max ≤ 10 mm；
- 相对路径主尺度误差 P95 ≤ 1%；
- 相对路径主尺度误差 max ≤ 2%；
- 所有预测 beta 位于机械边界内。

最终 Gate 沿用此前椭圆泛化实验的聚合口径：

- 总 seed × path 通过率 ≥ 80%；
- 每个非空形状、倾角、轴比和厚度 bin 通过率 ≥ 60%；
- 全体点绝对 P95/max 继续满足 5/10 mm；
- formal shell Gate 与 locked Student Gate 保持 PASS。

## 5. 正式结果

正式输出：`runs/bacra_v13_formal_3d_path_generalization_20260802`

最终 Gate：**PASS**。

### 全体结果

| 指标 | 结果 |
| --- | ---: |
| 唯一目标点 | 11,520 |
| 总预测点（3 seeds） | 34,560 |
| 全体 FK P50 | 0.146 mm |
| 全体 FK P95 | 0.362 mm |
| 全体 FK max | 1.463 mm |
| seed × path 通过 | 45/48 = 93.75% |
| 在三个 seed 上全部通过的路径 | 15/16 = 93.75% |
| beta step P95 | 0.111° |
| beta step max | 0.610° |
| 预测机械边界 | 100% 通过 |

### 分 seed 结果

| Seed | 点级 FK P50 | 点级 FK P95 | 点级 FK max | 路径通过率 |
| ---: | ---: | ---: | ---: | ---: |
| 20260738 | 0.128 mm | 0.249 mm | 1.088 mm | 93.75% |
| 20260739 | 0.151 mm | 0.349 mm | 1.270 mm | 93.75% |
| 20260740 | 0.162 mm | 0.418 mm | 1.463 mm | 93.75% |

### 分组最小通过率

- shape：75%；
- inclination bin：75%；
- design axis-ratio bin：75%；
- thickness amplitude：75%。

所有非空分组都高于预注册的 60% 下限。

## 6. 唯一未通过路径

`path_id=6` 是 57° 倾角、设计轴比 0.75、16 mm 厚度振幅的 `rounded_superellipse`。三个 seed 的绝对误差仍远低于 5/10 mm Gate，但都超过严格的相对 P95 1% 阈值：

| Seed | FK P95 | FK max | 相对 P95 | 相对 max |
| ---: | ---: | ---: | ---: | ---: |
| 20260738 | 0.825 mm | 1.088 mm | 1.142% | 1.506% |
| 20260739 | 0.996 mm | 1.270 mm | 1.378% | 1.758% |
| 20260740 | 1.167 mm | 1.463 mm | 1.615% | 2.025% |

这说明当前 Student 对大厚度振幅、带较尖曲率变化的 superellipse 存在一个可复现的局部相对精度弱点。由于绝对误差仍小，当前证据支持“壳体内广泛三维路径跟踪能力”，但不支持“所有形状均满足 1% 相对误差”。本轮不为消除这个边界项而改路径或重训模型。

## 7. 结论

V13 壳体数据集训练出的单一 known-chart Student 已经从二维椭圆截线泛化到显式使用壳体厚度的非共面三维闭环族：4 种形状、5°–82° 倾角、0.35–0.90 设计轴比和 4–16 mm 法向振幅下，全体点 FK P95 为 0.362 mm，15/16 条路径在三个 seed 上全部通过。

最重要的剩余 finding 是 `path_id=6` 的严格相对误差边界。下一步若要继续提高，不应放宽 Gate；应专门增加或重加权“高曲率 superellipse × 外层厚度”壳体样本，再用同一组 16 条锁定路径做 forward-only 回归验证。

## 8. 验证与产物

- Focused tests：
  `/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python -m pytest -q tests/test_bacra_v13_3d_path_generalization_runner.py tests/test_bacra_v13_ellipse_generalization_runner.py tests/test_bacra_v13_shell_student_runner.py tests/test_bacra_v13_ellipsoidal_shell_runner.py tests/test_ellipsoidal_shell_v13.py`
- 结果 Gate：`runs/bacra_v13_formal_3d_path_generalization_20260802/02_summary/gate.json`
- 路径级指标：`runs/bacra_v13_formal_3d_path_generalization_20260802/01_results/path_metrics.parquet`
- 点级指标：`runs/bacra_v13_formal_3d_path_generalization_20260802/01_results/point_metrics.parquet`
- 跟踪图：`runs/bacra_v13_formal_3d_path_generalization_20260802/03_visualization/tracking_grid.png`
- 路径 P95 图：`runs/bacra_v13_formal_3d_path_generalization_20260802/03_visualization/path_error_p95.png`
