# Hierarchical Beta FK 位姿数据集实验记录

## 背景

当前 fixed-layer 100k 数据集在 `x=1.0..1.2m` 内有大量点，但整体更像一张薄曲面，而不是有厚度的工作空间 patch。此前统计显示，fixed-layer 100k 在 20mm `y,z` cell 内的局部 `x` 厚度 p95 约为 `27mm`，导致带明显 x 方向变化的椭圆轨迹很容易离开数据流形。

本轮按“优先第三关节，其次第二关节，最后第一关节”的原则，生成 FK-only 位姿数据集。这里不做张力 PSO，不做 `rms_rnorm` gate，也不做 tension relabel；验收重点改为空间覆盖、稠密度、均匀性和几何泛化。

## 生成方式

脚本：

- `scripts/generate_hierarchical_beta_fk_dataset.py`

角度网格：

| beta | 范围 | 步长 | 档位 |
| --- | ---: | ---: | ---: |
| `beta1,beta2` | `[-5,5] deg` | `2.5 deg` | `5 x 5` |
| `beta3,beta4` | `[-5,5] deg` | `1.25 deg` | `9 x 9` |
| `beta5,beta6` | `[-15,15] deg` | `0.5 deg` | `61 x 61` |

总 FK 候选数：

`5 * 5 * 9 * 9 * 61 * 61 = 7,535,025`

输出：

- full FK pool: `data/hierarchical_beta_fk_x1p0_1p2_v1/full_fk_pool.parquet`
- x-filtered pool: `data/hierarchical_beta_fk_x1p0_1p2_v1/x1p0_1p2_pool.parquet`
- balanced 100k: `data/hierarchical_beta_fk_x1p0_1p2_v1/balanced_100k.parquet`
- balanced 500k: `data/hierarchical_beta_fk_x1p0_1p2_v1/balanced_500k.parquet`
- generation report: `runs/diagnostics/hierarchical_beta_fk_x1p0_1p2_v1/hierarchical_beta_fk_report.json`
- README: `runs/diagnostics/hierarchical_beta_fk_x1p0_1p2_v1/README.md`
- visualizations: `runs/diagnostics/hierarchical_beta_fk_x1p0_1p2_v1/visualizations/`

## 空间指标

`x=1.0..1.2m` 过滤结果：

| metric | value |
| --- | ---: |
| generated rows | `7,535,025` |
| x-filtered rows | `4,587,630` |
| x-filter ratio | `0.6088` |
| x-bin nonempty ratio | `1.0000` |
| x-bin count CV | `0.3323` |
| 20mm yz-cell x-range p95 | `198.74mm` |
| 10mm 3D voxel count | `150,490` |

balanced 子集：

| dataset | rows | x-bin CV | yz-cell x-range p95 | 10mm voxels | NN p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced 100k | `100,000` | `0.0000` | `190.65mm` | `69,870` | `10.78mm` |
| balanced 500k | `500,000` | `0.1599` | `197.18mm` | `143,978` | `6.13mm` |

结论：

1. 新数据集在 `x=1.0..1.2m` 内不缺样本，过滤后仍有 `458.8万` 点。
2. 局部 `x` 厚度从 fixed-layer 的约 `27mm` 提升到约 `190-199mm`，说明它已经不再是原来的薄曲面。
3. balanced 100k 的 `x` 分布非常均匀，但 NN p95 为 `10.78mm`，作为轨迹支持略稀。
4. balanced 500k 的 NN p95 为 `6.13mm`，更适合作为后续轨迹与模型训练主数据。

## Theta-only 几何模型 smoke

新增脚本：

- `scripts/baselines/run_theta_fk_baseline.py`

这个脚本只训练 `xyz -> theta`，不要求 tension 列，评价指标为 theta 误差和 FK 回代 EE 误差。

本轮先跑 `balanced_100k` 中随机 30k 子集的 sklearn MLP smoke：

- command log: `runs/logs/hierarchical_beta_fk_theta_30k_smoke.log`
- output: `runs/baselines_hierarchical_beta_fk_x1p0_1p2_30k_theta_smoke/`
- model: `mlp=(256,128,64,32)`
- max rows: `30,000`
- splits: `iid,x_slab,radius,beta_block,angular_sector`

结果：

| split | theta MAE deg | theta p95 deg | EE p95 mm | fit s |
| --- | ---: | ---: | ---: | ---: |
| iid | `2.4551` | `4.9079` | `84.90` | `25.6` |
| x_slab | `1.7665` | `2.9840` | `37.33` | `28.5` |
| radius | `1.7410` | `2.9027` | `38.56` | `14.4` |
| beta_block | `4.6063` | `7.8546` | `144.20` | `27.7` |
| angular_sector | `2.5494` | `4.7301` | `85.30` | `24.8` |

100k `mlp_l2=(512,256,128,64)` 训练尝试：

- output: `runs/baselines_hierarchical_beta_fk_x1p0_1p2_100k_theta_mlp_l2/`
- 状态：因 sklearn MLP 训练太慢，CPU 时间约 `75min` 只完成 IID 一个 split，已停止。
- 已保留 IID 指标：`runs/baselines_hierarchical_beta_fk_x1p0_1p2_100k_theta_mlp_l2/iid/mlp_l2/metrics.json`

## 当前结论

1. 数据生成方向是有效的：分层 beta 网格显著增加了 `x=1.0..1.2m` 内的 workspace 厚度。
2. 500k balanced 子集比 100k 更符合轨迹支持需求；100k 作为快速训练集可以保留，但不应作为最终稠密轨迹数据。
3. 当前 30k 小 MLP 的几何拟合还不够好，尤其 `beta_block` 和 `angular_sector` OOD 的 EE p95 偏大。
4. 这说明数据覆盖问题已经明显改善，但模型训练入口需要进一步升级。后续建议优先使用更快的 PyTorch/TF theta-only 训练，或用分区域/分 sector expert，而不是继续用 sklearn MLP 单线程长跑。
5. 张力标签仍应作为第二阶段：先把 `xyz -> theta` 几何拟合压下来，再从 500k balanced 中抽取 20k/100k 做张力 PSO/relabel。
