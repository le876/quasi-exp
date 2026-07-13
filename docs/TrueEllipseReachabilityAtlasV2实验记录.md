# True Ellipse Reachability Atlas V2 实验记录

日期：2026-07-10

## 目的

本轮根据 `docs/GPT5Pro七次反馈_真实椭圆半径扩展与数据生成优化问题.md` 末尾计划执行。核心目标不是继续把原 E75/E87.5/E100 强行塞入 A1 chart，而是从完整 `beta6` 构型空间中寻找真实二维闭合 `sin/sin/cos` 椭圆，并按以下 gate 判断是否可以进入 tube 数据生成和模型训练：

1. full `beta6` reachability pool 中几何支撑足够；
2. 完整 `beta6` 逐点 IK 可达；
3. 存在连续闭环 branch；
4. centerline 通过平滑性 gate；
5. local atlas tube 通过后再训练 `xyz -> beta6 -> theta30` 模型。

本轮仍不加入张力求解。

## 新增实现

新增脚本：

- `scripts/analysis/true_ellipse_atlas_utils.py`
- `scripts/analysis/run_true_ellipse_reachability_atlas_v2.py`

新增测试：

- `tests/test_true_ellipse_atlas_utils.py`
- `tests/test_true_ellipse_reachability_atlas_v2.py`

主输出目录：

```text
runs/true_ellipse_reachability_atlas_v2/
```

已验证：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python -m pytest \
  tests/test_true_ellipse_atlas_utils.py \
  tests/test_true_ellipse_reachability_atlas_v2.py -q
```

结果：`7 passed`。

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python -m py_compile \
  scripts/analysis/true_ellipse_atlas_utils.py \
  scripts/analysis/run_true_ellipse_reachability_atlas_v2.py
```

结果：通过。

## Phase 0-2：全局 reachability pool 与椭圆搜索

命令：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python \
  scripts/analysis/run_true_ellipse_reachability_atlas_v2.py \
  --preset pilot \
  --out-dir runs/true_ellipse_reachability_atlas_v2 \
  --phases phase0,phase1,phase2
```

耗时：约 `75.91s`。

生成了 1M full `beta6` Sobol reachability pool：

```text
runs/true_ellipse_reachability_atlas_v2/01_reachability_pool/reachability_pool_merged.parquet
```

pool 范围：

| source_domain | rows | x_min_m | x_max_m | y_min_m | y_max_m | z_min_m | z_max_m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 1048576 | 0.432576 | 1.214584 | -0.751343 | 0.746462 | -0.710971 | 0.710977 |

椭圆搜索结果：

```text
runs/true_ellipse_reachability_atlas_v2/02_ellipse_family_search/top_candidates_by_radius.csv
```

重要结果：

- `a=50mm` 最好 `NN p95 ~= 3.02mm`；
- `a=75mm` 最好 `NN p95 ~= 3.67mm`；
- `a=100mm` 最好 `NN p95 ~= 7.43mm`，`NN max ~= 7.70mm`。

因此，从几何支撑看，完整 `beta6` pool 中确实存在比旧 A1 chart 更大的真实二维闭合椭圆候选，尤其 `a=75mm` 和部分 `a=100mm` 候选值得进入逐点 IK。

## Phase 3：full-beta 逐点 IK

先对 `a=100mm` top 5 进行逐点 IK，耗时约 `603s`。重算 per-target best residual 后结果：

| candidate_id | success_ratio | residual_p95_mm | residual_max_mm | pointwise_gate_pass |
| --- | ---: | ---: | ---: | --- |
| c0273_a100_py150_pz30 | 0.9167 | 3.2730 | 4.6830 | False |
| c0273_a100_py210_pz330 | 0.9167 | 3.2730 | 4.6830 | False |
| c0426_a100_py180_pz300 | 0.9028 | 3.7540 | 4.8825 | False |
| c0426_a100_py180_pz60 | 0.9028 | 3.7540 | 4.8825 | False |
| c0426_a100_py210_pz330 | 0.9167 | 3.5976 | 4.9917 | False |

结论：当前 top `a=100mm` 椭圆在几何 NN 支撑上接近，但完整逐点 IK 仍有约 8-10% target 不能达到 2mm residual gate。不能进入 branch/tube。

随后对 `a=75mm` top 5 进行逐点 IK，耗时约 `696s`。结果：

| candidate_id | rows | target_rows | success_ratio | residual_p95_mm | residual_max_mm | pointwise_gate_pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| c0156_a75_py120_pz30 | 219 | 72 | 1.0 | 0.000985 | 0.002867 | True |
| c0156_a75_py240_pz330 | 221 | 72 | 1.0 | 0.001015 | 0.002850 | True |
| c0426_a75_py180_pz60 | 244 | 72 | 1.0 | 0.001514 | 0.003005 | True |
| c0426_a75_py180_pz300 | 245 | 72 | 1.0 | 0.001597 | 0.003015 | True |
| c0113_a75_py330_pz270 | 232 | 72 | 1.0 | 0.001275 | 0.001832 | True |

结论：`a=75mm` 真实二维闭合椭圆在完整 `beta6` 空间中逐点可达，而且 residual 极低。这证明上一轮 A1 residual 失败不是机器人全局不可达，而是 A1 chart 不覆盖目标。

## Phase 4-5：闭环连续 branch

命令：

```bash
/mnt/ML_projects/conda_envs/dante_env/bin/python \
  scripts/analysis/run_true_ellipse_reachability_atlas_v2.py \
  --preset pilot \
  --out-dir runs/true_ellipse_reachability_atlas_v2 \
  --phases phase4,phase5
```

严格 branch gate 结果：

```text
runs/true_ellipse_reachability_atlas_v2/04_cyclic_branch_linking/branch_candidate_summary.csv
```

| candidate_id | success | reason |
| --- | --- | --- |
| c0156_a75_py120_pz30 | False | no_closed_path |
| c0156_a75_py240_pz330 | False | no_closed_path |
| c0426_a75_py180_pz60 | False | no_closed_path |
| c0426_a75_py180_pz300 | False | no_closed_path |
| c0113_a75_py330_pz270 | False | no_closed_path |

centerline summary 为空：

```text
runs/true_ellipse_reachability_atlas_v2/05_centerline_trajectory_optimization/centerline_final_summary.csv
```

结论：本轮找到的 `a=75mm` 椭圆虽然逐点可达，但当前候选集合在 `3deg` edge gate 下找不到连续闭环 branch，因此不能进入 tube generation 和模型训练。

## Phase 10：失败诊断

输出：

```text
runs/true_ellipse_reachability_atlas_v2/10_failure_diagnostics/
```

关键文件：

- `best_residual_branch_smoothness.csv`
- `branch_threshold_sweep.csv`
- `failure_diagnosis.md`

best residual path 的平滑性显示：

| candidate_id | delta_beta_p95_deg | delta_beta_max_deg | seam_beta_rms_deg | residual_p95_mm |
| --- | ---: | ---: | ---: | ---: |
| c0156_a75_py120_pz30 | 6.71 | 12.14 | 0.167 | 0.000985 |
| c0156_a75_py240_pz330 | 9.62 | 12.14 | 0.643 | 0.001015 |
| c0426_a75_py180_pz60 | 11.54 | 14.36 | 1.884 | 0.001514 |
| c0426_a75_py180_pz300 | 12.09 | 14.40 | 1.267 | 0.001597 |
| c0113_a75_py330_pz270 | 14.40 | 15.80 | 0.223 | 0.001275 |

这说明如果只取每个 target 的最佳 residual 解，相邻 beta 变化会很大，存在明显 branch jump。

阈值 sweep 进一步显示：

- 严格 `3deg` 下，所有 `a=75mm` 候选均无闭环路径；
- `5deg` 下，`c0156_a75_py120_pz30` 和 `c0156_a75_py240_pz330` 可连上，但 delta beta p95 约 `2.16-2.21deg`，仍高于计划目标 `<=1deg`；
- 这说明问题不是单纯 FK 可达，而是当前 IK 候选不足以形成足够平滑的 canonical lift。

## Phase 6：tube gate

因为没有 centerline 通过，Phase 6 按 gate 停止：

```json
{
  "tube_gate_pass": false,
  "reason": "no_centerline_passed"
}
```

没有启动模型训练。

## 当前结论

本轮实验把问题进一步分清了：

1. 原 E75/E87.5/E100 不在 A1 chart 中，不代表完整机器人不可达。
2. 在完整 `beta6` 空间中，已经找到真实二维闭合 `a=75mm` 椭圆逐点可达，IK residual 极低。
3. 当前失败点转移到连续 branch：逐点 IK 解之间存在 branch jump，严格 `3deg` 闭环 branch gate 失败。
4. 放宽到 `5deg` 时部分 `a=75mm` 候选可连接，但 beta p95 仍约 `2.16deg`，不满足 `<=1deg` 的 canonical 连续性目标。
5. 因此不能生成 tube，也不能训练模型；否则会再次把多 branch 混入静态 `xyz -> beta/theta` 监督数据。

## 下一步建议

下一步不应扩大数据集或训练模型，而应优化 branch lifting：

1. 对最有希望的 `c0156_a75_py120_pz30` 进行更多 IK seed 复验，例如每 target 16-32 个 distinct candidates。
2. 在 Phase 4 中加入更强的 continuity-aware 解搜索，而不是只依赖 per-target least-squares 的随机/NN 初值。
3. 尝试 trajectory-level optimization：从 `5deg` 可连接路径初始化，加入速度/加速度/seam penalty，把 delta beta p95 从约 `2.16deg` 压到 `<=1deg`。
4. 如果仍不能压低，搜索相邻中心/相位，优先选择 threshold sweep 中 `3deg` 已接近可连的候选。
5. 只有某条 centerline 满足 residual 和 beta 平滑 gate 后，才进入 local atlas tube 和 MLP 训练。
