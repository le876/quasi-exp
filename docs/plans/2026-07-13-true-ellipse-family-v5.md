# True Ellipse Family Generalization V5 实验计划

## 1. 目标与结论边界

V5 的主目标是把当前 V4 的 `81.25 mm strict support-backed` 半径提升到：

- 主目标：`amp_xy = 87.5 mm`、`amp_z = 131.25 mm`，必须是 strict support-backed；
- stretch 目标：`amp_xy = 100 mm`、`amp_z = 150 mm`，使用相同硬门槛独立挑战；
- 100 mm 的“代码可执行”和“实验严格通过”分开报告，不以实现覆盖代替实验结论；
- 87.5 mm 与 100 mm 不互相替代，任何一项目标失败都保留实际 limiting factor。

本轮继续采用完整 `beta6` canonical branch、法向 tube 和 `xyz -> beta6` 模型，不引入 multi-chart、direct-theta、张力模型、FK loss 或 IK refinement，不修改 V3/V4 的硬 gate。

## 2. 固定不变的硬门槛

### 2.1 Pointwise IK

- 每个角度必须存在 `xyz residual <= 2 mm` 的解；
- 初始 seed budget 为 16，只有少量、连续且 `max residual <= 5 mm` 的失败角度才升级到 32/64；
- seed budget 是最大搜索预算；某个角度取得 `<= 2 mm` 的可达证书后停止该角度的剩余 seed，不改变通过判定；
- 一个半径锚点失败后，同一 family 的更大半径不再计入 connected radius。

### 2.2 360 点 canonical branch

- residual p95/max：`<= 2/5 mm`；
- delta beta p95/max：`<= 1/2 deg`；
- delta2 beta p95：`<= 0.25 deg`；
- seam beta RMS：`<= 0.75 deg`；
- sigma3 p05：`>= 0.0015 m`；
- kappa p95：`<= 150`；
- forward/reverse branch difference p95：`<= 1 deg`；
- 相同初值、方向和算法的确定性复跑必须一致。

### 2.3 5×5 法向 tube

- offsets 固定为 `[-5, -2.5, 0, 2.5, 5] mm` 的 25 个法向组合；
- target success ratio：`>= 0.99`；
- residual p95/max：`<= 1.5/3 mm`；
- 10 mm 邻域 beta RMS p95：`<= 1 deg`；
- multi-branch ratio：`0`；
- normal-grid coverage：`>= 0.95`；
- 不完整的 3×3、缺 offset 或未通过 branch robustness 的 tube 不进入正式数据集。

### 2.4 模型与 strict support

- 模型 gate 沿用 V4：EE p95/max、逐轴 p95、固定偏置、beta p95、周期一阶/二阶平滑、seam 和 beta bounds 全部通过；
- 正式稳定性要求至少 4/5 seeds 通过；
- strict support 只使用训练分区的非中心线样本重算：NN p95/max `<= 5/8 mm`，15 mm 邻域 count p10 `>= 32`；
- 87.5 mm 与 100 mm 的整个 `family@radius` tube 从训练中隔离，不允许中心线、offset 或同半径其他 family 泄漏；
- 主目标必须同时满足：robust trajectory 已物化、训练池 strict support、87.5 mm whole-radius holdout 4/5 seed model gate。

## 3. 实验分阶段执行

### A. V2/V3/V4 固化与输入审计

- 固化 checkpoint：`740d8c0`；
- 只读使用 V2 1M reachability pool、V2 E100 候选、V3 selected centerline 与 V3 9000-row tube；
- V5 写入独立目录，不覆盖 V2/V3/V4 结果。

### B. 固定 family 的多半径搜索

- seed family：V3 selected geometry，加两个去重的 E100 center；
- 每个 seed 生成 2048 个 bounded Sobol perturbations；
- 只改变一次 center 和 phase，随后在所有半径共享同一 geometry；
- 半径锚点：`75, 80, 82.5, 85, 87.5, 90, 92.5, 95, 97.5, 100 mm`；
- 主排名按 75→87.5 的最坏支持排序，stretch 排名按 75→100 的最坏支持排序；
- 支持指标从原始 reachability pool 精确重算，不继承旧候选表的指标。

### C. Pointwise IK 与半径 continuation

- 5 条代表性 family：V3 baseline、主排名 top、stretch 排名 top；
- 每个 family 从小到大推进半径；
- pointwise 角度任务使用独立 subprocess 真并行并逐 chunk checkpoint；
- 后续半径的 branch 初值来自上一半径已通过的 360 点 canonical branch，保持半径方向的 branch 连续性。

### D. Branch robustness 与 tube

- 搜索排名第一且能够覆盖 100 mm 的 family 优先，因为它同时覆盖主目标和 stretch；
- branch 正式验证全部 5 条 pointwise 代表 family：stretch 第一/第二候选、主排名第一候选、V3 几何的主排名扰动候选，以及原始 V3 baseline；避免候选预算在主目标、stretch 与基线之间产生覆盖漏检；
- forward、reverse 和确定性重复运行使用独立进程并行；
- 正反一致性失败时立即停止该 family 的后续优化；
- continuation 已满足全部 centerline gate 时不再执行冗余周期优化；
- 周期优化在第一个完全通过的 stage 停止；
- 每个通过的半径生成完整 360×25 法向 tube，并进行跨 offset 邻域一致性检查。

### E. 多轨迹数据集与泄漏审计

- 每条正式 tube 的 `trajectory_id = family@radius`；
- 所有通过 tube 的跨-family union 只用于 2 mm Cartesian voxel 冲突诊断，不把不同候选 family 的冲突标签直接混入监督训练；
- 正式训练集从单条固定 family 中确定性选择：内部 branch-conflict gate 通过、至少 3 条完整轨迹/3 个半径，并依次优先覆盖 87.5 mm、100 mm、最大连续半径和更多半径；多分支阈值固定为 3 deg；
- 至少 3 条完整轨迹、3 个不同半径且 branch-conflict gate 通过，才生成正式训练数据集；
- `sample_id`、angle×offset 键和 trajectory manifest 必须唯一、完整。

### F. 参数优化与 whole-radius 泛化评测

- V4 最优 `large + poly_heavy + ReLU + alpha=1e-6` 作为不可缺失基线；
- 受控网格：large/wide 两种宽度，raw/poly_medium/poly_heavy，ReLU/tanh，alpha `1e-6/1e-4`；
- 85 mm 整条轨迹用于模型选择；
- 87.5 mm 与 100 mm 整条轨迹作为最终 holdout；
- 最终模型使用 5 seeds，radius sweep 同时报告 model、training-only support 和 trajectory materialization。

## 4. 正式输出

- `runs/true_ellipse_family_expansion_v5/`：family search、pointwise、branch、tube、dataset 与阶段总结；
- `runs/true_ellipse_family_training_v5/`：split、training-only support、模型筛选、5-seed checkpoint、radius sweep 与最终结论；
- `docs/TrueEllipseFamilyGeneralizationV5实验记录.md`：正式实验记录；
- 主结论只使用 87.5 mm strict goal；100 mm 永远以独立 stretch 字段报告。

## 5. 失败时的 limiting factor

按最早失败阶段分类，不做模糊归因：

1. `geometric_support`：精确 pool NN 支持不足；
2. `pointwise_ik`：存在角度无法达到 2 mm；
3. `branch`：平滑、闭环、条件数或正反一致性失败；
4. `tube`：法向覆盖、残差或局部多分支失败；
5. `data_support`：训练分区 strict support 失败；
6. `model_generalization`：whole-radius holdout 的 4/5 seed gate 失败；
7. `combined`：support 与模型同时失败。

任何失败都保留失败半径、family、per-angle/per-seed 指标和最后一个通过的 connected radius。

## 6. 执行状态（2026-07-14 已完成）

本计划已按正式默认协议执行完毕，并在项目声明的 Python 3.11 数值栈中淘汰旧缓存后完整复核：

- formal expansion protocol gate：`True`；
- 87.5 mm pointwise：`5/5` family 通过；100 mm pointwise：`2/5` family 通过；
- 全部 5 条代表 family 已完成 360 点 branch 验证；87.5/100 mm 均在 branch 阶段失败；
- `v3_selected_s1008` 的 75/80/82.5/85 mm branch 与完整 360×25 tube 连续通过，trajectory-only 上限为 `85 mm`；
- 正式单-family 数据集为 36,000 行、4 个半径，formal dataset gate：`True`；
- formal family coverage gate：`True`，pointwise 实选 5 个唯一 family，branch 选中并完整执行同一组 5 个唯一 family；
- formal training protocol gate：`True`，固定 85 mm validation、完整 24-config 网格、5 个唯一 seed 与 V4 baseline；formal training audit 的 36 项检查只因 `primary_radius_materialized=false` 阻断，24-config/5-seed 训练未启动；
- 最终判定：87.5 mm strict=`False`，100 mm strict=`False`，V4 `81.25 mm` strict checkpoint 保持不变。

实现同时加入 expansion/training 正式协议 gate、实际 5-family 覆盖 gate、全阶段输入/策略指纹、训练半径泄漏防护，以及 formal dataset parquet/manifest/robot config/pointwise/branch 证据的 path+SHA-256+360×25 完整性绑定；缩减 smoke/pilot、少于 5 条 family、截断 screen、替换 validation 或不完整训练网格可以用于诊断，但不能形成 strict 阳性。完整数值和产物指纹见 `docs/TrueEllipseFamilyGeneralizationV5实验记录.md` 与 `docs/checkpoints/2026-07-14-true-ellipse-v5.md`。
