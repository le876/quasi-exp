# E75 真实椭圆连续 Branch Lifting V3 实验总结

候选轨迹：`c0156_a75_py120_pz30`。本实验仅构造连续 `beta6` branch 与条件式法向 tube；未训练模型，未计算张力。

## 实验实现与边界

- 扩展 `scripts/analysis/true_ellipse_atlas_utils.py`：多 seed IK、正反 continuation、null-space seeds、soft cyclic DP、周期稀疏 least-squares、可重复性和全半径邻域一致性诊断。
- 新增 `scripts/analysis/run_true_ellipse_branch_lifting_v3.py`：按 `audit/continuation/enrich/link/optimize/robustness/tube/summary` 分阶段执行，并支持独立 worker 子进程和断点续跑。
- 实验只读取 V2 结果并写入独立 V3 目录，不覆盖 V2。
- 本轮固定 E75，不扩大 Sobol pool、不搜索 E100、不训练模型、不计算张力。

## 0. V2 基线审计

- V2 pointwise rows：`219`；angle layers：`72`。
- 每层 distinct candidates min/p50/p95：`1/3.0/4.0`。
- 候选数不超过 2 的层数：`15`；相邻最小距离大于 1.5 deg 的瓶颈层数：`3`。
- 5 deg hard-link baseline 已从原始 pointwise candidates 重构，并与 V2 threshold sweep 数值一致。

## 1. 连续 branch 是否存在

- 结论：`存在`。
- V2 5deg linked delta beta p95: `2.15666 deg`
- Continuation residual p95: `3.08632e-05 mm`
- Continuation delta beta p95: `0.188475 deg`
- Continuation seam: `0.803237 deg`

## 2. Continuation 与候选增密的作用

- Continuation branch gate：`False`。
- Candidate enrichment：`True`。
- Soft cyclic DP branch gate：`False`。
- 判定原则：continuation 已显著修复相邻步进，但原始 72 点路径未满足当前 0.75 deg 接缝硬门槛；最终许可由后续优化、360 点 selected-branch robustness 与 tube gate 给出。

### Candidate density ablation

- `budget_4`：seed budget `4`，distinct candidates min/p50/p95/max = `2/4.0/4.0/4`。
- `budget_16`：seed budget `16`，distinct candidates min/p50/p95/max = `7/10.0/11.0/13`。
- `budget_32`：seed budget `32`，distinct candidates min/p50/p95/max = `20/24.0/26.0/29`。
- `adaptive`：seed budget `64`，distinct candidates min/p50/p95/max = `20/24.5/58.0/58`。
- 自适应扩展触发层数：`17`。
- Soft-DP selected delta beta p95: `0.190425 deg`
- Soft-DP selected delta2 beta p95: `0.245953 deg`
- Soft-DP selected seam: `0.803237 deg`

## 3. 72 点与 360 点 gate

- 72-point residual p95: `0.0463375 mm`
- 72-point delta beta p95: `0.175493 deg`
- 72-point delta2 beta p95: `0.0325741 deg`
- 360-point residual p95: `6.26488e-05 mm`
- 360-point delta beta p95: `0.0379594 deg`
- 360-point delta2 beta p95: `0.00164248 deg`
- 360-point seam: `0.665756 deg`
- 360-point sigma3 p05: `0.223026 m`
- 360-point kappa p95: `28.0842`
- 72 点最终来源：`soft_dp`；优化中间路径选择 stage：`stage_c`。
- 最终 tube 中心线来源：`v2_02_forward`。
- 优化中间路径 seam: `0.0325925 deg`
- `optimizer_success=false` 仅表示达到本轮 `max_nfev`；路径是否可用仍由独立 tracking、smoothness、conditioning 与正反一致性 gate 决定。

## 4. 正反向是否收敛到同一 canonical branch

- Robustness gate：`True`。
- Selected forward/reverse branch difference p95: `0.822468 deg`
- 检测到多个平滑 branch：`True`。
- 不同起点/方向通过率：`0.6428571428571429`。
- Robustness 许可由被选 branch 的正反可复现性、确定性复跑一致性和自身 centerline gate 共同决定；全部 run 通过率仅作为搜索诊断。
- 跨起点 branch difference p95 最大值: `10.477 deg`
- 多个平滑 branch 不允许直接混合标签；正式 tube 只使用通过正反一致性 gate 且经字典序 canonical tie-break 选中的单一 branch。

## 5. Tube 与下一阶段许可

- Tube gate：`True`。
- Tube success ratio: `1`
- Tube residual p95: `9.11061e-05 mm`
- Tube10 beta RMS p95: `0.519174 deg`
- Tube multi-branch ratio: `0`
- Tube full-radius neighbor pair count: `1596914`
- Tube normal-grid coverage: `1`
- 允许进入下一阶段模型训练：`True`。

## 关键产物

- `00_audit/audit_report.json`
- `01_continuation_ablation/best_continuation.parquet`
- `02_candidate_enrichment/candidates_adaptive.parquet`
- `03_soft_cyclic_linking/best_soft_linked.parquet`
- `04_trajectory_optimization/selected_centerline_360.parquet`
- `05_robustness/selected_centerline_360.parquet`
- `06_local_tube/tube_small.parquet`
- `06_local_tube/tube_quality_report.json`
- `07_summary/final_gate_summary.json`

## 停止规则

本次运行遵守 gate：若 360 点 centerline、robustness 或 tube 任一失败，则不会产生可供正式训练使用的 `tube_small.parquet`，也不会启动模型训练。
