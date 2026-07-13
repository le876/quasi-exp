# True Ellipse xyz -> beta6 Model Training V4 实验总结

本实验使用 V3 选定 canonical branch 的 `±5mm` 法向 tube，训练 `xyz -> beta6 -> theta30 -> FK`。E75 中心线未进入训练集。

## 数据与划分

- 输入数据审计：`True`；rows=`9000`，angles=`360`，offset curves=`25`。
- 筛选 train/validation/test rows：`7200/1440/360`。
- 最终训练 rows：`8640`；中心线泄漏：`0`。
- 轨迹相位保持为 py=`120.0deg`、pz=`30.0deg`。

## 模型筛选

- 最优配置：`mlp_beta6_large_poly_heavy_relu_a1em06`。
- 第一轮未通过后触发 tanh/regularization fallback：`False`。
- 最终 E75 4/5 seed gate：`True`（5/5）。
- E75 EE p95 median：`1.27609mm`；beta p95 median：`0.0213037deg`。
- E75 axis p95 median：`0.986264mm`；seed pass fraction：`1.000`。
- 330–360deg angle-sector holdout EE p95：`3.76146mm`；diagnostic gate：`True`。

## 最大稳定半径

- 正式 strict support-backed amp_xy：`81.25mm`；amp_z：`121.875mm`。
- relaxed support-backed amp_xy：`83.25mm`。
- model-only 外推 amp_xy：`84.25mm`，该值不作为可靠数据支撑结论。
- 代表 seed：`20260712`，按五 seed 中位表现选择。
- 正式最大半径的首要限制因素：`data_support`。

## 结论边界

- 正式半径同时受实际 8640 行训练池 support、4/5 seed 跟踪 gate、周期平滑和 beta bounds 约束。
- 本结果只适用于当前中心、py=120deg、pz=30deg 的局部 canonical tube，不证明全工作空间存在全局单值逆映射。
- 本阶段未训练张力、LGBM 或 direct-theta 模型，也未对模型输出执行 IK refinement。
