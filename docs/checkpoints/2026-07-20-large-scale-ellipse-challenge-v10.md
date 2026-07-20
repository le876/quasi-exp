# 2026-07-20 大尺度真实椭圆挑战 V10

## 结论

本实验直接挑战真实三维长半轴 `0.5 / 0.75 / 1.0 m`，不再扫描固定 family 的
`75–105 mm` 历史参数区间。

- **0.5 m：精确、平滑中心线成立，但严格数据质量 Gate 未闭合。** T3 与 T4
  都达到亚毫米 FK 残差，且 180 相位平滑/闭环 Gate 全部通过；T3 最小关节余量
  `1.432°`，比 `1.5°` 门槛少 `0.068°`。T4 最小余量 `1.412°`，同时 chart
  overlap gap `2.512° > 0.5°`。因此没有进入 tube，也不能声明完整数据集通过。
- **0.75 m：少数困难相位失败。** T4 的 coarse-screen FK 残差 P95 为
  `0.758 mm`，但最大残差 `16.933 mm` 且触及关节边界。绝大多数相位可拟合，
  尾部不满足 hard gate。
- **1.0 m：当前标准关节域下整体失败。** T4 coarse-screen FK 残差 P95/max 为
  `403.347 / 432.704 mm`，不是局部门槛微调可以解决的失败。
- **没有证据表明 T4 atlas 带来尺度质变。** 0.5 m 的最佳正式中心线由 T3 获得，
  T4 没有改善安全余量，反而增加 chart-overlap 失败。可以声明的是新中心/新平面
  几何使 0.5 m 中心线达到精确拟合；不能把这个提升归因给 T4。

## 冻结协议

- 分支：`codex/trajectory-canonical-teacher-v10`
- 实验代码提交：`946c2e7`
- 协议：`large-scale-ellipse-challenge-v10.3`
- 机器人最大伸展（由加载配置推导并核对）：`1.215498 m`
- 标准关节域：`±5°, ±5°, ±10°, ±10°, ±15°, ±15°`
- FK reachability atlas：131,073 行（131,072 Sobol + 零构型）
- pose search：每尺度 3 restarts，24 相位，joint-margin-aware objective
- screen：T1/T3/T4，24 相位；只用 tracking、solver 和实际越界决定是否升级
- formal upgrade：T3/T4，180 相位；只有中心线正式通过才运行 `3×3`、`±1 mm` tube

`R` 仍按历史 family 的奇异值换算，仅用于 lineage：

| 真实长半轴 | 历史参数 R | 相对旧 R=100 mm 的真实尺度 | a/L |
|---:|---:|---:|---:|
| 0.5 m | 255.963 mm | 2.560× | 0.411 |
| 0.75 m | 383.945 mm | 3.839× | 0.617 |
| 1.0 m | 511.927 mm | 5.119× | 0.823 |

## 工作空间证据

标准域 FK atlas 的轴向样本跨度为：

```text
x: 1.044309 m
y: 1.632485 m
z: 1.576223 m
```

数值优化找到一对相距 `1.809147 m` 的可达构型。这个数值是**工作空间直径的下界
witness**，不是严格上界；不得仅据此判定 2 m 长轴一定物理不可达。1.0 m 挑战的
精确 teacher 大残差才是本轮当前标准域失败的直接实验依据。

## 逐尺度结果

### 真实长半轴 0.5 m

优化 pose：

```text
center = [1.061638, -0.047114, -0.054579] m
max target norm = 1.187896 m
chain reach margin = 27.602 mm
```

180 相位正式中心线：

| 指标 | T3 | T4 | Gate |
|---|---:|---:|---:|
| FK residual P95 | 0.050 mm | 0.091 mm | ≤1 mm |
| FK residual max | 0.338 mm | 0.447 mm | ≤3 mm |
| Δβ RMS P95 | 0.234° | 0.233° | ≤1° |
| Δβ RMS max | 0.253° | 0.248° | ≤2° |
| acceleration P95 | 0.017° | 0.017° | ≤0.5° |
| seam | 0.175° | 0.199° | ≤1° |
| joint margin min | **1.432°** | **1.412°** | ≥1.5° |
| chart overlap P95 | N/A | **2.512°** | ≤0.5° |
| centerline formal | FAIL | FAIL | all checks |

T3 只因 joint margin 少 `0.068°` 失败。这是下一轮最小补证的首选目标，但本轮不得
事后放宽 Gate 改判。T4 同时有 margin 和 chart-overlap 两项失败。

### 真实长半轴 0.75 m

```text
center = [0.877435, 0.003044, -0.045738] m
max target norm = 1.171878 m
T4 residual P95/max = 0.758 / 16.933 mm
T4 joint margin min = 0°
```

由于 coarse screen 的最大残差和实际关节边界失败，没有升级到 formal/tube。

### 真实长半轴 1.0 m

```text
center = [0.657884, -0.029892, 0.003990] m
max target norm = 1.207553 m
atlas nearest P95/max = 120.421 / 146.787 mm
T4 residual P95/max = 403.347 / 432.704 mm
T4 joint margin min = 0°
```

链长的平移无关必要条件虽然没有单独排除 2 m 直径，但标准关节域的实际可达壳层
与完整闭合椭圆不匹配，三种 teacher 均失败。

## 证据等级与限制

- 0.5 m T3/T4 的 180 相位精确 FK、平滑和 margin 是正式协议产物。
- 0.75/1.0 m 是预注册 coarse-screen 失败证据；按协议未执行 tube。
- 三个尺度都没有完整中心线 formal pass，因此没有任何一个尺度运行 tube；这不是遗漏，
  而是 gate-controlled early stop。
- exact V7 T0 artifact 无法重放到新优化中心/平面。T1 只作 continuation 消融；方法归因
  使用更保守的 T3 graph + whole-trajectory comparator，并在报告中显式记录该语义。
- 本轮证明的是准静态虚拟环境中的几何/teacher 结果，不是动力学或真实硬件部署结果。

## 产物与哈希

- 根目录：`runs/trajectory_canonical_teacher_v10/07_large_scale_challenge/`
- 总报告 SHA-256：`84f6047d913869d6165e39c753ad1b15dd6b465a07bae17bb296737394d33c0c`
- manifest SHA-256：`32763d9fb1b5a3a6919545e14c3d93a14d2a1ce2dbdbf6b3ed78df9abf742389`
- workspace report SHA-256：`ae47a458acce583ec1c051227d8f1e8b197c8a474b8c74c7596d843bdbe76898`

## 下一步决策

1. 若目标是“精确中心线”，0.5 m 已获得 2.56× 尺度提升，而且 T3 优于 T4。
2. 若目标是“可生成严格安全数据集”，当前没有尺度通过；最小补证应只针对 0.5 m，
   将 pose objective 从有限 atlas margin 升级为 exact-teacher margin，并预注册保持全部 Gate
   不变。只需提升 `0.068°`，不应重新扫描 100 mm 附近。
3. 0.75 m 应针对失败相位做 workspace-sector/center/orientation 重搜索，而不是增加模型容量。
4. 1.0 m 应进入机器人/基座/关节域联合设计；继续调 T4 权重缺乏证据支持。
