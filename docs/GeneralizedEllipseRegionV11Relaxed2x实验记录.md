# Generalized Ellipse Region V11.2：2× Gate 放宽实验记录

日期：2026-07-21

协议：`generalized-ellipse-region-v11.2-relaxed2x`

实现提交：`ccdbcfd`

正式产物：`runs/generalized_ellipse_region_v11_relaxed2x/`

## 1. 实验目的与结论边界

本轮不是重写 V11 的严格结论，而是一个独立的诊断性续跑：保持候选数、phase 网格、family split、训练预算、sealed-test 隔离、关节界和产物哈希等不变量不变，只把数值质量 gate 放宽 2 倍，以观察旧实验越过 `1.5 deg` anchor margin 后会暴露什么下游行为。

放宽规则如下：

- 误差、残差、平滑性、覆盖距离和 condition number 等上限乘 2；
- success rate、joint margin、最小奇异值等下限除以 2；
- `required_seed_passes` 从 4/5 降为 2/5，但仍保留 5 个 seed；
- conflict 的 XYZ 邻域半径从 2 mm 缩到 1 mm，beta gap 阈值从 1 deg 增到 2 deg；
- hash、完整行数、family/trajectory 隔离、sealed-test 不泄漏、零关节越界和固定计算预算不放宽。

正式流水线最终在 Phase 1 停止。2× gate 成功越过了旧 V11 的 margin 层，但完整 traversal/cyclic-cut 审计首次暴露出 9.23--14.07 deg 的逆标签分支差；这远大于放宽后的 1.0 deg repeatability gate，因此没有合法 anchor 可以进入 tube。

## 2. Gate 变换

| Gate | V11 严格值 | V11.2 诊断值 |
|---|---:|---:|
| teacher success rate | 1.0 | 0.5 |
| residual P95 / max | 1 / 3 mm | 2 / 6 mm |
| joint margin min | 1.5 deg | 0.75 deg |
| phase beta RMS P95 / max | 1 / 2 deg | 2 / 4 deg |
| phase acceleration P95 | 0.5 deg | 1.0 deg |
| seam / surface edge / surface Laplacian | 1 deg | 2 deg |
| surface block update max | 0.05 deg | 0.10 deg |
| repeat / reverse-cut P95 | 0.2 / 0.5 deg | 0.4 / 1.0 deg |
| local 5 / 10 mm consistency | 0.5 / 1 deg | 1 / 2 deg |
| sigma-min P05 / kappa P95 | 0.05 / 100 | 0.025 / 200 |
| coverage P95 / max | 3 / 5 mm | 6 / 10 mm |
| student interpolation P95 | 5 mm | 10 mm |
| student near-OOD P95 | 7.5 mm | 15 mm |
| student absolute / relative max | 10 mm / 2% | 20 mm / 4% |
| student per-axis mean | 2 mm | 4 mm |
| required seed passes | 4/5 | 2/5 |

解析后的完整协议位于 `00_protocol/protocol_v11.yaml`，源配置为 `configs/generalized_ellipse_region_v11_relaxed2x.yaml`。

## 3. 阶段轨迹

| Phase | 状态 | 关键证据 |
|---|---|---|
| 0 protocol | PASS | 72 个冻结 family；主 split 14 train / 5 validation / 5 virgin-test；sealed 集与模型选择集不相交 |
| 1 anchor screen | 完成 | 193 proposals；固定选取 41 个 180-phase screen；Top-3 与旧 V11 相同 |
| 1 anchor verify | FAIL | 3 个候选均完成 primary、repeat 和 7 个 traversal/cut 变体；无候选通过完整 gate |
| 2 tube | 未启动 | 正确地被 Phase 1 gate 阻止 |
| 3--8 | 未启动 | 不存在伪造的下游 gate、模型或 `COMPLETED` 标记 |

运行时间：Phase 0 gate 于 09:04:29 写入，Phase 1 gate 于 10:41:16 写入，约 1 小时 37 分钟。正式目录包含 193 个文件，约 21 MiB。

## 4. 180-phase Screen Top-3

| 候选 | margin min | residual P95 | screen gate |
|---|---:|---:|---|
| `A4_A2_178_r0_reverse_c0045` | 1.491282 deg | 0.102668 mm | PASS |
| `A4_A2_143_r1_reverse_c0045` | 1.488005 deg | 0.032932 mm | PASS |
| `A4_A2_134_r0_reverse_c0045` | 1.487825 deg | 0.054238 mm | PASS |

## 5. 720-phase Primary 与不变量审计

三个 primary 在放宽后的 teacher gate 下全部通过：

| 候选 | primary margin | residual P95 / max | phase delta-beta P95 | seam |
|---|---:|---:|---:|---:|
| A2_134 | 1.497833 deg | 0.026188 / 0.036847 mm | 0.063599 deg | 0.042433 deg |
| A2_143 | 1.497257 deg | 0.006562 / 0.058943 mm | 0.052761 deg | 0.012661 deg |
| A2_178 | 1.216731 deg | 0.008922 / 0.269410 mm | 0.078420 deg | 0.050432 deg |

但是完整的不变量结果为：

| 候选 | repeat P95 | 最坏 traversal/cut P95 | 最坏变体 | beta gap > 1 deg 的 phase | 完整 gate |
|---|---:|---:|---|---:|---|
| A2_134 | 0.242327 deg | 12.557874 deg | `forward_cut0135` | 683/720 (94.86%) | FAIL |
| A2_143 | 0.291985 deg | 14.069262 deg | `reverse_cut0000` | 537/720 (74.58%) | FAIL |
| A2_178 | 1.108191 deg | 9.232195 deg | `forward_cut0045` | 650/720 (90.28%) | FAIL |

A2_134 与 A2_143 的 repeat 均通过放宽后的 0.4 deg gate；A2_178 连 repeat 也失败。三者都因最坏 traversal/cut 远超 1.0 deg 而失败。

## 6. 为什么这不是 phase 对齐错误

对每个 primary 与 7 个变体重新比较了冻结 parquet：

- 同一 `phase_idx` 的 target XYZ 最大差为 0 mm；
- 以 XYZ 最近邻重新配对后仍是恒等置换；
- `CanonicalTeacher.solve()` 会使用相同 traversal order 同步重排 target、`initial_beta_path` 和 `neighbor_anchor_path`，然后按原 phase 恢复输出；
- 六个 beta 均在小范围有界域内，不存在角度加减 360 deg 的等价 wrap。

因此大 gap 反映的是同一目标环在不同 continuation 起点/方向下进入不同逆解分支，而不是 phase/cut 索引错位。

最有代表性的证据是 A2_143 的 `reverse_cut0000`：该变体自身 residual max 仅 0.0415 mm、joint margin 1.4986 deg，teacher gate 通过，但相对 primary 的 beta gap P95 为 14.0693 deg。即两个标签在任务空间都准确、都不贴关节界，却不是同一 canonical inverse branch。

## 7. 当前解释

旧 V11 在 primary margin `1.4978 deg < 1.5 deg` 时 fail-fast，未运行这些昂贵的 traversal/cut 变体。因此旧结论“Phase 1 因 margin 停止”仍然正确，但不完整地描述了更深层风险。

本轮说明：简单把 margin 和其他数值 gate 放宽 2 倍，不能得到稳定的 canonical anchor。现阶段方法的主要瓶颈已经从 FK 几何精度转为 inverse-label branch identifiability / traversal invariance。继续生成 tube 会把相同或邻近 XYZ 对应到相差约 10 deg 的 beta 标签，后续静态 `xyz -> beta6` 很可能学习到混合分支或均值解。

## 8. 下一步边界

如果目的是保持正式证据有效，应先修改 anchor/canonical 选择目标，使 traversal/cut 共享同一 branch identity，再重跑 Phase 1；不能把 9--14 deg 事后放宽成通过。

如果目的只是查看“忽略分支不变量后”的下游 tube、family 和 student 表现，则应另建明确命名的 shadow/bypass 协议：保留当前 Phase 1 gate 为 FAIL，单独记录被强制选中的 anchor 和 bypass 原因，并禁止生成正式 `COMPLETED`。这属于比“所有数值 gate 放宽 2 倍”更强的协议变更，不能混入本轮正式目录。
