---
question_id: Q10
question_number: 10
question_confirmed_by_user: true
question_confirmation_summary: "请GPT-5 Pro基于Q10证据判断：当前是否应继续放宽numerical gate继续推进，还是应优先修复branch identity（不变性一致性）问题后重跑；并给出可执行的下一步决策边界。"
date: "2026-07-21"
status: ready-to-send
project: "Generalized Ellipse Region V11"
evidence_cutoff: "2026-07-21 10:41:16 +08:00"
source_snapshot: "worktree /mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11 | branch codex/generalized-ellipse-region-v11 | HEAD c817ecebf98ba6429c78070ae4475a745da217ee | worktree clean"
---

# GPT-5 Pro 第10次交接：放宽Gate与Branch-Identity取舍

## 0. 经用户确认的 GPT-5 Pro 任务

用户已经确认的任务原意：

> 请GPT-5 Pro基于Q10证据判断：当前是否应继续放宽数值 gate 继续实验，还是应优先修复 branch identity（即同一目标在不同 traversal/cut 下保持同一 inverse branch）后重跑；并给出可执行的下一步决策边界。

确认范围：仅在 `runs/generalized_ellipse_region_v11_relaxed2x` 的现有证据范围内比较“继续放宽数值门控”和“改进 branch identity”两条路径，不引入未执行的新实验结论。

## 1. 当前有效事实与最短摘要

### 1.1 证据截止时间

截止到 `2026-07-21 10:41:16 +08:00`（`runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/gate.json` 时间戳对应写入时间）。

### 1.2 当前有效事实

1. 本次为独立诊断实验，协议为 `generalized-ellipse-region-v11.2-relaxed2x`，未覆盖旧 V11 strict protocol（仅用于对比）。
2. Phase 0 通道通过（`runs/generalized_ellipse_region_v11_relaxed2x/00_protocol/gate.json` 中 `gate_pass=true`，并通过 72-family split 与 sealed-test 隔离校验）。
3. Phase 1 的 `01_anchor/gate.json` 显示 `gate_pass=false`，`passing_anchor_count=0`。
4. 三个候选（A2_134/A2_143/A2_178）均在 primary/teacher 数值门控下通过放宽值，但在 `repeat + 7 traversal/cut` 的不变性审计中失败。
5. 关键失败模式为 traversal/cut 版本间 beta 分支差异显著，`reverse/forward` 子变体与 primary 在同一 XYZ 索引对齐下出现高 `phase_beta` 偏差。

### 1.3 关键边界

1. 该结论基于 `runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/verification_ranking.csv` 和 `01_anchor/gate.json` 的门控统计。
2. 未执行新的 branch-identity 一致性改造；当前证据中没有出现“为同一目标在不同 traversal/cut 下共享同一 inverse-branch 起点与 canonical continuation 锚定”的实现或实验。
3. 不能把本轮失败解释为 phase 对齐错误：用户已提供的对齐复核结论说明同一 `phase_idx` 对应 XYZ 差值为 0，且最近邻重排仍为恒等映射。

## 2. 项目目标与本轮边界

### 2.1 用户给定的项目目标

在 V11 区域实验中，判断是否应靠门控放宽“突破 Phase 1”，还是先修复 traversal/cut 下的 branch identity 问题（避免相同目标落入不同逆解标签分支）。

### 2.2 当前实际覆盖范围

- 范围：`/.worktrees/generalized-ellipse-region-v11` 下的 `generalized_ellipse_region_v11_relaxed2x` 配置与其 01_anchor 结果。
- 形式：数值 gate 的 2×放宽对比 + 遍历/cut 一致性失败审计。
- 不覆盖：未执行任何新的 branch-identity 强化机制（如新 anchor canonical 策略、分支锚定规则重写）。

### 2.3 当前证据未覆盖的内容

- 在本轮产物内未见“跨 family 的新 branch 共享策略”实现细节。
- 未给出新一版 gate-relax 到 3×或 4× 的再放宽统计。
- 未给出后续 Tube、Phase3+、student 部分在 bypass 条件下的结果（本协议设计上并未启动）。

## 3. 相对上次材料新增的事实

若这是首次交接，明确写“首次交接，无上次材料”。

| 类型 | 已发生的变化 | 实际证据 | 证据等级 |
|---|---|---|---|
| 实现/配置 | 新增 `generalized_ellipse_region_v11.2-relaxed2x` 变体协议，独立 `output_root` 与门控参数 | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11/configs/generalized_ellipse_region_v11_relaxed2x.yaml` | formal |
| 实验 | 01_anchor 在放宽后完成 193 proposals，最终 3 candidate 无一通过；`passing_anchor_count=0` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/gate.json` | formal |
| 结论修正 | 明确失败主因从 `primary margin` 不足转移到 traversal/cut 下 branch 不一致 | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/gate.json` 与 `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/verification_ranking.csv` | formal |

## 4. 实验方法与复现口径

### 4.1 代码与工作树 fixed point

| 项目 | 值 |
|---|---|
| 项目根 | `/mnt/ML_projects/quasi_exp` |
| 分支 | `codex/generalized-ellipse-region-v11` |
| HEAD | `c817ecebf98ba6429c78070ae4475a745da217ee` |
| worktree | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11` |
| dirty 状态 | 主 worktree 有未跟踪文件；本 worktree 内无未提交变更（`git status` clean） |
| 仅 checkout HEAD 是否足以复现 | 是（需明确引用 `runs/generalized_ellipse_region_v11_relaxed2x` 已有产物） |

### 4.2 方法与参数

| 参数 | 实际值 | 来源 |
|---|---:|---|
| 基础协议 | `generalized_ellipse_region_v11.yaml`（严格） | `configs/generalized_ellipse_region_v11.yaml` |
| 放宽协议 | `generalized_ellipse_region_v11.2-relaxed2x`（上限翻倍，下限减半） | `configs/generalized_ellipse_region_v11_relaxed2x.yaml` |
| 输出目录 | `runs/generalized_ellipse_region_v11_relaxed2x` | `00_protocol/protocol_v11.yaml` + config |
| gate 判定层级 | `teacher_surface`, `repeatability`, `local_consistency`, `conflicts`, `coverage`, `student` | `01_anchor/gate.json` 与 config |

### 4.3 数据生成与划分

| 项目 | 实际口径 |
|---|---|
| 数据来源 | 统一 family split 的 frozen 家族与 phase family 生成产物 | 产物清单见 `00_protocol/gate.json` |
| 生成方式 | 继承 v11 规范流程，仅参数按 2× 放宽重算门控 |
| 样本数 | 193 proposals，41 screen candidates，3 verification candidates | `01_anchor/gate.json` |
| train | 14 |
| validation | 5 |
| test | 5 |
| seed | 5（protocol 固定 seed 配置） | `configs/generalized_ellipse_region_v11.yaml`（seed 段） |
| 泄漏审计 | sealed-test 非重叠通过 `source_fixed_point_frozen` 与 split 校验 | `00_protocol/gate.json` |

### 4.4 baseline、评价和 gate

| 项目 | 实际定义 | 是否正式 |
|---|---|---|
| baseline | V11 strict protocol 的 family/split/预算不变，仅数值门控放宽 | 是（用于诊断，保持 formal lineage 分离） |
| metric | 主要为 margin、residual、repeat/reverse-cut beta RMS P95、local consistency |
| gate | 3 阶段：phase0/pass -> anchor screen -> anchor verify。Phase1 最终 `gate_pass=false` | 是（严格 gate） |

### 4.5 复现入口与产物

```text
python3 .worktrees/generalized-ellipse-region-v11/scripts/analysis/run_generalized_ellipse_region_v11.py --config configs/generalized_ellipse_region_v11_relaxed2x.yaml
```

主要产物：

- `runs/generalized_ellipse_region_v11_relaxed2x/00_protocol/gate.json`
- `runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/gate.json`
- `runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/screen_ranking.csv`
- `runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/verification_ranking.csv`

## 5. 实验结果

| Fact ID | 已观察结果 | 数值 | 证据等级 | 实际附件/JSON key | 适用边界 |
|---|---|---:|---|---|---|
| F1 | Phase0 固定点与 split 校验通过 | `gate_pass=true` | formal | `runs/generalized_ellipse_region_v11_relaxed2x/00_protocol/gate.json` 的 checks | 仅严格 lineage 的冻结与隔离规则 |
| F2 | 01_anchor 最终未通过 Gate | `passing_anchor_count=0` | formal | `runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/gate.json` | 需同时满足 repeatable + reverse-cut + margin 条件 |
| F3 | 3候选 primary 通过放宽后 teacher 数值门控 | A2_134/A2_143/A2_178 均为 pass（primary 阶段） | formal | `verification_ranking.csv` 行 | 仅主候选的 primary 阶段 |
| F4 | traversal/cut 下 beta 一致性失败 | reverse-cut P95 分别 12.56° / 14.07° / 9.23°（A2_134/A2_143/A2_178） | formal | `verification_ranking.csv` | 决定性阻断 phase1 |

### 5.1 通过项

1. Phase0 与 split/seed/inventory 的不变式保持。
2. 3 个候选在 primary、repeat 与部分 teacher 指标下能达到放宽阈值。

### 5.2 失败与反例

1. `selected_margin_meets_protocol_minimum=false` 且 `selected_repeatability_passes=false` 且 `selected_reverse_cut_passes=false`（皆见于 `01_anchor/gate.json`）。
2. `gap > 1°` 的 720-phase 情况占比高，A2_143 达到 `537/720 (74.58%)`、A2_134 `683/720 (94.86%)`、A2_178 `650/720 (90.28%)`。

## 6. 证据等级、失败结果与未知事项

### 6.1 正式事实

- 该轮实验在 formal lineage 内通过了不变式（phase0）但未通过 anchor verify（phase1），不应进入 tube/phase3+。
- 主要阻断源来自 traversal/cut 不变性（branch 分支一致性）而非 residual/joint margin 的几何可解性。

### 6.2 诊断、探索、evidence-only、post-hoc 或历史事实

| 内容 | 等级 | 能支持什么 | 不能支持什么 |
|---|---|---|---|
| “phase_idx 对齐可能是根因” | diagnostic | 可支持：XYZ 重配仍为恒等、并非 index 漂移引发 | 不能支持：此为用户此前的额外核验结论之外的因果必然 |
| A2_143 的 residual 与 margin 同时通过但 beta 与 primary 相差 >10° | formal/evidence-only | 支持：同一任务下存在 branch 分支分歧 | 不能支持：单一机制解释为何会发生 |

### 6.3 已发现的实现或证据缺口

- 尚未补充 branch identity 的具体实现对比（如强制统一 continuation 起点）实验。
- 未有“继续再放宽 gate 到更高倍数”后是否收敛的后续曲线。

### 6.4 尚未知或尚未执行

- 未执行任何新修订版 canonical branch-sharing 或 anchor 锚定策略的正式对比。
- 未执行 shadow-bypass 的完整后续链路（tube/student）。

### 6.5 当前不能声称

- 不能把当前 failure 断言为算法理论上必然不可修复。
- 不能把“再放宽2×之外”作为当前唯一合理决策（缺少该分支证据）。

## 7. 决策所需的客观对照

本节只比较已有方法、协议或结果，不给出额外路线排序。

| 对照对象 | 已执行输入/方法 | 已观察结果 | 证据等级 | 已知限制 |
|---|---|---|---|---|
| V11 strict protocol | `generalized_ellipse_region_v11.yaml` | Phase1 被 margin 1.5° 提前阻断（未进入 traversal/cut） | formal | 未观察到 720-phase 不变性失效 |
| V11.2 relaxed2x protocol | `generalized_ellipse_region_v11_relaxed2x.yaml` | primary 可过，但 repeat/reverse-cut beta 不变性失败 | formal | 本轮只到 phase1，未进入 tube |

## 8. 经用户确认的问题

只保留用户明确确认的问题，不自动追加。

1. 请判断“继续放宽 gate 是否可行”与“是否应优先修复 branch identity 再重跑”，并给出你依据当前证据的建议方向（不基于未执行实验扩展）。

## 9. 经用户确认的回答要求

- 回答必须是二选一主判断：继续放宽到更大倍数，还是先修复 branch identity。
- 回答必须用现有证据支持阻断项（不能给出无依据猜测）。
- 回答必须给出最小下一步执行边界：`何时可重新启动 Phase1/何时应禁止进入 tube`。

## 10. 附件清单与复现信息

GPT-5 Pro 实际可见的是“上传文件名”列；原始 SSH 路径只用于 provenance。

GPT 可见文件数（`QUESTION.md + 实际证据附件`）：`20`（超过 15 个阈值，改为单一 ZIP 上传）。

| 上传文件名 | 原始 SSH 路径 | 大小 | 证据等级 | 支持事实 | 上传理由 |
|---|---|---:|---|---|---|
| `GeneralizedEllipseRegionV11Relaxed2x实验记录.md` | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11/docs/GeneralizedEllipseRegionV11Relaxed2x实验记录.md` | 6,652 B | formal | F1-F4, 实验轨迹 | 现象与阶段轨迹摘要 |
| `generalized_ellipse_region_v11_relaxed2x.yaml` | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11/configs/generalized_ellipse_region_v11_relaxed2x.yaml` | 1,662 B | formal | 配置与放宽规则 | 放宽边界定义 |
| `generalized_ellipse_region_v11.yaml` | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11/configs/generalized_ellipse_region_v11.yaml` | 4,795 B | formal | strict vs relaxed 对照 | 证据对照 |
| `00_protocol_gate.json` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/00_protocol/gate.json` | 860 B | formal | phase0 checks | 通过的基线约束 |
| `01_anchor_gate.json` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/gate.json` | 21,862 B | formal | passing_anchor_count、checks、candidate通过状态 | 决策核心 |
| `screen_ranking.csv` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/screen_ranking.csv` | 7,843 B | formal | Top候选与screen质量 | 关键候选来源 |
| `verification_ranking.csv` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/verification_ranking.csv` | 1,087 B | formal | 3候选在 repeat/reverse-cut 的数值 | 决策核心 |
| `region_audit.py` | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11/src/quasi_exp/teacher/region_audit.py` | 8,765 B | formal | 相关门控/一致性实现口径 | 能核验指标实现是否一致 |
| `region.py` | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11/src/quasi_exp/teacher/region.py` | 23,458 B | formal | teacher solve/surface 计算链路 | 对应输出统计可靠性 |
| `A2_134_primary_centerline.parquet` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/verify/A4_A2_134_r0_reverse_c0045/primary/centerline.parquet` | 499,334 B | `7905b5f887bd7bac904fedc60c5e7eb0a63453268a87f59f89d81a55f32e1e11` | F4 | A2_134 的教师主支轨迹（teacher gate pass），用于和逆变体对比 |
| `A2_134_forward_cut0135_centerline.parquet` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/verify/A4_A2_134_r0_reverse_c0045/forward_cut0135/centerline.parquet` | 498,928 B | `421b15fa558892809f172615718c17dd0a9bb70cfc6242aeb9304f9c7b682270` | F4 | A2_134 `forward_cut0135` 变体；与 primary 的 `phase_beta` 仍对齐目标但偏离 12.56°（P95） |
| `A2_143_primary_centerline.parquet` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/verify/A4_A2_143_r1_reverse_c0045/primary/centerline.parquet` | 499,334 B | `c21a0e2be7af447b807c2549d155bc1db143e814a735552a89c656d51effeb7b` | F4 | A2_143 的教师主支轨迹（teacher gate pass），用于和逆变体对比 |
| `A2_143_reverse_cut0000_centerline.parquet` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/verify/A4_A2_143_r1_reverse_c0045/reverse_cut0000/centerline.parquet` | 499,334 B | `bf3dc460b4c5ee0d660ee860d68de89f949d211e2c1a71dfa5d4713246abfaf0` | F4 | A2_143 的关键逆变体轨迹，最大发散变体 reverse_cut0000 |
| `A2_178_primary_centerline.parquet` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/verify/A4_A2_178_r0_reverse_c0045/primary/centerline.parquet` | 499,334 B | `51fc85acf652d2ecdfccd5267f7ac90db24a4eaa0c4cc9b0f28ce77912a853c7` | F4 | A2_178 的教师主支轨迹（teacher gate pass），用于和逆变体对比 |
| `A2_178_forward_cut0135_centerline.parquet` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/verify/A4_A2_178_r0_reverse_c0045/forward_cut0135/centerline.parquet` | 499,334 B | `68a15ab84e305d612b8545fa90f8e66d3a008ad132c8472fe105c097eec8de29` | F4 | A2_178 的关键逆变体轨迹，最大发散变体 forward_cut0135 |
| `A2_178_forward_cut0045_centerline.parquet` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_relaxed2x/01_anchor/verify/A4_A2_178_r0_reverse_c0045/forward_cut0045/centerline.parquet` | 499,334 B | `1163054efa3ee652f6cb872185556508e60d4845ad0dd25bb7d778d8a789b44b` | F4 | A2_178 的关键逆变体轨迹，另一个逆变体（用于交叉核验） |
| `run_generalized_ellipse_region_v11.py` | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11/scripts/analysis/run_generalized_ellipse_region_v11.py` | 131,097 B | formal | 实验入口参数与 pipeline | 复现场景入口 |
| `test_region_runner_v11.py` | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11/tests/test_region_runner_v11.py` | 8,505 B | formal | gate 与配置映射测试 | 边界与验证语义 |
| `test_region_audit_v11.py` | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11/tests/test_region_audit_v11.py` | 4,763 B | formal | gate 审计测试 | 结果可靠性约束 |

### 未上传的大文件或派生数据说明

未上传原始实验全量数据集（包含 193 proposals 的全部轨迹 parquet）。
本次上了 7 个关键原始轨迹 parquet 样本（3个候选共 7 条），用于证明 branch identity 反例是“真实样本路径层面”的差异，而非对齐或实现解码错误。

## 11. 给下一位讨论者的最短事实摘要

1. 本轮 V11.2 放宽2×仅提升了数值门控通过率，但决策点在 traversal/cut 不变性，`01_anchor_gate.json` 直接给出 0/3 pass。
2. 决策上当前已出现“主因转移”：旧 V11 fail-fast 在 primary margin 处停止，而 relaxed2x 揭示出 branch 不一致风险。
3. 证据中尚无 branch identity 改造后的实验结果；因此无法据此判断“再放宽”能否长期解决。
4. 用户已确认的问题：判断优先方向（继续放宽 vs 修 branch identity）。

---

## 主判断

**现在应先修复 (\beta) 的 branch identity，再继续后续实验。不要继续把 numerical gate 放宽到 (3\times)、(4\times)，也不要进入 tube、区域数据集或学生模型训练。**

但“停止实验”也不准确。正确做法是：

$$
\boxed{
\text{暂停下游实验}
\quad\rightarrow\quad
\text{继续执行 branch-identity 专项实验}
}
$$

证据非常明确：

- 放宽 (2\times) 后，A2_134、A2_143、A2_178 的 primary 数值求解已经能通过放宽后的主要 teacher 指标；
- 但相同的 (XYZ)、相同的 phase index，在不同 traversal/cut 下得到的 (\beta) RMS p95 仍分别达到 (12.56^\circ)、(14.07^\circ)、(9.23^\circ)；
- 相位中 (\beta) 差异超过 (1^\circ) 的比例达到 (74.58%\sim94.86%)；
- 这已确认不是 phase 对齐或重排问题，同一 phase 的 (XYZ) 完全一致。

这不是轻微数值噪声，也不是门槛设置略严，而是：

$$
x\ \text{相同}
\quad\Longrightarrow\quad
\beta^{\mathrm{forward}}\neq
\beta^{\mathrm{reverse/cut}}
$$

即教师在不同求解路径下选择了不同的逆解 branch。

如果继续放宽 gate，只会把一个多值、不确定的标签关系正式接纳进数据集：

$$
xyz\to{\beta^{(1)},\beta^{(2)},\ldots}
$$

随后无论用 MLP、LGBM 还是 Transformer，静态单值回归器都会被迫做 branch 平均。

---

# 一、当前问题不是一般意义上的“局部连续性差”

需要区分两类 (\beta) 一致性。

## 1. 局部轨迹连续性

同一次求解中：

$$
x_{i+1}\approx x_i
\Rightarrow
\beta_{i+1}\approx\beta_i
$$

它可通过一阶、二阶平滑和 seam 指标评价。

## 2. Branch identity 不变性

对于完全相同的目标轨迹：

$$
x_i^{(v)}=x_i
$$

不同的 traversal direction、cyclic cut、重复运行应产生：

$$
\beta_i^{(v)}\approx\beta_i^{\mathrm{canonical}}
$$

当前失败的是第二类，而且幅度高达 (9^\circ\sim14^\circ)。

即使每条单独求解出来的曲线内部都很平滑，它们也可能是两条不同的平滑 branch。因此，单纯增加曲线平滑权重并不足以解决问题。

---

# 二、为什么进一步放宽 numerical gate 无效

当前已经发生了“主矛盾转移”：

- strict V11 时，流程在 joint margin 处较早停止；
- relaxed (2\times) 后，数值可行性提高，反而暴露出 branch identity 的失败；
- 三个候选全部为 (0/3) 正式通过，关键阻断项是 repeatability 和 reverse/cut invariance。

即使把允许差异放宽到 (5^\circ)，当前 (9^\circ\sim14^\circ) 的 p95 仍然很大。而如果为了通过而放宽到 (15^\circ)，那相当于主动接受：

$$
\text{相同 }xyz
\to
\text{相差十几度的 }\beta
$$

这会直接破坏你希望建立的 deterministic canonical inverse policy。

因此：

> **数值 gate 可以在探索阶段用于暴露候选，但 branch-identity gate 不能通过继续放宽来绕过。**

---

# 三、提高 (\beta) 一致性的原则

提高一致性不能采用以下错误方式：

1. 不能把多条 branch 直接取均值；
2. 不能只在最终曲线上做低通滤波；
3. 不能把差异最大的 phase 删除；
4. 不能让每个 traversal/cut 独立选择自己的最优 seed；
5. 不能只最小化 FK residual，因为不同 branch 的 residual 都可能接近零。

正确目标是：

> **先定义一个 branch identity，然后要求所有 traversal、cut 和重复求解都围绕同一个 branch 求解。**

更准确地说，教师不应是：

$$
\operatorname*{argmin}_\beta
|F(\beta)-x|^2
$$

而应是：

$$
\operatorname*{argmin}*\beta
\left[
|F(\beta)-x|^2
+
\lambda*{\mathrm{branch}}
d_W^2(\beta,\beta_{\mathrm{ref}})
+
\lambda_{\mathrm{history}}
d_W^2(\beta,\beta_{\mathrm{prev}})
+
C_{\mathrm{canonical}}(\beta)
\right]
$$

其中 (\beta_{\mathrm{ref}}) 是已经确定的 reference branch，而不是每次运行独立生成的局部解。

---

# 四、推荐的下一轮专项实验

建议命名为：

```text
V11.3 Canonical Branch-Identity Repair
```

主输出目录：

```text
runs/generalized_ellipse_region_v11_branch_identity/
```

下一轮先只使用当前三个候选，不再扩大 family 数量：

- A2_134；
- A2_143；
- A2_178。

其中：

- A2_178 的 reverse/cut p95 最低，为 (9.23^\circ)；
- A2_143 的 (>1^\circ) phase 比例相对最低，为 (74.58%)；
- A2_134 作为较差负对照。

建议主 pilot 使用 A2_178 和 A2_143，A2_134 只做对照。

---

# 五、Phase BI-0：先定位 branch 差异的结构

## 目标

判断不同 traversal/cut 之间是：

1. 整条曲线落在不同的全局 branch；
2. 只有某些相位发生 branch switching；
3. 主要差异位于 Jacobian 零空间；
4. 某个单独关节或某一段主导差异。

## 输入

对每个候选读取：

- primary；
- repeat；
- forward cuts；
- reverse cuts；
- 共 7 或 8 个版本的 720-phase centerline。

## 计算

对每个 variant (v) 和 primary：

$$
d_\beta^{(v)}(i)
================

\sqrt{
\frac{1}{6}
\sum_{j=1}^6
\left(
\beta_j^{(v)}(i)
----------------

\beta_j^{(\mathrm{primary})}(i)
\right)^2
}
$$

输出：

- p50、p90、p95、max；
- (>0.5^\circ)、(>1^\circ)、(>2^\circ)、(>5^\circ) 比例；
- 每个 (\beta_j) 的差值曲线；
- 差异发生的连续相位区间；
- 每个 phase 的候选聚类数量。

## 零空间分析

在 primary 的每个 phase 计算：

$$
N_i=I-J_i^#J_i
$$

令：

$$
\Delta\beta_i^{(v)}
===================

## \beta_i^{(v)}

\beta_i^{(\mathrm{primary})}
$$

分解为：

$$
\Delta\beta_{\mathrm{null}}
===========================

N_i\Delta\beta_i^{(v)}
$$

$$
\Delta\beta_{\mathrm{task}}
===========================

(I-N_i)\Delta\beta_i^{(v)}
$$

若：

$$
|\Delta\beta_{\mathrm{null}}|
\gg
|\Delta\beta_{\mathrm{task}}|
$$

则可以确认：不同运行主要是在相同末端任务下选择了不同零空间构型。

## 输出

```text
00_branch_forensics/
  per_phase_variant_gap.parquet
  per_joint_gap.csv
  branch_clusters.parquet
  nullspace_gap_report.csv
  branch_transition_plots/
```

---

# 六、Phase BI-1：建立唯一 canonical root

当前 traversal/cut 很可能分别使用独立 seed 或独立起点，因此落入不同 branch。下一步必须首先固定一个 canonical root。

## 1. 选择 root phase

不要固定使用 (0^\circ)。从 720 个 phase 中选择同时满足：

- joint margin 较大；
- (\kappa) 较低；
- FK residual 较低；
- 候选 branch cluster 分离清晰；

的 phase：

$$
i^\star =
\operatorname*{argmax}_i
\left[
w_m m_i
- w_\kappa\log(1+\kappa_i)
- w_r r_i
\right]
$$

## 2. 在 root 处生成多候选

使用 32–64 个 full-(\beta_6) 初值求解同一个 root target，并在 (\beta) 空间聚类。

对每个 cluster 计算：

$$
C_{\mathrm{root}}
=================

w_r r_{\mathrm{FK}}
+
w_p C_{\mathrm{posture}}
+
w_m C_{\mathrm{margin}}
+
w_\kappa C_{\mathrm{condition}}
$$

选定唯一 root cluster：

```text
canonical_root_branch_id
canonical_root_beta
```

并冻结到 protocol。

## 3. 所有 traversal/cut 共享 root

- forward 从 (\beta^\star_{i^\star}) 出发；
- reverse 也从同一个 (\beta^\star_{i^\star}) 出发，只改变遍历方向；
- cut 版本不能在 cut phase 重新做独立 global IK；
- cut phase 的初值必须来自 reference branch 在该 phase 的 (\beta_{\mathrm{ref}})。

这是最小、最重要的改动。

---

# 七、Phase BI-2：reference-branch continuation

## 1. 构建 provisional reference branch

先用 canonical root 做一次高密度 forward + reverse continuation，得到 provisional：

$$
\beta_{\mathrm{ref}}(\phi)
$$

它还不是最终标签，只作为 branch anchor。

## 2. 每一步的求解目标

对 phase (i)，求：

$$
\min_{\beta_i}
\frac{|F(\beta_i)-x_i|^2}{\sigma_x^2}
+
\lambda_{\mathrm{ref}}
|\beta_i-\beta_{\mathrm{ref},i}|*W^2
+
\lambda*{\mathrm{prev}}
|\beta_i-\beta_{i-1}|*W^2
+
C*{\mathrm{canonical}}(\beta_i)
$$

其中：

$$
W=\operatorname{diag}(4,4,2,2,1,1)
$$

## 3. 初值来源

每个 phase 使用：

1. previous solution；
2. secant predictor：

$$
\beta_i^{(0)}
=============

\beta_{i-1}
+
(\beta_{i-1}-\beta_{i-2})
$$

1. Jacobian predictor：

$$
\beta_i^{(0)}
=============

\beta_{i-1}
+
J_{W,i-1}^{#}(x_i-x_{i-1})
$$

1. provisional reference (\beta_{\mathrm{ref},i})。

## 4. 信赖域限制

避免单步跳到另一 branch：

$$
|\beta_i-\beta_i^{(0)}|*{\mathrm{RMS}}
\le
\Delta*{\mathrm{trust}}
$$

Pilot 测试：

$$
\Delta_{\mathrm{trust}}
\in
{0.25^\circ,0.5^\circ,1^\circ}
$$

当无候选满足时，不允许立即跨 branch，而是触发局部多 seed candidate generation。

---

# 八、Phase BI-3：多候选 cyclic branch graph

仅靠 continuation 可能仍依赖局部初值，因此推荐把所有 traversal/cut 变成**候选生成器**，而不是让每个版本独立成为最终标签。

## 1. 每个 phase 的候选集合

收集：

- primary；
- repeat；
- forward/reverse cut；
- previous/secant/Jacobian predictor；
- root branch 周围零空间扰动；
- full-beta 局部多 seed 解。

在 (\beta) 空间聚类，建议阈值：

$$
0.5^\circ
$$

每个 phase 保留 (8\sim16) 个 distinct candidates。

## 2. 节点代价

$$
C_i(k)
======

w_x
\frac{|F(\beta_{i,k})-x_i|^2}{\sigma_x^2}
+
w_pC_{\mathrm{posture}}
+
w_mC_{\mathrm{margin}}
+
w_\kappa C_{\mathrm{condition}}
+
w_{\mathrm{ref}}
|\beta_{i,k}-\beta_{\mathrm{ref},i}|_W^2
$$

## 3. 边代价

$$
E_i(k,l)
========

w_v
|\beta_{i+1,l}-\beta_{i,k}|_W^2
$$

并加入二阶项：

$$
A_i(j,k,l)
==========

w_a
|\beta_{i+1,l}
-2\beta_{i,k}
+\beta_{i-1,j}|_W^2
$$

## 4. 闭环约束

最后一个 phase 与第一个 phase 必须连接：

$$
\beta_{719}\leftrightarrow\beta_0
$$

并且 root phase 必须处于冻结的 canonical root cluster。

## 5. 求解

采用：

- 二阶 Viterbi / dynamic programming；
- 或先一阶 DP，再 whole-curve acceleration correction。

最终只输出一条：

```text
canonical_consensus_branch
```

traversal/cut 不再分别定义不同标签。

---

# 九、Phase BI-4：共享 consensus 的整轨迹优化

对图连接得到的 branch 做联合优化：

$$
J_{\mathrm{curve}}
==================

\lambda_x
\sum_i
|F(\beta_i)-x_i|^2
+
\lambda_v
\sum_i
|\beta_{i+1}-\beta_i|_W^2
$$

$$
+
\lambda_a
\sum_i
|\beta_{i+1}-2\beta_i+\beta_{i-1}|*W^2
+
\lambda_p
\sum_iC*{\mathrm{posture}}(\beta_i)
$$

$$
+
\lambda_m
\sum_iC_{\mathrm{margin}}(\beta_i)
+
\lambda_\kappa
\sum_iC_{\mathrm{condition}}(\beta_i)
+
\lambda_{\mathrm{ref}}
\sum_i
|\beta_i-\beta_{\mathrm{ref},i}|_W^2
$$

循环条件：

$$
\beta_{720}=\beta_0
$$

分阶段优化：

1. 先保持 tracking；
2. 再提高一阶、二阶平滑；
3. 最后提高 margin 与 condition；
4. 不允许最后阶段跳出 canonical root branch。

---

# 十、Phase BI-5：让 traversal/cut 变成审计，而不是标签来源

得到最终 consensus branch 后，再重新执行：

- primary；
- repeat；
- forward traversal；
- reverse traversal；
- 所有 cyclic cuts。

但每次都必须：

1. 共享 canonical root；
2. 在相应 phase 用 consensus branch 初始化；
3. 使用相同 objective；
4. 使用相同 tie-break；
5. 禁止独立 global branch re-selection。

它们的作用是验证：

$$
\Pi_T(X^\star;\text{cut},\text{direction})
\approx
\beta_{\mathrm{consensus}}
$$

而不是分别生成可被接受的不同标签。

---

# 十一、并行消融实验

建议 Codex 同时实现五组：

| ID   | 修改                                         | 目的                    |
| ---- | ------------------------------------------ | --------------------- |
| BI-0 | 当前独立 traversal/cut                         | 基线                    |
| BI-1 | 只共享 canonical root                         | 判断独立 seed 是否为主因       |
| BI-2 | root + reference continuation              | 判断 branch anchor 是否足够 |
| BI-3 | cyclic candidate graph                     | 判断需要不需要全局 branch 选择   |
| BI-4 | graph + consensus whole-curve optimization | 推荐主方案                 |

先在 A2_178、A2_143 上跑 180-phase pilot；top 2 再跑 720-phase。

---

# 十二、建议的 (\beta) 一致性指标

## 1. Repeatability

相同 root、seed、参数重复运行：

$$
d_{\beta,\mathrm{repeat,p95}}
\le0.2^\circ
$$

## 2. Traversal/cut identity

至少满足当前协议已有的 (1^\circ) 判据：

$$
d_{\beta,\mathrm{variant,p95}}
\le1^\circ
$$

$$
\operatorname{ratio}(d_\beta>1^\circ)
\le1%
$$

$$
d_{\beta,\mathrm{variant,max}}
\le2^\circ
$$

推荐强目标：

$$
d_{\beta,\mathrm{variant,p95}}
\le0.5^\circ
$$

$$
\operatorname{ratio}(d_\beta>1^\circ)=0
$$

## 3. 同一 branch cluster

在每个 phase，所有 traversal/cut 的结果必须落入同一个 (\beta)-cluster。

## 4. 轨迹内部连续性

$$
\Delta\beta_{\mathrm{RMS,p95}}\le1^\circ
$$

$$
\Delta\beta_{\mathrm{RMS,max}}\le2^\circ
$$

$$
\Delta^2\beta_{\mathrm{RMS,p95}}\le0.5^\circ
$$

## 5. 闭环

$$
|\beta_{719}-\beta_0|_{\mathrm{RMS}}
\le0.5^\circ\sim1^\circ
$$

---

# 十三、重新进入 Phase 1 和 tube 的明确边界

## 可以重新运行正式 Phase 1 的条件

同一个 candidate 必须同时满足：

1. primary teacher numerical gate；
2. strict joint-margin gate；
3. repeatability gate；
4. forward/reverse/cut branch-identity gate；
5. closure 与 smoothness gate；
6. 所有变体处于同一个 branch cluster。

## 禁止进入 tube 的条件

只要存在以下任一项，就应 fail-closed：

- reverse/cut p95 (>1^\circ)；
- 大量 phase (>1^\circ)；
- 不同 variant 属于不同 branch cluster；
- selected margin 仍低于 protocol minimum；
- 依赖删除 phase 才能通过；
- 依赖平均两个不同 branch 才能形成标签。

## Branch 修复后 margin 仍失败

这是另一个独立问题。此时应：

1. 小范围 re-search center/plane/root；
2. 增强 margin-aware canonical cost；
3. 搜索另一 branch cluster；
4. 重新运行 strict gate。

不应再靠放宽 margin gate直接进入 tube。

---

# 十四、什么时候说明静态 (xyz\to\beta) 本身不成立

如果经过：

- shared root；
- reference continuation；
- cyclic graph；
- consensus whole-curve optimization；

仍然稳定得到两个或更多低 residual、连续、相互相差 (5^\circ\sim15^\circ) 的 branch cluster，那么这不是实现错误，而是任务本身确实多值。

此时不应强制把不同 branch 变成一致，而应选择以下之一：

### 方案 A：限制为一个 canonical chart

明确规定：

```text
branch_id = canonical_0
root_configuration_id = ...
```

只生成该 branch 的数据。

### 方案 B：增加 branch/chart 上下文

训练：

$$
(xyz,\mathrm{chart\ id})\to\beta_6
$$

### 方案 C：有状态教师和学生

训练：

$$
(x_t,\beta_{t-1})\to\Delta\beta_t
$$

当前不能在没有这个判定之前就把多 branch 平均成单一标签。

---

# 最终建议

**继续实验，但实验内容必须改成 branch-identity 修复；不要继续放宽 numerical gate，也不要启动 tube 或学生模型。**

下一步最有价值的主实验是：

$$
\boxed{
\text{shared canonical root}
+
\text{reference continuation}
+
\text{cyclic candidate graph}
+
\text{consensus whole-curve optimization}
}
$$

当前三个候选并不是“数值上完全不可行”，而是“相同目标下 branch 选择不确定”。所以你现在最需要提升的不是模型容量、样本数量或 gate 宽度，而是：

> **让慢速 teacher 本身成为一个确定的、cut-independent、direction-independent 的 canonical policy。**

只有该教师在相同 (XYZ) 上稳定输出同一条 (\beta) branch，后续生成的 tube、多 family 区域数据和学生模型才有可靠的监督目标。
