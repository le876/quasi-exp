---
question_id: Q11
question_number: 11
question_confirmed_by_user: true
question_confirmation_summary: "分析当前数据集生成瓶颈的核心原因，判断下一步应继续放宽 Gate 还是提升当前方法/模型表现，解释 root 逆解空间高度碎片化及单点 canonical branch 无法保证整环持续存在的问题，并据此设计下一步实验方案。"
date: "2026-07-23"
status: ready-to-send
project: "Generalized Ellipse Region V11"
evidence_cutoff: "2026-07-23 11:18:04 +08:00"
source_snapshot: "worktree /mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11 | branch codex/generalized-ellipse-region-v11 | HEAD d972de72aed9ead1361b53a3259e11868e2f96ee | worktree clean | formal artifacts generated at runtime git SHA fb9f5f8708b575022639ab42293bd6203de5b343"
---

# GPT-5 Pro 第 11 次交接：root 逆解空间碎片化与下一步数据集实验设计

## 0. 经用户确认的 GPT-5 Pro 任务

用户已经确认的任务原意：

> 针对当前实验瓶颈，分析为了生成合适的数据集并使用模型训练，下一步应该继续放宽 Gate，还是尝试提升当前方法或模型表现；找到限制实验继续进行的核心原因。重点解释：root 逆解空间高度碎片化，单个 root 点可以具有极低 residual 和较高 margin，但按单点 canonical score 选中的 branch 未必能够沿完整闭环持续存在。分析以上原因之后，设计下一步实验方案。

确认范围：审计当前 V11.3 branch-identity 实验的瓶颈；在“继续放宽 Gate”和“提升当前方法/模型表现”之间作出判断；解释 root 解空间碎片化与整环 branch 可持续性的关系；据此给出面向合适数据集生成与后续模型训练的下一步实验方案。除此之外不自动增加新的研究问题。

## 1. 当前有效事实与最短摘要

### 1.1 证据截止时间

本交接基于截至 `2026-07-23 11:18:04 +08:00` 已存在的代码、配置、正式 artifact 和实验记录。V11.3 Formal Gate 与 completion artifact 写入于 `2026-07-22 17:36:57 +08:00`；实验记录最后写入于 `2026-07-22 17:51:40 +08:00`。本交接在 2026-07-23 对 root 原始 CSV、JSON 与代码字段语义进行了额外只读核对，并按用户追加要求将完整正式实验目录打包为证据附件。

### 1.2 当前有效事实

1. V11.3 已完整执行 branch-identity forensics、canonical-root 搜索、180-phase Pilot 和 720-phase Formal；最终 `formal_stage_gate_pass=false`、`phase1_restart_authorized=false`、`tube_authorized=false`。
2. BI-2 和 BI-3 已将原先 BI-0 下约 `9–14°` 的 traversal/cut branch gap 降至 Formal P95 `0.004182–0.075522°`；四个 Formal 组合 repeat P95 均为 `0°`，`gap > 1°` 比例均为 `0%`。原来的大尺度 branch-identity 漂移已显著收敛。
3. 四个 Formal 组合仍全部失败。共同失败项为 audit variant solver success、audit variant trust preserved、primary residual P95/max、strict joint margin 和 trajectory velocity max。A2_143 的两个方法还未达到 single-cluster；A2_143/BI-3 还失败于 acceleration P95。
4. 两个候选各使用 64 个 root seeds；A2_143 得到 62 个 root-solution clusters，A2_178 得到 58 个。两个最终选中 cluster 的支持数都只有 1。
5. root 选择按单点的 residual、posture、margin deficit 和 conditioning 形成 canonical cost，再选择最低 cost cluster 的代表解；该 cost 没有使用后续短弧或整环 continuation 成功率。
6. A2_143 的 root JSON 存在字段语义混合：`canonical_root_beta` 来自选中的 candidate 21/cluster 21，但同一 JSON 中的 `teacher_fk_residual_mm`、`joint_margin_min_deg` 和 `kappa` 仍来自 root phase 的原 primary 点。原实验记录将后者写成了选中 root beta 的指标。A2_143 选中 candidate 21 的实际 CSV 指标为 residual `0.0136972995 mm`、margin `4.3522289620°`、kappa `8.7548653065`；JSON 中原 primary 点对应的是 residual `0.0000787748 mm`、margin `3.3630241336°`、kappa `9.7132658902`。
7. 运行器在后续实验中从 root JSON 读取 `phase_idx` 和 `canonical_root_beta`，而不是读取上述混合的 residual/margin/kappa 字段。因此现有证据表明该缺口会影响报告和 root 质量解释；现有代码读取路径没有显示它直接替换了实际用于 continuation 的 root beta。

### 1.3 关键边界

1. Formal 结果证明的是：当前选出的 canonical chart 在当前 solver、trust、graph cost 和严格 Gate 下不能覆盖完整闭环；它没有证明所有可能的 root cluster 或所有 center/plane 都不存在严格可行的整环 chart。
2. 当前没有运行“以短弧/整环可行性筛选 root cluster”的实验，也没有运行“把 residual 与 margin 作为图节点硬可行约束”的实验。
3. Tube、dataset generation 和 Student 训练均未启动，因此当前没有可用于判断下游学习效果的训练集、validation/test 指标或模型结果。
4. “root singleton 与 Formal 失败之间存在直接因果关系”尚未被受控实验验证；目前可以确认的是 singleton、解空间碎片化和整环数值失败同时存在。

## 2. 项目目标与本轮边界

### 2.1 用户给定的项目目标

生成几何可追踪、标签 branch 一致、能够用于模型训练并具有泛化价值的椭圆区域数据集，随后使用该数据集训练模型。当前需要先确定阻止数据集生成继续推进的核心原因，并决定下一步实验方向。

### 2.2 当前实际覆盖范围

- 对象：V11.2 relaxed2x 选出的 A2_143、A2_178；A2_134 仅作为冻结负对照。
- 方法：BI-0 至 BI-4 的 branch-identity 策略；Formal 仅运行 Pilot 冻结选出的 BI-3 和 BI-2。
- 分辨率：forensics 720 phase；Pilot 180 phase；Formal 720 phase。
- 审计 variants：forward/reverse，cyclic cuts 0/45/90/135。
- 当前终点：Formal branch-identity Gate；没有越过失败 Gate 生成 Tube 或 Student 数据。

### 2.3 当前证据未覆盖的内容

- 没有对全部 58/62 个 root clusters 分别运行短弧或完整闭环 continuation。
- 没有把 root cluster 支持数、局部 basin 稳定性或整环存活率纳入 root selection objective。
- 没有在图候选生成阶段把 strict residual、joint margin 和 trust preservation 全部设置为硬可行条件后重跑。
- 没有证明当前 center/plane 上存在或不存在至少一个满足全部严格约束的全环 chart。
- 没有生成正式训练数据，也没有训练或比较 Student 模型。
- 没有执行新的 Gate 放宽倍数实验；V11.3 恢复的是严格 Gate。

## 3. 相对上次材料新增的事实

上次交接为 Q10，证据截止于 `2026-07-21 10:41:16 +08:00`，当时只知道 relaxed2x 暴露了 `9–14°` traversal/cut branch 差异，尚未实现 branch-identity repair。

| 类型 | 已发生的变化 | 实际证据 | 证据等级 |
|---|---|---|---|
| 实现/配置 | 新增共享 root、reference continuation、cyclic candidate graph、whole-curve consensus 以及 fail-closed audit | `branch_identity.py`、`run_generalized_ellipse_branch_identity_v11_3.py`、`generalized_ellipse_region_v11_branch_identity.yaml` | formal implementation |
| 实验 | 完成 BI-0 至 BI-4 的 180-phase Pilot；BI-3、BI-2 进入 720-phase Formal | `pilot_method_ranking.csv`、`formal_ranking.csv` | formal |
| 正向结果 | Formal traversal/cut P95 降至 `0.004182–0.075522°`，repeat P95 全部为 `0°` | `formal_ranking.csv` | formal |
| 失败结果 | 四个 Formal 组合全部 Gate FAIL；Phase 1、Tube 与 Student 未获授权 | `formal_gate.json`、四个 per-method Gate、`BRANCH_IDENTITY_EXPERIMENT_COMPLETED.json` | formal |
| root 证据 | 64 seeds 形成 A2_143 的 62 簇和 A2_178 的 58 簇；两个选中簇均为 singleton | 两组 `root_candidates.csv`、`root_clusters.csv`、`canonical_root.json` | formal artifact |
| 事实更正 | A2_143 JSON/旧实验记录把 root-phase primary 指标与 selected canonical beta 指标混写；选中 candidate 21 的真实指标应从 CSV 读取 | `A2_143_canonical_root.json`、`A2_143_root_candidates.csv`、`A2_143_root_clusters.csv` 与 runner payload 代码 | post-run evidence audit |

## 4. 实验方法与复现口径

### 4.1 代码与工作树 fixed point

| 项目 | 值 |
|---|---|
| 项目根 | `/mnt/ML_projects/quasi_exp` |
| 分支 | `codex/generalized-ellipse-region-v11` |
| 当前 HEAD | `d972de72aed9ead1361b53a3259e11868e2f96ee` |
| artifact runtime git SHA | `fb9f5f8708b575022639ab42293bd6203de5b343` |
| worktree | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11` |
| dirty 状态 | V11.3 worktree clean；项目主 worktree 另有用户未跟踪/修改文件，不属于 V11.3 fixed point |
| 仅 checkout 当前 HEAD 是否足以复现 | 当前 HEAD 包含 runtime code 与后续文档修订；仍需使用冻结输入 artifacts、指定配置和同一环境。现有正式 artifacts 的 provenance 记录 runtime SHA `fb9f5f8`。 |

### 4.2 方法与参数

| 方法 | 实际定义 |
|---|---|
| BI-0 | 冻结的 V11.2 独立 traversal/cut，作为 branch-gap 基线和负对照 |
| BI-1 | 共享 canonical root，但后续 traversal 独立 continuation |
| BI-2 | 共享 root + frozen reference continuation + trust radius |
| BI-3 | 每 phase 生成不同候选节点，以带 velocity、acceleration、reference 和 closure cost 的 cyclic graph 选闭环路径 |
| BI-4 | 在 BI-3 基础上做 consensus whole-curve optimization；只进入 Pilot，未进入 Formal |

| 参数 | 实际值 | 来源 |
|---|---:|---|
| root seed count | 64 | `generalized_ellipse_region_v11_branch_identity.yaml` |
| root cluster threshold | `0.5°` | 同上 |
| root safe margin | `1.5°` | 同上 |
| selected trust radius | `0.5°` | 同上 |
| graph candidates per phase | 最少 8，最多 16 | 同上 |
| graph seed attempts | 24 | 同上 |
| graph velocity/acceleration/reference/closure weights | `1.0 / 0.5 / 2.0 / 5.0` | 同上 |
| teacher solver seed | `20260725` | 同上 |
| Pilot/Formal phase count | `180 / 720` | 同上 |

root 单点 canonical cost 的代码定义为：

```text
residual_mm^2
+ 10 * posture_cost
+ 100 * max(1.5° - margin, 0)^2
+ 0.001 * log(1 + kappa)
```

该 cost 在 `select_canonical_root_solution` 中对 root phase 的候选解排序。当前实现没有在该 cost 中加入 continuation length、整环存活率、全环 residual、全环最低 margin 或全环 trust-preservation。

### 4.3 数据生成与划分

| 项目 | 实际口径 |
|---|---|
| 数据来源 | V11.2 relaxed2x 冻结的 A2_143、A2_178 target centerline 与 variants；A2_134 为负对照 |
| 生成方式 | 对冻结闭环 target 做 root 多 seed IK、branch repair、traversal/cut audit 和 Formal Gate |
| root 样本数 | 每个候选 64 个 seed solution |
| trajectory 样本数 | Pilot 每轨迹 180 phase；Formal 每轨迹 720 phase |
| train | 未生成 |
| validation | 未生成 |
| test | 未生成 |
| seed | teacher solver seed `20260725`；root seeds 为 primary beta 加 atlas 最近邻确定性 seeds |
| 泄漏审计 | 本阶段没有构造 Student dataset；继承的 V11 source protocol 保持冻结，不能据此声称已完成新数据集泄漏审计 |

### 4.4 baseline、评价和 Gate

| 项目 | 实际定义 | 是否正式 |
|---|---|---|
| baseline | BI-0，即 V11.2 独立 traversal/cut 的冻结结果 | 是，branch-identity 负对照 |
| branch metrics | repeat P95、variant P95/max、`gap > 1°` 比例、branch cluster count | 是 |
| numerical metrics | FK residual P95/max、solver success、trust preservation | 是 |
| feasibility/smoothness | joint margin min、velocity、acceleration、closure | 是 |
| Gate | residual P95/max `1/3 mm`；margin `>=1.5°`；repeat P95 `<=0.2°`；variant P95/max `<=1/2°`；`gap>1° <=1%`；acceleration P95 `<=0.5°`；single-cluster | 是 |

### 4.5 复现入口与产物

```text
PYTHONPATH=src \
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python3.11 \
scripts/analysis/run_generalized_ellipse_branch_identity_v11_3.py \
  --config configs/generalized_ellipse_region_v11_branch_identity.yaml \
  --preset formal \
  --stage all
```

主要产物：

- `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/01_canonical_root/`
- `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/02_pilot/`
- `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/03_formal/`
- `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/BRANCH_IDENTITY_EXPERIMENT_COMPLETED.json`

## 5. 实验结果

| Fact ID | 已观察结果 | 数值 | 证据等级 | 实际附件/字段 | 适用边界 |
|---|---|---:|---|---|---|
| F1 | A2_178/BI-2 Formal | residual P95/max `15.683221/18.887969 mm`；variant P95/max `0.004823/0.378088°`；cluster 1；9/15 | formal | `formal_ranking.csv` | 720 phase、0.5° trust |
| F2 | A2_178/BI-3 Formal | residual P95/max `2.169650/8.988982 mm`；variant P95/max `0.035090/0.476667°`；cluster 1；9/15 | formal | `formal_ranking.csv` | 720 phase、graph |
| F3 | A2_143/BI-2 Formal | residual P95/max `4.849352/8.237505 mm`；variant P95/max `0.004182/0.602651°`；cluster 2；8/15 | formal | `formal_ranking.csv` | 720 phase、0.5° trust |
| F4 | A2_143/BI-3 Formal | residual P95/max `1.492651/4.823111 mm`；variant P95/max `0.075522/0.664155°`；cluster 2；7/15 | formal | `formal_ranking.csv` | 720 phase、graph |
| F5 | 四个 Formal 的 repeat 与大 gap | repeat P95 全为 `0°`；`gap>1°` 比例全为 `0%` | formal | `formal_ranking.csv` | 仅已选 BI-2/BI-3 和两个候选 |
| F6 | 四个 Formal 的最低 margin | 全为 `0°` | formal | `formal_ranking.csv` | 不能满足 1.5° strict Gate |
| F7 | A2_143 root 解空间 | 64 candidates、62 clusters、selected cluster 21、支持数 1 | formal artifact | `A2_143_root_candidates.csv`、`A2_143_root_clusters.csv`、`A2_143_canonical_root.json` | 单一 root phase 21 |
| F8 | A2_178 root 解空间 | 64 candidates、58 clusters、selected cluster 0、支持数 1 | formal artifact | `A2_178_root_candidates.csv`、`A2_178_root_clusters.csv`、`A2_178_canonical_root.json` | 单一 root phase 62 |
| F9 | A2_143 selected beta 的真实单点指标 | residual `0.013697 mm`、margin `4.352229°`、kappa `8.754865` | post-run artifact audit | candidate 21 in `A2_143_root_candidates.csv` | 更正旧记录的字段解释 |
| F10 | A2_178 selected beta 的单点指标 | residual `0.000323 mm`、margin `4.021413°`、kappa `6.859870` | formal artifact | candidate 0 in `A2_178_root_candidates.csv` | 与 JSON 中数值一致 |

### 5.1 通过项

- 00_protocol、branch forensics 和 canonical-root 阶段执行完成。
- BI-3 图层实际达到每 phase 8–9 个不同候选节点，并启用了二阶 acceleration DP。
- BI-2/BI-3 的 repeatability、traversal/cut P95、traversal/cut max、`gap > 1°` 比例和 closure 检查通过。
- A2_178 的两个 Formal 方法达到 single-cluster。
- Formal 方法目录严格只包含 Pilot 预选的 BI-2 和 BI-3。

### 5.2 失败与反例

- 所有 Pilot 方法都是 `0/2` candidate Gate pass；BI-3、BI-2 的排名仅表示累计通过检查数最高，不表示策略 Gate 已通过。
- 四个 Formal 组合全部失败于 residual、strict margin、solver/trust 和 velocity max。
- BI-2 是“branch identity 很强但数值 tracking 很差”的直接反例：variant P95 低于 `0.005°`，但 residual P95 达 `4.85–15.68 mm`。
- BI-3 将 residual P95 相对 BI-2 降至 `1.49–2.17 mm`，但仍未满足 `1 mm` P95、`3 mm` max 和 `1.5°` margin。
- A2_143 在 BI-2/BI-3 下仍有两个 branch clusters。
- 单点 root 指标较好没有保证整环最低 margin：两个 selected root 单点 margin 均高于 `4°`，但四条 Formal 轨迹的最低 margin 都为 `0°`。

## 6. 证据等级、失败结果与未知事项

### 6.1 正式事实

- V11.3 实验完整执行，但 Formal Gate 失败；Phase 1、Tube、dataset 和 Student 均未获授权。
- branch identity 相对 BI-0 已显著改善，但 numerical feasibility 与 smoothness 没有同时满足严格 Gate。
- 当前 root 多 seed 解高度分散，两个 selected clusters 均为 singleton。
- 当前 root selection objective 是单点 objective，不包含整环可行性。

### 6.2 诊断、解释性陈述与边界

| 内容 | 等级 | 能支持什么 | 不能支持什么 |
|---|---|---|---|
| V11.2 variant-primary 差主要沿 Jacobian nullspace 展开 | diagnostic-only | 支持不同 beta 可命中相同 XYZ，且不是简单 phase 对齐错误 | 不能单独证明哪个 root cluster 能覆盖整环 |
| “当前 selected chart 不能在严格条件下覆盖完整闭环” | formal result boundary | 支持当前方法、当前 selected root 和当前两个候选失败 | 不能外推为所有 root/center/plane 都不可行 |
| “singleton root 导致 Formal 失败” | 未验证因果假设 | 可作为 GPT 分析对象 | 不能写成已证明根因 |
| “应通过整环 feasibility 选择 root” | 未执行的实验方向 | 可由 GPT 在回答用户确认的问题时评价或形成实验设计 | 不能写成当前实验结论 |

### 6.3 已发现的实现或证据缺口

- A2_143 `canonical_root.json` 将 root-phase primary 的 residual/margin/kappa 与 selected candidate 21 的 beta 放在同一 payload 中，字段没有明确区分 provenance；旧实验记录据此误写了 selected root 指标。
- `root_clusters.csv` 的 selected cluster 支持数为 1，但当前 root objective 没有 basin size、seed support 或 continuation survival 项。
- 当前图优化用 residual 作为软 unary cost，并用 reference/smoothness/closure 共同加权；现有 artifact 没有证明 hard-feasible node 子图是否存在。
- 当前未保存“每个 root cluster 经过固定长度 continuation 后的存活/失败曲线”，无法直接把 Formal 失败定位到某个 root cluster disappearance phase。

### 6.4 尚未知或尚未执行

- 是否有其他 root cluster 能在全部 720 phases 同时满足 residual、margin、trust、smoothness、closure 和 single-cluster。
- 当前 target center/plane 下是否存在任意严格可行全环 chart。
- 当 residual/margin/trust 改为图节点硬约束后，候选图会保留闭环路径，还是出现某些 phase 空层。
- Gate 再次放宽能否产生对 Student 训练有用且物理/几何可靠的标签；当前没有此项实验。
- 如果先生成带较大 residual 或零 margin 的数据，Student 的训练稳定性、测试误差和部署安全性会怎样；当前没有数据或训练结果。

### 6.5 当前不能声称

- 不能声称静态 `xyz -> beta` 映射从理论上必然不存在。
- 不能声称继续放宽 Gate 一定能或一定不能得到可训练数据；目前只知道已有 relaxed2x 曾暴露 branch gap，而 V11.3 strict Gate 下修复 branch 后仍有数值不可行。
- 不能声称当前 selected root 是稳定多数解；两个 selected clusters 都只有一个 seed。
- 不能将 A2_143 JSON 中 `0.000079 mm / 3.363024°` 当成 selected candidate 21 的 residual/margin。
- 不能声称已经存在合格数据集或已有 Student 模型表现。

## 7. 决策所需的客观对照

本节只比较已经执行的方法和结果，不给出 Codex 推荐或路线排序。

| 对照对象 | 已执行输入/方法 | 已观察结果 | 证据等级 | 已知限制 |
|---|---|---|---|---|
| V11.2 relaxed2x | numerical Gate 上限放宽 2 倍、下限减半 | primary 数值项可推进，但 traversal/cut P95 出现 `9.23–14.07°` branch gap，0/3 anchors pass | historical formal | 未修 branch identity |
| BI-1 | 共享 root，后续独立 continuation | Pilot 最坏 variant P95 `12.944281°`，0/2 pass | formal Pilot | 共享单点不阻止后续离支 |
| BI-2 | root + reference continuation | Formal variant P95 `0.004182–0.004823°`，但 residual P95 `4.849–15.683 mm`，margin `0°` | formal | branch identity 强；tracking/trust 失败 |
| BI-3 | 多候选 cyclic graph | Formal residual P95 `1.493–2.170 mm`、variant P95 `0.035090–0.075522°`，仍 0/2 pass | formal | residual 改善但未过 strict Gate；A2_143 仍双簇 |
| BI-4 | graph + consensus whole-curve optimization | Pilot variant P95 较低，但 residual/margin/其他项仍失败，0/2 pass | formal Pilot | 未按 Pilot 排名进入 Formal |
| 当前 root selection | 64-seed 单点 IK 聚类后按 canonical cost 选 cluster representative | 58/62 clusters，两个 selected cluster 都为 singleton；单点 margin >4°，全环 Formal margin 0° | formal artifact | 没有 continuation feasibility admission |

## 8. 经用户确认的问题

1. 当前阻止“生成合适数据集并进入模型训练”的核心瓶颈是什么？请区分已由实验直接证明的现象、可以由现有证据支持的机制解释，以及仍需实验验证的原因。
2. 基于当前证据，下一步应该继续放宽 Gate，还是优先提升当前方法或模型表现？请围绕数据标签质量、整环可行性和后续训练用途作出明确判断。
3. 如何解释以下关键现象：root 逆解空间高度碎片化；单个 root 点可以具有极低 residual 和较高 margin，但按单点 canonical score 选中的 branch 未必能够沿完整闭环持续存在？
4. 在完成以上分析后，请设计下一步实验方案，使其直接服务于判断能否生成合适数据集并继续模型训练。

## 9. 经用户确认的回答要求

- 返回对当前实验瓶颈和核心原因的分析。
- 明确判断下一步应放宽 Gate，还是尝试提升当前方法或模型表现。
- 重点解释 root 解空间碎片化、单点 canonical score 与完整闭环 branch 可持续性之间的问题。
- 在上述分析基础上给出可执行的下一步实验方案，目标是判断并推进合适数据集生成与后续模型训练。
- 不把尚未执行的实验或未经验证的因果解释写成既定事实。

## 10. 附件清单与复现信息

GPT-5 Pro 实际可见的是 ZIP 内文件名。原始 SSH 路径只用于 provenance。

GPT 可见文件数（`QUESTION.md + 19 个实际证据附件`）：`20`。达到 16 个及以上，因此打包器应生成 `Q11_GPT5Pro_upload.zip`，网页端只上传该 ZIP。

| 上传文件名 | 原始 SSH 路径 | 大小 | 证据等级 | 支持事实 | 上传理由 |
|---|---|---:|---|---|---|
| `GeneralizedEllipseRegionV11BranchIdentity实验记录.md` | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11/docs/GeneralizedEllipseRegionV11BranchIdentity实验记录.md` | 14,459 B | formal summary，含一项已在 Q11 更正的字段解释 | F1–F10、实验流程 | 提供完整实验上下文；Q11 明确指出 A2_143 root 指标混写 |
| `generalized_ellipse_region_v11_branch_identity.yaml` | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11/configs/generalized_ellipse_region_v11_branch_identity.yaml` | 2,820 B | formal config | 方法参数和严格 Gate | 核验 threshold、phase、seed、graph cost |
| `branch_identity.py` | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11/src/quasi_exp/teacher/branch_identity.py` | 44,040 B | formal implementation | root cost、聚类、reference/graph 方法 | 核验 root 选择目标和 branch 方法 |
| `run_generalized_ellipse_branch_identity_v11_3.py` | `/mnt/ML_projects/quasi_exp/.worktrees/generalized-ellipse-region-v11/scripts/analysis/run_generalized_ellipse_branch_identity_v11_3.py` | 54,807 B | formal implementation | 实验流程、root payload、Formal 筛选 | 核验 JSON 字段来源和执行口径 |
| `A2_143_canonical_root.json` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/01_canonical_root/A4_A2_143_r1_reverse_c0045/canonical_root.json` | 4,124 B | formal artifact | F7、字段混写 | 核验 selected beta/cluster 与 root-phase 字段 |
| `A2_143_root_candidates.csv` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/01_canonical_root/A4_A2_143_r1_reverse_c0045/root_candidates.csv` | 14,968 B | formal artifact | F7、F9 | 核验 64 candidates 和 candidate 21 真实指标 |
| `A2_143_root_clusters.csv` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/01_canonical_root/A4_A2_143_r1_reverse_c0045/root_clusters.csv` | 5,285 B | formal artifact | F7 | 核验 62 clusters、selected singleton 和排序 |
| `A2_178_canonical_root.json` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/01_canonical_root/A4_A2_178_r0_reverse_c0045/canonical_root.json` | 4,167 B | formal artifact | F8、F10 | 核验 selected root |
| `A2_178_root_candidates.csv` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/01_canonical_root/A4_A2_178_r0_reverse_c0045/root_candidates.csv` | 14,844 B | formal artifact | F8、F10 | 核验 64 candidates 和 selected candidate 0 |
| `A2_178_root_clusters.csv` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/01_canonical_root/A4_A2_178_r0_reverse_c0045/root_clusters.csv` | 4,923 B | formal artifact | F8 | 核验 58 clusters 和 selected singleton |
| `pilot_method_ranking.csv` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/02_pilot/method_ranking.csv` | 220 B | formal Pilot | BI-1 至 BI-4 对照 | 核验进入 Formal 的方法排序不是 Gate pass |
| `formal_ranking.csv` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/03_formal/formal_ranking.csv` | 754 B | formal | F1–F6 | 核验四个 Formal 组合的核心数值 |
| `formal_gate.json` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/03_formal/gate.json` | 9,262 B | formal | Formal 总 Gate | 核验无方法通过、无 Phase 1/Tube 授权 |
| `A2_143_BI-2_gate.json` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/03_formal/A4_A2_143_r1_reverse_c0045/BI-2/trust_0p5deg/gate.json` | 3,193 B | formal | F3 和具体 checks | 核验失败项 |
| `A2_143_BI-3_gate.json` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/03_formal/A4_A2_143_r1_reverse_c0045/BI-3/gate.json` | 3,564 B | formal | F4 和具体 checks | 核验失败项与 graph report |
| `A2_178_BI-2_gate.json` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/03_formal/A4_A2_178_r0_reverse_c0045/BI-2/trust_0p5deg/gate.json` | 3,124 B | formal | F1 和具体 checks | 核验失败项 |
| `A2_178_BI-3_gate.json` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/03_formal/A4_A2_178_r0_reverse_c0045/BI-3/gate.json` | 3,438 B | formal | F2 和具体 checks | 核验失败项与 graph report |
| `BRANCH_IDENTITY_EXPERIMENT_COMPLETED.json` | `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/BRANCH_IDENTITY_EXPERIMENT_COMPLETED.json` | 199 B | formal | 最终授权状态 | 核验实验完成但下游未授权 |
| `generalized_ellipse_region_v11_branch_identity_full.zip` | `/mnt/ML_projects/quasi_exp/runs/gpt5pro_handoff_sources/Q11/generalized_ellipse_region_v11_branch_identity_full.zip` | 16,542,714 B | complete formal artifact archive | F1–F10 与完整 per-phase 证据 | 按用户要求提供 314 个正式实验文件的完整目录树，使 GPT-5 Pro 可以检查全部 forensics、Pilot、Formal、Parquet、CSV、PNG、协议和 Gate |

### 完整实验目录归档与派生数据说明

按用户追加要求，现已完整上传 `/mnt/ML_projects/quasi_exp/runs/generalized_ellipse_region_v11_branch_identity/`，没有抽样或删减。为保留目录结构，该目录先被机械打包为 `generalized_ellipse_region_v11_branch_identity_full.zip`，再作为附件进入网页端唯一上传包 `Q11_GPT5Pro_upload.zip`。

- 完整源目录文件数：`314`；
- 完整源目录文件总字节数：`49,310,743 B`；
- 内部完整实验 ZIP 大小：`16,542,714 B`；
- 内部完整实验 ZIP SHA-256：`f586dd29ca7acd86cefc3d0cebdb4ec85b2fe484cccafa312fe7366d5e723ccd`；
- 归档方式：在 `/mnt/ML_projects/quasi_exp/runs/` 下对 `generalized_ellipse_region_v11_branch_identity` 执行 ZIP deflate level 9，保留顶层目录和全部相对路径；
- 生成时间：`2026-07-23 11:17:27 +08:00`；
- 完整性验证：内部 ZIP 含 `314` 个普通文件和 `547` 个总 entries（含目录 entries），CRC 通过；逐文件比较源文件与 ZIP member 的 SHA-256，未发现缺失、额外文件或内容差异；
- 抽样方式与随机种子：不适用；这是完整机械归档，不是派生样本；
- 敏感信息审计：源树没有符号链接或特殊文件；未发现 `.env`、SSH key、credential、auth、token、secret/private-key 命名文件，文本模式扫描未命中 API key、password、authorization bearer 或 private/secret key 字段。

外层仍单独保留关键小文件，虽然这些文件也存在于内部完整归档中。这是用户要求完整目录后的有意重复：单独文件便于 GPT-5 Pro 直接读取关键事实，内部完整归档用于独立检查所有 per-phase trajectory、分支转换、nullspace、Pilot 和 Formal 证据。GPT-5 Pro 现在可以访问完整正式实验目录，但运行代码仍需要正文所列环境与依赖。

## 11. 给下一位讨论者的最短事实摘要

1. V11.3 已把 traversal/cut branch gap 从约 `9–14°` 降到 P95 `0.004–0.076°`，但四个 Formal 组合仍因 residual、margin、solver/trust 和 velocity 失败；这不是单纯的 branch-identity 失败。
2. 64 个 root seeds 分别形成 62/58 个 clusters，两个 selected clusters 都是 singleton；当前 root cost 只评价单点，不评价 continuation 或整环存活。
3. A2_143 旧记录混写了 root-phase primary 指标与 selected candidate 指标；selected candidate 21 的真实 residual/margin 是 `0.013697 mm / 4.352229°`。该字段问题影响 root 质量解释，现有读取路径没有显示它替换了用于 continuation 的 selected beta。
4. 当前没有正式数据集或 Student 训练结果，也没有证明其他 root clusters 或其他 center/plane 不存在严格可行的完整闭环 chart。
5. 用户已确认的问题：判断当前核心瓶颈，决定下一步应放宽 Gate 还是提升当前方法/模型表现，解释 root 碎片化与整环 branch 可持续性，并设计下一步实验方案。

---
# 独立审计结论

我已按要求阅读外层 `QUESTION.md`、关键汇总文件和代码，并解压检查了完整实验归档中的 root candidates、root clusters、branch-transition、nullspace、180-phase Pilot、720-phase Formal、所有 Formal consensus/audit Parquet 和 Gate。完整归档记录了 314 个正式实验文件；V11.3 最终确实为 `formal_stage_gate_pass=false`、`tube_authorized=false`，没有生成训练数据或启动 Student。

最重要的结论是：

$$
\boxed{
\text{现在不应继续放宽最终 Gate，也不应训练模型。}
}
$$

但当前核心瓶颈也已经**不再是原来的 traversal/cut branch identity 漂移**。V11.3 已把此前约 (9^\circ\sim14^\circ) 的 branch gap 压到了 (0.004^\circ\sim0.076^\circ) 的 P95。现在真正阻止数据生成的是：

$$
\boxed{
\text{尚未找到一条在全部 720 个相位上同时满足 tracking、joint margin、连续性和闭环约束的严格可行 branch。}
}
$$

更具体地说，现有 teacher 已经能让不同 traversal/cut **输出近似相同的标签**，但尚不能保证这个共同标签在每个相位都是**准确、安全且可连续实现的标签**。

---

# 一、实验直接证明的事实

## 1. Branch identity 的大尺度漂移已经基本被控制

Formal 结果如下：

| 候选/方法         | traversal/cut gap P95 |       gap max | (>1^\circ) 比例 |
| ------------- | --------------------: | ------------: | ------------: |
| A2_178 / BI-2 |      (0.004823^\circ) | (0.378^\circ) |          (0%) |
| A2_178 / BI-3 |      (0.035090^\circ) | (0.477^\circ) |          (0%) |
| A2_143 / BI-2 |      (0.004182^\circ) | (0.603^\circ) |          (0%) |
| A2_143 / BI-3 |      (0.075522^\circ) | (0.664^\circ) |          (0%) |

四个 Formal 组合的 repeat P95 都为 (0^\circ)。

相较 V11.2 的 (9^\circ\sim14^\circ) 分支差异，这是明确改善。此前差异也不是相位错位，而主要位于 Jacobian null space。

我对完整 `nullspace_gap_report.csv` 重新统计后发现：对于原来 gap (>1^\circ) 的样本，null-space 分量占总 gap 的中位比例约为：

* A2_134：(0.922)；
* A2_143：(0.952)；
* A2_178：(0.994)。

这支持原先的判断：早期 traversal/cut 差异主要是冗余零空间内选了不同构型，而不是末端目标不同。

---

## 2. 但 Formal 的数值可行性全部失败

Formal 核心指标是：

| 候选/方法         | residual P95 | residual max | 最小 margin | (\Delta\beta) max | branch clusters |
| ------------- | -----------: | -----------: | --------: | ----------------: | --------------: |
| A2_178 / BI-2 |   (15.683)mm |   (18.888)mm | (0^\circ) |     (5.597^\circ) |               1 |
| A2_178 / BI-3 |    (2.170)mm |    (8.989)mm | (0^\circ) |     (9.757^\circ) |               1 |
| A2_143 / BI-2 |    (4.849)mm |    (8.238)mm | (0^\circ) |     (8.471^\circ) |               2 |
| A2_143 / BI-3 |    (1.493)mm |    (4.823)mm | (0^\circ) |     (6.253^\circ) |               2 |

因此，失败不是只差一个小阈值，而是同时存在：

* residual 超标；
* joint margin 降到 (0^\circ)；
* 局部 (\beta) 跳变超过 (2^\circ)；
* audit solver success 失败；
* trust preservation 失败。

A2_143 / BI-3 还失败于 acceleration P95。

---

## 3. 逐相位审计显示：真正严重的是连续相位区间触碰关节边界

我对四条 Formal consensus Parquet 逐行复算，结果如下：

| 方法            | margin (=0^\circ) 的相位数 | margin (<1.5^\circ) 的相位数 | residual (>3)mm 的相位数 |
| ------------- | ---------------------: | -----------------------: | -------------------: |
| A2_143 / BI-2 |                    200 |                      213 |                   81 |
| A2_143 / BI-3 |                     94 |                      192 |                    2 |
| A2_178 / BI-2 |                    170 |                      175 |                  105 |
| A2_178 / BI-3 |                     59 |                      123 |                   18 |

这些不是零散数值噪声，而是连续相位区间。例如：

* A2_178 / BI-3 在 phase (497\sim519) 和 (595\sim629) 出现边界区；
* A2_143 / BI-3 在 phase (111\sim161)、(566\sim576)、(587\sim607) 等区间出现边界区。

主要触边变量是：

$$
\beta_5,\beta_6=\pm15^\circ
$$

而不是前四个变量。

与此同时，这些困难区的 position Jacobian 条件并不极端：

| 方法            | (\sigma_3) p05 | (\kappa) p95 |
| ------------- | -------------: | -----------: |
| A2_143 / BI-3 |        (0.508) |      (11.83) |
| A2_178 / BI-3 |        (0.317) |      (19.47) |

因此，现有证据更支持：

> 当前 selected branch 在部分相位被推到了 joint-bound / active-constraint 边界。

而不是：

> 这些失败主要由普通 Jacobian 奇异性造成。

---

## 4. A2_143 的“双 cluster”不是广泛分支分裂

逐相位查看 `audit_branch_clusters.parquet` 后：

* A2_143 / BI-2 只有 phase 107 出现 2 clusters，间距约 (0.603^\circ)；
* A2_143 / BI-3 只有 phase 610 出现 2 clusters，间距约 (0.601^\circ)；
* 其他 719 个相位都是单 cluster；
* A2_178 所有 720 个相位都是单 cluster。

因此，A2_143 的 `single_branch_cluster=false` 是正式 Gate 失败项，但不是目前最主要、最广泛的失败模式。广泛失败仍然是 margin、residual、solver/trust 和局部大跳变。

---

# 二、完整代码审计发现的两个关键实现问题

## 1. BI-2 的超小 branch gap 部分来自 fail-closed reference copying

`ReferenceBranchTeacher` 的实际逻辑是：

当某一相位没有候选同时满足 reference trust 时，不选择另一个可行 branch，而是直接写入：

$$
\beta_i=
\beta_{\mathrm{reference},i}
$$

或 root beta。

这样做确实能防止 traversal/cut 重新跳 branch，但如果 reference 本身不能准确到达该目标，就会出现：

* branch gap 极小；
* residual 很大；
* joint margin 为 0；
* `trust_violation_count>0`；
* `solver_success=false`。

这解释了为什么 BI-2 可以同时出现：

$$
\text{variant P95}<0.005^\circ
$$

和：

$$
\text{residual P95}=4.85\sim15.68\text{mm}.
$$

因此，V11.3 证明的是：

> 共享 reference 能强制各 traversal 保持同一 label identity。

但它尚未证明：

> 该 reference 是一条严格可行的物理 branch。

Gate 中的 `audit_variant_solver_success=false` 和 `audit_variant_trust_preserved=false` 正确地阻止了这种结果被误判为成功。

---

## 2. BI-3 图中允许存在未经充分求解的节点，而且 margin 不是硬约束

代码中的 `_candidate_layers` 有一段非常关键的逻辑：

> 如果不同 null-space seeds 经过 corrector 后坍缩成的 distinct candidates 少于最低候选数，就保留未充分修正的 bounded predictor / null seeds，以免 candidate layer 少于 (K)。

也就是说，图层中的部分节点可以是：

* residual 较大的节点；
* 位于 joint boundary 的节点；
* 仅被 clip 到 bounds 的 predictor；
* 不是严格 IK 解的节点。

随后 BI-3 的 unary cost 只有：

$$
C_i(k)
======

e_{i,k}^2
+
\lambda_{\mathrm{ref}}
d_W^2(\beta_{i,k},\beta_{\mathrm{ref},i})
$$

它没有将以下条件作为硬节点准入条件：

$$
e_{i,k}\le3\text{mm}
$$

$$
m(\beta_{i,k})\ge1.5^\circ
$$

所以图优化可能为了降低 reference、velocity、acceleration 和 closure 总成本，选择 residual 或 margin 不合格的节点。

这与 Formal 结果完全吻合：

* 图 `success=true`；
* 每层有 8–11 个候选；
* 但选出的环仍有 margin (0^\circ)、residual 超标和 (6^\circ\sim10^\circ) 的局部跳变。

因此，目前还不能从 BI-3 失败推出：

> 当前中心和平面不存在严格可行闭环 branch。

现有实验只证明：

> 当前的软代价 candidate graph 没有找到一条满足全部严格 Gate 的 branch。

因为实验尚未检查“只保留严格可行节点后的图”到底是：

* 仍有闭环；
* 某些 phase 为空；
* 还是存在多个不连通的局部 chart。

---

# 三、如何正确理解“root 逆解空间高度碎片化”

## 1. 直接事实

在当前固定的 (0.5^\circ) RMS 聚类阈值下：

* A2_143：64 个 root solutions，62 个 clusters；
* A2_178：64 个 root solutions，58 个 clusters；
* 两个 selected cluster 都只有 1 个 seed 支持。

而 selected root 单点表现很好：

* A2_143 candidate 21：

$$
e=0.013697\text{mm},\quad
m=4.352229^\circ,\quad
\kappa=8.754865
$$

* A2_178 candidate 0：

$$
e=0.000323\text{mm},\quad
m=4.021413^\circ,\quad
\kappa=6.859870.
$$

A2_143 JSON 确实混写了 selected beta 与原 primary root 的指标；运行器实际使用的 selected beta 没有被这些混写指标替换，但报告和解释必须修正。

---

## 2. 不能直接把 62/58 clusters 理解成 62/58 个拓扑分离 branch

我对同一批 root candidates 做了聚类阈值敏感性复算：

|      RMS 阈值 | A2_143 component 数 | A2_178 component 数 |
| ----------: | -----------------: | -----------------: |
| (0.5^\circ) |                 62 |                 58 |
| (1.0^\circ) |                 54 |                 47 |
| (1.5^\circ) |                 43 |                 33 |
| (2.0^\circ) |                 24 |                 21 |
| (3.0^\circ) |                  1 |       2，大小为 63 和 1 |
| (5.0^\circ) |                  1 |                  1 |

最近邻 RMS 距离中位数分别约为：

* A2_143：(1.47^\circ)；
* A2_178：(1.11^\circ)。

这说明“62/58 个 cluster”强烈依赖 (0.5^\circ) 的人为阈值。

对于：

$$
F:\mathbb R^6\rightarrow\mathbb R^3
$$

在正常满秩点附近，一个固定 (xyz) 的逆像一般具有约 3 个连续自运动自由度。因此更谨慎的解释是：

> 64 个 seed 在一个较宽的 self-motion 解集上分布得很稀；以 (0.5^\circ) 为连接半径时，大部分样本无法相连。

而不是已经证明：

> 存在 62 或 58 个互不连通的物理 branch。

真正的 branch 连通性需要通过：

* null-space continuation；
* kNN/geodesic graph；
* 连续路径可达性；
* 或完整 trajectory lift；

来判断，不能只靠一个 root 点上的小阈值聚类。

---

## 3. 为什么单点 canonical score 不能保证整环可持续

当前 root score 只评价：

$$
J_{\mathrm{root}}(\beta_0)
==========================

e_0^2
+
10C_{\mathrm{posture}}(\beta_0)
+
100P_{\mathrm{margin}}(\beta_0)
+
0.001\log(1+\kappa_0).
$$

它回答的是：

> 在单独的 root 目标点 (x_0) 上，哪个 (\beta_0) 看起来最好？

但你真正需要的是：

$$
\beta(\phi)
\in
\mathcal S\bigl(x(\phi)\bigr)
$$

在整个 (\phi\in[0,2\pi]) 上连续存在，并满足：

$$
m(\beta(\phi))\ge1.5^\circ
$$

$$
e(\phi)\le3\text{mm}
$$

$$
\beta(0)\approx\beta(2\pi).
$$

随着目标沿轨迹变化，root 对应的局部解支可能：

* 逐渐接近 joint boundary；
* 与其他支汇合或分离；
* 离开安全关节域；
* 需要超过 trust radius 才能继续；
* 或者虽逐点存在，但不能形成平滑闭环。

因此：

$$
\boxed{
\text{一个优秀的 root 点，不等于一条优秀的闭环 branch。}
}
$$

这不是当前实验新发现的某个 bug，而是从“单点 IK 选择”转向“完整轨迹 path lifting”时必须面对的核心区别。

---

# 四、对 QUESTION.md 第 8 节四个问题的明确回答

## 问题 1：当前核心瓶颈是什么？

当前阻止合适数据集生成的直接瓶颈是：

> **没有找到一条在全部 720 相位上同时满足严格 tracking、joint margin、continuity、closure 和 audit solver/trust 的 canonical branch。**

更具体地说，V11.3 当前存在三个层次的问题：

1. **root selection 不面向整环可行性**；
2. **candidate graph 不是 feasibility-first graph**；
3. **reference trust 通过复制不可行 reference 来维持 identity**。

Branch identity 的大尺度漂移已经被明显降低，但 numerical feasibility 和安全余量没有同时成立。

---

## 问题 2：下一步应继续放宽 Gate，还是提升方法/模型？

明确判断：

$$
\boxed{
\text{优先提升 teacher / branch-lifting 方法，不继续放宽最终数据 Gate。}
}
$$

目前没有训练数据，因此“提高 Student 模型表现”还不是当前阶段的问题。

以下 Gate 应继续 fail-closed：

* joint bounds；
* 最小 joint margin；
* residual P95/max；
* solver success；
* full-loop continuity；
* closure；
* branch audit。

可以调整和扫描的是**搜索算法参数**：

* root seed 数；
* null-space exploration；
* root 聚类尺度；
* candidate layer 数；
* corrector budget；
* graph connectivity；
* trust radius；
* trajectory optimizer 参数。

其中 trust radius 是算法参数，不是物理安全标准。但 Pilot 中 (0.25^\circ,0.5^\circ,1^\circ) 的 BI-2 trust sweep 没有改善 residual 或 margin，因此“仅继续放宽 trust”也不是充分方案。

允许设置 diagnostic slack 来定位最小不可行程度，但 slack 结果不能进入正式数据集。

---

## 问题 3：root 碎片化与整环持续性的关系是什么？

当前证据支持：

* root 解在 (\beta) 空间中分布很宽；
* 不同 seed 会落到相距较远的冗余解；
* 单点 cost 会选到一个局部表现很好的 singleton；
* 该 singleton 对应的 continuation 在部分相位触碰 (\beta_5/\beta_6) 边界。

但当前还没有证明：

* singleton 本身导致失败；
* 其他 57/61 个 cluster 全部不能覆盖整环；
* root 解集真的有 58/62 个不连通 branch。

因此，真正缺少的实验是：

> 对多个 root 解进行 continuation-survival 检验，而不是继续分析单点 canonical cost。

---

## 问题 4：下一步如何直接服务于数据生成和模型训练？

下一阶段应改成：

$$
\boxed{
\text{cycle-first / viability-first canonical branch discovery}
}
$$

即先寻找严格可行的闭环 branch，再为它定义 canonical cost；不再先选一个单点最优 root，然后要求整条轨迹跟随它。

---

# 五、推荐下一步实验：V11.4 Full-Loop Feasible Branch Discovery

## Phase 0：修复证据与增加可观测性

首先修正 A2_143 root JSON 字段来源：

```text
selected_root_residual_mm
selected_root_margin_deg
selected_root_kappa
primary_root_residual_mm
primary_root_margin_deg
primary_root_kappa
```

下一轮必须额外保存每个 phase、每个 candidate 的：

```text
candidate_source
candidate_beta
corrector_success
residual_mm
joint_margin_deg
within_bounds
reference_gap_deg
previous_gap_deg
trust_pass
kappa
```

同时保存每个 audit variant 的：

```text
trust_violation_count
solver_failure_phase_indices
reference_copy_phase_indices
```

禁止再只保存汇总布尔值。

---

## Phase 1：root 解纤维映射，而不是固定 (0.5^\circ) 聚类

对 A2_143 和 A2_178 并行执行。

### 1.1 增加 root 解采样

比较：

$$
N_{\mathrm{root}}\in{64,256,1024}
$$

候选来源不只使用 atlas 近邻，还要包含：

* canonical primary；
* Sobol/full-beta seeds；
* root 解附近的 Jacobian null-space continuation；
* 沿 3 个 null-space 基向量的正负步进；
* 局部 predictor-corrector walk。

### 1.2 连通性审计

同时报告：

$$
\epsilon\in
{0.25^\circ,0.5^\circ,1^\circ,1.5^\circ,2^\circ,3^\circ,5^\circ}
$$

下的 connected components。

再建立 kNN/geodesic graph，判断哪些 root 解能通过一系列小步 IK 解相连。

不要再把 (0.5^\circ) component 数直接称为 branch 数。

---

## Phase 2：按短弧和整环存活率选择 root

这是下一轮最关键的实验。

对每个 root representative，在正、反两个方向运行：

| 层级 | continuation 长度 |
| -- | --------------: |
| L1 |      (15^\circ) |
| L2 |      (30^\circ) |
| L3 |      (60^\circ) |
| L4 |      (90^\circ) |
| L5 |     (180^\circ) |
| L6 |     (360^\circ) |

每个 phase 必须记录：

* residual；
* margin；
* trust；
* corrector success；
* (\Delta\beta)；
* 第一个失败 phase。

root 排序必须改为字典序：

1. 是否完成 (360^\circ)；
2. strict-feasible phase 比例；
3. 最小 margin；
4. residual max；
5. residual P95；
6. velocity/acceleration；
7. 最后才是 posture 和 (\kappa)。

即：

$$
\boxed{
\text{full-loop viability}
\succ
\text{single-point canonical score}
}
$$

建议第一轮保留 top 16 个完整或最长存活 root lineages。

这一实验将直接检验：

* selected singleton 是否只是差的 branch；
* 其他 root 是否能覆盖整环；
* 当前 center/plane 是否根本没有严格安全 branch。

---

## Phase 3：建立“硬可行节点图”

对每一个 phase (i)，定义：

$$
\mathcal C_i^{\mathrm{hard}}
============================

\left{
\beta_{i,k}\ \middle|
\begin{aligned}
&e_{i,k}\le3\text{mm}\
&m_{i,k}\ge1.5^\circ\
&\beta_{\min}\le\beta_{i,k}\le\beta_{\max}\
&\text{corrector success}
\end{aligned}
\right}.
$$

候选来源包括：

* Phase 2 存活 lineages；
* forward/reverse continuation；
* secant/Jacobian predictor；
* local null-space walk；
* full-beta multi-seed IK；
* 困难 phase 的 targeted global search。

每个 phase 目标保留：

$$
K\in{16,32,64}
$$

个**严格可行** distinct candidates。

关键改变：

> 如果某 phase 严格可行候选少于 (K)，就如实保留少于 (K)；绝不能再用未修正 raw seeds 填满层数。

输出最重要的诊断是：

```text
feasible_candidate_count_by_phase
empty_feasible_layer_indices
connected_component_count
```

### 解释

* 若某些 phase 为空层：当前候选生成器没有找到严格可行解；
* 对空层做 128–512 seed targeted global IK；
* targeted search 后仍为空，才有证据认为该 center/plane 在 (1.5^\circ) margin 下局部不可行。

---

## Phase 4：cycle-first 图选择

在所有 hard-feasible nodes 上建立循环图。

边可采用：

$$
E_{i}(k,l)
==========

\lambda_vd_W^2(\beta_{i,k},\beta_{i+1,l})
+
\lambda_ad_W^2(
\beta_{i+1,l}-2\beta_{i,k}+\beta_{i-1,j}
).
$$

建议先测试边剪枝：

$$
d_W\le
{1^\circ,2^\circ,3^\circ}
$$

但最终闭环必须满足现有：

$$
\Delta\beta_{\max}\le2^\circ.
$$

不再预先固定一个 canonical root。算法应：

1. 枚举每个 root-layer feasible candidate；
2. 对每个 candidate 求最优闭环 cycle；
3. 丢弃不能闭环者；
4. 对所有严格可行 cycle 再比较：

$$
J_{\mathrm{cycle}}
==================

\lambda_vJ_v+
\lambda_aJ_a+
\lambda_pJ_{\mathrm{posture}}+
\lambda_\kappa J_\kappa.
$$

也就是：

> 先由整环可行性决定哪些 root 是候选，再由 canonical cost 选其中一个。

完成后再按整条曲线距离聚类：

$$
d_{\mathrm{curve}}(B^{(a)},B^{(b)})
===================================

\operatorname{p95}_{i}
d_W(\beta_i^{(a)},\beta_i^{(b)}).
$$

真正的 branch 应由完整闭环曲线定义，而不是 root 点定义。

---

## Phase 5：带硬约束的 whole-trajectory correction

对图选出的 cycle 做连续优化：

$$
\min_{\beta_{0:N-1}}
\lambda_v\sum_i|\beta_{i+1}-\beta_i|*W^2
+
\lambda_a\sum_i
|\beta*{i+1}-2\beta_i+\beta_{i-1}|*W^2
+
\lambda_p\sum_iC*{\mathrm{posture}}(\beta_i)
$$

约束：

$$
|F(\beta_i)-x_i|\le3\text{mm}
$$

$$
\operatorname{p95}_i
|F(\beta_i)-x_i|
\le1\text{mm}
$$

$$
\beta_{\min}+1.5^\circ
\le
\beta_i
\le
\beta_{\max}-1.5^\circ
$$

$$
\beta_N=\beta_0.
$$

建议使用 constrained SQP、`trust-constr` 或 augmented Lagrangian，而不是把 margin 继续作为很小的软惩罚。

同时运行一个 minimal-slack diagnostic：

$$
\min
\sum_i
\left[
(s_i^{e})^2+
\lambda_m(s_i^{m})^2
\right]
$$

用于识别：

* 需要多少 residual slack；
* 需要多少 margin slack；
* slack 集中在哪些 phase。

slack 只用于定位几何瓶颈，不能进入训练数据。

---

# 六、实验矩阵

建议先在 180 phase 上运行：

| 实验 | root 选择                | 节点 Gate | 图选择            | whole-curve |
| -- | ---------------------- | ------- | -------------- | ----------- |
| E0 | 当前单点 root              | soft    | 当前 BI-3        | 无           |
| E1 | 当前 root                | hard    | cycle graph    | 无           |
| E2 | viability-ranked roots | hard    | cycle-first    | 无           |
| E3 | viability-ranked roots | hard    | cycle-first    | constrained |
| E4 | all viable roots       | hard    | top-(M) cycles | constrained |

候选数：

$$
K\in{16,32,64}.
$$

root seeds：

$$
N_{\mathrm{root}}\in{256,1024}.
$$

选出 180-phase strict-pass 方法后，再固定配置运行 720-phase Formal。

A2_143 与 A2_178 均应保留：

* A2_143 当前 residual 更接近 Gate，但有单相位 cluster 和 acceleration 问题；
* A2_178 branch audit 更干净，但 residual 和局部大跳变更严重。

---

# 七、结果对应的决策边界

## 情况 A：找到至少一条严格可行闭环

要求：

$$
\mathrm{residual}_{95}\le1\text{mm}
$$

$$
\mathrm{residual}_{\max}\le3\text{mm}
$$

$$
m_{\min}\ge1.5^\circ
$$

$$
\Delta\beta_{95}\le1^\circ
$$

$$
\Delta\beta_{\max}\le2^\circ
$$

$$
\Delta^2\beta_{95}\le0.5^\circ
$$

$$
\mathrm{seam}\le1^\circ
$$

并通过 traversal/cut solver、trust 和 cluster audit。

此时才授权：

1. 生成 (\pm0.5)mm tube；
2. 再扩展到 (\pm1)mm；
3. 审计 tube 内 branch 一致性；
4. 生成训练/validation/test；
5. 启动 (xyz\to\beta_6) Student。

---

## 情况 B：每个 phase 都有严格可行点，但没有严格闭环 cycle

这表示逐点可达，但连续 lifting 受阻。

下一步应考虑：

* 重新搜索 center/plane；
* multi-chart teacher；
* stateful mapping：

$$
(x_t,\beta_{t-1})\to\Delta\beta_t;
$$

* 或显式 branch/chart context。

不能将所有逐点解混成静态 (xyz\to\beta) 标签。

---

## 情况 C：少数 phase 在 targeted global IK 后仍没有 strict-feasible candidate

这支持：

> 当前 center/plane 与 (1.5^\circ) safety margin 不兼容。

此时应重搜椭圆 center、plane orientation 或 radius，而不是放宽 Gate。

如果项目认为 (1.5^\circ) 本身过于保守，需要另立一个预注册安全协议和物理依据；不能根据本次失败事后改判。

---

## 情况 D：只有放宽 margin 或 residual 才能闭环

只能作为 evidence-only：

```text
diagnostic_reachable_but_not_dataset_eligible
```

不得进入正式 Student 数据。

---

# 八、最终建议

V11.3 的最大贡献是证明：

$$
\text{branch identity}
\quad\text{与}\quad
\text{branch feasibility}
$$

是两个不同的问题。

当前 branch identity 的大尺度漂移已经显著下降，但部分改善来自强制 reference / consensus，因此不能单凭极小 variant gap 宣称 teacher 已经正确。当前真正缺少的是一条：

$$
\boxed{
\text{严格可行、完整闭环、可重复的 canonical path lift}
}
$$

下一步最重要的算法改变不是再放宽 Gate，而是：

$$
\boxed{
\text{single-point root-first}
\quad\longrightarrow\quad
\text{full-loop viability-first / cycle-first}
}
$$

只有先证明某一 root lineage 能在全部 720 个相位上保持 residual、margin、continuity 和 closure，后续 tube、区域数据集和机器学习模型才有可信的监督目标。
---
