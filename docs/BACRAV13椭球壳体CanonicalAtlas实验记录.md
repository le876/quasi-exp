# BACRA V13 椭球壳体 Canonical Atlas 实验记录

日期：2026-08-01\
实验目标：生成末端位置分布具有明确物理厚度、canonical 运动策略一致且可由单个 Student 学习的数据集，并在同一几何域上验证多尺度、多轴比、多中心偏移和多姿态椭圆泛化。

## 1. 最终结论

本轮主目标已达到，正式 claim 限定为：

```text
simulation_ellipsoidal_shell_known_chart_dataset
simulation_formal_shell_known_chart_student_learnability
simulation_locked_student_analytical_ellipse_generalization
```

正式数据不是历史椭圆轨迹的膨胀走廊，而是先定义中心椭球，再沿物理法向构造 `rho ∈ [-20, 20] mm` 的显式三维壳体。正式结果为：

- 中心面面积覆盖率 `100%`；same-chart 壳体体积覆盖率 `100%`；
- 面积加权总厚度 `min/P10/P50/P90 = 20/20/20/20 mm`，即每个保留面元都有至少 `±10 mm` 的连续法向厚度；
- 数据集 `200,000 × 36`，density CV `0.0962`，fill-distance P95/max `2.209/3.534 mm`；
- canonical 标签 parent gap P95/max `0.0407/0.2703°`，Teacher FK residual P95/max `0.877/2.171 mm`；
- 三个相同架构 Student 在随机 sealed shell points 上的最差 FK P95/max 为 `0.369/2.960 mm`；
- 最终封存的 80 条解析椭圆覆盖 `16.71–116.25 mm` 半长轴（`6.956×`）、轴比 `0.457–0.979`、12 个姿态 bin 和 4 个 offset bin；
- 80 条 sealed 椭圆中 `69/80 = 86.25%` 在三个 seed 上同时通过；172,800 个 sealed seed-point 的相对半长轴误差 P95/max 为 `0.8686%/1.5891%`。

所有正式 Gate 均通过，但 `deployment_claim_gate_pass=false`。本轮仍使用已知 `chart_id`，不声称自动 chart routing 或真实机器人部署。

## 2. Fixed point 与代码 lineage

隔离 worktree：

```text
/mnt/ML_projects/quasi_exp/.worktrees/bacra-v13-ellipsoidal-shell-atlas
```

分支：

```text
codex/bacra-v13-ellipsoidal-shell-atlas
```

基线：

```text
8afde48d8642defff7738de9aa31cc75826ca49c
```

实现提交：

| Commit | 作用 |
|---|---|
| `a451b9c` | 新增显式椭球壳体、icosphere、atlas、radial labeling 和 dense dataset pipeline |
| `2be4e25` | 调整 smoke 壳体搜索参数 |
| `7e495db` | scientific Gate 失败时 fail closed，进程返回 `2` |
| `b9df691` | 修复 dense 全局截断破坏 cell-volume quota 的问题，支持 frozen-attempt replay |
| `77f23d5` | 新增三 seed formal shell Student 训练、model lock 和 sealed evaluation |
| `efedd7b` | 新增 locked Student 的解析椭圆平面截线泛化评估 |
| `301118d` | 将边际覆盖种子置于 sealed catalog，并使覆盖 Gate 直接作用于 sealed cycles |

正式壳体搜索和原始 attempts 绑定 `7e495db80096f89fe68971fe6c4e15bab0d2b92e`；dense quota replay 绑定 `b9df6917171c636eefa43b11912941fcb38522d8`；最终椭圆 retry1 绑定 `301118d494325442d8d0832da020ab9ee5d59d2d`。

主工作树 `/mnt/ML_projects/quasi_exp` 的既有 dirty 修改未被加入本分支；全部实现、测试和提交均在上述隔离 worktree 中进行。

## 3. 方法

### 3.1 显式壳体几何

中心面定义为旋转椭球：

```text
s(u) = c + Q D u,  u ∈ S²
```

壳体点定义为：

```text
x(u, rho) = s(u) + rho n(u)
```

其中 `n(u)` 是椭球物理单位法向。正式 mesh 使用 subdivision-4 icosphere，共 `2,562` 个顶点和 `5,120` 个三角面。每个三角面和相邻 radial levels 构成曲边近似的 triangular prism，并拆成 tetrahedra 计算真实物理体积；dense quota 按这些 cell 的体积分配，而不是按历史轨迹密度分配。

正式搜索得到：

```text
center_m    = [1.0602209133, -0.0602444198, -0.3734256326]
semiaxes_m = [0.1162773412, 0.0666491111, 0.0519067561]
```

### 3.2 canonical atlas 与一致性

每个表面节点从完整 capability beta pool、已知 Chart A/B anchors 和相邻 surface nodes 取得多个候选。候选通过 predictor-corrector continuation 建立 task/configuration product graph；相容前沿合并，不相容前沿保留不同 `chart_id`。

中心面通过后，每个 radial point 同时使用多个独立 parent 做 corrector。只有 residual、actual bounds 和 parent-candidate gap 同时满足阈值才接受。因此标签定义的是同一局部 canonical branch 的连续延拓，而不是每个 XYZ 独立求任意 IK。

atlas 提取了 8 个已知 charts；`chart_00` 和 `chart_01` 都覆盖完整中心面，其他 charts 只保留局部重叠。最终 dense 数据选择 `chart_00` 的一致分支覆盖整个壳体，因此本次正式数据实际只有一个 `chart_id`。训练代码仍使用任意 K 的 one-hot known-chart 输入，不把当前 K=1 误写成自动 router 成果。

### 3.3 dense 数据生成和 split

正式 run 先生成 `400,000` 条全部接受的 attempts。最初实现按 cell 产生 quota 后又对全表全局取前 `200,000` 行，破坏了体积分层配额；该 run 因 density CV `0.3595 > 0.20` 正确失败。

修复后只重放 frozen attempts 的 deterministic exact-volume quota selection：

- 不重新求解 Teacher；
- 不增加 padding；
- 不改变 shell、atlas、radial 或 dense Gate 阈值；
- 锁定原 attempts SHA256；
- 重新计算 split 和所有 dense metrics。

最终 `200,000` 行按 cell hash 分成：

| Split | Rows |
|---|---:|
| train | 139,991 |
| validation | 29,980 |
| sealed | 30,029 |

schema 共 36 列，显式保存 `shell_id`、`chart_id`、`surface_id`、`face_id`、`radial_interval_id`、`cell_id`、单位球坐标、barycentric 坐标、`rho_m`、物理法向、XYZ、六维 beta、parent ids、candidate gaps、residual、quality、overlap 和 split。

质量分布：`195,712 RegionGold + 4,288 RegionSilver`。

### 3.4 Student

Student 保持既有实验风格，不同时引入新模型因素：

- 输入：`(x, y, z, one_hot(known_chart_id))`；
- hidden units：`[128, 128, 64]`，GELU；
- 输出：bounded tanh `beta1..beta6`；
- loss：beta loss + `lambda_fk=1.0`；
- seeds：`20260738, 20260739, 20260740`；
- model selection 只读取 train/validation；
- 三个模型及 validation reports 全部写入 model lock 后才首次读取 sealed parquet；
- 不训练 automatic chart classifier。

### 3.5 解析椭圆封存评估

平面与中心椭球的解析交线直接给出完整椭圆。候选通过 scramble Sobol 生成 plane normal 和 offset；从 4,096 个候选中先显式保留半长轴极值及所有 size/axis-ratio/orientation/offset 边际 bins，再用标准化几何特征的 deterministic maximin 选出 120 条。

- 80 条 sealed：边际覆盖种子和前续 maximin 样本；
- 40 条 development：其余样本，只作诊断；
- 每条 720 点；
- 三个 model bytes 在预测前逐一复核 SHA256；
- 通过条件：每个 seed-cycle 的相对半长轴误差 P95 `≤1%` 且 max `≤2%`；
- 总 Gate：至少 `80%` sealed cycles 在三个 seeds 上同时通过，每个非空 sealed bin 的 seed-cycle 通过率至少 `60%`。

Student 与 model lock 在 ellipse catalog 生成前已经固定，ellipse runner 不包含 `.fit()`。

## 4. 正式结果

### 4.1 Shell search

| 指标 | 结果 | Gate |
|---|---:|---:|
| center support | `97.9703%` | pass |
| `-10 mm` support | `98.7510%` | pass |
| `+10 mm` support | `94.4575%` | pass |
| connected supported surface | `95.0146%` | pass |
| mesh | `2,562 vertices / 5,120 faces` | registered |

### 4.2 Surface atlas 与 radial shell

| 指标 | 结果 | Gate |
|---|---:|---:|
| surface union | `100%` | pass |
| known chart count | `8` | evidence |
| ambiguous known-chart overlap | `0` | pass |
| accepted radial labels | `47,187` | evidence |
| same-chart surface area | `100%` | pass |
| same-chart shell volume | `100%` | pass |
| shell physical volume | `0.0030336009 m³` | evidence |
| center surface area | `0.0742424892 m²` | evidence |
| total thickness min/P10/P50/P90 | `20/20/20/20 mm` | pass |
| radial parent gap P95/max | `0.0632/0.7241°` | pass |

这里的 `20 mm` 是已接受的内外总厚度，不是单侧 `20 mm`。正式 radial levels 虽然尝试到 `±20 mm`，最终连续 same-chart 体积 Gate 支持的是总厚度 `20 mm`，对应至少 `±10 mm`。

### 4.3 Dense dataset

| 指标 | 首次 formal | frozen-attempt replay | Gate |
|---|---:|---:|---:|
| attempts SHA256 | `dfcc5320...ad14` | 同一字节 | locked |
| attempts accepted | `400,000/400,000` | 同源 | pass |
| written rows | `200,000` | `200,000` | pass |
| density CV | `0.359456` | `0.096193` | replay pass |
| fill distance P95/max | `2.249/3.441 mm` | `2.209/3.534 mm` | pass |
| parent gap P95/max | `0.0406/0.2703°` | `0.0407/0.2703°` | pass |
| residual P95/max | `0.8770/2.1711 mm` | `0.8774/2.1711 mm` | pass |

最终 dataset：

```text
/mnt/ML_projects/quasi_exp/runs/
  bacra_v13_ellipsoidal_shell_atlas_formal_dense_replay1_20260801/
  04_dense_dataset/A3_shell_dataset.parquet
```

SHA256：

```text
528c8a6dd902c6d665a1d28378c30eb19c6969f314a44399c18872e1a1a6f1a8
```

### 4.4 Student random shell points

Model lock SHA256：

```text
50506e7d48ed690ded0bc07d620457ca17eb49fb0acb7ffa8020d5655c4b73fb
```

| Seed | Validation rows | Validation FK P50/P95/max mm | Sealed rows | Sealed FK P50/P95/max mm | Sealed beta RMS P95° |
|---:|---:|---:|---:|---:|---:|
| 20260738 | 29,980 | `0.133/0.252/2.728` | 30,029 | `0.133/0.255/2.374` | `0.629` |
| 20260739 | 29,980 | `0.156/0.342/2.910` | 30,029 | `0.155/0.344/2.559` | `0.851` |
| 20260740 | 29,980 | `0.148/0.369/3.388` | 30,029 | `0.147/0.369/2.960` | `0.877` |

三个 seeds 的 validation 与 sealed prediction 全部在 actual beta bounds 内；Student Gate 全部通过。

### 4.5 Sealed analytical ellipses

最终正式目录：

```text
/mnt/ML_projects/quasi_exp/runs/
  bacra_v13_formal_ellipse_generalization_retry1_20260801
```

protocol closure：

| 锁定项 | SHA256 / commit |
|---|---|
| implementation | `301118d494325442d8d0832da020ab9ee5d59d2d` |
| shell dataset | `528c8a6d...f1a8` |
| model lock | `50506e7d...3fb` |
| Student summary Gate | `8290dced...d48` |
| ellipsoid definition | `b16d438b...e5b8` |

sealed 参数域与结果：

| 指标 | 结果 | Gate |
|---|---:|---:|
| registered cycles | `120 = 80 sealed + 40 development` | pass |
| points | `259,200 seed-points` | evidence |
| sealed points | `172,800 seed-points` | registered |
| phase points per ellipse | `720` | pass |
| semimajor range | `16.712–116.250 mm` | pass |
| semimajor ratio | `6.956×` | pass (`≥3×`) |
| axis-ratio range | `0.457–0.979` | pass |
| sealed orientation bins | `12/12` | pass |
| sealed offset bins | `4/4` | pass |
| all-seed cycle pass | `69/80 = 86.25%` | pass (`≥80%`) |
| sealed point relative P95 | `0.8686%` | pass (`≤1%`) |
| sealed point relative max | `1.5891%` | pass (`≤2%`) |
| minimum nonempty-bin pass ratio | `77.78%` | pass (`≥60%`) |

第一次 ellipse run 的旧 Gate 返回通过，但 sealed 80 条只覆盖 11 个 orientation bins，第 12 个只存在于 development。该结果保留在：

```text
/mnt/ML_projects/quasi_exp/runs/
  bacra_v13_formal_ellipse_generalization_20260801
```

它不作为最终结论。修复没有调整误差或通过率阈值，只改变预注册 role 分配并把覆盖 Gate 限定到 sealed catalog；retry1 覆盖 12/12 orientation bins 后仍通过。

## 5. 失败历史与处理边界

### 5.1 保留的失败/非最终产物

| Run | 状态 | 原因 | 处理 |
|---|---|---|---|
| `...shell_atlas_smoke` | failed/old | 初始 smoke 搜索不足 | 原目录保留，参数调整后运行 `smoke_retry1` |
| `...pilot_20260731` | nonfinal | overlap Gate 语义错误且旧 runner 未 fail closed | 修正语义和退出码，运行 `pilot_retry1` |
| `...formal_20260801` | failed `rc=2` | density CV `0.3595` | 保留完整结果，用同一 attempts 做 quota replay |
| `...ellipse_generalization_20260801` | superseded | sealed orientation 仅 11 bins | 固定 sealed 分层并运行 retry1 |

没有覆盖或删除上述产物，也没有通过放宽科学阈值把失败改写成成功。

### 5.2 本轮可以声称

- 已经存在一个显式坐标化、具有连续物理厚度和可计算体积的椭球壳体数据域；
- 正式数据按 shell cell 物理体积均匀分层，而不是按椭圆或历史 trajectory 密度生成；
- canonical Teacher labels 在 dense 数据上具有低 multi-parent gap，且单个固定架构 Student 能学习该映射；
- locked Student 能在该中心椭球的封存解析平面截线上跨尺度、轴比、中心偏移和姿态工作。

### 5.3 本轮不能声称

- 自动 chart classifier、未知 chart routing、跨 chart 在线切换；
- 真实机器人或 deployment readiness；
- 椭球壳体之外的任意 workspace 泛化；
- 对所有可能相交平面都成功，正式结论只覆盖预注册 strata 和 Gate；
- Q15 建议中的 A1 独立 IK、A2 单 root continuation 因果消融已经执行；
- D0/D1/D2 等样本量对照或 legacy final8 retention 已由本轮 V13 runner 重跑。

A0 的现有 V12.16C corridor formal artifact 仍作为历史 baseline；本轮集中完成目标方法 A3/D2 的构造、均匀性、一致性、Student learnability 和 sealed ellipse 泛化。A1/A2 与等样本量 D1 属于“改进原因的因果归因”，不是上述主 claim 的证据。本记录明确保留该缺口，避免把主目标通过扩大成完整消融通过。

## 6. 验证

直接相关测试：

```text
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python \
  -m pytest -q \
  tests/test_bacra_v13_ellipse_generalization_runner.py \
  tests/test_ellipsoidal_shell_v13.py
```

结果：`8 passed`。

修改公共数值逻辑后运行全量测试。标准入口：

```text
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python -m pytest -q
```

受环境中 `/home/ubuntu/.local/lib/python3.11/site-packages/tests/__init__.py` 抢占项目 `tests` namespace 影响，标准入口在 collection 阶段报 `tests.test_segmented_tension_solver` import error。没有修改 site-packages；使用 process-local namespace workaround 运行同一全量 suite：

```text
/mnt/ML_projects/conda_envs/quasi_exp_tf221_cu125_py311/bin/python -c \
'import pathlib,sys,types,pytest; \
module=types.ModuleType("tests"); \
module.__path__=[str(pathlib.Path.cwd()/"tests")]; \
sys.modules["tests"]=module; \
raise SystemExit(pytest.main(["-q"]))'
```

结果：到 `100%`，exit code `0`；只有现有 matplotlib/pyparsing deprecation warnings 和一个既有 sklearn convergence warning。

canonical longrun 最终状态：

```text
task: bacra_v13_ellipse_retry1_20260801
status: success
pid: none
rc: 0
```

## 7. 主要 artifact

| Artifact | 路径 |
|---|---|
| Formal source shell gates | `/mnt/ML_projects/quasi_exp/runs/bacra_v13_ellipsoidal_shell_atlas_formal_20260801` |
| Final replayed shell dataset | `/mnt/ML_projects/quasi_exp/runs/bacra_v13_ellipsoidal_shell_atlas_formal_dense_replay1_20260801` |
| Three-seed Student + model lock | `/mnt/ML_projects/quasi_exp/runs/bacra_v13_formal_shell_student_20260801` |
| Final sealed ellipse evaluation | `/mnt/ML_projects/quasi_exp/runs/bacra_v13_formal_ellipse_generalization_retry1_20260801` |
| Shell final Gate | `...formal_dense_replay1_20260801/05_summary/gate.json` |
| Student final Gate | `...formal_shell_student_20260801/04_summary/gate.json` |
| Ellipse final Gate | `...formal_ellipse_generalization_retry1_20260801/03_summary/gate.json` |
