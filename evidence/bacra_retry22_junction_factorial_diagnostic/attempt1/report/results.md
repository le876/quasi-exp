# retry22：交界加采样与去 β 标签项的对照结果

## 结论

2026-09-13，完成用户授权的四组×两种子对照。**训练抽样分布是接缝失败的可干预因素，但当前加采样配比把部分失败移到了别处，并损失了原近轴精度；单独去掉 β 标签项没有稳定收益。** 没有组同时满足“两种子严重坏点至少减半且八条近轴路径仍全点≤3mm”，不替换现有模型，不追加训练预算。

数据坐标、标签、split、网络和训练步数在各组间保持一致。这个结果支持关注两域衔接和训练曝光；它不证明几何缺点是唯一原因，也不证明现有标签都错误。当前实验没有扩充空间范围或壳厚度。

## 实验与公平性

协议：[41-交界采样与标签牵制对照](protocols/41-BACRA-retry22-交界采样与标签牵制对照协议.md)。源码 `e63318d68016d8a60f35c832318c03a729d9fdd3`，执行绑定 `f94d8743ee9cf35f25f6c6f26d2c8e9bed1e1190`。运行目录 `runs/bacra_retry22_junction_attempt1`；独立 launcher task `bacra_retry22_junction_attempt1`，2026-09-13 01:10:15 至 01:34:40，success/rc0，科学程序耗时 1461.588 秒。

保留 L1 的 74,696 行（train 58,629 / validation 7,795 / test 8,272）、35条路径与3,000个off-grid test。两seed20260925/20260926，每seed共享500步warmup后分叉为四组，各继续至10,000步，共77,000次optimizer更新。预检六文件有效diagnostic_only，学习成功not_bounded；formal/global replacement均false。

四组为control、junction、task_only、junction_task_only。junction从 `X=1015.498±60mm` 的train池抽128行，替换原batch中128行uniform；其余core205/transition256/tip154保留。task_only在501步起把组合loss的β系数从.01改为0。零位结构、FK任务项、模型尺寸、Adam及学习率一致。

每seed四组初始网络/Adam/采样state SHA一致；相同采样的不同loss组完整batch序列SHA一致。两个control与retry21对应种子的完整on-grid预测β及误差均逐值相同，最大差0。所有评估使用重载final.keras，未按test选模。

## 全 test 尾部与近轴保留

以下按seed20260925 / seed20260926排列；严重坏点定义为FK误差>50mm，分母均为8272。

| 设置 | >50mm点数 | test最大误差mm | 近轴960点最大误差mm | 近轴≤3mm点数 |
| --- | --- | --- | --- | --- |
| control | 27 / 33 | 125.648 / 125.116 | 2.572 / 1.906 | 960 / 960 |
| junction | 9 / 15 | 83.445 / 124.595 | 5.181 / 3.500 | 697 / 947 |
| task_only | 24 / 36 | 132.389 / 125.545 | 2.726 / 1.996 | 960 / 960 |
| junction_task_only | 9 / 26 | 83.914 / 112.848 | 4.540 / 2.089 | 782 / 960 |

junction把严重坏点总数减少66.7%/54.5%，但两seed的近轴≤3mm保留都失败。组合干预在seed1与junction相近，seed2只减少21.2%的严重坏点；说明存在种子与干预之间的交互，不能把两个主效应简单相加。候选判据并未通过。

## 接缝改善与失败迁移

以最近32个train邻居同时包含primary/outer的1111个test点为固定混合邻域：control有22/30个严重坏点，junction降到3/0，task_only为19/32，组合为3/24。junction在这个固定区域确实改善。

但junction原来27/33个>50mm点全部降到50mm以内后，分别新增9/15个>50mm点。seed2新增坏点在X=1045.804–1112.796mm、rho=181.126–241.054mm，已偏向primary内部。seed1新增点在X=1000.498–1068.477mm、rho=207.218–346.858mm。不能只回看原坏点而称尾部消失。

原core/transition/tip的抽样配额没有减少，但近轴仍会退化。这说明共享网络的参数更新可能在区域之间相互影响，不能把近轴退化解释为“本轮直接少抽了core”。原uniform份额确实从409降到281，这一改变可能影响全域约束；本轮未分离这种作用与增加junction曝光的作用。

去β标签项的单独干预在两seed的坏点数为24/36，对比27/33没有一致改善。由此不支持“删除β项就能稳定解决现有尾部”。这不否认Q33已经验证的合法标签平均机制；也没有证明局部分支一致性重选无用，因为去掉损失约束与重选Teacher分支是不同操作。

## 全域百分位、off-grid与大轨迹

| 设置 | on-grid test P95 mm | off-grid test P95 mm | off-grid max mm |
| --- | --- | --- | --- |
| control | 9.833 / 11.341 | 7.545 / 9.694 | 71.830 / 101.276 |
| junction | 9.669 / 11.207 | 7.788 / 9.316 | 91.284 / 77.598 |
| task_only | 10.138 / 11.147 | 7.523 / 9.498 | 61.563 / 102.069 |
| junction_task_only | 9.549 / 10.956 | 7.829 / 8.951 | 89.214 / 94.519 |

junction还改善了部分大轨迹：large_star的最大误差从55.760/47.801降到22.960/18.105mm；large_helix从68.584/42.339降到17.959/17.128mm。但large_tapered_helix最大误差从14.989/16.188升到18.156/19.507mm。完整35条路径都保留，不能只报告改善的形状。

## 对数据增厚的含义

前置[壳与接缝审计](audits/2026-09-13-retry21-shell-junction.md)已用参考FK验证55个当前X范围内、outer壳内侧的合法构形位置，说明有增厚候选。那是旧FK样本的几何见证，本轮未将其加入监督或训练。

本轮进一步说明下一步不能只增加点数：在数据坐标不变时，采样就能修复原严重坏点，也能在别处制造新尾部。因此建议后续先解决跨区域精度取舍，再用等新增样本预算比较“原域加密”和“过渡带/内侧增厚”。保留本轮揭示的新失败位置作回归诊断，并另设独立确认点/路径。扩充时必须联合检查XYZ覆盖和β分支衔接，不通过平均不同分支β制造平滑。

当前有依据推进有界增厚pilot，但尚无“增厚后现有网络能稳定覆盖全部新区域”的实验证据。

## 验证与产物

首次owning/下游检查52项通过；将artifact检查单列后6项聚焦检查通过。没有改共享FK/旧模型实现，未运行全量pytest。结构验证governance_structure通过，保留历史canonical_layer_field protocol inventory warning；结构通过不作为科学通过。

独立数值复核记录 `runs/diagnostics/retry22_junction_verification/numerical_verification.json`由 `scripts/reports/verify_retry22_junction_evidence.py` 完成：84个artifact hash、671120行预测、416行分区/路径指标、48行配对诊断指标。batch FK与记录scalar FK最大差8.882e-16m，百分位最大差7.106e-15mm；分叉与batch配对验证通过，control复现差0。

实际结果文件为 `04_evaluation/metrics.csv`、`05_diagnosis/paired_metrics.csv`、逐点预测/增量Parquet、`03_training/paired_state_proof.json`、`model_lock.json`、`summary.json`、`completion_manifest.json`。所有原始目录保持冻结。
