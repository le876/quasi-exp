# 共享 HTML 模板：使用、理由与迭代

维护入口是 [report_template.html](../assets/report_template.html)。它是生成源文件，需要填入插槽后才能作为报告打开。生成后的 HTML 内联样式、Plotly、数据与图表脚本，可独立打开；不要把带实验数值的旧 HTML 当成新实验模板。

## 文件职责与设计理由

| 文件 | 负责什么 | 为什么放在这里 |
| --- | --- | --- |
| `assets/report_template.html` | 页面骨架、卡片/表格/响应式样式、明暗配色、系列图例、路径导航 | 同一视觉问题只修一处，重新生成时自动应用；普通 HTML/CSS/JS 可以直接检查，不增加构建框架。 |
| 本文 | 插槽用法、每一步的理由、验收和修改范围 | 让后续 AI 理解约束，而不是只模仿外观或盲目复制代码。 |
| 实验自己的 `build_report.py` | 读取证据、单位转换、确定性抽样、章节文字、实验图表、插槽填充 | 各实验的数据 schema 和科学含义不同，不能让模板替实验决定分母、模型、阈值和结论。 |
| 报告目录 | 生成 HTML、模板副本、生成器、输入与输出 hash、浏览器检查 | 保留交付物实际使用的字节，允许追溯模板修订；副本是快照，持续维护入口仍是 Skill 内的模板。 |

当前接入实例：项目根目录下 `runs/reports/bacra_retry20_candidate_diagnostic_attempt1/build_report.py`。这是本地报告生成器，目录可能被 Git 忽略；它不是共享模板的存储位置。共享模板和本文应随仓库版本管理，交付报告时同时保留生成器和模板副本。

## 生成一份报告，每一步为什么这么做

1. **绑定当前 evidence。** 确认 worktree、source/binding、attempt、artifact 与 observation date，按需核对输入 hash。目的是防止把旧实验的结论或另一个 attempt 的数据混入当前页面。局部样式修改保留原 evidence cutoff，另记 presentation 更新时间。
2. **准备当前实验的数据与正文。** 由生成器提供统一物理坐标、单位、完整统计分母、图上显示点数、路径和模型锁。复用版式不等于复用旧数值。没有证据的章节写明缺口，不为填满模板而重新训练或补造结果。
3. **按阅读问题组织正文。** 推荐顺序是实验身份与主要结论、方法/数据关系、覆盖与零位参照、训练与锁定、轨迹/误差、完整指标、限制与溯源。这样读者先知道在看什么、比较什么，再解释图与数值。章节可以随证据调整；原理、coverage、tracking、training、claim boundary 的适用内容不能因套模板而丢失。核心内容默认展开，细节溯源可折叠。
4. **读取共享模板并填充插槽。** 样式与公共交互始终从模板读取，实验脚本只提供数据和具体绘图逻辑。这样改一次图例或路径导航，接入的报告重建后就使用同一实现。
5. **用同一配色创建图形与说明。** 先取 `ReportPlot.palette()`，再把系列颜色显式传给 trace；图例从 trace 读取同一个颜色，正文中的系列名用 `data-series`。避免正文说紫色，实际图形却因未定义变量而使用 Plotly 默认色。
6. **连接路径、重绘与主题。** 下拉列表是路径选择的唯一状态来源，导航按钮修改选择后触发同一个 `change` 事件；该事件更新轨迹、当前指标与相邻步长图。主题切换先换 palette，再更新正文系列说明和所有已绘制图。视口外的图下次渲染时读取新 palette。
7. **运行浏览器验收。** 检查的是实际文字颜色、字号、点击效果、主题、窄屏和真实曲线，而不只是 JavaScript 没有语法错误。3D 图继续按视口创建/释放，离开再返回也要检查图例恢复。截图用于发现裁剪、重叠或错误相机视角。
8. **封装报告并更新清单。** 记录共享模板与生成 HTML 的 SHA-256，浏览器记录绑定最终 HTML hash；所有检查完成后刷新清单中的衍生文件 hash。模板版本或浏览器通过只说明展示实现，不改变 scientific Gate。

## 插槽约定

插槽使用 `@@NAME@@`，一次替换完成，避免后续替换意外修改已注入的数据。

| 插槽 | 内容 | 生成器处理 |
| --- | --- | --- |
| `TITLE` | 页面标题 | `html.escape(title)` |
| `BODY` | `<main>` 内部正文，包括 header、section、图容器和控件 | 生成可信 HTML；数据中的文本单独 HTML 转义。模板已提供 `<main>`，不要重复嵌套。 |
| `PLOTLY` | 本地 Plotly 的完整 JavaScript | 内联已验证的依赖，不插入 CDN 链接；记录依赖版本或文件 hash。 |
| `DATA` | 当前报告的 JSON，模板声明为 `const D` | `json.dumps(..., allow_nan=False)`；非有限值预先变成 `null`；把 `</` 替换为 `<\/`，防止数据终止 script。 |
| `SCRIPT` | 实验自己的图构造、事件和 lazy-render 逻辑 | 普通 JavaScript，不再需要 Python f-string 的双花括号；不能含未转义的动态 HTML/script 结束标记。 |

最小填充方式（省略实验证据读取）：

```python
slots = {
    'TITLE': html.escape(title), 'BODY': body,
    'PLOTLY': plotly_js,
    'DATA': json.dumps(data, ensure_ascii=False, allow_nan=False).replace('</', '<\\/'),
    'SCRIPT': chart_script,
}
rendered = re.sub(r'@@([A-Z]+)@@', lambda m: slots[m[1]], template)
```

保持未知插槽报错；不要静默用空字符串替代，缺失图表或证据段落会因此难以察觉。

## 图例与配色接口

```javascript
let color = ReportPlot.palette();
const traces = [{
  type: 'scatter', mode: 'lines', name: 'Teacher',
  x: D.x, y: D.teacher,
  line: {color: color.teacher, width: 3}
}];
await ReportPlot.react('error-plot', traces, layout, config);
```

- 图容器使用 `.plot`，并提供 `id` 和说明性的 `aria-label`。`ReportPlot.react` 在容器下生成可换行的 HTML 图例，关闭内置 Plotly 图例，避免长标签被图框裁掉；图例文字至少 14px，颜色来自对应 trace。
- 点击图例按钮调用 `Plotly.restyle` 切换该 trace；隐藏项用划线、虚线边框和 `aria-pressed=false` 表达，不把颜色褪成不可读灰字。键盘可聚焦并用 Enter/Space 操作；重绘保留图例焦点。
- 线型和点形状也出现在图例中。颜色说明“属于哪个系列”，虚线、点线、实心/空心等编码说明“属于哪个比较条件”；颜色不能取代这些语义。
- 当前 helper 服务于明确单色的分类系列，包括 `scatter` / `scatter3d` 的线和点。每条 trace 显式提供 `line.color` 或 `marker.color`。新增连续数值着色时保留 Plotly colorbar，另行处理代表性图例，不把数值颜色数组传入此分类图例。
- 正文和开关可使用 `<span data-series="teacher">Teacher：橙色粗线</span>`。主题切换后调用 `ReportPlot.syncLabels()`，并用新 palette 重建 trace。新增角色时给 `light` / `dark` 两套 palette 都加颜色，并检查对比度。
- L0/L1 是 retry20 使用的对照层角色键，可供同语义比较复用；新实验不能只因为有两组数据就继承其科学含义。Teacher、raw、corrected 的角色也必须来自当前证据。

## 路径上下切换

在 `.toolbar` 内提供 `<label>路径<select id="trajectory">…</select></label>`，填充 option 并设定初始选择后调用：

```javascript
ReportPlot.pathNavigation('trajectory');
document.getElementById('trajectory').onchange = renderSelectedPath;
```

模板生成“↑ 上一条”“↓ 下一条”和 `当前 / 总数`。顺序就是已有 option 顺序；首尾禁用相应按钮，不循环跳回，避免误把第 16 条之后当成第 17 条。保留下拉列表用于直接定位。切换时保留用户当前投影和系列开关，除非实验页面明确要求重置。初始化只调用一次；后续由用户操作或程序派发 `change` 同步显示位置。

## 固定大尺度测试轨迹，为什么需要独立资产

[q31_large_shapes_v1.json](../assets/q31_large_shapes_v1.json) 固定 Q31 更新后真实评估过的三条路径，共1504个有序点，坐标为机器人基座frame下的m。展示时统一乘1000转为mm；不能为适配图框移动或缩放数据。`geometry`中的尺寸和历史支持统计来自原`geometry.json`，`source`记录原输入、数值来源与verified package来源，二者不能混称同一个source。

| 固定ID | 实际几何 | 拓扑与点数 |
| --- | --- | --- |
| `large_star` | x=1030mm，外半径600mm，Y/Z跨度约1038×1095mm | 闭合，407点 |
| `large_helix` | x=1020→1140mm，半径400mm | 开放，523点 |
| `large_tapered_helix` | x=1020→1180mm，半径580→300mm | 开放，574点 |

这些路径替代“只展示靠近零位的小星形/小螺旋”的默认展示习惯。保留确切坐标比只记“画一个大星形”更可靠：不同报告的模型面对同一输入，形状规模不会因重写绘图脚本而漂移。原搜索属于有限候选搜索中的largest-found，不是数学全局最大。坐标固定也不保证它们始终“顶住”新数据集边界；新实验必须报告自己的覆盖/支持关系。

生成器读取`tracks[].waypoints_m`、`closed`与`id`，使用当前锁定模型生成新的Teacher/raw/corrected结果，并绑定套件hash。已有当前结果时直接复用；只有用户授权补做评估时才启动新计算。Q31旧模型的跟踪结果、旧support pass不能当作新实验结果。

页面把三条固定大尺度路径作为可直接选择的组，默认展示大星形，并保留原协议路径组。组切换后更新上下按钮、计数、指标和附图；不把新增3条计入原16条的冻结分母。对于完整星形，绘图和jump指标包含首尾边；两条螺旋必须保持开放，不能为“看起来完整”补一条跨越整个空间的闭合线。

验收除公共交互外，还要核对原1504个有序坐标、407/523/574点数、开闭属性、span与suite hash；检查当前模型hash、每类输出的缺失点、成功率分母和当前数据支持距离。Teacher失败保留断线；不允许用Target填充，也不允许用DLS2曲线替代raw。

## 迭代与验收范围

改变公共字体、图例、主题或导航时，修改模板；改变 artifact schema、计算或实验结论时，修改对应生成器。不要在生成的 HTML 中单独修补后丢失源修改，也不要为一个实验的特殊图添加无实际消费者的通用配置层。

每次模板修改至少用一个真实接入报告重新生成并检查：

- 每张图逐项比较图例文字的 computed color 与实际 trace 的系列颜色；检查字号、对比度、标签换行和窄屏边界。
- 明暗主题、路径切换、重置、系列开关、lazy-render 释放/重建后再检查颜色与曲线；图例本身也要点击验证。
- 路径导航验证中间位置、首尾禁用、直接下拉选择后的计数，以及轨迹/指标/附图都跟随当前路径。
- 检查控制台错误并实际看截图；确认数据和指标没有因展示重构改变。静态结构检查不能代替这些行为检查。

已交付旧报告保留原模板副本和 hash；用户要求更新某份报告时才重新生成对应衍生文件。PDF 仍按 Skill 的 opt-in 规则处理，不能把未验证的 WebGL 页面直接当作可靠打印版。
