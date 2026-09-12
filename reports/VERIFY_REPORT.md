# 验证和验收报告（问题二论文 `paper/t2/`）

- 论文入口：`paper/t2/main.tex`（LaTeX / xelatex）
- 正文章节：`paper/t2/sections/1_restatement … 10_evaluation`、`A_code`
- 参考文献：`paper/t2/references.tex`
- 论文图：`paper/t2/figures/{t2_suitability.pdf,t2_criteria.pdf}`（由 `results/t2/` 复制）
- 结果记录：`results/t2/t2_second_site.json`（结论与自校验）、
  `reports/问题二第二检测点候选区域.md`（技术报告）
- 验收日期：2026-09-12

## 结论

**PASS**（硬错误 0 项；`writing_check.sh` 输出 `FAIL=0 WARN=0`；xelatex 编译通过、28 页 A4）。
唯一未执行的是**视觉逐页检查**，原因见下方"PDF 视觉检查"一节——当前模型不具备图像识别能力，
已用可程序化的等价检查替代。

## 检查项

| 检查项 | 结果 | 说明 |
| --- | --- | --- |
| 论文入口与章节文件 | PASS | `main.tex` 以 `\input` 引入 10 个正文节 + 附录 A，文件全部存在 |
| 章节数量与顺序 | PASS | 1_restatement → 10_evaluation → A_code，前缀顺序与标题语义一致，无重复标题 |
| 图表引用 | PASS | 2 张成果图 + 1 张 TikZ 技术路线图，均含 `\caption` 并在正文被 `\ref` 引用 |
| 图片文件存在 | PASS | `../figures/*.pdf` 相对章节文件可解析（`\graphicspath{{figures/}}` 同时满足编译解析） |
| 数值一致性 | PASS | 31 项关键数值逐一与 `t2_second_site.json` 对账，0 冲突（见下） |
| 文本质量门禁 | PASS | `writing_check.sh`：FAIL=0、WARN=0；无占位符、无内部工作流路径泄露 |
| 参考文献与引用 | PASS | 6 条真实文献，正文 `\cite` 标记 11 处，全部可对应 |
| 编译 | PASS | `xelatex` 跑两遍，重复编译无 Overfull/Missing character/未解析引用 |
| PDF 元数据 | PASS | 28 页、A4（595.28×841.89 pt）、863 KB、非空 |
| PDF 视觉检查 | 未执行（已替代） | 见下节；已做空白页、边缘出血、公式溢出（Overfull）等程序化检查 |
| 代码可复现性 | PASS | `T2.py` 无随机数，六个产物两次运行逐字节一致（见 §7 与 `AGENTS.md`） |

## 章节结构

| # | 文件 | 一级标题 |
| --- | --- | --- |
| 1 | `1_restatement.tex` | 问题重述与分析 |
| 2 | `2_analysis.tex` | 总体思路（含 TikZ 技术路线图，图 1） |
| 3 | `3_assumptions.tex` | 模型假设 |
| 4 | `4_symbols.tex` | 符号说明 |
| 5 | `5_model.tex` | 选点模型 |
| 6 | `6_literature.tex` | 文献判据：CRLB、GDOP 与几何稀释 |
| 7 | `7_algorithm.tex` | 算法设计 |
| 8 | `8_results.tex` | 求解结果（含图 2、图 3） |
| 9 | `9_sensitivity.tex` | 灵敏度分析与稳健性 |
| 10 | `10_evaluation.tex` | 模型校验、评价与推广 |
| A | `A_code.tex` | 附录 A 核心代码与复现 |

模板沿用本项目 `paper/t4/` 的 CUMCM 中文 LaTeX 版式（封面 + 中文摘要与关键字 + 目录 + 中文数字
编号章节 + 参考文献 + 附录），字体 `ctex fontset=fandol`。

## 图表引用

| 图 | 位置 | 文件 | 说明 |
| --- | --- | --- | --- |
| 图 1 | §2 总体思路 | TikZ 内联 | 建模与认证技术路线 |
| 图 2 | §8.4 适合度图 | `paper/t2/figures/t2_suitability.pdf` | 全域 / 放大 / 最坏情形内嵌三视图 |
| 图 3 | §8.5 文献判据对照 | `paper/t2/figures/t2_criteria.pdf` | 交会角影响曲线 + 判据一致性散点 |

表格 9 张（参数口径、最优解、候选区域、判据对照、偏差分桶、三准则对照、灵敏度、校验清单等），
全部为三线表并含表题与标签，正文均有引用。图注已按门禁提示精简到阈值以内。

## 数值一致性

以 `results/t2/t2_second_site.json` 为唯一数据源，程序化核对论文正文/表格中的 31 项关键数值
（$S_2^*$、$r^*$、$\varphi^*$、$J^*$、提升倍数、可行域面积、候选区域阈值/范围/面积、最坏情形
几何、五种判据半径与偏差、秩相关、认证差值、余量等）：**0 项冲突**。三准则对照表的三行取自
`literature_criteria.points_xy` 与 `literature_criteria.points`；灵敏度分析（可测性口径 1500 m、
距离先验 $d\le1000$ m、$\eta=5\%/2\%$）取自按 §9 所述参数改参复算的运行记录。

## 文本质量门禁

`/home/ylxc/.dsh/skills/6verity/scripts/writing_check.sh` 运行结果：`FAIL=0 WARN=0`。
过程中修复的问题：

1. **图片路径不可解析**（原 `../../results/t2/*.pdf`）：改为论文自己的图目录
   `paper/t2/figures/`，引用 `../figures/*.pdf`，并在 `main.tex` 加 `\graphicspath{{figures/}}`，
   使"相对章节文件可解析"与"xelatex 按工作目录解析"两个口径同时成立；重新生成结果后需同步
   复制该两个 PDF。
2. **正文无引用标记**：在 §1、§6、§8、附录 A 加入 11 处 `\cite`，覆盖 6 条参考文献。
3. **列表过密**（§10 原有 4 个列表）：把"模型优点""推广"改为连贯散文，保留"局限"为列表。
4. **图注过长**：两张成果图图注精简到门禁阈值内（信息移入正文）。

## 编译

```bash
cd paper/t2 && xelatex -interaction=nonstopmode main.tex   # 跑两遍
```

第二遍输出：28 页，无 `Overfull \hbox`、无 `Missing character`、无 `LaTeX Warning: Reference … undefined`。
首轮曾出现两处问题并已修复：`\lesssim` 未定义（补 `amssymb`）；表格超出页宽 100 pt
（三张表加 `\small` 并收窄列间距、缩短表头）；代码清单中 ε/γ/σ/R₁ 等字符在等宽字体缺字
（改为 ASCII 记号，避免缺字）。

## PDF 视觉检查

**未执行逐页人眼/RGB 视觉检查**：当前模型（deepseek-v4-flash）不具备图像输入能力，无法查看渲染
结果。作为替代，已执行以下可程序化检查：

| 程序化检查 | 结果 |
| --- | --- |
| 页数与页面尺寸 | 28 页，A4 595.28×841.89 pt |
| 空白页 / 缺页 | 无（28 页均有墨迹，逐页墨迹量 > 阈值） |
| 页边出血 | 页面最外 1.2 cm 带内无墨迹（无内容越出页边距） |
| 公式与表格溢出 | 编译日志 Overfull 计数 0（LaTeX 的溢出判据） |
| 缺字 / 乱码 | 编译日志 Missing character 计数 0 |
| 交叉引用 | PDF 文本层无 `??`，图表/公式编号与正文引用一致 |
| 图片真实嵌入 | 图 2/图 3 所在页的文本层含图内文字（如"可行域（透镜）"），证明矢量图已嵌入而非占位 |
| 结构完整性 | 封面、摘要页、目录、参考文献、附录 A 均在（目录页列出至附录 A） |

仍建议由人类（或具备视觉能力的模型）抽查一次图 2/图 3 的版面细节（figures 自身的文本越界/重叠
此前已用包围盒 + 像素复核法检查，未发现真实重叠）。

## 仍需处理的问题

1. **论文图与结果图同步**：`paper/t2/figures/*.pdf` 是 `results/t2/*.pdf` 的复制件；重新运行
   `T2.py` 后需重新复制（已在 `main.tex` 与附录 A 注明）。
2. ~~`paper/t4/` 的图路径 (`../figures/…`) 指向的文件当前不存在~~ → 已解决（2026-09-13）：
   `paper/t4/figures/` 由 `paper/t4/make_figures.py` 生成，`main.tex` 改为 `\graphicspath{{figures/}}`
   + `figures/fig_t4_*.pdf`，论文已可正常编译（详见文末追加验收）。
3. 视觉逐页检查未执行（原因见上），已用程序化检查替代；如需最终交付，建议人工抽查一次。

---

# 追加验收：`paper/t3/` 与 `paper/t4/`（2026-09-13 全量重跑后）

## 结论

**PASS**。两篇论文的正文数字与当前求解产物逐项对齐，编译零错误、无缺图/未定义引用，
两题的自检脚本全部通过，`T3.py/T4.py --plan-only` 产物与提交前逐字节一致。

## 编译

| 论文 | 引擎 | 结果 | 页数 | 警告 |
| --- | --- | --- | --- | --- |
| `paper/t3/main.tex` | xelatex ×3 | rc=0 | 9 页 A4 | 仅 hyperref 的 PDF 书签字符串警告（数学符号，历史存在） |
| `paper/t4/main.tex` | xelatex ×3 | rc=0 | 21 页 A4 | 无（缺图 / 未定义引用 / 找不到文件均为 0） |

图片真实嵌入证据：`paper/t4/main.pdf` 第 15 页含 2 个位图对象（两张 NN 对比 PNG，2307×1204 与
1659×911），6 张矢量 PDF 图在文本层可检索到图内文字；`pdftotext` 抽查新数字（`6445`、`99.9891`、
`6206.6`、`256/256`）均已在 PDF 中。

## 数值一致性（论文 ↔ 产物）

| paper/t4 说法 | 产物来源 | 一致 |
| --- | --- | --- |
| 20 局平均虚拟 6 445 s（5 580--7 861）、里程 24 392 m、测向 247 次/局、顺路 8.6 个 | `results/t4/t4_survey.json`（20 局同 seed 复算） | ✓ |
| 检测扫描 16 996 m、20 个测量位置、定案顺序 | `results/t4/t4_sweep_plan.json` | ✓ |
| 听率 99.9891%（4 224 640 例漏 460：MC 320 / 贴边 0 / 精细 140） | `t4_sweep_plan.json.verification` + `cumcm.t4.sweep` | ✓ |
| 定位误差 均值 7.66 m / 最差 19.67 m、256/256 全清、首听 ≤19 步 | `t4_survey.json` | ✓ |
| 布局裁决表 人工 20 点 6 445 / 人工 23 点 6 688 / NN 12 外圈 6 947 / NN 11 外圈 6 724（255/256）/ NN 自由 7 697 s | `results/t4/.nn_arm_speed.json`（`T4.py --practice 20 --layout-file` 同口径实测） | ✓ |
| 听率 99.9974% / 99.9218% / 99.7778% / 99.9419% | `verify_hearing_stats` 对同一 4 224 640 例口径重算 | ✓ |

| paper/t3 说法 | 产物来源 | 一致 |
| --- | --- | --- |
| 7 站均匀正七边形（$r=1000$ m）、最坏最近距离 1000.0 m @ 原点、余量 0 | `results/t3/t3_cover_plan.json` + `cumcm.t3.covering` 自检 | ✓ |
| 站点坐标表（7 行） | `t3_cover_plan.json.centers` | ✓ |
| 巡视里程 6 206.6 m、顺序 $2\to1\to0\to6\to5\to4\to3$ | `t3_cover_plan.json.survey_length_m / survey_order`（与 Held--Karp 精确枚举 7! 的最优值一致） | ✓ |
| 20 局 256/256、3 590 s（2 558--4 312）、13 980 m、121 次/局、定位误差 7.69/19.24 m、顺路 110/25 | `results/t3/t3_survey.json` | ✓ |

## 自检与不变量

- `python -m cumcm.t4.sweep`：批量判据 vs 单例参考逐例一致；4 224 640 例漏 460（99.9891%）、贴边 0 漏 ✓
- `python -m cumcm.t3.covering`：均匀 7 点布局（正七边形 / 零余量 / 任意旋转不变 / 里程 6 206.6 m）全部通过 ✓
- `python -m cumcm.analysis.undefined_names`：46 个文件无漏定义/缺失参数 ✓
- `T3.py --plan-only`、`T4.py --plan-only` 产物 sha256 与改动前一致（本次只加 `plan_from_points` 复用与
  `--layout-file` 入口，不改默认路径）✓

## 论文图

`paper/t4/figures/` 8 张图全部由 `paper/t4/make_figures.py` 从当前产物生成（图内数字不写死，
PDF 去时间戳可逐字节复现）；每张图的图例/标题文字均由脚本按产物动态拼装。视觉细节抽查见下节。
