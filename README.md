# 2026 CUMCM B 题：机器狗搜索与清除干扰源

半径 1800 m 的圆形作业区内分散若干全向干扰源（10~16 个），机器狗以 5 m/s 行驶、每次测向
耗时 5 s（含换频道 1 s）、示向度误差 ±1°，需在限时内定位并走到 20 m 内清除。

## 目录结构

```
T1.py / T3.py / T3_ga.py / T3_validate.py / T3_figures.py   顶层薄入口（只调用 cumcm.*，用法不变）
sim_api.py                                                  模拟器接口的兼容垫片
cumcm/
  common/    跨题通用：geometry 几何、routing 路线算子、sim_client 接口、practice_arena 演练场、
             plotting 绘图、records 落盘、paths 路径、console 控制台
  t1/        问题一：交会定位区域 TriangulationRegion
  t3/        问题三主线（确定性策略）：config / covering / regions / probing / strategy /
             report / plotting / cli
  t3ga/      问题三 GA 对照方案：config / localize / routing_ga / ga_ops / covering /
             strategy / plotting / training / cli
  analysis/  验证与出图：validate（六组独立验证）、figures（论文图表）
jammers-py/  本地演练场（复刻模拟器；data/behavior-logs/ 会随每次演练累积日志，可随时清空）
results/     运行产物，按两套方案分成两棵子树：
  t3/        确定性方案（T3.py）：覆盖圆方案、巡视统计、逐条观测、10 局轨迹
  t3_ga/      GA 对照方案（T3_ga.py）：GA 训练记录、逐局统计、接口日志、验证结果、
             20 局轨迹，以及 holdout/（泛化批）与 official/（官方模式实测留档）
figures/     论文用图表 PDF 与背后的数据 CSV
reports/     方案说明与结果报告
References/  参考文献（PDF）
Problem/     赛题材料（已在 .gitignore 中，不入库）
```

> `results/` 与 `figures/` 是**求解产物**：删掉后重跑命令即可原样重建（同 seed 逐字节一致）。
> `jammers-py/` 与 `References/` 不是代码依赖，只在本地演练/撰写论文时需要。

## 依赖与运行

```bash
uv sync                              # 或 pip install -e .
.venv/bin/python T1.py               # 问题一示例
.venv/bin/python T3.py               # 只求覆盖圆方案（不连模拟器）
.venv/bin/python T3.py --practice 3  # 本地演练 3 局（结果落 results/t3/）
.venv/bin/python T3_ga.py --practice 3        # 结果落 results/t3_ga/
.venv/bin/python T3_validate.py      # 六组独立验证（默认审计 results/t3_ga/ + 其 holdout/）
.venv/bin/python T3_figures.py       # 生成论文图表（默认读 results/t3_ga/）
.venv/bin/python -m cumcm.t3.covering   # 覆盖圆方案自检（旋转不破坏保证 / 密集扇区选向 / 最少圆数）
```

`T3.py` 的 `--no-rotate` 可关掉"起始扫描后把覆盖圆环转到源最密集方向"这一步，用于对照实验。

**为什么是 7 个覆盖圆**（而不是更少）：半径 1000 m 的圆盘覆盖半径 1800 m 的圆域，属于经典的
*disk covering problem*。k 个等半径圆盘覆盖一个圆所需的最小半径比 ρ_k 已有证明：ρ_5 = 0.6093829
（Bezdek 1983）、ρ_6 = 0.5559052（Bezdek 1979）、ρ_7 = 0.5。本题 r/R = 1000/1800 = 5/9 ≈
0.5555556 落在 ρ_7 与 ρ_6 之间 ⇒ **7 个够、6 个不够，故最少 7 个**。注意 6 个只差 0.063%：
即便六圆摆到最优也需覆盖半径 1000.629 m，比可用的 1000 m 只多 0.629 m。

**7 个圆怎么摆**：最少个数定下来之后，圆心的摆放还要在"保证不漏源"的前提下让巡视里程最短。
经典做法是"1 个中心圆 + 6 个正六边形环上圆"，其里程恒为 6d（d 为环半径），最好情形
d = d_min = 1122.9558 m 时也是 6737.73 m。但**六边形并不是最优的**：它把 1 个圆心压在原点，
而原点是覆盖效率最低的位置（离圆域边界 1800 m）；把 7 个圆心全部推到距原点约 1000 m 处
（原点自己就落在它们的覆盖范围内，无需专门的中心圆）可把里程压到 **6167.32 m**，省 570 m。
该布局由数值优化得到，并用三种互相独立的方式复核了覆盖（解析最坏点候选集、3000×20000 稠密
极坐标网格、1×10^7 点蒙特卡洛），三方一致给出最坏最近距离 **994.9998 m ≤ 1000 m**（余量 5 m）。
因此默认采用它（`config.SURVEY_CENTERS`），六边形族保留作对照（`--hex-layout`）。

同 seed 配对实测（各 10 局、均 131/131 全清）：虚拟时间 4008 → **3796 s**（省 212.1 s，5.3%，
t = 3.97），里程 16419 → **15038 m**（省 1381 m，8.4%，t = 5.90）。

分层规则见 `cumcm/__init__.py`：依赖只能向下（common ← t1/t3/t3ga ← analysis），
严禁下层反向导入上层。
