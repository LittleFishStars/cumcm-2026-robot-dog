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
.venv/bin/python -m cumcm.t3.covering   # 覆盖圆方案自检（旋转不破坏覆盖保证 / 密集扇区选向）
```

`T3.py` 的 `--no-rotate` 可关掉"起始扫描后把覆盖圆环转到源最密集方向"这一步，用于对照实验。

分层规则见 `cumcm/__init__.py`：依赖只能向下（common ← t1/t3/t3ga ← analysis），
严禁下层反向导入上层。
