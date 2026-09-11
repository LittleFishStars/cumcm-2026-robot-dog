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
jammers-py/  本地演练场（模拟器）
results/     运行产物（训练结果、接口日志、轨迹、验证结果）
figures/     论文用图表 PDF 与背后的数据 CSV
report/      论文与说明文档
```

## 依赖与运行

```bash
uv sync                              # 或 pip install -e .
.venv/bin/python T1.py               # 问题一示例
.venv/bin/python T3.py               # 只求覆盖圆方案（不连模拟器）
.venv/bin/python T3.py --practice 3  # 本地演练 3 局（自动拉起 jammers-py）
.venv/bin/python T3_ga.py --practice 3
.venv/bin/python T3_validate.py      # 六组独立验证
.venv/bin/python T3_figures.py       # 生成论文图表
```

分层规则见 `cumcm/__init__.py`：依赖只能向下（common ← t1/t3/t3ga ← analysis），
严禁下层反向导入上层。
