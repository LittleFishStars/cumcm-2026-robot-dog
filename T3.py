#!/usr/bin/env python3
"""T3.py —— 问题 3：机器狗搜索与清除干扰源（确定性策略）的顶层入口。

本文件只是薄入口，真正的实现在 `cumcm/t3/` 包里：

    cumcm/t3/config.py     全部可调参数与物理常量
    cumcm/t3/covering.py   覆盖圆求解（1 中心 + 6 环的正六边形拼接）、连续域校验、最少圆数判定
    cumcm/t3/regions.py    ProbRegion：把每次测量（含"听不到"）变成硬约束
    cumcm/t3/probing.py    按文献准则（Fisher σ / 交会几何）选补测点
    cumcm/t3/strategy.py   RobotDog：巡视扫描 + 定位清除的全部决策
    cumcm/t3/report.py     真值核对、整批汇总、结果落盘
    cumcm/t3/plotting.py   逐局轨迹图
    cumcm/t3/cli.py        运行编排与命令行入口

完整的方法说明、公式推导与实测数据见 `cumcm/t3/strategy.py` 与各模块文档字符串；
策略层面的结论性说明（为什么最少要 7 个覆盖圆、为什么取 1200 m 环半径、为什么"就近试清"）
保留在 `docs/` 与报告里。其中"为什么是 7 个"归到经典的 disk covering problem：用 k 个半径 r
的圆盘覆盖半径 R 的圆，所需最小半径比 ρ_k = r/R 有已证明的最优值（ρ_5 = 0.6093829、
ρ_6 = 0.5559052、ρ_7 = 0.5），而本题 r/R = 1000/1800 = 5/9 ≈ 0.5555556 落在 ρ_7 与 ρ_6 之间，
故 6 个不够（只差 0.063%）、7 个够，最少 7 个 —— 详见 `cumcm.t3.covering.min_circle_count`。

用法：

    python T3.py --practice 1                    # 本地演练 1 局（需 jammers-py/）
    python T3.py --practice 10 --seed 0          # 10 局，seed 0~9
    python T3.py --practice 3 --survey-only      # 只做阶段一（巡视扫描 + 覆盖核对）
    python T3.py --practice 5 --no-reuse --console-port 8095 --robot-port 2027
                                                 # 另起独占演练实例（多会话并行时避免抢端口）
    python T3.py                                 # 官方模式（连 --base-url）

结果默认写到 results/t3/（--save-dir 改），轨迹图落在其下的 trajectory/（--no-plot 关闭，
--traj-dir 改目录）。GA 对照方案的结果树在 results/t3_ga/，两者互相独立。
"""

from __future__ import annotations

import sys

from cumcm.t3.cli import main

if __name__ == "__main__":
    sys.exit(main())
