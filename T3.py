#!/usr/bin/env python3
"""T3.py —— 问题 3：机器狗搜索与清除干扰源（确定性策略）的顶层入口。

本文件只是薄入口，真正的实现在 `cumcm/t3/` 包里：

    cumcm/t3/config.py     全部可调参数与物理常量
    cumcm/t3/covering.py   覆盖圆求解（最少 7 个；一般 7 点最优摆放）、连续域校验、最少圆数判定
    cumcm/t3/regions.py    ProbRegion：把每次测量（含"听不到"）变成硬约束
    cumcm/t3/probing.py    按文献准则（Fisher σ / 交会几何）选补测点
    cumcm/t3/strategy.py   RobotDog：巡视扫描 + 定位清除的全部决策
    cumcm/t3/report.py     真值核对、整批汇总、结果落盘
    cumcm/t3/plotting.py   逐局轨迹图
    cumcm/t3/cli.py        运行编排与命令行入口

完整的方法说明、公式推导与实测数据见 `cumcm/t3/strategy.py` 与各模块文档字符串；
策略层面的结论性说明（为什么最少要 7 个覆盖圆、7 个圆心怎么摆最省里程、为什么"就近试清"）
保留在 `docs/` 与报告里。其中"为什么是 7 个"归到经典的 disk covering problem：用 k 个半径 r
的圆盘覆盖半径 R 的圆，所需最小半径比 ρ_k = r/R 有已证明的最优值（ρ_5 = 0.6093829、
ρ_6 = 0.5559052、ρ_7 = 0.5），而本题 r/R = 1000/1800 = 5/9 ≈ 0.5555556 落在 ρ_7 与 ρ_6 之间，
故 6 个不够（只差 0.063%）、7 个够，最少 7 个 —— 详见 `cumcm.t3.covering.min_circle_count`。

用法（**不加参数即连官方模拟器**）：

    python T3.py                                 # 官方模式：连 http://127.0.0.1:2026 跑完一局
    python T3.py --base-url http://127.0.0.1:8080   # 换地址（模拟器控制台改了端口时）
    python T3.py --practice 1                    # 本地演练 1 局（需 jammers-py/，自带真值）
    python T3.py --practice 10 --seed 0          # 10 局，seed 0~9
    python T3.py --practice 3 --survey-only      # 只做阶段一（巡视扫描 + 覆盖核对）
    python T3.py --practice 5 --no-reuse --console-port 8095 --robot-port 2027
                                                 # 另起独占演练实例（多会话并行时避免抢端口）
    python T3.py --plan-only                     # 只求覆盖圆方案，不连任何模拟器

三种模式共用同一次覆盖圆求解与同一份策略代码，差别只在"场景从哪来"与"结果写哪去"：

| 模式 | 触发 | 场景来源 | 真值 | 结果目录 |
|---|---|---|---|---|
| 官方 | 不加参数（或 --base-url） | 官方模拟器 | **不可见** | `results/t3/` |
| 演练 | `--practice N` | jammers-py（自动拉起） | 可见，可核对覆盖保证 | `results/t3/` |
| 仅求解 | `--plan-only` | 无 | — | `results/t3/` |

官方模式与演练**写同一目录** `results/t3/`（--save-dir 可改）：一个结果目录 = 最新一次运行，
不按模式分家。因此官方跑一局就会覆盖该目录里的演练批产物 —— 想跑验证或出论文图，需先跑一次
演练批（`--practice 20 --seed 0`，逐字节可复现）。产物里记了 `meta.mode` 说明来源。
接口调用全程落盘到 `<save-dir>/api_calls.jsonl`（--api-log 改路径、传空串关闭）—— 官方模式
拿不到真值，这份逐次请求/响应的记录就是唯一的证据链。

**图与日志只保留最新一局**（每局开头清空重写，避免新旧混在一起难以分辨）：

| 产物 | 内容 |
|---|---|
| `<save-dir>/trajectory/epNN_seedM.png` + `.csv` | 该局的**总轨迹图**（整局行驶路径 + 全部动作点）与同名轨迹表 |
| `<save-dir>/scan/epNN_seedM_s00_起点全频道扫描.png` | 第 0 步：出发点全频道扫描的结果图 |
| `<save-dir>/scan/epNN_seedM_sNN_…png` | 其后每一步巡视扫描各一张（该步测向点、示向度射线、当时的可能源区域与已清除数） |
| `<save-dir>/api_calls.jsonl` | 该局逐次接口调用（含原始响应） |
| `--log FILE` | 该局的过程日志（缺省不写文件，只打终端） |

逐步扫描图的价值在于：整局的轨迹图信息密度太高（上百次测向挤在一张 3600 m 见方的图上），
而策略的全部信息都来自这一步步扫描 —— 每步单独出图才能看清"哪一步听到了什么、区域收缩到
什么程度"。所有图都在 `/exit` 之后生成，不占用现实时间预算（--no-plot 关闭出图）。

官方模式运行环境见 `reports/OFFICIAL_PLATFORM_GUIDE.md`（模拟器只监听 127.0.0.1，须同机运行）。
"""

from __future__ import annotations

import sys

from cumcm.t3.cli import main

if __name__ == "__main__":
    sys.exit(main())
