#!/usr/bin/env python3
"""T4.py —— 问题 4：机器狗搜索与清除干扰源（定向 + 全向混合，确定性策略）的顶层入口。

本文件只是薄入口，真正的实现在 `cumcm/t4/` 包里：

    cumcm/t4/config.py     全部可调参数与物理常量（含定向源波束角、拖网栅距）
    cumcm/t4/sweep.py      寻向拖网：保证命中任意半圆盘的检测路点设计与 131 万算例校验
    cumcm/t4/regions.py    DirProbRegion：定位区域（楔形交 + 接收半径内包，无 no_signal 硬约束）
    cumcm/t4/probing.py    按文献准则（Fisher σ / 交会几何）选补测点（与 t3 同源）
    cumcm/t4/strategy.py   RobotDog：寻向拖网 + 定位清除的全部决策
    cumcm/t4/report.py     真值核对（检测保证 + 清除误差）、整批汇总、结果落盘
    cumcm/t4/plotting.py   逐局轨迹图（含定向源波束扇形）与逐步拖网扫描图
    cumcm/t4/cli.py        运行编排与命令行入口

与问题三的差异，一句话：**问题四引入定向干扰源（只在定向方向 ±90° 内有信号），"搜索不到"
不再等价于"源太远"，而可能是方向不对**。因此：

* 检测阶段从"7 点覆盖圆心巡视"换成**寻向拖网**：700 m 栅格（取到半径 2270 m，共 37 点），
  使**任意**半圆盘（任意中心/朝向/半径 ≥ 1000）都被至少一个测量点命中 —— 131 万算例、0
  失败（最坏命中深度 0.792）。这保证任何源在拖网结束前至少被听到一次。
* 定位区域不再把 no_signal 当作"源在接收半径之外"的硬约束（它可能是方向不对）；
* homing 兜底增加"起点在波束背面时先朝测量点质心回撤"的修正。

用法（与 T3.py 一致，**不加参数即连官方模拟器**）：

    python T4.py                                 # 官方模式：连 http://127.0.0.1:2026
    python T4.py --practice 1                    # 本地演练 1 局（jammers-py 问题四场景，带真值）
    python T4.py --practice 10 --seed 0          # 10 局，seed 0~9（逐字节可复现）
    python T4.py --practice 3 --survey-only      # 只做阶段一（寻向拖网）
    python T4.py --practice 5 --no-reuse --console-port 8095 --robot-port 2027
                                                 # 另起独占演练实例（并行时避免抢端口）
    python T4.py --plan-only                     # 只求拖网方案 + 半圆盘命中校验，不连模拟器

官方模式与演练写同一目录 `results/t4/`（--save-dir 可改）：一个结果目录 = 最新一次运行。
接口调用全程落盘到 `<save-dir>/api_calls.jsonl` —— 官方模式拿不到真值，这份逐次请求/响应的
记录就是唯一的证据链。
"""

from cumcm.t4.cli import main

if __name__ == "__main__":
    raise SystemExit(main())