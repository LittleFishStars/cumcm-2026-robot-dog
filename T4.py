#!/usr/bin/env python3
"""T4.py —— 问题 4：机器狗搜索与清除干扰源（定向 + 全向混合，确定性策略）的顶层入口。

本文件只是薄入口，真正的实现在 `cumcm/t4/` 包里：

    cumcm/t4/config.py     全部可调参数与物理常量（含定向源波束角、中点外推系数）
    cumcm/t4/sweep.py      扫描布局：7 覆盖基点 + 21 外推中点（复用问题三布局）与听到率统计
    cumcm/t4/regions.py    DirProbRegion：定位区域（楔形交 + 接收半径内包，无 no_signal 硬约束）
    cumcm/t4/probing.py    按文献准则（Fisher σ / 交会几何）选补测点（与 t3 同源）
    cumcm/t4/strategy.py   RobotDog：扫描 + 定位清除的全部决策
    cumcm/t4/report.py     真值核对（检测保证 + 清除误差）、整批汇总、结果落盘
    cumcm/t4/plotting.py   逐局轨迹图（含定向源波束扇形）与逐步扫描图
    cumcm/t4/cli.py        运行编排与命令行入口

与问题三的差异，一句话：**问题四引入定向干扰源（只在定向方向 ±90° 内有信号），"搜索不到"
不再等价于"源太远"，而可能是方向不对**。因此：

* 检测阶段**复用问题三的 7 个覆盖基点并加密**：任意两两基点的中点沿径向外推 ×2.6（上限
  2270 m），共 29 个测量位置（原点起点 + 7 基点 + 21 中点）。中点外推补上了定向源朝向
  圆域外/边缘的"迎光面"——这是"尽量复用问题三布局"的轻量折中，实测听到率 ~99.8%
  （200 万随机算例 + 31 万对抗算例，见 cumcm.t4.sweep.verify_hearing_stats），
  **非严格保证**（残余漏例 = 贴边 + 波束精确朝外的源，约 0.2%）。
* 定位区域不再把 no_signal 当作"源在接收半径之外"的硬约束（它可能是方向不对）；
* homing 兜底增加"起点在波束背面时先朝测量点质心回撤"的修正。

用法（与 T3.py 一致，**不加参数即连官方模拟器**）：

    python T4.py                                 # 官方模式：连 http://127.0.0.1:2026
    python T4.py --practice 1                    # 本地演练 1 局（jammers-py 问题四场景，带真值）
    python T4.py --practice 10 --seed 0          # 10 局，seed 0~9（逐字节可复现）
    python T4.py --practice 3 --survey-only      # 只做阶段一（扫描）
    python T4.py --practice 5 --no-reuse --console-port 8095 --robot-port 2027
                                                 # 另起独占演练实例（并行时避免抢端口）
    python T4.py --plan-only                     # 只求扫描方案 + 听到率统计，不连模拟器

官方模式与演练写同一目录 `results/t4/`（--save-dir 可改）：一个结果目录 = 最新一次运行。
接口调用全程落盘到 `<save-dir>/api_calls.jsonl` —— 官方模式拿不到真值，这份逐次请求/响应的
记录就是唯一的证据链。
"""

from cumcm.t4.cli import main

if __name__ == "__main__":
    raise SystemExit(main())