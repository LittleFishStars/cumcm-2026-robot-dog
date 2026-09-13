"""t3：问题 3 的求解代码，本包是主线，入口是 `python -m t3`

模块划分如下。依赖只能向下，同层之间按箭头顺序：

    config.py     全部可调参数与物理常量
    covering.py   覆盖圆：1 中心 + 6 环的正六边形拼接、解析与数值校验
    regions.py    定位区域 ProbRegion：楔形交 ∩ 圆域，再叠加测量给出的圆盘硬约束
    probing.py    按文献准则选补测点（Fisher σ / 交会几何）
    strategy.py   RobotDog：巡视扫描 + 定位清除的全部决策
    report.py     真值核对、整批汇总、结果落盘
    plotting.py   逐局轨迹图
    cli.py        运行编排与命令行入口

    covering ─┐
    regions ──┼─> probing ──> strategy ──> cli
              │                    └────> report / plotting
    config ───┘
"""

__all__ = ["config", "covering", "regions", "probing", "strategy", "report",
           "plotting", "cli"]
