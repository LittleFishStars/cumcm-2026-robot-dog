"""cumcm.t3ga —— 问题 3 的 GA 对照方案（对应顶层入口 T3_ga.py）。

与确定性方案 cumcm.t3 并列，两者共用 cumcm.common 与 cumcm.t1，但策略、参数、结果文件名
各自独立，可对照"确定性方法 vs 启发式方法"的时间与精度。

    config.py      GA 超参与本题常量
    localize.py    遗传算法一：测向定位（实数编码）
    routing_ga.py  遗传算法二：巡视路线（排列编码，开路径 TSP）
    covering.py    贪心集合覆盖求路点
    strategy.py    RobotDog：覆盖路点巡视 + GA 定位/补测 + 就近清除
    plotting.py    逐局轨迹图（写到 <save-dir>/trajectory/）
    training.py    训练记录落盘（ga_training.json / ga_convergence.csv / episodes.csv）
    cli.py         运行编排与命令行入口
"""

__all__ = ["config", "localize", "routing_ga", "covering", "strategy", "plotting",
           "training", "cli"]
