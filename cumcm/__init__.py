"""cumcm —— 2026 年 CUMCM B 题（机器狗搜索干扰源）的求解包。

目录约定
--------
    cumcm/common/     跨题通用层：几何、路由、模拟器接口、演练场、绘图、落盘
    cumcm/t1/         问题 1：交会定位区域
    cumcm/t3/         问题 3：确定性策略（与 T3.py 入口对应）
    cumcm/t3ga/       问题 3：GA 对照方案（与 T3_ga.py 入口对应）
    cumcm/t4/         问题 4：定向 + 全向混合干扰源的确定性策略（与 T4.py 入口对应）
    cumcm/analysis/   验证脚本与论文出图

顶层仍保留同名薄入口（T1.py / T3.py / T3_ga.py / T3_validate.py / T3_figures.py /
T4.py / T4_figures.py / sim_api.py），所以原有的命令行用法一字不改：

    python T3.py --practice 10 --seed 0
    python T3_ga.py --practice 3
    python T3_validate.py
    python T3_figures.py
    python T4.py --practice 20 --seed 0
    python T4_figures.py

分层原则：**依赖只能向下**。common 不 import 任何题目模块；t1/t3/t3ga/t4 可 import common，
t4 另可 import t1（定位区域几何底座）；analysis 可 import 全部。t4 与 t3 是并列叶子，t4 不
import t3 —— 与 t3 同源的 probing 手法以"复制 + 改导入"落在 t4 包内，避免叶子间横向依赖。
"""

__all__ = ["common", "t1", "t3", "t3ga", "t4", "analysis"]
__version__ = "0.2.1"
