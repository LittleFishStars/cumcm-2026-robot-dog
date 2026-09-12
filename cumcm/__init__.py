"""cumcm —— 2026 年 CUMCM B 题（机器狗搜索干扰源）的求解包。

目录约定
--------
    cumcm/common/     跨题通用层：几何、路由、模拟器接口、演练场、绘图、落盘
    cumcm/t1/         问题 1：交会定位区域
    cumcm/t2/         问题 2：第二个检测点的选择与候选区域（与 T2.py 入口对应）
    cumcm/t3/         问题 3：确定性策略（与 T3.py 入口对应）
    cumcm/t4/         问题 4：定向 + 全向混合干扰源的确定性策略（与 T4.py 入口对应）
    cumcm/analysis/   分析工具（图纸已按要求清理，2026-09-12）

顶层仍保留同名薄入口（T1.py / T2.py / T3.py / T4.py），所以原有的
命令行用法一字不改：

    python T2.py                                  # 问题 2：最优第二检测点 + 适合度图
    python T3.py --practice 10 --seed 0
    python T4.py --practice 20 --seed 0

分层原则：**依赖只能向下**。common 不 import 任何题目模块；t1/t2/t3/t4 可 import common，
t2 可 import t1（定位区域的几何底座：同一套楔形交口径，t2 只是把它向量化到成千上万个候选点），
t4 另可 import t1 与 t3.probing（补测选点与 t3 完全同源，直接复用、不复制）。analysis 可 import
全部。t2/t3/t4 是并列叶子。
"""

__all__ = ["common", "t1", "t2", "t3", "t4", "analysis"]
__version__ = "0.3.0"
