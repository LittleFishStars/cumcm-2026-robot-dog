#!/usr/bin/env python3
"""T1.py —— 问题 1：交会定位法的定位区域（顶层薄入口）。

实现在 `cumcm/t1/triangulation.py`：

    from cumcm.t1 import TriangulationRegion
    from T1 import TriangulationRegion        # 兼容写法，两种都可以

检测点 S 处测得示向度 theta、测向误差 ±err（题目 err=1°），干扰源必落在以 S 为顶点、张角
2*err 的楔形内。把楔形写成一个足够长的三角形多边形，则

    定位区域 = 所有楔形之交 ∩ 目标圆域

增量性：add_node 只追加检测点，region 惰性求解并缓存"已并入多少个"，故逐点读结果时每次
只对新楔形求一次交，不重算历史约束。问题三的 ProbRegion 正是复用这个增量接口。

命令行入口（无参数）跑一个 4 检测点的示例并打印定位区域：

    .venv/bin/python T1.py
"""

from __future__ import annotations

from cumcm.t1.triangulation import TriangulationRegion, _demo

__all__ = ["TriangulationRegion"]


if __name__ == "__main__":
    _demo()
