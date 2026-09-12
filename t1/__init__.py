"""t1 —— 问题 1：交会定位法的定位区域。

    triangulation.py    TriangulationRegion：楔形交 ∩ 目标圆域，及其几何量

该类的 `add_node` / `region` 增量式设计被问题三直接复用（见 `t3.regions.ProbRegion`
在其上加挂"圆盘型硬约束"），所以它放在独立的 t1 子包里而不是 common —— common 只管与题目
无关的基础设施。
"""

from t1.triangulation import TriangulationRegion

__all__ = ["TriangulationRegion"]
