"""问题四的定位区域：只把"direction / near"当作硬约束，no_signal 不作为区域约束。

问题四同时存在全向源与定向源，且**无法预先知道某个频道属于哪一类**，所以"收不到信号"不再
像问题三那样等价于"源在检测点 1000 m 之外"——它可能是太远，也可能是机器狗站在定向源的
背面（方向不对，见 config.DIR_BEAM_HALF_DEG）。因此本模块的"可能源集合"（DirProbRegion）
只叠加两类可证明约束：

    direction ：收得到 ⇒ d ≤ R_rec ≤ 1500 ⇒ 源在"以测量点为心、1500 m"的圆盘**内**；
    near      ：d ≤ 5 m（题目近距阈值）  ⇒ 源在"以测量点为心、5 m"的圆盘**内**。

no_signal 一律**不**作为区域约束（它给出的"源在覆盖范围之外或背向"无法转成可证明的圆盘
内/外约束）。代价是单条射线只能给出"一条贯穿圆域的长带 ∩ 接收半径内包"，定位收敛完全靠
**广角度的多视角交会**——这正由扫描阶段 20 个测量位置自然提供。

保守性由两种相反方向的多边形近似保证：交（"在内"）用外接多边形、差（"在外"）用内接多边形。
本类只做交集运算（楔形交 ∩ 圆域 ∩ 圆盘内包），全部是凸集交集，故区域保持凸、不会裂成多块。
叠加机制本身（增量、惰性、缓存）在 `common.disc_region`。

`Obs` 是一条示向度（用于射线交会）；`Meas` 是一次测量的完整记录（含 no_signal，保留用于
核对与出图——只是不作为区域约束）。
"""

from __future__ import annotations

from typing import NamedTuple

from common.disc_region import DiscConstraintMixin
from t1 import TriangulationRegion
from t4.config import EXCL_QUAD


class Obs(NamedTuple):
    """一次 direction 检测：在 (x, y) 处测得频道 channel 的示向度 theta（度）。

    stage 记录该次观测来自哪个阶段（sweep = 扫描，refine = 文献准则补测，clear = 清除）。
    """

    channel: int
    x: float
    y: float
    theta: float
    stage: str = "sweep"


class Meas(NamedTuple):
    """一次测量的完整记录（**含 no_signal**）。

    no_signal 在问题四里**不是**区域约束（可能是方向不对），但仍完整记录：核对/出图要用它
    说明"这步在这里测了哪些频道、结果如何"。
    """

    channel: int
    x: float
    y: float
    outcome: str                        # direction / near / no_signal
    theta: float | None = None       # 仅 direction 时有值
    stage: str = "sweep"


class DirProbRegion(DiscConstraintMixin, TriangulationRegion):
    """问题四的"可能源集合"：交会楔形 ∩ 目标圆域，再叠加"direction ⇒ 接收半径内包"。

    与问题三 ProbRegion 的差别：**只有 add_inside（direction/near 给出的圆盘内约束），没有
    add_outside** —— `WITH_OUTSIDE = False`，一旦误调 add_outside 会直接报错，不会悄悄改变
    区域语义。原因是定向源使 no_signal 无法转成"源在圆盘外"的可证明约束（见模块文档）。

    所有运算都是凸集交集，区域保持凸，因此 `vertices` / `diameter` / `enclosing_circle` 按
    凸多边形的口径计算即可（`vertices` 的多块分支在这里永不触发）。
    """

    WITH_OUTSIDE = False

    def __init__(self, err: float = 1.0, radius: float | None = None,
                 sides: int | None = None, quad: int = EXCL_QUAD) -> None:
        """err / radius / sides 见父类；quad 为圆盘近似的正多边形精度（见 config.EXCL_QUAD）。"""
        super().__init__(err, radius, sides, quad)

    @property
    def diameter(self) -> float:
        """区域直径 / m：父类按顶点算最远点对（本题区域恒为凸，即为精确值）。"""
        return self._memo("diameter", lambda: TriangulationRegion.diameter.fget(self))
