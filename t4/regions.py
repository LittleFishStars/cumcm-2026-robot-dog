"""问题四的定位区域：硬约束只有 direction 与 near，no_signal 不参与

问题四里全向源和定向源混在一起，没法事先判断某个频道属于哪一类，于是"收不到信号"不再像
问题三那样等于"源在检测点 1000 m 之外"：它可能只是太远，也可能是机器狗站到了定向源背面，
方向不对，见 config.DIR_BEAM_HALF_DEG。所以本模块的"可能源集合"，也就是 DirProbRegion，
只叠加两类能证明的约束：

    direction ：收得到，则 d ≤ R_rec ≤ 1500，源在"以测量点为心、1500 m"的圆盘内；
    near      ：d ≤ 5 m，题目近距阈值，源在"以测量点为心、5 m"的圆盘内。

no_signal 一律不当作区域约束，它给出的信息是"源在覆盖范围之外或者背向"，转不成可证明的
圆盘内外约束。代价是单条射线只能给出"一条贯穿圆域的长带 ∩ 接收半径内包"，收不紧，定位
收敛全指望广角度的多视角交会，这事交给扫描阶段的 20 个测量位置去办。

保守性靠两套方向相反的多边形近似兜底：求交也就是"在内"用外接多边形，求差也就是"在外"用
内接多边形。本类只做交集，楔形交圆域再交圆盘内包，凸集交凸集，所以区域始终是凸的，不会
裂成几块。叠加机制本身怎么写，增量、惰性、缓存那些，在 `common.disc_region` 里。

`Obs` 是一条示向度，射线交会要用；`Meas` 是一次测量的完整记录，含 no_signal，留着核对和
出图，只是不拿它当区域约束。
"""

from __future__ import annotations

from typing import NamedTuple

from common.disc_region import DiscConstraintMixin
from t1 import TriangulationRegion
from t4.config import EXCL_QUAD


class Obs(NamedTuple):
    """一次 direction 检测：在 (x, y) 处测得频道 channel 的示向度 theta，单位为度

    stage 记这次观测出自哪个阶段，sweep 是扫描，refine 是补测，clear 是清除。
    """

    channel: int
    x: float
    y: float
    theta: float
    stage: str = "sweep"


class Meas(NamedTuple):
    """一次测量的完整记录，no_signal 也照记

    no_signal 在问题四里不当区域约束，因为可能只是方向不对，但核对和出图要拿它
    说清"这步在这里测了哪些频道、结果如何"，所以照样留全。
    """

    channel: int
    x: float
    y: float
    outcome: str                        # direction / near / no_signal
    theta: float | None = None       # 仅 direction 时有值
    stage: str = "sweep"


class DirProbRegion(DiscConstraintMixin, TriangulationRegion):
    """问题四的可能源集合：交会楔形交目标圆域，再叠加 direction 给的接收半径内包

    跟问题三 ProbRegion 只差一处，这里只有 add_inside，管 direction 和 near 给出的圆盘内约束，
    没有 add_outside：`WITH_OUTSIDE = False`，误调会直接报错而不会悄悄改掉区域语义，因为定向源
    让 no_signal 转不成"源在圆盘外"的可证明约束。

    所有运算都是凸集求交，区域保持 convex，所以 `vertices`、`diameter`、`enclosing_circle` 按
    凸多边形口径算就是对的，`vertices` 里那个多块分支在这里永远不走。
    """

    WITH_OUTSIDE = False

    def __init__(self, err: float = 1.0, radius: float | None = None,
                 sides: int | None = None, quad: int = EXCL_QUAD) -> None:
        """err / radius / sides 都在父类里；quad 照 config.EXCL_QUAD，是圆盘近似的正多边形精度"""
        super().__init__(err, radius, sides, quad)

    @property
    def diameter(self) -> float:
        """区域直径，单位 m：父类按顶点取最远点对，本题区域恒为凸，算出来就是精确值"""
        return self._memo("diameter", lambda: TriangulationRegion.diameter.fget(self))
