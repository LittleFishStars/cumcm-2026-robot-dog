"""定位区域：把每一次测量，包括"听不到"，都变成一条可证明的硬约束

问题三只有全向源，模拟器的 scenario 对 problem_no=3 硬校验禁止定向源，有效接收半径
R_rec ∈ [1000, 1500] m。于是每次测量结果都是一条硬约束，不是启发式：

    direction ：收得到 ⇒ d ≤ R_rec ≤ 1500  ⇒ 源在"以测量点为心、1500 m"的圆盘内；
    near      ：d ≤ 5 m，题目给的近距阈值  ⇒ 源在"以测量点为心、5 m"的圆盘内；
    no_signal ：收不到 ⇒ d > R_rec ≥ 1000  ⇒ 源在"以测量点为心、1000 m"的圆盘外。

no_signal 尤其值钱。它把"什么都没听到"变成一条实质的排除约束：只拿到一条射线的频道原本
是贯穿圆域的一条长带，叠上它之后缩到有限的一段。策略里那句"跳过必然无信号的测量"也靠它。

保守性由两种朝相反方向偏的多边形近似保证。交，也就是"在内"，用外接多边形；差，也就是
"在外"，用内接多边形。真源因此永远不会被切掉。`enclosing_circle` 半径能直接当"定位精度"
用，前提就是这一条。叠加机制本身，增量也罢、惰性也罢、缓存也罢，都在 `common.disc_region`，
本模块只交代本题有哪几类约束。

`Obs` 是一条示向度，给射线交会用；`Meas` 是一次测量的完整记录，连 no_signal 一起记。两者
刻意分开：前者给人看、给交会用，后者才是硬约束的来源。
"""

from __future__ import annotations

from typing import NamedTuple

from common.disc_region import DiscConstraintMixin
from t1 import TriangulationRegion
from t3.config import EXCL_QUAD


class Obs(NamedTuple):
    """一次 direction 检测：在 (x, y) 处测得频道 channel 的示向度 theta，单位是度

    stage 记该次观测来自哪个阶段：survey 是巡视扫描，refine 是文献准则补测与试清复测，
    clear 是清除阶段的复测。
    """

    channel: int
    x: float
    y: float
    theta: float
    stage: str = "survey"


class Meas(NamedTuple):
    """一次测量的完整记录，no_signal 也在内

    no_signal 在问题三里是实质证据：收不到，源就必然在接收半径下界 1000 m 之外。只记
    direction 会白丢这部分信息，所以全部测量都留下来。
    """

    channel: int
    x: float
    y: float
    outcome: str                        # direction / near / no_signal
    theta: float | None = None       # 仅 direction 时有值
    stage: str = "survey"


class ProbRegion(DiscConstraintMixin, TriangulationRegion):
    """问题三的"可能源集合"：交会楔形 ∩ 目标圆域，再叠加接收半径给出的硬约束

    为什么能叠加。问题三只有全向源，模拟器对 problem_no=3 硬校验禁止定向源，有效接收半径
    R_rec ∈ [1000, 1500] m，于是每次测量结果都对应一条可证明的约束：

        direction ：收得到 ⇒ d ≤ R_rec ≤ 1500 ⇒ 源落在以测量点为心、半径 1500 m 的圆盘内；
        near      ：d ≤ 5 m，近距阈值      ⇒ 源落在以测量点为心、半径 5 m 的圆盘内；
        no_signal ：收不到 ⇒ d > R_rec ≥ 1000 ⇒ 源落在以测量点为心、半径 1000 m 的圆盘外。

    三者都不是启发式。no_signal 尤其宝贵：它把"什么都没听到"变成一条实质的排除约束。单条
    射线原本只给出贯穿圆域的一条长带，叠上 1000~1500 m 的环带与各站的 no_signal 禁区之后，
    常常直接压到可清除的量级。起始先做一次全频道扫描之所以划算，额外收益就在这里。

    保守性：真源永远留在区域内。圆盘用正多边形近似，两个方向各自偏保守。算"源在圆盘内"的
    交集时用外接多边形，半径 r / cos(π/n) ⊇ 真圆盘；算"源在圆盘外"的差集时用内接多边形，
    半径 r ⊆ 真圆盘。该大的交集合偏大、该小的减集合偏小，真源自然切不掉。

    非凸与多块。减掉若干圆盘之后区域可能不再凸，甚至裂成多块。`vertices` 按整个几何处理，
    各分块的外环顶点全收，所以直径与最小覆盖圆都盖住全部可能位置，最远点对必在顶点上，判据
    只会更保守，出不了危险。可是"直径 < 40 m ⇒ 最小覆盖圆半径 < 20 m"这条依赖凸性，非凸时
    不成立，所以本包的清除判据一律直接用最小覆盖圆半径，见 RobotDog._precise，不用直径。

    本题的 no_signal 是可证明约束，所以 `WITH_OUTSIDE = True`。问题四正好相反，见 t4.regions。
    """

    WITH_OUTSIDE = True

    def __init__(self, err: float = 1.0, radius: float | None = None,
                 sides: int | None = None, quad: int = EXCL_QUAD) -> None:
        """err / radius / sides 三个参数见父类，quad 是圆盘近似的正多边形精度，见 config.EXCL_QUAD"""
        super().__init__(err, radius, sides, quad)

    @property
    def diameter(self) -> float:
        """区域直径 / m，父类按顶点算最远点对；非凸多块时它是保守上界，见类文档"""
        return self._memo("diameter", lambda: TriangulationRegion.diameter.fget(self))
