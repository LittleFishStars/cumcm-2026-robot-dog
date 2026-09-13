"""问题四的定位区域，只叠 direction 与 near 两类约束"""

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

    所有运算都是凸集求交，区域保持凸，所以 `vertices`、`diameter`、`enclosing_circle` 按
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
