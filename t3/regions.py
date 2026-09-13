"""定位区域：把每一次测量，包括听不到，都变成一条可证明的硬约束"""

from __future__ import annotations

from typing import NamedTuple

from common.disc_region import DiscConstraintMixin
from t1 import TriangulationRegion
from t3.config import EXCL_QUAD


class Obs(NamedTuple):
    """一次 direction 检测：在 (x, y) 处测得频道 channel 的示向度 theta，单位是度
    stage 记该次观测来自哪个阶段：survey 巡视扫描，refine 补测与试清，clear 清除阶段复测。"""

    channel: int
    x: float
    y: float
    theta: float
    stage: str = "survey"


class Meas(NamedTuple):
    """一次测量的完整记录，no_signal 也在内，收不到意味着源在接收半径下界 1000 m 之外"""

    channel: int
    x: float
    y: float
    outcome: str                        # direction / near / no_signal
    theta: float | None = None       # 仅 direction 时有值
    stage: str = "survey"


class ProbRegion(DiscConstraintMixin, TriangulationRegion):
    """问题三的"可能源集合"：交会楔形 ∩ 目标圆域，再叠加接收半径给出的硬约束
    接收半径 1000~1500 m，所以 direction 在 1500 m 圆盘内，no_signal 在 1000 m 圆盘外。"""

    # no_signal 在问题三是可证明的硬约束，所以打开圆盘外约束；问题四正好相反，见 t4.regions
    WITH_OUTSIDE = True

    def __init__(self, err: float = 1.0, radius: float | None = None,
                 sides: int | None = None, quad: int = EXCL_QUAD) -> None:
        """err / radius / sides 三个参数见父类，quad 是圆盘近似的正多边形精度，见 config.EXCL_QUAD"""
        super().__init__(err, radius, sides, quad)

    @property
    def diameter(self) -> float:
        """区域直径 / m，父类按顶点算最远点对；非凸多块时它是保守上界，见类文档"""
        return self._memo("diameter", lambda: TriangulationRegion.diameter.fget(self))
