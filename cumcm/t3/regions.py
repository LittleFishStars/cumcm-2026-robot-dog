"""定位区域：把每一次测量（含"听不到"）都变成一条可证明的硬约束。

问题三只有全向源（模拟器的 scenario 对 problem_no=3 硬校验禁止定向源），且有效接收半径
R_rec ∈ [1000, 1500] m，于是每次测量结果都是一条**硬约束**而不是启发式：

    direction ：收得到 ⇒ d ≤ R_rec ≤ 1500  ⇒ 源在"以测量点为心、1500 m"的圆盘内；
    near      ：d ≤ 5 m（题目近距阈值）    ⇒ 源在"以测量点为心、5 m"的圆盘内；
    no_signal ：收不到 ⇒ d > R_rec ≥ 1000  ⇒ 源在"以测量点为心、1000 m"的圆盘外。

no_signal 尤其宝贵：它把"什么都没听到"变成实质的排除约束，让只拿到一条射线的频道从"贯穿
圆域的长带"缩到有限的一段，也直接支撑策略里的"跳过必然无信号的测量"。

保守性靠两种**相反方向**的多边形近似保证：交（"在内"）用外接多边形、差（"在外"）用内接
多边形，于是真源永远不会被切掉。这也是 `enclosing_circle` 半径可以直接当"定位精度"用的前提。

`Obs` 是一条示向度（用于射线交会）；`Meas` 是一次测量的完整记录（含 no_signal），二者
刻意分开：前者给人看、给交会用，后者是硬约束的来源。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import shapely
from shapely import Point

from cumcm.t1 import TriangulationRegion
from cumcm.t3.config import (EXCL_QUAD)


class Obs(NamedTuple):
    """一次 direction 检测：在 (x, y) 处测得频道 channel 的示向度 theta（度）。

    stage 记录该次观测来自哪个阶段（survey = 巡视扫描，refine = 文献准则补测/试清复测，
    clear = 清除阶段的复测）。
    """

    channel: int
    x: float
    y: float
    theta: float
    stage: str = "survey"


class Meas(NamedTuple):
    """一次测量的完整记录（**含 no_signal**）。

    no_signal 在问题三里是一条实质证据：源既然收不到，就必然在接收半径下界 1000 m 之外。
    只记录 direction 观测会丢掉这部分信息，故本程序记录全部测量。
    """

    channel: int
    x: float
    y: float
    outcome: str                        # direction / near / no_signal
    theta: Optional[float] = None       # 仅 direction 时有值
    stage: str = "survey"


class ProbRegion(TriangulationRegion):
    """问题三的"可能源集合"：交会楔形 ∩ 目标圆域，再叠加接收半径给出的**硬约束**。

    为什么可以叠加。问题三只有全向源（模拟器对 problem_no=3 硬校验禁止定向源），且有效接收
    半径 R_rec ∈ [1000, 1500] m，于是每次测量结果都对应一条可证明的约束：

        direction ：收得到 ⇒ d ≤ R_rec ≤ 1500 ⇒ 源落在以测量点为心、半径 1500 m 的圆盘**内**；
        near      ：d ≤ 5 m（近距阈值）      ⇒ 源落在以测量点为心、半径 5 m 的圆盘**内**；
        no_signal ：收不到 ⇒ d > R_rec ≥ 1000 ⇒ 源落在以测量点为心、半径 1000 m 的圆盘**外**。

    三者都不是启发式。no_signal 尤其宝贵：它把"什么都没听到"变成一条实质的排除约束 —— 单条
    射线原本只给出"一条贯穿圆域的长带"，叠加"环带 1000~1500 m"与各站的 no_signal 禁区后，
    常常直接压到可清除量级。这正是"起始先做一次全频道扫描"的额外收益来源。

    保守性（真源永远留在区域内）。圆盘用正多边形近似，两个方向必须各自偏保守：
        "源在圆盘内"的**交集**用外接多边形（半径 r / cos(π/n) ⊇ 真圆盘）；
        "源在圆盘外"的**差集**用内接多边形（半径 r ⊆ 真圆盘）。
    所交集合偏大、所减集合偏小，故真源不会被切掉。

    非凸与多块。减去若干圆盘后区域可能不再凸、甚至裂成多块，本类按"整个几何"处理：vertices
    收集各分块的外环顶点，故直径与最小覆盖圆都覆盖全部可能位置（最远点对必在顶点上），判据只
    会更保守 —— 安全。但"直径 < 40 m ⇒ 最小覆盖圆半径 < 20 m"依赖凸性、非凸时不再成立，故
    T3.py 的清除判据一律直接用最小覆盖圆半径（见 RobotDog._precise），不用直径。
    """

    def __init__(self, err: float = 1.0, radius: Optional[float] = None,
                 sides: Optional[int] = None) -> None:
        super().__init__(err, radius, sides)
        self._inside: List[Tuple[float, float, float]] = []      # 源在此圆盘内
        self._outside: List[Tuple[float, float, float]] = []     # 源在此圆盘外
        self._applied_in = 0
        self._applied_out = 0
        self._cache: Dict[str, Any] = {}

    # ---- 硬约束 ----
    def add_inside(self, x: float, y: float, r: float) -> "ProbRegion":
        """叠加"源在以 (x, y) 为心、r 为半径的圆盘内"。"""
        self._inside.append((float(x), float(y), float(r)))
        return self

    def add_outside(self, x: float, y: float, r: float) -> "ProbRegion":
        """叠加"源在以 (x, y) 为心、r 为半径的圆盘外"。"""
        self._outside.append((float(x), float(y), float(r)))
        return self

    def disc(self, x: float, y: float, r: float, inscribed: bool) -> Polygon:
        """圆盘的正多边形近似：inscribed=True 内接（⊆ 真圆盘），False 外接（⊇ 真圆盘）。"""
        rr = float(r) if inscribed else float(r) / math.cos(math.pi / (4.0 * EXCL_QUAD))
        return Point(float(x), float(y)).buffer(rr, quad_segs=EXCL_QUAD)

    # ---- 几何：楔形交（父类增量）之上再叠加圆盘约束（同样增量、惰性）----
    @property
    def region(self):
        geom = super().region
        if self._applied_in < len(self._inside) or self._applied_out < len(self._outside):
            for x, y, r in self._inside[self._applied_in:]:
                geom = geom.intersection(self.disc(x, y, r, False))
            self._applied_in = len(self._inside)
            for x, y, r in self._outside[self._applied_out:]:
                geom = geom.difference(self.disc(x, y, r, True))
            self._applied_out = len(self._outside)
            self._region = geom                 # 覆盖父类缓存，后续增量楔形交由此继续
            self._cache.clear()
        return self._region

    def _sig(self) -> tuple:
        return (len(self._nodes), self._done, self._applied_in, self._applied_out)

    def _memo(self, key: str, compute):
        self.region                             # 先让几何追平，再取签名
        sig = self._sig()
        if self._cache.get("sig") != sig:
            self._cache = {"sig": sig}
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    @property
    def vertices(self):
        """区域顶点：单块取外环，多块（被禁区切开）时收集各块外环顶点。"""
        geom = self.region
        if geom.is_empty:
            return []
        if geom.geom_type == "Polygon":
            parts = [geom]
        elif geom.geom_type == "MultiPolygon":
            parts = list(geom.geoms)
        else:
            return []
        pts: List[Tuple[float, float]] = []
        for poly in parts:
            pts.extend((x, y) for x, y, *_ in poly.exterior.coords[:-1])
        if len(parts) == 1:
            return pts if parts[0].exterior.is_ccw else pts[::-1]
        return pts

    @property
    def diameter(self):
        return self._memo("diameter", lambda: TriangulationRegion.diameter.fget(self))

    @property
    def enclosing_circle(self):
        def compute():
            if not self.vertices:
                return None
            center = shapely.minimum_bounding_circle(self.region).centroid
            return (center.x, center.y, float(shapely.minimum_bounding_radius(self.region)))
        return self._memo("mec", compute)

    def min_distance_to(self, p: Sequence[float]) -> float:
        """区域（可能源集合）到点 p 的最小距离 / m；区域为空时返回 0。"""
        geom = self.region
        if geom.is_empty:
            return 0.0
        return float(geom.distance(Point(float(p[0]), float(p[1]))))
