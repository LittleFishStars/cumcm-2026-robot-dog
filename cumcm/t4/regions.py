"""问题四的定位区域：只把"direction / near"当作硬约束，no_signal 不作为区域约束。

问题四同时存在全向源与定向源，且**无法预先知道某个频道属于哪一类**，所以"收不到信号"不再
像问题三那样等价于"源在检测点 1000 m 之外"——它可能是太远，也可能是机器狗站在定向源的
背面（方向不对，见 config.DIR_BEAM_HALF_DEG）。因此本模块的"可能源集合"（DirProbRegion）
只叠加两类可证明约束：

    direction ：收得到 ⇒ d ≤ R_rec ≤ 1500 ⇒ 源在"以测量点为心、1500 m"的圆盘**内**；
    near      ：d ≤ 5 m（题目近距阈值）  ⇒ 源在"以测量点为心、5 m"的圆盘**内**。

no_signal 一律**不**作为区域约束（它给出的"源在覆盖范围之外或背向"无法转成可证明的圆盘
内/外约束）。代价是单条射线只能给出"一条贯穿圆域的长带 ∩ 接收半径内包"，定位收敛完全靠
**广角度的多视角交会**——这正由寻向拖网阶段的多测量点自然提供。

保守性由两种相反方向的多边形近似保证：交（"在内"）用外接多边形、差（"在外"）用内接多边形。
本类只做交集运算（楔形交 ∩ 圆域 ∩ 圆盘内包），全部是凸集交集，故区域保持凸、不会裂成多块。

`Obs` 是一条示向度（用于射线交会）；`Meas` 是一次测量的完整记录（含 no_signal，保留用于
核对与出图——只是不作为区域约束）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import shapely
from shapely import Point

from cumcm.t1 import TriangulationRegion
from cumcm.t4.config import EXCL_QUAD, NEAR_RADIUS, RECEIVE_MAX


class Obs(NamedTuple):
    """一次 direction 检测：在 (x, y) 处测得频道 channel 的示向度 theta（度）。

    stage 记录该次观测来自哪个阶段（sweep = 寻向拖网，refine = 文献准则补测，clear = 清除）。
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
    theta: Optional[float] = None       # 仅 direction 时有值
    stage: str = "sweep"


class DirProbRegion(TriangulationRegion):
    """问题四的"可能源集合"：交会楔形 ∩ 目标圆域，再叠加"direction ⇒ 接收半径内包"。

    与问题三 ProbRegion 的差别：**只有 add_inside（direction/near 给出的圆盘内约束），没有
    add_outside**。原因是定向源使 no_signal 无法转成"源在圆盘外"的可证明约束（见模块文档）。

    所有运算都是凸集交集，区域保持凸，因此 `vertices`/`diameter`/`enclosing_circle` 直接按
    父类（t1.TriangulationRegion）的凸多边形口径计算即可；这里只把圆盘内包增量叠加上去并
    用 shapely 的最小覆盖圆给出 `enclosing_circle`。
    """

    def __init__(self, err: float = 1.0, radius: Optional[float] = None,
                 sides: Optional[int] = None) -> None:
        super().__init__(err, radius, sides)
        self._inside: List[Tuple[float, float, float]] = []      # 源在此圆盘内
        self._applied_in = 0
        self._cache: Dict[str, Any] = {}

    def add_inside(self, x: float, y: float, r: float) -> "DirProbRegion":
        """叠加"源在以 (x, y) 为心、r 为半径的圆盘内"（direction 用 1500，near 用 5）。"""
        self._inside.append((float(x), float(y), float(r)))
        return self

    def disc(self, x: float, y: float, r: float, inscribed: bool):
        """圆盘的正多边形近似：inscribed=True 内接（⊆ 真圆盘），False 外接（⊇ 真圆盘）。

        本类只用外接（交集方向），inscribed 参数保留以与 t3.regions 的接口保持一致。
        """
        rr = float(r) if inscribed else float(r) / math.cos(math.pi / (4.0 * EXCL_QUAD))
        return Point(float(x), float(y)).buffer(rr, quad_segs=EXCL_QUAD)

    @property
    def region(self):
        """楔形交 ∩ 圆域 之上再叠加圆盘内包（增量、惰性，语义见父类）。"""
        geom = super().region
        if self._applied_in < len(self._inside):
            for x, y, r in self._inside[self._applied_in:]:
                geom = geom.intersection(self.disc(x, y, r, False))
            self._applied_in = len(self._inside)
            self._region = geom                 # 覆盖父类缓存，后续增量楔形交由此继续
            self._cache.clear()
        return self._region

    def _sig(self) -> tuple:
        return (len(self._nodes), self._done, self._applied_in)

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
        """区域顶点（凸多边形外环，逆时针）；区域为空/退化时返回 []。"""
        region = self.region
        if region.is_empty or region.geom_type != "Polygon":
            return []
        pts = [(x, y) for x, y, *_ in region.exterior.coords[:-1]]
        return pts if region.exterior.is_ccw else pts[::-1]

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