"""圆盘硬约束的"可能源集合"：问题三、问题四共用的区域叠加机制

它只做一件事：在父类 `t1.TriangulationRegion` 的"楔形交 ∩ 目标圆域"之上，再叠加每次测量给出的
圆盘硬约束。源在某圆盘内用交，拿外接多边形近似；源在某圆盘外用差，拿内接多边形近似。两种近似
方向相反，所以真源永远不会被切掉。论证见 `t3/regions.py` 与 `t4/regions.py`，那两处只需说明
本题有哪些约束，不必再各写一遍同样的几何机制。

为什么是混合器而不是基类：分层约定是 `common <- t1/t3/t4`，`common` 不能反向依赖 `t1`，所以这里
不继承 `TriangulationRegion`，只约定宿主类必须把它排在 MRO 中 `TriangulationRegion` 之前：

    class ProbRegion(DiscConstraintMixin, TriangulationRegion):
        WITH_OUTSIDE = True          # 本题的"收不到 ⇒ 源在圆盘外"是可证明约束

宿主类须提供下面这些，都由 `TriangulationRegion` 给出：

* `region` 属性：楔形交 ∩ 圆域的增量几何。本类用 `super().region` 取它，拿到后继续增量维护
  `_region`，圆盘约束与楔形约束共享同一份缓存；
* `_nodes` / `_done` / `_region`：父类的增量状态，几何签名靠它们判断要不要重算。

增量性是这么来的：`add_inside` / `add_outside` 只追加约束，`region` 惰性求解并缓存"已并入几条"。
反复读直径、最小覆盖圆时只补新增的那几条，历史约束不重算。

近似参数 `quad`：圆盘用正 `4×quad` 边形近似，取各题 config 的 `EXCL_QUAD`，16 就是正 64 边形。
交用外接，半径 ×1/cos(π/4q)，是 ⊇ 真圆盘；差用内接，半径不变，是 ⊆ 真圆盘。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import shapely
from shapely import Point, Polygon

__all__ = ["DiscConstraintMixin"]

# "区域到点的最小距离 > 接收半径"要留一点数值余量再判"必然收不到"。两者都在米级，1 m 的余量
# 足以盖住多边形近似（弦偏差 ≤ 2.17 m 的那部分是圆域自身的，圆盘近似偏差同量级）与浮点误差，
# 又远小于任何一个动作的物理尺度。
OUT_OF_REACH_MARGIN = 1.0


class DiscConstraintMixin:
    """把"源在圆盘内 / 圆盘外"的硬约束增量叠加到宿主类的 `region` 上

    子类只要声明本题有哪几类约束，也就是 `WITH_OUTSIDE`，再绑定各自的圆盘近似精度，
    几何机制全在这里。
    """

    WITH_OUTSIDE = False        # 子类按题意打开：可证明"收不到 ⇒ 源在圆盘外"时才开

    def __init__(self, err: float = 1.0, radius: float | None = None,
                 sides: int | None = None, quad: int = 16) -> None:
        """初始化圆盘约束叠层，把 err / radius / sides 原样转发给宿主类

        Args:
            err: 示向度误差半宽 / 度，透传给宿主类的定位区域
            radius: 目标圆域半径 / m；None 表示沿用宿主类的缺省值
            sides: 目标圆域内接正多边形边数；None 表示沿用宿主类的缺省值
            quad: 圆盘近似的正多边形精度，边数 = 4 × quad，见模块文档
        """
        super().__init__(err, radius, sides)
        self.quad = int(quad)                            # 圆盘近似的正多边形精度，见模块文档
        self._inside: list[tuple[float, float, float]] = []      # 源在此圆盘内
        self._outside: list[tuple[float, float, float]] = []     # 源在此圆盘外
        self._applied_in = 0
        self._applied_out = 0
        self._cache: dict[str, Any] = {}

    # ---- 硬约束 ----
    def add_inside(self, x: float, y: float, r: float) -> "DiscConstraintMixin":
        """叠加一条"源在以 (x, y) 为心、r 为半径的圆盘内"。direction 用接收半径上限，near 用 5 m"""
        self._inside.append((float(x), float(y), float(r)))
        return self

    def add_outside(self, x: float, y: float, r: float) -> "DiscConstraintMixin":
        """叠加一条"源在以 (x, y) 为心、r 为半径的圆盘外"，只有 `WITH_OUTSIDE = True` 的题能用"""
        if not self.WITH_OUTSIDE:
            raise RuntimeError(
                f"{type(self).__name__} 不支持圆盘外约束：本题的 no_signal 不是可证明约束，"
                f"比如问题四的定向源，收不到也可能只是方向不对，见模块文档")
        self._outside.append((float(x), float(y), float(r)))
        return self

    def disc(self, x: float, y: float, r: float, inscribed: bool) -> Polygon:
        """圆盘的正多边形近似：inscribed=True 内接，⊆ 真圆盘；False 外接，⊇ 真圆盘"""
        rr = float(r) if inscribed else float(r) / math.cos(math.pi / (4.0 * self.quad))
        return Point(float(x), float(y)).buffer(rr, quad_segs=self.quad)

    # ---- 几何：在父类的增量楔形交之上再叠加圆盘约束，同样增量、同样惰性 ----
    @property
    def region(self) -> shapely.geometry.base.BaseGeometry:
        """楔形交与圆盘约束叠加后的可能源集合，增量维护并缓存，见模块文档"""
        geom = super().region
        n_in, n_out = len(self._inside), len(self._outside)
        if self._applied_in < n_in or self._applied_out < n_out:
            # 只把新增的约束并进几何：先逐个求交"源在圆盘内"，再逐个减去"源在圆盘外"
            for x, y, r in self._inside[self._applied_in:]:
                geom = geom.intersection(self.disc(x, y, r, False))
            self._applied_in = n_in
            for x, y, r in self._outside[self._applied_out:]:
                geom = geom.difference(self.disc(x, y, r, True))
            self._applied_out = n_out
            self._region = geom                 # 覆盖父类缓存，后续增量楔形交由此继续
            self._cache.clear()
        return self._region

    @property
    def vertices(self) -> list[tuple[float, float]]:
        """区域顶点：单块取外环，被圆盘外约束切开的多个块则收集各块外环顶点

        只做交集的题目区域恒为单块，走的是与父类完全相同的那条分支。
        """
        geom = self.region
        if geom.is_empty:
            return []
        if geom.geom_type == "Polygon":
            parts = [geom]
        elif geom.geom_type == "MultiPolygon":
            parts = list(geom.geoms)
        else:
            return []
        pts: list[tuple[float, float]] = []
        for poly in parts:
            pts.extend((x, y) for x, y, *_ in poly.exterior.coords[:-1])
        if len(parts) == 1:
            return pts if parts[0].exterior.is_ccw else pts[::-1]
        return pts

    @property
    def enclosing_circle(self) -> tuple[float, float, float] | None:
        """区域的最小覆盖圆 (cx, cy, r)；区域为空或退化时返回 None

        "真源必在区域内 ⊆ 覆盖圆内"对任何集合都成立，不要求凸性，所以多块、非凸时照样能用。
        这是"走到圆心即可清除"的依据。
        """
        def compute() -> tuple[float, float, float] | None:
            """按需算出最小覆盖圆，交给 _memo 缓存"""
            if not self.vertices:
                return None
            # GEOS 的最小包围圆：质心就是圆心，最小包围半径就是覆盖半径
            center = shapely.minimum_bounding_circle(self.region).centroid
            return (center.x, center.y, float(shapely.minimum_bounding_radius(self.region)))
        return self._memo("mec", compute)

    def min_distance_to(self, p: Sequence[float]) -> float:
        """可能源集合到点 p 的最小距离 / m；区域为空时返回 0"""
        geom = self.region
        if geom.is_empty:
            return 0.0
        return float(geom.distance(Point(float(p[0]), float(p[1]))))

    def provably_out_of_reach(self, at: Sequence[float], receive_max: float) -> bool:
        """能否证明"在 at 处测这个频道必然收不到"，能证明就省掉这次测量

        可能源集合是真实源位置的超集。区域到 at 的最小距离大于 receive_max，真源到 at 的距离
        就也大于 receive_max，也就是超过有效接收半径上限，那么必然收不到信号，这次测量不带
        任何新信息。区域为空、无从判断时返回 False。
        """
        if self.region.is_empty:
            return False
        return self.min_distance_to(at) > receive_max + OUT_OF_REACH_MARGIN

    # ---- 几何量的增量缓存 ----
    def _sig(self) -> tuple:
        """几何签名：四项增量里任何一个变了，缓存就得作废"""
        return (len(self._nodes), self._done, self._applied_in, self._applied_out)

    def _memo(self, key: str, compute: Callable[[], Any]) -> Any:
        """按几何签名缓存一个几何量；先让 `region` 追平，再取签名"""
        self.region                             # 先让几何追平，再取签名
        sig = self._sig()
        if self._cache.get("sig") != sig:
            self._cache = {"sig": sig}
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]
