"""问题 1：交会定位法的定位区域，基于 shapely/GEOS 与 numpy"""

from math import atan2, cos, degrees, hypot, radians, sin

import numpy as np
import shapely
from shapely import Point, Polygon

from common.console import print_table


class TriangulationRegion:
    """交会定位区域：累积检测点后求定位区域凸多边形及其几何量"""

    RADIUS = 1800.0  # 目标圆域半径（米），题目给定（直径 3600 m）
    SIDES = 64       # 目标圆域的内接正多边形边数，须为 4 的倍数
    TOUCH = 1.0      # 顶点贴合圆域边界的判定阈值（米）

    def __init__(self, err: float = 1.0, radius: float | None = None, sides: int | None = None) -> None:
        """构造交会定位区域，err 为示向度误差半宽（度）"""
        self.err = float(err)
        self.radius = self.RADIUS if radius is None else float(radius)
        self.sides = self.SIDES if sides is None else int(sides)
        if self.sides % 4:
            raise ValueError("sides 必须是 4 的倍数")
        self._nodes = []
        self._clip = None    # 目标圆域的内接正多边形（惰性生成）
        self._region = None  # 已并入前 _done 个检测点的交集；None 表示尚未求解
        self._done = 0

    # ---------- 检测点管理 ----------

    def add_node(self, x: float, y: float, theta: float) -> TriangulationRegion:
        """加入一个检测点，坐标为米、示向度为度，返回自身便于链式调用"""
        self._nodes.append((float(x), float(y), float(theta) % 360.0))
        return self

    @classmethod
    def from_nodes(cls, nodes: list[tuple[float, float, float]], err: float = 1.0,
                   radius: float | None = None, sides: int | None = None) -> TriangulationRegion:
        """由检测点列表 [(x, y, theta), ...] 构造定位区域，坐标为米、示向度为度"""
        region = cls(err, radius, sides)
        for x, y, theta in nodes:
            region.add_node(x, y, theta)
        return region

    # ---------- 定位区域与几何量 ----------

    @property
    def clip(self) -> Polygon:
        """目标圆域，也就是内接正 sides 边形，顶点精确落在半径 radius 的圆上"""
        if self._clip is None:
            self._clip = Point(0.0, 0.0).buffer(self.radius, quad_segs=self.sides // 4)
        return self._clip

    @property
    def region(self) -> Polygon:
        """定位区域的 shapely 几何，空区域返回空几何"""
        if self._region is None:
            # 楔形张角为 0，也就是 err<=0 时，区域退化成零面积
            self._region = self.clip if self.err > 0.0 else Polygon()
            self._done = 0
        for x, y, theta in self._nodes[self._done:]:
            self._region = self._region.intersection(self._wedge(x, y, theta))
            self._done += 1
            if self._region.is_empty:  # 空区域与任何楔形相交仍为空
                self._done = len(self._nodes)
                break
        return self._region

    @property
    def vertices(self) -> list[tuple[float, float]]:
        """定位区域顶点坐标列表 [(x, y), ...]，统一成逆时针，空区域返回空表"""
        region = self.region
        if region.is_empty or region.geom_type != "Polygon":
            return []
        pts = [(x, y) for x, y, *_ in region.exterior.coords[:-1]]
        return pts if region.exterior.is_ccw else pts[::-1]

    @property
    def area(self) -> float:
        """定位区域面积 / m²"""
        return float(self.region.area)

    @property
    def diameter(self) -> float:
        """定位区域直径：区域内任意两点距离的最大值 / m"""
        pts = np.asarray(self.vertices, dtype=float)
        if len(pts) < 2:
            return 0.0
        return float(np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1).max())

    @property
    def bounded(self) -> bool:
        """定位是否只靠检测点就已确定，没被目标圆域截断"""
        # 有顶点落在圆域边界上，说明那个方向约束不足
        pts = self.vertices
        return bool(pts) and all(self.clip.exterior.distance(Point(*p)) > self.TOUCH for p in pts)

    @property
    def enclosing_circle(self) -> tuple[float, float, float] | None:
        """定位区域的最小覆盖圆 (cx, cy, r) / m，区域为空或退化时返回 None"""
        # 问题 1 第二问的判据（以区域直径为直径的圆能否盖住区域）就是看 r <= diameter / 2
        if not self.vertices:
            return None
        # 最小包围圆是正内接多边形，它的质心就是圆心
        center = shapely.minimum_bounding_circle(self.region).centroid
        return (center.x, center.y, float(shapely.minimum_bounding_radius(self.region)))

    def contains(self, p: tuple[float, float]) -> bool:
        """点 p 是否落在定位区域内，圆域按内接多边形算，圆边上的月牙不算"""
        return bool(self.region.covers(Point(p)))

    # ---------- 基础几何 ----------

    def _wedge(self, x: float, y: float, theta: float) -> Polygon:
        """单个检测点的楔形多边形，以 (x, y) 为顶点，边长足够覆盖整个目标圆域"""
        length = 2.0 * (hypot(x, y) + self.radius)
        a1, a2 = radians(theta - self.err), radians(theta + self.err)
        return Polygon([(x, y),
                        (x + length * cos(a1), y + length * sin(a1)),
                        (x + length * cos(a2), y + length * sin(a2))])

    @staticmethod
    def bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
        """a 点观测 b 点的方位角 / 度，x 轴正向逆时针为正，取值 [0, 360)"""
        return degrees(atan2(b[1] - a[1], b[0] - a[0])) % 360.0

    def __repr__(self) -> str:
        """对象的可读表示，汇总检测点数、顶点数、直径与 bounded"""
        return (f"TriangulationRegion(检测点 {len(self._nodes)} 个, 顶点 {len(self.vertices)} 个, "
                f"直径 {self.diameter:.4f} m, 仅靠检测点可确定 {self.bounded})")


def demo() -> None:
    """示例：4 个检测点对同一个干扰源交会定位"""
    # 没有实测数据，先假定真值 G，再按几何关系反推各点示向度，保证区域非空
    g = (500.0, 400.0)
    sites = [(0.0, 0.0), (1000.0, 0.0), (200.0, 900.0), (800.0, 900.0)]
    region = TriangulationRegion.from_nodes(
        [(x, y, TriangulationRegion.bearing((x, y), g)) for x, y in sites])

    cx, cy, r = region.enclosing_circle
    print("问题一：4 个检测点的交会定位")
    print_table(["量", "数值", "单位"],
                [["定位区域直径", f"{region.diameter:.4f}", "m"],
                 ["最小覆盖圆圆心 x", f"{cx:.4f}", "m"],
                 ["最小覆盖圆圆心 y", f"{cy:.4f}", "m"],
                 ["最小覆盖圆半径", f"{r:.4f}", "m"]],
                align="lrr")


if __name__ == '__main__':
    demo()
