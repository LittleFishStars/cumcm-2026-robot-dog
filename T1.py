"""T1.py —— 问题 1：交会定位法的定位区域（基于 shapely/GEOS 与 numpy）

检测点 S 处测得示向度 theta、测向误差 ±err（题目 err=1°），干扰源必落在以 S 为顶点、张角
2*err 的楔形内。把楔形写成一个足够长的三角形多边形，则

    定位区域 = 所有楔形之交 ∩ 目标圆域

目标圆域用 buffer 生成内接正 64 边形（顶点精确落在半径 1800 m 的圆上，圆域横跨 3600 m；
弦与真圆最大偏差 1800*(1-cos(π/64)) ≈ 2.17 m）。几何运算全部交给库：交集 shapely(GEOS)、
面积 region.area、直径 numpy 成对距离、最小覆盖圆 minimum_bounding_circle、包含 covers。

增量性：add_node 只追加检测点，region 惰性求解并缓存"已并入多少个"，故逐点读结果时每次
只对新楔形求一次交，不重算历史约束。

语义：区域已被圆域截断，故 area / diameter / enclosing_circle 都是截断后的有限值；bounded
另行回答"定位是否只靠检测点就确定了"——有顶点落在圆域边界上（约束不足）时为 False。

依赖 shapely>=2.1、numpy；运行 `.venv/bin/python T1.py` 或 `uv run T1.py`。
"""

from math import atan2, cos, degrees, hypot, radians, sin

import numpy as np
import shapely
from shapely import Point, Polygon


class TriangulationRegion:
    """交会定位区域：累积检测点（坐标 + 示向度），求定位区域凸多边形及其几何量。"""

    RADIUS = 1800.0  # 目标圆域半径（米），题目给定（直径 3600 m）
    SIDES = 64       # 目标圆域的内接正多边形边数，须为 4 的倍数
    TOUCH = 1.0      # 顶点贴合圆域边界的判定阈值（米）

    def __init__(self, err=1.0, radius=None, sides=None):
        """err：示向度误差半宽（度），对所有检测点相同。

        radius：目标圆域半径（米），缺省取 RADIUS（题目为 1800）。
        sides ：目标圆域的内接正多边形边数，缺省取 SIDES，须为 4 的倍数。
        """
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

    def add_node(self, x, y, theta):
        """加入一个检测点：坐标 (x, y)（米）与该点测得的示向度 theta（度）。"""
        self._nodes.append((float(x), float(y), float(theta) % 360.0))
        return self

    @classmethod
    def from_nodes(cls, nodes, err=1.0, radius=None, sides=None):
        """由检测点列表 [(x, y, theta), ...] 构造定位区域。"""
        region = cls(err, radius, sides)
        for x, y, theta in nodes:
            region.add_node(x, y, theta)
        return region

    @property
    def nodes(self):
        """已加入的检测点列表 [(x, y, theta), ...]。"""
        return list(self._nodes)

    # ---------- 定位区域与几何量 ----------

    @property
    def clip(self):
        """目标圆域：内接正 sides 边形（顶点精确落在半径 radius 的圆上）。"""
        if self._clip is None:
            self._clip = Point(0.0, 0.0).buffer(self.radius, quad_segs=self.sides // 4)
        return self._clip

    @property
    def region(self):
        """定位区域的 shapely 几何（空区域返回空几何）。"""
        if self._region is None:
            # 楔形张角为 0（err<=0）时区域退化为零面积
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
    def vertices(self):
        """定位区域顶点坐标列表 [(x, y), ...]（统一为逆时针）；区域为空或退化时返回 []。"""
        region = self.region
        if region.is_empty or region.geom_type != "Polygon":
            return []
        pts = [(x, y) for x, y, *_ in region.exterior.coords[:-1]]
        return pts if region.exterior.is_ccw else pts[::-1]

    @property
    def area(self):
        """定位区域面积（平方米）。"""
        return float(self.region.area)

    @property
    def diameter(self):
        """定位区域直径：区域内任意两点距离的最大值。

        区域是凸多边形，故直径即顶点间最大两两距离，用 numpy 的成对距离矩阵一次求出。"""
        pts = np.asarray(self.vertices, dtype=float)
        if len(pts) < 2:
            return 0.0
        return float(np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1).max())

    @property
    def bounded(self):
        """定位是否只靠检测点就已确定（区域未被目标圆域截断）。

        有顶点落在圆域边界上即说明该方向约束不足（例如只测一次示向度、或检测点与干扰源近似
        共线），此时返回 False；区域为空同样返回 False。"""
        pts = self.vertices
        return bool(pts) and all(self.clip.exterior.distance(Point(*p)) > self.TOUCH for p in pts)

    @property
    def enclosing_circle(self):
        """定位区域的最小覆盖圆 (cx, cy, r)；区域为空或退化时返回 None。

        由 GEOS 的最小包围圆给出。问题 1 第二问"以定位区域直径为直径的圆能否覆盖此定位区域"
        的判据即 r <= diameter / 2。"""
        if not self.vertices:
            return None
        # 最小包围圆是正内接多边形，其质心即圆心
        center = shapely.minimum_bounding_circle(self.region).centroid
        return (center.x, center.y, float(shapely.minimum_bounding_radius(self.region)))

    def contains(self, p):
        """点 p 是否落在定位区域内（圆域为内接多边形，真圆靠边的月牙不在此区域内）。"""
        return bool(self.region.covers(Point(p)))

    # ---------- 基础几何 ----------

    def _wedge(self, x, y, theta):
        """单个检测点的楔形多边形：以 (x, y) 为顶点的三角形，边长足以覆盖整个目标圆域。"""
        length = 2.0 * (hypot(x, y) + self.radius)
        a1, a2 = radians(theta - self.err), radians(theta + self.err)
        return Polygon([(x, y),
                        (x + length * cos(a1), y + length * sin(a1)),
                        (x + length * cos(a2), y + length * sin(a2))])

    @staticmethod
    def bearing(a, b):
        """在 a 点观测 b 点的方位角（度，[0, 360)）：x 轴正向逆时针旋转到 a→b 的夹角。"""
        return degrees(atan2(b[1] - a[1], b[0] - a[0])) % 360.0

    def __repr__(self):
        return (f"TriangulationRegion(检测点 {len(self._nodes)} 个, 顶点 {len(self.vertices)} 个, "
                f"直径 {self.diameter:.4f} m, 仅靠检测点可确定 {self.bounded})")


if __name__ == "__main__":
    # 示例：4 个检测点对同一干扰源交会定位。
    # 无实测数据时，先假定干扰源真值 G，再由几何关系反推各点示向度，保证区域非空。
    G = (500.0, 400.0)
    sites = [(0.0, 0.0), (1000.0, 0.0), (200.0, 900.0), (800.0, 900.0)]
    region = TriangulationRegion.from_nodes(
        [(x, y, TriangulationRegion.bearing((x, y), G)) for x, y in sites])

    print(f"定位区域直径：{region.diameter:.4f} m")
    cx, cy, r = region.enclosing_circle
    print(f"最小覆盖圆：圆心 ({cx:.4f}, {cy:.4f})，半径 {r:.4f} m")
