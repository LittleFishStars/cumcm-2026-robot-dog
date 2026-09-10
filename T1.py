"""T1.py —— 问题 1：交会定位法的定位区域（基于 shapely/GEOS 与 numpy）

原理
----
1. 检测点 S 处测得示向度 theta、测向误差 ±err（题目 err=1°），干扰源必落在以 S 为顶点、
   张角 2*err 的楔形内。把每个楔形写成一个（足够长的）三角形多边形，再并入一个大包围盒
   （保证区域有界），于是 定位区域 = 所有楔形与包围盒的交集。

2. 几何计算全部交给成熟库，不再手写几何算法：
   ● 交集       → shapely（GEOS 布尔运算），且交集满足结合律，新检测点只需与已有区域补裁一次
   ● 面积       → region.area
   ● 直径       → numpy 在区域顶点上求最大两两距离（区域是凸多边形，故顶点对最大距离即直径）
   ● 最小覆盖圆  → shapely.minimum_bounding_circle / minimum_bounding_radius（GEOS 最小包围圆）
   ● 点包含判定  → region.covers

3. 增量性：`region` 惰性求解且缓存"已并入多少个检测点"，`__add__` 把该缓存传给新对象，
   故 `r = r + node` 后读取 `r.region` 只会对新楔形求一次交，不会重算历史约束。

依赖：shapely>=2.1、numpy（见 pyproject.toml；用 `uv run T1.py` 或 `.venv/bin/python T1.py` 运行）
"""

from math import atan2, cos, degrees, hypot, radians, sin

import numpy as np
import shapely
from shapely import Point, Polygon, box

# 顶点贴合包围盒的判定阈值（米）：小于该值即认为顶点落在包围盒边上、区域无界
_TOUCH = 1.0


class TriangulationRegion:
    """交会定位区域：累积检测点（坐标 + 示向度），求定位区域凸多边形及其几何量。"""

    def __init__(self, err=1.0, bound=1e5):
        """err：示向度误差半宽（度），对所有检测点相同；bound：包围盒半边长（米）。"""
        self.err = float(err)
        self.bound = float(bound)
        self._nodes = []
        self._region = None  # 已并入前 _done 个检测点的交集；None 表示尚未求解
        self._done = 0

    # ---------- 检测点管理（加法返回新区域，原对象不变） ----------

    def __add__(self, node):
        """region + (x, y, theta)：加入一个检测点（坐标米、示向度度），返回新的定位区域。"""
        x, y, theta = (float(v) for v in node)
        new = self.__class__(self.err, self.bound)
        new._nodes = self._nodes + [(x, y, theta % 360.0)]
        new._region, new._done = self._region, self._done  # 继承缓存，新点只需补裁
        return new

    @classmethod
    def from_nodes(cls, nodes, err=1.0, bound=1e5):
        """由检测点列表 [(x, y, theta), ...] 构造定位区域。"""
        region = cls(err, bound)
        for node in nodes:
            region = region + node
        return region

    @property
    def nodes(self):
        """已加入的检测点列表 [(x, y, theta), ...]。"""
        return list(self._nodes)

    # ---------- 定位区域与几何量 ----------

    @property
    def region(self):
        """定位区域的 shapely 几何（Polygon；空区域返回空几何）。"""
        if self._region is None:
            # 楔形张角为 0（err<=0）时区域退化为零面积
            self._region = box(-self.bound, -self.bound, self.bound, self.bound) \
                if self.err > 0.0 else Polygon()
            self._done = 0
        for x, y, theta in self._nodes[self._done:]:
            self._region = self._region.intersection(self._wedge(x, y, theta))
            if self._region.is_empty:  # 空区域与任何楔形相交仍为空
                self._done = len(self._nodes)
                break
            self._done += 1
        return self._region

    @property
    def vertices(self):
        """定位区域顶点坐标列表 [(x, y), ...]（统一为逆时针）；区域为空或退化时返回 []。
        若某顶点贴在包围盒上（|x| 或 |y| ≈ bound），说明该方向未被约束住、区域无界。"""
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

        区域是凸多边形，故直径等于顶点间最大两两距离，用 numpy 的成对距离矩阵一次求出。"""
        pts = np.asarray(self.vertices, dtype=float)
        if len(pts) < 2:
            return 0.0
        return float(np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1).max())

    @property
    def bounded(self):
        """定位区域是否被检测点完全框住（有界）。"""
        pts = self.vertices
        return bool(pts) and all(min(self.bound - abs(v) for v in p) > _TOUCH for p in pts)

    @property
    def enclosing_circle(self):
        """定位区域的最小覆盖圆 (cx, cy, r)；区域为空、无界或退化时返回 None。

        由 GEOS 的最小包围圆给出。问题 1 第二问“以定位区域直径为直径的圆能否覆盖此定位
        区域”的判据即 r <= diameter / 2。"""
        if not self.bounded:
            return None
        center = shapely.minimum_bounding_circle(self.region).centroid  # 正内接多边形的质心即圆心
        return (center.x, center.y, float(shapely.minimum_bounding_radius(self.region)))

    def contains(self, p):
        """点 p 是否落在定位区域内（区域已被包围盒截断，包围盒外的点判为不在区域内）。"""
        return bool(self.region.covers(Point(p)))

    # ---------- 基础几何 ----------

    def _wedge(self, x, y, theta):
        """单个检测点的楔形多边形：以 (x, y) 为顶点的三角形，边长足以覆盖整个包围盒。"""
        length = 2.0 * (hypot(x, y) + self.bound)
        a1 = radians(theta - self.err)
        a2 = radians(theta + self.err)
        return Polygon([(x, y),
                        (x + length * cos(a1), y + length * sin(a1)),
                        (x + length * cos(a2), y + length * sin(a2))])

    @staticmethod
    def bearing(a, b):
        """在 a 点观测 b 点的方位角（度，[0, 360)）：x 轴正向逆时针旋转到 a→b 的夹角。"""
        return degrees(atan2(b[1] - a[1], b[0] - a[0])) % 360.0

    def __repr__(self):
        return (f"TriangulationRegion(检测点 {len(self._nodes)} 个, 顶点 {len(self.vertices)} 个, "
                f"直径 {self.diameter:.4f} m, 有界 {self.bounded})")


if __name__ == "__main__":
    # 自测：先假定干扰源真值 G，反推各检测点示向度，保证定位区域非空
    G = (500.0, 400.0)
    sites = [(0.0, 0.0), (1000.0, 0.0), (200.0, 900.0), (800.0, 900.0)]
    region = TriangulationRegion.from_nodes(
        [(x, y, TriangulationRegion.bearing((x, y), G)) for x, y in sites])

    print(region)
    for i, (x, y) in enumerate(region.vertices, 1):
        print(f"  P{i} = ({x:.6f}, {y:.6f})")
    print(f"  面积 {region.area:.4f} m²，示向度误差 ±{region.err}°")
    print(f"两点交会：{TriangulationRegion.from_nodes(region.nodes[:2])}")

    circle = region.enclosing_circle
    if circle:
        cx, cy, r = circle
        print(f"最小覆盖圆：圆心 ({cx:.4f}, {cy:.4f})，半径 {r:.4f} m；"
              f"直径的一半 {region.diameter / 2:.4f} m")
        print(f"以定位区域直径为直径的圆能否覆盖定位区域："
              f"{'能' if r <= region.diameter / 2 + 1e-6 else '否'}")
    print(f"真值 G 是否落在定位区域内：{region.contains(G)}")
