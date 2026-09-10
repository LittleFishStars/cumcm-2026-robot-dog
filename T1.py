"""T1.py —— 问题 1：交会定位法的定位区域（基于 shapely/GEOS 与 numpy）

原理
----
1. 题目给的目标区域是以原点为圆心、半径 1800 m 的圆域（**直径 3600 m**）。检测点 S 处测得
   示向度 theta、测向误差 ±err（题目 err=1°），干扰源必落在以 S 为顶点、张角 2*err 的楔形
   内。把楔形写成一个（足够长的）三角形多边形，则

       定位区域 = 所有楔形之交 ∩ 目标圆域

   目标圆域用 shapely 的 buffer 生成内接正 64 边形：`Point(0, 0).buffer(1800, quad_segs=16)`
   —— 64 个顶点精确落在半径 1800 的圆上，多边形横跨 3600 m；边是弦，与真圆的最大偏差为
   1800*(1-cos(π/64)) ≈ 2.17 m（相对半径约 0.12%），多边形面积/圆面积 ≈ 0.9984。

2. 几何计算全部交给成熟库，不手写几何算法：
   ● 交集       → shapely（GEOS 布尔运算），且交集满足结合律，新检测点只需与已有区域补裁一次
   ● 面积       → region.area
   ● 直径       → numpy 在区域顶点上求最大两两距离（区域是凸多边形，故顶点对最大距离即直径）
   ● 最小覆盖圆  → shapely.minimum_bounding_circle / minimum_bounding_radius（GEOS 最小包围圆）
   ● 点包含判定  → region.covers
   ● 有界判定    → 顶点到目标区域边界的距离（shapely distance）

3. 增量性：`region` 惰性求解且缓存"已并入多少个检测点"，`__add__` 把该缓存传给新对象，
   故 `r = r + node` 后读取 `r.region` 只会对新楔形求一次交，不重算历史约束。

关于"有界"（bounded）
--------------------
区域已被目标圆域截断，故 `area` / `diameter` / `enclosing_circle` 描述的都是**这一截断后的
区域**，恒为有限值。`bounded` 另行回答一个不同的问题：**定位是否只靠检测点就确定了**——
若区域有顶点落在目标区域边界上（被圆域截断），说明该方向的约束不足（只测一次示向度、
或检测点与干扰源近似共线时），此时 `bounded=False`。

依赖：shapely>=2.1、numpy（见 pyproject.toml；用 `uv run T1.py` 或 `.venv/bin/python T1.py` 运行）
"""

from math import atan2, cos, degrees, hypot, radians, sin

import numpy as np
import shapely
from shapely import Point, Polygon

# 顶点贴合目标区域边界的判定阈值（米）：小于该值即认为区域被圆域截断、约束不足
_TOUCH = 1.0

# 目标区域默认参数（题目给定）：半径 1800 m 的圆域
RADIUS = 1800.0
SIDES = 64


class TriangulationRegion:
    """交会定位区域：累积检测点（坐标 + 示向度），求定位区域凸多边形及其几何量。"""

    def __init__(self, err=1.0, radius=RADIUS, sides=SIDES):
        """err：示向度误差半宽（度），对所有检测点相同。

        radius：目标圆域半径（米），题目为 1800（即圆域直径 3600 m）。
        sides ：目标圆域的内接正多边形边数，须为 4 的倍数，默认 64（用 quad_segs=16 生成）。
        """
        if sides % 4:
            raise ValueError("sides 必须是 4 的倍数（buffer 生成内接多边形时 quad_segs = sides/4）")
        self.err = float(err)
        self.radius = float(radius)
        self.sides = int(sides)
        self._nodes = []
        self._clip = None    # 目标圆域的内接正多边形（惰性生成）
        self._region = None  # 已并入前 _done 个检测点的交集；None 表示尚未求解
        self._done = 0

    # ---------- 检测点管理（加法返回新区域，原对象不变） ----------

    def __add__(self, node):
        """region + (x, y, theta)：加入一个检测点（坐标米、示向度度），返回新的定位区域。"""
        x, y, theta = (float(v) for v in node)
        new = self.__class__(self.err, self.radius, self.sides)
        new._nodes = self._nodes + [(x, y, theta % 360.0)]
        new._clip = self._clip
        new._region, new._done = self._region, self._done  # 继承缓存，新点只需补裁
        return new

    @classmethod
    def from_nodes(cls, nodes, err=1.0, radius=RADIUS, sides=SIDES):
        """由检测点列表 [(x, y, theta), ...] 构造定位区域。"""
        region = cls(err, radius, sides)
        for node in nodes:
            region = region + node
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
        """定位区域的 shapely 几何（Polygon；区域为空返回空几何）。"""
        if self._region is None:
            # 楔形张角为 0（err<=0）时区域退化为零面积
            self._region = self.clip if self.err > 0.0 else Polygon()
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
        若某顶点落在目标区域边界上（到目标区域边界距离 ≈ 0），说明该方向未被检测点约束住、
        区域是被目标区域截断的（见 bounded）。"""
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
        """定位是否只靠检测点就已确定（区域未被目标圆域截断）。

        有顶点落在目标区域边界上即说明该方向约束不足（例如只测一次示向度、或检测点与干扰源
        近似共线），此时返回 False；区域为空同样返回 False。"""
        pts = self.vertices
        if not pts:
            return False
        boundary = self.clip.boundary
        return all(boundary.distance(Point(*p)) > _TOUCH for p in pts)

    @property
    def enclosing_circle(self):
        """定位区域的最小覆盖圆 (cx, cy, r)；区域为空或退化时返回 None。

        由 GEOS 的最小包围圆给出。问题 1 第二问“以定位区域直径为直径的圆能否覆盖此定位
        区域”的判据即 r <= diameter / 2。"""
        if not self.vertices:
            return None
        center = shapely.minimum_bounding_circle(self.region).centroid  # 正内接多边形的质心即圆心
        return (center.x, center.y, float(shapely.minimum_bounding_radius(self.region)))

    def contains(self, p):
        """点 p 是否落在定位区域内。

        注意：目标圆域是内接 64 边形，真圆上靠边的小月牙区域（宽 ≤ 2.17 m）不在多边形内。"""
        return bool(self.region.covers(Point(p)))

    # ---------- 基础几何 ----------

    def _wedge(self, x, y, theta):
        """单个检测点的楔形多边形：以 (x, y) 为顶点的三角形，边长足以覆盖整个目标圆域。"""
        length = 2.0 * (hypot(x, y) + self.radius)
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

