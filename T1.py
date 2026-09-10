"""T1.py —— 问题 1：交会定位法的定位区域（排序增量半平面交，O(n log n)）

原理
----
1. 检测点 S 处测得示向度 theta、测向误差 ±err（题目 err=1°），干扰源必落在以 S 为顶点、
   张角 2*err 的楔形内。因 2*err < 180°，楔形 = 两条边界射线给出的两个半平面之交；再并入
   一个大包围盒（保证区域有界），于是 定位区域 = 全部半平面 {a*x + b*y <= d} 的交。

2. 排序增量法（半平面交的标准算法，O(n log n)）：按法向 (a, b) 的极角升序排序，同向平行
   的半平面只留最紧的一个；再用双端队列逐个插入——新半平面若使队尾（或队首）两个平面
   已冗余（其交点落在外侧）就弹出，全部插入后再用队首约束查队尾、队尾约束查队首，处理
   环形边界。队列中相邻半平面的交点就是定位区域顶点，按队列顺序构成逆时针凸多边形。

3. 新检测点带来的半平面极角位置不定、会改变排序，故 `region + (x, y, theta)` 后需重跑一次
   O(n log n) 求解（惰性执行，只在首次读取结果时算）；n = 4 + 2*检测点数。

仅使用标准库。
"""

from collections import deque
from itertools import combinations
from math import atan2, cos, degrees, hypot, radians, sin

EPS = 1e-9  # 坐标单位为米，容差远小于题目精度要求


class TriangulationRegion:
    """交会定位区域：累积检测点（坐标 + 示向度），求定位区域凸多边形及其几何量。"""

    def __init__(self, err=1.0, bound=1e5):
        """err：示向度误差半宽（度），对所有检测点相同；bound：包围盒半边长（米）。"""
        self.err = float(err)
        self.bound = float(bound)
        self._nodes = []
        self._vertices = None  # 惰性缓存；[] 表示区域为空或退化

    # ---------- 检测点管理（加法返回新区域，原对象不变） ----------

    def __add__(self, node):
        """region + (x, y, theta)：加入一个检测点（坐标米、示向度度），返回新的定位区域。"""
        x, y, theta = (float(v) for v in node)
        new = self.__class__(self.err, self.bound)
        new._nodes = self._nodes + [(x, y, theta % 360.0)]
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
    def vertices(self):
        """定位区域顶点坐标列表 [(x, y), ...]（逆时针）；区域为空或退化时返回 []。
        若某顶点贴在包围盒上（|x| 或 |y| ≈ bound），说明该方向未被约束住、区域无界。"""
        if self._vertices is None:
            self._vertices = self._solve()
        return list(self._vertices)

    @property
    def area(self):
        """定位区域面积（平方米）。"""
        pts = self.vertices
        return abs(self._signed_area(pts)) if len(pts) > 2 else 0.0

    @property
    def diameter(self):
        """定位区域直径：区域内任意两点距离的最大值（凸多边形即顶点间最大距离）。"""
        pts = self.vertices
        return max((hypot(p[0] - q[0], p[1] - q[1]) for p, q in combinations(pts, 2)),
                   default=0.0)

    @property
    def bounded(self):
        """定位区域是否被检测点完全框住（有界）。"""
        pts = self.vertices
        return bool(pts) and all(min(self.bound - abs(v) for v in p) > 1.0 for p in pts)

    @property
    def enclosing_circle(self):
        """定位区域的最小覆盖圆 (cx, cy, r)；区域为空、无界或退化时返回 None。

        问题 1 第二问“以定位区域直径为直径的圆能否覆盖此定位区域”的判据即 r <= diameter/2。
        顶点数很少，故直接枚举 2/3 顶点定出的候选圆取最小者。"""
        pts = self.vertices
        if not pts or not self.bounded:
            return None
        best = None
        for k in (2, 3):
            for combo in combinations(pts, k):
                c = self._circle_through(*combo)
                if c and (best is None or c[2] < best[2]) and \
                        all(hypot(p[0] - c[0], p[1] - c[1]) <= c[2] + 1e-7 for p in pts):
                    best = c
        return best

    def contains(self, p):
        """点 p 是否可能是干扰源，即是否落在所有检测点楔形之交内。"""
        return all(abs((self.bearing((x, y), p) - theta + 180.0) % 360.0 - 180.0)
                   <= self.err + EPS for x, y, theta in self._nodes)

    # ---------- 排序增量半平面交 ----------

    def _solve(self):
        """求定位区域顶点：排序增量半平面交，返回逆时针顶点列表；空区域返回 []。"""
        b = self.bound
        # 半平面 (a, b, d) 表示 a*x + b*y <= d，内含包围盒以保证区域有界
        planes = [(1.0, 0.0, b), (-1.0, 0.0, b), (0.0, 1.0, b), (0.0, -1.0, b)]
        for x, y, theta in self._nodes:
            for angle, side in ((theta - self.err, -1.0), (theta + self.err, 1.0)):
                # 楔形内部满足 n·(p - S) >= 0，改写为 (-n)·p <= -n·S
                r = radians(angle)
                nx, ny = side * sin(r), -side * cos(r)
                planes.append((-nx, -ny, -(nx * x + ny * y)))

        # ① 按法向极角排序；同向平行的半平面只保留最紧的一个
        planes.sort(key=lambda h: atan2(h[1], h[0]))
        ordered = []
        for h in planes:
            if ordered:
                a1, b1, d1 = ordered[-1]
                if abs(a1 * h[1] - b1 * h[0]) <= EPS and a1 * h[0] + b1 * h[1] > 0.0:
                    if h[2] < d1:
                        ordered[-1] = h
                    continue
            ordered.append(h)

        tol = 1e-9 * max(1.0, b)  # 距离容差（米）

        def meet(h1, h2):
            """两半平面边界直线的交点；平行返回 None。"""
            det = h1[0] * h2[1] - h2[0] * h1[1]
            if abs(det) <= EPS:
                return None
            return ((h1[2] * h2[1] - h2[2] * h1[1]) / det,
                    (h1[0] * h2[2] - h2[0] * h1[2]) / det)

        def outside(p, h):
            """点 p 是否在半平面 h 的外侧。"""
            return p is not None and h[0] * p[0] + h[1] * p[1] > h[2] + tol

        # ② 逐个插入；队尾/队首已冗余的平面随之弹出
        dq = deque()
        for h in ordered:
            while len(dq) >= 2 and outside(meet(dq[-2], dq[-1]), h):
                dq.pop()
            while len(dq) >= 2 and outside(meet(dq[0], dq[1]), h):
                dq.popleft()
            dq.append(h)
        # ③ 收尾：处理环形边界处新暴露的冗余平面
        while len(dq) >= 3 and outside(meet(dq[-2], dq[-1]), dq[0]):
            dq.pop()
        while len(dq) >= 3 and outside(meet(dq[0], dq[1]), dq[-1]):
            dq.popleft()

        # ④ 相邻半平面求交得顶点
        n = len(dq)
        if n < 3:
            return []
        pts = []
        for i in range(n):
            v = meet(dq[i], dq[(i + 1) % n])
            if v is None:  # 相邻平行（去重后不应出现）
                return []
            if not pts or hypot(v[0] - pts[-1][0], v[1] - pts[-1][1]) > EPS:
                pts.append(v)
        if len(pts) > 2 and hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= EPS:
            pts.pop()

        # ⑤ 校验：退化或违反任一约束即视为空区域，否则统一为逆时针
        sa = self._signed_area(pts) if len(pts) > 2 else 0.0
        if abs(sa) <= EPS or any(a * v[0] + bb * v[1] > d + 1e3 * tol
                                 for v in pts for a, bb, d in ordered):
            return []
        if sa < 0:
            pts.reverse()
        return pts

    # ---------- 基础几何 ----------

    @staticmethod
    def _signed_area(poly):
        """多边形有向面积（逆时针为正）。"""
        n = len(poly)
        return sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                   for i in range(n)) / 2.0

    @staticmethod
    def _circle_through(*pts):
        """过 2 个顶点的最小圆（以连线为直径）/ 过 3 个顶点的外接圆；三点共线返回 None。"""
        if len(pts) == 2:
            (ax, ay), (bx, by) = pts
            return ((ax + bx) / 2.0, (ay + by) / 2.0, hypot(ax - bx, ay - by) / 2.0)
        (ax, ay), (bx, by), (cx, cy) = pts
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < EPS:
            return None
        a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
        ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
        uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
        return (ux, uy, hypot(ux - ax, uy - ay))

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
