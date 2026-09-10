"""T1.py —— 问题 1：交会定位法的定位区域（排序增量半平面交，O(n log n)）

原理
----
1. 检测点 S 处测得示向度 theta、测向误差 ±err（题目 err=1°），干扰源必落在以 S 为顶点、
   张角 2*err 的楔形内。因 2*err < 180°，楔形 = 两条边界射线给出的两个半平面之交；再
   并入一个大包围盒（保证区域有界），于是定位区域 = 全部半平面 {a*x + b*y <= d} 的交。

2. 排序增量法（sort-and-increment，半平面交的标准 O(n log n) 算法）：
   ① 排序：按半平面法向 (a, b) 的极角 atan2(b, a) 升序排列；同向平行的半平面只保留
      最紧的一个（d 最小者）。
   ② 增量插入：用双端队列维护"当前交集的边界平面序列"。加入新半平面 h 时，若队尾两个
      平面（或队首两个平面）的交点落在 h 的外侧，说明该平面已冗余，弹出之；重复直到
      交点位于 h 内侧，再把 h 压入队尾。队列中平面的先后顺序始终等于极角顺序。
   ③ 收尾：用队首约束再检查一次队尾、用队尾约束再检查队首（处理环形边界处新出现的
      冗余平面）。
   ④ 取解：队列中相邻两个半平面的交点就是定位区域的顶点，按队列顺序连成凸多边形
      （逆时针）。
   ⑤ 校验：所得顶点须满足全部半平面约束，否则交集为空区域，返回 []。

   复杂度：排序 O(n log n)、插入 O(n)，n = 4 + 2*检测点数。

注意：这里的"增量"指按极角顺序逐个插入半平面；跨检测点时（region + (x, y, theta)）新
半平面的极角位置不确定，排序结果会变，故仍需整体重算一次（一次 O(n log n)，微秒级）。

仅使用标准库。
"""

from collections import deque
from itertools import combinations
from math import atan2, cos, degrees, hypot, radians, sin

EPS = 1e-9  # 坐标单位为米，容差远小于题目精度要求


class TriangulationRegion:
    """交会定位区域：累积检测点（坐标 + 示向度），用排序增量法求定位区域凸多边形及几何量。"""

    def __init__(self, err=1.0, bound=1e5):
        """err：示向度误差半宽（度），对所有检测点相同；bound：包围盒半边长（米）。"""
        self.err = float(err)
        self.bound = float(bound)
        self._nodes = []
        self._vertices = None  # 惰性缓存：None 未算、[] 表示区域为空或退化

    # ---------- 检测点管理（加法返回新区域，原对象不变） ----------

    def __add__(self, node):
        """region + (x, y, theta)：加入一个检测点（坐标米、示向度度），返回新的定位区域。

        新检测点带来两个半平面，其极角位置不确定，故新区域需重跑一次排序增量（首次读取
        vertices 时进行，O(n log n)）。"""
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
        self._solve()
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

        问题 1 第二问“以定位区域直径为直径的圆能否覆盖此定位区域”的判据即
        r <= diameter / 2。顶点数很少（两点交会 4 个、多点交会一般不超过十几个），
        故直接枚举 2/3 顶点确定的候选圆，代价 O(n^4)。"""
        pts = self.vertices
        if not pts or not self.bounded:
            return None
        best = None
        for k in (2, 3):
            for combo in combinations(pts, k):
                c = self._circle_through(*combo)
                if c is None:
                    continue
                if (best is None or c[2] < best[2]) and \
                        all(hypot(p[0] - c[0], p[1] - c[1]) <= c[2] + 1e-7 for p in pts):
                    best = c
        return best

    def contains(self, p):
        """点 p 是否可能是干扰源，即是否落在所有检测点楔形之交内。"""
        return all(abs((self.bearing((x, y), p) - theta + 180.0) % 360.0 - 180.0)
                   <= self.err + EPS for x, y, theta in self._nodes)

    # ---------- 排序增量半平面交 ----------

    def _solve(self):
        """排序增量法求定位区域顶点（结果缓存）。"""
        if self._vertices is not None:
            return
        planes = self._sort_unique(self._planes())
        tol = 1e-9 * max(1.0, self.bound)  # 距离容差（米）
        hull = self._incremental(planes, tol)
        self._vertices = self._polygon(hull, planes, tol)

    @staticmethod
    def _incremental(planes, tol):
        """排序增量主循环：返回构成交集边界的半平面序列（双端队列内容）。"""
        dq = deque()
        for h in planes:
            # 队尾／队首已冗余的平面弹出（其与邻居的交点落在外侧）
            while len(dq) >= 2 and TriangulationRegion._outside(
                    TriangulationRegion._meet(dq[-2], dq[-1]), h, tol):
                dq.pop()
            while len(dq) >= 2 and TriangulationRegion._outside(
                    TriangulationRegion._meet(dq[0], dq[1]), h, tol):
                dq.popleft()
            dq.append(h)
        # 收尾：环形边界处可能又出现冗余平面
        while len(dq) >= 3 and TriangulationRegion._outside(
                TriangulationRegion._meet(dq[-2], dq[-1]), dq[0], tol):
            dq.pop()
        while len(dq) >= 3 and TriangulationRegion._outside(
                TriangulationRegion._meet(dq[0], dq[1]), dq[-1], tol):
            dq.popleft()
        return list(dq)

    @classmethod
    def _polygon(cls, hull, planes, tol):
        """队列中相邻半平面求交得到区域顶点；不成多边形或违反约束时返回 []。"""
        n = len(hull)
        if n < 3:
            return []
        pts = []
        for i in range(n):
            v = cls._meet(hull[i], hull[(i + 1) % n])
            if v is None:  # 相邻平面平行（正常已在去重时排除）
                return []
            if not pts or hypot(v[0] - pts[-1][0], v[1] - pts[-1][1]) > EPS:
                pts.append(v)
        if len(pts) > 2 and hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= EPS:
            pts.pop()
        signed = cls._signed_area(pts) if len(pts) > 2 else 0.0
        if abs(signed) <= EPS * max(1.0, tol):
            return []
        if signed < 0:
            pts.reverse()  # 统一为逆时针
        # 校验：所得顶点必须满足全部约束，否则交集实为空区域
        if any(a * v[0] + b * v[1] > d + 1e3 * tol for v in pts for a, b, d in planes):
            return []
        return pts

    def _planes(self):
        """全部半平面约束 (a, b, d)（已归一化为 a*x + b*y <= d、|(a,b)| = 1）。含包围盒。"""
        b = self.bound
        planes = [(1.0, 0.0, b), (-1.0, 0.0, b), (0.0, 1.0, b), (0.0, -1.0, b)]
        for x, y, theta in self._nodes:
            for nx, ny in self._wedge_normals(theta):
                # 楔形内部 n·(p - S) >= 0 改写为 (-n)·p <= -n·S
                planes.append((-nx, -ny, -(nx * x + ny * y)))
        return planes

    def _wedge_normals(self, theta):
        """楔形两条边界射线的内法向：内部满足 n·(p - S) >= 0（单位向量）。"""
        for angle, side in ((theta - self.err, -1.0), (theta + self.err, 1.0)):
            r = radians(angle)
            yield side * sin(r), -side * cos(r)

    @staticmethod
    def _sort_unique(planes):
        """按法向极角升序排序；同向平行的半平面只保留最紧的一个。"""
        ordered = sorted(planes, key=lambda h: atan2(h[1], h[0]))
        out = []
        for h in ordered:
            if out:
                a1, b1, d1 = out[-1]
                cross = a1 * h[1] - b1 * h[0]
                dot = a1 * h[0] + b1 * h[1]
                if abs(cross) <= EPS and dot > 0.0:  # 同向平行
                    if h[2] < d1:  # 新平面更紧则替换，否则丢弃
                        out[-1] = h
                    continue
            out.append(h)
        return out

    @staticmethod
    def _meet(h1, h2):
        """两条边界直线 a1*x + b1*y = d1 与 a2*x + b2*y = d2 的交点；平行返回 None。"""
        a1, b1, d1 = h1
        a2, b2, d2 = h2
        det = a1 * b2 - a2 * b1
        if abs(det) <= EPS:
            return None
        return ((d1 * b2 - d2 * b1) / det, (a1 * d2 - a2 * d1) / det)

    @staticmethod
    def _outside(p, h, tol):
        """点 p 是否在半平面 h 的外侧（严格超出容差）。p 为 None（平行）时返回 False。"""
        return p is not None and h[0] * p[0] + h[1] * p[1] > h[2] + tol

    @staticmethod
    def _signed_area(poly):
        """多边形有向面积（逆时针为正）。"""
        n = len(poly)
        return sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                   for i in range(n)) / 2.0

    @staticmethod
    def bearing(a, b):
        """在 a 点观测 b 点的方位角（度，[0, 360)）：x 轴正向逆时针旋转到 a→b 的夹角。"""
        return degrees(atan2(b[1] - a[1], b[0] - a[0])) % 360.0

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

    def __repr__(self):
        pts = self.vertices
        return (f"TriangulationRegion(检测点 {len(self._nodes)} 个, 顶点 {len(pts)} 个, "
                f"直径 {self.diameter:.4f} m, 有界 {self.bounded})")


if __name__ == "__main__":
    # 自测：先假定干扰源真值 G，反推各检测点示向度，保证定位区域非空
    G = (500.0, 400.0)
    sites = [(0.0, 0.0), (1000.0, 0.0), (200.0, 900.0), (800.0, 900.0)]

    region = TriangulationRegion()
    for x, y in sites:
        region = region + (x, y, TriangulationRegion.bearing((x, y), G))
    print(region)
    for i, (x, y) in enumerate(region.vertices, 1):
        print(f"  P{i} = ({x:.6f}, {y:.6f})")
    print(f"  面积 {region.area:.4f} m²，示向度误差 ±{region.err}°")

    # 排序增量法的中间量：半平面数 → 去重后 → 参与构成区域的平面数（即顶点数）
    planes = region._planes()
    hull = region._incremental(region._sort_unique(planes), 1e-9 * max(1.0, region.bound))
    print(f"  半平面 {len(planes)} 个 → 排序去重后 {len(region._sort_unique(planes))} 个 "
          f"→ 构成区域的边界平面 {len(hull)} 个")

    print(f"两点交会：{TriangulationRegion.from_nodes(region.nodes[:2])}")

    circle = region.enclosing_circle
    if circle:
        cx, cy, r = circle
        print(f"最小覆盖圆：圆心 ({cx:.4f}, {cy:.4f})，半径 {r:.4f} m；"
              f"直径的一半 {region.diameter / 2:.4f} m")
        print(f"以定位区域直径为直径的圆能否覆盖定位区域："
              f"{'能' if r <= region.diameter / 2 + 1e-6 else '否'}")
    print(f"真值 G 是否落在定位区域内：{region.contains(G)}")
