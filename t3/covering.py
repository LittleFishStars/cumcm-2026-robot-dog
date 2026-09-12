"""覆盖圆的求解与校验（阶段一的几何底座）。

问题：用尽量少的半径 1000 m 的圆盖住半径 1800 m 的作业圆域，使**圆域内任意点**到最近圆心
的距离 ≤ 1000 m（= 有效接收半径下界），从而走到任一圆心都能听到全域的源。

最少个数由 disk covering problem 的已证明最优值确定为 7（见 min_circle_count）。圆心的**摆放**
则是：让 7 个点在保证"圆域内任一点到最近圆心 ≤ 1000 m"的前提下，使从原点出发走遍它们的开放
路径尽量短，同时给各方位源良好的交会几何。经典的正六边形族（1 个中心圆 + 6 个环圆）里程恒为
6d、最好 6737.73 m；现有两种一般 7 点布局（CLI `--layout` 切换，**缺省 uniform**）：
* uniform（SURVEY_CENTERS_UNIFORM）：7 点均匀分布在半径 1000 m 的圆上（正七边形），里程
  ~6207 m、零余量，但圆周对称 → 定位误差更均衡（实测 −14%）；
* optimized（SURVEY_CENTERS）：把 7 点推到距原点约 1000 m 处的数值优化布局，里程 ~6167 m、
  余量 5 m。
本模块负责：解析地给出最坏最近距离 D(d) = max(d/√3, g₂(d))、可行环半径区间、最优环半径 d*，
以及**在连续圆域上**求最坏点的数值校验（网格上"看起来满足"不等于满足 —— 曾因此漏掉圆域
边缘的源）。所有结果显示在 print_cover_report 里。

    from t3.covering import solve_covering_circles
    res = solve_covering_circles()          # 缺省用 config.SURVEY_CENTERS_UNIFORM（均匀布局）
    res = solve_covering_circles(use_uniform=False)  # 或优化布局（config.SURVEY_CENTERS）
    res = solve_covering_circles(1200.0)    # 或指定六边形族的环半径（对照/扫描）

注：`nearest_order` / `path_length` 复用 common.routing 中的同名实现（后者按公共模块命名
为 open_path_length，这里保留别名以贴合"走遍圆心的总里程"这一语义）。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from common.routing import dist_matrix, exact_open_order, nearest_order
from common.routing import open_path_length as path_length
from t3.config import (BOUNDARY_SAMPLES, CHOSEN_RING_RADIUS, COARSE_BOUNDARY,
                             COARSE_STEP, COVER_RADIUS, DISK_RATIO_5, DISK_RATIO_6,
                             DISK_RATIO_7, GRID_STEP, REGION_RADIUS, SURVEY_CENTERS,
                             SURVEY_CENTERS_UNIFORM, SURVEY_ROUTE_M, SURVEY_ROUTE_UNIFORM,
                             SURVEY_WORST_M, SURVEY_WORST_UNIFORM, TOL)


@dataclass(frozen=True, eq=False)
class CoverPlan:
    """覆盖圆方案：所有覆盖圆的圆心的位置（这些圆心就是机器狗的巡视路点）。

    两种构造方式：
      * centers 给定时按给定坐标摆放（**默认**，见 config.SURVEY_CENTERS 的 7 点一般布局）；
      * centers 为 None 时按"1 个中心 + 6 个正六边形环心"生成（ring_radius 为环半径，
        可用于对照或参数扫描）。

    两者都保证"圆域内任一点到最近圆心的距离 ≤ cover_radius"，即走遍圆心就必不漏源。
    """

    region_radius: float            # 目标圆域半径 / m
    cover_radius: float             # 覆盖圆半径（= 有效接收半径下界）/ m
    ring_radius: float              # 六边形环上圆心到原点的距离 d / m（仅六边形族用）
    rotation: float = 0.0           # 布局整体绕原点的旋转角 / rad
    layout: tuple[tuple[float, float], ...] | None = None
    #   ^ 显式给定的圆心（一般布局）。为 None 时按"1 中心 + 6 环"的六边形族由 ring_radius 生成
    waypoints: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """按 layout 或 ring_radius 算出各覆盖圆的圆心并写入冻结字段 waypoints"""
        if self.layout is None:
            wp = hex_layout(self.ring_radius, self.rotation)
        else:
            wp = np.asarray(self.layout, dtype=float)
            if abs(self.rotation) > 1e-12:
                c, s = math.cos(self.rotation), math.sin(self.rotation)
                wp = wp @ np.array([[c, s], [-s, c]])      # 绕原点旋转
        object.__setattr__(self, "waypoints", wp)

    def rotated(self, rotation: float) -> "CoverPlan":
        """返回把整个布局绕原点旋转到给定角度的新方案。

        旋转不改变覆盖条件（只取决于点间距离与点到原点的距离）与巡视路径长度，
        故旋转只是"把站点的朝向对准源密集方向"的零成本自由度。
        """
        return replace(self, rotation=float(rotation))

    @property
    def centers(self) -> list[tuple[float, float]]:
        """各覆盖圆的圆心坐标列表（巡视路点，顺序与 waypoints 一致）"""
        return [(float(x), float(y)) for x, y in self.waypoints]


def hex_layout(ring_radius: float, rotation: float = 0.0,
               n_ring: int = 6) -> np.ndarray:
    """正六边形布局的圆心：第 0 个在原点，其余 n_ring 个在半径 ring_radius 的环上。

    `rotation` 把整个环绕原点转一个角度（正六边形有 6 重旋转对称，故只有模 2π/n_ring 有区别）。
    中心圆恒在原点，所以旋转不改变覆盖保证，也不改变巡视里程 6d —— 但它决定"环心朝向哪里"，
    从而决定巡视路线扫过哪一片区域、最后停在哪个方向（见 strategy._align_face）。
    """
    ang = rotation + np.arange(n_ring) * (2.0 * math.pi / n_ring)
    ring = np.stack((ring_radius * np.cos(ang), ring_radius * np.sin(ang)), axis=1)
    return np.vstack([[0.0, 0.0], ring])


def optimal_ring_radius(region_radius: float = REGION_RADIUS) -> float:
    """最优环半径 d* = √3·R/2：使最坏最近距离 D(d) 最小的解析解。"""
    return math.sqrt(3.0) * region_radius / 2.0


def feasible_ring_interval(region_radius: float = REGION_RADIUS,
                           cover_radius: float = COVER_RADIUS) -> tuple[float, float]:
    """满足覆盖保证 D(d) ≤ r 的环半径区间 [d_min, d_max]（解析）。

    两个约束各给一段区间，取交集：
      内圈约束 d/√3 ≤ r           → d ≤ √3·r
      边界约束 g₂(d) ≤ r（解 d² - √3R·d + R² - r² = 0）
                                  → d ∈ [(√3R - √(4r²-R²))/2, (√3R + √(4r²-R²))/2]
    区间两端点都是零余量（最坏距离恰好 = r），并随 d 单调变化。
    """
    root = math.sqrt(max(4.0 * cover_radius ** 2 - region_radius ** 2, 0.0))
    lo = (math.sqrt(3.0) * region_radius - root) / 2.0
    hi = min(math.sqrt(3.0) * cover_radius,
             (math.sqrt(3.0) * region_radius + root) / 2.0)
    return lo, hi


def analytic_worst(ring_radius: float, region_radius: float = REGION_RADIUS) -> float:
    """最坏最近距离的解析值 D(d) = max(d/√3, g₂)（推导见模块文档）。

    d/√3 是内圈（只有中心圆能覆盖到）的最坏距离；g₂ 是圆域边界上、与某环心夹角 30° 的点
    到最近环心的距离。两项在 d* = √3·R/2 处同时等于 R/2，即最优点的最坏距离。
    """
    def g(rho: float) -> float:
        """半径 rho 的圆域边界点、与最近环心夹角 30° 时到该环心的距离（解析的边界项 g₂）"""
        return math.sqrt(max(rho * rho + ring_radius * ring_radius
                             - math.sqrt(3.0) * rho * ring_radius, 0.0))
    return max(ring_radius / math.sqrt(3.0), g(region_radius))


def min_circle_count(region_radius: float = REGION_RADIUS,
                     cover_radius: float = COVER_RADIUS) -> dict[str, Any]:
    """覆盖半径 R 的圆域最少需要几个半径 r 的圆盘（本题 r/R = 5/9 = 0.5555556）。

    这是经典的 **disk covering problem**（用若干等半径圆盘覆盖一个圆）。k = 5、6 的最优值
    都已被证明，k 个圆盘能覆盖一个圆所需的最小半径比 ρ_k = r/R 为

        ρ_5 = 0.6093828641（Bezdek 1983）  ρ_6 = 0.5559052114（Bezdek 1979）  ρ_7 = 0.5

    判定因此只看 5/9 落在哪两个 ρ_k 之间：

        ρ_7 = 0.5000 < 5/9 = 0.5555556 < 0.5559052 = ρ_6

    ⇒ **7 个够、6 个不够，故最少 7 个**。值得注意的是 6 个只差一点点：即便六个圆盘摆到最优，
    也需要覆盖半径 ≥ ρ_6·R = 1000.629 m，只比可用的 1000 m 多 0.629 m（0.0629%）。
    换句话说，"再少几个圆也该行"的直觉几乎是对的 —— 只差千分之零点六，但仍是不够。

    只靠面积或弧长只能给出更弱的下界（面积/密度 → n ≥ 4；边界弧 → n ≥ 6），
    都不足以排除 6 个，必须用上面已证明的最优值。

    上面是引用文献的判定。为避免"只有引用、没有自查"，另做了一次**独立的数值搜索**复核
    （归一化到 R=1，圆心不设圆域约束 —— 这是比本题更宽松的情形，故其最优值必 ≤ 受约束时的
    最优值，构成下界）：以罗盘模式搜索从大量随机起点与结构族出发最小化最坏距离，结果为
    k=3 → 0.8660254（= √3/2 ✓）、k=4 → 0.7071425（≈ 1/√2）、k=5 → 0.6102517、
    k=6 → 0.5575510、k=7 → 0.5000000（= 1/2 ✓），与文献值分别相差 2e-16、3.6e-5、8.7e-4、
    1.7e-3、2e-16。搜索在 k=3、k=7 上精确复现解析最优，故求值器与搜索可信；k=5、k=6 的
    搜索值略高于证明值属正常的局部最优残留，但**方向一致且已足以判定**：即便按这个（比证明值
    更大的）数值结果，6 个圆盘也只能做到 0.5575510 > 5/9 = 0.5555556，6 个依然不够。

    判定所依据的两个比值差距很小，故这里格外注意精度：5/9 = 0.5555555556 与
    ρ_6 = 0.5559052114 只差 3.5e-4，用 4 位小数的近似值（0.5556 vs 0.5559）虽然结论相同，
    但余量会被吃掉大半；因此常量一律写足位数并注明来源。
    """
    ratio = cover_radius / region_radius
    arc_deg = 2.0 * math.degrees(math.asin(min(ratio, 1.0)))
    six_need = DISK_RATIO_6 * region_radius
    return {
        "min_count": 7,
        "ratio_r_over_R": round(ratio, 10),
        "disk_ratios": {"5": DISK_RATIO_5, "6": DISK_RATIO_6, "7": DISK_RATIO_7},
        "boundary_arc_lower": int(math.ceil(360.0 / arc_deg)),
        "area_density_lower": int(math.ceil(2.0 * math.pi / (3.0 * math.sqrt(3.0))
                                            * (region_radius / cover_radius) ** 2)),
        "six_required_cover_radius_m": round(six_need, 4),
        "six_shortfall_m": round(six_need - cover_radius, 4),
        "six_shortfall_ratio": round(six_need / cover_radius - 1.0, 6),
        "six_max_region_radius_m": round(cover_radius / DISK_RATIO_6, 4),
        "seven_required_cover_radius_m": round(DISK_RATIO_7 * region_radius, 4),
    }


SECTOR_HALF_DEG = 30.0          # 扇区半宽 / 度（正六边形相邻环心隔 60°，各管一半）
def _sector_score(bearings_rad: np.ndarray, phi: float) -> tuple[int, float]:
    """以 phi 为扇区中心的评分：(装入的源个数, 这些源的 ΣcosΔθ)。

    个数是主指标（"哪一片源最多"），ΣcosΔθ 只在个数相同时破平（片内的源越居中越好）。
    """
    d = (bearings_rad - phi + math.pi) % (2.0 * math.pi) - math.pi
    inside = np.abs(d) <= math.radians(SECTOR_HALF_DEG)
    cnt = int(inside.sum())
    return cnt, (float(np.cos(d[inside]).sum()) if cnt else 0.0)


def dense_sector_rotation(bearings: Sequence[float],
                          step_deg: float = 1.0) -> float | None:
    """选一个旋转角，使某个环心正对"听到的源最多"的 60° 扇区。

    6 个环心彼此相隔 60°，故"环心能对准哪些方向"只由旋转角模 60° 决定。于是把圆周切成 360/60
    个候选 60° 扇区（1° 一格，等效地把整个圆周扫一遍），取装入源最多的那一片的中心方向 θ*，
    返回 θ* 本身（落在 [0, 2π)）作为方向角。具体把哪一条旋转量施加到布局上由
    strategy._orient_route 决定（它同时还要选落脚站）；本函数只负责回答"源最密集的方向是哪个"。

    为什么按"个数最多"而不是"方位角均值"：均值是合向量方向，面对两个相距几十度的等量簇时
    会落在两簇之间（谁也没对准）；本题要的是"把圆心摆到源密集的那一侧"，取众数才符合意图。
    同分时用扇区内 ΣcosΔθ 破平，再尝试用片内源的质量中心做一次亚度精修（只在个数不减少时采纳）。

    入参是示向度序列（度）。一个源都没听到时返回 None（无从判断，保持 0°）。
    """
    if not len(bearings):
        return None
    beta = np.asarray([math.radians(float(b)) for b in bearings], dtype=float)
    # 候选：把整个圆周按 1° 扫一遍（每个候选都是"某个环心正对的方向"）。严格大于才替换，
    # 故同分时取角度最小的那个，结果确定。
    best = None
    for k in range(int(round(360.0 / step_deg))):
        phi = k * math.radians(step_deg)
        score = _sector_score(beta, phi)
        if best is None or score > best[0]:
            best = (score, phi)
    score0, phi0 = best
    # 亚度精修：把扇区中心挪到片内源的质量中心（片内跨度 ≤ 60°，圆均值无歧义）。
    # 只在"装入个数不减少"时采纳，避免挪动窗口反而漏掉边缘的源。
    d = (beta - phi0 + math.pi) % (2.0 * math.pi) - math.pi
    inside = np.abs(d) <= math.radians(SECTOR_HALF_DEG)
    if inside.any():
        phi1 = phi0 + math.atan2(float(np.sin(d[inside]).mean()),
                                 float(np.cos(d[inside]).mean()))
        if _sector_score(beta, phi1) > score0:
            phi0 = phi1
    return float(phi0 % (2.0 * math.pi))


@lru_cache(maxsize=8)
def region_samples(step: float, n_boundary: int) -> np.ndarray:
    """目标圆域的采样点：细网格 + 圆边界均匀采样（最坏点常在边界上，须精细采样）。"""
    ax = np.arange(-REGION_RADIUS, REGION_RADIUS + TOL, step)
    gx, gy = np.meshgrid(ax, ax)
    pts = np.stack((gx.ravel(), gy.ravel()), axis=1)
    # 圆域判据用平方距离（同 cover_counts：TOL 余量远大于浮点末位，逐点同判）
    pts = pts[(pts[:, 0] * pts[:, 0] + pts[:, 1] * pts[:, 1]) <= (REGION_RADIUS + TOL) ** 2]
    t = np.linspace(0.0, 2.0 * math.pi, n_boundary, endpoint=False)
    ring = np.stack((REGION_RADIUS * np.cos(t), REGION_RADIUS * np.sin(t)), axis=1)
    return np.vstack([pts, ring])


def nearest_distances(points: np.ndarray, waypoints: np.ndarray,
                      chunk: int = 200_000) -> np.ndarray:
    """每个采样点到最近圆心的距离（分块计算，避免大矩阵占内存）。

    实现上是"逐圆心一列、比平方距离、最后只对 N 个最小值开方"：
      * 不构造 `(N, n_circles, 2)` 三维临时数组，也不在对轴长约 2 的轴上做 `np.linalg.norm`
        的多维归约（那种归约要按行走通用累加路径，实测比逐列循环慢数倍）；
      * 开方放在最后 —— 开方单调，故 √(min d²) = min √(d²) 逐位相同，只是把 7N 次开方
        省成 N 次。
    """
    pts = np.asarray(points, dtype=float)
    wp = np.asarray(waypoints, dtype=float)
    out = np.empty(pts.shape[0], dtype=float)
    for i in range(0, pts.shape[0], chunk):
        blk = pts[i:i + chunk]
        best2 = np.full(blk.shape[0], np.inf, dtype=float)
        for cx, cy in wp:
            dx = blk[:, 0] - cx
            dy = blk[:, 1] - cy
            np.minimum(best2, dx * dx + dy * dy, out=best2)
        out[i:i + chunk] = np.sqrt(best2)
    return out


def worst_candidates(ring_radius: float, region_radius: float = REGION_RADIUS) -> np.ndarray:
    """解析给出的最坏点候选：12 个角平分线方向 × {ρ* = d/√3, ρ = R}。

    圆域内的最坏点必落在某个"角平分线方向"上（与最近环心夹角 30°）：内圈最坏点在 ρ* = d/√3
    处（此时它到最近环心的距离恰为 ρ*，与中心圆打平），外圈最坏点在圆域边界 ρ = R 处。把这
    些点放进采样集合，解析解与实算解就能一致到机器精度，网格只用于旁证。
    """
    ang = np.arange(12) * (math.pi / 6.0) + math.pi / 6.0
    pts = [(r * math.cos(a), r * math.sin(a))
           for r in (ring_radius / math.sqrt(3.0), region_radius) for a in ang]
    return np.array(pts, dtype=float)


def worst_candidates_general(centers: np.ndarray,
                            region_radius: float = REGION_RADIUS) -> np.ndarray:
    """任意圆心布局的**精确**最坏点候选集（不依赖网格采样）。

    函数 f(p) = min_i |p - c_i| 在圆域上的最大值必出现在以下三类点之一：
      1. 到三个圆心等距的点（三角形外心）落在圆域内 —— 该点是局部极大；
      2. 到两个圆心等距的中垂线与圆域边界的交点；
      3. 圆域内到某个圆心最远/最近的点，即沿 c_i 方向的原点与对径边界点（含原点本身）。
    把这三类点全部算入候选，实算最坏距离即可与真值一致到机器精度（网格只作旁证）。
    """
    C = np.asarray(centers, dtype=float)
    n = len(C)
    R = region_radius
    cand = [np.zeros(2)]
    # 1) 三角形外心
    for i, j, k in itertools.combinations(range(n), 3):
        a, b, c = C[i], C[j], C[k]
        det = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(det) < 1e-12:
            continue
        na, nb, nc = float(a @ a), float(b @ b), float(c @ c)
        p = np.array([(na * (b[1] - c[1]) + nb * (c[1] - a[1]) + nc * (a[1] - b[1])) / det,
                      (na * (c[0] - b[0]) + nb * (a[0] - c[0]) + nc * (b[0] - a[0])) / det])
        if float(p @ p) <= R * R + TOL:
            cand.append(p)
    # 2) 中垂线 ∩ 圆域边界；3) 圆心方向的边界点与原点
    for i in range(n):
        ci = C[i]
        ri = float(np.hypot(*ci))
        if ri > 1e-12:
            cand.append(ci / ri * R)
            cand.append(-ci / ri * R)
        for j in range(i + 1, n):
            v = C[j] - ci
            vn2 = float(v @ v)
            if vn2 < 1e-12:
                continue
            base = (0.5 * float(C[j] @ C[j] - ci @ ci) / vn2) * v
            perp = np.array([-v[1], v[0]]) / math.sqrt(vn2)
            rem = R * R - float(base @ base)
            if rem < -TOL:
                continue
            t = math.sqrt(max(rem, 0.0))
            cand.append(base + t * perp)
            cand.append(base - t * perp)
    return np.array(cand, dtype=float)


def cover_counts(points: np.ndarray, waypoints: np.ndarray, radius: float,
                 chunk: int = 200_000) -> np.ndarray:
    """每个采样点被几个覆盖圆同时覆盖（覆盖重数）。

    判据写成"平方距离 ≤ (r + TOL)²"：两侧都非负、且 TOL 的余量（d 上 1e-9、d² 上 2e-6）比
    浮点最后一位（d = 1000 处约 1e-13）大七个数量级，故与"距离 ≤ r + TOL"逐点同判。好处是不必
    对每个点都开方，也避免构造 `(N, n_circles, 2)` 三维临时数组。
    """
    pts = np.asarray(points, dtype=float)
    wp = np.asarray(waypoints, dtype=float)
    lim = (float(radius) + TOL) ** 2
    out = np.empty(pts.shape[0], dtype=np.int64)
    for i in range(0, pts.shape[0], chunk):
        blk = pts[i:i + chunk]
        cnt = np.zeros(blk.shape[0], dtype=np.int64)
        for cx, cy in wp:
            dx = blk[:, 0] - cx
            dy = blk[:, 1] - cy
            cnt += (dx * dx + dy * dy) <= lim
        out[i:i + chunk] = cnt
    return out


def nearest_order(waypoints: np.ndarray, start: Sequence[float] = (0.0, 0.0)) -> list[int]:
    """确定性最近邻访问顺序（并列时取编号小者）：从 start 出发依次走遍所有圆心。"""
    rest = list(range(len(waypoints)))
    order: list[int] = []
    cur = np.asarray(start, dtype=float)
    while rest:
        k = min(rest, key=lambda j: (float(np.linalg.norm(waypoints[j] - cur)), j))
        order.append(k)
        rest.remove(k)
        cur = waypoints[k]
    return order


def path_length(waypoints: np.ndarray, order: Sequence[int],
                start: Sequence[float] = (0.0, 0.0)) -> float:
    """按给定顺序走遍各圆心的总里程 / m。"""
    total, cur = 0.0, np.asarray(start, dtype=float)
    for j in order:
        total += float(np.linalg.norm(waypoints[j] - cur))
        cur = waypoints[j]
    return total


@dataclass
class CoverSolveResult:
    """覆盖圆求解的完整结果：方案 + 实算校验 + 权衡分析 + 文献方法对照。"""

    plan: CoverPlan
    worst_distance: float               # 实算最坏最近距离 / m（网格 + 边界 + 解析候选点）
    worst_point: tuple[float, float]    # 实算最坏点位置
    analytic_worst: float               # 解析最坏最近距离 / m
    coverage_ratio: float               # 被覆盖的采样点比例（1.0 表示全覆盖）
    multiplicity: dict[int, int]        # 覆盖重数 → 采样点数
    single_ratio: float                 # 单重覆盖（只有 1 条射线可用）的占比
    multi_ratio: float                  # 二重及以上覆盖的占比
    mean_multiplicity: float            # 圆域内平均覆盖重数（重复率 = 平均值 - 1）
    survey_order: list[int]             # 巡视顺序（圆心编号）
    survey_length: float                # 巡视总里程 / m
    feasible_interval: tuple[float, float]   # 满足覆盖保证的环半径可行区间 / m
    tradeoff: list[dict[str, float]]    # 环半径权衡：d、最坏距离、余量、里程、时间
    lattice: dict[str, float]           # 参考文献紧贴六边形栅格（间距 √3·r）对照
    six_circle_worst: float             # 6 圆方案的实算最坏距离（不可行对照）/ m
    six_circle_radius: float            # 6 圆方案的最优环半径 / m（仅六边形环这一族）
    min_circles: dict[str, Any]         # 最少圆数的判定依据（经典 disk covering problem）
    use_uniform: bool = False           # True=正七边形均匀布局（半径 1000 m 圆上均匀分布）

    @property
    def margin(self) -> float:
        """最坏最近距离相对覆盖半径（1000 m）的余量 / m。"""
        return COVER_RADIUS - self.worst_distance

    def to_json(self) -> dict[str, Any]:
        """把方案与校验结果导出成 JSON 可序列化的字典（落盘 t3_cover_plan.json 用）

        Returns:
            dict[str, Any]: 圆心坐标、最坏距离、覆盖重数、可行区间、权衡表与文献对照等字段
        """
        plan = self.plan
        is_hex = plan.layout is None
        return {
            "region_radius_m": plan.region_radius,
            "cover_radius_m": plan.cover_radius,
            "layout_kind": ("hex 1+6（1 个中心圆 + 6 个正六边形环上圆）" if is_hex
                else ("uniform 7 点（config.SURVEY_CENTERS_UNIFORM：7 个圆心均匀分布在"
                      "半径 1000 m 的圆上）" if self.use_uniform
                      else f"general {len(plan.waypoints)} 点（config.SURVEY_CENTERS）")),
            "ring_radius_m": round(plan.ring_radius, 3) if is_hex else None,
            "rotation_deg": round(math.degrees(plan.rotation), 4),
            "ring_radius_chosen": (abs(plan.ring_radius - CHOSEN_RING_RADIUS) < 1e-9
                                   if is_hex else False),
            "ring_radius_max_margin_m": round(optimal_ring_radius(), 3),
            "n_circles": len(plan.waypoints),
            "analytic_worst_m": round(self.analytic_worst, 4),
            "computed_worst_m": round(self.worst_distance, 4),
            "margin_m": round(self.margin, 4),
            "worst_point": [round(v, 2) for v in self.worst_point],
            "coverage_ratio": self.coverage_ratio,
            "multiplicity_points": self.multiplicity,
            "single_ratio": self.single_ratio,
            "multi_ratio": self.multi_ratio,
            "mean_multiplicity": round(self.mean_multiplicity, 4),
            "survey_order": self.survey_order,
            "survey_length_m": round(self.survey_length, 2),
            "centers": [{"id": i,
                         "kind": ("中心圆" if i == 0 else "环上圆") if is_hex else f"巡视站{i + 1}",
                         "x": round(x, 3), "y": round(y, 3),
                         "rho_m": round(math.hypot(x, y), 3)}
                        for i, (x, y) in enumerate(plan.centers)],
            "feasible_ring_interval_m": [round(v, 3) for v in self.feasible_interval],
            "ring_radius_tradeoff": self.tradeoff,
            "reference_hex_lattice": self.lattice,
            "compare_six_circles": {"n_circles": 6, "best_ring_radius_m": self.six_circle_radius,
                                    "computed_worst_m": round(self.six_circle_worst, 4),
                                    "feasible": self.six_circle_worst <= COVER_RADIUS,
                                    "family": "仅「1 中心 + 6 环上圆」这一族；"
                                              "更一般布局的严格下界见 min_circle_count"},
            "min_circle_count": self.min_circles,
            "grid_step_m": GRID_STEP,
            "boundary_samples": BOUNDARY_SAMPLES,
        }


def _six_circle_best(step: float = COARSE_STEP,
                     n_boundary: int = COARSE_BOUNDARY) -> tuple[float, float]:
    """6 个覆盖圆的**族内**对照，不是全局最优：只允许「1 中心 + 6 环上圆」这一个受限族，
    故得到的只是"这一族里最好能到多少"，不能用来证明 6 个不行。

    真正的判定见 `min_circle_count()` —— 它引用经典 disk covering problem 的已证明最优值
    ρ_6 = 0.5559052 > 5/9，说明**任意** 6 个圆盘（不限这一族）都不够。本函数保留下来只为
    在报告里给一个直观的对照数字。

    返回（该族最优环半径 / m, 该半径下的最坏最近距离 / m）。粗网格仅用于可行性判断。
    """
    pts = region_samples(step, n_boundary)
    best = (float("inf"), 0.0)
    for d in np.arange(200.0, COVER_RADIUS + 1.0, 10.0):
        wp = hex_layout(float(d))[1:]          # 只有环上 6 个圆，没有中心圆
        m = float(nearest_distances(pts, wp).max())
        if m < best[0]:
            best = (m, float(d))
    return best[1], best[0]


def layout_metrics(ring_radius: float, pts: np.ndarray,
                   compute_multiplicity: bool = True) -> dict[str, Any]:
    """给定环半径，报出该布局的最坏最近距离、余量、覆盖重数与巡视里程。"""
    wp = hex_layout(float(ring_radius))
    near = nearest_distances(pts, wp)
    order = nearest_order(wp)
    length = path_length(wp, order)
    row: dict[str, Any] = {
        "ring_radius_m": round(float(ring_radius), 3),
        "computed_worst_m": round(float(near.max()), 3),
        "analytic_worst_m": round(analytic_worst(float(ring_radius)), 3),
        "margin_m": round(COVER_RADIUS - float(near.max()), 3),
        "survey_length_m": round(length, 1),
        "survey_time_s": round(length / 5.0, 1),
    }
    if compute_multiplicity:
        cnt = cover_counts(pts, wp, COVER_RADIUS)
        row["mean_multiplicity"] = round(float(cnt.mean()), 4)
        row["min_multiplicity"] = int(cnt.min())
    return row


def tradeoff_table(interval: tuple[float, float], pts: np.ndarray) -> list[dict[str, Any]]:
    """可行区间内取若干代表环半径，给出"余量 vs 里程"的权衡表。"""
    lo, hi = interval
    radii = [lo, 1200.0, 1300.0, 1400.0, optimal_ring_radius(), 1700.0, hi]
    radii = [r for r in radii if lo - 1e-9 <= r <= hi + 1e-9]
    return [layout_metrics(r, pts) for r in sorted(set(round(r, 3) for r in radii))]


def solve_covering_circles(ring_radius: float | None = None,
                          use_hex: bool = False,
                          use_uniform: bool = False) -> CoverSolveResult:
    """求解 1000 m 覆盖圆的位置，并做实算校验、可行区间与权衡分析、文献方法对照。

    三种一般布局：
      * 默认优化布局（config.SURVEY_CENTERS，里程 ~6167 m，余量 ~5 m）；
      * use_uniform=True 时用"7 个圆心均匀分布在半径 1000 m 的圆上"的正七边形布局
        （config.SURVEY_CENTERS_UNIFORM，里程 ~6207 m，**零余量** —— 最坏点恰在原点）；
      * 给出 ring_radius 或 use_hex=True 时改用"1 中心 + 6 正六边形环心"的经典族（里程 6d），
        用于对照与参数扫描。
    """
    if use_hex or ring_radius is not None:
        d = CHOSEN_RING_RADIUS if ring_radius is None else float(ring_radius)
        plan = CoverPlan(REGION_RADIUS, COVER_RADIUS, d)
        assert len(plan.waypoints) == 7, "覆盖圆个数应为 7（1 中心 + 6 环）"
        # 采样点 = 细网格 + 圆边界精细扫描 + 解析给出的最坏点候选（保证解析/实算一致）
        pts = np.vstack([region_samples(GRID_STEP, BOUNDARY_SAMPLES),
                         worst_candidates(d)])
    else:
        layout = tuple((float(x), float(y)) for x, y in
                       (SURVEY_CENTERS_UNIFORM if use_uniform else SURVEY_CENTERS))
        plan = CoverPlan(REGION_RADIUS, COVER_RADIUS, CHOSEN_RING_RADIUS, layout=layout)
        assert len(plan.waypoints) == 7, "覆盖圆个数应为 7"
        pts = np.vstack([region_samples(GRID_STEP, BOUNDARY_SAMPLES),
                         worst_candidates_general(plan.waypoints)])
    near = nearest_distances(pts, plan.waypoints)
    k = int(near.argmax())
    worst = float(near[k])
    worst_point = (float(pts[k][0]), float(pts[k][1]))

    cnt = cover_counts(pts, plan.waypoints, COVER_RADIUS)
    multiplicity = {int(a): int(b) for a, b in zip(*np.unique(cnt, return_counts=True))}
    single = multiplicity.get(1, 0) / len(pts)
    multi = sum(v for kk, v in multiplicity.items() if kk >= 2) / len(pts)

    # 巡视顺序：7 个点规模小，直接求精确最短开放路径（Held-Karp），比最近邻更短且确定
    order = exact_open_order(len(plan.waypoints),
                             dist_matrix(plan.waypoints, (0.0, 0.0)))
    length = path_length(plan.waypoints, order)
    if worst > COVER_RADIUS:
        raise AssertionError(f"覆盖保证被破坏：最坏距离 {worst:.1f} m > {COVER_RADIUS:.0f} m")
    if use_hex or ring_radius is not None:
        ana = analytic_worst(d)
        if abs(worst - ana) > 1e-6:
            raise AssertionError(f"解析最坏距离 {ana:.6f} m 与实算 {worst:.6f} m 不一致")
        if abs(length - 6.0 * d) > 1e-6:
            raise AssertionError(f"巡视里程 {length:.3f} m 与解析值 6d = {6.0 * d:.3f} m 不一致")
    else:
        ana = SURVEY_WORST_UNIFORM if use_uniform else SURVEY_WORST_M
        if abs(worst - ana) > 1e-3:
            raise AssertionError(f"一般布局最坏距离 {worst:.6f} m 与记录的 {ana:.6f} m 不一致")
        if abs(length - (SURVEY_ROUTE_UNIFORM if use_uniform else SURVEY_ROUTE_M)) > 1e-3:
            raise AssertionError(f"一般布局里程 {length:.3f} m 与记录的 {SURVEY_ROUTE_M:.3f} m 不一致")

    interval = feasible_ring_interval()
    # 参考文献（赵一骁 2024）的紧贴六边形栅格：相邻圆心间距 √3·r，即环半径 = √3·r
    lattice_d = math.sqrt(3.0) * COVER_RADIUS
    lattice = layout_metrics(lattice_d, pts)

    six_d, six_worst = _six_circle_best()
    return CoverSolveResult(
        plan=plan, worst_distance=worst, worst_point=worst_point, analytic_worst=ana,
        coverage_ratio=float((cnt >= 1).mean()), multiplicity=multiplicity,
        single_ratio=single, multi_ratio=multi, mean_multiplicity=float(cnt.mean()),
        survey_order=order, survey_length=length,
        feasible_interval=interval, tradeoff=tradeoff_table(interval, pts),
        lattice=lattice, six_circle_worst=six_worst, six_circle_radius=six_d,
        min_circles=min_circle_count(), use_uniform=use_uniform,
    )


def print_cover_report(res: CoverSolveResult) -> None:
    """打印覆盖圆求解报告（圆心位置、覆盖校验、最少个数、可行区间与权衡、文献对照）。"""
    plan = res.plan
    d = plan.ring_radius
    is_hex = plan.layout is None
    print("=" * 78)
    print("一、1000 m 覆盖圆的位置")
    print("=" * 78)
    print(f"目标圆域半径 R = {plan.region_radius:.0f} m，覆盖圆半径 r = {plan.cover_radius:.0f} m"
          f"（= 有效接收半径下界），共 {len(plan.waypoints)} 个覆盖圆")
    if is_hex:
        is_chosen = abs(d - CHOSEN_RING_RADIUS) < 1e-6
        print(f"布局：1 个中心圆（圆心在原点）+ 6 个环上圆（正六边形顶点）")
        print(f"环半径 d = {d:.2f} m（正六边形边长 = d）"
              + ("（选定设计半径：时间优先）" if is_chosen else "（由 --ring-radius 指定）"))
    else:
        print(f"布局：一般 7 点（非六边形；7 个点全部落在距原点约 1000 m 处，"
              f"原点本身落在它们的覆盖内，无需专门的中心点）")
    print()
    print(f"{'序号':<6}{'类型':<12}{'x / m':>12}{'y / m':>12}{'距原点 / m':>12}")
    for i, (x, y) in enumerate(plan.centers):
        kind = "中心圆" if (is_hex and i == 0) else ("环上圆" if is_hex else f"站 {i + 1}")
        rho = math.hypot(x, y)
        print(f"{i:<6}{kind:<12}{x:>12.2f}{y:>12.2f}{rho:>12.2f}")
    print()
    print("=" * 78)
    print(f"二、覆盖校验（细网格 {GRID_STEP:.0f} m + 圆边界 {BOUNDARY_SAMPLES} 点 + 解析最坏点候选）")
    print("=" * 78)
    if is_hex:
        print(f"解析最坏最近距离 D(d) = max(d/√3, g₂) = {res.analytic_worst:.3f} m"
              + ("（d = d* 时内圈项与边界项相等，即余量最大的最优性条件）"
                 if abs(d - optimal_ring_radius()) < 1e-6 else ""))
    else:
        print(f"参考最坏最近距离 = {res.analytic_worst:.3f} m（一般布局无 d 的闭式，"
              f"由 config.{('SURVEY_WORST_UNIFORM' if res.use_uniform else 'SURVEY_WORST_M')} "
              f"记录，来源见该常量注释）")
    print(f"实算最坏最近距离 = {res.worst_distance:.3f} m @ "
          f"({res.worst_point[0]:.1f}, {res.worst_point[1]:.1f})"
          + (f"；另一族最坏点在 ρ = d/√3 = {d / math.sqrt(3):.1f} m 的角平分线方向上"
             if is_hex else "（一般布局；候选集为三角形外心 + 中垂线∩边界 + 圆心对径点）"))
    print(f"对覆盖半径 1000 m 的余量 = {res.margin:.3f} m，"
          f"采样点被覆盖比例 = {res.coverage_ratio * 100:.2f}%")
    print(f"覆盖重数分布：" + "、".join(f"{k} 重 {v} 点" for k, v in sorted(res.multiplicity.items())))
    print(f"→ 平均覆盖重数 {res.mean_multiplicity:.3f}（重复率 "
          f"{(res.mean_multiplicity - 1) * 100:.1f}%）；单重覆盖（只有 1 条射线，无法直接交会）占 "
          f"{res.single_ratio * 100:.1f}%，二重及以上占 {res.multi_ratio * 100:.1f}%"
          f"（第二阶段需对单射线频道补测第二视角）")
    print()
    print("=" * 78)
    print("三、为什么是 7 个（最少数）")
    print("=" * 78)
    mc = res.min_circles
    print(f"归类为经典 disk covering problem：用 k 个半径 r 的圆盘覆盖半径 R 的圆，最少要几个？")
    print(f"归一化后本题只有 r/R = 1000/1800 = {mc['ratio_r_over_R']:.10f} 一个参数。k 个圆盘所需")
    print(f"的最小半径比 ρ_k 已有证明：ρ_5 = {DISK_RATIO_5:.10f}（Bezdek 1983）、")
    print(f"                          ρ_6 = {DISK_RATIO_6:.10f}（Bezdek 1979）、ρ_7 = 0.5（构造）")
    print(f"ρ_7 = 0.5 < {mc['ratio_r_over_R']:.10f} < ρ_6 ⇒ 7 个够、6 个不够 ⇒ 最少 7 个 ✓")
    print(f"  6 个差多少：即便六圆摆到最优，也需覆盖半径 ≥ {mc['six_required_cover_radius_m']:.3f} m，")
    print(f"             比可用的 1000 m 多 {mc['six_shortfall_m']:.3f} m"
          f"（{mc['six_shortfall_ratio'] * 100:.3f}%）——只差千分之零点六，但仍是不够")
    print(f"             等价地：6 个半径 1000 m 的圆盘最多覆盖半径 "
          f"{mc['six_max_region_radius_m']:.3f} m 的圆域，而作业圆域是 1800 m")
    print(f"  弱下界（不足以排除 6）：面积/密度 → n ≥ {mc['area_density_lower']}；"
          f"边界弧 → n ≥ {mc['boundary_arc_lower']}；故必须用上面已证明的最优值")
    print(f"  族内对照（仅「1 中心 + 6 环上圆」这一族，非全局最优）：最优环半径 "
          f"{res.six_circle_radius:.0f} m 时最坏 {res.six_circle_worst:.1f} m，离 1000 m 更远")
    print(f"  7 个的余量：本方案最坏距离 {res.worst_distance:.1f} m ≤ 1000 m ✓"
          f"（余量 {res.margin:.1f} m）")
    print()
    print("=" * 78)
    print("四、可行区间与权衡（余量 vs 巡视里程）")
    print("=" * 78)
    lo, hi = res.feasible_interval
    if not is_hex:
        print("（下表是「1 中心 + 6 环上圆」族的权衡，用于说明为什么不用六边形、"
              "以及环半径取多小会破坏保证；默认的一般 7 点布局不在这一族内。）")
    print(f"覆盖保证等价于 D(d) ≤ 1000 m，可行区间 d ∈ [{lo:.2f}, {hi:.2f}] m"
          f"（两端点余量为 0）")
    print(f"六边形族内余量最大的是 d* = {optimal_ring_radius():.2f} m（余量 100.00 m，"
          f"里程 {6 * optimal_ring_radius():.1f} m）——取更小的 d 即「用余量换时间」")
    print(f"六边形族巡视里程 = 6d（原点→环心 d，再走 5 条六边形边），故 d 越小越省时间；")
    print(f"但它的下界是 6·d_min = {6 * lo:.1f} m（d_min = {lo:.2f} m 时余量为 0），"
          f"仍比默认的一般布局长 {6 * lo - res.survey_length:.1f} m")
    print()
    print(f"{'环半径 d / m':>13}{'最坏距离 / m':>13}{'余量 / m':>11}"
          f"{'平均重数':>10}{'里程 / m':>11}{'移动时间 / s':>13}")
    for row in res.tradeoff:
        mark = ""
        if abs(row["ring_radius_m"] - d) < 1e-6:
            mark = "  ← 六边形族内时间优先（已被默认布局取代）"
        elif abs(row["ring_radius_m"] - optimal_ring_radius()) < 1e-6:
            mark = "  ← d*（余量最大）"
        print(f"{row['ring_radius_m']:>13.2f}{row['computed_worst_m']:>13.2f}"
              f"{row['margin_m']:>11.2f}{row.get('mean_multiplicity', float('nan')):>10.3f}"
              f"{row['survey_length_m']:>11.1f}{row['survey_time_s']:>13.1f}{mark}")
    print()
    lat = res.lattice
    print(f"参考文献紧贴栅格对照（相邻圆心间距 √3·r，即环半径 {lat['ring_radius_m']:.2f} m）："
          f"最坏距离 {lat['computed_worst_m']:.2f} m（余量 {lat['margin_m']:.2f} m）、"
          f"里程 {lat['survey_length_m']:.1f} m")
    print(f"默认布局：最坏距离 {res.worst_distance:.2f} m（余量 {res.margin:.2f} m）、"
          f"里程 {res.survey_length:.1f} m → 覆盖余量优于紧贴栅格、里程省 "
          f"{lat['survey_length_m'] - res.survey_length:.1f} m")
    print(f"六边形族最优（d = d_min = {lo:.2f} m，余量 0.00 m）：里程 {6 * lo:.1f} m，"
          f"仍比默认布局长 {6 * lo - res.survey_length:.1f} m"
          f"（六边形把 1 个点压在原点，而原点是覆盖效率最低的位置）")
    print()
    print("=" * 78)
    print("五、巡视顺序（从原点出发的精确最短开放路径，Held-Karp 枚举）")
    print("=" * 78)
    seq = " → ".join(str(i) for i in res.survey_order)
    print(f"顺序：{seq}（编号见第一节；{'圆心 0 在原点' if is_hex else '所有站都在距原点约 1 km 处'}）")
    print("说明：以上是设计基准（站点方位固定）。把整个布局绕原点旋转一个角时，覆盖条件只依赖"
          "点间距离与\n      点到原点的距离、巡视路径长度只依赖点间距离，故最坏最近距离与里程都"
          "不变（见 python -m t3.covering 的实算自检）。实际作业在起始全频道扫描\n      "
          "之后把布局转到最密集的 60° 扇区方向，使巡视终点落在源密集方向；逐局旋转角记录在 "
          "t3_survey.json。")
    print(f"总里程 = {res.survey_length:.1f} m，纯移动时间 = {res.survey_length / 5.0:.1f} s"
          f"（速度 5 m/s）")
    print("=" * 78)


# ----------------------------------------------------------------------------
# 自检：`python -m t3.covering`
# ----------------------------------------------------------------------------
def _selftest() -> int:
    """三项自检：覆盖保证在旋转下成立（六边形族 + 一般布局）；密集扇区选向；最少圆数为 7。

    旋转是"整个 7 圆布局绕原点转一个角"，而覆盖条件只取决于圆心之间的距离与它们到原点的
    距离，两者在共同旋转下都不变，所以保证**理论上**恒定。但"理论上不变"和"实现上确实不变"
    是两回事（例如只转了采样点没转圆心就会静默出错），故这里实算校验。
    """
    import numpy as np
    rng = np.random.default_rng(2026)
    rots = np.concatenate([np.arange(0.0, 60.0, 1.0), rng.uniform(0.0, 360.0, 40)])
    ok1 = True

    # (a) 经典六边形族：解析值 D(d) 应被 100 个旋转角一致复现
    d = CHOSEN_RING_RADIUS
    pts = np.vstack([region_samples(GRID_STEP, BOUNDARY_SAMPLES), worst_candidates(d)])
    ana = analytic_worst(d)
    worst_seen, worst_at = 0.0, 0.0
    for rot in rots:
        plan = CoverPlan(REGION_RADIUS, COVER_RADIUS, d, float(math.radians(rot)))
        w = float(nearest_distances(pts, plan.waypoints).max())
        if w > worst_seen:
            worst_seen, worst_at = w, rot
    ok1a = abs(worst_seen - ana) < 1e-6 and worst_seen <= COVER_RADIUS + 1e-9
    ok1 &= ok1a
    print(f"[1] 六边形族在 {len(rots)} 个旋转角下：最坏最近距离最大 {worst_seen:.6f} m"
          f"（出现在 {worst_at:.1f}°），解析值 {ana:.6f} m → "
          f"{'✓ 与解析一致且未破坏保证' if ok1a else '✗ 不一致'}")

    # (b) 默认的一般 7 点布局：旋转不变 + 覆盖成立 + 里程等于记录值
    layout = tuple((float(x), float(y)) for x, y in SURVEY_CENTERS)
    base = CoverPlan(REGION_RADIUS, COVER_RADIUS, CHOSEN_RING_RADIUS, layout=layout)
    pts2 = np.vstack([region_samples(GRID_STEP, BOUNDARY_SAMPLES),
                      worst_candidates_general(base.waypoints)])
    gworst, gat = 0.0, 0.0
    for rot in rots:
        w = float(nearest_distances(pts2, base.rotated(float(rot)).waypoints).max())
        if w > gworst:
            gworst, gat = w, rot
    order = exact_open_order(7, dist_matrix(base.waypoints, (0.0, 0.0)))
    glen = path_length(base.waypoints, order)
    ok1b = (abs(gworst - SURVEY_WORST_M) < 1e-3 and gworst <= COVER_RADIUS + 1e-9
            and abs(glen - SURVEY_ROUTE_M) < 1e-3)
    ok1 &= ok1b
    print(f"[1] 一般 7 点布局（默认）在 {len(rots)} 个旋转角下：最坏 {gworst:.6f} m"
          f"（记录 {SURVEY_WORST_M:.4f}，余量 {COVER_RADIUS - gworst:.4f} m），"
          f"精确里程 {glen:.4f} m（记录 {SURVEY_ROUTE_M:.4f}）→ "
          f"{'✓ 覆盖成立且旋转不变' if ok1b else '✗ 不一致'}")

    cases = [([95.0, 100.0, 105.0, 200.0, 300.0], 100.0),
             ([130.0, 131.0, 10.0], 130.5),
             ([59.0, 59.0, 59.0], 59.0),
             ([350.0, 350.0, 170.0], 350.0)]
    ok2 = True
    for bearings, want in cases:
        got = math.degrees(dense_sector_rotation(bearings))
        good = min(abs(got - want), abs(got - want - 360.0), abs(got - want + 360.0)) < 1.5
        ok2 &= good
        print(f"[2] 密集扇区选向 {str(bearings)[:34]:36s} → {got:7.2f}°（期望 ≈{want}°）"
              f"{'✓' if good else '✗'}")
    ok2 &= dense_sector_rotation([]) is None
    print(f"    空输入（未听到任何源）→ {dense_sector_rotation([])}（应为 None）"
          f"{'✓' if dense_sector_rotation([]) is None else '✗'}")
    # [3] 最少圆数：7（6 个被已证明的 ρ_6 排除）
    mc = min_circle_count()
    ok3 = (mc["min_count"] == 7
           and mc["six_required_cover_radius_m"] > COVER_RADIUS
           and DISK_RATIO_7 * REGION_RADIUS <= COVER_RADIUS
           and mc["boundary_arc_lower"] <= 6)
    print(f"[3] 最少圆数：ρ_6·R = {mc['six_required_cover_radius_m']:.3f} m > 1000 m ⇒ 6 个不够；"
          f"ρ_7·R = {mc['seven_required_cover_radius_m']:.0f} m ≤ 1000 m ⇒ 7 个够")
    print(f"    → 最少 {mc['min_count']} 个（6 个只差 {mc['six_shortfall_ratio'] * 100:.3f}%）；"
          f"弱下界 面积≥{mc['area_density_lower']}、弧长≥{mc['boundary_arc_lower']}"
          f" → {'✓' if ok3 else '✗'}")

    # [4] 均匀 7 点布局（用户指定排布）：圆心都在半径 1000 m 的圆上且均匀分布，覆盖必须成立
    U = np.asarray([(float(x), float(y)) for x, y in SURVEY_CENTERS_UNIFORM], dtype=float)
    ok4 = True
    rho = np.linalg.norm(U, axis=1)
    ok4 &= np.allclose(rho, 1000.0, atol=1e-6)                    # 都在 r=1000 圆上
    ang = np.sort(np.mod(np.arctan2(U[:, 1], U[:, 0]), 2.0 * math.pi))
    ok4 &= np.allclose(np.diff(np.concatenate([ang, [ang[0] + 2.0 * math.pi]])),
                       2.0 * math.pi / 7, atol=1e-6)              # 均匀分布（正七边形）
    ptsU = np.vstack([region_samples(GRID_STEP, BOUNDARY_SAMPLES),
                      worst_candidates_general(U)])
    wU = float(nearest_distances(ptsU, U).max())
    ok4 &= wU <= COVER_RADIUS + 1e-9                              # 覆盖成立（等号，零余量）
    for th in (0.0, 0.3, 1.1):                                    # 旋转不变
        c, s = math.cos(th), math.sin(th)
        Ur = U @ np.array([[c, s], [-s, c]])
        ptsUr = np.vstack([region_samples(GRID_STEP, BOUNDARY_SAMPLES),
                           worst_candidates_general(Ur)])
        w = float(nearest_distances(ptsUr, Ur).max())
        ok4 &= abs(w - wU) <= 1e-6
    orderU = exact_open_order(7, dist_matrix(U, (0.0, 0.0)))
    LU = path_length(U, orderU)
    ok4 &= abs(LU - SURVEY_ROUTE_UNIFORM) <= 1e-3                 # 里程与记录一致
    print(f"[4] 均匀 7 点布局：圆心都在 r=1000 圆上且均匀（正七边形）"
          f" {'✓' if np.allclose(rho, 1000.0, atol=1e-6) else '✗'}；"
          f"最坏最近距离 {wU:.4f} m（零余量，最坏点恰在原点）"
          f" {'✓' if wU <= COVER_RADIUS + 1e-9 else '✗'}；"
          f"任意旋转不变 {'✓' if ok4 else '✗'}；里程 {LU:.1f} m"
          f"（记录 {SURVEY_ROUTE_UNIFORM}）{'✓' if abs(LU - SURVEY_ROUTE_UNIFORM) <= 1e-3 else '✗'}")

    ok = ok1 and ok2 and ok3 and ok4
    print("\n自检结果：" + ("全部通过 ✓" if ok else "存在失败 ✗"))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(_selftest())
