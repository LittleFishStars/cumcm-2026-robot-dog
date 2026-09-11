"""覆盖圆的求解与校验（阶段一的几何底座）。

问题：用尽量少的半径 1000 m 的圆盖住半径 1800 m 的作业圆域，使**圆域内任意点**到最近圆心
的距离 ≤ 1000 m（= 有效接收半径下界），从而走到任一圆心都能听到全域的源。

方案是"1 个中心圆 + 6 个环圆"的正六边形拼接（7 个圆心，文献中的最小圆覆盖 + 正六边形拼接）。
本模块负责：解析地给出最坏最近距离 D(d) = max(d/√3, g₂(d))、可行环半径区间、最优环半径 d*，
以及**在连续圆域上**求最坏点的数值校验（网格上"看起来满足"不等于满足 —— 曾因此漏掉圆域
边缘的源）。所有结果显示在 print_cover_report 里。

    from cumcm.t3.covering import solve_covering_circles
    res = solve_covering_circles()          # 缺省用 config.CHOSEN_RING_RADIUS

注：`nearest_order` / `path_length` 复用 common.routing 中的同名实现（后者按公共模块命名
为 open_path_length，这里保留别名以贴合"走遍圆心的总里程"这一语义）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cumcm.common.routing import nearest_order
from cumcm.common.routing import open_path_length as path_length
from cumcm.t3.config import (BOUNDARY_SAMPLES, CHOSEN_RING_RADIUS, COARSE_BOUNDARY,
                             COARSE_STEP, COVER_RADIUS, GRID_STEP,
                             REGION_RADIUS, TOL)


@dataclass(frozen=True, eq=False)
class CoverPlan:
    """覆盖圆方案：所有覆盖圆的圆心的位置。

    圆心编号：0 号为圆心在原点的那一个（中心圆），1~6 号为正六边形环上的圆心（逆时针，
    从 0° 方位角起）。这些圆心就是机器狗的巡视路点。
    """

    region_radius: float            # 目标圆域半径 / m
    cover_radius: float             # 覆盖圆半径（= 有效接收半径下界）/ m
    ring_radius: float              # 六边形环上圆心到原点的距离 d / m
    rotation: float = 0.0           # 环的旋转角 / rad（见 hex_layout；0 = 环心在 0°/60°/…）
    waypoints: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "waypoints",
                           hex_layout(self.ring_radius, self.rotation))

    def rotated(self, rotation: float) -> "CoverPlan":
        """返回把环旋转到给定角度的新方案（中心圆恒在原点，故不动）。"""
        return replace(self, rotation=float(rotation))

    @property
    def centers(self) -> List[Tuple[float, float]]:
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
                           cover_radius: float = COVER_RADIUS) -> Tuple[float, float]:
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
        return math.sqrt(max(rho * rho + ring_radius * ring_radius
                             - math.sqrt(3.0) * rho * ring_radius, 0.0))
    return max(ring_radius / math.sqrt(3.0), g(region_radius))


SECTOR_HALF_DEG = 30.0          # 扇区半宽 / 度（正六边形相邻环心隔 60°，各管一半）
def _sector_score(bearings_rad: np.ndarray, phi: float) -> Tuple[int, float]:
    """以 phi 为扇区中心的评分：(装入的源个数, 这些源的 ΣcosΔθ)。

    个数是主指标（"哪一片源最多"），ΣcosΔθ 只在个数相同时破平（片内的源越居中越好）。
    """
    d = (bearings_rad - phi + math.pi) % (2.0 * math.pi) - math.pi
    inside = np.abs(d) <= math.radians(SECTOR_HALF_DEG)
    cnt = int(inside.sum())
    return cnt, (float(np.cos(d[inside]).sum()) if cnt else 0.0)


def dense_sector_rotation(bearings: Sequence[float],
                          step_deg: float = 1.0) -> Optional[float]:
    """选一个旋转角，使某个环心正对"听到的源最多"的 60° 扇区。

    6 个环心彼此相隔 60°，故"环心能对准哪些方向"只由旋转角模 60° 决定。于是把圆周切成 360/60
    个候选 60° 扇区（1° 一格，等效地把整个圆周扫一遍），取装入源最多的那一片的中心方向 θ*，
    返回 θ* 本身（落在 [0, 2π)，不做模 60° 归约）作为旋转角 —— 这样 1 号环心正好落在 θ* 上，日志与结果图可以直接读成"1 号环心
    的方位角"。模 60° 归约在几何上等价，但不利于阅读，故不做。

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
    pts = pts[np.linalg.norm(pts, axis=1) <= REGION_RADIUS + TOL]
    t = np.linspace(0.0, 2.0 * math.pi, n_boundary, endpoint=False)
    ring = np.stack((REGION_RADIUS * np.cos(t), REGION_RADIUS * np.sin(t)), axis=1)
    return np.vstack([pts, ring])


def nearest_distances(points: np.ndarray, waypoints: np.ndarray,
                      chunk: int = 200_000) -> np.ndarray:
    """每个采样点到最近圆心的距离（分块计算，避免大矩阵占内存）。"""
    out = np.empty(len(points), dtype=float)
    for i in range(0, len(points), chunk):
        blk = points[i:i + chunk]
        d = np.linalg.norm(blk[:, None, :] - waypoints[None, :, :], axis=2)
        out[i:i + chunk] = d.min(axis=1)
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


def cover_counts(points: np.ndarray, waypoints: np.ndarray, radius: float,
                 chunk: int = 200_000) -> np.ndarray:
    """每个采样点被几个覆盖圆同时覆盖（覆盖重数）。"""
    out = np.empty(len(points), dtype=np.int64)
    for i in range(0, len(points), chunk):
        blk = points[i:i + chunk]
        d = np.linalg.norm(blk[:, None, :] - waypoints[None, :, :], axis=2)
        out[i:i + chunk] = (d <= radius + TOL).sum(axis=1)
    return out


def nearest_order(waypoints: np.ndarray, start: Sequence[float] = (0.0, 0.0)) -> List[int]:
    """确定性最近邻访问顺序（并列时取编号小者）：从 start 出发依次走遍所有圆心。"""
    rest = list(range(len(waypoints)))
    order: List[int] = []
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
    worst_point: Tuple[float, float]    # 实算最坏点位置
    analytic_worst: float               # 解析最坏最近距离 / m
    coverage_ratio: float               # 被覆盖的采样点比例（1.0 表示全覆盖）
    multiplicity: Dict[int, int]        # 覆盖重数 → 采样点数
    single_ratio: float                 # 单重覆盖（只有 1 条射线可用）的占比
    multi_ratio: float                  # 二重及以上覆盖的占比
    mean_multiplicity: float            # 圆域内平均覆盖重数（重复率 = 平均值 - 1）
    survey_order: List[int]             # 巡视顺序（圆心编号）
    survey_length: float                # 巡视总里程 / m
    feasible_interval: Tuple[float, float]   # 满足覆盖保证的环半径可行区间 / m
    tradeoff: List[Dict[str, float]]    # 环半径权衡：d、最坏距离、余量、里程、时间
    lattice: Dict[str, float]           # 参考文献紧贴六边形栅格（间距 √3·r）对照
    six_circle_worst: float             # 6 圆方案的实算最坏距离（不可行对照）/ m
    six_circle_radius: float            # 6 圆方案的最优环半径 / m

    @property
    def margin(self) -> float:
        """最坏最近距离相对覆盖半径（1000 m）的余量 / m。"""
        return COVER_RADIUS - self.worst_distance

    def to_json(self) -> Dict[str, Any]:
        plan = self.plan
        return {
            "region_radius_m": plan.region_radius,
            "cover_radius_m": plan.cover_radius,
            "ring_radius_m": round(plan.ring_radius, 3),
            "rotation_deg": round(math.degrees(plan.rotation), 4),
            "ring_radius_chosen": abs(plan.ring_radius - CHOSEN_RING_RADIUS) < 1e-9,
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
            "centers": [{"id": i, "kind": "中心圆" if i == 0 else "环上圆",
                         "x": round(x, 3), "y": round(y, 3)}
                        for i, (x, y) in enumerate(plan.centers)],
            "feasible_ring_interval_m": [round(v, 3) for v in self.feasible_interval],
            "ring_radius_tradeoff": self.tradeoff,
            "reference_hex_lattice": self.lattice,
            "compare_six_circles": {"n_circles": 6, "best_ring_radius_m": self.six_circle_radius,
                                    "computed_worst_m": round(self.six_circle_worst, 4),
                                    "feasible": self.six_circle_worst <= COVER_RADIUS},
            "grid_step_m": GRID_STEP,
            "boundary_samples": BOUNDARY_SAMPLES,
        }


def _six_circle_best(step: float = COARSE_STEP,
                     n_boundary: int = COARSE_BOUNDARY) -> Tuple[float, float]:
    """6 个覆盖圆的可行性对照：环半径必须 ≤ 1000 m 才能盖住原点，扫描取最优。

    返回（最优环半径 / m, 该半径下的最坏最近距离 / m）。粗网格仅用于可行性判断。
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
                   compute_multiplicity: bool = True) -> Dict[str, Any]:
    """给定环半径，报出该布局的最坏最近距离、余量、覆盖重数与巡视里程。"""
    wp = hex_layout(float(ring_radius))
    near = nearest_distances(pts, wp)
    order = nearest_order(wp)
    length = path_length(wp, order)
    row: Dict[str, Any] = {
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


def tradeoff_table(interval: Tuple[float, float], pts: np.ndarray) -> List[Dict[str, Any]]:
    """可行区间内取若干代表环半径，给出"余量 vs 里程"的权衡表。"""
    lo, hi = interval
    radii = [lo, 1200.0, 1300.0, 1400.0, optimal_ring_radius(), 1700.0, hi]
    radii = [r for r in radii if lo - 1e-9 <= r <= hi + 1e-9]
    return [layout_metrics(r, pts) for r in sorted(set(round(r, 3) for r in radii))]


def solve_covering_circles(ring_radius: Optional[float] = None) -> CoverSolveResult:
    """求解 1000 m 覆盖圆的位置，并做实算校验、可行区间与权衡分析、文献方法对照。"""
    d = CHOSEN_RING_RADIUS if ring_radius is None else float(ring_radius)
    plan = CoverPlan(REGION_RADIUS, COVER_RADIUS, d)
    assert len(plan.waypoints) == 7, "覆盖圆个数应为 7（1 中心 + 6 环）"

    # 采样点 = 细网格 + 圆边界精细扫描 + 解析给出的最坏点候选（保证解析/实算一致）
    pts = np.vstack([region_samples(GRID_STEP, BOUNDARY_SAMPLES),
                     worst_candidates(d)])
    near = nearest_distances(pts, plan.waypoints)
    k = int(near.argmax())
    worst = float(near[k])
    worst_point = (float(pts[k][0]), float(pts[k][1]))

    cnt = cover_counts(pts, plan.waypoints, COVER_RADIUS)
    multiplicity = {int(a): int(b) for a, b in zip(*np.unique(cnt, return_counts=True))}
    single = multiplicity.get(1, 0) / len(pts)
    multi = sum(v for kk, v in multiplicity.items() if kk >= 2) / len(pts)

    order = nearest_order(plan.waypoints)
    length = path_length(plan.waypoints, order)
    ana = analytic_worst(d)
    if abs(worst - ana) > 1e-6:
        raise AssertionError(f"解析最坏距离 {ana:.6f} m 与实算 {worst:.6f} m 不一致")
    if worst > COVER_RADIUS:
        raise AssertionError(f"覆盖保证被破坏：最坏距离 {worst:.1f} m > {COVER_RADIUS:.0f} m")
    if abs(length - 6.0 * d) > 1e-6:
        raise AssertionError(f"巡视里程 {length:.3f} m 与解析值 6d = {6.0 * d:.3f} m 不一致")

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
    )


def print_cover_report(res: CoverSolveResult) -> None:
    """打印覆盖圆求解报告（圆心位置、覆盖校验、最少个数、可行区间与权衡、文献对照）。"""
    plan = res.plan
    d = plan.ring_radius
    is_chosen = abs(d - CHOSEN_RING_RADIUS) < 1e-6
    print("=" * 78)
    print("一、1000 m 覆盖圆的位置")
    print("=" * 78)
    print(f"目标圆域半径 R = {plan.region_radius:.0f} m，覆盖圆半径 r = {plan.cover_radius:.0f} m"
          f"（= 有效接收半径下界），共 {len(plan.waypoints)} 个覆盖圆")
    print(f"布局：1 个中心圆（圆心在原点）+ 6 个环上圆（正六边形顶点）")
    print(f"环半径 d = {d:.2f} m（正六边形边长 = d）"
          + ("（选定设计半径：时间优先）" if is_chosen else "（由 --ring-radius 指定）"))
    print()
    print(f"{'序号':<6}{'类型':<10}{'x / m':>12}{'y / m':>12}")
    for i, (x, y) in enumerate(plan.centers):
        print(f"{i:<6}{'中心圆' if i == 0 else '环上圆':<10}{x:>12.2f}{y:>12.2f}")
    print()
    print("=" * 78)
    print(f"二、覆盖校验（细网格 {GRID_STEP:.0f} m + 圆边界 {BOUNDARY_SAMPLES} 点 + 解析最坏点候选）")
    print("=" * 78)
    print(f"解析最坏最近距离 D(d) = max(d/√3, g₂) = {res.analytic_worst:.3f} m"
          + ("（d = d* 时内圈项与边界项相等，即余量最大的最优性条件）"
             if abs(d - optimal_ring_radius()) < 1e-6 else ""))
    print(f"实算最坏最近距离 = {res.worst_distance:.3f} m @ "
          f"({res.worst_point[0]:.1f}, {res.worst_point[1]:.1f})"
          f"；另一族最坏点在 ρ = d/√3 = {d / math.sqrt(3):.1f} m 的角平分线方向上")
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
    print(f"6 个覆盖圆：环心须落在距原点 1000 m 内才能盖住原点；最优环半径 "
          f"{res.six_circle_radius:.0f} m 时实算最坏距离 {res.six_circle_worst:.1f} m "
          f"> 1000 m ✗（圆域边缘漏源）")
    print(f"7 个覆盖圆：最坏距离 {res.worst_distance:.1f} m ≤ 1000 m ✓（余量 "
          f"{res.margin:.1f} m）→ 7 个即最少可行个数")
    print()
    print("=" * 78)
    print("四、可行区间与权衡（余量 vs 巡视里程）")
    print("=" * 78)
    lo, hi = res.feasible_interval
    print(f"覆盖保证等价于 D(d) ≤ 1000 m，可行区间 d ∈ [{lo:.2f}, {hi:.2f}] m"
          f"（两端点余量为 0）")
    print(f"现用 d = {d:.2f} m（余量 {res.margin:.2f} m）；余量最大的是 d* = "
          f"{optimal_ring_radius():.2f} m（余量 100.00 m，里程 9353 m）"
          f"——取更小的 d 即「用余量换时间」，两者都可证明不漏源")
    print(f"巡视里程 = 6d（原点→环心 d，再走 5 条六边形边），故 d 越小越省时间")
    print()
    print(f"{'环半径 d / m':>13}{'最坏距离 / m':>13}{'余量 / m':>11}"
          f"{'平均重数':>10}{'里程 / m':>11}{'移动时间 / s':>13}")
    for row in res.tradeoff:
        mark = ""
        if abs(row["ring_radius_m"] - d) < 1e-6:
            mark = "  ← 现用（时间优先）"
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
    print(f"现用 d = {d:.2f} m：最坏距离 {res.worst_distance:.2f} m（余量 {res.margin:.2f} m）、"
          f"里程 {res.survey_length:.1f} m → 覆盖余量优于紧贴栅格、里程省 "
          f"{lat['survey_length_m'] - res.survey_length:.1f} m")
    print()
    print("=" * 78)
    print("五、巡视顺序（确定性最近邻：从原点出发，圆心 0 就在原点）")
    print("=" * 78)
    seq = " → ".join(str(i) for i in res.survey_order)
    print(f"顺序：{seq}")
    print("说明：以上是设计基准（环心在 0°/60°/…）。7 个圆心绕原点整体旋转时，覆盖条件只依赖"
          "圆心间距与圆心\n      到原点的距离，二者均不变，故最坏最近距离与巡视里程 6d 都不变"
          "（见 python -m cumcm.t3.covering 的实算自检）。\n      实际作业在起始全频道扫描之后"
          "把环转到“1 号环心正对源最密集的 60° 扇区”，使巡视的往返向径穿过源密集区、\n      "
          "巡视终点落在该方向（于是清除阶段从源最集中处开始）；逐局旋转角记录在 t3_survey.json。")
    print(f"总里程 = {res.survey_length:.1f} m，纯移动时间 = {res.survey_length / 5.0:.1f} s"
          f"（速度 5 m/s）")
    print("=" * 78)


# ----------------------------------------------------------------------------
# 自检：`python -m cumcm.t3.covering`
# ----------------------------------------------------------------------------
def _selftest() -> int:
    """两项自检：旋转不破坏覆盖保证；密集扇区选向符合预期。

    旋转是"整个 7 圆布局绕原点转一个角"，而覆盖条件只取决于圆心之间的距离与它们到原点的
    距离，两者在共同旋转下都不变，所以保证**理论上**恒定。但"理论上不变"和"实现上确实不变"
    是两回事（例如只转了采样点没转圆心就会静默出错），故这里实算校验。
    """
    import numpy as np
    d = CHOSEN_RING_RADIUS
    pts = np.vstack([region_samples(GRID_STEP, BOUNDARY_SAMPLES), worst_candidates(d)])
    ana = analytic_worst(d)
    worst_seen, worst_at = 0.0, 0.0
    rng = np.random.default_rng(2026)
    for rot in np.concatenate([np.arange(0.0, 60.0, 1.0), rng.uniform(0.0, 360.0, 40)]):
        plan = CoverPlan(REGION_RADIUS, COVER_RADIUS, d, float(math.radians(rot)))
        w = float(nearest_distances(pts, plan.waypoints).max())
        if w > worst_seen:
            worst_seen, worst_at = w, rot
    ok1 = worst_seen <= COVER_RADIUS + 1e-9 and abs(worst_seen - ana) < 1e-6
    print(f"[1] 旋转下的覆盖保证：{len(np.arange(0.0, 60.0, 1.0)) + 40} 个角度实算，"
          f"最坏最近距离最大 {worst_seen:.6f} m（出现在 {worst_at:.1f}°），"
          f"解析值 {ana:.6f} m，覆盖半径 {COVER_RADIUS:.0f} m → "
          f"{'✓ 与解析一致且未破坏保证' if ok1 else '✗ 不一致'}")

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
    ok = ok1 and ok2
    print("\n自检结果：" + ("全部通过 ✓" if ok else "存在失败 ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
