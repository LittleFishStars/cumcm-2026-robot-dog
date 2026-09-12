"""扫描布局：原点 + 7 覆盖基点 + 12 个均匀方位外圈点（问题四检测阶段）。

定向源在检测点 p 处可被听到 ⟺

    |p − g| ≤ R  且  (p − g) · u(θ) ≥ 0        （R 为接收半径 1000~1500，θ 为定向方向，含边界）

即检测区是"接收圆 ∩ 前向半平面"= 半径 R 的**半圆盘**，圆心 g、半径 R、朝向 θ 全部未知。
问题三的 7 点覆盖只保证"圆域内任意点到最近巡视点 ≤ 1000 m"（对全向源必听到）；对定向源，
**背对着所有巡视点的源即使近在咫尺也听不到**（引擎返回 no_signal，见 engine.py 的
_in_directional_coverage —— 这就是"搜索不到还有可能是方向不对"）。

本模块的布局（最终定案：20 个测量位置）：

  1. **原点**：起点全频道扫描（位置成本为零）；
  2. **7 个覆盖基点** = 问题三巡视站（t3.config.SURVEY_CENTERS，直接复用，全向覆盖保证）；
  3. **12 个均匀方位外圈点**，半径 OUTER_RING_RAD = 1850 m（MID_RING_N = 0，不加内部补点），
     起始方位自 0° 旋转 OUTER_RING_ROT_DEG = 7.5°（2026-09-12 定案：与内圈基点方位错开，
     方位互补；两组 seed A/B 复核 0°→7.5°：seed 2026-2030 时间持平、误差 8.61→7.15 m，
     seed 2031-2035 时间 6858→6411 s（−447 s）、里程 −1880 m）。

为什么外圈取 1850 m、12 个方位（这是对"7+21 中点外推"布局的结构优化）：

  * **半径 1850 m**：任意源的迎光区在源前方 [|g|, |g|+R] 纵深；源最远半径 1770，所以**任意
    贴边源**的迎光区内沿 = 1770 m，外圈设在其内侧 1850 m 时径向偏差 < 80 m。原"中点外推"
    布局把多数点推到 2270 m，反而离内部源的迎光面太远（实测只剩 99.81%）；1850 m 单圈 +
    12 个方位同时覆盖贴边源与内部半径源的迎光区，实测 **99.9891%**（4 224 640 个算例、漏
    460，其中贴边对抗 4.3 万例 0 漏）；
  * **12 个方位（间隔 30°）**：相邻外圈点最大距离 ≈ 1850·sin15° ≈ 479 m < 接收半径一半，
    任何方向的源都能被方向差 ≤ 15° 的外圈点听到。**为什么不是更少的点**：试过 11 点
    （间隔 32.7°、布局路线 16 725 m），但方位间隔变宽使顺路清除的方位扇区也变宽、绕路更多，
    实战总里程反升 464 m——12 点是"减少路程"的实战最优；
  * **里程**：20 个测量位置的最短开放路径 16 995.65 m（外圈起始角 7.5° 与内圈错开）—— 比
    "7+21 中点外推"（29 点、21 580 m）省 21%，还少 9 个测量位置。
  * **为什么不加内部补点（20 点定案）**：曾试过加 3 个内部补点（r = 850 m，23 个位置）把听率
    提到 99.9974%，但 20 局演练实测多 ~34 次测向、慢 80 s；20 点版听率 99.9891%（422.5 万
    算例、漏 460；贴边对抗 0 漏）对每局 16 个源的实战任务 20 局实测 256/256 全清，耗时与完成
    率双优。

仍是**非严格保证**（残余漏例约 0.01%），论文按实测统计报告（verify_hearing_stats），不声称
严格不漏。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from common.routing import dist_matrix, nearest_order, two_opt_first, two_opt_greedy
from t3.config import SURVEY_CENTERS
from t4.config import (MID_RING_N, MID_RING_RAD, MID_RING_ROT_DEG,
                             OUTER_RING_N, OUTER_RING_RAD, OUTER_RING_ROT_DEG)


def measure_layout() -> list[tuple[float, float]]:
    """20 个测量位置：原点 + 7 覆盖基点 + 12 均匀方位外圈点（MID_RING_N=0 无内部补点）。

    定案布局（见模块文档）：听率 99.9891% + 贴边对抗 0 漏 + 20 局演练 6 445 s、24 392 m
    （20 局 256/256 全清）；比曾用的 23 点版（+3 内部补点，该轮听率 99.9974%）少 ~34 次测向。
    """
    base = np.asarray(SURVEY_CENTERS, dtype=float)
    ang = np.arange(OUTER_RING_N, dtype=float) * (2.0 * math.pi / OUTER_RING_N) \
        + math.radians(OUTER_RING_ROT_DEG)   # 外圈起始旋转角：与内圈基点方位错开（2026-09-12）
    ring = np.stack((OUTER_RING_RAD * np.cos(ang), OUTER_RING_RAD * np.sin(ang)), axis=1)
    mid_pts = np.empty((0, 2))
    if MID_RING_N > 0:                        # MID_RING_N=0 时无内部补点（20 点定案）
        mid = np.arange(MID_RING_N, dtype=float) * (2.0 * math.pi / MID_RING_N) \
            + math.radians(MID_RING_ROT_DEG)
        mid_pts = np.stack((MID_RING_RAD * np.cos(mid), MID_RING_RAD * np.sin(mid)), axis=1)
    return [(0.0, 0.0)] + [(float(x), float(y)) for x, y in base] + \
        [(float(x), float(y)) for x, y in mid_pts] + \
        [(float(x), float(y)) for x, y in ring]


@dataclass(frozen=True)
class SweepPlan:
    """扫描方案：测量点集合 + 访问顺序（从原点出发的开放路径）。

    `points` 的第 0 个恒为原点 (0, 0)；`route` 是 points 的下标列表，表示机器狗的访问顺序
    （以 0 开头）。`route_m` 为按该顺序走完的里程。`verification` 保存听到率统计
    （见 verify_hearing_stats），供报告与校验脚本引用。
    """

    outer_n: int
    outer_radius: float
    points: np.ndarray = field(repr=False)
    route: list[int] = field(repr=False)
    route_m: float = 0.0
    verification: dict[str, Any] | None = None
    interior_n: int = 0
    interior_radius: float = 0.0

    @property
    def n_points(self) -> int:
        """测量位置个数（含原点的起点扫描）"""
        return len(self.points)

    def route_points(self) -> list[tuple[float, float]]:
        """按访问顺序列出测量位置坐标（从原点起，顺序与 route 一致）"""
        return [(float(self.points[i][0]), float(self.points[i][1])) for i in self.route]

    def to_json(self) -> dict[str, Any]:
        """把扫描方案导出成 JSON 可序列化的字典（落盘 t4_sweep_plan.json 用）

        Returns:
            dict[str, Any]: 布局参数、测量点数、访问里程、坐标、访问顺序与听到率统计
        """
        return {
            "outer_n": int(self.outer_n),
            "outer_radius_m": float(self.outer_radius),
            "interior_n": int(self.interior_n),
            "interior_radius_m": float(self.interior_radius),
            "n_measure_points": int(self.n_points),
            "n_base": 7,
            "n_outer": int(self.outer_n),
            "route_m": round(float(self.route_m), 2),
            "verification": self.verification,
            "points": [[float(x), float(y)] for x, y in self.points],
            "route": list(self.route),
        }


def build_sweep_plan() -> SweepPlan:
    """构造默认扫描方案：20 个测量位置，从原点出发的最近邻 + 2-opt 精修开路径。

    顺序与里程不参与检测效果（效果只取决于**点集**），只影响行驶耗时，故这里取确定性下
    里程较短的一种；`nearest_order`/`two_opt_*` 都是无随机算子（见 common.routing）。
    """
    arr = np.asarray(measure_layout(), dtype=float)
    D = dist_matrix(arr, (0.0, 0.0))
    order = two_opt_greedy(two_opt_first(nearest_order(arr, start=(0.0, 0.0)), D), D)
    route = list(order)
    route_m = 0.0
    prev = arr[0]
    for i in order:
        route_m += float(np.hypot(*(arr[i] - prev)))
        prev = arr[i]
    return SweepPlan(outer_n=OUTER_RING_N, outer_radius=OUTER_RING_RAD,
                     interior_n=MID_RING_N, interior_radius=MID_RING_RAD,
                     points=arr, route=route, route_m=route_m)


def plan_from_points(points: Sequence[Sequence[float]]) -> SweepPlan:
    """按给定点集构造扫描方案（供布局对照实验复现同一路线口径）。

    与 :func:`build_sweep_plan` 用完全相同的确定性排序（最近邻 + 2-opt 开路径），故
    "某布局在 20 局演练里耗时多少"这一对照口径与定案布局一致。点集第 0 个应为原点。
    报告用的 `outer_n` / `interior_n` 按半径粗分（> 1200 m 记外圈），只影响报告文字。
    """
    arr = np.asarray(points, dtype=float)
    assert arr.ndim == 2 and arr.shape[1] == 2, "点集形状应为 (K, 2)"
    D = dist_matrix(arr, (0.0, 0.0))
    order = two_opt_greedy(two_opt_first(nearest_order(arr, start=(0.0, 0.0)), D), D)
    route = list(order)
    route_m = 0.0
    prev = arr[0]
    for i in order:
        route_m += float(np.hypot(*(arr[i] - prev)))
        prev = arr[i]
    n_outer = int((np.hypot(arr[:, 0], arr[:, 1]) > 1200.0).sum())
    return SweepPlan(outer_n=n_outer, outer_radius=OUTER_RING_RAD,
                     interior_n=max(0, len(arr) - 1 - n_outer), interior_radius=MID_RING_RAD,
                     points=arr, route=route, route_m=route_m)


def _hit_report(pts: np.ndarray, g: np.ndarray, theta_rad: float, R: float) -> tuple[bool, float]:
    """单个算例：是否存在测量点在（距离 ≤ R 且 在光束内）；返回 (命中?, 命中深度 |m−g|/R)。

    保留为**单例参考实现**：批量路径（`_hit_cases`）必须与它逐例等价，`python -m t4.sweep`
    的随机对照就是这么核的。热点不在这里 —— 整套统计有 422 万个算例，逐个调用本函数时开销全在
    numpy 调用本身（每个算例只算 20 个点），故统计走批量路径。
    """
    d = pts - g
    r2 = np.einsum("ij,ij->i", d, d)
    ok = (r2 <= R * R + 1e-9) & (
        d[:, 0] * math.cos(theta_rad) + d[:, 1] * math.sin(theta_rad) >= -1e-9)
    if not ok.any():
        return False, math.inf
    return True, float(np.sqrt(r2[ok]).min() / R)


def _hit_cases(pts: np.ndarray, gx: np.ndarray, gy: np.ndarray, theta: np.ndarray,
               R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """**一批**算例的命中掩码与命中深度；逐例与 `_hit_report` 同一套算式（逐元素对应）。

    批量口径与逐例完全一致的三处细节：距离仍是"两分量平方和"（与 `einsum` 两次乘加同序）；
    命中深度仍是"先取命中点里最小的 r²，再开一次方除以 R"（开方单调，故 √(min r²) = min √(r²)）；
    判定用的容差项 1e-9 原样保留。
    """
    pts = np.asarray(pts, dtype=float)
    dx = pts[None, :, 0] - np.asarray(gx, dtype=float)[:, None]
    dy = pts[None, :, 1] - np.asarray(gy, dtype=float)[:, None]
    r2 = dx * dx + dy * dy
    rr = np.asarray(R, dtype=float)[:, None]
    el = np.asarray(theta, dtype=float)
    ok = (r2 <= rr * rr + 1e-9) & (dx * np.cos(el)[:, None] + dy * np.sin(el)[:, None] >= -1e-9)
    hit = ok.any(axis=1)
    depth = np.sqrt(np.where(ok, r2, np.inf).min(axis=1)) / np.asarray(R, dtype=float)
    return hit, depth


def _scan_cases(pts: np.ndarray, gx: np.ndarray, gy: np.ndarray, theta: np.ndarray,
                R: np.ndarray, chunk: int = 20_000) -> tuple[int, int, dict | None]:
    """按**枚举顺序**分批扫描算例，返回 (算例数, 漏例数, 首个漏例描述)。

    首个漏例取"顺序上最早的那个"，与原先逐例扫描的口径一致（报告里的 `worst_fail`）。

    `chunk` 只影响一次批处理多大、不改变任何数值：批太大时中间数组超出 CPU 缓存，元素处理
    反而变慢。本机实测 422 万算例的自检耗时：200 000 → 2.8 s、50 000 → 2.2 s、
    **20 000 → 1.5 s**、5 000 → 1.6 s，故取 20 000。
    """
    gx = np.asarray(gx, dtype=float)
    n_cases = n_fail = 0
    worst: dict | None = None
    for i in range(0, gx.size, chunk):
        hit, _ = _hit_cases(pts, gx[i:i + chunk], gy[i:i + chunk], theta[i:i + chunk],
                            R[i:i + chunk])
        n_cases += int(hit.size)
        miss = ~hit
        n_miss = int(miss.sum())
        n_fail += n_miss
        if n_miss and worst is None:
            k = int(np.argmax(miss))                      # 本批里第一个漏例
            worst = {"g": [float(gx[i + k]), float(gy[i + k])],
                     "theta_deg": round(math.degrees(float(theta[i + k])), 1),
                     "R_m": float(R[i + k])}
    return n_cases, n_fail, worst


@lru_cache(maxsize=8)
def adversarial_cases(n_azi: int = 360, n_off: int = 40, radius: float = 1770.0,
                      Rs: Sequence[float] = (1000.0, 1250.0, 1500.0),
                      span_deg: float = 8.0, wrap: bool = True
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """贴边对抗算例 `(gx, gy, θ/rad, R)`：源贴在源生成圆盘边缘、θ 取径向 ±span/2 内的偏角。

    枚举顺序是 R → 方位 → 偏角，与原先的双层循环完全一致（`wrap=True` 时 θ 取模 2π，与
    `verify_hearing_stats` 的贴边对抗口径相同；布局搜索里的快速子集版不取模，故用 `wrap=False`）。

    结果按参数缓存（返回的是**共享的只读数组**，调用方不要就地修改）：算例集合只与参数有关、
    与测量点布局无关，而布局搜索会拿同一批算例评估成千上万个候选布局。
    """
    gx: list[float] = []
    gy: list[float] = []
    th: list[float] = []
    rr: list[float] = []
    for R in Rs:
        for ka in range(int(n_azi)):
            a = 2.0 * math.pi * ka / n_azi
            cx = radius * math.cos(a)
            cy = radius * math.sin(a)
            for ko in range(int(n_off)):
                off = (ko / (n_off - 1) - 0.5) * span_deg
                t = a + math.radians(off)
                gx.append(cx)
                gy.append(cy)
                th.append(t % (2.0 * math.pi) if wrap else t)
                rr.append(R)
    return (np.asarray(gx, dtype=float), np.asarray(gy, dtype=float),
            np.asarray(th, dtype=float), np.asarray(rr, dtype=float))


def verify_hearing_stats(points: Sequence[Sequence[float]],
                         mc_n: int = 4_000_000,
                         seed: int = 2026) -> dict[str, Any]:
    """对测量点集合做"半圆盘命中"统计校验，返回听到率与最坏漏例。

    校验 1（贴边对抗，最严）：g 贴在源生成圆盘边缘（1770 m）、θ 取径向 ±8° 内 40 个偏角 ×
     360 方位、R ∈ {1000,1250,1500}——精确打击"外翻光束"刀口（1755/1765 亦含在 0~1800 枚举）。
    校验 2（精细对抗枚举）：g 取半径 {0,300,…,1770} × 48 方位，θ 取 180 个等分角，R 取 3 档。
    校验 3（蒙特卡洛）：g 在圆域内面积均匀、θ 均匀、R ∈ [1000,1500] 均匀，共 mc_n 例。

    返回：{n_cases, n_fail, hear_rate, worst_fail(首例漏例), mc_miss, edge_miss}。
    注意这是**统计**口径（本布局仍非严格 0 漏），调用方不得把它当"严格不漏"使用。

    422 万个算例按批（5 万例一批）过 `_hit_cases`：逐例调用时每个算例只算 20 个测量点、时间
    几乎全花在 numpy 调用开销上，批量后同一份算式快一个数量级；枚举顺序与随机数抽取顺序都
    保持不变，故统计口径（含"首个漏例"）与逐例完全一致。
    """
    P = np.asarray(points, dtype=float)
    n_cases = 0
    n_fail = 0
    worst_fail: dict | None = None
    edge_miss = 0

    # 校验 1：贴边对抗（枚举顺序：R → 方位 → 径向偏角）
    n1, nf1, w1 = _scan_cases(P, *adversarial_cases(360, 40))
    n_cases += n1
    n_fail += nf1
    worst_fail = w1
    edge_miss = nf1

    # 校验 2：精细对抗枚举（枚举顺序：R → 半径 → 方位 → 光束角）
    g2x: list[float] = []
    g2y: list[float] = []
    t2: list[float] = []
    r2_list: list[float] = []
    for R in (1000.0, 1250.0, 1500.0):
        for r in list(range(0, 1800, 300)) + [1770]:
            for k in range(48):
                a = 2.0 * math.pi * k / 48
                gx = r * math.cos(a)
                gy = r * math.sin(a)
                for t in (2.0 * math.pi * kk / 180 for kk in range(180)):
                    g2x.append(gx)
                    g2y.append(gy)
                    t2.append(t)
                    r2_list.append(R)
    n2, nf2, w2 = _scan_cases(P, g2x, g2y, np.asarray(t2, dtype=float),
                              np.asarray(r2_list, dtype=float))
    n_cases += n2
    n_fail += nf2
    if worst_fail is None:
        worst_fail = w2

    # 校验 3：蒙特卡洛（抽取顺序与原先逐例循环完全一致：g_r → g_a → θ → R）
    rng = np.random.default_rng(seed)
    mc_miss = 0
    for _ in range(max(1, mc_n // 1000)):
        g_r = 1770.0 * np.sqrt(rng.random(1000))
        g_a = rng.random(1000) * 2.0 * math.pi
        gs = np.stack((g_r * np.cos(g_a), g_r * np.sin(g_a)), axis=1)
        th = rng.random(1000) * 2.0 * math.pi
        RR = rng.uniform(1000.0, 1500.0, 1000)
        hit, _ = _hit_cases(P, gs[:, 0], gs[:, 1], th, RR)
        miss = ~hit
        n_miss = int(miss.sum())
        n_cases += int(hit.size)
        n_fail += n_miss
        mc_miss += n_miss
        if n_miss and worst_fail is None:
            k = int(np.argmax(miss))
            worst_fail = {"g": [float(gs[k, 0]), float(gs[k, 1])],
                          "theta_deg": round(math.degrees(float(th[k])), 1),
                          "R_m": float(RR[k])}

    return {
        "n_cases": n_cases,
        "n_fail": n_fail,
        "hear_rate": round(1.0 - n_fail / n_cases, 6),
        "mc_miss": mc_miss,
        "mc_n": mc_n,
        "edge_miss": edge_miss,
        "worst_fail": worst_fail,
        "guaranteed": False,
        "note": "原点 + 7 覆盖基点 + 12 均匀外圈点（20 点定案）：实测听到率统计（非严格不漏）",
    }


def print_sweep_report(plan: SweepPlan, verify: dict[str, Any] | None) -> None:
    """打印扫描方案与听到率统计报告（命令行 --plan-only / 每局开头使用）。"""
    print(f"扫描方案：7 覆盖基点"
          + (f" + {plan.interior_n} 内部补点" if plan.interior_n > 0 else "")
          + f" + {plan.outer_n} 均匀外圈点"
          f"（r={plan.outer_radius:.0f} m），测量位置 {plan.n_points} 个（含原点起点扫描），"
          f"访问里程 {plan.route_m:.0f} m")
    if verify is None:
        print("  （未做听到率统计）")
        return
    if verify["n_fail"] == 0:
        print(f"  听到率统计：{verify['n_cases']} 个算例全部命中（本布局恰好 0 漏）")
    else:
        print(f"  听到率统计：{verify['n_cases']} 个算例中漏 {verify['n_fail']} 个"
              f"（{verify['hear_rate'] * 100:.4f}% 听到）；贴边对抗漏 "
              f"{verify['edge_miss']}、蒙特卡洛漏 {verify['mc_miss']}/{verify['mc_n']}；"
              f"首例漏 {verify['worst_fail']} —— 非严格保证，按实测报告")


if __name__ == '__main__':
    # 自检：`python -m t4.sweep` 直接打听到率统计（无需连模拟器）
    p = build_sweep_plan()
    # 批量判据的不变式核验：随机的源位置/光束角/半径上，批量与单例参考实现必须逐例同判
    rng = np.random.default_rng(11)
    gx = rng.uniform(-1800.0, 1800.0, 200)
    gy = rng.uniform(-1800.0, 1800.0, 200)
    th = rng.uniform(0.0, 2.0 * math.pi, 200)
    RR = rng.uniform(1000.0, 1500.0, 200)
    batch_hit, _ = _hit_cases(p.points, gx, gy, th, RR)
    ref = np.array([_hit_report(p.points, np.array([x, y]), t, r)[0]
                    for x, y, t, r in zip(gx, gy, th, RR)])
    same_trig = np.array_equal(np.cos(th), np.array([math.cos(float(t)) for t in th]))
    print(f"  批量判据 vs 单例参考：逐例一致 {bool(np.array_equal(batch_hit, ref))}（200 个随机算例）；"
          f"np.cos 与 math.cos 在这批角度上逐位一致 {bool(same_trig)}")
    v = verify_hearing_stats(p.points)
    print_sweep_report(p, v)