"""扫描布局：原点 + 7 覆盖基点 + 12 个均匀方位外圈点（问题四检测阶段）。

定向源在检测点 p 处可被听到 ⟺

    |p − g| ≤ R  且  (p − g) · u(θ) ≥ 0        （R 为接收半径 1000~1500，θ 为定向方向，含边界）

即检测区是"接收圆 ∩ 前向半平面"= 半径 R 的**半圆盘**，圆心 g、半径 R、朝向 θ 全部未知。
问题三的 7 点覆盖只保证"圆域内任意点到最近巡视点 ≤ 1000 m"（对全向源必听到）；对定向源，
**背对着所有巡视点的源即使近在咫尺也听不到**（引擎返回 no_signal，见 engine.py 的
_in_directional_coverage —— 这就是"搜索不到还有可能是方向不对"）。

本模块的布局（"尽量减少路程"优化后的 20 个测量位置）：

  1. **原点**：起点全频道扫描（位置成本为零）；
  2. **7 个覆盖基点** = 问题三巡视站（cumcm.t3.config.SURVEY_CENTERS，直接复用，全向覆盖保证）；
  3. **12 个均匀方位外圈点**，半径 OUTER_RING_RAD = 1850 m。

为什么外圈取 1850 m、12 个方位（这是对"7+21 中点外推"布局的结构优化）：

  * **半径 1850 m**：任意源的迎光区在源前方 [|g|, |g|+R] 纵深；源最远半径 1770，所以**任意
    贴边源**的迎光区内沿 = 1770 m，外圈设在其内侧 1850 m 时径向偏差 < 80 m。原"中点外推"
    布局把多数点推到 2270 m，反而离内部源的迎光面太远（实测只剩 99.81%）；1850 m 单圈、
    12 个方位同时覆盖贴边源与内部半径源的迎光区，实测 **99.99%**（400 万算例、4 seed 稳健，
    贴边对抗 4.3 万例 0 漏）；
  * **12 个方位（间隔 30°）**：相邻外圈点最大距离 ≈ 1850·sin15° ≈ 479 m < 接收半径一半，
    任何方向的源都能被方向差 ≤ 15° 的外圈点听到。**为什么不是更少的点**：试过 11 点
    （间隔 32.7°、布局路线 16 725 m），但方位间隔变宽使顺路清除的方位扇区也变宽、绕路更多，
    实战总里程反升 464 m——12 点是"减少路程"的实战最优；
  * **里程**：20 个测量位置的最短开放路径 17 019 m —— 比"7+21 中点外推"（29 点、21 580 m）
    省 21%，还少 9 个测量位置。

仍是**非严格保证**（内部半径源 + 波束朝外 + 方向恰落点族空隙的残余约 0.01%），论文按实测
统计报告（verify_hearing_stats），不声称严格不漏。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cumcm.common.routing import dist_matrix, nearest_order, two_opt_first, two_opt_greedy
from cumcm.t3.config import SURVEY_CENTERS
from cumcm.t4.config import OUTER_RING_N, OUTER_RING_RAD


def measure_layout() -> List[Tuple[float, float]]:
    """20 个测量位置：原点 + 7 覆盖基点 + 12 均匀方位外圈点（半径 OUTER_RING_RAD）。"""
    base = np.asarray(SURVEY_CENTERS, dtype=float)
    ang = np.arange(OUTER_RING_N, dtype=float) * (2.0 * math.pi / OUTER_RING_N)
    ring = np.stack((OUTER_RING_RAD * np.cos(ang), OUTER_RING_RAD * np.sin(ang)), axis=1)
    return [(0.0, 0.0)] + [(float(x), float(y)) for x, y in base] + \
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
    route: List[int] = field(repr=False)
    route_m: float = 0.0
    verification: Optional[Dict[str, Any]] = None

    @property
    def n_points(self) -> int:
        return len(self.points)

    def route_points(self) -> List[Tuple[float, float]]:
        return [(float(self.points[i][0]), float(self.points[i][1])) for i in self.route]

    def to_json(self) -> Dict[str, Any]:
        return {
            "outer_n": int(self.outer_n),
            "outer_radius_m": float(self.outer_radius),
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
                     points=arr, route=route, route_m=route_m)


def _hit_report(pts: np.ndarray, g: np.ndarray, theta_rad: float, R: float) -> Tuple[bool, float]:
    """单个算例：是否存在测量点在（距离 ≤ R 且 在光束内）；返回 (命中?, 命中深度 |m−g|/R)。"""
    d = pts - g
    r2 = np.einsum("ij,ij->i", d, d)
    ok = (r2 <= R * R + 1e-9) & (
        d[:, 0] * math.cos(theta_rad) + d[:, 1] * math.sin(theta_rad) >= -1e-9)
    if not ok.any():
        return False, math.inf
    return True, float(np.sqrt(r2[ok]).min() / R)


def verify_hearing_stats(points: Sequence[Sequence[float]],
                         mc_n: int = 4_000_000,
                         seed: int = 2026) -> Dict[str, Any]:
    """对测量点集合做"半圆盘命中"统计校验，返回听到率与最坏漏例。

    校验 1（贴边对抗，最严）：g 贴在源生成圆盘边缘（1770 m）、θ 取径向 ±8° 内 40 个偏角 ×
     360 方位、R ∈ {1000,1250,1500}——精确打击"外翻光束"刀口（1755/1765 亦含在 0~1800 枚举）。
    校验 2（精细对抗枚举）：g 取半径 {0,300,…,1770} × 48 方位，θ 取 180 个等分角，R 取 3 档。
    校验 3（蒙特卡洛）：g 在圆域内面积均匀、θ 均匀、R ∈ [1000,1500] 均匀，共 mc_n 例。

    返回：{n_cases, n_fail, hear_rate, worst_fail(首例漏例), mc_miss, edge_miss}。
    注意这是**统计**口径（本布局仍非严格 0 漏），调用方不得把它当"严格不漏"使用。
    """
    P = np.asarray(points, dtype=float)
    n_fail = 0
    n_cases = 0
    worst_fail: Optional[dict] = None
    edge_miss = 0

    def _case(g, t, R):
        nonlocal n_fail, n_cases, worst_fail
        ok, _ = _hit_report(P, np.asarray(g, float), t, R)
        n_cases += 1
        if not ok:
            n_fail += 1
            if worst_fail is None:
                worst_fail = {"g": [float(g[0]), float(g[1])],
                              "theta_deg": round(math.degrees(t), 1), "R_m": float(R)}

    for R in (1000.0, 1250.0, 1500.0):
        for r in (1770.0,):
            for ka in range(360):
                a = 2.0 * math.pi * ka / 360
                g0 = np.array([r * math.cos(a), r * math.sin(a)])
                for ko in range(40):
                    off = (ko / 39.0 - 0.5) * 8.0
                    _case(g0, (a + math.radians(off)) % (2.0 * math.pi), R)
    edge_miss = n_fail

    for R in (1000.0, 1250.0, 1500.0):
        for r in list(range(0, 1800, 300)) + [1770]:
            for k in range(48):
                a = 2.0 * math.pi * k / 48
                g0 = np.array([r * math.cos(a), r * math.sin(a)])
                for t in (2.0 * math.pi * kk / 180 for kk in range(180)):
                    _case(g0, t, R)

    rng = np.random.default_rng(seed)
    mc_miss = 0
    for _ in range(max(1, mc_n // 1000)):
        g_r = 1770.0 * np.sqrt(rng.random(1000))
        g_a = rng.random(1000) * 2.0 * math.pi
        gs = np.stack((g_r * np.cos(g_a), g_r * np.sin(g_a)), axis=1)
        th = rng.random(1000) * 2.0 * math.pi
        RR = rng.uniform(1000.0, 1500.0, 1000)
        for g, t, R in zip(gs, th, RR):
            ok, _ = _hit_report(P, g, t, R)
            n_cases += 1
            if not ok:
                n_fail += 1
                mc_miss += 1
                if worst_fail is None:
                    worst_fail = {"g": [float(g[0]), float(g[1])],
                                  "theta_deg": round(math.degrees(t), 1), "R_m": float(R)}

    return {
        "n_cases": n_cases,
        "n_fail": n_fail,
        "hear_rate": round(1.0 - n_fail / n_cases, 6),
        "mc_miss": mc_miss,
        "mc_n": mc_n,
        "edge_miss": edge_miss,
        "worst_fail": worst_fail,
        "guaranteed": False,
        "note": "原点 + 7 覆盖基点 + 12 均匀外圈点（r=1850 m）：实测听到率统计（非严格不漏）",
    }


def print_sweep_report(plan: SweepPlan, verify: Optional[Dict[str, Any]]) -> None:
    """打印扫描方案与听到率统计报告（命令行 --plan-only / 每局开头使用）。"""
    print(f"扫描方案：7 覆盖基点 + {plan.outer_n} 均匀外圈点（r={plan.outer_radius:.0f} m），"
          f"测量位置 {plan.n_points} 个（含原点起点扫描），访问里程 {plan.route_m:.0f} m")
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


if __name__ == "__main__":
    # 自检：`python -m cumcm.t4.sweep` 直接打听到率统计（无需连模拟器）
    p = build_sweep_plan()
    v = verify_hearing_stats(p.points)
    print_sweep_report(p, v)