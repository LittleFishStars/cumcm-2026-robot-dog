"""寻向拖网：保证命中任意"半圆盘"的检测路点设计与校验（问题四的核心改动）。

问题三的 7 点覆盖只对**全向源**成立：它保证"圆域内任意点到最近巡视点 ≤ 1000 m"，于是走到
巡视点必能听到全域的源。问题四引入定向源后这一保证失效——定向源只在其"定向方向 ±90°"内有
信号，**背对着所有巡视点的源即使近在咫尺也听不到**（引擎返回 no_signal，见 engine.py 的
_in_directional_coverage；这就是"搜索不到还有可能是方向不对"）。

定向源在检测点 p 处可被听到 ⟺

    |p − g| ≤ R  且  (p − g) · u(θ) ≥ 0        （R 为接收半径 1000~1500，θ 为定向方向，含边界）

即检测区是"接收圆 ∩ 前向半平面"= 半径 R 的**半圆盘**，圆心 g、半径 R、朝向 θ 全部未知。
因此检测阶段必须满足：**测量点集合命中任意可能的半圆盘**（g ∈ 圆域、θ ∈ [0°,360°)、
R ≥ 1000）。

本模块的布局：间距 700 m 的方形格点，保留 |p| ≤ 2270 m 者（**含圆域外 470 m**）。两条理由
叠加成严格的解析保证：

1. 半圆盘 H(g,θ,R) 含内切圆 disk(g + (R/2)u, R/2)，其半径为 R/2 ≥ 500，圆心最远可达
   |g| + R/2 ≤ 1770 + 500 = 2270（R = 1000 m 时的最坏情形）；
2. 方形格点的覆盖半径 ρ = 700/√2 ≈ 495 ≤ 500，且格点铺满到半径 2270 之外，故**任意**上述
   内切圆内必有一个格点 ⇒ 任意半圆盘被命中（R > 1000 时内切圆更大、圆心更近，自动成立）。

于是"机器人每次都测所有未发现频道"这一朴素策略就能保证：**任何干扰源（全向或定向）在拖网
结束时都至少被听到一次**。这是第四问"确保所有干扰源被清除"的检测侧依据（清除侧见
cumcm.t4.strategy）。

数值校验（`verify_detection_guarantee`）逐一确证，三层全部 0 失败：
  * 精细对抗：半径 {0,300,…,1770} × 48 方位 × 180 光束方向 × {1000,1250,1500} R，181440 例
    （最坏命中深度 0.792，即最差情形命中点距源 792 m）；
  * 贴边对抗：源贴生成圆盘边缘（r ∈ {1755,1765,1770}）、光束取径向 ±4°，360 方位 × 40 偏角
    × 3 R，129600 例（深度 0.566）；
  * 蒙特卡洛：默认 100 万随机 (g, θ, R)。
命中深度 = min |m−g|/R（越小越深、越稳），用于量化"离漏检边界还有多远"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cumcm.common.routing import dist_matrix, nearest_order, two_opt_first, two_opt_greedy
from cumcm.t4.config import GRID_LATTICE_RAD, REGION_RADIUS, SWEEP_SPACING


def grid_points() -> List[Tuple[float, float]]:
    """拖网格点：间距 SWEEP_SPACING 的方形格点中 |p| ≤ GRID_LATTICE_RAD 者（含圆域外）。

    原点 (0, 0) 含在其中；build_sweep_plan 把它重排为访问顺序的首位（机器狗开局就在原点）。
    """
    hi = int(math.ceil((GRID_LATTICE_RAD + SWEEP_SPACING) / SWEEP_SPACING)) * SWEEP_SPACING
    ax = np.arange(-hi, hi + 1e-9, SWEEP_SPACING)
    return [(float(x), float(y)) for x in ax for y in ax
            if math.hypot(x, y) <= GRID_LATTICE_RAD + 1e-9]


@dataclass(frozen=True)
class SweepPlan:
    """寻向拖网方案：测量点集合 + 访问顺序（从原点出发的开放路径）。

    `points` 的第 0 个恒为原点 (0, 0)；`route` 是 points 的下标列表，表示机器狗的访问顺序
    （以 0 开头）。`route_m` 为按该顺序走完的里程。`verification` 保存检测保证校验结果
    （见 verify_detection_guarantee），供报告与校验脚本引用。
    """

    spacing: float
    lattice_radius: float
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
            "spacing_m": float(self.spacing),
            "lattice_radius_m": float(self.lattice_radius),
            "n_points": int(self.n_points),
            "route_m": round(float(self.route_m), 2),
            "verification": self.verification,
            "points": [[float(x), float(y)] for x, y in self.points],
            "route": list(self.route),
        }


def build_sweep_plan() -> SweepPlan:
    """构造默认的寻向拖网方案：全部格点，从原点出发的最近邻 + 2-opt 精修开路径。

    顺序与里程不参与检测保证（保证只取决于**点集**），只影响行驶耗时，故这里取确定性下
    里程较短的一种；`nearest_order`/`two_opt_*` 都是无随机算子（见 common.routing）。
    原点强制排在 points 的第 0 位（机器狗开局就在原点，起点拖网点即第 0 个测量簇）。
    """
    grid = grid_points()
    if abs(grid[0][0]) > 1e-9 or abs(grid[0][1]) > 1e-9:
        grid = [(0.0, 0.0)] + [p for p in grid if abs(p[0]) > 1e-9 or abs(p[1]) > 1e-9]
    arr = np.asarray(grid, dtype=float)
    D = dist_matrix(arr, (0.0, 0.0))
    order = two_opt_greedy(two_opt_first(nearest_order(arr, start=(0.0, 0.0)), D), D)
    route = list(order)
    route_m = 0.0
    prev = arr[0]
    for i in order:
        route_m += float(np.hypot(*(arr[i] - prev)))
        prev = arr[i]
    return SweepPlan(spacing=SWEEP_SPACING, lattice_radius=GRID_LATTICE_RAD,
                     points=arr, route=route, route_m=route_m)


def _in_beam(m: np.ndarray, g: np.ndarray, theta_rad: float) -> np.ndarray:
    """测量点 m 是否在源 g 的定向光束内：(m−g)·u(θ) ≥ 0（含边界）。"""
    d = m - g
    return d[:, 0] * math.cos(theta_rad) + d[:, 1] * math.sin(theta_rad) >= -1e-9


def _hit_report(pts: np.ndarray, g: np.ndarray, theta_rad: float, R: float) -> Tuple[bool, float]:
    """单个算例：是否存在测量点在（距离 ≤ R 且 在光束内）；返回 (命中?, 命中深度 |m−g|/R)。"""
    d = pts - g
    r2 = np.einsum("ij,ij->i", d, d)
    ok = (r2 <= R * R + 1e-9) & _in_beam(pts, g, theta_rad)
    if not ok.any():
        return False, math.inf
    return True, float(np.sqrt(r2[ok]).min() / R)


def verify_detection_guarantee(points: Sequence[Sequence[float]],
                               edge_extra: bool = True,
                               mc_n: int = 1_000_000,
                               seed: int = 2026) -> Dict[str, Any]:
    """对测量点集合做"半圆盘命中"三层校验，返回统计与最坏情形。

    校验 1（精细对抗枚举）：g 取半径 {0,300,…,1770} × 48 方位，θ 取 180 个等分角，R 取
    {1000,1250,1500}，共 181440 例。
    校验 2（贴边对抗，edge_extra=True）：g 贴在源生成圆盘边缘（1755/1765/1770），θ 取径向
    ±4° 内 40 个偏角 × 360 方位，3 个 R，共 129600 例——专门打击"外翻光束"的刀口情形。
    校验 3（蒙特卡洛）：g 在圆域内面积均匀、θ 均匀、R ∈ [1000,1500] 均匀，共 mc_n 例。

    返回：all_pass / n_cases / n_fail / worst_depth / margin_m / worst_fail（首例失败）。
    """
    P = np.asarray(points, dtype=float)
    radii = list(range(0, 1800, 300)) + [1770]
    thetas = [2.0 * math.pi * k / 180 for k in range(180)]
    Rs = (1000.0, 1250.0, 1500.0)
    worst_depth = 0.0
    n_fail = 0
    n_cases = 0
    worst_fail: Optional[dict] = None

    for R in Rs:
        for r in radii:
            for k in range(48):
                a = 2.0 * math.pi * k / 48
                g = np.array([r * math.cos(a), r * math.sin(a)])
                for t in thetas:
                    ok, depth = _hit_report(P, g, t, R)
                    n_cases += 1
                    if not ok:
                        n_fail += 1
                        if worst_fail is None:
                            worst_fail = {"g": [float(g[0]), float(g[1])],
                                          "theta_deg": round(math.degrees(t), 1),
                                          "R_m": float(R)}
                    worst_depth = max(worst_depth, depth if ok else 1.0)

    if edge_extra:
        for R in Rs:
            for r in (1755.0, 1765.0, 1770.0):
                for ka in range(360):
                    a = 2.0 * math.pi * ka / 360
                    g = np.array([r * math.cos(a), r * math.sin(a)])
                    for ko in range(40):
                        off = (ko / 39.0 - 0.5) * 8.0
                        t = (a + math.radians(off)) % (2.0 * math.pi)
                        ok, depth = _hit_report(P, g, t, R)
                        n_cases += 1
                        if not ok:
                            n_fail += 1
                            if worst_fail is None:
                                worst_fail = {"g": [float(g[0]), float(g[1])],
                                              "theta_deg": round(math.degrees(t), 1),
                                              "R_m": float(R)}
                        worst_depth = max(worst_depth, depth if ok else 1.0)

    rng = np.random.default_rng(seed)
    mc_fail = 0
    for _ in range(max(1, mc_n // 1000)):
        g_r = REGION_RADIUS * np.sqrt(rng.random(1000))
        g_a = rng.random(1000) * 2.0 * math.pi
        gs = np.stack((g_r * np.cos(g_a), g_r * np.sin(g_a)), axis=1)
        th = rng.random(1000) * 2.0 * math.pi
        RR = rng.uniform(1000.0, 1500.0, 1000)
        for g, t, R in zip(gs, th, RR):
            n_cases += 1
            ok, depth = _hit_report(P, g, t, R)
            if not ok:
                n_fail += 1
                mc_fail += 1
                if worst_fail is None:
                    worst_fail = {"g": [float(g[0]), float(g[1])],
                                  "theta_deg": round(math.degrees(t), 1), "R_m": float(R)}
            worst_depth = max(worst_depth, depth if ok else 1.0)

    return {
        "all_pass": n_fail == 0,
        "n_cases": n_cases,
        "n_fail": n_fail,
        "worst_depth": round(worst_depth, 4),
        "mc_fail": mc_fail,
        "worst_fail": worst_fail,
        "margin_m": round((1.0 - worst_depth) * 1000.0, 1) if n_fail == 0 else None,
    }


def print_sweep_report(plan: SweepPlan, verify: Dict[str, Any]) -> None:
    """打印拖网方案与覆盖校验报告（命令行 --plan-only / 每局开头使用）。"""
    print(f"寻向拖网方案：栅距 {plan.spacing:.0f} m、格点保留半径 {plan.lattice_radius:.0f} m，"
          f"测量点 {plan.n_points} 个，访问里程 {plan.route_m:.0f} m")
    if verify is None:
        print("  （未做半圆盘命中校验）")
        return
    if verify["all_pass"]:
        print(f"  半圆盘命中校验：{verify['n_cases']} 个算例全部命中，"
              f"最坏命中深度 {verify['worst_depth']:.3f}"
              f"（离漏检边界的余量 {verify['margin_m']:.0f} m）")
    else:
        print(f"  ⚠ 半圆盘命中校验失败 {verify['n_fail']}/{verify['n_cases']}："
              f"首例 {verify['worst_fail']} —— 布局不可用，请调整参数！")


if __name__ == "__main__":
    # 自检：`python -m cumcm.t4.sweep` 直接打校验报告（无需连模拟器）
    p = build_sweep_plan()
    v = verify_detection_guarantee(p.points)
    print_sweep_report(p, v)