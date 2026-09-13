"""问题四的扫描布局与听到率统计"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from common.console import EMPTY_CELL, print_table
from common.routing import dist_matrix, nearest_order, two_opt_first, two_opt_greedy
from t3.config import SURVEY_CENTERS
from t4.config import (MID_RING_N, MID_RING_RAD, MID_RING_ROT_DEG,
                             OUTER_RING_N, OUTER_RING_RAD, OUTER_RING_ROT_DEG)


def measure_layout() -> list[tuple[float, float]]:
    """20 个测量位置：原点 + 7 覆盖基点 + 12 均匀方位外圈点"""
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
    """扫描方案：测量点集合 + 访问顺序，从原点出发的开放路径"""
    # points 的第 0 个恒为原点 (0, 0)，route 是 points 的下标列表、以 0 开头
    # route_m 是按该顺序走完的里程，verification 存 verify_hearing_stats 的统计
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
        """测量位置个数，含原点那一次起点扫描"""
        return len(self.points)

    def route_points(self) -> list[tuple[float, float]]:
        """按访问顺序列出测量位置坐标，从原点起，顺序与 route 一致"""
        return [(float(self.points[i][0]), float(self.points[i][1])) for i in self.route]

    def to_json(self) -> dict[str, Any]:
        """把扫描方案导出成 JSON 可序列化的字典，落盘 t4_sweep_plan.json 用"""
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
    """构造默认扫描方案：20 个测量位置，从原点出发，最近邻 + 2-opt 精修开路径"""
    # 顺序与里程都不参与检测效果，只影响行驶耗时，所以取确定性下里程较短的一种，
    # nearest_order 与 two_opt_* 都是无随机算子，见 common.routing
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
    """按给定点集构造扫描方案，供布局对照实验复现同一路线口径"""
    arr = np.asarray(points, dtype=float)
    # 点集来自 --layout-file，是外部输入，用 raise 而不是 assert，免得 -O 下这道校验被摘掉
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
        raise ValueError(f"点集形状应为 (K, 2)，K ≥ 1，实际是 {arr.shape}")
    D = dist_matrix(arr, (0.0, 0.0))
    order = two_opt_greedy(two_opt_first(nearest_order(arr, start=(0.0, 0.0)), D), D)
    route = list(order)
    route_m = 0.0
    prev = arr[0]
    for i in order:
        route_m += float(np.hypot(*(arr[i] - prev)))
        prev = arr[i]
    n_outer = int((np.hypot(arr[:, 0], arr[:, 1]) > 1200.0).sum())   # 只影响报告里的内外圈计数
    return SweepPlan(outer_n=n_outer, outer_radius=OUTER_RING_RAD,
                     interior_n=max(0, len(arr) - 1 - n_outer), interior_radius=MID_RING_RAD,
                     points=arr, route=route, route_m=route_m)


def _hit_report(pts: np.ndarray, g: np.ndarray, theta_rad: float, R: float) -> tuple[bool, float]:
    """单个算例：是否存在测量点在距离 ≤ R 且在光束内；返回 (命中?, 命中深度 |m−g|/R)"""
    d = pts - g
    r2 = np.einsum("ij,ij->i", d, d)
    ok = (r2 <= R * R + 1e-9) & (
        d[:, 0] * math.cos(theta_rad) + d[:, 1] * math.sin(theta_rad) >= -1e-9)
    if not ok.any():
        return False, math.inf
    return True, float(np.sqrt(r2[ok]).min() / R)


def _hit_cases(pts: np.ndarray, gx: np.ndarray, gy: np.ndarray, theta: np.ndarray,
               R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """一批算例的命中掩码与命中深度；逐例与 `_hit_report` 同一套算式，逐元素对应"""
    # 批量口径要逐例等价：距离仍按两分量平方和算，命中深度仍是先取最小 r² 再开方除以 R，
    # 判定容差 1e-9 原样保留，`python -m t4.sweep` 的随机对照就是这么核的
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
    """按枚举顺序分批扫描算例，返回 (算例数, 漏例数, 首个漏例描述)"""
    # chunk 只定一次批处理多大，不改数值：批太大中间数组出 CPU 缓存反而变慢，本机 422 万算例
    # 的自检耗时 200 000 → 2.8 s、20 000 → 1.5 s，所以取 20 000
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


@lru_cache(maxsize=8)                     # 算例只与参数有关，布局搜索会拿同一批反复评估候选
def adversarial_cases(n_azi: int = 360, n_off: int = 40, radius: float = 1770.0,
                      Rs: Sequence[float] = (1000.0, 1250.0, 1500.0),
                      span_deg: float = 8.0, wrap: bool = True
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """贴边对抗算例 `(gx, gy, θ/rad, R)`：源贴在源生成圆盘边缘、θ 取径向 ±span/2 内的偏角"""
    # 返回的是缓存里的共享只读数组，调用方不要就地修改；枚举顺序 R → 方位 → 偏角
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
                th.append(t % (2.0 * math.pi) if wrap else t)   # 不取模的那版供布局搜索用
                rr.append(R)
    return (np.asarray(gx, dtype=float), np.asarray(gy, dtype=float),
            np.asarray(th, dtype=float), np.asarray(rr, dtype=float))


def verify_hearing_stats(points: Sequence[Sequence[float]],
                         mc_n: int = 4_000_000,
                         seed: int = 2026) -> dict[str, Any]:
    """对测量点集合做"半圆盘命中"统计校验，返回听到率与最坏漏例
    这是统计口径，本布局仍有零星漏例，调用方不得把它当"严格不漏"用
    """
    P = np.asarray(points, dtype=float)
    n_cases = 0
    n_fail = 0
    worst_fail: dict | None = None
    edge_miss = 0

    # 校验 1：贴边对抗（枚举顺序：R → 方位 → 径向偏角），g 贴 1770 m 圆盘边缘、θ 取径向
    # ±8° 内 40 个偏角乘 360 方位，这一档专打"外翻光束"的刀口，也是最严的一档
    n1, nf1, w1 = _scan_cases(P, *adversarial_cases(360, 40))
    n_cases += n1
    n_fail += nf1
    worst_fail = w1
    edge_miss = nf1

    # 校验 2：精细对抗枚举，g 取半径 {0,300,…,1770} × 48 方位，θ 取 180 个等分角，R 取 3 档
    # 枚举顺序：R → 半径 → 方位 → 光束角
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

    # 校验 3：蒙特卡洛，g 在圆域内面积均匀、θ 均匀、R ∈ [1000,1500] 均匀
    # 抽取顺序与原先逐例循环完全一致：g_r → g_a → θ → R
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
        "note": "原点 + 7 覆盖基点 + 12 均匀外圈点（20 点定案）：实测听到率统计，不是严格不漏",
    }


def point_rows(plan: SweepPlan) -> list[list[Any]]:
    """测量点清单的行：序号 / x / y / 是否在圆域内，序号按访问顺序，与落盘 CSV 对齐"""
    rows: list[list[Any]] = []
    for k, i in enumerate(plan.route):
        x, y = plan.points[i]
        rows.append([k, f"{x:.2f}", f"{y:.2f}",
                     "是" if float(np.hypot(x, y)) <= 1800.0 else "否"])
    return rows


def plan_summary_rows(plan: SweepPlan) -> list[list[Any]]:
    """扫描方案概况的行：测量位置分区计数与访问里程"""
    rows: list[list[Any]] = [["测量位置合计", plan.n_points], ["覆盖基点 + 原点", 8]]
    # 内部补点那档现在是 0，方案里没有这一项就不占表里一行
    if plan.interior_n > 0:
        rows.append(["内部补点", plan.interior_n])
    rows.append([f"均匀外圈点 r={plan.outer_radius:.0f} m", plan.outer_n])
    rows.append(["访问里程 / km", f"{plan.route_m / 1000:.2f}"])
    return rows


def verify_rows(verify: dict[str, Any], verbose: bool = False) -> list[list[Any]]:
    """听到率统计的行：算例数 / 漏数 / 听到率 / 贴边漏 / 蒙特卡洛漏，verbose 时补首例漏"""
    rows: list[list[Any]] = [
        ["算例数", verify["n_cases"], EMPTY_CELL],
        ["漏数", verify["n_fail"], f"听到率 {verify['hear_rate'] * 100:.4f}%"],
        ["贴边对抗漏", verify["edge_miss"], EMPTY_CELL],
        ["蒙特卡洛漏", verify["mc_miss"], f"共 {verify['mc_n']}"],
    ]
    if verbose:
        w = verify["worst_fail"] or {}
        g = w.get("g") or [None, None]
        rows.append(["首例漏", f"g=({g[0]}, {g[1]})",
                     f"θ={w.get('theta_deg')}°，R={w.get('R_m')} m"])
    else:
        rows.append(["首例漏", EMPTY_CELL, "非严格保证，按实测报告"])
    return rows


def print_point_table(plan: SweepPlan, title: str = "测量点清单") -> None:
    """打印 20 个测量位置的清单表，逐点一行"""
    print(title)
    print_table(["序号", "x / m", "y / m", "在圆域内"], point_rows(plan), align="rrrl")


def print_sweep_summary(plan: SweepPlan, verify: dict[str, Any] | None,
                        src: str | None = None) -> None:
    """打印扫描方案概况与听到率统计两张精简表，缺省档用，src 是布局文件来源"""
    print(f"扫描方案（布局来自 {src}）" if src else "扫描方案")
    print_table(["项目", "数值"], plan_summary_rows(plan), align="lr")
    if verify is None:
        print("听到率统计：未做统计")
        return
    print("听到率统计")
    print_table(["指标", "数值", "备注"], verify_rows(verify), align="lrl")


def print_sweep_report(plan: SweepPlan, verify: dict[str, Any] | None) -> None:
    """打印扫描方案与听到率统计报告，详细档用，逐点清单与逐项校验都成表"""
    print("扫描方案（含原点起点扫描）")
    print_table(["项目", "数值"], plan_summary_rows(plan), align="lr")
    print_point_table(plan)
    if verify is None:
        print("听到率统计：未做统计")
        return
    print("听到率统计")
    print_table(["指标", "数值", "备注"], verify_rows(verify, verbose=True), align="lrl")
    # 严格不漏与否只认统计口径这一句，guaranteed 标记留着是给以后换成严格布局用的
    print(f"  统计口径：{verify['note']}")


def print_check_table(rows: Sequence[Sequence[Any]]) -> None:
    """打印自检的检查项表：检查项 | 结果 | 数值，见 `python -m t4.sweep`"""
    print_table(["检查项", "结果", "数值"], [list(r) for r in rows], align="lll")


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
    print_check_table([
        ["批量判据 vs 单例参考", "逐例一致" if bool(np.array_equal(batch_hit, ref)) else "不一致",
         "200 个随机算例"],
        ["np.cos vs math.cos", "逐位一致" if bool(same_trig) else "有位差",
         f"{th.size} 个随机角度"],
    ])
    v = verify_hearing_stats(p.points)
    print_sweep_report(p, v)
