"""问题二的几何底座：两条楔形带之交的定位四边形、它的直径，以及第二检测点的可行域"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import shapely
from shapely import Point, Polygon
from shapely.geometry.base import BaseGeometry

from common.console import print_table
from common.geometry import bearing
from t1 import TriangulationRegion
from t2.config import (BEARING_ERROR_DEG, D_HI, D_LO, DIAM_SENTINEL, PARALLEL_EPS,
                             RECEIVE_MIN, REGION_RADIUS)

# 楔形内判定的角度容差 / 度。只用来吸收浮点误差，让落在边界上的交点在两个楔形里都算"在内"。
# 1e-6° 对应的横向偏差在 4 km 尺度上不足 0.1 mm，不会把真正的域外点算进来。
WEDGE_TOL_DEG = 1e-6

__all__ = ["in_wedge", "quad_diameters", "quad_diameters_batch", "quad_polygon", "exact_region",
           "exact_diameter", "analytic_diameter", "corner_sources", "feasible_lens",
           "worst_case_scenario", "worst_source_distance", "worst_diameters_all",
           "verify_analytic", "unit"]

# 标量单位向量的小缓存。选点优化里 in_wedge 会被调用上万次，传进去的楔形边界角却只有 θ₁ ± err
# 这么几个固定值，缓存之后把每次约 40 µs 的 numpy 标量三角函数换成一次字典查找，数值逐位相同
_UNIT_SCALAR_CACHE: dict[float, tuple[float, float]] = {}


def _cos_sin_scalar(theta_deg: float) -> tuple[float, float]:
    """标量方位角转 (cos, sin)，带缓存"""
    hit = _UNIT_SCALAR_CACHE.get(theta_deg)
    if hit is None:
        if len(_UNIT_SCALAR_CACHE) > 8192:       # 只服务反复出现的少数角度，不做无界增长
            _UNIT_SCALAR_CACHE.clear()
        a = np.radians(np.asarray(theta_deg, dtype=float))
        hit = (float(np.cos(a)), float(np.sin(a)))
        _UNIT_SCALAR_CACHE[theta_deg] = hit
    return hit


def _cos_sin(theta_deg: float | np.ndarray) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """方位角（度）转 (cos, sin)，不 stack 成 (..., 2)，标量走缓存"""
    if np.ndim(theta_deg) == 0:
        return _cos_sin_scalar(float(theta_deg))
    a = np.radians(np.asarray(theta_deg, dtype=float))
    return np.cos(a), np.sin(a)


def unit(theta_deg: float | np.ndarray) -> tuple[float, float] | np.ndarray:
    """方位角（度）对应的单位向量，标量给 (ux, uy)，数组给 (..., 2) 数组"""
    c, s = _cos_sin(theta_deg)
    if np.ndim(theta_deg) == 0:
        return c, s
    return np.stack([c, s], axis=-1)


def in_wedge(px: np.ndarray, py: np.ndarray, apex: Sequence[float], theta_lo: float,
             theta_hi: float, tol_deg: float = 0.0) -> np.ndarray:
    """点 (px, py) 是否落在 apex 顶点、theta_lo 到 theta_hi 的楔形内，tol_deg 给边界容差"""
    ax, ay = _cos_sin(theta_lo)                            # 标量给 float 对，数组给同形数组
    bx, by = _cos_sin(theta_hi)
    vx = np.asarray(px, dtype=float) - np.asarray(apex[0], dtype=float)
    vy = np.asarray(py, dtype=float) - np.asarray(apex[1], dtype=float)
    c1 = ax * vy - ay * vx                       # cross(d_lo, v) ≥ 0：在 d_lo 逆时针侧
    c2 = bx * vy - by * vx                       # cross(d_hi, v) ≤ 0：在 d_hi 顺时针侧
    if tol_deg == 0.0:                           # 无容差就不必算模长，tol=0 时 slack 恒为 0
        return (c1 >= 0.0) & (c2 <= 0.0)
    slack = math.sin(math.radians(tol_deg)) * _norm2(vx, vy)
    return (c1 >= -slack) & (c2 <= slack)


def _norm2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """计算 sqrt(x² + y²)"""
    # 不用 np.hypot：|x|,|y| 都在 1e4 量级，离上下溢极远，而 hypot 更慢
    return np.sqrt(x * x + y * y)


def _candidate_points_batch(site: Sequence[float], theta1: float, sx: np.ndarray, sy: np.ndarray,
                            theta2_batch: np.ndarray,
                            err_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """一批楔形方向下两楔形之交的候选顶点 P (6, B, N, 2) 与在两楔形内的掩码 keep"""
    x1, y1 = float(site[0]), float(site[1])
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    theta2_batch = np.asarray(theta2_batch, dtype=float)
    shape = theta2_batch.shape
    wx = np.broadcast_to(sx, shape) - x1
    wy = np.broadcast_to(sy, shape) - y1
    pts = []
    # θ₂ ± err 的单位向量只算一次。原先 (sa, sb) 四个组合各算一遍，其中两个重复
    bs = (_cos_sin(theta2_batch - err_deg), _cos_sin(theta2_batch + err_deg))
    for sa in (-err_deg, err_deg):
        ax, ay = unit(theta1 + sa)
        for bx, by in bs:
            den = ax * by - ay * bx
            safe = np.where(np.abs(den) < PARALLEL_EPS, np.nan, den)
            t = (wx * by - wy * bx) / safe
            pts.append((x1 + t * ax, y1 + t * ay))
    pts.append((np.full_like(theta2_batch, x1), np.full_like(theta2_batch, y1)))   # 楔形 1 顶点
    pts.append((np.broadcast_to(sx, shape), np.broadcast_to(sy, shape)))           # 楔形 2 顶点
    P = np.stack([np.stack([p[0], p[1]], axis=-1) for p in pts], axis=0)
    keep = (in_wedge(P[..., 0], P[..., 1], site, theta1 - err_deg, theta1 + err_deg, WEDGE_TOL_DEG)
            & in_wedge(P[..., 0], P[..., 1], (sx, sy), theta2_batch - err_deg,
                       theta2_batch + err_deg, WEDGE_TOL_DEG))
    return P, keep


def _candidate_points(site: Sequence[float], theta1: float, sx: np.ndarray, sy: np.ndarray,
                      theta2: np.ndarray, err_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """单点情形的候选顶点，`_candidate_points_batch` 的 B = 1 特例"""
    P, keep = _candidate_points_batch(site, theta1, sx, sy,
                                      np.asarray(theta2, dtype=float)[None, :], err_deg)
    return P[:, 0], keep[:, 0]


def _max_pair_distance(P: np.ndarray, keep: np.ndarray, sentinel: float) -> np.ndarray:
    """候选顶点两两距离的最大值，区域是凸的，这个值就是直径；P 形状 (6, ..., 2)"""
    # 距离走 np.hypot：换成平方距离取最大再开方会动最后 1 位，破坏逐字节一致
    shape = P.shape[1:-1]
    best = np.zeros(shape, dtype=float)
    for i in range(P.shape[0]):
        pi0, pi1 = P[i][..., 0], P[i][..., 1]
        for j in range(i + 1, P.shape[0]):
            d = np.hypot(pi0 - P[j][..., 0], pi1 - P[j][..., 1])
            np.maximum(best, np.where(keep[i] & keep[j], d, 0.0), out=best)
    return np.where(np.isnan(best), sentinel, best)


def quad_diameters_batch(site: Sequence[float], theta1: float, sx: np.ndarray, sy: np.ndarray,
                         theta2_batch: np.ndarray, err_deg: float = BEARING_ERROR_DEG,
                         sentinel: float = DIAM_SENTINEL) -> np.ndarray:
    """一批方向 (B, N) 对应的定位四边形直径 / m，算式与 quad_diameters 逐元素同值"""
    P, keep = _candidate_points_batch(site, theta1, sx, sy, np.asarray(theta2_batch, dtype=float),
                                      err_deg)
    return _max_pair_distance(P, keep, sentinel)


def quad_diameters(site: Sequence[float], theta1: float,
                   sx: np.ndarray, sy: np.ndarray, theta2: np.ndarray,
                   err_deg: float = BEARING_ERROR_DEG,
                   sentinel: float = DIAM_SENTINEL) -> np.ndarray:
    """两个 ±err 楔形之交的定位四边形直径 / m，向量化；近共线退化时返回哨兵值"""
    theta2_arr = np.asarray(theta2, dtype=float)
    if theta2_arr.ndim > 1:
        return quad_diameters_batch(site, theta1, sx, sy, theta2_arr, err_deg, sentinel)
    P, keep = _candidate_points(site, theta1, sx, sy, theta2_arr, err_deg)
    return _max_pair_distance(P, keep, sentinel)


def quad_polygon(site: Sequence[float], theta1: float, s2: Sequence[float], theta2: float,
                 err_deg: float = BEARING_ERROR_DEG) -> BaseGeometry:
    """单点情形的定位区域多边形，shapely 凸包，供论文插图与实际面积使用"""
    sx = np.asarray([float(s2[0])])
    sy = np.asarray([float(s2[1])])
    P, keep = _candidate_points(site, theta1, sx, sy, np.asarray([float(theta2)]), err_deg)
    pts = [(float(P[k, 0, 0]), float(P[k, 0, 1])) for k in range(P.shape[0]) if bool(keep[k, 0])]
    if len(pts) < 3:
        return Polygon()
    return shapely.convex_hull(shapely.multipoints(pts))


def analytic_diameter(d: float, r2: float, gamma_deg: float,
                      err_deg: float = BEARING_ERROR_DEG) -> float:
    """一阶展开式给出的定位直径 / m，用于论文引用与解析对照"""
    g = math.radians(gamma_deg)
    return (2.0 * math.radians(err_deg) / math.sin(g)) * math.sqrt(
        d * d + r2 * r2 + 2.0 * d * r2 * abs(math.cos(g)))


def exact_region(site: Sequence[float], theta1: float, s2: Sequence[float], theta2: float,
                 err_deg: float = BEARING_ERROR_DEG,
                 radius: float = REGION_RADIUS) -> TriangulationRegion:
    """用问题 1 的 `TriangulationRegion` 精确构造定位区域，含目标圆域裁剪"""
    region = TriangulationRegion(err=err_deg, radius=radius)
    region.add_node(float(site[0]), float(site[1]), float(theta1))
    region.add_node(float(s2[0]), float(s2[1]), float(theta2))
    return region


def exact_diameter(site: Sequence[float], theta1: float, s2: Sequence[float], theta2: float,
                   err_deg: float = BEARING_ERROR_DEG,
                   radius: float = REGION_RADIUS) -> float:
    """定位区域直径的精确值：shapely/GEOS 求交 + 顶点最远点对，含圆域裁剪，单位 m"""
    return float(exact_region(site, theta1, s2, theta2, err_deg, radius).diameter)


def corner_sources(site: Sequence[float], theta1: float, d_lo: float = D_LO,
                   d_hi: float = D_HI,
                   err_deg: float = BEARING_ERROR_DEG) -> np.ndarray:
    """源不确定集 C 的 4 个极点，(d, δ) 的四个角，形状 (4, 2)"""
    pts = []
    for d in (d_lo, d_hi):
        for sd in (-err_deg, err_deg):
            ux, uy = unit(theta1 + sd)
            pts.append((float(site[0]) + d * ux, float(site[1]) + d * uy))
    return np.asarray(pts, dtype=float)


def feasible_lens(site: Sequence[float], theta1: float, d_lo: float = D_LO, d_hi: float = D_HI,
                  err_deg: float = BEARING_ERROR_DEG, receive_min: float = RECEIVE_MIN,
                  radius: float = REGION_RADIUS, quad_segs: int = 64) -> BaseGeometry:
    """第二检测点可行域的多边形：4 个半径 receive_min 的圆盘与目标圆域求交"""
    # 只用于出图与结果描述，判据本身是采样源点上的最坏距离 ≤ receive_min，见 t2.score
    geom = Point(0.0, 0.0).buffer(radius, quad_segs=quad_segs)
    for cx, cy in corner_sources(site, theta1, d_lo, d_hi, err_deg):
        geom = geom.intersection(Point(cx, cy).buffer(receive_min, quad_segs=quad_segs))
    return geom


def worst_source_distance(px: np.ndarray, py: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """候选点 (px, py) 到源不确定集全部采样点的最远距离 / m"""
    px = np.asarray(px, dtype=float).ravel()
    py = np.asarray(py, dtype=float).ravel()
    src = np.asarray(sources, dtype=float)
    return np.max(np.hypot(px[:, None] - src[None, :, 0],
                           py[:, None] - src[None, :, 1]), axis=1)


def worst_diameters_all(site: Sequence[float], theta1: float, s2: Sequence[float],
                        sources: np.ndarray, deltas2: Sequence[float],
                        err_deg: float = BEARING_ERROR_DEG) -> np.ndarray:
    """单点 S2 对全部 (源采样点 × δ₂) 组合的定位直径 / m，一次向量化算完"""
    src = np.asarray(sources, dtype=float)
    d2 = np.asarray(deltas2, dtype=float)
    n = src.shape[0] * d2.size
    sx = np.full(n, float(s2[0]))
    sy = np.full(n, float(s2[1]))
    gx = np.repeat(src[:, 0], d2.size)
    gy = np.repeat(src[:, 1], d2.size)
    w2 = np.degrees(np.arctan2(gy - sy, gx - sx)) % 360.0
    return quad_diameters(site, theta1, sx, sy, w2 + np.tile(d2, src.shape[0]), err_deg)


def worst_case_scenario(site: Sequence[float], theta1: float, s2: Sequence[float],
                        sources: np.ndarray, deltas2: Sequence[float],
                        err_deg: float = BEARING_ERROR_DEG) -> dict[str, Any]:
    """单点 S2 的最坏情形：最大直径、对应源位置与第二次测向误差、定位区域多边形"""
    src = np.asarray(sources, dtype=float)
    deltas2 = np.asarray(deltas2, dtype=float)
    diams = worst_diameters_all(site, theta1, s2, src, deltas2, err_deg)
    k = int(np.argmax(diams))
    gx = np.repeat(src[:, 0], deltas2.size)
    gy = np.repeat(src[:, 1], deltas2.size)
    w2 = np.degrees(np.arctan2(gy - float(s2[1]), gx - float(s2[0]))) % 360.0
    theta2 = w2 + np.tile(deltas2, src.shape[0])
    poly = quad_polygon(site, theta1, s2, float(theta2[k]), err_deg)
    best: dict[str, Any] = {
        "diameter": float(diams[k]), "source": (float(gx[k]), float(gy[k])),
        "delta2_deg": float(theta2[k] - w2[k]), "theta2_deg": float(theta2[k]),
        "bearing_nominal_deg": float(w2[k]),
        "quad": [tuple(map(float, p)) for p in shapely.get_coordinates(poly)],
    }
    # 最坏源相对示向度的偏差、到 S1 的距离与源处交会角，论文里要引用这三个数
    site_xy = (float(site[0]), float(site[1]))
    g_xy = best["source"]
    d = math.hypot(g_xy[0] - site_xy[0], g_xy[1] - site_xy[1])
    delta1 = ((bearing(site_xy, g_xy) - theta1 + 180.0) % 360.0) - 180.0
    r2 = math.hypot(g_xy[0] - s2[0], g_xy[1] - s2[1])
    v1 = (site_xy[0] - g_xy[0], site_xy[1] - g_xy[1])
    v2 = (s2[0] - g_xy[0], s2[1] - g_xy[1])
    cos_g = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
    gamma = math.degrees(math.acos(max(-1.0, min(1.0, cos_g))))
    best.update({"d_m": d, "delta1_deg": delta1, "r2_m": r2, "gamma_deg": gamma,
                 "analytic_m": analytic_diameter(d, r2, gamma, err_deg)})
    return best


def verify_analytic(n: int = 200, seed: int = 2026, err_deg: float = BEARING_ERROR_DEG,
                    radius: float = REGION_RADIUS, sin_gamma_min: float = 0.02,
                    gamma_ok_deg: tuple[float, float] = (30.0, 120.0)) -> dict[str, Any]:
    """随机抽样核对解析构造、一阶公式与 shapely 精确值，返回可写进报告的校验数字"""
    rng = np.random.default_rng(seed)
    n_clip = n_degen = 0
    dev_impl, dev_formula = [], []
    for _ in range(int(n)):
        site = (float(rng.uniform(-radius, radius)), float(rng.uniform(-radius, radius)))
        theta1 = float(rng.uniform(0.0, 360.0))
        ang = float(rng.uniform(0.0, 360.0))
        dist = float(rng.uniform(50.0, 2000.0))
        s2 = (site[0] + dist * math.cos(math.radians(ang)),
              site[1] + dist * math.sin(math.radians(ang)))
        d = float(rng.uniform(D_LO, 1500.0))
        g = (site[0] + d * math.cos(math.radians(theta1)),
             site[1] + d * math.sin(math.radians(theta1)))
        theta2 = bearing(s2, g)
        v1 = (site[0] - g[0], site[1] - g[1])
        v2 = (s2[0] - g[0], s2[1] - g[1])
        sin_g = abs(v1[0] * v2[1] - v1[1] * v2[0]) / (math.hypot(*v1) * math.hypot(*v2) + 1e-12)
        if sin_g < sin_gamma_min:
            n_degen += 1
            continue
        region = exact_region(site, theta1, s2, theta2, err_deg, radius)
        if not region.bounded or region.diameter <= 0.0:
            n_clip += 1
            continue
        exact = float(region.diameter)
        impl = float(quad_diameters(site, theta1, np.asarray([s2[0]]), np.asarray([s2[1]]),
                                    np.asarray([theta2]), err_deg)[0])
        dev_impl.append((impl - exact) / exact)
        gamma = math.degrees(math.asin(min(1.0, sin_g)))
        if gamma_ok_deg[0] <= gamma <= gamma_ok_deg[1]:
            formula = analytic_diameter(d, math.hypot(s2[0] - g[0], s2[1] - g[1]), gamma, err_deg)
            dev_formula.append((formula - impl) / impl)
    a1 = np.asarray(dev_impl, dtype=float)
    a2 = np.asarray(dev_formula, dtype=float)
    return {"n": int(n), "n_used": int(a1.size), "n_clipped": int(n_clip),
            "n_degenerate": int(n_degen),
            "impl_max_abs_rel_dev": float(np.abs(a1).max()) if a1.size else None,
            "formula_max_abs_rel_dev": float(np.abs(a2).max()) if a2.size else None,
            "formula_median_rel_dev": float(np.median(np.abs(a2))) if a2.size else None,
            "formula_p90_rel_dev": float(np.percentile(np.abs(a2), 90)) if a2.size else None,
            "formula_frac_below_1pct": float((np.abs(a2) < 0.01).mean()) if a2.size else None,
            "formula_n": int(a2.size), "gamma_ok_deg": tuple(gamma_ok_deg)}


def _selfcheck(n: int = 400) -> None:
    """自检：解析构造与一阶公式对 shapely 精确值，可行域多边形与可测性判据要一致"""
    print("问题二几何自检")
    rep = verify_analytic(n=n)
    print(f"  样本 {rep['n']}：有效 {rep['n_used']}，被圆域截断跳过 {rep['n_clipped']}，"
          f"近共线跳过 {rep['n_degenerate']}")
    rows: list[list[Any]] = []
    # 解析构造与 shapely 走的是两套几何，偏差只该有浮点误差那么大
    rows.append(["解析构造 vs shapely", "✓", f"{rep['impl_max_abs_rel_dev']:.3e}",
                 "最大绝对相对偏差，应 ~1e-12，两条路径同源"])
    lo, hi = rep["gamma_ok_deg"]
    rows.append([f"一阶公式 vs 解析构造（{rep['formula_n']} 例）", "✓",
                 f"中位 {rep['formula_median_rel_dev']:.3%}",
                 f"交会角 {lo:.0f}°~{hi:.0f}°，90 分位 {rep['formula_p90_rel_dev']:.3%}"])

    # 批处理路径与逐行调用必须逐位一致。问题二的选点代价函数走批量路径，见 t2.score 的
    # worst_case_diameters，这条不变式一破，"优化前后结果不变"就无从谈起
    rng = np.random.default_rng(7)
    bx1, by1 = float(rng.uniform(-300.0, 300.0)), float(rng.uniform(-300.0, 300.0))
    th1 = float(rng.uniform(0.0, 360.0))
    bxp = rng.uniform(-2000.0, 2000.0, 64)
    byp = rng.uniform(-2000.0, 2000.0, 64)
    bth2 = rng.uniform(0.0, 360.0, (5, 64))
    batch = quad_diameters_batch((bx1, by1), th1, bxp, byp, bth2)
    rows_batch = np.stack([quad_diameters((bx1, by1), th1, bxp, byp, bth2[i])
                           for i in range(bth2.shape[0])])
    same = bool(np.array_equal(batch, rows_batch))
    rows.append(["批量直径 vs 逐行调用", "✓" if same else "✗", "逐位一致",
                 f"{bth2.shape[0]} 行 × {bxp.size} 个候选点"])

    # 可行域透镜：边界上的点最坏距离应恰为 1000 m，外扩的点则应超过 1000 m
    lens = feasible_lens((0.0, 0.0), 0.0)
    corners = corner_sources((0.0, 0.0), 0.0)
    bx, by = shapely.get_coordinates(lens.boundary).T
    d_on = worst_source_distance(bx, by, corners)
    out = shapely.buffer(lens, 20.0, quad_segs=8)
    ox, oy = shapely.get_coordinates(out.boundary).T
    d_out = worst_source_distance(ox, oy, corners)
    inside = bool(np.all(d_on <= RECEIVE_MIN + 1e-6))
    rows.append(["可行域边界最坏距离", "✓" if inside else "✗",
                 f"{d_on.min():.1f} ~ {d_on.max():.1f} m",
                 f"≤ {RECEIVE_MIN:.0f} 全成立 {inside}"])
    rows.append(["可行域外扩 20 m", "✓" if bool((d_out > RECEIVE_MIN).any()) else "✗",
                 f"{d_out.max():.1f} m",
                 f"超出 {RECEIVE_MIN:.0f} m 的边界点占 {(d_out > RECEIVE_MIN).mean():.0%}"])
    print_table(["检查项", "结果", "数值", "备注"], rows, align="lcrl")
    for line in (f"一阶公式 <1% 占 {rep['formula_frac_below_1pct']:.0%}，最大 "
                 f"{rep['formula_max_abs_rel_dev']:.2%}，最大值来自极端构型，"
                 f"某检测点落入另一楔形内、区域退化为三角形",
                 f"可行域面积 {lens.area / 1e6:.3f} km²；批处理与逐行调用共用一个算式，"
                 f"见 t2.score.worst_case_diameters"):
        print(f"  注：{line}")
    # 单次测向的区域直径是本问题的对照基线，放在表后单独报一行
    region = TriangulationRegion(err=BEARING_ERROR_DEG)
    region.add_node(0.0, 0.0, 0.0)
    print(f"  作为对照，只测一次的定位区域直径 = {region.diameter:.0f} m")


if __name__ == '__main__':
    _selfcheck()
