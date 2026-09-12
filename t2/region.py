"""问题二的几何底座：交会定位四边形（附录图 2 的红色四边形）、它的直径与第二检测点的可行域。

**为什么是四边形**。一次测向只给出"源在一条 ±1° 的楔形带里"。第二条示向度（在 S2 处测得）
再给一条 ±1° 的楔形带，两条带的**交**就是附录图 2 的定位区域：通常是一个由 4 条边界射线围成的
四边形（顶点 = 一条边界与另一楔形一条边界的交点）；当一个检测点恰好落在另一条楔形内部时，它
退化成三边形（顶点含该检测点本身，`_candidate_points` 把两类候选点一并取上再筛），源离某个
检测点很近、两条射线近同向时就会出现。问题 1 用直径度量它的"大小"，本模块沿用同一口径，故与
`t1.TriangulationRegion` 完全一致 —— 区别只是这里要**成千上万个候选点**各算一次，故
给出向量化的解析构造（`quad_diameters`），并用 shapely 精确值抽样校核（`verify_analytic`）。

**解析式（一阶展开，供论文引用）**。设源 G 处的交会角为 γ、r₁ = |S1G|、r₂ = |S2G|，则

    直径 ≈ (2ε / sin γ) · √(r₁² + r₂² + 2·r₁·r₂·|cos γ|)        （ε = 1° = 0.01745 rad）

推导：4 个顶点是两条示向度各自的 ±ε 边界射线的交点，把交点位置对"源的真实方向"做一阶展开，
位移形如 −(B·b + A·a·cos γ)·û₁ + A·a·û₁⊥（a、b 为两侧射线的角偏差，A = r₁ε/sin γ、
B = r₂ε/sin γ）；对 (a, b) ∈ {±ε}² 取两两最大距离，即得以 2A、2B 为邻边、夹角由 γ 决定的
平行四边形对角线。该式说明"交会角越接近 90°、两个检测点离源越近，四边形越小"，并给出退化
判据：γ → 0° 或 180°（两射线近共线）时直径发散 —— 正是要避开的构型。

**它的适用范围要说清楚**（否则会被读成"精确公式"）：一阶式只在小角偏差下成立，实测
（`verify_analytic`）在 γ ∈ [30°, 120°] 内中位偏差 0.08%、98% 的样本 < 1%；但**最坏情形**里
第一次测向误差 δ₁ 与第二次测向误差 δ₂ 会同向叠加，使两侧射线相对源真实方向的有效偏差达到
2ε（而不是 ±ε），这时二阶项不可忽略 —— 本模型的最优点上，一阶式给 121.5 m 而精确值是
134.1 m（低估 9.4%）。所以：论文用该式解释"为什么往远处、往侧面走"，**选点一律用精确构造**
（与 shapely 精确值逐位一致，见下），不用公式。

**校核**：`quad_diameters`（候选点交的解析构造）对 shapely+GEOS 精确求交的直径，在未被目标
圆域截断的样本上最大相对偏差 1e-14 量级 —— 即同一套几何，不是近似。被圆域截断的样本不计入
（截断只会让区域更小，本模块给出的是保守上界，选点因此偏安全）。

**第二检测点的可行域（保证可测）**。源的接收半径 R_rec ∈ [1000, 1500] 且逐源固定但未知，
第二点要"一定测得到"就必须 |S2G| ≤ 1000 = R_rec 的下界。而源的位置只知落在楔形 C 内，
故要求**对 C 中每个点**都成立。凸函数在紧集上的最大值必在极点取得，C 的极点就是它 4 个角
（近端 d = D_LO 的两个角、远端 d = D_HI 的两个角），于是

    可行域 = 圆域 ∩ D(角₁, 1000) ∩ D(角₂, 1000) ∩ D(角₃, 1000) ∩ D(角₄, 1000)

是 4 个半径 1000 m 圆盘的交（`feasible_lens` 给出它的多边形，仅用于出图与描述；判据本身
用采样点的精确最坏距离，见 `t2/score`）。

依赖 shapely>=2.1、numpy；命令行自检见文件末尾 `python -m t2.region`。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import shapely
from shapely import Point, Polygon
from shapely.geometry.base import BaseGeometry

from common.geometry import bearing
from t1 import TriangulationRegion
from t2.config import (BEARING_ERROR_DEG, D_HI, D_LO, DIAM_SENTINEL, PARALLEL_EPS,
                             RECEIVE_MIN, REGION_RADIUS)

# 楔形内判定的角度容差 / 度：只用来吸收浮点误差，让落在边界上的交点在两个楔形里都算"在内"。
# 取 1e-6° 对应的横向偏差在 4 km 尺度上不足 0.1 mm，不会把真正的域外点算进来。
WEDGE_TOL_DEG = 1e-6

__all__ = ["in_wedge", "quad_diameters", "quad_diameters_batch", "quad_polygon", "exact_region",
           "exact_diameter", "analytic_diameter", "corner_sources", "feasible_lens",
           "worst_case_scenario", "worst_source_distance", "worst_diameters_all",
           "verify_analytic", "unit"]

# 标量单位向量的小缓存：详见 unit() 的说明
_UNIT_SCALAR_CACHE: dict[float, tuple[float, float]] = {}


def _cos_sin_scalar(theta_deg: float) -> tuple[float, float]:
    """标量方位角 → (cos, sin)，带缓存（选点循环里同一批角度会被反复用到）。"""
    hit = _UNIT_SCALAR_CACHE.get(theta_deg)
    if hit is None:
        if len(_UNIT_SCALAR_CACHE) > 8192:       # 只服务"反复出现的少数角度"，不做无界增长
            _UNIT_SCALAR_CACHE.clear()
        a = np.radians(np.asarray(theta_deg, dtype=float))
        hit = (float(np.cos(a)), float(np.sin(a)))
        _UNIT_SCALAR_CACHE[theta_deg] = hit
    return hit


def _cos_sin(theta_deg: float | np.ndarray) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """方位角（度）→ (cos, sin)。标量返回一对 float（带缓存），数组返回两个同形数组。

    与 `unit` 的唯一区别是不把结果 `stack` 成 `(..., 2)`：热点里的调用方本来就分别要用 cos 与
    sin（叉积），而 `np.stack` 在大数组上约 110 ns/元素、比一次 `np.cos` 还贵。数值与 `unit`
    完全相同（同一批浮点运算，只是不装箱）。
    """
    if np.ndim(theta_deg) == 0:
        return _cos_sin_scalar(float(theta_deg))
    a = np.radians(np.asarray(theta_deg, dtype=float))
    return np.cos(a), np.sin(a)


def unit(theta_deg: float | np.ndarray) -> tuple[float, float] | np.ndarray:
    """方位角（度）对应的单位向量；标量返回 (ux, uy) 二元组，数组返回 (..., 2) 数组。

    角度约定同 `common.geometry.bearing`（x 轴正向为 0、逆时针为正）。

    标量走一层缓存：选点优化里 `in_wedge` 要被调用上万次，传进去的楔形边界角却只有少数几个
    固定值（θ₁ ± err 等），而每次 numpy 标量三角函数（radians/cos/sin 各建一个 0 维数组）要
    约 40 µs —— 缓存后只剩一次字典查找。数组入参不缓存（既无法用 float 作键，结果也可能很大）。
    缓存内容与直接计算逐位相同，故不影响任何数值。
    """
    c, s = _cos_sin(theta_deg)
    if np.ndim(theta_deg) == 0:
        return c, s
    return np.stack([c, s], axis=-1)


def in_wedge(px: np.ndarray, py: np.ndarray, apex: Sequence[float], theta_lo: float,
             theta_hi: float, tol_deg: float = 0.0) -> np.ndarray:
    """点 (px, py) 是否落在"以 apex 为顶点、方向从 theta_lo 到 theta_hi（逆时针 ≤ 180°）的楔形"内。

    用两个叉积判：点在 theta_lo 的逆时针一侧、且在 theta_hi 的顺时针一侧。tol_deg 为正时把
    边界向外放宽（数值容差），落在边界上的点算"在内"。
    """
    ax, ay = _cos_sin(theta_lo)                            # 标量给 float 对，数组给同形数组对
    bx, by = _cos_sin(theta_hi)
    vx = np.asarray(px, dtype=float) - np.asarray(apex[0], dtype=float)
    vy = np.asarray(py, dtype=float) - np.asarray(apex[1], dtype=float)
    c1 = ax * vy - ay * vx                       # cross(d_lo, v) ≥ 0：在 d_lo 逆时针侧
    c2 = bx * vy - by * vx                       # cross(d_hi, v) ≤ 0：在 d_hi 顺时针侧
    if tol_deg == 0.0:                           # 无容差：不必算模长（tol=0 时 slack 恒为 0）
        return (c1 >= 0.0) & (c2 <= 0.0)
    slack = math.sin(math.radians(tol_deg)) * _norm2(vx, vy)
    return (c1 >= -slack) & (c2 <= slack)


def _norm2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """sqrt(x² + y²)（|x|,|y| 都在 1e4 量级、离上溢下溢极远，故不必用更慢的 `np.hypot`）。"""
    return np.sqrt(x * x + y * y)


def _candidate_points_batch(site: Sequence[float], theta1: float, sx: np.ndarray, sy: np.ndarray,
                            theta2_batch: np.ndarray,
                            err_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """**一批**楔形方向下，两楔形之交的候选顶点与各自的"是否同时在两个楔形内"掩码。

    交区域的顶点只可能来自两类点：①两条边界射线的交点（4 个组合）；②某个楔形的顶点（检测点
    本身）落在另一个楔形内时，它就是交区域的一个顶点（此时交区域是三边形而不是四边形 ——
    源离一个检测点很近、两条射线近同向时就是这样）。两类都取上，再用"在两楔形内"筛掉
    延长线上的伪交点，剩下的点都落在交区域内，其两两最大距离即直径。

    `theta2_batch` 形状 `(B, N)`（B 个候选方向、N 个候选点），返回 `P` 形状 `(6, B, N, 2)`
    与 `keep` 形状 `(6, B, N)`。单点情形（`_candidate_points`）就是 B = 1 的特例：两条路径
    共用同一套算式，故"向量化前后逐元素一致"是构造上的保证，而不是巧合。

    Returns:
        tuple[np.ndarray, np.ndarray]: 候选顶点 `P`（形状 `(6, B, N, 2)`）与"该点同时落在两个
        楔形内"的掩码 `keep`（形状 `(6, B, N)`）
    """
    x1, y1 = float(site[0]), float(site[1])
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    theta2_batch = np.asarray(theta2_batch, dtype=float)
    shape = theta2_batch.shape
    wx = np.broadcast_to(sx, shape) - x1
    wy = np.broadcast_to(sy, shape) - y1
    pts = []
    # θ₂ ± err 的单位向量只算一次：原先把 (sa, sb) 四个组合各算一遍，其中两个是重复的
    bs = (_cos_sin(theta2_batch - err_deg), _cos_sin(theta2_batch + err_deg))
    for sa in (-err_deg, err_deg):
        ax, ay = unit(theta1 + sa)
        for bx, by in bs:
            den = ax * by - ay * bx
            safe = np.where(np.abs(den) < PARALLEL_EPS, np.nan, den)
            t = (wx * by - wy * bx) / safe
            pts.append((x1 + t * ax, y1 + t * ay))
    pts.append((np.full_like(theta2_batch, x1), np.full_like(theta2_batch, y1)))   # 楔形 1 的顶点
    pts.append((np.broadcast_to(sx, shape), np.broadcast_to(sy, shape)))           # 楔形 2 的顶点
    P = np.stack([np.stack([p[0], p[1]], axis=-1) for p in pts], axis=0)
    keep = (in_wedge(P[..., 0], P[..., 1], site, theta1 - err_deg, theta1 + err_deg, WEDGE_TOL_DEG)
            & in_wedge(P[..., 0], P[..., 1], (sx, sy), theta2_batch - err_deg,
                       theta2_batch + err_deg, WEDGE_TOL_DEG))
    return P, keep


def _candidate_points(site: Sequence[float], theta1: float, sx: np.ndarray, sy: np.ndarray,
                      theta2: np.ndarray, err_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """单点情形的候选顶点（`_candidate_points_batch` 的 B = 1 特例）。"""
    P, keep = _candidate_points_batch(site, theta1, sx, sy,
                                      np.asarray(theta2, dtype=float)[None, :], err_deg)
    return P[:, 0], keep[:, 0]


def _max_pair_distance(P: np.ndarray, keep: np.ndarray, sentinel: float) -> np.ndarray:
    """候选顶点两两距离的最大值（区域是凸的，故这就是直径）；`P` 形状 (6, ..., 2)。

    只用两处逐位不变的提速：索引外提（`P[i][..., 0]` 只取一次）+ `np.maximum(..., out=)` 原地
    更新，省掉每个点对一次临时数组。实测（真实最大分块 6×210×712）：21.8 → 21.5 ms/次。
    试过但**否决**的写法：掩码全为假时跳过整个点对（原式此时只写入 0.0，而 `best` 初值 0 且
    始终非负，故数学上确为空操作）—— 那多出的 `keep[i] & keep[j]` 布尔归约比省下的 hypot 更贵，
    实测 21.5 → 24.0 ms/次，故不做。

    距离仍用 `np.hypot` 算 —— 试过"先在平方距离上取最大、最后开一次方"（快约 3 倍），代价是
    距离的舍入差最后 1 位（hypot 多做一次规格化），会让 `t2_second_site.json` 里的浮点尾数变成
    另一个值；本项目的验收口径是"同一 seed 逐字节一致"，故这里选择保守的 hypot。
    """
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
    """**一批**方向 `(B, N)` 对应的定位四边形直径，返回 `(B, N)`；单位米。

    与 `quad_diameters` 逐元素同一套算式（后者是 B = 1 的特例）。存在的理由：选点代价函数要对
    "源采样点 × δ₂ 误差"的整套组合求最坏，逐组合调用一次 Python 函数时开销全在调用上（实测单次
    调用只有几百微秒的有效计算、却要 2 ms 以上），把组合按批喂进来能把这一层摊掉。
    """
    P, keep = _candidate_points_batch(site, theta1, sx, sy, np.asarray(theta2_batch, dtype=float),
                                      err_deg)
    return _max_pair_distance(P, keep, sentinel)


def quad_diameters(site: Sequence[float], theta1: float,
                   sx: np.ndarray, sy: np.ndarray, theta2: np.ndarray,
                   err_deg: float = BEARING_ERROR_DEG,
                   sentinel: float = DIAM_SENTINEL) -> np.ndarray:
    """定位四边形（两个 ±err 楔形之交）的直径，**向量化**；单位米。

    区域是凸的，故直径 = 区域内任意两点距离的最大值 = 候选顶点两两距离的最大值。两射线近
    共线（交区域退化为一条细带、面积趋零）时返回哨兵值：数值上等价于"无穷大"（无法定位），
    但保证后续统计与出图都是有限数。

    `theta2` 给 `(B, N)` 时按批处理（等价于逐行调用本函数，见 `quad_diameters_batch`）。
    """
    theta2_arr = np.asarray(theta2, dtype=float)
    if theta2_arr.ndim > 1:
        return quad_diameters_batch(site, theta1, sx, sy, theta2_arr, err_deg, sentinel)
    P, keep = _candidate_points(site, theta1, sx, sy, theta2_arr, err_deg)
    return _max_pair_distance(P, keep, sentinel)


def quad_polygon(site: Sequence[float], theta1: float, s2: Sequence[float], theta2: float,
                 err_deg: float = BEARING_ERROR_DEG) -> BaseGeometry:
    """单点情形的定位区域多边形（shapely，凸包），供论文插图与实际面积使用。"""
    sx = np.asarray([float(s2[0])])
    sy = np.asarray([float(s2[1])])
    P, keep = _candidate_points(site, theta1, sx, sy, np.asarray([float(theta2)]), err_deg)
    pts = [(float(P[k, 0, 0]), float(P[k, 0, 1])) for k in range(P.shape[0]) if bool(keep[k, 0])]
    if len(pts) < 3:
        return Polygon()
    return shapely.convex_hull(shapely.multipoints(pts))


def analytic_diameter(d: float, r2: float, gamma_deg: float,
                      err_deg: float = BEARING_ERROR_DEG) -> float:
    """解析式（模块文档中的一阶展开）给出的直径 / m。

    d = |S1G|、r2 = |S2G|、gamma_deg = 源处的交会角。仅用于论文引用与解析/数值对照，
    选点代价函数一律走 `quad_diameters`（精确到浮点）。
    """
    g = math.radians(gamma_deg)
    return (2.0 * math.radians(err_deg) / math.sin(g)) * math.sqrt(
        d * d + r2 * r2 + 2.0 * d * r2 * abs(math.cos(g)))


def exact_region(site: Sequence[float], theta1: float, s2: Sequence[float], theta2: float,
                 err_deg: float = BEARING_ERROR_DEG,
                 radius: float = REGION_RADIUS) -> TriangulationRegion:
    """用问题 1 的 `TriangulationRegion` 精确构造定位区域（含目标圆域裁剪）。"""
    region = TriangulationRegion(err=err_deg, radius=radius)
    region.add_node(float(site[0]), float(site[1]), float(theta1))
    region.add_node(float(s2[0]), float(s2[1]), float(theta2))
    return region


def exact_diameter(site: Sequence[float], theta1: float, s2: Sequence[float], theta2: float,
                   err_deg: float = BEARING_ERROR_DEG,
                   radius: float = REGION_RADIUS) -> float:
    """定位区域直径的精确值（shapely/GEOS 求交 + 顶点最远点对，含圆域裁剪）/ m。"""
    return float(exact_region(site, theta1, s2, theta2, err_deg, radius).diameter)


def corner_sources(site: Sequence[float], theta1: float, d_lo: float = D_LO,
                   d_hi: float = D_HI,
                   err_deg: float = BEARING_ERROR_DEG) -> np.ndarray:
    """源不确定集 C 的 4 个极点：(d, δ) 四个角，形状 (4, 2)。"""
    pts = []
    for d in (d_lo, d_hi):
        for sd in (-err_deg, err_deg):
            ux, uy = unit(theta1 + sd)
            pts.append((float(site[0]) + d * ux, float(site[1]) + d * uy))
    return np.asarray(pts, dtype=float)


def feasible_lens(site: Sequence[float], theta1: float, d_lo: float = D_LO, d_hi: float = D_HI,
                  err_deg: float = BEARING_ERROR_DEG, receive_min: float = RECEIVE_MIN,
                  radius: float = REGION_RADIUS, quad_segs: int = 64) -> BaseGeometry:
    """第二检测点可行域的多边形（4 个半径 receive_min 圆盘 ∩ 目标圆域）。

    仅用于出图与结果描述：判据本身是"采样源点上的最坏距离 ≤ receive_min"（见 t2.score），
    两者在采样分辨率内一致。圆盘用正 4×quad_segs 边形近似（缺省 256 边形，半径误差 < 0.01%）。
    """
    geom = Point(0.0, 0.0).buffer(radius, quad_segs=quad_segs)
    for cx, cy in corner_sources(site, theta1, d_lo, d_hi, err_deg):
        geom = geom.intersection(Point(cx, cy).buffer(receive_min, quad_segs=quad_segs))
    return geom


def worst_source_distance(px: np.ndarray, py: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """候选点 (px, py) 到"源不确定集"全部采样点的最远距离 / m（可测性判据的左边）。

    只有该值 ≤ R_rec 的下界 1000 m，才能保证第二次测向一定成功（与源的实际接收半径无关）。
    """
    px = np.asarray(px, dtype=float).ravel()
    py = np.asarray(py, dtype=float).ravel()
    src = np.asarray(sources, dtype=float)
    return np.max(np.hypot(px[:, None] - src[None, :, 0],
                           py[:, None] - src[None, :, 1]), axis=1)


def worst_diameters_all(site: Sequence[float], theta1: float, s2: Sequence[float],
                        sources: np.ndarray, deltas2: Sequence[float],
                        err_deg: float = BEARING_ERROR_DEG) -> np.ndarray:
    """单点 S2 对**全部** (源采样点 × δ₂) 组合的定位直径 / m，一次向量化算完。

    逐个组合调用 `quad_diameters` 是 Python 层循环（1.2 万次），开销全在调用上；这里把组合
    展平成一个长数组，一次算完 —— 最坏情形分析与离散化加密校验都用它，快两个数量级。
    """
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
    """单点 S2 的最坏情形：返回最大直径及其对应的源位置、第二次测向误差与定位区域多边形。

    供论文插图与结果说明使用（"最坏的那一次"到底长什么样）。实现上先把全部 (源, δ₂) 组合
    拼成一个大数组一次算完直径（与 `quad_diameters` 同一套公式），只有最后那一次最优组合才
    去构造多边形 —— 否则 O(源×δ₂) 次 shapely 调用会吃掉几秒。
    """
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
    # 最坏源相对示向度的偏差、到 S1 的距离与源处交会角（论文里要引用这三个数）
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
    """随机抽样核对两层实现与 shapely 精确值，返回可直接写进报告的校验数字。

    抽样分三类：
      * **被目标圆域截断**（`bounded` 为假）：四边形越大越容易被圆域切掉，解析构造不知道圆域，
        必然 ≥ 精确值（保守）。这类样本不计偏差，只报个数；
      * **近共线**（源处交会角 |sin γ| < sin_gamma_min）：四边形病态、一阶展开失效，单独计数；
      * **正常样本**：核对两件事 ——
        ① `quad_diameters`（候选点交的解析构造）对 shapely+GEOS 精确值：应逐位一致（残差 ~1e-12）；
        ② 一阶展开式 `analytic_diameter` 对解析构造：交会角落在 gamma_ok_deg 内时中位偏差
           应在 1‰ 量级、绝大多数 < 1%（它是趋势解释，不是选点依据，见模块开头）。

    "①逐位一致、②只在交会角两端才失准"就是本模块把解析构造当作选点代价函数的依据；选点用
    精确构造，论文里的公式只是它的直观解释与灵敏度量纲。实测偏差随交会角的变化（见报告）：
    中位偏差 < 1‰，两者相差大的只有"近共线"与"检测点落入另一楔形内"两类极端构型 —— 正是
    选点要避开的构型，二者不冲突。
    """
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
    """自检：解析构造/一阶公式 vs shapely 精确值；可行域多边形与可测性判据一致。"""
    print("问题二几何自检")
    rep = verify_analytic(n=n)
    print(f"  样本 {rep['n']}：有效 {rep['n_used']}，被圆域截断跳过 {rep['n_clipped']}，"
          f"近共线跳过 {rep['n_degenerate']}")
    print(f"  ① 解析构造 vs shapely 精确值：最大绝对相对偏差 "
          f"{rep['impl_max_abs_rel_dev']:.3e}（应 ~1e-12，即同一套几何）")
    lo, hi = rep["gamma_ok_deg"]
    print(f"  ② 一阶公式 vs 解析构造（交会角 ∈ [{lo:.0f}°, {hi:.0f}°]，{rep['formula_n']} 例）："
          f"中位 {rep['formula_median_rel_dev']:.3%}，90 分位 {rep['formula_p90_rel_dev']:.3%}，"
          f"<1% 占 {rep['formula_frac_below_1pct']:.0%}，最大 {rep['formula_max_abs_rel_dev']:.2%}"
          f"（极端构型：某检测点落入另一楔形内、区域退化为三角形）")

    # 批处理路径与逐行调用必须逐位一致 —— 问题二的选点代价函数走批量路径（见 t2.score
    # 的 worst_case_diameters），这条不变式一旦破了，"优化前后结果不变"就无从谈起
    rng = np.random.default_rng(7)
    bx1, by1 = float(rng.uniform(-300.0, 300.0)), float(rng.uniform(-300.0, 300.0))
    th1 = float(rng.uniform(0.0, 360.0))
    bxp = rng.uniform(-2000.0, 2000.0, 64)
    byp = rng.uniform(-2000.0, 2000.0, 64)
    bth2 = rng.uniform(0.0, 360.0, (5, 64))
    batch = quad_diameters_batch((bx1, by1), th1, bxp, byp, bth2)
    rows = np.stack([quad_diameters((bx1, by1), th1, bxp, byp, bth2[i]) for i in range(bth2.shape[0])])
    print(f"  批量直径 vs 逐行调用：逐位一致 {bool(np.array_equal(batch, rows))}"
          f"（{bth2.shape[0]} 行 × {bxp.size} 个候选点）")

    # 可行域透镜：边界上的点最坏距离应恰为 1000 m，外扩的点则应超过 1000 m
    lens = feasible_lens((0.0, 0.0), 0.0)
    corners = corner_sources((0.0, 0.0), 0.0)
    bx, by = shapely.get_coordinates(lens.boundary).T
    d_on = worst_source_distance(bx, by, corners)
    out = shapely.buffer(lens, 20.0, quad_segs=8)
    ox, oy = shapely.get_coordinates(out.boundary).T
    d_out = worst_source_distance(ox, oy, corners)
    print(f"  可行域：边界最坏距离 ∈ [{d_on.min():.1f}, {d_on.max():.1f}] m"
          f"（≤1000 全成立 {bool(np.all(d_on <= RECEIVE_MIN + 1e-6))}）；外扩 20 m 后最坏距离"
          f"最大 {d_out.max():.1f} m（超出 1000 m 的边界点占 {(d_out > RECEIVE_MIN).mean():.0%}）；"
          f"面积 {lens.area / 1e6:.3f} km²")
    region = TriangulationRegion(err=BEARING_ERROR_DEG)
    region.add_node(0.0, 0.0, 0.0)
    print(f"  （对照）只测一次的定位区域直径 = {region.diameter:.0f} m")


if __name__ == '__main__':
    _selfcheck()
