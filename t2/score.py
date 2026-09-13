"""问题二的求解算法：源不确定集、可测性保证、最坏定位直径与候选区域"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import shapely
from shapely import Polygon

from common.geometry import bearing
from t1 import TriangulationRegion
from t2 import config as cfg
from t2 import theory
from t2.region import (corner_sources, exact_region, feasible_lens, quad_diameters_batch,
                             quad_polygon, verify_analytic, worst_case_scenario,
                             worst_diameters_all, worst_source_distance)

__all__ = ["Band", "SolveResult", "source_samples", "delta2_grid", "worst_case_diameters",
           "suitability", "coarse_fields", "refine_best", "candidate_band", "solve",
           "scenario_quad"]


# 批量喂给 quad_diameters_batch 的"组合 × 候选点"元素个数上限。它不动任何数值，批内每行与
# 逐组合调用逐元素同值，只决定一次批处理多大。而这个大小很影响速度：批太大时中间数组超出 CPU
# 缓存，`in_wedge` 的元素处理反而变慢。本机实测，完整跑一遍 `python -m t2`，16 核无其他负载：
#   元素上限  150 000 → 16.5 s；20 000 → 12.9 s；6 000 → 12.3 s；3 000 → 12.4 s；1 500 → 12.8 s
# 取 6 000，约等于细网格 700 多个候选点时 8 行、粗网格 41 个点时 146 行，既在缓存友好的平台区
# 里，也不至于把 Python 层调用次数放得太大。
_COMBO_CHUNK_ELEMS = 6_000


# ----------------------------------------------------------------------------
# ① 源不确定集与第二次测向误差的离散
# ----------------------------------------------------------------------------

def source_samples(site: Sequence[float], theta1: float, d_lo: float = cfg.D_LO,
                   d_hi: float = cfg.D_HI, n_d: int = cfg.D_SAMPLES,
                   n_delta: int = cfg.DELTA_SAMPLES,
                   err_deg: float = cfg.BEARING_ERROR_DEG,
                   radius: float = cfg.REGION_RADIUS) -> np.ndarray:
    """源不确定集 C 的采样点，形状 (M, 2)，圆域之外的样本已剔除"""
    # 两端都取上：最坏情况出现在端点上，取最坏时拿到的才是真实极值候选
    ds = np.linspace(float(d_lo), float(d_hi), int(n_d))
    deltas = np.linspace(-err_deg, err_deg, int(n_delta))
    pts = []
    for d in ds:
        for sd in deltas:
            a = math.radians(theta1 + sd)
            x, y = float(site[0]) + d * math.cos(a), float(site[1]) + d * math.sin(a)
            if math.hypot(x, y) <= radius + 1e-9:
                pts.append((x, y))
    return np.asarray(pts, dtype=float)


def delta2_grid(n: int = cfg.DELTA2_SAMPLES,
                err_deg: float = cfg.BEARING_ERROR_DEG) -> np.ndarray:
    """第二次测向误差 δ₂ 的采样，含 ±1° 两端"""
    return np.linspace(-err_deg, err_deg, int(n))


def worst_case_diameters(site: Sequence[float], sx: np.ndarray, sy: np.ndarray, theta1: float,
                         sources: np.ndarray, deltas2: np.ndarray,
                         err_deg: float = cfg.BEARING_ERROR_DEG,
                         reduce: str = "max") -> np.ndarray:
    """对源采样点与第二次测向误差取最坏或期望的定位直径 J(S2) / m，逐候选点向量化"""
    # reduce="max" 是本文的 minimax 口径，"mean" 是文献常用的期望口径
    sx = np.asarray(sx, dtype=float).ravel()
    sy = np.asarray(sy, dtype=float).ravel()
    if reduce not in ("max", "mean"):
        raise ValueError(f"reduce 只能是 'max' 或 'mean'，收到 {reduce!r}")
    src = np.asarray(sources, dtype=float)
    d2s = np.asarray(deltas2, dtype=float)
    # 行序就是原先双重循环的顺序，源采样点在外、δ₂ 在内，保证 mean 的逐行累加与原先同序
    w2 = np.degrees(np.arctan2(src[:, 1][:, None] - sy[None, :],
                               src[:, 0][:, None] - sx[None, :])) % 360.0          # (M, N)
    combos = (w2[:, None, :] + d2s[None, :, None]).reshape(-1, sx.size)           # (M·K, N)
    rows = int(_COMBO_CHUNK_ELEMS // max(sx.size, 1))
    rows = max(1, min(rows, combos.shape[0]))
    if reduce == "mean":
        acc = np.zeros(sx.shape, dtype=float)
        for i in range(0, combos.shape[0], rows):
            block = quad_diameters_batch(site, theta1, sx, sy, combos[i:i + rows], err_deg)
            for r in range(block.shape[0]):        # 逐行累加：与原先的求和顺序逐位一致
                acc += block[r]
        return acc / max(combos.shape[0], 1)
    best = np.zeros(sx.shape, dtype=float)
    for i in range(0, combos.shape[0], rows):
        block = quad_diameters_batch(site, theta1, sx, sy, combos[i:i + rows], err_deg)
        best = np.maximum(best, block.max(axis=0))     # 取最大：与逐组合取最大逐位同值，无非结合性
    return best


def suitability(j: np.ndarray, j_star: float) -> np.ndarray:
    """适合度 F = J*/J，取值 (0, 1]，哨兵值按 0 处理"""
    j = np.asarray(j, dtype=float)
    safe = np.where(j > 0.0, j, np.inf)
    f = float(j_star) / safe
    return np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0), 0.0, 1.0)


# ----------------------------------------------------------------------------
# ② 全域粗网格与局部细化
# ----------------------------------------------------------------------------

def coarse_fields(site: Sequence[float], theta1: float, sources: np.ndarray,
                  deltas2: np.ndarray, step: float = cfg.COARSE_STEP,
                  err_deg: float = cfg.BEARING_ERROR_DEG,
                  radius: float = cfg.REGION_RADIUS,
                  receive_min: float = cfg.RECEIVE_MIN) -> dict[str, np.ndarray]:
    """全域粗网格上的坐标网格、最坏源距、可测性掩码与最坏定位直径四个场"""
    # 只在可行点上算 J，其余记哨兵值：可行域只是圆域里很小的一块透镜，能省九成计算
    axis = np.arange(-radius, radius + 0.5 * step, step)
    xx, yy = np.meshgrid(axis, axis)
    in_region = np.hypot(xx, yy) <= radius + 1e-9
    hear = worst_source_distance(xx.ravel(), yy.ravel(), sources).reshape(xx.shape)
    feasible = in_region & (hear <= receive_min)
    j = np.full(xx.shape, cfg.DIAM_SENTINEL, dtype=float)
    if feasible.any():
        j[feasible] = worst_case_diameters(site, xx[feasible], yy[feasible], theta1, sources,
                                           deltas2, err_deg)
    return {"x": xx, "y": yy, "hear": hear, "feasible": feasible, "j": j}


# ----------------------------------------------------------------------------
# ②b 文献判据层：全域快筛、判据对照与全局认证，代码在 t2/theory.py
# ----------------------------------------------------------------------------

def closed_form_table(d: float, r2: float, gamma_deg: float, exact_radius: float,
                      err_deg: float = cfg.BEARING_ERROR_DEG) -> dict[str, Any]:
    """在最坏情形几何上对照本文闭式、三条文献判据与精确最坏半径"""
    closed = {
        "minimax_radius_m": theory.minimax_radius(d, r2, gamma_deg, err_deg),
        "gdop_m": theory.gdop(d, r2, gamma_deg, err_deg),
        "crlb_major_m": theory.crlb_major(d, r2, gamma_deg, err_deg),
        "crlb_area_m2": theory.crlb_area(d, r2, gamma_deg, err_deg),
        "foy_rmec_m": theory.foy_rmec(d, r2, gamma_deg, err_deg),
    }
    # 面积以 m² 计，半径以 m 计，两者不比相对偏差，面积只列绝对值
    dev = {k: (v - exact_radius) / exact_radius for k, v in closed.items()
           if k != "crlb_area_m2" and math.isfinite(v)}
    return {"worst_case_geometry": {"d_m": float(d), "r2_m": float(r2),
                                    "gamma_deg": float(gamma_deg),
                                    "exact_worst_radius_m": float(exact_radius)},
            "closed_form_m": closed, "closed_form_rel_dev": dev}


def theory_analysis(site: Sequence[float], theta1: float, sources: np.ndarray,
                    deltas2: np.ndarray, best: tuple[float, float], j_star: float,
                    d_lo: float, d_hi: float, eta: float,
                    err_deg: float = cfg.BEARING_ERROR_DEG,
                    radius: float = cfg.REGION_RADIUS,
                    receive_min: float = cfg.RECEIVE_MIN,
                    step: float = cfg.CERTIFY_STEP, top: int = cfg.CERTIFY_TOP,
                    probe_n: int = cfg.PROBE_N,
                    expect_step: float = cfg.EXPECT_STEP,
                    certify_max: int = cfg.CERTIFY_MAX,
                    expect_max: int = cfg.EXPECT_MAX) -> dict[str, Any]:
    """把文献判据接到本文算法上，做全域认证、判据一致性与准则对照三件事"""
    lens = feasible_lens(site, theta1, d_lo, d_hi, err_deg, receive_min, radius)
    x0, y0, x1, y1 = lens.bounds
    # 可行域大时，比如换了第一检测点，5 m 网格会涨到几十万点，那就按点数上限自动放宽步长。
    # 缺省情形的步长不受影响，17 861 点远在上限以下；只在必要时变粗，并如实记进结果。
    step = max(float(step), math.sqrt(max((x1 - x0) * (y1 - y0), 1e-9) / float(certify_max)))
    expect_step = max(float(expect_step),
                      math.sqrt(max((x1 - x0) * (y1 - y0), 1e-9) / float(expect_max)))
    xs = np.arange(x0, x1 + 0.5 * step, step)
    ys = np.arange(y0, y1 + 0.5 * step, step)
    xx, yy = np.meshgrid(xs, ys)
    px, py = xx.ravel(), yy.ravel()
    in_region = np.hypot(px, py) <= radius + 1e-9
    feasible = in_region & (worst_source_distance(px, py, sources) <= receive_min)
    pts = np.column_stack([px[feasible], py[feasible]])
    gdop_field = theory.radius_field(site, theta1, pts, sources, mode="gdop", err_deg=err_deg)

    # ① 全域认证：GDOP 最小的一批候选 + 当前最优点 → 精确复核
    order = np.argsort(gdop_field)
    cand = np.vstack([pts[order[:int(top)]], np.asarray([best], dtype=float)])
    j_cand = worst_case_diameters(site, cand[:, 0], cand[:, 1], theta1, sources, deltas2, err_deg)
    k = int(np.argmin(j_cand))
    # 快筛出来的点落在 5 m 网格上，再按 REFINE_STEP 细化一次，才是能与粗搜解比较的"认证解"
    cert = refine_best(site, theta1, sources, deltas2, float(cand[k, 0]), float(cand[k, 1]),
                       half=step, step=cfg.REFINE_STEP, err_deg=err_deg,
                       receive_min=receive_min)

    # ② 判据对照：秩相关 + 三条文献判据在最坏情形上的偏差
    sub = pts[:: max(1, pts.shape[0] // int(probe_n))]
    j_sub = worst_case_diameters(site, sub[:, 0], sub[:, 1], theta1, sources, deltas2, err_deg)
    g_sub = theory.radius_field(site, theta1, sub, sources, mode="gdop", err_deg=err_deg)
    rho = theory.spearman(g_sub, j_sub)
    sc = worst_case_scenario(site, theta1, best, sources, deltas2, err_deg)
    cf = closed_form_table(float(sc["d_m"]), float(sc["r2_m"]), float(sc["gamma_deg"]),
                           0.5 * float(sc["diameter"]), err_deg)

    # ③ 准则对照：GDOP 判据最优点、期望口径最优点
    p_gdop = (float(pts[order[0], 0]), float(pts[order[0], 1]))
    ex_xs = np.arange(x0, x1 + 0.5 * expect_step, expect_step)
    ex_ys = np.arange(y0, y1 + 0.5 * expect_step, expect_step)
    exx, exy = np.meshgrid(ex_xs, ex_ys)
    epx, epy = exx.ravel(), exy.ravel()
    efeas = (np.hypot(epx, epy) <= radius + 1e-9) & (
        worst_source_distance(epx, epy, sources) <= receive_min)
    epts = np.column_stack([epx[efeas], epy[efeas]])
    mean_field = worst_case_diameters(site, epts[:, 0], epts[:, 1], theta1, sources, deltas2,
                                      err_deg, reduce="mean")
    ke = int(np.argmin(mean_field))
    p_exp = refine_best(site, theta1, sources, deltas2, float(epts[ke, 0]), float(epts[ke, 1]),
                        half=expect_step, step=cfg.REFINE_STEP, err_deg=err_deg,
                        receive_min=receive_min, reduce="mean")
    return {
        "certify": {"step_m": float(step), "expect_step_m": float(expect_step),
                    "n_grid": int(pts.shape[0]), "n_exact": int(cand.shape[0]),
                    "xy_m": [cert[0], cert[1]], "worst_diam_m": cert[2],
                    # 解关于 θ1 方向严格镜像对称，比距离时取到粗解或其镜像的较小者
                    "vs_coarse_m": min(math.hypot(cert[0] - float(best[0]), cert[1] - float(best[1])),
                                       math.hypot(cert[0] - float(best[0]), cert[1] + float(best[1]))),
                    "vs_coarse_j_m": cert[2] - float(j_star),
                    "better_than_coarse_m": float(j_star) - cert[2]},
        "probe": {"gdop_m": g_sub, "worst_diam_m": j_sub},
        "criteria": {"n_probe": int(sub.shape[0]), "gdop_vs_exact_spearman": rho, **cf},
        "points_xy": {"gdop": [p_gdop[0], p_gdop[1]], "expected": [p_exp[0], p_exp[1]]},
        "citations": theory.CITATIONS,
    }


def criteria_points(site: Sequence[float], theta1: float, sources: np.ndarray,
                    deltas2: np.ndarray, points_xy: dict[str, tuple[float, float]],
                    j_star: float, mirror: bool = True,
                    err_deg: float = cfg.BEARING_ERROR_DEG) -> dict[str, Any]:
    """三个口径的最优点在同一张表上互评，指标是最坏直径、期望直径与 GDOP"""
    ang = math.radians(float(theta1))
    table: dict[str, Any] = {}
    for name, (bx, by) in points_xy.items():
        phi = _rel_angle(bearing(site, (bx, by)), theta1)
        if mirror and phi < 0.0:
            bx = float(site[0]) + math.cos(ang) * (float(bx) - float(site[0])) \
                 + math.sin(ang) * (float(by) - float(site[1]))
            by = float(site[1]) + math.sin(ang) * (float(bx) - float(site[0])) \
                 - math.cos(ang) * (float(by) - float(site[1]))
            phi = _rel_angle(bearing(site, (bx, by)), theta1)
        px, py = np.asarray([float(bx)]), np.asarray([float(by)])
        jj = float(worst_case_diameters(site, px, py, theta1, sources, deltas2, err_deg)[0])
        jm = float(worst_case_diameters(site, px, py, theta1, sources, deltas2, err_deg,
                                        reduce="mean")[0])
        gg = float(theory.radius_field(site, theta1, np.column_stack([px, py]), sources,
                                       mode="gdop", err_deg=err_deg)[0])
        table[name] = {
            "xy_m": [float(bx), float(by)],
            "r_m": math.hypot(float(bx) - float(site[0]), float(by) - float(site[1])),
            "phi_deg": phi,
            "worst_diam_m": jj,
            "worst_diam_loss_pct": 100.0 * (jj - float(j_star)) / float(j_star),
            "mean_diam_m": jm,
            "gdop_m": gg,
        }
    return table


def refine_best(site: Sequence[float], theta1: float, sources: np.ndarray, deltas2: np.ndarray,
                x0: float, y0: float, half: float = cfg.REFINE_HALF,
                step: float = cfg.REFINE_STEP, err_deg: float = cfg.BEARING_ERROR_DEG,
                receive_min: float = cfg.RECEIVE_MIN,
                reduce: str = "max") -> tuple[float, float, float]:
    """在粗解附近做局部细化，返回 (x, y, J)"""
    xs = np.arange(x0 - half, x0 + half + 0.5 * step, step)
    ys = np.arange(y0 - half, y0 + half + 0.5 * step, step)
    xx, yy = np.meshgrid(xs, ys)
    px, py = xx.ravel(), yy.ravel()
    feasible = worst_source_distance(px, py, sources) <= receive_min
    if not feasible.any():                       # 理论上不会发生，x0, y0 本就可行的
        return float(x0), float(y0), float("nan")
    j = worst_case_diameters(site, px[feasible], py[feasible], theta1, sources, deltas2, err_deg,
                             reduce=reduce)
    k = int(np.argmin(j))
    return float(px[feasible][k]), float(py[feasible][k]), float(j[k])


# ----------------------------------------------------------------------------
# ③ 候选区域：弧带
# ----------------------------------------------------------------------------

@dataclass
class Band:
    """候选区域的解析描述：每瓣一条径向弧带，外加多边形边界与面积"""

    level_m: float                                  # 阈值 J ≤ level_m
    lobes: list[dict[str, Any]] = field(default_factory=list)
    area_m2: float = 0.0
    radial_gaps: int = 0
    fine: dict[str, np.ndarray] = field(default_factory=dict)   # 极坐标细网格场，放大图要用

    @property
    def r_lo(self) -> float:
        """候选区域所有瓣的径向区间下界里最小的那个 / m，无瓣时为 nan"""
        return min(l["r_lo"] for l in self.lobes) if self.lobes else float("nan")

    @property
    def r_hi(self) -> float:
        """候选区域所有瓣的径向区间上界里最大的那个 / m，无瓣时为 nan"""
        return max(l["r_hi"] for l in self.lobes) if self.lobes else float("nan")

    def describe(self) -> str:
        """一句话描述，控制台与报告共用：极径范围，加上每瓣的方位范围"""
        if not self.lobes:
            return "空"
        parts = [f"r ∈ [{self.r_lo:.0f}, {self.r_hi:.0f}] m"]
        for i, l in enumerate(self.lobes, 1):
            parts.append(f"第 {i} 瓣方位差 ∈ [{l['phi_lo']:+.1f}°, {l['phi_hi']:+.1f}°]")
        parts.append(f"合计面积 {self.area_m2 / 1e6:.3f} km²")
        return "；".join(parts)


def lens_ray_interval(site: Sequence[float], theta1: float, phi_deg: np.ndarray, d_lo: float,
                      d_hi: float, err_deg: float, receive_min: float,
                      radius: float) -> tuple[np.ndarray, np.ndarray]:
    """可行域透镜沿某个方位绕 S1 的精确径向区间 (r_in, r_out) / m，无交时 r_out 为 -inf"""
    phi = np.asarray(phi_deg, dtype=float)
    ux, uy = np.cos(np.radians(theta1 + phi)), np.sin(np.radians(theta1 + phi))
    r_in = np.zeros_like(phi)
    r_out = np.full_like(phi, np.inf)
    for cx, cy in np.vstack([corner_sources(site, theta1, d_lo, d_hi, err_deg), [site]]):
        rr = radius if (float(cx) == float(site[0]) and float(cy) == float(site[1])) else receive_min
        wx, wy = float(cx) - float(site[0]), float(cy) - float(site[1])
        proj = ux * wx + uy * wy
        disc = np.maximum(proj * proj + rr * rr - (wx * wx + wy * wy), 0.0)
        hit = proj * proj + rr * rr - (wx * wx + wy * wy) >= 0.0
        r_in = np.where(hit, np.maximum(r_in, proj - np.sqrt(disc)), r_in)
        r_out = np.where(hit, np.minimum(r_out, proj + np.sqrt(disc)), -np.inf)
    return r_in, r_out


def candidate_band(site: Sequence[float], theta1: float, sources: np.ndarray, deltas2: np.ndarray,
                   level: float, r_center: float, d_lo: float = cfg.D_LO, d_hi: float = cfg.D_HI,
                   err_deg: float = cfg.BEARING_ERROR_DEG,
                   receive_min: float = cfg.RECEIVE_MIN,
                   radius: float = cfg.REGION_RADIUS) -> Band:
    """在绕 S1 的极坐标网格上提取可行且 J ≤ level 的弧带，按 φ 的正负分瓣"""
    # 外弧取网格外缘与透镜精确边界的较小者，内弧外扩半格，最优点恰在外缘上也能落进带内
    phis = np.arange(-cfg.BAND_PHI_SPAN, cfg.BAND_PHI_SPAN + 0.5 * cfg.BAND_PHI_STEP,
                     cfg.BAND_PHI_STEP)
    # 径向网格按透镜的精确径向范围设定，留半步余量：放大图必须覆盖整个可行域，
    # 只按粗解 ±BAND_R_SPAN 取会漏掉透镜内侧，本算例里内缘在 500 m，外缘在 1005 m。
    lim_in, lim_out = lens_ray_interval(site, theta1, phis, d_lo, d_hi, err_deg, receive_min, radius)
    alive = np.isfinite(lim_out) & (lim_out > lim_in)      # 该方位上透镜确实存在
    r_lo = max(0.0, float(lim_in[alive].min()) - cfg.BAND_R_STEP) if alive.any() else 0.0
    r_hi = float(lim_out[alive].max()) + cfg.BAND_R_STEP if alive.any() else 0.0
    r = np.arange(r_lo, r_hi + 0.5 * cfg.BAND_R_STEP, cfg.BAND_R_STEP)
    rr, pp = np.meshgrid(r, phis, indexing="ij")
    ang = np.radians(theta1 + pp)
    px = float(site[0]) + rr * np.cos(ang)
    py = float(site[1]) + rr * np.sin(ang)
    feasible = worst_source_distance(px.ravel(), py.ravel(),
                                     sources).reshape(rr.shape) <= receive_min
    j = np.full(rr.shape, cfg.DIAM_SENTINEL, dtype=float)
    if feasible.any():
        j[feasible] = worst_case_diameters(site, px[feasible], py[feasible], theta1, sources,
                                           deltas2, err_deg)
    sel = feasible & (j <= level)
    # 每个方位角的精确透镜径向区间，弧带内外缘都可能被它限制；顺便给出半格外扩量
    half_r = 0.5 * cfg.BAND_R_STEP

    band = Band(level_m=float(level))
    for want_pos in (True, False):                            # 正/负方位各成一瓣
        cols = [k for k in range(phis.size) if (phis[k] > 0) == want_pos and phis[k] != 0.0]
        run: list[int] = []
        for k in cols:
            idx = np.flatnonzero(sel[:, k])
            if idx.size == 0:
                if run:
                    band.lobes.append(_lobe(r, phis, run, sel, feasible, theta1, site, lim_in,
                                            lim_out, half_r))
                    run = []
                continue
            if np.any(np.diff(idx) > 1):                      # 同一方位上两段：取包络并计数
                band.radial_gaps += 1
            run.append(k)
        if run:
            band.lobes.append(_lobe(r, phis, run, sel, feasible, theta1, site, lim_in, lim_out,
                                    half_r))
    band.area_m2 = float(sum(Polygon(l["boundary"]).area for l in band.lobes))
    band.lobes.sort(key=lambda l: -l["phi_hi"])               # 确定性顺序：先正后负
    band.fine = {"x": px, "y": py, "j": j, "feasible": feasible, "sel": sel,
                 "r": rr, "phi_deg": pp}
    return band


def _lobe(r: np.ndarray, phis: np.ndarray, cols: list[int], sel: np.ndarray,
          feasible: np.ndarray, theta1: float, site: Sequence[float], lim_in: np.ndarray,
          lim_out: np.ndarray, half_r: float) -> dict[str, Any]:
    """把一串连续方位角上的径向区间组装成一瓣弧带，外弧正序内弧逆序"""
    outer, inner = [], []
    for k in cols:
        idx = np.flatnonzero(sel[:, k])
        i, j = int(idx[-1]), int(idx[0])
        r_out = (float(lim_out[k]) if (i + 1 >= r.size or not feasible[i + 1, k])
                 else min(r[i] + half_r, float(lim_out[k])))
        r_in = (float(lim_in[k]) if (j - 1 < 0 or not feasible[j - 1, k])
                else max(r[j] - half_r, float(lim_in[k])))
        outer.append((r_out, phis[k]))
        inner.append((r_in, phis[k]))
    r_out = np.array([o[0] for o in outer])
    r_in = np.array([i[0] for i in inner])
    ph_out = np.array([o[1] for o in outer])
    boundary = np.vstack([_polar_to_xy(r_out, ph_out, theta1, site),
                          _polar_to_xy(r_in[::-1], ph_out[::-1], theta1, site)])
    return {"phi_lo": float(ph_out.min()), "phi_hi": float(ph_out.max()),
            "r_lo": float(r_in.min()), "r_hi": float(r_out.max()),
            "n_cols": int(len(cols)), "boundary": boundary,
            "area_m2": float(Polygon(boundary).area)}


def _polar_to_xy(r: np.ndarray, phi_deg: np.ndarray, theta1: float,
                 site: Sequence[float]) -> np.ndarray:
    """绕 S1 的极坐标转直角坐标，φ 相对示向度，返回形状 (N, 2)"""
    a = np.radians(theta1 + phi_deg)
    return np.stack([float(site[0]) + r * np.cos(a), float(site[1]) + r * np.sin(a)], axis=-1)


# ----------------------------------------------------------------------------
# ④ 总入口
# ----------------------------------------------------------------------------

@dataclass
class SolveResult:
    """问题二的完整解：最优第二检测点、最优区域即候选弧带、可行域，以及全部校验"""

    site: tuple[float, float]
    theta1: float
    j_star: float                                   # 最优最坏定位直径 / m
    best: tuple[float, float]                       # 最优第二检测点坐标 / m
    best_r: float                                   # 到 S1 的距离 / m
    best_phi: float                                 # 相对示向度的方位差 / 度
    band: Band                                      # 候选区域，弧带
    single_m: float                                 # 对照：只有一次测向时的区域直径 / m
    lens: Any                                       # 可行域多边形，shapely 对象
    lens_area_m2: float
    sources: np.ndarray
    corners: np.ndarray
    fields: dict[str, np.ndarray]
    scenario: dict[str, Any]                        # 最优点的最坏情形，论文插图要用
    checks: dict[str, Any]
    theory: dict[str, Any] = field(default_factory=dict)   # 文献判据层，见 t2/theory.py

    @property
    def improvement(self) -> float:
        """单次测向直径除以最优最坏直径，也就是第二次测向把最坏定位直径缩小了多少倍"""
        return self.single_m / self.j_star if self.j_star else float("inf")

    def summary(self) -> str:
        """控制台与报告共用的一段话小结"""
        x, y = self.best
        return (f"最优第二检测点 ({x:.0f}, {y:.0f}) m：距 S1 {self.best_r:.0f} m、"
                f"相对示向度 {self.best_phi:+.1f}°，最坏定位直径 J* = {self.j_star:.1f} m"
                f"（最坏交会角 {self.scenario['gamma_deg']:.1f}°）；候选区域 {self.band.describe()}")


def solve(site: Sequence[float] = cfg.DEFAULT_SITE, theta1: float = cfg.DEFAULT_BEARING,
          eta: float = cfg.ETA, d_lo: float = cfg.D_LO, d_hi: float = cfg.D_HI,
          err_deg: float = cfg.BEARING_ERROR_DEG, radius: float = cfg.REGION_RADIUS,
          receive_min: float = cfg.RECEIVE_MIN, verify_n: int = 400) -> SolveResult:
    """问题二的完整求解，确定性；`verify_n` > 0 时附带解析与 shapely 抽样校验"""
    sources = source_samples(site, theta1, d_lo, d_hi, err_deg=err_deg, radius=radius)
    deltas2 = delta2_grid(err_deg=err_deg)

    # ① 全域粗搜 → ② 局部细化
    fields = coarse_fields(site, theta1, sources, deltas2, err_deg=err_deg, radius=radius,
                           receive_min=receive_min)
    js = np.where(fields["feasible"], fields["j"], np.inf)
    k = int(np.argmin(js))
    x0, y0 = float(fields["x"].ravel()[k]), float(fields["y"].ravel()[k])
    bx, by, j_star = refine_best(site, theta1, sources, deltas2, x0, y0, err_deg=err_deg,
                                 receive_min=receive_min)

    # ②b 文献判据层：全域认证用 5 m 细网格 GDOP 快筛加精确复核，另有判据一致性与准则对照
    thy = theory_analysis(site, theta1, sources, deltas2, (bx, by), j_star, d_lo, d_hi, eta,
                          err_deg=err_deg, radius=radius, receive_min=receive_min)
    # ③ 两条路线取更优者：25 m 粗搜加局部细化，细网格全域快筛加局部细化
    cx, cy, cj = (float(thy["certify"]["xy_m"][0]), float(thy["certify"]["xy_m"][1]),
                  float(thy["certify"]["worst_diam_m"]))
    if cj < j_star:
        bx, by, j_star = cx, cy, cj
    phi_best = _rel_angle(bearing(site, (bx, by)), theta1)
    # ④ 解关于过 S1 的示向度方向严格镜像对称，源集、δ 网格、圆域都对称，所以 φ* < 0 时
    #    报告 +φ 那一支；镜像后 J 必须逐位一致，不一致就说明哪里不对称，直接报出来。
    mirror_rel_dev = 0.0
    if phi_best < 0.0:
        ang = math.radians(float(theta1))
        mx = site[0] + math.cos(ang) * (bx - site[0]) + math.sin(ang) * (by - site[1])
        my = site[1] + math.sin(ang) * (bx - site[0]) - math.cos(ang) * (by - site[1])
        jm = float(worst_case_diameters(site, np.asarray([mx]), np.asarray([my]), theta1, sources,
                                        deltas2, err_deg)[0])
        mirror_rel_dev = abs(jm - j_star) / max(abs(j_star), 1e-9)
        bx, by = float(mx), float(my)

    # ⑤ 候选区域
    band = candidate_band(site, theta1, sources, deltas2, j_star * (1.0 + eta),
                          r_center=math.hypot(bx - site[0], by - site[1]), d_lo=d_lo, d_hi=d_hi,
                          err_deg=err_deg, receive_min=receive_min, radius=radius)

    # ⑥ 最优点的最坏情形，连圆域截断一起判，另加各项校验
    scenario = worst_case_scenario(site, theta1, (bx, by), sources, deltas2, err_deg)
    exact = exact_region(site, theta1, (bx, by), scenario["theta2_deg"], err_deg, radius)
    dense = float(np.max(worst_diameters_all(
        site, theta1, (bx, by),
        source_samples(site, theta1, d_lo, d_hi, n_d=cfg.D_SAMPLES * cfg.DENSE_FACTOR,
                       n_delta=cfg.DELTA_SAMPLES * cfg.DENSE_FACTOR, err_deg=err_deg,
                       radius=radius),
        delta2_grid(cfg.DELTA2_SAMPLES * cfg.DENSE_FACTOR, err_deg), err_deg)))
    lens = feasible_lens(site, theta1, d_lo, d_hi, err_deg, receive_min, radius)
    single = _single_measurement_diameter(site, theta1, err_deg, radius)
    # ⑦ 闭式对照表与三点互评表都用最终解重算，theory_analysis 里那次是认证前的中间值
    thy["criteria"].update(closed_form_table(float(scenario["d_m"]), float(scenario["r2_m"]),
                                             float(scenario["gamma_deg"]),
                                             0.5 * float(scenario["diameter"]), err_deg))
    thy["points"] = criteria_points(site, theta1, sources, deltas2,
                                    {"minimax": (float(bx), float(by)),
                                     "gdop": tuple(thy["points_xy"]["gdop"]),
                                     "expected": tuple(thy["points_xy"]["expected"])},
                                    float(j_star), err_deg=err_deg)
    checks: dict[str, Any] = {
        "scenario_exact_m": float(exact.diameter),
        "scenario_analytic_m": float(scenario["diameter"]),
        "scenario_rel_dev": abs(float(exact.diameter) - float(scenario["diameter"]))
                            / max(float(exact.diameter), 1e-9),
        "scenario_clipped": not bool(exact.bounded),
        "scenario_n_vertices": len(exact.vertices),
        "discretisation_dense_m": float(dense),
        "discretisation_rel_dev": abs(float(dense) - float(j_star)) / max(float(j_star), 1e-9),
        "band_radial_gaps": int(band.radial_gaps),
        "band_level_m": float(band.level_m),
        "feasible_inside_region": bool(shapely.within(lens, shapely.Point(0.0, 0.0)
                                                      .buffer(radius, quad_segs=256))),
        "best_hears_worst_source_m": float(worst_source_distance(np.asarray([bx]),
                                                                 np.asarray([by]), sources)[0]),
        "n_source_samples": int(sources.shape[0]),
        "d_lo": float(d_lo), "d_hi": float(d_hi), "eta": float(eta),
        "certify_worst_diam_m": float(thy["certify"]["worst_diam_m"]),
        "certify_j_rel_dev": abs(float(thy["certify"]["worst_diam_m"]) - float(j_star))
                             / max(abs(float(j_star)), 1e-9),
        "certify_vs_coarse_m": float(thy["certify"]["vs_coarse_m"]),
        "gdop_vs_exact_spearman": float(thy["criteria"]["gdop_vs_exact_spearman"]),
        "theory_closed_form_rel_dev": thy["criteria"]["closed_form_rel_dev"],
        "mirror_rel_dev": float(mirror_rel_dev),
    }
    if verify_n > 0:
        checks["analytic_vs_shapely"] = verify_analytic(n=verify_n, err_deg=err_deg, radius=radius)
        checks["theory_closed_forms"] = theory.verify_theory(max(40, verify_n // 4),
                                                             err_deg=err_deg, radius=radius)
    return SolveResult(site=(float(site[0]), float(site[1])), theta1=float(theta1),
                       theory=thy,  # 认证细节在 thy["certify"]；最优点已按上面两条路线择优
                       j_star=float(j_star), best=(bx, by),
                       best_r=math.hypot(bx - site[0], by - site[1]),
                       best_phi=_rel_angle(bearing(site, (bx, by)), theta1),
                       band=band, single_m=single, lens=lens, lens_area_m2=float(lens.area),
                       sources=sources, corners=corner_sources(site, theta1, d_lo, d_hi, err_deg),
                       fields=fields, scenario=scenario, checks=checks)


def _rel_angle(angle: float, ref: float) -> float:
    """angle 相对 ref 的方位差，取值 (−180, 180]"""
    return ((angle - ref + 180.0) % 360.0) - 180.0


def _single_measurement_diameter(site: Sequence[float], theta1: float, err_deg: float,
                                 radius: float) -> float:
    """对照量：只做这一次测向时的定位区域直径 / m"""
    region = TriangulationRegion(err=err_deg, radius=radius)
    region.add_node(float(site[0]), float(site[1]), float(theta1))
    return float(region.diameter)


def scenario_quad(result: SolveResult) -> Any:
    """最坏情形下定位区域的多边形，shapely 对象，供插图使用"""
    s = result.scenario
    return quad_polygon(result.site, result.theta1, result.best, s["theta2_deg"],
                        cfg.BEARING_ERROR_DEG)
