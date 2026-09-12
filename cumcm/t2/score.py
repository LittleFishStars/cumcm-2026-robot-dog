"""问题二的求解算法：源不确定集 → 可测性保证 → 最坏定位直径 → 最优区域（候选区域）。

三个环节，对应题面"给出第二个检测点的选择策略……并给出第二个检测点的候选区域"：

1. **源不确定集**（`source_samples`）。只测了一次示向度，源的位置只知落在楔形
   C = { d ∈ [D_LO, D_HI]、相对示向度偏离 |δ| ≤ 1° } ∩ 目标圆域内。D_HI = 1500 m 来自
   "能被 S1 听到"，D_LO = 5 m 来自附录 2(9)（近距测不到示向度）。C 是**连续**的，算法用
   d×δ 的确定性网格离散它，并包含 4 个角点（`corner_sources`）——最坏情形就出现在角上。

2. **可测性保证**（`coarse_fields` 里的 feasible）。第二点要测到源须 |S2G| ≤ R_rec；
   R_rec ∈ [1000, 1500] 逐源固定但未知，故取最保守的 1000 m，并要求**对 C 中每个采样点**都
   成立。由此得到的可行域是"以 S1、以及楔形远端的两个角为心、半径 1000 m 的圆盘之交"，是
   圆域内很小的一块透镜（面积 0.45 km²，约占圆域的 4e-5）。落在透镜外的第二点，一旦源比
   预想更近就什么都测不到，第二次测向白跑。

3. **最坏定位直径与最优区域**（`worst_case_diameters` / `solve`）。选定 S2 后，两次测向把源
   约束在一个四边形（附录图 2）内，用它的直径度量定位效果（与问题 1 同口径）。源在 C 内的
   位置与第二次测向误差 δ₂ ∈ [−1°, 1°] 都未知，故取**最坏情况**：

       J(S2) = max_{G ∈ C, |δ₂| ≤ 1°} 直径( 楔形(S1, θ1) ∩ 楔形(S2, θ₂+δ₂) ∩ 圆域 )

   于是问题二化为"在可行域上最小化 J"。适合度 F(S2) = J*/J(S2) ∈ (0,1]（1 = 最优），
   候选区域 = { F ≥ 1/(1+η) }（缺省 η = 10%，即 J 不超过最优值的 1.1 倍）。

**算法结构**（全部确定性，无随机数）：
   ① 全域粗搜：直角网格（缺省 25 m）算可测性与 J，取可行域内最小点；
   ② 局部细化：以 ① 的解为中心、±80 m、步长 2 m 的直角网格再搜（J 在最优点附近光滑，
      25 m 粗网格必然把最优圈进窗口）；
   ③ 最优区域：绕 S1 的极坐标网格上取"可行 ∩ J ≤ (1+η)J*"的点，逐方位角求径向区间并组装成
      弧带多边形（解关于示向度方向镜像对称，故通常是两瓣）；
   ④ 校验：解析构造 vs shapely 精确直径、最坏情形用 shapely 复算、离散化加密复算、圆域截断
      与弧带连续性，全部写进 `checks`（缺省还做一次抽样校验，见 `solve` 的 verify_n）。

命令行入口见 `cumcm/t2/cli.py`（顶层 `T2.py`）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import shapely
from shapely import Polygon

from cumcm.common.geometry import bearing
from cumcm.t1 import TriangulationRegion
from cumcm.t2 import config as cfg
from cumcm.t2.region import (corner_sources, exact_region, feasible_lens, quad_diameters,
                             quad_polygon, verify_analytic, worst_case_scenario,
                             worst_diameters_all, worst_source_distance)

__all__ = ["Band", "SolveResult", "source_samples", "delta2_grid", "worst_case_diameters",
           "suitability", "coarse_fields", "refine_best", "candidate_band", "solve",
           "scenario_quad"]


# ----------------------------------------------------------------------------
# ① 源不确定集与第二次测向误差的离散
# ----------------------------------------------------------------------------

def source_samples(site: Sequence[float], theta1: float, d_lo: float = cfg.D_LO,
                   d_hi: float = cfg.D_HI, n_d: int = cfg.D_SAMPLES,
                   n_delta: int = cfg.DELTA_SAMPLES,
                   err_deg: float = cfg.BEARING_ERROR_DEG,
                   radius: float = cfg.REGION_RADIUS) -> np.ndarray:
    """源不确定集 C 的采样点（形状 (M, 2)），已剔除落在目标圆域之外的样本。

    d 取 [d_lo, d_hi] 的等距网格、δ 取 [−1°, 1°] 的等距网格，两端都包含（最坏情况常出现在
    端点：最远的源 + 楔形边缘），保证取最坏时拿到的是真实的极值候选。
    """
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
    """第二次测向误差 δ₂ 的采样（含 ±1° 两端）。"""
    return np.linspace(-err_deg, err_deg, int(n))


def worst_case_diameters(site: Sequence[float], sx: np.ndarray, sy: np.ndarray, theta1: float,
                         sources: np.ndarray, deltas2: np.ndarray,
                         err_deg: float = cfg.BEARING_ERROR_DEG) -> np.ndarray:
    """J(S2)：对源采样点与第二次测向误差取最坏的定位直径 / m（向量化，逐候选点）。

    对每个候选点 S2 与每个源采样点 G，第二次示向度的名义值是 θ₂ = ∠(S2→G)；实际读数再叠加
    误差 δ₂，故把 θ₂ + δ₂ 代进四边形构造取最大。源采样点已含 δ₁ ∈ [−1°, 1°]（源在楔形内的
    角向不确定性），于是这一层 max 覆盖了"源在楔形内何处 + 第二次测向偏多少"的全部组合。
    """
    sx = np.asarray(sx, dtype=float).ravel()
    sy = np.asarray(sy, dtype=float).ravel()
    best = np.zeros(sx.shape, dtype=float)
    for gx, gy in np.asarray(sources, dtype=float):
        w2 = np.degrees(np.arctan2(gy - sy, gx - sx)) % 360.0
        for d2 in np.asarray(deltas2, dtype=float):
            best = np.maximum(best, quad_diameters(site, theta1, sx, sy, w2 + d2, err_deg))
    return best


def suitability(j: np.ndarray, j_star: float) -> np.ndarray:
    """适合度 F = J*/J ∈ (0,1]；J 为哨兵值（退化构型，等于无法定位）时按 0 处理。"""
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
                  receive_min: float = cfg.RECEIVE_MIN) -> Dict[str, np.ndarray]:
    """全域粗网格上的四个场：坐标网格、到源集的最坏距离、可测性掩码、最坏定位直径。

    J 只在**可行点**上算 —— 不可行点根本不会成为第二检测点，算了也是浪费（可行域是圆域里
    很小的一块透镜，能省下 90% 以上的计算量）。不可行点的 J 记为哨兵值。
    """
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


def refine_best(site: Sequence[float], theta1: float, sources: np.ndarray, deltas2: np.ndarray,
                x0: float, y0: float, half: float = cfg.REFINE_HALF,
                step: float = cfg.REFINE_STEP, err_deg: float = cfg.BEARING_ERROR_DEG,
                receive_min: float = cfg.RECEIVE_MIN) -> Tuple[float, float, float]:
    """在粗解附近做局部细化，返回 (x, y, J)。

    J 在最优点附近光滑、"太大/近共线"的区域被哨兵值挡住，故 25 m 粗网格必然把最优点圈进
    ±80 m 的窗口里；窗口内 2 m 步长足以把最优值定到米级（再细的差异远小于 ±1° 测向误差对应
    的几十米量级）。
    """
    xs = np.arange(x0 - half, x0 + half + 0.5 * step, step)
    ys = np.arange(y0 - half, y0 + half + 0.5 * step, step)
    xx, yy = np.meshgrid(xs, ys)
    px, py = xx.ravel(), yy.ravel()
    feasible = worst_source_distance(px, py, sources) <= receive_min
    if not feasible.any():                       # 理论上不会发生（x0, y0 本就可行）
        return float(x0), float(y0), float("nan")
    j = worst_case_diameters(site, px[feasible], py[feasible], theta1, sources, deltas2, err_deg)
    k = int(np.argmin(j))
    return float(px[feasible][k]), float(py[feasible][k]), float(j[k])


# ----------------------------------------------------------------------------
# ③ 候选区域（弧带）
# ----------------------------------------------------------------------------

@dataclass
class Band:
    """候选区域的解析描述：若干"径向弧带"（每瓣一个）+ 多边形边界（绘图与面积用）。

    `radial_gaps` 记录"某个方位角上出现两段径向区间"的次数 —— 弧带组装取包络，若该值非 0
    说明描述略偏乐观，会打印出来（实测恒为 0）。
    """

    level_m: float                                  # 阈值 J ≤ level_m
    lobes: List[Dict[str, Any]] = field(default_factory=list)
    area_m2: float = 0.0
    radial_gaps: int = 0
    fine: Dict[str, np.ndarray] = field(default_factory=dict)   # 极坐标细网格场（放大图用）

    @property
    def r_lo(self) -> float:
        return min(l["r_lo"] for l in self.lobes) if self.lobes else float("nan")

    @property
    def r_hi(self) -> float:
        return max(l["r_hi"] for l in self.lobes) if self.lobes else float("nan")

    def describe(self) -> str:
        """一句话描述（控制台与报告共用）：极径范围 + 每瓣的方位范围。"""
        if not self.lobes:
            return "空"
        parts = [f"r ∈ [{self.r_lo:.0f}, {self.r_hi:.0f}] m"]
        for i, l in enumerate(self.lobes, 1):
            parts.append(f"第 {i} 瓣方位差 ∈ [{l['phi_lo']:+.1f}°, {l['phi_hi']:+.1f}°]")
        parts.append(f"合计面积 {self.area_m2 / 1e6:.3f} km²")
        return "；".join(parts)


def lens_ray_interval(site: Sequence[float], theta1: float, phi_deg: np.ndarray, d_lo: float,
                      d_hi: float, err_deg: float, receive_min: float, radius: float):
    """可行域透镜沿某个方位（绕 S1）的**精确**径向区间 (r_in, r_out) / m。

    透镜 = 圆域 ∩ 4 个半径 1000 m 圆盘（圆盘心为源不确定集的 4 个极点）。沿方向 u 从 S1 出发
    的射线与以 C 为心、半径 r 的圆盘相交的极径区间是 [proj − √D, proj + √D]（proj = u·w、
    D = proj² + r² − |w|²，w = C − S1；D < 0 表示该方位与圆盘无交）。5 个约束取交即得透镜沿该
    方位的精确区间 —— 它同时给出弧带的外缘（常被可测性限制住）与内缘（可能被两个远端圆盘
    的"阴影"顶出），也用来确定细网格的径向范围（否则放大图会漏掉透镜内侧的一半）。
    """
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
    """在绕 S1 的极坐标网格上提取 { 可行 ∩ J ≤ level } 的弧带。

     网格是极坐标下的矩形（r × φ），J 的次水平集在**每个方位角上是一段连续径向区间**（实测
    无例外，见 Band.radial_gaps），于是把每列的首/末极径连起来就得到弧带边界：外弧按 φ 升序、
    内弧按 φ 降序，闭合成一条简单多边形。分瓣按 φ 的符号（解关于示向度方向对称时恰好两瓣）。

    边界口径（关系到报告里的面积与范围，必须说清）：**外弧取"网格外缘与透镜精确外边界
    （`lens_ray_limit`）的较小者"**，因为弧带外缘通常正是被可测性限制住的（这时用精确值，
    最优点恰在外缘上也能如实落进带内）；**内弧取"网格内缘再向外半格"**，即把最后被选中的
    单元中心外扩半个径向步长（5 m 步长 ⇒ 2.5 m）。因此报告的范围/面积与真实次水平集的差别
    不超过半个网格单元（径向 ≤ 2.5 m、方位 ≤ 0.25°）。
    """
    phis = np.arange(-cfg.BAND_PHI_SPAN, cfg.BAND_PHI_SPAN + 0.5 * cfg.BAND_PHI_STEP,
                     cfg.BAND_PHI_STEP)
    # 径向网格按透镜的精确径向范围设定（含半步余量）：放大图必须覆盖整个可行域，
    # 只按粗解 ±BAND_R_SPAN 取会漏掉透镜内侧（本算例里内缘在 500 m，外缘在 1005 m）。
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
    # 每个方位角的精确透镜径向区间（弧带内外缘都可能是被它限制的），以及半格外扩量
    half_r = 0.5 * cfg.BAND_R_STEP

    band = Band(level_m=float(level))
    for want_pos in (True, False):                            # 正/负方位各成一瓣
        cols = [k for k in range(phis.size) if (phis[k] > 0) == want_pos and phis[k] != 0.0]
        run: List[int] = []
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


def _lobe(r: np.ndarray, phis: np.ndarray, cols: List[int], sel: np.ndarray,
          feasible: np.ndarray, theta1: float, site: Sequence[float], lim_in: np.ndarray,
          lim_out: np.ndarray, half_r: float) -> Dict[str, Any]:
    """把一串连续方位角上的径向区间组装成一瓣弧带（外弧正序 + 内弧逆序）。

    边界口径：**紧邻的下一个网格点是否可行**决定该侧是被可测性限制还是被 J 限制 ——
    被可测性限制（下一点不可行）时直接用透镜的精确边界 `lim_out`/`lim_in`（这是精确值，
    最优点恰在外缘上也能落进带内）；否则用网格单元中心外扩/内缩半格。
    """
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
    """绕 S1 的极坐标 (r, φ 相对示向度) → 直角坐标，形状 (N, 2)。"""
    a = np.radians(theta1 + phi_deg)
    return np.stack([float(site[0]) + r * np.cos(a), float(site[1]) + r * np.sin(a)], axis=-1)


# ----------------------------------------------------------------------------
# ④ 总入口
# ----------------------------------------------------------------------------

@dataclass
class SolveResult:
    """问题二的完整解：最优第二检测点、最优区域（候选弧带）、可行域与全部校验。"""

    site: Tuple[float, float]
    theta1: float
    j_star: float                                   # 最优最坏定位直径 / m
    best: Tuple[float, float]                       # 最优第二检测点坐标 / m
    best_r: float                                   # 到 S1 的距离 / m
    best_phi: float                                 # 相对示向度的方位差 / 度
    band: Band                                      # 候选区域（弧带）
    single_m: float                                 # 对照：只有一次测向时的区域直径 / m
    lens: Any                                       # 可行域多边形（shapely）
    lens_area_m2: float
    sources: np.ndarray
    corners: np.ndarray
    fields: Dict[str, np.ndarray]
    scenario: Dict[str, Any]                        # 最优点的最坏情形（论文插图用）
    checks: Dict[str, Any]

    @property
    def improvement(self) -> float:
        """单次测向直径 / 最优最坏直径 —— "第二次测向把最坏定位直径缩小了多少倍"。"""
        return self.single_m / self.j_star if self.j_star else float("inf")

    def summary(self) -> str:
        """控制台/报告用的一段话小结。"""
        x, y = self.best
        return (f"最优第二检测点 ({x:.0f}, {y:.0f}) m：距 S1 {self.best_r:.0f} m、"
                f"相对示向度 {self.best_phi:+.1f}°，最坏定位直径 J* = {self.j_star:.1f} m"
                f"（最坏交会角 {self.scenario['gamma_deg']:.1f}°）；候选区域 {self.band.describe()}")


def solve(site: Sequence[float] = cfg.DEFAULT_SITE, theta1: float = cfg.DEFAULT_BEARING,
          eta: float = cfg.ETA, d_lo: float = cfg.D_LO, d_hi: float = cfg.D_HI,
          err_deg: float = cfg.BEARING_ERROR_DEG, radius: float = cfg.REGION_RADIUS,
          receive_min: float = cfg.RECEIVE_MIN, verify_n: int = 400) -> SolveResult:
    """问题二的完整求解（确定性；`verify_n` > 0 时附带解析/shapely 抽样校验）。"""
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

    # ③ 候选区域
    band = candidate_band(site, theta1, sources, deltas2, j_star * (1.0 + eta),
                          r_center=math.hypot(bx - site[0], by - site[1]), d_lo=d_lo, d_hi=d_hi,
                          err_deg=err_deg, receive_min=receive_min, radius=radius)

    # ④ 最优点的最坏情形（含圆域截断判定）与各项校验
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
    checks: Dict[str, Any] = {
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
    }
    if verify_n > 0:
        checks["analytic_vs_shapely"] = verify_analytic(n=verify_n, err_deg=err_deg, radius=radius)
    return SolveResult(site=(float(site[0]), float(site[1])), theta1=float(theta1),
                       j_star=float(j_star), best=(bx, by),
                       best_r=math.hypot(bx - site[0], by - site[1]),
                       best_phi=_rel_angle(bearing(site, (bx, by)), theta1),
                       band=band, single_m=single, lens=lens, lens_area_m2=float(lens.area),
                       sources=sources, corners=corner_sources(site, theta1, d_lo, d_hi, err_deg),
                       fields=fields, scenario=scenario, checks=checks)


def _rel_angle(angle: float, ref: float) -> float:
    """angle 相对 ref 的方位差，取值 (−180, 180]。"""
    return ((angle - ref + 180.0) % 360.0) - 180.0


def _single_measurement_diameter(site: Sequence[float], theta1: float, err_deg: float,
                                 radius: float) -> float:
    """对照量：只做这一次测向时的定位区域直径（补测前的"最坏起点"）。"""
    region = TriangulationRegion(err=err_deg, radius=radius)
    region.add_node(float(site[0]), float(site[1]), float(theta1))
    return float(region.diameter)


def scenario_quad(result: SolveResult) -> Any:
    """最坏情形下定位区域的多边形（shapely），供插图使用。"""
    s = result.scenario
    return quad_polygon(result.site, result.theta1, result.best, s["theta2_deg"],
                        cfg.BEARING_ERROR_DEG)
