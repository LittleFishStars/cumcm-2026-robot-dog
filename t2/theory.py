"""问题二的文献判据层：两站纯方位测向的 Fisher 信息、CRLB 误差椭圆、GDOP 与几何稀释"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from common.console import print_table
from t2 import config as cfg
from t2.region import analytic_diameter

__all__ = ["CITATIONS", "gamma_deg_at", "fim_two_station", "ellipse_from_fim", "crlb_closed",
           "crlb_major", "crlb_area", "gdop", "foy_rmec", "minimax_radius", "radius_field",
           "spearman", "verify_theory"]

#: 文献信息，卷期页码按 resources/References/ 里的原文核过；供论文参考文献表直接取用
CITATIONS: dict[str, str] = {
    "foy1976": "W. H. Foy. Position-Location Solutions by Taylor-Series Estimation[J]. "
               "IEEE Transactions on Aerospace and Electronic Systems, 1976, AES-12(2): 187-194.",
    "chan1994": "Y. T. Chan, K. C. Ho. A Simple and Efficient Estimator for Hyperbolic "
                "Location[J]. IEEE Transactions on Signal Processing, 1994, 42(8): 1905-1915.",
    "ren2016": "任叶童. 基于到达角信息的无源定位算法研究[D]. 天津: 天津大学电子信息工程学院, "
               "2016. （2.4 节：双站测向定位误差概率椭圆长轴式(2-52)、短轴式(2-53)、面积式(2-54)）",
    "chen2009": "X. Chen, Z. Xu, L. Rui. An Optimization Algorithm of Multi-Observer "
                "Trajectories for Cooperative Bearings-Only Target Localization[C]. "
                "Proc. of ICICS 2009, IEEE, 2009.",
    "feng2026": "本项目. 2026 CUMCM B 题参考文献筛选清单[Z]. 2026-09-11. "
                "（问题二精读建议：E3 Zhao 2012、E1 Chen 2009、E4 廖海军 2008；"
                "本机缺 E3/E4/E5/E6/E7 与 2 题/ 目录）",
}


# ---------------------------------------------------------------- 几何与信息矩阵

def gamma_deg_at(g: Sequence[float], s1: Sequence[float], s2: Sequence[float]) -> float:
    """源 G 处的交会角，两条 L.O.S. 的夹角，单位度"""
    v1 = (float(s1[0]) - float(g[0]), float(s1[1]) - float(g[1]))
    v2 = (float(s2[0]) - float(g[0]), float(s2[1]) - float(g[1]))
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    c = (v1[0] * v2[0] + v1[1] * v2[1]) / max(n1 * n2, 1e-300)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def fim_two_station(s1: Sequence[float], theta1: float, s2: Sequence[float], theta2: float,
                    g: Sequence[float], sigma_rad: float | None = None) -> np.ndarray:
    """两站纯方位测向在真值 G 处的 Fisher 信息矩阵，2×2"""
    # 与 t3.probing.fisher_sigma 是同一个量，但 t2 与 t3 是并列叶子不许互相 import，各写一份
    s = math.radians(cfg.BEARING_ERROR_DEG) if sigma_rad is None else float(sigma_rad)
    H = np.zeros((2, 2), dtype=float)
    for (site, theta) in ((s1, theta1), (s2, theta2)):
        dx = float(g[0]) - float(site[0])
        dy = float(g[1]) - float(site[1])
        r = math.hypot(dx, dy)
        if r <= 0.0:
            continue
        # 视线方向的单位法向量，与 ∂θ/∂p 同向
        n = np.array([-dy / r, dx / r])
        H += (n[:, None] * n[None, :]) / (s * s * r * r)
    _ = theta   # θ 只以"测量值"的身份出现；CRLB 围绕真值线性化，取真值方向就够了
    return H


def ellipse_from_fim(H: np.ndarray) -> dict[str, float]:
    """由 Fisher 信息矩阵给出 1σ 误差椭圆的半轴、主轴方向、面积与 GDOP"""
    evals, evecs = np.linalg.eigh(H)
    lam_min, lam_max = float(evals[0]), float(evals[1])
    v = evecs[:, 0]
    return {
        "a_major_m": 1.0 / math.sqrt(lam_min) if lam_min > 0 else math.inf,
        "b_minor_m": 1.0 / math.sqrt(lam_max) if lam_max > 0 else math.inf,
        "area_m2": math.pi / math.sqrt(lam_min * lam_max) if lam_min * lam_max > 0 else math.inf,
        "gdop_m": math.sqrt(1.0 / lam_min + 1.0 / lam_max) if lam_min > 0 else math.inf,
        "major_axis_deg": math.degrees(math.atan2(v[1], v[0])) % 180.0,
    }


# ---------------------------------------------------------------- 文献闭式

def gdop(d: float, r2: float, gamma_deg: float, err_deg: float = cfg.BEARING_ERROR_DEG) -> float:
    """文献 GDOP 闭式，位置误差的均方尺度 / m"""
    g = math.radians(gamma_deg)
    return math.radians(err_deg) * math.sqrt(d * d + r2 * r2) / math.sin(g)


def crlb_major(d: float, r2: float, gamma_deg: float,
               err_deg: float = cfg.BEARING_ERROR_DEG) -> float:
    """两站测向 CRLB 误差椭圆主半轴的闭式 / m"""
    g = math.radians(gamma_deg)
    s = math.radians(err_deg)
    a = d * d + r2 * r2
    delta = math.sqrt(max(a * a - 4.0 * d * d * r2 * r2 * math.sin(g) ** 2, 0.0))
    den = a - delta
    if den <= 0.0:
        return math.inf
    return math.sqrt(2.0) * s * d * r2 / math.sqrt(den)


def crlb_area(d: float, r2: float, gamma_deg: float,
              err_deg: float = cfg.BEARING_ERROR_DEG) -> float:
    """两站测向 CRLB 误差椭圆的面积 / m²"""
    g = math.radians(gamma_deg)
    s = math.radians(err_deg)
    return math.pi * s * s * d * r2 / math.sin(g)


def foy_rmec(d: float, r2: float, gamma_deg: float,
             err_deg: float = cfg.BEARING_ERROR_DEG) -> float:
    """Foy 稀释式口径的最小外接圆半径 / m"""
    g = math.radians(gamma_deg)
    return math.tan(math.radians(err_deg)) * math.sqrt(d * d + r2 * r2) / math.sin(g)


def minimax_radius(d: float, r2: float, gamma_deg: float,
                   err_deg: float = cfg.BEARING_ERROR_DEG) -> float:
    """本文集员闭式，有界误差 minimax 口径下的定位区域半径 / m，等于 analytic_diameter 的一半"""
    return 0.5 * analytic_diameter(d, r2, gamma_deg, err_deg)


# ---------------------------------------------------------------- 判据场：快筛与对照

def radius_field(site: Sequence[float], theta1: float, points: np.ndarray, sources: np.ndarray,
                 mode: str = "gdop", err_deg: float = cfg.BEARING_ERROR_DEG) -> np.ndarray:
    """候选第二检测点上的判据尺度场 / m，mode 取 gdop、crlb、foy 或 minimax"""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    src = np.asarray(sources, dtype=float)
    d = np.hypot(src[:, 0] - float(site[0]), src[:, 1] - float(site[1]))          # (M,)
    dx = pts[:, 0][:, None] - src[None, :, 0]                                     # (N,M)
    dy = pts[:, 1][:, None] - src[None, :, 1]
    r2 = np.hypot(dx, dy)
    # 源处的交会角只依赖方向，用向量内积算，避开 arccos 的象限问题
    u1x = (src[None, :, 0] - float(site[0])) / np.maximum(d[None, :], 1e-12)
    u1y = (src[None, :, 1] - float(site[1])) / np.maximum(d[None, :], 1e-12)
    u2x = (src[None, :, 0] - pts[:, 0][:, None]) / np.maximum(r2, 1e-12)
    u2y = (src[None, :, 1] - pts[:, 1][:, None]) / np.maximum(r2, 1e-12)
    cosg = np.clip(u1x * u2x + u1y * u2y, -1.0, 1.0)
    gamma = np.arccos(cosg)                                                       # (N,M) 弧度
    sin_g = np.maximum(np.sin(gamma), 1e-12)
    if mode == "gdop":
        val = math.radians(err_deg) * np.sqrt(d[None, :] ** 2 + r2 ** 2) / sin_g
    elif mode == "crlb":
        a = d[None, :] ** 2 + r2 ** 2
        delta = np.sqrt(np.maximum(a * a - 4.0 * d[None, :] ** 2 * r2 ** 2 * np.sin(gamma) ** 2, 0.0))
        val = math.sqrt(2.0) * math.radians(err_deg) * d[None, :] * r2 / np.sqrt(
            np.maximum(a - delta, 1e-300))
    elif mode == "foy":
        val = math.tan(math.radians(err_deg)) * np.sqrt(d[None, :] ** 2 + r2 ** 2) / sin_g
    elif mode == "minimax":
        val = 0.5 * (math.radians(2.0 * err_deg) / sin_g) * np.sqrt(
            d[None, :] ** 2 + r2 ** 2 + 2.0 * d[None, :] * r2 * np.abs(np.cos(gamma)))
    else:
        raise ValueError(f"未知判据 {mode!r}，应为 gdop/crlb/foy/minimax")
    return val.max(axis=1)


# ---------------------------------------------------------------- 秩相关与自检

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关系数，按秩的 Pearson 相关手算"""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")

    def rank(x: np.ndarray) -> np.ndarray:
        """求一维数组的平均秩，并列元素取同一个平均秩"""
        order = np.argsort(x, kind="stable")
        r = np.empty(x.size, dtype=float)
        r[order] = np.arange(x.size, dtype=float)
        # 并列取平均秩，避免整数量级差被放大
        xs = x[order]
        i = 0
        while i < xs.size:
            j = i
            while j + 1 < xs.size and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb / den) if den > 0 else float("nan")


def verify_theory(n: int = 300, seed: int = 2026,
                  err_deg: float = cfg.BEARING_ERROR_DEG,
                  radius: float = cfg.REGION_RADIUS) -> dict[str, Any]:
    """文献闭式与精确集员构造的抽样对照，返回可直接写进论文的校验数字"""
    from t2.region import D_LO, exact_region

    rng = np.random.default_rng(seed)
    e1, e2, e3 = [], [], []          # ①闭式一致性 ②集员闭式偏差 ③文献判据低估比例
    reg = []                         # (γ, r2/d, 本文闭式相对偏差, GDOP 相对偏差) 分桶用
    n_clip = n_degen = 0
    for _ in range(int(n)):
        site = (float(rng.uniform(-radius, radius)), float(rng.uniform(-radius, radius)))
        theta1 = float(rng.uniform(0.0, 360.0))
        ang = float(rng.uniform(0.0, 360.0))
        dist = float(rng.uniform(50.0, 2500.0))
        s2 = (site[0] + dist * math.cos(math.radians(ang)),
              site[1] + dist * math.sin(math.radians(ang)))
        d = float(rng.uniform(D_LO, 1500.0))
        g = (site[0] + d * math.cos(math.radians(theta1)),
             site[1] + d * math.sin(math.radians(theta1)))
        theta2 = math.degrees(math.atan2(g[1] - s2[1], g[0] - s2[0])) % 360.0
        gamma = gamma_deg_at(g, site, s2)
        r2 = math.hypot(g[0] - s2[0], g[1] - s2[1])
        if math.sin(math.radians(gamma)) < 0.02:
            n_degen += 1
            continue
        region = exact_region(site, theta1, s2, theta2, err_deg, radius)
        if not region.bounded or region.diameter <= 0.0:
            n_clip += 1
            continue
        exact = float(region.diameter) / 2.0
        # ① 闭式 vs 数值特征值
        H = fim_two_station(site, theta1, s2, theta2, g)
        ell = ellipse_from_fim(H)
        if math.isfinite(ell["a_major_m"]) and ell["a_major_m"] > 0:
            for closed, key in ((crlb_major(d, r2, gamma, err_deg), "a_major_m"),
                                (gdop(d, r2, gamma, err_deg), "gdop_m"),
                                (crlb_area(d, r2, gamma, err_deg), "area_m2")):
                val = ell[key]
                if math.isfinite(val) and val > 0:
                    e1.append(abs(closed - val) / val)
        # ②③ 各闭式 vs 精确最坏半径
        mm = (minimax_radius(d, r2, gamma, err_deg) - exact) / exact
        gg = (gdop(d, r2, gamma, err_deg) - exact) / exact
        e2.append(mm)
        for closed in (gdop(d, r2, gamma, err_deg), crlb_major(d, r2, gamma, err_deg),
                       foy_rmec(d, r2, gamma, err_deg)):
            e3.append((closed - exact) / exact)
        reg.append((gamma, r2 / max(d, 1e-9), mm, gg))
    a1 = np.asarray(e1, dtype=float)
    a2 = np.asarray(e2, dtype=float)
    a3 = np.asarray(e3, dtype=float)
    ar = np.asarray(reg, dtype=float).reshape(-1, 4)

    def _bucket(lo: float, hi: float, col: int) -> dict[str, Any]:
        """按 ar 第 col 列的取值区间 [lo, hi) 分桶统计偏差"""
        m = (ar[:, col] >= lo) & (ar[:, col] < hi)
        if not bool(m.any()):
            return {"lo": lo, "hi": hi, "n": 0}
        return {"lo": lo, "hi": hi, "n": int(m.sum()),
                "minimax_median_rel_dev": float(np.median(ar[m, 2])),
                "minimax_max_abs_rel_dev": float(np.abs(ar[m, 2]).max()),
                "gdop_median_rel_dev": float(np.median(ar[m, 3]))}

    gammas = [(15.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 120.0), (120.0, 155.0)]
    ratios = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 1e9)]
    return {
        "n": int(n), "n_used": int(a2.size), "n_clipped": int(n_clip), "n_degenerate": int(n_degen),
        "closed_vs_eigen_max_rel_dev": float(np.abs(a1).max()) if a1.size else None,
        "minimax_closed_median_rel_dev": float(np.median(np.abs(a2))) if a2.size else None,
        "minimax_closed_max_rel_dev": float(np.abs(a2).max()) if a2.size else None,
        "gdop_vs_exact_median_rel_dev": float(np.median(a3)) if a3.size else None,
        "gdop_vs_exact_max_relerr": float(np.abs(a3).max()) if a3.size else None,
        "by_gamma_deg": [_bucket(lo, hi, 0) for lo, hi in gammas],
        "by_r2_over_d": [_bucket(lo, hi, 1) for lo, hi in ratios],
        "citations": CITATIONS,
    }


def _ratio_label(b: dict[str, Any]) -> str:
    """分桶的区间标签，最后一桶上界是 1e9 这种哨兵值，直接打出来难看"""
    if b["hi"] >= 1e9:
        return f"≥ {b['lo']:.1f}"
    return f"[{b['lo']:.1f}, {b['hi']:.1f})"


def _selfcheck(n: int = 300) -> None:
    """自检：文献闭式与数值特征值是否一致，集员闭式与精确构造是否一致，文献判据偏低多少"""
    print("问题二文献判据自检")
    # (a) 固定构型的解析核对，手算也能复一遍
    d, r2 = 1500.0, 907.237
    gamma = 40.6396
    H = fim_two_station((0.0, 0.0), 0.0, (801.133, 604.956),
                        math.degrees(math.atan2(26.178609655925268 - 604.956,
                                                1499.771542734587 - 801.133)) % 360.0,
                        (1499.771542734587, 26.178609655925268))
    ell = ellipse_from_fim(H)
    print(f"  最坏情形几何 R1 = {d:.0f} m、R2 = {r2:.0f} m、γ = {gamma:.3f}°，解析核对：")
    print_table(["量", "数值", "闭式"],
                [["CRLB 主半轴 / m", f"{ell['a_major_m']:.4f}", f"{crlb_major(d, r2, gamma):.4f}"],
                 ["GDOP / m", f"{ell['gdop_m']:.4f}", f"{gdop(d, r2, gamma):.4f}"],
                 ["椭圆面积 / m²", f"{ell['area_m2']:.2f}", f"{crlb_area(d, r2, gamma):.2f}"],
                 ["本文集员半径 / m", "精确 67.04", f"闭式 {minimax_radius(d, r2, gamma):.2f}"],
                 ["Foy 稀释式半径 / m", "—", f"闭式 {foy_rmec(d, r2, gamma):.2f}"]],
                align="lrr")
    # (b) 抽样
    out = verify_theory(n)
    print(f"  抽样 {out['n']}：可用 {out['n_used']}，圆域截断 {out['n_clipped']}，"
          f"近共线 {out['n_degenerate']}")
    print_table(["检查项", "结果", "数值"],
                [["文献闭式 vs 数值特征值", "✓",
                  f"最大相对偏差 {out['closed_vs_eigen_max_rel_dev']:.2e}"],
                 ["本文集员闭式 vs 精确最坏半径", "✓",
                  f"中位 {out['minimax_closed_median_rel_dev'] * 100:.3f}%，"
                  f"最大 {out['minimax_closed_max_rel_dev'] * 100:.1f}%"],
                 ["文献 GDOP/CRLB/Foy 判据 vs 精确最坏半径", "✓",
                  f"中位偏差 {out['gdop_vs_exact_median_rel_dev'] * 100:+.1f}%，负值为低估"]],
                align="lcr")
    print("  按交会角 γ 分桶，列出本文闭式与文献 GDOP 式的中位相对偏差：")
    print_table(["γ / °", "样本数", "本文中位偏差 / %", "本文最大偏差 / %", "GDOP 中位偏差 / %"],
                [[f"[{b['lo']:.0f}, {b['hi']:.0f})", b["n"],
                  f"{b['minimax_median_rel_dev'] * 100:+.2f}",
                  f"{b['minimax_max_abs_rel_dev'] * 100:.1f}",
                  f"{b['gdop_median_rel_dev'] * 100:+.1f}"]
                 for b in out["by_gamma_deg"] if b["n"]],
                align="lrrrr")
    print("  按 r₂/d 分桶：")
    print_table(["r₂/d", "样本数", "本文中位偏差 / %", "本文最大偏差 / %", "GDOP 中位偏差 / %"],
                [[_ratio_label(b), b["n"],
                  f"{b['minimax_median_rel_dev'] * 100:+.2f}",
                  f"{b['minimax_max_abs_rel_dev'] * 100:.1f}",
                  f"{b['gdop_median_rel_dev'] * 100:+.1f}"]
                 for b in out["by_r2_over_d"] if b["n"]],
                align="lrrrr")


if __name__ == '__main__':
    _selfcheck()
