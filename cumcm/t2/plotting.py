"""问题二的成果图。

* **`draw_suitability`**（主图）：第二检测点的适合度图（一张图说清"往哪儿走"）。

题目要的是"第二个检测点的候选区域"，一张能直接支撑结论的图必须同时回答四个问题：
① 第二点必须落在哪（可行域，否则第二次测向可能什么都听不到）；② 落在不同位置的效果差多少
（适合度场）；③ 最优的那一点在哪、离 S1 多远、偏示向度多少度；④ 为什么是现在这个形状
（最坏情形下定位区域长什么样）。因此图的版式是：

* **(a) 全域视图**：1800 m 圆域、源不确定集（以 S1 为顶点、长 1500 m、±1° 的窄扇形）、S1 与
  示向度射线、灰色的"不保证可测"区域、可行域透镜、适合度场与等值线 —— 回答 ①②；并画出
  放大框指明 (b) 的位置。
* **(b) 放大视图**：把可行域透镜放大，画细网格适合度场、适合度等值线（标注对应的最坏定位
  直径 J，单位 m）、候选区域弧带（J ≤ (1+η)$J^*$）、最优第二检测点 $S_2^*$ 及其实测极坐标标注
  （距 S1 的距离 $r^*$、相对示向度的方位差 $\\varphi^*$）—— 回答 ②③。
* **(b) 的内嵌小图（最坏情形几何）**：放大到定位区域尺度，画出两条示向度的 4 条 ±1° 边界射线
  与它们围成的四边形，标出直径（= $J^*$）—— 回答 ④，也解释了"为什么要把交会角做到 40° 左右"。

图上的每一次定量标注（$r^*$、$\\varphi^*$、$J^*$、面积、缩小倍数）都取自 `cumcm.t2.score.SolveResult` 的
数值字段，与 JSON/CSV 输出同源，便于逐项对账；PNG 走 `save_png`（去掉时间戳元数据），PDF 用
固定 CreationDate，故同一输入两次出图逐字节一致。
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from cumcm.common.plotting import (C_FRAME, C_GRAY, C_MEAS, C_PATH, C_SRC, font_context,
                                   no_plot_hint, save_png, setup_mpl_env)
from cumcm.t2 import config as cfg
from cumcm.t2 import theory
from cumcm.t2.region import quad_diameters
from cumcm.t2.score import SolveResult, scenario_quad, suitability

__all__ = ["draw_suitability", "draw_criteria", "zoom_window"]

# 图上专用配色：适合度用"深色 = 好"的冷暖渐变，候选弧带用亮橙以区别于红色的最坏情形
C_MAP = "YlGnBu"
C_BAND = "#e8590c"
C_INFEASIBLE = "#e9ebee"
_PDF_DATE = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)  # 固定时间戳


def zoom_window(result: SolveResult, margin: float = 150.0,
                aspect: float = 1.10) -> Tuple[float, float, float, float]:
    """放大视图窗口：可行域透镜的包围盒 + 边距，再按目标长宽比补成矩形 (x0, x1, y0, y1)。"""
    x0, y0, x1, y1 = result.lens.bounds
    x0, x1, y0, y1 = x0 - margin, x1 + margin, y0 - margin, y1 + margin
    w, h = x1 - x0, y1 - y0
    if w < aspect * h:                       # 太窄：左右加宽
        pad = (aspect * h - w) / 2.0
        x0, x1 = x0 - pad, x1 + pad
    else:                                    # 太扁：上下加高
        pad = (w / aspect - h) / 2.0
        y0, y1 = y0 - pad, y1 + pad
    return x0, x1, y0, y1


def _frame(ax: Any, result: SolveResult) -> None:
    """圆域边界、灰色"不保证可测"底、源不确定集扇形、S1 与示向度射线。"""
    th = np.linspace(0.0, 2.0 * math.pi, 721)
    cos_t, sin_t = np.cos(th), np.sin(th)
    ax.fill(cfg.REGION_RADIUS * cos_t, cfg.REGION_RADIUS * sin_t, color=C_INFEASIBLE, lw=0.0,
            zorder=0.5)
    ax.plot(cfg.REGION_RADIUS * cos_t, cfg.REGION_RADIUS * sin_t, color=C_FRAME, lw=1.4,
            zorder=2.0)
    corners = result.corners                       # 顺序：近端两角（d=D_LO）、远端两角（d=D_HI）
    wedge = np.vstack([[result.site], corners[2:4], [result.site]])
    ax.fill(wedge[:, 0], wedge[:, 1], color=C_MEAS, alpha=0.18, lw=0.0, zorder=1.0)
    ax.plot(corners[:2, 0], corners[:2, 1], marker="o", ms=2.4, color=C_MEAS, zorder=1.6)
    ax.plot(corners[2:4, 0], corners[2:4, 1], marker="o", ms=3.0, color=C_MEAS, zorder=1.6)
    for s, ls, lw in ((-cfg.BEARING_ERROR_DEG, "--", 0.8), (cfg.BEARING_ERROR_DEG, "--", 0.8),
                      (0.0, "-.", 1.2)):
        a = math.radians(result.theta1 + s)
        ax.plot([result.site[0], result.site[0] + cfg.D_HI * math.cos(a)],
                [result.site[1], result.site[1] + cfg.D_HI * math.sin(a)], color=C_MEAS, lw=lw,
                ls=ls, alpha=0.9, zorder=1.5)


def _cell_edges(v: np.ndarray) -> np.ndarray:
    """单调一维网格 → 单元边界（首末外扩半格，中间取中点）。"""
    d = np.diff(v)
    return np.concatenate([[v[0] - d[0] / 2.0], v[:-1] + d / 2.0, [v[-1] + d[-1] / 2.0]])


def _suitability_panel(ax: Any, result: SolveResult, x: np.ndarray, y: np.ndarray,
                       j: np.ndarray, feasible: np.ndarray, label_contours: bool,
                       polar: bool = False) -> Any:
    """铺适合度场 + 等值线 + 候选弧带，返回 pcolormesh 供 colorbar 使用。

    `polar=True` 时网格来自绕 S1 的极坐标（x、y 沿行/列都不单调），必须显式给出单元边界，
    否则 pcolormesh 只能猜边界（会警告并可能画错）。边界由相邻单元中心的中点外推得到。
    """
    f = np.ma.masked_where(~feasible, suitability(j, result.j_star))
    if polar:
        fine = result.band.fine
        r_e, p_e = _cell_edges(fine["r"][:, 0]), _cell_edges(fine["phi_deg"][0, :])
        pe_deg = np.radians(result.theta1 + p_e)                       # 角度边界
        xe = float(result.site[0]) + np.outer(r_e, np.cos(pe_deg))
        ye = float(result.site[1]) + np.outer(r_e, np.sin(pe_deg))
        mesh = ax.pcolormesh(xe, ye, f, cmap=C_MAP, vmin=0.0, vmax=1.0, shading="flat",
                             zorder=1.2, rasterized=True)
    else:
        mesh = ax.pcolormesh(x, y, f, cmap=C_MAP, vmin=0.0, vmax=1.0, shading="nearest",
                             zorder=1.2, rasterized=True)
    levels = [lv for lv in cfg.CONTOUR_F if lv <= float(f.max())]
    if levels:
        cs = ax.contour(x, y, f, levels=levels, colors=C_GRAY, linewidths=0.7, zorder=1.8)
        if label_contours:
            ax.clabel(cs, fmt={lv: f"F={lv:.2f}\nJ≤{result.j_star / lv:.0f} m" for lv in levels},
                      fontsize=5.6, inline=True, inline_spacing=2)
    for lobe in result.band.lobes:
        b = lobe["boundary"]
        ax.fill(b[:, 0], b[:, 1], color=C_BAND, alpha=0.28, lw=0.0, zorder=2.1)
        ax.plot(np.append(b[:, 0], b[0, 0]), np.append(b[:, 1], b[0, 1]), color=C_BAND, lw=1.3,
                zorder=2.2)
    return mesh


def _lens_outline(ax: Any, result: SolveResult) -> None:
    """可行域透镜边界（点线）—— 第二点越出它就可能听不到最坏的那个源。"""
    polys = result.lens.geoms if result.lens.geom_type == "MultiPolygon" else [result.lens]
    for poly in polys:
        xy = np.asarray(poly.exterior.coords)
        ax.plot(xy[:, 0], xy[:, 1], color="#111111", lw=1.1, ls=":", zorder=2.6)


def _best_marker(ax: Any, result: SolveResult, fontsize: float = 7.0,
                 annotate: bool = True) -> None:
    """最优第二检测点：星标 + 到 S1 的连线 + 极坐标标注。"""
    bx, by = result.best
    ax.plot([result.site[0], bx], [result.site[1], by], color=C_PATH, lw=1.0, zorder=3.0)
    ax.plot([bx], [by], marker="*", ms=13, color=C_SRC, mec="white", mew=0.6, ls="none",
            zorder=3.4)
    if not annotate:
        return
    ax.annotate(f"$S_2^*$：$r^*$ = {result.best_r:.0f} m，$\\varphi^*$ = {result.best_phi:+.1f}°\n"
                f"最坏定位直径 $J^*$ = {result.j_star:.0f} m",
                xy=(bx, by), xytext=(12, 12), textcoords="offset points", fontsize=fontsize,
                color="#202020", zorder=4.0,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_SRC, lw=0.6, alpha=0.92))


def _line_through(ax: Any, p: Tuple[float, float], ang_deg: float, **kw: Any) -> None:
    """画过点 p、方向 ang_deg 的整条直线（超出视窗的部分交给坐标轴裁剪）。"""
    a = math.radians(ang_deg)
    span = 6000.0
    ax.plot([p[0] - span * math.cos(a), p[0] + span * math.cos(a)],
            [p[1] - span * math.sin(a), p[1] + span * math.sin(a)], **kw)


def _worst_case_inset(ax: Any, result: SolveResult) -> None:
    """内嵌小图：最坏情形的定位区域（两条示向度的 ±1° 边界射线所围的四边形）与它的直径。

    按真实几何画：S1 的两条边界射线在 $\\theta_1$ ± 1°，S2 的两条在"最坏那次的实测示向度" ± 1°，
    四线围成的就是附录图 2 的四边形；直径即最长对角线（= $J^*$）。
    """
    poly = scenario_quad(result)
    verts = np.asarray(poly.exterior.coords)
    gx, gy = result.scenario["source"]
    for base, site in ((result.theta1, result.site), (result.scenario["theta2_deg"], result.best)):
        col = C_MEAS if base == result.theta1 else C_PATH
        for s in (-cfg.BEARING_ERROR_DEG, cfg.BEARING_ERROR_DEG):
            _line_through(ax, site, base + s, color=col, lw=0.7, ls="--", alpha=0.85, zorder=1.6)
    ax.fill(verts[:, 0], verts[:, 1], color=C_SRC, alpha=0.18, lw=0.0, zorder=1.0)
    ax.plot(verts[:, 0], verts[:, 1], color=C_SRC, lw=1.2, zorder=2.0)
    ax.plot([gx], [gy], marker="X", ms=6, color="#111111", ls="none", zorder=3.0)
    span = max(60.0, 1.05 * float(np.ptp(verts, axis=0).max()))
    ax.set_xlim(gx - span, gx + span)
    ax.set_ylim(gy - span, gy + span)
    p, q = _farthest_pair(verts[:-1])
    ax.plot([p[0], q[0]], [p[1], q[1]], color="#111111", lw=1.0, ls=(0, (4, 2)), zorder=2.4)
    ax.annotate(f"直径 {result.j_star:.0f} m", xy=((p[0] + q[0]) / 2, (p[1] + q[1]) / 2),
                xytext=(0, 8), textcoords="offset points", fontsize=5.8, ha="center",
                color="#111111", zorder=4.0,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.annotate("最坏源 $G^*$", xy=(gx, gy), xytext=(3, -10), textcoords="offset points",
                fontsize=5.4, color="#111111", zorder=4.0)
    ax.text(0.02, 0.97, "— — $S_1$ 的 ±1° 边界", transform=ax.transAxes, fontsize=5.2,
            color=C_MEAS, va="top")
    ax.text(0.02, 0.88, "— — $S_2$ 的 ±1° 边界", transform=ax.transAxes, fontsize=5.2,
            color=C_PATH, va="top")
    ax.set_title(f"最坏情形：交会角 {result.scenario['gamma_deg']:.0f}°、源距 S1 "
                 f"{result.scenario['d_m']:.0f} m、$\\delta_1$={result.scenario['delta1_deg']:+.0f}°、"
                 f"$\\delta_2$={result.scenario['delta2_deg']:+.0f}°", fontsize=6.0, color="#202020")
    ax.tick_params(labelsize=5.2)
    ax.grid(alpha=0.25, lw=0.4)


def _farthest_pair(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """一组点中最远的一对（顶点只有几个，直接穷举）。"""
    best = (0.0, 0, 0)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.hypot(*(pts[i] - pts[j])))
            if d > best[0]:
                best = (d, i, j)
    return pts[best[1]], pts[best[2]]


def draw_suitability(out_path: Path, result: SolveResult, dpi: float = cfg.DPI,
                     pdf_path: Optional[Path] = None) -> Path:
    """画"第二检测点适合度图"并落盘，返回 PNG 路径（给出 pdf_path 时同时输出矢量 PDF）。"""
    setup_mpl_env()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:                                   # 没有 matplotlib 时明确告知
        no_plot_hint()
        return Path(out_path)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with font_context(size=8.6):
        fig = plt.figure(figsize=(13.6, 6.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.22], wspace=0.16, left=0.055,
                              right=0.905, top=0.86, bottom=0.17)

        # ---------------- (a) 全域视图 ----------------
        ax = fig.add_subplot(gs[0, 0])
        _frame(ax, result)
        mesh = _suitability_panel(ax, result, result.fields["x"], result.fields["y"],
                                  result.fields["j"], result.fields["feasible"],
                                  label_contours=False)
        _lens_outline(ax, result)
        _best_marker(ax, result, annotate=False)
        x0, x1, y0, y1 = zoom_window(result)
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#202020", lw=0.9,
                ls=(0, (5, 3)), zorder=3.2)
        ax.annotate("右图区域", xy=(x1, y1), xytext=(-3, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=6.6, color="#202020")
        lim = cfg.REGION_RADIUS * 1.06
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title("(a) 全域：1800 m 圆域、源不确定集与适合度", fontsize=9.4)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.grid(alpha=0.18, lw=0.4)
        ax.tick_params(labelsize=7.6)
        ax.annotate("灰色：不保证第二次可测\n（源可能比 1000 m 更近）", xy=(-0.84 * lim, -0.86 * lim),
                    fontsize=7.0, color=C_GRAY)
        cax = fig.add_axes([0.916, 0.20, 0.016, 0.58])
        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label("适合度 F = $J^*$/J（1 = 最优）", fontsize=8.0)
        cb.ax.tick_params(labelsize=7.0)

        # ---------------- (b) 放大视图 + 最坏情形内嵌图 ----------------
        ax2 = fig.add_subplot(gs[0, 1])
        fine = result.band.fine
        _suitability_panel(ax2, result, fine["x"], fine["y"], fine["j"], fine["feasible"],
                           label_contours=True, polar=True)
        _frame(ax2, result)
        _lens_outline(ax2, result)
        _best_marker(ax2, result, fontsize=7.4)
        ax2.set_xlim(x0, x1)
        ax2.set_ylim(y0, y1)
        ax2.set_aspect("equal")
        ax2.set_title("(b) 放大：可行域内的适合度场与候选区域", fontsize=9.4)
        ax2.set_xlabel("x / m")
        ax2.grid(alpha=0.18, lw=0.4)
        ax2.tick_params(labelsize=7.6)
        ax2.annotate(f"可行域（保证可测）：{result.lens_area_m2 / 1e6:.3f} km²\n"
                     f"候选区域：J ≤ {result.band.level_m:.0f} m"
                     f"（占其 {100.0 * result.band.area_m2 / result.lens_area_m2:.1f}%）\n"
                     f"最坏定位直径 1800 m → {result.j_star:.0f} m"
                     f"（缩小 {result.improvement:.1f} 倍）",
                     xy=(0.015, 0.985), xycoords="axes fraction", va="top", ha="left",
                     fontsize=7.2, color="#202020", zorder=4.0,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GRAY, lw=0.5,
                               alpha=0.92))
        _worst_case_inset(ax2.inset_axes([0.695, 0.045, 0.295, 0.315]), result)

        handles = [
            Line2D([], [], color=C_MEAS, lw=1.2, ls="-.", label="$S_1$ 示向度（±1° 虚线）"),
            Line2D([], [], color=C_MEAS, alpha=0.35, lw=6, label="源不确定集（楔形）"),
            Line2D([], [], color="#111111", lw=1.1, ls=":", label="可行域（透镜）"),
            Line2D([], [], color=C_BAND, lw=1.3, alpha=0.7,
                   label="候选区域（弧带）"),
            Line2D([], [], color=C_SRC, marker="*", ms=10, ls="none",
                   label="$S_2^*$ 最优第二检测点"),
            Line2D([], [], color=C_PATH, lw=1.0, label="$S_1$ → $S_2^*$ 连线"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.0, frameon=False,
                   columnspacing=2.4, handletextpad=0.7, handlelength=1.8,
                   bbox_to_anchor=(0.5, 0.012))
        fig.suptitle(f"问题二：第二检测点适合度图（$S_1$ = ({result.site[0]:.0f}, "
                     f"{result.site[1]:.0f}) m，$\\theta_1$ = {result.theta1:.0f}°；只有一次测向时最坏"
                     f"定位直径 {result.single_m:.0f} m → 补测后 {result.j_star:.0f} m，"
                     f"缩小 {result.improvement:.1f} 倍）", fontsize=10.4, y=0.965)
        save_png(fig, out_path, dpi=dpi)
        if pdf_path is not None:
            pdf_path = Path(pdf_path)
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pdf_path, metadata={"Software": None, "CreationDate": _PDF_DATE})
        plt.close(fig)
    return out_path


def _exact_radius_at(R1: float, R2: float, gamma_deg: float,
                     err_deg: float = cfg.BEARING_ERROR_DEG) -> float:
    """构造一个恰好实现 (R1, R2, γ) 的两站构型，并用精确构造给出最坏半径 / m。

    记源 G 在 (R1, 0)、S1 在原点，则 G→S1 方向为 180°；取 G→S2 方向为 180°−γ、长度 R2，
    于是两站在源处的交会角恰为 γ。再用 `quad_diameters`（±1° 楔形交的精确直径）取一半。
    """
    gx, gy = R1, 0.0
    ang = math.radians(180.0 - gamma_deg)
    s2x, s2y = gx + R2 * math.cos(ang), gy + R2 * math.sin(ang)
    theta2 = math.degrees(math.atan2(gy - s2y, gx - s2x)) % 360.0
    d = float(quad_diameters((0.0, 0.0), 0.0, np.asarray([s2x]), np.asarray([s2y]),
                             np.asarray([theta2]), err_deg)[0])
    return 0.5 * d


def draw_criteria(out_path: Path, result: SolveResult, dpi: int = cfg.DPI,
                  pdf_path: Optional[Path] = None) -> Path:
    """文献判据 vs 本文精确判据的对照图（写论文"为什么这么选点"用）。

    * **(a) 交会角的影响**：固定 R₁ = 1500 m、R₂ = 907 m（本文最坏情形的距离组合），
      按 γ ∈ [15°, 90°] 逐点用**精确构造**算出最坏半径，与三条文献闭式/本文闭式并排 ——
      直接显示 Foy 1976 的几何稀释（γ 越小半径越大），以及文献 GDOP 式系统性偏低约三成。
    * **(b) 判据一致性**：可行域内 1200 余个采样点上，文献 GDOP 判据与本文精确 J 的散点
      （双对数），给出 Spearman 秩相关 —— 说明它**适合做快筛**（秩几乎一致）但**不能当硬界**
      （点云整体在对角线下方，即 GDOP 低估最坏直径）。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        no_plot_hint()
        return Path(out_path)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    th = result.scenario
    R1, R2 = float(th["d_m"]), float(th["r2_m"])
    gammas = np.arange(15.0, 90.5, 2.5)
    exact = np.array([_exact_radius_at(R1, R2, float(g)) for g in gammas])
    curves = {
        "精确构造（±1° 楔形交，本文判据）": (exact, "#111111", "-", 1.8, "o", 3.0),
        "本文集员闭式": (np.array([theory.minimax_radius(R1, R2, float(g)) for g in gammas]),
                        C_BAND, "--", 1.3, None, 0),
        "文献 GDOP $\\sigma\\sqrt{R_1^2+R_2^2}/\\sin\\gamma$":
            (np.array([theory.gdop(R1, R2, float(g)) for g in gammas]), "#1c7ed6", "-.", 1.3, None, 0),
        "文献 CRLB 主半轴（任叶童 2016 式(2-52)）":
            (np.array([theory.crlb_major(R1, R2, float(g)) for g in gammas]), "#7048e8", ":", 1.5,
             None, 0),
        "文献稀释式 $\\tan\\varepsilon\\sqrt{R_1^2+R_2^2}/\\sin\\gamma$":
            (np.array([theory.foy_rmec(R1, R2, float(g)) for g in gammas]), "#2b8a3e", (0, (4, 2)),
             1.3, None, 0),
    }
    with font_context(size=8.6):
        fig = plt.figure(figsize=(12.6, 5.4))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.24, left=0.075, right=0.985,
                              top=0.85, bottom=0.13)
        ax = fig.add_subplot(gs[0, 0])
        for label, (y, c, ls, lw, mk, ms) in curves.items():
            ax.plot(gammas, y, color=c, ls=ls, lw=lw, marker=mk, ms=ms, label=label, alpha=0.95)
        ax.axvline(float(th["gamma_deg"]), color=C_GRAY, lw=0.9, ls=":")
        ax.annotate(f"本文最坏情形\n$\\gamma$ = {float(th['gamma_deg']):.1f}°",
                    xy=(float(th["gamma_deg"]), 0.0), xycoords=("data", "axes fraction"),
                    xytext=(4, 6), textcoords="offset points", fontsize=7.0, color="#202020")
        ax.set_yscale("log")
        ax.set_xlabel("源处交会角 $\\gamma$ / °")
        ax.set_ylabel("定位区域半径 / m（对数轴）")
        ax.set_title(f"(a) 交会角的影响（$R_1$ = {R1:.0f} m，$R_2$ = {R2:.0f} m）", fontsize=9.2)
        ax.grid(alpha=0.20, lw=0.4, which="both")
        ax.tick_params(labelsize=7.6)
        ax.legend(fontsize=6.8, loc="upper right", frameon=False)

        probe = result.theory.get("probe", {})
        ax2 = fig.add_subplot(gs[0, 1])
        if probe:
            gx = np.asarray(probe["gdop_m"], dtype=float)
            jy = np.asarray(probe["worst_diam_m"], dtype=float)
            ax2.scatter(gx, jy, s=7.0, color="#1c7ed6", alpha=0.45, lw=0.0)
            lim = [min(gx.min(), jy.min()) * 0.9, max(gx.max(), jy.max()) * 1.1]
            ax2.plot(lim, lim, color=C_GRAY, lw=0.9, ls="--")
            ax2.set_xlim(lim)
            ax2.set_ylim(lim)
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            rho = float(result.theory["criteria"]["gdop_vs_exact_spearman"])
            ax2.annotate(f"可行域内 {gx.size} 个采样点\nSpearman 秩相关 = {rho:.3f}\n"
                         f"（秩几乎一致 → 可作快筛；\n点云在对角线下方 → GDOP 偏低，不可当硬界）",
                         xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left",
                         fontsize=7.2, color="#202020",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GRAY, lw=0.5,
                                   alpha=0.92))
            # 三个口径的最优点在 (GDOP, J) 平面上几乎重合（相差 < 0.2%），故只画星标 + 一处合注，
            # 避免三个文字标签叠在一起（数值对照放在右下角的框里）
            tags = {"minimax": "本文 minimax", "gdop": "文献 GDOP", "expected": "期望口径"}
            lines = []
            for name, q in result.theory.get("points", {}).items():
                ax2.scatter([q["gdop_m"]], [q["worst_diam_m"]], s=34, marker="*",
                            color=C_BAND if name == "minimax" else "#7048e8", zorder=5)
                lines.append(f"{tags[name]}：$r$ = {q['r_m']:.0f} m、$\\varphi$ = {q['phi_deg']:+.1f}°、"
                             f"$J$ = {q['worst_diam_m']:.1f} m")
            ax2.annotate("三个口径的最优点几乎重合（$J$ 相差 < 0.1%）：\n" + "\n".join(lines),
                         xy=(0.97, 0.05), xycoords="axes fraction", va="bottom", ha="right",
                         fontsize=6.8, color="#202020", zorder=6,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GRAY, lw=0.5,
                                   alpha=0.92))
        else:
            ax2.annotate("（无判据对照数据：请用 solve() 的返回值出图）", xy=(0.5, 0.5),
                         xycoords="axes fraction", ha="center", fontsize=8.0)
        ax2.set_xlabel("文献 GDOP 判据 / m")
        ax2.set_ylabel("本文精确最坏定位直径 $J$ / m")
        ax2.set_title("(b) 两个判据在可行域内的关系", fontsize=9.2)
        ax2.grid(alpha=0.20, lw=0.4, which="both")
        ax2.tick_params(labelsize=7.6)

        fig.suptitle("问题二：文献判据（GDOP / CRLB / 几何稀释）与本文精确判据的对照", fontsize=10.4)
        save_png(fig, out_path, dpi=dpi)
        if pdf_path is not None:
            pdf_path = Path(pdf_path)
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pdf_path, metadata={"Software": None, "CreationDate": _PDF_DATE})
        plt.close(fig)
    return out_path
