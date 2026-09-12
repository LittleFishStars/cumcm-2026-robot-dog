"""逐局轨迹图（问题四）：行驶轨迹、测量位置、测量结果与真值波束方向。

绘图在 `/exit` 之后进行，不占用现实时间预算，也不影响实时决策 —— 图上画的就是
`RobotDog.actions` 里的实际动作点，可与过程日志逐点对账；同名 CSV 轨迹表让"图上每个点"
都能被逐行复核。

绘制内容：作业圆域 1800 m 与源生成域 1770 m、扫描测量位置（原点起点 + 7 覆盖基点 +
3 内部补点 + 12 外圈点）、从原点起的行驶路径、按结果分类的动作点、干扰源真值（**定向源额外画其 ±90°
波束扇形**，一眼看出"背对波束的点听不到它"）、20 m 清除半径。

与问题三共用的那部分画法（动作点分类、真值源与波束方向短射线、清除半径小圆、图例排版、
轨迹表与落盘编排）在 `cumcm.common.trajfigure`；本模块只管问题四独有的元素（测量位置、
定向源 ±90° 波束扇形、图例里"测向"口径的文字）。

逐步扫描图（<save-dir>/scan/ 下）复用 common.scanfigure 的通用画法：覆盖圆信息对问题四无
意义（没有 7 个覆盖圆），故传空，仅保留该步测向点、示向度射线、行驶路径、估计区域与真值。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cumcm.common.plotting import (C_FRAME, C_GRAY, C_PATH, C_SRC, font_context, no_plot_hint,
                                   save_png, setup_mpl_env)
from cumcm.common.scanfigure import (STEP_DIR_NAME, draw_scan_step, scan_figure_path,
                                     scan_step_of)
from cumcm.common.trajfigure import (MEASURE_LABELS_T4, plot_action_points,
                                     plot_source_markers, truth_points)
from cumcm.common.trajfigure import save_trajectory as _save_trajectory
from cumcm.t4.config import (CLEAR_RADIUS, DIR_BEAM_HALF_DEG, RECEIVE_MAX, REGION_RADIUS,
                             TRAJ_DIR, TRAJ_DPI)
from cumcm.t4.sweep import SweepPlan

__all__ = ["truth_points", "draw_trajectory", "save_trajectory", "save_scan_figures"]

# 定向源在其波束方向上画的短射线长度 / m：只用来标出"它朝哪打"，太长会与示向度射线混淆
_DIR_RAY_M = 260.0


def _draw_beams(ax: "Axes", sources: Sequence[dict[str, Any]], th: np.ndarray) -> None:
    """定向源画 ±90° 波束扇形（用沿波束方向的两条半径 + 弧近似）

    Args:
        ax: matplotlib 轴对象
        sources: 真值源列表（定向源须带 kind="directional" 与 direction_deg）
        th: 圆周角度采样数组（统一绘图辅助函数签名保留，本函数未用到）
    """
    for s in sources:
        if s.get("kind") != "directional" or s.get("direction_deg") is None:
            continue
        a = math.radians(s["direction_deg"])
        lo, hi = a - math.radians(DIR_BEAM_HALF_DEG), a + math.radians(DIR_BEAM_HALF_DEG)
        # 扇形边界折线 = 圆心 → 波束角范围内的圆弧（半径取接收半径上界）→ 回到圆心
        xx = np.concatenate([[s["x"]], s["x"] + RECEIVE_MAX * np.cos(np.linspace(lo, hi, 41)),
                             [s["x"]]])
        yy = np.concatenate([[s["y"]], s["y"] + RECEIVE_MAX * np.sin(np.linspace(lo, hi, 41)),
                             [s["y"]]])
        ax.fill(xx, yy, color=C_SRC, alpha=0.10, lw=0.0, zorder=1.2)


def draw_trajectory(out_path: Path, actions: Sequence[dict[str, Any]],
                    plan: SweepPlan, order: Sequence[int] = (),
                    sources: Sequence[dict[str, Any]] = (),
                    title: str | None = None) -> Path:
    """画一局的轨迹图并落盘（PNG）。"""
    setup_mpl_env()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with font_context(size=10):
        fig, ax = plt.subplots(figsize=(9.0, 7.6))
        th = np.linspace(0.0, 2.0 * math.pi, 361)
        cos_th, sin_th = np.cos(th), np.sin(th)
        ax.plot(REGION_RADIUS * cos_th, REGION_RADIUS * sin_th, color=C_FRAME, lw=1.4)
        ax.plot((REGION_RADIUS - 30.0) * cos_th, (REGION_RADIUS - 30.0) * sin_th,
                color=C_FRAME, lw=0.8, ls="--", alpha=0.85)
        handles = [Line2D([], [], color=C_FRAME, lw=1.4, label="作业圆域 1800 m"),
                   Line2D([], [], color=C_FRAME, lw=0.8, ls="--", alpha=0.85,
                          label="源生成域 1770 m")]

        # 扫描测量位置：圆域内（原点 + 7 基点 + 3 内部补点）/ 圆域外（12 外圈点）分开标注
        wp = plan.points
        inner = wp[np.linalg.norm(wp, axis=1) <= REGION_RADIUS + 1e-9]
        outer = wp[np.linalg.norm(wp, axis=1) > REGION_RADIUS + 1e-9]
        ax.plot(inner[:, 0], inner[:, 1], marker=".", ms=3.5, ls="none", color=C_GRAY,
                alpha=0.8, zorder=1.5)
        if len(outer):
            ax.plot(outer[:, 0], outer[:, 1], marker=".", ms=3.0, ls="none", color=C_GRAY,
                    alpha=0.45, zorder=1.4)
        handles.append(Line2D([], [], marker=".", ms=3.5, ls="none", color=C_GRAY,
                              label=f"扫描测量位置 {len(wp)}（含圆域外 {len(outer)} 个）"))

        # 真值波束扇形（定向源）放最底层
        _draw_beams(ax, sources, th)

        # 行驶路径与动作点
        if actions:
            path = np.asarray([(0.0, 0.0)] + [(a["x"], a["y"]) for a in actions], dtype=float)
            ax.plot(path[:, 0], path[:, 1], color=C_PATH, lw=1.0, alpha=0.9, zorder=4.5)
            handles.append(Line2D([], [], color=C_PATH, lw=1.0,
                                  label=f"行驶路径（{len(actions)} 次动作）"))
            # 动作点按结果分类（测向有示向度 / 测向无信号 / 近距 / 清除尝试 / 清除成功）见公共画法
            plot_action_points(ax, actions, handles, MEASURE_LABELS_T4)

        if sources:
            plot_source_markers(ax, sources, cos_th, sin_th, CLEAR_RADIUS, ray_len=_DIR_RAY_M)
            n_dir = sum(1 for s in sources if s.get("kind") == "directional")
            handles.append(Line2D([], [], marker="X", ms=8, ls="none", color=C_SRC,
                                  label=f"干扰源真值（{len(sources)} 个，定向 {n_dir} 个）"))
            handles.append(Line2D([], [], color=C_SRC, lw=0.7, alpha=0.75,
                                  label=f"清除半径 {CLEAR_RADIUS:.0f} m"))
            if n_dir:
                handles.append(Line2D([], [], color=C_SRC, lw=1.2, alpha=0.5,
                                      label="定向源波束方向（±90° 见填充）"))

        ax.plot(0.0, 0.0, marker="s", ms=6, color="black")
        handles.append(Line2D([], [], marker="s", ms=6, ls="none", color="black",
                              label="起点（原点）"))
        ax.set_aspect("equal")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        if title:
            ax.set_title(title, fontsize=10)
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07),
                  ncol=3, fontsize=7.5, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5)
        fig.tight_layout()
        save_png(fig, out_path, dpi=TRAJ_DPI)
        plt.close(fig)
    return out_path


def save_trajectory(save_dir: Path, name: str, actions: Sequence[dict[str, Any]],
                    plan: SweepPlan, order: Sequence[int] = (),
                    sources: Sequence[dict[str, Any]] = (),
                    title: str | None = None,
                    traj_dir: str = TRAJ_DIR) -> list[Path]:
    """落盘一局的轨迹：PNG（图）与 CSV（轨迹表），返回已写出的文件列表。"""
    return _save_trajectory(
        save_dir, name, actions,
        lambda png: draw_trajectory(png, actions, plan, order, sources, title),
        traj_dir)


def save_scan_figures(save_dir: Path, name: str, steps: Sequence[dict[str, Any]],
                      plan: SweepPlan, order: Sequence[int] = (),
                      sources: Sequence[dict[str, Any]] = (),
                      step_dir: str = STEP_DIR_NAME) -> list[Path]:
    """把一局内**每一步扫描测量**各画一张结果图，落在 <save-dir>/<step_dir>/ 下。

    文件名形如 `ep01_s00_起点全频道扫描.png`、`ep01_s04_测量位置4.png`，排序后与执行顺序一致。
    问题四没有覆盖圆，故覆盖圆参数一律传空，画面上保留该步测向点、示向度射线、行驶路径、
    当时估计区域与真值（含定向源波束扇形）。
    """
    out_dir = save_dir / step_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for k, raw in enumerate(steps):
        out = scan_figure_path(out_dir, name, k, raw)
        try:
            draw_scan_step(out, scan_step_of(raw, k), cover_centers=(), visit_order=(),
                           visited=(), cover_radius=0.0, region_radius=REGION_RADIUS,
                           gen_radius=REGION_RADIUS - 30.0, ray_len=RECEIVE_MAX,
                           sources=sources, clear_radius=CLEAR_RADIUS,
                           title=f"第 {k} 步扫描 / 共 {len(steps)} 步：{raw.get('label', '')}")
            paths.append(out)
        except ImportError:
            no_plot_hint()
            return paths
    return paths
