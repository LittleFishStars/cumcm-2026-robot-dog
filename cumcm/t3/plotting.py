"""逐局轨迹图：把机器狗的行驶轨迹、覆盖圆与测量结果画成图。

绘图在 `/exit` 之后进行（`save_trajectory` 由运行编排在收尾时调用），**不占用现实时间预算**、
也不影响任何实时决策 —— 图上画的就是 `RobotDog.actions` 里记录的实际动作点，可与过程日志
逐点对账。同名 CSV 轨迹表让"图上每个点"都能被逐行复核。

绘制内容：作业圆域 1800 m 与源生成域 1770 m、7 个半径 1000 m 的覆盖圆与圆心（标注访问
序号）、从原点起的行驶路径、按结果分类的动作点（测得示向度 / 无信号 / 近距 / 清除尝试 /
清除成功）、干扰源真值与 20 m 清除半径。

与问题四共用的那部分画法（动作点分类、真值源、清除半径小圆、图例排版、轨迹表与落盘编排）
在 `cumcm.common.trajfigure`；本模块只管问题三独有的元素（7 个覆盖圆与圆心访问序号）与
问题三口径的图例文字。

输出目录是 `<save-dir>/trajectory/`，落在 results/t3/ 结果树下。
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cumcm.common.plotting import (C_COVER, C_FRAME, C_PATH, C_SRC, font_context, no_plot_hint,
                                   save_png, setup_mpl_env)
from cumcm.common.scanfigure import (STEP_DIR_NAME, draw_scan_step, scan_figure_path,
                                     scan_step_of)
from cumcm.common.trajfigure import (MEASURE_LABELS_T3, plot_action_points,
                                     plot_source_markers, truth_points)
from cumcm.common.trajfigure import save_trajectory as _save_trajectory
from cumcm.t3.config import (CLEAR_RADIUS, COVER_RADIUS, RECEIVE_MAX, REGION_RADIUS,
                             TRAJ_DIR, TRAJ_DPI)
from cumcm.t3.covering import CoverPlan

__all__ = ["truth_points", "draw_trajectory", "save_trajectory", "save_scan_figures"]


def draw_trajectory(out_path: Path, actions: Sequence[dict[str, Any]],
                    plan: CoverPlan, order: Sequence[int] = (),
                    sources: Sequence[dict[str, Any]] = (),
                    title: str | None = None,
                    figsize: tuple[float, float] = (9.0, 7.6)) -> Path:
    """把一局的轨迹画成图并存盘（格式由后缀决定，.png / .pdf）。

    - `actions`：逐次动作记录（RobotDog.actions），含测向与清除的落点、结果类型、虚拟时刻；
    - `plan`：覆盖圆方案，用来画 7 个半径 1000 m 的覆盖圆与圆心（巡视航路点）；
    - `order`：巡视访问顺序，用于给圆心标序号，直观看出"依次到圆心"的路线；
    - `sources`：干扰源真值（仅演练模式有），画成红叉并按 20 m 清除半径画圈；
    - `title`：图内小标题。论文用图传 None（大标题交给 caption）。

    matplotlib 只在本函数内导入，故未安装时只会抛 ImportError、由调用方忽略 —— 官方测试机上
    没有 matplotlib 也能正常完成整局。出图去掉时间戳类元数据，同一输入两次出图逐字节一致。
    """
    # 配置目录与中文字体的统一处理见 cumcm.common.plotting（此处只调用）
    setup_mpl_env()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with font_context(size=10):
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots(figsize=figsize)
        th = np.linspace(0.0, 2.0 * math.pi, 361)
        cos_th, sin_th = np.cos(th), np.sin(th)

        # 作业圆域 1800 m 与源生成域 1770 m
        ax.plot(REGION_RADIUS * cos_th, REGION_RADIUS * sin_th,
                color=C_FRAME, lw=1.5)
        gen_r = REGION_RADIUS - 30.0
        ax.plot(gen_r * cos_th, gen_r * sin_th, color=C_FRAME, lw=0.8, ls="--", alpha=0.85)

        # 7 个覆盖圆（半径 1000 m）与圆心：逐圆画但不逐个进图例（否则图例会被撑爆）
        wp = np.asarray(plan.waypoints, dtype=float)
        for cx, cy in wp:
            ax.plot(cx + COVER_RADIUS * cos_th, cy + COVER_RADIUS * sin_th,
                    color=C_COVER, lw=0.7, alpha=0.5, zorder=1.0)
        ax.plot(wp[:, 0], wp[:, 1], marker="o", ms=4.0, ls="none", mfc="none",
                mec=C_COVER, mew=1.2)
        for k, idx in enumerate(order or range(len(wp))):
            ax.annotate(f"{k + 1}", (wp[idx, 0], wp[idx, 1]), textcoords="offset points",
                        xytext=(5, 4), fontsize=8, color=C_COVER)

        # 行驶路径与各类动作点
        handles = [Line2D([], [], color=C_FRAME, lw=1.5,
                          label=f"作业圆域 {REGION_RADIUS:.0f} m"),
                   Line2D([], [], color=C_FRAME, lw=0.8, ls="--", alpha=0.85,
                          label=f"干扰源生成域 {gen_r:.0f} m"),
                   Line2D([], [], color=C_COVER, lw=0.7, alpha=0.5,
                          label=f"{len(wp)} 个覆盖圆（半径 {COVER_RADIUS:.0f} m）"),
                   Line2D([], [], marker="o", ms=4.0, ls="none", mfc="none", mec=C_COVER,
                          mew=1.2, label="覆盖圆圆心（航路点，标注访问序号）")]
        if actions:
            xs = [a["x"] for a in actions]
            ys = [a["y"] for a in actions]
            ax.plot([0.0] + xs, [0.0] + ys, "-", lw=1.0, color=C_PATH, alpha=0.8,
                    zorder=4.0)
            handles.append(Line2D([], [], color=C_PATH, lw=1.0,
                                  label=f"行驶路径（{len(actions)} 次动作）"))
            # 动作点按结果分类（测得示向度 / 无信号 / 近距 / 清除尝试 / 清除成功）见公共画法
            plot_action_points(ax, actions, handles, MEASURE_LABELS_T3)

        # 干扰源真值（仅演练模式）与 20 m 清除半径
        if sources:
            plot_source_markers(ax, sources, cos_th, sin_th, CLEAR_RADIUS)
            handles.append(Line2D([], [], marker="X", ms=8, ls="none", color=C_SRC,
                                  label=f"干扰源真值（{len(sources)} 个）"))
            handles.append(Line2D([], [], color=C_SRC, lw=0.7, alpha=0.75,
                                  label=f"清除半径 {CLEAR_RADIUS:.0f} m（全图视场 3600 m，"
                                        f"需放大才可见）"))

        ax.plot(0.0, 0.0, marker="s", ms=6, color="black")
        handles.append(Line2D([], [], marker="s", ms=6, ls="none", color="black",
                              label="起点（原点）"))
        ax.set_aspect("equal")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        if title:
            ax.set_title(title, fontsize=10)
        # 图例放到坐标轴下方：图内 3600 m 见方的圆域几乎没有空白，放进去必然压住轨迹
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07),
                  ncol=3, fontsize=7.5, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5)
        fig.tight_layout()
        save_png(fig, out_path, dpi=TRAJ_DPI)
        plt.close(fig)
    return out_path


def save_trajectory(save_dir: Path, name: str, actions: Sequence[dict[str, Any]],
                    plan: CoverPlan, order: Sequence[int] = (),
                    sources: Sequence[dict[str, Any]] = (),
                    title: str | None = None,
                    traj_dir: str = TRAJ_DIR) -> list[Path]:
    """落盘一局的轨迹：同名 PNG（图）与 CSV（轨迹表），返回已写出的文件列表。

    轨迹表让"图上每个点"都能与过程日志逐点对账（序号、动作类型、阶段、坐标、结果、频道、
    虚拟时刻、累计里程）。matplotlib 缺失只提示一次并跳过出图，轨迹表照常写出。
    """
    return _save_trajectory(
        save_dir, name, actions,
        lambda png: draw_trajectory(png, actions, plan, order, sources, title),
        traj_dir)


def save_scan_figures(save_dir: Path, name: str, steps: Sequence[dict[str, Any]],
                      plan: CoverPlan, order: Sequence[int] = (),
                      sources: Sequence[dict[str, Any]] = (),
                      step_dir: str = STEP_DIR_NAME) -> list[Path]:
    """把一局内**每一步扫描**各画一张结果图，落在 <save-dir>/<step_dir>/ 下。

    文件名形如 `ep01_s00_起点全频道扫描.png`、`ep01_s03_巡视站3.png`：局号 + 步序，排序后
    与执行顺序一致，便于按时间顺次翻阅"信息是怎么一步步积累起来的"。
    与轨迹图一样在 /exit 之后调用，不占现实时间预算；缺 matplotlib 只提示一次并跳过。
    """
    out_dir = save_dir / step_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    wp = [tuple(map(float, c)) for c in plan.waypoints]
    # 各站"已访问"集合：到第 i 步时，计划顺序里前 i 个站已走过（起点扫描不含任何站）
    visited: list[int] = []
    paths: list[Path] = []
    for k, raw in enumerate(steps):
        idx = int(raw.get("index", k))
        if idx > 0:                                  # 第 idx 个巡视站访问完，加进已访问集合
            # 标签里带了圆心号（六边形族下"第 1 站"可能就是原点站），故从标签里取圆心编号
            wp_i = _waypoint_of_label(str(raw.get("label", "")), order, idx)
            if wp_i is not None and wp_i not in visited:
                visited.append(wp_i)
        # 文件名只用 ASCII 与安全字符，避免不同文件系统下的编码问题
        out = scan_figure_path(out_dir, name, k, raw)
        try:
            draw_scan_step(out, scan_step_of(raw, k), cover_centers=wp, visit_order=list(order),
                           visited=list(visited), cover_radius=COVER_RADIUS,
                           region_radius=REGION_RADIUS, gen_radius=REGION_RADIUS - 30.0,
                           ray_len=RECEIVE_MAX, sources=sources, clear_radius=CLEAR_RADIUS,
                           title=f"第 {k} 步扫描 / 共 {len(steps)} 步：{raw.get('label', '')}")
            paths.append(out)
        except ImportError:
            no_plot_hint()
            return paths
    return paths


def _waypoint_of_label(label: str, order: Sequence[int], step_i: int) -> int | None:
    """从步骤标签里取出圆心编号（标签形如 "巡视站 3（圆心 5）"）。取不到时按顺序退推。"""
    m = re.search(r"圆心\s*(\d+)", label)
    if m:
        return int(m.group(1))
    if 1 <= step_i <= len(order):
        return int(order[step_i - 1])
    return None
