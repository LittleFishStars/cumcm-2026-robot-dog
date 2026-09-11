"""逐局轨迹图：把机器狗的行驶轨迹、覆盖圆与测量结果画成图。

绘图在 `/exit` 之后进行（`save_trajectory` 由运行编排在收尾时调用），**不占用现实时间预算**、
也不影响任何实时决策 —— 图上画的就是 `RobotDog.actions` 里记录的实际动作点，可与过程日志
逐点对账。同名 CSV 轨迹表让"图上每个点"都能被逐行复核。

绘制内容：作业圆域 1800 m 与源生成域 1770 m、7 个半径 1000 m 的覆盖圆与圆心（标注访问
序号）、从原点起的行驶路径、按结果分类的动作点（测得示向度 / 无信号 / 近距 / 清除尝试 /
清除成功）、干扰源真值与 20 m 清除半径。

输出目录刻意写成 `<save-dir>/trajectory/t3/`：T3_ga.py（GA 对照方案）会把它的轨迹图写在
`<save-dir>/trajectory/epNN_seedMM.png`，两边同名且同目录会互相覆盖（实测一次性覆盖掉对方
10 个已提交文件），故分开。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cumcm.common.plotting import (C_COVER, C_DIR, C_FRAME, C_HIT, C_NEAR, C_NOSIG,
                                   C_PATH, C_SRC, C_TRY, font_context, save_png,
                                   setup_mpl_env)
from cumcm.t3.config import CLEAR_RADIUS, COVER_RADIUS, REGION_RADIUS, TRAJ_DIR, TRAJ_DPI
from cumcm.t3.covering import CoverPlan


def truth_points(truth: Optional[Sequence[dict]]) -> List[Dict[str, float]]:
    """把引擎的源真值统一成 {channel, x, y}，供绘图使用（raw 引擎格式与核对行都能吃）。"""
    out: List[Dict[str, float]] = []
    for j in truth or []:
        if "position" in j:                       # 引擎原始格式：{"position": {"x": .., "y": ..}}
            pos = j["position"]
            x, y = float(pos["x"]), float(pos["y"])
        else:                                     # 核对行格式：{"x": .., "y": ..}
            x, y = float(j["x"]), float(j["y"])
        out.append({"channel": float(j["channel"]), "x": x, "y": y})
    return out


def draw_trajectory(out_path: Path, actions: Sequence[Dict[str, Any]],
                    plan: CoverPlan, order: Sequence[int] = (),
                    sources: Sequence[Dict[str, Any]] = (),
                    title: Optional[str] = None,
                    figsize: Tuple[float, float] = (9.0, 7.6)) -> Path:
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
            groups = (("direction", dict(marker=".", ms=5, ls="none", color=C_DIR),
                       "测得示向度"),
                      ("no_signal", dict(marker="x", ms=3.5, ls="none", color=C_NOSIG),
                       "无信号"),
                      ("near", dict(marker="o", ms=6, ls="none", mfc="none", mec=C_NEAR,
                                    mew=1.4), "近距 near"))
            for kind, style, label in groups:
                sel = [(a["x"], a["y"]) for a in actions
                       if a["kind"] == "measure" and a["outcome"] == kind]
                if not sel:
                    continue
                arr = np.asarray(sel, dtype=float)
                ax.plot(arr[:, 0], arr[:, 1], zorder=5.0, **style)
                handles.append(Line2D([], [], label=f"{label}（{len(arr)} 次）", **style))
            tries = [(a["x"], a["y"]) for a in actions if a["kind"] == "clear"]
            if tries:
                arr = np.asarray(tries, dtype=float)
                ax.plot(arr[:, 0], arr[:, 1], marker="^", ms=5.5, ls="none", mfc="none",
                        mec=C_TRY, mew=1.2, zorder=6.0)
                handles.append(Line2D([], [], marker="^", ms=5.5, ls="none", mfc="none",
                                      mec=C_TRY, mew=1.2,
                                      label=f"清除尝试（{len(arr)} 次）"))
                ok = np.asarray([(a["x"], a["y"]) for a in actions
                                 if a["kind"] == "clear" and a["outcome"] == "success"],
                                dtype=float)
                if len(ok):
                    # 清除落点必然紧贴真值（20 m 内），故画在最上层才看得见
                    ax.plot(ok[:, 0], ok[:, 1], marker="*", ms=11, ls="none", color=C_HIT,
                            zorder=8.0)
                    handles.append(Line2D([], [], marker="*", ms=11, ls="none",
                                          color=C_HIT, label=f"清除成功（{len(ok)} 个）"))

        # 干扰源真值（仅演练模式）与 20 m 清除半径
        if sources:
            arr = np.asarray([(s["x"], s["y"]) for s in sources], dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], marker="X", ms=8, ls="none", color=C_SRC,
                    zorder=7.0)
            # 每个源一个半径 20 m 的圆：列方向必须是"每个圆一列"，否则会被连成一团
            ax.plot(arr[:, 0][None, :] + CLEAR_RADIUS * cos_th[:, None],
                    arr[:, 1][None, :] + CLEAR_RADIUS * sin_th[:, None],
                    color=C_SRC, lw=0.7, alpha=0.75, zorder=1.5)
            handles.append(Line2D([], [], marker="X", ms=8, ls="none", color=C_SRC,
                                  label=f"干扰源真值（{len(arr)} 个）"))
            handles.append(Line2D([], [], color=C_SRC, lw=0.7, alpha=0.75,
                                  label=f"清除半径 {CLEAR_RADIUS:.0f} m（全图视场 3600 m，需放大才可见）"))

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


def save_trajectory(save_dir: Path, name: str, actions: Sequence[Dict[str, Any]],
                    plan: CoverPlan, order: Sequence[int] = (),
                    sources: Sequence[Dict[str, Any]] = (),
                    title: Optional[str] = None,
                    traj_dir: str = TRAJ_DIR) -> List[Path]:
    """落盘一局的轨迹：同名 PNG（图）与 CSV（轨迹表），返回已写出的文件列表。

    轨迹表让"图上每个点"都能与过程日志逐点对账（序号、动作类型、阶段、坐标、结果、频道、
    虚拟时刻、累计里程）。matplotlib 缺失只提示一次并跳过出图，轨迹表照常写出。
    """
    out_dir = save_dir / traj_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "kind", "stage", "x_m", "y_m", "channel", "outcome",
                    "theta_deg", "virtual_time_s", "travel_m"])
        for a in actions:
            w.writerow([a["seq"], a["kind"], a["stage"], f"{a['x']:.2f}", f"{a['y']:.2f}",
                        a["channel"], a["outcome"] or "",
                        "" if a["theta"] is None else f"{a['theta']:.2f}",
                        f"{a['virtual_time_s']:.3f}", f"{a['travel_m']:.2f}"])
    paths = [csv_path]
    try:
        paths.insert(0, draw_trajectory(out_dir / f"{name}.png", actions, plan, order,
                                        sources, title))
    except ImportError:
        global _PLOT_HINTED
        if not _PLOT_HINTED:
            _PLOT_HINTED = True
            print("提示：未安装 matplotlib，已跳过轨迹图（pip install matplotlib 后可自动生成）")
    return paths


_PLOT_HINTED = False           # 缺 matplotlib 的提示只打印一次，避免每局刷屏
