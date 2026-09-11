"""逐局轨迹图（问题四）：行驶轨迹、拖网格点、测量结果与真值波束方向。

绘图在 `/exit` 之后进行，不占用现实时间预算，也不影响实时决策 —— 图上画的就是
`RobotDog.actions` 里的实际动作点，可与过程日志逐点对账；同名 CSV 轨迹表让"图上每个点"
都能被逐行复核。

绘制内容：作业圆域 1800 m 与源生成域 1770 m、拖网格点（700 m 格点，|p| ≤ 2270，含圆域外
470 m）、从原点起的行驶路径、按结果分类的动作点、干扰源真值（**定向源额外画其 ±90° 波束
扇形**，一眼看出"背对波束的点听不到它"）、20 m 清除半径。

逐步扫描图（<save-dir>/scan/ 下）复用 common.scanfigure 的通用画法：覆盖圆信息对问题四无
意义（没有 7 个覆盖圆），故传空，仅保留该步测向点、示向度射线、行驶路径、估计区域与真值。
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cumcm.common.plotting import (C_DIR, C_FRAME, C_GRAY, C_HIT, C_MEAS, C_NEAR, C_NOSIG,
                                   C_PATH, C_SRC, C_TRY, font_context, hint_plot_once,
                                   save_png, setup_mpl_env, slug)
from cumcm.common.scanfigure import STEP_DIR_NAME, ScanStep, draw_scan_step
from cumcm.t4.config import (CLEAR_RADIUS, DIR_BEAM_HALF_DEG, RECEIVE_MAX, REGION_RADIUS,
                             TRAJ_DIR, TRAJ_DPI)
from cumcm.t4.sweep import SweepPlan


def truth_points(truth: Optional[Sequence[dict]]) -> List[Dict[str, Any]]:
    """把引擎源真值统一成 {channel, x, y, kind, direction_deg} 供绘图。"""
    out: List[Dict[str, Any]] = []
    for j in truth or []:
        if "position" in j:
            pos = j["position"]
        elif "x" in j and "y" in j:
            pos = j
        else:
            continue
        d = j.get("direction_deg")
        out.append({"channel": int(j["channel"]), "x": float(pos["x"]), "y": float(pos["y"]),
                    "kind": "directional" if d is not None else "omni",
                    "direction_deg": round(float(d), 2) if d is not None else None})
    return out


def _draw_beams(ax, sources: Sequence[Dict[str, Any]], th: np.ndarray) -> None:
    """定向源画 ±90° 波束扇形（用沿波束方向的两条半径 + 弧近似）。"""
    for s in sources:
        if s.get("kind") != "directional" or s.get("direction_deg") is None:
            continue
        a = math.radians(s["direction_deg"])
        lo, hi = a - math.radians(DIR_BEAM_HALF_DEG), a + math.radians(DIR_BEAM_HALF_DEG)
        xx = np.concatenate([[s["x"]], s["x"] + RECEIVE_MAX * np.cos(np.linspace(lo, hi, 41)),
                             [s["x"]]])
        yy = np.concatenate([[s["y"]], s["y"] + RECEIVE_MAX * np.sin(np.linspace(lo, hi, 41)),
                             [s["y"]]])
        ax.fill(xx, yy, color=C_SRC, alpha=0.10, lw=0.0, zorder=1.2)


def draw_trajectory(out_path: Path, actions: Sequence[Dict[str, Any]],
                    plan: SweepPlan, order: Sequence[int] = (),
                    sources: Sequence[Dict[str, Any]] = (),
                    title: Optional[str] = None) -> Path:
    """画一局的轨迹图并落盘（PNG）。"""
    setup_mpl_env()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with font_context(size=10):
        fig, ax = plt.subplots(figsize=(9.6, 9.2))
        th = np.linspace(0.0, 2.0 * math.pi, 361)
        cos_th, sin_th = np.cos(th), np.sin(th)
        ax.plot(REGION_RADIUS * cos_th, REGION_RADIUS * sin_th, color=C_FRAME, lw=1.4)
        ax.plot((REGION_RADIUS - 30.0) * cos_th, (REGION_RADIUS - 30.0) * sin_th,
                color=C_FRAME, lw=0.8, ls="--", alpha=0.85)
        handles = [Line2D([], [], color=C_FRAME, lw=1.4, label="作业圆域 1800 m"),
                   Line2D([], [], color=C_FRAME, lw=0.8, ls="--", alpha=0.85,
                          label="源生成域 1770 m")]

        # 拖网格点：|p| ≤ 2270 的 700 m 格点中，圆域内 / 外分开标注
        wp = plan.points
        inner = wp[np.linalg.norm(wp, axis=1) <= REGION_RADIUS + 1e-9]
        outer = wp[np.linalg.norm(wp, axis=1) > REGION_RADIUS + 1e-9]
        ax.plot(inner[:, 0], inner[:, 1], marker=".", ms=3.5, ls="none", color=C_GRAY,
                alpha=0.8, zorder=1.5)
        if len(outer):
            ax.plot(outer[:, 0], outer[:, 1], marker=".", ms=3.0, ls="none", color=C_GRAY,
                    alpha=0.45, zorder=1.4)
        handles.append(Line2D([], [], marker=".", ms=3.5, ls="none", color=C_GRAY,
                              label=f"拖网格点 {len(wp)}（含圆域外 {len(outer)}）"))

        # 真值波束扇形（定向源）放最底层
        _draw_beams(ax, sources, th)

        # 行驶路径与动作点
        if actions:
            path = np.asarray([(0.0, 0.0)] + [(a["x"], a["y"]) for a in actions], dtype=float)
            ax.plot(path[:, 0], path[:, 1], color=C_PATH, lw=1.0, alpha=0.9, zorder=4.5)
            handles.append(Line2D([], [], color=C_PATH, lw=1.0,
                                  label=f"行驶路径（{len(actions)} 次动作）"))
            for label, key, style in (("测向有示向度", "direction", dict(
                    marker="o", ms=4.5, ls="none", mfc=C_DIR, mec=C_DIR)),
                    ("测向无信号", "no_signal", dict(marker="o", ms=4.0, ls="none",
                                                     mfc=C_NOSIG, mec=C_NOSIG)),
                    ("近距", "near", dict(marker="o", ms=5.0, ls="none", mfc=C_NEAR,
                                          mec=C_NEAR))):
                sel = [(a["x"], a["y"]) for a in actions
                       if a["kind"] == "measure" and a["outcome"] == key]
                if sel:
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
                    ax.plot(ok[:, 0], ok[:, 1], marker="*", ms=11, ls="none", color=C_HIT,
                            zorder=8.0)
                    handles.append(Line2D([], [], marker="*", ms=11, ls="none", color=C_HIT,
                                          label=f"清除成功（{len(ok)} 个）"))

        if sources:
            arr = np.asarray([(s["x"], s["y"]) for s in sources], dtype=float)
            for s in sources:
                ax.plot(s["x"], s["y"], marker="X", ms=8, ls="none", color=C_SRC, zorder=7.0)
                # 定向源在其波束方向画短射线（标出"它朝哪打"）
                if s.get("kind") == "directional" and s.get("direction_deg") is not None:
                    a = math.radians(s["direction_deg"])
                    ax.plot([s["x"], s["x"] + 260 * math.cos(a)],
                            [s["y"], s["y"] + 260 * math.sin(a)], color=C_SRC, lw=1.2,
                            zorder=7.0)
            ax.plot(arr[:, 0][None, :] + CLEAR_RADIUS * cos_th[:, None],
                    arr[:, 1][None, :] + CLEAR_RADIUS * sin_th[:, None],
                    color=C_SRC, lw=0.7, alpha=0.75, zorder=1.5)
            n_dir = sum(1 for s in sources if s.get("kind") == "directional")
            handles.append(Line2D([], [], marker="X", ms=8, ls="none", color=C_SRC,
                                  label=f"干扰源真值（{len(arr)} 个，定向 {n_dir} 个）"))
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
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
                  ncol=2, fontsize=7.5, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5)
        fig.tight_layout()
        save_png(fig, out_path, dpi=TRAJ_DPI)
        plt.close(fig)
    return out_path


def save_trajectory(save_dir: Path, name: str, actions: Sequence[Dict[str, Any]],
                    plan: SweepPlan, order: Sequence[int] = (),
                    sources: Sequence[Dict[str, Any]] = (),
                    title: Optional[str] = None,
                    traj_dir: str = TRAJ_DIR) -> List[Path]:
    """落盘一局的轨迹：PNG（图）与 CSV（轨迹表），返回已写出的文件列表。"""
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
        hint_plot_once("提示：未安装 matplotlib，已跳过出图。装上即可自动生成：\n"
                       "      .venv/bin/pip install matplotlib")
    return paths


def save_scan_figures(save_dir: Path, name: str, steps: Sequence[Dict[str, Any]],
                      plan: SweepPlan, order: Sequence[int] = (),
                      sources: Sequence[Dict[str, Any]] = (),
                      step_dir: str = STEP_DIR_NAME) -> List[Path]:
    """把一局内**每一步拖网扫描**各画一张结果图，落在 <save-dir>/<step_dir>/ 下。

    文件名形如 `ep01_s00_起点全频道扫描.png`、`ep01_s04_拖网点4.png`，排序后与执行顺序一致。
    问题四没有覆盖圆，故覆盖圆参数一律传空，画面上保留该步测向点、示向度射线、行驶路径、
    当时估计区域与真值（含定向源波束扇形）。
    """
    out_dir = save_dir / step_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for k, raw in enumerate(steps):
        idx = int(raw.get("index", k))
        safe = slug(str(raw.get("label", f"step{k}")))
        out = out_dir / f"{Path(name).name}_s{k:02d}_{safe}.png"
        try:
            draw_scan_step(out, ScanStep(
                index=idx, label=str(raw.get("label", "")),
                x=float(raw["x"]), y=float(raw["y"]),
                n_channels=int(raw.get("n_channels", 0)),
                counts=dict(raw.get("counts", {})),
                virtual_time_s=float(raw.get("virtual_time_s", 0.0)),
                travel_m=float(raw.get("travel_m", 0.0)),
                measures=list(raw.get("measures", [])),
                clears=list(raw.get("clears", [])),
                path=[tuple(map(float, p)) for p in raw.get("path", [])],
                cleared=list(raw.get("cleared", [])),
                regions=raw.get("regions") or None,
                estimates=raw.get("estimates") or None,
            ), cover_centers=(), visit_order=(), visited=(),
                cover_radius=0.0, region_radius=REGION_RADIUS,
                gen_radius=REGION_RADIUS - 30.0, ray_len=RECEIVE_MAX,
                sources=sources, clear_radius=CLEAR_RADIUS,
                title=f"第 {k} 步拖网 / 共 {len(steps)} 步：{raw.get('label', '')}")
            paths.append(out)
        except ImportError:
            hint_plot_once("提示：未安装 matplotlib，已跳过出图。装上即可自动生成：\n"
                           "      .venv/bin/pip install matplotlib")
            return paths
    return paths


def _no_plot_hint() -> None:
    hint_plot_once("提示：未安装 matplotlib，已跳过出图。装上即可自动生成：\n"
                   "      .venv/bin/pip install matplotlib")