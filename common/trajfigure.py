"""逐局轨迹图：问题三、问题四共用的那部分画法与落盘"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from common.plotting import (C_DIR, C_HIT, C_NEAR, C_NOSIG, C_SRC, C_TRY,
                                   no_plot_hint)

__all__ = ["truth_points", "plot_action_points", "plot_source_markers",
           "write_trajectory_csv", "save_trajectory", "MEASURE_LABELS_T3", "MEASURE_LABELS_T4"]

# 图例文字的两题口径：分类与画法相同，只是措辞不同，问题四明确写"测向"
MEASURE_LABELS_T3 = ("测得示向度", "无信号", "近距 near")
MEASURE_LABELS_T4 = ("测向有示向度", "测向无信号", "近距")

# 测量结果到点型的映射。两题共用这一套：测得示向度 / 无信号 / 近距，颜色也取自公共配色表
_MEASURE_STYLES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("direction", dict(marker=".", ms=5, ls="none", color=C_DIR)),
    ("no_signal", dict(marker="x", ms=3.5, ls="none", color=C_NOSIG)),
    ("near", dict(marker="o", ms=6, ls="none", mfc="none", mec=C_NEAR, mew=1.4)),
)
_TRY_STYLE: dict[str, Any] = dict(marker="^", ms=5.5, ls="none", mfc="none", mec=C_TRY, mew=1.2)
_HIT_STYLE: dict[str, Any] = dict(marker="*", ms=11, ls="none", color=C_HIT)


# 本模块只收两题逐字相同的那部分画法，点位与图例文字由调用方传入
def truth_points(truth: Sequence[dict] | None) -> list[dict[str, Any]]:
    """把引擎的源真值统一成 `{channel, x, y, kind, direction_deg}`，供绘图使用"""
    out: list[dict[str, Any]] = []
    for j in truth or []:
        # 两种输入格式都要吃：引擎的 {"position": {...}}，以及核对行的 {"x": .., "y": ..}
        if "position" in j:                       # 引擎原始格式
            pos = j["position"]
        elif "x" in j and "y" in j:               # 核对行格式
            pos = j
        else:
            continue
        d = j.get("direction_deg")                # 为空就是全向源，问题三恒为空
        out.append({"channel": int(j["channel"]), "x": float(pos["x"]), "y": float(pos["y"]),
                    "kind": "directional" if d is not None else "omni",
                    "direction_deg": round(float(d), 2) if d is not None else None})
    return out


def plot_action_points(ax: Any, actions: Sequence[dict[str, Any]], handles: list[Any],
                       measure_labels: Sequence[str] = MEASURE_LABELS_T3) -> None:
    """画本局的动作点，测向按结果分类，清除分尝试与成功，并把图例项追加到 handles"""
    from matplotlib.lines import Line2D

    for (kind, style), label in zip(_MEASURE_STYLES, measure_labels):
        sel = [(a["x"], a["y"]) for a in actions
               if a["kind"] == "measure" and a["outcome"] == kind]
        if not sel:
            continue
        arr = np.asarray(sel, dtype=float)
        ax.plot(arr[:, 0], arr[:, 1], zorder=5.0, **style)
        handles.append(Line2D([], [], label=f"{label}（{len(arr)} 次）", **style))

    tries = [(a["x"], a["y"]) for a in actions if a["kind"] == "clear"]
    if not tries:
        return
    arr = np.asarray(tries, dtype=float)
    ax.plot(arr[:, 0], arr[:, 1], zorder=6.0, **_TRY_STYLE)      # 清除落点紧贴真值，要压在上层
    handles.append(Line2D([], [], label=f"清除尝试（{len(arr)} 次）", **_TRY_STYLE))
    ok = np.asarray([(a["x"], a["y"]) for a in actions
                     if a["kind"] == "clear" and a["outcome"] == "success"], dtype=float)
    if len(ok):
        ax.plot(ok[:, 0], ok[:, 1], zorder=8.0, **_HIT_STYLE)
        handles.append(Line2D([], [], label=f"清除成功（{len(ok)} 个）", **_HIT_STYLE))


def plot_source_markers(ax: Any, sources: Sequence[dict[str, Any]],
                        cos_th: np.ndarray, sin_th: np.ndarray,
                        clear_radius: float, ray_len: float = 0.0) -> None:
    """画真值源的红叉、定向源那条 ray_len 长的波束短射线，以及清除半径小圆"""
    for s in sources:
        ax.plot(s["x"], s["y"], marker="X", ms=8, ls="none", color=C_SRC, zorder=7.0)
        if ray_len and s.get("kind") == "directional" and s.get("direction_deg") is not None:
            a = math.radians(s["direction_deg"])
            ax.plot([s["x"], s["x"] + ray_len * math.cos(a)],
                    [s["y"], s["y"] + ray_len * math.sin(a)], color=C_SRC, lw=1.2, zorder=7.0)
    arr = np.asarray([(s["x"], s["y"]) for s in sources], dtype=float)
    # 列方向必须是"每个圆一列"，写成 arr[:, 0][None, :] + r * cos[:, None]，否则所有点连成一团
    ax.plot(arr[:, 0][None, :] + clear_radius * cos_th[:, None],
            arr[:, 1][None, :] + clear_radius * sin_th[:, None],
            color=C_SRC, lw=0.7, alpha=0.75, zorder=1.5)


def write_trajectory_csv(csv_path: Path, actions: Sequence[dict[str, Any]]) -> Path:
    """写出轨迹表，列与精度两题一致，逐点可与过程日志对账"""
    with Path(csv_path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "kind", "stage", "x_m", "y_m", "channel", "outcome",
                    "theta_deg", "virtual_time_s", "travel_m"])
        for a in actions:
            w.writerow([a["seq"], a["kind"], a["stage"], f"{a['x']:.2f}", f"{a['y']:.2f}",
                        a["channel"], a["outcome"] or "",
                        "" if a["theta"] is None else f"{a['theta']:.2f}",
                        f"{a['virtual_time_s']:.3f}", f"{a['travel_m']:.2f}"])
    return Path(csv_path)


def save_trajectory(save_dir: Path, name: str, actions: Sequence[dict[str, Any]],
                    draw_png: Callable[[Path], Path], traj_dir: str) -> list[Path]:
    """落盘一局的轨迹：同名 CSV 轨迹表加 PNG 图，返回已写出的文件列表"""
    out_dir = Path(save_dir) / traj_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_trajectory_csv(out_dir / f"{name}.csv", actions)
    paths = [csv_path]
    try:
        paths.insert(0, draw_png(out_dir / f"{name}.png"))
    except ImportError:
        no_plot_hint()
    return paths
