"""逐局轨迹图：问题三、问题四**共用**的那部分画法与落盘。

为什么只抽一部分：两题的轨迹图骨架相同（圆域 → 本题特有点位 → 行驶路径 → 按结果分类的动作点
→ 真值源 → 起点 → 图例），但"本题特有点位"与图例文字口径不同 —— 问题三画 7 个覆盖圆与圆心
访问序号，问题四画 20 个测量位置与定向源波束扇形。故这里只收**逐字相同**的部分：

* `truth_points`：把引擎的源真值（原始格式 / 核对行格式）统一成绘图用的最小字典；
* `plot_action_points`：按测量结果（测得示向度 / 无信号 / 近距）与清除动作（尝试 / 成功）
  分类画点并生成图例项 —— 分类规则、点型、层次与图例计数两题完全一致，**只有图例文字**由
  各题传入（问题四更强调"测向"二字）；
* `plot_source_markers`：真值源的红叉与每个源一圈的 20 m 清除半径（"每圆一列"的列方向写法
  极易写错，只留一份）；
* `write_trajectory_csv`：轨迹表（11 列的格式与精度两题必须一致，逐点可与过程日志对账）；
* `save_trajectory`：落盘编排（建目录 → 写轨迹表 → 出图 → 缺 matplotlib 时只提示一次）；
* `NO_PLOT_HINT`：缺 matplotlib 的唯一提示文案（原先在两题里写了三遍）。

matplotlib 仍只在绘图函数内导入：官方测试机上没有它也能正常完成整局。
"""

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

# 图例文字的两题口径：分类与画法相同，只是措辞不同（问题四明确写"测向"）
MEASURE_LABELS_T3 = ("测得示向度", "无信号", "近距 near")
MEASURE_LABELS_T4 = ("测向有示向度", "测向无信号", "近距")

# 测量结果 → 点型。两题共用同一套：测得示向度 / 无信号 / 近距，颜色也来自公共配色表
_MEASURE_STYLES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("direction", dict(marker=".", ms=5, ls="none", color=C_DIR)),
    ("no_signal", dict(marker="x", ms=3.5, ls="none", color=C_NOSIG)),
    ("near", dict(marker="o", ms=6, ls="none", mfc="none", mec=C_NEAR, mew=1.4)),
)
_TRY_STYLE: dict[str, Any] = dict(marker="^", ms=5.5, ls="none", mfc="none", mec=C_TRY, mew=1.2)
_HIT_STYLE: dict[str, Any] = dict(marker="*", ms=11, ls="none", color=C_HIT)


def truth_points(truth: Sequence[dict] | None) -> list[dict[str, Any]]:
    """把引擎的源真值统一成 `{channel, x, y, kind, direction_deg}`，供绘图使用。

    两种输入格式都要吃：引擎原始格式 `{"position": {"x": .., "y": ..}}`（演练模式）
    与核对行格式 `{"x": .., "y": ..}`。`direction_deg` 为空即全向源（问题三恒为空）。
    """
    out: list[dict[str, Any]] = []
    for j in truth or []:
        if "position" in j:                       # 引擎原始格式
            pos = j["position"]
        elif "x" in j and "y" in j:               # 核对行格式
            pos = j
        else:
            continue
        d = j.get("direction_deg")
        out.append({"channel": int(j["channel"]), "x": float(pos["x"]), "y": float(pos["y"]),
                    "kind": "directional" if d is not None else "omni",
                    "direction_deg": round(float(d), 2) if d is not None else None})
    return out


def plot_action_points(ax: Any, actions: Sequence[dict[str, Any]], handles: list[Any],
                       measure_labels: Sequence[str] = MEASURE_LABELS_T3) -> None:
    """画本局的动作点（测向按结果分类、清除尝试与成功）并把对应图例项追加到 handles。

    `actions` 是 `RobotDog.actions`（逐次动作记录）：`kind` 为 measure / clear，
    measure 的 `outcome` 决定点型。清除落点必然紧贴真值（20 m 内），故画在最上层才看得见。
    """
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
    ax.plot(arr[:, 0], arr[:, 1], zorder=6.0, **_TRY_STYLE)
    handles.append(Line2D([], [], label=f"清除尝试（{len(arr)} 次）", **_TRY_STYLE))
    ok = np.asarray([(a["x"], a["y"]) for a in actions
                     if a["kind"] == "clear" and a["outcome"] == "success"], dtype=float)
    if len(ok):
        ax.plot(ok[:, 0], ok[:, 1], zorder=8.0, **_HIT_STYLE)
        handles.append(Line2D([], [], label=f"清除成功（{len(ok)} 个）", **_HIT_STYLE))


def plot_source_markers(ax: Any, sources: Sequence[dict[str, Any]],
                        cos_th: np.ndarray, sin_th: np.ndarray,
                        clear_radius: float, ray_len: float = 0.0) -> None:
    """画真值源的红叉（定向源另画一条 ray_len 长的波束方向短射线）与清除半径小圆。

    清除范围那圈的列方向必须是"每个圆一列"（`arr[:, 0][None, :] + r * cos[:, None]`），
    否则所有点会被连成一团 —— 这正是把这段收进公共层的理由。`ray_len = 0`（问题三恒如此，
    全是全向源）时不画射线。
    """
    for s in sources:
        ax.plot(s["x"], s["y"], marker="X", ms=8, ls="none", color=C_SRC, zorder=7.0)
        if ray_len and s.get("kind") == "directional" and s.get("direction_deg") is not None:
            a = math.radians(s["direction_deg"])
            ax.plot([s["x"], s["x"] + ray_len * math.cos(a)],
                    [s["y"], s["y"] + ray_len * math.sin(a)], color=C_SRC, lw=1.2, zorder=7.0)
    arr = np.asarray([(s["x"], s["y"]) for s in sources], dtype=float)
    ax.plot(arr[:, 0][None, :] + clear_radius * cos_th[:, None],
            arr[:, 1][None, :] + clear_radius * sin_th[:, None],
            color=C_SRC, lw=0.7, alpha=0.75, zorder=1.5)


def write_trajectory_csv(csv_path: Path, actions: Sequence[dict[str, Any]]) -> Path:
    """写出轨迹表（列与精度两题一致，逐点可与过程日志对账）。"""
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
    """落盘一局的轨迹：同名 CSV（轨迹表）+ PNG（图），返回已写出的文件列表。

    `draw_png` 由各题传入（各题的图内元素不同，见模块文档）：传"目标 PNG 路径 → 落盘路径"的
    可调用对象。matplotlib 缺失时只提示一次并跳过出图，轨迹表照常写出。
    """
    out_dir = Path(save_dir) / traj_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_trajectory_csv(out_dir / f"{name}.csv", actions)
    paths = [csv_path]
    try:
        paths.insert(0, draw_png(out_dir / f"{name}.png"))
    except ImportError:
        no_plot_hint()
    return paths
