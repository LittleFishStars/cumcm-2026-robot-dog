"""单步扫描结果图：把"某一步扫描"看到的信息单独画成一张图。

**为什么要逐步骤出图**：一整局的轨迹图信息密度太高 —— 上百次测向、几十次清除尝试全部挤在
一张 3600 m 见方的圆域里，看不清"某一步到底听到了什么、哪些频道还是空白、当前估计收缩到
什么程度"。而策略的全部信息都来自这一步步扫描，所以每步单独出图既能讲清"信息是怎么积累
起来的"，也便于逐站复核（图上每个点都能与过程日志、api_calls.jsonl 对上）。

**一步扫描是什么**：机器狗停在某个位置，把"尚未采够示向度且未清除"的频道按频道号升序测
一遍。T3 的一步 = 起点全频道扫描 + 每个巡视站各一次。

本模块只负责画，不负责取数：调用方把自己的记录整理成 `ScanStep` 传进来，两族画法共用同一套
视觉约定。

matplotlib 只在绘图函数内导入，故未安装时只会抛 `ImportError`、由调用方忽略 ——
官方测试机上没有 matplotlib 也能正常完成整局（`.venv/bin/pip install matplotlib` 即可启用）。
出图去掉时间戳类元数据，同一输入两次出图逐字节一致。
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cumcm.common.plotting import (C_COVER, C_DIR, C_FRAME, C_HIT, C_MEAS, C_NEAR,
                                   C_NOSIG, C_PATH, C_SRC, C_TRY, font_context, save_png,
                                   setup_mpl_env)

__all__ = ["ScanStep", "draw_scan_step", "reset_dir", "STEP_DIR_NAME"]

STEP_DIR_NAME = "scan"          # 扫描图落在 <save-dir>/scan/ 下（与 trajectory/ 并列）

# 画图的几何参数由调用方从各自 config 传入，本模块不绑定任何一族的常量
Point = Tuple[float, float]


@dataclass
class ScanStep:
    """一步扫描的完整记录（画图所需的最小契约）。

    `measures` / `clears` 只描述**本步**发生的动作；`path` 是到本步为止的行驶路径（含起点），
    用于给这一步提供空间上下文；`regions` / `estimates` 是本步结束时的估计状态快照。
    所有这些字段都能与 `api_calls.jsonl` 里的逐次调用对上，故图可反向核对。
    """

    index: int                                  # 0 = 起始扫描；1..N = 巡视站序号
    label: str                                  # 展示名，如 "起点全频道扫描"
    x: float
    y: float
    n_channels: int
    counts: Dict[str, int] = field(default_factory=dict)
    virtual_time_s: float = 0.0
    travel_m: float = 0.0
    measures: List[Dict[str, Any]] = field(default_factory=list)
    clears: List[Dict[str, Any]] = field(default_factory=list)
    path: List[Point] = field(default_factory=list)
    cleared: List[int] = field(default_factory=list)
    regions: Optional[Dict[int, Sequence[Point]]] = None      # 频道 → 多边形的外环顶点
    estimates: Optional[Dict[int, Tuple[float, float, float]]] = None   # 频道 → (x, y, σ)

    def summary(self) -> str:
        """一行摘要（图内副标题与终端输出共用，保证图上文字与日志口径一致）。"""
        c = self.counts
        parts = [f"测向 {len(self.measures)} 次"]
        if c.get("direction"):
            parts.append(f"有示向度 {c['direction']}")
        if c.get("no_signal"):
            parts.append(f"无信号 {c['no_signal']}")
        if c.get("near"):
            parts.append(f"近距 {c['near']}")
        if c.get("skip"):
            parts.append(f"判定必无信号跳过 {c['skip']}")
        parts.append(f"累计清除 {len(self.cleared)}")
        parts.append(f"虚拟 {self.virtual_time_s:.0f} s")
        parts.append(f"里程 {self.travel_m:.0f} m")
        return "，".join(parts)


def reset_dir(path: Path) -> None:
    """清空一个输出目录（只保留最新一局用；目录不存在则什么都不做）。

    逐局重画时先清空，避免上一局的图与新图混在同一目录里 —— 文件名带局号，肉眼很难分辨
    哪张是这一轮的。只删该目录自身，不动 <save-dir> 下的 json / csv 结果表。
    """
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)


def draw_scan_step(out_path: Path, step: ScanStep, *,
                   cover_centers: Sequence[Point] = (),
                   visit_order: Sequence[int] = (),
                   visited: Sequence[int] = (),
                   cover_radius: float = 1000.0,
                   region_radius: float = 1800.0,
                   gen_radius: Optional[float] = None,
                   ray_len: float = 1500.0,
                   sources: Sequence[Dict[str, Any]] = (),
                   clear_radius: float = 20.0,
                   title: Optional[str] = None,
                   fonts: Optional[Sequence[str]] = None,
                   figsize: Tuple[float, float] = (9.2, 7.8)) -> Path:
    """画一步扫描的结果图并存盘（格式由后缀决定，.png / .pdf）。

    - `step`：本步记录（见 `ScanStep`）；
    - `cover_centers` / `visit_order` / `visited`：覆盖圆布局、计划巡视顺序、到本步为止已访问
      的站序号 —— 用来标出"路线走到哪了"，并把当前站突出显示；
    - `ray_len`：示向度的射线长度（取接收半径上限，画出来就是"源必在这条射线方向上"）；
    - `sources`：干扰源真值（仅演练模式有）；官方模式拿不到真值，传空即可；
    - `fonts`：中文字体候选链。传各自使用的链以保持与既有产物一致；不传则用公共默认链。

    图上元素：作业圆域、各覆盖圆、已访问/未访问圆心（当前站加粗）、到本步的行驶路径、
    本步测向点（按结果分类）与示向度射线、本步近距清除、本步结束时的估计（区域轮廓或 σ 圆）、
    干扰源真值。
    """
    setup_mpl_env()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with font_context(size=10, fonts=fonts):
        fig, ax = plt.subplots(figsize=figsize)
        th = np.linspace(0.0, 2.0 * math.pi, 361)
        cos_th, sin_th = np.cos(th), np.sin(th)

        # 作业圆域与源生成域
        ax.plot(region_radius * cos_th, region_radius * sin_th, color=C_FRAME, lw=1.4)
        handles = [Line2D([], [], color=C_FRAME, lw=1.4,
                          label=f"作业圆域 {region_radius:.0f} m")]
        if gen_radius:
            ax.plot(gen_radius * cos_th, gen_radius * sin_th, color=C_FRAME, lw=0.8,
                    ls="--", alpha=0.85)
            handles.append(Line2D([], [], color=C_FRAME, lw=0.8, ls="--", alpha=0.85,
                                  label=f"源生成域 {gen_radius:.0f} m"))

        # 覆盖圆：本步所在的站用粗线强调，其余细线
        wp = np.asarray(cover_centers, dtype=float) if len(cover_centers) else np.zeros((0, 2))
        cur_idx = -1
        if len(wp):
            for i, (cx, cy) in enumerate(wp):
                if float(np.hypot(cx - step.x, cy - step.y)) <= 1.0:
                    cur_idx = i
                ax.plot(cx + cover_radius * cos_th, cy + cover_radius * sin_th,
                        color=C_COVER, lw=0.6, alpha=0.42, zorder=1.0)
            if cur_idx >= 0:                       # 当前站所在的覆盖圆描粗，一眼看出人在哪
                cx, cy = wp[cur_idx]
                ax.plot(cx + cover_radius * cos_th, cy + cover_radius * sin_th,
                        color=C_COVER, lw=1.6, alpha=0.9, zorder=1.2)
            if len(visited):
                vis = wp[np.asarray(list(visited), dtype=int)]
                ax.plot(vis[:, 0], vis[:, 1], marker="o", ms=5.0, ls="none",
                        mfc=C_COVER, mec=C_COVER, alpha=0.75, zorder=3.0)
            ax.plot(wp[:, 0], wp[:, 1], marker="o", ms=4.5, ls="none", mfc="none",
                    mec=C_COVER, mew=1.2, zorder=3.0)
            for k, idx in enumerate(visit_order or range(len(wp))):
                ax.annotate(f"{k + 1}", (wp[idx, 0], wp[idx, 1]), textcoords="offset points",
                            xytext=(5, 4), fontsize=8, color=C_COVER)
            handles.append(Line2D([], [], color=C_COVER, lw=0.6, alpha=0.42,
                                  label=f"{len(wp)} 个覆盖圆（半径 {cover_radius:.0f} m）"))
            handles.append(Line2D([], [], marker="o", ms=5.0, ls="none", mfc=C_COVER,
                                  mec=C_COVER, alpha=0.75, label="已访问的圆心"))

        # 到本步为止的行驶路径
        if step.path:
            arr = np.asarray(step.path, dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], "-", lw=1.0, color=C_PATH, alpha=0.75, zorder=4.0)
            handles.append(Line2D([], [], color=C_PATH, lw=1.0,
                                  label=f"到本步的行驶路径（累计 {step.travel_m:.0f} m）"))

        # 本步的测向点，按结果分类；示向度另画一条射线（源必在该方向上）
        groups = (("direction", dict(marker="o", ms=6.0, ls="none", color=C_DIR), "有示向度"),
                  ("no_signal", dict(marker="x", ms=5.0, ls="none", color=C_NOSIG), "无信号"),
                  ("near", dict(marker="o", ms=7.0, ls="none", mfc="none", mec=C_NEAR,
                                mew=1.5), "近距 near"))
        for kind, style, label in groups:
            sel = [m for m in step.measures if m.get("outcome") == kind]
            if not sel:
                continue
            arr = np.asarray([(step.x, step.y)] * len(sel), dtype=float)   # 本步都在同一站
            ax.plot(arr[:, 0], arr[:, 1], zorder=6.0, **style)
            handles.append(Line2D([], [], label=f"{label}（{len(sel)} 次）", **style))
        rays = [m for m in step.measures
                if m.get("outcome") == "direction" and m.get("theta") is not None]
        if rays:
            # theta 为示向度（度）：从测量点沿该方向画到接收半径上限，直观看出"这条约束
            # 把源限制在哪条射线上"。逐条画线但不逐条进图例，否则图例会被撑爆。
            for m in rays:
                rad = math.radians(float(m["theta"]))
                ax.plot([step.x, step.x + ray_len * math.cos(rad)],
                        [step.y, step.y + ray_len * math.sin(rad)],
                        color=C_DIR, lw=0.8, alpha=0.55, zorder=5.0)
            handles.append(Line2D([], [], color=C_DIR, lw=0.8, alpha=0.55,
                                  label=f"示向度射线（长 {ray_len:.0f} m，{len(rays)} 条）"))

        # 本步就地清除（近距命中）
        if step.clears:
            ok = [(step.x, step.y) for c in step.clears if c.get("success")]
            if ok:
                arr = np.asarray(ok, dtype=float)
                # 就地清除与扫描点在同一坐标，必须画在扫描点之上才看得见
                ax.plot(arr[:, 0], arr[:, 1], marker="*", ms=15, ls="none", color=C_HIT,
                        zorder=11.0)
                handles.append(Line2D([], [], marker="*", ms=13, ls="none", color=C_HIT,
                                      label=f"本步就地清除（{len(ok)} 个）"))
            bad = len(step.clears) - len(ok)
            if bad:
                arr = np.asarray([(step.x, step.y)] * bad, dtype=float)
                ax.plot(arr[:, 0], arr[:, 1], marker="^", ms=6.5, ls="none", mfc="none",
                        mec=C_TRY, mew=1.3, zorder=7.0)
                handles.append(Line2D([], [], marker="^", ms=6.5, ls="none", mfc="none",
                                      mec=C_TRY, mew=1.3, label=f"清除未中（{bad} 次）"))

        # 本步结束时的估计：定位估计的 1σ 圆
        if step.estimates:
            for ch, (ex, ey, sigma) in sorted(step.estimates.items()):
                ax.plot(ex + sigma * cos_th, ey + sigma * sin_th, color=C_MEAS, lw=0.8,
                        alpha=0.8, zorder=6.5)
                ax.plot([ex], [ey], marker="+", ms=7, mew=1.4, ls="none", color=C_MEAS,
                        zorder=6.6)
            handles.append(Line2D([], [], color=C_MEAS, lw=0.8,
                                  label=f"定位估计 1σ（{len(step.estimates)} 个频道）"))
        if step.regions:
            widths = []
            for ch, ring in sorted(step.regions.items()):
                pts = list(ring)
                if len(pts) < 3:
                    continue
                arr = np.asarray(pts + [pts[0]], dtype=float)
                ax.plot(arr[:, 0], arr[:, 1], color=C_MEAS, lw=0.8, alpha=0.7, zorder=5.5)
                # 区域"宽度"取包围盒对角线：用来量化"信息收缩到什么程度"（逐步对比即可看到
                # 从几百米收到几十米）。用包围盒而非精确直径，是画图取值的廉价近似。
                widths.append(float(np.hypot(np.max(arr[:, 0]) - np.min(arr[:, 0]),
                                             np.max(arr[:, 1]) - np.min(arr[:, 1]))))
            med = float(np.median(widths)) if widths else 0.0
            handles.append(Line2D([], [], color=C_MEAS, lw=0.8, alpha=0.7,
                                  label=f"可能源区域（{len(widths)} 个频道，中位宽 {med:.0f} m）"))

        # 干扰源真值（仅演练模式有）与清除半径
        if sources:
            arr = np.asarray([(s["x"], s["y"]) for s in sources], dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], marker="X", ms=8, ls="none", color=C_SRC,
                    zorder=8.0)
            ax.plot(arr[:, 0][None, :] + clear_radius * cos_th[:, None],
                    arr[:, 1][None, :] + clear_radius * sin_th[:, None],
                    color=C_SRC, lw=0.7, alpha=0.7, zorder=1.5)
            handles.append(Line2D([], [], marker="X", ms=8, ls="none", color=C_SRC,
                                  label=f"干扰源真值（{len(arr)} 个）"))

        # 当前站位置：画在最上层，是本图的主角
        ax.plot(step.x, step.y, marker="s", ms=7, color="black", zorder=10.0)
        handles.append(Line2D([], [], marker="s", ms=8, ls="none", color="black",
                              label="本步扫描点"))

        ax.set_aspect("equal")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        head = title or f"第 {step.index} 步扫描：{step.label}"
        ax.set_title(f"{head}\n{step.summary()}", fontsize=10)
        # 图例放坐标轴下方：3600 m 见方的圆域几乎没有空白，放图内必然压住内容
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07),
                  ncol=3, fontsize=7.5, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5)
        fig.tight_layout()
        save_png(fig, out_path, dpi=160.0)
        plt.close(fig)
    return out_path
