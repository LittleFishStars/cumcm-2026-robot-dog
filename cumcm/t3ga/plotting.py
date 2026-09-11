"""逐局轨迹图与同名轨迹表（GA 对照方案）。

输出到 `<save-dir>/trajectory/epNN_seedMM.png` + 同名 CSV。本方案结果树落在
results/t3_ga/，确定性方案（cumcm.t3.plotting）落在 results/t3/，两者的 trajectory/ 天然
隔离，以免同名文件互相覆盖。
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from cumcm.common.plotting import (C_FRAME, C_HIT, C_PATH, C_SRC, font_context,
                                   hint_plot_once, setup_mpl_env, slug)
from cumcm.common.scanfigure import STEP_DIR_NAME, ScanStep, draw_scan_step
from cumcm.t3ga.config import (CLEAR_RADIUS, COVER_RADIUS, MAX_RECEPTION,
                               REGION_RADIUS)
from cumcm.t3ga.covering import covering_waypoints

# 旧名 → 统一配色（cumcm.common.plotting）。保留别名是为了让绘制函数体一字不改。
_C_PLOT_PATH = C_PATH
_C_PLOT_SRC = C_SRC
_C_PLOT_HIT = C_HIT
_C_PLOT_FRAME = C_FRAME

TRAJ_DIR_NAME = "trajectory"        # 每局轨迹图落在 <save-dir>/trajectory/ 下
TRAJ_DPI = 160.0                    # 位图分辨率
# 中文字体候选：本族沿用原有顺序（优先 Windows 自带字体）。注意**顺序会影响渲染结果** ——
# 本机装了微软雅黑，若改用 common.plotting 的默认链（Noto 优先）会换字体、图也变样，故
# 这里显式指定，保持 GA 版图与历史产物逐字节一致。
TRAJ_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
              "WenQuanYi Zen Hei", "DejaVu Sans")

def draw_trajectory(out_path: Path, track: Sequence[Sequence[float]],
                    sources: Sequence[Sequence[float]] = (),
                    hits: Sequence[Sequence[float]] = (),
                    marks: Sequence[Tuple[float, float, str]] = (),
                    title: Optional[str] = None, figsize: Tuple[float, float] = (6.4, 6.1),
                    dpi: float = 160.0) -> Path:
    """把机器狗轨迹画成图并存盘；格式由文件名后缀决定（.png / .pdf）。

    - `track`：按时间排序的动作点 [(x, y), ...]（含 /measure 与 /clear 的落点）
    - `sources`：干扰源真值 [(x, y, channel), ...]；官方模式拿不到真值，传空即可
    - `hits`：成功清除的落点 [(x, y), ...]
    - `marks`：动作点及类型 [(x, y, "measure"|"clear"), ...]，用于区分测向与清除
    - `title`：图内标题；论文用图传 None（标题交给 caption），每局诊断图传一句概况
    - `dpi`：位图分辨率（矢量格式忽略此项）

    matplotlib 只在本函数内导入：未安装会抛 ImportError，由调用方决定是否忽略——
    这样 T3_ga.py 在没有 matplotlib 的机器（如官方测试机）上仍能完成整局测试。
    """
    setup_mpl_env()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with font_context(size=10, fonts=TRAJ_FONTS):
        fig, ax = plt.subplots(figsize=figsize)
        th = np.linspace(0.0, 2.0 * math.pi, 400)

        # 作业圆域（题目 1800 m）与官方生成器源域（1770 m）
        ax.plot(REGION_RADIUS * np.cos(th), REGION_RADIUS * np.sin(th),
                color=_C_PLOT_FRAME, lw=1.4, label="作业圆域 1800 m")
        gen_r = REGION_RADIUS - 30.0
        ax.plot(gen_r * np.cos(th), gen_r * np.sin(th), color=_C_PLOT_FRAME,
                lw=0.8, ls="--", alpha=0.8, label="干扰源生成域 1770 m")

        # 轨迹（连线表示行驶路径，动作点按类型分开画）
        pts = np.asarray(track, dtype=float)
        if len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.0, color=_C_PLOT_PATH,
                    alpha=0.85, label=f"行驶路径（{len(pts) - 1} 次动作）")
        mk = [(x, y, k) for x, y, k in marks]
        for kind, style, label in (("measure", dict(marker=".", ms=4.5, ls="none",
                                                   color=_C_PLOT_PATH), "测向点 /measure"),
                                   ("clear", dict(marker="^", ms=5.5, ls="none",
                                                 mfc="none", mec=_C_PLOT_HIT, mew=1.2),
                                    "清除尝试 /clear")):
            sel = [(x, y) for x, y, k in mk if k == kind]
            if sel:
                arr = np.asarray(sel, dtype=float)
                ax.plot(arr[:, 0], arr[:, 1], label=f"{label}（{len(sel)} 次）", **style)

        # 干扰源真值与清除半径
        for i, (sx, sy, ch) in enumerate(sources):
            ax.add_patch(plt.Circle((sx, sy), CLEAR_RADIUS, fill=False, color=_C_PLOT_SRC,
                                    lw=0.6, alpha=0.55))
            ax.plot([sx], [sy], "x", ms=8, mew=1.7, color=_C_PLOT_SRC,
                    label="干扰源真值（20 m 清除半径）" if i == 0 else None)
            # 标签交替错开，缓解密集处互相压字
            dx, dy, ha = ((8, 4, "left") if i % 2 == 0 else (-8, -10, "right"))
            ax.annotate(f"ch{ch}", (sx, sy), textcoords="offset points", xytext=(dx, dy),
                        fontsize=7.5, color=_C_PLOT_SRC, ha=ha)

        # 成功清除落点
        if len(hits):
            hp = np.asarray(hits, dtype=float)
            ax.plot(hp[:, 0], hp[:, 1], "o", ms=6.5, mfc="none", mec=_C_PLOT_HIT,
                    mew=1.5, label=f"成功清除落点（{len(hp)} 处）")

        ax.plot([0.0], [0.0], marker="*", ms=15, color="#c8871b",
                label="起点 / 结束点")

        ax.set_aspect("equal")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        pad = 240.0
        ax.set_xlim(-REGION_RADIUS - pad, REGION_RADIUS + pad)
        ax.set_ylim(-REGION_RADIUS - pad, REGION_RADIUS + pad)
        ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
        if title:
            ax.set_title(title, fontsize=10.5, pad=8)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    return out_path


def save_trajectory_plot(save_dir: str, name: str, track: Sequence[Sequence[float]],
                         sources: Sequence[Sequence[float]] = (),
                         hits: Sequence[Sequence[float]] = (),
                         marks: Sequence[Tuple[float, float, str]] = (),
                         title: Optional[str] = None) -> Optional[Path]:
    """保存一局的轨迹图与同名轨迹表；失败只提示、不影响测试结果（返回图路径或 None）。

    在 /exit 之后调用，因此绘图耗时不计入现实运行时间预算。绘图或落盘出错都不应
    影响已完成的测试，故这里吞掉异常（含缺 matplotlib 的情况）。

    同时写一份同名 .csv（`step,x,y,kind`），使图上的动作点可被逐条核对——否则
    图片只是一张无法验证的图。CSV 与 PNG 同目录同名，一一对应。
    """
    out = Path(save_dir) / TRAJ_DIR_NAME / f"{name}.png"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_track_csv(out.with_suffix(".csv"), track, marks)
    except OSError as exc:
        hint_plot_once(f"提示：轨迹表写入失败（{type(exc).__name__}: {exc}）")
    try:
        return draw_trajectory(out, track, sources, hits, marks, title)
    except ImportError:
        hint_plot_once("提示：未安装 matplotlib，已跳过出图。装上即可自动生成：\n"
                       "      .venv/bin/pip install matplotlib")
    except Exception as exc:        # 字体/磁盘/权限等问题都不该影响测试结论
        hint_plot_once(f"提示：轨迹图生成失败，已跳过（{type(exc).__name__}: {exc}）")
    return None


def _write_track_csv(path: Path, track: Sequence[Sequence[float]],
                     marks: Sequence[Tuple[float, float, str]]) -> None:
    """把轨迹写成 `step,x,y,kind`，与轨迹图上的点逐条对应。

    step 0 固定为起点 (0,0)；其后每个动作点对应一次 /measure 或 /clear，
    kind 取 measure / clear，便于与 api_calls.jsonl 对账。
    """
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "x", "y", "kind"])
        w.writerow([0, f"{float(track[0][0]):.3f}", f"{float(track[0][1]):.3f}", "start"])
        for i, (x, y, kind) in enumerate(marks, start=1):
            w.writerow([i, f"{float(x):.3f}", f"{float(y):.3f}", kind])


def save_scan_figures(save_dir: str, name: str, steps: Sequence[dict],
                      sources: Sequence[Sequence[float]] = (),
                      step_dir: str = STEP_DIR_NAME) -> List[Path]:
    """把一局内**每一步扫描**各画一张结果图，落在 <save-dir>/<step_dir>/ 下。

    文件名形如 `ep01_s00_起点全频道扫描.png`，排序后与执行顺序一致。绘图放在 /exit 之后，
    不占现实时间预算；缺 matplotlib 只提示一次并跳过（与轨迹图同样的容错口径）。

    与确定性方案的差别：路点由贪心集合覆盖给出（8 个，半径 920 m 的覆盖保证，而确定性方案
    是固定 7 个半径 1000 m 的圆），且本族估计形态是"点 + 位置 1σ"而非多边形区域 ——
    故图上画 σ 圆。这些差异由调用方通过参数表达，绘制本身复用 cumcm.common.scanfigure。
    """
    out_dir = Path(save_dir) / step_dir
    paths: List[Path] = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        hint_plot_once(f"提示：扫描图目录创建失败（{type(exc).__name__}: {exc}）")
        return paths
    # 路点布局用**本族真正的覆盖路点集**（贪心集合覆盖的产物，COVER_RADIUS = 920 m 的保证
    # 就来自它）；起点扫描的位置不属于该集合，故它只作为"本步扫描点"出现，不画成路点。
    wp = [tuple(map(float, p)) for p in covering_waypoints()]
    for k, raw in enumerate(steps):
        # 到第 k 步为止已扫过的路点：按坐标匹配（容差 1 m，与图上判定当前站的容差一致）
        seen = [(float(s_["x"]), float(s_["y"])) for s_ in steps[:k + 1]
                if int(s_.get("index", 0)) > 0]
        visited = [i for i, w in enumerate(wp)
                   if any(math.hypot(w[0] - sx, w[1] - sy) <= 1.0 for sx, sy in seen)]
        safe = slug(str(raw.get("label", f"step{k}")))
        out = out_dir / f"{name}_s{k:02d}_{safe}.png"
        try:
            draw_scan_step(out, ScanStep(
                index=int(raw.get("index", k)), label=str(raw.get("label", "")),
                x=float(raw["x"]), y=float(raw["y"]),
                n_channels=int(raw.get("n_channels", 0)),
                counts=dict(raw.get("counts", {})),
                virtual_time_s=float(raw.get("virtual_time_s", 0.0)),
                travel_m=float(raw.get("travel_m", 0.0)),
                measures=list(raw.get("measures", [])),
                clears=list(raw.get("clears", [])),
                path=[tuple(map(float, p)) for p in raw.get("path", [])],
                cleared=list(raw.get("cleared", [])),
                estimates=raw.get("estimates") or None,
            ), cover_centers=wp, visit_order=list(range(len(wp))),
                visited=visited,
                cover_radius=COVER_RADIUS, region_radius=REGION_RADIUS,
                gen_radius=REGION_RADIUS - 30.0, ray_len=MAX_RECEPTION,
                sources=[{"x": float(a), "y": float(b)} for a, b, _ in sources],
                clear_radius=CLEAR_RADIUS,
                title=f"第 {k} 步扫描 / 共 {len(steps)} 步：{raw.get('label', '')}",
                fonts=TRAJ_FONTS)
            paths.append(out)
        except ImportError:
            hint_plot_once("提示：未安装 matplotlib，已跳过出图。装上即可自动生成：\n"
                           "      .venv/bin/pip install matplotlib")
            return paths
        except Exception as exc:        # 字体/磁盘等问题都不该影响测试结论
            hint_plot_once(f"提示：扫描图生成失败，已跳过（{type(exc).__name__}: {exc}）")
            return paths
    return paths


def sources_of(truth: Optional[Sequence[dict]]) -> List[Tuple[float, float, int]]:
    """把模拟器真值整理成绘图用的 [(x, y, channel), ...]。"""
    return [(float(j["position"]["x"]), float(j["position"]["y"]), int(j["channel"]))
            for j in (truth or [])]
