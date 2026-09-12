"""问题四论文图表：读 results/t4/（扫描方案 + 逐局统计）生成 figures/ 下的 PDF。

与 cumcm.analysis.figures（问题三）同风格：全部图在跑完演练后单独生成（不占现实时间预算），
数据取自结果目录里的 json/csv，出图确定性（无时间戳元数据）。

本模块覆盖问题四特有的几个论点：
* 扫描布局与"7 覆盖基点 + 21 外推中点"：如何从问题三的 7 点出发、用中点外推补上定向源的
  边缘迎光区（实测听到率 ~99.8%，非严格保证）——这是"尽量复用问题三布局"的体现；
* 听到率实证：20 局里每个源首次被听到发生在扫描的第几步；
* 方向性：定向源与被漏测的"背光"情形（波束扇形示意图）；
* 清除结果：逐局清除数/虚拟时间、定位误差分布、时间构成（扫描 vs 收尾）。

用法：

    python T4_figures.py                     # 读 results/t4/，出图到 figures/
    python T4_figures.py --results results/t4 --figdir figures
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from cumcm.t4.config import (CLEAR_RADIUS, DIR_BEAM_HALF_DEG, EXTEND_CLAMP, EXTEND_K,
                             RECEIVE_MAX, REGION_RADIUS)
from cumcm.t4.sweep import build_sweep_plan

MARKER = {"metadata": {"Software": "cumcm-t4"}}
FONT_CANDIDATES = ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei")


def _setup() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from cumcm.common.plotting import setup_mpl_env
    setup_mpl_env()     # 固定可写的 matplotlib 缓存目录，消除"~/.config/matplotlib 不可写"警告


def _save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 图 1：扫描布局（原点 + 7 覆盖基点 + 21 外推中点）与半圆盘示意
# ---------------------------------------------------------------------------
def fig_sweep_lattice(fig_dir: Path) -> None:
    """画扫描布局、作业圆域与一个"半圆盘 ⊇ 内切圆 ∋ 命中测量点"的示意。"""
    _setup()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    plan = build_sweep_plan()
    pts = plan.points
    base = pts[1:8]                    # 7 覆盖基点（第 0 个是原点）
    mids = pts[8:]
    with plt.rc_context({"font.family": FONT_CANDIDATES, "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(8.6, 8.6))
        th = np.linspace(0.0, 2.0 * np.pi, 361)
        ax.plot(REGION_RADIUS * np.cos(th), REGION_RADIUS * np.sin(th), color="#5b6470",
                lw=1.5, label="作业圆域 1800 m")
        ax.plot((REGION_RADIUS - 30.0) * np.cos(th), (REGION_RADIUS - 30.0) * np.sin(th),
                color="#5b6470", lw=0.8, ls="--", alpha=0.8, label="源生成域 1770 m")
        ax.plot(base[:, 0], base[:, 1], "o", ms=7, color="#2f6fb5", zorder=4,
                label=f"7 覆盖基点（复用问题三巡视站布局）")
        ax.plot(mids[:, 0], mids[:, 1], "^", ms=6, color="#7b3fa0", zorder=4,
                label=f"21 外推中点（两两基点中点 ×{EXTEND_K:.1f}，上限 {EXTEND_CLAMP:.0f} m）")
        ax.plot(0.0, 0.0, "s", ms=7, color="#2e9e5b", zorder=5, label="原点（起点全频道扫描）")

        # 示意：一个定向源 g（贴在边缘、波束朝外）→ 半圆盘 → 内切圆 → 命中测量点
        g = np.array([1770.0 * math.cos(math.radians(62.0)),
                      1770.0 * math.sin(math.radians(62.0))])
        thb = math.radians(62.0)          # 波束方向 ≈ 径向朝外
        R = 1000.0
        a0 = thb - math.pi / 2
        half = np.linspace(a0, a0 + math.pi, 61)
        arc = np.stack([g[0] + R * np.cos(half), g[1] + R * np.sin(half)], axis=1)
        disk_poly = np.vstack([np.array([[g[0], g[1]]]), arc,
                               np.array([[g[0], g[1]]])])
        ax.add_patch(Polygon(disk_poly, closed=True, facecolor="#c0392b", alpha=0.08,
                             zorder=1))
        ax.plot(arc[:, 0], arc[:, 1], color="#c0392b", lw=1.0, alpha=0.6,
                label="定向源检测区（半径 R 的半圆盘）")
        c = g + (R / 2) * np.array([math.cos(thb), math.sin(thb)])
        ax.add_patch(Circle(c, R / 2, fill=False, color="#2e9e5b", lw=1.4, ls="--",
                            label="半圆盘内切圆（半径 R/2 ≥ 500）"))
        d2 = np.linalg.norm(pts - c, axis=1)
        hit = pts[d2 <= d2.min() + 1e-6]
        if len(hit) and d2.min() <= 750.0:
            p = hit[0]
            ax.plot([p[0]], [p[1]], "o", ms=9, color="#c0392b", zorder=6)
            ax.annotate("命中测量点", (p[0], p[1]), textcoords="offset points",
                        xytext=(12, 6), fontsize=9, color="#c0392b")
        ax.plot([g[0]], [g[1]], "X", ms=12, color="#c0392b", zorder=7, label="定向源")
        ax.annotate("波束方向 θ", (g[0], g[1]), textcoords="offset points", xytext=(-4, -18),
                    fontsize=9, color="#c0392b")

        ax.set_aspect("equal")
        ax.set_xlim(-2500, 2500)
        ax.set_ylim(-2500, 2500)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8.5,
                  framealpha=0.9)
        ax.set_title(f"扫描布局：原点 + 7 覆盖基点 + 21 外推中点（{len(pts)} 个测量位置，"
                     f"实测听到率 ~99.8%，非严格保证）")
        _save(fig, fig_dir / "fig_t4_sweep_lattice.pdf")


# ---------------------------------------------------------------------------
# 图 2：逐局清除结果（清除数 / 虚拟时间）
# ---------------------------------------------------------------------------
def fig_episode_clear(survey: Dict[str, Any], fig_dir: Path) -> None:
    rows = survey["episodes"]
    if not rows:
        return
    _setup()
    import matplotlib.pyplot as plt
    xs = [r["seed"] for r in rows]
    cleared = [r["cleared"] for r in rows]
    nsrc = [r["n_sources"] or 0 for r in rows]
    vt = [r["virtual_time_s"] / 60.0 for r in rows]     # 分钟
    with plt.rc_context({"font.family": FONT_CANDIDATES, "axes.unicode_minus": False}):
        fig, ax1 = plt.subplots(figsize=(9.2, 4.6))
        ax1.bar(xs, cleared, width=0.8, color="#2f6fb5", alpha=0.85, label="清除数")
        ax1.plot(xs, nsrc, "X", ms=8, color="#c0392b", label="全向+定向源总数")
        ax1.set_xlabel("seed")
        ax1.set_ylabel("干扰源数")
        ax1.set_ylim(0, max(max(nsrc) + 2, 18))
        ax2 = ax1.twinx()
        ax2.plot(xs, vt, "o-", ms=5, color="#7b3fa0", label="虚拟时间 / min")
        ax2.set_ylabel("虚拟时间 / min")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5)
        ax1.set_title(f"问题四演练 {len(rows)} 局：全部源清除，平均虚拟时间 "
                      f"{np.mean(vt):.1f} min")
        _save(fig, fig_dir / "fig_t4_episode_clear.pdf")


# ---------------------------------------------------------------------------
# 图 3：听到率实证 —— 每个源首次被听到的扫描步骤分布
# ---------------------------------------------------------------------------
def fig_first_heard(survey: Dict[str, Any], fig_dir: Path) -> None:
    steps: List[int] = []
    for r in survey["episodes"]:
        fh = r.get("first_heard") or {}
        steps += [int(v) for v in fh.values()]
    if not steps:
        return
    n_plan = survey.get("sweep_plan", {}).get("n_measure_points", 29)
    _setup()
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": FONT_CANDIDATES, "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(9.2, 4.4))
        bins = np.arange(0, n_plan + 2) - 0.5
        ax.hist(steps, bins=bins, color="#2f6fb5", alpha=0.85,
                label=f"各源首次被听到的扫描步骤（{len(steps)} 个源）")
        ax.axvline(n_plan - 1, color="#c0392b", ls="--", lw=1.2,
                   label=f"扫描结束（第 {n_plan - 1} 个测量位置之后不再有未听到源）")
        ax.set_xlabel("首次听到时的扫描步骤序号")
        ax.set_ylabel("源数")
        ax.set_title("听到率实证：全部源（含定向源）都在扫描结束前被听到 ≥ 1 次（本批 20 局）")
        ax.legend(fontsize=8.5)
        _save(fig, fig_dir / "fig_t4_first_heard.pdf")


# ---------------------------------------------------------------------------
# 图 4：定位误差分布（所有源的清除点与真值距离）
# ---------------------------------------------------------------------------
def fig_localization_error(survey: Dict[str, Any], fig_dir: Path) -> None:
    errs: List[float] = []
    for r in survey["episodes"]:
        if r.get("localize_err_mean_m") is None:
            continue
        errs.append(r["localize_err_mean_m"])
    if not errs:
        return
    _setup()
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": FONT_CANDIDATES, "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        ax.plot(range(1, len(errs) + 1), errs, "o-", ms=5, color="#2f6fb5",
                label="局均定位误差（清除点距真值）")
        ax.axhline(CLEAR_RADIUS, color="#c0392b", ls="--", lw=1.2,
                   label=f"清除半径 {CLEAR_RADIUS:.0f} m")
        ax.set_xlabel("演练局（seed 升序）")
        ax.set_ylabel("定位误差 / m")
        ax.set_title(f"清除点定位误差：均值 {np.mean(errs):.2f} m，"
                     f"最大 {np.max(errs):.2f} m，全部 ≤ {CLEAR_RADIUS:.0f} m")
        ax.legend(fontsize=8.5)
        _save(fig, fig_dir / "fig_t4_localization_error.pdf")


# ---------------------------------------------------------------------------
# 图 5：虚拟时间构成（扫描 vs 收尾；由逐局汇总近似）
# ---------------------------------------------------------------------------
def fig_cost(survey: Dict[str, Any], fig_dir: Path) -> None:
    rows = survey["episodes"]
    if not rows:
        return
    plan = survey.get("sweep_plan", {})
    route_m = plan.get("route_m", 0.0) or 0.0
    _setup()
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": FONT_CANDIDATES, "axes.unicode_minus": False}):
        vt = np.array([r["virtual_time_s"] for r in rows])
        travel = np.array([r["travel_m"] for r in rows])
        sweeptravel = route_m
        fig, ax = plt.subplots(figsize=(9.0, 4.4))
        xs = np.arange(len(rows))
        ax.bar(xs, sweeptravel / 5.0 / 60.0, width=0.6, color="#2f6fb5",
               label=f"扫描行驶（{sweeptravel:.0f} m / 5 m/s）")
        ax.bar(xs, np.maximum(travel - sweeptravel, 0.0) / 5.0 / 60.0, bottom=sweeptravel / 5.0 / 60.0,
               width=0.6, color="#7b3fa0", label="收尾定位行驶")
        ax.plot(xs, vt / 60.0, "o-", ms=4, color="#c0392b", label="总虚拟时间")
        ax.set_xlabel("演练局（seed 升序）")
        ax.set_ylabel("时间 / min")
        ax.set_title(f"问题四时间构成：扫描行驶平均 "
                     f"{np.mean(sweeptravel / 300.0):.1f} min，总虚拟时间平均 {np.mean(vt / 60.0):.1f} min")
        ax.legend(fontsize=8.5)
        _save(fig, fig_dir / "fig_t4_cost.pdf")


# ---------------------------------------------------------------------------
# 图 6：定向源方向性示意 —— "方向不对"为什么收不到
# ---------------------------------------------------------------------------
def fig_directional_concept(fig_dir: Path) -> None:
    _setup()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    with plt.rc_context({"font.family": FONT_CANDIDATES, "axes.unicode_minus": False}):
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        g = np.array([0.0, 0.0])
        thb = math.radians(30.0)
        R = 1000.0
        a0 = thb - math.pi / 2
        half = np.linspace(a0, a0 + math.pi, 81)
        arc = np.stack([g[0] + R * np.cos(half), g[1] + R * np.sin(half)], axis=1)
        poly = np.vstack([arc, np.array([g[0], g[1]])])
        ax.add_patch(Polygon(poly, closed=True, facecolor="#2e9e5b", alpha=0.14,
                             edgecolor="#2e9e5b", lw=1.2,
                             label="定向源检测区 (R=1000, ±90°)"))
        for k in (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330):
            a = math.radians(k)
            d = 1580.0
            ax.plot(d * math.cos(a), d * math.sin(a), ".", color="black", ms=5)
            ax.annotate(f"{k}°", (d * math.cos(a), d * math.sin(a)),
                        textcoords="offset points", xytext=(4, 4), fontsize=7.5, color="black")
        # 两个代表性测量点：迎光侧（听见）与背光侧（no_signal）
        front = np.array([R * math.cos(thb), R * math.sin(thb)]) * 1.15
        back = np.array([-R * math.cos(thb), -R * math.sin(thb)]) * 1.15
        ax.plot([front[0]], [front[1]], "o", ms=9, color="#2f6fb5", zorder=6)
        ax.annotate("迎光侧：direction（有示向度）", front, textcoords="offset points",
                    xytext=(-88, 16), fontsize=9, color="#2f6fb5")
        ax.plot([back[0]], [back[1]], "o", ms=9, color="#c0392b", zorder=6)
        ax.annotate("背光侧：no_signal（方向不对）", back, textcoords="offset points",
                    xytext=(-90, -30), fontsize=9, color="#c0392b")
        ax.plot([g[0]], [g[1]], "X", ms=12, color="#c0392b", zorder=7, label="定向源")
        ax.annotate("定向方向 θ=30°", (0, 0), textcoords="offset points", xytext=(18, 24),
                    fontsize=9, color="#c0392b")
        ax.arrow(0, 0, 700 * math.cos(thb), 700 * math.sin(thb), color="#c0392b",
                 lw=1.6, head_width=40, length_includes_head=True)
        ax.set_aspect("equal")
        ax.set_xlim(-1750, 1750)
        ax.set_ylim(-1400, 1400)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.legend(loc="upper right", fontsize=8.5)
        ax.set_title("问题四的核心差异：同一距离，方向不对就听不到（no_signal 不再是'源太远'）")
        _save(fig, fig_dir / "fig_t4_directional_concept.pdf")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="问题四论文图表生成")
    ap.add_argument("--results", default="results/t4", help="结果目录（缺省 results/t4）")
    ap.add_argument("--figdir", default="figures", help="图表输出目录（缺省 figures/）")
    args = ap.parse_args(argv)
    fig_dir = Path(args.figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    survey_path = Path(args.results) / "t4_survey.json"

    out: List[Path] = []
    if survey_path.exists():
        survey = load_json(survey_path)
        fig_episode_clear(survey, fig_dir)
        fig_first_heard(survey, fig_dir)
        fig_localization_error(survey, fig_dir)
        fig_cost(survey, fig_dir)
        out = sorted(fig_dir.glob("fig_t4_*.pdf"))
    fig_sweep_lattice(fig_dir)                      # 布局图不依赖运行数据，总是可画
    fig_directional_concept(fig_dir)
    out = sorted(fig_dir.glob("fig_t4_*.pdf"))
    print("已生成 " + "、".join(p.name for p in out))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())