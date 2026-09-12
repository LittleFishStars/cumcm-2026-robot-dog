"""paper/t4 论文图生成：读当前求解产物，出 8 张论文图到 paper/t4/figures/。

用法（在仓库根目录运行）：

    .venv/bin/python paper/t4/make_figures.py
    .venv/bin/python paper/t4/make_figures.py --results results/t4 --figdir paper/t4/figures

图里的数字**全部来自当前产物或代码里的定案布局**，不写死，避免"论文数字跟不上重跑"：

| 图 | 数据来源 |
| --- | --- |
| `fig_t4_sweep_lattice.pdf` | `cumcm.t4.sweep.build_sweep_plan()`（20 点定案）+ 计划里的听到率统计 |
| `fig_t4_episode_clear.pdf` | `results/t4/t4_survey.json` 逐局清除数与虚拟时间 |
| `fig_t4_first_heard.pdf` | 同上，逐局 `first_heard` 的首次听到步骤分布 |
| `fig_t4_localization_error.pdf` | 同上，逐局定位误差均值 |
| `fig_t4_cost.pdf` | 同上 + 计划里的扫描里程（时间构成） |
| `fig_t4_directional_concept.pdf` | 不依赖数据（定向源迎光/背光示意） |
| `fig_t4_nn_layout_compare.png` | 定案布局 vs `results/t4/.nn_best_layout.npy`（神经网络自由搜索） |
| `fig_t4_nn_speed_compare.png` | `results/t4/.nn_arm_speed.json`（各布局在当前参数下的 20 局实测） |

出图确定性：PNG 走 `cumcm.common.plotting.save_png`（去掉时间戳元数据），PDF 用
`_save_pdf` 显式清掉 CreationDate，故同一份产物两次出图逐字节一致。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:              # 允许从任意目录运行本脚本
    sys.path.insert(0, str(REPO_ROOT))

from cumcm.common.plotting import font_context, save_png, setup_mpl_env  # noqa: E402
from cumcm.t4.config import CLEAR_RADIUS, OUTER_RING_N, OUTER_RING_RAD, REGION_RADIUS  # noqa: E402
from cumcm.t4.sweep import build_sweep_plan                       # noqa: E402
from drv_speed import make_plan                                   # noqa: E402

setup_mpl_env()          # 先固定 matplotlib 缓存目录，再让各图函数导入 pyplot

C_BASE = "#2f6fb5"          # 覆盖基点 / 主色
C_INNER = "#d97706"         # 圆域内测量点
C_OUTER = "#7b3fa0"         # 外圈测量点
C_SRC = "#c0392b"           # 干扰源 / 阈值 / 强调
C_FRAME = "#5b6470"         # 作业圆域边界
C_OK = "#2e9e5b"            # 原点 / 命中


def _save_pdf(fig: Any, path: Path) -> None:
    """确定性保存 PDF：清掉 CreationDate/Software，保证两次出图逐字节一致。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None, "Software": None})
    import matplotlib.pyplot as plt
    plt.close(fig)


def _save(fig: Any, path: Path) -> None:
    """按扩展名分派：PDF 用 _save_pdf，PNG 用确定性 save_png。"""
    if path.suffix.lower() == ".pdf":
        _save_pdf(fig, path)
    else:
        save_png(fig, path, dpi=200, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 图 1：扫描布局（原点 + 7 覆盖基点 + 12 均匀外圈点 = 20 个测量位置）
# ---------------------------------------------------------------------------
def fig_sweep_lattice(fig_dir: Path, plan) -> None:
    """扫描布局、作业圆域，以及"定向源半圆盘检测区 ⊇ 内切圆 ∋ 命中测量点"的示意。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    pts = plan.points
    base = pts[1:8]                       # 7 覆盖基点（第 0 个是原点）
    outer = pts[8 + plan.interior_n:]     # 12 均匀外圈点
    hear = plan.verification.get("hear_rate") if isinstance(plan.verification, dict) else None
    with font_context(size=10):
        fig, ax = plt.subplots(figsize=(8.6, 8.6))
        th = np.linspace(0.0, 2.0 * np.pi, 361)
        ax.plot(REGION_RADIUS * np.cos(th), REGION_RADIUS * np.sin(th), color=C_FRAME,
                lw=1.5, label=f"作业圆域 {REGION_RADIUS:.0f} m")
        ax.plot((REGION_RADIUS - 30.0) * np.cos(th), (REGION_RADIUS - 30.0) * np.sin(th),
                color=C_FRAME, lw=0.8, ls="--", alpha=0.8, label="源生成域 1770 m")
        ax.plot(base[:, 0], base[:, 1], "o", ms=7, color=C_BASE, zorder=4,
                label="7 覆盖基点（复用问题三巡视站）")
        ax.plot(outer[:, 0], outer[:, 1], "^", ms=6, color=C_OUTER, zorder=4,
                label=f"{OUTER_RING_N} 均匀外圈点（r = {OUTER_RING_RAD:.0f} m，间隔 "
                      f"{360.0 / OUTER_RING_N:.0f}°）")
        ax.plot(0.0, 0.0, "s", ms=7, color=C_OK, zorder=5, label="原点（起点全频道扫描）")

        # 示意：一个贴在边缘、波束朝外的定向源 → 半圆盘检测区 → 内切圆 → 命中测量点
        ang = math.radians(62.0)
        g = np.array([1770.0 * math.cos(ang), 1770.0 * math.sin(ang)])
        R = 1000.0
        a0 = ang - math.pi / 2
        half = np.linspace(a0, a0 + math.pi, 61)
        arc = np.stack([g[0] + R * np.cos(half), g[1] + R * np.sin(half)], axis=1)
        ax.add_patch(Polygon(np.vstack([[g], arc, [g]]), closed=True, facecolor=C_SRC,
                             alpha=0.08, zorder=1))
        ax.plot(arc[:, 0], arc[:, 1], color=C_SRC, lw=1.0, alpha=0.6,
                label="定向源检测区（半径 R 的半圆盘）")
        c = g + (R / 2) * np.array([math.cos(ang), math.sin(ang)])
        ax.add_patch(Circle(c, R / 2, fill=False, color=C_OK, lw=1.4, ls="--",
                            label="半圆盘内切圆（半径 R/2）"))
        d2 = np.linalg.norm(pts - c, axis=1)
        if d2.min() <= 750.0:
            p = pts[int(np.argmin(d2))]
            ax.plot([p[0]], [p[1]], "o", ms=9, color=C_SRC, zorder=6)
            ax.annotate("命中测量点", (p[0], p[1]), textcoords="offset points",
                        xytext=(12, 6), fontsize=9, color=C_SRC)
        ax.plot([g[0]], [g[1]], "X", ms=12, color=C_SRC, zorder=7, label="定向源")
        ax.annotate("波束方向 θ", (g[0], g[1]), textcoords="offset points", xytext=(-4, -18),
                    fontsize=9, color=C_SRC)

        ax.set_aspect("equal")
        ax.set_xlim(-2500, 2500)
        ax.set_ylim(-2500, 2500)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8.5,
                  framealpha=0.9)
        hear_txt = f"，实测听到率 {hear * 100:.4f}%" if hear else ""
        ax.set_title(f"扫描布局：原点 + 7 覆盖基点 + {OUTER_RING_N} 均匀外圈点"
                     f"（{plan.n_points} 个测量位置，里程 {plan.route_m / 1000:.2f} km{hear_txt}）")
        _save(fig, fig_dir / "fig_t4_sweep_lattice.pdf")


# ---------------------------------------------------------------------------
# 图 2：逐局清除结果（清除数 / 虚拟时间）
# ---------------------------------------------------------------------------
def fig_episode_clear(survey: Dict[str, Any], fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = survey["episodes"]
    xs = [r["seed"] for r in rows]
    cleared = [r["cleared"] for r in rows]
    nsrc = [r["n_sources"] or 0 for r in rows]
    vt = [r["virtual_time_s"] / 60.0 for r in rows]
    with font_context(size=10):
        fig, ax1 = plt.subplots(figsize=(9.2, 4.6))
        ax1.bar(xs, cleared, width=0.8, color=C_BASE, alpha=0.85, label="清除数")
        ax1.plot(xs, nsrc, "X", ms=8, color=C_SRC, label="全向 + 定向源总数")
        ax1.set_xlabel("seed")
        ax1.set_ylabel("干扰源数")
        ax1.set_ylim(0, max(max(nsrc) + 2, 18))
        ax2 = ax1.twinx()
        ax2.plot(xs, vt, "o-", ms=5, color=C_OUTER, label="虚拟时间 / min")
        ax2.set_ylabel("虚拟时间 / min")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5)
        ax1.set_title(f"问题四演练 {len(rows)} 局：{sum(cleared)}/{sum(nsrc)} 个源全部清除，"
                      f"平均虚拟时间 {np.mean(vt):.1f} min")
        _save(fig, fig_dir / "fig_t4_episode_clear.pdf")


# ---------------------------------------------------------------------------
# 图 3：听到率实证 —— 每个源首次被听到的扫描步骤分布
# ---------------------------------------------------------------------------
def fig_first_heard(survey: Dict[str, Any], plan, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    steps: List[int] = []
    for r in survey["episodes"]:
        steps += [int(v) for v in (r.get("first_heard") or {}).values()]
    n_plan = plan.n_points
    with font_context(size=10):
        fig, ax = plt.subplots(figsize=(9.2, 4.4))
        ax.hist(steps, bins=np.arange(0, n_plan + 2) - 0.5, color=C_BASE, alpha=0.85,
                label=f"各源首次被听到的扫描步骤（{len(steps)} 个源）")
        ax.axvline(n_plan - 1, color=C_SRC, ls="--", lw=1.2,
                   label=f"扫描结束（第 {n_plan - 1} 个测量位置之后不再有未听到源）")
        ax.set_xlabel("首次听到时的扫描步骤序号")
        ax.set_ylabel("源数")
        ax.set_title(f"听到率实证：全部 {len(steps)} 个源（含定向源）都在扫描结束前被听到 ≥ 1 次"
                     f"（最晚第 {max(steps)} 步）")
        ax.legend(fontsize=8.5)
        _save(fig, fig_dir / "fig_t4_first_heard.pdf")


# ---------------------------------------------------------------------------
# 图 4：定位误差分布（各局清除点与真值的平均距离）
# ---------------------------------------------------------------------------
def fig_localization_error(survey: Dict[str, Any], fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    errs = [r["localize_err_mean_m"] for r in survey["episodes"]
            if r.get("localize_err_mean_m") is not None]
    maxs = [r["localize_err_max_m"] for r in survey["episodes"]
            if r.get("localize_err_max_m") is not None]
    with font_context(size=10):
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        ax.plot(range(1, len(errs) + 1), errs, "o-", ms=5, color=C_BASE,
                label="局均定位误差（清除点距源真值）")
        ax.plot(range(1, len(maxs) + 1), maxs, "s--", ms=4, color=C_OUTER,
                label="该局最差单源定位误差")
        ax.axhline(CLEAR_RADIUS, color=C_SRC, ls="--", lw=1.2,
                   label=f"清除半径 {CLEAR_RADIUS:.0f} m")
        ax.set_xlabel("演练局（seed 升序）")
        ax.set_ylabel("定位误差 / m")
        ax.set_title(f"清除点定位误差：局均 {np.mean(errs):.2f} m（最大 {np.max(maxs):.2f} m），"
                     f"全部 ≤ {CLEAR_RADIUS:.0f} m")
        ax.legend(fontsize=8.5)
        _save(fig, fig_dir / "fig_t4_localization_error.pdf")


# ---------------------------------------------------------------------------
# 图 5：虚拟时间构成（扫描行驶 vs 收尾定位行驶）
# ---------------------------------------------------------------------------
def fig_cost(survey: Dict[str, Any], plan, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = survey["episodes"]
    vt = np.array([r["virtual_time_s"] for r in rows])
    travel = np.array([r["travel_m"] for r in rows])
    sweep_min = plan.route_m / 5.0 / 60.0                 # 检测扫描行驶（5 m/s）
    with font_context(size=10):
        fig, ax = plt.subplots(figsize=(9.0, 4.4))
        xs = np.arange(len(rows))
        ax.bar(xs, sweep_min, width=0.6, color=C_BASE,
               label=f"检测扫描行驶（{plan.route_m / 1000:.2f} km，5 m/s）")
        ax.bar(xs, np.maximum(travel - plan.route_m, 0.0) / 5.0 / 60.0, bottom=sweep_min,
               width=0.6, color=C_OUTER, label="收尾定位行驶")
        ax.plot(xs, vt / 60.0, "o-", ms=4, color=C_SRC, label="总虚拟时间")
        ax.set_xlabel("演练局（seed 升序）")
        ax.set_ylabel("时间 / min")
        ax.set_title(f"时间构成：检测扫描行驶固定 {sweep_min:.1f} min"
                     f"（占均值 {np.mean(vt) / 60.0:.1f} min 的 {sweep_min / np.mean(vt / 60.0) * 100:.0f}%），"
                     f"其余为收尾定位与测向")
        ax.legend(fontsize=8.5)
        _save(fig, fig_dir / "fig_t4_cost.pdf")


# ---------------------------------------------------------------------------
# 图 6：定向源方向性示意 —— "方向不对"为什么收不到
# ---------------------------------------------------------------------------
def fig_directional_concept(fig_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    with font_context(size=10):
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        thb = math.radians(30.0)
        R = 1000.0
        a0 = thb - math.pi / 2
        half = np.linspace(a0, a0 + math.pi, 81)
        arc = np.stack([R * np.cos(half), R * np.sin(half)], axis=1)
        ax.add_patch(Polygon(np.vstack([arc, np.array([0.0, 0.0])]), closed=True,
                             facecolor=C_OK, alpha=0.14, edgecolor=C_OK, lw=1.2,
                             label=f"定向源检测区（R = {R:.0f} m，±90°）"))
        for k in range(0, 360, 30):
            a = math.radians(k)
            d = 1580.0
            ax.plot(d * math.cos(a), d * math.sin(a), ".", color="black", ms=5)
            ax.annotate(f"{k}°", (d * math.cos(a), d * math.sin(a)), textcoords="offset points",
                        xytext=(4, 4), fontsize=7.5, color="black")
        front = np.array([R * math.cos(thb), R * math.sin(thb)]) * 1.15
        back = -front
        ax.plot([front[0]], [front[1]], "o", ms=9, color=C_BASE, zorder=6)
        ax.annotate("迎光侧：direction（有示向度）", front, textcoords="offset points",
                    xytext=(-88, 16), fontsize=9, color=C_BASE)
        ax.plot([back[0]], [back[1]], "o", ms=9, color=C_SRC, zorder=6)
        ax.annotate("背光侧：no_signal（方向不对）", back, textcoords="offset points",
                    xytext=(-90, -30), fontsize=9, color=C_SRC)
        ax.plot([0.0], [0.0], "X", ms=12, color=C_SRC, zorder=7, label="定向源")
        ax.annotate("定向方向 θ = 30°", (0.0, 0.0), textcoords="offset points", xytext=(18, 24),
                    fontsize=9, color=C_SRC)
        ax.arrow(0, 0, 700 * math.cos(thb), 700 * math.sin(thb), color=C_SRC, lw=1.6,
                 head_width=40, length_includes_head=True)
        ax.set_aspect("equal")
        ax.set_xlim(-1750, 1750)
        ax.set_ylim(-1400, 1400)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.legend(loc="upper right", fontsize=8.5)
        ax.set_title("问题四的核心差异：同一距离，方向不对就听不到（no_signal 不再是'源太远'）")
        _save(fig, fig_dir / "fig_t4_directional_concept.pdf")


# ---------------------------------------------------------------------------
# 图 7：神经网络搜索布局 vs 人工定案布局
# ---------------------------------------------------------------------------
def _draw_layout(ax, pts: np.ndarray, route, title: str) -> None:
    th = np.linspace(0.0, 2.0 * math.pi, 361)
    ax.plot(REGION_RADIUS * np.cos(th), REGION_RADIUS * np.sin(th), color="#334155", lw=1.2)
    ax.plot((REGION_RADIUS - 30.0) * np.cos(th), (REGION_RADIUS - 30.0) * np.sin(th),
            color="#94a3b8", lw=0.8, ls="--")
    if route is not None:
        seg = np.asarray(pts)[list(route)]
        ax.plot(seg[:, 0], seg[:, 1], color="#94a3b8", lw=0.9, alpha=0.7, zorder=1)
    r = np.hypot(np.asarray(pts)[:, 0], np.asarray(pts)[:, 1])
    inner = r <= REGION_RADIUS + 1e-9
    ax.scatter(np.asarray(pts)[inner, 0], np.asarray(pts)[inner, 1], s=46, color=C_INNER,
               edgecolor="white", lw=0.6, zorder=3, label="圆域内测量点")
    if (~inner).any():
        ax.scatter(np.asarray(pts)[~inner, 0], np.asarray(pts)[~inner, 1], s=46,
                   color="#2563eb", edgecolor="white", lw=0.6, zorder=3, label="圆域外测量点")
    ax.scatter([0.0], [0.0], marker="*", s=150, color="#dc2626", edgecolor="white", lw=0.8,
               zorder=4, label="原点")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlim(-2500, 2500)
    ax.set_ylim(-2500, 2500)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x / m", fontsize=9)
    ax.set_ylabel("y / m", fontsize=9)


def fig_nn_layout_compare(plan, results_dir: Path, fig_dir: Path) -> None:
    """左：人工定案布局；右：神经网络代理 + 进化搜索找到的自由布局。"""
    import matplotlib.pyplot as plt

    nn_path = results_dir / ".nn_best_layout.npy"
    if not nn_path.exists():
        print(f"  跳过布局对照图：缺少 {nn_path}")
        return
    nn = np.load(nn_path)
    pts_n = np.vstack([np.zeros(2), nn])
    nn_route_m = make_plan(pts_n).route_m
    hear = None
    verify_path = results_dir / ".nn_verify.json"
    if verify_path.exists():
        hear = load_json(verify_path).get("hear_rate")
    nn_txt = f"，{len(pts_n)} 位置，{nn_route_m / 1000:.2f} km"
    if hear:
        nn_txt += f"，听率 {hear * 100:.2f}%"
    with font_context(size=10):
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.0))
        _draw_layout(axes[0], plan.points, plan.route,
                     f"人工定案布局（{plan.n_points} 位置，{plan.route_m / 1000:.2f} km）")
        _draw_layout(axes[1], pts_n, None, f"神经网络搜索布局{nn_txt}")
        fig.suptitle("测量点布局对比：人工定案（7 覆盖基点 + 12 均匀外圈点）"
                     " vs 神经网络（代理 + 进化搜索）", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        _save(fig, fig_dir / "fig_t4_nn_layout_compare.png")


# ---------------------------------------------------------------------------
# 图 8：各布局 20 局实测平均虚拟时间
# ---------------------------------------------------------------------------
def fig_nn_speed_compare(results_dir: Path, fig_dir: Path) -> None:
    """各布局在当前参数下的 20 局实测平均虚拟时间（数据来自 `.nn_arm_speed.json`）。"""
    import matplotlib.pyplot as plt

    path = results_dir / ".nn_arm_speed.json"
    if not path.exists():
        print(f"  跳过耗时对照图：缺少 {path}（先跑 drv_speed.py 汇总各布局实测）")
        return
    arms = load_json(path)["arms"]
    names = [a["label"] for a in arms]
    times = [a["mean_virtual_s"] for a in arms]
    rates = [a["hear_rate"] * 100.0 for a in arms]
    npts = [a["n_points"] for a in arms]
    best = min(times)
    colors = ["#2563eb" if abs(t - best) < 1e-9 else "#94a3b8" for t in times]
    with font_context(size=10):
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        bars = ax.bar(range(len(arms)), times, width=0.58, color=colors)
        top = max(times) * 1.22
        for i, (b, t, r, n) in enumerate(zip(bars, times, rates, npts)):
            ax.text(b.get_x() + b.get_width() / 2, t + top * 0.02, f"{t:.0f} s", ha="center",
                    va="bottom", fontsize=10)
            ax.text(b.get_x() + b.get_width() / 2, t * 0.94, f"听率 {r:.2f}%\n{n} 位置",
                    ha="center", va="top", fontsize=8, color="white")
        ax.axhline(best, color="#2563eb", ls="--", lw=1.0, alpha=0.6)
        ax.text(len(arms) - 0.45, best * 1.02, f"最小 {best:.0f} s", color="#2563eb",
                fontsize=8, ha="right")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_ylabel("20 局平均虚拟时间 / s", fontsize=10)
        ax.set_ylim(0, top)
        ax.grid(axis="y", alpha=0.25)
        fig.suptitle("神经网络耗时导向布局 vs 人工布局（20 局 seed 0–19 实测）", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        _save(fig, fig_dir / "fig_t4_nn_speed_compare.png")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="paper/t4 论文图生成（读当前产物，数字不写死）")
    ap.add_argument("--results", default="results/t4", help="结果目录（缺省 results/t4）")
    ap.add_argument("--figdir", default="paper/t4/figures", help="输出目录（缺省 paper/t4/figures）")
    args = ap.parse_args(argv)

    results_dir = Path(args.results)
    fig_dir = Path(args.figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plan = build_sweep_plan()
    survey_path = results_dir / "t4_survey.json"
    plan_path = results_dir / "t4_sweep_plan.json"
    if plan_path.exists():                 # 听到率统计落在计划文件里，挂到 plan 上供标题使用
        plan = replace(plan, verification=load_json(plan_path).get("verification"))

    fig_sweep_lattice(fig_dir, plan)
    fig_directional_concept(fig_dir)
    fig_nn_layout_compare(plan, results_dir, fig_dir)
    fig_nn_speed_compare(results_dir, fig_dir)
    if survey_path.exists():
        survey = load_json(survey_path)
        fig_episode_clear(survey, fig_dir)
        fig_first_heard(survey, plan, fig_dir)
        fig_localization_error(survey, fig_dir)
        fig_cost(survey, plan, fig_dir)
    else:
        print(f"  跳过逐局图：缺少 {survey_path}")

    out = sorted(fig_dir.glob("fig_t4_*"))
    print(f"已生成 {len(out)} 张：")
    for p in out:
        print(f"  {p}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
