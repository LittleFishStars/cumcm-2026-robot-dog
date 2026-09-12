"""figures_t4_nn.py：神经网络搜索布局 vs 手工优化布局 对比图（论文第 6 章图 6.6）。

左：手工 23 点（原点 + 7 基点 + 3 内部补点 + 12 外圈点，结果见 t4_sweep_plan.json）；
右：神经网络代理 + 精英保真遗传算法搜索的 23 点（results/t4/.nn_best_layout.npy）。
两者同点数，直接对比"结构是否同构 / 覆盖侧重差异"。

运行：python -X utf8 -m cumcm.analysis.figures_t4_nn
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cumcm.common.plotting import font_context
from cumcm.t4.config import REGION_RADIUS
from cumcm.t4.sweep import measure_layout, build_sweep_plan

OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)


def _draw_ax(ax, pts, route, title):
    th = np.linspace(0.0, 2.0 * math.pi, 361)
    ax.plot(REGION_RADIUS * np.cos(th), REGION_RADIUS * np.sin(th),
            color="#334155", lw=1.2)
    ax.plot((REGION_RADIUS - 30.0) * np.cos(th), (REGION_RADIUS - 30.0) * np.sin(th),
            color="#94a3b8", lw=0.8, ls="--")
    r = np.hypot(pts[:, 0], pts[:, 1])
    inner = r <= REGION_RADIUS + 1e-9
    # 访问路径
    if route is not None:
        seg = pts[route]
        ax.plot(seg[:, 0], seg[:, 1], color="#94a3b8", lw=0.9, alpha=0.7,
                zorder=1)
    ax.scatter(pts[inner, 0], pts[inner, 1], s=46, color="#d97706",
               edgecolor="white", lw=0.6, zorder=3, label="圆域内测量点")
    outer = ~inner
    if outer.any():
        ax.scatter(pts[outer, 0], pts[outer, 1], s=46, color="#2563eb",
                   edgecolor="white", lw=0.6, zorder=3, label="圆域外测量点")
    ax.scatter([0.0], [0.0], marker="*", s=150, color="#dc2626",
               edgecolor="white", lw=0.8, zorder=4, label="原点")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlim(-2500, 2500)
    ax.set_ylim(-2500, 2500)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x / m", fontsize=9)
    ax.set_ylabel("y / m", fontsize=9)


def main() -> None:
    manual = build_sweep_plan()
    pts_m = manual.points
    nn = np.load("results/t4/.nn_best_layout.npy")
    pts_n = np.vstack([np.zeros(2), nn])

    with font_context(size=10):
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.0))
        _draw_ax(axes[0], pts_m, manual.route,
                 "手工优化布局（23 位置，17 358 m）")
        _draw_ax(axes[1], pts_n, None,
                 "神经网络搜索布局（23 位置，19 658 m）")
        fig.suptitle("测量点布局对比：手工（文献多环带） vs 神经网络（代理 + 进化搜索）",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = OUT / "fig_t4_nn_layout_compare.png"
        fig.savefig(out, dpi=200)
        print(f"已写出：{out}")


if __name__ == "__main__":
    main()