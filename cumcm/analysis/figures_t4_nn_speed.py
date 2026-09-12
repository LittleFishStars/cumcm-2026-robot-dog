"""figures_t4_nn_speed.py：NN 耗时导向布局 vs 人工布局的 20 局虚拟时间对比（论文图 6.7）。

数据全部来自 20 局固定 seed 0-19 的演练实测（results/t4/.speed20_p1.log、
.speed20_nn11.log 与 310aa3d 一期 23 点基线）：
  人工 23 点定案 6930s / 人工 20 点版(12 均匀环+7 基点) 6850s /
  NN 12 外圈非均匀 7318s / NN 11 外圈非均匀 7079s。
运行：python -X utf8 -m cumcm.analysis.figures_t4_nn_speed
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cumcm.common.plotting import font_context, save_png

OUT = Path("figures")

# (名称, 20 局平均虚拟时间 s, 听率%, 位置数, 测向注释)
ROWS = [
    ("人工 23 点\n(定案)", 6930, 99.9974, 23),
    ("人工 20 点版\n(12 均匀环+7 基点)", 6850, 99.9885, 20),
    ("神经网络 12 外圈\n(非均匀摆位)", 7318, 99.93, 20),
    ("神经网络 11 外圈\n(非均匀摆位)", 7079, 99.81, 19),
]


def main() -> None:
    names = [r[0] for r in ROWS]
    times = [r[1] for r in ROWS]
    rates = [r[2] for r in ROWS]
    npts = [r[3] for r in ROWS]
    best = min(times)
    with font_context(size=10):
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        bars = ax.bar(range(len(ROWS)), times, width=0.58,
                      color=["#94a3b8", "#2563eb", "#d97706", "#d97706"])
        for i, (b, t, r, n) in enumerate(zip(bars, times, rates, npts)):
            ax.text(i, t + 30, f"{t} s", ha="center", va="bottom", fontsize=10)
            ax.text(i, t - 180, f"听率 {r}%\n{n} 位置",
                    ha="center", va="top", fontsize=8, color="white")
        ax.axhline(best, color="#2563eb", ls="--", lw=1.0, alpha=0.6)
        ax.text(3.35, best + 15, f"最小 {best} s", color="#2563eb", fontsize=8)
        ax.set_xticks(range(len(ROWS)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("20 局平均虚拟时间 / s", fontsize=10)
        ax.set_ylim(0, 8600)
        ax.grid(axis="y", alpha=0.25)
        fig.suptitle("神经网络耗时导向布局 vs 人工布局（20 局 seed 0-19 实测）",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_png(fig, OUT / "fig_t4_nn_speed_compare.png", dpi=200)
        print("已写出：figures/fig_t4_nn_speed_compare.png")


if __name__ == "__main__":
    main()
