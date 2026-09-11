"""由问题三的运行结果生成论文用数据型图表（PDF 矢量图 + 背后的数据 CSV）。

数据来源全部是已落盘的产物，本脚本不重新求解、不连接模拟器：
    results/ga_training.json     GA 训练记录 + 逐局统计（含干扰源真值）
    results/ga_convergence.csv   逐代收敛曲线
    results/episodes.csv         逐局统计（同上，便于绘图脚本直接读表）
    results/api_calls.jsonl      逐次接口调用（用于重建机器狗轨迹）
    results/validation.json      验证结果（σ 标定、参数敏感性等）
    results/holdout/             独立 seed 批（泛化检验）

用法：
    python T3_figures.py                    # 生成全部图表到 figures/
    python T3_figures.py --figures figures  # 指定输出目录

注意：本模块自带一套排版样式（字号 10.5、开网格、配色等），与 common.plotting 的默认样式
**不同**，且已写进论文，故不并入公共样式，只复用其中的环境设置与中文字体回退链。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from cumcm.common.console import relax_console_encoding
from cumcm.common.plotting import setup_mpl_env
from cumcm.t3ga.covering import covering_waypoints
from cumcm.t3ga.config import COVER_RADIUS, REGION_RADIUS

# matplotlib 配置目录统一交给公共层（原先是 mkdtemp，每次运行换目录、字体缓存反复重建）
setup_mpl_env()

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 10.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})

C_SRC = "#2f6fb5"        # 干扰源 / 主色
C_CLEAR = "#2e9e5b"      # 成功清除
C_MEAS = "#c8871b"       # 测向 / 次色
C_WARN = "#c0392b"       # 阈值 / 失败
C_GRAY = "#6b7280"
C_BAND = "#9dbfe0"

# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_calls(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def dump_data(out_dir: Path, name: str, header: Sequence[str],
              rows: Sequence[Sequence[Any]]) -> None:
    """同步保存图表背后的数据，便于论文复核与复现。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / name).open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def save(fig, path: Path) -> None:
    """存图。显式清掉元数据里的生成时间，使同一份输入每次产出逐字节一致的 PDF——
    否则每重新出一次图 PDF 都因内嵌时间戳而变，既无法验证可复现性，也污染版本历史。
    """
    meta = {"CreationDate": None} if path.suffix.lower() == ".pdf" else None
    fig.savefig(path, metadata=meta)
    plt.close(fig)
    print(f"  √ {path}")


# ---------------------------------------------------------------------------
# 图 1：逐局清除完成情况
# ---------------------------------------------------------------------------
def fig_episode_clear(train: Dict[str, Any], fig_dir: Path, data_dir: Path) -> None:
    eps = train["episodes"]
    n = [e["n_sources"] for e in eps]
    c = [e["cleared"] for e in eps]
    x = np.arange(len(eps))

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.bar(x, n, width=0.66, color=C_BAND, edgecolor=C_SRC, linewidth=0.7,
           label="干扰源数量")
    ax.bar(x, c, width=0.34, color=C_CLEAR, edgecolor="white", linewidth=0.5,
           label="成功清除数量")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e["seed"]) for e in eps], fontsize=8.5)
    ax.set_xlabel("随机场景编号（seed）")
    ax.set_ylabel("干扰源个数 / 个")
    ax.set_ylim(0, max(n) * 1.28)
    ax.legend(loc="upper left", ncol=2)
    tot, cl = sum(n), sum(c)
    ax.text(0.995, 0.94, f"合计清除 {cl}/{tot}（{cl / tot * 100:.1f}%）",
            transform=ax.transAxes, ha="right", va="top", color=C_CLEAR, fontsize=9.5)
    save(fig, fig_dir / "fig_t3_episode_clear.pdf")
    dump_data(data_dir, "episode_clear.csv",
              ["episode", "seed", "n_sources", "cleared"],
              [(e["episode"], e["seed"], e["n_sources"], e["cleared"]) for e in eps])


# ---------------------------------------------------------------------------
# 图 2：逐局定位误差
# ---------------------------------------------------------------------------
def fig_localization_error(train: Dict[str, Any], fig_dir: Path, data_dir: Path) -> None:
    eps = [e for e in train["episodes"] if e["localize_err_mean_m"] is not None]
    mean = np.array([e["localize_err_mean_m"] for e in eps])
    mx = np.array([e["localize_err_max_m"] for e in eps])
    x = np.arange(len(eps))

    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.vlines(x, mean, mx, color=C_BAND, linewidth=2.4, zorder=1,
              label="该局误差范围（均值→最坏）")
    ax.plot(x, mean, "o", ms=4.6, color=C_SRC, zorder=3, label="逐局平均定位误差")
    ax.plot(x, mx, "s", ms=3.8, color=C_MEAS, zorder=3, label="逐局最坏定位误差")
    ax.axhline(20.0, color=C_WARN, ls="--", lw=1.1)
    ax.text(0.4, 20.8, "清除半径 20 m", color=C_WARN, fontsize=9, ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e["seed"]) for e in eps], fontsize=8.5)
    ax.set_xlabel("随机场景编号（seed）")
    ax.set_ylabel("定位误差 / m")
    ax.set_ylim(0, max(mx.max() * 1.30, 25.0))
    ax.legend(loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.22))
    save(fig, fig_dir / "fig_t3_localization_error.pdf")
    dump_data(data_dir, "localization_error.csv",
              ["episode", "seed", "err_mean_m", "err_max_m"],
              [(e["episode"], e["seed"], round(e["localize_err_mean_m"], 4),
                round(e["localize_err_max_m"], 4)) for e in eps])


# ---------------------------------------------------------------------------
# 图 3：逐局成本（虚拟时间与测向次数）
# ---------------------------------------------------------------------------
def fig_cost(train: Dict[str, Any], fig_dir: Path, data_dir: Path) -> None:
    eps = train["episodes"]
    x = np.arange(len(eps))
    vt = np.array([e["virtual_time_s"] for e in eps])
    per = np.array([e["avg_time_s"] for e in eps])
    meas = np.array([e["n_measure"] for e in eps])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3))
    fig.subplots_adjust(wspace=0.46)

    ax = axes[0]
    ax.bar(x, vt, width=0.62, color=C_BAND, edgecolor=C_SRC, linewidth=0.7)
    ax.set_ylabel("虚拟时间 / s", color=C_SRC)
    ax.tick_params(axis="y", labelcolor=C_SRC)
    ax.set_ylim(0, max(vt) * 1.25)
    ax2 = ax.twinx()
    ax2.plot(x, per, "o-", ms=4.2, lw=1.3, color=C_MEAS, label="平均每个源耗时")
    ax2.axhline(per.mean(), color=C_MEAS, ls=":", lw=1.0)
    ax2.set_ylabel("平均每源耗时 / s", color=C_MEAS, labelpad=8)
    ax2.tick_params(axis="y", labelcolor=C_MEAS, labelsize=9)
    ax2.set_yticks([0, 200, 400])
    ax2.grid(False)
    ax2.set_ylim(0, max(per) * 1.35)
    ax2.text(len(eps) - 0.5, per.mean() * 1.03, f"均值 {per.mean():.0f} s",
             color=C_MEAS, fontsize=9, ha="right", va="bottom")
    ax.set_xlabel("随机场景编号（seed）")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e["seed"]) for e in eps], fontsize=8)
    ax.set_title("虚拟时间开销", fontsize=10.5)

    ax = axes[1]
    ax.bar(x, meas, width=0.62, color=C_MEAS, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.axhline(meas.mean(), color=C_SRC, ls=":", lw=1.2)
    ax.text(len(eps) - 0.5, meas.mean() + 3, f"均值 {meas.mean():.0f} 次",
            color=C_SRC, fontsize=9, ha="right", va="bottom")
    ax.set_ylabel("测向（/measure）次数 / 次")
    ax.set_xlabel("随机场景编号（seed）")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e["seed"]) for e in eps], fontsize=8)
    ax.set_ylim(0, max(meas) * 1.22)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_title("测向通信开销", fontsize=10.5)

    save(fig, fig_dir / "fig_t3_cost.pdf")
    dump_data(data_dir, "cost.csv",
              ["episode", "seed", "virtual_time_s", "avg_time_per_source_s", "n_measure",
               "n_clear"],
              [(e["episode"], e["seed"], round(e["virtual_time_s"], 3),
                round(e["avg_time_s"], 3), e["n_measure"], e["n_clear"]) for e in eps])


# ---------------------------------------------------------------------------
# 图 4：定位 GA 收敛曲线
# ---------------------------------------------------------------------------
def _curve(rows: List[Dict[str, str]], ga: str, label: str | None = None
           ) -> Dict[int, List[float]]:
    """把收敛记录整理成 {代数: [各次运行的最优值]}。"""
    out: Dict[int, List[float]] = {}
    for r in rows:
        if r["ga"] != ga:
            continue
        if label is not None and r["label"] != label:
            continue
        out.setdefault(int(r["generation"]), []).append(float(r["best"]))
    return out


def fig_ga_convergence_loc(train: Dict[str, Any], rows: List[Dict[str, str]],
                           fig_dir: Path, data_dir: Path) -> None:
    full = [r for r in train["ga_runs"] if r["ga"] == "localize" and not r["converged"]]
    gens = sorted(_curve([r for r in rows if r["ga"] == "localize"], "localize").keys())
    med, q1, q3, best = [], [], [], []
    by_gen = _curve([r for r in rows if r["ga"] == "localize"], "localize")
    for g in gens:
        v = np.array(by_gen[g])
        med.append(float(np.median(v)))
        q1.append(float(np.percentile(v, 25)))
        q3.append(float(np.percentile(v, 75)))
        best.append(float(v.min()))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))

    ax = axes[0]
    ax.fill_between(gens, q1, q3, color=C_BAND, alpha=0.55,
                    label="四分位区间（全部定位任务）")
    ax.plot(gens, med, lw=1.6, color=C_SRC, label="中位数")
    ax.plot(gens, best, lw=1.2, color=C_CLEAR, ls="--", label="最好的一次")
    ax.axhline(1e-4, color=C_WARN, ls=":", lw=1.0)
    ax.text(gens[-1], 1.2e-4, "收敛判据 $10^{-4}$", color=C_WARN, fontsize=8.5,
            ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("进化代数")
    ax.set_ylabel("适应度（示向度残差 RMS）/ 度")
    ax.set_title("定位 GA 适应度收敛", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper right")

    # 右图：初始最好值 → 最终值的改进分布
    ax = axes[1]
    i0 = np.array([r["initial_best_g0"] for r in full]) if full else np.array([0.0])
    i1 = np.array([r["final_fitness"] for r in full]) if full else np.array([0.0])
    gain = (i0 - i1) / i0 * 100.0
    ax.hist(gain, bins=18, color=C_BAND, edgecolor=C_SRC, linewidth=0.7)
    ax.axvline(float(np.mean(gain)), color=C_WARN, ls="--", lw=1.2)
    ax.text(float(np.mean(gain)), ax.get_ylim()[1] * 0.92,
            f"均值 {np.mean(gain):.1f}%", color=C_WARN, fontsize=9, ha="left",
            va="top", rotation=90)
    ax.set_xlabel("相对初始种群的适应度改进 / %")
    ax.set_ylabel("定位任务数 / 次")
    ax.set_title(f"跑满 {int(np.median([r['generations'] for r in full]) if full else 0)} 代的任务收益",
                 fontsize=10.5)

    save(fig, fig_dir / "fig_t3_ga_convergence_loc.pdf")
    dump_data(data_dir, "ga_convergence_loc.csv",
              ["generation", "q1", "median", "q3", "best"],
              [(gens[i], round(q1[i], 8), round(med[i], 8), round(q3[i], 8),
                round(best[i], 8)) for i in range(len(gens))])


# ---------------------------------------------------------------------------
# 图 5：路线 GA 收敛与改进
# ---------------------------------------------------------------------------
def fig_ga_convergence_route(train: Dict[str, Any], rows: List[Dict[str, str]],
                             fig_dir: Path, data_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))

    # 左图：两类路线任务的相对超额率收敛曲线（消除绝对里程差异）
    ax = axes[0]
    styles = {"覆盖路点巡回": (C_SRC, "-", "覆盖路点巡回"), "清除顺序": (C_MEAS, "-", "清除顺序")}
    for label, (color, ls, name) in styles.items():
        runs = _curve(rows, "route", label)
        if not runs:
            continue
        gens = sorted(runs)
        rel = []
        for g in gens:
            # 以每条运行各自的最终值为基准，换算成"相对最优的超出率"
            finals = [r["final_length_m"] for r in train["ga_runs"]
                      if r["ga"] == "route" and r["label"] == label]
            base = float(np.median(finals))
            rel.append(float(np.median([(v - base) / base * 100.0 for v in runs[g]])))
        ax.plot(gens, rel, lw=1.7, color=color, ls=ls, label=name)
    ax.set_xlabel("进化代数")
    ax.set_ylabel("相对最优解的路程超出率 / %")
    ax.set_title("路线 GA 收敛过程", fontsize=10.5)
    ax.legend(fontsize=9)

    # 右图：初始解 → GA 终值的改进率分布（抖动散点 + 箱线，避免点重叠在一根线上）
    ax = axes[1]
    runs = [r for r in train["ga_runs"] if r["ga"] == "route"]
    rng = np.random.default_rng(7)
    names = ["覆盖路点巡回", "清除顺序"]
    data, colors = [], []
    for label in names:
        sub = [r for r in runs if r["label"] == label]
        data.append([(r["initial_best_g0"] - r["final_length_m"]) / r["final_length_m"] * 100.0
                     for r in sub])
        colors.append(styles[label][0])
    bp = ax.boxplot(data, vert=False, widths=0.42, patch_artist=True,
                    showfliers=False, medianprops=dict(color="white", lw=1.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, alpha=0.30, edgecolor=c)
    for i, (v, c) in enumerate(zip(data, colors), start=1):
        v = np.asarray(v, dtype=float)
        ax.scatter(v, i + rng.uniform(-0.13, 0.13, len(v)), s=30, color=c, alpha=0.85,
                   edgecolor="white", linewidth=0.5, zorder=3)
        if len(v):
            ax.text(0.995, i + 0.34, f"中位 {np.median(v):.1f}%",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    color=c, fontsize=9)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(names)
    ax.set_ylim(0.5, 2.6)
    ax.set_xlabel("GA 相对初始解的路程改进率 / %")
    ax.set_title("初始解质量与 GA 的改进空间", fontsize=10.5)
    ax.grid(axis="y", visible=False)

    save(fig, fig_dir / "fig_t3_ga_convergence_route.pdf")
    out = []
    for r in runs:
        excess = (r["initial_best_g0"] - r["final_length_m"]) / r["final_length_m"] * 100.0
        out.append((r["episode"], r["label"], r["n_points"], round(r["initial_best_g0"], 2),
                    round(r["final_length_m"], 2), round(excess, 3)))
    dump_data(data_dir, "route_ga.csv",
              ["episode", "label", "n_points", "initial_m", "final_m", "improve_pct"], out)


# ---------------------------------------------------------------------------
# 图 6：σ 随交会几何的变化（验证 A 组结论）
# ---------------------------------------------------------------------------
def fig_sigma_geometry(val: Dict[str, Any], fig_dir: Path, data_dir: Path) -> None:
    stats = val["metrics"].get("a_sigma_by_geometry")
    if not stats:
        print("  ! 验证结果缺少 σ 标定数据，跳过 fig_t3_sigma_geometry.pdf")
        return
    order = ["全方位", "中等张角", "窄张角(病态)"]
    order = [k for k in order if k in stats] or list(stats)
    sig = [stats[k]["sigma_med"] for k in order]
    err = [stats[k]["err_med"] for k in order]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    w = 0.36
    ax.bar(x - w / 2, sig, w, color=C_BAND, edgecolor=C_SRC, linewidth=0.8,
           label="位置不确定度 1σ（估计值）")
    ax.bar(x + w / 2, err, w, color=C_MEAS, alpha=0.85, edgecolor="white", linewidth=0.6,
           label="实际定位误差（中位数）")
    ax.axhline(20.0, color=C_WARN, ls="--", lw=1.1)
    ax.text(0.985, 0.60, "清除半径 20 m", color=C_WARN, fontsize=9, ha="right",
            va="center", transform=ax.transAxes)
    for i, (s, e) in enumerate(zip(sig, err)):
        ax.text(i - w / 2, s * 1.06, f"{s:.0f}", ha="center", va="bottom", fontsize=9,
                color=C_SRC)
        ax.text(i + w / 2, e * 1.06, f"{e:.0f}", ha="center", va="bottom", fontsize=9,
                color=C_MEAS)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("距离 / m")
    ax.set_ylim(0, max(max(sig), max(err)) * 1.30)
    ax.legend(fontsize=9, loc="upper left")
    save(fig, fig_dir / "fig_t3_sigma_geometry.pdf")
    dump_data(data_dir, "sigma_geometry.csv",
              ["geometry", "sigma_median_m", "err_median_m"],
              [(k, round(stats[k]["sigma_med"], 3), round(stats[k]["err_med"], 3))
               for k in order])


# ---------------------------------------------------------------------------
# 图 7：覆盖路点布局与最坏点
# ---------------------------------------------------------------------------
def fig_coverage(fig_dir: Path, data_dir: Path) -> None:
    way = covering_waypoints()
    R = REGION_RADIUS

    # 连续圆域上的最坏点（2 m 网格初筛 + 局部爬山）
    ax_grid = np.arange(-R, R + 1e-9, 2.0)
    gx, gy = np.meshgrid(ax_grid, ax_grid)
    P = np.stack((gx.ravel(), gy.ravel()), axis=1)
    P = P[np.linalg.norm(P, axis=1) <= R]
    d = np.linalg.norm(P[:, None, :] - way[None, :, :], axis=2).min(axis=1)
    wp = P[int(np.argmax(d))].copy()
    worst = float(d.max())

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    th = np.linspace(0, 2 * math.pi, 400)

    ax = axes[0]
    ax.plot(R * np.cos(th), R * np.sin(th), color=C_GRAY, lw=1.2)
    ax.plot((R - 30) * np.cos(th), (R - 30) * np.sin(th), color=C_GRAY, lw=0.8, ls="--")
    ax.text(0, R - 60, "官方生成器源域边界（1770 m）", color=C_GRAY, fontsize=8.5,
            ha="center", va="top")
    for i, (wx, wy) in enumerate(way):
        ax.add_patch(plt.Circle((wx, wy), COVER_RADIUS, fill=False,
                                color=C_BAND, lw=0.7, alpha=0.9))
        ax.plot(wx, wy, "o", ms=6, color=C_SRC, zorder=3)
        ax.text(wx, wy, str(i), color="white", fontsize=7, ha="center", va="center",
                zorder=4)
    ax.plot([0], [0], marker="*", ms=13, color=C_MEAS, zorder=4, label="起点")
    ax.plot([wp[0]], [wp[1]], marker="X", ms=9, color=C_WARN, zorder=5,
            label=f"最坏点（{worst:.0f} m）")
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(f"覆盖路点布局（{len(way)} 个，半径 {COVER_RADIUS:.0f} m）", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[1]
    ax.hist(d, bins=120, color=C_BAND, edgecolor=C_SRC, linewidth=0.4)
    ax.axvline(COVER_RADIUS, color=C_SRC, ls="--", lw=1.2)
    ax.text(COVER_RADIUS - 8, ax.get_ylim()[1] * 0.95, f"设计半径 {COVER_RADIUS:.0f} m",
            color=C_SRC, fontsize=9, ha="right", va="top")
    ax.axvline(1000.0, color=C_WARN, ls="--", lw=1.3)
    ax.text(1000.0 - 8, ax.get_ylim()[1] * 0.78, "接收半径下界 1000 m",
            color=C_WARN, fontsize=9, ha="right", va="top")
    ax.set_xlabel("圆域内各点到最近路点的距离 / m")
    ax.set_ylabel("网格点数 / 个")
    ax.set_title("全域距离分布（无一处超出接收下界）", fontsize=10.5)

    save(fig, fig_dir / "fig_t3_coverage.pdf")
    dump_data(data_dir, "coverage_worst.csv",
              ["nominal_radius_m", "worst_point_x", "worst_point_y", "worst_dist_m",
               "min_reception_m"],
              [(R, round(float(wp[0]), 2), round(float(wp[1]), 2), round(worst, 3), 1000.0)])
    dump_data(data_dir, "coverage_waypoints.csv", ["index", "x", "y"],
              [(i, round(float(w[0]), 2), round(float(w[1]), 2)) for i, w in enumerate(way)])


# ---------------------------------------------------------------------------
# 图 8：单局机器狗轨迹（由接口日志重建）
# ---------------------------------------------------------------------------
def fig_trajectory(calls: List[Dict[str, Any]], train: Dict[str, Any], fig_dir: Path,
                   data_dir: Path, episode: int = 1) -> None:
    recs = sorted([c for c in calls if c["episode"] == episode], key=lambda r: r["seq"])
    pos = np.array([[0.0, 0.0]] + [[r["x"], r["y"]] for r in recs
                                   if r["call"] in ("/measure", "/clear")])
    cleared = [(r["x"], r["y"], r["channel"]) for r in recs
               if r["call"] == "/clear" and (r.get("response") or {}).get("clear_result") == "success"]
    truth = train["episodes"][episode - 1]["truth"]

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    th = np.linspace(0, 2 * math.pi, 400)
    ax.plot(1800 * np.cos(th), 1800 * np.sin(th), color=C_GRAY, lw=1.4)
    ax.plot(1770 * np.cos(th), 1770 * np.sin(th), color=C_GRAY, lw=0.8, ls="--")
    ax.plot(pos[:, 0], pos[:, 1], "-", lw=0.9, color=C_SRC, alpha=0.85,
            label="机器狗轨迹")
    for t in truth:
        ax.plot([t["x"]], [t["y"]], "x", ms=9, mew=1.8, color=C_WARN, zorder=4,
                label="干扰源真值" if t is truth[0] else None)
        ax.text(t["x"] + 26, t["y"] + 26, f"ch{t['channel']}", fontsize=7.6,
                color=C_WARN)
        ax.add_patch(plt.Circle((t["x"], t["y"]), 20.0, fill=False, color=C_WARN,
                                lw=0.6, alpha=0.6))
    if cleared:
        cx, cy, _ = zip(*cleared)
        ax.plot(cx, cy, "o", ms=6.5, mfc="none", mec=C_CLEAR, mew=1.5, zorder=5,
                label="清除动作落点")
    ax.plot([0], [0], marker="*", ms=15, color=C_MEAS, zorder=6, label="起点 / 结束点")
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.legend(fontsize=8.5, loc="upper center", ncol=2)
    save(fig, fig_dir / "fig_t3_trajectory.pdf")
    dump_data(data_dir, f"trajectory_ep{episode}.csv", ["step", "x", "y"],
              [(i, round(float(p[0]), 2), round(float(p[1]), 2)) for i, p in enumerate(pos)])


# ---------------------------------------------------------------------------
# 图 9：GA 参数敏感性
# ---------------------------------------------------------------------------
def fig_sensitivity(val: Dict[str, Any], fig_dir: Path, data_dir: Path) -> None:
    m = val["metrics"]
    loc_g = m.get("d_loc_gens") or {}
    loc_p = m.get("d_loc_pop") or {}
    r_g = m.get("d_route_gens") or {}
    if not (loc_g and r_g):
        print("  ! 验证结果缺少敏感性数据，跳过 fig_t3_sensitivity.pdf")
        return

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.1))

    def nums(d: Dict[str, float], prefix: str) -> List[int]:
        return [int(k[len(prefix):]) for k in d]

    def param_axis(ax, keys: Sequence[str], prefix: str, values: Sequence[float],
                   ylabel: str, title: str, color: str) -> None:
        """等距索引轴画参数扫描：log 轴会同时绘制主/次刻度标签而重叠，这里显式标注真值。"""
        x = np.arange(len(values))
        ax.plot(x, values, "o-", ms=5.5, color=color)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.4g}", ha="center", va="bottom",
                    fontsize=8.5, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([k[len(prefix):] for k in keys])
        ax.set_xlim(-0.5, len(values) - 0.5)
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * 0.35, hi * 0.02, 1e-9)
        ax.set_ylim(lo - pad, hi + pad * 2.2)
        ax.set_xlabel({"gens": "代数", "pop": "种群规模"}[prefix])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.3)

    ax = axes[0]
    param_axis(ax, list(loc_g), "gens", [loc_g[k] for k in loc_g],
               "平均定位误差 / m", "代数对定位精度的影响（几乎无影响）", C_SRC)

    ax = axes[1]
    param_axis(ax, list(loc_p), "pop", [loc_p[k] for k in loc_p],
               "平均定位误差 / m", "种群规模对定位精度的影响（几乎无影响）", C_MEAS)

    ax = axes[2]
    rg = [r_g[k] * 100.0 for k in r_g]
    param_axis(ax, list(r_g), "gens", rg,
               "相对穷举最优的超出率 / %", "代数对路线质量的影响（150 代后最优）", C_CLEAR)

    save(fig, fig_dir / "fig_t3_sensitivity.pdf")
    rows = [("localize", "gens", int(k[4:]), v) for k, v in loc_g.items()]
    rows += [("localize", "pop", int(k[3:]), v) for k, v in loc_p.items()]
    rows += [("route", "gens", int(k[4:]), v * 100.0) for k, v in r_g.items()]
    dump_data(data_dir, "sensitivity.csv", ["ga", "param", "value", "metric"], rows)


# ---------------------------------------------------------------------------
# 图 10：独立 seed 批的泛化检验
# ---------------------------------------------------------------------------
def fig_generalization(main: Dict[str, Any], hold: Dict[str, Any], fig_dir: Path,
                       data_dir: Path) -> None:
    groups = [("主训练批（seed 0–19）", main["episodes"], C_SRC),
              ("独立检验批（seed 100–109）", hold["episodes"], C_MEAS)]
    labels = ["清除比例 / %", "平均误差 / m", "最坏误差 / m",
              "平均每源耗时 / s", "每局测向次数 / 次"]

    fig, axes = plt.subplots(1, 5, figsize=(12.6, 3.0))
    vals = {}
    for name, eps, _ in groups:
        vals[name] = [
            float(np.mean([e["clear_ratio"] for e in eps])) * 100.0,
            float(np.mean([e["localize_err_mean_m"] for e in eps])),
            float(np.max([e["localize_err_max_m"] for e in eps])),
            float(np.mean([e["avg_time_s"] for e in eps])),
            float(np.mean([e["n_measure"] for e in eps])),
        ]
    for i, ax in enumerate(axes):
        ax.bar([0, 1], [vals[g[0]][i] for g in groups], width=0.6,
               color=[C_SRC, C_MEAS], edgecolor="white", linewidth=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["主批", "独立批"], fontsize=9)
        ax.set_title(labels[i], fontsize=10)
        top = max(vals[g[0]][i] for g in groups)
        for j, g in enumerate(groups):
            ax.text(j, vals[g[0]][i] * 1.02, f"{vals[g[0]][i]:.1f}", ha="center",
                    va="bottom", fontsize=9)
        ax.set_ylim(0, top * 1.22)
        ax.grid(axis="x", visible=False)
    fig.subplots_adjust(wspace=0.45)
    save(fig, fig_dir / "fig_t3_generalization.pdf")
    dump_data(data_dir, "generalization.csv", ["metric"] + [g[0] for g in groups],
              [(labels[i],) + tuple(round(vals[g[0]][i], 4) for g in groups)
               for i in range(len(labels))])


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成问题三论文图表（PDF）")
    ap.add_argument("--results", default="results", help="结果目录")
    ap.add_argument("--holdout", default="results/holdout", help="独立 seed 批结果目录")
    ap.add_argument("--figures", default="figures", help="图表输出目录")
    ap.add_argument("--data", default="figures/data", help="图表数据输出目录")
    args = ap.parse_args(argv)

    relax_console_encoding()         # 中文控制台下也能正常打印进度
    res, fig_dir, data_dir = Path(args.results), Path(args.figures), Path(args.data)
    fig_dir.mkdir(parents=True, exist_ok=True)

    train = load_json(res / "ga_training.json")
    rows = load_csv(res / "ga_convergence.csv")
    calls = load_calls(res / "api_calls.jsonl")
    val = load_json(res / "validation.json") if (res / "validation.json").exists() else {}

    print("生成图表：")
    fig_episode_clear(train, fig_dir, data_dir)
    fig_localization_error(train, fig_dir, data_dir)
    fig_cost(train, fig_dir, data_dir)
    fig_ga_convergence_loc(train, rows, fig_dir, data_dir)
    fig_ga_convergence_route(train, rows, fig_dir, data_dir)
    fig_sigma_geometry(val, fig_dir, data_dir)
    fig_coverage(fig_dir, data_dir)
    fig_trajectory(calls, train, fig_dir, data_dir)
    fig_sensitivity(val, fig_dir, data_dir)
    if Path(args.holdout, "ga_training.json").exists():
        fig_generalization(train, load_json(Path(args.holdout) / "ga_training.json"),
                           fig_dir, data_dir)
    print(f"图表数据已保存：{data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
