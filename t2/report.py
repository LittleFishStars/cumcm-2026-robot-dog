"""问题二的成果落盘：CSV 适合度表、JSON 结论与控制台报表"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from t2 import config as cfg
from t2.score import SolveResult, suitability

__all__ = ["save_map_csv", "save_summary_json", "print_report", "write_outputs"]


def save_map_csv(save_dir: Path, result: SolveResult, name: str = cfg.MAP_CSV) -> Path:
    """把全域粗网格的适合度场写成 CSV，列为 x, y, 到源集最坏距离, 可测, J, F"""
    path = Path(save_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    x = result.fields["x"].ravel()
    y = result.fields["y"].ravel()
    hear = result.fields["hear"].ravel()
    feas = result.fields["feasible"].ravel()
    j = result.fields["j"].ravel()
    f = suitability(result.fields["j"], result.j_star).ravel()
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x_m", "y_m", "worst_source_dist_m", "testable", "worst_diam_m",
                    "suitability"])
        for i in range(x.size):
            w.writerow([f"{x[i]:.1f}", f"{y[i]:.1f}", f"{hear[i]:.1f}", int(feas[i]),
                        f"{j[i]:.2f}", f"{f[i]:.6f}"])
    return path


def save_summary_json(save_dir: Path, result: SolveResult,
                      name: str = cfg.SUMMARY_JSON) -> Path:
    """把结论与校验写成 JSON。报告和论文引用的数字统一从这里取"""
    path = Path(save_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    scenario = {k: v for k, v in result.scenario.items() if k != "quad"}
    payload: dict[str, Any] = {
        "problem": "CUMCM 2026 B 问题二：第二检测点的选择",
        "first_site_m": list(result.site),
        "first_bearing_deg": result.theta1,
        "second_site_best_m": list(result.best),
        "second_site_best_polar": {"r_m": result.best_r, "phi_from_bearing_deg": result.best_phi},
        "worst_diameter_best_m": result.j_star,
        "single_measurement_diameter_m": result.single_m,
        "improvement_factor": result.improvement,
        "candidate_region": {
            "criterion": f"J <= (1+eta)*J*, eta = {result.checks['eta']}",
            "level_m": result.band.level_m,
            "r_range_m": [result.band.r_lo, result.band.r_hi],
            "lobes": [{"phi_from_bearing_deg": [l["phi_lo"], l["phi_hi"]],
                       "r_m": [l["r_lo"], l["r_hi"]], "area_km2": l["area_m2"] / 1e6}
                      for l in result.band.lobes],
            "total_area_km2": result.band.area_m2 / 1e6,
        },
        "feasible_lens": {"area_km2": result.lens_area_m2 / 1e6,
                          "bounds_m": [list(result.lens.bounds[:2]), list(result.lens.bounds[2:])]},
        "source_uncertainty": {"d_range_m": [result.checks["d_lo"], result.checks["d_hi"]],
                               "half_width_deg": cfg.BEARING_ERROR_DEG,
                               "n_samples": result.checks["n_source_samples"],
                               "corners_m": result.corners.tolist()},
        "worst_case": scenario,
        "literature_criteria": _theory_json(result),
        "checks": result.checks,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _theory_json(result: SolveResult) -> dict[str, Any]:
    """文献判据层里能序列化的那部分。探针数组只拿来出图，不进 JSON"""
    thy = dict(result.theory)
    thy.pop("probe", None)
    return thy


def print_report(result: SolveResult, figure: Path | None = None,
                 files: Sequence[Path] = ()) -> None:
    """控制台报表：结论、关键数字、校验状态。异常直接标 ⚠"""
    c = result.checks
    x, y = result.best
    sc = result.scenario
    print("问题二：第二检测点的选择与候选区域")
    print("=" * 74)
    print(f"输入：S1 = ({result.site[0]:.0f}, {result.site[1]:.0f}) m，"
          f"θ1 = {result.theta1:.1f}°（误差 ±{cfg.BEARING_ERROR_DEG:.0f}°）；"
          f"源到 S1 距离 d ∈ [{c['d_lo']:.0f}, {c['d_hi']:.0f}] m")
    print(f"可测性：要求对楔形内任意源都有 |S2−G| ≤ {cfg.RECEIVE_MIN:.0f} m（接收半径下界）")
    print(f"  可行域（透镜）面积 {result.lens_area_m2 / 1e6:.3f} km²"
          f"；最优点对最坏源的距离 {c['best_hears_worst_source_m']:.1f} m")
    print("-" * 74)
    print(f"最优第二检测点 S2* = ({x:.0f}, {y:.0f}) m：距 S1 {result.best_r:.0f} m、"
          f"相对示向度 {result.best_phi:+.1f}°")
    print(f"  最坏定位直径 J* = {result.j_star:.1f} m（最坏：源距 S1 {sc['d_m']:.0f} m、"
          f"δ1 = {sc['delta1_deg']:+.0f}°、δ2 = {sc['delta2_deg']:+.0f}°、"
          f"交会角 {sc['gamma_deg']:.1f}°）")
    print(f"  对照：只做这一次测向时定位区域直径 {result.single_m:.0f} m → 最坏直径缩小 "
          f"{result.improvement:.1f} 倍")
    print(f"候选区域（J ≤ {result.band.level_m:.0f} m，即最优值的 "
          f"{1.0 + c['eta']:.2f} 倍）：{result.band.describe()}")
    for i, lobe in enumerate(result.band.lobes, 1):
        print(f"  第 {i} 瓣：r ∈ [{lobe['r_lo']:.0f}, {lobe['r_hi']:.0f}] m，方位差 ∈ "
              f"[{lobe['phi_lo']:+.1f}°, {lobe['phi_hi']:+.1f}°]，面积 "
              f"{lobe['area_m2'] / 1e6:.3f} km²")
    print("-" * 74)
    thy = result.theory
    if thy:
        ce = thy["certify"]
        cr = thy["criteria"]
        print("文献判据（GDOP / CRLB / 几何稀释）与全局认证：")
        print(f"  全域认证：可行域包围盒 {ce['step_m']:.0f} m 细网格 {ce['n_grid']} 点 → "
              f"GDOP 快筛取 {ce['n_exact']} 个候选精确复核 → 细化后 J = {ce['worst_diam_m']:.4f} m")
        print(f"    与 25 m 粗搜+细化解的距离 {ce['vs_coarse_m']:.1f} m、"
              f"最坏直径差 {ce['vs_coarse_j_m']:+.2e} m（未漏掉更好的盆地）")
        print(f"  判据一致性：可行域内 {cr['n_probe']} 个采样点上，文献 GDOP 与本文精确直径的 "
              f"Spearman 秩相关 = {cr['gdop_vs_exact_spearman']:.3f}")
        wc = cr["worst_case_geometry"]
        print(f"  最坏情形（d = {wc['d_m']:.0f} m、r2 = {wc['r2_m']:.0f} m、γ = "
              f"{wc['gamma_deg']:.1f}°）的半径：精确构造 {wc['exact_worst_radius_m']:.2f} m；"
              f"本文闭式 {cr['closed_form_m']['minimax_radius_m']:.2f} m"
              f"（{cr['closed_form_rel_dev']['minimax_radius_m']*100:+.1f}%）、"
              f"GDOP {cr['closed_form_m']['gdop_m']:.2f} m"
              f"（{cr['closed_form_rel_dev']['gdop_m']*100:+.1f}%）。文献式偏低，只能当快筛")
        print("  准则对照（同一张表三个口径互评）：")
        for name, tag in (("minimax", "本文 minimax"), ("gdop", "文献 GDOP  "),
                          ("expected", "期望口径  ")):
            q = thy["points"][name]
            print(f"    {tag} ({q['xy_m'][0]:6.0f}, {q['xy_m'][1]:5.0f}) r = {q['r_m']:.0f} m、"
                  f"φ = {q['phi_deg']:+.1f}°：最坏直径 {q['worst_diam_m']:.2f} m"
                  f"（比最优差 {q['worst_diam_loss_pct']:+.2f}%）、期望直径 {q['mean_diam_m']:.2f} m、"
                  f"GDOP {q['gdop_m']:.1f} m")
    print("-" * 74)
    ok = lambda b: "✓" if b else "⚠ 需检查"          # noqa: E731（局部小工具，意思一眼就懂）
    print("校验：")
    print(f"  {ok(c['scenario_rel_dev'] < 1e-9)} 最坏情形：解析构造 {c['scenario_analytic_m']:.4f} m"
          f" vs shapely 精确 {c['scenario_exact_m']:.4f} m（相对偏差 {c['scenario_rel_dev']:.1e}，"
          f"顶点 {c['scenario_n_vertices']} 个，圆域截断 {c['scenario_clipped']}）")
    print(f"  {ok(c['discretisation_rel_dev'] < 1e-3)} 离散化加密 "
          f"{cfg.DENSE_FACTOR}× 后 J* = {c['discretisation_dense_m']:.4f} m"
          f"（相对偏差 {c['discretisation_rel_dev']:.1e}）")
    print(f"  {ok(c['band_radial_gaps'] == 0)} 候选弧带：每个方位角上径向区间连续"
          f"（断口 {c['band_radial_gaps']} 处）")
    print(f"  {ok(c['certify_j_rel_dev'] < 1e-9)} 全局认证（细网格快筛+精确复核）"
          f"J = {c['certify_worst_diam_m']:.4f} m，与粗搜细化解相对偏差 {c['certify_j_rel_dev']:.1e}")
    if "analytic_vs_shapely" in c:
        v = c["analytic_vs_shapely"]
        print(f"  {ok(v['impl_max_abs_rel_dev'] < 1e-9)} 四边形构造 vs shapely："
              f"{v['n_used']} 个有效样本，最大相对偏差 {v['impl_max_abs_rel_dev']:.1e}"
              f"（另有圆域截断 {v['n_clipped']} 例、近共线 {v['n_degenerate']} 例按规则跳过）")
        print(f"  {ok(v['formula_frac_below_1pct'] > 0.95)} 一阶公式（论文引用）：中位偏差 "
              f"{v['formula_median_rel_dev']:.3%}，<1% 占 {v['formula_frac_below_1pct']:.0%}")
    if "theory_closed_forms" in c:
        tv = c["theory_closed_forms"]
        print(f"  {ok(tv['closed_vs_eigen_max_rel_dev'] < 1e-9)} 文献闭式 vs 数值特征值："
              f"最大相对偏差 {tv['closed_vs_eigen_max_rel_dev']:.1e}（{tv['n_used']} 个有效样本）")
        print(f"  {ok(tv['minimax_closed_median_rel_dev'] < 0.01)} 本文闭式 vs 精确最坏半径："
              f"中位 {tv['minimax_closed_median_rel_dev']:.2%}；文献 GDOP 类判据中位偏差 "
              f"{tv['gdop_vs_exact_median_rel_dev']:+.1%}（低估了，所以不能作保险判据）")
    for p in list(files) + ([figure] if figure else []):
        print(f"  → {p}")


def write_outputs(save_dir: Path, result: SolveResult, figure: Path | None = None,
                  csv_name: str = cfg.MAP_CSV, json_name: str = cfg.SUMMARY_JSON) -> dict[str, Path]:
    """一次写全 CSV 与 JSON，返回路径表。出图与否由调用方决定，好配合 --no-plot"""
    save_dir = Path(save_dir)
    return {"csv": save_map_csv(save_dir, result, csv_name),
            "json": save_summary_json(save_dir, result, json_name)}
