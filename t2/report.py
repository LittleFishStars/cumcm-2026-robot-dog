"""问题二的成果落盘：CSV 适合度表、JSON 结论与控制台报表"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from common.console import print_table
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
    """把结论与校验写成 JSON，报告与论文引用的数字统一从这里取"""
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
    """文献判据层里能序列化的那部分，探针数组只拿来出图，不进 JSON"""
    thy = dict(result.theory)
    thy.pop("probe", None)
    return thy


def _mark(ok: bool) -> str:
    """校验结果列的记号，通过打 ✓，不通过打 ✗"""
    return "✓" if ok else "✗ 需检查"


def print_report(result: SolveResult, figure: Path | None = None,
                 files: Sequence[Path] = ()) -> None:
    """控制台报表：结论、关键数字与校验状态，每样都是一张对齐表"""
    c = result.checks
    x, y = result.best
    sc = result.scenario
    print("问题二：第二检测点的选择与候选区域")
    print_table(["输入项", "数值", "单位"],
                [["第一检测点 S1", f"({result.site[0]:.0f}, {result.site[1]:.0f})", "m"],
                 ["第一示向度 θ1", f"{result.theta1:.1f}", "°"],
                 ["示向度误差半宽", f"±{cfg.BEARING_ERROR_DEG:.0f}", "°"],
                 ["源到 S1 距离 d", f"[{c['d_lo']:.0f}, {c['d_hi']:.0f}]", "m"],
                 ["可测性要求 |S2−G|", f"≤ {cfg.RECEIVE_MIN:.0f}", "m"],
                 ["可行域（透镜）面积", f"{result.lens_area_m2 / 1e6:.3f}", "km²"],
                 ["最优点对最坏源的距离", f"{c['best_hears_worst_source_m']:.1f}", "m"]],
                align="lrr")
    print()
    print("最优第二检测点")
    print_table(["项目", "数值", "单位"],
                [["最优第二检测点 S2*", f"({x:.0f}, {y:.0f})", "m"],
                 ["S2* 距 S1", f"{result.best_r:.0f}", "m"],
                 ["S2* 相对示向度", f"{result.best_phi:+.1f}", "°"],
                 ["最坏定位直径 J*", f"{result.j_star:.1f}", "m"],
                 ["最坏交会角 γ", f"{sc['gamma_deg']:.1f}", "°"],
                 ["最坏情形源距 S1 d", f"{sc['d_m']:.0f}", "m"],
                 ["最坏情形 δ1", f"{sc['delta1_deg']:+.0f}", "°"],
                 ["最坏情形 δ2", f"{sc['delta2_deg']:+.0f}", "°"],
                 ["单次测向直径", f"{result.single_m:.0f}", "m"],
                 ["最坏直径缩小倍数", f"{result.improvement:.2f}", "—"]],
                align="lrr")
    print()
    print("候选区域（弧带）")
    print(f"  判据 J ≤ {result.band.level_m:.0f} m，是最优值的 {1.0 + c['eta']:.2f} 倍；"
          f"合计面积 {result.band.area_m2 / 1e6:.3f} km²")
    print_table(["瓣号", "径向范围 / m", "方位差范围 / °", "面积 / km²"],
                [[i, f"[{lobe['r_lo']:.0f}, {lobe['r_hi']:.0f}]",
                  f"[{lobe['phi_lo']:+.1f}, {lobe['phi_hi']:+.1f}]",
                  f"{lobe['area_m2'] / 1e6:.3f}"]
                 for i, lobe in enumerate(result.band.lobes, 1)],
                align="rrrr")
    print()
    thy = result.theory
    if thy:
        ce = thy["certify"]
        cr = thy["criteria"]
        wc = cr["worst_case_geometry"]
        print("文献判据（GDOP / CRLB / 几何稀释）与全局认证")
        print_table(["环节", "规模", "结果"],
                    [["全域认证", f"{ce['step_m']:.0f} m 细网格，{ce['n_grid']} 点",
                      f"GDOP 快筛取 {ce['n_exact']} 个候选精确复核，J = "
                      f"{ce['worst_diam_m']:.4f} m"],
                     ["与粗搜细化解对照", f"相距 {ce['vs_coarse_m']:.1f} m",
                      f"最坏直径差 {ce['vs_coarse_j_m']:+.2e} m，未漏掉更好的盆地"],
                     ["判据一致性", f"可行域内 {cr['n_probe']} 个采样点",
                      f"GDOP 与精确直径的 Spearman 秩相关 "
                      f"{cr['gdop_vs_exact_spearman']:.3f}"],
                     ["最坏情形几何", f"d = {wc['d_m']:.0f} m、r2 = {wc['r2_m']:.0f} m",
                      f"交会角 γ = {wc['gamma_deg']:.1f}°"]],
                    align="lll")
        print_table(["半径口径", "数值 / m", "相对精确构造"],
                    [["精确构造", f"{wc['exact_worst_radius_m']:.2f}", "0.00%"],
                     ["本文闭式", f"{cr['closed_form_m']['minimax_radius_m']:.2f}",
                      f"{cr['closed_form_rel_dev']['minimax_radius_m'] * 100:+.1f}%"],
                     ["文献 GDOP", f"{cr['closed_form_m']['gdop_m']:.2f}",
                      f"{cr['closed_form_rel_dev']['gdop_m'] * 100:+.1f}%"]],
                    align="lrr")
        print("  文献式偏低，只能当快筛")
        print("  准则对照（同一张表三个口径互评）：")
        labels = (("minimax", "本文 minimax"), ("gdop", "文献 GDOP"),
                  ("expected", "期望口径"))
        print_table(["准则", "坐标 / m", "r / m", "φ / °", "最坏直径 / m", "比最优差 / %",
                     "期望直径 / m", "GDOP / m"],
                    [[label, f"({thy['points'][name]['xy_m'][0]:.0f}, "
                             f"{thy['points'][name]['xy_m'][1]:.0f})",
                      f"{thy['points'][name]['r_m']:.0f}",
                      f"{thy['points'][name]['phi_deg']:+.1f}",
                      f"{thy['points'][name]['worst_diam_m']:.2f}",
                      f"{thy['points'][name]['worst_diam_loss_pct']:+.2f}",
                      f"{thy['points'][name]['mean_diam_m']:.2f}",
                      f"{thy['points'][name]['gdop_m']:.1f}"]
                     for name, label in labels],
                    align="lrrrrrrr")
    print()
    print("校验")
    rows: list[list[Any]] = [
        ["最坏情形解析构造 vs shapely",
         _mark(c["scenario_rel_dev"] < 1e-9),
         f"{c['scenario_analytic_m']:.4f} vs {c['scenario_exact_m']:.4f} m，"
         f"偏差 {c['scenario_rel_dev']:.1e}，顶点 {c['scenario_n_vertices']} 个"],
        ["离散化加密",
         _mark(c["discretisation_rel_dev"] < 1e-3),
         f"{cfg.DENSE_FACTOR}× 后 J* = {c['discretisation_dense_m']:.4f} m，"
         f"偏差 {c['discretisation_rel_dev']:.1e}"],
        ["候选弧带径向连续",
         _mark(c["band_radial_gaps"] == 0),
         f"断口 {c['band_radial_gaps']} 处"],
        ["全局认证 J",
         _mark(c["certify_j_rel_dev"] < 1e-9),
         f"{c['certify_worst_diam_m']:.4f} m，偏差 {c['certify_j_rel_dev']:.1e}"],
    ]
    if "analytic_vs_shapely" in c:
        v = c["analytic_vs_shapely"]
        rows += [
            ["四边形构造 vs shapely",
             _mark(v["impl_max_abs_rel_dev"] < 1e-9),
             f"{v['n_used']} 个有效样本，最大偏差 {v['impl_max_abs_rel_dev']:.1e}"],
            ["一阶公式（论文引用）",
             _mark(v["formula_frac_below_1pct"] > 0.95),
             f"中位偏差 {v['formula_median_rel_dev']:.3%}，"
             f"<1% 占 {v['formula_frac_below_1pct']:.0%}"],
        ]
    if "theory_closed_forms" in c:
        tv = c["theory_closed_forms"]
        rows += [
            ["文献闭式 vs 数值特征值",
             _mark(tv["closed_vs_eigen_max_rel_dev"] < 1e-9),
             f"{tv['n_used']} 个有效样本，最大偏差 {tv['closed_vs_eigen_max_rel_dev']:.1e}"],
            ["本文闭式 vs 精确最坏半径",
             _mark(tv["minimax_closed_median_rel_dev"] < 0.01),
             f"中位 {tv['minimax_closed_median_rel_dev']:.2%}；文献 GDOP 中位偏差 "
             f"{tv['gdop_vs_exact_median_rel_dev']:+.1%}"],
        ]
    print_table(["检查项", "结果", "数值"], rows, align="lcr")
    for p in list(files) + ([figure] if figure else []):
        print(f"  → {p}")


def write_outputs(save_dir: Path, result: SolveResult, figure: Path | None = None,
                  csv_name: str = cfg.MAP_CSV, json_name: str = cfg.SUMMARY_JSON) -> dict[str, Path]:
    """一次写全 CSV 与 JSON，返回路径表；出图与否由调用方决定"""
    save_dir = Path(save_dir)
    return {"csv": save_map_csv(save_dir, result, csv_name),
            "json": save_summary_json(save_dir, result, json_name)}
