"""问题二的命令行入口：一条命令给出最优第二检测点、候选区域与适合度图"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from common.console import (LEVEL_NORMAL, LEVEL_QUIET, LEVEL_VERBOSE, is_quiet, is_verbose,
                            print_paths, print_table, relax_console_encoding, set_level)
from t2 import config as cfg
from t2.report import print_report, save_summary_json, write_outputs
from t2.score import solve

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器，问题二的选项全在这儿，缺省值取自 `t2.config`"""
    p = argparse.ArgumentParser(
        prog="python -m t2",
        description="CUMCM 2026 B 问题二：第二个检测点的选择与候选区域（最坏情况最小化）")
    p.add_argument("--site", nargs=2, type=float, metavar=("X", "Y"),
                   default=list(cfg.DEFAULT_SITE),
                   help="第一检测点坐标 / m（缺省原点，即机器狗起点）")
    p.add_argument("--bearing", type=float, default=cfg.DEFAULT_BEARING,
                   help=f"第一检测点测得的示向度 / 度（缺省 {cfg.DEFAULT_BEARING:.0f}，正东）")
    p.add_argument("--eta", type=float, default=cfg.ETA,
                   help=f"候选区域阈值：J ≤ (1+η)·J*（缺省 {cfg.ETA:.2f}）")
    p.add_argument("--d-lo", type=float, default=cfg.D_LO,
                   help=f"源到 S1 的距离下界 / m（缺省 {cfg.D_LO:.0f}：附录 2(9) 近距测不到示向度）")
    p.add_argument("--d-hi", type=float, default=cfg.D_HI,
                   help=f"源到 S1 的距离上界 / m（缺省 {cfg.D_HI:.0f}：接收半径上限）")
    p.add_argument("--verify", type=int, default=400,
                   help="解析构造 vs shapely 的抽样校核数（0 = 跳过，缺省 400）")
    p.add_argument("--save-dir", type=Path, default=Path(cfg.RESULTS_DIR),
                   help=f"结果目录（缺省 {cfg.RESULTS_DIR}）")
    p.add_argument("--no-plot", action="store_true", help="不出图（只要数值与 CSV/JSON）")
    p.add_argument("--no-csv", action="store_true", help="不写 CSV（全域网格表较大）")
    p.add_argument("--quiet", action="store_true",
                   help="只输出最终结论与产物路径，比缺省还安静，给批处理用")
    p.add_argument("--verbose", action="store_true",
                   help="打印完整过程：候选区域逐瓣明细、文献判据对照表、全部校验明细与逐条产物"
                        "路径。与 --quiet 同时给出时以它为准")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """命令行主流程：解析参数、求解、写结果与图，返回进程退出码"""
    relax_console_encoding()
    args = build_parser().parse_args(argv)
    set_level(LEVEL_VERBOSE if args.verbose else LEVEL_QUIET if args.quiet else LEVEL_NORMAL)
    result = solve(site=tuple(args.site), theta1=args.bearing, eta=args.eta, d_lo=args.d_lo,
                   d_hi=args.d_hi, verify_n=args.verify)

    files: dict[str, Path] = {}
    if args.no_csv:
        files["json"] = save_summary_json(args.save_dir, result)
    else:
        files = write_outputs(args.save_dir, result)

    figure: Path | None = None
    if not args.no_plot:
        from t2.plotting import draw_criteria, draw_suitability

        pdf_path = args.save_dir / cfg.FIGURE_PDF
        figure = draw_suitability(args.save_dir / cfg.FIGURE_PNG, result, pdf_path=pdf_path)
        draw_criteria(args.save_dir / cfg.CRITERIA_PNG, result,
                      pdf_path=args.save_dir / cfg.CRITERIA_PDF)
    if is_verbose():
        shown = [files[k] for k in ("csv", "json") if k in files]
        if figure is not None:
            shown += [figure, args.save_dir / cfg.FIGURE_PDF,
                      args.save_dir / cfg.CRITERIA_PNG, args.save_dir / cfg.CRITERIA_PDF]
        print_report(result, figure=figure, files=shown)
    elif is_quiet():
        # --quiet：只留产物路径，一行一个，批处理直接取用。六个产物一次给全
        print_paths(_artifact_paths(args.save_dir, files, figure))
    else:
        _print_conclusion(result, _artifact_paths(args.save_dir, files, figure))
    return 0


def _artifact_paths(save_dir: Path, files: dict[str, Path],
                    figure: Path | None) -> list[Path]:
    """本次运行写出的全部产物路径，缺省档与 --quiet 共用"""
    paths = [files[k] for k in ("csv", "json") if k in files]
    if figure is not None:
        paths += [figure, save_dir / cfg.FIGURE_PDF,
                  save_dir / cfg.CRITERIA_PNG, save_dir / cfg.CRITERIA_PDF]
    out: list[Path] = []
    for p in paths:                                 # `figure` 常与 FIGURE_PNG 同一路径，去重
        if p not in out:
            out.append(p)
    return out


def _print_conclusion(result: "SolveResult", paths: Sequence[Path]) -> None:
    """缺省档结论：一张关键结论表，末尾跟上逐条产物路径"""
    x, y = result.best
    ce = result.theory["certify"]
    cr = result.theory["criteria"]
    rows = [
        ["最优第二检测点 S2* / m", f"({x:.0f}, {y:.0f})"],
        ["S2* 距 S1 / m", f"{result.best_r:.0f}"],
        ["S2* 相对示向度 / °", f"{result.best_phi:+.1f}"],
        ["最坏定位直径 J* / m", f"{result.j_star:.1f}"],
        ["最坏交会角 γ / °", f"{result.scenario['gamma_deg']:.1f}"],
        ["单次测向时的直径 / m", f"{result.single_m:.0f}"],
        ["最坏直径缩小倍数", f"{result.improvement:.1f}"],
        ["候选区域判据 J / m", f"≤ {result.band.level_m:.0f}"],
        ["候选区域径向范围 / m", f"[{result.band.r_lo:.0f}, {result.band.r_hi:.0f}]"],
        ["候选区域方位差范围 / °",
         "；".join(f"[{l['phi_lo']:+.0f}, {l['phi_hi']:+.0f}]" for l in result.band.lobes) or "—"],
        ["候选区域面积 / km²", f"{result.band.area_m2 / 1e6:.3f}"],
        ["认证网格点数", str(ce["n_grid"])],
        ["精确复核点数", str(ce["n_exact"])],
        ["认证最坏直径 / m", f"{ce['worst_diam_m']:.4f}"],
        ["GDOP 与精确直径秩相关", f"{cr['gdop_vs_exact_spearman']:.3f}"],
    ]
    print_table(["关键结论", "数值"], rows, align="lr")
    print("结果已保存：")
    print_paths(paths)


if __name__ == '__main__':
    raise SystemExit(main())
