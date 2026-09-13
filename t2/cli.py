"""问题二的命令行入口：一条命令给出最优第二检测点、候选区域与适合度图。

    python -m t2                                    # 缺省：S1 = 原点、θ1 = 0°
    python -m t2 --site 200 -300 --bearing 45       # 换个第一检测点与示向度
    python -m t2 --eta 0.05                         # 候选区域收紧到 J ≤ 1.05 J*
    python -m t2 --d-lo 800 --d-hi 1200             # 先前若还测到了大致距离，可收窄源不确定集
    python -m t2 --no-plot                          # 只算数与文件，没装 matplotlib 也能跑

入口就是本模块，`python -m t2` 和 `python -m t2.cli` 都行。输出分三档，档位定义在
common.console：缺省每步 1~2 行关键结论；`--verbose` 还原完整过程，含 `=` 分隔线、候选区域
逐瓣明细、判据对照表与全部校验；`--quiet` 只留最终产物路径。三档只是打印多少的区别，算出来
的数和写出去的文件完全一样，跑完一律返回 0。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from common.console import (LEVEL_NORMAL, LEVEL_QUIET, LEVEL_VERBOSE, is_quiet, is_verbose,
                            relax_console_encoding, set_level)
from t2 import config as cfg
from t2.report import print_report, save_summary_json, write_outputs
from t2.score import solve

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器，问题二的选项都在这儿，缺省值取自 `t2.config`

    位置参数一个都没有，全部靠开关控制。--site、--bearing、--eta、--d-lo、--d-hi、--verify、
    --save-dir、--no-plot、--no-csv、--quiet、--verbose 都在这里注册，别处不再加
    """
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
    """问题二的命令行主流程：解析参数 → 求解最优第二检测点 → 写结果文件与图

    缺省输出只报关键结论，标题、S2*、J* 与改善倍数、判据/认证、产物路径，一共 4~6 行。
    `--verbose` 把 `print_report` 的逐行明细全放出来，`--quiet` 只留产物路径那一行。

    Args:
        argv: 命令行参数列表；缺省取 sys.argv[1:]

    Returns:
        int: 进程退出码，正常结束恒为 0
    """
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
        # --quiet：只留一行产物路径，批处理直接取用。六个产物一次给全
        print("结果已保存：" + "，".join(str(p) for p in _artifact_paths(args.save_dir, files, figure)))
    else:
        _print_conclusion(result, _artifact_paths(args.save_dir, files, figure))
    return 0


def _artifact_paths(save_dir: Path, files: dict[str, Path],
                    figure: Path | None) -> list[Path]:
    """本次运行写出的全部产物路径，缺省档与 --quiet 共用那一行

    Args:
        save_dir: 结果目录
        files: `write_outputs` / `save_summary_json` 返回的路径表；--no-csv 时只有 json
        figure: 适合度图路径。--no-plot 时为 None，这时不再列四张图

    Returns:
        list[Path]: 按 CSV → JSON → 两张图的 png/pdf 排好并去重后的路径
    """
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
    """缺省档结论：最优第二检测点、判据/认证与产物路径

    每步 1~2 行，全文 4~6 行。只拿已经算好的值拼字符串，不为这几行重算任何量。过程明细，
    也就是逐瓣候选区域、判据对照表、全部校验，交给 `--verbose` 的 `print_report`。

    Args:
        result: 问题二的求解结果
        paths: 本次运行写出的产物路径，取 `_artifact_paths` 去重后的结果
    """
    x, y = result.best
    ce = result.theory["certify"]
    cr = result.theory["criteria"]
    # 候选区域关于示向度方向镜像对称，通常两瓣，所以只取正方位那一瓣报"方位差范围"
    lobe = result.band.lobes[0] if result.band.lobes else None
    print("问题二：第二检测点的选择与候选区域")
    print(f"最优第二检测点 S2* = ({x:.0f}, {y:.0f}) m：距 S1 {result.best_r:.0f} m、"
          f"相对示向度 {result.best_phi:+.1f}°")
    print(f"最坏定位直径 J* = {result.j_star:.1f} m（最坏交会角 "
          f"{result.scenario['gamma_deg']:.1f}°），比只做一次测向（{result.single_m:.0f} m）"
          f"缩小 {result.improvement:.1f} 倍")
    print(f"候选区域（J ≤ {result.band.level_m:.0f} m）："
          + (f"r ∈ [{result.band.r_lo:.0f}, {result.band.r_hi:.0f}] m、方位差 ∈ "
             f"[{lobe['phi_lo']:+.0f}°, {lobe['phi_hi']:+.0f}°]、"
             if lobe else "")
          + f"合计面积 {result.band.area_m2 / 1e6:.3f} km²")
    print(f"认证与判据：{ce['step_m']:.0f} m 细网格 {ce['n_grid']} 点快筛 + "
          f"{ce['n_exact']} 点精确复核 → 认证 J = {ce['worst_diam_m']:.4f} m"
          f"（与粗搜细化解相距 {ce['vs_coarse_m']:.1f} m）；文献 GDOP 判据与精确直径的 "
          f"Spearman 秩相关 {cr['gdop_vs_exact_spearman']:.3f}")
    print("结果已保存：" + "，".join(str(p) for p in paths))


if __name__ == '__main__':
    raise SystemExit(main())
