"""问题二的命令行入口：一条命令给出最优第二检测点、候选区域与适合度图。

    python T2.py                                    # 缺省：S1 = 原点、θ1 = 0°
    python T2.py --site 200 -300 --bearing 45       # 换个第一检测点与示向度
    python T2.py --eta 0.05                         # 候选区域收紧到 J ≤ 1.05 J*
    python T2.py --d-lo 800 --d-hi 1200             # 若先前还测到了大致距离，可收窄源不确定集
    python T2.py --no-plot                          # 只算数与文件（无 matplotlib 时也能跑）

顶层 `T2.py` 只是薄壳，真正实现在 `t2/`。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from t2 import config as cfg
from t2.report import print_report, save_summary_json, write_outputs
from t2.score import solve

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（T2.py 的全部选项，缺省值取自 `t2.config`）

    Returns:
        argparse.ArgumentParser: 已注册 --site/--bearing/--eta/--d-lo/--d-hi/--verify/
        --save-dir/--no-plot/--no-csv/--quiet 的解析器
    """
    p = argparse.ArgumentParser(
        prog="T2.py",
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
    p.add_argument("--quiet", action="store_true", help="只打印结论，不打印校验明细")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """问题二的命令行主流程：解析参数 → 求解最优第二检测点 → 写结果文件与图

    Args:
        argv: 命令行参数列表；缺省取 sys.argv[1:]

    Returns:
        int: 进程退出码（正常结束恒为 0）
    """
    args = build_parser().parse_args(argv)
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
    if args.quiet:
        print(result.summary())
        print(f"结果：{files['json']}" + (f"；图：{figure}" if figure else ""))
    else:
        shown = [files[k] for k in ("csv", "json") if k in files]
        if figure is not None:
            shown += [figure, args.save_dir / cfg.FIGURE_PDF,
                      args.save_dir / cfg.CRITERIA_PNG, args.save_dir / cfg.CRITERIA_PDF]
        print_report(result, figure=figure, files=shown)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
