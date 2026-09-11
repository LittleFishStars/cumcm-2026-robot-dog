"""运行编排与命令行入口（两种模式 + 参数解析）。

* `run_practice` —— 本地演练模式：用 jammers-py 生成固定场景，能拿到真值，可核对覆盖保证；
* `run_official` —— 正式模式：连官方模拟器，真值不可见，策略完全相同。

顶层 T3.py 只调用本模块的 main()，故命令行用法与拆分前一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from cumcm.common.console import relax_console_encoding
from cumcm.common.paths import default_jammers_dir
from cumcm.common.practice_arena import PracticeArena
from cumcm.common.sim_client import API_LOG_NAME, BASE_URL, ROBOT_ID, Simulator
from cumcm.common.sim_client import api_log as _api_log_raw
from cumcm.t3.config import (BEARING_ERROR_DEG, CHOSEN_RING_RADIUS, CLEAR_RADIUS, COVER_RADIUS, INLINE_CLEAR_MEC_MAX_M, INLINE_PROBE_DIAM_M, INLINE_RADIUS_MAX_M, INLINE_SECTOR_DEG, K_CLEAR_MAX, RESULTS_DIR, SEED, TRAJ_DIR)
from cumcm.t3.covering import (CoverSolveResult, optimal_ring_radius, print_cover_report, solve_covering_circles)
from cumcm.common.scanfigure import STEP_DIR_NAME, reset_dir
from cumcm.t3.plotting import save_scan_figures, save_trajectory, truth_points
from cumcm.t3.report import (episode_row, save_plan, save_survey, truth_check, observation_rows)
from cumcm.t3.strategy import RobotDog


def _api_log(args: argparse.Namespace, echo: bool):
    """接口日志上下文（把 CLI 参数拆成公共层 api_log 所需的参数）。

    路径优先取 --api-log；未指定时用 <save-dir>/api_calls.jsonl；显式传空串则关闭日志。
    与 GA 方案（cumcm.t3ga.cli）同一套机制，便于两份结果用同样的方式审计。
    """
    return _api_log_raw(args.save_dir, args.api_log, echo)


def _episode_printer(clear: bool):
    """按模式打印本局小结（只在需要时输出清除相关字段）。"""
    def show(stats: dict, check: dict, n_sources: Optional[int]) -> None:
        print(f"本局：巡视 {stats['waypoints_visited']} 个圆心，里程 {stats['travel_m']:.0f} m，"
              f"虚拟时间 {stats['virtual_time_s']:.0f} s，测向 {stats['n_measure']} 次；"
              f"听到 {stats['channels_heard']}/{n_sources} 个源"
              f"（{stats['n_bearings']} 条示向度）")
        if clear:
            print(f"  阶段2：就近试清命中 {stats['cleared']} 个（其中补测后再清 "
                  f"{stats['n_refined']} 个，共补测 {stats['n_probe']} 次）；"
                  f"巡视后估计已够准的（覆盖圆半径 < {CLEAR_RADIUS:.0f} m，诊断）"
                  f"{stats['n_precise_at_survey']} 个；判定必无信号跳过测量 "
                  f"{stats['n_skip_measure']} 次；途中顺路清除 {stats['n_inline_cleared']} 个"
                  f"（白跑 {stats['n_inline_fail']} 次）")
            print(f"  清除：{stats['cleared']}/{n_sources}（平均 {stats['avg_time_s']:.1f} s/个），"
                  f"定位误差均值 {check['localize_err_mean_m']} m / 最大 "
                  f"{check['localize_err_max_m']} m，清除点在 20 m 内 "
                  f"{check['n_within_clear_radius']} 个")
        print(f"  覆盖核对：源到最近圆心最坏距离 {check['worst_nearest_m']:.1f} m ≤ "
              f"{COVER_RADIUS:.0f} m（{check['all_within_cover']}），"
              f"漏听 {check['missed_channels'] or '无'}")
    return show


def run_practice(args: argparse.Namespace, res: CoverSolveResult, save_dir: Path) -> int:
    """本地演练：自动拉起 jammers-py，跑 N 局（巡视扫描 + 定位 + 清除）并汇总。"""
    jammers_dir = (Path(args.jammers_dir) if args.jammers_dir
                   else default_jammers_dir())
    clear = not args.survey_only
    show = _episode_printer(clear)
    rows: List[dict] = []
    observations: List[dict] = []
    with _api_log(args, not args.quiet) as api_log, \
            PracticeArena(jammers_dir, robot_id=args.robot_id,
                          robot_port=args.robot_port, console_port=args.console_port,
                          reuse_existing=not args.no_reuse) as arena:
        print(f"jammers-py 已就绪：机器狗接口 {arena.robot_url}，控制台 {arena.console_url}")
        for ep in range(args.practice):
            seed = args.seed + ep
            truth = arena.start_episode(seed)
            print(f"\n----- 演练第 {ep + 1}/{args.practice} 局（seed={seed}，"
                  f"干扰源 {len(truth)} 个）-----")
            dog = RobotDog(Simulator(robot_id=args.robot_id, base_url=arena.robot_url,
                                             timeout=args.timeout),
                           verbose=not args.quiet, logfile=args.log, episode=ep + 1,
                           clear=clear, k_clear_max=args.k_clear_max,
                           inline_sector_deg=args.inline_sector_deg,
                           inline_radius_max=args.inline_radius_max,
                           inline_probe_diam=args.inline_probe_diam,
                           inline_clear_mec_max=args.inline_clear_mec_max,
                           rotate=not args.no_rotate, api_log=api_log)
            stats = dog.run(res.plan, res.survey_order)
            arena.finish_episode()
            # 用 dog.plan / dog.survey_order_used：布局在起始扫描后按源密集方向旋转过，
            # 这一局真正的圆心位置与巡视顺序才是核对、落表、出图应依据的几何。
            check = truth_check(truth, dog.plan, dog.obs, dog.cleared, dog.tracks)
            rows.append(episode_row(ep + 1, seed, truth, dog, stats, check))
            observations.extend(observation_rows(ep + 1, dog.plan, dog.meas))
            show(stats, check, len(truth))
            if not args.no_plot:                    # 出图在 /exit 之后，不占现实时间预算
                name = f"ep{ep + 1:02d}_seed{seed}"
                tp = truth_points(truth)
                # 每一局先清掉上一局的图：图形目录只保留最新一局，避免新旧图混在一起
                # （文件名带局号，肉眼很难分辨哪张属于这一轮）。结果表（json/csv）不受影响。
                reset_dir(save_dir / args.traj_dir)
                reset_dir(save_dir / STEP_DIR_NAME)
                files = save_trajectory(
                    save_dir, name, dog.actions, dog.plan, dog.survey_order_used, tp,
                    title=f"第 {ep + 1} 局（seed={seed}）：清除 {stats['cleared']}/{len(truth)}、"
                          f"里程 {stats['travel_m']:.0f} m、虚拟时间 {stats['virtual_time_s']:.0f} s",
                    traj_dir=args.traj_dir)
                scans = save_scan_figures(save_dir, name, dog.scan_steps, dog.plan,
                                          dog.survey_order_used, tp)
                # 打印**绝对**路径：--save-dir 缺省是相对路径，在不同工作目录下运行会把图
                # 写到别处，只报相对路径时"图在哪"很容易看岔（报 None 张更会让人以为没出图）。
                print("  总轨迹图：" + "，".join(str(Path(f).resolve()) for f in files))
                print(f"  逐步扫描结果图：{len(scans)} 张 → "
                      f"{(save_dir / STEP_DIR_NAME).resolve()}/"
                      + ("（已清掉上一局的图，只保留本局）" if ep else ""))
    print("\n" + "=" * 78)
    if clear:
        print(f"汇总（{len(rows)} 局）：平均清除比例 "
              f"{np.mean([r['clear_ratio'] for r in rows]):.4f}"
              f"（{sum(r['cleared'] for r in rows)}/{sum(r['n_sources'] for r in rows)}），"
              f"平均 {np.mean([r['avg_time_s'] for r in rows if r['avg_time_s']]):.1f} s/个，"
              f"平均虚拟时间 {np.mean([r['virtual_time_s'] for r in rows]):.0f} s，"
              f"平均测向 {np.mean([r['n_measure'] for r in rows]):.0f} 次"
              f"（其中补测 {np.mean([r['n_probe'] for r in rows]):.0f} 次；"
              f"判定必无信号而跳过 {np.mean([r['n_skip_measure'] for r in rows]):.0f} 次）")
        print(f"  定位误差：均值 "
              f"{np.mean([r['localize_err_mean_m'] for r in rows if r['localize_err_mean_m']]):.2f}"
              f" m，最差单源 "
              f"{max([r['localize_err_max_m'] for r in rows if r['localize_err_max_m']] or [0]):.2f}"
              f" m；巡视后估计已够准的源（诊断）"
              f"{sum(r['n_precise_at_survey'] for r in rows)}/{sum(r['n_sources'] for r in rows)} 个")
        print("逐局：" + "  ".join(f"seed{r['seed']}={r['cleared']}/{r['n_sources']}"
                                  for r in rows))
        print(f"  布局旋转：平均 {np.mean([r['rotation_deg'] for r in rows]):.1f}°"
              f"（起始扫描平均听到 {np.mean([r['n_face_scanned'] for r in rows]):.1f} 个源，"
              f"旋转把落脚站对准源最密集的方向；零成本）")
    else:
        print(f"汇总（{len(rows)} 局，仅巡视扫描）：平均里程 "
              f"{np.mean([r['travel_m'] for r in rows]):.0f} m，平均虚拟时间 "
              f"{np.mean([r['virtual_time_s'] for r in rows]):.0f} s，平均测向 "
              f"{np.mean([r['n_measure'] for r in rows]):.0f} 次，"
              f"共听到 {sum(r['channels_heard'] for r in rows)}/"
              f"{sum(r['n_sources'] for r in rows)} 个源")
        print("逐局：" + "  ".join(f"seed{r['seed']}={r['channels_heard']}/{r['n_sources']}"
                                  for r in rows))
    print("=" * 78)
    paths = (save_plan(res, save_dir)
             + save_survey(save_dir, rows, observations, res.to_json(),
                           {"mode": "practice", "problem_no": 3,
                            "robot_id": args.robot_id, "seed0": args.seed,
                            "episodes": args.practice,
                            "bearing_error_deg": BEARING_ERROR_DEG}))
    print("结果已保存：" + "，".join(str(p) for p in paths))
    if api_log is not None:
        api_log.report()
    return 0


def run_official(args: argparse.Namespace, res: CoverSolveResult, save_dir: Path) -> int:
    """官方评测接口模式：连 127.0.0.1 上已开放接口的模拟器，跑完整一局。

    与演练的两处差别：① **拿不到干扰源真值**，故定位误差等需要真值的指标留空（覆盖核对也
    无从做）；② 结果与演练写同一目录（<RESULTS_DIR>），不按模式分家。
    策略代码与参数完全一致，因此演练里验证过的行为在正式模式同样成立。

    接口调用全程落盘到 <save-dir>/api_calls.jsonl（--api-log 改路径、传空串关闭）：官方模式
    没有真值，这份逐次请求/响应的记录就是唯一的证据链，事后可据此复核每次测量与清除。
    """
    sim = Simulator(robot_id=args.robot_id, base_url=args.base_url, timeout=args.timeout)
    print(f"连接模拟器 {args.base_url}（robot_id={args.robot_id}）")
    with _api_log(args, not args.quiet) as api_log:
        dog = RobotDog(sim, verbose=not args.quiet, logfile=args.log, episode=1,
                       clear=not args.survey_only, k_clear_max=args.k_clear_max,
                       inline_sector_deg=args.inline_sector_deg,
                           inline_radius_max=args.inline_radius_max,
                           inline_probe_diam=args.inline_probe_diam,
                           inline_clear_mec_max=args.inline_clear_mec_max,
                       rotate=not args.no_rotate, api_log=api_log)
        stats = dog.run(res.plan, res.survey_order)
        print(f"完成：清除 {stats['cleared']} 个，巡视 {stats['waypoints_visited']} 个圆心，"
              f"里程 {stats['travel_m']:.0f} m，虚拟时间 {stats['virtual_time_s']:.0f} s，"
              f"测向 {stats['n_measure']} 次（补测 {stats['n_probe']} 次），"
              f"听到 {stats['channels_heard']} 个频道（{stats['n_bearings']} 条示向度）")
        # 官方模式的场景由平台生成，不受我们的 --seed 控制，故 seed 记为 None 以免误读
        row = episode_row(1, None, None, dog, stats,
                          truth_check(None, dog.plan, dog.obs, dog.cleared, dog.tracks))
        if not args.no_plot:
            reset_dir(save_dir / args.traj_dir)          # 只保留本次运行的图
            reset_dir(save_dir / STEP_DIR_NAME)
            files = save_trajectory(
                save_dir, "ep01", dog.actions, dog.plan,
                dog.survey_order_used, (),
                title=f"官方模式：清除 {stats['cleared']} 个、里程 {stats['travel_m']:.0f} m、"
                      f"虚拟时间 {stats['virtual_time_s']:.0f} s（无真值可比）",
                traj_dir=args.traj_dir)
            scans = save_scan_figures(save_dir, "ep01", dog.scan_steps,
                                      dog.plan, dog.survey_order_used, ())
            print("总轨迹图：" + "，".join(str(Path(f).resolve()) for f in files))
            print(f"逐步扫描结果图：{len(scans)} 张 → "
                  f"{(save_dir / STEP_DIR_NAME).resolve()}/")
        paths = (save_plan(res, save_dir)
                 + save_survey(save_dir, [row], observation_rows(1, dog.plan, dog.meas),
                               res.to_json(),
                               {"mode": "official", "problem_no": 3,
                                "base_url": args.base_url, "robot_id": args.robot_id,
                                "episodes": 1,
                                "bearing_error_deg": BEARING_ERROR_DEG,
                                "note": "官方模式接口不返回真值，故真值相关字段为 null"}))
        print("结果已保存：" + "，".join(str(p) for p in paths))
        if api_log is not None:
            api_log.report()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2026 CUMCM B 题问题三（第一阶段）：1000 m 覆盖圆求解 + 依次到圆心巡视扫描")
    p.add_argument("--practice", type=int, nargs="?", const=1, default=0,
                   help="本地演练局数：自动拉起 jammers-py 跑 N 局（缺省 1 局）")
    p.add_argument("--base-url", default=BASE_URL,
                   help=f"官方模拟器地址（缺省 {BASE_URL}）；不带任何参数即连它跑完一局。"
                        f"只想求覆盖圆方案、不连模拟器时用 --plan-only")
    p.add_argument("--plan-only", action="store_true",
                   help="只求解并保存 1000 m 覆盖圆方案（含校验与文献对照），不连模拟器")
    p.add_argument("--api-log", default=None,
                   help=f"接口调用日志路径（缺省 <save-dir>/{API_LOG_NAME}；传空串关闭）。"
                        f"官方模式下这是唯一的证据链，建议保留")
    p.add_argument("--robot-id", default=ROBOT_ID, help="参赛队号（须与模拟器一致）")
    p.add_argument("--timeout", type=float, default=5.0, help="HTTP 超时 / s")
    p.add_argument("--ring-radius", type=float, default=None,
                   help=f"（仅六边形族）覆盖圆环半径 d / m（缺省 {CHOSEN_RING_RADIUS:.0f}；"
                        f"传 {optimal_ring_radius():.3f} 取余量最大的 d*，里程增加约 2153 m）。"
                        f"指定本项即自动改用六边形族")
    p.add_argument("--hex-layout", action="store_true",
                   help="改用经典「1 中心 + 6 正六边形环心」布局（缺省是一般 7 点布局，"
                        "里程 ~6167 m 对六边形族最优的 6737.7 m）")
    p.add_argument("--jammers-dir", default=None,
                   help="jammers-py 目录（缺省为本仓库根目录下的 jammers-py/）")
    p.add_argument("--console-port", type=int, default=8090,
                   help="演练时 jammers-py 控制台端口（缺省 8090，被占用则自动顺延）")
    p.add_argument("--robot-port", type=int, default=2026,
                   help="演练时 jammers-py 的机器狗接口端口（缺省 2026，与官方一致）")
    p.add_argument("--no-reuse", action="store_true",
                   help="不使用已在运行的 jammers-py，另起一个独占实例（多个会话并行演练时用；"
                        "配合 --console-port/--robot-port 避免端口冲突）")
    p.add_argument("--seed", type=int, default=SEED,
                   help="演练第 1 局的种子（场景布局与示向度噪声都由它确定，可复现）")
    p.add_argument("--save-dir", default=None,
                   help=f"结果输出目录（缺省 {RESULTS_DIR}/，不按演练/官方分家："
                        f"一个目录 = 最新一次运行）")
    p.add_argument("--log", default=None, help="过程日志文件（逐站扫描的文字过程，每局重写，只留最新一局）")
    p.add_argument("--k-clear-max", type=int, default=K_CLEAR_MAX,
                   help=f"试清未中后最多再补清几个点（用 K 个半径 20 m 的圆覆盖定位区域；"
                        f"缺省 {K_CLEAR_MAX}，只在能盖满区域时才用，盖不满则转入补测）")
    p.add_argument("--inline-probe-diam", type=float, default=INLINE_PROBE_DIAM_M,
                   help=f"顺路清除时顺手补测的直径阈值 / m（区域直径 > 此值的已扫频道在清除点补测一次"
                        f"；缺省 {INLINE_PROBE_DIAM_M:.0f}，0 = 关闭）")
    p.add_argument("--inline-clear-mec-max", type=float, default=INLINE_CLEAR_MEC_MAX_M,
                   help=f"扇区内只清估计覆盖圆（MEC）半径 ≤ 此值的点 / m（缺省 "
                        f"{INLINE_CLEAR_MEC_MAX_M:.0f}；覆盖圆更大的点留阶段二，避免白跑）")
    p.add_argument("--inline-radius-max", type=float, default=INLINE_RADIUS_MAX_M,
                   help=f"顺路清除扇区的半径上界 / m（缺省 {INLINE_RADIUS_MAX_M:.0f} = 区域半径 = "
                        f"圆域内都算；曾收紧到两站半径以压白跑，用户确认放宽）")
    p.add_argument("--inline-sector-deg", type=float, default=INLINE_SECTOR_DEG,
                   help=f"巡视途中顺路清除的方位扇区半张角 / 度（缺省 {INLINE_SECTOR_DEG:.0f}；"
                        f"判据：估计点与圆心的连线方向落在「本站→圆心」与「下一站→圆心」"
                        f"两条连线之间即清；本参数只用于起点(原点)→第一站的兜底扇形）")
    p.add_argument("--survey-only", action="store_true",
                   help="只做阶段一（巡视扫描 + 覆盖核对），不做定位与清除")
    p.add_argument("--traj-dir", default=TRAJ_DIR,
                   help=f"轨迹图输出子目录（相对 --save-dir；缺省 {TRAJ_DIR}，"
                        f"与 T3_ga.py 的 trajectory/ 分开以免互相覆盖）")
    p.add_argument("--no-rotate", action="store_true",
                   help="不做起始扫描后的布局旋转（保持设计基准朝向，用于对照实验）")
    p.add_argument("--no-plot", action="store_true",
                   help=f"不出逐局轨迹图（缺省每局在 <save-dir>/{TRAJ_DIR}/ 生成同名 png + csv）")
    p.add_argument("--quiet", action="store_true", help="只输出汇总，不打印过程")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """按模式分派：`--practice` → 本地演练；`--plan-only` → 只求覆盖圆；其余 → 官方模拟器。

    即 `python T3.py` 不带任何参数时**直接连官方模拟器**（缺省 http://127.0.0.1:2026）跑完一局，
    与 GA 方案 T3_ga.py 的手感一致。三种模式共用同一次覆盖圆求解与同一份策略代码，差别只在
    "场景从哪来"与"结果写哪去"。
    """
    relax_console_encoding()
    args = build_parser().parse_args(argv)
    official = not args.practice and not args.plan_only
    # 官方模式与演练写同一目录：一个结果目录 = 最新一次运行，不按模式分家。
    # 代价是官方跑一局就会覆盖该目录里的演练批产物，故"先跑演练批 → 验证/出图"是一个
    # 固定次序（演练批可由 --practice N --seed M 逐字节复现，重跑一次即可）。
    if args.save_dir is None:
        args.save_dir = RESULTS_DIR
    save_dir = Path(args.save_dir)

    print("=" * 78)
    print("2026 CUMCM B 题 · 问题三：机器狗搜索与清除干扰源（确定性策略）")
    print("=" * 78)

    res = solve_covering_circles(args.ring_radius, use_hex=args.hex_layout)   # 第一步：求 1000 m 覆盖圆的位置
    if not args.quiet:
        print_cover_report(res)
    paths = save_plan(res, save_dir)
    print("覆盖圆方案已保存：" + "，".join(str(p) for p in paths))

    try:
        if args.practice:                   # 第二步：依次移动到圆心进行扫描
            return run_practice(args, res, save_dir)
        if official:
            return run_official(args, res, save_dir)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except OSError as exc:
        sys.stdout.flush()          # 先冲掉缓存的正常输出，错误信息才会出现在末尾而非表头前
        print(f"连接模拟器失败：{exc}", file=sys.stderr)
        print(f"请确认模拟器已启动并处于测试窗口内（缺省地址 {BASE_URL}）。", file=sys.stderr)
        if official:
            print("本地验证策略可用 --practice N（自动拉起 jammers-py，无需官方模拟器）。",
                  file=sys.stderr)
        return 1
    print("提示：不加参数即连官方模拟器；加 --practice N 跑本地演练（自动拉起 jammers-py）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
