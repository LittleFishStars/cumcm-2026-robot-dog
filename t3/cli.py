"""运行编排与命令行入口，本地演练与官方两种模式加参数解析"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from common.console import (LEVEL_NORMAL, LEVEL_QUIET, LEVEL_VERBOSE, is_quiet, is_verbose,
                            relax_console_encoding, set_level)
from common.paths import default_jammers_dir
from common.practice_arena import PracticeArena
from common.sim_client import API_LOG_NAME, BASE_URL, Simulator
from common.sim_client import api_log as _api_log_raw
from t3.config import (BEARING_ERROR_DEG, CHOSEN_RING_RADIUS, CLEAR_RADIUS, COVER_RADIUS,
                             INLINE_MAX_MEC_R, INLINE_NEAR_R, INLINE_R_MAX, INLINE_R_MIN,
                             K_CLEAR_MAX, RESULTS_DIR, SEED, TRAJ_DIR)
from t3.covering import (CoverSolveResult, optimal_ring_radius, print_cover_report,
                               solve_covering_circles)
from common.scanfigure import STEP_DIR_NAME, reset_dir
from t3.plotting import save_scan_figures, save_trajectory, truth_points
from t3.report import (episode_row, save_plan, save_survey, truth_check, observation_rows)
from t3.strategy import RobotDog


def _api_log(args: argparse.Namespace, echo: bool) -> "AbstractContextManager[ApiLog | None]":
    """接口日志上下文，把 CLI 参数拆成公共层 api_log 需要的参数，传空串的 --api-log 就是关闭"""
    return _api_log_raw(args.save_dir, args.api_log, echo)


def _episode_printer(clear: bool) -> "Callable[[dict, dict, int | None], None]":
    """按模式打印本局小结，只在需要时输出清除相关字段，过程明细仅 --verbose 时输出"""
    def show(stats: dict, check: dict, n_sources: int | None) -> None:
        """打印一局的巡视小结与覆盖核对结果，详细档走这里，缺省档只在外面打一行结论"""
        if not is_verbose():
            return
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
    """本地演练：自动拉起 jammers-py 跑 N 局，巡视扫描加定位加清除，最后汇总"""
    jammers_dir = (Path(args.jammers_dir) if args.jammers_dir
                   else default_jammers_dir())
    clear = not args.survey_only
    show = _episode_printer(clear)
    rows: list[dict] = []
    observations: list[dict] = []
    with _api_log(args, is_verbose()) as api_log, \
            PracticeArena(jammers_dir, robot_id=args.robot_id,
                          robot_port=args.robot_port, console_port=args.console_port,
                          reuse_existing=not args.no_reuse) as arena:
        # 演练场地址只报一次：缺省档并进"演练 N 局"那一行，详细档单独一行
        if is_verbose():
            print(f"jammers-py 已就绪：机器狗接口 {arena.robot_url}，控制台 {arena.console_url}")
        elif not is_quiet():
            print(f"演练 {args.practice} 局（jammers-py：机器狗接口 {arena.robot_url}，"
                  f"控制台 {arena.console_url}）")
        for ep in range(args.practice):
            seed = args.seed + ep
            truth = arena.start_episode(seed)
            if is_verbose():
                print(f"\n----- 演练第 {ep + 1}/{args.practice} 局（seed={seed}，"
                      f"干扰源 {len(truth)} 个）-----")
            dog = RobotDog(Simulator(robot_id=args.robot_id, base_url=arena.robot_url,
                                             timeout=args.timeout),
                           verbose=is_verbose(), logfile=args.log, episode=ep + 1,
                           clear=clear, k_clear_max=args.k_clear_max,
                           inline_r_min=args.inline_r_min,
                           inline_r_max=args.inline_r_max,
                           inline_near_r=args.inline_near_r,
                           inline_max_mec_r=args.inline_max_mec_r,
                           rotate=not args.no_rotate, api_log=api_log)
            stats = dog.run(res.plan, res.survey_order)
            arena.finish_episode()
            # 用 dog.plan / dog.survey_order_used：布局在起始扫描后按源密集方向旋转过，
            # 这一局真正的圆心位置与巡视顺序才是核对、落表、出图应依据的几何。
            check = truth_check(truth, dog.plan, dog.obs, dog.cleared, dog.tracks)
            rows.append(episode_row(ep + 1, seed, truth, dog, stats, check))
            observations.extend(observation_rows(ep + 1, dog.plan, dog.meas))
            show(stats, check, len(truth))
            if not is_verbose() and not is_quiet():        # 缺省档：每局一行结论
                print(f"第 {ep + 1}/{args.practice} 局（seed={seed}）：清除 "
                      f"{stats['cleared']}/{len(truth)}，里程 {stats['travel_m']:.0f} m，"
                      f"虚拟时间 {stats['virtual_time_s']:.0f} s，测向 {stats['n_measure']} 次")
            if not args.no_plot:                    # 出图在 /exit 之后，不占现实时间预算
                name = f"ep{ep + 1:02d}_seed{seed}"
                tp = truth_points(truth)
                # 每一局先清掉上一局的图：图形目录只保留最新一局，避免新旧图混在一起
                # 文件名带局号，肉眼很难分辨哪张属于这一轮。结果表 json/csv 不受影响。
                reset_dir(save_dir / args.traj_dir)
                reset_dir(save_dir / STEP_DIR_NAME)
                files = save_trajectory(
                    save_dir, name, dog.actions, dog.plan, dog.survey_order_used, tp,
                    title=f"第 {ep + 1} 局（seed={seed}）：清除 {stats['cleared']}/{len(truth)}、"
                          f"里程 {stats['travel_m']:.0f} m、虚拟时间 {stats['virtual_time_s']:.0f} s",
                    traj_dir=args.traj_dir)
                scans = save_scan_figures(save_dir, name, dog.scan_steps, dog.plan,
                                          dog.survey_order_used, tp)
                # 打印绝对路径：--save-dir 缺省是相对路径，换个工作目录运行就把图写到别处，
                # 只报相对路径时"图在哪"很容易看岔，报 None 张更会让人以为没出图。
                if is_verbose():
                    print("  总轨迹图：" + "，".join(str(Path(f).resolve()) for f in files))
                    print(f"  逐步扫描结果图：{len(scans)} 张 → "
                          f"{(save_dir / STEP_DIR_NAME).resolve()}/"
                          + ("（已清掉上一局的图，只保留本局）" if ep else ""))
        if not args.no_plot and not is_verbose() and not is_quiet():
            # 缺省档：图形目录只留最新一局，所以报一行"图在哪"就够了
            print(f"图只保留最新一局：{(save_dir / args.traj_dir).resolve()}/、"
                  f"{(save_dir / STEP_DIR_NAME).resolve()}/")
    if is_verbose():
        print("\n" + "=" * 78)
    if not is_quiet():
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
                  f"{sum(r['n_precise_at_survey'] for r in rows)}/"
                  f"{sum(r['n_sources'] for r in rows)} 个")
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
    if is_verbose():
        print("=" * 78)
    paths = (save_plan(res, save_dir)
             + save_survey(save_dir, rows, observations, res.to_json(),
                           {"mode": "practice", "problem_no": 3,
                            "seed0": args.seed, "episodes": args.practice,
                            "bearing_error_deg": BEARING_ERROR_DEG}))
    print("结果已保存：" + "，".join(str(p) for p in paths))
    if api_log is not None and not is_quiet():
        api_log.report()
    return 0


def run_official(args: argparse.Namespace, res: CoverSolveResult, save_dir: Path) -> int:
    """官方评测接口模式：连 127.0.0.1 上已开放接口的模拟器跑完整一局，接口调用日志即证据链"""
    sim = Simulator(robot_id=args.robot_id, base_url=args.base_url, timeout=args.timeout)
    if is_verbose():
        print(f"连接模拟器 {args.base_url}（robot_id={args.robot_id}）")
    with _api_log(args, is_verbose()) as api_log:
        dog = RobotDog(sim, verbose=is_verbose(), logfile=args.log, episode=1,
                       clear=not args.survey_only, k_clear_max=args.k_clear_max,
                       inline_r_min=args.inline_r_min,
                       inline_r_max=args.inline_r_max,
                       inline_near_r=args.inline_near_r,
                       inline_max_mec_r=args.inline_max_mec_r,
                       rotate=not args.no_rotate, api_log=api_log)
        stats = dog.run(res.plan, res.survey_order)
        if not is_quiet():
            print(f"完成：清除 {stats['cleared']} 个，巡视 {stats['waypoints_visited']} 个圆心，"
                  f"里程 {stats['travel_m']:.0f} m，虚拟时间 {stats['virtual_time_s']:.0f} s，"
                  f"测向 {stats['n_measure']} 次（补测 {stats['n_probe']} 次），"
                  f"听到 {stats['channels_heard']} 个频道（{stats['n_bearings']} 条示向度）")
        # 官方模式的场景由平台生成，不受我们的 --seed 控制，所以 seed 记为 None 以免误读
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
            if is_verbose():
                print("总轨迹图：" + "，".join(str(Path(f).resolve()) for f in files))
                print(f"逐步扫描结果图：{len(scans)} 张 → "
                      f"{(save_dir / STEP_DIR_NAME).resolve()}/")
            elif not is_quiet():
                print(f"图：{(save_dir / args.traj_dir).resolve()}/、"
                      f"{(save_dir / STEP_DIR_NAME).resolve()}/")
        paths = (save_plan(res, save_dir)
                 + save_survey(save_dir, [row], observation_rows(1, dog.plan, dog.meas),
                               res.to_json(),
                               {"mode": "official", "problem_no": 3,
                                "base_url": args.base_url, "episodes": 1,
                                "bearing_error_deg": BEARING_ERROR_DEG,
                                "note": "官方模式接口不返回真值，真值相关字段为 null"}))
        print("结果已保存：" + "，".join(str(p) for p in paths))
        if api_log is not None and not is_quiet():
            api_log.report()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器，演练、官方、仅规划三种模式的全部选项都在这里配上"""
    p = argparse.ArgumentParser(
        description="2026 CUMCM B 题问题三（第一阶段）：1000 m 覆盖圆求解 + 依次到圆心巡视扫描")
    p.add_argument("--practice", type=int, nargs="?", const=1, default=0,
                   help="本地演练局数：自动拉起 jammers-py 跑 N 局，缺省 1 局")
    p.add_argument("--base-url", default=BASE_URL,
                   help=f"官方模拟器地址，缺省 {BASE_URL}。不带任何参数就连它跑完一局；"
                        f"只想求覆盖圆方案、不连模拟器时用 --plan-only")
    p.add_argument("--plan-only", action="store_true",
                   help="只求解并保存 1000 m 覆盖圆方案，含校验与文献对照，不连模拟器")
    p.add_argument("--api-log", default=None,
                   help=f"接口调用日志路径，缺省 <save-dir>/{API_LOG_NAME}，传空串关闭。"
                        f"官方模式下这是唯一的证据链，建议保留")
    p.add_argument("--robot-id", default=None,
                   help="参赛队号，跑演练和官方模式时必须显式给出，代码里不存队号。"
                        "官方模式须与模拟器登录的队号一致，演练模式会把它传给模拟器的 --team")
    p.add_argument("--timeout", type=float, default=5.0, help="HTTP 超时 / s")
    p.add_argument("--ring-radius", type=float, default=None,
                   help=f"覆盖圆环半径 d / m，只对六边形族有效。缺省 {CHOSEN_RING_RADIUS:.0f}，"
                        f"传 {optimal_ring_radius():.3f} 则取余量最大的 d*，里程增加约 2153 m。"
                        f"指定本项就自动改用六边形族")
    p.add_argument("--hex-layout", action="store_true",
                   help="改用经典「1 中心 + 6 正六边形环心」布局。缺省是一般 7 点布局，"
                        "里程 ~6167 m，六边形族最优也只有 6737.7 m")
    p.add_argument("--layout", choices=("uniform", "optimized"), default="uniform",
                   help="一般 7 点布局的两个变体，缺省 uniform，用户 2026-09-12 指定。"
                        "uniform 把 7 个圆心均匀铺在半径 1000 m 的圆上，正七边形，零余量，"
                        "里程 ~6207 m；optimized 是数值优化的最短路径布局，余量 ~5 m，"
                        "里程 ~6167 m。仅当未用六边形族时生效")
    p.add_argument("--jammers-dir", default=None,
                   help="jammers-py 目录，缺省是本仓库根目录下的 resources/jammers-py/")
    p.add_argument("--console-port", type=int, default=8090,
                   help="演练时 jammers-py 控制台端口，缺省 8090，被占用则自动顺延")
    p.add_argument("--robot-port", type=int, default=2026,
                   help="演练时 jammers-py 的机器狗接口端口，缺省 2026，与官方一致")
    p.add_argument("--no-reuse", action="store_true",
                   help="不使用已在运行的 jammers-py，另起一个独占实例。多个会话并行演练时"
                        "用它，配合 --console-port/--robot-port 避免端口冲突")
    p.add_argument("--seed", type=int, default=SEED,
                   help="演练第 1 局的种子，场景布局与示向度噪声都由它确定，可复现")
    p.add_argument("--save-dir", default=None,
                   help=f"结果输出目录，缺省 {RESULTS_DIR}/。不按演练和官方分家，"
                        f"一个目录就是最新一次运行")
    p.add_argument("--log", default=None,
                   help="过程日志文件，记逐站扫描的文字过程；每局重写，只留最新一局")
    p.add_argument("--k-clear-max", type=int, default=K_CLEAR_MAX,
                   help=f"试清未中后最多再补清几个点：用 K 个半径 20 m 的圆覆盖定位区域。"
                        f"缺省 {K_CLEAR_MAX}，只在能盖满区域时才用，盖不满就转入补测")
    p.add_argument("--inline-r-min", type=float, default=INLINE_R_MIN,
                   help=f"站点间前向顺路清除的估计点半径下界 / m，缺省 {INLINE_R_MIN:.0f}")
    p.add_argument("--inline-r-max", type=float, default=INLINE_R_MAX,
                   help=f"站点间前向顺路清除的估计点半径上界 / m，缺省 {INLINE_R_MAX:.0f}")
    p.add_argument("--inline-near-r", type=float, default=INLINE_NEAR_R,
                   help=f"每站到站后的近距顺路清除半径 / m，缺省 {INLINE_NEAR_R:.0f}")
    p.add_argument("--inline-max-mec-r", type=float, default=INLINE_MAX_MEC_R,
                   help=f"参与顺路清除的区域最大最小覆盖圆半径 / m，缺省 {INLINE_MAX_MEC_R:.0f}。"
                        f"区域更大的频道不参与顺路，留给阶段二")
    p.add_argument("--survey-only", action="store_true",
                   help="只做阶段一：巡视扫描加覆盖核对，不做定位与清除")
    p.add_argument("--traj-dir", default=TRAJ_DIR,
                   help=f"轨迹图输出子目录，相对 --save-dir，缺省 {TRAJ_DIR}")
    p.add_argument("--no-rotate", action="store_true",
                   help="不做起始扫描后的布局旋转，保持设计基准朝向，用于对照实验")
    p.add_argument("--no-plot", action="store_true",
                   help=f"不出逐局轨迹图。缺省每局在 <save-dir>/{TRAJ_DIR}/ 生成同名 png + csv")
    p.add_argument("--quiet", action="store_true",
                   help="只输出最终结论与产物路径，比缺省更安静，供批处理用")
    p.add_argument("--verbose", action="store_true",
                   help="打印完整过程，含覆盖校验明细、每局阶段统计、接口回显等。"
                        "与 --quiet 同时给出时以它为准")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """按模式分派：--practice 走本地演练，--plan-only 只求覆盖圆，其余连官方模拟器"""
    relax_console_encoding()
    args = build_parser().parse_args(argv)
    set_level(LEVEL_VERBOSE if args.verbose else LEVEL_QUIET if args.quiet else LEVEL_NORMAL)
    official = not args.practice and not args.plan_only
    # 队号一律运行时传入，代码里不留任何队号；只有纯离线的 --plan-only 不需要
    if args.robot_id is None and (args.practice or official):
        print("错误：跑演练/官方模式必须用 --robot-id 传入参赛队号（如 --robot-id <12 位队号>）。",
              file=sys.stderr)
        return 2
    # 官方模式与演练写同一目录：一个结果目录 = 最新一次运行，不按模式分家。
    # 代价是官方跑一局就会覆盖该目录里的演练批产物，所以"先跑演练批，再验证和出图"是一个
    # 固定次序。演练批可由 --practice N --seed M 逐字节复现，重跑一次就行。
    if args.save_dir is None:
        args.save_dir = RESULTS_DIR
    save_dir = Path(args.save_dir)

    if is_verbose():
        print("=" * 78)
    if not is_quiet():
        print("2026 CUMCM B 题 · 问题三：机器狗搜索与清除干扰源（确定性策略）")
    if is_verbose():
        print("=" * 78)

    res = solve_covering_circles(args.ring_radius, use_hex=args.hex_layout,
                                 use_uniform=not args.hex_layout
                                 and args.ring_radius is None
                                 and args.layout == "uniform")
    if is_verbose():
        print_cover_report(res)
    elif not is_quiet():                        # 缺省档：覆盖方案一行结论
        plan = res.plan
        print(f"覆盖圆：{len(plan.waypoints)} 个（r = {plan.cover_radius:.0f} m），"
              f"最坏最近距离 {res.worst_distance:.3f} m（余量 {res.margin:.3f} m、"
              f"采样覆盖 {res.coverage_ratio * 100:.2f}%），巡视里程 {res.survey_length:.0f} m")
    paths = save_plan(res, save_dir)
    # 覆盖圆方案是三种模式共同的产物，所以这一行在缺省与 --quiet 下都保留，--quiet 只留路径
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
    if is_verbose():
        print("提示：不加参数就连官方模拟器；加 --practice N 跑本地演练（自动拉起 jammers-py）。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
