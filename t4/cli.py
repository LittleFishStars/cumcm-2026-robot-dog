"""运行编排与命令行入口（问题四）：两种模式加参数解析

* `run_practice`：本地演练。jammers-py 生成问题四场景，定向源与全向源混合，能拿到真值；
* `run_official`：正式模式。连官方模拟器，真值不可见，策略与演练完全一样。

命令行入口就是本模块，`python -m t4`，用法跟 `python -m t3` 保持一致。--practice 跑演练，
--plan-only 只求方案，不带参数就连官方模拟器。三档输出见 common/console：缺省每步 1~2 行关键
结论，--verbose 还原完整过程，含 `=` 分隔线、每局阶段统计明细与图路径，--quiet 只留最终产物路径。
"""

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
from common.scanfigure import STEP_DIR_NAME, reset_dir
from common.sim_client import API_LOG_NAME, BASE_URL, Simulator
from common.sim_client import api_log as _api_log_raw
from t4.config import (BEARING_ERROR_DEG, CLEAR_RADIUS, INLINE_MAX_MEC_R, INLINE_NEAR_R,
                             INLINE_R_MAX, INLINE_R_MAX_IN, INLINE_R_MIN, INLINE_R_MIN_IN,
                             INLINE_R_MIN_OUT, K_CLEAR_MAX, PROBLEM_NO,
                             RESULTS_DIR, SEED, TRAJ_DIR)
from t4.plotting import save_scan_figures, save_trajectory, truth_points
from t4.report import (episode_row, observation_rows, save_plan, save_survey,
                             truth_check)
from t4.strategy import RobotDog
from t4.sweep import (SweepPlan, build_sweep_plan, plan_from_points,
                            print_sweep_report, verify_hearing_stats)


def _api_log(args: argparse.Namespace, echo: bool) -> "AbstractContextManager[ApiLog | None]":
    """接口日志上下文，便于用同一方式审计

    Args:
        args: 已解析的命令行参数
        echo: 是否把每次接口调用回显到终端

    Returns:
        AbstractContextManager[ApiLog | None]: 上下文管理器。进入时拿到日志对象，
            日志被关掉时也就是 --api-log 传了空串，则是 None
    """
    return _api_log_raw(args.save_dir, args.api_log, echo)


def _episode_printer(clear: bool) -> "Callable[[dict, dict, int | None], None]":
    """按模式打印本局小结，过程明细只有 --verbose 才输出

    Args:
        clear: 是否做了阶段二，也就是定位与清除；False 时只报扫描相关字段

    Returns:
        Callable[[dict, dict, int | None], None]: 打印函数 show(stats, check, n_sources)
    """
    def show(stats: dict, check: dict, n_sources: int | None) -> None:
        """打印一局的扫描小结与命中核对结果，走详细档；缺省档只在外面打一行结论

        Args:
            stats: 本局统计，里程、虚拟时间、测向次数、清除个数、首次听到步数都在里面
            check: 真值核对结果，定位误差、命中情况等；官方模式没有真值，为 None
            n_sources: 本局干扰源个数；官方模式没有真值，为 None
        """
        if not is_verbose():
            return
        fh = [v for v in stats["first_heard"].values()]
        worst = max(fh) if fh else 0
        print(f"本局：扫描 {stats['travel_m']:.0f} m + 收尾，虚拟时间 {stats['virtual_time_s']:.0f} s，"
              f"测向 {stats['n_measure']} 次；扫描结束听到 {stats['channels_heard']}/{n_sources} "
              f"个源，全部在扫描第 {worst} 步内听到，共 {stats['n_bearings']} 条示向度，"
              f"顺路补测 {stats['n_side_scan']} 次，途中顺路清除命中 {stats['n_inline']} 个")
        if clear:
            print(f"  清除：{stats['cleared']}/{n_sources}，平均 {stats['avg_time_s']:.1f} s/个；"
                  f"定位误差均值 {check['localize_err_mean_m']} m / 最大 "
                  f"{check['localize_err_max_m']} m；方法分布 {stats['methods']}")
        print(f"  命中核对：{check['n_heard']}/{check['n_sources']} 个源被听到，"
              f"全部听到为 {check['all_heard']}；定位误差在清除半径内 "
              f"{check['n_within_clear_radius']} 个")
    return show


def run_practice(args: argparse.Namespace, plan: SweepPlan, verify: dict, save_dir: Path) -> int:
    """本地演练：自动拉起 jammers-py，问题四场景，跑 N 局并汇总"""
    jammers_dir = (Path(args.jammers_dir) if args.jammers_dir else default_jammers_dir())
    clear = not args.survey_only
    show = _episode_printer(clear)
    rows: list[dict] = []
    observations: list[dict] = []
    with _api_log(args, is_verbose()) as api_log, \
            PracticeArena(jammers_dir, robot_id=args.robot_id,
                          robot_port=args.robot_port, console_port=args.console_port,
                          reuse_existing=not args.no_reuse,
                          problem_no=PROBLEM_NO) as arena:
        # 演练场地址只报一次：缺省档并进"演练 N 局"那一行，详细档单独占一行
        if is_verbose():
            print(f"jammers-py 已就绪：机器狗接口 {arena.robot_url}，控制台 {arena.console_url}，"
                  f"问题四场景，定向 + 全向混合")
        elif not is_quiet():
            print(f"演练 {args.practice} 局，演练场：机器狗接口 {arena.robot_url}，"
                  f"控制台 {arena.console_url}；问题四场景，定向 + 全向混合")
        for ep in range(args.practice):
            seed = args.seed + ep
            truth = arena.start_episode(seed)
            if is_verbose():
                print(f"\n----- 演练第 {ep + 1}/{args.practice} 局，seed={seed}，"
                      f"干扰源 {len(truth)} 个 -----")
            dog = RobotDog(Simulator(robot_id=args.robot_id, base_url=arena.robot_url,
                                     timeout=args.timeout),
                           verbose=is_verbose(), logfile=args.log, episode=ep + 1,
                           clear=clear, k_clear_max=args.k_clear_max,
                           inline_r_min=args.inline_r_min,
                           inline_r_max=args.inline_r_max,
                           inline_r_min_in=args.inline_r_min_in,
                           inline_r_max_in=args.inline_r_max_in,
                           inline_r_min_out=args.inline_r_min_out,
                           inline_near_r=args.inline_near_r,
                           inline_max_mec_r=args.inline_max_mec_r,
                           api_log=api_log)
            stats = dog.run(plan)
            arena.finish_episode()
            check = truth_check(truth, plan, dog.obs, dog.cleared, dog.tracks,
                                dog.first_heard)
            rows.append(episode_row(ep + 1, seed, truth, dog, stats, check))
            observations.extend(observation_rows(ep + 1, plan, dog.meas))
            show(stats, check, len(truth))
            if not is_verbose() and not is_quiet():        # 缺省档：每局一行结论
                print(f"第 {ep + 1}/{args.practice} 局，seed={seed}：清除 "
                      f"{stats['cleared']}/{len(truth)}，里程 {stats['travel_m']:.0f} m，"
                      f"虚拟时间 {stats['virtual_time_s']:.0f} s，测向 {stats['n_measure']} 次")
            if not args.no_plot:
                name = f"ep{ep + 1:02d}_seed{seed}"
                tp = truth_points(truth)
                reset_dir(save_dir / args.traj_dir)
                files = save_trajectory(
                    save_dir, name, dog.actions, plan, (), tp,
                    title=f"第 {ep + 1} 局（seed={seed}）：清除 {stats['cleared']}/{len(truth)}、"
                          f"里程 {stats['travel_m']:.0f} m、虚拟时间 "
                          f"{stats['virtual_time_s']:.0f} s",
                    traj_dir=args.traj_dir)
                if ep == args.practice - 1:      # 逐步扫描图只保留最新一局，37 步 × N 局太慢
                    reset_dir(save_dir / STEP_DIR_NAME)
                    scans = save_scan_figures(save_dir, name, dog.scan_steps, plan, (), tp)
                else:
                    scans = []
                # 图路径属过程明细：缺省档只说图在哪，详细档才逐局打印
                if is_verbose():
                    print("  总轨迹图：" + "，".join(str(Path(f).resolve()) for f in files))
                    if scans:
                        print(f"  逐步扫描结果图：{len(scans)} 张 → "
                              f"{(save_dir / STEP_DIR_NAME).resolve()}/")
    if is_verbose():
        print("\n" + "=" * 78)
    if clear:
        if is_verbose():
            print(f"汇总 {len(rows)} 局：平均清除比例 "
                  f"{np.mean([r['clear_ratio'] for r in rows]):.4f}，"
                  f"{sum(r['cleared'] for r in rows)}/{sum(r['n_sources'] for r in rows)} 个源被清，"
                  f"平均虚拟时间 {np.mean([r['virtual_time_s'] for r in rows]):.0f} s，"
                  f"平均里程 {np.mean([r['travel_m'] for r in rows]):.0f} m，"
                  f"平均测向 {np.mean([r['n_measure'] for r in rows]):.0f} 次；其中文献补测 "
                  f"{np.mean([r['n_probe'] for r in rows]):.0f} 次、顺路补测 "
                  f"{np.mean([r['n_side_scan'] for r in rows]):.0f} 次、途中顺路清除命中 "
                  f"{np.mean([r['n_inline'] for r in rows]):.1f} 个、"
                  f"判定必无信号跳过 {np.mean([r['n_skip_measure'] for r in rows]):.0f} 次")
        worst_err = max([r['localize_err_max_m'] for r in rows if r['localize_err_max_m']] or [0])
        worst_step = max([r['worst_first_heard_step'] for r in rows
                          if r['worst_first_heard_step'] is not None] or [0])
        if is_verbose():
            print(f"  定位误差：均值 "
                  f"{np.mean([r['localize_err_mean_m'] for r in rows if r['localize_err_mean_m']]):.2f}"
                  f" m，最差单源 {worst_err:.2f} m；最晚首次听到发生在第 "
                  f"{worst_step} 步，共 {plan.n_points} 个测量位置")
            print("逐局：" + "  ".join(f"seed{r['seed']}={r['cleared']}/{r['n_sources']}"
                                      f"({r['virtual_time_s']:.0f}s)" for r in rows))
        elif not is_quiet():                        # 缺省档：汇总 2~3 行
            print(f"汇总 {len(rows)} 局：清除比例 "
                  f"{np.mean([r['clear_ratio'] for r in rows]):.4f}，"
                  f"{sum(r['cleared'] for r in rows)}/{sum(r['n_sources'] for r in rows)} 个源被清，"
                  f"平均里程 {np.mean([r['travel_m'] for r in rows]):.0f} m，"
                  f"平均虚拟时间 {np.mean([r['virtual_time_s'] for r in rows]):.0f} s，"
                  f"平均测向 {np.mean([r['n_measure'] for r in rows]):.0f} 次")
            print(f"  定位误差：均值 "
                  f"{np.mean([r['localize_err_mean_m'] for r in rows if r['localize_err_mean_m']]):.2f}"
                  f" m，最差单源 {worst_err:.2f} m")
            print("逐局：" + "  ".join(f"seed{r['seed']}={r['cleared']}/{r['n_sources']}"
                                      for r in rows))
    else:
        if is_verbose():
            print(f"汇总 {len(rows)} 局，仅扫描：平均里程 "
                  f"{np.mean([r['travel_m'] for r in rows]):.0f} m，"
                  f"平均虚拟时间 {np.mean([r['virtual_time_s'] for r in rows]):.0f} s，"
                  f"平均测向 {np.mean([r['n_measure'] for r in rows]):.0f} 次，"
                  f"共听到 {sum(r['heard'] for r in rows)}/{sum(r['n_sources'] for r in rows)} 个源")
            print("逐局：" + "  ".join(f"seed{r['seed']}={r['heard']}/{r['n_sources']}"
                                      for r in rows))
        elif not is_quiet():                        # 缺省档：汇总 2 行
            print(f"汇总 {len(rows)} 局，仅扫描：平均里程 "
                  f"{np.mean([r['travel_m'] for r in rows]):.0f} m，"
                  f"平均虚拟时间 {np.mean([r['virtual_time_s'] for r in rows]):.0f} s，"
                  f"平均测向 {np.mean([r['n_measure'] for r in rows]):.0f} 次，"
                  f"共听到 {sum(r['heard'] for r in rows)}/{sum(r['n_sources'] for r in rows)} 个源")
            print("逐局：" + "  ".join(f"seed{r['seed']}={r['heard']}/{r['n_sources']}"
                                      for r in rows))
    if is_verbose():
        print("=" * 78)
    paths = (save_plan(plan, save_dir, verify)
             + save_survey(save_dir, rows, observations, plan.to_json(),
                           {"mode": "practice", "problem_no": PROBLEM_NO,
                            "seed0": args.seed, "episodes": args.practice,
                            "bearing_error_deg": BEARING_ERROR_DEG}))
    print("结果已保存：" + "，".join(str(p) for p in paths))
    if api_log is not None and not is_quiet():
        api_log.report()
    return 0


def run_official(args: argparse.Namespace, plan: SweepPlan, verify: dict, save_dir: Path) -> int:
    """官方评测接口模式：连 127.0.0.1 上已开放接口的模拟器，跑完整一局

    拿不到真值，真值相关字段一律留空。扫描布局在 t4.sweep 里实测听到率 99.9891%，加上演练里
    验证过的行为，在正式模式下同样成立。接口调用全程落盘 api_calls.jsonl，这是官方模式唯一
    的证据链。
    """
    sim = Simulator(robot_id=args.robot_id, base_url=args.base_url, timeout=args.timeout)
    if is_verbose():
        print(f"连接模拟器 {args.base_url}，robot_id={args.robot_id}，问题四")
    with _api_log(args, is_verbose()) as api_log:
        dog = RobotDog(sim, verbose=is_verbose(), logfile=args.log, episode=1,
                       clear=not args.survey_only, k_clear_max=args.k_clear_max,
                       inline_r_min=args.inline_r_min,
                       inline_r_max=args.inline_r_max,
                       inline_r_min_in=args.inline_r_min_in,
                       inline_r_max_in=args.inline_r_max_in,
                       inline_r_min_out=args.inline_r_min_out,
                       inline_near_r=args.inline_near_r,
                       inline_max_mec_r=args.inline_max_mec_r,
                       api_log=api_log)
        stats = dog.run(plan)
        # 完成一行是关键结论，缺省与 --quiet 之外都保留
        if not is_quiet():
            print(f"完成：清除 {stats['cleared']} 个，里程 {stats['travel_m']:.0f} m，"
                  f"虚拟时间 {stats['virtual_time_s']:.0f} s，测向 {stats['n_measure']} 次，"
                  f"含补测 {stats['n_probe']} 次；扫描结束听到 {stats['channels_heard']} 个频道，"
                  f"{stats['n_bearings']} 条示向度")
        row = episode_row(1, None, None, dog, stats,
                          truth_check(None, plan, dog.obs, dog.cleared, dog.tracks,
                                      dog.first_heard))
        if not args.no_plot:
            reset_dir(save_dir / args.traj_dir)
            files = save_trajectory(
                save_dir, "ep01", dog.actions, plan, (), (),
                title=f"官方模式（问题四）：清除 {stats['cleared']} 个、里程 "
                      f"{stats['travel_m']:.0f} m、虚拟时间 {stats['virtual_time_s']:.0f} s"
                      f"（无真值可比）",
                traj_dir=args.traj_dir)
            reset_dir(save_dir / STEP_DIR_NAME)
            scans = save_scan_figures(save_dir, "ep01", dog.scan_steps, plan, (), ())
            if is_verbose():
                print("总轨迹图：" + "，".join(str(Path(f).resolve()) for f in files))
                print(f"逐步扫描结果图：{len(scans)} 张 → "
                      f"{(save_dir / STEP_DIR_NAME).resolve()}/")
            elif not is_quiet():
                print(f"图：{(save_dir / args.traj_dir).resolve()}/、"
                      f"{(save_dir / STEP_DIR_NAME).resolve()}/")
        paths = (save_plan(plan, save_dir, verify)
                 + save_survey(save_dir, [row], observation_rows(1, plan, dog.meas),
                               plan.to_json(),
                               {"mode": "official", "problem_no": PROBLEM_NO,
                                "base_url": args.base_url, "episodes": 1,
                                "bearing_error_deg": BEARING_ERROR_DEG,
                                "note": "官方模式接口不返回真值，真值相关字段一律为 null"}))
        print("结果已保存：" + "，".join(str(p) for p in paths))
        if api_log is not None and not is_quiet():
            api_log.report()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器

    Returns:
        argparse.ArgumentParser: 演练 / 官方 / 仅规划三种模式的选项都配齐了的解析器
    """
    p = argparse.ArgumentParser(
        description="2026 CUMCM B 题问题四：定向 + 全向混合干扰源的搜索与清除，扫描加定位清除")
    p.add_argument("--practice", type=int, nargs="?", const=1, default=0,
                   help="本地演练局数：自动拉起 jammers-py 跑 N 局，缺省 1 局")
    p.add_argument("--base-url", default=BASE_URL,
                   help=f"官方模拟器地址，缺省 {BASE_URL}；不带任何参数就直接连它跑完一局。"
                        f"只想看扫描方案、不连模拟器时用 --plan-only")
    p.add_argument("--plan-only", action="store_true",
                   help="只求解并保存扫描方案，含听到率统计，不连模拟器")
    p.add_argument("--api-log", default=None,
                   help=f"接口调用日志路径，缺省 <save-dir>/{API_LOG_NAME}；传空串关闭")
    p.add_argument("--robot-id", default=None,
                   help="参赛队号，跑演练/官方模式时必须显式给出，代码里不存队号；"
                        "官方模式须与模拟器登录的队号一致，演练模式会把它传给模拟器的 --team")
    p.add_argument("--timeout", type=float, default=5.0, help="HTTP 超时 / s")
    p.add_argument("--jammers-dir", default=None,
                   help="jammers-py 目录，缺省为本仓库根目录下的 resources/jammers-py/")
    p.add_argument("--console-port", type=int, default=8090,
                   help="演练时 jammers-py 控制台端口，缺省 8090，被占用则自动顺延")
    p.add_argument("--robot-port", type=int, default=2026,
                   help="演练时 jammers-py 的机器狗接口端口，缺省 2026，与官方一致")
    p.add_argument("--no-reuse", action="store_true",
                   help="不使用已在运行的 jammers-py，另起独占实例，并行演练时用")
    p.add_argument("--seed", type=int, default=SEED,
                   help="演练第 1 局的种子；场景布局与示向度噪声都由它确定，可复现")
    p.add_argument("--save-dir", default=None,
                   help=f"结果输出目录，缺省 {RESULTS_DIR}/；演练与官方不分家，共用一个目录")
    p.add_argument("--log", default=None,
                   help="过程日志文件，逐点扫描的文字过程，每局重写")
    p.add_argument("--k-clear-max", type=int, default=K_CLEAR_MAX,
                   help=f"试清未中后最多再补清几个点（缺省 {K_CLEAR_MAX}）")
    p.add_argument("--inline-r-min", type=float, default=INLINE_R_MIN,
                   help=f"站点间前向顺路清除的估计点半径下界 / m，缺省 {INLINE_R_MIN:.0f}；"
                        f"未单独指定内外圈时先作用于两组")
    p.add_argument("--inline-r-max", type=float, default=INLINE_R_MAX,
                   help=f"站点间前向顺路清除的估计点半径上界 / m，缺省 {INLINE_R_MAX:.0f}）")
    p.add_argument("--inline-r-min-in", type=float, default=INLINE_R_MIN_IN,
                   help=f"内圈站，距原点 ≤ INLINE_OUTER_R_M，前向顺路半径下界 / m，"
                        f"缺省 {INLINE_R_MIN_IN:.0f}，二分搜索收敛值")
    p.add_argument("--inline-r-max-in", type=float, default=INLINE_R_MAX_IN,
                   help=f"内圈站前向顺路半径上界 / m，缺省 {INLINE_R_MAX_IN:.0f}）")
    p.add_argument("--inline-r-min-out", type=float, default=INLINE_R_MIN_OUT,
                   help=f"外圈站，距原点 > INLINE_OUTER_R_M，也就是 1850 m 外圈点，其前向顺路半径"
                        f"下界 / m，缺省 {INLINE_R_MIN_OUT:.0f}）")
    p.add_argument("--inline-near-r", type=float, default=INLINE_NEAR_R,
                   help=f"每站到站后的近距顺路清除半径 / m，缺省 {INLINE_NEAR_R:.0f}）")
    p.add_argument("--inline-max-mec-r", type=float, default=INLINE_MAX_MEC_R,
                   help=f"参与顺路清除的区域最大最小覆盖圆半径 / m，缺省 {INLINE_MAX_MEC_R:.0f}；"
                        f"区域更大的频道不参与顺路，留给收尾阶段")
    p.add_argument("--survey-only", action="store_true",
                   help="只做阶段一，也就是扫描，不做定位与清除")
    p.add_argument("--traj-dir", default=TRAJ_DIR,
                   help=f"轨迹图输出子目录，相对 --save-dir，缺省 {TRAJ_DIR}）")
    p.add_argument("--no-plot", action="store_true",
                   help=f"不出逐局轨迹图；缺省每局在 <save-dir>/{TRAJ_DIR}/ 生成同名 png + csv")
    p.add_argument("--layout-file", default=None,
                   help="改用指定 (K,2) 布局 npy，论文布局对照实验用；缺省用 solver 里的 20 点定案")
    p.add_argument("--quiet", action="store_true",
                   help="只输出最终结论与产物路径，比缺省更安静，供批处理用")
    p.add_argument("--verbose", action="store_true",
                   help="打印完整过程：扫描方案明细、每局阶段统计、接口回显、图路径等；"
                        "与 --quiet 同时给出时以它为准")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """按模式分派：`--practice` 走本地演练，`--plan-only` 只求扫描方案，其余连官方

    缺省输出只报关键结论：标题、扫描方案一行、每局一行、汇总几行、产物路径。`--verbose` 还原
    完整过程，含 `=` 分隔线、扫描方案明细、每局阶段统计、图路径。`--quiet` 只留最终产物路径。
    """
    relax_console_encoding()
    args = build_parser().parse_args(argv)
    set_level(LEVEL_VERBOSE if args.verbose else LEVEL_QUIET if args.quiet else LEVEL_NORMAL)
    official = not args.practice and not args.plan_only
    # 队号一律运行时传入，代码里不留任何队号；只有纯离线的 --plan-only 不需要
    if args.robot_id is None and (args.practice or official):
        print("错误：跑演练/官方模式必须用 --robot-id 传入参赛队号，如 --robot-id <12 位队号>。",
              file=sys.stderr)
        return 2
    if args.save_dir is None:
        args.save_dir = RESULTS_DIR
    save_dir = Path(args.save_dir)

    if is_verbose():
        print("=" * 78)
    if not is_quiet():
        print("2026 CUMCM B 题 · 问题四：机器狗搜索与清除干扰源，定向 + 全向混合，确定性策略")
    if is_verbose():
        print("=" * 78)

    # 第一步：求扫描方案并做听到率统计，只算一次，随后所有局共用这份点集
    src = None
    if args.layout_file:
        plan = plan_from_points(np.load(args.layout_file))
        src = args.layout_file                     # 缺省档并进"扫描方案"一行，详细档单独一行
        if is_verbose():
            print(f"布局来源：{args.layout_file}，{plan.n_points} 个测量位置，里程 "
                  f"{plan.route_m / 1000:.2f} km")
    else:
        plan = build_sweep_plan()
    verify = verify_hearing_stats(plan.points)
    if is_verbose():
        print_sweep_report(plan, verify)
    elif not is_quiet():                            # 缺省档：扫描方案一行结论
        print((f"扫描方案（布局来自 {src}）：" if src else "扫描方案：")
              + f"7 覆盖基点"
              + (f" + {plan.interior_n} 内部补点" if plan.interior_n > 0 else "")
              + f" + {plan.outer_n} 均匀外圈点，测量位置 {plan.n_points} 个，"
              f"访问里程 {plan.route_m / 1000:.2f} km，"
              f"听到率 {verify['hear_rate'] * 100:.4f}%，共 {verify['n_cases']} 个算例，漏 "
              f"{verify['n_fail']} 个")
    if not plan.verification:
        plan = SweepPlan(outer_n=plan.outer_n, outer_radius=plan.outer_radius,
                         points=plan.points, route=plan.route, route_m=plan.route_m,
                         verification=verify)
    paths = save_plan(plan, save_dir, verify)
    # 扫描方案是三种模式共同的产物，所以缺省与 --quiet 下这一行都保留；--quiet 只留产物路径
    print("扫描方案已保存：" + "，".join(str(p) for p in paths))

    try:
        if args.practice:
            return run_practice(args, plan, verify, save_dir)
        if official:
            return run_official(args, plan, verify, save_dir)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except OSError as exc:
        sys.stdout.flush()
        print(f"连接模拟器失败：{exc}", file=sys.stderr)
        print(f"请确认模拟器已启动并处于测试窗口内，缺省地址 {BASE_URL}。", file=sys.stderr)
        if official:
            print("本地验证策略可用 --practice N，自动拉起 jammers-py，不需要官方模拟器。",
                  file=sys.stderr)
        return 1
    if is_verbose():
        print("提示：不加参数就连官方模拟器；加 --practice N 跑本地演练，自动拉起 jammers-py。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())