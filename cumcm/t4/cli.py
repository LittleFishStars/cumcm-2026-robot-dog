"""运行编排与命令行入口（问题四）：两种模式 + 参数解析。

* `run_practice` —— 本地演练：jammers-py 生成**问题四场景**（定向 + 全向混合），能拿到真值；
* `run_official` —— 正式模式：连官方模拟器，真值不可见，策略完全相同。

顶层 T4.py 只调用本模块的 main()，用法与 T3.py 保持一致（--practice / --plan-only / 无参数
即连官方模拟器）。
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
from cumcm.common.scanfigure import STEP_DIR_NAME, reset_dir
from cumcm.common.sim_client import API_LOG_NAME, BASE_URL, ROBOT_ID, Simulator
from cumcm.common.sim_client import api_log as _api_log_raw
from cumcm.t4.config import (BEARING_ERROR_DEG, CLEAR_RADIUS, K_CLEAR_MAX, PROBLEM_NO,
                             RESULTS_DIR, SEED, TRAJ_DIR)
from cumcm.t4.plotting import save_scan_figures, save_trajectory, truth_points
from cumcm.t4.report import (episode_row, observation_rows, save_plan, save_survey,
                             truth_check)
from cumcm.t4.strategy import RobotDog
from cumcm.t4.sweep import SweepPlan, build_sweep_plan, print_sweep_report, verify_hearing_stats


def _api_log(args: argparse.Namespace, echo: bool):
    """接口日志上下文（与 t3/t3ga 同一套机制，便于用同样方式审计）。"""
    return _api_log_raw(args.save_dir, args.api_log, echo)


def _episode_printer(clear: bool):
    """按模式打印本局小结。"""
    def show(stats: dict, check: dict, n_sources: Optional[int]) -> None:
        fh = [v for v in stats["first_heard"].values()]
        worst = max(fh) if fh else 0
        print(f"本局：扫描 {stats['travel_m']:.0f} m + 收尾，虚拟时间 {stats['virtual_time_s']:.0f} s，"
              f"测向 {stats['n_measure']} 次；扫描结束听到 {stats['channels_heard']}/{n_sources} "
              f"个源（全部在扫描第 {worst} 步内听到，共 {stats['n_bearings']} 条示向度，"
              f"顺路补测 {stats['n_side_scan']} 次）")
        if clear:
            print(f"  清除：{stats['cleared']}/{n_sources}（平均 {stats['avg_time_s']:.1f} s/个），"
                  f"定位误差均值 {check['localize_err_mean_m']} m / 最大 "
                  f"{check['localize_err_max_m']} m；方法分布 {stats['methods']}")
        print(f"  命中核对：{check['n_heard']}/{check['n_sources']} 个源被听到"
              f"（{check['all_heard']}），定位误差在清除半径内 "
              f"{check['n_within_clear_radius']} 个")
    return show


def run_practice(args: argparse.Namespace, plan: SweepPlan, verify: dict, save_dir: Path) -> int:
    """本地演练：自动拉起 jammers-py（问题四场景），跑 N 局并汇总。"""
    jammers_dir = (Path(args.jammers_dir) if args.jammers_dir else default_jammers_dir())
    clear = not args.survey_only
    show = _episode_printer(clear)
    rows: List[dict] = []
    observations: List[dict] = []
    with _api_log(args, not args.quiet) as api_log, \
            PracticeArena(jammers_dir, robot_id=args.robot_id,
                          robot_port=args.robot_port, console_port=args.console_port,
                          reuse_existing=not args.no_reuse,
                          problem_no=PROBLEM_NO) as arena:
        print(f"jammers-py 已就绪：机器狗接口 {arena.robot_url}，控制台 {arena.console_url}"
              f"（问题四场景：定向 + 全向混合）")
        for ep in range(args.practice):
            seed = args.seed + ep
            truth = arena.start_episode(seed)
            print(f"\n----- 演练第 {ep + 1}/{args.practice} 局（seed={seed}，"
                  f"干扰源 {len(truth)} 个）-----")
            dog = RobotDog(Simulator(robot_id=args.robot_id, base_url=arena.robot_url,
                                     timeout=args.timeout),
                           verbose=not args.quiet, logfile=args.log, episode=ep + 1,
                           clear=clear, k_clear_max=args.k_clear_max, api_log=api_log)
            stats = dog.run(plan)
            arena.finish_episode()
            check = truth_check(truth, plan, dog.obs, dog.cleared, dog.tracks,
                                dog.first_heard)
            rows.append(episode_row(ep + 1, seed, truth, dog, stats, check))
            observations.extend(observation_rows(ep + 1, plan, dog.meas))
            show(stats, check, len(truth))
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
                if ep == args.practice - 1:      # 逐步扫描图只保留最新一局（37 步 × N 局太慢）
                    reset_dir(save_dir / STEP_DIR_NAME)
                    scans = save_scan_figures(save_dir, name, dog.scan_steps, plan, (), tp)
                else:
                    scans = []
                print("  总轨迹图：" + "，".join(str(Path(f).resolve()) for f in files))
                if scans:
                    print(f"  逐步扫描结果图：{len(scans)} 张 → "
                          f"{(save_dir / STEP_DIR_NAME).resolve()}/")
    print("\n" + "=" * 78)
    if clear:
        print(f"汇总（{len(rows)} 局）：平均清除比例 "
              f"{np.mean([r['clear_ratio'] for r in rows]):.4f}"
              f"（{sum(r['cleared'] for r in rows)}/{sum(r['n_sources'] for r in rows)}），"
              f"平均虚拟时间 {np.mean([r['virtual_time_s'] for r in rows]):.0f} s，"
              f"平均里程 {np.mean([r['travel_m'] for r in rows]):.0f} m，"
              f"平均测向 {np.mean([r['n_measure'] for r in rows]):.0f} 次"
              f"（文献补测 {np.mean([r['n_probe'] for r in rows]):.0f} 次、顺路补测 "
              f"{np.mean([r['n_side_scan'] for r in rows]):.0f} 次、"
              f"判定必无信号跳过 {np.mean([r['n_skip_measure'] for r in rows]):.0f} 次）")
        print(f"  定位误差：均值 "
              f"{np.mean([r['localize_err_mean_m'] for r in rows if r['localize_err_mean_m']]):.2f}"
              f" m，最差单源 "
              f"{max([r['localize_err_max_m'] for r in rows if r['localize_err_max_m']] or [0]):.2f}"
              f" m；最晚首次听到发生在第 "
              f"{max([r['worst_first_heard_step'] for r in rows if r['worst_first_heard_step'] is not None] or [0])}"
              f" 步（共 {plan.n_points} 个测量位置）")
        print("逐局：" + "  ".join(f"seed{r['seed']}={r['cleared']}/{r['n_sources']}"
                                  f"({r['virtual_time_s']:.0f}s)" for r in rows))
    else:
        print(f"汇总（{len(rows)} 局，仅扫描）：平均里程 {np.mean([r['travel_m'] for r in rows]):.0f} m，"
              f"平均虚拟时间 {np.mean([r['virtual_time_s'] for r in rows]):.0f} s，"
              f"平均测向 {np.mean([r['n_measure'] for r in rows]):.0f} 次，"
              f"共听到 {sum(r['heard'] for r in rows)}/{sum(r['n_sources'] for r in rows)} 个源")
        print("逐局：" + "  ".join(f"seed{r['seed']}={r['heard']}/{r['n_sources']}"
                                  for r in rows))
    print("=" * 78)
    paths = (save_plan(plan, save_dir, verify)
             + save_survey(save_dir, rows, observations, plan.to_json(),
                           {"mode": "practice", "problem_no": PROBLEM_NO,
                            "robot_id": args.robot_id, "seed0": args.seed,
                            "episodes": args.practice,
                            "bearing_error_deg": BEARING_ERROR_DEG}))
    print("结果已保存：" + "，".join(str(p) for p in paths))
    if api_log is not None:
        api_log.report()
    return 0


def run_official(args: argparse.Namespace, plan: SweepPlan, verify: dict, save_dir: Path) -> int:
    """官方评测接口模式：连 127.0.0.1 上已开放接口的模拟器，跑完整一局。

    拿不到真值，故真值相关字段留空；扫描布局（实测听到率 ~99.8%）与演练里验证过的行为在
    正式模式同样成立。接口调用全程落盘 api_calls.jsonl —— 官方模式唯一证据链。
    """
    sim = Simulator(robot_id=args.robot_id, base_url=args.base_url, timeout=args.timeout)
    print(f"连接模拟器 {args.base_url}（robot_id={args.robot_id}，问题四）")
    with _api_log(args, not args.quiet) as api_log:
        dog = RobotDog(sim, verbose=not args.quiet, logfile=args.log, episode=1,
                       clear=not args.survey_only, k_clear_max=args.k_clear_max,
                       api_log=api_log)
        stats = dog.run(plan)
        print(f"完成：清除 {stats['cleared']} 个，里程 {stats['travel_m']:.0f} m，"
              f"虚拟时间 {stats['virtual_time_s']:.0f} s，测向 {stats['n_measure']} 次"
              f"（补测 {stats['n_probe']} 次），扫描结束听到 {stats['channels_heard']} 个频道"
              f"（{stats['n_bearings']} 条示向度）")
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
            print("总轨迹图：" + "，".join(str(Path(f).resolve()) for f in files))
            print(f"逐步扫描结果图：{len(scans)} 张 → "
                  f"{(save_dir / STEP_DIR_NAME).resolve()}/")
        paths = (save_plan(plan, save_dir, verify)
                 + save_survey(save_dir, [row], observation_rows(1, plan, dog.meas),
                               plan.to_json(),
                               {"mode": "official", "problem_no": PROBLEM_NO,
                                "base_url": args.base_url, "robot_id": args.robot_id,
                                "episodes": 1, "bearing_error_deg": BEARING_ERROR_DEG,
                                "note": "官方模式接口不返回真值，故真值相关字段为 null"}))
        print("结果已保存：" + "，".join(str(p) for p in paths))
        if api_log is not None:
            api_log.report()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2026 CUMCM B 题问题四：定向 + 全向混合干扰源的搜索与清除（扫描 + 定位清除）")
    p.add_argument("--practice", type=int, nargs="?", const=1, default=0,
                   help="本地演练局数：自动拉起 jammers-py 跑 N 局（缺省 1 局）")
    p.add_argument("--base-url", default=BASE_URL,
                   help=f"官方模拟器地址（缺省 {BASE_URL}）；不带任何参数即连它跑完一局。"
                        f"只想看扫描方案、不连模拟器时用 --plan-only")
    p.add_argument("--plan-only", action="store_true",
                   help="只求解并保存扫描方案（含听到率统计），不连模拟器")
    p.add_argument("--api-log", default=None,
                   help=f"接口调用日志路径（缺省 <save-dir>/{API_LOG_NAME}；传空串关闭）")
    p.add_argument("--robot-id", default=ROBOT_ID, help="参赛队号（须与模拟器一致）")
    p.add_argument("--timeout", type=float, default=5.0, help="HTTP 超时 / s")
    p.add_argument("--jammers-dir", default=None,
                   help="jammers-py 目录（缺省为本仓库根目录下的 jammers-py/）")
    p.add_argument("--console-port", type=int, default=8090,
                   help="演练时 jammers-py 控制台端口（缺省 8090，被占用则自动顺延）")
    p.add_argument("--robot-port", type=int, default=2026,
                   help="演练时 jammers-py 的机器狗接口端口（缺省 2026，与官方一致）")
    p.add_argument("--no-reuse", action="store_true",
                   help="不使用已在运行的 jammers-py，另起独占实例（并行演练时用）")
    p.add_argument("--seed", type=int, default=SEED,
                   help="演练第 1 局的种子（场景布局与示向度噪声都由它确定，可复现）")
    p.add_argument("--save-dir", default=None,
                   help=f"结果输出目录（缺省 {RESULTS_DIR}/，不按演练/官方分家）")
    p.add_argument("--log", default=None, help="过程日志文件（逐点扫描的文字过程，每局重写）")
    p.add_argument("--k-clear-max", type=int, default=K_CLEAR_MAX,
                   help=f"试清未中后最多再补清几个点（缺省 {K_CLEAR_MAX}）")
    p.add_argument("--survey-only", action="store_true",
                   help="只做阶段一（扫描），不做定位与清除")
    p.add_argument("--traj-dir", default=TRAJ_DIR,
                   help=f"轨迹图输出子目录（相对 --save-dir；缺省 {TRAJ_DIR}）")
    p.add_argument("--no-plot", action="store_true",
                   help=f"不出逐局轨迹图（缺省每局在 <save-dir>/{TRAJ_DIR}/ 生成同名 png + csv）")
    p.add_argument("--quiet", action="store_true", help="只输出汇总，不打印过程")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """按模式分派：`--practice` → 本地演练；`--plan-only` → 只求扫描方案；其余 → 官方。"""
    relax_console_encoding()
    args = build_parser().parse_args(argv)
    official = not args.practice and not args.plan_only
    if args.save_dir is None:
        args.save_dir = RESULTS_DIR
    save_dir = Path(args.save_dir)

    print("=" * 78)
    print("2026 CUMCM B 题 · 问题四：机器狗搜索与清除干扰源（定向 + 全向混合，确定性策略）")
    print("=" * 78)

    # 第一步：求扫描方案并做听到率统计（只需一次；随后所有局共用这份点集）
    plan = build_sweep_plan()
    verify = verify_hearing_stats(plan.points)
    if not args.quiet:
        print_sweep_report(plan, verify)
    if not plan.verification:
        plan = SweepPlan(extend_k=plan.extend_k, extend_clamp=plan.extend_clamp,
                         points=plan.points, route=plan.route, route_m=plan.route_m,
                         verification=verify)
    paths = save_plan(plan, save_dir, verify)
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
        print(f"请确认模拟器已启动并处于测试窗口内（缺省地址 {BASE_URL}）。", file=sys.stderr)
        if official:
            print("本地验证策略可用 --practice N（自动拉起 jammers-py，无需官方模拟器）。",
                  file=sys.stderr)
        return 1
    print("提示：不加参数即连官方模拟器；加 --practice N 跑本地演练（自动拉起 jammers-py）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())