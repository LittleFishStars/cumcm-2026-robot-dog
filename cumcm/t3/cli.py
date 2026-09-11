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
from cumcm.common.sim_client import ROBOT_ID, Simulator
from cumcm.t3.config import (CHOSEN_RING_RADIUS, CLEAR_RADIUS, COVER_RADIUS, INLINE_DETOUR, INLINE_TRY_RADIUS, K_CLEAR_MAX, RESULTS_DIR, SEED, TRAJ_DIR)
from cumcm.t3.covering import (CoverSolveResult, optimal_ring_radius, print_cover_report, solve_covering_circles)
from cumcm.t3.plotting import save_trajectory, truth_points
from cumcm.t3.report import (episode_row, save_plan, save_survey, truth_check, observation_rows)
from cumcm.t3.strategy import RobotDog


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
    with PracticeArena(jammers_dir, robot_id=args.robot_id,
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
                           inline_try_radius=args.inline_try_radius,
                           inline_detour=args.inline_detour,
                           rotate=not args.no_rotate)
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
                files = save_trajectory(
                    save_dir, name, dog.actions, dog.plan, dog.survey_order_used,
                    truth_points(truth),
                    title=f"第 {ep + 1} 局（seed={seed}）：清除 {stats['cleared']}/{len(truth)}、"
                          f"里程 {stats['travel_m']:.0f} m、虚拟时间 {stats['virtual_time_s']:.0f} s",
                    traj_dir=args.traj_dir)
                print("  轨迹图：" + "，".join(str(f) for f in files))
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
             + save_survey(save_dir, rows, observations, res.to_json()))
    print("结果已保存：" + "，".join(str(p) for p in paths))
    return 0


def run_official(args: argparse.Namespace, res: CoverSolveResult, save_dir: Path) -> int:
    """官方评测接口模式：连 127.0.0.1 上已开放接口的模拟器，跑完整一局。

    与演练的唯一差别是**拿不到干扰源真值**，因此定位误差等需要真值的指标留空。
    """
    sim = Simulator(robot_id=args.robot_id, base_url=args.base_url, timeout=args.timeout)
    print(f"连接模拟器 {args.base_url}（robot_id={args.robot_id}）")
    dog = RobotDog(sim, verbose=not args.quiet, logfile=args.log, episode=1,
                   clear=not args.survey_only, k_clear_max=args.k_clear_max,
                   inline_try_radius=args.inline_try_radius,
                   inline_detour=args.inline_detour,
                   rotate=not args.no_rotate)
    stats = dog.run(res.plan, res.survey_order)
    print(f"完成：清除 {stats['cleared']} 个，巡视 {stats['waypoints_visited']} 个圆心，"
          f"里程 {stats['travel_m']:.0f} m，虚拟时间 {stats['virtual_time_s']:.0f} s，"
          f"测向 {stats['n_measure']} 次（补测 {stats['n_probe']} 次），"
          f"听到 {stats['channels_heard']} 个频道（{stats['n_bearings']} 条示向度）")
    row = episode_row(1, args.seed, None, dog, stats,
                      truth_check(None, dog.plan, dog.obs, dog.cleared, dog.tracks))
    if not args.no_plot:
        files = save_trajectory(
            save_dir, f"ep01_seed{args.seed}", dog.actions, dog.plan,
            dog.survey_order_used, (),
            title=f"官方模式：清除 {stats['cleared']} 个、里程 {stats['travel_m']:.0f} m、"
                  f"虚拟时间 {stats['virtual_time_s']:.0f} s（无真值可比）",
            traj_dir=args.traj_dir)
        print("轨迹图：" + "，".join(str(f) for f in files))
    paths = (save_plan(res, save_dir)
             + save_survey(save_dir, [row], observation_rows(1, dog.plan, dog.meas),
                           res.to_json()))
    print("结果已保存：" + "，".join(str(p) for p in paths))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2026 CUMCM B 题问题三（第一阶段）：1000 m 覆盖圆求解 + 依次到圆心巡视扫描")
    p.add_argument("--practice", type=int, nargs="?", const=1, default=0,
                   help="本地演练局数：自动拉起 jammers-py 跑 N 局（缺省 1 局）")
    p.add_argument("--base-url", default=None,
                   help="官方模拟器地址（赛期用，如 http://127.0.0.1:2026）；缺省只求解覆盖圆")
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
    p.add_argument("--save-dir", default=RESULTS_DIR,
                   help=f"结果输出目录（缺省 {RESULTS_DIR}/）")
    p.add_argument("--log", default=None, help="过程日志文件（逐站扫描的文字过程，追加写入）")
    p.add_argument("--k-clear-max", type=int, default=K_CLEAR_MAX,
                   help=f"试清未中后最多再补清几个点（用 K 个半径 20 m 的圆覆盖定位区域；"
                        f"缺省 {K_CLEAR_MAX}，只在能盖满区域时才用，盖不满则转入补测）")
    p.add_argument("--inline-try-radius", type=float, default=INLINE_TRY_RADIUS,
                   help=f"巡视途中顺路试清允许的覆盖圆半径上限 / m（缺省 {INLINE_TRY_RADIUS:.0f}；"
                        f"设成 {CLEAR_RADIUS:.0f} 表示只清估计已够准的）")
    p.add_argument("--inline-detour", type=float, default=INLINE_DETOUR,
                   help=f"巡视途中顺路试清允许的绕行里程上限 / m（缺省 {INLINE_DETOUR:.0f}）")
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
    relax_console_encoding()
    args = build_parser().parse_args(argv)
    save_dir = Path(args.save_dir)

    res = solve_covering_circles(args.ring_radius, use_hex=args.hex_layout)   # 第一步：求 1000 m 覆盖圆的位置
    if not args.quiet:
        print_cover_report(res)
    paths = save_plan(res, save_dir)
    print("覆盖圆方案已保存：" + "，".join(str(p) for p in paths))

    try:
        if args.practice:                   # 第二步：依次移动到圆心进行扫描
            return run_practice(args, res, save_dir)
        if args.base_url:
            return run_official(args, res, save_dir)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except OSError as exc:
        print(f"连接模拟器失败：{exc}", file=sys.stderr)
        print("请确认模拟器已启动并处于测试窗口内（默认地址 http://127.0.0.1:2026）。",
              file=sys.stderr)
        return 1
    print("提示：加 --practice N 跑本地演练（自动拉起 jammers-py），"
          "或加 --base-url 连官方模拟器执行巡视扫描。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
