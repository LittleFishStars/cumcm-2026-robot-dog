"""运行编排与命令行入口（GA 对照方案）。

顶层 T3_ga.py 只调用本模块的 main()，故命令行用法与拆分前一致。
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
from cumcm.common.sim_client import BASE_URL, ROBOT_ID, Simulator
from cumcm.common.sim_client import api_log as _api_log_raw
from cumcm.t3ga.config import (
API_LOG_NAME, RESULTS_DIR)
from cumcm.t3ga.covering import covering_waypoints
from cumcm.t3ga.plotting import TRAJ_DIR_NAME, save_trajectory_plot, sources_of
from cumcm.t3ga.strategy import RobotDog
from cumcm.t3ga.training import (episode_row, ga_meta, print_ga_summary,
                                 save_training_results)

def _api_log(args: argparse.Namespace, echo: bool):
    """接口日志上下文（把 CLI 参数拆成公共层 api_log 所需的参数）。

    路径优先取 --api-log；未指定时用 <save-dir>/api_calls.jsonl；显式传空串则关闭日志。
    """
    return _api_log_raw(args.save_dir, args.api_log, echo)


def _save_and_report(args: argparse.Namespace, episodes: List[dict], ga_runs: List[dict],
                     meta: dict, api_log: Optional[ApiLog] = None,
                     traj_paths: Optional[Sequence[Path]] = None) -> None:
    """打印 GA 训练摘要，写出训练结果，汇总轨迹图与接口调用日志（产物路径并列在尾部）。"""
    print_ga_summary(ga_runs)
    paths = save_training_results(Path(args.save_dir), episodes, ga_runs, meta)
    print("训练结果已保存：" + "，".join(str(p) for p in paths))
    if traj_paths:
        where = Path(args.save_dir) / TRAJ_DIR_NAME
        print(f"轨迹图已保存：{where}/ 共 {len(traj_paths)} 张（机器狗行驶轨迹，逐局一张）")
    if api_log is not None:
        api_log.report()


def run_official(args: argparse.Namespace) -> int:
    """官方评测平台的正式流程：连 127.0.0.1 上已开放接口的模拟器，跑完一局。

    与演练的关键差别：**拿不到干扰源真值**，因此定位误差等需要真值的指标留空；
    结果默认写到 results/official/，不覆盖演练训练批的数据。
    """
    if args.save_dir is None:
        args.save_dir = str(Path(RESULTS_DIR) / "official")
    sim = Simulator(robot_id=args.robot_id, base_url=args.base_url, timeout=args.timeout)
    print(f"连接模拟器 {args.base_url}（robot_id={args.robot_id}）")
    with _api_log(args, not args.quiet) as api_log:
        dog = RobotDog(sim, verbose=not args.quiet, logfile=args.log, seed=args.seed,
                       episode=1, api_log=api_log)
        stats = dog.run()
        print(f"完成：清除 {stats['cleared']} 个，虚拟总时间 {stats['total_time_s']:.1f} s，"
              f"平均 {stats['avg_time_s']:.1f} s/个，测向 {stats['n_measure']} 次，"
              f"清除动作 {stats['n_clear']} 次")
        row = episode_row(1, args.seed, None, dog, stats)
        meta = {"mode": "official", "base_url": args.base_url, "robot_id": args.robot_id,
                "seed": args.seed, **ga_meta()}
        traj = (None if args.no_plot else
                save_trajectory_plot(args.save_dir, "ep01", dog.track, (), dog.hits, dog.marks,
                                     title=f"官方测试 · 清除 {stats['cleared']} 个 · "
                                           f"虚拟时间 {stats['total_time_s']:.0f} s"))
        _save_and_report(args, [row], dog.ga_runs, meta, api_log,
                         [traj] if traj else [])
    return 0


def run_practice(args: argparse.Namespace) -> int:
    """本地演练：自动拉起 jammers-py，跑 N 局场景并汇总。

    第 i 局用 seed=args.seed+i：场景布局与示向度噪声都由它确定，因此同一 --seed 的整轮
    演练完全可复现（含各次 GA 的解），可用于新旧策略的严格对比。
    """
    if args.save_dir is None:
        args.save_dir = RESULTS_DIR
    jammers_dir = (Path(args.jammers_dir) if args.jammers_dir
                   else default_jammers_dir())
    rows: List[dict] = []
    ga_runs: List[dict] = []
    traj_paths: List[Path] = []
    with _api_log(args, not args.quiet) as api_log, \
            PracticeArena(jammers_dir, robot_id=args.robot_id,
                          console_port=args.console_port) as arena:
        print(f"jammers-py 已就绪：机器狗接口 {arena.robot_url}，控制台 {arena.console_url}")
        for ep in range(args.practice):
            seed = args.seed + ep
            truth = arena.start_episode(seed)
            print(f"\n----- 演练第 {ep + 1}/{args.practice} 局（seed={seed}，"
                  f"干扰源 {len(truth)} 个）-----")
            dog = RobotDog(Simulator(robot_id=args.robot_id, base_url=arena.robot_url,
                                             timeout=args.timeout),
                           verbose=not args.quiet, logfile=args.log, seed=seed, episode=ep + 1,
                           api_log=api_log)
            stats = dog.run()
            ga_runs.extend(dog.ga_runs)
            engine = arena.finish_episode()
            rows.append(episode_row(ep + 1, seed, truth, dog, stats, engine))
            r = rows[-1]
            avg_txt = f"{r['avg_time_s']:.1f} s/个" if r["avg_time_s"] else "—"
            err_txt = (f"定位误差 {r['localize_err_mean_m']:.1f} m 均值 / "
                       f"{r['localize_err_max_m']:.1f} m 最大"
                       if r["n_located"] else "无定位误差数据")
            print(f"本局：清除 {r['cleared']}/{r['n_sources']}（{r['clear_ratio']:.3f}），"
                  f"虚拟总时间 {r['virtual_time_s']:.1f} s，平均 {avg_txt}，"
                  f"测向 {r['n_measure']} 次，{err_txt}")
            # 轨迹图在 /exit 之后画，不占用现实时间预算
            traj = (None if args.no_plot else
                    save_trajectory_plot(args.save_dir, f"ep{ep + 1:02d}_seed{seed}",
                                         dog.track, sources_of(truth), dog.hits, dog.marks,
                                         title=f"第 {ep + 1} 局 · seed {seed} · "
                                               f"清除 {r['cleared']}/{r['n_sources']} · "
                                               f"虚拟时间 {r['virtual_time_s']:.0f} s"))
            if traj:
                traj_paths.append(traj)
                if not args.quiet:
                    print(f"  轨迹图 {traj}")
    print("\n" + "=" * 74)
    avgs = [r["avg_time_s"] for r in rows if r["avg_time_s"]]
    print(f"汇总（{len(rows)} 局）：平均清除比例 {np.mean([r['clear_ratio'] for r in rows]):.4f}，"
          + (f"平均 {np.mean(avgs):.1f} s/个，" if avgs else "")
          + f"平均虚拟总时间 {np.mean([r['virtual_time_s'] for r in rows]):.1f} s")
    print("逐局：" + "  ".join(f"seed{r['seed']}={r['cleared']}/{r['n_sources']}" for r in rows))
    print("=" * 74)
    meta = {"mode": "practice", "robot_id": args.robot_id, "seed0": args.seed,
            "episodes": args.practice, **ga_meta(),
            "coverage_waypoints": len(covering_waypoints())}
    _save_and_report(args, rows, ga_runs, meta, api_log, traj_paths)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2026 CUMCM B 题问题三：遗传算法机器狗自动定位与清除")
    p.add_argument("--robot-id", default=ROBOT_ID, help="参赛队号（须与模拟器一致）")
    p.add_argument("--base-url", default=BASE_URL, help="官方模拟器地址")
    p.add_argument("--timeout", type=float, default=5.0, help="HTTP 超时 / s")
    p.add_argument("--practice", type=int, nargs="?", const=1, default=0,
                   help="本地演练局数：自动拉起 jammers-py 跑 N 局（缺省 1 局）")
    p.add_argument("--jammers-dir", default=None, help="jammers-py 目录（缺省为本仓库根目录下的 jammers-py/）")
    p.add_argument("--console-port", type=int, default=8090,
                   help="演练时 jammers-py 控制台端口（缺省 8090，被占用则自动顺延）")
    p.add_argument("--seed", type=int, default=0,
                   help="随机种子（演练第 1 局的场景布局、示向度噪声与 GA 都由它确定）")
    p.add_argument("--save-dir", default=None,
                   help=f"结果输出目录（演练缺省 {RESULTS_DIR}/，官方测试缺省 "
                        f"{RESULTS_DIR}/official/；写入 GA 训练记录与逐局统计）")
    p.add_argument("--log", default=None, help="过程日志文件（阶段/清除等文字过程，追加写入）")
    p.add_argument("--api-log", default=None,
                   help=f"接口调用日志（逐条记录 /enter /measure /clear /exit 的请求与"
                        f"原始响应；缺省 <save-dir>/{API_LOG_NAME}，传空字符串则关闭）")
    p.add_argument("--no-plot", action="store_true",
                   help=f"关闭每局轨迹图（缺省每局结束后在 <save-dir>/{TRAJ_DIR_NAME}/ 生成 PNG）")
    p.add_argument("--quiet", action="store_true", help="只输出汇总，不打印过程")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    relax_console_encoding()
    args = build_parser().parse_args(argv)
    print("=" * 74)
    print("2026 CUMCM B 题 · 问题三：遗传算法自动定位与清除")
    print("=" * 74)
    try:
        return run_practice(args) if args.practice else run_official(args)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except OSError as exc:      # 连接被拒/超时：多半是模拟器没启动或测试窗口未开放
        print(f"连接模拟器失败：{exc}", file=sys.stderr)
        print("请确认模拟器已启动并处于测试窗口内（默认地址 http://127.0.0.1:2026）。",
              file=sys.stderr)
        return 1
