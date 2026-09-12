"""drv_speed.py：演练实测"耗时导向"布局的虚拟时间（对比基线）。

模式：
  --baseline20  人工 20 点版：7 覆盖基点 + 12×1850 均匀外圈（文献环带结构）；
  --elite N     NN-GA 在外圈约束下搜索的精英榜（results/t4/.nn_speed_elite.npz）；
  --best        当前最优（.nn_speed_best.npy）。

布局统一组装：原点 + 7 基点 + 外圈点；路线 = 最近邻 + 2-opt（确定性）。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from cumcm.common.practice_arena import PracticeArena
from cumcm.common.routing import dist_matrix, nearest_order, two_opt_first, two_opt_greedy
from cumcm.common.sim_client import Simulator
from cumcm.t3.config import SURVEY_CENTERS
from cumcm.t4.nn_layout_speed import assemble
from cumcm.t4.strategy import RobotDog
from cumcm.t4.sweep import SweepPlan


def make_plan(points: np.ndarray) -> SweepPlan:
    pts = np.asarray(points, dtype=float)
    D = dist_matrix(pts, (0.0, 0.0))
    order = two_opt_greedy(two_opt_first(nearest_order(pts, start=(0.0, 0.0)), D), D)
    m = 0.0
    prev = pts[0]
    for i in order:
        m += float(np.hypot(*(pts[i] - prev)))
        prev = pts[i]
    n_outer = int((np.hypot(pts[:, 0], pts[:, 1]) > 1200.0).sum())
    return SweepPlan(outer_n=n_outer, outer_radius=1850.0,
                     interior_n=max(0, len(pts) - 1 - n_outer),
                     interior_radius=900.0, points=pts, route=list(order), route_m=m)


def baseline20() -> np.ndarray:
    a = 2.0 * math.pi * np.arange(12) / 12
    ring = np.stack([1850.0 * np.cos(a), 1850.0 * np.sin(a)], 1)
    return np.vstack([np.zeros(2), np.asarray(SURVEY_CENTERS, dtype=float), ring])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline20", action="store_true")
    ap.add_argument("--elite", type=int, default=0, help="精英榜前 N 个逐演（0=不演）")
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--robot-port", type=int, default=2001)
    ap.add_argument("--console-port", type=int, default=8095)
    a = ap.parse_args()

    with PracticeArena(Path(__file__).resolve().parent / "jammers-py", robot_id="A",
                       robot_port=a.robot_port, console_port=a.console_port,
                       reuse_existing=False, problem_no=4) as arena:
        def run(label: str, points: np.ndarray, seeds):
            plan = make_plan(points)
            for sd in seeds:
                arena.start_episode(sd)
                dog = RobotDog(Simulator(robot_id="A", base_url=arena.robot_url,
                                         timeout=120.0), verbose=False,
                               episode=sd + 1, clear=True)
                st = dog.run(plan)
                arena.finish_episode()
                print(f"{label} seed{sd}: 虚拟 {st['virtual_time_s']:.0f}s "
                      f"里程 {st['travel_m']:.0f}m 测向 {st['n_measure']} "
                      f"顺路 {st['n_inline']}h/{st['n_inline_fail']}f 清 {st['cleared']}")

        if a.baseline20:
            run("baseline20(7基点+12环)", baseline20(), a.seeds)
        if a.best:
            run("nn-best", assemble(np.load("results/t4/.nn_speed_best.npy")), a.seeds)
        if a.elite > 0:
            z = np.load("results/t4/.nn_speed_elite.npz")
            for i in range(min(a.elite, len(z["dir"]))):
                run(f"elite#{i+1}(dir {z['dir'][i]*100:.3f}%,{z['route'][i]:.0f}m)",
                    assemble(z["pts"][i]), a.seeds)


if __name__ == "__main__":
    main()