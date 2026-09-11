#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T3.py —— 2026 高教社杯全国大学生数学建模竞赛 B 题 · 问题三（第一阶段）

1000 m 覆盖圆求解 + 依次移动到圆心巡视扫描
==========================================

问题三要求机器狗从原点出发，在半径 1800 m 的目标圆域内自动定位并清除 10~16 个全向干扰源。
本程序实现整套策略的**第一阶段**：先用尽可能少的、半径为 1000 m 的"覆盖圆"铺满目标圆域，
求出这些覆盖圆的圆心的位置，再依次移动到各圆心处对 20 个频道扫描测向，把一个源的多视角
示向度采集齐全（第二阶段的交会定位与逐个清除将在此观测基础上继续开发）。

为什么覆盖圆半径取 1000 m
-------------------------
题目给出每个干扰源的有效接收半径在 1000~1500 m 之间，即"检测点距源不超过 1000 m 就一定
听得到"。因此只要一组半径 1000 m 的圆**完全覆盖**目标圆域，机器狗依次走遍这些圆的圆心并
在每个圆心停下测向，就必然不会漏掉任何干扰源 —— 这就是本策略"不漏源"的全部依据，不需要
对源的位置分布做任何先验假设。

覆盖圆的位置（解析解）
---------------------
布局取"1 个中心圆 + 6 个环上圆"（正六边形）：中心圆心在原点，6 个环心位于半径 d 的正六边形
顶点。由对称性，圆域内最坏点必落在某个环心方位角的角平分线上（与最近环心夹角 30°）：设其极径
为 ρ，则到最近环心的距离

    f(ρ) = sqrt(ρ² + d² - √3·ρ·d)

在 ρ = √3d/2 处取最小，且该处的距离恰为 d/2。记 ρ* = d/√3，它是 f(ρ) = ρ 的唯一正根，
把圆域分成两段：

    ρ ≤ ρ*： 最近的是中心圆，距离 = ρ，最坏值 = ρ* = d/√3        ← 内圈最坏（环圆与中心圆
                                                                   覆盖边界的交点）
    ρ ≥ ρ*： 最近的是环上圆，距离 = f(ρ)（凸函数），最大值必在端点：f(ρ*) = ρ* 或 f(R) = g₂

    g₂ = f(R) = sqrt(R² + d² - √3·R·d) = sqrt((d - √3R/2)² + R²/4) ≥ R/2

    全域最坏最近距离  D(d) = max( d/√3 ,  g₂ )      ← 内圈与圆域边界两处竞争

d/√3 随 d 单调增，g₂ 在 d = √3R/2 处取最小值 R/2，两者恰在该点相等：d*/√3 = R/2 = g₂(d*)。
于是

    最优环半径 d* = √3·R/2 = 900√3 ≈ 1558.85 m，
    全域最坏最近距离 = R/2 = 900.0 m ≤ 1000 m（余量 100 m）。        ★

（最坏点由此有两族：ρ* = 900 m 的角平分线方向点，以及圆域边界上 ρ = R、与某环心夹角 30° 的
点，两者到最近圆心的距离都恰为 900 m。）

6 个覆盖圆做不到：6 个圆全在环上（环心必须落在距原点 1000 m 内才能盖住原点），实测最优
（环半径 1000 m）最坏最近距离仍有 1059 m > 1000 m，圆域边缘会漏源 —— 所以 **7 个是最少个数**。
17 个可行半径的扫描、6 圆方案对照、解析值与实算值的交叉验证都由本程序自动完成。

覆盖重数（决定第二阶段的补测需求）
---------------------------------
7 个覆盖圆中，约 64% 的圆域面积只被 1 个圆覆盖：这些位置上的源只能得到 1 条示向度射线，
无法直接交会定距；约 34% 被 2 个圆覆盖、约 3% 被 3 个圆覆盖，可直接交会。本阶段采集到的
示向度因此在第二阶段需要配合"沿示向度前移补测"来处理单射线频道。

环半径的权衡与可行区间
----------------------
覆盖保证只要求 D(d) ≤ 1000 m，由此得到可行区间（两端点均为零余量）：

    内圈约束：d/√3 ≤ r  →  d ≤ √3·r = 1732.05 m
    边界约束：g₂(d) ≤ r →  d ∈ [1122.96, 1994.74] m
    合起来：  d ∈ [1122.96, 1732.05] m

而巡视里程恰为 6d（原点 → 一个环心 = d，再沿正六边形走 5 条边 = 5d），随 d 单调增。于是
"余量最大"与"里程最短"是一对矛盾：d* = 1558.85 m 使余量最大（100 m）、里程 9353 m；
若取可行下界 d = 1122.96 m，里程最短（6737 m，比 d* 省 2616 m ≈ 523 s 移动时间），但圆域
边缘的最坏距离恰好 1000 m、余量为 0，一旦源的接收半径正好取到下界 1000 m 就有漏源风险。
本程序默认取 d*（最稳），可用 --ring-radius 指定其他值，报告里给出权衡表供选择。

与参考文献的关系
----------------
* 赵一骁, 何航天, 李雨楠, 等. 基于最小圆覆盖的多无人机协同螺旋式搜索优化算法[J].
  指挥信息系统与技术, 2024, 15(4): 56-62.
  该文提出"圆覆盖 + 圆内接正六边形拼接"的环境建模：以传感器搜索半径 r 为覆盖圆，相邻圆心
  间距取紧贴的 √3·r（两圆重叠 5.77%、利用率 94.23%），覆盖圆圆心即航路关键点，只需遍历
  全部关键点就完成了区域覆盖搜索；圆域上螺旋遍历优于平行搜索。本题覆盖圆的个数与结构与
  之一致（都是"1 中心 + 6 环"共 7 个），但我们把环半径从紧贴间距 √3·r = 1732.05 m 改进为
  √3·R/2 = 1558.85 m：最坏最近距离由 1000.0 m（零余量）降到 900.0 m（余量 100 m），巡视
  里程同时由 10392 m 降到 9353 m，两项同时更优。差别来自边界条件：紧贴间距是"无限平面"
  最小圆覆盖问题的解，本题是有界圆域，最坏点由内圈与圆域边界两处共同决定（见上节）。
* 第二阶段（交会定位、第二检测点选择）的理论依据：
  任叶童. 基于到达角信息的无源定位算法研究[D]. 天津大学, 2016：双站交会的定位模糊区面积
  S = 4R²Δθ²ₘₐₓsinα₁sinα₂/sin³(α₁+α₂) 在 α₁ = α₂ = π/3 时最小；多站交会等价于以
  1/(σᵢRᵢ) 为权的加权最小二乘（最小化到各示向度线的垂距加权平方和）。
  Chen X, Xu Z, Rui L. An optimization algorithm of multi-observer trajectories for
  cooperative bearings-only target localization[C]. ICICS 2009：滤波均方位置误差对观测站到
  目标的距离呈平方反比（∝1/(σᵢ²rᵢ²)），最优观测轨迹是"贴近目标"与"拉开角度分集"的折衷。

运行
====
    python T3.py                                    # 只求解覆盖圆并打印报告（不连模拟器）
    python T3.py --ring-radius 1200                 # 改用其他环半径（里程更短、余量更小）
    python T3.py --practice 3                       # 本地演练 3 局（自动拉起 jammers-py）
    python T3.py --base-url http://127.0.0.1:2026   # 官方评测接口模式（赛期，先开模拟器）

结果落盘（--save-dir，缺省 results/，文件名固定便于论文与绘图引用）
    t3_cover_plan.json    覆盖圆求解结果（环半径、圆心、最坏距离、覆盖重数、巡视顺序）
    t3_cover_circles.csv  7 个覆盖圆的圆心坐标（序号、类型、x、y）
    t3_survey.json        逐局巡视扫描统计（含真值核对：每个源到最近圆心的距离、是否被听到）
    t3_observations.csv   逐条示向度观测（局号、频道、检测点坐标、示向度）

依赖：numpy；HTTP 层复用同目录 sim_api.py；本地演练用同目录 jammers-py/（纯标准库）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

import sim_api

# ----------------------------------------------------------------------------
# 常量（与题目附录 1 / 附录 2 一致）
# ----------------------------------------------------------------------------
REGION_RADIUS = 1800.0          # 目标圆域半径 / m
COVER_RADIUS = 1000.0           # 覆盖圆半径 = 有效接收半径下界 / m
CHANNELS: Tuple[int, ...] = tuple(range(1, 21))      # 20 个频道
COORD_LIMIT = 2.0e6             # 坐标分量绝对值上限 / m（协议规定）
REGION_MARGIN = 1.0             # 坐标裁剪保留的数值余量 / m
OBS_CAP = 3                     # 每个频道最多采集的示向度条数（够了就不再重复测）
SAFETY_MARGIN = 30.0            # 现实时限预留余量 / s

GRID_STEP = 5.0                 # 覆盖校验的细网格 / m
BOUNDARY_SAMPLES = 20000        # 覆盖校验的边界采样点数（圆域边界是"最坏点"候选密集区）
COARSE_STEP = 15.0              # 6 圆方案对照用的粗网格 / m
COARSE_BOUNDARY = 4000          # 6 圆方案对照的边界采样点数
TOL = 1e-9

SEED = 2026

RESULTS_DIR = "results"
PLAN_JSON = "t3_cover_plan.json"        # 覆盖圆求解结果
PLAN_CSV = "t3_cover_circles.csv"       # 覆盖圆圆心坐标
SURVEY_JSON = "t3_survey.json"          # 逐局巡视扫描统计
OBS_CSV = "t3_observations.csv"         # 逐条示向度观测


# ----------------------------------------------------------------------------
# 基础几何工具
# ----------------------------------------------------------------------------
def dist(a: Sequence[float], b: Sequence[float]) -> float:
    """两点距离 / m。"""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def bearing(a: Sequence[float], b: Sequence[float]) -> float:
    """a → b 的方位角（度，x 轴正向逆时针，[0, 360)）。"""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def clamp_to_region(x: float, y: float,
                    radius: float = REGION_RADIUS - REGION_MARGIN) -> Tuple[float, float]:
    """把坐标拉回作业圆域内（出域时沿原方向缩到边界）。"""
    r = math.hypot(x, y)
    if r <= radius or r == 0.0:
        return x, y
    k = radius / r
    return x * k, y * k


# ----------------------------------------------------------------------------
# 覆盖圆求解
# ----------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class CoverPlan:
    """覆盖圆方案：所有覆盖圆的圆心的位置。

    圆心编号：0 号为圆心在原点的那一个（中心圆），1~6 号为正六边形环上的圆心（逆时针，
    从 0° 方位角起）。这些圆心就是机器狗的巡视路点。
    """

    region_radius: float            # 目标圆域半径 / m
    cover_radius: float             # 覆盖圆半径（= 有效接收半径下界）/ m
    ring_radius: float              # 六边形环上圆心到原点的距离 d / m
    waypoints: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "waypoints", hex_layout(self.ring_radius))

    @property
    def centers(self) -> List[Tuple[float, float]]:
        return [(float(x), float(y)) for x, y in self.waypoints]


def hex_layout(ring_radius: float, n_ring: int = 6) -> np.ndarray:
    """正六边形布局的圆心：第 0 个在原点，其余 n_ring 个在半径 ring_radius 的环上。"""
    ang = np.arange(n_ring) * (2.0 * math.pi / n_ring)
    ring = np.stack((ring_radius * np.cos(ang), ring_radius * np.sin(ang)), axis=1)
    return np.vstack([[0.0, 0.0], ring])


def optimal_ring_radius(region_radius: float = REGION_RADIUS) -> float:
    """最优环半径 d* = √3·R/2：使最坏最近距离 D(d) 最小的解析解。"""
    return math.sqrt(3.0) * region_radius / 2.0


def feasible_ring_interval(region_radius: float = REGION_RADIUS,
                           cover_radius: float = COVER_RADIUS) -> Tuple[float, float]:
    """满足覆盖保证 D(d) ≤ r 的环半径区间 [d_min, d_max]（解析）。

    两个约束各给一段区间，取交集：
      内圈约束 d/√3 ≤ r           → d ≤ √3·r
      边界约束 g₂(d) ≤ r（解 d² - √3R·d + R² - r² = 0）
                                  → d ∈ [(√3R - √(4r²-R²))/2, (√3R + √(4r²-R²))/2]
    区间两端点都是零余量（最坏距离恰好 = r），并随 d 单调变化。
    """
    root = math.sqrt(max(4.0 * cover_radius ** 2 - region_radius ** 2, 0.0))
    lo = (math.sqrt(3.0) * region_radius - root) / 2.0
    hi = min(math.sqrt(3.0) * cover_radius,
             (math.sqrt(3.0) * region_radius + root) / 2.0)
    return lo, hi


def analytic_worst(ring_radius: float, region_radius: float = REGION_RADIUS,
                   cover_radius: float = COVER_RADIUS) -> float:
    """最坏最近距离的解析值 D(d) = max(d/√3, g₂)（推导见模块文档）。

    d/√3 是内圈（只有中心圆能覆盖到）的最坏距离；g₂ 是圆域边界上、与某环心夹角 30° 的点
    到最近环心的距离。两项在 d* = √3·R/2 处同时等于 R/2，即最优点的最坏距离。
    """
    def g(rho: float) -> float:
        return math.sqrt(max(rho * rho + ring_radius * ring_radius
                             - math.sqrt(3.0) * rho * ring_radius, 0.0))
    return max(ring_radius / math.sqrt(3.0), g(region_radius))


@lru_cache(maxsize=8)
def region_samples(step: float, n_boundary: int) -> np.ndarray:
    """目标圆域的采样点：细网格 + 圆边界均匀采样（最坏点常在边界上，须精细采样）。"""
    ax = np.arange(-REGION_RADIUS, REGION_RADIUS + TOL, step)
    gx, gy = np.meshgrid(ax, ax)
    pts = np.stack((gx.ravel(), gy.ravel()), axis=1)
    pts = pts[np.linalg.norm(pts, axis=1) <= REGION_RADIUS + TOL]
    t = np.linspace(0.0, 2.0 * math.pi, n_boundary, endpoint=False)
    ring = np.stack((REGION_RADIUS * np.cos(t), REGION_RADIUS * np.sin(t)), axis=1)
    return np.vstack([pts, ring])


def nearest_distances(points: np.ndarray, waypoints: np.ndarray,
                      chunk: int = 200_000) -> np.ndarray:
    """每个采样点到最近圆心的距离（分块计算，避免大矩阵占内存）。"""
    out = np.empty(len(points), dtype=float)
    for i in range(0, len(points), chunk):
        blk = points[i:i + chunk]
        d = np.linalg.norm(blk[:, None, :] - waypoints[None, :, :], axis=2)
        out[i:i + chunk] = d.min(axis=1)
    return out


def worst_candidates(ring_radius: float, region_radius: float = REGION_RADIUS) -> np.ndarray:
    """解析给出的最坏点候选：12 个角平分线方向 × {ρ* = d/√3, ρ = R}。

    圆域内的最坏点必落在某个"角平分线方向"上（与最近环心夹角 30°）：内圈最坏点在 ρ* = d/√3
    处（此时它到最近环心的距离恰为 ρ*，与中心圆打平），外圈最坏点在圆域边界 ρ = R 处。把这
    些点放进采样集合，解析解与实算解就能一致到机器精度，网格只用于旁证。
    """
    ang = np.arange(12) * (math.pi / 6.0) + math.pi / 6.0
    pts = [(r * math.cos(a), r * math.sin(a))
           for r in (ring_radius / math.sqrt(3.0), region_radius) for a in ang]
    return np.array(pts, dtype=float)


def cover_counts(points: np.ndarray, waypoints: np.ndarray, radius: float,
                 chunk: int = 200_000) -> np.ndarray:
    """每个采样点被几个覆盖圆同时覆盖（覆盖重数）。"""
    out = np.empty(len(points), dtype=np.int64)
    for i in range(0, len(points), chunk):
        blk = points[i:i + chunk]
        d = np.linalg.norm(blk[:, None, :] - waypoints[None, :, :], axis=2)
        out[i:i + chunk] = (d <= radius + TOL).sum(axis=1)
    return out


def nearest_order(waypoints: np.ndarray, start: Sequence[float] = (0.0, 0.0)) -> List[int]:
    """确定性最近邻访问顺序（并列时取编号小者）：从 start 出发依次走遍所有圆心。"""
    rest = list(range(len(waypoints)))
    order: List[int] = []
    cur = np.asarray(start, dtype=float)
    while rest:
        k = min(rest, key=lambda j: (float(np.linalg.norm(waypoints[j] - cur)), j))
        order.append(k)
        rest.remove(k)
        cur = waypoints[k]
    return order


def path_length(waypoints: np.ndarray, order: Sequence[int],
                start: Sequence[float] = (0.0, 0.0)) -> float:
    """按给定顺序走遍各圆心的总里程 / m。"""
    total, cur = 0.0, np.asarray(start, dtype=float)
    for j in order:
        total += float(np.linalg.norm(waypoints[j] - cur))
        cur = waypoints[j]
    return total


@dataclass
class CoverSolveResult:
    """覆盖圆求解的完整结果：方案 + 实算校验 + 权衡分析 + 文献方法对照。"""

    plan: CoverPlan
    worst_distance: float               # 实算最坏最近距离 / m（网格 + 边界 + 解析候选点）
    worst_point: Tuple[float, float]    # 实算最坏点位置
    analytic_worst: float               # 解析最坏最近距离 / m
    coverage_ratio: float               # 被覆盖的采样点比例（1.0 表示全覆盖）
    multiplicity: Dict[int, int]        # 覆盖重数 → 采样点数
    single_ratio: float                 # 单重覆盖（只有 1 条射线可用）的占比
    multi_ratio: float                  # 二重及以上覆盖的占比
    mean_multiplicity: float            # 圆域内平均覆盖重数（重复率 = 平均值 - 1）
    survey_order: List[int]             # 巡视顺序（圆心编号）
    survey_length: float                # 巡视总里程 / m
    feasible_interval: Tuple[float, float]   # 满足覆盖保证的环半径可行区间 / m
    tradeoff: List[Dict[str, float]]    # 环半径权衡：d、最坏距离、余量、里程、时间
    lattice: Dict[str, float]           # 参考文献紧贴六边形栅格（间距 √3·r）对照
    six_circle_worst: float             # 6 圆方案的实算最坏距离（不可行对照）/ m
    six_circle_radius: float            # 6 圆方案的最优环半径 / m

    @property
    def margin(self) -> float:
        """最坏最近距离相对覆盖半径（1000 m）的余量 / m。"""
        return COVER_RADIUS - self.worst_distance

    def to_json(self) -> Dict[str, Any]:
        plan = self.plan
        return {
            "region_radius_m": plan.region_radius,
            "cover_radius_m": plan.cover_radius,
            "ring_radius_m": round(plan.ring_radius, 3),
            "n_circles": len(plan.waypoints),
            "analytic_worst_m": round(self.analytic_worst, 4),
            "computed_worst_m": round(self.worst_distance, 4),
            "margin_m": round(self.margin, 4),
            "worst_point": [round(v, 2) for v in self.worst_point],
            "coverage_ratio": self.coverage_ratio,
            "multiplicity_points": self.multiplicity,
            "single_ratio": self.single_ratio,
            "multi_ratio": self.multi_ratio,
            "mean_multiplicity": round(self.mean_multiplicity, 4),
            "survey_order": self.survey_order,
            "survey_length_m": round(self.survey_length, 2),
            "centers": [{"id": i, "kind": "中心圆" if i == 0 else "环上圆",
                         "x": round(x, 3), "y": round(y, 3)}
                        for i, (x, y) in enumerate(plan.centers)],
            "feasible_ring_interval_m": [round(v, 3) for v in self.feasible_interval],
            "ring_radius_tradeoff": self.tradeoff,
            "reference_hex_lattice": self.lattice,
            "compare_six_circles": {"n_circles": 6, "best_ring_radius_m": self.six_circle_radius,
                                    "computed_worst_m": round(self.six_circle_worst, 4),
                                    "feasible": self.six_circle_worst <= COVER_RADIUS},
            "grid_step_m": GRID_STEP,
            "boundary_samples": BOUNDARY_SAMPLES,
        }


def _six_circle_best(step: float = COARSE_STEP,
                     n_boundary: int = COARSE_BOUNDARY) -> Tuple[float, float]:
    """6 个覆盖圆的可行性对照：环半径必须 ≤ 1000 m 才能盖住原点，扫描取最优。

    返回（最优环半径 / m, 该半径下的最坏最近距离 / m）。粗网格仅用于可行性判断。
    """
    pts = region_samples(step, n_boundary)
    best = (float("inf"), 0.0)
    for d in np.arange(200.0, COVER_RADIUS + 1.0, 10.0):
        wp = hex_layout(float(d))[1:]          # 只有环上 6 个圆，没有中心圆
        m = float(nearest_distances(pts, wp).max())
        if m < best[0]:
            best = (m, float(d))
    return best[1], best[0]


def layout_metrics(ring_radius: float, pts: np.ndarray,
                   compute_multiplicity: bool = True) -> Dict[str, Any]:
    """给定环半径，报出该布局的最坏最近距离、余量、覆盖重数与巡视里程。"""
    wp = hex_layout(float(ring_radius))
    near = nearest_distances(pts, wp)
    order = nearest_order(wp)
    length = path_length(wp, order)
    row: Dict[str, Any] = {
        "ring_radius_m": round(float(ring_radius), 3),
        "computed_worst_m": round(float(near.max()), 3),
        "analytic_worst_m": round(analytic_worst(float(ring_radius)), 3),
        "margin_m": round(COVER_RADIUS - float(near.max()), 3),
        "survey_length_m": round(length, 1),
        "survey_time_s": round(length / 5.0, 1),
    }
    if compute_multiplicity:
        cnt = cover_counts(pts, wp, COVER_RADIUS)
        row["mean_multiplicity"] = round(float(cnt.mean()), 4)
        row["min_multiplicity"] = int(cnt.min())
    return row


def tradeoff_table(interval: Tuple[float, float], pts: np.ndarray) -> List[Dict[str, Any]]:
    """可行区间内取若干代表环半径，给出"余量 vs 里程"的权衡表。"""
    lo, hi = interval
    radii = [lo, 1200.0, 1300.0, 1400.0, optimal_ring_radius(), 1700.0, hi]
    radii = [r for r in radii if lo - 1e-9 <= r <= hi + 1e-9]
    return [layout_metrics(r, pts) for r in sorted(set(round(r, 3) for r in radii))]


def solve_covering_circles(ring_radius: Optional[float] = None) -> CoverSolveResult:
    """求解 1000 m 覆盖圆的位置，并做实算校验、可行区间与权衡分析、文献方法对照。"""
    d_star = optimal_ring_radius()
    d = d_star if ring_radius is None else float(ring_radius)
    plan = CoverPlan(REGION_RADIUS, COVER_RADIUS, d)
    assert len(plan.waypoints) == 7, "覆盖圆个数应为 7（1 中心 + 6 环）"

    # 采样点 = 细网格 + 圆边界精细扫描 + 解析给出的最坏点候选（保证解析/实算一致）
    pts = np.vstack([region_samples(GRID_STEP, BOUNDARY_SAMPLES),
                     worst_candidates(d)])
    near = nearest_distances(pts, plan.waypoints)
    k = int(near.argmax())
    worst = float(near[k])
    worst_point = (float(pts[k][0]), float(pts[k][1]))

    cnt = cover_counts(pts, plan.waypoints, COVER_RADIUS)
    multiplicity = {int(a): int(b) for a, b in zip(*np.unique(cnt, return_counts=True))}
    single = multiplicity.get(1, 0) / len(pts)
    multi = sum(v for kk, v in multiplicity.items() if kk >= 2) / len(pts)

    order = nearest_order(plan.waypoints)
    length = path_length(plan.waypoints, order)
    ana = analytic_worst(d)
    if abs(worst - ana) > 1e-6:
        raise AssertionError(f"解析最坏距离 {ana:.6f} m 与实算 {worst:.6f} m 不一致")
    if worst > COVER_RADIUS:
        raise AssertionError(f"覆盖保证被破坏：最坏距离 {worst:.1f} m > {COVER_RADIUS:.0f} m")
    if abs(length - 6.0 * d) > 1e-6:
        raise AssertionError(f"巡视里程 {length:.3f} m 与解析值 6d = {6.0 * d:.3f} m 不一致")

    interval = feasible_ring_interval()
    # 参考文献（赵一骁 2024）的紧贴六边形栅格：相邻圆心间距 √3·r，即环半径 = √3·r
    lattice_d = math.sqrt(3.0) * COVER_RADIUS
    lattice = layout_metrics(lattice_d, pts)

    six_d, six_worst = _six_circle_best()
    return CoverSolveResult(
        plan=plan, worst_distance=worst, worst_point=worst_point, analytic_worst=ana,
        coverage_ratio=float((cnt >= 1).mean()), multiplicity=multiplicity,
        single_ratio=single, multi_ratio=multi, mean_multiplicity=float(cnt.mean()),
        survey_order=order, survey_length=length,
        feasible_interval=interval, tradeoff=tradeoff_table(interval, pts),
        lattice=lattice, six_circle_worst=six_worst, six_circle_radius=six_d,
    )


def print_cover_report(res: CoverSolveResult) -> None:
    """打印覆盖圆求解报告（圆心位置、覆盖校验、最少个数、可行区间与权衡、文献对照）。"""
    plan = res.plan
    d = plan.ring_radius
    is_optimal = abs(d - optimal_ring_radius()) < 1e-6
    print("=" * 78)
    print("一、1000 m 覆盖圆的位置")
    print("=" * 78)
    print(f"目标圆域半径 R = {plan.region_radius:.0f} m，覆盖圆半径 r = {plan.cover_radius:.0f} m"
          f"（= 有效接收半径下界），共 {len(plan.waypoints)} 个覆盖圆")
    print(f"布局：1 个中心圆（圆心在原点）+ 6 个环上圆（正六边形顶点）")
    print(f"环半径 d = {d:.2f} m（正六边形边长 = d）"
          + ("（= 最优环半径 d* = √3·R/2 = 900√3）" if is_optimal else "（由 --ring-radius 指定）"))
    print()
    print(f"{'序号':<6}{'类型':<10}{'x / m':>12}{'y / m':>12}")
    for i, (x, y) in enumerate(plan.centers):
        print(f"{i:<6}{'中心圆' if i == 0 else '环上圆':<10}{x:>12.2f}{y:>12.2f}")
    print()
    print("=" * 78)
    print(f"二、覆盖校验（细网格 {GRID_STEP:.0f} m + 圆边界 {BOUNDARY_SAMPLES} 点 + 解析最坏点候选）")
    print("=" * 78)
    print(f"解析最坏最近距离 D(d) = max(d/√3, g₂) = {res.analytic_worst:.3f} m"
          + ("（= R/2；内圈项 d/√3 与边界项 g₂ 在该点相等，即最优性条件）" if is_optimal else ""))
    print(f"实算最坏最近距离 = {res.worst_distance:.3f} m @ "
          f"({res.worst_point[0]:.1f}, {res.worst_point[1]:.1f})"
          f"；另一族最坏点在 ρ = d/√3 = {d / math.sqrt(3):.1f} m 的角平分线方向上")
    print(f"对覆盖半径 1000 m 的余量 = {res.margin:.3f} m，"
          f"采样点被覆盖比例 = {res.coverage_ratio * 100:.2f}%")
    print(f"覆盖重数分布：" + "、".join(f"{k} 重 {v} 点" for k, v in sorted(res.multiplicity.items())))
    print(f"→ 平均覆盖重数 {res.mean_multiplicity:.3f}（重复率 "
          f"{(res.mean_multiplicity - 1) * 100:.1f}%）；单重覆盖（只有 1 条射线，无法直接交会）占 "
          f"{res.single_ratio * 100:.1f}%，二重及以上占 {res.multi_ratio * 100:.1f}%"
          f"（第二阶段需对单射线频道补测第二视角）")
    print()
    print("=" * 78)
    print("三、为什么是 7 个（最少数）")
    print("=" * 78)
    print(f"6 个覆盖圆：环心须落在距原点 1000 m 内才能盖住原点；最优环半径 "
          f"{res.six_circle_radius:.0f} m 时实算最坏距离 {res.six_circle_worst:.1f} m "
          f"> 1000 m ✗（圆域边缘漏源）")
    print(f"7 个覆盖圆：最坏距离 {res.worst_distance:.1f} m ≤ 1000 m ✓（余量 "
          f"{res.margin:.1f} m）→ 7 个即最少可行个数")
    print()
    print("=" * 78)
    print("四、可行区间与权衡（余量 vs 巡视里程）")
    print("=" * 78)
    lo, hi = res.feasible_interval
    print(f"覆盖保证等价于 D(d) ≤ 1000 m，可行区间 d ∈ [{lo:.2f}, {hi:.2f}] m"
          f"（两端点余量为 0）；d* = {optimal_ring_radius():.2f} m 处余量最大")
    print(f"巡视里程 = 6d（原点→环心 d，再走 5 条六边形边），故 d 越小越省时间")
    print()
    print(f"{'环半径 d / m':>13}{'最坏距离 / m':>13}{'余量 / m':>11}"
          f"{'平均重数':>10}{'里程 / m':>11}{'移动时间 / s':>13}")
    for row in res.tradeoff:
        mark = ""
        if abs(row["ring_radius_m"] - d) < 1e-6:
            mark = "  ← 现用"
        elif abs(row["ring_radius_m"] - optimal_ring_radius()) < 1e-6:
            mark = "  ← d*（余量最大）"
        print(f"{row['ring_radius_m']:>13.2f}{row['computed_worst_m']:>13.2f}"
              f"{row['margin_m']:>11.2f}{row.get('mean_multiplicity', float('nan')):>10.3f}"
              f"{row['survey_length_m']:>11.1f}{row['survey_time_s']:>13.1f}{mark}")
    print()
    lat = res.lattice
    print(f"参考文献紧贴栅格对照（相邻圆心间距 √3·r，即环半径 {lat['ring_radius_m']:.2f} m）："
          f"最坏距离 {lat['computed_worst_m']:.2f} m（余量 {lat['margin_m']:.2f} m）、"
          f"里程 {lat['survey_length_m']:.1f} m")
    print(f"现用 d = {d:.2f} m：最坏距离 {res.worst_distance:.2f} m（余量 {res.margin:.2f} m）、"
          f"里程 {res.survey_length:.1f} m → 覆盖余量与里程同时优于紧贴栅格")
    print()
    print("=" * 78)
    print("五、巡视顺序（确定性最近邻：从原点出发，圆心 0 就在原点）")
    print("=" * 78)
    seq = " → ".join(str(i) for i in res.survey_order)
    print(f"顺序：{seq}")
    print(f"总里程 = {res.survey_length:.1f} m，纯移动时间 = {res.survey_length / 5.0:.1f} s"
          f"（速度 5 m/s）")
    print("=" * 78)


# ----------------------------------------------------------------------------
# 机器狗：依次移动到圆心并扫描测向
# ----------------------------------------------------------------------------
class Obs(NamedTuple):
    """一次 direction 检测：在 (x, y) 处测得频道 channel 的示向度 theta（度）。"""

    channel: int
    x: float
    y: float
    theta: float


class RobotDog:
    """第一阶段机器狗：按顺序走遍 7 个覆盖圆的圆心，在每个圆心扫描测向。

    只做"移动 + 扫描 + 记录"，不做定位与主动清除（第二阶段内容）；唯一的例外是扫描
    过程中若某个频道返回 `near`（源就在 5 m 内），则就地 /clear —— 这是白捡的清除。
    """

    def __init__(self, sim, verbose: bool = True, logfile: Optional[str] = None,
                 episode: int = 0) -> None:
        self.sim = sim
        self.verbose = verbose
        self._logfile = open(logfile, "a", encoding="utf-8") if logfile else None
        self.obs: Dict[int, List[Obs]] = defaultdict(list)
        self.cleared: set = set()
        self.pos = np.zeros(2)
        self.vt = 0.0
        self.n_measure = 0
        self.n_clear = 0
        self.episode = episode
        self.deadline = float("inf")
        self.waypoint_stats: List[Dict[str, Any]] = []      # 逐圆心扫描统计
        self.travel_m = 0.0                                 # 实际走过的里程 / m

    # ---- 日志 ----
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
        if self._logfile:
            print(msg, file=self._logfile, flush=True)

    def close(self) -> None:
        """关闭日志文件（幂等）。"""
        if self._logfile:
            self._logfile.close()
            self._logfile = None

    # ---- 原子动作 ----
    def measure(self, x: float, y: float, channel: int) -> dict:
        """测向；direction 时自动记录示向度。"""
        x, y = float(x), float(y)
        r = self.sim.measure(x, y, channel)
        if not r.get("accepted"):
            raise RuntimeError(f"/measure 被拒绝：{r}")
        self.travel_m += dist(self.pos, (x, y))
        self.pos, self.vt = np.array([x, y]), float(r["virtual_time_s"])
        self.n_measure += 1
        if r.get("measure_result") == "direction":
            self.obs[channel].append(Obs(channel, x, y, float(r["svd_deg"])))
        return r

    def clear(self, x: float, y: float, channel: int) -> bool:
        """清除；返回是否成功。"""
        x, y = float(x), float(y)
        r = self.sim.clear(x, y, channel)
        if not r.get("accepted"):
            raise RuntimeError(f"/clear 被拒绝：{r}")
        self.travel_m += dist(self.pos, (x, y))
        self.pos, self.vt = np.array([x, y]), float(r["virtual_time_s"])
        self.n_clear += 1
        ok = r.get("clear_result") == "success"
        if ok:
            self.cleared.add(channel)
        return ok

    def _out_of_time(self) -> bool:
        return time.monotonic() > self.deadline

    # ---- 扫描 ----
    def _active_channels(self) -> List[int]:
        """仍需测向的频道：未清除、且示向度条数未达上限。"""
        return [c for c in CHANNELS
                if c not in self.cleared and len(self.obs.get(c, ())) < OBS_CAP]

    def _sweep(self, channels: Sequence[int], at: Sequence[float]) -> Dict[str, int]:
        """在 at 处按频道号升序逐频道测向（升序可省频道切换时间）；near 就地清除。"""
        counts = {"direction": 0, "near": 0, "no_signal": 0}
        for ch in sorted(channels):
            if self._out_of_time():
                break
            res = self.measure(at[0], at[1], ch).get("measure_result", "no_signal")
            counts[res] = counts.get(res, 0) + 1
            if res == "near":
                self.log(f"    [near] 频道{ch}：源在 5 m 内，就地清除"
                         f"{'成功' if self.clear(at[0], at[1], ch) else '失败'}")
        return counts

    def survey(self, waypoints: np.ndarray, order: Sequence[int]) -> None:
        """依次移动到各圆心并扫描：这是本阶段的主体。"""
        self.log(f"巡视扫描：依次访问 {len(order)} 个圆心"
                 f"（顺序 {' → '.join(str(i) for i in order)}）")
        for step_i, idx in enumerate(order, 1):
            wp = waypoints[idx]
            active = self._active_channels()
            if not active:
                self.log(f"  第 {step_i} 站：圆心 {idx} @ ({wp[0]:.1f}, {wp[1]:.1f})，"
                         f"所有频道已采够或已清除，巡视提前结束")
                break
            if self._out_of_time():
                self.log(f"  第 {step_i} 站：现实时间不足，巡视提前结束")
                break
            self.log(f"  第 {step_i} 站：圆心 {idx} @ ({wp[0]:.1f}, {wp[1]:.1f})，"
                     f"扫描 {len(active)} 个频道")
            counts = self._sweep(active, wp)
            self.waypoint_stats.append({
                "waypoint": int(idx), "x": float(wp[0]), "y": float(wp[1]),
                "n_channels": len(active), **counts,
                "virtual_time_s": round(self.vt, 3), "travel_m": round(self.travel_m, 2),
            })
            self.log(f"    有示向度 {counts['direction']}、无信号 {counts['no_signal']}、"
                     f"近距清除 {counts['near']}，累计里程 {self.travel_m:.0f} m，"
                     f"虚拟时刻 {self.vt:.0f} s")

    def run(self, plan: CoverPlan, order: Sequence[int]) -> Dict[str, Any]:
        """/enter → 依次到各圆心扫描 → /exit。"""
        enter = self.sim.enter()
        if not enter.get("accepted"):
            raise RuntimeError(f"/enter 被拒绝：{enter}")
        left = float(enter.get("remaining_real_duration_s", 1200.0))
        self.deadline = time.monotonic() + max(left - SAFETY_MARGIN, 0.0)
        self.log(f"/enter 成功：虚拟时刻 {enter.get('virtual_time_s')} s，"
                 f"现实剩余 {left:.0f} s（本阶段仅巡视扫描，不做定位与清除）")
        try:
            self.survey(plan.waypoints, order)
        finally:
            try:
                self.sim.exit()
            except OSError as exc:
                self.log(f"    [警告] /exit 失败：{exc}")
            self.close()
        return {
            "waypoints_visited": len(self.waypoint_stats),
            "travel_m": self.travel_m,
            "virtual_time_s": self.vt,
            "n_measure": self.n_measure,
            "n_clear": self.n_clear,
            "channels_heard": len(self.obs),
            "n_bearings": sum(len(v) for v in self.obs.values()),
        }


# ----------------------------------------------------------------------------
# 覆盖保证的真值核对（只有演练模式拿得到真值）
# ----------------------------------------------------------------------------
def truth_check(truth: Optional[Sequence[dict]], plan: CoverPlan,
                obs: Dict[int, List[Obs]], cleared: set) -> Dict[str, Any]:
    """逐源核对"不漏源"：到最近圆心的距离是否 ≤ 1000 m、是否真的听到了它的频道。"""
    rows: List[Dict[str, Any]] = []
    for j in truth or []:
        p = (float(j["position"]["x"]), float(j["position"]["y"]))
        d_near = float(np.linalg.norm(plan.waypoints - np.asarray(p), axis=1).min())
        heard = len(obs.get(j["channel"], ())) > 0 or j["channel"] in cleared
        rows.append({"channel": int(j["channel"]), "x": p[0], "y": p[1],
                     "receive_m": float(j["receive"]),
                     "nearest_center_m": round(d_near, 2),
                     "heard": bool(heard),
                     "n_bearings": len(obs.get(j["channel"], ()))})
    return {
        "sources": rows,
        "worst_nearest_m": round(max((r["nearest_center_m"] for r in rows), default=0.0), 2),
        "all_within_cover": all(r["nearest_center_m"] <= COVER_RADIUS for r in rows),
        "missed_channels": [r["channel"] for r in rows if not r["heard"]],
    }


# ----------------------------------------------------------------------------
# 本地演练场：拉起 jammers-py 并用其控制台 REST 开一局（与 T3_ga.py 同源实现）
# ----------------------------------------------------------------------------
def _free_port(preferred: int) -> int:
    """取一个可用端口：优先 preferred，被占用则依次向后试 20 个。"""
    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"端口 {preferred}~{preferred + 19} 都被占用")


class PracticeArena:
    """本地演练用的 jammers-py 模拟器：自动拉起进程（或复用已在运行的实例）+ 控制台 REST。"""

    REUSE_PORTS = (8090, 8080)

    def __init__(self, jammers_dir: Path, robot_id: str, robot_port: int = 2026,
                 console_port: int = 8090, countdown: int = 1) -> None:
        self.jammers_dir = Path(jammers_dir).resolve()
        self.robot_id = robot_id
        self.robot_port = robot_port
        self.console_port = console_port
        self.countdown = countdown
        self.robot_url = f"http://127.0.0.1:{robot_port}"
        self.console_url = f"http://127.0.0.1:{console_port}"
        self._proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "PracticeArena":
        existing = self._find_existing()
        if existing is not None:
            self.console_url, state = existing
            self.robot_port = int(state.get("config", {}).get("robot_port", self.robot_port))
            self.robot_url = f"http://127.0.0.1:{self.robot_port}"
            print(f"检测到已在运行的 jammers-py（控制台 {self.console_url}），直接复用")
            return self
        run_py = self.jammers_dir / "run.py"
        if not run_py.exists():
            raise FileNotFoundError(f"未找到 jammers-py 模拟器：{run_py}（用 --jammers-dir 指定）")
        self.console_port = _free_port(self.console_port)
        self.console_url = f"http://127.0.0.1:{self.console_port}"
        self._proc = subprocess.Popen(
            [sys.executable, str(run_py),
             "--robot-port", str(self.robot_port), "--web-port", str(self.console_port),
             "--countdown", str(self.countdown), "--team", self.robot_id],
            cwd=str(self.jammers_dir), stdout=subprocess.DEVNULL)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if self._state(self.console_url) is not None:
                return self
            if self._proc.poll() is not None:
                raise RuntimeError(f"jammers-py 启动失败：机器狗接口 {self.robot_port} 或控制台 "
                                   f"{self.console_port} 端口被占用")
            time.sleep(0.2)
        raise TimeoutError("等待 jammers-py 控制台就绪超时")

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    # ---- 控制台 REST ----
    def _request(self, path: str, payload: Optional[dict] = None, base: Optional[str] = None,
                 post: bool = False) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request((base or self.console_url) + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if post or payload is not None else "GET")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _state(self, base: str) -> Optional[dict]:
        try:
            state = self._request("/api/state", base=base)
            return state if "state" in state else None
        except Exception:
            return None

    def _find_existing(self) -> Optional[Tuple[str, dict]]:
        for port in dict.fromkeys((self.console_port, *self.REUSE_PORTS)):
            base = f"http://127.0.0.1:{port}"
            state = self._state(base)
            if state is not None:
                return base, state
        return None

    def _wait_state(self, target: str, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self._request("/api/state").get("state")
            if state == target:
                return
            if state == "finished":
                raise RuntimeError(f"演练局提前结束（期望状态 {target}）")
            time.sleep(0.2)
        raise TimeoutError(f"等待状态 {target} 超时")

    # ---- 一局演练 ----
    def start_episode(self, seed: int) -> List[dict]:
        """按种子生成固定场景并开一局，等接口开放后返回干扰源真值。

        覆盖保证的核对需要"整局完全可复现"：jammers-py 只把 seed 用于干扰源布局，示向度噪声
        种子 `noise_seed_hex` 每次随机 —— 这里把它改写成由 seed 派生的确定值，于是同一个
        --seed 必然得到同一份扫描结果，新旧策略也可严格对照。
        """
        scenario = self._request("/api/scenario", {"problem_no": 3, "seed": seed})["scenario"]
        scenario["noise_seed_hex"] = hashlib.blake2b(f"t3-practice-{seed}".encode(),
                                                    digest_size=8).hexdigest()
        self._request("/api/start", {"problem_no": 3, "scenario": scenario})
        self._wait_state("window_open")
        return scenario["jammers"]

    def finish_episode(self) -> dict:
        """收掉本局并返回模拟器引擎统计（真值清除数、虚拟时刻、检测次数）。"""
        snap = self._request("/api/state")
        engine = snap.get("engine") or {}
        if snap.get("state") != "finished":
            self._request("/api/abort", post=True)
        self._request("/api/clear", post=True)
        return engine


# ----------------------------------------------------------------------------
# 结果落盘
# ----------------------------------------------------------------------------
def save_plan(res: CoverSolveResult, save_dir: Path) -> List[Path]:
    """覆盖圆方案落盘：JSON（含校验与对照）+ CSV（圆心坐标）。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / PLAN_JSON
    json_path.write_text(json.dumps(res.to_json(), ensure_ascii=False, indent=2),
                         encoding="utf-8")
    csv_path = save_dir / PLAN_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "x_m", "y_m", "ring_radius_m", "cover_radius_m"])
        for i, (x, y) in enumerate(res.plan.centers):
            w.writerow([i, "center" if i == 0 else "ring", f"{x:.3f}", f"{y:.3f}",
                        f"{res.plan.ring_radius:.3f}", f"{res.plan.cover_radius:.1f}"])
    return [json_path, csv_path]


def save_survey(save_dir: Path, rows: List[dict], observations: List[dict],
                plan_json: Dict[str, Any]) -> List[Path]:
    """巡视扫描结果落盘：逐局统计 JSON + 逐条观测 CSV。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / SURVEY_JSON
    json_path.write_text(json.dumps({
        "stage": "第一阶段：1000 m 覆盖圆求解 + 依次到圆心巡视扫描",
        "cover_plan": plan_json,
        "episodes": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = save_dir / OBS_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "channel", "x_m", "y_m", "svd_deg", "nearest_center_m"])
        for o in observations:
            w.writerow([o["episode"], o["channel"], f"{o['x']:.2f}", f"{o['y']:.2f}",
                        f"{o['theta']:.2f}", f"{o['nearest_center_m']:.2f}"])
    return [json_path, csv_path]


def episode_row(ep: int, seed: int, truth: Optional[Sequence[dict]], dog: RobotDog,
                stats: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
    """单局汇总行。"""
    return {
        "episode": ep, "seed": seed,
        "n_sources": len(truth) if truth else None,
        **stats,
        "waypoint_stats": dog.waypoint_stats,
        "truth_check": check,
        "bearings_per_channel": {str(c): len(v) for c, v in sorted(dog.obs.items())},
        "cleared_channels": sorted(dog.cleared),
    }


# ----------------------------------------------------------------------------
# 两种运行模式
# ----------------------------------------------------------------------------
def _observations(ep: int, plan: CoverPlan, obs: Dict[int, List[Obs]]) -> List[dict]:
    """把本局观测整理成 CSV 行（附"观测点到最近圆心的距离"，便于核对覆盖保证）。"""
    rows = []
    for ch, ol in sorted(obs.items()):
        for o in ol:
            d = float(np.linalg.norm(plan.waypoints - np.array([o.x, o.y]), axis=1).min())
            rows.append({"episode": ep, "channel": ch, "x": o.x, "y": o.y,
                         "theta": o.theta, "nearest_center_m": d})
    return rows


def run_practice(args: argparse.Namespace, res: CoverSolveResult,
                 save_dir: Path) -> int:
    """本地演练：自动拉起 jammers-py，跑 N 局巡视扫描，核对覆盖保证并汇总。"""
    jammers_dir = (Path(args.jammers_dir) if args.jammers_dir
                   else Path(__file__).parent / "jammers-py")
    rows: List[dict] = []
    observations: List[dict] = []
    with PracticeArena(jammers_dir, robot_id=args.robot_id,
                       console_port=args.console_port) as arena:
        print(f"jammers-py 已就绪：机器狗接口 {arena.robot_url}，控制台 {arena.console_url}")
        for ep in range(args.practice):
            seed = args.seed + ep
            truth = arena.start_episode(seed)
            print(f"\n----- 演练第 {ep + 1}/{args.practice} 局（seed={seed}，"
                  f"干扰源 {len(truth)} 个）-----")
            dog = RobotDog(sim_api.Simulator(robot_id=args.robot_id, base_url=arena.robot_url,
                                             timeout=args.timeout),
                           verbose=not args.quiet, logfile=args.log, episode=ep + 1)
            stats = dog.run(res.plan, res.survey_order)
            arena.finish_episode()
            check = truth_check(truth, res.plan, dog.obs, dog.cleared)
            rows.append(episode_row(ep + 1, seed, truth, dog, stats, check))
            observations.extend(_observations(ep + 1, res.plan, dog.obs))
            print(f"本局：巡视 {stats['waypoints_visited']} 个圆心，里程 {stats['travel_m']:.0f} m，"
                  f"虚拟时间 {stats['virtual_time_s']:.0f} s，测向 {stats['n_measure']} 次；"
                  f"听到 {stats['channels_heard']}/{len(truth)} 个源"
                  f"（{stats['n_bearings']} 条示向度），顺手近距清除 {stats['n_clear']} 个")
            print(f"  覆盖核对：源到最近圆心最坏距离 {check['worst_nearest_m']:.1f} m ≤ "
                  f"{COVER_RADIUS:.0f} m（{check['all_within_cover']}），"
                  f"漏听 {check['missed_channels'] or '无'}")
    print("\n" + "=" * 78)
    print(f"汇总（{len(rows)} 局）：平均里程 "
          f"{np.mean([r['travel_m'] for r in rows]):.0f} m，平均虚拟时间 "
          f"{np.mean([r['virtual_time_s'] for r in rows]):.0f} s，平均测向 "
          f"{np.mean([r['n_measure'] for r in rows]):.0f} 次，"
          f"共听到 {sum(r['channels_heard'] for r in rows)}/"
          f"{sum(r['n_sources'] for r in rows)} 个源")
    print("逐局：" + "  ".join(f"seed{r['seed']}={r['channels_heard']}/{r['n_sources']}"
                              for r in rows))
    print("=" * 78)
    plan_json = res.to_json()
    paths = save_plan(res, save_dir) + save_survey(save_dir, rows, observations, plan_json)
    print("结果已保存：" + "，".join(str(p) for p in paths))
    return 0


def run_official(args: argparse.Namespace, res: CoverSolveResult, save_dir: Path) -> int:
    """官方评测接口模式：连 127.0.0.1 上已开放接口的模拟器，跑一局巡视扫描。

    注意：本阶段只巡视扫描、不做定位与清除，因此**不会清除干扰源**，仅用于在官方引擎上
    验证覆盖圆与扫描流程；正式测试请等第二阶段（定位 + 清除）完成后再跑。
    """
    sim = sim_api.Simulator(robot_id=args.robot_id, base_url=args.base_url, timeout=args.timeout)
    print(f"连接模拟器 {args.base_url}（robot_id={args.robot_id}）")
    dog = RobotDog(sim, verbose=not args.quiet, logfile=args.log, episode=1)
    stats = dog.run(res.plan, res.survey_order)
    print(f"完成：巡视 {stats['waypoints_visited']} 个圆心，里程 {stats['travel_m']:.0f} m，"
          f"虚拟时间 {stats['virtual_time_s']:.0f} s，测向 {stats['n_measure']} 次，"
          f"听到 {stats['channels_heard']} 个频道（{stats['n_bearings']} 条示向度）")
    row = episode_row(1, args.seed, None, dog, stats,
                      truth_check(None, res.plan, dog.obs, dog.cleared))
    paths = (save_plan(res, save_dir)
             + save_survey(save_dir, [row], _observations(1, res.plan, dog.obs), res.to_json()))
    print("结果已保存：" + "，".join(str(p) for p in paths))
    return 0


# ----------------------------------------------------------------------------
# 命令行
# ----------------------------------------------------------------------------
def _relax_console_encoding() -> None:
    """放宽控制台编码错误处理，避免个别字符（如 √、≤）在 GBK 控制台上中断运行。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2026 CUMCM B 题问题三（第一阶段）：1000 m 覆盖圆求解 + 依次到圆心巡视扫描")
    p.add_argument("--practice", type=int, nargs="?", const=1, default=0,
                   help="本地演练局数：自动拉起 jammers-py 跑 N 局（缺省 1 局）")
    p.add_argument("--base-url", default=None,
                   help="官方模拟器地址（赛期用，如 http://127.0.0.1:2026）；缺省只求解覆盖圆")
    p.add_argument("--robot-id", default=sim_api.ROBOT_ID, help="参赛队号（须与模拟器一致）")
    p.add_argument("--timeout", type=float, default=5.0, help="HTTP 超时 / s")
    p.add_argument("--ring-radius", type=float, default=None,
                   help="覆盖圆环半径 d / m（缺省取余量最大的 d* = √3·R/2 = 1558.85；"
                        "取小值可缩短巡视里程，但覆盖余量变小）")
    p.add_argument("--jammers-dir", default=None,
                   help="jammers-py 目录（缺省为本脚本旁的 jammers-py/）")
    p.add_argument("--console-port", type=int, default=8090,
                   help="演练时 jammers-py 控制台端口（缺省 8090，被占用则自动顺延）")
    p.add_argument("--seed", type=int, default=SEED,
                   help="演练第 1 局的种子（场景布局与示向度噪声都由它确定，可复现）")
    p.add_argument("--save-dir", default=RESULTS_DIR, help="结果输出目录（缺省 results/）")
    p.add_argument("--log", default=None, help="过程日志文件（逐站扫描的文字过程，追加写入）")
    p.add_argument("--quiet", action="store_true", help="只输出汇总，不打印过程")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    _relax_console_encoding()
    args = build_parser().parse_args(argv)
    save_dir = Path(args.save_dir)

    res = solve_covering_circles(args.ring_radius)   # 第一步：求 1000 m 覆盖圆的位置
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
