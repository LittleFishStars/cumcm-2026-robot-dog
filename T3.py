#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T3.py —— 2026 高教社杯全国大学生数学建模竞赛 B 题 · 问题三

覆盖圆求解 → 依次到圆心巡视扫描 → 交会定位 → 先清直径 < 40 m 的 → 其余按文献准则补测缩小后清除
==========================================================================================

问题三要求机器狗从原点出发，在半径 1800 m 的目标圆域内自动定位并清除 10~16 个全向干扰源
（个数未知）。本程序的策略分两阶段，全程确定性（无随机搜索、无遗传算法）：

  阶段一 巡视扫描：先用尽量少的半径 1000 m 的"覆盖圆"铺满目标圆域，求出圆心位置，再依次
          走到每个圆心扫描 20 个频道，把每个源的示向度采集齐全；
  阶段二 定位与清除：用问题 1 的交会定位区域（各 ±1° 楔形之交 ∩ 圆域）判断每个源的定位
          精度 —— **定位区域直径 < 40 m 的直接清掉**（此时区域的最小覆盖圆半径 ≤ 20 m，
          走到圆心必中），其余频道按文献准则选点补测，把区域缩小到 40 m 以内再清。

第一部分：为什么覆盖圆半径取 1000 m
-----------------------------------
题目给出每个干扰源的有效接收半径在 1000~1500 m 之间，即"检测点距源不超过 1000 m 就一定
听得到"。因此只要一组半径 1000 m 的圆**完全覆盖**目标圆域，机器狗依次走遍这些圆的圆心并
在每个圆心停下测向，就必然不会漏掉任何干扰源 —— 这就是本策略"不漏源"的全部依据，不需要
对源的位置分布做任何先验假设。

覆盖圆的位置（解析解）
----------------------
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
无法直接交会定距；约 34% 被 2 个圆覆盖、约 3% 被 3 个圆覆盖，可直接交会。

环半径的权衡与可行区间
----------------------
覆盖保证只要求 D(d) ≤ 1000 m，由此得到可行区间（两端点均为零余量）：

    内圈约束：d/√3 ≤ r  →  d ≤ √3·r = 1732.05 m
    边界约束：g₂(d) ≤ r →  d ∈ [1122.96, 1994.74] m
    合起来：  d ∈ [1122.96, 1732.05] m

而巡视里程恰为 6d（原点 → 一个环心 = d，再沿正六边形走 5 条边 = 5d），随 d 单调增。于是
"余量最大"与"里程最短"是一对矛盾：d* = 1558.85 m 使余量最大（100 m）、里程 9353 m；
若取 d = 1200 m，里程 7200 m（省 2153 m ≈ 431 s），余量仍有 31.1 m（仍可证明不漏源）。
本程序默认取 d*（最稳），可用 --ring-radius 指定其他值，报告里给出权衡表供选择。

第二部分：定位与清除（阶段二）
-----------------------------
**定位区域。** 直接复用问题 1 的 `TriangulationRegion`：检测点处 ±1° 的示向度误差使干扰源必
落在以该点为顶点、张角 2° 的楔形内，故"定位区域 = 所有楔形之交 ∩ 目标圆域"，它是凸多边形；
区域直径按问题 1 的定义取区域内任意两点距离的最大值。区域的最小覆盖圆半径 r ≤ 直径/2，且
圆心落在区域内（凸集的最小覆盖圆圆心必在凸包内），因此

    定位区域直径 < 40 m  ⇒  清除点（最小覆盖圆圆心）到区域内任意点（含真实干扰源）< 20 m
                        ⇒  直接走过去 /clear 必然命中

这就是"先清直径 < 40 m 的"这一判据的来源（40 m = 2 × 清除半径 20 m）。求交时圆域用内接 256
边形（与真圆的偏差 0.14 m），故实际判据取 `直径 + 2 × 0.14 < 40`，保证的是真圆域上的直径。

**补测选点（文献方法）。** 对直径仍 ≥ 40 m 的频道，在候选点中选"最能让定位区域变小"的位置去
补测一次。评价用测向的 Fisher 信息矩阵（1° 误差）：

    J = Σᵢ (1/σ²) · (1/rᵢ²) · nᵢnᵢᵀ ,   σ_pos = √tr(J⁻¹)

其中 rᵢ 为假设源位置到第 i 个检测点的距离；σ 是常数因子，不影响候选排序。该准则同时实现了
文献的两条结论：

* 任叶童《基于到达角信息的无源定位算法研究》（天津大学硕士论文, 2016）：多站交会的极大似然
  估计等价于以 1/(σᵢRᵢ) 为权的加权最小二乘（最小化到各示向度线的垂距加权平方和），本程序
  的 J 与之一致；该文还给出定位模糊区面积 S = 4R²Δθ²sinα₁sinα₂/sin³(α₁+α₂)（基线 R 固定、
  目标位置自由的口径），代码里保留 ambiguity_area() 以便对照。
* Chen X, Xu Z, Rui L. An optimization algorithm of multi-observer trajectories for
  cooperative bearings-only target localization[C]//ICICS 2009：滤波均方位置误差与观测站到
  目标的距离呈平方反比（∝ 1/(σ²r²)），最优观测轨迹是"贴近目标"与"拉开角度分集"的折衷。
  在 J 的口径下即：**距离越近、与已有射线的交角越接近正交，预测 σ 越小**（两观测等距 r 时
  tr(J⁻¹) = 2σ²r²/sin²φ，交角 φ = 90° 最优）。

候选点取每个假设源位置周围若干半径（150~800 m，保证落在源的有效接收半径内）× 24 个方位角
（15° 一格）的环上点；共线候选（与已有射线夹角 ≈ 0，σ = ∞，无法定距）直接剔除；对全部假设
源位置求平均 σ 后排序，并在 σ 与最优值相差 2% 以内的候选里取里程最短者。单射线频道的假设
源位置沿该射线按 200~1400 m 枚举（源的距离未知，只能枚举假设）。

**先清直径 < 40 m 的，其余再补测。** 流程按此执行，并加了两条只花几秒的省时捷径：
① 区域有界且估计不太差（最小覆盖圆半径 ≤ 400 m）时，先走到区域中心试清一次（失败只花 3 s，
   比绕路补测约 120 s 便宜），未命中时就地复测 —— 这正是"距离最近、权重最大"的观测点；
② 万一仍未清除，沿最新实测示向度以 16 m 步长逼近（步数按到区域最远顶点的距离自适应）。
10 局演练中 131 个源全部清除，其中"严格判据直接清"60 个、"就近试清"71 个，无一次需要兜底。

与参考文献（覆盖圆部分）的关系
------------------------------
赵一骁, 何航天, 李雨楠, 等. 基于最小圆覆盖的多无人机协同螺旋式搜索优化算法[J]. 指挥信息
系统与技术, 2024, 15(4): 56-62：提出"圆覆盖 + 圆内接正六边形拼接"的环境建模，以传感器搜索
半径 r 为覆盖圆，相邻圆心间距取紧贴的 √3·r（两圆重叠 5.77%、利用率 94.23%），覆盖圆圆心即
航路关键点，只需遍历全部关键点就完成了区域覆盖搜索。本题覆盖圆的个数与结构与之一致（都是
"1 中心 + 6 环"共 7 个），但环半径由紧贴间距 √3·r = 1732.05 m 改进为 √3·R/2 = 1558.85 m：
最坏最近距离由 1000.0 m（零余量）降到 900.0 m（余量 100 m），巡视里程同时由 10392 m 降到
9353 m，两项同时更优。差别来自边界条件：紧贴间距是"无限平面"最小圆覆盖问题的解，本题是
有界圆域，最坏点由内圈与圆域边界两处共同决定（见上节）。

运行
====
    python T3.py                                    # 只求解覆盖圆并打印报告（不连模拟器）
    python T3.py --ring-radius 1200                 # 改用其他环半径（里程更短、余量更小）
    python T3.py --practice 10 --seed 0             # 本地演练 10 局（自动拉起 jammers-py）
    python T3.py --practice 3 --survey-only         # 只做阶段一（巡视扫描 + 覆盖核对）
    python T3.py --base-url http://127.0.0.1:2026   # 官方评测接口模式（赛期，先开模拟器）

结果落盘（--save-dir，缺省 results/，文件名固定便于论文与绘图引用）
    t3_cover_plan.json    覆盖圆求解结果（环半径、圆心、最坏距离、覆盖重数、巡视顺序、权衡表）
    t3_cover_circles.csv  7 个覆盖圆的圆心坐标（序号、类型、x、y）
    t3_survey.json        逐局统计 + 逐源真值核对 + 逐频道定位/清除档案 + 整批汇总
    t3_observations.csv   逐条示向度观测（局号、频道、阶段、检测点坐标、示向度）

依赖：numpy（覆盖校验）、shapely（定位区域，经 T1.py）、matplotlib 不需要；
HTTP 层复用同目录 sim_api.py；本地演练用同目录 jammers-py/（纯标准库）。
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
from T1 import TriangulationRegion      # 问题 1 的交会定位区域（楔形交 ∩ 圆域）

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

# ---- 第二阶段常量：定位区域、清除判据、补测选点准则（文献方法）----
BEARING_ERROR_DEG = 1.0         # 示向度误差半宽 / 度（题目给定 |误差| ≤ 1°，楔形张角 2°）
SIGMA_DEG = BEARING_ERROR_DEG   # 同一误差的量级（Fisher 信息里的 σ）/ 度
SIGMA_RAD = math.radians(SIGMA_DEG)
CLEAR_RADIUS = 20.0             # 清除半径 / m（光学精确定位要求 ≤ 20 m）
NEAR_RADIUS = 5.0               # 近距阈值 / m（≤ 5 m 可跳过测向直接清除）
CLIP_SIDES = 256                # 定位区域求交时目标圆域的内接多边形边数
CLIP_ERR = REGION_RADIUS * (1.0 - math.cos(math.pi / CLIP_SIDES))   # 内接多边形与真圆的偏差 / m
DIAM_CLEAR = 2.0 * CLEAR_RADIUS  # 直径判据：< 40 m 则区域最小覆盖圆半径 ≤ 20 m，可直接清
PROBE_RADII = (150.0, 300.0, 450.0, 600.0, 800.0)   # 补测候选点到假设源位置的距离 / m
PROBE_ANGLES = 24               # 补测候选点的方位角格数（15° 一格）
PROBE_TRY = 3                   # 每轮补测最多试几个候选点（收不到信号就换下一个）
HYP_MAX = 8                     # 假设源位置最多取几个（最坏情形稳健）
HYP_GAP = 50.0                  # 假设点与已有检测点的最小间距 / m（太近无法估计距离）
PROBE_GAP = 60.0                # 补测点与已有检测点的最小间距 / m（同点复测不提供新信息）
SINGLE_HYP = (200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0)   # 单射线时沿射线的假设距离
REFINE_MAX = 6                  # 每个频道最多补测几轮
TRY_CLEAR_RADIUS = 400.0        # 就近试清的前提：最小覆盖圆半径 ≤ 该值（区域有界且估计不太差）
HOMING_STEP = 16.0              # 末端沿最新示向度逼近的步长 / m
HOMING_MAX = 24                 # 末端沿示向度逼近的最少迭代次数
HOMING_CAP = 120                # 末端逼近的迭代上限（离得远时按距离自适应加长，但不超过此值）


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
# 第二阶段工具：定位区域、Fisher 信息、补测选点准则（文献方法）
# ----------------------------------------------------------------------------
class Obs(NamedTuple):
    """一次 direction 检测：在 (x, y) 处测得频道 channel 的示向度 theta（度）。

    stage 记录该次观测来自哪个阶段（survey = 巡视扫描，refine = 文献准则补测/试清复测，
    clear = 清除阶段的复测）。
    """

    channel: int
    x: float
    y: float
    theta: float
    stage: str = "survey"


def fisher_sigma(p: Sequence[float], bearings: Sequence[Sequence[float]]) -> float:
    """在假设源位置 p 处、由一组 (检测点x, 检测点y, 示向度) 预测的位置 1σ / m。

    测向的观测方程 θ = atan2(Δy, Δx) + e，e 的标准差取 σ = 1°。梯度 ∂θ/∂p = n / r
    （n 为示向度方向的单位法向量，r 为检测点到源的距离），故 Fisher 信息矩阵

        J = Σᵢ (1/σ²) · (1/rᵢ²) · nᵢ nᵢᵀ,      预测协方差 C = J⁻¹,   σ_pos = √tr(C)

    权重 1/(σᵢ²rᵢ²) 与文献一致：任叶童（2016）多站交会的加权最小二乘权为 1/(σᵢRᵢ)；
    Chen 等（2009）证明观测站到目标的距离越近、均方位置误差越小（∝ 1/(σ²r²)）。
    J 退化（单条射线、近共线）时行列式 ≈ 0，返回 inf，表示"无法定距"。
    """
    J = np.zeros((2, 2))
    for qx, qy, theta in bearings:
        r = math.hypot(p[0] - qx, p[1] - qy)
        if r < 1.0:                     # 源几乎落在检测点上，距离不可估
            return math.inf
        t = math.radians(theta)
        n = np.array([-math.sin(t), math.cos(t)])
        J += np.outer(n, n) / (SIGMA_RAD ** 2 * r * r)
    det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    if det <= 1e-30:
        return math.inf
    return math.sqrt((J[0, 0] + J[1, 1]) / det)     # tr(J⁻¹) = (J₁₁ + J₀₀)/det


def hypothesis_points(region, obs_list: Sequence[Obs],
                      start: Sequence[float] = (0.0, 0.0)) -> List[Tuple[float, float]]:
    """补测选点用的"假设源位置"集合。

    源必落在定位区域内，故用区域内散布的点做最坏情形假设（对全部假设都好的补测点才选）：
    * 多条射线：取最小覆盖圆圆心 + 区域内最远点采样出的若干顶点；
    * 单条射线：区域是一条带状楔形，沿射线按若干可能距离取点（源距离未知，只能枚举）。
    """
    pts: List[Tuple[float, float]] = []
    if len(obs_list) < 2:
        o = obs_list[0]
        for s in SINGLE_HYP:
            hx = o.x + s * math.cos(math.radians(o.theta))
            hy = o.y + s * math.sin(math.radians(o.theta))
            if math.hypot(hx, hy) <= REGION_RADIUS:
                pts.append((hx, hy))
    else:
        mec = region.enclosing_circle
        if mec is not None:
            pts.append((mec[0], mec[1]))
        verts = list(region.vertices)
        while len(pts) < HYP_MAX and verts:      # 最远点采样：始终补"离已选点最远"的顶点
            far = max(verts, key=lambda v: min(dist(v, q) for q in pts))
            verts.remove(far)
            pts.append(far)
    # 与已有检测点太近的假设点无法估计距离（r → 0），剔除；并去重
    keep: List[Tuple[float, float]] = []
    for h in pts:
        if any(dist(h, (o.x, o.y)) < HYP_GAP for o in obs_list):
            continue
        if any(dist(h, k) < 1.0 for k in keep):
            continue
        keep.append(h)
    return keep


class Probe(NamedTuple):
    """一个补测候选点：坐标、对全部假设源位置的最坏预测 σ、从当前位置出发的里程。"""

    x: float
    y: float
    sigma: float
    travel: float


def probe_candidates(obs_list: Sequence[Obs], hyps: Sequence[Sequence[float]],
                     pos: Sequence[float]) -> List[Probe]:
    """按文献准则给补测点排序：最小化"对全部假设源位置的平均预测 σ"。

    候选点 = 每个假设源位置周围若干半径（PROBE_RADII，均 < 1000 m，保证落在源的有效接收
    半径内）× 若干方位角（PROBE_ANGLES）的环上点。评价用 Fisher 信息口径的预测 σ：
    该准则同时实现了文献的两条结论 —— 检测点越接近源 σ 越小（Chen 的定理：均方位置误差
    ∝ 1/(σ²r²)），且新射线与已有射线的交角越接近正交 σ 越小（角度分集）；单射线时
    J 退化，准则自动把补测点放到能"定距"的位置上，即完成单射线→双射线的补测。

    共线候选（与已有射线夹角 ≈ 0，σ = ∞，无法定距）直接剔除 —— 单射线频道的假设点排成一条
    直线，若用"对全部假设取最坏 σ"排序会退化成"所有候选都不可用"，从而按里程误选到共线上的
    点（实测踩过：测了 26 次仍没缩小区域）；取平均 σ 既保留"靠近源 + 拉开交角"的偏好，
    又不会因个别极远假设把好点全部否掉。

    排序规则（确定性）：先按平均 σ 升序；σ 与最优值相差 2% 以内的候选视为"同等好"，
    其中取里程最短者（省时间），里程并列时取坐标字典序最小者。返回前 PROBE_TRY 个候选。
    """
    base = [(o.x, o.y, o.theta) for o in obs_list]      # 已有观测的（检测点, 示向度）
    cands: List[Probe] = []
    seen = set()
    for hx, hy in hyps:
        for rho in PROBE_RADII:
            for k in range(PROBE_ANGLES):
                a = 2.0 * math.pi * k / PROBE_ANGLES
                qx, qy = hx + rho * math.cos(a), hy + rho * math.sin(a)
                if math.hypot(qx, qy) > REGION_RADIUS - REGION_MARGIN:   # 必须在作业圆域内
                    continue
                if any(math.hypot(qx - o.x, qy - o.y) < PROBE_GAP for o in obs_list):
                    continue                    # 离已有检测点太近：同处复测不提供新信息
                key = (round(qx, 1), round(qy, 1))
                if key in seen:
                    continue
                seen.add(key)
                total, ok = 0.0, True
                for h in hyps:                          # 对全部假设源位置求平均 σ（见下）
                    sig = fisher_sigma(h, base + [(qx, qy, bearing((qx, qy), h))])
                    if sig == math.inf:                 # 与已有射线共线：不提供任何新信息
                        ok = False
                        break
                    total += sig
                if ok:
                    cands.append(Probe(qx, qy, total / len(hyps), dist(pos, (qx, qy))))
    if not cands:
        return []
    best = min(c.sigma for c in cands)
    good = [c for c in cands if c.sigma <= best * 1.02] if best < math.inf \
        else [c for c in cands if c.sigma == math.inf]
    good.sort(key=lambda c: (c.travel, c.x, c.y))
    rest = sorted((c for c in cands if c not in good),
                  key=lambda c: (c.sigma, c.travel, c.x, c.y))
    return (good + rest)[:PROBE_TRY]


def ambiguity_area(baseline: float, alpha1_deg: float, alpha2_deg: float,
                   err_deg: float = SIGMA_DEG) -> float:
    """文献的定位模糊区面积口径（任叶童 2016 式 2-15，供对照/校核用）：/ m²

        S = 4·R²·Δθ²·sinα₁·sinα₂ / sin³(α₁+α₂)

    其中 R 为两检测点基线，αᵢ 为基线两端观测站处的内角，Δθ 为测向误差半宽（弧度）。
    该式在"基线 R 固定、目标位置自由"的口径下取最小值；与本文"检测点自由、最小化预测
    协方差"的口径不同，代码中保留此函数以便论文同时给出两种准则的结论。
    """
    a1, a2 = math.radians(alpha1_deg), math.radians(alpha2_deg)
    s = math.sin(a1) * math.sin(a2)
    den = math.sin(a1 + a2) ** 3
    return 4.0 * baseline ** 2 * math.radians(err_deg) ** 2 * s / den


# ----------------------------------------------------------------------------
# 机器狗：巡视扫描 → 定位区域分类 → 先清直径 < 40 m 的 → 其余按文献准则补测缩小后清除
# ----------------------------------------------------------------------------
class RobotDog:
    """两阶段机器狗。

    阶段一（巡视扫描）：按覆盖圆方案依次走到 7 个圆心，在每个圆心对未采够的频道测向，把
    每个源的示向度采集齐全（同一地点误差固定，故每频道最多采 OBS_CAP 条）。

    阶段二（定位与清除）：
      1. 用问题 1 的交会定位区域（各 ±1° 楔形之交 ∩ 圆域）处理每个频道；
      2. **先清掉定位区域直径 < 40 m 的**：此时区域的最小覆盖圆半径 ≤ 20 m，直接走到该
         圆心 /clear 就一定有源命中，不需要任何试探；
      3. **其余频道按文献准则补测**：在"最大化 Fisher 信息"（等价于"靠近源 + 拉开交角"）
         的候选点测一次，把区域直径压到 40 m 以下再清；单射线频道由此补齐第二视角。
      4. 兜底：万一清除失败，沿最新实测示向度以 HOMING_STEP 步长逼近（确定性，无随机）。
    """

    def __init__(self, sim, verbose: bool = True, logfile: Optional[str] = None,
                 episode: int = 0, clear: bool = True) -> None:
        self.sim = sim
        self.verbose = verbose
        self.clear_enabled = clear
        self._logfile = open(logfile, "a", encoding="utf-8") if logfile else None
        self.obs: Dict[int, List[Obs]] = defaultdict(list)
        self.regions: Dict[int, TriangulationRegion] = {}
        self.tracks: Dict[int, Dict[str, Any]] = {}      # 逐频道的定位/清除档案
        self.cleared: set = set()
        self.pos = np.zeros(2)
        self.vt = 0.0
        self.n_measure = 0
        self.n_clear = 0
        self.episode = episode
        self.deadline = float("inf")
        self.waypoint_stats: List[Dict[str, Any]] = []   # 逐圆心扫描统计
        self.travel_m = 0.0                              # 实际走过的里程 / m
        self.stage = "survey"                            # 观测所处阶段（survey / refine）

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
        """测向；direction 时记录示向度并同步进该频道的定位区域。"""
        x, y = float(x), float(y)
        r = self.sim.measure(x, y, channel)
        if not r.get("accepted"):
            raise RuntimeError(f"/measure 被拒绝：{r}")
        self.travel_m += dist(self.pos, (x, y))
        self.pos, self.vt = np.array([x, y]), float(r["virtual_time_s"])
        self.n_measure += 1
        if r.get("measure_result") == "direction":
            o = Obs(channel, x, y, float(r["svd_deg"]), self.stage)
            self.obs[channel].append(o)
            self.region(channel).add_node(o.x, o.y, o.theta)     # 增量并入楔形
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
            self.tracks.setdefault(channel, {})["clear_point"] = [x, y]
        return ok

    def _out_of_time(self) -> bool:
        return time.monotonic() > self.deadline

    def region(self, channel: int) -> TriangulationRegion:
        """取该频道的定位区域对象。

        惰性创建并把已有观测补进去；之后每次 measure() 用 add_node 增量并入一个新楔形，
        TriangulationRegion 内部按 _done 缓存已并入个数，故逐条读 diameter 只补一刀。
        """
        if channel not in self.regions:
            self.regions[channel] = TriangulationRegion(
                err=BEARING_ERROR_DEG, radius=REGION_RADIUS, sides=CLIP_SIDES)
            for o in self.obs.get(channel, ()):
                self.regions[channel].add_node(o.x, o.y, o.theta)
        return self.regions[channel]

    def diameter(self, channel: int) -> float:
        """该频道当前定位区域的直径 / m（0 表示区域为空/退化）。"""
        return float(self.region(channel).diameter)

    def _clearable(self, channel: int) -> bool:
        """判据：定位区域直径（含裁剪误差）< 40 m ⇒ 最小覆盖圆半径 < 20 m ⇒ 可直接清除。"""
        return self.diameter(channel) + 2.0 * CLIP_ERR < DIAM_CLEAR

    # ---- 阶段 1：巡视扫描 ----
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
        """依次移动到各圆心并扫描：阶段一的主体。"""
        self.log(f"阶段1 巡视扫描：依次访问 {len(order)} 个圆心"
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

    # ---- 阶段 2a：定位区域分类 ----
    def classify(self) -> Tuple[List[int], List[int]]:
        """按定位区域直径分类：返回（可直接清除的频道, 需补测缩小的频道）。"""
        ready, pending = [], []
        for ch in sorted(self.obs):
            if ch in self.cleared:
                continue
            d = self.diameter(ch)
            rec = self.tracks.setdefault(ch, {})
            rec.update({"n_obs_survey": len(self.obs[ch]),
                        "diameter_survey_m": round(d, 3),
                        "bounded_survey": bool(self.region(ch).bounded),
                        "n_probe": 0})
            (ready if self._clearable(ch) else pending).append(ch)
        self.log(f"阶段2 定位区域分类：可直接清除（直径 < {DIAM_CLEAR:.0f} m）"
                 f"{len(ready)} 个 {ready}；需补测缩小 {len(pending)} 个 {pending}")
        return ready, pending

    def _nearest_order(self, channels: Sequence[int]) -> List[int]:
        """确定性的访问顺序：最近邻给出初值，再用 2-opt 精修（固定起点、只接受严格下降）。

        访问点取各频道定位区域的最小覆盖圆圆心（清缺点），起点为机器狗当前位置。2-opt 是
        纯确定性精修（同一输入必得同一顺序），用于压缩"逐个清除"的里程 —— 里程占虚拟总
        时间的近九成，是本题时间指标的主要矛盾。
        """
        pts: Dict[int, np.ndarray] = {}
        for c in channels:
            mec = self.region(c).enclosing_circle
            pts[c] = np.array([mec[0], mec[1]]) if mec else np.array(self.pos, dtype=float)
        rest, order, cur = list(channels), [], np.array(self.pos, dtype=float)
        while rest:                                     # 最近邻初值
            nxt = min(rest, key=lambda c: (float(np.linalg.norm(pts[c] - cur)), c))
            order.append(nxt)
            rest.remove(nxt)
            cur = pts[nxt]
        return self._two_opt(order, pts, self.pos)

    @staticmethod
    def _two_opt(order: List[int], pts: Dict[int, np.ndarray],
                 start: Sequence[float]) -> List[int]:
        """开放路径 2-opt：反复尝试反转一段子路径，只接受里程严格下降的改动。"""
        def length(seq: Sequence[int]) -> float:
            total, cur = 0.0, np.array(start, dtype=float)
            for c in seq:
                total += float(np.linalg.norm(pts[c] - cur))
                cur = pts[c]
            return total
        best, cur_len = list(order), length(order)
        improved = True
        while improved:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    cand_len = length(cand)
                    if cand_len < cur_len - 1e-9:
                        best, cur_len, improved = cand, cand_len, True
        return best

    # ---- 阶段 2b：按文献准则补测缩小定位区域 ----
    def refine(self, channel: int) -> int:
        """补测直到定位区域直径 < 40 m（或达到轮次/候选上限）；返回实际补测次数。"""
        n_probe = 0
        for _ in range(REFINE_MAX):
            if self._clearable(channel) or self._out_of_time():
                break
            hyps = hypothesis_points(self.region(channel), self.obs[channel], self.pos)
            if not hyps:
                break
            cands = probe_candidates(self.obs[channel], hyps, self.pos)
            if not cands:
                break
            d0 = self.diameter(channel)
            got = False
            for c in cands:                       # 依次试候选：收不到信号就换下一个
                if self._out_of_time():
                    break
                self.stage = "refine"
                res = self.measure(c.x, c.y, channel).get("measure_result", "no_signal")
                n_probe += 1
                self.tracks.setdefault(channel, {})["n_probe"] = n_probe
                if res == "direction":
                    self.log(f"    [补测] 频道{channel} @ ({c.x:.0f}, {c.y:.0f})："
                             f"预测 σ {c.sigma:.2f} m，直径 {d0:.1f} → "
                             f"{self.diameter(channel):.1f} m")
                    got = True
                    break
                if res == "near":
                    self.log(f"    [补测] 频道{channel} @ ({c.x:.0f}, {c.y:.0f})："
                             f"源在 5 m 内，就地清除"
                             f"{'成功' if self.clear(c.x, c.y, channel) else '失败'}")
                    self.tracks.setdefault(channel, {})["method"] = "near@probe"
                    return n_probe
                self.log(f"    [补测] 频道{channel} @ ({c.x:.0f}, {c.y:.0f})：无信号，换候选点")
            if not got:
                break
        return n_probe

    # ---- 阶段 2c：清除 ----
    def _homing(self, channel: int) -> bool:
        """兜底：沿最新实测示向度以 HOMING_STEP 步长逼近，直到清除成功（确定性）。

        步数按"当前位置到定位区域最远顶点的距离"自适应（下限 HOMING_MAX、上限 HOMING_CAP）：
        起点离源很远时不能只走固定几步就放弃 —— 实测踩过：起点 636 m 远、只走 24×16 = 384 m
        就停手，把本可清掉的源漏掉。
        """
        budget = HOMING_MAX
        verts = self.region(channel).vertices
        if verts:
            far = max(dist(self.pos, v) for v in verts)
            budget = int(min(HOMING_CAP, max(HOMING_MAX, far / HOMING_STEP + 4.0)))
        p = np.array(self.pos, dtype=float)
        for _ in range(budget):
            if self._out_of_time():
                return False
            self.stage = "clear"
            r = self.measure(p[0], p[1], channel)
            res = r.get("measure_result")
            if res == "near":
                return self.clear(p[0], p[1], channel)
            if res != "direction":
                return False
            theta = math.radians(float(r["svd_deg"]))
            nxt = clamp_to_region(p[0] + HOMING_STEP * math.cos(theta),
                                  p[1] + HOMING_STEP * math.sin(theta))
            if self.clear(nxt[0], nxt[1], channel):
                return True
            p = np.array(nxt, dtype=float)
        return False

    def _try_clear(self, channel: int, tag: str) -> Optional[str]:
        """走到定位区域最小覆盖圆圆心，就地 /clear 一次；成功返回 tag，失败返回 None。

        这是全局唯一的清除位置：直径 < 40 m 时它保证命中（区域内任一点到圆心的距离
        ≤ 最小覆盖圆半径 < 20 m），直径略大时也常常命中，失败只花 3 s。
        """
        mec = self.region(channel).enclosing_circle
        if mec is None:
            return None
        cx, cy, r = mec
        if self.clear(cx, cy, channel):
            self.tracks.setdefault(channel, {})["clear_radius_m"] = round(r, 3)
            self.log(f"    [清除] 频道{channel} @ ({cx:.1f}, {cy:.1f}) 命中，"
                     f"最小覆盖圆半径 {r:.2f} m")
            return tag
        return None

    def _finish_clear(self, channel: int) -> str:
        """保证路径：先按区域中心清（失败则就地复测确认），最后兜底沿示向度逼近。"""
        if self._try_clear(channel, "direct"):
            return "direct"
        mec = self.region(channel).enclosing_circle
        if mec is not None:
            cx, cy = mec[0], mec[1]
            self.log(f"    [清除] 频道{channel} @ ({cx:.1f}, {cy:.1f}) 未命中，就地复测")
            self.stage = "clear"
            if self.measure(cx, cy, channel).get("measure_result") == "near" \
                    and self.clear(cx, cy, channel):
                return "near"
        if self._homing(channel):
            self.log(f"    [清除] 频道{channel} 兜底沿示向度逼近成功")
            return "homing"
        return "failed"

    def _opportunistic(self, channel: int) -> Optional[str]:
        """捷径：区域有界且估计不太差时，先走到区域中心试清一次。

        命中则省掉整轮补测（补测要绕 600 m 左右 ≈ 120 s 的里程，试清只需 3 s）；未命中时
        就站在离源很近的地方复测一次 —— 按 Fisher 信息口径，这正是"距离最近、权重最大"
        的最佳观测点，一条近距离射线往往直接把区域压到 40 m 以内。
        """
        if not self.clear_enabled or self._out_of_time():
            return None
        mec = self.region(channel).enclosing_circle
        if mec is None or not self.region(channel).bounded or mec[2] > TRY_CLEAR_RADIUS:
            return None
        cx, cy = mec[0], mec[1]
        if self.clear(cx, cy, channel):
            self.tracks.setdefault(channel, {})["clear_radius_m"] = round(mec[2], 3)
            self.log(f"    [就近试清] 频道{channel} @ ({cx:.1f}, {cy:.1f}) 直接命中"
                     f"（最小覆盖圆半径 {mec[2]:.2f} m，省掉一轮补测）")
            return "opportunistic"
        rec = self.tracks.setdefault(channel, {})
        rec["n_probe"] = int(rec.get("n_probe", 0)) + 1
        self.stage = "refine"
        res = self.measure(cx, cy, channel).get("measure_result", "no_signal")
        if res == "direction":
            self.log(f"    [就近试清] 频道{channel} @ ({cx:.1f}, {cy:.1f}) 未命中，"
                     f"就地复测（距离最近、信息量最大）：直径 "
                     f"{rec.get('diameter_survey_m', float('nan'))} → "
                     f"{self.diameter(channel):.1f} m")
        elif res == "near" and self.clear(cx, cy, channel):
            rec["method"] = "near@center"
            return "near@center"
        return None

    def process(self, channel: int) -> None:
        """一个频道的完整处理：能直接清的就直接清，其余补测缩小区域后再清。"""
        rec = self.tracks.setdefault(channel, {})
        if not self.clear_enabled:                   # 只定位模式：补测到直径 < 40 m 为止
            if not self._clearable(channel):
                self.refine(channel)
            method: Optional[str] = "skipped"
        elif self._clearable(channel):               # 保证路径：直径 < 40 m，走过去必中
            method = self._finish_clear(channel)
        else:                                        # 先就地试清（3 s），失败再按文献准则补测
            method = self._opportunistic(channel)
            if method is None:
                self.refine(channel)
                method = self._finish_clear(channel)
        d = self.diameter(channel)
        mec = self.region(channel).enclosing_circle
        rec.update({
            "diameter_final_m": round(d, 3),
            "clearable_final": self._clearable(channel),
            "mec_radius_m": round(mec[2], 3) if mec else None,
            "bounded_final": bool(self.region(channel).bounded),
            "n_obs_total": len(self.obs[channel]),
            "method": method,
            "cleared": channel in self.cleared,
        })

    # ---- 主流程 ----
    def run(self, plan: CoverPlan, order: Sequence[int]) -> Dict[str, Any]:
        """/enter → 巡视扫描 → 分类 → 先清直径<40 m 的 → 其余补测缩小后清除 → /exit。"""
        enter = self.sim.enter()
        if not enter.get("accepted"):
            raise RuntimeError(f"/enter 被拒绝：{enter}")
        left = float(enter.get("remaining_real_duration_s", 1200.0))
        self.deadline = time.monotonic() + max(left - SAFETY_MARGIN, 0.0)
        self.log(f"/enter 成功：虚拟时刻 {enter.get('virtual_time_s')} s，"
                 f"现实剩余 {left:.0f} s")
        try:
            self.survey(plan.waypoints, order)
            if self.clear_enabled:
                ready, pending = self.classify()
                for ch in self._nearest_order(ready):          # 先清直径 < 40 m 的
                    if self._out_of_time():
                        break
                    self.process(ch)
                for ch in self._nearest_order(pending):        # 其余补测缩小后再清
                    if self._out_of_time():
                        break
                    self.process(ch)
        finally:
            try:
                self.sim.exit()
            except OSError as exc:
                self.log(f"    [警告] /exit 失败：{exc}")
            self.close()
        n = len(self.cleared)
        return {
            "waypoints_visited": len(self.waypoint_stats),
            "travel_m": self.travel_m,
            "virtual_time_s": self.vt,
            "n_measure": self.n_measure,
            "n_clear": self.n_clear,
            "channels_heard": len(self.obs),
            "n_bearings": sum(len(v) for v in self.obs.values()),
            "cleared": n,
            "avg_time_s": self.vt / n if n else float("inf"),
            "n_ready_at_survey": sum(1 for r in self.tracks.values()
                                     if r.get("diameter_survey_m", float("inf"))
                                     + 2 * CLIP_ERR < DIAM_CLEAR),
            "n_refined": sum(1 for r in self.tracks.values() if r.get("n_probe")),
            "n_probe": sum(int(r.get("n_probe", 0)) for r in self.tracks.values()),
            "tracks": self.tracks,
        }


# ----------------------------------------------------------------------------
# 覆盖保证的真值核对（只有演练模式拿得到真值）
# ----------------------------------------------------------------------------
def truth_check(truth: Optional[Sequence[dict]], plan: CoverPlan,
                obs: Dict[int, List[Obs]], cleared: set,
                tracks: Optional[Dict[int, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """逐源核对（仅演练模式拿得到真值）：

    * 覆盖保证：源到最近覆盖圆圆心的距离是否 ≤ 1000 m、它的频道是否真的被听到；
    * 清除结果：清除点与真值的距离（定位误差）、是否落在 20 m 清除半径内。
    """
    tracks = tracks or {}
    rows: List[Dict[str, Any]] = []
    for j in truth or []:
        p = (float(j["position"]["x"]), float(j["position"]["y"]))
        d_near = float(np.linalg.norm(plan.waypoints - np.asarray(p), axis=1).min())
        heard = len(obs.get(j["channel"], ())) > 0 or j["channel"] in cleared
        rec = tracks.get(int(j["channel"]), {})
        cp = rec.get("clear_point")
        err = dist(cp, p) if cp else None
        rows.append({"channel": int(j["channel"]), "x": p[0], "y": p[1],
                     "receive_m": float(j["receive"]),
                     "nearest_center_m": round(d_near, 2),
                     "heard": bool(heard),
                     "n_bearings": len(obs.get(j["channel"], ())),
                     "cleared": j["channel"] in cleared,
                     "diameter_survey_m": rec.get("diameter_survey_m"),
                     "n_probe": rec.get("n_probe", 0),
                     "diameter_final_m": rec.get("diameter_final_m"),
                     "mec_radius_final_m": rec.get("mec_radius_m"),
                     "clear_radius_m": rec.get("clear_radius_m"),
                     "method": rec.get("method"),
                     "localize_err_m": None if err is None else round(err, 2)})
    errs = [r["localize_err_m"] for r in rows if r["localize_err_m"] is not None]
    return {
        "sources": rows,
        "worst_nearest_m": round(max((r["nearest_center_m"] for r in rows), default=0.0), 2),
        "all_within_cover": all(r["nearest_center_m"] <= COVER_RADIUS for r in rows),
        "missed_channels": [r["channel"] for r in rows if not r["heard"]],
        "n_cleared": sum(1 for r in rows if r["cleared"]),
        "localize_err_mean_m": round(float(np.mean(errs)), 3) if errs else None,
        "localize_err_max_m": round(float(np.max(errs)), 3) if errs else None,
        "n_within_clear_radius": sum(1 for e in errs if e <= CLEAR_RADIUS),
        "n_cleared_after_probe": sum(1 for r in rows if r["n_probe"] and r["cleared"]),
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


def summarize(rows: Sequence[dict]) -> Dict[str, Any]:
    """整批演练的阶段二汇总：清除率、时间、定位误差、清除方式分布。"""
    src = [s for r in rows for s in r.get("truth_check", {}).get("sources", ())]
    methods: Dict[str, int] = {}
    for s in src:
        methods[str(s.get("method"))] = methods.get(str(s.get("method")), 0) + 1
    errs = [s["localize_err_m"] for s in src if s.get("localize_err_m") is not None]
    n_src = sum(r["n_sources"] for r in rows if r["n_sources"])
    return {
        "episodes": len(rows),
        "n_sources": n_src,
        "n_cleared": sum(r.get("cleared", 0) for r in rows),
        "clear_ratio": (sum(r.get("cleared", 0) for r in rows) / n_src) if n_src else None,
        "avg_time_s": round(float(np.mean([r["avg_time_s"] for r in rows
                                           if r.get("avg_time_s")])), 3),
        "virtual_time_s_mean": round(float(np.mean([r["virtual_time_s"] for r in rows])), 3),
        "travel_m_mean": round(float(np.mean([r["travel_m"] for r in rows])), 1),
        "n_measure_mean": round(float(np.mean([r["n_measure"] for r in rows])), 1),
        "n_probe_mean": round(float(np.mean([r["n_probe"] for r in rows])), 2),
        "n_ready_at_survey": sum(r.get("n_ready_at_survey", 0) for r in rows),
        "n_refined": sum(r.get("n_refined", 0) for r in rows),
        "localize_err_mean_m": round(float(np.mean(errs)), 3) if errs else None,
        "localize_err_max_m": round(float(np.max(errs)), 3) if errs else None,
        "n_within_clear_radius": sum(1 for e in errs if e <= CLEAR_RADIUS),
        "methods": methods,
    }


def save_survey(save_dir: Path, rows: List[dict], observations: List[dict],
                plan_json: Dict[str, Any]) -> List[Path]:
    """巡视扫描结果落盘：逐局统计 JSON + 逐条观测 CSV。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / SURVEY_JSON
    json_path.write_text(json.dumps({
        "stage": "覆盖圆求解 + 依次到圆心巡视扫描 + 定位（先清直径<40 m，其余按文献准则补测缩小）",
        "cover_plan": plan_json,
        "clear_radius_m": CLEAR_RADIUS,
        "diam_clear_m": DIAM_CLEAR,
        "summary": summarize(rows),
        "episodes": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = save_dir / OBS_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "channel", "stage", "x_m", "y_m", "svd_deg",
                    "nearest_center_m"])
        for o in observations:
            w.writerow([o["episode"], o["channel"], o["stage"], f"{o['x']:.2f}",
                        f"{o['y']:.2f}", f"{o['theta']:.2f}",
                        f"{o['nearest_center_m']:.2f}"])
    return [json_path, csv_path]


def episode_row(ep: int, seed: int, truth: Optional[Sequence[dict]], dog: RobotDog,
                stats: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
    """单局汇总行：把引擎统计、真值核对与逐频道档案合成一行（供 JSON / 绘图使用）。"""
    n_src = len(truth) if truth else stats.get("channels_heard")
    cleared = stats.get("cleared", 0)
    return {
        "episode": ep, "seed": seed,
        "n_sources": len(truth) if truth else None,
        "clear_ratio": (cleared / n_src) if n_src else None,
        "localize_err_mean_m": check.get("localize_err_mean_m"),
        "localize_err_max_m": check.get("localize_err_max_m"),
        "n_within_clear_radius": check.get("n_within_clear_radius"),
        "n_cleared_after_probe": check.get("n_cleared_after_probe"),
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
                         "theta": o.theta, "stage": o.stage, "nearest_center_m": d})
    return rows


def _episode_printer(clear: bool):
    """按模式打印本局小结（只在需要时输出清除相关字段）。"""
    def show(stats: dict, check: dict, n_sources: Optional[int]) -> None:
        print(f"本局：巡视 {stats['waypoints_visited']} 个圆心，里程 {stats['travel_m']:.0f} m，"
              f"虚拟时间 {stats['virtual_time_s']:.0f} s，测向 {stats['n_measure']} 次；"
              f"听到 {stats['channels_heard']}/{n_sources} 个源"
              f"（{stats['n_bearings']} 条示向度）")
        if clear:
            print(f"  阶段2：巡视后直径已 < {DIAM_CLEAR:.0f} m 可直接清除 "
                  f"{stats['n_ready_at_survey']} 个；补测缩小 {stats['n_refined']} 个"
                  f"（共补测 {stats['n_probe']} 次）")
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
                   else Path(__file__).parent / "jammers-py")
    clear = not args.survey_only
    show = _episode_printer(clear)
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
                           verbose=not args.quiet, logfile=args.log, episode=ep + 1,
                           clear=clear)
            stats = dog.run(res.plan, res.survey_order)
            arena.finish_episode()
            check = truth_check(truth, res.plan, dog.obs, dog.cleared, dog.tracks)
            rows.append(episode_row(ep + 1, seed, truth, dog, stats, check))
            observations.extend(_observations(ep + 1, res.plan, dog.obs))
            show(stats, check, len(truth))
    print("\n" + "=" * 78)
    if clear:
        print(f"汇总（{len(rows)} 局）：平均清除比例 "
              f"{np.mean([r['clear_ratio'] for r in rows]):.4f}"
              f"（{sum(r['cleared'] for r in rows)}/{sum(r['n_sources'] for r in rows)}），"
              f"平均 {np.mean([r['avg_time_s'] for r in rows if r['avg_time_s']]):.1f} s/个，"
              f"平均虚拟时间 {np.mean([r['virtual_time_s'] for r in rows]):.0f} s，"
              f"平均测向 {np.mean([r['n_measure'] for r in rows]):.0f} 次"
              f"（其中补测 {np.mean([r['n_probe'] for r in rows]):.0f} 次）")
        print(f"  定位误差：均值 "
              f"{np.mean([r['localize_err_mean_m'] for r in rows if r['localize_err_mean_m']]):.2f}"
              f" m，最差单源 "
              f"{max([r['localize_err_max_m'] for r in rows if r['localize_err_max_m']] or [0]):.2f}"
              f" m；巡视后即可直接清除的源 "
              f"{sum(r['n_ready_at_survey'] for r in rows)}/{sum(r['n_sources'] for r in rows)} 个")
        print("逐局：" + "  ".join(f"seed{r['seed']}={r['cleared']}/{r['n_sources']}"
                                  for r in rows))
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
    sim = sim_api.Simulator(robot_id=args.robot_id, base_url=args.base_url, timeout=args.timeout)
    print(f"连接模拟器 {args.base_url}（robot_id={args.robot_id}）")
    dog = RobotDog(sim, verbose=not args.quiet, logfile=args.log, episode=1,
                   clear=not args.survey_only)
    stats = dog.run(res.plan, res.survey_order)
    print(f"完成：清除 {stats['cleared']} 个，巡视 {stats['waypoints_visited']} 个圆心，"
          f"里程 {stats['travel_m']:.0f} m，虚拟时间 {stats['virtual_time_s']:.0f} s，"
          f"测向 {stats['n_measure']} 次（补测 {stats['n_probe']} 次），"
          f"听到 {stats['channels_heard']} 个频道（{stats['n_bearings']} 条示向度）")
    row = episode_row(1, args.seed, None, dog, stats,
                      truth_check(None, res.plan, dog.obs, dog.cleared, dog.tracks))
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
    p.add_argument("--survey-only", action="store_true",
                   help="只做阶段一（巡视扫描 + 覆盖核对），不做定位与清除")
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
