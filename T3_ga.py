#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T3_ga.py —— 2026 全国大学生数学建模竞赛 B 题 · 问题三
机器狗自动定位并清除无线电干扰源（遗传算法求解）

策略（四个阶段，定位与路线全用遗传算法）
======================================
1. 起点扫描：在原点对全部 20 个频道测向，先拿到一批示向度。
2. 覆盖观测：贪心集合覆盖挑出尽量少的路点，使整个半径 1800 m 圆域内任意点到最近路点
   不超过 920 m（< 有效接收半径下界 1000 m，实算最坏点 962 m，余量 38 m），再用路线 GA
   定序依次访问；每个路点对尚未确认的频道测向。既不漏源，又天然形成多视角交会。
3. GA 定位：个体是干扰源位置 (x, y)，适应度为示向度残差 RMS（度）+ 越界惩罚（距离超过
   有效接收半径上界、位置越出圆域）；初始种子用射线两两交点加速收敛。若交会几何病态
   （位置 1σ 过大或射线近共线），或某频道只有单条射线，则补测"垂直视角"。
4. GA 规划清除：路线 GA 规划已定位源的访问顺序，逐个靠近后先 /clear；未成功则沿最新
   实测示向度以 16 m 步长末端逼近（±1° 的横向偏差约 0.3 m），直到返回 success。

运行
====
    # 正式测试：先在模拟器中开始演练/测试，再运行本程序
    python T3_ga.py --robot-id 202614023005

    # 本地演练（= GA 训练）：自动拉起同目录 jammers-py 模拟器，跑 5 局固定场景并汇总
    # 同一 --seed 完全可复现（场景布局与示向度噪声都由 seed 决定）
    python T3_ga.py --practice 5 --seed 100

训练结果落盘
============
每次运行都会把 GA 的"训练过程"（种群逐代进化）与逐局战绩写入 `--save-dir`（缺省 results/）：

    results/ga_training.json   训练配置 + 逐局统计 + 每次 GA 调用的最终解/收敛代数/残差
    results/ga_convergence.csv 逐代收敛曲线（局号, GA 类型, 对象, 代数, 最优/平均/标准差）
    results/episodes.csv       逐局战绩（清除数, 清除比例, 虚拟时间, 测向次数）
    results/api_calls.jsonl    每一次机器狗接口调用的记录（见下节）

即：定位 GA 的"模型参数"是各频道干扰源坐标与它收敛所需的代数，路线 GA 的"模型"是访问
序列表；两者都随 JSON 保存，收敛曲线可直接用于论文绘图。文件名固定，重复运行覆盖。

接口调用日志
============
每一次机器狗接口调用（/enter /measure /clear /exit）都会逐条记录到
`results/api_calls.jsonl`（每行一条，可用 --api-log 改路径、传空串关闭）：

    局号、本局内序号、接口名、request_id、相对时刻与耗时、请求参数（坐标/频道）、
    是否 accepted、原始响应全文；连不上或超时的调用也会留痕（ok=false + error）。

request_id 由本程序生成（`<接口>-<局号>-<序号>`），既保证跨局不重复，又能与模拟器
自身行为日志里的同一条请求逐行对照 —— 比赛现场据此复核每一次动作。

依赖：numpy；HTTP 层复用同目录 sim_api.py；本地演练用同目录 jammers-py/（纯标准库）。

官方评测平台用法（赛期）
------------------------------------------------
1. 在本机运行官方模拟器（Windows 程序，界面依赖 WebView2；Wine 下缺 WebView2 无法启动），
   联网登录后进入"问题3演练测试"或"问题3正式测试"，确认开始并等 5 秒倒计时结束。
2. 倒计时结束后机器狗接口才开放（默认 http://127.0.0.1:2026）。此时执行：
       python T3_ga.py                       # 连默认地址，robot_id 已填参赛队号
       python T3_ga.py --base-url http://127.0.0.1:8080    # 若在设置里改过端口
   程序会自行 /enter → 覆盖观测 → GA 定位 → GA 规划清除 → /exit，单局现实耗时约 3 秒
   （现实限时 20 分钟），结果默认落到 results/official/。
3. 接口只监听 127.0.0.1，因此本程序必须与模拟器在同一台机器上运行。
4. 接口未开放时连接会被直接关闭，程序会提示"请确认模拟器已启动并处于测试窗口内"。
   正式测试每题只有 3 次机会，务必先用演练测试跑通。
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
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

import sim_api

# ----------------------------------------------------------------------------
# 常量（与附件 1 / 附件 2 一致）
# ----------------------------------------------------------------------------
REGION_RADIUS = 1800.0          # 目标圆域半径 / m
MAX_RECEPTION = 1500.0          # 有效接收半径上界 / m（下界 1000 m，见 COVER_RADIUS）
BEARING_ERROR_DEG = 1.0         # 示向度误差半宽 / 度
CHANNELS: Tuple[int, ...] = tuple(range(1, 21))
COORD_LIMIT = 2.0e6             # 坐标分量绝对值上限 / m

# 覆盖路点设计：三个参数共同决定"圆域内任意点到最近路点 ≤ COVER_RADIUS"的保证强度。
# 离散化越细，实际能达到的最坏距离越接近 COVER_RADIUS。实算（连续圆域上求最坏点）：
#   100/120/950 → 8 路点，最坏 1008.4 m（>1000，圆域边缘存在听不到的源 ❌）
#    60/ 80/920 → 8 路点，最坏  962.4 m（余量 +37.6 m，巡视里程还短 480 m ✅ 现用）
COVER_RADIUS = 920.0            # 覆盖路点设计半径（<1000 接收下界）/ m
COVER_GRID = 60.0               # 目标圆域离散网格 / m
COVER_STEP = 80.0               # 候选路点网格 / m
REGION_MARGIN = 1.0             # 坐标裁剪时保留的数值余量 / m（机器狗须在圆域内）
OBS_PER_SOURCE = 3              # 每个频道尽量采集的示向度条数

# 位置 1σ 超过该值即视为交会几何病态并补测垂直视角。校准依据（200 组随机交会实验）：
# σ > 40 m 的频道约占 10%，其真实定位误差中位数 28 m（已超过 20 m 清除半径），而
# σ < 40 m 的频道误差中位数仅 4~6 m —— 阈值正好把"会失败的交会"挑出来。
GEOM_SIGMA = 40.0               # 位置 1σ 超过该值视为交会几何病态 / m
GEOM_SIN_MIN = 0.20             # 两射线夹角 |sin| 小于该值视为近共线
PERP_STEPS = (250.0, 500.0, 800.0)      # 补测垂直视角时外移的距离 / m
# 说明：补测点按"沿某方向外移固定距离"生成，源靠近圆域边缘时可能挪到圆域之外（机器狗
# 不允许离开作业圆域，虽然模拟器不会拒绝）。所有动作坐标统一由 clamp_to_region 拉回域内，
# 因此无需在各处候选点生成逻辑里重复裁剪，代价只是基线略短、σ 略大。
SINGLE_PROBES = ((600.0, 35.0), (400.0, 45.0), (800.0, 25.0))   # 单射线补测：前移距离/侧偏角
HOMING_STEP = 16.0              # 末端沿示向度逼近的步长 / m
HOMING_MAX = 24                 # 末端逼近最大迭代次数
SAFETY_MARGIN = 30.0            # 现实时限预留余量 / s

SEED = 2026

RESULTS_DIR = "results"         # 训练结果输出目录（可用 --save-dir 改）
API_LOG_NAME = "api_calls.jsonl"    # 接口调用日志文件名（落在 --save-dir 下）


@dataclass(frozen=True)
class GAParams:
    """GA 超参数：只在这里定义一次，GA 实例与训练记录的 meta 共用，避免两处不同步。"""

    pop: int
    gens: int
    pc: float
    pm: float


GA_LOC = GAParams(pop=80, gens=150, pc=0.90, pm=0.35)       # 定位 GA（实数编码）
GA_ROUTE = GAParams(pop=100, gens=300, pc=0.90, pm=0.35)    # 路线 GA（排列编码，开路径 TSP）
GA_LOC_DOMAIN = 2200.0          # 定位 GA 个体取值半径 / m
GA_RECORD_STRIDE = 5            # 训练记录抽稀步长：每多少代记一个收敛点
CONVERGED_FITNESS = 1e-4        # 定位 GA 提前收敛的适应度阈值


# ----------------------------------------------------------------------------
# 基础数学工具
# ----------------------------------------------------------------------------
def bearing(a: Sequence[float], b: Sequence[float]) -> float:
    """a → b 的方位角（度，x 轴正向逆时针，[0,360)）。"""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def ang_diff(a: float, b: float) -> float:
    """两角的最小绝对差（度）。"""
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class Obs:
    """一次 direction 检测：在 (x, y) 处测得频道 channel 的示向度 theta（度）。"""

    channel: int
    x: float
    y: float
    theta: float

    @property
    def point(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def direction(self) -> np.ndarray:
        """示向度单位向量。"""
        t = math.radians(self.theta)
        return np.array([math.cos(t), math.sin(t)], dtype=float)

    @property
    def normal(self) -> np.ndarray:
        """射线的单位法向量。"""
        t = math.radians(self.theta)
        return np.array([-math.sin(t), math.cos(t)], dtype=float)


class Estimate(NamedTuple):
    """单频道源的位置估计：坐标与位置 1σ 不确定度（米）。"""

    x: float
    y: float
    sigma: float

    @property
    def point(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


def ray_intersection(o1: Obs, o2: Obs) -> Optional[np.ndarray]:
    """两条射线所在直线的交点；近共线（行列式≈0）时返回 None。"""
    d1, d2 = o1.direction, o2.direction
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return None
    ex, ey = o2.x - o1.x, o2.y - o1.y
    t = (ex * d2[1] - ey * d2[0]) / cross
    return o1.point + t * d1


def max_ray_sine(obs_list: Sequence[Obs]) -> float:
    """所有射线两两夹角的正弦最大值：接近 0 表示近共线（无法定距）。"""
    if len(obs_list) < 2:
        return 0.0
    t = np.radians([o.theta for o in obs_list])
    dirs = np.stack((np.cos(t), np.sin(t)), axis=1)
    cross = np.abs(dirs[:, None, 0] * dirs[None, :, 1] - dirs[:, None, 1] * dirs[None, :, 0])
    return float(cross.max())


def position_sigma(cov: np.ndarray) -> float:
    """位置 1σ 近似：协方差阵迹的平方根。"""
    return float(math.sqrt(max(float(np.trace(cov)), 0.0)))


def _split_runs(ga_runs: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    """把训练记录按 GA 类型拆成（定位 GA 记录, 路线 GA 记录）。"""
    return ([r for r in ga_runs if r["ga"] == "localize"],
            [r for r in ga_runs if r["ga"] == "route"])


# ----------------------------------------------------------------------------
# GA 通用算子（两个 GA 共用）
# ----------------------------------------------------------------------------
def _record(record: Optional[List[List[float]]], gen: int, cost: np.ndarray) -> None:
    """把一个种群代次的最优/平均/标准差适应度追加进训练记录（record 为 None 时不做）。"""
    if record is not None:
        record.append([gen, float(cost.min()), float(cost.mean()), float(cost.std())])


def _pick(pop: np.ndarray, cost: np.ndarray, rng, k: int = 3) -> np.ndarray:
    """锦标赛选择单个个体（排列编码用）：抽 k 个候选，取代价最小者，返回副本。"""
    idx = rng.integers(0, len(pop), size=k)
    return pop[idx[int(np.argmin(cost[idx]))]].copy()


def _tournament(pop: np.ndarray, cost: np.ndarray, rng, k: int = 3) -> np.ndarray:
    """锦标赛选择整群（实数编码用，向量化）：每个子代抽 k 个候选，取代价最小者。"""
    idx = rng.integers(0, len(pop), size=(len(pop), k))
    winners = np.argmin(cost[idx], axis=1)
    return pop[idx[np.arange(len(pop)), winners]].copy()


# ----------------------------------------------------------------------------
# 遗传算法一：测向定位（实数编码）
# ----------------------------------------------------------------------------
class GeneticLocalizer:
    """极小化示向度残差的定位 GA。

    适应度 = RMS 示向度残差(度)
             + PENALTY_DIST   × Σ max(0, 观测点到源的距离 − 有效接收半径上界)
             + PENALTY_REGION × max(0, |G| − 圆域半径)

    后两项把"远处无信号"的区域排除掉，避免近共线时残差谷延伸到无穷远而不唯一。
    """

    PENALTY_DIST = 0.02
    PENALTY_REGION = 0.02

    def __init__(self, params: GAParams = GA_LOC, domain: float = GA_LOC_DOMAIN,
                 seed: int = SEED) -> None:
        self.pop, self.gens, self.pc, self.pm = params.pop, params.gens, params.pc, params.pm
        self.domain = domain
        self.seed = seed

    def _fitness(self, P: np.ndarray, S: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """整种群向量化适应度：P 形状 (n,2)，S/(theta) 为观测点与示向度。"""
        dx = P[:, 0, None] - S[None, :, 0]
        dy = P[:, 1, None] - S[None, :, 1]
        d = np.hypot(dx, dy)
        diff = ((np.degrees(np.arctan2(dy, dx)) - theta[None, :] + 180.0) % 360.0) - 180.0
        rms = np.sqrt(np.mean(diff * diff, axis=1))
        excess = np.maximum(d - MAX_RECEPTION, 0.0).sum(axis=1)
        outside = np.maximum(np.hypot(P[:, 0], P[:, 1]) - REGION_RADIUS, 0.0)
        return rms + self.PENALTY_DIST * excess + self.PENALTY_REGION * outside

    @staticmethod
    def _blx(A: np.ndarray, B: np.ndarray, rng, alpha: float = 0.30) -> Tuple[np.ndarray, np.ndarray]:
        """BLX-α 交叉（整群向量化）：子代取自两父代区间外扩 alpha 倍的范围内。"""
        lo, hi = np.minimum(A, B), np.maximum(A, B)
        span = (hi - lo) * (1.0 + 2.0 * alpha)
        return (lo - alpha * (hi - lo) + rng.random(A.shape) * span,
                lo - alpha * (hi - lo) + rng.random(A.shape) * span)

    def localize(self, obs_list: Sequence[Obs],
                 record: Optional[List[List[float]]] = None) -> Optional[np.ndarray]:
        """求该频道干扰源位置；少于 2 条射线时无法定距，返回 None。

        传入 record 时逐代追加 [代数, 最优适应度, 种群平均适应度, 种群标准差]，即 GA 的
        训练（进化）过程；为控制体量按 GA_RECORD_STRIDE 抽稀，并保证记录到最优收敛那一代。
        """
        m = len(obs_list)
        if m < 2:
            return None

        S = np.array([o.point for o in obs_list])
        theta = np.array([o.theta for o in obs_list])
        rng = np.random.default_rng(self.seed)
        n, dom = self.pop, self.domain

        # 初始种群：射线两两交点 + 观测点质心作种子，其余随机撒点
        seeds = [p for i in range(m) for j in range(i + 1, m)
                 if (p := ray_intersection(obs_list[i], obs_list[j])) is not None][: n - 1]
        seeds.append(S.mean(axis=0))
        P = rng.uniform(-dom, dom, size=(n, 2))
        P[: len(seeds)] = np.clip(np.array(seeds), -dom, dom)
        cost = self._fitness(P, S, theta)
        _record(record, 0, cost)

        for gen in range(self.gens):
            order = np.argsort(cost)
            P, cost = P[order], cost[order]
            sigma = 220.0 * (1.0 - gen / self.gens) + 25.0        # 变异强度随代数线性降温
            A, B = _tournament(P, cost, rng), _tournament(P, cost, rng)
            cross = (rng.random(len(A)) < self.pc)[:, None]
            c1, c2 = self._blx(A, B, rng)
            C = np.vstack((np.where(cross, c1, A), np.where(cross, c2, B)))
            C += (rng.random(C.shape) < self.pm) * rng.normal(0.0, sigma, size=C.shape)
            np.clip(C, -dom, dom, out=C)
            P = np.vstack((P[0], P[1], C))[:n]                    # 精英保留：最优两个个体
            cost = self._fitness(P, S, theta)
            converged = bool(cost[0] < CONVERGED_FITNESS)
            if converged or gen % GA_RECORD_STRIDE == 0 or gen == self.gens - 1:
                _record(record, gen + 1, cost)
            if converged:
                break

        return P[int(cost.argmin())].copy()

    @staticmethod
    def covariance(obs_list: Sequence[Obs], G: Sequence[float]) -> np.ndarray:
        """位置协方差：射线法向信息矩阵的逆（每条射线给出一个"垂向"约束）。

        信息矩阵按 Fisher 信息累积：A = Σ 1/(d_i·σ_θ)² · n_i n_iᵀ，其中 n_i 是第 i 条
        射线的单位法向、d_i 为观测点到解的距离（取 50 m 下限以免近距权重爆炸）。
        注意 A 必须从**零矩阵**起（不可加单位阵）：多一个 1 m⁻² 的先验会让 σ 恒等于
        1.4 m，既失去随几何变化的信息，也让 GEOM_SIGMA 判据永远不触发。
        射线近共线时 A 近乎奇异，此时返回大方差矩阵，由调用方补测视角。
        """
        A = np.zeros((2, 2))
        for o in obs_list:
            d = max(float(np.linalg.norm(np.asarray(G, dtype=float) - o.point)), 50.0)
            w = 1.0 / (d * math.radians(BEARING_ERROR_DEG)) ** 2
            A += w * np.outer(o.normal, o.normal)
        A += np.eye(2) * 1e-12              # 仅用于数值可逆，量级远小于真实信息
        try:
            return np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return np.eye(2) * 1e6


# ----------------------------------------------------------------------------
# 遗传算法二：访问顺序（排列编码，开路径 TSP）
# ----------------------------------------------------------------------------
def _dist_matrix(pts: np.ndarray, start: Sequence[float]) -> np.ndarray:
    """距离矩阵 D，形状 (n+1, n)：D[0, j] 为起点到点 j，D[1+i, j] 为点 i 到点 j。"""
    nodes = np.vstack((np.asarray(start, dtype=float)[None, :], np.asarray(pts, dtype=float)))
    pts = np.asarray(pts, dtype=float)
    return np.linalg.norm(nodes[:, None, :] - pts[None, :, :], axis=2)


def _path_len(order: Sequence[int], D: np.ndarray) -> float:
    o = np.asarray(order, dtype=int)
    return 0.0 if len(o) == 0 else float(D[0, o[0]] + D[1 + o[:-1], o[1:]].sum())


def _path_len_vec(orders: np.ndarray, D: np.ndarray) -> np.ndarray:
    """整种群路径长度（向量化），orders 形状 (pop, n)。"""
    return D[0, orders[:, 0]] + D[1 + orders[:, :-1], orders[:, 1:]].sum(axis=1)


def _nearest_order(n: int, D: np.ndarray) -> List[int]:
    """最近邻构造初始个体（0 行代表起点）。"""
    remaining, cur, order = list(range(n)), 0, []
    while remaining:
        j = min(remaining, key=lambda i: D[cur, i])
        order.append(j)
        cur = 1 + j
        remaining.remove(j)
    return order


def _ox(p1: np.ndarray, p2: np.ndarray, rng) -> np.ndarray:
    """顺序交叉（OX）：保留 p1 的一段，其余按 p2 的相对顺序补全。"""
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    child = np.empty(n, dtype=int)
    child[a:b + 1] = p1[a:b + 1]
    used = set(p1[a:b + 1].tolist())
    fill = (int(v) for v in p2 if int(v) not in used)
    for i in list(range(a)) + list(range(b + 1, n)):
        child[i] = next(fill)
    return child


def _mutate(order: np.ndarray, rng, pm: float) -> None:
    """倒位变异（等价一次 2-opt 走子）与交换变异，各自以概率 pm 触发。"""
    n = len(order)
    if n < 2:
        return
    if rng.random() < pm:
        i, j = sorted(rng.choice(n, 2, replace=False))
        order[i:j + 1] = order[i:j + 1][::-1]
    if rng.random() < pm:
        i, j = rng.choice(n, 2, replace=False)
        order[i], order[j] = order[j], order[i]


def two_opt(order: Sequence[int], D: np.ndarray) -> List[int]:
    """2-opt 精修（起点固定、终点自由的开路径），按增量代价翻转段，不重算整条路径。"""
    order = [int(v) for v in order]

    def d(a: int, b: int) -> float:
        return float(D[0, b] if a < 0 else D[1 + a, b])       # a<0 表示起点

    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            prev = order[i - 1] if i else -1
            for j in range(i + 1, len(order)):
                nxt = order[j + 1] if j + 1 < len(order) else -1
                old = d(prev, order[i]) + (0.0 if nxt < 0 else d(order[j], nxt))
                new = d(prev, order[j]) + (0.0 if nxt < 0 else d(order[i], nxt))
                if new < old - 1e-9:
                    order[i:j + 1] = reversed(order[i:j + 1])
                    improved = True
                    break
            if improved:
                break
    return order


def route_ga(
    pts: np.ndarray,
    start: Sequence[float],
    params: GAParams = GA_ROUTE,
    seed: int = SEED,
    record: Optional[List[List[float]]] = None,
) -> List[int]:
    """GA 求从 start 出发访问全部 pts 的近似最短开路径，返回访问顺序的下标。

    传入 record 时逐代追加 [代数, 最短路径长, 种群平均路径长, 标准差]，即路线 GA 的训练过程。
    """
    n = len(pts)
    if n == 0:
        return []
    if n == 1:
        return [0]

    pop, gens, pc, pm = params.pop, params.gens, params.pc, params.pm
    pop = max(pop, 4)                       # 至少留 2 个精英 + 1 对子代
    rng = np.random.default_rng(seed)
    D = _dist_matrix(pts, start)
    # 初始种群：一个最近邻个体 + 若干随机排列
    population = np.array([_nearest_order(n, D)] + [rng.permutation(n) for _ in range(pop - 1)])
    cost = _path_len_vec(population, D)
    _record(record, 0, cost)

    for gen in range(gens):
        order = np.argsort(cost)
        population, cost = population[order], cost[order]
        pm_gen = pm * (1.0 - 0.5 * gen / gens) + 1e-3
        newpop = [population[0], population[1]]   # 精英保留
        while len(newpop) < pop:
            a, b = _pick(population, cost, rng), _pick(population, cost, rng)
            if rng.random() < pc:
                c1, c2 = _ox(a, b, rng), _ox(b, a, rng)
            else:
                c1, c2 = a, b
            _mutate(c1, rng, pm_gen)
            _mutate(c2, rng, pm_gen)
            newpop.extend((c1, c2))
        population = np.array(newpop[:pop])
        cost = _path_len_vec(population, D)
        if gen % GA_RECORD_STRIDE == 0 or gen == gens - 1:
            _record(record, gen + 1, cost)

    best = population[int(cost.argmin())].tolist()
    return two_opt(best, D)


# ----------------------------------------------------------------------------
# 覆盖路点：贪心集合覆盖（保证圆域内任意点都在某路点 COVER_RADIUS 内）
# ----------------------------------------------------------------------------
def clamp_to_region(x: float, y: float,
                    radius: float = REGION_RADIUS - REGION_MARGIN) -> Tuple[float, float]:
    """把坐标拉回作业圆域内（出域时沿原方向缩到边界）。

    用于所有 /measure 与 /clear 的入口：定位 GA 的解、垂直/单射线补测点、末端归航步进都
    可能落在圆域之外，统一在此裁剪，内部逻辑无需各自判断。
    """
    r = math.hypot(x, y)
    if r <= radius or r == 0.0:
        return x, y
    k = radius / r
    return x * k, y * k


def _disk_grid(radius: float, step: float) -> np.ndarray:
    ax = np.arange(-radius, radius + 1e-9, step)
    gx, gy = np.meshgrid(ax, ax)
    pts = np.stack((gx.ravel(), gy.ravel()), axis=1)
    return pts[np.linalg.norm(pts, axis=1) <= radius + 1e-9]


@lru_cache(maxsize=1)
def covering_waypoints() -> np.ndarray:
    """用尽量少的路点覆盖整个目标圆域（每轮取"新增覆盖目标点"最多的候选路点）。"""
    targets = _disk_grid(REGION_RADIUS, COVER_GRID)
    centers = _disk_grid(REGION_RADIUS, COVER_STEP)
    reach = np.linalg.norm(centers[:, None, :] - targets[None, :, :], axis=2) <= COVER_RADIUS
    covered = np.zeros(len(targets), dtype=bool)
    chosen: List[np.ndarray] = []
    while not covered.all():
        gain = reach[:, ~covered].sum(axis=1)
        k = int(gain.argmax())
        if gain[k] == 0:
            break
        chosen.append(centers[k])
        covered |= reach[k]
    return np.array(chosen, dtype=float)


# ----------------------------------------------------------------------------
# 接口调用日志：逐条记录 /enter /measure /clear /exit
# ----------------------------------------------------------------------------
def _ms(t0: float) -> int:
    """自 t0 起的毫秒耗时。"""
    return int((time.monotonic() - t0) * 1000)


class ApiLog:
    """把每一次接口调用（请求参数 + 原始响应）写成 JSONL，并按需回显一行摘要。

    模拟器自身有行为日志，但那份记录不归我们掌握；这里留一份自己的痕迹：逐条含
    request_id，可与模拟器日志逐行对照。每次调用后立即 flush，即使中途断连或崩溃，
    已经发生的调用也不会丢。
    """

    def __init__(self, path: Path, echo: Optional[Callable[[str], None]] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._echo = echo
        self._t0 = time.monotonic()
        self.counts: Dict[str, int] = defaultdict(int)

    def elapsed(self) -> float:
        """自日志建立起的秒数（单调时钟），用于记录各次调用的相对时刻。"""
        return time.monotonic() - self._t0

    def write(self, rec: Dict[str, Any], line: str) -> None:
        """落盘一条调用记录，并把单行摘要交给终端。"""
        self.counts[rec["call"]] += 1
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        if self._echo:
            self._echo(line)

    def report(self) -> None:
        """打印一行统计，便于与模拟器自己的计数交叉核对。"""
        if self.counts:
            detail = "、".join(f"{call} {n}" for call, n in sorted(self.counts.items()))
            print(f"接口调用日志：共 {sum(self.counts.values())} 次（{detail}）→ {self.path}")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


@contextmanager
def _api_log(args: argparse.Namespace, echo: bool) -> Iterator[Optional[ApiLog]]:
    """接口日志的上下文：进入时建文件，退出时打印统计并关闭。

    路径取 --api-log；未指定时用 <save-dir>/api_calls.jsonl；显式传空串则关闭日志。
    """
    raw = args.api_log if args.api_log is not None else str(Path(args.save_dir) / API_LOG_NAME)
    log = ApiLog(Path(raw), echo=print if echo else None) if raw else None
    try:
        yield log
    finally:
        if log is not None:                 # 统计由 _save_and_report 统一在尾部打印
            log.close()


def _api_brief(call: str, resp: dict, elapsed_ms: int) -> str:
    """一次调用的结果摘要（终端可读的单行）。"""
    if not resp.get("accepted"):
        return "拒绝 accepted=false"
    tail = f"（{elapsed_ms} ms"
    if resp.get("virtual_time_s") is not None:
        tail += f"，虚拟 {resp['virtual_time_s']} s"
    tail += "）"
    if call == "/measure":
        result = resp.get("measure_result", "?")
        return (f"{result} {resp['svd_deg']}°{tail}" if result == "direction"
                else f"{result}{tail}")
    if call == "/clear":
        return f"{resp.get('clear_result', '?')}{tail}"
    if call == "/enter":
        return f"accepted，现实剩余 {resp.get('remaining_real_duration_s')} s{tail}"
    return f"accepted{tail}"


class RecordedSim:
    """Simulator 的记录代理：原样转发 4 个接口，并把每次调用交给 ApiLog 落盘。

    request_id 由本层生成（`<接口>-<局号>-<序号>`）：既保证跨局不重复，也让我们的日志
    能与模拟器行为日志里的同一条请求对上号。
    """

    def __init__(self, sim: "sim_api.Simulator", log: ApiLog, episode: int = 0) -> None:
        self._sim = sim
        self._log = log
        self.episode = episode
        self.seq = 0

    def enter(self) -> dict:
        return self._call("/enter", {}, lambda rid: self._sim.enter(request_id=rid))

    def measure(self, x: float, y: float, channel: int) -> dict:
        x, y = clamp_to_region(float(x), float(y))
        return self._call("/measure", {"x": x, "y": y, "channel": int(channel)},
                          lambda rid: self._sim.measure(x, y, int(channel), request_id=rid))

    def clear(self, x: float, y: float, channel: int) -> dict:
        x, y = clamp_to_region(float(x), float(y))
        return self._call("/clear", {"x": x, "y": y, "channel": int(channel)},
                          lambda rid: self._sim.clear(x, y, int(channel), request_id=rid))

    def exit(self) -> dict:
        return self._call("/exit", {}, lambda rid: self._sim.exit(request_id=rid))

    def _call(self, call: str, params: Dict[str, Any], send: Callable[[str], dict]) -> dict:
        """执行一次调用并记录：先落请求与时刻，再按成功/异常分别补全结果。"""
        self.seq += 1
        rid = f"{call.strip('/')}-{self.episode}-{self.seq}"
        rec: Dict[str, Any] = {"episode": self.episode, "seq": self.seq, "call": call,
                               "request_id": rid, "at_s": round(self._log.elapsed(), 3),
                               **params}
        where = "" if "channel" not in rec else f" ({rec['x']:.0f}, {rec['y']:.0f}) 频道{rec['channel']}"
        t0 = time.monotonic()
        try:
            resp = send(rid)
        except Exception as exc:            # 连不上/超时也要留痕，再把异常交回上层
            elapsed = _ms(t0)
            why = f"{type(exc).__name__}: {exc}"
            self._log.write({**rec, "ok": False, "elapsed_ms": elapsed, "error": why},
                            f"[接口] ep{self.episode} #{self.seq} {call}{where} → 失败 {why} "
                            f"（{elapsed} ms）")
            raise
        rec["elapsed_ms"] = _ms(t0)
        rec["ok"] = bool(resp.get("accepted"))
        rec["response"] = resp
        self._log.write(rec, f"[接口] ep{self.episode} #{self.seq} {call}{where} → "
                             f"{_api_brief(call, resp, rec['elapsed_ms'])}")
        return resp


# ----------------------------------------------------------------------------
# 机器狗策略
# ----------------------------------------------------------------------------
class RobotDog:
    """覆盖观测 → GA 定位 → GA 规划路线逐个清除。"""

    def __init__(self, sim, verbose: bool = True, logfile: Optional[str] = None,
                 seed: int = SEED, episode: int = 0,
                 api_log: Optional[ApiLog] = None) -> None:
        # 传入 api_log 时套一层记录代理：4 个接口的每一次调用都会落盘
        self.sim = sim if api_log is None else RecordedSim(sim, api_log, episode)
        self.verbose = verbose
        self._logfile = open(logfile, "a", encoding="utf-8") if logfile else None
        self.obs: Dict[int, List[Obs]] = defaultdict(list)
        self.cleared: set = set()
        self.pos = np.zeros(2)
        self.vt = 0.0
        self.n_measure = 0
        self.n_clear = 0
        self.deadline = float("inf")
        self.localizer = GeneticLocalizer(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.episode = episode      # 局号（用于训练记录分组）
        self.ga_runs: List[Dict[str, Any]] = []     # GA 训练记录（逐次调用一条）
        self.final_est: Dict[int, Estimate] = {}    # 本局各频道的最终定位解

    # ---- 日志 ----
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
        if self._logfile:
            print(msg, file=self._logfile, flush=True)

    def close(self) -> None:
        """关闭日志文件（幂等；run() 已在 finally 中调用）。"""
        if self._logfile:
            self._logfile.close()
            self._logfile = None

    # ---- 原子动作 ----
    def measure(self, x: float, y: float, channel: int) -> dict:
        """测向；返回原始响应，direction 时自动记录示向度。"""
        r = self.sim.measure(float(x), float(y), channel)
        if not r.get("accepted"):
            raise RuntimeError(f"/measure 被拒绝：{r}")
        self.pos, self.vt = np.array([float(x), float(y)]), float(r["virtual_time_s"])
        self.n_measure += 1
        if r.get("measure_result") == "direction":
            self.obs[channel].append(Obs(channel, float(x), float(y), float(r["svd_deg"])))
        return r

    def clear(self, x: float, y: float, channel: int) -> bool:
        """清除；返回是否成功。"""
        r = self.sim.clear(float(x), float(y), channel)
        if not r.get("accepted"):
            raise RuntimeError(f"/clear 被拒绝：{r}")
        self.pos, self.vt = np.array([float(x), float(y)]), float(r["virtual_time_s"])
        self.n_clear += 1
        ok = r.get("clear_result") == "success"
        if ok:
            self.cleared.add(channel)
        return ok

    def _out_of_time(self) -> bool:
        return time.monotonic() > self.deadline

    def _plan_route(self, pts: np.ndarray, label: str) -> Tuple[List[int], float]:
        """路线 GA 规划从当前位置出发访问 pts 的顺序；返回（访问顺序, 总里程 m）。

        顺带记录本次训练（进化）过程；里程在这里一次算好，供调用方直接使用。
        """
        seed = int(self._rng.integers(1 << 31))
        record: List[List[float]] = []
        order = route_ga(pts, self.pos, seed=seed, record=record)
        length = _path_len(order, _dist_matrix(pts, self.pos))
        first, last = record[0], record[-1]
        self.ga_runs.append({
            "episode": self.episode, "ga": "route", "label": label, "seed": seed,
            "n_points": len(pts), "initial_best_g0": first[1],
            "generations": int(last[0]), "final_length_m": length,
            "solution": order, "history": record,
        })
        return order, length

    def _sweep(self, channels: Sequence[int], at: Sequence[float]) -> Dict[str, int]:
        """在 at 处逐频道测向（按频道号升序以减少切换）；近距则就地清除。"""
        counts = {"direction": 0, "near": 0, "no_signal": 0}
        for ch in sorted(channels):
            if self._out_of_time():
                break
            res = self.measure(at[0], at[1], ch).get("measure_result", "no_signal")
            counts[res] = counts.get(res, 0) + 1
            if res == "near" and self.clear(at[0], at[1], ch):
                self.log(f"    [near] 频道{ch} 距离过近，就地清除成功")
        return counts

    def _probe(self, channel: int, candidates: Sequence[Sequence[float]]) -> bool:
        """依次在候选点测向：取到 direction 或就地清除后即停；返回是否新增示向度。"""
        for x, y in candidates:
            if self._out_of_time():
                return False
            if max(abs(x), abs(y)) > COORD_LIMIT:      # 超出协议允许的坐标范围
                continue
            res = self.measure(x, y, channel).get("measure_result", "no_signal")
            if res == "direction":
                return True
            if res == "near":
                self.clear(x, y, channel)
                return False
        return False

    def _estimate(self, channel: int) -> Optional[Estimate]:
        """定位 GA 求解该频道源位置，并给出位置 1σ；同时记录本次训练（进化）过程。"""
        ol = self.obs.get(channel, ())
        if len(ol) < 2:
            return None
        record: List[List[float]] = []
        G = self.localizer.localize(ol, record=record)
        if G is None:
            return None
        est = Estimate(float(G[0]), float(G[1]),
                       position_sigma(GeneticLocalizer.covariance(ol, G)))
        first, last = record[0], record[-1]
        self.ga_runs.append({
            "episode": self.episode, "ga": "localize", "label": f"频道{channel}",
            "seed": self.localizer.seed, "channel": channel, "n_obs": len(ol),
            "generations": int(last[0]),
            "converged": bool(last[1] < CONVERGED_FITNESS),
            "initial_best_g0": first[1], "final_fitness": last[1],
            "residual_rms_deg": self._residual(est.point, ol),
            "sigma_m": est.sigma, "solution": [est.x, est.y], "history": record,
        })
        return est

    @staticmethod
    def _residual(G: Sequence[float], obs_list: Sequence[Obs]) -> float:
        return float(np.sqrt(np.mean([ang_diff(bearing((o.x, o.y), G), o.theta) ** 2
                                       for o in obs_list])))

    # ---- 阶段 1：起点完整扫描 ----
    def _initial_scan(self) -> None:
        self.log(f"阶段1：起点完整扫描全部 {len(CHANNELS)} 个频道 @ "
                 f"({self.pos[0]:.0f}, {self.pos[1]:.0f})")
        counts = self._sweep(CHANNELS, self.pos)
        self.log(f"  有示向度 {counts['direction']} 个 {sorted(self.obs)}，"
                 f"近距清除 {counts['near']} 个，无信号 {counts['no_signal']} 个")

    # ---- 阶段 2：覆盖观测 ----
    def _coverage_survey(self) -> None:
        waypoints = covering_waypoints()
        route_idx, _ = self._plan_route(waypoints, "覆盖路点巡回")
        route = waypoints[route_idx]
        self.log(f"阶段2：覆盖观测，{len(route)} 个路点（GA 定序，覆盖半径 {COVER_RADIUS:.0f} m）")
        for wp in route:
            active = [c for c in CHANNELS if c not in self.cleared
                      and len(self.obs.get(c, ())) < OBS_PER_SOURCE]
            if not active or self._out_of_time():
                break
            self._sweep(active, wp)
        self.log(f"  观测结束，已探测频道 {len(self.obs)} 个")

    # ---- 阶段 3：定位 ----
    def _single_candidates(self, channel: int) -> List[Tuple[float, float]]:
        """单射线补测点：沿示向度前移并带侧偏，保证仍在接收范围内且拉开交会角。"""
        o = self.obs[channel][0]
        return [(o.x + t * math.cos(math.radians(o.theta + s * beta)),
                 o.y + t * math.sin(math.radians(o.theta + s * beta)))
                for t, beta in SINGLE_PROBES for s in (1.0, -1.0)]

    def _perp_candidates(self, channel: int, e: Estimate) -> List[Tuple[float, float]]:
        """几何病态时的补测点：沿"良态方向"外移。

        近共线时取公共直线的法向（新射线与原直线相交即可定距）；否则取位置协方差最小
        特征值方向，即当前最不确定的方向。
        """
        ol = self.obs[channel]
        if max_ray_sine(ol) < GEOM_SIN_MIN:
            t2 = 2.0 * np.radians([o.theta for o in ol])
            orient = 0.5 * math.atan2(float(np.sin(t2).mean()), float(np.cos(t2).mean()))
            nx, ny = -math.sin(orient), math.cos(orient)
        else:
            _, vecs = np.linalg.eigh(GeneticLocalizer.covariance(ol, e.point))
            nx, ny = float(vecs[0, 0]), float(vecs[1, 0])
        return [(e.x + r * nx, e.y + r * ny) for r in PERP_STEPS]

    def _resolve_singles(self) -> None:
        """对只有单条射线、无法定距的频道补测第二视角。"""
        singles = [c for c in sorted(self.obs) if c not in self.cleared and len(self.obs[c]) == 1]
        if not singles:
            return
        self.log(f"阶段3a：单射线频道补测第二视角 {singles}")
        for c in singles:
            self._probe(c, self._single_candidates(c))

    def _fix_geometry(self, channel: int) -> bool:
        """交会几何病态时补测垂直视角，最多 3 轮；返回是否有补测。"""
        probed = False
        for _ in range(3):
            e = self._estimate(channel)
            ol = self.obs.get(channel, ())
            if e is None or len(ol) < 2:
                return probed
            if e.sigma <= GEOM_SIGMA and max_ray_sine(ol) >= GEOM_SIN_MIN:
                return probed
            if not self._probe(channel, self._perp_candidates(channel, e)):
                return probed
            probed = True
        return probed

    def localize_all(self) -> Dict[int, Estimate]:
        """对每个已积累足够示向度的频道跑定位 GA，并修掉病态几何。"""
        self._resolve_singles()
        est: Dict[int, Estimate] = {}
        for c in sorted(self.obs):
            if c in self.cleared:
                continue
            e = self._estimate(c)
            if e is None:
                continue
            if e.sigma > GEOM_SIGMA or max_ray_sine(self.obs[c]) < GEOM_SIN_MIN:
                if self._fix_geometry(c):
                    e = self._estimate(c) or e
            est[c] = e
        if est:
            rms = float(np.mean([self._residual(e.point, self.obs[c]) for c, e in est.items()]))
            self.log(f"阶段3：GA 定位 {len(est)} 个源，示向度残差 RMS = {rms:.2f}°")
        return est

    # ---- 阶段 4：清除 ----
    def _home_and_clear(self, channel: int, est: Dict[int, Estimate]) -> None:
        """靠近并清除：先按定位解 /clear，未成功则沿最新实测示向度逐步逼近。

        示向度误差是"同一地点固定"的系统误差，仅靠多视角交会存在沿射线方向的偏移；
        靠近后直接沿最新示向度走一步（16 m 步长的横向误差约 0.3 m）即可稳定进入清除半径。
        """
        for _ in range(HOMING_MAX):
            e = est.get(channel)
            if e is None or self._out_of_time():
                return
            if self.clear(e.x, e.y, channel):
                self.log(f"    [清除] 频道{channel} 成功 @ ({e.x:.0f}, {e.y:.0f})")
                return
            r = self.measure(e.x, e.y, channel)
            res = r.get("measure_result")
            if res == "near":
                if self.clear(e.x, e.y, channel):
                    self.log(f"    [清除] 频道{channel} 近距命中")
                return
            if res == "direction":
                th = math.radians(float(r["svd_deg"]))
                nx, ny = clamp_to_region(e.x + HOMING_STEP * math.cos(th),
                                         e.y + HOMING_STEP * math.sin(th))
                est[channel] = Estimate(nx, ny, e.sigma)
                continue
            # no_signal：定位偏了，用新示向度重新定位；仍不行则补测视角
            ne = self._estimate(channel)
            if ne is None or dist(ne.point, e.point) <= 3.0:
                if len(self.obs[channel]) == 1:
                    self._probe(channel, self._single_candidates(channel))
                ne = self._estimate(channel)
            if ne is not None:
                est[channel] = ne
        self.log(f"    [警告] 频道{channel} 逼近 {HOMING_MAX} 次仍未能清除")

    def _clear_all(self, est: Dict[int, Estimate]) -> None:
        todo = [c for c in sorted(est) if c not in self.cleared]
        if not todo:
            return
        pts = np.array([est[c].point for c in todo])
        order, length = self._plan_route(pts, "清除顺序")
        self.log(f"阶段4：GA 规划 {len(todo)} 个源的清除顺序，路程 {length:.0f} m")
        for k, i in enumerate(order):
            if self._out_of_time():
                self.log(f"    [警告] 现实时间不足，剩余 {len(order) - k} 个源未处理")
                return
            self._home_and_clear(todo[i], est)

    # ---- 主流程 ----
    def run(self) -> Dict[str, Any]:
        """跑完一局：/enter → 起点扫描 → 覆盖观测 → GA 定位 → GA 规划清除 → /exit。"""
        self.log("=" * 74)
        self.log("策略：遗传算法（定位 GA + 路线 GA）")
        enter = self.sim.enter()
        if not enter.get("accepted"):
            raise RuntimeError(f"/enter 被拒绝：{enter}")
        left = float(enter.get("remaining_real_duration_s", 1200.0))
        self.deadline = time.monotonic() + max(left - SAFETY_MARGIN, 0.0)
        self.log(f"/enter 成功：虚拟时刻 {enter.get('virtual_time_s')} s，现实剩余 {left:.0f} s")

        try:
            self._initial_scan()
            self._coverage_survey()
            self.final_est = self.localize_all()    # 留存最终定位解，供与真值比对
            self._clear_all(self.final_est)
        finally:
            try:
                self.sim.exit()                     # 无论成功与否都要正常退出，保住测试记录
            except OSError as exc:
                self.log(f"    [警告] /exit 失败：{exc}")
            self.close()                            # 日志句柄随本局一并关闭

        n = len(self.cleared)
        return {
            "cleared": n,
            "total_time_s": self.vt,
            "avg_time_s": self.vt / n if n else float("inf"),
            "n_measure": self.n_measure,
            "n_clear": self.n_clear,
        }


# ----------------------------------------------------------------------------
# 本地演练场：拉起 jammers-py 并用其控制台 REST 开固定场景的一局
# ----------------------------------------------------------------------------
def _free_port(preferred: int) -> int:
    """取一个可用端口：优先 preferred，被占用则依次向后试 20 个。"""
    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:      # 无进程监听
                return port
    raise RuntimeError(f"端口 {preferred}~{preferred + 19} 都被占用")


class PracticeArena:
    """本地演练用的 jammers-py 模拟器：自动拉起进程（或复用已在运行的实例）+ 控制台 REST。"""

    REUSE_PORTS = (8090, 8080)      # 探测已有 jammers-py 的控制台端口（8080 为其默认端口）

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

    # ---- 生命周期 ----
    def __enter__(self) -> "PracticeArena":
        existing = self._find_existing()
        if existing is not None:
            self.console_url, state = existing
            # 复用已在运行的 jammers-py：机器狗接口以它的实际配置为准
            self.robot_port = int(state.get("config", {}).get("robot_port", self.robot_port))
            self.robot_url = f"http://127.0.0.1:{self.robot_port}"
            print(f"检测到已在运行的 jammers-py（控制台 {self.console_url}），"
                  f"直接复用，退出时不关闭它")
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
    def _request(self, path: str, payload: Optional[dict] = None,
                 base: Optional[str] = None, post: bool = False) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request((base or self.console_url) + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if post or payload is not None else "GET")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _state(self, base: str) -> Optional[dict]:
        """读取控制台状态；该地址不是 jammers-py 或未启动时返回 None。"""
        try:
            state = self._request("/api/state", base=base)
            return state if "state" in state else None
        except Exception:
            return None

    def _find_existing(self) -> Optional[Tuple[str, dict]]:
        """探测是否已有 jammers-py 在运行（端口按常驻优先顺序）。"""
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

        jammers-py 的 `generate_scenario` 只把 seed 用于干扰源布局，示向度噪声种子
        `noise_seed_hex` 是每次随机生成的 —— 于是同一个 seed 反复演练，噪声实现也不同。
        这里把噪声种子覆写成由 seed 派生的确定值，使整局（布局 + 噪声）完全可复现：
        同一个 `--seed` 必然得到同一份训练结果。
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
# 训练结果落盘：GA 逐代进化记录 + 逐局统计
# ----------------------------------------------------------------------------
def save_training_results(save_dir: Path, episodes: List[dict], ga_runs: List[dict],
                          meta: dict) -> List[Path]:
    """把 GA 训练结果写盘，返回生成的文件路径。

    - `ga_training.json`：训练配置、逐局统计、每次 GA 调用的摘要与最终解（不含逐代明细）
    - `ga_convergence.csv`：逐代收敛曲线（局号, GA 类型, 对象, 代数, 最优/平均/标准差）
    - `episodes.csv`：逐局战绩（清除数、清除比例、虚拟时间、测向次数）
    文件名固定，重复运行直接覆盖，便于论文与后续绘图脚本稳定引用。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "ga_training.json"
    curve_path = save_dir / "ga_convergence.csv"
    episodes_path = save_dir / "episodes.csv"

    summary = [{k: v for k, v in r.items() if k != "history"} for r in ga_runs]
    json_path.write_text(json.dumps({"meta": meta, "episodes": episodes, "ga_runs": summary},
                                    ensure_ascii=False, indent=2), encoding="utf-8")

    with curve_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["episode", "ga", "label", "seed", "generation",
                         "best", "mean", "std"])
        for r in ga_runs:
            for gen, best, mean, std in r["history"]:
                writer.writerow([r["episode"], r["ga"], r["label"], r["seed"],
                                 gen, f"{best:.6g}", f"{mean:.6g}", f"{std:.6g}"])

    keys = ["episode", "seed", "n_sources", "cleared", "clear_ratio", "virtual_time_s",
            "avg_time_s", "n_measure", "n_clear", "ga_runs", "localize_rms_deg",
            "n_located", "localize_err_mean_m", "localize_err_max_m"]
    with episodes_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(episodes)
    return [json_path, curve_path, episodes_path]


def print_ga_summary(ga_runs: Sequence[dict]) -> None:
    """打印 GA 训练摘要：定位 GA 的收敛代数与残差、路线 GA 的优化幅度。"""
    if not ga_runs:
        print("GA 训练摘要：本次没有产生训练记录")
        return
    loc, route = _split_runs(ga_runs)
    if loc:
        rms = np.array([r["residual_rms_deg"] for r in loc], dtype=float)
        fit = np.array([r["final_fitness"] for r in loc], dtype=float)
        n_conv = sum(1 for r in loc if r["converged"])
        print(f"定位 GA：{len(loc)} 次训练（每代 {GA_LOC.pop} 个体），"
              f"提前收敛 {n_conv}/{len(loc)} 次（判据 适应度<{CONVERGED_FITNESS:.0e}），"
              f"否则跑满 {GA_LOC.gens} 代；最终适应度均值 {fit.mean():.3f}°，"
              f"示向度残差 RMS 均值 {rms.mean():.2f}°")
    for r in route:
        g0, g1 = r["initial_best_g0"], r["final_length_m"]
        gain = (g0 - g1) / g0 * 100 if g0 else 0.0
        print(f"路线 GA：{r['label']}（{r['n_points']} 点）{g0:.0f} m → {g1:.0f} m，"
              f"改进 {gain:.1f}%")


def _ga_meta() -> Dict[str, dict]:
    """训练记录里的 GA 超参数快照，便于复现实验。"""
    return {
        "ga_loc": {**asdict(GA_LOC), "domain_m": GA_LOC_DOMAIN,
                   "record_stride": GA_RECORD_STRIDE},
        "ga_route": {**asdict(GA_ROUTE), "record_stride": GA_RECORD_STRIDE},
    }


def _mean_rms(ga_runs: Sequence[dict]) -> Optional[float]:
    """本局所有定位 GA 的示向度残差均值（度）。"""
    rms = [r["residual_rms_deg"] for r in _split_runs(ga_runs)[0]]
    return float(np.mean(rms)) if rms else None


def _episode_row(episode: int, seed: int, truth: Optional[List[dict]], dog: RobotDog,
                 stats: Dict[str, Any], engine: Optional[dict] = None) -> dict:
    """汇总一局战绩，供 CSV/JSON 落盘。

    演练模式传模拟器真值 `truth` 与引擎统计 `engine`，并逐源算定位误差；官方模式真值
    不可得（接口不返回），相关字段为 None。
    """
    engine = engine or {}
    cleared = int(engine.get("cleared_jammer_count", stats["cleared"]))
    total_time = float(engine.get("virtual_time_s", stats["total_time_s"]))
    errors = [dist((e.x, e.y), (j["position"]["x"], j["position"]["y"]))
              for j in (truth or []) if (e := dog.final_est.get(j["channel"]))]
    return {
        "episode": episode, "seed": seed, "n_sources": len(truth) if truth else None,
        "cleared": cleared, "clear_ratio": cleared / len(truth) if truth else None,
        "virtual_time_s": total_time,
        "avg_time_s": total_time / cleared if cleared else None,
        "n_measure": int(engine.get("measure_accepted_count", stats["n_measure"])),
        "n_clear": stats["n_clear"], "ga_runs": len(dog.ga_runs),
        "localize_rms_deg": _mean_rms(dog.ga_runs),
        "n_located": len(errors),
        "localize_err_mean_m": float(np.mean(errors)) if errors else None,
        "localize_err_max_m": float(np.max(errors)) if errors else None,
        "truth": [{"channel": j["channel"], "x": j["position"]["x"], "y": j["position"]["y"],
                   "receive_m": j["receive"]} for j in (truth or [])],
    }


def _save_and_report(args: argparse.Namespace, episodes: List[dict], ga_runs: List[dict],
                     meta: dict, api_log: Optional[ApiLog] = None) -> None:
    """打印 GA 训练摘要，写出训练结果，并汇总接口调用日志（四个产物路径并列在尾部）。"""
    print_ga_summary(ga_runs)
    paths = save_training_results(Path(args.save_dir), episodes, ga_runs, meta)
    print("训练结果已保存：" + "，".join(str(p) for p in paths))
    if api_log is not None:
        api_log.report()


def run_official(args: argparse.Namespace) -> int:
    """官方评测平台的正式流程：连 127.0.0.1 上已开放接口的模拟器，跑完一局。

    与演练的关键差别：**拿不到干扰源真值**，因此定位误差等需要真值的指标留空；
    结果默认写到 results/official/，不覆盖演练训练批的数据。
    """
    if args.save_dir is None:
        args.save_dir = str(Path(RESULTS_DIR) / "official")
    sim = sim_api.Simulator(robot_id=args.robot_id, base_url=args.base_url, timeout=args.timeout)
    print(f"连接模拟器 {args.base_url}（robot_id={args.robot_id}）")
    with _api_log(args, not args.quiet) as api_log:
        dog = RobotDog(sim, verbose=not args.quiet, logfile=args.log, seed=args.seed,
                       episode=1, api_log=api_log)
        stats = dog.run()
        print(f"完成：清除 {stats['cleared']} 个，虚拟总时间 {stats['total_time_s']:.1f} s，"
              f"平均 {stats['avg_time_s']:.1f} s/个，测向 {stats['n_measure']} 次，"
              f"清除动作 {stats['n_clear']} 次")
        row = _episode_row(1, args.seed, None, dog, stats)
        meta = {"mode": "official", "base_url": args.base_url, "robot_id": args.robot_id,
                "seed": args.seed, **_ga_meta()}
        _save_and_report(args, [row], dog.ga_runs, meta, api_log)
    return 0


def run_practice(args: argparse.Namespace) -> int:
    if args.save_dir is None:
        args.save_dir = RESULTS_DIR
    """本地演练：自动拉起 jammers-py，跑 N 局场景并汇总。

    第 i 局用 seed=args.seed+i：场景布局与示向度噪声都由它确定，因此同一 --seed 的整轮
    演练完全可复现（含各次 GA 的解），可用于新旧策略的严格对比。
    """
    jammers_dir = Path(args.jammers_dir) if args.jammers_dir else Path(__file__).parent / "jammers-py"
    rows: List[dict] = []
    ga_runs: List[dict] = []
    with _api_log(args, not args.quiet) as api_log, \
            PracticeArena(jammers_dir, robot_id=args.robot_id,
                          console_port=args.console_port) as arena:
        print(f"jammers-py 已就绪：机器狗接口 {arena.robot_url}，控制台 {arena.console_url}")
        for ep in range(args.practice):
            seed = args.seed + ep
            truth = arena.start_episode(seed)
            print(f"\n----- 演练第 {ep + 1}/{args.practice} 局（seed={seed}，"
                  f"干扰源 {len(truth)} 个）-----")
            dog = RobotDog(sim_api.Simulator(robot_id=args.robot_id, base_url=arena.robot_url,
                                             timeout=args.timeout),
                           verbose=not args.quiet, logfile=args.log, seed=seed, episode=ep + 1,
                           api_log=api_log)
            stats = dog.run()
            ga_runs.extend(dog.ga_runs)
            engine = arena.finish_episode()
            rows.append(_episode_row(ep + 1, seed, truth, dog, stats, engine))
            r = rows[-1]
            avg_txt = f"{r['avg_time_s']:.1f} s/个" if r["avg_time_s"] else "—"
            err_txt = (f"定位误差 {r['localize_err_mean_m']:.1f} m 均值 / "
                       f"{r['localize_err_max_m']:.1f} m 最大"
                       if r["n_located"] else "无定位误差数据")
            print(f"本局：清除 {r['cleared']}/{r['n_sources']}（{r['clear_ratio']:.3f}），"
                  f"虚拟总时间 {r['virtual_time_s']:.1f} s，平均 {avg_txt}，"
                  f"测向 {r['n_measure']} 次，{err_txt}")
    print("\n" + "=" * 74)
    avgs = [r["avg_time_s"] for r in rows if r["avg_time_s"]]
    print(f"汇总（{len(rows)} 局）：平均清除比例 {np.mean([r['clear_ratio'] for r in rows]):.4f}，"
          + (f"平均 {np.mean(avgs):.1f} s/个，" if avgs else "")
          + f"平均虚拟总时间 {np.mean([r['virtual_time_s'] for r in rows]):.1f} s")
    print("逐局：" + "  ".join(f"seed{r['seed']}={r['cleared']}/{r['n_sources']}" for r in rows))
    print("=" * 74)
    meta = {"mode": "practice", "robot_id": args.robot_id, "seed0": args.seed,
            "episodes": args.practice, **_ga_meta(),
            "coverage_waypoints": len(covering_waypoints())}
    _save_and_report(args, rows, ga_runs, meta, api_log)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2026 CUMCM B 题问题三：遗传算法机器狗自动定位与清除")
    p.add_argument("--robot-id", default=sim_api.ROBOT_ID, help="参赛队号（须与模拟器一致）")
    p.add_argument("--base-url", default=sim_api.BASE_URL, help="官方模拟器地址")
    p.add_argument("--timeout", type=float, default=5.0, help="HTTP 超时 / s")
    p.add_argument("--practice", type=int, nargs="?", const=1, default=0,
                   help="本地演练局数：自动拉起 jammers-py 跑 N 局（缺省 1 局）")
    p.add_argument("--jammers-dir", default=None, help="jammers-py 目录（缺省为本脚本旁的 jammers-py/）")
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
    p.add_argument("--quiet", action="store_true", help="只输出汇总，不打印过程")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
