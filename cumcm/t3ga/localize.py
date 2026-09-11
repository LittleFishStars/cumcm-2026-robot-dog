"""遗传算法一：测向定位（实数编码）。

适应度是"个体到各条示向度射线的角度残差平方和"，个体即源位置估计。配合两个几何量使用：
* `ray_intersection`  —— 两条射线直接交会（无噪时精确，共线时返回 None）；
* `position_sigma`    —— 由 Fisher 信息给出的位置 1σ，用于判定交会几何是否病态。

`_split_runs` 把"逐代记录"拆成两段（定位 GA 与路线 GA 的记录混在同一个列表里），供收敛曲线
与逐代统计使用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from cumcm.t3ga.ga_ops import _record, _tournament
from cumcm.t3ga.config import (
                              BEARING_ERROR_DEG, CONVERGED_FITNESS, GAParams,
                              GA_LOC, GA_LOC_DOMAIN, GA_RECORD_STRIDE,
                              MAX_RECEPTION, REGION_RADIUS, SEED)

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
