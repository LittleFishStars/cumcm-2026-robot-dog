"""遗传算法二：巡视路线（排列编码，开路径 TSP）。

与确定性方案的差别：这里用 GA 搜索访问顺序，而不是"最近邻 + 2-opt"。算子：顺序交叉（OX）
+ 倒位/交换变异，适应度为开路径总里程。`two_opt` 作为独立的精修算子，GA 收敛后仍会调用一次。

`_dist_matrix` / `_path_len` / `_path_len_vec` / `_nearest_order` / `two_opt` 全部复用
cumcm.common.routing 的实现（原先与 T3.py 各有一份重复代码），这里只保留 GA 专有的
`_ox` / `_mutate` / `route_ga`。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from cumcm.common.routing import dist_matrix as _dist_matrix
from cumcm.common.routing import nearest_order_from_D as _nearest_order
from cumcm.common.routing import path_len as _path_len
from cumcm.common.routing import path_len_vec as _path_len_vec
from cumcm.common.routing import two_opt_first as two_opt
from cumcm.t3ga.ga_ops import _pick, _record
from cumcm.t3ga.config import (
                              GAParams, GA_RECORD_STRIDE, GA_ROUTE, SEED)

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
