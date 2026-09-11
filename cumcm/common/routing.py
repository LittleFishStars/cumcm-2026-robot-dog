"""开路径 TSP 的确定性算子：最近邻构造、路径长度、2-opt 精修。

所有函数都**不含随机性**：并列时按编号取小，故同一输入必得同一输出（可复现性要求）。
"开路径"指从给定起点出发走遍所有点、终点自由、不要求回到起点 —— 本题两阶段的路线都是这种。

关于两个 2-opt 变体
-------------------
T3.py 与 T3_ga.py 原先各写了一份 2-opt，策略不同，而两者的数值结果都已写进报告，故这里
**同时保留**并各自命名，不做合并（合并会改变已公布的里程与时间）：

* `two_opt_first`  ：T3_ga 版。发现第一处改进就翻转并 break 内层循环，再从头重扫。
* `two_opt_greedy` ：T3.py 版。扫描过程中一旦发现更短就把当前解替换掉并继续往后扫（不回退），
                     整轮无改进才结束。

两者的共同前提是"距离矩阵 D 的第 0 行代表起点"（见 `dist_matrix`）。
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

__all__ = ["dist_matrix", "path_len", "path_len_vec", "nearest_order_from_D",
           "nearest_order", "open_path_length", "two_opt_first", "two_opt_greedy"]

_EPS = 1e-9


def dist_matrix(pts: np.ndarray, start: Sequence[float]) -> np.ndarray:
    """距离矩阵 D，形状 (n+1, n)：D[0, j] 为起点到点 j，D[1+i, j] 为点 i 到点 j。"""
    pts = np.asarray(pts, dtype=float)
    nodes = np.vstack((np.asarray(start, dtype=float)[None, :], pts))
    return np.linalg.norm(nodes[:, None, :] - pts[None, :, :], axis=2)


def path_len(order: Sequence[int], D: np.ndarray) -> float:
    """单条顺序的路径长度 / m（D 的第 0 行是起点）。"""
    o = np.asarray(order, dtype=int)
    return 0.0 if len(o) == 0 else float(D[0, o[0]] + D[1 + o[:-1], o[1:]].sum())


def path_len_vec(orders: np.ndarray, D: np.ndarray) -> np.ndarray:
    """整种群路径长度（向量化），orders 形状 (pop, n)。"""
    return D[0, orders[:, 0]] + D[1 + orders[:, :-1], orders[:, 1:]].sum(axis=1)


def nearest_order_from_D(n: int, D: np.ndarray) -> List[int]:
    """最近邻构造初始个体（第 0 行代表起点）；并列时取编号小者。"""
    remaining, cur, order = list(range(n)), 0, []
    while remaining:
        j = min(remaining, key=lambda i: D[cur, i])
        order.append(j)
        cur = 1 + j
        remaining.remove(j)
    return order


def nearest_order(waypoints: np.ndarray, start: Sequence[float] = (0.0, 0.0)) -> List[int]:
    """确定性最近邻访问顺序（并列时取编号小者）：从 start 出发依次走遍所有点。"""
    pts = np.asarray(waypoints, dtype=float)
    rest = list(range(len(pts)))
    order: List[int] = []
    cur = np.asarray(start, dtype=float)
    while rest:
        k = min(rest, key=lambda j: (float(np.linalg.norm(pts[j] - cur)), j))
        order.append(k)
        rest.remove(k)
        cur = pts[k]
    return order


def open_path_length(waypoints: np.ndarray, order: Sequence[int],
                     start: Sequence[float] = (0.0, 0.0)) -> float:
    """按给定顺序走遍各点的总里程 / m（终点自由）。"""
    pts = np.asarray(waypoints, dtype=float)
    total, cur = 0.0, np.asarray(start, dtype=float)
    for j in order:
        total += float(np.linalg.norm(pts[j] - cur))
        cur = pts[j]
    return total


def two_opt_first(order: Sequence[int], D: np.ndarray) -> List[int]:
    """2-opt 精修（第一改进）：发现改进即翻转该段并 break，然后从头重扫。"""
    order = [int(v) for v in order]

    def d(a: int, b: int) -> float:
        return float(D[0, b] if a < 0 else D[1 + a, b])       # a < 0 表示起点

    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            prev = order[i - 1] if i else -1
            for j in range(i + 1, len(order)):
                nxt = order[j + 1] if j + 1 < len(order) else -1
                old = d(prev, order[i]) + (0.0 if nxt < 0 else d(order[j], nxt))
                new = d(prev, order[j]) + (0.0 if nxt < 0 else d(order[i], nxt))
                if new < old - _EPS:
                    order[i:j + 1] = reversed(order[i:j + 1])
                    improved = True
                    break
            if improved:
                break
    return order


def two_opt_greedy(order: Sequence[int], D: np.ndarray) -> List[int]:
    """2-opt 精修（贪心就地替换）：扫描中一旦发现更短就替换当前解并继续往后扫。

    与 `two_opt_first` 的差别只在"发现改进后是 break 重扫还是继续往后扫"，但两者停在不同
    局部最优上，故都保留（见模块文档）。
    """
    order = [int(v) for v in order]

    def d(a: int, b: int) -> float:
        return float(D[0, b] if a < 0 else D[1 + a, b])

    def length(seq: Sequence[int]) -> float:
        total, prev = 0.0, -1
        for c in seq:
            total += d(prev, c)
            prev = c
        return total

    best, cur_len = list(order), length(order)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cand_len = length(cand)
                if cand_len < cur_len - _EPS:
                    best, cur_len, improved = cand, cand_len, True
    return best
