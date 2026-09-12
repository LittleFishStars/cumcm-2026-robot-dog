"""开路径 TSP 的确定性算子：最近邻构造、路径长度、2-opt 精修。

所有函数都**不含随机性**：并列时按编号取小，故同一输入必得同一输出（可复现性要求）。
"开路径"指从给定起点出发走遍所有点、终点自由、不要求回到起点 —— 本题两阶段的路线都是这种。

关于两个 2-opt 变体
-------------------
仓库历史上 T3.py 与已删除的 T3_ga.py 各带一份 2-opt，策略不同，数值结果都已写进报告，故这里
**同时保留**并各自命名，不做合并（合并会改变已公布的里程与时间）：

* `two_opt_first`  ：发现第一处改进就翻转并 break 内层循环，再从头重扫。
* `two_opt_greedy` ：扫描过程中一旦发现更短就把当前解替换掉并继续往后扫（不回退），
                     整轮无改进才结束。

两者都被问题四（sweep / nn_layout）与公共精确算子（exact_open_order 的降级路径）使用。

两者的共同前提是"距离矩阵 D 的第 0 行代表起点"（见 `dist_matrix`）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["dist_matrix", "path_len", "path_len_vec", "nearest_order_from_D",
           "nearest_order", "open_path_length", "two_opt_first", "two_opt_greedy",
           "exact_open_order", "exact_open_by_end"]

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


def nearest_order_from_D(n: int, D: np.ndarray) -> list[int]:
    """最近邻构造初始个体（第 0 行代表起点）；并列时取编号小者。"""
    remaining, cur, order = list(range(n)), 0, []
    while remaining:
        j = min(remaining, key=lambda i: D[cur, i])
        order.append(j)
        cur = 1 + j
        remaining.remove(j)
    return order


def nearest_order(waypoints: np.ndarray, start: Sequence[float] = (0.0, 0.0)) -> list[int]:
    """确定性最近邻访问顺序（并列时取编号小者）：从 start 出发依次走遍所有点。"""
    pts = np.asarray(waypoints, dtype=float)
    rest = list(range(len(pts)))
    order: list[int] = []
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


def two_opt_first(order: Sequence[int], D: np.ndarray) -> list[int]:
    """2-opt 精修（第一改进）：发现改进即翻转该段并 break，然后从头重扫。"""
    order = [int(v) for v in order]

    def d(a: int, b: int) -> float:
        """D 上从 a 到 b 的距离 / m（a < 0 表示起点）"""
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


def two_opt_greedy(order: Sequence[int], D: np.ndarray) -> list[int]:
    """2-opt 精修（贪心就地替换）：扫描中一旦发现更短就替换当前解并继续往后扫。

    与 `two_opt_first` 的差别只在"发现改进后是 break 重扫还是继续往后扫"，但两者停在不同
    局部最优上，故都保留（见模块文档）。
    """
    order = [int(v) for v in order]

    def d(a: int, b: int) -> float:
        """D 上从 a 到 b 的距离 / m（a < 0 表示起点）"""
        return float(D[0, b] if a < 0 else D[1 + a, b])

    def length(seq: Sequence[int]) -> float:
        """整条顺序的路径长度 / m：首点从起点起算，终点自由"""
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


def _held_karp(n: int, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Held-Karp 动态规划，返回 (dp, prev)。

    dp[mask, j] = 从起点出发、访问 mask 中的点、最后停在 j 的最短长度；
    prev[mask, j] = 对应路径上 j 的前一个点（-1 表示起点）。
    mask 的转移按 j 的候选集向量化，故 n=16（65536 个掩码、1.7e7 次松弛）也能秒级完成。

    Args:
        n: 待访问点个数
        D: 距离矩阵，形状 (n+1, n)，第 0 行代表起点

    Returns:
        tuple[np.ndarray, np.ndarray]: (dp, prev) 两张 (2**n, n) 的表
    """
    INF = float("inf")
    full = 1 << n
    bits = np.arange(n)
    dp = np.full((full, n), INF)
    prev = np.full((full, n), -1, dtype=np.int64)
    dp[(1 << bits), bits] = D[0, bits]
    for mask in range(1, full):
        ins = (mask >> bits) & 1
        cur = dp[mask]
        js = np.flatnonzero(ins & (cur < INF))
        if not len(js):
            continue
        ks = np.flatnonzero(~ins.astype(bool))
        if not len(ks):
            continue
        cand = cur[js][:, None] + D[1 + js][:, ks]          # (|js|, |ks|)
        best = np.argmin(cand, axis=0)
        vals = cand[best, np.arange(len(ks))]
        for t, k in enumerate(ks):
            nm = mask | (1 << int(k))
            if vals[t] < dp[nm, k] - _EPS:
                dp[nm, k] = vals[t]
                prev[nm, k] = js[best[t]]
    return dp, prev


def _hk_path(prev: np.ndarray, n: int, last: int) -> list[int]:
    """从 prev 表回溯出路径（不含起点）。"""
    order, mask, j = [], (1 << n) - 1, int(last)
    while j >= 0:
        order.append(j)
        pj = int(prev[mask, j])
        mask ^= (1 << j)
        j = pj
    order.reverse()
    return order


def exact_open_order(n: int, D: np.ndarray, limit: int = 16) -> list[int]:
    """从起点出发访问全部 n 个点、终点任意的**精确**最短开放路径（Held-Karp）。

    返回被访问点的编号序列（与 nearest_order / path_len 的约定一致，**不含**起点占位符）。
    n > limit 时退化为最近邻 + 2-opt（精确解规模 O(2^n·n²)：n=16 约 1.7e7，已实测秒级；
    本题最大的场景是 20 个频道，故 limit 留到 16，超出即降级）。

    确定性：遍历顺序固定；最优值并列时取字典序最小的路径，故同一输入必得同一输出。
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    if n > limit:
        return two_opt_greedy(two_opt_first(nearest_order_from_D(n, D), D), D)
    dp, prev = _held_karp(n, D)
    last = min(range(n), key=lambda j: (dp[(1 << n) - 1, j], j))
    return _hk_path(prev, n, last)


def exact_open_by_end(n: int, D: np.ndarray, limit: int = 16) -> list[tuple[int, float, list[int]]]:
    """返回 (终点, 长度, 顺序) 列表：对每个可能的终点给出从起点出发的最短开放路径。

    用于"终点本身也是决策变量"的场合 —— 例如巡视路线既要短，又希望终点落在一片指定区域
    附近（终点决定后续行程的起点）。n > limit 时退化为最近邻 + 2-opt（只给一个终点）。

    确定性：最优值并列时按 (长度, 终点编号) 排序，故同一输入必得同一输出。

    Args:
        n: 待访问点个数
        D: 距离矩阵，形状 (n+1, n)，第 0 行代表起点
        limit: 精确 DP 的规模上限，超过即退化为启发式

    Returns:
        list[tuple[int, float, list[int]]]: 每项为 (终点编号, 路径长度 / m, 访问顺序)
    """
    if n <= 0:
        return []
    if n == 1:
        return [(0, float(D[0, 0]), [0])]
    if n > limit:
        order = exact_open_order(n, D, limit)
        return [(order[-1], path_len(order, D), order)]
    dp, prev = _held_karp(n, D)
    out = []
    for last in range(n):
        total = float(dp[(1 << n) - 1, last])
        if total == float("inf"):
            continue
        out.append((last, total, _hk_path(prev, n, last)))
    out.sort(key=lambda t: (t[1], t[0]))
    return out
