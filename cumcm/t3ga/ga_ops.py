"""两个 GA 共用的基础算子：逐代记录、选择与锦标赛。

定位 GA（实数编码）与路线 GA（排列编码）除了交叉/变异不同，其余骨架一致，故把这三件
共用的事抽出来，避免在两边各写一份（原先就重复了一份）。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

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
