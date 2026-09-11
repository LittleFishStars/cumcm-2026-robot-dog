"""覆盖路点：贪心集合覆盖（保证圆域内任意点都在某路点 COVER_RADIUS 内）。

与确定性方案的 7 圆正六边形不同，这里用贪心集合覆盖自动求路点数量（本参数下得 8 个）。
两者的覆盖保证都在**连续圆域**上校验（网格上"看起来满足"不等于满足）。
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from cumcm.t3ga.config import COVER_GRID, COVER_RADIUS, COVER_STEP, REGION_RADIUS

def _disk_grid(radius: float, step: float) -> np.ndarray:
    ax = np.arange(-radius, radius + 1e-9, step)
    gx, gy = np.meshgrid(ax, ax)
    pts = np.stack((gx.ravel(), gy.ravel()), axis=1)
    return pts[np.linalg.norm(pts, axis=1) <= radius + 1e-9]


@lru_cache(maxsize=None)
def covering_waypoints(region_radius: float = REGION_RADIUS,
                       grid: float = COVER_GRID,
                       step: float = COVER_STEP,
                       cover_radius: float = COVER_RADIUS) -> np.ndarray:
    """用尽量少的路点覆盖整个目标圆域（每轮取"新增覆盖目标点"最多的候选路点）。

    四个离散化/覆盖参数**显式传入**（缺省取 config 里的常量）。原先这四个量直接读模块全局
    变量，验证脚本要扫描不同覆盖半径只能去改模块属性再清缓存 —— 那种隐式耦合一旦代码分层
    就会静默失效（实测：改的是别处的同名属性，扫描结果全都一样却不报错）。显式参数既让
    调用方一目了然，也让 lru_cache 能按参数分别缓存。
    """
    targets = _disk_grid(region_radius, grid)
    centers = _disk_grid(region_radius, step)
    reach = np.linalg.norm(centers[:, None, :] - targets[None, :, :], axis=2) <= cover_radius
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
