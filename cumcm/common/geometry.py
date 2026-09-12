"""平面几何工具：距离、方位角、角度差、圆域裁剪。

这些函数原先在各题入口里各有一份完全相同的实现（`dist`、`bearing`、
`clamp_to_region`），是典型的重复代码；抽取到此处后共用，改动只需一处。

约定：所有角度用**度**，方位角以 x 轴正向为 0、逆时针为正、取值 [0, 360)；
所有长度用**米**。
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

__all__ = ["dist", "bearing", "ang_diff", "clamp_to_region", "point_in_circle"]


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    """两点距离 / m。

    用 math.hypot 而非 np.linalg.norm：两者在极少数调用上会差 1 µs 级别的浮点误差，
    而虚拟时钟是按 5 m/s 精确复算的，用 numpy 会造成"时钟对不上"的假不一致（踩过）。
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bearing(a: Sequence[float], b: Sequence[float]) -> float:
    """a → b 的方位角（度，x 轴正向逆时针，[0, 360)）。"""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def ang_diff(a: float, b: float) -> float:
    """两角的最小绝对差（度），取值 [0, 180]。"""
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def clamp_to_region(x: float, y: float, radius: float) -> Tuple[float, float]:
    """把坐标拉回半径为 radius 的作业圆域内（出域时沿原方向缩到边界）。

    用于所有 /measure 与 /clear 的入口：定位解、补测点、末端归航步进都可能落在圆域之外，
    统一在此裁剪，内部逻辑无需各自判断。radius 由调用方给出（通常是
    `REGION_RADIUS - REGION_MARGIN`，留出数值余量）。
    """
    r = math.hypot(x, y)
    if r <= radius or r == 0.0:
        return x, y
    k = radius / r
    return x * k, y * k


def point_in_circle(p: Sequence[float], center: Sequence[float], radius: float,
                    tol: float = 1e-9) -> bool:
    """点 p 是否落在以 center 为心、radius 为半径的圆（含边界）内。"""
    return dist(p, center) <= radius + tol
