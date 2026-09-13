"""平面几何工具：距离、方位角、角度差、圆域裁剪，角度一律用度、长度用米"""

from __future__ import annotations

import math
from typing import Sequence

__all__ = ["dist", "bearing", "ang_diff", "clamp_to_region", "point_in_circle"]


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    """两点距离 / m

    用 math.hypot 而不是 np.linalg.norm：两者在极少数调用上会差出 1 µs 量级的浮点误差，
    而虚拟时钟按 5 m/s 精确复算，走 numpy 会造出"时钟对不上"的假不一致。这里踩过。
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bearing(a: Sequence[float], b: Sequence[float]) -> float:
    """a → b 的方位角 / 度，x 轴正向逆时针，取值 [0, 360)"""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def ang_diff(a: float, b: float) -> float:
    """两角的最小绝对差 / 度，取值 [0, 180]"""
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def clamp_to_region(x: float, y: float, radius: float) -> tuple[float, float]:
    """把坐标拉回半径为 radius 的作业圆域内，出域时沿原方向缩到边界

    所有 /measure 与 /clear 的入口都过这里：定位解、补测点、末端归航步进都可能落在圆域外，
    统一在这一处裁掉，内部逻辑就不用各自判断了。radius 由调用方给，通常是
    `REGION_RADIUS - REGION_MARGIN`，留出数值余量。
    """
    r = math.hypot(x, y)
    if r <= radius or r == 0.0:
        return x, y
    k = radius / r
    return x * k, y * k


def point_in_circle(p: Sequence[float], center: Sequence[float], radius: float,
                    tol: float = 1e-9) -> bool:
    """点 p 是否落在以 center 为心、radius 为半径的圆内，含边界"""
    return dist(p, center) <= radius + tol
