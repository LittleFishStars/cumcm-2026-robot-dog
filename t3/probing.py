"""按文献准则选补测点，用 Fisher 信息和交会几何"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import numpy as np

from common.geometry import bearing, dist
from t3.config import (HYP_GAP, HYP_MAX, PROBE_ANGLES, PROBE_GAP, PROBE_RADII, PROBE_TRY,
                             REGION_MARGIN, REGION_RADIUS, SIGMA_DEG, SIGMA_RAD, SINGLE_HYP)
from t3.regions import Obs


def fisher_sigma(p: Sequence[float], bearings: Sequence[Sequence[float]]) -> float:
    """在假设源位置 p 处，由一组 (检测点x, 检测点y, 示向度) 预测的位置 1σ / m
    权重取 Fisher 信息口径的 1/(σ²r²)，J 退化时返回 inf 表示定不了距。"""
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


def hypothesis_points(region: "ProbRegion",
                      obs_list: Sequence[Obs]) -> list[tuple[float, float]]:
    """补测选点用的"假设源位置"集合，也就是真源可能落在哪儿
    区域已知时取最小覆盖圆圆心加最远点采样出的顶点，只在区域退化时沿射线枚举可能距离。"""
    pts: list[tuple[float, float]] = []
    mec = region.enclosing_circle
    verts = list(region.vertices)
    if verts:
        # 区域已知，含 no_signal 禁区与接收半径环带：最小覆盖圆心加最远点采样出的几个顶点
        if mec is not None:
            pts.append((mec[0], mec[1]))
        while len(pts) < HYP_MAX and verts:
            far = max(verts, key=lambda v: min(dist(v, q) for q in pts))
            verts.remove(far)
            pts.append(far)
    elif len(obs_list) == 1:
        # 区域退化、还没有禁区约束：只能沿那条射线按可能距离枚举
        o = obs_list[0]
        for s in SINGLE_HYP:
            hx = o.x + s * math.cos(math.radians(o.theta))
            hy = o.y + s * math.sin(math.radians(o.theta))
            if math.hypot(hx, hy) <= REGION_RADIUS:
                pts.append((hx, hy))
    # 离已有检测点太近的假设点估不了距离，r → 0，剔除；顺手去重
    keep: list[tuple[float, float]] = []
    for h in pts:
        if any(dist(h, (o.x, o.y)) < HYP_GAP for o in obs_list):
            continue
        if any(dist(h, k) < 1.0 for k in keep):
            continue
        keep.append(h)
    return keep


class Probe(NamedTuple):
    """一个补测候选点：坐标、对全部假设源位置的最坏预测 σ、从当前位置出发的里程"""

    x: float
    y: float
    sigma: float
    travel: float


def probe_candidates(obs_list: Sequence[Obs], hyps: Sequence[Sequence[float]],
                     pos: Sequence[float]) -> list[Probe]:
    """按文献准则给补测点排序，目标是让"对全部假设源位置的平均预测 σ"最小
    候选点由距离环乘方位角格撒出，评价走 Fisher 信息口径，兼顾离源近与交角接近正交。"""
    base = [(o.x, o.y, o.theta) for o in obs_list]      # 已有观测：检测点加示向度
    cands: list[Probe] = []
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
                for h in hyps:                          # 对全部假设源位置求平均 σ
                    sig = fisher_sigma(h, base + [(qx, qy, bearing((qx, qy), h))])
                    if sig == math.inf:                 # 与已有射线共线，提供不了新信息
                        ok = False
                        break
                    total += sig
                if ok:
                    cands.append(Probe(qx, qy, total / len(hyps), dist(pos, (qx, qy))))
    if not cands:
        return []
    # 与最优 σ 相差 2% 以内算同等好，这类候选里取里程最短的，并列时再取坐标字典序最小
    best = min(c.sigma for c in cands)
    good = [c for c in cands if c.sigma <= best * 1.02] if best < math.inf \
        else [c for c in cands if c.sigma == math.inf]
    good.sort(key=lambda c: (c.travel, c.x, c.y))
    rest = sorted((c for c in cands if c not in good),
                  key=lambda c: (c.sigma, c.travel, c.x, c.y))
    return (good + rest)[:PROBE_TRY]


def ambiguity_area(baseline: float, alpha1_deg: float, alpha2_deg: float,
                   err_deg: float = SIGMA_DEG) -> float:
    """文献的定位模糊区面积口径，任叶童 2016 式 2-15，留给对照与校核用，单位 m²
    该式在"基线固定、目标位置自由"的口径下取最小值，与本文"检测点自由"的口径不同。"""
    a1, a2 = math.radians(alpha1_deg), math.radians(alpha2_deg)
    s = math.sin(a1) * math.sin(a2)
    den = math.sin(a1 + a2) ** 3
    return 4.0 * baseline ** 2 * math.radians(err_deg) ** 2 * s / den
