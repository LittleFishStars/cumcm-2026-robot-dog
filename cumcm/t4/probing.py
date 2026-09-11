"""按文献准则选补测点（Fisher 信息 / 交会几何）——从问题三复制适配。

本模块与 cumcm.t3.probing 同源（问题四的定位几何相同：direction 示向度误差 ±1°，交会收敛
规律不变），仅调整导入到问题四自己的 config/regions。出处：`cumcm.t3.probing`（任叶童
(2016) AOA 双站交会模糊区面积公式与多站加权最小二乘权重 1/(σᵢRᵢ)；Chen et al. (2009,
ICICS) 均方位置误差 ∝ 1/(σ²r²)、最优轨迹兼顾贴近与角度分集）。

* `fisher_sigma`   —— Fisher 信息矩阵给出的位置 1σ：J = Σ(1/σ²)(1/rᵢ²)nᵢnᵢᵀ，σ_pos = √tr(J⁻¹)。
* `probe_candidates` —— 枚举候选补测点并排序：剔除与已有观测共线的（交会退化），按对全部
  假设源位置的平均 σ 排序，2% 以内视为并列则取路近者，按字典序打破剩余并列（确定性）。
* `ambiguity_area` —— 文献公式本身，用于与 Fisher σ 并列对照。

与问题三的差异：问题四的定位区域不含 no_signal 硬约束（见 DirProbRegion），假设点集合的来源
变为"最小覆盖圆圆心 + 距已有检测点最远的若干顶点 + 单射线时沿射线按可能距离枚举"，规则与本
模块文档一致；补测点本身仍要求落在作业圆域内、与已有检测点保持间距。
"""

from __future__ import annotations

import math
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np

from cumcm.common.geometry import bearing, dist
from cumcm.t4.config import (HYP_GAP, HYP_MAX, PROBE_ANGLES, PROBE_GAP, PROBE_RADII,
                             PROBE_TRY, REGION_MARGIN, REGION_RADIUS, SIGMA_DEG,
                             SIGMA_RAD, SINGLE_HYP)
from cumcm.t4.regions import Obs


def fisher_sigma(p: Sequence[float], bearings: Sequence[Sequence[float]]) -> float:
    """在假设源位置 p 处、由一组 (检测点x, 检测点y, 示向度) 预测的位置 1σ / m。

    测向的观测方程 θ = atan2(Δy, Δx) + e，e 的标准差取 σ = 1°。梯度 ∂θ/∂p = n / r
    （n 为示向度方向的单位法向量，r 为检测点到源的距离），故 Fisher 信息矩阵

        J = Σᵢ (1/σ²) · (1/rᵢ²) · nᵢ nᵢᵀ,      预测协方差 C = J⁻¹,   σ_pos = √tr(C)

    权重 1/(σᵢ²rᵢ²) 与文献一致：任叶童（2016）多站交会的加权最小二乘权为 1/(σᵢRᵢ)；
    Chen 等（2009）证明观测站到目标的距离越近、均方位置误差越小（∝ 1/(σ²r²)）。
    J 退化（单条射线、近共线）时行列式 ≈ 0，返回 inf，表示"无法定距"。
    """
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


def hypothesis_points(region, obs_list: Sequence[Obs]) -> List[Tuple[float, float]]:
    """补测选点用的"假设源位置"集合。

    区域已知时统一取"最小覆盖圆圆心 + 距已有检测点最远的若干顶点"（问题四的单条射线经
    direction ⇒ 接收半径内包后区域是一段有限带，verts 恒非空）；区域退化（尚无任何
    direction 约束，理论上不会发生：拖网保证每个源 ≥ 1 次示向）时才回退到沿那条射线按
    可能距离枚举。
    """
    pts: List[Tuple[float, float]] = []
    mec = region.enclosing_circle
    verts = list(region.vertices)
    if verts:
        if mec is not None:
            pts.append((mec[0], mec[1]))
        while len(pts) < HYP_MAX and verts:
            far = max(verts, key=lambda v: min(dist(v, q) for q in pts))
            verts.remove(far)
            pts.append(far)
    elif len(obs_list) == 1:
        o = obs_list[0]
        for s in SINGLE_HYP:
            hx = o.x + s * math.cos(math.radians(o.theta))
            hy = o.y + s * math.sin(math.radians(o.theta))
            if math.hypot(hx, hy) <= REGION_RADIUS:
                pts.append((hx, hy))
    keep: List[Tuple[float, float]] = []
    for h in pts:
        if any(dist(h, (o.x, o.y)) < HYP_GAP for o in obs_list):
            continue
        if any(dist(h, k) < 1.0 for k in keep):
            continue
        keep.append(h)
    return keep


class Probe(NamedTuple):
    """一个补测候选点：坐标、对全部假设源位置的平均预测 σ、从当前位置出发的里程。"""

    x: float
    y: float
    sigma: float
    travel: float


def probe_candidates(obs_list: Sequence[Obs], hyps: Sequence[Sequence[float]],
                     pos: Sequence[float]) -> List[Probe]:
    """按文献准则给补测点排序：最小化"对全部假设源位置的平均预测 σ"。

    候选点 = 每个假设源位置周围若干半径（PROBE_RADII，均 < 1000 m，保证落在源的有效接收
    半径内）× 若干方位角（PROBE_ANGLES）的环上点。评价用 Fisher 信息口径的预测 σ：
    该准则同时实现文献两条结论（接近源 σ 小、与已有射线交角近正交 σ 小；见模块文档）。
    单射线时 J 退化，准则自动把补测点放到能"定距"的位置上（单射线→双射线的补测）。

    共线候选（与已有射线夹角 ≈ 0，σ = ∞）直接剔除；排序规则（确定性）：先按平均 σ 升序，
    σ 与最优相差 2% 以内视为同等好、取里程最短者，里程并列取坐标字典序最小者。
    返回前 PROBE_TRY 个候选。
    """
    base = [(o.x, o.y, o.theta) for o in obs_list]
    cands: List[Probe] = []
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
                for h in hyps:
                    sig = fisher_sigma(h, base + [(qx, qy, bearing((qx, qy), h))])
                    if sig == math.inf:         # 与已有射线共线：不提供任何新信息
                        ok = False
                        break
                    total += sig
                if ok:
                    cands.append(Probe(qx, qy, total / len(hyps), dist(pos, (qx, qy))))
    if not cands:
        return []
    best = min(c.sigma for c in cands)
    good = [c for c in cands if c.sigma <= best * 1.02] if best < math.inf \
        else [c for c in cands if c.sigma == math.inf]
    good.sort(key=lambda c: (c.travel, c.x, c.y))
    rest = sorted((c for c in cands if c not in good),
                  key=lambda c: (c.sigma, c.travel, c.x, c.y))
    return (good + rest)[:PROBE_TRY]


def ambiguity_area(baseline: float, alpha1_deg: float, alpha2_deg: float,
                   err_deg: float = SIGMA_DEG) -> float:
    """文献的定位模糊区面积口径（任叶童 2016 式 2-15，供对照/校核用）：/ m²

        S = 4·R²·Δθ²·sinα₁·sinα₂ / sin³(α₁+α₂)

    其中 R 为两检测点基线，αᵢ 为基线两端观测站处的内角，Δθ 为测向误差半宽（弧度）。
    """
    a1, a2 = math.radians(alpha1_deg), math.radians(alpha2_deg)
    s = math.sin(a1) * math.sin(a2)
    den = math.sin(a1 + a2) ** 3
    return 4.0 * baseline ** 2 * math.radians(err_deg) ** 2 * s / den