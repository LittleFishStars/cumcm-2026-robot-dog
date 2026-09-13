"""按文献准则选补测点，用 Fisher 信息和交会几何

文献依据是任叶童(2016) 的 AOA 双站交会模糊区面积公式、多站加权最小二乘权重 1/(σᵢRᵢ)，
以及 Chen et al.(2009, ICICS) 那句"均方位置误差 ∝ 1/(σ²r²)，最优轨迹兼顾贴近与角度分集"。
本模块把它们落成两个能算的东西：

* `fisher_sigma`：由 Fisher 信息矩阵给出位置的 1σ，J = Σ(1/σ²)(1/rᵢ²)nᵢnᵢᵀ，σ_pos = √tr(J⁻¹)。
  两条等距观测在夹角 90° 时最优，此时 tr(J⁻¹) = 2σ²r²/sin²φ。
* `probe_candidates`：枚举候选补测点并排序。先剔掉与已有观测共线的，那种交会已经退化；再按
  对全部假设源位置的平均 σ 排序，相差 2% 以内算并列，并列时取路近的那个；最后按字典序打破
  剩余并列，保证结果确定。
* `ambiguity_area`：文献公式本身，拿来和 Fisher σ 并列对照，论文里解释"为什么这样选点"。

`hypothesis_points` 给出"源可能在哪里"的代表点：优先从当前定位区域里取，最小覆盖圆圆心加
区域顶点的最远点采样；区域退化时才回退到沿单条射线枚举距离。
"""

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

    测向的观测方程是 θ = atan2(Δy, Δx) + e，e 的标准差取 σ = 1°。梯度 ∂θ/∂p = n / r，
    其中 n 是示向度方向的单位法向量，r 是检测点到源的距离。于是 Fisher 信息矩阵

        J = Σᵢ (1/σ²) · (1/rᵢ²) · nᵢ nᵢᵀ,      预测协方差 C = J⁻¹,   σ_pos = √tr(C)

    权重 1/(σᵢ²rᵢ²) 与文献对得上：任叶童(2016) 给多站交会的加权最小二乘权为 1/(σᵢRᵢ)；
    Chen 等(2009) 证明观测站离目标越近，均方位置误差越小，正比于 1/(σ²r²)。J 退化时
    单条射线或者近共线时行列式 ≈ 0，直接返回 inf，意思是"定不了距"。
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


def hypothesis_points(region: "ProbRegion",
                      obs_list: Sequence[Obs]) -> list[tuple[float, float]]:
    """补测选点用的"假设源位置"集合

    区域已知时统一取最小覆盖圆圆心加最远点采样出的几个顶点。单条射线也适用，因为叠上
    no_signal 禁区与接收半径环带之后，区域不再是无限长的一条带，而是有限的一段。只有区域
    退化、还没有禁区约束时，才退回"沿那条射线按可能距离枚举"。

    Args:
        region: 该频道当前的定位区域，也就是可能源集合
        obs_list: 该频道已有的示向度观测

    Returns:
        list[tuple[float, float]]: 假设源位置列表；区域已知时是最小覆盖圆圆心与最远点采样顶点
    """
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

    候选点是这么撒出来的：每个假设源位置周围取若干半径 PROBE_RADII，都小于 1000 m，保证
    落在源的有效接收半径内，再乘若干方位角 PROBE_ANGLES，得到一圈圈环上的点。评价用
    Fisher 信息口径的预测 σ，一条准则同时兑现了文献的两条结论：检测点越接近源，σ 越小，
    这是 Chen 的定理，均方位置误差正比于 1/(σ²r²)；新射线与已有射线的交角越接近正交，
    σ 也越小，也就是角度分集。单射线时 J 退化，这条准则会自动把补测点放到能"定距"的位置
    上，单射线到双射线的补测顺带完成。

    与已有射线夹角 ≈ 0 的共线候选，σ = ∞，定不了距，直接剔除。单射线频道的假设点本来就排成
    一条直线，若改用"对全部假设取最坏 σ"排序，会退化成"所有候选都不可用"，然后按里程误选到
    共线上的点。这里实测踩过：测了 26 次仍然没把区域缩小。取平均 σ 既保住了"靠近源加拉开
    交角"的偏好，又不会因为个别极远的假设把好点全部否掉。

    排序规则是确定的：先按平均 σ 升序；与最优值相差 2% 以内的候选算"同等好"，其中取里程
    最短的，省时间；里程并列时取坐标字典序最小的。返回前 PROBE_TRY 个候选。
    """
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

        S = 4·R²·Δθ²·sinα₁·sinα₂ / sin³(α₁+α₂)

    R 是两检测点基线，αᵢ 是基线两端观测站处的内角，Δθ 是测向误差半宽，按弧度算。该式在
    "基线 R 固定、目标位置自由"的口径下取最小值，与本文"检测点自由、最小化预测协方差"的
    口径不是一回事。函数留着，论文里好把两种准则的结论一起给出来。
    """
    a1, a2 = math.radians(alpha1_deg), math.radians(alpha2_deg)
    s = math.sin(a1) * math.sin(a2)
    den = math.sin(a1 + a2) ** 3
    return 4.0 * baseline ** 2 * math.radians(err_deg) ** 2 * s / den
