"""按文献准则选补测点（Fisher 信息 / 交会几何）——**直接复用**问题三的同名模块。

问题四与问题三的定位几何完全相同（direction 示向度误差 ±1°，交会收敛规律不变），且本模块用到的
全部参数在两套 config 中**逐值一致**（SIGMA_RAD、PROBE_RADII/ANGLES/TRY、HYP_MAX/HYP_GAP、
PROBE_GAP、SINGLE_HYP、REGION_RADIUS、REGION_MARGIN —— 见 cumcm.t3.config 与 cumcm.t4.config），
因此无需复制实现：这里只是 `cumcm.t3.probing` 的薄转发，行为与"各自维护一份"完全等价，
但只在一处维护代码。

所有函数签名与问题三一致：

* `fisher_sigma`       —— 由 Fisher 信息矩阵给出位置的 1σ（J = Σ(1/σ²)(1/rᵢ²)nᵢnᵢᵀ，
                          σ_pos = √tr(J⁻¹)）；
* `hypothesis_points`  —— 补测选点用的"假设源位置"集合（区域的最小覆盖圆心 + 最远顶点采样，
                          单射线退化时沿射线枚举距离）；
* `probe_candidates`   —— 枚举候选补测点并排序（剔除近共线退化，按平均 σ 升序，σ 差 ≤2% 取
                          路近者，字典序打破并列）；
* `ambiguity_area`     —— 文献的 AOA 双站交会模糊区面积公式（与 Fisher σ 并列对照用）。

注意：t3.probing 的类型标注引用 t3.regions.Obs；问题四传入的是 t4.regions.Obs，二者字段完全
一致（channel/x/y/theta/stage），函数只按字段名访问（鸭子类型），运行时无差别。观测的阶段标签
仍在 t4 侧自己的 record 里维护（"sweep" 等），与这里无关。

出处：`cumcm.t3.probing`（任叶童(2016) AOA 双站交会模糊区面积公式与多站加权最小二乘权重
1/(σᵢRᵢ)；Chen et al.(2009, ICICS) 均方位置误差 ∝ 1/(σ²r²)）。
"""

from __future__ import annotations

from cumcm.t3.probing import (ambiguity_area, fisher_sigma, hypothesis_points,
                              probe_candidates)

__all__ = ["fisher_sigma", "hypothesis_points", "probe_candidates", "ambiguity_area"]