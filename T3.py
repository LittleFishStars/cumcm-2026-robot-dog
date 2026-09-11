#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T3.py —— 2026 高教社杯全国大学生数学建模竞赛 B 题 · 问题三

覆盖圆求解 → 依次到圆心巡视扫描 → 交会定位 → 就近试清（未命中则按文献准则补测缩小后再清）
========================================================================================

问题三要求机器狗从原点出发，在半径 1800 m 的目标圆域内自动定位并清除 10~16 个全向干扰源
（个数未知）。本程序的策略分两阶段，全程确定性（无随机搜索、无遗传算法）：

  阶段一 巡视扫描：先用尽量少的半径 1000 m 的"覆盖圆"铺满目标圆域，求出圆心位置，再依次
          走到每个圆心扫描 20 个频道，把每个源的示向度采集齐全；
  阶段二 定位与清除：用问题 1 的交会定位区域（各 ±1° 楔形之交 ∩ 圆域）得到每个源的位置
          估计 —— 走到估计点（区域的最小覆盖圆圆心）**先试一次 /clear**：命中即完成（绝大
          多数都能命中，省掉整轮补测）；未命中则就地复测，再按文献准则选点补测把区域缩小，
          然后再走过去清。清除不存在"先判断够不够准、再决定清不清"的门槛。

第一部分：为什么覆盖圆半径取 1000 m
-----------------------------------
题目给出每个干扰源的有效接收半径在 1000~1500 m 之间，即"检测点距源不超过 1000 m 就一定
听得到"。因此只要一组半径 1000 m 的圆**完全覆盖**目标圆域，机器狗依次走遍这些圆的圆心并
在每个圆心停下测向，就必然不会漏掉任何干扰源 —— 这就是本策略"不漏源"的全部依据，不需要
对源的位置分布做任何先验假设。

覆盖圆的位置（解析解）
----------------------
布局取"1 个中心圆 + 6 个环上圆"（正六边形）：中心圆心在原点，6 个环心位于半径 d 的正六边形
顶点。由对称性，圆域内最坏点必落在某个环心方位角的角平分线上（与最近环心夹角 30°）：设其极径
为 ρ，则到最近环心的距离

    f(ρ) = sqrt(ρ² + d² - √3·ρ·d)

在 ρ = √3d/2 处取最小，且该处的距离恰为 d/2。记 ρ* = d/√3，它是 f(ρ) = ρ 的唯一正根，
把圆域分成两段：

    ρ ≤ ρ*： 最近的是中心圆，距离 = ρ，最坏值 = ρ* = d/√3        ← 内圈最坏（环圆与中心圆
                                                                   覆盖边界的交点）
    ρ ≥ ρ*： 最近的是环上圆，距离 = f(ρ)（凸函数），最大值必在端点：f(ρ*) = ρ* 或 f(R) = g₂

    g₂ = f(R) = sqrt(R² + d² - √3·R·d) = sqrt((d - √3R/2)² + R²/4) ≥ R/2

    全域最坏最近距离  D(d) = max( d/√3 ,  g₂ )      ← 内圈与圆域边界两处竞争

d/√3 随 d 单调增，g₂ 在 d = √3R/2 处取最小值 R/2，两者恰在该点相等：d*/√3 = R/2 = g₂(d*)。
于是

    d* = √3·R/2 = 900√3 ≈ 1558.85 m 时余量最大：
    全域最坏最近距离 = R/2 = 900.0 m ≤ 1000 m（余量 100.0 m）。

（最坏点由此有两族：ρ* = 900 m 的角平分线方向点，以及圆域边界上 ρ = R、与某环心夹角 30° 的
点，两者到最近圆心的距离都恰为 900 m。）d* 是"余量最大"的解；本程序实际取 d = 1200 m
（余量 31.10 m、里程最短），理由见"环半径的权衡"一节。

6 个覆盖圆做不到：6 个圆全在环上（环心必须落在距原点 1000 m 内才能盖住原点），实测最优
（环半径 1000 m）最坏最近距离仍有 1059 m > 1000 m，圆域边缘会漏源 —— 所以 **7 个是最少个数**。
环半径在可行区间内可自由选取，本程序取 d = 1200 m（见上）。
17 个可行半径的扫描、6 圆方案对照、解析值与实算值的交叉验证都由本程序自动完成。

覆盖重数（决定第二阶段的补测需求）
---------------------------------
7 个覆盖圆中，约 64% 的圆域面积只被 1 个圆覆盖：这些位置上的源只能得到 1 条示向度射线，
无法直接交会定距；约 34% 被 2 个圆覆盖、约 3% 被 3 个圆覆盖，可直接交会。

环半径的权衡与可行区间
----------------------
覆盖保证只要求 D(d) ≤ 1000 m，由此得到可行区间（两端点均为零余量）：

    内圈约束：d/√3 ≤ r  →  d ≤ √3·r = 1732.05 m
    边界约束：g₂(d) ≤ r →  d ∈ [1122.96, 1994.74] m
    合起来：  d ∈ [1122.96, 1732.05] m

而巡视里程恰为 6d（原点 → 一个环心 = d，再沿正六边形走 5 条边 = 5d），随 d 单调增。于是
"余量最大"与"里程最短"是一对矛盾，本程序在可行区间内**取 d = 1200 m 作为设计环半径**：

    余量：D(1200) = 968.90 m < 1000 m ⇒ 余量 31.10 m（> 0，不漏源仍可证明）；
    里程：6d = 7200 m，比 d* = 1558.85 m 的 9353 m 少 2153 m（≈ 431 s 纯移动时间）。

理由：里程占虚拟总时间约 85%，缩短里程直接改善题目考察的"平均定位清除时间"；而余量只要
严格为正，覆盖保证就成立。实测（10 局 seed 0-9）：d = 1200 时 131/131 全部清除、平均
346.5 s/个、平均虚拟时间 4464 s，比 d* 的 413.3 s/个 / 5338 s 快约 16%。若希望更大的安全
余量，用 --ring-radius 1558.846 切回 d*（余量 100.0 m），报告里给出完整权衡表。

第二部分：定位与清除（阶段二）
-----------------------------
**位置估计：把"听不到"也当成证据。** 直接复用问题 1 的交会定位区域：检测点处 ±1° 的示向度误差
使干扰源必落在以该点为顶点、张角 2° 的楔形内，故

    可能源集合 = 所有楔形之交 ∩ 目标圆域 ∩ （接收半径给出的圆盘约束）

求交时圆域用内接 256 边形（与真圆的偏差 0.14 m）。关键补充是**每次测量结果都是一条可证明的硬
约束** —— 问题三只有全向源、且有效接收半径 R_rec ∈ [1000, 1500] m，于是

    direction ：收得到 ⇒ d ≤ R_rec ≤ 1500        ⇒ 源在"以测量点为心、1500 m"的圆盘**内**；
    near      ：d ≤ 5 m（题目近距阈值）          ⇒ 源在"以测量点为心、5 m"的圆盘**内**；
    no_signal ：收不到 ⇒ d > R_rec ≥ 1000        ⇒ 源在"以测量点为心、1000 m"的圆盘**外**。

这三条都不是启发式。no_signal 尤其宝贵：它把"什么都没听到"变成一条实质的排除约束，让那些
只拿到一条射线的频道从"贯穿圆域的长带"缩到有限的一段，也直接支撑了下面的"跳过必然无信号的
测量"。保守性靠两种近似方向保证：交（"在内"）用**外接**多边形、差（"在外"）用**内接**多边形，
于是真源永远不会被切掉。

估计点取可能源集合的最小覆盖圆圆心 —— 真源必在集合内 ⊆ 覆盖圆内，故到圆心的距离 ≤ 覆盖圆半径。
**清除判据一律用最小覆盖圆半径，不用"直径 < 40 m"**：直径判据的依据是"凸集合的覆盖圆半径 ≤
直径/2"，而减去禁区后集合可能非凸、甚至裂成多块，凸性不再成立（这一点在实现中踩过，故写死）。

**清除一律"就近试清"，不设直径门槛。** 清除半径只有 20 m，**走到源的近处是清除的必要动作**，
绕不过去；所以到达估计点后顺手试一次 /clear 的边际代价只有失败时的 3 s，命中却省掉整轮补测
（一轮补测要绕几百米、约 100 s 量级的里程）。反过来，"先把可能源集合压到覆盖圆半径 < 20 m
保证必中、再决定清除"要额外花探针去换那个确定性，反而更贵。因此覆盖圆半径 < 20 m 只表示
"估计已够准"，用来决定**还要不要继续补测**，不决定清不清除。

**清除流程（无门槛、四级递进 + 巡视途中顺路清）。** 每个频道依次做：

① **就近试清**：走到估计点 /clear 一次。失败只花 3 s，命中即完成（10 局实测 70/131 个在这一步解决）。
② **多清几次**：未命中就在集合内取细网格为覆盖目标，贪心选至多 `--k-clear-max`（缺省 4）个补充
   清缺点，使这些半径 20 m 的圆把集合盖满，再由近及远逐个清 —— 盖满即保证命中，可省掉一轮补测；
   盖不满（集合被作业圆域截断、需要十几个圆）就不赌，直接进入 ③。实测这一步只覆盖到 2/131 个
   频道：试清未中的 30 个里 24 个集合被圆域截断、直径 > 320 m（需 K = 10~70 个圆），而补测一次
   约 120 s、K=16 个点硬清约 170 s，所以大区域仍是补测更划算。
③ **补测**：就地复测一次（该点是集合内离源最近、Fisher 权重最大的位置），若估计仍不够准则按
   文献准则选点补测，直到覆盖圆半径 < 20 m 或达到轮次上限，再到新的估计点清。
④ **兜底**：万一仍未清除，沿最新实测示向度以 16 m 步长逼近（步数按到集合最远顶点的距离自适应）。

此外，**巡视途中顺路清除**：每站扫描完，若某频道的估计点顺路（绕行 ≤ `--inline-detour`，缺省
400 m）且估计不太离谱（覆盖圆半径 ≤ `--inline-try-radius`，缺省 150 m），就当场清掉 —— 巡视本来
就要路过，绕一下的代价远小于留到阶段二专程跑一趟，清掉后该频道后续各站也不再测量。10 局实测
顺路清掉 34/131 个（26%）、白跑 8 次（每次 3 s）。参数由 4 组对照实验选出：半径 150 与 300 结果
相同（说明 150 已覆盖全部机会），绕行放宽到 600/700 m 反而更慢。

**行进线路。**
* 巡视：7 个圆心的访问长度恒为 6d（原点 → 一个环顶点 = d，再沿正六边形走 5 条边 = 5d），所以
  "从哪个环顶点开始、顺时针还是逆时针"共 12 条路径**长度完全相同**。第 1 站（原点）的起始
  全频道扫描一结束，就用它锚定到的源方位挑选落脚点 —— 让巡视终点靠近这些最先能清的源，
  清除阶段即可从那里开始，不必为它们单独折返（实测终点到已听源的估计距离由 1845 m 降到 1237 m）。
* 清除：按各频道估计点到当前位置的距离做最近邻 + 2-opt 的开放路径（清完不回原点）。实测
  "每清一个就重排"与"一次定序"结果完全一致，故保留更简单的一次定序。

**最省的那一步：起始全频道扫描。** 第 1 站就在原点，一次把 20 个频道全测一遍（20 次测向 ≈
20×5 s + 19×1 s 切换 ≈ 119 s），换来的是"哪些频道在 1000 m 内"这批最便宜的信息 —— 10 局实测
平均直接锚定 6.1 个源的方向（占全部源的三分之一），同时把其余频道标记为"源在 1000 m 之外"。
它还给后续带来两项收益：① 巡视绕向与落脚点由它决定（见上）；② 这些源只需再补一条射线就能
定位，于是很多站可以少测甚至不测。

**跳过"必然无信号"的测量。** 可能源集合是真实源位置的超集，故"集合到测量点的最小距离 > 1500 m"
⇒ 真源也在 1500 m 之外 ⇒ 必然收不到信号。此时这次测量不带来任何新信息，直接跳过。10 局实测
平均每局跳过 19 次测量（≈ 110 s）。加上"估计已够准就不再测"，每局测量次数由最初的 132 次降到
111 次。

与参考文献（覆盖圆部分）的关系
------------------------------
赵一骁, 何航天, 李雨楠, 等. 基于最小圆覆盖的多无人机协同螺旋式搜索优化算法[J]. 指挥信息
系统与技术, 2024, 15(4): 56-62：提出"圆覆盖 + 圆内接正六边形拼接"的环境建模，以传感器搜索
半径 r 为覆盖圆，相邻圆心间距取紧贴的 √3·r（两圆重叠 5.77%、利用率 94.23%），覆盖圆圆心即
航路关键点，只需遍历全部关键点就完成了区域覆盖搜索。本题覆盖圆的个数与结构与之一致（都是
"1 中心 + 6 环"共 7 个），但环半径由紧贴间距 √3·r = 1732.05 m 改进为 √3·R/2 = 1558.85 m：
最坏最近距离由 1000.0 m（零余量）降到 900.0 m（余量 100 m），巡视里程同时由 10392 m 降到
9353 m，两项同时更优。差别来自边界条件：紧贴间距是"无限平面"最小圆覆盖问题的解，本题是
有界圆域，最坏点由内圈与圆域边界两处共同决定（见上节）。

运行
====
    python T3.py                                    # 只求解覆盖圆并打印报告（不连模拟器）
    python T3.py --ring-radius 1558.846             # 切回余量最大的 d*（里程更长）
    python T3.py --practice 10 --seed 0             # 本地演练 10 局（自动拉起 jammers-py）
    python T3.py --practice 3 --survey-only         # 只做阶段一（巡视扫描 + 覆盖核对）
    python T3.py --practice 5 --no-reuse --console-port 8095 --robot-port 2027
                                                    # 另起独占演练实例（多会话并行时避免抢端口）
    python T3.py --base-url http://127.0.0.1:2026   # 官方评测接口模式（赛期，先开模拟器）

结果落盘（--save-dir，缺省 results/，文件名固定便于论文与绘图引用）
    t3_cover_plan.json    覆盖圆求解结果（环半径、圆心、最坏距离、覆盖重数、巡视顺序、权衡表）
    t3_cover_circles.csv  7 个覆盖圆的圆心坐标（序号、类型、x、y）
    t3_survey.json        逐局统计 + 逐源真值核对 + 逐频道定位/清除档案 + 整批汇总
    t3_observations.csv   逐条测量记录（局号、频道、阶段、检测点坐标、结果类型、示向度）
                          —— 含 no_signal，便于逐条复核"为什么这里必然听不到"
    trajectory/t3/        逐局轨迹图 epNN_seedMM.png 与同名轨迹表 .csv（--no-plot 可关闭，
                          --traj-dir 可改目录）。刻意与 T3_ga.py 的 trajectory/ 分开，避免同名覆盖

依赖：numpy（覆盖校验）、shapely（定位区域，经 T1.py）、matplotlib 不需要；
HTTP 层复用同目录 sim_api.py；本地演练用同目录 jammers-py/（纯标准库）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import shapely
from shapely import Point

import sim_api
from T1 import TriangulationRegion      # 问题 1 的交会定位区域（楔形交 ∩ 圆域）

# ----------------------------------------------------------------------------
# 常量（与题目附录 1 / 附录 2 一致）
# ----------------------------------------------------------------------------
REGION_RADIUS = 1800.0          # 目标圆域半径 / m
COVER_RADIUS = 1000.0           # 覆盖圆半径 = 有效接收半径下界 / m
RECEIVE_MAX = 1500.0            # 有效接收半径上界 / m（题目给定 1000~1500）
RECEIVE_MID = 0.5 * (COVER_RADIUS + RECEIVE_MAX)   # 接收半径中值，用于估源距离 / m
INLINE_TRY_RADIUS = 150.0       # 巡视途中顺路试清：估计点覆盖圆半径 ≤ 该值才值得绕 / m
INLINE_DETOUR = 400.0           # 巡视途中顺路试清允许的最大绕行里程 / m
                                # （两者缺省值由 5 局 × 4 组参数实测选出：半径 150 与 300 结果相同，
                                #   说明 150 已覆盖全部机会；绕行放宽到 600/700 m 反而更慢 ——
                                #   绕得越远，试清失败时就越是纯粹白跑）
EXCL_QUAD = 16                  # 圆盘约束的近似精度：正 4×EXCL_QUAD 边形（见 ProbRegion）
CHANNELS: Tuple[int, ...] = tuple(range(1, 21))      # 20 个频道
COORD_LIMIT = 2.0e6             # 坐标分量绝对值上限 / m（协议规定）
REGION_MARGIN = 1.0             # 坐标裁剪保留的数值余量 / m
OBS_CAP = 3                     # 每个频道最多采集的示向度条数（够了就不再重复测）
SAFETY_MARGIN = 30.0            # 现实时限预留余量 / s

GRID_STEP = 5.0                 # 覆盖校验的细网格 / m
BOUNDARY_SAMPLES = 20000        # 覆盖校验的边界采样点数（圆域边界是"最坏点"候选密集区）
COARSE_STEP = 15.0              # 6 圆方案对照用的粗网格 / m
COARSE_BOUNDARY = 4000          # 6 圆方案对照的边界采样点数
TOL = 1e-9

# 选定的设计环半径（本程序默认用它布点，时间优先）。可行的最小环半径是 1122.96 m（余量 0）；
# 取 1200 m 时最坏最近距离 968.90 m < 1000 m，余量 31.10 m —— 仍是**可证明**的不漏源保证，
# 而巡视里程只有 6d = 7200 m，比余量最大的 d* = 1558.85 m（里程 9353 m）少 2153 m，
# 实测单局平均虚拟时间由 5338 s 降到 4464 s（各类初始位置不同，省 8%~18%）。
# 若宁可要更大的安全余量，用 --ring-radius 1558.846 切回 d*（余量 100.0 m）。
CHOSEN_RING_RADIUS = 1200.0

SEED = 2026

RESULTS_DIR = "results"
PLAN_JSON = "t3_cover_plan.json"        # 覆盖圆求解结果
PLAN_CSV = "t3_cover_circles.csv"       # 覆盖圆圆心坐标
SURVEY_JSON = "t3_survey.json"          # 逐局巡视扫描统计
OBS_CSV = "t3_observations.csv"         # 逐条示向度观测
# 逐局轨迹图与轨迹表所在子目录（相对 --save-dir）。**刻意与 T3_ga.py 分开**：后者把 GA 版的
# 轨迹图写在 <save-dir>/trajectory/epNN_seedMM.png，两边同名且同目录会互相覆盖（实测踩过，
# 一次性覆盖掉对方 10 个已提交文件）。故本程序固定写到自己的子目录，互不干扰。
TRAJ_DIR = "trajectory/t3"
TRAJ_DPI = 160.0                        # 轨迹图位图分辨率

# ---- 第二阶段常量：定位区域、清除判据、补测选点准则（文献方法）----
BEARING_ERROR_DEG = 1.0         # 示向度误差半宽 / 度（题目给定 |误差| ≤ 1°，楔形张角 2°）
SIGMA_DEG = BEARING_ERROR_DEG   # 同一误差的量级（Fisher 信息里的 σ）/ 度
SIGMA_RAD = math.radians(SIGMA_DEG)
CLEAR_RADIUS = 20.0             # 清除半径 / m（光学精确定位要求 ≤ 20 m）
NEAR_RADIUS = 5.0               # 近距阈值 / m（≤ 5 m 可跳过测向直接清除）
CLIP_SIDES = 256                # 定位区域求交时目标圆域的内接多边形边数
CLIP_ERR = REGION_RADIUS * (1.0 - math.cos(math.pi / CLIP_SIDES))   # 内接多边形与真圆的偏差 / m
DIAM_PRECISE = 2.0 * CLEAR_RADIUS   # 仅供报告参考：凸区域下与"最小覆盖圆半径 < 20 m"等价
                                    # （只用来决定"要不要继续补测"，不作清除门槛，见 RobotDog 说明）
PROBE_RADII = (150.0, 300.0, 450.0, 600.0, 800.0)   # 补测候选点到假设源位置的距离 / m
PROBE_ANGLES = 24               # 补测候选点的方位角格数（15° 一格）
PROBE_TRY = 3                   # 每轮补测最多试几个候选点（收不到信号就换下一个）
HYP_MAX = 8                     # 假设源位置最多取几个（最坏情形稳健）
HYP_GAP = 50.0                  # 假设点与已有检测点的最小间距 / m（太近无法估计距离）
PROBE_GAP = 60.0                # 补测点与已有检测点的最小间距 / m（同点复测不提供新信息）
SINGLE_HYP = (200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0)   # 单射线时沿射线的假设距离
REFINE_MAX = 6                  # 每个频道最多补测几轮
TRY_CLEAR_RADIUS = 400.0        # 就近试清的前提：最小覆盖圆半径 ≤ 该值（估计不离谱，值得跑过去试）
K_CLEAR_MAX = 4                 # 试清未中后，最多再补清几个点（用 K 个半径 20 m 的圆覆盖定位区域）
K_COVER_SAMPLES = 40            # K 圆覆盖的采样格数（最长方向切成这么多格，据此定采样步长）
K_COVER_STEP_MIN = 3.0          # K 圆覆盖的采样步长下限 / m
HOMING_STEP = 16.0              # 末端沿最新示向度逼近的步长 / m
HOMING_MAX = 24                 # 末端沿示向度逼近的最少迭代次数
HOMING_CAP = 120                # 末端逼近的迭代上限（离得远时按距离自适应加长，但不超过此值）


# ----------------------------------------------------------------------------
# 基础几何工具
# ----------------------------------------------------------------------------
def dist(a: Sequence[float], b: Sequence[float]) -> float:
    """两点距离 / m。"""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def bearing(a: Sequence[float], b: Sequence[float]) -> float:
    """a → b 的方位角（度，x 轴正向逆时针，[0, 360)）。"""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def clamp_to_region(x: float, y: float,
                    radius: float = REGION_RADIUS - REGION_MARGIN) -> Tuple[float, float]:
    """把坐标拉回作业圆域内（出域时沿原方向缩到边界）。"""
    r = math.hypot(x, y)
    if r <= radius or r == 0.0:
        return x, y
    k = radius / r
    return x * k, y * k


# ----------------------------------------------------------------------------
# 覆盖圆求解
# ----------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class CoverPlan:
    """覆盖圆方案：所有覆盖圆的圆心的位置。

    圆心编号：0 号为圆心在原点的那一个（中心圆），1~6 号为正六边形环上的圆心（逆时针，
    从 0° 方位角起）。这些圆心就是机器狗的巡视路点。
    """

    region_radius: float            # 目标圆域半径 / m
    cover_radius: float             # 覆盖圆半径（= 有效接收半径下界）/ m
    ring_radius: float              # 六边形环上圆心到原点的距离 d / m
    waypoints: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "waypoints", hex_layout(self.ring_radius))

    @property
    def centers(self) -> List[Tuple[float, float]]:
        return [(float(x), float(y)) for x, y in self.waypoints]


def hex_layout(ring_radius: float, n_ring: int = 6) -> np.ndarray:
    """正六边形布局的圆心：第 0 个在原点，其余 n_ring 个在半径 ring_radius 的环上。"""
    ang = np.arange(n_ring) * (2.0 * math.pi / n_ring)
    ring = np.stack((ring_radius * np.cos(ang), ring_radius * np.sin(ang)), axis=1)
    return np.vstack([[0.0, 0.0], ring])


def optimal_ring_radius(region_radius: float = REGION_RADIUS) -> float:
    """最优环半径 d* = √3·R/2：使最坏最近距离 D(d) 最小的解析解。"""
    return math.sqrt(3.0) * region_radius / 2.0


def feasible_ring_interval(region_radius: float = REGION_RADIUS,
                           cover_radius: float = COVER_RADIUS) -> Tuple[float, float]:
    """满足覆盖保证 D(d) ≤ r 的环半径区间 [d_min, d_max]（解析）。

    两个约束各给一段区间，取交集：
      内圈约束 d/√3 ≤ r           → d ≤ √3·r
      边界约束 g₂(d) ≤ r（解 d² - √3R·d + R² - r² = 0）
                                  → d ∈ [(√3R - √(4r²-R²))/2, (√3R + √(4r²-R²))/2]
    区间两端点都是零余量（最坏距离恰好 = r），并随 d 单调变化。
    """
    root = math.sqrt(max(4.0 * cover_radius ** 2 - region_radius ** 2, 0.0))
    lo = (math.sqrt(3.0) * region_radius - root) / 2.0
    hi = min(math.sqrt(3.0) * cover_radius,
             (math.sqrt(3.0) * region_radius + root) / 2.0)
    return lo, hi


def analytic_worst(ring_radius: float, region_radius: float = REGION_RADIUS,
                   cover_radius: float = COVER_RADIUS) -> float:
    """最坏最近距离的解析值 D(d) = max(d/√3, g₂)（推导见模块文档）。

    d/√3 是内圈（只有中心圆能覆盖到）的最坏距离；g₂ 是圆域边界上、与某环心夹角 30° 的点
    到最近环心的距离。两项在 d* = √3·R/2 处同时等于 R/2，即最优点的最坏距离。
    """
    def g(rho: float) -> float:
        return math.sqrt(max(rho * rho + ring_radius * ring_radius
                             - math.sqrt(3.0) * rho * ring_radius, 0.0))
    return max(ring_radius / math.sqrt(3.0), g(region_radius))


@lru_cache(maxsize=8)
def region_samples(step: float, n_boundary: int) -> np.ndarray:
    """目标圆域的采样点：细网格 + 圆边界均匀采样（最坏点常在边界上，须精细采样）。"""
    ax = np.arange(-REGION_RADIUS, REGION_RADIUS + TOL, step)
    gx, gy = np.meshgrid(ax, ax)
    pts = np.stack((gx.ravel(), gy.ravel()), axis=1)
    pts = pts[np.linalg.norm(pts, axis=1) <= REGION_RADIUS + TOL]
    t = np.linspace(0.0, 2.0 * math.pi, n_boundary, endpoint=False)
    ring = np.stack((REGION_RADIUS * np.cos(t), REGION_RADIUS * np.sin(t)), axis=1)
    return np.vstack([pts, ring])


def nearest_distances(points: np.ndarray, waypoints: np.ndarray,
                      chunk: int = 200_000) -> np.ndarray:
    """每个采样点到最近圆心的距离（分块计算，避免大矩阵占内存）。"""
    out = np.empty(len(points), dtype=float)
    for i in range(0, len(points), chunk):
        blk = points[i:i + chunk]
        d = np.linalg.norm(blk[:, None, :] - waypoints[None, :, :], axis=2)
        out[i:i + chunk] = d.min(axis=1)
    return out


def worst_candidates(ring_radius: float, region_radius: float = REGION_RADIUS) -> np.ndarray:
    """解析给出的最坏点候选：12 个角平分线方向 × {ρ* = d/√3, ρ = R}。

    圆域内的最坏点必落在某个"角平分线方向"上（与最近环心夹角 30°）：内圈最坏点在 ρ* = d/√3
    处（此时它到最近环心的距离恰为 ρ*，与中心圆打平），外圈最坏点在圆域边界 ρ = R 处。把这
    些点放进采样集合，解析解与实算解就能一致到机器精度，网格只用于旁证。
    """
    ang = np.arange(12) * (math.pi / 6.0) + math.pi / 6.0
    pts = [(r * math.cos(a), r * math.sin(a))
           for r in (ring_radius / math.sqrt(3.0), region_radius) for a in ang]
    return np.array(pts, dtype=float)


def cover_counts(points: np.ndarray, waypoints: np.ndarray, radius: float,
                 chunk: int = 200_000) -> np.ndarray:
    """每个采样点被几个覆盖圆同时覆盖（覆盖重数）。"""
    out = np.empty(len(points), dtype=np.int64)
    for i in range(0, len(points), chunk):
        blk = points[i:i + chunk]
        d = np.linalg.norm(blk[:, None, :] - waypoints[None, :, :], axis=2)
        out[i:i + chunk] = (d <= radius + TOL).sum(axis=1)
    return out


def nearest_order(waypoints: np.ndarray, start: Sequence[float] = (0.0, 0.0)) -> List[int]:
    """确定性最近邻访问顺序（并列时取编号小者）：从 start 出发依次走遍所有圆心。"""
    rest = list(range(len(waypoints)))
    order: List[int] = []
    cur = np.asarray(start, dtype=float)
    while rest:
        k = min(rest, key=lambda j: (float(np.linalg.norm(waypoints[j] - cur)), j))
        order.append(k)
        rest.remove(k)
        cur = waypoints[k]
    return order


def path_length(waypoints: np.ndarray, order: Sequence[int],
                start: Sequence[float] = (0.0, 0.0)) -> float:
    """按给定顺序走遍各圆心的总里程 / m。"""
    total, cur = 0.0, np.asarray(start, dtype=float)
    for j in order:
        total += float(np.linalg.norm(waypoints[j] - cur))
        cur = waypoints[j]
    return total


@dataclass
class CoverSolveResult:
    """覆盖圆求解的完整结果：方案 + 实算校验 + 权衡分析 + 文献方法对照。"""

    plan: CoverPlan
    worst_distance: float               # 实算最坏最近距离 / m（网格 + 边界 + 解析候选点）
    worst_point: Tuple[float, float]    # 实算最坏点位置
    analytic_worst: float               # 解析最坏最近距离 / m
    coverage_ratio: float               # 被覆盖的采样点比例（1.0 表示全覆盖）
    multiplicity: Dict[int, int]        # 覆盖重数 → 采样点数
    single_ratio: float                 # 单重覆盖（只有 1 条射线可用）的占比
    multi_ratio: float                  # 二重及以上覆盖的占比
    mean_multiplicity: float            # 圆域内平均覆盖重数（重复率 = 平均值 - 1）
    survey_order: List[int]             # 巡视顺序（圆心编号）
    survey_length: float                # 巡视总里程 / m
    feasible_interval: Tuple[float, float]   # 满足覆盖保证的环半径可行区间 / m
    tradeoff: List[Dict[str, float]]    # 环半径权衡：d、最坏距离、余量、里程、时间
    lattice: Dict[str, float]           # 参考文献紧贴六边形栅格（间距 √3·r）对照
    six_circle_worst: float             # 6 圆方案的实算最坏距离（不可行对照）/ m
    six_circle_radius: float            # 6 圆方案的最优环半径 / m

    @property
    def margin(self) -> float:
        """最坏最近距离相对覆盖半径（1000 m）的余量 / m。"""
        return COVER_RADIUS - self.worst_distance

    def to_json(self) -> Dict[str, Any]:
        plan = self.plan
        return {
            "region_radius_m": plan.region_radius,
            "cover_radius_m": plan.cover_radius,
            "ring_radius_m": round(plan.ring_radius, 3),
            "ring_radius_chosen": abs(plan.ring_radius - CHOSEN_RING_RADIUS) < 1e-9,
            "ring_radius_max_margin_m": round(optimal_ring_radius(), 3),
            "n_circles": len(plan.waypoints),
            "analytic_worst_m": round(self.analytic_worst, 4),
            "computed_worst_m": round(self.worst_distance, 4),
            "margin_m": round(self.margin, 4),
            "worst_point": [round(v, 2) for v in self.worst_point],
            "coverage_ratio": self.coverage_ratio,
            "multiplicity_points": self.multiplicity,
            "single_ratio": self.single_ratio,
            "multi_ratio": self.multi_ratio,
            "mean_multiplicity": round(self.mean_multiplicity, 4),
            "survey_order": self.survey_order,
            "survey_length_m": round(self.survey_length, 2),
            "centers": [{"id": i, "kind": "中心圆" if i == 0 else "环上圆",
                         "x": round(x, 3), "y": round(y, 3)}
                        for i, (x, y) in enumerate(plan.centers)],
            "feasible_ring_interval_m": [round(v, 3) for v in self.feasible_interval],
            "ring_radius_tradeoff": self.tradeoff,
            "reference_hex_lattice": self.lattice,
            "compare_six_circles": {"n_circles": 6, "best_ring_radius_m": self.six_circle_radius,
                                    "computed_worst_m": round(self.six_circle_worst, 4),
                                    "feasible": self.six_circle_worst <= COVER_RADIUS},
            "grid_step_m": GRID_STEP,
            "boundary_samples": BOUNDARY_SAMPLES,
        }


def _six_circle_best(step: float = COARSE_STEP,
                     n_boundary: int = COARSE_BOUNDARY) -> Tuple[float, float]:
    """6 个覆盖圆的可行性对照：环半径必须 ≤ 1000 m 才能盖住原点，扫描取最优。

    返回（最优环半径 / m, 该半径下的最坏最近距离 / m）。粗网格仅用于可行性判断。
    """
    pts = region_samples(step, n_boundary)
    best = (float("inf"), 0.0)
    for d in np.arange(200.0, COVER_RADIUS + 1.0, 10.0):
        wp = hex_layout(float(d))[1:]          # 只有环上 6 个圆，没有中心圆
        m = float(nearest_distances(pts, wp).max())
        if m < best[0]:
            best = (m, float(d))
    return best[1], best[0]


def layout_metrics(ring_radius: float, pts: np.ndarray,
                   compute_multiplicity: bool = True) -> Dict[str, Any]:
    """给定环半径，报出该布局的最坏最近距离、余量、覆盖重数与巡视里程。"""
    wp = hex_layout(float(ring_radius))
    near = nearest_distances(pts, wp)
    order = nearest_order(wp)
    length = path_length(wp, order)
    row: Dict[str, Any] = {
        "ring_radius_m": round(float(ring_radius), 3),
        "computed_worst_m": round(float(near.max()), 3),
        "analytic_worst_m": round(analytic_worst(float(ring_radius)), 3),
        "margin_m": round(COVER_RADIUS - float(near.max()), 3),
        "survey_length_m": round(length, 1),
        "survey_time_s": round(length / 5.0, 1),
    }
    if compute_multiplicity:
        cnt = cover_counts(pts, wp, COVER_RADIUS)
        row["mean_multiplicity"] = round(float(cnt.mean()), 4)
        row["min_multiplicity"] = int(cnt.min())
    return row


def tradeoff_table(interval: Tuple[float, float], pts: np.ndarray) -> List[Dict[str, Any]]:
    """可行区间内取若干代表环半径，给出"余量 vs 里程"的权衡表。"""
    lo, hi = interval
    radii = [lo, 1200.0, 1300.0, 1400.0, optimal_ring_radius(), 1700.0, hi]
    radii = [r for r in radii if lo - 1e-9 <= r <= hi + 1e-9]
    return [layout_metrics(r, pts) for r in sorted(set(round(r, 3) for r in radii))]


def solve_covering_circles(ring_radius: Optional[float] = None) -> CoverSolveResult:
    """求解 1000 m 覆盖圆的位置，并做实算校验、可行区间与权衡分析、文献方法对照。"""
    d = CHOSEN_RING_RADIUS if ring_radius is None else float(ring_radius)
    plan = CoverPlan(REGION_RADIUS, COVER_RADIUS, d)
    assert len(plan.waypoints) == 7, "覆盖圆个数应为 7（1 中心 + 6 环）"

    # 采样点 = 细网格 + 圆边界精细扫描 + 解析给出的最坏点候选（保证解析/实算一致）
    pts = np.vstack([region_samples(GRID_STEP, BOUNDARY_SAMPLES),
                     worst_candidates(d)])
    near = nearest_distances(pts, plan.waypoints)
    k = int(near.argmax())
    worst = float(near[k])
    worst_point = (float(pts[k][0]), float(pts[k][1]))

    cnt = cover_counts(pts, plan.waypoints, COVER_RADIUS)
    multiplicity = {int(a): int(b) for a, b in zip(*np.unique(cnt, return_counts=True))}
    single = multiplicity.get(1, 0) / len(pts)
    multi = sum(v for kk, v in multiplicity.items() if kk >= 2) / len(pts)

    order = nearest_order(plan.waypoints)
    length = path_length(plan.waypoints, order)
    ana = analytic_worst(d)
    if abs(worst - ana) > 1e-6:
        raise AssertionError(f"解析最坏距离 {ana:.6f} m 与实算 {worst:.6f} m 不一致")
    if worst > COVER_RADIUS:
        raise AssertionError(f"覆盖保证被破坏：最坏距离 {worst:.1f} m > {COVER_RADIUS:.0f} m")
    if abs(length - 6.0 * d) > 1e-6:
        raise AssertionError(f"巡视里程 {length:.3f} m 与解析值 6d = {6.0 * d:.3f} m 不一致")

    interval = feasible_ring_interval()
    # 参考文献（赵一骁 2024）的紧贴六边形栅格：相邻圆心间距 √3·r，即环半径 = √3·r
    lattice_d = math.sqrt(3.0) * COVER_RADIUS
    lattice = layout_metrics(lattice_d, pts)

    six_d, six_worst = _six_circle_best()
    return CoverSolveResult(
        plan=plan, worst_distance=worst, worst_point=worst_point, analytic_worst=ana,
        coverage_ratio=float((cnt >= 1).mean()), multiplicity=multiplicity,
        single_ratio=single, multi_ratio=multi, mean_multiplicity=float(cnt.mean()),
        survey_order=order, survey_length=length,
        feasible_interval=interval, tradeoff=tradeoff_table(interval, pts),
        lattice=lattice, six_circle_worst=six_worst, six_circle_radius=six_d,
    )


def print_cover_report(res: CoverSolveResult) -> None:
    """打印覆盖圆求解报告（圆心位置、覆盖校验、最少个数、可行区间与权衡、文献对照）。"""
    plan = res.plan
    d = plan.ring_radius
    is_chosen = abs(d - CHOSEN_RING_RADIUS) < 1e-6
    print("=" * 78)
    print("一、1000 m 覆盖圆的位置")
    print("=" * 78)
    print(f"目标圆域半径 R = {plan.region_radius:.0f} m，覆盖圆半径 r = {plan.cover_radius:.0f} m"
          f"（= 有效接收半径下界），共 {len(plan.waypoints)} 个覆盖圆")
    print(f"布局：1 个中心圆（圆心在原点）+ 6 个环上圆（正六边形顶点）")
    print(f"环半径 d = {d:.2f} m（正六边形边长 = d）"
          + ("（选定设计半径：时间优先）" if is_chosen else "（由 --ring-radius 指定）"))
    print()
    print(f"{'序号':<6}{'类型':<10}{'x / m':>12}{'y / m':>12}")
    for i, (x, y) in enumerate(plan.centers):
        print(f"{i:<6}{'中心圆' if i == 0 else '环上圆':<10}{x:>12.2f}{y:>12.2f}")
    print()
    print("=" * 78)
    print(f"二、覆盖校验（细网格 {GRID_STEP:.0f} m + 圆边界 {BOUNDARY_SAMPLES} 点 + 解析最坏点候选）")
    print("=" * 78)
    print(f"解析最坏最近距离 D(d) = max(d/√3, g₂) = {res.analytic_worst:.3f} m"
          + ("（d = d* 时内圈项与边界项相等，即余量最大的最优性条件）"
             if abs(d - optimal_ring_radius()) < 1e-6 else ""))
    print(f"实算最坏最近距离 = {res.worst_distance:.3f} m @ "
          f"({res.worst_point[0]:.1f}, {res.worst_point[1]:.1f})"
          f"；另一族最坏点在 ρ = d/√3 = {d / math.sqrt(3):.1f} m 的角平分线方向上")
    print(f"对覆盖半径 1000 m 的余量 = {res.margin:.3f} m，"
          f"采样点被覆盖比例 = {res.coverage_ratio * 100:.2f}%")
    print(f"覆盖重数分布：" + "、".join(f"{k} 重 {v} 点" for k, v in sorted(res.multiplicity.items())))
    print(f"→ 平均覆盖重数 {res.mean_multiplicity:.3f}（重复率 "
          f"{(res.mean_multiplicity - 1) * 100:.1f}%）；单重覆盖（只有 1 条射线，无法直接交会）占 "
          f"{res.single_ratio * 100:.1f}%，二重及以上占 {res.multi_ratio * 100:.1f}%"
          f"（第二阶段需对单射线频道补测第二视角）")
    print()
    print("=" * 78)
    print("三、为什么是 7 个（最少数）")
    print("=" * 78)
    print(f"6 个覆盖圆：环心须落在距原点 1000 m 内才能盖住原点；最优环半径 "
          f"{res.six_circle_radius:.0f} m 时实算最坏距离 {res.six_circle_worst:.1f} m "
          f"> 1000 m ✗（圆域边缘漏源）")
    print(f"7 个覆盖圆：最坏距离 {res.worst_distance:.1f} m ≤ 1000 m ✓（余量 "
          f"{res.margin:.1f} m）→ 7 个即最少可行个数")
    print()
    print("=" * 78)
    print("四、可行区间与权衡（余量 vs 巡视里程）")
    print("=" * 78)
    lo, hi = res.feasible_interval
    print(f"覆盖保证等价于 D(d) ≤ 1000 m，可行区间 d ∈ [{lo:.2f}, {hi:.2f}] m"
          f"（两端点余量为 0）")
    print(f"现用 d = {d:.2f} m（余量 {res.margin:.2f} m）；余量最大的是 d* = "
          f"{optimal_ring_radius():.2f} m（余量 100.00 m，里程 9353 m）"
          f"——取更小的 d 即「用余量换时间」，两者都可证明不漏源")
    print(f"巡视里程 = 6d（原点→环心 d，再走 5 条六边形边），故 d 越小越省时间")
    print()
    print(f"{'环半径 d / m':>13}{'最坏距离 / m':>13}{'余量 / m':>11}"
          f"{'平均重数':>10}{'里程 / m':>11}{'移动时间 / s':>13}")
    for row in res.tradeoff:
        mark = ""
        if abs(row["ring_radius_m"] - d) < 1e-6:
            mark = "  ← 现用（时间优先）"
        elif abs(row["ring_radius_m"] - optimal_ring_radius()) < 1e-6:
            mark = "  ← d*（余量最大）"
        print(f"{row['ring_radius_m']:>13.2f}{row['computed_worst_m']:>13.2f}"
              f"{row['margin_m']:>11.2f}{row.get('mean_multiplicity', float('nan')):>10.3f}"
              f"{row['survey_length_m']:>11.1f}{row['survey_time_s']:>13.1f}{mark}")
    print()
    lat = res.lattice
    print(f"参考文献紧贴栅格对照（相邻圆心间距 √3·r，即环半径 {lat['ring_radius_m']:.2f} m）："
          f"最坏距离 {lat['computed_worst_m']:.2f} m（余量 {lat['margin_m']:.2f} m）、"
          f"里程 {lat['survey_length_m']:.1f} m")
    print(f"现用 d = {d:.2f} m：最坏距离 {res.worst_distance:.2f} m（余量 {res.margin:.2f} m）、"
          f"里程 {res.survey_length:.1f} m → 覆盖余量优于紧贴栅格、里程省 "
          f"{lat['survey_length_m'] - res.survey_length:.1f} m")
    print()
    print("=" * 78)
    print("五、巡视顺序（确定性最近邻：从原点出发，圆心 0 就在原点）")
    print("=" * 78)
    seq = " → ".join(str(i) for i in res.survey_order)
    print(f"顺序：{seq}")
    print(f"总里程 = {res.survey_length:.1f} m，纯移动时间 = {res.survey_length / 5.0:.1f} s"
          f"（速度 5 m/s）")
    print("=" * 78)


# ----------------------------------------------------------------------------
# 第二阶段工具：定位区域、Fisher 信息、补测选点准则（文献方法）
# ----------------------------------------------------------------------------
class Obs(NamedTuple):
    """一次 direction 检测：在 (x, y) 处测得频道 channel 的示向度 theta（度）。

    stage 记录该次观测来自哪个阶段（survey = 巡视扫描，refine = 文献准则补测/试清复测，
    clear = 清除阶段的复测）。
    """

    channel: int
    x: float
    y: float
    theta: float
    stage: str = "survey"


class Meas(NamedTuple):
    """一次测量的完整记录（**含 no_signal**）。

    no_signal 在问题三里是一条实质证据：源既然收不到，就必然在接收半径下界 1000 m 之外。
    只记录 direction 观测会丢掉这部分信息，故本程序记录全部测量。
    """

    channel: int
    x: float
    y: float
    outcome: str                        # direction / near / no_signal
    theta: Optional[float] = None       # 仅 direction 时有值
    stage: str = "survey"


class ProbRegion(TriangulationRegion):
    """问题三的"可能源集合"：交会楔形 ∩ 目标圆域，再叠加接收半径给出的**硬约束**。

    为什么可以叠加。问题三只有全向源（模拟器对 problem_no=3 硬校验禁止定向源），且有效接收
    半径 R_rec ∈ [1000, 1500] m，于是每次测量结果都对应一条可证明的约束：

        direction ：收得到 ⇒ d ≤ R_rec ≤ 1500 ⇒ 源落在以测量点为心、半径 1500 m 的圆盘**内**；
        near      ：d ≤ 5 m（近距阈值）      ⇒ 源落在以测量点为心、半径 5 m 的圆盘**内**；
        no_signal ：收不到 ⇒ d > R_rec ≥ 1000 ⇒ 源落在以测量点为心、半径 1000 m 的圆盘**外**。

    三者都不是启发式。no_signal 尤其宝贵：它把"什么都没听到"变成一条实质的排除约束 —— 单条
    射线原本只给出"一条贯穿圆域的长带"，叠加"环带 1000~1500 m"与各站的 no_signal 禁区后，
    常常直接压到可清除量级。这正是"起始先做一次全频道扫描"的额外收益来源。

    保守性（真源永远留在区域内）。圆盘用正多边形近似，两个方向必须各自偏保守：
        "源在圆盘内"的**交集**用外接多边形（半径 r / cos(π/n) ⊇ 真圆盘）；
        "源在圆盘外"的**差集**用内接多边形（半径 r ⊆ 真圆盘）。
    所交集合偏大、所减集合偏小，故真源不会被切掉。

    非凸与多块。减去若干圆盘后区域可能不再凸、甚至裂成多块，本类按"整个几何"处理：vertices
    收集各分块的外环顶点，故直径与最小覆盖圆都覆盖全部可能位置（最远点对必在顶点上），判据只
    会更保守 —— 安全。但"直径 < 40 m ⇒ 最小覆盖圆半径 < 20 m"依赖凸性、非凸时不再成立，故
    T3.py 的清除判据一律直接用最小覆盖圆半径（见 RobotDog._precise），不用直径。
    """

    def __init__(self, err: float = 1.0, radius: Optional[float] = None,
                 sides: Optional[int] = None) -> None:
        super().__init__(err, radius, sides)
        self._inside: List[Tuple[float, float, float]] = []      # 源在此圆盘内
        self._outside: List[Tuple[float, float, float]] = []     # 源在此圆盘外
        self._applied_in = 0
        self._applied_out = 0
        self._cache: Dict[str, Any] = {}

    # ---- 硬约束 ----
    def add_inside(self, x: float, y: float, r: float) -> "ProbRegion":
        """叠加"源在以 (x, y) 为心、r 为半径的圆盘内"。"""
        self._inside.append((float(x), float(y), float(r)))
        return self

    def add_outside(self, x: float, y: float, r: float) -> "ProbRegion":
        """叠加"源在以 (x, y) 为心、r 为半径的圆盘外"。"""
        self._outside.append((float(x), float(y), float(r)))
        return self

    def disc(self, x: float, y: float, r: float, inscribed: bool) -> Polygon:
        """圆盘的正多边形近似：inscribed=True 内接（⊆ 真圆盘），False 外接（⊇ 真圆盘）。"""
        rr = float(r) if inscribed else float(r) / math.cos(math.pi / (4.0 * EXCL_QUAD))
        return Point(float(x), float(y)).buffer(rr, quad_segs=EXCL_QUAD)

    # ---- 几何：楔形交（父类增量）之上再叠加圆盘约束（同样增量、惰性）----
    @property
    def region(self):
        geom = super().region
        if self._applied_in < len(self._inside) or self._applied_out < len(self._outside):
            for x, y, r in self._inside[self._applied_in:]:
                geom = geom.intersection(self.disc(x, y, r, False))
            self._applied_in = len(self._inside)
            for x, y, r in self._outside[self._applied_out:]:
                geom = geom.difference(self.disc(x, y, r, True))
            self._applied_out = len(self._outside)
            self._region = geom                 # 覆盖父类缓存，后续增量楔形交由此继续
            self._cache.clear()
        return self._region

    def _sig(self) -> tuple:
        return (len(self._nodes), self._done, self._applied_in, self._applied_out)

    def _memo(self, key: str, compute):
        self.region                             # 先让几何追平，再取签名
        sig = self._sig()
        if self._cache.get("sig") != sig:
            self._cache = {"sig": sig}
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    @property
    def vertices(self):
        """区域顶点：单块取外环，多块（被禁区切开）时收集各块外环顶点。"""
        geom = self.region
        if geom.is_empty:
            return []
        if geom.geom_type == "Polygon":
            parts = [geom]
        elif geom.geom_type == "MultiPolygon":
            parts = list(geom.geoms)
        else:
            return []
        pts: List[Tuple[float, float]] = []
        for poly in parts:
            pts.extend((x, y) for x, y, *_ in poly.exterior.coords[:-1])
        if len(parts) == 1:
            return pts if parts[0].exterior.is_ccw else pts[::-1]
        return pts

    @property
    def diameter(self):
        return self._memo("diameter", lambda: TriangulationRegion.diameter.fget(self))

    @property
    def enclosing_circle(self):
        def compute():
            if not self.vertices:
                return None
            center = shapely.minimum_bounding_circle(self.region).centroid
            return (center.x, center.y, float(shapely.minimum_bounding_radius(self.region)))
        return self._memo("mec", compute)

    def min_distance_to(self, p: Sequence[float]) -> float:
        """区域（可能源集合）到点 p 的最小距离 / m；区域为空时返回 0。"""
        geom = self.region
        if geom.is_empty:
            return 0.0
        return float(geom.distance(Point(float(p[0]), float(p[1]))))


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


def hypothesis_points(region, obs_list: Sequence[Obs],
                      start: Sequence[float] = (0.0, 0.0)) -> List[Tuple[float, float]]:
    """补测选点用的"假设源位置"集合。

    区域已知时统一取"最小覆盖圆圆心 + 最远点采样出的若干顶点"（单条射线也适用：叠加 no_signal
    禁区与接收半径环带后，区域不再是无限长的一条带，而是有限的一段）；区域退化时（尚无禁区
    约束）才退回"沿那条射线按可能距离枚举"。
    """
    pts: List[Tuple[float, float]] = []
    mec = region.enclosing_circle
    verts = list(region.vertices)
    if verts:
        # 区域已知（含 no_signal 禁区与接收半径环带）：最小覆盖圆心 + 最远点采样若干顶点
        if mec is not None:
            pts.append((mec[0], mec[1]))
        while len(pts) < HYP_MAX and verts:
            far = max(verts, key=lambda v: min(dist(v, q) for q in pts))
            verts.remove(far)
            pts.append(far)
    elif len(obs_list) == 1:
        # 区域退化（尚无禁区约束）：只能沿那条射线按可能距离枚举
        o = obs_list[0]
        for s in SINGLE_HYP:
            hx = o.x + s * math.cos(math.radians(o.theta))
            hy = o.y + s * math.sin(math.radians(o.theta))
            if math.hypot(hx, hy) <= REGION_RADIUS:
                pts.append((hx, hy))
    # 与已有检测点太近的假设点无法估计距离（r → 0），剔除；并去重
    keep: List[Tuple[float, float]] = []
    for h in pts:
        if any(dist(h, (o.x, o.y)) < HYP_GAP for o in obs_list):
            continue
        if any(dist(h, k) < 1.0 for k in keep):
            continue
        keep.append(h)
    return keep


class Probe(NamedTuple):
    """一个补测候选点：坐标、对全部假设源位置的最坏预测 σ、从当前位置出发的里程。"""

    x: float
    y: float
    sigma: float
    travel: float


def probe_candidates(obs_list: Sequence[Obs], hyps: Sequence[Sequence[float]],
                     pos: Sequence[float]) -> List[Probe]:
    """按文献准则给补测点排序：最小化"对全部假设源位置的平均预测 σ"。

    候选点 = 每个假设源位置周围若干半径（PROBE_RADII，均 < 1000 m，保证落在源的有效接收
    半径内）× 若干方位角（PROBE_ANGLES）的环上点。评价用 Fisher 信息口径的预测 σ：
    该准则同时实现了文献的两条结论 —— 检测点越接近源 σ 越小（Chen 的定理：均方位置误差
    ∝ 1/(σ²r²)），且新射线与已有射线的交角越接近正交 σ 越小（角度分集）；单射线时
    J 退化，准则自动把补测点放到能"定距"的位置上，即完成单射线→双射线的补测。

    共线候选（与已有射线夹角 ≈ 0，σ = ∞，无法定距）直接剔除 —— 单射线频道的假设点排成一条
    直线，若用"对全部假设取最坏 σ"排序会退化成"所有候选都不可用"，从而按里程误选到共线上的
    点（实测踩过：测了 26 次仍没缩小区域）；取平均 σ 既保留"靠近源 + 拉开交角"的偏好，
    又不会因个别极远假设把好点全部否掉。

    排序规则（确定性）：先按平均 σ 升序；σ 与最优值相差 2% 以内的候选视为"同等好"，
    其中取里程最短者（省时间），里程并列时取坐标字典序最小者。返回前 PROBE_TRY 个候选。
    """
    base = [(o.x, o.y, o.theta) for o in obs_list]      # 已有观测的（检测点, 示向度）
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
                for h in hyps:                          # 对全部假设源位置求平均 σ（见下）
                    sig = fisher_sigma(h, base + [(qx, qy, bearing((qx, qy), h))])
                    if sig == math.inf:                 # 与已有射线共线：不提供任何新信息
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
    该式在"基线 R 固定、目标位置自由"的口径下取最小值；与本文"检测点自由、最小化预测
    协方差"的口径不同，代码中保留此函数以便论文同时给出两种准则的结论。
    """
    a1, a2 = math.radians(alpha1_deg), math.radians(alpha2_deg)
    s = math.sin(a1) * math.sin(a2)
    den = math.sin(a1 + a2) ** 3
    return 4.0 * baseline ** 2 * math.radians(err_deg) ** 2 * s / den


# ----------------------------------------------------------------------------
# 机器狗：巡视扫描 → 对每个频道就近试清（未命中则按文献准则补测缩小后再清）
# ----------------------------------------------------------------------------
class RobotDog:
    """两阶段机器狗。

    阶段一（巡视扫描）：按覆盖圆方案依次走到 7 个圆心，在每个圆心对未采够的频道测向，把
    每个源的示向度采集齐全（同一地点误差固定，故每频道最多采 OBS_CAP 条）。

    阶段二（定位与清除）：对每个频道走同一条流程，**没有"够不够准"的清除门槛** ——
      1. 用问题 1 的交会定位区域（各 ±1° 楔形之交 ∩ 圆域）得到位置估计（区域最小覆盖圆圆心）；
      2. **就近试清**：走到估计点直接 /clear 一次。清除半径只有 20 m，走到源的近处本来就是
         清除的必要动作，所以到达后顺手一试的边际代价只有失败时的 3 s，命中却省掉整轮补测；
         未命中时就地复测（该点是区域内离源最近、Fisher 权重最大的位置）；
      3. 若估计仍不够集中（直径 ≥ 40 m），按文献准则在海选候选点里补测，把区域压到
         直径 < 40 m 以下，再到新的估计点试清；单射线频道由此补齐第二视角；
      4. 兜底：万一仍未清除，沿最新实测示向度以 HOMING_STEP 步长逼近（确定性，无随机）。
    其中"直径 < 40 m"只是**补测的收工条件**（表示估计已够准），不决定清不清除。
    """

    def __init__(self, sim, verbose: bool = True, logfile: Optional[str] = None,
                 episode: int = 0, clear: bool = True,
                 k_clear_max: int = K_CLEAR_MAX,
                 inline_try_radius: float = INLINE_TRY_RADIUS,
                 inline_detour: float = INLINE_DETOUR) -> None:
        self.sim = sim
        self.verbose = verbose
        self.clear_enabled = clear
        self.k_clear_max = int(k_clear_max)
        self.inline_try_radius = float(inline_try_radius)    # 顺路试清允许的覆盖圆半径上限 / m
        self.inline_detour = float(inline_detour)            # 顺路试清允许的绕行里程上限 / m
        self._logfile = open(logfile, "a", encoding="utf-8") if logfile else None
        self.obs: Dict[int, List[Obs]] = defaultdict(list)
        self.meas: Dict[int, List[Meas]] = defaultdict(list)   # 全部测量（含 no_signal）
        self.n_skip = 0                                        # 判定必无信号而跳过的测量次数
        self.n_inline_fail = 0                                 # 巡视途中顺路试清白跑的次数
        self.initial_scan: Dict[str, Any] = {}                 # 起始全频道扫描的统计
        self.actions: List[Dict[str, Any]] = []                # 逐次动作记录（供轨迹图/轨迹表）
        self.regions: Dict[int, ProbRegion] = {}
        self.tracks: Dict[int, Dict[str, Any]] = {}      # 逐频道的定位/清除档案
        self.cleared: set = set()
        self.pos = np.zeros(2)
        self.vt = 0.0
        self.n_measure = 0
        self.n_clear = 0
        self.episode = episode
        self.deadline = float("inf")
        self.waypoint_stats: List[Dict[str, Any]] = []   # 逐圆心扫描统计
        self.travel_m = 0.0                              # 实际走过的里程 / m
        self.stage = "survey"                            # 观测所处阶段（survey / refine）

    # ---- 日志 ----
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
        if self._logfile:
            print(msg, file=self._logfile, flush=True)

    def close(self) -> None:
        """关闭日志文件（幂等）。"""
        if self._logfile:
            self._logfile.close()
            self._logfile = None

    # ---- 原子动作 ----
    def measure(self, x: float, y: float, channel: int) -> dict:
        """测向；direction 时记录示向度并同步进该频道的定位区域。"""
        x, y = float(x), float(y)
        r = self.sim.measure(x, y, channel)
        if not r.get("accepted"):
            raise RuntimeError(f"/measure 被拒绝：{r}")
        self.travel_m += dist(self.pos, (x, y))
        self.pos, self.vt = np.array([x, y]), float(r["virtual_time_s"])
        self.n_measure += 1
        outcome = r.get("measure_result", "no_signal")
        reg = self.region(channel)          # 首次创建会重放此前测量，故必须先取再追加
        m = Meas(channel, x, y, outcome,
                 float(r["svd_deg"]) if outcome == "direction" else None, self.stage)
        self.meas[channel].append(m)
        self._apply_meas(reg, m)
        if outcome == "direction":
            self.obs[channel].append(Obs(channel, x, y, m.theta, self.stage))
        self._note("measure", x, y, channel, outcome=outcome, theta=m.theta)
        return r

    def _provable_no_signal(self, channel: int, at: Sequence[float]) -> bool:
        """能否证明"在 at 处测 channel 必然无信号"，从而省掉这次测量。

        可能源集合是真实源位置的超集，故"区域到 at 的最小距离 > 1500 m"⇒ 真源到 at 的距离也
        > 1500 m ≥ 有效接收半径 ⇒ 必然收不到信号。此时这次测量不会带来任何新信息，直接跳过。
        """
        reg = self.regions.get(channel)
        if reg is None or reg.region.is_empty:
            return False
        return reg.min_distance_to(at) > RECEIVE_MAX + 1.0

    def clear(self, x: float, y: float, channel: int) -> bool:
        """清除；返回是否成功。"""
        x, y = float(x), float(y)
        r = self.sim.clear(x, y, channel)
        if not r.get("accepted"):
            raise RuntimeError(f"/clear 被拒绝：{r}")
        self.travel_m += dist(self.pos, (x, y))
        self.pos, self.vt = np.array([x, y]), float(r["virtual_time_s"])
        self.n_clear += 1
        ok = r.get("clear_result") == "success"
        if ok:
            self.cleared.add(channel)
            self.tracks.setdefault(channel, {})["clear_point"] = [x, y]
        self._note("clear", x, y, channel, outcome="success" if ok else "no_target_in_range")
        return ok

    def _note(self, kind: str, x: float, y: float, channel: int,
              outcome: Optional[str] = None, theta: Optional[float] = None) -> None:
        """登记一次动作（/measure 或 /clear），供逐局轨迹图与轨迹表使用。

        记的是**动作点**（机器狗实际到达的坐标），与日志逐点对应；画图与落盘都在 /exit 之后
        进行，不占用现实时间预算，也不影响任何实时决策。
        """
        self.actions.append({
            "seq": len(self.actions), "kind": kind, "stage": self.stage,
            "x": float(x), "y": float(y), "channel": int(channel),
            "outcome": outcome, "theta": theta,
            "virtual_time_s": round(self.vt, 3), "travel_m": round(self.travel_m, 2),
        })

    def _out_of_time(self) -> bool:
        return time.monotonic() > self.deadline

    def region(self, channel: int) -> ProbRegion:
        """取该频道的"可能源集合"（ProbRegion）。

        惰性创建：首次访问时把该频道此前的全部测量（含 no_signal）作为硬约束补进去；之后每次
        measure() 直接叠加新约束。ProbRegion 对楔形交与圆盘约束都做增量缓存，故反复读直径或
        最小覆盖圆只会补上新增的那几条。
        """
        if channel not in self.regions:
            reg = ProbRegion(err=BEARING_ERROR_DEG, radius=REGION_RADIUS, sides=CLIP_SIDES)
            for m in self.meas.get(channel, ()):
                self._apply_meas(reg, m)
            self.regions[channel] = reg
        return self.regions[channel]

    @staticmethod
    def _apply_meas(reg: ProbRegion, m: Meas) -> None:
        """把一次测量的结果转成硬约束叠加到可能源集合上（依据见 ProbRegion 的类文档）。"""
        if m.outcome == "direction":
            reg.add_node(m.x, m.y, float(m.theta))
            reg.add_inside(m.x, m.y, RECEIVE_MAX)      # 收得到 ⇒ 源在接收半径上限之内
        elif m.outcome == "near":
            reg.add_inside(m.x, m.y, NEAR_RADIUS)      # 5 m 内 ⇒ 位置几乎确定
        elif m.outcome == "no_signal":
            reg.add_outside(m.x, m.y, COVER_RADIUS)    # 收不到 ⇒ 源在接收半径下界之外

    def diameter(self, channel: int) -> float:
        """该频道当前定位区域的直径 / m（0 表示区域为空/退化）。"""
        return float(self.region(channel).diameter)

    def _precise(self, channel: int) -> bool:
        """判据：可能源集合的最小覆盖圆半径 + 裁剪误差 < 20 m（= 清除半径）。

        注意它**不是清除门槛**：只回答"估计是否已经够准、可以停止补测"；清除一律由"走到估计点
        先试一次 /clear"决定（见 _nearby_try_clear）。

        这里用最小覆盖圆半径而不是"直径 < 40 m"：直径判据的依据是"凸区域的最小覆盖圆半径 ≤
        直径/2"，而叠加 no_signal 禁区后区域可能非凸、甚至裂成多块，凸性不再成立；直接看覆盖圆
        半径对任何集合都成立 —— 真源必在区域内 ⊆ 覆盖圆内，故走到圆心必在 20 m 以内。
        """
        mec = self.region(channel).enclosing_circle
        return mec is not None and mec[2] + CLIP_ERR < CLEAR_RADIUS

    # ---- 阶段 1：巡视扫描 ----
    def _active_channels(self) -> List[int]:
        """仍需测向的频道：未清除、示向度条数未达上限、且**定位还不够准**。

        第三条是关键：叠加 no_signal 禁区与接收半径环带后，很多频道两条射线就已把可能源集合压到
        覆盖圆半径 < 20 m，此时再测纯属浪费（每站 5 s 测向 + 可能的 1 s 切换）—— 实测前 3 站
        每站都要把 20 个频道全测一遍，而每站只有 4~6 次能测出方向。
        """
        return [c for c in CHANNELS
                if c not in self.cleared
                and len(self.obs.get(c, ())) < OBS_CAP
                and not self._precise(c)]

    def _sweep(self, channels: Sequence[int], at: Sequence[float]) -> Dict[str, int]:
        """在 at 处按频道号升序逐频道测向（升序可省频道切换时间）；near 就地清除。"""
        counts = {"direction": 0, "near": 0, "no_signal": 0, "skip": 0}
        for ch in sorted(channels):
            if self._out_of_time():
                break
            if self._provable_no_signal(ch, at):
                counts["skip"] += 1          # 区域整体在 1500 m 之外：必然无信号，不必测
                self.n_skip += 1
                continue
            res = self.measure(at[0], at[1], ch).get("measure_result", "no_signal")
            counts[res] = counts.get(res, 0) + 1
            if res == "near":
                self.log(f"    [near] 频道{ch}：源在 5 m 内，就地清除"
                         f"{'成功' if self.clear(at[0], at[1], ch) else '失败'}")
        return counts

    def survey(self, waypoints: np.ndarray, order: Sequence[int]) -> None:
        """依次移动到各圆心并扫描：阶段一的主体。

        第 1 站（原点）就是"起始全频道扫描"：一次把 20 个频道全测一遍，成本 20 次测向
        （≈ 20×5 s + 19×1 s 切换 ≈ 119 s），换来的是"哪些频道在 1000 m 内"这一批最便宜的信息
        —— 实测平均能直接锚定 6 个源的方向，同时把其余频道标记为"源在 1000 m 之外"（这条对后续
        跳过测量与区域收缩都有用）。扫描结束后立刻用这批方位调整后续路径的绕向与落脚点。
        """
        order = list(order)
        self.log(f"阶段1 巡视扫描：依次访问 {len(order)} 个圆心"
                 f"（顺序 {' → '.join(str(i) for i in order)}）")
        for step_i, idx in enumerate(order, 1):
            wp = waypoints[idx]
            active = self._active_channels()
            if not active:
                self.log(f"  第 {step_i} 站：圆心 {idx} @ ({wp[0]:.1f}, {wp[1]:.1f})，"
                         f"所有频道已采够或已清除，巡视提前结束")
                break
            if self._out_of_time():
                self.log(f"  第 {step_i} 站：现实时间不足，巡视提前结束")
                break
            self.log(f"  第 {step_i} 站：圆心 {idx} @ ({wp[0]:.1f}, {wp[1]:.1f})，"
                     f"扫描 {len(active)} 个频道")
            counts = self._sweep(active, wp)
            self.waypoint_stats.append({
                "waypoint": int(idx), "x": float(wp[0]), "y": float(wp[1]),
                "n_channels": len(active), **counts,
                "virtual_time_s": round(self.vt, 3), "travel_m": round(self.travel_m, 2),
            })
            self.log(f"    有示向度 {counts['direction']}、无信号 {counts['no_signal']}、"
                     f"近距清除 {counts['near']}、判定必无信号而跳过 {counts['skip']}，"
                     f"累计里程 {self.travel_m:.0f} m，虚拟时刻 {self.vt:.0f} s")
            if step_i == 1:
                self.initial_scan = dict(counts, n_channels=len(active))
                self.log(f"    [起始全频道扫描] 一次扫完 20 个频道：锚定 {counts['direction']} "
                         f"个源的方向，其余 {counts['no_signal']} 个判定为源在 "
                         f"{COVER_RADIUS:.0f} m 之外")
                self._orient_route(waypoints, order, wp)
            if step_i < len(order):
                self._inline_clear(wp, waypoints[order[step_i]])
        self.log(f"阶段1 完成：里程 {self.travel_m:.0f} m，虚拟时刻 {self.vt:.0f} s，"
                 f"累计示向度 {sum(len(v) for v in self.obs.values())} 条，"
                 f"途中顺路清除 {len(self.cleared)} 个")

    def _orient_route(self, waypoints: np.ndarray, order: List[int],
                      origin: Sequence[float]) -> None:
        """起始扫描之后，用听到的源方位调整巡视路径的绕向与落脚点（零成本）。

        7 个圆心的访问长度恒为 6d（原点 → 一个环顶点 = d，再沿正六边形走 5 条边 = 5d），因此
        "从哪个环顶点开始、顺时针还是逆时针"共 12 条路径**长度完全相同**。选择依据：让巡视的
        最后落脚点靠近"起始扫描已经听到的源"（这些是最先能清的源），清除阶段就能直接从它们开始，
        不必为它们单独折返。
        """
        ring = [i for i in order if i != 0]
        if len(ring) < 3 or not self.obs:
            return
        # 起始扫描听到的源：方位已知、距离未知，取接收半径区间中点作估计
        est = []
        for ml in self.meas.values():
            for m in ml:
                if m.theta is None or dist((m.x, m.y), origin) > 1e-6:
                    continue
                est.append(np.array([origin[0] + RECEIVE_MID * math.cos(math.radians(m.theta)),
                                     origin[1] + RECEIVE_MID * math.sin(math.radians(m.theta))]))
        if not est:
            return
        best, best_score = None, None
        for s in range(len(ring)):
            for direction in (1, -1):
                path = [ring[(s + direction * k) % len(ring)] for k in range(len(ring))]
                end = np.asarray(waypoints[path[-1]], dtype=float)
                score = float(np.mean([np.linalg.norm(end - p) for p in est]))
                if best_score is None or score < best_score - 1e-9:
                    best, best_score = path, score
        if best is None:
            return
        new_order = [order[0]] + best
        if new_order != list(order):
            old_end = np.asarray(waypoints[order[-1]], dtype=float)
            self.log(f"    [线路] 起始扫描已锚定 {len(est)} 个源的方向，据此把巡视绕向与落脚点从 "
                     f"圆心{order[-1]} 调整到圆心{new_order[-1]}"
                     f"（路径长度不变，终点到已听源的估计距离 "
                     f"{np.mean([np.linalg.norm(old_end - p) for p in est]):.0f} → "
                     f"{best_score:.0f} m）")
        order[:] = new_order

    def _inline_clear(self, at: Sequence[float], next_wp: Sequence[float]) -> int:
        """巡视途中顺路清除：估计点就在路线上（绕行代价小）的频道，当场清掉。

        巡视本来就要路过这些位置，绕一下的额外里程 ≤ INLINE_DETOUR；而留到阶段二再清，至少要从
        别处专程跑一趟（几百米）。清掉后该频道后续各站都不再测量，双向省时间。
        """
        if not self.clear_enabled:
            return 0
        n = 0
        for ch in sorted(self.obs):
            if ch in self.cleared or self._out_of_time():
                continue
            mec = self.region(ch).enclosing_circle
            if mec is None or mec[2] + CLIP_ERR >= self.inline_try_radius:
                continue                                  # 估计太离谱就不值得绕
            est = (mec[0], mec[1])
            extra = (dist(at, est) + dist(est, next_wp) - dist(at, next_wp))
            if extra > self.inline_detour:
                continue
            self.stage = "survey-clear"
            if self.clear(est[0], est[1], ch):
                self.tracks.setdefault(ch, {}).update({"method": "survey-inline",
                                                       "clear_point": [est[0], est[1]]})
                n += 1
                self.log(f"    [顺路清除] 频道{ch} @ ({est[0]:.1f}, {est[1]:.1f}) 命中"
                         f"（绕行 {extra:.0f} m，估计误差 "
                         f"{dist(est, (mec[0], mec[1])):.0f} m，省下专程往返）")
            else:
                self.n_inline_fail += 1
                self.log(f"    [顺路清除] 频道{ch} @ ({est[0]:.1f}, {est[1]:.1f}) 未命中"
                         f"（绕行 {extra:.0f} m、覆盖圆半径 {mec[2]:.0f} m，留到阶段二处理）")
        if n:
            self.log(f"    本段顺路清除 {n} 个，累计已清 {len(self.cleared)} 个")
        return n

    # ---- 阶段 2a：巡视后的定位诊断（只记录，不改变处理流程）----
    def diagnose(self) -> Dict[str, int]:
        """记录每个频道巡视后的定位区域直径/有界性，并统计"估计已够准"的个数。

        这是纯诊断：清除流程对所有频道一视同仁（都先就近试清），不按直径分叉。
        """
        precise = 0
        for ch in sorted(self.obs):
            if ch in self.cleared:
                continue
            d = self.diameter(ch)
            mec = self.region(ch).enclosing_circle
            self.tracks.setdefault(ch, {}).update({
                "n_obs_survey": len(self.obs[ch]),
                "n_no_signal_survey": sum(1 for m in self.meas.get(ch, ())
                                          if m.stage == "survey" and m.outcome == "no_signal"),
                "diameter_survey_m": round(d, 3),
                "mec_radius_survey_m": round(mec[2], 3) if mec else None,
                "bounded_survey": bool(self.region(ch).bounded),
                "n_probe": 0,
            })
            precise += self._precise(ch)
        self.log(f"阶段2 定位与清除：{len(self.obs)} 个频道，其中巡视后估计已够准"
                 f"（最小覆盖圆半径 < {CLEAR_RADIUS:.0f} m，诊断值）{precise} 个；"
                 f"全部频道都走同一流程：就近试清 → 未命中则补测缩小再清")
        return {"n_channels": len(self.obs), "n_precise": precise,
                "n_skip": self.n_skip}

    def _nearest_order(self, channels: Sequence[int]) -> List[int]:
        """确定性的访问顺序：最近邻给出初值，再用 2-opt 精修（固定起点、只接受严格下降）。

        访问点取各频道定位区域的最小覆盖圆圆心（清缺点），起点为机器狗当前位置。2-opt 是
        纯确定性精修（同一输入必得同一顺序），用于压缩"逐个清除"的里程 —— 里程占虚拟总
        时间的近九成，是本题时间指标的主要矛盾。
        """
        pts: Dict[int, np.ndarray] = {}
        for c in channels:
            mec = self.region(c).enclosing_circle
            pts[c] = np.array([mec[0], mec[1]]) if mec else np.array(self.pos, dtype=float)
        rest, order, cur = list(channels), [], np.array(self.pos, dtype=float)
        while rest:                                     # 最近邻初值
            nxt = min(rest, key=lambda c: (float(np.linalg.norm(pts[c] - cur)), c))
            order.append(nxt)
            rest.remove(nxt)
            cur = pts[nxt]
        return self._two_opt(order, pts, self.pos)

    @staticmethod
    def _two_opt(order: List[int], pts: Dict[int, np.ndarray],
                 start: Sequence[float]) -> List[int]:
        """开放路径 2-opt：反复尝试反转一段子路径，只接受里程严格下降的改动。"""
        def length(seq: Sequence[int]) -> float:
            total, cur = 0.0, np.array(start, dtype=float)
            for c in seq:
                total += float(np.linalg.norm(pts[c] - cur))
                cur = pts[c]
            return total
        best, cur_len = list(order), length(order)
        improved = True
        while improved:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    cand_len = length(cand)
                    if cand_len < cur_len - 1e-9:
                        best, cur_len, improved = cand, cand_len, True
        return best

    # ---- 阶段 2b：按文献准则补测缩小定位区域 ----
    def refine(self, channel: int) -> int:
        """补测直到定位区域直径 < 40 m（或达到轮次/候选上限）；返回实际补测次数。"""
        n_probe = 0
        for _ in range(REFINE_MAX):
            if self._precise(channel) or self._out_of_time():
                break
            hyps = hypothesis_points(self.region(channel), self.obs[channel], self.pos)
            if not hyps:
                break
            cands = probe_candidates(self.obs[channel], hyps, self.pos)
            if not cands:
                break
            d0 = self.diameter(channel)
            got = False
            for c in cands:                       # 依次试候选：收不到信号就换下一个
                if self._out_of_time():
                    break
                self.stage = "refine"
                res = self.measure(c.x, c.y, channel).get("measure_result", "no_signal")
                n_probe += 1
                self.tracks.setdefault(channel, {})["n_probe"] = n_probe
                if res == "direction":
                    self.log(f"    [补测] 频道{channel} @ ({c.x:.0f}, {c.y:.0f})："
                             f"预测 σ {c.sigma:.2f} m，直径 {d0:.1f} → "
                             f"{self.diameter(channel):.1f} m")
                    got = True
                    break
                if res == "near":
                    self.log(f"    [补测] 频道{channel} @ ({c.x:.0f}, {c.y:.0f})："
                             f"源在 5 m 内，就地清除"
                             f"{'成功' if self.clear(c.x, c.y, channel) else '失败'}")
                    self.tracks.setdefault(channel, {})["method"] = "near@probe"
                    return n_probe
                self.log(f"    [补测] 频道{channel} @ ({c.x:.0f}, {c.y:.0f})：无信号，换候选点")
            if not got:
                break
        return n_probe

    # ---- 阶段 2c：清除 ----
    def _homing(self, channel: int) -> bool:
        """兜底：沿最新实测示向度以 HOMING_STEP 步长逼近，直到清除成功（确定性）。

        步数按"当前位置到定位区域最远顶点的距离"自适应（下限 HOMING_MAX、上限 HOMING_CAP）：
        起点离源很远时不能只走固定几步就放弃 —— 实测踩过：起点 636 m 远、只走 24×16 = 384 m
        就停手，把本可清掉的源漏掉。
        """
        budget = HOMING_MAX
        verts = self.region(channel).vertices
        if verts:
            far = max(dist(self.pos, v) for v in verts)
            budget = int(min(HOMING_CAP, max(HOMING_MAX, far / HOMING_STEP + 4.0)))
        p = np.array(self.pos, dtype=float)
        for _ in range(budget):
            if self._out_of_time():
                return False
            self.stage = "clear"
            r = self.measure(p[0], p[1], channel)
            res = r.get("measure_result")
            if res == "near":
                return self.clear(p[0], p[1], channel)
            if res != "direction":
                return False
            theta = math.radians(float(r["svd_deg"]))
            nxt = clamp_to_region(p[0] + HOMING_STEP * math.cos(theta),
                                  p[1] + HOMING_STEP * math.sin(theta))
            if self.clear(nxt[0], nxt[1], channel):
                return True
            p = np.array(nxt, dtype=float)
        return False

    def _try_clear(self, channel: int, tag: str) -> Optional[str]:
        """走到定位区域的最小覆盖圆圆心（当前的位置估计），就地 /clear 一次。

        成功返回 tag，失败返回 None。这是全局唯一的清除位置：区域最小覆盖圆半径 < 20 m 时
        它必然命中（区域内任一点到圆心 ≤ 该半径），估计略差时也常常命中，失败只花 3 s。
        """
        mec = self.region(channel).enclosing_circle
        if mec is None:
            return None
        cx, cy, r = mec
        if self.clear(cx, cy, channel):
            self.tracks.setdefault(channel, {})["clear_radius_m"] = round(r, 3)
            self.log(f"    [清除] 频道{channel} @ ({cx:.1f}, {cy:.1f}) 命中，"
                     f"最小覆盖圆半径 {r:.2f} m")
            return tag
        return None

    def _nearby_try_clear(self, channel: int) -> Optional[str]:
        """就近试清：走到当前估计点，直接 /clear 一次；未命中则就地复测。

        为什么"试"而不是"先判断能不能清"。清除半径只有 20 m，**走到源的近处是清除的必要
        动作**，绕不过去；因此到达估计点后顺手试一次 /clear，边际代价只有失败时的 3 s，命中
        却能省掉整轮补测（一轮补测要绕几百米，约 100 s 量级的里程）。相比之下，"先把定位区域
        压到直径 < 40 m 再清"要额外花探针去换那个确定性，反而更贵 —— 所以不再用直径判据决定
        清不清，直径只用来决定还要不要继续补测。

        唯一的前提（"就近"之意）：估计点不能太离谱，才值得跑过去试 ——
        ① 定位区域有界（方向由示向度真正约束住，而不是被作业圆域截断后"随便落"），且
        ② 最小覆盖圆半径 ≤ TRY_CLEAR_RADIUS（估计够集中）。
        否则先按文献准则就地补测一次把区域压小，再过去：跑一趟很远的错点比就近补测更贵
        （实测：放开 ① 后平均里程多 275 m）。

        未命中时就地复测也不是白花的：该点是区域内"离源最近"的位置，按 Fisher 信息口径权重
        最大，一条近距离射线往往直接把区域压到 40 m 以内。
        """
        if not self.clear_enabled or self._out_of_time():
            return None
        mec = self.region(channel).enclosing_circle
        if mec is None or mec[2] > TRY_CLEAR_RADIUS:
            return None
        if not self.region(channel).bounded:     # 区域被圆域截断：方向还存在"跑掉"的可能
            return None
        if self._try_clear(channel, "try"):
            return "try"
        rec = self.tracks.setdefault(channel, {})
        rec["n_probe"] = int(rec.get("n_probe", 0)) + 1
        self.stage = "refine"
        cx, cy = mec[0], mec[1]
        res = self.measure(cx, cy, channel).get("measure_result", "no_signal")
        if res == "direction":
            self.log(f"    [就近试清] 频道{channel} @ ({cx:.1f}, {cy:.1f}) 未命中，"
                     f"就地复测（区域中离源最近、信息量最大的点）：直径 "
                     f"{rec.get('diameter_survey_m', float('nan'))} → "
                     f"{self.diameter(channel):.1f} m")
        elif res == "near" and self.clear(cx, cy, channel):
            return "near@center"
        return None

    def _k_cover_points(self, channel: int, k_extra: int,
                        radius: float = CLEAR_RADIUS) -> Tuple[List[Tuple[float, float]], float]:
        """用"当前估计点已清过一次"为起点，再贪心选 k_extra 个清除点，使半径 radius 的圆尽量
        覆盖整个定位区域；返回（补充清除点列表, 未被覆盖的目标点比例）。

        思路（"区域大就多清几次"）：清除半径只有 20 m，一个点保证不了命中时就多清几个点 ——
        只要这几个半径 20 m 的圆把定位区域盖满，逐个清过去必然命中，省掉一轮补测（补测要绕
        几百米、约 100 s 量级里程）。补清一个点的边际代价只有一个点间距的移动（≤ 40 m ≈ 8 s）
        加失败时的 3 s，比补测便宜一个量级。

        点怎么选：在区域内取细网格作为目标点，候选点同样取区域内的网格；估计点（已经去过、
        已失败）的圆所覆盖的目标点先划掉，然后每轮选"新增覆盖目标点最多"的候选点（贪心最大
        覆盖，与覆盖圆求解里的贪心集合覆盖同一手法）。比例 = 0 表示覆盖完整，可以保证命中。
        """
        region = self.region(channel)
        mec = region.enclosing_circle
        if mec is None or region.region.is_empty:
            return [], 1.0
        b = region.region.bounds
        span = max(b[2] - b[0], b[3] - b[1])
        step = max(K_COVER_STEP_MIN, span / K_COVER_SAMPLES)
        gx, gy = np.meshgrid(np.arange(b[0], b[2] + 1e-9, step),
                             np.arange(b[1], b[3] + 1e-9, step))
        grid = np.stack((gx.ravel(), gy.ravel()), axis=1)
        inside = np.array([region.contains(p) for p in grid])
        targets = grid[inside]
        if len(targets) == 0:
            return [], 1.0
        cand = targets
        if len(cand) > 2500:                    # 候选点抽稀，控制距离矩阵规模
            cand = cand[:: len(cand) // 2500 + 1]

        covered = np.linalg.norm(targets - np.asarray(mec[:2]), axis=1) <= radius + TOL
        points: List[Tuple[float, float]] = []
        for _ in range(k_extra):
            if covered.all():
                break
            dists = np.linalg.norm(cand[:, None, :] - targets[None, :, :], axis=2)
            reach = (dists <= radius + TOL) & ~covered[None, :]
            gain = reach.sum(axis=1)
            j = int(gain.argmax())
            if gain[j] == 0:
                break
            points.append((float(cand[j][0]), float(cand[j][1])))
            covered |= dists[j] <= radius + TOL
        return points, float(1.0 - covered.mean())

    def _multi_try_clear(self, channel: int) -> Optional[str]:
        """试清未中后就地"多清几次"：用至多 k_clear_max 个半径 20 m 的圆覆盖定位区域，依次补清。

        只在"这几个圆能把区域盖满"时才动手（否则白跑，直接交给补测）；顺序按最近邻，从当前
        位置由近及远，命中即停。返回方法标签，覆盖不全或都没命中时返回 None。
        """
        if not self.clear_enabled or self._out_of_time():
            return None
        pts, leftover = self._k_cover_points(channel, self.k_clear_max)
        if not pts or leftover > 1e-9:          # 盖不满整个区域：不值得赌，交给补测
            return None
        rec = self.tracks.setdefault(channel, {})
        rec["k_clear"] = len(pts) + 1
        rec["k_cover_leftover"] = round(leftover, 6)
        self.log(f"    [多清几次] 频道{channel}：{len(pts) + 1} 个半径 {CLEAR_RADIUS:.0f} m 的圆"
                 f"可覆盖整个定位区域，依次补清")
        cur = np.array(self.pos, dtype=float)
        rest = list(pts)
        while rest:                             # 最近邻：从当前位置由近及远
            k = min(range(len(rest)),
                    key=lambda i: (float(np.linalg.norm(np.asarray(rest[i]) - cur)), i))
            x, y = rest.pop(k)
            if self.clear(x, y, channel):
                self.log(f"    [多清几次] 频道{channel} @ ({x:.1f}, {y:.1f}) 命中")
                return "multi"
            cur = np.array([x, y], dtype=float)
        self.log(f"    [多清几次] 频道{channel}：{len(pts) + 1} 个点都未命中，转入补测")
        return None

    def _finish_clear(self, channel: int) -> str:
        """补测之后的收尾：再到新的估计点试清，失败则就地复测，最后兜底沿示向度逼近。"""
        if self._try_clear(channel, "try-refined"):
            return "try-refined"
        mec = self.region(channel).enclosing_circle
        if mec is not None:
            cx, cy = mec[0], mec[1]
            self.log(f"    [清除] 频道{channel} @ ({cx:.1f}, {cy:.1f}) 未命中，就地复测")
            self.stage = "clear"
            if self.measure(cx, cy, channel).get("measure_result") == "near" \
                    and self.clear(cx, cy, channel):
                return "near"
        if self._homing(channel):
            self.log(f"    [清除] 频道{channel} 兜底沿示向度逼近成功")
            return "homing"
        return "failed"

    def process(self, channel: int) -> None:
        """一个频道的完整处理：走到估计点先试清，未命中再补测缩小、然后再清。"""
        rec = self.tracks.setdefault(channel, {})
        if channel in self.cleared:                  # 巡视阶段已就地清除（near）
            rec.update({"method": "survey-near", "cleared": True})
            return
        if not self.clear_enabled:                   # 只定位模式：补测到估计够准为止
            if not self._precise(channel):
                self.refine(channel)
            method: Optional[str] = "skipped"
        else:
            method = self._nearby_try_clear(channel)     # ① 就近试清
            if method is None:                           # ② 未命中：多清几次（几个 20 m 圆盖满区域）
                method = self._multi_try_clear(channel)
            if method is None:                           # ③ 还不行：按文献准则补测缩小范围
                self.refine(channel)
                method = self._finish_clear(channel)
        d = self.diameter(channel)
        mec = self.region(channel).enclosing_circle
        rec.update({
            "diameter_final_m": round(d, 3),
            "precise_final": self._precise(channel),
            "mec_radius_m": round(mec[2], 3) if mec else None,
            "bounded_final": bool(self.region(channel).bounded),
            "n_obs_total": len(self.obs[channel]),
            "method": method,
            "cleared": channel in self.cleared,
        })

    # ---- 主流程 ----
    def run(self, plan: CoverPlan, order: Sequence[int]) -> Dict[str, Any]:
        """/enter → 巡视扫描 → 诊断 → 逐频道就近试清（未命中则补测缩小后再清）→ /exit。"""
        enter = self.sim.enter()
        if not enter.get("accepted"):
            raise RuntimeError(f"/enter 被拒绝：{enter}")
        left = float(enter.get("remaining_real_duration_s", 1200.0))
        self.deadline = time.monotonic() + max(left - SAFETY_MARGIN, 0.0)
        self.log(f"/enter 成功：虚拟时刻 {enter.get('virtual_time_s')} s，"
                 f"现实剩余 {left:.0f} s")
        try:
            self.survey(plan.waypoints, order)
            if self.clear_enabled:
                self.diagnose()
                # 阶段二的访问顺序：按各频道估计点到当前位置的距离做最近邻 + 2-opt（省里程）。
                # 实测"每处理一个就重排"与"一次定序"结果完全相同（补测位移不足以改变最近邻
                # 首元素），故保留更简单的一次定序。
                for ch in self._nearest_order([c for c in sorted(self.obs)
                                               if c not in self.cleared]):
                    if self._out_of_time():
                        break
                    self.process(ch)
        finally:
            try:
                self.sim.exit()
            except OSError as exc:
                self.log(f"    [警告] /exit 失败：{exc}")
            self.close()
        n = len(self.cleared)
        return {
            "waypoints_visited": len(self.waypoint_stats),
            "travel_m": self.travel_m,
            "virtual_time_s": self.vt,
            "n_measure": self.n_measure,
            "n_clear": self.n_clear,
            "channels_heard": len(self.obs),
            "n_bearings": sum(len(v) for v in self.obs.values()),
            "cleared": n,
            "avg_time_s": self.vt / n if n else float("inf"),
            "n_precise_at_survey": sum(1 for r in self.tracks.values()
                                      if r.get("mec_radius_survey_m") is not None
                                      and r["mec_radius_survey_m"] + CLIP_ERR < CLEAR_RADIUS),
            "n_skip_measure": self.n_skip,
            "initial_scan": self.initial_scan,
            "n_inline_cleared": sum(1 for r in self.tracks.values()
                                    if r.get("method") == "survey-inline"),
            "n_inline_fail": self.n_inline_fail,
            "n_refined": sum(1 for r in self.tracks.values() if r.get("n_probe")),
            "n_probe": sum(int(r.get("n_probe", 0)) for r in self.tracks.values()),
            "tracks": self.tracks,
        }


# ----------------------------------------------------------------------------
# 覆盖保证的真值核对（只有演练模式拿得到真值）
# ----------------------------------------------------------------------------
def truth_check(truth: Optional[Sequence[dict]], plan: CoverPlan,
                obs: Dict[int, List[Obs]], cleared: set,
                tracks: Optional[Dict[int, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """逐源核对（仅演练模式拿得到真值）：

    * 覆盖保证：源到最近覆盖圆圆心的距离是否 ≤ 1000 m、它的频道是否真的被听到；
    * 清除结果：清除点与真值的距离（定位误差）、是否落在 20 m 清除半径内。
    """
    tracks = tracks or {}
    rows: List[Dict[str, Any]] = []
    for j in truth or []:
        p = (float(j["position"]["x"]), float(j["position"]["y"]))
        d_near = float(np.linalg.norm(plan.waypoints - np.asarray(p), axis=1).min())
        heard = len(obs.get(j["channel"], ())) > 0 or j["channel"] in cleared
        rec = tracks.get(int(j["channel"]), {})
        cp = rec.get("clear_point")
        err = dist(cp, p) if cp else None
        rows.append({"channel": int(j["channel"]), "x": p[0], "y": p[1],
                     "receive_m": float(j["receive"]),
                     "nearest_center_m": round(d_near, 2),
                     "heard": bool(heard),
                     "n_bearings": len(obs.get(j["channel"], ())),
                     "cleared": j["channel"] in cleared,
                     "diameter_survey_m": rec.get("diameter_survey_m"),
                     "n_probe": rec.get("n_probe", 0),
                     "diameter_final_m": rec.get("diameter_final_m"),
                     "mec_radius_final_m": rec.get("mec_radius_m"),
                     "clear_radius_m": rec.get("clear_radius_m"),
                     "method": rec.get("method"),
                     "localize_err_m": None if err is None else round(err, 2)})
    errs = [r["localize_err_m"] for r in rows if r["localize_err_m"] is not None]
    return {
        "sources": rows,
        "worst_nearest_m": round(max((r["nearest_center_m"] for r in rows), default=0.0), 2),
        "all_within_cover": all(r["nearest_center_m"] <= COVER_RADIUS for r in rows),
        "missed_channels": [r["channel"] for r in rows if not r["heard"]],
        "n_cleared": sum(1 for r in rows if r["cleared"]),
        "localize_err_mean_m": round(float(np.mean(errs)), 3) if errs else None,
        "localize_err_max_m": round(float(np.max(errs)), 3) if errs else None,
        "n_within_clear_radius": sum(1 for e in errs if e <= CLEAR_RADIUS),
        "n_cleared_after_probe": sum(1 for r in rows if r["n_probe"] and r["cleared"]),
    }


# ----------------------------------------------------------------------------
# 本地演练场：拉起 jammers-py 并用其控制台 REST 开一局（与 T3_ga.py 同源实现）
# ----------------------------------------------------------------------------
def _free_port(preferred: int) -> int:
    """取一个可用端口：优先 preferred，被占用则依次向后试 20 个。"""
    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"端口 {preferred}~{preferred + 19} 都被占用")


class PracticeArena:
    """本地演练用的 jammers-py 模拟器：自动拉起进程（或复用已在运行的实例）+ 控制台 REST。"""

    REUSE_PORTS = (8090, 8080)

    def __init__(self, jammers_dir: Path, robot_id: str, robot_port: int = 2026,
                 console_port: int = 8090, countdown: int = 1,
                 reuse_existing: bool = True) -> None:
        self.reuse_existing = reuse_existing
        self.jammers_dir = Path(jammers_dir).resolve()
        self.robot_id = robot_id
        self.robot_port = robot_port
        self.console_port = console_port
        self.countdown = countdown
        self.robot_url = f"http://127.0.0.1:{robot_port}"
        self.console_url = f"http://127.0.0.1:{console_port}"
        self._proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "PracticeArena":
        existing = self._find_existing() if self.reuse_existing else None
        if existing is not None:
            self.console_url, state = existing
            self.robot_port = int(state.get("config", {}).get("robot_port", self.robot_port))
            self.robot_url = f"http://127.0.0.1:{self.robot_port}"
            print(f"检测到已在运行的 jammers-py（控制台 {self.console_url}），直接复用"
                  f"（如需独占实例，加 --no-reuse）")
            return self
        run_py = self.jammers_dir / "run.py"
        if not run_py.exists():
            raise FileNotFoundError(f"未找到 jammers-py 模拟器：{run_py}（用 --jammers-dir 指定）")
        self.console_port = _free_port(self.console_port)
        self.console_url = f"http://127.0.0.1:{self.console_port}"
        self._proc = subprocess.Popen(
            [sys.executable, str(run_py),
             "--robot-port", str(self.robot_port), "--web-port", str(self.console_port),
             "--countdown", str(self.countdown), "--team", self.robot_id],
            cwd=str(self.jammers_dir), stdout=subprocess.DEVNULL)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if self._state(self.console_url) is not None:
                return self
            if self._proc.poll() is not None:
                raise RuntimeError(f"jammers-py 启动失败：机器狗接口 {self.robot_port} 或控制台 "
                                   f"{self.console_port} 端口被占用")
            time.sleep(0.2)
        raise TimeoutError("等待 jammers-py 控制台就绪超时")

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    # ---- 控制台 REST ----
    def _request(self, path: str, payload: Optional[dict] = None, base: Optional[str] = None,
                 post: bool = False) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request((base or self.console_url) + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if post or payload is not None else "GET")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _state(self, base: str) -> Optional[dict]:
        try:
            state = self._request("/api/state", base=base)
            return state if "state" in state else None
        except Exception:
            return None

    def _find_existing(self) -> Optional[Tuple[str, dict]]:
        for port in dict.fromkeys((self.console_port, *self.REUSE_PORTS)):
            base = f"http://127.0.0.1:{port}"
            state = self._state(base)
            if state is not None:
                return base, state
        return None

    def _wait_state(self, target: str, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self._request("/api/state").get("state")
            if state == target:
                return
            if state == "finished":
                raise RuntimeError(f"演练局提前结束（期望状态 {target}）")
            time.sleep(0.2)
        raise TimeoutError(f"等待状态 {target} 超时")

    # ---- 一局演练 ----
    def start_episode(self, seed: int) -> List[dict]:
        """按种子生成固定场景并开一局，等接口开放后返回干扰源真值。

        覆盖保证的核对需要"整局完全可复现"：jammers-py 只把 seed 用于干扰源布局，示向度噪声
        种子 `noise_seed_hex` 每次随机 —— 这里把它改写成由 seed 派生的确定值，于是同一个
        --seed 必然得到同一份扫描结果，新旧策略也可严格对照。
        """
        scenario = self._request("/api/scenario", {"problem_no": 3, "seed": seed})["scenario"]
        scenario["noise_seed_hex"] = hashlib.blake2b(f"t3-practice-{seed}".encode(),
                                                    digest_size=8).hexdigest()
        self._request("/api/start", {"problem_no": 3, "scenario": scenario})
        self._wait_state("window_open")
        return scenario["jammers"]

    def finish_episode(self) -> dict:
        """收掉本局并返回模拟器引擎统计（真值清除数、虚拟时刻、检测次数）。"""
        snap = self._request("/api/state")
        engine = snap.get("engine") or {}
        if snap.get("state") != "finished":
            self._request("/api/abort", post=True)
        self._request("/api/clear", post=True)
        return engine


# ----------------------------------------------------------------------------
# 结果落盘
# ----------------------------------------------------------------------------
def save_plan(res: CoverSolveResult, save_dir: Path) -> List[Path]:
    """覆盖圆方案落盘：JSON（含校验与对照）+ CSV（圆心坐标）。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / PLAN_JSON
    json_path.write_text(json.dumps(res.to_json(), ensure_ascii=False, indent=2),
                         encoding="utf-8")
    csv_path = save_dir / PLAN_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "x_m", "y_m", "ring_radius_m", "cover_radius_m"])
        for i, (x, y) in enumerate(res.plan.centers):
            w.writerow([i, "center" if i == 0 else "ring", f"{x:.3f}", f"{y:.3f}",
                        f"{res.plan.ring_radius:.3f}", f"{res.plan.cover_radius:.1f}"])
    return [json_path, csv_path]


def summarize(rows: Sequence[dict]) -> Dict[str, Any]:
    """整批演练的阶段二汇总：清除率、时间、定位误差、清除方式分布。"""
    src = [s for r in rows for s in r.get("truth_check", {}).get("sources", ())]
    methods: Dict[str, int] = {}
    for s in src:
        methods[str(s.get("method"))] = methods.get(str(s.get("method")), 0) + 1
    errs = [s["localize_err_m"] for s in src if s.get("localize_err_m") is not None]
    n_src = sum(r["n_sources"] for r in rows if r["n_sources"])
    return {
        "episodes": len(rows),
        "n_sources": n_src,
        "n_cleared": sum(r.get("cleared", 0) for r in rows),
        "clear_ratio": (sum(r.get("cleared", 0) for r in rows) / n_src) if n_src else None,
        "avg_time_s": round(float(np.mean([r["avg_time_s"] for r in rows
                                           if r.get("avg_time_s")])), 3),
        "virtual_time_s_mean": round(float(np.mean([r["virtual_time_s"] for r in rows])), 3),
        "travel_m_mean": round(float(np.mean([r["travel_m"] for r in rows])), 1),
        "n_measure_mean": round(float(np.mean([r["n_measure"] for r in rows])), 1),
        "n_probe_mean": round(float(np.mean([r["n_probe"] for r in rows])), 2),
        "n_precise_at_survey": sum(r.get("n_precise_at_survey", 0) for r in rows),
        "n_skip_measure": sum(r.get("n_skip_measure", 0) for r in rows),
        "n_inline_cleared": sum(r.get("n_inline_cleared", 0) for r in rows),
        "n_inline_fail": sum(r.get("n_inline_fail", 0) for r in rows),
        "n_refined": sum(r.get("n_refined", 0) for r in rows),
        "localize_err_mean_m": round(float(np.mean(errs)), 3) if errs else None,
        "localize_err_max_m": round(float(np.max(errs)), 3) if errs else None,
        "n_within_clear_radius": sum(1 for e in errs if e <= CLEAR_RADIUS),
        "methods": methods,
    }


def save_survey(save_dir: Path, rows: List[dict], observations: List[dict],
                plan_json: Dict[str, Any]) -> List[Path]:
    """巡视扫描结果落盘：逐局统计 JSON + 逐条观测 CSV。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / SURVEY_JSON
    json_path.write_text(json.dumps({
        "stage": "覆盖圆求解 + 依次到圆心巡视扫描 + 就近试清（未命中按文献准则补测缩小后再清）",
        "cover_plan": plan_json,
        "clear_radius_m": CLEAR_RADIUS,
        "receive_radius_m": [COVER_RADIUS, RECEIVE_MAX],
        "summary": summarize(rows),
        "episodes": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = save_dir / OBS_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "channel", "stage", "x_m", "y_m", "outcome", "svd_deg",
                    "nearest_center_m"])
        for o in observations:
            w.writerow([o["episode"], o["channel"], o["stage"], f"{o['x']:.2f}",
                        f"{o['y']:.2f}", o["outcome"],
                        "" if o["theta"] is None else f"{o['theta']:.2f}",
                        f"{o['nearest_center_m']:.2f}"])
    return [json_path, csv_path]


def episode_row(ep: int, seed: int, truth: Optional[Sequence[dict]], dog: RobotDog,
                stats: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
    """单局汇总行：把引擎统计、真值核对与逐频道档案合成一行（供 JSON / 绘图使用）。"""
    n_src = len(truth) if truth else stats.get("channels_heard")
    cleared = stats.get("cleared", 0)
    return {
        "episode": ep, "seed": seed,
        "n_sources": len(truth) if truth else None,
        "clear_ratio": (cleared / n_src) if n_src else None,
        "localize_err_mean_m": check.get("localize_err_mean_m"),
        "localize_err_max_m": check.get("localize_err_max_m"),
        "n_within_clear_radius": check.get("n_within_clear_radius"),
        "n_cleared_after_probe": check.get("n_cleared_after_probe"),
        **stats,
        "waypoint_stats": dog.waypoint_stats,
        "truth_check": check,
        "bearings_per_channel": {str(c): len(v) for c, v in sorted(dog.obs.items())},
        "cleared_channels": sorted(dog.cleared),
    }


# ----------------------------------------------------------------------------
# 两种运行模式
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 逐局轨迹图：跑完一局后把机器狗的行驶轨迹、覆盖圆与测量结果画成图
# ----------------------------------------------------------------------------
# 中文字体候选：优先本地常见 CJK 字体，最后 DejaVu 兜底（缺字时只影响字形，不影响出图）
TRAJ_FONTS = ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
              "Microsoft YaHei", "SimHei", "DejaVu Sans")
_C_PATH = "#2f6fb5"            # 行驶路径
_C_DIR = "#1f4e79"             # 有示向度的测量点
_C_NOSIG = "#9aa3ad"           # 必然听不到的测量点
_C_NEAR = "#f0a020"            # 近距（可跳过测向）
_C_TRY = "#7b3fa0"             # 清除尝试
_C_HIT = "#2e9e5b"             # 成功清除落点
_C_SRC = "#c0392b"             # 干扰源真值
_C_COVER = "#6fae6f"           # 覆盖圆
_C_FRAME = "#5b6470"           # 圆域边界


def truth_points(truth: Optional[Sequence[dict]]) -> List[Dict[str, float]]:
    """把引擎的源真值统一成 {channel, x, y}，供绘图使用（raw 引擎格式与核对行都能吃）。"""
    out: List[Dict[str, float]] = []
    for j in truth or []:
        if "position" in j:                       # 引擎原始格式：{"position": {"x": .., "y": ..}}
            pos = j["position"]
            x, y = float(pos["x"]), float(pos["y"])
        else:                                     # 核对行格式：{"x": .., "y": ..}
            x, y = float(j["x"]), float(j["y"])
        out.append({"channel": float(j["channel"]), "x": x, "y": y})
    return out


def draw_trajectory(out_path: Path, actions: Sequence[Dict[str, Any]],
                    plan: CoverPlan, order: Sequence[int] = (),
                    sources: Sequence[Dict[str, Any]] = (),
                    title: Optional[str] = None,
                    figsize: Tuple[float, float] = (9.0, 7.6)) -> Path:
    """把一局的轨迹画成图并存盘（格式由后缀决定，.png / .pdf）。

    - `actions`：逐次动作记录（RobotDog.actions），含测向与清除的落点、结果类型、虚拟时刻；
    - `plan`：覆盖圆方案，用来画 7 个半径 1000 m 的覆盖圆与圆心（巡视航路点）；
    - `order`：巡视访问顺序，用于给圆心标序号，直观看出"依次到圆心"的路线；
    - `sources`：干扰源真值（仅演练模式有），画成红叉并按 20 m 清除半径画圈；
    - `title`：图内小标题。论文用图传 None（大标题交给 caption）。

    matplotlib 只在本函数内导入，故未安装时只会抛 ImportError、由调用方忽略 —— 官方测试机上
    没有 matplotlib 也能正常完成整局。出图去掉时间戳类元数据，同一输入两次出图逐字节一致。
    """
    # matplotlib 的默认配置目录常不可写（沙箱/只读 home），会退化成 /tmp 下的随机目录并报警告；
    # 显式指到一个固定可写目录，既消除警告也让两次运行共用同一缓存（出图更快）。
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(),
                                                       "cumcm-mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"font.sans-serif": list(TRAJ_FONTS),
                                "font.family": "sans-serif",
                                "axes.unicode_minus": False, "font.size": 10}):
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots(figsize=figsize)
        th = np.linspace(0.0, 2.0 * math.pi, 361)
        cos_th, sin_th = np.cos(th), np.sin(th)

        # 作业圆域 1800 m 与源生成域 1770 m
        ax.plot(REGION_RADIUS * cos_th, REGION_RADIUS * sin_th,
                color=_C_FRAME, lw=1.5)
        gen_r = REGION_RADIUS - 30.0
        ax.plot(gen_r * cos_th, gen_r * sin_th, color=_C_FRAME, lw=0.8, ls="--", alpha=0.85)

        # 7 个覆盖圆（半径 1000 m）与圆心：逐圆画但不逐个进图例（否则图例会被撑爆）
        wp = np.asarray(plan.waypoints, dtype=float)
        for cx, cy in wp:
            ax.plot(cx + COVER_RADIUS * cos_th, cy + COVER_RADIUS * sin_th,
                    color=_C_COVER, lw=0.7, alpha=0.5, zorder=1.0)
        ax.plot(wp[:, 0], wp[:, 1], marker="o", ms=4.0, ls="none", mfc="none",
                mec=_C_COVER, mew=1.2)
        for k, idx in enumerate(order or range(len(wp))):
            ax.annotate(f"{k + 1}", (wp[idx, 0], wp[idx, 1]), textcoords="offset points",
                        xytext=(5, 4), fontsize=8, color=_C_COVER)

        # 行驶路径与各类动作点
        handles = [Line2D([], [], color=_C_FRAME, lw=1.5,
                          label=f"作业圆域 {REGION_RADIUS:.0f} m"),
                   Line2D([], [], color=_C_FRAME, lw=0.8, ls="--", alpha=0.85,
                          label=f"干扰源生成域 {gen_r:.0f} m"),
                   Line2D([], [], color=_C_COVER, lw=0.7, alpha=0.5,
                          label=f"{len(wp)} 个覆盖圆（半径 {COVER_RADIUS:.0f} m）"),
                   Line2D([], [], marker="o", ms=4.0, ls="none", mfc="none", mec=_C_COVER,
                          mew=1.2, label="覆盖圆圆心（航路点，标注访问序号）")]
        if actions:
            xs = [a["x"] for a in actions]
            ys = [a["y"] for a in actions]
            ax.plot([0.0] + xs, [0.0] + ys, "-", lw=1.0, color=_C_PATH, alpha=0.8,
                    zorder=4.0)
            handles.append(Line2D([], [], color=_C_PATH, lw=1.0,
                                  label=f"行驶路径（{len(actions)} 次动作）"))
            groups = (("direction", dict(marker=".", ms=5, ls="none", color=_C_DIR),
                       "测得示向度"),
                      ("no_signal", dict(marker="x", ms=3.5, ls="none", color=_C_NOSIG),
                       "无信号"),
                      ("near", dict(marker="o", ms=6, ls="none", mfc="none", mec=_C_NEAR,
                                    mew=1.4), "近距 near"))
            for kind, style, label in groups:
                sel = [(a["x"], a["y"]) for a in actions
                       if a["kind"] == "measure" and a["outcome"] == kind]
                if not sel:
                    continue
                arr = np.asarray(sel, dtype=float)
                ax.plot(arr[:, 0], arr[:, 1], zorder=5.0, **style)
                handles.append(Line2D([], [], label=f"{label}（{len(arr)} 次）", **style))
            tries = [(a["x"], a["y"]) for a in actions if a["kind"] == "clear"]
            if tries:
                arr = np.asarray(tries, dtype=float)
                ax.plot(arr[:, 0], arr[:, 1], marker="^", ms=5.5, ls="none", mfc="none",
                        mec=_C_TRY, mew=1.2, zorder=6.0)
                handles.append(Line2D([], [], marker="^", ms=5.5, ls="none", mfc="none",
                                      mec=_C_TRY, mew=1.2,
                                      label=f"清除尝试（{len(arr)} 次）"))
                ok = np.asarray([(a["x"], a["y"]) for a in actions
                                 if a["kind"] == "clear" and a["outcome"] == "success"],
                                dtype=float)
                if len(ok):
                    # 清除落点必然紧贴真值（20 m 内），故画在最上层才看得见
                    ax.plot(ok[:, 0], ok[:, 1], marker="*", ms=11, ls="none", color=_C_HIT,
                            zorder=8.0)
                    handles.append(Line2D([], [], marker="*", ms=11, ls="none",
                                          color=_C_HIT, label=f"清除成功（{len(ok)} 个）"))

        # 干扰源真值（仅演练模式）与 20 m 清除半径
        if sources:
            arr = np.asarray([(s["x"], s["y"]) for s in sources], dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], marker="X", ms=8, ls="none", color=_C_SRC,
                    zorder=7.0)
            # 每个源一个半径 20 m 的圆：列方向必须是"每个圆一列"，否则会被连成一团
            ax.plot(arr[:, 0][None, :] + CLEAR_RADIUS * cos_th[:, None],
                    arr[:, 1][None, :] + CLEAR_RADIUS * sin_th[:, None],
                    color=_C_SRC, lw=0.7, alpha=0.75, zorder=1.5)
            handles.append(Line2D([], [], marker="X", ms=8, ls="none", color=_C_SRC,
                                  label=f"干扰源真值（{len(arr)} 个）"))
            handles.append(Line2D([], [], color=_C_SRC, lw=0.7, alpha=0.75,
                                  label=f"清除半径 {CLEAR_RADIUS:.0f} m（全图视场 3600 m，需放大才可见）"))

        ax.plot(0.0, 0.0, marker="s", ms=6, color="black")
        handles.append(Line2D([], [], marker="s", ms=6, ls="none", color="black",
                              label="起点（原点）"))
        ax.set_aspect("equal")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        if title:
            ax.set_title(title, fontsize=10)
        # 图例放到坐标轴下方：图内 3600 m 见方的圆域几乎没有空白，放进去必然压住轨迹
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07),
                  ncol=3, fontsize=7.5, framealpha=0.9)
        ax.grid(alpha=0.25, lw=0.5)
        fig.tight_layout()
        fig.savefig(out_path, dpi=TRAJ_DPI, metadata={"Software": None})
        plt.close(fig)
    return out_path


def save_trajectory(save_dir: Path, name: str, actions: Sequence[Dict[str, Any]],
                    plan: CoverPlan, order: Sequence[int] = (),
                    sources: Sequence[Dict[str, Any]] = (),
                    title: Optional[str] = None,
                    traj_dir: str = TRAJ_DIR) -> List[Path]:
    """落盘一局的轨迹：同名 PNG（图）与 CSV（轨迹表），返回已写出的文件列表。

    轨迹表让"图上每个点"都能与过程日志逐点对账（序号、动作类型、阶段、坐标、结果、频道、
    虚拟时刻、累计里程）。matplotlib 缺失只提示一次并跳过出图，轨迹表照常写出。
    """
    out_dir = save_dir / traj_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "kind", "stage", "x_m", "y_m", "channel", "outcome",
                    "theta_deg", "virtual_time_s", "travel_m"])
        for a in actions:
            w.writerow([a["seq"], a["kind"], a["stage"], f"{a['x']:.2f}", f"{a['y']:.2f}",
                        a["channel"], a["outcome"] or "",
                        "" if a["theta"] is None else f"{a['theta']:.2f}",
                        f"{a['virtual_time_s']:.3f}", f"{a['travel_m']:.2f}"])
    paths = [csv_path]
    try:
        paths.insert(0, draw_trajectory(out_dir / f"{name}.png", actions, plan, order,
                                        sources, title))
    except ImportError:
        global _PLOT_HINTED
        if not _PLOT_HINTED:
            _PLOT_HINTED = True
            print("提示：未安装 matplotlib，已跳过轨迹图（pip install matplotlib 后可自动生成）")
    return paths


_PLOT_HINTED = False           # 缺 matplotlib 的提示只打印一次，避免每局刷屏


def _observations(ep: int, plan: CoverPlan, meas: Dict[int, List[Meas]]) -> List[dict]:
    """把本局全部测量整理成 CSV 行（含 no_signal，并附"测量点到最近圆心的距离"）。"""
    rows = []
    for ch, ml in sorted(meas.items()):
        for m in ml:
            d = float(np.linalg.norm(plan.waypoints - np.array([m.x, m.y]), axis=1).min())
            rows.append({"episode": ep, "channel": ch, "x": m.x, "y": m.y,
                         "outcome": m.outcome, "theta": m.theta, "stage": m.stage,
                         "nearest_center_m": d})
    return rows


def _episode_printer(clear: bool):
    """按模式打印本局小结（只在需要时输出清除相关字段）。"""
    def show(stats: dict, check: dict, n_sources: Optional[int]) -> None:
        print(f"本局：巡视 {stats['waypoints_visited']} 个圆心，里程 {stats['travel_m']:.0f} m，"
              f"虚拟时间 {stats['virtual_time_s']:.0f} s，测向 {stats['n_measure']} 次；"
              f"听到 {stats['channels_heard']}/{n_sources} 个源"
              f"（{stats['n_bearings']} 条示向度）")
        if clear:
            print(f"  阶段2：就近试清命中 {stats['cleared']} 个（其中补测后再清 "
                  f"{stats['n_refined']} 个，共补测 {stats['n_probe']} 次）；"
                  f"巡视后估计已够准的（覆盖圆半径 < {CLEAR_RADIUS:.0f} m，诊断）"
                  f"{stats['n_precise_at_survey']} 个；判定必无信号跳过测量 "
                  f"{stats['n_skip_measure']} 次；途中顺路清除 {stats['n_inline_cleared']} 个"
                  f"（白跑 {stats['n_inline_fail']} 次）")
            print(f"  清除：{stats['cleared']}/{n_sources}（平均 {stats['avg_time_s']:.1f} s/个），"
                  f"定位误差均值 {check['localize_err_mean_m']} m / 最大 "
                  f"{check['localize_err_max_m']} m，清除点在 20 m 内 "
                  f"{check['n_within_clear_radius']} 个")
        print(f"  覆盖核对：源到最近圆心最坏距离 {check['worst_nearest_m']:.1f} m ≤ "
              f"{COVER_RADIUS:.0f} m（{check['all_within_cover']}），"
              f"漏听 {check['missed_channels'] or '无'}")
    return show


def run_practice(args: argparse.Namespace, res: CoverSolveResult, save_dir: Path) -> int:
    """本地演练：自动拉起 jammers-py，跑 N 局（巡视扫描 + 定位 + 清除）并汇总。"""
    jammers_dir = (Path(args.jammers_dir) if args.jammers_dir
                   else Path(__file__).parent / "jammers-py")
    clear = not args.survey_only
    show = _episode_printer(clear)
    rows: List[dict] = []
    observations: List[dict] = []
    with PracticeArena(jammers_dir, robot_id=args.robot_id,
                       robot_port=args.robot_port, console_port=args.console_port,
                       reuse_existing=not args.no_reuse) as arena:
        print(f"jammers-py 已就绪：机器狗接口 {arena.robot_url}，控制台 {arena.console_url}")
        for ep in range(args.practice):
            seed = args.seed + ep
            truth = arena.start_episode(seed)
            print(f"\n----- 演练第 {ep + 1}/{args.practice} 局（seed={seed}，"
                  f"干扰源 {len(truth)} 个）-----")
            dog = RobotDog(sim_api.Simulator(robot_id=args.robot_id, base_url=arena.robot_url,
                                             timeout=args.timeout),
                           verbose=not args.quiet, logfile=args.log, episode=ep + 1,
                           clear=clear, k_clear_max=args.k_clear_max,
                           inline_try_radius=args.inline_try_radius,
                           inline_detour=args.inline_detour)
            stats = dog.run(res.plan, res.survey_order)
            arena.finish_episode()
            check = truth_check(truth, res.plan, dog.obs, dog.cleared, dog.tracks)
            rows.append(episode_row(ep + 1, seed, truth, dog, stats, check))
            observations.extend(_observations(ep + 1, res.plan, dog.meas))
            show(stats, check, len(truth))
            if not args.no_plot:                    # 出图在 /exit 之后，不占现实时间预算
                name = f"ep{ep + 1:02d}_seed{seed}"
                files = save_trajectory(
                    save_dir, name, dog.actions, res.plan, res.survey_order,
                    truth_points(truth),
                    title=f"第 {ep + 1} 局（seed={seed}）：清除 {stats['cleared']}/{len(truth)}、"
                          f"里程 {stats['travel_m']:.0f} m、虚拟时间 {stats['virtual_time_s']:.0f} s",
                    traj_dir=args.traj_dir)
                print("  轨迹图：" + "，".join(str(f) for f in files))
    print("\n" + "=" * 78)
    if clear:
        print(f"汇总（{len(rows)} 局）：平均清除比例 "
              f"{np.mean([r['clear_ratio'] for r in rows]):.4f}"
              f"（{sum(r['cleared'] for r in rows)}/{sum(r['n_sources'] for r in rows)}），"
              f"平均 {np.mean([r['avg_time_s'] for r in rows if r['avg_time_s']]):.1f} s/个，"
              f"平均虚拟时间 {np.mean([r['virtual_time_s'] for r in rows]):.0f} s，"
              f"平均测向 {np.mean([r['n_measure'] for r in rows]):.0f} 次"
              f"（其中补测 {np.mean([r['n_probe'] for r in rows]):.0f} 次；"
              f"判定必无信号而跳过 {np.mean([r['n_skip_measure'] for r in rows]):.0f} 次）")
        print(f"  定位误差：均值 "
              f"{np.mean([r['localize_err_mean_m'] for r in rows if r['localize_err_mean_m']]):.2f}"
              f" m，最差单源 "
              f"{max([r['localize_err_max_m'] for r in rows if r['localize_err_max_m']] or [0]):.2f}"
              f" m；巡视后估计已够准的源（诊断）"
              f"{sum(r['n_precise_at_survey'] for r in rows)}/{sum(r['n_sources'] for r in rows)} 个")
        print("逐局：" + "  ".join(f"seed{r['seed']}={r['cleared']}/{r['n_sources']}"
                                  for r in rows))
    else:
        print(f"汇总（{len(rows)} 局，仅巡视扫描）：平均里程 "
              f"{np.mean([r['travel_m'] for r in rows]):.0f} m，平均虚拟时间 "
              f"{np.mean([r['virtual_time_s'] for r in rows]):.0f} s，平均测向 "
              f"{np.mean([r['n_measure'] for r in rows]):.0f} 次，"
              f"共听到 {sum(r['channels_heard'] for r in rows)}/"
              f"{sum(r['n_sources'] for r in rows)} 个源")
        print("逐局：" + "  ".join(f"seed{r['seed']}={r['channels_heard']}/{r['n_sources']}"
                                  for r in rows))
    print("=" * 78)
    paths = (save_plan(res, save_dir)
             + save_survey(save_dir, rows, observations, res.to_json()))
    print("结果已保存：" + "，".join(str(p) for p in paths))
    return 0


def run_official(args: argparse.Namespace, res: CoverSolveResult, save_dir: Path) -> int:
    """官方评测接口模式：连 127.0.0.1 上已开放接口的模拟器，跑完整一局。

    与演练的唯一差别是**拿不到干扰源真值**，因此定位误差等需要真值的指标留空。
    """
    sim = sim_api.Simulator(robot_id=args.robot_id, base_url=args.base_url, timeout=args.timeout)
    print(f"连接模拟器 {args.base_url}（robot_id={args.robot_id}）")
    dog = RobotDog(sim, verbose=not args.quiet, logfile=args.log, episode=1,
                   clear=not args.survey_only, k_clear_max=args.k_clear_max,
                   inline_try_radius=args.inline_try_radius,
                   inline_detour=args.inline_detour)
    stats = dog.run(res.plan, res.survey_order)
    print(f"完成：清除 {stats['cleared']} 个，巡视 {stats['waypoints_visited']} 个圆心，"
          f"里程 {stats['travel_m']:.0f} m，虚拟时间 {stats['virtual_time_s']:.0f} s，"
          f"测向 {stats['n_measure']} 次（补测 {stats['n_probe']} 次），"
          f"听到 {stats['channels_heard']} 个频道（{stats['n_bearings']} 条示向度）")
    row = episode_row(1, args.seed, None, dog, stats,
                      truth_check(None, res.plan, dog.obs, dog.cleared, dog.tracks))
    if not args.no_plot:
        files = save_trajectory(
            save_dir, f"ep01_seed{args.seed}", dog.actions, res.plan, res.survey_order, (),
            title=f"官方模式：清除 {stats['cleared']} 个、里程 {stats['travel_m']:.0f} m、"
                  f"虚拟时间 {stats['virtual_time_s']:.0f} s（无真值可比）",
            traj_dir=args.traj_dir)
        print("轨迹图：" + "，".join(str(f) for f in files))
    paths = (save_plan(res, save_dir)
             + save_survey(save_dir, [row], _observations(1, res.plan, dog.meas), res.to_json()))
    print("结果已保存：" + "，".join(str(p) for p in paths))
    return 0


# ----------------------------------------------------------------------------
# 命令行
# ----------------------------------------------------------------------------
def _relax_console_encoding() -> None:
    """放宽控制台编码错误处理，避免个别字符（如 √、≤）在 GBK 控制台上中断运行。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2026 CUMCM B 题问题三（第一阶段）：1000 m 覆盖圆求解 + 依次到圆心巡视扫描")
    p.add_argument("--practice", type=int, nargs="?", const=1, default=0,
                   help="本地演练局数：自动拉起 jammers-py 跑 N 局（缺省 1 局）")
    p.add_argument("--base-url", default=None,
                   help="官方模拟器地址（赛期用，如 http://127.0.0.1:2026）；缺省只求解覆盖圆")
    p.add_argument("--robot-id", default=sim_api.ROBOT_ID, help="参赛队号（须与模拟器一致）")
    p.add_argument("--timeout", type=float, default=5.0, help="HTTP 超时 / s")
    p.add_argument("--ring-radius", type=float, default=None,
                   help=f"覆盖圆环半径 d / m（缺省 {CHOSEN_RING_RADIUS:.0f}，时间优先、余量 31 m；"
                        f"传 {optimal_ring_radius():.3f} 可取余量最大的 d*，里程增加约 2153 m）")
    p.add_argument("--jammers-dir", default=None,
                   help="jammers-py 目录（缺省为本脚本旁的 jammers-py/）")
    p.add_argument("--console-port", type=int, default=8090,
                   help="演练时 jammers-py 控制台端口（缺省 8090，被占用则自动顺延）")
    p.add_argument("--robot-port", type=int, default=2026,
                   help="演练时 jammers-py 的机器狗接口端口（缺省 2026，与官方一致）")
    p.add_argument("--no-reuse", action="store_true",
                   help="不使用已在运行的 jammers-py，另起一个独占实例（多个会话并行演练时用；"
                        "配合 --console-port/--robot-port 避免端口冲突）")
    p.add_argument("--seed", type=int, default=SEED,
                   help="演练第 1 局的种子（场景布局与示向度噪声都由它确定，可复现）")
    p.add_argument("--save-dir", default=RESULTS_DIR, help="结果输出目录（缺省 results/）")
    p.add_argument("--log", default=None, help="过程日志文件（逐站扫描的文字过程，追加写入）")
    p.add_argument("--k-clear-max", type=int, default=K_CLEAR_MAX,
                   help=f"试清未中后最多再补清几个点（用 K 个半径 20 m 的圆覆盖定位区域；"
                        f"缺省 {K_CLEAR_MAX}，只在能盖满区域时才用，盖不满则转入补测）")
    p.add_argument("--inline-try-radius", type=float, default=INLINE_TRY_RADIUS,
                   help=f"巡视途中顺路试清允许的覆盖圆半径上限 / m（缺省 {INLINE_TRY_RADIUS:.0f}；"
                        f"设成 {CLEAR_RADIUS:.0f} 表示只清估计已够准的）")
    p.add_argument("--inline-detour", type=float, default=INLINE_DETOUR,
                   help=f"巡视途中顺路试清允许的绕行里程上限 / m（缺省 {INLINE_DETOUR:.0f}）")
    p.add_argument("--survey-only", action="store_true",
                   help="只做阶段一（巡视扫描 + 覆盖核对），不做定位与清除")
    p.add_argument("--traj-dir", default=TRAJ_DIR,
                   help=f"轨迹图输出子目录（相对 --save-dir；缺省 {TRAJ_DIR}，"
                        f"与 T3_ga.py 的 trajectory/ 分开以免互相覆盖）")
    p.add_argument("--no-plot", action="store_true",
                   help=f"不出逐局轨迹图（缺省每局在 <save-dir>/{TRAJ_DIR}/ 生成同名 png + csv）")
    p.add_argument("--quiet", action="store_true", help="只输出汇总，不打印过程")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    _relax_console_encoding()
    args = build_parser().parse_args(argv)
    save_dir = Path(args.save_dir)

    res = solve_covering_circles(args.ring_radius)   # 第一步：求 1000 m 覆盖圆的位置
    if not args.quiet:
        print_cover_report(res)
    paths = save_plan(res, save_dir)
    print("覆盖圆方案已保存：" + "，".join(str(p) for p in paths))

    try:
        if args.practice:                   # 第二步：依次移动到圆心进行扫描
            return run_practice(args, res, save_dir)
        if args.base_url:
            return run_official(args, res, save_dir)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except OSError as exc:
        print(f"连接模拟器失败：{exc}", file=sys.stderr)
        print("请确认模拟器已启动并处于测试窗口内（默认地址 http://127.0.0.1:2026）。",
              file=sys.stderr)
        return 1
    print("提示：加 --practice N 跑本地演练（自动拉起 jammers-py），"
          "或加 --base-url 连官方模拟器执行巡视扫描。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
