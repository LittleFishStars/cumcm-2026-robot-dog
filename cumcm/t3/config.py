"""问题三的全部可调参数与物理常量（原 T3.py 顶部常量区）。

集中在此便于"一处改、全局生效"：覆盖圆半径、覆盖判据、清除流程阈值、补测选点准则、
结果文件名等。凡是经过实验选定或校准的常量，都在原注释里保留了依据（例如
`CHOSEN_RING_RADIUS` 的权衡、`GRID_STEP` 的精度、`K_CLEAR_MAX` 的实测取舍），
改之前请先读注释。
"""

from __future__ import annotations

import math

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
# 精确定位判据用"最小覆盖圆半径 < CLEAR_RADIUS"而不是"区域直径 < 2*CLEAR_RADIUS"：区域被
# 圆域与圆盘约束裁剪后不再是凸集，直径本身无法再衡量"能不能一步清除"。二者只在凸区域下等价，
# 故不设直径常量，相关论证见 cumcm/t3/regions.py 与 RobotDog._precise。
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
