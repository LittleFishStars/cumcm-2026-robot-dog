"""GA 对照方案的可调参数（原 T3_ga.py 顶部常量区）。

与确定性方案（cumcm.t3）**各自独立**：两者的覆盖路点设计、补测判据、末端逼近步长等取自不同
的调参过程，数值也已分别写进报告，因此不强行统一 —— 共用的只有与题目本身有关的物理量
（圆域半径、接收半径上下界、示向度误差、频道数、坐标上限），它们在两边数值相同。

覆盖路点设计：三个参数共同决定"圆域内任意点到最近路点 ≤ COVER_RADIUS"的保证强度。
离散化越细，实际能达到的最坏距离越接近 COVER_RADIUS。实算（连续圆域上求最坏点）：
    100/120/950 → 8 路点，最坏 1008.4 m（> 1000，圆域边缘存在听不到的源 ×）
     60/ 80/920 → 8 路点，最坏  962.4 m（余量 +37.6 m，巡视里程还短 480 m √ 现用）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# ----------------------------------------------------------------------------
# 常量（与附件 1 / 附件 2 一致）
# ----------------------------------------------------------------------------
REGION_RADIUS = 1800.0          # 目标圆域半径 / m
MAX_RECEPTION = 1500.0          # 有效接收半径上界 / m（下界 1000 m，见 COVER_RADIUS）
BEARING_ERROR_DEG = 1.0         # 示向度误差半宽 / 度
CHANNELS: Tuple[int, ...] = tuple(range(1, 21))
COORD_LIMIT = 2.0e6             # 坐标分量绝对值上限 / m

# 覆盖路点设计：三个参数共同决定"圆域内任意点到最近路点 ≤ COVER_RADIUS"的保证强度。
# 离散化越细，实际能达到的最坏距离越接近 COVER_RADIUS。实算（连续圆域上求最坏点）：
#   100/120/950 → 8 路点，最坏 1008.4 m（>1000，圆域边缘存在听不到的源 ×）
#    60/ 80/920 → 8 路点，最坏  962.4 m（余量 +37.6 m，巡视里程还短 480 m √ 现用）
COVER_RADIUS = 920.0            # 覆盖路点设计半径（<1000 接收下界）/ m
COVER_GRID = 60.0               # 目标圆域离散网格 / m
COVER_STEP = 80.0               # 候选路点网格 / m
REGION_MARGIN = 1.0             # 坐标裁剪时保留的数值余量 / m（机器狗须在圆域内）
OBS_PER_SOURCE = 3              # 每个频道尽量采集的示向度条数

# 位置 1σ 超过该值即视为交会几何病态并补测垂直视角。校准依据（200 组随机交会实验）：
# σ > 40 m 的频道约占 10%，其真实定位误差中位数 28 m（已超过 20 m 清除半径），而
# σ < 40 m 的频道误差中位数仅 4~6 m —— 阈值正好把"会失败的交会"挑出来。
GEOM_SIGMA = 40.0               # 位置 1σ 超过该值视为交会几何病态 / m
GEOM_SIN_MIN = 0.20             # 两射线夹角 |sin| 小于该值视为近共线
PERP_STEPS = (250.0, 500.0, 800.0)      # 补测垂直视角时外移的距离 / m
# 说明：补测点按"沿某方向外移固定距离"生成，源靠近圆域边缘时可能挪到圆域之外（机器狗
# 不允许离开作业圆域，虽然模拟器不会拒绝）。所有动作坐标统一由 clamp_to_region 拉回域内，
# 因此无需在各处候选点生成逻辑里重复裁剪，代价只是基线略短、σ 略大。
SINGLE_PROBES = ((600.0, 35.0), (400.0, 45.0), (800.0, 25.0))   # 单射线补测：前移距离/侧偏角
HOMING_STEP = 16.0              # 末端沿示向度逼近的步长 / m
HOMING_MAX = 24                 # 末端逼近最大迭代次数
SAFETY_MARGIN = 30.0            # 现实时限预留余量 / s

SEED = 2026

RESULTS_DIR = "results/t3_ga"    # 训练结果输出目录（可用 --save-dir 改）
API_LOG_NAME = "api_calls.jsonl"    # 接口调用日志文件名（落在 --save-dir 下）
CLEAR_RADIUS = 20.0             # 清除半径 / m（题目给定；GA 版原先在绘图中写作字面量 20.0）


@dataclass(frozen=True)
class GAParams:
    """GA 超参数：只在这里定义一次，GA 实例与训练记录的 meta 共用，避免两处不同步。"""

    pop: int
    gens: int
    pc: float
    pm: float


GA_LOC = GAParams(pop=80, gens=150, pc=0.90, pm=0.35)       # 定位 GA（实数编码）
GA_ROUTE = GAParams(pop=100, gens=300, pc=0.90, pm=0.35)    # 路线 GA（排列编码，开路径 TSP）
GA_LOC_DOMAIN = 2200.0          # 定位 GA 个体取值半径 / m
GA_RECORD_STRIDE = 5            # 训练记录抽稀步长：每多少代记一个收敛点
CONVERGED_FITNESS = 1e-4        # 定位 GA 提前收敛的适应度阈值


@dataclass(frozen=True)
class GAParams:
    """GA 超参数：只在这里定义一次，GA 实例与训练记录的 meta 共用，避免两处不同步。"""

    pop: int
    gens: int
    pc: float
    pm: float
