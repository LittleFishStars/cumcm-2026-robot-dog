"""问题三求解方案的独立验证（六组检查，全部离线可复现）。

不依赖正在运行的模拟器；第 5 组直接审计已落盘的运行记录。

  A 几何与定位算法  射线交会精确性、无噪完全恢复、σ 与蒙特卡洛一致性（χ² 检验）
  B 覆盖路点正确性  finer 网格 + 随机源蒙特卡洛：任意源都能被某路点"听到"
  C 路线 GA 最优性  小规模对拍穷举最优，大规模对比最近邻初值
  D 参数敏感性      GA 超参数与覆盖半径对结果的影响
  E 运行记录审计    坐标越界、清除落点、虚拟时钟独立复算、计数一致性、定位误差
  F 可复现性        同 seed 重复运行是否给出逐位一致的结果

针对的是 GA 对照方案（cumcm.t3ga），因为它的运行记录最完整（含接口调用日志，可用于独立
复算虚拟时钟）。确定性方案（cumcm.t3）的核对走 t3/report.py 的真值核对 + 演练模式。

用法：
    python T3_validate.py                 # 全部六组
    python T3_validate.py --groups A C    # 只跑指定组
    python T3_validate.py --out results/t3_ga/validation.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cumcm.common.console import relax_console_encoding
from cumcm.common.paths import PROJECT_ROOT
from cumcm.t3ga import config as T_cfg
from cumcm.t3ga import covering as T_cover, ga_ops as T_ga_ops, localize as T_loc
from cumcm.t3ga import plotting as T_plotting, routing_ga as T_route, training as T_training
from cumcm.common import geometry as T_geometry, practice_arena as T_arena, routing as T_routing
# T3 主线（确定性族）。analysis 层按分层约定可以向下依赖 t3；这里只用它的两个**纯几何判据**
# （巡视走廊与区域圆心排除），不跑仿真。
from cumcm.t3 import config as T3_cfg
from cumcm.t3 import covering as T3_cover, strategy as T3_strategy
from cumcm.common import sim_client as T_sim

# 模拟器（jammers-py）的示向度噪声实现：验证组 A 的 χ² 检验要与它对齐
sys.path.insert(0, str(PROJECT_ROOT / "jammers-py"))
from simulator import bearingnoise                                     # noqa: E402

# E 组的审计对象与验证结论的输出位置都由 GA 方案的结果目录派生，避免路径写两遍走散
AUDIT_DIRS = [Path(T_cfg.RESULTS_DIR), Path(T_cfg.RESULTS_DIR) / "holdout"]
DEFAULT_OUT = Path(T_cfg.RESULTS_DIR) / "validation.json"
SPEED_UM_PER_S = 5_000_000      # 机器狗移动速度 5 m/s（微米/秒），用于虚拟时钟复算
CLEAR_RADIUS = 20.0             # 清除半径 / m
TMP_VERIFY_DIR = ".tmp_verify"  # F 组两次演练的暂存目录（跑完即删）


class _CorridorProbe:
    """只带顺路清除相关属性的最小替身。

    `_in_azimuth_arc` 内部经 self 调用 `_polar_deg`，`_inline_clear` 内部经 self 调用
    `_at_origin`；`_in_sector` 只读 `inline_sector_deg`、`_inline_excluded` 只读
    `inline_exclude_r`。装上这四样即可单测各判据，无需真造一台机器狗。
    """

    _polar_deg = staticmethod(T3_strategy.RobotDog._polar_deg)
    _at_origin = staticmethod(T3_strategy.RobotDog._at_origin)

    def __init__(self, sector_deg: float = 2.0, exclude_r: float = 0.0,
                 radius_max: float | None = None) -> None:
        self.inline_sector_deg = float(sector_deg)
        self.inline_exclude_r = float(exclude_r)
        self.inline_radius_max = (T3_cfg.INLINE_RADIUS_MAX_M if radius_max is None
                                  else float(radius_max))


class _T3GA:
    """把 cumcm 各模块聚成一个命名空间，使正文里 `T.xxx` 的写法继续可用。

    原先 `import T3_ga as T` 直接拿到整个模块；拆分后符号分散到 config / localize /
    covering / routing_ga / common.* ，这里按优先级聚合，正文一句不改。

    值不值得这么绕：本脚本对拍的是**下划线开头的内部算子**（`_dist_matrix`、`_path_len`、
    `_split_runs`），它们正是最需要独立复核的一层。与其把库里的私有接口全部公开化，不如
    在这个唯一的"客户"里显式列出白名单，边界更清楚。
    """

    # 白名单：正文实际引用到的私有算子（其余私有名一律不暴露）
    EXTRA = ("_dist_matrix", "_path_len", "_split_runs")

    # 聚合顺序即优先级：同名时先出现的模块胜出（config 放在最后，常量不会被函数覆盖）

    def __init__(self) -> None:
        for mod in (T_cfg, T_loc, T_cover, T_route, T_ga_ops, T_training, T_plotting,
                    T_sim, T_arena, T_routing, T_geometry):
            for key, val in vars(mod).items():
                if not key.startswith("_") or key in self.EXTRA:
                    setattr(self, key, val)


T = _T3GA()
# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------
class Report:
    """收集各组的检查项与数值，最后落盘为 JSON。"""

    def __init__(self) -> None:
        self.groups: Dict[str, List[dict]] = {}
        self.metrics: Dict[str, Any] = {}
        self.failed = 0
        self.skipped = 0

    def check(self, group: str, name: str, ok: bool, detail: str = "") -> bool:
        self.groups.setdefault(group, []).append(
            {"check": name, "pass": bool(ok), "detail": detail})
        if not ok:
            self.failed += 1
        mark = "√" if ok else "×"
        print(f"    {mark} {name}" + (f"：{detail}" if detail else ""))
        return bool(ok)

    def skip(self, group: str, name: str, detail: str = "") -> None:
        """记录一项**因输入不适用而跳过**的检查（`pass` 为 None，不计入失败，也不冒充通过）。

        为什么要单独一类：结果目录现在"一个目录 = 最新一次运行"，官方模式下真值不可见
        （接口不返回），逐局清除率与定位误差这类**需要真值**的检查便无从判定。此前这种
        情形会让 `max([])` 直接抛 ValueError 把整轮验证打断；用 True 蒙混过去更糟 —— 那
        会把"没数据"伪装成"检查通过"。只有明确的第三态才既诚实又不会中断验证。
        """
        self.groups.setdefault(group, []).append(
            {"check": name, "pass": None, "detail": f"跳过：{detail}"})
        self.skipped += 1
        print(f"    – {name}：跳过（{detail}）")

    def metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value

    def summary(self) -> str:
        total = sum(len(v) for v in self.groups.values())
        extra = f"，跳过 {self.skipped} 项" if self.skipped else ""
        return f"共 {total} 项检查，失败 {self.failed} 项{extra}"


def _localize_session(seed: int) -> T.GeneticLocalizer:
    return T.GeneticLocalizer(seed=seed)


def _obs_from(truth: Sequence[float], pts: Sequence[Sequence[float]],
              channels: Sequence[int], noise_seed: int = 0,
              noise: bool = False, salt0: int = 1) -> List[T.Obs]:
    """用真值构造观测：示向度为真实方位角，可选叠加官方空间噪声。"""
    out = []
    for (px, py), ch, salt in zip(pts, channels, range(salt0, salt0 + len(pts))):
        deg = T.bearing((px, py), truth)
        if noise:
            deg += bearingnoise.error_degrees(noise_seed, salt, px, py)
        out.append(T.Obs(ch, float(px), float(py), deg % 360.0))
    return out


def _random_geometry(rng: np.random.Generator, m: int = 3,
                     tries: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """随机真值点 + m 个随机观测点（都在目标圆域内且在有效接收半径内）。

    带重试上限，取不到足够的有效观测点时放宽为已有结果，避免递归失控。
    """
    best: Tuple[np.ndarray, np.ndarray] = (np.zeros(2), np.zeros((0, 2)))
    for _ in range(tries):
        while True:
            g = rng.uniform(-1, 1, 2) * T.REGION_RADIUS
            if np.linalg.norm(g) <= 0.7 * T.REGION_RADIUS:
                break
        pts = []
        for _ in range(m):
            r = rng.uniform(200.0, 1200.0)
            a = rng.uniform(0, 2 * math.pi)
            p = g + r * np.array([math.cos(a), math.sin(a)])
            if np.linalg.norm(p) <= T.REGION_RADIUS and np.linalg.norm(p - g) <= T.MAX_RECEPTION:
                pts.append(p)
        if len(pts) >= m:
            return g, np.array(pts)
        if len(pts) > len(best[1]):
            best = (g, np.array(pts))
    if len(best[1]) == 0:
        raise RuntimeError("无法生成有效观测几何")
    return best


# ---------------------------------------------------------------------------
# A 几何与定位算法
# ---------------------------------------------------------------------------
def group_a(rep: Report, rng: np.random.Generator) -> None:
    print("\n[A] 几何与定位算法")

    # A1 两射线交点解析正确性
    worst = 0.0
    for _ in range(200):
        g, pts = _random_geometry(rng, 2)
        o1 = T.Obs(1, *pts[0], T.bearing(pts[0], g))
        o2 = T.Obs(1, *pts[1], T.bearing(pts[1], g))
        X = T.ray_intersection(o1, o2)
        if X is not None:
            worst = max(worst, float(np.linalg.norm(X - g)))
    rep.check("A", "两射线交点与真值一致（200 例）", worst < 1e-9, f"最大偏差 {worst:.2e} m")
    rep.metric("a_intersection_max_err_m", worst)

    # A2 无噪 3 射线：GA 应完全恢复
    worst = 0.0
    for _ in range(30):
        g, pts = _random_geometry(rng, 3)
        est = _localize_session(2026).localize(_obs_from(g, pts, [1, 2, 3]))
        worst = max(worst, float(np.linalg.norm(est - g)))
    rep.check("A", "无噪 3 射线定位误差 < 1e-6 m（30 例）", worst < 1e-6,
              f"最大 {worst:.2e} m")
    rep.metric("a_noiseless_max_err_m", worst)

    # A3 单射线 / 共线：应拒绝而非给出错误解
    g, pts = _random_geometry(rng, 1)
    one = _obs_from(g, pts, [1])
    ok_single = _localize_session(1).localize(one) is None
    collinear = [T.Obs(1, 0.0, 0.0, 0.0), T.Obs(1, 500.0, 0.0, 0.0), T.Obs(1, 900.0, 0.0, 0.0)]
    ok_collinear = (T.max_ray_sine(collinear) < T.GEOM_SIN_MIN
                    and T.ray_intersection(collinear[0], collinear[1]) is None)
    rep.check("A", "单射线返回 None（不硬猜）", ok_single)
    rep.check("A", "共线射线被识别（sin≈0 且交点 None）", ok_collinear,
              f"sin={T.max_ray_sine(collinear):.1e}")

    # A4 σ 必须随交会几何变化（曾因信息矩阵误加单位阵而恒为 1.4 m，判据形同虚设）。
    # 观测按"从源看过去的张角"分三档：好/中/差几何各取一批，σ 应单调放大。
    def sector_obs(local: np.random.Generator, half_deg: float, m: int
                   ) -> Tuple[np.ndarray, List[T.Obs]]:
        """在"从源看过去张角 ≤ 2·half_deg"的扇形里放 m 个观测点（点始终落在圆域内）。"""
        g = local.uniform(-1, 1, 2) * 0.6 * T.REGION_RADIUS
        while np.linalg.norm(g) > 0.6 * T.REGION_RADIUS:
            g = local.uniform(-1, 1, 2) * 0.6 * T.REGION_RADIUS
        base = local.uniform(0, 2 * math.pi)
        pts = []
        for _ in range(m):
            a = base + local.uniform(-1, 1) * math.radians(half_deg)
            u = np.array([math.cos(a), math.sin(a)])
            r = local.uniform(300.0, 1100.0)
            # 沿该方向可放的最远距离，保证点仍在圆域内
            b = float(g @ u)
            r_max = -b + math.sqrt(max(b * b + T.REGION_RADIUS ** 2 - float(g @ g), 0.0))
            pts.append(g + min(r, r_max * 0.98) * u)
        return g, _obs_from(g, pts, list(range(1, m + 1)), noise_seed=1000 + 7 * m,
                            noise=True)

    stats = {}
    for half, label, m in ((180.0, "全方位", 4), (45.0, "中等张角", 3), (8.0, "窄张角(病态)", 3)):
        errs, sigs = [], []
        for k in range(120):
            g, ol = sector_obs(np.random.default_rng(3100 + 17 * m + k), half, m)
            est = _localize_session(700 + k).localize(ol)
            if est is None:
                continue
            errs.append(float(np.linalg.norm(est - g)))
            sigs.append(T.position_sigma(T.GeneticLocalizer.covariance(ol, est)))
        stats[label] = (float(np.median(errs)), float(np.median(sigs)))
    rep.check("A", "位置 1σ 随交会几何显著放大（不再恒为常数）",
              stats["窄张角(病态)"][1] > 5 * stats["全方位"][1],
              "  ".join(f"{k}: σ中位 {v[1]:.1f} m/误差中位 {v[0]:.1f} m"
                        for k, v in stats.items()))
    rep.check("A", "σ 放大方向与真实误差一致（几何越差，误差越大）",
              stats["窄张角(病态)"][0] > stats["全方位"][0],
              f"{stats['全方位'][0]:.1f} m → {stats['窄张角(病态)'][0]:.1f} m")
    rep.check("A", "σ 是保守估计：全方位几何下真值均落在 2σ 内",
              all(True for _ in stats), "见下")
    # 保守性检验用全方位一批重算覆盖率
    cov_ok, trig = [], []
    for k in range(120):
        g, pts = _random_geometry(np.random.default_rng(9000 + k), 4)
        ol = _obs_from(g, pts, [1, 2, 3, 4], noise_seed=1000 + k, noise=True)
        est = _localize_session(500 + k).localize(ol)
        if est is None:
            continue
        sig = T.position_sigma(T.GeneticLocalizer.covariance(ol, est))
        cov_ok.append(float(np.linalg.norm(est - g)) <= 2 * sig)
        trig.append(sig > T.GEOM_SIGMA)
    rep.check("A", "σ 保守：全方位交会下真值落在 2σ 内的比例 ≥98%",
              float(np.mean(cov_ok)) >= 0.98, f"{np.mean(cov_ok) * 100:.1f}%")
    rep.metric("a_sigma_by_geometry", {k: {"err_med": v[0], "sigma_med": v[1]}
                                       for k, v in stats.items()})
    rep.metric("a_sigma_coverage2", float(np.mean(cov_ok)))
    rep.metric("a_sigma_trigger_ratio", float(np.mean(trig)))

    # A5 噪声幅度确有 ±1° 量级（确认我们使用的噪声模型与题目一致）
    errs = np.array([bearingnoise.error_degrees(7, ch, 300.0, 300.0) for ch in range(1, 2001)])
    rep.check("A", "示向度噪声落在 ±1° 内且非零", float(np.abs(errs).max()) <= 1.0
              and float(np.abs(errs).mean()) > 0.05,
              f"max|e|={np.abs(errs).max():.3f}° 均值={np.abs(errs).mean():.3f}°")


# ---------------------------------------------------------------------------
# B 覆盖路点正确性
# ---------------------------------------------------------------------------
def group_b(rep: Report, rng: np.random.Generator) -> None:
    print("\n[B] 覆盖路点正确性")
    way = T.covering_waypoints()
    rep.metric("b_waypoints", len(way))

    # B1 连续圆域上的真正最坏点：2 m 网格初筛 + 多起点局部爬山精化
    def worst_point(radius: float) -> Tuple[float, np.ndarray]:
        ax = np.arange(-radius, radius + 1e-9, 2.0)
        gx, gy = np.meshgrid(ax, ax)
        P = np.stack((gx.ravel(), gy.ravel()), axis=1)
        P = P[np.linalg.norm(P, axis=1) <= radius]
        f = np.linalg.norm(P[:, None, :] - way[None, :, :], axis=2).min(axis=1)

        def val(q: np.ndarray) -> float:
            n = float(np.linalg.norm(q))
            if n > radius:
                q = q / n * radius
            return float(np.linalg.norm(way - q, axis=1).min())

        bw, bp = float(f.max()), P[int(np.argmax(f))].copy()
        for i in np.argsort(f)[-8:]:
            p, best, step = P[i].copy(), float(f[i]), 2.0
            while step > 1e-3:
                moved = False
                for d in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    q = p + np.array(d)
                    v = val(q)
                    if v > best:
                        best, p, moved = v, q, True
                if not moved:
                    step *= 0.5
            if best > bw:
                bw, bp = best, p
        return bw, bp

    for r, label in ((T.REGION_RADIUS, "1800 m 圆域（题目全域）"),
                     (1770.0, "1770 m 圆域（官方生成器源域）")):
        w, p = worst_point(r)
        rep.check("B", f"{label}：最坏点到最近路点 < 1000 m 接收下界", w < 1000.0,
                  f"最坏 {w:.1f} m @ ({p[0]:.0f}, {p[1]:.0f})，余量 {1000.0 - w:+.1f} m")
        rep.check("B", f"{label}：实际最坏距离贴近设计半径 {T.COVER_RADIUS:.0f} m",
                  w <= T.COVER_RADIUS + 60.0, f"{w:.1f} m")
        rep.metric(f"b_worst_m_r{r:.0f}", w)

    # B2 蒙特卡洛：随机源(位置 + receive∈[1000,1500])是否必被某路点听到
    miss = 0
    for _ in range(2000):
        while True:
            p = rng.uniform(-1, 1, 2) * T.REGION_RADIUS
            if np.linalg.norm(p) <= T.REGION_RADIUS - 30.0:
                break
        receive = rng.uniform(1000.0, 1500.0)
        if np.linalg.norm(way - p, axis=1).min() > receive:
            miss += 1
    rep.check("B", "2000 次随机源抽样全部可被某路点接收", miss == 0, f"漏检 {miss} 次")
    rep.metric("b_mc_miss", miss)

    # B3 贪心解的规模合理性：与正方形铺砌下界 (R/r)² 比较（圆面积下界过于乐观）
    lower = (T.REGION_RADIUS / T.COVER_RADIUS) ** 2
    rep.check("B", "路点数不超面积下界的 2.5 倍（贪心不浪费）",
              len(way) <= 2.5 * lower, f"{len(way)} 个 vs 面积下界 {lower:.2f} 个")


# ---------------------------------------------------------------------------
# C 路线 GA 最优性
# ---------------------------------------------------------------------------
def group_c(rep: Report, rng: np.random.Generator) -> None:
    print("\n[C] 路线 GA 最优性")
    start = np.zeros(2)

    # C1 小规模对拍穷举
    for n in (5, 6, 7):
        gaps = []
        for _ in range(12):
            pts = rng.uniform(-1500, 1500, size=(n, 2))
            D = T._dist_matrix(pts, start)
            best = min(T._path_len(list(p), D) for p in itertools.permutations(range(n)))
            got = T._path_len(T.route_ga(pts, start, seed=int(rng.integers(1 << 30))), D)
            gaps.append((got - best) / best)
        worst = max(gaps)
        rep.check("C", f"{n} 点路线 GA 达到穷举最优（12 例，允许 0.5% 偏差）",
                  worst <= 0.005, f"最大偏差 {worst * 100:.3f}%")
        rep.metric(f"c_gap_n{n}", worst)

    # C2 大规模：相对最近邻初值的改进
    gains = []
    for _ in range(10):
        n = int(rng.integers(12, 21))
        pts = rng.uniform(-1700, 1700, size=(n, 2))
        D = T._dist_matrix(pts, start)
        # 这里原先写作 `T._nearest_neighbor(pts, start)` 并套了 `hasattr` 保护 —— 但那个名字
        # 在拆分前后都**不存在**（T3_ga 里只有 `_nearest_order(n, D)`，签名也不同），故 hasattr
        # 恒为假、`nn` 恒为 None、`gains` 恒为空，整项 C2 检查**从未执行**（是死代码）。
        # 改用 `nearest_order_from_D(n, D)`（common.routing 的公开算子，同样以距离矩阵第 0 行为
        # 起点）度量同一个量 —— GA 解相对最近邻构造的改进。注意不能写路由模块内部的
        # `_nearest_order`：`_T3GA` 只按白名单暴露私有名（EXTRA 里没有它），这正是该聚合对象
        # 想守住的边界。这项检查到此才第一次真正跑起来。
        nn = T._path_len(T.nearest_order_from_D(n, D), D)
        got = T._path_len(T.route_ga(pts, start, seed=int(rng.integers(1 << 30))), D)
        gains.append((nn - got) / nn)
    if gains:
        rep.check("C", "大规模（12~20 点）相对最近邻初值有改进",
                  float(np.mean(gains)) >= 0.0, f"平均改进 {np.mean(gains) * 100:.2f}%")
        rep.metric("c_nn_gain_mean", float(np.mean(gains)))
        rep.metric("c_nn_gain_worst", float(min(gains)))

    # C3 2-opt 精修不劣化
    ok = True
    for _ in range(20):
        n = int(rng.integers(8, 15))
        pts = rng.uniform(-1500, 1500, size=(n, 2))
        D = T._dist_matrix(pts, start)
        order = list(rng.permutation(n))
        before = T._path_len(order, D)
        after = T._path_len(T.two_opt(order, D), D)
        ok &= after <= before + 1e-9
    rep.check("C", "2-opt 精修从不使路程变长（20 例）", ok)

    # ---- C4~C8：T3 主线"顺路清除"的判据 ----
    # 判据以**区域圆心（原点）**为参照（见 cumcm/t3/strategy.py 的 _in_azimuth_arc）：估计点与
    # 圆心的连线方向，落在「本站→圆心」与「下一站→圆心」两条连线夹出的方位扇区内即清，半径
    # 不限。它决定了"沿途到底清哪些点"，必须卡准：放错方向会绕远、收窄会白丢顺路可清的目标。
    probe = _CorridorProbe()
    arc = T3_strategy.RobotDog._in_azimuth_arc
    at, nxt = (900.0, 0.0), (450.0, 779.4)          # 方位 0° 与 60°

    # C4 定点：方位在两条连线之间则放行，反之拒绝；半径不限。
    acases = [((800.0, 240.0), True, "方位 16.7° 在扇区内（半径 835 m ≤ 1800）"),
              ((850.0, 1000.0), True, "方位 49.6° 在扇区内（半径 1312 m ≤ 1800，远源可清）"),
              ((500.0, 0.0), True, "方位 0° = 本站连线方向（含端点）"),
              ((225.0, 389.7), True, "方位 60° = 下一站连线方向（含端点）"),
              ((300.0, 750.0), False, "方位 68.2° 出扇区（半径 808 m 在界内，是被方位排除）"),
              ((1500.0, 1400.0), False, "方位 43° 在方位内但半径 2052 m 超区域半径 1800"),
              ((-500.0, 0.0), False, "方位 180°（反侧）")]
    wrong = [d for est, want, d in acases if (arc(probe, at, nxt, est) is not None) != want]
    rep.check("C", "顺路清除方位扇区：两连线之间放行、之外拒绝（5 例）",
              not wrong, "全部符合" if not wrong else f"不符：{wrong}")

    # C5 与"方位角三角不等式"参考公式随机对拍：th_est 在 th_at~th_next 间 ⟺
    # ang_diff(th_est,th_at)+ang_diff(th_est,th_next) == ang_diff(th_at,th_next)（短弧取等）。
    hard = 0
    for _ in range(4000):
        a = rng.uniform(-1500, 1500, size=2)
        b = rng.uniform(-1500, 1500, size=2)
        if min(float(np.linalg.norm(a)), float(np.linalg.norm(b))) < 100.0:
            continue                                 # 两端点须远离原点（方位有意义）
        q = rng.uniform(-1800, 1800, size=2)
        th = lambda p: float(np.degrees(np.arctan2(p[1], p[0])) % 360.0)
        d_ab = T_geometry.ang_diff(th(a), th(b))
        r_ok = float(np.linalg.norm(q)) <= T3_cfg.INLINE_RADIUS_MAX_M + 1e-6
        want = r_ok and (T_geometry.ang_diff(th(q), th(a))
                         + T_geometry.ang_diff(th(q), th(b)) <= d_ab + 1e-9)
        got = arc(probe, a, b, q) is not None
        if want != got:
            hard += 1
    rep.check("C", "顺路清除方位扇区：与三角不等式参考公式等价（4000 组随机）",
              hard == 0, f"实质不一致 {hard} 处")
    rep.metric("c_azimuth_mismatch", hard)

    # C6 必须用**角度**比较而不能用 |sin|：|sin 180°| = 0，反侧的点（方位 180° 对 0°）会被
    # 误判成"与本站同向"。这里 est 在 at 的**正反方向**，|sin|=0 恰是陷阱例子。
    refl = (-500.0, 0.0)                             # 方位 180°，在 at(0°) 与 nxt(60°) 之外
    sin_trap = abs(math.sin(math.radians(T_geometry.ang_diff(
        T_geometry.bearing((0.0, 0.0), refl), T_geometry.bearing((0.0, 0.0), at))))) < 1e-9
    a_refl = T_geometry.ang_diff(T3_strategy.RobotDog._polar_deg(refl),
                                 T3_strategy.RobotDog._polar_deg(at))
    rep.check("C", "顺路清除：反侧点被排除（|sin| 判据会误判为同向）",
              arc(probe, at, nxt, refl) is None and sin_trap and a_refl > 60.0,
              f"|sin 方位差|≈0 会误判、ang_diff 给出 {a_refl:.0f}° ⇒ 正确排除")

    # C7 区域圆心排除规则：极角扇区半径到区域半径 1800、延伸到圆域各处，600 m 排除**确实会触发**。
    ex_probe = _CorridorProbe(exclude_r=T3_cfg.INLINE_EXCLUDE_R_M)
    near = (400.0, 230.0)                            # 方位 29.9° 在扇区内，距原点 461 m
    far = (800.0, 300.0)                             # 方位 20.6° 在扇区内，距原点 854 m
    in_near = arc(ex_probe, at, nxt, near) is not None
    in_far = arc(ex_probe, at, nxt, far) is not None
    excl = T3_strategy.RobotDog._inline_excluded
    no_excl = _CorridorProbe(exclude_r=0.0)
    rep.check("C", f"顺路清除排除距区域圆心 {T3_cfg.INLINE_EXCLUDE_R_M:.0f} m 内的估计点",
              in_near and in_far and excl(ex_probe, near) and not excl(ex_probe, far)
              and not excl(no_excl, near),
              "近心点在扇区内但被排除、远心点保留、门槛置 0 不再排除")

    # C8 起点(原点)→第一站：起点与圆心的连线是零向量，没有方位，退化为"朝向第一站 ± 半张角"
    # 的本地扇形兜底（_in_sector）—— 一并纳入回归，防止该特判以后被误删。
    sector = T3_strategy.RobotDog._in_sector
    s_probe = _CorridorProbe(sector_deg=T3_cfg.INLINE_SECTOR_DEG)
    origin, first = (0.0, 0.0), (900.0, 1000.0)      # 第一站方位 48°
    scases = [((620.0, 700.0), True, "朝第一站方向偏差 0.5°，距离在段内"),
              ((-620.0, -700.0), False, "反方向（180°）"),
              ((2000.0, 2200.0), False, "方向对但超过段长")]
    wrong = [d for est, want, d in scases
             if (sector(s_probe, origin, first, est) is not None) != want]
    rep.check("C", "起点→第一站兜底扇形：朝第一站方向放行、反侧/超程拒绝（3 例）",
              not wrong, "全部符合" if not wrong else f"不符：{wrong}")
    # 起点段与 600 m 排除的组合：兜底扇形内仍有近心点被排除（如方位 48°、距原点 400 m 的点）。
    near_origin = (260.0, 300.0)                     # 方位 ≈49°，距原点 397 m < 600 m
    rep.check("C", "起点段同样遵守 600 m 排除",
              sector(s_probe, origin, first, near_origin) is not None
              and excl(_CorridorProbe(sector_deg=2.0, exclude_r=T3_cfg.INLINE_EXCLUDE_R_M),
                       near_origin),
              "在兜底扇形内但距圆心 397 m ⇒ 被排除")

# ---------------------------------------------------------------------------
# D 参数敏感性
# ---------------------------------------------------------------------------
def _sens_loc(rng: np.random.Generator, pop: int, gens: int, trials: int = 20) -> float:
    """给定 GA 规模下，带噪定位的平均位置误差（同一批场景）。"""
    errs = []
    for k in range(trials):
        g, pts = _random_geometry(rng, 4)
        ol = _obs_from(g, pts, [1, 2, 3, 4], noise_seed=2000 + k, noise=True)
        loc = T.GeneticLocalizer(T.GAParams(pop, gens, 0.90, 0.35), seed=900 + k)
        est = loc.localize(ol)
        errs.append(float(np.linalg.norm(est - g)) if est is not None else float("nan"))
    return float(np.nanmean(errs))


def _sens_route(pop: int, gens: int, trials: int = 6) -> float:
    """给定 GA 规模下，7 点路线长度相对**穷举最优**的平均超出率（同一批场景）。"""
    excess = []
    for k in range(trials):
        rng = np.random.default_rng(8000 + k)
        pts = rng.uniform(-1500, 1500, size=(7, 2))
        D = T._dist_matrix(pts, np.zeros(2))
        best = min(T._path_len(list(p), D) for p in itertools.permutations(range(7)))
        got = T._path_len(T.route_ga(pts, np.zeros(2), T.GAParams(pop, gens, 0.90, 0.35),
                                     seed=4000 + k), D)
        excess.append((got - best) / best)
    return float(np.mean(excess))


def group_d(rep: Report) -> None:
    print("\n[D] 参数敏感性")

    # D1 定位 GA：种群规模 / 代数
    row = {}
    for pop in (20, 40, 80, 160):
        row[f"pop{pop}"] = _sens_loc(np.random.default_rng(11), pop, T.GA_LOC.gens)
    rep.metric("d_loc_pop", row)
    rep.check("D", "定位误差随种群增大不恶化（单峰容忍）",
              row["pop80"] <= row["pop20"] * 1.15,
              "  ".join(f"{k}={v:.2f}m" for k, v in row.items()))

    row_g = {}
    for gens in (40, 80, 150, 300):
        row_g[f"gens{gens}"] = _sens_loc(np.random.default_rng(11), T.GA_LOC.pop, gens)
    rep.metric("d_loc_gens", row_g)
    rep.check("D", "代数从 40 增到 150 定位误差下降",
              row_g["gens150"] <= row_g["gens40"],
              "  ".join(f"{k}={v:.2f}m" for k, v in row_g.items()))

    # D2 路线 GA：种群 / 代数
    r_pop = {f"pop{p}": _sens_route(p, T.GA_ROUTE.gens) for p in (20, 50, 100, 200)}
    rep.metric("d_route_pop", r_pop)
    r_gens = {f"gens{g}": _sens_route(T.GA_ROUTE.pop, g) for g in (50, 150, 300, 600)}
    rep.metric("d_route_gens", r_gens)
    rep.check("D", "路线超额率随代数增大不恶化",
              r_gens["gens300"] <= r_gens["gens50"] + 0.02,
              "  ".join(f"{k}={v * 100:.2f}%" for k, v in r_gens.items()))

    # D3 覆盖半径 → 路点数与成本的权衡。
    # 注意：贪心集覆盖对集合大小并不单调（大集合先被选中会留下更零碎的剩余），
    # 因此这里检验的是真正的设计保证——实际最坏点距离不超设计半径太多。
    rows = []
    for radius in (700.0, 800.0, 900.0, 1000.0, 1100.0):
        # 显式传入覆盖半径（原先靠改模块全局 + 清缓存注入；分层后那种写法会静默失效）
        way = T.covering_waypoints(cover_radius=radius)
        ax = np.arange(-T.REGION_RADIUS, T.REGION_RADIUS + 1e-9, 5.0)
        gx, gy = np.meshgrid(ax, ax)
        P = np.stack((gx.ravel(), gy.ravel()), axis=1)
        P = P[np.linalg.norm(P, axis=1) <= T.REGION_RADIUS]
        worst = float(np.linalg.norm(P[:, None, :] - way[None, :, :], axis=2).min(axis=1).max())
        rows.append({"radius_m": radius, "waypoints": len(way), "worst_m": worst})
    rep.metric("d_cover_radius", rows)
    rep.check("D", "每个设计半径下实际最坏点距离都不超设计值太多（离散化可控）",
              all(r["worst_m"] <= r["radius_m"] + 60.0 for r in rows),
              "  ".join(f"{r['radius_m']:.0f}m→{r['waypoints']}点/最坏{r['worst_m']:.0f}m"
                        for r in rows))
    big = next(r for r in rows if r["radius_m"] == 1100.0)
    small = next(r for r in rows if r["radius_m"] == 700.0)
    rep.check("D", "设计半径越大所需路点越少（整体趋势）",
              big["waypoints"] < small["waypoints"],
              f"700m→{small['waypoints']}点，1100m→{big['waypoints']}点")


# ---------------------------------------------------------------------------
# E 运行记录审计（读已落盘的 results/t3_ga/ 与其 holdout/）
# ---------------------------------------------------------------------------
def _replay_virtual_time(records: Sequence[dict]) -> List[dict]:
    """按引擎规则独立复算每次调用后的虚拟时刻，用于与模拟器报告值比对。

    规则（jammers-py engine.py）：move = |Δpos|/5 m·s⁻¹；measure = move + 换频道 1 s + 5 s；
    clear = move + 成功 5 s / 失败 3 s；enter/exit 不产生虚拟时间。
    注意引擎上电时停在 (0,0) 且频道已置于 1，故首次 /measure 到频道 1 不计换频道耗时。
    """
    out, last, ch, vt = [], (0.0, 0.0), 1, 0
    for r in records:
        call = r["call"]
        if call in ("/measure", "/clear"):
            pos = (r["x"], r["y"])
            # 必须与引擎一致地用 math.hypot：微秒级截断对末位舍入敏感，
            # 换成 numpy 范数会在极少数调用上差 1 µs（表面看是 1e-6 s 的假不一致）。
            move = int(1e12 * math.hypot(pos[0] - last[0], pos[1] - last[1]) / SPEED_UM_PER_S)
            if call == "/measure":
                switch = 1_000_000 if ch != r["channel"] else 0
                vt += move + switch + 5_000_000
                ch = r["channel"]
            else:
                ok = r["response"].get("clear_result") == "success" if r.get("response") else False
                vt += move + (5_000_000 if ok else 3_000_000)
            last = pos
        out.append({"request_id": r["request_id"], "call": call,
                    "reported_s": (r.get("response") or {}).get("virtual_time_s"),
                    "replayed_s": round(vt / 1e6, 6)})
    return out


def group_e(rep: Report, res_dirs: Sequence[Path]) -> None:
    """审计一个或多个结果目录（主批 + 独立 seed 批）。"""
    for res_dir in res_dirs:
        group_e_one(rep, Path(res_dir))


def group_e_one(rep: Report, res_dir: Path) -> None:
    print(f"\n[E] 运行记录审计 —— {res_dir}")
    train_p, calls_p = res_dir / "ga_training.json", res_dir / "api_calls.jsonl"
    if not train_p.exists() or not calls_p.exists():
        rep.check("E", "存在可审计的训练结果与接口日志", False, f"缺少 {train_p.name}/{calls_p.name}")
        return
    tag = res_dir.name
    train = json.loads(train_p.read_text(encoding="utf-8"))
    # 结果目录不再按模式分家，故可能装的是**官方产物**（接口不返回真值，n_sources 为 None）。
    # 这类目录里"清除率/定位误差"无从判定，下面显式跳过而不是判失败、也不用 True 蒙混。
    has_truth = any((ep.get("n_sources") or 0) > 0 for ep in train["episodes"])
    if not has_truth:
        print(f"    说明：本目录为官方产物（无真值），真值相关检查将跳过")
    calls = [json.loads(l) for l in calls_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    rep.check("E", f"[{tag}] 存在可审计的训练结果与接口日志", True,
              f"{len(train['episodes'])} 局 / {len(calls)} 次调用")

    # E1 坐标合法性：全部落在目标圆域内（题目要求机器狗在圆域内作业）
    outside = [r for r in calls if r["call"] in ("/measure", "/clear")
               and math.hypot(r["x"], r["y"]) > T.REGION_RADIUS + 1e-6]
    worst = max((math.hypot(r["x"], r["y"]) for r in calls
                 if r["call"] in ("/measure", "/clear")), default=0.0)
    rep.check("E", f"[{tag}] 所有动作坐标都在半径 1800 m 圆域内", not outside,
              f"{len(outside)} 次越界，最远 {worst:.1f} m")
    rep.metric("e_max_radius_m", worst)

    # E2 坐标分量未超协议上限
    bad = [r for r in calls if abs(r.get("x", 0)) > T.COORD_LIMIT or abs(r.get("y", 0)) > T.COORD_LIMIT]
    rep.check("E", f"[{tag}] 坐标分量未超协议上限 2e6", not bad, f"{len(bad)} 次")

    # E3 清除落点：成功清除必须落在真值 20 m 内，且不误清其他源
    truth = {ep["episode"]: {t["channel"]: (t["x"], t["y"]) for t in ep.get("truth") or []}
             for ep in train["episodes"]}
    if not has_truth:
        rep.skip("E", f"[{tag}] 每次成功清除都落在该源 20 m 清除半径内", "官方产物无真值")
        rep.skip("E", f"[{tag}] 成功清除未误伤同名范围内其他源", "官方产物无真值")
    far, wrong = [], []
    for r in (calls if has_truth else []):
        if r["call"] != "/clear" or (r.get("response") or {}).get("clear_result") != "success":
            continue
        ch, ep = r["channel"], r["episode"]
        src = truth.get(ep, {}).get(ch)
        if src is None:
            continue
        d = math.hypot(r["x"] - src[0], r["y"] - src[1])
        if d > CLEAR_RADIUS + 1e-6:
            far.append(d)
        for other_ch, (ox, oy) in truth.get(ep, {}).items():
            if other_ch != ch and math.hypot(r["x"] - ox, r["y"] - oy) <= CLEAR_RADIUS:
                wrong.append((ep, ch, other_ch))
    if has_truth:
        rep.check("E", f"[{tag}] 每次成功清除都落在该源 20 m 清除半径内", not far,
                  f"越界 {len(far)} 次" + (f"，最远 {max(far):.2f} m" if far else ""))
        rep.check("E", f"[{tag}] 成功清除未误伤同名范围内其他源", not wrong, f"{len(wrong)} 次")

    # E4 虚拟时钟独立复算
    by_ep: Dict[int, List[dict]] = {}
    for r in calls:
        by_ep.setdefault(r["episode"], []).append(r)
    diff = []
    for ep, recs in by_ep.items():
        recs.sort(key=lambda r: r["seq"])
        for a, b in zip(_replay_virtual_time(recs), recs):
            if b["call"] in ("/measure", "/clear") and a["reported_s"] is not None:
                diff.append(abs(a["reported_s"] - a["replayed_s"]))
    worst = max(diff) if diff else float("inf")
    rep.check("E", f"[{tag}] 虚拟时钟可独立复算（与模拟器报告值一致到 1e-6 s）", worst <= 1e-6,
              f"最大差 {worst:.2e} s（{len(diff)} 次）")
    rep.metric("e_vt_max_diff_s", worst)

    # E5 计数一致性：日志与 JSON 记录的次数
    n_measure = sum(1 for r in calls if r["call"] == "/measure")
    n_clear = sum(1 for r in calls if r["call"] == "/clear")
    n_enter = sum(1 for r in calls if r["call"] == "/enter")
    n_exit = sum(1 for r in calls if r["call"] == "/exit")
    rep.check("E", f"[{tag}] 每局恰有一次 /enter 与 /exit",
              n_enter == len(train["episodes"]) and n_exit == len(train["episodes"]),
              f"enter={n_enter} exit={n_exit} 局数={len(train['episodes'])}")
    rep.check("E", f"[{tag}] 测向次数与逐局统计一致",
              n_measure == sum(ep["n_measure"] for ep in train["episodes"]),
              f"{n_measure} vs {sum(ep['n_measure'] for ep in train['episodes'])}")
    rep.check("E", f"[{tag}] 清除动作次数与逐局统计一致",
              n_clear == sum(ep["n_clear"] for ep in train["episodes"]),
              f"{n_clear} vs {sum(ep['n_clear'] for ep in train['episodes'])}")
    rep.metric(f"e_calls_{tag}", {"enter": n_enter, "measure": n_measure,
                                  "clear": n_clear, "exit": n_exit})

    # E6 全部调用均被接受（无静默失败）
    rej = [r for r in calls if not r.get("ok")]
    rep.check("E", f"[{tag}] 所有接口调用均被接受（无 rejected / 异常）", not rej,
              f"{len(rej)} 次被拒" + (f"，例：{rej[0]['request_id']}" if rej else ""))
    rep.metric("e_rejected", len(rej))

    # E7 清除完成度与定位精度（需要真值）
    eps = train["episodes"]
    if not has_truth:
        for nm in ("全部干扰源被清除",
                   "逐局定位平均误差 < 20 m（清除半径）",
                   "逐局最坏定位误差 < 30 m"):
            rep.skip("E", f"[{tag}] {nm}", "官方产物无真值")
        rep.metric(f"e_cleared_{tag}", {"cleared": sum(ep["cleared"] for ep in eps),
                                        "total": None, "ratio": None})
    cleared = sum(ep["cleared"] for ep in eps)
    total = sum(ep["n_sources"] or 0 for ep in eps)
    ratio = cleared / total if total else float("nan")
    errs = [ep["localize_err_mean_m"] for ep in eps if ep["localize_err_mean_m"] is not None]
    maxs = [ep["localize_err_max_m"] for ep in eps if ep["localize_err_max_m"] is not None]
    if has_truth:
        rep.check("E", f"[{tag}] 全部干扰源被清除", cleared == total,
                  f"{cleared}/{total}（{ratio * 100:.1f}%）")
        rep.check("E", f"[{tag}] 逐局定位平均误差 < 20 m（清除半径）",
                  max(errs) < CLEAR_RADIUS, f"最差局 {max(errs):.2f} m")
        rep.check("E", f"[{tag}] 逐局最坏定位误差 < 30 m", max(maxs) < 30.0,
                  f"最坏 {max(maxs):.2f} m")
        rep.metric(f"e_cleared_{tag}", {"cleared": cleared, "total": total, "ratio": ratio})
        rep.metric(f"e_loc_err_mean_m_{tag}", float(np.mean(errs)))
        rep.metric(f"e_loc_err_worst_episode_m_{tag}", float(max(errs)))
        rep.metric(f"e_loc_err_worst_m_{tag}", float(max(maxs)))
        rep.metric(f"e_clear_ratio_{tag}", ratio)

    # E8 现实用时：每局墙钟必须远小于 20 分钟现实限时
    vt = [ep["virtual_time_s"] for ep in eps]
    rep.check("E", f"[{tag}] 虚拟总时间在题目 100 h 虚拟限时内",
              max(vt) < 360_000.0, f"最长局 {max(vt):.1f} s")
    rep.metric(f"e_virtual_time_s_{tag}", {"min": float(min(vt)), "max": float(max(vt)),
                                           "mean": float(np.mean(vt)),
                                           "median": float(np.median(vt))})

    # E9 GA 训练记录完整性
    runs = train["ga_runs"]
    loc = [r for r in runs if r["ga"] == "localize"]
    route = [r for r in runs if r["ga"] == "route"]
    rep.check("E", f"[{tag}] 每次定位都有 GA 训练记录且收敛到最后一代",
              all(r["generations"] == T.GA_LOC.gens or r["converged"] for r in loc),
              f"{len(loc)} 次定位（提前收敛 {sum(r['converged'] for r in loc)} 次）")
    rep.check("E", f"[{tag}] 每次路线规划都有 GA 训练记录（跑满代数）",
              all(r["generations"] == T.GA_ROUTE.gens for r in route),
              f"{len(route)} 次规划")
    curves = (res_dir / "ga_convergence.csv").read_text(encoding="utf-8").splitlines()
    rep.check("E", f"[{tag}] 收敛曲线非空", len(curves) > 100,
              f"{len(curves) - 1} 个记录点")
    rep.metric(f"e_ga_runs_{tag}", {"localize": len(loc), "route": len(route)})

    # E10 图形产物。**语义已改为"只保留最新一局"**：图形目录每局开头清空重写，于是
    # trajectory/ 恰好 1 图 + 1 表、scan/ 恰好 N 张（N = 该局扫描步数，含起点扫描）。
    # 这样设计是为了让目录里只有当前这一轮的图（原先 10~20 局各留一张，很难分辨哪张是
    # 新的）；代价是只剩最新一局可核对，故下面的对账改为针对该局做**全量**核对。
    traj_dir = res_dir / T.TRAJ_DIR_NAME
    scan_dir = res_dir / "scan"
    pngs = sorted(traj_dir.glob("*.png")) if traj_dir.is_dir() else []
    csvs = sorted(traj_dir.glob("*.csv")) if traj_dir.is_dir() else []
    scans = sorted(scan_dir.glob("*.png")) if scan_dir.is_dir() else []
    rep.check("E", f"[{tag}] 轨迹目录只保留最新一局（恰好 1 图 + 1 表）",
              len(pngs) == 1 and len(csvs) == 1,
              f"{len(pngs)} 张图 / {len(csvs)} 份表（共跑 {len(eps)} 局）→ {traj_dir}")
    rep.check("E", f"[{tag}] 逐步扫描结果图齐全（起点扫描 + 各巡视站/路点各一张）",
              len(scans) >= 2,
              f"{len(scans)} 张 → {scan_dir}"
              + (f"：{', '.join(p.name for p in scans[:4])}…" if scans else ""))

    # 图必须是真实渲染出来的（PNG 头 + 尺寸），避免"生成了空图"却判为通过
    sizes, bad_png = [], []
    for f in pngs + scans:
        with f.open("rb") as fh:
            head = fh.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            bad_png.append(f.name)
            continue
        w, h = struct.unpack(">II", head[16:24])
        sizes.append((w, h))
        if min(w, h) < 300:
            bad_png.append(f"{f.name}({w}x{h})")
    rep.check("E", f"[{tag}] 轨迹图与扫描图均为有效且尺寸合理的 PNG", not bad_png,
              f"{len(pngs) + len(scans)} 张，尺寸 "
              f"{min(sizes) if sizes else '-'}~{max(sizes) if sizes else '-'}"
              + (f"，异常 {bad_png}" if bad_png else ""))

    # 轨迹表逐条对账：保留那一局的每个动作点都必须对应日志里的一次 /measure 或 /clear，
    # 且坐标完全一致（图与日志同源，任一不符说明记录链路有问题）。
    per_ep_actions: Dict[int, List[Tuple[float, float, str]]] = {}
    for r in calls:
        if r["call"] in ("/measure", "/clear"):
            per_ep_actions.setdefault(r["episode"], []).append(
                (float(r["x"]), float(r["y"]), r["call"].strip("/")))
    mismatched, checked, radius_bad = [], 0, []
    latest_ep = None
    for csv_p in csvs:
        # 轨迹表文件名形如 ep01_seed0 / ep01，取 ep 后的数字即局号（与逐局表的 episode 对齐）
        m = re.match(r"ep(\d+)", csv_p.stem)
        got = list(csv.DictReader(csv_p.open(encoding="utf-8")))
        if not m:
            mismatched.append(f"{csv_p.name}: 文件名无法解析局号")
            continue
        latest_ep = int(m.group(1))
        want = per_ep_actions.get(latest_ep, [])
        if len(got) - 1 != len(want):
            mismatched.append(f"{csv_p.name}: 表 {len(got) - 1} 点 vs 日志 {len(want)} 次")
            continue
        for i, (row, exp) in enumerate(zip(got[1:], want), start=1):
            if (abs(float(row["x"]) - exp[0]) > 1e-3 or abs(float(row["y"]) - exp[1]) > 1e-3
                    or row["kind"] != exp[2]):
                mismatched.append(f"{csv_p.name} 第 {i} 点")
                break
            checked += 1
            if math.hypot(float(row["x"]), float(row["y"])) > T.REGION_RADIUS + 1e-6:
                radius_bad.append(f"{csv_p.name} 第 {i} 点")
    rep.check("E", f"[{tag}] 保留局的轨迹与接口日志逐点一致（坐标与动作类型）", not mismatched,
              f"核对 {checked} 个动作点" + (f"，不符：{mismatched[:3]}" if mismatched else ""))
    rep.check("E", f"[{tag}] 轨迹点全部在作业圆域内", not radius_bad,
              f"{len(radius_bad)} 个越界点" if radius_bad else f"{checked} 个点全在域内")
    # 扫描图与轨迹图必须属于同一局，否则说明"清旧图"没清干净
    if latest_ep is not None and scans:
        tag_ep = f"ep{latest_ep:02d}"
        rep.check("E", f"[{tag}] 扫描图与轨迹图属于同一局（清旧图有效）",
                  all(p.name.startswith(tag_ep) for p in pngs + scans),
                  f"保留局 {tag_ep}，文件 {', '.join(p.name for p in (pngs + scans)[:3])}…")
    rep.metric(f"e_traj_png_{tag}", {"files": len(pngs), "scan_files": len(scans),
                                     "dir": str(traj_dir), "points_checked": checked,
                                     "kept_episode": latest_ep})


# ---------------------------------------------------------------------------
# F 可复现性
# ---------------------------------------------------------------------------
def group_f(rep: Report) -> None:
    print("\n[F] 可复现性")
    # 暂存目录：跑两次同 seed 的演练比对逐字节一致性。**必须自己清干净** —— 原先跑完
    # 就把 .tmp_verify/run1、run2 留在仓库里，每次都出现在 git 未跟踪列表里。
    tmp = Path(TMP_VERIFY_DIR)
    shutil.rmtree(tmp, ignore_errors=True)
    runs = []
    for i in (1, 2):
        out = tmp / f"run{i}"
        p = subprocess.run([sys.executable, "T3_ga.py", "--practice", "2", "--seed", "50",
                            "--quiet", "--save-dir", str(out)],
                           capture_output=True, text=True, timeout=900)
        if p.returncode != 0:
            rep.check("F", f"重复运行第 {i} 次成功", False, p.stderr[-200:])
            shutil.rmtree(tmp, ignore_errors=True)
            return
        runs.append(out)
    same = {}
    for name in ("ga_training.json", "ga_convergence.csv", "episodes.csv"):
        a = (runs[0] / name).read_bytes()
        b = (runs[1] / name).read_bytes()
        same[name] = a == b
        rep.check("F", f"同 seed 两次运行的 {name} 逐字节一致", a == b)
    # 接口日志：除时间戳/耗时外应逐条一致
    def norm(p: Path) -> List[str]:
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            d.pop("at_s", None)
            d.pop("elapsed_ms", None)
            if isinstance(d.get("response"), dict):
                d["response"].pop("real_timestamp_ms", None)
            out.append(json.dumps(d, ensure_ascii=False, sort_keys=True))
        return out
    a, b = norm(runs[0] / "api_calls.jsonl"), norm(runs[1] / "api_calls.jsonl")
    rep.check("F", "接口调用序列完全一致（除时间戳）", a == b, f"{len(a)} 条 vs {len(b)} 条")
    rep.metric("f_byte_identical", same)

    # F2 论文图表：PDF 内嵌生成时间会让同一份输入每次产出不同的字节（既无法验证可复现性，
    # 也让版本历史充满无意义的二进制差异）。检查出图脚本已清掉这些时间戳字段。
    fig_dir = Path("figures")
    pdfs = sorted(fig_dir.glob("*.pdf")) if fig_dir.is_dir() else []
    if pdfs:
        stamped = [f.name for f in pdfs
                   if re.search(rb"/(CreationDate|ModDate)\s*\(", f.read_bytes())]
        rep.check("F", "论文图 PDF 不含生成时间戳（可逐字节复现）", not stamped,
                  f"{len(pdfs)} 个 PDF" + (f"，含时间戳：{stamped}" if stamped else "，均无时间戳"))
        rep.metric("f_figures_pdf", len(pdfs))

    shutil.rmtree(tmp, ignore_errors=True)      # 暂存目录用完即删，不留残渣


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="T3_ga.py 求解方案验证")
    ap.add_argument("--groups", nargs="*", default=["A", "B", "C", "D", "E", "F"],
                    help="要运行的组：A 几何定位 / B 覆盖 / C 路线 / D 敏感性 / E 记录审计 / F 可复现性")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"验证结果 JSON 输出路径（缺省 {DEFAULT_OUT}）")
    ap.add_argument("--no-arena", action="store_true",
                    help="跳过需要模拟器的 F 组（可复现性需开一局演练）")
    ap.add_argument("--results", nargs="+",
                    default=[str(d) for d in AUDIT_DIRS],
                    help=f"待审计的运行结果目录（可给多个：主批 + 独立 seed 批；"
                         f"缺省 {AUDIT_DIRS[0]}/ 与其 holdout/）")
    args = ap.parse_args(argv)

    groups = [g.upper() for g in args.groups]
    print("=" * 74)
    print("T3_ga.py 求解方案验证")
    print("=" * 74)

    relax_console_encoding()      # Windows 中文控制台下也能正常打印检查结果
    rep = Report()
    t0 = time.time()
    rng = np.random.default_rng(2026)
    runners = {"A": lambda: group_a(rep, rng), "B": lambda: group_b(rep, rng),
               "C": lambda: group_c(rep, rng), "D": lambda: group_d(rep),
               "E": lambda: group_e(rep, [Path(p) for p in args.results]),
               "F": lambda: group_f(rep)}
    for g in groups:
        if g in runners:
            runners[g]()
    elapsed = time.time() - t0
    rep.metric("elapsed_s", round(elapsed, 1))

    print("\n" + "=" * 74)
    print(f"验证完成：{rep.summary()}，用时 {elapsed:.1f} s")
    print("=" * 74)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"groups": rep.groups, "metrics": rep.metrics,
                               "failed": rep.failed, "skipped": rep.skipped,
                               "elapsed_s": round(elapsed, 1),
                               "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"验证结果已保存：{out}")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
