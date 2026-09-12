"""机器狗策略（问题四）：扫描（阶段一）+ 定向感知定位清除（阶段二）。

与问题三的差别（其余手法——就近试清、K 圆覆盖、Fisher 准则补测、沿示向度逼近——同源复用，
见 t3.strategy 的模块文档）：

* 阶段一**复用问题三的 7 个覆盖基点并外扩一圈**：按 t4.sweep 的方案依次走到 20 个
  测量点（原点 + 7 覆盖基点 + 12 个均匀方位的 1850 m 外圈点），在每个点对"还没听到过"的
  频道测向，另对已听到但估计还不够准、且当前点就在其估计附近的频道顺路补一两个视角。
  **布局效果**：实测听到率 99.9891%（4 224 640 个算例、漏 460），20 局演练 256/256 全清，
  故绝大多数干扰源（全向或定向）在扫描结束前被听到——
  "确保所有干扰源被清除"的检测侧依据（清除侧见下）。
* no_signal 不再是"源在接收半径之外"的硬约束（可能是方向不对，见 DirProbRegion 与 config），
  定位区域只用 direction/near 两类约束；"跳过必无信号的测量"仍保留（区域整体在 1500 m 外才
  可证），但作用大幅收窄。
* homing 兜底针对"起始点可能恰在波束背面"加了反方向修正：起点实测无信号时，先朝该频道**测量
  点质心**（必在迎光侧）步进，直到重新采到示向度再正常逼近——半圆盘是凸集，从迎光侧向源
  逼近始终留在波束内，故最终必能贴到 20 m 内清除。
* 阶段二仍对"扫描结束时已听到的频道"全部处理，逐频道四级清除。
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from functools import partial
from typing import Any, Sequence

import numpy as np

from common.actions import ActionRecorder
from common.geometry import ang_diff
from common.geometry import clamp_to_region as _clamp_to_region
from common.geometry import dist
from common.routing import dist_matrix, exact_open_order
from common.sim_client import RecordedSim
from t4.config import (BEARING_ERROR_DEG, CHANNELS, CLEAR_RADIUS, CLIP_ERR,
                             CLIP_SIDES, HOMING_CAP, HOMING_MAX, HOMING_STEP,
                             INLINE_MAX_MEC_R, INLINE_NEAR_R, INLINE_OUTER_R_M,
                             INLINE_R_MAX, INLINE_R_MIN,
                             K_CLEAR_MAX, K_COVER_SAMPLES, K_COVER_STEP_MIN, NEAR_RADIUS,
                             OBS_CAP, RECEIVE_MAX, REFINE_MAX, REGION_MARGIN, REGION_RADIUS, SAFETY_MARGIN,
                             SIDE_SCAN_DIAM, SWEEP_OPP_RADIUS, TOL, TRY_CLEAR_RADIUS)
from t4.probing import hypothesis_points, probe_candidates
from t4.regions import DirProbRegion, Meas, Obs
from t4.sweep import SweepPlan

clamp_to_region = partial(_clamp_to_region, radius=REGION_RADIUS - REGION_MARGIN)


class RobotDog(ActionRecorder):
    """两阶段机器狗（问题四）。

    阶段一（扫描）：在出发点做全频道扫描（原点测量位置），随后按扫描方案依次走到其余
    测量点；每点对"未听到过的频道"测向（近距就地清除），保证任何源都被听到 ≥ 1 次；
    顺路对估计附近的频道补测第二/第三视角；每段站间移动时按问题三机制做**途中顺路清除**
    （_inline_clear：方位扇区内的未清频道估计点顺路试清），扫描结束已清掉约 60% 的源。

    阶段二（定位与清除）：对扫描结束时已听到的每个频道，四级清除；**每次清除动作后
    顺路补测**所有"直径比较大（> 200 m）且观测未满"的未清频道（多视角缩小定位区域，
    见 _side_scan）：
      1. 就近试清（走到定位区域最小覆盖圆圆心直接 /clear 一次，未命中就地复测）；
      2. 多清几次（K 个半径 20 m 的圆盖满区域外，依次补清）；
      3. 按文献准则补测缩小区域后再清（含单射线局面）；
      4. 兜底沿最新示向度以 16 m 步长逼近；起点在波束背面时先朝测量点质心回撤再逼近。
    """

    def __init__(self, sim: "Simulator", verbose: bool = True, logfile: str | None = None,
                 episode: int = 0, clear: bool = True,
                 k_clear_max: int = K_CLEAR_MAX,
                 sweep_opp_radius: float = SWEEP_OPP_RADIUS,
                 inline_r_min: float = INLINE_R_MIN,
                 inline_r_max: float = INLINE_R_MAX,
                 inline_r_min_in: float | None = None,
                 inline_r_max_in: float | None = None,
                 inline_r_min_out: float | None = None,
                 inline_r_max_out: float | None = None,
                 inline_near_r: float = INLINE_NEAR_R,
                 inline_max_mec_r: float = INLINE_MAX_MEC_R,
                 inline_outer_r: float = INLINE_OUTER_R_M,
                 api_log: "ApiLog" | None = None) -> None:
        """构造两阶段机器狗（问题四）

        Args:
            sim: 模拟器接口（4 条指令的薄封装）；给了 api_log 就再包一层记录代理
            verbose: 是否实时回显过程日志
            logfile: 过程日志文件路径；None 表示日志只回显不落盘
            episode: 局号，写进日志与 request_id，便于与模拟器行为日志对照
            clear: 是否启用清除（False 时只扫描定位，对应 --survey-only）
            k_clear_max: 覆盖定位区域的 20 m 圆最多补选几个
            sweep_opp_radius: 顺路补测半径 / m
            inline_r_min: 前向顺路半径下界 / m（内外圈缺省值都源自它）
            inline_r_max: 前向顺路半径上界 / m
            inline_r_min_in: 内圈前向半径下界 / m；None 跟随 inline_r_min
            inline_r_max_in: 内圈前向半径上界 / m；None 跟随 inline_r_max
            inline_r_min_out: 外圈前向半径下界 / m；None 跟随 inline_r_min
            inline_r_max_out: 外圈前向半径上界 / m；None 取 +inf（外圈不设上界）
            inline_near_r: 近距顺路清除半径 / m
            inline_max_mec_r: 参与顺路的区域最大 mec 半径 / m
            inline_outer_r: 内外圈分界：本站距原点超过它按外圈处理
            api_log: 接口调用日志；None 表示不记录
        """
        self.sim = sim if api_log is None else RecordedSim(sim, api_log, episode)
        self.verbose = verbose
        self.clear_enabled = bool(clear)
        self.k_clear_max = int(k_clear_max)
        self.sweep_opp_radius = float(sweep_opp_radius)   # 顺路补测半径 / m
        self.inline_r_min = float(inline_r_min)           # 前向顺路半径下界 / m（内外组缺省源自它）
        self.inline_r_max = float(inline_r_max)           # 前向顺路半径上界 / m
        # 内圈 / 外圈前向半径：None 时内圈跟随全局 inline_r_min/r_max（2026-09-12 内外分组）。
        # **外圈不设半径上界**（= +inf）：源生成域半径 1770 m，估计点半径 r_est 恒 ≤ 1770，故
        # 上界只在设到 <1770 时才起作用、且必然更差（实测 1533→6727 s、1689→6430 s；≥1767 全
        # 为 6363 s 的平台），对 r_est 无任何约束意义，故不作为可调参数。
        self.inline_r_min_in = float(inline_r_min if inline_r_min_in is None else inline_r_min_in)
        self.inline_r_max_in = float(inline_r_max if inline_r_max_in is None else inline_r_max_in)
        self.inline_r_min_out = float(inline_r_min if inline_r_min_out is None else inline_r_min_out)
        self.inline_r_max_out = math.inf if inline_r_max_out is None else float(inline_r_max_out)
        self.inline_outer_r = float(inline_outer_r)       # 内外圈分界：本站距原点 > 该值按外圈
        self.inline_near_r = float(inline_near_r)         # 近距顺路清除半径 / m
        self.inline_max_mec_r = float(inline_max_mec_r)   # 参与顺路的区域最大 mec 半径 / m
        self._logfile = open(logfile, "w", encoding="utf-8") if logfile else None
        self.obs: dict[int, list[Obs]] = defaultdict(list)
        self.meas: dict[int, list[Meas]] = defaultdict(list)
        self.n_skip = 0
        self.actions: list[dict[str, Any]] = []
        self.regions: dict[int, DirProbRegion] = {}
        self.tracks: dict[int, dict[str, Any]] = {}
        self.cleared: set = set()
        self.pos = np.zeros(2)
        self.vt = 0.0
        self.n_measure = 0
        self.n_clear = 0
        self.n_side_scan = 0        # 清除动作时对"直径大频道"的顺路补测次数
        self.n_inline = 0            # 途中顺路清除命中的次数（沿用问题三机制）
        self.n_inline_fail = 0       # 途中顺路清除未命中的次数
        self.n_side_cand = 0        # 顺路补测候选数（直径大且未满且够近）
        self.n_side_skip_far = 0    # 因距估计点 >1500m 必然无信号而跳过的候选
        self.episode = episode
        self.deadline = float("inf")
        self.travel_m = 0.0
        self.stage = "sweep"
        self.plan: SweepPlan | None = None
        self.first_heard: dict[int, int] = {}       # 频道 → 首次听到时的测量位置序号
        self.first_heard_at: dict[int, tuple[float, float]] = {}
        self.scan_steps: list[dict[str, Any]] = []
        self._cur_step: dict[str, Any] | None = None

    # ---- 日志与原子动作（log / close / clear / _note / _polar_deg 等）见 common.actions ----
    def measure(self, x: float, y: float, channel: int) -> dict:
        """在 (x, y) 处测量 channel，并把结果并入定位区域与观测记录

        Args:
            x: 测量点 x 坐标 / m
            y: 测量点 y 坐标 / m
            channel: 频道号

        Returns:
            dict: 模拟器原始响应（含 measure_result / svd_deg / virtual_time_s）

        Raises:
            RuntimeError: 模拟器拒绝本次 /measure
        """
        x, y = float(x), float(y)
        r = self.sim.measure(x, y, channel)
        if not r.get("accepted"):
            raise RuntimeError(f"/measure 被拒绝：{r}")
        self.travel_m += dist(self.pos, (x, y))
        self.pos, self.vt = np.array([x, y]), float(r["virtual_time_s"])
        self.n_measure += 1
        outcome = r.get("measure_result", "no_signal")
        reg = self.region(channel)
        m = Meas(channel, x, y, outcome,
                 float(r["svd_deg"]) if outcome == "direction" else None, self.stage)
        self.meas[channel].append(m)
        self._apply_meas(reg, m)
        if outcome == "direction":
            self.obs[channel].append(Obs(channel, x, y, m.theta, self.stage))
            if channel not in self.first_heard:
                self.first_heard[channel] = int(self._cur_step["index"]) if self._cur_step else -1
                self.first_heard_at[channel] = (x, y)
        self._note("measure", x, y, channel, outcome=outcome, theta=m.theta)
        if self._cur_step is not None:
            self._cur_step["measures"].append({"channel": int(channel), "outcome": outcome,
                                               "theta": None if m.theta is None
                                               else float(m.theta)})
        return r

    def _provable_no_signal(self, channel: int, at: Sequence[float]) -> bool:
        """能否证明"在 at 处测 channel 必然无信号"，从而省掉这次测量。

        可能源集合是真实源位置的超集，故"区域到 at 的最小距离 > 1500 m"⇒ 真源到 at 的距离也
        > 1500 m ≥ 有效接收半径 ⇒ 无论全向还是定向都必然收不到。此时测量不带新信息，跳过。
        """
        reg = self.regions.get(channel)
        return reg is not None and reg.provably_out_of_reach(at, RECEIVE_MAX)

    def _end_scan_step(self, counts: dict[str, int]) -> None:
        """收尾当前扫描步骤：补上统计、路径、区域与估计点后追加进 scan_steps

        Args:
            counts: 本站各测量结果的计数（direction / near / no_signal / skip）
        """
        step = self._cur_step
        self._cur_step = None
        if step is None:
            return
        step["counts"] = dict(counts)
        step["virtual_time_s"] = round(self.vt, 3)
        step["travel_m"] = round(self.travel_m, 2)
        step["cleared"] = sorted(self.cleared)
        # 路径由已登记的动作点重建，开头补上出发点 (0, 0)
        step["path"] = [(0.0, 0.0)] + [(a["x"], a["y"]) for a in self.actions]
        step["regions"] = {ch: verts for ch, reg in sorted(self.regions.items())
                           if len(verts := list(reg.vertices)) >= 3}
        step["estimates"] = {ch: (mec[0], mec[1], mec[2])
                             for ch, reg in sorted(self.regions.items())
                             if (mec := reg.enclosing_circle) is not None}
        self.scan_steps.append(step)

    def region(self, channel: int) -> DirProbRegion:
        """取频道 channel 的定位区域（首次访问时按已有测量惰性重建）

        Args:
            channel: 频道号

        Returns:
            DirProbRegion: 该频道的可能源集合
        """
        if channel not in self.regions:
            reg = DirProbRegion(err=BEARING_ERROR_DEG, radius=REGION_RADIUS, sides=CLIP_SIDES)
            # 首次建区域：把该频道此前的测量按顺序重放成硬约束
            for m in self.meas.get(channel, ()):
                self._apply_meas(reg, m)
            self.regions[channel] = reg
        return self.regions[channel]

    @staticmethod
    def _apply_meas(reg: DirProbRegion, m: Meas) -> None:
        """把一次测量转成硬约束（问题四：no_signal 不作为区域约束——可能是方向不对）。"""
        if m.outcome == "direction":
            reg.add_node(m.x, m.y, float(m.theta))
            reg.add_inside(m.x, m.y, RECEIVE_MAX)      # 收得到 ⇒ 源在接收半径上限之内
        elif m.outcome == "near":
            reg.add_inside(m.x, m.y, NEAR_RADIUS)      # 5 m 内 ⇒ 位置几乎确定

    def diameter(self, channel: int) -> float:
        """取频道 channel 定位区域的直径 / m

        Args:
            channel: 频道号

        Returns:
            float: 可能源集合的直径，越小表示定位越准
        """
        return float(self.region(channel).diameter)

    def _precise(self, channel: int) -> bool:
        """判据：可能源集合的最小覆盖圆半径 + 裁剪误差 < 20 m（= 清除半径）。

        只决定"还要不要继续补测"，不决定清不清除（清除一律先就近试一下）。
        """
        mec = self.region(channel).enclosing_circle
        return mec is not None and mec[2] + CLIP_ERR < CLEAR_RADIUS

    # ---- 阶段 1：扫描 ----
    def _est(self, channel: int) -> tuple[float, float] | None:
        """取频道 channel 的位置估计（区域最小覆盖圆圆心）

        Args:
            channel: 频道号

        Returns:
            tuple[float, float] | None: 估计点坐标；区域为空或退化时为 None
        """
        mec = self.region(channel).enclosing_circle
        return (mec[0], mec[1]) if mec is not None else None

    def _sweep_channels(self, at: Sequence[float]) -> list[int]:
        """该测量位置要测的频道：未清除 且（从未听到 或（顺路补测视角））。

        * 从未听到的频道：**必测**——扫描布局的作用就靠"每个测量位置都测所有未听到频道"；
        * 已听到但视角不足（< OBS_CAP）且估计还不够准、当前点又在其估计附近（≤ sweep_opp_
          radius）的频道：顺路补测，多一条不同角度的射线（对仅单侧可听的定向源尤其宝贵）。
        """
        out: list[int] = []
        for c in CHANNELS:
            if c in self.cleared:
                continue
            if len(self.obs.get(c, ())) == 0:
                out.append(c)
                continue
            if len(self.obs[c]) >= OBS_CAP or self._precise(c):
                continue
            est = self._est(c)
            if est is not None and dist(at, est) <= self.sweep_opp_radius:
                out.append(c)
        return out

    def _sweep_at(self, at: Sequence[float], label: str, index: int) -> dict[str, int]:
        """在 at 处对"该测的频道"按频道号升序测向（升序可省切换时间）；near 就地清除。"""
        channels = self._sweep_channels(at)
        self._begin_scan_step(index, label, at, len(channels))
        counts = {"direction": 0, "near": 0, "no_signal": 0, "skip": 0}
        for ch in sorted(channels):
            if self._out_of_time():
                break
            if self._provable_no_signal(ch, at):
                counts["skip"] += 1
                self.n_skip += 1
                continue
            res = self.measure(at[0], at[1], ch).get("measure_result", "no_signal")
            counts[res] = counts.get(res, 0) + 1
            if res == "near":
                ok = self.clear(at[0], at[1], ch)
                self.log(f"    [near] 频道{ch} 就在脚下，就地清除"
                         f"{'成功' if ok else '失败'}")
        self._end_scan_step(counts)
        return counts

    def sweep(self, plan: SweepPlan) -> None:
        """阶段一主体：原点全频道扫描 + 其余 28 个测量位置依次测向。"""
        self.plan = plan
        pts = plan.points
        self.log(f"── 阶段一：扫描（{plan.n_points} 个测量位置，里程 "
                 f"{plan.route_m:.0f} m）──")
        for k, i in enumerate(plan.route):
            at = (float(pts[i][0]), float(pts[i][1]))
            if k == 0:
                label = "起点全频道扫描（原点测量位置）"
            else:
                label = f"测量位置 {k}"
            counts = self._sweep_at(at, label, k)
            heard = sum(1 for c in self.obs if c not in self.cleared)
            skip_note = f"，跳过 {counts['skip']}" if counts["skip"] else ""
            n_meas = counts['direction'] + counts['near'] + counts['no_signal']
            self.log(f"  点{k} ({at[0]:.0f}, {at[1]:.0f})：测 {n_meas}"
                     f" 次（示向度 {counts['direction']}，无信号 {counts['no_signal']}"
                     f"{skip_note}），"
                     f"累计听到 {heard} 个频道")
            if k == 0:
                continue          # 起点段顺路清除已按问题三优化删除（2026-09-12 同步）
            # 到站后顺路清除（问题三优化机制）：
            #   1) 近距清：距本站 INLINE_NEAR_R 以内的估计点全清（不分方位，_nearby_clear）；
            #   2) 前向清：去下一站方向的方位扇区内、估计点距原点 [INLINE_R_MIN, INLINE_R_MAX]
            #      且区域小的频道（_inline_clear）。
            self._nearby_clear()
            if k + 1 < len(plan.route):
                nxt = plan.route[k + 1]
                self._inline_clear(at, (float(pts[nxt][0]), float(pts[nxt][1])))

    # ---- 阶段 2a：诊断 ----
    def diagnose(self) -> dict[str, int]:
        """阶段二开头的诊断：统计已听到频道及其中的"估计已够准"数，并初始化 tracks

        Returns:
            dict[str, int]: 已听到频道数、估计够准数与跳过测量次数
        """
        precise = 0
        for ch in sorted(self.obs):
            if ch in self.cleared:
                continue
            d = self.diameter(ch)
            mec = self.region(ch).enclosing_circle
            # 先把扫描阶段的统计落进 tracks，后续逐频道的记录在此之上累加
            self.tracks.setdefault(ch, {}).update({
                "n_obs_survey": len(self.obs[ch]),
                "first_heard_step": self.first_heard.get(ch),
                "diameter_survey_m": round(d, 3),
                "mec_radius_survey_m": round(mec[2], 3) if mec else None,
                "n_probe": 0,
            })
            precise += self._precise(ch)
        self.log(f"── 阶段二：定位与清除（扫描结束听到 {len(self.obs)} 个频道，"
                 f"其中估计已够准 {precise} 个）──")
        return {"n_channels": len(self.obs), "n_precise": precise, "n_skip": self.n_skip}

    # ---- 阶段 2b：访问顺序 ----
    def _nearest_order(self, channels: Sequence[int]) -> list[int]:
        """精确最短开放路径定序（Held-Karp；见 t3.strategy 同名列注释）。

        空输入直接返回空（r0 顺路清除后可能出现"扫描即全清、收尾无剩余"的局）。
        """
        if not channels:
            return []
        pts = self._clear_points(channels)
        idx = exact_open_order(len(channels), dist_matrix(pts, self.pos))
        return [channels[int(i)] for i in idx]

    # ---- 阶段 2c：补测 ----
    def refine(self, channel: int) -> int:
        """按文献准则补测缩小频道 channel 的定位区域，直到够准或没有候选

        Args:
            channel: 频道号

        Returns:
            int: 本次实际补测的次数
        """
        n_probe = 0
        for _ in range(REFINE_MAX):
            if self._precise(channel) or self._out_of_time():
                break
            hyps = hypothesis_points(self.region(channel), self.obs[channel])
            if not hyps:
                break
            cands = probe_candidates(self.obs[channel], hyps, self.pos)
            if not cands:
                break
            # 补测前的直径，仅用于日志对比
            d0 = self.diameter(channel)
            got = False
            for c in cands:
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

    # ---- 阶段 2d：清除 ----
    def _homing(self, channel: int) -> bool:
        """兜底：沿最新实测示向度以 HOMING_STEP 步长逼近，直到清除成功（确定性）。

        第四问修正：起点可能在波束背面（实测 no_signal）——此时先朝该频道**测量点质心**步进
        （质心必在迎光侧：全部方向观测都来自迎光侧），采到示向度后再正常逼近。半圆盘是凸集，
        从迎光侧沿向源方向步进始终留在波束内，故最终必能贴上。
        """
        obs = self.obs.get(channel, ())
        centroid = (np.mean([o.x for o in obs]), np.mean([o.y for o in obs])) if obs else None
        budget = HOMING_MAX
        verts = self.region(channel).vertices
        if verts:
            far = max(dist(self.pos, v) for v in verts)
            budget = int(min(HOMING_CAP, max(HOMING_MAX, far / HOMING_STEP + 4.0)))
        p = np.array(self.pos, dtype=float)
        blind = 0
        for _ in range(budget):
            if self._out_of_time():
                return False
            self.stage = "clear"
            r = self.measure(p[0], p[1], channel)
            res = r.get("measure_result")
            if res == "near":
                return self.clear(p[0], p[1], channel)
            if res == "direction":
                theta = math.radians(float(r["svd_deg"]))
                nxt = clamp_to_region(p[0] + HOMING_STEP * math.cos(theta),
                                      p[1] + HOMING_STEP * math.sin(theta))
                if self.clear(nxt[0], nxt[1], channel):
                    return True
                p = np.array(nxt, dtype=float)
                blind = 0
                continue
            # no_signal：可能在波束背面/太远。朝测量点质心回撤一步，重新找迎光侧。
            if centroid is None:
                return False
            blind += 1
            if blind > HOMING_CAP:
                return False
            nxt = clamp_to_region(p[0] + HOMING_STEP * (centroid[0] - p[0])
                                  / max(dist(p, centroid), 1e-9),
                                  p[1] + HOMING_STEP * (centroid[1] - p[1])
                                  / max(dist(p, centroid), 1e-9))
            if self.clear(nxt[0], nxt[1], channel):     # 回撤途中也可能直接贴上（靠得够近）
                return True
            p = np.array(nxt, dtype=float)
        return False

    def _side_scan(self, exclude: int) -> None:
        """清除动作时顺路补测：对所有"直径比较大"（定位区域未收敛）的未清频道测一次。

        定向源只能迎光侧测向，多视角交会定位更宝贵；在每次计划性清除尝试前，把当前
        仍"不确定"（最小覆盖圆直径 > SIDE_SCAN_DIAM）且示向度条数未满（< OBS_CAP）的
        频道各补测一条，用当前这个路过的位置换一个新视角，缩小它们的可能区域，让
        后续清除更省（减少文献补测 / 兜底逼近）。
        """
        if self._out_of_time():
            return
        for c in sorted(self.obs):
            if c == exclude or c in self.cleared:
                continue
            if len(self.obs[c]) >= OBS_CAP:
                continue
            if self.diameter(c) <= SIDE_SCAN_DIAM:
                continue
            if self._provable_no_signal(c, self.pos):
                self.n_side_skip_far += 1
                continue
            self.n_side_cand += 1
            self.stage = "clear"
            d0 = self.diameter(c)
            r = self.measure(self.pos[0], self.pos[1], c)
            res = r.get("measure_result", "no_signal")
            self.n_side_scan += 1
            if res == "direction":
                self.log(f"    [顺路补测] 频道{c} @ ({self.pos[0]:.1f}, {self.pos[1]:.1f})："
                         f"直径 {d0:.0f} → {self.diameter(c):.0f} m，新视角 +1")
            elif res == "near" and self.clear(self.pos[0], self.pos[1], c):
                self.log(f"    [顺路补测] 频道{c}：源就在 5 m 内，就地清除成功")
                self.tracks.setdefault(c, {})["method"] = "near@side"
            # no_signal 对定向源不构成硬约束（可能背光），无新增区域信息，正常跳过

    # ---- 途中顺路清除（沿用问题三优化后的机制与参数，见 t3.strategy，2026-09-12 同步）----
    def _in_azimuth_arc(self, at: Sequence[float], next_wp: Sequence[float],
                        est: Sequence[float]) -> tuple[float, float, float] | None:
        """估计点 est 是否落在「本站→圆心」与「下一站→圆心」两条连线之间的扇区内。

        判据（以区域圆心为参照的极角扇区，半径按**本站内外圈**分别限定）：
          · 方位：est 与圆心的连线方向 th_est 位于 th_at 与 th_next 夹出的**较短弧**上，即
                ang_diff(th_est, th_at) + ang_diff(th_est, th_next) == ang_diff(th_at, th_next)
            （三角不等式取等；ang_diff ∈ [0,180]，较短弧是唯一候选）；
          · 半径：本站为内圈（距原点 ≤ inline_outer_r）用 [INLINE_R_MIN_IN, INLINE_R_MAX_IN]；
            外圈用 [INLINE_R_MIN_OUT, ∞) —— **外圈不设上界**（源生成域 1770 m，r_est 恒 ≤1770，
            上界只在设到 <1770 时才起作用且必然更差，故无意义）。下界承接原"排除近圆心点"
            语义并入判据。
        反侧点（th_est 与 th_at 差 180°）由三角不等式排除；必须用角度而不用 sin（|sin 180°|=0
        会把反侧点误判成同向）。返回 (短弧进度 0~1, 距圆心半径, 与 th_at 的角度差)。
        """
        if math.hypot(at[0], at[1]) > self.inline_outer_r:   # 本站为外圈点
            r_min, r_max = self.inline_r_min_out, self.inline_r_max_out
        else:                                                # 本站为内圈点
            r_min, r_max = self.inline_r_min_in, self.inline_r_max_in
        r_est = math.hypot(est[0], est[1])
        if r_est < r_min - 1e-6:
            return None                              # 距原点太近，不在前向清的范围
        if r_est > r_max + 1e-6:
            return None                              # 超出半径上界
        th_at = self._polar_deg(at)
        th_next = self._polar_deg(next_wp)
        th_est = self._polar_deg(est)
        d = ang_diff(th_at, th_next)
        a1 = ang_diff(th_est, th_at)
        if d <= 1e-9:                            # 两端点同方位：退化为单方向
            if a1 > 1e-9:
                return None
            return (0.0, r_est, 0.0)
        if a1 + ang_diff(th_est, th_next) > d + 1e-9:
            return None
        return (a1 / d, r_est, a1)

    def _cover_clear(self, channel: int, why: str) -> int:
        """在"覆盖该频道定位区域的半径 20 m 圆"的圆心处清除（顺路清除的新清法）。

        用户 2026-09-12 指定（问题三同步）：在**区域的覆盖圆圆心**处清 —— 用半径
        `CLEAR_RADIUS`（20 m）的圆覆盖定位区域；区域用一个圆盖不住（最小覆盖圆半径 > 20 m）
        就按贪心补选多个半径 20 m 的圆（复用 _k_cover_points：第 1 个圆 = 区域最小覆盖圆的
        圆心），逐个在圆心处 /clear，**命中即停**；k_clear_max 个圆盖不满区域时只清第 1 个
        圆心（大区域多清命中率趋零，纯白跑）。圆心顺序按"距当前位置由近及远"（最近邻，命中
        概率最高的先试）。返回清除成功数（0 或 1：命中后频道即加入 cleared，不再试剩余圆）。
        """
        region = self.region(channel)
        mec = region.enclosing_circle
        if mec is None or region.region.is_empty:
            return 0
        center = (float(mec[0]), float(mec[1]))
        extra, leftover = self._k_cover_points(channel, self.k_clear_max)
        if leftover > 1e-9:
            extra = []                 # 盖不满区域：只清区域最小覆盖圆圆心一次，不白跑多个
        pts = [center] + list(extra)
        cur = np.array(self.pos, dtype=float)
        rest = list(pts)
        order: list[tuple[float, float]] = []
        while rest:                                   # 最近邻：从当前位置由近及远
            k = min(range(len(rest)),
                    key=lambda i: (float(np.linalg.norm(np.asarray(rest[i]) - cur)), i))
            p = rest.pop(k)
            order.append((float(p[0]), float(p[1])))
            cur = np.asarray(p, dtype=float)
        self.log(f"    [顺路清除] 频道{channel}（{why}）：{len(order)} 个半径 "
                 f"{CLEAR_RADIUS:.0f} m 覆盖圆圆心依次试清"
                 f"（区域最小覆盖圆半径 {mec[2]:.0f} m）")
        for i, (x, y) in enumerate(order, 1):
            if self._out_of_time():
                break
            self.stage = "survey-clear"
            if self.clear(x, y, channel):
                self.tracks.setdefault(channel, {}).update({
                    "method": "inline", "clear_point": [x, y],
                    "n_cover_clear": len(order)})
                self.n_inline += 1
                self.log(f"    [顺路清除] 频道{channel} @ ({x:.1f}, {y:.1f}) 命中"
                         f"（{why}，第 {i}/{len(order)} 个覆盖圆）")
                return 1
            self.n_inline_fail += 1
            self.log(f"    [顺路清除] 频道{channel} @ ({x:.1f}, {y:.1f}) 未命中"
                     f"（{why}，留到收尾阶段处理）")
        return 0

    def _nearby_clear(self, radius: float | None = None) -> int:
        """每站到站后的近距顺路清除：距当前点 radius（缺省 self.inline_near_r）以内的估计点全清。

        用户 2026-09-12 指定（问题三同步）：不分方位，只要估计点距当前点 ≤ 180 m 就清，清法
        同为"覆盖圆圆心处清、多个圆就清多次"（_cover_clear）。顺序按距离由近到远。只有
        **区域小**（最小覆盖圆半径 ≤ self.inline_max_mec_r）的频道才参与。返回本段清除数。
        """
        radius = self.inline_near_r if radius is None else float(radius)
        if not self.clear_enabled or self._out_of_time():
            return 0
        cur = np.array(self.pos, dtype=float)
        cand = []
        for ch in sorted(self.obs):
            if ch in self.cleared:
                continue
            mec = self.region(ch).enclosing_circle
            if mec is None or mec[2] > self.inline_max_mec_r:
                continue              # 区域为空/退化，或区域太大不参与顺路
            d = float(np.linalg.norm(np.asarray(mec[:2]) - cur))
            if d <= radius + 1e-6:
                cand.append((d, ch))
        if not cand:
            return 0
        cand.sort(key=lambda p: (p[0], p[1]))         # 由近到远
        n = 0
        for _d, ch in cand:
            if self._out_of_time():
                break
            n += self._cover_clear(ch, f"距本站 {_d:.0f} m ≤ {radius:.0f} m")
        if n:
            self.log(f"    本段近距清除 {n} 个，累计已清 {len(self.cleared)} 个")
        return n

    def _inline_clear(self, at: Sequence[float], next_wp: Sequence[float]) -> int:
        """途中**前向**顺路清除：把去下一站方向扇区内的估计点顺路清掉。

        判据（用户 2026-09-12 改，问题三同步）：以区域圆心（原点）为参照的方位扇区
        （_in_azimuth_arc）——估计点与圆心的连线方向落在「本站→圆心」与「下一站→圆心」两条
        连线之间（较短弧），且估计点距原点半径在 [INLINE_R_MIN, INLINE_R_MAX] 之间。不设
        "估计可信度"门槛，也没有起点段兜底（起点段顺路清除已删除，本站恒为测量位置、at 非
        原点）与独立的 600 m 排除规则（其语义由半径下界并入判据）。**只清区域小的频道**：
        定位区域最小覆盖圆半径 ≤ INLINE_MAX_MEC_R（75 m）才参与顺路，大区域命中率低、留给
        收尾阶段专程处理。

        成本：清点要走到估计点再回来——但这些源收尾阶段反正要清，顺路清掉省的是"从别处专程
        跑一趟"的里程；清点固定 5 s（命中）/ 3 s（未命中），命中后该频道后续测量点都不再测向。
        清法（_cover_clear）：在覆盖定位区域的半径 20 m 圆的圆心处清，区域大需多个圆则逐个
        清、命中即停。
        """
        if not self.clear_enabled or self._out_of_time():
            return 0
        picked: list[tuple[float, int, Sequence[float]]] = []
        n_too_big = 0                                  # 区域太大不参与顺路的频道数
        for ch in sorted(self.obs):
            if ch in self.cleared:
                continue
            mec = self.region(ch).enclosing_circle
            if mec is None:
                continue
            if mec[2] > self.inline_max_mec_r:
                n_too_big += 1
                continue
            est = (float(mec[0]), float(mec[1]))
            got = self._in_azimuth_arc(at, next_wp, est)
            if got is None:
                continue
            picked.append((got[0], ch, est))
        if not picked:
            return 0
        picked.sort(key=lambda p: (p[0], p[1]))  # 沿扇区方位由近到远依次清
        th_a = self._polar_deg(at)
        th_b = self._polar_deg(next_wp)
        if math.hypot(at[0], at[1]) > self.inline_outer_r:
            r_min, r_max = self.inline_r_min_out, self.inline_r_max_out
            ring = "外圈"
        else:
            r_min, r_max = self.inline_r_min_in, self.inline_r_max_in
            ring = "内圈"
        self.log(f"    [顺路清除] 本站 ({at[0]:.0f}, {at[1]:.0f}) → 下一站"
                 f" ({next_wp[0]:.0f}, {next_wp[1]:.0f})：方位扇区 {th_a:.0f}° ~ {th_b:.0f}°"
                 f"（{ring}）、半径 {r_min:.0f}~"
                 + ("∞（外圈不设上界）" if math.isinf(r_max) else f"{r_max:.0f}")
                 + f" m、区域覆盖圆半径 ≤ {self.inline_max_mec_r:.0f} m，有 {len(picked)} 个估计点"
                 + (f"（跳过 {n_too_big} 个区域过大的频道）" if n_too_big else ""))
        n = 0
        for _key, ch, _est in picked:
            if self._out_of_time():
                break
            n += self._cover_clear(ch, "前向扇区")
        if n:
            self.log(f"    本段顺路清除 {n} 个，累计已清 {len(self.cleared)} 个")
        return n

    def _try_clear(self, channel: int, tag: str) -> str | None:
        """走到区域最小覆盖圆圆心清除一次，并在清除点顺路补测其它大直径频道

        Args:
            channel: 频道号
            tag: 命中时写进 tracks 的方法标记

        Returns:
            str | None: 命中返回 tag；未命中返回 None
        """
        mec = self.region(channel).enclosing_circle
        if mec is None:
            return None
        cx, cy, r = mec
        hit = self.clear(cx, cy, channel)   # 到达清除点并尝试清除（成败都会更新 self.pos）
        # 每次清除动作后在清除点顺路补测"直径比较大"的其它频道（多视角缩小其定位区域）
        self._side_scan(channel)
        if hit:
            self.tracks.setdefault(channel, {})["clear_radius_m"] = round(r, 3)
            self.log(f"    [清除] 频道{channel} @ ({cx:.1f}, {cy:.1f}) 命中，"
                     f"最小覆盖圆半径 {r:.2f} m")
            return tag
        return None

    def _nearby_try_clear(self, channel: int) -> str | None:
        """就近试清：走到当前估计点直接 /clear 一次；未命中则就地复测。"""
        if not self.clear_enabled or self._out_of_time():
            return None
        mec = self.region(channel).enclosing_circle
        if mec is None or mec[2] > TRY_CLEAR_RADIUS:
            return None
        if not self.region(channel).bounded:
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
                     f"就地复测：直径 {rec.get('diameter_survey_m', float('nan'))} → "
                     f"{self.diameter(channel):.1f} m")
        elif res == "near" and self.clear(cx, cy, channel):
            return "near@center"
        return None

    def _k_cover_points(self, channel: int, k_extra: int,
                        radius: float = CLEAR_RADIUS) -> tuple[list[tuple[float, float]], float]:
        """贪心补选至多 k_extra 个半径 radius 的圆，覆盖频道 channel 的定位区域

        第 1 个圆固定是区域最小覆盖圆（由调用方使用），本函数只返回补选的圆心。

        Args:
            channel: 频道号
            k_extra: 最多补选的圆数
            radius: 每个覆盖圆的半径 / m

        Returns:
            tuple[list[tuple[float, float]], float]: 补选的圆心与未被覆盖的目标点占比
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
        if len(cand) > 2500:
            # 候选抽稀：给贪心每轮的成对距离矩阵限规模
            cand = cand[:: len(cand) // 2500 + 1]
        covered = np.linalg.norm(targets - np.asarray(mec[:2]), axis=1) <= radius + TOL
        points: list[tuple[float, float]] = []
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

    def _multi_try_clear(self, channel: int) -> str | None:
        """多清几次：区域能被 k_clear_max 个 20 m 圆盖满时，在这些圆心依次补清（就近优先）

        Args:
            channel: 频道号

        Returns:
            str | None: 命中返回 "multi"；盖不满区域或全部未命中返回 None
        """
        if not self.clear_enabled or self._out_of_time():
            return None
        pts, leftover = self._k_cover_points(channel, self.k_clear_max)
        if not pts or leftover > 1e-9:
            return None
        rec = self.tracks.setdefault(channel, {})
        # +1 是区域最小覆盖圆本身（第 1 个清点）
        rec["k_clear"] = len(pts) + 1
        rec["k_cover_leftover"] = round(leftover, 6)
        self.log(f"    [多清几次] 频道{channel}：{len(pts) + 1} 个半径 {CLEAR_RADIUS:.0f} m 的圆"
                 f"可覆盖整个定位区域，依次补清")
        cur = np.array(self.pos, dtype=float)
        rest = list(pts)
        while rest:
            k = min(range(len(rest)),
                    key=lambda i: (float(np.linalg.norm(np.asarray(rest[i]) - cur)), i))
            x, y = rest.pop(k)
            if self.clear(x, y, channel):
                self.log(f"    [多清几次] 频道{channel} @ ({x:.1f}, {y:.1f}) 命中")
                return "multi"
            cur = np.array([x, y], dtype=float)
        self.log(f"    [多清几次] 频道{channel}：{len(pts) + 1} 个点都未命中，转入补测")
        return None

    def process(self, channel: int) -> None:
        """四级清除频道 channel（就近试清 → 多清几次 → 补测 → 兜底逼近）并落 tracks

        Args:
            channel: 频道号
        """
        rec = self.tracks.setdefault(channel, {})
        if channel in self.cleared:
            rec.update({"method": "survey-near", "cleared": True})
            return
        if not self.clear_enabled:
            if not self._precise(channel):
                self.refine(channel)
            method: str | None = "skipped"
        else:
            method = self._nearby_try_clear(channel)
            if method is None:
                method = self._multi_try_clear(channel)
            if method is None:
                self.refine(channel)
                method = self._finish_clear(channel)
        # 收尾统一记录该频道最终的定位精度与清除方式
        d = self.diameter(channel)
        mec = self.region(channel).enclosing_circle
        rec.update({
            "diameter_final_m": round(d, 3),
            "precise_final": self._precise(channel),
            "mec_radius_m": round(mec[2], 3) if mec else None,
            "n_obs_total": len(self.obs[channel]),
            "method": method,
            "cleared": channel in self.cleared,
        })

    # ---- 主流程 ----
    def run(self, plan: SweepPlan) -> dict[str, Any]:
        """/enter → 扫描 → 诊断 → 逐频道定位清除 → /exit。"""
        enter = self.sim.enter()
        if not enter.get("accepted"):
            raise RuntimeError(f"/enter 被拒绝：{enter}")
        left = float(enter.get("remaining_real_duration_s", 1200.0))
        self.deadline = time.monotonic() + max(left - SAFETY_MARGIN, 0.0)
        self.log(f"/enter 成功：虚拟时刻 {enter.get('virtual_time_s')} s，现实剩余 {left:.0f} s")
        try:
            self.sweep(plan)
            if self.clear_enabled:
                self.diagnose()
                for ch in self._nearest_order([c for c in sorted(self.obs)
                                               if c not in self.cleared]):
                    if self._out_of_time():
                        self.log("    [超时] 中断处理剩余频道")
                        break
                    self.log(f"── 频道 {ch} ──")
                    self.process(ch)
        finally:
            try:
                self.sim.exit()
            except OSError as exc:
                self.log(f"    [警告] /exit 失败：{exc}")
            self.close()
        n = len(self.cleared)
        return {
            "travel_m": round(self.travel_m, 2),
            "virtual_time_s": round(self.vt, 3),
            "n_measure": self.n_measure,
            "n_clear": self.n_clear,
            "channels_heard": len(self.obs),
            "n_bearings": sum(len(v) for v in self.obs.values()),
            "cleared": n,
            "all_cleared": n == sum(1 for v in self.obs.values() if v),
            "avg_time_s": round(self.vt / n, 2) if n else float("inf"),
            "n_skip_measure": self.n_skip,
            "first_heard": dict(self.first_heard),
            "n_refined": sum(1 for r in self.tracks.values() if r.get("n_probe")),
            "n_probe": sum(int(r.get("n_probe", 0)) for r in self.tracks.values()),
            "n_side_scan": self.n_side_scan,
            "n_side_cand": self.n_side_cand,
            "n_side_skip_far": self.n_side_skip_far,
            "n_inline": self.n_inline,
            "n_inline_fail": self.n_inline_fail,
            "methods": {m: sum(1 for r in self.tracks.values() if r.get("method") == m)
                        for m in ("try", "try-refined", "multi", "near", "homing",
                                  "survey-near", "near@center", "near@probe",
                                  "near@side", "inline", "failed")},
            "tracks": self.tracks,
        }