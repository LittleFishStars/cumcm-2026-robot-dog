"""机器狗策略，问题四：先扫描，再按定向感知定位清除

其余手法都跟问题三同源复用，就近试清、K 圆覆盖、Fisher 准则补测、沿示向度逼近的细节见 t3.strategy
的模块文档，下面只写不一样的地方。

阶段一复用问题三的 7 个覆盖基点，外圈再扩一圈，按 t4.sweep 给的方案依次走 20 个测量点，也就是原点、
7 个覆盖基点、12 个均匀方位的 1850 m 外圈点。每个点都对"还没听到过"的频道测向，已经听到但估计还
不够准、且当前点就在其估计附近的频道顺路补一两个视角。布局效果按实测报：听到率 99.9891%，
4 224 640 个算例漏 460，20 局演练 256/256 全清，绝大多数干扰源不管全向还是定向，扫描没结束就被
听到了，这就是"确保所有干扰源被清除"的检测侧依据。

no_signal 不再是"源在接收半径之外"这种硬约束，因为方向可能不对，见 DirProbRegion 与 config，定位
区域只用 direction 和 near 两类约束；"跳过必无信号的测量"这一条留着，区域整体落在 1500 m 外才算
可证，但适用范围比问题三小得多。

homing 兜底针对"起始点可能恰好落在波束背面"补了反方向修正：起点实测没信号就朝该频道的测量点质心
步进，质心必在迎光侧，等重新采到示向度再正常逼近，半圆盘是凸集，从迎光侧朝源逼近一路都留在波束内，
最后总能贴到 20 m 内清掉。

阶段二对"扫描结束时已听到的频道"一个不落，逐频道四级清除。
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
    """两阶段机器狗，这是问题四的实现

    阶段一扫描：先在出发点把全频道扫一遍，那是原点测量位置，然后按扫描方案依次走到其余测量点，每点对
    "还没听到过"的频道测向，近距的就地清掉，保证任何源至少被听到 1 次，估计附近的频道顺路补第二、
    第三视角；站间移动时按问题三的机制做 _inline_clear，方位扇区里那些未清频道的估计点顺路试清，扫描
    结束大约能清掉 60% 的源。

    阶段二定位与清除：扫描结束时已听到的频道逐个走四级清除，每次清除动作之后还要顺路补测所有"直径偏大、
    超过 200 m，而且观测没满"的未清频道，靠多视角把定位区域压小，见 _side_scan。

      1. 就近试清：走到定位区域最小覆盖圆圆心直接 /clear 一次，没命中就就地复测；
      2. 多清几次：区域用 K 个半径 20 m 的圆盖满，逐个圆心依次补清；
      3. 按文献准则补测把区域缩小之后再清，单射线局面也走这条；
      4. 兜底沿最新示向度以 16 m 步长逼近；起点在波束背面时先朝测量点质心回撤，再逼近。
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
        """构造两阶段机器狗

        Args:
            sim: 模拟器接口，就是那 4 条指令的薄封装；给了 api_log 会再包一层记录代理
            verbose: 要不要实时回显过程日志
            logfile: 过程日志写到哪个文件，None 表示只回显不落盘
            episode: 局号，会写进日志和 request_id，方便跟模拟器行为日志对着看
            clear: 是否启用清除，False 只扫描定位，对应 --survey-only
            k_clear_max: 覆盖定位区域的 20 m 圆最多补选几个
            sweep_opp_radius: 顺路补测半径 / 米
            inline_r_min: 前向顺路半径下界 / 米，内外圈的缺省值都从它来
            inline_r_max: 前向顺路半径上界 / 米
            inline_r_min_in: 内圈前向半径下界 / 米，None 就跟 inline_r_min
            inline_r_max_in: 内圈前向半径上界 / 米，None 就跟 inline_r_max
            inline_r_min_out: 外圈前向半径下界 / 米，None 就跟 inline_r_min
            inline_r_max_out: 外圈前向半径上界 / 米，None 取 +inf，也就是外圈不设上界
            inline_near_r: 近距顺路清除半径 / 米
            inline_max_mec_r: 肯参与顺路的区域最大 mec 半径 / 米
            inline_outer_r: 内外圈分界，本站距原点超过它按外圈处理
            api_log: 接口调用日志，None 表示不记录
        """
        self.sim = sim if api_log is None else RecordedSim(sim, api_log, episode)
        self.verbose = verbose
        self.clear_enabled = bool(clear)
        self.k_clear_max = int(k_clear_max)
        self.sweep_opp_radius = float(sweep_opp_radius)   # 顺路补测半径 / 米
        self.inline_r_min = float(inline_r_min)           # 前向顺路半径下界 / 米，内外组缺省源自它
        self.inline_r_max = float(inline_r_max)           # 前向顺路半径上界 / 米
        # 内外圈的前向半径，None 时内圈跟随全局 inline_r_min/inline_r_max。外圈不设半径上界：
        # 源生成域半径 1770 m 而 r_est 恒 ≤ 1770，上界要设到 1770 以下才有用，实测只会更差，
        # 1533 对应 6727 s、1689 对应 6430 s，1767 以上全是 6363 s 的平台。
        self.inline_r_min_in = float(inline_r_min if inline_r_min_in is None else inline_r_min_in)
        self.inline_r_max_in = float(inline_r_max if inline_r_max_in is None else inline_r_max_in)
        self.inline_r_min_out = float(inline_r_min if inline_r_min_out is None else inline_r_min_out)
        self.inline_r_max_out = math.inf if inline_r_max_out is None else float(inline_r_max_out)
        self.inline_outer_r = float(inline_outer_r)       # 内外圈分界：本站距原点 > 该值按外圈
        self.inline_near_r = float(inline_near_r)         # 近距顺路清除半径 / 米
        self.inline_max_mec_r = float(inline_max_mec_r)   # 参与顺路的区域最大 mec 半径 / 米
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
        self.n_inline = 0            # 途中顺路清除命中的次数，沿用问题三那套机制
        self.n_inline_fail = 0       # 途中顺路清除没命中的次数
        self.n_side_cand = 0        # 顺路补测候选数，要求直径大、视角未满、距离够近
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

    # ---- 日志与原子动作见 common.actions：log / close / clear / _note / _polar_deg 都在那里 ----
    def measure(self, x: float, y: float, channel: int) -> dict:
        """在坐标 (x, y) 处测频道 channel，结果一并进定位区域和观测记录

        返回模拟器原始响应，含 measure_result / svd_deg / virtual_time_s，被拒绝了就抛 RuntimeError。
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
        """能不能证明"在 at 处测 channel 必然没有信号"，能证明就省掉这次测量

        可能源集合是真实源位置的超集。区域到 at 的最小距离一旦超过 1500 m，真源到 at 的距离
        也就超过 1500 m，而 1500 m 已经顶到有效接收半径，所以不管全向还是定向都必然收不到。
        这次测量带不来新信息，跳过。
        """
        reg = self.regions.get(channel)
        return reg is not None and reg.provably_out_of_reach(at, RECEIVE_MAX)

    def _end_scan_step(self, counts: dict[str, int]) -> None:
        """收尾当前扫描步骤：补齐统计、路径、区域与估计点，然后追加进 scan_steps

        counts 是本站各测量结果的计数，键为 direction / near / no_signal / skip。
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
        """取频道 channel 的定位区域，头一次访问时按已有测量惰性重建
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
        """把一次测量转成硬约束。问题四里 no_signal 不当约束，它可能只是方向不对"""
        if m.outcome == "direction":
            reg.add_node(m.x, m.y, float(m.theta))
            reg.add_inside(m.x, m.y, RECEIVE_MAX)      # 收得到，所以源在接收半径上限之内
        elif m.outcome == "near":
            reg.add_inside(m.x, m.y, NEAR_RADIUS)      # 落在 5 m 内，位置几乎确定

    def diameter(self, channel: int) -> float:
        """取频道 channel 定位区域的直径，单位米，越小说明定位越准
        """
        return float(self.region(channel).diameter)

    def _precise(self, channel: int) -> bool:
        """判据：可能源集合的最小覆盖圆半径加上裁剪误差还不到 20 m，也就是清除半径

        它只管"还要不要继续补测"，不管清不清除。清除一律先就近试一下。
        """
        mec = self.region(channel).enclosing_circle
        return mec is not None and mec[2] + CLIP_ERR < CLEAR_RADIUS

    # ---- 阶段一：扫描 ----
    def _est(self, channel: int) -> tuple[float, float] | None:
        """取频道 channel 的位置估计，就是区域最小覆盖圆的圆心，区域为空或退化时给 None
        """
        mec = self.region(channel).enclosing_circle
        return (mec[0], mec[1]) if mec is not None else None

    def _sweep_channels(self, at: Sequence[float]) -> list[int]:
        """该测量位置要测哪些频道：没清除的，而且要么从未听到，要么值得顺路补一个视角

        从未听到的频道必须测，扫描布局能起作用靠的就是"每个测量位置都把没听到的频道测一遍"；已经听到但视角还
        没满、少于 OBS_CAP，估计也还不够准，当前点又落在其估计点 sweep_opp_radius 以内的频道，顺路补一条不同
        角度的射线，对只能单侧听到的定向源尤其金贵。
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
        """在 at 处把该测的频道按频道号升序测一遍，升序省切换时间；碰上 near 就地清除"""
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
        """阶段一主体：先在原点全频道扫描，其余 19 个测量位置依次测向"""
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
                continue          # 起点段的顺路清除已按问题三的优化删掉，2026-09-12 同步
            # 到站后顺路清除，沿用问题三的两档：_nearby_clear 清距本站 INLINE_NEAR_R 以内
            # 不分方位的估计点，_inline_clear 清去下一站方位扇区里、估计点距原点落在
            # [INLINE_R_MIN, INLINE_R_MAX] 且区域够小的频道。
            self._nearby_clear()
            if k + 1 < len(plan.route):
                nxt = plan.route[k + 1]
                self._inline_clear(at, (float(pts[nxt][0]), float(pts[nxt][1])))

    # ---- 阶段 2a：诊断 ----
    def diagnose(self) -> dict[str, int]:
        """阶段二开头先诊断：数一数已听到多少频道、其中几个估计已经够准，顺带初始化 tracks

        返回已听到频道数、估计够准的个数与跳过测量的次数。
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
        """精确最短开放路径定序，用 Held-Karp，算法本身见 t3.strategy 同名函数的注释

        输入为空就直接返回空：r0 顺路清除之后真会出现"扫描即全清、收尾没剩下什么"的局。
        """
        if not channels:
            return []
        pts = self._clear_points(channels)
        idx = exact_open_order(len(channels), dist_matrix(pts, self.pos))
        return [channels[int(i)] for i in idx]

    # ---- 阶段 2c：补测 ----
    def refine(self, channel: int) -> int:
        """按文献准则补测缩小频道 channel 的定位区域，直到够准或没有候选，返回实际补测次数
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
            # 补测前的直径，只留给日志做对比
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
        """兜底：沿最新实测示向度按 HOMING_STEP 的步长逼近，直到清掉，过程是确定性的

        问题四补的修正是起点有可能恰好落在波束背面，实测就是 no_signal，这时先朝该频道的测量点质心步进（质心
        必在迎光侧，方向观测全来自迎光侧），采到示向度后再照常逼近；半圆盘是凸集，从迎光侧沿向源方向步进始终
        留在波束内，最后一定贴得上。
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
            # 无信号：可能站在波束背面，也可能太远。朝测量点质心回撤一步，重新找迎光侧。
            if centroid is None:
                return False
            blind += 1
            if blind > HOMING_CAP:
                return False
            nxt = clamp_to_region(p[0] + HOMING_STEP * (centroid[0] - p[0])
                                  / max(dist(p, centroid), 1e-9),
                                  p[1] + HOMING_STEP * (centroid[1] - p[1])
                                  / max(dist(p, centroid), 1e-9))
            if self.clear(nxt[0], nxt[1], channel):     # 回撤路上也可能直接贴上，距离够近就行
                return True
            p = np.array(nxt, dtype=float)
        return False

    def _side_scan(self, exclude: int) -> None:
        """清除动作发生时顺路补测：把所有"直径比较大"、也就是定位区域还没收敛的未清频道测一次

        定向源只有迎光侧测得出方向，多视角交会就格外值钱：每次计划性清除尝试之前，把最小覆盖圆直径超过
        SIDE_SCAN_DIAM 且示向度条数还没满、少于 OBS_CAP 的频道各补测一条，等于拿这个正好路过的位置换一个新视
        角，后面清除就能少走文献补测和兜底逼近。
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
            # no_signal 对定向源算不上硬约束，可能就是背光，区域里添不了新信息，照常跳过

    # ---- 途中顺路清除，沿用问题三调好的机制与参数，见 t3.strategy，2026-09-12 同步 ----
    def _in_azimuth_arc(self, at: Sequence[float], next_wp: Sequence[float],
                        est: Sequence[float]) -> tuple[float, float, float] | None:
        """估计点 est 落没落在「本站→圆心」和「下一站→圆心」两条连线夹出的扇区里

        判据以区域圆心为参照看极角扇区：方位上要求 est 与圆心的连线方向 th_est 落在 th_at 和 th_next 夹出的
        较短弧上，也就是 ang_diff(th_est, th_at) + ang_diff(th_est, th_next) == ang_diff(th_at, th_next)，这
        是三角不等式取等，而 ang_diff 值域 [0,180] 让较短弧成为唯一候选；半径上本站算内圈，距原点不超过
        inline_outer_r，就用 [INLINE_R_MIN_IN, INLINE_R_MAX_IN]，算外圈就用 [INLINE_R_MIN_OUT, ∞)，外圈不设
        上界是因为源生成域 1770 m 而 r_est 恒不超过 1770，上界设到 1770 以下才起作用且必然更差，原来的"排除
        近圆心点"语义则由下界接着。

        反侧点由三角不等式自己排除，也就是 th_est 与 th_at 差 180° 的那种；这里必须比角度而不能拿 sin
        比，因为 |sin 180°| 等于 0 会把反侧点误判成同向。返回值是短弧上的进度 0~1、距圆心的半径、与
        th_at 的角度差这三项。
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
        """在覆盖该频道定位区域的那个半径 20 m 圆的圆心处清除，这是顺路清除的新清法

        2026-09-12 用户指定、问题三同步：清点取区域的覆盖圆圆心，拿半径 `CLEAR_RADIUS` 的圆去盖定位区域，一
        个圆盖不住、最小覆盖圆半径超过 20 m 的话，就按贪心补选几个半径 20 m 的圆，走 _k_cover_points，它的第
        1 个圆就是区域最小覆盖圆的圆心；然后逐个圆心 /clear，命中就停，k_clear_max 个圆还盖不满区域时只清第
        1 个圆心，因为大区域多清几次命中率趋近于零，纯属白跑。圆心顺序按距当前位置由近及远排，最近邻，命中概
        率最高的先试。清除成功数只可能是 0 或 1，命中后频道就进 cleared，剩下的圆不再试。
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
        """每站到站后的近距顺路清除：把距当前点 radius 以内的估计点全清掉

        2026-09-12 用户指定、问题三同步：这里不分方位，估计点距当前点只要不超过 180 m 就清，清法仍是
        _cover_clear 那一套，在覆盖圆圆心处清、多个圆就清多次，顺序按距离由近到远，只有最小覆盖圆半径不超过
        self.inline_max_mec_r 的小区域频道才参与。返回本段清掉几个。
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
                continue              # 区域为空或退化，或者区域太大，不参与顺路
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
        """途中的前向顺路清除：去下一站方向上，扇区里的估计点顺手清掉

        判据是 2026-09-12 用户改的、问题三同步：以区域圆心，也就是原点，为参照看方位扇区，判断交给
        _in_azimuth_arc，要求估计点与圆心的连线方向落在「本站→圆心」和「下一站→圆心」夹出的较短弧上，且估计
        点距原点的半径落在 [INLINE_R_MIN, INLINE_R_MAX] 之间；这里没有"估计可信度"门槛，也没有起点段兜底，起
        点段顺路清除已删，本站恒为测量位置、at 不会是原点，原来那条独立的 600 m 排除规则也撤了，语义并进半径
        下界。

        只清区域小的频道，也就是最小覆盖圆半径不超过 INLINE_MAX_MEC_R、即 75 m 的频道，因为大区域清中的概率
        太低，留给收尾阶段专程处理。成本上清点要走到估计点再走回来，但这些源收尾阶段反正也要清，顺路清掉省下
        的是"从别处专程跑一趟"的里程；清点固定 5 s 命中、3 s 未命中，命中之后该频道在后续测量点都不再测向，
        清法仍是 _cover_clear。
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
        """走到区域最小覆盖圆圆心清一次，顺便在清除点补测其它大直径频道

        tag 是命中时写进 tracks 的方法标记，命中就把它原样返回，没命中返回 None。
        """
        mec = self.region(channel).enclosing_circle
        if mec is None:
            return None
        cx, cy, r = mec
        hit = self.clear(cx, cy, channel)   # 走到清除点试着清一次，成不成都会更新 self.pos
        # 每次清除动作之后，在清除点顺路补测"直径比较大"的其它频道，多视角缩小它们的定位区域
        self._side_scan(channel)
        if hit:
            self.tracks.setdefault(channel, {})["clear_radius_m"] = round(r, 3)
            self.log(f"    [清除] 频道{channel} @ ({cx:.1f}, {cy:.1f}) 命中，"
                     f"最小覆盖圆半径 {r:.2f} m")
            return tag
        return None

    def _nearby_try_clear(self, channel: int) -> str | None:
        """就近试清：走到当前估计点直接 /clear 一次，没命中就就地复测"""
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
        """贪心补选至多 k_extra 个半径 radius 的圆，盖住频道 channel 的定位区域

        第 1 个圆固定是区域最小覆盖圆，那个由调用方自己用，本函数只返回补选出来的圆心，以及没被盖住的目标点
        占比。
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
        """多清几次：区域能被 k_clear_max 个 20 m 圆盖满时，就在这些圆心依次补清，就近优先

        命中返回 "multi"，盖不满区域或者全都没命中则返回 None。
        """
        if not self.clear_enabled or self._out_of_time():
            return None
        pts, leftover = self._k_cover_points(channel, self.k_clear_max)
        if not pts or leftover > 1e-9:
            return None
        rec = self.tracks.setdefault(channel, {})
        # +1 是区域最小覆盖圆，也就是第 1 个清点
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
        """对频道 channel 走四级清除：就近试清、多清几次、补测、兜底逼近，最后落进 tracks
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
        # 收尾统一记下该频道最终的定位精度和清除方式
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
        """/enter，扫描，诊断，然后逐频道定位清除，最后 /exit"""
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