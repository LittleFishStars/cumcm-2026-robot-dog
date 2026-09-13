"""问题四的机器狗策略：先扫描，再定位清除"""

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
    """两阶段机器狗，先扫描再逐频道定位清除"""

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
        """构造两阶段机器狗；给了 api_log 就在模拟器外面再包一层记录代理"""
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
        """在坐标 (x, y) 处测频道 channel，结果一并进定位区域、观测记录和首次听到记录"""
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
        """能不能证明"在 at 处测 channel 必然没有信号"，能证明就省掉这次测量"""
        reg = self.regions.get(channel)
        return reg is not None and reg.provably_out_of_reach(at, RECEIVE_MAX)

    def _end_scan_step(self, counts: dict[str, int]) -> None:
        """收尾当前扫描步骤：补齐统计、路径、区域与估计点，然后追加进 scan_steps"""
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
        """取频道 channel 的定位区域，头一次访问时按已有测量惰性重建"""
        if channel not in self.regions:
            reg = DirProbRegion(err=BEARING_ERROR_DEG, radius=REGION_RADIUS, sides=CLIP_SIDES)
            # 首次建区域：把该频道此前的测量按顺序重放成硬约束
            for m in self.meas.get(channel, ()):
                self._apply_meas(reg, m)
            self.regions[channel] = reg
        return self.regions[channel]

    @staticmethod
    def _apply_meas(reg: DirProbRegion, m: Meas) -> None:
        """把一次测量转成硬约束，no_signal 不管"""
        if m.outcome == "direction":
            # 定向源让 no_signal 转不成硬约束，它可能只是方向不对，所以这里没有对应分支
            reg.add_node(m.x, m.y, float(m.theta))
            reg.add_inside(m.x, m.y, RECEIVE_MAX)      # 收得到，所以源在接收半径上限之内
        elif m.outcome == "near":
            reg.add_inside(m.x, m.y, NEAR_RADIUS)      # 落在 5 m 内，位置几乎确定

    def diameter(self, channel: int) -> float:
        """取频道 channel 定位区域的直径，单位米，越小说明定位越准"""
        return float(self.region(channel).diameter)

    def _precise(self, channel: int) -> bool:
        """判据：可能源集合的最小覆盖圆半径加上裁剪误差还不到 20 m，也就是清除半径"""
        # 它只管"还要不要继续补测"，不管清不清除，清除一律先就近试一下
        mec = self.region(channel).enclosing_circle
        return mec is not None and mec[2] + CLIP_ERR < CLEAR_RADIUS

    # ---- 阶段一：扫描 ----
    def _est(self, channel: int) -> tuple[float, float] | None:
        """取频道 channel 的位置估计，就是区域最小覆盖圆的圆心，区域为空或退化时给 None"""
        mec = self.region(channel).enclosing_circle
        return (mec[0], mec[1]) if mec is not None else None

    def _sweep_channels(self, at: Sequence[float]) -> list[int]:
        """该测量位置要测哪些频道：没清除的，而且要么从未听到，要么值得顺路补一个视角"""
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
        """阶段一的主体：先在原点全频道扫描，其余 19 个测量位置依次测向"""
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

    # ---- 阶段二 a：诊断 ----
    def diagnose(self) -> dict[str, int]:
        """阶段二开头先诊断：数一数已听到多少频道、其中几个估计已经够准，顺带初始化 tracks"""
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

    # ---- 阶段二 b：访问顺序 ----
    def _nearest_order(self, channels: Sequence[int]) -> list[int]:
        """精确最短开放路径定序，用 Held-Karp，算法本身见 t3.strategy 同名函数的注释"""
        # 空输入要给空：r0 顺路清除之后真会出现"扫描时全清、收尾没剩下什么"的局
        if not channels:
            return []
        pts = self._clear_points(channels)
        idx = exact_open_order(len(channels), dist_matrix(pts, self.pos))
        return [channels[int(i)] for i in idx]

    # ---- 阶段二 c：补测 ----
    def refine(self, channel: int) -> int:
        """按文献准则补测缩小频道 channel 的定位区域，直到够准或没有候选，返回实际补测次数"""
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

    # ---- 阶段二 d：清除 ----
    def _homing(self, channel: int) -> bool:
        """兜底：沿最新实测示向度按 HOMING_STEP 的步长逼近，直到清掉，过程是确定性的"""
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
            # 质心必在迎光侧，方向观测全来自迎光侧；半圆盘是凸集，从迎光侧沿向源方向步进
            # 始终留在波束内，最后一定贴得上
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
        """清除动作发生时顺路补测：把直径还没收敛的未清频道在当前位置补一个视角"""
        # 定向源只有迎光侧测得出方向，多视角交会格外值钱，顺手换来的视角能省掉后面的文献补测
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
        以区域圆心为参照看极角扇区，返回短弧进度、距圆心半径与到 th_at 的角差
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
        # 必须比角度，拿 sin 比会把反侧点误判成同向，|sin 180°| 等于 0
        a1 = ang_diff(th_est, th_at)
        if d <= 1e-9:                            # 两端点同方位：退化为单方向
            if a1 > 1e-9:
                return None
            return (0.0, r_est, 0.0)
        if a1 + ang_diff(th_est, th_next) > d + 1e-9:
            return None
        return (a1 / d, r_est, a1)

    def _cover_clear(self, channel: int, why: str) -> int:
        """在覆盖该频道定位区域的半径 20 m 圆的圆心处顺路清除，命中就停"""
        region = self.region(channel)
        mec = region.enclosing_circle
        if mec is None or region.region.is_empty:
            return 0
        center = (float(mec[0]), float(mec[1]))
        # 一个半径 CLEAR_RADIUS 的圆盖不满区域时用 _k_cover_points 贪心补选，它的第 1 个圆就是
        # 此处的最小覆盖圆圆心；圆心按距当前位置由近及远试，命中率最高的先试
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
        """每站到站后的近距顺路清除，不分方位，只清区域不大的频道，返回本段清掉几个"""
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
        """从当前站走到下一站，途中的前向顺路清除，只清区域小的那些频道"""
        # 以区域圆心、也就是原点为参照看方位扇区，判据交给 _in_azimuth_arc；本站恒为测量位置，
        # 起点段那一段顺路清除已删，原来独立的 600 m 排除规则也撤了，语义并进半径下界
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
                # 大区域清中的概率太低，不参与顺路，留给收尾阶段专程处理
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
        """走到区域最小覆盖圆圆心清一次，顺便在清除点补测其它大直径频道"""
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
        """贪心补选至多 k_extra 个半径 radius 的圆盖住定位区域，返回补选的圆心与未覆盖占比"""
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
        """多清几次：区域能被 k_clear_max 个 20 m 圆盖满时，就在这些圆心依次补清，就近优先"""
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
        """对频道 channel 走四级清除：就近试清、多清几次、补测、兜底逼近，最后落进 tracks"""
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
