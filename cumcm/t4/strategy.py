"""机器狗策略（问题四）：扫描（阶段一）+ 定向感知定位清除（阶段二）。

与问题三的差别（其余手法——就近试清、K 圆覆盖、Fisher 准则补测、沿示向度逼近——同源复用，
见 cumcm.t3.strategy 的模块文档）：

* 阶段一**复用问题三的 7 个覆盖基点并加密**：按 cumcm.t4.sweep 的方案依次走到 29 个
  测量点（原点 + 覆盖到半径 2270 m 的 700 m 格点），在每个点对"还没听到过"的频道测向，另对
  已听到但估计还不够准、且当前点就在其估计附近的频道顺路补一两个视角。**布局效果**：实测
  听到率 ~99.99%（400 万随机算例），故绝大多数干扰源（全向或定向）在扫描结束前被听到——
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cumcm.common.geometry import clamp_to_region as _clamp_to_region
from cumcm.common.geometry import dist
from cumcm.common.routing import dist_matrix, exact_open_order
from cumcm.common.sim_client import RecordedSim
from cumcm.t4.config import (BEARING_ERROR_DEG, CHANNELS, CLEAR_RADIUS, CLIP_ERR,
                             CLIP_SIDES, HOMING_CAP, HOMING_MAX, HOMING_STEP,
                             K_CLEAR_MAX, K_COVER_SAMPLES, K_COVER_STEP_MIN, NEAR_RADIUS,
                             OBS_CAP, RECEIVE_MAX, REFINE_MAX, REGION_MARGIN, REGION_RADIUS, SAFETY_MARGIN,
                             SIDE_SCAN_DIAM, SWEEP_OPP_RADIUS, TOL, TRY_CLEAR_RADIUS)
from cumcm.t4.probing import hypothesis_points, probe_candidates
from cumcm.t4.regions import DirProbRegion, Meas, Obs
from cumcm.t4.sweep import SweepPlan

clamp_to_region = partial(_clamp_to_region, radius=REGION_RADIUS - REGION_MARGIN)


class RobotDog:
    """两阶段机器狗（问题四）。

    阶段一（扫描）：在出发点做全频道扫描（原点测量位置），随后按扫描方案依次走到其余
    测量点；每点对"未听到过的频道"测向（近距就地清除），保证任何源都被听到 ≥ 1 次；
    顺路对估计附近的频道补测第二/第三视角。

    阶段二（定位与清除）：对扫描结束时已听到的每个频道，四级清除；**每次清除动作后
    顺路补测**所有"直径比较大（> 200 m）且观测未满"的未清频道（多视角缩小定位区域，
    见 _side_scan）：
      1. 就近试清（走到定位区域最小覆盖圆圆心直接 /clear 一次，未命中就地复测）；
      2. 多清几次（K 个半径 20 m 的圆盖满区域外，依次补清）；
      3. 按文献准则补测缩小区域后再清（含单射线局面）；
      4. 兜底沿最新示向度以 16 m 步长逼近；起点在波束背面时先朝测量点质心回撤再逼近。
    """

    def __init__(self, sim, verbose: bool = True, logfile: Optional[str] = None,
                 episode: int = 0, clear: bool = True,
                 k_clear_max: int = K_CLEAR_MAX,
                 sweep_opp_radius: float = SWEEP_OPP_RADIUS,
                 api_log=None) -> None:
        self.sim = sim if api_log is None else RecordedSim(sim, api_log, episode)
        self.verbose = verbose
        self.clear_enabled = bool(clear)
        self.k_clear_max = int(k_clear_max)
        self.sweep_opp_radius = float(sweep_opp_radius)   # 顺路补测半径 / m
        self._logfile = open(logfile, "w", encoding="utf-8") if logfile else None
        self.obs: Dict[int, List[Obs]] = defaultdict(list)
        self.meas: Dict[int, List[Meas]] = defaultdict(list)
        self.n_skip = 0
        self.actions: List[Dict[str, Any]] = []
        self.regions: Dict[int, DirProbRegion] = {}
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.cleared: set = set()
        self.pos = np.zeros(2)
        self.vt = 0.0
        self.n_measure = 0
        self.n_clear = 0
        self.n_side_scan = 0        # 清除动作时对"直径大频道"的顺路补测次数
        self.n_side_cand = 0        # 顺路补测候选数（直径大且未满且够近）
        self.n_side_skip_far = 0    # 因距估计点 >1500m 必然无信号而跳过的候选
        self.episode = episode
        self.deadline = float("inf")
        self.travel_m = 0.0
        self.stage = "sweep"
        self.plan: Optional[SweepPlan] = None
        self.first_heard: Dict[int, int] = {}       # 频道 → 首次听到时的测量位置序号
        self.first_heard_at: Dict[int, Tuple[float, float]] = {}
        self.scan_steps: List[Dict[str, Any]] = []
        self._cur_step: Optional[Dict[str, Any]] = None

    # ---- 日志 ----
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
        if self._logfile:
            print(msg, file=self._logfile, flush=True)

    def close(self) -> None:
        if self._logfile:
            self._logfile.close()
            self._logfile = None

    # ---- 原子动作 ----
    def measure(self, x: float, y: float, channel: int) -> dict:
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
        if reg is None or reg.region.is_empty:
            return False
        return reg.min_distance_to(at) > RECEIVE_MAX + 1.0

    def clear(self, x: float, y: float, channel: int) -> bool:
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
        if self._cur_step is not None:
            self._cur_step["clears"].append({"channel": int(channel), "success": bool(ok)})
        return ok

    def _note(self, kind: str, x: float, y: float, channel: int,
              outcome: Optional[str] = None, theta: Optional[float] = None) -> None:
        self.actions.append({
            "seq": len(self.actions), "kind": kind, "stage": self.stage,
            "x": float(x), "y": float(y), "channel": int(channel),
            "outcome": outcome, "theta": theta,
            "virtual_time_s": round(self.vt, 3), "travel_m": round(self.travel_m, 2),
        })

    def _begin_scan_step(self, index: int, label: str, at: Sequence[float],
                         n_channels: int) -> None:
        self._cur_step = {
            "index": int(index), "label": label,
            "x": float(at[0]), "y": float(at[1]), "n_channels": int(n_channels),
            "counts": {}, "measures": [], "clears": [],
            "virtual_time_s": round(self.vt, 3), "travel_m": round(self.travel_m, 2),
        }

    def _end_scan_step(self, counts: Dict[str, int]) -> None:
        step = self._cur_step
        self._cur_step = None
        if step is None:
            return
        step["counts"] = dict(counts)
        step["virtual_time_s"] = round(self.vt, 3)
        step["travel_m"] = round(self.travel_m, 2)
        step["cleared"] = sorted(self.cleared)
        step["path"] = [(0.0, 0.0)] + [(a["x"], a["y"]) for a in self.actions]
        step["regions"] = {ch: verts for ch, reg in sorted(self.regions.items())
                           if len(verts := list(reg.vertices)) >= 3}
        step["estimates"] = {ch: (mec[0], mec[1], mec[2])
                             for ch, reg in sorted(self.regions.items())
                             if (mec := reg.enclosing_circle) is not None}
        self.scan_steps.append(step)

    def _out_of_time(self) -> bool:
        return time.monotonic() > self.deadline

    def region(self, channel: int) -> DirProbRegion:
        if channel not in self.regions:
            reg = DirProbRegion(err=BEARING_ERROR_DEG, radius=REGION_RADIUS, sides=CLIP_SIDES)
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
        return float(self.region(channel).diameter)

    def _precise(self, channel: int) -> bool:
        """判据：可能源集合的最小覆盖圆半径 + 裁剪误差 < 20 m（= 清除半径）。

        只决定"还要不要继续补测"，不决定清不清除（清除一律先就近试一下）。
        """
        mec = self.region(channel).enclosing_circle
        return mec is not None and mec[2] + CLIP_ERR < CLEAR_RADIUS

    # ---- 阶段 1：扫描 ----
    def _est(self, channel: int) -> Optional[Tuple[float, float]]:
        mec = self.region(channel).enclosing_circle
        return (mec[0], mec[1]) if mec is not None else None

    def _sweep_channels(self, at: Sequence[float]) -> List[int]:
        """该测量位置要测的频道：未清除 且（从未听到 或（顺路补测视角））。

        * 从未听到的频道：**必测**——扫描布局的作用就靠"每个测量位置都测所有未听到频道"；
        * 已听到但视角不足（< OBS_CAP）且估计还不够准、当前点又在其估计附近（≤ sweep_opp_
          radius）的频道：顺路补测，多一条不同角度的射线（对仅单侧可听的定向源尤其宝贵）。
        """
        out: List[int] = []
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

    def _sweep_at(self, at: Sequence[float], label: str, index: int) -> Dict[str, int]:
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
            self.log(f"  点{k} ({at[0]:.0f}, {at[1]:.0f})：测 {counts['direction'] + counts['near'] + counts['no_signal']}"
                     f" 次（示向度 {counts['direction']}，无信号 {counts['no_signal']}"
                     f"{skip_note}），"
                     f"累计听到 {heard} 个频道")

    # ---- 阶段 2a：诊断 ----
    def diagnose(self) -> Dict[str, int]:
        precise = 0
        for ch in sorted(self.obs):
            if ch in self.cleared:
                continue
            d = self.diameter(ch)
            mec = self.region(ch).enclosing_circle
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
    def _clear_points(self, channels: Sequence[int]) -> np.ndarray:
        pts = []
        for c in channels:
            mec = self.region(c).enclosing_circle
            pts.append(np.array([mec[0], mec[1]]) if mec else np.array(self.pos, dtype=float))
        return np.asarray(pts, dtype=float)

    def _nearest_order(self, channels: Sequence[int]) -> List[int]:
        """精确最短开放路径定序（Held-Karp；见 cumcm.t3.strategy 同名列注释）。"""
        pts = self._clear_points(channels)
        idx = exact_open_order(len(channels), dist_matrix(pts, self.pos))
        return [channels[int(i)] for i in idx]

    # ---- 阶段 2c：补测 ----
    def refine(self, channel: int) -> int:
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

    def _try_clear(self, channel: int, tag: str) -> Optional[str]:
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

    def _nearby_try_clear(self, channel: int) -> Optional[str]:
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
                        radius: float = CLEAR_RADIUS) -> Tuple[List[Tuple[float, float]], float]:
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
        if not self.clear_enabled or self._out_of_time():
            return None
        pts, leftover = self._k_cover_points(channel, self.k_clear_max)
        if not pts or leftover > 1e-9:
            return None
        rec = self.tracks.setdefault(channel, {})
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

    def _finish_clear(self, channel: int) -> str:
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
        rec = self.tracks.setdefault(channel, {})
        if channel in self.cleared:
            rec.update({"method": "survey-near", "cleared": True})
            return
        if not self.clear_enabled:
            if not self._precise(channel):
                self.refine(channel)
            method: Optional[str] = "skipped"
        else:
            method = self._nearby_try_clear(channel)
            if method is None:
                method = self._multi_try_clear(channel)
            if method is None:
                self.refine(channel)
                method = self._finish_clear(channel)
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
    def run(self, plan: SweepPlan) -> Dict[str, Any]:
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
            "methods": {m: sum(1 for r in self.tracks.values() if r.get("method") == m)
                        for m in ("try", "try-refined", "multi", "near", "homing",
                                  "survey-near", "near@center", "near@probe", "failed")},
            "tracks": self.tracks,
        }