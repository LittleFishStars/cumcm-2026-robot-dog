"""机器狗策略：巡视扫描（阶段一）+ 定位与清除（阶段二）。

流程总览（细节见各方法文档）：

    阶段一  依次移动到 7 个覆盖圆圆心，每站扫描所有"还值得测"的频道；
            第 1 站就在原点，于是它天然是一次**起始全频道扫描**（一次拿到"哪些频道在
            1000 m 内"），并用它锚定到的方位调整巡视绕向与落脚点；每站结束还会顺路清除。
    阶段二  诊断 -> 按估计点距离做最近邻 + 2-opt 的访问顺序 -> 逐频道四级清除：
            就近试清 -> 多清几次（K 个半径 20 m 的圆盖满区域）-> 补测后清 -> 沿示向度逼近兜底。

贯穿其中的两条"省时间"原则：
* 硬约束（见 regions.ProbRegion）让"必然听不到"的测量可直接**跳过**（省 6 s/次）；
* 清除一律"就近试清"，不设直径门槛 —— 走到源的近处是清除的必要动作，绕不过去。
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from functools import partial

from cumcm.common.geometry import dist
from cumcm.common.geometry import clamp_to_region as _clamp_to_region
from cumcm.common.routing import dist_matrix, nearest_order, two_opt_greedy
from cumcm.t3.config import (BEARING_ERROR_DEG, CHANNELS, CLEAR_RADIUS, CLIP_ERR, CLIP_SIDES,
                             COVER_RADIUS, HOMING_CAP, HOMING_MAX, HOMING_STEP,
                             INLINE_DETOUR, INLINE_TRY_RADIUS, K_CLEAR_MAX, K_COVER_SAMPLES,
                             K_COVER_STEP_MIN, NEAR_RADIUS, OBS_CAP, RECEIVE_MAX, RECEIVE_MID,
                             REFINE_MAX, REGION_MARGIN, REGION_RADIUS, SAFETY_MARGIN, TOL,
                             TRY_CLEAR_RADIUS)
from cumcm.t3.covering import CoverPlan
from cumcm.t3.probing import hypothesis_points, probe_candidates
from cumcm.t3.regions import Meas, Obs, ProbRegion

# 把坐标拉回作业圆域（半径留 1 m 数值余量）。公共层不内置默认半径，这里按本题常量绑定，
# 于是所有动作入口都能直接写 clamp_to_region(x, y)，无需各处重复算半径。
clamp_to_region = partial(_clamp_to_region, radius=REGION_RADIUS - REGION_MARGIN)


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

        最近邻与 2-opt 都复用 cumcm.common.routing（原先本类自带一份实现，与 T3_ga.py 的
        那份重复；抽取后只保留"把频道的估计点拼成数组"这一层业务逻辑）。

        注意下标与频道号的映射：公共算子返回的是**数组下标**，而本类的调用方按**频道号**
        使用顺序。`channels` 由调用方保证升序（`sorted(self.obs)` 过滤而来），而下标并列时
        公共算子取小下标，故映射回来后与"并列取小频道号"完全等价。
        """
        pts = self._clear_points(channels)
        idx = nearest_order(pts, self.pos)
        if len(idx) >= 3:                               # 2-opt 对 0/1/2 个点无意义
            idx = two_opt_greedy(idx, dist_matrix(pts, self.pos))
        return [channels[i] for i in idx]

    def _clear_points(self, channels: Sequence[int]) -> np.ndarray:
        """各频道的清缺点（定位区域最小覆盖圆圆心）；拿不到估计点的退回当前位置。

        返回的数组与 `channels` 同序，故 2-opt 的下标可直接映射回频道号。
        """
        pts = []
        for c in channels:
            mec = self.region(c).enclosing_circle
            pts.append(np.array([mec[0], mec[1]]) if mec else np.array(self.pos, dtype=float))
        return np.asarray(pts, dtype=float)

    # ---- 阶段 2b：按文献准则补测缩小定位区域 ----
    def refine(self, channel: int) -> int:
        """补测直到定位区域直径 < 40 m（或达到轮次/候选上限）；返回实际补测次数。"""
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
