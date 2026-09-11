"""机器狗策略（GA 对照方案）：滚动重规划（覆盖巡视 + 定位/补测 + 就地清除合并为一条路线）。

与确定性方案（cumcm.t3.strategy）的目标相同、路线不同，用于在论文中对照"确定性方法 vs
启发式方法"的时间与定位精度。所有动作坐标统一由 clamp_to_region 拉回作业圆域内。

**非分段式**：不再"先把所有覆盖路点巡视完、再统一定位、最后统一清除"，而是把"尚未访问的
覆盖路点（硬约束，保证不漏源）"与"已听到且估计够准的源"放进**同一条 GA 规划路线**，每步只
执行第一个目标并重排（滚动时域）。下一站是源就当场逼近清除，是路点就扫过并更新观测。这样
机器狗不必把 1800 m 圆域跑两遍，越早清除越省专程往返。滚动中未能清掉的源由收尾的
`localize_all` + `_clear_all` 兜底（补测、修病态几何、沿示向度逼近），保证 100% 清除。
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cumcm.common.geometry import ang_diff, bearing, dist
from cumcm.common.geometry import clamp_to_region as _clamp_to_region
from cumcm.common.sim_client import RecordedSim
from cumcm.t3ga.config import (CHANNELS, CONVERGED_FITNESS, COORD_LIMIT, COVER_RADIUS, GEOM_SIGMA, GEOM_SIN_MIN, HOMING_MAX, HOMING_STEP, OBS_PER_SOURCE, PERP_STEPS, PRECISE_SIGMA, REGION_MARGIN, REGION_RADIUS, RHO_GATE, SAFETY_MARGIN, SEED, SINGLE_PROBES)
from cumcm.t3ga.localize import (Estimate, GeneticLocalizer, Obs, max_ray_sine, position_sigma)
from cumcm.t3ga.covering import covering_waypoints
from cumcm.common.routing import dist_matrix as _dist_matrix
from cumcm.common.routing import path_len as _path_len
from cumcm.t3ga.routing_ga import route_ga

clamp_to_region = partial(_clamp_to_region, radius=REGION_RADIUS - REGION_MARGIN)

# ----------------------------------------------------------------------------
# 机器狗策略
# ----------------------------------------------------------------------------
class RobotDog:
    """滚动重规划：覆盖路点与已定位源共用一条 GA 路线，边巡视边定位边清除。"""

    def __init__(self, sim, verbose: bool = True, logfile: Optional[str] = None,
                 seed: int = SEED, episode: int = 0,
                 api_log: Optional[ApiLog] = None) -> None:
        # 传入 api_log 时套一层记录代理：4 个接口的每一次调用都会落盘
        self.sim = sim if api_log is None else RecordedSim(sim, api_log, episode)
        self.verbose = verbose
        # 过程日志用 "w"：每局开头重写，于是整份日志只描述**最新一局**（原先 "a" 追加，
        # 跨局、跨运行无限累积，几轮演练后文件里混着几百局的内容难以查阅）。
        self._logfile = open(logfile, "w", encoding="utf-8") if logfile else None
        self.obs: Dict[int, List[Obs]] = defaultdict(list)
        self.cleared: set = set()
        self.pos = np.zeros(2)
        self.vt = 0.0
        self.n_measure = 0
        self.n_clear = 0
        self.deadline = float("inf")
        self.localizer = GeneticLocalizer(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.episode = episode      # 局号（用于训练记录分组）
        self.ga_runs: List[Dict[str, Any]] = []     # GA 训练记录（逐次调用一条）
        self.final_est: Dict[int, Estimate] = {}    # 本局各频道的最终定位解
        # 滚动重规划的定位缓存：以"示向度条数"为键，避免同一批观测重复跑定位 GA
        self._est_cache: Dict[int, Optional[Estimate]] = {}
        self._est_key: Dict[int, int] = {}
        # 轨迹记录：起始于 (0,0)，之后每次实际动作点依次追加，用于每局结束后出轨迹图
        self.track: List[Tuple[float, float]] = [(0.0, 0.0)]
        self.marks: List[Tuple[float, float, str]] = []   # 动作点及其类型（measure/clear）
        self.hits: List[Tuple[float, float]] = []   # 成功清除的落点
        # 逐步扫描记录：一步 = 机器狗停在某处把当前该测的频道测一遍（起点全频道扫描 + 滚动
        # 重规划里每到一个覆盖路点）。仅登记、不在此处绘图 —— 绘图统一在 /exit 之后做，
        # 不占用现实时间预算。
        self.scan_steps: List[Dict[str, Any]] = []
        self._cur_step: Optional[Dict[str, Any]] = None

    # ---- 日志 ----
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
        if self._logfile:
            print(msg, file=self._logfile, flush=True)

    def close(self) -> None:
        """关闭日志文件（幂等；run() 已在 finally 中调用）。"""
        if self._logfile:
            self._logfile.close()
            self._logfile = None

    # ---- 原子动作 ----
    def measure(self, x: float, y: float, channel: int) -> dict:
        """测向；返回原始响应，direction 时自动记录示向度。"""
        x, y = clamp_to_region(float(x), float(y))      # 机器狗不得离开作业圆域
        r = self.sim.measure(x, y, channel)
        if not r.get("accepted"):
            raise RuntimeError(f"/measure 被拒绝：{r}")
        self.pos, self.vt = np.array([x, y]), float(r["virtual_time_s"])
        self.track.append((x, y))
        self.marks.append((x, y, "measure"))
        self.n_measure += 1
        outcome = r.get("measure_result", "no_signal")
        if outcome == "direction":
            self.obs[channel].append(Obs(channel, x, y, float(r["svd_deg"])))
        if self._cur_step is not None:
            self._cur_step["measures"].append(
                {"channel": int(channel), "outcome": outcome,
                 "theta": float(r["svd_deg"]) if outcome == "direction" else None})
        return r

    def clear(self, x: float, y: float, channel: int) -> bool:
        """清除；返回是否成功。"""
        x, y = clamp_to_region(float(x), float(y))      # 与测向同一处裁剪
        r = self.sim.clear(x, y, channel)
        if not r.get("accepted"):
            raise RuntimeError(f"/clear 被拒绝：{r}")
        self.pos, self.vt = np.array([x, y]), float(r["virtual_time_s"])
        self.track.append((x, y))
        self.marks.append((x, y, "clear"))
        self.n_clear += 1
        ok = r.get("clear_result") == "success"
        if ok:
            self.cleared.add(channel)
            self.hits.append((x, y))
        if self._cur_step is not None:
            self._cur_step["clears"].append({"channel": int(channel), "success": bool(ok)})
        return ok

    def _out_of_time(self) -> bool:
        return time.monotonic() > self.deadline

    def _plan_route(self, pts: np.ndarray, label: str) -> Tuple[List[int], float]:
        """路线 GA 规划从当前位置出发访问 pts 的顺序；返回（访问顺序, 总里程 m）。

        顺带记录本次训练（进化）过程；里程在这里一次算好，供调用方直接使用。

        **恰有一个目标（或没有）时不存在"路线"问题**：直接从当前位置过去即可，故跳过 GA、
        也不登记训练记录。两个原因：一是 `route_ga` 对 n ≤ 1 会提前返回、不写进化记录，
        原先在此处无条件读 `record[0]` 会 IndexError（滚动重规划里"只剩最后一个路点且没有
        待清源"时正好命中，实测跑到第 7 局才触发）；二是若为它补记一条 generations=0 的
        记录，会破坏"每次路线规划都跑满代数"这条可核验性（见 T3_validate 的 E9）。
        """
        pts = np.asarray(pts, dtype=float)
        if len(pts) <= 1:
            order = list(range(len(pts)))
            return order, _path_len(order, _dist_matrix(pts, self.pos))
        seed = int(self._rng.integers(1 << 31))
        record: List[List[float]] = []
        order = route_ga(pts, self.pos, seed=seed, record=record)
        length = _path_len(order, _dist_matrix(pts, self.pos))
        first, last = record[0], record[-1]
        self.ga_runs.append({
            "episode": self.episode, "ga": "route", "label": label, "seed": seed,
            "n_points": len(pts), "initial_best_g0": first[1],
            "generations": int(last[0]), "final_length_m": length,
            "solution": order, "history": record,
        })
        return order, length

    def _begin_scan_step(self, index: int, label: str, at: Sequence[float],
                         n_channels: int) -> None:
        """开始记录一步扫描（见 `scan_steps`）。动作由 measure/clear 自动挂到当前步上。"""
        self._cur_step = {
            "index": int(index), "label": label,
            "x": float(at[0]), "y": float(at[1]), "n_channels": int(n_channels),
            "counts": {}, "measures": [], "clears": [],
            "virtual_time_s": round(self.vt, 3), "travel_m": round(self._travel_m(), 2),
        }

    def _end_scan_step(self, counts: Dict[str, int]) -> None:
        """收尾一步扫描：填统计与状态快照，然后登记（供收尾统一出图）。"""
        step = self._cur_step
        self._cur_step = None
        if step is None:
            return
        step["counts"] = dict(counts)
        step["virtual_time_s"] = round(self.vt, 3)
        step["travel_m"] = round(self._travel_m(), 2)
        step["cleared"] = sorted(self.cleared)
        step["path"] = [(0.0, 0.0)] + [(float(t[0]), float(t[1])) for t in self.track]
        # GA 的估计形态是"点 + 位置 1σ"（不是多边形），故快照 σ 圆；官方模式下同样可用
        step["estimates"] = {ch: (e.x, e.y, e.sigma)
                             for ch, e in sorted(self.final_est.items())
                             if ch not in self.cleared}
        self.scan_steps.append(step)

    def _travel_m(self) -> float:
        """累计行驶里程（沿 track 逐段累加；track[0] 是起点 (0,0)）。"""
        pts = np.asarray(self.track, dtype=float)
        return float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))) if len(pts) > 1 else 0.0

    def _sweep(self, channels: Sequence[int], at: Sequence[float],
               label: Optional[str] = None, index: int = 0) -> Dict[str, int]:
        """在 at 处逐频道测向（按频道号升序以减少切换）；近距则就地清除。

        传入 `label` 时把这一步登记进 `scan_steps`（收尾逐步骤出"扫描结果图"）。
        """
        if label is not None:
            self._begin_scan_step(index, label, at, len(channels))
        counts = {"direction": 0, "near": 0, "no_signal": 0}
        for ch in sorted(channels):
            if self._out_of_time():
                break
            res = self.measure(at[0], at[1], ch).get("measure_result", "no_signal")
            counts[res] = counts.get(res, 0) + 1
            if res == "near" and self.clear(at[0], at[1], ch):
                self.log(f"    [near] 频道{ch} 距离过近，就地清除成功")
        if label is not None:
            self._end_scan_step(counts)
        return counts

    def _probe(self, channel: int, candidates: Sequence[Sequence[float]]) -> bool:
        """依次在候选点测向：取到 direction 或就地清除后即停；返回是否新增示向度。"""
        for x, y in candidates:
            if self._out_of_time():
                return False
            if max(abs(x), abs(y)) > COORD_LIMIT:      # 超出协议允许的坐标范围
                continue
            res = self.measure(x, y, channel).get("measure_result", "no_signal")
            if res == "direction":
                return True
            if res == "near":
                self.clear(x, y, channel)
                return False
        return False

    def _estimate(self, channel: int) -> Optional[Estimate]:
        """定位 GA 求解该频道源位置，并给出位置 1σ；同时记录本次训练（进化）过程。"""
        ol = self.obs.get(channel, ())
        if len(ol) < 2:
            return None
        record: List[List[float]] = []
        G = self.localizer.localize(ol, record=record)
        if G is None:
            return None
        est = Estimate(float(G[0]), float(G[1]),
                       position_sigma(GeneticLocalizer.covariance(ol, G)))
        self.final_est[channel] = est       # 滚动清除的源也要留档，供与真值比对
        first, last = record[0], record[-1]
        self.ga_runs.append({
            "episode": self.episode, "ga": "localize", "label": f"频道{channel}",
            "seed": self.localizer.seed, "channel": channel, "n_obs": len(ol),
            "generations": int(last[0]),
            "converged": bool(last[1] < CONVERGED_FITNESS),
            "initial_best_g0": first[1], "final_fitness": last[1],
            "residual_rms_deg": self._residual(est.point, ol),
            "sigma_m": est.sigma, "solution": [est.x, est.y], "history": record,
        })
        return est

    @staticmethod
    def _residual(G: Sequence[float], obs_list: Sequence[Obs]) -> float:
        return float(np.sqrt(np.mean([ang_diff(bearing((o.x, o.y), G), o.theta) ** 2
                                       for o in obs_list])))

    # ---- 阶段 1：起点完整扫描 ----
    def _initial_scan(self) -> None:
        self.log(f"阶段1：起点完整扫描全部 {len(CHANNELS)} 个频道 @ "
                 f"({self.pos[0]:.0f}, {self.pos[1]:.0f})")
        counts = self._sweep(CHANNELS, self.pos, label="起点全频道扫描", index=0)
        self.log(f"  有示向度 {counts['direction']} 个 {sorted(self.obs)}，"
                 f"近距清除 {counts['near']} 个，无信号 {counts['no_signal']} 个")

    # ---- 阶段 2：滚动重规划（覆盖巡视与定位清除合并）----
    def _precise(self, channel: int) -> bool:
        """该频道当前估计是否已足够准（位置 1σ ≤ PRECISE_SIGMA），无需再测向。

        只用已缓存的定位解判断，且要求缓存与当前观测条数一致，避免拿过期解做决定。缓存由
        本类每轮先调用的 `_localize_ready` 填充；首轮缓存为空时返回 False（照常测向），故
        不改变第一轮行为。PRECISE_SIGMA=0 时恒为 False（关闭本优化）。
        """
        if PRECISE_SIGMA <= 0:
            return False
        e = self._est_cache.get(channel)
        return (e is not None
                and self._est_key.get(channel) == len(self.obs.get(channel, ()))
                and e.sigma <= PRECISE_SIGMA)

    def _active_channels(self) -> List[int]:
        """仍需测向的频道：未清除、示向度条数未达上限，且**估计还不够准**。

        第三条（`_precise`）是滚动重规划版的"提前收工"：估计已够准的频道再测只能把 σ 从
        20 m 压到 15 m，却要每站多花 5 s；它会被当作已定位源插入同一条路线，走到就近清掉，
        而真正的兜底（收尾 localize_all + _clear_all）保证仍能 100% 清除。未听到的频道不在
        任何缓存里，`_precise` 返回 False，故绝不会因本优化而漏源。
        """
        return [c for c in CHANNELS if c not in self.cleared
                and len(self.obs.get(c, ())) < OBS_PER_SOURCE
                and not self._precise(c)]

    def _localize_ready(self) -> Dict[int, Estimate]:
        """对"已有 ≥2 条示向度、尚未清除"的频道跑定位 GA，返回估计字典。

        以观测条数为缓存键：同一批观测只训练一次，避免滚动循环每步重复跑 GA 记录训练数据。
        """
        out: Dict[int, Estimate] = {}
        for ch in sorted(self.obs):
            if ch in self.cleared:
                continue
            n = len(self.obs[ch])
            if n < 2:
                continue
            if self._est_key.get(ch) != n:
                self._est_cache[ch] = self._estimate(ch)
                self._est_key[ch] = n
            e = self._est_cache.get(ch)
            if e is not None:
                out[ch] = e
        return out

    def _gated(self, attempted: set) -> Dict[int, Estimate]:
        """挑出"值得插入当前路线顺路清除"的已定位源。

        条件：位置 1σ ≤ RHO_GATE（估计可信），且不在 `attempted` 里 —— 后者是"已试过一次
        （无论成败）"的频道集合，避免在同一估计上反复绕路白跑；没清掉的源留给收尾的
        localize_all + _clear_all 专程处理。
        """
        out: Dict[int, Estimate] = {}
        for ch, e in self._localize_ready().items():
            if e.sigma > RHO_GATE or ch in attempted:
                continue
            out[ch] = e
        return out

    def _rolling_survey_clear(self) -> None:
        """把覆盖路点与已定位源放进同一条 GA 路线，每步只执行第一个目标并重排。

        约束：覆盖路点是**硬任务**（必须全部访问或确认无信息），保证 1000 m 覆盖不漏源；
        已定位源只是**可选任务**，仅当位置 1σ ≤ RHO_GATE 时才插入，防止为不可信估计白跑。
        每步更新观测后重排，是"边测边定位边清"的关键：清掉的源立即从后续路线消失，省掉
        巡视结束后的专程往返。未能就地清除的源留给收尾的 localize_all + _clear_all 兜底。

        两处相对旧版的收紧（都为省时，且不削弱"不漏源 + 必清除"的保证）：
        ① `_active_channels` 对已够准（σ≤PRECISE_SIGMA）的频道停测，只清不测 —— 继续测只
           能小幅降 σ，却要每个路点多花 5 s 测向 + 1 s 换频；
        ② 当没有任何频道需要测（`active` 为空）时，不再走访剩余路点（走访只为测量），直接
           沿路线把已定位源清掉，然后收尾 —— 路点覆盖保证的是"能听到"，不是"必须走完"。
        实测（20 局 seed 0~19）虚拟时间 4220→4135 s、测向 2816→2663 次，256/256 全清。
        """
        waypoints = covering_waypoints()
        visited = [False] * len(waypoints)
        attempted: set = set()          # 已在滚动中试清过（无论成败）的频道，收尾再处理
        n_step = 1                      # 步骤编号（0 已被起点全频道扫描占用）
        self.log(f"阶段2：滚动重规划，{len(waypoints)} 个覆盖路点（覆盖半径 "
                 f"{COVER_RADIUS:.0f} m，插入门 σ≤{RHO_GATE:.0f} m，停测门 "
                 f"σ≤{PRECISE_SIGMA:.0f} m）")
        while not self._out_of_time():
            gated = self._gated(attempted)      # 先跑定位、填缓存，再判是否还需测向
            active = self._active_channels()
            pending = [i for i in range(len(waypoints)) if not visited[i]]
            has_station = bool(active) and bool(pending)
            if not has_station and not gated:
                break
            blocks = []
            if has_station:
                blocks.append(waypoints[pending])
            if gated:
                blocks.append(np.array([gated[ch].point for ch in gated], dtype=float))
            pts = blocks[0] if len(blocks) == 1 else np.vstack(blocks)
            n_station = len(pending) if has_station else 0
            order, _ = self._plan_route(pts, "滚动重规划")
            nxt = order[0]
            if nxt >= n_station:                # 下一站是源：就地逼近清除
                ch = list(gated)[nxt - n_station]
                attempted.add(ch)
                self.log(f"  滚动：先清频道{ch}（估计 σ={gated[ch].sigma:.1f} m，"
                         f"剩余路点 {len(pending)} 个、待清源 {len(gated)} 个）")
                self._home_and_clear(ch, gated)
            else:                               # 下一站是路点：移动并扫描
                idx = pending[nxt]
                visited[idx] = True
                wp = waypoints[idx]
                self.log(f"  滚动：下一站路点 {idx} @ ({wp[0]:.0f}, {wp[1]:.0f})"
                         f"（剩余 {len(pending) - 1} 个路点，待清源 {len(gated)} 个）")
                self._sweep(self._active_channels(), wp,
                            label=f"覆盖路点 #{idx + 1}", index=n_step)
                n_step += 1
        self.log(f"阶段2 完成：虚拟时刻 {self.vt:.0f} s，已清除 {len(self.cleared)} 个，"
                 f"滚动中试清 {len(attempted)} 个源")

    # ---- 阶段 3：定位 ----
    def _single_candidates(self, channel: int) -> List[Tuple[float, float]]:
        """单射线补测点：沿示向度前移并带侧偏，保证仍在接收范围内且拉开交会角。"""
        o = self.obs[channel][0]
        return [(o.x + t * math.cos(math.radians(o.theta + s * beta)),
                 o.y + t * math.sin(math.radians(o.theta + s * beta)))
                for t, beta in SINGLE_PROBES for s in (1.0, -1.0)]

    def _perp_candidates(self, channel: int, e: Estimate) -> List[Tuple[float, float]]:
        """几何病态时的补测点：沿"良态方向"外移。

        近共线时取公共直线的法向（新射线与原直线相交即可定距）；否则取位置协方差最小
        特征值方向，即当前最不确定的方向。
        """
        ol = self.obs[channel]
        if max_ray_sine(ol) < GEOM_SIN_MIN:
            t2 = 2.0 * np.radians([o.theta for o in ol])
            orient = 0.5 * math.atan2(float(np.sin(t2).mean()), float(np.cos(t2).mean()))
            nx, ny = -math.sin(orient), math.cos(orient)
        else:
            _, vecs = np.linalg.eigh(GeneticLocalizer.covariance(ol, e.point))
            nx, ny = float(vecs[0, 0]), float(vecs[1, 0])
        return [(e.x + r * nx, e.y + r * ny) for r in PERP_STEPS]

    def _resolve_singles(self) -> None:
        """对只有单条射线、无法定距的频道补测第二视角。"""
        singles = [c for c in sorted(self.obs) if c not in self.cleared and len(self.obs[c]) == 1]
        if not singles:
            return
        self.log(f"阶段3a：单射线频道补测第二视角 {singles}")
        for c in singles:
            self._probe(c, self._single_candidates(c))

    def _fix_geometry(self, channel: int) -> bool:
        """交会几何病态时补测垂直视角，最多 3 轮；返回是否有补测。"""
        probed = False
        for _ in range(3):
            e = self._estimate(channel)
            ol = self.obs.get(channel, ())
            if e is None or len(ol) < 2:
                return probed
            if e.sigma <= GEOM_SIGMA and max_ray_sine(ol) >= GEOM_SIN_MIN:
                return probed
            if not self._probe(channel, self._perp_candidates(channel, e)):
                return probed
            probed = True
        return probed

    def localize_all(self) -> Dict[int, Estimate]:
        """对每个已积累足够示向度的频道跑定位 GA，并修掉病态几何。"""
        self._resolve_singles()
        est: Dict[int, Estimate] = {}
        for c in sorted(self.obs):
            if c in self.cleared:
                continue
            e = self._estimate(c)
            if e is None:
                continue
            if e.sigma > GEOM_SIGMA or max_ray_sine(self.obs[c]) < GEOM_SIN_MIN:
                if self._fix_geometry(c):
                    e = self._estimate(c) or e
            est[c] = e
        if est:
            rms = float(np.mean([self._residual(e.point, self.obs[c]) for c, e in est.items()]))
            self.log(f"阶段3：GA 定位 {len(est)} 个源，示向度残差 RMS = {rms:.2f}°")
        return est

    # ---- 阶段 4：清除 ----
    def _home_and_clear(self, channel: int, est: Dict[int, Estimate]) -> None:
        """靠近并清除：先按定位解 /clear，未成功则沿最新实测示向度逐步逼近。

        示向度误差是"同一地点固定"的系统误差，仅靠多视角交会存在沿射线方向的偏移；
        靠近后直接沿最新示向度走一步（16 m 步长的横向误差约 0.3 m）即可稳定进入清除半径。

        **逼近中每次更新都要回写 `final_est`**：滚动重规划传入的 `est` 是局部字典 `gated`，
        不回写的话 `final_est` 会永远停在最早那次（示向度很少的）粗糙解，逐局定位误差是被
        记录下来的那个陈旧值而不是实际达到的精度 —— 实测最坏由 19.7 m 虚高到 68.9 m，
        而同一局每个清除点其实都落在源 20 m 内。分段式路径下 `est` 就是 `final_est`，
        同步写是幂等的，故这一改动对旧路径无影响。
        """
        for _ in range(HOMING_MAX):
            e = est.get(channel)
            if e is None or self._out_of_time():
                return
            if self.clear(e.x, e.y, channel):
                self.log(f"    [清除] 频道{channel} 成功 @ ({e.x:.0f}, {e.y:.0f})")
                return
            r = self.measure(e.x, e.y, channel)
            res = r.get("measure_result")
            if res == "near":
                if self.clear(e.x, e.y, channel):
                    self.log(f"    [清除] 频道{channel} 近距命中")
                return
            if res == "direction":
                th = math.radians(float(r["svd_deg"]))
                nx, ny = clamp_to_region(e.x + HOMING_STEP * math.cos(th),
                                         e.y + HOMING_STEP * math.sin(th))
                est[channel] = Estimate(nx, ny, e.sigma)
                self.final_est[channel] = est[channel]    # 见下
                continue
            # no_signal：定位偏了，用新示向度重新定位；仍不行则补测视角
            ne = self._estimate(channel)
            if ne is None or dist(ne.point, e.point) <= 3.0:
                if len(self.obs[channel]) == 1:
                    self._probe(channel, self._single_candidates(channel))
                ne = self._estimate(channel)
            if ne is not None:
                est[channel] = ne
                self.final_est[channel] = ne
        self.log(f"    [警告] 频道{channel} 逼近 {HOMING_MAX} 次仍未能清除")

    def _clear_all(self, est: Dict[int, Estimate]) -> None:
        todo = [c for c in sorted(est) if c not in self.cleared]
        if not todo:
            return
        pts = np.array([est[c].point for c in todo])
        order, length = self._plan_route(pts, "清除顺序")
        self.log(f"阶段4：GA 规划 {len(todo)} 个源的清除顺序，路程 {length:.0f} m")
        for k, i in enumerate(order):
            if self._out_of_time():
                self.log(f"    [警告] 现实时间不足，剩余 {len(order) - k} 个源未处理")
                return
            self._home_and_clear(todo[i], est)

    # ---- 主流程 ----
    def run(self) -> Dict[str, Any]:
        """跑完一局：/enter → 起点扫描 → 滚动重规划（边巡视边定位边清）→ 兜底清除 → /exit。"""
        self.log("=" * 74)
        self.log("策略：滚动重规划（定位 GA + 路线 GA，覆盖巡视与清除合并）")
        enter = self.sim.enter()
        if not enter.get("accepted"):
            raise RuntimeError(f"/enter 被拒绝：{enter}")
        left = float(enter.get("remaining_real_duration_s", 1200.0))
        self.deadline = time.monotonic() + max(left - SAFETY_MARGIN, 0.0)
        self.log(f"/enter 成功：虚拟时刻 {enter.get('virtual_time_s')} s，现实剩余 {left:.0f} s")

        try:
            self._initial_scan()
            self._rolling_survey_clear()
            self.localize_all()                     # 兜底定位（解随 _estimate 写入 final_est）
            self._clear_all(self.final_est)         # 兜底清除滚动中未解决的源
        finally:
            try:
                self.sim.exit()                     # 无论成功与否都要正常退出，保住测试记录
            except OSError as exc:
                self.log(f"    [警告] /exit 失败：{exc}")
            self.close()                            # 日志句柄随本局一并关闭

        n = len(self.cleared)
        return {
            "cleared": n,
            "total_time_s": self.vt,
            "avg_time_s": self.vt / n if n else float("inf"),
            "n_measure": self.n_measure,
            "n_clear": self.n_clear,
        }
