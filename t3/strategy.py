"""机器狗策略：阶段一巡视扫描，阶段二定位与清除

流程总览，细节见各方法自己的文档：

    阶段一  依次移动到 7 个覆盖圆圆心，每站扫描所有"还值得测"的频道。
            第 1 站就在原点，它天然就是一次起始全频道扫描，一趟拿到"哪些频道在 1000 m
            内"，再用它锚定到的方位调整巡视绕向与落脚点。每站结束还会顺路清除。
    阶段二  诊断 -> 按估计点距离求精确最短开放路径定序 -> 逐频道四级清除：
            就近试清，多清几次，用 K 个半径 20 m 的圆盖满区域；补测后清；沿示向度逼近兜底。

贯穿其中的两条"省时间"原则：
* 硬约束让"必然听不到"的测量可以直接跳过，省 6 s/次，见 regions.ProbRegion；
* 清除一律"就近试清"，不设直径门槛。走到源的近处是清除的必要动作，绕不过去。

负结果实验存档，2026-09-12 清理 feature/dynamic-cover 分支时并入主线备查：
动态覆盖圆在"替代固定站 / 省里程"上不成立。圆域覆盖需 ≥ 7 个半径 1000 m 圆（ρ₆ 下界），
而走廊切段点恰好总落在两站边界弧交界的弱区，单点替代某固定站的概率仅 ~17~21%（贴站
150~400 m 随机方向实测），需约 4 个贴点联合才平均替代 1 站，替代后全布局最坏距离压到
~1000 m，零余量，只靠源生成域 1770 m 的 30 m 冗余兜底，扫描成本远大于省下的里程。唯一的
"多视角"收益，测向多所以定位略准，也不显著：step 600~1500 五档实测定位误差
7.94/8.10/8.38/8.66/8.10 m，无系统改善，虚拟时间反增 0~16%，所以默认关闭。
"路线式顺路清除" feature/route-clear 同批清理：纯开关实验，无单独结论。
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from functools import partial

from common.actions import ActionRecorder
from common.geometry import ang_diff, dist
from common.geometry import clamp_to_region as _clamp_to_region
from common.routing import dist_matrix, exact_open_by_end, exact_open_order
from common.sim_client import RecordedSim
from t3.config import (BEARING_ERROR_DEG, CHANNELS, CLEAR_RADIUS, CLIP_ERR, CLIP_SIDES,
                             COVER_RADIUS, HOMING_CAP, HOMING_MAX, HOMING_STEP,
                             INLINE_MAX_MEC_R, INLINE_NEAR_R, INLINE_R_MAX, INLINE_R_MIN,
                             K_CLEAR_MAX,
                             K_COVER_SAMPLES,
                             K_COVER_STEP_MIN, NEAR_RADIUS, OBS_CAP, RECEIVE_MAX, RECEIVE_MID,
                             REFINE_MAX, REGION_MARGIN, REGION_RADIUS, SAFETY_MARGIN, TOL,
                             TRY_CLEAR_RADIUS)
from t3.covering import CoverPlan, dense_sector_rotation
from t3.probing import hypothesis_points, probe_candidates
from t3.regions import Meas, Obs, ProbRegion

# 把坐标拉回作业圆域，半径留 1 m 数值余量。公共层不内置默认半径，这里按本题常量绑定，
# 于是所有动作入口都能直接写 clamp_to_region(x, y)，无需各处重复算半径。
clamp_to_region = partial(_clamp_to_region, radius=REGION_RADIUS - REGION_MARGIN)


class RobotDog(ActionRecorder):
    """两阶段机器狗

    阶段一，巡视扫描。按覆盖圆方案依次走到 7 个圆心，在每个圆心对未采够的频道测向，把每个
    源的示向度采集齐全。同一地点误差固定，所以每频道最多采 OBS_CAP 条。起始点原点的那一站
    是"全频道扫描"，扫完立刻用这批示向度做两件零成本的事：把覆盖圆布局转到最密集的扇区
    方向，也就是角度频率最高的那个 60° 区间的中点；再把布局转到"第一个巡视点正对源最密集
    方向顺时针旋转 90° 处"，见 `_orient_route`。旋转不动覆盖保证，覆盖只看点间距离和点到
    原点的距离；也不动巡回路径长度，那只依赖点间距离。它是纯零成本自由度。

    阶段二，定位与清除。对每个频道走同一条流程，没有"够不够准"的清除门槛。
      1. 用问题 1 的交会定位区域得到位置估计，区域是各 ±1° 楔形之交 ∩ 圆域，取最小覆盖圆圆心；
      2. 就近试清：走到估计点直接 /clear 一次。清除半径只有 20 m，走到源的近处本来就是清除
         的必要动作，所以到达后顺手一试的边际代价只有失败时的 3 s，命中却省掉整轮补测。未
         命中时就地复测，该点是区域内离源最近、Fisher 权重最大的位置；
      3. 若估计仍不够集中，直径 ≥ 40 m，就按文献准则在海选候选点里补测，把区域压到
         直径 < 40 m 以下，再到新的估计点试清；单射线频道由此补齐第二视角；
      4. 兜底：万一仍未清除，沿最新实测示向度以 HOMING_STEP 步长逼近，确定性，无随机。
    其中"直径 < 40 m"只是补测的收工条件，表示估计已够准，不决定清不清除。
    """

    def __init__(self, sim: "Simulator", verbose: bool = True,
                 logfile: str | None = None,
                 episode: int = 0, clear: bool = True,
                 k_clear_max: int = K_CLEAR_MAX,
                 inline_r_min: float = INLINE_R_MIN,
                 inline_r_max: float = INLINE_R_MAX,
                 inline_near_r: float = INLINE_NEAR_R,
                 inline_max_mec_r: float = INLINE_MAX_MEC_R,
                 inline_radius_max: float | None = None,  # None=前向上界取 inline_r_max
                 rotate: bool = True,
                 api_log: "ApiLog | None" = None) -> None:
        """构造两阶段机器狗策略

        Args:
            sim: 模拟器客户端。传的是原始 Simulator；给 api_log 时会自动套一层记录代理
            verbose: 是否打印过程日志
            logfile: 过程日志文件路径，每局开头重写，只留最新一局
            episode: 局号，写进接口日志的 request_id，便于与模拟器行为日志对账
            clear: 是否做阶段二。False 时只做巡视扫描
            k_clear_max: 覆盖定位区域的半径 20 m 圆最多补几个
            inline_r_min: 站点间前向顺路清除的估计点半径下界 / m
            inline_r_max: 站点间前向顺路清除的估计点半径上界 / m
            inline_near_r: 每站到站后的近距顺路清除半径 / m
            inline_max_mec_r: 参与顺路清除的区域最大最小覆盖圆半径 / m
            inline_radius_max: 兼容旧参数，覆盖前向上界的统一值 / m。None 时取 inline_r_max
            rotate: 起始全频道扫描后是否把覆盖圆布局旋转到源最密集方向
            api_log: 接口调用日志。传入后 4 个接口的每次调用都会落盘
        """
        # 传入 api_log 时套一层记录代理，4 个接口的每一次调用都会落盘。官方模式下这就是唯一
        # 的证据链：真值拿不到，但每次请求/响应都有记录。
        self.sim = sim if api_log is None else RecordedSim(sim, api_log, episode)
        self.verbose = verbose
        self.clear_enabled = clear
        self.k_clear_max = int(k_clear_max)                   # 覆盖定位区域的 20m 圆最多补几个
        self.inline_r_min = float(inline_r_min)               # 前向顺路半径下界 / m
        self.inline_r_max = float(inline_r_max)               # 前向顺路半径上界 / m
        self.inline_near_r = float(inline_near_r)             # 近距顺路清除半径 / m，相对当前点
        self.inline_max_mec_r = float(inline_max_mec_r)       # 参与顺路的区域最大 mec 半径 / m
        self.inline_radius_max = (None if inline_radius_max is None
                                  else float(inline_radius_max))  # 兼容旧参数：覆盖前向上界 / m
        self.rotate = bool(rotate)                           # 起始扫描后是否旋转覆盖圆布局
        # 过程日志用 "w"：每局开头重写，于是整份日志只描述最新一局。
        # 原先用 "a" 追加，跨局、跨运行无限累积，几轮演练后文件里混着几百局的内容难以查阅。
        self._logfile = open(logfile, "w", encoding="utf-8") if logfile else None
        self.obs: dict[int, list[Obs]] = defaultdict(list)
        self.meas: dict[int, list[Meas]] = defaultdict(list)   # 全部测量，含 no_signal
        self.n_skip = 0                                        # 判定必无信号而跳过的测量次数
        self.n_inline_fail = 0                                 # 巡视途中顺路试清白跑的次数
        self.initial_scan: dict[str, Any] = {}                 # 起始全频道扫描的统计
        self.actions: list[dict[str, Any]] = []                # 逐次动作记录，供轨迹图/轨迹表
        self.regions: dict[int, ProbRegion] = {}
        self.tracks: dict[int, dict[str, Any]] = {}      # 逐频道的定位/清除档案
        self.cleared: set = set()
        self.pos = np.zeros(2)
        self.vt = 0.0
        self.n_measure = 0
        self.n_clear = 0
        self.episode = episode
        self.deadline = float("inf")
        self.waypoint_stats: list[dict[str, Any]] = []   # 逐圆心扫描统计
        self.travel_m = 0.0                              # 实际走过的里程 / m
        self.stage = "survey"                            # 观测所处阶段，survey / refine
        self.plan: CoverPlan | None = None            # 本局实际使用的覆盖圆方案，含旋转
        self.rotation_deg = 0.0                          # 布局旋转角 / 度，起始扫描后确定
        self.dense_dir_deg: float | None = None       # 源最密集的扇区中心方位 / 度
        self.n_face_scanned = 0                          # 起始扫描听到的源个数，选向依据
        self.survey_order_used: list[int] = []           # 本局实际的巡视顺序
        # 逐步扫描记录：一步 = 机器狗停在某处把当前该测的频道测一遍，起点全频道扫描与各巡视
        # 站都算。这里只登记、不绘图，绘图统一在 /exit 之后做，不占用现实时间预算。
        self.scan_steps: list[dict[str, Any]] = []
        self._cur_step: dict[str, Any] | None = None

    # ---- 日志与原子动作见 common.actions：log / close / clear / _note / _polar_deg 等 ----
    def measure(self, x: float, y: float, channel: int) -> dict:
        """测向；direction 时记录示向度并同步进该频道的定位区域"""
        x, y = float(x), float(y)
        r = self.sim.measure(x, y, channel)
        if not r.get("accepted"):
            raise RuntimeError(f"/measure 被拒绝：{r}")
        self.travel_m += dist(self.pos, (x, y))
        self.pos, self.vt = np.array([x, y]), float(r["virtual_time_s"])
        self.n_measure += 1
        outcome = r.get("measure_result", "no_signal")
        reg = self.region(channel)          # 首次创建会重放此前测量，所以必须先取再追加
        m = Meas(channel, x, y, outcome,
                 float(r["svd_deg"]) if outcome == "direction" else None, self.stage)
        self.meas[channel].append(m)
        self._apply_meas(reg, m)
        if outcome == "direction":
            self.obs[channel].append(Obs(channel, x, y, m.theta, self.stage))
        self._note("measure", x, y, channel, outcome=outcome, theta=m.theta)
        if self._cur_step is not None:
            self._cur_step["measures"].append({"channel": int(channel), "outcome": outcome,
                                              "theta": None if m.theta is None
                                              else float(m.theta)})
        return r

    def _provable_no_signal(self, channel: int, at: Sequence[float]) -> bool:
        """能否证明"在 at 处测 channel 必然无信号"，从而省掉这次测量

        可能源集合是真实源位置的超集，所以"区域到 at 的最小距离 > 1500 m"意味着真源到 at 的
        距离也 > 1500 m ≥ 有效接收半径，必然收不到信号。这次测量不会带来任何新信息，直接跳过。
        """
        reg = self.regions.get(channel)
        return reg is not None and reg.provably_out_of_reach(at, RECEIVE_MAX)

    def _end_scan_step(self, counts: dict[str, int]) -> None:
        """收尾一步扫描：填统计与本步结束时的状态快照，然后登记，供收尾统一出图"""
        step = self._cur_step
        self._cur_step = None
        if step is None:
            return
        step["counts"] = dict(counts)
        step["virtual_time_s"] = round(self.vt, 3)
        step["travel_m"] = round(self.travel_m, 2)
        step["cleared"] = sorted(self.cleared)
        # 到本步为止的行驶路径，起点也画上，给出"路线走到哪了"的空间上下文
        step["path"] = [(0.0, 0.0)] + [(a["x"], a["y"]) for a in self.actions]
        # 本步结束时的可能源区域轮廓。T3 的估计形态就是多边形，不另算 σ 圆。
        # 顶点不足 3 个的多边形画不出来，跳过就行。画图是辅助手段，不影响任何决策。
        step["regions"] = {ch: verts for ch, reg in sorted(self.regions.items())
                           if len(verts := list(reg.vertices)) >= 3}
        self.scan_steps.append(step)

    def region(self, channel: int) -> ProbRegion:
        """取该频道的"可能源集合"，也就是 ProbRegion

        惰性创建：首次访问时把该频道此前的全部测量，连 no_signal 一起，作为硬约束补进去；之后每次
        measure() 直接叠加新约束。ProbRegion 对楔形交与圆盘约束都做增量缓存，所以反复读直径
        或最小覆盖圆只会补上新增的那几条。
        """
        if channel not in self.regions:
            reg = ProbRegion(err=BEARING_ERROR_DEG, radius=REGION_RADIUS, sides=CLIP_SIDES)
            for m in self.meas.get(channel, ()):
                self._apply_meas(reg, m)
            self.regions[channel] = reg
        return self.regions[channel]

    @staticmethod
    def _apply_meas(reg: ProbRegion, m: Meas) -> None:
        """把一次测量的结果转成硬约束叠加到可能源集合上，依据见 ProbRegion 的类文档"""
        if m.outcome == "direction":
            reg.add_node(m.x, m.y, float(m.theta))
            reg.add_inside(m.x, m.y, RECEIVE_MAX)      # 收得到，源在接收半径上限之内
        elif m.outcome == "near":
            reg.add_inside(m.x, m.y, NEAR_RADIUS)      # 5 m 内，位置几乎确定
        elif m.outcome == "no_signal":
            reg.add_outside(m.x, m.y, COVER_RADIUS)    # 收不到，源在接收半径下界之外

    def diameter(self, channel: int) -> float:
        """该频道当前定位区域的直径 / m；0 表示区域为空或退化"""
        return float(self.region(channel).diameter)

    def _precise(self, channel: int) -> bool:
        """判据：可能源集合的最小覆盖圆半径 + 裁剪误差 < 20 m，也就是清除半径

        注意它不是清除门槛，只回答"估计是否已经够准、可以停止补测"。清除一律由"走到估计点
        先试一次 /clear"决定，见 _nearby_try_clear。

        这里用最小覆盖圆半径，而不是"直径 < 40 m"。直径判据的依据是"凸区域的最小覆盖圆半径
        ≤ 直径/2"，而叠加 no_signal 禁区后区域可能非凸、甚至裂成多块，凸性不再成立。直接看
        覆盖圆半径对任何集合都成立：真源必在区域内 ⊆ 覆盖圆内，所以走到圆心必在 20 m 以内。
        """
        mec = self.region(channel).enclosing_circle
        return mec is not None and mec[2] + CLIP_ERR < CLEAR_RADIUS

    # ---- 阶段 1：巡视扫描 ----
    def _active_channels(self) -> list[int]:
        """仍需测向的频道：未清除、示向度条数未达上限、且定位还不够准

        第三条是关键。叠加 no_signal 禁区与接收半径环带后，很多频道两条射线就已把可能源集合
        压到覆盖圆半径 < 20 m，此时再测纯属浪费，每站 5 s 测向加上可能的 1 s 切换。实测前 3
        站每站都要把 20 个频道全测一遍，而每站只有 4~6 次能测出方向。
        """
        return [c for c in CHANNELS
                if c not in self.cleared
                and len(self.obs.get(c, ())) < OBS_CAP
                and not self._precise(c)]

    def _sweep(self, channels: Sequence[int], at: Sequence[float],
               label: str | None = None, index: int = 0) -> dict[str, int]:
        """在 at 处按频道号升序逐频道测向，升序可省频道切换时间；near 就地清除

        传入 `label` 时把这一步登记进 `scan_steps`，收尾逐步骤出"扫描结果图"。
        """
        if label is not None:
            self._begin_scan_step(index, label, at, len(channels))
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
        if label is not None:
            self._end_scan_step(counts)
        return counts

    def survey(self, order: Sequence[int]) -> None:
        """依次移动到各圆心并扫描，阶段一的主体

        第一步是在出发点原点做一次全频道扫描，里程 0 m，详见下面。一次把 20 个频道全测
        一遍，成本 20 次测向，约 20×5 s 加 19×1 s 切换约 119 s，换来的是"哪些频道在 1000 m
        内"这一批最便宜的信息。实测平均能直接锚定 6 个源的方向，同时把其余频道标记为"源在
        1000 m 之外"，这条对后续跳过测量与区域收缩都有用。扫描结束后立刻用这批方位决定后续
        巡视的绕向与落脚点。
        """
        order = list(order)
        waypoints = self.plan.waypoints

        # ---- 第 0 站：出发点原点的全频道扫描 ----
        # 机器狗就在原点，这一步的里程为 0，却是整局最划算的一批信息：一次扫完 20 个频道，
        # 约 20×5 s 测向加 19×1 s 切换共 ≈ 119 s，就能拿到"哪些频道在 1000 m 内"以及它们的
        # 方向。默认的 7 点布局不把原点当作巡视站：原点是覆盖效率最低的位置，占了它反而使
        # 里程变长。所以这里把它当免费的第 0 站单独扫一遍，之后才出发巡视。
        origin = (0.0, 0.0)
        # 若某个巡视站本身就在出发点，也就是经典六边形族的 1 号站，则不必单独扫描。巡回到那里时
        # 自然会扫；重复扫同一位置只会再花 119 s，得到的结果一模一样，本题测量是确定性的。
        at_home = any(float(np.hypot(*waypoints[i])) <= 1.0 for i in range(len(waypoints)))
        home = [] if at_home else self._active_channels()
        self._orient_after_first_scan = bool(at_home)
        if at_home:
            self.log("阶段1 起始全频道扫描：某巡视站就在出发点，将在巡视到该站时扫描，"
                     "并在该站决定巡视绕向与落脚点，不另扫")
        if home:
            self.log(f"阶段1 起始全频道扫描：在出发点 (0.0, 0.0) 扫描 {len(home)} 个频道"
                     f"，里程 0 m")
            counts0 = self._sweep(home, origin, label="起点全频道扫描", index=0)
            self.initial_scan = dict(counts0, n_channels=len(home))
            self.log(f"    有示向度 {counts0['direction']}、无信号 {counts0['no_signal']}、"
                     f"近距清除 {counts0['near']}、判定必无信号而跳过 {counts0['skip']}")
            self.log(f"    [起始全频道扫描] 一次扫完 {len(home)} 个频道：锚定 "
                     f"{counts0['direction']} 个源的方向，其余 {counts0['no_signal']} 个判定为"
                     f"源在 {COVER_RADIUS:.0f} m 之外")
            self._orient_after_first_scan = False
            bearings = self._bearings_at(origin)
            self.n_face_scanned = len(bearings)
            # 用这批方位联合决定"布局旋转 + 巡视绕向 + 落脚点"，零成本，见 _orient_route
            self._orient_route(order, origin, bearings)
            waypoints = self.plan.waypoints

        self.log(f"阶段1 巡视扫描：依次访问 {len(order)} 个圆心"
                 f"，顺序 {' → '.join(str(i) for i in order)}")
        # 起点段原点→第一站的顺路清除已按用户要求删除，2026-09-12 定的：顺路清除只在巡视站
        # 到站后做，见下方循环内。
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
            # label 会进扫描图的文件名，保持原样别动
            counts = self._sweep(active, wp, label=f"巡视站 {step_i}（圆心 {idx}）",
                                 index=step_i)
            self.waypoint_stats.append({
                "waypoint": int(idx), "x": float(wp[0]), "y": float(wp[1]),
                "n_channels": len(active), **counts,
                "virtual_time_s": round(self.vt, 3), "travel_m": round(self.travel_m, 2),
            })
            self.log(f"    有示向度 {counts['direction']}、无信号 {counts['no_signal']}、"
                     f"近距清除 {counts['near']}、判定必无信号而跳过 {counts['skip']}，"
                     f"累计里程 {self.travel_m:.0f} m，虚拟时刻 {self.vt:.0f} s")
            if getattr(self, "_orient_after_first_scan", False):
                # 六边形族下出发点本身就是巡视站：首次扫描发生在本站，定向决策也在这里做。
                # 这一步已按"巡视站"登记，标签沿用就行。
                self._orient_after_first_scan = False
                bearings = self._bearings_at(wp)
                self.n_face_scanned = len(bearings)
                self._orient_route(order, wp, bearings)
                waypoints = self.plan.waypoints
            # 到站后顺路清除。用户 2026-09-12 改判据，起点段与站点间后向都已删除：
            #   1) 近距清：距本站 inline_near_r 以内的估计点全清，不分方位，见 _nearby_clear；
            #   2) 前向清：去下一站方向的方位扇区内、且估计点距原点落在
            #      [inline_r_min, inline_r_max] 之间的点，
            #      见 _inline_clear。
            self._nearby_clear()
            if step_i < len(order):
                self._inline_clear(wp, waypoints[order[step_i]])
        self.survey_order_used = list(order)
        self.log(f"阶段1 完成：里程 {self.travel_m:.0f} m，虚拟时刻 {self.vt:.0f} s，"
                 f"累计示向度 {sum(len(v) for v in self.obs.values())} 条，"
                 f"途中顺路清除 {len(self.cleared)} 个")

    def _bearings_at(self, at: Sequence[float]) -> list[float]:
        """在 at 处测得的示向度（度），也就是这个点听到的源各自在哪个方向"""
        out = []
        for ml in self.meas.values():
            for m in ml:
                if m.theta is not None and dist((m.x, m.y), at) <= 1e-6:
                    out.append(float(m.theta))
        return out

    def _orient_route(self, order: list[int], origin: Sequence[float],
                      bearings: Sequence[float]) -> None:
        """用起始扫描听到的方位，联合决定"巡视路线绕向 + 布局旋转 + 落脚点"，零成本

        自由度在这儿。把整个 7 点布局绕原点整体旋转，覆盖条件不变，它只取决于点间距离和点到
        原点的距离；巡回路径长度也不变，它只取决于点间距离。所以"7 个站分别朝向哪"是纯自由的。

        决策，用户指定：
          1. 最密集方向 θ* = "角度频率最高的 60° 区间中点"，见 dense_sector_rotation，众数赢；
          2. 布局旋转使第一个巡视点 path[0] 正对"θ* 顺时针旋转 90°"的方向。atan2 坐标系下
             顺时针就是角度 −90°，所以目标方位 = θ* − 90°；
          3. 巡视顺序仍由"Held-Karp 最短开放路径 + 终点靠源密集区径向"选取，此轮不变。

        为什么这样摆。旋转是零成本自由度；把起点对准密集方向旁 90°、而非正对，是用户对"第一
        阶段先扫外围、把密簇留给之后处理"的排布偏好。覆盖与路径长度均不受旋转影响。

        `--no-rotate` 时不做旋转，仅在原朝向下选终点。
        """
        wps = self.plan.waypoints
        n = len(wps)
        dense = dense_sector_rotation(list(bearings)) if bearings else None
        self.n_face_scanned = len(bearings)
        if dense is None:
            self.log(f"    [朝向] 起始扫描在原点未听到任何源（听到 {len(bearings)} 条），"
                     f"保持现有布局朝向与顺序")
            return
        # 源密集方向的方位估计：方位取 θ*，径向距离未知，取接收半径区间中点
        rho_est = RECEIVE_MID
        self.rotation_deg = math.degrees(dense % (2.0 * math.pi))
        self.dense_dir_deg = self.rotation_deg

        D = dist_matrix(wps, np.asarray(origin, dtype=float))
        by_end = exact_open_by_end(n, D)
        if not by_end:
            return
        L_min = by_end[0][1]
        best = None
        for end, length, path in by_end:
            rho_j = float(np.hypot(*wps[end]))
            score = (length - L_min) + abs(rho_est - rho_j)
            if best is None or score < best[0] - 1e-9:
                best = (score, end, length, path)
        _, end, length, path = best

        if self.rotate:
            # 把"第一个巡视点"转到 θ* 顺时针旋转 90° 的方向上。
            # atan2 坐标系 x 右、y 上、逆时针为正：顺时针 90° 就是角度 −90°，目标方位 = θ* − π/2。
            first = int(path[0])
            cur = math.atan2(float(wps[first][1]), float(wps[first][0]))
            target = (dense - math.pi / 2.0) % (2.0 * math.pi)
            self.plan = self.plan.rotated(target - cur)
            wps = self.plan.waypoints
            self.rotation_deg = math.degrees((target - cur) % (2.0 * math.pi))
        else:
            rho_j = float(np.hypot(*wps[end]))
            self.log(f"    [朝向] --no-rotate：不做布局旋转，仅在现有朝向下选终点")

        first = int(path[0])
        # 巡视是否会因布局旋转而偏移：旋转后第一个站点正对最密集方向
        rotate_note = (f"布局旋转 {self.rotation_deg:.1f}° 使" if self.rotate
                       else "未旋转（--no-rotate），")
        self.log(f"    [朝向] 起始扫描听到 {len(bearings)} 个源，最密集方向 "
                 f"{math.degrees(dense % (2.0 * math.pi)):.1f}°，其顺时针 90° 处 = "
                 f"{math.degrees((dense - math.pi / 2.0) % (2.0 * math.pi)):.1f}°；"
                 f"{rotate_note}"
                 f"第一个巡视点站点{first} 正对该方位。巡视终点为站点{end}"
                 f"（终点路径长 {length:.0f} m，比最短路径多 {length - L_min:.0f} m）")
        order[:] = list(path)
        self.survey_order_planned = list(path)
        self._survey_path_len = length

    @staticmethod
    def _at_origin(p: Sequence[float]) -> bool:
        """是否近似位于区域圆心也就是原点，起点 (0,0) 要特判"""
        return math.hypot(p[0], p[1]) <= 1.0

    def _in_azimuth_arc(self, at: Sequence[float], next_wp: Sequence[float],
                        est: Sequence[float]) -> tuple[float, float, float] | None:
        """估计点 est 是否落在「本站→圆心」与「下一站→圆心」两条连线之间的扇区内

        判据以区域圆心为参照的极角扇区，半径限定在 [INLINE_R_MIN, INLINE_R_MAX]：
          · 方位：est 与圆心的连线方向 th_est 位于 th_at 与 th_next 夹出的较短弧上，也就是
                ang_diff(th_est, th_at) + ang_diff(th_est, th_next) == ang_diff(th_at, th_next)
            三角不等式取等。ang_diff 取值 [0,180]，所以"较短弧"是唯一候选；
          · 半径：INLINE_R_MIN ≤ r_est ≤ r_max。用户 2026-09-12 指定前向顺路只清距原点
            下界承接原"排除近圆心点"的语义并入判据本身，上界比原来的"两站所在半径"更宽。
            两个边界按定位方差最优选出来，见 config 里 INLINE_R_MIN / INLINE_R_MAX 的注释。
            上界可用 inline_r_max，或兼容参数 inline_radius_max 覆盖，也就是 CLI 的 --inline-r-max。

        反侧点就是 th_est 与 th_at 差 180° 的点，会被三角不等式排除。这里必须用角度比较而
        不能用 sin：|sin 180°| = 0，反侧点会被误判成同向。

        返回在短弧上的进度 0~1、距圆心半径、与 th_at 的角度差；并列时按频道号定序。
        """
        r_est = math.hypot(est[0], est[1])
        if r_est < self.inline_r_min - 1e-6:
            return None                              # 距原点太近，不在前向清的范围
        r_max = (self.inline_radius_max if self.inline_radius_max is not None
                 else self.inline_r_max)
        if r_est > r_max + 1e-6:
            return None                              # 超出半径上界
        th_at = self._polar_deg(at)
        th_next = self._polar_deg(next_wp)
        th_est = self._polar_deg(est)
        d = ang_diff(th_at, th_next)
        a1 = ang_diff(th_est, th_at)
        if d <= 1e-9:                                # 两端点同方位：退化为单方向
            if a1 > 1e-9:
                return None
            return (0.0, r_est, 0.0)
        if a1 + ang_diff(th_est, th_next) > d + 1e-9:
            return None
        return (a1 / d, r_est, a1)

    def _cover_clear(self, channel: int, why: str) -> int:
        """在覆盖该频道定位区域的半径 20 m 圆的圆心处清除，顺路清除的新清法

        用户 2026-09-12 指定，清法改为在区域的覆盖圆圆心处清。用半径 `CLEAR_RADIUS` 也就是
        20 m 的圆去盖定位区域；一个圆盖不住，最小覆盖圆半径 > 20 m，就按贪心补选多个 20 m
        圆，复用阶段二 _k_cover_points 的手法。第 1 个圆取区域最小覆盖圆的圆心，逐个在圆心
        处 /clear，命中就停。圆心顺序按"距当前位置由近及远"，最近邻，命中概率最高的先试。

        返回清除成功的圆心数，0 或 1。命中之后频道就进了 cleared，剩下的圆心不再试。
        """
        region = self.region(channel)
        mec = region.enclosing_circle
        if mec is None or region.region.is_empty:
            return 0
        center = (float(mec[0]), float(mec[1]))
        extra, leftover = self._k_cover_points(channel, self.k_clear_max)
        if leftover > 1e-9:
            # k_clear_max 个半径 20 m 的圆盖不满区域，区域太大：多清几个命中率仍趋零，纯白跑。
            # 此时只清区域覆盖圆圆心也就是 mec 圆心，碰一次运气，交给阶段二处理。
            extra = []
        pts = [center] + list(extra)                  # 第 1 个 = 区域覆盖圆圆心，也就是 mec 圆心
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
                    "method": "survey-inline", "clear_point": [x, y],
                    "n_cover_clear": len(order)})
                self.log(f"    [顺路清除] 频道{channel} @ ({x:.1f}, {y:.1f}) 命中"
                         f"（{why}，第 {i}/{len(order)} 个覆盖圆）")
                return 1
            self.n_inline_fail += 1
            self.log(f"    [顺路清除] 频道{channel} @ ({x:.1f}, {y:.1f}) 未命中"
                     f"（{why}，留到阶段二再处理）")
        return 0

    def _nearby_clear(self, radius: float | None = None) -> int:
        """每站到站后的近距顺路清除：距当前点 radius 以内的估计点全清，缺省用 self.inline_near_r

        用户 2026-09-12 指定，替代原来的"站点间后向"：不分方位，只要估计点距当前点 ≤ self.inline_near_r
        就清，清法同为"覆盖圆圆心处清、多个圆就清多次"，见 _cover_clear。顺序按距离由近到远。
        只有区域小的频道才参与，也就是最小覆盖圆半径 ≤ self.inline_max_mec_r。大区域命中率低，
        不值得顺路试，交给阶段二。返回本段清除的频道数。
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
                continue                              # 区域为空/退化，或区域太大不参与顺路
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
        """巡视途中的前向顺路清除：把去下一站方向扇区内的估计点顺路清掉

        判据，用户 2026-09-12 改：以区域圆心也就是原点为参照的方位扇区，见 _in_azimuth_arc。
        估计点与圆心的连线方向落在「本站→圆心」与「下一站→圆心」两条连线之间，取较短弧，且
        估计点距原点半径在 [INLINE_R_MIN, INLINE_R_MAX] 之间。不设"估计可信度"门槛，扇区内都要清；也没有起点段
        兜底，起点段顺路清除已按用户要求删除，本站恒为巡视站、at 非原点；也没有独立的 600 m
        排除规则，其语义由半径下界 600 并入判据本身。只清区域小的频道：定位区域最小覆盖圆
        半径 ≤ INLINE_MAX_MEC_R 才参与顺路，大区域命中率低、不值得绕路试，留给阶段二
        专程处理。

        顺路的成本：极角扇区内半径不限，清点要走到估计点再回来。但只要这些源阶段二反正要清，
        顺路清掉省的就是"从别处专程跑一趟"的里程。清点本身固定花 5 s 命中或 3 s 未命中，且
        命中后该频道后续各站都不再测量。

        清法见 _cover_clear：在覆盖定位区域的半径 20 m 圆的圆心处清，区域大需多个圆就逐个清，
        命中就停。
        """
        if not self.clear_enabled or self._out_of_time():
            return 0
        picked: list[tuple[float, int, Sequence[float]]] = []
        n_too_big = 0                                  # 区域太大不参与顺路的频道数（INLINE_MAX_MEC_R）
        for ch in sorted(self.obs):
            if ch in self.cleared:
                continue
            mec = self.region(ch).enclosing_circle
            if mec is None:
                continue                              # 区域为空/退化：连估计点都没有
            if mec[2] > self.inline_max_mec_r:
                n_too_big += 1
                continue                              # 区域太大：命中率低，不参与顺路（阶段二处理）
            est = (float(mec[0]), float(mec[1]))
            got = self._in_azimuth_arc(at, next_wp, est)
            if got is None:
                continue
            picked.append((got[0], ch, est))
        if not picked:
            return 0
        picked.sort(key=lambda p: (p[0], p[1]))       # 沿扇区方位由近到远依次清
        th_a = self._polar_deg(at)
        th_b = self._polar_deg(next_wp)
        r_hi = self.inline_radius_max if self.inline_radius_max is not None else self.inline_r_max
        self.log(f"    [顺路清除] 本站 ({at[0]:.0f}, {at[1]:.0f}) → 下一站"
                 f" ({next_wp[0]:.0f}, {next_wp[1]:.0f})：与圆心的连线方向在 {th_a:.0f}° ~ "
                 f"{th_b:.0f}° 之间（方位扇区 {ang_diff(th_a, th_b):.0f}°）、半径 "
                 f"{self.inline_r_min:.0f}~{r_hi:.0f} m、区域覆盖圆半径 ≤ {self.inline_max_mec_r:.0f} m，"
                 f"有 {len(picked)} 个估计点" + (f"（跳过 {n_too_big} 个区域过大的频道）"
                                                if n_too_big else ""))
        n = 0
        for _key, ch, _est in picked:
            if self._out_of_time():
                break
            n += self._cover_clear(ch, "前向扇区")
        if n:
            self.log(f"    本段顺路清除 {n} 个，累计已清 {len(self.cleared)} 个")
        return n

    # ---- 阶段 2a：巡视后的定位诊断，只记录，不改变处理流程 ----
    def diagnose(self) -> dict[str, int]:
        """记录每个频道巡视后的定位区域直径与有界性，并统计"估计已够准"的个数

        这是纯诊断。清除流程对所有频道一视同仁，都先就近试清，不按直径分叉。
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

    def _nearest_order(self, channels: Sequence[int]) -> list[int]:
        """确定性的访问顺序：直接求精确最短开放路径，用 Held-Karp 动态规划

        访问点取各频道定位区域的最小覆盖圆圆心，也就是清缺点，起点为机器狗当前位置。实测
        n=10~16、300 个随机算例下，"最近邻 + 2-opt"平均比精确解长 2.2%、最坏长 20%，而精确解
        在本规模下只需 15 ms~1.3 s，一次定序只算一次，所以直接用精确解。n > 16 时公共算子自动
        降级为最近邻 + 2-opt，本题不会出现，频道一共 20 个。里程占虚拟总时间的八成以上，这一
        项是纯改进。

        确定性：Held-Karp 的遍历顺序固定、并列取编号小者，所以同一输入必得同一顺序。原先的
        2-opt 也是确定性的，两者都不依赖随机数。

        注意下标与频道号的映射：公共算子返回的是数组下标，而本类的调用方按频道号使用顺序。
        `channels` 由调用方保证升序，来自 `sorted(self.obs)` 过滤；下标并列时公共算子取小下标，
        所以映射回来后与"并列取小频道号"完全等价。
        """
        pts = self._clear_points(channels)
        idx = exact_open_order(len(channels), dist_matrix(pts, self.pos))
        return [channels[int(i)] for i in idx]

    # ---- 阶段 2b：按文献准则补测缩小定位区域 ----
    def refine(self, channel: int) -> int:
        """补测直到定位区域直径 < 40 m，或达到轮次/候选上限；返回实际补测次数"""
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
        """兜底：沿最新实测示向度以 HOMING_STEP 步长逼近，直到清除成功，确定性

        步数按"当前位置到定位区域最远顶点的距离"自适应，下限 HOMING_MAX、上限 HOMING_CAP。
        起点离源很远时不能只走固定几步就放弃。实测踩过这个坑：起点 636 m 远、只走
        24×16 = 384 m 就停手，把本可清掉的源漏掉。
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

    def _try_clear(self, channel: int, tag: str) -> str | None:
        """走到定位区域的最小覆盖圆圆心，也就是当前位置估计，就地 /clear 一次

        成功返回 tag，失败返回 None。这是全局唯一的清除位置：区域最小覆盖圆半径 < 20 m 时它
        必然命中，区域内任一点到圆心 ≤ 该半径；估计略差时也常常命中，失败只花 3 s。
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

    def _nearby_try_clear(self, channel: int) -> str | None:
        """就近试清：走到当前估计点，直接 /clear 一次；未命中则就地复测

        为什么"试"而不是"先判断能不能清"。清除半径只有 20 m，走到源的近处是清除的必要动作，
        绕不过去。所以到达估计点后顺手试一次 /clear，边际代价只有失败时的 3 s，命中却能省掉
        整轮补测，一轮补测要绕几百米，约 100 s 量级的里程。相比之下，"先把定位区域压到
        直径 < 40 m 再清"要额外花探针去换那个确定性，反而更贵。所以不再用直径判据决定清不
        清，直径只用来决定还要不要继续补测。

        唯一的前提，这也是"就近"之意：估计点不能太离谱，才值得跑过去试。
        ① 定位区域有界，方向由示向度真正约束住，而不是被作业圆域截断后"随便落"；
        ② 最小覆盖圆半径 ≤ TRY_CLEAR_RADIUS，估计够集中。
        两条都不满足就先按文献准则就地补测一次把区域压小，再过去。跑一趟很远的错点比就近补测
        更贵，实测放开 ① 后平均里程多 275 m。

        未命中时就地复测也不是白花的：该点是区域内离源最近的位置，按 Fisher 信息口径权重最大，
        一条近距离射线往往直接把区域压到 40 m 以内。
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
                        radius: float = CLEAR_RADIUS) -> tuple[list[tuple[float, float]], float]:
        """用"当前估计点已清过一次"为起点，再贪心选 k_extra 个清除点，使半径 radius 的圆尽量
        覆盖整个定位区域；返回补充清除点列表和未被覆盖的目标点比例。

        思路是"区域大就多清几次"。清除半径只有 20 m，一个点保证不了命中时就多清几个点。只要
        这几个半径 20 m 的圆把定位区域盖满，逐个清过去必然命中，省掉一轮补测，而补测要绕几百
        米、约 100 s 量级里程。补清一个点的边际代价只有一个点间距的移动（≤ 40 m ≈ 8 s）加失败
        时的 3 s，比补测便宜一个量级。

        点怎么选：在区域内取细网格作为目标点，候选点同样取区域内的网格。估计点已经去过、已经
        失败，它的圆所覆盖的目标点先划掉，然后每轮选"新增覆盖目标点最多"的候选点，贪心最大覆盖，
        与覆盖圆求解里的贪心集合覆盖同一手法。比例 = 0 表示覆盖完整，可以保证命中。
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
        """试清未中后就地"多清几次"：用至多 k_clear_max 个半径 20 m 的圆覆盖定位区域，依次补清

        只在"这几个圆能把区域盖满"时才动手，否则白跑，直接交给补测。顺序按最近邻，从当前位置
        由近及远，命中就停。返回方法标签，覆盖不全或都没命中时返回 None。
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

    def process(self, channel: int) -> None:
        """一个频道的完整处理：走到估计点先试清，未命中再补测缩小、然后再清"""
        rec = self.tracks.setdefault(channel, {})
        if channel in self.cleared:                  # 巡视阶段已就地清除，near
            rec.update({"method": "survey-near", "cleared": True})
            return
        if not self.clear_enabled:                   # 只定位模式：补测到估计够准为止
            if not self._precise(channel):
                self.refine(channel)
            method: str | None = "skipped"
        else:
            method = self._nearby_try_clear(channel)     # ① 就近试清
            if method is None:                           # ② 未命中：多清几次，几个 20 m 圆盖满区域
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
    def run(self, plan: CoverPlan, order: Sequence[int]) -> dict[str, Any]:
        """/enter → 巡视扫描 → 诊断 → 逐频道就近试清，未命中则补测缩小后再清 → /exit"""
        enter = self.sim.enter()
        if not enter.get("accepted"):
            raise RuntimeError(f"/enter 被拒绝：{enter}")
        left = float(enter.get("remaining_real_duration_s", 1200.0))
        self.deadline = time.monotonic() + max(left - SAFETY_MARGIN, 0.0)
        self.log(f"/enter 成功：虚拟时刻 {enter.get('virtual_time_s')} s，"
                 f"现实剩余 {left:.0f} s")
        self.plan = plan
        try:
            self.survey(order)
            if self.clear_enabled:
                self.diagnose()
                # 阶段二的访问顺序：按各频道估计点到当前位置的距离做最近邻 + 2-opt，省里程。
                # 实测"每处理一个就重排"与"一次定序"结果完全相同，补测位移不足以改变最近邻
                # 首元素，所以保留更简单的一次定序。
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
            "rotation_deg": round(self.rotation_deg, 4),
            "dense_dir_deg": (None if self.dense_dir_deg is None
                              else round(self.dense_dir_deg, 4)),
            "n_face_scanned": self.n_face_scanned,
            "survey_order_used": list(self.survey_order_used),
            "n_inline_cleared": sum(1 for r in self.tracks.values()
                                    if r.get("method") == "survey-inline"),
            "n_inline_fail": self.n_inline_fail,
            "n_refined": sum(1 for r in self.tracks.values() if r.get("n_probe")),
            "n_probe": sum(int(r.get("n_probe", 0)) for r in self.tracks.values()),
            "tracks": self.tracks,
        }
