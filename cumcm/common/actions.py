"""机器狗的动作记录、过程日志与"与模拟器交互的基础动作"：问题三、问题四共用。

这三个东西在两题里是**同一份契约**，不该各写一遍：

* **动作记录 schema**（`actions` 里的键）被 `cumcm.common.trajfigure` 的轨迹图与轨迹表按名读取，
  而轨迹图/轨迹表又是公共代码 —— 谁改了键名却忘改另一处，图上就会静默少一类点；
* **一步扫描的记录 schema**（`scan_steps` 里的键）被 `cumcm.common.scanfigure` 的逐步扫描图读取，
  同理；
* **原子动作**（`clear`、`_note`）对 `travel_m`/`vt`/`n_clear`/`cleared`/`tracks` 的更新顺序，
  两题必须一致，否则"里程与虚拟时刻"的口径会在两题之间漂移。

故这里把这些**逐字相同**的实现收成一份混合器，各题 `RobotDog` 只继承：

    class RobotDog(ActionRecorder):
        '''...本题特有的两阶段策略...'''

本模块**不**绑定任何一题的 config：用到阈值/半径的地方（`_provable_no_signal`、`_apply_meas`
等）仍留在各题策略里，那里能直接引用各自的 config 常量。

`_finish_clear` 虽含清除流程，但两题逐字相同（都调子类的 `_try_clear` / `_homing`），故一并放此。
"""

from __future__ import annotations

import math
import time
from typing import Optional, Sequence

import numpy as np

from cumcm.common.geometry import dist

__all__ = ["ActionRecorder"]


class ActionRecorder:
    """机器狗的基础行为：日志与文件句柄、动作登记、与模拟器的原子动作、几何小工具。

    宿主类（各题的 `RobotDog`）须在其 `__init__` 里准备好这些属性：
    `sim`（含 `measure` / `clear` 的模拟器客户端）、`verbose`、`_logfile`、`deadline`、
    `actions`、`scan_steps`、`_cur_step`、`meas`、`regions`、`tracks`、`cleared`、
    `pos`、`vt`、`travel_m`、`n_measure`、`n_clear`、`stage`。
    """

    # ---- 日志 ----
    def log(self, msg: str) -> None:
        """打印一条过程日志（同时写入 `--log` 指定的文件，若有）。"""
        if self.verbose:
            print(msg, flush=True)
        if self._logfile:
            print(msg, file=self._logfile, flush=True)

    def close(self) -> None:
        """关闭日志文件（幂等）。"""
        if self._logfile:
            self._logfile.close()
            self._logfile = None

    def _out_of_time(self) -> bool:
        """是否已到本局的现实时限（到了就尽快收尾，不再开展新动作）。"""
        return time.monotonic() > self.deadline

    # ---- 原子动作 ----
    def clear(self, x: float, y: float, channel: int) -> bool:
        """清除；返回是否成功。

        清除半径只有 20 m，故"走到源附近"本身就是必要动作 —— 各题策略都据此把"到达后顺手
        试一次"当作首选，不设"够不够准"的门槛。
        """
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

    # ---- 动作与逐步扫描的记录 ----
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

    def _begin_scan_step(self, index: int, label: str, at: Sequence[float],
                         n_channels: int) -> None:
        """开始记录一步扫描（见 `scan_steps`）。动作由 measure/clear 自动挂到当前步上。"""
        self._cur_step = {
            "index": int(index), "label": label,
            "x": float(at[0]), "y": float(at[1]), "n_channels": int(n_channels),
            "counts": {}, "measures": [], "clears": [],
            "virtual_time_s": round(self.vt, 3), "travel_m": round(self.travel_m, 2),
        }

    # ---- 几何小工具 ----
    @staticmethod
    def _polar_deg(p: Sequence[float]) -> float:
        """点 p 相对区域圆心（原点）的方位角 / 度，[0, 360)。起点 (0,0) 的方位未定义，
        调用方须先用 _at_origin 排除。

        用 math 而非 numpy 的三角函数：虚拟时钟按 5 m/s 精确复算，浮点末位差异会造成"时钟
        对不上"的假不一致（同类教训见 cumcm.common.geometry.dist）。
        """
        return math.degrees(math.atan2(p[1], p[0])) % 360.0

    def _clear_points(self, channels: Sequence[int]) -> np.ndarray:
        """各频道的清缺点（定位区域最小覆盖圆圆心）；拿不到估计点的退回当前位置。

        返回的数组与 `channels` 同序，故定序算子的下标可直接映射回频道号。
        """
        pts = []
        for c in channels:
            mec = self.region(c).enclosing_circle
            pts.append(np.array([mec[0], mec[1]]) if mec else np.array(self.pos, dtype=float))
        return np.asarray(pts, dtype=float)

    # ---- 清除收尾 ----
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
