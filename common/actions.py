"""动作记录、过程日志与基础动作，问题三、问题四共用一份"""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

from common.geometry import dist

__all__ = ["ActionRecorder"]


class ActionRecorder:
    """机器狗的基础行为：日志与文件句柄、动作登记、与模拟器的原子动作、几何小工具

    宿主类，也就是各题的 `RobotDog`，须在其 `__init__` 里准备好这些属性：
    `sim`，带 `measure` / `clear` 的模拟器客户端，另有 `verbose`、`_logfile`、`deadline`、
    `actions`、`scan_steps`、`_cur_step`、`meas`、`regions`、`tracks`、`cleared`、
    `pos`、`vt`、`travel_m`、`n_measure`、`n_clear`、`stage`。
    """

    # ---- 日志 ----
    def log(self, msg: str) -> None:
        """打印一条过程日志；有 `--log` 指定的文件就同时写进去"""
        if self.verbose:
            print(msg, flush=True)
        if self._logfile:
            print(msg, file=self._logfile, flush=True)

    def close(self) -> None:
        """关闭日志文件，幂等"""
        if self._logfile:
            self._logfile.close()
            self._logfile = None

    def _out_of_time(self) -> bool:
        """是否已到本局的现实时限；到了就尽快收尾，别再开新动作"""
        return time.monotonic() > self.deadline

    # ---- 原子动作 ----
    def clear(self, x: float, y: float, channel: int) -> bool:
        """在该点清除，返回是否成功

        清除半径只有 20 m，走到源附近本身就是必要动作，所以各题策略都把"到达后顺手试一次"
        当首选，不设"够不够准"的门槛。
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
              outcome: str | None = None, theta: float | None = None) -> None:
        """登记一次动作，/measure 或 /clear，供逐局轨迹图与轨迹表使用

        记的是动作点，也就是机器狗实际到达的坐标，与日志逐点对应。画图和落盘都在 /exit
        之后进行，不占用现实时间预算，也不影响任何实时决策。键名被 `common.trajfigure`
        和 `common.scanfigure` 按名字读取，改名要同步这两处。
        """
        self.actions.append({
            "seq": len(self.actions), "kind": kind, "stage": self.stage,
            "x": float(x), "y": float(y), "channel": int(channel),
            "outcome": outcome, "theta": theta,
            "virtual_time_s": round(self.vt, 3), "travel_m": round(self.travel_m, 2),
        })

    def _begin_scan_step(self, index: int, label: str, at: Sequence[float],
                         n_channels: int) -> None:
        """开始记录一步扫描，字段见 `scan_steps`。动作由 measure/clear 自动挂到当前步上"""
        self._cur_step = {
            "index": int(index), "label": label,
            "x": float(at[0]), "y": float(at[1]), "n_channels": int(n_channels),
            "counts": {}, "measures": [], "clears": [],
            "virtual_time_s": round(self.vt, 3), "travel_m": round(self.travel_m, 2),
        }

    # ---- 几何小工具 ----
    @staticmethod
    def _polar_deg(p: Sequence[float]) -> float:
        """点 p 相对区域圆心，也就是原点的方位角 / 度，[0, 360)。起点 (0,0) 的方位没有定义，
        调用方得先用 _at_origin 把它排除掉

        用 math 而不是 numpy 的三角函数：虚拟时钟按 5 m/s 精确复算，浮点末位差一点就会造出
        "时钟对不上"的假不一致。同类教训见 common.geometry.dist。
        """
        return math.degrees(math.atan2(p[1], p[0])) % 360.0

    def _clear_points(self, channels: Sequence[int]) -> np.ndarray:
        """各频道的清缺点，取定位区域最小覆盖圆的圆心；拿不到估计点的就退回当前位置

        返回的数组与 `channels` 同序，定序算子的下标可以直接映射回频道号。
        """
        pts = []
        for c in channels:
            mec = self.region(c).enclosing_circle
            pts.append(np.array([mec[0], mec[1]]) if mec else np.array(self.pos, dtype=float))
        return np.asarray(pts, dtype=float)

    # ---- 清除收尾 ----
    def _finish_clear(self, channel: int) -> str:
        """补测之后的收尾：先到新的估计点试清，不成就在地儿复测，最后兜底沿示向度逼近"""
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
