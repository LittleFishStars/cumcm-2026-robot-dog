"""模拟器 HTTP+JSON 接口层（题目附件 2 的 4 条指令）与其调用日志。

包含三件东西，从薄到厚：

* `Simulator`    —— 最薄的一层：补齐 arena_id/robot_id、生成 request_id、POST 并解析 JSON。
* `ApiLog`       —— 把每次调用（请求参数 + 原始响应）写成 JSONL，可与模拟器日志逐行对照。
* `RecordedSim`  —— 记录代理：原样转发 4 个接口，并把每次调用交给 ApiLog 落盘。

    from cumcm.common.sim_client import Simulator

    sim = Simulator()                        # 默认队号 202614023005，地址 http://127.0.0.1:2026
    sim.enter()
    r = sim.measure(300, 400, 1)             # r["measure_result"]: no_signal / near / direction
    if r["measure_result"] == "direction":
        r["svd_deg"]                         # 示向度，含 ±1° 误差
    sim.clear(300, 0, 3)                     # r["clear_result"]: success / no_target_in_range
    sim.exit()

协议要点：HTTP 错误由 urllib 抛 HTTPError，业务拒绝（HTTP 200 但 accepted=false）不抛异常，
由返回值里的 accepted 判断。网络中断后重试同一动作时，传入原 request_id 复用原请求。

原 sim_api.py 保留为兼容垫片（`sim_api.Simulator` 等名字照旧可用），故 T1/T3 之外的既有脚本
不必改动。
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO

__all__ = ["ROBOT_ID", "BASE_URL", "API_LOG_NAME", "Simulator", "ApiLog", "RecordedSim",
           "api_brief", "ms_since", "api_log"]

ROBOT_ID = "202614023005"                   # 参赛队号，必须与模拟器登录的队号一致
BASE_URL = "http://127.0.0.1:2026"          # 官方默认地址，模拟器只监听本机回环
API_LOG_NAME = "api_calls.jsonl"            # 接口调用日志文件名（落在 --save-dir 下）


class Simulator:
    """模拟器接口的薄封装：4 个方法直接返回 JSON 响应（dict）。"""

    def __init__(self, robot_id: str = ROBOT_ID, base_url: str = BASE_URL,
                 timeout: float = 5.0) -> None:
        """初始化模拟器接口封装

        Args:
            robot_id: 参赛队号，必须与模拟器登录的队号一致
            base_url: 模拟器地址（只监听本机回环）
            timeout: 单次 HTTP 请求超时 / s
        """
        self.robot_id = robot_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._seq = 0

    def _post(self, path: str, request_id: str | None = None, **fields: Any) -> dict:
        """POST 一个接口并解析返回的 JSON

        Args:
            path: 接口路径，如 "/measure"
            request_id: 断线重试时传入原 request_id 以复用原请求；None 表示按序号新生成
            **fields: 该接口的业务字段，直接并入请求体

        Returns:
            dict: 模拟器返回的 JSON 响应
        """
        self._seq += 1
        payload = {
            "arena_id": "default",
            "robot_id": self.robot_id,
            "request_id": request_id or f"{path.strip('/')}-{self._seq}",
            **fields,
        }
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def enter(self, request_id: str | None = None) -> dict:
        """进入模拟器（/enter），此后开始占用现实时间预算"""
        return self._post("/enter", request_id)

    def measure(self, x: float, y: float, channel: int,
                request_id: str | None = None) -> dict:
        """在 (x, y) 处测量频道 channel 的信号（/measure）"""
        return self._post("/measure", request_id, position={"x": x, "y": y}, channel=channel)

    def clear(self, x: float, y: float, channel: int,
              request_id: str | None = None) -> dict:
        """在 (x, y) 处对频道 channel 执行清除（/clear）"""
        return self._post("/clear", request_id, position={"x": x, "y": y}, channel=channel)

    def exit(self, request_id: str | None = None) -> dict:
        """退出模拟器（/exit）并结束计时"""
        return self._post("/exit", request_id)


def ms_since(t0: float) -> int:
    """自 t0 起的毫秒耗时。"""
    return int((time.monotonic() - t0) * 1000)


def api_brief(call: str, resp: dict, elapsed_ms: int) -> str:
    """一次调用的结果摘要（终端可读的单行）。"""
    if not resp.get("accepted"):
        return "拒绝 accepted=false"
    tail = f"（{elapsed_ms} ms"
    if resp.get("virtual_time_s") is not None:
        tail += f"，虚拟 {resp['virtual_time_s']} s"
    tail += "）"
    if call == "/measure":
        result = resp.get("measure_result", "?")
        return (f"{result} {resp['svd_deg']}°{tail}" if result == "direction"
                else f"{result}{tail}")
    if call == "/clear":
        return f"{resp.get('clear_result', '?')}{tail}"
    if call == "/enter":
        return f"accepted，现实剩余 {resp.get('remaining_real_duration_s')} s{tail}"
    return f"accepted{tail}"


class ApiLog:
    """把每一次接口调用（请求参数 + 原始响应）写成 JSONL，并按需回显一行摘要。

    模拟器自身有行为日志，但那份记录不归我们掌握；这里留一份自己的痕迹：逐条含
    request_id，可与模拟器日志逐行对照。每次调用后立即 flush，即使中途断连或崩溃，
    已经发生的调用也不会丢。

    **落盘策略（避免一次失败运行抹掉上一次成功的证据链）**：本轮第一次要落盘时才决定打开方式 ——

    * 本轮已经发生过成功调用（正常情形，`/enter` 就是第一条且成功）→ **清空重写**，
      即"一个结果目录 = 最新一次运行"，与直觉一致；
    * 本轮至今全是失败（模拟器没启动、地址敲错而立刻连接失败）→ **追加**，并在记录里标
      `"appended": true`：上一次成功运行的完整日志得以保全，本次失败也留了痕。

    官方测试只有 3 次机会，若把上一次成功的记录交给一次敲错地址的运行清空，损失无法挽回。

    路径为 --api-log 指定的文件，或 <save-dir>/api_calls.jsonl。
    """

    def __init__(self, path: Path, echo: Callable[[str], None] | None = None) -> None:
        """初始化接口日志

        Args:
            path: 日志文件路径（JSONL，一行一次调用）
            echo: 每条记录的单行摘要回调；None 表示不回显
        """
        self.path = Path(path)
        self._fh: TextIO | None = None
        self._echo = echo
        self._t0 = time.monotonic()
        self.counts: dict[str, int] = defaultdict(int)

    def elapsed(self) -> float:
        """自日志建立起的秒数（单调时钟），用于记录各次调用的相对时刻。"""
        return time.monotonic() - self._t0

    def _ensure_open(self, ok: bool, rec: dict[str, Any]) -> TextIO:
        """本轮第一次落盘时决定打开方式（说明见类文档：成功过就重写，纯失败则追加）。"""
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not ok and self.path.exists() and self.path.stat().st_size > 0:
                self._fh = self.path.open("a", encoding="utf-8")
                rec["appended"] = True          # 标记"这一条不与上一次运行同批"
            else:
                self._fh = self.path.open("w", encoding="utf-8")
        return self._fh

    def write(self, rec: dict[str, Any], line: str) -> None:
        """落盘一条调用记录，并把单行摘要交给终端。"""
        fh = self._ensure_open(bool(rec.get("ok")), rec)
        self.counts[rec["call"]] += 1
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        if self._echo:
            self._echo(line)

    def report(self) -> None:
        """打印一行统计，便于与模拟器自己的计数交叉核对。"""
        if self.counts:
            detail = "、".join(f"{call} {n}" for call, n in sorted(self.counts.items()))
            print(f"接口调用日志：共 {sum(self.counts.values())} 次（{detail}）→ {self.path}")

    def close(self) -> None:
        """关闭日志文件句柄（从未落盘过则什么都不做）"""
        if self._fh is not None:
            self._fh.close()
            self._fh = None


@contextmanager
def api_log(save_dir: str, raw: str | None, echo: bool) -> Iterator[ApiLog | None]:
    """接口日志的上下文：进入时建文件，退出时关闭。

    路径优先用 raw（--api-log）；未指定时用 <save-dir>/api_calls.jsonl；显式传空串则关闭日志。
    """
    path = raw if raw is not None else str(Path(save_dir) / API_LOG_NAME)
    log = ApiLog(Path(path), echo=print if echo else None) if path else None
    try:
        yield log
    finally:
        if log is not None:                 # 统计由调用方在尾部统一打印
            log.close()


class RecordedSim:
    """Simulator 的记录代理：原样转发 4 个接口，并把每次调用交给 ApiLog 落盘。

    request_id 由本层生成（`<接口>-<局号>-<序号>`）：既保证跨局不重复，也让我们的日志
    能与模拟器行为日志里的同一条请求对上号。
    """

    def __init__(self, sim: Simulator, log: ApiLog, episode: int = 0) -> None:
        """初始化记录代理

        Args:
            sim: 被代理的 Simulator 实例
            log: 调用落盘用的 ApiLog
            episode: 局号，参与生成 request_id，保证跨局不重复
        """
        self._sim = sim
        self._log = log
        self.episode = episode
        self.seq = 0

    def enter(self) -> dict:
        """记录并转发 /enter 调用"""
        return self._call("/enter", {}, lambda rid: self._sim.enter(request_id=rid))

    def measure(self, x: float, y: float, channel: int) -> dict:
        """记录并转发 /measure 调用"""
        return self._call("/measure", {"x": x, "y": y, "channel": int(channel)},
                          lambda rid: self._sim.measure(x, y, int(channel), request_id=rid))

    def clear(self, x: float, y: float, channel: int) -> dict:
        """记录并转发 /clear 调用"""
        return self._call("/clear", {"x": x, "y": y, "channel": int(channel)},
                          lambda rid: self._sim.clear(x, y, int(channel), request_id=rid))

    def exit(self) -> dict:
        """记录并转发 /exit 调用"""
        return self._call("/exit", {}, lambda rid: self._sim.exit(request_id=rid))

    def _call(self, call: str, params: dict[str, Any], send: Callable[[str], dict]) -> dict:
        """执行一次调用并记录：先落请求与时刻，再按成功/异常分别补全结果。"""
        self.seq += 1
        rid = f"{call.strip('/')}-{self.episode}-{self.seq}"
        rec: dict[str, Any] = {"episode": self.episode, "seq": self.seq, "call": call,
                               "request_id": rid, "at_s": round(self._log.elapsed(), 3),
                               **params}
        where = ("" if "channel" not in rec
                 else f" ({rec['x']:.0f}, {rec['y']:.0f}) 频道{rec['channel']}")
        t0 = time.monotonic()
        try:
            resp = send(rid)
        except Exception as exc:            # 连不上/超时也要留痕，再把异常交回上层
            elapsed = ms_since(t0)
            why = f"{type(exc).__name__}: {exc}"
            self._log.write({**rec, "ok": False, "elapsed_ms": elapsed, "error": why},
                            f"[接口] ep{self.episode} #{self.seq} {call}{where} → 失败 {why} "
                            f"（{elapsed} ms）")
            raise
        rec["elapsed_ms"] = ms_since(t0)
        rec["ok"] = bool(resp.get("accepted"))
        rec["response"] = resp
        self._log.write(rec, f"[接口] ep{self.episode} #{self.seq} {call}{where} → "
                             f"{api_brief(call, resp, rec['elapsed_ms'])}")
        return resp
