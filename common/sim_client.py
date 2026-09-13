"""模拟器 HTTP+JSON 接口层，也就是题目附件 2 的 4 条指令，外加调用日志

里面三个东西，从薄到厚：

* `Simulator`：最薄的一层。补齐 arena_id/robot_id、生成 request_id、POST 出去、解析 JSON。
* `ApiLog`：把每次调用的请求参数和原始响应写成 JSONL，能与模拟器自己的日志逐行对照。
* `RecordedSim`：记录代理。4 个接口原样转发，每次调用顺手交给 ApiLog 落盘。

    from common.sim_client import Simulator

    sim = Simulator(robot_id="<参赛队号>")   # 队号一律运行时传入，代码里不留任何队号
    sim.enter()
    r = sim.measure(300, 400, 1)             # r["measure_result"]: no_signal / near / direction
    if r["measure_result"] == "direction":
        r["svd_deg"]                         # 示向度，含 ±1° 误差
    sim.clear(300, 0, 3)                     # r["clear_result"]: success / no_target_in_range
    sim.exit()

协议上有两点要留意。HTTP 错误由 urllib 抛 HTTPError；业务拒绝不一样，它是 HTTP 200 但
accepted=false，不抛异常，看返回值里的 accepted 判断。网络中断后重试同一动作时，把原
request_id 传进来复用原请求，别让一次断线在模拟器侧变成两次动作。

顶层原先还有个 `sim_api.py` 兼容垫片，已经删了。要这些名字直接从本模块导入：
`from common.sim_client import Simulator`。
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO

__all__ = ["BASE_URL", "API_LOG_NAME", "Simulator", "ApiLog", "RecordedSim",
           "api_brief", "ms_since", "api_log"]

BASE_URL = "http://127.0.0.1:2026"          # 官方默认地址，模拟器只监听本机回环
API_LOG_NAME = "api_calls.jsonl"            # 接口调用日志的文件名，落在 --save-dir 下


class Simulator:
    """模拟器接口的薄封装，4 个方法各自直接返回 JSON 响应，都是 dict"""

    def __init__(self, robot_id: str, base_url: str = BASE_URL,
                 timeout: float = 5.0) -> None:
        """初始化模拟器接口封装

        Args:
            robot_id: 参赛队号。运行时传入，没有缺省值：代码里不写队号，免得上身份信息随
                源码和提交包扩散。官方模式必须与模拟器登录的队号一致
            base_url: 模拟器地址，只监听本机回环
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
            request_id: 断线重试时传入原 request_id 以复用原请求；None 表示按序号新生成一个
            **fields: 该接口的业务字段，直接并进请求体

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
        """进入模拟器 /enter，此后开始占用现实时间预算"""
        return self._post("/enter", request_id)

    def measure(self, x: float, y: float, channel: int,
                request_id: str | None = None) -> dict:
        """在 (x, y) 处测频道 channel 的信号，走 /measure"""
        return self._post("/measure", request_id, position={"x": x, "y": y}, channel=channel)

    def clear(self, x: float, y: float, channel: int,
              request_id: str | None = None) -> dict:
        """在 (x, y) 处对频道 channel 执行清除，走 /clear"""
        return self._post("/clear", request_id, position={"x": x, "y": y}, channel=channel)

    def exit(self, request_id: str | None = None) -> dict:
        """退出模拟器 /exit，计时到此结束"""
        return self._post("/exit", request_id)


def ms_since(t0: float) -> int:
    """自 t0 起的毫秒耗时"""
    return int((time.monotonic() - t0) * 1000)


def api_brief(call: str, resp: dict, elapsed_ms: int) -> str:
    """一次调用的结果摘要，终端上看得懂的单行"""
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
    """把每一次接口调用的请求参数与原始响应写成 JSONL，需要时再回显一行摘要

    模拟器自己也有行为日志，那份不归我们掌握，这里就留一份自己的痕迹：逐条带 request_id，
    可以和模拟器日志逐行对照。每次调用后立即 flush，中途断连或者崩溃，已经发生的调用也不会丢。

    落盘策略，为的是别让一次失败运行把上一次成功的证据链抹掉。本轮第一次要落盘时才决定
    打开方式：

    * 本轮已经成功调用过，正常情形就是 `/enter` 这条且成功：清空重写，一个结果目录对应
      最新一次运行，与直觉一致；
    * 本轮至今全是失败，比如模拟器没启动、地址敲错当场连不上：追加写，并在记录里标
      `"appended": true`。上一次成功运行的完整日志得以保全，本次失败也留了痕。

    官方测试只有 3 次机会。真把上一次成功的记录交给一次敲错地址的运行清空，损失无法挽回。

    路径取 --api-log 指定的文件，没给就是 <save-dir>/api_calls.jsonl。
    """

    def __init__(self, path: Path, echo: Callable[[str], None] | None = None) -> None:
        """初始化接口日志

        Args:
            path: 日志文件路径，JSONL，一行一次调用
            echo: 每条记录的单行摘要回调；None 表示不回显
        """
        self.path = Path(path)
        self._fh: TextIO | None = None
        self._echo = echo
        self._t0 = time.monotonic()
        self.counts: dict[str, int] = defaultdict(int)

    def elapsed(self) -> float:
        """自日志建立起的秒数，走单调时钟，用来记各次调用的相对时刻"""
        return time.monotonic() - self._t0

    def _ensure_open(self, ok: bool, rec: dict[str, Any]) -> TextIO:
        """本轮第一次落盘时决定打开方式；成功过就重写，纯失败则追加，详见类文档"""
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not ok and self.path.exists() and self.path.stat().st_size > 0:
                self._fh = self.path.open("a", encoding="utf-8")
                rec["appended"] = True          # 标记"这一条不与上一次运行同批"
            else:
                self._fh = self.path.open("w", encoding="utf-8")
        return self._fh

    def write(self, rec: dict[str, Any], line: str) -> None:
        """落盘一条调用记录，并把单行摘要交给终端"""
        fh = self._ensure_open(bool(rec.get("ok")), rec)
        self.counts[rec["call"]] += 1
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        if self._echo:
            self._echo(line)

    def report(self) -> None:
        """打印一行统计，便于和模拟器自己的计数交叉核对"""
        if self.counts:
            detail = "、".join(f"{call} {n}" for call, n in sorted(self.counts.items()))
            print(f"接口调用日志：共 {sum(self.counts.values())} 次（{detail}）→ {self.path}")

    def close(self) -> None:
        """关闭日志文件句柄；一次都没落过盘就什么都不做"""
        if self._fh is not None:
            self._fh.close()
            self._fh = None


@contextmanager
def api_log(save_dir: str, raw: str | None, echo: bool) -> Iterator[ApiLog | None]:
    """接口日志的上下文：进入时建文件，退出时关闭

    路径优先用 raw，也就是 --api-log；没指定就用 <save-dir>/api_calls.jsonl；
    显式传空串则不记日志。
    """
    path = raw if raw is not None else str(Path(save_dir) / API_LOG_NAME)
    log = ApiLog(Path(path), echo=print if echo else None) if path else None
    try:
        yield log
    finally:
        if log is not None:                 # 统计由调用方在尾部统一打印
            log.close()


class RecordedSim:
    """Simulator 的记录代理：4 个接口原样转发，每次调用顺手交给 ApiLog 落盘

    request_id 在这一层生成，形如 `<接口>-<局号>-<序号>`。跨局不会重复，也让我们的日志
    能和模拟器行为日志里的同一条请求对上号。
    """

    def __init__(self, sim: Simulator, log: ApiLog, episode: int = 0) -> None:
        """初始化记录代理

        Args:
            sim: 被代理的 Simulator 实例
            log: 调用落盘用的 ApiLog
            episode: 局号，参与生成 request_id，跨局不重复
        """
        self._sim = sim
        self._log = log
        self.episode = episode
        self.seq = 0

    def enter(self) -> dict:
        """记录并转发一次 /enter"""
        return self._call("/enter", {}, lambda rid: self._sim.enter(request_id=rid))

    def measure(self, x: float, y: float, channel: int) -> dict:
        """记录并转发一次 /measure"""
        return self._call("/measure", {"x": x, "y": y, "channel": int(channel)},
                          lambda rid: self._sim.measure(x, y, int(channel), request_id=rid))

    def clear(self, x: float, y: float, channel: int) -> dict:
        """记录并转发一次 /clear"""
        return self._call("/clear", {"x": x, "y": y, "channel": int(channel)},
                          lambda rid: self._sim.clear(x, y, int(channel), request_id=rid))

    def exit(self) -> dict:
        """记录并转发一次 /exit"""
        return self._call("/exit", {}, lambda rid: self._sim.exit(request_id=rid))

    def _call(self, call: str, params: dict[str, Any], send: Callable[[str], dict]) -> dict:
        """执行一次调用并记录：先把请求与时刻落下去，再按成功或异常分别补全结果"""
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
