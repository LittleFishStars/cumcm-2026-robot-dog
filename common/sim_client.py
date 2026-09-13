"""模拟器 HTTP+JSON 接口层与调用日志，也就是附件 2 的 4 条指令"""

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
        """初始化模拟器接口封装，队号必须与模拟器登录的那个一致"""
        self.robot_id = robot_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._seq = 0

    def _post(self, path: str, request_id: str | None = None, **fields: Any) -> dict:
        """POST 一个接口并解析返回的 JSON，断线重试时复用原 request_id"""
        self._seq += 1
        payload = {
            "arena_id": "default",
            "robot_id": self.robot_id,
            # request_id 没给就按接口名加序号新生成一个
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
    """把每一次接口调用的请求与原始响应写成 JSONL，需要时再回显一行摘要"""

    def __init__(self, path: Path, echo: Callable[[str], None] | None = None) -> None:
        """初始化接口日志，echo 为 None 时不回显"""
        self.path = Path(path)
        self._fh: TextIO | None = None
        self._echo = echo
        self._t0 = time.monotonic()
        self.counts: dict[str, int] = defaultdict(int)

    def elapsed(self) -> float:
        """自日志建立起的秒数，走单调时钟"""
        return time.monotonic() - self._t0

    def _ensure_open(self, ok: bool, rec: dict[str, Any]) -> TextIO:
        """本轮第一次落盘时决定打开方式：成功过就清空重写，纯失败则追加，详见类文档"""
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 一次敲错地址的运行不该把上一次成功的证据链抹掉，官方测试只有 3 次机会，所以
            # 本轮至今全是失败时改为追加写，并给这条记录标上 appended
            if not ok and self.path.exists() and self.path.stat().st_size > 0:
                self._fh = self.path.open("a", encoding="utf-8")
                rec["appended"] = True          # 标记"这一条不与上一次运行同批"
            else:
                self._fh = self.path.open("w", encoding="utf-8")
        return self._fh

    def write(self, rec: dict[str, Any], line: str) -> None:
        """落盘一条调用记录并把单行摘要交给终端，写完立即 flush"""
        fh = self._ensure_open(bool(rec.get("ok")), rec)
        self.counts[rec["call"]] += 1
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        if self._echo:
            self._echo(line)

    def report(self) -> None:
        """打印一行调用次数统计，便于和模拟器自己的计数交叉核对"""
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
    """接口日志的上下文：进入时建文件，退出时关闭，路径与落盘策略见 `ApiLog`"""
    path = raw if raw is not None else str(Path(save_dir) / API_LOG_NAME)
    log = ApiLog(Path(path), echo=print if echo else None) if path else None
    try:
        yield log
    finally:
        if log is not None:                 # 统计由调用方在尾部统一打印
            log.close()


class RecordedSim:
    """Simulator 的记录代理：4 个接口原样转发，每次调用顺手交给 ApiLog 落盘"""

    def __init__(self, sim: Simulator, log: ApiLog, episode: int = 0) -> None:
        """初始化记录代理，局号参与生成 request_id"""
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
        # request_id 形如 <接口>-<局号>-<序号>，跨局不重复，能和模拟器行为日志里的同一条请求对上
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
