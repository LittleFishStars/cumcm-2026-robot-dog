"""mock_simulator.py —— 本地离线「无线电干扰源环境模拟器」（按附件 2 协议实现）

用途：真模拟器需要联网登录、有测试窗口限制、正式测试只有 3 次机会；开发策略时用本文件
在本地随时开一局，反复演练、复现问题、单元测试接口层。它实现了附件 2 的：

* 4 条指令（/enter、/measure、/clear、/exit）与全部请求字段校验；
* HTTP 状态码语义（400 / 404 / 405 / 409 / 413 / 415）与 accepted=false 分支；
* request_id 幂等（同 ID 同内容返回首次响应、同 ID 不同内容返回 409）；
* 虚拟时钟与耗时规则（移动 5 m/s、切换频道 1 s、检测 5 s、清除 3 s / 5 s）；
* 检测结果 no_signal / near / direction（含 ±1° 示向度误差）与清除结果 success /
  no_target_in_range；有效接收半径 1000~1500 m、定向源 ±90° 覆盖、5 m 近距、20 m 清除半径。

它不做的事：不校验登录、不实现 25 分钟窗口与 20 分钟程序运行时间、不生成加密日志
（加密行为日志只能由真模拟器导出）。

运行：
    python mock_simulator.py --port 2026 --robot-id TESTTEAM --seed 42
    # 另开终端
    python sim_client.py --demo --base-url http://127.0.0.1:2026 --robot-id TESTTEAM

常用选项：
    --sources 12          固定干扰源个数（缺省 10~16 随机）
    --channels 1,2,5     固定频道集合（缺省从 1..20 随机不重复）
    --directional-ratio 0    问题 3 场景（全为全向源）；缺省 0.3 对应问题 4
    --seed 42             固定案例，便于复现
    --case-out case.json  把案例真值写出（策略评估用；真模拟器不提供真值）
    --close-after-exit    /exit 后直接断开连接，模拟真模拟器关闭接口的行为
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# 与 sim_client.py 保持一致的协议常量
ARENA_ID = "default"
PATHS = ("/enter", "/measure", "/clear", "/exit")
MAX_COORD = 2_000_000.0
MAX_BODY_BYTES = 65_536
MAX_CHANNEL = 20
MOVE_SPEED_MPS = 5.0
MEASURE_S = 5.0
CHANNEL_SWITCH_S = 1.0
CLEAR_SCAN_S = 3.0
CLEAR_KILL_S = 2.0
NEAR_RADIUS_M = 5.0
CLEAR_RADIUS_M = 20.0
ARENA_RADIUS_M = 1800.0
MIN_RX_RADIUS_M, MAX_RX_RADIUS_M = 1000.0, 1500.0

ALLOWED_FIELDS = {
    "/enter": {"arena_id", "robot_id", "request_id"},
    "/measure": {"arena_id", "robot_id", "request_id", "position", "channel"},
    "/clear": {"arena_id", "robot_id", "request_id", "position", "channel"},
    "/exit": {"arena_id", "robot_id", "request_id"},
}
POSITION_FIELDS = {"x", "y"}


class BadRequest(Exception):
    """请求结构错误 → HTTP 400。"""


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BadRequest(f"JSON 里出现重复键 {key!r}")
        result[key] = value
    return result


def _depth(value: Any, level: int = 1) -> int:
    if isinstance(value, dict):
        return max([level] + [_depth(v, level + 1) for v in value.values()])
    if isinstance(value, list):
        return max([level] + [_depth(v, level + 1) for v in value])
    return level


def _clean_time(seconds: float) -> float | int:
    """按协议：内部按微秒累计，最多保留 6 位小数并删掉无意义的末尾零。"""
    micro = round(seconds * 1e6) / 1e6
    return int(micro) if micro == int(micro) else micro


def _bearing_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """从 (x1, y1) 指向 (x2, y2) 的方位角，单位度，范围 [0, 360)。"""
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360.0


@dataclass
class Source:
    """一个干扰源的真值。"""

    channel: int
    x: float
    y: float
    rx_radius_m: float
    directional: bool = False
    heading_deg: float = 0.0
    cleared: bool = False

    @property
    def kind(self) -> str:
        return "定向" if self.directional else "全向"

    def covers(self, x: float, y: float) -> bool:
        """检测点是否位于本干扰源的有效覆盖角度范围内（全向源恒为真）。"""
        if not self.directional:
            return True
        d = abs((_bearing_deg(self.x, self.y, x, y) - self.heading_deg + 180.0) % 360.0 - 180.0)
        return d <= 90.0 + 1e-9

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(x - self.x, y - self.y)

    def as_truth(self) -> dict:
        return {
            "channel": self.channel, "x": round(self.x, 3), "y": round(self.y, 3),
            "rx_radius_m": round(self.rx_radius_m, 3), "kind": self.kind,
            "heading_deg": round(self.heading_deg, 3) if self.directional else None,
        }


@dataclass
class Session:
    """一局测试的状态机（虚拟时钟、位置、频道、幂等表）。"""

    sources: list[Source]
    robot_id: str | None = None            # None 表示接受任意 robot_id
    rng: random.Random = field(default_factory=lambda: random.Random())
    close_after_exit: bool = False
    quiet: bool = False

    entered: bool = False
    exited: bool = False
    position: tuple[float, float] = (0.0, 0.0)
    channel: int = 1
    virtual_time_s: float = 0.0
    idempotent: dict[str, tuple[str, int, dict]] = field(default_factory=dict)
    request_count: int = 0
    rejected_count: int = 0
    error_count: int = 0
    cleared_channels: list[int] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -------------------------------------------------- 案例

    @staticmethod
    def make_case(rng: random.Random, count: int, channels: list[int] | None,
                  directional_ratio: float) -> list[Source]:
        if channels is None:
            if not 1 <= count <= 20:
                raise SystemExit("干扰源个数必须在 1..20 之间（频道互不相同）")
            channels = sorted(rng.sample(range(1, MAX_CHANNEL + 1), count))
        sources = []
        for channel in channels:
            angle = rng.uniform(0.0, 2 * math.pi)
            radius = ARENA_RADIUS_M * math.sqrt(rng.random())      # 圆域内均匀分布
            directional = rng.random() < directional_ratio
            sources.append(Source(
                channel=channel,
                x=radius * math.cos(angle),
                y=radius * math.sin(angle),
                rx_radius_m=rng.uniform(MIN_RX_RADIUS_M, MAX_RX_RADIUS_M),
                directional=directional,
                heading_deg=rng.uniform(0.0, 360.0) if directional else 0.0,
            ))
        return sources

    # -------------------------------------------------- 动作实现

    def handle(self, path: str, payload: dict) -> tuple[int | None, dict | None]:
        """执行一条已通过结构校验的指令，返回 (HTTP 状态, 响应体)。

        返回 (None, None) 表示接口已关闭：真模拟器在测试结束后会直接断开连接、不给响应体。
        """
        with self._lock:
            if self.close_after_exit and self.exited:
                self._log("[接口关闭] 测试已结束，直接断开连接")
                return None, None
            self.request_count += 1
            fingerprint = json.dumps({k: v for k, v in payload.items() if k != "request_id"},
                                     sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            request_id = payload["request_id"]

            # 1) 未声明字段 / arena_id / robot_id 不匹配：accepted=false，不占用 request_id
            unknown = set(payload) - ALLOWED_FIELDS[path]
            if unknown:
                return 200, self._reject(f"请求包含未声明字段：{sorted(unknown)}")
            position = payload.get("position")
            if isinstance(position, dict):
                unknown_pos = set(position) - POSITION_FIELDS
                if unknown_pos:
                    return 200, self._reject(f"position 包含未声明字段：{sorted(unknown_pos)}")
            if payload.get("arena_id") != ARENA_ID:
                return 200, self._reject(f'arena_id 必须是 "{ARENA_ID}"')
            if self.robot_id is not None and payload.get("robot_id") != self.robot_id:
                return 200, self._reject("robot_id 与当前登录参赛队号不一致")

            # 2) 幂等：同 ID 同内容返回首次响应；同 ID 不同内容 409
            cached = self.idempotent.get(request_id)
            if cached is not None:
                cached_fingerprint, cached_status, cached_body = cached
                if cached_fingerprint != fingerprint:
                    self.error_count += 1
                    return 409, self._base(accepted=False, extra={
                        "error": "同一 request_id 对应了不同动作",
                        "http_note": "HTTP 409",
                    })
                self._log(f"[幂等重放] {path} request_id={request_id}")
                return cached_status, cached_body

            # 3) 业务状态校验：accepted=false，不占用 request_id
            if path == "/enter":
                if self.entered:
                    return 200, self._reject("重复调用 /enter")
            else:
                if not self.entered:
                    return 200, self._reject("尚未成功调用 /enter")
                if self.exited:
                    return 200, self._reject("机器狗已经退出，测试已结束")

            # 4) 执行并推进虚拟时钟
            if path == "/enter":
                self.entered = True
                body = self._base(accepted=True, extra={
                    "max_virtual_duration_s": 360_000,
                    "max_real_duration_s": 1200,
                    "remaining_real_duration_s": 1200,
                })
                self._log(f"[虚拟 {self.virtual_time_s:.3f}s] /enter 开始测试")
            elif path == "/measure":
                body = self._measure(payload)
            elif path == "/clear":
                body = self._clear(payload)
            else:
                self.exited = True
                body = self._base(accepted=True, extra={"exit_reason": "user_exit"})
                self._log(f"[虚拟 {self.virtual_time_s:.3f}s] /exit 测试结束")

            self.idempotent[request_id] = (fingerprint, 200, body)
            return 200, body

    def _measure(self, payload: dict) -> dict:
        x, y = float(payload["position"]["x"]), float(payload["position"]["y"])
        channel = int(payload["channel"])
        move_s = math.hypot(x - self.position[0], y - self.position[1]) / MOVE_SPEED_MPS
        switch_s = CHANNEL_SWITCH_S if channel != self.channel else 0.0
        self.virtual_time_s += move_s + switch_s + MEASURE_S
        self.position = (x, y)
        self.channel = channel

        source = self._active_source(channel)
        extra: dict[str, Any]
        if source is None:
            extra = {"measure_result": "no_signal"}
            note = "no_signal（该频道无未清除干扰源）"
        else:
            distance = source.distance_to(x, y)
            if distance > source.rx_radius_m:
                extra = {"measure_result": "no_signal"}
                note = f"no_signal（距离 {distance:.1f} m > 有效接收半径 {source.rx_radius_m:.1f} m）"
            elif not source.covers(x, y):
                extra = {"measure_result": "no_signal"}
                note = f"no_signal（定向源覆盖 {source.heading_deg:.1f}°±90° 之外）"
            elif distance <= NEAR_RADIUS_M:
                extra = {"measure_result": "near"}
                note = f"near（距离 {distance:.2f} m ≤ 5 m，无示向度）"
            else:
                true_bearing = _bearing_deg(x, y, source.x, source.y)
                measured = (true_bearing + self.rng.uniform(-1.0, 1.0)) % 360.0
                extra = {"measure_result": "direction", "svd_deg": round(measured, 2)}
                note = (f"direction {extra['svd_deg']:.2f}°（真值 {true_bearing:.2f}°，误差 "
                        f"{((extra['svd_deg'] - true_bearing + 180) % 360 - 180):+.2f}°，"
                        f"距离 {distance:.1f} m）")
        self._log(f"[虚拟 {self.virtual_time_s:.3f}s] /measure ({x:.1f},{y:.1f}) ch={channel} "
                  f"移动 {move_s:.3f}s 切换 {switch_s:.0f}s -> {note}")
        return self._base(accepted=True, extra=extra)

    def _clear(self, payload: dict) -> dict:
        x, y = float(payload["position"]["x"]), float(payload["position"]["y"])
        channel = int(payload["channel"])
        move_s = math.hypot(x - self.position[0], y - self.position[1]) / MOVE_SPEED_MPS
        self.position = (x, y)

        source = self._active_source(channel)
        hit = source is not None and source.distance_to(x, y) <= CLEAR_RADIUS_M
        if hit:
            assert source is not None
            source.cleared = True
            self.cleared_channels.append(channel)
            self.virtual_time_s += move_s + CLEAR_SCAN_S + CLEAR_KILL_S
            note = (f"success（ch={channel} 距 {source.distance_to(x, y):.2f} m ≤ 20 m，"
                    f"已清除第 {len(self.cleared_channels)} 个）")
            extra = {"clear_result": "success"}
        else:
            self.virtual_time_s += move_s + CLEAR_SCAN_S
            extra = {"clear_result": "no_target_in_range"}
            if source is None:
                note = f"no_target_in_range（ch={channel} 无未清除干扰源）"
            else:
                note = f"no_target_in_range（最近目标 {source.distance_to(x, y):.1f} m > 20 m）"
        self._log(f"[虚拟 {self.virtual_time_s:.3f}s] /clear ({x:.1f},{y:.1f}) ch={channel} "
                  f"移动 {move_s:.3f}s -> {note}")
        return self._base(accepted=True, extra=extra)

    def _active_source(self, channel: int) -> Source | None:
        for source in self.sources:
            if source.channel == channel:
                return None if source.cleared else source
        return None

    # -------------------------------------------------- 响应构造

    def _base(self, *, accepted: bool, extra: dict | None = None) -> dict:
        body = {
            "accepted": accepted,
            "real_timestamp_ms": int(datetime.now().timestamp() * 1000),
            "virtual_time_s": _clean_time(self.virtual_time_s),
        }
        if not accepted:
            body["virtual_time_s"] = 0          # 协议：accepted=false 时恒为 0
        if extra:
            body.update(extra)
        return body

    def _reject(self, reason: str) -> dict:
        self.rejected_count += 1
        self._log(f"[拒绝] {reason}")
        return self._base(accepted=False, extra={"reject_reason": reason})

    def _log(self, message: str) -> None:
        if not self.quiet:
            print(f"  {message}", flush=True)

    # -------------------------------------------------- 汇总

    def summary(self) -> str:
        cleared = len(self.cleared_channels)
        total = len(self.sources)
        ratio = cleared / total if total else 0.0
        average = self.virtual_time_s / cleared if cleared else 0.0
        lines = [
            "",
            "======== 本地演练汇总（真模拟器的真值不会在正式测试中显示）========",
            f"干扰源总数：{total}（全向 {sum(1 for s in self.sources if not s.directional)}，"
            f"定向 {sum(1 for s in self.sources if s.directional)}）",
            f"已清除：{cleared} 个（{', '.join(f'ch{c}' for c in self.cleared_channels) or '无'}）",
            f"被清除干扰源个数的比例：{ratio:.2%}",
            f"定位清除总时间（虚拟）：{self.virtual_time_s:.3f} s",
            f"平均定位清除时间：{average:.3f} s/个",
            f"请求总数：{self.request_count}（被拒绝 {self.rejected_count}，错误 {self.error_count}）",
            "未清除的干扰源真值：",
        ]
        for source in self.sources:
            mark = "已清除" if source.cleared else "未清除"
            lines.append(f"  ch{source.channel:>2} {source.kind} ({source.x:8.1f}, {source.y:8.1f}) "
                         f"有效接收半径 {source.rx_radius_m:7.1f} m "
                         f"{'朝向 ' + format(source.heading_deg, '6.1f') + '°' if source.directional else ''} [{mark}]")
        lines.append("==============================================================")
        return "\n".join(lines)


class Handler(BaseHTTPRequestHandler):
    server_version = "MockSimulator/1.0"
    protocol_version = "HTTP/1.1"

    session: Session                       # 由 create_server 注入

    def log_message(self, fmt: str, *args: Any) -> None:    # 静音默认访问日志
        pass

    # -------------------------------------------------- 方法分发

    def do_POST(self) -> None:
        path = self.path
        if path not in PATHS:
            self._send_json(404, self.session._base(accepted=False, extra={
                "error": "路径未知或路径不精确（不接受尾随斜线或查询参数）"}))
            return

        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._send_json(400, self.session._base(accepted=False, extra={"error": "缺少 Content-Length"}))
            return
        if int(content_length) > MAX_BODY_BYTES:
            self._send_json(413, self.session._base(accepted=False, extra={
                "error": f"请求体超过 {MAX_BODY_BYTES} 字节"}))
            return

        content_type = (self.headers.get("Content-Type") or "").strip()
        if content_type.lower() not in ("application/json", "application/json; charset=utf-8"):
            self._send_json(415, self.session._base(accepted=False, extra={
                "error": "Content-Type 必须是 application/json，可且仅可带 charset=utf-8"}))
            return
        encoding = (self.headers.get("Content-Encoding") or "identity").strip().lower()
        if encoding not in ("", "identity"):
            self._send_json(415, self.session._base(accepted=False, extra={
                "error": "Content-Encoding 只能是 identity 或省略"}))
            return

        raw = self.rfile.read(int(content_length))
        if raw.startswith(b"\xef\xbb\xbf"):
            self._send_json(400, self.session._base(accepted=False, extra={"error": "请求体不能带 BOM"}))
            return

        try:
            payload = self._parse(raw)
            self._check_common(path, payload)
        except BadRequest as exc:
            self.session.error_count += 1
            self.session._log(f"[HTTP 400] {path} 结构错误：{exc}")
            self._send_json(400, self.session._base(accepted=False, extra={"error": str(exc)}))
            return

        status, body = self.session.handle(path, payload)
        if body is None:                       # 接口已关闭：模拟器直接断开，不返回任何响应
            self.close_connection = True
            return
        self._send_json(status, body)

    def _method_not_allowed(self) -> None:
        path = self.path
        status = 405 if path in PATHS else 404
        self.session.error_count += 1
        note = "该路径只接受 POST" if status == 405 else "路径未知或路径不精确"
        self._send_json(status, self.session._base(accepted=False, extra={"error": note}))

    do_GET = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _method_not_allowed

    # -------------------------------------------------- 校验与解析

    @staticmethod
    def _parse(raw: bytes) -> dict:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BadRequest("请求体不是合法 UTF-8") from exc
        text = text.strip()
        if not text:
            raise BadRequest("请求体为空")
        try:
            payload = json.loads(text, object_pairs_hook=_no_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise BadRequest(f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(payload, dict):
            raise BadRequest("请求体必须是 JSON 对象")
        if _depth(payload) > 16:
            raise BadRequest("JSON 嵌套层数超过 16")
        return payload

    @staticmethod
    def _check_common(path: str, payload: dict) -> None:
        for key in ("arena_id", "robot_id", "request_id"):
            value = payload.get(key)
            if not isinstance(value, str) or value == "":
                raise BadRequest(f"缺少或类型错误的字段 {key}（必须是非空字符串）")
        identifier_limits = {"robot_id": 64, "request_id": 128}
        for key, limit in identifier_limits.items():
            value = payload[key]
            if len(value.encode("utf-8")) > limit:
                raise BadRequest(f"{key} 的 UTF-8 长度不得超过 {limit} 字节")
            for ch in value:
                if ch < " " or ch == "\x7f" or category_of(ch).startswith("C"):
                    raise BadRequest(f"{key} 不能包含控制字符或不可见格式字符")
        if path in ("/measure", "/clear"):
            position = payload.get("position")
            if not isinstance(position, dict):
                raise BadRequest("缺少 position 对象")
            for axis in ("x", "y"):
                if axis not in position:
                    raise BadRequest(f"缺少 position.{axis}")
                value = position[axis]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise BadRequest(f"position.{axis} 必须是数值")
                if not math.isfinite(value) or abs(value) > MAX_COORD:
                    raise BadRequest(f"position.{axis} 必须是有限数值且绝对值不超过 {MAX_COORD:.0f}")
            channel = payload.get("channel")
            if isinstance(channel, bool) or not isinstance(channel, (int, float)):
                raise BadRequest("channel 必须是 1..20 的整数")
            if not float(channel).is_integer() or not 1 <= int(channel) <= MAX_CHANNEL:
                raise BadRequest("channel 必须是 1..20 的整数")

    # -------------------------------------------------- 输出

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def category_of(ch: str) -> str:
    import unicodedata
    return unicodedata.category(ch)


def create_server(host: str, port: int, session: Session) -> ThreadingHTTPServer:
    """把本次会话绑到处理器上并监听端口。"""

    class BoundHandler(Handler):
        pass

    BoundHandler.session = session

    class Server(ThreadingHTTPServer):
        daemon_threads = True

    return Server((host, port), BoundHandler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地离线模拟器（按 2026 CUMCM B 题附件 2 协议实现）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认只监听本机回环")
    parser.add_argument("--port", type=int, default=2026, help="监听端口，默认 2026")
    parser.add_argument("--robot-id", default=None,
                        help="只接受该 robot_id；缺省接受任意 robot_id（便于自检）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，固定案例便于复现")
    parser.add_argument("--sources", type=int, default=None, help="固定干扰源个数（缺省 10~16 随机）")
    parser.add_argument("--channels", default=None, help="固定频道，如 1,3,5,7")
    parser.add_argument("--directional-ratio", type=float, default=0.3,
                        help="定向干扰源比例，0 表示全部全向（问题 3 场景）")
    parser.add_argument("--case-out", default=None, help="把案例真值写到该 JSON 文件")
    parser.add_argument("--close-after-exit", action="store_true",
                        help="/exit 后的请求直接断开连接，模拟真模拟器关闭接口")
    parser.add_argument("--quiet", action="store_true", help="不打印每次交互")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)

    channels = None
    if args.channels:
        channels = [int(part) for part in args.channels.replace(" ", "").split(",") if part]
    count = args.sources if args.sources is not None else (len(channels) if channels else rng.randint(10, 16))
    sources = Session.make_case(rng, count, channels, args.directional_ratio)

    session = Session(sources=sources, robot_id=args.robot_id, rng=rng,
                      close_after_exit=args.close_after_exit, quiet=args.quiet)
    try:
        server = create_server(args.host, args.port, session)
    except OSError as exc:
        print(f"无法监听 {args.host}:{args.port}：{exc}\n"
              f"端口可能被占用（真模拟器默认也用 2026），请换 --port 或关闭占用程序。")
        return 1

    print(f"本地模拟器已启动：http://{args.host}:{args.port}")
    print(f"案例：{len(sources)} 个干扰源（全向 {sum(1 for s in sources if not s.directional)}，"
          f"定向 {sum(1 for s in sources if s.directional)}），seed={args.seed}")
    print(f"robot_id：{args.robot_id or '接受任意值'}")
    print("提示：接口需在 /enter 之后才接受动作；Ctrl-C 结束后打印本局汇总。\n")

    if args.case_out:
        Path(args.case_out).write_text(json.dumps(
            {"seed": args.seed, "sources": [s.as_truth() for s in sources]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"案例真值已写入 {args.case_out}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n收到中断，正在结束本局……")
    finally:
        server.server_close()
        print(session.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
