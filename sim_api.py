"""sim_api.py —— 无线电干扰源环境模拟器 官方 API 客户端（单文件）

本文件把《模拟器通信接口说明及编程指南》（题目附件 2）定义的 HTTP+JSON 协议完整封装成
4 个方法，上层的搜索/清除策略只需：

    from sim_api import SimulatorClient

    robot = SimulatorClient(robot_id="<参赛队号>")   # 官方默认地址 http://127.0.0.1:2026
    info = robot.enter()                            # info.remaining_real_duration_s 本局现实预算
    r = robot.measure(300, 400, 1)                  # r.result: no_signal / near / direction
    if r.result == "direction":
        print(r.svd_deg)                            # 仅 direction 有示向度，含 ±1° 误差
    robot.clear(300, 0, 3)                          # clear_result: success / no_target_in_range
    robot.exit()

官方协议要点
------------------------------------------------------------------
* 4 条指令（全部 POST + JSON）：/enter、/measure、/clear、/exit。模拟器只监听本机回环。
* 每个请求必须带 arena_id="default"、robot_id（当前登录参赛队号）、request_id（幂等键）。
* 移动与切换频道没有独立指令：位置由 /measure、/clear 的 position 体现；切换频道只由
  /measure 的 channel 体现。/clear 的 channel 是"目标干扰源频道"，不切换测向机频道、
  也不产生切换耗时。
* 虚拟耗时：移动 = 直线距离 / 5 (m/s)；切换频道 = 1 s（仅当合法 /measure 的频道与当前不同）；
  检测 = 5 s；清除未发现 = 3 s、成功 = 5 s。虚拟时间不要求现实等待。
* 必须同时检查 HTTP 状态码与响应里的 accepted：只有 HTTP 200 且 accepted=true 动作才生效；
  accepted=false 时 virtual_time_s 恒为 0，不能当作当前虚拟时刻。
* 幂等：同一 request_id + 相同内容 = 同一次动作，重放不重复计费；同 ID 换内容返回 HTTP 409。
  网络超时/连接中断后重试必须原样复用同一份请求。（本客户端自动完成）
* 接口只在测试窗口内开放：倒计时期间、非测试期间、测试结束后会直接断开连接（无 JSON 体）。
* 现实时间预算取自 /enter 的 remaining_real_duration_s（0~1200，不一定是 1200）。
* 测试结束后不要调用 /exit 查询原因（接口已关闭）。

本客户端额外提供
------------------------------------------------------------------
* 本地预检：坐标 ±2e6/NaN、频道必须是 1..20 整数、robot_id/request_id 长度与控制字符，
  在浪费一次模拟器交互之前就报错。
* 自动重试：连接中断、读超时、HTTP 500、HTTP 429 时退避重试，且复用同一 request_id。
* 虚拟时钟记账：只采纳 accepted=true 的 virtual_time_s，并跟踪测向机当前频道与最后合法位置。
* 交互日志：每次请求/响应按 JSON Lines 落盘（题目要求程序自行记录指令序列与响应信息，
  正式测试的支撑材料也需要），默认 logs/robot-<队号>-<时间戳>.jsonl。
* 耗时预估：predict() 按官方公式算出下一步将消耗多少虚拟秒，供策略层比较代价（不发请求）。

运行
------------------------------------------------------------------
    python sim_api.py --robot-id <参赛队号>              # 打印用法
    python sim_api.py --robot-id <参赛队号> --demo       # 在真模拟器上跑一遍最小连通性验证
    python sim_api.py --help

注意：--demo 会向模拟器发送真实动作（/measure、/clear 会推进虚拟时钟）。只在已进入测试窗口、
准备消耗该局时间时使用。
"""

from __future__ import annotations

import argparse
import dataclasses
import http.client
import json
import math
import os
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

# ================================================================ 官方协议常量

ARENA_ID = "default"                       # arena_id 固定为 "default"
DEFAULT_BASE_URL = "http://127.0.0.1:2026"  # 官方默认服务地址
ROBOT_ID_ENV = "CUMCM_ROBOT_ID"            # 参赛队号可用环境变量提供

PATH_ENTER = "/enter"
PATH_MEASURE = "/measure"
PATH_CLEAR = "/clear"
PATH_EXIT = "/exit"
PATHS = (PATH_ENTER, PATH_MEASURE, PATH_CLEAR, PATH_EXIT)

MAX_COORD = 2_000_000.0                    # 坐标分量绝对值上限（米）
CHANNEL_MIN, CHANNEL_MAX = 1, 20           # 频道范围
MAX_BODY_BYTES = 65_536                    # 请求体上限
MAX_ROBOT_ID_BYTES = 64
MAX_REQUEST_ID_BYTES = 128
CONTENT_TYPE = "application/json; charset=utf-8"

# 结果码：判断时用这些字符串，不要依赖模拟器界面的中文说明
NO_SIGNAL = "no_signal"                    # 无信号（无该频道源 / 超有效接收半径 / 定向源未覆盖）
NEAR = "near"                              # 距离 ≤ 5 m 且被覆盖：无示向度，可直接 /clear
DIRECTION = "direction"                    # 返回示向度 svd_deg（含 ±1° 误差）
SUCCESS = "success"                        # /clear 成功清除
NO_TARGET_IN_RANGE = "no_target_in_range"  # /clear 的 20 m 内无该频道未清除干扰源
EXIT_USER = "user_exit"                    # /exit 的退出原因

# 官方时间与几何参数（predict 与策略层复用）
MOVE_SPEED_MPS = 5.0                       # 机器狗移动速度
MEASURE_S = 5.0                            # 一次检测动作耗时
CHANNEL_SWITCH_S = 1.0                     # 任意两频道间切换耗时
CLEAR_SCAN_S = 3.0                         # 光学精确定位耗时（未发现时只有这一段）
CLEAR_KILL_S = 2.0                         # 激光清除耗时（成功时 3 + 2 = 5 s）
NEAR_RADIUS_M = 5.0                        # 近距离阈值
CLEAR_RADIUS_M = 20.0                      # 清除半径
ARENA_RADIUS_M = 1800.0                    # 目标区域半径
RX_RADIUS_MIN_M, RX_RADIUS_MAX_M = 1000.0, 1500.0   # 有效接收半径范围（接口不返回）

# 可重试的传输层异常：连不上、连上后被关闭、读超时
_RETRYABLE = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError)


# ================================================================ 异常

class SimulatorError(Exception):
    """本模块所有异常的基类。"""


class ProtocolError(SimulatorError):
    """本地预检不通过，或服务端返回了不该出现的 HTTP 状态/响应结构。

    这类问题来自代码缺陷或调用方式，重放同样的请求没有意义。
    """

    def __init__(self, message: str, *, status: int | None = None, response: dict | None = None):
        super().__init__(message)
        self.status = status
        self.response = response


class RequestRejected(SimulatorError):
    """HTTP 200 但 accepted=false：动作没有生效。

    常见原因：尚未 /enter、重复 /enter、测试已结束、请求含未声明字段、arena_id 或 robot_id
    不匹配。这类请求不占用 request_id，修正内容后可以复用该 ID。
    """

    def __init__(self, message: str, *, response: dict):
        super().__init__(message)
        self.response = response


class ConnectionFailure(SimulatorError):
    """重试若干次后仍连不上模拟器（接口未开放、测试已结束、网络中断）。"""


# ================================================================ 响应结果

@dataclasses.dataclass(frozen=True)
class EnterResult:
    """POST /enter 的成功响应。"""

    virtual_time_s: float
    real_timestamp_ms: float
    max_virtual_duration_s: float
    max_real_duration_s: float
    remaining_real_duration_s: float


@dataclasses.dataclass(frozen=True)
class MeasureResult:
    """POST /measure 的成功响应。result 取 "no_signal" / "near" / "direction"。"""

    result: str
    svd_deg: float | None                  # 仅 result == "direction" 时非 None
    virtual_time_s: float
    real_timestamp_ms: float


@dataclasses.dataclass(frozen=True)
class ClearResult:
    """POST /clear 的成功响应。result 取 "success" / "no_target_in_range"。"""

    result: str
    virtual_time_s: float
    real_timestamp_ms: float


@dataclasses.dataclass(frozen=True)
class ExitResult:
    """POST /exit 的成功响应。"""

    exit_reason: str
    virtual_time_s: float
    real_timestamp_ms: float


@dataclasses.dataclass(frozen=True)
class CostEstimate:
    """predict() 给出的下一步耗时预估（纯计算，不消耗虚拟时间）。"""

    distance_m: float
    move_s: float
    switch_s: float
    action_s: float

    @property
    def total_s(self) -> float:
        return self.move_s + self.switch_s + self.action_s


# ================================================================ 交互日志

class TranscriptRecorder:
    """把每次请求与响应按 JSON Lines 追加写盘。

    题目要求"机器狗程序应自行记录测试过程中的指令序列、响应信息等内容"，正式测试的支撑
    材料也需要这些记录，所以接口层默认开启。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, event: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TranscriptRecorder":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ================================================================ 本地预检

def _check_identifier(value: str, what: str, max_bytes: int) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{what} 必须是非空字符串，收到 {value!r}")
    if len(value.encode("utf-8")) > max_bytes:
        raise ProtocolError(f"{what} 的 UTF-8 长度不得超过 {max_bytes} 字节：{value!r}")
    for ch in value:
        if unicodedata.category(ch).startswith("C"):     # Cc/Cf/Cs/Co 等控制与不可见格式字符
            raise ProtocolError(f"{what} 不能包含控制字符或不可见格式字符：{value!r}")
    return value


def _check_coord(value: Any, axis: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"坐标 {axis} 必须是数值，收到 {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError(f"坐标 {axis} 必须是有限数值（NaN/无穷大会被 HTTP 400 拒绝）")
    if abs(number) > MAX_COORD:
        raise ProtocolError(f"坐标 {axis}={number} 超出 ±{MAX_COORD:.0f} m 限制")
    return number


def _check_channel(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"channel 必须是整数，收到 {value!r}")
    number = float(value)
    if not number.is_integer():
        raise ProtocolError(f"channel 必须是整数（1.5 会被 HTTP 400 拒绝），收到 {value!r}")
    channel = int(number)
    if not CHANNEL_MIN <= channel <= CHANNEL_MAX:
        raise ProtocolError(f"channel={channel} 超出 {CHANNEL_MIN}..{CHANNEL_MAX}")
    return channel


# ================================================================ 客户端

class SimulatorClient:
    """官方模拟器接口的封装：4 条指令 + 重试/幂等 + 虚拟时钟与现实预算记账。

    官方要求"逐次等待响应，不得并发发送不同动作"，本类不做并发保护，请从单线程串行调用。
    """

    def __init__(
        self,
        robot_id: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 5.0,
        max_attempts: int = 3,
        backoff_s: float = 0.5,
        transcript: TranscriptRecorder | None = None,
        transcript_path: str | Path | None = None,
        echo: bool = True,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.robot_id = _check_identifier(robot_id, "robot_id", MAX_ROBOT_ID_BYTES)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_s = backoff_s
        self.echo = echo
        self._sleep = sleeper

        if transcript is None and transcript_path is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            safe_id = "".join(c if c.isalnum() else "_" for c in self.robot_id)
            transcript_path = Path("logs") / f"robot-{safe_id}-{stamp}.jsonl"
        self.transcript = transcript if transcript is not None else TranscriptRecorder(transcript_path)

        # ---- 会话状态 ----
        self.entered = False
        self.exited = False
        self.virtual_time_s: float | None = None             # 最近一次 accepted=true 的虚拟时刻
        self.max_virtual_duration_s: float | None = None
        self.position: tuple[float, float] | None = None      # 上一次合法动作的位置
        self.current_channel = 1                             # 测向机当前频道（初始 1，仅合法 /measure 改变）
        self.cleared_channels: set[int] = set()              # 本地记录：已确认清除成功的频道

        self._action_seq = 0
        self._fingerprints: dict[str, str] = {}              # request_id -> 内容指纹（防同 ID 换内容）
        self._real_deadline: float | None = None             # monotonic 现实截止

    # ---------------------------------------------------- 会话状态

    @property
    def remaining_real_s(self) -> float | None:
        """本局还剩下多少现实秒（取自 /enter 的预算；未 /enter 时返回 None）。"""
        if self._real_deadline is None:
            return None
        return max(0.0, self._real_deadline - time.monotonic())

    @property
    def remaining_virtual_s(self) -> float | None:
        """虚拟世界还剩多少秒（默认限时 360000 s，一般不是约束）。"""
        if self.max_virtual_duration_s is None or self.virtual_time_s is None:
            return None
        return max(0.0, self.max_virtual_duration_s - self.virtual_time_s)

    @property
    def out_of_time(self) -> bool:
        """现实时间预算是否已用完（用完后模拟器会结束测试并关闭接口）。"""
        remaining = self.remaining_real_s
        return remaining is not None and remaining <= 0.0

    # ---------------------------------------------------- 4 条官方指令

    def enter(self, *, request_id: str | None = None) -> EnterResult:
        """POST /enter：进入目标区域并开始本次运行。不推进虚拟时间，但现实计时从此开始。"""
        if self.entered and not self.exited:
            raise ProtocolError("本局已经调用过 /enter，重复调用会被 accepted=false 拒绝")
        response = self._request(PATH_ENTER, request_id=request_id)
        self.entered = True
        self.exited = False
        self.max_virtual_duration_s = float(response.get("max_virtual_duration_s", 360_000))
        remaining = float(response.get("remaining_real_duration_s", 0))
        self._real_deadline = time.monotonic() + remaining
        return EnterResult(
            virtual_time_s=float(response["virtual_time_s"]),
            real_timestamp_ms=float(response["real_timestamp_ms"]),
            max_virtual_duration_s=self.max_virtual_duration_s,
            max_real_duration_s=float(response.get("max_real_duration_s", 1200)),
            remaining_real_duration_s=remaining,
        )

    def measure(self, x: float, y: float, channel: int, *, request_id: str | None = None) -> MeasureResult:
        """POST /measure：到 (x, y) 对 channel 检测。

        耗时 = 移动 + 可能的 1 s 切换频道 + 5 s 检测。
        """
        response = self._request(
            PATH_MEASURE,
            position=self._position_payload(x, y),
            channel=_check_channel(channel),
            request_id=request_id,
        )
        result = response.get("measure_result")
        if result not in (NO_SIGNAL, NEAR, DIRECTION):
            raise ProtocolError(f"/measure 返回了未知的 measure_result={result!r}", response=response)
        svd = response.get("svd_deg")
        return MeasureResult(
            result=result,
            svd_deg=None if svd is None else float(svd),
            virtual_time_s=float(response["virtual_time_s"]),
            real_timestamp_ms=float(response["real_timestamp_ms"]),
        )

    def clear(self, x: float, y: float, channel: int, *, request_id: str | None = None) -> ClearResult:
        """POST /clear：到 (x, y) 尝试清除 channel 的干扰源（清除半径 20 m，与定向朝向无关）。

        注意 /clear 的 channel 不切换测向机频道，因此不产生切换耗时。
        """
        response = self._request(
            PATH_CLEAR,
            position=self._position_payload(x, y),
            channel=_check_channel(channel),
            request_id=request_id,
        )
        result = response.get("clear_result")
        if result not in (SUCCESS, NO_TARGET_IN_RANGE):
            raise ProtocolError(f"/clear 返回了未知的 clear_result={result!r}", response=response)
        if result == SUCCESS:
            self.cleared_channels.add(int(channel))
        return ClearResult(
            result=result,
            virtual_time_s=float(response["virtual_time_s"]),
            real_timestamp_ms=float(response["real_timestamp_ms"]),
        )

    def exit(self, *, request_id: str | None = None) -> ExitResult:
        """POST /exit：主动结束本次测试。调用后接口关闭，不要再发动作。"""
        response = self._request(PATH_EXIT, request_id=request_id)
        self.exited = True
        return ExitResult(
            exit_reason=str(response.get("exit_reason", "unknown")),
            virtual_time_s=float(response["virtual_time_s"]),
            real_timestamp_ms=float(response["real_timestamp_ms"]),
        )

    def close(self) -> None:
        """关闭本地交互日志文件。"""
        self.transcript.close()

    def __enter__(self) -> "SimulatorClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---------------------------------------------------- 耗时预估（官方公式）

    def predict(
        self,
        action: str,
        x: float | None = None,
        y: float | None = None,
        channel: int | None = None,
        *,
        from_position: tuple[float, float] | None = None,
        hit: bool = False,
    ) -> CostEstimate:
        """按官方公式预测一次动作将消耗的虚拟秒数（供策略层比较代价，不发请求）。

        action 取 "measure" 或 "clear"。位置缺省表示原地不动；起点缺省用上一次合法位置。
        hit 只对 "clear" 有意义：True 表示预计清除成功（动作耗时 5 s），否则按未发现计 3 s。
        """
        if action not in ("measure", "clear"):
            raise ProtocolError(f"action 只能是 'measure' 或 'clear'，收到 {action!r}")

        start = from_position if from_position is not None else self.position
        if start is None:
            start = (0.0, 0.0)                          # /enter 后的初始位置
        target = start if x is None or y is None else (float(x), float(y))
        distance = math.hypot(target[0] - start[0], target[1] - start[1])

        if action == "measure":
            switch_s = CHANNEL_SWITCH_S if channel is not None and _check_channel(channel) != self.current_channel else 0.0
            action_s = MEASURE_S
        else:
            switch_s = 0.0                              # /clear 不切换频道
            action_s = CLEAR_SCAN_S + CLEAR_KILL_S if hit else CLEAR_SCAN_S

        return CostEstimate(
            distance_m=distance,
            move_s=distance / MOVE_SPEED_MPS,
            switch_s=switch_s,
            action_s=action_s,
        )

    # ---------------------------------------------------- 内部实现

    def _position_payload(self, x: float, y: float) -> dict[str, float]:
        return {"x": _check_coord(x, "x"), "y": _check_coord(y, "y")}

    def _resolve_request_id(self, tag: str, request_id: str | None, fingerprint: str) -> str:
        """生成或校验 request_id，并保证同一个 ID 不会对应两种不同内容。"""
        if request_id is None:
            self._action_seq += 1
            request_id = f"{tag}-{self._action_seq}"
        else:
            _check_identifier(request_id, "request_id", MAX_REQUEST_ID_BYTES)
        previous = self._fingerprints.get(request_id)
        if previous is not None and previous != fingerprint:
            raise ProtocolError(
                f"request_id {request_id!r} 已经用于另一份内容；"
                f"重试必须原样复用请求，新的动作必须使用新的 request_id"
            )
        self._fingerprints[request_id] = fingerprint
        return request_id

    def _request(self, path: str, *, request_id: str | None, **fields: Any) -> dict:
        """发送一条指令：处理重试、幂等、HTTP 状态与 accepted，并回写会话状态。"""
        if self.exited:
            raise ProtocolError("本局已经 /exit，官方接口已关闭，不能再发送任何动作")

        payload: dict[str, Any] = {
            "arena_id": ARENA_ID,
            "robot_id": self.robot_id,
            "request_id": "",                            # 占位，稍后填入真实 ID
            **fields,
        }
        tag = path.lstrip("/")
        fingerprint = json.dumps({k: v for k, v in payload.items() if k != "request_id"},
                                 sort_keys=True, ensure_ascii=False)
        payload["request_id"] = self._resolve_request_id(tag, request_id, fingerprint)
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(body) > MAX_BODY_BYTES:
            raise ProtocolError(f"请求体 {len(body)} 字节超过 {MAX_BODY_BYTES} 字节上限")

        response = self._post_with_retry(path, payload, body)
        self._absorb(path, payload, response)
        return response

    def _post_with_retry(self, path: str, payload: dict, body: bytes) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            try:
                status, raw = self._send(path, body)
            except _RETRYABLE as exc:
                # 连接被拒绝/中断/读超时：可能接口未开放，也可能是网络抖动。重试必须复用同一个
                # request_id 与同一份内容，官方模拟器不会因此重复计费。
                last_error = exc
                self._record(path, payload, attempt, None, None, None, repr(exc), time.monotonic() - started)
                if self.echo:
                    print(f"  ! {path} 第 {attempt} 次尝试失败（{exc}），"
                          f"{'重试同一 request_id' if attempt < self.max_attempts else '放弃'}")
                if attempt < self.max_attempts:
                    self._sleep(self.backoff_s * 2 ** (attempt - 1))
                continue

            elapsed = time.monotonic() - started
            try:
                response = json.loads(raw.decode("utf-8")) if raw else None
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._record(path, payload, attempt, status, raw, None, f"响应不是合法 JSON: {exc}", elapsed)
                raise ProtocolError(f"{path} 的响应不是合法 JSON（HTTP {status}）", status=status) from exc

            if not isinstance(response, dict) or "accepted" not in response:
                self._record(path, payload, attempt, status, raw, None, "响应缺少 accepted 字段", elapsed)
                raise ProtocolError(f"{path} 的响应缺少 accepted 字段（HTTP {status}）",
                                    status=status, response=response)

            self._record(path, payload, attempt, status, raw, response, None, elapsed)

            if status == 200:
                if response["accepted"] is True:
                    return response
                reason = self._reject_reason()
                raise RequestRejected(f"{path} 被拒绝（HTTP 200 但 accepted=false）{reason}", response=response)

            if status in (429, 500) and attempt < self.max_attempts:
                # 连接/无效流量保护或模拟器内部错误：幂等键保证重放安全，退避后重试同一请求。
                if self.echo:
                    print(f"  ! {path} 返回 HTTP {status}，退避后重试同一 request_id")
                self._sleep(self.backoff_s * 2 ** (attempt - 1))
                continue

            hint = {
                400: "请求结构错误（字段缺失/类型/范围/重复键），修正后可复用该 request_id",
                404: "路径不正确（必须精确为 /enter、/measure、/clear、/exit，不能带尾随斜线或查询参数）",
                405: "该路径只接受 POST",
                409: "同一 request_id 对应了不同动作，或并发发送了不同动作",
                413: f"请求体超过 {MAX_BODY_BYTES} 字节",
                415: "Content-Type / Content-Encoding 不受支持",
                429: "连接或无效流量保护触发，或本局幂等记录达到上限",
                500: "模拟器内部错误",
            }.get(status, "")
            raise ProtocolError(f"{path} 返回 HTTP {status}。{hint}", status=status, response=response)

        raise ConnectionFailure(
            f"连接 {self.base_url}{path} 失败（尝试 {self.max_attempts} 次）：{last_error!r}。"
            f"官方接口只在测试窗口内开放；倒计时未结束、非测试期间或测试已结束时会直接断开连接。"
        )

    def _send(self, path: str, body: bytes) -> tuple[int, bytes]:
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": CONTENT_TYPE},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:           # 4xx/5xx 同样带 JSON 响应体
            try:
                return exc.code, exc.read()
            finally:
                exc.close()

    def _absorb(self, path: str, payload: dict, response: dict) -> None:
        """把成功响应对应的会话状态写回本地（只有 accepted=true 才会走到这里）。"""
        self.virtual_time_s = float(response["virtual_time_s"])
        if path in (PATH_MEASURE, PATH_CLEAR):
            self.position = (float(payload["position"]["x"]), float(payload["position"]["y"]))
        if path == PATH_MEASURE:
            self.current_channel = int(payload["channel"])

        if self.echo:
            detail = ""
            if path == PATH_MEASURE:
                result = response.get("measure_result")
                detail = {DIRECTION: f"direction {response.get('svd_deg')}°",
                          NEAR: "near（≤5 m，可直接 clear）",
                          NO_SIGNAL: "no_signal"}.get(str(result), str(result))
            elif path == PATH_CLEAR:
                detail = "success" if response.get("clear_result") == SUCCESS else NO_TARGET_IN_RANGE
            elif path == PATH_EXIT:
                detail = str(response.get("exit_reason"))
            elif path == PATH_ENTER:
                detail = f"可用现实时间 {response.get('remaining_real_duration_s')} s"
            where = payload.get("position")
            where = f" ({where['x']:.1f},{where['y']:.1f})" if where else ""
            channel = f" ch={payload['channel']}" if "channel" in payload else ""
            print(f"  [{self.virtual_time_s:>9.3f}s] {path}{where}{channel} -> {detail}")

    @staticmethod
    def _reject_reason() -> str:
        return ("（可能原因：尚未成功 /enter、重复 /enter、测试已结束、请求含未声明字段、"
                "arena_id 或 robot_id 不匹配；此类请求不占用 request_id，修正后可复用该 ID）")

    def _record(self, path: str, payload: dict, attempt: int, status: int | None,
                raw: bytes | None, response: dict | None, error: str | None, elapsed: float) -> None:
        self.transcript.log({
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "path": path,
            "request": payload,
            "attempt": attempt,
            "http_status": status,
            "response": response,
            "raw_response": None if raw is None else raw.decode("utf-8", errors="replace"),
            "error": error,
            "elapsed_ms": round(elapsed * 1000, 3),
            "accepted": None if response is None else response.get("accepted"),
            "virtual_time_s": self.virtual_time_s,
        })


# ================================================================ 命令行

# 附件 2 第 10 节给出的官方计时示例动作序列：(path, x, y, channel, 期望虚拟时刻或 None)
_EXAMPLE_STEPS = [
    (PATH_MEASURE, 300.0, 400.0, 1, 105.0),
    (PATH_MEASURE, 300.0, 400.0, 2, 111.0),
    (PATH_CLEAR, 300.0, 0.0, 3, 194.0),
    (PATH_MEASURE, 300.0, 0.0, 2, 199.0),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="无线电干扰源环境模拟器官方 API 客户端（单文件，仅含通信接口，不含搜索策略）")
    parser.add_argument("--robot-id", default=None,
                        help=f"当前登录的参赛队号，缺省读环境变量 {ROBOT_ID_ENV}")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"模拟器地址，官方默认 {DEFAULT_BASE_URL}")
    parser.add_argument("--timeout", type=float, default=5.0, help="单次 HTTP 超时（秒）")
    parser.add_argument("--demo", action="store_true",
                        help="在真模拟器上发送官方附件 2 第 10 节的示例动作序列（会消耗本局虚拟时间）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    robot_id = args.robot_id or os.environ.get(ROBOT_ID_ENV)

    if not robot_id:
        print(f"需要参赛队号：--robot-id <参赛队号>，或设置环境变量 {ROBOT_ID_ENV}\n")
        print("本文件只提供官方模拟器通信接口，不含搜索策略。用法：")
        print('    from sim_api import SimulatorClient')
        print('    robot = SimulatorClient(robot_id="<参赛队号>")')
        print("    info = robot.enter()          # 取 remaining_real_duration_s 作为现实预算")
        print("    r = robot.measure(x, y, ch)   # r.result: no_signal / near / direction")
        print("    robot.clear(x, y, ch)         # success / no_target_in_range")
        print("    robot.exit()")
        return 2

    if not args.demo:
        print(f"robot_id = {robot_id}，目标地址 = {args.base_url}")
        print("本文件只提供通信接口，不含搜索策略，因此默认不发任何请求。用法见 --help；")
        print("要跑一遍官方附件 2 第 10 节的示例序列，加 --demo（会消耗本局虚拟时间）。")
        return 0

    print(f"连接 {args.base_url}，robot_id={robot_id}")
    with SimulatorClient(robot_id=robot_id, base_url=args.base_url, timeout=args.timeout) as robot:
        info = robot.enter()
        print(f"/enter 成功：可用现实时间 {info.remaining_real_duration_s:.0f} s，"
              f"虚拟限时 {info.max_virtual_duration_s:.0f} s")
        for path, x, y, channel, expected in _EXAMPLE_STEPS:
            if path == PATH_MEASURE:
                result = robot.measure(x, y, channel)
                detail = result.result if result.result != DIRECTION else f"{result.result} {result.svd_deg}°"
            else:
                result = robot.clear(x, y, channel)
                detail = result.result
            note = ""
            if expected is not None:
                note = " 与官方示例一致" if abs(result.virtual_time_s - expected) < 1e-6 else \
                       f" （官方示例为 {expected:g} s：/clear 若命中目标会多耗 2 s）"
            print(f"  -> {detail}{note}")
        robot.exit()
        print(f"已退出；最终虚拟时刻 {robot.virtual_time_s} s，交互日志：{robot.transcript.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
