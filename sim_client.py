"""sim_client.py —— 2026 CUMCM B 题「无线电干扰源环境模拟器」通信接口层

本文件只做一件事：把附件 2 的 HTTP+JSON 协议封装成 4 个可靠的方法，供上层的
搜索/清除策略调用。它本身不含任何搜索策略（那是问题 3、4 的算法层）。

协议要点（摘自附件 2）
------------------------------------------------------------------
* 4 条指令：POST /enter、/measure、/clear、/exit；默认地址 http://127.0.0.1:2026，
  模拟器只监听本机回环。
* 每个请求都带 arena_id="default"、robot_id（当前登录的参赛队号）、request_id（幂等键）。
* 移动与切换频道没有独立指令：位置由 /measure、/clear 的 position 体现；切换频道只由
  /measure 的 channel 体现。/clear 的 channel 是"目标干扰源频道"，既不切换测向机频道，
  也不产生切换耗时。
* 虚拟耗时：移动 = 直线距离 / 5 (m/s)；切换频道 = 1 s（仅当合法 /measure 的频道与当前
  频道不同）；检测 = 5 s；清除未发现 = 3 s、清除成功 = 5 s。虚拟时间只增加，不要求现实等待。
* 幂等：同一个 request_id + 完全相同的内容 = 同一次动作。网络超时/连接中断后重试时必须
  原样重放（同样的 request_id、同样的字段），模拟器不会重复计费。
* 必须同时检查 HTTP 状态码和响应里的 accepted：只有 HTTP 200 且 accepted=true 动作才生效。
  accepted=false 的响应里 virtual_time_s 恒为 0，不能当作当前虚拟时刻。
* 接口在倒计时未结束、测试未开始或测试已结束时可能直接断开连接（没有 JSON 响应体），
  所以"连不上"和"收到 JSON"都要处理。
* 现实时间预算用 /enter 返回的 remaining_real_duration_s（0~1200，不一定是 1200），
  不是固定 20 分钟。
* 测试结束后不要再调用 /exit 去查询原因。

用法
------------------------------------------------------------------
    from sim_client import SimulatorClient

    robot = SimulatorClient(robot_id="<参赛队号>")        # 默认连 127.0.0.1:2026
    info = robot.enter()
    print(info.remaining_real_duration_s)                 # 本局可用现实秒数
    r = robot.measure(300, 400, 1)
    if r.result == "direction":
        print(r.svd_deg)
    robot.clear(300, 0, 3)
    robot.exit()

命令行自检（会真的发动作，只用于连接本地 mock_simulator.py，不要连正式测试）：
    python sim_client.py --demo --base-url http://127.0.0.1:2026 --robot-id TESTTEAM
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
import unicodedata
import urllib.error
import urllib.request
import http.client
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

# ---------------------------------------------------------------- 协议常量

ARENA_ID = "default"
DEFAULT_BASE_URL = "http://127.0.0.1:2026"
DEFAULT_ROBOT_ID_ENV = "CUMCM_ROBOT_ID"

PATH_ENTER = "/enter"
PATH_MEASURE = "/measure"
PATH_CLEAR = "/clear"
PATH_EXIT = "/exit"
PATHS = (PATH_ENTER, PATH_MEASURE, PATH_CLEAR, PATH_EXIT)

MAX_COORD = 2_000_000.0       # 坐标绝对值上限（米）
CHANNEL_MIN, CHANNEL_MAX = 1, 20
MAX_BODY_BYTES = 65_536
MAX_ROBOT_ID_BYTES = 64
MAX_REQUEST_ID_BYTES = 128
CONTENT_TYPE = "application/json; charset=utf-8"

# 检测与清除结果码（判断时用这些字符串，不要依赖界面中文说明）
NO_SIGNAL = "no_signal"        # 无信号：无该频道干扰源 / 超有效接收半径 / 定向源未覆盖此处
NEAR = "near"                  # 距离 ≤ 5 m 且被覆盖：信号过强，无示向度，可直接 /clear
DIRECTION = "direction"        # 正常返回示向度 svd_deg（含 ±1° 误差）
SUCCESS = "success"            # /clear 成功清除
NO_TARGET_IN_RANGE = "no_target_in_range"   # /clear 20 m 内没有该频道的未清除干扰源
EXIT_USER = "user_exit"

# 虚拟世界的时间参数（用于本地预估耗时，见 SimulatorClient.predict）
MOVE_SPEED_MPS = 5.0
MEASURE_S = 5.0
CHANNEL_SWITCH_S = 1.0
CLEAR_SCAN_S = 3.0             # 光学精确定位
CLEAR_KILL_S = 2.0             # 激光清除（成功时与精确定位合计 5 s）
NEAR_RADIUS_M = 5.0
CLEAR_RADIUS_M = 20.0
ARENA_RADIUS_M = 1800.0

# 可重试的传输层异常：连不上、连上后被关闭、读超时
_RETRYABLE = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError)


# ---------------------------------------------------------------- 异常

class SimulatorError(Exception):
    """本模块所有异常的基类。"""


class ProtocolError(SimulatorError):
    """本地预检不通过，或服务端给出了不该出现的 HTTP 状态 / 响应结构。

    这类错误来自代码缺陷或调用方式，重试同样的请求没有意义。
    """

    def __init__(self, message: str, *, status: int | None = None, response: dict | None = None):
        super().__init__(message)
        self.status = status
        self.response = response


class RequestRejected(SimulatorError):
    """HTTP 200 但 accepted=false：动作没有生效（未 /enter、重复 /enter、测试已结束、
    含未声明字段、arena_id 或 robot_id 不匹配等）。

    这种请求不占用 request_id，修正内容后可以复用该 ID。
    """

    def __init__(self, message: str, *, response: dict):
        super().__init__(message)
        self.response = response


class ConnectionFailure(SimulatorError):
    """重试若干次后仍然连不上模拟器（接口未开放、测试已结束或网络中断）。"""


# ---------------------------------------------------------------- 返回结果

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
    svd_deg: float | None
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
    """本地按协议规则预估的一次动作耗时（不消耗虚拟时间，纯计算）。"""

    distance_m: float
    move_s: float
    switch_s: float
    action_s: float

    @property
    def total_s(self) -> float:
        return self.move_s + self.switch_s + self.action_s


# ---------------------------------------------------------------- 交互记录

class TranscriptRecorder:
    """把每次请求与响应按 JSON Lines 追加写盘。

    题目要求"机器狗程序应自行记录测试过程中的指令序列、响应信息等内容"，
    正式测试的支撑材料也需要这些记录，所以接口层默认开启。
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


# ---------------------------------------------------------------- 本地预检

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


# ---------------------------------------------------------------- 客户端

class SimulatorClient:
    """与模拟器通信的接口层：4 条指令 + 重试/幂等 + 虚拟时钟与现实预算记账。

    线程模型：模拟器要求"逐次等待响应，不得并发发送不同动作"，所以本类不做并发保护，
    请从单个线程串行调用。
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

        # 会话状态
        self.entered = False
        self.exited = False
        self.virtual_time_s: float | None = None      # 最近一次 accepted=true 的虚拟时刻
        self.max_virtual_duration_s: float | None = None
        self.position: tuple[float, float] | None = None   # 上一次合法动作的位置
        self.current_channel = 1                      # 测向机当前频道（初始 1，仅合法 /measure 改变）
        self.cleared_channels: set[int] = set()       # 本地推断：已确认清除成功的频道

        self._action_seq = 0
        self._fingerprints: dict[str, str] = {}       # request_id -> 内容指纹（防同 ID 换内容）
        self._real_deadline: float | None = None      # monotonic 现实截止（来自 /enter）
        self._enter_monotonic: float | None = None

    # ---------------------------------------------------- 属性

    @property
    def remaining_real_s(self) -> float | None:
        """本局还剩下多少现实秒（来自 /enter 的预算，未 enter 时返回 None）。"""
        if self._real_deadline is None:
            return None
        return max(0.0, self._real_deadline - time.monotonic())

    @property
    def remaining_virtual_s(self) -> float | None:
        """虚拟世界还剩多少秒（默认限时 360000 s，实际上不构成约束）。"""
        if self.max_virtual_duration_s is None or self.virtual_time_s is None:
            return None
        return max(0.0, self.max_virtual_duration_s - self.virtual_time_s)

    @property
    def out_of_time(self) -> bool:
        """现实时间预算是否已经用完（用完后模拟器会结束测试并关闭接口）。"""
        remaining = self.remaining_real_s
        return remaining is not None and remaining <= 0.0

    # ---------------------------------------------------- 4 条指令

    def enter(self, *, request_id: str | None = None) -> EnterResult:
        """进入目标区域并开始本次运行。虚拟时间不推进，但现实计时从此开始。"""
        if self.entered and not self.exited:
            raise ProtocolError("本局已经调用过 /enter，重复调用会被 accepted=false 拒绝")
        response = self._request(PATH_ENTER, request_id=request_id)
        self.entered = True
        self.exited = False
        self.max_virtual_duration_s = float(response.get("max_virtual_duration_s", 360_000))
        remaining = float(response.get("remaining_real_duration_s", 0))
        self._enter_monotonic = time.monotonic()
        self._real_deadline = self._enter_monotonic + remaining
        return EnterResult(
            virtual_time_s=float(response["virtual_time_s"]),
            real_timestamp_ms=float(response["real_timestamp_ms"]),
            max_virtual_duration_s=self.max_virtual_duration_s,
            max_real_duration_s=float(response.get("max_real_duration_s", 1200)),
            remaining_real_duration_s=remaining,
        )

    def measure(self, x: float, y: float, channel: int, *, request_id: str | None = None) -> MeasureResult:
        """到 (x, y) 对 channel 检测。耗时 = 移动 + 可能的 1 s 切换频道 + 5 s 检测。"""
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
        """到 (x, y) 尝试清除 channel 的干扰源（清除半径 20 m，与定向朝向无关）。

        注意：/clear 的 channel 不切换测向机频道，所以不产生切换耗时。
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
        """主动结束本次测试。调用后接口会关闭，不要再发动作。"""
        response = self._request(PATH_EXIT, request_id=request_id)
        self.exited = True
        return ExitResult(
            exit_reason=str(response.get("exit_reason", "unknown")),
            virtual_time_s=float(response["virtual_time_s"]),
            real_timestamp_ms=float(response["real_timestamp_ms"]),
        )

    def close(self) -> None:
        """关闭本地交互记录文件。"""
        self.transcript.close()

    def __enter__(self) -> "SimulatorClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---------------------------------------------------- 本地耗时预估

    def predict(
        self,
        action: str,
        x: float | None = None,
        y: float | None = None,
        channel: int | None = None,
        *,
        from_position: tuple[float, float] | None = None,
    ) -> CostEstimate:
        """按协议规则预测一次动作将消耗的虚拟秒数（用于策略层的代价比较，不访问网络）。

        action 取 "measure" 或 "clear"。位置缺省时视为原地不动；起点缺省时用上一次合法位置。
        """
        if action not in ("measure", "clear"):
            raise ProtocolError(f"action 只能是 'measure' 或 'clear'，收到 {action!r}")

        start = from_position if from_position is not None else self.position
        if start is None:
            start = (0.0, 0.0)                 # /enter 后的初始位置
        target = start if x is None or y is None else (float(x), float(y))
        distance = math.hypot(target[0] - start[0], target[1] - start[1])

        switch_s = 0.0
        action_s = CLEAR_SCAN_S + CLEAR_KILL_S   # /clear 未发现时只有 3 s，见下
        if action == "measure":
            action_s = MEASURE_S
            if channel is not None and _check_channel(channel) != self.current_channel:
                switch_s = CHANNEL_SWITCH_S
        else:
            action_s = CLEAR_SCAN_S              # 保守估计：按"未发现"计 3 s

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
        """生成或校验 request_id，并保证同一 ID 不会对应两种不同内容。"""
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
        """发送一条指令，处理重试、幂等、HTTP 状态与 accepted，并回写会话状态。"""
        if self.exited:
            raise ProtocolError("本局已经 /exit，模拟器接口已关闭，不能再发送任何动作")

        payload: dict[str, Any] = {
            "arena_id": ARENA_ID,
            "robot_id": self.robot_id,
            "request_id": "",                       # 占位，稍后填入真实 ID
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
                # 连接被拒绝 / 中断 / 超时：接口可能未开放，也可能是网络抖动。
                # 重试必须复用同一 request_id 与同一份内容，模拟器不会重复计费。
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
                reason = self._reject_reason(path, payload)
                raise RequestRejected(f"{path} 被拒绝（HTTP 200 但 accepted=false）{reason}", response=response)

            if status in (429, 500):
                # 连接/流量保护或模拟器内部错误：幂等键保证重放安全，退避后重试同一请求。
                if self.echo:
                    print(f"  ! {path} 返回 HTTP {status}，"
                          f"{'退避后重试同一 request_id' if attempt < self.max_attempts else '重试次数用尽'}")
                if attempt < self.max_attempts:
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
            raise ProtocolError(f"{path} 返回 HTTP {status}。{hint}",
                                status=status, response=response)

        raise ConnectionFailure(
            f"连接 {self.base_url}{path} 失败（尝试 {self.max_attempts} 次）：{last_error!r}。"
            f"接口只在测试窗口内开放；倒计时未结束或测试已结束时会直接断开连接。"
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
        except urllib.error.HTTPError as exc:      # 4xx/5xx 同样带 JSON 体
            try:
                return exc.code, exc.read()
            finally:
                exc.close()

    def _absorb(self, path: str, payload: dict, response: dict) -> None:
        """把成功响应对应的会话状态写回本地（只有 accepted=true 才走到这里）。"""
        self.virtual_time_s = float(response["virtual_time_s"])
        if path in (PATH_MEASURE, PATH_CLEAR):
            self.position = (float(payload["position"]["x"]), float(payload["position"]["y"]))
        if path == PATH_MEASURE:
            self.current_channel = int(payload["channel"])

        if self.echo:
            detail = ""
            if path == PATH_MEASURE:
                result = response.get("measure_result")
                detail = {"direction": f"direction {response.get('svd_deg')}°",
                          "near": "near（≤5 m，可直接 clear）",
                          "no_signal": "no_signal"}.get(str(result), str(result))
            elif path == PATH_CLEAR:
                detail = "success" if response.get("clear_result") == SUCCESS else "no_target_in_range"
            elif path == PATH_EXIT:
                detail = str(response.get("exit_reason"))
            elif path == PATH_ENTER:
                detail = f"可用现实时间 {response.get('remaining_real_duration_s')} s"
            where = payload.get("position")
            where = f" ({where['x']:.1f},{where['y']:.1f})" if where else ""
            channel = f" ch={payload['channel']}" if "channel" in payload else ""
            print(f"  [{self.virtual_time_s:>9.3f}s] {path}{where}{channel} -> {detail}")

    @staticmethod
    def _reject_reason(path: str, payload: dict) -> str:
        if path != PATH_ENTER and payload.get("channel") is None:
            return "（可能原因：尚未成功 /enter、测试已结束、robot_id 与登录队号不一致）"
        return ("（可能原因：请求含未声明字段、重复 /enter、尚未 /enter、测试已结束、"
                "robot_id 与登录队号不一致；此类请求不占用 request_id，修正后可复用）")

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


# ---------------------------------------------------------------- 命令行自检

def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="与无线电干扰源环境模拟器通信的接口层（本文件只含通信，不含策略）")
    parser.add_argument("--robot-id", default=None,
                        help=f"当前登录的参赛队号，缺省读环境变量 {DEFAULT_ROBOT_ID_ENV}")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="模拟器地址，默认 127.0.0.1:2026")
    parser.add_argument("--timeout", type=float, default=5.0, help="单次 HTTP 超时（秒）")
    parser.add_argument("--demo", action="store_true",
                        help="发送附件 2 第 10 节的示例动作序列；只在连本地 mock_simulator.py 时使用")
    return parser


# 附件 2 第 10 节的示例序列：(path, x, y, channel, 期望的虚拟时刻)
_EXAMPLE_STEPS = [
    (PATH_MEASURE, 300.0, 400.0, 1, 105.0),
    (PATH_MEASURE, 300.0, 400.0, 2, 111.0),
    (PATH_CLEAR, 300.0, 0.0, 3, None),
    (PATH_MEASURE, 300.0, 0.0, 2, None),
]


def main(argv: list[str] | None = None) -> int:
    import os

    args = _build_parser().parse_args(argv)
    robot_id = args.robot_id or os.environ.get(DEFAULT_ROBOT_ID_ENV)
    if not robot_id:
        print("需要参赛队号：--robot-id <参赛队号>，或设置环境变量 "
              f"{DEFAULT_ROBOT_ID_ENV}。\n")
        return 2

    if not args.demo:
        print(__doc__.split("用法")[0].strip())
        print(f"\nrobot_id = {robot_id}，目标地址 = {args.base_url}")
        print("本文件只提供通信接口，不含搜索策略。用法：")
        print("    from sim_client import SimulatorClient")
        print("    robot = SimulatorClient(robot_id=...); robot.enter(); robot.measure(...)")
        print("\n要跑一遍协议自检（发动作），加 --demo 并指向本地 mock_simulator.py。")
        return 0

    with SimulatorClient(robot_id=robot_id, base_url=args.base_url, timeout=args.timeout) as robot:
        print(f"连接 {args.base_url}，robot_id={robot_id}")
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
            if expected is not None and abs(result.virtual_time_s - expected) > 1e-6:
                print(f"  !! 虚拟时刻 {result.virtual_time_s} 与附件 2 示例的 {expected} 不一致"
                      f"（本地 mock 按同一规则计时，应当一致）")
            else:
                print(f"  -> {detail}")
        print("说明：/clear 命中目标时会多耗 2 s，后续时刻随之为 196/199；未命中时与示例一致。")
        robot.exit()
        print(f"已退出；本地记录的虚拟时刻 = {robot.virtual_time_s} s，"
              f"交互日志：{robot.transcript.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
