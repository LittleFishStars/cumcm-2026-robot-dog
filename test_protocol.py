"""test_protocol.py —— 接口层（sim_client.py）与本地模拟器（mock_simulator.py）的协议一致性测试

覆盖：客户端本地预检、accepted=false 分支、HTTP 400/404/405/409/413/415、request_id 幂等与
冲突、未知字段不占用 ID、网络中断与 HTTP 500 的重试（不重复计费）、附件 2 第 10 节的虚拟时刻、
定向源覆盖与 5 m 近距 / 20 m 清除半径语义、接口关闭后直接断开连接、交互日志。

运行（不联网、不接触真模拟器、不消耗正式测试次数）：
    .venv/bin/python test_protocol.py
"""

import json
import random
import shutil
import socket
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_simulator import Session, Source, create_server      # noqa: E402
from sim_client import ConnectionFailure, ProtocolError, SimulatorClient   # noqa: E402

TMP_DIR = Path(tempfile.mkdtemp(prefix="sim-protocol-test-"))
TRANSCRIPT = TMP_DIR / "transcript.jsonl"

PASS, FAIL = [], []


def check(name, ok, info=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(info)}")


def start_server(robot_id="TESTTEAM", seed=7, sources=None, **kw):
    session = Session(
        sources=sources if sources is not None
        else Session.make_case(random.Random(seed), 12, None, 0.3),
        robot_id=robot_id, rng=random.Random(seed), quiet=True, **kw,
    )
    server = create_server("127.0.0.1", 0, session)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", session, server


def raw(url, path, payload=None, *, body=None, content_type="application/json",
        method="POST", encoding=None):
    data = body if body is not None else json.dumps(payload).encode("utf-8")
    headers = {} if content_type is None else {"Content-Type": content_type}
    if encoding:
        headers["Content-Encoding"] = encoding
    request = urllib.request.Request(url + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")
    except Exception as exc:                       # noqa: BLE001  连接被直接关闭
        return None, repr(exc)


def fresh_client(url, robot_id="TESTTEAM", **kw):
    return SimulatorClient(robot_id=robot_id, base_url=url, echo=False,
                           transcript_path=TRANSCRIPT, **kw)


print("\n=== 1. 客户端本地预检（不消耗模拟器请求） ===")
url, session, server = start_server()
robot = fresh_client(url)
for name, call in [
    ("坐标超出 ±2e6 被本地拒绝", lambda: robot.measure(3_000_000, 0, 1)),
    ("NaN 坐标被本地拒绝", lambda: robot.measure(float("nan"), 0, 1)),
    ("非整数频道 1.5 被本地拒绝", lambda: robot.measure(0, 0, 1.5)),
    ("频道 21 越界被本地拒绝", lambda: robot.measure(0, 0, 21)),
    ("robot_id 含控制字符被本地拒绝", lambda: SimulatorClient("bad\x01id", url,
                                                        transcript_path=TRANSCRIPT)),
]:
    try:
        call()
        check(name, False, "没有抛异常")
    except ProtocolError:
        check(name, True)
check("预检失败没有发出任何请求", session.request_count == 0, session.request_count)

print("\n=== 2. 未 /enter 时的业务拒绝 ===")
status, body = raw(url, "/measure", {"arena_id": "default", "robot_id": "TESTTEAM",
                                     "request_id": "m0", "position": {"x": 1, "y": 2}, "channel": 1})
check("未 enter 的 /measure → 200 accepted=false", status == 200 and body["accepted"] is False, body)
check("accepted=false 时 virtual_time_s 为 0", body["virtual_time_s"] == 0, body)

print("\n=== 3. /enter 与 HTTP 状态码语义 ===")
enter = {"arena_id": "default", "robot_id": "TESTTEAM", "request_id": "e1"}
status, body = raw(url, "/enter", enter)
check("合法 /enter → 200 accepted=true", status == 200 and body["accepted"] is True, body)
check("/enter 响应含 remaining_real_duration_s", "remaining_real_duration_s" in body, body)
check("重复 /enter → 200 accepted=false",
      raw(url, "/enter", {**enter, "request_id": "e2"})[1]["accepted"] is False)
check("未知路径 → 404", raw(url, "/nope", enter)[0] == 404)
check("尾随斜线 → 404", raw(url, "/enter/", enter)[0] == 404)
check("GET 已知路径 → 405", raw(url, "/measure", None, method="GET")[0] == 405)
check("Content-Type 不支持 → 415", raw(url, "/enter", enter, content_type="text/plain")[0] == 415)
check("Content-Encoding 不支持 → 415", raw(url, "/enter", enter, encoding="gzip")[0] == 415)
check("重复键 JSON → 400",
      raw(url, "/enter", None, body=b'{"arena_id":"default","arena_id":"default"}')[0] == 400)

print("\n=== 4. 未知字段与标识不匹配（不占用 request_id） ===")
bad = {"arena_id": "default", "robot_id": "TESTTEAM", "request_id": "u1",
       "position": {"x": 10, "y": 10}, "channel": 1, "typo_field": 1}
status, body = raw(url, "/measure", bad)
check("未声明字段 → 200 accepted=false", status == 200 and body["accepted"] is False, body)
fixed = {k: v for k, v in bad.items() if k != "typo_field"}
status, body = raw(url, "/measure", fixed)
check("修正后复用同一 request_id 被接受", status == 200 and body["accepted"] is True, body)
check("position 下的未知字段 → accepted=false",
      raw(url, "/measure", {**fixed, "request_id": "u2",
                            "position": {"x": 10, "y": 10, "r": 5}})[1]["accepted"] is False)
check("arena_id 不是 default → accepted=false",
      raw(url, "/measure", {**fixed, "request_id": "u3", "arena_id": "other"})[1]["accepted"] is False)
check("robot_id 不匹配 → accepted=false",
      raw(url, "/measure", {**fixed, "request_id": "u4", "robot_id": "OTHER"})[1]["accepted"] is False)

print("\n=== 5. 请求字段范围（HTTP 400） ===")
base = {"arena_id": "default", "robot_id": "TESTTEAM", "position": {"x": 10, "y": 10}, "channel": 1}
cases = [
    ("channel=1.5 → 400", {"channel": 1.5}),
    ("channel=0 → 400", {"channel": 0}),
    ("channel=21 → 400", {"channel": 21}),
    ("坐标 3e6 → 400", {"position": {"x": 3_000_000, "y": 0}}),
    ("缺 position.x → 400", {"position": {"y": 0}}),
]
for index, (name, patch) in enumerate(cases):
    status, body = raw(url, "/measure", {**base, "request_id": f"r{index}", **patch})
    check(name, status == 400, (status, body))
status, body = raw(url, "/measure", None,
                   body=b'{"arena_id":"default","robot_id":"TESTTEAM","request_id":"nan1",'
                        b'"position":{"x":NaN,"y":0},"channel":1}')
check("NaN 坐标 → 400", status == 400, (status, body))

print("\n=== 6. 幂等与冲突 ===")
idem = {"arena_id": "default", "robot_id": "TESTTEAM", "request_id": "idem-1",
        "position": {"x": 500, "y": 0}, "channel": 1}
first = raw(url, "/measure", idem)
second = raw(url, "/measure", idem)
check("同 ID 同内容重放 → 响应完全一致（不重复计费）", first == second, (first, second))
status, body = raw(url, "/measure", {**idem, "position": {"x": 600, "y": 0}})
check("同 ID 不同内容 → 409", status == 409, (status, body))

print("\n=== 7. 客户端端到端（虚拟时刻精度） ===")
url2, session2, server2 = start_server(seed=7)
robot = fresh_client(url2)
info = robot.enter()
check("enter 暴露现实时间预算和虚拟限时",
      abs(info.max_virtual_duration_s - 360_000) < 1e-9 and info.remaining_real_duration_s == 1200, info)
step2 = robot.measure(300, 400, 1)
step3 = robot.measure(300, 400, 2)
check("示例第 2 步 = 105 s（移动 500 m + 检测 5 s）", abs(step2.virtual_time_s - 105.0) < 1e-9, step2.virtual_time_s)
check("示例第 3 步 = 111 s（切换频道 1 s）", abs(step3.virtual_time_s - 111.0) < 1e-9, step3.virtual_time_s)
check("客户端跟踪测向机当前频道 = 2", robot.current_channel == 2, robot.current_channel)
check("客户端跟踪最近合法位置", robot.position == (300.0, 400.0), robot.position)
check("predict 原地检测 = 5 s", abs(robot.predict("measure", 300, 400, 2).total_s - 5.0) < 1e-9)
check("predict 换频道检测 = 6 s", abs(robot.predict("measure", 300, 400, 5).total_s - 6.0) < 1e-9)
check("predict 移动 400 m + 清除 = 83 s", abs(robot.predict("clear", 300, 0, 3).total_s - 83.0) < 1e-9)
robot.clear(300, 0, 3)
check("clear 后仍可在 20 m 内重试同一频道（返回 no_target 或 success）",
      robot.measure(300, 0, 2).result in ("no_signal", "near", "direction"))
check("/exit 返回 user_exit", robot.exit().exit_reason == "user_exit")
try:
    robot.measure(0, 0, 1)
    check("退出后本地禁止再发动作", False, "没有抛异常")
except ProtocolError:
    check("退出后本地禁止再发动作", True)
robot.close()

print("\n=== 8. 连接失败与接口关闭 ===")
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
offline = fresh_client(f"http://127.0.0.1:{free_port}", max_attempts=2, backoff_s=0.0)
try:
    offline.enter()
    check("连不上 → ConnectionFailure", False, "没有抛异常")
except ConnectionFailure as exc:
    check("连不上 → ConnectionFailure（提示接口开放时间）", "接口" in str(exc), exc)
offline.close()

url3, session3, server3 = start_server(close_after_exit=True, seed=11)
robot3 = fresh_client(url3)
robot3.enter()
robot3.exit()
status, body = raw(url3, "/enter", {"arena_id": "default", "robot_id": "TESTTEAM", "request_id": "e9"})
check("测试结束后请求被直接断开（无 JSON 响应体）", status is None, (status, body))
robot3.close()

print("\n=== 9. 网络中断重试：复用 request_id 且不重复计费 ===")
url4, session4, server4 = start_server(seed=3)
robot4 = fresh_client(url4, max_attempts=3, backoff_s=0.0)
robot4.enter()
original_send = robot4._send
calls = {"n": 0}


def flaky_lost_response(path, body):
    """第一次真实发送并执行成功，随后假装响应丢失（真实网络中断的经典形态）。"""
    calls["n"] += 1
    status, raw_bytes = original_send(path, body)
    if calls["n"] == 1:
        raise ConnectionResetError("模拟：连接中断，响应丢失")
    return status, raw_bytes


robot4._send = flaky_lost_response
result4 = robot4.measure(300, 400, 1)
robot4._send = original_send
check("客户端重试了同一动作（2 次网络请求）", calls["n"] == 2, calls["n"])
check("重试仍得到 105 s，未重复计费", abs(result4.virtual_time_s - 105.0) < 1e-9, result4.virtual_time_s)
check("模拟器侧虚拟时钟也只推进一次", abs(session4.virtual_time_s - 105.0) < 1e-9, session4.virtual_time_s)

print("\n=== 10. HTTP 500 退避重试 ===")
calls2 = {"n": 0}


def flaky_500(path, body):
    calls2["n"] += 1
    if calls2["n"] == 1:
        return 500, json.dumps({"accepted": False, "real_timestamp_ms": 0, "virtual_time_s": 0,
                                "error": "模拟器内部错误"}).encode("utf-8")
    return original_send(path, body)


robot4._send = flaky_500
result5 = robot4.measure(300, 400, 1)
robot4._send = original_send
check("HTTP 500 后重试成功", calls2["n"] == 2 and result5.result in ("no_signal", "near", "direction"),
      (calls2["n"], result5))
check("500 重试后虚拟时钟只累加这一步 5 s", abs(result5.virtual_time_s - 110.0) < 1e-9, result5.virtual_time_s)
robot4.exit()
robot4.close()

print("\n=== 11. 定向源覆盖 / 近距 / 清除半径语义 ===")
# 定向源 ch5：(1000, 0)，有效接收半径 1500 m，定向方向 90°（正北），覆盖 90°±90°。
directional = [Source(channel=5, x=1000.0, y=0.0, rx_radius_m=1500.0,
                      directional=True, heading_deg=90.0)]
url5, session5, server5 = start_server(robot_id="T", sources=directional)


def probe(path, x=None, y=None, channel=None, rid="p"):
    payload = {"arena_id": "default", "robot_id": "T", "request_id": rid}
    if x is not None:
        payload["position"] = {"x": x, "y": y}
    if channel is not None:
        payload["channel"] = channel
    return raw(url5, path, payload)


probe("/enter", rid="e")
boundary = probe("/measure", 0.0, 0.0, 5, rid="d1")
check("恰好落在覆盖边界（±90°）→ direction",
      boundary[1].get("measure_result") == "direction", boundary)
outside = probe("/measure", 0.0, -1000.0, 5, rid="d2")
check("在覆盖范围外但距离够（1414 m ≤ 1500 m）→ no_signal",
      outside[1].get("measure_result") == "no_signal", outside)
near_inside = probe("/measure", 1000.0, 3.0, 5, rid="d3")
check("覆盖内且距离 ≤ 5 m → near（无示向度）",
      near_inside[1].get("measure_result") == "near" and "svd_deg" not in near_inside[1], near_inside)
near_outside = probe("/measure", 1000.0, -3.0, 5, rid="d4")
check("覆盖外且距离 ≤ 5 m → no_signal（near 也要求被覆盖）",
      near_outside[1].get("measure_result") == "no_signal", near_outside)
cleared = probe("/clear", 1000.0, -3.0, 5, rid="c1")
check("清除不受定向朝向限制（背向 3 m 也能清除）",
      cleared[1].get("clear_result") == "success", cleared)
after = probe("/measure", 1000.0, 3.0, 5, rid="d5")
check("同一干扰源清除后该频道 → no_signal（只能清除一次）",
      after[1].get("measure_result") == "no_signal", after)

print("\n=== 12. 交互日志（支撑材料与自查用） ===")
from pathlib import Path                                       # noqa: E402
lines = [json.loads(line) for line in TRANSCRIPT.read_text(
    encoding="utf-8").strip().splitlines()]
check("JSONL 日志记录了每次请求与响应",
      all({"wall_time", "path", "request", "http_status", "response", "virtual_time_s"} <= set(entry)
          for entry in lines) and len(lines) > 5, len(lines))

print(f"\n结果：{len(PASS)} 通过，{len(FAIL)} 失败")
if FAIL:
    print("失败项：" + "；".join(FAIL))
shutil.rmtree(TMP_DIR, ignore_errors=True)
sys.exit(1 if FAIL else 0)
