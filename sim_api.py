"""sim_api.py —— 模拟器 HTTP+JSON 接口的薄封装（题目附件 2 的 4 条指令）

    from sim_api import Simulator

    sim = Simulator()                       # 默认队号 202614023005，地址 http://127.0.0.1:2026
    sim.enter()
    r = sim.measure(300, 400, 1)            # r["measure_result"]: no_signal / near / direction
    if r["measure_result"] == "direction":
        r["svd_deg"]                        # 示向度，含 ±1° 误差
    sim.clear(300, 0, 3)                    # r["clear_result"]: success / no_target_in_range
    sim.exit()

4 个方法直接返回模拟器的 JSON 响应（dict），只做三件协议要求的事：补齐 arena_id/robot_id、
自动生成互不重复的 request_id、POST 并解析 JSON。HTTP 错误由 urllib 抛 HTTPError，
业务拒绝（HTTP 200 但 accepted=false）不抛异常，由返回值里的 accepted 判断。

网络中断后重试同一动作时，传入原 request_id 复用原请求，例如：
    sim.measure(300, 400, 1, request_id="measure-7")     # 重试时仍用 "measure-7"
"""

import json
import urllib.request

ROBOT_ID = "202614023005"                   # 参赛队号，必须与模拟器登录的队号一致
BASE_URL = "http://127.0.0.1:2026"          # 官方默认地址，模拟器只监听本机回环


class Simulator:
    def __init__(self, robot_id=ROBOT_ID, base_url=BASE_URL, timeout=5.0):
        self.robot_id = robot_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._seq = 0

    def _post(self, path, request_id=None, **fields):
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

    def enter(self, request_id=None):
        return self._post("/enter", request_id)

    def measure(self, x, y, channel, request_id=None):
        return self._post("/measure", request_id, position={"x": x, "y": y}, channel=channel)

    def clear(self, x, y, channel, request_id=None):
        return self._post("/clear", request_id, position={"x": x, "y": y}, channel=channel)

    def exit(self, request_id=None):
        return self._post("/exit", request_id)


if __name__ == "__main__":
    sim = Simulator()
    print(sim.enter())
    print(sim.exit())
