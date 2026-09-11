"""sim_api.py —— 模拟器接口的兼容垫片（题目附件 2 的 4 条指令）。

实现在 `cumcm/common/sim_client.py`，这里只是把原来的名字照旧导出，使既有的
`from sim_api import Simulator` 写法继续可用：

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

from __future__ import annotations

from cumcm.common.sim_client import (API_LOG_NAME, BASE_URL, ROBOT_ID, ApiLog,
                                     RecordedSim, Simulator, api_brief, api_log,
                                     ms_since)

__all__ = ["ROBOT_ID", "BASE_URL", "API_LOG_NAME", "Simulator", "ms_since",
           "api_brief", "ApiLog", "api_log", "RecordedSim"]

if __name__ == "__main__":                   # 保留原有自检：连一次模拟器走通进出场
    sim = Simulator()
    print(sim.enter())
    print(sim.exit())
