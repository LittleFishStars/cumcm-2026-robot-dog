"""本地演练场：拉起或复用 jammers-py，并操作它的控制台 REST 接口"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

__all__ = ["PracticeArena", "free_port"]

PROBLEM_NO = 3                      # 默认题目编号，问题三；问题四由调用方传 problem_no=4
START_RETRIES = 6                   # 开局撞上 409 时的重试次数，仅演练场用
START_RETRY_WAIT_S = 1.0            # 每次重试前的等待 / s


def free_port(preferred: int) -> int:
    """取一个可用端口：优先 preferred，被占了就依次往后试 20 个"""
    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:      # 无进程监听
                return port
    raise RuntimeError(f"端口 {preferred}~{preferred + 19} 都被占用")


class PracticeArena:
    """本地演练用的 jammers-py 模拟器：自动拉起进程或复用已在运行的实例，另加控制台 REST"""

    REUSE_PORTS = (8090, 8080)      # 探测已有 jammers-py 的控制台端口，8080 是它的默认端口

    def __init__(self, jammers_dir: Path, robot_id: str, robot_port: int = 2026,
                 console_port: int = 8090, countdown: int = 1,
                 reuse_existing: bool = True, problem_no: int = PROBLEM_NO) -> None:
        """初始化演练场

        Args:
            jammers_dir: jammers-py 所在目录，里头有 run.py
            robot_id: 参赛队号，传给模拟器的 --team
            robot_port: 机器狗接口端口，由模拟器监听
            console_port: 控制台 REST 端口；被占用时会自动往后换
            countdown: 开局倒计时秒数
            reuse_existing: 是否复用已在运行的 jammers-py 实例
            problem_no: 题目编号，3 是全向源，4 是定向 + 全向混合
        """
        self.reuse_existing = reuse_existing
        self.jammers_dir = Path(jammers_dir).resolve()
        self.robot_id = robot_id
        self.robot_port = robot_port
        self.console_port = console_port
        self.countdown = countdown
        self.problem_no = int(problem_no)       # 题目编号：3 = 全向源，4 = 定向+全向混合
        self.robot_url = f"http://127.0.0.1:{robot_port}"
        self.console_url = f"http://127.0.0.1:{console_port}"
        self._proc: subprocess.Popen | None = None

    # ---- 生命周期 ----
    def __enter__(self) -> "PracticeArena":
        """进入演练场：复用已有实例，或者拉起一个新的 jammers-py 等它就绪"""
        existing = self._find_existing() if self.reuse_existing else None
        if existing is not None:
            self.console_url, state = existing
            # 复用已在运行的 jammers-py：机器狗接口以它的实际配置为准
            self.robot_port = int(state.get("config", {}).get("robot_port", self.robot_port))
            self.robot_url = f"http://127.0.0.1:{self.robot_port}"
            print(f"检测到已在运行的 jammers-py，控制台 {self.console_url}，直接复用"
                  f"，要独占实例就加 --no-reuse")
            return self

        run_py = self.jammers_dir / "run.py"
        if not run_py.exists():
            raise FileNotFoundError(f"未找到 jammers-py 模拟器：{run_py}，用 --jammers-dir 换一个目录")
        self.console_port = free_port(self.console_port)
        self.console_url = f"http://127.0.0.1:{self.console_port}"
        self._proc = subprocess.Popen(
            [sys.executable, str(run_py),
             "--robot-port", str(self.robot_port), "--web-port", str(self.console_port),
             "--countdown", str(self.countdown), "--team", self.robot_id],
            cwd=str(self.jammers_dir), stdout=subprocess.DEVNULL)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if self._state(self.console_url) is not None:
                return self
            if self._proc.poll() is not None:
                raise RuntimeError(f"jammers-py 启动失败：机器狗接口 {self.robot_port} 或控制台 "
                                   f"{self.console_port} 端口被占用")
            time.sleep(0.2)
        raise TimeoutError("等待 jammers-py 控制台就绪超时")

    def __exit__(self, *exc: Any) -> bool:
        """退出演练场：收掉自己拉起的那个进程

        Args:
            *exc: with 块内的异常三元组；本演练场不改写异常传播

        Returns:
            bool: 恒为 False，异常照常往外抛
        """
        self.close()
        return False

    def close(self) -> None:
        """停止本实例拉起的 jammers-py；复用的那个实例不动"""
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    # ---- 控制台 REST ----
    def _request(self, path: str, payload: dict | None = None,
                 base: str | None = None, post: bool = False) -> dict:
        """向控制台 REST 发一次请求并返回解析后的 JSON

        Args:
            path: 接口路径，如 "/api/state"
            payload: 请求体；None 表示不带请求体
            base: 基地址；None 表示用本实例的 console_url，探测其他端口时会显式传进来
            post: 是否强制用 POST

        Returns:
            dict: 控制台返回的 JSON
        """
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        # 带请求体就走 POST；post=True 时即使没有请求体也发 POST，如 /api/abort、/api/clear
        req = urllib.request.Request((base or self.console_url) + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if post or payload is not None else "GET")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _state(self, base: str) -> dict | None:
        """读取控制台状态；这个地址不是 jammers-py 或者它没启动就返回 None"""
        try:
            state = self._request("/api/state", base=base)
            return state if "state" in state else None
        except Exception:
            return None

    def _find_existing(self) -> tuple[str, dict] | None:
        """探测是不是已经有 jammers-py 在跑，端口按常驻优先的顺序试"""
        for port in dict.fromkeys((self.console_port, *self.REUSE_PORTS)):
            base = f"http://127.0.0.1:{port}"
            state = self._state(base)
            if state is not None:
                return base, state
        return None

    def _wait_state(self, target: str, timeout: float = 30.0) -> None:
        """轮询控制台，直到模拟器状态等于 target

        Args:
            target: 期望的状态名，如 "window_open"
            timeout: 最长等待时间 / s

        Raises:
            RuntimeError: 演练局提前进入 finished，不可能再到达 target
            TimeoutError: 等超时了还没到 target
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self._request("/api/state").get("state")
            if state == target:
                return
            if state == "finished":
                raise RuntimeError(f"演练局提前结束，本来在等状态 {target}")
            time.sleep(0.2)
        raise TimeoutError(f"等待状态 {target} 超时")

    # ---- 一局演练 ----
    def start_episode(self, seed: int) -> list[dict]:
        """按种子生成固定场景并开一局，等接口开放后返回干扰源真值

        噪声种子被覆写成由 seed 派生的确定值，整局的布局与噪声都完全可复现：同一个 `--seed`
        必然得到同一份结果，新旧策略也能严格对照。盐里带题目编号，免得不同题目、相同 seed
        碰巧共用同一条噪声流。
        """
        scenario = self._request("/api/scenario", {"problem_no": self.problem_no,
                                                   "seed": seed})["scenario"]
        scenario["noise_seed_hex"] = hashlib.blake2b(
            f"t{self.problem_no}-practice-{seed}".encode(), digest_size=8).hexdigest()
        # 连续多局时，上一局的会话可能还没在模拟器侧完全释放，/api/start 会返回 409
        # Conflict。这是演练场的时序问题，不是策略问题，短暂等一会儿再试就行。
        # 只在演练场重试。官方模式保持"发一次就是一次"的语义，免得掩盖真实故障。
        last: Exception | None = None
        for attempt in range(START_RETRIES):
            try:
                self._request("/api/start", {"problem_no": self.problem_no,
                                             "scenario": scenario})
                break
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code != 409 or attempt == START_RETRIES - 1:
                    raise
                time.sleep(START_RETRY_WAIT_S)
        else:                                   # pragma: no cover - 循环必然 break 或 raise
            raise RuntimeError(f"多次重试后仍无法开始演练局：{last}")
        self._wait_state("window_open")
        return scenario["jammers"]

    def finish_episode(self) -> dict:
        """收掉本局并返回模拟器引擎统计：真值清除数、虚拟时刻、检测次数"""
        snap = self._request("/api/state")
        engine = snap.get("engine") or {}
        if snap.get("state") != "finished":
            self._request("/api/abort", post=True)
        self._request("/api/clear", post=True)
        return engine
