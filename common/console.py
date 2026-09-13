"""控制台输出：统一编码错误处理与三档详略"""

from __future__ import annotations

import sys

__all__ = ["relax_console_encoding", "LEVEL_QUIET", "LEVEL_NORMAL", "LEVEL_VERBOSE",
           "set_level", "get_level", "is_quiet", "is_verbose", "detail"]

LEVEL_QUIET = 0
LEVEL_NORMAL = 1
LEVEL_VERBOSE = 2

_level = LEVEL_NORMAL


def relax_console_encoding() -> None:
    """放宽控制台编码的错误处理，免得 √、≤ 这类字符在 GBK 控制台上把运行打断"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:               # 某些流不支持 reconfigure，忽略即可
                pass


def set_level(level: int) -> None:
    """设置输出详略档位，取值 LEVEL_QUIET / LEVEL_NORMAL / LEVEL_VERBOSE"""
    global _level
    _level = level


def get_level() -> int:
    """取当前输出档位，缺省是 LEVEL_NORMAL"""
    return _level


def is_quiet() -> bool:
    """当前是否在最简档：只留最终结论与产物路径"""
    return _level <= LEVEL_QUIET


def is_verbose() -> bool:
    """当前是否在详细档：过程明细照打"""
    return _level >= LEVEL_VERBOSE


def detail(msg: str = "") -> None:
    """打印一条过程明细，只有 `--verbose` 下看得见，不给 msg 就是打一个空行"""
    if is_verbose():
        print(msg)
