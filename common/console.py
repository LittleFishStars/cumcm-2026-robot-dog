"""控制台与终端输出的小工具，跨题通用

原先每个入口脚本里都抄了一份"放宽控制台编码"的样板代码，这里合并成一份。顺便把输出详略
统一成三档，各题的入口脚本共用同一套语义。缺省最简洁，要看过程明细就加 `--verbose`：

* `LEVEL_QUIET` 是 0，`--quiet` 只留最终结论与产物路径，给批处理、自动化用；
* `LEVEL_NORMAL` 是 1，缺省，每一步 1~2 行关键结论；
* `LEVEL_VERBOSE` 是 2，`--verbose` 打完整过程，等价于 2026-09-13 精简前的缺省输出。

约定：关键结论走普通 `print`，除非 quiet 否则始终可见；过程明细走本模块的 `detail()`。
成段的过程报告函数，比如 `print_cover_report`，由调用方先问 `is_verbose()` 再调用，
函数内部不用再判断。
"""

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
    """设置输出详略档位，取值 LEVEL_QUIET / LEVEL_NORMAL / LEVEL_VERBOSE

    各入口脚本解析完 `--verbose` / `--quiet` 之后调一次，全局只此一处入口。
    """
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
    """打印一条过程明细，只有 `--verbose` 下看得见，缺省档位直接丢掉

    Args:
        msg: 要打印的一行；不给就是打一个空行
    """
    if is_verbose():
        print(msg)
