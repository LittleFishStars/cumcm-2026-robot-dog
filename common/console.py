"""控制台与终端输出的小工具（跨题通用）。

原先各入口脚本里各写了一份"放宽控制台编码"的样板代码，这里合并；另外统一输出的三档详略，
各题的入口脚本共用同一套语义（缺省最简洁，过程明细要看时加 `--verbose`）：

* `LEVEL_QUIET`（0，`--quiet`）—— 只留最终结论与产物路径，给批处理/自动化用；
* `LEVEL_NORMAL`（1，缺省）—— 每一步 1~2 行关键结论；
* `LEVEL_VERBOSE`（2，`--verbose`）—— 完整过程，等价于 2026-09-13 精简前的缺省输出。

约定：**关键结论**用普通 `print`（始终可见，除非 quiet），**过程明细**用本模块的 `detail()`；
成段的过程报告函数（如 `print_cover_report`）由调用方先问 `is_verbose()` 再调用，函数内部不改。
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
    """放宽控制台编码错误处理，避免个别字符（如 √、≤）在 GBK 控制台上中断运行。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:               # 某些流不支持 reconfigure，忽略即可
                pass


def set_level(level: int) -> None:
    """设置输出详略档位（各入口脚本解析完 `--verbose` / `--quiet` 后调用一次）

    Args:
        level: LEVEL_QUIET / LEVEL_NORMAL / LEVEL_VERBOSE
    """
    global _level
    _level = level


def get_level() -> int:
    """取当前输出档位

    Returns:
        int: 当前档位（缺省 LEVEL_NORMAL）
    """
    return _level


def is_quiet() -> bool:
    """当前是否为最简档（只留最终结论与产物路径）"""
    return _level <= LEVEL_QUIET


def is_verbose() -> bool:
    """当前是否为详细档（打印过程明细）"""
    return _level >= LEVEL_VERBOSE


def detail(msg: str = "") -> None:
    """打印过程明细：仅 `--verbose` 时可见，缺省档位下静默丢弃

    Args:
        msg: 要打印的一行（缺省空串即空行）
    """
    if is_verbose():
        print(msg)
