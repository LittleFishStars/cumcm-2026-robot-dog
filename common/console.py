"""控制台输出：统一编码错误处理、三档详略与对齐表格"""

from __future__ import annotations

import sys
import unicodedata
from typing import Any, Sequence

__all__ = ["relax_console_encoding", "LEVEL_QUIET", "LEVEL_NORMAL", "LEVEL_VERBOSE",
           "set_level", "get_level", "is_quiet", "is_verbose", "detail",
           "display_width", "print_table", "print_paths", "EMPTY_CELL"]

LEVEL_QUIET = 0
LEVEL_NORMAL = 1
LEVEL_VERBOSE = 2

# 表格里的空值统一写这个，留白看不出是"本来没有"还是"忘了填"
EMPTY_CELL = "—"

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


def display_width(text: str) -> int:
    """一段文字在终端里占几列：全角字符算两列，其余算一列"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, right: bool) -> str:
    """把单元格补到指定列宽，right 为真时右对齐"""
    space = " " * max(0, width - display_width(text))
    return space + text if right else text + space


def print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
                align: str = "", indent: int = 0) -> None:
    """打印一张对齐表格，列宽按显示宽度算，表头下面画一条虚线

    align 是每列的对齐方式，`l` 或 `r`，缺的按左对齐；数值列给 `r` 更整齐。
    """
    cols = len(headers)
    cells = [[("" if c is None else str(c)) for c in row] for row in rows]
    widths = [display_width(h) for h in headers]
    for row in cells:
        for i in range(min(cols, len(row))):
            widths[i] = max(widths[i], display_width(row[i]))
    right = [(i < len(align) and align[i] == "r") for i in range(cols)]
    pad = " " * indent
    head = "  ".join(_pad(h, widths[i], right[i]) for i, h in enumerate(headers))
    print(pad + head.rstrip())
    print(pad + "  ".join("-" * w for w in widths))
    for row in cells:
        line = "  ".join(_pad(row[i] if i < len(row) else "", widths[i], right[i])
                         for i in range(cols))
        print(pad + line.rstrip())


def print_paths(paths: Sequence[Any]) -> None:
    """产物路径一行一个，四五个路径挤成一行会顶到 130 列开外"""
    for p in paths:
        print(str(p))
