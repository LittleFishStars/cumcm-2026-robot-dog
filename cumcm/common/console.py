"""控制台与终端输出的小工具（跨题通用）。

原先 T3.py / T3_ga.py / T3_validate.py 里各写了一份"放宽控制台编码"的样板代码，这里合并。
"""

from __future__ import annotations

import sys

__all__ = ["relax_console_encoding"]


def relax_console_encoding() -> None:
    """放宽控制台编码错误处理，避免个别字符（如 √、≤）在 GBK 控制台上中断运行。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:               # 某些流不支持 reconfigure，忽略即可
                pass
