"""问题一的命令行入口：4 个检测点的交会定位区域示例"""

from __future__ import annotations

import argparse
from typing import Sequence

from common.console import relax_console_encoding
from t1.triangulation import demo


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器，问题一是几何示例，没有可调参数"""
    return argparse.ArgumentParser(
        prog="python -m t1",
        description="CUMCM 2026 B 问题一：交会定位法的定位区域（4 个检测点示例）")


def main(argv: Sequence[str] | None = None) -> int:
    """命令行主流程：解析参数并跑示例，返回进程退出码"""
    relax_console_encoding()
    build_parser().parse_args(argv)
    demo()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
