"""项目内路径的统一定位：包内的按本文件位置，数据与结果按当前工作目录"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PROJECT_ROOT", "PACKAGE_ROOT", "RESOURCES_DIR_NAME",
           "JAMMERS_REL_DIR", "default_jammers_dir"]

RESOURCES_DIR_NAME = "resources"          # 本机材料目录：赛题与参考文献都在这，不入库
JAMMERS_REL_DIR = "jammers-py"            # 本地演练场相对仓库根，随提交包交付，内容是复刻模拟器

PACKAGE_ROOT = Path(__file__).resolve().parent.parent      # 仓库根，也就是 common/ 的上一级
PROJECT_ROOT = PACKAGE_ROOT                                # 拍平之后包根就是仓库根


def default_jammers_dir() -> Path:
    """演练场目录，也就是 --jammers-dir 的缺省值：先看仓库根，再看当前工作目录

    为什么要看工作目录：代码一旦用 `python -m zipapp` 打成单文件，`__file__` 就落在 zip 内部，
    包相对路径会指向 `submit.pyz/jammers-py/...` 这种不存在的路径。那就按"仓库相对，然后
    当前工作目录"的顺序取第一个真实存在的目录，打包版与源码版的行为就一致了。
    """
    for cand in (PROJECT_ROOT / JAMMERS_REL_DIR, Path.cwd() / JAMMERS_REL_DIR):
        if (cand / "run.py").exists():
            return cand
    return PROJECT_ROOT / JAMMERS_REL_DIR         # 都不在就返回仓库相对路径，便于报错定位
