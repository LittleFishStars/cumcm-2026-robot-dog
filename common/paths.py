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
    """演练场目录，也是 --jammers-dir 的缺省值：先看仓库根，再看当前工作目录"""
    # 打成单文件 zipapp 之后 __file__ 落在 zip 内部，包相对路径会指向不存在的路径，
    # 所以按"仓库相对，然后当前工作目录"取第一个真实存在的目录
    for cand in (PROJECT_ROOT / JAMMERS_REL_DIR, Path.cwd() / JAMMERS_REL_DIR):
        if (cand / "run.py").exists():
            return cand
    return PROJECT_ROOT / JAMMERS_REL_DIR         # 都不在就返回仓库相对路径，便于报错定位
