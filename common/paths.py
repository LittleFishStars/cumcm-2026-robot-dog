"""项目内路径的统一定位。

"相对于脚本所在目录"的写法会随模块位置漂移（实测：按调用方所在目录去找演练场，会落到
`t3/…` 这种并不存在的路径而报错）。故这里显式给出两个基准：

* `PACKAGE_ROOT` / `PROJECT_ROOT` —— 由本文件位置反推，与调用方在哪个子模块无关；
  代码已拍平到仓库根（common/、t1/… 都是根目录下的一级包），故两者指向同一个目录；
* `JAMMERS_REL_DIR` —— 本地演练场相对仓库根的路径（赛题、文献、演练场都在 `resources/` 下）；
* 数据与结果的相对路径则一律以**当前工作目录**为基准（沿用拆分前的行为，
  例如 `--save-dir results/t3` 始终相对 CWD）。
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PROJECT_ROOT", "PACKAGE_ROOT", "RESOURCES_DIR_NAME",
           "JAMMERS_REL_DIR", "default_jammers_dir"]

RESOURCES_DIR_NAME = "resources"          # 外部材料目录：赛题、参考文献与本地演练场都在这里
JAMMERS_REL_DIR = f"{RESOURCES_DIR_NAME}/jammers-py"   # 本地演练场（复刻模拟器）相对仓库根

PACKAGE_ROOT = Path(__file__).resolve().parent.parent      # 仓库根（common/ 的上一级）
PROJECT_ROOT = PACKAGE_ROOT                                # 拍平后包根就是仓库根


def default_jammers_dir() -> Path:
    """演练场目录（缺省 --jammers-dir）：先看仓库根，再看当前工作目录。

    为什么要看 CWD：代码一旦用 `python -m zipapp` 打成单文件，`__file__` 就落在 zip 内部，
    包相对路径会指向 `submit.pyz/resources/jammers-py/...` 这种不存在的路径。此时按
    "仓库相对 → 当前工作目录"的顺序取第一个真实存在的目录，打包版与源码版行为就一致了。
    """
    for cand in (PROJECT_ROOT / JAMMERS_REL_DIR, Path.cwd() / JAMMERS_REL_DIR):
        if (cand / "run.py").exists():
            return cand
    return PROJECT_ROOT / JAMMERS_REL_DIR         # 都不存在时返回仓库相对路径，便于报错定位
