"""项目内路径的统一定位。

拆分成包以后，"相对于脚本所在目录"的写法会随模块位置漂移（实测：演练场默认去
`cumcm/t3/jammers-py/run.py` 找模拟器而报错）。故这里显式给出两个基准：

* `PACKAGE_ROOT` / `PROJECT_ROOT` —— 由本文件位置反推，与调用方在哪个子模块无关；
* 数据与结果的相对路径则一律以**当前工作目录**为基准（沿用拆分前的行为，
  例如 `--save-dir results/t3` 始终相对 CWD）。
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PROJECT_ROOT", "PACKAGE_ROOT", "JAMMERS_DIR_NAME",
           "default_jammers_dir"]

JAMMERS_DIR_NAME = "jammers-py"      # 本地演练场（复刻模拟器）目录名

PACKAGE_ROOT = Path(__file__).resolve().parent.parent      # .../cumcm
PROJECT_ROOT = PACKAGE_ROOT.parent                         # 仓库根（含 T1.py / T3.py）


def default_jammers_dir() -> Path:
    """演练场目录（缺省 --jammers-dir）：先看仓库根，再看当前工作目录。

    为什么要看 CWD：代码一旦用 `python -m zipapp` 打成单文件，`__file__` 就落在 zip 内部，
    包相对路径会指向 `submit.pyz/jammers-py/...` 这种不存在的路径。此时按"包相对 → 当前
    工作目录"的顺序取第一个真实存在的目录，打包版与源码版行为就一致了。
    """
    for cand in (PROJECT_ROOT / JAMMERS_DIR_NAME, Path.cwd() / JAMMERS_DIR_NAME):
        if (cand / "run.py").exists():
            return cand
    return PROJECT_ROOT / JAMMERS_DIR_NAME        # 都不存在时返回包相对路径，便于报错定位
