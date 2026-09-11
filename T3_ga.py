#!/usr/bin/env python3
"""T3_ga.py —— 问题 3 的 GA 对照方案（顶层薄入口）。

实现在 `cumcm/t3ga/` 包里；确定性方案见 `cumcm/t3/`（顶层入口 T3.py）。两者共用
`cumcm/common/` 与 `cumcm/t1/`，对照"确定性方法 vs 启发式方法"的指标。

    python T3_ga.py --practice 3                  # 本地演练 3 局
    python T3_ga.py --practice 20 --seed 0        # 20 局
    python T3_ga.py --practice 3 --no-plot        # 不出轨迹图

轨迹图写到 <save-dir>/trajectory/（与 T3.py 的 trajectory/t3/ 分开，避免同名覆盖）。
"""

from __future__ import annotations

import sys

from cumcm.t3ga.cli import main

if __name__ == "__main__":
    sys.exit(main())
