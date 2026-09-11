#!/usr/bin/env python3
"""T3_ga.py —— 问题 3 的 GA 对照方案（顶层薄入口）。

实现在 `cumcm/t3ga/` 包里；确定性方案见 `cumcm/t3/`（顶层入口 T3.py）。两者共用
`cumcm/common/` 与 `cumcm/t1/`，对照"确定性方法 vs 启发式方法"的指标。

    python T3_ga.py --practice 3                  # 本地演练 3 局
    python T3_ga.py --practice 20 --seed 0        # 20 局
    python T3_ga.py --practice 3 --no-plot        # 不出轨迹图

结果写到 <save-dir>（缺省 results/t3_ga/），轨迹图落在其下的 trajectory/。两套方案的
结果树互相独立（T3.py 写 results/t3/），早期共用同一目录且轨迹同名曾互相覆盖。
"""

from __future__ import annotations

import sys

from cumcm.t3ga.cli import main

if __name__ == "__main__":
    sys.exit(main())
