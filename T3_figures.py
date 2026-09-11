#!/usr/bin/env python3
"""T3_figures.py —— 问题三论文图表生成的顶层薄入口（实现在 cumcm/analysis/figures.py）。

    python T3_figures.py                              # 读 results/t3_ga/，出图到 figures/
    python T3_figures.py --results results/t3_ga       # 指定结果目录
"""

from __future__ import annotations

import sys

from cumcm.analysis.figures import main

if __name__ == "__main__":
    sys.exit(main())
