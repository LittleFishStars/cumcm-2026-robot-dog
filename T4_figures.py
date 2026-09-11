#!/usr/bin/env python3
"""T4_figures.py —— 问题四论文图表生成的顶层薄入口（实现在 cumcm/analysis/figures_t4.py）。

    python T4_figures.py                        # 读 results/t4/，出图到 figures/
    python T4_figures.py --results results/t4 --figdir figures
"""

from __future__ import annotations

import sys

from cumcm.analysis.figures_t4 import main

if __name__ == "__main__":
    sys.exit(main())