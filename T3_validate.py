#!/usr/bin/env python3
"""T3_validate.py —— 问题三方案六组独立验证的顶层薄入口（实现在 cumcm/analysis/validate.py）。

    python T3_validate.py                 # 全部六组（默认审计 results/t3_ga/ + 其 holdout/）
    python T3_validate.py --groups A C    # 只跑指定组
"""

from __future__ import annotations

import sys

from cumcm.analysis.validate import main

if __name__ == "__main__":
    sys.exit(main())
