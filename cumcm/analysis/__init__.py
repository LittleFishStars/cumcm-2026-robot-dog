"""cumcm.analysis —— 验证脚本与论文出图。

    validate.py   六组独立验证（几何/覆盖/路线最优性/参数敏感性/记录审计/可复现性）
    figures.py    论文用图表（PDF + 背后的数据 CSV）

顶层保留同名薄入口 T3_validate.py / T3_figures.py，命令行用法不变。
"""

__all__ = ["validate", "figures"]
