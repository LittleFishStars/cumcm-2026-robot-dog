#!/usr/bin/env python3
"""T2.py —— 问题 2：第二个检测点的选择与候选区域（顶层薄入口）。

实现在 `cumcm/t2/` 包里：

    cumcm/t2/config.py    全部参数与物理常量（含"为什么源到 S1 距离 ∈ [5, 1500] m"的依据）
    cumcm/t2/region.py    几何底座：两楔形之交（附录图 2 的定位四边形）直径、可行域透镜、校核
    cumcm/t2/score.py     算法主体：源不确定集 → 可测性保证 → 最坏定位直径 J → 最优区域（弧带）
    cumcm/t2/report.py    CSV 适合度表、JSON 结论、控制台报表
    cumcm/t2/plotting.py  成果图：全域视图 + 放大视图 + 最坏情形几何内嵌图
    cumcm/t2/cli.py       命令行入口

模型一句话：只有一次测向（S1、θ1，误差 ±1°）时，源的可能位置是一个以 S1 为顶点、长
1500 m、张角 2° 的窄楔形；第二个检测点要"一定听得到"就必须落在以楔形 4 个极点为心、半径
1000 m（接收半径下界）的圆盘之交里，再在这个可行域上最小化"最坏情况下的定位区域直径"。

用法：

    python T2.py                                    # 缺省：S1 = 原点、θ1 = 0°
    python T2.py --site 200 -300 --bearing 45       # 换个第一检测点与示向度
    python T2.py --eta 0.05                         # 候选区域收紧到 J ≤ 1.05 J*
    python T2.py --no-plot                          # 只算数与文件
    python -m cumcm.t2.region                       # 几何自检（解析构造 vs shapely 精确值）

产物：`results/t2/` 下的 `t2_suitability.png`（论文可用图）、`t2_suitability.pdf`（矢量版）、
`t2_suitability.csv`（全域逐格适合度表）、`t2_second_site.json`（结论与全部校验数字）。
"""

from cumcm.t2.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
