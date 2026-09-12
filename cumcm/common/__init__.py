"""cumcm.common —— 跨题通用层。

本层刻意不含任何"问题 1/2/3"的业务概念，只提供可复用的基础设施：

    geometry.py        平面几何：距离、方位角、角度差、圆域裁剪
    routing.py         开路径 TSP 的确定性算子：最近邻、路径长度、2-opt
    sim_client.py      模拟器 HTTP+JSON 接口薄封装（题目附件 2 的 4 条指令）
    practice_arena.py  本地演练场：拉起/复用 jammers-py 与其控制台 REST
    plotting.py        matplotlib 中文环境、确定性出图、统一配色
    records.py         结果落盘：CSV/JSON 写出与接口调用日志
    paths.py           项目内路径定位（包根 / 仓库根 / 默认演练场目录）

原先这些实现散落在 T1.py / T3.py / sim_api.py 里各有一份（dist、bearing、
clamp_to_region 等完全重复），抽取后各题共用同一实现。
"""

__all__ = ["geometry", "routing", "sim_client", "practice_arena", "plotting",
           "records", "paths", "console"]
