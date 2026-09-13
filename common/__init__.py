"""common：跨题通用层

本层刻意不放任何"问题 1/2/3"的业务概念，只有可复用的基础设施：

    geometry.py        平面几何：距离、方位角、角度差、圆域裁剪
    routing.py         开路径 TSP 的确定性算子：最近邻、路径长度、2-opt
    sim_client.py      模拟器 HTTP+JSON 接口薄封装，对应题目附件 2 的 4 条指令
    practice_arena.py  本地演练场：拉起或复用 jammers-py 及其控制台 REST
    plotting.py        matplotlib 中文环境、确定性出图、统一配色
    paths.py           项目内路径定位：包根、仓库根、默认演练场目录
    console.py         控制台输出：放宽编码、统一打印

这些实现原先散在各题入口脚本里，dist、bearing、clamp_to_region 之类完全重复，抽出来之后
各题共用同一份。
"""

__all__ = ["geometry", "routing", "sim_client", "practice_arena", "plotting",
           "paths", "console"]
