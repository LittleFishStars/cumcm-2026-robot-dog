"""2026 年 CUMCM B 题（机器狗搜索干扰源）的求解包。

目录约定
--------
    common/     跨题通用层：几何、路由、模拟器接口、演练场、绘图、落盘
    t1/         问题 1：交会定位区域
    t2/         问题 2：第二个检测点的选择与候选区域
    t3/         问题 3：确定性策略
    t4/         问题 4：定向 + 全向混合干扰源的确定性策略
    analysis/   分析工具

代码已拍平到仓库根：`common` / `t1` / `t2` / `t3` / `t4` / `analysis` 都是根目录下的一级包，
不再套一层包名，也不再有 T1.py 之类的薄入口。各入口就是各模块自己，例如：

    python -m t1.triangulation                     # 问题 1：4 检测点示例
    python -m t2.cli                               # 问题 2：最优第二检测点 + 出图
    python -m t3.cli --practice 10 --seed 0        # 问题 3：本地演练 10 局
    python -m t4.cli --plan-only                   # 问题 4：只求扫描方案
    python -m analysis.undefined_names             # 静态自检：漏定义/缺失参数

分层原则：**依赖只能向下**。common 不 import 任何题目模块；t1/t2/t3/t4 可 import common，
t2 可 import t1（定位区域的几何底座：同一套楔形交口径，t2 只是把它向量化到成千上万个候选点），
t4 另可 import t1 与 t3.probing（补测选点与 t3 完全同源，直接复用、不复制）。analysis 可 import
全部。t2/t3/t4 是并列叶子。
"""

__all__ = ["common", "t1", "t2", "t3", "t4", "analysis"]
__version__ = "0.3.0"
