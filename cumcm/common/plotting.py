"""matplotlib 中文环境与确定性出图。

两个反复踩到的环境坑，集中在这里解决：

1. **字体**：默认字体没有汉字，图上的中文会变成方框。这里给出一份候选字体链（本机装的是
   Noto Sans CJK，Windows 上是微软雅黑/黑体），并统一关掉负号被替换成方框的问题。
2. **配置目录**：matplotlib 的默认配置目录（`~/.config/matplotlib`）在只读 home 或沙箱里
   不可写，于是它退化成 /tmp 下的随机目录并打印警告、每次重新扫字体（很慢）。这里显式
   `setdefault` 到一个固定可写目录，既消除警告又复用字体缓存。

确定性：`save_png` 会去掉 PNG 里的时间戳类元数据，故同一份输入两次出图**逐字节一致** ——
这是"结果可复现"验收的一部分。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

__all__ = ["setup_mpl_env", "font_context", "save_png", "TRAJ_FONTS",
           "C_PATH", "C_DIR", "C_NOSIG", "C_NEAR", "C_TRY", "C_HIT", "C_SRC",
           "C_COVER", "C_FRAME", "C_MEAS", "C_WARN", "C_GRAY"]

# 中文字体候选：优先 Linux 常见 CJK 字体，再退到 Windows 自带，最后 DejaVu 兜底
# （缺字只影响字形，不影响出图与数值）
TRAJ_FONTS = ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
              "Microsoft YaHei", "SimHei", "DejaVu Sans")

# 统一配色：两族策略（t3 / t3ga）与论文出图共用，避免同一含义在不同图里颜色不一致
C_PATH = "#2f6fb5"          # 行驶路径 / 主色
C_DIR = "#1f4e79"           # 有示向度的测量点
C_NOSIG = "#9aa3ad"         # 无信号的测量点
C_NEAR = "#f0a020"          # 近距（可跳过测向）
C_TRY = "#7b3fa0"           # 清除尝试
C_HIT = "#2e9e5b"           # 成功清除落点
C_SRC = "#c0392b"           # 干扰源真值 / 阈值 / 失败
C_COVER = "#6fae6f"         # 覆盖圆
C_FRAME = "#5b6470"         # 圆域边界
C_MEAS = "#c8871b"          # 测向 / 次色
C_WARN = "#c0392b"          # 阈值警告
C_GRAY = "#6b7280"          # 次要文字

_MPL_READY = False


def setup_mpl_env() -> None:
    """把 matplotlib 的配置目录指到固定可写位置（幂等，可反复调用）。"""
    os.environ.setdefault("MPLCONFIGDIR",
                          os.path.join(tempfile.gettempdir(), "cumcm-mplconfig"))


def font_context(size: Optional[float] = None, fonts: Optional[Sequence[str]] = None):
    """返回一个 rc_context，用于统一中文字体、坐标轴负号与（可选）字号。

        with font_context(size=10):
            fig, ax = plt.subplots(); ...
    """
    setup_mpl_env()
    import matplotlib

    rc = {"font.sans-serif": list(fonts or TRAJ_FONTS),
          "font.family": "sans-serif", "axes.unicode_minus": False}
    if size is not None:
        rc["font.size"] = size
    return matplotlib.rc_context(rc)


def save_png(fig: Any, path: Path, dpi: float = 160.0, **kwargs: Any) -> Path:
    """确定性保存 PNG：去掉时间戳类元数据，使同一输入两次出图逐字节一致。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, metadata={"Software": None}, **kwargs)
    return path


def apply_fonts(plt: Any, size: Optional[float] = None,
                extra: Optional[Sequence[str]] = None) -> None:
    """给已导入的 pyplot 设置中文字体（供 T3_figures.py 这类脚本式出图使用）。"""
    setup_mpl_env()
    plt.rcParams["font.sans-serif"] = list(extra or TRAJ_FONTS)
    plt.rcParams["axes.unicode_minus"] = False
    if size is not None:
        plt.rcParams["font.size"] = size
