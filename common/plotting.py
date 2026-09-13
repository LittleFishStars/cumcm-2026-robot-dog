"""matplotlib 中文环境与确定性出图：配好字体链，PNG 去掉时间戳一类元数据"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

__all__ = ["setup_mpl_env", "font_context", "save_png", "slug", "hint_plot_once",
           "TRAJ_FONTS",
           "C_PATH", "C_DIR", "C_NOSIG", "C_NEAR", "C_TRY", "C_HIT", "C_SRC",
           "C_COVER", "C_FRAME", "C_MEAS", "C_WARN", "C_GRAY"]

# 中文字体候选：先挑 Linux 上常见的 CJK 字体，再退到 Windows 自带的，最后 DejaVu 兜底。
# 缺字只影响字形，不影响出图与数值
TRAJ_FONTS = ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
              "Microsoft YaHei", "SimHei", "DejaVu Sans")

# 统一配色：各题策略与论文出图共用，免得上同一个含义在不同图里颜色不一致
C_PATH = "#2f6fb5"          # 行驶路径 / 主色
C_DIR = "#1f4e79"           # 有示向度的测量点
C_NOSIG = "#9aa3ad"         # 无信号的测量点
C_NEAR = "#f0a020"          # 近距，可跳过测向
C_TRY = "#7b3fa0"           # 清除尝试
C_HIT = "#2e9e5b"           # 成功清除落点
C_SRC = "#c0392b"           # 干扰源真值 / 阈值 / 失败
C_COVER = "#6fae6f"         # 覆盖圆
C_FRAME = "#5b6470"         # 圆域边界
C_MEAS = "#c8871b"          # 测向 / 次色
C_WARN = "#c0392b"          # 阈值警告
C_GRAY = "#6b7280"          # 次要文字


def setup_mpl_env() -> None:
    """把 matplotlib 的配置目录指到固定可写位置，幂等，可以反复调用"""
    os.environ.setdefault("MPLCONFIGDIR",
                          os.path.join(tempfile.gettempdir(), "sm-mplconfig"))


def font_context(size: float | None = None, fonts: Sequence[str] | None = None) -> Any:
    """返回一个 rc_context，统一中文字体与坐标轴负号，给了 size 就一并设字号"""
    setup_mpl_env()
    import matplotlib

    rc = {"font.sans-serif": list(fonts or TRAJ_FONTS),
          "font.family": "sans-serif", "axes.unicode_minus": False}
    if size is not None:
        rc["font.size"] = size
    return matplotlib.rc_context(rc)


def save_png(fig: Any, path: Path, dpi: float = 160.0, **kwargs: Any) -> Path:
    """确定性保存 PNG：去掉时间戳一类元数据，同一输入两次出图逐字节一致"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, metadata={"Software": None}, **kwargs)
    return path


def slug(text: str) -> str:
    """把标签压成安全的文件名片段：中文与字母数字留着，其余换成下划线，压完为空就退回 step"""
    keep = [c if (c.isalnum() or c in "._-") else "_" for c in text]
    return re.sub(r"_+", "_", "".join(keep)).strip("_") or "step"


# "出图失败或被跳过的提示"只打印一次。逐局、逐步骤出图时同一个原因会重复几十次，刷屏反而
# 把真正重要的战报埋掉，一次说清并给出处置办法就够了。
_PLOT_HINTED = False


def hint_plot_once(reason: str) -> None:
    """报告一次"图没能生成"，首次调用时打印，其后静默"""
    global _PLOT_HINTED
    if not _PLOT_HINTED:
        _PLOT_HINTED = True
        print(reason)


# "这一局的图没能生成"的唯一文案。原先在两族绘图里各抄了一遍，问题四还多留了一份死代码；
# 集中到这里之后，改处置办法只需改一处，也不会出现两族口径不一致。
NO_PLOT_HINT = ("提示：未安装 matplotlib，已跳过出图。装上即可自动生成：\n"
                "      .venv/bin/pip install matplotlib")


def no_plot_hint() -> None:
    """说明"图没能生成"以及怎么才能生成，只打印一次，免得逐局、逐步骤刷屏"""
    hint_plot_once(NO_PLOT_HINT)

