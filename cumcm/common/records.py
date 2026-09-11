"""结果落盘的小工具：CSV 写出、JSON 写出、目录准备。

原先各脚本里散落着 `json.dumps(..., ensure_ascii=False, indent=2)` 与
`csv.writer(...)` 的手写循环（T3.py、T3_ga.py、T3_figures.py 各一份），这里统一：
**中文一律不转义**（`ensure_ascii=False`），便于直接阅读结果文件。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Union

__all__ = ["ensure_dir", "write_json", "write_rows", "read_json", "read_rows"]

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """确保目录存在并返回 Path（parents=True）。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: PathLike, payload: Any, indent: int = 2) -> Path:
    """写 JSON：中文不转义、缩进便于人读。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
    return p


def read_json(path: PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_rows(path: PathLike, header: Sequence[str], rows: Iterable[Sequence[Any]],
               newline: str = "") -> Path:
    """写 CSV：第一行是表头，其余逐行写出（值由调用方自行格式化）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline=newline, encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for row in rows:
            w.writerow(list(row))
    return p


def read_rows(path: PathLike) -> List[dict]:
    """读 CSV 成 list[dict]（全部字段为字符串，沿用 DictReader 语义）。"""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
