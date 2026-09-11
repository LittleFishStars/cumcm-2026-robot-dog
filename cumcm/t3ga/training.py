"""训练结果落盘：GA 逐代进化记录 + 逐局统计。

*t3_ga_training.json*  —— GA 超参、逐代收敛记录、逐局汇总（含定位误差与清除方式）；
*ga_convergence.csv*   —— 抽稀后的逐代收敛曲线数据（论文收敛图的数据源）；
*episodes.csv*         —— 逐局统计（清除比例、里程、虚拟时间、定位误差）。

文件名的 `t3_` 前缀与确定性方案（cumcm.t3.report 写 `t3_*.json`）区分，两者可共存于
同一个 --save-dir。
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from cumcm.common.geometry import dist
from cumcm.t3ga.localize import _split_runs

from cumcm.t3ga.config import (CONVERGED_FITNESS, GA_LOC, GA_LOC_DOMAIN, GA_RECORD_STRIDE, GA_ROUTE)

# ----------------------------------------------------------------------------
# 训练结果落盘：GA 逐代进化记录 + 逐局统计
# ----------------------------------------------------------------------------
def save_training_results(save_dir: Path, episodes: List[dict], ga_runs: List[dict],
                          meta: dict) -> List[Path]:
    """把 GA 训练结果写盘，返回生成的文件路径。

    - `ga_training.json`：训练配置、逐局统计、每次 GA 调用的摘要与最终解（不含逐代明细）
    - `ga_convergence.csv`：逐代收敛曲线（局号, GA 类型, 对象, 代数, 最优/平均/标准差）
    - `episodes.csv`：逐局战绩（清除数、清除比例、虚拟时间、测向次数）
    文件名固定，重复运行直接覆盖，便于论文与后续绘图脚本稳定引用。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "ga_training.json"
    curve_path = save_dir / "ga_convergence.csv"
    episodes_path = save_dir / "episodes.csv"

    summary = [{k: v for k, v in r.items() if k != "history"} for r in ga_runs]
    json_path.write_text(json.dumps({"meta": meta, "episodes": episodes, "ga_runs": summary},
                                    ensure_ascii=False, indent=2), encoding="utf-8")

    with curve_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["episode", "ga", "label", "seed", "generation",
                         "best", "mean", "std"])
        for r in ga_runs:
            for gen, best, mean, std in r["history"]:
                writer.writerow([r["episode"], r["ga"], r["label"], r["seed"],
                                 gen, f"{best:.6g}", f"{mean:.6g}", f"{std:.6g}"])

    keys = ["episode", "seed", "n_sources", "cleared", "clear_ratio", "virtual_time_s",
            "avg_time_s", "n_measure", "n_clear", "ga_runs", "localize_rms_deg",
            "n_located", "localize_err_mean_m", "localize_err_max_m"]
    with episodes_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(episodes)
    return [json_path, curve_path, episodes_path]


def print_ga_summary(ga_runs: Sequence[dict]) -> None:
    """打印 GA 训练摘要：定位 GA 的收敛代数与残差、路线 GA 的优化幅度。"""
    if not ga_runs:
        print("GA 训练摘要：本次没有产生训练记录")
        return
    loc, route = _split_runs(ga_runs)
    if loc:
        rms = np.array([r["residual_rms_deg"] for r in loc], dtype=float)
        fit = np.array([r["final_fitness"] for r in loc], dtype=float)
        n_conv = sum(1 for r in loc if r["converged"])
        print(f"定位 GA：{len(loc)} 次训练（每代 {GA_LOC.pop} 个体），"
              f"提前收敛 {n_conv}/{len(loc)} 次（判据 适应度<{CONVERGED_FITNESS:.0e}），"
              f"否则跑满 {GA_LOC.gens} 代；最终适应度均值 {fit.mean():.3f}°，"
              f"示向度残差 RMS 均值 {rms.mean():.2f}°")
    for r in route:
        g0, g1 = r["initial_best_g0"], r["final_length_m"]
        gain = (g0 - g1) / g0 * 100 if g0 else 0.0
        print(f"路线 GA：{r['label']}（{r['n_points']} 点）{g0:.0f} m → {g1:.0f} m，"
              f"改进 {gain:.1f}%")


def ga_meta() -> Dict[str, dict]:
    """训练记录里的 GA 超参数快照，便于复现实验。"""
    return {
        "ga_loc": {**asdict(GA_LOC), "domain_m": GA_LOC_DOMAIN,
                   "record_stride": GA_RECORD_STRIDE},
        "ga_route": {**asdict(GA_ROUTE), "record_stride": GA_RECORD_STRIDE},
    }


def mean_rms(ga_runs: Sequence[dict]) -> Optional[float]:
    """本局所有定位 GA 的示向度残差均值（度）。"""
    rms = [r["residual_rms_deg"] for r in _split_runs(ga_runs)[0]]
    return float(np.mean(rms)) if rms else None


def episode_row(episode: int, seed: int, truth: Optional[List[dict]], dog: RobotDog,
                 stats: Dict[str, Any], engine: Optional[dict] = None) -> dict:
    """汇总一局战绩，供 CSV/JSON 落盘。

    演练模式传模拟器真值 `truth` 与引擎统计 `engine`，并逐源算定位误差；官方模式真值
    不可得（接口不返回），相关字段为 None。
    """
    engine = engine or {}
    cleared = int(engine.get("cleared_jammer_count", stats["cleared"]))
    total_time = float(engine.get("virtual_time_s", stats["total_time_s"]))
    errors = [dist((e.x, e.y), (j["position"]["x"], j["position"]["y"]))
              for j in (truth or []) if (e := dog.final_est.get(j["channel"]))]
    return {
        "episode": episode, "seed": seed, "n_sources": len(truth) if truth else None,
        "cleared": cleared, "clear_ratio": cleared / len(truth) if truth else None,
        "virtual_time_s": total_time,
        "avg_time_s": total_time / cleared if cleared else None,
        "n_measure": int(engine.get("measure_accepted_count", stats["n_measure"])),
        "n_clear": stats["n_clear"], "ga_runs": len(dog.ga_runs),
        "localize_rms_deg": mean_rms(dog.ga_runs),
        "n_located": len(errors),
        "localize_err_mean_m": float(np.mean(errors)) if errors else None,
        "localize_err_max_m": float(np.max(errors)) if errors else None,
        "truth": [{"channel": j["channel"], "x": j["position"]["x"], "y": j["position"]["y"],
                   "receive_m": j["receive"]} for j in (truth or [])],
    }
