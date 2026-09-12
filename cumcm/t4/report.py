"""结果核对与落盘（问题四）：真值核对、整批汇总、拖网方案/逐局统计/观测明细的写出。

演练模式下能拿到真值，于是逐源核对两件本该保证的事：
* **检测保证**：每个源（含定向源）的频道在拖网结束前确实被听到过（拖网命中该源的半圆盘），
  统计首次听到发生在拖网的第几步；
* **清除结果**：清除点与真值的距离（定位误差）是否落在 20 m 清除半径内。

正式模式下真值不可见，这些字段自动省略，但落盘格式与逐局统计保持不变。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from cumcm.common.geometry import dist
from cumcm.t4.config import (CLEAR_RADIUS, RECEIVE_MAX, OBS_CSV, PLAN_CSV, PLAN_JSON,
                             RESULTS_DIR, SURVEY_JSON)
from cumcm.t4.regions import Meas, Obs
from cumcm.t4.sweep import SweepPlan


def truth_check(truth: Optional[Sequence[dict]], plan: SweepPlan,
                obs: Dict[int, List[Obs]], cleared: set, tracks: dict,
                first_heard: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
    """逐源核对（仅演练模式有真值）：

    * 检测：该源频道是否被听到（拖网保证每个源 ≥ 1 次示向度），记录首次听到的拖网步骤；
      对定向源额外记录其定向方向，用于说明"方向不对"的场景也被拖网兜住（听到即证明命中）；
    * 清除：清除点与真值的距离（定位误差）、判定是否在 20 m 清除半径内。
    """
    tracks = tracks or {}
    first_heard = first_heard or {}
    rows: List[Dict[str, Any]] = []
    for j in truth or []:
        p = (float(j["position"]["x"]), float(j["position"]["y"]))
        ch = int(j["channel"])
        heard = len(obs.get(ch, ())) > 0 or ch in cleared
        dir_deg = j.get("direction_deg")
        rec = tracks.get(ch, {})
        cp = rec.get("clear_point")
        err = dist(cp, p) if cp else None
        rows.append({
            "channel": ch, "x": p[0], "y": p[1],
            "kind": "directional" if dir_deg is not None else "omni",
            "direction_deg": round(float(dir_deg), 2) if dir_deg is not None else None,
            "receive_m": round(float(j.get("receive", 0.0)), 1),
            "heard": bool(heard),
            "first_heard_step": first_heard.get(ch),
            "n_bearings": len(obs.get(ch, ())),
            "cleared": ch in cleared,
            "diameter_survey_m": rec.get("diameter_survey_m"),
            "n_probe": rec.get("n_probe", 0),
            "diameter_final_m": rec.get("diameter_final_m"),
            "mec_radius_final_m": rec.get("mec_radius_m"),
            "clear_radius_m": rec.get("clear_radius_m"),
            "method": rec.get("method"),
            "localize_err_m": None if err is None else round(err, 2),
        })
    errs = [r["localize_err_m"] for r in rows if r["localize_err_m"] is not None]
    return {
        "sources": rows,
        "n_sources": len(rows),
        "n_heard": sum(1 for r in rows if r["heard"]),
        "all_heard": all(r["heard"] for r in rows),
        "n_cleared": sum(1 for r in rows if r["cleared"]),
        "localize_err_mean_m": round(float(np.mean(errs)), 2) if errs else None,
        "localize_err_max_m": round(float(np.max(errs)), 2) if errs else None,
        "n_within_clear_radius": sum(1 for r in rows if r["cleared"]),
        "worst_first_heard_step": max((r["first_heard_step"] for r in rows
                                       if r["first_heard_step"] is not None), default=None),
    }


def episode_row(ep: int, seed: Optional[int], truth: Optional[Sequence[dict]],
                dog, stats: dict, check: dict) -> dict:
    """一局的汇总行（整批汇总表 + 落盘 JSON 都用它）。"""
    return {
        "episode": int(ep),
        "seed": seed,
        "mode": "practice" if truth is not None else "official",
        "n_sources": check.get("n_sources"),
        "heard": check.get("n_heard"),
        "cleared": stats["cleared"],
        "clear_ratio": (check["n_cleared"] / check["n_sources"]
                        if check.get("n_sources") else None),
        "virtual_time_s": stats["virtual_time_s"],
        "travel_m": stats["travel_m"],
        "n_measure": stats["n_measure"],
        "n_clear": stats["n_clear"],
        "n_bearings": stats["n_bearings"],
        "n_probe": stats["n_probe"],
        "n_refined": stats["n_refined"],
        "n_side_scan": stats.get("n_side_scan", 0),
        "n_side_cand": stats.get("n_side_cand", 0),
        "n_side_skip_far": stats.get("n_side_skip_far", 0),
        "n_skip_measure": stats["n_skip_measure"],
        "avg_time_s": stats["avg_time_s"],
        "localize_err_mean_m": check.get("localize_err_mean_m"),
        "localize_err_max_m": check.get("localize_err_max_m"),
        "worst_first_heard_step": check.get("worst_first_heard_step"),
        "first_heard": stats.get("first_heard"),
        "methods": stats.get("methods"),
    }


def observation_rows(ep: int, plan: SweepPlan, meas: Dict[int, List[Meas]]) -> List[dict]:
    """逐条观测明细（channel, x, y, outcome, theta, stage, episode）。"""
    rows = []
    for ch in sorted(meas):
        for m in meas[ch]:
            rows.append({"episode": ep, "channel": ch,
                         "x": round(m.x, 2), "y": round(m.y, 2),
                         "outcome": m.outcome,
                         "theta_deg": None if m.theta is None else round(m.theta, 3),
                         "stage": m.stage})
    return rows


def save_plan(plan: SweepPlan, save_dir: Path, verify: Optional[dict] = None) -> List[Path]:
    """扫描方案落盘：JSON（含听到率统计）与 CSV（测量点坐标）。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    plan = SweepPlan(outer_n=plan.outer_n, outer_radius=plan.outer_radius,
                     points=plan.points, route=plan.route, route_m=plan.route_m,
                     verification=verify if verify is not None else plan.verification)
    paths = [save_dir / PLAN_JSON]
    (save_dir / PLAN_JSON).write_text(
        json.dumps(plan.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = save_dir / PLAN_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "x_m", "y_m", "in_region"])
        for k, i in enumerate(plan.route):
            x, y = plan.points[i]
            w.writerow([k, f"{x:.2f}", f"{y:.2f}",
                        "yes" if float(np.hypot(x, y)) <= 1800.0 else "no"])
    paths.append(csv_path)
    return paths


def save_survey(save_dir: Path, rows: Sequence[dict], observations: Sequence[dict],
                plan_json: dict, meta: dict) -> List[Path]:
    """逐局统计（JSON）与观测明细（CSV）落盘。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": meta,
        "sweep_plan": plan_json,
        "episodes": [dict(r) for r in rows],
    }
    paths = [save_dir / SURVEY_JSON]
    (save_dir / SURVEY_JSON).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = save_dir / OBS_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "channel", "x_m", "y_m", "outcome", "theta_deg", "stage"])
        for o in observations:
            w.writerow([o["episode"], o["channel"], o["x"], o["y"], o["outcome"],
                        "" if o["theta_deg"] is None else f"{o['theta_deg']:.2f}", o["stage"]])
    paths.append(csv_path)
    return paths