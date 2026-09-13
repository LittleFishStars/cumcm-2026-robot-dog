"""问题四的结果核对与落盘"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common.geometry import dist
from t4.config import OBS_CSV, PLAN_CSV, PLAN_JSON, SURVEY_JSON
from t4.regions import Meas, Obs
from t4.sweep import SweepPlan


def truth_check(truth: Sequence[dict] | None, plan: SweepPlan,
                obs: dict[int, list[Obs]], cleared: set, tracks: dict,
                first_heard: dict[int, int] | None = None) -> dict[str, Any]:
    """逐源核对，只有演练模式拿得到真值

    * 检测：该源频道是否被听到，拖网保证每个源 ≥ 1 次示向度，同时记下首次听到的拖网步骤；
      定向源额外记下它的定向方向，用来交代"方向不对"的场景也被拖网兜住了，听到就证明命中；
    * 清除：清除点与真值的距离（定位误差）、判定是否在 20 m 清除半径内。
    """
    tracks = tracks or {}
    first_heard = first_heard or {}
    rows: list[dict[str, Any]] = []
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


def summarize(rows: Sequence[dict]) -> dict[str, Any]:
    """整批演练汇总：清除率、时间、里程、测向次数、定位误差与清除方式分布

    键名与问题三的 summarize 对齐，两份 survey 可以并排读；问题四另外给出拖网特有的两项：
    首次听到的最晚步骤，以及顺路补测次数。清除方式由逐局的 methods 累加。
    官方模式拿不到真值，定位误差与首次听到步会留空，其余字段照常。
    """
    n_src = sum(r["n_sources"] for r in rows if r.get("n_sources"))
    n_cleared = sum(r.get("cleared", 0) for r in rows)
    avg_times = [r["avg_time_s"] for r in rows if r.get("avg_time_s")]
    means = [r["localize_err_mean_m"] for r in rows if r.get("localize_err_mean_m") is not None]
    maxes = [r["localize_err_max_m"] for r in rows if r.get("localize_err_max_m") is not None]
    heard = [r["worst_first_heard_step"] for r in rows
             if r.get("worst_first_heard_step") is not None]
    methods: dict[str, int] = {}
    for r in rows:
        for name, cnt in (r.get("methods") or {}).items():
            methods[str(name)] = methods.get(str(name), 0) + int(cnt)
    # 策略侧会把没用到的清除方式也记成 0，分布里只留真出现过的
    methods = {k: v for k, v in methods.items() if v}
    return {
        "episodes": len(rows),
        "n_sources": n_src,
        "n_cleared": n_cleared,
        "clear_ratio": (n_cleared / n_src) if n_src else None,
        "avg_time_s": round(float(np.mean(avg_times)), 3) if avg_times else None,
        "virtual_time_s_mean": round(float(np.mean([r["virtual_time_s"] for r in rows])), 3),
        "virtual_time_s_min": round(float(np.min([r["virtual_time_s"] for r in rows])), 3),
        "virtual_time_s_max": round(float(np.max([r["virtual_time_s"] for r in rows])), 3),
        "travel_m_mean": round(float(np.mean([r["travel_m"] for r in rows])), 1),
        "n_measure_mean": round(float(np.mean([r["n_measure"] for r in rows])), 1),
        "n_probe_mean": round(float(np.mean([r["n_probe"] for r in rows])), 2),
        "n_skip_measure": sum(r.get("n_skip_measure", 0) for r in rows),
        "n_inline_cleared": sum(r.get("n_inline", 0) for r in rows),
        "n_inline_fail": sum(r.get("n_inline_fail", 0) for r in rows),
        "n_side_scan_mean": round(float(np.mean([r.get("n_side_scan", 0) for r in rows])), 2),
        "localize_err_mean_m": round(float(np.mean(means)), 3) if means else None,
        "localize_err_max_m": round(float(np.max(maxes)), 3) if maxes else None,
        "worst_first_heard_step": int(max(heard)) if heard else None,
        "methods": methods,
    }


def episode_row(ep: int, seed: int | None, truth: Sequence[dict] | None,
                dog: "RobotDog", stats: dict, check: dict) -> dict:
    """一局的汇总行，整批汇总表与落盘 JSON 都用它

    Args:
        ep: 局号，从 1 开始
        seed: 该局的种子；官方模式的场景不由 --seed 控制，记 None 以免误读
        truth: 干扰源真值列表，官方模式没有，为 None
        dog: 该局的机器人策略实例；本行字段取自 stats 与 check，这个参数只为统一调用口径留着
        stats: dog.run() 返回的本局统计
        check: truth_check() 给出的真值核对结果

    Returns:
        dict: 汇总字段，清除比例、里程、虚拟时间、测向次数、定位误差等
    """
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
        "n_inline": stats.get("n_inline", 0),
        "n_inline_fail": stats.get("n_inline_fail", 0),
        "n_skip_measure": stats["n_skip_measure"],
        "avg_time_s": stats["avg_time_s"],
        "localize_err_mean_m": check.get("localize_err_mean_m"),
        "localize_err_max_m": check.get("localize_err_max_m"),
        "worst_first_heard_step": check.get("worst_first_heard_step"),
        "first_heard": stats.get("first_heard"),
        "methods": stats.get("methods"),
    }


def observation_rows(ep: int, plan: SweepPlan, meas: dict[int, list[Meas]]) -> list[dict]:
    """逐条观测明细：channel, x, y, outcome, theta, stage, episode"""
    rows = []
    for ch in sorted(meas):
        for m in meas[ch]:
            rows.append({"episode": ep, "channel": ch,
                         "x": round(m.x, 2), "y": round(m.y, 2),
                         "outcome": m.outcome,
                         "theta_deg": None if m.theta is None else round(m.theta, 3),
                         "stage": m.stage})
    return rows


def save_plan(plan: SweepPlan, save_dir: Path, verify: dict | None = None) -> list[Path]:
    """扫描方案落盘：JSON 带听到率统计，CSV 存测量点坐标"""
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
                plan_json: dict, meta: dict) -> list[Path]:
    """逐局统计 JSON 与观测明细 CSV 落盘"""
    save_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": meta,
        "sweep_plan": plan_json,
        "summary": summarize(rows),
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