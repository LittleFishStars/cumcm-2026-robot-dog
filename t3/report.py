"""结果核对与落盘：真值核对、整批汇总、覆盖方案/巡视统计/观测明细的写出

演练模式下能拿到干扰源真值，于是可以逐源核对两件本该保证的事：
* 覆盖保证：源到最近覆盖圆圆心的距离 ≤ 1000 m，且它的频道确实被听到了；
* 清除结果：清除点与真值的距离，也就是定位误差，是否落在 20 m 清除半径内。

正式模式下真值不可见，这些字段自动省略，但落盘格式与逐局统计保持不变。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common.geometry import dist
from t3.config import (CLEAR_RADIUS, COVER_RADIUS, OBS_CSV, PLAN_CSV, PLAN_JSON,
                             RECEIVE_MAX, SURVEY_JSON)
from t3.covering import CoverPlan, CoverSolveResult
from t3.regions import Meas, Obs


def truth_check(truth: Sequence[dict] | None, plan: CoverPlan,
                obs: dict[int, list[Obs]], cleared: set,
                tracks: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    """逐源核对，只有演练模式拿得到真值

    核两件事。一是覆盖保证：源到最近覆盖圆圆心的距离是否 ≤ 1000 m，它的频道是否真的被听到。
    二是清除结果：清除点与真值的距离就是定位误差，看它是否落在 20 m 清除半径内。
    """
    tracks = tracks or {}
    rows: list[dict[str, Any]] = []
    for j in truth or []:
        p = (float(j["position"]["x"]), float(j["position"]["y"]))
        d_near = float(np.linalg.norm(plan.waypoints - np.asarray(p), axis=1).min())
        heard = len(obs.get(j["channel"], ())) > 0 or j["channel"] in cleared
        rec = tracks.get(int(j["channel"]), {})
        cp = rec.get("clear_point")
        err = dist(cp, p) if cp else None
        rows.append({"channel": int(j["channel"]), "x": p[0], "y": p[1],
                     "receive_m": float(j["receive"]),
                     "nearest_center_m": round(d_near, 2),
                     "heard": bool(heard),
                     "n_bearings": len(obs.get(j["channel"], ())),
                     "cleared": j["channel"] in cleared,
                     "diameter_survey_m": rec.get("diameter_survey_m"),
                     "n_probe": rec.get("n_probe", 0),
                     "diameter_final_m": rec.get("diameter_final_m"),
                     "mec_radius_final_m": rec.get("mec_radius_m"),
                     "clear_radius_m": rec.get("clear_radius_m"),
                     "method": rec.get("method"),
                     "localize_err_m": None if err is None else round(err, 2)})
    errs = [r["localize_err_m"] for r in rows if r["localize_err_m"] is not None]
    return {
        "sources": rows,
        "worst_nearest_m": round(max((r["nearest_center_m"] for r in rows), default=0.0), 2),
        "all_within_cover": all(r["nearest_center_m"] <= COVER_RADIUS for r in rows),
        "missed_channels": [r["channel"] for r in rows if not r["heard"]],
        "n_cleared": sum(1 for r in rows if r["cleared"]),
        "localize_err_mean_m": round(float(np.mean(errs)), 3) if errs else None,
        "localize_err_max_m": round(float(np.max(errs)), 3) if errs else None,
        "n_within_clear_radius": sum(1 for e in errs if e <= CLEAR_RADIUS),
        "n_cleared_after_probe": sum(1 for r in rows if r["n_probe"] and r["cleared"]),
    }


# ----------------------------------------------------------------------------
# 本地演练场：拉起 jammers-py 并用其控制台 REST 开一局

def save_plan(res: CoverSolveResult, save_dir: Path) -> list[Path]:
    """覆盖圆方案落盘：一份 JSON 带校验与对照，一份 CSV 记圆心坐标"""
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / PLAN_JSON
    json_path.write_text(json.dumps(res.to_json(), ensure_ascii=False, indent=2),
                         encoding="utf-8")
    csv_path = save_dir / PLAN_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "x_m", "y_m", "ring_radius_m", "cover_radius_m"])
        for i, (x, y) in enumerate(res.plan.centers):
            w.writerow([i, "center" if i == 0 else "ring", f"{x:.3f}", f"{y:.3f}",
                        f"{res.plan.ring_radius:.3f}", f"{res.plan.cover_radius:.1f}"])
    return [json_path, csv_path]


def summarize(rows: Sequence[dict]) -> dict[str, Any]:
    """整批演练的阶段二汇总：清除率、时间、定位误差、清除方式分布"""
    src = [s for r in rows for s in r.get("truth_check", {}).get("sources", ())]
    methods: dict[str, int] = {}
    for s in src:
        methods[str(s.get("method"))] = methods.get(str(s.get("method")), 0) + 1
    errs = [s["localize_err_m"] for s in src if s.get("localize_err_m") is not None]
    n_src = sum(r["n_sources"] for r in rows if r["n_sources"])
    return {
        "episodes": len(rows),
        "n_sources": n_src,
        "n_cleared": sum(r.get("cleared", 0) for r in rows),
        "clear_ratio": (sum(r.get("cleared", 0) for r in rows) / n_src) if n_src else None,
        "avg_time_s": round(float(np.mean([r["avg_time_s"] for r in rows
                                           if r.get("avg_time_s")])), 3),
        "virtual_time_s_mean": round(float(np.mean([r["virtual_time_s"] for r in rows])), 3),
        "travel_m_mean": round(float(np.mean([r["travel_m"] for r in rows])), 1),
        "n_measure_mean": round(float(np.mean([r["n_measure"] for r in rows])), 1),
        "n_probe_mean": round(float(np.mean([r["n_probe"] for r in rows])), 2),
        "n_precise_at_survey": sum(r.get("n_precise_at_survey", 0) for r in rows),
        "n_skip_measure": sum(r.get("n_skip_measure", 0) for r in rows),
        "n_inline_cleared": sum(r.get("n_inline_cleared", 0) for r in rows),
        "n_inline_fail": sum(r.get("n_inline_fail", 0) for r in rows),
        "n_refined": sum(r.get("n_refined", 0) for r in rows),
        "localize_err_mean_m": round(float(np.mean(errs)), 3) if errs else None,
        "localize_err_max_m": round(float(np.max(errs)), 3) if errs else None,
        "n_within_clear_radius": sum(1 for e in errs if e <= CLEAR_RADIUS),
        "methods": methods,
        # 布局旋转：每局按起始扫描听到的源把覆盖圆环转到"源最密集的 60° 扇区"，所以旋转角
        # 逐局不同。它是策略输出，不是设计参数。加 --no-rotate 时恒为 0。
        "rotation_deg_mean": round(float(np.mean([r.get("rotation_deg", 0.0)
                                                  for r in rows])), 3),
        "rotation_deg_min": round(float(np.min([r.get("rotation_deg", 0.0)
                                                for r in rows])), 3),
        "rotation_deg_max": round(float(np.max([r.get("rotation_deg", 0.0)
                                                for r in rows])), 3),
        "n_episodes_rotated": sum(1 for r in rows if abs(r.get("rotation_deg", 0.0)) > 1e-9),
        "n_face_scanned_mean": round(float(np.mean([r.get("n_face_scanned", 0)
                                                    for r in rows])), 2),
    }


def save_survey(save_dir: Path, rows: list[dict], observations: list[dict],
                plan_json: dict[str, Any], meta: dict | None = None) -> list[Path]:
    """巡视扫描结果落盘：逐局统计 JSON + 逐条观测 CSV

    `meta` 说明这次运行的来源，mode 取 practice 或 official，另含地址、局数等。这里不写参赛
    队号：队号一律运行时经 `--robot-id` 传入、只用于通信，落盘会把它带进交付物，而竞赛要求
    交付物里不含身份信息。结果目录已不按模式分家，目录名和文件名都不再透露模式，来源只能靠
    这份元数据交代。官方产物没有真值，接口不返回；`mode` 是事后判断"这批数字为何缺真值字段"
    的唯一线索。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / SURVEY_JSON
    json_path.write_text(json.dumps({
        "stage": "覆盖圆求解 + 依次到圆心巡视扫描 + 就近试清（未命中按文献准则补测缩小后再清）",
        "meta": meta or {},
        "cover_plan": plan_json,
        "clear_radius_m": CLEAR_RADIUS,
        "receive_radius_m": [COVER_RADIUS, RECEIVE_MAX],
        "summary": summarize(rows),
        "episodes": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = save_dir / OBS_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "channel", "stage", "x_m", "y_m", "outcome", "svd_deg",
                    "nearest_center_m"])
        for o in observations:
            w.writerow([o["episode"], o["channel"], o["stage"], f"{o['x']:.2f}",
                        f"{o['y']:.2f}", o["outcome"],
                        "" if o["theta"] is None else f"{o['theta']:.2f}",
                        f"{o['nearest_center_m']:.2f}"])
    return [json_path, csv_path]


def episode_row(ep: int, seed: int | None, truth: Sequence[dict] | None,
                dog: RobotDog,
                stats: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    """单局汇总行：把引擎统计、真值核对与逐频道档案合成一行，供 JSON 与绘图使用

    `seed` 在演练模式下是本局的随机种子；官方模式的场景由平台生成，不受我们控制，所以传 None。
    `truth` 同理：官方模式拿不到真值，传 None 之后需要真值的指标都留空，定位误差就是其中一个。
    """
    n_src = len(truth) if truth else stats.get("channels_heard")
    cleared = stats.get("cleared", 0)
    return {
        "episode": ep, "seed": seed,
        "n_sources": len(truth) if truth else None,
        "clear_ratio": (cleared / n_src) if n_src else None,
        "localize_err_mean_m": check.get("localize_err_mean_m"),
        "localize_err_max_m": check.get("localize_err_max_m"),
        "n_within_clear_radius": check.get("n_within_clear_radius"),
        "n_cleared_after_probe": check.get("n_cleared_after_probe"),
        **stats,
        "waypoint_stats": dog.waypoint_stats,
        "plan_centers": dog.plan.centers,          # 本局实际圆心（旋转后的位置）
        "truth_check": check,
        "bearings_per_channel": {str(c): len(v) for c, v in sorted(dog.obs.items())},
        "cleared_channels": sorted(dog.cleared),
    }

def observation_rows(ep: int, plan: CoverPlan, meas: dict[int, list[Meas]]) -> list[dict]:
    """把本局全部测量整理成 CSV 行，no_signal 也留在里面，另附测量点到最近圆心的距离"""
    rows = []
    for ch, ml in sorted(meas.items()):
        for m in ml:
            d = float(np.linalg.norm(plan.waypoints - np.array([m.x, m.y]), axis=1).min())
            rows.append({"episode": ep, "channel": ch, "x": m.x, "y": m.y,
                         "outcome": m.outcome, "theta": m.theta, "stage": m.stage,
                         "nearest_center_m": d})
    return rows
