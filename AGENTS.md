# AGENTS.md

2026 CUMCM B 题：机器狗搜索与清除干扰源（CUMCM 数学建模求解代码）。

## Repo layout

- Git repo root is **this** directory (`cumcm-2026-robot-dog/`). Run all commands from here.
- Top-level `T1.py` / `T3.py` / `T3_ga.py` / `T3_validate.py` / `T3_figures.py` / `sim_api.py`
  are thin shims. Real code is in `cumcm/` (`common`, `t1`, `t3`, `t3ga`, `analysis`). Edit the
  package, not the shims; keep the same CLI behavior.
- Layering is a hard convention: deps only point down `common <- t1/t3/t3ga <- analysis`.
  Never import upward (e.g. `common` must not import `t3`).
- `results/` and `figures/` are **committed solve artifacts** (not gitignored). Running the code
  regenerates them in place, and the same `--seed` is byte-identical. Expect a dirty tree after a run;
  do not commit regenerated artifacts unless asked.

## Setup / environment

- `pyproject.toml` requires **Python >= 3.14** and uses the Tsinghua PyPI mirror. This is
  authoritative — `reports/OFFICIAL_PLATFORM_GUIDE.md` still says 3.12/3.13 (stale).
- Use `uv sync` then `uv run python <script>`. README's `.venv/bin/python` is POSIX-only; on Windows
  use `uv run` or `.venv\Scripts\python.exe`. Prefer `python -X utf8 ...` so Chinese output renders.

## Commands that work offline

- `uv run python T1.py` — problem 1 demo.
- `uv run python T3.py --plan-only` — solve the 7 cover circles; no simulator, no ground truth.
- `uv run python -m cumcm.t3.covering` — cover-layout self-check (rotation/selection/min-circle).
- `uv run python T3_figures.py` — rebuild paper figures into `figures/` from `results/t3_ga/`.

## Local simulator (`jammers-py/`)

- `jammers-py/` is the local replica simulator used by `--practice` and by `T3_validate.py`
  (`validate.py` does `from simulator import bearingnoise` with `jammers-py` on `sys.path`).
- It is **not tracked by git** and may be absent in a fresh clone; if missing, only the offline
  commands above run. If present, `--practice` and the six-group validation work.
- `uv run python -m simulator --help` or `cd jammers-py; uv run python run.py`.

## Running against the official simulator

- `python -X utf8 T3.py` (no args) connects to `http://127.0.0.1:2026`; `T3_ga.py` has an identical CLI.
  The simulator is Windows-only and listens on localhost only, so it must run on the same machine.
- **Gotcha:** the official `jammers-simulator.exe` is a long-running process that holds port 2026.
  While it runs, a local `--practice` launch fails with `WinError 10013`/port-busy. Both `T3.py`
  and `T3_ga.py` accept `--robot-port`/`--console-port`; run practice on e.g.
  `--practice 20 --no-reuse --robot-port 2111 --console-port 8091`. Never test-run against the
  official simulator: it consumes one of only 3 official attempts.
- Official and practice modes write the **same** `--save-dir` (`results/t3/`, `results/t3_ga/`);
  one directory = the latest run. Fixed order: run a practice batch first, then validate/figures —
  an official run overwrites the practice artifacts. `--plan-only` writes only the plan.
- Every API call is appended to `<save-dir>/api_calls.jsonl`. In official mode there is no ground
  truth, so that log is the only evidence trail (check `meta.mode` to tell runs apart).
- Practice runs are byte-identical for a fixed `--seed` (scenario layout + bearing-noise seed are
  both derived from it), so A/B comparisons must use the same seed set.

## Verification (no test/lint/CI exists)

- There is no test suite, linter config, or CI workflow. Treat `python -m cumcm.t3.covering`, the
  six-group `T3_validate.py` (needs `jammers-py/`), and byte-identical fixed-seed reruns as the checks.
- Design rationale lives in the module docstrings, `README.md`, and `reports/RESULTS_REPORT.md`.
