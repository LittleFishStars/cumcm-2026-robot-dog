# AGENTS.md

2026 CUMCM Problem B: a robot dog searching for and clearing jammers (the solve code for CUMCM
mathematical-modelling problems 1-4).

## Repo layout

- Git repo root is **this** directory (`cumcm-2026-robot-dog/`). Run all commands from here.
- Code is flat at the repo root: `common/`, `t1/`, `t2/`, `t3/`, `t4/`, `analysis/` are
  top-level packages (no enclosing `cumcm` package, no `T*.py` shim scripts). Each problem
  package ships a `__main__.py`, so `python -m t3 --practice 3` replaces the old
  `python T3.py --practice 3`; `python -m t4.sweep`, `python -m analysis.undefined_names`, etc.
  run the self-checks.
- Layering is a hard convention: deps only point down `common <- t1/t2/t3/t4 <- analysis`
  (`t2` may import `t1` — same wedge-intersection duct, vectorized; `t4` may import `t1` and
  `t3.probing`).
  Never import upward (e.g. `common` must not import `t3`).
- `results/` holds the generated solve artifacts and is **gitignored**: `python -m t2` writes the six
  T2 files, `--plan-only` the T3/T4 plans, practice runs the per-episode CSV/JSON + `scan/` +
  `trajectory/`. Re-running regenerates them in place and the same `--seed` is byte-identical — the
  only exception is `api_calls.jsonl`, which carries wall-clock timestamps. Never `git add -f` them;
  prove reproducibility by hashing the files (see Verification) instead of with `git status`.
- Figures were deleted at the user's request: the old `figures/` directory, the `figures_t4*.py` /
  `T4_figures.py` generators, `paper/` (paper sources incl. `make_figures.py`) and `reports/` (stage
  reports) are no longer in the repo. `python -m t2` writes its figures into `results/t2/`, problem 4's
  plan/points into `results/t4/`; rebuild any paper output from those, not from a checked-in figure dir.

## Setup / environment

- `pyproject.toml` requires **Python >= 3.14** and uses the Tsinghua PyPI mirror; the stack is
  Python 3.14 + numpy/matplotlib/shapely.
- Use `uv sync` then `uv run python <script>`. README's `.venv/bin/python` is POSIX-only; on Windows
  use `uv run` or `.venv\Scripts\python.exe`. Prefer `python -X utf8 ...` so Chinese output renders.

## Commands that work offline

- `uv run python -m t1` — problem 1 demo.
- `uv run python -m t2` — problem 2: best second detection point + suitability figure + literature
  criteria figure (offline, ~15 s; writes `results/t2/{t2_suitability.png,pdf,t2_criteria.png,pdf,
  t2_suitability.csv,t2_second_site.json}`).
- `uv run python -m t2.region` — problem 2 geometry self-check (analytic quad vs shapely).
- `uv run python -m t2.theory` — problem 2 literature self-check (CRLB/GDOP closed forms vs
  the exact set-membership worst-case bound; prints per-γ and per-r₂/d deviation buckets).
- `uv run python -m t3 --plan-only` — solve the 7 cover circles; no simulator, no ground truth.
- `uv run python -m t3.covering` — cover-layout self-check (rotation/selection/min-circle).
- `uv run python -m t4 --plan-only` — solve the 20 measurement positions + hearing-rate statistics
  (~2 s; writes `results/t4/t4_sweep_plan.json` + `t4_sweep_points.csv`).
- `uv run python -m t4 --practice 20 --seed 0 --layout-file <layout.npy> --save-dir <dir>` — benchmark
  an alternative measurement layout with the same harness (benchmark any candidate layout on the same
  seed set).
- `uv run python -m t4.sweep` — problem 4 scan self-check (batched hearing predicate vs the
  single-case reference, then the 4.2 M-case statistics without touching any simulator).
- `uv run python -m analysis.undefined_names` — static scan of every repo `.py` for names read
  but never bound / parameters that silently do not exist (the de-facto lint).

## Local simulator (`resources/jammers-py/`)

- `resources/jammers-py/` is the local replica simulator used by `--practice` (`common/paths.py`
  resolves it via the `JAMMERS_REL_DIR` constant; `--jammers-dir` overrides it).
- It is **not tracked by git** and may be absent in a fresh clone; if missing, only the offline
  commands above run. If present, `--practice` and ground-truth checks work.
- `uv run python -m simulator --help` or `cd resources/jammers-py; uv run python run.py`.

## Running against the official simulator

- `python -X utf8 -m t3` (no args) connects to `http://127.0.0.1:2026`.
  The simulator is Windows-only and listens on localhost only, so it must run on the same machine.
- **Gotcha:** the official `jammers-simulator.exe` is a long-running process that holds port 2026.
  While it runs, a local `--practice` launch fails with `WinError 10013`/port-busy. `python -m t3`
  accepts `--robot-port`/`--console-port`; run practice on e.g.
  `--practice 20 --no-reuse --robot-port 2111 --console-port 8091`. Never test-run against the
  official simulator: it consumes one of only 3 official attempts.
- Official and practice modes write the **same** `--save-dir` (`results/t3/`);
  one directory = the latest run. `--plan-only` writes only the plan.
- Every API call is appended to `<save-dir>/api_calls.jsonl`. In official mode there is no ground
  truth, so that log is the only evidence trail (check `meta.mode` to tell runs apart).
- Practice runs are byte-identical for a fixed `--seed` (scenario layout + bearing-noise seed are
  both derived from it), so A/B comparisons must use the same seed set.

## Verification (no test/lint/CI exists)

- There is no test suite, linter config, or CI workflow. The checks are: the module self-checks
  (`t2.region`, `t2.theory`, `t3.covering`, `t4.sweep`,
  `analysis.undefined_names`) and byte-identical fixed-seed reruns. `python -m t2` is fully
  deterministic (no RNG): its six outputs are byte-identical across runs, and every run self-checks
  the analytic geometry against shapely (max rel. dev ~1e-15), a 3x-refined discretization,
  the literature CRLB/GDOP closed forms, and a 5 m global GDOP-prefiltered certification of
  the optimum.
- **Optimization invariant:** performance work must keep the artifacts byte-identical. The batched
  hot paths (`t2.region.quad_diameters_batch` / `t2.score.worst_case_diameters`, `t3.covering`
  distance kernels, `t4.sweep._hit_cases`) are built so each element performs exactly the same
  floating-point operations as the scalar reference; `python -m t2`, `python -m t3 --plan-only` and
  `python -m t4 --plan-only` outputs stay byte-identical before/after. Deviations that were rejected for
  this reason: replacing `np.hypot` by `sqrt(x²+y²)` in the T2 diameter pair loop (last-bit drift in
  `t2_second_site.json`).
- Design rationale lives in the module docstrings; `README.md` only covers environment and usage.

## Code style (follows the author's conventions in `~/Projects`)

Every Python file in this repo follows the conventions below. They are not an external standard but
the author's own habit across `~/Projects/python/MBridge`, `~/Projects/python/GNNU_API` and
`~/Projects/python/CPUISEditor`; new code and edits follow them too.

- **Docstrings**: every module, class, function and method has one, and in real code it is written in
  **Chinese** — that is a project convention, not an accident of this file. The one-line summary is a
  single line **with no trailing period** (e.g. `"""构建命令行解析器"""`); when parameters or return
  values need explaining, continue with Google-style sections:

  ```python
  """One-line Chinese summary, no trailing period

  Args:
      student_id: student id
      password: password for the unified verification platform

  Returns:
      tuple[bool, Any]: whether it succeeded, plus the resulting data
  """
  ```

  The sample above shows the layout only (4-space indent inside a section, written as
  `name: description` / `type: description`); the real text is Chinese. The maths, constraints and
  pitfalls already recorded in existing module docstrings are **content** — do not trim them away in
  the name of brevity.
- **Type annotations**: annotate parameters and return values on every function and method, including
  `-> None` on `__init__` and on internal `_helper`s. **Use only builtin generics and `|`**:
  `list[float]`, `tuple[float, float]`, `dict[str, Any]`, `float | None` — never
  `typing.List/Dict/Tuple/Optional/Union` (from `typing` keep only names with no builtin replacement,
  such as `Any`, `Sequence`, `Callable`, `Iterator`, `TextIO`, `NamedTuple`). numpy arrays are typed
  `np.ndarray`, shapely geometry `Polygon`/`BaseGeometry`, matplotlib axes `Axes`. A cross-module type
  may reference only names already imported in that file; otherwise use a string forward reference
  (`"SimClient"`) — **never add a runtime import just to write an annotation** (the layering rule is a
  hard convention).
- **Comments**: put a one-line Chinese comment above any non-obvious statement saying what it does or
  why; logs, errors and printed output are Chinese as well.
- **Imports**: stdlib → third-party → local packages, one blank line between the groups; absolute
  imports only (`from common.x import y`).
- **Naming and formatting**: modules/functions/variables snake_case, classes PascalCase, internal
  symbols prefixed with `_`, constants UPPER_CASE; double quotes for strings, the single exception
  being `if __name__ == '__main__':` written with single quotes (the author's habit in personal
  projects); lines ≤ 110 characters; no trailing whitespace, no tabs.
- **Entry points**: a directly runnable module uses `if __name__ == '__main__':` and parses its
  command line with `argparse`.

**Style changes are bound by the same optimization invariant**: after a purely stylistic change (extra
annotations, docstrings, comments, rewrapped lines) the artifacts of `python -m t2`,
`python -m t3 --plan-only` and `python -m t4 --plan-only` must stay byte-identical to before, and the
self-checks (`analysis.undefined_names` and the rest) must all pass. **Annotations and docstrings must
not change any expression, literal, control flow or output text.**

How to verify this class of change (there is no test suite; the three pieces of evidence are):

1. **Full self-checks + unchanged artifacts**: before touching anything, hash the baseline inside the
   workspace with `find results -type f | sort | xargs sha256sum > .results.sha256`; after running the
   five self-checks plus `python -m t2`, `python -m t3 --plan-only` and `python -m t4 --plan-only`,
   `sha256sum -c .results.sha256` must report every entry OK, then delete the baseline file. Three
   gotchas: `results/` is no longer tracked, so `git status` can no longer tell you this; keep the
   baseline inside the workspace (this machine wipes `/tmp` between commands, so a `/tmp` baseline is
   lost immediately); and `sha256sum -c` prints `OK` or, in a Chinese locale, `成功`, so accept both
   in scripts.
2. **AST normalization audit**: parse both the `HEAD` version and the working-tree version into an AST,
   strip docstrings, signature annotations, annotated-assignment annotations, type aliases and import
   name lists, then compare `ast.dump` — they must be identical (comments never reach the AST). This
   proves that only annotations/docstrings/comments/import spelling changed. Rewrapping that hoists a
   local variable will show up here, which is expected — verify those with step 3 instead.
3. **Real-harness A/B**: for files with statement-level changes, run `--practice N --seed 0` (never the
   official simulator) once per version on the same seed, and compare the `t3_survey.json`,
   `t3_observations.csv` (same for T4) and stdout under `--save-dir`; the artifacts must be
   byte-identical. The per-call wall-clock milliseconds in stdout are random by nature, so normalize
   them (e.g. strip `（N ms`) before comparing.
