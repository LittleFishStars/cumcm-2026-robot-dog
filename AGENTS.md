# AGENTS.md

2026 CUMCM Problem B: a robot dog searching for and clearing jammers (the solve code for CUMCM
mathematical-modelling problems 1-4).

## Repo layout

Everything runs from this directory, which is also the git root of `cumcm-2026-robot-dog/`.

The code is flat. `common/`, `t1/`, `t2/`, `t3/`, `t4/` are top-level packages, with no `cumcm`
wrapper and no `T*.py` shim scripts. Every problem package ships a `__main__.py`, so
`python -m t3 --practice 3` is what used to be `python T3.py --practice 3`, and the self-checks are
modules as well (`python -m t2.region`, `python -m t4.sweep`). The local simulator submodule
`jammers-py/` sits at the root too; it used to live under `resources/`, which is gitignored and never
submitted.

Dependencies only point down: `common <- t1/t2/t3/t4`. `t2` may import `t1` (same wedge-intersection
duct, vectorized), `t4` may import `t1` and `t3.probing`. Importing upward, `common` reaching into
`t3` for instance, is out of the question.

`results/` is gitignored output. `python -m t2` writes the six T2 files, `--plan-only` writes the
T3/T4 plans, a practice run writes the per-episode CSV/JSON plus `scan/` and `trajectory/`. Both
`t3_survey.json` and `t4_survey.json` carry a `summary` block under the same key names (`summarize()`
in `t3/report.py` and `t4/report.py`), so the two problems read side by side. Re-running rewrites the
files in place, and a fixed `--seed` gives byte-identical output; `api_calls.jsonl` is the one exception
since it carries wall-clock timestamps. Never `git add -f` them. Prove reproducibility by hashing the
files (see Verification), not with `git status`.

The old `figures/` directory, the `figures_t4*.py` / `T4_figures.py` generators, `paper/` (paper
sources, `make_figures.py` included) and `reports/` (stage reports) were deleted at the user's
request, so paper output has to be rebuilt from `results/t2/` (T2 figures) and `results/t4/` (problem
4's plan and points) instead of from a checked-in figure directory.

## Setup / environment

`pyproject.toml` requires Python >= 3.14 and points at the Tsinghua PyPI mirror. The stack is Python
3.14 plus numpy, matplotlib and shapely.

uv is mandatory: `uv sync`, then `uv run -m <package>` (long form `uv run python -m <package>`). Do
not call a bare system `python`. The `.venv` interpreter shipped with the submission package,
`.venv/bin/python`, is the only other supported entry; README documents both. Add `-X utf8` when
Chinese output has to render.

Console output has three levels, defined once in `common/console.py` and wired through each package's
CLI. Default prints a line or two of conclusions per step, `--verbose` prints the full process report,
`--quiet` prints artifact paths only. Keep new printing on the right side of that line: process detail
goes behind `is_verbose()`, conclusions stay on a plain `print`.

The team id is never stored in the code. Practice and official modes take `--robot-id` at run time,
and t3/t4 exit with code 2 when it is missing; `Simulator(robot_id=...)` has no default. `t1`, `t2`,
`--plan-only` and the module self-checks never contact a simulator, so they need no id. Keep it that
way: no team number, school or personal data in any tracked file.

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
- `uv run python -m t4 --practice 20 --seed 0 --robot-id <team id> --layout-file <layout.npy>
  --save-dir <dir>` — benchmark an alternative measurement layout with the same harness
  (benchmark any candidate layout on the same seed set).
- `uv run python -m t4.sweep` — problem 4 scan self-check (batched hearing predicate vs the
  single-case reference, then the 4.2 M-case statistics without touching any simulator).

## Local simulator (`jammers-py/`)

`jammers-py/` at the repo root is the local replica simulator that `--practice` pulls up.
`common/paths.py` finds it through the `JAMMERS_REL_DIR` constant, and `--jammers-dir` overrides
that. It sits at the root rather than under `resources/` because it ships inside the submission
package with the rest of the code, while `resources/` (problem statement, references) never does.

It is a git submodule on the private remote `LittleFishStars/jammers-py-simulator`, so a fresh clone
carries only its commit pointer. Run `git submodule update --init jammers-py` once. Without it only
the offline commands above work; with it, `--practice` and the ground-truth checks come alive.

No third-party packages are involved: it is standard library only. Its former `tools/` plot script,
the one that pulled in matplotlib and numpy, was deleted on request. The `data/` directory it creates
(practice statistics database plus behaviour logs, and those logs contain the team id) is runtime
output, and it stays out of the submission package.

`uv run python -m simulator --help` or `cd jammers-py; uv run python run.py` starts it by hand.

## Running against the official simulator

`uv run python -X utf8 -m t3` with no arguments connects to `http://127.0.0.1:2026`. The simulator is
Windows-only and listens on localhost, so it has to run on the same machine.

One trap: the official `jammers-simulator.exe` is a long-running process that holds port 2026. A local
`--practice` launch fails while it runs, with `WinError 10013` or a port-busy error. `python -m t3`
takes `--robot-port` and `--console-port`, so run practice as
`--practice 20 --no-reuse --robot-port 2111 --console-port 8091` for instance. Never test-run against
the official simulator: each run eats one of the 3 official attempts.

Official and practice modes write the same `--save-dir` (`results/t3/`), so one directory means the
latest run. `--plan-only` writes only the plan.

Every API call is appended to `<save-dir>/api_calls.jsonl`. Official mode has no ground truth, which
makes that log the only evidence trail; `meta.mode` tells runs apart.

A practice run is byte-identical for a fixed `--seed`, because both the scenario layout and the
bearing-noise seed derive from it. Any A/B comparison therefore has to use the same seed set.

## Verification (no test/lint/CI exists)

There is no test suite, linter config or CI workflow. What stands in for them: the four module
self-checks (`t2.region`, `t2.theory`, `t3.covering`, `t4.sweep`) and byte-identical fixed-seed
reruns. `python -m t2` is fully deterministic, no RNG anywhere; its six outputs are byte-identical
across runs, and each run checks its own analytic geometry against shapely (max relative deviation
around 1e-15), against a 3x-refined discretization, against the literature CRLB/GDOP closed forms,
and against a 5 m global GDOP-prefiltered certification of the optimum.

Optimization work is bound by an invariant: the artifacts have to stay byte-identical afterwards. The
batched hot paths (`t2.region.quad_diameters_batch` and `t2.score.worst_case_diameters`,
`t3.covering`'s distance kernels, `t4.sweep._hit_cases`) are written so each element performs exactly
the same floating-point operations as the scalar reference, which keeps `python -m t2`,
`python -m t3 --plan-only` and `python -m t4 --plan-only` byte-stable. One deviation was rejected for
this reason: replacing `np.hypot` by `sqrt(x²+y²)` in the T2 diameter pair loop, which shifted the
last bits of `t2_second_site.json`.

Design rationale lives in the docstrings, mostly on the functions and constants that carry it. `README.md`
only covers environment and usage.

## Code style (follows the author's conventions in `~/Projects`)

Every Python file in this repo follows the conventions below. They are not an external standard but
the author's own habit across `~/Projects/python/MBridge`, `~/Projects/python/GNNU_API` and
`~/Projects/python/CPUISEditor`; new code and edits follow them too.

- Voice: the Chinese here is prose written by the person who wrote the code, not boilerplate from
  a generator. Keep it uneven, concrete and a bit blunt. The following tics were all over the place
  before 2026-09-13 and are now banned:
  - no `**bold**` markup inside comments or docstrings. Emphasise by wording, not by punctuation;
  - no `——` used as a parenthetical dash: use a comma, a colon, or a separate sentence;
  - 全角括号 asides only when they carry numbers, units, symbol names or a real caveat. Anything that
    can be folded into the sentence should be folded, and 998 of them was far too many;
  - skip the written-register connectives (`故`, `即`, `综上`, `值得注意的是`). Plain 分句 or 所以
    reads better. 98 `故` in 10k lines is a fingerprint;
  - vary sentence length. A one-line jab ("跑一遍就知道", "这里别乱动") is welcome next to a long
    sentence that carries a whole derivation. Just don't give every paragraph the same rhythm.
- Docstrings are one-line introductions. Every module, class, function and method has one, in
  Chinese, saying what it does, with no trailing period on that line. Two lines is the ceiling, and
  only when the second one is genuinely load-bearing. No `Args:`/`Returns:` skeletons: if a parameter
  or return value really is not obvious from the call site, work it into the sentence.

  ```python
  """构建命令行解析器"""
  ```

  Rationale, derivations, measured numbers and traps do not belong in a docstring. When one of them
  has to stay near the code, put it in a `#` comment on the statement it explains, one line, and
  leave the docstring alone.
- Module docstrings follow the same rule: say what the file is for, one line. Constraints,
  derivations, traps and history belong in the comment or docstring that carries them, in the commit
  message, or in the paper; do not restate them at the top of the file.
- Type annotations: annotate parameters and return values on every function and method, including
  `-> None` on `__init__` and on internal `_helper`s. Use only builtin generics and `|`:
  `list[float]`, `tuple[float, float]`, `dict[str, Any]`, `float | None` — never
  `typing.List/Dict/Tuple/Optional/Union` (from `typing` keep only names with no builtin replacement,
  such as `Any`, `Sequence`, `Callable`, `Iterator`, `TextIO`, `NamedTuple`). numpy arrays are typed
  `np.ndarray`, shapely geometry `Polygon`/`BaseGeometry`, matplotlib axes `Axes`. A cross-module type
  may reference only names already imported in that file; otherwise use a string forward reference
  (`"SimClient"`) rather than adding a runtime import just to spell an annotation. That is the layering
  rule talking.
- Comments: put a one-line Chinese comment above any non-obvious statement saying what it does or
  why; logs, errors and printed output are Chinese as well.
- Imports: stdlib → third-party → local packages, one blank line between the groups; absolute
  imports only (`from common.x import y`).
- Naming and formatting: modules/functions/variables snake_case, classes PascalCase, internal
  symbols prefixed with `_`, constants UPPER_CASE; double quotes for strings, the single exception
  being `if __name__ == '__main__':` written with single quotes (the author's habit in personal
  projects); lines ≤ 110 characters; no trailing whitespace, no tabs.
- Entry points: a directly runnable module uses `if __name__ == '__main__':` and parses its
  command line with `argparse`.

A pure text pass, meaning comments and docstrings only, is still bound by the artifact invariant:
afterwards the artifacts of `python -m t2`, `python -m t3 --plan-only` and `python -m t4 --plan-only`
must be byte-identical to before, and the four module self-checks must pass. Rewording CLI help or
printed output is allowed and naturally changes stdout; when a descriptive string that goes into an
artifact changes along with it (`layout_kind`, `note`, `stage` and friends), compare artifacts by keys
and numbers instead of by bytes. Either way one rule holds: no expression, literal value, control flow
or numeric format may move. Wording only.

Three ways to check a change like this:

1. Self-checks plus artifacts. Hash the baseline inside the workspace first:
   `find results -type f | sort | xargs sha256sum > .results.sha256`. Then run the four self-checks and
   the three artifact commands, and finish with `sha256sum -c .results.sha256`, where every entry must
   come back OK. Delete the baseline file afterwards. Three things to watch: `results/` is untracked,
   so `git status` cannot tell you any of this; keep the baseline inside the workspace, since this
   machine wipes `/tmp` between commands; and `sha256sum -c` prints `成功` instead of `OK` under a
   Chinese locale, so accept both.
2. Text-only audit. Parse the `HEAD` version and the working-tree version, strip docstrings, blank out
   every string constant, and compare `ast.dump`. Identical dumps prove the edit touched text and
   nothing else. Rewrapping that hoists a local variable will show up here, which is what step 3 is
   for.
3. Real-harness A/B. For anything with a statement-level change, run `--practice N --seed 0` once per
   version, never against the official simulator, and compare `t3_survey.json` and
   `t3_observations.csv` (same pair for T4) under `--save-dir`; they must be byte-identical. The
   per-call wall-clock milliseconds in stdout are random by nature, so strip `（N ms` before diffing.
