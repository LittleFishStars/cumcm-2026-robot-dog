# AGENTS.md

2026 CUMCM B 题：机器狗搜索与清除干扰源（CUMCM 数学建模求解代码）。

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
- `results/` holds **committed solve artifacts** (not gitignored): the six T2 files, the T3 plan +
  per-episode CSV/JSON + `scan/` + `trajectory/`, and the T4 analogue. Running the code regenerates
  them in place, and the same `--seed` is byte-identical — the only exception is `api_calls.jsonl`,
  which carries wall-clock timestamps. Expect a dirty tree after a run; the committed artifacts are
  the **latest** run, so refresh them deliberately rather than mixing runs.
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
- Design rationale lives in the module docstrings and `README.md`.

## 代码风格（沿用作者在 `~/Projects` 下个人项目的约定）

本仓库所有 Python 代码统一按下面这套约定书写。它不是外部规范，而是作者自己在
`~/Projects/python/MBridge`、`~/Projects/python/GNNU_API`、`~/Projects/python/CPUISEditor`
等项目里一贯的写法，新代码与改动都照此办理。

- **docstring**：模块、类、函数、方法都有中文 docstring。一句话摘要写成一行、**结尾不加句号**
  （如 `"""构建命令行解析器"""`）；需要说明参数/返回值时接 Google 风格段：

  ```python
  """登陆验证

  Args:
      student_id: 学号
      password: 统一验证平台密码

  Returns:
      tuple[bool, Any]: 是否成功与结果数据
  """
  ```

  段内 4 空格缩进，写作 `名字: 说明`、`类型: 说明`。现有模块 docstring 里的数学推导、约束条件与
  踩过的坑属于**内容**，不要为了"简洁"压缩掉。
- **类型注解**：函数/方法一律标注参数与返回值，`__init__` 标 `-> None`，内部 `_helper` 同样标注。
  **只用内建泛型与 `|`**：`list[float]`、`tuple[float, float]`、`dict[str, Any]`、`float | None`，
  不写 `typing.List/Dict/Tuple/Optional/Union`（`typing` 里只留 `Any`、`Sequence`、`Callable`、
  `Iterator`、`TextIO`、`NamedTuple` 这类没有内建替身的名字）。numpy 数组用 `np.ndarray`、shapely
  几何用 `Polygon`/`BaseGeometry`、matplotlib 轴用 `Axes`。跨模块类型只引用本文件已导入的名字，
  否则用字符串前向引用（`"SimClient"`）——**不要为了写注解新增运行时 import**（分层依赖是硬约定）。
- **注释**：不明显的语句上方加一行中文注释说明"做什么/为什么"；日志、报错与打印文本一律中文。
- **导入**：stdlib → 第三方 → 本地包，三组之间空一行；只用绝对导入（`from common.x import y`）。
- **命名与格式**：模块/函数/变量 snake_case，类 PascalCase，内部符号 `_` 前缀，常量全大写；
  字符串统一双引号，唯一例外是 `if __name__ == '__main__':` 写成单引号（沿用作者个人项目习惯）；
  行长 ≤ 110 字符；不留行尾空白、不写制表符。
- **入口**：可直接运行的模块写 `if __name__ == '__main__':`，命令行解析统一用 `argparse`。

**风格改动同样受"优化不变量"约束**：纯风格改动（补注解、补 docstring、加注释、折行）后，
`python -m t2`、`python -m t3 --plan-only`、`python -m t4 --plan-only` 的产物必须与改动前逐字节一致，自检
（`analysis.undefined_names` 等）必须全过。**类型注解与 docstring 不得改变任何表达式、
字面量、控制流或输出文本。**

验证这类改动的做法（本仓库没有测试套件，靠下面三条证据）：

1. **全量自检 + 产物零改动**：跑完五条自检与 `python -m t2`、`python -m t3 --plan-only`、`python -m t4 --plan-only`，
   要求 `git status --porcelain -- results` 为空——空就说明产物与改动前逐字节一致。
2. **AST 归一化审计**：把 `HEAD` 版与工作区版都解析成 AST，剥离 docstring、函数签名注解、带注解
   赋值的注解、类型别名与 import 名单后比对 `ast.dump`，必须完全相同（注释本就不进 AST）。这一步能
   证明"只有注解/docstring/注释/导入写法变了"。为折行而提取局部变量之类的改动会被它标出来，属正常，
   但必须再用第 3 条单独验证。
3. **真题 A/B**：出现过语句层改动的文件，用 `--practice N --seed 0`（官方模拟器绝不参与）在
   两种版本下各跑一次同一 seed，比对 `--save-dir` 里的 `t3_survey.json`、`t3_observations.csv`
   （T4 同理）与 stdout；产物必须逐字节一致。stdout 里每次调用的墙钟毫秒数本就随机，比对时按
   `（N ms` 归一化即可。
