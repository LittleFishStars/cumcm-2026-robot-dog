# 2026 CUMCM B 题：机器狗搜索与清除干扰源

半径 1800 m 的圆形作业区内有 10~16 个干扰源（问题四为定向源 + 全向源混合），机器狗以 5 m/s
行驶、每次测向 5 s（含换频道 1 s）、示向度误差 ±1°，需走到 20 m 内清除。四个问题各一个包，
代码平铺在包根目录。

## 提交包内容

支撑材料压缩包里只有下列内容，赛题材料、参考文献、开发文档等一律不随包提供：

```
common/            跨题通用层：几何、路线算子、模拟器接口、演练场、绘图、落盘、路径、控制台
t1/ t2/ t3/ t4/    问题一~四，每个包都带 __main__.py
results/           最后一次正式运行的产物：t3/、t4/ 的逐局汇总、观测、轨迹与扫描图，t2/ 的成果图
README.md          本文件
pyproject.toml     Python 版本与依赖声明
.venv/             随包提供的 Python 3.14 运行环境（numpy / shapely / matplotlib）
```

因此包内**只能跑离线命令**：`--practice` 演练模式依赖的本地复刻模拟器是开发仓库里的
`resources/jammers-py/` 子仓库，不在包内；官方模式要求官方 `jammers-simulator.exe` 在本机运行，
且只有 3 次机会，**评审时请勿使用**。

## 运行环境

包里已带 `.venv/`，**无需安装任何依赖**即可直接运行：

```bash
.venv/bin/python -m t2                   # Linux / macOS
.venv\Scripts\python.exe -m t2           # Windows
```

本队的开发方式是 [uv](https://docs.astral.sh/uv/)：装了 uv 的机器上按下面这样用（推荐），uv 会
自动使用本目录的 `.venv`，不必手动激活：

```bash
uv sync                                  # 按 pyproject.toml 建/更新 .venv（首次需联网解析依赖）
uv run python -V                         # 应显示 3.14.x
```

没有 uv 时，把下一节命令里的 `uv run -m X` 换成 `.venv/bin/python -m X`
（Windows 为 `.venv\Scripts\python.exe -m X`），两者等价。请勿使用系统 PATH 里的 `python`。

- **Python ≥ 3.14**（`pyproject.toml` 声明；开发与验证都在 Arch Linux 上做）
- 依赖：`numpy ≥ 2.5`、`shapely ≥ 2.1`（问题一/二的楔形交计算）、`matplotlib ≥ 3.11`（出图）；
  求解与自检本身不需要 matplotlib（`--no-plot` 可只算数与文件）
- 中文输出若乱码，用 `uv run python -X utf8 -m <包名> ...`（或 `.venv/bin/python -X utf8 -m <包名> ...`）

## 使用说明

下面一律用 `uv run -m <包名>`（在包根目录执行）。每个包都带 `__main__.py`，等价写法是
`uv run python -m <包名>`。

```bash
# 四个问题的入口
uv run -m t1                            # 问题一：4 个检测点的交会定位区域示例
uv run -m t2                            # 问题二：最优第二检测点 + 成果图（离线，约 15 s）
uv run -m t3 --plan-only                # 问题三：只求覆盖圆方案（不连模拟器，约 0.6 s）
uv run -m t4 --plan-only                # 问题四：只求扫描方案 + 听到率统计（约 1.7 s）
uv run -m t3 --practice 3               # 问题三：本地演练 3 局（需本地复刻模拟器，不在提交包内）
uv run -m t4 --practice 20 --seed 0     # 问题四：本地演练 20 局（同上；固定 seed 逐字节可复现）

# 自检（都应正常退出）
uv run -m t2.region                     # 几何：解析构造 vs shapely 精确值（偏差 ~1e-15）
uv run -m t2.theory                     # 文献判据（CRLB/GDOP）vs 精确最坏界
uv run -m t3.covering                   # 覆盖圆方案：旋转/选向/最少圆数
uv run -m t4.sweep                      # 扫描判据：批量实现 vs 单例参考
```

- **离线命令**（`t1`、`t2`、两个 `--plan-only` 与上面四条自检）在提交包里直接可跑，只依赖
  `.venv` 里的 numpy / shapely / matplotlib。
- **演练模式**（`--practice N`）自动拉起本地复刻模拟器（自带真值，可核对覆盖保证）；该模拟器是
  开发仓库里的独立子仓库 `resources/jammers-py/`，**不在提交包内**，故包里跑不了。
  本队开发时先 `git submodule update --init resources/jammers-py`，`--jammers-dir` 可指定别处；
  多局并行或不想复用已有实例时加 `--no-reuse --robot-port 2111 --console-port 8091` 一类参数。
- **官方模式**（`uv run -m t3` / `uv run -m t4` 不带参数）连 `http://127.0.0.1:2026`，只有 3 次
  机会，**不要用它试跑**。官方与演练写同一个 `--save-dir`（一个目录 = 最新一次运行），且官方模式
  拿不到真值，逐次请求/响应只落盘到 `api_calls.jsonl`，那是唯一的证据链。
- **产物**落在 `results/`：`trajectory/` 总轨迹图 + 同名轨迹表、`scan/` 逐步扫描图、
  `*_survey.json` 逐局汇总、`*_observations.csv` 逐条观测、`api_calls.jsonl` 接口日志、以及
  plan 类 JSON/CSV。同 seed 逐字节可复现（唯一例外是 `api_calls.jsonl` 的现实时间戳）。
  包内这一份就是最后一次正式运行的产物，`api_calls.jsonl` 里的 `meta.mode` 可区分官方 / 演练；
  **重跑会就地覆盖**，评审时如要保留请先备份。

## 依赖方向

依赖方向是硬约定：`common ← t1/t2/t3/t4`，只能向下（`t2` 可 import `t1`，`t4` 可 import `t1` 与
`t3.probing`）。模型推导、算法选择与踩过的坑都写在对应模块的 docstring 里。
