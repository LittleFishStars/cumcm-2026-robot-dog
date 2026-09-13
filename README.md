# 2026 CUMCM B 题：机器狗搜索与清除干扰源

半径 1800 m 的圆形作业区内有 10~16 个干扰源（问题四为定向源 + 全向源混合），机器狗以 5 m/s
行驶、每次测向 5 s（含换频道 1 s）、示向度误差 ±1°，需走到 20 m 内清除。四个问题各一个包，
代码平铺在仓库根。

## 运行环境

**统一使用 [uv](https://docs.astral.sh/uv/)**：解释器、依赖与虚拟环境都由 uv 管理，不要用系统
`python` 直接跑（既省去手动激活 `.venv`，也保证版本与 `uv.lock` 一致）。

- **Python ≥ 3.14**（`pyproject.toml` 声明；开发与验证都在 Arch Linux 上做）
- 依赖：`numpy ≥ 2.5`、`shapely ≥ 2.1`（问题一/二的楔形交计算）、`matplotlib ≥ 3.11`（出图）；
  求解与自检本身不需要 matplotlib（`--no-plot` 可只算数与文件）
- 安装（在仓库根执行一次即可，uv 会自动准备 Python 3.14 与 `.venv`）：

  ```bash
  uv sync                              # 按 pyproject.toml + uv.lock 建/更新 .venv
  uv run python -V                     # 应显示 3.14.x
  ```

- 之后所有命令一律走 uv（见下一节的 `uv run -m <包名>`），无需 `source .venv/bin/activate`
- 中文输出若乱码，用 `uv run python -X utf8 -m <包名> ...` 代替 `uv run -m <包名> ...`

## 使用说明

下面一律用 `uv run -m <包名>`（在仓库根执行；uv 自动用本仓库的 `.venv`，不必先激活）。
每个包都带 `__main__.py`，等价写法是 `uv run python -m <包名>`。

```bash
# 四个问题的入口
uv run -m t1                            # 问题一：4 个检测点的交会定位区域示例
uv run -m t2                            # 问题二：最优第二检测点 + 成果图（离线，约 15 s）
uv run -m t3 --plan-only                # 问题三：只求覆盖圆方案（不连模拟器，约 0.6 s）
uv run -m t3 --practice 3               # 问题三：本地演练 3 局
uv run -m t4 --plan-only                # 问题四：只求扫描方案 + 听到率统计（约 1.7 s）
uv run -m t4 --practice 20 --seed 0     # 问题四：本地演练 20 局（固定 seed 逐字节可复现）

# 自检（都应正常退出）
uv run -m t2.region                     # 几何：解析构造 vs shapely 精确值（偏差 ~1e-15）
uv run -m t2.theory                     # 文献判据（CRLB/GDOP）vs 精确最坏界
uv run -m t3.covering                   # 覆盖圆方案：旋转/选向/最少圆数
uv run -m t4.sweep                      # 扫描判据：批量实现 vs 单例参考
uv run -m analysis.undefined_names      # 静态检查：漏定义 / 缺失参数
```

- **演练模式**（`--practice N`）自动拉起 `resources/jammers-py/` 里的本地复刻模拟器（自带真值，
  可核对覆盖保证）；该目录是独立子仓库（私有远端，克隆后先跑一次
  `git submodule update --init resources/jammers-py`），缺了就只能跑上面的离线命令，
  `--jammers-dir` 可指定别处。
  多局并行或不想复用已有实例时加 `--no-reuse --robot-port 2111 --console-port 8091` 一类参数。
- **官方模式**（`uv run -m t3` / `uv run -m t4` 不带参数）连 `http://127.0.0.1:2026`，只有 3 次
  机会，**不要用它试跑**。官方与演练写同一个 `--save-dir`（一个目录 = 最新一次运行），且官方模式
  拿不到真值，逐次请求/响应只落盘到 `api_calls.jsonl`，那是唯一的证据链。
- **产物**落在 `results/`（按题分子目录，不入库）：`trajectory/` 总轨迹图 + 同名轨迹表、`scan/`
  逐步扫描图、`*_survey.json` 逐局汇总、`*_observations.csv` 逐条观测、`api_calls.jsonl` 接口
  日志、以及 plan 类 JSON/CSV。同 seed 逐字节可复现（唯一例外是 `api_calls.jsonl` 的现实时间戳），
  删掉后重跑命令即可重建，交付时按需重新生成。

## 目录结构

```
common/            跨题通用层：几何、路线算子、模拟器接口、演练场、绘图、落盘、路径、控制台
t1/ t2/ t3/ t4/    问题一~四
analysis/          静态检查工具
resources/         本机材料（不入库）：Problem/ 赛题、References/ 文献、latex/ 论文源文件；
                   jammers-py/ 是子仓库
results/           运行产物（不入库）
```

依赖方向是硬约定：`common ← t1/t2/t3/t4 ← analysis`，只能向下（`t2` 可 import `t1`，`t4` 可
import `t1` 与 `t3.probing`）。模型推导、算法选择与踩过的坑都写在对应模块的 docstring 里；
仓库约定、验证办法与代码风格见 `AGENTS.md`。
