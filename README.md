# 2026 CUMCM B 题：机器狗搜索与清除干扰源

半径 1800 m 的圆形作业区内有 10~16 个干扰源（问题四为定向源 + 全向源混合），机器狗以 5 m/s
行驶、每次测向 5 s（含换频道 1 s）、示向度误差 ±1°，需走到 20 m 内清除。四个问题各一个包，
代码平铺在仓库根。

## 运行环境

- **Python ≥ 3.14**（`pyproject.toml` 声明；开发与验证都在 Arch Linux 上做）
- 依赖：`numpy ≥ 2.5`、`shapely ≥ 2.1`（问题一/二的楔形交计算）、`matplotlib ≥ 3.11`（出图）；
  求解与自检本身不需要 matplotlib（`--no-plot` 可只算数与文件）
- 安装（在仓库根执行，二选一）：

  ```bash
  uv sync                              # 按 pyproject.toml 建/更新 .venv
  .venv/bin/python -V                  # 应显示 3.14.x
  ```

  ```bash
  uv pip install --python .venv/bin/python numpy shapely matplotlib
  ```

- 中文输出建议加 `-X utf8`：`.venv/bin/python -X utf8 -m t3 --plan-only`

## 使用说明

下面用 `python` 代指上面装好的解释器（`.venv/bin/python`，或 `uv run python`）。

```bash
# 四个问题的入口（每个包都带 __main__.py）
python -m t1                            # 问题一：4 个检测点的交会定位区域示例
python -m t2                            # 问题二：最优第二检测点 + 成果图（离线，约 15 s）
python -m t3 --plan-only                # 问题三：只求覆盖圆方案（不连模拟器，约 0.6 s）
python -m t3 --practice 3               # 问题三：本地演练 3 局
python -m t4 --plan-only                # 问题四：只求扫描方案 + 听到率统计（约 1.7 s）
python -m t4 --practice 20 --seed 0     # 问题四：本地演练 20 局（固定 seed 逐字节可复现）

# 自检（都应正常退出）
python -m t2.region                     # 几何：解析构造 vs shapely 精确值（偏差 ~1e-15）
python -m t2.theory                     # 文献判据（CRLB/GDOP）vs 精确最坏界
python -m t3.covering                   # 覆盖圆方案：旋转/选向/最少圆数
python -m t4.sweep                      # 扫描判据：批量实现 vs 单例参考
python -m analysis.undefined_names      # 静态检查：漏定义 / 缺失参数
```

- **演练模式**（`--practice N`）自动拉起 `resources/jammers-py/` 里的本地复刻模拟器（自带真值，
  可核对覆盖保证）；该目录不入库，缺了就只能跑上面的离线命令，`--jammers-dir` 可指定别处。
  多局并行或不想复用已有实例时加 `--no-reuse --robot-port 2111 --console-port 8091` 一类参数。
- **官方模式**（`python -m t3` / `python -m t4` 不带参数）连 `http://127.0.0.1:2026`，只有 3 次
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
resources/         本机材料（不入库）：jammers-py/ 本地演练场、Problem/ 赛题、References/ 文献
results/           运行产物（不入库）
```

依赖方向是硬约定：`common ← t1/t2/t3/t4 ← analysis`，只能向下（`t2` 可 import `t1`，`t4` 可
import `t1` 与 `t3.probing`）。模型推导、算法选择与踩过的坑都写在对应模块的 docstring 里；
仓库约定、验证办法与代码风格见 `AGENTS.md`。
