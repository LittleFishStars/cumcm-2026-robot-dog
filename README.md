# 2026 CUMCM B 题：机器狗搜索与清除干扰源

半径 1800 m 的圆形作业区里有 10~16 个干扰源，问题四还会混进定向源。机器狗跑 5 m/s，测一次向要
5 s（换频道另算 1 s），示向度误差 ±1°，得走到 20 m 以内才算清除。四个问题各一个包，代码全平铺
在包根目录下。

## 包里有什么

支撑材料压缩包只有下面这些，赛题材料、参考文献、开发文档都不在里面：

```
common/            跨题通用层：几何、路线算子、模拟器接口、演练场、绘图、落盘、路径、控制台
t1/ t2/ t3/ t4/    问题一~四，每个包都带 __main__.py
jammers-py/        本地复刻模拟器，纯标准库，--practice 靠它
results/           正式测评与本地演练的产物，见下面"产物"一节
README.md          本文件
pyproject.toml     Python 版本与依赖声明
uv.lock            依赖版本锁，装出来的就是本队验证过的那几个版本
AI工具使用详情.pdf  按竞赛 AI 工具使用规定第 4 条附的详情表，独立文件，不进代码目录
```

包里不带 `.venv/`，环境按下面一节自己建，首次联网两分钟搞定。

`AI工具使用详情.pdf` 的源文件是 `resources/latex/ai-declaration.tex`，同一份文件独立编译出这张
详情表，被论文 `\input` 时就只出参考文献前那一句声明。`resources/` 整体不进包，所以这里单放了
编译好的 PDF。

所以包内能跑的是离线命令和演练；只有"连官方模拟器"那条路走不通，它要官方的
`jammers-simulator.exe` 在本机跑着，而且一共只有 3 次机会，评审时别用。

`jammers-py/data/` 里的演练统计库和行为日志没进包。那个目录跑一次就自己重建，日志里还带着参赛
队号，本来也不该往外给。

## 先让环境能跑

需要 Python ≥ 3.14，依赖 numpy ≥ 2.5、shapely ≥ 2.1（问题一、二的楔形交计算）和
matplotlib ≥ 3.11（出图）。纯求解和自检不用 matplotlib，加 `--no-plot` 就只算数与落地文件。
Python 不用自己备，uv 按 pyproject.toml 挑一个 3.14，本机没有就下装。

在包根目录建环境，本队统一走 uv：

```bash
uv init         # 只有源码目录、手上没有 pyproject.toml 时才需要先建工程。
                # 本包自带 pyproject.toml，跑它会报 Project is already initialized，跳过
uv sync         # 按 pyproject.toml 与 uv.lock 建 .venv 并装依赖，首次要联网
uv run python -V    # 应显示 3.14.x
```

之后每条命令都加 `uv run` 前缀，例如 `uv run -m t2`。uv 用本目录的 `.venv`，不必手动激活。
中文输出乱码加 `-X utf8`。

## 命令

```bash
# 四个问题的入口
uv run -m t1                            # 问题一：4 个检测点的交会定位区域示例
uv run -m t2                            # 问题二：最优第二检测点 + 成果图（离线，约 25 s）
uv run -m t3 --plan-only                # 问题三：只求覆盖圆方案（不连模拟器，约 1 s）
uv run -m t4 --plan-only                # 问题四：只求扫描方案 + 听到率统计（约 3 s）
uv run -m t3 --practice 1 --robot-id <队号>   # 问题三：本地演练 1 局（自动拉起包内 jammers-py/）
uv run -m t4 --practice 20 --seed 0 --robot-id <队号>   # 问题四：演练 20 局，固定 seed 可复现

# 自检（都应正常退出）
uv run -m t2.region                     # 几何：解析构造 vs shapely 精确值（偏差 ~1e-15）
uv run -m t2.theory                     # 文献判据（CRLB/GDOP）vs 精确最坏界
uv run -m t3.covering                   # 覆盖圆方案：旋转/选向/最少圆数
uv run -m t4.sweep                      # 扫描判据：批量实现 vs 单例参考
```

## 几个要点

输出默认只报每一步的关键结论，结论、清单与校验都是对齐的表格，量多的时候按行看就行，不用在一句
话里挑数字。想看过程加 `--verbose`（覆盖校验明细、每局阶段统计、接口回显、逐局图路径都在里面）；
只想拿产物路径加 `--quiet`，路径一行一个。`t1` 本来就只输出一张表，没有这两个开关。

参赛队号一律运行时用 `--robot-id` 传进来，代码里一个队号都不存。跑演练或官方模式不给就直接报错
退出，错误码 2，例如 `uv run -m t3 --practice 1 --robot-id <12 位队号>`。官方模式这个号必须和
模拟器登录的一致；演练模式随便填，它会作为 `--team` 传给本地模拟器。`t1`、`t2`、两个
`--plan-only` 和上面四条自检都不连模拟器，不需要队号。

演练模式（`--practice N`）自动拉起包里的 `jammers-py/`，它只用 Python 标准库，不用另装东西。
多局并行、或者不想复用已在跑的实例，加 `--no-reuse --robot-port 2111 --console-port 8091` 一类
参数；`--jammers-dir` 可以指向别处的副本。这个目录在开发仓库里是独立子仓库，克隆之后先跑一次
`git submodule update --init jammers-py`。

官方模式（`uv run -m t3` / `uv run -m t4` 不带参数）连 `http://127.0.0.1:2026`，只有 3 次机会，
别拿它试跑。官方和演练写同一个 `--save-dir`，一个目录就是最新一次运行；官方模式拿不到真值，
每次请求与响应都落进 `api_calls.jsonl`，那是唯一的证据链。

产物都在 `results/` 下，按来路分了四份：

```
results/t3/ results/t4/   正式测评（官方模式）的产物，各一局。t3 清除 16 个、虚拟耗时 4058.8 s、
                          里程 16419 m；t4 清除 10 个、虚拟耗时 6503.3 s、里程 23331 m。
                          meta.mode 是 official。官方不返回真值，源数这类字段为 null
results/t2/               问题二的六个产物，python -m t2 离线生成，随时可复现
results/practice/t3/ t4/  本地演练那批，seed 2026 起 10 局，meta.mode 是 practice
```

每一份里的构成一样：`trajectory/` 是总轨迹图加同名轨迹表，`scan/` 是逐步扫描图，
`*_survey.json` 汇总逐局，`*_observations.csv` 记逐条观测，`api_calls.jsonl` 记接口调用，
另外还有 plan 类的 JSON 和 CSV。

`results/practice/` 那批是演练跑满 10 局的产物，两条命令生成，同 seed 逐字节可复现，随时能重跑对照：

```bash
uv run -m t3 --practice 10 --seed 2026 --save-dir results/practice/t3
uv run -m t4 --practice 10 --seed 2026 --save-dir results/practice/t4
```

`api_calls.jsonl` 是唯一不可复现的产物，它写的是现实时间戳。官方模式更是一局一次机会，
那批跑完什么样就存什么样。

正式测评每个问题 3 次机会，都跑了一局，包里放的是最后一次。**重跑别用缺省 `--save-dir`**：
官方和演练都往 `results/t3`、`results/t4` 里写，会把正式测评的产物就地覆盖掉，演练请另给
`--save-dir` 目录。

## 依赖方向

`common ← t1/t2/t3/t4`，只能往下。`t1` 是共用的几何核心，只依赖 `common`；`t2`、`t3` 都 import `t1`；
`t4` 除了 `t1` 还 import `t3.probing` 和 `t3.config`，因为它的测量点按问题三的 7 个覆盖圆心摆。
模型怎么推出来、算法怎么选、踩过哪些坑，都写在对应模块的 docstring 里。
