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
results/           最后一次正式运行的产物
README.md          本文件
pyproject.toml     Python 版本与依赖声明
.venv/             本队在 Linux（Arch + Python 3.14.7）下建的虚拟环境
```

所以包内能跑的是离线命令和演练；只有"连官方模拟器"那条路走不通，它要官方的
`jammers-simulator.exe` 在本机跑着，而且一共只有 3 次机会，评审时别用。

`jammers-py/data/` 里的演练统计库和行为日志没进包。那个目录跑一次就自己重建，日志里还带着参赛
队号，本来也不该往外给。

## 先让环境能跑

需要 Python ≥ 3.14，依赖 numpy ≥ 2.5、shapely ≥ 2.1（问题一、二的楔形交计算）和
matplotlib ≥ 3.11（出图）。纯求解和自检不用 matplotlib，加 `--no-plot` 就只算数与落地文件。

装依赖两条路，挑一条（在包根目录执行）：

```bash
python -m pip install "numpy>=2.5" "shapely>=2.1" "matplotlib>=3.11"   # 用本机 Python
uv sync                     # 用 uv：本队开发方式，自动建/更新 .venv，首次要联网
uv run python -V            # 用 uv 的话应显示 3.14.x
```

包里那份 `.venv/` 是 Linux 上建的。`.venv/bin/python` 其实只是指向系统 `python3.14` 的软链接，
site-packages 里也全是 linux-x86_64 的扩展模块（numpy 2.5.3、shapely 2.1.2、matplotlib 3.11.2），
所以同类 Linux 上直接 `.venv/bin/python -m t2` 就能跑。换别的平台就按上面自己装：Windows 上连
`.venv\Scripts\` 都不存在，别照搬这一行。

本队开发时统一走 uv：`uv run -m <包名>`。uv 会用本目录的 .venv，不必手动激活；评审机上没装 uv，
把前缀换成 `python -m <包名>`（自己装好了依赖）或 `.venv/bin/python -m <包名>`（Linux 上直接用
包内环境），三者等价。中文输出乱码就加 `-X utf8`。

## 命令

```bash
# 四个问题的入口
uv run -m t1                            # 问题一：4 个检测点的交会定位区域示例
uv run -m t2                            # 问题二：最优第二检测点 + 成果图（离线，约 15 s）
uv run -m t3 --plan-only                # 问题三：只求覆盖圆方案（不连模拟器，约 0.6 s）
uv run -m t4 --plan-only                # 问题四：只求扫描方案 + 听到率统计（约 1.7 s）
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

产物都在 `results/` 下：`trajectory/` 是总轨迹图加同名轨迹表，`scan/` 是逐步扫描图，
`*_survey.json` 汇总逐局，`*_observations.csv` 记逐条观测，`api_calls.jsonl` 记接口调用，
另外还有 plan 类的 JSON 和 CSV。同一个 seed 重跑，产物逐字节一致，只有 `api_calls.jsonl` 因为
写的是现实时间戳而不同。包内这份就是最后一次正式运行的产物，`meta.mode` 能看出是官方还是演练；
重跑会就地覆盖，要留底先备份。

## 依赖方向

`common ← t1/t2/t3/t4`，只能往下（`t2` 可以 import `t1`，`t4` 可以 import `t1` 和 `t3.probing`）。
模型怎么推出来、算法怎么选、踩过哪些坑，都写在对应模块的 docstring 里。
