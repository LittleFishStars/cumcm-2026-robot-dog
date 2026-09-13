# 2026 CUMCM B 题：机器狗搜索与清除干扰源

## 项目结构

支撑材料压缩包只有下面这些，赛题材料、参考文献、开发文档都不在里面。

```
common/                跨题通用层：几何、路线算子、模拟器接口、演练场、绘图、落盘、路径、控制台
t1/ t2/ t3/ t4/        问题一~四，每个包都带 __main__.py，入口是 python -m tN
jammers-py/            本地复刻模拟器，纯标准库，--practice 靠它
results/               正式测评与本地演练的产物
  t3/1/ 2/ 3/          问题三正式测评三次，各一局，meta.mode 是 official
  t4/1/ 2/ 3/          问题四同上。官方不返回真值，源数这类字段为 null
  t2/                  问题二的六个产物
  practice/t3/ t4/     本地演练那批，seed 2026 起 10 局，meta.mode 是 practice
logs/                  正式测评的官方行为日志，六份 .jlog
README.md              本文件
pyproject.toml         Python 版本与依赖声明
AI工具使用详情.pdf     按竞赛 AI 工具使用规定第 4 条附的详情表
```

`results/` 每份里的构成一样：`trajectory/` 是总轨迹图加同名轨迹表，`scan/` 是逐步扫描图，
`*_survey.json` 汇总逐局，`*_observations.csv` 记逐条观测，`api_calls.jsonl` 记接口调用，
另外还有 plan 类的 JSON 和 CSV。

`logs/` 里是官方模拟器为每次正式测评写的现场日志，一份对一次：`formal-p3-1` 到 `formal-p3-3`
对应 `results/t3/1` 到 `t3/3`，问题四同理，中间的序号就是第几次正式测评。文件格式是 8 字节魔数
`JMBFLOG1` + 2 字节版本 + 4 字节封装长度 + 封装头 JSON + 行为记录，封装头里能读到参赛队号、题号、
第几次、case code 与提交时刻，六份的客户端版本都是 1.1.0。每份的内部结构见 `logs/README.md`。

包外还有两处，都不进包：`resources/` 放赛题材料、参考文献和论文源文件，`jammers-py/data/` 是
本地模拟器跑起来自己建的统计库与行为日志。`AI工具使用详情.pdf` 的源文件就是
`resources/latex/ai-declaration.tex`；`logs/` 与 `results/` 那批是从 `resources/Evaluation/`
的两个压缩包里取出来的。

## 运行环境

Python ≥ 3.14，依赖 numpy ≥ 2.5、shapely ≥ 2.1（问题一、二的楔形交计算）和 matplotlib ≥ 3.11
（出图）。纯求解和自检不用 matplotlib，加 `--no-plot` 就只算数与落地文件。Python 不用自己备，
uv 按 pyproject.toml 挑一个 3.14，本机没有就下装。

在包根目录建环境，本队统一走 uv：

```bash
uv init             # 只有源码目录、手上没有 pyproject.toml 时才需要先建工程。
                    # 本包自带 pyproject.toml，跑它会报 Project is already initialized，跳过
uv sync             # 按 pyproject.toml 建 .venv 并装依赖，首次要联网
uv run python -V    # 应显示 3.14.x
```

之后每条命令都加 `uv run` 前缀，例如 `uv run -m t2`。uv 用本目录的 `.venv`，不必手动激活。
中文输出乱码加 `-X utf8`。
