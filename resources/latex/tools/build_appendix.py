"""把四个问题的主要求解代码抽取成论文附录的 LaTeX 片段，并按行宽自动定代码字号"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[3]
TEX = ROOT / "resources" / "latex" / "无线电干扰源的快速自动定位与清除的研究.tex"

# 字体标记块，插在 \usepackage{listings} 之后，重复运行按标记整体替换
FONT_BEGIN = "% ---------- 附录代码块字体：拉丁用 DejaVu 等宽、中文用 Noto 等宽 ----------"
FONT_END = "% ---------- 附录代码块字体结束 ----------"
FONT_TEMPLATE = """{begin}
% 代码块里的希腊字母、≤、√、±、m² 在 Latin Modern Mono 里没有字形，换成 DejaVu 等宽；
% 中文改走 Noto 等宽。两个比例尺由 build_appendix.py 按最宽的一行代码反推：拉丁等宽字符
% 宽 0.602 em，中文全角宽 1 em，所以中文取 1.204 倍，一个汉字正好两个半角位，中英混排的
% 注释才对得齐。正文里的等宽字不受影响。
\\newfontfamily\\codefont{{DejaVu Sans Mono}}[Scale={scale:.3f}]
\\setCJKfamilyfont{{zhcode}}{{Noto Sans Mono CJK SC}}[Scale={cjk_scale:.3f}]
\\newcommand{{\\codefamily}}{{\\codefont\\CJKfamily{{zhcode}}}}
% 圈数字 ① 到 ⑦ 交给中文字体，DejaVu 等宽里没有这几个字形
\\xeCJKDeclareCharClass{{CJK}}{{"2460 -> "2473}}
\\lstset{{
    % 代码段落的中文行距会被 ctex 拉到 1.56 倍，附录三千多行按那个行距要多出二十几页，
    % 这里压到 0.85，一行 9.2 pt，汉字行高 1.27 倍，不挤也不散
    basicstyle=\\linespread{{0.85}}\\selectfont\\small\\codefamily,
    language=Python,
    keywordstyle=\\color{{blue!70!black}},
    commentstyle=\\color{{green!45!black}},
    stringstyle=\\color{{red!60!black}}
}}
{end}
"""

# 附录收录清单：收录从读入测量到给出答案的完整求解逻辑，命令行、绘图、报告、自检脚本
# 与模拟器接口封装（common/sim_client.py）不进附录
SECTIONS = [
    {
        "title": "附录 A \\quad 问题一：交会定位区域的求交与几何量",
        "intro": "问题一把每条示向度展开成张角 $2\\varepsilon$ 的楔形，检测点逐个与目标圆域求交，"
                 "再从交出的凸多边形上量直径、最小覆盖圆与是否被圆域截断。"
                 "下面是一个检测点集合求交与量测的完整实现。",
        "blocks": [
            {"file": "t1/triangulation.py", "symbols": ["TriangulationRegion"],
             "caption": "A.1 \\quad 定位区域类（t1/triangulation.py）"},
        ],
    },
    {
        "title": "附录 B \\quad 问题二：第二检测点的最坏定位直径寻优",
        "intro": "问题二在源不确定集 $\\mathcal{C}$ 与第二次测向误差上取最坏情形，"
                 "把最坏定位直径 $J(S_2)$ 作为目标函数：先采样源不确定集，"
                 "在全域粗网格上算出每个候选点的 $J$，再在粗解附近细化，"
                 "与文献判据（GDOP、CRLB、极小极大半径）的推荐落点对照，"
                 "最后按 $J\\le(1+\\eta)J^*$ 给出候选区域弧带。",
        "blocks": [
            {"file": "t2/region.py",
             "symbols": ["_candidate_points_batch", "_max_pair_distance", "quad_diameters_batch",
                         "analytic_diameter", "exact_diameter", "corner_sources"],
             "caption": "B.1 \\quad 两楔形之交的候选顶点与定位直径（t2/region.py）"},
            {"file": "t2/score.py",
             "symbols": ["source_samples", "delta2_grid", "worst_case_diameters", "suitability",
                         "coarse_fields", "refine_best"],
             "caption": "B.2 \\quad 源不确定集采样、最坏直径与两级搜索（t2/score.py）"},
            {"file": "t2/theory.py",
             "symbols": ["fim_two_station", "gdop", "crlb_major"],
             "caption": "B.3 \\quad 文献判据：Fisher 信息阵、GDOP 与 CRLB（t2/theory.py）"},
            {"file": "t2/score.py",
             "symbols": ["closed_form_table", "theory_analysis", "criteria_points"],
             "caption": "B.4 \\quad 闭式对照表与三条准则的落点互评（t2/score.py）"},
            {"file": "t2/score.py",
             "symbols": ["Band", "lens_ray_interval", "candidate_band", "_lobe", "_polar_to_xy",
                         "SolveResult", "solve", "_rel_angle", "_single_measurement_diameter",
                         "scenario_quad"],
             "caption": "B.5 \\quad 候选区域弧带与完整求解流程（t2/score.py）"},
        ],
    },
    {
        "title": "附录 C \\quad 问题三：覆盖布局、补测定位与巡视清除策略",
        "intro": "问题三分两段：先求 7 个半径 1000 m 的覆盖圆，环半径取解析最优值 "
                 "$\\sqrt{3}R/2$，并给出巡视顺序；再让机器狗按该顺序扫描，"
                 "把每一次测量（含听不到）都变成可能源集合上的硬约束，"
                 "按「就近试清、多点覆盖试清、Fisher 信息补测、末端逼近」四级递进定位并清除。",
        "blocks": [
            {"file": "common/disc_region.py", "symbols": ["DiscConstraintMixin"],
             "caption": "C.1 \\quad 圆盘内外的硬约束叠加（common/disc_region.py）"},
            {"file": "t3/regions.py", "symbols": ["Obs", "Meas", "ProbRegion"],
             "caption": "C.2 \\quad 问题三的可能源集合（t3/regions.py）"},
            {"file": "t3/covering.py",
             "symbols": ["CoverPlan", "hex_layout", "optimal_ring_radius", "feasible_ring_interval",
                         "analytic_worst", "min_circle_count", "_sector_score",
                         "dense_sector_rotation", "region_samples", "nearest_distances",
                         "worst_candidates", "worst_candidates_general", "cover_counts",
                         "nearest_order", "path_length", "CoverSolveResult", "_six_circle_best",
                         "layout_metrics", "tradeoff_table", "solve_covering_circles"],
             "caption": "C.3 \\quad 覆盖布局的解析解、求值与校验（t3/covering.py）"},
            {"file": "common/routing.py",
             "symbols": ["dist_matrix", "path_len", "path_len_vec", "nearest_order_from_D",
                         "nearest_order", "open_path_length", "two_opt_first", "two_opt_greedy",
                         "_held_karp", "_hk_path", "exact_open_order", "exact_open_by_end"],
             "caption": "C.4 \\quad 巡视路径：最近邻、2-opt 与 Held-Karp 精确解（common/routing.py）"},
            {"file": "t3/probing.py",
             "symbols": ["fisher_sigma", "hypothesis_points", "Probe", "probe_candidates"],
             "caption": "C.5 \\quad 补测点的 Fisher 信息选点（t3/probing.py）"},
            {"file": "common/actions.py", "symbols": ["ActionRecorder"],
             "caption": "C.6 \\quad 机器狗的基础动作与状态（common/actions.py）"},
            {"file": "t3/strategy.py", "symbols": ["RobotDog"],
             "caption": "C.7 \\quad 巡视扫描与清除策略（t3/strategy.py）"},
        ],
    },
    {
        "title": "附录 D \\quad 问题四：二十点测量布局与定向源清除策略",
        "intro": "问题四的测量位置由原点、问题三的 7 个覆盖基点与 12 个外圈点组成，"
                 "外圈点压在定向源迎光区内沿以内，保证任何方位、任何波束朝向的贴边源都有人听到；"
                 "定向源的 \\texttt{no\\_signal} 不再作硬约束，只保留 direction 与 near 两类，"
                 "清除沿用问题三的四级递进，并增加清除时刻的顺路补测换取新视角。",
        "blocks": [
            {"file": "t4/sweep.py",
             "symbols": ["measure_layout", "SweepPlan", "build_sweep_plan", "plan_from_points",
                         "_hit_report", "_hit_cases", "_scan_cases", "adversarial_cases",
                         "verify_hearing_stats"],
             "caption": "D.1 \\quad 测量布局、命中判据与听到率校验（t4/sweep.py）"},
            {"file": "t4/regions.py", "symbols": ["Obs", "Meas", "DirProbRegion"],
             "caption": "D.2 \\quad 问题四的可能源集合（t4/regions.py）"},
            {"file": "t4/strategy.py", "symbols": ["RobotDog"],
             "caption": "D.3 \\quad 扫描与清除策略（t4/strategy.py）"},
        ],
    },
]


def display_width(text: str) -> int:
    """字符串的显示宽度，全角字符按两列算"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def extract(file: str, symbols: list[str]) -> str:
    """按符号名从源文件里取出函数或类的原文，顺序按文件中的位置"""
    text = (ROOT / file).read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    found: dict[str, list[ast.AST]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.setdefault(node.name, []).append(node)
    picked: list[ast.AST] = []
    for name in symbols:
        if name not in found:
            raise SystemExit(f"{file} 里找不到 {name}")
        picked.append(found[name][0])
    picked.sort(key=lambda n: n.lineno)
    chunks = []
    for node in picked:
        start = node.lineno - 1
        if node.decorator_list:                      # 装饰器也要跟着函数一起取出
            start = min(d.lineno for d in node.decorator_list) - 1
        block = lines[start:node.end_lineno]
        while block and not block[-1].strip():
            block.pop()
        chunks.append("\n".join(block))
    return "\n\n".join(chunks)


def collect() -> list[tuple[dict, str, str]]:
    """取出全部代码段，返回 (小节, 标题, 代码) 列表"""
    out = []
    for section in SECTIONS:
        for block in section["blocks"]:
            out.append((section, block["caption"], extract(block["file"], block["symbols"])))
    return out


TEXT_WIDTH_PT = 415.0      # 正文宽度 / pt，21 cm 版心减两侧 3.18 cm 边距
CHAR_RATIO = 0.602         # DejaVu Sans Mono 的字符宽除以字号
BASE_PT = 9.0              # \small 在五号正文下的字号 / pt


def pick_scale(code_lines: list[str]) -> float:
    """按正文宽度反推代码字号，以最长的一行为准，一行都不用折；下限 0.62 免得字太小"""
    widths = [display_width(line) for line in code_lines if line.strip()]
    if not widths:
        return 0.80
    return min(0.80, max(0.62, TEXT_WIDTH_PT / (max(widths) * CHAR_RATIO * BASE_PT)))


def main() -> None:
    """生成附录并写回论文源文件，--dry-run 只打印统计"""
    parser = argparse.ArgumentParser(description="生成论文附录代码")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不改论文")
    args = parser.parse_args()

    blocks = collect()
    all_lines = [line for _, _, code in blocks for line in code.splitlines()]
    code_lines = sum(len(code.splitlines()) for _, _, code in blocks)
    scale = pick_scale(all_lines)
    # 一个汉字占两个半角位：中文全角宽 1 em、拉丁等宽宽 0.602 em，比例尺就得差 1.204 倍
    cjk_scale = min(1.0, scale * 2.0 * CHAR_RATIO)
    print(f"代码段 {len(blocks)} 个，共 {code_lines} 行，最长行 {max(map(display_width, all_lines))} 列，"
          f"字号比例 拉丁 {scale:.3f}、中文 {cjk_scale:.3f}")

    body = []
    for section in SECTIONS:
        body.append(f"\\subsection*{{{section['title']}}}")
        body.append("")
        body.append("\t\\noindent " + section["intro"])
        body.append("")
        for sec, caption, code in blocks:
            if sec is not section:
                continue
            body.append(f"\t\\subsubsection*{{{caption.replace('_', r'\_')}}}")
            body.append("\t\\begin{lstlisting}")
            body.append(code)
            body.append("\t\\end{lstlisting}")
            body.append("")

    head = ["\\section*{附录}", "",
            "\t\\noindent 附录收录四个问题从读入测量到给出答案的求解逻辑：问题一的定位区域求交，"
            "问题二的源不确定集采样、最坏定位直径与候选区域，问题三、四的覆盖布局、"
            "可能源集合的硬约束叠加、补测选点与巡视清除策略。命令行解析、绘图、报告输出、"
            "模块自检与模拟器接口封装（common/sim\\_client.py）不在附录内，完整工程代码见支撑材料。"
            "代码中出现的圆域半径、覆盖半径、采样格数等常数集中定义在各问题的 config 模块里。", ""]
    appendix = "\n".join(head + body).rstrip() + "\n"

    if args.dry_run:
        print(appendix[:400])
        return

    text = TEX.read_text(encoding="utf-8")
    font_block = FONT_TEMPLATE.format(begin=FONT_BEGIN, end=FONT_END, scale=scale,
                                      cjk_scale=cjk_scale)
    marker = re.compile(re.escape(FONT_BEGIN) + r".*?" + re.escape(FONT_END), re.S)
    if marker.search(text):
        text = marker.sub(lambda _: font_block.rstrip(), text, count=1)
    else:
        # 插在原 \lstset 之后：listings 的同名键以后设置的为准，插在前面会被 basicstyle 覆盖掉
        anchor = "\\usepackage{tikz}                                % 流程图/示意图\n"
        if anchor not in text:
            raise SystemExit("论文里找不到 tikz 的引入行，字体设置没处可插")
        text = text.replace(anchor, font_block + anchor, 1)

    pattern = re.compile(r"\\section\*\{附录\}.*?(?=\n\\end\{document\})", re.S)
    if not pattern.search(text):
        raise SystemExit("论文里找不到附录一节")
    text = pattern.sub(lambda _: appendix.rstrip(), text, count=1)
    TEX.write_text(text, encoding="utf-8")
    print(f"已写入 {TEX}")


if __name__ == '__main__':
    main()
