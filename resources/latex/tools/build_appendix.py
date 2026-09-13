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
% 中文改走 Noto 等宽，和中英混排的字符宽度对齐。字号 SCALE 由 build_appendix.py 按
% 正文中最宽的一行代码反推，正文的等宽字不受影响。
\\newfontfamily\\codefont{{DejaVu Sans Mono}}[Scale={scale:.3f}]
\\setCJKfamilyfont{{zhcode}}{{Noto Sans Mono CJK SC}}[Scale={scale:.3f}]
\\newcommand{{\\codefamily}}{{\\codefont\\CJKfamily{{zhcode}}}}
% 圈数字 ① 到 ⑦ 交给中文字体，DejaVu 等宽里没有这几个字形
\\xeCJKDeclareCharClass{{CJK}}{{"2460 -> "2473}}
\\lstset{{
    basicstyle=\\small\\codefamily,
    language=Python,
    keywordstyle=\\color{{blue!70!black}},
    commentstyle=\\color{{green!45!black}},
    stringstyle=\\color{{red!60!black}}
}}
{end}
"""

# 附录收录清单：每个问题只留与模型直接相关的算法函数，命令行、绘图、报告与控制流不再列出
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
                 "把最坏定位直径 $J(S_2)$ 作为目标函数。源采样、楔形交会的候选顶点、"
                 "全域粗搜与局部细化是这段代码的四步。",
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
        ],
    },
    {
        "title": "附录 C \\quad 问题三：覆盖圆布局与补测定位",
        "intro": "问题三先用 7 个半径 1000 m 的圆盘盖住整个目标圆域，环半径取解析最优值 "
                 "$\\sqrt{3}R/2$；巡视结束仍未清除的频段按 Fisher 信息准则选补测点，"
                 "逐轮把定位区域压到半径 20 m 以内。",
        "blocks": [
            {"file": "t3/covering.py",
             "symbols": ["hex_layout", "optimal_ring_radius", "feasible_ring_interval",
                         "analytic_worst", "dense_sector_rotation", "nearest_distances",
                         "cover_counts", "worst_candidates_general"],
             "caption": "C.1 \\quad 六边形覆盖布局与覆盖校验（t3/covering.py）"},
            {"file": "common/routing.py", "symbols": ["exact_open_order"],
             "caption": "C.2 \\quad 巡视顺序的精确最短开放路径（common/routing.py）"},
            {"file": "t3/probing.py",
             "symbols": ["fisher_sigma", "hypothesis_points", "probe_candidates"],
             "caption": "C.3 \\quad 补测点的 Fisher 信息选点（t3/probing.py）"},
        ],
    },
    {
        "title": "附录 D \\quad 问题四：二十点测量布局与听到率校验",
        "intro": "问题四的测量位置由原点、问题三的 7 个覆盖基点与 12 个外圈点组成，"
                 "外圈点压在定向源迎光区内沿以内，保证任何方位、任何波束朝向的贴边源都有人听到。"
                 "布局确定后按半圆盘命中判据做全口径蒙特卡洛与贴边对抗校验。",
        "blocks": [
            {"file": "t4/sweep.py",
             "symbols": ["measure_layout", "build_sweep_plan", "_hit_report", "_hit_cases",
                         "adversarial_cases", "verify_hearing_stats"],
             "caption": "D.1 \\quad 测量布局、命中判据与听到率校验（t4/sweep.py）"},
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


def pick_scale(code_lines: list[str]) -> float:
    """按正文宽度反推代码字号：0.80 的比例尺下一行放得下 86 个半角字符"""
    widths = sorted(display_width(line) for line in code_lines if line.strip())
    if not widths:
        return 0.80
    idx = min(len(widths) - 1, int(0.98 * len(widths)))
    target = max(widths[idx], 1)
    return min(0.80, max(0.58, 0.80 * 86.0 / target))


def main() -> None:
    """生成附录并写回论文源文件，--dry-run 只打印统计"""
    parser = argparse.ArgumentParser(description="生成论文附录代码")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不改论文")
    args = parser.parse_args()

    blocks = collect()
    all_lines = [line for _, _, code in blocks for line in code.splitlines()]
    code_lines = sum(len(code.splitlines()) for _, _, code in blocks)
    scale = pick_scale(all_lines)
    print(f"代码段 {len(blocks)} 个，共 {code_lines} 行，最长行 {max(map(display_width, all_lines))} 列，"
          f"字号比例 {scale:.3f}")

    body = []
    for section in SECTIONS:
        body.append(f"\\subsection*{{{section['title']}}}")
        body.append("")
        body.append("\t\\noindent " + section["intro"])
        body.append("")
        for sec, caption, code in blocks:
            if sec is not section:
                continue
            body.append(f"\t\\subsubsection*{{{caption}}}")
            body.append("\t\\begin{lstlisting}")
            body.append(code)
            body.append("\t\\end{lstlisting}")
            body.append("")

    head = ["\\section*{附录}", "",
            "\t\\noindent 附录给出四个问题求解程序的主要部分，完整的工程代码见支撑材料。"
            "为控制篇幅，这里只保留与模型直接对应的算法函数：命令行解析、绘图、报告输出、"
            "控制台日志一类的工程代码不再列出。代码中出现的圆域半径、覆盖半径、采样格数等"
            "常数集中定义在各问题的 config 模块里。", ""]
    appendix = "\n".join(head + body).rstrip() + "\n"

    if args.dry_run:
        print(appendix[:400])
        return

    text = TEX.read_text(encoding="utf-8")
    font_block = FONT_TEMPLATE.format(begin=FONT_BEGIN, end=FONT_END, scale=scale)
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
