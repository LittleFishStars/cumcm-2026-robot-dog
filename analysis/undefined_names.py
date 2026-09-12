"""漏定义名字的静态检查：函数里读到、但任何可见作用域都未绑定的名字。

**为什么需要它**（不是假想的用法，是被真事故逼出来的）：撤掉"边清边扫"功能时，我用字符串
替换删代码，连带删掉了 `_nearby_try_clear` 里的 `cx, cy = mec[0], mec[1]`，而后面三处还在用
这两个名字。`import t3.strategy` **完全正常** —— 导入不执行那条代码路径；只有跑到"就近
试清未命中、就地复测"这一步才会抛 NameError。本模块就是为这类"静态可查、导入查不到"的错
漏而写的，撤任何功能后都该跑一遍。

用法：`.venv/bin/python -m analysis.undefined_names`（缺省扫 `` 全包）。

判定的四条规则（每一条都是前一版踩过坑才定下来的）：
1. 必须**进类**：`_nearby_try_clear` 是类方法，只遍历模块级函数会整个漏掉它。
2. 作用域要**收紧**：模块级绑定只取模块体的直接语句。若把各函数体内的赋值也算作"模块可见"，
   那么 `cx` 在别的函数里赋过值就会显得处处可见，真漏定义反被掩盖（第一版正是如此）。
3. 必须处理**语句自身**的绑定效果（`import os` 没有子节点），否则函数内延迟 import 会误报。
4. **跳过注解**：本仓库用 `from __future__ import annotations`，注解不求值，且类型名常在
   `if TYPE_CHECKING:` 下导入，故注解里的名字不参与判定。
"""

import ast
import builtins
import sys
from pathlib import Path

BUILTINS = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__class__", "__spec__", "__loader__",
    "__build_class__", "__debug__"}


def collect(body: list[ast.stmt]) -> tuple[set[str], list[tuple[str, int]], list[ast.stmt]]:
    """扫描一个作用域：返回 (绑定名, [(读到的名, 行号)], [嵌套的 def/class 节点])。

    不下钻嵌套 def/lambda/class 的函数体（它们各自成作用域），但把它们收集起来单独递归。

    Args:
        body: 该作用域的语句列表（模块体或函数体）

    Returns:
        tuple[set[str], list[tuple[str, int]], list[ast.stmt]]:
        本作用域绑定的名字、读到的名字与行号、以及待递归的嵌套 def/class 节点
    """
    binds, loads, nested = set(), [], []

    def visit(node: ast.AST) -> None:
        """递归访问单个 AST 节点，把绑定与读取分别记进 binds / loads

        Args:
            node: 待访问的 AST 节点
        """
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            binds.add(node.name)
            nested.append(node)
            return                                   # 参数与函数体属于它自己的作用域
        if isinstance(node, ast.ClassDef):
            binds.add(node.name)
            nested.append(node)
            return
        if isinstance(node, ast.Lambda):
            return                                   # lambda 体自成作用域
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                binds.add(node.target.id)
            if node.value is not None:
                visit(node.value)
            return                                   # 注解不求值，跳过
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                binds.add((al.asname or al.name).split(".")[0])
            return
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                binds.add(node.id)
            else:
                loads.append((node.id, node.lineno))
            return
        if isinstance(node, ast.ExceptHandler) and node.name:
            binds.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            binds.update(node.names)
        for ch in ast.iter_child_nodes(node):
            visit(ch)

    for b in body:
        visit(b)
    return binds, loads, nested


def check(body: list[ast.stmt], visible: set[str], where: str,
          out: list[tuple[str, str, int]]) -> None:
    """递归检查一个作用域：读到但不可见的名字记进 out，再逐层下钻嵌套 def/class

    Args:
        body: 该作用域的语句列表（模块体或函数体）
        visible: 外层可见的名字（调用方负责并上本层的绑定）
        where: 出错时打印的作用域名（函数名或 "<module>"）
        out: 结果累积列表，元素为 (作用域名, 名字, 行号)
    """
    binds, loads, nested = collect(body)
    vis = visible | binds
    for nm, ln in loads:
        if nm not in vis:
            out.append((where, nm, ln))
    for nd in nested:
        # 函数要把自己的形参并进可见集，类体则沿用外层可见集
        if isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = nd.args
            params = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
            if a.vararg:
                params.add(a.vararg.arg)
            if a.kwarg:
                params.add(a.kwarg.arg)
            check(nd.body, vis | params, nd.name, out)
        else:
            check(nd.body, vis, nd.name, out)


def check_file(path: str) -> list[tuple[str, str, int]]:
    """检查单个 .py 文件，返回模块体以下各作用域的漏定义列表

    Args:
        path: 待检查的 Python 文件路径

    Returns:
        list[tuple[str, str, int]]: (作用域名, 未绑定的名字, 行号)；模块体本身的问题不计
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), str(path))
    out = []
    check(tree.body, set(BUILTINS), "<module>", out)
    return [p for p in out if p[0] != "<module>"]


def check_argparse_attrs(path: str) -> list[tuple[str, str, int]]:
    """同一类错漏的 argparse 版本：读到的 `args.X` 却没有任何 `add_argument` 产生。

    为什么要这一项：撤/换命令行参数时按"两处锚点之间的区间"删代码，很容易连带删掉夹在中间的
    其它参数（本项目真实发生过两次：撤 --piggyback 与换 --inline-* 时各误删一次
    `--survey-only`、`--traj-dir`）。这类错 import 检查同样抓不到 —— 只有真跑到那条分支才炸，
    而 `--survey-only` 恰好在 `run_practice` 开头就用到，属于"一跑就炸"的幸运情况。

    判定：先收齐本文件 `add_argument` 产生的 dest（长选项去 `--`、连字符换下划线；有 `dest=`
    则用它），再找出所有"命名空间变量"（`parse_args()` 的赋值目标，或注解为
    `argparse.Namespace` 的参数）的属性读取，凡 attr 不在 dest 集合里即报告。
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), path)
    dests, namespaces = set(), set()
    for n in ast.walk(tree):
        # add_argument("--foo", "-f", dest="bar")
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"):
            dest = None
            for kw in n.keywords:
                if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                    dest = str(kw.value.value)
            longs = [a.value for a in n.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if dest is None:
                cand = [x for x in longs if x.startswith("--")]
                if cand:
                    dest = cand[0][2:].replace("-", "_")
                elif longs:                      # 位置参数
                    dest = longs[0]
            if dest:
                dests.add(dest)
        # X = p.parse_args(...)
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) \
                and isinstance(n.value.func, ast.Attribute) \
                and n.value.func.attr == "parse_args":
            for t in n.targets:
                if isinstance(t, ast.Name):
                    namespaces.add(t.id)
        # def f(args: argparse.Namespace, ...)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in n.args.posonlyargs + n.args.args + n.args.kwonlyargs:
                ann = a.annotation
                if isinstance(ann, ast.Attribute) and ann.attr == "Namespace":
                    namespaces.add(a.arg)
    if not dests:
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
                and n.value.id in namespaces and n.attr not in dests:
            out.append((n.attr, n.value.id, n.lineno))
    return out


def _default_targets() -> list[str]:
    """缺省扫描范围：仓库根的 .py 文件与根目录下所有带 `__init__.py` 的包

    Returns:
        list[str]: 可直接交给 check_file 的路径列表
    """
    root = Path(__file__).resolve().parent.parent
    out = [str(f) for f in sorted(root.glob("*.py"))]
    out += [str(d) for d in sorted(root.iterdir()) if (d / "__init__.py").exists()]
    return out


def main(argv: list[str]) -> int:
    """扫若干文件或目录（缺省根目录下的全部包与脚本）；有漏定义返回 1，否则返回 0。"""
    args = argv or _default_targets()
    targets: list[str] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            targets.extend(str(f) for f in sorted(p.rglob("*.py"))
                           if "__pycache__" not in f.parts)
        else:
            targets.append(str(p))
    total = 0
    for t in targets:
        for where, nm, ln in check_file(t):
            print(f"  {t}:{ln}  {where}() 读到未绑定的 '{nm}'")
            total += 1
        for attr, ns, ln in check_argparse_attrs(t):
            print(f"  {t}:{ln}  读到的 '{ns}.{attr}' 没有任何 add_argument 产生")
            total += 1
    print(f"共 {total} 处漏定义（扫了 {len(targets)} 个文件）" if total
          else f"未发现漏定义或缺失参数 ✓（扫了 {len(targets)} 个文件）")
    return 1 if total else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
