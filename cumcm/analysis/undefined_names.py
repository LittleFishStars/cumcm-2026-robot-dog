"""漏定义名字的静态检查：函数里读到、但任何可见作用域都未绑定的名字。

**为什么需要它**（不是假想的用法，是被真事故逼出来的）：撤掉"边清边扫"功能时，我用字符串
替换删代码，连带删掉了 `_nearby_try_clear` 里的 `cx, cy = mec[0], mec[1]`，而后面三处还在用
这两个名字。`import cumcm.t3.strategy` **完全正常** —— 导入不执行那条代码路径；只有跑到"就近
试清未命中、就地复测"这一步才会抛 NameError。本模块就是为这类"静态可查、导入查不到"的错
漏而写的，撤任何功能后都该跑一遍。

用法：`.venv/bin/python -m cumcm.analysis.undefined_names`（缺省扫 `cumcm/` 全包）。

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
from typing import List

BUILTINS = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__class__", "__spec__", "__loader__",
    "__build_class__", "__debug__"}


def collect(body):
    """扫描一个作用域：返回 (绑定名, [(读到的名, 行号)], [嵌套的 def/class 节点])。

    不下钻嵌套 def/lambda/class 的函数体（它们各自成作用域），但把它们收集起来单独递归。
    """
    binds, loads, nested = set(), [], []

    def visit(node):
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


def check(body, visible, where, out):
    binds, loads, nested = collect(body)
    vis = visible | binds
    for nm, ln in loads:
        if nm not in vis:
            out.append((where, nm, ln))
    for nd in nested:
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


def check_file(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), str(path))
    out = []
    check(tree.body, set(BUILTINS), "<module>", out)
    return [p for p in out if p[0] != "<module>"]


def main(argv: List[str]) -> int:
    """扫若干文件或目录（缺省整个 `cumcm/` 包）；有漏定义返回 1，否则返回 0。"""
    args = argv or ["cumcm"]
    targets: List[str] = []
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
    print(f"共 {total} 处漏定义（扫了 {len(targets)} 个文件）" if total
          else f"未发现漏定义 ✓（扫了 {len(targets)} 个文件）")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
