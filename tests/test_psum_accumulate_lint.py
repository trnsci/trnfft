"""Static check: no matmul accumulates onto PSUM that a non-matmul instruction zeroed.

The `nki.isa.nc_matmul` documentation: "On NeuronCore-v3, matmul accumulating
into a PSUM location that was value-initialized by a non-matmul instruction
(e.g., memset, tensor_copy) is not supported and is undefined behavior on
hardware." The first matmul into a PSUM location must use accumulate=False.
Trn2 is NeuronCore-v3. The kernels used to open accumulation groups with
`nl.zeros(..., buffer=nl.psum)` (or a PSUM memset) followed by
`nc_matmul(..., accumulate=True)`; see trnsci/trnsci#185.

This test parses every module (no `nki` needed) and fails on a
`nc_matmul(dst, ..., accumulate=True)` whose `dst` is, or is a view of, a
tile that an `nl.zeros` / `nl.full` / `nl.ones` PSUM allocation or a
`memset` of PSUM initialised in the same function. A tile reaches the check
through names: assignments, views (subscripts, `.reshape`, ...), list
comprehensions, and helper functions that return such a tile. Names follow
Python scoping (a nested helper sees its enclosing function's names), and
the check is flow-insensitive within a function: a name that holds a zeroed
PSUM tile anywhere in it must not be the target of a literal
`accumulate=True` anywhere in it. `accumulate=False`, or an expression such
as `accumulate=k > 0` whose first write overwrites, passes.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

PACKAGE = "trnfft"
ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(
    p.relative_to(ROOT).as_posix()
    for p in [*(ROOT / PACKAGE).rglob("*.py"), *(ROOT / "scripts").glob("*.py")]
)

_FILL = {"zeros", "ones", "full", "zeros_like", "ones_like", "full_like"}
_ALLOC = _FILL | {"ndarray", "empty_like"}
_VIEWS = {"reshape", "permute", "broadcast", "broadcast_to", "view", "ap", "bitcast"}


def _attr(node) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _is_psum_alloc(node, fill_only: bool) -> bool:
    if not isinstance(node, ast.Call) or _attr(node.func) not in (_FILL if fill_only else _ALLOC):
        return False
    buf = _kw(node, "buffer")
    return _attr(buf) == "psum"


def _bases(expr) -> set[str]:
    """Names whose tile `expr` is (or is a view of)."""
    if isinstance(expr, ast.Name):
        return {expr.id}
    if isinstance(expr, (ast.Subscript, ast.Starred)):
        return _bases(expr.value)
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr in _VIEWS
    ):
        return _bases(expr.func.value)
    if isinstance(expr, ast.IfExp):
        return _bases(expr.body) | _bases(expr.orelse)
    if isinstance(expr, (ast.List, ast.Tuple)):
        return set().union(*(_bases(e) for e in expr.elts)) if expr.elts else set()
    if isinstance(expr, (ast.ListComp, ast.GeneratorExp)):
        return _bases(expr.elt)
    return set()


def _target_names(target) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Subscript, ast.Attribute, ast.Starred)):
        return _target_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(e) for e in target.elts)) if target.elts else set()
    return set()


def _contains(expr, pred) -> bool:
    return any(pred(n) for n in ast.walk(expr))


_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _own_nodes(fn):
    """Nodes of `fn`'s body outside nested function definitions."""
    stack = [c for c in fn.body if not isinstance(c, _FUNCS)]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(c for c in ast.iter_child_nodes(node) if not isinstance(c, _FUNCS))


def _bindings(fn):
    """(target names, value) for every binding in `fn`'s own body."""
    for node in _own_nodes(fn):
        if isinstance(node, ast.Assign):
            yield set().union(*(_target_names(t) for t in node.targets)), node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            if node.value is not None:
                yield _target_names(node.target), node.value
        elif isinstance(node, (ast.For, ast.comprehension)):
            yield _target_names(node.target), node.iter


def _calls(fn, name):
    for node in _own_nodes(fn):
        if isinstance(node, ast.Call) and _attr(node.func) == name:
            yield node


def _dst(call: ast.Call):
    dst = _kw(call, "dst")
    if dst is None and call.args:
        dst = call.args[0]
    return dst


def find_violations(tree: ast.AST) -> list[int]:
    """Line numbers of literal accumulate=True matmuls onto zero-filled PSUM.

    Names follow Python scoping: a name is local to the innermost function
    that binds it, and a nested helper reads its enclosing function's names.
    Helper functions returning a (zeroed) PSUM tile are matched by name.
    """
    parent = {}

    def visit(node, enclosing):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _FUNCS):
                parent[child] = enclosing
                visit(child, child)
            else:
                visit(child, enclosing)

    visit(tree, None)
    funcs = list(parent)
    local = {}
    for fn in funcs:
        a = fn.args
        names = {x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs]}
        names |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
        for node in _own_nodes(fn):  # names the function binds (not x[i] = ...)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
        names |= {c.name for c in parent if parent[c] is fn}  # nested defs
        local[fn] = names

    def key(fn, name):
        f = fn
        while f is not None:
            if name in local[f]:
                return (id(f), name)
            f = parent[f]
        return (None, name)

    psum: set = set()  # (scope, name) keys holding PSUM tiles
    zero: set = set()  # ... holding PSUM tiles a non-matmul instruction zeroed
    psum_fns: set[str] = set()
    zero_fns: set[str] = set()

    def keys(fn, expr):
        return {key(fn, n) for n in _bases(expr)}

    def source(fn, expr, tiles, fns, fill_only):
        return _contains(
            expr,
            lambda n: (
                _is_psum_alloc(n, fill_only) or (isinstance(n, ast.Call) and _attr(n.func) in fns)
            ),
        ) or bool(keys(fn, expr) & tiles)

    changed = True
    while changed:
        changed = False
        for fn in funcs:
            for names, value in _bindings(fn):
                bound = {key(fn, n) for n in names}
                for tiles, fns, fill in ((psum, psum_fns, False), (zero, zero_fns, True)):
                    if bound - tiles and source(fn, value, tiles, fns, fill):
                        tiles |= bound
                        changed = True
            for call in _calls(fn, "memset"):
                hit = keys(fn, _dst(call)) & psum
                if hit - zero:
                    zero |= hit
                    changed = True
            for ret in _own_nodes(fn):
                if not isinstance(ret, ast.Return) or ret.value is None:
                    continue
                for tiles, fns, fill in ((psum, psum_fns, False), (zero, zero_fns, True)):
                    if fn.name not in fns and source(fn, ret.value, tiles, fns, fill):
                        fns.add(fn.name)
                        changed = True
    lines = []
    for fn in funcs:
        for call in _calls(fn, "nc_matmul"):
            acc = _kw(call, "accumulate")
            if isinstance(acc, ast.Constant) and acc.value is True and keys(fn, _dst(call)) & zero:
                lines.append(call.lineno)
    return sorted(set(lines))


@pytest.mark.parametrize("path", SOURCES)
def test_no_matmul_accumulates_onto_zero_filled_psum(path):
    tree = ast.parse((ROOT / path).read_text(), filename=path)
    bad = find_violations(tree)
    assert not bad, (
        f"{path}: nc_matmul(accumulate=True) onto PSUM zeroed by nl.zeros/memset at lines "
        f"{bad}; open the accumulation group with accumulate=False (trnsci/trnsci#185)"
    )


_BAD = {
    "zeros_then_accumulate": """
        def k(a, b):
            psum = nl.zeros((128, 512), dtype=nl.float32, buffer=nl.psum)
            for t in nl.affine_range(4):
                nisa.nc_matmul(dst=psum, stationary=a, moving=b, accumulate=True)
        """,
    "zeros_view": """
        def k(a, b):
            y = nl.zeros((128, 4, 2, 64), dtype=nl.float32, buffer=nl.psum)
            for q in range(4):
                yq = y[:, q, :, :].reshape((128, 128))
                nisa.nc_matmul(dst=yq, stationary=a, moving=b, accumulate=True)
        """,
    "zeros_list": """
        def k(a, b, rows):
            ps = [nl.zeros((128, 512), dtype=nl.float32, buffer=nl.psum) for _ in rows]
            nisa.nc_matmul(ps[0], a, b, accumulate=True)
        """,
    "module_helper": """
        def acc(shape):
            return nl.zeros(shape, dtype=nl.float32, buffer=nl.psum)

        def k(a, b):
            ps = acc((128, 512))
            nisa.nc_matmul(ps, a, b, accumulate=True)
        """,
    "memset_helper": """
        def k(a, b):
            pss = {}
            pss["mv"] = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)

            def psum_zeros(width):
                t = pss["mv"]
                v = t[:, 0:width]
                nisa.memset(v, 0.0)
                return v

            ps = psum_zeros(64)
            nisa.nc_matmul(ps[:, 0:1], a, b, accumulate=True)
        """,
}

_GOOD = {
    "first_write_opens": """
        def k(a, b):
            psum = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            for t in range(4):
                nisa.nc_matmul(dst=psum, stationary=a, moving=b, accumulate=t > 0)
        """,
    "zeros_only_when_no_matmul": """
        def acc(shape, n):
            if n == 0:
                return nl.zeros(shape, dtype=nl.float32, buffer=nl.psum)
            return nl.ndarray(shape, dtype=nl.float32, buffer=nl.psum)

        def k(a, b, n):
            psum = acc((128, 512), n)
            for t in range(n):
                nisa.nc_matmul(psum, a, b, accumulate=t > 0)
        """,
    "matmul_opener": """
        def k(a, b, z):
            v = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(v, z, z, accumulate=False)
            nisa.nc_matmul(v, a, b, accumulate=True)
        """,
    "sbuf_memset": """
        def k(a, b):
            sb = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(sb, 0.0)
            ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(ps, a, b, accumulate=False)
            nisa.nc_matmul(ps, a, b, accumulate=True)
        """,
}


@pytest.mark.parametrize("name", sorted(_BAD))
def test_lint_flags_the_pattern(name):
    assert find_violations(ast.parse(textwrap.dedent(_BAD[name])))


@pytest.mark.parametrize("name", sorted(_GOOD))
def test_lint_passes_matmul_opened_groups(name):
    assert not find_violations(ast.parse(textwrap.dedent(_GOOD[name])))
