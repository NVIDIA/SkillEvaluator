# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure drift between the two copies of the Tier 3 grading checks.

``eval_core/checks.py`` (host) and ``harbor/templates/eval.py`` (the verifier
that runs alone inside the task container) carry the same security and grading
functions. This script lists the top-level functions and module constants in
each file, the names both define, and, for those, the ones whose normalized
ASTs differ (docstrings removed). It is a measuring tool, not a test.

    uv run python tests/tier3/_tools/measure_checks_template_drift.py [--diff NAME ...] [--json]
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import json
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src" / "skillevaluator" / "tier3"
HOST = _SRC / "eval_core" / "checks.py"
CONTAINER = _SRC / "harbor" / "templates" / "eval.py"
_SHARED_BLOCKS = ("Canary exfiltration", "Runtime security")


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        first = node.body[0] if node.body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
        return node

    def visit_FunctionDef(self, node):
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node):
        return self._strip(node)

    def visit_ClassDef(self, node):
        return self._strip(node)


class _StripAnnotations(ast.NodeTransformer):
    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.returns = None
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if node.value is None:
            return None  # a bare ``x: T`` declares nothing at run time
        return ast.Assign(targets=[node.target], value=node.value, lineno=node.lineno)


def _shared_block_lines(text: str) -> list[tuple[int, int]]:
    """Line spans of the byte-guarded shared blocks; a block whose markers are gone is skipped."""
    lines = text.splitlines()
    spans = []
    for name in _SHARED_BLOCKS:
        begin = next(
            (i for i, line in enumerate(lines, 1) if line.startswith(f"# ── {name} (begin shared block)")), None
        )
        end = next((i for i, line in enumerate(lines, 1) if line.startswith(f"# ── {name} (end shared block)")), None)
        if begin is not None and end is not None:
            spans.append((begin, end))
    return spans


def _top_level(path: Path):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    spans = _shared_block_lines(text)
    functions, constants = {}, {}
    for node in tree.body:
        in_block = any(b <= node.lineno <= e for b, e in spans)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            functions[node.name] = (node, in_block)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = (node, in_block)
    return functions, constants


def _normalize(node, *, annotations: bool = True) -> ast.AST:
    node = _StripDocstrings().visit(copy.deepcopy(node))
    if not annotations:
        node = _StripAnnotations().visit(node)
    return ast.fix_missing_locations(node)


def _dump(node, *, annotations: bool = True) -> str:
    return ast.dump(_normalize(node, annotations=annotations), include_attributes=False)


def _source(node) -> list[str]:
    return ast.unparse(_normalize(node)).splitlines()


def measure() -> dict:
    host_fn, host_const = _top_level(HOST)
    cont_fn, cont_const = _top_level(CONTAINER)
    shared_fn = sorted(set(host_fn) & set(cont_fn))
    shared_const = sorted(set(host_const) & set(cont_const))

    def differing(host, cont, names):
        rows = []
        for name in names:
            h, h_block = host[name]
            c, c_block = cont[name]
            if _dump(h) == _dump(c):
                continue
            diff = list(difflib.unified_diff(_source(h), _source(c), lineterm="", n=0))
            rows.append(
                {
                    "name": name,
                    "annotation_only": _dump(h, annotations=False) == _dump(c, annotations=False),
                    "in_shared_block": h_block and c_block,
                    "changed_lines": sum(1 for line in diff if line[:1] in "+-" and line[:3] not in ("+++", "---")),
                }
            )
        return rows

    return {
        "host_functions": len(host_fn),
        "container_functions": len(cont_fn),
        "shared_functions": len(shared_fn),
        "shared_functions_in_guarded_blocks": sum(1 for n in shared_fn if host_fn[n][1] and cont_fn[n][1]),
        "host_only_functions": sorted(set(host_fn) - set(cont_fn)),
        "differing_functions": differing(host_fn, cont_fn, shared_fn),
        "shared_constants": len(shared_const),
        "differing_constants": differing(host_const, cont_const, shared_const),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diff", nargs="*", default=[], help="print the normalized source diff for these names")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = measure()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"host top-level functions/classes:      {report['host_functions']}")
        print(f"container top-level functions/classes: {report['container_functions']}")
        print(f"shared names:                          {report['shared_functions']}")
        print(f"  of which inside guarded shared blocks: {report['shared_functions_in_guarded_blocks']}")
        print(f"host-only names:                       {len(report['host_only_functions'])}")
        print(f"shared names whose AST differs:        {len(report['differing_functions'])}")
        for row in report["differing_functions"]:
            tag = "annotations only" if row["annotation_only"] else f"{row['changed_lines']} changed lines"
            print(f"  - {row['name']}: {tag}")
        print(f"shared module constants:               {report['shared_constants']}")
        print(f"shared constants whose AST differs:    {len(report['differing_constants'])}")
        for row in report["differing_constants"]:
            print(f"  - {row['name']}: {row['changed_lines']} changed lines")
    host_fn, host_const = _top_level(HOST)
    cont_fn, cont_const = _top_level(CONTAINER)
    for name in args.diff:
        host = host_fn.get(name) or host_const.get(name)
        cont = cont_fn.get(name) or cont_const.get(name)
        print(f"\n===== {name} (--- host checks.py, +++ container eval.py)")
        if host is None or cont is None:
            print(f"not defined at top level in {'checks.py' if host is None else 'eval.py'}")
            continue
        print("\n".join(difflib.unified_diff(_source(host[0]), _source(cont[0]), "checks.py", "eval.py", lineterm="")))


if __name__ == "__main__":
    main()
