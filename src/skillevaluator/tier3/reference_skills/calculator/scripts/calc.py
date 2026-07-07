#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calculator — safe arithmetic evaluation and unit conversions."""

import ast
import operator
import re
import sys

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic expression using AST — no eval()."""
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            op = OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary: {type(node.op).__name__}")
            return op(_eval(node.operand))
        raise ValueError(f"Unsupported node: {type(node).__name__}")

    return _eval(tree)


CONVERSIONS = {
    ("celsius", "fahrenheit"): lambda c: c * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda f: (f - 32) * 5 / 9,
    ("miles", "km"): lambda m: m * 1.60934,
    ("km", "miles"): lambda k: k / 1.60934,
    ("kg", "lbs"): lambda k: k * 2.20462,
    ("lbs", "kg"): lambda l: l / 2.20462,
    ("meters", "feet"): lambda m: m * 3.28084,
    ("feet", "meters"): lambda f: f / 3.28084,
}


def try_conversion(expr: str) -> str | None:
    m = re.match(
        r"([\d.]+)\s+(\w+)\s+to\s+(\w+)",
        expr.strip(),
        re.IGNORECASE,
    )
    if not m:
        return None
    value = float(m.group(1))
    from_unit = m.group(2).lower()
    to_unit = m.group(3).lower()
    fn = CONVERSIONS.get((from_unit, to_unit))
    if fn is None:
        return None
    result = fn(value)
    return f"{value:g} {from_unit} = {result:.4g} {to_unit}"


def try_percentage(expr: str) -> str | None:
    m = re.match(r"([\d.]+)%\s+of\s+([\d.]+)", expr.strip(), re.IGNORECASE)
    if not m:
        return None
    pct = float(m.group(1))
    base = float(m.group(2))
    result = pct / 100 * base
    return f"Result: {result:g}"


def main():
    if len(sys.argv) < 2:
        print("Usage: calc.py <expression>", file=sys.stderr)
        sys.exit(1)

    expr = " ".join(sys.argv[1:])

    conv = try_conversion(expr)
    if conv:
        print(conv)
        return

    pct = try_percentage(expr)
    if pct:
        print(pct)
        return

    try:
        result = safe_eval(expr)
        if result == int(result):
            print(f"Result: {int(result)}")
        else:
            print(f"Result: {result:g}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
