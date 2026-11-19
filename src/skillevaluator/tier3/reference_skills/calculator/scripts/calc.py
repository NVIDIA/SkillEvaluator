#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calculator — safe arithmetic evaluation and unit conversions."""

import ast
import math
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

# Keep printed integers below Python's default int-to-str limit (4,300 digits).
MAX_RESULT_DIGITS = 1000


def _log10_int(n: int) -> float:
    """Return log10(n) for a positive int without converting the whole value to float."""
    text = str(n)
    lead = text[:12]
    significand = int(lead) / 10 ** (len(lead) - 1)
    return math.log10(significand) + len(text) - 1


def _int_digit_count(n: int) -> int:
    return len(str(abs(n)))


def _integer_pow_exceeds_digit_budget(base: object, exp: object) -> bool:
    if type(base) is not int or type(exp) is not int or exp <= 0:
        return False
    abs_base = abs(base)
    if abs_base <= 1:
        return False
    # |base| >= 2 gives at least exp * log10(2) > 0.3 * exp digits, so a large
    # exponent is over budget without the float estimate, which would overflow.
    if exp >= 4 * MAX_RESULT_DIGITS:
        return True
    digits_minus_one = exp * _log10_int(abs_base)
    return not math.isfinite(digits_minus_one) or digits_minus_one >= MAX_RESULT_DIGITS


def _check_result(value):
    if isinstance(value, complex):
        raise ValueError("result is not a real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result is too large")
    if type(value) is int and _int_digit_count(value) > MAX_RESULT_DIGITS:
        raise ValueError("result is too large")
    return value


def _apply(op, *args):
    try:
        return _check_result(op(*args))
    except OverflowError:
        raise ValueError("result is too large") from None


def _pow(base, exp):
    if _integer_pow_exceeds_digit_budget(base, exp):
        raise ValueError("result is too large")
    return _apply(operator.pow, base, exp)


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic expression using AST — no eval()."""
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError(f"Unsupported value: {type(node.value).__name__}")
            return _check_result(node.value)
        if isinstance(node, ast.BinOp):
            op = OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow):
                return _pow(left, right)
            return _apply(op, left, right)
        if isinstance(node, ast.UnaryOp):
            op = OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary: {type(node.op).__name__}")
            return _apply(op, _eval(node.operand))
        raise ValueError(f"Unsupported node: {type(node).__name__}")

    return _eval(tree)


# A signed decimal number: "-40", "+2", "3.5", ".5" or "5.".
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

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
    m = re.fullmatch(
        rf"({NUMBER})\s+(\w+)\s+to\s+(\w+)",
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
    m = re.fullmatch(rf"({NUMBER})%\s+of\s+({NUMBER})", expr.strip(), re.IGNORECASE)
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
