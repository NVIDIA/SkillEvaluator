---
name: calculator
description: Evaluate mathematical expressions and unit conversions. Handles arithmetic, percentages, exponents, and common unit conversions (temperature, distance, weight). No external dependencies.
compatibility: Python 3.10+, no external dependencies
metadata:
  author: Skill Evaluator Maintainers <maintainers@example.com>
---

# Calculator

Evaluate math expressions and perform unit conversions from the command line.

## Purpose

Provide a safe, sandboxed calculator for arithmetic expressions and common
unit conversions without relying on `eval()` or external services.

## Agent Instructions

1. Read this SKILL.md to understand capabilities.
2. Run `scripts/calc.py` with the user's expression or conversion.
3. Return the computed result to the user.

## Examples

### Usage

```bash
# Arithmetic expressions
python scripts/calc.py "2 + 3 * 4"
python scripts/calc.py "(100 - 25) / 3"
python scripts/calc.py "2 ** 10"

# Percentage calculations
python scripts/calc.py "15% of 200"

# Unit conversions
python scripts/calc.py "100 celsius to fahrenheit"
python scripts/calc.py "5 miles to km"
python scripts/calc.py "10 kg to lbs"
```

### Output

A single line with the result, e.g.:
```
Result: 14
```

For conversions:
```
100 celsius = 212.0 fahrenheit
```

## Limitations

- No symbolic algebra (no variables, no equations).
- Only supports basic arithmetic operators: `+`, `-`, `*`, `/`, `**`, `%`, parentheses.
- Unit conversions are limited to: celsius/fahrenheit, miles/km, kg/lbs, meters/feet.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Invalid expression` | Check for unsupported operators or variables |
| `Unknown conversion` | Use supported units: celsius, fahrenheit, miles, km, kg, lbs, meters, feet |
