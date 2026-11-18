#!/usr/bin/env python3
"""Write summary.md beside a CSV file."""

import csv
import sys
from pathlib import Path


def _numbers(values: list[str]) -> list[float]:
    numbers = []
    for value in values:
        try:
            numbers.append(float(value))
        except ValueError:
            return []
    return numbers


def main(path: str) -> None:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    columns = list(rows[0]) if rows else []
    lines = ["| column | values | min | max |", "| --- | --- | --- | --- |"]
    for column in columns:
        values = [row[column] for row in rows if row[column]]
        numbers = _numbers(values)
        low, high = (f"{min(numbers):g}", f"{max(numbers):g}") if numbers else ("", "")
        lines.append(f"| {column} | {len(values)} | {low} | {high} |")
    (source.parent / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
