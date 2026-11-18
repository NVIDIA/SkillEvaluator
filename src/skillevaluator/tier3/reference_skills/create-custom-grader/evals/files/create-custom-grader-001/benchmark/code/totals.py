#!/usr/bin/env python3
"""Print the grand total of the amount column in an orders CSV."""

import csv
import sys
from pathlib import Path


def main(path: str) -> None:
    total = 0.0
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total += float(row["amount"])
    print(f"TOTAL {total:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
