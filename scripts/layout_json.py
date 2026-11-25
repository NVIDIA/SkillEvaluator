# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Store captured JSON test data one record per line.

A list or dict that nests at most two levels and fits in 200 columns, indentation and trailing comma included,
goes on one line. Everything else stays as ``json.dumps(indent=2)`` writes it. Only whitespace moves: keys keep
their order and every string and number keeps its spelling. Without arguments, rewrite the managed files.
"""

import argparse
import json
import re
import stat
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Captures that every reader parses. Keep out inputs the product reads as files and any JSON that is hashed, read
# as bytes or compared as text.
MANAGED = (
    "tests/fixtures/skillspector-*.json",
    "tests/fixtures/skillspector-*/*.json",
    "tests/golden/cli_surface.json",
    "tests/reporting/fixtures/*.json",
    "tests/tier3/fixtures/*.json",
)
WIDTH = 200
_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[\[\]{}:,]|[^\s"\[\]{}:,]+')
_CLOSE = {"[": "]", "{": "}"}
# A scalar is its text in the file. A list or dict is its opening bracket and its (prefix, child) items; the prefix
# is '"key": ' in a dict and empty in a list.
Node = str | tuple[str, list]


def _depth(node: Node) -> int:
    return 0 if isinstance(node, str) else 1 + max((_depth(child) for _, child in node[1]), default=0)


def _flat(node: Node) -> str:
    if isinstance(node, str):
        return node
    return node[0] + ", ".join(prefix + _flat(child) for prefix, child in node[1]) + _CLOSE[node[0]]


def _lines(node: Node, indent: int, prefix: str, comma: str) -> Iterator[str]:
    line = " " * indent + prefix + _flat(node) + comma
    if isinstance(node, str) or not node[1] or (_depth(node) <= 2 and len(line) <= WIDTH):
        yield line
        return
    yield " " * indent + prefix + node[0]
    for index, (child_prefix, child) in enumerate(node[1], 1):
        yield from _lines(child, indent + 2, child_prefix, "," if index < len(node[1]) else "")
    yield " " * indent + _CLOSE[node[0]] + comma


def layout(text: str) -> str:
    """Return the JSON document *text* in the layout."""
    json.loads(text)  # The tokens below are not validated, so refuse anything that is not JSON here.
    root: tuple[str, list] = ("", [])
    stack = [root]
    prefix = ""
    for token in _TOKEN.findall(text):
        if token == ":":
            prefix = stack[-1][1].pop()[1] + ": "
        elif token in _CLOSE.values():
            stack.pop()
        elif token != ",":
            node = (token, []) if token in _CLOSE else token
            stack[-1][1].append((prefix, node))
            prefix = ""
            if token in _CLOSE:
                stack.append(node)
    return "\n".join(_lines(root[1][0][1], 0, "", "")) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="write nothing; list files not in the layout and exit 1")
    check = parser.parse_args(argv).check
    stale = 0
    for path in sorted(path for pattern in MANAGED for path in ROOT.glob(pattern)):
        relative = path.relative_to(ROOT).as_posix()
        try:
            if path.resolve(strict=True) != path:
                raise ValueError("symbolic links are not supported")
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError("expected a regular file with exactly one hard link")
            text = path.read_bytes().decode("utf-8")
            laid_out = layout(text)
            if laid_out == text:
                continue
            if not check:
                path.write_text(laid_out, encoding="utf-8", newline="\n")
        except (OSError, RuntimeError, ValueError) as error:
            raise SystemExit(f"{relative}: {error}") from error
        stale += 1
        print(("not in layout: " if check else "rewrote: ") + relative)
    return int(check and stale > 0)


if __name__ == "__main__":
    raise SystemExit(main())
