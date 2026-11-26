# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the frozen CLI surface snapshot.

Run from the repo root with the package importable, e.g.::

    PYTHONPATH=src python tests/golden/generate_cli_surface.py

The committed ``cli_surface.json`` is a golden baseline. Only regenerate it on a
*deliberate* CLI change and review the diff carefully.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

from _cli_surface import build_cli_surface
from scripts.layout_json import layout

from skillevaluator.cli import cli


def main() -> None:
    out = HERE / "cli_surface.json"
    out.write_text(layout(json.dumps(build_cli_surface(cli), sort_keys=True)), encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
