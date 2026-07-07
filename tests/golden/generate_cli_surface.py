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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli_surface import build_cli_surface

from skillevaluator.cli import cli


def main() -> None:
    snapshot = build_cli_surface(cli)
    out = Path(__file__).resolve().parent / "cli_surface.json"
    out.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
