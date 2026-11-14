#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SessionStart hook: report which work tree this session resolved."""

import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import plugin_paths


def main():
    with contextlib.suppress(ValueError):
        json.load(sys.stdin)
    root = plugin_paths.project_root()
    message = f"build-notes ready in {root}." if root else "build-notes: no work tree found."
    sys.stdout.write(json.dumps({"continue": True, "suppressOutput": False, "systemMessage": message}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
