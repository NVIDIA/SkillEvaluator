# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden regression guard for the CLI command/option surface.

If this test fails, the CLI command tree changed. If the change was intentional,
regenerate the snapshot with ``tests/golden/generate_cli_surface.py`` and review
the diff. This freezes structure (commands, options, choices) so renames and
refactors can be proven to preserve behavior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli_surface import build_cli_surface

from skillevaluator.cli import cli

GOLDEN = Path(__file__).resolve().parent / "cli_surface.json"


def test_callable_defaults_are_named_without_evaluation() -> None:
    def first_default():
        raise AssertionError("Golden serialization must not call a dynamic default")

    def second_default():
        raise AssertionError("Golden serialization must not call a dynamic default")

    group = click.Group(
        params=[click.Option(["--first"], default=first_default), click.Option(["--second"], default=second_default)]
    )
    defaults = [param["default"] for param in build_cli_surface(group)["params"]]
    assert defaults[0] != defaults[1]
    assert "first_default" in defaults[0] and "second_default" in defaults[1]
    assert all(value.startswith("<callable ") and "0x" not in value for value in defaults)


def test_cwd_default_uses_its_stable_public_identity() -> None:
    group = click.Group(params=[click.Option(["--repo"], default=Path.cwd)])
    assert build_cli_surface(group)["params"][0]["default"] == "<callable pathlib.Path.cwd>"


def test_cli_surface_matches_golden() -> None:
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    actual = build_cli_surface(cli)

    assert actual == expected, (
        "CLI surface drifted from the golden baseline. If intentional, run "
        "tests/golden/generate_cli_surface.py and review the diff."
    )
