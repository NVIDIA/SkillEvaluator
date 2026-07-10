# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 validation streams per-check progress.

All Tier 1 results used to be buffered until the final report; on slow
targets the run printed nothing for many minutes and looked hung. Each
check must announce itself when it starts and report its duration when
it completes, on the progress (stderr) console.
"""

import io
from pathlib import Path

import pytest
from rich.console import Console

from skillevaluator.constants import CONTENT_TYPE_PLUGIN
from skillevaluator.tier1 import commands


@pytest.fixture
def progress_capture(monkeypatch) -> io.StringIO:
    buffer = io.StringIO()
    monkeypatch.setattr(commands, "progress_console", Console(file=buffer, force_terminal=False, width=200))
    return buffer


@pytest.fixture
def tiny_skill(tmp_path) -> Path:
    skill = tmp_path / "tiny-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: tiny-skill\ndescription: minimal test skill\n---\n\n# Tiny\n\nDoes nothing.\n"
    )
    return skill


class TestPerCheckProgress:
    def test_enabled_checks_default_and_aliases_are_explicit(self):
        assert commands._enabled_checks(None) == set(commands.DEFAULT_CHECKS)
        assert commands._enabled_checks(" schema, code, dependencies, licence, script-lint, version ,, ") == {
            "schema",
            "code-integrity",
            "dependency",
            "license",
            "lint",
            "version",
        }

    def test_each_check_announces_start_and_duration(self, tiny_skill, progress_capture):
        commands.run_validation(tiny_skill, checks="schema,license")

        output = progress_capture.getvalue()
        assert "[1/2] schema ..." in output
        assert "[2/2] license ..." in output
        assert output.count("done in") == 2

    def test_completion_line_reports_outcome(self, tiny_skill, progress_capture):
        commands.run_validation(tiny_skill, checks="license")

        output = progress_capture.getvalue()
        assert "[1/1] license done in" in output
        assert "s (" in output  # duration followed by outcome summary

    def test_step_count_reflects_enabled_checks_only(self, tiny_skill, progress_capture):
        commands.run_validation(tiny_skill, checks="schema")

        output = progress_capture.getvalue()
        assert "[1/1] schema" in output
        assert "[1/2]" not in output

    def test_step_count_excludes_skill_only_checks_for_plugin(self, tmp_path, progress_capture):
        plugin = tmp_path / "my-plugin"
        plugin.mkdir()
        (plugin / "agent_plugin.yaml").write_text(
            """
name: my-bundle
author:
  email: dev@example.com
skills:
  refs:
    - "github::example-org/example-repo::skills::build-infra"
""",
            encoding="utf-8",
        )

        commands.run_validation(plugin, checks="schema,quality,lint", content_type=CONTENT_TYPE_PLUGIN)

        output = progress_capture.getvalue()
        assert "[1/1] schema ..." in output
        assert "quality" not in output
        assert "lint" not in output
