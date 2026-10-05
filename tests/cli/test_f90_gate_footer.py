# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof L5: the quiet footer says INCOMPLETE for an INCOMPLETE run (check-01 k05, k06, k08), not FAIL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _f90_gate_cli import callout, run_validate, scores, tier3_result

from skillevaluator.utils.tool_runner import ToolResult, Tools

# ---------------------------------------------------------------------------
# L5: the quiet footer says INCOMPLETE for an INCOMPLETE run
# ---------------------------------------------------------------------------


class _FakeClaude:
    def __init__(self, response: ToolResult) -> None:
        self.response = response

    @property
    def is_available(self) -> bool:
        return True

    def get_install_hint(self) -> str:
        return "Install Claude Code"

    def run(self, _args: list[str], **_kwargs: Any) -> ToolResult:
        return self.response


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(ToolResult(True, "", "fake claude: simulated crash", 3), id="k05-crash"),
        pytest.param(
            ToolResult(False, "", "", -1, error_message="Claude Code timed out after 120 seconds"), id="k06-timeout"
        ),
        pytest.param(ToolResult(True, "not a report", "", 0), id="k08-garbage"),
    ],
)
def test_incomplete_tier1_footer_says_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, response: ToolResult
) -> None:
    monkeypatch.setattr(Tools, "claude", _FakeClaude(response))

    exit_code, report, _benchmark, footer = run_validate(
        monkeypatch, tmp_path, None, tiers="1", checks="schema,claude-validate"
    )

    assert report["overall_status"] == "incomplete"
    assert exit_code == 1
    assert "INCOMPLETE" in footer
    assert "✗ FAIL" not in footer
    assert "failed in Tier 1" not in footer
    assert "did not complete in Tier 1" in footer


def test_incomplete_tier3_plugin_run_footer_says_incomplete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provenance = {"partial": True, "unresolved_skill_refs": ["github::example-org/example-repo::skills::missing"]}
    tier3 = tier3_result(scores(), plugin_provenance=provenance)

    exit_code, _report, _benchmark, footer = run_validate(monkeypatch, tmp_path, tier3, "--block-on-agent-eval")

    assert exit_code == 1
    assert "INCOMPLETE" in footer
    assert "✗ FAIL" not in footer


def test_real_failure_outranks_incomplete_in_the_footer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provenance = {"partial": True, "unresolved_skill_refs": ["github::example-org/example-repo::skills::missing"]}
    tier3 = tier3_result(scores(security=0.0), plugin_provenance=provenance)

    exit_code, _report, benchmark, footer = run_validate(monkeypatch, tmp_path, tier3, "--block-on-agent-eval")

    # The card says FAIL for a partial run whose verdict is FAIL; the footer must agree.
    assert "FAIL" in callout(benchmark)
    assert exit_code == 1
    assert "✗ FAIL" in footer
