# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof H1: a Tier 3 FAIL verdict fails ``validate`` only with ``--block-on-agent-eval``, and every surface says which.

Before the fix a FAIL verdict never failed ``validate``, even with the flag, while BENCHMARK.md said "Publication
blocked" (check-28 gate probe: exit 0 both ways). NEUTRAL never gates.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from _f90_gate_cli import callout, run_validate, scores, tier3_result
from click.testing import CliRunner

import skillevaluator.cli as cli_module

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# H1: a FAIL verdict gates only with --block-on-agent-eval, and every surface says which
# ---------------------------------------------------------------------------


def test_fail_verdict_fails_validate_with_block_on_agent_eval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tier3 = tier3_result(scores(security=0.0))

    exit_code, report, benchmark, footer = run_validate(monkeypatch, tmp_path, tier3, "--block-on-agent-eval")

    assert report["tier3"]["verdict"] == "fail"
    assert report["gating"]["tiers"]["3"]["blocking"] is True
    assert exit_code == 1
    assert report["overall_status"] == "failed"
    assert "Publication blocked" in callout(benchmark)
    assert "exit 1" in footer
    assert "Tier 3 verdict FAIL" in footer
    assert "all 2 tiers passed" not in footer


def test_fail_verdict_is_advisory_without_the_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tier3 = tier3_result(scores(security=0.0))

    exit_code, report, benchmark, footer = run_validate(monkeypatch, tmp_path, tier3)

    assert report["tier3"]["verdict"] == "fail"
    assert report["gating"]["tiers"]["3"]["blocking"] is False
    assert exit_code == 0
    assert report["overall_status"] == "passed"
    head_line = callout(benchmark)
    assert "FAIL" in head_line
    # Exit 0, so the card must not claim the publication was blocked; it says the Tier 3 result was advisory.
    assert "Publication blocked" not in benchmark
    assert "advisory" in benchmark.split("## Evaluation Metadata")[0]
    assert "--block-on-agent-eval" in benchmark
    assert "exit 0" in footer
    assert "all 2 tiers passed" not in footer
    assert "Tier 3 verdict FAIL is advisory" in footer


def test_neutral_verdict_never_fails_validate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tier3 = tier3_result(scores(security=0.45))

    exit_code, report, benchmark, _footer = run_validate(monkeypatch, tmp_path, tier3, "--block-on-agent-eval")

    assert report["tier3"]["verdict"] == "neutral"
    assert exit_code == 0
    assert report["overall_status"] == "passed"
    assert "Publication blocked" not in benchmark


def test_pass_verdict_with_flag_exits_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tier3 = tier3_result(scores())

    exit_code, report, benchmark, _footer = run_validate(monkeypatch, tmp_path, tier3, "--block-on-agent-eval")

    assert report["tier3"]["verdict"] == "pass"
    assert exit_code == 0
    assert "Publication blocked" not in benchmark


def test_validate_help_states_what_the_flag_gates() -> None:
    result = CliRunner().invoke(cli_module.cli, ["validate", "--help"], terminal_width=200)
    text = " ".join(result.output.split())

    assert result.exit_code == 0
    assert "FAIL verdict" in text
    assert "NEUTRAL" in text
