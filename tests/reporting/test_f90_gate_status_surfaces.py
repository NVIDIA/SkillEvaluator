# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proofs H1 and L5 (terminal and HTML reports): an advisory Tier 3 FAIL is marked advisory, a gating one says it
gated, and a partial Tier 3 run is INCOMPLETE, not FAIL."""

from __future__ import annotations

import re

import pytest
from _f90_gate_payloads import build_payload, scores, terminal, tier3

from skillevaluator.reporting import HTMLReporter

# ---------------------------------------------------------------------------
# H1 and L5: terminal summary for advisory FAIL and INCOMPLETE Tier 3 runs
# ---------------------------------------------------------------------------


def test_terminal_marks_an_advisory_fail_as_advisory() -> None:
    output = terminal([tier3(build_payload(scores(security=0.0)), blocking=False)])

    assert re.search(r"AGENT_EVAL\s*│\s*FAIL \(advisory\)", output)
    assert "Tier 3 verdict FAIL" in output
    assert "All validations passed" not in output
    assert "Required validations passed" in output


def test_terminal_marks_a_partial_tier3_run_incomplete() -> None:
    provenance = {"partial": True, "unresolved_skill_refs": ["github::example-org/example-repo::skills::missing"]}
    output = terminal([tier3(build_payload(scores(), plugin_provenance=provenance), blocking=True)])

    assert re.search(r"AGENT_EVAL\s*│\s*INCOMPLETE", output)
    assert re.search(r"AGENT_EVAL\s*│\s*FAIL", output) is None
    assert "1 unresolved skill ref(s)" in output


@pytest.mark.parametrize(
    ("blocking", "expected"),
    [
        (False, "Tier 3 was advisory in this run, so this FAIL did not change the exit code"),
        (True, "made this FAIL gate the exit code"),
    ],
)
def test_html_verdict_banner_says_whether_the_fail_gated(blocking: bool, expected: str) -> None:
    html = HTMLReporter(include_timestamp=False, content_label="Plugin").render_all(
        [tier3(build_payload(scores(security=0.0)), blocking=blocking)]
    )
    lead = re.search(r'<div class="t3-overview-lead">(.*?)</div>', html, re.DOTALL)

    assert lead is not None
    assert expected in " ".join(lead.group(1).split())
