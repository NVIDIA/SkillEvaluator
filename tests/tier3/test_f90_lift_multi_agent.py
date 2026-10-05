# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof M14 (Integration half): a multi-agent run shows one named Integration block per agent.

In the skeptic's two-agent live run (Claude Code and Codex, ``--lift-mode
both``) the report showed one unnamed Integration block, +0.1958, which was
Codex's; Claude Code's +0.04 appeared only as a statistics row. Every report
now names the agent of each Integration result.
"""

from __future__ import annotations

import re
from pathlib import Path

from _f90_lift_fixtures import benchmark, cli, collect, html, markdown, metrics, payload

CASES = [f"case-{index}" for index in range(1, 7)]


def _two_agent_run(tmp_path: Path) -> dict:
    def arms(plugin: float) -> dict:
        return {
            "with": {case: [metrics(plugin)] for case in CASES},
            "without": {case: [metrics(0.4, skill=False)] for case in CASES},
            "sumofparts": {case: [metrics(0.65)] for case in CASES},
        }

    collect(tmp_path, {"claude-code": arms(0.69), "codex": arms(0.85)}, n_attempts=1)
    return payload(tmp_path, "both")


def test_each_agent_gets_its_own_named_integration_block(tmp_path: Path) -> None:
    built = _two_agent_run(tmp_path)

    claude = built["agents"]["claude-code"]["integration"]
    codex = built["agents"]["codex"]["integration"]
    assert (claude["agent"], claude["integration_lift"]) == ("claude-code", 0.04)
    assert (codex["agent"], codex["integration_lift"]) == ("codex", 0.2)
    assert codex["verdict"] == "real_integration"
    assert claude["verdict"] != "real_integration"
    # The run-level block is the best agent's, and says whose it is.
    assert built["best_agent"] == "codex"
    assert built["integration"]["agent"] == "codex"
    assert built["integration"]["integration_lift"] == 0.2


def test_every_report_names_the_agent_of_each_integration_result(tmp_path: Path) -> None:
    built = _two_agent_run(tmp_path)

    report = markdown(built)
    assert re.search(r"\*\*claude-code — [^*]+:\*\* plugin 0\.69 vs sum-of-parts 0\.65 \(lift \+0\.04", report)
    assert "**codex — Real integration:** plugin 0.85 vs sum-of-parts 0.65 (lift +0.20" in report

    output = cli(built)
    assert re.search(r"Integration \(claude-code\):\s+\S+.*\(lift \+0\.04\)", output)
    assert re.search(r"Integration \(codex\):\s+REAL INTEGRATION \(lift \+0\.20\)", output)

    card = benchmark(built)
    assert "| Integration (plugin vs. its own parts) — claude-code |" in card
    assert "| Integration (plugin vs. its own parts) — codex | Real integration, +20 points |" in card

    page = html(built)
    section = page[page.index('id="tier3-integration"') :]
    section = section[: section.index("</details>")]
    assert "<h4>claude-code</h4>" in section and "<h4>codex</h4>" in section
    assert "+0.04" in section and "+0.20" in section
