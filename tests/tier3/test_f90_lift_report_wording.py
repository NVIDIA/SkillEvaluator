# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof H5 and M14 follow-ups: report wording names the right dimensions and the right agent.

- BENCHMARK's shared-basis note said every dimension missing from the lift
  "needs the plugin installed", including Correctness when no case had a
  ground_truth (N/A in both runs).
- A multi-agent Markdown report listed the best agent's lift intervals with no
  agent name, right after the per-agent Integration lines, and left out the
  other agent's interval (proof check-20 pos-04 and pos-06).
"""

from __future__ import annotations

from pathlib import Path

from _f90_lift_fixtures import NA, benchmark, collect, markdown, metrics, payload

CASES = [f"case-{index}" for index in range(1, 7)]


def test_benchmark_note_names_only_the_dimensions_that_need_the_plugin(tmp_path: Path) -> None:
    plugin = metrics(0.6, accuracy=NA, skill_execution=1.0, skill_efficiency=1.0)
    no_plugin = metrics(0.4, skill=False, accuracy=NA)
    arms = {"with": {case: [plugin] for case in CASES}, "without": {case: [no_plugin] for case in CASES}}
    collect(tmp_path, {"claude-code": arms}, n_attempts=1)

    card = " ".join(benchmark(payload(tmp_path)).split())

    assert "only the dimensions both runs scored (Security, Effectiveness) are compared" in card
    assert (
        "Discoverability and Efficiency need the plugin installed, so the run without it has no score for them." in card
    )
    assert "The lift also leaves out Correctness, which does not have a score in both runs." in card
    assert "Correctness, Discoverability and Efficiency need the plugin installed" not in card


def test_benchmark_note_for_one_dimension_is_singular(tmp_path: Path) -> None:
    plugin = metrics(0.6, skill_execution=1.0, skill_efficiency=1.0)
    no_plugin = metrics(0.4, skill=False, skill_efficiency=1.0)
    arms = {"with": {case: [plugin] for case in CASES}, "without": {case: [no_plugin] for case in CASES}}
    collect(tmp_path, {"claude-code": arms}, n_attempts=1)

    card = " ".join(benchmark(payload(tmp_path)).split())

    assert "Discoverability needs the plugin installed, so the run without it has no score for it." in card
    assert "leaves out" not in card


def test_multi_agent_markdown_names_the_agent_of_every_lift_interval(tmp_path: Path) -> None:
    # Proof check-20 pos-04 (claude-code, Integration +0.03) and pos-06 (codex, +0.20).
    cases = [f"case-{index:02d}" for index in range(1, 11)]

    def arms(plugin: float, parts: float) -> dict:
        return {
            "with": {case: [metrics(plugin)] for case in cases},
            "without": {case: [metrics(0.3)] for case in cases},
            "sumofparts": {case: [metrics(parts)] for case in cases},
        }

    collect(tmp_path, {"claude-code": arms(0.63, 0.6), "codex": arms(0.8, 0.6)}, n_attempts=1)
    report = markdown(payload(tmp_path, "both"))
    section = report.split("### Integration (advisory)", 1)[1].split("\n### ", 1)[0]
    bullets = [line for line in section.splitlines() if line.startswith("- ")]

    assert "- claude-code — Integration lift: +0.03" in "\n".join(bullets)
    assert "- codex — Integration lift: +0.20" in "\n".join(bullets)
    assert "- claude-code — Effectiveness lift:" in "\n".join(bullets)
    assert "- codex — Effectiveness lift:" in "\n".join(bullets)
    assert all(bullet.startswith(("- claude-code — ", "- codex — ")) for bullet in bullets)
