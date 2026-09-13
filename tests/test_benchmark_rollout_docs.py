# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the BENCHMARK.md backfill runbook aligned with the public contract."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_rollout_runbook_covers_safe_backfill_and_freshness() -> None:
    runbook = (REPO / "docs/benchmark-rollout.mdx").read_text(encoding="utf-8")

    assert "--output-dir ./benchmark-backfill/example-skill" in runbook
    assert "check_public_benchmarks.py" in runbook
    assert "assets/benchmark-card-before.png" in runbook
    assert "assets/benchmark-card-after.png" in runbook
    assert "--require-files" in runbook
    assert "Do not reuse a prior live score" in runbook
    assert "The generated evaluation date must come from the live run artifact" in runbook
    assert "Effectiveness=50% `goal_accuracy` + 50% `behavior_check`" in runbook
    assert "A dimension passes at 50%" in runbook
    assert "benchmark_policy.tier3_required = false" in runbook
    assert "publication `PASS`" in runbook
    assert "without completed required Tier 3 evidence" in runbook


def test_tier3_reference_uses_canonical_dimension_mapping() -> None:
    reference = (REPO / "docs/tier3-live-evaluation.mdx").read_text(encoding="utf-8")

    assert "| Security | Is it safe to use? | `security` | 1.0 |" in reference
    assert "| Correctness | Is the answer correct? | `accuracy` | 1.0 |" in reference
    assert "| Discoverability | Was the right skill loaded when needed? | `skill_execution` | 1.0 |" in reference
    assert (
        "| Effectiveness | Did the skill help complete the task? | `goal_accuracy` + `behavior_check` | 0.5 + 0.5 |"
    ) in reference
    assert "`goal_accuracy` + `behavior_check` + `accuracy`" not in reference
    assert "At least one successful agent has every dimension ≥ 0.50" in reference
