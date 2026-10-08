# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proof M15: a confirmed negative Skill Lift warns, and fails ``validate`` with ``--block-on-agent-eval``.

check-13 edge-05: lift -0.20 with the whole 95% interval below zero still gave PASS, no warning, exit 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _f90_gate_cli import EDGE05_INTERVAL, EDGE05_WITH, EDGE05_WITHOUT, callout, run_validate, tier3_result

# ---------------------------------------------------------------------------
# M15: a confirmed negative Skill Lift warns, and gates with --block-on-agent-eval
# ---------------------------------------------------------------------------


def test_confirmed_negative_lift_fails_validate_with_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tier3 = tier3_result(EDGE05_WITH, EDGE05_WITHOUT, lift_uncertainty=EDGE05_INTERVAL)

    exit_code, report, benchmark, footer = run_validate(monkeypatch, tmp_path, tier3, "--block-on-agent-eval")

    payload = report["tier3"]
    assert payload["verdict"] == "pass"  # the dimension gate alone still passes
    assert payload["overall_lift"] == pytest.approx(-0.1963, abs=1e-4)
    assert payload["lift_band"]["verdict"] == "fail"
    assert payload["lift_band"]["regression_confirmed"] is True
    assert exit_code == 1
    assert report["overall_status"] == "failed"
    assert "Publication blocked" in callout(benchmark)
    assert "Skill Lift" in footer


def test_confirmed_negative_lift_warns_without_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tier3 = tier3_result(EDGE05_WITH, EDGE05_WITHOUT, lift_uncertainty=EDGE05_INTERVAL)

    exit_code, report, benchmark, _footer = run_validate(monkeypatch, tmp_path, tier3)

    conclusions = report["tier3"]["conclusions"]
    regression = [item for item in conclusions if item.get("title") == "Skill Lift regression"]
    assert regression and regression[0]["severity"] == "fail"
    assert "-0.20" in regression[0]["message"]
    assert exit_code == 0
    head = benchmark.split("## Evaluation Metadata")[0]
    assert "Recommended for publication" not in head
    assert "Publication blocked" not in head
    assert "Skill Lift" in head and "-20 points" in head


def test_unconfirmed_negative_lift_warns_but_never_gates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    interval = json.loads(json.dumps(EDGE05_INTERVAL))
    interval["effectiveness"].update({"ci_low": -0.42, "ci_high": 0.03, "ci_includes_zero": True, "precision": "low"})
    tier3 = tier3_result(EDGE05_WITH, EDGE05_WITHOUT, lift_uncertainty=interval)

    exit_code, report, _benchmark, _footer = run_validate(monkeypatch, tmp_path, tier3, "--block-on-agent-eval")

    payload = report["tier3"]
    assert payload["lift_band"]["verdict"] == "fail"
    assert payload["lift_band"]["regression_confirmed"] is False
    warnings = [item for item in payload["conclusions"] if item.get("title") == "Negative Skill Lift"]
    assert warnings and warnings[0]["severity"] == "warn"
    assert exit_code == 0


def test_too_few_paired_cases_never_confirm_a_regression(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Two paired cases: the interval is only the spread of the resampled means, below zero or not.
    interval = json.loads(json.dumps(EDGE05_INTERVAL))
    interval["effectiveness"].update({"ci_low": -0.3, "ci_high": -0.12, "n_cases": 2, "precision": "insufficient"})
    tier3 = tier3_result(EDGE05_WITH, EDGE05_WITHOUT, lift_uncertainty=interval)

    exit_code, report, benchmark, _footer = run_validate(monkeypatch, tmp_path, tier3, "--block-on-agent-eval")

    payload = report["tier3"]
    assert payload["lift_band"]["verdict"] == "fail"
    assert payload["lift_band"]["regression_confirmed"] is False
    assert payload["lift_band"]["precision"] == "insufficient"
    warnings = [item for item in payload["conclusions"] if item.get("title") == "Negative Skill Lift"]
    assert warnings and "fewer than 5 paired cases" in warnings[0]["message"]
    assert "Skill Lift regression" not in benchmark
    assert exit_code == 0


def test_integration_only_lift_is_not_a_skill_lift_band(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_config = {
        "eval_target": {"kind": "plugin"},
        "lift_mode": {"requested": "integration", "effective": "integration"},
    }
    tier3 = tier3_result(EDGE05_WITH, EDGE05_WITHOUT, lift_uncertainty=EDGE05_INTERVAL, run_config=run_config)

    exit_code, report, _benchmark, _footer = run_validate(monkeypatch, tmp_path, tier3, "--block-on-agent-eval")

    # Integration compares the plugin with its own parts; it is advisory and never gates.
    assert report["tier3"].get("lift_band") is None
    assert exit_code == 0
