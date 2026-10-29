# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded report-only runtime evidence; no agent or provider invocation."""

from __future__ import annotations

import importlib
import json


def module():
    return importlib.import_module("skillevaluator.tier3.eval_core.runtime_evidence")


def record(hook="hook", event="PreToolUse", code=0, duration=5):
    return json.dumps({"hook_id": hook, "event": event, "exit_code": code, "duration_ms": duration})


def test_hook_outcomes_distinguish_deny_from_failed_start():
    result = module().parse_hook_census("\n".join(record(code=code) for code in (0, 2, 1, 126, 127)))
    assert result["status"] == "recorded"
    assert {key: result[key] for key in ("total_runs", "total_failures", "total_blocked", "total_not_started")} == {
        "total_runs": 5,
        "total_failures": 3,
        "total_blocked": 1,
        "total_not_started": 2,
    }
    assert result["hooks"][0]["total_duration_ms"] == 25


def test_bad_lines_and_boolean_exit_are_missing_evidence():
    result = module().parse_hook_census("\n".join([record(), "{", "[]", record(code=True)]))
    assert result["invalid_lines"] == 3
    assert result["total_runs"] == 1


def test_redaction_precedes_label_truncation():
    runtime = module()
    secret = "sk-" + "Aa1" * 20
    hook = "github_pat_" + "a" * 50 + " " + "x" * 190 + " " + secret
    parsed = runtime.parse_hook_census(record(hook=hook))
    serialized = json.dumps(parsed)
    assert secret not in serialized
    assert "sk-Aa1" not in serialized
    assert "github_pat_" + "a" * 20 not in serialized
    assert len(parsed["hooks"][0]["hook_id"]) <= runtime.MAX_HOOK_ID_CHARS


def test_runtime_census_caps_rows_and_lines(monkeypatch):
    runtime = module()
    monkeypatch.setattr(runtime, "MAX_HOOK_CENSUS_HOOKS", 2)
    monkeypatch.setattr(runtime, "MAX_HOOK_CENSUS_LINES", 3)
    result = runtime.parse_hook_census("\n".join(record(hook=f"h{i}") for i in range(5)))
    assert result["truncated"] is True
    assert [row["hook_id"] for row in result["hooks"]] == ["h0", "h1"]
    assert result["total_runs"] == 2


def test_census_reader_preserves_absent_recorded_and_unsafe(tmp_path):
    runtime = module()
    assert runtime.read_hook_census(tmp_path)["status"] == "absent"
    agent = tmp_path / "agent"
    agent.mkdir()
    path = agent / runtime.HOOK_CENSUS_FILENAME
    path.write_text(record())
    assert runtime.read_hook_census(tmp_path)["total_runs"] == 1
    path.unlink()
    outside = tmp_path / "outside.jsonl"
    outside.write_text(record(hook="outside"))
    path.symlink_to(outside)
    result = runtime.read_hook_census(tmp_path)
    assert result["status"] == "unreadable"
    assert result["total_runs"] == 0


def test_census_arm_summary_counts_missing_and_unreadable_trials():
    runtime = module()
    recorded = runtime.parse_hook_census(record())
    result = runtime.summarize_hook_census([recorded, None, runtime.empty_hook_census("unreadable")])
    assert (result["n_trials"], result["n_trials_with_census"], result["n_trials_unreadable"]) == (3, 1, 1)
    assert result["hooks"][0]["trials"] == 1
    assert result["total_runs"] == 1


def reward(*, planted=True, leaked=False, findings=()):
    return {
        "details": {
            "security": {
                "canary": {
                    "planted": planted,
                    "leaked": leaked,
                    "file_present": True,
                    "sinks": [{"kind": "network"}] if leaked else [],
                },
                "findings": list(findings),
            }
        }
    }


def test_canary_summary_compares_rates_and_keeps_unknown_attribution():
    runtime = module()
    with_arm = runtime.summarize_canary([reward(leaked=True)] * 2)
    baseline = runtime.summarize_canary([reward(leaked=True), reward(leaked=False)])
    assert with_arm["leak_rate"] == 1.0
    assert baseline["leak_rate"] == 0.5
    assert (
        runtime.canary_arm_comparison({"with_skill": with_arm, "without_skill": baseline})["plugin_attributable"]
        is True
    )
    assert runtime.canary_arm_comparison({"with_skill": with_arm})["plugin_attributable"] is None
    unknown = runtime.summarize_canary([reward(planted=False)])
    assert unknown["leak_rate"] is None
    assert (
        runtime.canary_arm_comparison({"with_skill": unknown, "without_skill": baseline})["plugin_attributable"] is None
    )


def test_no_canary_summary_can_still_report_protected_accesses():
    runtime = module()
    value = {
        "details": {
            "security": {
                "findings": [{"type": "sensitive_path_access", "score_impact": True, "evidence": "~/.aws/credentials"}]
            }
        }
    }
    assert runtime.summarize_canary([value]) is None
    summary = runtime.summarize_canary([value], without_canary=True)
    assert summary["canary_checked"] is False
    assert summary["credential_reads"] == 1
    assert summary["credential_stores"] == {"~/.aws/credentials": 1}


def test_counter_overflow_and_nonfinite_values_are_refused():
    runtime = module()
    result = runtime.parse_hook_census("\n".join([record(code=10**1000), record(duration=float("inf")), record()]))
    assert result["invalid_lines"] == 1
    assert result["total_runs"] == 2
    assert result["hooks"][0]["total_duration_ms"] == 5
