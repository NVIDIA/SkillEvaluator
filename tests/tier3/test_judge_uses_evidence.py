# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from test_behavior_evidence import _trajectory_with_late_write

from skillevaluator.tier3.eval_core import atif_helpers, llm_judge


def test_accuracy_judges_provided_evidence(monkeypatch):
    captured = {}

    def fake_hub(prompt, **_kw):
        captured["prompt"] = prompt
        return '{"score": 1.0, "reason": "ok", "criteria": {}}', None

    monkeypatch.setattr(llm_judge, "call_public_llm", fake_hub)
    bundles = atif_helpers.build_metric_evidence_bundles(
        _trajectory_with_late_write(), "q", ground_truth="gt", expected_behavior=[])
    llm_judge.judge_accuracy("q", "gt", bundles["accuracy"]["prompt_evidence"])
    assert "test_gpu_engine_selection.py" in captured["prompt"]  # late proof reached the judge


def test_goal_accuracy_custom_judges_end_state_evidence(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        llm_judge, "call_public_llm",
        lambda prompt, **_kw: (captured.__setitem__("prompt", prompt) or
                              ('{"achieved": true, "score": 1.0, "reason": "ok"}', None)))
    bundles = atif_helpers.build_metric_evidence_bundles(
        _trajectory_with_late_write(), "q", ground_truth="gt", expected_behavior=[])
    llm_judge.judge_goal_accuracy("q", "gt", bundles["goal_accuracy"]["prompt_evidence"], tool_summary="")
    assert "test_gpu_engine_selection.py" in captured["prompt"]


def test_behavior_check_does_not_recut_compiled_evidence(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        llm_judge, "call_public_llm",
        lambda prompt, **_kw: (captured.__setitem__("prompt", prompt) or
                              ('{"results": [], "score": 1.0, "summary": "ok"}', None)))
    big = "WRITE_MARKER " + ("e" * 6000) + " TAIL_MARKER"
    llm_judge.judge_behavior_check(big, ["did the agent write the file?"])
    assert "WRITE_MARKER" in captured["prompt"] and "TAIL_MARKER" in captured["prompt"]


def test_accuracy_keeps_evidence_past_old_3000_cutoff(monkeypatch):
    # Would FAIL under the old agent_text[:3000] truncation (marker sits at ~3515).
    captured = {}
    monkeypatch.setattr(
        llm_judge, "call_public_llm",
        lambda prompt, **_kw: (captured.__setitem__("prompt", prompt) or
                              ('{"score": 1.0, "reason": "ok", "criteria": {}}', None)))
    evidence = ("A" * 3500) + " LATE_ACC_MARKER"
    llm_judge.judge_accuracy("q", "gt", evidence)
    assert "LATE_ACC_MARKER" in captured["prompt"]


def test_goal_accuracy_keeps_evidence_past_old_3000_cutoff(monkeypatch):
    # Would FAIL under the old agent_text[:3000] truncation.
    captured = {}
    monkeypatch.setattr(
        llm_judge, "call_public_llm",
        lambda prompt, **_kw: (captured.__setitem__("prompt", prompt) or
                              ('{"achieved": true, "score": 1.0, "reason": "ok"}', None)))
    evidence = ("B" * 3500) + " LATE_GOAL_MARKER"
    llm_judge.judge_goal_accuracy("q", "gt", evidence, tool_summary="")
    assert "LATE_GOAL_MARKER" in captured["prompt"]


def test_behavior_keeps_middle_evidence_under_new_budget(monkeypatch):
    # MIDDLE marker is dropped by the old 4000-char middle truncation but kept at 8000.
    captured = {}
    monkeypatch.setattr(
        llm_judge, "call_public_llm",
        lambda prompt, **_kw: (captured.__setitem__("prompt", prompt) or
                              ('{"results": [], "score": 1.0, "summary": "ok"}', None)))
    conv = ("h" * 3000) + " MIDDLE_MARKER " + ("t" * 4000)  # ~7015 chars
    llm_judge.judge_behavior_check(conv, ["did the agent write the file?"])
    assert "MIDDLE_MARKER" in captured["prompt"]
