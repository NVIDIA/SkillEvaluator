# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

from test_behavior_evidence import _trajectory_with_late_write

from skillevaluator.tier3.eval_core import atif_helpers
from skillevaluator.tier3.eval_core.atif_helpers import build_conversation_summary

METRICS = ("accuracy", "goal_accuracy", "behavior_check")


def test_bundle_shape_has_all_metrics_and_fields() -> None:
    bundles = atif_helpers.build_metric_evidence_bundles(
        _trajectory_with_late_write(),
        "Update the suite for GPU execution.",
        ground_truth="A GPU test file is written to /workspace/output/.",
        expected_behavior=["The agent writes a GPU engine test."],
    )
    assert set(bundles) == set(METRICS)
    for metric in METRICS:
        b = bundles[metric]
        assert isinstance(b["prompt_evidence"], str) and b["prompt_evidence"].strip()
        assert isinstance(b["evidence_refs"], list)
        assert set(b["omitted"]) >= {"count", "truncated", "reason"}


def test_late_write_survives_into_accuracy_and_goal_evidence() -> None:
    traj = _trajectory_with_late_write()
    old = build_conversation_summary(traj, "question")[:3000]
    assert "test_gpu_engine_selection.py" not in old  # old judge would miss it

    bundles = atif_helpers.build_metric_evidence_bundles(
        traj, "question", ground_truth="GPU test written", expected_behavior=["writes GPU test"]
    )
    for metric in ("accuracy", "goal_accuracy", "behavior_check"):
        assert "test_gpu_engine_selection.py" in bundles[metric]["prompt_evidence"], metric


def test_over_budget_sets_truncated_and_reason_not_silent() -> None:
    steps = [{"source": "user", "message": "go"}]
    for i in range(60):
        steps.append({
            "source": "agent",
            "message": f"step {i} " + ("y" * 1200),
            "tool_calls": [{
                "tool_call_id": f"t{i}", "function_name": "bash",
                "arguments": {"command": f"echo {i} " + ("z" * 1200)},
            }],
            "observation": {"results": [{"source_call_id": f"t{i}", "content": "out " + ("w" * 1200)}]},
        })
    steps.append({"source": "agent", "message": "FINAL_MARKER done"})
    bundles = atif_helpers.build_metric_evidence_bundles({"steps": steps}, "q", ground_truth="gt")
    goal = bundles["goal_accuracy"]
    assert goal["omitted"]["truncated"] is True
    assert goal["omitted"]["count"] > 0
    assert goal["omitted"]["reason"]
    assert "FINAL_MARKER" in goal["prompt_evidence"]  # guaranteed-include survives truncation

    bc = bundles["behavior_check"]
    assert bc["omitted"]["truncated"] is True  # behavior truncation is reported, not silent
    assert bc["omitted"]["reason"]


def test_template_bundles_match_shared_helper() -> None:
    template_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "skillevaluator"
        / "tier3"
        / "harbor"
        / "templates"
        / "eval.py"
    )
    spec = importlib.util.spec_from_file_location("harbor_eval_template", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    traj = _trajectory_with_late_write()
    args = {"ground_truth": "GPU test written", "expected_behavior": ["writes GPU test"]}
    shared = atif_helpers.build_metric_evidence_bundles(traj, "question", **args)
    templated = module.build_metric_evidence_bundles(traj, "question", **args)

    for metric in METRICS:
        assert shared[metric]["prompt_evidence"] == templated[metric]["prompt_evidence"], metric
        assert shared[metric]["omitted"] == templated[metric]["omitted"], metric


def test_bundle_budgets_defaults_and_env_overrides(monkeypatch) -> None:
    # Defaults
    assert atif_helpers._accuracy_budget() == 8000
    assert atif_helpers._goal_accuracy_budget() == 12000
    budgets = atif_helpers._bundle_budgets()
    assert budgets["accuracy"] == 8000
    assert budgets["goal_accuracy"] == 12000
    assert budgets["behavior_check"] == 8000

    # Non-integer / invalid fallback
    monkeypatch.setenv("SKILL_EVAL_ACCURACY_BUDGET", "invalid")
    monkeypatch.setenv("SKILL_EVAL_GOAL_ACCURACY_BUDGET", "-10")
    assert atif_helpers._accuracy_budget() == 8000
    assert atif_helpers._goal_accuracy_budget() == 1  # max(1, -10) is 1

    # Valid environment overrides
    monkeypatch.setenv("SKILL_EVAL_ACCURACY_BUDGET", "15000")
    monkeypatch.setenv("SKILL_EVAL_GOAL_ACCURACY_BUDGET", "25000")
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", "10000")
    assert atif_helpers._accuracy_budget() == 15000
    assert atif_helpers._goal_accuracy_budget() == 25000
    budgets_custom = atif_helpers._bundle_budgets()
    assert budgets_custom["accuracy"] == 15000
    assert budgets_custom["goal_accuracy"] == 25000
    assert budgets_custom["behavior_check"] == 10000


def test_harbor_template_bundle_budgets_match_shared(monkeypatch) -> None:
    template_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "skillevaluator"
        / "tier3"
        / "harbor"
        / "templates"
        / "eval.py"
    )
    spec = importlib.util.spec_from_file_location("harbor_eval_template_budgets", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setenv("SKILL_EVAL_ACCURACY_BUDGET", "9000")
    monkeypatch.setenv("SKILL_EVAL_GOAL_ACCURACY_BUDGET", "18000")
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", "7000")

    assert module._accuracy_budget() == atif_helpers._accuracy_budget() == 9000
    assert module._goal_accuracy_budget() == atif_helpers._goal_accuracy_budget() == 18000
    assert module._bundle_budgets() == atif_helpers._bundle_budgets()


def test_verifier_env_vars_and_runner_include_budget_vars(monkeypatch) -> None:
    from skillevaluator.tier3.harbor.adapter import _VERIFIER_PROVIDER_ENV_VARS
    from skillevaluator.tier3.harbor.runner import ProviderConfig, _provider_environment

    budget_vars = {
        "SKILL_EVAL_ACCURACY_BUDGET",
        "SKILL_EVAL_BEHAVIOR_CHECK_BUDGET",
        "SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT",
        "SKILL_EVAL_GOAL_ACCURACY_BUDGET",
    }
    assert budget_vars.issubset(_VERIFIER_PROVIDER_ENV_VARS)

    monkeypatch.setenv("SKILL_EVAL_ACCURACY_BUDGET", "10000")
    monkeypatch.setenv("SKILL_EVAL_GOAL_ACCURACY_BUDGET", "20000")
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", "9000")
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", "1200")

    config = ProviderConfig(
        provider="openai",
        model="gpt-4o",
        api_key="dummy",
        base_url="https://api.openai.com/v1",
        litellm_model="openai/gpt-4o",
    )
    env = _provider_environment(config)
    assert env["SKILL_EVAL_ACCURACY_BUDGET"] == "10000"
    assert env["SKILL_EVAL_GOAL_ACCURACY_BUDGET"] == "20000"
    assert env["SKILL_EVAL_BEHAVIOR_CHECK_BUDGET"] == "9000"
    assert env["SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT"] == "1200"

