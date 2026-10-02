# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from test_behavior_evidence import _load_harbor_template_module, _trajectory_with_late_write

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
    module = _load_harbor_template_module()
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

    # Non-integer / invalid fallback across all 4 budget/limit env vars
    for invalid_val in ("invalid", "   ", "3.14", ""):
        monkeypatch.setenv("SKILL_EVAL_ACCURACY_BUDGET", invalid_val)
        monkeypatch.setenv("SKILL_EVAL_GOAL_ACCURACY_BUDGET", invalid_val)
        monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", invalid_val)
        monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", invalid_val)
        assert atif_helpers._accuracy_budget() == 8000
        assert atif_helpers._goal_accuracy_budget() == 12000
        assert atif_helpers._behavior_final_response_limit() == 800
        assert atif_helpers._behavior_check_budget() == 8000

    # Non-positive integer clamping and headroom reconciliation
    for non_positive in ("0", "-10", "-500"):
        monkeypatch.setenv("SKILL_EVAL_ACCURACY_BUDGET", non_positive)
        monkeypatch.setenv("SKILL_EVAL_GOAL_ACCURACY_BUDGET", non_positive)
        monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", non_positive)
        monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", non_positive)
        assert atif_helpers._accuracy_budget() == 1
        assert atif_helpers._goal_accuracy_budget() == 1
        assert atif_helpers._behavior_final_response_limit() == 1
        assert atif_helpers._behavior_check_budget() == 4001  # final_limit (1) + headroom (4000)

    # Valid environment overrides
    monkeypatch.setenv("SKILL_EVAL_ACCURACY_BUDGET", "15000")
    monkeypatch.setenv("SKILL_EVAL_GOAL_ACCURACY_BUDGET", "25000")
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", "800")
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", "10000")
    assert atif_helpers._accuracy_budget() == 15000
    assert atif_helpers._goal_accuracy_budget() == 25000
    budgets_custom = atif_helpers._bundle_budgets()
    assert budgets_custom["accuracy"] == 15000
    assert budgets_custom["goal_accuracy"] == 25000
    assert budgets_custom["behavior_check"] == 10000


def test_harbor_template_bundle_budgets_match_shared(monkeypatch) -> None:
    module = _load_harbor_template_module()

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


@pytest.mark.parametrize("budget_val", [100, 200])
def test_small_accuracy_and_goal_budgets_retain_final_response_excerpt(monkeypatch, budget_val: int) -> None:
    from skillevaluator.tier3.eval_core import llm_judge

    template_module = _load_harbor_template_module()
    monkeypatch.setenv("SKILL_EVAL_ACCURACY_BUDGET", str(budget_val))
    monkeypatch.setenv("SKILL_EVAL_GOAL_ACCURACY_BUDGET", str(budget_val))

    final_answer = (
        "FINAL_ANSWER_EXCERPT: The GPU execution failed because the CUDA driver was unavailable; "
        "switched to the CPU fallback configuration and documented the root cause in detail for the user."
        " Additional context padding to reach roughly two hundred sixty characters."
    )
    assert 230 <= len(final_answer) <= 270

    traj = {
        "steps": [
            {"source": "user", "message": "Run the GPU check and explain the result."},
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "t1",
                        "function_name": "bash",
                        "arguments": {"command": "pytest -q"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "t1",
                            "content": "ERROR: CUDA driver not found (exit 1)",
                        }
                    ]
                },
            },
            {"source": "agent", "message": final_answer},
        ]
    }
    question = "Run the GPU check and explain the result."
    ground_truth = "Explain that GPU execution failed due to missing CUDA driver."

    shared_bundles = atif_helpers.build_metric_evidence_bundles(traj, question, ground_truth=ground_truth)
    template_bundles = template_module.build_metric_evidence_bundles(traj, question, ground_truth=ground_truth)

    for metric in ("accuracy", "goal_accuracy"):
        shared_ev = shared_bundles[metric]["prompt_evidence"]
        template_ev = template_bundles[metric]["prompt_evidence"]
        assert shared_ev == template_ev, metric
        assert "FINAL RESPONSE" in shared_ev, metric
        assert "FINAL_ANSWER_EXCERPT:" in shared_ev, metric
        if budget_val >= 160:
            assert "ERROR: CUDA driver not found (exit 1)" in shared_ev, metric
        assert len(shared_ev) <= budget_val, metric
        assert shared_bundles[metric]["omitted"]["truncated"] is True, metric
        assert shared_bundles[metric]["omitted"]["count"] >= 1, metric

    captured_shared: dict[str, str] = {}
    captured_template: dict[str, str] = {}

    monkeypatch.setattr(
        llm_judge,
        "call_public_llm",
        lambda prompt, **_kw: (
            captured_shared.__setitem__("last", prompt)
            or (
                '{"achieved": true, "score": 1.0, "reason": "ok", "criteria": '
                '{"SKILL_IDENTIFIED": true, "ACTION_CORRECT": true, "FACTUALLY_ACCURATE": true, '
                '"TASK_ADDRESSED": true, "ACTIONABLE": true}}',
                None,
            )
        ),
    )
    monkeypatch.setattr(
        template_module,
        "call_public_llm",
        lambda prompt, **_kw: (
            captured_template.__setitem__("last", prompt)
            or (
                '{"achieved": true, "score": 1.0, "reason": "ok", "criteria": '
                '{"SKILL_IDENTIFIED": true, "ACTION_CORRECT": true, "FACTUALLY_ACCURATE": true, '
                '"TASK_ADDRESSED": true, "ACTIONABLE": true}}',
                None,
            )
        ),
    )

    llm_judge.judge_accuracy(question, ground_truth, shared_bundles["accuracy"]["prompt_evidence"])
    template_module.judge_accuracy(question, ground_truth, template_bundles["accuracy"]["prompt_evidence"])
    assert "FINAL_ANSWER_EXCERPT:" in captured_shared["last"]
    assert "FINAL_ANSWER_EXCERPT:" in captured_template["last"]

    llm_judge.judge_goal_accuracy(
        question, ground_truth, shared_bundles["goal_accuracy"]["prompt_evidence"], tool_summary=""
    )
    template_module.judge_goal_accuracy(
        question, ground_truth, template_bundles["goal_accuracy"]["prompt_evidence"], tool_summary=""
    )
    assert "FINAL_ANSWER_EXCERPT:" in captured_shared["last"]
    assert "FINAL_ANSWER_EXCERPT:" in captured_template["last"]


@pytest.mark.parametrize("budget", [10, 20, 26, 40])
def test_assemble_micro_budgets_and_missing_final_response_fallback(monkeypatch, budget: int) -> None:
    template_module = _load_harbor_template_module()
    sections = [
        ("FINAL RESPONSE", "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu."),
        ("KEY OBSERVATIONS", "Secondary observation that should be dropped."),
    ]
    shared_text, shared_drop, shared_trunc = atif_helpers._assemble(sections, budget)
    template_text, template_drop, template_trunc = template_module._assemble(sections, budget)

    assert shared_text == template_text
    assert shared_drop == template_drop == 2
    assert shared_trunc is True and template_trunc is True
    assert 0 < len(shared_text) <= budget

    # Sad/edge path: agent produced no final text message, only a large file write + tool observation
    monkeypatch.setenv("SKILL_EVAL_ACCURACY_BUDGET", "90")
    monkeypatch.setenv("SKILL_EVAL_GOAL_ACCURACY_BUDGET", "90")
    traj_no_final = {
        "steps": [
            {"source": "user", "message": "Generate output config."},
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "w1",
                        "function_name": "Write",
                        "arguments": {
                            "file_path": "/workspace/output/config.yaml",
                            "content": "setting: enabled\n" + ("# padding line\n" * 20),
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "w1", "content": "wrote config.yaml"}]},
            },
        ]
    }
    shared_b = atif_helpers.build_metric_evidence_bundles(traj_no_final, "Generate output config.")
    template_b = template_module.build_metric_evidence_bundles(traj_no_final, "Generate output config.")
    for metric in ("accuracy", "goal_accuracy"):
        assert shared_b[metric]["prompt_evidence"] == template_b[metric]["prompt_evidence"]
        assert len(shared_b[metric]["prompt_evidence"]) <= 90
        assert "config.yaml" in shared_b[metric]["prompt_evidence"]
        assert shared_b[metric]["omitted"]["truncated"] is True


@pytest.mark.parametrize("implementation", ["shared", "harbor"])
@pytest.mark.parametrize(
    "budgets", [None, (7000, 11000)], ids=["defaults", "custom"]
)
@pytest.mark.parametrize("metric", ["accuracy", "goal_accuracy"])
def test_tool_only_trajectory_retains_latest_verification_failure(monkeypatch, implementation, budgets, metric) -> None:
    for name in ("SKILL_EVAL_ACCURACY_BUDGET", "SKILL_EVAL_GOAL_ACCURACY_BUDGET"):
        monkeypatch.delenv(name, raising=False)
    accuracy_budget, goal_budget = budgets or (8000, 12000)
    if budgets:
        monkeypatch.setenv("SKILL_EVAL_ACCURACY_BUDGET", str(accuracy_budget))
        monkeypatch.setenv("SKILL_EVAL_GOAL_ACCURACY_BUDGET", str(goal_budget))

    steps = [{"source": "user", "message": "Write the GPU tests and verify the suite."}]
    for idx in range(7):
        steps.append({
            "source": "agent",
            "tool_calls": [{
                "tool_call_id": f"write-{idx}",
                "function_name": "Write",
                "arguments": {
                    "file_path": f"/workspace/output/test_gpu_{idx}.py",
                    "content": "\n".join(
                        f"def test_gpu_case_{case:03d}(): assert select_engine('gpu') == 'gpu'" for case in range(40)
                    ),
                },
            }],
        })
    failure = "FAILED tests/test_gpu.py::test_gpu_engine - AssertionError: expected gpu, got cpu"
    for idx in range(9):
        result = failure if idx == 8 else f"Verification run {idx + 1}: all GPU tests passed"
        output = "\n".join([
            result,
            "============================= test session starts ==============================",
            "platform linux -- Python 3.12.12, pytest-8.4.0",
            *(f"tests/test_gpu.py::test_gpu_engine[dataset_{case:02d}] PASSED [100%]" for case in range(14)),
            "============================= 14 tests completed ==============================",
        ])
        steps.append({
            "source": "agent",
            "tool_calls": [{
                "tool_call_id": f"verify-{idx}",
                "function_name": "bash",
                "arguments": {"command": "pytest -v tests/test_gpu.py"},
            }],
            "observation": {"results": [{"source_call_id": f"verify-{idx}", "content": output}]},
        })

    module = atif_helpers if implementation == "shared" else _load_harbor_template_module()
    bundles = module.build_metric_evidence_bundles({"steps": steps}, steps[0]["message"])
    bundle = bundles[metric]
    evidence = bundle["prompt_evidence"]
    assert evidence.find(failure) >= 0, "The latest failed verification must survive without a final response"
    assert "FINAL RESPONSE" not in evidence
    assert len(evidence) <= {"accuracy": accuracy_budget, "goal_accuracy": goal_budget}[metric]
    assert bundle["omitted"]["truncated"] is True
    assert bundle["omitted"]["count"] > 0
