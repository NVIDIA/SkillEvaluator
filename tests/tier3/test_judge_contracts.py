# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge reply contracts: verdicts in, scores computed in code, specific retries, saved evidence."""

from __future__ import annotations

import json
import random
import string
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from tests.conftest import load_harbor_eval_template

from skillevaluator.tier3.eval_core import atif_helpers, llm_judge
from skillevaluator.tier3.harbor.metrics import RESERVED_METRIC_NAMES

eval_template = load_harbor_eval_template("harbor_template_eval_judge_contracts")

_CRITERIA = {
    "ACTION_CORRECT": True,
    "FACTUALLY_ACCURATE": False,
    "TASK_ADDRESSED": True,
    "ACTIONABLE": False,
    "FORMAT_FOLLOWED": True,
}
_ACCURACY_RESPONSE = json.dumps({"criteria": _CRITERIA, "reason": "judged from evidence"})
_GOAL_RESPONSE = json.dumps(
    {"user_goal": "complete the task", "end_state": "task completed", "achieved": True, "reason": "met"}
)
_BEHAVIOR_RESPONSE = json.dumps(
    {
        "results": [
            {"step": 1, "passed": True, "reason": "ran the script"},
            {"step": 2, "passed": False, "reason": "no report"},
        ],
        "summary": "one of two",
    }
)


@pytest.fixture(params=["shared", "template"])
def judge_module(request: pytest.FixtureRequest) -> ModuleType:
    return llm_judge if request.param == "shared" else eval_template


def _script(responses: list[tuple[str | None, str | None]], calls: list[dict[str, Any]]):
    def call(prompt: str, **kwargs: Any):
        calls.append({"prompt": prompt, "kwargs": kwargs})
        return responses[min(len(calls) - 1, len(responses) - 1)]

    return call


def _use_goal_script(module: ModuleType, monkeypatch: pytest.MonkeyPatch, responses, calls) -> None:
    if module is llm_judge:
        monkeypatch.setattr(module, "call_public_llm", _script(responses, calls))
        return
    scripted = _script(responses, calls)

    def with_provenance(prompt: str, **kwargs: Any):
        content, error = scripted(prompt, **kwargs)
        return content, error, {"provider": "openai", "model": "judge-model"}

    monkeypatch.setattr(module, "_ragas_goal_accuracy_enabled", lambda: False)
    monkeypatch.setattr(module, "_call_public_llm_with_provenance", with_provenance)


def _prompt_example(prompt: str) -> dict[str, Any]:
    marker = "(the values are only an illustration):\n"
    example, _end = json.JSONDecoder().raw_decode(prompt[prompt.index(marker) + len(marker) :])
    return example


def _capture_prompts(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    prompts: list[str] = []

    def capture(prompt: str, **_kwargs: Any):
        prompts.append(prompt)
        return None, "stop after capturing the prompt"

    monkeypatch.setattr(module, "call_public_llm", capture)
    if module is eval_template:
        monkeypatch.setattr(module, "_ragas_goal_accuracy_enabled", lambda: False)
        monkeypatch.setattr(
            module, "_call_public_llm_with_provenance", lambda prompt, **kwargs: (*capture(prompt, **kwargs), {})
        )
    return prompts


# ── Prompt contracts ─────────────────────────────────────────────────────────


def test_every_judge_prompt_example_passes_its_own_validator(judge_module, monkeypatch) -> None:
    prompts = _capture_prompts(judge_module, monkeypatch)

    judge_module.judge_accuracy("q", "gt", "evidence")
    judge_module.judge_goal_accuracy("q", "gt", "evidence")
    judge_module.judge_behavior_check("conversation", ["first", "second"])

    accuracy, goal, behavior = (_prompt_example(prompt) for prompt in prompts)
    assert judge_module._accuracy_payload_error(accuracy) is None
    assert judge_module._goal_payload_error(goal) is None
    assert judge_module._behavior_payload_error(behavior, 2) is None
    for example in (accuracy, goal, behavior):
        assert "score" not in example


def test_placeholder_only_expected_behavior_is_not_applicable(judge_module, monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("the judge should not run without expected behaviors")

    monkeypatch.setattr(judge_module, "call_public_llm", fail_if_called)
    result = judge_module.judge_behavior_check("conversation", ["None", "  ", "n/A", None])

    assert result["score"] is None
    assert result["reason"] == "N/A: no expected_behavior defined for this eval case"
    assert result["results"] == []


@pytest.mark.parametrize("behavior_count", [1, 2, 3, 5, 12])
def test_behavior_prompt_example_is_a_valid_reply_for_any_number_of_behaviors(
    judge_module, monkeypatch, behavior_count: int
) -> None:
    prompts = _capture_prompts(judge_module, monkeypatch)

    judge_module.judge_behavior_check("conversation", [f"behavior {n}" for n in range(1, behavior_count + 1)])

    example = _prompt_example(prompts[0])
    assert judge_module._behavior_payload_error(example, behavior_count) is None
    assert [entry["step"] for entry in example["results"]] == list(range(1, behavior_count + 1))


def test_accuracy_prompt_gives_no_free_criterion_to_a_response_without_an_answer(judge_module, monkeypatch) -> None:
    # A trial that ran but gave no answer scored 0.0 before FORMAT_FOLLOWED
    # replaced SKILL_IDENTIFIED. Its "no format required" default must not turn
    # into a free point for an empty response.
    prompts = _capture_prompts(judge_module, monkeypatch)

    judge_module.judge_accuracy("Which version of the tool is installed?", "1.2.3", "")

    prompt = prompts[0]
    format_line = next(line for line in prompt.splitlines() if line.startswith("5. FORMAT_FOLLOWED:"))
    assert "Answer true when the question requires no specific format." not in format_line
    assert "answer true if the response gives an answer and false if it gives none" in format_line
    assert "If the evidence holds no answer to the question" in prompt
    assert "every criterion is false" in prompt
    # A correct refusal still scores when the expected answer is a refusal.
    assert "unless the expected answer is itself a refusal" in prompt


@pytest.mark.parametrize(
    ("requirement", "final_response"),
    [
        ("Return only a number.", "83"),
        ("Return only a number.", "82"),
        ("Return only a number.", "83\nFinished processing."),
        ("Return only a number.", "The computed quantity is 83.\nCalculation finished."),
        ("Return only a number.", "Quantity: 83"),
        ("Return only a number.", "```text\n83\n```"),
        ("Explain the quantity.", "The computed quantity is 83."),
        ("Use the prefix COUNT= before the quantity.", "COUNT=83"),
        ("Return the quantity with units and a short explanation.", "83 items remain after counting."),
    ],
    ids=[
        "bare-number",
        "wrong-value-valid-format",
        "valid-first-line-extra-line",
        "prose",
        "label",
        "fence",
        "no-format",
        "required-prefix",
        "required-units-prose",
    ],
)
def test_format_judges_receive_complete_final_response_separate_from_tool_and_reference(
    monkeypatch, requirement: str, final_response: str
) -> None:
    # Catch response extraction, evidence misframing, or drift between deployed
    # prompts. The provider is stopped at its boundary; no semantic verdict is faked.
    question = f"Count the available items. {requirement}"
    trajectory = {
        "steps": [
            {"source": "user", "message": question},
            {
                "source": "agent",
                "tool_calls": [{"tool_call_id": "count-1", "function_name": "count_items", "arguments": {}}],
                "observation": {"results": [{"source_call_id": "count-1", "content": "83"}]},
            },
            {"source": "agent", "message": final_response},
        ]
    }
    captured = []
    for module, builder in (
        (llm_judge, atif_helpers.build_metric_evidence_bundles),
        (eval_template, eval_template.build_metric_evidence_bundles),
    ):
        bundles = builder(trajectory, question, ground_truth="83", expected_behavior=[requirement])
        prompts = _capture_prompts(module, monkeypatch)
        module.judge_accuracy(question, "83", bundles["accuracy"]["prompt_evidence"])
        module.judge_behavior_check(bundles["behavior_check"]["prompt_evidence"], [requirement])
        assert len(prompts) == 2
        accuracy, behavior = prompts
        assert "EXPECTED ANSWER:\n83\n" in accuracy
        assert f"FINAL RESPONSE\n{final_response}\n\nKEY OBSERVATIONS\n83" in accuracy
        assert f"FINAL RESPONSE\n{final_response}\n\nUSER REQUEST\n{question}" in behavior
        assert "COMPACT TOOL HISTORY\n" in behavior
        assert "count_items" in behavior
        assert "83" in behavior.split("COMPACT TOOL HISTORY\n", 1)[1]
        assert f"EXPECTED BEHAVIORS:\n1. {requirement}" in behavior
        for prompt in prompts:
            assert "judge the entire final response literally" in prompt
            assert "independently of factual correctness" in prompt
            assert "Do not extract a compliant answer" in prompt
            assert "extra prose, a label, or a code fence violates that requirement" in prompt
            assert "Required prefixes, units, or prose are valid when they match the stated format" in prompt
            assert "Inspect every line of the final response, including text after an initially valid answer" in prompt
            assert "Line breaks do not end the final response" in prompt
            assert "Do not mentally remove labels or Markdown fence characters" in prompt
        assert "Wording leniency does not relax explicit format constraints" in accuracy
        assert "When the complete final response establishes a format violation, mark that behavior false" in behavior
        assert "keep null when missing or incomplete evidence prevents checking it" in behavior
        captured.append(prompts)
    assert captured[0] == captured[1]


def test_format_only_behavior_is_separate_from_a_stated_correctness_behavior(monkeypatch) -> None:
    # The same wrong value can satisfy structure but violate a separate factual
    # requirement. Capture the actual evidence/prompt, without faking a verdict.
    question = "Count the stored entries. Return only a number."
    behaviors = ["Return only a number.", "Return the correct count."]
    trajectory = {
        "steps": [
            {"source": "user", "message": question},
            {"source": "agent", "message": "82"},
        ]
    }
    captured = []
    for module, builder in (
        (llm_judge, atif_helpers.build_metric_evidence_bundles),
        (eval_template, eval_template.build_metric_evidence_bundles),
    ):
        bundles = builder(trajectory, question, ground_truth="83", expected_behavior=behaviors)
        prompts = _capture_prompts(module, monkeypatch)
        module.judge_behavior_check(bundles["behavior_check"]["prompt_evidence"], behaviors)
        prompt = prompts[0]
        assert f"FINAL RESPONSE\n82\n\nUSER REQUEST\n{question}" in prompt
        assert "EXPECTED BEHAVIORS:\n1. Return only a number.\n2. Return the correct count." in prompt
        assert "For each numbered behavior, check only what that behavior requires" in prompt
        assert "A format-only behavior does not require the answer's value to be factually correct" in prompt
        assert "If a behavior also requires factual correctness, evaluate that requirement too" in prompt
        assert "ensure each passed value agrees with its reason" in prompt
        assert "a reason establishing a violation requires false, not true" in prompt
        assert "an evidence gap that prevents checking requires null" in prompt
        captured.append(prompts)
    assert captured[0] == captured[1]


def test_format_judges_keep_direct_unsectioned_inputs_compatible(judge_module, monkeypatch) -> None:
    prompts = _capture_prompts(judge_module, monkeypatch)
    direct_response = "COUNT=83"

    judge_module.judge_accuracy("Use the prefix COUNT=.", "83", direct_response)
    judge_module.judge_behavior_check(direct_response, ["Use the prefix COUNT=."])

    assert prompts[0].endswith(
        "SELECTED EVIDENCE (final response + produced artifacts; low-relevance steps may be omitted):\nCOUNT=83"
    )
    assert "CONVERSATION:\nCOUNT=83\n\nEXPECTED BEHAVIORS:" in prompts[1]
    for prompt in prompts:
        assert "When a FINAL RESPONSE section is supplied" in prompt
        assert "its contents are the final answer" in prompt
        assert "context, not part of that answer" in prompt


def test_goal_prompt_omits_empty_tool_section_and_keeps_real_one(judge_module, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    _use_goal_script(judge_module, monkeypatch, [(_GOAL_RESPONSE, None)], calls)

    judge_module.judge_goal_accuracy("question", "ground truth", "agent evidence", tool_summary="")
    judge_module.judge_goal_accuracy("question", "ground truth", "agent evidence", tool_summary="Bash: make test")

    assert "AGENT'S TOOL CALLS" not in calls[0]["prompt"]
    assert "AGENT'S TOOL CALLS:\nBash: make test" in calls[1]["prompt"]


def test_shared_and_template_judges_use_the_same_schemas() -> None:
    assert eval_template.ACCURACY_JSON_SCHEMA == llm_judge.ACCURACY_JSON_SCHEMA
    assert eval_template.GOAL_ACCURACY_JSON_SCHEMA == llm_judge.GOAL_ACCURACY_JSON_SCHEMA
    assert eval_template.BEHAVIOR_CHECK_JSON_SCHEMA == llm_judge.BEHAVIOR_CHECK_JSON_SCHEMA
    for schema in (
        llm_judge.ACCURACY_JSON_SCHEMA,
        llm_judge.GOAL_ACCURACY_JSON_SCHEMA,
        llm_judge.BEHAVIOR_CHECK_JSON_SCHEMA,
    ):
        assert "score" not in schema["properties"]
        assert "score" not in schema["required"]


# ── Scores come from verdicts ────────────────────────────────────────────────


def test_goal_score_follows_achieved_not_a_contradicting_score(judge_module, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    reply = json.dumps({"achieved": False, "score": 1.0, "reason": "the expected file is missing"})
    _use_goal_script(judge_module, monkeypatch, [(reply, None)], calls)

    result = judge_module.judge_goal_accuracy("question", "ground truth", "agent evidence")

    assert result["score"] == 0.0
    assert result["achieved"] is False
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("results", "problem"),
    [
        pytest.param(
            [{"step": 1, "passed": True}, {"step": 1, "passed": True}],
            "steps judged more than once: [1]; steps not judged: [2]",
            id="duplicate-step",
        ),
        pytest.param(
            [{"step": 1, "passed": True}, {"step": 3, "passed": True}],
            "steps not judged: [2]; steps outside 1..2: [3]",
            id="out-of-range-step",
        ),
        pytest.param(
            [{"passed": True}, {"passed": True}],
            'every "results" entry needs an integer "step" number',
            id="missing-step",
        ),
        pytest.param(
            [{"step": True, "passed": True}, {"step": 2, "passed": True}],
            'every "results" entry needs an integer "step" number',
            id="boolean-step",
        ),
        pytest.param(
            [{"step": 1, "passed": True}],
            '"results" has 1 entries; it needs exactly 2, one per behavior',
            id="too-few",
        ),
    ],
)
def test_behavior_reply_must_judge_each_step_exactly_once(
    judge_module, monkeypatch, results: list[dict[str, Any]], problem: str
) -> None:
    calls: list[dict[str, Any]] = []
    invalid = json.dumps({"results": results, "summary": "all observed"})
    monkeypatch.setattr(judge_module, "call_public_llm", _script([(invalid, None), (invalid, None)], calls))

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2"])

    assert result["score"] is None
    assert result["status"] == "error"
    assert len(calls) == 2
    assert problem in calls[1]["prompt"]
    assert problem in result["reason"]


def test_behavior_salvage_rejects_repeated_steps(judge_module, monkeypatch) -> None:
    truncated = '{"results":[{"step":1,"passed":true},{"step":1,"passed":true},{"step":2,"passed":tr'
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(judge_module, "call_public_llm", _script([(truncated, None), (truncated, None)], calls))

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2", "b3"])

    assert result["score"] is None
    assert result["status"] == "error"


# ── Incomplete evidence gate (issue #234) ────────────────────────────────────


def _never_called(*args: Any, **kwargs: Any):
    raise AssertionError("the judge must not call a model on incomplete evidence")


def _behavior_bundle(
    module: ModuleType,
    traj: dict[str, Any],
    expected_behavior: list[str] | None = None,
) -> dict[str, Any]:
    """Build the behavior_check bundle with the module the judge belongs to."""
    builder = atif_helpers if module is llm_judge else module
    return builder.build_metric_evidence_bundles(
        traj, "Explain the migration.", expected_behavior=expected_behavior
    )["behavior_check"]


_VERIFIED_PATH = "/workspace/output/report.md"
_VERIFIED_BEHAVIOR = f"writes the file `{_VERIFIED_PATH}`"


def _traj_with_verified_write_and_long_answer(answer_chars: int = 9000) -> dict[str, Any]:
    """A run whose one expected behavior is provable from the write alone.

    The final answer is long enough that the final-response limit cuts its
    middle, which is the condition the evidence gate fires on.
    """
    return {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "write-1",
                        "function_name": "Write",
                        "arguments": {"file_path": _VERIFIED_PATH, "content": "# Report\nAll good.\n"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "write-1", "content": "File written"}]},
            },
            {"source": "agent", "message": _distinct_answer(answer_chars)},
        ]
    }


def _distinct_answer(length: int) -> str:
    """Build non-repetitive text, so a cut middle cannot be matched elsewhere."""
    rng = random.Random(20234)
    words: list[str] = []
    while sum(len(word) + 1 for word in words) < length:
        words.append("".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9))))
    return " ".join(words)[:length]


def test_incomplete_required_behavior_evidence_errors_without_a_model_call(judge_module, monkeypatch) -> None:
    # A behavior score read off a shortened final answer grades the evidence
    # loss, not the agent, so the judge returns an evaluation error instead.
    monkeypatch.setattr(judge_module, "call_public_llm", _never_called)
    omitted = {
        "count": 1,
        "truncated": True,
        "reason": "final response shortened to fit SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT",
        "required_complete": False,
        "required_missing": ["final response"],
        "incomplete_sections": ["final response"],
    }

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2"], evidence_omitted=omitted)

    assert result["score"] is None
    assert result["status"] == "error"
    assert "final response" in result["reason"]
    assert "SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT" in result["reason"]
    assert result["results"] == []
    assert [item["status"] for item in result["expectations"]] == ["not_checked", "not_checked"]


def test_complete_required_behavior_evidence_is_judged_normally(judge_module, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(judge_module, "call_public_llm", _script([(_BEHAVIOR_RESPONSE, None)], calls))
    omitted = {
        "count": 1,
        "truncated": True,
        "reason": "behavior evidence shortened to fit SKILL_EVAL_BEHAVIOR_CHECK_BUDGET",
        "required_complete": True,
        "required_missing": [],
        "incomplete_sections": ["compact tool history"],
    }

    result = judge_module.judge_behavior_check("conversation", ["b1", "b2"], evidence_omitted=omitted)

    # Optional exploration output dropped to fit the budget is ordinary; only
    # missing required evidence withholds the score.
    assert result["score"] == 0.5
    assert len(calls) == 1


def test_behavior_judge_refuses_to_cut_accounted_evidence_a_second_time(judge_module, monkeypatch) -> None:
    monkeypatch.setattr(judge_module, "call_public_llm", _never_called)
    oversized = "A" * (judge_module._behavior_check_budget() + 5000)
    omitted = {"count": 0, "truncated": False, "reason": "", "required_complete": True, "required_missing": []}

    result = judge_module.judge_behavior_check(oversized, ["b1"], evidence_omitted=omitted)

    assert result["score"] is None
    assert result["status"] == "error"
    assert "shortened again inside the judge" in result["reason"]


def test_oversized_final_answer_reaches_the_judge_gate_as_an_error(judge_module, monkeypatch) -> None:
    # End to end over the seam the verifier uses: bundle -> judge. No model is
    # called and no behavior score is produced.
    monkeypatch.setattr(judge_module, "call_public_llm", _never_called)
    marker = "MIGRATE_UNIQUE_COVERAGE_FIRST"
    answer = "A" * 8000 + marker + "B" * 8000
    bundle = _behavior_bundle(judge_module, {"steps": [{"source": "agent", "message": answer}]})

    assert marker not in bundle["prompt_evidence"]
    assert bundle["omitted"]["required_complete"] is False

    result = judge_module.judge_behavior_check(
        bundle["prompt_evidence"], ["explains the migration"], evidence_omitted=bundle["omitted"]
    )

    assert result["score"] is None
    assert result["status"] == "error"


def test_cut_answer_still_scores_when_verified_facts_settle_every_behavior(judge_module, monkeypatch) -> None:
    # The gate exists to stop a score being read off evidence the limits cut.
    # It must not discard a score the surviving evidence supports: here the one
    # expected behavior is a path the deterministic fact list already marks
    # OBSERVED, so the missing middle of the answer cannot change the answer.
    calls: list[dict[str, Any]] = []
    response = json.dumps({"results": [{"step": 1, "passed": True, "reason": "wrote the report"}]})
    monkeypatch.setattr(judge_module, "call_public_llm", _script([(response, None)], calls))

    bundle = _behavior_bundle(judge_module, _traj_with_verified_write_and_long_answer(), [_VERIFIED_BEHAVIOR])

    # The evidence really is incomplete — the gate's own precondition holds.
    assert bundle["omitted"]["required_complete"] is False
    assert bundle["omitted"]["required_missing"] == ["final response"]
    # And the deterministic facts settle the expectation regardless.
    assert bundle["omitted"]["expected_behaviors_verified"] is True
    assert f"[OBSERVED step 0] {_VERIFIED_PATH}" in bundle["prompt_evidence"]

    result = judge_module.judge_behavior_check(
        bundle["prompt_evidence"], [_VERIFIED_BEHAVIOR], evidence_omitted=bundle["omitted"]
    )

    assert len(calls) == 1
    assert result.get("status") != "error"
    assert result["score"] == 1.0

    # And the case keeps an overall score, instead of dropping out of every
    # gate and comparison keyed on one.
    from skillevaluator.tier3.harbor.metrics import overall_score_from_metrics

    scores = {
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": 1.0,
        "goal_accuracy": 1.0,
        "behavior_check": result["score"],
    }
    assert overall_score_from_metrics(scores) == 1.0


def test_cut_answer_is_still_gated_when_any_behavior_needs_the_answer(judge_module, monkeypatch) -> None:
    # A prose expectation names nothing checkable, so it can only be judged
    # from the text the limits cut. One such expectation keeps the gate, even
    # beside a behavior the facts do settle.
    monkeypatch.setattr(judge_module, "call_public_llm", _never_called)
    traj = _traj_with_verified_write_and_long_answer()

    for expectations in (
        ["explains the migration trade-offs"],
        [_VERIFIED_BEHAVIOR, "explains the migration trade-offs"],
    ):
        bundle = _behavior_bundle(judge_module, traj, expectations)
        assert bundle["omitted"]["required_complete"] is False
        assert bundle["omitted"]["expected_behaviors_verified"] is False

        result = judge_module.judge_behavior_check(
            bundle["prompt_evidence"], expectations, evidence_omitted=bundle["omitted"]
        )

        assert result["score"] is None
        assert result["status"] == "error"


def test_behavior_evidence_gap_treats_an_absent_verified_flag_as_unknown(judge_module, monkeypatch) -> None:
    # Records written before the flag existed, and records assembled by hand,
    # must keep the gate rather than skip it by default.
    monkeypatch.setattr(judge_module, "call_public_llm", _never_called)
    omitted = {"required_complete": False, "required_missing": ["final response"], "reason": "cut"}

    assert judge_module.behavior_evidence_gap(omitted)
    assert judge_module.behavior_evidence_gap({**omitted, "expected_behaviors_verified": False})
    assert judge_module.behavior_evidence_gap({**omitted, "expected_behaviors_verified": True}) == ""

    result = judge_module.judge_behavior_check("conversation", ["b1"], evidence_omitted=omitted)
    assert result["score"] is None
    assert result["status"] == "error"


def test_gated_behavior_result_never_becomes_a_zero_downstream(judge_module, monkeypatch) -> None:
    from skillevaluator.tier3.eval_core.declarative_checks import combine_behavior_check
    from skillevaluator.tier3.harbor.metrics import overall_score_from_metrics

    monkeypatch.setattr(judge_module, "call_public_llm", _never_called)
    omitted = {"required_complete": False, "required_missing": ["final response"], "reason": "cut"}
    gated = judge_module.judge_behavior_check("conversation", ["b1"], evidence_omitted=omitted)

    # The verifier's own normalization keeps it scoreless, even where an N/A
    # result would be accepted.
    normalized = eval_template._normalize_required_judge_result("behavior_check", gated, allow_not_applicable=True)
    assert normalized["score"] is None
    assert normalized["status"] == "error"

    # Direct declarative checks beside the judge must not fill the gap either.
    combined = combine_behavior_check(gated, [{"step": 1, "passed": True}], 1)
    assert combined["score"] is None
    assert combined["status"] == "error"

    # And a missing behavior score leaves the overall score undefined rather
    # than averaging a zero into the dimension mean (see PR #240).
    scores = {
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": 1.0,
        "goal_accuracy": 1.0,
        "behavior_check": None,
    }
    assert overall_score_from_metrics(scores) is None


# ── Specific retries ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("metric", "invalid", "problem"),
    [
        pytest.param("accuracy", "not-json", "the reply did not contain one JSON object", id="accuracy-malformed"),
        pytest.param("accuracy", '{"score": 0.7}', 'the reply has no "criteria" object', id="accuracy-score-only"),
        pytest.param(
            "accuracy",
            json.dumps({"criteria": {**_CRITERIA, "SKILL_IDENTIFIED": True}}),
            "criteria has unexpected keys SKILL_IDENTIFIED",
            id="accuracy-old-criterion",
        ),
        pytest.param(
            "accuracy",
            json.dumps({"criteria": {key: value for key, value in _CRITERIA.items() if key != "ACTIONABLE"}}),
            "criteria is missing ACTIONABLE",
            id="accuracy-missing-criterion",
        ),
        pytest.param("goal", '{"achieved": "true"}', '"achieved" must be true or false', id="goal-string-verdict"),
        pytest.param("goal", '{"score": 1.0}', '"achieved" must be true or false', id="goal-score-only"),
        pytest.param("goal", "", "the reply was empty", id="goal-empty"),
    ],
)
def test_invalid_reply_is_retried_once_with_the_specific_problem(
    judge_module, monkeypatch, metric: str, invalid: str, problem: str
) -> None:
    calls: list[dict[str, Any]] = []
    valid = _ACCURACY_RESPONSE if metric == "accuracy" else _GOAL_RESPONSE
    if metric == "accuracy":
        monkeypatch.setattr(judge_module, "call_public_llm", _script([(invalid, None), (valid, None)], calls))
        result = judge_module.judge_accuracy("question", "ground truth", "agent evidence")
    else:
        _use_goal_script(judge_module, monkeypatch, [(invalid, None), (valid, None)], calls)
        result = judge_module.judge_goal_accuracy("question", "ground truth", "agent evidence")

    assert result["score"] == (0.6 if metric == "accuracy" else 1.0)
    assert len(calls) == 2
    assert "previous reply could not be parsed or validated" not in calls[0]["prompt"]
    assert f"previous reply could not be parsed or validated: {problem}" in calls[1]["prompt"]


@pytest.mark.parametrize("applicable", [True, False])
def test_deprecated_skill_identification_keyword_is_accepted_and_ignored(
    judge_module, monkeypatch, applicable: bool
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(judge_module, "call_public_llm", _script([(_ACCURACY_RESPONSE, None)], calls))

    plain = judge_module.judge_accuracy("question", "ground truth", "agent evidence")
    legacy = judge_module.judge_accuracy(
        "question", "ground truth", "agent evidence", skill_identification_applicable=applicable
    )

    assert legacy == plain
    assert legacy["score"] == 0.6
    assert set(legacy["criteria"]) == set(_CRITERIA)
    assert "excluded_criteria" not in legacy
    assert calls[1]["prompt"] == calls[0]["prompt"]
    assert "skill_identification_applicable" not in calls[1]["kwargs"]


def test_required_judge_accepts_the_deprecated_accuracy_keyword(monkeypatch) -> None:
    monkeypatch.setattr(eval_template, "call_public_llm", _script([(_ACCURACY_RESPONSE, None)], []))

    result = eval_template._call_required_judge(
        "accuracy",
        eval_template.judge_accuracy,
        "question",
        "ground truth",
        "agent evidence",
        skill_identification_applicable=False,
    )

    assert result["score"] == 0.6
    assert result.get("status") != "error"


# ── Saved judge evidence and the reward contract ─────────────────────────────


def _stage_verifier(tmp_path: Path, entry: dict[str, Any] | None = None) -> ModuleType:
    module = load_harbor_eval_template(f"harbor_eval_judge_contracts_{tmp_path.name}")
    agent_dir = tmp_path / "logs" / "agent"
    verifier_dir = tmp_path / "logs" / "verifier"
    tests_dir = tmp_path / "tests"
    for directory in (agent_dir, verifier_dir, tests_dir):
        directory.mkdir(parents=True)
    module.VERIFIER_DIR = verifier_dir
    module.ATIF_PATH = agent_dir / "trajectory.json"
    module.ENTRY_PATH = tests_dir / "entry.json"
    module.REWARD_JSON = verifier_dir / "reward.json"
    module.REWARD_TXT = verifier_dir / "reward.txt"
    module.SKILL_EVALUATOR_REWARD_JSON = verifier_dir / "skill_evaluator_reward.json"
    module.ATIF_PATH.write_text(
        json.dumps({"steps": [{"source": "agent", "message": "I completed the task."}]}), encoding="utf-8"
    )
    module.ENTRY_PATH.write_text(
        json.dumps(
            entry
            or {
                "id": "case-1",
                "has_skill": False,
                "question": "Complete the task",
                "ground_truth": "Done",
                "expected_behavior": ["ran the script", "reported the result"],
            }
        ),
        encoding="utf-8",
    )
    module._ragas_goal_accuracy_enabled = lambda: False
    return module


def _routed(calls: list[str], **replies: str):
    """Answer each judge by its prompt, so a full verifier run can use one fake."""

    def call(prompt: str, **_kwargs: Any):
        calls.append(prompt)
        if "EXPECTED BEHAVIORS" in prompt:
            reply = replies["behavior"]
        elif "FORMAT_FOLLOWED" in prompt:
            reply = replies["accuracy"]
        else:
            reply = replies["goal"]
        return reply, None

    return call


def _attempts(verifier_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((verifier_dir / "judge_attempts.json").read_text(encoding="utf-8"))
    assert payload["judge_contract"] == eval_template.JUDGE_CONTRACT
    return payload["attempts"]


def test_checks_only_behavior_does_not_call_the_judge(tmp_path: Path) -> None:
    verifier = _stage_verifier(
        tmp_path,
        {
            "id": "case-1",
            "has_skill": False,
            "question": "Complete the task",
            "ground_truth": "",
            "expected_behavior": [],
            "checks": [{"type": "output_contains", "text": "completed"}],
        },
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("checks-only behavior must not invoke the judge")

    verifier.judge_behavior_check = fail_if_called
    verifier.call_public_llm = fail_if_called
    verifier.main()

    reward = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    behavior = reward["details"]["behavior_check"]
    assert reward["behavior_check"] == 1.0
    assert behavior["declarative_checks"]["summary"] == "Checked directly: 1, judged: 0, not checked: 0"


def test_scored_trial_records_judge_contract_and_every_attempt(tmp_path: Path) -> None:
    verifier = _stage_verifier(tmp_path)
    calls: list[str] = []
    route = _routed(calls, accuracy=_ACCURACY_RESPONSE, goal=_GOAL_RESPONSE, behavior=_BEHAVIOR_RESPONSE)
    verifier.call_public_llm = route
    verifier._call_public_llm_with_provenance = lambda prompt, **kwargs: (
        *route(prompt, **kwargs),
        {"provider": "openai", "model": "goal-model"},
    )

    verifier.main()

    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    numeric = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    assert rich["judge_contract"] == "skill-evaluator-judges-v2"
    assert "judge_contract" not in numeric
    assert rich["accuracy"] == numeric["accuracy"] == 0.6
    assert rich["goal_accuracy"] == numeric["goal_accuracy"] == 1.0
    assert rich["behavior_check"] == numeric["behavior_check"] == 0.5
    attempts = _attempts(verifier.VERIFIER_DIR)
    assert [(attempt["metric"], attempt["status"]) for attempt in attempts] == [
        ("accuracy", "accepted"),
        ("goal_accuracy", "accepted"),
        ("behavior_check", "accepted"),
    ]
    assert attempts[1]["model"] == "goal-model"
    assert attempts[0]["reply"] == _ACCURACY_RESPONSE
    assert len(attempts[0]["prompt_sha256"]) == 64
    assert "Complete the task" in attempts[0]["prompt"]


def test_trial_without_judge_references_writes_empty_attempt_artifact(tmp_path: Path) -> None:
    verifier = _stage_verifier(
        tmp_path,
        {"id": "case-1", "question": "Inspect the output.", "ground_truth": "", "expected_behavior": []},
    )
    verifier.call_public_llm = lambda *_args, **_kwargs: pytest.fail("no judge reference")
    verifier._call_public_llm_with_provenance = lambda *_args, **_kwargs: pytest.fail("no judge reference")

    verifier.main()

    assert _attempts(verifier.VERIFIER_DIR) == []


def test_failed_judge_keeps_its_rejected_attempts_without_a_score(tmp_path: Path) -> None:
    verifier = _stage_verifier(tmp_path)
    calls: list[str] = []
    route = _routed(calls, accuracy='{"score": 1.0}', goal=_GOAL_RESPONSE, behavior=_BEHAVIOR_RESPONSE)
    verifier.call_public_llm = route
    verifier._call_public_llm_with_provenance = lambda prompt, **kwargs: (*route(prompt, **kwargs), {})

    with pytest.raises(SystemExit):
        verifier.main()

    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    assert rich["accuracy"] is None
    assert 'the reply has no "criteria" object' in rich["evaluation_errors"]["accuracy"]
    rejected = [attempt for attempt in _attempts(verifier.VERIFIER_DIR) if attempt["metric"] == "accuracy"]
    assert [attempt["status"] for attempt in rejected] == ["rejected", "rejected"]
    assert rejected[1]["problem"] == 'the reply has no "criteria" object'
    assert "previous reply could not be parsed or validated" not in rejected[0]["prompt"]
    assert "previous reply could not be parsed or validated" in rejected[1]["prompt"]


def test_provider_error_is_saved_as_an_error_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = _stage_verifier(tmp_path)
    credential = "dummy-evidence-secret-DO-NOT-RETAIN"
    monkeypatch.setenv("OPENAI_API_KEY", credential)
    calls: list[str] = []
    route = _routed(calls, accuracy=_ACCURACY_RESPONSE, goal=_GOAL_RESPONSE, behavior=_BEHAVIOR_RESPONSE)
    verifier.call_public_llm = lambda prompt, **kwargs: (
        (None, f"HTTP 503 echoed {credential}") if "FORMAT_FOLLOWED" in prompt else route(prompt, **kwargs)
    )
    verifier._call_public_llm_with_provenance = lambda prompt, **kwargs: (*route(prompt, **kwargs), {})

    with pytest.raises(SystemExit):
        verifier.main()

    (attempt,) = [attempt for attempt in _attempts(verifier.VERIFIER_DIR) if attempt["metric"] == "accuracy"]
    assert attempt["status"] == "error"
    assert "HTTP 503" in attempt["problem"]
    assert "reply" not in attempt
    assert credential not in (verifier.VERIFIER_DIR / "judge_attempts.json").read_text(encoding="utf-8")


def test_saved_judge_evidence_masks_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configured = "dummy-configured-judge-credential"
    monkeypatch.setenv("ANTHROPIC_API_KEY", configured)
    leaked_key = "sk-" + "a" * 24
    userinfo = "dummy-userinfo-credential"
    verifier = _stage_verifier(
        tmp_path,
        {
            "id": "case-1",
            "question": f"Use {configured} and {leaked_key} to clone https://ci:{userinfo}@git.example.com/r.git",
            "ground_truth": "Done",
        },
    )
    calls: list[str] = []
    accuracy_reply = json.dumps({"criteria": _CRITERIA, "reason": f"saw {configured}"})
    route = _routed(calls, accuracy=accuracy_reply, goal=_GOAL_RESPONSE, behavior=_BEHAVIOR_RESPONSE)
    verifier.call_public_llm = route
    verifier._call_public_llm_with_provenance = lambda prompt, **kwargs: (*route(prompt, **kwargs), {})

    verifier.main()

    saved = (verifier.VERIFIER_DIR / "judge_attempts.json").read_text(encoding="utf-8")
    assert configured not in saved
    assert leaked_key not in saved
    assert userinfo not in saved
    assert "Done" in saved


# The curl examples use repeated-letter credentials, which stay below the
# entropy thresholds of Gitleaks' curl rules.
@pytest.mark.parametrize(
    ("text", "secret", "kept"),
    [
        pytest.param(
            "git clone https://deploy:url-userinfo-secret@git.example.com/repo.git",
            "url-userinfo-secret",
            "https://<redacted>@git.example.com/repo.git",
            id="url-userinfo",
        ),
        pytest.param(
            "curl -H 'Authorization: Bearer tttttttttttt' https://api.example.com",
            "tttttttttttt",
            "Authorization: Bearer <redacted>'",
            id="authorization-header",
        ),
        pytest.param(
            '{"headers": {"Authorization": "Basic b3BhcXVlLWJhc2ljLXNlY3JldA"}}',
            "b3BhcXVlLWJhc2ljLXNlY3JldA",
            '"Authorization": "Basic <redacted>"',
            id="authorization-json",
        ),
        pytest.param(
            "deploy --token flag-token-secret --dry-run", "flag-token-secret", "--token <redacted>", id="token"
        ),
        pytest.param("login --password=flag-password-secret", "flag-password-secret", "--password=<redacted>", id="pw"),
        pytest.param("curl -u uuuu:pppppppppppp https://x", "pppppppppppp", "-u <redacted>", id="user-pass"),
        pytest.param(
            "curl -H 'X-API-Key: kkkkkkkkkkkk' https://x",
            "kkkkkkkkkkkk",
            "X-API-Key: <redacted>'",
            id="api-key-header",
        ),
        pytest.param(
            '{"headers": {"x-api-key": "api-key-json-secret"}}',
            "api-key-json-secret",
            '"x-api-key": "<redacted>"',
            id="api-key-json",
        ),
        pytest.param(
            "curl -H 'PRIVATE-TOKEN: private-token-header-secret' https://x",
            "private-token-header-secret",
            "PRIVATE-TOKEN: <redacted>'",
            id="private-token-header",
        ),
        pytest.param(
            "curl -H 'Cookie: theme=dark; session=cookie-session-secret' https://x",
            "cookie-session-secret",
            "Cookie: <redacted>'",
            id="cookie-every-pair",
        ),
        pytest.param(
            'curl -H "Authorization: ApiKey aaaaaaaaaaaaaaaaaaaaaaaa" https://es.example',
            "aaaaaaaaaaaaaaaaaaaaaaaa",
            '"Authorization: <redacted>" https://es.example',
            id="authorization-apikey-scheme",
        ),
        pytest.param(
            "curl -H 'Authorization: Bot bbbbbbbbbbbbbbbbbbbbbbbb' https://x",
            "bbbbbbbbbbbbbbbbbbbbbbbb",
            "'Authorization: <redacted>' https://x",
            id="authorization-bot-scheme",
        ),
        pytest.param(
            "curl -H 'Authorization: SSWS ssssssssssssssssssssssss'",
            "ssssssssssssssssssssssss",
            "'Authorization: <redacted>'",
            id="authorization-ssws-scheme",
        ),
        pytest.param(
            'headers["Authorization"] = "Bearer subscript-bearer-secret"',
            "subscript-bearer-secret",
            'headers["Authorization"] = "Bearer <redacted>"',
            id="authorization-subscript",
        ),
        pytest.param(
            "headers['Authorization'] = f'Token fstring-token-secret'",
            "fstring-token-secret",
            "headers['Authorization'] = f'Token <redacted>'",
            id="authorization-subscript-fstring",
        ),
        pytest.param(
            'headers["X-API-Key"] = "subscript-api-key-secret"',
            "subscript-api-key-secret",
            'headers["X-API-Key"] = "<redacted>"',
            id="api-key-subscript",
        ),
        pytest.param(
            "headers['Cookie'] = 'session=subscript-cookie-secret'",
            "subscript-cookie-secret",
            "headers['Cookie'] = '<redacted>'",
            id="cookie-subscript",
        ),
        pytest.param(
            "curl -H 'Authorization: Bearer \"quoted-bearer-secret\"' https://x",
            "quoted-bearer-secret",
            'Authorization: Bearer "<redacted>"',
            id="authorization-quoted-token",
        ),
        pytest.param(
            """curl -H 'Authorization: Digest username="bob", response="digest-response-secret"' https://x""",
            "digest-response-secret",
            "Authorization: Digest <redacted>' https://x",
            id="authorization-digest",
        ),
        pytest.param(
            'curl -H "Authorization: Bearer ' + "L0ngOpaqueT0ken" * 60 + '" https://x',
            "L0ngOpaqueT0ken" * 3,
            'Authorization: Bearer <redacted>" https://x',
            id="authorization-value-over-256-chars",
        ),
        pytest.param(
            'curl -u "admin:quoted-user-password" https://x',
            "quoted-user-password",
            "curl -u <redacted> https://x",
            id="user-flag-double-quoted",
        ),
        pytest.param(
            "curl --user='admin:quoted-user-password' https://x",
            "quoted-user-password",
            "--user=<redacted> https://x",
            id="user-flag-single-quoted",
        ),
        pytest.param(
            "mysql --password='quoted-db-password' db",
            "quoted-db-password",
            "mysql --password=<redacted> db",
            id="password-flag-single-quoted",
        ),
        pytest.param(
            "tool --token 'quoted-token-value'",
            "quoted-token-value",
            "tool --token <redacted>",
            id="token-flag-quoted",
        ),
        pytest.param(
            'mysql --password="' + "p" * 300 + '" db',
            "p" * 300,
            "mysql --password=<redacted> db",
            id="password-flag-double-quoted-over-256-chars",
        ),
        pytest.param(
            "tool --token '" + "t" * 300 + "'",
            "t" * 300,
            "tool --token <redacted>",
            id="token-flag-single-quoted-over-256-chars",
        ),
        pytest.param(
            'curl -u "admin:' + "x" * 300 + '" https://x',
            "x" * 300,
            "curl -u <redacted> https://x",
            id="user-flag-double-quoted-over-256-chars",
        ),
        pytest.param(
            'tool --api-key="quoted-api-key-value" run',
            "quoted-api-key-value",
            "tool --api-key=<redacted> run",
            id="api-key-flag-double-quoted",
        ),
    ],
)
def test_judge_evidence_masks_credentials_without_a_key_shape(text: str, secret: str, kept: str) -> None:
    redacted = eval_template._redact_judge_evidence(text)

    assert secret not in redacted
    assert kept in redacted


# Behavior and goal prompts render tool-call arguments with json.dumps, so the
# quotes around header names and values reach the judge prompt escaped.
@pytest.mark.parametrize(
    ("command", "secret"),
    [
        pytest.param(
            """python -c 'requests.get(u, headers={"Authorization": "ApiKey escaped-dict-apikey-secret"})'""",
            "escaped-dict-apikey-secret",
            id="authorization-dict",
        ),
        pytest.param(
            """python -c 'h={}; h["Authorization"] = "Bearer escaped-subscript-bearer-secret"'""",
            "escaped-subscript-bearer-secret",
            id="authorization-subscript",
        ),
        pytest.param(
            """python -c 'h={}; h["X-API-Key"]="escaped-subscript-api-key-secret"'""",
            "escaped-subscript-api-key-secret",
            id="api-key-subscript",
        ),
        pytest.param(
            """python -c 'h={"Cookie": "session=escaped-cookie-secret"}'""",
            "escaped-cookie-secret",
            id="cookie-dict",
        ),
        pytest.param(
            """curl -H "Authorization: Digest username=\\"bob\\", response=\\"escaped-digest-secret\\"" https://x""",
            "escaped-digest-secret",
            id="authorization-digest",
        ),
    ],
)
def test_judge_evidence_masks_json_escaped_tool_call_headers(command: str, secret: str) -> None:
    trajectory = {
        "steps": [
            {"source": "agent", "tool_calls": [{"function_name": "Bash", "arguments": {"command": command}}]},
        ]
    }
    summary = eval_template.build_conversation_summary(trajectory, "Fetch the index.")
    assert secret in summary
    assert '\\"' in summary

    redacted = eval_template._redact_judge_evidence(summary)

    assert secret not in redacted
    assert "<redacted>" in redacted


@pytest.mark.parametrize(
    "text",
    [
        "python -u script.py && pip install -U pkg && pip install --user pkg && git push -u origin main",
        "judge --max-tokens 100 --token-limit 5",
        "see https://example.com/docs?a=b@c and mailto:user@example.com",
        "Authorization is required for this endpoint.",
        "curl --cookie-jar cookies.txt https://x; see the api-key rotation docs",
    ],
)
def test_judge_evidence_keeps_ordinary_text(text: str) -> None:
    assert eval_template._redact_judge_evidence(text) == text


def test_salvaged_behavior_score_is_saved_as_a_salvage_attempt(tmp_path: Path) -> None:
    verifier = _stage_verifier(tmp_path)
    truncated = '{"results":[{"step":1,"passed":true,"reason":"ran the script"},{"step":2,"passed":tr'
    calls: list[str] = []
    route = _routed(calls, accuracy=_ACCURACY_RESPONSE, goal=_GOAL_RESPONSE, behavior=truncated)
    verifier.call_public_llm = route
    verifier._call_public_llm_with_provenance = lambda prompt, **kwargs: (*route(prompt, **kwargs), {})

    verifier.main()

    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    assert rich["behavior_check"] == 0.5
    assert "Salvaged 1/2" in rich["details"]["behavior_check"]["reason"]
    attempts = [attempt for attempt in _attempts(verifier.VERIFIER_DIR) if attempt["metric"] == "behavior_check"]
    assert [(attempt["attempt"], attempt["method"], attempt["status"]) for attempt in attempts] == [
        (1, "llm", "rejected"),
        (2, "llm", "rejected"),
        (2, "salvage", "accepted"),
    ]
    salvaged = attempts[2]
    assert "prompt" not in salvaged
    assert salvaged["problem"] is None
    assert salvaged["model"] == attempts[1]["model"]
    assert json.loads(salvaged["reply"])["results"] == [{"step": 1, "passed": True, "reason": "ran the script"}]


def test_salvage_after_a_failed_retry_keeps_each_attempt_status(tmp_path: Path) -> None:
    # The first reply is cut off and the retry fails in transport: the LLM
    # attempts keep their own status, and the salvage entry is numbered like
    # the truncated attempt it was rebuilt from.
    verifier = _stage_verifier(tmp_path)
    truncated = '{"results":[{"step":1,"passed":true,"reason":"ran the script"},{"step":2,"passed":tr'
    calls: list[str] = []
    route = _routed(calls, accuracy=_ACCURACY_RESPONSE, goal=_GOAL_RESPONSE, behavior=truncated)
    behavior_calls: list[str] = []

    def call(prompt: str, **kwargs: Any):
        if "EXPECTED BEHAVIORS" in prompt:
            behavior_calls.append(prompt)
            return (truncated, None) if len(behavior_calls) == 1 else (None, "request timed out")
        return route(prompt, **kwargs)

    verifier.call_public_llm = call
    verifier._call_public_llm_with_provenance = lambda prompt, **kwargs: (*call(prompt, **kwargs), {})

    verifier.main()

    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    assert rich["behavior_check"] == 0.5
    attempts = [attempt for attempt in _attempts(verifier.VERIFIER_DIR) if attempt["metric"] == "behavior_check"]
    assert [(attempt["attempt"], attempt["method"], attempt["status"]) for attempt in attempts] == [
        (1, "llm", "rejected"),
        (2, "llm", "error"),
        (1, "salvage", "accepted"),
    ]
    assert attempts[1]["problem"] == "request timed out"


def _cut_point(limit: int) -> int:
    # Where _truncate_for_behavior ends the kept head of a text longer than *limit*.
    marker = "\n...[truncated]...\n"
    return (limit - len(marker)) * 2 // 3


def test_long_judge_evidence_is_bounded_and_redacted_before_it_is_cut(tmp_path: Path, monkeypatch) -> None:
    verifier = _stage_verifier(tmp_path)
    credential = "configured-cut-credential-DO-NOT-SAVE"
    monkeypatch.setenv("ANTHROPIC_API_KEY", credential)
    # Each credential crosses the point where the raw text would be cut, so
    # cutting before redaction would save its first characters in clear.
    prompt_cut = _cut_point(eval_template.JUDGE_EVIDENCE_PROMPT_CHARS)
    reply_cut = _cut_point(eval_template.JUDGE_EVIDENCE_REPLY_CHARS)
    prompt = "p" * (prompt_cut - 12) + credential + "p" * 50_000
    reply = "r" * (reply_cut - 12) + credential + "r" * 50_000

    token = verifier._ACTIVE_JUDGE_EVIDENCE.set([])
    try:
        verifier._record_judge_attempt("accuracy", 1, prompt=prompt, reply=reply, problem="too long")
    finally:
        verifier._ACTIVE_JUDGE_EVIDENCE.reset(token)

    saved_text = (verifier.VERIFIER_DIR / "judge_attempts.json").read_text(encoding="utf-8")
    (entry,) = _attempts(verifier.VERIFIER_DIR)
    assert entry["prompt_chars"] == len(prompt)
    assert entry["reply_chars"] == len(reply)
    assert len(entry["prompt"]) <= eval_template.JUDGE_EVIDENCE_PROMPT_CHARS
    assert len(entry["reply"]) <= eval_template.JUDGE_EVIDENCE_REPLY_CHARS
    assert "...[truncated]..." in entry["prompt"]
    assert "...[truncated]..." in entry["reply"]
    assert "[REDACTED]" in entry["prompt"]
    assert "[REDACTED]" in entry["reply"]
    assert credential[:8] not in saved_text
    assert credential[-8:] not in saved_text


def test_judge_evidence_write_failure_never_changes_the_score(tmp_path: Path, monkeypatch) -> None:
    verifier = _stage_verifier(tmp_path)
    calls: list[str] = []
    route = _routed(calls, accuracy=_ACCURACY_RESPONSE, goal=_GOAL_RESPONSE, behavior=_BEHAVIOR_RESPONSE)
    verifier.call_public_llm = route
    verifier._call_public_llm_with_provenance = lambda prompt, **kwargs: (*route(prompt, **kwargs), {})
    verifier.JUDGE_ATTEMPTS_FILE_NAME = "missing-directory/judge_attempts.json"
    (verifier.VERIFIER_DIR / "missing-directory").write_text("a file, not a directory", encoding="utf-8")

    verifier.main()

    numeric = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    assert numeric["accuracy"] == 0.6
    assert numeric["behavior_check"] == 0.5


def test_judge_deadline_is_not_swallowed_by_the_evidence_writer(tmp_path: Path) -> None:
    verifier = _stage_verifier(tmp_path)

    class _DeadlineDuringWrite:
        def __truediv__(self, _name: str):
            raise TimeoutError("LLM judge time budget exhausted")

    verifier._verifier_output_dir = _DeadlineDuringWrite
    token = verifier._ACTIVE_JUDGE_EVIDENCE.set([])
    try:
        with pytest.raises(TimeoutError):
            verifier._record_judge_attempt("accuracy", 1, prompt="p", reply="r", problem=None)
    finally:
        verifier._ACTIVE_JUDGE_EVIDENCE.reset(token)


def test_judges_outside_the_verifier_run_save_no_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(eval_template, "VERIFIER_DIR", tmp_path)
    monkeypatch.setattr(eval_template, "SKILL_EVALUATOR_REWARD_JSON", tmp_path / "skill_evaluator_reward.json")
    monkeypatch.setattr(eval_template, "REWARD_JSON", tmp_path / "reward.json")
    monkeypatch.setattr(eval_template, "call_public_llm", _script([(_ACCURACY_RESPONSE, None)], []))

    assert eval_template.judge_accuracy("question", "ground truth", "agent evidence")["score"] == 0.6
    assert not (tmp_path / "judge_attempts.json").exists()


@pytest.mark.parametrize("ragas_score", [0.5, float("nan")])
def test_non_binary_ragas_score_falls_back_to_the_prompt_judge(tmp_path: Path, ragas_score: float) -> None:
    verifier = _stage_verifier(tmp_path)
    verifier._ragas_goal_accuracy_enabled = lambda: True

    def ragas(*_args: Any, **_kwargs: Any):
        return {"score": ragas_score, "method": "ragas", "model": "ragas-model"}

    calls: list[str] = []
    route = _routed(calls, accuracy=_ACCURACY_RESPONSE, goal=_GOAL_RESPONSE, behavior=_BEHAVIOR_RESPONSE)
    verifier._judge_goal_accuracy_ragas = ragas
    verifier.call_public_llm = route
    verifier._call_public_llm_with_provenance = lambda prompt, **kwargs: (*route(prompt, **kwargs), {})

    verifier.main()

    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    assert rich["goal_accuracy"] == 1.0
    assert rich["details"]["goal_accuracy"]["method"] == "custom"
    goal_attempts = [attempt for attempt in _attempts(verifier.VERIFIER_DIR) if attempt["metric"] == "goal_accuracy"]
    assert [(attempt["method"], attempt["status"]) for attempt in goal_attempts] == [
        ("ragas", "error"),
        ("llm", "accepted"),
    ]
    assert "non-binary" in goal_attempts[0]["problem"]


def test_binary_ragas_score_is_saved_as_an_accepted_attempt(tmp_path: Path) -> None:
    verifier = _stage_verifier(tmp_path)
    verifier._ragas_goal_accuracy_enabled = lambda: True
    verifier._judge_goal_accuracy_ragas = lambda *_args, **_kwargs: {
        "score": 0.0,
        "achieved": False,
        "method": "ragas",
        "model": "ragas-model",
    }
    calls: list[str] = []
    route = _routed(calls, accuracy=_ACCURACY_RESPONSE, goal=_GOAL_RESPONSE, behavior=_BEHAVIOR_RESPONSE)
    verifier.call_public_llm = route

    verifier.main()

    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    assert rich["goal_accuracy"] == 0.0
    (attempt,) = [attempt for attempt in _attempts(verifier.VERIFIER_DIR) if attempt["metric"] == "goal_accuracy"]
    assert attempt == {
        "metric": "goal_accuracy",
        "attempt": 1,
        "method": "ragas",
        "model": "ragas-model",
        "status": "accepted",
        "problem": None,
        "score": 0.0,
    }


def test_judge_contract_is_reserved_reward_metadata() -> None:
    assert "judge_contract" in RESERVED_METRIC_NAMES


@pytest.mark.parametrize(
    "text",
    [
        '{"headers": {"Authorization": "Bearer pa\\"CANARY_JUDGE_SUFFIX"}}',
        'headers["Authorization"] = "Bearer pa\\"CANARY_JUDGE_SUFFIX"',
        'headers["Authorization"] = f"Bearer pa\\"CANARY_JUDGE_SUFFIX"',
        '{"headers": {"X-API-Key": "pa\\"CANARY_JUDGE_SUFFIX"}}',
        'headers["X-API-Key"] = "pa\\"CANARY_JUDGE_SUFFIX"',
        'curl -H \'Authorization: Bearer "pa\\"CANARY_JUDGE_SUFFIX"\' https://x',
    ],
)
def test_judge_header_format_preservation_masks_escaped_secret_suffixes(text: str) -> None:
    evidence = eval_template._redact_judge_evidence(text)
    assert "CANARY_JUDGE_SUFFIX" not in evidence
    assert "<redacted>" in evidence


_GATE_FACTS = [
    {"claim": "report.py", "observed": True},
    {"claim": "summary.md", "observed": True},
]


@pytest.mark.parametrize(
    ("expected_behavior", "verified", "why"),
    [
        (["writes `report.py`"], True, "a single observed artifact settles the expectation"),
        (
            ["writes `report.py`, `summary.md`"],
            True,
            "several observed artifacts in one clause still settle it",
        ),
        (["explains the trade-offs"], False, "plain prose names nothing checkable"),
        (["writes `missing.py`"], False, "an unobserved artifact settles nothing"),
        (
            ["writes `report.py` and explains why each column was dropped"],
            False,
            "the prose half needs the final response the limits cut",
        ),
        (["writes `report.py` then explains the result"], False, "clause joined by 'then'"),
        (["writes `report.py`; explains why"], False, "clause joined by a semicolon"),
        (["writes `report.py`. Explains why."], False, "a second sentence"),
        (["writes `report.py`", "explains why"], False, "a prose expectation beside an artifact one"),
    ],
)
def test_expected_behaviors_verified_requires_every_clause_to_be_checkable(
    expected_behavior: list[str], verified: bool, why: str
) -> None:
    # The gate exists to stop a score being withheld when the deterministic
    # facts already settle the expectation. Naming an observed artifact is not
    # enough on its own: an expectation that also asks for prose is settled
    # only in its artifact half, and the other half is exactly what the cut
    # final response would have answered. Keeping the gate withholds a score;
    # lifting it wrongly keeps a score the evidence does not support, so an
    # unrecognized clause shape must fall on the withholding side.
    assert atif_helpers._expected_behaviors_verified(expected_behavior, _GATE_FACTS) is verified, why


def test_expected_behaviors_verified_mixed_clause_does_not_lift_the_evidence_gate() -> None:
    # End-to-end through behavior_evidence_gap: the mixed expectation must
    # leave the gap in place, so behavior_check still withholds its score.
    omitted = {
        "required_complete": False,
        "required_missing": ["final response"],
        "reason": "final response truncated",
        "expected_behaviors_verified": atif_helpers._expected_behaviors_verified(
            ["writes `report.py` and explains why each column was dropped"], _GATE_FACTS
        ),
    }
    assert omitted["expected_behaviors_verified"] is False
    assert atif_helpers.behavior_evidence_gap(omitted) != ""

    omitted["expected_behaviors_verified"] = atif_helpers._expected_behaviors_verified(
        ["writes `report.py`"], _GATE_FACTS
    )
    assert omitted["expected_behaviors_verified"] is True
    assert atif_helpers.behavior_evidence_gap(omitted) == ""
