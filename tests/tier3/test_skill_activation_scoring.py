# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expected skill activation is required for positive execution credit."""

from __future__ import annotations

from types import ModuleType

import pytest

from skillevaluator.tier3.eval_core import checks
from skillevaluator.tier3.eval_core.codex_tool_call_normalizer import UNSUPPORTED_NATIVE_CODEX_EXEC


@pytest.fixture(params=("host", "verifier"))
def scorer(request: pytest.FixtureRequest, harbor_eval_template: ModuleType) -> ModuleType:
    return checks if request.param == "host" else harbor_eval_template


@pytest.mark.parametrize(
    "call",
    [
        {"action": "Read", "action_input": {"file_path": "/workspace/README.md"}},
        {"action": "Read", "action_input": {"file_path": "/skills/other/SKILL.md"}},
        {"action": "Read", "action_input": {"file_path": "/skills/demo/SKILL.md"}, "is_error": True},
        {"action": "Bash", "action_input": {"command": "python /skills/demo/scripts/run.py"}},
    ],
    ids=("unrelated-work", "wrong-skill", "failed-read", "script-without-activation"),
)
@pytest.mark.parametrize("expected_script", (None, "run.py"))
def test_zero_activation_cannot_receive_execution_credit(
    scorer: ModuleType, call: dict, expected_script: str | None
) -> None:
    result = scorer.score_skill_execution([call], "demo", expected_script)

    assert result["score"] == 0.0
    assert result["details"]["activation"]["passed"] is False
    assert result["details"]["activation"]["score"] == 0.0
    # Keep the independent evidence visible even though it cannot override activation.
    assert result["details"]["error_recovery"]["score"] == 1.0
    assert set(result["details"]) == {"activation", "script_execution", "workflow_order", "error_recovery"}


def test_empty_positive_trajectory_retains_zero_activation_evidence(scorer: ModuleType) -> None:
    result = scorer.score_skill_execution([], "demo")

    assert result["score"] == 0.0
    assert result["details"]["activation"]["passed"] is False
    assert result["details"]["activation"]["score"] == 0.0
    assert result["details"]["message"] == "No tool calls in trajectory"


@pytest.mark.parametrize(
    ("call", "options", "activation_score", "execution_score"),
    [
        ({"action": "Read", "action_input": {"file_path": "/skills/demo/SKILL.md"}}, {}, 1.0, 1.0),
        ({"action": "Bash", "action_input": {"command": "cat /skills/demo/SKILL.md"}}, {}, 0.75, 0.9375),
        (
            {"action": "Read", "action_input": {"file_path": "/skills/alternate/SKILL.md"}},
            {"acceptable_skills": ["alternate"]},
            0.75,
            0.9375,
        ),
        ({"action": "Skill", "action_input": {"skill": "demo"}}, {"skill_tool_names": ["demo"]}, 1.0, 1.0),
    ],
    ids=("target-read", "shell-partial-credit", "acceptable-alternate", "native-skill"),
)
def test_supported_activation_preserves_subcheck_average(
    scorer: ModuleType, call: dict, options: dict, activation_score: float, execution_score: float
) -> None:
    result = scorer.score_skill_execution([call], "demo", **options)

    assert result["details"]["activation"]["score"] == activation_score
    assert result["score"] == execution_score


def test_unsupported_activation_is_unknown_and_not_forced_to_zero(scorer: ModuleType) -> None:
    call = {
        "action": "functions.exec",
        "action_input": {"code": "await tools.exec_command(args)"},
        "normalization_status": UNSUPPORTED_NATIVE_CODEX_EXEC,
    }
    result = scorer.score_skill_execution([call], "demo", "run.py")

    assert result["details"]["activation"]["passed"] is None
    assert result["details"]["activation"]["supported"] is False
    assert result["score"] == 0.5


@pytest.mark.parametrize("expected_skill", (None, ""))
def test_negative_control_does_not_require_positive_activation(scorer: ModuleType, expected_skill: str | None) -> None:
    result = scorer.score_skill_execution(
        [{"action": "Read", "action_input": {"file_path": "/workspace/README.md"}}],
        expected_skill,
        should_trigger=False,
        evaluated_skill="demo",
        require_evaluated_skill=True,
    )

    assert result["score"] == 1.0
    assert result["details"]["negative_check"]["passed"] is True
    assert "activation" not in result["details"]


@pytest.mark.parametrize("expected_skill", (None, "", "demo"))
def test_unlabeled_case_remains_skipped(scorer: ModuleType, expected_skill: str | None) -> None:
    result = scorer.score_skill_execution([], expected_skill, should_trigger=None)

    assert result["score"] == 1.0
    assert "skipped" in result["details"]["message"]
    assert "activation" not in result["details"]


def test_missing_expected_skill_remains_skipped(scorer: ModuleType) -> None:
    result = scorer.score_skill_execution([], "", should_trigger=True)

    assert result["score"] == 1.0
    assert "skipped" in result["details"]["message"]
    assert "activation" not in result["details"]
