# SPDX-License-Identifier: Apache-2.0

"""Character metadata and host/standalone judge integration regressions."""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from tests.conftest import load_harbor_eval_template

from skillevaluator.tier3.eval_core import llm_judge
from skillevaluator.tier3.eval_core.character_facts import append_character_facts, format_character_facts
from skillevaluator.tier3.harbor import adapter, runner


@pytest.mark.parametrize("text", ["", " 42 \t\r\n", "café 中文 Ελληνικά 😀", r"\u00a0"])
def test_ordinary_text_has_no_character_facts(text):
    assert format_character_facts(text, label="evidence") == ""


@pytest.mark.parametrize(
    "character", ["\u00a0", "\u202f", "\u200b", "\u2060", "\ufeff", "\u00ad", "\u202e", "\U000e0001"]
)
def test_character_identity_and_ascii_context(character):
    text = f"😀a{character}b"
    facts = format_character_facts(text, label='field"\u00a0')
    assert f"line 1, column 3: U+{ord(character):04X}" in facts
    assert facts.isascii()
    assert json.loads(facts.split("escaped context: ")[1]) == text


def test_positions_count_code_points_and_only_lf_advances_lines():
    facts = format_character_facts("😀\u00a0\r\n\u2028x\u200b", label="evidence")
    assert "line 1, column 2: U+00A0 NO-BREAK SPACE" in facts
    assert "line 2, column 1: U+2028 LINE SEPARATOR" in facts
    assert "line 2, column 3: U+200B ZERO WIDTH SPACE" in facts


def test_occurrences_and_context_are_bounded():
    facts = format_character_facts("\u00a0" * 10_000, label="evidence", max_occurrences=2, context_radius=0)
    assert facts.count("- line ") == 2
    assert facts.count('escaped context: "\\u00a0"') == 2
    assert "Omitted 9998 additional matching characters." in facts
    assert len(facts) < 500
    assert "Omitted 1 additional matching characters." in format_character_facts("\u00a0" * 21, label="evidence")


def test_context_is_bounded_in_code_points():
    facts = format_character_facts("outside😀a\u00a0b😀outside", label="evidence", context_radius=2)
    assert json.loads(facts.split("escaped context: ")[1]) == "😀a\u00a0b😀"


@pytest.mark.parametrize(("maximum", "radius"), [(0, 20), (-1, 20), (20, -1)])
def test_invalid_bounds_are_rejected(maximum, radius):
    with pytest.raises(ValueError):
        format_character_facts("", label="evidence", max_occurrences=maximum, context_radius=radius)


@pytest.mark.parametrize("setting", ["", "0", "false", "no", "off"])
def test_disabled_metadata_preserves_prompt(monkeypatch, setting):
    monkeypatch.setenv("SKILL_EVAL_JUDGE_CHARACTER_FACTS", setting)
    assert append_character_facts("original", agent_text="\u00a0") == "original"


@pytest.mark.parametrize("setting", ["1", "true", "YES", " on "])
def test_enabled_metadata_is_separate_and_optional(monkeypatch, setting):
    monkeypatch.setenv("SKILL_EVAL_JUDGE_CHARACTER_FACTS", setting)
    assert append_character_facts("original", agent_text="ordinary text") == "original"
    assert append_character_facts("original", agent_text="\u00a0").startswith("original\n\nCharacter facts")


@pytest.mark.parametrize("implementation", ["host", "verifier"])
@pytest.mark.parametrize("judge_name", ["judge_accuracy", "judge_goal_accuracy", "judge_behavior_check"])
def test_all_judges_preserve_original_evidence_and_append_facts(monkeypatch, implementation, judge_name):
    judge = llm_judge if implementation == "host" else load_harbor_eval_template()
    payload = json.dumps(
        {
            "criteria": dict.fromkeys(
                ["SKILL_IDENTIFIED", "ACTION_CORRECT", "FACTUALLY_ACCURATE", "TASK_ADDRESSED", "ACTIONABLE"], True
            ),
            "score": 1.0,
            "reason": "test",
            "achieved": True,
            "user_goal": "test",
            "end_state": "test",
            "results": [{"step": 1, "passed": True, "reason": "test"}],
            "summary": "test",
        }
    )
    mock = MagicMock(return_value=(payload, None))
    monkeypatch.setattr(judge, "call_public_llm", mock)
    if implementation == "verifier":
        monkeypatch.setattr(judge, "_public_provider", lambda: "anthropic")
        if judge_name == "judge_goal_accuracy":
            mock = MagicMock(return_value=(payload, None, {}))
            monkeypatch.setattr(judge, "_call_public_llm_with_provenance", mock)
    answer = '{{env.SECRET}}\nassert f("\u00a0") is False'
    args = (
        (answer, ["Reject \u202f"])
        if judge_name == "judge_behavior_check"
        else ("Task \u202f", "Expected \u200b", answer)
    )
    if judge_name == "judge_goal_accuracy":
        args += ("Tool \u2060",)
    monkeypatch.delenv("SKILL_EVAL_JUDGE_CHARACTER_FACTS", raising=False)
    assert getattr(judge, judge_name)(*args)["score"] == 1.0
    original = mock.call_args.args[0]
    assert answer in original
    assert "Character facts for" not in original
    monkeypatch.setenv("SKILL_EVAL_JUDGE_CHARACTER_FACTS", "1")
    assert getattr(judge, judge_name)(*args)["score"] == 1.0
    augmented = mock.call_args.args[0]
    assert augmented.startswith(original + "\n\nCharacter facts for")
    assert answer in augmented
    assert "line 2, column 11: U+00A0 NO-BREAK SPACE" in augmented
    assert "U+202F" in augmented
    if judge_name != "judge_behavior_check":
        assert 'Character facts for "ground_truth"' in augmented
    if judge_name == "judge_goal_accuracy":
        assert 'Character facts for "tool_summary"' in augmented


@pytest.mark.parametrize("implementation", ["host", "verifier"])
def test_behavior_facts_describe_only_the_compacted_evidence(monkeypatch, implementation):
    judge = llm_judge if implementation == "host" else load_harbor_eval_template()
    monkeypatch.setenv("SKILL_EVAL_JUDGE_CHARACTER_FACTS", "1")
    monkeypatch.setattr(judge, "_compact_behavior_conversation", lambda _: "kept \u00a0")
    mock = MagicMock(
        return_value=(
            json.dumps({"results": [{"step": 1, "passed": True, "reason": "ok"}], "score": 1, "summary": "ok"}),
            None,
        )
    )
    monkeypatch.setattr(judge, "call_public_llm", mock)
    judge.judge_behavior_check("removed \u200b", ["check"])
    prompt = mock.call_args.args[0]
    assert "line 1, column 6: U+00A0" in prompt
    assert "U+200B" not in prompt


def test_opt_in_selects_prompt_goal_judge(monkeypatch):
    judge = load_harbor_eval_template()
    monkeypatch.setattr(judge, "_public_provider", lambda: "openai")
    monkeypatch.setattr(judge, "_selected_judge_model", lambda: "gpt-4.1-mini")
    monkeypatch.setattr(judge, "_resolve_url", lambda _: "https://api.openai.com/v1/chat/completions")
    monkeypatch.delenv("SKILL_EVAL_JUDGE_CHARACTER_FACTS", raising=False)
    assert judge._ragas_goal_accuracy_enabled()
    monkeypatch.setenv("SKILL_EVAL_JUDGE_CHARACTER_FACTS", "1")
    assert not judge._ragas_goal_accuracy_enabled()


def test_helper_is_bundled_and_works_without_installed_package(tmp_path):
    adapter._copy_verifier(tmp_path)
    helper = adapter._evaluator_tests_dir(tmp_path) / "character_facts.py"
    assert helper.is_file()
    script = "import runpy,sys; m=runpy.run_path(sys.argv[1]); print(m['format_character_facts']('\u00a0', label='evidence'))"
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(helper)], capture_output=True, text=True, check=True
    )
    assert "U+00A0 NO-BREAK SPACE" in result.stdout


def test_operator_flag_is_verifier_only_and_recorded(monkeypatch):
    from skillevaluator.provider_config import ProviderConfig

    name = "SKILL_EVAL_JUDGE_CHARACTER_FACTS"
    monkeypatch.setenv(name, "1")
    provider = ProviderConfig(
        provider="openai", model="gpt-4.1-mini", api_key="test-key", base_url=None, litellm_model="openai/gpt-4.1-mini"
    )
    environment = runner._provider_environment(provider)
    assert environment[name] == "1"
    assert runner._job_judge_verifier_env(environment, "default")[name] == "${" + name + "}"
    assert runner._job_judge_subprocess_env(environment, "default")[name] == "1"
    assert runner._job_judge_verifier_env(environment, "custom_only") == {}
    assert runner._job_judge_subprocess_env(environment, "custom_only") == {}
    assert runner._judge_model_config(provider, environment, "default")["character_facts"] == "1"
    resolved, errors = runner._resolve_runtime_env({name: "0"}, env_mode="docker")
    assert resolved == {}
    assert errors
