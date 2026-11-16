# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Case-first Tier 3 summary and How-to-improve ranking."""

import json
import time
from pathlib import Path

import pytest
from rich.console import Console

from skillevaluator.tier3.case_summary import _weakens_requirements, render_case_summary, verdict, what_to_fix
from skillevaluator.tier3.result_display import render_result


def _case(*passed: bool, missing: int = 0, prefix: str = "t") -> dict:
    return {
        "attempts": [
            {"attempt": i + 1, "trial": f"{prefix}{i}", "score": 0.9 if p else 0.2, "passed": p}
            for i, p in enumerate(passed)
        ],
        "attempts_missing": missing,
    }


def _agent(with_cases: dict, without_cases: dict) -> dict:
    return {
        "execution_status": "succeeded",
        "pass_at_k": {"with_skill": {"k": 2, "cases": with_cases}, "without_skill": {"k": 2, "cases": without_cases}},
        "lift": {"overall": {"with_skill": 0.7, "without_skill": 0.4, "delta": 0.3}},
        "dimensions_with_skill": {"effectiveness": {"score": 0.5}, "security": {"score": 1.0}},
        "dimensions_without_skill": {"effectiveness": {"score": 0.3}, "security": {"score": 1.0}},
    }


def _trial(run_dir: Path, agent: str, name: str, reward: dict, observation: str = "") -> None:
    trial = run_dir / agent / "with-skill" / "trials" / name
    trial.mkdir(parents=True)
    (trial / "reward.json").write_text(json.dumps(reward))
    steps = [{"source": "user", "message": "q"}, {"source": "agent", "message": "done", "observation": observation}]
    (trial / "trajectory.json").write_text(json.dumps({"steps": steps}))


def _reward(activated: bool, reason: str) -> dict:
    return {
        "goal_accuracy": 0.2,
        "details": {
            "skill_execution": {"activation": {"passed": activated}},
            "goal_accuracy": {"reason": reason},
            "behavior_check": {"results": [{"passed": False, "reason": reason}]},
        },
    }


@pytest.mark.parametrize(
    "action",
    [
        "In evals/evals.json, remove the expectation for syllable_count from the syllable-positive case.",
        "Loosen the assertion in evals/evals.json.",
        "Drop the syllable_count expectation from evals/evals.json.",
        "The assertion should be removed from the dataset.",
        "Lower the pass threshold to make this case pass.",
        "Remove the `expectation` for syllable_count from the dataset.",
        "Drop the `syllable_count` `expectation` from evals/evals.json.",
        "Loosen the `assertion` in evals/evals.json.",
        "The `expectation` should be removed from the dataset.",
        "The expectation for `syllable_count` should be removed from the dataset.",
        "The `pass threshold` for the syllable-positive case should be lowered.",
        "Remove the expected output mismatch by dropping the assertion from evals/evals.json.",
        "Remove the expected output for syllable_count from evals/evals.json.",
        "Remove the expected output mismatch by deleting the syllable-positive case from evals/evals.json.",
        "Remove the assertion failure by dropping the `syllable-positive` case from evals/evals.json.",
        "Remove the expected output mismatch by implementing syllable_count, then delete the syllable-positive case.",
        "Fix the mismatch by removing the syllable-positive entry from evals/evals.json.",
        "Remove the expected output mismatch by deleting the syllable-positive test case.",
        "Remove the failure by commenting out the syllable-positive case in evals/evals.json.",
        "Remove the mismatch by disabling the dataset entry for syllable-positive.",
        "Delete the syllable-positive row from the dataset.",
        "Skip the syllable-positive case in evals/evals.json.",
        "The syllable-positive case should be deleted from evals/evals.json.",
        "The syllable-positive entry in evals/evals.json should be removed.",
        "The syllable-positive test case must be dropped.",
        "Do not remove the expectation; delete the syllable-positive case from the dataset.",
        "Delete the syllable-positive case from evals/evals.json.",
        "Remove the test case from the dataset.",
        "Drop the eval case.",
        "The test case should be deleted.",
        "Remove the special case from evals/evals.json.",
        "In evals/evals.json, remove the special case for syllable-positive.",
        "In the dataset, delete the edge case.",
        "The special case in evals/evals.json should be removed.",
    ],
)
def test_case_action_rejects_requirement_weakening(tmp_path, monkeypatch, action):
    from skillevaluator.tier3.eval_core import llm_judge

    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: (json.dumps([action]), None))
    source = {"agents": {"a": _agent({"syllable-positive": _case(False)}, {})}}
    [fix] = what_to_fix(source, run_dir=tmp_path)

    assert fix["action"] == "Address in SKILL.md: Missing syllable_count"
    assert fix["source"] == "deterministic"
    assert fix["fallback_reason"] == "requirement_weakening"


@pytest.mark.parametrize(
    "case_id,action",
    [
        ("positive-001", "Delete the case for positive-001 so it stops failing."),
        ("positive-001", "Remove the case named positive-001 so the run passes."),
        ("positive-001", "Delete case positive-001."),
        ("positive-001", "Delete test case positive-001."),
        ("positive-001", "Fix the mismatch by deleting the case for positive-001."),
        ("positive-001", "Delete the entry for positive-001."),
        ("positive-001", "Remove positive-001 from evals/evals.json."),
        ("negative-007", "Remove the case for `negative-007`"),
        ("negative-007", 'Drop the "negative-007" case.'),
        ("syllable-count", "Delete the syllable-count case."),
        ("syllable-count", "Remove the case for syllable-count."),
        ("smoke", "Delete the smoke case."),
        ("smoke", "Remove the case for smoke so it stops failing."),
        ("positive-001", "The case for positive-001 should be deleted."),
        ("positive-001", "The positive-001 entry should be deleted."),
        ("positive-001", "The case for positive-001 in evals/evals.json can be removed."),
        ("syllable-count", "The syllable-count case should be removed."),
        ("smoke", "The smoke case must be dropped."),
        ("positive-001", "Exclude the case for positive-001 from the run."),
        ("positive-001", "Omit the case for positive-001."),
        ("positive-001", "Get rid of the positive-001 case."),
        ("positive-001", "Take out the case for positive-001."),
        ("positive-001", "Take positive-001 out of the dataset."),
        ("positive-001", "Take the positive-001 case out of evals/evals.json."),
        ("syllable-count", "Leave out the syllable-count case."),
        ("smoke", "Leave the smoke case out of the dataset."),
        ("positive-001", "Filter out the case for positive-001."),
        ("positive-001", "Filter positive-001 out of evals/evals.json."),
        ("smoke", "Discard the smoke case."),
        ("positive-001", "Strip the case for positive-001 from evals/evals.json."),
        ("positive-001", "Ignore the case for positive-001."),
        ("smoke", "Ignore the smoke case."),
        ("positive-001", "Stop running the case for positive-001."),
        ("syllable-count", "Retire the syllable-count case."),
        ("positive-001", "Replace the positive-001 case with an easier one."),
        ("positive-001", "Replace the case for positive-001 in evals/evals.json with a simpler one."),
        ("positive-001", "Replace positive-001 with a simpler case."),
        ("smoke", "Swap out the smoke case for an easier one."),
        ("positive-001", "Instead of keeping the case for positive-001, move on to the next one."),
        ("positive-001", "The positive-001 case should be excluded."),
        ("positive-001", "The case for positive-001 should be replaced with an easier one."),
        ("syllable-count", "The syllable-count case should be discarded."),
        ("smoke", "The smoke case can be left out."),
        ("positive-001", "The case for positive-001 should be ignored."),
        ("positive-001", "The positive-001 case should be taken out of the dataset."),
        ("syllable-count", "The syllable-count case must be filtered out."),
        ("positive-001", "The case for positive-001 can be stripped from evals/evals.json."),
        ("smoke", "The smoke case should be retired."),
    ],
)
def test_case_action_rejects_deleting_a_dataset_case_named_by_its_id(tmp_path, monkeypatch, case_id, action):
    from skillevaluator.tier3.eval_core import llm_judge

    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: (json.dumps([action]), None))
    source = {"agents": {"a": _agent({case_id: _case(False)}, {})}}
    [fix] = what_to_fix(source, run_dir=tmp_path)

    assert fix["action"] == "Address in SKILL.md: Missing syllable_count"
    assert fix["source"] == "deterministic"
    assert fix["fallback_reason"] == "requirement_weakening"


def test_case_action_rejects_deleting_another_case_named_in_the_dataset_snapshot(tmp_path, monkeypatch):
    from skillevaluator.tier3.eval_core import llm_judge

    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    (tmp_path / "dataset_snapshot.json").write_text(
        json.dumps({"dataset": [{"id": "syllable-positive", "question": "q"}, {"id": "smoke", "question": "q"}]})
    )
    action = "Delete the smoke case so syllable-positive stops failing."
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: (json.dumps([action]), None))
    [fix] = what_to_fix({"agents": {"a": _agent({"syllable-positive": _case(False)}, {})}}, run_dir=tmp_path)

    assert fix["action"] == "Address in SKILL.md: Missing syllable_count"
    assert fix["fallback_reason"] == "requirement_weakening"


@pytest.mark.parametrize(
    "case_id,action",
    [
        ("positive-001", "Remove the special case in parser.py for positive-001 and implement the general rule."),
        ("positive-001", "Remove the special case for positive-001 in parser.py and implement the general rule."),
        ("positive-001", "Remove the case for positive-001 from analyze_text.py and implement the general rule."),
        ("syllable-count", "Remove the syllable-count case in analyze_text.py and implement it generally."),
        ("smoke", "Remove the smoke case in analyze_text.py and implement it generally."),
        ("positive-001", "Remove the test case failure for positive-001 by implementing the rule in SKILL.md"),
        ("positive-001", "Do not delete the case for positive-001; implement syllable_count in analyze_text.py."),
        ("syllable-count", "Never remove the syllable-count case; implement syllable_count in analyze_text.py."),
        ("smoke", "Do not drop the smoke case; implement syllable_count in analyze_text.py."),
        ("positive-001", "Remove the special case for zero-length input in analyze_text.py."),
        ("positive-001", "Remove the special case for zero-length input and implement syllable_count generally."),
    ],
)
def test_case_action_keeps_advice_that_names_a_case_id_but_edits_the_implementation(
    tmp_path, monkeypatch, case_id, action
):
    from skillevaluator.tier3.eval_core import llm_judge

    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: (json.dumps([action]), None))
    source = {"agents": {"a": _agent({case_id: _case(False)}, {})}}
    [fix] = what_to_fix(source, run_dir=tmp_path)

    assert fix["action"] == action
    assert fix["source"] == "llm"
    assert "fallback_reason" not in fix


@pytest.mark.parametrize(
    "case_id,action",
    [
        ("positive-001", "Do not exclude the case for positive-001; implement the rule in SKILL.md."),
        ("positive-001", "Never replace the case for positive-001 with an easier one; implement the rule in SKILL.md."),
        ("smoke", "Never replace the smoke case; implement the missing step in SKILL.md."),
        ("syllable-count", "Don't ignore the syllable-count case; implement the counting rule in SKILL.md."),
        ("positive-001", "Do not take positive-001 out of the dataset; implement the rule in SKILL.md."),
        ("positive-001", "Instead of deleting the case for positive-001, implement the rounding rule in SKILL.md."),
        ("syllable-count", "Instead of removing the syllable-count case, implement the counting rule in SKILL.md."),
        ("smoke", "Implement the missing step in SKILL.md instead of dropping the smoke case."),
        ("positive-001", "Fix the rounding step rather than excluding the case for positive-001."),
        ("positive-001", "Make positive-001 pass without deleting the case for positive-001."),
        ("positive-001", "Discard the special case in parser.py for positive-001 and implement the general rule."),
        ("syllable-count", "Exclude the syllable-count case in analyze_text.py and implement it generally."),
        ("positive-001", "Take the special case for positive-001 out of parser.py and implement the general rule."),
        ("smoke", "Retire the smoke case from analyze_text.py and implement it generally."),
        ("positive-001", "Discard the test case failure for positive-001 by implementing the rule in SKILL.md."),
        ("positive-001", "Ignore case when comparing units, as positive-001 expects."),
        ("positive-001", "Strip whitespace from the answer, as positive-001 expects."),
        ("positive-001", "Take the case for positive-001 into account in SKILL.md."),
        ("positive-001", "Replace the special case for positive-001 in parser.py with the general rounding rule."),
    ],
)
def test_case_action_keeps_negated_instead_of_and_implementation_advice_for_the_wider_verbs(
    tmp_path, monkeypatch, case_id, action
):
    from skillevaluator.tier3.eval_core import llm_judge

    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: (json.dumps([action]), None))
    source = {"agents": {"a": _agent({case_id: _case(False)}, {})}}
    [fix] = what_to_fix(source, run_dir=tmp_path)

    assert fix["action"] == action
    assert fix["source"] == "llm"
    assert "fallback_reason" not in fix


@pytest.mark.parametrize(
    "case_id,action",
    [
        ("positive-001", "Run without failures by deleting the positive-001 case from evals/evals.json."),
        ("positive-001", "Make the run pass without failures by dropping the case for positive-001."),
        ("smoke", "Get a green run without extra work by excluding the smoke case."),
        ("positive-001", "Rather than a fix, delete the case for positive-001."),
        ("positive-001", "Instead of the old approach, delete the case for positive-001 from the dataset."),
    ],
)
def test_case_action_rejects_a_removal_when_the_negation_word_governs_something_else(
    tmp_path, monkeypatch, case_id, action
):
    from skillevaluator.tier3.eval_core import llm_judge

    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: (json.dumps([action]), None))
    [fix] = what_to_fix({"agents": {"a": _agent({case_id: _case(False)}, {})}}, run_dir=tmp_path)

    assert fix["action"] == "Address in SKILL.md: Missing syllable_count"
    assert fix["fallback_reason"] == "requirement_weakening"


@pytest.mark.parametrize(
    "case_id,action",
    [
        ("positive-001", "Make positive-001 pass without deleting the case for positive-001."),
        ("positive-001", "Make positive-001 pass without simply removing the case for positive-001."),
        ("positive-001", "Fix the rounding step for positive-001 without ever dropping its case."),
        ("syllable-count", "Implement the counting rule instead of just dropping the syllable-count case."),
        ("smoke", "Fix the missing step in SKILL.md rather than excluding the smoke case."),
        ("positive-001", "Remove the special case for UTF-8 and use the standard decoder for positive-001."),
        ("positive-001", "Drop the special case for ISO-8859 input and use the standard decoder for positive-001."),
        ("positive-001", "Delete the utf-16 case and decode with the standard library for positive-001."),
        ("positive-001", "Remove the UNIX case and normalise line endings generally for positive-001."),
        ("positive-001", "Remove the v2-api case and call the current endpoint for positive-001."),
        (
            "positive-001",
            "Exclude the base-64 case from the encoder and handle positive-001 with the standard library.",
        ),
    ],
)
def test_case_action_keeps_advice_whose_negation_or_identifier_is_not_about_a_dataset_case(
    tmp_path, monkeypatch, case_id, action
):
    from skillevaluator.tier3.eval_core import llm_judge

    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: (json.dumps([action]), None))
    [fix] = what_to_fix({"agents": {"a": _agent({case_id: _case(False)}, {})}}, run_dir=tmp_path)

    assert fix["action"] == action
    assert fix["source"] == "llm"
    assert "fallback_reason" not in fix


def test_case_action_rejects_an_unlisted_identifier_when_the_action_points_at_the_dataset(tmp_path, monkeypatch):
    from skillevaluator.tier3.eval_core import llm_judge

    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    action = "Delete the utf-16 case from evals/evals.json, then implement syllable_count."
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: (json.dumps([action]), None))
    [fix] = what_to_fix({"agents": {"a": _agent({"syllable-positive": _case(False)}, {})}}, run_dir=tmp_path)

    assert fix["action"] == "Address in SKILL.md: Missing syllable_count"
    assert fix["fallback_reason"] == "requirement_weakening"


def test_case_action_with_a_long_list_of_known_case_ids_is_checked_in_bounded_time(tmp_path, monkeypatch):
    from skillevaluator.tier3.eval_core import llm_judge

    ids = [f"case-{i:02d}" for i in range(20)]
    dataset = [{"id": case_id, "question": "q"} for case_id in ids + [f"case-{i:03d}" for i in range(230)]]
    (tmp_path / "dataset_snapshot.json").write_text(json.dumps({"dataset": dataset}))
    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    actions = {
        "Remove the " + ", ".join(ids) + " overrides in parser.py and implement the general rule.": "llm",
        "Remove " + " ".join(["a-b-c-d-e-f-g"] * 50) + " case": "llm",
        "Delete the " + ", ".join(ids[:5]) + " cases.": "deterministic",
    }
    for action, expected_source in actions.items():
        monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, action=action, **_k: (json.dumps([action]), None))
        started = time.perf_counter()
        [fix] = what_to_fix({"agents": {"a": _agent({"case-00": _case(False)}, {})}}, run_dir=tmp_path)
        assert time.perf_counter() - started < 1.0
        assert fix["source"] == expected_source


def test_case_action_fallback_text_with_a_long_judge_reason_is_checked_in_bounded_time(tmp_path):
    ids = [f"case-{i:03d}" for i in range(250)]
    (tmp_path / "dataset_snapshot.json").write_text(json.dumps({"dataset": [{"id": i, "question": "q"} for i in ids]}))
    reason = "Missing " + ", ".join(ids[:100]) + " and " + " ".join(["a-b-c-d-e-f-g"] * 200)
    _trial(tmp_path, "a", "t0", _reward(True, reason))
    started = time.perf_counter()
    [fix] = what_to_fix({"agents": {"a": _agent({"case-000": _case(False)}, {})}}, run_dir=tmp_path, use_llm=False)
    assert time.perf_counter() - started < 1.0
    assert fix["source"] == "deterministic"


# Verb modifiers a negator may govern: "without first deleting", "without ever having to delete the case".
_NEGATION_CHAINS = (
    "",
    "first",
    "outright",
    "ever",
    "even",
    "actually",
    "simply",
    "just",
    "also",
    "then",
    "again",
    "entirely",
    "completely",
    "fully",
    "altogether",
    "necessarily",
    "directly",
    "having to",
    "needing to",
    "need to",
    "the need to",
    "having to first",
    "being forced to",
    "being asked to",
    "trying to",
    "attempting to",
    "ever having to",
    "ever even having to first",
)
# (negator, whether the verb after it is in its base form): "without first deleting", "do not first delete".
_STRICT_NEGATORS = (
    ("without", False),
    ("instead of", False),
    ("rather than", False),
    ("there is no need to", True),
    ("should not", True),
)
_ALL_NEGATORS = (*_STRICT_NEGATORS, ("do not", True), ("never", True), ("don't", True), ("avoid", False))


# "not only deleting the case but also fixing the parser" recommends the deletion, so after "not", "do not",
# "should not", "never", "don't" or "avoid" these words void the negation. "without simply removing the case" and
# "instead of just dropping it" are different: there the negator governs the verb.
_ADDITIVE_WORDS = ("only", "just", "merely", "simply", "solely", "exclusively", "purely")
_ADDITIVE_PREFIXES = (*_ADDITIVE_WORDS, "necessarily just", "necessarily only", "even just")
_NOT_NEGATORS = (
    ("not", False),
    ("do not", True),
    ("should not", True),
    ("never", True),
    ("don't", True),
    ("avoid", False),
)


def _negated_advice(negators: tuple[tuple[str, bool], ...], case_subject: str) -> list[str]:
    sentences = []
    for negator, base_form in negators:
        for chain in _NEGATION_CHAINS:
            if negator in dict(_NOT_NEGATORS) and set(chain.split()) & set(_ADDITIVE_WORDS):
                continue
            base = base_form or chain.endswith(" to")
            for verb, subject in (("delete", case_subject), ("lower", "the pass threshold")):
                form = verb if base else verb.replace("lower", "lowering").replace("delete", "deleting")
                sentences.append(" ".join(f"Fix the parser {negator} {chain} {form} {subject}.".split()))
    return sentences


def _unrelated_negation_advice(negators: tuple[tuple[str, bool], ...], case_subject: str) -> list[str]:
    """A noun between the negator (and its modifiers) and the verb: the negator governs the noun, not the removal."""
    return [
        " ".join(f"Fix the parser {negator} {chain} {noun} by {verb} {subject}.".split())
        for negator, _ in negators
        for chain in _NEGATION_CHAINS
        for noun in ("failures", "errors")
        for verb, subject in (("deleting", case_subject), ("lowering", "the pass threshold"))
    ]


def _additive_advice(case_subject: str) -> list[str]:
    """Additive wording ("not only lowering the threshold but also fixing the parser") still recommends the removal."""
    sentences = []
    for negator, base_form in _NOT_NEGATORS:
        for additive in _ADDITIVE_PREFIXES:
            for verb, subject in (("delete", case_subject), ("lower", "the pass threshold")):
                form = verb if base_form else verb.replace("lower", "lowering").replace("delete", "deleting")
                sentences.append(f"Fix the parser and {negator} {additive} {form} {subject} but also add a test.")
    return sentences


@pytest.mark.parametrize(
    "case_id,action",
    [
        ("smoke", "Implement the missing field without having to delete the smoke case."),
        ("smoke", "Fix the parser without first deleting the smoke case."),
        ("smoke", "Fix the parser without outright deleting the smoke case."),
        ("smoke", "Fix the parser without ever having to delete the smoke case."),
        ("smoke", "Fix the parser without even needing to drop the smoke case."),
        ("smoke", "Fix the parser without being forced to exclude the smoke case."),
        ("smoke", "Fix the parser instead of first deleting the smoke case."),
        ("smoke", "Fix the parser rather than simply dropping the smoke case."),
        ("smoke", "Do not first delete the smoke case; fix the parser."),
        ("smoke", "There is no need to delete the smoke case once the parser is fixed."),
        ("smoke", "You should not delete the smoke case; fix the parser."),
        ("positive-001", "Fix the parser without trying to delete the case for positive-001."),
        ("smoke", "Get smoke to pass without first lowering the pass threshold."),
        ("smoke", "Fix the parser so smoke passes without having to loosen the expected output."),
        ("smoke", "Make smoke pass without ever relaxing the assertions."),
        ("smoke", "Make smoke pass instead of outright softening the ground truth."),
    ],
)
def test_case_action_keeps_advice_whose_negator_governs_the_verb_through_modifiers(
    tmp_path, monkeypatch, case_id, action
):
    from skillevaluator.tier3.eval_core import llm_judge

    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: (json.dumps([action]), None))
    [fix] = what_to_fix({"agents": {"a": _agent({case_id: _case(False)}, {})}}, run_dir=tmp_path)

    assert fix["action"] == action
    assert fix["source"] == "llm"
    assert "fallback_reason" not in fix


@pytest.mark.parametrize(
    "case_id,action",
    [
        ("smoke", "Run without failures by deleting the smoke case from evals/evals.json."),
        ("smoke", "Run without errors by removing the smoke case from evals/evals.json."),
        ("positive-001", "Pass without errors by removing the positive-001 case from evals/evals.json."),
        ("smoke", "Pass without regressions by dropping the smoke case."),
        ("smoke", "Fix the parser without delay, deleting the smoke case."),
        ("smoke", "Fix the parser without having to wait, delete the smoke case."),
        ("smoke", "Fix the parser without ever fixing the parser by deleting the smoke case."),
        ("smoke", "Fix the parser without first fixing the parser. Delete the smoke case."),
        ("smoke", "Get smoke to pass without failures by lowering the pass threshold."),
        ("smoke", "Get smoke to pass without errors by loosening the expected output."),
        ("smoke", "Get smoke to pass without first fixing the parser, then relax the assertions."),
    ],
)
def test_case_action_rejects_a_removal_that_a_negator_with_modifiers_does_not_govern(
    tmp_path, monkeypatch, case_id, action
):
    from skillevaluator.tier3.eval_core import llm_judge

    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: (json.dumps([action]), None))
    [fix] = what_to_fix({"agents": {"a": _agent({case_id: _case(False)}, {})}}, run_dir=tmp_path)

    assert fix["action"] == "Address in SKILL.md: Missing syllable_count"
    assert fix["fallback_reason"] == "requirement_weakening"


@pytest.mark.parametrize("known", [("smoke",), ()])
def test_a_negator_and_its_verb_modifiers_negate_the_removal_and_the_weakening(known):
    subject = "the smoke case" if known else "the case for positive-001"
    offenders = [s for s in _negated_advice(_ALL_NEGATORS, subject) if _weakens_requirements(s, known)]
    assert offenders == []


@pytest.mark.parametrize("known", [("smoke",), ()])
def test_a_noun_between_the_negator_and_the_verb_breaks_the_negation(known):
    subject = "the smoke case from evals/evals.json" if known else "the case for positive-001 from evals/evals.json"
    missed = [s for s in _unrelated_negation_advice(_STRICT_NEGATORS, subject) if not _weakens_requirements(s, known)]
    assert missed == []


@pytest.mark.parametrize(
    "case_id,action",
    [
        ("smoke", "Make smoke pass by not only lowering the pass threshold but also fixing the parser."),
        ("smoke", "Not only delete the smoke case but also fix the parser."),
        ("smoke", "Make smoke pass by not just lowering the pass threshold but also fixing the parser."),
        ("smoke", "Make smoke pass by not merely relaxing the assertions but also fixing the parser."),
        ("smoke", "Make smoke pass by not simply loosening the expected output but also fixing the parser."),
        ("smoke", "Make smoke pass by not solely lowering the pass threshold but also fixing the parser."),
        ("smoke", "Make smoke pass by not exclusively softening the ground truth but also fixing the parser."),
        ("smoke", "Make smoke pass by not necessarily just lowering the pass threshold but also fixing the parser."),
        ("smoke", "Make smoke pass by not necessarily only lowering the pass threshold but also fixing the parser."),
        ("smoke", "Fix the parser by not only dropping the smoke case but also adding a unit test."),
        ("smoke", "Not just removing the smoke case but also fixing the parser."),
        ("smoke", "Fix the parser by not merely excluding the smoke case but also adding a unit test."),
        ("positive-001", "Not simply delete the case for positive-001 but also fix the parser."),
        ("smoke", "Do not only lower the pass threshold for smoke but also fix the parser."),
        ("smoke", "Do not just delete the smoke case, fix the parser."),
        ("smoke", "Don't simply delete the smoke case; fix the parser."),
    ],
)
def test_case_action_rejects_additive_wording_that_still_recommends_the_removal(tmp_path, monkeypatch, case_id, action):
    from skillevaluator.tier3.eval_core import llm_judge

    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: (json.dumps([action]), None))
    [fix] = what_to_fix({"agents": {"a": _agent({case_id: _case(False)}, {})}}, run_dir=tmp_path)

    assert fix["action"] == "Address in SKILL.md: Missing syllable_count"
    assert fix["fallback_reason"] == "requirement_weakening"


@pytest.mark.parametrize(
    "case_id,action",
    [
        ("smoke", "You should not lower the pass threshold for smoke; fix the parser."),
        ("smoke", "You must not delete the smoke case; fix the parser."),
        ("smoke", "Do not lower the pass threshold for smoke."),
        ("smoke", "Do not delete the smoke case but also fix the parser."),
        ("smoke", "Fix the parser and do not delete the smoke case."),
        ("smoke", "Fix the parser; it should not drop the smoke case."),
        ("smoke", "Fix the parser without simply removing the smoke case."),
        ("smoke", "Fix the parser instead of just dropping the smoke case."),
        ("smoke", "Fix the parser rather than merely excluding the smoke case."),
    ],
)
def test_case_action_keeps_genuine_prohibitions_next_to_additive_wording(tmp_path, monkeypatch, case_id, action):
    from skillevaluator.tier3.eval_core import llm_judge

    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: (json.dumps([action]), None))
    [fix] = what_to_fix({"agents": {"a": _agent({case_id: _case(False)}, {})}}, run_dir=tmp_path)

    assert fix["action"] == action
    assert fix["source"] == "llm"
    assert "fallback_reason" not in fix


@pytest.mark.parametrize("known", [("smoke",), ()])
def test_additive_not_only_wording_does_not_negate_the_removal_or_the_weakening(known):
    subject = "the smoke case" if known else "the case for positive-001"
    missed = [s for s in _additive_advice(subject) if not _weakens_requirements(s, known)]
    assert missed == []


@pytest.mark.parametrize(
    "action,error,expected_source,expected_fallback",
    [
        ("Implement syllable_count in analyze_text.py.", None, "llm", None),
        ("Do not remove the expectation; implement it in analyze_text.py.", None, "llm", None),
        ("Do not remove the `expectation`; implement it in analyze_text.py.", None, "llm", None),
        ("Never drop the `expected output` for syllable_count; implement it in analyze_text.py.", None, "llm", None),
        ("Remove the expected output mismatch by implementing syllable_count in analyze_text.py.", None, "llm", None),
        ("Remove the assertion failure by implementing syllable_count in analyze_text.py.", None, "llm", None),
        ("Remove the `expected output` mismatch by implementing syllable_count in analyze_text.py.", None, "llm", None),
        (
            "The requirement violation should be removed by implementing syllable_count in analyze_text.py.",
            None,
            "llm",
            None,
        ),
        ("Remove temporary files after analysis.", None, "llm", None),
        ("Do not delete the case; implement syllable_count in analyze_text.py.", None, "llm", None),
        (
            "Never remove the syllable-positive entry from evals/evals.json; fix analyze_text.py instead.",
            None,
            "llm",
            None,
        ),
        (
            "Remove the expected output mismatch by implementing syllable_count in analyze_text.py; "
            "keep the syllable-positive case in evals/evals.json.",
            None,
            "llm",
            None,
        ),
        ("In this case, implement syllable_count in analyze_text.py.", None, "llm", None),
        ("The edge case should be handled in analyze_text.py.", None, "llm", None),
        ("Delete the cache entry, then implement syllable_count in analyze_text.py.", None, "llm", None),
        ("Remove temp files per case, then implement syllable_count in analyze_text.py.", None, "llm", None),
        (
            "Remove the special case in parser.py for positive-001 and implement the general rounding algorithm.",
            None,
            "llm",
            None,
        ),
        (
            "Remove the test case failure for positive-001 by implementing the rounding rule in SKILL.md",
            None,
            "llm",
            None,
        ),
        ("Remove the special case in the script and implement syllable_count generally.", None, "llm", None),
        ("Delete the default case branch in analyze_text.py; implement syllable_count.", None, "llm", None),
        ("Remove the case mismatch by implementing syllable_count in analyze_text.py.", None, "llm", None),
        ("Remove the test case violation by implementing syllable_count in analyze_text.py.", None, "llm", None),
        (
            "The test case failure should be removed by implementing syllable_count in analyze_text.py.",
            None,
            "llm",
            None,
        ),
        ("Never remove the test case; implement syllable_count in analyze_text.py.", None, "llm", None),
        (None, "Connection error", "deterministic", "llm_unavailable_or_invalid"),
        ("", None, "deterministic", "llm_unavailable_or_invalid"),
    ],
)
def test_case_action_provenance_reaches_canonical_recommendation(
    tmp_path, monkeypatch, action, error, expected_source, expected_fallback
):
    from skillevaluator.evaluation.tier3_report import _apply_case_findings
    from skillevaluator.tier3.eval_core import llm_judge

    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    monkeypatch.setattr(llm_judge, "call_public_llm", lambda *_a, **_k: (json.dumps([action]), error))
    source = {"agents": {"a": _agent({"syllable-positive": _case(False)}, {"syllable-positive": _case(False)})}}
    payload = {}
    _apply_case_findings(payload, source, tmp_path, tmp_path, use_llm=True)
    [fix] = json.loads((tmp_path / "what_to_fix.json").read_text())
    [recommendation] = payload["recommendations"]

    for record in (fix, recommendation):
        assert record["source"] == expected_source
        assert record.get("fallback_reason") == expected_fallback
    assert fix["action"] in recommendation["message"]


def test_case_action_disabled_is_not_a_failed_llm_fallback(tmp_path, monkeypatch):
    from skillevaluator.tier3.eval_core import llm_judge

    def unexpected(*args, **kwargs):
        pytest.fail("LLM should not be called")

    monkeypatch.setattr(llm_judge, "call_public_llm", unexpected)
    _trial(tmp_path, "a", "t0", _reward(True, "Missing syllable_count"))
    [fix] = what_to_fix({"agents": {"a": _agent({"c": _case(False)}, {})}}, run_dir=tmp_path, use_llm=False)
    assert fix["source"] == "deterministic"
    assert "fallback_reason" not in fix


def test_case_action_deterministic_remedy_does_not_repeat_weakening_reason(tmp_path):
    _trial(tmp_path, "a", "t0", _reward(True, "Remove the expectation for syllable_count"))
    [fix] = what_to_fix({"agents": {"a": _agent({"c": _case(False)}, {})}}, run_dir=tmp_path, use_llm=False)
    assert "Remove the expectation" not in fix["action"]


@pytest.mark.parametrize("source", ["result", "saved", "provenance", "variant", "other_variant", "default"])
def test_incomplete_reconstruction_preserves_configured_threshold(tmp_path: Path, source: str) -> None:
    from skillevaluator.tier3.case_summary import _cases

    _trial(tmp_path, "a", "strict__one", {"overall": 0.95, "goal_accuracy": 1.0, "skill_invoked": True})
    agent = _agent({}, {})
    agent.update(output_dir=str(tmp_path / "a"), execution_status="failed")
    result = {"agents": {"a": agent}}
    policy = {"pass_threshold": 0.99}
    if source == "result":
        result["attempt_policy"] = policy
    elif source == "saved":
        (tmp_path / "result.json").write_text(json.dumps({"attempt_policy": policy}))
    elif source == "provenance":
        (tmp_path / "result.json").write_text(json.dumps({"provenance": {"attempt_policy": policy}}))
    elif source in {"variant", "other_variant"}:
        agent["pass_at_k"]["with_skill" if source == "variant" else "without_skill"]["pass_threshold"] = 0.99

    fixes = what_to_fix(result, run_dir=tmp_path, use_llm=False)

    assert bool(fixes) is (source != "default")
    if fixes:
        assert (fixes[0]["case"], fixes[0]["kind"], fixes[0]["attempts_lost"]) == ("strict", "skill", 1)
    if source != "result":
        assert _cases(agent, "with_skill")["strict"]["attempts"][0]["passed"] is (source == "default")


def test_incomplete_reconstruction_without_saved_result_keeps_default(tmp_path: Path) -> None:
    _trial(tmp_path, "a", "ordinary__one", {"overall": 0.2, "goal_accuracy": 1.0, "skill_invoked": True})
    agent = dict(_agent({}, {}), output_dir=str(tmp_path / "a"), execution_status="failed")
    (tmp_path / "result.json").write_text("invalid JSON")

    [fix] = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)

    assert (fix["case"], fix["kind"]) == ("ordinary", "skill")


def test_grid_shows_one_row_per_case_with_attempt_marks_per_agent() -> None:
    result = {
        "execution_status": "succeeded",
        "agents": {
            "claude-code": _agent({"c1": _case(True, True), "c2": _case(False, True)}, {"c1": _case(False, False)}),
            "codex": _agent({"c1": _case(True, False), "c2": _case(False, missing=1)}, {"c2": _case(True, True)}),
        },
    }
    output = render_result(result)
    rows = {line.split()[0]: line for line in output.splitlines() if line.startswith("c")}
    lines = output.splitlines()
    header = lines[next(i for i, line in enumerate(lines) if line.startswith("test case")) - 1]
    assert header.split() == ["claude-code", "codex"]
    assert "✗✗ → ✓✓  +100" in rows["c1"] and "- → ✓✗  ?" in rows["c1"]
    assert "- → ✗✓  ?" in rows["c2"] and "✓✓ → ✗⊘  ?" in rows["c2"]
    rates = next(line for line in output.splitlines() if line.startswith("success rate"))
    assert "0% → 75%  ?" in rates and rates.rstrip().endswith("?")
    assert output.index("effectiveness") < output.index("security")
    assert output.count("(without → with skill)") == 2 and "change in per-attempt success rate" in output
    assert "inconclusive" in output and "conditions cover different cases" in output


def test_multi_agent_diagnostics_are_arm_labeled_after_rate_change() -> None:
    from skillevaluator.tier3.case_summary import _agent_cell

    with_case = {
        "attempts": [{"attempt": 1, "trial": "case__attempt1", "score": 0.2, "passed": False, "reason": "goal missed"}]
    }
    without_case = {
        "attempts": [
            {"attempt": 1, "trial": "case__attempt1", "score": 0.2, "passed": False, "reason": "security issue"}
        ]
    }
    output = _agent_cell(with_case, without_case, safe=str).plain

    assert "FAIL: security issue" in output
    assert "FAIL: goal missed" in output
    assert output.index("No skill:") < output.index("FAIL: security issue")
    assert output.index("FAIL: security issue") < output.index("With skill:")
    assert output.index("With skill:") < output.index("FAIL: goal missed")
    assert output.index("·") < output.index("No skill:")


def test_verdict_thresholds() -> None:
    helps = {"a": _agent({"c": _case(True, True)}, {"c": _case(False, False)})}
    flat = {"a": _agent({"c": _case(True, False)}, {"c": _case(False, True)})}
    hurts = {"a": _agent({"c": _case(False, False)}, {"c": _case(True, True)})}
    assert verdict(helps) == ("skill helps", "a +100 pp")
    assert verdict(flat)[0] == "no clear effect"
    assert verdict(hurts)[0] == "skill hurts"
    partial = {"a": _agent({"c": _case(True, missing=1)}, {"c": _case(False, False)})}
    assert verdict(partial) == ("inconclusive", "a: 1 attempt(s) did not run")


def test_live_mark_merges_verifier_security_sidecar(tmp_path: Path) -> None:
    from skillevaluator.tier3.case_summary import _live_mark

    trial = tmp_path / "trial"
    verifier = trial / "verifier"
    verifier.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "finished_at": "now",
                "verifier_result": {"rewards": {"overall": 0.9, "security": 0.0, "goal_accuracy": 1.0}},
            }
        )
    )
    (verifier / "skill_evaluator_reward.json").write_text(
        json.dumps(
            {
                "details": {
                    "security": {
                        "supported": False,
                        "findings": [
                            {
                                "type": "unsupported_tool_wrapper",
                                "score_impact": True,
                                "message": "unsupported wrapper",
                            }
                        ],
                    }
                }
            }
        )
    )

    assert _live_mark(trial, tmp_path, 0.5) == "✓"


def test_what_to_fix_classifies_and_ranks_by_attempts_lost(tmp_path: Path) -> None:
    agents = {
        "opencode": _agent(
            {
                "unused": _case(False, False, prefix="u"),
                "env": _case(False, True, prefix="e"),
                "hurts": _case(True, False, prefix="h"),
                "broken": _case(missing=2),
                "fine": _case(True, True, prefix="f"),
            },
            {"hurts": _case(True, True), "unused": _case(False, False)},
        )
    }
    for name in ("u0", "u1"):
        _trial(tmp_path, "opencode", name, _reward(False, "never read SKILL.md"))
    _trial(tmp_path, "opencode", "e0", _reward(True, "no output"), "ModuleNotFoundError: No module named 'pypdf'")
    _trial(tmp_path, "opencode", "h1", _reward(True, "followed the wrong script"))
    for name in ("broken__a", "broken__b"):
        (tmp_path / "opencode" / "with-skill" / "trials" / name).mkdir(parents=True)
        (tmp_path / "opencode" / "with-skill" / "trials" / name / "failure.json").write_text(
            json.dumps({"error": {"type": "NonZeroAgentExitCodeError", "message": "EPIPE"}})
        )

    fixes = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert [(f["case"], f["kind"], f["attempts_lost"]) for f in fixes] == [
        ("unused", "skill not used", 2),
        ("env", "environment", 1),
        ("hurts", "skill hurts", 1),
    ]
    assert "No module named 'pypdf'" in fixes[1]["what"]
    assert fixes[0]["evidence"] == "opencode/with-skill/trials/u0/trajectory.json"
    assert all(f["action"] for f in fixes)


def test_run_errors_rank_after_skill_issues_and_are_not_blamed_on_the_skill(tmp_path: Path) -> None:
    agents = {"opencode": _agent({"broken": _case(missing=2)}, {})}
    trial = tmp_path / "opencode" / "with-skill" / "trials" / "broken__a"
    trial.mkdir(parents=True)
    (trial / "failure.json").write_text(json.dumps({"error": {"type": "AgentTimeoutError", "message": "900s"}}))

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert fix["kind"] == "run error"
    assert "AgentTimeoutError" in fix["what"]
    assert fix["action"].startswith("Not a skill problem")
    assert fix["evidence"] == "opencode/with-skill/trials/broken__a/failure.json"


@pytest.mark.parametrize("should_trigger", [True, False])
@pytest.mark.parametrize("status", ["failed", "error"])
@pytest.mark.parametrize("record_failure", [True, False])
def test_judge_failures_are_unscored_infrastructure_advice(
    tmp_path: Path, should_trigger: bool, status: str, record_failure: bool
) -> None:
    from skillevaluator.tier3.case_summary import ERROR, _cases, _marks

    _trial(
        tmp_path,
        "a",
        "c__1",
        {"overall": 0.0, "evaluation_status": status, "evaluation_errors": {"accuracy": "Judge reply was empty"}},
    )
    (tmp_path / "dataset_snapshot.json").write_text(
        json.dumps({"dataset": [{"id": "c", "should_trigger": should_trigger}]})
    )
    agent = _agent({}, {})
    agent["output_dir"] = str(tmp_path / "a")
    agent["skill_activation"] = {"status": "never_activated"}
    if record_failure:
        agent["trial_failures"] = {"with_skill": [{"trial": "c__1", "reason": "Required judge evaluation failed"}]}

    assert _marks(_cases(agent, "with_skill")["c"]) == [ERROR]
    [fix] = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)
    assert fix["kind"] == "run error"
    assert fix["attempts_lost"] == 1
    assert "judge" in fix["action"].lower()
    assert "Not a skill problem" in fix["action"]
    assert "scored below" not in fix["what"]
    assert fix["evidence"].endswith("reward.json")


def test_judge_failure_does_not_change_genuine_scored_failure_advice(tmp_path: Path) -> None:
    _trial(
        tmp_path,
        "a",
        "c__1",
        {"overall": 0.0, "evaluation_status": "failed", "evaluation_errors": {"accuracy": "Judge unavailable"}},
    )
    _trial(tmp_path, "a", "c__2", {"overall": 0.2, **_reward(True, "omitted validation")})
    agent = _agent({}, {})
    agent["output_dir"] = str(tmp_path / "a")
    agent["trial_failures"] = {"with_skill": [{"trial": "c__1", "reason": "Required judge evaluation failed"}]}

    fixes = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)
    assert [(f["kind"], f["attempts_lost"]) for f in fixes] == [("skill", 1), ("run error", 1)]
    assert "omitted validation" in fixes[0]["action"]
    assert "judge" in fixes[1]["action"].lower()


@pytest.mark.parametrize(
    ("errors", "reason", "action_text"),
    [
        ({"environment": "required dependency unavailable"}, "required dependency unavailable", "environment"),
        ({"collector": "Judge sidecar could not be read; trial was not scored"}, "could not be read", "verifier"),
    ],
)
def test_environment_and_collector_failures_do_not_get_judge_advice(
    tmp_path: Path, errors: dict, reason: str, action_text: str
) -> None:
    _trial(tmp_path, "a", "c__1", {"overall": 0.0, "evaluation_status": "failed", "evaluation_errors": errors})
    agent = _agent({}, {})
    agent["output_dir"] = str(tmp_path / "a")
    agent["trial_failures"] = {"with_skill": [{"trial": "c__1", "reason": "Evaluation failed"}]}

    [fix] = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)

    assert (fix["kind"], fix["attempts_lost"]) == ("run error", 1)
    assert reason in fix["what"]
    assert "required judge evaluation failed" not in fix["what"]
    assert fix["action"].startswith("Not a skill problem")
    assert action_text in fix["action"]
    assert "judge" not in fix["action"].lower() and "credentials" not in fix["action"]
    assert fix["evidence"].endswith("c__1/reward.json")


def test_runtime_and_judge_failures_in_one_case_keep_their_own_causes_and_remedies(tmp_path: Path) -> None:
    _trial(
        tmp_path,
        "a",
        "c__judge",
        {"overall": 0.0, "evaluation_status": "failed", "evaluation_errors": {"accuracy": "Judge reply was empty"}},
    )
    timeout = tmp_path / "a" / "with-skill" / "trials" / "c__timeout"
    timeout.mkdir(parents=True)
    (timeout / "failure.json").write_text(json.dumps({"error": {"type": "AgentTimeoutError", "message": "900s"}}))
    agent = _agent({}, {})
    agent["output_dir"] = str(tmp_path / "a")
    agent["trial_failures"] = {
        "with_skill": [
            {"trial": "c__timeout", "reason": "Agent timed out"},
            {"trial": "c__judge", "reason": "Required judge evaluation failed"},
        ]
    }

    fixes = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)

    assert [(f["kind"], f["attempts_lost"]) for f in fixes] == [("run error", 1), ("run error", 1)]
    timeout_fix, judge_fix = fixes
    assert "AgentTimeoutError" in timeout_fix["what"] and "judge" not in timeout_fix["what"].lower()
    assert "agent/gateway logs" in timeout_fix["action"]
    assert timeout_fix["evidence"].endswith("c__timeout/failure.json")
    assert "Judge reply was empty" in judge_fix["what"]
    assert "judge model configuration" in judge_fix["action"]
    assert judge_fix["evidence"].endswith("c__judge/reward.json")


def test_run_errors_with_different_causes_stay_separate_across_agents(tmp_path: Path) -> None:
    timeout = tmp_path / "a" / "with-skill" / "trials" / "c__timeout"
    timeout.mkdir(parents=True)
    (timeout / "failure.json").write_text(json.dumps({"error": {"type": "AgentTimeoutError", "message": "900s"}}))
    judge_failed = {
        "overall": 0.0,
        "evaluation_status": "failed",
        "evaluation_errors": {"accuracy": "Judge unavailable"},
    }
    for name in ("b", "c"):
        _trial(tmp_path, name, "c__1", judge_failed)
    agents = {}
    for name, trial, reason in (
        ("a", "c__timeout", "Agent timed out"),
        ("b", "c__1", "Required judge evaluation failed"),
        ("c", "c__1", "Required judge evaluation failed"),
    ):
        agents[name] = _agent({}, {})
        agents[name]["output_dir"] = str(tmp_path / name)
        agents[name]["trial_failures"] = {"with_skill": [{"trial": trial, "reason": reason}]}

    judge_fix, timeout_fix = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert (timeout_fix["scope"], timeout_fix["attempts_lost"]) == ("a only", 1)
    assert "AgentTimeoutError" in timeout_fix["what"] and "agent/gateway logs" in timeout_fix["action"]
    assert (judge_fix["scope"], judge_fix["attempts_lost"]) == ("b, c only", 2)
    assert "judge" in judge_fix["what"].lower() and "judge model configuration" in judge_fix["action"]


def test_multi_agent_issue_scope_names_the_failing_agent(tmp_path: Path) -> None:
    agents = {
        "claude-code": _agent({"c": _case(True, True)}, {}),
        "codex": _agent({"c": _case(False, False, prefix="x")}, {}),
    }
    for name in ("x0", "x1"):
        _trial(tmp_path, "codex", name, _reward(True, "skipped validation step"))

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert (fix["scope"], fix["kind"]) == ("codex only", "skill")
    assert "skipped validation step" in fix["what"]


def test_trial_names_cannot_escape_the_run_directory(tmp_path: Path) -> None:
    agents = {"opencode": _agent({"c": {"attempts": [{"trial": "../../etc", "passed": False}]}}, {})}
    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)
    assert fix["evidence"] == ""


def test_eval_prompt_that_restates_the_description_is_flagged_as_an_eval_problem(tmp_path: Path) -> None:
    description = "Use this skill whenever the user wants to create, read, edit, or manipulate Word documents."
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(f"---\nname: docx\ndescription: {description}\n---\n# Docx\n")
    run = tmp_path / "run"
    cases = {f"c{i}": _case(False, prefix=f"c{i}_") for i in (1, 2)}
    entries = [{"id": f"c{i}", "question": f"I need help: {description.lower()}"} for i in (1, 2)]
    run.mkdir()
    (run / "dataset_snapshot.json").write_text(json.dumps({"dataset": entries}))
    for i in (1, 2):
        _trial(run, "claude-code", f"c{i}_0", _reward(False, "asked what to do"))

    [fix] = what_to_fix({"agents": {"claude-code": _agent(cases, {})}}, run_dir=run, skill_path=skill, use_llm=False)

    assert (fix["case"], fix["kind"], fix["attempts_lost"]) == ("all 2 cases", "eval case", 2)
    assert fix["action"].startswith("Not a skill problem: rewrite this prompt")


def test_setup_command_failure_reports_its_stdout_and_merges_by_error_type(tmp_path: Path) -> None:
    agents = {"claude-code": _agent({"a": _case(missing=1), "b": _case(missing=1)}, {})}
    for case, code in (("a", 35), ("b", 56)):
        trial = tmp_path / "claude-code" / "with-skill" / "trials" / f"{case}__x"
        trial.mkdir(parents=True)
        message = f"Command failed (exit 1): set -euo pipefail; curl -fsSL https://x | bash stdout: curl: ({code}) Recv failure  stderr: None"
        (trial / "failure.json").write_text(
            json.dumps({"error": {"type": "NetworkConnectionError", "message": message}})
        )

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert fix["case"] == "all 2 cases"
    assert fix["what"] == "2 attempt(s) never produced a score: NetworkConnectionError: curl: (35) Recv failure"


def test_failed_condition_rebuilds_case_rows_from_trial_rewards(tmp_path: Path) -> None:
    trials = tmp_path / "claude-code" / "with-skill" / "trials"
    for name, overall in (("ok__1", 0.9), ("low__1", 0.3)):
        (trials / name).mkdir(parents=True)
        (trials / name / "reward.json").write_text(json.dumps({"overall": overall}))
    (trials / "gone__1").mkdir()
    agent = _agent({}, {"ok": _case(False)})
    agent["pass_at_k"]["without_skill"]["pass_threshold"] = 0.5
    agent.update(output_dir=str(tmp_path / "claude-code"), trial_failures={"with_skill": [{"trial": "gone__1"}]})

    output = render_result({"execution_status": "failed", "agents": {"claude-code": agent}})

    assert "1/1 ✓ (0.90)" in output and "0/1 ✗ (0.30)" in output and "⊘" in output


# Harbor names a trial after the first 32 characters of its task directory, so a trial's name need not
# start with its case: a native task keeps its own directory name, and a longer case ID is cut.
_LONG_CASE = "convert-quarterly-report-to-markdown-table"
_TIMEOUT = {"error": {"type": "AgentTimeoutError", "message": "900s"}}
_JUDGE_FAILED = {
    "overall": 0.0,
    "evaluation_status": "failed",
    "evaluation_errors": {"accuracy": "Judge reply was empty"},
}


def _trial_records(trials: Path, name: str, records: dict[str, dict]) -> Path:
    trial = trials / name
    trial.mkdir(parents=True)
    for file_name, payload in records.items():
        (trial / file_name).write_text(json.dumps(payload))
    return trial


@pytest.mark.parametrize(
    ("case_id", "trial_name", "records"),
    [
        ("case-alpha", "native-task-a__AbC1234", {"failure.json": {**_TIMEOUT, "entry_id": "case-alpha"}}),
        (_LONG_CASE, f"{_LONG_CASE[:32]}__AbC1234", {"failure.json": {**_TIMEOUT, "entry_id": _LONG_CASE}}),
        # A run from before failure.json named the case: the trial's config names the whole task directory.
        (
            _LONG_CASE,
            f"{_LONG_CASE[:32]}__AbC1234",
            {"failure.json": _TIMEOUT, "config.json": {"task": {"path": f"/run/_harbor-tasks/a/with/{_LONG_CASE}"}}},
        ),
    ],
    ids=["native-task", "long-case-id", "long-case-id-without-recorded-case"],
)
def test_run_error_is_found_for_a_trial_whose_name_does_not_start_with_its_case(
    tmp_path: Path, case_id: str, trial_name: str, records: dict
) -> None:
    _trial_records(tmp_path / "a" / "with-skill" / "trials", trial_name, records)

    [fix] = what_to_fix({"agents": {"a": _agent({case_id: _case(missing=1)}, {})}}, run_dir=tmp_path, use_llm=False)

    assert (fix["case"], fix["kind"]) == (case_id, "run error")
    assert fix["what"] == "1 attempt(s) never produced a score: AgentTimeoutError: 900s"
    assert fix["evidence"] == f"a/with-skill/trials/{trial_name}/failure.json"


def test_judge_failure_of_a_native_task_gets_judge_advice_under_the_case_the_collector_recorded(
    tmp_path: Path,
) -> None:
    # The judge's own entry_id is not authoritative; the collector's record in failure.json is.
    _trial_records(
        tmp_path / "a" / "with-skill" / "trials",
        "native-task-a__AbC1234",
        {
            "reward.json": {**_JUDGE_FAILED, "entry_id": "native-task-a"},
            "failure.json": {"status": "unscored", "evaluation_status": "failed", "entry_id": "case-alpha"},
        },
    )

    [fix] = what_to_fix(
        {"agents": {"a": _agent({"case-alpha": _case(missing=1)}, {})}}, run_dir=tmp_path, use_llm=False
    )

    assert (fix["case"], fix["kind"], fix.get("cause")) == ("case-alpha", "run error", "judge")
    assert "Judge reply was empty" in fix["what"]
    assert "judge model configuration" in fix["action"]
    assert fix["evidence"] == "a/with-skill/trials/native-task-a__AbC1234/reward.json"


def test_failed_condition_rebuilds_native_task_rows_under_the_case_the_collector_recorded(tmp_path: Path) -> None:
    from skillevaluator.tier3.case_summary import ERROR, PASS, _cases, _marks

    trials = tmp_path / "a" / "with-skill" / "trials"
    _trial_records(trials, "native-task-a__pass", {"reward.json": {"overall": 0.9, "entry_id": "case-alpha"}})
    _trial_records(
        trials,
        "native-task-a__judge",
        {"reward.json": _JUDGE_FAILED, "failure.json": {"evaluation_status": "failed", "entry_id": "case-alpha"}},
    )
    _trial_records(trials, "native-task-a__timeout", {"failure.json": {**_TIMEOUT, "entry_id": "case-alpha"}})
    agent = _agent({}, {})
    agent.update(
        output_dir=str(tmp_path / "a"),
        trial_failures={
            "with_skill": [
                {"trial": "native-task-a__judge", "reason": "Required judge evaluation failed"},
                {"trial": "native-task-a__timeout", "reason": "Agent timed out"},
            ]
        },
    )

    assert {case: _marks(row) for case, row in _cases(agent, "with_skill").items()} == {
        "case-alpha": [PASS, ERROR, ERROR]
    }
    fixes = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)
    assert sorted((f["case"], f["kind"], f.get("cause") or "") for f in fixes) == [
        ("case-alpha", "run error", ""),
        ("case-alpha", "run error", "judge"),
    ]


def test_failed_condition_rows_are_rebuilt_once_per_what_to_fix_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3 import case_summary

    # A failed condition with several cases, each with one scored attempt and one that timed out.
    trials = tmp_path / "a" / "with-skill" / "trials"
    failed = []
    for index in range(4):
        case_id = f"case-{index}"
        _trial_records(trials, f"native-{index}__pass", {"reward.json": {"overall": 0.9, "entry_id": case_id}})
        _trial_records(trials, f"native-{index}__timeout", {"failure.json": {**_TIMEOUT, "entry_id": case_id}})
        failed.append({"trial": f"native-{index}__timeout", "reason": "Agent timed out"})
    agent = _agent({}, {})
    agent.update(output_dir=str(tmp_path / "a"), trial_failures={"with_skill": failed})
    rebuilds: list[str] = []
    rebuild = case_summary._rebuilt_cases

    def counted(agent: object, key: str) -> object:
        rebuilds.append(key)
        return rebuild(agent, key)

    monkeypatch.setattr(case_summary, "_rebuilt_cases", counted)

    fixes = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)

    # Rebuilding reads every trial's files, so each failed condition is rebuilt once for the whole
    # call rather than again for each of its cases.
    assert [fix["case"] for fix in fixes] == ["all 4 cases"]
    assert sorted(rebuilds) == ["with_skill", "without_skill"]
    # The next call reads the trials again: they can change between calls.
    what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)
    assert sorted(rebuilds) == ["with_skill", "with_skill", "without_skill", "without_skill"]


@pytest.mark.parametrize(
    "records",
    [{}, {"config.json": {"task": {"path": "/run/_harbor-tasks/a/with/plain-case"}}}],
    ids=["no-harbor-files", "config-written-at-trial-start"],
)
def test_trial_whose_reward_identity_the_collector_rejected_keeps_its_own_case(tmp_path: Path, records: dict) -> None:
    from skillevaluator.tier3.case_summary import _cases
    from skillevaluator.tier3.harbor.collector import UNSAFE_REWARD_IDENTITY_REASON

    # A trial stopped before Harbor wrote result.json names no task the collector can map to a case, so
    # the collector rejects its reward and records the placeholder entry_id "unknown", which is no case.
    rejected = {
        "overall": 0.9,
        "evaluation_status": "failed",
        "evaluation_errors": {"collector": UNSAFE_REWARD_IDENTITY_REASON},
        "entry_id": "unknown",
    }
    trial = "plain-case__Zx81kQa"
    _trial_records(tmp_path / "a" / "with-skill" / "trials", trial, {"reward.json": rejected, **records})
    failed_arm = _agent({}, {})
    failed_arm.update(
        output_dir=str(tmp_path / "a"),
        trial_failures={"with_skill": [{"trial": trial, "reason": "Required judge evaluation failed"}]},
    )
    scored_arm = _agent({"plain-case": _case(missing=1)}, {})

    assert set(_cases(failed_arm, "with_skill")) == {"plain-case"}
    for agent in (failed_arm, scored_arm):
        [fix] = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)
        assert (fix["case"], fix["kind"], fix.get("cause")) == ("plain-case", "run error", "collector")
        assert fix["evidence"] == f"a/with-skill/trials/{trial}/reward.json"


def test_recorded_case_is_not_read_through_a_link(tmp_path: Path) -> None:
    from skillevaluator.tier3.case_summary import _cases

    # A link to a trial directory, or to its failure.json, must not move the trial to another case.
    outside = tmp_path / "outside"
    _trial_records(outside, "dir", {"failure.json": {**_TIMEOUT, "entry_id": "outside-case"}})
    (outside / "failure.json").write_text(json.dumps({"entry_id": "outside-case"}))
    trials = tmp_path / "run" / "a" / "with-skill" / "trials"
    judged = _trial_records(trials, "t__judge", {"reward.json": _JUDGE_FAILED})
    (judged / "failure.json").symlink_to(outside / "failure.json")
    (trials / "t__link").symlink_to(outside / "dir", target_is_directory=True)
    agent = _agent({}, {})
    agent.update(output_dir=str(tmp_path / "run" / "a"), trial_failures={"with_skill": [{"trial": "t__link"}]})

    assert set(_cases(agent, "with_skill")) == {"t"}


def test_live_marks_key_a_trial_by_its_whole_task_directory(tmp_path: Path) -> None:
    from skillevaluator.tier3.case_summary import PASS, live_case_marks

    staged = tmp_path / "_harbor-tasks" / "a" / "with" / _LONG_CASE
    staged.mkdir(parents=True)
    jobs = tmp_path / "_harbor-jobs" / "job-a-with"
    done = {"finished_at": "t", "verifier_result": {"rewards": {"overall": 0.9}}}
    config = {"task": {"path": str(staged)}}
    _trial_records(jobs, f"{_LONG_CASE[:32]}__AbC1234", {"config.json": config, "result.json": done})
    # Harbor writes config.json as the trial starts, before any result.
    _trial_records(jobs, f"{_LONG_CASE[:32]}__DeF5678", {"config.json": config})

    marks, running = live_case_marks(tmp_path, ["a"])

    assert marks == {"a": {"with_skill": {_LONG_CASE: [PASS]}}}
    assert running == {("a", "with_skill", _LONG_CASE)}


def test_agent_asking_for_files_the_case_lacks_is_flagged_as_missing_input(tmp_path: Path) -> None:
    agents = {"claude-code": _agent({"app": _case(False, False, prefix="m")}, {})}
    for name in ("m0", "m1"):
        trial = tmp_path / "claude-code" / "with-skill" / "trials" / name
        _trial(tmp_path, "claude-code", name, _reward(True, "did not run Playwright"))
        steps = [{"source": "agent", "message": "The workspace appears to be empty. Could you share your app?"}]
        (trial / "trajectory.json").write_text(json.dumps({"steps": steps}))

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert (fix["kind"], fix["attempts_lost"]) == ("missing input", 2)
    assert "workspace appears to be empty" in fix["what"]
    assert fix["action"].startswith("Not a skill problem: add the files")


def test_baseline_passing_every_case_is_flagged_as_unable_to_show_lift(tmp_path: Path) -> None:
    agents = {"claude-code": _agent({"a": _case(True, True)}, {"a": _case(True, True)})}

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert (fix["case"], fix["kind"], fix["attempts_lost"]) == ("all cases", "eval case", 0)
    assert fix["action"].startswith("Add cases the agent fails without the skill")
    console = Console(record=True, width=160)
    render_case_summary({"agents": agents}, console=console, fixes=[fix])
    assert "all cases  [eval case] · claude-code\n" in console.export_text()


def _scored(*scores: float) -> dict:
    return {"attempts": [{"attempt": i + 1, "score": s, "passed": s >= 0.5} for i, s in enumerate(scores)]}


def test_effect_column_names_what_the_skill_changed_per_case() -> None:
    with_cases = {
        "helps": _scored(0.8, 0.8),
        "hurts": _scored(0.3, 0.2),
        "both-fail": _scored(0.4, 0.3),
        "better": _scored(0.9, 0.9),
        "worse": _scored(0.6, 0.6),
        "same": _scored(0.9, 0.85),
        "partial": {**_scored(0.9), "attempts_missing": 1},
    }
    without_cases = {
        "helps": _scored(0.3, 0.6),
        "hurts": _scored(0.9, 0.9),
        "both-fail": _scored(0.2, 0.1),
        "better": _scored(0.6, 0.6),
        "worse": _scored(0.9, 0.8),
        "same": _scored(0.9, 0.9),
        "partial": _scored(0.9, 0.9),
    }
    output = render_result({"execution_status": "succeeded", "agents": {"a": _agent(with_cases, without_cases)}})
    effect = {
        line.split()[0]: line.rstrip().split("  ")[-1].strip() for line in output.splitlines() if line[:1].isalpha()
    }

    assert effect["helps"] == "helps"
    assert effect["hurts"] == "hurts"
    assert effect["both-fail"] == "both fail"
    assert effect["better"] == "better score (+0.30)"
    assert effect["worse"] == "worse score (-0.25)"
    assert effect["same"] == "no change"
    assert effect["partial"] == "inconclusive"
    assert "1/1 ✓⊘ (0.90)" in output and "(0.82) mean score 0-1" in output


def test_multi_agent_grid_wraps_instead_of_truncating_on_an_80_column_terminal() -> None:
    cases = {f"internal-comms-with-a-long-case-name-00{i}": _scored(0.81, 0.81) for i in (1, 2)}
    baseline = {f"internal-comms-with-a-long-case-name-00{i}": _scored(0.53, 0.2) for i in (1, 2)}
    console = Console(record=True, width=80)
    render_case_summary(
        {"agents": {"claude-code": _agent(cases, baseline), "codex": _agent(cases, baseline)}},
        console=console,
        fixes=[],
    )
    output = console.export_text()
    assert "…" not in output
    assert output.count("✓✗ → ✓✓  +50") == 4 and "codex" in output
    assert max(map(len, output.splitlines())) <= 80


def test_multi_agent_dimensions_read_without_then_with_skill() -> None:
    agent = _agent({"c1": _case(True, True)}, {"c1": _case(False, False)})
    console = Console(record=True, width=120)
    render_case_summary({"agents": {"a": agent, "b": agent}}, console=console, fixes=[])
    rows = {line.split()[0]: line for line in console.export_text().splitlines() if line.strip()}
    assert rows["effectiveness"].count("0.30 → 0.50  +0.20") == 2
    assert rows["security"].count("1.00 → 1.00  +0.00") == 2


def test_how_to_improve_says_nothing_to_fix_only_when_the_list_was_computed_and_empty() -> None:
    result = {"agents": {"a": _agent({"c1": _case(True, True)}, {"c1": _case(False, False)})}}
    for fixes, expected in (([], True), (None, False)):
        console = Console(record=True, width=120)
        render_case_summary(result, console=console, fixes=fixes)
        assert ("How to improve: nothing to fix in the available case evidence." in console.export_text()) is expected


def test_passing_attempts_that_miss_the_goal_are_reported_with_a_prompt_fix(tmp_path: Path) -> None:
    agents = {
        name: _agent({"c1": _case(True, True, prefix=f"{name}1-"), "c2": _case(True, prefix=f"{name}2-")}, {})
        for name in ("a", "b")
    }
    asked = "I need a few quick details: team name, dates, progress."
    for name in ("a", "b"):
        for trial in (f"{name}1-0", f"{name}1-1", f"{name}2-0"):
            reward = {"goal_accuracy": 0.0, "details": {"goal_accuracy": {"reason": "no update", "end_state": "asked"}}}
            _trial(tmp_path, name, trial, reward)
            steps = [{"source": "user", "message": "q"}, {"source": "agent", "message": asked}]
            (tmp_path / name / "with-skill" / "trials" / trial / "trajectory.json").write_text(
                json.dumps({"steps": steps})
            )
    _trial(tmp_path, "a", "unused", {"goal_accuracy": 1.0})

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert (fix["case"], fix["kind"], fix["scope"], fix["attempts_lost"]) == (
        "all 2 cases",
        "goal not met",
        "all agents",
        6,
    )
    assert fix["what"].endswith("goal accuracy 0.00: asked")
    assert fix["action"].startswith("The agent stopped to ask for input")


def test_goal_met_or_unscored_passes_are_not_flagged(tmp_path: Path) -> None:
    agents = {"a": _agent({"c1": _case(True, True)}, {})}
    _trial(tmp_path, "a", "t0", {"goal_accuracy": 0.6})
    _trial(tmp_path, "a", "t1", {})

    assert what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False) == []


def test_never_activated_scores_do_not_claim_a_skill_effect(tmp_path: Path) -> None:
    agent = _agent({"c": _case(False)}, {"c": _case(True)})
    agent["skill_activation"] = {"status": "never_activated"}
    _trial(tmp_path, "a", "t0", _reward(False, "the skill was not loaded"))
    assert verdict({"a": agent})[0] == "inconclusive"
    fixes = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)
    assert all(fix["kind"] != "skill hurts" for fix in fixes)
    output = render_result({"execution_status": "succeeded", "agents": {"a": agent}})
    assert "skill hurts" not in output
    assert "inconclusive" in output


@pytest.mark.parametrize(
    "negative_routing",
    [{"expected_skill": None}, {"should_trigger": False, "expected_skill": "release-review"}],
)
def test_mixed_activation_miss_does_not_blame_passing_negative(tmp_path: Path, negative_routing: dict) -> None:
    from skillevaluator.tier3.harbor.activation import summarize_skill_activation
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

    negative = {
        **dict.fromkeys(DEFAULT_METRICS, 1.0),
        "_trial_name": "n0",
        "details": {"skill_execution": {"should_trigger": False, "negative_check": {"passed": True}}},
    }
    positive = {
        **dict.fromkeys(DEFAULT_METRICS, 0.2),
        "_trial_name": "p0",
        "details": {"skill_execution": {"should_trigger": True, "activation": {"passed": False, "score": 0.0}}},
    }
    agent = _agent(
        {"negative": _case(True, prefix="n"), "positive": _case(False, prefix="p")},
        {"negative": _case(True), "positive": _case(False)},
    )
    agent["skill_activation"] = summarize_skill_activation([negative, positive])
    (tmp_path / "dataset_snapshot.json").write_text(
        json.dumps(
            {
                "dataset": [
                    {"id": "negative", **negative_routing},
                    {"id": "positive", "expected_skill": "release-review"},
                ]
            }
        )
    )
    _trial(tmp_path, "a", "n0", negative)
    _trial(tmp_path, "a", "p0", positive)

    [fix] = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)

    assert (fix["case"], fix["kind"], fix["attempts_lost"]) == ("positive", "skill not used", 1)
    assert "frontmatter description" in fix["action"]
    console = Console(record=True, width=160)
    render_case_summary({"agents": {"a": agent}}, console=console, fixes=[fix])
    advice = console.export_text().split("How to improve", 1)[1]
    assert "positive" in advice and "negative" not in advice


@pytest.mark.parametrize("activation_status", ["never_activated", "unknown"])
@pytest.mark.parametrize("failure_kind", ["skill", "environment", "run error"])
def test_failed_negative_keeps_task_diagnostics(tmp_path: Path, activation_status: str, failure_kind: str) -> None:
    case = _case(missing=1) if failure_kind == "run error" else _case(False)
    agent = _agent({"negative": case}, {})
    agent["skill_activation"] = {"status": activation_status}
    (tmp_path / "dataset_snapshot.json").write_text(
        json.dumps({"dataset": [{"id": "negative", "should_trigger": False, "expected_skill": "release-review"}]})
    )
    if failure_kind == "run error":
        trial = tmp_path / "a" / "with-skill" / "trials" / "negative__attempt1"
        trial.mkdir(parents=True)
        (trial / "failure.json").write_text(json.dumps({"error": {"type": "AgentTimeoutError", "message": "900s"}}))
    else:
        reward = _reward(False, "the answer omitted the requested total")
        reward["details"]["skill_execution"]["should_trigger"] = False
        observation = "ModuleNotFoundError: No module named 'pypdf'" if failure_kind == "environment" else ""
        _trial(tmp_path, "a", "t0", reward, observation)

    [fix] = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)

    assert (fix["case"], fix["kind"], fix["attempts_lost"]) == ("negative", failure_kind, 1)
    assert "frontmatter description" not in fix["action"]
    if failure_kind == "environment":
        assert "No module named 'pypdf'" in fix["what"]
    elif failure_kind == "run error":
        assert "AgentTimeoutError" in fix["what"]
    else:
        assert "omitted the requested total" in fix["what"]


@pytest.mark.parametrize("routing", [{"expected_skill": None}, {"should_trigger": False}])
@pytest.mark.parametrize("passed", [False, True])
@pytest.mark.parametrize("use_llm", [False, True])
def test_correctly_unused_negative_answer_does_not_advise_skill_edits(
    tmp_path: Path, monkeypatch, routing: dict, passed: bool, use_llm: bool
) -> None:
    monkeypatch.setattr(
        "skillevaluator.tier3.case_summary._llm_actions",
        lambda *_args: pytest.fail("Correct non-use must not request a skill-edit action"),
    )
    (tmp_path / "dataset_snapshot.json").write_text(json.dumps({"dataset": [{"id": "negative", **routing}]}))
    reward = _reward(False, "the answer omitted the requested total")
    reward.update(
        entry_id="negative",
        skill_execution=1.0,
        goal_accuracy=0.0,
        skill_invoked=False,
        invocation_evidence_source="trajectory",
        invocation_evidence_scope="logical_trial",
    )
    _trial(tmp_path, "a", "t0", reward)
    agent = _agent({"negative": _case(passed)}, {})

    [fix] = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=use_llm)

    assert fix["kind"] == "task outcome"
    assert fix["source"] == "deterministic"
    assert "correctly unused" in fix["action"]
    assert "SKILL.md" not in fix["action"]
    assert "requested total" in fix["what"] or "requested total" in " ".join(fix["reasons"])
    assert fix["attempts_lost"] == 1


@pytest.mark.parametrize("evidence", ["invoked", "unknown", "missing", "error", "custom", "step", "missing scope"])
def test_negative_answer_requires_complete_trusted_non_use_evidence(tmp_path: Path, evidence: str) -> None:
    (tmp_path / "dataset_snapshot.json").write_text(
        json.dumps({"dataset": [{"id": "negative", "expected_skill": None}]})
    )
    if evidence != "missing":
        reward = _reward(False, "wrong total")
        reward.update(
            entry_id="negative",
            skill_execution=1.0,
            skill_invoked=evidence == "invoked",
            invocation_evidence_source="trajectory",
            invocation_evidence_scope="logical_trial",
        )
        if evidence == "unknown":
            reward.pop("invocation_evidence_source")
        elif evidence == "error":
            reward["evaluation_status"] = "error"
        elif evidence == "custom":
            reward["metric_set"] = "custom-metrics"
        elif evidence == "step":
            reward["invocation_evidence_scope"] = "step"
        elif evidence == "missing scope":
            reward.pop("invocation_evidence_scope")
        _trial(tmp_path, "a", "t0", reward)
    agent = _agent({"negative": _case(False)}, {})

    fixes = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)

    assert all(fix["kind"] != "task outcome" for fix in fixes)


@pytest.mark.parametrize(
    "metadata",
    [
        {"attempt_details_truncated": True},
        {"attempts_recovered": 1},
        {"attempts_missing": 1},
        *(
            {key: value}
            for key in ("attempts_used", "attempt_details_total", "attempt_details_shown")
            for value in (2, -1, True, 1.0, "1", None)
        ),
    ],
)
def test_negative_non_use_requires_complete_attempt_details(tmp_path: Path, metadata: dict) -> None:
    from skillevaluator.tier3.case_summary import _correctly_unused_negative

    reward = _reward(False, "wrong total")
    reward.update(
        entry_id="negative",
        skill_execution=1.0,
        skill_invoked=False,
        invocation_evidence_source="trajectory",
        invocation_evidence_scope="logical_trial",
    )
    _trial(tmp_path, "a", "t0", reward)
    case = {**_case(False), **metadata}
    agent = _agent({"negative": case}, {})

    assert not _correctly_unused_negative(tmp_path, "a", agent, "negative", {"expected_skill": None})


def test_collector_capped_attempts_do_not_claim_complete_negative_non_use(tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor.collector import PUBLISHED_ATTEMPT_DETAILS_PER_CASE_MAX, _pass_summary

    (tmp_path / "dataset_snapshot.json").write_text(
        json.dumps({"dataset": [{"id": "negative", "expected_skill": None}]})
    )
    rewards = []
    for index in range(PUBLISHED_ATTEMPT_DETAILS_PER_CASE_MAX + 1):
        reward = _reward(False, "wrong total")
        reward.update(
            entry_id="negative",
            _trial_name=f"t{index}",
            skill_execution=1.0,
            skill_invoked=index == PUBLISHED_ATTEMPT_DETAILS_PER_CASE_MAX,
            invocation_evidence_source="trajectory",
            invocation_evidence_scope="logical_trial",
        )
        rewards.append(reward)
        _trial(tmp_path, "a", f"t{index}", reward)
    summary = _pass_summary(
        rewards,
        n_attempts=len(rewards),
        pass_threshold=0.5,
        expected_cases=1,
        expected_case_ids=["negative"],
        scorer=lambda _reward: 0.2,
    )
    case = summary["cases"]["negative"]
    assert case["attempt_details_truncated"]
    assert len(case["attempts"]) == PUBLISHED_ATTEMPT_DETAILS_PER_CASE_MAX
    assert case["attempts_used"] == len(rewards)
    agent = _agent({}, {})
    agent["pass_at_k"]["with_skill"] = summary

    fixes = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)

    assert fixes
    assert all(fix["kind"] != "task outcome" and "correctly unused" not in fix["action"] for fix in fixes)


def test_negative_task_outcome_scope_deduplicates_agent_and_sums_losses(tmp_path: Path) -> None:
    (tmp_path / "dataset_snapshot.json").write_text(
        json.dumps({"dataset": [{"id": "negative", "expected_skill": None}]})
    )
    for name, attempts in (("A", 2), ("B", 1)):
        for i in range(attempts):
            reward = _reward(False, "wrong total" if name == "A" else "correct total")
            reward.update(
                entry_id="negative",
                skill_execution=1.0,
                goal_accuracy=0.0 if name == "A" else 1.0,
                skill_invoked=False,
                invocation_evidence_source="trajectory",
                invocation_evidence_scope="logical_trial",
            )
            _trial(tmp_path, name, f"t{i}", reward)
    agents = {
        "A": _agent({"negative": _case(False, True)}, {}),
        "B": _agent({"negative": _case(True)}, {}),
    }

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert fix["kind"] == "task outcome"
    assert fix["scope"] == "A only"
    assert fix["scope"].count("A") == 1
    assert fix["attempts_lost"] == 2


@pytest.mark.parametrize("passed", [False, True])
def test_negative_task_outcome_copy_does_not_blame_skill_or_call_passed_attempts_lost(
    tmp_path: Path, passed: bool
) -> None:
    (tmp_path / "dataset_snapshot.json").write_text(
        json.dumps({"dataset": [{"id": "negative", "expected_skill": None}]})
    )
    reward = _reward(False, "wrong total")
    reward.update(
        entry_id="negative",
        goal_accuracy=0.0,
        skill_execution=1.0,
        skill_invoked=False,
        invocation_evidence_source="trajectory",
        invocation_evidence_scope="logical_trial",
    )
    _trial(tmp_path, "a", "t0", reward)
    result = {"agents": {"a": _agent({"negative": _case(passed)}, {"negative": _case(True)})}}

    fixes = what_to_fix(result, run_dir=tmp_path, use_llm=False)
    [fix] = [item for item in fixes if item["kind"] == "task outcome"]
    assert "without the skill" not in fix["what"]
    assert "wrong total" in fix["what"]
    console = Console(record=True, width=180)
    render_case_summary(result, console=console, fixes=[fix])
    advice = console.export_text().split("How to improve", 1)[1]
    assert "missed the case requirements" in advice
    assert "lost " not in advice


def test_negative_task_outcomes_merge_without_hiding_a_skill_fix(tmp_path: Path) -> None:
    negative_ids = [f"negative-{index}" for index in range(3)]
    (tmp_path / "dataset_snapshot.json").write_text(
        json.dumps(
            {
                "dataset": [
                    *({"id": case, "expected_skill": None} for case in negative_ids),
                    {"id": "positive", "should_trigger": True},
                ]
            }
        )
    )
    cases = {}
    for case_id in [*negative_ids, "positive"]:
        negative = case_id in negative_ids
        cases[case_id] = _case(*([False, False] if negative else [False]), prefix=case_id)
        for index in range(2 if negative else 1):
            reward = _reward(not negative, "wrong total" if negative else "missing required feature")
            reward.update(
                entry_id=case_id,
                skill_execution=1.0,
                skill_invoked=not negative,
                invocation_evidence_source="trajectory",
                invocation_evidence_scope="logical_trial",
            )
            _trial(tmp_path, "a", f"{case_id}{index}", reward)

    fixes = what_to_fix({"agents": {"a": _agent(cases, {})}}, run_dir=tmp_path, use_llm=False)

    assert len(fixes) == 2
    [task_outcome] = [fix for fix in fixes if fix["kind"] == "task outcome"]
    assert task_outcome["cases"] == negative_ids
    assert task_outcome["attempts_lost"] == 6
    assert any(fix["case"] == "positive" and fix["kind"] == "skill" for fix in fixes)


def test_trusted_observed_activation_is_a_public_metrics_helper() -> None:
    from skillevaluator.tier3 import case_summary
    from skillevaluator.tier3.harbor import metrics

    helper = metrics.trusted_observed_activation
    trusted = {"skill_execution": 1.0, "invocation_evidence_source": "trajectory"}
    assert helper({**trusted, "skill_invoked": True}) == "invoked"
    assert helper({**trusted, "skill_invoked": False}) == "not_invoked"
    assert helper({**trusted, "skill_invoked": "no"}) == "unknown"
    assert helper({"skill_execution": 1.0, "skill_invoked": False}) == "unknown"
    assert helper({**trusted, "skill_invoked": False, "metric_set": "custom-metrics"}) == "unknown"
    assert helper({"invocation_evidence_source": "trajectory", "skill_invoked": False}) == "unknown"
    assert case_summary.trusted_observed_activation is helper


def test_report_activation_validation_uses_the_shared_trusted_helper(monkeypatch) -> None:
    from skillevaluator.evaluation.tier3_report import _trial_activation_validation
    from skillevaluator.tier3.harbor import metrics

    monkeypatch.setattr(metrics, "trusted_observed_activation", lambda _reward: "shared-helper")
    reward = {"invocation_evidence_scope": "logical_trial"}
    trial = {"scores": {}}

    validation = _trial_activation_validation([reward], trial)

    assert validation["observed_activation"] == "shared-helper"
    assert validation["evidence_scopes"][0]["observed_activation"] == "shared-helper"


def test_case_summary_import_does_not_load_the_report_module() -> None:
    import subprocess
    import sys

    source = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(source)!r}); "
        "import skillevaluator.tier3.case_summary; "
        f"assert skillevaluator.tier3.case_summary.__file__ == {str(source / 'skillevaluator/tier3/case_summary.py')!r}; "
        "sys.exit(1 if 'skillevaluator.evaluation.tier3_report' in sys.modules else 0)"
    )
    completed = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("snapshot", [None, "{broken", '{"dataset": [{"id": "c"}]}'])
@pytest.mark.parametrize("activation_status", ["never_activated", "unknown"])
def test_unlabeled_snapshot_preserves_legacy_activation_diagnosis(
    tmp_path: Path, snapshot: str | None, activation_status: str
) -> None:
    agent = _agent({"c": _case(True)}, {})
    agent["skill_activation"] = {"status": activation_status}
    if snapshot is not None:
        (tmp_path / "dataset_snapshot.json").write_text(snapshot)

    fixes = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)

    if activation_status == "never_activated":
        [fix] = fixes
        assert (fix["case"], fix["kind"], fix["attempts_lost"]) == ("c", "skill not used", 1)
    else:
        assert fixes == []


@pytest.mark.parametrize("case_count", [20, 100])
@pytest.mark.parametrize("agent_count", [1, 2])
@pytest.mark.parametrize("scenario", ["passing", "mixed negative", "empty snapshot"])
def test_case_advice_reads_snapshot_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case_count: int, agent_count: int, scenario: str
) -> None:
    from skillevaluator.utils.secure_fs import SecureRoot

    case_ids = [f"case-{i:03d}" for i in range(case_count)]
    entries = [
        {"id": case_id, "should_trigger": bool(i % 2), "expected_skill": "release-review"}
        for i, case_id in enumerate(case_ids)
    ]
    (tmp_path / "dataset_snapshot.json").write_text(
        json.dumps({"dataset": [] if scenario == "empty snapshot" else entries})
    )
    cases = {case_id: _case(scenario != "mixed negative" or i % 2 == 0) for i, case_id in enumerate(case_ids)}
    agents = {f"agent-{i}": _agent(cases, {}) for i in range(agent_count)}
    for agent in agents.values():
        agent["skill_activation"] = {"status": "unknown" if scenario == "passing" else "never_activated"}
    snapshot_reads = 0
    read_bytes = SecureRoot.read_bytes

    def count_reads(self, path, *args, **kwargs):
        nonlocal snapshot_reads
        if str(path) == "dataset_snapshot.json":
            snapshot_reads += 1
        return read_bytes(self, path, *args, **kwargs)

    monkeypatch.setattr(SecureRoot, "read_bytes", count_reads)

    fixes = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    if scenario == "passing":
        assert fixes == []
    else:
        [fix] = fixes
        expected_cases = case_ids[1::2] if scenario == "mixed negative" else case_ids
        assert fix["cases"] == expected_cases
        assert fix["kind"] == "skill not used"
        assert fix["attempts_lost"] == len(expected_cases) * agent_count
    assert snapshot_reads == 1


def test_direct_case_diagnosis_still_loads_snapshot(tmp_path: Path) -> None:
    from skillevaluator.tier3.case_summary import _diagnose_agent_case

    (tmp_path / "dataset_snapshot.json").write_text(
        json.dumps({"dataset": [{"id": "negative", "should_trigger": False, "expected_skill": "release-review"}]})
    )
    agent = _agent({"negative": _case(True)}, {})
    agent["skill_activation"] = {"status": "never_activated"}

    assert _diagnose_agent_case(tmp_path, "a", agent, "negative") == []


def test_truncated_attempts_cannot_produce_a_complete_effect_or_clean_bill(tmp_path: Path) -> None:
    case = {**_case(True), "attempt_details_total": 2, "attempt_details_truncated": True}
    agent = _agent({"c": case}, {"c": _case(False, False)})
    assert verdict({"a": agent})[0] == "inconclusive"
    console = Console(record=True, width=160)
    render_case_summary({"agents": {"a": agent}}, console=console, fixes=[])
    output = console.export_text()
    assert "inconclusive" in output and "nothing to fix" not in output


def test_truncated_case_sample_does_not_claim_baseline_saturation(tmp_path: Path) -> None:
    agent = _agent({"c": _case(True)}, {"c": _case(True)})
    agent["pass_at_k"]["without_skill"].update(case_details_truncated=True, total_cases=100)
    assert verdict({"a": agent})[0] == "inconclusive"
    assert what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False) == []


def test_new_summary_redacts_known_credentials_and_keeps_case_names_literal(tmp_path: Path, monkeypatch) -> None:
    secret = "review-fake-environment-credential-7135"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    case_id = f"[bold]case[/bold]-{secret}"
    agent = _agent({case_id: _case(False)}, {case_id: _case(False)})
    (tmp_path / "what_to_fix.json").write_text(
        json.dumps(
            [
                {
                    "case": case_id,
                    "scope": secret,
                    "kind": "skill",
                    "what": secret,
                    "action": secret,
                    "evidence": secret,
                    "attempts_lost": 1,
                }
            ]
        )
    )
    output = render_result({"execution_status": "succeeded", "run_dir": str(tmp_path), "agents": {secret: agent}})
    assert secret not in output
    assert "[bold]case[/bold]" in output


def test_trial_artifact_symlinks_cannot_supply_advice(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "reward.json").write_text(json.dumps(_reward(True, "outside-private-reason")))
    (outside / "trajectory.json").write_text(
        json.dumps({"steps": [{"source": "agent", "message": "outside-private-answer"}]})
    )
    run = tmp_path / "run"
    trials = run / "a" / "with-skill" / "trials"
    trials.mkdir(parents=True)
    (trials / "t0").symlink_to(outside, target_is_directory=True)
    [fix] = what_to_fix({"agents": {"a": _agent({"c": _case(False)}, {})}}, run_dir=run, use_llm=False)
    assert "outside-private" not in json.dumps(fix)
    assert fix["evidence"] == ""


def test_symlinked_reward_file_cannot_supply_advice(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(_reward(True, "outside-private-reason")))
    trial = tmp_path / "run" / "a" / "with-skill" / "trials" / "t0"
    trial.mkdir(parents=True)
    (trial / "reward.json").symlink_to(outside)
    [fix] = what_to_fix({"agents": {"a": _agent({"c": _case(False)}, {})}}, run_dir=tmp_path / "run", use_llm=False)
    assert "outside-private-reason" not in json.dumps(fix)


def test_parent_agent_component_cannot_escape_run_directory(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    trial = tmp_path / "with-skill" / "trials" / "t0"
    trial.mkdir(parents=True)
    (trial / "reward.json").write_text(json.dumps(_reward(True, "outside-private-reason")))
    [fix] = what_to_fix({"agents": {"..": _agent({"c": _case(False)}, {})}}, run_dir=run, use_llm=False)
    assert "outside-private-reason" not in json.dumps(fix)
    assert fix["evidence"] == ""


def test_advice_is_redacted_before_it_is_saved(tmp_path: Path, monkeypatch) -> None:
    secret = "review-fake-credential-in-judge-reason-8745"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    _trial(tmp_path, "a", "t0", _reward(True, secret))
    fixes = what_to_fix({"agents": {"a": _agent({"c": _case(False)}, {})}}, run_dir=tmp_path, use_llm=False)
    assert secret not in json.dumps(fixes)
    assert "<redacted>" in json.dumps(fixes)


def test_issue_scope_names_a_subset_of_more_than_two_agents(tmp_path: Path) -> None:
    agents = {name: _agent({"c": _case(name == "c")}, {}) for name in ("a", "b", "c")}
    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)
    assert fix["scope"] == "a, b only"


def test_missing_attempts_suppress_the_single_agent_summary_delta() -> None:
    agent = _agent({"c": _case(True, missing=1)}, {"c": _case(False)})
    output = render_result({"execution_status": "succeeded", "agents": {"a": agent}})
    row = next(line for line in output.splitlines() if line.startswith("success rate"))
    assert "+100" not in row


def test_advisory_prompt_redacts_credentials_before_the_model_call(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.tier3.eval_core import llm_judge

    secret = "review-fake-advisory-prompt-secret-9375"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    _trial(tmp_path, "a", "t0", _reward(True, secret))
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(f"# Skill\n\nPrivate setup: {secret}")
    prompts = []

    def call(prompt, **kwargs):
        prompts.append(prompt)
        return '["Improve the skill instructions."]', None

    monkeypatch.setattr(llm_judge, "call_public_llm", call)
    what_to_fix({"agents": {"a": _agent({"c": _case(False)}, {})}}, run_dir=tmp_path, skill_path=skill)
    assert len(prompts) == 1 and secret not in prompts[0]
    assert "\n\nSKILL.md:\n# Skill\n" in prompts[0]


def test_long_error_logs_keep_the_root_error_from_the_bounded_tail(tmp_path: Path) -> None:
    trial = tmp_path / "a" / "with-skill" / "trials" / "c__1"
    trial.mkdir(parents=True)
    (trial / "failure.json").write_text(json.dumps({"error": {"type": "AgentExitError", "message": "wrapper failed"}}))
    (trial / "agent.txt").write_text("irrelevant log output\n" * 10000 + "Error: specific-tail-cause\n")
    [fix] = what_to_fix({"agents": {"a": _agent({"c": _case(missing=1)}, {})}}, run_dir=tmp_path, use_llm=False)
    assert "specific-tail-cause" in fix["what"]


def test_failed_condition_rebuild_does_not_follow_a_symlinked_trials_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "c__1"
    outside.mkdir(parents=True)
    (outside / "reward.json").write_text(json.dumps({"overall": 0.9}))
    agent_dir = tmp_path / "run" / "a"
    (agent_dir / "with-skill").mkdir(parents=True)
    (agent_dir / "with-skill" / "trials").symlink_to(outside.parent, target_is_directory=True)
    agent = _agent({}, {})
    agent["output_dir"] = str(agent_dir)
    console = Console(record=True, width=160)
    assert not render_case_summary({"agents": {"a": agent}}, console=console, fixes=None)


def test_different_condition_score_contracts_do_not_claim_a_causal_skill_effect(tmp_path: Path) -> None:
    for fields in (
        {"metrics_with_skill": ["security"], "metrics_without_skill": ["accuracy"]},
        {"score_policy_with_skill": "dimension-mean/v1", "score_policy_without_skill": "legacy-flat/v1"},
    ):
        agent = {**_agent({"c": _case(False)}, {"c": _case(True)}), **fields}
        assert verdict({"a": agent})[0] == "inconclusive"
        fixes = what_to_fix({"agents": {"a": agent}}, run_dir=tmp_path, use_llm=False)
        assert all(fix["kind"] != "skill hurts" for fix in fixes)
        output = render_result({"execution_status": "succeeded", "agents": {"a": agent}})
        assert "skill hurts" not in output
        rates = next(line for line in output.splitlines() if line.startswith("success rate"))
        assert "-100" not in rates


def test_disjoint_case_coverage_does_not_produce_an_overall_effect() -> None:
    agent = _agent(
        {"shared": _case(True), "with-only": _case(True)}, {"shared": _case(False), "baseline-only": _case(False)}
    )
    assert verdict({"a": agent})[0] == "inconclusive"


def test_logical_multistep_goal_counts_average_steps_once(tmp_path: Path) -> None:
    from skillevaluator.tier3.case_summary import goal_not_met_count

    for label, goals, expected in (("missed", [0.0, 0.0], 1), ("mixed", [0.0, 1.0], 0)):
        run = tmp_path / label
        trial_id = "case__attempt1"
        for i, goal in enumerate(goals, 1):
            _trial(run, "a", f"{trial_id}__step-{i}", {"trial_id": trial_id, "goal_accuracy": goal})
        case = {"attempts": [{"trial": trial_id, "passed": True, "score": 0.9}]}
        source = {"agents": {"a": _agent({"case": case}, {"case": _case(False)})}}
        assert goal_not_met_count(source, run_dir=run) == expected
        fixes = what_to_fix(source, run_dir=run, use_llm=False)
        assert sum(f["attempts_lost"] for f in fixes if f["kind"] == "goal not met") == expected


def test_failed_conditions_keep_rebuilt_scores_diagnostic(tmp_path: Path) -> None:
    agent = _agent({"c": _case(True)}, {"c": _case(False)})
    agent["conditions"] = {
        "with_skill": {"execution_status": "failed"},
        "without_skill": {"execution_status": "succeeded"},
    }
    assert verdict({"a": agent})[0] == "inconclusive"
    output = render_result({"execution_status": "failed", "agents": {"a": agent}})
    assert "skill helps" not in output


def test_saved_fixes_do_not_follow_a_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    content = json.dumps(
        [
            {
                "case": "private-external-case",
                "scope": "private",
                "kind": "skill",
                "what": "private",
                "action": "private",
                "evidence": "private",
                "attempts_lost": 1,
            }
        ]
    )
    outside.write_text(content)
    run = tmp_path / "run"
    run.mkdir()
    (run / "what_to_fix.json").symlink_to(outside)
    agent = _agent({"c": _case(True)}, {"c": _case(False)})
    output = render_result({"execution_status": "succeeded", "run_dir": str(run), "agents": {"a": agent}})
    assert "private-external-case" not in output
    assert outside.read_text() == content


def _goal_miss(run_dir: Path, agent: str, trial: str, end_state: str) -> None:
    _trial(run_dir, agent, trial, {"goal_accuracy": 0.0, "details": {"goal_accuracy": {"end_state": end_state}}})


def test_goal_not_met_uses_the_canned_fix_only_when_the_agent_stopped_for_input(tmp_path: Path) -> None:
    outcomes = {
        "The agent made HTTP requests to the API and returned the wrong totals.": False,
        "The agent fixed an import that was not found and wrote the report.": False,
        "The agent asked the user for the report file.": True,
        "report.pdf does not exist in the workspace, so nothing was extracted.": True,
    }
    for index, (outcome, stopped) in enumerate(outcomes.items()):
        run_dir = tmp_path / str(index)
        _goal_miss(run_dir, "a", "x0", outcome)

        [fix] = what_to_fix(
            {"agents": {"a": _agent({"c1": _case(True, prefix="x")}, {})}}, run_dir=run_dir, use_llm=False
        )

        assert fix["asks_for_input"] is stopped, outcome
        assert fix["action"].startswith("The agent stopped to ask for input") is stopped, outcome


def test_merged_goal_issue_keeps_the_canned_fix_only_if_every_agent_stopped_for_input(tmp_path: Path) -> None:
    agents = {name: _agent({"c1": _case(True, prefix=name)}, {}) for name in ("a", "b")}
    _goal_miss(tmp_path, "a", "a0", "The agent asked for the sprint details.")
    _goal_miss(tmp_path, "b", "b0", "The agent wrote the wrong section headings.")

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert (fix["scope"], fix["asks_for_input"]) == ("all agents", False)


def test_case_with_a_failure_also_reports_passes_that_missed_the_goal(tmp_path: Path) -> None:
    agents = {"a": _agent({"c1": _case(True, False, prefix="m")}, {})}
    _goal_miss(tmp_path, "a", "m0", "The agent asked for the data.")
    _trial(tmp_path, "a", "m1", _reward(True, "wrong totals"))

    kinds = {f["kind"] for f in what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)}

    assert kinds == {"skill", "goal not met"}


def test_fixes_redact_environment_secrets(tmp_path: Path, monkeypatch) -> None:
    secret = "review-fake-goal-credential-5521"
    monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", secret)
    agents = {"a": _agent({"c1": _case(True, prefix="s")}, {})}
    _goal_miss(tmp_path, "a", "s0", f"The agent printed {secret} and asked for the file.")

    [fix] = what_to_fix({"agents": agents}, run_dir=tmp_path, use_llm=False)

    assert secret not in json.dumps(fix)
    assert "<redacted>" in fix["what"]


def test_goal_not_met_heading_counts_attempts_that_missed_the_goal() -> None:
    result = {"agents": {"a": _agent({"c1": _case(True)}, {"c1": _case(False)})}}
    fix = {"case": "c1", "kind": "goal not met", "scope": "a", "attempts_lost": 2, "what": "w", "action": "f"}
    console = Console(record=True, width=120)

    render_case_summary(result, console=console, fixes=[fix])

    text = console.export_text()
    assert "2 attempt(s) missed the goal" in text
    assert "lost 2" not in text
