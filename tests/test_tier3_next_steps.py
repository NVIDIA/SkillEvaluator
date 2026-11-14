# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-level Tier 3 next steps: grounded, distinct, and honest about weak evidence."""

import json
from pathlib import Path

from tests.reporting.test_case_findings_report import _run
from tests.test_tier3_case_summary import _agent, _case

from skillevaluator.evaluation.tier3_report import (
    _validation_result_from_payload,
    agent_eval_result_from_directory,
    build_agent_eval_payload,
    render_agent_eval_html_report,
)
from skillevaluator.reporting import MarkdownReporter
from skillevaluator.tier3.next_steps import HYPOTHESIS, LIMITED, SUPPORTED, next_steps
from skillevaluator.tier3.result_display import render_result

_IDS = ("c1", "c2", "c3", "c4", "c5")


def _dataset(run: Path, *, negative: bool = True, ids: tuple[str, ...] = _IDS) -> None:
    entries = [{"id": case_id, "question": "q", "expected_skill": "demo"} for case_id in ids]
    if negative:
        entries.append({"id": "neg", "question": "unrelated", "expected_skill": None})
    run.mkdir(parents=True, exist_ok=True)
    (run / "dataset_snapshot.json").write_text(json.dumps({"dataset": entries}))


def _helps(ids: tuple[str, ...] = _IDS, *, negative: bool = True) -> dict:
    with_cases = {case_id: _case(True, True) for case_id in ids}
    without_cases = {case_id: _case(False, False) for case_id in ids}
    if negative:
        with_cases["neg"] = _case(True, True)
        without_cases["neg"] = _case(True, True)
    return _agent(with_cases, without_cases)


def _kinds(steps: list[dict]) -> list[str]:
    return [step["kind"] for step in steps]


def test_clean_run_with_enough_cases_and_a_negative_control_needs_no_next_step(tmp_path: Path) -> None:
    _dataset(tmp_path)
    assert next_steps({"agents": {"a": _helps()}}, run_dir=tmp_path) == []


def test_missing_attempts_come_first_and_limit_the_other_steps(tmp_path: Path) -> None:
    _dataset(tmp_path, ids=("c1", "c2"))
    agent = _helps(("c1", "c2"))
    agent["pass_at_k"]["with_skill"]["cases"]["c1"] = _case(True, missing=1)

    steps = next_steps({"agents": {"a": agent}}, run_dir=tmp_path)

    assert _kinds(steps) == ["incomplete run", "more cases"]
    assert steps[0]["strength"] == SUPPORTED and steps[0]["target"] == "evaluation"
    assert "a/c1 (with-skill)" in steps[0]["evidence"]
    assert steps[1]["strength"] == LIMITED


def test_run_errors_already_listed_per_case_are_not_repeated(tmp_path: Path) -> None:
    _dataset(tmp_path, ids=("c1", "c2"))
    agent = _helps(("c1", "c2"))
    agent["pass_at_k"]["with_skill"]["cases"]["c1"] = _case(True, missing=1)

    steps = next_steps({"agents": {"a": agent}}, run_dir=tmp_path, fixes=[{"kind": "run error"}])

    assert _kinds(steps) == ["more cases"]
    assert steps[0]["strength"] == LIMITED


def test_mixed_attempts_ask_for_more_attempts_before_editing_anything(tmp_path: Path) -> None:
    _dataset(tmp_path)
    agent = _helps()
    agent["pass_at_k"]["with_skill"]["cases"]["c2"] = _case(True, False)

    [step] = next_steps({"agents": {"a": agent}}, run_dir=tmp_path)

    assert step["kind"] == "inconsistent results" and step["target"] == "evaluation"
    assert "`--n-attempts 4`" in step["action"]
    assert step["evidence"] == "a/c2 with-skill ✓✗"

    result = {"agents": {"a": agent}, "attempt_policy": {"stop_on_pass": True}}
    [stopped_step] = next_steps(result, run_dir=tmp_path)
    assert "`--n-attempts 4 --no-stop-on-pass`" in stopped_step["action"]


def test_single_attempt_runs_are_flagged_but_stop_on_pass_skips_are_not(tmp_path: Path) -> None:
    _dataset(tmp_path)
    once = _agent(
        {case_id: _case(True) for case_id in (*_IDS, "neg")},
        {case_id: _case(False) for case_id in _IDS} | {"neg": _case(True)},
    )
    assert _kinds(next_steps({"agents": {"a": once}}, run_dir=tmp_path)) == ["single attempt"]

    skipped = {**_case(True), "attempts_skipped": 2}
    stopped = _agent(
        {case_id: dict(skipped) for case_id in (*_IDS, "neg")},
        {case_id: _case(False, False, False) for case_id in _IDS} | {"neg": dict(skipped)},
    )
    assert next_steps({"agents": {"a": stopped}}, run_dir=tmp_path) == []


def test_agent_dependent_effect_is_a_hypothesis_to_test(tmp_path: Path) -> None:
    _dataset(tmp_path)
    flat = _agent(
        {case_id: _case(True, True) for case_id in (*_IDS, "neg")},
        {case_id: _case(True, True) for case_id in (*_IDS, "neg")},
    )

    [step] = next_steps({"agents": {"claude-code": _helps(), "codex": flat}}, run_dir=tmp_path)

    assert step["kind"] == "agent-dependent effect" and step["strength"] == HYPOTHESIS
    assert "claude-code +83 pp" in step["problem"] and "codex +0 pp" in step["problem"]
    assert step["action"].startswith("Test the hypothesis that codex")


def test_never_activated_agent_does_not_trigger_single_agent_experiment(tmp_path: Path) -> None:
    _dataset(tmp_path)
    never = {**_helps(), "skill_activation": {"status": "never_activated"}}
    flat = _agent(
        {case_id: _case(True, True) for case_id in (*_IDS, "neg")},
        {case_id: _case(True, True) for case_id in (*_IDS, "neg")},
    )

    steps = next_steps({"agents": {"claude-code": never, "codex": flat}}, run_dir=tmp_path)

    assert "agent experiment" not in _kinds(steps)


def test_few_cases_ask_for_more_cases_and_negative_controls_do_not_count(tmp_path: Path) -> None:
    _dataset(tmp_path, ids=("c1", "c2"))

    [step] = next_steps({"agents": {"a": _helps(("c1", "c2"))}}, run_dir=tmp_path)

    assert step["kind"] == "more cases" and step["target"] == "dataset"
    assert step["problem"].startswith("only 2 skill case(s)") and step["evidence"] == "c1, c2"
    assert "keep the existing cases and their expectations" in step["action"]
    rewritten = [{"kind": "eval case", "case": "c1"}]
    assert _kinds(next_steps({"agents": {"a": _helps(("c1", "c2"))}}, run_dir=tmp_path, fixes=rewritten)) == [
        "more cases"
    ]
    saturated = [{"kind": "eval case", "case": "all cases"}]
    assert next_steps({"agents": {"a": _helps(("c1", "c2"))}}, run_dir=tmp_path, fixes=saturated) == []


def test_missing_negative_control_is_reported_only_with_a_dataset_snapshot(tmp_path: Path) -> None:
    result = {"agents": {"a": _helps(negative=False)}}
    assert next_steps(result, run_dir=tmp_path) == []

    _dataset(tmp_path, negative=False)
    [step] = next_steps(result, run_dir=tmp_path)
    assert step["kind"] == "negative controls" and step["target"] == "dataset"
    assert '"expected_skill": null' in step["action"]


def test_single_agent_without_a_clear_gain_suggests_an_agent_experiment(tmp_path: Path) -> None:
    _dataset(tmp_path)
    flat = _agent(
        {case_id: _case(True, True) for case_id in (*_IDS, "neg")},
        {case_id: _case(True, True) for case_id in (*_IDS, "neg")},
    )

    [step] = next_steps({"agents": {"opencode": flat}}, run_dir=tmp_path)

    assert step["kind"] == "agent experiment" and step["strength"] == HYPOTHESIS
    assert "`--agents opencode,<agent>`" in step["action"] and "keeping the dataset unchanged" in step["action"]


def test_steps_are_capped_distinct_and_never_weaken_the_dataset(tmp_path: Path) -> None:
    _dataset(tmp_path, ids=("c1",), negative=False)
    agent = _agent({"c1": _case(True, False, missing=1)}, {"c1": _case(True, False, False)})

    steps = next_steps({"agents": {"opencode": agent}}, run_dir=tmp_path)

    assert _kinds(steps) == ["incomplete run", "inconsistent results", "more cases"]
    text = " ".join(value for step in steps for value in step.values()).lower()
    assert not any(word in text for word in ("remove", "delete", "loosen", "lower the"))


def test_baseline_only_and_never_activated_runs_get_no_next_steps(tmp_path: Path) -> None:
    baseline_only = {**_helps(), "conditions": {"with_skill": {"execution_status": "skipped"}}}
    never = {**_helps(), "skill_activation": {"status": "never_activated"}}
    assert next_steps({"agents": {"a": baseline_only}}, run_dir=tmp_path) == []
    assert next_steps({"agents": {"a": never}}, run_dir=tmp_path) == []


def test_cli_prints_next_steps_after_the_case_summary(tmp_path: Path) -> None:
    _dataset(tmp_path, ids=("c1", "c2"))
    result = {"execution_status": "succeeded", "run_dir": str(tmp_path), "agents": {"a": _helps(("c1", "c2"))}}

    output = render_result(result)

    assert output.index("How to improve") < output.index("Next steps")
    assert "1. only 2 skill case(s) were evaluated" in output and "[dataset]" in output
    assert "check: With 5 or more skill cases the verdict keeps its direction." in output


def test_html_report_shows_next_steps_under_the_case_fixes(tmp_path: Path) -> None:
    run = _run(tmp_path, goal_accuracy=1.0)

    html = render_agent_eval_html_report(tmp_path, run, use_llm_judge=False).read_text(encoding="utf-8")

    preview = " ".join(
        html.split('id="tier3-next-steps-preview"', 1)[1].split('id="tier3-dimension-lift"', 1)[0].split()
    )
    assert preview.index("Nothing to fix in the available case evidence.") < preview.index("Next steps for the run:")
    assert (
        "<strong>each case ran once, so a single attempt decides every pass or fail mark</strong> [evaluation]:"
        in preview
    )
    assert "Check: With 5 or more skill cases the verdict keeps its direction. Evidence: case" in preview


def test_markdown_report_renders_next_step_strength() -> None:
    agent = {
        "num_trials": 1,
        "num_trials_baseline": 1,
        "with_skill": {"goal_accuracy": 0.9},
        "without_skill": {"goal_accuracy": 0.4},
        "lift": {"overall": {"delta": 0.5}},
    }
    payload = build_agent_eval_payload("demo", {"opencode": agent}, use_llm_judge=False)
    payload["next_steps"] = [
        {
            "kind": "agent experiment",
            "target": "agent",
            "problem": "only opencode was evaluated",
            "evidence": "opencode +0 pp",
            "action": "Run the same cases with a second agent.",
            "check": "A second agent shows the same direction.",
            "strength": HYPOTHESIS,
        }
    ]

    markdown = MarkdownReporter(include_timestamp=False).render_all([_validation_result_from_payload(payload)])

    assert "### Next Steps" in markdown
    assert (
        "1. **only opencode was evaluated** [agent · hypothesis]: Run the same cases with a second agent." in markdown
    )
    assert "   - Check: A second agent shows the same direction." in markdown
    assert "   - Evidence: opencode +0 pp" in markdown


def test_markdown_report_uses_next_steps_from_saved_run_findings(tmp_path: Path) -> None:
    run = _run(tmp_path, goal_accuracy=1.0)
    render_agent_eval_html_report(tmp_path, run, use_llm_judge=False)

    result = agent_eval_result_from_directory(tmp_path, run, use_llm_judge=False)
    assert result is not None
    assert result.metadata["agent_eval"]["next_steps"]

    markdown = MarkdownReporter(include_timestamp=False).render_all([result])

    assert "### Next Steps" in markdown
    assert "each case ran once, so a single attempt decides every pass or fail mark" in markdown
    assert "Check: With 5 or more skill cases the verdict keeps its direction." in markdown
