# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge fallback exclusions survive collection and canonical report reloads."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from skillevaluator.evaluation import tier3_report
from skillevaluator.evaluation.tier3_report import (
    _validation_result_from_payload,
    agent_eval_result_from_directory,
    build_agent_eval_payload,
    render_agent_eval_html_report,
)
from skillevaluator.reporting import HTMLReporter
from skillevaluator.reporting.html import _sanitize_tier3_display_payload
from skillevaluator.tier3.harbor.collector import _extract_rewards, collect_harbor_results
from skillevaluator.tier3.harbor.metrics import (
    DEFAULT_METRIC_SET,
    DEFAULT_METRICS,
    DEFAULT_SCORE_POLICY,
    mark_not_applicable,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_job(
    job: Path,
    accuracies: list[float],
    judges: list[tuple[str | None, ...]],
    *,
    native: bool,
    judge_metric: str = "accuracy",
    activated: bool | None = None,
) -> None:
    trials = []
    for index, (accuracy, step_judges) in enumerate(zip(accuracies, judges, strict=True), 1):
        case = f"case-{index}"
        trial_name = f"{case}__trial"
        trials.append(trial_name)
        trial = job / trial_name
        scores = {**dict.fromkeys(DEFAULT_METRICS, 0.5), judge_metric: accuracy}
        steps = []
        for step_index, judge in enumerate(step_judges):
            name = f"step-{step_index}"
            verifier = trial / "steps" / name / "verifier" if native else trial / "verifier"
            rich = {
                **scores,
                "entry_id": case,
                "metric_set": DEFAULT_METRIC_SET,
                "score_policy": DEFAULT_SCORE_POLICY,
                "details": {judge_metric: {"score": accuracy, "judge_model_used": judge}} if judge else {},
            }
            if activated is not None:
                rich["details"]["skill_execution"] = {"activation": {"passed": activated, "score": float(activated)}}
            _write_json(verifier / "reward.json", scores if native else rich)
            if native:
                _write_json(verifier / "skill_evaluator_reward.json", rich)
                steps.append({"step_name": name, "verifier_result": {"rewards": scores}})
        if native:
            _write_json(
                trial / "result.json",
                {
                    "trial_name": trial_name,
                    "task_name": case,
                    "verifier_result": {"rewards": scores},
                    "step_results": steps,
                },
            )
    _write_json(
        job / "result.json",
        {
            "n_total_trials": len(trials),
            "stats": {
                "n_completed_trials": len(trials),
                "n_errored_trials": 0,
                "n_running_trials": 0,
                "n_pending_trials": 0,
                "n_cancelled_trials": 0,
                "n_retries": 0,
                "evals": {
                    "agent__model___tasks": {
                        "n_trials": len(trials),
                        "n_errors": 0,
                        "reward_stats": {"overall": {"0.5": trials}},
                    }
                },
            },
        },
    )


def _collect(
    root: Path,
    *,
    native: bool = False,
    all_mixed: bool = False,
    judge_free: bool = False,
    cases: int = 2,
    judge_metric: str = "accuracy",
    activated: bool | None = None,
) -> tuple[dict, Path]:
    judges_with = [("primary", "fallback" if index == 0 or all_mixed else "primary") for index in range(cases)]
    judges_without = [("primary", "primary")] * cases
    accuracies_with = [1.0, *([0.0] * (cases - 1))]
    accuracies_without = [0.0, *([0.4] * (cases - 1))]
    if judge_free:
        judges_with.append((None, None))
        judges_without.append((None, None))
        accuracies_with.append(1.0)
        accuracies_without.append(1.0)
    if not native:
        judges_with = [(row[-1],) for row in judges_with]
        judges_without = [(row[-1],) for row in judges_without]
    _write_job(
        root / "jobs" / "demo-codex-with",
        accuracies_with,
        judges_with,
        native=native,
        judge_metric=judge_metric,
        activated=activated,
    )
    _write_job(
        root / "jobs" / "demo-codex-without",
        accuracies_without,
        judges_without,
        native=native,
        judge_metric=judge_metric,
    )
    run = root / "run"
    collected = collect_harbor_results(
        "demo",
        ["codex"],
        run,
        root / "jobs",
        expected_case_ids=[f"case-{index + 1}" for index in range(len(accuracies_with))],
        expected_trials=len(accuracies_with),
    )
    assert collected["execution_status"] == "succeeded"
    return collected, run


def _report(run: Path) -> dict:
    return agent_eval_result_from_directory(run.parent / "demo", run, use_llm_judge=False).metadata["agent_eval"]


@pytest.mark.parametrize("activated", [False, True])
def test_judge_filtered_lift_respects_activation_eligibility(tmp_path: Path, activated: bool) -> None:
    collected, run = _collect(tmp_path, activated=activated)
    report = _report(run)
    agent = report["agents"]["codex"]
    assert agent["skill_activation"]["status"] == ("activated" if activated else "never_activated")
    assert report["overall_lift"] == agent["lift"] == (-0.08 if activated else None)
    assert agent["evaluators"]["accuracy"]["lift"] == (-0.4 if activated else None)
    correctness = next(dim for dim in agent["dimensions"] if dim["id"] == "correctness")
    assert correctness["lift"] == (-0.4 if activated else None)
    if activated:
        assert collected["agents"]["codex"]["lift"]["judge_mixed_cases"] == {"accuracy": 1}
    else:
        assert report["verdict"] == "inconclusive"
        assert collected["agents"]["codex"]["lift"] == {}
        assert all(dimension["lift"] is None for dimension in agent["dimensions"])


@pytest.mark.parametrize("native", [False, True])
def test_report_retains_judge_filtered_regression(tmp_path: Path, native: bool) -> None:
    collected, run = _collect(tmp_path, native=native)
    lift = collected["agents"]["codex"]["lift"]
    assert lift["judge_mixed_cases"] == {"accuracy": 1}
    assert lift["accuracy"]["delta"] == -0.4
    assert lift["overall"]["delta"] == -0.08

    # A stale cache must not reverse the comparison or its explanation.
    cached = run / "codex" / "lift.json"
    payload = json.loads(cached.read_text())
    payload["accuracy"]["delta"] = 0.9
    payload["overall"]["delta"] = 0.9
    _write_json(cached, payload)
    summary = run / "codex" / "with-skill" / "summary.json"
    payload = json.loads(summary.read_text())
    payload["dimensions"]["correctness"].update(
        explanation="+0.30 lift over baseline 0.20.",
        reasoning_bullets=["Improvement!"],
        sources={"goal_accuracy": 0.5},
    )
    _write_json(summary, payload)

    report = _report(run)
    agent = report["agents"]["codex"]
    assert report["overall_lift"] == agent["lift"] == -0.08
    assert agent["with_skill"] == 0.5
    assert agent["baseline"] == 0.44
    assert agent["evaluators"]["accuracy"]["lift"] == -0.4
    correctness = next(dim for dim in agent["dimensions"] if dim["id"] == "correctness")
    assert correctness["lift"] == -0.4
    assert "-0.40 regression versus baseline 0.40" in correctness["explanation"]
    assert "different judge models" in correctness["explanation"]
    assert "+0.30" not in correctness["explanation"]


@pytest.mark.parametrize("native", [False, True])
def test_all_mixed_judges_leave_metric_dimension_and_overall_unavailable(tmp_path: Path, native: bool) -> None:
    collected, run = _collect(tmp_path, native=native, all_mixed=True)
    lift = collected["agents"]["codex"]["lift"]
    assert lift["judge_mixed_cases"] == {"accuracy": 2}
    assert "accuracy" not in lift
    assert "overall" not in lift
    report = _report(run)
    agent = report["agents"]["codex"]
    assert report["overall_lift"] is None
    assert agent["lift"] is None
    assert agent["evaluators"]["accuracy"]["lift"] is None
    correctness = next(dim for dim in agent["dimensions"] if dim["id"] == "correctness")
    assert correctness["lift"] is None
    assert "Lift unavailable" in correctness["explanation"]
    assert "No baseline run" not in correctness["explanation"]


@pytest.mark.parametrize(
    "missing_evidence", ["reward", "judge_metadata", "partial_judge_metadata", "retained_judge_metadata"]
)
def test_report_handles_incomplete_judge_evidence(tmp_path: Path, missing_evidence: str) -> None:
    collected, run = _collect(tmp_path, all_mixed=missing_evidence == "partial_judge_metadata")
    reward = next((run / "codex" / "with-skill" / "trials").glob("*/reward.json"))
    if missing_evidence == "reward":
        # Legacy artifacts without an explicit case basis retain their bounded judge fallback.
        lift_path = run / "codex" / "lift.json"
        legacy_lift = json.loads(lift_path.read_text())
        legacy_lift.pop("basis", None)
        _write_json(lift_path, legacy_lift)
        reward.unlink()
    else:
        for reward in (run / "codex").glob("*/trials/*/reward.json"):
            payload = json.loads(reward.read_text())
            if (
                missing_evidence in {"partial_judge_metadata", "retained_judge_metadata"}
                and payload["entry_id"] != "case-2"
            ):
                continue
            payload.pop("details", None)
            _write_json(reward, payload)
    report = _report(run)
    lift = collected["agents"]["codex"]["lift"]
    available = missing_evidence in {"reward", "retained_judge_metadata"}
    assert report["overall_lift"] == (lift["overall"]["delta"] if available else None)
    assert report["agents"]["codex"]["evaluators"]["accuracy"]["lift"] == (
        lift["accuracy"]["delta"] if available else None
    )


@pytest.mark.parametrize("native", [False, True])
def test_judge_free_cases_retain_collector_lift(tmp_path: Path, native: bool) -> None:
    collected, run = _collect(tmp_path, native=native, judge_free=True)
    lift = collected["agents"]["codex"]["lift"]
    assert lift["judge_mixed_cases"] == {"accuracy": 1}
    assert lift["accuracy"]["delta"] == -0.2
    assert lift["overall"]["delta"] == -0.04
    report = _report(run)
    agent = report["agents"]["codex"]
    assert report["overall_lift"] == agent["lift"] == lift["overall"]["delta"]
    assert agent["evaluators"]["accuracy"]["lift"] == lift["accuracy"]["delta"]
    correctness = next(dim for dim in agent["dimensions"] if dim["id"] == "correctness")
    assert correctness["lift"] == lift["accuracy"]["delta"]


@pytest.mark.parametrize("recorded_lift", [True, False])
@pytest.mark.parametrize(
    ("metric", "dimension", "dimension_lift"),
    [("accuracy", "correctness", -0.4), ("goal_accuracy", "effectiveness", -0.2)],
)
def test_capped_report_uses_recorded_lift_when_available(
    tmp_path: Path, recorded_lift: bool, metric: str, dimension: str, dimension_lift: float
) -> None:
    collected, run = _collect(tmp_path, cases=513, judge_metric=metric)
    lift = collected["agents"]["codex"]["lift"]
    assert lift["judge_mixed_cases"] == {metric: 1}
    assert lift[metric]["delta"] == -0.4
    assert lift["overall"]["delta"] == dimension_lift / 5
    if not recorded_lift:
        (run / "codex" / "lift.json").unlink()
    else:
        # This compatibility test covers pre-basis runs; explicit-basis runs require full rewards.
        legacy_lift = dict(lift)
        legacy_lift.pop("basis", None)
        _write_json(run / "codex" / "lift.json", legacy_lift)
    report = _report(run)
    agent = report["agents"]["codex"]
    assert len(agent["trials"]) == 512
    assert report["overall_lift"] == agent["lift"] == (lift["overall"]["delta"] if recorded_lift else None)
    assert agent["evaluators"][metric]["lift"] == (lift[metric]["delta"] if recorded_lift else None)
    dimension = next(dim for dim in agent["dimensions"] if dim["id"] == dimension)
    assert dimension["lift"] == (dimension_lift if recorded_lift else None)


@pytest.mark.parametrize("recorded_mixed", [None, {"accuracy": 0}, {"accuracy": True}])
def test_partial_load_does_not_trust_lift_without_recorded_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded_mixed: dict | None
) -> None:
    from skillevaluator.tier3.harbor import report_data

    _, run = _collect(tmp_path)
    path = run / "codex" / "lift.json"
    lift = json.loads(path.read_text())
    lift["judge_mixed_cases"] = recorded_mixed
    _write_json(path, lift)
    monkeypatch.setattr(report_data, "_MAX_TRIALS_PER_CONDITION", 1)
    report = _report(run)
    assert report["overall_lift"] is None
    assert report["agents"]["codex"]["evaluators"]["accuracy"]["lift"] is None


def test_native_aggregate_keeps_numeric_authority_and_only_declared_step_judges(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _write_job(job, [0.25], [("primary", "fallback")], native=True)
    trial = job / "case-1__trial"
    for sidecar in trial.glob("steps/*/verifier/skill_evaluator_reward.json"):
        payload = json.loads(sidecar.read_text())
        payload["accuracy"] = 1.0
        _write_json(sidecar, payload)
    _write_json(
        trial / "steps" / "orphan" / "verifier" / "skill_evaluator_reward.json",
        {"details": {"accuracy": {"judge_model_used": "unrelated"}}},
    )
    rows = _extract_rewards(job)
    assert len(rows) == 1
    assert rows[0]["accuracy"] == 0.25
    assert rows[0]["_judge_models"] == {"accuracy": ["fallback", "primary"]}
    assert rows[0]["details"]["accuracy"]["judge_models"] == ["fallback", "primary"]


# Matches the note in every wording; the stylesheet names the class without this attribute.
_LIFT_NOTE_DIV = 'class="t3-dim-hint t3-eval-lift-note"'
_TRIAL_LABEL = "Left out of lift: different judge models"
_BASIS_REASON = (
    "Cases whose arms were scored by different judge models are left out of this metric's lift; "
    "the arm scores include them."
)


def _card(agent: dict, metric: str = "accuracy") -> dict:
    return next(card for card in agent["evaluator_cards"] if card["id"] == metric)


def _card_html(html: str, label: str) -> str:
    """Return one evaluator card's markup, failing unless its ``div`` elements close."""
    for match in re.finditer(r'<div class="t3-eval-card ', html):
        depth = 0
        for tag in re.finditer(r"<(/?)div\b", html[match.start() :]):
            depth += -1 if tag.group(1) else 1
            if depth == 0:
                card = html[match.start() : match.start() + tag.end()]
                break
        else:
            raise AssertionError("unbalanced evaluator card markup")
        if f'<div class="t3-eval-title">{label}</div>' in card:
            return card
    raise AssertionError(f"no {label} evaluator card")


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment)).strip()


def _render(payload: dict) -> str:
    result = _validation_result_from_payload(payload)
    assert result is not None
    return HTMLReporter(include_timestamp=False).render_all([result])


@pytest.mark.parametrize("all_mixed", [False, True])
def test_evaluator_card_names_the_cases_its_lift_left_out(tmp_path: Path, all_mixed: bool) -> None:
    _, run = _collect(tmp_path, all_mixed=all_mixed)
    report = _report(run)
    agent = report["agents"]["codex"]
    mixed = ["case-1", "case-2"] if all_mixed else ["case-1"]
    evaluator = agent["evaluators"]["accuracy"]
    # The arm scores keep every scored case; only the lift leaves the mixed cases out.
    assert (evaluator["with_skill"], evaluator["baseline"]) == (0.5, 0.2)
    assert evaluator["lift"] == (None if all_mixed else -0.4)
    assert evaluator["lift_excluded_cases"] == len(mixed)
    assert all(
        "lift_excluded_cases" not in agent["evaluators"][metric] for metric in DEFAULT_METRICS if metric != "accuracy"
    )
    card = _card(agent)
    assert (card["with_skill"], card["baseline"], card["lift"]) == (0.5, 0.2, evaluator["lift"])
    assert card["comparison_basis"] == {
        "excluded_cases": len(mixed),
        "case_ids": mixed,
        "excluded_attempts": len(mixed),
        "contributing_attempts": 0 if all_mixed else 1,
        "reason": _BASIS_REASON,
    }
    assert [item["entry_id"] for item in card["evidence"] if item.get("comparison_excluded")] == mixed
    assert all("comparison_basis" not in other for other in agent["evaluator_cards"] if other["id"] != "accuracy")
    assert _card(report) == card

    skill = tmp_path / "demo"
    skill.mkdir()
    html = render_agent_eval_html_report(skill, run, use_llm_judge=False).read_text(encoding="utf-8")
    card_html = _card_html(html, "Accuracy")
    note = (
        "No lift: 2 cases were scored by different judge models and no comparable result remains; "
        "the arm scores include them."
        if all_mixed
        else "Lift leaves out 1 case scored by different judge models; the arm scores include it."
    )
    assert note in _text(card_html)
    assert ("Lift leaves out" in _text(card_html)) is not all_mixed
    assert ("-0.40 lift" in _text(card_html)) is not all_mixed
    assert card_html.count(_TRIAL_LABEL) == len(mixed)
    assert html.count(_LIFT_NOTE_DIV) == 1


def _attempt(case: str, trial_id: str, accuracy: float | None, judge: str | None) -> dict:
    reward = {**dict.fromkeys(DEFAULT_METRICS, 0.5), "entry_id": case, "trial_id": trial_id, "details": {}}
    if accuracy is None:
        mark_not_applicable(reward, "accuracy")
    else:
        reward.update(accuracy=accuracy, details={"accuracy": {"score": accuracy, "judge_model_used": judge}})
    return reward


def _info(with_rewards: list[dict], baseline_rewards: list[dict]) -> dict:
    succeeded = {"execution_status": "succeeded"}
    return {
        "execution_status": "succeeded",
        "expected_attempts": len(with_rewards) + len(baseline_rewards),
        "scored_attempts": len(with_rewards) + len(baseline_rewards),
        "num_trials": len(with_rewards),
        "num_trials_baseline": len(baseline_rewards),
        "with_skill": {**dict.fromkeys(DEFAULT_METRICS, 0.5), "accuracy": 0.5},
        "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.5),
        "rewards": with_rewards,
        "rewards_baseline": baseline_rewards,
        "rewards_complete": True,
        "rewards_baseline_complete": True,
        "conditions": {"with_skill": succeeded, "without_skill": succeeded},
    }


def test_comparison_basis_counts_repeated_attempts_apart_from_na_and_unpaired_cases() -> None:
    info = _info(
        [
            _attempt("kept", "kept__a1", 1.0, "primary"),
            _attempt("mixed", "mixed__a1", 0.0, "fallback"),
            _attempt("mixed", "mixed__a2", 0.5, "primary"),
            _attempt("mixed", "mixed__a3", None, None),
        ],
        [
            _attempt("kept", "kept__b1", 0.5, "primary"),
            _attempt("mixed", "mixed__b1", 0.5, "primary"),
            # A case only the baseline scored is unpaired, not judge-mixed.
            _attempt("baseline-only", "baseline-only__b1", 0.5, "fallback"),
        ],
    )
    payload = build_agent_eval_payload("demo", {"codex": info}, use_llm_judge=False)
    assert payload is not None
    agent = payload["agents"]["codex"]
    assert agent["evaluators"]["accuracy"]["lift"] == 0.5
    assert agent["evaluators"]["accuracy"]["lift_excluded_cases"] == 1
    basis = _card(agent)["comparison_basis"]
    # Two scored attempts of the mixed case are left out; its N/A attempt never had a score.
    assert (basis["case_ids"], basis["excluded_attempts"], basis["contributing_attempts"]) == (["mixed"], 2, 1)
    flagged = {item["entry_id"]: item.get("comparison_excluded", False) for item in _card(agent)["evidence"]}
    assert flagged == {"kept": False, "mixed": True}


def test_matching_judges_add_no_lift_note() -> None:
    info = _info(
        [_attempt("case-1", "case-1__a1", 1.0, "primary"), _attempt("case-2", "case-2__a1", 0.0, "primary")],
        [_attempt("case-1", "case-1__b1", 0.5, "primary"), _attempt("case-2", "case-2__b1", 0.5, "primary")],
    )
    payload = build_agent_eval_payload("demo", {"codex": info}, use_llm_judge=False)
    assert payload is not None
    agent = payload["agents"]["codex"]
    assert agent["evaluators"]["accuracy"]["lift"] == 0.0
    assert all("lift_excluded_cases" not in evaluator for evaluator in agent["evaluators"].values())
    assert all("comparison_basis" not in card for card in agent["evaluator_cards"])
    assert all("comparison_excluded" not in item for card in agent["evaluator_cards"] for item in card["evidence"])
    html = _render(payload)
    assert _LIFT_NOTE_DIV not in html
    assert _TRIAL_LABEL not in html


def test_saved_run_without_the_mixed_case_rewards_shows_only_the_recorded_count(tmp_path: Path) -> None:
    collected, run = _collect(tmp_path)
    lift_path = run / "codex" / "lift.json"
    legacy_lift = json.loads(lift_path.read_text(encoding="utf-8"))
    legacy_lift.pop("basis", None)
    _write_json(lift_path, legacy_lift)
    for reward in (run / "codex" / "with-skill" / "trials").glob("*/reward.json"):
        if json.loads(reward.read_text(encoding="utf-8"))["entry_id"] == "case-1":
            reward.unlink()
    agent = _report(run)["agents"]["codex"]
    # The recorded lift and its exclusion count stand; no loaded reward names the case.
    assert agent["evaluators"]["accuracy"]["lift"] == collected["agents"]["codex"]["lift"]["accuracy"]["delta"]
    assert agent["evaluators"]["accuracy"]["lift_excluded_cases"] == 1
    card = _card(agent)
    assert card["comparison_basis"] == {
        "excluded_cases": 1,
        "case_ids": [],
        "excluded_attempts": None,
        "contributing_attempts": None,
        "reason": _BASIS_REASON,
    }
    assert [item["entry_id"] for item in card["evidence"]] == ["case-2"]
    assert not any(item.get("comparison_excluded") for item in card["evidence"])

    skill = tmp_path / "demo"
    skill.mkdir()
    card_html = _card_html(
        render_agent_eval_html_report(skill, run, use_llm_judge=False).read_text(encoding="utf-8"), "Accuracy"
    )
    assert "Lift leaves out 1 case scored by different judge models" in _text(card_html)
    assert _TRIAL_LABEL not in card_html


def test_trimmed_card_evidence_keeps_the_lift_note(monkeypatch: pytest.MonkeyPatch) -> None:
    info = _info(
        [_attempt("case-1", "case-1__a1", 1.0, "fallback"), _attempt("case-2", "case-2__a1", 0.0, "primary")],
        [_attempt("case-1", "case-1__b1", 0.5, "primary"), _attempt("case-2", "case-2__b1", 0.5, "primary")],
    )
    real_size = tier3_report._serialized_payload_size

    def size_until_evidence_trimmed(payload: dict) -> int:
        cards = [card for agent in payload.get("agents", {}).values() for card in agent.get("evaluator_cards", [])]
        evidence = any(card.get("evidence") for card in cards)
        return real_size(payload) + (tier3_report._MAX_EMBEDDED_REPORT_BYTES if evidence else 0)

    monkeypatch.setattr(tier3_report, "_serialized_payload_size", size_until_evidence_trimmed)
    payload = build_agent_eval_payload("demo", {"codex": info}, use_llm_judge=False)
    assert payload is not None
    assert payload["report_truncation"]["omitted"]["evidence_entries"] > 0
    card = _card(payload["agents"]["codex"])
    assert card["evidence"] == []
    assert card["comparison_basis"]["excluded_cases"] == 1
    card_html = _card_html(_render(payload), "Accuracy")
    assert "Lift leaves out 1 case scored by different judge models" in _text(card_html)
    assert _TRIAL_LABEL not in card_html


def _mixed_info() -> dict:
    """One case scored by a fallback judge in the with-skill arm, one by matching judges."""
    return _info(
        [_attempt("case-1", "case-1__a1", 1.0, "fallback"), _attempt("case-2", "case-2__a1", 0.0, "primary")],
        [_attempt("case-1", "case-1__b1", 0.5, "primary"), _attempt("case-2", "case-2__b1", 0.5, "primary")],
    )


def test_never_activated_agent_gets_no_lift_note() -> None:
    info = {**_mixed_info(), "skill_activation": {"status": "never_activated"}}
    payload = build_agent_eval_payload("demo", {"codex": info}, use_llm_judge=False)
    assert payload is not None
    agent = payload["agents"]["codex"]
    assert agent["verdict"] == tier3_report.VERDICT_INCONCLUSIVE
    # The agent has no lift at all, so nothing explains what its lift left out.
    assert agent["evaluators"]["accuracy"]["lift"] is None
    assert "lift_excluded_cases" not in agent["evaluators"]["accuracy"]
    card = _card(agent)
    assert card["lift"] is None
    assert "comparison_basis" not in card
    assert not any(item.get("comparison_excluded") for item in card["evidence"])
    html = _render(payload)
    assert _LIFT_NOTE_DIV not in html
    assert _TRIAL_LABEL not in html


@pytest.mark.parametrize(
    "basis",
    [
        {"excluded_cases": True},
        {"excluded_cases": "3"},
        {"excluded_cases": -1},
        {"excluded_cases": 2**70},
        {"excluded_cases": 1.0},
        ["excluded_cases", 1],
    ],
)
def test_html_ignores_a_malformed_comparison_basis_and_flag(basis: object) -> None:
    info = _info(
        [_attempt("case-1", "case-1__a1", 1.0, "primary"), _attempt("case-2", "case-2__a1", 0.0, "primary")],
        [_attempt("case-1", "case-1__b1", 0.5, "primary"), _attempt("case-2", "case-2__b1", 0.5, "primary")],
    )
    payload = build_agent_eval_payload("demo", {"codex": info}, use_llm_judge=False)
    assert payload is not None
    for card in (_card(payload["agents"]["codex"]), _card(payload)):
        card["comparison_basis"] = basis
        for item in card["evidence"]:
            item["comparison_excluded"] = "yes"
    html = _render(payload)
    assert _LIFT_NOTE_DIV not in html
    assert _TRIAL_LABEL not in html


def test_html_keeps_comparison_basis_fields_only_in_their_types() -> None:
    payload = build_agent_eval_payload("demo", {"codex": _mixed_info()}, use_llm_judge=False)
    assert payload is not None
    for attempts in (True, "1", -1, 2**70, 1.0):
        card = _card(payload["agents"]["codex"])
        card["comparison_basis"] = {
            "excluded_cases": 1,
            "case_ids": "case-1",
            "excluded_attempts": attempts,
            "contributing_attempts": attempts,
            "reason": 5,
        }
        card["evidence"][0]["comparison_excluded"] = 1
        sanitized = _card(_sanitize_tier3_display_payload(payload)["agents"]["codex"])
        assert sanitized["comparison_basis"] == {
            "excluded_cases": 1,
            "case_ids": [],
            "excluded_attempts": None,
            "contributing_attempts": None,
            "reason": "",
        }
        assert not any("comparison_excluded" in item for item in sanitized["evidence"])


def test_comparison_basis_caps_case_ids_and_lists_each_cut_id_once() -> None:
    with_rewards = [_attempt("kept", "kept__a1", 1.0, "primary")]
    baseline_rewards = [_attempt("kept", "kept__b1", 0.5, "primary")]
    for index in range(70):
        case = f"case-{index:03d}-" + "x" * 300
        with_rewards.append(_attempt(case, f"{case}__a1", 1.0, "fallback"))
        baseline_rewards.append(_attempt(case, f"{case}__b1", 0.5, "primary"))
    # Two keys that match in their first 256 characters and sort first.
    shared_prefix = "a" * 260
    for case in (f"{shared_prefix}-a", f"{shared_prefix}-b"):
        with_rewards.append(_attempt(case, f"{case}__a1", 1.0, "fallback"))
        baseline_rewards.append(_attempt(case, f"{case}__b1", 0.5, "primary"))
    payload = build_agent_eval_payload("demo", {"codex": _info(with_rewards, baseline_rewards)}, use_llm_judge=False)
    assert payload is not None
    agent = payload["agents"]["codex"]
    assert agent["evaluators"]["accuracy"]["lift_excluded_cases"] == 72
    basis = _card(agent)["comparison_basis"]
    # The count stays exact while the list stops at 64 cut keys, and the two
    # keys that match once cut appear once.
    assert basis["excluded_cases"] == 72
    assert basis["case_ids"] == ["a" * 256, *(f"case-{index:03d}-".ljust(256, "x") for index in range(63))]
    assert len(set(basis["case_ids"])) == 64
    assert (basis["excluded_attempts"], basis["contributing_attempts"]) == (72, 1)


def test_comparison_basis_leaves_out_trials_without_a_resolved_case() -> None:
    info = _info(
        [
            _attempt("kept", "kept__a1", 1.0, "primary"),
            _attempt("mixed", "mixed__a1", 0.0, "fallback"),
            {**_attempt("", "unresolved__a1", 1.0, "fallback"), "entry_id": ""},
        ],
        [
            _attempt("kept", "kept__b1", 0.5, "primary"),
            _attempt("mixed", "mixed__b1", 0.5, "primary"),
            {**_attempt("", "unresolved__b1", 0.5, "primary"), "entry_id": ""},
        ],
    )
    payload = build_agent_eval_payload("demo", {"codex": info}, use_llm_judge=False)
    assert payload is not None
    agent = payload["agents"]["codex"]
    # Lift pairing never compares a trial without a case, so it is not a left-out case.
    assert agent["evaluators"]["accuracy"]["lift_excluded_cases"] == 1
    card = _card(agent)
    basis = card["comparison_basis"]
    assert (basis["case_ids"], basis["excluded_attempts"], basis["contributing_attempts"]) == (["mixed"], 1, 1)
    assert [item["entry_id"] for item in card["evidence"] if item.get("comparison_excluded")] == ["mixed"]


def _split_trial_info() -> dict:
    """A with-skill trial whose rows name two cases: the logical load is incomplete."""
    return _info(
        [
            _attempt("mixed", "mixed__a1", 1.0, "fallback"),
            _attempt("kept", "kept__a1", 0.0, "primary"),
            _attempt("odd-a", "shared", 0.5, "primary"),
            _attempt("odd-b", "shared", 0.5, "primary"),
        ],
        [
            _attempt("mixed", "mixed__b1", 0.5, "primary"),
            _attempt("kept", "kept__b1", 0.5, "primary"),
            _attempt("odd-a", "odd-a__b1", 0.5, "primary"),
            _attempt("odd-b", "odd-b__b1", 0.5, "primary"),
        ],
    )


def _recorded_count_info() -> dict:
    """``lift.json`` records two mixed cases; the loaded rewards identify one."""
    return {**_mixed_info(), "lift": {"basis": {}, "judge_mixed_cases": {"accuracy": 2}}}


@pytest.mark.parametrize(
    ("make_info", "excluded_cases"),
    [(_split_trial_info, 1), (_recorded_count_info, 2)],
    ids=["trial-rows-name-two-cases", "recorded-count-exceeds-identified-cases"],
)
def test_attempt_counts_stay_null_unless_the_load_identifies_every_left_out_case(
    make_info, excluded_cases: int
) -> None:
    payload = build_agent_eval_payload("demo", {"codex": make_info()}, use_llm_judge=False)
    assert payload is not None
    agent = payload["agents"]["codex"]
    # Both arms' rewards loaded and the lift basis is published, but it does not
    # describe every left-out case, so neither attempt count is published.
    assert agent["lift_basis"]["metrics"]["accuracy"]["trials"]["with_skill"] == 1
    assert agent["evaluators"]["accuracy"]["lift"] is None
    assert agent["evaluators"]["accuracy"]["lift_excluded_cases"] == excluded_cases
    basis = _card(agent)["comparison_basis"]
    assert basis["excluded_cases"] == excluded_cases
    assert basis["case_ids"] == ["case-1" if excluded_cases == 2 else "mixed"]
    assert (basis["excluded_attempts"], basis["contributing_attempts"]) == (None, None)
    card_html = _text(_card_html(_render(payload), "Accuracy"))
    noun = "1 case was" if excluded_cases == 1 else "2 cases were"
    # A matching-judge case can still exist, so the note does not say none remains.
    assert (
        f"No lift: {noun} scored by different judge models and the loaded results cannot confirm the comparison; "
        f"the arm scores include {'it' if excluded_cases == 1 else 'them'}."
    ) in card_html
    assert "no comparable result remains" not in card_html


def _doc_block(name: str, lead: str) -> str:
    """Return the one blank-line separated block of ``name`` that starts with ``lead``, whitespace-normalized."""
    text = (Path(__file__).resolve().parents[1] / name).read_text(encoding="utf-8")
    [block] = [" ".join(part.split()) for part in re.split(r"\n\s*\n", text) if " ".join(part.split()).startswith(lead)]
    return block


def test_docs_and_changelog_say_when_each_no_lift_note_applies() -> None:
    """A card with no lift says nothing comparable remains only when no attempt fed the lift."""
    entry = _doc_block("CHANGELOG.md", "- A Tier 3 HTML evaluator card with no lift for a judge metric no longer says")
    assert (
        "A card with no lift says that no comparable result remains when the loaded results show that no attempt "
        "fed the lift, and otherwise that the loaded results cannot confirm the comparison"
    ) in entry
    docs = _doc_block("docs/reports.mdx", "For each LLM metric (`accuracy`, `goal_accuracy`, and `behavior_check`)")
    assert (
        "When it has no lift, the card says that no comparable result remains if its `comparison_basis` records "
        "`contributing_attempts` as 0, and otherwise that the loaded results cannot confirm the comparison."
    ) in docs


def test_lift_note_mentions_arm_scores_only_when_both_arms_have_one() -> None:
    info = _mixed_info()
    # The baseline arm failed, so it has no arm score, although its rewards loaded.
    info["conditions"]["without_skill"] = {"execution_status": "failed"}
    payload = build_agent_eval_payload("demo", {"codex": info}, use_llm_judge=False)
    assert payload is not None
    agent = payload["agents"]["codex"]
    evaluator = agent["evaluators"]["accuracy"]
    assert (evaluator["baseline"], evaluator["lift_excluded_cases"]) == (None, 1)
    basis = _card(agent)["comparison_basis"]
    assert (
        basis["reason"] == "Cases whose arms were scored by different judge models are left out of this metric's lift."
    )
    card_html = _text(_card_html(_render(payload), "Accuracy"))
    assert "Lift leaves out 1 case scored by different judge models." in card_html
    assert "the arm scores include" not in card_html


@pytest.mark.parametrize(
    ("recorded", "published"),
    [(3, 3), (2**53 - 1, 2**53 - 1), (2**53, None), (10**30, None)],
)
def test_recorded_exclusion_count_is_published_only_when_json_carries_it_exactly(
    recorded: int, published: int | None
) -> None:
    info = _info(
        [_attempt("case-1", "case-1__a1", 1.0, "primary"), _attempt("case-2", "case-2__a1", 0.0, "primary")],
        [_attempt("case-1", "case-1__b1", 0.5, "primary"), _attempt("case-2", "case-2__b1", 0.5, "primary")],
    )
    info.update(lift={"judge_mixed_cases": {"accuracy": recorded}}, rewards_complete=False)
    payload = build_agent_eval_payload("demo", {"codex": info}, use_llm_judge=False)
    assert payload is not None
    agent = payload["agents"]["codex"]
    assert agent["evaluators"]["accuracy"].get("lift_excluded_cases", None) == published
    assert ("lift_excluded_cases" in agent["evaluators"]["accuracy"]) is (published is not None)
    card = _card(agent)
    assert ("comparison_basis" in card) is (published is not None)
    if published is not None:
        assert card["comparison_basis"]["excluded_cases"] == published


@pytest.mark.parametrize(
    ("complete", "metrics_basis", "attempts"),
    [
        (True, {"accuracy": {"cases": 1, "trials": {"with_skill": 1, "without_skill": 1}}}, (1, 1)),
        (False, {"accuracy": {"cases": 1, "trials": {"with_skill": 1, "without_skill": 1}}}, (None, None)),
        # A metric the basis does not describe publishes neither count.
        (True, {}, (None, None)),
        (True, {"accuracy": {"cases": 1, "trials": {"with_skill": "1"}}}, (None, None)),
    ],
)
def test_comparison_basis_publishes_both_attempt_counts_or_neither(
    complete: bool, metrics_basis: dict, attempts: tuple[int | None, int | None]
) -> None:
    basis = tier3_report._card_comparison_basis(
        "accuracy",
        1,
        {"case-1"},
        {"case_ids": {"case-1"}, "excluded_attempts": 1, "complete": complete},
        {"metrics": metrics_basis},
        arm_scores=True,
    )
    assert (basis["excluded_attempts"], basis["contributing_attempts"]) == attempts
