# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private comparisons share the existing launcher and its trial budget."""

from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from skillevaluator.tier3.harbor import runner


def arguments(tmp_path):
    return {
        "skill_name": "bundle",
        "agent": "claude-code",
        "model": "fixture-model",
        "env_mode": "docker",
        "with_skill": tmp_path / "with",
        "baseline": tmp_path / "without",
        "jobs_dir": tmp_path / "jobs",
        "run_env": {},
        "n_attempts": 3,
        "n_concurrent": 2,
        "timeout_multiplier": 1.0,
        "override_cpus": None,
        "override_memory_mb": None,
        "override_storage_mb": None,
        "expected_trials": 6,
        "agent_import_path": "fixture:ordinary",
        "with_agent_import_path": "fixture:native",
        "additional_arms": (
            ("sumofparts", tmp_path / "members", "fixture:ordinary"),
            ("looseparts", tmp_path / "loose", "fixture:loose"),
        ),
    }


def test_four_arms_share_the_per_agent_trial_budget(tmp_path, monkeypatch):
    calls, active, maximum = [], 0, 0
    lock, barrier = Lock(), Barrier(2)

    def launch(**kwargs):
        nonlocal active, maximum
        with lock:
            calls.append(kwargs)
            active += 1
            maximum = max(maximum, active)
        barrier.wait(timeout=10)
        with lock:
            active -= 1
        return True, ""

    monkeypatch.setattr(runner, "_run_harbor", launch)
    assert runner._run_agent_pair(**arguments(tmp_path)) == []
    assert maximum == 2
    by_variant = {call["job_name"].rsplit("-", 1)[1]: call for call in calls}
    assert set(by_variant) == {"with", "without", "sumofparts", "looseparts"}
    assert {variant: call["agent_import_path"] for variant, call in by_variant.items()} == {
        "with": "fixture:native",
        "without": "fixture:ordinary",
        "sumofparts": "fixture:ordinary",
        "looseparts": "fixture:loose",
    }
    assert all(call["n_concurrent"] == 1 and call["n_attempts"] == 3 for call in calls)
    assert all(call["model"] == "fixture-model" and call["expected_trials"] == 6 for call in calls)


def test_added_arm_launch_failures_stay_advisory(tmp_path, monkeypatch):
    def launch(**kwargs):
        return (False, "fixture setup failure") if kwargs["job_name"].endswith("looseparts") else (True, "")

    monkeypatch.setattr(runner, "_run_harbor", launch)
    failures = {}
    assert runner._run_agent_pair(**arguments(tmp_path), additional_arm_errors=failures) == []
    assert failures == {"looseparts": "fixture setup failure"}


def test_stop_on_pass_applies_the_same_policy_to_each_private_arm(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "_run_stop_on_pass_variant", lambda **kw: calls.append(kw) or [])
    assert runner._run_agent_pair(**arguments(tmp_path), stop_on_pass=True, task_names=["case-a", "case-b"]) == []
    assert [call["variant"] for call in calls] == ["with", "without", "sumofparts", "looseparts"]
    assert all(call["n_attempts"] == 3 and call["task_names"] == ["case-a", "case-b"] for call in calls)


def test_reserved_or_duplicate_private_variants_are_refused(tmp_path):
    kwargs = arguments(tmp_path)
    kwargs["additional_arms"] = (("with", Path("fixture"), None),)
    with pytest.raises(ValueError, match="variant"):
        runner._run_agent_pair(**kwargs)


def test_private_comparison_options_refuse_an_incomplete_ladder():
    from skillevaluator.tier3.plugin_comparisons import ComparisonOptions

    with pytest.raises(ValueError, match="member"):
        ComparisonOptions(member_skills=False, loose_parts=True)
    with pytest.raises(ValueError, match="boolean"):
        ComparisonOptions(member_skills="yes")


@pytest.mark.parametrize("loose_parts", [False, True])
def test_baseline_only_keeps_one_arm_and_explains_unmeasured_private_controls(tmp_path, monkeypatch, loose_parts):
    from skillevaluator.tier3 import plugin_comparisons as comparisons

    package = SimpleNamespace(native_source=SimpleNamespace(member_skills=(tmp_path / "member",)))
    runtime = comparisons.ComparisonRuntime(
        SimpleNamespace(),
        package,
        comparisons.ComparisonOptions(loose_parts=loose_parts),
        ["claude-code"],
        baseline=True,
    )
    if loose_parts:
        runtime.loose_plan.arms["claude-code"] = object()
    baseline = tmp_path / "baseline" / "case"
    baseline.mkdir(parents=True)
    monkeypatch.setattr(comparisons, "verify_source_identity", lambda _package: None)

    def unexpected_control(*_args):
        pytest.fail("baseline-only execution emitted an additional comparison arm")

    jobs = runtime.stage_additional("claude-code", unexpected_control, {"without_skill": [baseline]}, None)

    assert jobs == ()
    runtime.bind_execution(["case"], n_attempts=1, stop_on_pass=False, pass_threshold=0.5)
    assert runtime.task_sets["claude-code"] == {"without_skill": (baseline,)}
    assert runtime.loose_plan.arm_for("claude-code") is None
    assert "with-skill arm disabled" in runtime.loose_plan.not_measured["claude-code"]


def test_usage_of_unscored_attempts_is_kept_without_inventing_scores():
    from skillevaluator.tier3.plugin_comparisons import arm_observations

    records = [
        SimpleNamespace(
            case_id="case-a",
            reward=None,
            usage={"total_prompt_tokens": 17, "total_completion_tokens": 3, "total_cost_usd": 0},
            first_turn_usage={"prompt_tokens": 17},
            evidence_status="recorded",
        )
    ]
    observed = arm_observations(
        records,
        expected_case_ids=["case-a"],
        n_attempts=1,
        stop_on_pass=False,
        pass_threshold=0.5,
        job_failure="setup failed",
    )
    assert observed.execution_status == "failed"
    assert not observed.trials
    assert observed.unscored[0].case_id == "case-a"
    assert observed.unscored[0].usage == {
        "prompt_tokens": 17,
        "completion_tokens": 3,
        "cost_usd": 0,
        "first_turn_prompt_tokens": 17,
    }


def test_first_turn_hosted_search_is_excluded_even_with_a_token_counter():
    from skillevaluator.tier3.plugin_comparisons import observation_usage

    record = SimpleNamespace(
        usage={"prompt_tokens": 10}, first_turn_usage={"prompt_tokens": 10, "first_turn_hosted_search": 1}
    )
    usage = observation_usage(record)
    assert usage["first_turn_hosted_search"] == 1
    assert "first_turn_prompt_tokens" not in usage


def test_stop_on_pass_extra_exception_preserves_the_completed_primary_arms(tmp_path, monkeypatch):
    calls = []

    def launch(**kwargs):
        calls.append(kwargs["variant"])
        if kwargs["variant"] == "sumofparts":
            raise OSError("fixture control failure")
        return []

    monkeypatch.setattr(runner, "_run_stop_on_pass_variant", launch)
    failures = {}
    assert runner._run_agent_pair(**arguments(tmp_path), stop_on_pass=True, additional_arm_errors=failures) == []
    assert calls == ["with", "without", "sumofparts", "looseparts"]
    assert "fixture control failure" in failures["sumofparts"]


@pytest.mark.parametrize("missing_case", [False, True])
def test_missing_recorded_attempts_cannot_become_complete_cost(missing_case):
    from skillevaluator.tier3.plugin_comparisons import arm_observations

    record = SimpleNamespace(
        case_id="case-a", reward=None, usage={"prompt_tokens": 10}, first_turn_usage={}, evidence_status="recorded"
    )
    observed = arm_observations(
        [record],
        expected_case_ids=["case-a", "case-b"] if missing_case else ["case-a"],
        n_attempts=1 if missing_case else 2,
        stop_on_pass=False,
        pass_threshold=0.5,
    )
    assert observed.observations_complete is False


def test_early_pass_is_complete_only_with_contiguous_attempt_evidence():
    from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS
    from skillevaluator.tier3.plugin_comparisons import arm_observations

    def observed(ordinal):
        record = SimpleNamespace(
            case_id="case-a",
            usage={},
            first_turn_usage={},
            evidence_status="recorded",
            reward={
                "entry_id": "case-a",
                "_attempt_ordinal": ordinal,
                "metric_set": DEFAULT_METRIC_SET,
                **dict.fromkeys(DEFAULT_METRICS, 1.0),
            },
        )
        return arm_observations(
            [record], expected_case_ids=["case-a"], n_attempts=3, stop_on_pass=True, pass_threshold=0.5
        )

    assert observed(1).observations_complete is True
    assert observed(2).observations_complete is False
