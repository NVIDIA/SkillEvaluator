# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A run whose trials all time out without a model token is flagged as an endpoint failure."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor")

from tests.test_harbor_task_toml_retry_env import _stub_harbor_runner

from skillevaluator.tier3.harbor import endpoint_failure, runner
from skillevaluator.tier3.harbor.endpoint_failure import (
    DIAGNOSTIC_KIND,
    ENDPOINT_FAILURE_THRESHOLD,
    detect_model_endpoint_failure,
    endpoint_failure_warning,
    provider_hosts,
    selected_agent_job_dirs,
)


def _generated_skill(tmp_path: Path) -> Path:
    skill = tmp_path / "demo-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo-skill\ndescription: Demo skill.\n---\n# Demo\n", encoding="utf-8")
    (skill / "evals" / "evals.json").write_text(
        '[{"id": "case-001", "question": "Say ok.", "expected_answer": "ok", "files": []}]\n', encoding="utf-8"
    )
    return skill


def _trial_result(
    name: str,
    *,
    exception_type: str | None = "AgentTimeoutError",
    n_input: int | None = 0,
    n_output: int | None = 0,
    agent: str = "codex",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "trial_name": name,
        "agent_info": {"name": agent},
        "agent_result": {"n_input_tokens": n_input, "n_output_tokens": n_output},
    }
    if exception_type is not None:
        result["exception_info"] = {
            "exception_type": exception_type,
            "exception_message": "Agent execution timed out after 300.0 seconds",
        }
    return result


def _write_trial(job_dir: Path, name: str, payload: dict[str, Any] | str) -> Path:
    trial_dir = job_dir / name
    trial_dir.mkdir(parents=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (trial_dir / "result.json").write_text(text, encoding="utf-8")
    return trial_dir


def _write_job(job_dir: Path, results: list[dict[str, Any]]) -> Path:
    for result in results:
        _write_trial(job_dir, result["trial_name"], result)
    return job_dir


def _timeouts(count: int, *, prefix: str = "t", **kwargs: Any) -> list[dict[str, Any]]:
    return [_trial_result(f"{prefix}{index}", **kwargs) for index in range(count)]


def test_every_trial_timing_out_without_tokens_is_flagged(tmp_path: Path) -> None:
    job = _write_job(tmp_path / "job", _timeouts(3, agent="codex") + _timeouts(2, prefix="c", agent="claude-code"))

    diagnostic = detect_model_endpoint_failure([job])

    assert diagnostic is not None
    assert diagnostic["kind"] == DIAGNOSTIC_KIND == "model_endpoint_unreachable"
    assert (diagnostic["matched_trials"], diagnostic["total_trials"]) == (5, 5)
    assert diagnostic["agents"] == ["claude-code", "codex"]
    assert set(diagnostic) == {"kind", "matched_trials", "total_trials", "agents"}


def test_exactly_the_threshold_is_flagged_and_below_it_is_not(tmp_path: Path) -> None:
    assert ENDPOINT_FAILURE_THRESHOLD == 0.8
    four_of_five = _write_job(tmp_path / "four", [*_timeouts(4), _trial_result("ok", exception_type=None, n_input=9)])
    three_of_five = _write_job(
        tmp_path / "three",
        [*_timeouts(3), *(_trial_result(f"ok{i}", exception_type=None, n_input=9) for i in range(2))],
    )

    diagnostic = detect_model_endpoint_failure([four_of_five])

    assert diagnostic is not None
    assert (diagnostic["matched_trials"], diagnostic["total_trials"]) == (4, 5)
    assert detect_model_endpoint_failure([three_of_five]) is None


def test_the_ratio_is_taken_over_all_jobs_of_the_run(tmp_path: Path) -> None:
    failing = _write_job(tmp_path / "with", _timeouts(4))
    working = _write_job(
        tmp_path / "without", [_trial_result(f"w{i}", exception_type=None, n_input=5) for i in range(4)]
    )

    assert detect_model_endpoint_failure([failing, working]) is None
    assert detect_model_endpoint_failure([failing]) is not None


@pytest.mark.parametrize(
    "result",
    [
        _trial_result("t0", n_input=12, n_output=0),
        _trial_result("t0", n_input=0, n_output=3),
        _trial_result("t0", exception_type="RuntimeError"),
        _trial_result("t0", exception_type="NonZeroAgentExitCodeError"),
        _trial_result("t0", exception_type=None),
    ],
    ids=["input tokens", "output tokens", "other exception", "other agent exception", "no exception"],
)
def test_a_trial_that_used_the_model_or_failed_otherwise_is_not_a_match(tmp_path: Path, result: dict[str, Any]) -> None:
    job = _write_job(tmp_path / "job", [result])

    assert detect_model_endpoint_failure([job]) is None


def test_null_and_missing_token_counts_count_as_zero(tmp_path: Path) -> None:
    missing = _trial_result("t1")
    del missing["agent_result"]
    job = _write_job(tmp_path / "job", [_trial_result("t0", n_input=None, n_output=None), missing])

    diagnostic = detect_model_endpoint_failure([job])

    assert diagnostic is not None
    assert (diagnostic["matched_trials"], diagnostic["total_trials"]) == (2, 2)


def test_a_timeout_whose_trajectory_shows_model_output_is_not_a_match(tmp_path: Path) -> None:
    job = tmp_path / "job"
    trial = _write_trial(job, "t0", _trial_result("t0", n_input=None, n_output=None))
    (trial / "agent").mkdir()
    (trial / "agent" / "trajectory.json").write_text(
        json.dumps({"final_metrics": {"total_completion_tokens": 900}}), encoding="utf-8"
    )

    assert detect_model_endpoint_failure([job]) is None


@pytest.mark.skipif(os.name != "posix", reason="needs POSIX symlinks")
def test_links_nested_copies_and_malformed_results_are_ignored(tmp_path: Path) -> None:
    used_model = _trial_result("x", exception_type=None, n_input=4)
    job = tmp_path / "job"
    job.mkdir()
    # The one trial that counts, and it matches.
    _write_trial(job, "real", _trial_result("real"))
    # Everything below would pull the ratio under the threshold if it were counted.
    # A symlinked trial directory.
    elsewhere = _write_trial(tmp_path / "elsewhere", "linked", {**used_model, "trial_name": "linked"})
    (job / "linked").symlink_to(elsewhere, target_is_directory=True)
    # A trial whose result.json is a symlink.
    symlinked_result = job / "symlinked-result"
    symlinked_result.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({**used_model, "trial_name": "symlinked-result"}), encoding="utf-8")
    (symlinked_result / "result.json").symlink_to(outside)
    # Copies kept inside a counted trial's artifacts.
    _write_trial(job / "real" / "artifacts", "nested", {**used_model, "trial_name": "nested"})
    # Malformed, unidentified and mismatched results.
    _write_trial(job, "invalid", "{not json")
    _write_trial(job, "no-agent", {"trial_name": "no-agent", "agent_result": {"n_input_tokens": 3}})
    _write_trial(job, "renamed", {**used_model, "trial_name": "some-other-name"})
    _write_trial(job, "a-list", "[]")
    (job / "stray-file").write_text("{}", encoding="utf-8")

    diagnostic = detect_model_endpoint_failure([job])

    assert diagnostic is not None
    assert (diagnostic["matched_trials"], diagnostic["total_trials"]) == (1, 1)


@pytest.mark.skipif(os.name != "posix", reason="needs POSIX symlinks")
def test_a_symlinked_job_directory_is_ignored(tmp_path: Path) -> None:
    real_job = _write_job(tmp_path / "real-job", _timeouts(3))
    link = tmp_path / "linked-job"
    link.symlink_to(real_job, target_is_directory=True)

    assert detect_model_endpoint_failure([link]) is None
    assert detect_model_endpoint_failure([real_job]) is not None


def test_no_trials_or_missing_directories_give_no_diagnostic(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    assert detect_model_endpoint_failure([]) is None
    assert detect_model_endpoint_failure([empty, tmp_path / "missing"]) is None


def test_selected_job_directories_follow_the_arms_that_ran(tmp_path: Path) -> None:
    both = selected_agent_job_dirs(tmp_path, "demo", ["codex"], skip_baseline=False, skip_with_skill=False)
    with_only = selected_agent_job_dirs(
        tmp_path, "demo", ["codex", "opencode"], skip_baseline=True, skip_with_skill=False
    )
    without_only = selected_agent_job_dirs(tmp_path, "demo", ["codex"], skip_baseline=False, skip_with_skill=True)

    assert both == [tmp_path / "demo-codex-with", tmp_path / "demo-codex-without"]
    assert with_only == [tmp_path / "demo-codex-with", tmp_path / "demo-opencode-with"]
    assert without_only == [tmp_path / "demo-codex-without"]


def test_warning_names_only_hosts_and_the_environment() -> None:
    diagnostic = {"matched_trials": 4, "total_trials": 5}
    hosts = provider_hosts(
        [
            "https://user:hunter2@integrate.api.nvidia.com/v1/chat?key=abc",
            "https://api.openai.com:8443/v1",
            "https://integrate.api.nvidia.com/other",
            None,
            "",
            "not a url",
        ]
    )

    warning = endpoint_failure_warning(diagnostic, env_mode="docker", hosts=hosts)

    assert hosts == ["integrate.api.nvidia.com", "api.openai.com"]
    assert warning.startswith("Likely model endpoint failure: 4 of 5 trials ended in AgentTimeoutError")
    assert "without a single model token (0 input, 0 output)" in warning
    assert "model endpoint (integrate.api.nvidia.com, api.openai.com). The timeouts most likely reflect" in warning
    assert "from the docker environment" in warning
    for leaked in ("hunter2", "user:", "/v1", "key=abc", "8443"):
        assert leaked not in warning


def test_warning_without_hosts_omits_the_parenthesis() -> None:
    warning = endpoint_failure_warning({"matched_trials": 1, "total_trials": 1}, env_mode="local", hosts=[])

    assert "from the model endpoint. The timeouts" in warning
    assert "()" not in warning


def test_provider_hosts_survives_a_malformed_authority() -> None:
    assert provider_hosts(["http://[::1/v1", "https://ok.example/v1"]) == ["ok.example"]


def _run_with_collected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    diagnostic: dict[str, Any] | None,
    collected_warnings: list[str] | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    skill = _generated_skill(tmp_path)
    _stub_harbor_runner(monkeypatch, tmp_path, task_source="evals_json")
    seen: list[Any] = []

    def collect(**_kwargs: Any) -> dict[str, Any]:
        collected: dict[str, Any] = {
            "execution_status": "complete",
            "execution_errors": [],
            "metrics": [],
            "agents": {},
        }
        if collected_warnings is not None:
            collected["warnings"] = list(collected_warnings)
        return collected

    def detect(job_dirs: Any) -> dict[str, Any] | None:
        seen.extend(job_dirs)
        return diagnostic

    monkeypatch.setattr(runner, "collect_harbor_results", collect)
    monkeypatch.setattr(runner, "detect_model_endpoint_failure", detect)
    result = runner.run_harbor_eval(
        skill,
        ["codex"],
        agent_models={"codex": "gpt-4o-mini"},
        output_dir=tmp_path / "results",
        env_mode="docker",
        skip_baseline=True,
        keep_harbor_jobs=True,
        agent_runtime_preflight=False,
    )
    return result, seen


def test_run_result_carries_the_diagnostic_and_a_host_only_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    diagnostic = {
        "kind": DIAGNOSTIC_KIND,
        "matched_trials": 1,
        "total_trials": 1,
        "agents": ["codex"],
    }

    result, seen = _run_with_collected(monkeypatch, tmp_path, diagnostic=diagnostic)

    assert result["model_endpoint_diagnostic"] == diagnostic
    (warning,) = [item for item in result["warnings"] if item.startswith("Likely model endpoint failure")]
    assert "(api.openai.com)" in warning
    assert "from the docker environment" in warning
    assert "/v1" not in warning
    assert result["execution_status"] == "complete"
    assert result["execution_errors"] == []
    # Only the arm that ran is read: the with-skill job of the one agent.
    assert [path.name for path in seen] == ["demo-skill-codex-with"]
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert persisted["model_endpoint_diagnostic"] == diagnostic
    assert warning in persisted["warnings"]


def test_run_result_keeps_the_warnings_the_collector_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    diagnostic = {"kind": DIAGNOSTIC_KIND, "matched_trials": 2, "total_trials": 2, "agents": []}

    result, _seen = _run_with_collected(
        monkeypatch, tmp_path, diagnostic=diagnostic, collected_warnings=["earlier warning"]
    )

    earlier, added = result["warnings"]
    assert earlier == "earlier warning"
    assert added.startswith("Likely model endpoint failure")


def test_run_without_a_matching_pattern_adds_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, _seen = _run_with_collected(monkeypatch, tmp_path, diagnostic=None)

    assert "model_endpoint_diagnostic" not in result
    assert not [item for item in result.get("warnings", []) if "model endpoint" in item]


def test_detector_reads_through_the_bounded_collector_reader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from skillevaluator.tier3.harbor import collector

    job = _write_job(tmp_path / "job", _timeouts(2))
    reads: list[Path] = []
    real_read = collector._read_json
    monkeypatch.setattr(endpoint_failure, "_read_json", lambda path: reads.append(path) or real_read(path))

    assert detect_model_endpoint_failure([job]) is not None
    assert sorted(path.parent.name for path in reads) == ["t0", "t1"]
