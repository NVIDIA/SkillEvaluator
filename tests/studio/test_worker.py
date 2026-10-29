# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from skillevaluator.studio.worker import _RedactedStream, run_job

_DOC = """---
name: word-counter
description: Count words in a supplied plain text string. Use when the user requests a word count.
---

# Word counter

Split the user's text on whitespace and return the number of resulting words.

## Example

Input: red blue green
Output: 3
"""


def _job(tmp_path: Path, action: str, **extra: object) -> Path:
    job = tmp_path / "job"
    skill = job / "source"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(_DOC)
    (job / "request.json").write_text(
        json.dumps({"action": action, "skill_path": str(skill), "results_dir": str(job / "results"), **extra})
    )
    return job


def test_quality_failure_does_not_become_a_worker_execution_failure(tmp_path: Path) -> None:
    job = _job(tmp_path, "check")
    (job / "source" / "SKILL.md").write_text("Do the thing.")
    assert run_job(job) == 0
    result = json.loads((job / "result.json").read_text())
    assert result["execution_status"] == "succeeded"
    assert result["passed"] is False


def test_real_quality_check_has_native_output_and_completed_events(tmp_path: Path) -> None:
    job = _job(tmp_path, "check")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"NVIDIA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SKILL_EVAL_LLM_API_KEY"}
    }
    process = subprocess.run(
        [sys.executable, "-m", "skillevaluator.studio.worker", str(job)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads((job / "result.json").read_text())
    assert result["kind"] == "check"
    assert isinstance(result["passed"], bool)
    assert result["raw"][0]["validator"]
    assert result["raw"][0]["success_details"]
    assert 0 <= result["quality"]["overall_score"] <= 100
    events = [json.loads(line) for line in (job / "events.jsonl").read_text().splitlines()]
    assert events[-1]["state"] == "complete"
    assert '"stage": "check"' in process.stdout


def test_known_request_failure_always_writes_machine_readable_result(tmp_path: Path) -> None:
    job = _job(tmp_path, "evaluate", options={"env_mode": "unknown"})
    assert run_job(job) == 1
    result = json.loads((job / "result.json").read_text())
    assert result["execution_status"] == "failed"
    assert result["summary"]["lift"] is None
    assert result["suggestions"] == []


@pytest.mark.parametrize("environment", ["local", "docker"])
@pytest.mark.parametrize(
    ("provider", "key_name", "model", "namespace"),
    [
        ("nv_build", "NVIDIA_API_KEY", "nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia"),
        ("openai", "OPENAI_API_KEY", "gpt-5.4-mini", "openai"),
        ("anthropic", "ANTHROPIC_API_KEY", "claude-opus-5-5", "anthropic"),
    ],
)
def test_catalog_model_reaches_opencode_with_runtime_namespace(
    tmp_path, monkeypatch, environment, provider, key_name, model, namespace
):
    from skillevaluator.evaluation import EvaluationService
    from skillevaluator.provider_config import resolve_llm_provider
    from skillevaluator.studio.worker import EventReporter, _evaluate
    from skillevaluator.tier3.harbor.runner import _model_for_agent

    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", provider)
    monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", model)
    monkeypatch.setenv(key_name, "test-credential")
    for name in ("SKILL_EVAL_LLM_API_KEY", "SKILL_EVAL_LLM_BASE_URL", "SKILL_EVAL_LLM_CREDENTIAL_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    captured = []
    monkeypatch.setattr(EvaluationService, "evaluate", lambda _self, options, **_kw: captured.append(options) or {})
    monkeypatch.setattr(EvaluationService, "failure_reason", lambda *_args: None)
    job = _job(tmp_path, "evaluate")
    _evaluate(
        {"options": {"agent": "opencode", "env_mode": environment, "model": model}},
        job / "source",
        job,
        EventReporter(job),
    )
    options = captured[0]
    runtime_model = options.agent_model[0].split("=", 1)[1]
    selected, _ = _model_for_agent(
        "opencode", cli_model=runtime_model, config_agents={}, provider=resolve_llm_provider()
    )
    assert options.model == model
    assert selected.split("/", 1) == [namespace, model]


def test_bad_action_type_also_retains_a_failure_result(tmp_path: Path) -> None:
    job = _job(tmp_path, "check")
    request = json.loads((job / "request.json").read_text())
    request["action"] = []
    (job / "request.json").write_text(json.dumps(request))
    assert run_job(job) == 1
    assert json.loads((job / "result.json").read_text())["kind"] == "error"


def test_missing_provider_fails_without_launching_agent_or_claiming_zero_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("NVIDIA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SKILL_EVAL_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    job = _job(tmp_path, "evaluate", options={"env_mode": "local", "agent": "opencode"})
    assert run_job(job) == 1
    result = json.loads((job / "result.json").read_text())
    assert "NVIDIA_API_KEY" in result["error"]
    assert result["summary"]["scored_attempts"] is None


def test_native_optional_checks_keep_missing_scanners_and_providers_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("NVIDIA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SKILL_EVAL_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("SKILL_EVAL_EMBEDDING_PROVIDER", "nv_build")
    monkeypatch.setenv("PATH", str(tmp_path))
    for name in ("SKILLSPECTOR", "BANDIT", "SEMGREP"):
        monkeypatch.setenv(f"SKILLEVALUATOR_{name}_PATH", str(tmp_path / "missing-scanner"))
    job = _job(tmp_path, "check", options={"tier1": True, "tier1_llm": True, "dedup": True})
    skill = job / "source"
    (skill / "reference.md").write_text(
        "# Input handling\n"
        + "Split the supplied plain text on whitespace and return the number of words. " * 3
        + "\n# Output handling\n"
        + "Return only the integer count, preserving the user's original input and files. " * 3
    )
    before = {path.name: path.read_bytes() for path in skill.iterdir()}
    process = subprocess.run(
        [sys.executable, "-m", "skillevaluator.studio.worker", str(job)],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads((job / "result.json").read_text())
    report = result["optional_checks"]
    assert result["kind"] == "check" and result["execution_status"] == "succeeded"
    assert result["passed"] is False and report["status"] == "incomplete"
    assert {"skillspector-llm", "skillspector", "gitleaks", "embedding-provider"} <= {
        name for item in report["results"] for name in item["incomplete_scans"]
    }
    assert any(item["metadata"].get("quality_scores") for item in report["results"])
    targets = [
        item["metadata"]["publication_target"] for item in report["results"] if "publication_target" in item["metadata"]
    ]
    assert targets and all(target == targets[0] for target in targets)
    assert report == json.loads((job / "optional-checks.json").read_text())
    assert {path.name: path.read_bytes() for path in skill.iterdir()} == before
    assert "summary" not in result and not (skill / "evals").exists()


def test_optional_findings_do_not_gate_tier3_and_survive_its_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.models.result import ValidationResult
    from skillevaluator.studio import worker
    from skillevaluator.tier1 import commands

    for incomplete in (False, True):
        check = ValidationResult(validator_name="Controlled native check")
        check.add_error("A check found a problem.")
        if incomplete:
            check.mark_scan_incomplete("controlled-scanner")
        monkeypatch.setattr(commands, "run_validation", lambda *_args, _check=check, **_kwargs: [_check])
        job = _job(tmp_path / str(incomplete), "evaluate", options={"tier1": True})

        def evaluate(request, skill, job_dir, reporter, expected=job / "source", should_fail=incomplete):
            assert skill == expected
            assert json.loads((job_dir / "optional-checks.json").read_text())["results"][0]["errors"]
            if should_fail:
                raise RuntimeError("Controlled agent failure")
            return {"kind": "evaluation", "summary": {"lift": 0.125}}

        monkeypatch.setattr(worker, "_evaluate", evaluate)
        assert run_job(job) == int(incomplete)
        result = json.loads((job / "result.json").read_text())
        assert result["optional_checks"]["status"] == ("incomplete" if incomplete else "failed")
        assert result["summary"]["lift"] == (None if incomplete else 0.125)
        assert result["execution_status"] == ("failed" if incomplete else "succeeded")


def test_native_generator_exit_retains_diagnostic_in_failed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skillevaluator.tier3 import generate_dataset

    def stop(_arguments: list[str]) -> None:
        failure = SystemExit(2)
        failure.diagnostic = "The native generator could not prepare this dataset."
        raise failure

    monkeypatch.setattr(generate_dataset, "main", stop)
    job = _job(tmp_path, "draft", options={"no_llm": True})
    assert run_job(job) == 1
    result = json.loads((job / "result.json").read_text())
    assert result["kind"] == "dataset"
    assert result["execution_status"] == "failed"
    assert result["error"] == "DatasetGenerationError: The native generator could not prepare this dataset."


def test_stream_redacts_credentials_split_between_native_print_writes() -> None:
    output = io.StringIO()
    stream = _RedactedStream(output, {"unusual-opaque-credential"})
    stream.write("failure unusual-opaque-")
    stream.write("credential\n")
    stream.flush()
    assert "unusual-opaque" not in output.getvalue()
    assert "<redacted>" in output.getvalue()


def test_sigterm_preserves_interrupted_worker_result(tmp_path: Path) -> None:
    job = _job(tmp_path, "check")
    # A controlled long action verifies the real process/signal boundary. It
    # does not claim that a credentialed Harbor job's cleanup was exercised.
    code = """import sys,time
from pathlib import Path
from skillevaluator.studio import worker
job=Path(sys.argv[1])
def slow_action(skill):
    (job/'ready').write_text('ready')
    time.sleep(60)
worker._check=slow_action
raise SystemExit(worker.main([str(job)]))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(job)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 10
        while not (job / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (job / "ready").exists()
        process.send_signal(signal.SIGTERM)
        _stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, stderr
        result = json.loads((job / "result.json").read_text())
        assert result["execution_status"] == "interrupted"
        assert result["kind"] == "check"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
