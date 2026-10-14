# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset preparation must remain visibly active without polluting stdout."""

from __future__ import annotations

import json
import threading
from io import StringIO

import click
import httpx
import pytest
from click.testing import CliRunner
from openai import APIStatusError, APITimeoutError

from skillevaluator.cli import _ensure_autopilot_dataset
from skillevaluator.inference.client import LLMClient


@pytest.fixture
def generation_skill(tmp_path, monkeypatch):
    skill = tmp_path / "demo"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Summarize supplied text.\n---\n# Demo\n")
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-never-sent")
    monkeypatch.delenv("SKILL_EVAL_LLM_MODEL", raising=False)
    case = {
        "id": "demo-001",
        "question": "Summarize the supplied text.",
        "expected_skill": "demo",
        "expected_script": None,
        "ground_truth": "A concise summary.",
        "expected_behavior": ["The agent provides a summary."],
    }
    monkeypatch.setattr(LLMClient, "completions", lambda *_args, **_kwargs: json.dumps([case]))
    return skill


def test_generation_announces_provider_and_handoff_on_stderr(generation_skill):
    @click.command()
    def command():
        _ensure_autopilot_dataset(generation_skill)

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "nv_build / nvidia/" in result.stderr
    assert "evaluation source ready" in result.stderr
    assert (generation_skill / "evals" / "evals.json").exists()


def test_quiet_generation_has_no_provider_output(generation_skill):
    @click.command()
    def command():
        _ensure_autopilot_dataset(generation_skill, quiet=True)

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert result.output == ""
    assert (generation_skill / "evals" / "evals.json").exists()


class _ObservedStream(StringIO):
    def __init__(self):
        super().__init__()
        self.heartbeat_received = threading.Event()

    def write(self, text):
        count = super().write(text)
        if "live evaluation has not started" in text:
            self.heartbeat_received.set()
        return count


def test_heartbeat_reports_wait_before_completion_and_stops(monkeypatch):
    from skillevaluator.tier3 import dataset_progress

    monkeypatch.setattr(dataset_progress, "HEARTBEAT_INTERVAL_SECONDS", 0.01)
    stream = _ObservedStream()
    with dataset_progress.dataset_generation_progress(enabled=True, stream=stream):
        assert stream.heartbeat_received.wait(2), "generation wait must be visible before it completes"
        assert "elapsed" in stream.getvalue()
        assert "evaluation source ready" not in stream.getvalue()
    assert "evaluation source ready" in stream.getvalue()
    assert not any(thread.name == "dataset-generation-progress" for thread in threading.enumerate())


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_failed_or_interrupted_generation_stops_without_success(monkeypatch, error):
    from skillevaluator.tier3 import dataset_progress

    monkeypatch.setattr(dataset_progress, "HEARTBEAT_INTERVAL_SECONDS", 0.01)
    stream = _ObservedStream()
    with pytest.raises(error), dataset_progress.dataset_generation_progress(enabled=True, stream=stream):
        assert stream.heartbeat_received.wait(2)
        raise error()
    assert "evaluation source ready" not in stream.getvalue()
    assert not any(thread.name == "dataset-generation-progress" for thread in threading.enumerate())


def test_closed_progress_stream_does_not_fail_generation():
    from skillevaluator.tier3.dataset_progress import dataset_generation_progress

    stream = StringIO()
    stream.close()
    with dataset_generation_progress(enabled=True, stream=stream):
        pass


def test_off_suppresses_progress_but_preserves_setup_notice(generation_skill):
    @click.command()
    def command():
        _ensure_autopilot_dataset(generation_skill, progress="off")

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "Autopilot: generating one case" in result.stderr
    assert "elapsed" not in result.stderr
    assert "evaluation source ready" not in result.stderr


def test_interrupted_autopilot_does_not_fallback_or_write_dataset(generation_skill, monkeypatch):
    from skillevaluator.evaluation import EvaluationService

    calls = []

    def interrupt(_self, _path, *, use_llm):
        calls.append(use_llm)
        raise KeyboardInterrupt

    monkeypatch.setattr(EvaluationService, "create_autopilot_dataset", interrupt)

    @click.command()
    def command():
        _ensure_autopilot_dataset(generation_skill)

    result = CliRunner().invoke(command)
    assert result.exit_code == 1
    assert calls == [True]
    assert "evaluation source ready" not in result.output
    assert not (generation_skill / "evals").exists()


@pytest.mark.parametrize(
    "error, expected",
    [
        (
            APIStatusError(
                "private-response-content",
                response=httpx.Response(429, request=httpx.Request("POST", "https://private-endpoint.invalid")),
                body={"error": {"code": "rate_limit_exceeded", "message": "private-response-content"}},
            ),
            "HTTP 429",
        ),
        (APITimeoutError(request=httpx.Request("POST", "https://private-endpoint.invalid")), "timed out"),
        (json.JSONDecodeError("private-response-content", "private-response-content", 0), "structured-output"),
        (RuntimeError("private-response-content"), "Unexpected LLM"),
    ],
)
def test_fallback_explains_safe_failure_and_labels_starter(generation_skill, monkeypatch, error, expected):
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(LLMClient, "completions", fail)
    notes = []

    @click.command()
    def command():
        notes.append(_ensure_autopilot_dataset(generation_skill, progress="off"))

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert expected in result.stderr
    assert "falling back" in result.stderr.lower()
    assert "deterministic starter" in result.stderr.lower()
    assert "deterministic starter" in notes[0].lower()
    assert expected in notes[0]
    assert "private-response-content" not in result.output + notes[0]
    assert "private-endpoint" not in result.output + notes[0]
    assert "Tier 2" not in result.output + notes[0]
    assert len(json.loads((generation_skill / "evals" / "evals.json").read_text())["evals"]) == 1


def test_quiet_fallback_preserves_reason_in_returned_note(generation_skill, monkeypatch):
    def fail(*_args, **_kwargs):
        raise TimeoutError("private-response-content")

    monkeypatch.setattr(LLMClient, "completions", fail)
    notes = []

    @click.command()
    def command():
        notes.append(_ensure_autopilot_dataset(generation_skill, quiet=True))

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert result.output == ""
    assert "timed out" in notes[0]
    assert "deterministic starter" in notes[0].lower()
    assert "private-response-content" not in notes[0]
