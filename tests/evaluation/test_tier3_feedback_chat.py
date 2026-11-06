# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the interactive ``tier3 feedback`` chat over one completed run."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from skillevaluator.cli import cli
from skillevaluator.evaluation import tier3_feedback_chat
from skillevaluator.evaluation.tier3_feedback_chat import (
    Tier3FeedbackSession,
    _Scrubber,
    _summarize_cases_for_feedback,
    compact_run_context,
    load_agent_eval_payload,
    open_feedback_session,
    resolve_skill_path_for_run,
)
from skillevaluator.inference.types import LLMClientError, LLMClientRefusedError, LLMClientTruncatedError
from skillevaluator.provider_config import ProviderConfigurationError
from skillevaluator.tier3 import commands
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

RUN_ID = "20261027_120000"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _completed_run(parent: Path, run_id: str = RUN_ID, **result: Any) -> Path:
    """A completed run directory: ``run_config.json`` plus a ``result.json`` owned by the run."""
    run = parent / run_id
    run.mkdir(parents=True)
    _write_json(run / "run_config.json", {})
    _write_json(run / "result.json", {"run_id": run_id, **result})
    return run


def _write_arm(run: Path, arm: str, *, overall: float, tokens: int, seconds: int, warning: str | None = None) -> None:
    """One Harbor arm for agent ``opencode`` with one trial of ``case-1``."""
    directory = run / "opencode" / arm
    scores = dict.fromkeys(DEFAULT_METRICS, overall)
    _write_json(
        directory / "summary.json",
        {
            "scores": scores,
            "metrics": list(DEFAULT_METRICS),
            "num_trials": 1,
            "expected_attempts": 1,
            "scored_attempts": 1,
            "execution_status": "succeeded",
        },
    )
    trial = directory / "trials" / "case-1__1"
    reward: dict[str, Any] = {"entry_id": "case-1", **scores}
    if warning:
        reward["warnings"] = [warning]
    _write_json(trial / "reward.json", reward)
    _write_json(
        trial / "trajectory.json",
        {"steps": [{}], "final_metrics": {"total_prompt_tokens": tokens, "total_completion_tokens": 10}},
    )
    _write_json(
        trial / "result.json",
        {"agent_execution": {"started_at": "2026-10-27T12:00:00Z", "finished_at": f"2026-10-27T12:00:0{seconds}Z"}},
    )


def _real_run(parent: Path, *, skill_digest: str = "sha256:" + "0" * 64) -> Path:
    """A completed run with real Harbor artifacts and saved report feedback."""
    run = _completed_run(
        parent,
        skill_name="demo-skill",
        publication_target={
            "skill_name": "demo-skill",
            "skill_digest": skill_digest,
            "skill_digest_algorithm": "sha256",
        },
        dataset=[
            {"id": "case-1", "question": "Create the report.", "ground_truth": "A report", "should_trigger": True}
        ],
        tier3_feedback={
            "schema_version": "1.0",
            "conclusions": [{"severity": "warn", "title": "Saved conclusion", "message": "From this run"}],
            "recommendations": [],
            "suggestions": ["Saved suggestion"],
            "suggestions_v2": [],
        },
    )
    _write_arm(run, "with-skill", overall=0.9, tokens=20, seconds=3, warning=f"retry under {run}")
    _write_arm(run, "without-skill", overall=0.5, tokens=40, seconds=5)
    return run


def _chat_response(text: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=text, refusal=None))]
    )


def _use_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SKILL_EVAL_LLM_MODEL", "gpt-5.4-mini")
    for name in ("SKILL_EVAL_LLM_BASE_URL", "OPENAI_BASE_URL", "SKILL_EVAL_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def plain_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Print without color or wrapping, whatever the CI terminal settings are."""
    monkeypatch.setattr(commands, "console", Console(emoji=False, color_system=None, width=400))


def _cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return _summarize_cases_for_feedback(payload, _Scrubber.for_run(Path("/runs/demo") / RUN_ID))


def _invoke(args: list[str], user_input: str | None = None):
    return CliRunner().invoke(cli, ["tier3", "feedback", *args], input=user_input)


def _invoke_with_session(
    monkeypatch: pytest.MonkeyPatch, skill: Path, run: Path, session: Tier3FeedbackSession, user_input: str
):
    skill.mkdir(parents=True)
    run.mkdir(parents=True)
    monkeypatch.setattr(tier3_feedback_chat, "assert_completed_run_dir", lambda run_dir: run_dir.expanduser().resolve())
    monkeypatch.setattr(tier3_feedback_chat, "open_feedback_session", MagicMock(return_value=session))
    return _invoke([str(run), "--skill", str(skill)], user_input)


# -- skill resolution ---------------------------------------------------------


def test_resolve_skill_path_from_publication_target(tmp_path: Path) -> None:
    run = _completed_run(tmp_path, skill_name="staged-copy", publication_target={"skill_name": "demo-skill"})
    assert resolve_skill_path_for_run(run, None) == Path("demo-skill")


def test_resolve_skill_path_ignores_a_target_from_another_run(tmp_path: Path) -> None:
    run = tmp_path / RUN_ID
    run.mkdir()
    _write_json(
        run / "result.json",
        {"run_id": "20250101_000000", "skill_name": "demo-skill", "publication_target": {"skill_name": "other"}},
    )
    assert resolve_skill_path_for_run(run, None) == Path("demo-skill")


def test_resolve_skill_path_requires_hint_when_missing(tmp_path: Path) -> None:
    run = tmp_path / RUN_ID
    run.mkdir()
    with pytest.raises(FileNotFoundError, match="--skill"):
        resolve_skill_path_for_run(run, None)


def test_resolve_skill_path_checks_explicit_skill_against_run(tmp_path: Path) -> None:
    run = _completed_run(tmp_path / "runs", skill_name="demo-skill")
    skill = tmp_path / "demo-skill"
    skill.mkdir()
    assert resolve_skill_path_for_run(run, skill) == skill.resolve()
    other = tmp_path / "other-skill"
    other.mkdir()
    with pytest.raises(ValueError, match=r"does not match the skill this run evaluated \('demo-skill'\)"):
        resolve_skill_path_for_run(run, other)
    # A run that recorded no skill name keeps accepting the explicit path.
    bare_run = tmp_path / "20261027_130000"
    bare_run.mkdir()
    assert resolve_skill_path_for_run(bare_run, other) == other.resolve()


def test_resolve_skill_path_ignores_case_of_skill_directory(tmp_path: Path) -> None:
    run = _completed_run(tmp_path / "runs", publication_target={"skill_name": "Demo-Skill"})
    skill = tmp_path / "demo-skill"
    skill.mkdir()
    assert resolve_skill_path_for_run(run, skill) == skill.resolve()


def test_resolve_skill_path_accepts_a_decomposed_spelling_of_the_recorded_name(tmp_path: Path) -> None:
    # Runs record the NFC name; macOS paths often arrive decomposed (NFD).
    run = _completed_run(tmp_path / "runs", publication_target={"skill_name": "café-skill"})
    skill = tmp_path / unicodedata.normalize("NFD", "café-skill")
    skill.mkdir()
    assert resolve_skill_path_for_run(run, skill) == skill.resolve()


# -- run context ----------------------------------------------------------------


def test_compact_run_context_keeps_agent_resources() -> None:
    resources = {
        "scope": "observed agent-only trial totals",
        "pairing": "matching case counts",
        "delta_definition": "with skill minus baseline",
        "tokens": {"with_skill": {"total": 12}, "baseline": {"total": 22}, "delta": -10},
        "agent_seconds": {"with_skill": {"total": 3}, "baseline": {"total": 5}, "delta": -2},
    }
    context = compact_run_context({"agents": {"codex": {"resources": resources}}}, run_dir=Path("/runs/demo"))
    compact = context["agents"]["codex"]["resources"]
    assert compact["tokens"] == resources["tokens"]
    assert compact["agent_seconds"]["delta"] == -2
    assert "scope" not in compact and "delta_definition" not in compact


@pytest.mark.parametrize("runtime", [0.0, True, -1, float("nan"), float("inf"), "42"])
def test_invalid_runtime_is_unavailable(runtime: object) -> None:
    context = compact_run_context({"runtime_seconds": runtime}, run_dir=Path("/runs/demo"))
    assert context["runtime_seconds"] is None


def test_compact_run_context_includes_comparisons() -> None:
    payload = {
        "skill_name": "demo",
        "overall_score": 0.8,
        "overall_lift": 0.1,
        "runtime_seconds": 42.0,
        "dimensions": [],
        "evaluators": {"accuracy": {"with_skill": 0.9, "baseline": 0.8, "lift": 0.1}},
        "trials": [],
        "dataset": [],
        "conclusions": [{"title": "t", "message": "m", "evidence": {"path": "/tmp/x"}}],
        "recommendations": [{"title": "case-1 · skill", "message": "Fix step 1", "severity": "warn"}],
        "suggestions": ["Fix step 1"],
    }
    context = compact_run_context(payload, run_dir=Path("/runs/demo"))
    assert context["overall_lift"] == 0.1
    assert context["runtime_seconds"] == 42.0
    assert context["best_agent_evaluators"]["accuracy"]["lift"] == 0.1
    assert context["report_feedback"] == {
        "conclusions": [{"title": "t", "message": "m"}],
        "recommendations": [{"severity": "warn", "title": "case-1 · skill", "message": "Fix step 1"}],
        "suggestions": ["Fix step 1"],
    }


def test_context_preserves_other_agents_and_failure_scope() -> None:
    payload = {
        "best_agent": "codex",
        "agents": {
            "codex": {"with_skill": 0.8, "baseline": 0.7, "lift": 0.1},
            "claude": {
                "execution_status": "failed",
                "execution_errors": ["sandbox setup failed", *[f"error {index}" for index in range(10)]],
                "skill_activation": {"status": "never_activated", "reason": "No activation", "trials": [1, 2]},
                "conditions": {
                    "with_skill": {
                        "execution_status": "failed",
                        "expected_attempts": 2,
                        "scored_attempts": 0,
                        "duration_ms": 1234,
                        "recovered_attempts": ["not needed"],
                    }
                },
                "trials": [{"raw_trajectory": "must not copy"}],
            },
        },
    }
    context = compact_run_context(payload, run_dir=Path("/runs/demo"))
    assert context["agents"]["codex"]["lift"] == 0.1
    other = context["agents"]["claude"]
    assert other["execution_errors"][0] == "sandbox setup failed"
    assert len(other["execution_errors"]) == 5
    assert other["skill_activation"] == {"status": "never_activated", "reason": "No activation"}
    assert other["conditions"]["with_skill"] == {
        "execution_status": "failed",
        "expected_attempts": 2,
        "scored_attempts": 0,
        "duration_ms": 1234,
    }
    assert "trials" not in other
    assert "must not copy" not in json.dumps(context)
    assert compact_run_context({}, run_dir=Path("/runs/legacy"))["agents"] == {}


def test_context_sends_run_id_not_host_path(tmp_path: Path) -> None:
    run = tmp_path / "results" / "demo" / RUN_ID
    payload = {"agents": {"codex": {"execution_errors": [f"missing {run / 'opencode' / 'summary.json'}"]}}}
    context = compact_run_context(payload, run_dir=run)
    assert context["run_id"] == RUN_ID
    evidence = Tier3FeedbackSession(run_context=context, assistant=MagicMock()).transcript[0]["content"]
    assert RUN_ID in evidence
    assert "summary.json" in evidence
    assert str(tmp_path) not in evidence
    assert tmp_path.as_posix() not in evidence


def test_context_masks_the_path_a_copied_run_had_elsewhere(tmp_path: Path) -> None:
    # A copied run keeps its ID, but text written where it first ran still
    # names that machine's path.
    posix = f"/home/ci-runner/work/acme-skill/evals/results/{RUN_ID}"
    windows = f"C:\\Users\\builder\\acme-skill\\results\\{RUN_ID}"
    escaped = windows.replace("\\", "\\\\")
    kept = f"kept results/{RUN_ID}/jobs, /tmp/{RUN_ID}_old and /tmp/{RUN_ID}.log"
    payload = {
        "dataset": [{"id": "case-1", "question": f"Read {posix}/inputs/a.txt", "ground_truth": f"path={posix}"}],
        "agents": {
            "codex": {
                "execution_errors": [
                    f"Trial failed: {posix}/jobs/t1 exited 1",
                    f'{{"cwd":"{escaped}\\\\jobs"}}',
                    f"see ~/acme-skill/results/{RUN_ID}, then ({windows}\\jobs).",
                    kept,
                ],
            },
            "opencode": {
                "execution_errors": [
                    f"see ~builder/acme-skill/results/{RUN_ID}/jobs, $HOME/acme-skill/{RUN_ID}, "
                    f"${{HOME}}/acme-skill/{RUN_ID} and %USERPROFILE%\\acme-skill\\{RUN_ID}",
                    # A path joined to another path or option keeps the other one.
                    f"tried /opt/tool/bin,{posix}/jobs; PATH=/usr/bin:{posix} log:/var/log/x({posix}) "
                    f"PATH=C:\\bin;{windows}",
                    # A separator that does not start another absolute path is part of this one.
                    f"in /home/ci-runner/my(project),v2/{RUN_ID}",
                ],
            },
        },
    }

    context = compact_run_context(payload, run_dir=tmp_path / "Downloads" / RUN_ID)

    serialized = json.dumps(context)
    for leaked in ("ci-runner", "acme-skill", "builder"):
        assert leaked not in serialized
    errors = context["agents"]["codex"]["execution_errors"]
    assert errors[0] == f"Trial failed: {RUN_ID}/jobs/t1 exited 1"
    assert errors[1] == f'{{"cwd":"{RUN_ID}\\\\jobs"}}'
    assert errors[2] == f"see {RUN_ID}, then ({RUN_ID}\\jobs)."
    # A relative path and other names that only start with the run ID are left alone.
    assert errors[3] == kept
    assert context["agents"]["opencode"]["execution_errors"] == [
        f"see {RUN_ID}/jobs, {RUN_ID}, {RUN_ID} and {RUN_ID}",
        f"tried /opt/tool/bin,{RUN_ID}/jobs; PATH=/usr/bin:{RUN_ID} log:/var/log/x({RUN_ID}) PATH=C:\\bin;{RUN_ID}",
        f"in {RUN_ID}",
    ]
    (row, _other_agent_row) = context["cases"]
    assert row["question"] == f"Read {RUN_ID}/inputs/a.txt"
    assert row["ground_truth"] == f"path={RUN_ID}"


def test_context_replaces_the_skill_and_home_directories(tmp_path: Path) -> None:
    # Paths outside the run directory still name the user and where the skill lives.
    home = tmp_path / "home" / "alice"
    skill = home / "work" / "acme-skill"
    run = skill / "evals" / "results" / RUN_ID
    payload = {
        "dataset": [{"id": "case-1", "question": f"Read {skill / 'references' / 'a.md'}"}],
        "agents": {
            "codex": {
                "execution_errors": [
                    f"Skill validation failed: {skill / 'SKILL.md'} missing name",
                    f"cache {home / '.cache'} is full; see {run / 'jobs'}",
                    f"kept {home}-old, {skill}2 and {home.parent}",
                ],
            }
        },
    }

    context = compact_run_context(payload, run_dir=run, skill_dir=skill, home=home)

    errors = context["agents"]["codex"]["execution_errors"]
    assert errors[0] == f"Skill validation failed: {Path('acme-skill') / 'SKILL.md'} missing name"
    assert errors[1] == f"cache {Path('~') / '.cache'} is full; see {Path(RUN_ID) / 'jobs'}"
    # A directory whose name only starts with the same name is a different directory.
    assert errors[2] == f"kept {home}-old, {Path('~') / 'work' / 'acme-skill2'} and {home.parent}"
    assert context["cases"][0]["question"] == f"Read {Path('acme-skill') / 'references' / 'a.md'}"


def test_context_redacts_credentials_but_keeps_measurements() -> None:
    payload = {
        "agents": {
            "codex": {
                "execution_errors": ["agent failed: OPENAI_API_KEY=xxxxxxxxxxxx rejected"],
                "resources": {"tokens": {"with_skill": {"total": 12}, "baseline": {"total": 22}, "delta": -10}},
                "trials": [
                    {
                        "entry_id": "case-1",
                        "overall": 0.5,
                        "tokens": {"prompt": 5, "completion": 7},
                        "warnings": ["request sent Authorization: Bearer aaaaaaaaaaaaaaaa"],
                    }
                ],
            }
        }
    }
    context = compact_run_context(payload, run_dir=Path("/runs/demo"))
    serialized = json.dumps(context)
    assert "xxxxxxxxxxxx" not in serialized
    assert "aaaaaaaaaaaaaaaa" not in serialized
    assert "<redacted>" in context["agents"]["codex"]["execution_errors"][0]
    assert context["agents"]["codex"]["resources"]["tokens"]["delta"] == -10
    assert context["cases"][0]["with_skill"][0]["tokens"] == 12


# Built at runtime so secret scanners never see a literal credential.
_FAKE_GITHUB_TOKEN = "ghp_" + "a" * 36
_FAKE_AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
_PRIVATE_RUN = Path("/home/alice/projects/acme-secret-skill/evals/results") / RUN_ID


def _across_limit(limit: int, text: str, inside: int) -> str:
    """Filler, then ``text`` placed so that only its first ``inside`` characters fit within ``limit``."""
    return "x" * (limit - inside - 1) + " " + text + " done"


def test_context_masks_text_before_shortening_it() -> None:
    # Each credential or run path starts just inside its field's limit. Cut
    # first, it would keep a prefix that no redaction pattern or path matches.
    limits = tier3_feedback_chat
    path_text = f"failed at {_PRIVATE_RUN / 'jobs' / 'trial'}"
    path_inside = len("failed at ") + len(str(_PRIVATE_RUN)) - 5

    def token_at(limit: int) -> str:
        return _across_limit(limit, f"token {_FAKE_GITHUB_TOKEN}", len("token ") + 9)

    def key_at(limit: int) -> str:
        return _across_limit(limit, f"key={_FAKE_AWS_KEY}", len("key=") + 15)

    explanation = token_at(limits._MAX_EXPLANATION_CHARS)
    payload = {
        "dataset": [
            {
                "id": "case-1",
                "question": token_at(limits._MAX_QUESTION_CHARS),
                "ground_truth": key_at(limits._MAX_GROUND_TRUTH_CHARS),
            }
        ],
        "dimensions": [{"id": "accuracy", "explanation": explanation}, {"id": "speed", "explanation": 5}],
        "conclusions": [{"title": "t", "message": token_at(limits._MAX_FEEDBACK_CHARS)}],
        "suggestions": [key_at(limits._MAX_FEEDBACK_CHARS)],
        "agents": {
            "codex": {
                "execution_errors": [
                    key_at(limits._MAX_ERROR_CHARS),
                    _across_limit(limits._MAX_ERROR_CHARS, path_text, path_inside),
                ],
                "dimensions": [{"id": "accuracy", "explanation": explanation}],
                "trials": [
                    {"entry_id": "case-1", "overall": 1.0, "warnings": [token_at(limits._MAX_WARNING_CHARS)]},
                    {
                        "entry_id": "case-1",
                        "overall": 1.0,
                        "warnings": [_across_limit(limits._MAX_WARNING_CHARS, path_text, path_inside)],
                    },
                ],
            }
        },
    }

    context = compact_run_context(payload, run_dir=_PRIVATE_RUN)

    serialized = json.dumps(context)
    for leaked in ("ghp_", "AKIA", "alice", "acme-secret-skill"):
        assert leaked not in serialized
    agent = context["agents"]["codex"]
    (row,) = context["cases"]
    token_warning, path_warning = (attempt["warning"] for attempt in row["with_skill"])
    assert f"failed at {RUN_ID}" in path_warning
    assert f"failed at {RUN_ID}" in agent["execution_errors"][1]
    # The placeholder is longer than the key, so the shortened text ends in the marker.
    assert agent["execution_errors"][0].endswith(" key=a...<truncated>")
    assert context["dimensions"][1]["explanation"] == ""
    # Masking first still keeps every field within its limit.
    for text, limit in (
        (row["question"], limits._MAX_QUESTION_CHARS),
        (row["ground_truth"], limits._MAX_GROUND_TRUTH_CHARS),
        (token_warning, limits._MAX_WARNING_CHARS),
        (path_warning, limits._MAX_WARNING_CHARS),
        (agent["execution_errors"][0], limits._MAX_ERROR_CHARS),
        (agent["execution_errors"][1], limits._MAX_ERROR_CHARS),
        (context["dimensions"][0]["explanation"], limits._MAX_EXPLANATION_CHARS),
        (agent["dimensions"][0]["explanation"], limits._MAX_EXPLANATION_CHARS),
        (context["report_feedback"]["conclusions"][0]["message"], limits._MAX_FEEDBACK_CHARS),
        (context["report_feedback"]["suggestions"][0], limits._MAX_FEEDBACK_CHARS),
    ):
        assert len(text) <= limit


def test_context_keeps_long_text_without_spaces() -> None:
    # Compact JSON and minified errors have no space to cut at near the limit.
    # Masking must not blank them; they keep a masked prefix and a marker.
    limits = tier3_feedback_chat
    ground_truth = json.dumps({"files": [f"src/module_{index}.py" for index in range(60)]}, separators=(",", ":"))
    minified = "{" + ",".join(f'"k{index}":"v{index}"' for index in range(400)) + "}"
    token_inside = "x" * (limits._MAX_ERROR_CHARS - 100) + "," + _FAKE_GITHUB_TOKEN + "," + "y" * 3000
    payload = {
        "dataset": [{"id": "case-1", "question": "Which files? " + minified, "ground_truth": ground_truth}],
        "agents": {
            "codex": {
                "execution_errors": [minified, "Error: " + minified, token_inside, "z" * 100_000],
                "trials": [{"entry_id": "case-1", "overall": 1.0, "warnings": [minified]}],
            }
        },
    }

    context = compact_run_context(payload, run_dir=_PRIVATE_RUN)

    assert len(ground_truth) > limits._MAX_GROUND_TRUTH_CHARS * 4 + 256
    (row,) = context["cases"]
    errors = context["agents"]["codex"]["execution_errors"]
    for text, prefix, limit in (
        (row["ground_truth"], '{"files":["src/module_0.py",', limits._MAX_GROUND_TRUTH_CHARS),
        (row["question"], 'Which files? {"k0":"v0",', limits._MAX_QUESTION_CHARS),
        (row["with_skill"][0]["warning"], '{"k0":"v0",', limits._MAX_WARNING_CHARS),
        (errors[0], '{"k0":"v0",', limits._MAX_ERROR_CHARS),
        (errors[1], 'Error: {"k0":"v0",', limits._MAX_ERROR_CHARS),
    ):
        assert text.startswith(prefix)
        assert text.endswith("...<truncated>")
        assert len(text) == limit
    assert "ghp_" not in errors[2]
    assert errors[2].startswith("x" * 100) and "github-token-<redacted>" in errors[2]
    # Past the larger window the cut token is still dropped, and the field says so.
    assert errors[3] == "...<truncated>"


# -- case summaries -------------------------------------------------------------


def test_summarize_cases_covers_every_dataset_entry() -> None:
    payload = {
        "best_agent": "codex",
        "dataset": [
            {"id": "case-01", "question": "Q1", "should_trigger": True},
            {"id": "case-02", "question": "Q2"},
            {"id": "case-09", "question": "Q9", "should_trigger": False},
        ],
        "trials": [
            {"agent": "codex", "entry_id": "case-01", "overall": 0.9, "baseline_overall": 0.8, "status": "scored"},
            {"agent": "codex", "entry_id": "case-07", "overall": None, "status": "timeout", "warnings": ["timed out"]},
        ],
    }
    cases = _cases(payload)
    assert {row["entry_id"] for row in cases} == {"case-01", "case-02", "case-07", "case-09"}
    case_09 = next(row for row in cases if row["entry_id"] == "case-09")
    assert case_09["question"] == "Q9"
    assert case_09["case_type"] == "negative_control"
    assert case_09["with_skill"] == [] and case_09["baseline"] == []
    case_07 = next(row for row in cases if row["entry_id"] == "case-07")
    assert case_07["with_skill"] == [{"overall": None, "status": "timeout", "warning": "timed out"}]
    assert "case_type" not in next(row for row in cases if row["entry_id"] == "case-02")


def test_summarize_cases_keeps_every_attempt() -> None:
    with_skill = [
        {"entry_id": "case-02", "overall": 0.1, "status": "scored", "warnings": ["wrong file edited"]},
        {"entry_id": "case-02", "overall": 0.9, "status": "scored", "agent_seconds": 12.34, "tokens": {"prompt": 1}},
    ]
    payload = {
        "best_agent": "codex",
        "dataset": [{"id": "case-02", "question": "Q2"}, {"id": "case-03", "question": "Q3"}],
        "agents": {
            "codex": {
                "trials": with_skill,
                "trials_baseline": [
                    {"entry_id": "case-02", "overall": 0.5, "status": "scored"},
                    {"entry_id": "case-02", "overall": 0.3, "status": "scored"},
                ],
            },
            "claude-code": {"execution_status": "failed", "trials": [], "trials_baseline": []},
        },
        # The payload builder mirrors with-skill trials here; they must not be counted twice.
        "trials": [{**trial, "agent": "codex"} for trial in with_skill],
    }
    cases = _cases(payload)
    row = next(row for row in cases if row["agent"] == "codex" and row["entry_id"] == "case-02")
    assert row["with_skill"] == [
        {"overall": 0.1, "status": "scored", "warning": "wrong file edited"},
        # Incomplete token counts are left out rather than guessed.
        {"overall": 0.9, "status": "scored", "agent_seconds": 12.3},
    ]
    assert [attempt["overall"] for attempt in row["baseline"]] == [0.5, 0.3]
    assert "lift_scores" not in row and "baseline_overall" not in row
    # An agent that never ran a case still lists it, with no attempts.
    assert {(r["agent"], r["entry_id"]) for r in cases} == {
        ("codex", "case-02"),
        ("codex", "case-03"),
        ("claude-code", "case-02"),
        ("claude-code", "case-03"),
    }
    failed = next(row for row in cases if row["agent"] == "claude-code" and row["entry_id"] == "case-02")
    assert failed["with_skill"] == [] and failed["baseline"] == []


def test_summarize_cases_keeps_per_metric_scores_for_each_attempt() -> None:
    payload = {
        "dataset": [{"id": "case-03", "question": "Q3"}],
        "agents": {
            "codex": {
                "trials": [
                    {
                        "entry_id": "case-03",
                        "overall": 0.4,
                        "scores": {"security": 0.2, "accuracy": 0.61234567, "bad": float("nan"), "flag": True},
                        # Derived from the baseline list below; must not be copied.
                        "baseline_scores": {"security": 0.9, "accuracy": 0.5},
                        "lift_scores": {"security": -0.7, "accuracy": 0.1123},
                    },
                    {"entry_id": "case-03", "overall": 0.7, "scores": {"security": 0.8, "huge": 10**400, "text": "1"}},
                ],
                "trials_baseline": [
                    {"entry_id": "case-03", "overall": 0.7, "scores": {"security": 0.9, "accuracy": 0.5}},
                    {"entry_id": "case-03", "overall": 0.6, "scores": None},
                ],
            }
        },
    }
    row = next(iter(_cases(payload)))
    assert [attempt.get("scores") for attempt in row["with_skill"]] == [
        {"security": 0.2, "accuracy": 0.6123},
        {"security": 0.8},
    ]
    assert [attempt.get("scores") for attempt in row["baseline"]] == [{"security": 0.9, "accuracy": 0.5}, None]
    assert "lift_scores" not in json.dumps(row) and "baseline_scores" not in json.dumps(row)


def test_session_evidence_with_per_attempt_scores_stays_compact() -> None:
    metrics = ("security", "skill_execution", "skill_efficiency", "accuracy", "goal_accuracy", "behavior_check")
    n_cases, n_attempts = 100, 2

    def trials(score: float) -> list[dict]:
        return [
            {"entry_id": f"case-{i:03d}", "overall": score, "scores": dict.fromkeys(metrics, score)}
            for i in range(n_cases)
            for _ in range(n_attempts)
        ]

    payload = {
        "dataset": [{"id": f"case-{i:03d}", "question": f"Question {i}"} for i in range(n_cases)],
        "agents": {
            name: {"trials": trials(0.8125), "trials_baseline": trials(0.5)} for name in ("codex", "claude-code")
        },
    }
    context = compact_run_context(payload, run_dir=Path("/runs/demo") / RUN_ID)
    attempts = [attempt for row in context["cases"] for arm in ("with_skill", "baseline") for attempt in row[arm]]
    assert len(attempts) == 2 * 2 * n_cases * n_attempts
    assert "cases_omitted" not in context
    assert all(attempt["scores"] == dict.fromkeys(metrics, attempt["overall"]) for attempt in attempts)
    evidence = Tier3FeedbackSession(run_context=context, assistant=MagicMock()).transcript[0]["content"]
    # Re-sent every turn, so the evidence is compact JSON (no indentation).
    assert "\n " not in evidence
    assert len(evidence) < 250 * len(attempts)


def test_case_evidence_is_capped_for_large_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tier3_feedback_chat, "_MAX_CASE_EVIDENCE_CHARS", 1_000)
    payload = {"dataset": [{"id": f"case-{i:03d}", "question": "Q" * 100} for i in range(50)]}
    context = compact_run_context(payload, run_dir=Path("/runs/demo"))
    assert 0 < len(context["cases"]) < 50
    assert context["cases_omitted"] == {"default": 50 - len(context["cases"])}


def test_case_evidence_cap_keeps_every_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Over the budget, each agent keeps the same leading cases; no agent is dropped whole.
    monkeypatch.setattr(tier3_feedback_chat, "_MAX_CASE_EVIDENCE_CHARS", 6_000)
    agents = ("codex", "gemini-cli", "opencode")
    n_cases = 40

    def trials() -> list[dict[str, Any]]:
        return [{"entry_id": f"case-{i:02d}", "overall": 0.5} for i in range(n_cases) for _ in range(3)]

    payload = {
        "dataset": [{"id": f"case-{i:02d}", "question": f"Question {i}"} for i in range(n_cases)],
        "agents": {name: {"trials": trials(), "trials_baseline": trials()} for name in agents},
    }
    context = compact_run_context(payload, run_dir=Path("/runs/demo") / RUN_ID)

    kept: dict[str, list[str]] = {name: [] for name in agents}
    for row in context["cases"]:
        kept[row["agent"]].append(row["entry_id"])
    assert all(kept[name] for name in agents)
    # Rows stay grouped by agent and in case order.
    assert [row["agent"] for row in context["cases"]] == sorted(row["agent"] for row in context["cases"])
    assert all(ids == sorted(ids) for ids in kept.values())
    assert max(map(len, kept.values())) - min(map(len, kept.values())) <= 1
    assert context["cases_omitted"] == {name: n_cases - len(kept[name]) for name in agents}


# -- session --------------------------------------------------------------------


def test_session_reply_uses_chat() -> None:
    assistant = MagicMock()
    assistant.chat.return_value = "Try clarifying step 2 in the skill."
    session = Tier3FeedbackSession(run_context={"skill_name": "demo", "overall_score": 0.5}, assistant=assistant)

    answer = session.reply("How can I improve?")

    assert answer == "Try clarifying step 2 in the skill."
    assert assistant.chat.call_count == 1
    messages = assistant.chat.call_args.args[0]
    assert messages[-1] == {"role": "user", "content": "How can I improve?"}
    assert session.transcript[-1] == {"role": "assistant", "content": answer}
    assert (session.answered_turns, session.failed_turns) == (1, 0)


def test_session_history_alternates_roles_and_is_stable() -> None:
    assistant = MagicMock()
    assistant.chat.side_effect = ["First answer", "Second answer"]
    session = Tier3FeedbackSession(run_context={"skill_name": "demo"}, assistant=assistant)
    session.reply("First question")
    first_request = assistant.chat.call_args.args[0]
    transcript = session.transcript
    transcript[0]["content"] = "overwritten"
    session.reply("Second question")
    second_request = assistant.chat.call_args.args[0]

    assert first_request[-1] == {"role": "user", "content": "First question"}
    assert session.transcript[0]["content"] != "overwritten"
    # The evidence travels with the system prompt, so user and assistant turns alternate.
    assert [message["role"] for message in second_request] == ["system", "user", "assistant", "user"]
    assert "Run evidence" in second_request[0]["content"]
    assert second_request[-2:] == [
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]


def test_session_reply_llm_error_message() -> None:
    assistant = MagicMock()
    assistant.chat.side_effect = LLMClientError("No API key resolved for openai.")
    session = Tier3FeedbackSession(run_context={"skill_name": "demo"}, assistant=assistant)
    before = session.transcript
    answer = session.reply("hello")
    assert answer == "The inference request failed: No API key resolved for openai."
    assert session.transcript == before
    assert (session.answered_turns, session.failed_turns) == (0, 1)


class _FakeStatusError(Exception):
    """Shaped like an SDK status error: a status code and an untrusted response body."""

    def __init__(self, status_code: int) -> None:
        super().__init__("provider said: secret-body-text")
        self.status_code = status_code
        self.body = {"error": {"message": "secret-body-text", "code": "invalid_api_key"}}


def _wrapped_configuration_error() -> LLMClientError:
    try:
        raise ProviderConfigurationError("OPENAI_API_KEY or SKILL_EVAL_LLM_API_KEY is required.")
    except ProviderConfigurationError as exc:
        try:
            raise LLMClientError(str(exc)) from exc
        except LLMClientError as wrapped:
            return wrapped


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LLMClientRefusedError("LLM provider declined the request (stop reason: refusal)"), "declined"),
        (LLMClientTruncatedError("LLM response stopped at the output-token limit"), "output limit"),
        (_FakeStatusError(401), "HTTP 401"),
        (ProviderConfigurationError("OPENAI_API_KEY or SKILL_EVAL_LLM_API_KEY is required."), "not configured"),
        (_wrapped_configuration_error(), "Inference is not configured: OPENAI_API_KEY"),
        (IndexError("list index out of range"), "inference request failed"),
    ],
    ids=["refusal", "truncated", "http-401", "provider-config", "wrapped-provider-config", "unexpected"],
)
def test_session_reply_failure_messages(error: Exception, expected: str) -> None:
    assistant = MagicMock()
    assistant.chat.side_effect = error
    session = Tier3FeedbackSession(run_context={"skill_name": "demo"}, assistant=assistant)
    before = session.transcript
    answer = session.reply("hello")
    assert expected in answer
    # Provider bodies are never echoed.
    assert "secret-body-text" not in answer
    if isinstance(error, LLMClientRefusedError | LLMClientTruncatedError):
        # A model refusal or ceiling is not a credential problem.
        assert "credential" not in answer.lower()
    assert session.transcript == before


def test_session_reply_keyboard_interrupt_keeps_history() -> None:
    assistant = MagicMock()
    assistant.chat.side_effect = [KeyboardInterrupt(), "Second answer"]
    session = Tier3FeedbackSession(run_context={"skill_name": "demo"}, assistant=assistant)
    before = session.transcript
    with pytest.raises(KeyboardInterrupt):
        session.reply("First question")
    assert session.transcript == before
    assert session.reply("Second question") == "Second answer"
    assert session.transcript[len(before) :] == [
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
    ]


def test_session_survives_an_sdk_authentication_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from openai import AuthenticationError

    _use_openai(monkeypatch)
    sdk = MagicMock()
    sdk.chat.completions.create.side_effect = [
        AuthenticationError("Unauthorized", response=MagicMock(status_code=401), body=None),
        _chat_response("Second answer"),
    ]
    with patch("openai.OpenAI", return_value=sdk):
        session = Tier3FeedbackSession(run_context={"skill_name": "demo"})
        first = session.reply("First question")
        second = session.reply("Second question")

    assert first.startswith("The inference request failed: HTTP 401")
    assert second == "Second answer"
    assert [message["role"] for message in sdk.chat.completions.create.call_args.kwargs["messages"]] == [
        "system",
        "user",
    ]


def test_open_feedback_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = {"skill_name": "demo", "dimensions": [], "evaluators": {}, "trials": [], "dataset": []}
    monkeypatch.setattr(tier3_feedback_chat, "assert_completed_run_dir", lambda run: run)
    monkeypatch.setattr(tier3_feedback_chat, "load_agent_eval_payload", lambda *_: payload)
    monkeypatch.setattr(tier3_feedback_chat, "resolve_llm_provider", lambda: None)

    session = open_feedback_session(Path("demo"), tmp_path / RUN_ID)

    assert session.run_context["skill_name"] == "demo"
    assert session.run_context["run_id"] == RUN_ID


@pytest.mark.parametrize("explicit", [False, True], ids=["inferred-skill", "explicit-skill"])
def test_open_feedback_session_replaces_the_skill_and_home_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, explicit: bool
) -> None:
    home = tmp_path / "home" / "alice"
    skill = home / "work" / "acme-skill"
    # An inferred skill is only a name; a run in the skill's evals/results still names its directory.
    # An explicit --skill also covers a run kept under another results directory.
    run = home / "results" / "acme-skill" / RUN_ID if explicit else skill / "evals" / "results" / RUN_ID
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    error = f"Skill validation failed: {skill / 'SKILL.md'} missing name; cache {home / '.cache'}"
    payload = {"skill_name": "acme-skill", "agents": {"codex": {"execution_errors": [error]}}}
    monkeypatch.setattr(tier3_feedback_chat, "assert_completed_run_dir", lambda run: run)
    monkeypatch.setattr(tier3_feedback_chat, "load_agent_eval_payload", lambda *_: payload)
    monkeypatch.setattr(tier3_feedback_chat, "resolve_llm_provider", lambda: None)

    session = open_feedback_session(skill if explicit else Path("acme-skill"), run)

    assert session.run_context["agents"]["codex"]["execution_errors"] == [
        f"Skill validation failed: {Path('acme-skill') / 'SKILL.md'} missing name; cache {Path('~') / '.cache'}"
    ]


def test_open_feedback_session_rejects_misconfigured_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SKILL_EVAL_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("SKILL_EVAL_LLM_BASE_URL", raising=False)
    loader = MagicMock()
    monkeypatch.setattr(tier3_feedback_chat, "assert_completed_run_dir", lambda run: run)
    monkeypatch.setattr(tier3_feedback_chat, "load_agent_eval_payload", loader)

    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        open_feedback_session(Path("demo"), tmp_path)
    loader.assert_not_called()


# -- loader ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("run_id", "schema_version", "restored"),
    [(RUN_ID, "1.0", True), ("20250101_000000", "1.0", False), (RUN_ID, "0.9", False)],
    ids=["this-run", "another-run", "unknown-schema"],
)
def test_loader_reuses_only_run_bound_feedback_without_new_judge_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_id: str, schema_version: str, restored: bool
) -> None:
    from skillevaluator.evaluation import tier3_report

    run = tmp_path / RUN_ID
    run.mkdir()
    _write_json(
        run / "result.json",
        {
            "run_id": run_id,
            "tier3_feedback": {
                "schema_version": schema_version,
                "conclusions": [{"title": "Saved conclusion", "message": "Evidence from this run"}],
                "recommendations": [],
                "suggestions": ["Saved suggestion"],
                "suggestions_v2": [],
            },
        },
    )
    payload = {"skill_name": "demo", "conclusions": [{"title": "Deterministic conclusion"}]}
    builder = MagicMock(return_value=SimpleNamespace(metadata={"agent_eval": payload}))
    monkeypatch.setattr(tier3_report, "agent_eval_result_from_directory", builder)

    loaded = load_agent_eval_payload(Path("demo"), run)

    assert builder.call_args.kwargs["use_llm_judge"] is False
    assert builder.call_args.kwargs["enforce_embedded_size"] is False
    assert loaded["conclusions"][0]["title"] == ("Saved conclusion" if restored else "Deterministic conclusion")
    assert payload["conclusions"][0]["title"] == "Deterministic conclusion"


def test_loader_builds_context_from_a_real_run(tmp_path: Path) -> None:
    run = _real_run(tmp_path / "results")
    before = sorted(path.relative_to(run).as_posix() for path in run.rglob("*"))

    payload = load_agent_eval_payload(Path("demo-skill"), run)
    context = compact_run_context(payload, run_dir=run)

    assert context["run_id"] == RUN_ID
    assert context["skill_name"] == "demo-skill"
    agent = context["agents"]["opencode"]
    assert agent["resources"]["tokens"]["delta"] == -20
    assert agent["resources"]["agent_seconds"]["delta"] == -2
    (row,) = context["cases"]
    assert row["entry_id"] == "case-1"
    assert row["question"] == "Create the report."
    assert row["case_type"] == "skill_activation"
    (with_skill,) = row["with_skill"]
    assert with_skill["overall"] == 0.9
    assert with_skill["tokens"] == 30
    assert with_skill["agent_seconds"] == 3
    assert set(with_skill["scores"]) == set(DEFAULT_METRICS)
    assert with_skill["warning"] == f"retry under {RUN_ID}"
    assert row["baseline"][0]["tokens"] == 50
    assert context["report_feedback"]["conclusions"] == [
        {"severity": "warn", "title": "Saved conclusion", "message": "From this run"}
    ]
    assert context["report_feedback"]["suggestions"] == ["Saved suggestion"]
    assert str(tmp_path) not in json.dumps(context)
    # Reading a run never writes to it.
    assert sorted(path.relative_to(run).as_posix() for path in run.rglob("*")) == before


# -- CLI ------------------------------------------------------------------------


def test_tier3_help_lists_feedback() -> None:
    group_help = CliRunner().invoke(cli, ["tier3", "--help"])
    command_help = _invoke(["--help"])
    assert group_help.exit_code == 0, group_help.output
    assert "feedback" in group_help.output
    assert command_help.exit_code == 0, command_help.output
    assert "--skill" in command_help.output


def test_skill_folder_named_feedback_needs_an_explicit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The registered command wins over a same-named folder, as the docs say.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "feedback").mkdir()
    chat = CliRunner().invoke(cli, ["tier3", "feedback"])
    assert chat.exit_code == 2
    assert "RUN_DIR" in chat.output
    group = cli.commands["tier3"]
    direct = group.make_context("tier3", ["./feedback"])
    assert direct.meta["direct_tier_args"] == ["./feedback"]
    expert = group.make_context("tier3", ["evaluate", "feedback"])
    assert "direct_tier_args" not in expert.meta
    _name, command, args = group.resolve_command(expert, ["evaluate", "feedback"])
    assert command is group.commands["evaluate"]
    assert args == ["feedback"]


def test_feedback_cli_preserves_tradeoffs_and_renders_model_text_literally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plain_console: None
) -> None:
    assistant = MagicMock()
    assistant.chat.side_effect = ["Consider [red]step 2[/red].", "The 10% token budget remains a preference."]
    session = Tier3FeedbackSession(run_context={"skill_name": "demo"}, assistant=assistant)
    skill, run = tmp_path / "demo", tmp_path / RUN_ID
    opened = MagicMock(return_value=session)
    skill.mkdir()
    run.mkdir()
    monkeypatch.setattr(tier3_feedback_chat, "assert_completed_run_dir", lambda run_dir: run_dir.expanduser().resolve())
    monkeypatch.setattr(tier3_feedback_chat, "open_feedback_session", opened)

    result = _invoke(
        [str(run), "--skill", str(skill)],
        "I accept 10% more tokens for speed.\nWhich case should I compare?\nexit\n",
    )

    assert result.exit_code == 0, result.output
    opened.assert_called_once_with(skill.resolve(), run.resolve())
    assert "Assistant: Consider [red]step 2[/red]." in result.output
    assert assistant.chat.call_count == 2
    second_request = assistant.chat.call_args.args[0]
    assert second_request[-3]["content"] == "I accept 10% more tokens for speed."
    assert second_request[-1]["content"] == "Which case should I compare?"


def test_feedback_cli_provider_misconfig_before_first_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plain_console: None
) -> None:
    run = _real_run(tmp_path / "results")
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    for name in ("OPENAI_API_KEY", "SKILL_EVAL_LLM_API_KEY", "OPENAI_BASE_URL", "SKILL_EVAL_LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    result = _invoke([str(run)], "hello\n")

    assert result.exit_code == 1
    assert "Error: OPENAI_API_KEY or SKILL_EVAL_LLM_API_KEY is required." in result.output
    assert "You:" not in result.output


def test_feedback_cli_rejects_incomplete_run_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plain_console: None
) -> None:
    skill = tmp_path / "demo"
    skill.mkdir()
    run = tmp_path / RUN_ID
    run.mkdir()
    opened = MagicMock()
    monkeypatch.setattr(tier3_feedback_chat, "open_feedback_session", opened)

    result = _invoke([str(run), "--skill", str(skill)], "exit\n")

    assert result.exit_code == 1, result.output
    assert "Not a completed Tier 3 run directory" in result.output
    opened.assert_not_called()


def test_feedback_cli_rejects_skill_that_does_not_match_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plain_console: None
) -> None:
    other = tmp_path / "other-skill"
    other.mkdir()
    run = _completed_run(tmp_path / "results", skill_name="demo-skill")
    opened = MagicMock()
    monkeypatch.setattr(tier3_feedback_chat, "open_feedback_session", opened)

    result = _invoke([str(run), "--skill", str(other)], "exit\n")

    assert result.exit_code == 1, result.output
    assert "does not match" in result.output
    opened.assert_not_called()


def test_feedback_cli_strips_terminal_escapes_from_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plain_console: None
) -> None:
    assistant = MagicMock()
    assistant.chat.return_value = "Use \x1b]52;c;ZXZpbA==\x07step 2\x1b[2J\x1b[31m now\x1b[0m\x9b\x00"
    session = Tier3FeedbackSession(run_context={"skill_name": "demo"}, assistant=assistant)

    result = _invoke_with_session(monkeypatch, tmp_path / "demo", tmp_path / RUN_ID, session, "why?\nexit\n")

    assert result.exit_code == 0, result.output
    # The console prints without color, so no escape or control byte may remain.
    for injected in ("\x1b", "\x07", "\x9b", "\x00", "ZXZpbA=="):
        assert injected not in result.output
    assert "Use step 2 now" in result.output
    # History keeps the model's raw answer; only the terminal copy is sanitized.
    assert session.transcript[-1]["content"] == assistant.chat.return_value


def test_feedback_cli_ctrl_c_during_answer_returns_to_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plain_console: None
) -> None:
    assistant = MagicMock()
    assistant.chat.side_effect = [KeyboardInterrupt(), "second answer"]
    session = Tier3FeedbackSession(run_context={"skill_name": "demo"}, assistant=assistant)

    result = _invoke_with_session(monkeypatch, tmp_path / "demo", tmp_path / RUN_ID, session, "first\nsecond\nexit\n")

    assert result.exit_code == 0, result.output
    assert "Interrupted; that question was discarded." in result.output
    assert "second answer" in result.output
    second_request = assistant.chat.call_args.args[0]
    assert [message["content"] for message in second_request if message["role"] == "user"] == ["second"]


def test_feedback_cli_escapes_panel_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plain_console: None) -> None:
    session = Tier3FeedbackSession(run_context={"skill_name": "sk[old]ll"}, assistant=MagicMock())

    result = _invoke_with_session(
        monkeypatch, tmp_path / "sk[old]ll", tmp_path / "x[" / "y]" / RUN_ID, session, "exit\n"
    )

    assert result.exit_code == 0, result.output
    assert f"sk[old]ll · {RUN_ID}" in result.output
    assert str(Path("x[") / "y]" / RUN_ID) in result.output


def test_feedback_cli_eof_exits_zero_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plain_console: None
) -> None:
    run = _real_run(tmp_path / "results")
    _use_openai(monkeypatch)
    sdk_factory = MagicMock()

    with patch("openai.OpenAI", sdk_factory):
        result = _invoke([str(run)], "")

    assert result.exit_code == 0, result.output
    assert "Post-run feedback" in result.output
    assert f"demo-skill · {RUN_ID}" in result.output
    # No client is created and no request is sent before the first question.
    sdk_factory.assert_not_called()


def test_feedback_cli_sends_alternating_history_for_a_real_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plain_console: None
) -> None:
    run = _real_run(tmp_path / "results")
    _use_openai(monkeypatch)
    sdk = MagicMock()
    sdk.chat.completions.create.side_effect = [_chat_response("Case 1 improved."), _chat_response("Keep it.")]

    with patch("openai.OpenAI", return_value=sdk):
        result = _invoke([str(run)], "Which case improved?\nShould I keep the skill?\n")

    assert result.exit_code == 0, result.output
    assert "Assistant: Case 1 improved." in result.output
    assert "Assistant: Keep it." in result.output
    first, second = (call.kwargs["messages"] for call in sdk.chat.completions.create.call_args_list)
    assert [message["role"] for message in first] == ["system", "user"]
    assert [message["role"] for message in second] == ["system", "user", "assistant", "user"]
    evidence = first[0]["content"]
    assert f'"run_id":"{RUN_ID}"' in evidence
    assert "Saved suggestion" in evidence
    assert str(tmp_path) not in evidence
    assert tmp_path.as_posix() not in evidence


def test_feedback_cli_fails_when_every_question_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plain_console: None
) -> None:
    assistant = MagicMock()
    assistant.chat.side_effect = LLMClientRefusedError("declined")
    session = Tier3FeedbackSession(run_context={"skill_name": "demo"}, assistant=assistant)

    result = _invoke_with_session(monkeypatch, tmp_path / "demo", tmp_path / RUN_ID, session, "one\ntwo\n")

    assert result.exit_code == 1, result.output
    assert result.output.count("The model declined to answer that question.") == 2


@pytest.mark.parametrize("changed", [True, False], ids=["changed", "unchanged"])
def test_feedback_cli_notes_a_skill_that_changed_since_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plain_console: None, changed: bool
) -> None:
    from skillevaluator.publication_identity import publication_target_from_path

    skill = tmp_path / "demo-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: demo-skill\ndescription: Demo.\n---\n", encoding="utf-8")
    _write_json(skill / "evals" / "evals.json", [{"id": "case-1", "question": "Create the report."}])
    target = publication_target_from_path(skill)
    assert target is not None
    # The run lives under the skill's evals/results/, which the digest leaves out.
    run = _real_run(
        skill / "evals" / "results",
        skill_digest="sha256:" + "0" * 64 if changed else target["skill_digest"],
    )
    _use_openai(monkeypatch)

    result = _invoke([str(run), "--skill", str(skill)], "")

    assert result.exit_code == 0, result.output
    assert ("The skill has changed since this run" in result.output) is changed
