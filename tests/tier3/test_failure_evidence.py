# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for harness-native agent failure evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor import failure_evidence
from skillevaluator.tier3.harbor.failure_evidence import (
    FailureEvidence,
    codex_presemantic_failure_marker_from_text,
    failure_from_agent_event,
    first_agent_log_failure,
    first_trial_agent_failure,
)

OLD_FAILURE_MARKERS = (
    "api error:",
    "incorrect api key",
    "unauthorized",
    "thinking.type.enabled",
    "configuration is invalid",
    "failed to connect to websocket",
    "model_not_found",
)


def _write_jsonl(path: Path, *events: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(event)}\n" for event in events), encoding="utf-8")


def _codex_presemantic_marker(path: Path) -> dict[str, str] | None:
    return codex_presemantic_failure_marker_from_text(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("marker", OLD_FAILURE_MARKERS)
def test_claude_tool_result_error_words_are_not_failure(marker: str) -> None:
    event = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "content": f"Documentation mentions {marker}",
                    "is_error": False,
                }
            ]
        },
    }

    assert failure_from_agent_event("claude-code", event, "claude-code.txt") is None


def test_claude_terminal_result_is_failure() -> None:
    evidence = failure_from_agent_event(
        "claude-code",
        {"type": "result", "is_error": True, "result": "API Error: 401"},
        "claude-code.txt",
    )

    assert evidence is not None
    assert evidence.component == "agent"
    assert evidence.phase == "execute"
    assert evidence.event_type == "result"
    assert evidence.kind == "agent_runtime"
    assert evidence.source == "claude-code.txt"
    assert evidence.message == "API Error: 401"
    assert evidence.retryable is None


def test_claude_success_result_is_not_failure() -> None:
    event = {
        "type": "result",
        "is_error": False,
        "result": "Explained why configuration is invalid in the source text",
    }

    assert failure_from_agent_event("claude-code", event, "claude-code.txt") is None


@pytest.mark.parametrize(
    "is_error",
    ["false", "true", "True", 1, 0, None],
    ids=["str-false", "str-true", "str-True", "int-1", "int-0", "json-null"],
)
def test_claude_non_boolean_is_error_never_classifies(is_error: object) -> None:
    """Only the JSON boolean ``true`` marks a claude-code result as terminal.

    Truthy coercion (``bool(event.get("is_error"))``) would classify the
    string values and ``1`` here; the identity check must reject them all.
    """
    event = {"type": "result", "is_error": is_error, "result": "API Error: 401"}

    assert failure_from_agent_event("claude-code", event, "claude-code.txt") is None


def test_claude_result_missing_is_error_is_not_failure() -> None:
    event = {"type": "result", "result": "API Error: 401"}

    assert failure_from_agent_event("claude-code", event, "claude-code.txt") is None


@pytest.mark.parametrize("marker", OLD_FAILURE_MARKERS)
def test_codex_command_output_is_not_failure(marker: str) -> None:
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "aggregated_output": f"Documentation mentions {marker}",
        },
    }

    assert failure_from_agent_event("codex", event, "codex.txt") is None


def test_codex_transient_error_followed_by_completion_is_not_failure(tmp_path: Path) -> None:
    path = tmp_path / "codex.txt"
    path.write_text(
        "not JSON diagnostic prose\n"
        + json.dumps({"type": "error", "message": "failed to connect to websocket"})
        + "\n\n"
        + json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10}})
        + "\n",
        encoding="utf-8",
    )

    assert first_agent_log_failure(path) is None


def test_codex_transient_error_then_turn_failed_reports_terminal_event(tmp_path: Path) -> None:
    path = tmp_path / "codex.txt"
    _write_jsonl(
        path,
        {"type": "error", "message": "transient stream error"},
        {"type": "item.completed", "item": {"type": "command_execution", "aggregated_output": "ok"}},
        {"type": "turn.failed", "error": {"message": "terminal provider failure"}},
    )

    evidence = first_agent_log_failure(path)

    assert evidence is not None
    assert evidence.event_type == "turn.failed"
    assert evidence.message == "terminal provider failure"


def test_codex_turn_failed_is_failure() -> None:
    evidence = failure_from_agent_event(
        "codex",
        {"type": "turn.failed", "error": {"message": "401 Unauthorized"}},
        "codex.txt",
    )

    assert evidence is not None
    assert evidence.event_type == "turn.failed"
    assert evidence.kind == "agent_runtime"
    assert evidence.message == "401 Unauthorized"


def test_codex_presemantic_failure_marker_accepts_observed_lifecycle(tmp_path: Path) -> None:
    path = tmp_path / "codex.txt"
    _write_jsonl(
        path,
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {"type": "error", "message": "Bad Request"},
        {"type": "turn.failed", "error": {"message": "Bad Request"}},
    )

    assert _codex_presemantic_marker(path) == {
        "stage": "agent_adapter_bootstrap",
        "reason_code": "adapter_model_protocol_negotiation_failed",
    }


def test_codex_presemantic_failure_marker_accepts_current_cli_preamble(tmp_path: Path) -> None:
    path = tmp_path / "codex.txt"
    path.write_text(
        "WARNING: proceeding, even though we could not create PATH aliases\n"
        "Reading additional input from stdin...\n"
        + "\n".join(
            json.dumps(event)
            for event in (
                {"type": "thread.started", "thread_id": "thread-1"},
                {
                    "type": "item.completed",
                    "item": {"type": "error", "message": "model metadata diagnostic"},
                },
                {"type": "turn.started"},
                {"type": "error", "message": "provider rejected request"},
                {"type": "turn.failed", "error": {"message": "provider rejected request"}},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert _codex_presemantic_marker(path) is not None


def test_codex_presemantic_failure_marker_accepts_timestamped_cli_diagnostics() -> None:
    transcript = "\n".join(
        (
            "Reading additional input from stdin...",
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"item.completed","item":{"type":"error","message":"metadata"}}',
            '{"type":"turn.started"}',
            "2026-07-15T08:02:51.293729Z  WARN codex_core::responses_retry: retrying",
            '{"type":"error","message":"Reconnecting 1/5"}',
            "2026-07-15T08:02:52.156670Z  WARN codex_core::responses_retry: retrying",
            '{"type":"error","message":"Reconnecting 2/5"}',
            '{"type":"error","message":"provider rejected request"}',
            '{"type":"turn.failed","error":{"message":"provider rejected request"}}',
        )
    )

    assert codex_presemantic_failure_marker_from_text(transcript) is not None


def test_codex_presemantic_failure_marker_accepts_terminal_without_separate_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "codex.txt"
    _write_jsonl(
        path,
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {"type": "turn.failed", "error": {"message": "request failed"}},
    )

    assert _codex_presemantic_marker(path) is not None


def test_codex_presemantic_failure_marker_accepts_typed_error_item(tmp_path: Path) -> None:
    path = tmp_path / "codex.txt"
    _write_jsonl(
        path,
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "error", "message": "model rerouted"}},
        {"type": "turn.failed", "error": {"message": "request failed"}},
    )

    assert _codex_presemantic_marker(path) is not None


def test_codex_presemantic_failure_marker_accepts_diagnostic_before_turn(tmp_path: Path) -> None:
    path = tmp_path / "codex.txt"
    _write_jsonl(
        path,
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "item.completed", "item": {"type": "error", "message": "config warning"}},
        {"type": "turn.started"},
        {"type": "turn.failed", "error": {"message": "request failed"}},
    )

    assert _codex_presemantic_marker(path) is not None


def test_codex_presemantic_failure_marker_rejects_invalid_unicode() -> None:
    assert codex_presemantic_failure_marker_from_text("\ud800") is None


def test_codex_presemantic_failure_marker_rejects_bom_only_line_without_crashing() -> None:
    assert codex_presemantic_failure_marker_from_text("\ufeff\n") is None


def test_codex_presemantic_failure_marker_rejects_oversized_transcript() -> None:
    transcript = "\n".join(
        (
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"turn.started"}',
            '{"type":"error","message":"' + ("x" * (256 * 1024)) + '"}',
            '{"type":"turn.failed","error":{"message":"request failed"}}',
        )
    )

    assert codex_presemantic_failure_marker_from_text(transcript) is None


def test_codex_presemantic_failure_marker_rejects_event_after_terminal() -> None:
    transcript = "\n".join(
        (
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"turn.started"}',
            '{"type":"turn.failed","error":{"message":"request failed"}}',
            '{"type":"error","message":"late event"}',
        )
    )

    assert codex_presemantic_failure_marker_from_text(transcript) is None


def test_codex_presemantic_failure_marker_rejects_non_json_after_thread_start() -> None:
    transcript = "\n".join(
        (
            '{"type":"thread.started","thread_id":"thread-1"}',
            "unexpected launcher output",
            '{"type":"turn.started"}',
            '{"type":"turn.failed","error":{"message":"request failed"}}',
        )
    )

    assert codex_presemantic_failure_marker_from_text(transcript) is None


def test_codex_presemantic_failure_marker_rejects_duplicate_json_keys() -> None:
    transcript = "\n".join(
        (
            '{"type":"diagnostic","type":"error"}',
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"turn.started"}',
            '{"type":"turn.failed","error":{"message":"request failed"}}',
        )
    )

    assert codex_presemantic_failure_marker_from_text(transcript) is None


def test_codex_presemantic_failure_marker_rejects_json_like_malformed_preamble() -> None:
    transcript = "\n".join(
        (
            '{"type":"diagnostic"',
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"turn.started"}',
            '{"type":"turn.failed","error":{"message":"request failed"}}',
        )
    )

    assert codex_presemantic_failure_marker_from_text(transcript) is None


@pytest.mark.parametrize(
    "extra_event",
    [
        {"type": "item.started", "item": {"type": "reasoning"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "hello"}},
        {"type": "item.completed", "item": {"type": "command_execution", "aggregated_output": "done"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1}},
        {"type": "future.lifecycle.event"},
    ],
    ids=["reasoning", "assistant", "tool", "completed", "unknown"],
)
def test_codex_presemantic_failure_marker_rejects_semantic_or_unknown_events(
    tmp_path: Path, extra_event: dict[str, object]
) -> None:
    path = tmp_path / "codex.txt"
    _write_jsonl(
        path,
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        extra_event,
        {"type": "error", "message": "Bad Request"},
        {"type": "turn.failed", "error": {"message": "Bad Request"}},
    )

    assert _codex_presemantic_marker(path) is None


@pytest.mark.parametrize(
    "events",
    [
        ({"type": "turn.started"}, {"type": "error"}, {"type": "turn.failed"}),
        ({"type": "thread.started"}, {"type": "error"}, {"type": "turn.failed"}),
        ({"type": "thread.started"}, {"type": "turn.started"}, {"type": "error"}),
    ],
    ids=["missing-thread", "missing-turn", "missing-terminal"],
)
def test_codex_presemantic_failure_marker_requires_complete_failure_lifecycle(
    tmp_path: Path, events: tuple[dict[str, object], ...]
) -> None:
    path = tmp_path / "codex.txt"
    _write_jsonl(path, *events)

    assert _codex_presemantic_marker(path) is None


def test_codex_presemantic_failure_marker_rejects_malformed_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "codex.txt"
    path.write_text(
        json.dumps({"type": "thread.started"})
        + "\n"
        + "not-json\n"
        + json.dumps({"type": "turn.started"})
        + "\n"
        + json.dumps({"type": "error"})
        + "\n"
        + json.dumps({"type": "turn.failed"})
        + "\n",
        encoding="utf-8",
    )

    assert _codex_presemantic_marker(path) is None


def test_opencode_structured_error_is_failure() -> None:
    evidence = failure_from_agent_event(
        "opencode",
        {
            "type": "error",
            "error": {"name": "APIError", "data": {"statusCode": 401}},
        },
        "opencode.txt",
    )

    assert evidence is not None
    assert evidence.event_type == "error"
    assert evidence.kind == "agent_runtime"
    assert "APIError" in evidence.message
    assert "401" in evidence.message


@pytest.mark.parametrize("event_type", ["message", "message.updated", "assistant"])
def test_opencode_error_words_inside_normal_content_are_not_failure(event_type: str) -> None:
    event = {
        "type": event_type,
        "message": {"content": "API Error: incorrect api key; model_not_found"},
    }

    assert failure_from_agent_event("opencode", event, "opencode.txt") is None


def test_opencode_unstructured_error_is_not_failure() -> None:
    event = {"type": "error", "error": "401 Unauthorized"}

    assert failure_from_agent_event("opencode", event, "opencode.txt") is None


@pytest.mark.parametrize(
    ("filename", "event", "expected_type"),
    [
        ("claude-code.txt", {"type": "result", "is_error": True, "result": "failed"}, "result"),
        ("codex.txt", {"type": "turn.failed", "error": {"message": "failed"}}, "turn.failed"),
        ("opencode.txt", {"type": "error", "error": {"message": "failed"}}, "error"),
    ],
)
def test_agent_is_inferred_from_log_basename(
    tmp_path: Path,
    filename: str,
    event: dict[str, object],
    expected_type: str,
) -> None:
    path = tmp_path / filename
    _write_jsonl(path, {"type": "message", "content": "benign"}, event)

    evidence = first_agent_log_failure(path)

    assert evidence is not None
    assert evidence.event_type == expected_type
    assert evidence.source == str(path)


def test_explicit_agent_allows_nonstandard_log_basename(tmp_path: Path) -> None:
    path = tmp_path / "agent-output.jsonl"
    _write_jsonl(path, {"type": "turn.failed", "error": {"message": "terminal"}})

    evidence = first_agent_log_failure(path, agent="codex")

    assert evidence is not None
    assert evidence.event_type == "turn.failed"


@pytest.mark.parametrize("nested", [False, True])
def test_trial_directory_searches_root_and_agent_logs(tmp_path: Path, nested: bool) -> None:
    trial_dir = tmp_path / ("nested" if nested else "root")
    path = trial_dir / "opencode.txt"
    if nested:
        path = trial_dir / "agent" / "opencode.txt"
    _write_jsonl(path, {"type": "error", "error": {"message": "terminal failure"}})

    evidence = first_trial_agent_failure(trial_dir)

    assert evidence is not None
    assert evidence.event_type == "error"
    assert evidence.source == str(path)


@pytest.mark.parametrize("nested", [False, True], ids=["step-root", "step-agent"])
def test_trial_directory_searches_multistep_step_logs(tmp_path: Path, nested: bool) -> None:
    """Harbor multi-step layouts keep agent logs under steps/<step>/(agent/)."""
    trial_dir = tmp_path / "trial"
    parent = trial_dir / "steps" / "s1"
    if nested:
        parent = parent / "agent"
    path = parent / "claude-code.txt"
    _write_jsonl(path, {"type": "result", "is_error": True, "result": "step terminal"})

    evidence = first_trial_agent_failure(trial_dir)

    assert evidence is not None
    assert evidence.event_type == "result"
    assert evidence.message == "step terminal"
    assert evidence.source == str(path)


def test_trial_directory_skips_benign_log_before_failure(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    _write_jsonl(
        trial_dir / "claude-code.txt",
        {"type": "result", "is_error": False, "result": "unauthorized docs"},
    )
    failing_path = trial_dir / "agent" / "codex.txt"
    _write_jsonl(failing_path, {"type": "turn.failed", "error": {"message": "terminal failure"}})

    evidence = first_trial_agent_failure(trial_dir)

    assert evidence is not None
    assert evidence.event_type == "turn.failed"
    assert evidence.source == str(failing_path)


def test_zero_token_terminal_trajectory_api_error_is_failure(tmp_path: Path) -> None:
    """Supported adapters without native log schemas retain a structured backstop."""
    trial_dir = tmp_path / "trial"
    trajectory = trial_dir / "agent" / "trajectory.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(
        json.dumps(
            {
                "steps": [{"source": "agent", "message": "API Error: 401 Unauthorized"}],
                "final_metrics": {
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = first_trial_agent_failure(trial_dir)

    assert evidence is not None
    assert evidence.event_type == "trajectory.zero_token_api_error"
    assert evidence.message == "API Error: 401 Unauthorized"
    assert evidence.source == str(trajectory)


def test_trajectory_failure_cap_is_independent_from_jsonl_line_cap(tmp_path: Path, monkeypatch) -> None:
    trial_dir = tmp_path / "trial"
    trajectory = trial_dir / "agent" / "trajectory.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "source": "agent",
                        "message": "API Error: 401 Unauthorized " + "x" * 256,
                    }
                ],
                "final_metrics": {
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(failure_evidence, "_MAX_EVENT_LINE_BYTES", 64)

    assert first_trial_agent_failure(trial_dir) is not None


@pytest.mark.parametrize(
    "payload",
    [
        {
            "steps": [
                {"source": "agent", "message": "API Error: 401 Unauthorized"},
                {"source": "agent", "message": "Recovered and completed"},
            ],
            "final_metrics": {"total_prompt_tokens": 0, "total_completion_tokens": 0},
        },
        {
            "steps": [{"source": "agent", "message": "API Error: 401 Unauthorized"}],
            "final_metrics": {"total_prompt_tokens": 1, "total_completion_tokens": 0},
        },
        {
            "steps": [{"message": "API Error: 401 Unauthorized"}],
            "final_metrics": {"total_prompt_tokens": 0, "total_completion_tokens": 0},
        },
        {
            "steps": [
                {
                    "source": "agent",
                    "message": "Documentation example: API Error: 401 Unauthorized",
                }
            ],
            "final_metrics": {"total_prompt_tokens": 0, "total_completion_tokens": 0},
        },
    ],
    ids=["recovered", "used-tokens", "untyped-source", "documentary-prefix"],
)
def test_terminal_trajectory_api_error_backstop_rejects_ambiguous_prose(tmp_path: Path, payload: dict) -> None:
    trial_dir = tmp_path / "trial"
    trajectory = trial_dir / "agent" / "trajectory.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(json.dumps(payload), encoding="utf-8")

    assert first_trial_agent_failure(trial_dir) is None


def test_terminal_message_is_redacted_and_bounded() -> None:
    secret = f"{'sk'}-abcdefghijklmnop"
    evidence = failure_from_agent_event(
        "codex",
        {
            "type": "turn.failed",
            "error": {"message": f"OPENAI_API_KEY={secret} " + "x" * 5000},
        },
        "codex.txt",
    )

    assert evidence is not None
    assert secret not in evidence.message
    assert "<redacted>" in evidence.message
    assert len(evidence.message) <= 2048
    assert evidence.message.endswith("...<truncated>")


def test_two_mib_terminal_event_line_is_detected(tmp_path: Path) -> None:
    """Regression: a ~2MiB terminal event line must yield evidence.

    The previous 1MiB streaming line bound silently skipped it, so a genuine
    claude-code ``result``/``is_error: true`` with a large payload produced no
    evidence and a zero reward shell aggregated as valid.
    """
    path = tmp_path / "claude-code.txt"
    event = {
        "type": "result",
        "is_error": True,
        "result": "API Error: 401 " + "x" * (2 * 1024 * 1024),
    }
    _write_jsonl(path, {"type": "message", "content": "benign"}, event)

    evidence = first_agent_log_failure(path)

    assert evidence is not None
    assert evidence.event_type == "result"
    assert evidence.message.startswith("API Error: 401")
    assert len(evidence.message) <= 2048


def test_line_over_bound_is_stream_skipped_and_later_terminal_event_found(tmp_path: Path) -> None:
    """A single line above the bound is skipped without loading it into memory.

    This is the documented residual gap: a terminal event on a >10MiB line is
    invisible to the parser (Harbor's result.json typed-exception backstop
    remains the authoritative catch); a later normal terminal event must still
    be found, and skipping must stream rather than materialize the line.
    """
    import tracemalloc

    from skillevaluator.tier3.harbor.failure_evidence import _MAX_EVENT_LINE_BYTES

    giant_payload_bytes = 2 * _MAX_EVENT_LINE_BYTES
    path = tmp_path / "claude-code.txt"
    with path.open("wb") as stream:
        stream.write(b'{"type": "result", "is_error": true, "result": "giant ')
        stream.write(b"x" * giant_payload_bytes)
        stream.write(b'"}\n')
        stream.write(
            json.dumps({"type": "result", "is_error": True, "result": "second terminal"}).encode("utf-8") + b"\n"
        )

    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        evidence = first_agent_log_failure(path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert evidence is not None
    assert evidence.message == "second terminal"
    # Streaming skip: peak allocation stays near the per-read chunk bound
    # (empirically ~3 chunks) and far below the giant line itself.
    assert peak - baseline < 4 * _MAX_EVENT_LINE_BYTES


def test_agent_log_scan_stops_at_aggregate_byte_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(failure_evidence, "_MAX_EVENT_LOG_BYTES", 256)
    path = tmp_path / "claude-code.txt"
    path.write_text(
        json.dumps({"type": "message", "content": "x" * 180})
        + "\n"
        + json.dumps({"type": "result", "is_error": True, "result": "too late"})
        + "\n",
        encoding="utf-8",
    )

    assert first_agent_log_failure(path) is None


def test_huge_integer_literal_line_does_not_crash_scan(tmp_path: Path) -> None:
    """A >4300-digit integer literal raises plain ValueError from json.loads.

    That is an agent-writable transcript line; it must be treated as an
    unparseable line, not crash the whole evaluation, and a later terminal
    event must still be found.
    """
    path = tmp_path / "claude-code.txt"
    path.write_text(
        "9" * 5000 + "\n" + json.dumps({"type": "result", "is_error": True, "result": "terminal"}) + "\n",
        encoding="utf-8",
    )

    evidence = first_agent_log_failure(path)

    assert evidence is not None
    assert evidence.message == "terminal"


def test_utf8_bom_on_first_line_terminal_event_classifies(tmp_path: Path) -> None:
    """A UTF-8 BOM must not make a first-line terminal event unparseable."""
    path = tmp_path / "claude-code.txt"
    payload = json.dumps({"type": "result", "is_error": True, "result": "API Error: 401"})
    path.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8") + b"\n")

    evidence = first_agent_log_failure(path)

    assert evidence is not None
    assert evidence.event_type == "result"
    assert evidence.message == "API Error: 401"


def test_terminal_message_with_bare_github_token_is_redacted() -> None:
    """Attacker-influenced transcript text must not leak bare tokens into evidence."""
    token = f"{'ghp'}_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef7890"
    evidence = failure_from_agent_event(
        "claude-code",
        {
            "type": "result",
            "is_error": True,
            "result": f"git push failed for {token} during the task",
        },
        "claude-code.txt",
    )

    assert evidence is not None
    assert token not in evidence.message
    assert "ghp_<redacted>" in evidence.message


def test_terminal_message_redacts_strong_key_glued_into_identifier() -> None:
    token = f"{'sk'}-Abcdefghijklmnop12345678"
    evidence = failure_from_agent_event(
        "codex",
        {
            "type": "turn.failed",
            "error": {"message": f"trial-case-{token} failed"},
        },
        "codex.txt",
    )

    assert evidence is not None
    assert token not in evidence.message
    assert "sk-<redacted>" in evidence.message


def test_failure_evidence_is_frozen() -> None:
    evidence = FailureEvidence(
        component="agent",
        phase="execute",
        kind="agent_runtime",
        source="codex.txt",
        event_type="turn.failed",
        message="failed",
    )

    with pytest.raises(AttributeError):
        evidence.message = "changed"  # type: ignore[misc]
