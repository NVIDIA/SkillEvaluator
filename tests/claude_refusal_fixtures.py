# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code stream-json fixtures for a model-provider safety refusal.

They follow the event shapes Claude Code 2.1.292 writes when the provider's
safety classifier stops the selected model (refusal category ``cyber``):

- with the refusal model fallback on, behind a gateway that serves only the
  selected model: the CLI retries on a bare Anthropic id and gets a 403;
- with ``CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1``: the CLI reports
  ``model_refusal_no_fallback`` and stops.

Session and message ids are placeholders; nothing here came from a credential.
"""

from __future__ import annotations

import json

SELECTED_MODEL = "claude-opus-5-5"
FALLBACK_MODEL = "claude-opus-4-8"

_SESSION = "00000000-0000-4000-8000-000000000000"
GATEWAY_FALLBACK_403 = (
    f"Failed to authenticate. API Error: 403 Model '{FALLBACK_MODEL}' is not available on this gateway route."
)
NO_FALLBACK_TEXT = (
    "API Error: Opus 5.5's safeguards flagged this message (https://www.anthropic.com/legal/aup). "
    "Try rephrasing the request or starting a new session."
)


def line(event: dict) -> str:
    return json.dumps({**event, "session_id": _SESSION}, separators=(",", ":"))


def init_event() -> str:
    return line(
        {
            "type": "system",
            "subtype": "init",
            "model": SELECTED_MODEL,
            "permissionMode": "bypassPermissions",
            "claude_code_version": "2.1.292",
            "apiKeySource": "ANTHROPIC_API_KEY",
        }
    )


def tool_round(output: str) -> list[str]:
    return [
        line(
            {
                "type": "assistant",
                "message": {
                    "model": SELECTED_MODEL,
                    "role": "assistant",
                    "stop_reason": "tool_use",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}],
                },
            }
        ),
        line(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": output}],
                },
            }
        ),
    ]


def result_event(text: str, *, status: int | None = None) -> str:
    return line(
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "terminal_reason": "api_error",
            **({"api_error_status": status} if status is not None else {}),
            "safety_stops": 2,
            "result": text,
        }
    )


def synthetic_frame(text: str, *, stop_reason: str, error: str | None = None) -> str:
    return line(
        {
            "type": "assistant",
            "message": {
                "model": "<synthetic>",
                "role": "assistant",
                "stop_reason": stop_reason,
                "stop_details": {"category": "cyber"} if stop_reason == "refusal" else None,
                "content": [{"type": "text", "text": text}],
            },
            **({"error": error} if error else {}),
            "is_api_error_message": True,
        }
    )


def gateway_fallback_403_stream() -> str:
    """Fallback on: the CLI swapped models after a cyber refusal, and the gateway answered 403."""
    return "\n".join(
        [
            init_event(),
            *tool_round("README.md\nbuild.sh"),
            line(
                {
                    "type": "system",
                    "subtype": "informational",
                    "level": "notice",
                    "content": "Opus 5.5's safeguards stopped the response above · continuing once with that noted",
                }
            ),
            line(
                {
                    "type": "system",
                    "subtype": "model_refusal_fallback",
                    "trigger": "refusal",
                    "direction": "retry",
                    "scope": "session",
                    "original_model": SELECTED_MODEL,
                    "fallback_model": FALLBACK_MODEL,
                    "request_id": None,
                    "api_refusal_category": "cyber",
                    "content": "Opus 5.5's safeguards flagged this session. Opus 4.8 is answering instead.",
                }
            ),
            synthetic_frame(GATEWAY_FALLBACK_403, stop_reason="stop_sequence", error="authentication_failed"),
            result_event(GATEWAY_FALLBACK_403, status=403),
        ]
    )


def no_fallback_stream() -> str:
    """The same refusal with CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1."""
    return "\n".join(
        [
            init_event(),
            *tool_round("README.md\nbuild.sh"),
            line(
                {
                    "type": "system",
                    "subtype": "model_refusal_no_fallback",
                    "level": "warning",
                    "original_model": SELECTED_MODEL,
                    "request_id": None,
                    "api_refusal_category": "cyber",
                    "content": "Opus 5.5's safeguards flagged this message.",
                }
            ),
            synthetic_frame(NO_FALLBACK_TEXT, stop_reason="refusal"),
            result_event(NO_FALLBACK_TEXT),
        ]
    )
