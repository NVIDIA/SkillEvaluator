# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-valid ATIF message parts must reach judge evidence and references."""

from __future__ import annotations

import json
from types import ModuleType

import pytest

from skillevaluator.tier3.eval_core import atif_helpers


def _trajectory(messages: list) -> dict:
    from harbor.models.trajectories.trajectory import Trajectory

    return Trajectory.model_validate(
        {
            "schema_version": "ATIF-v1.8",
            "agent": {"name": "test-agent", "version": "test"},
            "steps": [
                {"step_id": index, "source": "agent", "message": message}
                for index, message in enumerate(messages, start=1)
            ],
        }
    ).to_json_dict()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("The answer is 4.", "The answer is 4."),
        ([{"type": "text", "text": "The answer is 4."}], "The answer is 4."),
        (
            [{"type": "text", "text": "First line."}, {"type": "text", "text": "Second line."}],
            "First line.\nSecond line.",
        ),
        ([], ""),
        ([{"type": "text", "text": "   "}], ""),
    ],
)
def test_message_content_reaches_all_text_extractors(
    harbor_eval_template: ModuleType,
    message: str | list,
    expected: str,
) -> None:
    trajectory = _trajectory([message])

    assert atif_helpers.get_agent_text(trajectory) == expected
    assert atif_helpers.get_final_response(trajectory) == expected
    assert harbor_eval_template.get_agent_text(trajectory) == expected
    assert harbor_eval_template._get_final_response(trajectory) == expected
    assert atif_helpers.build_conversation_summary(trajectory, "question") == (
        harbor_eval_template.build_conversation_summary(trajectory, "question")
    )


def test_last_nonempty_structured_answer_reaches_all_judges_and_refs(harbor_eval_template: ModuleType) -> None:
    trajectory = _trajectory(["Earlier answer.", [{"type": "text", "text": "The answer is 4."}], []])

    assert atif_helpers.get_final_response(trajectory) == "The answer is 4."
    assert harbor_eval_template._get_final_response(trajectory) == "The answer is 4."
    assert atif_helpers.get_agent_text(trajectory) == "Earlier answer.\nThe answer is 4."
    assert "Agent: The answer is 4." in atif_helpers.build_conversation_summary(trajectory, "What is 2+2?")
    shared = atif_helpers.build_metric_evidence_bundles(trajectory, "What is 2+2?", ground_truth="4")
    verifier = harbor_eval_template.build_metric_evidence_bundles(trajectory, "What is 2+2?", ground_truth="4")
    assert shared == verifier
    for bundle in shared.values():
        assert "The answer is 4." in bundle["prompt_evidence"]
        final_refs = [ref for ref in bundle["evidence_refs"] if ref["kind"] == "final_response"]
        assert len(final_refs) == 1
        assert final_refs[0]["json_pointer"] == "/steps/1"
        assert final_refs[0]["excerpt"] == "The answer is 4."


@pytest.mark.parametrize("media_type", ("image", "audio"))
def test_media_only_followup_does_not_override_text_answer(
    harbor_eval_template: ModuleType, media_type: str
) -> None:
    trajectory = _trajectory(
        [
            "The answer is 4.",
            [{"type": media_type, "source": {
                "media_type": "image/png" if media_type == "image" else "audio/wav",
                "path": "/private/media",
            }}],
        ]
    )
    assert atif_helpers.get_final_response(trajectory) == "The answer is 4."
    assert harbor_eval_template._get_final_response(trajectory) == "The answer is 4."
    shared = atif_helpers.build_metric_evidence_bundles(trajectory, "What is 2+2?", ground_truth="4")
    verifier = harbor_eval_template.build_metric_evidence_bundles(trajectory, "What is 2+2?", ground_truth="4")
    assert shared == verifier
    for metric in ("accuracy", "goal_accuracy"):
        bundle = shared[metric]
        assert "FINAL RESPONSE\nThe answer is 4." in bundle["prompt_evidence"]
        refs = [ref for ref in bundle["evidence_refs"] if ref["kind"] == "final_response"]
        assert len(refs) == 1
        assert refs[0]["json_pointer"] == "/steps/0"
        assert refs[0]["excerpt"] == "The answer is 4."
    assert "/private/media" not in json.dumps(shared)


def test_media_only_step_is_not_selected_as_final_text_response(harbor_eval_template: ModuleType) -> None:
    trajectory = _trajectory([[{
        "type": "audio", "source": {"media_type": "audio/wav", "path": "/private/media"}
    }]])
    assert atif_helpers.get_final_response(trajectory) == ""
    assert harbor_eval_template._get_final_response(trajectory) == ""
    assert atif_helpers._final_response_ref(trajectory) == []
    assert harbor_eval_template._final_response_ref(trajectory) == []


def test_multimodal_answer_uses_markers_without_media_paths(harbor_eval_template: ModuleType) -> None:
    trajectory = _trajectory(
        [
            [
                {"type": "text", "text": "Result attached."},
                {"type": "image", "source": {"media_type": "image/png", "path": "/private/image-result.png"}},
                {"type": "audio", "source": {"media_type": "audio/wav", "path": "/private/audio-result.wav"}},
            ]
        ]
    )
    expected = "Result attached.\n[image]\n[audio]"

    assert atif_helpers.get_final_response(trajectory) == expected
    assert harbor_eval_template._get_final_response(trajectory) == expected
    shared = atif_helpers.build_metric_evidence_bundles(trajectory, "question")
    verifier = harbor_eval_template.build_metric_evidence_bundles(trajectory, "question")
    assert shared == verifier
    for bundle in shared.values():
        assert expected in bundle["prompt_evidence"]
        assert "/private/" not in json.dumps(bundle)
