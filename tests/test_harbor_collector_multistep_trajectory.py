# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor 0.22 native multi-step trajectory merge regressions."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
from harbor.models.trajectories import Trajectory

from skillevaluator.tier3.harbor import report_data
from skillevaluator.tier3.harbor.collector import (
    COLLECTED_REWARD_JSON_MAX_BYTES,
    _materialize_trajectory_file,
    _merged_step_trajectory,
    _redacted_artifact_text,
    _save_trials,
    _save_unscored_trials,
)
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS


def _trajectory(
    session_id: str,
    labels: tuple[str, ...],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    cost_usd: float,
) -> dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": session_id,
        "trajectory_id": f"trajectory-{session_id}",
        "agent": {"name": "codex", "version": "test"},
        "steps": [
            {
                "step_id": index,
                "source": "agent",
                "message": label,
                "tool_calls": [],
            }
            for index, label in enumerate(labels, start=1)
        ],
        "final_metrics": {
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_cached_tokens": 0,
            "total_cost_usd": cost_usd,
            "extra": {
                "reasoning_output_tokens": reasoning_tokens,
                "finish_reason": "stop",
            },
        },
    }


def _write_multistep_trial(
    trial_root: Path,
    trajectories: tuple[tuple[str, dict[str, Any]], ...],
    *,
    resume_trajectory: bool,
    load_trajectory: str | None = None,
) -> None:
    agent_config: dict[str, Any] = {"resume_trajectory": resume_trajectory}
    if load_trajectory is not None:
        agent_config["load_trajectory"] = load_trajectory
    (trial_root / "result.json").parent.mkdir(parents=True, exist_ok=True)
    (trial_root / "result.json").write_text(
        json.dumps(
            {
                "config": {"agent": agent_config},
                "step_results": [{"step_name": name} for name, _ in trajectories],
            }
        ),
        encoding="utf-8",
    )
    for step_name, trajectory in trajectories:
        agent_dir = trial_root / "steps" / step_name / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")


def test_merge_resumed_multistep_trajectory_appends_only_cumulative_suffix(
    tmp_path: Path,
) -> None:
    first = _trajectory(
        "session-1",
        ("loaded", "one"),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.125,
    )
    second = _trajectory(
        "session-1",
        ("loaded", "one", "two"),
        prompt_tokens=25,
        completion_tokens=5,
        reasoning_tokens=4,
        cost_usd=0.25,
    )
    for copied in second["steps"][:2]:
        copied["is_copied_context"] = True
    _write_multistep_trial(
        tmp_path,
        (
            ("prepare", first),
            ("finish", second),
        ),
        resume_trajectory=True,
        load_trajectory="/seed/trajectory.json",
    )

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    Trajectory.model_validate(merged)
    assert [step["message"] for step in merged["steps"]] == ["loaded", "one", "two"]
    assert [step["step_id"] for step in merged["steps"]] == [1, 2, 3]
    assert [step["extra"]["harbor_step_name"] for step in merged["steps"]] == [
        "prepare",
        "prepare",
        "finish",
    ]
    assert merged["final_metrics"] == {
        "total_prompt_tokens": 25,
        "total_completion_tokens": 5,
        "total_cached_tokens": 0,
        "total_cost_usd": 0.25,
        "total_steps": 3,
        "extra": {
            "reasoning_output_tokens": 4,
            "finish_reason": "stop",
            "harbor_multi_step": True,
        },
    }


def test_merge_resumed_multistep_accepts_retained_copied_context_suffix(
    tmp_path: Path,
) -> None:
    first = _trajectory(
        "session-1",
        ("one", "two", "three"),
        prompt_tokens=30,
        completion_tokens=6,
        reasoning_tokens=3,
        cost_usd=0.3,
    )
    second = _trajectory(
        "session-1",
        ("two", "three", "four"),
        prompt_tokens=40,
        completion_tokens=8,
        reasoning_tokens=4,
        cost_usd=0.4,
    )
    for copied in second["steps"][:2]:
        copied["is_copied_context"] = True
    _write_multistep_trial(
        tmp_path,
        (("prepare", first), ("finish", second)),
        resume_trajectory=True,
    )

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    Trajectory.model_validate(merged)
    assert [step["message"] for step in merged["steps"]] == ["one", "two", "three", "four"]
    assert merged["final_metrics"]["total_prompt_tokens"] == 40


def test_merge_resumed_multistep_anchors_duplicate_copied_context_to_terminal_suffix(
    tmp_path: Path,
) -> None:
    first = _trajectory(
        "session-1",
        ("x", "a", "b", "a", "b"),
        prompt_tokens=30,
        completion_tokens=6,
        reasoning_tokens=3,
        cost_usd=0.3,
    )
    second = _trajectory(
        "session-1",
        ("a", "b", "c"),
        prompt_tokens=40,
        completion_tokens=8,
        reasoning_tokens=4,
        cost_usd=0.4,
    )
    for copied in second["steps"][:2]:
        copied["is_copied_context"] = True
    _write_multistep_trial(tmp_path, (("prepare", first), ("finish", second)), resume_trajectory=True)

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    assert [step["message"] for step in merged["steps"]] == ["x", "a", "b", "a", "b", "c"]


@pytest.mark.parametrize(
    ("previous_labels", "current_labels"),
    [
        (("a", "b", "x", "a"), ("a", "b", "c")),
        (("a", "b", "c"), ("c", "b", "d")),
        (("a", "b"), ("a", "b")),
    ],
    ids=("earlier-only-match", "reordered-suffix", "all-copied-no-new-step"),
)
def test_merge_resumed_multistep_rejects_ambiguous_or_empty_copied_context(
    tmp_path: Path,
    previous_labels: tuple[str, ...],
    current_labels: tuple[str, ...],
) -> None:
    first = _trajectory(
        "session-1",
        previous_labels,
        prompt_tokens=30,
        completion_tokens=6,
        reasoning_tokens=3,
        cost_usd=0.3,
    )
    second = _trajectory(
        "session-1",
        current_labels,
        prompt_tokens=40,
        completion_tokens=8,
        reasoning_tokens=4,
        cost_usd=0.4,
    )
    copied_count = 2
    for copied in second["steps"][:copied_count]:
        copied["is_copied_context"] = True
    _write_multistep_trial(tmp_path, (("prepare", first), ("finish", second)), resume_trajectory=True)

    assert _merged_step_trajectory(tmp_path) is None


def test_merge_copilot_placeholder_session_does_not_deduplicate_independent_fragments(
    tmp_path: Path,
) -> None:
    first = _trajectory(
        "copilot-cli",
        ("same output",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    second = _trajectory(
        "copilot-cli",
        ("same output",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    first["agent"]["name"] = "copilot"
    second["agent"]["name"] = "copilot"
    _write_multistep_trial(
        tmp_path,
        (("one", first), ("two", second)),
        resume_trajectory=True,
    )

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    Trajectory.model_validate(merged)
    assert [step["message"] for step in merged["steps"]] == ["same output", "same output"]
    assert merged["final_metrics"]["total_prompt_tokens"] == 20


def test_merge_unmarked_unique_session_strict_prefix_remains_cumulative(tmp_path: Path) -> None:
    first = _trajectory(
        "unique-session",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    second = _trajectory(
        "unique-session",
        ("one", "two"),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    _write_multistep_trial(tmp_path, (("one", first), ("two", second)), resume_trajectory=True)

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    assert [step["message"] for step in merged["steps"]] == ["one", "two"]
    assert merged["final_metrics"]["total_prompt_tokens"] == 20


def test_merge_fails_closed_for_mismatched_cross_step_copied_context(tmp_path: Path) -> None:
    first = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    second = _trajectory(
        "session-1",
        ("different", "two"),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    second["steps"][0]["is_copied_context"] = True
    _write_multistep_trial(tmp_path, (("one", first), ("two", second)), resume_trajectory=True)

    assert _merged_step_trajectory(tmp_path) is None


def test_merge_nonresumed_multistep_trajectory_keeps_independent_fragments(
    tmp_path: Path,
) -> None:
    _write_multistep_trial(
        tmp_path,
        (
            (
                "prepare",
                _trajectory(
                    "session-1",
                    ("one",),
                    prompt_tokens=10,
                    completion_tokens=2,
                    reasoning_tokens=1,
                    cost_usd=0.125,
                ),
            ),
            (
                "finish",
                _trajectory(
                    "session-2",
                    ("two",),
                    prompt_tokens=25,
                    completion_tokens=5,
                    reasoning_tokens=4,
                    cost_usd=0.25,
                ),
            ),
        ),
        resume_trajectory=False,
    )

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    Trajectory.model_validate(merged)
    assert [step["message"] for step in merged["steps"]] == ["one", "two"]
    assert merged["final_metrics"]["total_prompt_tokens"] == 35
    assert merged["final_metrics"]["total_completion_tokens"] == 7
    assert merged["final_metrics"]["total_cost_usd"] == 0.375
    assert merged["final_metrics"]["extra"]["reasoning_output_tokens"] == 5


def test_merge_resume_flag_sums_fragments_when_session_prefix_is_not_cumulative(
    tmp_path: Path,
) -> None:
    _write_multistep_trial(
        tmp_path,
        (
            (
                "prepare",
                _trajectory(
                    "session-1",
                    ("one",),
                    prompt_tokens=10,
                    completion_tokens=2,
                    reasoning_tokens=1,
                    cost_usd=0.125,
                ),
            ),
            (
                "finish",
                _trajectory(
                    "session-1",
                    ("different",),
                    prompt_tokens=25,
                    completion_tokens=5,
                    reasoning_tokens=4,
                    cost_usd=0.25,
                ),
            ),
        ),
        resume_trajectory=True,
    )

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    Trajectory.model_validate(merged)
    assert [step["message"] for step in merged["steps"]] == ["one", "different"]
    assert merged["final_metrics"]["total_prompt_tokens"] == 35
    assert merged["final_metrics"]["total_completion_tokens"] == 7
    assert merged["final_metrics"]["total_cost_usd"] == 0.375
    assert merged["final_metrics"]["extra"]["reasoning_output_tokens"] == 5


def test_merge_resumed_copied_context_prefix_ignores_marker_metric_and_note_changes(tmp_path: Path) -> None:
    first = _trajectory(
        "session-1",
        ("one", "two"),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.125,
    )
    first["steps"][0]["metrics"] = {"prompt_tokens": 4, "completion_tokens": 1}
    second = _trajectory(
        "session-1",
        ("one", "two", "three"),
        prompt_tokens=25,
        completion_tokens=5,
        reasoning_tokens=4,
        cost_usd=0.25,
    )
    for copied in second["steps"][:2]:
        copied["is_copied_context"] = True
        copied.pop("metrics", None)
        copied["extra"] = {"note": "Copied context; metrics already recorded"}
    _write_multistep_trial(
        tmp_path,
        (("prepare", first), ("finish", second)),
        resume_trajectory=True,
    )

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    Trajectory.model_validate(merged)
    assert [step["message"] for step in merged["steps"]] == ["one", "two", "three"]
    assert merged["final_metrics"]["total_prompt_tokens"] == 25


def test_merge_fails_closed_when_result_lists_a_missing_fragment(tmp_path: Path) -> None:
    _write_multistep_trial(
        tmp_path,
        (
            (
                "one",
                _trajectory(
                    "session-1",
                    ("one",),
                    prompt_tokens=10,
                    completion_tokens=2,
                    reasoning_tokens=1,
                    cost_usd=0.1,
                ),
            ),
        ),
        resume_trajectory=False,
    )
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    result["step_results"].append({"step_name": "missing"})
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")

    assert _merged_step_trajectory(tmp_path) is None


def test_merge_fails_closed_for_invalid_atif_fragment(tmp_path: Path) -> None:
    invalid = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    invalid["steps"].append("not-a-step")
    _write_multistep_trial(tmp_path, (("one", invalid),), resume_trajectory=False)

    assert _merged_step_trajectory(tmp_path) is None


def test_merge_materializes_continuation_and_external_subagent_refs(tmp_path: Path) -> None:
    root = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    root["notes"] = "root note"
    root["extra"] = {"root": True}
    root["continued_trajectory_ref"] = "trajectory.cont-1.json"
    continuation = _trajectory(
        "continuation-session-1",
        ("one", "two"),
        prompt_tokens=25,
        completion_tokens=5,
        reasoning_tokens=4,
        cost_usd=0.25,
    )
    continuation["trajectory_id"] = "continuation-1"
    continuation["notes"] = "continuation note"
    continuation["extra"] = {"continuation": True}
    continuation["steps"][0]["is_copied_context"] = True
    continuation["steps"][1]["observation"] = {
        "results": [
            {
                "content": "delegated",
                "subagent_trajectory_ref": [{"trajectory_path": "trajectory.subagent.json"}],
            }
        ]
    }
    subagent = _trajectory(
        "subagent-session",
        ("subagent",),
        prompt_tokens=3,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.01,
    )
    subagent.pop("trajectory_id")
    _write_multistep_trial(tmp_path, (("only", root),), resume_trajectory=False)
    agent_dir = tmp_path / "steps" / "only" / "agent"
    (agent_dir / "trajectory.cont-1.json").write_text(json.dumps(continuation), encoding="utf-8")
    (agent_dir / "trajectory.subagent.json").write_text(json.dumps(subagent), encoding="utf-8")

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    Trajectory.model_validate(merged)
    assert [step["message"] for step in merged["steps"]] == ["one", "two"]
    assert merged["final_metrics"]["total_prompt_tokens"] == 25
    assert "continued_trajectory_ref" not in merged
    assert "root note" in merged["notes"]
    assert "continuation note" in merged["notes"]
    embedded_id = merged["subagent_trajectories"][0]["trajectory_id"]
    assert embedded_id.startswith("skillevaluator-scoped-subagent-")
    ref = merged["steps"][1]["observation"]["results"][0]["subagent_trajectory_ref"][0]
    assert ref["trajectory_id"] == embedded_id
    assert "trajectory_path" not in ref
    source = merged["extra"]["harbor_multi_step"]["source_trajectories"][0]
    assert source["trajectory_id"].startswith("skillevaluator-continuation-")


def test_merge_recursively_materializes_refs_inside_embedded_subagent(tmp_path: Path) -> None:
    root = _trajectory(
        "root-session",
        ("delegate",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    root["steps"][0]["observation"] = {
        "results": [
            {
                "content": "child",
                "subagent_trajectory_ref": [{"trajectory_id": "embedded-child"}],
            }
        ]
    }
    child = _trajectory(
        "child-session",
        ("child root",),
        prompt_tokens=3,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.01,
    )
    child["trajectory_id"] = "embedded-child"
    child["continued_trajectory_ref"] = "trajectory.child-cont.json"
    root["subagent_trajectories"] = [child]

    child_continuation = _trajectory(
        "child-session",
        ("child root", "nested delegate"),
        prompt_tokens=6,
        completion_tokens=2,
        reasoning_tokens=0,
        cost_usd=0.02,
    )
    child_continuation["trajectory_id"] = "child-continuation"
    child_continuation["steps"][0]["is_copied_context"] = True
    child_continuation["steps"][1]["observation"] = {
        "results": [
            {
                "content": "grandchild",
                "subagent_trajectory_ref": [{"trajectory_path": "trajectory.grandchild.json"}],
            }
        ]
    }
    grandchild = _trajectory(
        "grandchild-session",
        ("grandchild",),
        prompt_tokens=1,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.001,
    )
    grandchild.pop("trajectory_id")

    _write_multistep_trial(tmp_path, (("only", root),), resume_trajectory=False)
    agent_dir = tmp_path / "steps" / "only" / "agent"
    (agent_dir / "trajectory.child-cont.json").write_text(
        json.dumps(child_continuation), encoding="utf-8"
    )
    (agent_dir / "trajectory.grandchild.json").write_text(json.dumps(grandchild), encoding="utf-8")

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    Trajectory.model_validate(merged)
    embedded_child = merged["subagent_trajectories"][0]
    assert embedded_child["trajectory_id"].startswith("skillevaluator-scoped-subagent-")
    assert embedded_child["extra"]["harbor_continuation"]["segment_count"] == 2
    assert "continued_trajectory_ref" not in embedded_child
    assert [step["message"] for step in embedded_child["steps"]] == ["child root", "nested delegate"]
    nested_ref = embedded_child["steps"][1]["observation"]["results"][0][
        "subagent_trajectory_ref"
    ][0]
    assert "trajectory_path" not in nested_ref
    assert nested_ref["trajectory_id"] == embedded_child["subagent_trajectories"][0]["trajectory_id"]


def test_merge_fails_closed_for_continuation_cycle(tmp_path: Path) -> None:
    root = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    root["continued_trajectory_ref"] = "trajectory.cont-1.json"
    continuation = _trajectory(
        "session-1",
        ("two",),
        prompt_tokens=25,
        completion_tokens=5,
        reasoning_tokens=4,
        cost_usd=0.25,
    )
    continuation["continued_trajectory_ref"] = "trajectory.json"
    _write_multistep_trial(tmp_path, (("only", root),), resume_trajectory=False)
    agent_dir = tmp_path / "steps" / "only" / "agent"
    (agent_dir / "trajectory.cont-1.json").write_text(json.dumps(continuation), encoding="utf-8")

    assert _merged_step_trajectory(tmp_path) is None


def test_materialize_preserves_exact_whitespace_in_continuation_source_name(tmp_path: Path) -> None:
    root = _trajectory(
        "session-1",
        ("root",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    root["continued_trajectory_ref"] = "continuation.json "
    exact = _trajectory(
        "session-1",
        ("exact continuation",),
        prompt_tokens=5,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.05,
    )
    wrong = _trajectory(
        "session-1",
        ("wrong clean-name continuation",),
        prompt_tokens=500,
        completion_tokens=100,
        reasoning_tokens=50,
        cost_usd=5.0,
    )
    (tmp_path / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")
    (tmp_path / "continuation.json ").write_text(json.dumps(exact), encoding="utf-8")
    (tmp_path / "continuation.json").write_text(json.dumps(wrong), encoding="utf-8")

    materialized, _reference_key = _materialize_trajectory_file(tmp_path, "trajectory.json")

    messages = [step["message"] for step in materialized["steps"]]
    assert "exact continuation" in messages
    assert "wrong clean-name continuation" not in messages


@pytest.mark.skipif(os.name == "nt", reason="backslash is a separator on Windows, matching Harbor")
def test_materialize_preserves_literal_posix_backslash_in_continuation_source_name(tmp_path: Path) -> None:
    root = _trajectory(
        "session-1",
        ("root",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    reference = r"cont\next.json"
    root["continued_trajectory_ref"] = reference
    literal = _trajectory(
        "session-1",
        ("harbor literal",),
        prompt_tokens=5,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.05,
    )
    rewritten = _trajectory(
        "session-1",
        ("collector rewritten",),
        prompt_tokens=500,
        completion_tokens=100,
        reasoning_tokens=50,
        cost_usd=5.0,
    )
    (tmp_path / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")
    (tmp_path / reference).write_text(json.dumps(literal), encoding="utf-8")
    rewritten_path = tmp_path / "cont" / "next.json"
    rewritten_path.parent.mkdir()
    rewritten_path.write_text(json.dumps(rewritten), encoding="utf-8")

    materialized, _reference_key = _materialize_trajectory_file(tmp_path, "trajectory.json")

    messages = [step["message"] for step in materialized["steps"]]
    assert "harbor literal" in messages
    assert "collector rewritten" not in messages


def test_merge_trusts_explicit_continuation_copied_context_markers(tmp_path: Path) -> None:
    root = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    root["continued_trajectory_ref"] = "trajectory.cont-1.json"
    continuation = _trajectory(
        "session-1",
        ("different", "two"),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    continuation["steps"][0]["is_copied_context"] = True
    _write_multistep_trial(tmp_path, (("only", root),), resume_trajectory=False)
    (tmp_path / "steps" / "only" / "agent" / "trajectory.cont-1.json").write_text(
        json.dumps(continuation), encoding="utf-8"
    )

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    assert [step["message"] for step in merged["steps"]] == ["one", "two"]


def test_materialize_flattens_continuation_provenance_and_reconciles_total_steps(tmp_path: Path) -> None:
    first = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    first["trajectory_id"] = "first"
    first["continued_trajectory_ref"] = "trajectory.cont-1.json"
    second = _trajectory(
        "session-1",
        ("one", "two"),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    second["trajectory_id"] = "second"
    second["steps"][0]["is_copied_context"] = True
    second["steps"][1]["observation"] = {
        "results": [
            {
                "content": "child",
                "subagent_trajectory_ref": [{"trajectory_path": "trajectory.child.json"}],
            }
        ]
    }
    second["continued_trajectory_ref"] = "trajectory.cont-2.json"
    third = _trajectory(
        "session-1",
        ("two", "three"),
        prompt_tokens=30,
        completion_tokens=6,
        reasoning_tokens=3,
        cost_usd=0.3,
    )
    third["trajectory_id"] = "third"
    third["steps"][0]["is_copied_context"] = True
    third["final_metrics"]["total_steps"] = 2
    child = _trajectory(
        "child-session",
        ("child",),
        prompt_tokens=1,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.01,
    )
    child.pop("trajectory_id")
    for name, trajectory in (
        ("trajectory.json", first),
        ("trajectory.cont-1.json", second),
        ("trajectory.cont-2.json", third),
    ):
        (tmp_path / name).write_text(json.dumps(trajectory), encoding="utf-8")
    (tmp_path / "trajectory.child.json").write_text(json.dumps(child), encoding="utf-8")

    materialized, _ = _materialize_trajectory_file(tmp_path, "trajectory.json")

    assert [step["message"] for step in materialized["steps"]] == ["one", "two", "three"]
    assert materialized["final_metrics"]["total_steps"] == 3
    provenance = materialized["extra"]["harbor_continuation"]
    assert provenance["segment_count"] == 3
    assert provenance["source_trajectory_ids"] == ["first", "second", "third"]
    assert len(materialized["subagent_trajectories"]) == 1


def test_materialize_prefers_embedded_subagent_when_ref_also_has_missing_path(tmp_path: Path) -> None:
    root = _trajectory(
        "root-session",
        ("delegate",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    child = _trajectory(
        "child-session",
        ("child",),
        prompt_tokens=1,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.01,
    )
    child["trajectory_id"] = "child"
    root["subagent_trajectories"] = [child]
    root["steps"][0]["observation"] = {
        "results": [
            {
                "content": "child",
                "subagent_trajectory_ref": [
                    {"trajectory_id": "child", "trajectory_path": "missing-sidecar.json"}
                ],
            }
        ]
    }
    (tmp_path / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")

    materialized, _ = _materialize_trajectory_file(tmp_path, "trajectory.json")

    ref = materialized["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"][0]
    assert ref == {"trajectory_id": "child"}


@pytest.mark.parametrize("dual_key_first", [True, False])
def test_materialize_reuses_embedded_alias_for_path_only_missing_sidecar(
    tmp_path: Path,
    dual_key_first: bool,
) -> None:
    root = _trajectory(
        "root-session",
        ("delegate",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    child = _trajectory(
        "child-session",
        ("child",),
        prompt_tokens=1,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.01,
    )
    child["trajectory_id"] = "child"
    root["subagent_trajectories"] = [child]
    dual_key = {"trajectory_id": "child", "trajectory_path": "missing-sidecar.json"}
    path_only = {"trajectory_path": "missing-sidecar.json"}
    root["steps"][0]["observation"] = {
        "results": [
            {
                "content": "child",
                "subagent_trajectory_ref": [dual_key, path_only]
                if dual_key_first
                else [path_only, dual_key],
            }
        ]
    }
    (tmp_path / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")

    materialized, _ = _materialize_trajectory_file(tmp_path, "trajectory.json")

    refs = materialized["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"]
    assert refs == [{"trajectory_id": "child"}, {"trajectory_id": "child"}]
    assert [item["trajectory_id"] for item in materialized["subagent_trajectories"]] == ["child"]


def test_materialize_rejects_conflicting_ids_for_same_external_subagent_file(tmp_path: Path) -> None:
    root = _trajectory(
        "root-session",
        ("first", "second"),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    for step, trajectory_id in zip(root["steps"], ("child-a", "child-b"), strict=True):
        step["observation"] = {
            "results": [
                {
                    "content": "child",
                    "subagent_trajectory_ref": [
                        {"trajectory_id": trajectory_id, "trajectory_path": "trajectory.child.json"}
                    ],
                }
            ]
        }
    child = _trajectory(
        "child-session",
        ("child",),
        prompt_tokens=1,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.01,
    )
    child.pop("trajectory_id")
    (tmp_path / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")
    (tmp_path / "trajectory.child.json").write_text(json.dumps(child), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting trajectory_id aliases"):
        _materialize_trajectory_file(tmp_path, "trajectory.json")


@pytest.mark.parametrize(
    "refs",
    [
        (
            {"trajectory_id": "child", "trajectory_path": "trajectory.child.json"},
            {"trajectory_path": "trajectory.child.json"},
        ),
        (
            {"trajectory_path": "trajectory.child.json"},
            {"trajectory_id": "child", "trajectory_path": "trajectory.child.json"},
        ),
    ],
)
def test_materialize_reuses_supplied_id_for_same_path_regardless_of_ref_order(
    tmp_path: Path,
    refs: tuple[dict[str, str], dict[str, str]],
) -> None:
    root = _trajectory(
        "root-session",
        ("first", "second"),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    for step, ref in zip(root["steps"], refs, strict=True):
        step["observation"] = {
            "results": [{"content": "child", "subagent_trajectory_ref": [ref]}]
        }
    child = _trajectory(
        "child-session",
        ("child",),
        prompt_tokens=1,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.01,
    )
    child.pop("trajectory_id")
    (tmp_path / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")
    (tmp_path / "trajectory.child.json").write_text(json.dumps(child), encoding="utf-8")

    materialized, _ = _materialize_trajectory_file(tmp_path, "trajectory.json")

    assert [item["trajectory_id"] for item in materialized["subagent_trajectories"]] == ["child"]
    assert [
        step["observation"]["results"][0]["subagent_trajectory_ref"][0]["trajectory_id"]
        for step in materialized["steps"]
    ] == ["child", "child"]


def test_materialize_mixed_embedded_and_path_refs_share_one_canonical_child(tmp_path: Path) -> None:
    root = _trajectory(
        "root-session",
        ("first", "second"),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    child = _trajectory(
        "child-session",
        ("child",),
        prompt_tokens=1,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.01,
    )
    child["trajectory_id"] = "canonical-child"
    root["subagent_trajectories"] = [child]
    refs = (
        {"trajectory_id": "canonical-child", "trajectory_path": "trajectory.child.json"},
        {"trajectory_path": "trajectory.child.json"},
    )
    for step, ref in zip(root["steps"], refs, strict=True):
        step["observation"] = {
            "results": [{"content": "child", "subagent_trajectory_ref": [ref]}]
        }
    sidecar = dict(child)
    sidecar.pop("trajectory_id")
    (tmp_path / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")
    (tmp_path / "trajectory.child.json").write_text(json.dumps(sidecar), encoding="utf-8")

    materialized, _ = _materialize_trajectory_file(tmp_path, "trajectory.json")

    assert [item["trajectory_id"] for item in materialized["subagent_trajectories"]] == ["canonical-child"]
    assert [
        step["observation"]["results"][0]["subagent_trajectory_ref"][0]["trajectory_id"]
        for step in materialized["steps"]
    ] == ["canonical-child", "canonical-child"]


def test_materialize_does_not_trust_source_forged_continuation_provenance(tmp_path: Path) -> None:
    first = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    first["trajectory_id"] = "real-a"
    first["continued_trajectory_ref"] = "trajectory.cont-1.json"
    first["extra"] = {
        "harbor_continuation": {
            "segment_count": 999,
            "source_session_ids": ["forged-session"],
            "source_trajectory_ids": ["forged-id"],
            "source_root_extra": [{"forged": True}],
        }
    }
    second = _trajectory(
        "session-1",
        ("one", "two"),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    second["trajectory_id"] = "real-b"
    second["steps"][0]["is_copied_context"] = True
    (tmp_path / "trajectory.json").write_text(json.dumps(first), encoding="utf-8")
    (tmp_path / "trajectory.cont-1.json").write_text(json.dumps(second), encoding="utf-8")

    materialized, _ = _materialize_trajectory_file(tmp_path, "trajectory.json")

    provenance = materialized["extra"]["harbor_continuation"]
    assert provenance["segment_count"] == 2
    assert provenance["source_session_ids"] == ["session-1", "session-1"]
    assert provenance["source_trajectory_ids"] == ["real-a", "real-b"]


@pytest.mark.parametrize("poison", [math.nan, math.inf, -math.inf])
def test_merge_omits_nonfinite_standard_and_extra_aggregate_metrics(
    tmp_path: Path,
    poison: float,
) -> None:
    first = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    second = _trajectory(
        "session-2",
        ("two",),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    first["final_metrics"]["total_cost_usd"] = poison
    first["final_metrics"]["extra"]["reasoning_output_tokens"] = poison
    _write_multistep_trial(tmp_path, (("one", first), ("two", second)), resume_trajectory=False)

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    assert "total_cost_usd" not in merged["final_metrics"]
    assert "reasoning_output_tokens" not in merged["final_metrics"]["extra"]
    assert "NaN" not in json.dumps(merged)
    assert "Infinity" not in json.dumps(merged)


def test_merge_omits_unrepresentable_integer_aggregate_metrics(tmp_path: Path) -> None:
    first = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    second = _trajectory(
        "session-2",
        ("two",),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    first["final_metrics"]["total_prompt_tokens"] = 10**400
    first["final_metrics"]["extra"]["reasoning_output_tokens"] = 10**400
    _write_multistep_trial(tmp_path, (("one", first), ("two", second)), resume_trajectory=False)

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    assert "total_prompt_tokens" not in merged["final_metrics"]
    assert "reasoning_output_tokens" not in merged["final_metrics"]["extra"]
    json.dumps(merged, allow_nan=False)


def test_single_step_redaction_omits_unrepresentable_integer_aggregate_metrics(tmp_path: Path) -> None:
    trajectory = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    trajectory["final_metrics"]["total_prompt_tokens"] = 10**400
    trajectory["final_metrics"]["total_steps"] = 10**400
    trajectory["final_metrics"]["extra"]["reasoning_output_tokens"] = 10**400
    trajectory["steps"][0]["llm_call_count"] = 10**400
    trajectory["steps"][0]["metrics"] = {
        "prompt_tokens": 10**400,
        "completion_tokens": 10**400,
        "cached_tokens": 10**400,
        "extra": {"reasoning_output_tokens": 10**400},
    }

    redacted = _redacted_artifact_text(tmp_path / "trajectory.json", json.dumps(trajectory))

    assert redacted is not None
    persisted = json.loads(redacted)
    assert "total_prompt_tokens" not in persisted["final_metrics"]
    assert "total_steps" not in persisted["final_metrics"]
    assert "reasoning_output_tokens" not in persisted["final_metrics"]["extra"]
    assert "llm_call_count" not in persisted["steps"][0]
    step_metrics = persisted["steps"][0]["metrics"]
    assert "prompt_tokens" not in step_metrics
    assert "completion_tokens" not in step_metrics
    assert "cached_tokens" not in step_metrics
    assert "reasoning_output_tokens" not in step_metrics["extra"]
    Trajectory.model_validate(persisted)


def test_single_step_redaction_drops_credential_shaped_extra_key(tmp_path: Path) -> None:
    credential = "sk-abcdefghijk"
    trajectory = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    trajectory["extra"] = {credential: "value", "phase": "test"}

    redacted = _redacted_artifact_text(tmp_path / "trajectory.json", json.dumps(trajectory))

    assert redacted is not None
    assert credential not in redacted
    assert json.loads(redacted)["extra"] == {"phase": "test"}


@pytest.mark.parametrize("location", ["root", "agent", "step", "final_metrics"])
def test_single_step_redaction_rejects_browser_unsafe_integer_in_arbitrary_extra(
    tmp_path: Path,
    location: str,
) -> None:
    trajectory = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    if location == "root":
        trajectory["extra"] = {"custom_counter": 10**400}
    elif location == "agent":
        trajectory["agent"]["extra"] = {"custom_counter": 10**400}
    elif location == "step":
        trajectory["steps"][0]["extra"] = {"custom_counter": 10**400}
    else:
        trajectory["final_metrics"]["extra"]["custom_counter"] = 10**400

    assert _redacted_artifact_text(tmp_path / "trajectory.json", json.dumps(trajectory)) is None


def test_saved_deep_trajectory_surfaces_bounded_omission_to_report_loader(tmp_path: Path) -> None:
    trajectory = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    nested: dict[str, Any] = {}
    for _index in range(70):
        nested = {"next": nested}
    trajectory["extra"] = nested
    job_dir = tmp_path / "jobs"
    agent_dir = job_dir / "case-001" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    trials_dir = tmp_path / "results" / "opencode" / "with-skill" / "trials"
    reward = {
        "entry_id": "case-001",
        "metric_set": DEFAULT_METRIC_SET,
        **dict.fromkeys(DEFAULT_METRICS, 1.0),
        "_trial_name": "case-001",
        "_trial_root_name": "case-001",
    }

    _save_trials(
        [reward],
        trials_dir,
        job_dir,
        skill_name="demo",
        agent="opencode",
        variant="with_skill",
    )

    trial_out = trials_dir / "case-001"
    assert not (trial_out / "trajectory.json").exists()
    manifest = json.loads((trial_out / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert any(item.get("name") == "trajectory.json" for item in manifest["skipped"])
    persisted_reward = json.loads((trial_out / "reward.json").read_text(encoding="utf-8"))
    assert any("trajectory" in warning.casefold() for warning in persisted_reward["warnings"])

    summary_dir = trials_dir.parent
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "scores": dict.fromkeys(DEFAULT_METRICS, 1.0),
                "metrics": list(DEFAULT_METRICS),
                "num_trials": 1,
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 1,
                "scored_attempts": 1,
            }
        ),
        encoding="utf-8",
    )
    loaded = report_data.load_agent_data(tmp_path / "results")["opencode"]
    assert any("trajectory" in warning.casefold() for warning in loaded["rewards"][0]["warnings"])


def test_structural_reward_fallback_bounds_identity_and_model_before_publication(tmp_path: Path) -> None:
    trials_dir = tmp_path / "results" / "opencode" / "with-skill" / "trials"
    reward = {
        "entry_id": "e" * 2_100_000,
        "metric_set": DEFAULT_METRIC_SET,
        **dict.fromkeys(DEFAULT_METRICS, 1.0),
        "_trial_name": "case-001",
        "_trial_root_name": "case-001",
    }

    _save_trials(
        [reward],
        trials_dir,
        None,
        skill_name="demo",
        agent="opencode",
        variant="with_skill",
        agent_model="m" * 600_000,
    )

    reward_path = trials_dir / "case-001" / "reward.json"
    assert reward_path.stat().st_size <= report_data._MAX_JSON_BYTES
    diagnostics: list[dict[str, Any]] = []
    persisted = report_data._load_bounded_json(reward_path, diagnostics, artifact="reward")
    assert isinstance(persisted, dict)
    assert persisted["evaluation_status"] == "failed"
    assert "structural limits" in persisted["evaluation_errors"]["collector"]
    assert len(persisted["entry_id"]) <= 512
    assert len(persisted["model"]) <= 512
    assert diagnostics == []


def test_collected_reward_byte_reserve_bounds_model_metadata_and_keeps_scoreable(tmp_path: Path) -> None:
    trials_dir = tmp_path / "results" / "opencode" / "with-skill" / "trials"
    reward = {
        "entry_id": "case-001",
        "metric_set": DEFAULT_METRIC_SET,
        **dict.fromkeys(DEFAULT_METRICS, 1.0),
        "details": {"padding": ""},
        "_trial_name": "case-001",
        "_trial_root_name": "case-001",
    }
    initial_size = len(json.dumps(reward, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    reward["details"]["padding"] = "x" * (COLLECTED_REWARD_JSON_MAX_BYTES - initial_size)

    _save_trials(
        [reward],
        trials_dir,
        None,
        skill_name="demo",
        agent="opencode",
        variant="with_skill",
        agent_model="m" * 300_000,
        agent_model_source="cli",
    )

    reward_path = trials_dir / "case-001" / "reward.json"
    assert reward_path.stat().st_size <= report_data._MAX_JSON_BYTES
    persisted = json.loads(reward_path.read_text(encoding="utf-8"))
    assert persisted.get("evaluation_status") != "failed"
    assert all(persisted[metric] == 1.0 for metric in DEFAULT_METRICS)
    assert persisted["model"].startswith("m")
    assert len(persisted["model"]) <= 512
    assert persisted["model_source"] == "cli"


def test_merge_overwrites_source_spoofed_step_provenance(tmp_path: Path) -> None:
    trajectory = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    trajectory["steps"][0]["extra"] = {
        "harbor_step_name": "forged",
        "harbor_original_step_id": 999,
    }
    _write_multistep_trial(tmp_path, (("real", trajectory),), resume_trajectory=False)

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    assert merged["steps"][0]["extra"]["harbor_step_name"] == "real"
    assert merged["steps"][0]["extra"]["harbor_original_step_id"] == 1


def test_merge_scopes_same_embedded_id_from_independent_step_parents(tmp_path: Path) -> None:
    parents: list[tuple[str, dict[str, Any]]] = []
    for step_name, child_message in (("one", "child one"), ("two", "child two")):
        parent = _trajectory(
            f"session-{step_name}",
            (f"parent {step_name}",),
            prompt_tokens=10,
            completion_tokens=2,
            reasoning_tokens=1,
            cost_usd=0.1,
        )
        child = _trajectory(
            f"child-session-{step_name}",
            (child_message,),
            prompt_tokens=1,
            completion_tokens=1,
            reasoning_tokens=0,
            cost_usd=0.01,
        )
        child["trajectory_id"] = "child"
        parent["subagent_trajectories"] = [child]
        parent["steps"][0]["observation"] = {
            "results": [
                {
                    "content": "child",
                    "subagent_trajectory_ref": [{"trajectory_id": "child"}],
                }
            ]
        }
        parents.append((step_name, parent))
    _write_multistep_trial(tmp_path, tuple(parents), resume_trajectory=False)

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    Trajectory.model_validate(merged)
    embedded_ids = [child["trajectory_id"] for child in merged["subagent_trajectories"]]
    assert len(embedded_ids) == len(set(embedded_ids)) == 2
    assert [child["steps"][0]["message"] for child in merged["subagent_trajectories"]] == [
        "child one",
        "child two",
    ]
    refs = [
        step["observation"]["results"][0]["subagent_trajectory_ref"][0]["trajectory_id"]
        for step in merged["steps"]
    ]
    assert refs == embedded_ids
    for source_step, child in zip(("one", "two"), merged["subagent_trajectories"], strict=True):
        scope = child["extra"]["harbor_parent_scope"]
        assert scope["original_trajectory_id"] == "child"
        assert scope["parent_scope"] == f"harbor-step:{0 if source_step == 'one' else 1}"


def test_explicit_continuation_scopes_same_embedded_id_from_each_parent(tmp_path: Path) -> None:
    root = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    continuation = _trajectory(
        "session-1",
        ("one", "two"),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    root["continued_trajectory_ref"] = "trajectory.cont-1.json"
    continuation["steps"][0]["is_copied_context"] = True
    for parent, child_message, step_indexes in (
        (root, "child one", (0,)),
        (continuation, "child two", (0, 1)),
    ):
        child = _trajectory(
            f"session-{child_message.replace(' ', '-')}",
            (child_message,),
            prompt_tokens=1,
            completion_tokens=1,
            reasoning_tokens=0,
            cost_usd=0.01,
        )
        child["trajectory_id"] = "child"
        parent["subagent_trajectories"] = [child]
        for index in step_indexes:
            parent["steps"][index]["observation"] = {
                "results": [
                    {
                        "content": "child",
                        "subagent_trajectory_ref": [{"trajectory_id": "child"}],
                    }
                ]
            }
    (tmp_path / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")
    (tmp_path / "trajectory.cont-1.json").write_text(json.dumps(continuation), encoding="utf-8")

    materialized, _ = _materialize_trajectory_file(tmp_path, "trajectory.json")

    Trajectory.model_validate(materialized)
    embedded_ids = [child["trajectory_id"] for child in materialized["subagent_trajectories"]]
    assert len(embedded_ids) == len(set(embedded_ids)) == 2
    refs = [
        step["observation"]["results"][0]["subagent_trajectory_ref"][0]["trajectory_id"]
        for step in materialized["steps"]
    ]
    assert refs == embedded_ids


def test_merge_fails_closed_for_escaping_trajectory_reference(tmp_path: Path) -> None:
    root = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    root["continued_trajectory_ref"] = "../outside.json"
    _write_multistep_trial(tmp_path, (("only", root),), resume_trajectory=False)
    (tmp_path / "steps" / "only" / "outside.json").write_text(json.dumps(root), encoding="utf-8")

    assert _merged_step_trajectory(tmp_path) is None


def test_merge_omits_unknown_independent_metrics_instead_of_reporting_partial_totals(tmp_path: Path) -> None:
    first = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    second = _trajectory(
        "session-2",
        ("two",),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    second["final_metrics"] = {}
    _write_multistep_trial(tmp_path, (("one", first), ("two", second)), resume_trajectory=False)

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    Trajectory.model_validate(merged)
    assert set(merged["final_metrics"]) == {"total_steps", "extra"}
    assert merged["final_metrics"]["extra"] == {"harbor_multi_step": True}


def test_merge_omits_metric_when_terminal_cumulative_fragment_is_unknown(tmp_path: Path) -> None:
    first = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    second = _trajectory(
        "session-1",
        ("one", "two"),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    second["final_metrics"] = {}
    second["steps"][0]["is_copied_context"] = True
    _write_multistep_trial(tmp_path, (("one", first), ("two", second)), resume_trajectory=True)

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    Trajectory.model_validate(merged)
    assert set(merged["final_metrics"]) == {"total_steps", "extra"}


def test_merge_omits_unknown_aggregate_extra_metric_instead_of_terminal_partial_value(
    tmp_path: Path,
) -> None:
    first = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    second = _trajectory(
        "session-2",
        ("two",),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    del first["final_metrics"]["extra"]["reasoning_output_tokens"]
    _write_multistep_trial(tmp_path, (("one", first), ("two", second)), resume_trajectory=False)

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    Trajectory.model_validate(merged)
    assert "reasoning_output_tokens" not in merged["final_metrics"]["extra"]
    assert merged["final_metrics"]["extra"]["finish_reason"] == "stop"


def test_synthetic_multistep_id_changes_when_trajectory_content_changes(tmp_path: Path) -> None:
    ids: list[str] = []
    for directory, message in (("first", "alpha"), ("second", "beta")):
        trial_root = tmp_path / directory
        trajectory = _trajectory(
            "same-session",
            (message,),
            prompt_tokens=10,
            completion_tokens=2,
            reasoning_tokens=1,
            cost_usd=0.1,
        )
        _write_multistep_trial(trial_root, (("only", trajectory),), resume_trajectory=False)
        merged = _merged_step_trajectory(trial_root)
        assert merged is not None
        ids.append(merged["trajectory_id"])

    assert ids[0] != ids[1]


def test_synthetic_multistep_id_changes_when_authoritative_step_names_change(tmp_path: Path) -> None:
    ids: list[str] = []
    trajectory = _trajectory(
        "same-session",
        ("same",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    for directory, step_name in (("first", "one"), ("second", "renamed")):
        trial_root = tmp_path / directory
        _write_multistep_trial(trial_root, ((step_name, trajectory),), resume_trajectory=False)
        merged = _merged_step_trajectory(trial_root)
        assert merged is not None
        ids.append(merged["trajectory_id"])

    assert ids[0] != ids[1]


def test_synthetic_multistep_id_does_not_oracle_redacted_step_name_values(tmp_path: Path) -> None:
    ids: list[str] = []
    trajectory = _trajectory(
        "same-session",
        ("same",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    for directory, step_name in (("first", "api_key=one"), ("second", "api_key=two")):
        trial_root = tmp_path / directory
        _write_multistep_trial(trial_root, ((step_name, trajectory),), resume_trajectory=False)
        merged = _merged_step_trajectory(trial_root)
        assert merged is not None
        ids.append(merged["trajectory_id"])

    assert ids[0] == ids[1]


def test_synthetic_multistep_id_changes_with_resume_merge_semantics(tmp_path: Path) -> None:
    ids: list[str] = []
    step_counts: list[int] = []
    first = _trajectory(
        "same-session",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    second = _trajectory(
        "same-session",
        ("one", "two"),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    second["steps"][0]["is_copied_context"] = True
    for directory, resume in (("resumed", True), ("independent", False)):
        trial_root = tmp_path / directory
        _write_multistep_trial(
            trial_root,
            (("one", first), ("two", second)),
            resume_trajectory=resume,
        )
        merged = _merged_step_trajectory(trial_root)
        assert merged is not None
        ids.append(merged["trajectory_id"])
        step_counts.append(len(merged["steps"]))

    assert step_counts == [2, 3]
    assert ids[0] != ids[1]


def test_synthetic_multistep_id_does_not_oracle_redacted_secret_values(tmp_path: Path) -> None:
    ids: list[str] = []
    for directory, secret in (("first", "synthetic-secret-one"), ("second", "synthetic-secret-two")):
        trial_root = tmp_path / directory
        trajectory = _trajectory(
            "same-session",
            ("same",),
            prompt_tokens=10,
            completion_tokens=2,
            reasoning_tokens=1,
            cost_usd=0.1,
        )
        trajectory["steps"][0]["tool_calls"] = [
            {
                "tool_call_id": "call-1",
                "function_name": "request",
                "arguments": {"api_key": secret},
            }
        ]
        _write_multistep_trial(trial_root, (("only", trajectory),), resume_trajectory=False)
        merged = _merged_step_trajectory(trial_root)
        assert merged is not None
        ids.append(merged["trajectory_id"])

    assert ids[0] == ids[1]


def test_synthetic_multistep_id_does_not_oracle_redacted_root_extra_values(tmp_path: Path) -> None:
    ids: list[str] = []
    for directory, secret in (("first", "synthetic-secret-one"), ("second", "synthetic-secret-two")):
        trial_root = tmp_path / directory
        trajectory = _trajectory(
            "same-session",
            ("same",),
            prompt_tokens=10,
            completion_tokens=2,
            reasoning_tokens=1,
            cost_usd=0.1,
        )
        trajectory["extra"] = {"api_key": secret}
        _write_multistep_trial(trial_root, (("only", trajectory),), resume_trajectory=False)
        merged = _merged_step_trajectory(trial_root)
        assert merged is not None
        ids.append(merged["trajectory_id"])

    assert ids[0] == ids[1]


def test_minted_subagent_ids_are_stable_across_redacted_secret_only_path_changes(tmp_path: Path) -> None:
    ids: list[str] = []
    for directory, reference in (("first", "api_key=one.json"), ("second", "api_key=two.json")):
        agent_dir = tmp_path / directory
        agent_dir.mkdir()
        root = _trajectory(
            "root-session",
            ("delegate",),
            prompt_tokens=10,
            completion_tokens=2,
            reasoning_tokens=1,
            cost_usd=0.1,
        )
        root["steps"][0]["observation"] = {
            "results": [{"content": "child", "subagent_trajectory_ref": [{"trajectory_path": reference}]}]
        }
        child = _trajectory(
            "child-session",
            ("same child",),
            prompt_tokens=1,
            completion_tokens=1,
            reasoning_tokens=0,
            cost_usd=0.01,
        )
        child.pop("trajectory_id")
        (agent_dir / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")
        (agent_dir / reference).write_text(json.dumps(child), encoding="utf-8")

        materialized, _ = _materialize_trajectory_file(agent_dir, "trajectory.json")
        ids.append(materialized["subagent_trajectories"][0]["trajectory_id"])

    assert ids[0] == ids[1]


def test_distinct_secret_named_sidecars_receive_distinct_parent_local_ids(tmp_path: Path) -> None:
    root = _trajectory(
        "root-session",
        ("one", "two"),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    references = ("api_key=one.json", "api_key=two.json")
    child = _trajectory(
        "child-session",
        ("same child",),
        prompt_tokens=1,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.01,
    )
    child.pop("trajectory_id")
    for step, reference in zip(root["steps"], references, strict=True):
        step["observation"] = {
            "results": [{"content": "child", "subagent_trajectory_ref": [{"trajectory_path": reference}]}]
        }
        (tmp_path / reference).write_text(json.dumps(child), encoding="utf-8")
    (tmp_path / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")

    materialized, _ = _materialize_trajectory_file(tmp_path, "trajectory.json")

    embedded_ids = [item["trajectory_id"] for item in materialized["subagent_trajectories"]]
    assert len(embedded_ids) == len(set(embedded_ids)) == 2


def test_embedded_secret_only_id_collisions_are_remapped_without_oracle(tmp_path: Path) -> None:
    emitted_ids: list[list[str]] = []
    for directory, source_ids in (
        ("first", ("api_key=one", "api_key=two")),
        ("second", ("api_key=alpha", "api_key=beta")),
    ):
        agent_dir = tmp_path / directory
        agent_dir.mkdir()
        root = _trajectory(
            "root-session",
            ("delegate",),
            prompt_tokens=10,
            completion_tokens=2,
            reasoning_tokens=1,
            cost_usd=0.1,
        )
        children: list[dict[str, Any]] = []
        refs: list[dict[str, str]] = []
        for source_id in source_ids:
            child = _trajectory(
                "child-session",
                ("child",),
                prompt_tokens=1,
                completion_tokens=1,
                reasoning_tokens=0,
                cost_usd=0.01,
            )
            child["trajectory_id"] = source_id
            children.append(child)
            refs.append({"trajectory_id": source_id})
        root["subagent_trajectories"] = children
        root["steps"][0]["observation"] = {
            "results": [{"content": "children", "subagent_trajectory_ref": refs}]
        }
        (agent_dir / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")

        materialized, _ = _materialize_trajectory_file(agent_dir, "trajectory.json")
        redacted = _redacted_artifact_text(agent_dir / "trajectory.json", json.dumps(materialized))

        assert redacted is not None
        persisted = json.loads(redacted)
        Trajectory.model_validate(persisted)
        ids = [child["trajectory_id"] for child in persisted["subagent_trajectories"]]
        assert len(ids) == len(set(ids)) == 2
        resolved_refs = persisted["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"]
        assert [ref["trajectory_id"] for ref in resolved_refs] == ids
        emitted_ids.append(ids)

    assert emitted_ids[0] == emitted_ids[1]


def test_external_idless_continuation_chains_receive_distinct_reference_scoped_ids(tmp_path: Path) -> None:
    root = _trajectory(
        "root-session",
        ("one", "two"),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    for step, reference in zip(root["steps"], ("base-a.json", "base-b.json"), strict=True):
        step["observation"] = {
            "results": [{"content": "child", "subagent_trajectory_ref": [{"trajectory_path": reference}]}]
        }
    (tmp_path / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")
    for suffix, secret in (("a", "secret-one"), ("b", "secret-two")):
        base = _trajectory(
            f"base-{suffix}",
            ("base",),
            prompt_tokens=1,
            completion_tokens=1,
            reasoning_tokens=0,
            cost_usd=0.01,
        )
        continuation = _trajectory(
            f"continuation-{suffix}",
            ("next",),
            prompt_tokens=2,
            completion_tokens=2,
            reasoning_tokens=0,
            cost_usd=0.02,
        )
        for trajectory in (base, continuation):
            trajectory.pop("session_id")
            trajectory.pop("trajectory_id")
        base["extra"] = {"api_key": secret}
        base["continued_trajectory_ref"] = f"cont-{suffix}.json"
        (tmp_path / f"base-{suffix}.json").write_text(json.dumps(base), encoding="utf-8")
        (tmp_path / f"cont-{suffix}.json").write_text(json.dumps(continuation), encoding="utf-8")

    materialized, _ = _materialize_trajectory_file(tmp_path, "trajectory.json")

    embedded_ids = [item["trajectory_id"] for item in materialized["subagent_trajectories"]]
    assert len(embedded_ids) == len(set(embedded_ids)) == 2


def test_embedded_sibling_continuations_receive_distinct_parent_local_ids(tmp_path: Path) -> None:
    root = _trajectory(
        "root-session",
        ("delegate",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    children: list[dict[str, Any]] = []
    refs: list[dict[str, str]] = []
    for suffix in ("one", "two"):
        child = _trajectory(
            f"child-{suffix}",
            ("base",),
            prompt_tokens=1,
            completion_tokens=1,
            reasoning_tokens=0,
            cost_usd=0.01,
        )
        child["trajectory_id"] = f"api_key={suffix}"
        child["continued_trajectory_ref"] = f"continuation-{suffix}.json"
        continuation = _trajectory(
            f"continuation-{suffix}",
            ("next",),
            prompt_tokens=2,
            completion_tokens=2,
            reasoning_tokens=0,
            cost_usd=0.02,
        )
        continuation["trajectory_id"] = f"continuation-api_key={suffix}"
        children.append(child)
        refs.append({"trajectory_id": child["trajectory_id"]})
        (tmp_path / f"continuation-{suffix}.json").write_text(json.dumps(continuation), encoding="utf-8")
    root["subagent_trajectories"] = children
    root["steps"][0]["observation"] = {
        "results": [{"content": "children", "subagent_trajectory_ref": refs}]
    }
    (tmp_path / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")

    materialized, _ = _materialize_trajectory_file(tmp_path, "trajectory.json")

    embedded_ids = [item["trajectory_id"] for item in materialized["subagent_trajectories"]]
    assert len(embedded_ids) == len(set(embedded_ids)) == 2
    resolved_refs = materialized["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"]
    assert [ref["trajectory_id"] for ref in resolved_refs] == embedded_ids


def test_merge_preserves_source_agent_and_custom_metric_extras_in_provenance(tmp_path: Path) -> None:
    first = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    second = _trajectory(
        "session-2",
        ("two",),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    first["agent"]["extra"] = {"phase": "base"}
    second["agent"]["extra"] = {"phase": "next"}
    first["final_metrics"]["extra"]["llm_calls"] = 1
    second["final_metrics"]["extra"]["llm_calls"] = 2
    _write_multistep_trial(tmp_path, (("one", first), ("two", second)), resume_trajectory=False)

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    assert "llm_calls" not in merged["final_metrics"]["extra"]
    sources = merged["extra"]["harbor_multi_step"]["source_trajectories"]
    assert [source["agent_extra"] for source in sources] == [{"phase": "base"}, {"phase": "next"}]
    assert [source["final_metrics_extra"]["llm_calls"] for source in sources] == [1, 2]


@pytest.mark.parametrize("location", ["custom_step_metric", "step_extra", "custom_final_metric"])
def test_merge_fails_closed_for_nonfinite_values_outside_known_aggregate_fields(
    tmp_path: Path,
    location: str,
) -> None:
    trajectory = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    if location == "custom_step_metric":
        trajectory["steps"][0]["metrics"] = {"custom_metric": math.nan}
    elif location == "step_extra":
        trajectory["steps"][0]["extra"] = {"custom_metric": math.inf}
    else:
        trajectory["final_metrics"]["extra"]["llm_calls"] = -math.inf
    _write_multistep_trial(tmp_path, (("only", trajectory),), resume_trajectory=False)

    assert _merged_step_trajectory(tmp_path) is None


def test_merge_handles_json_escaped_lone_surrogate_without_crashing(tmp_path: Path) -> None:
    trajectory = _trajectory(
        "session-1",
        ("\ud800",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    _write_multistep_trial(tmp_path, (("only", trajectory),), resume_trajectory=False)

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    assert "\\ud800" in json.dumps(merged, ensure_ascii=True, allow_nan=False)


def test_merge_fails_closed_for_non_utf8_trajectory_reference(tmp_path: Path) -> None:
    trajectory = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    trajectory["continued_trajectory_ref"] = "\ud800.json"
    _write_multistep_trial(tmp_path, (("only", trajectory),), resume_trajectory=False)

    assert _merged_step_trajectory(tmp_path) is None


def test_maximum_step_count_retains_independent_subagent_reference_budget(tmp_path: Path) -> None:
    trajectories: list[tuple[str, dict[str, Any]]] = []
    for index in range(64):
        trajectory = _trajectory(
            f"session-{index}",
            (f"step-{index}",),
            prompt_tokens=1,
            completion_tokens=1,
            reasoning_tokens=0,
            cost_usd=0.01,
        )
        trajectories.append((f"step-{index}", trajectory))
    first = trajectories[0][1]
    first["steps"][0]["observation"] = {
        "results": [
            {
                "content": "child",
                "subagent_trajectory_ref": [{"trajectory_path": "trajectory.child.json"}],
            }
        ]
    }
    _write_multistep_trial(tmp_path, tuple(trajectories), resume_trajectory=False)
    child = _trajectory(
        "child-session",
        ("child",),
        prompt_tokens=1,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.01,
    )
    child.pop("trajectory_id")
    (tmp_path / "steps" / "step-0" / "agent" / "trajectory.child.json").write_text(
        json.dumps(child),
        encoding="utf-8",
    )

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    assert len(merged["steps"]) == 64
    assert len(merged["subagent_trajectories"]) == 1


def test_native_resume_deduplicates_collector_materialized_explicit_continuation_prefix(
    tmp_path: Path,
) -> None:
    first_root = _trajectory(
        "segment-a",
        ("a",),
        prompt_tokens=10,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.1,
    )
    first_continuation = _trajectory(
        "segment-b",
        ("b",),
        prompt_tokens=20,
        completion_tokens=2,
        reasoning_tokens=0,
        cost_usd=0.2,
    )
    second_root = _trajectory(
        "segment-a",
        ("a",),
        prompt_tokens=10,
        completion_tokens=1,
        reasoning_tokens=0,
        cost_usd=0.1,
    )
    second_continuation = _trajectory(
        "segment-c",
        ("b", "c"),
        prompt_tokens=30,
        completion_tokens=3,
        reasoning_tokens=0,
        cost_usd=0.3,
    )
    first_root["continued_trajectory_ref"] = "trajectory.cont.json"
    second_root["continued_trajectory_ref"] = "trajectory.cont.json"
    _write_multistep_trial(
        tmp_path,
        (("first", first_root), ("second", second_root)),
        resume_trajectory=True,
    )
    (tmp_path / "steps" / "first" / "agent" / "trajectory.cont.json").write_text(
        json.dumps(first_continuation),
        encoding="utf-8",
    )
    (tmp_path / "steps" / "second" / "agent" / "trajectory.cont.json").write_text(
        json.dumps(second_continuation),
        encoding="utf-8",
    )

    merged = _merged_step_trajectory(tmp_path)

    assert merged is not None
    assert [step["message"] for step in merged["steps"]] == ["a", "b", "c"]
    assert merged["final_metrics"]["total_steps"] == 3
    assert merged["final_metrics"]["total_prompt_tokens"] == 30


def test_minted_external_subagent_id_changes_when_content_changes(tmp_path: Path) -> None:
    ids: list[str] = []
    for directory, message in (("first", "alpha"), ("second", "beta")):
        agent_dir = tmp_path / directory
        agent_dir.mkdir()
        root = _trajectory(
            "root-session",
            ("delegate",),
            prompt_tokens=10,
            completion_tokens=2,
            reasoning_tokens=1,
            cost_usd=0.1,
        )
        root["steps"][0]["observation"] = {
            "results": [
                {
                    "content": "delegated",
                    "subagent_trajectory_ref": [{"trajectory_path": "trajectory.subagent.json"}],
                }
            ]
        }
        subagent = _trajectory(
            "same-subagent-session",
            (message,),
            prompt_tokens=1,
            completion_tokens=1,
            reasoning_tokens=0,
            cost_usd=0.01,
        )
        subagent.pop("trajectory_id")
        (agent_dir / "trajectory.json").write_text(json.dumps(root), encoding="utf-8")
        (agent_dir / "trajectory.subagent.json").write_text(json.dumps(subagent), encoding="utf-8")

        materialized, _ = _materialize_trajectory_file(agent_dir, "trajectory.json")
        ids.append(materialized["subagent_trajectories"][0]["trajectory_id"])

    assert ids[0] != ids[1]


def test_unscored_multistep_trial_persists_materialized_trajectory(tmp_path: Path) -> None:
    trial_root = tmp_path / "jobs" / "trial-1"
    first = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    second = _trajectory(
        "session-2",
        ("two",),
        prompt_tokens=20,
        completion_tokens=4,
        reasoning_tokens=2,
        cost_usd=0.2,
    )
    _write_multistep_trial(trial_root, (("one", first), ("two", second)), resume_trajectory=False)

    _save_unscored_trials(
        [],
        tmp_path / "results",
        tmp_path / "jobs",
        agent="opencode",
        variant="with",
    )

    trial_out = tmp_path / "results" / "trial-1"
    persisted = json.loads((trial_out / "trajectory.json").read_text(encoding="utf-8"))
    Trajectory.model_validate(persisted)
    assert [step["message"] for step in persisted["steps"]] == ["one", "two"]
    failure = json.loads((trial_out / "failure.json").read_text(encoding="utf-8"))
    assert "trajectory.json" in failure["artifacts"]


def test_trajectory_redaction_omits_token_id_arrays_and_remains_atif_valid(tmp_path: Path) -> None:
    trajectory = _trajectory(
        "session-1",
        ("one",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    trajectory["steps"][0]["metrics"] = {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "prompt_token_ids": [101, 102],
        "completion_token_ids": [201],
    }
    Trajectory.model_validate(trajectory)

    redacted = _redacted_artifact_text(tmp_path / "trajectory.json", json.dumps(trajectory))

    assert redacted is not None
    persisted = json.loads(redacted)
    Trajectory.model_validate(persisted)
    metrics = persisted["steps"][0]["metrics"]
    assert "prompt_token_ids" not in metrics
    assert "completion_token_ids" not in metrics


def test_trajectory_redaction_masks_uri_userinfo_and_remains_atif_valid(tmp_path: Path) -> None:
    trajectory = _trajectory(
        "session-1",
        ("request https://alice:correct@horse@example.test/path failed",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )

    redacted = _redacted_artifact_text(tmp_path / "trajectory.json", json.dumps(trajectory))

    assert redacted is not None
    assert "alice" not in redacted
    assert "correct@horse" not in redacted
    assert "<redacted>@example.test" in redacted
    Trajectory.model_validate_json(redacted)


def test_trajectory_redaction_masks_plural_and_nonnumeric_token_fields(tmp_path: Path) -> None:
    trajectory = _trajectory(
        "session-1",
        ("request",),
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=1,
        cost_usd=0.1,
    )
    trajectory["steps"][0]["tool_calls"] = [
        {
            "tool_call_id": "call-1",
            "function_name": "request",
            "arguments": {
                "tokens": ["correct-horse-battery"],
                "passwords": ["hunter-two-secret"],
                "access_tokens": ["access-token-secret"],
            },
        }
    ]

    redacted = _redacted_artifact_text(tmp_path / "trajectory.json", json.dumps(trajectory))

    assert redacted is not None
    assert "correct-horse-battery" not in redacted
    assert "hunter-two-secret" not in redacted
    assert "access-token-secret" not in redacted
    persisted = json.loads(redacted)
    arguments = persisted["steps"][0]["tool_calls"][0]["arguments"]
    assert arguments == {
        "tokens": "<redacted>",
        "passwords": "<redacted>",
        "access_tokens": "<redacted>",
    }
    assert persisted["final_metrics"]["total_prompt_tokens"] == 10
    Trajectory.model_validate(persisted)
