# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillevaluator.evaluation import DatasetGenerationError
from skillevaluator.inference import client as inference_client
from skillevaluator.tier3 import generate_dataset
from skillevaluator.tier3.evals_spec import validate_tier3_source


def _skill_dir(tmp_path: Path, *, guidance: str | None = None) -> Path:
    skill = tmp_path / "sample-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: sample-skill\ndescription: Does useful work\n---\n# Skill\n",
        encoding="utf-8",
    )
    if guidance is not None:
        evals = skill / "evals"
        evals.mkdir()
        (evals / "EVAL.md").write_text(guidance, encoding="utf-8")
    return skill


def test_intent_grounding_requires_author_guidance(tmp_path):
    skill = _skill_dir(tmp_path)

    with pytest.raises(DatasetGenerationError, match="requires author guidance"):
        generate_dataset.main([str(skill), "--no-llm", "--grounding", "intent"])


def test_intent_records_eval_provenance_and_template_mode(tmp_path, capsys):
    guidance = "## Questions\n- Reconcile the invoice using the documented workflow.\n"
    skill = _skill_dir(tmp_path, guidance=guidance)

    result = generate_dataset.main([str(skill), "--no-llm", "--grounding", "intent"])

    generation = result.dataset["generation"]
    eval_input = next(record for record in generation["inputs"] if record["kind"] == "eval_md")
    assert eval_input == {
        "kind": "eval_md",
        "path": "evals/EVAL.md",
        "sha256": hashlib.sha256(guidance.encode()).hexdigest(),
    }
    assert generation["mode"] == "template_requested"
    assert generation["grounding"] == "intent"
    assert "Grounding: author intent from evals/EVAL.md" in capsys.readouterr().out


def test_trace_preserves_authored_ground_truth_and_flags_mismatches():
    cases = [
        {
            "id": "case-1",
            "question": "Do the task",
            "ground_truth": "Authored expected answer",
            "expected_behavior": ["The agent uses the approved workflow"],
            "expected_skill": "sample-skill",
        }
    ]
    trajectories = {
        "case-1": {
            "steps": [
                {"source": "agent", "message": "Observed different answer from the run."},
            ]
        }
    }

    refined = generate_dataset._refine_from_trajectory_template(cases, trajectories)
    review = refined[0]["review"]

    assert refined[0]["ground_truth"] == "Authored expected answer"
    assert review["status"] == "needs_review"
    assert "agent answer differs from authored ground truth" in review["reasons"]
    assert "The agent uses the approved workflow" in refined[0]["expected_behavior"]
    assert not any(reason.startswith("refinement dropped authored step:") for reason in review["reasons"])


def test_llm_refinement_restores_and_flags_dropped_authored_behavior(monkeypatch):
    class FakeLLMClient:
        def __init__(self, **_kwargs):
            pass

        def completions(self, *_args):
            return json.dumps(
                [
                    {
                        "id": "case-1",
                        "question": "Do the task",
                        "ground_truth": "Authored expected answer",
                        "expected_behavior": ["The agent uses a refined workflow"],
                        "expected_skill": "sample-skill",
                    }
                ]
            )

    monkeypatch.setattr(
        "skillevaluator.provider_config.resolve_llm_provider",
        lambda: SimpleNamespace(
            provider="test",
            model="test",
        ),
    )
    monkeypatch.setattr(inference_client, "LLMClient", FakeLLMClient)
    cases = [
        {
            "id": "case-1",
            "question": "Do the task",
            "ground_truth": "Authored expected answer",
            "expected_behavior": ["The agent uses the approved workflow"],
            "expected_skill": "sample-skill",
        }
    ]

    refined = asyncio.run(
        generate_dataset._refine_with_llm(
            {"name": "sample-skill"},
            cases,
            {"case-1": {"steps": []}},
        )
    )

    assert refined[0]["expected_behavior"][:2] == [
        "The agent uses the approved workflow",
        "The agent uses a refined workflow",
    ]
    assert refined[0]["review"]["reasons"] == ["refinement dropped authored step: The agent uses the approved workflow"]


def test_trace_proposes_unauthored_observed_answer_and_flags_review():
    cases = [{"id": "case-1", "question": "Do the task", "expected_behavior": [], "expected_skill": "sample-skill"}]
    trajectories = {
        "case-1": {"steps": [{"source": "agent", "message": "Observed answer that is longer than twenty characters."}]}
    }

    refined = generate_dataset._refine_from_trajectory_template(cases, trajectories)

    assert refined[0]["ground_truth"] == "Observed answer that is longer than twenty characters."
    assert refined[0]["review"]["status"] == "needs_review"
    assert "expected outcome taken from an observed run; confirm it is correct" in refined[0]["review"]["reasons"]


def test_default_skill_grounding_keeps_template_cases_and_records_generation(tmp_path):
    skill = _skill_dir(tmp_path)
    parsed = generate_dataset._parse_skill(skill)
    expected_cases = generate_dataset._generate_simple(parsed)

    result = generate_dataset.main([str(skill), "--no-llm"])

    assert result.dataset["generation"]["grounding"] == "skill"
    assert result.dataset["generation"]["mode"] == "template_requested"
    assert result.dataset["evals"] == generate_dataset._to_agentskills_dataset("sample-skill", expected_cases)["evals"]


def test_trace_grounding_fails_when_dry_run_has_no_trajectory(tmp_path, monkeypatch):
    skill = _skill_dir(tmp_path)
    monkeypatch.setattr(generate_dataset, "_discover_trajectories", lambda *_args, **_kwargs: ({}, {}))

    with pytest.raises(DatasetGenerationError, match="Trace grounding requires at least one trajectory"):
        generate_dataset.main([str(skill), "--no-llm", "--dry-run", "--grounding", "trace"])


def test_validator_accepts_generation_and_case_review_metadata(tmp_path):
    skill = _skill_dir(tmp_path)
    evals = skill / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(
        json.dumps(
            {
                "skill_name": "sample-skill",
                "generation": {
                    "mode": "template_requested",
                    "grounding": "intent",
                    "inputs": [{"kind": "skill_md", "path": "SKILL.md", "sha256": "0" * 64}],
                    "generated_at": "2026-01-01T00:00:00+00:00",
                    "review_state": "needs_review",
                },
                "evals": [
                    {
                        "id": "case-1",
                        "prompt": "Do the task",
                        "expected_output": "A result",
                        "assertions": ["Use the approved workflow"],
                        "review": {"status": "needs_review", "reasons": ["Check this case"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    source, checks = validate_tier3_source(skill)

    assert source == "evals_json"
    assert not [check for check in checks if check.status == "error"]


def test_llm_generation_records_mode_and_trace_refinement_flag(tmp_path, monkeypatch):
    skill = _skill_dir(tmp_path)
    case = {
        "id": "case-1",
        "question": "Do the task",
        "expected_skill": "sample-skill",
        "expected_script": None,
        "ground_truth": "A useful result",
        "expected_behavior": ["The agent completes the task"],
    }

    async def generated(*_args, **_kwargs):
        return [case]

    monkeypatch.setattr(generate_dataset, "_generate_with_llm", generated)
    llm_result = generate_dataset.generate_one_case(skill, use_llm=True)
    llm_record = json.loads(llm_result.read_text())["generation"]
    assert llm_record["mode"] == "llm"
    assert llm_record["grounding"] == "skill"
    assert "refined" not in llm_record


def test_template_fallback_records_safe_reason(tmp_path, monkeypatch):
    from skillevaluator.provider_config import ProviderConfigurationError

    skill = _skill_dir(tmp_path)

    def provider_unavailable():
        raise ProviderConfigurationError("credential-value-that-must-not-leak")

    monkeypatch.setattr("skillevaluator.provider_config.resolve_llm_provider", provider_unavailable)
    result = generate_dataset.main([str(skill), "--dry-run"])

    record = result.dataset["generation"]
    assert record["mode"] == "template_fallback"
    assert record["fallback_reason"] == "public LLM provider is not configured"
    assert "credential-value-that-must-not-leak" not in json.dumps(record)


def test_trace_refinement_keeps_template_mode_and_records_refined(tmp_path, monkeypatch):
    skill = _skill_dir(tmp_path)
    case = {
        "id": "case-1",
        "question": "Do the task",
        "expected_skill": "sample-skill",
        "ground_truth": "A useful result",
        "expected_behavior": ["The agent completes the task"],
    }
    monkeypatch.setattr(generate_dataset, "_generate_simple", lambda _skill: [case])
    monkeypatch.setattr(
        generate_dataset,
        "_discover_trajectories",
        lambda *_args, **_kwargs: ({"case-1": {"steps": []}}, {}),
    )

    result = generate_dataset.main([str(skill), "--no-llm", "--grounding", "trace", "--dry-run"])

    record = result.dataset["generation"]
    assert record["mode"] == "template_requested"
    assert record["grounding"] == "trace"
    assert record["refined"] is True


def test_autopilot_records_llm_and_template_fallback(tmp_path, monkeypatch):
    async def generated(*_args, **_kwargs):
        return [
            {
                "id": "case-1",
                "question": "Do the task",
                "expected_skill": "sample-skill",
                "expected_script": None,
                "ground_truth": "A useful result",
                "expected_behavior": ["The agent completes the task"],
            }
        ]

    monkeypatch.setattr(generate_dataset, "_generate_with_llm", generated)
    llm_skill = _skill_dir(tmp_path)
    llm_record = json.loads(generate_dataset.generate_one_case(llm_skill, use_llm=True).read_text())["generation"]
    assert llm_record["mode"] == "llm"
    assert llm_record["grounding"] == "skill"

    fallback_root = tmp_path / "fallback"
    fallback_root.mkdir()
    fallback_skill = _skill_dir(fallback_root)
    fallback_record = json.loads(
        generate_dataset.generate_one_case(
            fallback_skill,
            use_llm=False,
            fallback_reason="public LLM provider is not configured",
        ).read_text()
    )["generation"]
    assert fallback_record["mode"] == "template_fallback"
    assert fallback_record["grounding"] == "skill"
    assert fallback_record["fallback_reason"] == "public LLM provider is not configured"
