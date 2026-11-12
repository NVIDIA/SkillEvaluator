# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import re
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


def _write_trace_results(
    skill: Path,
    results_root: Path,
    case_id: str,
    trajectory: dict,
) -> Path:
    run_id = "20260224_120000"
    run_dir = results_root / skill.name / run_id
    trial = run_dir / "claude-code" / "with-skill" / "trials" / case_id
    trial.mkdir(parents=True)
    trajectory_path = trial / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    latest = results_root / skill.name / "latest"
    if not latest.exists():
        latest.symlink_to(run_id)
    return trajectory_path


def _invoice_trajectory() -> dict:
    return {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "function_name": "read_file",
                        "arguments": {"path": "/workspace/skills/sample-skill/SKILL.md"},
                    }
                ],
            },
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "function_name": "bash",
                        "arguments": {"command": "python reconcile.py inv.csv"},
                    }
                ],
            },
            {
                "source": "agent",
                "message": "Reconciled 42 invoices; three mismatches were written to mismatches.csv.",
            },
        ]
    }


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


def test_intent_case_002_keeps_all_authored_behaviors():
    cases = generate_dataset._generate_full(
        {
            "name": "sample-skill",
            "description": "Does useful work",
            "scripts": ["reconcile.py"],
            "grounding": "intent",
            "eval_prompt": (
                "## Behaviors\n"
                "- The agent validates the invoice total against the ledger\n"
                "- The agent reports each mismatched row\n"
            ),
        }
    )

    case_002 = cases[1]

    assert case_002["expected_behavior"][:2] == [
        "The agent validates the invoice total against the ledger",
        "The agent reports each mismatched row",
    ]
    # The no-skill baseline is graded on the same behaviors, so none grades skill use.
    assert not [item for item in case_002["expected_behavior"] if re.search(r"SKILL\.md|reconcile\.py", item)]


@pytest.mark.parametrize("existing_dataset", [False, True], ids=["generated-cases", "authored-cases"])
def test_main_trace_refines_real_trajectory_without_losing_authored_content(
    tmp_path,
    monkeypatch,
    existing_dataset,
):
    skill = _skill_dir(tmp_path)
    trajectory_path = _write_trace_results(
        skill,
        tmp_path / "results",
        "sample-skill-001",
        _invoice_trajectory(),
    )
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    if existing_dataset:
        evals_dir = skill / "evals"
        evals_dir.mkdir()
        _write_trace_results(
            skill,
            tmp_path / "results",
            "unmatched-case",
            {"steps": [{"source": "agent", "message": "This trajectory belongs to another case."}]},
        )
        (evals_dir / "evals.json").write_text(
            json.dumps(
                {
                    "skill_name": "sample-skill",
                    "evals": [
                        {
                            "id": "sample-skill-001",
                            "prompt": "AUTHORED: reconcile these invoices",
                            "expected_output": "AUTHORED: the ledger was reconciled",
                            "assertions": ["AUTHORED: validate each ledger row"],
                            "expected_skill": "sample-skill",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    result = generate_dataset.main([str(skill), "--no-llm", "--dry-run", "--grounding", "trace"])

    case = result.dataset["evals"][0]
    if existing_dataset:
        assert case["prompt"] == "AUTHORED: reconcile these invoices"
        assert case["expected_output"] == "AUTHORED: the ledger was reconciled"
        assert "AUTHORED: validate each ledger row" in case["assertions"]
        assert "agent answer differs from authored ground truth" in case["review"]["reasons"]
    else:
        assert case["expected_output"] == "Reconciled 42 invoices; three mismatches were written to mismatches.csv."
        assert "expected outcome taken from an observed run; confirm it is correct" in case["review"]["reasons"]
        assert not any("used the sample-skill skill" in behavior for behavior in case["assertions"])

    # Reading the skill's SKILL.md is skill use, which the no-skill baseline is not graded on.
    assert not [assertion for assertion in case["assertions"] if "SKILL.md" in assertion]
    assert "The agent executed: python reconcile.py inv.csv" in case["assertions"]
    trajectory_inputs = [entry for entry in result.dataset["generation"]["inputs"] if entry["kind"] == "trajectory"]
    assert len(trajectory_inputs) == 1
    assert trajectory_inputs[0]["sha256"] == hashlib.sha256(trajectory_path.read_bytes()).hexdigest()


def test_trace_aborts_when_no_trajectory_id_matches_existing_cases(tmp_path, monkeypatch):
    skill = _skill_dir(tmp_path)
    evals_dir = skill / "evals"
    evals_dir.mkdir()
    output_path = evals_dir / "evals.json"
    original = json.dumps(
        {
            "skill_name": "sample-skill",
            "evals": [{"id": "authored-case", "prompt": "Do it", "expected_output": "Keep this"}],
        }
    ).encode()
    output_path.write_bytes(original)
    _write_trace_results(skill, tmp_path / "results", "unmatched-case", _invoice_trajectory())
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    with pytest.raises(DatasetGenerationError, match="--grounding trace found trajectories, but none matched"):
        generate_dataset.main([str(skill), "--no-llm", "--dry-run", "--grounding", "trace"])

    assert output_path.read_bytes() == original


def test_trace_restores_authored_dataset_when_collection_fails(tmp_path, monkeypatch):
    skill = _skill_dir(tmp_path)
    evals_dir = skill / "evals"
    evals_dir.mkdir()
    output_path = evals_dir / "evals.json"
    original = json.dumps(
        {
            "skill_name": "sample-skill",
            "evals": [{"id": "authored-case", "prompt": "Do it", "expected_output": "Keep this"}],
        }
    ).encode()
    output_path.write_bytes(original)
    monkeypatch.setattr(generate_dataset, "_discover_trajectories", lambda *_args, **_kwargs: ({}, {}))

    def failed_collection(*_args, **_kwargs):
        output_path.write_text("temporary replacement", encoding="utf-8")
        raise DatasetGenerationError("Docker daemon is not running")

    monkeypatch.setattr(generate_dataset, "_run_agent_collect_trajectories", failed_collection)

    with pytest.raises(DatasetGenerationError, match="--grounding trace collection failed: Docker daemon"):
        generate_dataset.main([str(skill), "--no-llm", "--grounding", "trace"])

    assert output_path.read_bytes() == original


def test_main_uses_collected_trajectory_sources_for_generation_inputs(tmp_path, monkeypatch):
    skill = _skill_dir(tmp_path)
    trajectory = _invoice_trajectory()
    trajectory_path = tmp_path / "collected-trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    discoveries = iter([({}, {}), ({"sample-skill-001": trajectory}, {"sample-skill-001": trajectory_path})])
    monkeypatch.setattr(generate_dataset, "_discover_trajectories", lambda *_args, **_kwargs: next(discoveries))

    def collect(*_args, **kwargs):
        assert kwargs["return_sources"] is True
        return {"sample-skill-001": trajectory}, {"sample-skill-001": trajectory_path}

    monkeypatch.setattr(generate_dataset, "_run_agent_collect_trajectories", collect)

    result = generate_dataset.main([str(skill), "--no-llm", "--grounding", "trace"])

    trajectory_inputs = [entry for entry in result.dataset["generation"]["inputs"] if entry["kind"] == "trajectory"]
    assert len(trajectory_inputs) == 1
    assert trajectory_inputs[0]["sha256"] == hashlib.sha256(trajectory_path.read_bytes()).hexdigest()


def test_main_llm_trace_uses_observed_answer_for_generated_cases(tmp_path, monkeypatch):
    skill = _skill_dir(tmp_path)
    _write_trace_results(
        skill,
        tmp_path / "results",
        "sample-skill-001",
        _invoice_trajectory(),
    )
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    async def generate_cases(skill, **_kwargs):
        return generate_dataset._generate_simple(skill)

    monkeypatch.setattr(generate_dataset, "_generate_with_llm", generate_cases)

    class FakeLLMClient:
        def __init__(self, **_kwargs):
            pass

        def completions(self, *_args):
            return json.dumps(
                [
                    {
                        "id": "sample-skill-001",
                        "question": "Use sample-skill.",
                        "ground_truth": "Generated placeholder answer",
                        "expected_behavior": [],
                        "expected_skill": "sample-skill",
                    }
                ]
            )

    monkeypatch.setattr(
        "skillevaluator.provider_config.resolve_llm_provider",
        lambda: SimpleNamespace(provider="test", model="test"),
    )
    monkeypatch.setattr(inference_client, "LLMClient", FakeLLMClient)

    result = generate_dataset.main([str(skill), "--dry-run", "--grounding", "trace"])

    case = result.dataset["evals"][0]
    assert case["expected_output"] == "Reconciled 42 invoices; three mismatches were written to mismatches.csv."
    assert "expected outcome taken from an observed run; confirm it is correct" in case["review"]["reasons"]
    assert "The agent used generated expectations" not in case["assertions"]


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


def test_template_refinement_deduplicates_standard_behaviors_and_keeps_trajectory_actions():
    case = {
        "id": "case-1",
        "question": "Do the task",
        "ground_truth": "Authored answer",
        "expected_skill": "sample-skill",
        "expected_behavior": [
            "The agent uses the approved workflow",
            generate_dataset._FINAL_RESPONSE_BEHAVIOR,
            generate_dataset.SECURITY_BEHAVIOR,
            generate_dataset._FINAL_RESPONSE_BEHAVIOR,
            generate_dataset.SECURITY_BEHAVIOR,
        ],
    }
    trajectory = {
        "steps": [
            {
                "tool_calls": [
                    {"function_name": "read_file", "arguments": {"path": "/tmp/SKILL.md"}},
                    {"function_name": "bash", "arguments": {"command": "python reconcile.py inv.csv"}},
                ]
            }
        ]
    }

    refined = generate_dataset._refine_from_trajectory_template([case], {"case-1": trajectory})[0]

    behaviors = refined["expected_behavior"]
    # The SKILL.md read is skill use, which the no-skill baseline is not graded on, so it is skipped.
    assert behaviors[:2] == [
        "The agent uses the approved workflow",
        "The agent executed: python reconcile.py inv.csv",
    ]
    assert not [behavior for behavior in behaviors if "SKILL.md" in behavior]
    assert behaviors.count(generate_dataset._FINAL_RESPONSE_BEHAVIOR) == 1
    assert behaviors.count(generate_dataset.SECURITY_BEHAVIOR) == 1
    assert behaviors[-2:] == [generate_dataset._FINAL_RESPONSE_BEHAVIOR, generate_dataset.SECURITY_BEHAVIOR]


def test_authored_restore_does_not_treat_generic_standard_lines_as_authored():
    case = {}

    restored = generate_dataset._restore_authored_behaviors(
        case,
        [generate_dataset._FINAL_RESPONSE_BEHAVIOR, generate_dataset.SECURITY_BEHAVIOR],
        [],
    )

    assert restored == []
    assert "review" not in case


def test_llm_refinement_falls_back_and_flags_missing_case_ids(monkeypatch):
    prompts = []

    class FakeLLMClient:
        def __init__(self, **_kwargs):
            pass

        def completions(self, *_args):
            prompts.append(_args[-1])
            return json.dumps(
                [
                    {
                        "id": "case-1",
                        "ground_truth": "Different answer",
                        "expected_behavior": ["The agent uses a refined workflow"],
                    },
                    {
                        "id": "case-2-renamed",
                        "ground_truth": "Observed answer",
                        "expected_behavior": [],
                    },
                    {
                        "id": "case-3",
                        "ground_truth": "Unobserved model answer",
                        "expected_behavior": ["The agent records a result"],
                    },
                ]
            )

    monkeypatch.setattr(
        "skillevaluator.provider_config.resolve_llm_provider",
        lambda: SimpleNamespace(provider="test", model="test"),
    )
    monkeypatch.setattr(inference_client, "LLMClient", FakeLLMClient)
    cases = [
        {
            "id": "case-1",
            "question": "Do the task",
            "ground_truth": "Authored answer",
            "expected_behavior": ["The agent uses the approved workflow"],
            "expected_skill": "sample-skill",
        },
        {
            "id": "case-2",
            "question": "Do the other task",
            "ground_truth": "Authored second answer",
            "expected_behavior": ["The agent reports every mismatch"],
            "expected_skill": "sample-skill",
        },
        {
            "id": "case-3",
            "question": "Do one more task",
            "ground_truth": "Authored third answer",
            "expected_behavior": ["The agent records the result"],
            "expected_skill": "sample-skill",
        },
    ]
    trajectories = {
        case_id: {
            "steps": [{"source": "agent", "message": f"Observed answer for {case_id} with enough detail to be used."}]
        }
        for case_id in ("case-1", "case-2")
    }

    refined = asyncio.run(
        generate_dataset._refine_with_llm(
            {"name": "sample-skill"},
            cases,
            trajectories,
            authored_case_ids={"case-1", "case-2", "case-3"},
        )
    )

    assert [case["id"] for case in refined] == ["case-1", "case-2", "case-3"]
    assert refined[0]["ground_truth"] == "Authored answer"
    assert refined[0]["review"]["reasons"] == [
        "refinement dropped authored step: The agent uses the approved workflow",
        "agent answer differs from authored ground truth",
    ]
    assert "refinement output did not return this case" in refined[1]["review"]["reasons"]
    assert refined[1]["ground_truth"] == "Authored second answer"
    assert refined[2]["ground_truth"] == "Authored third answer"
    assert "agent answer differs from authored ground truth" not in refined[2].get("review", {}).get("reasons", [])
    assert generate_dataset.SECURITY_BEHAVIOR in prompts[0]


def test_explicit_skill_grounding_keeps_template_cases_and_records_generation(tmp_path):
    skill = _skill_dir(tmp_path)
    parsed = generate_dataset._parse_skill(skill)
    expected_cases = generate_dataset._generate_simple(parsed)

    result = generate_dataset.main([str(skill), "--no-llm", "--grounding", "skill"])

    assert result.dataset["generation"]["grounding"] == "skill"
    assert result.dataset["generation"]["mode"] == "template_requested"
    assert result.dataset["evals"] == generate_dataset._to_agentskills_dataset("sample-skill", expected_cases)["evals"]


def test_trace_grounding_fails_when_dry_run_has_no_trajectory(tmp_path, monkeypatch):
    skill = _skill_dir(tmp_path)
    monkeypatch.setattr(generate_dataset, "_discover_trajectories", lambda *_args, **_kwargs: ({}, {}))

    with pytest.raises(DatasetGenerationError, match="--grounding trace requires a trajectory"):
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
    result = generate_dataset.main([str(skill), "--dry-run", "--grounding", "skill"])

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
