# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone guided generation selects evidence before any expensive or writing work."""

import hashlib
import json

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.evaluation import DatasetGenerationError, DatasetOptions, EvaluationService
from skillevaluator.tier3 import generate_dataset


def _skill(tmp_path, guidance=None):
    skill = tmp_path / "sample-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: sample-skill\ndescription: Reconcile invoices\n---\n")
    if guidance is not None:
        (skill / "evals").mkdir()
        (skill / "evals" / "EVAL.md").write_text(guidance)
    return skill


def _trace(skill, results_root, case_id="sample-skill-001", *, steps=None, skill_name=None):
    run = results_root / (skill_name or skill.name) / "20261112_120000"
    trial = run / "claude-code" / "with-skill" / "trials" / case_id
    trial.mkdir(parents=True)
    source = trial / "trajectory.json"
    source.write_text(
        json.dumps(
            {
                "steps": steps
                if steps is not None
                else [{"source": "agent", "message": "Reconciled 42 invoices; three mismatches were recorded."}]
            }
        )
    )
    (run / "run_config.json").write_text("{}")
    (run / "result.json").write_text(json.dumps({"run_id": run.name, "skill_name": skill_name or skill.name}))
    return source


def _forbid_generation(*_args, **_kwargs):
    pytest.fail("auto preflight must finish before generation or trajectory collection")


@pytest.mark.parametrize("explicit_auto", [False, True])
def test_auto_prefers_nonblank_author_intent_to_retained_trace(tmp_path, monkeypatch, explicit_auto):
    guidance = "## Questions\n- Check all invoices\n## Behaviors\n- Report each mismatch\n"
    skill = _skill(tmp_path, guidance)
    _trace(skill, tmp_path / "results")
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    result = generate_dataset.main(
        [str(skill), "--no-llm", "--dry-run", *(["--grounding", "auto"] if explicit_auto else [])]
    )

    assert result.dataset["generation"]["grounding"] == "intent"
    assert result.dataset["evals"][0]["prompt"] == "Check all invoices"
    assert result.dataset["evals"][0]["assertions"][0] == "Report each mismatch"
    assert not [item for item in result.dataset["generation"]["inputs"] if item["kind"] == "trajectory"]


def test_auto_prompt_overrides_eval_guidance_and_records_effective_source(tmp_path):
    skill = _skill(tmp_path, "## Questions\n- Wrong prompt\n")
    prompt = tmp_path / "intent.md"
    prompt.write_text("## Questions\n- Reconcile the authored invoice\n")

    result = generate_dataset.main([str(skill), "--no-llm", "--prompt", str(prompt)])

    assert result.dataset["evals"][0]["prompt"] == "Reconcile the authored invoice"
    assert result.dataset["generation"]["grounding"] == "intent"
    sources = result.dataset["generation"]["inputs"]
    assert {"kind": "prompt", "path": "intent.md", "sha256": hashlib.sha256(prompt.read_bytes()).hexdigest()} in sources
    assert not [item for item in sources if item["kind"] == "eval_md"]


@pytest.mark.parametrize("entrypoint", ["direct", "service", "cli"])
def test_auto_keeps_existing_dataset_without_explicit_refinement(tmp_path, monkeypatch, entrypoint):
    skill = _skill(tmp_path, "\n")
    output = skill / "evals" / "evals.json"
    original = (
        b'{"skill_name":"sample-skill","evals":'
        b'[{"id":"authored","prompt":"Reconcile","expected_output":"Keep","assertions":["Keep"]}]}\n'
    )
    output.write_bytes(original)
    _trace(skill, tmp_path / "results", case_id="authored")
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(generate_dataset, "_generate_with_llm", _forbid_generation)
    monkeypatch.setattr(generate_dataset, "_run_agent_collect_trajectories", _forbid_generation)

    if entrypoint == "direct":
        result = generate_dataset.main([str(skill), "--no-llm"])
        assert result.status == "unchanged"
    elif entrypoint == "service":
        result = EvaluationService().create_dataset(DatasetOptions(skill_path=skill, no_llm=True))
        assert result.status == "unchanged"
    else:
        result = CliRunner().invoke(cli, ["create-eval-dataset", str(skill), "--no-llm"])
        assert result.exit_code == 0, result.output
        assert "Dataset already exists" in result.output
    assert output.read_bytes() == original


def test_auto_unchanged_dataset_does_not_parse_the_skill(tmp_path, monkeypatch):
    skill = _skill(tmp_path, "\n")
    output = skill / "evals" / "evals.json"
    original = b'{"evals": []}'
    output.write_bytes(original)
    monkeypatch.setattr(generate_dataset, "_parse_skill", _forbid_generation)

    result = generate_dataset.main([str(skill)])

    assert result.status == "unchanged"
    assert output.read_bytes() == original


@pytest.mark.parametrize("use_llm", [False, True])
@pytest.mark.parametrize("authored", [False, True])
def test_auto_trace_keeps_skill_use_out_of_new_expected_outputs(tmp_path, monkeypatch, use_llm, authored):
    skill = _skill(tmp_path, "\n")
    observed = "Using the sample-skill skill I read SKILL.md and ran reconcile.py: 42 invoices reconciled"
    if authored:
        (skill / "evals" / "evals.json").write_text(
            json.dumps(
                {
                    "skill_name": "sample-skill",
                    "evals": [{"id": "sample-skill-001", "prompt": "Reconcile", "expected_output": observed}],
                }
            )
        )
    _trace(skill, tmp_path / "results", steps=[{"source": "agent", "message": observed}])
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    async def refine(_skill, cases, _trajectories, **_kwargs):
        return [{**case, "ground_truth": observed} for case in cases]

    monkeypatch.setattr(generate_dataset, "_refine_with_llm", refine)
    result = generate_dataset.main(
        [str(skill), *([] if use_llm else ["--no-llm"]), *(["--refine"] if authored else [])]
    )

    case = result.dataset["evals"][0]
    assert case["expected_output"] == (observed if authored else "")
    if not authored:
        assert any("skill use" in reason for reason in case["review"]["reasons"])
        assert case["review"]["status"] == "needs_review"


@pytest.mark.parametrize("retained_trace", [False, True])
def test_auto_does_not_treat_headings_alone_as_author_guidance(tmp_path, monkeypatch, retained_trace):
    skill = _skill(tmp_path, "# Evaluation\n\n## Questions\n\n## Behaviors\n\n## Notes\n")
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))
    if retained_trace:
        _trace(skill, tmp_path / "results")
        result = generate_dataset.main([str(skill), "--no-llm"])
        assert result.dataset["generation"]["grounding"] == "trace"
    else:
        with pytest.raises(DatasetGenerationError, match="--grounding auto requires"):
            generate_dataset.main([str(skill), "--no-llm"])
        assert not (skill / "evals" / "evals.json").exists()


def test_auto_force_invalid_dataset_explains_the_explicit_starter_option(tmp_path, monkeypatch):
    skill = _skill(tmp_path, "\n")
    output = skill / "evals" / "evals.json"
    original = b"{}"
    output.write_bytes(original)
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "absent"))

    with pytest.raises(DatasetGenerationError) as error:
        generate_dataset.main([str(skill), "--no-llm", "--force"])

    assert "--grounding skill --force" in str(error.value)
    assert output.read_bytes() == original


@pytest.mark.parametrize("authored_output", [42, 0, False, {"answer": 42}, ["42"]])
def test_auto_refine_preserves_typed_authored_expected_outputs(tmp_path, monkeypatch, authored_output):
    skill = _skill(tmp_path, "\n")
    (skill / "evals" / "evals.json").write_text(
        json.dumps(
            {
                "skill_name": "sample-skill",
                "evals": [{"id": "authored", "prompt": "Reconcile", "expected_output": authored_output}],
            }
        )
    )
    _trace(skill, tmp_path / "results", case_id="authored")
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    result = generate_dataset.main([str(skill), "--no-llm", "--refine"])

    assert result.dataset["evals"][0]["expected_output"] == authored_output


@pytest.mark.parametrize("options", [[], ["--refine"], ["--no-llm"], ["--no-llm", "--refine", "--force"]])
def test_auto_missing_input_fails_before_provider_collection_or_writes(tmp_path, monkeypatch, options):
    skill = _skill(tmp_path, " \n\t")
    monkeypatch.setattr(generate_dataset, "_generate_with_llm", _forbid_generation)
    monkeypatch.setattr(generate_dataset, "_run_agent_collect_trajectories", _forbid_generation)
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "absent"))

    with pytest.raises(DatasetGenerationError, match="--grounding auto requires") as error:
        generate_dataset.main([str(skill), *options])

    for recovery in ("EVAL.md", "--prompt", "trace", "--grounding skill"):
        assert recovery in str(error.value)
    assert not (skill / "evals" / "evals.json").exists()
    assert sorted(path.name for path in (skill / "evals").iterdir()) == ["EVAL.md"]


@pytest.mark.parametrize("source_option", ["env", "--results-dir", "--from-results"])
def test_auto_falls_back_to_matching_retained_trace_and_flags_observation(tmp_path, monkeypatch, source_option):
    skill = _skill(tmp_path, "\n")
    source = _trace(skill, tmp_path / "results")
    monkeypatch.setattr(generate_dataset, "_generate_with_llm", _forbid_generation)
    options = []
    if source_option == "env":
        monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))
    else:
        options = [
            source_option,
            str(tmp_path / "results") if source_option == "--results-dir" else str(source.parents[4]),
        ]

    result = generate_dataset.main([str(skill), "--no-llm", *options])

    generation = result.dataset["generation"]
    assert generation["grounding"] == "trace"
    assert generation["refined"] is True
    assert generation["review_state"] == "needs_review"
    assert {
        "kind": "trajectory",
        "path": "trajectory.json",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    } in generation["inputs"]
    case = result.dataset["evals"][0]
    assert case["id"] == "sample-skill-001"
    assert case["expected_output"] == "Reconciled 42 invoices; three mismatches were recorded."
    assert case["review"]["status"] == "needs_review"


@pytest.mark.parametrize(
    "steps", [[], [{}], [{"source": "user", "message": "Do the task"}], [{"source": "agent", "message": " "}]]
)
def test_auto_rejects_empty_or_non_agent_trace_evidence(tmp_path, monkeypatch, steps):
    skill = _skill(tmp_path)
    _trace(skill, tmp_path / "results", steps=steps)
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    with pytest.raises(DatasetGenerationError, match="--grounding auto requires"):
        generate_dataset.main([str(skill), "--no-llm"])
    assert not (skill / "evals" / "evals.json").exists()


def test_auto_rejects_unmatched_case_ids_before_generation(tmp_path, monkeypatch):
    skill = _skill(tmp_path)
    _trace(skill, tmp_path / "results", case_id="unrelated")
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(generate_dataset, "_generate_with_llm", _forbid_generation)

    with pytest.raises(DatasetGenerationError, match="--grounding auto requires"):
        generate_dataset.main([str(skill)])
    assert not (skill / "evals" / "evals.json").exists()


def test_auto_rejects_foreign_explicit_results_even_with_matching_case_id(tmp_path):
    skill = _skill(tmp_path)
    source = _trace(skill, tmp_path / "results", skill_name="another-skill")

    with pytest.raises(DatasetGenerationError, match="--grounding auto requires"):
        generate_dataset.main([str(skill), "--no-llm", "--from-results", str(source.parents[4])])
    assert not (skill / "evals" / "evals.json").exists()


def test_auto_trace_preserves_authored_cases_and_flags_unmatched_coverage(tmp_path, monkeypatch):
    skill = _skill(tmp_path, "\n")
    dataset = skill / "evals" / "evals.json"
    dataset.write_text(
        json.dumps(
            {
                "skill_name": "sample-skill",
                "evals": [
                    {
                        "id": "authored",
                        "prompt": "Check this invoice",
                        "expected_output": "Correct authored answer",
                        "assertions": ["Keep authored behavior"],
                    },
                    {
                        "id": "unmatched",
                        "prompt": "Check another invoice",
                        "expected_output": "Another authored answer",
                    },
                ],
            }
        )
    )
    source = _trace(skill, tmp_path / "results", case_id="authored")
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    result = generate_dataset.main([str(skill), "--no-llm", "--refine"])

    assert [case["id"] for case in result.dataset["evals"]] == ["authored", "unmatched"]
    first, second = result.dataset["evals"]
    assert first["expected_output"] == "Correct authored answer"
    assert "Keep authored behavior" in first["assertions"]
    assert second["expected_output"] == "Another authored answer"
    assert second["review"]["status"] == "needs_review"
    assert "no matching retained trajectory for this case" in second["review"]["reasons"]
    assert result.dataset["generation"]["needs_review_count"] == 2
    assert hashlib.sha256(source.read_bytes()).hexdigest() in json.dumps(result.dataset["generation"])


def test_auto_intent_refine_retains_collection_compatibility(tmp_path, monkeypatch):
    skill = _skill(tmp_path, "## Questions\n- Check this invoice\n")
    monkeypatch.setattr(generate_dataset, "_discover_trajectories", lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(generate_dataset, "_run_agent_collect_trajectories", lambda *_args, **_kwargs: ({}, {}))

    result = generate_dataset.main([str(skill), "--no-llm", "--refine"])

    assert result.dataset["generation"]["grounding"] == "intent"
    assert result.dataset["evals"][0]["prompt"] == "Check this invoice"


@pytest.mark.parametrize("entrypoint", ["service", "cli", "tier3-cli"])
def test_guided_default_is_shared_by_cli_and_service(tmp_path, entrypoint):
    skill = _skill(tmp_path)
    if entrypoint == "service":
        with pytest.raises(DatasetGenerationError, match="--grounding auto requires"):
            EvaluationService().create_dataset(DatasetOptions(skill_path=skill, no_llm=True))
    else:
        command = ["tier3"] if entrypoint == "tier3-cli" else []
        result = CliRunner().invoke(cli, [*command, "create-eval-dataset", str(skill), "--no-llm"])
        assert result.exit_code == 1
        assert "--grounding auto requires" in result.output
    assert not (skill / "evals" / "evals.json").exists()


def test_service_explicit_skill_preserves_legacy_template_workflow(tmp_path):
    skill = _skill(tmp_path)

    result = EvaluationService().create_dataset(DatasetOptions(skill_path=skill, no_llm=True, grounding="skill"))

    assert result.dataset["generation"]["grounding"] == "skill"
    assert result.path.is_file()


def test_auto_missing_input_preserves_existing_dataset_when_forced(tmp_path, monkeypatch):
    skill = _skill(tmp_path, "\n")
    output = skill / "evals" / "evals.json"
    original = b'{"skill_name":"sample-skill","evals":[{"id":"authored","prompt":"Check","expected_output":"Keep"}]}'
    output.write_bytes(original)
    monkeypatch.setattr(generate_dataset, "_generate_with_llm", _forbid_generation)
    monkeypatch.setattr(generate_dataset, "_run_agent_collect_trajectories", _forbid_generation)

    with pytest.raises(DatasetGenerationError, match="--grounding auto requires"):
        generate_dataset.main([str(skill), "--force", "--refine"])
    assert output.read_bytes() == original


def test_auto_trace_does_not_regenerate_ids_before_llm_refinement(tmp_path, monkeypatch):
    skill = _skill(tmp_path)
    _trace(skill, tmp_path / "results")
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(generate_dataset, "_generate_with_llm", _forbid_generation)
    monkeypatch.setattr(generate_dataset, "_run_agent_collect_trajectories", _forbid_generation)

    async def refine(parsed, cases, trajectories, **kwargs):
        return generate_dataset._refine_from_trajectory_template(cases, trajectories, skill=parsed, **kwargs)

    monkeypatch.setattr(generate_dataset, "_refine_with_llm", refine)

    result = generate_dataset.main([str(skill)])

    assert result.dataset["generation"]["grounding"] == "trace"
    assert result.dataset["evals"][0]["id"] == "sample-skill-001"
    assert result.dataset["evals"][0]["review"]["status"] == "needs_review"


def test_auto_rejects_incomplete_run_and_conflicting_skill_identity(tmp_path, monkeypatch):
    skill = _skill(tmp_path)
    source = _trace(skill, tmp_path / "results")
    run = source.parents[4]
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))
    (run / "result.json").write_text(json.dumps({"run_id": run.name, "skill_name": "another-skill"}))
    with pytest.raises(DatasetGenerationError, match="--grounding auto requires"):
        generate_dataset.main([str(skill), "--no-llm"])
    (run / "result.json").unlink()
    with pytest.raises(DatasetGenerationError, match="--grounding auto requires"):
        generate_dataset.main([str(skill), "--no-llm"])
    assert not (skill / "evals" / "evals.json").exists()


@pytest.mark.parametrize("tool_calls", [7, True, {"function_name": "bash"}, "bash"])
def test_auto_ignores_malformed_tool_call_steps(tmp_path, monkeypatch, tool_calls):
    skill = _skill(tmp_path)
    _trace(skill, tmp_path / "results", steps=[{"source": "agent", "tool_calls": tool_calls}])
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    with pytest.raises(DatasetGenerationError, match="--grounding auto requires"):
        generate_dataset.main([str(skill), "--no-llm"])


def test_auto_existing_empty_dataset_is_unchanged_without_force(tmp_path):
    skill = _skill(tmp_path, "\n")
    output = skill / "evals" / "evals.json"
    original = '{"skill_name":"sample-skill","evals":[]}'
    output.write_text(original)

    result = generate_dataset.main([str(skill), "--no-llm"])

    assert result.status == "unchanged"
    assert output.read_text() == original


@pytest.mark.parametrize("message", [7, {"text": "not ATIF content"}, [{"type": "image", "source": "/tmp/image.png"}]])
def test_auto_requires_text_or_tool_actions_beyond_placeholder_messages(tmp_path, monkeypatch, message):
    skill = _skill(tmp_path)
    _trace(skill, tmp_path / "results", steps=[{"source": "agent", "message": message}])
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    with pytest.raises(DatasetGenerationError, match="--grounding auto requires"):
        generate_dataset.main([str(skill), "--no-llm"])


@pytest.mark.parametrize("higher_case_id", ["unrelated", "sample-skill-001"])
def test_auto_searches_past_an_agent_without_usable_matching_evidence(tmp_path, monkeypatch, higher_case_id):
    skill = _skill(tmp_path)
    source = _trace(
        skill,
        tmp_path / "results",
        case_id=higher_case_id,
        steps=[
            {"source": "agent" if higher_case_id == "unrelated" else "user", "message": "This is only the user request"}
        ],
    )
    fallback = source.parents[4] / "codex" / "with-skill" / "trials" / "sample-skill-001" / "trajectory.json"
    fallback.parent.mkdir(parents=True)
    fallback.write_text(
        json.dumps({"steps": [{"source": "agent", "message": "Reconciled all invoices; the totals now match."}]})
    )
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    result = generate_dataset.main([str(skill), "--no-llm"])

    assert result.dataset["generation"]["grounding"] == "trace"
    hashes = [item["sha256"] for item in result.dataset["generation"]["inputs"] if item["kind"] == "trajectory"]
    assert hashes == [hashlib.sha256(fallback.read_bytes()).hexdigest()]
    assert result.dataset["evals"][0]["expected_output"] == "Reconciled all invoices; the totals now match."


@pytest.mark.parametrize("use_llm", [False, True])
@pytest.mark.parametrize("wrapper", ["evals", "cases"])
def test_auto_trace_preserves_authored_prompt_metadata_and_review_history(tmp_path, monkeypatch, use_llm, wrapper):
    from types import SimpleNamespace

    from skillevaluator.inference import client as inference_client

    skill = _skill(tmp_path, "\n")
    output = skill / "evals" / "evals.json"
    output.write_text(
        json.dumps(
            {
                "skill_name": "sample-skill",
                "dataset_version": "author-v2",
                "source": {"kind": "reviewed_import"},
                wrapper: [
                    {
                        "id": "authored",
                        "prompt": "Check this exact authored invoice",
                        "expected_output": "Authored answer",
                        "expected_skill": "sample-skill",
                        "assertions": ["Keep authored behavior"],
                        "files": ["files/invoice.csv"],
                        "behavior_checks": [{"kind": "final_response", "pattern": "total"}],
                        "review": {"status": "needs_review", "reasons": ["Original reviewer concern"]},
                    }
                ],
            }
        )
    )
    original_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    _trace(skill, tmp_path / "results", case_id="authored")
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))

    class FakeLLMClient:
        def __init__(self, **_kwargs):
            pass

        def completions(self, *_args):
            return json.dumps(
                [
                    {
                        "id": "authored",
                        "question": "Changed prompt from model",
                        "ground_truth": "Changed expected answer",
                        "expected_skill": "different-skill",
                        "expected_behavior": ["Keep authored behavior"],
                    }
                ]
            )

    monkeypatch.setattr(
        "skillevaluator.provider_config.resolve_llm_provider", lambda: SimpleNamespace(provider="test", model="test")
    )
    monkeypatch.setattr(inference_client, "LLMClient", FakeLLMClient)

    result = generate_dataset.main([str(skill), "--refine", *([] if use_llm else ["--no-llm"])])

    case = result.dataset["evals"][0]
    assert case["prompt"] == "Check this exact authored invoice"
    assert case["expected_output"] == "Authored answer"
    assert case["expected_skill"] == "sample-skill"
    assert case["files"] == ["files/invoice.csv"]
    assert case["behavior_checks"] == [{"kind": "final_response", "pattern": "total"}]
    assert "Original reviewer concern" in case["review"]["reasons"]
    assert result.dataset["dataset_version"] == "author-v2"
    assert result.dataset["source"] == {"kind": "reviewed_import"}
    assert "cases" not in result.dataset
    assert result.dataset["generation"]["grounding"] == "trace"
    assert result.dataset["generation"]["mode"] == ("llm" if use_llm else "template_requested")
    assert {"kind": "dataset", "path": "evals/evals.json", "sha256": original_hash} in result.dataset["generation"][
        "inputs"
    ]


@pytest.mark.parametrize(
    "filename", ["trajectory.json", "claude-code.txt", "cursor-cli.txt", "cline.txt", "opencode.txt", "codex.txt"]
)
def test_auto_rejects_nonregular_trace_sources_before_reading(tmp_path, monkeypatch, filename):
    import os
    from pathlib import Path

    skill = _skill(tmp_path)
    source = _trace(skill, tmp_path / "results")
    source.unlink()
    blocked = source.parent / filename
    os.mkfifo(blocked)
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))
    original_read = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path == blocked:
            pytest.fail("auto discovery must reject a FIFO before attempting to read it")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    with pytest.raises(DatasetGenerationError, match="--grounding auto requires"):
        generate_dataset.main([str(skill), "--no-llm"])
    assert not (skill / "evals" / "evals.json").exists()


@pytest.mark.parametrize("relative", ["reward.json", "verifier/reward.json", "result.json"])
def test_auto_rejects_linked_case_identity_before_reading(tmp_path, monkeypatch, relative):
    from pathlib import Path

    skill = _skill(tmp_path)
    source = _trace(skill, tmp_path / "results")
    outside = tmp_path / "foreign-result.json"
    outside.write_text(json.dumps({"entry_id": "sample-skill-001"}))
    linked = source.parent / relative
    linked.parent.mkdir(exist_ok=True)
    linked.symlink_to(outside)
    monkeypatch.setenv("SKILLEVALUATOR_RESULTS_DIR", str(tmp_path / "results"))
    original_read = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path == linked:
            pytest.fail("auto discovery must reject linked identity inputs before reading them")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    with pytest.raises(DatasetGenerationError, match="--grounding auto requires"):
        generate_dataset.main([str(skill), "--no-llm"])
    assert not (skill / "evals" / "evals.json").exists()


@pytest.mark.parametrize("existing_authored", [False, True])
@pytest.mark.parametrize(
    ("provider_behavior", "expected_reason"),
    [
        ("unconfigured", "public LLM provider is not configured"),
        (
            "request_failure",
            "Unexpected LLM request error. Check the selected provider and model configuration, then retry.",
        ),
    ],
)
def test_cli_auto_trace_fallback_records_template_mode_and_preserves_sources(
    tmp_path,
    monkeypatch,
    existing_authored,
    provider_behavior,
    expected_reason,
):
    from skillevaluator.inference import client as inference_client
    from skillevaluator.provider_config import resolve_llm_provider
    from skillevaluator.tier3.dataset_provenance import dataset_provenance_display

    skill = _skill(tmp_path)
    source = _trace(skill, tmp_path / "results")
    output = skill / "evals" / "evals.json"
    original_hash = None
    if existing_authored:
        output.parent.mkdir()
        output.write_text(
            json.dumps(
                {
                    "skill_name": skill.name,
                    "dataset_version": "author-v3",
                    "source": {"kind": "authored"},
                    "evals": [
                        {
                            "id": "sample-skill-001",
                            "prompt": "Keep this authored invoice task",
                            "expected_output": "Keep this independent human expectation",
                            "expected_skill": skill.name,
                            "assertions": ["Keep the authored acceptance behavior"],
                            "files": ["files/invoice.csv"],
                            "review": {"status": "needs_review", "reasons": ["Keep the original review concern"]},
                        }
                    ],
                }
            )
        )
        original_hash = hashlib.sha256(output.read_bytes()).hexdigest()

    def resolve_provider():
        environment = {"SKILL_EVAL_LLM_PROVIDER": "openai"}
        if provider_behavior != "unconfigured":
            environment["OPENAI_API_KEY"] = "test-provider-key"
        return resolve_llm_provider(environment)

    class FailingClient:
        def __init__(self, **_kwargs):
            pass

        def completions(self, *_args):
            if provider_behavior == "unconfigured":
                pytest.fail("unconfigured provider must not attempt inference")
            raise RuntimeError("private-provider-detail")

    monkeypatch.setattr("skillevaluator.provider_config.resolve_llm_provider", resolve_provider)
    monkeypatch.setattr(inference_client, "LLMClient", FailingClient)
    result = CliRunner().invoke(
        cli,
        [
            "create-eval-dataset",
            str(skill),
            "--results-dir",
            str(tmp_path / "results"),
            *(["--refine"] if existing_authored else []),
        ],
    )

    assert result.exit_code == 0, result.output
    dataset = json.loads(output.read_text())
    generation = dataset["generation"]
    assert generation["mode"] == "template_fallback"
    assert generation["grounding"] == "trace"
    assert generation["fallback_reason"] == expected_reason
    assert generation["refined"] is True
    assert generation["review_state"] == "needs_review"
    assert "private-provider-detail" not in json.dumps(generation)
    assert "private-provider-detail" not in result.output
    assert hashlib.sha256(source.read_bytes()).hexdigest() in json.dumps(generation["inputs"])
    case = dataset["evals"][0]
    assert case["review"]["status"] == "needs_review"
    if existing_authored:
        assert case["prompt"] == "Keep this authored invoice task"
        assert case["expected_output"] == "Keep this independent human expectation"
        assert "Keep the authored acceptance behavior" in case["assertions"]
        assert case["files"] == ["files/invoice.csv"]
        assert "Keep the original review concern" in case["review"]["reasons"]
        assert dataset["dataset_version"] == "author-v3"
        assert dataset["source"] == {"kind": "authored"}
        assert {"kind": "dataset", "path": "evals/evals.json", "sha256": original_hash} in generation["inputs"]
    else:
        assert "expected outcome taken from an observed run; confirm it is correct" in case["review"]["reasons"]
    display, warning = dataset_provenance_display(generation)
    assert "generated from templates (fallback:" in display
    assert expected_reason in display
    assert warning is True
