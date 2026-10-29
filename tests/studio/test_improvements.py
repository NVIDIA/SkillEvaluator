# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import zipfile

import pytest

from skillevaluator.studio import improvements
from skillevaluator.studio.jobs import Studio
from skillevaluator.studio.store import Workspace, new_id, write_json
from skillevaluator.tier3.improve import snapshot_tree


@pytest.fixture
def retained(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("SKILL.md", "---\nname: example\ndescription: Count words.\n---\nUse the broken helper.\n")
        script = zipfile.ZipInfo("scripts/helper.py")
        script.external_attr = 0o100755 << 16
        output.writestr(script, "def count_words(text):\n    return 0\n")
        output.writestr(
            "evals/evals.json",
            '{"skill_name":"example","evals":[{"id":"case","prompt":"Count red blue","expected_output":"2"}]}',
        )
        output.writestr("evals/files/incoming/SKILL.md", "A legitimate input skill fixture.\n")
        output.writestr("evals/files/__pycache__/fixture.pyc", b"frozen fixture bytes")
    skill = workspace.create(
        {"filename": "example.zip", "archive_base64": base64.b64encode(archive.getvalue()).decode()}
    )
    job_id = new_id()
    folder = workspace.jobs / job_id
    source = folder / "source/example"
    improvements.copy_package(workspace.directory(skill["id"]), source)
    run = folder / "results/native"
    task = run / "_harbor-tasks/agent/with/case"
    improvements.copy_package(source, task / "environment/skills/example")
    (task / "task.toml").write_text("[metadata]\n")
    write_json(run / "result.json", {"skill_name": "example"})
    options = Studio._options({}, "advanced-original-model")
    write_json(
        folder / "metadata.json",
        {
            "id": job_id,
            "skill_id": skill["id"],
            "action": "evaluate",
            "status": "succeeded",
            "provider": "nv_build",
            "options": options,
        },
    )
    write_json(folder / "request.json", {"skill_path": str(source), "options": options})
    write_json(
        folder / "result.json", {"kind": "evaluation", "execution_status": "succeeded", "raw": {"run_dir": str(run)}}
    )
    signals = {
        "schema_version": 1,
        "execution_status": "succeeded",
        "dataset_digest": "dataset",
        "evidence_issues": [],
        "agents": {
            "agent": {
                "execution_status": "succeeded",
                "cases": [
                    {
                        "entry_id": "case",
                        "split": "discovery",
                        "attempts": [
                            {
                                "trial_id": "trial",
                                "findings": [{"id": "failed-helper", "score": 0, "metric": "helper_contract"}],
                            }
                        ],
                        "baseline_attempts": [],
                    }
                ],
            }
        },
    }
    monkeypatch.setattr("skillevaluator.tier3.improvement_signals.load_signals", lambda *_args, **_kwargs: signals)
    monkeypatch.setattr("skillevaluator.reporting.dashboard_improvements._source_matches", lambda *_args: True)
    return workspace, skill, job_id, source, signals


def prepare(retained, mode, **configuration):
    workspace, skill, discovery_id, original, _signals = retained
    folder = workspace.jobs / new_id()
    source = folder / "source/example"
    improvements.copy_package(original, source)
    request = {
        "skill_id": skill["id"],
        "discovery_job_id": discovery_id,
        "options": {"model": "unused-model"},
        "improvement": {"mode": mode, "include_agent_observations": False, **configuration},
    }
    result = improvements.prepare(workspace, request, source, folder)
    write_json(
        folder / "metadata.json",
        {"id": folder.name, "skill_id": skill["id"], "action": "improve", "status": "succeeded"},
    )
    write_json(folder / "request.json", request)
    write_json(folder / "result.json", result)
    return folder, result


def test_readiness_checks_current_identity_and_counts_only_discovery(retained):
    workspace, skill, job_id, _source, signals = retained
    signals["agents"]["agent"]["cases"].append(
        {
            "entry_id": "heldout",
            "split": "holdout",
            "attempts": [{"findings": [{"id": "private", "score": 0}]}],
            "baseline_attempts": [],
        }
    )
    ready = improvements.readiness(workspace, job_id, skill["id"])
    assert ready["eligible"] and ready["signals_summary"] == {"cases": 1, "findings": 1, "observations": 0}
    current = workspace.directory(skill["id"])
    original_mode = current.stat().st_mode
    current.chmod(original_mode ^ 0o010)
    assert not improvements.readiness(workspace, job_id, skill["id"])["source_matches"]
    current.chmod(original_mode)
    other = workspace.sample()
    assert not improvements.readiness(workspace, job_id, other["id"])["eligible"]
    cases = [
        {
            **skill["cases"][0],
            "authoring_status": "qualified",
            "unresolved_requirements": ["Need a trusted expected count"],
        }
    ]
    with pytest.raises(ValueError, match="Resolve"):
        workspace.save_cases(skill["id"], cases)
    cases[0]["unresolved_requirements"] = []
    workspace.save_cases(skill["id"], cases)
    assert workspace.describe(skill["id"])["cases"][0]["authoring_status"] == "qualified"
    assert not improvements.readiness(workspace, job_id, skill["id"])["source_matches"]


def test_native_reviewed_proposal_imports_whole_candidate_and_freezes_fixtures(retained):
    workspace, skill, discovery_id, source, _signals = retained
    original = snapshot_tree(source, include_generated=True)
    _export, exported = prepare(retained, "export")
    proposal = {
        "source_digest": exported["receipt"]["source_digest"],
        "edits": [
            {
                "path": "SKILL.md",
                "old_text": "broken helper",
                "new_text": "repaired helper",
                "reason": "Use the repaired helper",
                "evidence_ids": ["failed-helper"],
            },
            {
                "path": "scripts/helper.py",
                "old_text": "return 0",
                "new_text": "return len(text.split())",
                "reason": "Count actual words",
                "evidence_ids": ["failed-helper"],
            },
        ],
    }
    folder, result = prepare(retained, "deterministic", edits=proposal)
    candidate = workspace.describe(result["candidate_id"])
    assert result["receipt"]["changed_files"] == ["SKILL.md", "scripts/helper.py"]
    assert candidate["parent_id"] == skill["id"] and candidate["proposal_job_id"] == folder.name
    assert candidate["discovery_job_id"] == discovery_id
    package = workspace.directory(candidate["id"])
    assert (package / "scripts/helper.py").stat().st_mode & 0o777 == 0o755
    assert (package / "evals/files/__pycache__/fixture.pyc").read_bytes() == b"frozen fixture bytes"
    assert snapshot_tree(source, include_generated=True) == original
    with zipfile.ZipFile(io.BytesIO(workspace.export(candidate["id"]))) as archive:
        assert archive.read("evals/files/incoming/SKILL.md") == b"A legitimate input skill fixture.\n"
        assert archive.read("evals/files/__pycache__/fixture.pyc") == b"frozen fixture bytes"
    other = workspace.sample()
    with pytest.raises(ValueError, match="does not belong"):
        improvements.validation_reference(workspace, folder.name, other["id"], discovery_id)
    original_metadata = workspace.jobs / discovery_id / "metadata.json"
    write_json(original_metadata, {**json.loads(original_metadata.read_text()), "status": "failed"})
    with pytest.raises(ValueError, match="complete original"):
        improvements.validation_reference(workspace, folder.name, candidate["id"], discovery_id)
    (package / "scripts/helper.py").write_text("A later unrelated edit.\n")
    with pytest.raises(ValueError, match="changed after"):
        improvements.validation_reference(workspace, folder.name, candidate["id"], discovery_id)


def test_native_noop_keeps_original_without_importing_identical_candidate(retained):
    workspace, _skill, _discovery, _source, signals = retained
    signals["agents"]["agent"]["cases"][0]["attempts"][0]["findings"] = []
    count = len(workspace.list())
    _folder, result = prepare(retained, "llm")
    assert result["receipt"]["status"] == "unchanged"
    assert result["receipt"]["editor_invoked"] is False
    assert result["candidate_id"] is None and len(workspace.list()) == count


@pytest.mark.parametrize(
    "findings,observations,include,requires_key",
    [(0, 0, False, False), (0, 1, False, False), (0, 1, True, True), (1, 0, False, True)],
)
def test_noop_dispatch_needs_a_key_only_when_editor_has_feedback(
    retained, monkeypatch, findings, observations, include, requires_key
):
    from skillevaluator.studio.jobs import PROVIDER_KEYS

    for name in PROVIDER_KEYS.values():
        monkeypatch.delenv(name, raising=False)
    workspace, skill, discovery_id, _source, signals = retained
    attempt = signals["agents"]["agent"]["cases"][0]["attempts"][0]
    attempt["findings"] = attempt["findings"][:findings]
    attempt["agent_observations"] = [{"id": "observation", "text": "Inspect the helper"}] * observations
    dispatched = []

    def stop_dispatch(*_args, **_kwargs):
        dispatched.append(True)
        raise OSError("Controlled dispatch interception")

    monkeypatch.setattr("skillevaluator.studio.jobs.subprocess.Popen", stop_dispatch)
    service = Studio(workspace.root)
    with pytest.raises(ValueError, match="provider key" if requires_key else "could not start"):
        service.start(
            {
                "action": "improve",
                "skill_id": skill["id"],
                "discovery_job_id": discovery_id,
                "improvement": {"mode": "llm", "include_agent_observations": include},
            }
        )
    assert bool(dispatched) is not requires_key


def test_candidate_preflight_rejects_changed_or_missing_installed_verifier(retained):
    from skillevaluator.tier3.harbor.adapter import _copy_custom_grader, _copy_verifier, _evaluator_tests_dir

    workspace, skill, job_id, source, _signals = retained
    run = improvements.evaluation_run(workspace, job_id, skill_id=skill["id"])
    task = run / "_harbor-tasks/agent/with/case"
    _copy_verifier(task)
    _copy_custom_grader(task, source, "default")
    improvements.require_current_verifier(run, source)
    verifier = _evaluator_tests_dir(task) / "eval.py"
    original = verifier.read_bytes()
    verifier.write_bytes(original + b"\n# older or changed evaluator\n")
    with pytest.raises(ValueError, match="Evaluate the original skill again"):
        improvements.require_current_verifier(run, source)
    verifier.write_bytes(original)
    (_evaluator_tests_dir(task) / "deterministic_checks.py").unlink()
    with pytest.raises(ValueError, match="Evaluate the original skill again"):
        improvements.require_current_verifier(run, source)


def test_managed_native_run_never_follows_an_outside_pointer(retained, tmp_path):
    workspace, skill, job_id, _source, _signals = retained
    outside = tmp_path / "outside"
    outside.mkdir()
    write_json(outside / "result.json", {"must_not_be_read": True})
    write_json(workspace.jobs / job_id / "result.json", {"kind": "evaluation", "raw": {"run_dir": str(outside)}})
    with pytest.raises(ValueError, match="contained native"):
        improvements.evaluation_run(workspace, job_id, skill_id=skill["id"])


def test_candidate_evaluation_replays_advanced_model_without_changing_provider(retained, monkeypatch):
    workspace, _skill, discovery_id, _source, _signals = retained
    _folder, exported = prepare(retained, "export")
    edits = {
        "source_digest": exported["receipt"]["source_digest"],
        "edits": [
            {
                "path": "SKILL.md",
                "old_text": "broken helper",
                "new_text": "fixed helper",
                "reason": "Fix observed helper",
                "evidence_ids": ["failed-helper"],
            }
        ],
    }
    proposal_folder, result = prepare(retained, "deterministic", edits=edits)
    studio = Studio(workspace.root)
    studio.configure(
        {"name": "nv_build", "model": "a-different-connection-default", "api_key": "synthetic-never-dispatched"}
    )
    monkeypatch.setattr(Studio, "capabilities", lambda _self: {"tier3": True, "docker": {"available": True}})
    captured = []

    def stop_dispatch(command, **_kwargs):
        captured.append(json.loads((Path(command[-1]) / "request.json").read_text()))
        raise OSError("Controlled dispatch interception")

    from pathlib import Path

    monkeypatch.setattr("skillevaluator.studio.jobs.subprocess.Popen", stop_dispatch)
    with pytest.raises(ValueError, match="could not start"):
        studio.start({"action": "evaluate", "skill_id": result["candidate_id"], "validation_of": proposal_folder.name})
    assert captured[0]["options"]["model"] == "advanced-original-model"
    assert captured[0]["before_job_id"] == discovery_id
    assert captured[0]["validation_of"] == proposal_folder.name
