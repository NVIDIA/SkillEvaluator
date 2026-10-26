# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate boundary tests; provider and full evaluation runs are separate integration checks."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from skillevaluator.tier3.improve import discovery_packet, prepare_improvement


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    skill = tmp_path / "original" / "example"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("Use broken command.\n")
    (skill / "evals" / "evals.json").write_text('{"private_holdout": "SECRET"}\n')
    (skill / "evals" / "grader.py").write_text("assert True\n")
    run = tmp_path / "run"
    task = run / "_harbor-tasks" / "opencode" / "with" / "case"
    retained = task / "environment" / "skills" / skill.name
    shutil.copytree(skill, retained, ignore=shutil.ignore_patterns("evals"))
    (task / "task.toml").write_text("[metadata]\n")
    (run / "result.json").write_text(json.dumps({"skill_name": skill.name}))
    signals = {
        "schema_version": 1,
        "execution_status": "succeeded",
        "dataset_digest": "dataset",
        "agents": {
            "agent": {
                "execution_status": "succeeded",
                "model": "test-model",
                "aggregate": {"secret_holdout_average": 0.1},
                "cases": [
                    {
                        "entry_id": "discovery",
                        "split": "discovery",
                        "attempts": [
                            {
                                "trial_id": "trial",
                                "scores": {"accuracy": 0},
                                "findings": [
                                    {
                                        "id": "agent:case:trial:accuracy",
                                        "score": 0,
                                        "reasons": ["Wrong command result"],
                                    },
                                    {"id": "passing", "score": 1},
                                ],
                            }
                        ],
                        "baseline_attempts": [
                            {"scores": {"accuracy": 0}, "findings": [{"id": "baseline", "score": 0}]}
                        ],
                    },
                    {"entry_id": "SECRET", "split": "holdout", "attempts": [], "baseline_attempts": []},
                ],
            }
        },
    }
    monkeypatch.setattr("skillevaluator.tier3.improvement_signals.load_signals", lambda _, **_kwargs: signals)
    return skill, run, signals


def proposal_for(tmp_path, skill, run, **edit_changes):
    exported = prepare_improvement(skill, run, tmp_path / "export")
    edit = {
        "path": "SKILL.md",
        "old_text": "broken",
        "new_text": "correct",
        "reason": "Repair the command behind the observed incorrect result",
        "evidence_ids": ["agent:case:trial:accuracy"],
        **edit_changes,
    }
    proposal = {"source_digest": exported["source_digest"], "edits": [edit]}
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(proposal))
    return path


def test_deterministic_candidate_preserves_source_and_evaluation(tmp_path, workspace):
    skill, run, _ = workspace
    proposal = proposal_for(tmp_path, skill, run)
    receipt = prepare_improvement(skill, run, tmp_path / "edit", mode="deterministic", edits_path=proposal)
    candidate = Path(receipt["candidate_path"])
    assert candidate.name == skill.name
    assert (candidate / "SKILL.md").read_text() == "Use correct command.\n"
    assert (skill / "SKILL.md").read_text() == "Use broken command.\n"
    assert (candidate / "evals" / "evals.json").read_bytes() == (skill / "evals" / "evals.json").read_bytes()
    assert receipt["source_tree_before"] == receipt["source_tree_after"]
    assert receipt["evals_digest"] == receipt["candidate_evals_digest"]
    assert receipt["status"] == "candidate_created"
    assert receipt["comparison_status"] == "not_evaluated"
    assert receipt["evaluated_source_verified"] is True
    assert "--harbor-keep-jobs" in receipt["next_step"]
    assert "-Use broken command." in (tmp_path / "edit" / "changes.diff").read_text()


@pytest.mark.parametrize("split", ["holdout", "heldout", "held-out"])
def test_export_excludes_holdout_aggregates_and_frozen_files(tmp_path, workspace, split):
    skill, run, signals = workspace
    signals["agents"]["agent"]["cases"][1]["split"] = split
    receipt = prepare_improvement(skill, run, tmp_path / "export")
    assert not receipt["candidate_ready"]
    assert not (Path(receipt["candidate_path"]) / "evals").exists()
    text = (tmp_path / "export" / "evidence.json").read_text()
    assert "SECRET" not in text and "secret_holdout_average" not in text
    cases = discovery_packet(signals)["agents"]["agent"]["cases"]
    assert len(cases) == 1
    assert [f["id"] for f in cases[0]["attempts"][0]["findings"]] == ["agent:case:trial:accuracy"]
    assert "findings" not in cases[0]["baseline_attempts"][0]


@pytest.mark.parametrize(
    "edit_changes,match",
    [
        ({"path": "../SKILL.md"}, "included"),
        ({"path": "evals/grader.py"}, "included"),
        ({"old_text": "missing"}, "exactly once"),
        ({"evidence_ids": ["invented"]}, "existing failed"),
    ],
)
def test_invalid_proposal_fails_without_changing_content(tmp_path, workspace, edit_changes, match):
    skill, run, _ = workspace
    proposal = proposal_for(tmp_path, skill, run, **edit_changes)
    with pytest.raises(ValueError, match=match):
        prepare_improvement(skill, run, tmp_path / "edit", mode="deterministic", edits_path=proposal)
    receipt = json.loads((tmp_path / "edit" / "receipt.json").read_text())
    assert receipt["status"] == "failed" and not receipt["candidate_ready"]
    assert (Path(receipt["candidate_path"]) / "SKILL.md").read_bytes() == (skill / "SKILL.md").read_bytes()


def test_all_edits_validate_before_any_write(tmp_path, workspace):
    skill, run, _ = workspace
    path = proposal_for(tmp_path, skill, run)
    proposal = json.loads(path.read_text())
    proposal["edits"].append({**proposal["edits"][0], "old_text": "missing"})
    path.write_text(json.dumps(proposal))
    with pytest.raises(ValueError, match="exactly once"):
        prepare_improvement(skill, run, tmp_path / "edit", mode="deterministic", edits_path=path)
    assert (tmp_path / "edit" / "candidate" / "example" / "SKILL.md").read_bytes() == (skill / "SKILL.md").read_bytes()


def test_stale_run_and_stale_proposal_are_rejected(tmp_path, workspace):
    skill, run, _ = workspace
    path = proposal_for(tmp_path, skill, run)
    proposal = json.loads(path.read_text())
    proposal["source_digest"] = "old-version"
    path.write_text(json.dumps(proposal))
    with pytest.raises(ValueError, match="source_digest"):
        prepare_improvement(skill, run, tmp_path / "stale-proposal", mode="deterministic", edits_path=path)
    (skill / "SKILL.md").write_text("A new skill version\n")
    with pytest.raises(ValueError, match="retained evaluated skill"):
        prepare_improvement(skill, run, tmp_path / "stale-run", mode="llm")
    exported = prepare_improvement(skill, run, tmp_path / "unverified-export")
    assert exported["evaluated_source_verified"] is False


@pytest.mark.parametrize("mode", ["llm", "agent-command"])
def test_no_failed_findings_skips_editor_calls(tmp_path, workspace, monkeypatch, mode):
    skill, run, signals = workspace
    signals["agents"]["agent"]["cases"][0]["attempts"][0]["findings"] = []
    signals["agents"]["agent"]["cases"][0]["attempts"][0]["agent_observations"] = [
        {"id": "agent:case:trial:agent_observation", "text": "I worked around a broken helper."}
    ]

    def unexpected(*args, **kwargs):
        pytest.fail("Editor must not run without failed discovery evidence")

    monkeypatch.setattr("skillevaluator.inference.client.LLMClient", unexpected)
    monkeypatch.setattr("skillevaluator.tier3.improve._run_proposal_command", unexpected)
    receipt = prepare_improvement(
        skill, run, tmp_path / "edit", mode=mode, agent_command=["unused"] if mode == "agent-command" else None
    )
    assert receipt["status"] == "unchanged"
    assert receipt["changed_files"] == []
    if mode == "agent-command":
        assert receipt["agent_command_execution"]["invoked"] is False


def test_real_local_proposal_command_protocol(tmp_path, workspace):
    skill, run, _ = workspace
    command = """import json, os, sys
from pathlib import Path
instructions = sys.stdin.read()
assert 'SECRET' not in instructions
assert not Path('evals').exists()
packet = json.loads(Path(os.environ['SKILL_IMPROVER_EVIDENCE']).read_text())
proposal = {'source_digest': packet['source_digest'], 'edits': [{'path': 'SKILL.md',
'old_text': 'broken', 'new_text': 'correct', 'reason': 'Repair command',
'evidence_ids': ['agent:case:trial:accuracy']}]}
Path(os.environ['SKILL_IMPROVER_EDITS']).write_text(json.dumps(proposal))
print('proposal written')
"""
    receipt = prepare_improvement(
        skill, run, tmp_path / "edit", mode="agent-command", agent_command=[sys.executable, "-c", command], timeout=10
    )
    assert receipt["status"] == "candidate_created"
    assert (tmp_path / "edit" / "agent-output.log").read_text().strip() == "proposal written"
    assert receipt["agent_command_execution"] == {
        "boundary": "local_host",
        "sandboxed": False,
        "access": ["host_filesystem", "inherited_environment"],
        "invoked": True,
    }


def test_direct_command_edits_are_rejected(tmp_path, workspace):
    skill, run, _ = workspace
    command = """import json, os, sys
from pathlib import Path
sys.stdin.read()
packet = json.loads(Path(os.environ['SKILL_IMPROVER_EVIDENCE']).read_text())
Path('SKILL.md').write_text('unauthorized direct edit')
Path(os.environ['SKILL_IMPROVER_EDITS']).write_text(json.dumps({'source_digest': packet['source_digest'], 'edits': []}))
"""
    with pytest.raises(ValueError, match="direct edits"):
        prepare_improvement(
            skill,
            run,
            tmp_path / "edit",
            mode="agent-command",
            agent_command=[sys.executable, "-c", command],
            timeout=10,
        )
    receipt = json.loads((tmp_path / "edit" / "receipt.json").read_text())
    assert not receipt["candidate_ready"]
    assert (skill / "SKILL.md").read_text() == "Use broken command.\n"


def test_incomplete_execution_and_symlinks_are_rejected(tmp_path, workspace):
    skill, run, signals = workspace
    signals["agents"]["agent"]["execution_status"] = "failed"
    with pytest.raises(ValueError, match="complete successful"):
        prepare_improvement(skill, run, tmp_path / "failed", mode="llm")
    (skill / "linked.md").symlink_to(skill / "SKILL.md")
    with pytest.raises(ValueError, match="regular files"):
        prepare_improvement(skill, run, tmp_path / "linked")


def test_context_failure_still_writes_failed_receipt(tmp_path, workspace):
    skill, run, _ = workspace
    (skill / "SKILL.md").write_bytes(b"x" * (128 * 1024 + 1))
    with pytest.raises(ValueError, match="context limit"):
        prepare_improvement(skill, run, tmp_path / "oversized")
    receipt = json.loads((tmp_path / "oversized" / "receipt.json").read_text())
    assert receipt["status"] == "failed" and not receipt["candidate_ready"]


def test_fixture_tasks_do_not_participate_in_source_verification(tmp_path, workspace):
    skill, run, _ = workspace
    fixture = run / "_harbor-tasks/opencode/with/case/environment/input/project"
    other_skill = fixture / "environment" / "skills" / skill.name
    other_skill.mkdir(parents=True)
    (fixture / "task.toml").write_text("[metadata]\n")
    (other_skill / "SKILL.md").write_text("A different skill copied as task input\n")
    proposal = proposal_for(tmp_path, skill, run)
    receipt = prepare_improvement(skill, run, tmp_path / "edit", mode="deterministic", edits_path=proposal)
    assert receipt["evaluated_source_verified"] is True


@pytest.mark.skipif(sys.platform == "win32", reason="Executable permission bits are POSIX semantics")
def test_changed_script_permissions_do_not_match_evaluated_source(tmp_path, workspace):
    skill, run, _ = workspace
    retained = run / "_harbor-tasks/opencode/with/case/environment/skills" / skill.name
    for root in (skill, retained):
        script = root / "run.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
    (skill / "run.sh").chmod(0o644)
    exported = prepare_improvement(skill, run, tmp_path / "export")
    assert exported["evaluated_source_verified"] is False
    with pytest.raises(ValueError, match="does not match the retained evaluated skill"):
        prepare_improvement(skill, run, tmp_path / "llm", mode="llm")


def test_workspace_skill_cannot_impersonate_evaluated_target(tmp_path, workspace):
    skill, run, _ = workspace
    (run / "result.json").write_text(json.dumps({"skill_name": "different-evaluated-skill"}))
    exported = prepare_improvement(skill, run, tmp_path / "export")
    assert exported["evaluated_source_verified"] is False
    assert exported["evaluated_skill_name"] == "different-evaluated-skill"
    with pytest.raises(ValueError, match="retained evaluated skill"):
        prepare_improvement(skill, run, tmp_path / "llm", mode="llm")


def test_existing_generated_results_are_not_copied_or_exposed(tmp_path, workspace):
    skill, run, _ = workspace
    results = skill / "evals" / "results"
    results.mkdir()
    (results / "latest").symlink_to(run, target_is_directory=True)
    (results / "old-output.txt").write_text("OLD_RUN_PRIVATE_DATA")
    proposal = proposal_for(tmp_path, skill, run)
    receipt = prepare_improvement(skill, run, tmp_path / "edit", mode="deterministic", edits_path=proposal)
    assert not (Path(receipt["candidate_path"]) / "evals" / "results").exists()
    assert "OLD_RUN_PRIVATE_DATA" not in (tmp_path / "edit" / "instructions.md").read_text()
    assert (results / "latest").is_symlink()
    assert receipt["evals_digest"] == receipt["candidate_evals_digest"]


def test_legacy_evaluation_files_are_hidden_frozen_and_not_editable(tmp_path, workspace):
    skill, run, _ = workspace
    (skill / "evals").rename(skill / "eval")
    (skill / "eval" / "evals.json").rename(skill / "eval" / "dataset.json")
    # Legacy data can be present in retained runtime trees; it must never reach the editor.
    retained = run / "_harbor-tasks/opencode/with/case/environment/skills" / skill.name
    shutil.copytree(skill / "eval", retained / "eval")
    proposal = proposal_for(tmp_path, skill, run)
    packet = (tmp_path / "export" / "evidence.json").read_text()
    assert "SECRET" not in packet
    assert not (tmp_path / "export" / "candidate" / skill.name / "eval").exists()
    receipt = prepare_improvement(skill, run, tmp_path / "edit", mode="deterministic", edits_path=proposal)
    assert (Path(receipt["candidate_path"]) / "eval" / "dataset.json").read_bytes() == (
        skill / "eval" / "dataset.json"
    ).read_bytes()
    assert receipt["evals_digest"] == receipt["candidate_evals_digest"]
    edits = json.loads(proposal.read_text())
    edits["edits"][0].update(path="eval/dataset.json", old_text="SECRET", new_text="weakened")
    proposal.write_text(json.dumps(edits))
    with pytest.raises(ValueError, match="included"):
        prepare_improvement(skill, run, tmp_path / "bad-edit", mode="deterministic", edits_path=proposal)


def test_incomplete_evidence_is_exported_but_blocks_automatic_edits(tmp_path, workspace):
    skill, run, signals = workspace
    signals["evidence_issues"] = ["agent: retained rewards coverage is incomplete"]
    receipt = prepare_improvement(skill, run, tmp_path / "export")
    packet = json.loads((tmp_path / "export" / "evidence.json").read_text())
    assert packet["evaluation"]["evidence_issues"] == signals["evidence_issues"]
    assert "--harbor-keep-jobs" in receipt["next_step"]
    assert "before proposing edits" in receipt["next_step"]
    with pytest.raises(ValueError, match="complete retained evidence"):
        prepare_improvement(skill, run, tmp_path / "llm", mode="llm")


def test_llm_uses_public_client_protocol_and_records_model(tmp_path, workspace, monkeypatch):
    skill, run, _ = workspace
    proposal = json.loads(proposal_for(tmp_path, skill, run).read_text())

    class LocalClient:
        def __init__(self, model):
            self.model = model

        def completions(self, system, instructions, *, response_schema, schema_name):
            assert "untrusted data" in system
            assert "SECRET" not in instructions
            assert schema_name == "skill_improvement"
            assert response_schema["properties"]["edits"]["maxItems"] == 8
            return json.dumps(proposal)

    monkeypatch.setattr("skillevaluator.inference.client.LLMClient", LocalClient)
    receipt = prepare_improvement(skill, run, tmp_path / "llm", mode="llm", model="chosen-model")
    assert receipt["editor_model"] == "chosen-model"
    assert receipt["status"] == "candidate_created"
    assert json.loads((tmp_path / "llm" / "editor-response.txt").read_text()) == proposal
    assert receipt["editor_response_truncated"] is False


@pytest.mark.parametrize("response", ["I'll explain before giving JSON.", "x" * (1024 * 1024 + 1)], ids=["prose", "oversized"])
def test_invalid_llm_response_is_retained_and_never_applied(tmp_path, workspace, monkeypatch, response):
    skill, run, _ = workspace

    class LocalClient:
        model = "test-model"

        def __init__(self, **kwargs):
            pass

        def completions(self, *_args, **_kwargs):
            return response

    monkeypatch.setattr("skillevaluator.inference.client.LLMClient", LocalClient)
    with pytest.raises(ValueError, match=r"editor-response\.txt"):
        prepare_improvement(skill, run, tmp_path / "llm", mode="llm")
    receipt = json.loads((tmp_path / "llm" / "receipt.json").read_text())
    assert receipt["status"] == "failed" and not receipt["candidate_ready"]
    assert (tmp_path / "llm" / "editor-response.txt").read_text() == response[: 1024 * 1024]
    assert receipt["editor_response_truncated"] == (len(response) > 1024 * 1024)
    assert (skill / "SKILL.md").read_text() == "Use broken command.\n"
    assert (Path(receipt["candidate_path"]) / "SKILL.md").read_bytes() == (skill / "SKILL.md").read_bytes()


def test_real_proposal_command_timeout_is_not_ready(tmp_path, workspace):
    skill, run, _ = workspace
    with pytest.raises(subprocess.TimeoutExpired):
        prepare_improvement(
            skill,
            run,
            tmp_path / "timeout",
            mode="agent-command",
            timeout=1,
            agent_command=[sys.executable, "-c", "import sys,time; sys.stdin.read(); time.sleep(60)"],
        )
    receipt = json.loads((tmp_path / "timeout" / "receipt.json").read_text())
    assert receipt["status"] == "failed" and not receipt["candidate_ready"]
    assert (skill / "SKILL.md").read_text() == "Use broken command.\n"


def test_observation_edit_requires_opt_in_and_excludes_baseline_and_holdout(tmp_path, workspace):
    from skillevaluator.tier3.improve import _evidence_ids

    skill, run, signals = workspace
    cases = signals["agents"]["agent"]["cases"]
    observation = {"id": "agent:case:trial:agent_observation", "text": "The helper is broken; I worked around it."}
    cases[0]["attempts"][0].update(findings=[], agent_observations=[observation])
    cases[0]["baseline_attempts"][0]["agent_observations"] = [{**observation, "id": "baseline-observation"}]
    cases[1]["attempts"] = [{"findings": [], "agent_observations": [{**observation, "id": "holdout-observation"}]}]
    assert _evidence_ids(discovery_packet(signals)) == set()
    packet = discovery_packet(signals, include_agent_observations=True)
    assert _evidence_ids(packet) == {observation["id"]}
    assert "baseline-observation" not in json.dumps(packet) and "holdout-observation" not in json.dumps(packet)
    exported = prepare_improvement(skill, run, tmp_path / "export", include_agent_observations=True)
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({"source_digest": exported["source_digest"], "edits": [{
        "path": "SKILL.md", "old_text": "broken", "new_text": "correct", "reason": "Source confirms the report",
        "evidence_ids": [observation["id"]],
    }]}))
    with pytest.raises(ValueError, match="explicitly enabled"):
        prepare_improvement(skill, run, tmp_path / "disabled", mode="deterministic", edits_path=proposal)
    result = prepare_improvement(skill, run, tmp_path / "enabled", mode="deterministic", edits_path=proposal,
                                 include_agent_observations=True)
    assert result["status"] == "candidate_created" and result["include_agent_observations"] is True
    assert "Unverified" in result["observation_policy"]


@pytest.mark.parametrize("evaluation_dir", ["evals", "eval"])
def test_frozen_evaluation_cache_fixtures_are_copied_and_hashed(tmp_path, workspace, evaluation_dir):
    from skillevaluator.tier3.improve import _digest, snapshot_tree

    skill, run, _ = workspace
    fixture = skill / evaluation_dir / "files/discovery-case/incoming/log-summary/__pycache__/helper.pyc"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_bytes(b"\x00intentional binary evaluation fixture\xff")
    runtime_cache = skill / "__pycache__/generated.pyc"
    runtime_cache.parent.mkdir()
    runtime_cache.write_bytes(b"runtime cache")
    if evaluation_dir == "eval":
        retained = run / "_harbor-tasks/opencode/with/case/environment/skills" / skill.name
        shutil.copytree(skill / "eval", retained / "eval")
    proposal = proposal_for(tmp_path, skill, run)
    receipt = prepare_improvement(skill, run, tmp_path / "edit", mode="deterministic", edits_path=proposal)
    candidate = Path(receipt["candidate_path"])
    assert (candidate / fixture.relative_to(skill)).read_bytes() == fixture.read_bytes()
    assert not (candidate / "__pycache__").exists()
    assert "helper.pyc" not in (tmp_path / "edit/evidence.json").read_text()
    exact = snapshot_tree(skill, include_generated=True)
    assert receipt["evals_digest"] == _digest({
        name: entry for name, entry in exact.items() if name.split("/")[0] in {"eval", "evals"}
    })
    assert receipt["evals_digest"] == receipt["candidate_evals_digest"]
