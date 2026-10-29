# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity and provenance boundaries; actual retained-run/browser checks are separate."""

import json
import shutil

import pytest

from skillevaluator.reporting.dashboard_improvements import _runtime_digest, _source_matches, load_skill_change
from skillevaluator.tier3.improve import _digest, snapshot_tree


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def proposal(tmp_path):
    candidate = tmp_path / "candidate" / "example"
    candidate.mkdir(parents=True)
    (candidate / "SKILL.md").write_text("Use the repaired helper.\n")
    _write(candidate / "evals" / "evals.json", {"split": "discovery"})
    _write(
        tmp_path / "receipt.json",
        {
            "status": "candidate_created",
            "evaluated_skill_name": "example",
            "source_digest": "source-id",
            "evaluated_source_verified": True,
            "candidate_digest": _digest(snapshot_tree(candidate, include_generated=True)),
            "changed_files": ["SKILL.md"],
        },
    )
    _write(
        tmp_path / "evidence.json",
        {
            "source_digest": "source-id",
            "source_verified": True,
            "skill_files": {"SKILL.md": "Use the broken helper.\n"},
            "evaluation": {
                "agents": {
                    "agent": {
                        "cases": [
                            {
                                "entry_id": "discovery",
                                "attempts": [
                                    {
                                        "trial_id": "trial",
                                        "findings": [{"id": "finding", "score": 0, "reasons": ["Helper failed"]}],
                                    }
                                ],
                            }
                        ]
                    }
                }
            },
        },
    )
    _write(
        tmp_path / "edits.json",
        {
            "source_digest": "source-id",
            "edits": [
                {
                    "path": "SKILL.md",
                    "old_text": "broken",
                    "new_text": "repaired",
                    "reason": "Repair observed failure",
                    "evidence_ids": ["finding"],
                }
            ],
        },
    )
    return tmp_path


def test_proposal_links_finding_and_exact_diff_without_claiming_improvement(proposal):
    data = load_skill_change(proposal)
    assert data["issues"] == []
    assert data["findings"]["finding"]["trial"] == "trial"
    assert "-Use the broken helper.\n+Use the repaired helper.\n" in data["diff"]
    assert data["association"] == "Not evaluated" and data["comparison"] is None
    # Stale source or an invented citation must not silently look like an evaluated edit.
    (proposal / "candidate/example/SKILL.md").write_text("An unrelated candidate.\n")
    assert "differs from its proposal receipt" in load_skill_change(proposal)["issues"][0]
    edits = json.loads((proposal / "edits.json").read_text())
    edits["edits"][0]["evidence_ids"] = ["invented"]
    _write(proposal / "edits.json", edits)
    with pytest.raises(ValueError, match="existing failed discovery finding IDs"):
        load_skill_change(proposal)


def test_every_retained_task_must_match_runtime_source_including_modes(proposal, tmp_path):
    candidate = proposal / "candidate/example"
    expected = _runtime_digest(candidate)
    run = tmp_path / "run"
    _write(run / "result.json", {"skill_name": "example"})
    sources = []
    for case in ("one", "two"):
        task = run / "_harbor-tasks/agent/with" / case
        source = task / "environment/skills/example"
        shutil.copytree(candidate, source)
        (task / "task.toml").write_text("[metadata]\n")
        _write(task / "tests/entry.json", {"id": case})
        # Different heldout evaluation files do not change the runtime skill identity.
        _write(source / "evals/evals.json", {"split": "heldout", "case": case})
        sources.append(source)
    cases = {("agent", "one"), ("agent", "two")}
    assert _source_matches(run, "example", expected, cases)
    last_task = sources[-1].parents[2]
    (last_task / "task.toml").unlink()
    assert not _source_matches(run, "example", expected, cases)
    (last_task / "task.toml").write_text("[metadata]\n")
    _write(last_task / "tests/entry.json", {"id": "one"})
    assert not _source_matches(run, "example", expected, cases)
    _write(last_task / "tests/entry.json", {"id": "two"})
    changed = sources[-1] / "SKILL.md"
    changed.chmod(changed.stat().st_mode | 0o111)
    assert not _source_matches(run, "example", expected, cases)
    shutil.rmtree(last_task)
    assert not _source_matches(run, "example", expected, cases)


def test_linked_artifact_and_embedded_paths_are_not_followed(proposal, tmp_path):
    receipt = json.loads((proposal / "receipt.json").read_text())
    receipt["candidate_path"] = "/must/not/be/read"
    receipt["original_run_dir"] = "/also/not/read"
    _write(proposal / "receipt.json", receipt)
    assert load_skill_change(proposal)["issues"] == []
    outside = tmp_path / "outside.json"
    (proposal / "evidence.json").rename(outside)
    (proposal / "evidence.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink or external"):
        load_skill_change(proposal)


def test_failed_receipt_without_edits_or_evidence_remains_visible(proposal):
    receipt = json.loads((proposal / "receipt.json").read_text())
    receipt.update(status="failed", error="Editor returned invalid JSON", candidate_ready=False)
    _write(proposal / "receipt.json", receipt)
    (proposal / "edits.json").unlink()
    (proposal / "evidence.json").unlink()
    data = load_skill_change(proposal)
    assert data["receipt"]["status"] == "failed"
    assert data["edits"] == [] and data["comparison"] is None


@pytest.mark.parametrize("edits", [None, False, 0, "", {}])
def test_malformed_saved_edits_are_rejected_before_rendering(proposal, edits):
    _write(proposal / "edits.json", {"edits": edits})
    with pytest.raises(ValueError, match="edits list"):
        load_skill_change(proposal)


def test_actual_dashboard_form_opens_proposal_and_rejects_invalid_reload(proposal):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string(
        "from skillevaluator.reporting.dashboard_app import main\nmain([])", default_timeout=60
    ).run()
    app.text_input(key="change_proposal").set_value(str(proposal))
    app.button[0].click().run()
    assert not app.exception
    assert any("candidate_created" in text.value for text in app.text)
    assert "Use the repaired helper" in app.code[0].value
    assert any("No validation pair" in caption.value for caption in app.caption)
    app.text_input(key="change_proposal").set_value(str(proposal / "missing"))
    app.button[0].click().run()
    assert not app.exception and app.error
    assert not app.code  # Do not retain stale candidate success after a failed reload.
