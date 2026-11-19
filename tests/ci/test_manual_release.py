# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
VERSION = "2.3.4"
SHA = "1" * 40


def test_release_is_only_dispatched_manually_and_defaults_to_validation():
    workflow = yaml.load((ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    phase = workflow["on"]["workflow_dispatch"]["inputs"]["phase"]
    assert phase["default"] == "Validate release"
    assert phase["options"] == ["Validate release", "Build CI image and bundle", "Publish release"]
    jobs = workflow["jobs"]
    assert "workflow_dispatch" in jobs["preflight"]["if"]
    assert "refs/heads/main" in jobs["preflight"]["if"]
    assert jobs["image"]["uses"] == "./.github/workflows/release-tier1-image.yml"
    assert jobs["stage"]["environment"] == "release-github"
    assert "Publish release" in jobs["stage"]["if"]
    assert jobs["kitmaker"]["needs"] == ["preflight", "stage"]
    assert jobs["finalize"]["needs"] == ["preflight", "verify-indexes"]
    (preflight,) = [step for step in jobs["prepare"]["steps"] if "inspect-indexes" in step.get("run", "")]
    assert preflight["if"] == "inputs.phase == 'Validate release'"


@pytest.fixture
def release():
    spec = importlib.util.spec_from_file_location("github_release", ROOT / "ci/github_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conflicting_tag_blocks_all_release_writes(release, monkeypatch, tmp_path):
    monkeypatch.setattr(release, "tag_target", lambda _: "2" * 40)
    publish = Mock()
    monkeypatch.setattr(release, "api", publish)
    with pytest.raises(ValueError, match="points to a different commit"):
        release.stage(VERSION, SHA, tmp_path, "123-1")
    publish.assert_not_called()


def test_partial_draft_requires_the_original_run_artifact(release, monkeypatch, tmp_path):
    monkeypatch.setattr(release, "tag_target", lambda _: SHA)
    monkeypatch.setattr(
        release,
        "get_release",
        lambda _: {"draft": True, "body": "<!-- skillevaluator-stage: 123-2 -->", "assets": []},
    )
    download = Mock()
    monkeypatch.setattr(release, "gh", download)
    monkeypatch.setattr(release, "api", Mock(return_value=release_run()))
    assert release.restore(VERSION, SHA, tmp_path)
    download.assert_called_once_with(
        "run", "download", "123", "--name", f"product-release-{VERSION}-123-2", "--dir", str(tmp_path)
    )


def test_draft_with_conflicting_bytes_is_never_published(release, monkeypatch, tmp_path):
    tag = f"v{VERSION}"
    (tmp_path / "SHA256SUMS.txt").write_bytes(b"verified checksums")
    monkeypatch.setattr(release, "tag_target", lambda _: SHA)
    monkeypatch.setattr(release, "verify_local", Mock())
    monkeypatch.setattr(release, "release_notes", lambda *_: "Frozen release notes")
    monkeypatch.setattr(release.product_release, "inspect_public_indexes", Mock())
    monkeypatch.setattr(
        release,
        "get_release",
        lambda selected: {
            "tag_name": selected,
            "draft": True,
            "assets": [{"name": "SHA256SUMS.txt", "id": 123, "state": "uploaded"}],
        },
    )
    monkeypatch.setattr(release.subprocess, "run", Mock(return_value=Mock(stdout=b"different bytes")))
    write_api = Mock()
    monkeypatch.setattr(release, "api", write_api)
    with pytest.raises(ValueError, match="differs from the verified release bytes"):
        release.stage(VERSION, SHA, tmp_path, "123-1")
    write_api.assert_not_called()
    assert release.get_release(tag)["draft"] is True


def test_public_verification_rejects_a_mutable_release(release, monkeypatch, tmp_path):
    monkeypatch.setattr(release, "tag_target", lambda _: SHA)
    monkeypatch.setattr(release, "get_release", lambda _: {"draft": False, "immutable": False})
    download = Mock()
    monkeypatch.setattr(release, "urlopen", download)
    with pytest.raises(ValueError, match="immutable, published"):
        release.verify_public(VERSION, SHA, tmp_path)
    download.assert_not_called()


def test_interrupted_github_upload_recovers_original_artifact(release, monkeypatch, tmp_path):
    names = sorted(release.expected_filenames(VERSION))
    assets = [{"name": name, "state": "uploaded"} for name in names]
    assets[0].update(state="starter", size=0)
    draft = {"draft": True, "body": "<!-- skillevaluator-stage: 123-2 -->", "assets": assets}
    monkeypatch.setattr(release, "tag_target", lambda _: SHA)
    monkeypatch.setattr(release, "get_release", lambda _: draft)
    download = Mock()
    monkeypatch.setattr(release, "gh", download)
    monkeypatch.setattr(release, "api", Mock(return_value=release_run()))
    output = tmp_path / "release"
    assert release.restore(VERSION, SHA, output)
    download.assert_called_once_with(
        "run", "download", "123", "--name", f"product-release-{VERSION}-123-2", "--dir", str(output)
    )


def test_registry_revision_must_match_the_reviewed_digest(release, monkeypatch, tmp_path):
    repository = "ghcr.io/nvidia/skillevaluator-tier1"
    (tmp_path / "artifact-manifest.json").write_text(
        json.dumps({"image": f"{repository}@sha256:{'a' * 64}", "image_source_commit": SHA})
    )
    monkeypatch.setattr(release.product_release, "verify_assets", Mock())
    resolve = Mock(return_value=f"{repository}@sha256:{'b' * 64}")
    monkeypatch.setattr(release.release_image, "resolve_digest", resolve)
    with pytest.raises(ValueError, match="does not match the reviewed release digest"):
        release.verify_local(VERSION, SHA, tmp_path)
    resolve.assert_called_once_with(repository, SHA)


def release_run(**overrides):
    return {
        "id": 123,
        "run_attempt": 2,
        "head_branch": "main",
        "event": "workflow_dispatch",
        "path": ".github/workflows/release.yml",
        "head_repository": {"full_name": "NVIDIA/SkillEvaluator"},
        **overrides,
    }


def test_complete_draft_restores_original_actions_bytes(release, monkeypatch, tmp_path):
    draft = {
        "draft": True,
        "body": "<!-- skillevaluator-stage: 123-2 -->",
        "assets": [{"name": name, "state": "uploaded"} for name in release.expected_filenames(VERSION)],
    }
    monkeypatch.setattr(release, "tag_target", lambda _: SHA)
    monkeypatch.setattr(release, "get_release", lambda _: draft)
    run = Mock(return_value=release_run())
    run_download = Mock()
    release_download = Mock()
    monkeypatch.setattr(release, "api", run)
    monkeypatch.setattr(release, "gh", run_download)
    monkeypatch.setattr(release, "download_asset", release_download)
    assert release.restore(VERSION, SHA, tmp_path / "staged")
    run.assert_called_once_with("actions/runs/123/attempts/2")
    run_download.assert_called_once()
    release_download.assert_not_called()


def test_draft_cannot_recover_artifacts_from_a_pull_request(release, monkeypatch, tmp_path):
    draft = {"draft": True, "body": "<!-- skillevaluator-stage: 123-2 -->", "assets": []}
    monkeypatch.setattr(release, "tag_target", lambda _: SHA)
    monkeypatch.setattr(release, "get_release", lambda _: draft)
    monkeypatch.setattr(release, "api", Mock(return_value=release_run(event="pull_request")))
    run_download = Mock()
    monkeypatch.setattr(release, "gh", run_download)
    with pytest.raises(ValueError, match="upstream manual release run"):
        release.restore(VERSION, SHA, tmp_path / "staged")
    run_download.assert_not_called()


def test_initial_publication_checks_package_conflicts_before_tagging(release, monkeypatch, tmp_path):
    monkeypatch.setattr(release, "tag_target", lambda _: None)
    monkeypatch.setattr(release, "get_release", lambda _: None)
    monkeypatch.setattr(release, "verify_local", Mock(return_value={}))
    monkeypatch.setattr(release, "release_notes", lambda *_: "Frozen release notes")
    inspect_indexes = Mock(side_effect=ValueError("Package indexes are inconsistent"))
    monkeypatch.setattr(release.product_release, "inspect_public_indexes", inspect_indexes)
    writes = Mock()
    monkeypatch.setattr(release, "api", writes)
    with pytest.raises(ValueError, match="Package indexes are inconsistent"):
        release.stage(VERSION, SHA, tmp_path, "123-2")
    inspect_indexes.assert_called_once_with(VERSION, {})
    writes.assert_not_called()


def test_immutable_resume_reaches_kitmaker_even_when_an_index_is_pending(release, monkeypatch, tmp_path):
    monkeypatch.setattr(release, "tag_target", lambda _: SHA)
    monkeypatch.setattr(release, "get_release", lambda _: {"draft": False, "immutable": True})
    monkeypatch.setattr(release, "verify_local", Mock(return_value={}))
    monkeypatch.setattr(release, "verify_staged", Mock())
    monkeypatch.setattr(release, "release_notes", lambda *_: "Frozen release notes")
    inspect_indexes = Mock(side_effect=ValueError("An artifact is present on only one index"))
    monkeypatch.setattr(release.product_release, "inspect_public_indexes", inspect_indexes)
    release.stage(VERSION, SHA, tmp_path, "123-2")
    inspect_indexes.assert_not_called()
