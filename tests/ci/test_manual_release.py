# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

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
    assert any("inspect-indexes" in step.get("run", "") for step in jobs["prepare"]["steps"])


@pytest.mark.parametrize("name", ["release", "release-tier1-image", "release-kitmaker"])
def test_release_workflows_scope_permissions_and_credentials(name):
    source = (ROOT / ".github" / "workflows" / f"{name}.yml").read_text(encoding="utf-8")
    workflow = yaml.load(source, Loader=yaml.BaseLoader)
    assert workflow["permissions"] == {}
    assert re.search(r"\bsecrets\s*(?:\.|\[|:)", source, re.IGNORECASE) is None
    checkouts = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert checkouts
    assert all(step["with"]["persist-credentials"] == "false" for step in checkouts)


def test_release_dispatches_serialize_without_replacing_pending_runs():
    source = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    workflow = yaml.load(source, Loader=yaml.BaseLoader)
    assert workflow["concurrency"] == {
        "group": "skillevaluator-release",
        "cancel-in-progress": "false",
        "queue": "max",
    }


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
        "head_sha": SHA,
        "head_repository": {"id": 42, "full_name": "NVIDIA/SkillEvaluator"},
        "repository": {"id": 42, "full_name": "NVIDIA/SkillEvaluator"},
        "display_title": f"Validate release · SkillEvaluator {VERSION}",
        "status": "completed",
        "conclusion": "success",
        **overrides,
    }


def test_validation_is_the_only_phase_that_builds_and_smoke_tests_release_assets():
    workflow = yaml.load((ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert {"validation_run", "validation_manifest_sha256"} <= inputs.keys()
    jobs = workflow["jobs"]
    assert jobs["prepare"]["if"] == "inputs.phase == 'Validate release'"
    assert "prepare" not in jobs["stage"]["needs"]
    steps = jobs["stage"]["steps"]
    retain = next(i for i, step in enumerate(steps) if "upload-artifact@" in step.get("uses", ""))
    stage = next(i for i, step in enumerate(steps) if "github_release.py stage" in step.get("run", ""))
    assert retain < stage


@pytest.mark.parametrize("run_ref, manifest", [("", "a" * 64), ("123", "a" * 64), ("123-2", "A" * 64)])
def test_new_release_requires_a_validation_attempt_and_manifest_digest(
    release, monkeypatch, tmp_path, run_ref, manifest
):
    monkeypatch.setattr(release, "tag_target", lambda _: None)
    monkeypatch.setattr(release, "get_release", lambda _: None)
    origin = Mock()
    monkeypatch.setattr(release, "api", origin)
    with pytest.raises(ValueError, match="successful Validate release"):
        release.acquire_publication(VERSION, SHA, tmp_path / "assets", run_ref, manifest)
    origin.assert_not_called()


@pytest.mark.parametrize(
    "metadata",
    [
        {"conclusion": "failure"},
        {"status": "in_progress"},
        {"display_title": f"Publish release · SkillEvaluator {VERSION}"},
        {"display_title": "Validate release · SkillEvaluator 9.9.9"},
        {"run_attempt": 3},
        {"id": 456},
        {"path": ".github/workflows/other.yml"},
        {"event": "pull_request"},
        {"head_branch": "feature"},
        {"head_repository": {"full_name": "fork/SkillEvaluator"}},
        {"repository": {"full_name": "fork/SkillEvaluator"}},
    ],
)
def test_new_release_rejects_unqualified_validation_attempts(release, monkeypatch, tmp_path, metadata):
    monkeypatch.setattr(release, "tag_target", lambda _: None)
    monkeypatch.setattr(release, "get_release", lambda _: None)
    origin = Mock(return_value=release_run(**metadata))
    monkeypatch.setattr(release, "api", origin)
    download = Mock()
    monkeypatch.setattr(release.subprocess, "run", download)
    with pytest.raises(ValueError, match="successful Validate release"):
        release.acquire_publication(VERSION, SHA, tmp_path / "assets", "123-2", "a" * 64)
    origin.assert_called_once_with("actions/runs/123/attempts/2")
    download.assert_not_called()


@pytest.mark.parametrize("retained", [False, True], ids=["missing", "expired"])
def test_missing_or_expired_validation_artifact_requires_revalidation(release, monkeypatch, tmp_path, retained):
    monkeypatch.setattr(release, "tag_target", lambda _: None)
    monkeypatch.setattr(release, "get_release", lambda _: None)
    artifact = {
        "id": 789,
        "name": f"validated-release-{VERSION}-123-2",
        "expired": True,
        "size_in_bytes": 1000,
        "digest": "sha256:" + "a" * 64,
        "workflow_run": {
            "id": 123,
            "repository_id": 42,
            "head_repository_id": 42,
            "head_branch": "main",
            "head_sha": SHA,
        },
    }
    artifacts = [artifact] if retained else []
    requests = Mock(side_effect=[release_run(), {"total_count": len(artifacts), "artifacts": artifacts}])
    monkeypatch.setattr(release, "api", requests)
    download = Mock()
    monkeypatch.setattr(release.subprocess, "run", download)
    with pytest.raises(ValueError, match="Run Validate release again"):
        release.acquire_publication(VERSION, SHA, tmp_path / "assets", "123-2", "a" * 64)
    assert [call.args for call in requests.call_args_list] == [
        ("actions/runs/123/attempts/2",),
        (f"actions/runs/123/artifacts?name=validated-release-{VERSION}-123-2",),
    ]
    download.assert_not_called()


@pytest.fixture
def draft_recovery(release, monkeypatch):
    def configure(state="empty", fault=None):
        contents = {
            "artifact-manifest.json": b"original manifest",
            "SHA256SUMS.txt": b"original checksums",
            f"skillevaluator-{VERSION}-py3-none-any.whl": b"original wheel",
            f"skillevaluator-{VERSION}.tar.gz": b"original sdist",
            f"skillevaluator-ci-{VERSION}.zip": b"original CI bundle",
        }
        staged = [{"name": name, "state": "uploaded"} for name in contents] if state != "empty" else []
        if state == "interrupted":
            staged[0].update(state="starter", size=0)
        draft = {"draft": True, "body": "<!-- skillevaluator-stage: 123-2 -->", "assets": staged}
        monkeypatch.setattr(release, "tag_target", lambda _: SHA)
        monkeypatch.setattr(release, "get_release", lambda _: draft)

        archive = io.BytesIO()
        with ZipFile(archive, "w") as files:
            for name, data in contents.items():
                files.writestr(name, data)
            if fault == "extra-file":
                files.writestr("unexpected.bin", b"extra file")
        zipped = archive.getvalue()
        artifact = {
            "id": 789,
            "name": f"product-release-{VERSION}-123-2",
            "expired": fault == "expired",
            "size_in_bytes": len(zipped) + (fault == "size"),
            "digest": "sha256:" + hashlib.sha256(b"different bytes" if fault == "digest" else zipped).hexdigest(),
            "workflow_run": {
                "id": 456 if fault == "lineage" else 123,
                "repository_id": 42,
                "head_repository_id": 42,
                "head_branch": "main",
                "head_sha": SHA,
            },
        }
        artifacts = {
            "missing": [],
            "duplicate": [artifact, {**artifact, "id": 790}],
            "name": [{**artifact, "name": f"product-release-{VERSION}-123-1"}],
        }.get(fault, [artifact])
        requests = Mock(
            side_effect=[
                release_run(conclusion="failure", display_title=f"Publish release · SkillEvaluator {VERSION}"),
                {"total_count": len(artifacts), "artifacts": artifacts},
            ]
        )
        monkeypatch.setattr(release, "api", requests)

        def download(command, *, check, stdout):
            assert command == ["gh", "api", "repos/NVIDIA/SkillEvaluator/actions/artifacts/789/zip"]
            assert check
            assert fault in (None, "size", "digest", "extra-file")
            stdout.write(zipped)

        monkeypatch.setattr(release.subprocess, "run", download)
        legacy_download = Mock(side_effect=AssertionError("Recover draft archives by numeric artifact ID"))
        monkeypatch.setattr(release, "gh", legacy_download)
        return contents, requests

    return configure


@pytest.mark.parametrize("draft_state", ["empty", "interrupted", "complete"])
def test_publish_recovers_the_original_draft_without_a_validation_artifact(
    release, draft_recovery, tmp_path, draft_state
):
    contents, requests = draft_recovery(state=draft_state)
    output = tmp_path / "assets"
    assert release.acquire_publication(VERSION, SHA, output, "", "") is False
    assert {path.name: path.read_bytes() for path in output.iterdir()} == contents
    assert [call.args for call in requests.call_args_list] == [
        ("actions/runs/123/attempts/2",),
        (f"actions/runs/123/artifacts?name=product-release-{VERSION}-123-2",),
    ]


@pytest.mark.parametrize(
    "fault",
    ["missing", "duplicate", "name", "expired", "lineage", "size", "digest", "extra-file"],
)
def test_draft_rejects_an_unverified_recovery_archive(release, draft_recovery, tmp_path, fault):
    draft_recovery(fault=fault)
    output = tmp_path / "assets"
    with pytest.raises(ValueError, match=r"original draft artifact.*Reconcile the draft before retrying"):
        release.acquire_publication(VERSION, SHA, output, "", "")
    assert not output.exists()


def test_publish_recovers_immutable_bytes_without_a_validation_artifact(release, monkeypatch, tmp_path):
    names = release.expected_filenames(VERSION)
    monkeypatch.setattr(release, "tag_target", lambda _: SHA)
    monkeypatch.setattr(
        release,
        "get_release",
        lambda _: {
            "draft": False,
            "immutable": True,
            "assets": [{"name": name, "state": "uploaded"} for name in names],
        },
    )
    monkeypatch.setattr(release, "download_asset", lambda asset, path: path.write_text(asset["name"], encoding="utf-8"))
    checkout_run = Mock()
    monkeypatch.setattr(release, "api", checkout_run)
    output = tmp_path / "assets"

    assert release.acquire_publication(VERSION, SHA, output, "", "") is False
    assert {path.name: path.read_text(encoding="utf-8") for path in output.iterdir()} == {name: name for name in names}
    checkout_run.assert_not_called()


def test_draft_cannot_recover_artifacts_from_a_pull_request(release, monkeypatch, tmp_path):
    draft = {"draft": True, "body": "<!-- skillevaluator-stage: 123-2 -->", "assets": []}
    monkeypatch.setattr(release, "tag_target", lambda _: SHA)
    monkeypatch.setattr(release, "get_release", lambda _: draft)
    monkeypatch.setattr(release, "api", Mock(return_value=release_run(event="pull_request")))
    run_download = Mock()
    monkeypatch.setattr(release.subprocess, "run", run_download)
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
