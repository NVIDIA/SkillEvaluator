# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
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
    assert phase["options"] == [
        "Validate release",
        "Validate and publish",
        "Full release",
        "Build CI image and bundle",
        "Publish release",
    ]
    jobs = workflow["jobs"]
    assert "workflow_dispatch" in jobs["preflight"]["if"]
    assert "refs/heads/main" in jobs["preflight"]["if"]
    reviewed = next(step for step in jobs["preflight"]["steps"] if step.get("id") == "reviewed")
    assert reviewed["env"]["PROMOTE_STABLE"] == "${{ inputs.promote_stable }}"
    assert jobs["image"]["uses"] == "./.github/workflows/release-tier1-image.yml"
    assert jobs["stage"]["environment"] == "release-github"
    assert "Publish release" in jobs["stage"]["if"]
    assert jobs["kitmaker"]["needs"] == ["release-source", "stage"]
    assert jobs["finalize"]["needs"] == ["release-source", "verify-indexes"]
    assert any("inspect-indexes" in step.get("run", "") for step in jobs["prepare"]["steps"])


@pytest.mark.parametrize("publish_testpypi", ["false", "true"])
@pytest.mark.parametrize("promote_stable", ["false", "true"])
@pytest.mark.parametrize(
    ("phase", "publishes"),
    [
        ("Validate release", False),
        ("Build CI image and bundle", False),
        ("Validate and publish", True),
        ("Full release", True),
        ("Publish release", True),
    ],
)
def test_preflight_validates_stable_promotion_phase(phase, publishes, promote_stable, publish_testpypi, tmp_path):
    workflow = yaml.load((ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    step = next(step for step in workflow["jobs"]["preflight"]["steps"] if step.get("id") == "reviewed")
    command = tmp_path / "python3"
    command.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$PREFLIGHT_ARGS"\n', encoding="utf-8")
    command.chmod(0o755)
    args_file = tmp_path / "args"
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "PHASE": phase,
            "PROMOTE_STABLE": promote_stable,
            "PUBLISH_TESTPYPI": publish_testpypi,
            "VERSION": VERSION,
            "SOURCE_SHA": SHA,
            "GITHUB_SHA": SHA,
            "PREFLIGHT_ARGS": str(args_file),
            "GITHUB_OUTPUT": str(output),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    if publish_testpypi == "true" and phase != "Validate release":
        assert result.returncode == 1
        assert "TestPyPI requires the Validate release phase." in result.stderr
        assert not args_file.exists() and not output.exists()
    elif promote_stable == "true" and not publishes:
        assert result.returncode == 1
        assert "Stable promotion requires Full release, Validate and publish, or Publish release." in result.stderr
        assert not args_file.exists() and not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        arguments = args_file.read_text(encoding="utf-8").splitlines()
        assert arguments[arguments.index("--phase") + 1] == phase
        assert output.read_text(encoding="utf-8") == f"version={VERSION}\nsource_sha={SHA}\n"


def test_testpypi_upload_is_opt_in_and_uses_only_validated_distributions():
    workflow = yaml.load((ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    testpypi = workflow["on"]["workflow_dispatch"]["inputs"]["publish_testpypi"]
    assert testpypi["type"] == "boolean"
    assert testpypi["default"] == "false"

    jobs = workflow["jobs"]
    publish = jobs["testpypi"]
    assert publish["needs"] == ["release-source", "prepare"]
    assert publish["if"] == "inputs.publish_testpypi && inputs.phase == 'Validate release'"
    assert publish["environment"] == "pypi-test"
    assert publish["permissions"] == {"id-token": "write"}
    assert not any("checkout@" in step.get("uses", "") for step in publish["steps"])
    download = next(step for step in publish["steps"] if "download-artifact@" in step.get("uses", ""))
    assert download["with"]["name"] == (
        "validated-release-${{ needs.release-source.outputs.version }}-${{ needs.prepare.outputs.run }}"
    )
    command = publish["steps"][-1]["run"]
    assert "--trusted-publishing always" in command
    assert "--publish-url https://test.pypi.org/legacy/" in command
    assert "--check-url https://test.pypi.org/simple/" in command
    assert '"$assets/skillevaluator-$VERSION-py3-none-any.whl"' in command
    assert '"$assets/skillevaluator-$VERSION.tar.gz"' in command

    verify = jobs["verify-testpypi"]
    assert verify["needs"] == ["release-source", "prepare", "testpypi"]
    assert verify["permissions"] == {"contents": "read"}
    commands = "\n".join(step.get("run", "") for step in verify["steps"])
    assert "ci/product_release.py verify-testpypi" in commands
    assert "--default-index https://pypi.org/simple" in commands
    assert 'ci/smoke_distribution.py "$VERSION"' in commands


def test_validate_checks_release_metadata_with_the_current_workflow_from_the_reviewed_checkout():
    workflow = yaml.load((ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["prepare"]
    steps = job["steps"]
    metadata = [i for i, step in enumerate(steps) if "github_release.py verify-release-metadata" in step.get("run", "")]
    assert len(metadata) == 1
    (metadata_index,) = metadata
    assert "../release-control/ci/github_release.py" in steps[metadata_index]["run"]
    assert job["defaults"]["run"]["working-directory"] == "source"
    assert job["permissions"] == {"actions": "read", "contents": "read"}
    checkout_refs = {
        step["with"]["path"]: step["with"]["ref"]
        for step in steps[:metadata_index]
        if step.get("uses", "").startswith("actions/checkout@")
    }
    assert checkout_refs == {
        "release-control": "${{ github.workflow_sha }}",
        "source": "${{ needs.release-source.outputs.source_sha }}",
    }
    verify = next(i for i, step in enumerate(steps) if "product_release.py verify-assets" in step.get("run", ""))
    retain = next(i for i, step in enumerate(steps) if "upload-artifact@" in step.get("uses", ""))
    assert verify < metadata_index < retain


@pytest.mark.parametrize("phase", ["Validate release", "Validate and publish", "Full release"])
def test_prepare_summarizes_the_exact_validated_release_before_publication(phase, tmp_path):
    workflow = yaml.load((ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["prepare"]["steps"]
    summary = next(step for step in steps if "$GITHUB_STEP_SUMMARY" in step.get("run", ""))
    output = tmp_path / "summary"
    subprocess.run(
        ["bash", "-e", "-c", summary["run"]],
        env={
            **os.environ,
            "PHASE": phase,
            "VERSION": VERSION,
            "SOURCE_SHA": SHA,
            "VALIDATION_RUN": "42-1",
            "VALIDATION_ARTIFACT_ID": "123",
            "MANIFEST_SHA256": "b" * 64,
            "GITHUB_STEP_SUMMARY": str(output),
        },
        check=True,
    )
    contents = output.read_text(encoding="utf-8")
    assert f"| `version` | `{VERSION}` |" in contents
    assert f"| `source_sha` | `{SHA}` |" in contents
    assert "| `validation_run` | `42-1` |" in contents
    assert f"| `validation_manifest_sha256` | `{'b' * 64}` |" in contents
    if phase == "Validate release":
        assert contents.startswith("### Publish release inputs\n")
        assert "Validation artifact ID" not in contents
    else:
        assert contents.startswith("### Validated release artifacts\n")
        assert "| Validation artifact ID | `123` |" in contents
    assert "if" not in summary
    assert summary["env"]["VALIDATION_ARTIFACT_ID"] == "${{ steps.retained.outputs.artifact-id }}"


@pytest.mark.parametrize("name", ["release", "release-tier1-image", "release-kitmaker"])
def test_release_workflows_scope_permissions_and_credentials(name):
    source = (ROOT / ".github" / "workflows" / f"{name}.yml").read_text(encoding="utf-8")
    workflow = yaml.load(source, Loader=yaml.BaseLoader)
    assert workflow["permissions"] == {}
    secrets = re.findall(r"\bsecrets\.([A-Z_]+)", source)
    if name == "release":
        assert sorted(secrets) == [
            "RELEASE_BUNDLE_APP_PRIVATE_KEY",
            "RELEASE_BUNDLE_APP_PRIVATE_KEY",
            "STABLE_CHANNEL_APP_PRIVATE_KEY",
        ]
        assert workflow["jobs"]["bundle-pr"]["environment"] == "release-bundle-pr"
        assert workflow["jobs"]["bundle-configuration"]["environment"] == "release-bundle-pr"
        assert workflow["jobs"]["promote-stable"]["secrets"] == {
            "STABLE_CHANNEL_APP_PRIVATE_KEY": "${{ secrets.STABLE_CHANNEL_APP_PRIVATE_KEY }}"
        }
    else:
        assert not secrets
    assert not re.search(r"\bsecrets\s*(?:\[|:\s*inherit)", source, re.IGNORECASE)
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


def test_publish_rechecks_the_public_image_before_tagging(release, monkeypatch, tmp_path):
    repository = "ghcr.io/nvidia/skillevaluator-tier1"
    (tmp_path / "artifact-manifest.json").write_text(
        json.dumps({"image": f"{repository}@sha256:{'a' * 64}", "image_source_commit": SHA})
    )
    monkeypatch.setattr(release, "tag_target", lambda _: None)
    monkeypatch.setattr(release, "get_release", lambda _: None)
    monkeypatch.setattr(release.product_release, "verify_assets", Mock())
    resolve = Mock(return_value=f"{repository}@sha256:{'b' * 64}")
    monkeypatch.setattr(release.release_image, "resolve_digest", resolve)
    writes = Mock()
    monkeypatch.setattr(release, "api", writes)
    with pytest.raises(ValueError, match="does not match the reviewed release digest"):
        release.stage(VERSION, SHA, tmp_path, "123-2")
    resolve.assert_called_once_with(repository, SHA)
    writes.assert_not_called()


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


def test_publish_uses_the_required_validation_artifact_and_retains_recovery_bytes():
    workflow = yaml.load((ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert {"validation_run", "validation_manifest_sha256"} <= inputs.keys()
    jobs = workflow["jobs"]
    prepare = jobs["prepare"]
    assert "inputs.phase == 'Publish release'" not in prepare["if"]
    assert prepare["outputs"] == {
        "run": "${{ steps.artifact.outputs.run }}",
        "artifact_id": "${{ steps.retained.outputs.artifact-id }}",
        "artifact_digest": "${{ steps.retained.outputs.artifact-digest }}",
        "manifest_sha256": "${{ steps.verified.outputs.manifest_sha256 }}",
    }
    stage_job = jobs["stage"]
    assert stage_job["needs"] == ["release-source", "prepare"]
    assert "inputs.phase == 'Publish release' && needs.prepare.result == 'skipped'" in stage_job["if"]
    assert "needs.prepare.result == 'success'" in stage_job["if"]
    for name in ("prepare", "stage", "kitmaker", "verify-indexes", "finalize"):
        assert "!cancelled()" in jobs[name]["if"]
        for dependency in jobs[name]["needs"] if isinstance(jobs[name]["needs"], list) else [jobs[name]["needs"]]:
            assert f"needs.{dependency}.result" in jobs[name]["if"]
    acquire = next(step for step in stage_job["steps"] if step.get("id") == "acquire")
    assert acquire["env"]["VALIDATION_ARTIFACT_ID"] == "${{ needs.prepare.outputs.artifact_id }}"
    assert acquire["env"]["VALIDATION_ARTIFACT_DIGEST"] == "${{ needs.prepare.outputs.artifact_digest }}"
    assert acquire["env"]["VALIDATION_RUN"].endswith(" || needs.prepare.outputs.run }}")
    assert acquire["env"]["MANIFEST_SHA256"].endswith(" || needs.prepare.outputs.manifest_sha256 }}")
    assert '--validation-artifact-id "$VALIDATION_ARTIFACT_ID"' in acquire["run"]
    assert '--validation-artifact-digest "$VALIDATION_ARTIFACT_DIGEST"' in acquire["run"]
    steps = jobs["stage"]["steps"]
    retain = next(i for i, step in enumerate(steps) if "upload-artifact@" in step.get("uses", ""))
    stage = next(i for i, step in enumerate(steps) if "github_release.py stage" in step.get("run", ""))
    assert retain < stage


@pytest.mark.parametrize("phase", ["Publish release", "Validate and publish", "Full release"])
def test_stage_passes_same_run_artifact_options_only_for_combined_phases(phase, tmp_path):
    workflow = yaml.load((ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    step = next(step for step in workflow["jobs"]["stage"]["steps"] if step.get("id") == "acquire")
    command = tmp_path / "python3"
    command.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ACQUIRE_ARGS"\nprintf "true\\n"\n', encoding="utf-8")
    command.chmod(0o755)
    args_file = tmp_path / "args"
    output = tmp_path / "output"
    subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "PHASE": phase,
            "VERSION": VERSION,
            "SOURCE_SHA": SHA,
            "RUNNER_TEMP": str(tmp_path),
            "VALIDATION_RUN": "42-1",
            "MANIFEST_SHA256": "b" * 64,
            "VALIDATION_ARTIFACT_ID": "123",
            "VALIDATION_ARTIFACT_DIGEST": "c" * 64,
            "ACQUIRE_ARGS": str(args_file),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_RUN_ID": "100",
            "GITHUB_RUN_ATTEMPT": "2",
        },
        check=True,
    )
    arguments = args_file.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--validation-run") + 1] == "42-1"
    if phase == "Publish release":
        assert "--validation-artifact-id" not in arguments
        assert "--validation-artifact-digest" not in arguments
    else:
        assert arguments[arguments.index("--validation-artifact-id") + 1] == "123"
        assert arguments[arguments.index("--validation-artifact-digest") + 1] == "c" * 64
    assert "new_release=true" in output.read_text(encoding="utf-8")


def test_full_release_uses_an_approved_bundle_pr_and_only_promotes_after_finalization():
    workflow = yaml.load((ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    assert "needs.bundle-configuration.result == 'success'" in jobs["image"]["if"]
    bundle = jobs["bundle-pr"]
    assert bundle["needs"] == ["preflight", "image"]
    assert bundle["permissions"] == {"actions": "read", "contents": "read"}
    verify = next(i for i, step in enumerate(bundle["steps"]) if "sha256sum --check" in step.get("run", ""))
    writer = next(i for i, step in enumerate(bundle["steps"]) if step.get("id") == "writer")
    assert verify < writer
    assert jobs["bundle-merge"]["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert "bundle-pr" in jobs["bundle-merge"]["needs"]
    source = jobs["release-source"]
    assert "inputs.phase == 'Full release' && needs.bundle-merge.result == 'success'" in source["if"]
    assert jobs["promotion-candidate"]["needs"] == ["release-source", "finalize"]
    assert "needs.promotion-candidate.outputs.is_latest == 'true'" in jobs["promote-stable"]["if"]
    assert jobs["promote-stable"]["with"]["expected-release"] == "v${{ needs.release-source.outputs.version }}"


@pytest.mark.parametrize(
    ("phase", "merged", "expected"),
    [("Full release", "2" * 40, "2" * 40), ("Full release", "", None), ("Validate and publish", "", SHA)],
)
def test_full_release_cannot_fall_back_to_the_image_candidate(phase, merged, expected, tmp_path):
    workflow = yaml.load((ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
    run = workflow["jobs"]["release-source"]["steps"][0]["run"]
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-e", "-c", run],
        env={**os.environ, "PHASE": phase, "CANDIDATE_SHA": SHA, "MERGED_SHA": merged, "GITHUB_OUTPUT": str(output)},
        check=False,
    )
    if expected is None:
        assert result.returncode != 0
        assert not output.exists()
    else:
        assert result.returncode == 0
        assert output.read_text(encoding="utf-8") == f"sha={expected}\n"


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


def test_publish_rechecks_the_dated_changelog_before_tagging(release, monkeypatch, tmp_path):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(f"## {VERSION}\n\nPublished changes.\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    git = release.product_release.git
    git("init", "-q")
    git("add", "CHANGELOG.md")
    git("-c", "user.name=Release tests", "-c", "user.email=test@example.com", "commit", "-qm", "Undated changelog")
    source = git("rev-parse", "HEAD").decode().strip()
    changelog.write_text(f"## {VERSION} - 2026-11-20\n\nWorktree changes.\n", encoding="utf-8")
    (tmp_path / "artifact-manifest.json").write_text(
        json.dumps({"image": f"ghcr.io/nvidia/skillevaluator-tier1@sha256:{'a' * 64}"}), encoding="utf-8"
    )
    monkeypatch.setattr(release, "tag_target", lambda _: None)
    monkeypatch.setattr(release, "get_release", lambda _: None)
    monkeypatch.setattr(release, "verify_local", Mock(return_value={}))
    monkeypatch.setattr(release.product_release, "inspect_public_indexes", Mock())
    writes = Mock()
    monkeypatch.setattr(release, "api", writes)
    with pytest.raises(ValueError, match="no dated entry"):
        release.stage(VERSION, source, tmp_path, "123-2")
    writes.assert_not_called()


def test_immutable_resume_reaches_kitmaker_even_when_an_index_is_pending(release, monkeypatch, tmp_path):
    monkeypatch.setattr(release, "tag_target", lambda _: SHA)
    monkeypatch.setattr(release, "get_release", lambda _: {"draft": False, "immutable": True})
    monkeypatch.setattr(release, "verify_local", Mock(return_value={}))
    monkeypatch.setattr(release, "verify_staged", Mock())
    notes = Mock(side_effect=ValueError("CHANGELOG.md has no dated release entry."))
    monkeypatch.setattr(release, "release_notes", notes)
    inspect_indexes = Mock(side_effect=ValueError("An artifact is present on only one index"))
    monkeypatch.setattr(release.product_release, "inspect_public_indexes", inspect_indexes)
    release.stage(VERSION, SHA, tmp_path, "123-2")
    notes.assert_not_called()
    inspect_indexes.assert_not_called()
