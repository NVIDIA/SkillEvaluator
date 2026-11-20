# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from ci import github_release

VERSION = "2.3.4"
SOURCE = "1" * 40
WORKFLOW_COMMIT = "2" * 40
RUN = "45112"
PRODUCER_ATTEMPT = "2"
ARTIFACT_ID = "789"
PHASE = "Validate and publish"
PREFIX = "repos/NVIDIA/SkillEvaluator/"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def producer(monkeypatch):
    wheel = f"skillevaluator-{VERSION}-py3-none-any.whl"
    sdist = f"skillevaluator-{VERSION}.tar.gz"
    bundle = f"skillevaluator-ci-{VERSION}.zip"
    contents = {wheel: b"validated wheel", sdist: b"validated sdist", bundle: b"validated bundle"}

    def entry(name):
        return {"filename": name, "size": len(contents[name]), "sha256": sha256(contents[name])}

    contents["artifact-manifest.json"] = json.dumps(
        {
            "version": VERSION,
            "source_commit": SOURCE,
            "artifacts": [entry(wheel), entry(sdist)],
            "ci_bundle": entry(bundle),
        }
    ).encode()
    contents["SHA256SUMS.txt"] = "".join(f"{sha256(data)}  {name}\n" for name, data in contents.items()).encode()
    archive = io.BytesIO()
    with ZipFile(archive, "w") as files:
        for name, data in contents.items():
            files.writestr(name, data)
    zipped = archive.getvalue()
    artifact_digest = sha256(zipped)

    origin = {
        "id": int(RUN),
        "run_attempt": int(PRODUCER_ATTEMPT),
        "event": "workflow_dispatch",
        "path": ".github/workflows/release.yml",
        "head_branch": "main",
        "head_sha": WORKFLOW_COMMIT,
        "repository": {"id": 42, "full_name": "NVIDIA/SkillEvaluator"},
        "head_repository": {"id": 42, "full_name": "NVIDIA/SkillEvaluator"},
        "status": "in_progress",
        "conclusion": None,
        "display_title": f"{PHASE} · SkillEvaluator {VERSION}",
    }
    name = f"validated-release-{VERSION}-{RUN}-{PRODUCER_ATTEMPT}"
    artifact = {
        "name": name,
        "id": int(ARTIFACT_ID),
        "expired": False,
        "size_in_bytes": len(zipped),
        "digest": f"sha256:{artifact_digest}",
        "workflow_run": {
            "id": int(RUN),
            "head_branch": "main",
            "head_sha": WORKFLOW_COMMIT,
            "repository_id": 42,
            "head_repository_id": 42,
        },
    }
    attempt_path = f"actions/runs/{RUN}/attempts/{PRODUCER_ATTEMPT}"
    artifact_path = f"actions/runs/{RUN}/artifacts?name={name}"
    api = Mock(
        side_effect={attempt_path: origin, artifact_path: {"total_count": 1, "artifacts": [artifact]}}.__getitem__
    )
    producer = SimpleNamespace(
        origin=origin,
        artifact=artifact,
        digest=artifact_digest,
        zipped=zipped,
        manifest_digest=sha256(contents["artifact-manifest.json"]),
        contents=contents,
        api=api,
        attempt_path=attempt_path,
        artifact_path=artifact_path,
    )

    def download(args, *, stdout, check):
        assert args == ["gh", "api", f"{PREFIX}actions/artifacts/{ARTIFACT_ID}/zip"]
        assert check
        stdout.write(producer.zipped)

    producer.download = Mock(side_effect=download)
    monkeypatch.setenv("GITHUB_RUN_ID", RUN)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", PRODUCER_ATTEMPT)
    monkeypatch.setattr(github_release, "tag_target", lambda _: None)
    monkeypatch.setattr(github_release, "get_release", lambda _: None)
    monkeypatch.setattr(github_release, "api", api)
    monkeypatch.setattr(github_release.subprocess, "run", producer.download)
    return producer


def acquire(producer, output, **overrides):
    arguments = {
        "version": VERSION,
        "source_sha": SOURCE,
        "output": output,
        "run_ref": f"{RUN}-{PRODUCER_ATTEMPT}",
        "manifest_sha256": producer.manifest_digest,
        "phase": PHASE,
        "artifact_id": ARTIFACT_ID,
        "artifact_digest": producer.digest,
    }
    return github_release.acquire_publication(**(arguments | overrides))


@pytest.mark.parametrize(
    ("phase", "publishing_attempt", "conclusion"),
    [(PHASE, "2", None), ("Full release", "3", "failure")],
)
def test_same_run_acquires_exact_producer_bytes(
    producer, monkeypatch, tmp_path, capsys, phase, publishing_attempt, conclusion
):
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", publishing_attempt)
    producer.origin.update(
        display_title=f"{phase} · SkillEvaluator {VERSION}",
        status="completed" if conclusion else "in_progress",
        conclusion=conclusion,
    )
    output = tmp_path / "assets"
    supplied_digest = f"sha256:{producer.digest}" if conclusion else producer.digest

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "github_release.py",
            "acquire",
            "--version",
            VERSION,
            "--source-sha",
            SOURCE,
            "--assets",
            str(output),
            "--phase",
            phase,
            "--validation-run",
            f"{RUN}-{PRODUCER_ATTEMPT}",
            "--validation-manifest-sha256",
            producer.manifest_digest,
            "--validation-artifact-id",
            ARTIFACT_ID,
            "--validation-artifact-digest",
            supplied_digest,
        ],
    )

    assert github_release.main() == 0
    assert capsys.readouterr().out == "true\n"
    assert {path.name: path.read_bytes() for path in output.iterdir()} == producer.contents
    assert [call.args[0] for call in producer.api.call_args_list] == [producer.attempt_path, producer.artifact_path]
    producer.download.assert_called_once()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"artifact_id": None}, "artifact ID and SHA-256 digest"),
        ({"artifact_digest": None}, "artifact ID and SHA-256 digest"),
        ({"artifact_id": "", "artifact_digest": ""}, "artifact ID and SHA-256 digest"),
        ({"artifact_id": None, "artifact_digest": None}, "artifact ID and SHA-256 digest"),
        ({"artifact_id": "0"}, "artifact ID and SHA-256 digest"),
        ({"artifact_digest": "invalid"}, "artifact ID and SHA-256 digest"),
        ({"phase": "Publish release"}, "Validate and publish or Full release"),
        ({"run_ref": "99999-2"}, "current workflow run"),
        ({"run_ref": f"{RUN}-3"}, "current workflow run"),
        ({"manifest_sha256": ""}, "run attempt and manifest SHA-256"),
    ],
)
def test_same_run_rejects_invalid_producer_inputs(producer, tmp_path, overrides, message):
    output = tmp_path / "assets"
    with pytest.raises(ValueError, match=message):
        acquire(producer, output, **overrides)
    assert not output.exists()
    producer.api.assert_not_called()
    producer.download.assert_not_called()


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-phase",
        "wrong-version",
        "wrong-attempt",
        "wrong-workflow",
        "fork",
        "wrong-lineage",
        "wrong-name",
        "wrong-id",
        "wrong-digest",
    ],
)
def test_same_run_rejects_wrong_actions_identity_before_downloading(producer, tmp_path, mutation):
    changes = {
        "wrong-phase": (producer.origin, "display_title", f"Full release · SkillEvaluator {VERSION}"),
        "wrong-version": (producer.origin, "display_title", f"{PHASE} · SkillEvaluator 9.9.9"),
        "wrong-attempt": (producer.origin, "run_attempt", 1),
        "wrong-workflow": (producer.origin, "path", ".github/workflows/other.yml"),
        "fork": (producer.origin["head_repository"], "full_name", "someone/SkillEvaluator"),
        "wrong-lineage": (producer.artifact["workflow_run"], "head_sha", "3" * 40),
        "wrong-name": (producer.artifact, "name", f"validated-release-{VERSION}-{RUN}-3"),
        "wrong-id": (producer.artifact, "id", 900),
        "wrong-digest": (producer.artifact, "digest", "sha256:" + "f" * 64),
    }
    metadata, key, value = changes[mutation]
    metadata[key] = value
    output = tmp_path / "assets"
    with pytest.raises(ValueError, match=r"validation artifact|upstream manual release run"):
        acquire(producer, output)
    assert not output.exists()
    producer.download.assert_not_called()


@pytest.mark.parametrize("mutation", ["archive", "manifest", "source"])
def test_same_run_rejects_different_release_bytes(producer, tmp_path, mutation):
    overrides = {}
    if mutation == "archive":
        producer.zipped = b"!" + producer.zipped[1:]
    if mutation == "manifest":
        overrides["manifest_sha256"] = "c" * 64
    if mutation == "source":
        overrides["source_sha"] = "d" * 40

    with pytest.raises(ValueError, match=r"ZIP differs|manifest SHA-256 or source commit"):
        acquire(producer, tmp_path / "assets", **overrides)


@pytest.mark.parametrize("conclusion", ["success", "failure"])
def test_standalone_publish_checks_completed_validation_without_same_run_outputs(
    producer, monkeypatch, tmp_path, conclusion
):
    monkeypatch.setenv("GITHUB_RUN_ID", "9000")
    producer.origin.update(
        display_title=f"Validate release · SkillEvaluator {VERSION}", status="completed", conclusion=conclusion
    )
    output = tmp_path / "assets"
    arguments = {"phase": "Publish release", "artifact_id": None, "artifact_digest": None}
    if conclusion == "failure":
        with pytest.raises(ValueError, match="successful Validate release"):
            acquire(producer, output, **arguments)
        producer.download.assert_not_called()
    else:
        assert acquire(producer, output, **arguments) is True
        assert {path.name: path.read_bytes() for path in output.iterdir()} == producer.contents


def test_same_run_command_preserves_restore_first(producer, monkeypatch, tmp_path, capsys):
    restore = Mock(return_value=True)
    monkeypatch.setattr(github_release, "restore", restore)
    output = tmp_path / "assets"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "github_release.py",
            "acquire",
            "--version",
            VERSION,
            "--source-sha",
            SOURCE,
            "--assets",
            str(output),
            "--phase",
            PHASE,
            "--validation-artifact-id",
            "invalid",
        ],
    )

    assert github_release.main() == 0
    assert capsys.readouterr().out == "false\n"
    restore.assert_called_once_with(VERSION, SOURCE, output)
    producer.api.assert_not_called()


def test_full_release_preflight_rejects_an_existing_version_before_creating_a_bundle_pr(monkeypatch):
    def git_output(args, *, text):
        assert text
        if args == ["git", "rev-list", "--first-parent", WORKFLOW_COMMIT]:
            return f"{WORKFLOW_COMMIT}\n{SOURCE}\n"
        assert args == ["git", "show", f"{SOURCE}:pyproject.toml"]
        return f'[project]\nversion = "{VERSION}"\n'

    monkeypatch.setattr(github_release.subprocess, "check_output", git_output)
    tag_target = Mock(side_effect=[None, "3" * 40, "3" * 40])
    monkeypatch.setattr(github_release, "tag_target", tag_target)

    github_release.preflight(VERSION, SOURCE, WORKFLOW_COMMIT, "Full release")
    with pytest.raises(ValueError, match=r"already exists\. Resume Publish release"):
        github_release.preflight(VERSION, SOURCE, WORKFLOW_COMMIT, "Full release")
    with pytest.raises(ValueError, match="points to a different commit"):
        github_release.preflight(VERSION, SOURCE, WORKFLOW_COMMIT, "Publish release")
    assert [call.args for call in tag_target.call_args_list] == [(f"v{VERSION}",)] * 3
