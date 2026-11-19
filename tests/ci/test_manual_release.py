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
