# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from tests.ci.test_kitmaker_release import MemoryReceipts, Portal, check_absent

from ci import github_release
from ci import kitmaker_release as kitmaker

VERSION = "2.3.4"
SOURCE = "a" * 40
TAG = "v2.3.4-assets"
BASE = f"https://github.com/NVIDIA/SkillEvaluator/releases/download/{TAG}/"


@pytest.fixture
def recovery(monkeypatch):
    wheel = {"filename": "skillevaluator-2.3.4-py3-none-any.whl", "size": 5, "sha256": "c" * 64}
    sdist = {"filename": "skillevaluator-2.3.4.tar.gz", "size": 6, "sha256": "d" * 64}
    bundle = {"filename": "skillevaluator-ci-2.3.4.zip", "size": 7, "sha256": "e" * 64}
    manifest = {
        "version": VERSION,
        "source_commit": SOURCE,
        "asset_tag": TAG,
        "artifacts": [wheel, sdist],
        "ci_bundle": bundle,
    }
    raw = json.dumps(manifest).encode()
    digest = hashlib.sha256(raw).hexdigest()
    sums = "".join(f"{item['sha256']}  {item['filename']}\n" for item in (wheel, sdist, bundle))
    sums += f"{digest}  artifact-manifest.json\n"
    release = {"immutable": True, "draft": False, "prerelease": False, "published_at": "2026-11-23T00:00:00Z"}
    names = [item["filename"] for item in (wheel, sdist, bundle)] + ["artifact-manifest.json", "SHA256SUMS.txt"]
    responses = {}
    for tag in (f"v{VERSION}", TAG):
        responses[f"/releases/tags/{tag}"] = {
            **release,
            "tag_name": tag,
            "assets": [{"name": name, "state": "uploaded"} for name in names] if tag == TAG else [],
        }
        responses[f"/git/ref/tags/{tag}"] = {"ref": f"refs/tags/{tag}", "object": {"type": "commit", "sha": SOURCE}}
    downloads = {BASE + "artifact-manifest.json": raw, BASE + "SHA256SUMS.txt": sums.encode()}
    calls = []
    monkeypatch.setenv("GH_TOKEN", "test-read-token")
    monkeypatch.setattr(
        kitmaker,
        "request_json",
        lambda url, **_: responses[url.removeprefix("https://api.github.com/repos/NVIDIA/SkillEvaluator")],
    )
    monkeypatch.setattr(kitmaker, "request_bytes", lambda url, **_: downloads[url])
    monkeypatch.setattr(kitmaker, "verify_download", lambda url, artifact: calls.append((url, artifact)))
    return digest, responses, downloads, calls


def test_recovery_verifies_both_tag_identities_and_downloads_all_artifacts(recovery):
    digest, _, _, calls = recovery
    candidate = kitmaker.verify_github_release(VERSION, SOURCE, digest, TAG)
    assert candidate.version == VERSION
    assert candidate.asset_base == BASE
    assert {url for url, _ in calls} == {
        BASE + name
        for name in (
            "skillevaluator-ci-2.3.4.zip",
            "skillevaluator-2.3.4-py3-none-any.whl",
            "skillevaluator-2.3.4.tar.gz",
        )
    }
    portal, receipts = Portal(), MemoryReceipts()
    kitmaker.release_to_kitmaker("dry-run", candidate, portal, receipts, check_absent)
    kitmaker.release_to_kitmaker("publish", candidate, portal, receipts, check_absent)
    assert all(url.startswith(BASE) for urls, _ in portal.submitted for url in urls)
    assert receipts.data["binding"]["asset_tag"] == TAG


@pytest.mark.parametrize(
    "tag", ["v2.3.5-assets", "v2.3.4-assets/extra", "../v2.3.4", "v2.3.4-assets?x=1", "https://other"]
)
def test_recovery_rejects_other_versions_or_arbitrary_locations(tag):
    with pytest.raises(kitmaker.ReleaseError, match="Asset tag"):
        kitmaker.checked_asset_tag(VERSION, tag)


@pytest.mark.parametrize("tag", [f"v{VERSION}", TAG])
@pytest.mark.parametrize(
    "field,value", [("immutable", False), ("draft", True), ("prerelease", True), ("published_at", None)]
)
def test_recovery_requires_both_releases_to_be_published_and_immutable(recovery, tag, field, value):
    digest, responses, _, calls = recovery
    responses[f"/releases/tags/{tag}"][field] = value
    with pytest.raises(kitmaker.ReleaseError, match="immutable GitHub release"):
        kitmaker.verify_github_release(VERSION, SOURCE, digest, TAG)
    assert calls == []


@pytest.mark.parametrize("tag", [f"v{VERSION}", TAG])
def test_recovery_rejects_a_tag_at_a_different_source(recovery, tag):
    digest, responses, _, calls = recovery
    responses[f"/git/ref/tags/{tag}"]["object"]["sha"] = "f" * 40
    with pytest.raises(kitmaker.ReleaseError, match="approved source"):
        kitmaker.verify_github_release(VERSION, SOURCE, digest, TAG)
    assert calls == []


@pytest.mark.parametrize("fault", ["missing", "duplicate", "incomplete", "extra"])
def test_recovery_rejects_an_incomplete_five_asset_release(recovery, fault):
    digest, responses, _, calls = recovery
    assets = responses[f"/releases/tags/{TAG}"]["assets"]
    if fault == "missing":
        assets.pop()
    elif fault == "duplicate":
        assets[-1] = dict(assets[0])
    elif fault == "incomplete":
        assets[-1]["state"] = "new"
    else:
        assets.append({"name": "extra.txt", "state": "uploaded"})
    with pytest.raises(kitmaker.ReleaseError, match="all five"):
        kitmaker.verify_github_release(VERSION, SOURCE, digest, TAG)
    assert calls == []


def test_recovery_checksum_file_cannot_substitute_a_different_bundle(recovery):
    digest, _, downloads, _ = recovery
    key = BASE + "SHA256SUMS.txt"
    downloads[key] = downloads[key].replace(b"e" * 64, b"f" * 64)
    with pytest.raises(kitmaker.ReleaseError, match="exact contents"):
        kitmaker.verify_github_release(VERSION, SOURCE, digest, TAG)


def test_recovery_requires_manifest_to_bind_the_asset_tag(recovery):
    _, _, downloads, _ = recovery
    raw = json.loads(downloads[BASE + "artifact-manifest.json"])
    del raw["asset_tag"]
    raw = json.dumps(raw).encode()
    with pytest.raises(kitmaker.ReleaseError, match="bind the approved asset tag"):
        kitmaker.parse_manifest(VERSION, SOURCE, hashlib.sha256(raw).hexdigest(), raw, TAG)


def test_recovery_receipt_cannot_be_reused_for_the_product_download_location(recovery):
    digest, _, _, _ = recovery
    candidate = kitmaker.verify_github_release(VERSION, SOURCE, digest, TAG)
    receipts, portal = MemoryReceipts(), Portal()
    kitmaker.release_to_kitmaker("dry-run", candidate, portal, receipts, check_absent)
    with pytest.raises(kitmaker.ReleaseError, match="different release"):
        kitmaker.release_to_kitmaker("publish", replace(candidate, asset_tag=""), portal, receipts, check_absent)
    assert len(portal.submitted) == 1


@pytest.mark.parametrize("complete", [False, True])
def test_index_verification_never_connects_to_vault_or_submits_to_kitmaker(recovery, monkeypatch, complete, capsys):
    digest, _, _, _ = recovery
    candidate = kitmaker.verify_github_release(VERSION, SOURCE, digest, TAG)
    monkeypatch.setattr(kitmaker, "check_indexes", lambda _: set(candidate.artifacts) if complete else set())
    monkeypatch.setattr(kitmaker.Vault, "connect", lambda: pytest.fail("Read-only verification must not access Vault"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kitmaker_release",
            "verify",
            "--version",
            VERSION,
            "--source-sha",
            SOURCE,
            "--manifest-sha256",
            digest,
            "--asset-tag",
            TAG,
        ],
    )
    assert kitmaker.main() == (0 if complete else 1)
    output = capsys.readouterr()
    assert ("Verified the exact" in output.out) is complete


def test_recovery_preflight_preserves_normal_tag_guard(monkeypatch):
    monkeypatch.setattr(
        github_release.subprocess,
        "check_output",
        lambda args, **_: SOURCE + "\n" if "rev-list" in args else f'[project]\nversion = "{VERSION}"\n',
    )
    monkeypatch.setattr(github_release, "tag_target", lambda tag: SOURCE if tag == f"v{VERSION}" else None)
    published = {"immutable": True, "draft": False, "prerelease": False, "published_at": "today", "assets": []}
    monkeypatch.setattr(github_release, "get_release", lambda _: published)
    github_release.preflight(VERSION, SOURCE, SOURCE, "Build recovery CI bundle")
    with pytest.raises(ValueError, match="already exists"):
        github_release.preflight(VERSION, SOURCE, SOURCE, "Build CI image and bundle")
    published["assets"] = [{"name": "already-present.whl"}]
    with pytest.raises(ValueError, match="no uploaded assets"):
        github_release.preflight(VERSION, SOURCE, SOURCE, "Build recovery CI bundle")


def test_recovery_workflow_keeps_publication_separate_and_protected():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.load((root / ".github/workflows/release-recovery.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["on"]["workflow_dispatch"]["inputs"]["phase"]["default"] == "Build recovery CI bundle"
    assert "refs/heads/main" in workflow["jobs"]["preflight"]["if"]
    assert workflow["jobs"]["image"]["if"] == "inputs.phase == 'Build recovery CI bundle'"
    assert workflow["jobs"]["kitmaker"]["if"] == "inputs.phase == 'Publish recovery packages'"
    assert workflow["jobs"]["kitmaker"]["with"]["asset_tag"] == "v${{ inputs.version }}-assets"
    assert workflow["jobs"]["verify-indexes"]["needs"] == "kitmaker"
    assert all(job.get("permissions", {}).get("contents") != "write" for job in workflow["jobs"].values())
