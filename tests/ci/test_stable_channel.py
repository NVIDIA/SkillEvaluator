# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "ghcr.io/nvidia/skillevaluator-tier1@sha256:" + "a" * 64
SCAN_REF = "b" * 40
PUBLISHER_REF = "c" * 40
STABLE_SCAN = "NVIDIA/SkillEvaluator/.github/workflows/skill-checks-stable.yml@stable"


@pytest.fixture
def bundle(tmp_path):
    path = tmp_path / "1.2.3"
    path.mkdir()
    manifest = {
        "repository": "NVIDIA/SkillEvaluator",
        "workflow_ref": SCAN_REF,
        "publisher_ref": PUBLISHER_REF,
        "image": IMAGE,
    }
    receipt = {
        "all_verified": True,
        "image": IMAGE,
        "source_revision": "d" * 40,
        "public_manifest_matches_smoke_tested_image": True,
        "anonymous_pull_verified": True,
    }
    for name, content in {"release.json": manifest, "verification-receipt.json": receipt}.items():
        (path / name).write_text(json.dumps(content), encoding="utf-8")
    return path


def test_stable_workflows_pair_the_release_with_delegated_provenance(bundle):
    from ci.stable_channel import render

    files = render(bundle)
    scan = yaml.load(files["skill-checks-stable.yml"], Loader=yaml.BaseLoader)
    assert list(scan["jobs"]) == ["tier1"]
    job = scan["jobs"]["tier1"]
    assert "permissions" not in scan and "permissions" not in job
    assert job["uses"] == f"NVIDIA/SkillEvaluator/.github/workflows/evaluate-skills.yml@{SCAN_REF}"
    assert job["with"]["image"] == IMAGE
    assert "image" not in scan["on"]["workflow_call"]["inputs"]
    publisher = yaml.load(files["publish-skill-reports-stable.yml"], Loader=yaml.BaseLoader)
    job = publisher["jobs"]["publish"]
    assert job["uses"] == f"NVIDIA/SkillEvaluator/.github/workflows/publish-skill-reports.yml@{PUBLISHER_REF}"
    assert job["with"]["tools-ref"] == PUBLISHER_REF
    assert job["with"]["trusted-wrapper-workflows"] == job["with"]["delegated-evaluator-wrapper"] == STABLE_SCAN
    assert job["with"]["trusted-caller-blobs"] == "${{ inputs.trusted-caller-blobs }}"
    assert "trusted-evaluator-shas" not in job["with"]


@pytest.mark.parametrize(
    ("filename", "key", "value", "match"),
    [
        ("release.json", "repository", "another/repository", "official"),
        ("release.json", "workflow_ref", "main", "workflow_ref"),
        ("release.json", "publisher_ref", "stable", "publisher_ref"),
        ("release.json", "image", "ghcr.io/nvidia/skillevaluator-tier1:latest", "image"),
        ("verification-receipt.json", "all_verified", False, "verification"),
        ("verification-receipt.json", "image", None, "verification"),
        ("verification-receipt.json", "source_revision", "main", "verification"),
    ],
)
def test_rejects_unverified_or_mutable_release(bundle, filename, key, value, match):
    from ci.stable_channel import render

    path = bundle / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, key: value}), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        render(bundle)


def test_checked_in_workflows_match_the_named_release_bundle():
    from ci.stable_channel import RELEASE_HEADER, render

    workflows = ROOT / ".github/workflows"
    version = (workflows / "skill-checks-stable.yml").read_text(encoding="utf-8").splitlines()[1]
    assert version.startswith(RELEASE_HEADER)
    bundle = ROOT / "ci/bundles" / version.removeprefix(RELEASE_HEADER).removeprefix("v")
    for name, expected in render(bundle).items():
        assert (workflows / name).read_text(encoding="utf-8") == expected


def test_published_manifest_digest_is_verified_before_use(bundle, tmp_path):
    from ci.stable_channel import verify_publication

    manifest = tmp_path / "artifact-manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="published artifact manifest"):
        verify_publication(bundle, manifest, SCAN_REF, "sha256:" + "e" * 64, 2, "sha256:" + "f" * 64, 8)


@pytest.mark.skipif(os.name != "posix", reason="The release promotion workflow runs Bash on Linux.")
def test_promotion_bootstraps_an_existing_release_and_moves_both_workflows(tmp_path, bundle):
    repo = tmp_path / "repo"
    repo.mkdir()
    upstream = tmp_path / "upstream.git"

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE).strip()

    git("init", "--bare", str(upstream))
    git("init", "--initial-branch=main")
    git("config", "user.name", "Release Test")
    git("config", "user.email", "release-test@example.com")
    git("config", "commit.gpgsign", "false")
    git("remote", "add", "origin", str(upstream))
    workflows = repo / ".github/workflows"
    workflows.mkdir(parents=True)
    (repo / "ci").mkdir()
    for name in ("evaluate-skills.yml", "publish-skill-reports.yml"):
        (workflows / name).write_text("name: Released workflow\n", encoding="utf-8")
    (repo / "ci/Dockerfile.tier1").write_text("FROM scratch\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "Reviewed workflow and image source")
    source = git("rev-parse", "HEAD")

    def prepare_bundle(version, publisher_ref):
        dest = repo / "ci/bundles" / version
        shutil.copytree(bundle, dest)
        pins = dest / "release.json"
        pins.write_text(
            json.dumps({**json.loads(pins.read_text()), "workflow_ref": source, "publisher_ref": publisher_ref}),
            encoding="utf-8",
        )
        receipt = dest / "verification-receipt.json"
        receipt.write_text(json.dumps({**json.loads(receipt.read_text()), "source_revision": source}), encoding="utf-8")
        with ZipFile(dest / f"skillevaluator-ci-{version}.zip", "w") as archive:
            for path in (pins, receipt):
                archive.write(path, arcname=path.name)
        return dest

    initial = prepare_bundle("1.2.3", source)
    git("add", "ci/bundles")
    git("commit", "-m", "Immutable published release")
    git("tag", "v1.2.3")

    for name in ("github_api.py", "stable_channel.py", "promote_stable.sh"):
        shutil.copyfile(ROOT / "ci" / name, repo / "ci" / name)
    shutil.copytree(ROOT / "ci/templates/stable", repo / "ci/templates/stable")

    def prepare_release(release_bundle):
        subprocess.run(
            ["python3", "ci/stable_channel.py", str(release_bundle)],
            cwd=repo,
            check=True,
        )
        git("add", ".")
        git("commit", "-m", f"Prepare stable workflows for {release_bundle.name}")
        return git("rev-parse", "HEAD")

    def published_ref():
        return git("ls-remote", "origin", "refs/tags/stable").split()[0]

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_gh = bin_dir / "gh"
    fake_gh.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        '    api) printf "%s\\n" "$TEST_RELEASE_RECORD" ;;\n'
        '    release) cp "$TEST_MANIFEST" "$9/artifact-manifest.json" ;;\n'
        "    *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)

    def promote(tag, *, source_tree=None):
        version = tag.removeprefix("v")
        commit = git("rev-parse", f"refs/tags/{tag}^{{commit}}")
        archive = repo / "ci/bundles" / version / f"skillevaluator-ci-{version}.zip"
        archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive_size = archive.stat().st_size
        data = {
            "version": version,
            "source_commit": commit,
            "source_tree": source_tree or git("rev-parse", f"{commit}^{{tree}}"),
            "image_source_commit": source,
            "image": IMAGE,
            "ci_bundle": {
                "filename": archive.name,
                "size": archive_size,
                "sha256": archive_digest,
            },
        }
        manifest = tmp_path / "artifact-manifest.json"
        body = json.dumps(data).encode()
        manifest.write_bytes(body)
        return subprocess.run(
            ["bash", "ci/promote_stable.sh"],
            cwd=repo,
            env={
                **os.environ,
                "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                "TEST_RELEASE_RECORD": (
                    f"{tag}\tsha256:{hashlib.sha256(body).hexdigest()}\t{len(body)}"
                    f"\tsha256:{archive_digest}\t{archive_size}"
                ),
                "TEST_MANIFEST": str(manifest),
                "STABLE_PUSH_TOKEN": "synthetic-test-token",
            },
            check=False,
            text=True,
            capture_output=True,
        )

    first = prepare_release(initial)
    git("push", "origin", "main", "v1.2.3")
    promoted = promote("v1.2.3")
    assert promoted.returncode == 0, promoted.stderr
    assert published_ref() == first
    assert "already serves" in promote("v1.2.3").stdout

    next_bundle = prepare_bundle("1.2.4", first)
    second = prepare_release(next_bundle)
    git("tag", "v1.2.4")
    git("push", "origin", "main", "v1.2.4")
    rejected = promote("v1.2.4", source_tree="f" * 40)
    assert rejected.returncode != 0 and published_ref() == first
    assert "does not match the tagged CI release" in rejected.stderr
    promoted = promote("v1.2.4")
    assert promoted.returncode == 0, promoted.stderr
    assert published_ref() == second

    git("checkout", "--detach", first)
    stale = promote("v1.2.3")
    assert stale.returncode != 0
    assert "already serves a newer release" in stale.stderr
    assert published_ref() == second
