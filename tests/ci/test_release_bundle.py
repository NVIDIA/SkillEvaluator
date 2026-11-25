# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).parents[2]
SOURCE = "a" * 40
IMAGE = "ghcr.io/nvidia/skillevaluator-tier1@sha256:" + "b" * 64
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
BUILD_RUN = "https://github.com/NVIDIA/SkillEvaluator/actions/runs/123"
CASES = ("all-pass", "invalid-name", "changed-one", "no-change", "missing-llm-key")


@pytest.fixture
def qualification(tmp_path):
    smoke_path = tmp_path / "smoke-receipt.json"
    smoke_path.write_text(
        json.dumps(
            {
                "image": IMAGE,
                "all_verified": True,
                "runtime": {
                    "id": "sha256:" + "c" * 64,
                    "digests": [IMAGE],
                    "os": "linux",
                    "architecture": "amd64",
                    "source_revision": SOURCE,
                },
                "cases": [{"case": case, "verified": True} for case in CASES],
            }
        )
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'test "$1 $2 $3 $4 $5 $6 $7" = "run --rm --pull=never --network=none --entrypoint $6 $EXPECTED_IMAGE" || exit 9\n'
        'case "$6" in\n'
        '  skillevaluator) printf "%s\n" "skillevaluator $EXPECTED_VERSION" ;;\n'
        '  semgrep) printf "%s\n" "1.182.0" ;;\n'
        '  skillspector) printf "%s\n" "SkillSpector v2.13.6" ;;\n'
        '  pip-audit) printf "%s\n" "pip-audit 2.10.1" ;;\n'
        '  bandit) printf "%s\n" "bandit 1.9.4" ;;\n'
        '  gitleaks) printf "%s\n" "8.30.0" ;;\n'
        "  *) exit 9 ;;\n"
        "esac\n"
    )
    docker.chmod(0o755)
    return smoke_path


def build(output, receipt, *, version=VERSION):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "ci/release_bundle.py"),
            "--version",
            version,
            "--source-sha",
            SOURCE,
            "--image",
            IMAGE,
            "--smoke-receipt",
            str(receipt),
            "--build-run",
            BUILD_RUN,
            "--output",
            str(output),
        ],
        env={
            "PATH": f"{receipt.parent / 'bin'}:/usr/bin:/bin",
            "EXPECTED_IMAGE": IMAGE,
            "EXPECTED_VERSION": version,
        },
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_release_bundle_contains_deterministic_pins_and_verified_evidence(qualification, tmp_path):
    output = tmp_path / "new"
    repeated = tmp_path / "again"
    for target in (output, repeated):
        result = build(target, qualification)
        assert result.returncode == 0, result.stderr
    bundle = output / "ci/bundles" / VERSION
    archive_name = f"skillevaluator-ci-{VERSION}.zip"
    archive = bundle / archive_name
    assert archive.read_bytes() == (repeated / "ci/bundles" / VERSION / archive_name).read_bytes()
    with ZipFile(archive) as released:
        assert released.namelist() == sorted(path.name for path in bundle.iterdir() if path != archive)
        assert all(released.read(name) == (bundle / name).read_bytes() for name in released.namelist())
    sums = (bundle / "SHA256SUMS.txt").read_text().splitlines()
    assert sums == [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(bundle.iterdir())
        if path not in {archive, bundle / "SHA256SUMS.txt"}
    ]
    report = json.loads((bundle / "verification-receipt.json").read_text())
    assert report["image"] == IMAGE
    assert report["source_revision"] == SOURCE
    assert report["platform"] == {"os": "linux", "architecture": "amd64"}
    assert (
        report["all_verified"]
        and report["anonymous_pull_verified"]
        and report["public_manifest_matches_smoke_tested_image"]
    )
    assert report["official_build_run"] == BUILD_RUN
    inventory = json.loads((bundle / "scanner-inventory.json").read_text())
    assert inventory["skillspector"]["version"] == "2.13.6"
    assert inventory["skillevaluator"]["version"] == VERSION
    descriptors = output / "src/skillevaluator"
    tier1 = json.loads((descriptors / "ci-release.json").read_text())
    all_tiers = json.loads((descriptors / "ci-release-all-tiers.json").read_text())
    assert tier1["schema_version"] == 1
    assert all_tiers["schema_version"] == 2
    for descriptor in (tier1, all_tiers):
        assert descriptor["workflow_ref"] == descriptor["publisher_ref"] == SOURCE
        assert descriptor["image"] == IMAGE
    assert all_tiers["tools_ref"] == all_tiers["gitlab_ref"] == all_tiers["feedback_ref"] == SOURCE
    assert all_tiers["evaluator_version"] == VERSION
    assert all_tiers["runtime_lock_sha256"] == hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    for name in ("skill-checks-stable.yml", "comment-skill-checks-stable.yml", "publish-skill-reports-stable.yml"):
        workflow = (output / ".github/workflows" / name).read_text()
        assert f"# SkillEvaluator release: v{VERSION}" in workflow
        assert SOURCE in workflow
        assert "RELEASE_" not in workflow
    assert IMAGE in (output / ".github/workflows/skill-checks-stable.yml").read_text()
    if (ROOT / "ci/templates/stable/stable-tier1.yml").exists():
        include = (output / "ci/gitlab/stable-tier1.yml").read_text()
        assert f"# SkillEvaluator release: v{VERSION}" in include
        assert SOURCE in include and IMAGE in include
        assert not (output / ".github/workflows/stable-tier1.yml").exists()


@pytest.mark.parametrize(
    "case",
    ("wrong-platform", "wrong-revision", "wrong-digest", "missing-case", "failed-case", "wrong-version"),
)
def test_release_bundle_rejects_unverified_image_and_version(qualification, tmp_path, case):
    smoke = json.loads(qualification.read_text())
    version = VERSION
    if case == "wrong-platform":
        smoke["runtime"]["architecture"] = "arm64"
    elif case == "wrong-revision":
        smoke["runtime"]["source_revision"] = "d" * 40
    elif case == "wrong-digest":
        smoke["runtime"]["digests"] = [IMAGE[:-1] + "d"]
    elif case == "missing-case":
        smoke["cases"].pop()
    elif case == "failed-case":
        smoke["cases"][0]["verified"] = False
    else:
        version = "99.0.0"
    qualification.write_text(json.dumps(smoke))
    output = tmp_path / "new"
    result = build(output, qualification, version=version)
    assert result.returncode == 1, result.stderr
    assert not output.exists()
