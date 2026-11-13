# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SPEC = importlib.util.spec_from_file_location("render_templates", Path(__file__).parents[2] / "ci/render_templates.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_release_templates_are_copyable_and_bind_the_actual_caller(tmp_path):
    output = tmp_path / "onboarding"
    image = "ghcr.io/nvidia/skillevaluator-tier1@sha256:" + "a" * 64
    receipt = release.render_templates(output, "b" * 40, image)
    scan_path = output / "skill-checks.yml"
    blob = subprocess.check_output(["git", "hash-object", str(scan_path)], text=True).strip()
    assert receipt["caller_blob"] == blob
    assert receipt["publisher_ref"] == receipt["workflow_ref"] == "b" * 40
    scan = yaml.load(scan_path.read_text(), Loader=yaml.BaseLoader)
    assert scan["jobs"]["skills"]["with"]["image"] == image
    assert scan["jobs"]["skills"]["uses"].endswith("@" + "b" * 40)
    publisher = yaml.load((output / "publish-skill-reports.yml").read_text(), Loader=yaml.BaseLoader)
    assert publisher["jobs"]["reports"]["with"]["trusted-caller-blobs"] == blob
    assert publisher["jobs"]["reports"]["with"]["tools-ref"] == "b" * 40
    comments = yaml.load((output / "comment-skill-checks.yml").read_text(), Loader=yaml.BaseLoader)
    assert comments["permissions"] == {"contents": "read", "actions": "read", "pull-requests": "write"}
    assert comments["on"]["workflow_run"] == {"workflows": ["Skill checks"], "types": ["completed"]}
    assert set(comments["on"]["workflow_dispatch"]["inputs"]) == {"run-id", "run-attempt"}
    assert all(value["required"] == "true" for value in comments["on"]["workflow_dispatch"]["inputs"].values())
    assert comments["jobs"]["comment"]["with"]["trusted-caller-blobs"] == blob
    assert comments["jobs"]["comment"]["with"]["trusted-evaluator-shas"] == "b" * 40
    assert comments["jobs"]["comment"]["with"]["tools-ref"] == "b" * 40
    assert "report-base-url" not in comments["jobs"]["comment"]["with"]
    for template in output.glob("*.yml"):
        assert "RELEASE_" not in template.read_text()
    assert "YOUR_REPORT_BASE_URL" in (output / "publish-skill-reports.yml").read_text()


def test_separate_publisher_revision_preserves_scanner_and_caller_trust(tmp_path):
    output = tmp_path / "onboarding"
    scanner_ref, publisher_ref = "b" * 40, "c" * 40
    image = "ghcr.io/nvidia/skillevaluator-tier1@sha256:" + "a" * 64
    subprocess.run(
        [
            sys.executable,
            str(SPEC.origin),
            "--workflow-ref",
            scanner_ref,
            "--publisher-ref",
            publisher_ref,
            "--image",
            image,
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    receipt = json.loads((output / "release.json").read_text())
    assert receipt["workflow_ref"] == scanner_ref
    assert receipt["publisher_ref"] == publisher_ref
    assert receipt["image"] == image
    blob = subprocess.check_output(["git", "hash-object", str(output / "skill-checks.yml")], text=True).strip()
    assert receipt["caller_blob"] == blob
    publisher = yaml.load((output / "publish-skill-reports.yml").read_text(), Loader=yaml.BaseLoader)["jobs"]["reports"]
    assert publisher["uses"].endswith("@" + publisher_ref)
    assert publisher["with"]["tools-ref"] == publisher_ref
    assert publisher["with"]["trusted-evaluator-shas"] == scanner_ref
    assert publisher["with"]["trusted-caller-blobs"] == blob
    comments = yaml.load((output / "comment-skill-checks.yml").read_text(), Loader=yaml.BaseLoader)["jobs"]["comment"]
    assert comments["uses"].endswith("@" + publisher_ref)
    assert comments["with"]["tools-ref"] == publisher_ref
    assert comments["with"]["trusted-evaluator-shas"] == scanner_ref
    assert comments["with"]["trusted-caller-blobs"] == blob
    assert publisher_ref in (output / "README.md").read_text()
    for name in ("skill-checks.yml", "skill-checks-llm.yml", "gitlab-skill-checks.yml"):
        content = (output / name).read_text()
        assert scanner_ref in content and image in content
        assert publisher_ref not in content
        assert "RELEASE_" not in content


@pytest.mark.parametrize("publisher_ref", ["", "main", "a" * 39, "a" * 41])
def test_invalid_publisher_revision_is_rejected_before_writing(tmp_path, publisher_ref):
    output = tmp_path / "onboarding"
    with pytest.raises(ValueError, match="publisher commit SHA"):
        release.render_templates(
            output, "b" * 40, "ghcr.io/nvidia/scan@sha256:" + "a" * 64, publisher_ref=publisher_ref
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "ref,image,repository",
    [
        ("main", "ghcr.io/nvidia/scan@sha256:" + "a" * 64, "NVIDIA/SkillEvaluator"),
        ("b" * 40, "ghcr.io/nvidia/scan:latest", "NVIDIA/SkillEvaluator"),
        ("b" * 40, "ghcr.io/nvidia/scan@sha256:" + "a" * 64, "bad repo"),
    ],
)
def test_unpinned_or_invalid_release_identity_is_rejected(tmp_path, ref, image, repository):
    with pytest.raises(ValueError):
        release.render_templates(tmp_path / "out", ref, image, repository)
    assert not (tmp_path / "out").exists()


def test_fork_verification_templates_point_to_the_fork(tmp_path):
    output = tmp_path / "out"
    release.render_templates(output, "b" * 40, "ghcr.io/example/scan@sha256:" + "a" * 64, "example/scan")
    for filename in ("publish-skill-reports.yml", "comment-skill-checks.yml"):
        text = (output / filename).read_text()
        assert "example/scan/.github/workflows/" in text
        assert "tools-repository: example/scan" in text


def test_existing_release_output_is_preserved(tmp_path):
    with pytest.raises(ValueError, match="new output directory"):
        release.render_templates(tmp_path, "b" * 40, "ghcr.io/nvidia/scan@sha256:" + "a" * 64)


def test_gitlab_consumer_pins_image_and_reviewed_job_and_enables_mr_rules(tmp_path):
    output = tmp_path / "out"
    image = "ghcr.io/nvidia/skillevaluator-tier1@sha256:" + "a" * 64
    release.render_templates(output, "b" * 40, image)
    config = yaml.safe_load((output / "gitlab-skill-checks.yml").read_text())
    assert config["variables"]["SKILL_EVAL_IMAGE"] == image
    assert config["include"][0]["remote"] == (
        "https://raw.githubusercontent.com/NVIDIA/SkillEvaluator/" + "b" * 40 + "/ci/gitlab/skill-evaluation.yml"
    )
    assert any("merge_request_event" in rule["if"] for rule in config["skill-evaluation"]["rules"])
