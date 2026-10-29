# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
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
    scan = yaml.load(scan_path.read_text(), Loader=yaml.BaseLoader)
    assert scan["jobs"]["skills"]["with"]["image"] == image
    assert scan["jobs"]["skills"]["uses"].endswith("@" + "b" * 40)
    publisher = yaml.load((output / "publish-skill-reports.yml").read_text(), Loader=yaml.BaseLoader)
    assert publisher["jobs"]["reports"]["with"]["trusted-caller-blobs"] == blob
    assert publisher["jobs"]["reports"]["with"]["tools-ref"] == "b" * 40
    for template in output.glob("*.yml"):
        assert "RELEASE_" not in template.read_text()
    assert "YOUR_REPORT_BASE_URL" in (output / "publish-skill-reports.yml").read_text()


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
    text = (output / "publish-skill-reports.yml").read_text()
    assert "example/scan/.github/workflows/" in text
    assert "tools-repository: example/scan" in text


def test_existing_release_output_is_preserved(tmp_path):
    with pytest.raises(ValueError, match="new output directory"):
        release.render_templates(tmp_path, "b" * 40, "ghcr.io/nvidia/scan@sha256:" + "a" * 64)
