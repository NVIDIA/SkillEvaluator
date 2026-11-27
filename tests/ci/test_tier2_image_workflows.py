# SPDX-License-Identifier: Apache-2.0
"""Keep Tier 2 image verification outside the protected registry publisher."""

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).parents[2] / ".github/workflows"


def test_tier2_image_verifies_pull_requests_without_registry_write():
    workflow = yaml.load((WORKFLOWS / "verify-tier2-image.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["permissions"] == {"contents": "read"}
    assert "ci/**" in workflow["on"]["pull_request"]["paths"]
    assert workflow["on"]["push"]["branches"] == ["main"]
    steps = workflow["jobs"]["verify"]["steps"]
    build = next(step for step in steps if "docker/build-push-action@" in step.get("uses", ""))
    assert build["with"]["file"] == "ci/Dockerfile.tier2"
    assert build["with"]["load"] == "true"
    assert "push" not in build["with"]
    assert "github.event_name != 'pull_request'" in build["with"]["cache-to"]
    assert any("ci/smoke_tier2_image.py" in step.get("run", "") for step in steps)


def test_tier2_release_publishes_only_a_smoked_reviewed_revision():
    workflow = yaml.load((WORKFLOWS / "release-tier2-image.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    publish = workflow["jobs"]["publish"]
    assert publish["environment"] == "release-image"
    assert "github.ref == 'refs/heads/main'" in publish["if"]
    assert "github.repository == 'NVIDIA/SkillEvaluator'" in publish["if"]
    assert publish["env"]["IMAGE"] == "ghcr.io/nvidia/skillevaluator-tier2"
    steps = [step.get("run", "") for step in publish["steps"]]
    ancestry = next(i for i, step in enumerate(steps) if "git rev-list --first-parent" in step)
    smoke = next(i for i, step in enumerate(steps) if "ci/smoke_tier2_image.py" in step)
    push = next(i for i, step in enumerate(steps) if "docker push" in step)
    assert ancestry < smoke < push
    assert "${{ github.run_id }}-${{ github.run_attempt }}" in publish["env"]["PUSH_TAG"]
    qualify = workflow["jobs"]["qualify"]
    assert qualify["needs"] == "publish"
    assert qualify["permissions"] == {"contents": "read"}
    assert any('docker pull "$IMAGE_DIGEST"' in step.get("run", "") for step in qualify["steps"])
