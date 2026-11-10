# SPDX-License-Identifier: Apache-2.0
"""Keep this repository's consumer callers compatible with trusted upstream CI."""

import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
SCAN = ".github/workflows/skill-checks-all-tiers.yml"
FEEDBACK = ".github/workflows/skill-feedback-all-tiers.yml"


def workflow(path):
    return yaml.load((ROOT / path).read_text(), Loader=yaml.BaseLoader)


def test_repository_skill_checks_use_reviewed_full_three_tier_configuration():
    scan = workflow(SCAN)
    assert set(scan["on"]) == {"pull_request", "push", "workflow_dispatch"}
    assert scan["on"]["push"]["branches"] == ["main"]
    assert scan["permissions"] == {
        "contents": "read",
        "actions": "read",
        "pull-requests": "read",
        "id-token": "write",
    }
    job = scan["jobs"]["skills"]
    prefix = "NVIDIA/SkillEvaluator/.github/workflows/evaluate-all-tiers.yml@"
    assert job["uses"].startswith(prefix)
    runtime = job["uses"].removeprefix(prefix)
    assert re.fullmatch(r"[0-9a-f]{40}", runtime)
    inputs = job["with"]
    assert re.fullmatch(r"ghcr.io/nvidia/skillevaluator-tier1@sha256:[0-9a-f]{64}", inputs["image"])
    assert inputs["tools-repository"] == "NVIDIA/SkillEvaluator"
    assert inputs["tools-ref"] == runtime
    assert inputs["tiers"] == "1,2,3"
    assert inputs["paths"] == "skills"
    assert inputs["scan-mode"] == "auto"
    assert inputs["env-mode"] == "docker"
    assert inputs["min-score"] == "70"
    assert inputs["provider"] == inputs["embedding-provider"] == "nv_build"
    assert inputs["environment"] == "skillevaluator"
    assert inputs["n-attempts"] == inputs["n-concurrent"] == "1"
    assert "tier3-required" not in inputs
    assert job["secrets"] == {
        key: "${{ secrets." + key + " }}" for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NVIDIA_API_KEY")
    }
    assert "steps" not in job


def test_repository_feedback_approves_exact_caller_and_preserves_public_publisher():
    feedback = workflow(FEEDBACK)
    assert feedback["on"] == {"workflow_run": {"workflows": [workflow(SCAN)["name"]], "types": ["completed"]}}
    assert feedback["permissions"] == {"contents": "read", "actions": "read", "pull-requests": "write"}
    job = feedback["jobs"]["feedback"]
    # A consumer keeps its reviewed runtime when a later release descriptor changes.
    runtime = workflow(SCAN)["jobs"]["skills"]["with"]["tools-ref"]
    assert job["uses"] == ("NVIDIA/SkillEvaluator/.github/workflows/feedback-all-tiers.yml@" + runtime)
    assert "steps" not in job
    assert "secrets" not in job
    inputs = job["with"]
    assert inputs["tools-ref"] == runtime
    assert inputs["trusted-workflow-ref"] == workflow(SCAN)["jobs"]["skills"]["uses"]
    # Apply the checkout's clean filters exactly as committing the caller does.
    blob = subprocess.check_output(["git", "hash-object", "--path", SCAN, SCAN], cwd=ROOT, text=True).strip()
    assert inputs["trusted-caller-blobs"] == blob
    public = workflow(".github/workflows/feedback-all-tiers.yml")
    assert set(public["on"]) == {"workflow_call"}
    assert public["jobs"]["feedback"]["steps"]


def test_repository_skill_has_committed_complete_dataset_prerequisites():
    from ci.run_tiers import dataset_preflight

    skills = sorted((ROOT / "skills").glob("*/SKILL.md"))
    assert skills
    for manifest in skills:
        digests = dataset_preflight(manifest.parent)
        assert "evals/evals.json" in digests
