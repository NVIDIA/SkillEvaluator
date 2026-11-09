# SPDX-License-Identifier: Apache-2.0
"""Shared CI wiring for enriched security and independent live-agent models."""

import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]


def workflow():
    return yaml.load((ROOT / ".github/workflows/evaluate-all-tiers.yml").read_text(), Loader=yaml.BaseLoader)


def test_enriched_tier1_requires_admission_and_keeps_the_keyless_path():
    document = workflow()
    assert document["on"]["workflow_call"]["inputs"]["tier1-llm"]["default"] == "false"
    baseline = document["jobs"]["tier1"]
    assert baseline["with"]["llm"] == "false"
    assert "always()" in baseline["if"]
    assert "needs.eligibility.outputs.allowed" in baseline["with"]["artifact-name"]
    assert baseline["with"]["artifact-name"].endswith("|| 'skill-evaluator-tier1' }}")
    enriched = document["jobs"]["tier1-llm"]
    # A native local Tier 3 runner need not have Docker. Tier 1 keeps the
    # separate image-capable runner used by its keyless baseline.
    assert enriched["runs-on"] == "ubuntu-24.04"
    assert "inputs.tier1-llm" in enriched["if"]
    assert "needs.eligibility.outputs.allowed == 'true'" in enriched["if"]
    assert enriched["environment"] == "${{ inputs.environment }}"
    check = next(step for step in enriched["steps"] if step.get("id") == "current-head")
    evaluate = next(step for step in enriched["steps"] if "run_tier1_image.py" in step.get("run", ""))
    assert enriched["steps"].index(check) < enriched["steps"].index(evaluate)
    assert "steps.current-head.outputs.allowed == 'true'" in evaluate["if"]
    assert "secrets." not in str(check)
    assert "tier1-llm" in document["jobs"]["aggregate"]["needs"]


def test_native_jobs_forward_gateway_settings_only_at_protected_execution_steps():
    document = workflow()
    inputs = document["on"]["workflow_call"]["inputs"]
    for name in ("base-url", "embedding-base-url", "tier3-provider", "tier3-model", "tier3-base-url", "agent-models"):
        assert inputs[name]["default"] == ""
    paid = document["jobs"]["paid-tiers"]
    assert paid["continue-on-error"] == "true"
    assert paid["env"]["SKILL_EVAL_TIER3_PROVIDER"] == "${{ inputs.tier3-provider }}"
    assert paid["env"]["SKILL_EVAL_AGENT_MODELS"] == "${{ inputs.agent-models }}"
    assert "secrets." not in str(paid["env"])
    for step in paid["steps"]:
        if "secrets.LLM_API_KEY" in str(step):
            assert "steps.current-head.outputs.allowed == 'true'" in step["if"]
            assert step["env"]["SKILL_EVAL_LLM_API_KEY"] == "${{ secrets.LLM_API_KEY }}"


def test_gitlab_enrichment_is_protected_and_does_not_need_docker_in_docker():
    document = yaml.safe_load((ROOT / "ci/gitlab/all-tiers.yml").read_text())
    job = document["skill-evaluation-tier1-llm"]
    assert job["environment"]["name"] == "$SKILL_EVAL_ENVIRONMENT"
    assert job["services"] == []
    assert "tags" not in job  # Use the existing untagged Tier 1 image runner.
    script = "\n".join(job["script"])
    assert script.index("eligible_gitlab.py") < script.index("run_tier1_image.py")
    assert "--in-image" in script
    rules = str(job["rules"])
    assert 'SKILL_EVAL_TIER1_LLM != "true"' in rules
    assert "CI_MERGE_REQUEST_SOURCE_PROJECT_ID != $CI_PROJECT_ID" in rules
    needs = document["skill-evaluation-aggregate"]["needs"]
    assert any(item["job"] == "skill-evaluation-tier1-llm" for item in needs)


def _aggregate_selects_enriched(env):
    document = yaml.safe_load((ROOT / "ci/gitlab/all-tiers.yml").read_text())
    script = "\n".join(document["skill-evaluation-aggregate"]["script"])
    selection = script[script.index("enriched=false") : script.index('if [ "$enriched" = true ]')]
    result = subprocess.run(
        ["sh", "-c", "set -eu\n" + selection + 'printf %s "$enriched"'],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout == "true"


@pytest.mark.parametrize(
    ("source", "admitted"),
    [("push", True), ("web", True), ("schedule", False), ("api", False), ("trigger", False), ("pipeline", False)],
)
def test_gitlab_enriched_tier1_matches_default_branch_admission(source, admitted):
    """Sources that admission rejects keep the keyless baseline instead of failing Tier 1."""
    env = {
        "SKILL_EVAL_TIER1_LLM": "true",
        "CI_PIPELINE_SOURCE": source,
        "CI_PROJECT_ID": "42",
        "CI_COMMIT_BRANCH": "main",
        "CI_DEFAULT_BRANCH": "main",
        "CI_COMMIT_REF_PROTECTED": "true",
    }
    assert _aggregate_selects_enriched(env) is admitted
    spec = importlib.util.spec_from_file_location("gate_sources", ROOT / "ci/eligible_gitlab.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    assert gate._default_branch_candidate({**env, "CI_COMMIT_SHA": "a" * 40}) is admitted
    document = yaml.safe_load((ROOT / "ci/gitlab/all-tiers.yml").read_text())
    default_rule = document["skill-evaluation-tier1-llm"]["rules"][-1]["if"]
    assert '$CI_PIPELINE_SOURCE == "push" || $CI_PIPELINE_SOURCE == "web"' in default_rule


def test_gitlab_same_project_mr_still_selects_enriched_tier1():
    env = {
        "SKILL_EVAL_TIER1_LLM": "true",
        "CI_PIPELINE_SOURCE": "merge_request_event",
        "CI_PROJECT_ID": "42",
        "CI_MERGE_REQUEST_SOURCE_PROJECT_ID": "42",
        "CI_DEFAULT_BRANCH": "main",
    }
    assert _aggregate_selects_enriched(env)
    assert not _aggregate_selects_enriched({**env, "CI_MERGE_REQUEST_SOURCE_PROJECT_ID": "7"})


def test_benchmark_publisher_is_separate_from_read_only_pr_feedback():
    document = yaml.load((ROOT / ".github/workflows/publish-benchmarks.yml").read_text(), Loader=yaml.BaseLoader)
    job = document["jobs"]["benchmarks"]
    assert document["permissions"]["contents"] == "write"
    assert "workflow_run" in job["if"] and "push" in job["if"] and "workflow_dispatch" in job["if"]
    assert "head_repository.full_name == github.repository" in job["if"]
    checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["ref"] == "${{ inputs.tools-ref }}"
    assert "pull_request.head" not in str(job)
    assert "secrets." not in str(job)
    feedback = yaml.load((ROOT / ".github/workflows/feedback-all-tiers.yml").read_text(), Loader=yaml.BaseLoader)
    assert feedback["permissions"]["contents"] == "read"
