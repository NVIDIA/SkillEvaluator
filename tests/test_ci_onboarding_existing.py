# SPDX-License-Identifier: Apache-2.0
"""Existing consumer workflows remain the source of repository-specific policy."""

import pytest
import yaml

from skillevaluator import _ci_onboarding_workflows as workflows
from skillevaluator._ci_onboarding_git import SetupError

CALLER = """# Keep the team's scan configuration.
name: Skill checks
on:
  pull_request:
    paths: [libraries/**]
  push:
    branches: [trunk]
permissions:
  contents: read
jobs:
  catalog:
    name: Catalog Tier 1 checks
    permissions:
      contents: read
    uses: example/components/.github/workflows/skill-checks.yml@v3
    with:
      paths: |
        libraries
        reference-implementations
      min-score: 83
      scan-mode: all
      scan-workers: 4
"""


def test_existing_component_is_preserved_except_for_required_job_permission():
    config = workflows.existing_scan(CALLER)
    assert config.roots == ["libraries", "reference-implementations"]
    assert config.score == 83
    assert config.scan_mode == "all"
    assert config.job == "catalog"
    updated = config.with_identity_permission()
    assert updated.startswith("# Keep the team's scan configuration.")
    old, new = yaml.safe_load(CALLER), yaml.safe_load(updated)
    new["jobs"]["catalog"]["permissions"].pop("id-token")
    assert old == new
    assert config.delegated_wrapper == "example/components/.github/workflows/skill-checks.yml@v3"
    assert workflows.existing_scan(updated).with_identity_permission() == updated


def test_explicit_job_permissions_override_root_identity_permission():
    content = CALLER.replace("permissions:\n  contents: read", "permissions:\n  contents: read\n  id-token: write")
    updated = workflows.existing_scan(content).with_identity_permission()
    assert yaml.safe_load(updated)["jobs"]["catalog"]["permissions"]["id-token"] == "write"


def test_inherited_permission_keeps_exact_existing_bytes():
    content = CALLER.replace("    permissions:\n      contents: read\n", "")
    content = content.replace("permissions:\n  contents: read", "permissions:\n  contents: read\n  id-token: write")
    assert workflows.existing_scan(content).with_identity_permission() == content


@pytest.mark.parametrize("ref", ["main", "refs/heads/v3", "${{ inputs.version }}", "v3/unsafe"])
def test_existing_mutable_branch_or_dynamic_wrapper_is_not_silently_approved(ref):
    with pytest.raises(SetupError):
        workflows.existing_scan(CALLER.replace("@v3", "@" + ref))


@pytest.mark.parametrize("extra", ["    strategy: {matrix: {test: [a, b]}}\n", "    secrets: inherit\n"])
def test_matrix_or_inherited_secrets_are_not_silently_approved(extra):
    # An existing secret-passing workflow requires deliberate owner review;
    # report onboarding must not expand its trust automatically.
    with pytest.raises(SetupError):
        workflows.existing_scan(CALLER.replace("    name:", extra + "    name:"))


def test_duplicate_yaml_keys_are_rejected_instead_of_approving_ambiguous_bytes():
    with pytest.raises(SetupError):
        workflows.existing_scan(CALLER.replace("      min-score: 83", "      min-score: 83\n      min-score: 0"))


def test_direct_evaluator_keeps_its_existing_pin_and_inputs():
    pin = "b" * 40
    content = CALLER.replace(
        "example/components/.github/workflows/skill-checks.yml@v3",
        "NVIDIA/SkillEvaluator/.github/workflows/evaluate-skills.yml@" + pin,
    )
    config = workflows.existing_scan(content)
    assert config.evaluator_sha == pin
    assert config.delegated_wrapper is None


def test_multiple_jobs_require_manual_review():
    with pytest.raises(SetupError):
        workflows.existing_scan(CALLER + "  other:\n    runs-on: ubuntu-latest\n    steps: []\n")


def test_job_permission_insertion_does_not_drop_other_inherited_permissions():
    content = CALLER.replace("    permissions:\n      contents: read\n", "")
    content = content.replace("permissions:\n  contents: read", "permissions:\n  contents: read\n  checks: read")
    updated = workflows.existing_scan(content).with_identity_permission()
    assert yaml.safe_load(updated)["jobs"]["catalog"]["permissions"] == {
        "contents": "read",
        "checks": "read",
        "id-token": "write",
    }
