# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

REQUIRED_CI_JOBS = {
    "test-python-312": "Tests (Python 3.12)",
    "test-python-313": "Tests (Python 3.13)",
    "package": "Package",
    "rhel8-security-install": "RHEL 8 security install",
    "tier2-macos": "Tier 2 (macos-latest)",
    "tier2-windows": "Tier 2 (windows-latest)",
    "tier3-macos": "Tier 3 macOS contract and progress",
    "native-windows-local-mode": "Native Windows local mode fails closed",
}
HEAVY_CI_JOBS = set(REQUIRED_CI_JOBS) - {"test-python-312"}
RUN_UNLESS_CANCELLED_IF = "${{ !cancelled() }}"
FULL_LANE_IF = "${{ !cancelled() && needs.classify-changes.outputs.docs_only != 'true' }}"
DOCS_ONLY_IF = "${{ needs.classify-changes.outputs.docs_only == 'true' }}"
NOT_DOCS_ONLY_IF = "${{ needs.classify-changes.outputs.docs_only != 'true' }}"
PR_CONCURRENCY = {
    "group": "${{ github.workflow }}-${{ github.event.pull_request.number || github.run_id }}",
    "cancel-in-progress": "true",
}


def _load(name: str) -> dict[str, Any]:
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _assert_no_path_filter(workflow: dict[str, Any], event: str = "pull_request") -> None:
    trigger = workflow["on"][event]
    if isinstance(trigger, dict):
        assert "paths" not in trigger
        assert "paths-ignore" not in trigger


def _runs(job: dict[str, Any]) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def _all_uses(workflow: dict[str, Any]) -> list[str]:
    return [step["uses"] for job in workflow["jobs"].values() for step in job.get("steps", []) if "uses" in step]


def _all_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


def test_ci_preserves_required_contexts_as_explicit_jobs() -> None:
    ci = _load("ci.yml")

    assert {job_id: ci["jobs"][job_id]["name"] for job_id in REQUIRED_CI_JOBS} == REQUIRED_CI_JOBS
    assert all("matrix." not in ci["jobs"][job_id]["name"] for job_id in REQUIRED_CI_JOBS)
    _assert_no_path_filter(ci)


def test_ci_classifier_is_pull_request_only_and_exports_docs_only() -> None:
    ci = _load("ci.yml")
    classifier = ci["jobs"]["classify-changes"]

    assert ci["concurrency"] == PR_CONCURRENCY
    assert classifier["name"] == "Classify changes"
    assert classifier["if"] == "${{ github.event_name == 'pull_request' }}"
    assert classifier["outputs"]["docs_only"] == "${{ steps.changes.outputs.docs_only }}"
    assert classifier["steps"][0]["with"]["fetch-depth"] == "0"
    assert classifier["steps"][0]["with"]["persist-credentials"] == "false"
    assert classifier["steps"][1]["id"] == "changes"
    classifier_run = classifier["steps"][1]["run"]
    assert 'git show "$BASE_SHA:scripts/classify_ci_changes.py"' in classifier_run
    assert 'python3 "$classifier"' in classifier_run
    assert 'echo "docs_only=false" >> "$GITHUB_OUTPUT"' in classifier_run
    assert "python3 scripts/classify_ci_changes.py" not in classifier_run
    assert classifier["steps"][1]["env"] == {
        "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
        "HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
    }


def test_ci_docs_lane_uses_the_required_python_312_context() -> None:
    job = _load("ci.yml")["jobs"]["test-python-312"]

    assert job["needs"] == "classify-changes"
    assert job["if"] == RUN_UNLESS_CANCELLED_IF
    assert job["runs-on"] == "ubuntu-latest"
    assert job["steps"][0]["with"]["persist-credentials"] == "false"

    full_lane_step_names = {
        "Set up Python",
        "Set up uv",
        "Install dependencies",
        "Scan OSS source boundary",
        "Lint",
        "Run tests with coverage",
    }
    full_lane_steps = {step.get("name"): step for step in job["steps"] if step.get("name") in full_lane_step_names}
    assert set(full_lane_steps) == full_lane_step_names
    assert all(step["if"] == NOT_DOCS_ONLY_IF for step in full_lane_steps.values())

    node_step = next(step for step in job["steps"] if step.get("name") == "Set up Node.js for docs")
    docs_step = next(step for step in job["steps"] if step.get("name") == "Validate Fern documentation")
    assert node_step["if"] == DOCS_ONLY_IF
    assert len(node_step["uses"].split("@", 1)[1]) == 40
    assert docs_step["if"] == DOCS_ONLY_IF
    assert "fern/fern.config.json" in docs_step["run"]
    assert 'npm install --global "fern-api@$FERN_VERSION"' in docs_step["run"]
    assert "fern check" in docs_step["run"]
    assert "GITHUB_STEP_SUMMARY" in docs_step["run"]


def test_ci_skips_every_other_required_job_only_after_classification() -> None:
    jobs = _load("ci.yml")["jobs"]

    for job_id in HEAVY_CI_JOBS:
        assert jobs[job_id]["needs"] == "classify-changes"
        assert jobs[job_id]["if"] == FULL_LANE_IF


def test_full_lane_keeps_the_existing_commands_and_runners() -> None:
    jobs = _load("ci.yml")["jobs"]

    assert jobs["test-python-313"]["runs-on"] == "ubuntu-latest"
    assert "uv run pytest -q" in _runs(jobs["test-python-313"])
    assert jobs["tier2-macos"]["runs-on"] == "macos-latest"
    assert jobs["tier2-windows"]["runs-on"] == "windows-latest"
    assert "tests/embedding" in _runs(jobs["tier2-macos"])
    assert "tests/embedding" in _runs(jobs["tier2-windows"])
    assert jobs["tier3-macos"]["runs-on"] == "macos-latest"
    assert "tests/test_tier3_progress.py" in _runs(jobs["tier3-macos"])
    assert jobs["native-windows-local-mode"]["runs-on"] == "windows-latest"
    assert "tests/test_harbor_local_mode.py" in _runs(jobs["native-windows-local-mode"])
    assert jobs["rhel8-security-install"]["container"] == "rockylinux/rockylinux:8.10"
    assert "uv build --wheel" in _runs(jobs["rhel8-security-install"])
    assert "twine==6.2.0" in _runs(jobs["package"])


def test_security_keeps_gitleaks_always_on_and_skips_only_nonessential_jobs() -> None:
    security = _load("security.yml")
    jobs = security["jobs"]

    _assert_no_path_filter(security)
    assert security["concurrency"] == PR_CONCURRENCY
    assert "if" not in jobs["gitleaks"]
    assert "needs" not in jobs["gitleaks"]
    assert jobs["classify-changes"]["if"] == "${{ github.event_name == 'pull_request' }}"
    assert jobs["classify-changes"]["outputs"]["docs_only"] == "${{ steps.changes.outputs.docs_only }}"
    assert jobs["classify-changes"]["steps"][0]["with"]["persist-credentials"] == "false"
    classifier_run = jobs["classify-changes"]["steps"][1]["run"]
    assert 'git show "$BASE_SHA:scripts/classify_ci_changes.py"' in classifier_run
    assert 'python3 "$classifier"' in classifier_run
    assert 'echo "docs_only=false" >> "$GITHUB_OUTPUT"' in classifier_run
    assert "python3 scripts/classify_ci_changes.py" not in classifier_run

    dependency_if = " ".join(jobs["dependency-review"]["if"].split())
    codeql_if = " ".join(jobs["codeql"]["if"].split())
    for job_id in ("dependency-review", "codeql"):
        assert jobs[job_id]["needs"] == "classify-changes"
        assert "!cancelled()" in jobs[job_id]["if"]
        assert "always()" not in jobs[job_id]["if"]
        assert "needs.classify-changes.outputs.docs_only != 'true'" in jobs[job_id]["if"]
    assert "github.event_name == 'pull_request'" in dependency_if
    assert "github.event.repository.private == false" in dependency_if
    assert "vars.ENABLE_GITHUB_ADVANCED_SECURITY == 'true'" in dependency_if
    assert "github.event.repository.private == false" in codeql_if
    assert "vars.ENABLE_GITHUB_ADVANCED_SECURITY == 'true'" in codeql_if


def test_dco_stays_unconditional_and_has_no_path_filter() -> None:
    dco = _load("dco.yml")

    _assert_no_path_filter(dco)
    assert "if" not in dco["jobs"]["dco"]
    assert "needs" not in dco["jobs"]["dco"]


def test_non_pr_workflow_triggers_are_preserved() -> None:
    ci = _load("ci.yml")
    security = _load("security.yml")

    assert ci["on"]["push"] == {"branches": ["main"]}
    assert security["on"]["push"] == {"branches": ["main"]}
    assert security["on"]["schedule"] == [{"cron": "23 7 * * 1"}]
    assert "workflow_dispatch" in security["on"]


def test_changed_workflows_pin_every_action_to_a_commit() -> None:
    for workflow_name in ("ci.yml", "security.yml"):
        for uses in _all_uses(_load(workflow_name)):
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", uses), uses


def test_changed_workflows_do_not_persist_checkout_credentials() -> None:
    for workflow_name in ("ci.yml", "security.yml"):
        checkout_steps = [step for step in _all_steps(_load(workflow_name)) if step.get("uses", "").startswith("actions/checkout@")]
        assert checkout_steps
        assert all(step.get("with", {}).get("persist-credentials") == "false" for step in checkout_steps)
