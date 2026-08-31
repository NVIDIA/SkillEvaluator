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


def _workflow_names() -> list[str]:
    """Every workflow on disk, so a new one is covered the day it lands."""
    names = sorted(path.name for pattern in ("*.yml", "*.yaml") for path in WORKFLOWS.glob(pattern))
    assert names, "no workflows found"
    return names


def _assert_no_path_filter(workflow: dict[str, Any], event: str = "pull_request") -> None:
    trigger = workflow["on"][event]
    if isinstance(trigger, dict):
        assert "paths" not in trigger
        assert "paths-ignore" not in trigger


def _runs(job: dict[str, Any]) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def _all_uses(workflow: dict[str, Any]) -> list[str]:
    """Every action reference: step-level actions and job-level reusable workflows."""
    step_uses = [step["uses"] for job in workflow["jobs"].values() for step in job.get("steps", []) if "uses" in step]
    job_uses = [job["uses"] for job in workflow["jobs"].values() if "uses" in job]
    return step_uses + job_uses


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
    assert 'npm install --global --ignore-scripts --omit=optional "fern-api@$FERN_VERSION"' in docs_step["run"]
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


def test_gitleaks_uses_event_specific_full_history_scopes() -> None:
    job = _load("security.yml")["jobs"]["gitleaks"]
    checkout = next(
        step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@")
    )
    verify = next(step for step in job["steps"] if step.get("name") == "Verify full Git checkout")
    scan = next(step for step in job["steps"] if step.get("name") == "Scan Git history")

    assert str(checkout["with"]["fetch-depth"]) == "0"
    assert 'test "$(git rev-parse --is-shallow-repository)" = "false"' in verify["run"]
    assert 'git rev-parse --verify "HEAD^{commit}"' in verify["run"]
    assert (
        "gitleaks:v8.30.0@sha256:691af3c7c5a48b16f187ce3446d5f194838f91238f27270ed36eef6359a574d9"
        in scan["run"]
    )
    scan_run = scan["run"]
    head_opts = "--full-history --diff-filter=tuxdb HEAD --"
    audit_opts = "--full-history --all --diff-filter=tuxdb --"
    assignments = re.findall(r'log_opts="([^"]+)"', scan_run)

    assert assignments == [head_opts, audit_opts]
    assert scan["shell"] == "bash"
    assert "set -euo pipefail" in scan_run
    assert 'case "$GITHUB_EVENT_NAME" in' in scan_run
    assert re.search(
        rf'pull_request\|push\)\s+log_opts="{re.escape(head_opts)}"\s+;;', scan_run
    )
    assert re.search(rf'\*\)\s+log_opts="{re.escape(audit_opts)}"\s+;;', scan_run)
    assert '--log-opts="$log_opts"' in scan_run


def test_dco_stays_unconditional_and_has_no_path_filter() -> None:
    dco = _load("dco.yml")

    _assert_no_path_filter(dco)
    assert "if" not in dco["jobs"]["dco"]
    assert "needs" not in dco["jobs"]["dco"]


def test_non_pr_workflow_triggers_are_preserved() -> None:
    ci = _load("ci.yml")
    security = _load("security.yml")

    assert ci["on"]["push"] == {"branches": ["main"]}
    assert set(security["on"]) == {"pull_request", "push", "schedule", "workflow_dispatch"}
    assert security["on"]["push"] == {"branches": ["main"]}
    assert security["on"]["schedule"] == [{"cron": "23 7 * * 1"}]
    assert "workflow_dispatch" in security["on"]


def _is_local_reference(uses: str) -> bool:
    """A same-repo composite action or reusable workflow, e.g. ``./.github/actions/x``.

    GitHub always resolves these from the caller's own commit, so nothing about
    them can float and there is no ``@ref`` to pin.
    """
    return uses.startswith("./")


def _local_reference_escapes_repo(uses: str) -> bool:
    return ".." in Path(uses).parts


def test_every_workflow_pins_every_action_to_a_commit() -> None:
    for workflow_name in _workflow_names():
        for uses in _all_uses(_load(workflow_name)):
            if _is_local_reference(uses):
                assert not _local_reference_escapes_repo(uses), f"{workflow_name}: {uses}"
                continue
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", uses), f"{workflow_name}: {uses}"


def test_every_workflow_does_not_persist_checkout_credentials() -> None:
    checkout_steps = [
        (workflow_name, step)
        for workflow_name in _workflow_names()
        for step in _all_steps(_load(workflow_name))
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert checkout_steps
    for workflow_name, step in checkout_steps:
        assert step.get("with", {}).get("persist-credentials") == "false", workflow_name


def test_publish_docs_installs_the_fern_version_the_repository_pins() -> None:
    """Docs are checked and published by the same CLI version.

    ci.yml derives it from fern/fern.config.json; publishing must not hardcode a
    second version that can drift from the one validation ran against.
    """
    job = _load("publish-docs.yml")["jobs"]["run"]
    install_step = next((step for step in job["steps"] if "npm install" in step.get("run", "")), None)

    assert install_step is not None, "publish-docs.yml no longer installs the Fern CLI with npm"
    assert "fern/fern.config.json" in install_step["run"]
    assert "fern-api@$FERN_VERSION" in install_step["run"]
    assert not re.search(r"fern-api@\d", install_step["run"]), "Fern CLI version is hardcoded"


def test_every_workflow_declares_least_privilege_permissions() -> None:
    """A job-level permissions: block overrides the workflow-level one, so check both."""
    for workflow_name in _workflow_names():
        workflow = _load(workflow_name)
        assert workflow.get("permissions") is not None, (
            f"{workflow_name} inherits the repository default token scope"
        )
        for scope, permissions in [("workflow", workflow["permissions"])] + [
            (f"job {job_id}", job["permissions"]) for job_id, job in workflow["jobs"].items() if "permissions" in job
        ]:
            assert permissions != "write-all", f"{workflow_name}: {scope}"


NPM_INSTALL_LINE = re.compile(r"^.*\bnpm install\b.*$", re.MULTILINE)


def test_every_workflow_npm_install_ignores_lifecycle_scripts() -> None:
    """A floating transitive dependency must not get to run install-time code.

    npm re-resolves the whole tree on every install, so pinning the top-level
    package does not pin what its dependencies can execute at install time.
    """
    for workflow_name in _workflow_names():
        for step in _all_steps(_load(workflow_name)):
            for line in NPM_INSTALL_LINE.findall(step.get("run", "")):
                assert "--ignore-scripts" in line, f"{workflow_name}: {line.strip()}"
