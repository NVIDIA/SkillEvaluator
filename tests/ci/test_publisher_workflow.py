# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow contracts for attempt-bound publication and queued PR feedback."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def load(path):
    return yaml.load((ROOT / path).read_text(), Loader=yaml.BaseLoader)


def test_completed_event_and_manual_backfill_pass_an_explicit_attempt():
    caller = load("ci/templates/publish-skill-reports.yml")
    assert caller["on"]["workflow_dispatch"]["inputs"]["run-attempt"]["required"] == "true"
    assert (
        caller["jobs"]["reports"]["with"]["run-attempt"]
        == "${{ format('{0}', github.event.workflow_run.run_attempt || inputs.run-attempt) }}"
    )
    workflow = load(".github/workflows/publish-skill-reports.yml")
    assert workflow["on"]["workflow_call"]["inputs"]["run-attempt"]["required"] == "true"
    step = next(step for step in workflow["jobs"]["prepare"]["steps"] if step.get("id") == "publish")
    assert step["env"]["REPORT_RUN_ATTEMPT"] == "${{ inputs.run-attempt }}"


def test_privileged_preparation_has_pr_read_and_queued_deploy_preserves_feedback():
    jobs = load(".github/workflows/publish-skill-reports.yml")["jobs"]
    assert jobs["prepare"]["permissions"]["pull-requests"] == "read"
    for name in ("deploy", "comment"):
        assert jobs[name]["concurrency"]["queue"] == "max"
        assert jobs[name]["concurrency"]["cancel-in-progress"] == "false"
    assert "needs.deploy.result == 'success'" in jobs["comment"]["if"]


def test_skipped_stale_run_does_not_deploy_or_upload_feedback():
    jobs = load(".github/workflows/publish-skill-reports.yml")["jobs"]
    assert jobs["prepare"]["outputs"]["skipped"] == "${{ steps.publish.outputs.skipped }}"
    assert "needs.prepare.outputs.skipped != 'true'" in jobs["deploy"]["if"]
    upload = next(step for step in jobs["prepare"]["steps"] if "upload-artifact@" in step.get("uses", ""))
    assert "steps.publish.outputs.skipped != 'true'" in upload["if"]


def test_feedback_artifact_is_reusable_when_only_a_failed_job_is_retried():
    jobs = load(".github/workflows/publish-skill-reports.yml")["jobs"]
    upload = next(step for step in jobs["prepare"]["steps"] if "upload-artifact@" in step.get("uses", ""))
    assert upload["with"]["name"] == "report-feedback"
    assert upload["with"]["overwrite"] == "true"
    download = next(step for step in jobs["comment"]["steps"] if "gh run download" in step.get("run", ""))
    assert "--name report-feedback" in download["run"]


def test_publisher_uses_verified_node24_action_releases():
    jobs = load(".github/workflows/publish-skill-reports.yml")["jobs"]
    uses = {step.get("uses", "") for job in jobs.values() for step in job["steps"]}
    assert "actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9" in uses
    assert "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9" in uses
    assert "actions/deploy-pages@368f82528645a54fb793d4d04e342629a3f51346" in uses


def test_privileged_publisher_installs_the_reviewed_hash_locked_bootstrap():
    steps = load(".github/workflows/publish-skill-reports.yml")["jobs"]["prepare"]["steps"]
    install = next(step for step in steps if step.get("name") == "Install the trusted native report renderer")
    assert install["working-directory"] == "publisher"
    assert "python3 -m pip install --require-hashes -r ci/requirements/uv.txt" in install["run"]
    assert "uv sync --frozen --no-dev --no-editable" in install["run"]
