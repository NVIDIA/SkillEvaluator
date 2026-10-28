# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow contracts for attempt-bound publication and queued PR feedback."""

import os
import re
import subprocess
from pathlib import Path

import pytest
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
    lines = [line.strip() for line in install["run"].replace("\\\n", " ").splitlines()]
    # uv comes from its hash-locked requirement, installed into a private environment.
    assert lines[0] == 'python3 -m venv "$RUNNER_TEMP/uv"'
    assert lines[1].startswith('"$RUNNER_TEMP/uv/bin/python" -m pip install ')
    for flag in ("--require-hashes", "--no-deps", "--only-binary :all:", "-r ci/requirements/uv.txt"):
        assert flag in lines[1]
    assert lines[2] == '"$RUNNER_TEMP/uv/bin/uv" sync --frozen --no-dev --no-editable'
    assert len(lines) == 3
    # The runner's system Python and pip are never modified.
    assert "python3 -m pip" not in install["run"]
    # This job can push to gh-pages, so it never restores a dependency cache that
    # other default-branch workflows could have written.
    for step in steps:
        assert "actions/cache" not in step.get("uses", "")
        assert "setup-uv" not in step.get("uses", "")


def expression(text: str, context: dict) -> bool:
    """Evaluate the small GitHub expression subset these job conditions use."""
    tokens = re.findall(r"'[^']*'|==|!=|&&|\|\||!|\(|\)|always\(\)|[A-Za-z_][\w.-]*", text)
    assert "".join(tokens) == re.sub(r"\s+", "", text)
    translated = {"&&": " and ", "||": " or ", "!": " not ", "always()": "True"}
    python = []
    for token in tokens:
        if token.startswith("'"):
            python.append(repr(token[1:-1]))
        elif token in translated:
            python.append(translated[token])
        elif token in {"==", "!=", "(", ")"}:
            python.append(token)
        else:
            python.append(repr(context[token]))
    # The input is a workflow file from this repository, translated token by token.
    return eval("".join(python), {"__builtins__": {}})


def job_context(committed, event="workflow_run", attempt="1", deploy="success", skipped=""):
    return {
        "needs.prepare.result": "success",
        "needs.prepare.outputs.skipped": skipped,
        "needs.prepare.outputs.committed": committed,
        "needs.prepare.outputs.pr_number": "1",
        "needs.prepare.outputs.is_current": "true",
        "needs.deploy.result": deploy,
        "github.event_name": event,
        "github.run_attempt": attempt,
    }


@pytest.mark.parametrize(
    ("committed", "event", "attempt", "deploys"),
    [
        ("true", "workflow_run", "1", True),
        ("false", "workflow_run", "1", False),
        ("false", "workflow_dispatch", "1", True),
        ("false", "workflow_run", "2", True),
        ("true", "workflow_run", "3", True),
    ],
)
def test_deploy_is_skipped_only_when_an_automatic_first_attempt_committed_nothing(committed, event, attempt, deploys):
    deploy = load(".github/workflows/publish-skill-reports.yml")["jobs"]["deploy"]
    assert deploy["needs"] == "prepare"
    # No status function, so GitHub also requires prepare to succeed.
    assert "always()" not in deploy["if"]
    assert expression(deploy["if"], job_context(committed, event, attempt)) is deploys


@pytest.mark.parametrize(
    ("committed", "deploy", "comments"),
    [
        ("true", "success", True),
        ("false", "success", True),
        ("false", "skipped", True),
        ("true", "skipped", False),
        ("true", "failure", False),
        ("false", "failure", False),
        ("true", "cancelled", False),
    ],
)
def test_comment_needs_a_successful_deploy_for_new_content(committed, deploy, comments):
    job = load(".github/workflows/publish-skill-reports.yml")["jobs"]["comment"]
    assert expression(job["if"], job_context(committed, deploy=deploy)) is comments
    assert not expression(job["if"], {**job_context(committed, deploy=deploy), "needs.prepare.result": "failure"})


@pytest.mark.parametrize("committed", ["true", "false"])
@pytest.mark.parametrize(("event", "attempt"), [("workflow_run", "1"), ("workflow_dispatch", "1"), ("workflow_run", "2")])
def test_skipped_stale_publication_never_deploys_or_comments(committed, event, attempt):
    # A stale run can follow an earlier attempt that already wrote pr_number and is_current.
    jobs = load(".github/workflows/publish-skill-reports.yml")["jobs"]
    for deploy in ("success", "skipped"):
        context = job_context(committed, event, attempt, deploy, skipped="true")
        assert not expression(jobs["deploy"]["if"], context)
        assert not expression(jobs["comment"]["if"], context)


def run_publish_loop(tmp_path, *, diffs, pushes):
    """Run the real publish step shell with stand-ins for git and the publisher."""
    step = next(
        step
        for step in load(".github/workflows/publish-skill-reports.yml")["jobs"]["prepare"]["steps"]
        if step.get("id") == "publish"
    )
    workspace, temp, bin_dir = tmp_path / "workspace", tmp_path / "temp", tmp_path / "bin"
    python = workspace / "publisher/.venv/bin/python"
    for directory in (workspace / "site", python.parent, temp, bin_dir):
        directory.mkdir(parents=True, exist_ok=True)
    python.write_text('#!/bin/sh\nmkdir -p "$OUTPUT_DIR"\necho summary > "$OUTPUT_DIR/summary.md"\n')
    git = bin_dir / "git"
    # Each `diff --cached --quiet` and `push` pops its exit code from a queue file.
    git.write_text(
        "#!/bin/sh\n"
        'echo "$*" >> "$CALLS"\n'
        'pop() { code=$(head -n 1 "$1"); tail -n +2 "$1" > "$1.next"; mv "$1.next" "$1"; exit "$code"; }\n'
        'case "$3" in diff) pop "$DIFFS";; push) pop "$PUSHES";; esac\n'
    )
    for script in (python, git):
        script.chmod(0o755)
    (tmp_path / "diffs").write_text("".join(f"{code}\n" for code in diffs))
    (tmp_path / "pushes").write_text("".join(f"{code}\n" for code in pushes))
    output = tmp_path / "github-output"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_WORKSPACE": str(workspace),
        "RUNNER_TEMP": str(temp),
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "step-summary"),
        "REPORT_RUN_ID": "123",
        "CALLS": str(tmp_path / "calls"),
        "DIFFS": str(tmp_path / "diffs"),
        "PUSHES": str(tmp_path / "pushes"),
    }
    subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
        cwd=workspace,
        env=env,
        check=True,
        capture_output=True,
    )
    calls = (tmp_path / "calls").read_text().splitlines()
    return output.read_text().splitlines(), calls


@pytest.mark.parametrize(
    ("diffs", "pushes", "committed", "commits"),
    [
        ([0], [], "false", 0),
        ([1], [0], "true", 1),
        ([1, 1], [1, 0], "true", 2),
        # A concurrent writer already published this exact run attempt.
        ([1, 0], [1], "false", 1),
    ],
)
def test_publish_step_reports_whether_it_pushed_new_content(tmp_path, diffs, pushes, committed, commits):
    output, calls = run_publish_loop(tmp_path, diffs=diffs, pushes=pushes)
    assert output == [f"committed={committed}"]
    assert sum(call.startswith("-C site commit") for call in calls) == commits
    assert load(".github/workflows/publish-skill-reports.yml")["jobs"]["prepare"]["outputs"]["committed"] == (
        "${{ steps.publish.outputs.committed }}"
    )
