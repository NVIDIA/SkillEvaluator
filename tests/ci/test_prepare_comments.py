# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comments-only publication must share native output and verified evidence."""

import copy
import re
from pathlib import Path

import pytest
import test_comment_report as feedback_fixtures
import test_publish_reports as evidence_fixtures
import yaml
from test_publish_reports import trusted_workflow_pins  # noqa: F401 -- shared signed-receipt fixture

from ci import prepare_comments as comments

pytestmark = pytest.mark.usefixtures("trusted_workflow_pins")
REPOSITORY = "owner/consumer"


class FakeAPI(evidence_fixtures.FakeAPI):
    def __init__(self, *, payload=None, record=None, **kwargs):
        record = record or {
            **evidence_fixtures.run_record(),
            "event": "pull_request",
            "pull_requests": [{"number": 1}],
        }
        super().__init__(record=record, **kwargs)
        self.latest = copy.deepcopy(self.record)
        self.payload = payload or evidence_fixtures.valid_archive()
        self.feedback = feedback_fixtures.FakeAPI()
        self.downloads = []

    def get(self, path):
        if path.endswith("/actions/runs/123"):
            self.paths.append(path)
            return self.latest
        if "/issues/" in path:
            self.paths.append(path)
            return self.feedback.get(path)
        return super().get(path)

    def post(self, path, payload):
        return self.feedback.post(path, payload)

    def patch(self, path, payload):
        return self.feedback.patch(path, payload)

    def download_artifact(self, artifact_id):
        self.downloads.append(artifact_id)
        if artifact_id == 456:
            return self.payload
        return super().download_artifact(artifact_id)


def invoke(api, output):
    return comments.prepare_comments(api, REPOSITORY, 123, 1, output)


def assert_no_writes(api):
    assert not any(call[0] in {"POST", "PATCH"} for call in api.feedback.calls)


@pytest.mark.parametrize("status", ["passed", "failed", "skipped", "advisory"])
def test_comment_is_byte_identical_to_native_renderer_with_scores_fixes_and_noop(tmp_path, status):
    from ci import native_reports

    summary = evidence_fixtures.summary()
    report = evidence_fixtures.report()
    if status in {"failed", "advisory"}:
        summary = evidence_fixtures.failed_summary(advisory_downgraded=status == "advisory")
        report = evidence_fixtures.failed_report()
    if status == "skipped":
        summary = evidence_fixtures.skipped_summary()
    report["quality_summary"] = [{"overall_score": 85.8, "grade": "B", "dimensions": {}}]
    files = [("summary.json", summary)]
    if status != "skipped":
        files.append(("skill-001/skillevaluator-tier1.json", report))
    # The adapter never trusts artifact Markdown or executes artifact code.
    files.extend([("pr-comment.md", b"forged score @everyone"), ("payload.py", b"raise RuntimeError('executed')")])
    api = FakeAPI(payload=evidence_fixtures.archive(files))
    if status == "failed":
        api.record["conclusion"] = api.latest["conclusion"] = "failure"
    output = tmp_path / "feedback"
    assert invoke(api, output)["comment_action"] == "created"
    expected = tmp_path / "native"
    comments.publish_reports._dump(expected / "summary.json", summary)
    if status != "skipped":
        comments.publish_reports._dump(expected / "skill-001/skillevaluator-tier1.json", report)
    native_reports.render_native_reports(
        expected,
        summary,
        {
            "run_url": f"https://github.com/{REPOSITORY}/actions/runs/123",
            "commit": "a" * 40,
            "image": "",
            "llm_enabled": None,
            "ci_conclusion": api.record["conclusion"],
        },
    )
    body = (output / "pr-comment.md").read_bytes()
    assert body == (expected / "pr-comment.md").read_bytes()
    assert body.decode() in api.feedback.comments[0]["body"]
    assert b"forged score" not in body
    assert not list(output.rglob("*.html"))
    assert not (output / "payload.py").exists()
    assert not any("/pages" in path for path in api.paths)
    if status == "skipped":
        assert b"NO SKILLS CHANGED" in body and b"did not run" in body
        assert b"85.8" not in body
    else:
        assert b"85.8 / B" in body
        assert (b"What to fix" in body) is (status in {"failed", "advisory"})
    # A second invocation follows the same existing upsert protocol.
    assert invoke(api, output)["comment_action"] == "unchanged"


@pytest.mark.parametrize("changes", [{"run_attempt": 2}, {"status": "in_progress"}])
def test_stale_attempt_skips_before_artifact_download(tmp_path, changes):
    api = FakeAPI()
    api.latest.update(changes)
    assert invoke(api, tmp_path)["comment_action"] == "skipped"
    assert api.downloads == []
    assert_no_writes(api)


@pytest.mark.parametrize("failure", ["closed", "old_head", "fork", "ambiguous", "push"])
def test_only_unique_current_same_repository_pr_gets_feedback(tmp_path, failure):
    api = FakeAPI()
    if failure == "old_head":
        api.branch_sha = "e" * 40
    elif failure == "fork":
        api.record["head_repository"] = {"full_name": "attacker/fork"}
    elif failure == "ambiguous":
        api.record["pull_requests"] = [{"number": 1}, {"number": 2}]
    elif failure == "push":
        api.record["event"] = "push"
    elif failure == "closed":
        original_get = api.get

        def get(path):
            result = original_get(path)
            if "/pulls/" in path:
                result["state"] = "closed"
            return result

        api.get = get
    assert invoke(api, tmp_path)["comment_action"] == "skipped"
    assert_no_writes(api)


@pytest.mark.parametrize(
    "failure", ["missing_json", "malformed", "bad_caller", "bad_evaluator", "bad_receipt", "verdict", "noop_head"]
)
def test_untrusted_or_inconsistent_evidence_never_posts(tmp_path, failure, monkeypatch):
    api = FakeAPI()
    if failure == "missing_json":
        api.payload = evidence_fixtures.archive([("pr-comment.md", b"PASS")])
    elif failure == "malformed":
        api.payload = b"not a ZIP"
    elif failure == "bad_caller":
        monkeypatch.setenv("TRUSTED_CALLER_BLOBS", "e" * 40)
    elif failure == "bad_evaluator":
        monkeypatch.setenv("TRUSTED_EVALUATOR_SHA", "e" * 40)
    elif failure == "bad_receipt":
        original = api.download_artifact
        api.download_artifact = lambda identifier: b"forged" if identifier == 457 else original(identifier)
    elif failure == "verdict":
        api.record["conclusion"] = "failure"
    elif failure == "noop_head":
        summary = evidence_fixtures.skipped_summary()
        summary["selection"]["head"] = "e" * 40
        api.payload = evidence_fixtures.archive([("summary.json", summary)])
    with pytest.raises(comments.publish_reports.PublicationError):
        invoke(api, tmp_path)
    assert_no_writes(api)


def test_rerun_started_during_rendering_suppresses_comment(tmp_path, monkeypatch):
    api = FakeAPI()
    render = comments.report_views.render_reports

    def changing_render(*args, **kwargs):
        render(*args, **kwargs)
        api.latest.update(run_attempt=2, status="in_progress")

    monkeypatch.setattr(comments.report_views, "render_reports", changing_render)
    assert invoke(api, tmp_path)["comment_action"] == "skipped"
    assert_no_writes(api)


def test_comments_workflow_privileges_and_trusted_code_boundary():
    root = Path(__file__).parents[2]
    workflow = yaml.load((root / ".github/workflows/comment-skill-checks.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["permissions"] == {}
    assert set(workflow["on"]) == {"workflow_call"}
    inputs = workflow["on"]["workflow_call"]["inputs"]
    for name in ("run-id", "run-attempt", "tools-ref", "trusted-caller-blobs"):
        assert inputs[name]["required"] == "true"
    assert "report-base-url" not in inputs and "retention-runs" not in inputs
    assert set(workflow["jobs"]) == {"comment"}
    job = workflow["jobs"]["comment"]
    assert job["permissions"] == {"contents": "read", "actions": "read", "pull-requests": "write"}
    for guard in (
        "workflow_run",
        "workflow_dispatch",
        "github.event.workflow_run.status == 'completed'",
        "github.event.repository.default_branch",
        "github.event.workflow_run.event == 'pull_request'",
        "github.event.workflow_run.head_repository.full_name == github.repository",
    ):
        assert guard in job["if"]
    assert job["concurrency"] == {
        "group": "skill-report-comment-${{ github.repository }}",
        "cancel-in-progress": "false",
        "queue": "max",
    }
    checkouts = [step for step in job["steps"] if "uses" in step]
    assert len(checkouts) == 1
    assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", checkouts[0]["uses"])
    assert checkouts[0]["with"]["repository"] == "${{ inputs.tools-repository }}"
    assert checkouts[0]["with"]["ref"] == "${{ inputs.tools-ref }}"
    assert checkouts[0]["with"]["persist-credentials"] == "false"
    hosted = yaml.load((root / ".github/workflows/publish-skill-reports.yml").read_text(), Loader=yaml.BaseLoader)
    installs = [step for step in job["steps"] if step.get("working-directory") == "publisher"]
    original = next(step for step in hosted["jobs"]["prepare"]["steps"] if step.get("working-directory") == "publisher")
    assert installs == [original]
    assert hosted["jobs"]["comment"]["permissions"]["actions"] == "read"
    last = job["steps"][-1]
    assert last["run"] == "publisher/.venv/bin/python publisher/ci/prepare_comments.py"
    for name in (
        "TRUSTED_EVALUATOR_SHA",
        "TRUSTED_CALLER_BLOBS",
        "TRUSTED_WRAPPER_WORKFLOWS",
        "DELEGATED_EVALUATOR_WRAPPER",
    ):
        assert last["env"][name] == hosted["jobs"]["prepare"]["steps"][4]["env"][name]
