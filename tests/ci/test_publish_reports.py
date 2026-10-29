# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundaries for privileged publication of untrusted CI report data."""

import importlib.util
import io
import json
import stat
import urllib.error
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("publish_reports", Path(__file__).parents[2] / "ci/publish_reports.py")
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


@pytest.fixture(autouse=True)
def trusted_workflow_pins(monkeypatch):
    monkeypatch.setenv("TRUSTED_EVALUATOR_SHA", "c" * 40)
    monkeypatch.setenv("TRUSTED_CALLER_BLOBS", "b" * 40)


def archive(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as handle:
        for name, content in files:
            handle.writestr(name, content if isinstance(content, bytes) else json.dumps(content))
    return output.getvalue()


def summary():
    return {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "incomplete": 0,
        "overall_passed": True,
        "overall_status": "passed",
        "total_validators": 1,
        "total_errors": 0,
        "total_warnings": 0,
        "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        "skills": [
            {
                "path": "skills/example",
                "report_dir": "skill-001",
                "exit_code": 0,
                "overall_status": "passed",
                "incomplete_scans": [],
                "passed": True,
                "total_validators": 1,
                "total_errors": 0,
                "total_warnings": 0,
                "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            }
        ],
    }


def report():
    return {
        "overall_passed": True,
        "overall_status": "passed",
        "incomplete_scans": [],
        "total_validators": 1,
        "total_errors": 0,
        "total_warnings": 0,
        "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        "results": [
            {
                "validator": "Example",
                "description": "Example validator",
                "findings": [],
                "success_details": [],
                "legacy": {"errors": [], "warnings": [], "messages": []},
                "passed": True,
                "status": "passed",
                "incomplete_scans": [],
                "summary": {
                    "files_scanned": 1,
                    "checks_performed": 1,
                    "errors": 0,
                    "warnings": 0,
                    "critical_count": 0,
                    "high_count": 0,
                    "medium_count": 0,
                    "low_count": 0,
                },
            }
        ],
    }


def valid_archive():
    return archive(
        [
            ("summary.json", summary()),
            ("skill-001/skillevaluator-tier1.json", report()),
            ("skill-001/skillevaluator-tier1.html", b"<script>bad()</script>"),
        ]
    )


def test_only_json_is_loaded_and_html_is_ignored():
    loaded, reports = publisher.read_artifact(valid_archive())
    assert loaded["total"] == 1
    assert list(reports) == ["skill-001"]


@pytest.mark.parametrize(
    "name", ["../summary.json", "/summary.json", "a/../../summary.json", "C:\\summary.json", "a\\summary.json"]
)
def test_zip_paths_cannot_escape(name):
    with pytest.raises(publisher.PublicationError):
        publisher.read_artifact(archive([(name, {})]))


def test_duplicate_members_are_rejected():
    with pytest.warns(UserWarning), pytest.raises(publisher.PublicationError):
        publisher.read_artifact(archive([("summary.json", {}), ("summary.json", {})]))


def test_symlink_members_are_rejected():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as handle:
        info = zipfile.ZipInfo("summary.json")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        handle.writestr(info, "outside")
    with pytest.raises(publisher.PublicationError):
        publisher.read_artifact(output.getvalue())


def test_archive_count_and_expansion_limits(monkeypatch):
    monkeypatch.setattr(publisher, "MAX_FILES", 1)
    with pytest.raises(publisher.PublicationError):
        publisher.read_artifact(valid_archive())
    monkeypatch.setattr(publisher, "MAX_FILES", 1000)
    monkeypatch.setattr(publisher, "MAX_TOTAL_BYTES", 10)
    with pytest.raises(publisher.PublicationError):
        publisher.read_artifact(valid_archive())


@pytest.mark.parametrize(
    "change",
    [
        {"total": 2},
        {"passed": 0},
        {"overall_passed": "true"},
        {"skills": [{**summary()["skills"][0], "report_dir": "../bad"}]},
    ],
)
def test_summary_schema_and_counts_are_checked(change):
    with pytest.raises(publisher.PublicationError):
        publisher.read_artifact(
            archive([("summary.json", {**summary(), **change}), ("skill-001/skillevaluator-tier1.json", report())])
        )


def test_missing_report_and_mismatched_verdict_are_rejected():
    with pytest.raises(publisher.PublicationError):
        publisher.read_artifact(archive([("summary.json", summary())]))
    with pytest.raises(publisher.PublicationError):
        publisher.read_artifact(
            archive(
                [
                    ("summary.json", summary()),
                    ("skill-001/skillevaluator-tier1.json", {**report(), "overall_passed": False}),
                ]
            )
        )


def test_no_target_configuration_error_can_be_published():
    payload = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "incomplete": 0,
        "overall_passed": False,
        "overall_status": "error",
        "skills": [],
        "error": "No targets",
        "total_validators": 0,
        "total_errors": 0,
        "total_warnings": 0,
        "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0},
    }
    assert publisher.read_artifact(archive([("summary.json", payload)]))[0] == payload


def test_alias_order_is_run_then_attempt():
    assert publisher.is_newer({"run_id": 5, "run_attempt": 2}, {"run_id": 5, "run_attempt": 1})
    assert not publisher.is_newer({"run_id": 4, "run_attempt": 99}, {"run_id": 5, "run_attempt": 1})


def test_large_raw_artifact_ignores_non_json_members_without_extracting():
    files = [("summary.json", summary()), ("skill-001/skillevaluator-tier1.json", report())]
    files.extend((f"ignored-{index}.html", b"<script>never_execute()</script>") for index in range(700))
    assert publisher.read_artifact(archive(files))[0]["total"] == 1


def test_redirect_does_not_forward_api_authorization():
    api = publisher.GitHubAPI("owner/consumer", "private-token")
    requests = []

    class Response(io.BytesIO):
        @property
        def headers(self):
            return {}

    class Opener:
        def open(self, request, timeout):
            assert timeout == 60
            requests.append(request)
            if len(requests) == 1:
                raise urllib.error.HTTPError(
                    request.full_url,
                    302,
                    "redirect",
                    {
                        "Location": "https://productionresultssa1.blob.core.windows.net/archive?sig=test",
                    },
                    None,
                )
            return Response(b"archive-bytes")

    api.opener = Opener()
    assert api.download_artifact(456) == b"archive-bytes"
    assert requests[0].get_header("Authorization") == "Bearer private-token"
    assert requests[1].get_header("Authorization") is None


def test_pr_provenance_compares_recorded_revisions_through_real_github_client():
    api = publisher.GitHubAPI("owner/consumer", "private-token")
    record = run_record()
    record["event"] = "pull_request"
    record["pull_requests"] = [{"base": {"sha": "d" * 40}, "head": {"sha": "a" * 40}}]
    paths = []

    class Response(io.BytesIO):
        @property
        def headers(self):
            return {}

    class Opener:
        def open(self, request, timeout):
            assert timeout == 60
            paths.append(request.full_url)
            if "/compare/" in request.full_url:
                data = {"merge_base_commit": {"sha": "e" * 40}}
            else:
                sha = request.full_url.rsplit("?ref=", 1)[1]
                data = {"type": "file", "path": publisher.WORKFLOW_PATH, "sha": ("b" if sha == "a" * 40 else "f") * 40}
            return Response(json.dumps(data).encode())

    api.opener = Opener()
    publisher._validate_workflow_provenance(api, "owner/consumer", record)
    assert any("/compare/" in path for path in paths)


@pytest.mark.parametrize(
    "path",
    [
        "/repos/owner/consumer-other/actions/runs",
        "/repos/owner/consumer/../other/actions",
        "/repos/owner/consumer/compare/" + "a" * 40 + "..." + "b" * 40 + "/extra",
        "/repos/owner/consumer/compare/" + "a" * 40 + "..." + "b" * 39,
    ],
)
def test_github_client_still_rejects_untrusted_paths(path):
    api = publisher.GitHubAPI("owner/consumer", "private-token")
    with pytest.raises(publisher.PublicationError):
        api._request(path)


def test_github_client_accepts_repository_root_for_default_branch_lookup():
    api = publisher.GitHubAPI("owner/consumer", "private-token")

    class Response(io.BytesIO):
        @property
        def headers(self):
            return {}

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://api.github.com/repos/owner/consumer"
            assert timeout == 60
            return Response(b'{"default_branch":"main"}')

    api.opener = Opener()
    assert api.get("/repos/owner/consumer")["default_branch"] == "main"


def test_signed_download_hosts_are_explicitly_allowlisted():
    assert publisher.allowed_download_url("https://productionresultssa1.blob.core.windows.net/results/a?sig=x")
    assert publisher.allowed_download_url("https://objects.githubusercontent.com/a")
    assert not publisher.allowed_download_url("http://objects.githubusercontent.com/a")
    assert not publisher.allowed_download_url("https://objects.githubusercontent.com.evil.invalid/a")
    assert not publisher.allowed_download_url("https://token@objects.githubusercontent.com/a")


def run_record():
    return {
        "id": 123,
        "run_attempt": 1,
        "status": "completed",
        "conclusion": "success",
        "path": ".github/workflows/skill-checks.yml",
        "event": "push",
        "head_branch": "main",
        "head_sha": "a" * 40,
        "repository": {"full_name": "owner/consumer"},
        "head_repository": {"full_name": "owner/consumer"},
        "pull_requests": [],
        "run_started_at": "2026-10-21T12:00:00Z",
        "updated_at": "2026-10-21T12:02:00Z",
        "referenced_workflows": [
            {"path": "NVIDIA/SkillEvaluator/.github/workflows/evaluate-skills.yml@" + "c" * 40, "sha": "c" * 40}
        ],
    }


def fork_record(**changes):
    # GitHub never fills pull_requests for a run from a fork PR.
    return {
        **run_record(),
        "event": "pull_request",
        "head_branch": "feature/fork",
        "head_repository": {"full_name": "contributor/consumer"},
        **changes,
    }


def fork_pull(head_sha="a" * 40, *, base_sha="d" * 40, repo=None):
    return {
        "number": 7,
        "state": "open",
        "head": {"sha": head_sha, "repo": repo or {"full_name": "contributor/consumer"}},
        "base": {"sha": base_sha, "repo": {"full_name": "owner/consumer"}},
    }


class FakeAPI:
    def __init__(self, *, branch_sha="a" * 40, record=None):
        self.record = record or run_record()
        self.branch_sha = branch_sha
        self.paths = []

    def get(self, path):
        self.paths.append(path)
        if path.endswith("/actions/runs/123/attempts/1"):
            return self.record
        if "/contents/.github/workflows/skill-checks.yml?ref=" in path:
            return {"type": "file", "path": publisher.WORKFLOW_PATH, "sha": "b" * 40}
        if path.endswith("/artifacts?per_page=100"):
            return {
                "total_count": 1,
                "artifacts": [
                    {
                        "id": 456,
                        "name": "skill-evaluator-tier1",
                        "expired": False,
                        "size_in_bytes": len(valid_archive()),
                        "created_at": "2026-10-21T12:01:00Z",
                        "workflow_run": {"id": 123, "head_sha": "a" * 40},
                    }
                ],
            }
        if path == "/repos/owner/consumer":
            return {"default_branch": "main"}
        if "/branches/" in path:
            return {"commit": {"sha": self.branch_sha}}
        if "/pulls/" in path:
            return {
                "number": 1,
                "state": "open",
                "head": {"sha": self.branch_sha, "repo": self.record.get("head_repository")},
                "base": {"repo": {"full_name": "owner/consumer"}},
            }
        if "/pulls?state=open&head=" in path:
            return [fork_pull(self.branch_sha, repo=self.record.get("head_repository"))]
        if path.endswith("/pulls"):
            return [{"number": 1}]
        raise AssertionError(path)

    def download_artifact(self, artifact_id):
        assert artifact_id == 456
        return valid_archive()


def fake_render(directory, _summary, context):
    (directory / "index.html").write_text("trusted renderer", encoding="utf-8")
    (directory / "summary.md").write_text(context["run_url"], encoding="utf-8")
    (directory / "pr-comment.md").write_text("trusted comment", encoding="utf-8")


@pytest.mark.parametrize("child", [None, b"not valid JSON", report()])
def test_rejected_child_evidence_is_not_loaded_or_published(child):
    payload = summary()
    payload.update(
        passed=0, failed=1, incomplete=1, overall_passed=False, overall_status="incomplete", total_validators=0
    )
    payload["skills"][0].update(
        passed=False, overall_status="error", exit_code=1, total_validators=0, error="Rejected child evidence"
    )
    files = [("summary.json", payload)]
    if child is not None:
        files.append(("skill-001/skillevaluator-tier1.json", child))
    loaded, reports = publisher.read_artifact(archive(files))
    assert loaded["skills"][0]["error"] == "Rejected child evidence"
    assert reports == {}


def test_error_row_with_rejected_json_gets_a_native_unavailable_report(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "ci"))
    payload = summary()
    payload.update(
        passed=0, failed=1, incomplete=1, overall_passed=False, overall_status="incomplete", total_validators=0
    )
    payload["skills"][0].update(
        passed=False, overall_status="error", exit_code=1, total_validators=0, error="Rejected child evidence"
    )

    class ErrorAPI(FakeAPI):
        def download_artifact(self, _artifact_id):
            return archive([("summary.json", payload), ("skill-001/skillevaluator-tier1.json", b"not JSON")])

    publisher.publish(
        ErrorAPI(record={**run_record(), "conclusion": "failure"}),
        "owner/consumer",
        123,
        "https://owner.github.io/consumer",
        tmp_path / "site",
        tmp_path / "feedback",
    )
    child = tmp_path / "site/runs/123/1/skill-001"
    assert "Report unavailable" in (child / "skillevaluator-tier1.html").read_text()
    assert not (child / "skillevaluator-tier1.json").exists()
    assert "Rejected child evidence" in (tmp_path / "feedback/pr-comment.md").read_text()


def test_bad_typed_child_evidence_is_rejected_before_writing_a_receipt(tmp_path):
    class BadTypedAPI(FakeAPI):
        def download_artifact(self, _artifact_id):
            child = report()
            child["results"][0]["findings"] = "invalid findings"
            return archive([("summary.json", summary()), ("skill-001/skillevaluator-tier1.json", child)])

    with pytest.raises(publisher.PublicationError, match="invalid retained evidence"):
        publisher.publish(
            BadTypedAPI(),
            "owner/consumer",
            123,
            "https://owner.github.io/consumer",
            tmp_path / "site",
            tmp_path / "feedback",
            render=fake_render,
        )
    assert not list((tmp_path / "site").rglob("receipt.json"))


def test_unexpected_render_failure_cannot_publish_a_passing_receipt(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "ci"))

    class BadDisplayAPI(FakeAPI):
        def download_artifact(self, _artifact_id):
            payload = summary()
            payload["skills"][0]["path"] = "skills/bad\tpath"
            return archive([("summary.json", payload), ("skill-001/skillevaluator-tier1.json", report())])

    with pytest.raises(publisher.PublicationError, match="native renderer"):
        publisher.publish(
            BadDisplayAPI(),
            "owner/consumer",
            123,
            "https://owner.github.io/consumer",
            tmp_path / "site",
            tmp_path / "feedback",
        )
    assert not list((tmp_path / "site").rglob("receipt.json"))


@pytest.mark.parametrize("separator", [",", ", ", "\n"])
def test_trusted_pins_accept_documented_list_separators(monkeypatch, separator):
    monkeypatch.setenv("PINS", separator.join(["a" * 40, "b" * 40]) + "\n")
    assert publisher._trusted_pins("PINS") == {"a" * 40, "b" * 40}


@pytest.mark.parametrize("latest_status", ["in_progress", "completed"])
def test_publication_uses_triggering_attempt_even_after_a_rerun(tmp_path, latest_status):
    class RerunAPI(FakeAPI):
        def get(self, path):
            if path.endswith("/actions/runs/123"):
                return {**run_record(), "run_attempt": 2, "status": latest_status}
            result = super().get(path)
            if path.endswith("/artifacts?per_page=100"):
                result["artifacts"].append({**result["artifacts"][0], "id": 789, "created_at": "2026-10-21T13:01:00Z"})
                result["total_count"] = 2
            return result

    metadata = publisher.publish(
        RerunAPI(),
        "owner/consumer",
        123,
        "https://owner.github.io/consumer",
        tmp_path / "site",
        tmp_path / "feedback",
        run_attempt=1,
        render=fake_render,
    )
    assert metadata["run_attempt"] == "1"
    assert metadata["report_url"].endswith("/123/1/")
    receipt = json.loads((tmp_path / "site/runs/123/1/receipt.json").read_text())
    assert receipt["artifact_id"] == 456


def test_requested_second_attempt_gets_only_its_own_artifact(tmp_path):
    class SecondAttemptAPI(FakeAPI):
        def get(self, path):
            if path.endswith("/actions/runs/123/attempts/2"):
                return {
                    **run_record(),
                    "run_attempt": 2,
                    "run_started_at": "2026-10-21T13:00:00Z",
                    "updated_at": "2026-10-21T13:02:00Z",
                }
            result = super().get(path)
            if path.endswith("/artifacts?per_page=100"):
                result["artifacts"].append(
                    {
                        **result["artifacts"][0],
                        "id": 789,
                        "name": "skill-evaluator-tier1-attempt-2",
                        "created_at": "2026-10-21T13:01:00Z",
                    }
                )
                result["total_count"] = 2
            return result

        def download_artifact(self, artifact_id):
            assert artifact_id == 789
            return valid_archive()

    metadata = publisher.publish(
        SecondAttemptAPI(),
        "owner/consumer",
        123,
        "https://owner.github.io/consumer",
        tmp_path / "site",
        tmp_path / "feedback",
        run_attempt=2,
        render=fake_render,
    )
    assert metadata["run_attempt"] == "2"


def test_attempt_artifact_name_cannot_claim_another_attempt():
    class WrongNameAPI(FakeAPI):
        def get(self, path):
            result = super().get(path)
            result["artifacts"][0]["name"] = "skill-evaluator-tier1-attempt-2"
            return result

    with pytest.raises(publisher.PublicationError, match="no current report artifact"):
        publisher._select_artifact(WrongNameAPI(), "owner/consumer", run_record())


@pytest.mark.parametrize("repo", [{"full_name": "attacker/fork"}, None])
def test_fork_and_deleted_fork_from_commit_fallback_get_no_alias_or_comment(repo):
    class ForkAPI(FakeAPI):
        def get(self, path):
            result = super().get(path)
            if path.endswith("/pulls/1"):
                result["head"]["repo"] = repo
            return result

    assert publisher._current_routes(ForkAPI(), "owner/consumer", {**run_record(), "event": "pull_request"}) == ([], "")


@pytest.mark.parametrize(
    "blobs,expected",
    [
        ({"a": "1", "d": "2", "e": "1"}, "2"),
        ({"a": "2", "d": "1", "e": "1"}, "2"),
        ({"a": "2", "d": "2", "e": "1"}, "2"),
        ({"a": "2", "d": "3", "e": "1"}, None),
    ],
)
def test_pr_caller_uses_unambiguous_merged_blob_not_pr_head(blobs, expected):
    class MergeAPI(FakeAPI):
        def get(self, path):
            if "/compare/" in path:
                assert path.endswith("d" * 40 + "..." + "a" * 40)
                return {"merge_base_commit": {"sha": "e" * 40}}
            result = super().get(path)
            if "/contents/" in path:
                result["sha"] = blobs[path.rsplit("=", 1)[1][0]] * 40
            return result

    run = {
        **run_record(),
        "event": "pull_request",
        "pull_requests": [{"number": 1, "head": {"sha": "a" * 40}, "base": {"sha": "d" * 40}}],
    }
    if expected is None:
        with pytest.raises(publisher.PublicationError, match="Both branches changed"):
            publisher._executed_caller_blob(MergeAPI(), "owner/consumer", run)
    else:
        assert publisher._executed_caller_blob(MergeAPI(), "owner/consumer", run) == expected * 40


def test_pr_caller_does_not_trust_missing_or_current_mutable_base():
    with pytest.raises(publisher.StalePublication, match="no longer associated"):
        publisher._executed_caller_blob(FakeAPI(), "owner/consumer", {**run_record(), "event": "pull_request"})


def test_open_fork_pr_run_is_published_without_alias_or_comment(tmp_path):
    api = FakeAPI(record=fork_record())
    metadata = publisher.publish(
        api,
        "owner/consumer",
        123,
        "https://owner.github.io/consumer",
        tmp_path / "site",
        tmp_path / "output",
        render=fake_render,
    )
    assert metadata["pr_number"] == ""
    assert metadata["is_current"] == "false"
    assert (tmp_path / "site/runs/123/1/index.html").is_file()
    assert not (tmp_path / "site/pr").exists()
    assert [item["run_id"] for item in json.loads((tmp_path / "site/history.json").read_text())] == [123]
    # The fork PR is found from API run metadata, never from artifact data.
    assert "/repos/owner/consumer/pulls?state=open&head=contributor:feature%2Ffork&per_page=100" in api.paths


def test_history_retention_preserves_current_main_and_open_pr_and_prunes_closed(tmp_path):
    history = []
    for number in range(1, 6):
        receipt = {"run_id": number, "run_attempt": 1, "report_url": f"https://example.org/runs/{number}/1/"}
        history.append(receipt)
        publisher._dump(tmp_path / f"runs/{number}/1/receipt.json", receipt)
    publisher._alias(tmp_path, "main/latest", history[0])
    publisher._alias(tmp_path, "pr/1", history[1])
    publisher._alias(tmp_path, "pr/2", history[2])

    class ClosedAPI(FakeAPI):
        def get(self, path):
            result = super().get(path)
            result["state"] = "closed" if path.endswith("/2") else "open"
            return result

    retained = publisher._prune_history(ClosedAPI(), "owner/consumer", tmp_path, history, history[-1], 1)
    assert [item["run_id"] for item in retained] == [5, 2, 1]
    assert (tmp_path / "runs/2/1/receipt.json").exists()
    assert not (tmp_path / "runs/3/1").exists()
    assert not (tmp_path / "pr/2").exists()


def test_history_byte_budget_prunes_only_unprotected_reports(tmp_path, monkeypatch):
    history = [{"run_id": number, "run_attempt": 1} for number in range(1, 4)]
    for item in history:
        publisher._write(tmp_path / f"runs/{item['run_id']}/1/index.html", "x" * 100)
    monkeypatch.setattr(publisher, "MAX_HISTORY_BYTES", 150)
    assert publisher._prune_history(FakeAPI(), "owner/consumer", tmp_path, history, history[-1], 100) == [history[-1]]
    monkeypatch.setattr(publisher, "MAX_HISTORY_BYTES", 50)
    with pytest.raises(publisher.PublicationError, match="site budget"):
        publisher._prune_history(FakeAPI(), "owner/consumer", tmp_path, [history[-1]], history[-1], 100)


def test_publish_keeps_history_and_updates_only_current_main(tmp_path):
    site, output = tmp_path / "site", tmp_path / "output"
    metadata = publisher.publish(
        FakeAPI(), "owner/consumer", 123, "https://owner.github.io/consumer", site, output, render=fake_render
    )
    assert (site / "runs/123/1/index.html").read_text() == "trusted renderer"
    assert (site / "main/latest/index.html").is_file()
    assert metadata["is_current"] == "true"
    assert metadata["report_url"].endswith("/runs/123/1/")
    assert not list(site.rglob("*.log"))
    assert not (site / "runs/123/1/skill-001/skillevaluator-tier1.html").exists()
    # Re-publication is idempotent and does not rewrite immutable report content.
    publisher.publish(
        FakeAPI(),
        "owner/consumer",
        123,
        "https://owner.github.io/consumer",
        site,
        output,
        render=lambda *_: pytest.fail("immutable report was re-rendered"),
    )
    assert len(json.loads((site / "history.json").read_text())) == 1


def test_old_head_is_historical_without_latest_alias(tmp_path):
    site = tmp_path / "site"
    metadata = publisher.publish(
        FakeAPI(branch_sha="b" * 40),
        "owner/consumer",
        123,
        "https://owner.github.io/consumer",
        site,
        tmp_path / "output",
        render=fake_render,
    )
    assert metadata["is_current"] == "false"
    assert not (site / "main/latest/index.html").exists()


def test_pr_alias_uses_verified_api_head(tmp_path):
    record = {
        **run_record(),
        "event": "pull_request",
        "pull_requests": [{"number": 1, "head": {"sha": "a" * 40}, "base": {"sha": "d" * 40}}],
    }
    metadata = publisher.publish(
        FakeAPI(record=record),
        "owner/consumer",
        123,
        "https://owner.github.io/consumer",
        tmp_path / "site",
        tmp_path / "output",
        render=fake_render,
    )
    assert metadata["pr_number"] == "1"
    assert (tmp_path / "site/pr/1/index.html").is_file()


def test_manual_skill_checks_run_can_update_current_main(tmp_path):
    metadata = publisher.publish(
        FakeAPI(record={**run_record(), "event": "workflow_dispatch"}),
        "owner/consumer",
        123,
        "https://owner.github.io/consumer",
        tmp_path / "site",
        tmp_path / "output",
        render=fake_render,
    )
    assert metadata["is_current"] == "true"
    assert (tmp_path / "site/main/latest/index.html").is_file()


def test_previous_attempt_artifact_is_not_reused(tmp_path):
    class OldArtifactAPI(FakeAPI):
        def get(self, path):
            result = super().get(path)
            if path.endswith("/artifacts?per_page=100"):
                result["artifacts"][0]["created_at"] = "2026-10-20T12:00:00Z"
            return result

    with pytest.raises(publisher.PublicationError):
        publisher.publish(
            OldArtifactAPI(),
            "owner/consumer",
            123,
            "https://owner.github.io/consumer",
            tmp_path / "site",
            tmp_path / "output",
            render=fake_render,
        )


@pytest.mark.parametrize(
    "recorded,fallback,expected",
    [
        (
            "ghcr.io/owner/evaluator@sha256:" + "d" * 64,
            "ghcr.io/owner/evaluator@sha256:" + "e" * 64,
            "ghcr.io/owner/evaluator@sha256:" + "d" * 64,
        ),
        (None, "", ""),
        (None, "ghcr.io/owner/evaluator@sha256:" + "e" * 64, "ghcr.io/owner/evaluator@sha256:" + "e" * 64),
    ],
)
def test_recorded_image_provenance_wins_and_legacy_stays_unknown(tmp_path, recorded, fallback, expected):
    document = summary()
    if recorded is not None:
        document["execution"] = {"image": recorded}

    class ImageAPI(FakeAPI):
        def download_artifact(self, _artifact_id):
            return archive([("summary.json", document), ("skill-001/skillevaluator-tier1.json", report())])

    contexts = []

    def render(directory, payload, context):
        contexts.append(context)
        fake_render(directory, payload, context)

    publisher.publish(
        ImageAPI(),
        "owner/consumer",
        123,
        "https://owner.github.io/consumer",
        tmp_path / "site",
        tmp_path / "output",
        render=render,
        image=fallback,
    )
    assert contexts[0]["image"] == expected
    receipt = json.loads((tmp_path / "site/runs/123/1/receipt.json").read_text())
    history = json.loads((tmp_path / "site/history.json").read_text())
    assert receipt.get("image", "") == expected
    assert history[0].get("image", "") == expected
    if not expected:
        assert "image" not in receipt


@pytest.mark.parametrize(
    "execution",
    [[], {"image": ["invalid"]}, {"image": "untrusted\nimage"}, {"image": "ghcr.io/owner/evaluator:latest"}],
)
def test_invalid_recorded_image_provenance_is_rejected(tmp_path, execution):
    class BadImageAPI(FakeAPI):
        def download_artifact(self, _artifact_id):
            return archive(
                [
                    ("summary.json", {**summary(), "execution": execution}),
                    ("skill-001/skillevaluator-tier1.json", report()),
                ]
            )

    with pytest.raises(publisher.PublicationError):
        publisher.publish(
            BadImageAPI(),
            "owner/consumer",
            123,
            "https://owner.github.io/consumer",
            tmp_path / "site",
            tmp_path / "output",
            render=fake_render,
        )


@pytest.mark.parametrize("variable", ["TRUSTED_EVALUATOR_SHA", "TRUSTED_CALLER_BLOBS"])
def test_publisher_requires_explicit_trust_configuration(tmp_path, monkeypatch, variable):
    monkeypatch.delenv(variable)
    with pytest.raises(publisher.PublicationError):
        publisher.publish(
            FakeAPI(),
            "owner/consumer",
            123,
            "https://owner.github.io/consumer",
            tmp_path / "site",
            tmp_path / "output",
            render=fake_render,
        )


@pytest.mark.parametrize(
    "reference",
    [
        [],
        [{"path": "attacker/repo/.github/workflows/evaluate-skills.yml@" + "c" * 40, "sha": "c" * 40}],
        [{"path": "NVIDIA/SkillEvaluator/.github/workflows/evaluate-skills.yml@main", "sha": "c" * 40}],
        [
            {
                "path": "NVIDIA/SkillEvaluator/.github/workflows/evaluate-skills.yml@" + "c" * 40,
                "sha": "d" * 40,
            }
        ],
    ],
)
def test_untrusted_reusable_workflow_is_rejected(tmp_path, reference):
    with pytest.raises(publisher.PublicationError):
        publisher.publish(
            FakeAPI(record={**run_record(), "referenced_workflows": reference}),
            "owner/consumer",
            123,
            "https://owner.github.io/consumer",
            tmp_path / "site",
            tmp_path / "output",
            render=fake_render,
        )


def test_modified_caller_with_trusted_reference_cannot_publish(tmp_path):
    class ChangedCallerAPI(FakeAPI):
        def get(self, path):
            result = super().get(path)
            if "/contents/" in path:
                result["sha"] = "f" * 40
            return result

    with pytest.raises(publisher.PublicationError):
        publisher.publish(
            ChangedCallerAPI(),
            "owner/consumer",
            123,
            "https://owner.github.io/consumer",
            tmp_path / "site",
            tmp_path / "output",
            render=fake_render,
        )


def test_failed_workflow_cannot_publish_a_passing_artifact(tmp_path):
    with pytest.raises(publisher.PublicationError):
        publisher.publish(
            FakeAPI(record={**run_record(), "conclusion": "failure"}),
            "owner/consumer",
            123,
            "https://owner.github.io/consumer",
            tmp_path / "site",
            tmp_path / "output",
            render=fake_render,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"total_errors": 3},
        {"severity_counts": {"critical": 0, "high": 1, "medium": 0, "low": 0}},
        {"overall_status": "failed"},
        {"incomplete_scans": ["semgrep"]},
    ],
)
def test_report_counts_status_and_completeness_must_agree(change):
    with pytest.raises(publisher.PublicationError):
        publisher.read_artifact(
            archive([("summary.json", summary()), ("skill-001/skillevaluator-tier1.json", {**report(), **change})])
        )


def test_failed_summary_cannot_display_a_passing_skill_report():
    payload = summary()
    payload.update(passed=0, failed=1, overall_passed=False, overall_status="failed")
    payload["skills"][0].update(passed=False, overall_status="failed", exit_code=1)
    with pytest.raises(publisher.PublicationError):
        publisher.read_artifact(archive([("summary.json", payload), ("skill-001/skillevaluator-tier1.json", report())]))


@pytest.mark.parametrize(
    "changes",
    [
        {"path": ".github/workflows/evil.yml"},
        {"status": "in_progress"},
        {"repository": {"full_name": "other/repo"}},
        {"event": "workflow_run"},
    ],
)
def test_unexpected_run_provenance_is_rejected(tmp_path, changes):
    with pytest.raises(publisher.PublicationError):
        publisher.publish(
            FakeAPI(record={**run_record(), **changes}),
            "owner/consumer",
            123,
            "https://owner.github.io/consumer",
            tmp_path / "site",
            tmp_path / "output",
            render=fake_render,
        )


def skipped_summary():
    return {
        **summary(),
        "total": 0,
        "passed": 0,
        "failed": 0,
        "incomplete": 0,
        "overall_passed": False,
        "overall_status": "skipped",
        "skills": [],
        "total_validators": 0,
        "total_errors": 0,
        "total_warnings": 0,
        "selection": {
            "mode": "changed",
            "base": "d" * 40,
            "head": "a" * 40,
            "discovered": 5,
            "selected": [],
            "deleted": [],
            "changed_files": 1,
        },
    }


def test_no_change_artifact_retains_skipped_without_claiming_pass():
    payload, reports = publisher.read_artifact(archive([("summary.json", skipped_summary())]))
    assert payload["overall_status"] == "skipped"
    assert payload["overall_passed"] is False
    assert reports == {}


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "all"},
        {"head": "main"},
        {"selected": ["skills/example"]},
        {"discovered": True},
        {"changed_files": -1},
        {"deleted": ["valid", 1]},
    ],
)
def test_no_change_requires_consistent_selection_evidence(change):
    payload = skipped_summary()
    payload["selection"].update(change)
    with pytest.raises(publisher.PublicationError):
        publisher.read_artifact(archive([("summary.json", payload)]))


@pytest.mark.parametrize(
    "head,conclusion,accepted",
    [
        ("a" * 40, "success", True),
        ("e" * 40, "success", False),
        ("a" * 40, "failure", False),
    ],
)
def test_no_change_publication_matches_run_head_and_success(tmp_path, head, conclusion, accepted):
    class NoChangeAPI(FakeAPI):
        def download_artifact(self, artifact_id):
            assert artifact_id == 456
            payload = skipped_summary()
            payload["selection"]["head"] = head
            return archive([("summary.json", payload)])

    api = NoChangeAPI(record={**run_record(), "conclusion": conclusion})
    args = (api, "owner/consumer", 123, "https://owner.github.io/consumer", tmp_path / "site", tmp_path / "out")
    if accepted:
        publisher.publish(*args, render=fake_render)
        receipt = json.loads((tmp_path / "site/runs/123/1/receipt.json").read_text())
        assert receipt["overall_status"] == "skipped"
    else:
        with pytest.raises(publisher.PublicationError):
            publisher.publish(*args, render=fake_render)


def test_explicit_fork_evaluator_identity_is_required(monkeypatch):
    record = run_record()
    record["referenced_workflows"][0]["path"] = "example/evaluator/.github/workflows/evaluate-skills.yml@" + "c" * 40
    with pytest.raises(publisher.PublicationError):
        publisher._validate_workflow_provenance(FakeAPI(), "owner/consumer", record)
    monkeypatch.setenv("EVALUATOR_REPOSITORY", "example/evaluator")
    publisher._validate_workflow_provenance(FakeAPI(), "owner/consumer", record)


class ForkLookupAPI(FakeAPI):
    def __init__(self, pulls, **kwargs):
        super().__init__(**kwargs)
        self.pulls = pulls

    def get(self, path):
        if "/pulls?state=open&head=" in path:
            self.paths.append(path)
            return self.pulls
        return super().get(path)


@pytest.mark.parametrize(
    "record,pulls,reason",
    [
        pytest.param(
            {**run_record(), "event": "pull_request"},
            None,
            "The PR is closed or no longer associated",
            id="same-repo-pr-merged-after-scan",
        ),
        pytest.param(
            {
                **run_record(),
                "event": "pull_request",
                "pull_requests": [{"base": {"sha": "d" * 40}, "head": {"sha": "e" * 40}}],
            },
            None,
            "The PR head moved",
            id="same-repo-push-during-scan",
        ),
        pytest.param(fork_record(), [], "The fork PR is closed or merged", id="fork-pr-closed-after-scan"),
        pytest.param(fork_record(), [fork_pull("e" * 40)], "The PR head moved", id="fork-push-during-scan"),
        pytest.param(fork_record(head_repository=None), None, "fork for this PR run was deleted", id="deleted-fork"),
    ],
)
def test_stale_or_closed_pr_is_a_skipped_publisher_run(tmp_path, monkeypatch, record, pulls, reason):
    api = ForkLookupAPI(pulls, record=record)
    monkeypatch.setattr(publisher, "GitHubAPI", lambda *_args: api)
    for key, value in {
        "GH_TOKEN": "dummy-token",
        "GITHUB_REPOSITORY": "owner/consumer",
        "REPORT_RUN_ID": "123",
        "REPORT_RUN_ATTEMPT": "1",
        "PAGES_BASE_URL": "https://owner.github.io/consumer",
        "SITE_DIR": str(tmp_path / "site"),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "GITHUB_OUTPUT": str(tmp_path / "github-output"),
    }.items():
        monkeypatch.setenv(key, value)
    assert publisher.main() == 0
    assert "skipped=true" in (tmp_path / "github-output").read_text()
    assert "skipped" in (tmp_path / "output/summary.md").read_text().lower()
    assert reason in (tmp_path / "output/summary.md").read_text()
    assert not (tmp_path / "site/runs").exists()
    # Only a fork run with a known head repository looks up its PR.
    assert any("/pulls?" in path for path in api.paths) == (pulls is not None)


@pytest.mark.parametrize(
    "record,pulls",
    [
        pytest.param({**run_record(), "event": "pull_request", "pull_requests": None}, None, id="same-repo-missing"),
        pytest.param(
            {
                **run_record(),
                "event": "pull_request",
                "pull_requests": [{"base": {"sha": "d" * 40}, "head": {"sha": "a" * 40}}] * 2,
            },
            None,
            id="same-repo-two-prs",
        ),
        pytest.param(
            {**run_record(), "event": "pull_request", "pull_requests": [{"base": {"sha": "d" * 40}, "head": {}}]},
            None,
            id="same-repo-malformed-head",
        ),
        pytest.param(fork_record(pull_requests=None), None, id="fork-missing"),
        pytest.param(fork_record(head_repository="contributor/consumer"), None, id="fork-malformed-head-repository"),
        pytest.param(fork_record(head_branch=None), None, id="fork-missing-branch"),
        pytest.param(fork_record(), {"items": []}, id="fork-malformed-listing"),
        pytest.param(fork_record(), [fork_pull(), fork_pull()], id="fork-two-prs"),
        pytest.param(fork_record(), [fork_pull(base_sha="main")], id="fork-malformed-base"),
    ],
)
def test_malformed_pr_association_still_fails(record, pulls):
    with pytest.raises(publisher.PublicationError) as raised:
        publisher._validate_workflow_provenance(ForkLookupAPI(pulls, record=record), "owner/consumer", record)
    assert not isinstance(raised.value, publisher.StalePublication)


@pytest.mark.parametrize("enabled", [True, False, None])
def test_published_report_preserves_recorded_llm_request(tmp_path, enabled):
    class RecordedAPI(FakeAPI):
        def download_artifact(self, artifact_id):
            assert artifact_id == 456
            payload = summary()
            if enabled is not None:
                payload["execution"] = {"llm_enabled": enabled}
            return archive([("summary.json", payload), ("skill-001/skillevaluator-tier1.json", report())])

    def capture(directory, payload, context):
        assert context["llm_enabled"] is enabled
        fake_render(directory, payload, context)

    publisher.publish(
        RecordedAPI(),
        "owner/consumer",
        123,
        "https://owner.github.io/consumer",
        tmp_path / "site",
        tmp_path / "out",
        render=capture,
    )
