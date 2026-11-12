# SPDX-License-Identifier: Apache-2.0
"""Benchmark publication regenerates evidence and restricts all repository writes."""

import copy
import importlib
import json
import subprocess
import sys
import urllib.error

import pytest
from tests.ci.test_feedback import BLOB, REPO, SHA, WORKFLOW, GitHub


def adapter():
    return importlib.import_module("ci.benchmark_publish")


class Publisher(GitHub):
    def __init__(self):
        super().__init__()
        self.run.update(event="push", head_branch="main", pull_requests=[])
        self.entries = [
            {"path": "skills", "mode": "040000", "type": "tree", "sha": "e" * 40},
            {"path": "skills/demo", "mode": "040000", "type": "tree", "sha": "f" * 40},
            {"path": "skills/demo/SKILL.md", "mode": "100644", "type": "blob", "sha": BLOB},
        ]
        self.branch = None
        self.commit_count = 0
        self.commit_messages = {}
        self.generated_entries = None
        self.pull = None
        self.current_sha = SHA
        self.calls = 0
        self.stale_at = None

    def get(self, path):
        root = "/repos/" + REPO
        if path.startswith(root + "/contents/.github/workflows/"):
            assert path.endswith("?ref=" + SHA)
            return {"type": "file", "path": ".github/workflows/skill-checks-all-tiers.yml", "sha": BLOB}
        if path == root:
            return {"default_branch": "main", "full_name": REPO}
        if path == root + "/git/ref/heads/main":
            self.calls += 1
            return {"object": {"type": "commit", "sha": "9" * 40 if self.stale_at == self.calls else self.current_sha}}
        if path == root + "/git/ref/heads/skillevaluator/benchmark-update":
            if self.branch is None:
                raise urllib.error.HTTPError("", 404, "", {}, None)
            return {"object": {"type": "commit", "sha": self.branch}}
        if path.startswith(root + "/git/commits/"):
            return {
                "tree": {"sha": "1" * 40 if path.endswith(SHA) else "2" * 40},
                "message": self.commit_messages.get(
                    path.rsplit("/", 1)[-1],
                    "Update eligible SkillEvaluator benchmark cards\n\nSource: "
                    + SHA
                    + "\nRun: 123/1\nBenchmark-Cards: "
                    + json.dumps({"skills/demo/BENCHMARK.md": adapter()._blob("generated card\n")}),
                ),
            }
        if path.startswith(root + "/git/trees/"):
            return {
                "truncated": False,
                "tree": copy.deepcopy(
                    self.generated_entries
                    if "/" + "2" * 40 in path and self.generated_entries is not None
                    else self.entries
                ),
            }
        if path == root + "/pulls/8":
            return copy.deepcopy(self.pull)
        if path.startswith(root + "/pulls?"):
            return [copy.deepcopy(self.pull)] if self.pull else []
        return super().get(path)

    def post(self, path, payload):
        self.writes.append(("post", path, payload))
        if path.endswith("/git/trees"):
            self.generated_entries = copy.deepcopy(self.entries)
            changed = {row["path"] for row in payload["tree"]}
            self.generated_entries = [row for row in self.generated_entries if row["path"] not in changed]
            self.generated_entries.extend(
                {"path": row["path"], "mode": row["mode"], "type": row["type"], "sha": adapter()._blob(row["content"])}
                for row in payload["tree"]
            )
            return {"sha": "2" * 40}
        if path.endswith("/git/commits"):
            self.commit_count += 1
            sha = ("3" if self.commit_count == 1 else "8") * 40
            self.commit_messages[sha] = payload["message"]
            return {"sha": sha}
        if path.endswith("/git/refs"):
            self.branch = payload["sha"]
            return {"object": {"sha": self.branch}}
        if path.endswith("/pulls"):
            self.pull = {
                "number": 8,
                "html_url": f"https://github.com/{REPO}/pull/8",
                "state": "open",
                "head": {"ref": "skillevaluator/benchmark-update", "repo": {"full_name": REPO}, "sha": self.branch},
                "base": {"ref": "main", "repo": {"full_name": REPO}},
                **{k: v for k, v in payload.items() if k not in {"head", "base"}},
                "user": {"login": "github-actions[bot]", "type": "Bot"},
            }
            return self.pull
        raise AssertionError(path)

    def patch(self, path, payload):
        self.writes.append(("patch", path, payload))
        if "/git/refs/" in path:
            self.branch = payload["sha"]
            if self.pull:
                self.pull["head"]["sha"] = self.branch
            return {"object": {"sha": self.branch}}
        if "/pulls/" in path:
            self.pull.update(payload)
            return self.pull
        raise AssertionError(path)


@pytest.fixture
def trusted(monkeypatch):
    monkeypatch.setenv("SKILL_EVAL_TRUSTED_WORKFLOW_REF", WORKFLOW)
    monkeypatch.setenv("SKILL_EVAL_TRUSTED_CALLER_BLOBS", BLOB)
    from ci import feedback

    monkeypatch.setattr(feedback, "verify_receipt", lambda *_a, **_kw: {"workflow_sha": SHA})
    return Publisher()


def documents(monkeypatch, *, eligible=True):
    module = adapter()
    monkeypatch.setattr(
        module,
        "benchmark_documents",
        lambda _manifest, _native: [
            {"path": "skills/demo", "eligible": eligible, "benchmark_content": "generated card\n"}
        ],
    )
    return module


def test_only_regenerated_cards_written_and_reviewable_pr(trusted, monkeypatch):
    module = documents(monkeypatch)
    result = module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert result["benchmark_action"] == "created"
    tree = next(payload for _, path, payload in trusted.writes if path.endswith("/git/trees"))
    assert tree["tree"] == [
        {"path": "skills/demo/BENCHMARK.md", "mode": "100644", "type": "blob", "content": "generated card\n"}
    ]
    assert all("/contents/" not in path for _, path, _ in trusted.writes)
    pr = next(payload for _, path, payload in trusted.writes if path.endswith("/pulls"))
    assert pr["base"] == "main" and pr["head"] == module.BRANCH
    assert SHA in pr["body"] and "/actions/runs/123/artifacts/21" in pr["body"]


@pytest.mark.parametrize("change", ["fork", "attempt", "event", "archive"])
def test_rejects_unauthorized_evidence_without_writes(trusted, monkeypatch, change):
    module = documents(monkeypatch)
    if change == "fork":
        trusted.run["head_repository"]["full_name"] = "foreign/repo"
    elif change == "attempt":
        trusted.run["run_attempt"] = 2
    elif change == "event":
        trusted.run["event"] = "pull_request"
    else:
        trusted.payload = b"not zip"
    with pytest.raises(ValueError):
        module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert trusted.writes == []


def test_initial_stale_default_head_is_successful_no_write_skip(trusted, monkeypatch, capsys):
    module = documents(monkeypatch)
    trusted.current_sha = "9" * 40
    monkeypatch.setenv("GITHUB_TOKEN", "publisher-token")
    monkeypatch.setattr(module.publish_reports, "GitHubAPI", lambda *_args: trusted)

    def unexpected_documents(*_args):
        pytest.fail("A stale producer must skip before regenerating cards.")

    monkeypatch.setattr(module, "benchmark_documents", unexpected_documents)
    assert module.main(["--repository", REPO, "--run-id", "123", "--run-attempt", "1"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["benchmark_action"] == "skipped"
    assert output.err == ""
    assert trusted.writes == []


def test_ineligible_noop(trusted, monkeypatch):
    module = documents(monkeypatch, eligible=False)
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "unchanged"
    assert trusted.writes == []


def test_stale_after_read_never_mutates(trusted, monkeypatch):
    module = documents(monkeypatch)
    trusted.stale_at = 2
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "skipped"
    assert trusted.writes == []


@pytest.mark.parametrize("mode,kind", [("120000", "blob"), ("040000", "tree"), ("160000", "commit")])
def test_target_requires_regular_file(trusted, monkeypatch, mode, kind):
    module = documents(monkeypatch)
    trusted.entries.append({"path": "skills/demo/BENCHMARK.md", "mode": mode, "type": kind, "sha": BLOB})
    with pytest.raises(ValueError):
        module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert trusted.writes == []


def native_archive(tmp_path, *, tamper=False, conflict=False):
    import json

    from tests.ci.test_benchmark_reports import retain
    from tests.ci.test_feedback import archive, manifest

    payload = manifest()
    payload["tiers"] = retain(tmp_path)["tiers"]
    extra = {}
    for tier in payload["tiers"]:
        for skill in tier["skills"]:
            relative = skill["report_json"]
            skill["report_html"] = relative.removesuffix(".json") + ".html"
            native = json.loads((tmp_path / relative).read_text())
            if tier["tier"] == 3:
                native["tier3"]["mode"] = "both"
                native["tier3"]["summary"]["mode"] = "both"
                for result in native["results"]:
                    result["tier3"]["mode"] = "both"
                    result["tier3"]["summary"]["mode"] = "both"
            if conflict:
                native["evaluated_source"] = {"repository": "foreign/repo", "commit": SHA}
            if tamper and tier["tier"] == 1:
                native["results"][0]["passed"] = False
            extra[relative] = json.dumps(native)
            extra[skill["report_html"]] = "native html not interpreted"
    extra["benchmarks/skill-001/BENCHMARK.md"] = "MALICIOUS CARD CONTENT"
    extra["benchmarks/skill-001/skillevaluator.json"] = '{"publication":{"eligible":true}}'
    return archive(payload, extra)


def test_real_native_reports_regenerate_and_ignore_artifact_cards(trusted, tmp_path):
    trusted.payload = native_archive(tmp_path)
    module = adapter()
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "created"
    tree = next(payload for _, path, payload in trusted.writes if path.endswith("/git/trees"))
    card = tree["tree"][0]["content"]
    assert "MALICIOUS" not in card
    assert "Source digest:" in card


def test_conflicting_native_source_cannot_publish(trusted, tmp_path):
    trusted.payload = native_archive(tmp_path, conflict=True)
    assert adapter().github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "unchanged"
    assert trusted.writes == []


def test_invalid_native_report_cannot_publish(trusted, tmp_path):
    trusted.payload = native_archive(tmp_path, tamper=True)
    with pytest.raises(ValueError):
        adapter().github_benchmark_publish(trusted, REPO, 123, 1)
    assert trusted.writes == []


def test_read_results_exposes_only_fully_validated_native_reports(tmp_path):
    from ci import feedback

    native = {}
    feedback.read_results(native_archive(tmp_path), native_reports=native)
    assert len(native) == 3
    assert all(path.startswith("tier") for path in native)
    retained = {"unchanged": {}}
    with pytest.raises(ValueError):
        feedback.read_results(native_archive(tmp_path, tamper=True), native_reports=retained)
    assert retained == {"unchanged": {}}


def test_unchanged_cards_do_not_create_branch(trusted, monkeypatch):
    module = documents(monkeypatch)
    trusted.entries.append(
        {"path": "skills/demo/BENCHMARK.md", "mode": "100644", "type": "blob", "sha": module._blob("generated card\n")}
    )
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "unchanged"
    assert trusted.writes == []


def test_rerun_reuses_owned_branch_and_pr(trusted, monkeypatch):
    module = documents(monkeypatch)
    module.github_benchmark_publish(trusted, REPO, 123, 1)
    source_entries = copy.deepcopy(trusted.entries)
    branch_entries = [
        *source_entries,
        {"path": "skills/demo/BENCHMARK.md", "mode": "100644", "type": "blob", "sha": module._blob("generated card\n")},
    ]
    original_get = trusted.get

    def get(path):
        if "/git/trees/" in path:
            return {"truncated": False, "tree": branch_entries if "/" + "2" * 40 in path else source_entries}
        return original_get(path)

    monkeypatch.setattr(trusted, "get", get)
    trusted.writes.clear()
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "unchanged"
    assert trusted.writes == []


def test_existing_branch_unrelated_changes_rejected(trusted, monkeypatch):
    module = documents(monkeypatch)
    trusted.branch = "3" * 40
    original_get = trusted.get

    def get(path):
        result = original_get(path)
        if "/git/trees/" + "2" * 40 in path:
            result["tree"].append({"path": "unrelated.txt", "mode": "100644", "type": "blob", "sha": BLOB})
        return result

    monkeypatch.setattr(trusted, "get", get)
    with pytest.raises(ValueError):
        module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert trusted.writes == []


def test_actions_cannot_create_pr_fixed_manual_fallback(trusted, monkeypatch):
    module = documents(monkeypatch)
    original_post = trusted.post

    def post(path, payload):
        if path.endswith("/pulls"):
            raise urllib.error.HTTPError("", 403, "secret provider text", {}, None)
        return original_post(path, payload)

    monkeypatch.setattr(trusted, "post", post)
    result = module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert result == {"benchmark_action": "branch_ready", "benchmark_reason": module.PR_DISABLED}
    assert "secret" not in str(result)
    assert trusted.branch is not None


def test_manual_run_on_current_default_branch_is_eligible(trusted, monkeypatch):
    module = documents(monkeypatch)
    trusted.run["event"] = "workflow_dispatch"
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "created"


def test_recheck_superseded_attempt_before_writes(trusted, monkeypatch):
    module = documents(monkeypatch)
    original_get = trusted.get

    def get(path):
        result = original_get(path)
        if "/git/trees/" in path:
            trusted.run["run_attempt"] = 2
        return result

    monkeypatch.setattr(trusted, "get", get)
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "skipped"
    assert trusted.writes == []


def test_foreign_existing_pr_is_not_modified(trusted, monkeypatch):
    module = documents(monkeypatch)
    module.github_benchmark_publish(trusted, REPO, 123, 1)
    trusted.pull["user"] = {"login": "human", "type": "User"}
    trusted.writes.clear()
    with pytest.raises(ValueError):
        module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert trusted.writes == []


def test_owned_pr_updated_without_touching_source_pr(trusted, monkeypatch):
    module = documents(monkeypatch)
    module.github_benchmark_publish(trusted, REPO, 123, 1)
    monkeypatch.setattr(
        module,
        "benchmark_documents",
        lambda _manifest, _native: [{"path": "skills/demo", "eligible": True, "benchmark_content": "new card\n"}],
    )
    trusted.pull["body"] = module.MARKER + "\nOld evaluation report"
    trusted.writes.clear()
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "updated"
    updates = [(path, payload) for method, path, payload in trusted.writes if method == "patch"]
    assert updates[-1][0] == "/repos/" + REPO + "/pulls/8"
    assert all("/pulls/7" not in path for _, path, _ in trusted.writes)
    branch = next(payload for method, path, payload in trusted.writes if method == "patch" and "/git/refs/" in path)
    assert branch["force"] is False


def test_existing_pr_update_tolerates_live_repository_and_user_metadata(trusted, monkeypatch):
    module = documents(monkeypatch)
    module.github_benchmark_publish(trusted, REPO, 123, 1)
    trusted.pull["body"] = module.MARKER + "\nOld evaluation report"
    trusted.pull["base"]["repo"].update(
        pushed_at="2026-11-09T20:30:00Z", updated_at="2026-11-09T20:30:00Z", open_issues_count=8
    )
    trusted.pull["user"]["avatar_url"] = "https://avatars.githubusercontent.com/u/old"
    monkeypatch.setattr(
        module,
        "benchmark_documents",
        lambda _manifest, _native: [{"path": "skills/demo", "eligible": True, "benchmark_content": "new card\n"}],
    )
    original_patch = trusted.patch

    def patch(path, payload):
        response = original_patch(path, payload)
        if "/git/refs/" in path:
            # A push updates live repository metadata included in a fresh PR response.
            trusted.pull["base"]["repo"].update(
                pushed_at="2026-11-09T20:34:00Z", updated_at="2026-11-09T20:34:00Z", open_issues_count=9
            )
            trusted.pull["user"]["avatar_url"] = "https://avatars.githubusercontent.com/u/current"
        return response

    monkeypatch.setattr(trusted, "patch", patch)
    trusted.writes.clear()
    result = module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert result["benchmark_action"] == "updated"
    assert SHA in trusted.pull["body"] and "attempt 1" in trusted.pull["body"]
    assert trusted.writes[-1][0:2] == ("patch", "/repos/" + REPO + "/pulls/8")
    branch = next(payload for _, path, payload in trusted.writes if "/git/refs/" in path)
    assert branch["force"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        (("number",), 9),
        (("body",), "Human edit"),
        (("title",), "Human title"),
        (("state",), "closed"),
        (("base", "ref"), "other"),
        (("base", "repo", "full_name"), "foreign/repo"),
        (("user", "login"), "another[bot]"),
        (("user", "type"), "User"),
        (("head", "ref"), "other"),
        (("head", "repo", "full_name"), "foreign/repo"),
        (("head", "sha"), "9" * 40),
    ],
)
def test_changed_pr_before_patch_is_not_overwritten(trusted, monkeypatch, field, value):
    module = documents(monkeypatch)
    module.github_benchmark_publish(trusted, REPO, 123, 1)
    trusted.pull["body"] = module.MARKER + "\nOld evaluation report"
    trusted.writes.clear()
    original_get = trusted.get

    def get(path):
        result = original_get(path)
        if path.endswith("/pulls/8"):
            target = result
            for part in field[:-1]:
                target = target[part]
            target[field[-1]] = value
        return result

    monkeypatch.setattr(trusted, "get", get)
    with pytest.raises(ValueError, match=r"PR (head )?changed"):
        module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert all("/pulls/" not in path for _, path, _ in trusted.writes)


def test_unrelated_repository_filenames_are_preserved_during_publication(trusted, monkeypatch):
    module = documents(monkeypatch)
    unrelated = [
        {"path": "docs", "mode": "040000", "type": "tree", "sha": "7" * 40},
        *[
            {"path": path, "mode": "100644", "type": "blob", "sha": BLOB}
            for path in (
                "docs/c#-notes.md",
                "docs/100%-notes.md",
                "docs/why?.md",
                "docs/colon:notes.md",
                r"docs/back\slash.md",
            )
        ],
    ]
    trusted.entries.extend(copy.deepcopy(unrelated))
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "created"
    monkeypatch.setattr(
        module,
        "benchmark_documents",
        lambda _manifest, _native: [{"path": "skills/demo", "eligible": True, "benchmark_content": "new card\n"}],
    )
    trusted.writes.clear()
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "updated"
    assert all(entry in trusted.generated_entries for entry in unrelated)
    tree = next(payload for _, path, payload in trusted.writes if path.endswith("/git/trees"))
    assert tree == {
        "base_tree": "1" * 40,
        "tree": [{"path": "skills/demo/BENCHMARK.md", "mode": "100644", "type": "blob", "content": "new card\n"}],
    }
    assert all("/heads/main" not in path and "/pulls/7" not in path for _, path, _ in trusted.writes)


@pytest.mark.parametrize("path", [None, "", "x" * 4097, "skills/demo/SKILL.md"])
def test_invalid_or_duplicate_repository_tree_paths_rejected(trusted, monkeypatch, path):
    module = documents(monkeypatch)
    trusted.entries.append({"path": path, "mode": "100644", "type": "blob", "sha": BLOB})
    with pytest.raises(ValueError, match="invalid path"):
        module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert trusted.writes == []


def test_unrelated_tree_entry_still_requires_git_object_identity(trusted, monkeypatch):
    module = documents(monkeypatch)
    trusted.entries.append({"path": "notes.md", "mode": "100644", "type": "blob", "sha": "invalid"})
    with pytest.raises(ValueError, match="object identity"):
        module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert trusted.writes == []


def test_superseded_default_head_skips_without_writes(trusted, monkeypatch):
    # A newer default-branch push during a long run is routine, not a failure.
    module = documents(monkeypatch)
    trusted.current_sha = "9" * 40
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "skipped"
    assert trusted.writes == []


def test_unrelated_repository_paths_do_not_block_publication(trusted, monkeypatch):
    module = documents(monkeypatch)
    trusted.entries[:0] = [
        {"path": "docs", "mode": "040000", "type": "tree", "sha": "c" * 40},
        {"path": "docs/c#-notes 100%.md", "mode": "100644", "type": "blob", "sha": "d" * 40},
    ]
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "created"


def test_owned_pr_update_ignores_live_repository_metadata(trusted, monkeypatch):
    # GitHub returns base.repo as the live repository; the branch push changes pushed_at.
    module = documents(monkeypatch)
    module.github_benchmark_publish(trusted, REPO, 123, 1)
    monkeypatch.setattr(
        module,
        "benchmark_documents",
        lambda _manifest, _native: [{"path": "skills/demo", "eligible": True, "benchmark_content": "new card\n"}],
    )
    trusted.pull["body"] = module.MARKER + "\nOld evaluation report"
    trusted.pull["base"]["repo"]["pushed_at"] = "2026-11-10T00:00:00Z"
    trusted.writes.clear()
    original_get = trusted.get

    def get(path):
        result = original_get(path)
        if path.endswith("/pulls/8"):
            result["base"]["repo"]["pushed_at"] = "2026-11-10T00:05:00Z"
            result["base"]["sha"] = "b" * 40
        return result

    monkeypatch.setattr(trusted, "get", get)
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "updated"
    assert any(method == "patch" and path.endswith("/pulls/8") for method, path, _ in trusted.writes)


def test_changed_pr_body_before_patch_is_not_overwritten(trusted, monkeypatch):
    module = documents(monkeypatch)
    module.github_benchmark_publish(trusted, REPO, 123, 1)
    trusted.pull["body"] = module.MARKER + "\nOld evaluation report"
    trusted.writes.clear()
    original_get = trusted.get

    def get(path):
        result = original_get(path)
        if path.endswith("/pulls/8"):
            result["body"] = "Human edit"
        return result

    monkeypatch.setattr(trusted, "get", get)
    with pytest.raises(ValueError, match="PR changed"):
        module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert all("/pulls/" not in path for _, path, _ in trusted.writes)


def previous_source_branch(trusted, monkeypatch, *, unrelated=False, trailer=None):
    module = documents(monkeypatch)
    old_source = "4" * 40
    trusted.branch = "3" * 40
    trusted.commit_count = 1
    trusted.pull = {
        "number": 8,
        "state": "open",
        "title": "Update SkillEvaluator benchmark cards",
        "body": module.MARKER + "\nSource commit: " + old_source,
        "head": {"ref": module.BRANCH, "repo": {"full_name": REPO}, "sha": trusted.branch},
        "base": {"ref": "main", "repo": {"full_name": REPO}},
        "user": {"login": "github-actions[bot]", "type": "Bot"},
    }
    old_entries = copy.deepcopy(trusted.entries)
    trusted.entries[-1]["sha"] = "6" * 40  # current main advanced its skill source
    old_branch = [
        *copy.deepcopy(old_entries),
        {"path": "skills/demo/BENCHMARK.md", "mode": "100644", "type": "blob", "sha": BLOB},
        {"path": "BENCHMARK.md", "mode": "100644", "type": "blob", "sha": BLOB},
    ]
    if unrelated:
        old_branch[-3]["sha"] = "7" * 40
    original_get = trusted.get

    def get(path):
        if path.endswith("/compare/" + old_source + "..." + SHA):
            return {"status": "ahead", "merge_base_commit": {"sha": old_source}}
        if path.endswith("/git/commits/" + old_source):
            return {"tree": {"sha": "5" * 40}}
        if path.endswith("/git/commits/" + "3" * 40):
            return {
                "tree": {"sha": "2" * 40},
                "message": trailer
                if trailer is not None
                else "Update eligible SkillEvaluator benchmark cards\n\nSource: "
                + old_source
                + "\nRun: 122/1\nBenchmark-Cards: "
                + json.dumps({"skills/demo/BENCHMARK.md": BLOB, "BENCHMARK.md": BLOB}),
                "parents": [{"sha": old_source}],
            }
        if "/git/trees/" + "5" * 40 in path:
            return {"truncated": False, "tree": old_entries}
        if "/git/trees/" + "2" * 40 in path:
            return {"truncated": False, "tree": old_branch}
        return original_get(path)

    monkeypatch.setattr(trusted, "get", get)
    return module


def test_previous_publication_source_advances_without_old_source_overwrite(trusted, monkeypatch):
    module = previous_source_branch(trusted, monkeypatch)
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "updated"
    tree = next(payload for _, path, payload in trusted.writes if path.endswith("/git/trees"))
    assert tree["base_tree"] == "1" * 40
    assert [row["path"] for row in tree["tree"]] == ["skills/demo/BENCHMARK.md"]
    commit = next(payload for _, path, payload in trusted.writes if path.endswith("/git/commits"))
    assert commit["parents"] == ["3" * 40, SHA]
    assert "Source: " + SHA in commit["message"]
    receipt_line = next(line for line in commit["message"].splitlines() if line.startswith("Benchmark-Cards: "))
    assert json.loads(receipt_line.removeprefix("Benchmark-Cards: ")) == {
        "skills/demo/BENCHMARK.md": module._blob("generated card\n")
    }
    update = next(payload for _, path, payload in trusted.writes if "/git/refs/" in path)
    assert update["force"] is False


def test_previous_source_branch_unrelated_code_edit_rejected(trusted, monkeypatch):
    module = previous_source_branch(trusted, monkeypatch, unrelated=True)
    with pytest.raises(ValueError):
        module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert trusted.writes == []


@pytest.mark.parametrize(
    "trailer",
    [
        "Unknown commit",
        "Update eligible SkillEvaluator benchmark cards\n\nSource: invalid",
        "Update eligible SkillEvaluator benchmark cards\n\nSource: " + SHA + "\nSource: " + SHA,
    ],
)
def test_previous_source_missing_or_ambiguous_trailer_rejected(trusted, monkeypatch, trailer):
    module = previous_source_branch(trusted, monkeypatch, trailer=trailer)
    with pytest.raises(ValueError):
        module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert trusted.writes == []


def test_source_advance_during_prior_branch_read_skips(trusted, monkeypatch):
    module = previous_source_branch(trusted, monkeypatch)
    trusted.stale_at = 2
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "skipped"
    assert trusted.writes == []


def test_recorded_source_must_be_current_default_history(trusted, monkeypatch):
    module = previous_source_branch(trusted, monkeypatch)
    original_get = trusted.get

    def get(path):
        if "/compare/" in path:
            return {"status": "diverged", "merge_base_commit": {"sha": "8" * 40}}
        return original_get(path)

    monkeypatch.setattr(trusted, "get", get)
    with pytest.raises(ValueError, match="default-branch history"):
        module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert trusted.writes == []


def test_existing_pr_tracks_new_head_sha_and_receives_current_body(trusted, monkeypatch):
    module = previous_source_branch(trusted, monkeypatch)
    old_head = trusted.pull["head"]["sha"]
    assert module.github_benchmark_publish(trusted, REPO, 123, 1)["benchmark_action"] == "updated"
    assert trusted.pull["head"]["sha"] != old_head
    assert SHA in trusted.pull["body"] and "attempt 1" in trusted.pull["body"]


@pytest.mark.parametrize("manual_change", ["new_card", "modified_owned_card"])
def test_unproven_card_edits_are_not_discarded(trusted, monkeypatch, manual_change):
    module = previous_source_branch(trusted, monkeypatch)
    original_get = trusted.get

    def get(path):
        result = copy.deepcopy(original_get(path))
        if "/git/trees/" + "2" * 40 in path:
            if manual_change == "new_card":
                result["tree"].append(
                    {"path": "skills/BENCHMARK.md", "mode": "100644", "type": "blob", "sha": "7" * 40}
                )
            else:
                next(row for row in result["tree"] if row["path"] == "skills/demo/BENCHMARK.md")["sha"] = "7" * 40
        return result

    monkeypatch.setattr(trusted, "get", get)
    with pytest.raises(ValueError):
        module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert trusted.writes == []


def test_package_import_identity_after_bootstrap_import():
    code = "import sys; from pathlib import Path; from ci import bootstrap_tiers; sys.path.insert(0, str(Path(bootstrap_tiers.__file__).parent)); from ci import benchmark_publish, feedback; assert benchmark_publish.feedback is feedback"
    completed = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("receipt", [None, "duplicate", "invalid-json", "unsafe-path"])
def test_prior_card_receipt_required_and_unambiguous(trusted, monkeypatch, receipt):
    module = previous_source_branch(trusted, monkeypatch)
    original_get = trusted.get

    def get(path):
        result = original_get(path)
        if path.endswith("/git/commits/" + "3" * 40):
            message = result["message"]
            original = next(line for line in message.splitlines() if line.startswith("Benchmark-Cards:"))
            if receipt is None:
                result["message"] = message.replace("\n" + original, "")
            elif receipt == "duplicate":
                result["message"] += "\n" + original
            elif receipt == "invalid-json":
                result["message"] = message.replace(original, "Benchmark-Cards: {invalid}")
            else:
                result["message"] = message.replace(original, 'Benchmark-Cards: {"../BENCHMARK.md":"' + BLOB + '"}')
        return result

    monkeypatch.setattr(trusted, "get", get)
    with pytest.raises(ValueError):
        module.github_benchmark_publish(trusted, REPO, 123, 1)
    assert trusted.writes == []
