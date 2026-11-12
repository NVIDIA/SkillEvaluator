# SPDX-License-Identifier: Apache-2.0
"""Execute the actual credential-admission workflow scripts against GitHub data."""

import io
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SHA = "a" * 40


def run_admission(
    tmp_path,
    monkeypatch,
    stage,
    *,
    permission="write",
    event_name="pull_request",
    head=SHA,
    fork=False,
    actor="alice",
    ref="refs/heads/main",
):
    workflow = yaml.load((ROOT / ".github/workflows/evaluate-all-tiers.yml").read_text(), Loader=yaml.BaseLoader)
    if stage == "eligibility":
        shell = workflow["jobs"]["eligibility"]["steps"][0]["run"]
    else:
        shell = next(s["run"] for s in workflow["jobs"]["paid-tiers"]["steps"] if s.get("id") == "current-head")
    event = {"repository": {"default_branch": "main"}}
    current = {"state": "open", "head": {"sha": head, "repo": {"full_name": "acme/skills"}}}
    if event_name == "pull_request":
        event["pull_request"] = {
            "number": 7,
            "head": {"sha": SHA, "repo": {"full_name": "outsider/fork" if fork else "acme/skills"}},
        }
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event))
    output = tmp_path / "outputs"
    environment = {
        "TOOLS_REF": SHA,
        "ENV_MODE": "docker",
        "TIERS": "1,2,3",
        "SCAN_MODE": "auto",
        "GH_TOKEN": "read-only-token",
        "API_URL": "https://api.github.example",
        "GITHUB_REPOSITORY": "acme/skills",
        "GITHUB_ACTOR": actor,
        "GITHUB_TRIGGERING_ACTOR": "rerun-user",
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(output),
        "GITHUB_REF": ref,
        "GITHUB_SHA": SHA,
        "GITHUB_EVENT_NAME": event_name,
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    requested = []

    def respond(request, **_kwargs):
        url = request.full_url
        requested.append(url)
        if url.endswith("/collaborators/" + urllib.parse.quote(actor, safe="") + "/permission"):
            if permission is None:
                raise urllib.error.HTTPError(url, 404, "No access", {}, None)
            payload = {"permission": permission}
        elif url.endswith("/pulls/7"):
            payload = current
        elif "/pulls/7/files?" in url:
            payload = []
        elif url.endswith("/repos/acme/skills"):
            payload = {"default_branch": "main"}
        elif url.endswith("/branches/main"):
            payload = {"commit": {"sha": head}}
        else:
            raise AssertionError(f"Unexpected API request {url}")
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    try:
        exec(compile("\n".join(shell.splitlines()[1:-1]), "<workflow admission>", "exec"), {})
    except SystemExit as error:
        if error.code not in (None, 0):
            return False, requested
    allowed = output.exists() and "allowed=true\n" in output.read_text()
    return allowed, requested


@pytest.mark.parametrize("stage", ["eligibility", "recheck"])
@pytest.mark.parametrize("permission", ["read", "triage", "none", None])
@pytest.mark.parametrize("event_name", ["pull_request", "push", "workflow_dispatch"])
def test_provider_jobs_require_proven_write_permission(tmp_path, monkeypatch, stage, permission, event_name):
    allowed, requested = run_admission(tmp_path, monkeypatch, stage, permission=permission, event_name=event_name)
    assert not allowed
    assert any("/collaborators/alice/permission" in url for url in requested)


@pytest.mark.parametrize("stage", ["eligibility", "recheck"])
@pytest.mark.parametrize("permission", ["write", "maintain", "admin"])
def test_initiating_writer_can_run_current_same_repo_head(tmp_path, monkeypatch, stage, permission):
    allowed, requested = run_admission(tmp_path, monkeypatch, stage, permission=permission)
    assert allowed
    assert any("/collaborators/alice/permission" in url for url in requested)
    assert not any("rerun-user" in url for url in requested)


@pytest.mark.parametrize("stage", ["eligibility", "recheck"])
@pytest.mark.parametrize("event_name", ["push", "workflow_dispatch"])
def test_default_branch_execution_requires_current_commit(tmp_path, monkeypatch, stage, event_name):
    allowed, _ = run_admission(tmp_path, monkeypatch, stage, event_name=event_name, head="b" * 40)
    assert not allowed


@pytest.mark.parametrize("stage", ["eligibility", "recheck"])
@pytest.mark.parametrize("event_name", ["push", "workflow_dispatch"])
def test_current_default_branch_writer_is_eligible(tmp_path, monkeypatch, stage, event_name):
    allowed, requested = run_admission(tmp_path, monkeypatch, stage, event_name=event_name)
    assert allowed
    assert any(url.endswith("/branches/main") for url in requested)


@pytest.mark.parametrize("stage", ["eligibility", "recheck"])
def test_a_fork_head_cannot_use_writer_permissions(tmp_path, monkeypatch, stage):
    allowed, _ = run_admission(tmp_path, monkeypatch, stage, fork=True)
    assert not allowed


@pytest.mark.parametrize("stage", ["eligibility", "recheck"])
def test_nondefault_branch_is_not_authorized_for_manual_model_work(tmp_path, monkeypatch, stage):
    allowed, _ = run_admission(tmp_path, monkeypatch, stage, event_name="workflow_dispatch", ref="refs/heads/feature")
    assert not allowed


@pytest.mark.parametrize("stage", ["eligibility", "recheck"])
def test_app_bot_needs_the_same_api_proven_write_role(tmp_path, monkeypatch, stage):
    allowed, requested = run_admission(tmp_path, monkeypatch, stage, actor="my-app[bot]")
    assert allowed
    assert any("/collaborators/my-app%5Bbot%5D/permission" in url for url in requested)


def _eligibility_outputs(tmp_path, monkeypatch, files, *, scan_mode="auto", paths="skills"):
    """Run the actual eligibility script for an eligible same-repository PR."""
    workflow = yaml.load((ROOT / ".github/workflows/evaluate-all-tiers.yml").read_text(), Loader=yaml.BaseLoader)
    shell = workflow["jobs"]["eligibility"]["steps"][0]["run"]
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"pull_request": {"number": 7, "head": {"sha": SHA, "repo": {"full_name": "acme/skills"}}}})
    )
    output = tmp_path / "outputs"
    for key, value in {
        "TOOLS_REF": SHA,
        "ENV_MODE": "docker",
        "TIERS": "1,2,3",
        "SCAN_MODE": scan_mode,
        "SKILL_PATHS": paths,
        "GH_TOKEN": "read-only-token",
        "API_URL": "https://api.github.example",
        "GITHUB_REPOSITORY": "acme/skills",
        "GITHUB_ACTOR": "alice",
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(output),
        "GITHUB_REF": "refs/pull/7/merge",
        "GITHUB_SHA": SHA,
        "GITHUB_EVENT_NAME": "pull_request",
    }.items():
        monkeypatch.setenv(key, value)
    pages = [files[index : index + 100] for index in range(0, max(len(files), 1), 100)] or [[]]

    def respond(request, **_kwargs):
        url = request.full_url
        if url.endswith("/permission"):
            payload = {"permission": "write"}
        elif url.endswith("/pulls/7"):
            payload = {"state": "open", "head": {"sha": SHA, "repo": {"full_name": "acme/skills"}}}
        elif "/pulls/7/files?" in url:
            page = int(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["page"][0])
            payload = pages[page - 1] if page <= len(pages) else []
        else:
            raise AssertionError(f"Unexpected API request {url}")
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    exec(compile("\n".join(shell.splitlines()[1:-1]), "<workflow eligibility>", "exec"), {})
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


@pytest.mark.parametrize(
    ("changed", "skills_changed", "scan_mode"),
    [
        ([{"filename": "skills/alpha/SKILL.md"}], "true", "auto"),
        ([{"filename": "docs/readme.md"}], "false", "auto"),
        ([{"filename": ".github/workflows/skill-checks-all-tiers.yml"}], "false", "all"),
        ([{"filename": ".github/workflows/ci.yml"}, {"filename": "skills/alpha/run.py"}], "true", "all"),
        ([{"filename": "docs/alpha.md", "previous_filename": "skills/alpha/notes.md"}], "true", "auto"),
        ([{"filename": "skills-archive/alpha/SKILL.md"}], "false", "auto"),
    ],
)
def test_eligibility_reports_whether_a_pr_changes_a_skill(tmp_path, monkeypatch, changed, skills_changed, scan_mode):
    """Setup-only and unrelated PRs skip the paid LLM Tier 1 pass; the keyless scan stays."""
    outputs = _eligibility_outputs(tmp_path, monkeypatch, changed)
    assert outputs["allowed"] == "true"
    assert outputs["skills-changed"] == skills_changed
    assert outputs["scan-mode"] == scan_mode


def test_eligibility_skill_predicate_matches_the_shared_helper(tmp_path, monkeypatch):
    from ci.select_changed_skills import skill_root_changed

    roots = "skills\nplugins/tools/"
    cases = [
        ["skills/a/SKILL.md"],
        ["plugins/tools/x/SKILL.md"],
        ["plugins/toolsmith/SKILL.md"],
        ["plugins/README.md"],
        ["skills"],
        ["README.md", ".gitlab-ci.yml"],
    ]
    normalized = ["skills", "plugins/tools"]
    for index, case in enumerate(cases):
        directory = tmp_path / str(index)
        directory.mkdir()
        outputs = _eligibility_outputs(directory, monkeypatch, [{"filename": path} for path in case], paths=roots)
        assert (outputs["skills-changed"] == "true") is skill_root_changed(case, normalized), case


def test_explicit_full_scans_and_long_listings_keep_the_enriched_pass(tmp_path, monkeypatch):
    outputs = _eligibility_outputs(tmp_path, monkeypatch, [{"filename": "docs/readme.md"}], scan_mode="all")
    assert outputs["skills-changed"] == "true"
    many = [{"filename": f"docs/{index}.md"} for index in range(3000)]
    long_listing = tmp_path / "long"
    long_listing.mkdir()
    outputs = _eligibility_outputs(long_listing, monkeypatch, many)
    assert outputs["skills-changed"] == "true"  # A truncated listing cannot prove no skill changed.
    assert outputs["scan-mode"] == "all"


def test_enriched_job_and_gate_artifact_share_the_skill_change_condition():
    workflow = yaml.load((ROOT / ".github/workflows/evaluate-all-tiers.yml").read_text(), Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    condition = "needs.eligibility.outputs.skills-changed == 'true'"
    assert jobs["eligibility"]["outputs"]["skills-changed"] == "${{ steps.check.outputs.skills-changed }}"
    assert condition in jobs["tier1-llm"]["if"]
    assert condition in jobs["tier1"]["with"]["artifact-name"]
    assert jobs["eligibility"]["steps"][0]["env"]["SKILL_PATHS"] == "${{ inputs.paths }}"


def test_paid_tiers_share_the_skill_change_condition_and_aggregate_reason():
    workflow = yaml.load((ROOT / ".github/workflows/evaluate-all-tiers.yml").read_text(), Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    condition = "needs.eligibility.outputs.skills-changed == 'true'"
    assert condition in jobs["paid-tiers"]["if"]
    assert "needs.eligibility.outputs.allowed == 'true'" in jobs["paid-tiers"]["if"]
    assert "eligibility" in jobs["aggregate"]["needs"]
    assert jobs["aggregate"]["env"]["SKILL_EVAL_SKILLS_CHANGED"] == "${{ needs.eligibility.outputs.skills-changed }}"


def test_fork_prs_never_claim_that_no_skill_changed(tmp_path, monkeypatch):
    """Fork file lists are not read, so their skipped paid tiers keep the withheld-job reason."""
    workflow = yaml.load((ROOT / ".github/workflows/evaluate-all-tiers.yml").read_text(), Loader=yaml.BaseLoader)
    shell = workflow["jobs"]["eligibility"]["steps"][0]["run"]
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"pull_request": {"number": 7, "head": {"sha": SHA, "repo": {"full_name": "outsider/fork"}}}})
    )
    output = tmp_path / "outputs"
    for key, value in {
        "TOOLS_REF": SHA,
        "ENV_MODE": "docker",
        "TIERS": "1,2,3",
        "SCAN_MODE": "auto",
        "SKILL_PATHS": "skills",
        "GH_TOKEN": "read-only-token",
        "API_URL": "https://api.github.example",
        "GITHUB_REPOSITORY": "acme/skills",
        "GITHUB_ACTOR": "alice",
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(output),
        "GITHUB_EVENT_NAME": "pull_request",
    }.items():
        monkeypatch.setenv(key, value)

    def respond(request, **_kwargs):
        if request.full_url.endswith("/permission"):
            return io.BytesIO(json.dumps({"permission": "write"}).encode())
        raise AssertionError(f"Fork PR eligibility must not list files: {request.full_url}")

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    exec(compile("\n".join(shell.splitlines()[1:-1]), "<workflow eligibility>", "exec"), {})
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs == {"allowed": "false", "scan-mode": "auto", "skills-changed": "true"}
