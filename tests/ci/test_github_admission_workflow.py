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
