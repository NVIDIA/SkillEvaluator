# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Privacy, ownership, and URL boundaries for GitHub Pages onboarding."""

import pytest

from skillevaluator import _ci_onboarding_pages as pages_helpers
from skillevaluator import ci_onboarding as wizard
from skillevaluator._ci_onboarding_git import SetupError

NAME = "example-org/private-skills"
PRIVATE_URL = "https://quiet-forest-123.pages.github.io/"


@pytest.fixture
def info():
    return {"private": True, "visibility": "private", "owner": {"type": "Organization"}, "permissions": {"admin": True}}


@pytest.fixture
def pages():
    return {"public": False, "build_type": "workflow", "cname": None, "html_url": PRIVATE_URL, "status": None}


@pytest.mark.parametrize("visibility", ["private", "internal"])
def test_private_and_internal_repositories_reuse_verified_site(info, pages, visibility):
    info["visibility"] = visibility
    private_url = pages_helpers.private_pages_url(NAME, info, pages)
    assert private_url == PRIVATE_URL
    assert pages_helpers.report_url(NAME, verified_private_url=private_url) == PRIVATE_URL
    assert pages_helpers.report_url(NAME, PRIVATE_URL.rstrip("/"), verified_private_url=private_url) == PRIVATE_URL


@pytest.mark.parametrize("value", [None, True, 0, "false"])
def test_repository_privacy_must_be_explicit_boolean_false(value):
    assert pages_helpers.is_nonpublic({"private": value, "visibility": "public"})


def test_public_repository_is_recognized_and_default_url_is_preserved():
    assert not pages_helpers.is_nonpublic({"private": False, "visibility": "public"})
    assert not pages_helpers.is_nonpublic({"private": False})
    assert pages_helpers.is_nonpublic({"private": False, "visibility": "internal"})
    assert pages_helpers.report_url("Example-Org/Skills") == "https://example-org.github.io/Skills/"
    assert pages_helpers.report_url("Example-Org/Skills", "https://example-org.github.io/Skills") == (
        "https://example-org.github.io/Skills/"
    )


@pytest.mark.parametrize("value", [None, True, 0, "false"])
def test_private_site_requires_exact_false_visibility(info, pages, value):
    pages["public"] = value
    with pytest.raises(SetupError, match="select Private visibility and GitHub Actions"):
        pages_helpers.private_pages_url(NAME, info, pages)


def test_missing_site_reports_plan_and_policy_prerequisites(info):
    with pytest.raises(SetupError) as error:
        pages_helpers.private_pages_url(NAME, info, None)
    message = str(error.value)
    assert "GitHub Enterprise Cloud" in message
    assert "organization policy" in message
    assert "--checks-only" in message
    assert f"https://github.com/{NAME}/settings/pages" in message
    assert "A public site will not be created" in message


@pytest.mark.parametrize("owner", [{"type": "User"}, {}, None])
def test_personal_or_unknown_owner_is_not_eligible(info, pages, owner):
    info["owner"] = owner
    with pytest.raises(SetupError, match="Personal repositories cannot publish private Pages"):
        pages_helpers.private_pages_url(NAME, info, pages)


def test_organization_site_cannot_use_private_access_control(info, pages):
    with pytest.raises(SetupError, match="unavailable for an organization site"):
        pages_helpers.private_pages_url("example-org/Example-Org.github.io", info, pages)


@pytest.mark.parametrize("permissions", [{}, {"admin": False}, {"admin": 1}, None])
def test_private_setup_requires_verified_admin_permission(info, pages, permissions):
    info["permissions"] = permissions
    with pytest.raises(SetupError, match="repository admin access"):
        pages_helpers.private_pages_url(NAME, info, pages)


@pytest.mark.parametrize("private,visibility", [(False, "internal"), (True, "public"), (1, "private"), (True, None)])
def test_inconsistent_repository_visibility_fails_closed(info, pages, private, visibility):
    info.update(private=private, visibility=visibility)
    with pytest.raises(SetupError, match="did not confirm this repository"):
        pages_helpers.private_pages_url(NAME, info, pages)


@pytest.mark.parametrize(
    "url",
    [
        "https://quiet-forest-123.pages.github.io.attacker.example/",
        "https://pages.github.io/",
        "https://.pages.github.io/",
        "https://-bad.pages.github.io/",
        "https://bad-.pages.github.io/",
        "https://bad_name.pages.github.io/",
        "https://example-org.github.io/private-skills/",
        "http://quiet-forest-123.pages.github.io/",
        "https://token@quiet-forest-123.pages.github.io/",
        "https://quiet-forest-123.pages.github.io:443/",
        "https://quiet-forest-123.pages.github.io:/",
        "https://quiet-forest-123.pages.github.io:bad/",
        "https://quiet-forest-123.pages.github.io/?token=secret",
        "https://quiet-forest-123.pages.github.io/?",
        "https://quiet-forest-123.pages.github.io/#",
        "https://quiet-forest-123.pages.github.io/subpath/",
        "https://quiet-forest-123.pages.github.io//",
        "https://quiet-forest-123.pages.github.io/../",
        "https://quiet-forest-123.pages.github.io/\\host",
        "https://quiet-forest-123.pages.github.io/\n",
        " https://quiet-forest-123.pages.github.io/",
        "https://qu\u0131et.pages.github.io/",
        "https://[bad/",
        "",
        None,
    ],
)
def test_private_url_rejects_ambiguous_or_unsupported_origins(info, pages, url):
    pages["html_url"] = url
    with pytest.raises(SetupError, match="report URL"):
        pages_helpers.private_pages_url(NAME, info, pages)


@pytest.mark.parametrize("url", [PRIVATE_URL, "https://elsewhere.example/"])
def test_unverified_custom_url_cannot_bind_a_repository_to_a_private_site(url):
    with pytest.raises(SetupError, match="must match this repository"):
        pages_helpers.report_url(NAME, url)


def test_private_custom_url_must_match_the_exact_verified_site():
    with pytest.raises(SetupError, match="must match this repository"):
        pages_helpers.report_url(NAME, "https://another-site.pages.github.io/", verified_private_url=PRIVATE_URL)


@pytest.mark.parametrize("url", ["", "https://example-org.github.io/private-skills//"])
def test_public_url_rejects_empty_or_repeated_trailing_slashes(url):
    with pytest.raises(SetupError):
        pages_helpers.report_url(NAME, url)


def test_fresh_private_workflow_site_can_be_adopted_after_no_history_or_deployments(pages):
    pages_helpers.check_site(NAME, pages, history_exists=False, deployments=[])


@pytest.mark.parametrize("deployments", [None, {}, {"deployments": []}, [{"id": 1}], ""])
def test_unowned_site_requires_verified_empty_deployments(pages, deployments):
    with pytest.raises(SetupError, match="not verified as a fresh empty private site"):
        pages_helpers.check_site(NAME, pages, history_exists=False, deployments=deployments)


@pytest.mark.parametrize("status", ["built", "building", "errored", "", False])
def test_unowned_site_with_any_publication_status_is_not_adopted(pages, status):
    pages["status"] = status
    with pytest.raises(SetupError, match="not verified as a fresh empty private site"):
        pages_helpers.check_site(NAME, pages, history_exists=False, deployments=[])


def test_missing_status_is_not_evidence_that_site_is_fresh(pages):
    del pages["status"]
    with pytest.raises(SetupError, match="not verified as a fresh empty private site"):
        pages_helpers.check_site(NAME, pages, history_exists=False, deployments=[])


@pytest.mark.parametrize("public", [True, None, "false", 0])
def test_only_private_sites_can_be_adopted(pages, public):
    pages["public"] = public
    with pytest.raises(SetupError, match="not verified as a fresh empty private site"):
        pages_helpers.check_site(NAME, pages, history_exists=False, deployments=[])


def test_unowned_legacy_site_cannot_be_adopted(pages):
    pages["build_type"] = "legacy"
    with pytest.raises(SetupError, match="not verified as a fresh empty private site"):
        pages_helpers.check_site(NAME, pages, history_exists=False, deployments=[])


@pytest.mark.parametrize("build_type", ["workflow", "legacy"])
def test_owned_site_can_reuse_history_and_previous_deployments(pages, build_type):
    pages.update(build_type=build_type, status="built")
    pages_helpers.check_site(NAME, pages, history_exists=True, deployments=[{"id": 1}])


@pytest.mark.parametrize("history_exists", [True, False])
@pytest.mark.parametrize("changes", [{"cname": "docs.example.com"}, {"build_type": "unknown"}])
def test_owned_or_empty_site_cannot_override_unsupported_configuration(pages, history_exists, changes):
    pages.update(changes)
    with pytest.raises(SetupError, match="outside SkillEvaluator"):
        pages_helpers.check_site(NAME, pages, history_exists=history_exists, deployments=[])


def test_missing_public_site_needs_no_existing_site_adoption():
    pages_helpers.check_site(NAME, None, history_exists=False)


def _remote_preflight(monkeypatch, tmp_path, info, pages, *, history=False, deployments=None, full=True):
    calls = []
    info["default_branch"] = "main"

    def api(method, endpoint, payload=None, **kwargs):
        calls.append((method, endpoint, payload))
        assert method == "GET", "Remote preflight must never mutate settings"
        if endpoint == f"repos/{NAME}":
            return info
        if endpoint == f"repos/{NAME}/pages":
            return pages
        if endpoint == f"repos/{NAME}/actions/permissions":
            return {"enabled": True, "allowed_actions": "all"}
        if endpoint == f"repos/{NAME}/deployments?per_page=1":
            return deployments
        raise AssertionError(endpoint)

    monkeypatch.setattr(wizard, "github_api", api)
    monkeypatch.setattr(wizard.git_helpers, "require_clean", lambda *_: None)
    monkeypatch.setattr(wizard.git_helpers, "git", lambda *_args, **_kwargs: "same-sha")
    monkeypatch.setattr(wizard, "history_preflight", lambda *_: history)
    return wizard.remote_preflight(tmp_path, NAME, "main", full), calls


@pytest.mark.parametrize("visibility", ["private", "internal"])
def test_preflight_binds_report_url_to_verified_private_site(monkeypatch, tmp_path, info, pages, visibility):
    info["visibility"] = visibility
    state, calls = _remote_preflight(monkeypatch, tmp_path, info, pages, deployments=[])
    assert state["report_url"] == PRIVATE_URL
    assert state["verified_private_url"] == PRIVATE_URL
    assert not state["history_exists"]
    assert any("/deployments?" in endpoint for _, endpoint, _ in calls)


def test_owned_private_site_does_not_require_empty_deployments(monkeypatch, tmp_path, info, pages):
    pages["status"] = "built"
    state, calls = _remote_preflight(monkeypatch, tmp_path, info, pages, history=True)
    assert state["history_exists"]
    assert all("/deployments?" not in endpoint for _, endpoint, _ in calls)


def test_private_adoption_rejects_deployments_from_another_environment(monkeypatch, tmp_path, info, pages):
    with pytest.raises(SetupError, match="not verified as a fresh empty private site"):
        _remote_preflight(monkeypatch, tmp_path, info, pages, deployments=[{"id": 1, "environment": "documentation"}])


def test_private_checks_only_never_reads_pages(monkeypatch, tmp_path, info, pages):
    state, calls = _remote_preflight(monkeypatch, tmp_path, info, pages, full=False)
    assert state["verified_private_url"] is None
    assert state["report_url"] is None
    assert all(not endpoint.endswith("/pages") for _, endpoint, _ in calls)


@pytest.mark.parametrize("visibility", [True, None, "false"])
def test_preflight_rejects_unverified_visibility_before_git_fetch(monkeypatch, tmp_path, info, pages, visibility):
    pages["public"] = visibility
    info["default_branch"] = "main"
    monkeypatch.setattr(wizard.git_helpers, "require_clean", lambda *_: None)

    def api(method, endpoint, payload=None, **kwargs):
        assert method == "GET"
        if endpoint == f"repos/{NAME}":
            return info
        if endpoint == f"repos/{NAME}/pages":
            return pages
        raise AssertionError("Private visibility must be verified before other work")

    def unexpected_git(*_args, **_kwargs):
        raise AssertionError("Private visibility must be verified before fetching")

    monkeypatch.setattr(wizard, "github_api", api)
    monkeypatch.setattr(wizard.git_helpers, "git", unexpected_git)
    with pytest.raises(SetupError, match="has not verified a private Pages site"):
        wizard.remote_preflight(tmp_path, NAME, "main", True)


def _enable_private(monkeypatch, tmp_path, info, pages, *, history=False, readback=None, deployments=None):
    calls = []
    events = []
    state = {
        "info": info,
        "pages": dict(pages),
        "verified_private_url": PRIVATE_URL,
        "report_url": PRIVATE_URL,
        "history_exists": history,
        "actions": {"enabled": True},
    }
    provider = dict(pages)

    def api(method, endpoint, payload=None, **kwargs):
        calls.append((method, endpoint, payload))
        assert method != "POST", "An existing private site must never be recreated"
        if method == "PUT" and endpoint.endswith("/pages"):
            assert history, "Only owned legacy history can switch the publishing source"
            assert payload == {"build_type": "workflow"}
            provider["build_type"] = "workflow"
            return {}
        assert method == "GET"
        if endpoint == f"repos/{NAME}/pages":
            return dict(provider if readback is None else readback)
        if endpoint == f"repos/{NAME}/deployments?per_page=1":
            return [] if deployments is None else deployments
        raise AssertionError(endpoint)

    monkeypatch.setattr(wizard, "github_api", api)
    monkeypatch.setattr(wizard, "initialize_history", lambda *_: events.append("history"))
    monkeypatch.setattr(wizard, "ensure_environment", lambda *_: events.append("environment"))
    progress = []
    wizard.enable_hosting(tmp_path, NAME, "main", state, progress)
    return calls, events, progress


def test_existing_private_site_is_read_back_without_recreation(monkeypatch, tmp_path, info, pages):
    calls, events, progress = _enable_private(monkeypatch, tmp_path, info, pages)
    assert all(method == "GET" for method, _, _ in calls)
    assert events == ["history", "environment"]
    assert "Report history branch initialized" in progress


def test_owned_legacy_private_site_keeps_history_when_switching_source(monkeypatch, tmp_path, info, pages):
    pages.update(build_type="legacy", status="built")
    calls, events, _ = _enable_private(monkeypatch, tmp_path, info, pages, history=True)
    assert [(method, payload) for method, _, payload in calls if method == "PUT"] == [
        ("PUT", {"build_type": "workflow"})
    ]
    assert events == ["environment"]


@pytest.mark.parametrize("changes", [{"public": True}, {"html_url": "https://different.pages.github.io/"}])
def test_changed_private_site_stops_before_history_or_environment(monkeypatch, tmp_path, info, pages, changes):
    with pytest.raises(SetupError):
        _enable_private(monkeypatch, tmp_path, info, pages, readback={**pages, **changes})


def test_private_adoption_rechecks_publication_before_initializing_history(monkeypatch, tmp_path, info, pages):
    with pytest.raises(SetupError, match="not verified as a fresh empty private site"):
        _enable_private(monkeypatch, tmp_path, info, pages, deployments=[{"id": 123}])


def test_private_missing_site_never_creates_pages_or_enables_actions(monkeypatch, tmp_path, info):
    state = {"info": info, "pages": None, "history_exists": False, "actions": {"enabled": False}}

    def unexpected_mutation(*_args, **_kwargs):
        raise AssertionError("No API call is safe before a private Pages site is verified")

    monkeypatch.setattr(wizard, "github_api", unexpected_mutation)
    with pytest.raises(SetupError, match="has not verified a private Pages site"):
        wizard.enable_hosting(tmp_path, NAME, "main", state, [])
