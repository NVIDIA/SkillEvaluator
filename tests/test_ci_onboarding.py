# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral safety and output checks; live GitHub is verified separately."""

import hashlib
import json
import subprocess

import pytest
import yaml
from click.testing import CliRunner

from skillevaluator import ci_onboarding as wizard
from skillevaluator.cli import cli

RELEASE = {
    "schema_version": 1,
    "repository": "NVIDIA/SkillEvaluator",
    "workflow_ref": "e4cda13a85f34fa7a112b3529feaf24cfe587caf",
    "publisher_ref": "1f3c98cf1b178d5446e31cc187c97a6a697d599a",
    "image": "ghcr.io/nvidia/skillevaluator-tier1@sha256:" + "c" * 64,
}


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "consumer"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "-C", str(repo), *args],
            text=True,
        )

    git("init", "-b", "trunk")
    git("config", "user.name", "Onboarding tests")
    git("config", "user.email", "test@example.com")
    git("remote", "add", "origin", "https://github.com/example-org/consumer.git")
    for name in [
        "skills/a/SKILL.md",
        "skills/b/SKILL.md",
        ".agents/skills/c/SKILL.md",
        "tests/fixtures/invalid/SKILL.md",
    ]:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("---\nname: example\n---\n")
    existing = repo / ".github/workflows/existing.yml"
    existing.parent.mkdir(parents=True)
    existing.write_text("name: Preserve me\n")
    git("add", ".")
    git("commit", "-m", "Create test skills")
    return repo


@pytest.fixture
def release_file(tmp_path):
    path = tmp_path / "release.json"
    path.write_text(json.dumps(RELEASE))
    return path


def invoke(repository, release_file, *args, input_text=None):
    return CliRunner().invoke(
        cli, ["ci", "init", "--repo", str(repository), "--release", str(release_file), *args], input=input_text
    )


def test_full_preview_needs_no_login_and_leaves_checkout_unchanged(repository, release_file, monkeypatch):
    # A preview accidentally calling GitHub or writing workflows must fail this test.
    monkeypatch.setenv("GH_HOST", "wrong.example")

    def no_api(*args, **kwargs):
        raise AssertionError("Preview must not use GitHub")

    monkeypatch.setattr(wizard, "github_api", no_api)
    before = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"])
    result = invoke(repository, release_file, "--no-input", "--mode", "preview")
    assert result.exit_code == 0, result.output
    assert "PR comment" in result.output
    assert "Pages" in result.output
    assert "No changes applied" in result.output
    assert not (repository / ".github/workflows/skill-checks.yml").exists()
    assert not (repository / ".github/workflows/publish-skill-reports.yml").exists()
    assert subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"]) == before


@pytest.mark.parametrize("answer", ["1,2", "1, 2", "1 2", "1, 2 1"])
def test_folder_input_accepts_commas_and_spaces(repository, release_file, tmp_path, answer):
    receipt = tmp_path / "selection.json"
    result = invoke(
        repository,
        release_file,
        "--mode",
        "preview",
        "--min-score",
        "70",
        "--scan-mode",
        "auto",
        "--full",
        "--output",
        str(receipt),
        input_text=answer + "\n",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(receipt.read_text())["selected_roots"] == [".agents/skills", "skills"]
    assert not (repository / wizard.SCAN).exists()


def test_invalid_folder_choice_can_be_corrected_in_same_interview(repository, release_file, tmp_path):
    receipt = tmp_path / "recovered.json"
    before = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"])
    result = invoke(
        repository,
        release_file,
        "--mode",
        "preview",
        "--min-score",
        "70",
        "--scan-mode",
        "auto",
        "--full",
        "--output",
        str(receipt),
        input_text="0\n1,,2\n9\n1 2\n",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(receipt.read_text())["selected_roots"] == [".agents/skills", "skills"]
    assert subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"]) == before


def test_interactive_selection_can_recover_from_skill_limit(monkeypatch):
    answers = iter(["1 2", "1"])
    monkeypatch.setattr(wizard.click, "prompt", lambda *_args, **_kwargs: next(answers))
    roots = [{"root": "skills", "count": 100, "default": True}, {"root": "extra", "count": 1, "default": False}]
    assert wizard.select_roots(roots, (), False) == ["skills"]
    with pytest.raises(wizard.SetupError, match="at most 100"):
        wizard.select_roots(roots, ("skills", "extra"), True)


@pytest.mark.parametrize("optional_folders", [1, 2])
def test_optional_only_roots_require_explicit_numbers(repository, release_file, optional_folders):
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repository), *args], text=True
        )

    git("rm", "-r", "--cached", "skills", ".agents")
    if optional_folders == 2:
        sample = repository / "examples/demo/SKILL.md"
        sample.parent.mkdir(parents=True)
        sample.write_text("---\nname: demo\n---\n")
        git("add", "examples")
    assert not any(root["default"] for root in wizard.git_helpers.discover(repository))
    result = invoke(
        repository,
        release_file,
        "--mode",
        "preview",
        "--min-score",
        "70",
        "--scan-mode",
        "auto",
        "--checks-only",
        input_text="\n1\n",
    )
    assert result.exit_code == 0, result.output
    assert "No default: enter a number to continue." in result.output
    assert "Enter accepts" not in result.output
    assert "press Enter" not in result.output
    assert "Folders []" not in result.output
    assert "No changes applied" in result.output
    unattended = invoke(repository, release_file, "--mode", "preview", "--no-input")
    assert unattended.exit_code != 0
    assert "Select at least one displayed skill folder" in unattended.output


@pytest.mark.parametrize("width", [80, 140])
def test_completion_keeps_long_pr_links_copyable(monkeypatch, tmp_path, width):
    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    monkeypatch.setattr(wizard, "console", Console(file=stream, width=width, color_system=None))
    url = "https://github.com/example-org/skill-evaluator-example/pull/123456789"
    wizard.show_next_steps("example-org/skill-evaluator-example", True, url, repo=tmp_path)
    wizard.show_test_pr("example-org/skill-evaluator-example", url)
    assert stream.getvalue().count(url) == 2


def test_full_save_approves_exact_scan_bytes_and_preserves_existing_workflow(repository, release_file):
    result = invoke(repository, release_file, "--no-input", "--mode", "save", "--paths", "skills")
    assert result.exit_code == 0, result.output
    scan_path = repository / ".github/workflows/skill-checks.yml"
    scan_bytes = scan_path.read_bytes()
    scan = yaml.safe_load(scan_bytes)
    publisher = yaml.safe_load((repository / ".github/workflows/publish-skill-reports.yml").read_text())
    inputs = publisher["jobs"]["reports"]["with"]
    # Git's blob identity, calculated independently from the renderer under test.
    expected = hashlib.sha1(f"blob {len(scan_bytes)}\0".encode() + scan_bytes).hexdigest()
    assert inputs["trusted-caller-blobs"] == expected
    assert scan["permissions"] == {"contents": "read", "id-token": "write"}
    assert scan["jobs"]["skills"]["with"]["paths"] == "skills"
    assert scan["jobs"]["skills"]["with"]["report-base-url"] == inputs["report-base-url"]
    assert publisher["jobs"]["reports"]["uses"].endswith("@" + RELEASE["publisher_ref"])
    assert inputs["tools-ref"] == RELEASE["publisher_ref"]
    assert inputs["trusted-evaluator-shas"] == RELEASE["workflow_ref"]
    assert (repository / ".github/workflows/existing.yml").read_text() == "name: Preserve me\n"
    assert "Remote settings were not changed" in result.output


def test_scan_only_does_not_request_identity_or_generate_publisher(repository, release_file):
    result = invoke(repository, release_file, "--no-input", "--mode", "save", "--checks-only")
    assert result.exit_code == 0, result.output
    scan = yaml.safe_load((repository / ".github/workflows/skill-checks.yml").read_text())
    assert scan["permissions"] == {"contents": "read"}
    assert "report-base-url" not in scan["jobs"]["skills"]["with"]
    assert not (repository / ".github/workflows/publish-skill-reports.yml").exists()


def test_conflicting_publisher_does_not_partially_write_scan(repository, release_file):
    target = repository / ".github/workflows/publish-skill-reports.yml"
    target.write_text("name: My existing publisher\n")
    result = invoke(repository, release_file, "--no-input", "--mode", "save")
    assert result.exit_code != 0
    assert "different content" in result.output
    assert not (repository / ".github/workflows/skill-checks.yml").exists()
    assert target.read_text() == "name: My existing publisher\n"


def test_save_is_idempotent_even_before_files_are_committed(repository, release_file):
    first = invoke(repository, release_file, "--no-input", "--mode", "save")
    assert first.exit_code == 0, first.output
    before = (repository / ".github/workflows/skill-checks.yml").stat().st_mtime_ns
    second = invoke(repository, release_file, "--no-input", "--mode", "save")
    assert second.exit_code == 0, second.output
    assert "already match" in second.output
    assert (repository / ".github/workflows/skill-checks.yml").stat().st_mtime_ns == before


@pytest.mark.parametrize(
    "url",
    [
        "http://example-org.github.io/example/",
        "https://evil.example/",
        "https://user:secret@example-org.github.io/example/",
        "https://example-org.github.io/example/?x=1",
    ],
)
def test_report_url_rejects_external_or_credential_bearing_sites(repository, release_file, url):
    result = invoke(repository, release_file, "--no-input", "--mode", "preview", "--report-url", url)
    assert result.exit_code != 0
    assert "report URL" in result.output


def test_invalid_release_pin_stops_before_writing(repository, release_file):
    data = dict(RELEASE, publisher_ref="main")
    release_file.write_text(json.dumps(data))
    result = invoke(repository, release_file, "--no-input", "--mode", "save")
    assert result.exit_code != 0
    assert "immutable" in result.output
    assert not (repository / ".github/workflows/skill-checks.yml").exists()


def test_symlink_parent_is_refused_without_writing_outside(repository, release_file, tmp_path):
    old = repository / ".github/workflows/existing.yml"
    old.unlink()
    old.parent.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    old.parent.symlink_to(outside, target_is_directory=True)
    result = invoke(repository, release_file, "--no-input", "--mode", "save")
    assert result.exit_code != 0
    assert "Symlink" in result.output
    assert list(outside.iterdir()) == []


def test_dirty_repository_is_refused_before_pr_setup(repository, release_file):
    (repository / "unrelated.txt").write_text("Keep my changes")
    result = invoke(repository, release_file, "--no-input", "--mode", "pr")
    assert result.exit_code != 0
    assert "existing changes" in result.output
    assert (repository / "unrelated.txt").read_text() == "Keep my changes"


def test_declined_apply_is_read_only(repository, release_file):
    result = invoke(repository, release_file, "--mode", "pr", input_text="\n\n\n\nn\n")
    assert result.exit_code == 0, result.output
    assert "Cancelled" in result.output
    assert not (repository / ".github/workflows/skill-checks.yml").exists()


def test_existing_history_requires_this_repositorys_ownership_marker(monkeypatch):
    marker = {"purpose": "skillevaluator-reports", "repository": "example-org/another-repo"}
    import base64

    def api(method, endpoint, **kwargs):
        if endpoint.endswith("/git/ref/heads/gh-pages"):
            return {"ref": "refs/heads/gh-pages", "object": {"sha": "a" * 40}}
        if "/contents/.skillevaluator-report-site.json" in endpoint:
            return {"encoding": "base64", "content": base64.b64encode(json.dumps(marker).encode()).decode()}
        raise AssertionError("Unowned history must not be accessed or changed")

    monkeypatch.setattr(wizard, "github_api", api)
    with pytest.raises(wizard.SetupError, match="not owned"):
        wizard.history_preflight("example-org/this-repo")


def test_environment_approval_requirements_are_never_removed(monkeypatch):
    def api(method, endpoint, *args, **kwargs):
        if method != "GET":
            raise AssertionError("Never remove an existing approval requirement")
        return {
            "name": "github-pages",
            "protection_rules": [
                {"type": "required_reviewers", "reviewers": [{"type": "User", "reviewer": {"login": "reviewer"}}]}
            ],
            "deployment_branch_policy": None,
        }

    monkeypatch.setattr(wizard, "github_api", api)
    with pytest.raises(wizard.SetupError, match="requires approval"):
        wizard.ensure_environment("example-org/test", "main")


def test_setup_status_detects_a_changed_scan_without_approving_it(repository, release_file, monkeypatch):
    import base64

    result = invoke(repository, release_file, "--no-input", "--mode", "save")
    assert result.exit_code == 0, result.output

    def api(method, endpoint, *args, **kwargs):
        if endpoint == "repos/example-org/consumer":
            return {"default_branch": "trunk"}
        if endpoint.endswith("/pages"):
            return {"html_url": "https://example-org.github.io/prototype-unit-only/", "build_type": "workflow"}
        for path in [wizard.SCAN, wizard.PUBLISH]:
            if f"/contents/{path}" in endpoint:
                content = (repository / path).read_bytes()
                if path == wizard.SCAN:
                    content += b"# A later edit, not approved by the publisher\n"
                return {"encoding": "base64", "content": base64.b64encode(content).decode()}
        if "/runs?" in endpoint:
            return {"workflow_runs": []}
        raise AssertionError(endpoint)

    monkeypatch.setattr(wizard, "github_api", api)
    result = CliRunner().invoke(cli, ["ci", "status", "--repo", str(repository)])
    assert result.exit_code == 0, result.output
    assert "Needs refresh" in result.output


def test_pages_is_enabled_before_history_push_can_auto_enable_legacy_pages(monkeypatch, tmp_path):
    state = {"actions": {"enabled": True}, "pages": None, "history_exists": False}
    provider = {"build_type": None, "history_initialized": False}

    def api(method, endpoint, payload=None, **kwargs):
        if method == "POST" and endpoint.endswith("/pages"):
            provider["build_type"] = payload["build_type"]
            return {"build_type": provider["build_type"]}
        if method == "GET" and endpoint.endswith("/pages"):
            return {
                "build_type": provider["build_type"],
                "html_url": "https://example-org.github.io/test/",
                "cname": None,
            }
        raise AssertionError((method, endpoint))

    def history(repo, name):
        assert provider["build_type"] == "workflow", "Creating gh-pages first can auto-enable legacy Pages"
        provider["history_initialized"] = True

    monkeypatch.setattr(wizard, "github_api", api)
    monkeypatch.setattr(wizard, "initialize_history", history)
    monkeypatch.setattr(wizard, "ensure_environment", lambda *_args: None)
    wizard.enable_hosting(tmp_path, "example-org/test", "main", state, [])
    assert provider == {"build_type": "workflow", "history_initialized": True}


def test_owned_legacy_pages_setup_can_resume_without_resetting_history(monkeypatch, tmp_path):
    provider = {"build_type": "legacy", "history": [{"run_id": "123"}]}
    state = {"actions": {"enabled": True}, "pages": {"build_type": "legacy", "cname": None}, "history_exists": True}

    def api(method, endpoint, payload=None, **kwargs):
        if method == "PUT" and endpoint.endswith("/pages"):
            provider["build_type"] = payload["build_type"]
            return {}
        if method == "GET" and endpoint.endswith("/pages"):
            return {
                "build_type": provider["build_type"],
                "html_url": "https://example-org.github.io/test/",
                "cname": None,
            }
        raise AssertionError((method, endpoint))

    def no_reset(*args):
        raise AssertionError("Existing history must not be recreated")

    monkeypatch.setattr(wizard, "github_api", api)
    monkeypatch.setattr(wizard, "initialize_history", no_reset)
    monkeypatch.setattr(wizard, "ensure_environment", lambda *_args: None)
    wizard.enable_hosting(tmp_path, "example-org/test", "main", state, [])
    assert provider == {"build_type": "workflow", "history": [{"run_id": "123"}]}


def test_demo_requires_merged_setup_before_pushing(repository, monkeypatch):
    def api(method, endpoint, *args, **kwargs):
        if endpoint == "repos/example-org/consumer":
            return {"default_branch": "trunk", "permissions": {"admin": True}, "archived": False}
        if "/contents/" in endpoint:
            return None
        raise AssertionError("Demo must stop before any remote mutation")

    monkeypatch.setattr(wizard, "github_api", api)
    result = CliRunner().invoke(cli, ["ci", "demo", "--repo", str(repository), "--no-input"])
    assert result.exit_code != 0
    assert "Merge the setup PR" in result.output
    assert not (repository / ".github/workflows/skill-checks.yml").exists()


def _remote_state(monkeypatch, info, *, actions=None, pages=None):
    calls = []
    git_calls = []

    def api(method, endpoint, payload=None, **kwargs):
        calls.append((method, endpoint, payload))
        assert method == "GET", "Preflight must not change remote settings"
        if endpoint.endswith("/actions/permissions"):
            return actions or {"enabled": True, "allowed_actions": "all"}
        if endpoint.endswith("/pages"):
            return pages
        return info

    def git(repo, *args, **kwargs):
        git_calls.append(args)
        if args[0] == "rev-parse":
            return "a" * 40
        return ""

    monkeypatch.setattr(wizard, "github_api", api)
    monkeypatch.setattr(wizard.git_helpers, "git", git)
    monkeypatch.setattr(wizard, "history_preflight", lambda _name: False)
    return calls, git_calls


@pytest.mark.parametrize("owner", ["example-org", "another-user", "EnterpriseTeam"])
def test_automatic_setup_supports_any_github_owner(repository, monkeypatch, owner):
    info = {"default_branch": "trunk", "permissions": {"admin": True}, "private": False, "visibility": "public"}
    calls, _ = _remote_state(monkeypatch, info)
    state = wizard.remote_preflight(repository, f"{owner}/skills", "trunk", True)
    assert state["info"] == info
    assert calls[0][1] == f"repos/{owner}/skills"


@pytest.mark.parametrize("visibility", ["private", "internal"])
def test_private_checks_only_does_not_access_pages(repository, monkeypatch, visibility):
    info = {"default_branch": "trunk", "permissions": {"admin": True}, "private": True, "visibility": visibility}
    calls, _ = _remote_state(monkeypatch, info)
    state = wizard.remote_preflight(repository, "example-org/private-skills", "trunk", False)
    assert state["pages"] is None
    assert all(not endpoint.endswith("/pages") for _, endpoint, _ in calls)


@pytest.mark.parametrize(
    "private,visibility", [(True, "private"), (True, "internal"), (False, "internal"), (None, "public")]
)
def test_nonpublic_hosting_stops_before_fetch_or_settings(repository, monkeypatch, private, visibility):
    info = {"default_branch": "trunk", "permissions": {"admin": True}, "private": private, "visibility": visibility}
    calls, git_calls = _remote_state(monkeypatch, info)
    with pytest.raises(wizard.SetupError, match="--checks-only"):
        wizard.remote_preflight(repository, "example-org/private-skills", "trunk", True)
    assert len(calls) == 1
    assert all(args[0] == "status" for args in git_calls)


@pytest.mark.parametrize("archived,permissions", [(True, {"admin": True}), (False, {"push": True}), (False, {})])
def test_remote_setup_requires_active_repo_and_admin(repository, monkeypatch, archived, permissions):
    info = {"default_branch": "trunk", "permissions": permissions, "private": False, "archived": archived}
    calls, git_calls = _remote_state(monkeypatch, info)
    with pytest.raises(wizard.SetupError, match="admin access"):
        wizard.remote_preflight(repository, "example-org/skills", "trunk", False)
    assert len(calls) == 1
    assert all(args[0] == "status" for args in git_calls)


def test_restricted_actions_are_preserved_and_point_to_manual_setup(repository, monkeypatch):
    info = {"default_branch": "trunk", "permissions": {"admin": True}, "private": False}
    calls, _ = _remote_state(monkeypatch, info, actions={"enabled": True, "allowed_actions": "selected"})
    with pytest.raises(wizard.SetupError, match="--mode save"):
        wizard.remote_preflight(repository, "example-org/skills", "trunk", True)
    assert all(method == "GET" for method, _, _ in calls)


def test_existing_unowned_pages_site_is_not_replaced(repository, monkeypatch):
    info = {"default_branch": "trunk", "permissions": {"admin": True}, "private": False}
    calls, _ = _remote_state(monkeypatch, info, pages={"build_type": "workflow", "cname": None})
    with pytest.raises(wizard.SetupError, match="will not be replaced"):
        wizard.remote_preflight(repository, "example-org/skills", "trunk", True)
    assert all(method == "GET" for method, _, _ in calls)


def test_default_selection_is_not_specific_to_a_demo_folder():
    roots = [{"root": "sandbox-skills", "count": 2, "default": True}, {"root": "skills", "count": 3, "default": True}]
    assert wizard.select_roots(roots, (), True) == ["sandbox-skills", "skills"]


@pytest.mark.parametrize("content", [b"not a workflow", b"jobs: [", b"jobs: {}", b"jobs: {skills: {with: 3}}"])
def test_malformed_existing_workflow_has_actionable_error(content):
    import base64

    value = {"encoding": "base64", "content": base64.b64encode(content).decode()}
    with pytest.raises(wizard.SetupError, match="workflow"):
        wizard.workflow_inputs(value, "skills")


def test_bundled_release_and_ci_namespace_are_available():
    from importlib.resources import files

    release = wizard.load_release(files("skillevaluator").joinpath("ci-release.json"))
    assert release["repository"] == "NVIDIA/SkillEvaluator"
    assert release["schema_version"] == 1
    assert "ci" in cli.commands
    for command in ["init", "status", "demo"]:
        result = CliRunner().invoke(cli, ["ci", command, "--help"])
        assert result.exit_code == 0, result.output


def test_default_branch_mismatch_stops_before_fetch(repository, monkeypatch):
    info = {"default_branch": "main", "permissions": {"admin": True}, "private": False}
    calls, git_calls = _remote_state(monkeypatch, info)
    with pytest.raises(wizard.SetupError, match="--default-branch"):
        wizard.remote_preflight(repository, "example-org/skills", "trunk", True)
    assert len(calls) == 1
    assert all(args[0] == "status" for args in git_calls)


@pytest.mark.parametrize("console_script", [True, False])
def test_windows_followup_quotes_powershell_paths(monkeypatch, tmp_path, console_script):
    install = tmp_path / "User's Python"
    install.mkdir()
    python = install / "python.exe"
    if console_script:
        (install / "skillevaluator.exe").touch()
    monkeypatch.setattr(wizard.sys, "platform", "win32")
    monkeypatch.setattr(wizard.sys, "executable", str(python))
    command = wizard.cli_command(tmp_path / "consumer", "ci", "demo")
    assert command.startswith("& '")
    assert "User''s Python" in command
    assert command.endswith("'ci' 'demo'")
    assert ("skillevaluator.exe" in command) == console_script
    assert ("'-m' 'skillevaluator.cli'" in command) != console_script


@pytest.mark.parametrize("full", [True, False])
def test_missing_git_identity_stops_before_github_settings(repository, monkeypatch, full):
    def no_identity(_repo):
        raise wizard.SetupError("Set your Git commit identity")

    def unexpected_remote(*args, **kwargs):
        raise AssertionError("Missing identity must be checked before GitHub changes")

    monkeypatch.setattr(wizard.git_helpers, "require_identity", no_identity)
    monkeypatch.setattr(wizard, "remote_preflight", unexpected_remote)
    with pytest.raises(wizard.SetupError, match="Git commit identity"):
        wizard.create_setup(repository, "example-org/consumer", "trunk", {}, full, {})


def test_pages_protected_branch_endpoint_escapes_special_characters(monkeypatch):
    calls = []

    def api(method, endpoint, *args, **kwargs):
        calls.append((method, endpoint))
        if endpoint.endswith("/environments/github-pages"):
            return {"deployment_branch_policy": {"protected_branches": True}, "protection_rules": []}
        return {"protected": True}

    monkeypatch.setattr(wizard, "github_api", api)
    wizard.ensure_environment("example-org/consumer", "release/2026#11&stable")
    assert calls[-1] == ("GET", "repos/example-org/consumer/branches/release%2F2026%2311%26stable")
    assert all(method == "GET" for method, _ in calls)
