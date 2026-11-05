# SPDX-License-Identifier: Apache-2.0
"""Interactive GitHub CI onboarding with pinned workflows and optional hosted reports."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import click
import yaml
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from skillevaluator import _ci_onboarding_all_tiers as all_tier_helpers
from skillevaluator import _ci_onboarding_git as git_helpers
from skillevaluator import _ci_onboarding_pages as pages_helpers
from skillevaluator import _ci_onboarding_workflows as workflow_helpers

SCAN = ".github/workflows/skill-checks.yml"
PUBLISH = ".github/workflows/publish-skill-reports.yml"
BRANCH = "ci/skillevaluator-setup"
DEMO_BRANCH = "ci/skillevaluator-test"
MARKER = ".skillevaluator-report-site.json"
console = Console(highlight=False)
SetupError = git_helpers.SetupError


def symbol(value: str, fallback: str) -> str:
    """Keep status labels readable in terminals without Unicode support."""
    try:
        value.encode(console.encoding)
    except (UnicodeError, LookupError):
        return fallback
    return value


def question(number: int, title: str, *content) -> None:
    console.print(
        Panel(
            Group(*(Text(item) if isinstance(item, str) else item for item in content)),
            title=Text(f"{number}/5  {title}", style="bold cyan"),
            title_align="left",
            border_style="cyan",
            width=min(console.width, 92),
        )
    )


def print_link(url: str) -> None:
    # Keep a literal, untruncated URL in captured output, even on narrow screens.
    console.print(Text(url, style=f"link {url}"), soft_wrap=True)


def results_help(name: str, full: bool) -> str:
    text = f"Checks {symbol('→', '>')} Skill checks"
    if full:
        text += f"  |  Bot comment {symbol('→', '>')} Open full report"
    else:
        text += f"  |  Actions {symbol('→', '>')} Download report"
    return text


def cli_command(repo: Path, *arguments: str) -> str:
    """Reuse this interpreter's CLI; Windows output targets PowerShell."""
    # Do not resolve the interpreter symlink: that would leave its virtualenv.
    windows = sys.platform == "win32"
    entrypoint = Path(sys.executable).parent / ("skillevaluator.exe" if windows else "skillevaluator")
    if entrypoint.is_file():
        try:
            command = str(entrypoint.relative_to(repo.resolve()))
            if windows:
                command = ".\\" + command
            elif "/" not in command:
                command = "./" + command
        except ValueError:
            command = str(entrypoint)
        parts = [command, *arguments]
    else:
        parts = [sys.executable, "-m", "skillevaluator.cli", *arguments]
    if windows:
        return "& " + " ".join("'" + part.replace("'", "''") + "'" for part in parts)
    return shlex.join(parts)


def show_next_steps(name: str, full: bool, setup_url: str | None = None, *, repo: Path | None = None) -> None:
    if setup_url:
        heading = f"{symbol('✓', '[OK]')} Setup PR ready"
        body = "Merge this PR to enable checks."
    else:
        heading = f"{symbol('✓', '[OK]')} Setup is configured"
        body = "Ready for your first skill-change PR."
    console.print(
        Panel(Text(body), title=Text(heading), title_align="left", border_style="green", width=min(console.width, 92))
    )
    if setup_url:
        print_link(setup_url)
    console.print("After merging, optionally create a test PR:" if setup_url else "Optional: create a test PR:")
    if sys.platform == "win32":
        console.print("Run in PowerShell:")
    console.print(Text("  " + cli_command(repo or Path.cwd(), "ci", "demo")), soft_wrap=True)


def show_test_pr(name: str, url: str, *, existing: bool = False, full: bool = True) -> None:
    heading = f"{symbol('✓', '[OK]')} " + ("Test PR already exists" if existing else "Test PR created")
    body = results_help(name, full) + (
        "\nComment follows publication. Close this PR when done."
        if full
        else "\nClose this PR when done; no merge needed."
    )
    console.print(
        Panel(Text(body), title=Text(heading), title_align="left", border_style="green", width=min(console.width, 92))
    )
    print_link(url)


def load_release(path: Path) -> dict:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and data.get("schema_version") == 2:
        return all_tier_helpers.validate_release(data)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise SetupError("The CI release descriptor must use schema_version 1.")
    if data.get("repository") != "NVIDIA/SkillEvaluator":
        raise SetupError("The CI release must use NVIDIA/SkillEvaluator as its workflow provider.")
    for field in ("workflow_ref", "publisher_ref"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(data.get(field, ""))):
            raise SetupError(f"{field} must be an immutable full commit SHA.")
    if not re.fullmatch(r"ghcr\.io/nvidia/skillevaluator-tier1@sha256:[0-9a-f]{64}", str(data.get("image", ""))):
        raise SetupError("The image must use the official immutable digest.")
    return data


def report_url(name: str, custom: str | None = None, *, verified_private_url: str | None = None) -> str:
    return pages_helpers.report_url(name, custom, verified_private_url=verified_private_url)


def blob_id(content: str) -> str:
    data = content.encode("utf-8")
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def yaml_text(value: dict) -> str:
    return yaml.dump(value, Dumper=git_helpers.WorkflowDumper, sort_keys=False, allow_unicode=True, width=1000)


def committed_blob(repo: Path, path: str, content: str) -> str:
    """Hash using the same path filters Git applies when staging this file."""
    with tempfile.TemporaryDirectory(prefix="skill-caller-blob-") as temporary:
        source = Path(temporary) / "scan.yml"
        source.write_bytes(content.encode("utf-8"))
        blob = git_helpers.git(repo, "hash-object", "--path", path, "--", str(source)).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", blob):
        raise SetupError("Git did not return the scan workflow's commit identity.")
    return blob


def bind_caller_approval(repo: Path, files: dict[str, str]) -> dict[str, str]:
    """Approve the bytes Git will commit, including configured line endings."""
    if PUBLISH not in files:
        return files
    blob = committed_blob(repo, SCAN, files[SCAN])
    publisher = workflow_helpers.parse_workflow(files[PUBLISH])
    publisher["jobs"]["reports"]["with"]["trusted-caller-blobs"] = blob
    return {
        **files,
        PUBLISH: "# Generated by SkillEvaluator CI onboarding; publisher pinned to a verified commit.\n"
        + yaml_text(publisher),
    }


def caller_approval(files: dict[str, str]) -> str:
    if PUBLISH in files:
        return workflow_helpers.parse_workflow(files[PUBLISH])["jobs"]["reports"]["with"]["trusted-caller-blobs"]
    return blob_id(files[SCAN])


def render_files(
    release: dict,
    roots: list[str],
    score: int,
    scan_mode: str,
    branch: str,
    name: str,
    full: bool = True,
    url: str | None = None,
    *,
    existing: workflow_helpers.ExistingScan | None = None,
    verified_private_url: str | None = None,
) -> dict[str, str]:
    url = report_url(name, url, verified_private_url=verified_private_url) if full else None
    if (
        full
        and existing
        and existing.data["jobs"][existing.job]["with"].get("artifact-name", "skill-evaluator-tier1")
        != "skill-evaluator-tier1"
    ):
        raise SetupError("Reporting requires artifact-name: skill-evaluator-tier1 in the existing scan.")
    scan = {
        "name": "Skill checks",
        "on": {"pull_request": None, "push": {"branches": [branch]}, "workflow_dispatch": None},
        "permissions": {"contents": "read"},
        "jobs": {
            "skills": {
                "uses": f"{release['repository']}/.github/workflows/evaluate-skills.yml@{release['workflow_ref']}",
                "with": {
                    "image": release["image"],
                    "paths": "\n".join(roots),
                    "min-score": score,
                    "scan-mode": scan_mode,
                    "llm": False,
                },
            }
        },
    }
    if full:
        scan["permissions"]["id-token"] = "write"
        scan["jobs"]["skills"]["with"]["report-base-url"] = url
    scan_text = (
        (existing.with_identity_permission() if full else existing.text)
        if existing
        else "# Generated by SkillEvaluator CI onboarding.\n" + yaml_text(scan)
    )
    files = {SCAN: scan_text}
    if full:
        publisher = {
            "name": "Publish skill reports",
            "on": {
                "workflow_run": {"workflows": ["Skill checks"], "types": ["completed"]},
                "workflow_dispatch": {
                    "inputs": {
                        "run-id": {"description": "Completed Skill checks run", "required": True, "type": "string"},
                        "run-attempt": {"description": "Completed run attempt", "required": True, "type": "string"},
                    }
                },
            },
            "permissions": {
                "contents": "write",
                "actions": "read",
                "pages": "write",
                "id-token": "write",
                "pull-requests": "write",
            },
            "jobs": {
                "reports": {
                    "uses": f"{release['repository']}/.github/workflows/publish-skill-reports.yml@{release['publisher_ref']}",
                    "with": {
                        "run-id": "${{ format('{0}', github.event.workflow_run.id || inputs.run-id) }}",
                        "run-attempt": "${{ format('{0}', github.event.workflow_run.run_attempt || inputs.run-attempt) }}",
                        "report-base-url": url,
                        "trusted-evaluator-shas": existing.evaluator_sha if existing else release["workflow_ref"],
                        "trusted-caller-blobs": blob_id(scan_text),
                        "tools-ref": release["publisher_ref"],
                        "tools-repository": release["repository"],
                    },
                }
            },
        }
        if existing and existing.delegated_wrapper:
            inputs = publisher["jobs"]["reports"]["with"]
            inputs.pop("trusted-evaluator-shas")
            inputs["trusted-wrapper-workflows"] = existing.delegated_wrapper
            inputs["delegated-evaluator-wrapper"] = existing.delegated_wrapper
        files[PUBLISH] = (
            "# Generated by SkillEvaluator CI onboarding; publisher pinned to a verified commit.\n"
            + yaml_text(publisher)
        )
    return files


def check_files(repo: Path, files: dict[str, str], expected: dict[str, str] | None = None) -> list[str]:
    changed = []
    for name, content in files.items():
        git_helpers.no_symlinks(repo, name)
        target = repo / name
        if target.exists():
            if not target.is_file() or target.read_bytes() != content.encode():
                if (
                    not target.is_file()
                    or name not in (expected or {})
                    or target.read_bytes() != expected[name].encode()
                ):
                    raise SetupError(f"{name} already exists with different content; it will not be replaced.")
                changed.append(name)
        else:
            changed.append(name)
    return changed


def save_files(repo: Path, files: dict[str, str], expected: dict[str, str] | None = None) -> None:
    import stat
    import uuid
    from contextlib import suppress

    missing = check_files(repo, files, expected)
    if not missing:
        return
    git_helpers.require_clean(repo)
    directory_handles = []
    written = []

    def parent_handle(target: Path):
        git_helpers.no_symlinks(repo, target.relative_to(repo).as_posix())
        if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
            target.parent.mkdir(parents=True, exist_ok=True)
            git_helpers.no_symlinks(repo, target.relative_to(repo).as_posix())
            return None
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        handle = os.open(repo, flags)
        try:
            for part in target.relative_to(repo).parts[:-1]:
                try:
                    child = os.open(part, flags, dir_fd=handle)
                except FileNotFoundError:
                    os.mkdir(part, dir_fd=handle)
                    child = os.open(part, flags, dir_fd=handle)
                os.close(handle)
                handle = child
        except BaseException:
            os.close(handle)
            raise
        directory_handles.append(handle)
        return handle

    def validate_parent(target: Path, handle) -> None:
        git_helpers.no_symlinks(repo, target.relative_to(repo).as_posix())
        if handle is not None:
            current, opened = target.parent.stat(), os.fstat(handle)
            if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                raise SetupError("A workflow directory changed during setup; no replacement was approved.")

    def snapshot(target: Path, handle):
        validate_parent(target, handle)
        kwargs = {"dir_fd": handle} if handle is not None else {}
        path = target.name if handle is not None else target
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        try:
            descriptor = os.open(path, flags, **kwargs)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise SetupError("The workflow target is not a regular file.")
            content = stream.read()
        return content, stat.S_IMODE(metadata.st_mode), metadata.st_dev, metadata.st_ino

    def replace(target: Path, handle, content: bytes, previous):
        validate_parent(target, handle)
        stage_name = f".skillevaluator-{uuid.uuid4().hex}.tmp"
        stage = stage_name if handle is not None else target.parent / stage_name
        destination = target.name if handle is not None else target
        kwargs = {"dir_fd": handle} if handle is not None else {}
        pair_kwargs = {"src_dir_fd": handle, "dst_dir_fd": handle} if handle is not None else {}
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        descriptor = os.open(stage, flags, previous[1] if previous is not None else 0o644, **kwargs)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                if previous is not None:
                    if hasattr(os, "fchmod"):
                        os.fchmod(stream.fileno(), previous[1])
                    else:
                        Path(stage).chmod(previous[1])
                metadata = os.fstat(stream.fileno())
            if snapshot(target, handle) != previous:
                raise SetupError("A workflow changed during setup; no replacement was approved.")
            validate_parent(target, handle)
            if previous is None:
                # Hard-link publication is exclusive: a concurrently created
                # target is never overwritten, even after the last readback.
                os.link(stage, destination, **pair_kwargs)
            else:
                # Replace the directory entry, never write through a target
                # that could have become a symlink after its final readback.
                os.replace(stage, destination, **pair_kwargs)  # noqa: PTH105 -- preserve directory-descriptor binding
            return content, stat.S_IMODE(metadata.st_mode), metadata.st_dev, metadata.st_ino
        finally:
            # Cleanup must not follow a redirected parent or turn a completed
            # replacement into an untracked write if unlinking the temp fails.
            with suppress(OSError, SetupError):
                if handle is None:
                    validate_parent(target, handle)
                os.unlink(stage, **kwargs)

    try:
        for name in missing:
            target = repo / name
            handle = parent_handle(target)
            previous = snapshot(target, handle)
            if previous is None and name in (expected or {}):
                raise SetupError("A workflow disappeared during setup; no replacement was approved.")
            if previous is not None and (name not in (expected or {}) or previous[0] != expected[name].encode()):
                raise SetupError("A workflow changed during setup; no replacement was approved.")
            installed = replace(target, handle, files[name].encode(), previous)
            written.append((target, handle, previous, installed))
    except (OSError, SetupError) as exc:
        incomplete = False
        for target, handle, previous, installed in reversed(written):
            try:
                if snapshot(target, handle) != installed:
                    incomplete = True
                    continue
                if previous is None:
                    os.unlink(target.name, dir_fd=handle) if handle is not None else target.unlink()
                else:
                    replace(target, handle, previous[0], installed)
            except (OSError, SetupError):
                incomplete = True
        if incomplete:
            raise SetupError(
                "Workflow saving failed and a target changed concurrently. Changed targets were left untouched; "
                "inspect the checkout before retrying."
            ) from exc
        raise
    finally:
        for handle in directory_handles:
            os.close(handle)


def github_api(method: str, endpoint: str, payload: dict | None = None, *, missing_ok=False):
    args = [
        "gh",
        "api",
        "--hostname",
        "github.com",
        "--include",
        "--method",
        method,
        endpoint,
        "--header",
        "Accept: application/vnd.github+json",
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k != "GH_REPO"}
    env["GH_HOST"] = "github.com"
    with tempfile.TemporaryDirectory(prefix="skill-setup-api-") as temporary:
        if payload is not None:
            path = Path(temporary) / "request.json"
            path.write_text(json.dumps(payload))
            args += ["--input", str(path)]
        result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=90)
    response = result.stdout.replace("\r\n", "\n")
    match = re.search(r"^HTTP/[\d.]+\s+(\d+)", response)
    status = int(match[1]) if match else 0
    if missing_ok and status == 404:
        return None
    if result.returncode or not 200 <= status < 300:
        raise SetupError(
            f"GitHub {method} {endpoint} failed (HTTP {status or 'unknown'}). "
            "Check your gh login and repository permissions. No credential was printed."
        )
    body = response.partition("\n\n")[2].strip()
    return json.loads(body) if body else {}


def decode_content(value: dict) -> bytes:
    if value.get("encoding") != "base64" or not isinstance(value.get("content"), str):
        raise SetupError("GitHub did not return the expected bounded file content.")
    return base64.b64decode(value["content"], validate=False)


def workflow_inputs(value: dict, job: str) -> dict:
    """Validate the small workflow interface that status and demo consume."""
    try:
        content = yaml.safe_load(decode_content(value))
        inputs = content["jobs"][job]["with"]
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise SetupError("The existing workflow does not match SkillEvaluator's expected setup.") from exc
    if not isinstance(inputs, dict):
        raise SetupError("The existing workflow has invalid SkillEvaluator inputs.")
    return inputs


def history_preflight(name: str) -> bool:
    prefix = f"repos/{name}"
    ref = github_api("GET", prefix + "/git/ref/heads/gh-pages", missing_ok=True)
    if ref is None:
        return False
    marker = github_api("GET", prefix + f"/contents/{MARKER}?ref=gh-pages", missing_ok=True)
    if marker is None or json.loads(decode_content(marker)) != {
        "purpose": "skillevaluator-reports",
        "repository": name,
    }:
        raise SetupError(
            "An existing gh-pages branch is not owned by SkillEvaluator. Its site/history will not be overwritten."
        )
    history = github_api("GET", prefix + "/contents/history.json?ref=gh-pages")
    if not isinstance(json.loads(decode_content(history)), list):
        raise SetupError("Existing report history is invalid; inspect it before continuing.")
    return True


def remote_preflight(repo: Path, name: str, branch: str, full: bool) -> dict:
    git_helpers.require_clean(repo)
    info = github_api("GET", f"repos/{name}")
    if info.get("archived") or not info.get("permissions", {}).get("admin"):
        raise SetupError(
            "Automatic GitHub setup requires repository admin access. Use --mode save to prepare files for an owner."
        )
    pages = None
    private_url = None
    if full and pages_helpers.is_nonpublic(info):
        pages_helpers.check_private_repository(name, info)
        try:
            pages = github_api("GET", f"repos/{name}/pages", missing_ok=True)
        except SetupError as exc:
            raise SetupError(
                "Private Pages could not be verified. Check your gh login, Pages read permission, and repository "
                "admin access. Private hosting requires an Enterprise Cloud organization that allows private Pages; "
                "use --checks-only to keep reports in Actions artifacts."
            ) from exc
        private_url = pages_helpers.private_pages_url(name, info, pages)
    if info["default_branch"] != branch:
        raise SetupError("The selected branch differs from GitHub's default branch. Use --default-branch.")
    canonical = f"https://github.com/{name}.git"
    git_helpers.git(
        repo, "fetch", "--no-tags", canonical, f"refs/heads/{branch}:refs/remotes/origin/{branch}", network=True
    )
    if (
        git_helpers.git(repo, "rev-parse", "HEAD").strip()
        != git_helpers.git(repo, "rev-parse", f"origin/{branch}").strip()
    ):
        raise SetupError("Update your checkout to the current remote default branch before creating the setup PR.")
    actions = github_api("GET", f"repos/{name}/actions/permissions")
    if actions.get("allowed_actions") != "all":
        raise SetupError(
            "This repository restricts external actions. Automatic setup preserves that policy; use --mode save and have an owner review the workflow allowlist."
        )
    state = {
        "info": info,
        "actions": actions,
        "history_exists": False,
        "pages": pages,
        "verified_private_url": private_url,
        "report_url": report_url(name, verified_private_url=private_url) if full else None,
    }
    if full:
        state["history_exists"] = history_preflight(name)
        if private_url is None:
            state["pages"] = github_api("GET", f"repos/{name}/pages", missing_ok=True)
        deployments = None
        if private_url is not None and not state["history_exists"]:
            deployments = github_api("GET", f"repos/{name}/deployments?per_page=1")
        pages_helpers.check_site(name, state["pages"], history_exists=state["history_exists"], deployments=deployments)
    return state


def branch_preflight(
    repo: Path, name: str, branch: str, files: dict[str, str], changed_files: list[str]
) -> dict | None:
    canonical = f"https://github.com/{name}.git"
    remote = git_helpers.git(repo, "ls-remote", "--heads", canonical, f"refs/heads/{BRANCH}", network=True).strip()
    prs = json.loads(
        git_helpers.run(
            ["gh", "pr", "list", "--repo", name, "--head", BRANCH, "--state", "open", "--json", "url,baseRefName"]
        )
    )
    if len(prs) > 1 or (prs and prs[0]["baseRefName"] != branch):
        raise SetupError("An unexpected onboarding PR exists. Inspect it before continuing.")
    if not remote:
        if prs:
            raise SetupError("The onboarding PR has no matching branch. Inspect it manually.")
        return None
    git_helpers.git(repo, "fetch", "--no-tags", canonical, f"refs/heads/{BRANCH}", network=True)
    changed = git_helpers.git(repo, "diff", "--name-only", f"origin/{branch}...FETCH_HEAD").splitlines()
    if sorted(changed) != sorted(changed_files):
        raise SetupError("The existing setup branch has other changes; it will not be reused.")
    for path, content in files.items():
        if git_helpers.git(repo, "rev-parse", f"FETCH_HEAD:{path}").strip() != committed_blob(repo, path, content):
            raise SetupError("The existing setup branch has a different configuration; it will not be replaced.")
    return {"pr_url": prs[0]["url"] if prs else None}


def initialize_history(repo: Path, name: str) -> None:
    with tempfile.TemporaryDirectory(prefix="skill-report-history-") as temporary:
        work = Path(temporary)
        git_helpers.git(work, "init", "-b", "gh-pages")
        for key in ("user.name", "user.email"):
            value = git_helpers.git(repo, "config", "--get", key).strip()
            git_helpers.git(work, "config", "--local", key, value)
        (work / ".nojekyll").write_text("")
        (work / "history.json").write_text("[]\n")
        (work / MARKER).write_text(json.dumps({"purpose": "skillevaluator-reports", "repository": name}) + "\n")
        git_helpers.git(work, "add", "--", ".nojekyll", "history.json", MARKER)
        git_helpers.git(
            work,
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--signoff",
            "-m",
            "ci: initialize SkillEvaluator report history",
        )
        git_helpers.git(work, "push", f"https://github.com/{name}.git", "HEAD:refs/heads/gh-pages", network=True)


def ensure_environment(name: str, branch: str) -> None:
    endpoint = f"repos/{name}/environments/github-pages"
    env = github_api("GET", endpoint, missing_ok=True)
    if env:
        for rule in env.get("protection_rules", []):
            if rule.get("type") in {"required_reviewers", "custom_deployment_protection_rule"} or rule.get(
                "wait_timer", 0
            ):
                raise SetupError(
                    "The Pages environment requires approval. Existing protection was preserved; configure it manually."
                )
        policy = env.get("deployment_branch_policy")
        if policy and policy.get("protected_branches"):
            if not github_api("GET", f"repos/{name}/branches/{quote(branch, safe='')}").get("protected"):
                raise SetupError("The Pages environment allows protected branches only; its restriction was preserved.")
            return
    else:
        policy = None
    if not policy:
        github_api(
            "PUT", endpoint, {"deployment_branch_policy": {"protected_branches": False, "custom_branch_policies": True}}
        )
    rules = github_api("GET", endpoint + "/deployment-branch-policies")
    if not any(
        item.get("name") == branch and item.get("type", "branch") == "branch"
        for item in rules.get("branch_policies", [])
    ):
        github_api("POST", endpoint + "/deployment-branch-policies", {"name": branch, "type": "branch"})


def enable_hosting(repo: Path, name: str, branch: str, state: dict, progress: list[str]) -> None:
    private_url = state.get("verified_private_url")
    if private_url is not None or ("info" in state and pages_helpers.is_nonpublic(state["info"])):
        private_url = pages_helpers.private_pages_url(name, state.get("info", {}), state["pages"])
        current = github_api("GET", f"repos/{name}/pages", missing_ok=True)
        if pages_helpers.private_pages_url(name, state["info"], current) != private_url:
            raise SetupError("The private Pages URL changed after preflight; rerun setup before publishing.")
    expected_url = report_url(name, verified_private_url=private_url)
    if state.get("report_url", expected_url) != expected_url:
        raise SetupError("Pages preflight does not match the requested report URL; rerun setup.")
    if not state["actions"].get("enabled"):
        github_api("PUT", f"repos/{name}/actions/permissions", {"enabled": True, "allowed_actions": "all"})
        progress.append("GitHub Actions enabled")
    # Creating gh-pages first can auto-enable GitHub's legacy publishing source.
    # Enable the intended workflow source before that branch exists.
    if state["pages"] is None:
        github_api("POST", f"repos/{name}/pages", {"build_type": "workflow"})
        progress.append("GitHub Pages enabled with Actions source")
    elif state["pages"].get("build_type") == "legacy":
        # Preflight requires SkillEvaluator's ownership marker, so an unrelated
        # pre-existing website is never converted or taken over here.
        github_api("PUT", f"repos/{name}/pages", {"build_type": "workflow"})
        progress.append("Owned report site switched to Actions source")
    pages = github_api("GET", f"repos/{name}/pages")
    if private_url is not None and pages_helpers.private_pages_url(name, state["info"], pages) != private_url:
        raise SetupError("The private Pages URL changed after preflight; rerun setup before publishing.")
    if (
        pages.get("build_type") != "workflow"
        or pages.get("cname")
        or not isinstance(pages.get("html_url"), str)
        or report_url(name, pages["html_url"], verified_private_url=private_url) != expected_url
    ):
        raise SetupError("Pages readback does not match the requested workflow report site.")
    if not state["history_exists"]:
        if private_url is not None:
            deployments = github_api("GET", f"repos/{name}/deployments?per_page=1")
            pages_helpers.check_site(name, pages, history_exists=False, deployments=deployments)
        initialize_history(repo, name)
        progress.append("Report history branch initialized")
    ensure_environment(name, branch)
    progress.append("Pages deployment allowed from the default branch")


def component_preflight(existing: workflow_helpers.ExistingScan | None) -> None:
    """Use the operator's login to check the shared component once, before writes."""
    if not existing or not existing.delegated_wrapper:
        return
    match = workflow_helpers.WORKFLOW.fullmatch(existing.uses)
    repository, path, ref = match["repository"], match["path"], match["ref"]
    if ref.startswith("v"):
        tag = github_api("GET", f"repos/{repository}/git/ref/tags/{quote(ref, safe='')}")
        obj = tag.get("object", {})
        for _ in range(5):
            if obj.get("type") != "tag":
                break
            obj = github_api("GET", f"repos/{repository}/git/tags/{obj['sha']}").get("object", {})
        if obj.get("type") != "commit" or not re.fullmatch(r"[0-9a-f]{40}", str(obj.get("sha", ""))):
            raise SetupError("The component version must resolve to a Git commit tag.")
        ref = obj["sha"]
    value = github_api("GET", f"repos/{repository}/contents/{path}?ref={ref}")
    data = workflow_helpers.parse_workflow(decode_content(value).decode("utf-8"))
    jobs = data.get("jobs")
    if not isinstance(jobs, dict) or len(jobs) != 1:
        raise SetupError("Automatic reporting currently supports a component with one direct evaluator job.")
    job_id, job = next(iter(jobs.items()))
    if (
        not isinstance(job, dict)
        or not re.fullmatch(re.escape(workflow_helpers.EVALUATOR) + r"@[0-9a-f]{40}", str(job.get("uses", "")))
        or "strategy" in job
    ):
        raise SetupError("The shared component must call one SHA-pinned NVIDIA SkillEvaluator workflow.")
    if not workflow_helpers.identity_allowed(data, job):
        raise SetupError(
            f"The shared component needs one central update: grant 'id-token: write' alongside 'contents: read' "
            f"on job '{job_id}' in {repository}/{path}, then update its approved version and rerun this command. "
            "This lets scans sign their report identity. No repository settings were changed."
        )
    if job.get("with", {}).get("artifact-name", "skill-evaluator-tier1") != "skill-evaluator-tier1":
        raise SetupError("The shared component must keep artifact-name: skill-evaluator-tier1 for reporting.")


def create_setup(
    repo: Path, name: str, branch: str, files: dict[str, str], full: bool, plan: dict, *, expected=None
) -> str | None:
    git_helpers.require_identity(repo)
    state = remote_preflight(repo, name, branch, full)
    if full and state.get("report_url") != plan["report_url"]:
        raise SetupError("The report site changed after preview. Rerun setup to review its current URL.")
    if full and expected and SCAN in expected:
        component_preflight(workflow_helpers.existing_scan(expected[SCAN]))
    existing = branch_preflight(repo, name, branch, files, plan["changed_files"]) if plan["changed_files"] else None
    progress = plan.setdefault("completed_steps", [])
    if full:
        enable_hosting(repo, name, branch, state, progress)
    elif not state["actions"].get("enabled"):
        github_api("PUT", f"repos/{name}/actions/permissions", {"enabled": True, "allowed_actions": "all"})
        progress.append("GitHub Actions enabled")
    if not plan["changed_files"]:
        plan["action"] = "configured"
        return None
    if existing and existing["pr_url"]:
        plan["action"] = "existing-pr"
        return existing["pr_url"]
    canonical = f"https://github.com/{name}.git"
    if not existing:
        temporary = Path(tempfile.mkdtemp(prefix="skill-onboarding-"))
        work = temporary / "checkout"
        added = False
        try:
            git_helpers.git(repo, "worktree", "add", "--detach", str(work), f"origin/{branch}")
            added = True
            save_files(work, files, expected)
            git_helpers.git(work, "add", "--", *files)
            git_helpers.git(
                work,
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--signoff",
                "-m",
                "ci: onboard SkillEvaluator checks and reports",
            )
            git_helpers.git(work, "push", canonical, f"HEAD:refs/heads/{BRANCH}", network=True)
            progress.append("Setup branch pushed")
        finally:
            if added:
                git_helpers.git(repo, "worktree", "remove", str(work), optional=True)
            if not work.exists():
                temporary.rmdir()
    with tempfile.TemporaryDirectory(prefix="skill-setup-pr-") as temporary:
        body = Path(temporary) / "body.md"
        outputs = "downloadable HTML reports" if not full else "hosted HTML reports and one updated PR comment"
        body.write_text(
            f"Bring SkillEvaluator Tier 1 checks, including SkillSpector, to this repository with {outputs}.\n\n"
            f"Skill folders: `{', '.join(plan['selected_roots'])}`. Minimum score: **{plan['min_score']}**. "
            f"PR scan selection: **{plan['scan_mode']}**. No model key is used.\n\n"
            "**After merging:** open a new PR changing a selected skill. The check runs automatically. "
            + (
                "The publisher then deploys the HTML and posts or updates its PR comment.\n\n"
                if full
                else "Download the HTML from Actions.\n\n"
            )
            + "Generated by SkillEvaluator CI onboarding. "
            + (
                f"The approved shared component `{plan['component']}` maintains evaluator versions and policy. "
                if plan.get("component")
                else "Evaluator workflow and image versions are pinned. "
            )
            + "This setup PR is not an evaluation pass. Existing workflows and merge rules are preserved.\n",
        )
        url = git_helpers.run(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                name,
                "--base",
                branch,
                "--head",
                BRANCH,
                "--title",
                "Set up SkillEvaluator checks and reports",
                "--body-file",
                str(body),
            ]
        ).strip()
    if not re.fullmatch(r"https://github\.com/" + re.escape(name) + r"/pull/[1-9][0-9]*", url):
        raise SetupError("The setup branch was pushed, but the PR URL needs manual inspection.")
    plan["action"] = "created-pr"
    return url


def select_roots(roots: list[dict], explicit: tuple[str, ...], no_input: bool) -> list[str]:
    suggested = [item for item in roots if item["default"]]
    defaults = [str(roots.index(item) + 1) for item in suggested]
    known = {item["root"]: item["count"] for item in roots}
    if explicit:
        selected = list(dict.fromkeys(git_helpers.safe_relative(value) for value in explicit))
    elif no_input:
        selected = [item["root"] for item in suggested]
    else:
        while True:
            answer = click.prompt("Folders", default=",".join(defaults) or None).strip()
            if not re.fullmatch(r"\d+(?:(?:\s*,\s*|\s+)\d+)*", answer):
                console.print("Use listed numbers: 1,2 or 1 2.", style="yellow")
                continue
            try:
                indices = list(dict.fromkeys(int(value) for value in re.split(r"[,\s]+", answer)))
            except ValueError:
                console.print("Use the short folder numbers shown above.", style="yellow")
                continue
            if any(i < 1 or i > len(roots) for i in indices):
                console.print(f"Choose folder numbers from 1 to {len(roots)}.", style="yellow")
                continue
            selected = [roots[i - 1]["root"] for i in indices]
            if sum(known[value] for value in selected) > 100:
                console.print("Choose fewer folders: 100 skills maximum.", style="yellow")
                continue
            break
    if not selected or any(value not in known for value in selected):
        raise SetupError("Select at least one displayed skill folder.")
    if sum(known[value] for value in selected) > 100:
        raise SetupError("Select at most 100 skills for one CI workflow.")
    return selected


@click.group()
def ci():
    """Set up checks, HTML reports and PR comments."""


ALL_TIER_ONLY_OPTIONS = (
    "platform",
    "env_mode",
    "provider",
    "embedding_provider",
    "model",
    "embedding_model",
    "agents",
    "runner",
    "n_attempts",
    "n_concurrent",
    "timeout_minutes",
    "environment",
)


@ci.command("init")
@click.option("--tiers", default=None, help="Opt in with 1,2,3; requires a compatible candidate/release descriptor.")
@click.option("--platform", type=click.Choice(["auto", "github", "gitlab"]), default="auto")
@click.option("--env-mode", type=click.Choice(["docker", "local"]), default="docker")
@click.option("--provider", type=click.Choice(["openai", "anthropic", "nv_build"]), default="openai")
@click.option("--embedding-provider", type=click.Choice(["openai", "nv_build"]), default=None)
@click.option("--model", default=None)
@click.option("--embedding-model", default=None)
@click.option("--agents", default=None)
@click.option("--runner", default=None, help="GitHub runner label or GitLab runner tag.")
@click.option(
    "--tier3-required", is_flag=True, help="Deprecated and ignored; Tier 2 and Tier 3 CI results are advisory."
)
@click.option("--n-attempts", type=click.IntRange(1, 10), default=1)
@click.option("--n-concurrent", type=click.IntRange(1, 16), default=1)
@click.option("--timeout-minutes", type=click.IntRange(1, 360), default=300)
@click.option("--environment", default="skillevaluator")
@click.option("--repo", type=click.Path(path_type=Path, exists=True, file_okay=False), default=Path.cwd)
@click.option(
    "--release",
    "release_path",
    default=None,
    help="Pinned CI release descriptor; defaults to the bundled release.",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option("--paths", multiple=True)
@click.option("--min-score", type=click.IntRange(0, 100), default=None)
@click.option("--scan-mode", type=click.Choice(["auto", "all"]), default=None)
@click.option("--default-branch", default=None)
@click.option("--full/--checks-only", default=None, help="Full setup includes hosted reports and PR comments.")
@click.option("--report-url", "custom_url", default=None)
@click.option("--mode", type=click.Choice(["preview", "save", "pr"]), default=None)
@click.option(
    "--no-input", is_flag=True, help="Use defaults; requires --mode. --mode pr authorizes the displayed setup."
)
@click.option("--show-files", is_flag=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional new receipt outside the consumer checkout.",
)
def init(
    tiers,
    platform,
    env_mode,
    provider,
    embedding_provider,
    model,
    embedding_model,
    agents,
    runner,
    tier3_required,
    n_attempts,
    n_concurrent,
    timeout_minutes,
    environment,
    repo,
    release_path,
    paths,
    min_score,
    scan_mode,
    default_branch,
    full,
    custom_url,
    mode,
    no_input,
    show_files,
    output,
):
    plan = None
    if tier3_required:
        console.print(
            "Warning: --tier3-required is deprecated and ignored; Tier 1 is the only blocking CI tier.",
            style="yellow",
        )
    try:
        if tiers is not None:
            return all_tier_helpers.initialize(
                sys.modules[__name__],
                repo=repo,
                release_path=release_path,
                paths=paths,
                min_score=min_score,
                scan_mode=scan_mode,
                default_branch=default_branch,
                full=full,
                custom_url=custom_url,
                mode=mode,
                no_input=no_input,
                show_files=show_files,
                output=output,
                tiers=tiers,
                platform=platform,
                env_mode=env_mode,
                provider=provider,
                embedding_provider=embedding_provider,
                model=model,
                embedding_model=embedding_model,
                agents=agents,
                runner=runner,
                tier3_required=False,
                n_attempts=n_attempts,
                n_concurrent=n_concurrent,
                timeout_minutes=timeout_minutes,
                environment=environment,
            )
        context = click.get_current_context()
        ignored = [
            "--" + parameter.replace("_", "-")
            for parameter in ALL_TIER_ONLY_OPTIONS
            if context.get_parameter_source(parameter) not in (None, click.core.ParameterSource.DEFAULT)
        ]
        if ignored:
            raise SetupError(f"{', '.join(ignored)} applies only to all-tier onboarding; add --tiers 1,2,3.")
        repo = git_helpers.repository_root(repo)
        if output:
            output = output.resolve()
            if output.exists() or output.is_relative_to(repo):
                raise SetupError("Choose a new receipt path outside the consumer checkout.")
        if no_input and mode is None:
            raise SetupError("--no-input requires an explicit --mode.")
        release = load_release(release_path or Path(__file__).with_name("ci-release.json"))
        if release["schema_version"] == 2:
            raise SetupError("Use --tiers 1,2,3 with an all-tier release descriptor.")
        name = git_helpers.parse_remote(git_helpers.git(repo, "remote", "get-url", "origin").strip())
        branch = git_helpers.infer_branch(repo, default_branch)
        roots = git_helpers.discover(repo)
        git_helpers.no_symlinks(repo, SCAN)
        existing = (
            workflow_helpers.existing_scan((repo / SCAN).read_bytes().decode("utf-8"))
            if (repo / SCAN).exists()
            else None
        )
        expected = {SCAN: existing.text} if existing else None
        console.print("Bring SkillEvaluator to your repository", style="bold green")
        console.print("GitHub | Tier 1 + SkillSpector | no model key", style="dim")
        console.print(f"{name}  |  {branch}", markup=False)
        table = Table("#", "Skill folder", "Skills", "Suggested")
        for i, item in enumerate(roots, 1):
            suggested = item["default"]
            table.add_row(str(i), Text(item["root"]), str(item["count"]), "Yes" if suggested else "Optional")
        selection_hint = "Use folder numbers: 1 or 1,2. Spaces work too."
        if len(roots) == 1:
            selection_hint = "Use folder number 1."
        if any(item["default"] for item in roots):
            selection_hint += " Enter accepts the default."
        else:
            selection_hint += "\nNo default: enter a number to continue."
        if existing:
            question(
                1,
                "Keep your existing skill checks",
                Text(existing.uses),
                "Your folders, thresholds and triggers stay as configured.",
            )
            if paths or min_score is not None or scan_mode is not None:
                raise SetupError(
                    "Edit the existing scan workflow to change its policy; onboarding preserves that configuration."
                )
            selected = existing.roots
            manifests = git_helpers.git(repo, "ls-files", "-z", "--cached").split("\0")
            selected_count = sum(
                PurePosixPath(path).name == "SKILL.md"
                and any(root == "." or path.startswith(root + "/") for root in selected)
                for path in manifests
                if path
            )
            if not 1 <= selected_count <= 100:
                raise SetupError("The existing scan must select between 1 and 100 tracked skills.")
        else:
            question(1, "Which skill folders?", table, selection_hint)
            selected = select_roots(roots, paths, no_input)
            selected_count = sum(item["count"] for item in roots if item["root"] in selected)
        console.print(f"{symbol('✓', '[OK]')} Selected: {', '.join(selected)} ({selected_count} skills)", markup=False)
        question(2, "Minimum quality score?", "Fail below this score (0-100). Other required checks also apply.")
        score = (
            existing.score
            if existing
            else min_score
            if existing or min_score is not None
            else (70 if no_input else click.prompt("Score", default=70, type=click.IntRange(0, 100)))
        )
        question(
            3,
            "What should PRs check?",
            "auto = changed skills (recommended)\nall  = every skill in your selected folders",
        )
        scan = (
            existing.scan_mode
            if existing
            else scan_mode
            if existing or scan_mode is not None
            else ("auto" if no_input else click.prompt("PR scope", default="auto", type=click.Choice(["auto", "all"])))
        )
        question(
            4,
            "Add hosted reports and PR comments?",
            "Always included: checks + downloadable HTML.\nOptional: hosted HTML + one updated PR comment. Private repos require private Pages.",
        )
        if full is None:
            full = True if no_input else click.confirm("Add hosted feedback?", default=True)
        # Private site URLs are bound to an authenticated Pages readback in PR
        # mode below. Offline preview/save never treats a typed URL as verified.
        initial_url = None if mode in {None, "pr"} else custom_url
        url = report_url(name, initial_url) if full else None
        files = render_files(release, selected, score, scan, branch, name, full, url, existing=existing)
        files = bind_caller_approval(repo, files)
        # A private site's actual URL is discovered in PR mode. Do not compare
        # its existing publisher with a provisional public URL first.
        changed = check_files(repo, files, expected) if mode in {"preview", "save"} else list(files)
        plan = {
            "repository": name,
            "default_branch": branch,
            "selected_roots": selected,
            "min_score": score,
            "scan_mode": scan,
            "full": full,
            "report_url": url,
            "changed_files": changed,
            "scan_blob": caller_approval(files),
            "workflow_ref": existing.evaluator_sha if existing else release["workflow_ref"],
            "publisher_ref": release["publisher_ref"] if full else None,
            "image": existing.data["jobs"][existing.job]["with"].get("image") if existing else release["image"],
            "scan_workflow": existing.uses if existing else f"{workflow_helpers.EVALUATOR}@{release['workflow_ref']}",
            "component": existing.delegated_wrapper if existing else None,
            "completed_steps": [],
        }
        summary = Table.grid(padding=(0, 2))
        summary.add_row("Skills", Text(f"{selected_count} in {', '.join(selected)}"))
        summary.add_row("Quality gate", f"{score}/100")
        summary.add_row("PR checks", "Changed skills" if scan == "auto" else "All selected skills")
        if full:
            summary.add_row("Feedback", "Hosted HTML + updated PR comment")
            summary.add_row("Setup", "Keep scan + add publisher" if existing else "2 workflow files + GitHub Pages")
            if existing and existing.delegated_wrapper:
                summary.add_row("Version policy", "Trust the approved component to choose evaluator versions")
        else:
            summary.add_row("Feedback", "Check + downloadable HTML")
            summary.add_row("Setup", "1 workflow file")
        question(
            5, "Ready to set up?", summary, "pr = set up GitHub + open PR | preview = no changes | save = files only"
        )
        if not changed:
            console.print("Workflow files already match.")
        if mode is None:
            mode = click.prompt("Next", default="pr", type=click.Choice(["preview", "save", "pr"]))
        plan["requested_action"] = mode
        if mode in {"preview", "save"}:
            if full:
                report_url(name, custom_url)
            changed = check_files(repo, files, expected)
            plan["changed_files"] = changed
        if mode == "preview":
            plan["action"] = "preview"
            console.print("No changes applied. Preview needs no login; PR mode verifies access and report visibility.")
        elif mode == "save":
            if not no_input and changed and not click.confirm("Save these workflow files locally?", default=False):
                plan["action"] = "cancelled"
                console.print("Cancelled. No changes applied.")
            else:
                save_files(repo, files, expected)
                plan["action"] = "saved" if changed else "unchanged"
                console.print("Files saved. Remote settings were not changed. Use PR mode for automatic GitHub setup.")
        else:
            if not no_input and not click.confirm("Configure GitHub and open the setup PR?", default=False):
                plan["action"] = "cancelled"
                console.print("Cancelled. No changes applied.")
                return None
            console.print("Checking GitHub access…")
            state = remote_preflight(repo, name, branch, full)
            if full:
                url = report_url(
                    name, custom_url or state["report_url"], verified_private_url=state["verified_private_url"]
                )
                files = render_files(
                    release,
                    selected,
                    score,
                    scan,
                    branch,
                    name,
                    full,
                    url,
                    existing=existing,
                    verified_private_url=state["verified_private_url"],
                )
                files = bind_caller_approval(repo, files)
                plan.update(report_url=url, scan_blob=caller_approval(files))
                console.print("Private report site:" if state["verified_private_url"] else "Public report site:")
                print_link(url)
            plan["changed_files"] = check_files(repo, files, expected)
            url_pr = create_setup(repo, name, branch, files, full, plan, expected=expected)
            if url_pr:
                plan["pr_url"] = url_pr
                show_next_steps(name, full, url_pr, repo=repo)
            else:
                show_next_steps(name, full, repo=repo)
        if show_files:
            for path, content in files.items():
                console.print(f"\n{path}\n{content}", markup=False, soft_wrap=True)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                json.dump(plan, stream, indent=2)
                stream.write("\n")
    except (SetupError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        if plan and plan.get("completed_steps"):
            console.print("Completed before the interruption: " + "; ".join(plan["completed_steps"]), markup=False)
            console.print("Rerun the same setup to resume; existing files/history will be preserved.")
        raise click.ClickException(str(exc)) from exc


@ci.command("status")
@click.option("--platform", type=click.Choice(["auto", "github", "gitlab"]), default="auto")
@click.option("--tiers", default=None)
@click.option("--repo", type=click.Path(path_type=Path, exists=True, file_okay=False), default=Path.cwd)
def status(repo, platform, tiers):
    """Read the live setup state without changing settings or starting scans."""
    try:
        repo = git_helpers.repository_root(repo)
        if all_tier_helpers.is_all_tier(repo, platform, tiers):
            return all_tier_helpers.inspect(sys.modules[__name__], repo, platform, tiers)
        name = git_helpers.parse_remote(git_helpers.git(repo, "remote", "get-url", "origin").strip())
        info = github_api("GET", f"repos/{name}")
        branch = info["default_branch"]
        pages = github_api("GET", f"repos/{name}/pages", missing_ok=True)
        scan = github_api("GET", f"repos/{name}/contents/{SCAN}?ref={quote(branch, safe='')}", missing_ok=True)
        publisher = github_api("GET", f"repos/{name}/contents/{PUBLISH}?ref={quote(branch, safe='')}", missing_ok=True)
        table = Table("Component", "Live state")
        table.add_row("Checks", "Configured on default branch" if scan else "Awaiting setup PR merge")
        table.add_row("Publisher", "Configured on default branch" if publisher else "Not enabled (optional)")
        table.add_row("Pages", pages["html_url"] if pages else "Not enabled")
        if scan and publisher:
            scan_bytes = decode_content(scan)
            blob = hashlib.sha1(f"blob {len(scan_bytes)}\0".encode() + scan_bytes).hexdigest()
            inputs = workflow_inputs(publisher, "reports")
            approved = inputs.get("trusted-caller-blobs")
            if not isinstance(approved, str):
                raise SetupError("The publisher workflow is missing its scan-file approval.")
            trusted = re.split(r"[,\s]+", approved)
            table.add_row(
                "Scan approval",
                "Matches exact workflow" if blob in trusted else "Needs refresh; publisher will reject it",
            )
        console.print(table)
        for filename, label in [
            ("skill-checks.yml", "Latest scan"),
            ("publish-skill-reports.yml", "Latest publication"),
        ]:
            value = github_api("GET", f"repos/{name}/actions/workflows/{filename}/runs?per_page=1", missing_ok=True)
            if value and value.get("workflow_runs"):
                item = value["workflow_runs"][0]
                console.print(f"{label}: {item.get('conclusion') or item['status']}\n{item['html_url']}", markup=False)
        console.print(
            "Configuration alone is not a verified scan. Open a skill PR to inspect its check, comment and report."
        )
    except (SetupError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise click.ClickException(str(exc)) from exc


@ci.command("demo")
@click.option("--platform", type=click.Choice(["auto", "github", "gitlab"]), default="auto")
@click.option("--tiers", default=None)
@click.option("--repo", type=click.Path(path_type=Path, exists=True, file_okay=False), default=Path.cwd)
@click.option("--no-input", is_flag=True, help="Authorize opening one small skill-change PR.")
def demo(repo, no_input, platform, tiers):
    """After merging setup, open a small skill PR to see automatic feedback."""
    try:
        console.print("Try your first skill check", style="bold green")
        repo = git_helpers.repository_root(repo)
        git_helpers.require_clean(repo)
        if all_tier_helpers.is_all_tier(repo, platform, tiers):
            return all_tier_helpers.demo(sys.modules[__name__], repo, platform, tiers, no_input)
        name = git_helpers.parse_remote(git_helpers.git(repo, "remote", "get-url", "origin").strip())
        info = github_api("GET", f"repos/{name}")
        permissions = info.get("permissions", {})
        if info.get("archived") or not (permissions.get("push") or permissions.get("admin")):
            raise SetupError("Your GitHub login needs write access to create a test PR.")
        branch = info["default_branch"]
        scan_file = github_api("GET", f"repos/{name}/contents/{SCAN}?ref={quote(branch, safe='')}", missing_ok=True)
        if scan_file is None:
            raise SetupError("Merge the setup PR first. The scan workflow is not on the default branch yet.")
        publisher_file = github_api(
            "GET", f"repos/{name}/contents/{PUBLISH}?ref={quote(branch, safe='')}", missing_ok=True
        )
        full = publisher_file is not None
        configuration = workflow_helpers.existing_scan(decode_content(scan_file).decode("utf-8"))
        inputs = configuration.data["jobs"][configuration.job]["with"]
        paths = inputs.get("paths")
        if not isinstance(paths, str):
            raise SetupError("The scan workflow is missing its configured skill paths.")
        roots = [git_helpers.safe_relative(value.strip()) for value in paths.splitlines() if value.strip()]
        if not roots:
            raise SetupError("No skill roots are configured on the default branch.")
        demo_branch = DEMO_BRANCH
        existing = json.loads(
            git_helpers.run(
                ["gh", "pr", "list", "--repo", name, "--head", demo_branch, "--state", "open", "--json", "url"]
            )
        )
        if existing:
            show_test_pr(name, existing[0]["url"], existing=True, full=full)
            return None
        canonical = f"https://github.com/{name}.git"
        if git_helpers.git(repo, "ls-remote", "--heads", canonical, f"refs/heads/{demo_branch}", network=True).strip():
            raise SetupError(
                "A previous demo branch exists without an open PR. Inspect it manually; it will not be replaced."
            )
        console.print(
            Panel(
                Text(
                    "Add a Markdown comment to one selected skill on a new branch.\nClose the test PR afterward; no merge needed."
                ),
                title="Create a test PR?",
                title_align="left",
                border_style="cyan",
                width=min(console.width, 92),
            )
        )
        if not no_input and not click.confirm("Create test PR?", default=False):
            console.print("Cancelled. No changes applied.")
            return None
        git_helpers.require_identity(repo)
        git_helpers.git(
            repo, "fetch", "--no-tags", canonical, f"refs/heads/{branch}:refs/remotes/origin/{branch}", network=True
        )
        temporary = Path(tempfile.mkdtemp(prefix="skill-feedback-demo-"))
        work = temporary / "checkout"
        added = False
        try:
            git_helpers.git(repo, "worktree", "add", "--detach", str(work), f"origin/{branch}")
            added = True
            manifests = git_helpers.git(work, "ls-files", "-z").split("\0")
            candidates = sorted(
                path
                for path in manifests
                if PurePosixPath(path).name == "SKILL.md"
                and any(root == "." or path.startswith(root + "/") for root in roots)
            )
            if not candidates:
                raise SetupError("No tracked skill manifest matches the configured roots.")
            target = candidates[0]
            git_helpers.no_symlinks(work, target)
            path = work / target
            if not path.is_file():
                raise SetupError("The demonstration manifest is missing.")
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write("\n<!-- SkillEvaluator onboarding demo: exercise automatic PR feedback. -->\n")
            git_helpers.git(work, "add", "--", target)
            git_helpers.git(
                work,
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--signoff",
                "-m",
                "test: try automatic SkillEvaluator PR feedback",
            )
            git_helpers.git(work, "push", canonical, f"HEAD:refs/heads/{demo_branch}", network=True)
            body = temporary / "body.md"
            feedback = (
                "Its bot comment links to the HTML report after publication."
                if full
                else "Download the HTML report from the Skill checks run."
            )
            body.write_text(
                f"Make a small Markdown-comment change to `{target}` to try SkillEvaluator.\n\nThe check evaluates this changed skill. {feedback}\n"
            )
            url = git_helpers.run(
                [
                    "gh",
                    "pr",
                    "create",
                    "--repo",
                    name,
                    "--base",
                    branch,
                    "--head",
                    demo_branch,
                    "--title",
                    "Try automatic SkillEvaluator PR feedback",
                    "--body-file",
                    str(body),
                ]
            ).strip()
            if not re.fullmatch(r"https://github\.com/" + re.escape(name) + r"/pull/[1-9][0-9]*", url):
                raise SetupError("The test branch was pushed, but its PR URL needs manual inspection.")
            show_test_pr(name, url, full=full)
        finally:
            if added:
                git_helpers.git(repo, "worktree", "remove", str(work), optional=True)
            if not work.exists():
                for child in temporary.iterdir():
                    if child.is_file():
                        child.unlink()
                temporary.rmdir()
    except (SetupError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise click.ClickException(str(exc)) from exc
