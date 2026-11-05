# SPDX-License-Identifier: Apache-2.0
"""Opt-in all-tier callers and host-aware, API-only consumer setup."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request

import click
import yaml

from skillevaluator import _ci_onboarding_git as git

SCAN = ".github/workflows/skill-checks-all-tiers.yml"
FEEDBACK = ".github/workflows/feedback-all-tiers.yml"
GL_SCAN = ".gitlab-ci.yml"
GL_FEEDBACK = ".gitlab/skill-feedback.yml"
SETUP_BRANCH = "ci/skillevaluator-all-tiers-setup"
DEMO_BRANCH = "ci/skillevaluator-all-tiers-test"
SHA = re.compile(r"[0-9a-f]{40}")


class _GitLabVariable(str):
    """Force GitLab's YAML parser to preserve a CI variable's string value."""

    __slots__ = ()


class _GitLabDumper(git.WorkflowDumper):
    pass


def _represent_gitlab_variable(dumper, value):
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style='"')


_GitLabDumper.add_representer(_GitLabVariable, _represent_gitlab_variable)


def _gitlab_yaml_text(document):
    # GitLab's Ruby parser reads an unquoted 1,2,3 as integer 123. Quote CI
    # variable scalars without changing typed include inputs or job settings.
    rendered = dict(document)
    if "variables" in rendered:
        rendered["variables"] = {
            key: _GitLabVariable(value) if isinstance(value, str) else value
            for key, value in rendered["variables"].items()
        }
    return yaml.dump(rendered, Dumper=_GitLabDumper, sort_keys=False, allow_unicode=True, width=1000)


def validate_release(data: dict) -> dict:
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 2
        or data.get("repository") != "NVIDIA/SkillEvaluator"
    ):
        raise git.SetupError("All-tier onboarding needs an explicit schema_version 2 CI release descriptor.")
    caps = data.get("capabilities")
    if not isinstance(caps, list) or not caps or any(c not in {"all-tiers-github", "all-tiers-gitlab"} for c in caps):
        raise git.SetupError("The all-tier release descriptor must declare supported capabilities.")
    for key in ("workflow_ref", "feedback_ref", "gitlab_ref", "tools_ref"):
        if not SHA.fullmatch(str(data.get(key, ""))):
            raise git.SetupError(f"{key} must be an immutable full commit SHA in the all-tier release.")
    if not re.fullmatch(r"ghcr\.io/nvidia/skillevaluator-tier1@sha256:[0-9a-f]{64}", str(data.get("image", ""))):
        raise git.SetupError("The Tier 1 image must use the official immutable digest.")
    return data


def validate_tiers(value: str) -> str:
    values = [part.strip() for part in value.split(",")]
    if values != ["1", "2", "3"]:
        raise git.SetupError("All-tier onboarding requires --tiers 1,2,3; omit --tiers for the released Tier 1 setup.")
    return "1,2,3"


def remote(repo: Path, platform: str = "auto") -> tuple[str, str, str]:
    value = git.git(repo, "remote", "get-url", "origin").strip()
    match = re.fullmatch(r"git@([A-Za-z0-9.-]+):(.+)", value)
    if match:
        host, name = match.groups()
        origin = "https://" + host
    else:
        parts = urlsplit(value)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise git.SetupError("Use a credential-free HTTPS or git@host SSH repository origin.")
        host = parts.hostname
        origin = f"https://{parts.netloc}"
        name = parts.path.lstrip("/")
    name = name.removesuffix(".git")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+", name) or any(
        p in {".", ".."} for p in name.split("/")
    ):
        raise git.SetupError("The origin does not identify a supported project path.")
    detected = "github" if host == "github.com" else "gitlab"
    selected = detected if platform == "auto" else platform
    if selected == "github" and host != "github.com":
        raise git.SetupError("GitHub onboarding currently supports github.com origins.")
    if selected == "gitlab" and host == "github.com":
        raise git.SetupError("The selected platform does not match the repository origin.")
    return selected, origin, name


def gitlab_api(origin: str, method: str, endpoint: str, payload=None, *, missing_ok=False):
    """Use a process-scoped token and the origin host; never mutate glab auth."""
    api = os.environ.get("SKILLEVALUATOR_GITLAB_API_URL", origin + "/api/v4").rstrip("/")
    parsed, expected = urlsplit(api), urlsplit(origin)
    if (
        parsed.scheme != "https"
        or parsed.netloc != expected.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise git.SetupError("The GitLab API URL must use the HTTPS origin host without credentials.")
    token = os.environ.get("SKILLEVALUATOR_GITLAB_TOKEN") or os.environ.get("GITLAB_TOKEN")
    if not token:
        raise git.SetupError(
            "Set SKILLEVALUATOR_GITLAB_TOKEN for this GitLab host before using PR/MR or live status commands."
        )
    request = Request(
        api + "/" + endpoint.lstrip("/"),
        method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"PRIVATE-TOKEN": token, "Content-Type": "application/json"},
    )
    # Disallow redirects so a different host never receives the scoped credential.
    from urllib.request import HTTPRedirectHandler, build_opener

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, _req, _fp, _code, _msg, _headers, _newurl):
            return None

    try:
        with build_opener(NoRedirect).open(request, timeout=60) as response:
            body = response.read(8 * 1024 * 1024 + 1)
        if len(body) > 8 * 1024 * 1024:
            raise git.SetupError("The GitLab API response exceeded the size limit.")
        return json.loads(body)
    except HTTPError as exc:
        if missing_ok and exc.code == 404:
            return None
        raise git.SetupError(
            f"GitLab API request failed (HTTP {exc.code}); check host, token scope and project access."
        ) from None
    except (URLError, ValueError) as exc:
        raise git.SetupError("GitLab API request failed; check host, authentication and connectivity.") from exc


class Client:
    def __init__(self, wizard, platform, origin, name):
        self.wizard, self.platform, self.origin, self.name = wizard, platform, origin, name
        self.project = f"projects/{quote(name, safe='')}" if platform == "gitlab" else f"repos/{name}"

    def api(self, method, suffix="", payload=None, *, missing_ok=False):
        if self.platform == "github":
            return self.wizard.github_api(method, self.project + suffix, payload, missing_ok=missing_ok)
        return gitlab_api(self.origin, method, self.project + suffix, payload, missing_ok=missing_ok)

    def info(self):
        info = self.api("GET")
        if info.get("archived"):
            raise git.SetupError("Archived repositories cannot be onboarded.")
        return info

    def file(self, path, ref):
        suffix = (
            f"/contents/{path}?ref={quote(ref, safe='')}"
            if self.platform == "github"
            else f"/repository/files/{quote(path, safe='')}?ref={quote(ref, safe='')}"
        )
        value = self.api("GET", suffix, missing_ok=True)
        if value is None:
            return None
        if value.get("encoding") != "base64":
            raise git.SetupError("The remote CI configuration is not a supported text file.")
        try:
            return base64.b64decode(value["content"], validate=False).decode("utf-8")
        except (KeyError, ValueError, UnicodeError) as exc:
            raise git.SetupError("The remote CI configuration could not be decoded.") from exc

    def reviews(self, branch):
        query = (
            urlencode({"state": "open", "head": self.name.split("/")[0] + ":" + branch})
            if self.platform == "github"
            else urlencode({"state": "opened", "source_branch": branch})
        )
        return self.api("GET", ("/pulls?" if self.platform == "github" else "/merge_requests?") + query)

    def create(self, branch, base, files, title, body, *, setup=False):
        existing = self.reviews(branch)
        branch_suffix = "/git/ref/heads/" if self.platform == "github" else "/repository/branches/"
        ref = self.api("GET", branch_suffix + quote(branch, safe=""), missing_ok=True)
        if ref is not None:
            if self.platform == "github":
                comparison = self.api("GET", "/compare/" + quote(base, safe="") + "..." + quote(branch, safe=""))
                changed_paths = [item["filename"] for item in comparison.get("files", [])]
                complete = len(changed_paths) < 300
            else:
                comparison = self.api("GET", "/repository/compare?" + urlencode({"from": base, "to": branch}))
                changed_paths = [item["new_path"] for item in comparison.get("diffs", [])]
                complete = not comparison.get("compare_timeout") and not comparison.get("overflow")
            if not complete or sorted(changed_paths) != sorted(files):
                raise git.SetupError("The existing onboarding branch has other changes; it will not be reused.")
            if any(self.file(path, branch) != content for path, content in files.items()):
                raise git.SetupError("The existing onboarding branch has different content; it will not be replaced.")
            if existing:
                return existing[0].get("html_url") or existing[0]["web_url"]
            raise git.SetupError("An onboarding branch already exists without an open PR/MR; inspect it manually.")
        if existing:
            raise git.SetupError("An open PR/MR exists without its expected branch; inspect it manually.")
        if self.platform == "gitlab":
            actions = [
                {"action": "create" if setup else "update", "file_path": p, "content": c} for p, c in files.items()
            ]
            self.api(
                "POST",
                "/repository/commits",
                {"branch": branch, "start_branch": base, "commit_message": title, "actions": actions},
            )
            review = self.api(
                "POST",
                "/merge_requests",
                {
                    "source_branch": branch,
                    "target_branch": base,
                    "title": title,
                    "description": body,
                    "remove_source_branch": True,
                },
            )
        else:
            base_ref = self.api("GET", "/git/ref/heads/" + quote(base, safe=""))
            parent = base_ref["object"]["sha"]
            commit = self.api("GET", "/git/commits/" + parent)
            tree = self.api(
                "POST",
                "/git/trees",
                {
                    "base_tree": commit["tree"]["sha"],
                    "tree": [{"path": p, "mode": "100644", "type": "blob", "content": c} for p, c in files.items()],
                },
            )
            created = self.api("POST", "/git/commits", {"message": title, "tree": tree["sha"], "parents": [parent]})
            self.api("POST", "/git/refs", {"ref": "refs/heads/" + branch, "sha": created["sha"]})
            review = self.api("POST", "/pulls", {"base": base, "head": branch, "title": title, "body": body})
        url = review.get("html_url") or review.get("web_url")
        parsed = urlsplit(url or "")
        if (
            parsed.scheme != "https"
            or parsed.netloc != urlsplit(self.origin).netloc
            or not parsed.path.startswith("/" + self.name + "/")
        ):
            raise git.SetupError("The branch was created, but its PR/MR URL needs manual inspection.")
        return url


def render(wizard, repo, release, platform, branch, roots, options):
    common = {
        "tiers": "1,2,3",
        "paths": "\n".join(roots),
        "min-score": options["min_score"],
        "scan-mode": options["scan_mode"],
        "env-mode": options["env_mode"],
        "provider": options["provider"],
        "embedding-provider": options["embedding_provider"],
        "model": options["model"] or "",
        "embedding-model": options["embedding_model"] or "",
        "agents": options["agents"] or "",
        "runner": options["runner"],
        "tools-repository": release["repository"],
        "tools-ref": release["tools_ref"],
        "image": release["image"],
        "tier3-required": options["tier3_required"],
        "n-attempts": options["n_attempts"],
        "n-concurrent": options["n_concurrent"],
        "timeout-minutes": options["timeout_minutes"],
        "environment": options["environment"],
    }
    header = "# Generated by SkillEvaluator all-tier CI onboarding.\n"
    if platform == "github":
        scan = {
            "name": "Skill checks (all tiers)",
            "on": {"pull_request": None, "push": {"branches": [branch]}, "workflow_dispatch": None},
            "permissions": {"contents": "read", "id-token": "write", "pull-requests": "read", "actions": "read"},
            "jobs": {
                "skills": {
                    "uses": f"{release['repository']}/.github/workflows/evaluate-all-tiers.yml@{release['workflow_ref']}",
                    "with": common,
                    "secrets": {
                        key: "${{ secrets." + key + " }}"
                        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NVIDIA_API_KEY")
                    },
                }
            },
        }
        scan_text = header + wizard.yaml_text(scan)
        feedback = {
            "name": "Skill feedback (all tiers)",
            "on": {"workflow_run": {"workflows": ["Skill checks (all tiers)"], "types": ["completed"]}},
            "permissions": {"actions": "read", "contents": "read", "pull-requests": "write"},
            "jobs": {
                "feedback": {
                    "uses": f"{release['repository']}/.github/workflows/feedback-all-tiers.yml@{release['feedback_ref']}",
                    "with": {
                        "tools-repository": release["repository"],
                        "tools-ref": release["tools_ref"],
                        "trusted-workflow-ref": f"{release['repository']}/.github/workflows/evaluate-all-tiers.yml@{release['workflow_ref']}",
                        "trusted-caller-blobs": wizard.committed_blob(repo, SCAN, scan_text),
                    },
                }
            },
        }
        return {SCAN: scan_text, FEEDBACK: header + wizard.yaml_text(feedback)}
    variables = {
        "SKILL_EVAL_" + key.upper().replace("-", "_"): str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in common.items()
        if key not in {"paths", "image", "tools-repository", "tools-ref", "environment", "runner", "timeout-minutes"}
    }
    variables.update(
        {
            "SKILL_EVAL_PATHS": "\n".join(roots),
            "SKILL_EVAL_IMAGE": release["image"],
            "SKILL_EVAL_TIMEOUT_SECONDS": str(options["timeout_minutes"] * 60),
            "SKILL_EVAL_TOOLS_REPOSITORY": release["repository"],
            "SKILL_EVAL_TOOLS_REF": release["tools_ref"],
            "SKILL_EVAL_ENVIRONMENT": options["environment"],
            "SKILL_EVAL_RUNNER": options["runner"],
        }
    )
    upstream = f"https://raw.githubusercontent.com/{release['repository']}/{release['gitlab_ref']}/ci/gitlab/"
    scan = {
        "include": [
            {"remote": upstream + "all-tiers.yml"},
            {
                "local": GL_FEEDBACK,
                "rules": [
                    {
                        "if": '$CI_PIPELINE_SOURCE == "pipeline" && $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH && $SKILL_EVAL_FEEDBACK == "true"'
                    }
                ],
            },
        ],
        "variables": variables,
        "workflow": {
            "rules": [
                {
                    "if": '$CI_PIPELINE_SOURCE == "pipeline" && $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH && $SKILL_EVAL_FEEDBACK == "true"'
                },
                {"if": '$SKILL_EVAL_FEEDBACK == "true"', "when": "never"},
                {"if": '$CI_PIPELINE_SOURCE == "merge_request_event"'},
                {"if": "$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH"},
                {"if": '$CI_PIPELINE_SOURCE == "web"'},
                {"when": "never"},
            ]
        },
        "skill-evaluation": {
            "rules": [
                {"if": '$SKILL_EVAL_FEEDBACK == "true"', "when": "never"},
                {"if": '$CI_PIPELINE_SOURCE == "merge_request_event"'},
                {"if": "$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH"},
                {"if": '$CI_PIPELINE_SOURCE == "web"'},
            ]
        },
    }
    # GitLab 17.8 validates job timeout before expanding CI variables.
    scan["skill-evaluation-native"] = {"timeout": str(options["timeout_minutes"]) + " minutes"}
    if options["env_mode"] == "local":
        scan["skill-evaluation-native"]["services"] = []
    _, origin, project = remote(repo, "gitlab")
    feedback_include = {
        "remote": upstream + "feedback.yml",
        "inputs": {
            "tools-ref": release["tools_ref"],
            "tools-repository": release["repository"],
            "api-url": origin + "/api/v4",
            "project-id": project,
        },
    }
    return {
        GL_SCAN: header + _gitlab_yaml_text(scan),
        GL_FEEDBACK: header + _gitlab_yaml_text({"include": [feedback_include]}),
    }


def initialize(
    wizard,
    *,
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
    tiers,
    platform,
    **options,
):
    validate_tiers(tiers)
    repo = git.repository_root(repo)
    if no_input and mode is None:
        raise git.SetupError("--no-input requires an explicit --mode.")
    if custom_url:
        raise git.SetupError(
            "All-tier onboarding currently links private CI artifacts; --report-url applies to the legacy Tier 1 setup."
        )
    if full is False:
        raise git.SetupError("All-tier onboarding includes trusted PR/MR feedback. Omit --checks-only.")
    descriptor = release_path or Path(wizard.__file__).with_name("ci-release-all-tiers.json")
    if not descriptor.is_file():
        raise git.SetupError(
            "This installation has no packaged all-tier release descriptor; supply a reviewed candidate using --release."
        )
    release = validate_release(json.loads(descriptor.read_text()))
    platform, origin, name = remote(repo, platform)
    if "all-tiers-" + platform not in release["capabilities"]:
        raise git.SetupError(f"The all-tier release does not declare {platform} support.")
    branch = git.infer_branch(repo, default_branch)
    roots = git.discover(repo)
    for index, item in enumerate(roots, 1):
        wizard.console.print(f"{index}. {item['root']} ({item['count']} skills)", markup=False)
    selected = wizard.select_roots(roots, paths, no_input)
    options.update(min_score=70 if min_score is None else min_score, scan_mode=scan_mode or "auto")
    options["embedding_provider"] = options["embedding_provider"] or (
        "openai" if options["provider"] == "anthropic" else options["provider"]
    )
    options["runner"] = options["runner"] or ("ubuntu-24.04" if platform == "github" else "docker")
    files = render(wizard, repo, release, platform, branch, selected, options)
    changed = wizard.check_files(repo, files)
    if output:
        output = output.resolve()
        if output.exists() or output.is_relative_to(repo):
            raise git.SetupError("Choose a new receipt path outside the consumer checkout.")
    wizard.console.print(
        f"{platform.title()} | Tier 1 + Tier 2 required; Tier 3 "
        + ("required" if options["tier3_required"] else "advisory"),
        markup=False,
    )
    wizard.console.print(
        f"Environment: {options['env_mode']} | provider: {options['provider']} | runner: {options['runner']}",
        markup=False,
    )
    wizard.console.print(
        "Tier 2 needs chat and embedding credentials. Tier 3 needs committed native eval datasets and runs paired coverage."
    )
    wizard.console.print(
        "Store provider keys in CI secrets and authorize the protected "
        + options["environment"]
        + " environment before model execution.",
        markup=False,
    )
    if platform == "gitlab":
        wizard.console.print(
            "GitLab: protect the default branch and add a scoped protected SKILLEVALUATOR_GITLAB_TOKEN for trusted feedback. Enable job-token pipeline triggers for this project."
        )
    mode = mode or click.prompt("Next", default="preview", type=click.Choice(["preview", "save", "pr"]))
    plan = {
        "repository": name,
        "platform": platform,
        "default_branch": branch,
        "selected_roots": selected,
        "tiers": [1, 2, 3],
        "release": release,
        "options": options,
        "changed_files": changed,
        "requested_action": mode,
    }
    if mode == "preview":
        plan["action"] = "preview"
        wizard.console.print(
            "No changes applied. These are candidate pinned workflows; configuration is not a verified evaluation."
        )
    elif not no_input and not click.confirm(
        "Save files?" if mode == "save" else "Open the setup PR/MR?", default=False
    ):
        plan["action"] = "cancelled"
        wizard.console.print("Cancelled. No changes applied.")
    elif mode == "save":
        wizard.save_files(repo, files)
        plan["action"] = "saved" if changed else "unchanged"
        wizard.console.print("Files saved. Remote settings were not changed.")
    else:
        git.require_clean(repo)
        if platform == "github":
            # Git API commits bypass local clean filters; approve the uploaded bytes.
            feedback = yaml.safe_load(files[FEEDBACK])
            feedback["jobs"]["feedback"]["with"]["trusted-caller-blobs"] = wizard.blob_id(files[SCAN])
            files[FEEDBACK] = "# Generated by SkillEvaluator all-tier CI onboarding.\n" + wizard.yaml_text(feedback)
        client = Client(wizard, platform, origin, name)
        info = client.info()
        if info.get("default_branch") != branch:
            raise git.SetupError("The live default branch differs; rerun with --default-branch.")
        pending = {}
        for path, content in files.items():
            current = client.file(path, branch)
            if current is not None and current != content:
                raise git.SetupError(f"{path} differs on the default branch; it will not be replaced.")
            if current is None:
                pending[path] = content
        if pending:
            url = client.create(
                SETUP_BRANCH,
                branch,
                pending,
                "Set up SkillEvaluator all-tier checks and feedback",
                "Add pinned Tier 1 and Tier 2 required checks, advisory paired Tier 3 evaluation, and trusted feedback linking retained private CI artifacts.\n\nMerge this setup, configure provider credentials and the protected execution environment, then run `skillevaluator ci demo`. This setup is not an evaluation pass.",
                setup=True,
            )
            plan.update(action="created-pr", pr_url=url)
            wizard.console.print("Setup PR/MR ready. Merge it before running ci demo.")
            wizard.print_link(url)
        else:
            plan["action"] = "configured"
            wizard.console.print("All-tier files already match the default branch.")
    if show_files:
        for path, text in files.items():
            wizard.console.print(path, markup=False)
            wizard.console.print(text, markup=False, soft_wrap=True)
    if output:
        with output.open("x") as stream:
            json.dump(plan, stream, indent=2)
            stream.write("\n")
    return plan


def is_all_tier(repo, platform="auto", tiers=None):
    selected, _, _ = remote(repo, platform)
    return selected == "gitlab" or tiers is not None or (repo / SCAN).exists()


def inspect(wizard, repo, platform="auto", tiers=None):
    if tiers is not None:
        validate_tiers(tiers)
    platform, origin, name = remote(repo, platform)
    client = Client(wizard, platform, origin, name)
    info = client.info()
    branch = info["default_branch"]
    scan_path, feedback_path = (SCAN, FEEDBACK) if platform == "github" else (GL_SCAN, GL_FEEDBACK)
    scan, feedback = client.file(scan_path, branch), client.file(feedback_path, branch)
    wizard.console.print(f"{platform.title()} {name} | default branch: {branch}", markup=False)
    wizard.console.print("All-tier checks: " + ("configured" if scan else "awaiting setup PR/MR merge"))
    wizard.console.print("Trusted feedback: " + ("configured" if feedback else "missing"))
    if scan and feedback and platform == "github":
        inputs = yaml.safe_load(feedback).get("jobs", {}).get("feedback", {}).get("with", {})
        blob = wizard.blob_id(scan)
        wizard.console.print(
            "Scan approval: "
            + (
                "matches exact workflow"
                if blob in str(inputs.get("trusted-caller-blobs", "")).split()
                else "Needs refresh; feedback will reject it"
            )
        )
    suffix = (
        "/actions/workflows/skill-checks-all-tiers.yml/runs?per_page=1"
        if platform == "github"
        else "/pipelines?per_page=1"
    )
    runs = client.api("GET", suffix, missing_ok=True)
    records = runs.get("workflow_runs", []) if isinstance(runs, dict) else runs or []
    if records:
        latest = records[0]
        wizard.console.print("Latest pipeline: " + str(latest.get("conclusion") or latest.get("status")))
        wizard.print_link(latest.get("html_url") or latest.get("web_url"))
    wizard.console.print(
        "Configuration alone is not a verified scan. Inspect the demo check, comment and artifact reports."
    )


def demo(wizard, repo, platform="auto", tiers=None, no_input=False):
    if tiers is not None:
        validate_tiers(tiers)
    git.require_clean(repo)
    platform, origin, name = remote(repo, platform)
    client = Client(wizard, platform, origin, name)
    info = client.info()
    branch = info["default_branch"]
    scan_path = SCAN if platform == "github" else GL_SCAN
    scan = client.file(scan_path, branch)
    if scan is None:
        raise git.SetupError("Merge the setup PR/MR first. The all-tier scan is not on the default branch yet.")
    data = yaml.safe_load(scan)
    try:
        paths = (
            data["jobs"]["skills"]["with"]["paths"] if platform == "github" else data["variables"]["SKILL_EVAL_PATHS"]
        )
        roots = [git.safe_relative(value.strip()) for value in paths.splitlines() if value.strip()]
    except (KeyError, TypeError, AttributeError) as exc:
        raise git.SetupError("The all-tier scan is missing configured skill paths.") from exc
    if not roots:
        raise git.SetupError("No skill roots are configured.")
    reviews = client.reviews(DEMO_BRANCH)
    if reviews:
        wizard.console.print("Demo PR/MR already exists.")
        wizard.print_link(reviews[0].get("html_url") or reviews[0]["web_url"])
        return
    # Enumerate the live default branch, avoiding any local branch drift.
    if platform == "github":
        tree = client.api("GET", "/git/trees/" + quote(branch, safe="") + "?recursive=1")
        if tree.get("truncated"):
            raise git.SetupError("The remote tree is truncated; create a demo skill change manually.")
        manifests = [i["path"] for i in tree.get("tree", []) if i.get("type") == "blob"]
    else:
        manifests = []
        for page in range(1, 101):
            items = client.api(
                "GET",
                "/repository/tree?" + urlencode({"ref": branch, "recursive": "true", "per_page": 100, "page": page}),
            )
            manifests.extend(i["path"] for i in items if i.get("type") == "blob")
            if len(items) < 100:
                break
        else:
            raise git.SetupError("Remote discovery exceeded its size limit.")
    candidates = sorted(
        p
        for p in manifests
        if PurePosixPath(p).name == "SKILL.md" and any(r == "." or p.startswith(r + "/") for r in roots)
    )
    if not candidates:
        raise git.SetupError("No tracked skill manifest matches the configured roots.")
    target = git.safe_relative(candidates[0])
    content = client.file(target, branch)
    if content is None:
        raise git.SetupError("The demonstration manifest is missing.")
    if not no_input and not click.confirm("Open a small skill-change demo PR/MR?", default=False):
        wizard.console.print("Cancelled. No changes applied.")
        return
    # GitLab commits need update actions for existing files.
    url = client.create(
        DEMO_BRANCH,
        branch,
        {target: content + "\n<!-- SkillEvaluator onboarding demo: exercise automatic feedback. -->\n"},
        "Try automatic SkillEvaluator all-tier feedback",
        "Exercise the configured required Tier 1/Tier 2 and paired Tier 3 evaluation. Inspect the CI checks, updated comment and retained artifact reports. Close this demo without merging.",
    )
    wizard.console.print("Demo PR/MR created. Close it when finished; no merge needed.")
    wizard.print_link(url)
