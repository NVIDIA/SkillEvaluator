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
MODEL_ROUTING_CAPABILITY = "ci-model-routing-v1"
BENCHMARK_PR_CAPABILITY = "ci-benchmark-pr-v1"
MODEL_ROUTING_OPTIONS = (
    "tier1_llm",
    "tier1_provider",
    "tier1_model",
    "base_url",
    "embedding_base_url",
    "tier3_provider",
    "tier3_model",
    "tier3_base_url",
    "agent_model",
)
# Feedback pipelines run on the protected default branch; other root jobs must skip them.
GL_FEEDBACK_GUARD = {"if": '$SKILL_EVAL_FEEDBACK == "true"', "when": "never"}
GL_NOT_FEEDBACK = ".skill-evaluation-not-feedback"
_GL_GUARD_IF = re.compile(r"""\s*\$SKILL_EVAL_FEEDBACK\s*==\s*(["'])true\1\s*""")
_GL_GLOBAL_KEYS = {
    "after_script",
    "before_script",
    "cache",
    "default",
    "image",
    "include",
    "services",
    "stages",
    "variables",
    "workflow",
}
_GL_EVALUATOR_JOBS = {
    "skill-evaluation",
    "skill-evaluation-native",
    "skill-evaluation-native-tier2",
    "skill-evaluation-tier1-llm",
    "skill-evaluation-aggregate",
    "skill-evaluation-request-feedback",
    "skill-evaluation-feedback",
}


def evaluator_timeout_seconds(minutes: int) -> int:
    """End the evaluator before the CI job limit so setup and evidence retention fit."""
    return (minutes - min(20, minutes // 4)) * 60


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
    if (
        not isinstance(caps, list)
        or not caps
        or any(
            c not in {"all-tiers-github", "all-tiers-gitlab", MODEL_ROUTING_CAPABILITY, BENCHMARK_PR_CAPABILITY}
            for c in caps
        )
    ):
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
    host, origin, name = git.parse_origin(repo, _parse_origin)
    detected = "github" if host == "github.com" else "gitlab"
    selected = detected if platform == "auto" else platform
    if selected == "github" and host != "github.com":
        raise git.SetupError("GitHub onboarding currently supports github.com origins.")
    if selected == "gitlab" and host == "github.com":
        raise git.SetupError("The selected platform does not match the repository origin.")
    return selected, origin, name


def _parse_origin(value: str) -> tuple[str, str, str]:
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
    return host, origin, name


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
    # Generic GITLAB_TOKEN values are often issued for a different instance than
    # this origin; only the tool-scoped variable is sent to the origin host.
    token = os.environ.get("SKILLEVALUATOR_GITLAB_TOKEN")
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


def gitlab_admission_prerequisite(origin, *, online):
    requirement = {"server_version": None, "read_token_required": None, "verified": False}
    if not online:
        return requirement
    try:
        result = gitlab_api(origin, "GET", "version")
    except git.SetupError:
        return requirement
    version = result.get("version") if isinstance(result, dict) else None
    if not isinstance(version, str):
        return requirement
    match = re.match(r"^(\d{1,3})\.(\d{1,3})(?:\.|-|$)", version)
    if match:
        major, minor = map(int, match.groups())
        requirement.update(
            server_version=f"{major}.{minor}", read_token_required=(major, minor) < (18, 4), verified=True
        )
    return requirement


def _print_gitlab_admission_prerequisite(wizard, requirement):
    if not requirement["verified"]:
        wizard.console.print("GitLab server version unverified; admission readiness is unverified.")
        wizard.console.print(
            "GitLab before 18.4 needs SKILLEVALUATOR_GITLAB_READ_TOKEN (project Reporter/read_api); configure it until the server version is verified."
        )
    elif requirement["read_token_required"]:
        wizard.console.print(
            f"GitLab {requirement['server_version']} requires SKILLEVALUATOR_GITLAB_READ_TOKEN (project Reporter/read_api) for current-MR admission."
        )
    else:
        wizard.console.print(f"GitLab {requirement['server_version']} supports CI_JOB_TOKEN for current-MR admission.")
    wizard.console.print(
        "Admission credential validity is not verified by local setup/status; inspect the actual CI run."
    )


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
        "n-attempts": options["n_attempts"],
        "n-concurrent": options["n_concurrent"],
        "timeout-minutes": options["timeout_minutes"],
        "environment": options["environment"],
    }
    common.update(
        {
            key.replace("_", "-"): options[key]
            for key in MODEL_ROUTING_OPTIONS
            if key != "agent_model" and options.get(key)
        }
    )
    if options.get("agent_model"):
        common["agent-models"] = "\n".join(options["agent_model"])
    generic_chat = "openai-compatible" in {options["provider"], options.get("tier3_provider")}
    generic_embedding = options["embedding_provider"] == "openai-compatible"
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
        secrets = scan["jobs"]["skills"]["secrets"]
        if generic_chat:
            secrets["LLM_API_KEY"] = "${{ secrets." + options.get("gateway_secret", "SKILL_EVAL_LLM_API_KEY") + " }}"
        if generic_embedding:
            secrets["EMBEDDING_API_KEY"] = "${{ secrets.SKILL_EVAL_EMBEDDING_API_KEY }}"
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
        if options.get("benchmark_pr"):
            feedback["jobs"]["benchmarks"] = {
                "uses": f"{release['repository']}/.github/workflows/publish-benchmarks.yml@{release['feedback_ref']}",
                "permissions": {"contents": "write", "actions": "read", "pull-requests": "write"},
                "with": dict(feedback["jobs"]["feedback"]["with"]),
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
            "SKILL_EVAL_TIMEOUT_SECONDS": str(evaluator_timeout_seconds(options["timeout_minutes"])),
            "SKILL_EVAL_TOOLS_REPOSITORY": release["repository"],
            "SKILL_EVAL_TOOLS_REF": release["tools_ref"],
            "SKILL_EVAL_ENVIRONMENT": options["environment"],
            "SKILL_EVAL_RUNNER": options["runner"],
        }
    )
    if generic_chat and options.get("gateway_secret", "SKILL_EVAL_LLM_API_KEY") != "SKILL_EVAL_LLM_API_KEY":
        variables["SKILL_EVAL_LLM_API_KEY"] = "$" + options["gateway_secret"]
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


def _gitlab_root_needs_merge(repo, platform, files) -> bool:
    """A different existing root .gitlab-ci.yml is merged by hand, never replaced."""
    target = repo / GL_SCAN
    return (
        platform == "gitlab"
        and not target.is_symlink()
        and target.is_file()
        and target.read_bytes() != files[GL_SCAN].encode()
    )


def _refuse_gitlab_root_replacement(mode):
    if mode in {"save", "pr"}:
        raise git.SetupError(
            f"{GL_SCAN} already exists with different content; it will not be replaced. "
            "Run --mode preview to print the rendered configuration and the manual merge steps, "
            "then merge and commit it yourself."
        )


def _print_gitlab_root_merge_steps(wizard, existing):
    unguarded = gitlab_unguarded_jobs(existing)
    steps = [
        f"{GL_SCAN} already exists; it will not be replaced. Merge the rendered files below by hand:",
        f"1. Make the root include list exactly the two rendered entries: the pinned all-tiers.yml remote and "
        f"the {GL_FEEDBACK} local include with its rule. Remove the Tier 1 skill-evaluation.yml include and any "
        "other include; trusted feedback rejects other includes. Keep the root plain YAML (no !reference tags).",
        "2. Add the rendered variables, replacing earlier SKILL_EVAL_* values. If the root defines stages, "
        "keep test in the list; every SkillEvaluator job uses stage test.",
        "3. Use the rendered workflow rules. Keep any of your own workflow rules after the two SKILL_EVAL_FEEDBACK "
        "rules and before the final when: never, and only if they still allow merge-request, default-branch "
        "and web pipelines.",
        "4. Replace any skill-evaluation job override with the rendered skill-evaluation and "
        "skill-evaluation-native entries.",
        "5. Feedback pipelines run on the default branch. Start the rules of every other job with this "
        "guard; give a job without rules, only or except the guard followed by - when: on_success:"
        "\n     - if: '$SKILL_EVAL_FEEDBACK == \"true\"'\n       when: never",
        (f"   Jobs to guard: {', '.join(unguarded)}. " if unguarded else "   ")
        + "Merge the guards to the default branch first: the setup MR's own feedback request runs that "
        f"branch's current configuration. Write those guards as rules: extends: {GL_NOT_FEEDBACK} works only "
        "after the all-tiers include is on the default branch, for example for a job added later.",
        f"6. Create {GL_FEEDBACK} with the rendered content.",
        "7. Commit both through a merge request; after it merges, ci status --tiers 1,2,3 reports whether "
        "the root is recognized.",
    ]
    for line in steps:
        wizard.console.print(line, markup=False)


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
    if options.get("benchmark_pr"):
        if platform != "github":
            raise git.SetupError(
                "benchmark-update PR publication currently requires GitHub; GitLab cards retained in reports."
            )
        if BENCHMARK_PR_CAPABILITY not in release["capabilities"]:
            raise git.SetupError(f"--benchmark-pr requires a release declaring {BENCHMARK_PR_CAPABILITY}.")
    secret = options.get("gateway_secret", "SKILL_EVAL_LLM_API_KEY")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", secret) or secret.upper().startswith("GITHUB_"):
        raise git.SetupError(
            "--gateway-secret must name a valid CI secret (letters, digits, underscores; no GITHUB_ prefix)."
        )
    routing = (
        any(options.get(key) for key in MODEL_ROUTING_OPTIONS)
        or any(options.get(key) == "openai-compatible" for key in ("provider", "embedding_provider"))
        or secret != "SKILL_EVAL_LLM_API_KEY"
    )
    if routing and MODEL_ROUTING_CAPABILITY not in release["capabilities"]:
        raise git.SetupError(
            f"These options require a release declaring {MODEL_ROUTING_CAPABILITY}; old pinned workflows cannot accept them."
        )
    selected_agents = set()
    for entry in options.get("agent_model", ()):
        if not re.fullmatch(r"(?:claude-code|codex|opencode)=[^\s=,]+", entry) or len(entry) > 280:
            raise git.SetupError(
                "--agent-model requires a supported AGENT=MODEL without whitespace (claude-code, codex, opencode)."
            )
        agent = entry.split("=", 1)[0]
        if agent in selected_agents:
            raise git.SetupError(f"--agent-model: specify only one model for {agent}.")
        selected_agents.add(agent)
    branch = git.infer_branch(repo, default_branch)
    roots = git.discover(repo)
    for index, item in enumerate(roots, 1):
        wizard.console.print(f"{index}. {item['root']} ({item['count']} skills)", markup=False)
    selected = wizard.select_roots(roots, paths, no_input)
    options.update(
        min_score=70 if min_score is None else min_score, scan_mode=scan_mode or "auto", tier3_required=False
    )
    options["embedding_provider"] = options["embedding_provider"] or (
        "openai" if options["provider"] == "anthropic" else options["provider"]
    )
    options["runner"] = options["runner"] or ("ubuntu-24.04" if platform == "github" else "docker")
    files = render(wizard, repo, release, platform, branch, selected, options)
    merge_root = _gitlab_root_needs_merge(repo, platform, files)
    if merge_root:
        _refuse_gitlab_root_replacement(mode)
    changed = wizard.check_files(repo, {p: c for p, c in files.items() if not (merge_root and p == GL_SCAN)})
    if merge_root:
        changed.insert(0, GL_SCAN)
    if output:
        output = output.resolve()
        if output.exists() or output.is_relative_to(repo):
            raise git.SetupError("Choose a new receipt path outside the consumer checkout.")
        # Fail before saving files or opening a PR/MR, never after.
        output.parent.mkdir(parents=True, exist_ok=True)
    wizard.console.print(
        f"{platform.title()} | Tier 1 required; Tiers 2 and 3 advisory",
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
            "GitLab: protect the default branch and add a scoped protected SKILLEVALUATOR_GITLAB_TOKEN for trusted feedback. Permit job-token pipeline triggers; the initiating user needs permission to run protected-default-branch pipelines, pass pipeline variables and run the feedback environment."
        )
    mode = mode or click.prompt("Next", default="preview", type=click.Choice(["preview", "save", "pr"]))
    if merge_root:
        _refuse_gitlab_root_replacement(mode)
    admission = None
    if platform == "gitlab":
        admission = gitlab_admission_prerequisite(origin, online=mode == "pr")
        _print_gitlab_admission_prerequisite(wizard, admission)
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
    if admission is not None:
        plan["gitlab_admission"] = admission
    if mode == "preview":
        plan["action"] = "preview"
        if merge_root:
            _print_gitlab_root_merge_steps(wizard, (repo / GL_SCAN).read_text(errors="replace"))
            show_files = True
        if changed:
            wizard.console.print(
                "Proposed files:" if show_files else "Proposed files (add --show-files to print them):"
            )
            for path in changed:
                wizard.console.print("  " + path, markup=False)
        else:
            wizard.console.print("The all-tier files already match this checkout.")
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
                raise git.SetupError(
                    f"{path} differs on the default branch; it will not be replaced."
                    + (
                        " Update your checkout to the default branch, then run --mode preview to print the"
                        " manual merge steps."
                        if path == GL_SCAN
                        else ""
                    )
                )
            if current is None:
                pending[path] = content
        if pending:
            url = client.create(
                SETUP_BRANCH,
                branch,
                pending,
                "Set up SkillEvaluator all-tier checks and feedback",
                "Add pinned Tier 1 required checks, Tier 2 and Tier 3 advisory evaluation, and trusted feedback linking retained private CI artifacts.\n\nMerge this setup, configure provider credentials and the protected execution environment, then run `skillevaluator ci demo --tiers 1,2,3`. This setup is not an evaluation pass.",
                setup=True,
            )
            plan.update(action="created-pr", pr_url=url)
            wizard.console.print("Setup PR/MR ready. Merge it, then run skillevaluator ci demo --tiers 1,2,3.")
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
    if tiers is not None or platform == "gitlab" or (repo / SCAN).exists():
        return True
    if selected == "gitlab":
        if (repo / GL_FEEDBACK).exists():
            return True
        # Never contact an unrecognized origin host without an explicit opt-in.
        raise git.SetupError("This origin is not github.com. For GitLab all-tier onboarding, add --tiers 1,2,3.")
    return False


def trusted_blobs(raw) -> list[str] | None:
    """Parse approved caller blobs exactly like the trusted feedback job."""
    pins = [part.strip() for part in re.split(r"[,\n]", str(raw or ""))]
    return pins if 0 < len(pins) <= 32 and all(SHA.fullmatch(pin) for pin in pins) else None


def _github_workflow_inputs(text: str, job_name: str, workflow: str) -> tuple[dict, str | None]:
    """Recognize pinned all-tier callers without treating arbitrary YAML as setup."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}, "invalid YAML"
    jobs = document.get("jobs") if isinstance(document, dict) else None
    job = jobs.get(job_name) if isinstance(jobs, dict) else None
    expected = rf"NVIDIA/SkillEvaluator/\.github/workflows/{re.escape(workflow)}@[0-9a-f]{{40}}"
    if not isinstance(job, dict) or not re.fullmatch(expected, str(job.get("uses", ""))):
        return {}, "missing the immutable reusable workflow"
    inputs = job.get("with")
    if not isinstance(inputs, dict):
        return {}, "missing reusable workflow inputs"
    if job_name == "skills" and inputs.get("tiers", "1,2,3") != "1,2,3":
        return {}, "not all three tiers are selected"
    if not SHA.fullmatch(str(inputs.get("tools-ref", ""))):
        return {}, "missing an immutable tools-ref"
    if job_name == "skills":
        if not re.fullmatch(r"ghcr\.io/nvidia/skillevaluator-tier1@sha256:[0-9a-f]{64}", str(inputs.get("image", ""))):
            return {}, "missing the immutable Tier 1 image"
    else:
        producer = r"NVIDIA/SkillEvaluator/\.github/workflows/evaluate-all-tiers\.yml@[0-9a-f]{40}"
        if not re.fullmatch(producer, str(inputs.get("trusted-workflow-ref", ""))):
            return {}, "missing the immutable trusted-workflow-ref"
        blobs = inputs.get("trusted-caller-blobs")
        if not isinstance(blobs, str) or not blobs.strip():
            return {}, "missing trusted-caller-blobs"
    return inputs, None


def gitlab_config_problem(text: str) -> str | None:
    """Mirror ci/feedback.py validate_gitlab_config for the default-branch root config."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return "invalid YAML"
    includes = document.get("include") if isinstance(document, dict) else None
    if not isinstance(includes, list) or len(includes) != 2:
        return "missing the pinned evaluator and trusted feedback includes"
    expected_rule = (
        '$CI_PIPELINE_SOURCE == "pipeline" && $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH && $SKILL_EVAL_FEEDBACK == "true"'
    )
    pinned = re.compile(
        r"https://raw\.githubusercontent\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[0-9a-f]{40}/ci/gitlab/all-tiers\.yml"
    )
    remotes = [i for i in includes if isinstance(i, dict) and set(i) == {"remote"}]
    locals_ = [i for i in includes if i == {"local": GL_FEEDBACK, "rules": [{"if": expected_rule}]}]
    if len(remotes) != 1 or not isinstance(remotes[0]["remote"], str) or not pinned.fullmatch(remotes[0]["remote"]):
        return "missing an immutable evaluator include"
    if len(locals_) != 1:
        return "missing the trusted downstream feedback include"
    return None


def _gitlab_rules(document, job, depth=0):
    if "rules" in job or depth > 10:
        return job.get("rules")
    extends = job.get("extends")
    for parent in reversed([extends] if isinstance(extends, str) else extends if isinstance(extends, list) else []):
        if parent == GL_NOT_FEEDBACK:
            return [GL_FEEDBACK_GUARD]
        base = document.get(parent)
        rules = _gitlab_rules(document, base, depth + 1) if isinstance(base, dict) else None
        if rules is not None:
            return rules
    return None


_GL_UNSET = object()


def _gitlab_inherited(document, job, key, depth=0):
    """Resolve key as extends does: the job's own value, even null, wins, then the last parent defining it."""
    if not isinstance(job, dict) or depth > 10:
        return _GL_UNSET
    if key in job:
        return job[key]
    extends = job.get("extends")
    for parent in reversed([extends] if isinstance(extends, str) else extends if isinstance(extends, list) else []):
        value = (
            _gitlab_inherited(document, document.get(parent), key, depth + 1) if isinstance(parent, str) else _GL_UNSET
        )
        if value is not _GL_UNSET:
            return value
    return _GL_UNSET


def _gitlab_only_except(document, job) -> bool:
    """GitLab rejects only/except beside rules, including inherited ones; a null value drops an inherited key."""
    values = [_gitlab_inherited(document, job, key) for key in ("only", "except")]
    return any(value is not None and value is not _GL_UNSET for value in values)


def gitlab_unguarded_jobs(text: str) -> list[str]:
    """Name root jobs that a trusted default-branch feedback pipeline would also run."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return []
    if not isinstance(document, dict):
        return []
    unguarded = []
    for name, job in document.items():
        if (
            not isinstance(name, str)
            or not isinstance(job, dict)
            or name.startswith(".")
            or name in _GL_GLOBAL_KEYS
            or name in _GL_EVALUATOR_JOBS
        ):
            continue
        rules = _gitlab_rules(document, job)
        first = rules[0] if isinstance(rules, list) and rules else None
        # A job keeping only/except cannot also carry the guard rules.
        if _gitlab_only_except(document, job) or not (
            isinstance(first, dict)
            and set(first) == {"if", "when"}
            and first["when"] == "never"
            and isinstance(first["if"], str)
            and _GL_GUARD_IF.fullmatch(first["if"])
        ):
            unguarded.append(name)
    return unguarded


def inspect(wizard, repo, platform="auto", tiers=None):
    if tiers is not None:
        validate_tiers(tiers)
    platform, origin, name = remote(repo, platform)
    client = Client(wizard, platform, origin, name)
    info = client.info()
    branch = info["default_branch"]
    scan_path, feedback_path = (SCAN, FEEDBACK) if platform == "github" else (GL_SCAN, GL_FEEDBACK)
    scan, feedback = client.file(scan_path, branch), client.file(feedback_path, branch)
    problem, feedback_problem, feedback_inputs = None, None, {}
    if platform == "github":
        if scan is not None:
            _, problem = _github_workflow_inputs(scan, "skills", "evaluate-all-tiers.yml")
        if feedback is not None:
            feedback_inputs, feedback_problem = _github_workflow_inputs(feedback, "feedback", "feedback-all-tiers.yml")
    elif scan is not None:
        problem = gitlab_config_problem(scan)
    wizard.console.print(f"{platform.title()} {name} | default branch: {branch}", markup=False)
    wizard.console.print(
        "All-tier checks: "
        + (
            "awaiting setup PR/MR merge"
            if scan is None
            else f"{scan_path} is present but not an all-tier configuration ({problem})"
            if problem
            else "configured"
        ),
        markup=False,
    )
    wizard.console.print(
        "Trusted feedback: "
        + (
            "missing"
            if feedback is None
            else f"{feedback_path} is present but not a trusted feedback configuration ({feedback_problem})"
            if feedback_problem
            else "configured"
        ),
        markup=False,
    )
    # Advisory only; the trusted include check above keeps parity with feedback.
    unguarded = gitlab_unguarded_jobs(scan) if scan and platform == "gitlab" and not problem else []
    if unguarded:
        wizard.console.print(
            "Warning: trusted feedback pipelines on the default branch also run "
            + ", ".join(unguarded)
            + ". Start their rules with if: '$SKILL_EVAL_FEEDBACK == \"true\"' and when: never; a job without "
            + "rules, only or except can extend "
            + GL_NOT_FEEDBACK
            + ".",
            markup=False,
        )
    if platform == "gitlab":
        _print_gitlab_admission_prerequisite(wizard, gitlab_admission_prerequisite(origin, online=True))
    if scan is not None and feedback is not None and platform == "github" and not problem and not feedback_problem:
        pins = trusted_blobs(feedback_inputs.get("trusted-caller-blobs"))
        wizard.console.print(
            "Scan approval: "
            + (
                "Malformed approval list; feedback will reject it"
                if pins is None
                else "matches exact workflow"
                if wizard.blob_id(scan) in pins
                else "Needs refresh; feedback will reject it"
            )
        )
    suffix = (
        "/actions/workflows/skill-checks-all-tiers.yml/runs?per_page=1"
        if platform == "github"
        else "/pipelines?" + urlencode({"ref": branch, "per_page": 1})
    )
    if problem:
        suffix = ""
    runs = client.api("GET", suffix, missing_ok=True) if suffix else None
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
        "Exercise the configured required Tier 1, advisory Tier 2 and paired Tier 3 evaluation. Inspect the CI checks, updated comment and retained artifact reports. Close this demo without merging.",
    )
    wizard.console.print("Demo PR/MR created. Close it when finished; no merge needed.")
    wizard.print_link(url)
