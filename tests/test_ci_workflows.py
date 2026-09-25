# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
REVIEWED_NPM_COMMAND = ["npm", "ci", "--prefix", "fern", "--ignore-scripts", "--omit=optional"]

REQUIRED_CI_JOBS = {
    "test-python-312": "Tests (Python 3.12)",
    "test-python-313": "Tests (Python 3.13)",
    "package": "Package",
    "rhel8-security-install": "RHEL 8 security install",
    "tier2-macos": "Tier 2 (macos-latest)",
    "tier2-windows": "Tier 2 (windows-latest)",
    "tier3-macos": "Tier 3 macOS contract and progress",
    "native-windows-local-mode": "Native Windows local mode fails closed",
}
HEAVY_CI_JOBS = set(REQUIRED_CI_JOBS) - {"test-python-312"}
RUN_UNLESS_CANCELLED_IF = "${{ !cancelled() }}"
FULL_LANE_IF = "${{ !cancelled() && needs.classify-changes.outputs.docs_only != 'true' }}"
NOT_DOCS_ONLY_IF = "${{ needs.classify-changes.outputs.docs_only != 'true' }}"
PR_CONCURRENCY = {
    "group": "${{ github.workflow }}-${{ github.event.pull_request.number || github.run_id }}",
    "cancel-in-progress": "true",
}


def _load(name: str) -> dict[str, Any]:
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _workflow_names() -> list[str]:
    """Every workflow on disk, so a new one is covered the day it lands."""
    names = sorted(path.name for pattern in ("*.yml", "*.yaml") for path in WORKFLOWS.glob(pattern))
    assert names, "no workflows found"
    return names


def _assert_no_path_filter(workflow: dict[str, Any], event: str = "pull_request") -> None:
    trigger = workflow["on"][event]
    if isinstance(trigger, dict):
        assert "paths" not in trigger
        assert "paths-ignore" not in trigger


def _runs(job: dict[str, Any]) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def _all_uses(workflow: dict[str, Any]) -> list[str]:
    """Every action reference: step-level actions and job-level reusable workflows."""
    step_uses = [step["uses"] for job in workflow["jobs"].values() for step in job.get("steps", []) if "uses" in step]
    job_uses = [job["uses"] for job in workflow["jobs"].values() if "uses" in job]
    return step_uses + job_uses


def _all_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


def _action_repository(uses: str) -> str:
    """The ``owner/repo`` half of a ``uses:`` value, folded for comparison.

    GitHub resolves repository names case-insensitively, so ``Actions/Checkout``
    runs exactly the action ``actions/checkout`` names. A guard that filters on
    one literal spelling therefore skips a step that really does run -- which is
    a silent hole, not a cosmetic one.
    """
    return uses.split("@", 1)[0].casefold()


def _is_checkout_step(step: dict[str, Any]) -> bool:
    """Whether ``step`` runs ``actions/checkout``, however GitHub spells it.

    An unpinned ``uses: actions/checkout`` counts too: the pin guard is what
    rejects that, and until it does the step still checks out a credentialed
    workspace.
    """
    return _action_repository(step.get("uses", "")) == "actions/checkout"


def test_ci_preserves_required_contexts_as_explicit_jobs() -> None:
    ci = _load("ci.yml")

    assert {job_id: ci["jobs"][job_id]["name"] for job_id in REQUIRED_CI_JOBS} == REQUIRED_CI_JOBS
    assert all("matrix." not in ci["jobs"][job_id]["name"] for job_id in REQUIRED_CI_JOBS)
    _assert_no_path_filter(ci)


def test_ci_classifier_is_pull_request_only_and_exports_docs_only() -> None:
    ci = _load("ci.yml")
    classifier = ci["jobs"]["classify-changes"]

    assert ci["concurrency"] == PR_CONCURRENCY
    assert classifier["name"] == "Classify changes"
    assert classifier["if"] == "${{ github.event_name == 'pull_request' }}"
    assert classifier["outputs"]["docs_only"] == "${{ steps.changes.outputs.docs_only }}"
    assert classifier["steps"][0]["with"]["fetch-depth"] == "0"
    assert classifier["steps"][0]["with"]["persist-credentials"] == "false"
    assert classifier["steps"][1]["id"] == "changes"
    classifier_run = classifier["steps"][1]["run"]
    assert 'git show "$BASE_SHA:scripts/classify_ci_changes.py"' in classifier_run
    assert 'python3 "$classifier"' in classifier_run
    assert 'echo "docs_only=false" >> "$GITHUB_OUTPUT"' in classifier_run
    assert "python3 scripts/classify_ci_changes.py" not in classifier_run
    assert classifier["steps"][1]["env"] == {
        "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
        "HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
    }


def test_ci_validates_fern_on_the_required_python_312_context() -> None:
    job = _load("ci.yml")["jobs"]["test-python-312"]

    assert job["needs"] == "classify-changes"
    assert job["if"] == RUN_UNLESS_CANCELLED_IF
    assert job["runs-on"] == "ubuntu-latest"
    assert job["steps"][0]["with"]["persist-credentials"] == "false"

    full_lane_step_names = {
        "Set up Python",
        "Set up uv",
        "Install dependencies",
        "Scan OSS source boundary",
        "Lint",
        "Run tests with coverage",
    }
    full_lane_steps = {step.get("name"): step for step in job["steps"] if step.get("name") in full_lane_step_names}
    assert set(full_lane_steps) == full_lane_step_names
    assert all(step["if"] == NOT_DOCS_ONLY_IF for step in full_lane_steps.values())

    node_step = next(step for step in job["steps"] if step.get("name") == "Set up Node.js for docs")
    docs_step = next(step for step in job["steps"] if step.get("name") == "Validate Fern documentation")
    assert "if" not in node_step, "Fern validation must run for mixed and Fern dependency PRs"
    assert len(node_step["uses"].split("@", 1)[1]) == 40
    assert "if" not in docs_step, "Fern validation must run for mixed and Fern dependency PRs"
    assert "npm ci --prefix fern --ignore-scripts --omit=optional" in docs_step["run"]
    assert "./fern/node_modules/.bin/fern check" in docs_step["run"]
    assert "GITHUB_STEP_SUMMARY" in docs_step["run"]
    assert "platform test matrix was skipped" not in docs_step["run"]


def test_ci_skips_every_other_required_job_only_after_classification() -> None:
    jobs = _load("ci.yml")["jobs"]

    for job_id in HEAVY_CI_JOBS:
        assert jobs[job_id]["needs"] == "classify-changes"
        assert jobs[job_id]["if"] == FULL_LANE_IF


def test_full_lane_keeps_the_existing_commands_and_runners() -> None:
    jobs = _load("ci.yml")["jobs"]

    assert jobs["test-python-313"]["runs-on"] == "ubuntu-latest"
    assert "uv run pytest -q" in _runs(jobs["test-python-313"])
    assert jobs["tier2-macos"]["runs-on"] == "macos-latest"
    assert jobs["tier2-windows"]["runs-on"] == "windows-latest"
    assert "tests/embedding" in _runs(jobs["tier2-macos"])
    assert "tests/embedding" in _runs(jobs["tier2-windows"])
    assert jobs["tier3-macos"]["runs-on"] == "macos-latest"
    assert "tests/test_tier3_progress.py" in _runs(jobs["tier3-macos"])
    assert jobs["native-windows-local-mode"]["runs-on"] == "windows-latest"
    assert "tests/test_harbor_local_mode.py" in _runs(jobs["native-windows-local-mode"])
    assert jobs["rhel8-security-install"]["container"] == "rockylinux/rockylinux:8.10"
    assert "uv build --wheel" in _runs(jobs["rhel8-security-install"])
    assert "twine==6.2.0" in _runs(jobs["package"])


def test_security_keeps_gitleaks_always_on_and_skips_only_nonessential_jobs() -> None:
    security = _load("security.yml")
    jobs = security["jobs"]

    _assert_no_path_filter(security)
    assert security["concurrency"] == PR_CONCURRENCY
    assert "if" not in jobs["gitleaks"]
    assert "needs" not in jobs["gitleaks"]
    assert jobs["classify-changes"]["if"] == "${{ github.event_name == 'pull_request' }}"
    assert jobs["classify-changes"]["outputs"]["docs_only"] == "${{ steps.changes.outputs.docs_only }}"
    assert jobs["classify-changes"]["steps"][0]["with"]["persist-credentials"] == "false"
    classifier_run = jobs["classify-changes"]["steps"][1]["run"]
    assert 'git show "$BASE_SHA:scripts/classify_ci_changes.py"' in classifier_run
    assert 'python3 "$classifier"' in classifier_run
    assert 'echo "docs_only=false" >> "$GITHUB_OUTPUT"' in classifier_run
    assert "python3 scripts/classify_ci_changes.py" not in classifier_run

    dependency_if = " ".join(jobs["dependency-review"]["if"].split())
    codeql_if = " ".join(jobs["codeql"]["if"].split())
    for job_id in ("dependency-review", "codeql"):
        assert jobs[job_id]["needs"] == "classify-changes"
        assert "!cancelled()" in jobs[job_id]["if"]
        assert "always()" not in jobs[job_id]["if"]
        assert "needs.classify-changes.outputs.docs_only != 'true'" in jobs[job_id]["if"]
    assert "github.event_name == 'pull_request'" in dependency_if
    assert "github.event.repository.private == false" in dependency_if
    assert "vars.ENABLE_GITHUB_ADVANCED_SECURITY == 'true'" in dependency_if
    assert "github.event.repository.private == false" in codeql_if
    assert "vars.ENABLE_GITHUB_ADVANCED_SECURITY == 'true'" in codeql_if


def test_gitleaks_uses_event_specific_full_history_scopes() -> None:
    job = _load("security.yml")["jobs"]["gitleaks"]
    checkout = next(step for step in job["steps"] if _is_checkout_step(step))
    verify = next(step for step in job["steps"] if step.get("name") == "Verify full Git checkout")
    scan = next(step for step in job["steps"] if step.get("name") == "Scan Git history")

    assert str(checkout["with"]["fetch-depth"]) == "0"
    assert 'test "$(git rev-parse --is-shallow-repository)" = "false"' in verify["run"]
    assert 'git rev-parse --verify "HEAD^{commit}"' in verify["run"]
    assert (
        "gitleaks:v8.30.0@sha256:691af3c7c5a48b16f187ce3446d5f194838f91238f27270ed36eef6359a574d9"
        in scan["run"]
    )
    scan_run = scan["run"]
    head_opts = "--full-history --diff-filter=tuxdb HEAD --"
    audit_opts = "--full-history --all --diff-filter=tuxdb --"
    assignments = re.findall(r'log_opts="([^"]+)"', scan_run)

    assert assignments == [head_opts, audit_opts]
    assert scan["shell"] == "bash"
    assert "set -euo pipefail" in scan_run
    assert 'case "$GITHUB_EVENT_NAME" in' in scan_run
    assert re.search(
        rf'pull_request\|push\)\s+log_opts="{re.escape(head_opts)}"\s+;;', scan_run
    )
    assert re.search(rf'\*\)\s+log_opts="{re.escape(audit_opts)}"\s+;;', scan_run)
    assert '--log-opts="$log_opts"' in scan_run


def test_dco_stays_unconditional_and_has_no_path_filter() -> None:
    dco = _load("dco.yml")

    _assert_no_path_filter(dco)
    assert "if" not in dco["jobs"]["dco"]
    assert "needs" not in dco["jobs"]["dco"]


def test_non_pr_workflow_triggers_are_preserved() -> None:
    ci = _load("ci.yml")
    security = _load("security.yml")

    assert ci["on"]["push"] == {"branches": ["main"]}
    assert set(security["on"]) == {"pull_request", "push", "schedule", "workflow_dispatch"}
    assert security["on"]["push"] == {"branches": ["main"]}
    assert security["on"]["schedule"] == [{"cron": "23 7 * * 1"}]
    assert "workflow_dispatch" in security["on"]


def _is_local_reference(uses: str) -> bool:
    """A same-repo composite action or reusable workflow, e.g. ``./.github/actions/x``.

    GitHub always resolves these from the caller's own commit, so nothing about
    them can float and there is no ``@ref`` syntax for one. A reference that is
    both ``./``-prefixed and carries an ``@`` is therefore not this syntax --
    treat it as a normal reference so it still has to satisfy the SHA-pin check.
    """
    return uses.startswith("./") and "@" not in uses


def _local_reference_escapes_repo(uses: str) -> bool:
    return ".." in Path(uses).parts


def test_every_workflow_pins_every_action_to_a_commit() -> None:
    for workflow_name in _workflow_names():
        for uses in _all_uses(_load(workflow_name)):
            if _is_local_reference(uses):
                assert not _local_reference_escapes_repo(uses), f"{workflow_name}: {uses}"
                continue
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", uses), f"{workflow_name}: {uses}"


def test_every_workflow_does_not_persist_checkout_credentials() -> None:
    checkout_steps = [
        (workflow_name, step)
        for workflow_name in _workflow_names()
        for step in _all_steps(_load(workflow_name))
        if _is_checkout_step(step)
    ]
    assert checkout_steps
    for workflow_name, step in checkout_steps:
        assert step.get("with", {}).get("persist-credentials") == "false", workflow_name


def test_publish_docs_installs_the_fern_cli_from_the_committed_lockfile() -> None:
    """The secret-bearing job runs a CLI whose whole tree was reviewed, not resolved.

    ``npm ci`` reproduces ``fern/package-lock.json`` exactly -- every version and
    integrity hash -- so nothing between the commit and the run can change what
    executes next to ``FERN_TOKEN``.
    """
    job = _load("publish-docs.yml")["jobs"]["run"]
    install_commands = [argv for step in job["steps"] for argv in _npm_commands(step.get("run", ""))]
    assert install_commands == [REVIEWED_NPM_COMMAND], (
        "publish-docs.yml must install the pinned Fern CLI with the reviewed command"
    )

    publish_step = next(step for step in job["steps"] if "fern generate" in step.get("run", ""))
    assert "fern/node_modules/.bin/fern" in publish_step["run"], "run the installed CLI, not one from PATH"


def test_every_workflow_declares_explicit_permissions() -> None:
    """A job-level permissions: block overrides the workflow-level one, so check both.

    This rejects the write-all shorthand and the inherited-default token scope, not
    every broad grant -- a granular write like `contents: write` is a legitimate
    choice for a release workflow and is not this test's concern.
    """
    for workflow_name in _workflow_names():
        workflow = _load(workflow_name)
        assert workflow.get("permissions") is not None, (
            f"{workflow_name} inherits the repository default token scope"
        )
        for scope, permissions in [("workflow", workflow["permissions"])] + [
            (f"job {job_id}", job["permissions"]) for job_id, job in workflow["jobs"].items() if "permissions" in job
        ]:
            assert permissions != "write-all", f"{workflow_name}: {scope}"


# npm is not always the first word of a line: `cd fern && npm install`, `env npm
# install` and `npm --prefix fern install` are all real installs that a
# line-anchored pattern never sees. These guards therefore split each `run:`
# block into the commands a shell would run and inspect each one's argv -- and
# fail closed on any npm mention this parser cannot resolve to a plain
# invocation, because a guard that silently sees nothing is worse than one that
# is noisy.
NPM_EXECUTABLE = re.compile(r"^(?:.*[/\\])?npm(?:\.cmd|\.exe|\.ps1)?$", re.IGNORECASE)
NPM_MENTION = re.compile(
    r"""(?:^|[\s'"`(;&|=])(?:[\w./\\-]*[/\\])?npm(?:\.cmd|\.exe|\.ps1)?(?=$|[\s'"`);&|])""",
    re.IGNORECASE,
)
SHELL_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
SHELL_OPERATOR = re.compile(r"&&|\|\||[;&|()]")
SHELL_WRAPPERS = frozenset({"command", "env", "exec", "ionice", "nice", "setsid", "stdbuf", "sudo", "time"})
SHELL_KEYWORDS = frozenset({"!", "do", "elif", "else", "if", "then", "until", "while", "{"})


def _logical_lines(run: str) -> list[str]:
    """``run`` split into lines, with backslash continuations joined back up."""
    return re.sub(r"\\\n", " ", run).splitlines()


def _shell_segments(line: str) -> list[str]:
    """``line`` cut into command texts at every *unquoted* shell operator.

    Quoting is tracked by hand rather than left to ``shlex``'s lexer, because
    the lexer answers two questions wrongly for this job. It ends a word at any
    ``#``, so the URL in ``curl https://host/doc#install && npm install`` would
    swallow the install; and once a token is quoted it no longer says whether
    that token was an operator, so an argument like ``";"`` would cut the
    command in half. Both mistakes end with an npm command going unguarded,
    which is the failure this parser exists to prevent.
    """
    segments: list[str] = []
    current: list[str] = []
    quote = ""
    index = 0
    while index < len(line):
        char = line[index]
        if quote:
            current.append(char)
            quote = "" if char == quote else quote
            index += 1
            continue
        if char in "'\"":
            quote = char
            current.append(char)
            index += 1
            continue
        if char == "\\" and index + 1 < len(line):
            current.extend(line[index : index + 2])
            index += 2
            continue
        if char == "#" and (not current or current[-1].isspace() or current[-1] in "(;&|"):
            break  # A word-initial # opens a comment: the rest of the line is not a command.
        operator = SHELL_OPERATOR.match(line, index)
        if operator:
            segments.append("".join(current))
            current = []
            index = operator.end()
            continue
        current.append(char)
        index += 1
    if quote:
        raise ValueError(f"unbalanced {quote} quote")
    segments.append("".join(current))
    return [segment for segment in segments if segment.strip()]


def _shell_commands(line: str) -> list[list[str]]:
    """``line`` split into the separate commands a shell would run.

    Raises ``ValueError`` on a line no shell could read either, e.g. one
    holding an unbalanced quote; the callers decide what that means.
    """
    return [shlex.split(segment, comments=False) for segment in _shell_segments(line)]


def _invoked_command(command: Sequence[str]) -> list[str]:
    """``command`` with what stands between the shell and the program removed.

    Environment assignments, wrappers such as ``env``, and the keywords that
    open a compound statement all sit in front of a command that really does
    run, so a guard that reads only the first word misses it.
    """
    argv = list(command)
    while argv and (
        SHELL_ASSIGNMENT.match(argv[0])
        or argv[0] in SHELL_WRAPPERS
        or argv[0] in SHELL_KEYWORDS
        or argv[0].startswith("-")
    ):
        argv = argv[1:]
    return argv


def _npm_commands(run: str) -> list[list[str]]:
    """Every npm invocation in a ``run:`` block, as its argv tokens.

    Anything holding the word ``npm`` that does not come out of this as a plain
    npm invocation -- a line the lexer cannot read, an install hidden inside
    ``bash -c "..."``, a wrapper this parser does not model -- raises instead of
    being skipped. Guards that quietly match nothing are the bug this replaces.
    """
    invocations: list[list[str]] = []
    for line in _logical_lines(run):
        try:
            commands = _shell_commands(line)
        except ValueError as error:
            assert not NPM_MENTION.search(line), f"unreadable npm command line ({error}): {line.strip()}"
            continue
        for command in commands:
            argv = _invoked_command(command)
            if argv and NPM_EXECUTABLE.match(argv[0]):
                invocations.append(argv)
                continue
            assert not any(NPM_MENTION.search(token) for token in command), (
                f"npm invocation these guards cannot classify: {line.strip()}"
            )
    return invocations


def test_npm_commands_sees_invocations_that_do_not_open_the_line() -> None:
    """The forms a line-anchored pattern silently missed."""
    assert _npm_commands("cd fern && npm install --ignore-scripts") == [["npm", "install", "--ignore-scripts"]]
    assert _npm_commands("env npm install") == [["npm", "install"]]
    assert _npm_commands("npm --prefix fern install") == [["npm", "--prefix", "fern", "install"]]
    assert _npm_commands("CI=true npm ci") == [["npm", "ci"]]
    assert _npm_commands("./node_modules/.bin/../../bin/npm ci") == [["./node_modules/.bin/../../bin/npm", "ci"]]
    assert _npm_commands("npm ci --ignore-scripts; npm run build") == [
        ["npm", "ci", "--ignore-scripts"],
        ["npm", "run", "build"],
    ]


def test_npm_commands_recognizes_windows_case_insensitive_executable_names() -> None:
    assert _npm_commands("NPM install --ignore-scripts") == [["NPM", "install", "--ignore-scripts"]]
    assert _npm_commands("NpM.CmD ci --ignore-scripts") == [["NpM.CmD", "ci", "--ignore-scripts"]]
    assert _npm_commands("npm.ps1 ci --ignore-scripts") == [["npm.ps1", "ci", "--ignore-scripts"]]


def test_npm_commands_reads_quoting_and_comments_the_way_a_shell_does() -> None:
    """Where the line ends and where an argument ends are both easy to get wrong.

    A ``#`` inside a URL does not open a comment, a quoted ``;`` is an argument
    rather than a command break, and a command inside ``if``/``then`` still runs.
    Each mistake hides a real npm invocation from the guards below.
    """
    assert _npm_commands("curl https://host/doc#install && npm install") == [["npm", "install"]]
    assert _npm_commands("npm ci --ignore-scripts # installs the pinned CLI") == [["npm", "ci", "--ignore-scripts"]]
    assert _npm_commands("if [ -d fern ]; then npm ci --ignore-scripts; fi") == [["npm", "ci", "--ignore-scripts"]]

    quoted = _npm_commands("npm ci --ignore-scripts ';' --no-ignore-scripts")
    assert quoted == [["npm", "ci", "--ignore-scripts", ";", "--no-ignore-scripts"]]


def test_npm_commands_ignores_npm_outside_a_command() -> None:
    assert _npm_commands("# npm install is what this step replaces") == []
    assert _npm_commands("uv run pytest") == []
    assert _npm_commands("echo 'no npm-cache here' > /dev/null") == []


def test_npm_commands_fails_closed_on_forms_it_cannot_classify() -> None:
    """An npm this parser cannot resolve must be loud, never skipped."""
    for run in (
        'bash -c "npm install"',
        "sudo -u builder npm ci --ignore-scripts",
        "xargs npm install --ignore-scripts",
        'NPM_BIN=npm && "$NPM_BIN" ci --ignore-scripts',
        "$(which npm) ci --ignore-scripts",
        "npm ci --ignore-scripts 'unbalanced",
    ):
        with pytest.raises(AssertionError):
            _npm_commands(run)


@pytest.mark.parametrize(
    "run",
    [
        "npm install --package-lock-only --ignore-scripts",
        "npm ci --ignore-scripts --omit=optional",
        "npm ci --prefix fern --ignore-scripts --omit=optional --no-omit",
        "npm exec --yes --ignore-scripts --package-lock-only install",
        "npm ci -- --ignore-scripts",
        "npm ci --prefix fern --ignore-scripts --no-ignore-s --omit=optional",
        "npm cit --prefix fern --ignore-scripts --omit=optional",
    ],
)
def test_unreviewed_npm_commands_fail_the_workflow_guard(
    run: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "probe.yml").write_text(
        yaml.safe_dump({"jobs": {"probe": {"steps": [{"run": run}]}}}), encoding="utf-8"
    )
    monkeypatch.setattr(sys.modules[__name__], "WORKFLOWS", tmp_path)

    with pytest.raises(AssertionError):
        test_every_workflow_uses_only_the_reviewed_npm_command()


def test_every_workflow_uses_only_the_reviewed_npm_command() -> None:
    """The reviewed lockfile install is the only npm command in workflows.

    npm accepts option abbreviations, aliases, and a changing command grammar.
    An exact argv comparison keeps those forms from silently broadening what
    runs in a workflow. Any new npm use needs explicit review here.
    """
    for workflow_name in _workflow_names():
        for step in _all_steps(_load(workflow_name)):
            for argv in _npm_commands(step.get("run", "")):
                assert argv == REVIEWED_NPM_COMMAND, f"{workflow_name}: unreviewed npm command: {' '.join(argv)}"


def test_the_pinned_fern_cli_version_matches_the_fern_config() -> None:
    """The Fern config, manifest, and lockfile describe one reviewed CLI tree.

    If they drift, docs may be validated or published with a different CLI or
    additional packages that were not reviewed for the token-bearing job.
    """
    manifest = json.loads((ROOT / "fern" / "package.json").read_text(encoding="utf-8"))
    fern_config = json.loads((ROOT / "fern" / "fern.config.json").read_text(encoding="utf-8"))
    lockfile = json.loads((ROOT / "fern" / "package-lock.json").read_text(encoding="utf-8"))

    declared = manifest["dependencies"]["fern-api"]
    assert declared == fern_config["version"], "fern/package.json and fern/fern.config.json disagree"
    assert re.fullmatch(r"\d+\.\d+\.\d+", declared), f"pin an exact version, not {declared!r}"
    assert manifest["dependencies"] == {"fern-api": declared}, "review new direct Fern dependencies"
    root_package = lockfile["packages"][""]
    assert root_package["dependencies"] == manifest["dependencies"], "lockfile root is stale"
    for dependency_group in ("devDependencies", "optionalDependencies", "peerDependencies"):
        assert not manifest.get(dependency_group), f"review new {dependency_group} in the Fern manifest"
        assert not root_package.get(dependency_group), f"review new {dependency_group} in the Fern lockfile"
    assert lockfile["packages"]["node_modules/fern-api"]["version"] == declared, "lockfile is stale"
    for path, package in lockfile["packages"].items():
        if not path:
            continue
        assert package.get("resolved", "").startswith("https://registry.npmjs.org/"), path
        assert package.get("integrity", "").startswith("sha512-"), path

    # NVIDIA's Fern organization rejects the authenticated docs-publish path below this
    # version (a review comment on PR #126 surfaced the "Org 'nvidia' requires Fern CLI
    # >= 5.106.0" error); `fern check` alone does not exercise that authenticated path,
    # so nothing else here would have caught a stale pin.
    minimum_required = (5, 106, 0)
    assert tuple(int(part) for part in declared.split(".")) >= minimum_required, (
        f"fern-api {declared} is older than the {'.'.join(map(str, minimum_required))} NVIDIA's org requires"
    )


@pytest.mark.parametrize("mutation", ["extra_dependency", "root_drift", "missing_integrity"])
def test_fern_pin_guard_rejects_unreviewed_dependency_changes(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_fern = ROOT / "fern"
    isolated_fern = tmp_path / "fern"
    isolated_fern.mkdir()
    for name in ("package.json", "package-lock.json", "fern.config.json"):
        (isolated_fern / name).write_bytes((source_fern / name).read_bytes())

    manifest_path = isolated_fern / "package.json"
    lockfile_path = isolated_fern / "package-lock.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lockfile = json.loads(lockfile_path.read_text(encoding="utf-8"))
    if mutation == "extra_dependency":
        manifest["dependencies"]["extra-package"] = "1.0.0"
    elif mutation == "root_drift":
        lockfile["packages"][""]["dependencies"]["fern-api"] = "0.0.0"
    else:
        del lockfile["packages"]["node_modules/fern-api"]["integrity"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    lockfile_path.write_text(json.dumps(lockfile), encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)

    with pytest.raises(AssertionError):
        test_the_pinned_fern_cli_version_matches_the_fern_config()
