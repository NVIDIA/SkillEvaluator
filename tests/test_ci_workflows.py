# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

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
DOCS_ONLY_IF = "${{ needs.classify-changes.outputs.docs_only == 'true' }}"
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


def test_ci_docs_lane_uses_the_required_python_312_context() -> None:
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
    assert node_step["if"] == DOCS_ONLY_IF
    assert len(node_step["uses"].split("@", 1)[1]) == 40
    assert docs_step["if"] == DOCS_ONLY_IF
    assert "npm ci --prefix fern --ignore-scripts --omit=optional" in docs_step["run"]
    assert "./fern/node_modules/.bin/fern check" in docs_step["run"]
    assert "GITHUB_STEP_SUMMARY" in docs_step["run"]


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
    install_command = next(
        (
            argv
            for argv in install_commands
            if _npm_subcommands(argv) & NPM_INSTALL_SUBCOMMANDS
            and not _npm_subcommands(argv) & NPM_REGISTRY_SUBCOMMANDS
        ),
        None,
    )

    assert install_command is not None, "publish-docs.yml no longer installs the Fern CLI from the lockfile"
    assert _npm_flag_active(install_command, "ignore-scripts")
    assert not any(token.startswith("fern-api@") for token in install_command), (
        "the version belongs in fern/package.json"
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
NPM_EXECUTABLE = re.compile(r"^(?:.*[/\\])?npm(?:\.cmd|\.exe)?$")
NPM_MENTION = re.compile(r"""(?:^|[\s'"`(;&|])(?:[\w./\\-]*[/\\])?npm(?:\.cmd|\.exe)?(?=$|[\s'"`);&|])""")
SHELL_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
SHELL_OPERATOR = re.compile(r"&&|\|\||[;&|()]")
SHELL_WRAPPERS = frozenset({"command", "env", "exec", "ionice", "nice", "setsid", "stdbuf", "sudo", "time"})
SHELL_KEYWORDS = frozenset({"!", "do", "elif", "else", "if", "then", "until", "while", "{"})
NPM_INSTALL_SUBCOMMANDS = frozenset({"add", "ci", "cit", "i", "install", "install-ci-test", "install-test", "it"})
NPM_REGISTRY_SUBCOMMANDS = NPM_INSTALL_SUBCOMMANDS - {"ci", "cit", "install-ci-test"}
NPM_OTHER_SUBCOMMANDS = frozenset(
    {
        "audit",
        "cache",
        "config",
        "docs",
        "help",
        "list",
        "ls",
        "outdated",
        "ping",
        "prefix",
        "root",
        "run",
        "run-script",
        "test",
        "version",
        "view",
        "whoami",
        "why",
    }
)
# `npm --version` names no subcommand at all, and is no reason to fail a build.
NPM_SELF_REPORTING_FLAGS = frozenset({"-h", "-v", "--help", "--version"})


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


def _npm_subcommands(argv: Sequence[str]) -> set[str]:
    """Every npm subcommand named in ``argv``.

    npm's own options can take a value (``npm --prefix fern install``), so the
    subcommand is not simply the first non-flag token. Every recognised name is
    collected instead, and a command naming more than one is held to the
    strictest guard that applies -- a value that happens to read as a
    subcommand can only make this stricter, never blinder.
    """
    known = NPM_INSTALL_SUBCOMMANDS | NPM_OTHER_SUBCOMMANDS | NPM_SELF_REPORTING_FLAGS
    found = {token for token in argv[1:] if token in known}
    assert found, f"npm invocation naming no subcommand these guards know: {' '.join(argv)}"
    return found


def _npm_flag_active(argv: Sequence[str], flag: str) -> bool:
    """Whether ``--flag`` is in effect on an npm command's argv.

    A boolean npm flag can be switched back off three ways after it appears to
    be set: ``--no-<flag>``, ``--<flag>=false``, and ``--<flag> false`` -- npm
    consumes a following literal ``true``/``false`` as the flag's value, which
    running ``npm --ignore-scripts false config get ignore-scripts`` confirms.
    A substring check for ``--<flag>`` cannot tell any of those from the flag
    being on, which is the bypass this exists to close. Anything else npm
    rejects outright (``--<flag>=0`` exits with its usage text), so this reads
    only the two literals it accepts and treats every other value as off --
    failing the guard rather than trusting a form npm will not run. When the
    flag appears more than once npm applies the last occurrence, so this does too.
    """
    state = False
    for index, token in enumerate(argv):
        name, separator, value = token.partition("=")
        if not name.startswith("--"):
            continue
        negated = name.startswith("--no-")
        if (name[len("--no-") :] if negated else name[len("--") :]) != flag:
            continue
        if negated:
            state = False
        elif separator:
            state = value == "true"
        else:
            state = (argv[index + 1] if index + 1 < len(argv) else "") != "false"
    return state


def test_npm_flag_active_rejects_negated_and_false_valued_flags() -> None:
    assert _npm_flag_active(shlex.split("npm ci --ignore-scripts"), "ignore-scripts")
    assert _npm_flag_active(shlex.split("npm ci --ignore-scripts=true"), "ignore-scripts")
    assert not _npm_flag_active(shlex.split("npm ci --ignore-scripts=false"), "ignore-scripts")
    assert not _npm_flag_active(shlex.split("npm ci --ignore-scripts=0"), "ignore-scripts")
    assert not _npm_flag_active(shlex.split("npm ci --no-ignore-scripts"), "ignore-scripts")
    assert not _npm_flag_active(shlex.split("npm ci"), "ignore-scripts")
    assert not _npm_flag_active(shlex.split("npm install --package-lock-only=false"), "package-lock-only")


def test_npm_flag_active_reads_the_space_separated_value_npm_accepts() -> None:
    """``npm ci --ignore-scripts false`` really does turn the flag off."""
    assert not _npm_flag_active(shlex.split("npm ci --ignore-scripts false"), "ignore-scripts")
    assert _npm_flag_active(shlex.split("npm ci --ignore-scripts true"), "ignore-scripts")
    assert _npm_flag_active(shlex.split("npm ci --ignore-scripts --omit=optional"), "ignore-scripts")
    assert not _npm_flag_active(shlex.split("npm install --package-lock-only false"), "package-lock-only")


def test_npm_flag_active_is_not_fooled_by_a_look_alike_flag_name() -> None:
    assert not _npm_flag_active(shlex.split("npm ci --ignore-scripts-if-untrusted"), "ignore-scripts")


def test_npm_flag_active_takes_the_last_occurrence_like_npm_does() -> None:
    assert _npm_flag_active(shlex.split("npm ci --no-ignore-scripts --ignore-scripts"), "ignore-scripts")
    assert not _npm_flag_active(shlex.split("npm ci --ignore-scripts --no-ignore-scripts"), "ignore-scripts")


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
    assert not _npm_flag_active(quoted[0], "ignore-scripts"), "a quoted argument must not truncate the command"


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
        "$(which npm) ci --ignore-scripts",
        "npm ci --ignore-scripts 'unbalanced",
    ):
        with pytest.raises(AssertionError):
            _npm_commands(run)


def test_npm_subcommands_fails_closed_on_a_subcommand_it_does_not_know() -> None:
    assert _npm_subcommands(shlex.split("npm --prefix fern install")) == {"install"}
    assert _npm_subcommands(shlex.split("npm ci --prefix fern")) == {"ci"}
    assert _npm_subcommands(shlex.split("npm --version")) == {"--version"}
    with pytest.raises(AssertionError):
        _npm_subcommands(shlex.split("npm exec --yes some-package"))


def test_an_option_value_reading_as_a_subcommand_cannot_pass_for_a_lockfile_install() -> None:
    """``npm install --prefix ci`` names ``ci`` without being one.

    Collecting every recognised name keeps the install guards strict, but a
    check that merely asks whether ``ci`` is in that set would accept this as
    the reviewed, lockfile-pinned install it is not.
    """
    argv = shlex.split("npm install --prefix ci fern-api")
    subcommands = _npm_subcommands(argv)

    assert subcommands == {"ci", "install"}
    assert subcommands & NPM_REGISTRY_SUBCOMMANDS, "a registry install must stay visible as one"


def test_every_workflow_npm_command_ignores_lifecycle_scripts() -> None:
    """A dependency must not get to run install-time code in a CI job.

    ``--ignore-scripts`` is the only half of this that a global install honours,
    so it is asserted on every npm install regardless of how the tree is resolved.
    """
    for workflow_name in _workflow_names():
        for step in _all_steps(_load(workflow_name)):
            for argv in _npm_commands(step.get("run", "")):
                if _npm_subcommands(argv) & NPM_INSTALL_SUBCOMMANDS:
                    assert _npm_flag_active(argv, "ignore-scripts"), f"{workflow_name}: {' '.join(argv)}"


def test_no_workflow_resolves_a_node_dependency_tree_from_the_registry() -> None:
    """Only ``npm ci`` against the committed lockfile may install into a job.

    ``npm install`` re-resolves every transitive dependency from semver ranges on
    each run, so pinning the top-level version pins nothing beneath it. ``npm ci``
    installs exactly the versions and integrity hashes in ``fern/package-lock.json``.
    """
    for workflow_name in _workflow_names():
        for step in _all_steps(_load(workflow_name)):
            for argv in _npm_commands(step.get("run", "")):
                if not _npm_subcommands(argv) & NPM_REGISTRY_SUBCOMMANDS:
                    continue
                if not _npm_flag_active(argv, "package-lock-only"):
                    raise AssertionError(f"{workflow_name}: use `npm ci` against the lockfile, not `{' '.join(argv)}`")


def test_the_pinned_fern_cli_version_matches_the_fern_config() -> None:
    """``fern/package.json`` and ``fern/fern.config.json`` both name a CLI version.

    They are two declarations of one fact. If they drift, docs are validated and
    published by a different CLI than the one Fern itself is configured for.
    """
    manifest = json.loads((ROOT / "fern" / "package.json").read_text(encoding="utf-8"))
    fern_config = json.loads((ROOT / "fern" / "fern.config.json").read_text(encoding="utf-8"))
    lockfile = json.loads((ROOT / "fern" / "package-lock.json").read_text(encoding="utf-8"))

    declared = manifest["dependencies"]["fern-api"]
    assert declared == fern_config["version"], "fern/package.json and fern/fern.config.json disagree"
    assert re.fullmatch(r"\d+\.\d+\.\d+", declared), f"pin an exact version, not {declared!r}"
    assert lockfile["packages"]["node_modules/fern-api"]["version"] == declared, "lockfile is stale"

    # NVIDIA's Fern organization rejects the authenticated docs-publish path below this
    # version (a review comment on PR #126 surfaced the "Org 'nvidia' requires Fern CLI
    # >= 5.106.0" error); `fern check` alone does not exercise that authenticated path,
    # so nothing else here would have caught a stale pin.
    minimum_required = (5, 106, 0)
    assert tuple(int(part) for part in declared.split(".")) >= minimum_required, (
        f"fern-api {declared} is older than the {'.'.join(map(str, minimum_required))} NVIDIA's org requires"
    )
