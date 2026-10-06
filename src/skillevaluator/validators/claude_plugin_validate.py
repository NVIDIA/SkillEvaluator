# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in parity check against Claude Code's own plugin validator.

When the plugin's selected manifest is ``.claude-plugin/plugin.json`` and the
``claude`` CLI is on ``PATH``, ``claude plugin validate <root> --json`` is run
once against the plugin root (``--json`` needs Claude Code v2.1.259 or later;
older builds are rerun without it and their text report is parsed). It runs
without ``--strict``: the comparison is about what Claude Code refuses to load,
and its warnings (which ``--strict`` also fails) are still listed, with the
verdict ``--strict`` would give recorded as ``strict_verdict``.
Codex, Cursor, Agent Plugins, and bundle-reference plugins are not Claude Code
plugins, so the check is recorded as not applicable for them.

``validate`` only reads the plugin: it installs, enables, and fetches nothing.
The subprocess still runs with the auto-updater and non-essential traffic
disabled, from an empty temporary working directory, with a temporary HOME and
Claude config directory, and with an environment built from a short allowlist
(``PATH``, locale, temp dir, terminal, proxy and CA settings, and where Volta,
asdf, mise, or nvm keep their tools), so no credential from the caller's
environment reaches it.

The Tier 1 result records Claude Code's verdict, errors, and warnings next to
SkillEvaluator's own manifest verdict, with file paths relative to the plugin
root. The two are compared field by field as well as by verdict: they agree
only when the verdicts match, every manifest field Claude Code reports an error
on has a blocking SkillEvaluator finding, and every manifest field
SkillEvaluator's schema check fails is one Claude Code also reports. Claude
Code's errors are MEDIUM and its warnings LOW (advisory; a policy overlay can
raise them), and a disagreement is an INFO finding that names the fields. Without the CLI the check is skipped with a reason; a CLI that crashes,
times out, or prints a report with no verdict leaves the check INCOMPLETE.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

from skillevaluator.constants import PLUGIN_CONTAINED_MANIFEST_TYPE
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.plugin_manifest import PluginManifestPathError, locate_plugin_manifest
from skillevaluator.utils.redaction import redact_sensitive_text
from skillevaluator.utils.tool_runner import ExternalTool, Tools, parse_json_output
from skillevaluator.validators.mcp_static import CATEGORY as MCP_CATEGORY

CATEGORY = "PLUGIN_PARITY"
VALIDATOR_NAME = "Claude Plugin Validate Parity"
VALIDATOR_DESCRIPTION = "Compare SkillEvaluator's plugin manifest findings with `claude plugin validate`"
SCAN_NAME = "claude plugin validate"
TIMEOUT_SECONDS = 120
MAX_MESSAGES = 50
MAX_MESSAGE_CHARS = 300
PLUGIN_ROOT_LABEL = "<plugin-root>"
# The only caller variables the child sees. Everything else, including every
# credential, provider, and cloud variable, is dropped. HOME and the Claude
# config directory are set to a temporary directory instead.
_CHILD_ENV_NAMES = frozenset(
    {
        "PATH",
        "LANG",
        "LANGUAGE",
        "TMPDIR",
        "TMP",
        "TEMP",
        "TERM",
        "NO_PROXY",
        "no_proxy",
        "NODE_EXTRA_CA_CERTS",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        # Windows needs these to start a process at all.
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
    }
)
_CHILD_ENV_PREFIXES = ("LC_",)
# Version managers (Volta, asdf, mise, nvm) put a claude or node shim on PATH. The
# shim finds the real binary through these variables, or under HOME when they are
# unset. The child's HOME is a throwaway directory, so these are passed on and,
# when unset, filled from the caller's HOME. ASDF_DIR (where asdf itself lives) is
# passed on only when set: asdf works it out from its own path otherwise.
_VERSION_MANAGER_ENV_NAMES = ("VOLTA_HOME", "ASDF_DIR", "ASDF_DATA_DIR", "MISE_DATA_DIR", "NVM_DIR")
# asdf (and mise, for asdf users) read the global version pins from this file in
# HOME, so it is copied into the throwaway HOME. mise's own global config is not:
# it can set environment variables for every tool it runs.
_VERSION_PIN_FILES = (".tool-versions",)
_VERSION_PIN_MAX_BYTES = 64 * 1024
# Proxy URLs are passed with any user:password@ part removed.
_CHILD_PROXY_ENV_NAMES = frozenset({"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"})
_PROXY_USERINFO_RE = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)?[^/\s]*@")
_VERDICT_RE = re.compile(r"Validation (passed|failed)[^\n]*")
# "Validating plugin manifest: /abs/plugin/.claude-plugin/plugin.json" starts the report for one file.
_FILE_HEADER_RE = re.compile(r"^Validating [^:]+: (?P<file>.+)$")
# "Found 3 errors:" / "Found 1 warning:" opens a section whose lines are the messages.
# The CLI prints the header, a blank line, the messages, and a blank line. After
# that come the file's notes: message lines with no header of their own.
_SECTION_RE = re.compile(r"^Found \d+ (?P<kind>error|warning)s?:?$", re.IGNORECASE)
# The heavy right-pointing angle (and the single angle quote) that starts each message line.
_MESSAGE_MARKERS = "\u276f\u203a"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Status glyphs and bullets that prefix report lines (escaped to keep the source ASCII):
# heavy/ballot X, multiplication sign, cross mark, warning sign, check marks, bullet,
# and the heavy right-pointing angle that marks each message.
_BULLETS = "\u2718\u2717\u00d7\u274c\u26a0\u2714\u2713\u2022\u276f\u203a*- "
_UNKNOWN_OPTION_RE = re.compile(r"(?i)(unknown option|unexpected argument|unrecognized)[^\n]*--json")
# The top-level manifest field a report path or message names: "name", "dependencies.0", "skills[1]", ...
_FIELD_RE = re.compile(r"^(?P<field>[A-Za-z_$][\w$-]*)(?:[.\[][^:]*)?$")
_TEXT_FIELD_RE = re.compile(r"^(?:(?:error|warning)\s*:\s*)?(?P<path>[A-Za-z_$][\w$.\[\]-]*)\s*:", re.IGNORECASE)
# Report paths that mean the whole manifest (it is not JSON, not an object, or not readable).
_ROOT_PATHS = frozenset({"", "root", "json", "file"})
ROOT_FIELD = "<root>"
# SkillEvaluator findings about the whole selected manifest.
_MANIFEST_ROOT_CHECKS = frozenset(
    {
        "manifest_invalid_json",
        "manifest_not_object",
        "manifest_complexity_limit",
        "manifest_unsafe",
        "manifest_unreadable",
        "manifest_outside_root",
        "manifest_linked",
        "manifest_hardlinked",
    }
)
# Component path findings start with the manifest field they are about: "'skills' path ...", or a
# path into it: "'commands['x'].source' path ...".
_COMPONENT_FIELD_RE = re.compile(r"^'(?P<field>[A-Za-z_$][\w$]*)['.\[]")


def _strip_userinfo(value: str) -> str:
    return _PROXY_USERINFO_RE.sub(lambda match: match.group("scheme") or "", value, count=1)


def _caller_home(source: Mapping[str, str]) -> Path | None:
    value = source.get("HOME") or (source.get("USERPROFILE") if os.name == "nt" else None)
    return Path(value) if value else None


def _version_manager_env(source: Mapping[str, str]) -> dict[str, str]:
    """Where the caller's version managers live, so a shim on PATH still finds the real claude."""
    env = {name: source[name] for name in _VERSION_MANAGER_ENV_NAMES if source.get(name)}
    home = _caller_home(source)
    if home is None:
        return env
    data_home = Path(source["XDG_DATA_HOME"]) if source.get("XDG_DATA_HOME") else home / ".local" / "share"
    defaults = {
        "VOLTA_HOME": home / ".volta",
        "ASDF_DATA_DIR": home / ".asdf",
        "MISE_DATA_DIR": data_home / "mise",
        "NVM_DIR": home / ".nvm",
    }
    for name, default in defaults.items():
        if name not in env and default.is_dir():
            env[name] = str(default)
    return env


def _copy_version_pins(home: Path, *, environ: Mapping[str, str] | None = None) -> None:
    """Copy the caller's global version pins (``~/.tool-versions``) into the throwaway *home*."""
    caller_home = _caller_home(os.environ if environ is None else environ)
    if caller_home is None:
        return
    for name in _VERSION_PIN_FILES:
        source = caller_home / name
        try:
            if source.is_file() and source.stat().st_size <= _VERSION_PIN_MAX_BYTES:
                (home / name).write_bytes(source.read_bytes())
        except OSError:
            continue  # best effort: without the pins a shim fails and the check is INCOMPLETE


def _child_env(home: Path, *, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the child environment from an allowlist, with HOME and the Claude config in *home*."""
    source = os.environ if environ is None else environ
    env: dict[str, str] = {}
    for key, value in source.items():
        if key in _CHILD_ENV_NAMES or key.startswith(_CHILD_ENV_PREFIXES):
            env[key] = value
        elif key in _CHILD_PROXY_ENV_NAMES:
            env[key] = _strip_userinfo(value)
    env.update(_version_manager_env(source))
    env["HOME"] = str(home)
    env["CLAUDE_CONFIG_DIR"] = str(home / ".claude")
    if os.name == "nt":
        env["USERPROFILE"] = str(home)
    env["DISABLE_AUTOUPDATER"] = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return env


def _root_forms(root: Path | str | None) -> tuple[str, ...]:
    """Lexical and resolved spellings of the plugin root (Claude Code prints resolved paths)."""
    if root is None:
        return ()
    lexical = os.path.abspath(root)  # noqa: PTH100 - lexical spelling the CLI was given
    resolved = os.path.realpath(root)
    return tuple(dict.fromkeys(form.rstrip("/\\") or form for form in (lexical, resolved)))


def _relative_file(file: str, roots: tuple[str, ...]) -> str:
    """Report a file the CLI names relative to the plugin root, never as an absolute path."""
    for root in roots:
        if file == root:
            return PLUGIN_ROOT_LABEL
        for separator in ("/", "\\"):
            if file.startswith(root + separator):
                return file[len(root) + 1 :].replace("\\", "/")
    if Path(file).is_absolute() or PureWindowsPath(file).is_absolute():
        # Outside the plugin root: keep only the file name, never the absolute path.
        return PureWindowsPath(file).name or PLUGIN_ROOT_LABEL
    return file


def _relativize_text(text: str, roots: tuple[str, ...]) -> str:
    for root in sorted(roots, key=len, reverse=True):
        text = text.replace(root + "/", "").replace(root + "\\", "").replace(root, PLUGIN_ROOT_LABEL)
    return text


def _top_field(path: str) -> str | None:
    """The top-level manifest field a Claude Code report path names (``"<root>"`` for the whole file)."""
    path = path.strip()
    if path in _ROOT_PATHS:
        return ROOT_FIELD
    match = _FIELD_RE.match(path)
    return match.group("field") if match else None


def _is_manifest_file(file: str | None) -> bool:
    """Whether a report section is about ``plugin.json`` (a section with no file is the manifest too)."""
    return file is None or file.replace("\\", "/").rsplit("/", 1)[-1] == "plugin.json"


def _item_field(item: Any) -> str | None:
    if isinstance(item, dict):
        where = item.get("path") or item.get("field")
        return _top_field(str(where)) if isinstance(where, str) else ROOT_FIELD
    match = _TEXT_FIELD_RE.match(str(item).strip())
    return _top_field(match.group("path")) if match else None


def _message(item: Any, file: str | None, roots: tuple[str, ...] = ()) -> str:
    if isinstance(item, dict):
        text = str(item.get("message") or item.get("msg") or item.get("error") or item)
        where = item.get("path") or item.get("field") or item.get("file")
        if where:
            text = f"{_relative_file(str(where), roots)}: {text}"
    else:
        text = str(item)
    if file:
        text = f"{_relative_file(file, roots)}: {text}"
    return redact_sensitive_text(" ".join(_relativize_text(text, roots).split()), max_len=MAX_MESSAGE_CHARS)


def parse_json_report(data: dict[str, Any], *, root: Path | str | None = None) -> dict[str, Any]:
    """Normalize ``claude plugin validate --json`` output into verdict, errors, and warnings.

    File paths in the messages are made relative to *root*. ``manifest_found`` is
    False only when Claude Code says it found no plugin manifest
    (``"manifest": null``); a report with no ``manifest`` key says nothing about it.
    """
    roots = _root_forms(root)
    errors: list[str] = []
    warnings: list[str] = []
    error_fields: set[str] = set()
    sections: list[tuple[Any, bool]] = []
    if isinstance(data.get("manifest"), dict):
        sections.append((data["manifest"], True))
    if isinstance(data.get("contents"), list):
        sections.extend((section, False) for section in data["contents"])
    for section, is_manifest in sections:
        if not isinstance(section, dict):
            continue
        file = section.get("file") if isinstance(section.get("file"), str) else None
        for item in section.get("errors") or []:
            errors.append(_message(item, file, roots))
            if is_manifest and (field := _item_field(item)) is not None:
                error_fields.add(field)
        for item in section.get("warnings") or []:
            warnings.append(_message(item, file, roots))
    success = data.get("success")
    return {
        "claude_verdict": "passed" if success is True else "failed" if success is False else "unknown",
        "errors": errors,
        "warnings": warnings,
        "error_fields": sorted(error_fields),
        "format": "json",
        "manifest_found": not ("manifest" in data and data["manifest"] is None),
    }


def parse_text_report(stdout: str, stderr: str, *, root: Path | str | None = None) -> dict[str, Any]:
    """Parse the human-readable report (Claude Code builds without ``--json``).

    The report names each file on a ``Validating <kind>: <path>`` line, opens a
    ``Found N error(s):`` or ``Found N warning(s):`` section, and lists one
    message per line under it. The first blank line after the messages closes
    the section. Message lines outside a section are notes, which the JSON
    report lists apart and this check ignores. Other lines outside a section
    that start with "error" or "warning" are kept, for builds that print no
    sections.
    """
    roots = _root_forms(root)
    text = _ANSI_RE.sub("", f"{stdout}\n{stderr}")
    verdicts = _VERDICT_RE.findall(text)
    verdict = verdicts[-1] if verdicts else "unknown"
    found: dict[str, list[str]] = {"error": [], "warning": []}
    error_fields: set[str] = set()
    section: str | None = None
    section_messages = 0
    file: str | None = None
    for raw in _ANSI_RE.sub("", stdout).splitlines():
        stripped = raw.strip()
        line = stripped.lstrip(_BULLETS).strip()
        if not line:
            if section is not None and section_messages:
                section = None  # the blank line after the messages; notes may follow
            continue
        if _VERDICT_RE.match(line):
            section = None
            continue
        if header := _FILE_HEADER_RE.match(line):
            file, section = header.group("file").strip(), None
            continue
        if opened := _SECTION_RE.match(line):
            section, section_messages = opened.group("kind").lower(), 0
            continue
        if section is not None:
            found[section].append(_message(line, file, roots))
            section_messages += 1
            if section == "error" and _is_manifest_file(file) and (field := _item_field(line)) is not None:
                error_fields.add(field)
            continue
        if stripped[:1] in _MESSAGE_MARKERS:
            continue  # a note: a message line under no error or warning section
        lowered = line.lower()
        if lowered.startswith("error") or " error" in lowered[:40]:
            found["error"].append(_message(line, None, roots))
            if _is_manifest_file(file) and (field := _item_field(line)) is not None:
                error_fields.add(field)
        elif lowered.startswith("warning") or " warning" in lowered[:40]:
            found["warning"].append(_message(line, None, roots))
    return {
        "claude_verdict": verdict,
        "errors": found["error"],
        "warnings": found["warning"],
        "error_fields": sorted(error_fields),
        "format": "text",
    }


def claude_plugin_applicability(root: Path) -> tuple[str, str] | None:
    """Return ``(status, reason)`` when the parity check cannot apply to *root*, else ``None``.

    ``claude plugin validate`` checks only ``.claude-plugin/plugin.json``, so any
    other selected manifest is ``not_applicable``. A manifest that cannot be
    located safely is an ``error``: the check cannot tell what it would validate.
    """
    try:
        located = locate_plugin_manifest(Path(root))
    except PluginManifestPathError as exc:
        return "error", redact_sensitive_text(
            f"Cannot locate the plugin manifest safely: {exc}", max_len=MAX_MESSAGE_CHARS
        )
    if located is None:
        return "not_applicable", "No supported plugin manifest was found, so there is nothing for Claude Code to check."
    if located.manifest_type != PLUGIN_CONTAINED_MANIFEST_TYPE:
        return "not_applicable", (
            f"Not a Claude Code plugin: the selected manifest is {located.manifest_filename}. "
            "claude plugin validate checks only .claude-plugin/plugin.json."
        )
    return None


class ClaudePluginValidateParity:
    """Run ``claude plugin validate`` and compare it with SkillEvaluator's manifest verdict and findings."""

    def __init__(self, tool: ExternalTool | None = None) -> None:
        self.tool = tool if tool is not None else Tools.claude

    @property
    def name(self) -> str:
        return VALIDATOR_NAME

    @property
    def description(self) -> str:
        return VALIDATOR_DESCRIPTION

    def validate(
        self,
        root: Path,
        *,
        skillevaluator_verdict: str | None = None,
        skillevaluator_fields: Mapping[str, Any] | None = None,
    ) -> ValidationResult:
        """Run Claude Code's validator once and compare it with SkillEvaluator.

        ``skillevaluator_fields`` comes from :func:`skillevaluator_manifest_fields`:
        the manifest fields with a blocking SkillEvaluator finding (``blocking``),
        and the subset its manifest schema check fails (``schema``). Without it
        only the verdicts are compared.
        """
        result = ValidationResult(validator_name=VALIDATOR_NAME, validator_description=VALIDATOR_DESCRIPTION)
        parity: dict[str, Any] = {
            "command": f"claude plugin validate {PLUGIN_ROOT_LABEL} --json",
            "strict": False,
            "skillevaluator_verdict": skillevaluator_verdict or "unknown",
        }
        result.metadata["plugin"] = {"validator_parity": parity}
        applicability = claude_plugin_applicability(Path(root))
        if applicability is not None and applicability[0] == "error":
            parity.update(status="error", reason=applicability[1])
            result.add_warning(f"claude plugin validate did not run: {applicability[1]}")
            result.mark_scan_incomplete(SCAN_NAME)
            return result
        if applicability is not None:
            return self._not_applicable(result, parity, applicability[1])
        if not self.tool.is_available:
            reason = f"claude CLI not found on PATH; parity check skipped. {self.tool.get_install_hint()}"
            parity.update(status="skipped", reason=reason)
            result.metadata["skipped"] = True
            result.add_message(reason)
            return result

        report = self._run(Path(root), parity)
        if report is None:
            result.add_warning(f"claude plugin validate did not produce a usable report: {parity.get('reason')}")
            result.mark_scan_incomplete(SCAN_NAME)
            return result
        if report.get("manifest_found") is False:
            # Claude Code found no .claude-plugin manifest, so it validated nothing.
            return self._not_applicable(
                result, parity, "Claude Code found no .claude-plugin/plugin.json, so it validated nothing."
            )

        errors: list[str] = report["errors"]
        warnings: list[str] = report["warnings"]
        claude_verdict = report["claude_verdict"]
        known = {"passed", "failed"}
        strict_verdict = ("failed" if errors or warnings else "passed") if claude_verdict in known else claude_verdict
        parity.update(
            status="compared",
            format=report["format"],
            claude_verdict=claude_verdict,
            strict_verdict=strict_verdict,
            error_count=len(errors),
            warning_count=len(warnings),
            errors=errors[:MAX_MESSAGES],
            warnings=warnings[:MAX_MESSAGES],
        )
        for message in errors[:MAX_MESSAGES]:
            result.add_finding(
                Finding(
                    category=CATEGORY,
                    severity=Severity.MEDIUM,
                    check_name="claude_validate_error",
                    message=f"claude plugin validate: {message}",
                    file_path=PLUGIN_ROOT_LABEL,
                    suggestion="Fix the reported problem; Claude Code reports the same error when it loads the plugin.",
                )
            )
        for message in warnings[:MAX_MESSAGES]:
            result.add_finding(
                Finding(
                    category=CATEGORY,
                    severity=Severity.LOW,
                    check_name="claude_validate_warning",
                    message=f"claude plugin validate warning (fails --strict): {message}",
                    file_path=PLUGIN_ROOT_LABEL,
                    suggestion="Fix the warning; claude plugin validate --strict treats it as an error in CI.",
                )
            )
        if claude_verdict in known and skillevaluator_verdict in known:
            details = self._compare_fields(report, skillevaluator_fields, parity)
            agree = claude_verdict == skillevaluator_verdict and not details
            parity["agree"] = agree
            if not agree:
                lead = (
                    f"claude plugin validate {claude_verdict} the plugin, but SkillEvaluator's manifest and "
                    f"component checks {skillevaluator_verdict} it"
                    if claude_verdict != skillevaluator_verdict
                    else f"claude plugin validate and SkillEvaluator both {claude_verdict} the plugin, for different "
                    "reasons"
                )
                result.add_finding(
                    Finding(
                        category=CATEGORY,
                        severity=Severity.INFO,
                        check_name="claude_validate_disagreement",
                        message="; ".join([lead, *details]),
                        file_path=PLUGIN_ROOT_LABEL,
                        suggestion=(
                            "Compare the two reports: SkillEvaluator adds security and dependency checks Claude Code "
                            "does not run, and a field one side fails and the other passes is a parity gap."
                        ),
                        metadata={
                            "claude_verdict": claude_verdict,
                            "skillevaluator_verdict": skillevaluator_verdict,
                            **parity.get("fields", {}),
                        },
                    )
                )
        else:
            parity["agree"] = None
        result.add_message(
            f"claude plugin validate: {claude_verdict} ({len(errors)} error(s), {len(warnings)} warning(s); "
            f"--strict: {strict_verdict}); SkillEvaluator: {skillevaluator_verdict or 'unknown'}"
        )
        return result

    @staticmethod
    def _compare_fields(
        report: Mapping[str, Any], skillevaluator_fields: Mapping[str, Any] | None, parity: dict[str, Any]
    ) -> list[str]:
        """Field-level differences between the two sides, as message fragments (empty when they match).

        Claude Code's manifest errors are compared with the fields that have a
        blocking SkillEvaluator finding; SkillEvaluator's manifest schema errors
        are compared with the fields Claude Code reports. Other blocking
        SkillEvaluator findings (MCP policy, dependencies, ...) only decide the
        verdict, because Claude Code does not run those checks.
        """
        if skillevaluator_fields is None:
            return []
        claude_fields = set(report.get("error_fields") or ())
        blocking = {str(field) for field in skillevaluator_fields.get("blocking") or ()}
        schema = {str(field) for field in skillevaluator_fields.get("schema") or ()}
        claude_only = sorted(claude_fields - blocking)
        skillevaluator_only = sorted(schema - claude_fields)
        parity["fields"] = {
            "claude_error_fields": sorted(claude_fields),
            "skillevaluator_error_fields": sorted(blocking),
            "claude_only": claude_only,
            "skillevaluator_only": skillevaluator_only,
        }
        details = []
        if claude_only:
            details.append(
                f"Claude Code reports errors on {', '.join(claude_only[:10])}, which SkillEvaluator does not block on"
            )
        if skillevaluator_only:
            details.append(
                f"SkillEvaluator's manifest check fails {', '.join(skillevaluator_only[:10])}, which Claude Code "
                "accepts"
            )
        return details

    @staticmethod
    def _not_applicable(result: ValidationResult, parity: dict[str, Any], reason: str) -> ValidationResult:
        parity.update(status="not_applicable", reason=reason, agree=None)
        result.metadata["skipped"] = True
        result.add_message(f"claude plugin validate: not applicable. {reason}")
        return result

    def _run(self, root: Path, parity: dict[str, Any]) -> dict[str, Any] | None:
        target = str(Path(os.path.abspath(root)))  # noqa: PTH100 - lexical, never resolved
        with tempfile.TemporaryDirectory(
            prefix="skillevaluator-claude-validate-", ignore_cleanup_errors=True
        ) as scratch:
            # An empty working directory, and a throwaway HOME and Claude config
            # directory: the CLI writes its state there, never into the caller's.
            cwd = Path(scratch) / "work"
            home = Path(scratch) / "home"
            cwd.mkdir()
            (home / ".claude").mkdir(parents=True)
            _copy_version_pins(home)
            env = _child_env(home)
            run = self.tool.run(
                ["plugin", "validate", target, "--json"],
                cwd=cwd,
                timeout=TIMEOUT_SECONDS,
                env=env,
                replace_env=True,
            )
            if run.error_message:
                parity.update(status="error", reason=run.error_message)
                return None
            data = parse_json_output(run.stdout)
            if isinstance(data, dict) and ("success" in data or "contents" in data):
                parity["exit_code"] = run.exit_code
                if not isinstance(data.get("success"), bool):
                    parity.update(status="error", reason="the JSON report has no success verdict")
                    return None
                if "manifest" not in data and not (isinstance(data.get("contents"), list) and data["contents"]):
                    # Not "manifest": null (nothing to validate): the report just does not say what it checked.
                    parity.update(status="error", reason="the JSON report names no manifest and no validated files")
                    return None
                return parse_json_report(data, root=root)
            if _UNKNOWN_OPTION_RE.search(f"{run.stdout}\n{run.stderr}"):
                # Claude Code before v2.1.259 has no --json; parse the text report instead.
                parity["command"] = f"claude plugin validate {PLUGIN_ROOT_LABEL}"
                run = self.tool.run(
                    ["plugin", "validate", target],
                    cwd=cwd,
                    timeout=TIMEOUT_SECONDS,
                    env=env,
                    replace_env=True,
                )
                if run.error_message:
                    parity.update(status="error", reason=run.error_message)
                    return None
        parity["exit_code"] = run.exit_code
        if run.exit_code not in (0, 1):
            detail = (run.stderr or "").strip().splitlines()
            parity.update(
                status="error",
                reason=redact_sensitive_text(
                    detail[-1] if detail else f"exit code {run.exit_code}", max_len=MAX_MESSAGE_CHARS
                ),
            )
            return None
        report = parse_text_report(run.stdout, run.stderr, root=root)
        if report["claude_verdict"] == "unknown":
            # Neither a JSON report nor a "Validation passed/failed" line: the
            # exit code alone does not say what was validated, so do not guess.
            parity.update(status="error", reason="the output had no JSON report and no Validation passed/failed line")
            return None
        return report


def skillevaluator_manifest_verdict(results: list[ValidationResult], schema_validator_name: str) -> str | None:
    """SkillEvaluator's own verdict from the plugin schema result: ``passed``, ``failed``, or ``None``."""
    for result in results:
        if result.validator_name != schema_validator_name:
            continue
        if isinstance(result.metadata, dict) and result.metadata.get("security_failure"):
            return "failed"
        blocking = any(finding.severity.is_error() for finding in result.findings)
        return "failed" if blocking else "passed"
    return None


def _is_claude_manifest_path(path: str | None) -> bool:
    """Whether a finding's file is the Claude Code manifest, ``.claude-plugin/plugin.json``."""
    return bool(path) and str(path).replace("\\", "/").rsplit("/", 2)[-2:] == [".claude-plugin", "plugin.json"]


def _finding_field(finding: Finding) -> str | None:
    """The top-level field of the selected Claude Code manifest a SkillEvaluator finding is about, if any."""
    metadata = finding.metadata if isinstance(finding.metadata, dict) else {}
    if metadata.get("manifest_type") not in (None, PLUGIN_CONTAINED_MANIFEST_TYPE):
        return None
    field = metadata.get("field")
    if isinstance(field, str) and field:
        return _top_field(field)
    if finding.check_name in _MANIFEST_ROOT_CHECKS:
        return ROOT_FIELD
    if finding.category == MCP_CATEGORY and _is_claude_manifest_path(finding.file_path):
        # A server declared inline in the manifest: Claude Code reports its problems under mcpServers.
        return "mcpServers"
    if "plugin_component_ref" in metadata and (match := _COMPONENT_FIELD_RE.match(finding.message or "")):
        return match.group("field")
    return None


def skillevaluator_manifest_fields(
    results: list[ValidationResult], schema_validator_name: str
) -> dict[str, Any] | None:
    """The manifest fields SkillEvaluator fails, for the field-level parity comparison.

    ``blocking`` lists every top-level manifest field with a CRITICAL or HIGH
    plugin-schema finding (the manifest schema check and the component path
    checks); ``schema`` lists the ones the manifest schema check itself fails,
    which claim that Claude Code refuses the field. ``None`` when the schema
    check did not run.
    """
    for result in results:
        if result.validator_name != schema_validator_name:
            continue
        blocking: set[str] = set()
        schema: set[str] = set()
        for finding in result.findings:
            if not finding.severity.is_error() or (field := _finding_field(finding)) is None:
                continue
            blocking.add(field)
            if finding.check_name.startswith(("schema:", "plugin_manifest_", "manifest_")):
                schema.add(field)
        return {"blocking": sorted(blocking), "schema": sorted(schema)}
    return None
