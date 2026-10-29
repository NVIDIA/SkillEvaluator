# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private static policy for component configurations, independent of inventory/report building."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any

from skillevaluator.models.result import Finding, Severity
from skillevaluator.plugin_component_risk import (
    MAX_QUOTED_ENTRIES,
    MAX_SCRIPT_BYTES,
    MONITOR_EVENT,
    MONITOR_HOOKS,
    PERMISSIVE_PERMISSION_MODES,
    HookAnalysis,
    HookAnalyzer,
    HookScriptUnreadable,
    hook_dialect,
)
from skillevaluator.plugin_formats import PROFILES, FormatProfile
from skillevaluator.plugin_paths import PluginRootReader, _plugin_finding
from skillevaluator.plugin_permissions import OverrideIssue, allowed_tools_flag_issues, is_broad_allow_rule
from skillevaluator.utils.secure_fs import SecurePathError
from skillevaluator.validators.mcp_static import (
    env_override_issues,
    env_tls_and_secret_issues,
    permission_flag_issues,
    validate_mcp_command,
    validate_mcp_pinning,
)
from skillevaluator.validators.url_policy import report_text


def _override_findings(
    issues: Iterable[OverrideIssue],
    file_path: str,
    *,
    where: str,
    component: tuple[str, str] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    metadata = {"plugin_component": {"type": component[0], "name": component[1]}} if component else None
    for issue in issues:
        findings.append(
            _plugin_finding(
                issue.severity,
                f"plugin_{issue.concept}",
                f"{where}: {issue.message}",
                file_path,
                issue.suggestion,
                metadata=dict(metadata) if metadata else None,
            )
        )
    return findings


def _agent_cli_flag_issues(value: Any) -> list[OverrideIssue]:
    """Agent-CLI flags in a hook, LSP, monitor, or settings config: permission bypasses, the permissive
    ``--permission-mode acceptEdits|auto`` (one :func:`permission_flag_issues` walk), and ``--allowedTools Bash``."""
    return [*permission_flag_issues(value), *allowed_tools_flag_issues(value)]


def _lsp_command_findings(name: str, server: dict[str, Any], file_path: str) -> list[Finding]:
    """The MCP stdio command-form checks for one LSP server's ``command`` and ``args``, as ``plugin_lsp_*``.

    Claude Code starts an LSP server like a stdio MCP server (argv, no shell),
    so a shell ``-c`` program, inline credentials, and unpinned package runners
    get the same findings, worded for an LSP server. Without a shell, ``;`` or
    ``|`` inside an argument is passed to the server literally and is not
    flagged (the MCP argv rule); an argument that is only an operator or
    carries command substitution is a LOW note, not the MCP CRITICAL.
    """
    if not isinstance(server.get("command"), str):
        return []
    raw: list[Finding] = []
    validate_mcp_command(name, server, file_path, raw)
    validate_mcp_pinning(name, server, file_path, raw)
    prefix = f"mcpServers['{name}']: "
    findings: list[Finding] = []
    for finding in raw:
        check = f"plugin_lsp_{finding.check_name.removeprefix('mcp_')}"
        severity, message, suggestion = finding.severity, finding.message.removeprefix(prefix), finding.suggestion
        if check == "plugin_lsp_command_shell_metacharacters":
            if any(item.check_name == check for item in findings):
                continue
            severity = Severity.LOW
            message = (
                "an argument contains shell metacharacters (; | & ` $() < >); Claude Code starts LSP servers "
                "without a shell, so they reach the server literally"
            )
            suggestion = "Check that the argument is meant literally; the server does not get a shell."
        findings.append(
            _plugin_finding(
                severity,
                check,
                f"lspServers['{name}']: {_lsp_wording(message)}",
                file_path,
                _lsp_wording(suggestion) if suggestion else suggestion,
                metadata={"plugin_component": {"type": "lsp", "name": name}},
            )
        )
    return findings


def _lsp_wording(text: str) -> str:
    """MCP check text reworded for an LSP server."""
    return (
        text.replace("runnable MCP ", "LSP server ")
        .replace("MCP commands run", "LSP servers run")
        .replace("MCP server", "LSP server")
    )


def _lsp_env_findings(name: str, env: Any, file_path: str) -> list[Finding]:
    """The MCP env checks for an LSP server's ``env``: TLS verification off and inline credentials (CRITICAL)."""
    return _override_findings(
        env_tls_and_secret_issues(env), file_path, where=f"lspServers['{name}']", component=("lsp", name)
    )


def _settings_findings(config: dict[str, Any], rel: str, display: str) -> list[Finding]:
    """The checks on shipped Claude Code settings: a settings file, or the manifest's inline ``settings``."""
    findings: list[Finding] = []
    permissions = config.get("permissions")
    if isinstance(permissions, dict):
        mode = permissions.get("defaultMode")
        if mode == "bypassPermissions":
            findings.append(
                _plugin_finding(
                    Severity.HIGH,
                    "plugin_settings_bypass_permissions",
                    f"shipped settings '{rel}' sets permissions.defaultMode to 'bypassPermissions', which "
                    "disables every tool-approval prompt",
                    display,
                    "Remove permissions.defaultMode (Claude Code ignores it from plugins, but shipping it is a "
                    "red flag and it applies if the file is copied into a project).",
                )
            )
        elif isinstance(mode, str) and mode in PERMISSIVE_PERMISSION_MODES:
            findings.append(
                _plugin_finding(
                    Severity.MEDIUM,
                    "plugin_settings_permission_mode",
                    f"shipped settings '{rel}' sets permissions.defaultMode to {mode!r}, which approves some "
                    "tool calls (file edits, or what a classifier allows) without a prompt",
                    display,
                    "Remove permissions.defaultMode; let the user choose the permission mode (Claude Code "
                    "ignores it from plugins, but it applies if the file is copied into a project).",
                )
            )
        allow = permissions.get("allow")
        if isinstance(allow, list):
            broad = sorted(
                {
                    str(rule)
                    for rule in allow  # every rule: already bounded by the structured-data limits
                    if isinstance(rule, str) and is_broad_allow_rule(rule)
                }
            )
            if broad:
                findings.append(
                    _plugin_finding(
                        Severity.HIGH,
                        "plugin_settings_broad_allow",
                        f"shipped settings '{rel}' pre-approves unrestricted tools or interpreter Bash rules "
                        f"that run any command {broad[:MAX_QUOTED_ENTRIES]}",
                        display,
                        "Remove blanket allow rules such as Bash / Bash(*) / Bash(python3:*); scope permissions "
                        "to exact commands.",
                    )
                )
    if config.get("enableAllProjectMcpServers") is True:
        findings.append(
            _plugin_finding(
                Severity.MEDIUM,
                "plugin_settings_auto_approve",
                f"shipped settings '{rel}' sets enableAllProjectMcpServers, auto-approving every project MCP server",
                display,
                "Remove enableAllProjectMcpServers; let users approve MCP servers explicitly.",
            )
        )
    enabled = config.get("enabledMcpjsonServers")
    if isinstance(enabled, list) and enabled:
        names = ", ".join(report_text(str(name), 40) for name in enabled[:8])
        findings.append(
            _plugin_finding(
                Severity.MEDIUM,
                "plugin_settings_auto_approve",
                f"shipped settings '{rel}' lists enabledMcpjsonServers ({names}), auto-approving those project "
                "MCP servers",
                display,
                "Remove enabledMcpjsonServers; let users approve MCP servers explicitly.",
            )
        )
    allow = permissions.get("allow") if isinstance(permissions, dict) else None
    if isinstance(allow, list) and any(
        isinstance(rule, str) and "".join(rule.split()).lower() in {"mcp__*", "mcp__*__*"} for rule in allow
    ):
        findings.append(
            _plugin_finding(
                Severity.MEDIUM,
                "plugin_settings_auto_approve",
                f"shipped settings '{rel}' pre-approves every MCP tool (mcp__*) in permissions.allow",
                display,
                "Remove the mcp__* allow rule; pre-approve only the MCP tools the plugin needs, by name.",
            )
        )
    findings.extend(_override_findings(env_override_issues(config.get("env")), display, where=rel))
    # Claude Code ignores env from plugin settings, but it applies if the file is copied into a project: HIGH.
    findings.extend(
        _override_findings(env_tls_and_secret_issues(config.get("env"), severity=Severity.HIGH), display, where=rel)
    )
    findings.extend(_override_findings(_agent_cli_flag_issues(config), display, where=rel))
    return findings


def _nested_hook_groups(config: Any) -> Any:
    """Return a hooks config in the nested shape the hook risk analyzer reads.

    Cursor, and Agent Plugins client extensions such as ``com.cursor/`` and
    ``com.github.copilot/``, list handlers directly under each event
    (``{"hooks": {"stop": [{"command": ...}]}}``); Claude Code and Codex nest
    them in matcher groups (``{"matcher": ..., "hooks": [{"type": "command",
    ...}]}``). Each flat handler becomes a one-handler group with its
    ``matcher``; the handler ``type`` defaults to ``command``, as in Cursor. A
    group (an entry with a ``hooks`` list) is left unchanged, so every format's
    config passes through here.
    """
    events = config.get("hooks") if isinstance(config, dict) and isinstance(config.get("hooks"), dict) else config
    if not isinstance(events, dict):
        return config
    nested: dict[str, Any] = {}
    for event, handlers in events.items():
        if not isinstance(handlers, list):
            nested[event] = handlers
            continue
        groups: list[Any] = []
        for handler in handlers:
            if isinstance(handler, dict) and not isinstance(handler.get("hooks"), list):
                inner = {key: value for key, value in handler.items() if key != "matcher"}
                inner.setdefault("type", "command")
                group: dict[str, Any] = {"hooks": [inner]}
                if "matcher" in handler:
                    group["matcher"] = handler["matcher"]
                groups.append(group)
            else:
                groups.append(handler)
        nested[event] = groups
    return {"hooks": nested}


def read_hook_script(reader: PluginRootReader, rel: PurePosixPath) -> str | None:
    """Text of a script a hook runs from the plugin root (``${CLAUDE_PLUGIN_ROOT}``), for the hook risk analyzer.

    Returns the whole file (at most ``MAX_SCRIPT_BYTES``) of a regular file,
    hard-linked or not, decoded as UTF-8 with replacement characters (so one
    non-UTF-8 byte does not hide the script), or ``None`` when nothing or a
    directory is at ``rel``. Raises :class:`HookScriptUnreadable` when
    something is there that cannot be read safely: a link, a special file, a
    file over the size limit, or an exhausted read budget.
    """
    kind = reader.kind(rel, allow_hard_links=True)
    if kind in {"missing", "dir"}:
        return None
    if kind == "link":
        raise HookScriptUnreadable(
            f"'{rel.as_posix()}' is (or passes through) a symlink or reparse point; it was not followed"
        )
    if kind != "file":
        raise HookScriptUnreadable(f"'{rel.as_posix()}' is a special file, or cannot be inspected")
    try:
        raw = reader.read_script_bytes(rel, MAX_SCRIPT_BYTES)
    except (SecurePathError, OSError) as exc:
        raise HookScriptUnreadable(f"'{rel.as_posix()}' could not be read safely: {exc}") from exc
    return raw.decode("utf-8-sig", errors="replace")


def analyze_hook_config(
    analyzer: HookAnalyzer, config: Any, *, source: str, file: str, display: str, manifest_type: str | None = None
) -> HookAnalysis:
    """Analyze a source using its native flat/nested hook dialect without executing it."""
    result = analyzer.analyze(
        _nested_hook_groups(config),
        source=source,
        file=file,
        display=display,
        dialect=hook_dialect(manifest_type, file),
    )
    result.findings[:0] = _override_findings(
        _agent_cli_flag_issues(config), display, where=f"hooks ({source})", component=("hook", source)
    )
    return result


def analyze_lsp_server(name: str, config: Any, display: str) -> list[Finding]:
    """Preserve LSP-specific argv, pinning, environment and permission checks."""
    component = ("lsp", name)
    findings = _override_findings(
        _agent_cli_flag_issues(config), display, where=f"lspServers['{name}']", component=component
    )
    if isinstance(config, dict):
        findings.extend(
            _override_findings(
                env_override_issues(config.get("env")), display, where=f"lspServers['{name}']", component=component
            )
        )
        findings.extend(_lsp_env_findings(name, config.get("env"), display))
        findings.extend(_lsp_command_findings(name, config, display))
    return findings


def analyze_settings(config: Any, path: str, display: str, *, name: str | None = None) -> list[Finding]:
    """Check a bounded parsed Claude settings object, with source-row attribution."""
    findings = _settings_findings(config, path, display) if isinstance(config, dict) else []
    for finding in findings:
        finding.metadata["plugin_component"] = {"type": "settings", "name": name or path}
    return findings


def analyze_codex_settings(config: Any, path: str, display: str) -> list[Finding]:
    """Check project/profile approval and sandbox settings without loading Codex."""
    if not isinstance(config, dict):
        return []
    findings: list[Finding] = []
    tables = [("", config)]
    profiles = config.get("profiles")
    if isinstance(profiles, dict):
        tables.extend((f"profiles.{key}.", value) for key, value in list(profiles.items())[:64])
    for prefix, table in tables:
        if not isinstance(table, dict):
            continue
        for key, value in (("approval_policy", "never"), ("sandbox_mode", "danger-full-access")):
            if isinstance(table.get(key), str) and table[key].strip().lower() == value:
                effect = (
                    "turns off every approval prompt" if key == "approval_policy" else "turns off the command sandbox"
                )
                findings.append(
                    _plugin_finding(
                        Severity.HIGH,
                        "plugin_settings_bypass_permissions",
                        f"shipped Codex config {path!r} sets {prefix}{key} = {value!r}, which {effect}",
                        display,
                        f"Remove {prefix}{key}; let the user choose Codex's approval policy and sandbox.",
                        metadata={"plugin_component": {"type": "settings", "name": path}},
                    )
                )
    findings.extend(
        _override_findings(_agent_cli_flag_issues(config), display, where=path, component=("settings", path))
    )
    return findings


def analyze_monitor(analyzer: HookAnalyzer, name: str, config: Any, file: str, display: str) -> HookAnalysis:
    """Analyze an unsandboxed monitor command and its permissions, without starting it."""
    result = HookAnalysis()
    if isinstance(config, dict) and isinstance(config.get("command"), str):
        handler: dict[str, Any] = {"type": "command", "command": config["command"]}
        if isinstance(config.get("args"), list):
            handler["args"] = config["args"]
        result = analyzer.analyze(
            {MONITOR_EVENT: [{"hooks": [handler]}]},
            source=f"monitor:{name}",
            file=file,
            display=display,
            dialect=MONITOR_HOOKS,
        )
        for finding in result.findings:
            finding.metadata["plugin_component"] = {"type": "monitor", "name": name}
    result.findings.extend(
        _override_findings(
            _agent_cli_flag_issues(config), display, where=f"monitors entry '{name}'", component=("monitor", name)
        )
    )
    return result


def create_hook_analyzer(
    reader: PluginRootReader,
    profile: FormatProfile,
    *,
    hook_allowed_urls: Iterable[str] = (),
    allowed_private_hosts: Iterable[str] = (),
) -> HookAnalyzer:
    """Bind bounded script reads and native path semantics to an independent analyzer.

    Agent Plugins extensions may use any supported client's root placeholder;
    Cursor and Copilot namespace scripts also resolve relative to their hooks file.
    """
    prefixes, relative = profile.root_prefixes, profile.relative_hook_scripts
    if profile.client_extensions:
        prefixes = tuple(dict.fromkeys(prefix for known in PROFILES.values() for prefix in known.root_prefixes))
        relative = any(known.relative_hook_scripts for known in PROFILES.values())
    return HookAnalyzer(
        read_script=lambda rel: read_hook_script(reader, rel),
        hook_allowed_urls=hook_allowed_urls,
        allowed_private_hosts=allowed_private_hosts,
        root_prefixes=prefixes,
        relative_scripts=relative,
    )
