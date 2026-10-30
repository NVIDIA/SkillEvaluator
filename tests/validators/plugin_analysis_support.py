# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only composition of private discovery and policy engines.

This does not register or stand in for the public schema validator. Records and
findings below are real engine outputs from bounded files on disk; report
formatting, severity overlays, scores and runtime integration are deferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillevaluator.models.result import Finding, Severity
from skillevaluator.plugin_component_risk import (
    HookAnalyzer,
    analyze_agent,
    analyze_command,
    analyze_skill,
    hook_risk_summary,
    mcp_server_is_read_only,
    privilege_summary,
)
from skillevaluator.plugin_config_risk import (
    analyze_codex_settings,
    analyze_hook_config,
    analyze_lsp_server,
    analyze_monitor,
    analyze_settings,
    create_hook_analyzer,
)
from skillevaluator.plugin_discovery import discover_plugin_members
from skillevaluator.plugin_formats import profile_for
from skillevaluator.plugin_paths import PluginRootReader
from skillevaluator.validators.mcp_static import validate_mcp_server_declaration


@dataclass
class AnalysisOptions:
    hook_allowed_urls: tuple[str, ...] = ()
    mcp_allowed_private_hosts: tuple[str, ...] = ()


@dataclass
class Analysis:
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_no_blocking_findings(self) -> bool:
        return not any(f.severity in {Severity.HIGH, Severity.CRITICAL} for f in self.findings)


def analyze_plugin(root: Path, policy: AnalysisOptions | None = None) -> Analysis:
    options = policy or AnalysisOptions()
    discovery = discover_plugin_members(root)
    assert discovery is not None
    reader = PluginRootReader(root)
    result = Analysis(list(discovery.findings))
    hooks, privileges = [], []
    analyzers: dict[str, HookAnalyzer] = {}
    write_mcp = [
        m.name
        for m in discovery.components
        if m.mcp and m.effective and m.mcp.runnable and not mcp_server_is_read_only(m.config)
    ]
    for member in discovery.components:
        profile = profile_for(member.manifest_type)
        if member.manifest_type not in analyzers:
            analyzers[member.manifest_type] = create_hook_analyzer(
                reader,
                profile,
                hook_allowed_urls=options.hook_allowed_urls,
                allowed_private_hosts=options.mcp_allowed_private_hosts,
            )
        analyzer = analyzers[member.manifest_type]
        path = member.path or member.declared_by
        display = reader.display(path)
        if member.type == "hook":
            analysis = analyze_hook_config(
                analyzer,
                member.config,
                source=member.name,
                file=path.removesuffix("#hooks"),
                display=display,
                manifest_type=member.manifest_type,
            )
            hooks.extend(analysis.records)
            result.findings.extend(analysis.findings)
        elif member.type == "monitor":
            analysis = analyze_monitor(analyzer, member.name, member.config, path, display)
            hooks.extend(analysis.records)
            result.findings.extend(analysis.findings)
        elif member.type in {"agent", "command", "skill"}:
            if member.problem:
                continue
            if member.type == "agent":
                record, findings = analyze_agent(
                    member.name,
                    member.path,
                    member.frontmatter,
                    display,
                    write_capable_mcp=write_mcp,
                    plugin_name="demo",
                )
            elif member.type == "command":
                record, findings = analyze_command(
                    member.name,
                    member.path,
                    member.frontmatter,
                    display,
                    entry=member.config,
                    client="codex" if profile.codex_path_rules else None,
                )
            else:
                record, findings = analyze_skill(
                    member.name,
                    member.path,
                    member.frontmatter,
                    display,
                    client="codex" if profile.codex_path_rules else None,
                )
            privileges.append(record)
            result.findings.extend(findings)
        elif member.type == "lsp":
            result.findings.extend(analyze_lsp_server(member.name, member.config, display))
        elif member.type == "settings":
            if path == ".codex/config.toml":
                result.findings.extend(analyze_codex_settings(member.config, path, display))
            else:
                result.findings.extend(analyze_settings(member.config, path, display, name=member.name))
        elif member.mcp:
            result.findings.extend(
                validate_mcp_server_declaration(
                    member.name,
                    member.config,
                    display,
                    allowed_private_hosts=options.mcp_allowed_private_hosts,
                    manifest_type=member.manifest_type,
                )
            )
    result.metadata["plugin"] = {"hook_risk": hook_risk_summary(hooks), "privileges": privilege_summary(privileges)}
    return result
