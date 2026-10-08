# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown reporter for PR comments and documentation.

This reporter produces Markdown output suitable for:
- GitHub pull request comments
- Documentation wikis
- Slack/Teams messages (with markdown support)
- README files

The output is optimized for readability in code review contexts.
"""

from __future__ import annotations

import html
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from skillevaluator.evidence import evidence_ref_identity
from skillevaluator.reporting.base import (
    ReporterBase,
    additional_errors,
    is_advisory_agent_eval_skip,
    passes_required_gate,
)
from skillevaluator.reporting.harbor_viewer import (
    harbor_evidence_link_text,
    normalize_harbor_viewer_for_display,
    safe_url,
)
from skillevaluator.reporting.plugin_sections import format_score, tier3_plugin_view
from skillevaluator.utils.rich_markup import strip_terminal_controls

if TYPE_CHECKING:
    from skillevaluator.models import Finding, ValidationResult


# Inline Markdown syntax that untrusted text could use for links, images or emphasis,
# escaped as narrowly as possible so ordinary text such as "run(s)", "agent_plugin"
# or "hooks[0]" stays readable in the raw file:
# - a backslash before ASCII punctuation (it would cancel the escapes below);
# - asterisks and tildes, which mark emphasis and strikethrough;
# - "_" unless it sits between two letters or digits (intraword "_" never emphasizes);
# - "(" or "[" right after "]": an inline or reference link needs that pair, and a
#   cell cannot start a line, so it cannot define a reference.
_MARKDOWN_INLINE_SPECIAL = re.compile(r"\\(?=[!-/:-@\[-`{-~])|[*~]|(?<=\])[\[(]|(?<![^\W_])_|_(?![^\W_])")


def _markdown_table_cell(value: object) -> str:
    """Return untrusted text as inert Markdown that fits in one table cell.

    Terminal control sequences are removed, Unicode format characters are
    shown as escapes, HTML is escaped, and link, image and emphasis syntax is
    backslash-escaped. The result is for plain text and ``<code>`` elements:
    Markdown shows backslash escapes and entities literally inside a backtick
    code span, so never wrap it in backticks.
    """
    normalized = strip_terminal_controls(str(value)).replace("\r\n", "\n").replace("\r", "\n")
    escaped = html.escape(normalized, quote=False)
    escaped = _MARKDOWN_INLINE_SPECIAL.sub(lambda match: "\\" + match.group(0), escaped)
    return escaped.replace("|", "&#124;").replace("`", "&#96;").replace("\n", "<br>")


def _markdown_output(text: str) -> str:
    """Final pass over a rendered report.

    Untrusted text may carry lone surrogates (which UTF-8 cannot encode) and
    terminal escape sequences (which run when someone prints the file).
    """
    return strip_terminal_controls(text)


def _related_paths(finding: Finding) -> list[str]:
    """Return distinct path-like string values carried in finding metadata."""
    metadata = finding.metadata if isinstance(finding.metadata, dict) else {}
    paths: list[str] = []
    for key, value in metadata.items():
        normalized_key = str(key).casefold()
        if not (normalized_key == "path" or normalized_key.startswith("path_") or normalized_key.endswith("_path")):
            continue
        if isinstance(value, str) and value and value not in paths:
            paths.append(value)
    return paths


def _table_header(*headers: str) -> list[str]:
    """Return a Markdown table's header line and its separator line."""
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("-" * (len(header) + 2) for header in headers) + "|",
    ]


# ---------------------------------------------------------------------------
# Plugin sections, one function per subsection (the HTML template's macros)
# ---------------------------------------------------------------------------

_PLUGIN_STATUS_MARKERS = {"failed": "❌ FAILED", "incomplete": "⚠️ INCOMPLETE", "passed": "✅ PASSED"}


def _plugin_overview(view: dict, lines: list[str]) -> None:
    cell = _markdown_table_cell
    lines.append("## Plugin")
    lines.append("")
    lines.extend(_table_header("Plugin", "Status", "Manifest", "Mode"))
    lines.append(
        f"| {cell(view['name'] or 'plugin')} | {_PLUGIN_STATUS_MARKERS.get(view['status'], '')} "
        f"| {cell(view['manifest_type'] or 'unknown')} | {cell(view['plugin_mode'] or 'unknown')} |"
    )
    lines.append("")
    _manifest_declarations(view.get("manifest_declarations"), lines)
    if view["declared_dependencies"]:
        declared = ", ".join(f"{cell(row['kind'])}={row['count']}" for row in view["declared_dependencies"])
        lines.append(f"**Declared dependencies:** {declared}")
        lines.append("")
    if view["bundled_skills"]:
        bundled = ", ".join(cell(name) for name in view["bundled_skills"])
        more = f" (+{view['bundled_skills_omitted']} more)" if view["bundled_skills_omitted"] else ""
        lines.append(f"**Bundled skills ({view['in_plugin_skills'] or len(view['bundled_skills'])}):** {bundled}{more}")
        lines.append("")


def _manifest_declarations(declarations: dict | None, lines: list[str]) -> None:
    if not declarations:
        return
    cell = _markdown_table_cell
    lines.append("### Plugin manifests")
    lines.append("")
    lines.extend(_table_header("Manifest", "Type", "Selected", "Status", "Name", "Version"))
    for row in declarations["rows"]:
        lines.append(
            f"| {cell(row['manifest_filename'])} | {cell(row['manifest_type'])} "
            f"| {'yes' if row['selected'] else 'no'} | {cell(row['status'])} "
            f"| {cell(row['name'])} | {cell(row['version'])} |"
        )
    lines.append("")
    for conflict in declarations["conflicts"]:
        lines.append(
            f"> ⚠️ **Manifest conflict:** {cell(conflict['manifest_filename'])} declares {cell(conflict['field'])} "
            f"<code>{cell(conflict['additional'])}</code>; the selected manifest declares "
            f"<code>{cell(conflict['selected'])}</code>."
        )
        lines.append("")
    lines.append(f"*{cell(declarations['note'])}*")
    lines.append("")


def _dependency_resolution(dependencies: dict | None, lines: list[str]) -> None:
    if not dependencies:
        return
    cell = _markdown_table_cell
    lines.append("### Dependency resolution")
    lines.append("")
    if dependencies["counts"]:
        counts = ", ".join(f"{cell(row['state'])}={row['count']}" for row in dependencies["counts"])
        lines.append(f"**Status counts:** {counts}")
        lines.append("")
    if dependencies["rows"]:
        lines.extend(_table_header("Kind", "Ref", "State", "Path", "Reason"))
        for row in dependencies["rows"]:
            lines.append(
                f"| {cell(row['kind'])} | {cell(row['ref'])} | {cell(row['state'])} "
                f"| {cell(row['path'] or '—')} | {cell(row['reason'])} |"
            )
        if dependencies["omitted"]:
            lines.append(f"| … | *{dependencies['omitted']} more refs* | | | |")
        lines.append("")


def _component_inventory(inventory: dict | None, lines: list[str]) -> None:
    if not inventory:
        return
    cell = _markdown_table_cell
    lines.append(f"### Component inventory ({inventory['total']})")
    lines.append("")
    if inventory.get("broken"):
        lines.append(
            f"> **{inventory['broken']} broken component(s):** the declaration is missing, escapes the "
            "plugin root, is a link, or is invalid, so the client cannot load it."
        )
        lines.append("")
    if inventory["unsupported_types"]:
        unsupported = ", ".join(cell(name) for name in inventory["unsupported_types"])
        lines.append(
            f"> ⚠️ **Unsupported component types present:** {unsupported}. {cell(inventory['unsupported_note'])}"
        )
        lines.append("")
    if inventory["rows"]:
        lines.extend(_table_header("Type", "Name", "Origin", "Support", "Findings"))
        for row in inventory["rows"]:
            lines.append(
                f"| {cell(row['type'])} | {cell(row['name'])} | {cell(row['origin'])} "
                f"| {cell(row['support_label'])} | {row['findings']} |"
            )
        if inventory["omitted"]:
            lines.append(f"| … | *{inventory['omitted']} more components* | | | |")
        lines.append("")


def _mcp_pinning(mcp: dict | None, lines: list[str]) -> None:
    if not (mcp and mcp["pinning"]):
        return
    cell = _markdown_table_cell
    pinning = mcp["pinning"]
    lines.append("### MCP pinning")
    lines.append("")
    lines.append(f"**Pinned:** {cell(pinning['summary'])} ({cell(pinning['ratio_label'])})")
    lines.append("")
    if mcp["unpinned"]:
        lines.extend(_table_header("Unpinned server", "Kind", "Detail"))
        for server in mcp["unpinned"]:
            lines.append(f"| {cell(server['name'])} | {cell(server['kind'])} | {cell(server['pin_detail'])} |")
        lines.append("")


def _context_cost(cost: dict | None, lines: list[str]) -> None:
    if not cost:
        return
    cell = _markdown_table_cell
    lines.append(f"### Context cost ({cell(cost['label'])})")
    lines.append("")
    lines.append(f"**Always-on:** {cost['always_on']} tokens · **On-demand:** {cost['on_demand']} tokens")
    lines.append("")
    lines.append(f"*{cell(cost['note'])}*")
    lines.append("")


def _catalog_similarity(similarity: dict | None, lines: list[str]) -> None:
    if not similarity:
        return
    cell = _markdown_table_cell
    lines.append(f"### {similarity['title']} (advisory)")
    lines.append("")
    summary = f" · {cell(similarity['summary'])}" if similarity["summary"] else ""
    lines.append(f"**Status:** {cell(similarity['status_label'])}{summary}")
    lines.append("")
    if similarity["matches"]:
        columns = similarity["columns"]
        lines.append("| " + " | ".join(column["label"] for column in columns) + " |")
        lines.append("|" + "|".join("---" for _ in columns) + "|")
        for match in similarity["matches"]:
            lines.append("| " + " | ".join(cell(match[column["key"]]) for column in columns) + " |")
        lines.append("")


def _privileges(privileges: dict | None, lines: list[str]) -> None:
    if not privileges:
        return
    cell = _markdown_table_cell
    lines.append(
        f"### Subagent, command, and skill privileges ({privileges['agents']} agents, "
        f"{privileges['commands']} commands, {privileges.get('skills', 0)} skills; "
        f"{privileges['flagged']} flagged)"
    )
    lines.append("")
    lines.extend(_table_header("Type", "Name", "Grants", "Model", "Permission mode", "Invocation", "Flags"))
    for row in privileges["rows"]:
        lines.append(
            f"| {cell(row['type'])} | {cell(row['name'])} | {cell(row['grants'] or '—')} "
            f"| {cell(row['model'] or '—')} | {cell(row['permission_mode'] or '—')} "
            f"| {cell(row['invocation'] or '—')} | {cell(', '.join(row['flags']) or '—')} |"
        )
    if privileges["omitted"]:
        lines.append(f"| … | *{privileges['omitted']} more* | | | | | |")
    lines.append("")


def _hook_risk(hooks: dict | None, lines: list[str]) -> None:
    if not hooks:
        return
    cell = _markdown_table_cell
    lines.append(f"### Hook risk ({hooks['total']} handlers; {hooks['flagged']} flagged)")
    lines.append("")
    if hooks["by_flag"]:
        flags = ", ".join(f"{cell(row['flag'])}={row['count']}" for row in hooks["by_flag"])
        lines.append(f"**Risk flags:** {flags}")
        lines.append("")
    lines.extend(_table_header("Event", "Matcher", "Handler", "Target", "Risk flags"))
    for row in hooks["rows"]:
        lines.append(
            f"| {cell(row['event'])} | {cell(row['matcher'])} | {cell(row['handler_type'])} "
            f"| {cell(row['target'] or '—')} | {cell(', '.join(row['flags']) or '—')} |"
        )
    if hooks["omitted"]:
        lines.append(f"| … | *{hooks['omitted']} more* | | | |")
    lines.append("")


def _cve_audit(cve: dict | None, lines: list[str]) -> None:
    if not cve:
        return
    cell = _markdown_table_cell
    lines.append("### Dependency CVE audit")
    lines.append("")
    lines.extend(_table_header("Ecosystem", "Status", "Scanner", "Audited", "Unverified", "Vulnerabilities"))
    for row in cve["rows"]:
        lines.append(
            f"| {cell(row['ecosystem'])} | {cell(row['status_label'])} | {cell(row['scanners'])} "
            f"| {row['audited']} | {row['unverified']} | {cell(row['severity_label'])} |"
        )
    lines.append("")
    for row in cve["rows"]:
        for error in row["errors"]:
            lines.append(f"> ⚠️ **{cell(row['ecosystem'])} audit incomplete:** {cell(error)}")
            lines.append("")


def _validator_parity(parity: dict | None, lines: list[str]) -> None:
    if not parity:
        return
    cell = _markdown_table_cell
    lines.append("### Claude plugin validate parity")
    lines.append("")
    if parity["status"] != "compared":
        lines.append(f"**Status:** {cell(parity['status_label'])} · {cell(parity['reason'])}")
        lines.append("")
        return
    lines.append(
        f"**claude plugin validate:** {cell(parity['claude_verdict'])} "
        f"({parity['error_count']} errors, {parity['warning_count']} warnings) · "
        f"**SkillEvaluator:** {cell(parity['skillevaluator_verdict'])} · "
        f"**Agreement:** {cell(parity['agreement'])}"
    )
    lines.append("")
    for error in parity["errors"]:
        lines.append(f"- error: {cell(error)}")
    if parity["errors_omitted"]:
        lines.append(f"- *(+{parity['errors_omitted']} more errors)*")
    for warning in parity["warnings"]:
        lines.append(f"- warning: {cell(warning)}")
    if parity["warnings_omitted"]:
        lines.append(f"- *(+{parity['warnings_omitted']} more warnings)*")
    if parity["errors"] or parity["warnings"]:
        lines.append("")


def _endpoint_checks(endpoints: dict | None, lines: list[str]) -> None:
    if not endpoints:
        return
    cell = _markdown_table_cell
    lines.append(f"### Endpoint DNS and redirect checks ({endpoints['total']} endpoints)")
    lines.append("")
    if not endpoints["rows"]:
        return
    lines.extend(_table_header("Kind", "Name", "URL", "Status", "Addresses", "HEAD", "Redirect"))
    for row in endpoints["rows"]:
        redirect = row["redirect"]
        if redirect and row["redirect_classification"]:
            redirect = f"{redirect} ({row['redirect_classification']})"
        lines.append(
            f"| {cell(row['kind'])} | {cell(row['name'])} | {cell(row['url'] or '—')} "
            f"| {cell(row['status'])} | {cell(row['addresses'] or '—')} | {cell(row['head'] or '—')} "
            f"| {cell(redirect or '—')} |"
        )
    if endpoints["omitted"]:
        lines.append(f"| … | *{endpoints['omitted']} more endpoints* | | | | | |")
    lines.append("")


def _component_coverage(coverage: dict | None, lines: list[str]) -> None:
    if not coverage:
        return
    cell = _markdown_table_cell
    lines.append("### Plugin Component Coverage")
    lines.append("")
    lines.append(f"**{cell(coverage['headline'])}** {cell(coverage['detail'])}.")
    lines.append("")
    lines.append(f"*{coverage['caveat']} {cell(coverage['note'])}*")
    lines.append("")
    unevaluated_rows = [*coverage["not_staged_rows"], *coverage.get("not_loaded_rows", [])]
    if unevaluated_rows:
        lines.extend(_table_header("Type", "Component", "State", "Reason"))
        for row in unevaluated_rows:
            lines.append(
                f"| {cell(row['type'])} | {cell(row['name'])} | {cell(row['state_label'])} | {cell(row['reason'])} |"
            )
        lines.append("")
    # The "Not evaluated" statement names only the first few of these.
    if coverage["staged_not_observed_rows"]:
        lines.append("Staged but not observed in any plugin trial:")
        lines.append("")
        lines.extend(
            f"- {cell(row['type'])} {cell(row['name'])} ({cell(row['observed'])})"
            for row in coverage["staged_not_observed_rows"]
        )
        lines.append("")


def _not_evaluated(statements: list[str], lines: list[str]) -> None:
    if not statements:
        return
    lines.append("**Not evaluated by this run:**")
    lines.append("")
    lines.extend(f"- {_markdown_table_cell(statement)}" for statement in statements)
    lines.append("")


def _plugin_loading(plugin_load: dict | None, lines: list[str]) -> None:
    if not plugin_load:
        return
    cell = _markdown_table_cell
    lines.append("### Plugin Loading")
    lines.append("")
    lines.append(f"**Requested:** <code>{cell(plugin_load['requested'])}</code>. *{cell(plugin_load['note'])}*")
    lines.append("")
    lines.extend(_table_header("Agent", "Mode", "Adapter", "Native", "Wrapper", "Unsupported", "Load census", "Reason"))
    for row in plugin_load["agents"]:
        census = row["census_summary"] if row["census"] else "none"
        lines.append(
            f"| {cell(row['agent'])} | {cell(row['mode'])} | {cell(row['adapter'])} "
            f"| {cell(', '.join(row['native']) or '-')} | {cell(', '.join(row['wrapper']) or '-')} "
            f"| {cell(', '.join(row['unsupported']) or '-')} | {cell(census)} | {cell(row['reason'])} |"
        )
    lines.append("")
    for row in plugin_load["agents"]:
        if row["unverified"]:
            lines.append(f"**INCOMPLETE:** {cell(row['unverified'])}")
            lines.append("")


def _integration(integration: dict | None, modes: dict | None, lines: list[str]) -> None:
    if not (integration or modes):
        return
    cell = _markdown_table_cell
    lines.append("### Integration (advisory)")
    lines.append("")
    if modes:
        lines.append(
            f"**Lift mode:** requested <code>{cell(modes['requested'])}</code>, "
            f"effective <code>{cell(modes['effective'])}</code>"
        )
        lines.append("")
    if not integration:
        return
    # One named Integration line per agent in a multi-agent run.
    per_agent = integration.get("per_agent")
    for entry in per_agent or [integration]:
        scope = f"{cell(entry['agent'])} — " if per_agent and entry.get("agent") else ""
        if entry["measured"]:
            # The interval only: the CI summary starts with the estimate, which
            # repeated the lift ("lift +0.12 +0.12 [...]").
            ci = entry["ci"]
            interval = f", {cell(ci['confidence'])} {cell(ci['interval'])}" if ci else ""
            point = f"; point estimate: {cell(entry['point_verdict_label'])}" if entry["point_verdict_label"] else ""
            lines.append(
                f"**{scope}{cell(entry['verdict_label'])}:** plugin {entry['with_plugin']} vs "
                f"sum-of-parts {entry['sum_of_parts']} (lift {entry['integration_lift']}{interval}{point})"
            )
            if entry["reason"]:
                lines.append(f"*{cell(entry['reason'])}*")
        else:
            lines.append(f"**{scope}INCONCLUSIVE:** {cell(entry['reason'])}")
        lines.append("")


def _lift_intervals(statistics: dict | None, lines: list[str]) -> None:
    if not statistics:
        return
    cell = _markdown_table_cell
    # A multi-agent run lists every agent's intervals, each named, next to the named Integration lines.
    scopes = statistics.get("scopes") or [statistics["primary"]]
    named = len(scopes) > 1
    for scope in scopes:
        prefix = f"{cell(scope['label'])} — " if named else ""
        for row in scope["lift_ci"]:
            warning = " — ⚠️ CI includes zero" if row["ci_includes_zero"] else ""
            partial = f" — {cell(row['partial_note'])}" if row.get("partial_note") else ""
            lines.append(
                f"- {prefix}{cell(row['label'])}: {cell(row['summary'])}, precision {cell(row['precision'])}"
                f"{warning}{partial}"
            )
    if any(scope["lift_ci"] for scope in scopes):
        lines.append("")


def _mcp_calls(signals: dict | None, lines: list[str]) -> None:
    """Render each arm's MCP call outcomes: succeeded, failed, and unknown, per server and tool."""
    entries = [entry for entry in (signals or {}).get("entries", []) if entry.get("mcp_calls")]
    if not entries:
        return
    cell = _markdown_table_cell
    lines.append("### MCP Calls (advisory)")
    lines.append("")
    lines.extend(
        _table_header("Agent · Arm", "Server", "Calls", "Succeeded", "Failed", "Unknown", "Success rate", "Tools")
    )
    for entry in entries:
        mcp = entry["mcp_calls"]
        scope = cell(f"{entry['scope']} · {entry['arm_label']}")
        rows = [
            ("all servers", mcp, ""),
            *((server["server"], server, server["tools"]) for server in mcp["servers"]),
        ]
        for server, counts, tools in rows:
            lines.append(
                f"| {scope} | {cell(server)} | {cell(counts['total'])} | {cell(counts['succeeded'])} "
                f"| {cell(counts['failed'])} | {cell(counts['unknown'])} | {cell(counts['success_rate'])} "
                f"| {cell(tools)} |"
            )
    lines.append("")
    lines.append("*The success rate is succeeded / (succeeded + failed); unknown calls are not in it.*")
    lines.append("")


def _canary(canary: dict | None, lines: list[str]) -> None:
    if not canary:
        return
    cell = _markdown_table_cell
    lines.append("### Canary Exfiltration")
    lines.append("")
    lines.append(f"_{cell(canary['note'])}_")
    lines.append("")
    for entry in canary["entries"]:
        lines.append(f"**{cell(entry['scope'])}:** {cell(entry['verdict'])}")
        lines.append("")
        for label, key in (("Credential reads", "credential_verdict"), ("Protected writes", "write_verdict")):
            if entry.get(key):
                lines.append(f"- {label}: {cell(entry[key])}")
        if entry.get("credential_verdict") or entry.get("write_verdict"):
            lines.append("")
        lines.extend(
            _table_header(
                "Arm", "Trials", "Planted", "Leaked", "Leak rate", "Sinks", "Credential reads", "Protected writes"
            )
        )
        for row in entry["rows"]:
            lines.append(
                f"| {cell(row['arm_label'])} | {cell(row['trials'])} | {cell(row['planted'])} | "
                f"{cell(row['leaked'])} | {cell(row['leak_rate'])} | {cell(row['sinks'])} | "
                f"{cell(row['credential_cell'])} | {cell(row['write_cell'])} |"
            )
        lines.append("")


def _hook_census(hook_census: dict | None, lines: list[str]) -> None:
    if not hook_census:
        return
    cell = _markdown_table_cell
    lines.append("### Plugin Hook Census (advisory)")
    lines.append("")
    for entry in hook_census["entries"]:
        lines.append(f"**{cell(entry['scope'])} · {cell(entry['arm_label'])}:** {cell(entry['summary'])}")
        lines.append("")
        if not entry["rows"]:
            continue
        lines.extend(
            _table_header("Hook", "Event", "Runs", "Blocked (exit 2)", "Failures", "Not started", "Failure rate")
        )
        for row in entry["rows"]:
            lines.append(
                f"| <code>{cell(row['hook_id'])}</code> | {cell(row['event'])} | {row['runs']} | {row['blocked']} | "
                f"{row['failures']} | {row['not_started']} | {cell(row['failure_rate'])} |"
            )
        lines.append("")


def _mcp_proof(mcp_proof: dict | None, lines: list[str]) -> None:
    if not mcp_proof:
        return
    cell = _markdown_table_cell
    lines.append("### MCP Proof (advisory)")
    lines.append("")
    lines.append(f"{cell(mcp_proof['headline'])}.")
    lines.append("")
    lines.extend(_table_header("Server", "Status", "Tools", "Detail"))
    for row in mcp_proof["rows"]:
        tools = ", ".join(row["tools"]) or "none"
        lines.append(f"| {cell(row['server'])} | {cell(row['status_label'])} | {cell(tools)} | {cell(row['detail'])} |")
    lines.append("")


class MarkdownReporter(ReporterBase):
    """Markdown report generator for PR comments.

    Produces clean, readable Markdown suitable for GitHub
    pull request comments and documentation.
    """

    def __init__(
        self,
        *,
        include_timestamp: bool = True,
        include_details: bool = True,
        max_findings_shown: int = 10,
    ) -> None:
        """Initialize Markdown reporter.

        Args:
            include_timestamp: Whether to include generation timestamp
            include_details: Whether to include expandable details sections
            max_findings_shown: Maximum findings to show per validator
        """
        self.include_timestamp = include_timestamp
        self.include_details = include_details
        self.max_findings_shown = max_findings_shown

    @property
    def name(self) -> str:
        return "markdown"

    @property
    def description(self) -> str:
        return "Markdown for PR comments and documentation"

    def render(self, result: ValidationResult) -> str:
        """Render single result to Markdown."""
        lines = []
        self._render_result(result, lines)
        return _markdown_output("\n".join(lines))

    def render_all(self, results: list[ValidationResult]) -> str:
        """Render all results to Markdown with summary."""
        lines = []

        # Header
        lines.append("# SkillEvaluator Validation Report")
        lines.append("")

        # Overall status
        all_passed = all(passes_required_gate(r) for r in results)
        has_incomplete = any(r.is_incomplete for r in results)
        advisory_skip_count = sum(1 for r in results if is_advisory_agent_eval_skip(r))
        status = "⚠️ INCOMPLETE" if has_incomplete else "✅ PASSED" if all_passed else "❌ FAILED"
        lines.append(f"**Status:** {status}")

        policy = next(
            (result.metadata.get("policy") for result in results if isinstance(result.metadata.get("policy"), dict)),
            None,
        )
        if policy is not None:
            lines.append(f"**Profile:** {policy.get('profile', 'external')}")
            if policy.get("digest"):
                lines.append(f"**Policy digest:** `{policy['digest']}`")

        if self.include_timestamp:
            timestamp = datetime.now(tz=UTC).strftime("%B %d, %Y at %I:%M %p UTC")
            lines.append(f"**Generated:** {timestamp}")
        lines.append("")

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Validator Results | {len(results)} |")
        lines.append(f"| ✅ Passed | {sum(1 for r in results if r.status == 'passed')} |")
        lines.append(
            f"| ❌ Failed | {sum(1 for r in results if r.status == 'failed' and not is_advisory_agent_eval_skip(r))} |"
        )
        lines.append(f"| ⚠️ Incomplete | {sum(1 for r in results if r.is_incomplete)} |")
        if advisory_skip_count:
            lines.append(f"| ⏭️ Advisory skips | {advisory_skip_count} |")

        total_errors = sum(r.summary.errors for r in results)
        total_warnings = sum(r.summary.warnings for r in results)
        critical = sum(r.summary.critical_count for r in results)
        high = sum(r.summary.high_count for r in results)
        medium = sum(r.summary.medium_count for r in results)

        severity_breakdown = []
        if critical > 0:
            severity_breakdown.append(f"{critical} critical")
        if high > 0:
            severity_breakdown.append(f"{high} high")
        if medium > 0:
            severity_breakdown.append(f"{medium} medium")

        issue_str = f"{total_errors + total_warnings}"
        if severity_breakdown:
            issue_str += f" ({', '.join(severity_breakdown)})"
        lines.append(f"| Total Issues | {issue_str} |")
        lines.append("")

        plugin_view = self._tier1_plugin_view(results)
        if plugin_view is not None:
            self._render_plugin_section(plugin_view, lines)

        # Quality Score summary (if any QUALITY results present)
        quality_results = [r for r in results if r.metadata.get("quality_scores")]
        if quality_results:
            lines.append("## Quality Score")
            lines.append("")
            lines.append("| Skill | Score | Grade | Type | Correctness | Discoverability | Reliability | Efficiency |")
            lines.append("|-------|-------|-------|------|-------------|-----------------|-------------|------------|")
            for qr in quality_results:
                qs = qr.metadata["quality_scores"]
                dims = qs.get("dimensions", {})
                lines.append(
                    f"| {_markdown_table_cell(qs.get('skill_name', '—'))} "
                    f"| {qs.get('overall_score', 0):.1f} "
                    f"| {qs.get('grade', '?')} "
                    f"| {qs.get('skill_type', '—')} "
                    f"| {dims.get('correctness', {}).get('score', 0):.1f} "
                    f"| {dims.get('discoverability', {}).get('score', 0):.1f} "
                    f"| {dims.get('reliability', {}).get('score', 0):.1f} "
                    f"| {dims.get('efficiency', {}).get('score', 0):.1f} |"
                )
            lines.append("")

        # Tier 3: Agent Evaluation summary (if present)
        tier3_results = [r for r in results if r.metadata.get("agent_eval")]
        if tier3_results:
            ae = tier3_results[0].metadata["agent_eval"]
            verdict = ae.get("verdict", "unknown").upper()
            composite = ae.get("composite_lift")
            runtime = ae.get("runtime_seconds", 0.0)

            lines.append("## Tier 3: Agent Evaluation")
            lines.append("")
            composite_text = f"{composite:+.2f}" if isinstance(composite, int | float) else "N/A"
            lines.append(f"**Verdict:** {verdict} (composite lift = {composite_text})")
            lines.append(f"**Runtime:** {runtime:.1f}s")
            harbor_viewer = normalize_harbor_viewer_for_display(ae)
            if harbor_viewer.get("job_url"):
                lines.append(f"**Harbor logs:** [Open Harbor logs]({harbor_viewer['job_url']})")
            if harbor_viewer.get("analysis_url"):
                lines.append(f"**Harbor analysis:** [Open Harbor analysis]({harbor_viewer['analysis_url']})")
            lines.append("")

            tier3_plugin = tier3_plugin_view(ae)
            if tier3_plugin is not None:
                self._render_tier3_plugin(tier3_plugin, lines)

            evaluators = ae.get("evaluators", {})
            if evaluators:
                lines.append("### Evaluator Scores")
                lines.append("")
                lines.append("| Evaluator | With Skill | Baseline | Lift |")
                lines.append("|-----------|-----------|----------|------|")
                for name, scores in evaluators.items():
                    scores = scores if isinstance(scores, dict) else {}
                    label = _markdown_table_cell(str(name).replace("_", " ").title())
                    ws = format_score(scores.get("with_skill"), ".2f")
                    bl = format_score(scores.get("baseline"), ".2f")
                    lift = format_score(scores.get("lift"), "+.2f")
                    lines.append(f"| {label} | {ws} | {bl} | {lift} |")
                lines.append("")

            insights = ae.get("insights", {})
            if any(v.get("score") is not None for v in insights.values()):
                lines.append("### LLM-as-Judge Insights")
                lines.append("")
                lines.append("| Dimension | Score | Explanation |")
                lines.append("|-----------|-------|-------------|")
                for dim, info in insights.items():
                    score = info.get("score")
                    if score is None:
                        continue
                    score_str = str(score).upper() if isinstance(score, str) else f"{score:.2f}"
                    explanation = _markdown_table_cell(str(info.get("explanation") or "")[:60])
                    label = _markdown_table_cell(str(dim).title())
                    lines.append(f"| {label} | {_markdown_table_cell(score_str)} | {explanation} |")
                lines.append("")

            suggestions_v2 = ae.get("suggestions_v2") or []
            if suggestions_v2:
                lines.append("### Evidence-Backed Suggestions")
                lines.append("")
                for idx, suggestion in enumerate(suggestions_v2, start=1):
                    recommendation = str(suggestion.get("recommendation") or "").strip()
                    if not recommendation:
                        continue
                    metric = _markdown_table_cell(suggestion.get("metric", "unknown"))
                    lines.append(f"{idx}. **{metric}**: {_markdown_table_cell(recommendation)}")
                    harbor_evidence = suggestion.get("harbor_evidence") or suggestion.get("evidence")
                    if isinstance(harbor_evidence, dict):
                        url = safe_url(harbor_evidence.get("url"))
                        if url:
                            label = _markdown_table_cell(harbor_evidence_link_text(harbor_evidence))
                            lines.append(f"   - Evidence: [{label}]({url})")
                    for ref in (suggestion.get("evidence_refs") or [])[:3]:
                        kind = _markdown_table_cell(ref.get("kind", "evidence"))
                        pointer = _markdown_table_cell(evidence_ref_identity(ref))
                        excerpt = _markdown_table_cell(str(ref.get("excerpt") or ref.get("label") or "")[:120])
                        lines.append(f"   - Evidence: <code>{kind}</code> <code>{pointer}</code> {excerpt}")
                lines.append("")
            elif ae.get("recommendations"):
                lines.append("### Recommendations")
                lines.append("")
                for idx, recommendation in enumerate(ae.get("recommendations") or [], start=1):
                    if not isinstance(recommendation, dict):
                        continue
                    message = str(recommendation.get("message") or recommendation.get("title") or "").strip()
                    if not message:
                        continue
                    lines.append(f"{idx}. {_markdown_table_cell(message)}")
                    evidence = recommendation.get("evidence")
                    if isinstance(evidence, dict):
                        url = safe_url(evidence.get("url"))
                        if url:
                            label = _markdown_table_cell(harbor_evidence_link_text(evidence))
                            lines.append(f"   - Evidence: [{label}]({url})")
                lines.append("")

        # Results per validator
        lines.append("## Results")
        lines.append("")

        for result in results:
            self._render_result(result, lines)

        # Footer
        lines.append("---")
        lines.append("*Generated by SkillEvaluator*")

        return _markdown_output("\n".join(lines))

    @staticmethod
    def _render_plugin_section(view: dict, lines: list[str]) -> None:
        """Render the Tier 1 plugin block: manifest, dependencies, components, MCP, context, risk, similarity."""
        _plugin_overview(view, lines)
        _dependency_resolution(view["dependencies"], lines)
        _component_inventory(view["inventory"], lines)
        _mcp_pinning(view["mcp"], lines)
        _context_cost(view["context_cost"], lines)
        if view.get("static_risk"):
            MarkdownReporter._render_plugin_static_risk(view["static_risk"], lines)
        _catalog_similarity(view["catalog_skill_similarity"], lines)
        _catalog_similarity(view["inter_plugin_similarity"], lines)

    @staticmethod
    def _render_plugin_static_risk(risk: dict, lines: list[str]) -> None:
        """Render subagent/command privileges, hook risk, the CVE audit, validator parity, and endpoint checks."""
        _privileges(risk.get("privileges"), lines)
        _hook_risk(risk.get("hooks"), lines)
        _cve_audit(risk.get("cve"), lines)
        _validator_parity(risk.get("parity"), lines)
        _endpoint_checks(risk.get("endpoints"), lines)

    @staticmethod
    def _render_manifest_declarations(declarations: dict | None, lines: list[str]) -> None:
        """Render every manifest in the plugin root when there is more than one."""
        _manifest_declarations(declarations, lines)

    @staticmethod
    def _render_tier3_plugin(view: dict, lines: list[str]) -> None:
        """Render what the Tier 3 plugin run did and did not demonstrate."""
        if view["partial"]:
            lines.append(
                f"> ⚠️ **INCOMPLETE: {_markdown_table_cell(view['incomplete_reason'])}.** "
                "This is a partial result, not a pass."
            )
            lines.append("")
        _component_coverage(view["coverage"], lines)
        _not_evaluated(view["excluded"], lines)
        _plugin_loading(view.get("plugin_load"), lines)
        _integration(view["integration"], view["lift_modes"], lines)
        _lift_intervals(view["statistics"], lines)
        _mcp_calls(view.get("signals"), lines)
        MarkdownReporter._render_tier3_runtime_evidence(view, lines)

    @staticmethod
    def _render_tier3_runtime_evidence(view: dict, lines: list[str]) -> None:
        """Render the canary, hook census, and MCP proof blocks of a plugin run."""
        _canary(view.get("canary"), lines)
        _hook_census(view.get("hook_census"), lines)
        _mcp_proof(view.get("mcp_proof"), lines)

    def _render_result(self, result: ValidationResult, lines: list[str]) -> None:
        """Render a single validation result."""
        qs = result.metadata.get("quality_scores")

        advisory_skip = is_advisory_agent_eval_skip(result)
        if result.is_incomplete:
            status_emoji = "⚠️ INCOMPLETE"
            lines.append(f"### {status_emoji} {result.validator_name}")
        elif advisory_skip:
            lines.append(f"### ⏭️ SKIPPED {result.validator_name}")
        elif qs and qs.get("grade"):
            grade = qs["grade"]
            status_emoji = "✅" if result.passed else "❌"
            lines.append(f"### {status_emoji} {grade} {result.validator_name}")
        else:
            status_emoji = "✅" if result.passed else "❌"
            lines.append(f"### {status_emoji} {result.validator_name}")

        if result.validator_description:
            lines.append(f"*{result.validator_description}*")
        lines.append("")

        # Quality dimension breakdown
        if qs and qs.get("dimensions"):
            score = qs.get("overall_score", 0)
            grade = qs.get("grade", "?")
            stype = qs.get("skill_type", "unknown")
            lines.append(f"**Overall: {score:.1f}/100 (Grade: {grade})** | Skill Type: {stype}")
            lines.append("")
            lines.append("| Dimension | Score | Weight |")
            lines.append("|-----------|-------|--------|")
            for dname, ddata in qs["dimensions"].items():
                lines.append(f"| {dname.title()} | {ddata.get('score', 0):.1f} | {ddata.get('weight', 0) * 100:.0f}% |")
            lines.append("")

        if result.is_incomplete:
            self._render_incomplete(result, lines)
        elif advisory_skip:
            payload = result.metadata.get("agent_eval", {}) if result.metadata else {}
            provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
            message = provenance.get("message") if isinstance(provenance, dict) else None
            lines.append(f"- {_markdown_table_cell(message) if message else 'Live evaluation did not run.'}")
        elif result.passed:
            self._render_success(result, lines)
            if result.findings:
                lines.append("")
                lines.append(f"**Non-blocking findings: {len(result.findings)}**")
                lines.append("")
                self._render_findings(result.findings, lines)
        else:
            self._render_failure(result, lines)

        lines.append("")

    def _render_success(self, result: ValidationResult, lines: list[str]) -> None:
        """Render success details."""
        if result.success_details:
            for detail in result.success_details:
                check = _markdown_table_cell(detail.check_name)
                lines.append(f"- [OK] **{check}**: {_markdown_table_cell(detail.message)}")
        elif result.messages:
            for msg in result.messages:
                lines.append(f"- {_markdown_table_cell(msg)}")
        else:
            lines.append("- All checks passed")

    def _render_failure(self, result: ValidationResult, lines: list[str]) -> None:
        """Render failure details with findings table and expandable details."""
        # Summary counts
        s = result.summary
        lines.append(f"**{s.errors} errors, {s.warnings} warnings**")
        lines.append("")

        errors = additional_errors(result)
        if errors:
            label = "Execution errors" if result.is_incomplete else "Errors"
            lines.extend([f"**{label}:**", ""])
            for error in errors[: self.max_findings_shown]:
                lines.append(f"- ❌ {_markdown_table_cell(error)}")
            remaining = len(errors) - self.max_findings_shown
            if remaining > 0:
                lines.append(f"- *... and {remaining} more errors*")
            lines.append("")

        if result.findings:
            self._render_findings(result.findings, lines)

    def _render_findings(self, findings: list[Finding], lines: list[str]) -> None:
        """Render a shared findings table for blocking and non-blocking results."""
        lines.append("| Severity | Issue | Location |")
        lines.append("|----------|-------|----------|")

        shown_findings = findings[: self.max_findings_shown]
        for finding in shown_findings:
            emoji = finding.severity.emoji
            severity_upper = finding.severity.value.upper()
            message = _markdown_table_cell(finding.message)
            location = _markdown_table_cell(finding.location)
            lines.append(f"| {emoji} {severity_upper} | {message} | <code>{location}</code> |")

        remaining = len(findings) - len(shown_findings)
        if remaining > 0:
            lines.append(f"| ... | *{remaining} more issues* | |")

        lines.append("")
        if self.include_details:
            lines.append("<details>")
            lines.append("<summary>View Details</summary>")
            lines.append("")

            for index, finding in enumerate(shown_findings, 1):
                self._render_finding_detail(index, finding, lines)

            if remaining > 0:
                lines.append(f"*... and {remaining} more issues*")
                lines.append("")

            lines.append("</details>")
            lines.append("")

    def _render_incomplete(self, result: ValidationResult, lines: list[str]) -> None:
        """Render missing scanner evidence without presenting a failure as a pass."""
        lines.append(f"**Incomplete scanners:** {_markdown_table_cell(', '.join(result.incomplete_scans))}")
        lines.append("")
        self._render_failure(result, lines)
        if not result.findings and not result.errors:
            for warning in result.warnings[: self.max_findings_shown]:
                lines.append(f"- ⚠️ {_markdown_table_cell(warning)}")

    def _render_finding_detail(self, index: int, finding: Finding, lines: list[str]) -> None:
        """Render detailed information for a single finding."""
        # Untrusted values go in <code> elements, not backtick code spans: a code
        # span would show the cell escapes literally.
        cell = _markdown_table_cell
        lines.append(f"**{index}. {cell(finding.message)}**")
        lines.append(f"- File: <code>{cell(finding.location)}</code>")
        lines.append(f"- Check: <code>{cell(finding.check_name)}</code>")
        related_paths = _related_paths(finding)
        if related_paths:
            lines.append(f"- Related paths: {' <-> '.join(f'<code>{cell(path)}</code>' for path in related_paths)}")

        if finding.line_content:
            content = finding.line_content.strip()
            if len(content) > 60:
                content = content[:57] + "..."
            lines.append(f"- Content: <code>{cell(content)}</code>")

        if finding.suggestion:
            lines.append(f"- Fix: {cell(finding.suggestion)}")

        lines.append("")

    def get_file_extension(self) -> str:
        return ".md"
