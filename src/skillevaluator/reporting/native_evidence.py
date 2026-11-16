# SPDX-License-Identifier: Apache-2.0
"""Bounded passive presentation of optional retained native Tier 3 fields.

This compatibility adapter consumes sanitized canonical report data. It never
loads sidecars, resolves endpoints, executes comparisons, changes scores or
certifies native policy claims. Unknown fields (including runtime hook/canary
censuses without a public producer) are deliberately not traversed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice

from skillevaluator.reporting.harbor_viewer import safe_url
from skillevaluator.reporting.plugin_static import _clean_text, _count, _number

_MAX_ROWS = 64
_MAX_TOTAL_ROWS = 256
_MAX_AGENTS = 32
_MAX_NOTES = 8
_MAX_TEXT_BYTES = 48 * 1024
_MAX_AGENT_LABEL_BYTES = 128
# Two attributed sections per agent must remain identifiable after other native
# evidence consumes its allowance. Reserve within the existing aggregate bound.
_AGENT_IDENTITY_BYTES = _MAX_AGENTS * 2 * _MAX_AGENT_LABEL_BYTES
_ACTIVE_DELIVERY = frozenset({"failed", "error", "partial", "incomplete", "indeterminate", "degraded", "upload_failed"})
_CONNECTED = frozenset({"reachable-in-agent", "reachable-host"})


@dataclass(frozen=True)
class NativeCell:
    text: str
    url: str | None = None


@dataclass(frozen=True)
class NativeSection:
    key: str
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[NativeCell, ...], ...]
    notes: tuple[str, ...]
    omitted_rows: int = 0
    omitted_notes: int = 0
    text_shortened: bool = False
    omitted_groups: int = 0


@dataclass(frozen=True)
class NativeNotice:
    title: str
    message: str
    target: str


@dataclass(frozen=True)
class NativeEvidence:
    warnings: tuple[NativeNotice, ...] = ()
    agents: tuple[NativeSection, ...] = ()
    trials: tuple[NativeSection, ...] = ()
    provenance: tuple[NativeSection, ...] = ()
    omitted_agents: int = 0


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _present(value: object) -> bool:
    return (isinstance(value, str) and bool(value.strip())) or type(value) is bool or _number(value) is not None


def _agent_label(name: object, index: int) -> tuple[str, bool]:
    prefix = f"Agent {index + 1} · "
    text, shortened = _clean_text(name) if isinstance(name, str) else ("unavailable", False)
    encoded = text.encode("utf-8")
    remaining = _MAX_AGENT_LABEL_BYTES - len(prefix.encode("utf-8"))
    if len(encoded) > remaining:
        text = encoded[: remaining - 3].decode("utf-8", "ignore") + "…"
        shortened = True
    return prefix + text, shortened


@dataclass
class _Budget:
    rows: int = _MAX_TOTAL_ROWS
    text: int = _MAX_TEXT_BYTES - _AGENT_IDENTITY_BYTES


@dataclass
class _Section:
    key: str
    title: str
    columns: tuple[str, ...]
    budget: _Budget
    rows: list[tuple[NativeCell, ...]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    omitted_rows: int = 0
    omitted_notes: int = 0
    shortened: bool = False
    omitted_groups: int = 0

    def text(self, value: object) -> str:
        if isinstance(value, str):
            text, cut = _clean_text(value)
            self.shortened |= cut
        elif type(value) is bool:
            text = "yes" if value else "no"
        elif _number(value) is not None:
            text = str(value)
        else:
            return "—"
        encoded = text.encode("utf-8")
        if len(encoded) > self.budget.text:
            text = encoded[: self.budget.text].decode("utf-8", "ignore")
            self.shortened = True
        self.budget.text -= len(text.encode("utf-8"))
        return text

    def row(self, *values: object, archive: bool = False) -> None:
        if len(self.rows) >= _MAX_ROWS or self.budget.rows <= 0 or self.budget.text <= 0:
            self.omitted_rows += 1
            self.shortened |= self.budget.text <= 0
            return
        self.budget.rows -= 1
        cells = []
        for index, value in enumerate(values):
            text = self.text(value)
            # Link only a whole, unmodified, explicitly supplied safe URL. Paths
            # remain literal references; no artifact or private URL is inferred.
            url = safe_url(value) if archive and index == len(values) - 1 else None
            cells.append(NativeCell(text, url if url == text else None))
        self.rows.append(tuple(cells))

    def note(self, value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        if len(self.notes) >= _MAX_NOTES:
            self.omitted_notes += 1
            return
        text = self.text(value)
        if text:
            self.notes.append(text)
        else:
            self.omitted_notes += 1

    def values(self, label: str, value: object) -> None:
        if _present(value):
            self.row(label, value)

    def listing(self, label: str, raw: object) -> None:
        if not isinstance(raw, (list, tuple)):
            return
        # Inspect at most the rows that can be shown. Count the rest without
        # traversing their content, including malformed oversized collections.
        limit = min(_MAX_ROWS - len(self.rows), self.budget.rows)
        for value in islice(raw, limit):
            if _present(value):
                self.row(label, value)
        self.omitted_rows += max(0, len(raw) - limit)

    def finish(self) -> NativeSection | None:
        if not (self.rows or self.notes or self.omitted_rows or self.omitted_notes or self.omitted_groups):
            return None
        return NativeSection(
            self.key,
            self.title,
            self.columns,
            tuple(self.rows),
            tuple(self.notes),
            self.omitted_rows,
            self.omitted_notes,
            self.shortened,
            self.omitted_groups,
        )


def _mcp_rows(section: _Section, raw: object, *, nested: bool) -> None:
    if not isinstance(raw, (list, tuple)):
        return
    limit = min(_MAX_ROWS - len(section.rows), section.budget.rows)
    for item in islice(raw, limit):
        record = _mapping(item)
        if not record:
            continue
        usage = _mapping(record.get("tool_usage")) if nested else record
        calls, failed = _count(usage.get("calls")), _count(usage.get("failed"))
        status = record.get("status")
        interpretation = None
        if failed is not None and failed > 0:
            interpretation = "Recorded tool calls failed; successful MCP-backed behavior is not established (advisory)."
        elif calls == 0 and isinstance(status, str) and status in _CONNECTED:
            interpretation = (
                "Recorded connected, no recorded calls; MCP-backed behavior was not demonstrated (advisory)."
            )
        tools = usage.get("tools")
        if isinstance(tools, (list, tuple)) and len(tools) > 16:
            section.note(f"{len(tools) - 16} tool entries omitted for {section.text(record.get('name'))}.")
        values = (record.get("name"), status, calls, failed, _joined(tools), interpretation)
        if nested:
            values += (
                _count(record.get("tool_count")),
                record.get("url") or _mapping(record.get("endpoint")).get("url"),
                record.get("reason"),
                record.get("remediation"),
            )
        section.row(*values)
    section.omitted_rows += max(0, len(raw) - limit)


def _joined(raw: object) -> str | None:
    """Bound a list before joining; the section's text cleaner bounds the cell."""
    if not isinstance(raw, (list, tuple)):
        return None
    texts = [value for value in islice(raw, 16) if isinstance(value, str) and len(value) <= 16384]
    if not texts:
        return None
    return ", ".join(texts) + (f" (+{len(raw) - 16} entries omitted)" if len(raw) > 16 else "")


def _plugin(payload: dict, budget: _Budget, warnings: list[NativeNotice]) -> list[NativeSection]:
    record = _mapping(payload.get("plugin_provenance"))
    section = _Section(
        "plugin-completeness", "Recorded plugin dependency completeness", ("Recorded field", "Value"), budget
    )
    if type(record.get("partial")) is bool:
        section.values("Partial coverage", record["partial"])
    section.values("Sidecar error", record.get("sidecar_error"))
    for key, label in (
        ("evaluated_member_skills", "Resolved skill"),
        ("staged_rules", "Resolved rule"),
        ("unresolved_skill_refs", "Unresolved skill"),
        ("unresolved_rule_refs", "Unresolved rule"),
        ("runnable_mcp_servers", "Runnable MCP server"),
        ("provider_only_mcp_servers", "Provider-only MCP server"),
        ("mcp_unresolved_servers", "Unresolved MCP server"),
    ):
        section.listing(label, record.get(key))
    if record.get("partial") is True or (
        isinstance(record.get("sidecar_error"), str) and record["sidecar_error"].strip()
    ):
        warnings.append(
            NativeNotice(
                "Plugin dependency coverage",
                "Recorded native evidence reports partial or unreadable dependency coverage.",
                section.key,
            )
        )
    result = [finished] if (finished := section.finish()) else []
    mcp = _Section(
        "plugin-mcp",
        "Recorded native MCP evidence",
        (
            "Server",
            "Status",
            "Calls",
            "Failed",
            "Tools",
            "Interpretation",
            "Available tools",
            "Recorded endpoint",
            "Reason",
            "Remediation",
        ),
        budget,
    )
    _mcp_rows(mcp, record.get("mcp_resolutions"), nested=True)
    if finished := mcp.finish():
        result.append(finished)
    configs = _Section(
        "plugin-mcp-configs", "Recorded native MCP configuration exclusions", ("Path", "Entries"), budget
    )
    untrusted = record.get("mcp_untrusted_configs")
    if isinstance(untrusted, (list, tuple)):
        for item in islice(untrusted, _MAX_ROWS):
            config = _mapping(item)
            if config:
                configs.row(config.get("path"), _count(config.get("entries")))
        configs.omitted_rows += max(0, len(untrusted) - _MAX_ROWS)
    if finished := configs.finish():
        result.append(finished)
    return result


def _composition(payload: dict, budget: _Budget) -> list[NativeSection]:
    record = _mapping(payload.get("integration"))
    section = _Section("integration", "Plugin composition · recorded ablation", ("Recorded field", "Value"), budget)
    for key, label in (
        ("with_plugin", "Coordinated plugin"),
        ("sum_of_parts", "Sum-of-parts baseline"),
        ("integration_lift", "Integration lift"),
        ("verdict", "Recorded verdict"),
        ("point_verdict", "Recorded point verdict"),
        ("requested_lift_mode", "Requested lift mode"),
        ("effective_lift_mode", "Effective lift mode"),
        ("complete", "Complete arm coverage"),
    ):
        value = record.get(key)
        if key in {"with_plugin", "sum_of_parts", "integration_lift"}:
            value = _number(value)
        elif key == "complete" and type(value) is not bool:
            value = None
        section.values(label, value)
    for key in ("reason", "interpretation"):
        section.note(record.get(key))
    uncertainty = _mapping(record.get("lift_uncertainty"))
    for key, label in (
        ("ci_low", "Interval lower bound"),
        ("ci_high", "Interval upper bound"),
        ("n_cases", "Paired cases"),
        ("precision", "Recorded precision"),
    ):
        value = uncertainty.get(key)
        if key in {"ci_low", "ci_high"}:
            value = _number(value)
        elif key == "n_cases":
            value = _count(value)
        section.values(label, value)
    section.listing("Member component", record.get("components"))
    completeness = _mapping(record.get("completeness"))
    section.listing("Missing from sum-of-parts arm", completeness.get("missing_sum_of_parts"))
    section.listing("Missing from plugin arm", completeness.get("missing_with_plugin"))
    signals = _mapping(record.get("component_signals"))
    section.values("Recorded multi-component rate", _number(signals.get("multi_component_rate")))
    unused = _mapping(signals.get("unused_declared"))
    for kind in islice(unused, 16):
        section.listing(f"Declared but unused · {section.text(kind)}", unused[kind])
    section.omitted_groups += max(0, len(unused) - 16)
    unavailable = signals.get("runtime_unavailable")
    if isinstance(unavailable, (list, tuple)):
        section.listing(
            "Unavailable at runtime",
            [_mapping(item).get("name") or _mapping(item).get("type") for item in islice(unavailable, _MAX_ROWS)],
        )
        section.omitted_rows += max(0, len(unavailable) - _MAX_ROWS)
    section.note(signals.get("convention_note"))
    mcp = _mapping(record.get("mcp"))
    for key, label in (
        ("declared", "MCP declared"),
        ("connected_in_agent", "MCP connected in agent"),
        ("used", "MCP servers used"),
        ("failed_calls", "MCP failed calls"),
    ):
        if _count(mcp.get(key)) is not None:
            section.values(label, mcp[key])
    if type(mcp.get("exercised")) is bool:
        section.values("Recorded MCP exercised", mcp["exercised"])
    result = []
    if section.rows or section.notes or section.omitted_rows or section.omitted_groups:
        section.note("Recorded native composition evidence is advisory; public scores and gates are unchanged.")
        result.append(section.finish())
    servers = _Section(
        "integration-mcp",
        "Composition · recorded MCP evidence",
        ("Server", "Status", "Calls", "Failed", "Tools", "Interpretation"),
        budget,
    )
    _mcp_rows(servers, mcp.get("servers"), nested=False)
    if finished := servers.finish():
        result.append(finished)
    return result


def _delivery(payload: dict, budget: _Budget, warnings: list[NativeNotice]) -> list[NativeSection]:
    record = _mapping(payload.get("harbor_viewer_delivery"))
    section = _Section(
        "viewer-delivery", "Recorded Harbor delivery", ("Job", "Status", "Error", "Retained archive"), budget
    )
    status = record.get("viewer_status")
    if isinstance(status, str) and status.strip():
        section.note(f"Recorded upload status: {status}")
    active = isinstance(status, str) and status in _ACTIVE_DELIVERY
    uploads = record.get("uploads")
    if isinstance(uploads, (list, tuple)):
        for item in islice(uploads, _MAX_ROWS):
            upload = _mapping(item)
            if not upload:
                continue
            upload_status = upload.get("status")
            active |= isinstance(upload_status, str) and upload_status in _ACTIVE_DELIVERY
            section.row(
                upload.get("job_name"),
                upload_status,
                upload.get("error"),
                upload.get("diagnostic_archive"),
                archive=True,
            )
        section.omitted_rows += max(0, len(uploads) - _MAX_ROWS)
    if active:
        warnings.append(
            NativeNotice(
                "Harbor delivery incomplete",
                "A recorded upload failed or remains incomplete. Evaluation evidence is separate from delivery status.",
                section.key,
            )
        )
    return [finished] if (finished := section.finish()) else []


def _agent(
    name: object, raw: object, index: int, budget: _Budget, warnings: list[NativeNotice]
) -> tuple[list[NativeSection], list[NativeSection]]:
    record = _mapping(raw)
    agent_label, label_shortened = _agent_label(name, index)
    basis = _Section(
        f"refusal-basis-{index}",
        f"Recorded refusal comparison basis · {agent_label}",
        ("Recorded field", "Value"),
        budget,
    )
    basis.shortened = label_shortened
    refusal = _mapping(record.get("safety_refusals"))
    lift = _mapping(_mapping(record.get("pass_at_k")).get("lift"))
    attempts, mixed = _count(refusal.get("attempts")), _count(refusal.get("model_mixed_attempts"))
    for label, value in (("Provider refusal attempts", attempts), ("Mixed-model attempts", mixed)):
        if value is not None:
            basis.values(label, value)
    for key, label in (("categories", "Provider categories"), ("cases", "Recorded refusal case")):
        basis.listing(label, refusal.get(key))
    basis.listing("Recorded excluded comparison case", lift.get("safety_refusal_cases"))
    if isinstance(lift.get("safety_refusal_cases"), (list, tuple)) and lift["safety_refusal_cases"]:
        basis.values("Recorded comparison rate with skill", _number(lift.get("with_skill")))
        basis.values("Recorded comparison rate without skill", _number(lift.get("without_skill")))
    agents = []
    if basis.rows or basis.omitted_rows:
        basis.note(
            "Recorded native refusal and comparison basis; public pass rates, exclusions and denominators are unchanged."
        )
        agents.append(basis.finish())
    cases = refusal.get("cases")
    active = (
        (attempts is not None and attempts > 0)
        or (mixed is not None and mixed > 0)
        or (
            isinstance(cases, (list, tuple))
            and any(isinstance(case, str) and case.strip() for case in islice(cases, _MAX_ROWS))
        )
    )
    # One concise aggregate warning avoids one repeated Overview tile per agent.
    if active and not any(notice.title == "Provider safety notice" for notice in warnings):
        warnings.append(
            NativeNotice(
                "Provider safety notice",
                "Recorded provider refusals or mixed-model attempts are present; inspect the recorded comparison basis.",
                basis.key,
            )
        )
    recovery = _Section(f"recovery-notes-{index}", f"Recorded recovery notes · {agent_label}", (), budget)
    recovery.shortened = label_shortened
    notes = record.get("recovery_notes")
    if isinstance(notes, (list, tuple)):
        for note in islice(notes, _MAX_NOTES):
            recovery.note(note)
        recovery.omitted_notes += max(0, len(notes) - _MAX_NOTES)
    diagnostics = []
    if recovery.notes or recovery.omitted_notes:
        diagnostics.append(recovery.finish())
    return agents, diagnostics


def build_native_evidence(payload: object = None) -> NativeEvidence:
    """Return independent typed display sections from known native fields only.

    Missing, null and malformed values are omitted rather than interpreted as
    zero, successful, complete or inconclusive. The supplied canonical payload
    is never mutated or copied into this view, and may contain unknown cycles.
    """
    record, budget = _mapping(payload), _Budget()
    warnings: list[NativeNotice] = []
    agents = _plugin(record, budget, warnings)
    trials = _composition(record, budget)
    provenance = _delivery(record, budget, warnings)
    raw_agents = _mapping(record.get("agents"))
    for index, (name, agent) in enumerate(islice(raw_agents.items(), _MAX_AGENTS)):
        agent_sections, diagnostic_sections = _agent(name, agent, index, budget, warnings)
        agents.extend(agent_sections)
        provenance.extend(diagnostic_sections)
    return NativeEvidence(
        tuple(warnings), tuple(agents), tuple(trials), tuple(provenance), max(0, len(raw_agents) - _MAX_AGENTS)
    )
