# SPDX-License-Identifier: Apache-2.0
"""Private bounded projection of passive plugin evidence; no discovery or execution."""

from __future__ import annotations

import json
import math
import posixpath
import unicodedata
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from skillevaluator.reporting.run_status import Status, result_status
from skillevaluator.utils.redaction import redact_sensitive_text
from skillevaluator.utils.rich_markup import strip_terminal_controls

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult

MAX_OUTPUT_BYTES = 256 * 1024
MAX_RESULTS = 256
MAX_INPUT_ROWS = 8192
MAX_LIST_ROWS = 4096
MAX_SECTION_ROWS = 64
MAX_TOTAL_ROWS = 256
MAX_TEXT_BYTES = 48 * 1024
MAX_FIELD_INPUT = 16384
MAX_CELL_CHARS = 300


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _number(value: object) -> int | float | None:
    if type(value) not in (int, float):
        return None
    if isinstance(value, int) and value.bit_length() > 53:
        return None
    return value if math.isfinite(value) else None


def _count(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 2**53 - 1 else None


def _path(value: object) -> str:
    if not isinstance(value, str) or len(value) > 4096:
        return ""
    return PurePosixPath(value.replace("\\", "/")).as_posix()


def _absolute(path: str) -> bool:
    return PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute()


def _clean_text(value: str) -> tuple[str, bool]:
    if len(value) > MAX_FIELD_INPUT:
        return "[text exceeds inspection bound]", True
    value = redact_sensitive_text(strip_terminal_controls(value))
    value = "".join(
        "\ufffd"
        if unicodedata.category(char) == "Cs"
        else f"\\u{ord(char):04x}"
        if unicodedata.category(char) == "Cf"
        else char
        for char in value
    )
    value = " ".join(value.split())
    truncated = len(value) > MAX_CELL_CHARS
    return (value[: MAX_CELL_CHARS - 1] + "…" if truncated else value), truncated


@dataclass(frozen=True)
class StaticPluginReport:
    """Immutable serialized evidence and a bounded lexical component index."""

    _serialized: str
    _root: str
    _components: tuple[tuple[str, str, str], ...]

    def payload(self) -> dict[str, Any]:
        return json.loads(self._serialized)

    def component_for(self, file_path: object, metadata: object = None) -> dict[str, str] | None:
        raw = _path(file_path)
        label = None
        if raw.startswith("[") and "] " in raw:
            label, raw = raw[1:].split("] ", 1)
        if ".." in PurePosixPath(raw).parts:
            return None
        if self._root and (raw == self._root or raw.startswith(self._root.rstrip("/") + "/")):
            raw = raw[len(self._root) :].lstrip("/")
        elif _absolute(raw):
            return None
        tags = _mapping(metadata)
        tagged = _mapping(tags.get("plugin_component"))
        if tagged:
            kind, name = tagged.get("type"), tagged.get("name")
            if kind == "hook" and isinstance(name, str) and name.startswith("monitor:"):
                kind, name = "monitor", name.removeprefix("monitor:")
            candidates = [row for row in self._components if row[:2] == (kind, name)]
        elif isinstance(tags.get("mcp_server"), str):
            candidates = [row for row in self._components if row[:2] == ("mcp", tags["mcp_server"])]
            same_file = [row for row in candidates if row[2] == raw]
            candidates = same_file or candidates
        elif isinstance(tags.get("plugin_component_ref"), str):
            ref = _path(tags["plugin_component_ref"])
            for prefix in ("${CLAUDE_PLUGIN_ROOT}", "${CURSOR_PLUGIN_ROOT}", "${PLUGIN_ROOT}"):
                if ref.startswith(prefix + "/"):
                    ref = ref[len(prefix) + 1 :]
                    break
            candidates = (
                []
                if _absolute(ref) or ".." in PurePosixPath(ref).parts
                else [row for row in self._components if row[1] == ref or row[2] == ref]
            )
        else:
            if label:
                skills = [
                    row
                    for row in self._components
                    if row[0] == "skill" and (row[1] == label or row[2] == label or row[2].endswith("/" + label))
                ]
                if len(skills) != 1:
                    return None
                if not (raw == skills[0][2] or raw.startswith(skills[0][2] + "/")):
                    raw = posixpath.join(skills[0][2], raw)
            candidates = [
                row
                for row in self._components
                if row[0] != "mcp" and row[2] and (raw == row[2] or raw.startswith(row[2].rstrip("/") + "/"))
            ]
            if candidates:
                longest = max(len(row[2]) for row in candidates)
                candidates = [row for row in candidates if len(row[2]) == longest]
        if len(candidates) != 1:
            return None
        kind, name, path = candidates[0]
        return {"type": _clean_text(kind)[0], "name": _clean_text(name)[0], "path": _clean_text(path)[0]}


class _Projection:
    def __init__(self) -> None:
        self.input_remaining = MAX_INPUT_ROWS
        self.text_remaining = MAX_TEXT_BYTES
        self.rows_remaining = MAX_TOTAL_ROWS
        self.truncated = False
        self.sections: list[dict[str, Any]] = []

    def records(self, value: object) -> list[dict]:
        if not isinstance(value, (list, tuple)):
            return []
        length = min(len(value), self.input_remaining, MAX_LIST_ROWS)
        self.input_remaining -= length
        self.truncated |= length < len(value)
        return [row for row in islice(value, length) if isinstance(row, dict)]

    def text(self, value: object, *, charge: bool = True) -> str | int | float | None:
        if value is None:
            return None
        if type(value) in (int, float):
            return _number(value)
        if isinstance(value, bool):
            return "yes" if value else "no"
        if not isinstance(value, str):
            return "unavailable"
        value, truncated = _clean_text(value)
        self.truncated |= truncated
        encoded = value.encode("utf-8")
        if charge:
            if len(encoded) > self.text_remaining:
                value = encoded[: self.text_remaining].decode("utf-8", "ignore")
                self.truncated = True
            self.text_remaining -= len(value.encode("utf-8"))
        return value

    def label(self, value: object) -> str:
        normalized = self.text(value, charge=False)
        return "unknown" if normalized is None else str(normalized)

    def names(self, value: object, limit: int = 12) -> str:
        if isinstance(value, str):
            return self.label(value)
        if not isinstance(value, (list, tuple)):
            return "unknown"
        chosen = [self.label(item) for item in islice(value, limit)]
        if len(value) > limit:
            chosen.append(f"(+{len(value) - limit} more)")
            self.truncated = True
        return ", ".join(chosen) or "none"

    def section(
        self,
        key: str,
        title: str,
        columns: tuple[str, ...],
        rows: list[list],
        *,
        total: int | None = None,
        notes: list | None = None,
    ) -> None:
        # Defer shared budgets until every section is known so risk evidence
        # can take priority without changing the report's display order.
        self.sections.append(
            {
                "key": key,
                "title": title,
                "columns": list(columns),
                "rows": rows[:MAX_SECTION_ROWS],
                "total": len(rows) if total is None else total,
                "notes": (notes or [])[:8],
            }
        )

    def _bounded_cells(self, values: list) -> list | None:
        cells = [self.text(value, charge=False) for value in values]
        size = sum(len(cell.encode("utf-8")) for cell in cells if isinstance(cell, str))
        if size > self.text_remaining:
            self.truncated = True
            return None
        self.text_remaining -= size
        return cells

    def finish(self) -> list[dict[str, Any]]:
        risk_keys = {"hooks", "privileges", "audits", "endpoints"}
        for section in sorted(self.sections, key=lambda item: item["key"] not in risk_keys):
            rows = []
            for row in section["rows"][: self.rows_remaining]:
                cells = self._bounded_cells(row)
                if cells is None:
                    break
                rows.append(cells)
            self.rows_remaining -= len(rows)
            section["rows"] = rows
            section["omitted"] = max(0, section["total"] - len(rows))
            self.truncated |= bool(section["omitted"])
            section["notes"] = [
                cells[0] for note in section["notes"] if (cells := self._bounded_cells([note])) is not None
            ]
        return self.sections

    def url(self, value: object) -> str:
        if not isinstance(value, str) or len(value) > MAX_FIELD_INPUT:
            return "unavailable"
        try:
            parsed = urlsplit(value)
            host = parsed.hostname
            port = parsed.port
            if parsed.scheme not in {"http", "https"} or not host or any(char.isspace() for char in parsed.netloc):
                return "unavailable"
            authority = f"[{host}]" if ":" in host else host
            if port is not None:
                authority += f":{port}"
            return urlunsplit((parsed.scheme, authority, parsed.path, "", ""))
        except ValueError:
            return "unavailable"


def _count_rows(value: object) -> int | None:
    return len(value) if isinstance(value, (list, tuple)) else None


def project_plugin_static(results: list[ValidationResult]) -> StaticPluginReport | None:
    """Reduce private static results once, keeping unknown and incomplete facts explicit."""
    projection = _Projection()
    merged: dict[str, Any] = {}
    sources = []
    endpoints = []
    conflicts = []
    blocked = False
    incomplete = len(results) > MAX_RESULTS
    known_keys = (
        "root",
        "name",
        "manifest_sha256",
        "manifest_declarations",
        "component_inventory",
        "dependency_resolution",
        "dependency_status_counts",
        "dependency_manifests",
        "mcp",
        "context_cost",
        "hook_risk",
        "privileges",
        "cve_summary",
        "validator_parity",
        "catalog_skill_similarity",
        "inter_plugin_similarity",
    )
    for result in islice(results, MAX_RESULTS):
        metadata = _mapping(result.metadata)
        block = _mapping(metadata.get("plugin"))
        severities: dict[str, Counter] = {}
        incomplete |= result.is_incomplete or len(result.findings) > MAX_INPUT_ROWS
        blocked |= result_status(result) is Status.FAILED and (
            bool(result.gate_failures) or metadata.get("optional_skip") is not True
        )
        for finding in islice(result.findings, MAX_INPUT_ROWS):
            severity = getattr(finding.severity, "value", finding.severity)
            blocked |= severity in {"critical", "high"}
            ecosystem = _mapping(finding.metadata).get("ecosystem")
            if isinstance(ecosystem, str) and finding.check_name != "dependency-not-audited":
                severities.setdefault(ecosystem, Counter())[severity] += 1
        sources.append((result, block, severities))
        for key in known_keys:
            if key not in block:
                continue
            value = block[key]
            if (
                key in {"root", "manifest_sha256", "name"}
                and isinstance(value, str)
                and key in merged
                and value != merged[key]
            ):
                conflicts.append(f"Conflicting source identity: {key}")
            if key not in merged or merged[key] is None:
                merged[key] = value
        endpoint = _mapping(metadata.get("endpoint_resolution"))
        if endpoint:
            endpoints.append(endpoint)
    projection.truncated |= len(results) > MAX_RESULTS
    if not merged:
        return None
    declarations = merged.get("manifest_declarations")
    if isinstance(declarations, dict):
        declarations = declarations.get("manifests")
    manifests = projection.records(declarations)
    selected = next((row for row in manifests if row.get("selected") is True), {})
    name = projection.text(merged.get("name") or selected.get("name") or "plugin")
    root = _path(merged.get("root"))
    # Gates and typed findings remain authoritative; metadata never declares a pass.
    status = "failed" if blocked else "incomplete" if incomplete or conflicts else "passed"
    if conflicts:
        projection.section(
            "identity_conflicts", "Source identity conflicts", ("Issue",), [[message] for message in conflicts]
        )
    if manifests:
        projection.section(
            "manifests",
            "Plugin manifests",
            ("Selected", "Manifest", "Type", "Name", "Version"),
            [
                [
                    row.get("selected"),
                    row.get("manifest_filename"),
                    row.get("manifest_type"),
                    row.get("name"),
                    row.get("version"),
                ]
                for row in manifests
            ],
            total=_count_rows(declarations),
        )
    hook_block = _mapping(merged.get("hook_risk"))
    privilege_block = _mapping(merged.get("privileges"))
    hook_records = projection.records(hook_block.get("hooks"))
    privilege_records = projection.records(privilege_block.get("components"))
    checked_types = {"monitor" if row.get("event") == "Monitor" else "hook" for row in hook_records}
    checked_types.update(row.get("type") for row in privilege_records if isinstance(row.get("type"), str))
    checked_types.update({"lsp", "monitor", "settings", "output_style"})

    def support(row: dict) -> object:
        if row.get("type") in checked_types:
            return "statically checked"
        return "inventoried only" if row.get("support") == "unsupported" else row.get("support")

    inventory = _mapping(merged.get("component_inventory"))
    components = projection.records(inventory.get("components"))
    index = tuple(
        (
            str(projection.text(row.get("type"), charge=False)),
            str(projection.text(row.get("name"), charge=False)),
            _path(row.get("path")),
        )
        for row in components
    )
    if inventory:
        complete = inventory.get("complete")
        complete_label = "yes" if complete is True else "no" if complete is False else "unknown"
        ordered = sorted(components, key=lambda row: (not bool(row.get("problem")), not bool(row.get("findings"))))
        projection.section(
            "inventory",
            "Component inventory and support",
            ("Type", "Name", "Origin", "Path", "Static support", "State", "Findings", "Declared by", "Client view"),
            [
                [
                    row.get("type"),
                    row.get("name"),
                    row.get("origin"),
                    row.get("path"),
                    support(row),
                    row.get("problem")
                    or row.get("dependency")
                    or ("overridden" if row.get("effective") is False else "declared"),
                    _count(row.get("findings")),
                    row.get("declared_by"),
                    row.get("loaded_by"),
                ]
                for row in ordered
            ],
            total=_count_rows(inventory.get("components")),
            notes=[
                "Static evidence does not demonstrate runtime loading or behavior.",
                f"Discovery complete: {complete_label}",
            ],
        )
    counts = _mapping(inventory.get("counts"))
    if counts:
        projection.section(
            "inventory_counts",
            "Component counts",
            ("Type", "Count"),
            [[kind, _count(amount)] for kind, amount in islice(counts.items(), 32)],
        )
    counts = _mapping(merged.get("dependency_status_counts"))
    if counts:
        projection.section(
            "dependency_counts",
            "Dependency state counts",
            ("State", "Count"),
            [[state, _count(amount)] for state, amount in islice(counts.items(), 16)],
        )
    resolution = _mapping(merged.get("dependency_resolution"))
    dependencies = []
    dependency_total = 0
    for kind, raw in islice(resolution.items(), 16):
        dependency_total += _count_rows(raw) or 0
        dependencies.extend((kind, row) for row in projection.records(raw))
    if resolution:
        dependencies.sort(
            key=lambda item: {"missing": 0, "unresolved": 1, "external": 2}.get(str(item[1].get("state")), 3)
        )
        projection.section(
            "dependencies",
            "Dependency resolution",
            ("Kind", "Reference", "State", "Path", "Declared by", "Reason"),
            [
                [kind, row.get("ref"), row.get("state"), row.get("path"), row.get("declared_by"), row.get("reason")]
                for kind, row in dependencies
            ],
            total=dependency_total,
        )
    mcp = _mapping(merged.get("mcp"))
    if mcp:
        projection.section(
            "mcp",
            "MCP declarations and pinning",
            ("Server", "Source", "Kind", "Transport", "Pinned", "Detail"),
            [
                [
                    row.get("name"),
                    row.get("source"),
                    row.get("kind"),
                    row.get("transport"),
                    row.get("pinned"),
                    row.get("pin_detail"),
                ]
                for row in projection.records(mcp.get("servers"))
            ],
            total=_count_rows(mcp.get("servers")),
        )
    pinning = _mapping(mcp.get("pinning"))
    if pinning:
        projection.section(
            "mcp_pinning",
            "MCP pinning totals",
            ("Total", "Pinned", "Unpinned", "Not applicable", "Ratio"),
            [
                [_count(pinning.get(key)) for key in ("total", "pinned", "unpinned", "not_applicable")]
                + [_number(pinning.get("ratio"))]
            ],
        )
    cost = _mapping(merged.get("context_cost"))
    if cost:
        summary = f"Static estimate {projection.label(cost.get('estimator'))}; {projection.label(cost.get('harness'))}/{projection.label(cost.get('load_mode'))}; always-on {projection.label(_number(cost.get('always_on_tokens')))}; on-demand {projection.label(_number(cost.get('on_demand_tokens')))}; lower bound {cost.get('lower_bound') is True}"
        projection.section(
            "context",
            "Estimated context cost",
            ("Type", "Name", "Always-on tokens", "On-demand tokens", "Basis", "Not counted"),
            [
                [
                    row.get("type"),
                    row.get("name"),
                    _number(row.get("always_on_tokens")),
                    _number(row.get("on_demand_tokens")),
                    row.get("basis"),
                    row.get("not_counted"),
                ]
                for row in projection.records(cost.get("by_component"))
            ],
            total=_count_rows(cost.get("by_component")),
            notes=[summary, projection.names(cost.get("not_counted")), projection.names(cost.get("notes"))],
        )
        projection.section(
            "context_harnesses",
            "Context estimates by harness",
            ("Harness", "Load mode", "Always-on tokens", "On-demand tokens", "Lower bound"),
            [
                [
                    row.get("harness"),
                    row.get("load_mode"),
                    _number(row.get("always_on_tokens")),
                    _number(row.get("on_demand_tokens")),
                    row.get("lower_bound"),
                ]
                for row in projection.records(cost.get("by_harness"))
            ],
            total=_count_rows(cost.get("by_harness")),
        )
    hooks = hook_block
    if hooks:
        rows = hook_records
        rows.sort(key=lambda row: not bool(row.get("risk_flags")))
        projection.section(
            "hooks",
            "Hook static risk",
            ("Event", "Hook", "Matcher", "Handler", "Target", "Flags"),
            [
                [
                    row.get("event"),
                    row.get("id"),
                    row.get("matcher") or "all",
                    row.get("handler_type"),
                    row.get("target"),
                    projection.names(row.get("risk_flags")),
                ]
                for row in rows
            ],
            total=_count_rows(hooks.get("hooks")),
        )
    privileges = privilege_block
    if privileges:
        rows = privilege_records
        rows.sort(key=lambda row: not bool(row.get("flags")))
        projection.section(
            "privileges",
            "Static component privileges",
            ("Type", "Name", "Path", "Grants", "Model", "Permission mode", "Flags"),
            [
                [
                    row.get("type"),
                    row.get("name"),
                    row.get("path"),
                    "inherits all tools"
                    if row.get("inherits_all_tools") is True
                    else projection.names(row.get("allowed_tools") or row.get("tools")),
                    row.get("model"),
                    row.get("permission_mode"),
                    projection.names(row.get("flags")),
                ]
                for row in rows
            ],
            total=_count_rows(privileges.get("components")),
        )
    audit_rows = []
    parity_rows = []
    for _result, block, severities in sources:
        for ecosystem, raw in islice(_mapping(_mapping(block.get("cve_summary")).get("ecosystems")).items(), 16):
            row = _mapping(raw)
            observed = severities.get(ecosystem, Counter())
            declared = row.get("vulnerabilities")
            if isinstance(declared, dict):
                observed = {key: _count(value) for key, value in islice(declared.items(), 5)}
                total = sum(value for value in observed.values() if value is not None)
            else:
                total = _count(declared)
            partial = row.get("findings_truncated") is True or row.get("status") in {"incomplete", "unverified"}
            if row.get("status") in {"disabled", "not_found", "unverified"} or (partial and total == 0):
                total = None
            severity = (
                ", ".join(f"{key}: {value}" for key, value in sorted(observed.items()) if value is not None)
                or "unknown"
            )
            audit_rows.append(
                [
                    ecosystem,
                    row.get("status", "unknown"),
                    _count(row.get("sources", row.get("declarations"))),
                    _count(row.get("audited")),
                    total,
                    ("partial; " if partial else "") + severity,
                    projection.names(row.get("scanners")),
                    projection.names(row.get("errors")),
                ]
            )
        parity = _mapping(block.get("validator_parity"))
        if parity:
            parity_rows.append(
                [
                    parity.get("status"),
                    parity.get("claude_verdict"),
                    parity.get("strict_verdict"),
                    parity.get("skillevaluator_verdict"),
                    parity.get("agree"),
                    projection.names(parity.get("errors")),
                    projection.names(parity.get("warnings")),
                    parity.get("reason"),
                ]
            )
    if audit_rows:
        projection.section(
            "audits",
            "Dependency vulnerability audits",
            (
                "Ecosystem",
                "Status",
                "Sources",
                "Audited",
                "Vulnerabilities",
                "Known severity counts",
                "Scanners",
                "Notes",
            ),
            audit_rows,
        )
    if parity_rows:
        projection.section(
            "parity",
            "Official validator parity",
            (
                "Status",
                "CLI verdict",
                "Strict verdict",
                "Evaluator verdict",
                "Agreement",
                "Errors",
                "Warnings",
                "Reason",
            ),
            parity_rows,
        )
    if endpoints:
        rows = [row for block in endpoints for row in projection.records(block.get("endpoints"))]
        total = sum(_count_rows(block.get("endpoints")) or 0 for block in endpoints)
        projection.section(
            "endpoints",
            "Endpoint checks",
            ("Kind", "Name", "URL", "Status", "HEAD", "Redirect", "Redirect classification", "Reason"),
            [
                [
                    row.get("kind"),
                    row.get("name"),
                    projection.url(row.get("url")),
                    row.get("status"),
                    _mapping(row.get("head")).get("status") or _mapping(row.get("head")).get("skipped") or "unknown",
                    projection.url(_mapping(row.get("redirect")).get("url")),
                    _mapping(row.get("redirect")).get("classification"),
                    row.get("reason"),
                ]
                for row in rows
            ],
            total=total,
        )
    for source_key, key, title in (
        ("catalog_skill_similarity", "catalog_skills", "Skills versus local catalog"),
        ("inter_plugin_similarity", "catalog_plugins", "Plugin versus local catalog"),
    ):
        comparison = _mapping(merged.get(source_key))
        if comparison:
            projection.section(
                key,
                title,
                ("Subject", "Match", "Similarity", "Member overlap", "Verdict"),
                [
                    [
                        row.get("skill") or row.get("name"),
                        row.get("match"),
                        row.get("similarity"),
                        row.get("member_overlap"),
                        row.get("verdict"),
                    ]
                    for row in projection.records(comparison.get("matches"))
                ],
                total=_count_rows(comparison.get("matches")),
                notes=[
                    f"Advisory; status {projection.text(comparison.get('status'), charge=False)}; catalog entries {projection.label(_count(comparison.get('catalog_entries')))}",
                    comparison.get("reason"),
                ],
            )
    sections = projection.finish()
    payload = {
        "schema_version": 1,
        "name": name,
        "status": status,
        "truncated": projection.truncated,
        "sections": sections,
    }
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # A final byte ceiling also covers structural overhead, not only text cells.
    while len(serialized.encode("utf-8")) > MAX_OUTPUT_BYTES:
        section = next((item for item in reversed(projection.sections) if item["rows"]), None)
        if section is None:
            raise ValueError("Static report structure exceeds output allowance")
        section["rows"].pop()
        section["omitted"] += 1
        payload["truncated"] = True
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return StaticPluginReport(serialized, root, index)
