# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin report sections shared by every reporter.

Plugin evaluation adds optional blocks to the Tier 1 plugin metadata and to
the Tier 3 payload: Tier 1 component and dependency metadata, Tier 3 component
coverage and provenance, per-arm plugin signals, lift statistics, and the
Integration block. Older runs predate these blocks, and any producer may be
absent, so every key here is optional.

The builders turn whatever is present into small, bounded, display-ready view
models. Every value a reporter prints is either a pre-formatted string or a
small count, and each reporter applies its own escaping (HTML autoescape,
Markdown cell escaping, Rich markup escaping, publication sanitizing). A
builder returns ``None`` when there is no plugin data, so skill reports render
exactly as before.

The coverage view keeps three claims apart: a component's files were staged,
the agent loaded or called it, and its behavior was verified. Staging proves
only the first, so reports built from these views count the components that
were not staged and, when trials recorded activation, the staged components no
trial exercised, instead of implying that staged means tested.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

MAX_TABLE_ROWS = 200
MAX_LIST_ITEMS = 64
MAX_TEXT_CHARS = 300
MAX_AGENTS = 16
MAX_TOP_FAILURES = 5
MAX_SERVERS = 32

DEPENDENCY_STATES = ("provided", "referenced", "missing", "external", "unresolved")
COVERAGE_STATES = ("staged", "not_staged", "unsupported", "unavailable", "invalid")
SUPPORT_LABELS = {
    "evaluated": "Evaluated",
    "static_only": "Static only",
    "unsupported": "Unsupported",
}
COVERAGE_LABELS = {
    "staged": "Staged",
    "not_staged": "Not staged",
    "unsupported": "Unsupported",
    "unavailable": "Unavailable",
    "invalid": "Invalid",
}
ARM_LABELS = {
    "with_skill": "Plugin",
    "with_plugin": "Plugin",
    "without_skill": "Baseline (no plugin)",
    "baseline": "Baseline (no plugin)",
    "sum_of_parts": "Sum of parts",
}
# In a legacy 2-arm ``--lift-mode integration`` run the only baseline arm stages
# the plugin's member components individually, so that arm and its interval
# describe the sum of parts, not a run without the plugin.
SUM_OF_PARTS_BASELINE_LABEL = "Sum-of-parts baseline"
_BASELINE_ARMS = frozenset({"without_skill", "baseline"})
_ARM_ORDER = ("with_skill", "with_plugin", "without_skill", "baseline", "sum_of_parts")
_LIFT_KINDS = (("effectiveness", "Effectiveness lift"), ("integration", "Integration lift"))
_SUM_OF_PARTS_LIFT_LABEL = "Integration lift (sum-of-parts baseline)"
_PRECISION_CLASSES = {"adequate": "ok", "low": "warn", "insufficient": "fail"}
_INTEGRATION_LIFT_MODES = frozenset({"integration", "both"})
_INTEGRATION_VERDICTS = {
    "real_integration": ("Real integration", "ok"),
    "cosmetic_bundling": ("Cosmetic bundling", "warn"),
    "negative_integration": ("Negative integration", "fail"),
    "inconclusive": ("Inconclusive", "warn"),
}
_SIGNAL_SECTIONS = (
    "tool_selection",
    "arguments",
    "mcp_calls",
    "order",
    "handoff",
    "conflict",
    "activation_coverage",
)
_STATISTIC_KEYS = (
    "lift_uncertainty",
    "reliability",
    "cost",
    "token_efficiency",
    "context_cost_measured",
    "integration_completeness",
)
_COMPLETENESS_ISSUE_KEYS = ("missing_cases", "failed_arms", "attempt_shortfall")

STAGED_IS_NOT_VERIFIED = (
    "Staged means the component's files were placed in the evaluation workspace. "
    "It does not show that the agent loaded the component, and it does not verify the component's behavior."
)
NOT_CONFIGURED = "not configured for this dataset"
SIGNALS_ADVISORY_NOTE = (
    "Advisory, report-only signals computed from agent trajectories. They never change a score or verdict."
)
STATIC_CONTEXT_COST_NOTE = (
    "Static estimate from component file sizes, not a measured token count. "
    "Always-on content loads with every session; on-demand content loads only when used."
)


# ---------------------------------------------------------------------------
# Value coercion and formatting
# ---------------------------------------------------------------------------


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else []


def text(value: object, *, limit: int = MAX_TEXT_CHARS) -> str:
    """Return one bounded, whitespace-collapsed display string."""
    if value is None or isinstance(value, Mapping | list | tuple | set):
        return ""
    collapsed = " ".join(str(value).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def number(value: object) -> float | None:
    """Return a finite number, rejecting booleans and non-numeric values."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    return numeric if math.isfinite(numeric) else None


def count(value: object) -> int | None:
    """Return a non-negative integer count, or ``None`` when absent or invalid."""
    numeric = number(value)
    if numeric is None or numeric < 0:
        return None
    return round(numeric)


def _names(value: object, *, limit: int = MAX_LIST_ITEMS) -> tuple[list[str], int]:
    """Return bounded, de-duplicated display names and how many were omitted."""
    names: list[str] = []
    for item in _sequence(value):
        if isinstance(item, Mapping):
            item = item.get("name") or item.get("ref") or item.get("id")
        name = text(item)
        if name and name not in names:
            names.append(name)
    return names[:limit], max(0, len(names) - limit)


def fmt_rate(value: object) -> str:
    numeric = number(value)
    return "n/a" if numeric is None else f"{numeric * 100:.0f}%"


def fmt_score(value: object) -> str:
    numeric = number(value)
    return "n/a" if numeric is None else f"{numeric:.2f}"


def fmt_signed(value: object) -> str:
    numeric = number(value)
    return "n/a" if numeric is None else f"{numeric:+.2f}"


def fmt_count(value: object) -> str:
    numeric = number(value)
    if numeric is None:
        return "n/a"
    if numeric.is_integer():
        return f"{int(numeric):,}"
    return f"{numeric:,.1f}"


def fmt_usd(value: object) -> str:
    numeric = number(value)
    return "n/a" if numeric is None else f"${numeric:,.4f}"


def _ratio(numerator: object, denominator: object) -> float | None:
    top = number(numerator)
    bottom = number(denominator)
    if top is None or bottom is None or bottom <= 0:
        return None
    return top / bottom


def _first(sources: Iterable[tuple[Mapping[str, Any], str]]) -> str:
    for source, key in sources:
        value = text(source.get(key))
        if value:
            return value
    return ""


def json_safe(value: Any, *, _depth: int = 0) -> Any:
    """Copy JSON-like data, replacing non-finite floats and unknown types."""
    if _depth > 32:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item, _depth=_depth + 1) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item, _depth=_depth + 1) for item in value]
    return str(value)


def _plural(value: int, noun: str) -> str:
    return f"{value} {noun}{'' if value == 1 else 's'}"


# ---------------------------------------------------------------------------
# Tier 1 (validators/plugin_schema.py metadata)
# ---------------------------------------------------------------------------


def plugin_block(metadata: object) -> dict[str, Any] | None:
    """Return the raw Tier 1 plugin block carried by one result's metadata."""
    source = _mapping(metadata)
    manifest_type = source.get("manifest_type")
    if not manifest_type:
        return None
    block: dict[str, Any] = dict(_mapping(source.get("plugin")))
    block["manifest_type"] = manifest_type
    block["plugin_mode"] = source.get("plugin_mode")
    return block


def tier1_plugin_view(
    block: object,
    *,
    status: str | None = None,
    bundled_skills: list[str] | None = None,
) -> dict[str, Any] | None:
    """Return the display model for the Tier 1 plugin section."""
    source = _mapping(block)
    if not source:
        return None
    bundled, bundled_omitted = _names(source.get("bundled_skills") if bundled_skills is None else bundled_skills)
    return {
        "name": text(source.get("name")),
        "manifest_type": text(source.get("manifest_type")),
        "plugin_mode": text(source.get("plugin_mode")),
        "manifest_filename": text(source.get("manifest_filename")),
        "status": status or "",
        "status_label": {"passed": "PASSED", "failed": "FAILED", "incomplete": "INCOMPLETE"}.get(status or "", ""),
        "declared_dependencies": _declared_dependencies(source.get("declared_dependencies")),
        "dependencies": dependency_view(source),
        "bundled_skills": bundled,
        "bundled_skills_omitted": bundled_omitted,
        "in_plugin_skills": count(source.get("in_plugin_skills")),
        "inventory": inventory_view(source.get("component_inventory")),
        "mcp": mcp_view(source.get("mcp")),
        "context_cost": context_cost_view(source.get("context_cost")),
        "catalog_skill_similarity": similarity_view(source.get("catalog_skill_similarity"), kind="skills"),
        "inter_plugin_similarity": similarity_view(source.get("inter_plugin_similarity"), kind="plugins"),
    }


def _fmt_overlap(value: object) -> str:
    numeric = number(value)
    if numeric is None:
        return text(value, limit=64) or "n/a"
    if isinstance(value, float) and 0.0 <= numeric <= 1.0:
        return fmt_rate(numeric)
    return fmt_count(numeric)


def similarity_view(value: object, *, kind: str) -> dict[str, Any] | None:
    """Return an advisory Tier 2 local-catalog similarity result (skills or plugins)."""
    similarity = _mapping(value)
    if not similarity:
        return None
    status = text(similarity.get("status"), limit=32).lower() or "unknown"
    matches: list[dict[str, str]] = []
    total = 0
    for match in _sequence(similarity.get("matches")):
        if not isinstance(match, Mapping):
            continue
        total += 1
        if len(matches) >= MAX_TABLE_ROWS:
            continue
        if kind == "skills":
            matches.append(
                {
                    "subject": text(match.get("skill")),
                    "match": text(match.get("match")),
                    "similarity": fmt_score(match.get("similarity")),
                }
            )
        else:
            matches.append(
                {
                    "subject": text(match.get("name")),
                    "similarity": fmt_score(match.get("similarity")),
                    "member_overlap": _fmt_overlap(match.get("member_overlap")),
                    "verdict": text(match.get("verdict"), limit=64),
                }
            )
    return {
        "status": status,
        "status_label": {"compared": "Compared", "skipped": "Skipped"}.get(status, status.replace("_", " ").title()),
        "catalog_entries": count(similarity.get("catalog_entries")),
        "matches": matches,
        "omitted": max(0, total - len(matches)),
        "reason": text(similarity.get("reason")),
    }


def _declared_dependencies(value: object) -> list[dict[str, Any]]:
    rows = []
    for key, raw in sorted(_mapping(value).items(), key=lambda item: str(item[0])):
        amount = count(raw)
        if amount is not None and text(key):
            rows.append({"kind": text(key), "count": amount})
    return rows[:MAX_LIST_ITEMS]


def _state_counts(value: object, ordered_states: tuple[str, ...]) -> list[dict[str, Any]]:
    counts = _mapping(value)
    rows = [{"state": state, "count": count(counts.get(state)) or 0} for state in ordered_states]
    for key, raw in sorted(counts.items(), key=lambda item: str(item[0])):
        state = text(key, limit=64)
        amount = count(raw)
        if state and state not in ordered_states and amount is not None:
            rows.append({"state": state, "count": amount})
    return rows[:MAX_LIST_ITEMS]


def dependency_view(block: object) -> dict[str, Any] | None:
    """Return declared-dependency resolution counts and per-ref rows."""
    source = _mapping(block)
    counts = _mapping(source.get("dependency_status_counts"))
    resolution = _mapping(source.get("dependency_resolution"))
    if not counts and not resolution:
        return None
    rows: list[dict[str, str]] = []
    kinds = ["skills", "rules", *sorted(str(key) for key in resolution if key not in {"skills", "rules"})]
    total_rows = 0
    for kind in kinds:
        for entry in _sequence(resolution.get(kind)):
            if not isinstance(entry, Mapping):
                continue
            total_rows += 1
            if len(rows) >= MAX_TABLE_ROWS:
                continue
            rows.append(
                {
                    "kind": text(kind, limit=32).rstrip("s") or "ref",
                    "ref": text(entry.get("ref")),
                    "state": text(entry.get("state"), limit=32) or "unknown",
                    "path": text(entry.get("path")),
                    "reason": text(entry.get("reason")),
                }
            )
    count_rows = _state_counts(counts, DEPENDENCY_STATES) if counts else []
    missing = next((row["count"] for row in count_rows if row["state"] == "missing"), 0)
    if not counts:
        missing = sum(1 for row in rows if row["state"] == "missing")
    return {
        "counts": count_rows,
        "total": sum(row["count"] for row in count_rows) if count_rows else total_rows,
        "rows": rows,
        "omitted": max(0, total_rows - len(rows)),
        "missing": missing,
    }


def inventory_view(value: object) -> dict[str, Any] | None:
    """Return the component inventory table and unsupported-type callout."""
    inventory = _mapping(value)
    if not inventory:
        return None
    rows: list[dict[str, Any]] = []
    total = 0
    computed_counts: dict[str, int] = {}
    unsupported_seen: list[str] = []
    for component in _sequence(inventory.get("components")):
        if not isinstance(component, Mapping):
            continue
        total += 1
        component_type = text(component.get("type"), limit=32) or "unknown"
        support = text(component.get("support"), limit=32)
        computed_counts[component_type] = computed_counts.get(component_type, 0) + 1
        if support == "unsupported" and component_type not in unsupported_seen:
            unsupported_seen.append(component_type)
        if len(rows) >= MAX_TABLE_ROWS:
            continue
        rows.append(
            {
                "type": component_type,
                "name": text(component.get("name")),
                "origin": text(component.get("origin"), limit=64),
                "path": text(component.get("path")),
                "support": support,
                "support_label": SUPPORT_LABELS.get(support, support or "unknown"),
                "findings": count(component.get("findings")) or 0,
            }
        )
    declared_counts = _mapping(inventory.get("counts"))
    count_source = declared_counts or computed_counts
    counts = [
        {"type": text(key, limit=32), "count": amount}
        for key, raw in sorted(count_source.items(), key=lambda item: str(item[0]))
        if (amount := count(raw)) is not None and text(key, limit=32)
    ][:MAX_LIST_ITEMS]
    if "unsupported_types_present" in inventory:
        unsupported, _omitted = _names(inventory.get("unsupported_types_present"))
    else:
        unsupported = unsupported_seen
    return {
        "rows": rows,
        "omitted": max(0, total - len(rows)),
        "total": total or sum(item["count"] for item in counts),
        "counts": counts,
        "unsupported_types": unsupported,
    }


def pinning_view(value: object, servers: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Return MCP pinning totals, deriving them from server rows when absent."""
    pinning = _mapping(value)
    servers = servers or []
    if not pinning and not servers:
        return None
    total = count(pinning.get("total"))
    pinned = count(pinning.get("pinned"))
    unpinned = count(pinning.get("unpinned"))
    not_applicable = count(pinning.get("not_applicable"))
    if total is None and servers:
        total = len(servers)
    if pinned is None and servers:
        pinned = sum(1 for server in servers if server.get("pinned") is True)
    if unpinned is None and servers:
        unpinned = sum(1 for server in servers if server.get("pinned") is False)
    if not_applicable is None and servers:
        not_applicable = sum(1 for server in servers if server.get("pinned") is None)
    ratio = number(pinning.get("ratio"))
    applicable = None if total is None else max(0, total - (not_applicable or 0))
    if ratio is None:
        ratio = _ratio(pinned, applicable)
    return {
        "total": total,
        "pinned": pinned,
        "unpinned": unpinned,
        "not_applicable": not_applicable,
        "ratio": ratio,
        "ratio_label": fmt_rate(ratio),
        "summary": (
            f"{pinned}/{applicable} pinned" if pinned is not None and applicable is not None else "pinning not recorded"
        ),
    }


def mcp_view(value: object) -> dict[str, Any] | None:
    """Return MCP servers, their pinning ratio, and the unpinned servers."""
    mcp = _mapping(value)
    if not mcp:
        return None
    servers: list[dict[str, Any]] = []
    total = 0
    for server in _sequence(mcp.get("servers")):
        if not isinstance(server, Mapping):
            continue
        total += 1
        if len(servers) >= MAX_TABLE_ROWS:
            continue
        pinned = server.get("pinned")
        servers.append(
            {
                "name": text(server.get("name")) or "unnamed",
                "source": text(server.get("source"), limit=64),
                "kind": text(server.get("kind"), limit=32),
                "transport": text(server.get("transport"), limit=64),
                "pinned": pinned if isinstance(pinned, bool) else None,
                "pin_detail": text(server.get("pin_detail")),
            }
        )
    unpinned = [server for server in servers if server["pinned"] is False]
    return {
        "servers": servers,
        "omitted": max(0, total - len(servers)),
        "pinning": pinning_view(mcp.get("pinning"), servers),
        "unpinned": unpinned,
    }


def context_cost_view(value: object) -> dict[str, Any] | None:
    """Return the static always-on versus on-demand context estimate."""
    cost = _mapping(value)
    if not cost:
        return None
    always_on = number(cost.get("always_on_tokens"))
    on_demand = number(cost.get("on_demand_tokens"))
    components: list[dict[str, Any]] = []
    for entry in _sequence(cost.get("by_component")):
        if not isinstance(entry, Mapping):
            continue
        components.append(
            {
                "type": text(entry.get("type"), limit=32),
                "name": text(entry.get("name")),
                "always_on_value": number(entry.get("always_on_tokens")) or 0.0,
                "always_on": fmt_count(entry.get("always_on_tokens")),
                "on_demand": fmt_count(entry.get("on_demand_tokens")),
                "basis": text(entry.get("basis")),
            }
        )
    if always_on is None and on_demand is None and not components:
        return None
    components.sort(key=lambda row: (-row["always_on_value"], row["type"], row["name"]))
    notes, _omitted = _names(cost.get("notes"), limit=8)
    estimator = text(cost.get("estimator"), limit=64)
    method = text(cost.get("method"), limit=64) or "static_estimate"
    label = "Static estimate"
    if estimator == "chars_div_4":
        label = "Static estimate (characters ÷ 4)"
    elif estimator:
        label = f"Static estimate ({estimator})"
    return {
        "method": method,
        "estimator": estimator,
        "label": label,
        "always_on": fmt_count(always_on),
        "on_demand": fmt_count(on_demand),
        "components": components[:MAX_TABLE_ROWS],
        "omitted": max(0, len(components) - MAX_TABLE_ROWS),
        "notes": notes,
        "note": STATIC_CONTEXT_COST_NOTE,
    }


def component_for_path(file_path: object, block: object) -> dict[str, str] | None:
    """Return the inventory component whose root-relative path contains *file_path*.

    Findings carry either root-relative or absolute paths. Absolute paths are
    made root-relative against the plugin root; the longest matching component
    path wins, so a file inside ``skills/foo`` maps to that skill rather than
    to a broader component.
    """
    raw = text(file_path, limit=4096)
    source = _mapping(block)
    if not raw:
        return None
    if raw.startswith("[") and "]" in raw:
        raw = raw[raw.index("]") + 1 :].strip()
    normalized = raw.replace("\\", "/")
    root = text(source.get("root"), limit=4096).replace("\\", "/").rstrip("/")
    if root and (normalized == root or normalized.startswith(root + "/")):
        normalized = normalized[len(root) :].lstrip("/")
    elif PurePosixPath(normalized).is_absolute() or PureWindowsPath(raw).is_absolute():
        return None
    normalized = normalized.removeprefix("./")
    best: dict[str, str] | None = None
    best_length = -1
    inventory = _mapping(source.get("component_inventory"))
    for component in _sequence(inventory.get("components"))[: MAX_TABLE_ROWS * 5]:
        if not isinstance(component, Mapping):
            continue
        component_path = text(component.get("path"), limit=4096).replace("\\", "/").removeprefix("./").rstrip("/")
        if not component_path:
            continue
        contains = normalized == component_path or normalized.startswith(component_path + "/")
        if contains and len(component_path) > best_length:
            best_length = len(component_path)
            best = {
                "type": text(component.get("type"), limit=32),
                "name": text(component.get("name")),
                "path": component_path,
                "support": text(component.get("support"), limit=32),
            }
    return best


# ---------------------------------------------------------------------------
# Tier 3 (canonical agent_eval payload or raw engine result)
# ---------------------------------------------------------------------------


def _plugin_provenance(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("plugin_provenance")) or _mapping(
        _mapping(payload.get("summary")).get("plugin_provenance")
    )


def _agents(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    agents = _mapping(payload.get("agents"))
    return [
        (text(name, limit=64), agent) for name, agent in list(agents.items())[:MAX_AGENTS] if isinstance(agent, Mapping)
    ]


def is_plugin_payload(payload: object) -> bool:
    """Return whether a Tier 3 payload or engine result describes a plugin run."""
    source = _mapping(payload)
    if not source:
        return False
    if _plugin_provenance(source) or _mapping(source.get("integration")):
        return True
    if text(source.get("lift_mode_requested")) or text(source.get("lift_mode_effective")):
        return True
    for target in (source.get("eval_target"), _mapping(source.get("run_config")).get("eval_target")):
        if text(_mapping(target).get("kind")) == "plugin":
            return True
    if _mapping(source.get("plugin_signals_summary")):
        return True
    return any(_arm_signal_summaries(agent) for _name, agent in _agents(source))


def tier3_plugin_view(payload: object) -> dict[str, Any] | None:
    """Return the display model for every Tier 3 plugin block, or ``None``."""
    source = _mapping(payload)
    if not is_plugin_payload(source):
        return None
    provenance = _plugin_provenance(source)
    statistics = statistics_view(source)
    signals = signals_view(source)
    coverage = coverage_view(provenance.get("component_coverage"), signals)
    integration = integration_view(source, provenance, statistics)
    completeness = completeness_view(provenance)
    partial = bool(completeness and completeness["partial"])
    dataset = {
        "cases": count(provenance.get("dataset_case_count")),
        "cross_component_cases": count(provenance.get("cross_component_case_count")),
        "integration_evidence_ready": (
            provenance.get("integration_evidence_ready")
            if isinstance(provenance.get("integration_evidence_ready"), bool)
            else None
        ),
    }
    dependency_counts = _mapping(provenance.get("dependency_status_counts"))
    view = {
        "plugin_name": text(provenance.get("plugin_name")),
        "partial": partial,
        "incomplete_reason": completeness["reason"] if completeness and partial else "",
        "completeness": completeness,
        "dataset": dataset if any(value is not None for value in dataset.values()) else None,
        "coverage": coverage,
        "static_context_cost": context_cost_view(provenance.get("context_cost")),
        "mcp_pinning": pinning_view(provenance.get("mcp_pinning")),
        "dependency_counts": _state_counts(dependency_counts, DEPENDENCY_STATES) if dependency_counts else [],
        "lift_modes": _lift_modes(source, provenance),
        "sum_of_parts_baseline": baseline_is_sum_of_parts(source),
        "integration": integration,
        "statistics": statistics,
        "signals": signals,
    }
    view["excluded"] = excluded_behavior(view, provenance)
    content_keys = (
        "completeness",
        "coverage",
        "static_context_cost",
        "mcp_pinning",
        "lift_modes",
        "integration",
        "statistics",
        "signals",
    )
    if not any(view[key] for key in content_keys):
        return None
    return view


_COMPLETENESS_FIELDS = {
    "skills_resolved": "evaluated_member_skills",
    "rules_resolved": "staged_rules",
    "mcp_runnable": "runnable_mcp_servers",
    "skills_unresolved": "unresolved_skill_refs",
    "rules_unresolved": "unresolved_rule_refs",
    "mcp_provider_only": "provider_only_mcp_servers",
    "mcp_unsupported_config": "mcp_unsupported_config",
}


def sidecar_error_reason(provenance: object) -> str:
    """Return why an unusable provenance sidecar keeps a plugin run INCOMPLETE, or ``""``.

    ``_read_plugin_provenance`` records ``sidecar_error`` when the run's
    ``plugin_provenance.json`` exists but cannot be used. The run is then
    INCOMPLETE because its evaluated components are unknown, not because a
    listed dependency failed to resolve.
    """
    code = text(_mapping(provenance).get("sidecar_error"), limit=64)
    if not code:
        return ""
    return f"plugin provenance sidecar unreadable ({code}), so the components evaluated at Tier 3 are unknown"


def completeness_view(provenance: object) -> dict[str, Any] | None:
    """Return resolved versus deferred declared components for a plugin run."""
    source = _mapping(provenance)
    if not source:
        return None
    sidecar_reason = sidecar_error_reason(source)
    groups = {
        "skills_resolved": _names(source.get("evaluated_member_skills")),
        "rules_resolved": _names(source.get("staged_rules")),
        "mcp_runnable": _names(source.get("runnable_mcp_servers")),
        "skills_unresolved": _names(source.get("unresolved_skill_refs")),
        "rules_unresolved": _names(source.get("unresolved_rule_refs")),
        "mcp_provider_only": _names(source.get("provider_only_mcp_servers")),
        "mcp_unsupported_config": _names(source.get("mcp_unsupported_config")),
    }
    counts = {key: len(_sequence(source.get(field))) for key, field in _COMPLETENESS_FIELDS.items()}
    deferred = [
        (counts["skills_unresolved"], "unresolved skill ref"),
        (counts["rules_unresolved"], "unresolved rule ref"),
        (counts["mcp_provider_only"], "provider-only MCP server"),
        (counts["mcp_unsupported_config"], "MCP server declaring config the runtime cannot apply"),
    ]
    declared_partial = source.get("partial") is True
    computed_partial = any(amount for amount, _label in deferred)
    detail = ", ".join(_plural(amount, label) for amount, label in deferred if amount)
    return {
        "partial": declared_partial or computed_partial or bool(sidecar_reason),
        "counts": counts,
        "names": {key: names for key, (names, _omitted) in groups.items()},
        "sidecar_error": text(source.get("sidecar_error"), limit=64) if sidecar_reason else "",
        "reason": sidecar_reason
        or f"{detail or 'required declared components'} could not be resolved or evaluated at Tier 3",
    }


def coverage_view(value: object, signals: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return per-component coverage with prominent not-staged and not-observed counts.

    The headline counts components that were not staged. Staging is not
    evaluation, so when trials recorded activation the view also counts the
    staged components no plugin trial exercised, and ``all_exercised`` is true
    only when every component was staged and exercised. Without activation
    data nothing is known beyond staging, so ``all_exercised`` stays false.
    """
    coverage = _mapping(value)
    if not coverage:
        return None
    activation = (signals or {}).get("activation")
    rows: list[dict[str, Any]] = []
    total = 0
    not_staged_rows: list[dict[str, Any]] = []
    unobserved_rows: list[dict[str, Any]] = []
    unobserved = 0
    computed_counts: dict[str, int] = {}
    for component in _sequence(coverage.get("components")):
        if not isinstance(component, Mapping):
            continue
        total += 1
        state = text(component.get("state"), limit=32) or "unknown"
        computed_counts[state] = computed_counts.get(state, 0) + 1
        row = {
            "type": text(component.get("type"), limit=32) or "unknown",
            "name": text(component.get("name")),
            "origin": text(component.get("origin"), limit=64),
            "path": text(component.get("path")),
            "state": state,
            "state_label": COVERAGE_LABELS.get(state, state),
            "staged": state == "staged",
            "reason": text(component.get("reason")),
            "observed": "",
        }
        if activation:
            row["observed"] = _observed_activation(row, activation)
        if len(rows) < MAX_TABLE_ROWS:
            rows.append(row)
        if state != "staged":
            if len(not_staged_rows) < MAX_TABLE_ROWS:
                not_staged_rows.append(row)
        elif activation and row["observed"] != "exercised":
            unobserved += 1
            if len(unobserved_rows) < MAX_TABLE_ROWS:
                unobserved_rows.append(row)
    declared_counts = _mapping(coverage.get("counts"))
    counts = _state_counts(declared_counts or computed_counts, COVERAGE_STATES)
    # The producer records the not-staged count as ``not_evaluated``.
    not_staged = count(coverage.get("not_evaluated"))
    if not_staged is None:
        not_staged = sum(row["count"] for row in counts if row["state"] != "staged")
    staged = next((row["count"] for row in counts if row["state"] == "staged"), 0)
    return {
        "rows": rows,
        "omitted": max(0, total - len(rows)),
        "total": total or sum(row["count"] for row in counts),
        "staged": staged,
        "counts": [row for row in counts if row["count"] or row["state"] in COVERAGE_STATES],
        "not_staged": not_staged,
        "not_staged_rows": not_staged_rows,
        "headline": f"{_plural(not_staged, 'component')} not staged",
        "staged_not_observed": unobserved if activation else None,
        "staged_not_observed_rows": unobserved_rows,
        "observed_headline": (
            f"{_plural(unobserved, 'staged component')} not observed in any plugin trial" if unobserved else ""
        ),
        "all_exercised": bool(activation) and staged > 0 and not_staged == 0 and unobserved == 0,
        "note": STAGED_IS_NOT_VERIFIED,
        "activation": activation,
    }


_ACTIVATION_TYPE_ALIASES = {"rule": ("rule", "rule_read"), "agent": ("agent", "subagent")}


def _observed_activation(row: Mapping[str, Any], activation: Mapping[str, Any]) -> str:
    """Return whether trials observed a coverage row's component (advisory)."""
    keys = {f"{kind}:{row['name']}" for kind in _ACTIVATION_TYPE_ALIASES.get(row["type"], (row["type"],))}
    if keys & set(activation.get("exercised") or []):
        return "exercised"
    if keys & set(activation.get("unavailable") or []):
        return "unavailable"
    if keys & set(activation.get("unverified") or []):
        return "unverified"
    return "not observed"


def _lift_modes(payload: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, str] | None:
    integration = _mapping(payload.get("integration"))
    summary = _mapping(payload.get("summary"))
    # The runner's own record, the only one a raw engine result carries.
    recorded = _mapping(_mapping(payload.get("run_config")).get("lift_mode"))
    requested = _first(
        (
            (payload, "lift_mode_requested"),
            (integration, "lift_mode_requested"),
            (integration, "requested_lift_mode"),
            (summary, "lift_mode_requested"),
            (provenance, "lift_mode_requested"),
            (provenance, "requested_lift_mode"),
            (recorded, "requested"),
        )
    )
    effective = _first(
        (
            (payload, "lift_mode_effective"),
            (integration, "lift_mode_effective"),
            (integration, "effective_lift_mode"),
            (summary, "lift_mode_effective"),
            (provenance, "lift_mode_effective"),
            (provenance, "effective_lift_mode"),
            (recorded, "effective"),
        )
    )
    if not requested and not effective:
        return None
    return {
        "requested": requested or "not recorded",
        "effective": effective or "not recorded",
        "fallback": bool(requested and effective and requested != effective),
    }


def integration_view(
    payload: Mapping[str, Any],
    provenance: Mapping[str, Any],
    statistics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the Integration block, including an explicit INCONCLUSIVE state."""
    integration = _mapping(payload.get("integration"))
    modes = _lift_modes(payload, provenance)
    requested = (modes or {}).get("requested", "")
    if not integration and requested not in _INTEGRATION_LIFT_MODES:
        return None
    lift = number(integration.get("integration_lift"))
    measured_flag = integration.get("measured")
    measured = measured_flag if isinstance(measured_flag, bool) else bool(integration) and lift is not None
    verdict = text(integration.get("verdict"), limit=64).lower() or "inconclusive"
    if not measured:
        verdict = "inconclusive"
    verdict_label, verdict_class = _INTEGRATION_VERDICTS.get(verdict, (verdict.replace("_", " ").title(), "warn"))
    reason = text(integration.get("reason")) or text(provenance.get("integration_skip_reason"))
    if not integration:
        reason = reason or "Integration was requested, but this run recorded no sum-of-parts comparison."
    elif not measured and not reason:
        reason = "No complete sum-of-parts comparison was produced for this run."
    components, components_omitted = _names(integration.get("components"))
    ci = None
    if statistics:
        ci = next(
            (row for row in statistics["primary"]["lift_ci"] if row["kind"] == "integration"),
            None,
        )
    uncertainty = _mapping(integration.get("lift_uncertainty"))
    if "estimate" not in uncertainty and "ci_low" not in uncertainty:
        uncertainty = _mapping(uncertainty.get("integration"))
    if uncertainty:
        ci = _ci_row("integration", "Integration lift", uncertainty)
    point_verdict = text(integration.get("point_verdict"), limit=64).lower()
    point_label = (
        _INTEGRATION_VERDICTS.get(point_verdict, (point_verdict.replace("_", " ").title(), "warn"))[0]
        if point_verdict and point_verdict != verdict
        else ""
    )
    sum_of_parts_baseline = bool(modes and modes["effective"] == "integration")
    completeness = completeness_issues_view(
        integration.get("completeness"), sum_of_parts_baseline=sum_of_parts_baseline
    ) or completeness_issues_view(integration, sum_of_parts_baseline=sum_of_parts_baseline)
    return {
        "modes": modes,
        "measured": measured,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "verdict_class": verdict_class,
        "point_verdict_label": point_label,
        "reason": reason,
        "with_plugin": fmt_score(integration.get("with_plugin")),
        "sum_of_parts": fmt_score(integration.get("sum_of_parts")),
        "integration_lift": fmt_signed(lift),
        "lift_value": lift,
        "components": components,
        "components_omitted": components_omitted,
        "interpretation": text(integration.get("interpretation")),
        "ci": ci,
        "completeness": completeness,
    }


def baseline_is_sum_of_parts(payload: object) -> bool:
    """Return whether the run's only baseline arm was the plugin's sum of parts.

    The legacy 2-arm ``--lift-mode integration`` (effective mode
    ``integration``) stages the member components individually in the
    baseline, so its ``without_skill`` arm and ``effectiveness`` interval
    compare the plugin with its parts. Plugin versus no plugin was not run.
    """
    source = _mapping(payload)
    modes = _lift_modes(source, _plugin_provenance(source))
    return bool(modes and modes["effective"] == "integration")


def _statistic_sources(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    provenance = _mapping(payload.get("provenance"))
    return [
        payload,
        _mapping(payload.get("statistics")),
        _mapping(payload.get("summary")),
        _mapping(provenance.get("comparison")),
        _mapping(payload.get("comparison")),
    ]


def _has_completeness_issue_keys(value: object) -> bool:
    return any(key in _mapping(value) for key in _COMPLETENESS_ISSUE_KEYS)


def _statistics_block(source: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        key: value
        for key in _STATISTIC_KEYS
        if (value := _mapping(source.get(key)))
        and (key != "integration_completeness" or _has_completeness_issue_keys(value))
    }


def statistics_view(payload: object) -> dict[str, Any] | None:
    """Return lift intervals, reliability, cost, and completeness per agent.

    The payload repeats the best agent's statistics at the top level; per-agent
    blocks win, the top level only fills gaps for the best agent (or stands in
    when no agent carries its own block), and the best agent's scope is primary.
    """
    source = _mapping(payload)
    best = text(source.get("best_agent") or _mapping(source.get("summary")).get("best_agent"), limit=64)
    sum_of_parts_baseline = baseline_is_sum_of_parts(source)
    run_statistics: dict[str, Mapping[str, Any]] = {}
    for key in _STATISTIC_KEYS:
        for candidate in _statistic_sources(source):
            value = _mapping(candidate.get(key))
            if value and (key != "integration_completeness" or _has_completeness_issue_keys(value)):
                run_statistics[key] = value
                break
    scopes: list[dict[str, Any]] = []
    for name, agent in _agents(source):
        agent_statistics = _statistics_block(agent)
        if name == best and agent_statistics:
            agent_statistics = {**run_statistics, **agent_statistics}
        if agent_statistics:
            scopes.append(_statistics_scope(name, agent_statistics, sum_of_parts_baseline=sum_of_parts_baseline))
    if not scopes and run_statistics:
        scopes.append(
            _statistics_scope(
                f"Best agent ({best})" if best else "All agents",
                run_statistics,
                sum_of_parts_baseline=sum_of_parts_baseline,
            )
        )
    if not scopes:
        return None
    primary = next((scope for scope in scopes if scope["label"] == best), scopes[0])
    return {"scopes": [primary, *(scope for scope in scopes if scope is not primary)], "primary": primary}


def _ci_row(kind: str, label: str, value: Mapping[str, Any]) -> dict[str, Any]:
    estimate = number(value.get("estimate"))
    low = number(value.get("ci_low"))
    high = number(value.get("ci_high"))
    confidence = number(value.get("confidence"))
    includes_zero = value.get("ci_includes_zero")
    if not isinstance(includes_zero, bool):
        includes_zero = low <= 0.0 <= high if low is not None and high is not None else None
    precision = text(value.get("precision"), limit=32).lower()
    interval = f"[{fmt_signed(low)}, {fmt_signed(high)}]" if low is not None and high is not None else "n/a"
    confidence_label = f"{confidence * 100:.0f}% CI" if confidence is not None else "CI"
    return {
        "kind": kind,
        "label": label,
        "estimate_value": estimate,
        "low_value": low,
        "high_value": high,
        "estimate": fmt_signed(estimate),
        "interval": interval,
        "confidence": confidence_label,
        "method": text(value.get("method"), limit=64),
        "resamples": fmt_count(value.get("resamples")),
        "n_cases": fmt_count(value.get("n_cases")),
        "precision": precision or "unknown",
        "precision_class": _PRECISION_CLASSES.get(precision, "neutral"),
        "ci_includes_zero": includes_zero,
        "summary": f"{fmt_signed(estimate)} {interval} ({confidence_label})",
    }


def _ordered_arms(*mappings: Mapping[str, Any]) -> list[str]:
    arms: list[str] = []
    for mapping in mappings:
        for key in mapping:
            arm = text(key, limit=64)
            if arm and arm not in arms:
                arms.append(arm)
    order = {arm: index for index, arm in enumerate(_ARM_ORDER)}
    return sorted(arms, key=lambda arm: (order.get(arm, len(order)), arm))[:MAX_LIST_ITEMS]


def arm_label(arm: str, *, sum_of_parts_baseline: bool = False) -> str:
    if sum_of_parts_baseline and arm in _BASELINE_ARMS:
        return SUM_OF_PARTS_BASELINE_LABEL
    return ARM_LABELS.get(arm, arm.replace("_", " ").title())


def _statistics_scope(
    label: str,
    statistics: Mapping[str, Mapping[str, Any]],
    *,
    sum_of_parts_baseline: bool = False,
) -> dict[str, Any]:
    uncertainty = _mapping(statistics.get("lift_uncertainty"))
    lift_ci = [
        _ci_row(
            kind,
            _SUM_OF_PARTS_LIFT_LABEL if sum_of_parts_baseline and kind == "effectiveness" else kind_label,
            _mapping(uncertainty.get(kind)),
        )
        for kind, kind_label in _LIFT_KINDS
        if _mapping(uncertainty.get(kind))
    ]
    reliability = _mapping(statistics.get("reliability"))
    cost = _mapping(statistics.get("cost"))
    efficiency = _mapping(statistics.get("token_efficiency"))
    arms = []
    for arm in _ordered_arms(reliability, cost, efficiency):
        arm_reliability = _mapping(reliability.get(arm))
        arm_cost = _mapping(cost.get(arm))
        arms.append(
            {
                "arm": arm,
                "label": arm_label(arm, sum_of_parts_baseline=sum_of_parts_baseline),
                "pass_at_k": fmt_rate(arm_reliability.get("pass_at_k")),
                "pass_hat_k": fmt_rate(arm_reliability.get("pass_hat_k")),
                "k": fmt_count(arm_reliability.get("k")),
                "n_cases": fmt_count(arm_reliability.get("n_cases")),
                "tokens_per_success": fmt_count(arm_cost.get("tokens_per_success")),
                "usd_per_success": fmt_usd(arm_cost.get("usd_per_success")),
                "total_tokens": fmt_count(arm_cost.get("total_tokens")),
                "successes": fmt_count(arm_cost.get("successes")),
                "token_efficiency": fmt_score(efficiency.get(arm)),
                "has_reliability": bool(arm_reliability),
                "has_tokens": number(arm_cost.get("tokens_per_success")) is not None,
                "has_usd": number(arm_cost.get("usd_per_success")) is not None,
                "has_efficiency": number(efficiency.get(arm)) is not None,
            }
        )
    return {
        "label": label,
        "lift_ci": lift_ci,
        "arms": arms,
        "has_reliability": any(row["has_reliability"] for row in arms),
        "has_tokens": any(row["has_tokens"] for row in arms),
        "has_usd": any(row["has_usd"] for row in arms),
        "has_efficiency": any(row["has_efficiency"] for row in arms),
        "context_measured": _context_measured_view(statistics.get("context_cost_measured")),
        "completeness": completeness_issues_view(
            statistics.get("integration_completeness"), sum_of_parts_baseline=sum_of_parts_baseline
        ),
    }


def _context_measured_view(value: object) -> dict[str, Any] | None:
    measured = _mapping(value)
    if not measured:
        return None
    status = text(measured.get("status"), limit=32).lower() or "unknown"
    delta = number(measured.get("delta_tokens_mean"))
    return {
        "status": status,
        "measured": status == "measured" and delta is not None,
        "delta": "n/a" if delta is None else f"{delta:+,.0f} tokens",
        "n_pairs": fmt_count(measured.get("n_pairs")),
        "method": text(measured.get("method"), limit=64),
        "reason": text(measured.get("reason")),
    }


def completeness_issues_view(value: object, *, sum_of_parts_baseline: bool = False) -> dict[str, Any] | None:
    """Return integration completeness issues (missing cases, failed arms, shortfalls)."""
    completeness = _mapping(value)
    if not completeness:
        return None
    complete = completeness.get("complete") if isinstance(completeness.get("complete"), bool) else None
    missing_cases, missing_omitted = _names(completeness.get("missing_cases"))
    failed_arms, _failed_omitted = _names(completeness.get("failed_arms"))
    shortfall = []
    for entry in _sequence(completeness.get("attempt_shortfall"))[:MAX_LIST_ITEMS]:
        if isinstance(entry, Mapping):
            arm = text(entry.get("arm"), limit=64)
            shortfall.append(
                {
                    "case": text(entry.get("case")),
                    "arm": arm_label(arm, sum_of_parts_baseline=sum_of_parts_baseline) if arm else "",
                    "expected": fmt_count(entry.get("expected")),
                    "observed": fmt_count(entry.get("observed")),
                }
            )
    if complete is None and not (missing_cases or failed_arms or shortfall):
        return None
    return {
        "complete": complete,
        "missing_cases": missing_cases,
        "missing_omitted": missing_omitted,
        "failed_arms": [arm_label(arm, sum_of_parts_baseline=sum_of_parts_baseline) for arm in failed_arms],
        "attempt_shortfall": shortfall,
        "issues": bool(missing_cases or failed_arms or shortfall or complete is False),
    }


def _arm_signal_summaries(agent: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    summaries = _signal_summaries(agent.get("plugin_signals_summary"))
    if summaries:
        return summaries
    conditions = _mapping(agent.get("conditions"))
    return {
        text(arm, limit=64): summary
        for arm, condition in conditions.items()
        if (summary := _mapping(_mapping(condition).get("plugin_signals_summary"))) and _is_signal_summary(summary)
    }


def _is_signal_summary(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in _SIGNAL_SECTIONS) or any(
        key in value for key in ("n_trials", "n_missing_trajectory", "activations")
    )


def _signal_summaries(value: object) -> dict[str, Mapping[str, Any]]:
    summary = _mapping(value)
    if not summary:
        return {}
    if _is_signal_summary(summary):
        return {"with_skill": summary}
    return {
        text(arm, limit=64): candidate
        for arm, raw in summary.items()
        if (candidate := _mapping(raw)) and _is_signal_summary(candidate)
    }


def signals_view(payload: object) -> dict[str, Any] | None:
    """Return advisory per-arm plugin signals and the union activation coverage.

    Per-agent summaries win; the payload's top-level copy (the best agent's)
    is used only when no agent carries its own, so nothing renders twice.
    """
    source = _mapping(payload)
    sum_of_parts_baseline = baseline_is_sum_of_parts(source)
    sources: list[tuple[str, dict[str, Mapping[str, Any]]]] = []
    for name, agent in _agents(source):
        summaries = _arm_signal_summaries(agent)
        if summaries:
            sources.append((name, summaries))
    if not sources:
        run_summaries = _signal_summaries(source.get("plugin_signals_summary"))
        if run_summaries:
            best = text(source.get("best_agent") or _mapping(source.get("summary")).get("best_agent"), limit=64)
            sources.append((best or "All agents", run_summaries))
    entries: list[dict[str, Any]] = []
    activation: dict[str, list[str]] = {"declared": [], "exercised": [], "unverified": [], "unavailable": []}
    for scope, summaries in sources:
        for arm in _ordered_arms(summaries):
            entry = _signal_entry(scope, arm, summaries[arm], sum_of_parts_baseline=sum_of_parts_baseline)
            entries.append(entry)
            if entry["activation"] and arm in {"with_skill", "with_plugin"}:
                for key, collected in activation.items():
                    for name in entry["activation"][key]:
                        if name not in collected and len(collected) < MAX_LIST_ITEMS:
                            collected.append(name)
    if not entries:
        return None
    has_activation = any(activation.values())
    return {
        "entries": entries,
        "note": SIGNALS_ADVISORY_NOTE,
        "activation": (
            {
                **activation,
                "summary": (
                    f"{len(activation['exercised'])} of {len(activation['declared'])} declared components "
                    "were exercised in at least one plugin trial"
                ),
            }
            if has_activation
            else None
        ),
    }


def _scored(section: Mapping[str, Any]) -> bool:
    return text(section.get("status"), limit=32).lower() != "not_applicable"


def _rate(section: Mapping[str, Any], *keys: str, numerator: str, denominator: str) -> float | None:
    """Return the first recorded rate, else ``numerator / denominator``; ``None`` for 0/0."""
    for key in keys:
        if key in section:
            return number(section.get(key))
    return _ratio(section.get(numerator), section.get(denominator))


def _check_row(label: str, section: Mapping[str, Any], passed_key: str, total_key: str) -> dict[str, Any] | None:
    if not section:
        return None
    applicable = _scored(section)
    passed = section.get(passed_key)
    total = section.get(total_key)
    rate = _rate(section, "pass_rate", "satisfaction_rate", numerator=passed_key, denominator=total_key)
    return {
        "name": label,
        "applicable": applicable,
        "passed": fmt_count(passed),
        "total": fmt_count(total),
        "rate": fmt_rate(rate),
        "n_scored": fmt_count(section.get("n_scored")),
        "label": (
            NOT_CONFIGURED if not applicable else f"{fmt_count(passed)}/{fmt_count(total)} passed ({fmt_rate(rate)})"
        ),
    }


def _signal_failures(arguments: Mapping[str, Any]) -> list[dict[str, str]]:
    failures = []
    for failure in _sequence(arguments.get("failures") or arguments.get("top_failures"))[:MAX_TOP_FAILURES]:
        if isinstance(failure, Mapping):
            failures.append(
                {
                    "tool": text(failure.get("tool")),
                    "arg": text(failure.get("arg")),
                    "rule": text(failure.get("rule"), limit=64),
                    "detail": text(failure.get("detail")),
                    "count": fmt_count(failure.get("count")) if failure.get("count") is not None else "",
                }
            )
    return failures


def _signal_mcp(mcp_calls: Mapping[str, Any]) -> dict[str, Any] | None:
    if not mcp_calls:
        return None
    servers = []
    for server, raw_stats in list(_mapping(mcp_calls.get("by_server")).items())[:MAX_SERVERS]:
        stats = _mapping(raw_stats)
        tools, tools_omitted = _names(stats.get("tools"), limit=8)
        servers.append(
            {
                "server": text(server) or "unnamed",
                "total": fmt_count(stats.get("total")),
                "succeeded": fmt_count(stats.get("succeeded")),
                "failed": fmt_count(stats.get("failed")),
                "unknown": fmt_count(stats.get("unknown")),
                "success_rate": fmt_rate(_rate(stats, "success_rate", numerator="succeeded", denominator="total")),
                "tools": ", ".join(tools) + (f" (+{tools_omitted} more)" if tools_omitted else ""),
            }
        )
    return {
        "total": fmt_count(mcp_calls.get("total")),
        "succeeded": fmt_count(mcp_calls.get("succeeded")),
        "failed": fmt_count(mcp_calls.get("failed")),
        "unknown": fmt_count(mcp_calls.get("unknown")),
        "success_rate": fmt_rate(_rate(mcp_calls, "success_rate", numerator="succeeded", denominator="total")),
        "servers": servers,
    }


def _signal_entry(
    scope: str,
    arm: str,
    summary: Mapping[str, Any],
    *,
    sum_of_parts_baseline: bool = False,
) -> dict[str, Any]:
    tool_selection = _mapping(summary.get("tool_selection"))
    arguments = _mapping(summary.get("arguments"))
    activation = _mapping(summary.get("activation_coverage"))
    activations = _mapping(summary.get("activations"))

    argument_view = None
    if arguments:
        argument_view = {
            "applicable": _scored(arguments),
            "checked": fmt_count(arguments.get("checked")),
            "passed": fmt_count(arguments.get("passed")),
            "pass_rate": fmt_rate(_rate(arguments, "pass_rate", numerator="passed", denominator="checked")),
            "failures": _signal_failures(arguments),
        }

    activation_view = None
    if activation:
        activation_view = {
            key: _names(activation.get(key))[0] for key in ("declared", "exercised", "unverified", "unavailable")
        }
        if "exercise_rate" in activation:
            exercise_rate = number(activation.get("exercise_rate"))
        else:
            exercise_rate = _ratio(len(activation_view["exercised"]), len(activation_view["declared"]))
        activation_view["exercise_rate"] = fmt_rate(exercise_rate)

    checks = [
        row
        for row in (
            _check_row("Order", _mapping(summary.get("order")), "satisfied", "edges"),
            _check_row("Handoff", _mapping(summary.get("handoff")), "passed", "checked"),
            _check_row("Conflict", _mapping(summary.get("conflict")), "passed", "checked"),
        )
        if row is not None
    ]
    missing = count(summary.get("n_missing_trajectory"))
    return {
        "scope": scope,
        "arm": arm,
        "arm_label": arm_label(arm, sum_of_parts_baseline=sum_of_parts_baseline),
        "n_trials": fmt_count(summary.get("n_trials")),
        "n_missing_trajectory": missing or 0,
        "activations_per_trial": fmt_count(activations.get("mean_per_trial")) if activations else "",
        "tool_selection": (
            {
                "applicable": _scored(tool_selection),
                "precision": fmt_rate(tool_selection.get("precision")),
                "recall": fmt_rate(tool_selection.get("recall")),
                "f1": fmt_rate(tool_selection.get("f1")),
                "decoy_calls": fmt_count(tool_selection.get("decoy_calls")),
                "decoy_call_rate": fmt_rate(tool_selection.get("decoy_call_rate")),
                "n_scored": fmt_count(tool_selection.get("n_scored")),
            }
            if tool_selection
            else None
        ),
        "arguments": argument_view,
        "mcp_calls": _signal_mcp(_mapping(summary.get("mcp_calls"))),
        "checks": checks,
        "activation": activation_view,
    }


def excluded_behavior(view: Mapping[str, Any], provenance: object) -> list[str]:
    """Return plain-language statements of what this plugin run did not evaluate."""
    source = _mapping(provenance)
    statements: list[str] = []
    sidecar_reason = sidecar_error_reason(source)
    if sidecar_reason:
        statements.append(sidecar_reason[:1].upper() + sidecar_reason[1:])
    coverage = _mapping(view.get("coverage"))
    if coverage and coverage.get("not_staged"):
        names = [f"{row['type']} {row['name']}".strip() for row in _sequence(coverage.get("not_staged_rows"))[:12]]
        statements.append(f"{coverage['headline']}: {', '.join(names)}" if names else str(coverage["headline"]))
    if coverage and coverage.get("staged_not_observed"):
        rows = _sequence(coverage.get("staged_not_observed_rows"))
        names = [f"{row['type']} {row['name']}".strip() for row in rows[:12]]
        omitted = max(0, int(coverage["staged_not_observed"]) - len(names))
        suffix = f" (+{omitted} more)" if omitted else ""
        statements.append(f"Staged but not observed in any plugin trial: {', '.join(names)}{suffix}")
    for label, field in (
        ("Provider-only MCP servers were not exercised", "provider_only_mcp_servers"),
        ("MCP servers declare configuration the runtime cannot apply", "mcp_unsupported_config"),
        ("Unresolved skill refs were not evaluated", "unresolved_skill_refs"),
        ("Unresolved rule refs were not evaluated", "unresolved_rule_refs"),
    ):
        names, omitted = _names(source.get(field), limit=12)
        if names:
            suffix = f" (+{omitted} more)" if omitted else ""
            statements.append(f"{label}: {', '.join(names)}{suffix}")
    integration = _mapping(view.get("integration"))
    if integration and not integration.get("measured"):
        statements.append(
            f"Integration (the plugin versus its own parts) was not measured: {integration.get('reason')}"
        )
    return statements
