# SPDX-License-Identifier: Apache-2.0
"""Passive, bounded views of retained private runtime observations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import PurePosixPath
from typing import Any

from skillevaluator.reporting.plugin_static import (  # noqa: F401 - re-exported bound
    MAX_OUTPUT_BYTES,
    _clean_text,
    _count,
    _mapping,
    _Projection,
)

SCHEMA_VERSION = 1
_PAIR_LABELS = {
    "effectiveness": "Effectiveness lift (full minus baseline)",
    "integration": "Bundle lift (full minus member skills)",
}
SECTIONS = {
    "run": ("Run and source", ("Field", "Value")),
    "diagnostics": ("Evidence limitations", ("Source", "Detail")),
    "conditions": (
        "Arm execution coverage",
        ("Agent", "Arm", "Execution", "Expected cases", "Scored cases", "Expected attempts", "Scored attempts"),
    ),
    "loading": ("Component loading", ("Agent", "Mode", "Component type", "Component", "State", "Evidence")),
    "coverage": ("Activation coverage", ("Agent", "Arm", "Component", "State", "Exercise rate")),
    "signals": ("Behavior signals", ("Agent", "Arm", "Signal", "Status", "Count", "Skipped", "Value")),
    "mcp": ("MCP proof", ("Server", "Proof", "Tools", "Detail")),
    "mcp_calls": ("Agent-observed MCP calls", ("Agent", "Arm", "Server", "Total", "Succeeded", "Failed", "Unknown")),
    "comparisons": (
        "Paired comparisons",
        ("Agent", "Measure", "Estimate", "CI low", "CI high", "Cases", "Precision", "Partial"),
    ),
    "eligibility": ("Measurement eligibility", ("Agent", "Measure", "Eligible", "Verdict", "Reason")),
    "exclusions": ("Excluded cases", ("Agent", "Case", "Reason")),
    "cost": ("Recorded cost", ("Agent", "Arm", "Tokens", "USD", "Successes", "Tokens/success", "USD/success")),
    "context": (
        "Measured context",
        ("Agent", "Status", "Delta tokens", "Pairs", "Partial", "Excluded hosted search", "Missing tokens", "Reason"),
    ),
}


@dataclass(frozen=True)
class RuntimePluginReport:
    """An immutable display snapshot; no trace, score or execution authority."""

    _serialized: str

    def payload(self) -> dict[str, Any]:
        return json.loads(self._serialized)


class _RuntimeProjection:
    def __init__(self) -> None:
        self.bound = _Projection()
        self.rows: dict[str, list[list]] = {key: [] for key in SECTIONS}
        self.totals = dict.fromkeys(SECTIONS, 0)
        self.complete = True

    def add(self, key: str, *cells: object) -> None:
        self.totals[key] += 1
        # Count every supplied row but retain only the section's display budget.
        if len(self.rows[key]) < 64:
            self.rows[key].append(list(cells))

    def items(self, raw: object, *, limit: int = 32):
        data = _mapping(raw)
        if len(data) > limit:
            self.bound.truncated = True
            self.issue("report bounds", f"{len(data) - limit} input groups omitted")
        return islice(data.items(), limit)

    def issue(self, source: str, detail: object) -> None:
        self.complete = False
        self.add("diagnostics", source, detail)

    def records(self, key: str, raw: object) -> list[dict]:
        records = self.bound.records(raw)
        if isinstance(raw, (list, tuple)):
            self.totals[key] += len(raw) - len(records)
        return records

    def labels(self, key: str, raw: object) -> list[str]:
        if not isinstance(raw, (list, tuple)):
            return []
        length = min(len(raw), self.bound.input_remaining, 4096)
        self.bound.input_remaining -= length
        labels = [item for item in islice(raw, length) if isinstance(item, str)]
        self.totals[key] += len(raw) - len(labels)
        self.bound.truncated |= length < len(raw)
        return labels

    def errors(self, source: str, raw: object) -> None:
        if isinstance(raw, list):
            for error in islice(raw, 64):
                if isinstance(error, str):
                    self.issue(source, error)
            if len(raw) > 64:
                self.bound.truncated = True
                self.issue(source, f"{len(raw) - 64} more diagnostics omitted")

    def finish(self) -> list[dict[str, Any]]:
        # Limitations and eligibility stay visible before large member inventories.
        order = [
            "run",
            "diagnostics",
            "eligibility",
            "conditions",
            "comparisons",
            "mcp",
            "context",
            "cost",
            "loading",
            "coverage",
            "signals",
            "mcp_calls",
            "exclusions",
        ]
        for key in order:
            title, columns = SECTIONS[key]
            self.bound.section(
                key,
                title,
                columns,
                self.rows[key],
                total=self.totals[key],
                notes=(
                    ["Skipped expectations concern components unsupported in that arm."]
                    if key == "signals"
                    else []
                    if self.totals[key]
                    else ["No retained observations for this section."]
                ),
            )
        return self.bound.finish()


def _load_views(projection: _RuntimeProjection, native: dict, decisions: dict) -> None:
    for agent, raw in projection.items(native):
        row = _mapping(raw)
        mode = row.get("mode") or _mapping(decisions.get(agent)).get("mode") or "unknown"
        for state in ("loaded", "listed", "staged", "not_loaded"):
            for item in projection.records("loading", row.get(state)):
                projection.add(
                    "loading",
                    agent,
                    mode,
                    item.get("type"),
                    item.get("name"),
                    state,
                    item.get("reason") or item.get("evidence"),
                )
        if row.get("fallback_trials") or row.get("listed") or row.get("staged"):
            projection.issue(agent, "Some trials have only staging evidence; harness loading is unverified")
    for agent, raw in projection.items(decisions):
        decision = _mapping(raw)
        projection.add("run", f"{agent} loading", decision.get("mode"))
        if decision.get("reason"):
            projection.add("run", f"{agent} loading reason", decision["reason"])
        if decision.get("mode") == "native" and not _mapping(native.get(agent)):
            projection.issue(agent, "Native loading evidence is unavailable")


def _signal_views(projection: _RuntimeProjection, agents: object) -> None:
    if not _mapping(agents):
        projection.issue("agents", "No retained agent observations")
    for agent, raw in projection.items(agents):
        info = _mapping(raw)
        for arm, raw_condition in projection.items(info.get("conditions"), limit=4):
            condition = _mapping(raw_condition)
            projection.add(
                "conditions",
                agent,
                arm,
                condition.get("execution_status"),
                *(
                    _count(condition.get(key))
                    for key in ("expected_cases", "scored_cases", "expected_attempts", "scored_attempts")
                ),
            )
        summaries = _mapping(info.get("plugin_signals_summary"))
        if not summaries:
            projection.issue(agent, "Runtime behavior evidence is unavailable")
        for arm, raw_summary in projection.items(summaries, limit=4):
            summary = _mapping(raw_summary)
            if summary.get("evidence_complete") is not True:
                projection.issue(f"{agent}/{arm}", "Runtime behavior evidence is incomplete")
            for key in ("n_trials", "n_missing_trials", "n_missing_trajectory", "n_incomplete_trajectory"):
                projection.add(
                    "signals",
                    agent,
                    arm,
                    key,
                    "recorded" if _count(summary.get(key)) is not None else "unknown",
                    _count(summary.get(key)),
                    None,
                    None,
                )
            for key, metric in (
                ("routing", "f1"),
                ("tool_selection", "f1"),
                ("arguments", "pass_rate"),
                ("order", "satisfaction_rate"),
                ("handoff", "pass_rate"),
                ("conflict", "pass_rate"),
            ):
                value = _mapping(summary.get(key))
                projection.add(
                    "signals",
                    agent,
                    arm,
                    key,
                    value.get("status") or "unavailable",
                    _count(value.get("n_scored")),
                    _count(value.get("skipped")),
                    value.get(metric),
                )
            coverage = _mapping(summary.get("activation_coverage"))
            declared = coverage.get("declared")
            if isinstance(declared, list):
                # Membership work is bounded even for an in-memory hostile input.
                states = {}
                for key in ("exercised", "unavailable"):
                    values = coverage.get(key, [])
                    if not isinstance(values, list):
                        projection.issue(f"{agent}/{arm}", f"Malformed {key} coverage evidence")
                        values = []
                    states[key] = {value for value in islice(values, 4096) if isinstance(value, str)}
                for label in islice(declared, 4096):
                    if isinstance(label, str):
                        state = (
                            "exercised"
                            if label in states["exercised"]
                            else "unavailable"
                            if label in states["unavailable"]
                            else "unverified"
                        )
                        projection.add(
                            "coverage", agent, arm, label, state, _mapping(coverage.get("exercise_rate")).get(label)
                        )
                projection.totals["coverage"] += max(0, len(declared) - 4096)
            for server, counts in projection.items(_mapping(summary.get("mcp_calls")).get("by_server"), limit=64):
                counts = _mapping(counts)
                projection.add(
                    "mcp_calls",
                    agent,
                    arm,
                    server,
                    *(_count(counts.get(field)) for field in ("total", "succeeded", "failed", "unknown")),
                )


def _comparison_views(projection: _RuntimeProjection, comparisons: dict) -> None:
    if comparisons and comparisons.get("status") != "recorded":
        projection.issue("comparisons", "Comparison evidence is incomplete")
    projection.errors("comparisons", comparisons.get("errors"))
    for agent, raw in projection.items(comparisons.get("agents")):
        stats = _mapping(raw)
        ladder = _mapping(stats.get("lift_ladder"))
        for key in ("basis", "expected_cases", "n_cases", "partial"):
            if key in ladder:
                projection.add("run", f"{agent} comparison {key}", ladder[key])
        for key in ("confidence", "resamples", "method", "interval"):
            if key in _mapping(ladder.get("bootstrap")):
                projection.add("run", f"{agent} bootstrap {key}", ladder["bootstrap"][key])
        estimates = [*projection.items(ladder.get("steps"), limit=4)]
        estimates.extend((name, ladder[name]) for name in ("bundle", "glue") if isinstance(ladder.get(name), dict))
        estimates.extend(
            (
                _PAIR_LABELS.get(name, _PAIR_LABELS["effectiveness"]),
                step,
            )
            for name, step in projection.items(stats.get("lift_uncertainty"), limit=2)
        )
        for name, raw_step in estimates:
            step = _mapping(raw_step)
            projection.add(
                "comparisons",
                agent,
                name,
                step.get("estimate", step.get("lift")),
                step.get("ci_low"),
                step.get("ci_high"),
                step.get("n_cases", step.get("n")),
                step.get("precision"),
                step.get("partial", ladder.get("partial")),
            )
        claim = _mapping(stats.get("integration_claim"))
        projection.add(
            "eligibility",
            agent,
            claim.get("label") or "Integration lift",
            claim.get("eligible"),
            claim.get("verdict"),
            claim.get("reason") or ("claim unavailable" if not claim else None),
        )
        loose = _mapping(stats.get("loose_parts"))
        if loose.get("measured") is not True:
            projection.add(
                "eligibility",
                agent,
                "Loose-parts control",
                False,
                "not measured",
                loose.get("not_measured_reason") or "Measurement evidence unavailable",
            )
        else:
            projection.add(
                "eligibility",
                agent,
                "Content parity",
                loose.get("equal_across_arms"),
                "recorded",
                loose.get("content_basis"),
            )
            for arm, pieces in projection.items(_mapping(loose.get("load_evidence")).get("pieces"), limit=4):
                for piece, raw_counts in projection.items(pieces, limit=256):
                    counts = _mapping(raw_counts)
                    state = (
                        "not_loaded"
                        if counts.get("not_loaded")
                        else "unverified"
                        if counts.get("unverifiable")
                        else "loaded"
                        if counts.get("loaded")
                        else "unknown"
                    )
                    projection.add(
                        "loading",
                        agent,
                        arm,
                        "comparison piece",
                        piece,
                        state,
                        f"loaded={_count(counts.get('loaded'))}; not_loaded={_count(counts.get('not_loaded'))}; unverified={_count(counts.get('unverifiable'))}",
                    )
        seen_exclusions = set()
        for entry in projection.records("exclusions", ladder.get("excluded_cases")):
            case, reason = entry.get("case_id", entry.get("case")), entry.get("reason")
            projection.add("exclusions", agent, case, reason)
            if isinstance(case, str) and isinstance(reason, str):
                seen_exclusions.add((case, reason))
        eligibility = _mapping(stats.get("comparison_eligibility"))
        for measure, label in _PAIR_LABELS.items():
            mismatches = _mapping(eligibility.get(measure)).get("score_policy_mismatches")
            for case in projection.labels("exclusions", mismatches):
                reason = f"{label}: score_policy_mismatch"
                if (case, reason) not in seen_exclusions:
                    projection.add("exclusions", agent, case, reason)
                    seen_exclusions.add((case, reason))
        for arm, raw_cost in projection.items(stats.get("cost"), limit=4):
            cost = _mapping(raw_cost)
            projection.add(
                "cost",
                agent,
                arm,
                *(
                    cost.get(key)
                    for key in ("total_tokens", "total_usd", "successes", "tokens_per_success", "usd_per_success")
                ),
            )
        context = _mapping(stats.get("context_cost_measured"))
        excluded = _mapping(context.get("excluded"))
        projection.add(
            "context",
            agent,
            context.get("status") or "unavailable",
            context.get("delta_tokens_mean"),
            context.get("n_pairs"),
            context.get("partial"),
            excluded.get("hosted_search"),
            excluded.get("missing_tokens"),
            context.get("reason"),
        )


def project_plugin_runtime(
    result: Mapping[str, Any], *, provenance: Mapping[str, Any] | None = None
) -> RuntimePluginReport | None:
    """Copy recorded observations into bounded views without rescoring or I/O."""
    provenance = dict(provenance or {})
    if not any(
        key in result for key in ("runtime_evidence", "native_evidence", "private_comparisons")
    ) and not provenance.get("plugin"):
        return None
    projection = _RuntimeProjection()
    observed = _mapping(provenance.get("runtime_evidence") or result.get("runtime_evidence"))
    comparisons = _mapping(provenance.get("private_comparisons") or result.get("private_comparisons"))
    native = _mapping(provenance.get("native_evidence") or result.get("native_evidence"))
    if observed.get("evidence_complete") is not True:
        projection.issue("runtime", "Runtime evidence is unavailable or incomplete")
    if provenance.get("complete") is False:
        projection.issue("preparation", "The prepared package has incomplete evidence")
    projection.errors("runtime", observed.get("errors"))
    # ``error`` is the engine's compatibility copy of ``execution_errors``.
    execution_errors = result.get("execution_errors")
    projection.errors("execution", execution_errors if isinstance(execution_errors, list) else result.get("error"))
    for key in ("native_evidence_error", "runtime_report_error"):
        if error := provenance.get(key) or result.get(key):
            projection.issue(key, error)
    for reason in islice(provenance.get("incomplete_reasons") or (), 64):
        projection.issue("preparation", reason)
    run_id = PurePosixPath(str(result.get("run_dir") or "unknown").replace("\\", "/")).name
    projection.add("run", "Run", run_id)
    projection.add("run", "Execution status", result.get("execution_status") or "unknown")
    if "attempts" in provenance:
        projection.add("run", "Setup attempts", _count(provenance["attempts"]))
    for reason in islice(provenance.get("retry_reasons") or (), 5):
        projection.add("run", "Setup retry reason", reason)
    source = _mapping(provenance.get("source") or observed.get("source"))
    source = {
        key: _clean_text(source[key])[0]
        for key in ("manifest", "manifest_sha256", "source_tree_sha256")
        if isinstance(source.get(key), str)
    }
    for key, value in source.items():
        projection.add("run", key, value)
    _load_views(projection, native, _mapping(provenance.get("load_decisions")))
    _signal_views(projection, result.get("agents"))
    for name, raw in projection.items(observed.get("mcp_proof"), limit=64):
        proof = _mapping(raw)
        projection.add(
            "mcp",
            name,
            proof.get("status") or "unknown",
            projection.bound.names(proof.get("tools")),
            proof.get("detail"),
        )
    _comparison_views(projection, comparisons)
    sections = projection.finish()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": _clean_text(run_id)[0],
        "name": _clean_text(str(provenance.get("plugin") or "Plugin runtime"))[0],
        "execution_status": _clean_text(str(result.get("execution_status") or "unknown"))[0],
        "status": "complete" if projection.complete else "incomplete",
        "evidence_complete": projection.complete,
        "truncated": projection.bound.truncated,
        "source": source,
        "sections": sections,
    }
    # Settle sanitization through the validator that replay uses, so the live
    # report, its provenance copy and the saved snapshot are the same bytes.
    from skillevaluator.reporting.plugin_runtime_io import _validate

    return _validate(payload)
