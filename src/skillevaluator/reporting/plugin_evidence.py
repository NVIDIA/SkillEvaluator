# SPDX-License-Identifier: Apache-2.0
"""Route bounded, retained plugin observations into the native report tabs.

This module consumes display snapshots only. It never discovers a plugin,
opens its source, runs an agent, or recomputes an observation. Canonical
snapshots and sanitized display sections have independent ownership.
"""

from __future__ import annotations

import copy
import json
import re

from skillevaluator.reporting.plugin_static import (
    MAX_FIELD_INPUT,
    MAX_OUTPUT_BYTES,
    MAX_SECTION_ROWS,
    MAX_TOTAL_ROWS,
    _clean_text,
    _count,
    _number,
)

_SLOTS = ("tier1", "tier2", "overview", "agents", "trials", "provenance")
_STATIC_SLOTS = {
    "catalog_skills": "tier2",
    "catalog_plugins": "tier2",
}
_RUNTIME_SLOTS = {
    "eligibility": "overview",
    "conditions": "overview",
    "loading": "agents",
    "coverage": "agents",
    "signals": "agents",
    "mcp": "agents",
    "mcp_calls": "agents",
    "comparisons": "trials",
    "comparison": "trials",  # Compatibility with early retained private views.
    "context": "trials",
    "cost": "trials",
}
_SECTION_FIELDS = {"key", "title", "columns", "rows", "total", "omitted", "notes"}


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(f"Invalid plugin evidence: {detail}")


def _snapshot(value: object, owner: str) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        payload = getattr(value, "payload", None)
        _require(callable(payload), "expected a snapshot or report with payload()")
        value = payload()
    _require(isinstance(value, dict), "expected a snapshot object")
    text_remaining = MAX_OUTPUT_BYTES

    def text(value: object) -> None:
        nonlocal text_remaining
        _require(isinstance(value, str) and len(value) <= MAX_FIELD_INPUT, "text bound or type")
        text_remaining -= len(value)
        _require(text_remaining >= 0, "aggregate text bound")

    fields = {"schema_version", "name", "status", "truncated", "sections"}
    if owner == "runtime":
        fields |= {"run_id", "execution_status", "evidence_complete", "source"}
    _require(value.keys() == fields, "snapshot fields")
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1, "schema version")
    for field in ("name", "status"):
        text(value[field])
    _require(type(value["truncated"]) is bool, "truncated must be boolean")
    if owner == "runtime":
        for field in ("run_id", "execution_status"):
            text(value[field])
        _require(type(value["evidence_complete"]) is bool, "evidence completeness must be boolean")
        _require(value["status"] in {"complete", "incomplete"}, "runtime status")
        _require(value["evidence_complete"] is (value["status"] == "complete"), "runtime completeness")
        source = value["source"]
        _require(
            isinstance(source, dict) and source.keys() <= {"manifest", "manifest_sha256", "source_tree_sha256"},
            "runtime source",
        )
        for label in source.values():
            text(label)
    else:
        _require(value["status"] in {"passed", "failed", "incomplete"}, "static status")
    sections = value["sections"]
    _require(isinstance(sections, list) and len(sections) <= 32, "section bound or type")
    seen, row_count = set(), 0
    for section in sections:
        _require(isinstance(section, dict) and section.keys() == _SECTION_FIELDS, "section fields")
        key = section["key"]
        _require(isinstance(key, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key) is not None, "section key")
        _require(key not in seen, "duplicate section")
        seen.add(key)
        text(section["title"])
        columns, rows, notes = section["columns"], section["rows"], section["notes"]
        _require(isinstance(columns, list) and 0 < len(columns) <= 32, "column bound or type")
        _require(isinstance(rows, list) and len(rows) <= MAX_SECTION_ROWS, "row bound or type")
        _require(isinstance(notes, list) and len(notes) <= 8, "note bound or type")
        for label in [*columns, *notes]:
            if label is not None:  # Static projections may retain an unavailable note.
                text(label)
        _require(all(isinstance(column, str) for column in columns), "column type")
        for row in rows:
            _require(isinstance(row, list) and len(row) == len(columns), "row width")
            for cell in row:
                if isinstance(cell, str):
                    text(cell)
                else:
                    _require(cell is None or _number(cell) is not None, "cell type or number")
        total, omitted = section["total"], section["omitted"]
        _require(_count(total) is not None and _count(omitted) is not None, "omission count type")
        _require(total == len(rows) + omitted, "omission counts")
        row_count += len(rows)
    _require(row_count <= MAX_TOTAL_ROWS, "total row bound")
    # The shape is now shallow, acyclic and scalar-only; serialization cannot
    # traverse arbitrary caller objects or unbounded nested structures.
    serialized = json.dumps(value, ensure_ascii=True, allow_nan=False)
    _require(len(serialized.encode("utf-8")) <= MAX_OUTPUT_BYTES, "snapshot byte bound")
    return copy.deepcopy(value)


def build_plugin_evidence(static: object = None, runtime: object = None) -> dict:
    """Accept existing reports or bounded snapshots, retaining their exact data.

    The six tab values contain independent display sections. ``static`` and
    ``runtime`` contain canonical snapshots for the existing HTML export
    envelope; they are never annotated or shortened for presentation.
    """
    view = {slot: [] for slot in _SLOTS}
    for owner, supplied in (("static", static), ("runtime", runtime)):
        snapshot = _snapshot(supplied, owner)
        view[owner] = snapshot
        if snapshot is None:
            continue
        destinations = _STATIC_SLOTS if owner == "static" else _RUNTIME_SLOTS
        for raw in snapshot["sections"]:
            section = copy.deepcopy(raw)
            shortened = False

            def clean(value: object) -> object:
                nonlocal shortened
                if not isinstance(value, str):
                    return value
                text, cut = _clean_text(value)
                shortened |= cut
                return text

            section["title"] = clean(section["title"])
            section["columns"] = [clean(column) for column in section["columns"]]
            section["rows"] = [[clean(cell) for cell in row] for row in section["rows"]]
            section["notes"] = [clean(note) for note in section["notes"]]
            section.update(
                owner=owner,
                source_label=f"Plugin {owner} evidence",
                name=clean(snapshot["name"]),
                status=clean(snapshot["status"]),
                truncated=snapshot["truncated"],
                run_id=clean(snapshot.get("run_id")),
                execution_status=clean(snapshot.get("execution_status")),
                text_shortened=shortened,
                first=False,
            )
            destination = destinations.get(section["key"], "tier1" if owner == "static" else "provenance")
            view[destination].append(section)
        first = next((section for slot in _SLOTS for section in view[slot] if section["owner"] == owner), None)
        if first is not None:
            first["first"] = True
    return view
