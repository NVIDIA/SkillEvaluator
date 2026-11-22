# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded projections of scanner completeness evidence, never scan verdicts."""

from __future__ import annotations

import unicodedata

# Per-result and per-record caps keep the JSON report bounded against a hostile
# or runaway scanner payload. Real libraries stay well below them: a 227-skill
# library with 521 ledger rows was cut to 268 under the old 20-row cap, which
# hid half of the incomplete scope (#861). Each record states how many rows it
# retains out of the total, so any truncation that does occur is visible.
_MAX_RECORDS = 512
_MAX_ROWS = 500
_MAX_LIMITATIONS = 8


def _text(value: object) -> str:
    # Result models use the merge helper during import-sensitive CLI paths.
    # Loading utils at module scope would re-import the partially defined models.
    from skillevaluator.utils.redaction import redact_printable_text

    if not isinstance(value, str):
        return "unknown"
    safe = redact_printable_text(value, max_input_chars=4096, max_len=240)
    return " ".join("".join(" " if unicodedata.category(c).startswith("C") else c for c in safe).split()) or "unknown"


def _count(value: object) -> int:
    return value if type(value) is int and 0 <= value <= 1_000_000_000 else 0


def _line_number(value: object) -> int | None:
    return value if type(value) is int and 0 < value <= 1_000_000_000 else None


def parser_diagnostic_record(value: object) -> dict | None:
    """Project ``parser_diagnostic`` without source text or byte offsets."""
    if not isinstance(value, dict):
        return None
    item: dict = {}
    for key in ("cause", "check", "unchecked_scope"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            item[key] = _text(raw)
    start = _line_number(value.get("start_line"))
    end = _line_number(value.get("end_line"))
    if start is not None and end is not None and end >= start:
        item["start_line"] = start
        item["end_line"] = end
    checked = value.get("full_input_checked")
    if type(checked) is bool:
        item["full_input_checked"] = checked
    return item or None


def completeness_record(completeness: dict, stage: str) -> dict:
    """Project only diagnostic fields; do not copy arbitrary scanner payloads."""
    rows = completeness.get("ledger_exceptions")
    rows = rows if isinstance(rows, list) else []
    limits = completeness.get("limitations")
    limits = limits if isinstance(limits, list) else []
    exceptions = []
    for row in rows[:_MAX_ROWS]:
        if not isinstance(row, dict):
            continue
        item = {key: _text(row.get(key)) for key in ("path", "phase", "reason_code")}
        for key in ("start_line", "end_line"):
            if _count(row.get(key)):
                item[key] = _count(row[key])
        analyzers = row.get("analyzers")
        if isinstance(analyzers, list):
            item["analyzers"] = [_text(value) for value in analyzers[:8]]
            item["omitted_analyzers"] = max(0, len(analyzers) - 8) + _count(row.get("omitted_analyzers"))
        diagnostic = parser_diagnostic_record(row.get("parser_diagnostic"))
        if diagnostic is not None:
            item["parser_diagnostic"] = diagnostic
        exceptions.append(item)
    omitted_exceptions = max(0, len(rows) - _MAX_ROWS)
    return {
        "stage": stage if isinstance(stage, str) and stage in {"skillspector", "skillspector-llm"} else "unknown",
        "reported_status": "partial",
        "ledger_exceptions": exceptions,
        "limitations": [_text(value) for value in limits[:_MAX_LIMITATIONS]],
        "omitted_exceptions": omitted_exceptions,
        # Rows this record retains out of every row the scanner reported, so a
        # reader can see "N of M" without summing two fields.
        "ledger_rows_retained": len(exceptions),
        "ledger_rows_total": len(exceptions) + omitted_exceptions,
        "omitted_limitations": max(0, len(limits) - _MAX_LIMITATIONS),
    }


def sanitize_completeness(value: object) -> dict:
    """Re-project saved metadata at report boundaries and preserve omission counts."""
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        return {}
    records = []
    for source in value["records"][:_MAX_RECORDS]:
        if not isinstance(source, dict):
            continue
        record = completeness_record(source, source.get("stage"))
        for key in ("omitted_exceptions", "omitted_limitations"):
            record[key] += _count(source.get(key))
        record["ledger_rows_total"] = record["ledger_rows_retained"] + record["omitted_exceptions"]
        if "skill" in source:
            record["skill"] = _text(source["skill"])
        records.append(record)
    return {
        "records": records,
        "omitted_records": _count(value.get("omitted_records")) + max(0, len(value["records"]) - _MAX_RECORDS),
    }


def merge_completeness(left: object, right: object, *, prefix: str | None = None) -> dict:
    """Keep both scan stages and skill identities when result objects are merged."""
    a, b = sanitize_completeness(left), sanitize_completeness(right)
    if not a and not b:
        return {}
    incoming = b.get("records", [])
    if prefix is not None and incoming:
        skill = _text(prefix)
        incoming = [{**record, "skill": skill} for record in incoming]
    records = a.get("records", []) + incoming
    omitted = _count(a.get("omitted_records", 0) + b.get("omitted_records", 0))
    # Both sets of records are already projected. Cap the combined list
    # without redacting all of its rows again.
    return {
        "records": records[:_MAX_RECORDS],
        "omitted_records": omitted + max(0, len(records) - _MAX_RECORDS),
    }
