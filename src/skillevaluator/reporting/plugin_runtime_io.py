# SPDX-License-Identifier: Apache-2.0
"""Versioned private display snapshots, independent of optional runtime packages."""

from __future__ import annotations

import json
from pathlib import Path

from skillevaluator.reporting.plugin_runtime import SCHEMA_VERSION, SECTIONS, RuntimePluginReport
from skillevaluator.reporting.plugin_static import MAX_OUTPUT_BYTES, _clean_text, _number
from skillevaluator.utils.secure_fs import SecureRoot
from skillevaluator.utils.structured_data import load_bounded_json

RUNTIME_REPORT_FILENAME = "plugin_runtime_report.json"


def _settle(value: str) -> tuple[str, bool]:
    # _clean_text redacts before it joins lines, so one pass over multi-line
    # text is not always stable. Repeat until a pass changes nothing: a saved
    # snapshot then replays to exactly the report that was saved. Also report
    # whether any pass cut the text at the display bound.
    cut = False
    for _ in range(8):
        cleaned, truncated = _clean_text(value)
        cut |= truncated
        if cleaned == value:
            break
        value = cleaned
    return value, cut


def _validate(payload: object) -> RuntimePluginReport:
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported private runtime report schema")
    if not isinstance(payload.get("status"), str) or payload["status"] not in {"complete", "incomplete"}:
        raise ValueError("Invalid runtime evidence status")
    for key in ("evidence_complete", "truncated"):
        if type(payload.get(key)) is not bool:
            raise ValueError(f"Invalid runtime report {key}")
    if payload["evidence_complete"] is not (payload["status"] == "complete"):
        raise ValueError("Inconsistent runtime report completeness")
    cut = False

    def settle(value: str) -> str:
        nonlocal cut
        value, truncated = _settle(value)
        cut |= truncated
        return value

    for key in ("name", "run_id", "execution_status"):
        if not isinstance(payload.get(key), str):
            raise ValueError(f"Invalid runtime report {key}")
        payload[key] = settle(payload[key])
    source = payload.get("source")
    if (
        not isinstance(source, dict)
        or not source.keys() <= {"manifest", "manifest_sha256", "source_tree_sha256"}
        or not all(isinstance(value, str) for value in source.values())
    ):
        raise ValueError("Invalid runtime report source")
    payload["source"] = {key: settle(value) for key, value in source.items()}
    sections = payload.get("sections")
    if not isinstance(sections, list) or len(sections) != len(SECTIONS):
        raise ValueError("Invalid runtime report sections")
    seen, row_count = set(), 0
    for section in sections:
        if (
            not isinstance(section, dict)
            or not isinstance(section.get("key"), str)
            or section["key"] not in SECTIONS
            or section["key"] in seen
        ):
            raise ValueError("Invalid runtime report section")
        key = section["key"]
        if section.keys() != {"key", "title", "columns", "rows", "total", "omitted", "notes"}:
            raise ValueError("Unexpected runtime report section fields")
        seen.add(key)
        title, columns = SECTIONS[key]
        if section.get("title") != title or section.get("columns") != list(columns):
            raise ValueError("Incompatible runtime report columns")
        rows, notes = section.get("rows"), section.get("notes")
        if not isinstance(rows, list) or len(rows) > 64 or not isinstance(notes, list) or len(notes) > 8:
            raise ValueError("Invalid runtime report rows")
        for row in rows:
            if not isinstance(row, list) or len(row) != len(columns):
                raise ValueError("Invalid runtime report row width")
            for index, value in enumerate(row):
                if value is not None and type(value) not in (str, int, float):
                    raise ValueError("Invalid runtime report cell")
                if type(value) in (int, float) and _number(value) is None:
                    raise ValueError("Invalid runtime report number")
                if isinstance(value, str):
                    row[index] = settle(value)
        if not all(isinstance(note, str) for note in notes):
            raise ValueError("Invalid runtime report notes")
        section["notes"] = [settle(note) for note in notes]
        total, omitted = section.get("total"), section.get("omitted")
        if (
            type(total) is not int
            or type(omitted) is not int
            or not 0 <= omitted <= total
            or total != len(rows) + omitted
        ):
            raise ValueError("Invalid runtime report omission counts")
        row_count += len(rows)
    if row_count > 256:
        raise ValueError("Runtime report row bound exceeded")
    allowed = {
        "schema_version",
        "run_id",
        "name",
        "execution_status",
        "status",
        "evidence_complete",
        "truncated",
        "source",
        "sections",
    }
    if payload.keys() != allowed:
        raise ValueError("Unexpected runtime report fields")
    # Text cut while settling is as incomplete as text cut during projection.
    payload["truncated"] = payload["truncated"] or cut
    serialized = json.dumps(payload, ensure_ascii=True, allow_nan=False)
    if len(serialized.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise ValueError("Runtime report exceeds its publication bound")
    return RuntimePluginReport(serialized)


def runtime_report_from_payload(payload: object) -> RuntimePluginReport:
    """Validate a bounded, detached snapshot supplied by an in-process caller."""
    chunks = []
    size = 0
    for chunk in json.JSONEncoder(ensure_ascii=True, allow_nan=False).iterencode(payload):
        size += len(chunk)
        if size > MAX_OUTPUT_BYTES:
            raise ValueError("Runtime report exceeds its publication bound")
        chunks.append(chunk)
    detached = load_bounded_json("".join(chunks), max_nodes=50_000, max_tokens=50_000, max_collection_items=4096)
    return _validate(detached)


def save_plugin_runtime_report(report: RuntimePluginReport, run_dir: Path) -> None:
    """Atomically retain one bounded sanitized snapshot in the selected run directory."""
    # Writing needs the tier3 output helpers; replay stays free of tier3 imports.
    from skillevaluator.tier3.output_provenance import write_output_file_atomically

    checked = _validate(report.payload())
    encoded = checked._serialized.encode("utf-8")
    write_output_file_atomically(run_dir / RUNTIME_REPORT_FILENAME, encoded)


def load_plugin_runtime_report(run_dir: Path) -> RuntimePluginReport:
    """Replay a selected snapshot; missing older snapshots are explicitly unavailable."""
    try:
        with SecureRoot(run_dir) as root:
            raw, _ = root.read_bytes(Path(RUNTIME_REPORT_FILENAME), MAX_OUTPUT_BYTES)
        payload = load_bounded_json(raw.decode("utf-8"), max_nodes=50_000, max_tokens=50_000, max_collection_items=4096)
        return _validate(payload)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Saved private runtime report unavailable: {exc}") from exc
