# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional local Streamlit dashboard for quantitative Tier 3 reports.

Run through ``skillevaluator dashboard`` so the dashboard extra is checked before
starting Streamlit. Only normalized quantitative data is displayed or exported.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

from skillevaluator.reporting.dashboard_data import comparison_rows, load_dashboard_path, parse_dashboard_report

_DIMENSIONS = ("skill", "model", "harness", "features")
_MAX_UPLOAD_BYTES = 16 * 1024 * 1024
_CONDITION_NAMES = {"with_skill": "With skill", "without_skill": "Without skill", "adverse": "Adverse skill"}
_USAGE_METRICS = ("total_tokens", "prompt_tokens", "completion_tokens", "cached_tokens", "duration_seconds", "cost_usd")
_PERFORMANCE_COLUMNS = (
    "skill",
    "model",
    "harness",
    "condition",
    "score",
    "total_tokens",
    "duration_seconds",
    "cost_usd",
    "status",
    "features",
    "pass_rate",
    "trials",
    "coverage",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "run_id",
    "source",
)
_COMPARISON_COLUMNS = (
    "skill",
    "model",
    "harness",
    "score_delta",
    "total_tokens_delta",
    "total_tokens_percent",
    "duration_seconds_delta",
    "duration_seconds_percent",
    "cost_usd_delta",
    "cost_usd_percent",
    "comparison_status",
    "features",
    "coverage",
    "with_score",
    "without_score",
    "with_pass_rate",
    "without_pass_rate",
    "pass_rate_delta",
    *(
        field
        for metric in _USAGE_METRICS
        for field in (f"with_{metric}", f"without_{metric}", f"{metric}_delta", f"{metric}_percent")
        if field
        not in {
            "total_tokens_delta",
            "total_tokens_percent",
            "duration_seconds_delta",
            "duration_seconds_percent",
            "cost_usd_delta",
            "cost_usd_percent",
        }
    ),
    "run_id",
    "source",
)


def rows_csv(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    """Export visible quantitative columns with spreadsheet-safe text cells."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        cells: dict[str, Any] = {}
        for column in columns:
            value = row.get(column)
            if isinstance(value, str) and (
                value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))
            ):
                value = "'" + value
            cells[column] = value
        writer.writerow(cells)
    return buffer.getvalue()


def _filter_rows(rows: list[dict[str, Any]], field: str, option_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    options = sorted({str(row.get(field) or "unknown") for row in option_rows}, key=str.casefold)
    selected = st.sidebar.multiselect(
        field.capitalize(),
        options,
        key=f"filter_{field}",
        format_func=(lambda value: _CONDITION_NAMES.get(value, value)) if field == "condition" else str,
    )
    return [row for row in rows if str(row.get(field) or "unknown") in selected] if selected else rows


def _performance_column_config() -> dict[str, Any]:
    config: dict[str, Any] = {
        "skill": st.column_config.TextColumn("Skill", width="medium"),
        "model": st.column_config.TextColumn("Model", width="medium"),
        "harness": st.column_config.TextColumn("Harness"),
        "features": st.column_config.TextColumn("Features", width="medium"),
        "condition": st.column_config.TextColumn("Condition"),
        "score": st.column_config.NumberColumn("Score", format="%.3f", width="small"),
        "pass_rate": st.column_config.NumberColumn("Pass rate", format="percent", width="small"),
        "duration_seconds": st.column_config.NumberColumn("Duration (s)", format="%.2f", width="small"),
        "cost_usd": st.column_config.NumberColumn("Cost (USD)", format="$%.4f", width="small"),
        "status": st.column_config.TextColumn("Status"),
        "trials": st.column_config.NumberColumn("Trials", format="%d"),
        "coverage": st.column_config.TextColumn("Measurement coverage", width="large"),
        "run_id": st.column_config.TextColumn("Run", width="medium"),
        "source": st.column_config.TextColumn("Source", width="large"),
    }
    for field in ("total_tokens", "prompt_tokens", "completion_tokens", "cached_tokens"):
        config[field] = st.column_config.NumberColumn(field.replace("_", " ").capitalize(), format="%d", width="small")
    return config


def _comparison_column_config(columns: tuple[str, ...]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    short_labels = {
        "score_delta": "Score Δ",
        "total_tokens_delta": "Tokens Δ",
        "total_tokens_percent": "Tokens Δ (%)",
        "duration_seconds_delta": "Time Δ (s)",
        "duration_seconds_percent": "Time Δ (%)",
        "cost_usd_delta": "Cost Δ (USD)",
        "cost_usd_percent": "Cost Δ (%)",
    }
    for field in columns:
        label = short_labels.get(field, field.replace("_", " ").capitalize())
        if field.endswith("_percent"):
            config[field] = st.column_config.NumberColumn(label, format="%.1f%%", width="small")
        elif "pass_rate" in field:
            config[field] = st.column_config.NumberColumn(label, format="percent", width="small")
        elif field.endswith("_score") or field == "score_delta":
            config[field] = st.column_config.NumberColumn(label, format="%.3f", width="small")
        elif "tokens" in field:
            config[field] = st.column_config.NumberColumn(label, format="%d", width="small")
        elif "duration_seconds" in field:
            config[field] = st.column_config.NumberColumn(label, format="%.2f", width="small")
        elif "cost_usd" in field:
            config[field] = st.column_config.NumberColumn(label, format="$%.4f", width="small")
        else:
            config[field] = st.column_config.TextColumn(label)
    return config


def _show_performance(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No rows match these filters. Clear a filter to include more results.")
        return
    visible = [{column: row.get(column) for column in _PERFORMANCE_COLUMNS} for row in rows]
    st.dataframe(
        visible,
        column_order=_PERFORMANCE_COLUMNS,
        column_config=_performance_column_config(),
        width="stretch",
        hide_index=True,
        key="performance_table",
    )
    st.download_button(
        "Download performance CSV",
        rows_csv(rows, _PERFORMANCE_COLUMNS),
        file_name="skill-performance.csv",
        mime="text/csv",
        key="download_performance",
    )


def _show_comparisons(rows: list[dict[str, Any]]) -> None:
    comparisons = comparison_rows(rows)
    st.caption("Conditions are paired within each run, skill, model, harness, and feature configuration.")
    if not comparisons:
        st.info("No comparisons are available. Load reports with with-skill and without-skill conditions.")
        return
    view = st.selectbox(
        "Comparison metrics",
        ("Overview", "Quality", "Tokens", "Duration", "Cost", "All metrics"),
        key="comparison_metrics",
    )
    identity = ("skill", "model", "harness")
    details = ("comparison_status", "features", "coverage", "run_id", "source")
    metric_sets = {
        "Overview": (
            "score_delta",
            "total_tokens_delta",
            "total_tokens_percent",
            "duration_seconds_delta",
            "duration_seconds_percent",
            "cost_usd_delta",
            "cost_usd_percent",
        ),
        "Quality": (
            "with_score",
            "without_score",
            "score_delta",
            "with_pass_rate",
            "without_pass_rate",
            "pass_rate_delta",
        ),
        "Tokens": tuple(
            field
            for metric in _USAGE_METRICS
            if "tokens" in metric
            for field in (f"with_{metric}", f"without_{metric}", f"{metric}_delta", f"{metric}_percent")
        ),
        "Duration": (
            "with_duration_seconds",
            "without_duration_seconds",
            "duration_seconds_delta",
            "duration_seconds_percent",
        ),
        "Cost": ("with_cost_usd", "without_cost_usd", "cost_usd_delta", "cost_usd_percent"),
    }
    columns = (*identity, *metric_sets[view], *details) if view in metric_sets else _COMPARISON_COLUMNS
    visible = [{column: row.get(column) for column in columns} for row in comparisons]
    st.dataframe(
        visible,
        column_order=columns,
        column_config=_comparison_column_config(columns),
        width="stretch",
        hide_index=True,
        key="comparison_table",
    )
    st.caption(
        "Deltas are with skill minus without skill. Usage percentages use the without-skill value as the baseline. "
        "Unavailable or ineligible comparisons stay blank. These are descriptive results, not significance tests."
    )
    st.download_button(
        "Download comparison CSV",
        rows_csv(visible, columns),
        file_name="skill-comparison.csv",
        mime="text/csv",
        key="download_comparisons",
    )


def main(paths: list[str] | None = None) -> None:
    """Render the app, accepting CLI paths or a supplied path list for embedding."""
    st.set_page_config(page_title="Skill performance · SkillEvaluator", page_icon="▦", layout="wide")
    st.title("Skill performance")
    st.caption("SKILLEVALUATOR · Compare agent quality, tokens, duration, and cost across evaluation runs.")
    st.sidebar.header("Reports")
    uploads = st.sidebar.file_uploader(
        "Upload JSON reports",
        type=["json"],
        accept_multiple_files=True,
        help="Use Tier 3 JSON reports or consolidated SkillEvaluator JSON reports, up to 16 MiB per file.",
        key="report_uploads",
    )
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_paths: set[Path] = set()
    for supplied_path in paths or []:
        try:
            path = Path(supplied_path).expanduser().resolve()
        except (OSError, ValueError, RuntimeError) as exc:
            warnings.append(f"{supplied_path}: cannot resolve dashboard input: {exc}")
            continue
        if path in seen_paths:
            continue
        seen_paths.add(path)
        data = load_dashboard_path(path)
        rows.extend(data.rows)
        warnings.extend(data.warnings)
    for index, uploaded in enumerate(uploads):
        contents = uploaded.getvalue()
        if len(contents) > _MAX_UPLOAD_BYTES:
            warnings.append(f"{uploaded.name}: report exceeds 16 MiB. Upload a smaller evaluation report.")
            continue
        try:
            payload = json.loads(contents)
        except (ValueError, UnicodeError, RecursionError):
            warnings.append(f"{uploaded.name}: invalid JSON. Upload a UTF-8 JSON evaluation report.")
            continue
        data = parse_dashboard_report(payload, source=f"upload:{index + 1}:{uploaded.name}")
        rows.extend(data.rows)
        warnings.extend(data.warnings)
    if warnings:
        with st.expander(f"Report notices ({len(warnings)})", expanded=True):
            for warning in dict.fromkeys(warnings):
                st.warning(warning)
    if not rows:
        st.info("Upload JSON evaluation reports to start comparing results.")
        st.caption("You can also open saved run directories with skillevaluator dashboard PATH [PATH …].")
        return

    st.sidebar.header("Filter results")
    st.sidebar.caption("Leave a filter empty to include all values.")
    dimension_rows = rows
    for field in _DIMENSIONS:
        dimension_rows = _filter_rows(dimension_rows, field, rows)
    dimension_rows = _filter_rows(dimension_rows, "status", rows)
    performance_rows = _filter_rows(dimension_rows, "condition", rows)
    st.sidebar.caption("The condition filter applies to performance. Comparisons retain both conditions.")

    run_count = len({(row["source"], row["run_id"]) for row in performance_rows})
    skill_count = len({row["skill"] for row in performance_rows})
    scored_count = sum(row.get("score") is not None for row in performance_rows)
    summary = st.columns(4)
    summary[0].metric("Runs", run_count)
    summary[1].metric("Skills", skill_count)
    summary[2].metric("Condition rows", len(performance_rows))
    summary[3].metric("Scored rows", scored_count)
    st.caption("Blank cells mean a measurement is missing, incomplete, or ambiguous. Unknown settings remain labeled unknown.")
    performance_tab, comparison_tab = st.tabs(["Performance", "With vs without skill"])
    with performance_tab:
        _show_performance(performance_rows)
    with comparison_tab:
        _show_comparisons(dimension_rows)


if __name__ == "__main__":
    main(sys.argv[1:])
