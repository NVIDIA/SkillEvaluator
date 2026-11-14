# SPDX-License-Identifier: Apache-2.0
"""Public reporting hooks over the shared passive plugin projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skillevaluator.reporting.plugin_runtime_io import runtime_report_from_payload
from skillevaluator.reporting.plugin_runtime_render import _augment as _augment_runtime
from skillevaluator.reporting.plugin_static import project_plugin_static
from skillevaluator.reporting.plugin_static_render import _augment as _augment_static

if TYPE_CHECKING:
    from pathlib import Path

    from skillevaluator.models.result import ValidationResult
    from skillevaluator.reporting.base import ReporterBase
    from skillevaluator.reporting.plugin_runtime import RuntimePluginReport


def mark_incomplete_plugin_package(result: ValidationResult, engine_result: object) -> None:
    """Mark the result incomplete when the prepared plugin package was incomplete.

    The package's incomplete reasons, such as skill or rule references that were
    not staged, keep this from being a complete plugin evaluation. Scores stay visible.
    """
    provenance = engine_result.get("plugin_provenance") if isinstance(engine_result, dict) else None
    if isinstance(provenance, dict) and provenance.get("complete") is False:
        result.mark_scan_incomplete("plugin-components")


def attach_saved_plugin_evidence(result: ValidationResult, run_dir: Path, engine_result: dict | None) -> None:
    """Attach passive evidence without changing the canonical score payload.

    The result is also marked incomplete when the prepared plugin package was incomplete.
    """
    from skillevaluator.reporting.plugin_runtime_io import RUNTIME_REPORT_FILENAME, load_plugin_runtime_report

    mark_incomplete_plugin_package(result, engine_result)
    live = None
    if isinstance(engine_result, dict):
        inventory = engine_result.get("plugin_inventory")
        if isinstance(inventory, dict):
            result.metadata["plugin"] = inventory
        live = engine_result.get("plugin_runtime_report")
    try:
        try:
            (run_dir / RUNTIME_REPORT_FILENAME).lstat()
        except FileNotFoundError:
            if live is None:
                return
            report = runtime_report_from_payload(live)
        else:
            report = load_plugin_runtime_report(run_dir)
        result.metadata["plugin_runtime_report"] = report.payload()
    except (OSError, ValueError):
        result.metadata["plugin_runtime_report_error"] = "Saved plugin runtime evidence is unavailable"
        result.add_warning("Saved plugin runtime evidence is unavailable")


def runtime_report_for_results(results: list[ValidationResult]) -> RuntimePluginReport | None:
    """Read a validated display snapshot without opening a source or run directory."""
    snapshots = [
        result.metadata["plugin_runtime_report"] for result in results if "plugin_runtime_report" in result.metadata
    ]
    if not snapshots:
        return None
    report = runtime_report_from_payload(snapshots[0])
    # A combined report must never silently choose between conflicting runs.
    for payload in snapshots[1:]:
        if runtime_report_from_payload(payload) != report:
            raise ValueError("Conflicting plugin runtime report snapshots")
    return report


def render_plugin_report(reporter: ReporterBase, results: list[ValidationResult]) -> str:
    """Compose the existing format with bounded static and retained runtime views."""
    return augment_plugin_report(reporter, reporter.render_all(results), results)


def augment_plugin_report(reporter: ReporterBase, document: str, results: list[ValidationResult]) -> str:
    """Append passive evidence to an already rendered document or CLI stream."""
    projection = project_plugin_static(results)
    if projection is not None:
        document = _augment_static(reporter, document, results, projection, projection.payload())
    report = runtime_report_for_results(results)
    if reporter.name == "benchmark" and (projection is not None or report is not None):
        document = document.replace("# Skill Benchmark:", "# Plugin Benchmark:", 1)
    return _augment_runtime(reporter, document, report.payload()) if report is not None else document
