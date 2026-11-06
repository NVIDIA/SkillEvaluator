# SPDX-License-Identifier: Apache-2.0
"""Private runtime adapters over one retained, passive evidence projection."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from skillevaluator.reporting.plugin_static_render import _cli, _markdown, render_plugin_static_reports

if TYPE_CHECKING:
    from collections.abc import Iterable

    from skillevaluator.models.result import ValidationResult
    from skillevaluator.reporting.base import ReporterBase
    from skillevaluator.reporting.plugin_runtime import RuntimePluginReport

_TITLE = "Plugin runtime evidence"
_NOTICE = "Advisory observations do not change scores or validation gates."
_TRUNCATION_NOTICE = "Some runtime evidence was omitted by report bounds; see section counts."


def _augment(reporter: ReporterBase, document: str, payload: dict) -> str:
    if reporter.name == "json":
        data = json.loads(document)
        data["plugin_runtime"] = payload
        return json.dumps(data, indent=reporter.indent, ensure_ascii=True, allow_nan=False)
    if reporter.name == "sarif":
        data = json.loads(document)
        data["runs"][0].setdefault("properties", {})["pluginRuntime"] = payload
        return json.dumps(data, indent=reporter.indent, ensure_ascii=True, allow_nan=False)
    if reporter.name == "html":
        fragment = reporter._env.get_template("plugin_runtime_sections.html.j2").render(plugin_runtime=payload)
        if "</main>" not in document:
            raise ValueError("The HTML reporter did not provide its main content boundary")
        return document.replace("</main>", fragment + "\n</main>", 1)
    options = {
        "title": _TITLE,
        "notice": _NOTICE,
        "truncation_notice": _TRUNCATION_NOTICE,
        "status_label": "Evidence status",
        "header_fields": (("Run", payload["run_id"]), ("Execution status", payload["execution_status"])),
    }
    if reporter.name in {"markdown", "benchmark"}:
        fragment = _markdown(payload, publication=reporter.name == "benchmark", **options)
        return document.rstrip() + "\n\n" + fragment.rstrip() + "\n"
    if reporter.name == "cli":
        return document.rstrip() + "\n\n" + _cli(payload, **options)
    raise ValueError(f"Unsupported private runtime report format: {reporter.name}")


def render_plugin_runtime_reports(
    results: list[ValidationResult],
    *,
    report: RuntimePluginReport | None,
    reporters: Iterable[ReporterBase],
) -> dict[str, str]:
    """Compose static and runtime evidence without collecting or changing results.

    A saved report follows the same path as a live report. The private adapter
    does not add public dispatch or require an optional execution dependency.
    """
    selected = tuple(reporters)
    rendered = render_plugin_static_reports(results, reporters=selected)
    if report is None:
        return rendered
    payload = report.payload()
    return {reporter.name: _augment(reporter, rendered[reporter.name], payload) for reporter in selected}
