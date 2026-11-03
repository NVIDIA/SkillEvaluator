# SPDX-License-Identifier: Apache-2.0
"""Explicit private adapters; ordinary reporter entrypoints keep their behavior."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

from skillevaluator.reporting.plugin_static import StaticPluginReport, project_plugin_static

if TYPE_CHECKING:
    from collections.abc import Iterable

    from skillevaluator.models.result import ValidationResult
    from skillevaluator.reporting.base import ReporterBase


def _display(value: object) -> str:
    return "unknown" if value is None else str(value)


def _markdown(payload: dict, *, publication: bool = False) -> str:
    if publication:
        from skillevaluator.reporting.benchmark import _md_cell as escape
    else:
        from skillevaluator.reporting.markdown import _markdown_table_cell as escape

    def cell(value: object) -> str:
        encoded = escape(_display(value))
        if publication:
            return encoded
        for character in "[]!*_\\":
            encoded = encoded.replace(character, f"&#{ord(character)};")
        return encoded

    lines = [
        "## Plugin static evidence",
        "",
        f"**{cell(payload['name'])}** — {payload['status']}",
        "",
        "Static evidence does not demonstrate runtime loading or behavior.",
        "",
    ]
    if payload["truncated"]:
        lines.extend(["Some static evidence was omitted by report bounds; see section counts.", ""])
    for section in payload["sections"]:
        lines.extend([f"### {section['title']}", ""])
        lines.extend(f"{cell(note)}\n" for note in section["notes"] if note is not None)
        if section["rows"]:
            lines.extend(
                [
                    "| " + " | ".join(section["columns"]) + " |",
                    "| " + " | ".join("---" for _ in section["columns"]) + " |",
                ]
            )
            lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in section["rows"])
        if section["omitted"]:
            lines.append(f"{section['omitted']} more rows omitted ({section['total']} total).")
        lines.append("")
    return "\n".join(lines)


def _cli(payload: dict) -> str:
    from rich.console import Console
    from rich.text import Text

    output = io.StringIO()
    console = Console(file=output, color_system=None, emoji=False)
    lines = [
        "Plugin static evidence",
        f"{payload['name']} — {payload['status']}",
        "Static evidence does not demonstrate runtime loading or behavior.",
    ]
    if payload["truncated"]:
        lines.append("Static evidence was omitted by report bounds; see section counts.")
    for section in payload["sections"]:
        lines.extend(["", section["title"], " | ".join(section["columns"])])
        lines.extend(_display(note) for note in section["notes"] if note is not None)
        lines.extend(" | ".join(_display(cell) for cell in row) for row in section["rows"])
        if section["omitted"]:
            lines.append(f"{section['omitted']} more rows omitted ({section['total']} total).")
    for line in lines:
        console.print(Text(line), soft_wrap=True)
    return output.getvalue()


def _augment(
    reporter: ReporterBase,
    document: str,
    results: list[ValidationResult],
    projection: StaticPluginReport,
    payload: dict,
) -> str:
    if reporter.name == "json":
        data = json.loads(document)
        data["plugin_static"] = payload
        return json.dumps(data, indent=reporter.indent, ensure_ascii=True, allow_nan=False)
    if reporter.name == "sarif":
        data = json.loads(document)
        run = data["runs"][0]
        run.setdefault("properties", {})["pluginStatic"] = payload
        findings = [finding for result in results for finding in result.findings]
        for row, finding in zip(run["results"], findings, strict=True):
            component = projection.component_for(finding.file_path, finding.metadata)
            if component is not None:
                row.setdefault("properties", {})["pluginComponent"] = component
        return json.dumps(data, indent=reporter.indent, ensure_ascii=True, allow_nan=False)
    if reporter.name == "html":
        fragment = reporter._env.get_template("plugin_static_sections.html.j2").render(plugin_static=payload)
        if "</main>" not in document:
            raise ValueError("The HTML reporter did not provide its main content boundary")
        return document.replace("</main>", fragment + "\n</main>", 1)
    if reporter.name in {"markdown", "benchmark"}:
        return document.rstrip() + "\n\n" + _markdown(payload, publication=reporter.name == "benchmark").rstrip() + "\n"
    if reporter.name == "cli":
        return document.rstrip() + "\n\n" + _cli(payload)
    raise ValueError(f"Unsupported private static report format: {reporter.name}")


def render_plugin_static_reports(
    results: list[ValidationResult], *, reporters: Iterable[ReporterBase]
) -> dict[str, str]:
    """Render a private static bundle through existing reporters with one reduction.

    This function is deliberately absent from CLI dispatch and the public
    reporting package exports. No reporter instance or validation result is
    mutated, and a bundle without plugin evidence uses existing output exactly.
    """
    projection = project_plugin_static(results)
    payload = projection.payload() if projection is not None else None
    rendered = {}
    for reporter in reporters:
        document = reporter.render_all(results)
        rendered[reporter.name] = (
            _augment(reporter, document, results, projection, payload) if projection is not None else document
        )
    return rendered
