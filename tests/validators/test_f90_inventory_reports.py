# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The component inventory shows broken components and counts their findings.

Rebuilt from the plugin-evaluation proof (M40): a declared path that does not
exist was listed as "Evaluated" with nothing saying it is missing (check-02
P01), broken MCP config files and invalid server names showed 0 findings
(check-04 pos-06), an agent declared from ``evals/`` lost its finding because
attribution compared the raw ref with the frontmatter name (check-02 P07),
missing bundle refs looked evaluated (check-03 pos-04), and a plugin reached
through a symlinked parent folder lost the findings of scanners that report
resolved paths (L15, check-10 e08).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.plugin_components import Component, attribute_findings
from skillevaluator.reporting.html import HTMLReporter
from skillevaluator.reporting.markdown import MarkdownReporter
from skillevaluator.reporting.plugin_sections import inventory_view
from skillevaluator.validators.plugin_schema import PluginSchemaValidator


def _write(root: Path, files: dict[str, str | dict]) -> Path:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _rows(result: ValidationResult, component_type: str | None = None) -> list[dict]:
    rows = result.metadata["plugin"]["component_inventory"]["components"]
    return [row for row in rows if component_type is None or row["type"] == component_type]


def _missing_each_type(tmp_path: Path) -> ValidationResult:
    """Proof check-02 P01: one missing path per declared field."""
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {
                "name": "missing-each",
                "version": "1.0.0",
                "skills": "./missing-skills/",
                "agents": "./missing-agents/reviewer.md",
                "commands": "./missing-commands/deploy.md",
                "mcpServers": "./missing.mcp.json",
                "outputStyles": "./missing-styles/",
            },
            "skills/greet/SKILL.md": "---\nname: greet\ndescription: Greets.\n---\nSay hi.\n",
        },
    )
    return PluginSchemaValidator().validate(root)


def test_inventory_rows_keep_the_problem_of_broken_components(tmp_path: Path) -> None:
    result = _missing_each_type(tmp_path)

    broken = {row["path"]: row for row in _rows(result) if "problem" in row}
    assert set(broken) == {
        "missing-skills",
        "missing-agents/reviewer.md",
        "missing-commands/deploy.md",
        "missing.mcp.json",
        "missing-styles",
    }
    assert {row["problem"] for row in broken.values()} == {"missing"}
    assert all(row["findings"] == 1 for row in broken.values())
    assert result.metadata["plugin"]["component_inventory"]["broken"] == 5
    [greet] = _rows(result, "skill")[1:]
    assert "problem" not in greet


def test_reports_never_label_a_broken_component_evaluated(tmp_path: Path) -> None:
    result = _missing_each_type(tmp_path)

    view = inventory_view(result.metadata["plugin"]["component_inventory"])
    assert view is not None
    assert view["broken"] == 5
    labels = {row["path"]: row["support_label"] for row in view["rows"]}
    assert labels["missing-skills"] == "Broken: missing"
    assert labels["skills/greet"] == "Evaluated"

    result.validator_name = "Plugin Schema & Bundle References"
    markdown = MarkdownReporter().render_all([result])
    assert "5 broken component(s)" in markdown
    assert "| skill | ./missing-skills/ | declared | Broken: missing | 1 |" in markdown

    html = HTMLReporter().render_all([result])
    assert 'id="plugin-broken-components"' in html
    assert '<span class="plugin-pill fail">Broken: missing</span>' in html


def test_broken_mcp_files_and_invalid_server_names_are_counted(tmp_path: Path) -> None:
    """Check-04 pos-06: './bad.json', './servers.txt', and 'bad name!' showed 0 findings."""
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {
                "name": "mcp-errors",
                "version": "1.0.0",
                "mcpServers": ["./bad.json", "./servers.txt", {"bad name!": {"command": "node", "args": ["x.js"]}}],
            },
            "bad.json": "{not json",
            "servers.txt": "{}",
        },
    )

    rows = {row["name"]: row for row in _rows(PluginSchemaValidator().validate(root), "mcp")}
    assert rows["./bad.json"]["problem"] == "invalid"
    assert rows["./bad.json"]["findings"] == 1
    assert rows["./servers.txt"]["findings"] == 1
    assert rows["bad name!"]["findings"] >= 1


def test_declared_agent_is_attributed_by_its_normalized_path(tmp_path: Path) -> None:
    """Check-02 P07: the unscanned-folder finding names './evals/agents/judge.md'; the row is named 'judge'."""
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {"name": "p07", "version": "1.0.0", "agents": ["./evals/agents/judge.md"]},
            "evals/agents/judge.md": "---\nname: judge\ndescription: Judges.\ntools: Read\n---\nJudge.\n",
        },
    )

    result = PluginSchemaValidator().validate(root)
    [row] = _rows(result, "agent")
    assert row["name"] == "judge"
    assert row["findings"] == 1


_POS04_SKILL = (
    "---\nname: {name}\ndescription: Formats release notes into a short changelog. Use when the user asks to "
    "summarize merged changes for a release.\nmetadata:\n  author: Demo Author <dev@example.com>\n---\n# {name}\n\n"
    "## Instructions\n1. Read the merged change titles.\n2. Write one line per change.\n\n"
    "## Examples\nInput: three merged fixes. Output: a Fixes list with three lines.\n"
)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")
def test_dependency_states_show_in_the_inventory(tmp_path: Path) -> None:
    """Check-03 pos-04: one skill ref in each state; the missing ref looked evaluated with 0 findings."""
    repo = tmp_path / "agent-catalog"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", "https://github.com/acme/agent-catalog.git"], check=True
    )
    refs = {
        "provided": "github::acme/agent-catalog::plugins::demo/skills/local-skill",
        "referenced": "github::acme/agent-catalog::skills::shared-skill",
        "missing": "github::acme/agent-catalog::skills::ghost-skill",
        "external": "github::other-org/other-repo::skills::shared-skill",
        "unresolved": "github::acme/agent-catalog::docs::guide",
    }
    manifest = (
        'name: demo-plugin\ndescription: One ref in each of the five states.\nversion: "1.0.0"\n'
        "author:\n  email: dev@example.com\nskills:\n  refs:\n" + "".join(f"    - {ref}\n" for ref in refs.values())
    )
    _write(
        repo,
        {
            "plugins/demo/agent_plugin.yaml": manifest,
            "plugins/demo/skills/local-skill/SKILL.md": _POS04_SKILL.format(name="local-skill"),
            "skills/shared-skill/SKILL.md": _POS04_SKILL.format(name="shared-skill"),
        },
    )

    result = PluginSchemaValidator().validate(repo / "plugins" / "demo")
    inventory = result.metadata["plugin"]["component_inventory"]
    rows = {row["name"]: row for row in inventory["components"]}
    # The packaged skill that provides the first ref is the same component, so five rows, not six.
    assert len(rows) == 5, sorted(rows)
    assert refs["provided"] not in rows
    assert rows["local-skill"]["origin"] == "declared+packaged"
    assert "problem" not in rows[refs["referenced"]] and "dependency" not in rows[refs["referenced"]]
    assert (rows[refs["missing"]]["problem"], rows[refs["missing"]]["findings"]) == ("missing", 1)
    assert rows[refs["external"]]["dependency"] == "external"
    assert rows[refs["unresolved"]]["dependency"] == "unresolved"
    assert [finding.check_name for finding in result.findings if finding.severity == Severity.HIGH] == [
        "plugin_dependency_missing"
    ]

    labels = {row["name"]: row["support_label"] for row in inventory_view(inventory)["rows"]}
    assert labels[refs["missing"]] == "Broken: missing"
    assert labels[refs["external"]] == "External ref (not evaluated)"
    assert labels[refs["unresolved"]] == "Unresolved ref (not evaluated)"


def test_findings_reported_through_the_resolved_root_keep_their_component(tmp_path: Path) -> None:
    """Check-10 e08: Semgrep and Bandit report resolved paths, which are not under a symlinked lexical root."""
    real = _write(tmp_path / "real" / "plugin", {"skills/helper/scripts/run.py": "print(1)\n"})
    (tmp_path / "via-link").symlink_to(tmp_path / "real")
    lexical_root = tmp_path / "via-link" / "plugin"
    components = [Component("skill", "helper", "packaged", "skills/helper", "evaluated")]
    finding = Finding(
        category="CODE_RISK",
        severity=Severity.MEDIUM,
        check_name="subprocess_shell_true",
        message="shell=True",
        file_path=str(real.resolve() / "skills/helper/scripts/run.py"),
    )

    attribute_findings(components, [finding], lexical_root)
    assert components[0].findings == 1
