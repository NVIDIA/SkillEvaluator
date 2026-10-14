# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BENCHMARK.md path redaction keeps relative paths and closing tags intact (proof M2).

The plugin is a small copy of the proof's check 2 examples: ``P01`` (every
component field points at a missing ``./...`` path), ``P05`` (the "does not
start with './'" message) and ``E11`` (a declared path holding ``</script>``).
The real Tier 1 schema check and the real Tier 3 package step feed the card,
so both the findings and the Component Coverage table are covered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter
from skillevaluator.reporting.benchmark import _publication_safe_inline
from skillevaluator.tier1.commands import run_validation
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

HOSTILE_AGENT = "./agents/<script>alert('x')</script>.md"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _card(tmp_path: Path) -> str:
    plugin = tmp_path / "missing-each-type"
    manifest = {
        "name": "missing-each-type",
        "version": "1.0.0",
        "description": "Every component field points at a path that does not exist.",
        "skills": "./missing-skills/",
        "agents": ["./missing-agents/reviewer.md", HOSTILE_AGENT],
        "commands": "commands/hello.md",
    }
    _write(plugin / ".claude-plugin" / "plugin.json", json.dumps(manifest))
    _write(plugin / "commands" / "hello.md", "---\ndescription: Says hello.\n---\nHello.\n")
    _write(plugin / "skills" / "greet" / "SKILL.md", "---\nname: greet\ndescription: Greets.\n---\nHi.\n")
    _write(plugin / "evals" / "evals.json", json.dumps([{"id": "c1", "prompt": "hi", "expected_output": "hi"}]))
    results = run_validation(plugin, checks="schema", content_type="plugin")
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")
    tier3 = ValidationResult(validator_name="AGENT_EVAL", validator_description="Tier 3 agent evaluation")
    tier3.metadata["agent_eval"] = {"skill_name": "missing-each-type", "plugin_provenance": package.provenance()}
    return BenchmarkReporter(include_timestamp=False, content_type="plugin", skill_name="missing-each-type").render_all(
        [*results, tier3]
    )


def test_benchmark_keeps_relative_declared_paths_in_findings(tmp_path: Path) -> None:
    card = _card(tmp_path)

    assert "'agents' path './missing-agents/reviewer.md' does not exist in the plugin" in card
    assert "'skills' path './missing-skills/' does not exist in the plugin" in card
    assert "'.reviewer.md'" not in card and "'.missing-skills'" not in card


def test_benchmark_keeps_closing_tags_escaped_not_rewritten(tmp_path: Path) -> None:
    card = _card(tmp_path)

    assert "\"./agents/&lt;script&gt;alert('x')&lt;/script&gt;.md\" does not exist" in card
    assert ".script&gt;.md" not in card
    assert "<script>" not in card


def test_benchmark_coverage_table_keeps_relative_component_names(tmp_path: Path) -> None:
    card = _card(tmp_path)

    assert "| ./missing-agents/reviewer.md | agent | Invalid | declared path does not exist |" in card
    assert "| ./missing-skills/ | skill | Invalid | declared path does not exist |" in card
    assert "| .reviewer.md |" not in card


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # P05: the path-style message quotes "./".
        (
            "'skills' path 'extra-skills/' does not start with './'",
            "'skills' path 'extra-skills/' does not start with './'",
        ),
        ("see ../shared/notes.md", "see ../shared/notes.md"),
        # Absolute paths are still reduced to their basename, also after an ellipsis or inside a tag.
        ("wrote /home/alice/plugin/hooks/run.sh", "wrote run.sh"),
        ("truncated .../home/alice/secret.txt", "truncated ...secret.txt"),
        ("value </home/alice/x>", "value &lt;x&gt;"),
        ("C:\\Users\\bob\\plugin\\x.md", "x.md"),
    ],
)
def test_publication_redaction_only_reduces_absolute_paths(raw: str, expected: str) -> None:
    assert _publication_safe_inline(raw) == expected
