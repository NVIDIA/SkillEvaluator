# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 1 plugin runs that used to fail falsely, crash, stop early, or skip the plugin.

Rebuilt from the plugin-evaluation proof: 300 LOW notes overflowed the
100-finding cap into a blocking HIGH (proof M20, check-02 E09); a 5000-deep
YAML skill crashed the default run with no reports (M20, P16); QUALITY scored
only ``skills/``, even when the declared folders replace it (M20, E02); a
non-UTF-8 ``SKILL.md`` was called an unsafe path and stopped every other scan
(L7, E12); and a plugin folder without a manifest was validated as a skill, so
its subagent and command checks never ran (M17, skeptic ``zz-nomanifest``).
"""

from __future__ import annotations

import json
from pathlib import Path

from skillevaluator.cli_core import detect_content_type
from skillevaluator.constants import CONTENT_TYPE_PLUGIN, CONTENT_TYPE_SKILL
from skillevaluator.models.result import Severity, ValidationResult
from skillevaluator.tier1.commands import run_validation
from skillevaluator.validators.frontmatter_parser import parse_frontmatter
from skillevaluator.validators.plugin_schema import PluginSchemaValidator

_SKILL = (
    "---\nname: {name}\ndescription: Formats changelog entries for this repository. Use when the user asks to "
    "tidy a changelog.\nmetadata:\n  author: Demo Author <demo@example.com>\n---\n# {name}\n\n"
    "## Instructions\n\n1. Read the changelog.\n2. Group entries under Added, Changed, and Fixed.\n\n"
    "## Examples\n\nInput: fixed crash. Output: a Fixed entry.\n"
)
_CODEX = {
    "name": "codex-replace",
    "version": "1.0.0",
    "description": "Accepted path replaces the default.",
    "author": {"name": "Example", "email": "example@example.com"},
    "interface": {
        "displayName": "Codex Replace",
        "shortDescription": "Replace",
        "longDescription": "Accepted path replaces the default.",
        "developerName": "Example",
        "category": "Productivity",
        "capabilities": ["Read"],
        "defaultPrompt": ["Hi"],
    },
}


def _write(root: Path, files: dict[str, str | bytes | dict]) -> Path:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return root


def _findings(results: list[ValidationResult] | ValidationResult, check_name: str) -> list:
    results = results if isinstance(results, list) else [results]
    return [finding for result in results for finding in result.findings if finding.check_name == check_name]


# --------------------------------------------------------------------------- #
# M20: the 100-finding cap keeps blocking findings and never blocks on its own #
# --------------------------------------------------------------------------- #
def _many_agents(tmp_path: Path, extra: dict | None = None) -> Path:
    files: dict[str, str | bytes | dict] = {
        ".claude-plugin/plugin.json": {"name": "many-agents", "version": "1.0.0", **(extra or {})},
    }
    for index in range(300):
        files[f"agents/agent-{index:03d}.md"] = (
            f"---\nname: agent-{index:03d}\ndescription: Agent number {index:03d}.\n---\n\nDo task {index:03d}.\n"
        )
    return _write(tmp_path / "p", files)


def test_low_notes_over_the_cap_do_not_fail_tier1(tmp_path: Path) -> None:
    """Check-02 E09: 300 LOW notes gave a HIGH 'schema_errors_truncated' and the LOW scan note was cut off."""
    result = PluginSchemaValidator().validate(_many_agents(tmp_path))

    assert not _findings(result, "schema_errors_truncated")
    [note] = _findings(result, "schema_findings_truncated")
    assert note.severity == Severity.LOW
    assert _findings(result, "plugin_component_scan_truncated")
    assert result.passed


def test_a_blocking_finding_after_many_notes_is_still_reported(tmp_path: Path) -> None:
    result = PluginSchemaValidator().validate(_many_agents(tmp_path, {"commands": "./missing-commands/deploy.md"}))

    [missing] = _findings(result, "plugin_component_path_missing")
    assert missing.severity == Severity.HIGH
    assert not _findings(result, "schema_errors_truncated")
    assert not result.passed


# --------------------------------------------------------------------------- #
# M20: deeply nested YAML no longer crashes the default run                    #
# --------------------------------------------------------------------------- #
def _deep_skill(tmp_path: Path) -> Path:
    depth = 5000
    deep = "---\nname: deep\ndescription: Deep nesting.\nmetadata: " + "[" * depth + "]" * depth + "\n---\n# deep\n"
    return _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {"name": "deep-yaml", "version": "1.0.0"},
            "skills/greet/SKILL.md": _SKILL.format(name="greet"),
            "skills/deep/SKILL.md": deep,
        },
    )


def test_deep_yaml_skill_is_reported_not_raised(tmp_path: Path) -> None:
    """Check-02 P16: RecursionError in the version validator; the CLI wrote no reports."""
    root = _deep_skill(tmp_path)

    results = run_validation(root, checks="schema,version,pii,quality", content_type=CONTENT_TYPE_PLUGIN)
    names = [result.validator_name for result in results]
    assert len(results) == 4, names
    # The skill's schema check reports the YAML error itself (it used to raise, caught as in_plugin_skill_error).
    [error] = _findings(results, "yaml_syntax")
    assert error.severity == Severity.HIGH
    assert "skills/deep" in error.file_path.replace("\\", "/")


def test_standalone_deep_yaml_skill_is_reported_not_raised(tmp_path: Path) -> None:
    """P16's skill on its own: the schema validator raised RecursionError on both the base and the fix."""
    skill = _deep_skill(tmp_path) / "skills" / "deep"

    results = run_validation(skill, checks="schema,version,pii,quality", content_type=CONTENT_TYPE_SKILL)
    assert len(results) == 4, [result.validator_name for result in results]
    [error] = _findings(results, "yaml_syntax")
    assert error.severity == Severity.HIGH
    assert "nested too deeply" in error.message


def test_frontmatter_parser_turns_deep_nesting_into_an_error(tmp_path: Path) -> None:
    manifest = _deep_skill(tmp_path) / "skills" / "deep" / "SKILL.md"

    parsed, result = parse_frontmatter(manifest)
    assert parsed is None
    assert any("nested too deeply" in error for error in result.errors)


# --------------------------------------------------------------------------- #
# M20: QUALITY scores the skills the plugin's client loads                     #
# --------------------------------------------------------------------------- #
def _quality_files(results: list[ValidationResult]) -> set[str]:
    [quality] = results  # the run asked for the quality check only
    paths = {finding.file_path.replace("\\", "/") for finding in quality.findings}
    paths |= {detail.check_name for detail in quality.success_details}
    return paths


def test_quality_scores_declared_skills_that_replace_skills_folder(tmp_path: Path) -> None:
    """Check-02 E02: Codex loads only my-skills/beta; QUALITY scored skills/Bad_Name and failed Tier 1 on it."""
    root = _write(
        tmp_path / "p",
        {
            ".codex-plugin/plugin.json": {**_CODEX, "skills": "./my-skills/"},
            "my-skills/beta/SKILL.md": _SKILL.format(name="beta"),
            "skills/Bad_Name/SKILL.md": "---\nname: Bad_Name\ndescription: x\n---\nx\n",
        },
    )

    results = run_validation(root, checks="quality", content_type=CONTENT_TYPE_PLUGIN)
    seen = " ".join(sorted(_quality_files(results)))
    assert "beta" in seen
    assert "Bad_Name" not in seen


def test_quality_runs_for_a_plugin_with_only_declared_skill_folders(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "p",
        {
            ".cursor-plugin/plugin.json": {"name": "cursor-demo", "skills": "my-skills/"},
            "my-skills/summarize/SKILL.md": _SKILL.format(name="summarize"),
        },
    )

    results = run_validation(root, checks="quality", content_type=CONTENT_TYPE_PLUGIN)
    assert "summarize" in " ".join(sorted(_quality_files(results)))


# --------------------------------------------------------------------------- #
# L7: a non-UTF-8 SKILL.md is its own finding and the other scans still run    #
# --------------------------------------------------------------------------- #
def test_non_utf8_skill_is_reported_and_other_checks_still_run(tmp_path: Path) -> None:
    """Check-02 E12: 'changed or became unsafe after discovery', security_failure, and only one validator ran."""
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {"name": "latin", "version": "1.0.0"},
            "skills/greet/SKILL.md": _SKILL.format(name="greet"),
            "skills/latin/SKILL.md": b"---\nname: latin\ndescription: Caf\xe9 helper.\n---\n\n# latin\n",
        },
    )

    results = run_validation(root, checks="schema,quality", content_type=CONTENT_TYPE_PLUGIN)
    assert len(results) == 2
    [schema] = results[:1]
    assert not schema.metadata.get("security_failure")
    [encoding] = _findings(schema, "bundled_skill_not_utf8")
    assert encoding.severity == Severity.HIGH
    assert not _findings(schema, "bundled_skill_path_unsafe")


# --------------------------------------------------------------------------- #
# M17: a plugin folder without a manifest is validated as a plugin             #
# --------------------------------------------------------------------------- #
def _no_manifest(tmp_path: Path) -> Path:
    """Skeptic zz-nomanifest: Claude Code --plugin-dir loads it; it was detected as a skill."""
    return _write(
        tmp_path / "plugin",
        {
            "agents/bypass.md": "---\nname: bypass\ndescription: Runs fixes.\ntools: Read\npermissionMode: bypassPermissions\n---\nx\n",
            "commands/any.md": "---\ndescription: Anything.\nallowed-tools: Bash\n---\nx\n",
            "skills/notes/SKILL.md": _SKILL.format(name="notes"),
        },
    )


def test_folder_with_plugin_components_and_no_manifest_is_a_plugin(tmp_path: Path) -> None:
    root = _no_manifest(tmp_path)

    assert detect_content_type(root) == CONTENT_TYPE_PLUGIN
    result = PluginSchemaValidator().validate(root)
    [missing] = _findings(result, "manifest_missing")
    assert missing.severity == Severity.MEDIUM
    assert _findings(result, "plugin_agent_bypass_permissions")
    assert _findings(result, "plugin_command_unrestricted_bash")
    types = {row["type"] for row in result.metadata["plugin"]["component_inventory"]["components"]}
    assert {"agent", "command", "skill"} <= types


def test_skill_collections_and_code_folders_are_not_plugins(tmp_path: Path) -> None:
    skills_only = _write(tmp_path / "skills-repo", {"skills/notes/SKILL.md": _SKILL.format(name="notes")})
    code = _write(tmp_path / "code", {"agents/__init__.py": "", "skills/notes/SKILL.md": _SKILL.format(name="notes")})

    assert detect_content_type(skills_only) == CONTENT_TYPE_SKILL
    assert detect_content_type(code) == CONTENT_TYPE_SKILL
    empty = PluginSchemaValidator().validate(_write(tmp_path / "empty", {"README.md": "x\n"}))
    [missing] = _findings(empty, "manifest_missing")
    assert missing.severity == Severity.HIGH
