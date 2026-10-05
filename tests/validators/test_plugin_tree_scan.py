# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-plugin Tier 1 scanners cover the entire plugin tree exactly once.

Regression: for a plugin that bundles skills under ``skills/``, the security,
PII, code-risk, secrets, hygiene, Unicode, license, and dependency scanners
only visited the bundled skill directories, so root-owned plugin content
(``scripts/``, ``hooks/``, ``.mcp.json``, ...) was never scanned.

External scanners are replaced by fakes that inspect the directory they are
actually handed, so the tests exercise the real exclusion and staging logic.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import pytest

from skillevaluator.constants import CONTENT_TYPE_PLUGIN, CONTENT_TYPE_SKILL
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.tier1.commands import run_validation
from skillevaluator.utils.tool_runner import GITLEAKS_FINDINGS_EXIT_CODE, ToolResult, Tools
from skillevaluator.validators.base import iter_scannable_files
from skillevaluator.validators.plugin_tree import (
    plugin_tree_exclusions,
    plugin_tree_scan_view,
    plugin_tree_scope,
    rebase_relative_finding_paths,
)

# Split so this test file does not itself look like it ships a credential.
AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
EVIL_PY = f'import os\nAWS_ACCESS_KEY_ID = "{AWS_KEY}"\ndef run(cmd):\n    eval(cmd)\n'
WHOLE_TREE_CHECKS = "security,pii,code-integrity"


def _skill(root: Path, name: str) -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Formats CSV files into Markdown tables on request.\n---\n# {name}\n",
        encoding="utf-8",
    )
    return skill


def _plugin(tmp_path: Path, *, root_script: str | None = EVIL_PY) -> Path:
    plugin = tmp_path / "probe-plugin"
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "probe-plugin"}), encoding="utf-8")
    _skill(plugin / "skills", "foo")
    if root_script is not None:
        (plugin / "scripts").mkdir()
        (plugin / "scripts" / "evil.py").write_text(root_script, encoding="utf-8")
    return plugin


def _hits(root: Path, needle: str) -> list[tuple[Path, int]]:
    """Return ``(file, line)`` for every line of every file below *root* containing *needle*."""
    hits: list[tuple[Path, int]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            path = Path(dirpath, name)
            text = path.read_text(encoding="utf-8", errors="ignore")
            hits.extend((path, number) for number, line in enumerate(text.splitlines(), 1) if needle in line)
    return hits


@dataclass
class FakeScanners:
    """Scanner fakes that report what is present in the directory they scan."""

    scanned: dict[str, list[Path]] = field(default_factory=dict)

    def _record(self, tool: str, root: Path) -> None:
        self.scanned.setdefault(tool, []).append(root)

    def skillspector(self, args: list[str], **_kwargs) -> ToolResult:
        root = Path(args[args.index("scan") + 1])
        self._record("skillspector", root)
        issues = [
            {
                "id": "SS001",
                "category": "execution",
                "pattern": "eval-usage",
                "severity": "HIGH",
                "confidence": 1.0,
                "location": {"file": path.relative_to(root).as_posix(), "start_line": line},
                "finding": "dangerous call",
            }
            for path, line in _hits(root, "eval(")
        ]
        risk = (
            {"score": 80, "severity": "HIGH", "recommendation": "DO_NOT_INSTALL"}
            if issues
            else {"score": 0, "severity": "LOW", "recommendation": "SAFE"}
        )
        report = {
            "skill": {"name": root.name, "source": str(root)},
            "risk_assessment": risk,
            "issues": issues,
            "metadata": {"skillspector_version": "1.0.0", "llm_requested": False, "llm_available": False},
        }
        return ToolResult(success=not issues, stdout=json.dumps(report), stderr="", exit_code=1 if issues else 0)

    def bandit(self, args: list[str], **_kwargs) -> ToolResult:
        root = Path(args[args.index("-r") + 1])
        self._record("bandit", root)
        issues = [
            {
                "filename": str(path),
                "line_number": line,
                "issue_severity": "MEDIUM",
                "issue_confidence": "HIGH",
                "test_id": "B307",
                "test_name": "blacklist",
                "issue_text": "Use of possibly insecure function",
                "issue_cwe": {"id": 78},
            }
            for path, line in _hits(root, "eval(")
            if path.suffix == ".py"
        ]
        output = json.dumps({"results": issues, "errors": []})
        return ToolResult(success=True, stdout=output, stderr="", exit_code=1 if issues else 0)

    def semgrep(self, args: list[str], **_kwargs) -> ToolResult:
        root = Path(args[-1])
        self._record("semgrep", root)
        results = [
            {
                "check_id": "skillevaluator.python.dynamic-code-execution",
                "path": str(path),
                "start": {"line": line},
                "extra": {"severity": "ERROR", "message": "Avoid evaluating dynamic code.", "metadata": {}},
            }
            for path, line in _hits(root, "eval(")
        ]
        output = json.dumps({"results": results, "errors": []})
        return ToolResult(success=True, stdout=output, stderr="", exit_code=1 if results else 0)

    def gitleaks(self, args: list[str], *, cwd: Path | None = None, **_kwargs) -> ToolResult:
        # Working-tree scans run from the scan root so findings stay relative to it.
        assert cwd is not None
        assert "--no-git" in args
        root = Path(cwd)
        self._record("gitleaks", root)
        findings = [
            {
                "RuleID": "aws-access-token",
                "Description": "AWS Access Key",
                "File": path.relative_to(root).as_posix(),
                "StartLine": line,
                "Tags": ["key", "AWS"],
            }
            for path, line in _hits(root, "AKIA")
        ]
        if not findings:
            return ToolResult(success=True, stdout="[]", stderr="", exit_code=0)
        return ToolResult(success=True, stdout=json.dumps(findings), stderr="", exit_code=GITLEAKS_FINDINGS_EXIT_CODE)

    def calls(self, tool: str) -> list[Path]:
        return self.scanned.get(tool, [])


@pytest.fixture
def scanners() -> Iterator[FakeScanners]:
    fakes = FakeScanners()
    with (
        patch.object(Tools.skillspector, "_path", "/usr/bin/skillspector"),
        patch.object(Tools.skillspector, "run", side_effect=fakes.skillspector),
        patch.object(Tools.bandit, "_path", "/usr/bin/bandit"),
        patch.object(Tools.bandit, "run", side_effect=fakes.bandit),
        patch.object(Tools.semgrep, "_path", "/usr/bin/semgrep"),
        patch.object(Tools.semgrep, "run", side_effect=fakes.semgrep),
        patch.object(Tools.gitleaks, "_path", "/usr/bin/gitleaks"),
        patch.object(Tools.gitleaks, "run", side_effect=fakes.gitleaks),
    ):
        yield fakes


def _by_name(results: list[ValidationResult]) -> dict[str, ValidationResult]:
    return {result.validator_name: result for result in results}


def _paths(result: ValidationResult) -> list[str]:
    return [finding.file_path for finding in result.findings]


# =============================================================================
# ROOT CONTENT OF A PLUGIN THAT BUNDLES SKILLS
# =============================================================================


def test_root_script_is_scanned_when_the_plugin_bundles_skills(tmp_path: Path, scanners: FakeScanners) -> None:
    plugin = _plugin(tmp_path)
    resolved_script = str(plugin.resolve() / "scripts" / "evil.py")

    results = _by_name(run_validation(plugin, checks=WHOLE_TREE_CHECKS, content_type=CONTENT_TYPE_PLUGIN))

    assert _paths(results["Security Scan"]) == ["scripts/evil.py"]
    assert not results["Security Scan"].passed
    pii = results["PII Scan"]
    assert [(f.check_name, f.file_path) for f in pii.findings] == [("aws_identifiers", "scripts/evil.py")]
    assert pii.findings[0].severity == Severity.CRITICAL
    code_risk = results["Code Risk Analysis"]
    assert sorted((f.category, f.file_path) for f in code_risk.findings) == [
        ("BANDIT", resolved_script),
        ("SEMGREP", resolved_script),
    ]
    secrets = results["Secrets Detection"]
    assert [(f.check_name, f.file_path) for f in secrets.findings] == [("aws-access-token", "scripts/evil.py")]
    assert not secrets.passed and not secrets.is_incomplete


def test_root_pass_never_hands_a_bundled_skill_to_an_external_scanner(tmp_path: Path, scanners: FakeScanners) -> None:
    plugin = _plugin(tmp_path)
    skill = plugin / "skills" / "foo"
    (skill / "helper.py").write_text("print('skill helper')\n", encoding="utf-8")
    snapshots: dict[str, list[set[str]]] = {}

    def snapshot(tool: str, run):
        def wrapped(args, **kwargs):
            if tool == "gitleaks":
                root = Path(kwargs["cwd"])
            elif tool == "skillspector":
                root = Path(args[args.index("scan") + 1])
            elif tool == "bandit":
                root = Path(args[args.index("-r") + 1])
            else:
                root = Path(args[-1])
            files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
            snapshots.setdefault(tool, []).append(files)
            return run(args, **kwargs)

        return wrapped

    with (
        patch.object(Tools.skillspector, "run", side_effect=snapshot("skillspector", scanners.skillspector)),
        patch.object(Tools.bandit, "run", side_effect=snapshot("bandit", scanners.bandit)),
        patch.object(Tools.semgrep, "run", side_effect=snapshot("semgrep", scanners.semgrep)),
        patch.object(Tools.gitleaks, "run", side_effect=snapshot("gitleaks", scanners.gitleaks)),
    ):
        run_validation(plugin, checks=WHOLE_TREE_CHECKS, content_type=CONTENT_TYPE_PLUGIN)

    for tool in ("skillspector", "bandit", "semgrep", "gitleaks"):
        # One per-skill pass plus one plugin-root pass, each on disjoint files.
        skill_files, root_files = snapshots[tool]
        assert skill_files == {"SKILL.md", "helper.py"}, tool
        assert "scripts/evil.py" in root_files, tool
        assert not any(path.startswith("skills/") for path in root_files), tool
        # Staged views are removed once the scan finishes.
        assert all(path == skill.resolve() or not path.exists() for path in scanners.calls(tool)[1:]), tool


def test_python_walkers_prune_bundled_skills_only_for_the_plugin_root(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path)
    skills = [plugin / "skills" / "foo"]
    (plugin / "skills" / "README.md").write_text("Shared notes for bundled skills.\n", encoding="utf-8")

    with plugin_tree_scope(plugin, skills):
        root_files = {p.relative_to(plugin).as_posix() for p in iter_scannable_files(plugin, {".md", ".py", ".json"})}
        skill_files = {p.relative_to(plugin).as_posix() for p in iter_scannable_files(skills[0], {".md"})}
    unscoped = {p.relative_to(plugin).as_posix() for p in iter_scannable_files(plugin, {".md", ".py", ".json"})}

    # Loose files under skills/ that belong to no bundled skill stay root-owned.
    assert root_files == {".claude-plugin/plugin.json", "scripts/evil.py", "skills/README.md"}
    assert skill_files == {"skills/foo/SKILL.md"}
    assert unscoped == root_files | skill_files


def test_nested_bundled_skill_is_scanned_by_its_own_pass_only(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path)
    outer = plugin / "skills" / "foo"
    inner = _skill(outer, "inner")

    with plugin_tree_scope(plugin, [outer, inner]):
        assert plugin_tree_exclusions(plugin) == {("skills", "foo"), ("skills", "foo", "inner")}
        assert plugin_tree_exclusions(outer) == {("inner",)}
        assert plugin_tree_exclusions(inner) == frozenset()
        outer_files = {p.relative_to(outer).as_posix() for p in iter_scannable_files(outer, {".md"})}
    assert outer_files == {"SKILL.md"}
    assert plugin_tree_exclusions(plugin) == frozenset()


# =============================================================================
# BUNDLED-SKILL FINDINGS: EXACTLY ONCE, ATTRIBUTED TO THE SKILL
# =============================================================================


def test_secret_inside_a_bundled_skill_is_reported_once_for_that_skill(tmp_path: Path, scanners: FakeScanners) -> None:
    plugin = _plugin(tmp_path, root_script="print('clean root script')\n")
    (plugin / "skills" / "foo" / "config.py").write_text(f'KEY = "{AWS_KEY}"\n', encoding="utf-8")

    results = run_validation(plugin, checks=f"schema,{WHOLE_TREE_CHECKS}", content_type=CONTENT_TYPE_PLUGIN)
    by_name = _by_name(results)

    assert _paths(by_name["Secrets Detection"]) == ["[foo] skills/foo/config.py"]
    assert [(f.check_name, f.file_path) for f in by_name["PII Scan"].findings] == [
        ("aws_identifiers", "[foo] skills/foo/config.py")
    ]
    # Each scanner ran once on the skill and once on the root.
    assert len(scanners.calls("gitleaks")) == 2
    all_keys = [
        (r.validator_name, f.check_name, f.file_path, f.line_number, f.message) for r in results for f in r.findings
    ]
    assert len(all_keys) == len(set(all_keys))

    # The plugin component inventory attributes the rebased paths to the skill.
    plugin_meta = next(r.metadata["plugin"] for r in results if "plugin" in r.metadata)
    rows = {(row["type"], row["name"]): row for row in plugin_meta["component_inventory"]["components"]}
    secret_and_pii = 2
    assert rows[("skill", "foo")]["findings"] >= secret_and_pii


def test_root_findings_are_not_attributed_to_a_bundled_skill(tmp_path: Path, scanners: FakeScanners) -> None:
    plugin = _plugin(tmp_path)

    results = run_validation(plugin, checks=f"schema,{WHOLE_TREE_CHECKS}", content_type=CONTENT_TYPE_PLUGIN)

    plugin_meta = next(r.metadata["plugin"] for r in results if "plugin" in r.metadata)
    rows = {(row["type"], row["name"]): row for row in plugin_meta["component_inventory"]["components"]}
    scanner_findings = [
        f for r in results if r.validator_name != "Plugin Schema & Bundle References" for f in r.findings
    ]
    assert scanner_findings
    assert all("skills/" not in f.file_path for f in scanner_findings)
    skill_findings = [f for r in results for f in r.findings if "skills/foo/" in f.file_path]
    assert rows[("skill", "foo")]["findings"] == len(skill_findings)


def test_root_is_scanned_even_after_a_critical_skill_finding_stops_the_skill_loop(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path)
    (plugin / "skills" / "foo" / "config.py").write_text(f'KEY = "{AWS_KEY}"\n', encoding="utf-8")
    _skill(plugin / "skills", "zeta")
    (plugin / "skills" / "zeta" / "config.py").write_text(f'KEY = "{AWS_KEY}"\n', encoding="utf-8")

    results = _by_name(run_validation(plugin, checks="pii", content_type=CONTENT_TYPE_PLUGIN))

    # Without --continue-on-failure the folder walk still stops after the
    # first skill with a CRITICAL finding, but the plugin root is always scanned.
    assert _paths(results["PII Scan"]) == ["[foo] skills/foo/config.py", "scripts/evil.py"]


def test_rebase_leaves_absolute_and_placeholder_paths_alone() -> None:
    result = ValidationResult()
    for path in ("scripts/run.py", "/abs/skills/foo/run.py", "<plugin-skills>", "", "[bar] nested.py"):
        result.add_finding(Finding(category="X", severity=Severity.LOW, check_name="c", message="m", file_path=path))

    rebase_relative_finding_paths(result, PurePosixPath("skills/foo"))

    assert _paths(result) == [
        "skills/foo/scripts/run.py",
        "/abs/skills/foo/run.py",
        "<plugin-skills>",
        "",
        "[bar] nested.py",
    ]


def test_license_check_covers_bundled_skills_when_the_root_looks_like_an_asset(tmp_path: Path) -> None:
    # A root-level rules file makes the plugin root look like a single asset;
    # it must still be split into bundled skills plus root-owned content.
    plugin = _plugin(tmp_path, root_script=None)
    (plugin / "legacy.mdc").write_text("Prefer small functions.\n", encoding="utf-8")
    (plugin / "skills" / "foo" / "run.py").write_text(
        "# SPDX-License-Identifier: GPL-3.0-only\nprint('run')\n", encoding="utf-8"
    )

    results = _by_name(run_validation(plugin, checks="license", content_type=CONTENT_TYPE_PLUGIN))

    findings = results["License Compliance"].findings
    assert [(f.check_name, f.file_path) for f in findings] == [("blocked_license", "[foo] skills/foo/run.py")]


# =============================================================================
# SKILL (NON-PLUGIN) VALIDATION IS UNCHANGED
# =============================================================================


def test_standalone_skill_validation_is_unchanged(tmp_path: Path, scanners: FakeScanners) -> None:
    skill = _skill(tmp_path, "solo")
    (skill / "scripts").mkdir()
    (skill / "scripts" / "evil.py").write_text(EVIL_PY, encoding="utf-8")

    results = _by_name(run_validation(skill, checks=WHOLE_TREE_CHECKS, content_type=CONTENT_TYPE_SKILL))

    assert _paths(results["Security Scan"]) == ["scripts/evil.py"]
    assert [f.file_path for f in results["PII Scan"].findings] == ["scripts/evil.py"]
    assert _paths(results["Secrets Detection"]) == ["scripts/evil.py"]
    # Every scanner ran exactly once, in place, on the skill itself.
    for tool in ("skillspector", "bandit", "semgrep", "gitleaks"):
        assert scanners.calls(tool) == [skill.resolve()], tool


def test_skill_collection_keeps_skill_relative_paths(tmp_path: Path, scanners: FakeScanners) -> None:
    collection = tmp_path / "collection"
    skill = _skill(collection, "foo")
    (skill / "config.py").write_text(f'KEY = "{AWS_KEY}"\n', encoding="utf-8")

    results = _by_name(run_validation(collection, checks="pii,code-integrity", content_type=CONTENT_TYPE_SKILL))

    assert _paths(results["Secrets Detection"]) == ["[foo] config.py"]
    assert _paths(results["PII Scan"]) == ["[foo] config.py"]
    assert scanners.calls("gitleaks") == [skill.resolve()]


# =============================================================================
# INPUT HARDENING: LINKED OR SPECIAL ROOT CONTENT IS STILL REJECTED
# =============================================================================


def _link_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")


@pytest.mark.parametrize("relative", ["scripts/linked.py", "scripts/lib/deep/linked.py", "hooks"])
def test_symlinked_root_content_is_rejected_before_any_scan(
    tmp_path: Path, scanners: FakeScanners, relative: str
) -> None:
    plugin = _plugin(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text(EVIL_PY, encoding="utf-8")
    link = plugin / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    _link_or_skip(link, outside if relative.endswith(".py") else tmp_path)

    results = run_validation(plugin, checks=WHOLE_TREE_CHECKS, content_type=CONTENT_TYPE_PLUGIN)

    assert [r.validator_name for r in results] == ["Plugin Tree Security"]
    [failure] = results
    assert failure.metadata["security_failure"] is True
    assert not failure.passed
    [finding] = failure.findings
    assert finding.check_name == "unsafe_plugin_filesystem"
    assert finding.file_path == relative
    assert scanners.scanned == {}


def test_linked_root_content_fails_closed_alongside_the_schema_result(tmp_path: Path, scanners: FakeScanners) -> None:
    plugin = _plugin(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text(EVIL_PY, encoding="utf-8")
    (plugin / "scripts" / "lib" / "deep").mkdir(parents=True)
    _link_or_skip(plugin / "scripts" / "lib" / "deep" / "linked.py", outside)

    results = run_validation(plugin, checks=f"schema,{WHOLE_TREE_CHECKS}", content_type=CONTENT_TYPE_PLUGIN)

    assert [r.validator_name for r in results] == ["Plugin Schema & Bundle References", "Plugin Tree Security"]
    assert results[-1].metadata["security_failure"] is True
    assert scanners.scanned == {}


def test_unsafe_bundled_skill_run_still_recounts_component_findings(tmp_path: Path) -> None:
    """Regression: a run stopped by an unsafe entry under skills/ skipped the component finding recount.

    The schema check counts component findings before it validates declared skill folders, so only the
    recount every other run path makes counted the finding of the declared skill below.
    """
    plugin = _plugin(tmp_path, root_script=None)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "probe-plugin", "skills": "./my-skills/"}), encoding="utf-8"
    )
    (plugin / "my-skills" / "x").mkdir(parents=True)
    (plugin / "my-skills" / "x" / "SKILL.md").write_text("---\nname: x\n---\nNo description.\n", encoding="utf-8")
    (tmp_path / "outside").mkdir()
    _link_or_skip(plugin / "skills" / "foo" / "linked", tmp_path / "outside")

    [result] = run_validation(plugin, checks="schema", content_type=CONTENT_TYPE_PLUGIN)

    assert result.metadata["security_failure"] is True
    assert "bundled_skill_path_unsafe" in {finding.check_name for finding in result.findings}
    rows = result.metadata["plugin"]["component_inventory"]["components"]
    [declared] = [row for row in rows if row["path"] == "my-skills/x"]
    assert declared["findings"] == 1


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special files")
def test_special_root_file_is_rejected(tmp_path: Path, scanners: FakeScanners) -> None:
    plugin = _plugin(tmp_path)
    os.mkfifo(plugin / "scripts" / "pipe.py")

    results = run_validation(plugin, checks="pii", content_type=CONTENT_TYPE_PLUGIN)

    assert [r.validator_name for r in results] == ["Plugin Tree Security"]
    assert results[0].findings[0].file_path == "scripts/pipe.py"


def test_hard_linked_root_file_is_rejected(tmp_path: Path, scanners: FakeScanners) -> None:
    plugin = _plugin(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text(EVIL_PY, encoding="utf-8")
    try:
        os.link(outside, plugin / "scripts" / "hard.py")
    except (OSError, NotImplementedError):
        pytest.skip("hard links are unavailable")

    results = run_validation(plugin, checks="pii", content_type=CONTENT_TYPE_PLUGIN)

    assert [r.validator_name for r in results] == ["Plugin Tree Security"]
    assert results[0].findings[0].file_path == "scripts/hard.py"


def test_schema_only_plugin_run_does_not_walk_the_tree(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    (plugin / "scripts" / "lib" / "deep").mkdir(parents=True)
    _link_or_skip(plugin / "scripts" / "lib" / "deep" / "linked.py", outside)

    results = run_validation(plugin, checks="schema", content_type=CONTENT_TYPE_PLUGIN)

    assert [r.validator_name for r in results] == ["Plugin Schema & Bundle References"]


def test_scan_view_drops_bundled_skills_and_links(tmp_path: Path) -> None:
    plugin = _plugin(tmp_path)
    (plugin / "results").mkdir()
    (plugin / "results" / "old.py").write_text("x = 1\n", encoding="utf-8")
    try:
        (plugin / "scripts" / "linked.py").symlink_to(tmp_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")

    with plugin_tree_scope(plugin, [plugin / "skills" / "foo"]):
        with plugin_tree_scan_view(plugin, excluded_dir_names={"results"}) as view:
            assert view is not None
            staged = {p.relative_to(view).as_posix() for p in view.rglob("*") if not p.is_dir()}
        with plugin_tree_scan_view(plugin / "skills" / "foo") as skill_view:
            assert skill_view is None

    assert staged == {".claude-plugin/plugin.json", "scripts/evil.py"}
    assert not view.exists()
