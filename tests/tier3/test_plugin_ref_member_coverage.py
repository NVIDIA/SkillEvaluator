# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage rows of skills and rules declared by reference match the member names runtime evidence uses.

A bundle manifest's ``skills.refs`` or ``rules.refs`` entry is a coverage row
named by its ref (``gitlab::example-group/agent-catalog::skills::release-notes``),
but staging names the member after the ref's trailing name (``release-notes``),
and so do the load census and the activation labels. Regression: such a row was
never promoted to ``loaded`` or ``exercised``, and reports listed a skill that
loaded and ran as "staged but not observed in any plugin trial". Nothing runs
an agent: packages are prepared and evidence is folded in as the collector
would after a run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.plugin_components import summarize_coverage
from skillevaluator.reporting.plugin_sections import coverage_view
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package
from skillevaluator.tier3.plugin_native import apply_load_census
from skillevaluator.tier3.plugin_runtime import apply_runtime_coverage

ORIGIN = "https://gitlab.example.com/example-group/agent-catalog.git"
REPO = "example-group/agent-catalog"
RELEASE_NOTES = f"gitlab::{REPO}::skills::release-notes"
TICKET_TRIAGE = f"gitlab::{REPO}::skills::ticket-triage"
CHANGELOG_DRAFT = f"git::{REPO}::skills::changelog-draft"
VENDORED = "github::example-org/agent-catalog::skills::vendor-skill"
UNRESOLVED = "github::example-org/agent-catalog::skills::not-supplied"
STYLE_RULE = f"gitlab::{REPO}::rules::style.md"
INIT_EVIDENCE = "claude-code system/init event: demo-plugin:release-notes in skills"
EVALS = {"skill_name": "demo-plugin", "evals": [{"id": "c1", "prompt": "Draft notes.", "expected_output": "Notes."}]}

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is unavailable")


def _clone(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "remote", "add", "origin", ORIGIN], check=True)
    return root


def _skill(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {directory.name}\ndescription: Skill {directory.name}.\n---\n# {directory.name}\n",
        encoding="utf-8",
    )
    return directory


def _plugin(plugin: Path, *, skills: list[Any], rules: list[Any] | None = None) -> Path:
    plugin.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"name": "demo-plugin", "version": "1.0.0", "skills": {"refs": skills}}
    if rules is not None:
        manifest["rules"] = {"refs": rules}
    (plugin / "agent_plugin.yaml").write_text(json.dumps(manifest), encoding="utf-8")
    (plugin / "evals").mkdir(exist_ok=True)
    (plugin / "evals" / "evals.json").write_text(json.dumps(EVALS), encoding="utf-8")
    return plugin


def _ref_plugin(tmp_path: Path, **kwargs: Any) -> tuple[Any, dict[str, dict[str, Any]]]:
    """A bundle plugin whose skills and rule are all declared by reference, prepared as Tier 3 does."""
    repo = _clone(tmp_path / "repo")
    for name in ("release-notes", "ticket-triage", "changelog-draft"):
        _skill(repo / "skills" / name)
    (repo / "rules").mkdir()
    (repo / "rules" / "style.md").write_text("# Style\nUse short lines.\n", encoding="utf-8")
    skills = [
        RELEASE_NOTES,
        {"source": "gitlab", "repo": REPO, "path": "skills/ticket-triage"},
        CHANGELOG_DRAFT,
        VENDORED,
        UNRESOLVED,
    ]
    plugin = _plugin(repo / "plugins" / "demo", skills=skills, rules=[STYLE_RULE])
    vendored = _skill(tmp_path / "vendored" / "vendor-skill")
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", include_skills=(vendored,), **kwargs)
    return package, _rows(package.provenance()["component_coverage"])


def _rows(coverage: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in coverage["components"]}


def _census(key: str, *entries: tuple[str, str]) -> dict[str, Any]:
    detail = "reason" if key == "not_loaded" else "evidence"
    items = [{"type": kind, "name": name, detail: INIT_EVIDENCE} for kind, name in entries]
    return {"claude-code": {"mode": "native", key: items}}


def _plugin_load(*kinds: str) -> dict[str, Any]:
    return {"by_agent": {"claude-code": {"mode": "native", "components": dict.fromkeys(kinds, "native")}}}


def _engine(*, exercised: tuple[str, ...] = (), unavailable: tuple[str, ...] = ()) -> dict[str, Any]:
    activation = {
        "declared": [*exercised, *unavailable],
        "exercised": list(exercised),
        "unavailable": list(unavailable),
    }
    summary = {"activation_coverage": activation, "hook_census": {"hooks": [], "total_runs": 0}}
    return {"agents": {"claude-code": {"plugin_signals_summary": {"with_skill": summary}}}}


def _row(kind: str, name: str, state: str = "staged", **extra: Any) -> dict[str, Any]:
    return {"type": kind, "name": name, "origin": "declared", "path": None, "state": state, "reason": "base", **extra}


# --------------------------------------------------------------------------- #
# The staged member name is recorded on the row                                #
# --------------------------------------------------------------------------- #
@requires_git
def test_resolved_refs_record_the_member_name_staging_gives_them(tmp_path: Path) -> None:
    package, rows = _ref_plugin(tmp_path)

    members = {name: row.get("member") for name, row in rows.items()}
    assert members == {
        RELEASE_NOTES: "release-notes",  # canonical gitlab ref, same repository
        TICKET_TRIAGE: "ticket-triage",  # gitlab selector, labelled by its canonical form
        CHANGELOG_DRAFT: "changelog-draft",  # git ref, same repository
        VENDORED: "vendor-skill",  # external ref supplied by --include-skills
        UNRESOLVED: None,  # not staged, so there is no member to name
        STYLE_RULE: "style.md",  # a rule is staged under its file name
    }
    assert rows[UNRESOLVED]["state"] == "unavailable"
    staged_names = {path.name for path in package.include_skills} | set(package.staged_rules)
    assert {member for member in members.values() if member} <= staged_names
    for name in (RELEASE_NOTES, TICKET_TRIAGE, CHANGELOG_DRAFT, VENDORED, STYLE_RULE):
        assert (rows[name]["path"], rows[name]["state"]) == (None, "staged")


@requires_git
def test_a_member_name_two_staged_skills_share_is_not_recorded(tmp_path: Path) -> None:
    """A bundled skill and an --include-skills directory both named release-notes: evidence cannot tell them apart."""
    repo = _clone(tmp_path / "repo")
    plugin = _plugin(repo / "plugins" / "demo", skills=["github::example-org/agent-catalog::skills::release-notes"])
    _skill(plugin / "skills" / "release-notes")
    vendored = _skill(tmp_path / "vendored" / "release-notes")

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", include_skills=(vendored,))

    assert [path.name for path in package.include_skills] == ["release-notes", "release-notes"]
    rows = _rows(package.provenance()["component_coverage"])
    assert "member" not in rows["github::example-org/agent-catalog::skills::release-notes"]
    assert "member" not in rows["release-notes"]  # a packaged row is unchanged

    census = _census("loaded", ("skill", "release-notes"))
    promoted = _rows(apply_load_census(package.provenance()["component_coverage"], census, _plugin_load("skill")))
    assert promoted["github::example-org/agent-catalog::skills::release-notes"]["state"] == "staged"
    assert promoted["release-notes"]["state"] == "loaded"


# --------------------------------------------------------------------------- #
# Load census                                                                  #
# --------------------------------------------------------------------------- #
@requires_git
def test_the_load_census_loads_a_ref_declared_skill_by_its_member_name(tmp_path: Path) -> None:
    package, _rows_before = _ref_plugin(tmp_path, plugin_load="native", agents="claude-code", env_mode="docker")
    coverage = package.provenance()["component_coverage"]
    census = _census("loaded", ("skill", "release-notes"), ("skill", "vendor-skill"), ("skill", "other-skill"))

    rows = _rows(apply_load_census(coverage, census, _plugin_load("skill", "rule")))

    for name in (RELEASE_NOTES, VENDORED):
        assert rows[name]["state"] == "loaded"
        assert rows[name]["reason"] == (
            "skill reference resolved to a local member skill; loaded natively by claude-code (load census, "
            f"harness evidence: {INIT_EVIDENCE})"
        )
    # No census entry for these members, and "other-skill" names no row.
    assert {rows[name]["state"] for name in (TICKET_TRIAGE, CHANGELOG_DRAFT, STYLE_RULE)} == {"staged"}
    assert rows[UNRESOLVED]["state"] == "unavailable"


@requires_git
def test_the_load_census_matches_a_ref_declared_rule_by_its_file_name(tmp_path: Path) -> None:
    package, _rows_before = _ref_plugin(tmp_path, plugin_load="native", agents="claude-code", env_mode="docker")
    coverage = package.provenance()["component_coverage"]

    listed = _rows(apply_load_census(coverage, _census("listed", ("rule", "style.md")), _plugin_load("rule")))
    failed = _rows(apply_load_census(coverage, _census("not_loaded", ("rule", "style.md")), _plugin_load("rule")))

    assert listed[STYLE_RULE]["state"] == "staged"
    assert "the load census listed it" in listed[STYLE_RULE]["reason"]
    assert failed[STYLE_RULE]["state"] == "not_loaded"
    assert f"not loaded natively by claude-code: {INIT_EVIDENCE}" in failed[STYLE_RULE]["reason"]


@requires_git
def test_a_ref_to_a_bundled_skill_and_its_packaged_row_load_together(tmp_path: Path) -> None:
    """The ref and the packaged row name the one staged directory, so one census entry loads both."""
    plugin = _plugin(_clone(tmp_path / "plugin"), skills=[RELEASE_NOTES])
    _skill(plugin / "skills" / "release-notes")

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")
    coverage = package.provenance()["component_coverage"]

    assert [path.name for path in package.include_skills] == ["release-notes"]
    assert _rows(coverage)[RELEASE_NOTES]["member"] == "release-notes"
    rows = _rows(apply_load_census(coverage, _census("loaded", ("skill", "release-notes")), _plugin_load("skill")))
    assert {name: row["state"] for name, row in rows.items()} == {RELEASE_NOTES: "loaded", "release-notes": "loaded"}


def test_a_census_entry_for_another_name_or_type_does_not_load_a_ref_row() -> None:
    coverage = summarize_coverage(
        [
            _row("skill", RELEASE_NOTES, member="release-notes"),
            _row("skill", TICKET_TRIAGE),  # no recorded member (unresolvable or ambiguous)
        ]
    )
    census = _census("loaded", ("skill", "ticket-triage"), ("skill", "release"), ("rule", "release-notes"))

    rows = _rows(apply_load_census(coverage, census, _plugin_load("skill", "rule")))

    assert {name: row["state"] for name, row in rows.items()} == {RELEASE_NOTES: "staged", TICKET_TRIAGE: "staged"}


# --------------------------------------------------------------------------- #
# Activation coverage                                                          #
# --------------------------------------------------------------------------- #
@requires_git
def test_activation_exercises_a_ref_declared_skill_by_its_member_name(tmp_path: Path) -> None:
    package, _rows_before = _ref_plugin(tmp_path)
    provenance = package.provenance()

    promoted = apply_runtime_coverage(
        provenance,
        _engine(exercised=("skill:release-notes", "skill:vendor-skill"), unavailable=("skill:ticket-triage",)),
    )

    rows = _rows(provenance["component_coverage"])
    assert promoted == 2
    for name in (RELEASE_NOTES, VENDORED):
        assert rows[name]["state"] == "exercised"
        assert rows[name]["reason"] == (
            "skill reference resolved to a local member skill; runtime evidence: activated in the claude-code "
            "with-plugin arm"
        )
    # Every activation failed, or none happened: still staged, never exercised.
    assert rows[TICKET_TRIAGE]["state"] == rows[CHANGELOG_DRAFT]["state"] == "staged"
    assert rows[UNRESOLVED]["state"] == "unavailable"


def test_activation_promotion_of_packaged_rows_and_hooks_is_unchanged() -> None:
    rows = [
        {**_row("skill", "alpha"), "origin": "packaged", "path": "skills/alpha"},
        _row("skill", RELEASE_NOTES, member="release-notes"),
        _row("skill", TICKET_TRIAGE),  # no recorded member: its ref name never matches a label
        _row("hook", "hooks/hooks.json"),
    ]
    provenance = {"component_coverage": summarize_coverage(deepcopy(rows))}
    engine = _engine(exercised=("skill:alpha", "skill:ticket-triage", "skill:release", "hook:hooks/hooks.json"))

    assert apply_runtime_coverage(provenance, engine) == 1

    states = {row["name"]: row["state"] for row in provenance["component_coverage"]["components"]}
    assert states == {
        "alpha": "exercised",
        RELEASE_NOTES: "staged",
        TICKET_TRIAGE: "staged",
        "hooks/hooks.json": "staged",
    }


@requires_git
def test_reports_count_an_exercised_ref_declared_skill_as_observed(tmp_path: Path) -> None:
    """The live false negative: a skill that loaded and ran was listed as staged but not observed."""
    package, _rows_before = _ref_plugin(tmp_path)
    provenance = package.provenance()
    labels = ["skill:release-notes", "skill:ticket-triage", "skill:changelog-draft", "skill:vendor-skill"]
    activation = {
        "declared": labels,
        "exercised": ["skill:release-notes"],
        "unverified": ["skill:changelog-draft", "skill:vendor-skill"],
        "unavailable": ["skill:ticket-triage"],
    }
    # The report reads the activation labels by member name, also on a payload not yet promoted.
    before = coverage_view(provenance["component_coverage"], {"activation": activation})
    assert apply_runtime_coverage(provenance, _engine(exercised=("skill:release-notes",))) == 1
    after = coverage_view(provenance["component_coverage"], {"activation": activation})

    for view in (before, after):
        assert view is not None
        observed = {row["name"]: row["observed"] for row in view["rows"]}
        assert observed[RELEASE_NOTES] == "exercised"
        assert observed[TICKET_TRIAGE] == "unavailable"
        assert observed[CHANGELOG_DRAFT] == observed[VENDORED] == "unverified"
        unobserved = {row["name"] for row in view["staged_not_observed_rows"]}
        assert unobserved == {TICKET_TRIAGE, CHANGELOG_DRAFT, VENDORED, STYLE_RULE}
        assert view["staged_not_observed"] == 4
    assert {row["name"]: row["state"] for row in after["rows"]}[RELEASE_NOTES] == "exercised"
