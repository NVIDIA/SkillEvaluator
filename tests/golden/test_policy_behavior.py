# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden behavior tests for the default public validation policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli

_BODY = """
# {name}

## Instructions

Do the task and report the result clearly.

## Examples

User: "do it"

Assistant: "done"
"""


def _write_skill(root: Path, name: str, author_line: str | None) -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    meta = f"metadata:\n  author: {author_line}\n" if author_line else ""
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A test skill for policy behavior checks.\n{meta}---\n"
        + _BODY.format(name=name),
        encoding="utf-8",
    )
    return skill


def _validate(skill: Path, out: Path, *args: str) -> int:
    result = CliRunner().invoke(
        cli,
        [
            "validate",
            str(skill),
            "--tiers",
            "1",
            "--no-llm",
            "--checks",
            "schema",
            "-r",
            "json,html",
            "-o",
            str(out),
            *args,
        ],
    )
    return result.exit_code


def test_default_policy_accepts_public_author_metadata(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    nvidia = _write_skill(tmp_path, "nvidia-skill", "Dev One <dev@nvidia.com>")
    external = _write_skill(tmp_path, "ext-skill", "Jane Doe <jane@example.com>")
    missing = _write_skill(tmp_path, "no-author-skill", None)
    team = _write_skill(tmp_path, "team-skill", "Example Platform Team")
    malformed = _write_skill(tmp_path, "malformed-skill", "dev@example.com")

    assert _validate(nvidia, out) == 0
    assert _validate(external, out) == 0
    # The Agent Skills baseline requires only name and description, so a missing
    # author or a team name without an email is advisory under the default policy.
    assert _validate(missing, out) == 0
    assert _validate(team, out) == 0
    # A malformed value (here a bare email) still blocks.
    assert _validate(malformed, out) == 1


def test_external_alias_matches_default_public_policy(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    external = _write_skill(tmp_path, "ext-skill", "Jane Doe <jane@example.com>")
    missing = _write_skill(tmp_path, "no-author-skill", None)

    assert _validate(external, out, "--external") == 0  # domain check disabled
    assert _validate(missing, out, "--external") == 0  # author is advisory


@pytest.mark.parametrize(
    "options",
    [[], ["--profile", "default"], ["--profile", "external"], ["--external"], ["--external", "--profile", "default"]],
)
def test_default_and_legacy_cli_forms_keep_author_validation(tmp_path: Path, options: list[str]) -> None:
    present = _write_skill(tmp_path, "attributed-skill", "Demo Author <author@example.com>")
    missing = _write_skill(tmp_path, "unattributed-skill", None)
    malformed = _write_skill(tmp_path, "malformed-skill", "Demo Author <author>")
    assert _validate(present, tmp_path / "passed", *options) == 0
    assert _validate(missing, tmp_path / "advisory", *options) == 0
    assert _validate(malformed, tmp_path / "failed", *options) == 1
    report = json.loads(next((tmp_path / "passed").glob("skillevaluator-output-*.json")).read_text())
    assert report["policy"]["profile"] == "default"
    assert report["policy"]["audience"] == "all"
    assert report["policy"]["metadata_id_requirement"] == "ignore"


def _write_skill_with_id(root: Path, name: str, author_line: str, metadata_id: str) -> Path:
    skill = _write_skill(root, name, author_line)
    manifest = skill / "SKILL.md"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("metadata:\n", f"metadata:\n  id: {metadata_id}\n", 1),
        encoding="utf-8",
    )
    return skill


def test_registry_profile_requires_contact_author_and_metadata_id(tmp_path: Path) -> None:
    complete = _write_skill_with_id(tmp_path, "complete-skill", "Demo Author <author@example.com>", "complete-skill")
    no_id = _write_skill(tmp_path, "no-id-skill", "Demo Author <author@example.com>")
    team = _write_skill_with_id(tmp_path, "team-skill", "Example Platform Team", "team-skill")
    bad_id = _write_skill_with_id(tmp_path, "bad-id-skill", "Demo Author <author@example.com>", "Bad_ID")

    assert _validate(complete, tmp_path / "complete", "--profile", "registry") == 0
    assert _validate(no_id, tmp_path / "no-id", "--profile", "registry") == 1
    assert _validate(team, tmp_path / "team", "--profile", "registry") == 1
    assert _validate(bad_id, tmp_path / "bad-id", "--profile", "registry") == 1
    # The same invalid ID is advisory under the default policy.
    assert _validate(bad_id, tmp_path / "bad-id-default") == 0

    report = json.loads(next((tmp_path / "complete").glob("skillevaluator-output-*.json")).read_text())
    assert report["policy"]["profile"] == "registry"
    assert report["policy"]["metadata_id_requirement"] == "require"


def test_benchmark_md_generated_for_skill(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    nvidia = _write_skill(tmp_path, "nvidia-skill", "Dev One <dev@nvidia.com>")
    assert _validate(nvidia, out) == 0
    assert (out / "BENCHMARK.md").exists()


def test_catalog_reports_shared_metadata_id_and_blocks_under_registry(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    _write_skill_with_id(catalog, "first-skill", "Demo Author <author@example.com>", "shared-id")
    _write_skill_with_id(catalog, "second-skill", "Demo Author <author@example.com>", "shared-id")
    _write_skill_with_id(catalog, "third-skill", "Demo Author <author@example.com>", "third-skill")

    # Each skill passes on its own; the shared ID is advisory under the default policy.
    assert _validate(catalog, tmp_path / "default") == 0
    summary = json.loads((tmp_path / "default" / "catalog-summary.json").read_text(encoding="utf-8"))
    assert summary["overall_passed"] is True
    assert summary["metadata_id_conflicts"] == [
        {
            "metadata_id": "shared-id",
            "skills": ["first-skill", "second-skill"],
            "severity": "medium",
            "blocking": False,
        }
    ]

    # The registry profile requires unique IDs, so both sharing skills fail.
    assert _validate(catalog, tmp_path / "registry", "--profile", "registry") == 1
    summary = json.loads((tmp_path / "registry" / "catalog-summary.json").read_text(encoding="utf-8"))
    assert summary["metadata_id_conflicts"][0]["blocking"] is True
    failed = {skill["name"]: skill["reason"] for skill in summary["skills"] if not skill["passed"]}
    assert failed == {
        "first-skill": "metadata.id 'shared-id' is also declared by second-skill",
        "second-skill": "metadata.id 'shared-id' is also declared by first-skill",
    }


def test_catalog_without_shared_ids_records_no_conflicts(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    _write_skill_with_id(catalog, "first-skill", "Demo Author <author@example.com>", "first-skill")
    _write_skill(catalog, "second-skill", "Demo Author <author@example.com>")

    assert _validate(catalog, tmp_path / "out") == 0
    summary = json.loads((tmp_path / "out" / "catalog-summary.json").read_text(encoding="utf-8"))
    assert summary["metadata_id_conflicts"] == []


def test_parallel_catalog_applies_the_registry_id_conflict(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    _write_skill_with_id(catalog, "first-skill", "Demo Author <author@example.com>", "shared-id")
    _write_skill_with_id(catalog, "second-skill", "Demo Author <author@example.com>", "shared-id")

    assert _validate(catalog, tmp_path / "out", "--profile", "registry", "--workers", "2") == 1
    summary = json.loads((tmp_path / "out" / "catalog-summary.json").read_text(encoding="utf-8"))
    assert [skill["passed"] for skill in summary["skills"]] == [False, False]
    assert summary["metadata_id_conflicts"][0]["skills"] == ["first-skill", "second-skill"]


def test_catalog_id_check_runs_only_when_the_schema_check_is_selected(tmp_path: Path) -> None:
    # --checks and --skip run exactly the checks they name: a catalog-wide
    # metadata.id conflict must not fail a run that did not select schema.
    catalog = tmp_path / "catalog"
    _write_skill_with_id(catalog, "first-skill", "Demo Author <author@example.com>", "shared-id")
    _write_skill_with_id(catalog, "second-skill", "Demo Author <author@example.com>", "shared-id")

    # schema selected: the duplicate blocks under registry, as before.
    assert _validate(catalog, tmp_path / "schema-on", "--profile", "registry") == 1

    # schema deselected: the per-skill version checks pass, so the run passes.
    assert (
        CliRunner()
        .invoke(
            cli,
            [
                "validate", str(catalog), "--tiers", "1", "--no-llm",
                "--checks", "version", "--profile", "registry",
                "-r", "json", "-o", str(tmp_path / "checks-version"),
            ],
        )
        .exit_code
        == 0
    )
    summary = json.loads((tmp_path / "checks-version" / "catalog-summary.json").read_text(encoding="utf-8"))
    assert summary["metadata_id_conflicts"] == []

    # --skip schema is the same contract.
    assert (
        CliRunner()
        .invoke(
            cli,
            [
                "validate", str(catalog), "--tiers", "1", "--no-llm",
                "--skip", "schema", "--profile", "registry",
                "-r", "json", "-o", str(tmp_path / "skip-schema"),
            ],
        )
        .exit_code
        == 0
    )
    summary = json.loads((tmp_path / "skip-schema" / "catalog-summary.json").read_text(encoding="utf-8"))
    assert summary["metadata_id_conflicts"] == []
