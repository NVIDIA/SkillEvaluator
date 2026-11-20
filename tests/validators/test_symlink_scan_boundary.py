# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: a Tier 1 report never quotes content a symlink pulls in from outside the skill."""

import json
from pathlib import Path

import pytest

from skillevaluator.reporting import HTMLReporter, JSONReporter, MarkdownReporter
from skillevaluator.tier1.commands import run_validation

HOST_MARKER = "HOSTMARKER-7f3a"
HOST_EMAIL = "alice.host@acme-widgets.net"


@pytest.fixture
def skill_with_outside_symlink(tmp_path: Path) -> Path:
    host_dir = tmp_path / "host-private"
    host_dir.mkdir()
    host_file = host_dir / "host_file.txt"
    host_file.write_text(f"contact {HOST_EMAIL} token={HOST_MARKER}\n", encoding="utf-8")

    skill = tmp_path / "demo-skill"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo skill used for testing. Use when demoing.\n---\n"
        "# Demo\nSee references/notes.md\n",
        encoding="utf-8",
    )
    try:
        (skill / "references" / "notes.md").symlink_to(host_file)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    return skill


def test_default_tier1_reports_do_not_contain_host_file_content(skill_with_outside_symlink: Path) -> None:
    results = run_validation(skill_with_outside_symlink, checks="pii,code-integrity", content_type="skill")

    rendered = [
        JSONReporter(include_timestamp=False).render_all(results),
        MarkdownReporter().render_all(results),
        HTMLReporter().render_all(results),
    ]
    for report in rendered:
        assert HOST_MARKER not in report
        assert HOST_EMAIL not in report
        assert "host-private" not in report
        assert "references/notes.md" in report

    report = json.loads(rendered[0])
    hygiene = next(r for r in report["results"] if r["validator"] == "Code Integrity & Hygiene")
    skipped = [f for f in hygiene["findings"] if f["check_name"] == "symlink_outside_skill"]
    assert [f["file_path"] for f in skipped] == ["references/notes.md"]
    assert skipped[0]["severity"] == "low"
    assert hygiene["passed"] is True


@pytest.mark.parametrize("requirements_name", ["requirements.txt", "requirements-dev.txt"])
def test_outside_requirements_symlink_is_reported_but_never_audited(tmp_path: Path, requirements_name: str) -> None:
    """The dependency audit must not read or quote a requirements file that a symlink pulls in from outside."""
    host_dir = tmp_path / "host-private"
    host_dir.mkdir()
    host_file = host_dir / "host_requirements.txt"
    host_file.write_text(f"{HOST_PACKAGE}\n", encoding="utf-8")

    skill = tmp_path / "demo-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo skill used for testing. Use when demoing.\n---\n# Demo\n",
        encoding="utf-8",
    )
    try:
        (skill / requirements_name).symlink_to(host_file)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    results = run_validation(skill, checks="pii,code-integrity", content_type="skill")

    rendered = [
        JSONReporter(include_timestamp=False).render_all(results),
        MarkdownReporter().render_all(results),
        HTMLReporter().render_all(results),
    ]
    for report in rendered:
        assert HOST_PACKAGE not in report
        assert "Unpinned" not in report
        assert "host-private" not in report
        assert requirements_name in report

    report = json.loads(rendered[0])
    hygiene = next(r for r in report["results"] if r["validator"] == "Code Integrity & Hygiene")
    skipped = [f for f in hygiene["findings"] if f["check_name"] == "symlink_outside_skill"]
    assert [f["file_path"] for f in skipped] == [requirements_name]
    assert hygiene["passed"] is True


def test_default_tier1_reports_do_not_contain_content_behind_an_outside_junction(tmp_path: Path, make_junction) -> None:
    """A directory junction has no symlink bit, so the walk must prune it explicitly."""
    host_dir = tmp_path / "host-private"
    host_dir.mkdir()
    (host_dir / "notes.md").write_text(f"contact {HOST_EMAIL} token={HOST_MARKER}\n", encoding="utf-8")

    skill = tmp_path / "demo-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo skill used for testing. Use when demoing.\n---\n"
        "# Demo\nSee references/notes.md\n",
        encoding="utf-8",
    )
    make_junction(skill / "references", host_dir)

    results = run_validation(skill, checks="pii,code-integrity", content_type="skill")

    rendered = [
        JSONReporter(include_timestamp=False).render_all(results),
        MarkdownReporter().render_all(results),
        HTMLReporter().render_all(results),
    ]
    for report in rendered:
        assert HOST_MARKER not in report
        assert HOST_EMAIL not in report
        assert "host-private" not in report

    report = json.loads(rendered[0])
    hygiene = next(r for r in report["results"] if r["validator"] == "Code Integrity & Hygiene")
    skipped = [f for f in hygiene["findings"] if f["check_name"] == "symlink_outside_skill"]
    assert [f["file_path"] for f in skipped] == ["references"]
    assert skipped[0]["severity"] == "low"
    assert hygiene["passed"] is True


def test_outside_license_symlink_is_reported_but_never_supplies_the_license(tmp_path: Path) -> None:
    """A skill with no declared license must not pick one up from a LICENSE symlink that leaves the skill."""
    host_dir = tmp_path / "host-private"
    host_dir.mkdir()
    (host_dir / "host_license.txt").write_text(
        "Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/\n"
        "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n",
        encoding="utf-8",
    )
    skill = tmp_path / "demo-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo skill used for testing. Use when demoing.\n---\n# Demo\n",
        encoding="utf-8",
    )
    try:
        (skill / "LICENSE").symlink_to(host_dir / "host_license.txt")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    results = run_validation(skill, checks="license,code-integrity", content_type="skill")

    license_result = next(r for r in results if r.validator_name == "License Compliance")
    assert license_result.metadata.get("license") is None
    assert license_result.metadata.get("license_status") != "allowed"
    assert license_result.metadata.get("license_source") != "license_file"

    rendered = [
        JSONReporter(include_timestamp=False).render_all(results),
        MarkdownReporter().render_all(results),
        HTMLReporter().render_all(results),
    ]
    for report in rendered:
        assert "host-private" not in report
        assert "LICENSE" in report
    # The HTML report carries its own SPDX header, so only the data reports are checked for the id.
    assert "Apache-2.0" not in rendered[0]
    assert "Apache-2.0" not in rendered[1]

    report = json.loads(rendered[0])
    hygiene = next(r for r in report["results"] if r["validator"] == "Code Integrity & Hygiene")
    skipped = [f for f in hygiene["findings"] if f["check_name"] == "symlink_outside_skill"]
    assert [f["file_path"] for f in skipped] == ["LICENSE"]
