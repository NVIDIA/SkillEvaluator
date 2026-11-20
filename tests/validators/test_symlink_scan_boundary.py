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
