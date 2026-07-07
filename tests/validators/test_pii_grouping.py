# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PII findings are grouped by matched value, not repeated per line.

A doc-heavy skill repeating the same placeholder address produced 76
near-identical PII errors for 22 unique addresses; the report must show
one finding per (check, matched value) with an occurrence count and the
line list, so repeated values read as one row instead of pages of noise.
"""

from pathlib import Path

import pytest

from skillevaluator.validators.security import SecurityValidator


@pytest.fixture
def skill_with_repeated_emails(tmp_path) -> Path:
    skill = tmp_path / "mail-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: mail-skill\ndescription: mail helper\n---\n")
    workflows = skill / "workflows"
    workflows.mkdir()
    (workflows / "guide.md").write_text(
        "Intro line without addresses.\n"
        "Contact boss@company.com for approval.\n"
        "Escalate to boss@company.com when urgent.\n"
        "CC alice@company.com on the thread.\n"
        "Archive mail from boss@company.com monthly.\n"
    )
    return skill


def _email_findings(result):
    return [f for f in result.findings if f.check_name == "emails"]


class TestPiiGroupingByValue:
    def test_repeated_value_collapses_to_one_finding(self, skill_with_repeated_emails):
        result = SecurityValidator().validate_pii_only(skill_with_repeated_emails)

        findings = _email_findings(result)
        by_value = {f.metadata.get("matched_value"): f for f in findings}
        assert set(by_value) == {"boss@company.com", "alice@company.com"}

        boss = by_value["boss@company.com"]
        assert boss.metadata["occurrence_count"] == 3
        assert [o["line"] for o in boss.metadata["occurrences"]] == [2, 3, 5]
        assert "boss@company.com" in boss.message
        assert "3 occurrences" in boss.message
        assert "workflows/guide.md" in boss.message
        assert boss.line_number == 2

    def test_single_occurrence_message_names_the_value(self, skill_with_repeated_emails):
        result = SecurityValidator().validate_pii_only(skill_with_repeated_emails)

        alice = {f.metadata.get("matched_value"): f for f in _email_findings(result)}["alice@company.com"]
        assert alice.metadata["occurrence_count"] == 1
        assert "alice@company.com" in alice.message
        assert "occurrences" not in alice.message

    def test_error_count_matches_unique_values(self, skill_with_repeated_emails):
        result = SecurityValidator().validate_pii_only(skill_with_repeated_emails)

        email_errors = [e for e in result.errors if "email" in e.lower()]
        assert len(email_errors) == 2

    def test_groups_span_files(self, skill_with_repeated_emails):
        (skill_with_repeated_emails / "workflows" / "extra.md").write_text("Also ping boss@company.com here.\n")

        result = SecurityValidator().validate_pii_only(skill_with_repeated_emails)

        boss = {f.metadata.get("matched_value"): f for f in _email_findings(result)}["boss@company.com"]
        assert boss.metadata["occurrence_count"] == 4
        files = {o["file"] for o in boss.metadata["occurrences"]}
        assert files == {"workflows/guide.md", "workflows/extra.md"}
        assert "workflows/extra.md" in boss.message

    def test_distinct_values_stay_separate(self, tmp_path):
        skill = tmp_path / "two-skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text("---\nname: two-skill\ndescription: two addresses\n---\n")
        (skill / "notes.md").write_text("Ping bob@company.com then carol@company.com.\n")

        result = SecurityValidator().validate_pii_only(skill)

        values = {f.metadata.get("matched_value") for f in _email_findings(result)}
        # regex reports the first match per line; a second value on another line groups separately
        (skill / "notes2.md").write_text("Direct line to carol@company.com.\n")
        result = SecurityValidator().validate_pii_only(skill)
        values = {f.metadata.get("matched_value") for f in _email_findings(result)}
        assert values == {"bob@company.com", "carol@company.com"}
