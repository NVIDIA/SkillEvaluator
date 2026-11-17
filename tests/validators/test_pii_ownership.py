# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership policy applies only to explicit contact metadata, never a domain."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models.result import Severity
from skillevaluator.validators.policy import ValidationPolicy, apply_policy
from skillevaluator.validators.security import SecurityValidator

ADDRESS = "maintainer@company.com"
POLICY = ValidationPolicy(severity_overrides={"PII.ownership_emails": Severity.INFO})


def scan(tmp_path: Path, frontmatter: str, body: str = "", name: str = "SKILL.md"):
    skill = tmp_path / "contact-skill"
    skill.mkdir(exist_ok=True)
    (skill / name).write_text(f"---\nname: contact-skill\n{frontmatter}\n---\n{body}\n", encoding="utf-8")
    result = SecurityValidator(submitter_usernames=[]).validate_pii_only(skill)
    return skill, result


@pytest.mark.parametrize("field", ["pic", "email"])
@pytest.mark.parametrize("value", [ADDRESS, f'"Maintainer <{ADDRESS}>"', f"'{ADDRESS}'"])
def test_ownership_requires_explicit_policy_and_remains_visible(tmp_path, field, value):
    _, result = scan(tmp_path, f"metadata:\n  {field}: {value}")
    assert not result.passed
    assert [(f.check_name, f.severity) for f in result.findings] == [("ownership_emails", Severity.HIGH)]
    apply_policy([result], POLICY)
    assert result.passed
    assert result.findings[0].severity == Severity.INFO
    assert result.findings[0].metadata["matched_value"] == ADDRESS


def test_same_address_in_body_and_metadata_stays_independently_blocking(tmp_path):
    _, result = scan(tmp_path, f"metadata:\n  pic: {ADDRESS}", f"Contact {ADDRESS} for access.")
    apply_policy([result], POLICY)
    assert not result.passed
    assert {f.check_name for f in result.findings} == {"emails", "ownership_emails"}
    assert all(f.metadata["occurrence_count"] == 1 for f in result.findings)


@pytest.mark.parametrize("order", ["owner-first", "body-first"])
def test_flow_metadata_does_not_hide_another_address_on_the_same_line(tmp_path, order):
    fields = [f"pic: {ADDRESS}", f"description: {ADDRESS}"]
    if order == "body-first":
        fields.reverse()
    _, result = scan(tmp_path, "metadata: {" + ", ".join(fields) + "}")
    apply_policy([result], POLICY)
    assert not result.passed
    assert {f.check_name for f in result.findings} == {"emails", "ownership_emails"}


@pytest.mark.parametrize("author", ["a@company.com", "long-author-address@company.com"])
@pytest.mark.parametrize("ordinary", [False, True])
def test_author_exemption_preserves_adjacent_flow_contact_offsets(tmp_path, author, ordinary):
    extra = f", description: {ADDRESS}" if ordinary else ""
    _, result = scan(
        tmp_path,
        f'metadata: {{\n  author: "Author <{author}>", pic: {ADDRESS}{extra}}}',
    )
    apply_policy([result], POLICY)
    assert result.passed is not ordinary
    expected = {"ownership_emails", "emails"} if ordinary else {"ownership_emails"}
    assert {f.check_name for f in result.findings} == expected
    assert all(f.metadata["matched_value"] == ADDRESS for f in result.findings)


@pytest.mark.parametrize(
    "frontmatter",
    [
        f"pic: {ADDRESS}",
        f"metadata:\n  description: {ADDRESS}",
        f"metadata:\n  nested:\n    pic: {ADDRESS}",
        f"metadata:\n  pic: Contact {ADDRESS} for support",
        f"metadata:\n  pic: 'Maintainer <{ADDRESS}'",
        f"metadata:\n  pic: '{ADDRESS}>'",
        f"metadata:\n  pic: |\n    {ADDRESS}",
        f"metadata:\n  pic: {ADDRESS}\n  pic: someone@company.com",
        f"metadata: [\n  pic: {ADDRESS}",
        f"original: &owner {ADDRESS}\nmetadata:\n  pic: *owner",
        f"original: &owner {{pic: {ADDRESS}}}\nmetadata:\n  <<: *owner",
        f"metadata:\n  pic: {ADDRESS}\nlarge: " + "x" * 65_536,
    ],
)
def test_ambiguous_or_unrecognized_metadata_retains_ordinary_email_policy(tmp_path, frontmatter):
    _, result = scan(tmp_path, frontmatter)
    apply_policy([result], POLICY)
    assert not result.passed
    assert {f.check_name for f in result.findings} == {"emails"}


def test_other_markdown_files_cannot_claim_skill_ownership_metadata(tmp_path):
    _, result = scan(tmp_path, f"metadata:\n  pic: {ADDRESS}", name="contacts.md")
    assert {f.check_name for f in result.findings} == {"emails"}


@pytest.mark.parametrize("name", ["SKILL.md", "skill.md"])
@pytest.mark.parametrize("nested", ["references", "references/vendored/deep", "scripts"])
def test_nested_skill_manifest_cannot_claim_ownership_metadata(tmp_path, nested, name):
    skill, _ = scan(tmp_path, "description: Plain skill")
    folder = skill / nested
    folder.mkdir(parents=True)
    (folder / name).write_text(
        f"---\nname: vendored\nmetadata:\n  pic: {ADDRESS}\n---\nVendored notes.\n", encoding="utf-8"
    )
    result = SecurityValidator(submitter_usernames=[]).validate_pii_only(skill)
    apply_policy([result], POLICY)
    assert not result.passed
    assert [(f.check_name, f.severity) for f in result.findings] == [("emails", Severity.HIGH)]
    assert result.findings[0].metadata["occurrences"] == [{"file": f"{nested}/{name}", "line": 4}]


def test_root_ownership_does_not_cover_nested_manifest_contacts(tmp_path):
    skill, _ = scan(tmp_path, f"metadata:\n  pic: {ADDRESS}")
    (skill / "references").mkdir()
    (skill / "references" / "SKILL.md").write_text(
        f"---\nname: vendored\nmetadata:\n  pic: {ADDRESS}\n  email: other@company.com\n---\n", encoding="utf-8"
    )
    result = SecurityValidator(submitter_usernames=[]).validate_pii_only(skill)
    apply_policy([result], POLICY)
    assert not result.passed
    by_check = {(f.check_name, f.metadata["matched_value"]): f.severity for f in result.findings}
    assert by_check == {
        ("ownership_emails", ADDRESS): Severity.INFO,
        ("emails", ADDRESS): Severity.HIGH,
        ("emails", "other@company.com"): Severity.HIGH,
    }


def test_nested_manifest_is_not_exempt_when_scanning_a_skill_collection(tmp_path):
    collection = tmp_path / "skills"
    skill = collection / "contact-skill"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: contact-skill\ndescription: Plain skill\n---\n", encoding="utf-8")
    (skill / "references" / "SKILL.md").write_text(
        f"---\nname: vendored\nmetadata:\n  pic: {ADDRESS}\n---\n", encoding="utf-8"
    )
    result = SecurityValidator(submitter_usernames=[]).validate_pii_only(collection)
    apply_policy([result], POLICY)
    assert not result.passed
    assert any(f.check_name == "emails" and f.severity == Severity.HIGH for f in result.findings)


def test_credentials_in_the_same_skill_still_block(tmp_path):
    _, result = scan(tmp_path, f"metadata:\n  pic: {ADDRESS}", 'password = "sample-sensitive-value"')
    apply_policy([result], POLICY)
    assert not result.passed
    assert any(f.category == "PII" and f.check_name != "ownership_emails" for f in result.findings)


@pytest.mark.parametrize("subcommand", [[], ["validate"]])
def test_tier1_policy_option_controls_real_pii_scan(tmp_path, subcommand):
    skill, _ = scan(tmp_path, f"metadata:\n  pic: {ADDRESS}")
    policy = tmp_path / "policy.yaml"
    policy.write_text("severity_overrides:\n  PII.ownership_emails: info\n", encoding="utf-8")
    args = [
        "tier1",
        *subcommand,
        str(skill),
        "--checks",
        "pii",
        "--no-llm",
        "-r",
        "json,html,markdown",
        "-o",
        str(tmp_path / "reports"),
    ]
    runner = CliRunner()
    assert runner.invoke(cli, args).exit_code == 1
    approved = runner.invoke(cli, [*args, "--policy", str(policy)])
    assert approved.exit_code == 0, approved.output
    import json

    report = json.loads((tmp_path / "reports" / "skillevaluator-tier1.json").read_text())
    assert report["overall_passed"]
    findings = [f for result in report["results"] for f in result["findings"]]
    assert any(f["check_name"] == "ownership_emails" and f["severity"] == "info" for f in findings)


def test_tier1_rejects_missing_policy_before_scanning(tmp_path):
    skill, _ = scan(tmp_path, f"metadata:\n  pic: {ADDRESS}")
    result = CliRunner().invoke(cli, ["tier1", str(skill), "--policy", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 2
