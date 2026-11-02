# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evidence-bound PII waivers.

A waiver accepts one reviewed value at one file and line. It must stop
applying, visibly, when the value, place, category, or review window changes,
and a waiver file that does not load must apply nothing and fail the scan.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from skillevaluator.reporting import JSONReporter, SARIFReporter
from skillevaluator.utils.secret_redaction import SENSITIVE_PII_CATEGORIES
from skillevaluator.validators import pii_waivers
from skillevaluator.validators.pii_waivers import PIIWaiver, PIIWaiverSet, pii_evidence_sha256, today_utc
from skillevaluator.validators.security import SecurityValidator

ADDRESS = "boss@company.com"


def _waiver(**overrides: object) -> dict:
    waiver = {
        "id": "PII-001",
        "category": "emails",
        "file": "skill/contact.md",
        "line": 2,
        "evidence_sha256": pii_evidence_sha256("emails", ADDRESS),
        "reason": "Public support address of the documented service",
        "owner": "docs-team",
        "created": "2026-01-01",
        "expires": (today_utc() + timedelta(days=30)).isoformat(),
    }
    waiver.update(overrides)
    return waiver


def _loaded(**overrides: object) -> dict:
    """Return :class:`PIIWaiver` fields for a waiver built without the loader."""
    waiver = _waiver(**overrides)
    return {
        "waiver_id": waiver.pop("id"),
        **waiver,
        "created": date.fromisoformat(waiver["created"]),
        "expires": date.fromisoformat(waiver["expires"]),
    }


def _write_waivers(root: Path, *waivers: dict) -> Path:
    path = root / "pii-waivers.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "waivers": list(waivers)}, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "contact.md").write_text(f"# Contact\nWrite to {ADDRESS} for access.\n", encoding="utf-8")
    return skill


def _scan(target: Path, waiver_path: Path | None):
    return SecurityValidator(submitter_usernames=[], pii_waiver_path=waiver_path).validate_pii_only(target)


def _pii(result) -> list:
    return [finding for finding in result.findings if finding.category == "PII"]


def _waiver_checks(result) -> list[str]:
    return [finding.check_name for finding in result.findings if finding.category == "PII_WAIVER"]


def test_pii_finding_reports_the_digest_a_waiver_needs(skill: Path) -> None:
    (finding,) = _pii(_scan(skill, None))

    assert finding.metadata["evidence_sha256"] == pii_evidence_sha256("emails", ADDRESS)
    assert finding.metadata["occurrences"] == [{"file": "contact.md", "line": 2}]


def test_matching_waiver_suppresses_the_finding_and_stays_visible(skill: Path, tmp_path: Path) -> None:
    result = _scan(skill, _write_waivers(tmp_path, _waiver()))

    assert result.passed
    assert _pii(result) == []
    (applied,) = [finding for finding in result.findings if finding.category == "PII_WAIVER"]
    assert applied.check_name == "applied-pii-waiver"
    assert applied.severity.value == "info"
    # The same path base as the PII findings of the run; the waiver's own path is kept in metadata.
    assert (applied.file_path, applied.line_number) == ("contact.md", 2)
    assert applied.metadata["waiver_file"] == "skill/contact.md"
    assert applied.metadata["waiver_id"] == "PII-001"
    assert applied.metadata["owner"] == "docs-team"
    assert "PII waivers suppressed 1 reviewed PII value(s)" in result.messages
    rendered = JSONReporter(include_timestamp=False).render(result)
    assert ADDRESS not in rendered
    # The waiver file holds the digest; the report identifies the waiver by id and file only.
    assert "evidence_sha256" not in applied.metadata
    assert pii_evidence_sha256("emails", ADDRESS) not in rendered


@pytest.mark.parametrize(
    ("content", "overrides"),
    [
        pytest.param(f"# Contact\n\nWrite to {ADDRESS} for access.\n", {}, id="line-moved"),
        pytest.param("# Contact\nWrite to alice@company.com for access.\n", {}, id="value-changed"),
        pytest.param(None, {"file": "skill/other.md"}, id="other-file"),
        pytest.param(None, {"category": "phone_numbers"}, id="other-category"),
    ],
)
def test_waiver_stops_applying_when_the_evidence_changes(
    skill: Path,
    tmp_path: Path,
    content: str | None,
    overrides: dict,
) -> None:
    if content is not None:
        (skill / "contact.md").write_text(content, encoding="utf-8")

    result = _scan(skill, _write_waivers(tmp_path, _waiver(**overrides)))

    assert not result.passed
    assert [finding.check_name for finding in _pii(result)] == ["emails"]
    assert _waiver_checks(result) == ["stale-pii-waiver"]


def test_waiver_for_a_file_outside_the_scan_is_neither_applied_nor_judged(skill: Path, tmp_path: Path) -> None:
    # Relative to the skill, not to the waiver file, so it names tmp_path/contact.md.
    result = _scan(skill, _write_waivers(tmp_path, _waiver(file="contact.md")))

    assert not result.passed
    assert [finding.check_name for finding in _pii(result)] == ["emails"]
    assert _waiver_checks(result) == []


def test_waived_value_does_not_hide_another_value_on_the_same_line(skill: Path, tmp_path: Path) -> None:
    (skill / "contact.md").write_text(f"# Contact\nWrite to {ADDRESS} or alice@company.com.\n", encoding="utf-8")

    result = _scan(skill, _write_waivers(tmp_path, _waiver()))

    assert not result.passed
    assert [finding.metadata["matched_value"] for finding in _pii(result)] == ["alice@company.com"]
    assert _waiver_checks(result) == ["applied-pii-waiver"]


def test_expired_waiver_is_reported_and_not_applied(skill: Path, tmp_path: Path) -> None:
    yesterday = (today_utc() - timedelta(days=1)).isoformat()

    result = _scan(skill, _write_waivers(tmp_path, _waiver(expires=yesterday)))

    assert not result.passed
    assert [finding.check_name for finding in _pii(result)] == ["emails"]
    (expired,) = [finding for finding in result.findings if finding.category == "PII_WAIVER"]
    assert (expired.check_name, expired.severity.value) == ("expired-pii-waiver", "medium")
    assert expired.file_path == "contact.md"


def test_waiver_applies_on_its_expiry_day(skill: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A waiver applies through ``expires``; the date is pinned so a run crossing UTC midnight cannot flake."""
    monkeypatch.setattr(pii_waivers, "today_utc", lambda: date(2026, 11, 2))

    result = _scan(skill, _write_waivers(tmp_path, _waiver(expires="2026-11-02")))

    assert result.passed
    assert _waiver_checks(result) == ["applied-pii-waiver"]


def test_created_may_be_one_day_ahead_of_utc(skill: Path, tmp_path: Path) -> None:
    """A reviewer east of UTC can enter their local date while UTC is still on the day before."""
    tomorrow = (today_utc() + timedelta(days=1)).isoformat()

    result = _scan(skill, _write_waivers(tmp_path, _waiver(created=tomorrow)))

    assert result.passed
    assert _waiver_checks(result) == ["applied-pii-waiver"]


_IN_TWO_DAYS = (today_utc() + timedelta(days=2)).isoformat()


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        pytest.param("version: [1\n", "invalid YAML", id="bad-yaml"),
        pytest.param({"version": 2, "waivers": []}, "version must be 1", id="unknown-version"),
        pytest.param({"version": 1, "waivers": [], "extra": True}, "top level", id="unknown-top-level-key"),
        pytest.param("version: 1\nversion: 1\nwaivers: []\n", "duplicate key", id="duplicate-key"),
        pytest.param({"version": 1, "waivers": {}}, "waivers must be a list", id="waivers-not-list"),
        pytest.param([_waiver(note="x")], "unknown field(s): note", id="unknown-field"),
        pytest.param([{k: v for k, v in _waiver().items() if k != "owner"}], "missing: owner", id="missing-field"),
        pytest.param([_waiver(), _waiver()], "used more than once", id="duplicate-id"),
        pytest.param([_waiver(category="nicknames")], "category", id="unknown-category"),
        pytest.param([_waiver(file="/etc/skill/contact.md")], ".file", id="absolute-path"),
        pytest.param([_waiver(file="../skill/contact.md")], ".file", id="parent-traversal"),
        pytest.param([_waiver(file="skill\\contact.md")], ".file", id="backslash-path"),
        pytest.param([_waiver(file="./skill/contact.md")], ".file", id="unnormalized-path"),
        pytest.param([_waiver(file="skill/a\x00b/contact.md")], ".file", id="nul-in-path"),
        pytest.param([_waiver(file="skill/contact.md\n")], ".file", id="control-character-in-path"),
        pytest.param([_waiver(line=0)], ".line", id="line-zero"),
        pytest.param([_waiver(line=True)], ".line", id="line-bool"),
        pytest.param([_waiver(evidence_sha256="abc")], "evidence_sha256", id="short-digest"),
        pytest.param([_waiver(reason="ok")], ".reason", id="short-reason"),
        pytest.param([_waiver(owner=" ")], ".owner", id="blank-owner"),
        pytest.param([_waiver(created="2026-13-01")], ".created", id="bad-date"),
        pytest.param([_waiver(created=_IN_TWO_DAYS)], "in the future", id="created-in-future"),
        pytest.param([_waiver(created="2026-02-01", expires="2026-01-01")], "before created", id="expires-first"),
    ],
)
def test_invalid_waiver_file_applies_nothing_and_fails_the_scan(
    skill: Path,
    tmp_path: Path,
    document: object,
    expected: str,
) -> None:
    waiver_path = tmp_path / "private" / "pii-waivers.yaml"
    waiver_path.parent.mkdir()
    if isinstance(document, list):
        document = {"version": 1, "waivers": document}
    text = document if isinstance(document, str) else yaml.safe_dump(document, sort_keys=False)
    waiver_path.write_text(text, encoding="utf-8")

    result = _scan(skill, waiver_path)

    assert not result.passed
    assert [finding.check_name for finding in _pii(result)] == ["emails"]
    (invalid,) = [finding for finding in result.findings if finding.category == "PII_WAIVER"]
    assert (invalid.check_name, invalid.severity.value) == ("invalid-pii-waiver-file", "high")
    assert expected in invalid.message
    assert invalid.file_path == "pii-waivers.yaml"
    assert str(tmp_path) not in JSONReporter(include_timestamp=False).render(result)


@pytest.mark.parametrize(
    ("waiver_dir", "waiver_file"),
    [
        pytest.param(".", ".claude/skills/foo/SKILL.md", id="through-the-link"),
        pytest.param(".", "skills/foo/SKILL.md", id="link-free-path"),
        pytest.param(".claude", "skills/foo/SKILL.md", id="waiver-file-beside-the-link"),
    ],
)
def test_waiver_through_a_symlinked_skill_directory_applies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    waiver_dir: str,
    waiver_file: str,
) -> None:
    """A waiver applies wherever it is judged, whichever path to the file it names."""
    _write_skills(tmp_path / "skills", "foo")
    (tmp_path / ".claude" / "skills").mkdir(parents=True)
    try:
        (tmp_path / ".claude" / "skills" / "foo").symlink_to(tmp_path / "skills" / "foo", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create a symlink here: {exc}")
    waiver_path = _write_waivers(tmp_path / waiver_dir, _waiver(file=waiver_file, line=5))
    monkeypatch.chdir(tmp_path)

    result = _scan(Path(".claude", "skills", "foo"), waiver_path)

    assert result.passed
    assert _pii(result) == []
    assert _dispositions(result) == [("applied-pii-waiver", "PII-001", "SKILL.md")]


def test_waiver_whose_file_cannot_be_located_is_not_judged(skill: Path, tmp_path: Path) -> None:
    """A path the OS rejects (here an embedded NUL) is skipped, never a crash of the scan."""
    waivers = PIIWaiverSet(
        path=tmp_path / "pii-waivers.yaml", waivers=[PIIWaiver(**_loaded(file="skill/a\x00b/contact.md"))]
    )
    waivers.cover(skill)

    assert waivers.matcher(skill / "contact.md") is None
    assert waivers.findings(skill) == []


def _write_skills(root: Path, *names: str, line: str = f"Write to {ADDRESS}.") -> None:
    for name in names:
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Contact helper\n---\n{line}\n", encoding="utf-8"
        )


def _dispositions(result) -> list[tuple[str, str, str]]:
    return [
        (finding.check_name, finding.metadata["waiver_id"], finding.file_path)
        for finding in result.findings
        if finding.category == "PII_WAIVER"
    ]


def test_folder_scan_binds_waivers_to_paths_under_the_waiver_file(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _write_skills(root, "one", "two")
    waiver_path = _write_waivers(root, _waiver(file="one/SKILL.md", line=5))

    result = _scan(root, waiver_path)

    assert [finding.file_path for finding in _pii(result)] == ["[two] SKILL.md"]
    # Folder runs name the waiver's file the way they name PII findings.
    assert _dispositions(result) == [("applied-pii-waiver", "PII-001", "[one] SKILL.md")]


def test_run_on_one_skill_judges_only_that_skills_waivers(tmp_path: Path) -> None:
    """One waiver file serves a whole folder; a run on one skill must not call the others stale."""
    root = tmp_path / "skills"
    _write_skills(root, "one", "two", "three")
    yesterday = (today_utc() - timedelta(days=1)).isoformat()
    waiver_path = _write_waivers(
        root,
        _waiver(id="PII-001", file="one/SKILL.md", line=5),
        _waiver(id="PII-002", file="two/SKILL.md", line=5),
        _waiver(id="PII-003", file="three/SKILL.md", line=5, expires=yesterday),
    )

    single = _scan(root / "one", waiver_path)

    assert single.passed
    assert _dispositions(single) == [("applied-pii-waiver", "PII-001", "SKILL.md")]
    folder = _scan(root, waiver_path)
    assert _dispositions(folder) == [
        ("applied-pii-waiver", "PII-001", "[one] SKILL.md"),
        ("applied-pii-waiver", "PII-002", "[two] SKILL.md"),
        ("expired-pii-waiver", "PII-003", "[three] SKILL.md"),
    ]


def test_waivers_of_skills_a_folder_run_stopped_before_are_not_judged(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _write_skills(root, "a-critical", line="SSN 219-09-9999 on file.")
    _write_skills(root, "b-contact")
    waiver_path = _write_waivers(root, _waiver(file="b-contact/SKILL.md", line=5))

    result = _scan(root, waiver_path)

    # A CRITICAL finding stops a folder run, so b-contact was never scanned.
    assert [finding.check_name for finding in _pii(result)] == ["ssn"]
    assert _waiver_checks(result) == []


def test_waiver_for_a_deleted_file_in_the_scan_is_stale(skill: Path, tmp_path: Path) -> None:
    result = _scan(skill, _write_waivers(tmp_path, _waiver(file="skill/removed.md")))

    assert _dispositions(result) == [("stale-pii-waiver", "PII-001", "removed.md")]


def test_waiver_dispositions_share_the_sarif_location_of_pii_findings(skill: Path, tmp_path: Path) -> None:
    (skill / "contact.md").write_text(f"# Contact\nWrite to {ADDRESS} or alice@company.com.\n", encoding="utf-8")
    result = _scan(skill, _write_waivers(tmp_path, _waiver()))

    sarif = json.loads(SARIFReporter(include_timestamp=False).render(result))
    uris = {
        entry["ruleId"]: entry["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        for entry in sarif["runs"][0]["results"]
    }

    assert set(uris.values()) == {"contact.md"}, uris


def test_waiver_file_inside_the_scanned_tree_is_scanned_like_any_other_file(skill: Path) -> None:
    """A waiver file kept in the skill ships with it, so its own text is PII-scanned."""
    waiver_path = _write_waivers(skill, _waiver(file="contact.md", owner="carol@company.com"))

    result = _scan(skill, waiver_path)

    assert not result.passed
    (owner,) = _pii(result)
    assert (owner.file_path, owner.metadata["matched_value"]) == ("pii-waivers.yaml", "carol@company.com")
    # The digest, dates, and paths in the file raise nothing; only the owner address does.
    assert _dispositions(result) == [("applied-pii-waiver", "PII-001", "contact.md")]


@pytest.mark.parametrize(
    ("category", "content", "value"),
    [
        pytest.param("home_paths", "Logs live in /home/bob/logs.\n", "/home/bob/", id="unrelated-home-directory"),
        pytest.param("credit_cards", "Card 4111 1111 1111 1112 failed.\n", "4111 1111 1111 1112", id="fails-luhn"),
    ],
)
def test_waiver_for_a_value_the_scan_does_not_report_is_stale(
    skill: Path,
    tmp_path: Path,
    category: str,
    content: str,
    value: str,
) -> None:
    (skill / "contact.md").write_text(f"# Contact\n{content}", encoding="utf-8")
    waiver = _waiver(category=category, evidence_sha256=pii_evidence_sha256(category, value))

    result = _scan(skill, _write_waivers(tmp_path, waiver))

    assert result.passed
    assert _pii(result) == []
    assert _waiver_checks(result) == ["stale-pii-waiver"]


@pytest.mark.parametrize(
    ("content", "check"),
    [
        pytest.param(
            "Connect with postgres://svc:a6@db.internal.corp/app now.\n", "database_credentials", id="credential"
        ),
        # The address is also the password, so the emails finding masks it as well.
        pytest.param('password = "carol@company.com"\n', "emails", id="value-holding-a-credential"),
    ],
)
def test_masked_values_carry_no_digest(skill: Path, content: str, check: str) -> None:
    """A digest of a masked value would let anyone holding the report test guesses for it offline."""
    (skill / "contact.md").write_text(f"# Database\n{content}", encoding="utf-8")

    (finding,) = [finding for finding in _pii(_scan(skill, None)) if finding.check_name == check]

    assert finding.metadata["matched_value"] == "[REDACTED]"
    assert "evidence_sha256" not in finding.metadata


@pytest.mark.parametrize(
    ("contact", "config"),
    [
        pytest.param(f"Write to {ADDRESS} for access.", f'password = "{ADDRESS}"\n', id="credential-elsewhere"),
        pytest.param(f'password = "{ADDRESS}"', None, id="value-became-a-credential"),
    ],
)
@pytest.mark.parametrize("expires_in", [30, -1], ids=["applied-or-stale", "expired"])
def test_waiver_findings_carry_no_digest(
    skill: Path,
    tmp_path: Path,
    contact: str,
    config: str | None,
    expires_in: int,
) -> None:
    """Once a waived value is masked anywhere, a digest in the report would let readers test guesses offline."""
    (skill / "contact.md").write_text(f"# Contact\n{contact}\n", encoding="utf-8")
    if config is not None:
        (skill / "config.py").write_text(config, encoding="utf-8")
    expires = (today_utc() + timedelta(days=expires_in)).isoformat()

    result = _scan(skill, _write_waivers(tmp_path, _waiver(expires=expires)))

    assert _waiver_checks(result)
    assert pii_evidence_sha256("emails", ADDRESS) not in JSONReporter(include_timestamp=False).render(result)


def test_digest_is_not_masked_by_a_credential_elsewhere_in_the_skill(skill: Path) -> None:
    """The digest is added after redaction, so a short password that occurs in its hex stays visible there."""
    digest = pii_evidence_sha256("emails", ADDRESS)
    password = next(digest[i : i + 2] for i in range(len(digest) - 1) if digest[i : i + 2].isdigit())
    (skill / "contact.md").write_text(
        f"# Contact\nWrite to {ADDRESS} for access.\nConnect with postgres://svc:{password}@db.internal.corp/app now.\n",
        encoding="utf-8",
    )

    findings = _pii(_scan(skill, None))

    assert "database_credentials" in [finding.check_name for finding in findings]
    (email,) = [finding for finding in findings if finding.check_name == "emails"]
    assert email.metadata["evidence_sha256"] == digest


@pytest.mark.parametrize("category", sorted(SENSITIVE_PII_CATEGORIES))
def test_credential_checks_cannot_be_waived(skill: Path, tmp_path: Path, category: str) -> None:
    """Credentials are removed or rotated, never waived, even with a digest computed from the source."""
    value = "postgres://svc:a6@db.internal.corp"
    (skill / "contact.md").write_text(f"# Database\nConnect with {value}/app now.\n", encoding="utf-8")
    waiver = _waiver(category=category, evidence_sha256=pii_evidence_sha256(category, value))

    result = _scan(skill, _write_waivers(tmp_path, waiver))

    assert not result.passed
    assert [finding.check_name for finding in _pii(result)] == ["database_credentials"]
    (invalid,) = [finding for finding in result.findings if finding.category == "PII_WAIVER"]
    assert invalid.check_name == "invalid-pii-waiver-file"
    assert f"category {category!r} is a credential check and cannot be waived" in invalid.message


def test_waiver_digest_follows_the_value_findings_report(
    skill: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waivers hash whatever a finding reports, so narrowing the reported value cannot break them."""
    monkeypatch.setattr(SecurityValidator, "_reported_value", staticmethod(lambda match: match.group().split("@")[0]))
    (finding,) = _pii(_scan(skill, None))
    assert finding.metadata["matched_value"] == "boss"

    result = _scan(skill, _write_waivers(tmp_path, _waiver(evidence_sha256=finding.metadata["evidence_sha256"])))

    assert result.passed
    assert _waiver_checks(result) == ["applied-pii-waiver"]


def test_waiver_dispositions_are_added_to_json_reports(skill: Path, tmp_path: Path) -> None:
    stale = _waiver(id="PII-002", line=9)
    result = _scan(skill, _write_waivers(tmp_path, _waiver(), stale))

    findings = json.loads(JSONReporter(include_timestamp=False).render(result))["findings"]

    assert sorted((finding["check_name"], finding["metadata"]["waiver_id"]) for finding in findings) == [
        ("applied-pii-waiver", "PII-001"),
        ("stale-pii-waiver", "PII-002"),
    ]
