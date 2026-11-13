# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential findings must not copy detected values into shared CI evidence."""

from __future__ import annotations

import html
import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote

import pytest

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting.html import HTMLReporter
from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.validators.security import SecurityValidator

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location("redaction_runner", ROOT / "ci/run_tier1.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _fixture_value() -> str:
    # Synthetic value, intentionally assembled at runtime; never a credential.
    return "nvapi-" + "Ab9" * 16 + "<&%"


def _skill(workspace: Path) -> Path:
    skill = workspace / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: A sample formatting skill.\n---\n# Demo\n")
    (skill / "source.py").write_text('API_KEY = "' + _fixture_value() + '"\n')
    return skill


def _assert_no_secret(value: str | bytes) -> None:
    raw = value.encode() if isinstance(value, str) else value
    # Do not render synthetic values in assertion output.
    for encoding in (_fixture_value(), html.escape(_fixture_value()), quote(_fixture_value(), safe="")):
        if encoding.encode() in raw:
            pytest.fail("A synthetic credential was present in a shared output.")


def _assert_finding_ids_match_text(document: dict) -> None:
    """Every finding ID in a shared report is hashed from that report's redacted text."""
    finding_ids = set()
    for row in document["results"]:
        for item in row["findings"]:
            finding = Finding(
                **{
                    key: item[key]
                    for key in (
                        "category",
                        "severity",
                        "check_name",
                        "message",
                        "file_path",
                        "line_number",
                        "line_content",
                        "suggestion",
                    )
                }
            )
            assert item["finding_id"] == finding.finding_id
            assert item["occurrence_id"] == finding.occurrence_id
            finding_ids.add(finding.finding_id)
    assert {group["finding_id"] for group in document["finding_groups"]} == finding_ids


def test_pii_findings_do_not_expose_detected_credential(tmp_path):
    skill = _skill(tmp_path)
    result = SecurityValidator(use_llm=False).validate_pii_only(skill)
    assert not result.passed
    assert any(f.check_name == "hardcoded_secrets" for f in result.findings)
    _assert_no_secret(JSONReporter(include_timestamp=False).render_all([result]))
    _assert_no_secret(repr(result))


def test_runner_and_retained_publisher_never_export_detected_value(tmp_path, capsys, monkeypatch):
    workspace, reports = tmp_path / "workspace", tmp_path / "reports"
    skill = _skill(workspace)
    result = SecurityValidator(use_llm=False).validate_pii_only(skill)
    result.validator_name = "PII Scan"
    result.validator_description = "Detect sensitive identifiers"
    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    # Another scanner can echo a value that PII also detected.
    finding = payload["results"][0]["findings"][0]
    finding["message"] += " scanner echoed " + _fixture_value()
    finding["line_content"] = "truncated source: " + _fixture_value()[:24]
    finding["metadata"]["nested"] = {"detail": _fixture_value()[:24]}
    payload["results"][0]["legacy"]["errors"].append("other scanner: " + _fixture_value())

    def fake_cli(argv, **kwargs):
        target = Path(argv[argv.index("-o") + 1])
        assert not target.is_relative_to(reports)
        kwargs["stdout"].write(("raw scanner output " + _fixture_value()[:24]).encode())
        kwargs["stderr"].write(("raw scanner error " + _fixture_value()).encode())
        (target / "skillevaluator-tier1.json").write_text(json.dumps(payload))
        for extension in ("html", "md"):
            (target / f"skillevaluator-tier1.{extension}").write_text(
                "other scanner "
                + _fixture_value()
                + " "
                + html.escape(_fixture_value())
                + " "
                + quote(_fixture_value(), safe="")
                + " truncated: "
                + _fixture_value()[:24]
            )
        (target / "skillevaluator-tier1.sarif.json").write_text(
            json.dumps({"message": "other scanner " + _fixture_value()[:24]})
        )
        return subprocess.CompletedProcess(argv, 1)

    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills", "GITHUB_ACTIONS": "true"}, execute=fake_cli) == 1
    _assert_no_secret(capsys.readouterr().out)
    for path in reports.rglob("*"):
        if path.is_file():
            _assert_no_secret(path.read_bytes())
            if _fixture_value()[:24].encode() in path.read_bytes():
                pytest.fail("A truncated synthetic credential was present in a report.")
    _assert_finding_ids_match_text(json.loads((reports / "skill-001/skillevaluator-tier1.json").read_text()))

    from ci import publish_reports

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in reports.rglob("*"):
            if path.is_file():
                zipped.write(path, path.relative_to(reports).as_posix())
    summary, retained = publish_reports.read_artifact(archive.getvalue())
    _assert_no_secret(json.dumps({"summary": summary, "retained": retained}))

    # A previous trusted run can still contain raw PII and another scanner's
    # copy. Publisher must sanitize it before writing retained history.
    document = retained["skill-001"]
    old_finding = document["results"][0]["findings"][0]
    old_finding["metadata"]["matched_value"] = 'API_KEY = "' + _fixture_value() + '"'
    old_finding["message"] = "Hardcoded credential: " + _fixture_value()
    old_finding["line_content"] = 'API_KEY = "' + _fixture_value() + '"'
    document["results"][0]["legacy"]["errors"].append("scanner echoed " + _fixture_value())
    summary["extra"] = "summary echoed " + _fixture_value()
    old_archive = io.BytesIO()
    with zipfile.ZipFile(old_archive, "w") as zipped:
        zipped.writestr("summary.json", json.dumps(summary))
        zipped.writestr("skill-001/skillevaluator-tier1.json", json.dumps(document))
    safe_summary, safe_reports = publish_reports.read_artifact(old_archive.getvalue())
    _assert_no_secret(json.dumps({"summary": safe_summary, "retained": safe_reports}))
    _assert_finding_ids_match_text(safe_reports["skill-001"])

    spec = importlib.util.spec_from_file_location("publisher_fixtures", ROOT / "tests/ci/test_publish_reports.py")
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)

    class ArtifactAPI(fixtures.FakeAPI):
        def download_artifact(self, artifact_id):
            if artifact_id == 457:
                return super().download_artifact(artifact_id)
            return old_archive.getvalue()

    monkeypatch.setenv("TRUSTED_EVALUATOR_SHA", "c" * 40)
    monkeypatch.setenv("TRUSTED_CALLER_BLOBS", "b" * 40)
    monkeypatch.syspath_prepend(str(ROOT / "ci"))
    import verify_provenance

    monkeypatch.setattr(verify_provenance, "fetch_signing_keys", lambda: fixtures.JWKS)
    site, feedback = tmp_path / "site", tmp_path / "feedback"
    fixtures.publisher.publish(
        ArtifactAPI(record={**fixtures.run_record(), "conclusion": "failure"}),
        "owner/consumer",
        123,
        "https://owner.github.io/consumer",
        site,
        feedback,
    )
    for directory in (site, feedback):
        for path in directory.rglob("*"):
            if path.is_file():
                _assert_no_secret(path.read_bytes())
    assert (site / "runs/123/1/skill-001/skillevaluator-tier1.html").is_file()
    assert (feedback / "pr-comment.md").is_file()

    # A legacy private-key header does not identify the private key body. Do
    # not publish an artifact when its other findings could still contain it.
    old_finding["check_name"] = "private_keys"
    old_finding["metadata"]["matched_value"] = "-----BEGIN PRIVATE KEY-----"
    old_finding["line_content"] = "-----BEGIN PRIVATE KEY-----"
    unsafe_archive = io.BytesIO()
    with zipfile.ZipFile(unsafe_archive, "w") as zipped:
        zipped.writestr("summary.json", json.dumps(summary))
        zipped.writestr("skill-001/skillevaluator-tier1.json", json.dumps(document))
    with pytest.raises(publish_reports.PublicationError, match="safely redacted"):
        publish_reports.read_artifact(unsafe_archive.getvalue())


def test_redacted_report_does_not_keep_ids_hashed_from_the_credential(tmp_path):
    # A finding ID hashed from text that held a short credential would let a
    # reader of the redacted report confirm guesses of that credential.
    skill = _skill(tmp_path / "workspace")
    pii = SecurityValidator(use_llm=False).validate_pii_only(skill)
    pii.validator_name = "PII Scan"
    echo = ValidationResult(validator_name="Code Risk", validator_description="Static analysis")
    echoed = Finding(
        "SECURITY",
        Severity.HIGH,
        "hardcoded-credential",
        "Credential in source: " + _fixture_value(),
        "source.py",
        1,
        line_content='API_KEY = "' + _fixture_value() + '"',
    )
    echo.add_finding(echoed)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "skillevaluator-tier1.json").write_text(JSONReporter(include_timestamp=False).render_all([pii, echo]))

    sanitized = runner.sanitized_outputs(raw, runner.collect_sensitive_values(skill))

    _assert_no_secret(sanitized["skillevaluator-tier1.json"])
    for raw_id in (echoed.finding_id, echoed.occurrence_id):
        assert raw_id.encode() not in sanitized["skillevaluator-tier1.json"]
    document = json.loads(sanitized["skillevaluator-tier1.json"])
    _assert_finding_ids_match_text(document)
    assert document["finding_counts"]["unique_findings"] == len(document["finding_groups"]) == 2
    assert document["results"][1]["finding_counts"]["raw_occurrences"] == 1


def test_redacted_html_does_not_keep_ids_hashed_from_the_credential(tmp_path):
    # The HTML report embeds finding and occurrence IDs. Passed through text
    # redaction alone, it would keep the IDs hashed before the value was masked.
    skill = _skill(tmp_path / "workspace")
    echo = ValidationResult(validator_name="Code Risk", validator_description="Static analysis")
    echoed = Finding(
        "SECURITY",
        Severity.HIGH,
        "hardcoded-credential",
        "Credential in source: " + _fixture_value(),
        "source.py",
        1,
        line_content='API_KEY = "' + _fixture_value() + '"',
    )
    echo.add_finding(echoed)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "skillevaluator-tier1.json").write_text(
        JSONReporter(include_timestamp=False).render_all([echo]), encoding="utf-8"
    )
    native = HTMLReporter(include_timestamp=False).render_all([echo])
    assert echoed.finding_id in native and echoed.occurrence_id in native
    (raw / "skillevaluator-tier1.html").write_text(native, encoding="utf-8")

    sanitized = runner.sanitized_outputs(raw, runner.collect_sensitive_values(skill))

    page = sanitized["skillevaluator-tier1.html"]
    _assert_no_secret(page)
    for raw_id in (echoed.finding_id, echoed.occurrence_id):
        assert raw_id.encode() not in page
    # The page carries the IDs of the redacted finding, as the JSON does.
    redacted = json.loads(sanitized["skillevaluator-tier1.json"])["results"][0]["findings"][0]
    assert redacted["finding_id"].encode() in page
    assert redacted["occurrence_id"].encode() in page


def _database_url_line() -> str:
    # Synthetic driver-qualified URL; the PII collector does not name its password.
    return 'DATABASE_URL = "postgresql+psycopg2://admin:' + "hunter" + '22@db.example.net/app"'


def test_ids_are_refreshed_when_only_database_userinfo_is_masked(tmp_path, monkeypatch):
    # The redactor masks database userinfo even when no token was collected.
    # IDs hashed from the unmasked line would let a reader confirm a guess.
    workspace, reports = tmp_path / "workspace", tmp_path / "reports"
    skill = workspace / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: A sample formatting skill.\n---\n# Demo\n")
    (skill / "config.py").write_text(_database_url_line() + "\n")
    assert runner.collect_sensitive_values(skill) == set()
    scan = ValidationResult(validator_name="SkillSpector", validator_description="Static analysis")
    quoted = Finding(
        "SECURITY",
        Severity.HIGH,
        "hardcoded-database-url",
        "Database URL with credentials",
        "config.py",
        1,
        line_content=_database_url_line(),
    )
    scan.add_finding(quoted)
    raw_report = JSONReporter(include_timestamp=False).render_all([scan])

    def fake_cli(argv, **kwargs):
        (Path(argv[argv.index("-o") + 1]) / "skillevaluator-tier1.json").write_text(raw_report)
        return subprocess.CompletedProcess(argv, 1)

    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills", "GITHUB_ACTIONS": "true"}, execute=fake_cli) == 1
    shared = (reports / "skill-001/skillevaluator-tier1.json").read_text(encoding="utf-8")
    assert "admin:hunter" not in shared
    assert quoted.occurrence_id not in shared
    document = json.loads(shared)
    assert "[REDACTED]@db.example.net" in document["results"][0]["findings"][0]["line_content"]
    _assert_finding_ids_match_text(document)

    # A retained report from an older run still quotes the URL with the IDs
    # hashed from it; the publisher masks it and refreshes the IDs.
    from ci import publish_reports

    stale = json.loads(shared)
    stale_finding = stale["results"][0]["findings"][0]
    stale_finding["line_content"] = _database_url_line()
    stale_finding["occurrence_id"] = quoted.occurrence_id
    stale["finding_groups"][0]["occurrences"][0]["occurrence_id"] = quoted.occurrence_id
    stale["finding_groups"][0]["occurrences"][0]["line_content"] = _database_url_line()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("summary.json", (reports / "summary.json").read_text(encoding="utf-8"))
        zipped.writestr("skill-001/skillevaluator-tier1.json", json.dumps(stale))
    _summary, retained = publish_reports.read_artifact(archive.getvalue())
    published = json.dumps(retained)
    assert "admin:hunter" not in published
    assert quoted.occurrence_id not in published
    _assert_finding_ids_match_text(retained["skill-001"])


def test_refreshing_ids_keeps_the_shape_of_reports_written_before_ids() -> None:
    from skillevaluator.reporting.finding_summary import refresh_finding_identities

    document = {
        "results": [
            {
                "validator": "Code Risk",
                "findings": [
                    {
                        "category": "SECURITY",
                        "severity": "high",
                        "check_name": "shell",
                        "message": "subprocess call with shell=True",
                        "file_path": "run.py",
                        "line_number": 3,
                        "line_content": None,
                        "suggestion": None,
                        "metadata": None,
                    }
                ],
            }
        ]
    }
    before = json.loads(json.dumps(document))

    refresh_finding_identities(document)

    assert document == before


def _pii_finding(value: str) -> Finding:
    from skillevaluator.validators.pii_waivers import pii_evidence_sha256

    return Finding(
        category="PII",
        severity=Severity.MEDIUM,
        check_name="email",
        message=f"Email address: {value}",
        file_path="SKILL.md",
        line_number=4,
        line_content=f"Contact {value}",
        metadata={"matched_value": value, "evidence_sha256": pii_evidence_sha256("email", value)},
    )


def test_redacting_a_reported_pii_value_drops_its_evidence_digest() -> None:
    from skillevaluator.reporting.finding_summary import refresh_finding_identities
    from skillevaluator.utils.secret_redaction import Redactor
    from skillevaluator.validators.pii_waivers import pii_evidence_sha256

    masked, kept = "ops-sampletoken@example.com", "team@example.com"
    result = ValidationResult(validator_name="PII Scan")
    result.add_structured_finding(_pii_finding(masked), is_error=False)
    result.add_structured_finding(_pii_finding(kept), is_error=False)
    document = Redactor({"sampletoken"}).object(json.loads(JSONReporter().render_all([result])))

    refresh_finding_identities(document)

    by_value = {f["metadata"]["matched_value"]: f["metadata"] for f in document["results"][0]["findings"]}
    assert masked not in by_value
    redacted = next(meta for value, meta in by_value.items() if value != kept)
    assert "evidence_sha256" not in redacted
    assert by_value[kept]["evidence_sha256"] == pii_evidence_sha256("email", kept)
    rendered = json.dumps(document)
    assert pii_evidence_sha256("email", masked) not in rendered
    assert pii_evidence_sha256("email", kept) in rendered


def test_a_digest_that_does_not_match_its_reported_value_is_dropped() -> None:
    from skillevaluator.reporting.finding_summary import refresh_finding_identities

    finding = _pii_finding("team@example.com").to_dict()
    finding["metadata"]["evidence_sha256"] = "0" * 64
    document = {"results": [{"validator": "PII Scan", "findings": [finding]}]}

    refresh_finding_identities(document)

    assert "evidence_sha256" not in document["results"][0]["findings"][0]["metadata"]


def test_refreshing_finding_ids_does_not_load_the_validators() -> None:
    probe = (
        "import sys, skillevaluator.reporting.finding_summary; "
        "print(sorted(m for m in sys.modules if m.startswith('skillevaluator.validators')))"
    )
    loaded = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout

    assert loaded.strip() == "[]"


def test_interrupted_scan_never_copies_raw_output(tmp_path, capsys):
    workspace, reports = tmp_path / "workspace", tmp_path / "reports"
    _skill(workspace)

    def interrupted(argv, **kwargs):
        target = Path(argv[argv.index("-o") + 1])
        kwargs["stdout"].write(_fixture_value().encode())
        (target / "skillevaluator-tier1.json").write_text(_fixture_value())
        raise subprocess.TimeoutExpired(argv, 1)

    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills"}, execute=interrupted) == 1
    _assert_no_secret(capsys.readouterr().out)
    for path in reports.rglob("*"):
        if path.is_file():
            _assert_no_secret(path.read_bytes())


def test_credential_detected_in_one_skill_is_redacted_from_another_skill(tmp_path, capsys):
    workspace, reports = tmp_path / "workspace", tmp_path / "reports"
    _skill(workspace)
    other = workspace / "skills" / "other"
    other.mkdir()
    (other / "SKILL.md").write_text("---\nname: other\ndescription: A formatting skill.\n---\n# Other\n")
    # This source mentions the same value without a credential assignment.
    (other / "reference.md").write_text("Reference " + _fixture_value())

    def fake_cli(argv, **kwargs):
        target = Path(argv[argv.index("-o") + 1])
        skill = Path(argv[2])
        result = SecurityValidator(use_llm=False).validate_pii_only(skill)
        result.validator_name = "PII Scan"
        result.validator_description = "Detect sensitive identifiers"
        result.add_warning("Another scanner echoed " + _fixture_value())
        payload = JSONReporter(include_timestamp=False).render_all([result])
        (target / "skillevaluator-tier1.json").write_text(payload)
        return subprocess.CompletedProcess(argv, int(not result.passed))

    runner.run(workspace, reports, {"SKILL_PATHS": "skills"}, execute=fake_cli)
    _assert_no_secret(capsys.readouterr().out)
    for path in reports.rglob("*"):
        if path.is_file():
            _assert_no_secret(path.read_bytes())


def test_detected_credential_in_skill_path_never_reaches_summary_or_log(tmp_path, capsys):
    workspace, reports = tmp_path / "workspace", tmp_path / "reports"
    skill = workspace / "skills" / _fixture_value()
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Formats text.\n---\n# Demo\n")
    (skill / "source.py").write_text('API_KEY = "' + _fixture_value() + '"\n')
    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills"}) == 1
    _assert_no_secret(capsys.readouterr().out)
    for path in reports.rglob("*"):
        if path.is_file():
            _assert_no_secret(path.read_bytes())


def test_redactor_removes_numeric_secret_from_json_number():
    from skillevaluator.utils.secret_redaction import Redactor

    token = "1234567890123456"
    result = Redactor({token}).object({"scanner_value": int(token)})
    if token in json.dumps(result):
        pytest.fail("A numeric synthetic credential was present in JSON.")


def test_redactor_refuses_to_change_schema_keys():
    from skillevaluator.utils.secret_redaction import Redactor

    with pytest.raises(ValueError, match="report keys"):
        Redactor({"occurrence-secret-value"}).object({"occurrence_count": 1})


def test_pii_source_masks_multiple_credentials_in_one_line(tmp_path):
    skill = _skill(tmp_path)
    second = "nvapi-" + "Cd4" * 16
    (skill / "source.py").write_text(
        'API_KEY = "' + _fixture_value() + '"; PASSWORD = "' + second + '"; contact="user@real-domain.org"\n'
    )
    result = SecurityValidator(use_llm=False).validate_pii_only(skill)
    rendered = JSONReporter(include_timestamp=False).render_all([result])
    _assert_no_secret(rendered)
    if second in rendered:
        pytest.fail("A second synthetic credential was present in PII output.")


def test_preflight_withholds_results_when_file_is_too_large(tmp_path, capsys, monkeypatch):
    from ci import secret_redaction

    workspace, reports = tmp_path / "workspace", tmp_path / "reports"
    _skill(workspace)
    monkeypatch.setattr(secret_redaction, "MAX_FILE_BYTES", 8)
    called = []

    def execute(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("Unsafe preflight must stop before the scanner starts")

    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills"}, execute=execute) == 1
    assert not called
    _assert_no_secret(capsys.readouterr().out)
    _assert_no_secret((reports / "summary.json").read_bytes())


@pytest.mark.parametrize("category", ["assignment", "jwt", "github", "aws", "webhook", "database", "private_key"])
def test_preflight_collects_credential_types_in_source_without_exposing_them(tmp_path, category):
    from ci.secret_redaction import collect_sensitive_values

    skill = _skill(tmp_path)
    token = "Zv8" * 16
    cases = {
        "assignment": ('API_KEY = "' + token + '"', token),
        "jwt": (
            "eyJ" + "A" * 16 + ".eyJ" + "B" * 16 + "." + "C" * 16,
            "eyJ" + "A" * 16 + ".eyJ" + "B" * 16 + "." + "C" * 16,
        ),
        "github": ("gh" + "p_" + "A" * 36, "gh" + "p_" + "A" * 36),
        "aws": ("AK" + "IA" + "A" * 16, "AK" + "IA" + "A" * 16),
        "webhook": ("https://hooks.slack.com/services/T" + "A" * 8 + "/B" + "B" * 8 + "/" + "C" * 24, "C" * 24),
        "database": ("postgres://account:" + token + "@host.local/db", token),
        "private_key": ("-----BEGIN PRIVATE KEY-----\n" + token + "\n-----END PRIVATE KEY-----", token),
    }
    content, expected = cases[category]
    (skill / "source.py").write_text(content + "\n")
    tokens = collect_sensitive_values(skill)
    if expected not in tokens:
        pytest.fail(f"Credential preflight did not identify the {category} example.")


def test_credential_preflight_scans_dot_env_and_credential_inside_comments(tmp_path):
    from ci.secret_redaction import collect_sensitive_values

    skill = _skill(tmp_path)
    token = "F4k" * 16
    (skill / ".env").write_text('# API_KEY = "' + token + '"\n')
    if token not in collect_sensitive_values(skill):
        pytest.fail("Credential preflight did not scan a dotenv comment.")


def test_optional_provider_key_is_redacted_from_scanner_output(tmp_path, capsys):
    workspace, reports = tmp_path / "workspace", tmp_path / "reports"
    skill = _skill(workspace)
    (skill / "source.py").write_text("value = 1\n")
    provider_key = "provider-" + "Z8b" * 16

    def fake_cli(argv, **kwargs):
        assert kwargs["env"]["OPENAI_API_KEY"] == provider_key
        kwargs["stdout"].write(("provider diagnostic " + provider_key).encode())
        result = SecurityValidator(use_llm=False).validate_pii_only(skill)
        result.validator_name = "PII Scan"
        result.validator_description = "Detect sensitive identifiers"
        result.add_message("provider diagnostic " + provider_key)
        target = Path(argv[argv.index("-o") + 1])
        (target / "skillevaluator-tier1.json").write_text(JSONReporter(include_timestamp=False).render_all([result]))
        return subprocess.CompletedProcess(argv, 0)

    environment = {
        "SKILL_PATHS": "skills",
        "LLM_ENABLED": "true",
        "LLM_PROVIDER": "openai",
        "LLM_API_KEY": provider_key,
    }
    assert runner.run(workspace, reports, environment, execute=fake_cli) == 0
    assert provider_key not in capsys.readouterr().out
    for path in reports.rglob("*"):
        if path.is_file() and provider_key.encode() in path.read_bytes():
            pytest.fail("A synthetic provider key reached a report artifact.")


def test_preflight_rejects_extremely_long_lines_before_scanning(tmp_path, capsys):
    workspace, reports = tmp_path / "workspace", tmp_path / "reports"
    skill = _skill(workspace)
    (skill / "source.py").write_text("connection" * 2000 + _fixture_value() + "\n")
    called = []

    def execute(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("Unsafe input must stop before the scanner starts")

    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills"}, execute=execute) == 1
    assert not called
    _assert_no_secret(capsys.readouterr().out)
    _assert_no_secret((reports / "summary.json").read_bytes())


def test_database_password_is_collected_in_decoded_form():
    from skillevaluator.utils.secret_redaction import sensitive_values

    matched = "postgres://user:encoded%2Fpassword@host.local/db"
    assert "encoded/password" in sensitive_values("database_credentials", matched)


def _assert_no_fragment(value: str | bytes, secret: str) -> None:
    text = value.decode() if isinstance(value, bytes) else value
    # Upstream text can start, end or be escaped inside a value. No window of
    # eight characters may remain in any encoding that a report uses.
    for start in range(len(secret) - 7):
        window = secret[start : start + 8]
        forms = {
            window,
            html.escape(window),
            json.dumps(window)[1:-1],
            quote(window, safe=""),
            html.escape(window, quote=False).replace("|", "&#124;").replace("`", "&#96;"),
        }
        if any(form in text for form in forms):
            pytest.fail("A fragment of a synthetic credential was present in a shared output.")


def _publish(tmp_path: Path, monkeypatch, reports: Path) -> tuple[Path, Path]:
    spec = importlib.util.spec_from_file_location("publisher_fixtures", ROOT / "tests/ci/test_publish_reports.py")
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in reports.rglob("*"):
            if path.is_file():
                zipped.write(path, path.relative_to(reports).as_posix())

    class ArtifactAPI(fixtures.FakeAPI):
        def download_artifact(self, artifact_id):
            if artifact_id == 457:
                return super().download_artifact(artifact_id)
            return archive.getvalue()

    monkeypatch.setenv("TRUSTED_EVALUATOR_SHA", "c" * 40)
    monkeypatch.setenv("TRUSTED_CALLER_BLOBS", "b" * 40)
    monkeypatch.syspath_prepend(str(ROOT / "ci"))
    import verify_provenance

    monkeypatch.setattr(verify_provenance, "fetch_signing_keys", lambda: fixtures.JWKS)
    site, feedback = tmp_path / "site", tmp_path / "feedback"
    fixtures.publisher.publish(
        ArtifactAPI(record={**fixtures.run_record(), "conclusion": "failure"}),
        "owner/consumer",
        123,
        "https://owner.github.io/consumer",
        site,
        feedback,
    )
    return site, feedback


def test_redactor_masks_values_that_upstream_text_cuts_or_escapes():
    from skillevaluator.utils.secret_redaction import Redactor

    value = "nvapi-" + "Qx7" * 16
    password = "Rt5" * 4 + "|" + "Wm8`" * 3
    redactor = Redactor({value, password})
    cell = html.escape(password, quote=False).replace("|", "&#124;").replace("`", "&#96;")
    for text in (
        # A context window that starts inside a value.
        "window: " + value[3:] + '"\nWEBHOOK = "https://hooks.example.invalid/x"',
        # A window that starts inside a value and is then cut again.
        "excerpt " + value[9:41] + "...",
        json.dumps({"snippet": value[13:]}),
        "| " + cell + " |",
    ):
        safe = redactor.text(text)
        _assert_no_fragment(safe, value)
        _assert_no_fragment(safe, password)
    ordinary = "Dangerous code execution: subprocess.run(cmd, shell=True) in scripts/run.py"
    assert redactor.text(ordinary) == ordinary


def test_redactor_masks_database_userinfo_when_the_text_ends_inside_it():
    from skillevaluator.utils.secret_redaction import Redactor

    password = "Kp4" * 4
    for redactor in (Redactor({password}), Redactor(set())):
        safe = redactor.text('    url = "postgresql://svc:' + password[:10])
        if password[:6] in safe or "postgresql://" not in safe:
            pytest.fail("A cut database password was not masked.")
    ordinary = [
        "postgresql://db.local:5432/app",
        'DSN = "postgresql://db.local:5432"',
        "mysql://reporting@db.local/app",
        "https://user.example.invalid:8443/path",
    ]
    assert [Redactor(set()).text(text) for text in ordinary] == ordinary
    # The connection string match is not a secret; its words stay readable.
    connection = Redactor({'conn_string = "Server=db.local;password='})
    assert connection.text("Hardcoded password in a connection string") == "Hardcoded password in a connection string"


def test_redactor_masks_a_short_database_password_when_the_text_starts_inside_it(tmp_path):
    from ci.secret_redaction import collect_sensitive_values

    from skillevaluator.utils.secret_redaction import Redactor

    skill = _skill(tmp_path)
    password = "Gh7" * 4 + "Gh"
    (skill / "source.py").write_text('DATABASE_URL = "postgresql://reporting:' + password + '@db.local/app"\n')
    redactor = Redactor(collect_sensitive_values(skill))
    for start in range(1, len(password) - 7):
        # A context window that starts inside the password cuts off the scheme.
        safe = redactor.text(password[start:] + '@db.local/app"\nWEBHOOK = "https://hooks.example.invalid/x"')
        _assert_no_fragment(safe, password)
        assert "db.local/app" in safe
    ordinary = "Send the export to reporting@db.local after the run."
    assert redactor.text(ordinary) == ordinary


def test_retained_markdown_and_sarif_are_rendered_from_redacted_json(tmp_path):
    workspace, reports = tmp_path / "workspace", tmp_path / "reports"
    _skill(workspace)
    value = _fixture_value()

    def fake_cli(argv, **kwargs):
        target = Path(argv[argv.index("-o") + 1])
        result = SecurityValidator(use_llm=False).validate_pii_only(Path(argv[2]))
        result.validator_name = "PII Scan"
        result.validator_description = "Detect sensitive identifiers"
        payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
        # Another scanner's context window started inside the value.
        payload["results"][0]["findings"][0]["line_content"] = value[3:]
        (target / "skillevaluator-tier1.json").write_text(json.dumps(payload))
        # The evaluator's own files escape and cut text in ways a literal
        # search cannot follow.
        (target / "skillevaluator-tier1.md").write_text("Raw evaluator Markdown " + value.replace("-", "\\-")[:30])
        (target / "skillevaluator-tier1.sarif.json").write_text(json.dumps({"raw": "Raw evaluator SARIF"}))
        return subprocess.CompletedProcess(argv, 1)

    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills"}, execute=fake_cli) == 1
    markdown = (reports / "skill-001/skillevaluator-tier1.md").read_text()
    sarif = json.loads((reports / "skill-001/skillevaluator-tier1.sarif.json").read_text())
    if "Raw evaluator" in markdown or not markdown.startswith("# SkillEvaluator Validation Report"):
        pytest.fail("Retained Markdown was not rendered from the redacted JSON.")
    assert sarif["version"] == "2.1.0"
    assert [item["properties"]["checkName"] for item in sarif["runs"][0]["results"]] == ["hardcoded_secrets"]
    for path in reports.rglob("*"):
        if path.is_file():
            _assert_no_fragment(path.read_bytes(), value)


def test_preflight_collects_unquoted_nvidia_key_and_bearer_values(tmp_path):
    from ci.secret_redaction import collect_sensitive_values

    skill = _skill(tmp_path)
    key = "nvapi-" + "kl" * 24
    bearer = "Tk9" * 8
    (skill / "source.py").write_text("value = 1\n")
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A sample formatting skill.\n---\n# Demo\n\n"
        "```bash\nexport NVIDIA_API_KEY=" + key + "\n"
        "curl -H 'Authorization: Bearer " + bearer + "' https://api.example.invalid\n```\n"
    )
    tokens = collect_sensitive_values(skill)
    if key not in tokens or bearer not in tokens:
        pytest.fail("Credential preflight did not identify an unquoted credential.")


def test_preflight_does_not_collect_ordinary_words_after_credential_names(tmp_path):
    from ci.secret_redaction import collect_sensitive_values

    skill = _skill(tmp_path)
    (skill / "source.py").write_text("value = 1\n")
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A sample formatting skill.\n---\n# Demo\n\n"
        "The password: required for the staging environment.\n"
        "Send the header Authorization: Bearer token_from_the_environment.\n"
        "export NVIDIA_API_KEY=$NVIDIA_API_KEY_FROM_THE_VAULT\n"
        "api_key = load_configuration_value(name)\n"
        "API_KEY_FILE=/run/secrets/provider_key_v2\n"
        # A long encoded blob is not a key; collecting it would stop the scan.
        "api_key: " + "Ab1" * 200 + "\n"
    )
    assert collect_sensitive_values(skill) == set()


@pytest.mark.parametrize("mark", ["\u200b", "\ufeff", "\u00a0"])
def test_preflight_matches_values_echoed_with_an_invisible_character_marker(tmp_path, mark):
    from ci.secret_redaction import collect_sensitive_values

    from skillevaluator.utils.secret_redaction import Redactor
    from skillevaluator.validators.unicode_smuggle import UnicodeSmuggleValidator

    skill = _skill(tmp_path)
    leading, middle = "nvapi-" + "Hj6" * 16, "Lm3" * 4 + "Pq2" * 6
    short = "Kp4" * 4
    lines = [
        'NVIDIA_API_KEY = "' + mark + leading + '"',
        'DB_PASSWORD = "' + middle[:12] + mark + middle[12:] + '"',
        # A short password is matched only as a whole value.
        'conn_string = "Server=db.local;password=' + mark + short + ';"',
    ]
    (skill / "source.py").write_text("\n".join(lines) + "\n")
    redactor = Redactor(collect_sensitive_values(skill))
    for line in lines:
        # The Unicode check echoes the line with the character shown as a marker.
        safe = redactor.text(UnicodeSmuggleValidator._safe_line_content(line))
        _assert_no_fragment(safe, leading)
        _assert_no_fragment(safe, middle)
        _assert_no_fragment(safe, short)


@pytest.mark.parametrize("separator", ["\u2028", "\u0085", "\x0c"])
def test_preflight_splits_lines_as_the_scanners_do(tmp_path, separator):
    from ci.secret_redaction import collect_sensitive_values

    from skillevaluator.utils.secret_redaction import Redactor
    from skillevaluator.validators.unicode_smuggle import UnicodeSmuggleValidator

    skill = _skill(tmp_path)
    value = "Hj6" * 6 + "Lm3" * 6
    line = 'API_SECRET = "' + value[:10] + separator + value[10:] + '"'
    (skill / "source.py").write_text(line + "\n", encoding="utf-8")
    # The PII scan splits only on newlines, so it still flags the value.
    result = SecurityValidator(use_llm=False).validate_pii_only(skill)
    assert any(f.check_name == "hardcoded_secrets" for f in result.findings)
    tokens = collect_sensitive_values(skill)
    if value[:10] + separator + value[10:] not in tokens:
        pytest.fail("Credential preflight split a quoted value at a line separator.")
    redactor = Redactor(tokens)
    # YARA echoes the raw line. The Unicode check shows the character as a marker.
    for echo in (line, UnicodeSmuggleValidator._safe_line_content(line)):
        _assert_no_fragment(redactor.text(echo), value)


def test_invisible_character_in_credential_never_reaches_reports_or_pages(tmp_path, capsys, monkeypatch):
    from skillevaluator.validators.unicode_smuggle import UnicodeSmuggleValidator

    workspace, reports = tmp_path / "workspace", tmp_path / "reports"
    skill = _skill(workspace)
    value = "nvapi-" + "Hj6" * 16
    (skill / "source.py").write_text('NVIDIA_API_KEY = "\u200b' + value + '"\n')

    def fake_cli(argv, **kwargs):
        target = Path(argv[argv.index("-o") + 1])
        results = [
            SecurityValidator(use_llm=False).validate_pii_only(Path(argv[2])),
            UnicodeSmuggleValidator().validate(Path(argv[2])),
        ]
        for result, name in zip(results, ("PII Scan", "Unicode Smuggling"), strict=True):
            result.validator_name = name
            result.validator_description = name
        assert any(finding.line_content for finding in results[1].findings)
        (target / "skillevaluator-tier1.json").write_text(JSONReporter(include_timestamp=False).render_all(results))
        return subprocess.CompletedProcess(argv, 1)

    environment = {"SKILL_PATHS": "skills", "GITHUB_ACTIONS": "true"}
    assert runner.run(workspace, reports, environment, execute=fake_cli) == 1
    _assert_no_fragment(capsys.readouterr().out, value)
    site, feedback = _publish(tmp_path, monkeypatch, reports)
    for directory in (reports, site, feedback):
        for path in directory.rglob("*"):
            if path.is_file():
                _assert_no_fragment(path.read_bytes(), value)
    assert (site / "runs/123/1/skill-001/skillevaluator-tier1.html").is_file()


@pytest.mark.parametrize(
    ("target", "path_override", "published"),
    [
        ("/home/runner/work/private/skills/demo", None, None),
        (r"C:\runner\work\private\skills\demo", None, None),
        ("C:/runner/work/private/skills/demo", None, None),
        (r"\\server\private\skills\demo", None, None),
        ("skills/demo", None, "skills/demo"),
        (".", None, "."),
        ("https://github.com/example/skills/tree/main/demo", None, "https://github.com/example/skills/tree/main/demo"),
        (None, None, None),
        ("/home/runner/work/private/skills/demo", "skills/demo", "skills/demo"),
        ("skills/demo", "skills/public-demo", "skills/public-demo"),
        ("/home/runner/work/private/skills/demo", "/private/skills/demo", None),
    ],
)
def test_published_json_omits_absolute_host_target_without_changing_evidence(
    tmp_path, target, path_override, published
):
    result = ValidationResult(validator_name="Schema", validator_description="Check skill manifest")
    result.add_success("manifest_exists", "Found SKILL.md")
    result.metadata["gating"] = {"tier": 1, "blocking": True}
    raw_document = json.loads(
        JSONReporter(include_timestamp=False, target_path=target, explicit_skill_name="demo").render_all([result])
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "skillevaluator-tier1.json").write_text(json.dumps(raw_document), encoding="utf-8")

    outputs = runner.sanitized_outputs(raw, set(), target_path=path_override)
    document = json.loads(outputs["skillevaluator-tier1.json"])

    assert raw_document["target_path"] == target
    assert document["target_path"] == published
    for field in ("skills", "gating", "version", "results"):
        assert document[field] == raw_document[field]


def test_published_json_redacts_known_credentials_in_repository_relative_target_override(tmp_path):
    token = "example-secret-0123456789"
    result = ValidationResult(validator_name="Schema", validator_description="Check skill manifest")
    result.add_success("manifest_exists", "Found SKILL.md")
    raw_document = JSONReporter(target_path="/runner/private/skills/demo", explicit_skill_name="demo").render_all(
        [result]
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "skillevaluator-tier1.json").write_text(raw_document, encoding="utf-8")

    outputs = runner.sanitized_outputs(raw, {token}, target_path=f"skills/{token}/demo")
    document = json.loads(outputs["skillevaluator-tier1.json"])

    assert document["target_path"] == "skills/[REDACTED]/demo"
    assert token.encode() not in outputs["skillevaluator-tier1.json"]
    assert (raw / "skillevaluator-tier1.json").read_text(encoding="utf-8") == raw_document
