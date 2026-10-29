# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential findings must not copy detected values into shared CI evidence."""

from __future__ import annotations

import html
import importlib.util
import io
import json
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import quote

import pytest

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

    spec = importlib.util.spec_from_file_location("publisher_fixtures", ROOT / "tests/ci/test_publish_reports.py")
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)

    class ArtifactAPI(fixtures.FakeAPI):
        def download_artifact(self, _artifact_id):
            return old_archive.getvalue()

    monkeypatch.setenv("TRUSTED_EVALUATOR_SHA", "c" * 40)
    monkeypatch.setenv("TRUSTED_CALLER_BLOBS", "b" * 40)
    monkeypatch.syspath_prepend(str(ROOT / "ci"))
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
        def download_artifact(self, _artifact_id):
            return archive.getvalue()

    monkeypatch.setenv("TRUSTED_EVALUATOR_SHA", "c" * 40)
    monkeypatch.setenv("TRUSTED_CALLER_BLOBS", "b" * 40)
    monkeypatch.syspath_prepend(str(ROOT / "ci"))
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
