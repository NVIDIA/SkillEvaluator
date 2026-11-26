# SPDX-License-Identifier: Apache-2.0
"""Guard selection wiring and execute the consumer shell's Docker contract."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting.sarif_reporter import SARIFReporter

ROOT = Path(__file__).parents[2]


def _workflow():
    return yaml.load((ROOT / ".github/workflows/evaluate-skills.yml").read_text(), Loader=yaml.BaseLoader)


def _job():
    return _workflow()["jobs"]["tier1"]


def test_pr_history_mode_and_attempt_artifact_wiring():
    job = _job()
    checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    scan = next(step for step in job["steps"] if step.get("name") == "Evaluate selected skills")
    env = scan["env"]
    # Steps inherit the job's scan mode, which also sizes the checkout.
    assert "SCAN_MODE" not in env
    assert (
        job["env"]["SCAN_MODE"]
        == "${{ inputs.scan-mode == 'auto' && (github.event_name == 'pull_request' && 'changed' || 'all') || inputs.scan-mode }}"
    )
    # Only changed mode fetches history, and it skips old file contents.
    # The numbers are ordered so that 0, which is falsy, is never the && operand.
    assert (
        checkout["with"]["fetch-depth"] == "${{ !inputs.security-baselines && env.SCAN_MODE != 'changed' && 1 || 0 }}"
    )
    assert (
        checkout["with"]["filter"]
        == "${{ !inputs.security-baselines && env.SCAN_MODE == 'changed' && 'blob:none' || '' }}"
    )
    assert checkout["with"]["ref"] == env["DIFF_HEAD_SHA"] == "${{ github.event.pull_request.head.sha || github.sha }}"
    assert env["DIFF_BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    assert env["POLICY_SOURCE_SHA"] == "${{ github.event.pull_request.base.sha || github.sha }}"
    assert env["SECURITY_BASELINES"] == "${{ inputs.security-baselines }}"
    assert env["LLM_API_KEY"] == "${{ inputs.llm && secrets.LLM_API_KEY || '' }}"
    assert env["CALLER_WORKFLOW_SHA"] == "${{ github.workflow_sha }}"
    assert env["SCAN_WORKERS"] == "${{ inputs.scan-workers }}"
    upload = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["with"]["name"] == "${{ inputs.artifact-name }}-attempt-${{ github.run_attempt }}"
    assert int(job["timeout-minutes"]) > 50


def test_enforcement_mode_defaults_to_strict():
    workflow = _workflow()
    mode = workflow["on"]["workflow_call"]["inputs"]["enforcement-mode"]
    assert (mode["type"], mode["default"]) == ("string", "strict")
    validate = next(step for step in _job()["steps"] if step.get("name") == "Validate enforcement mode")
    assert validate["env"]["ENFORCEMENT_MODE"] == "${{ inputs.enforcement-mode }}"
    scan = next(step for step in _job()["steps"] if step.get("name") == "Evaluate selected skills")
    assert scan.get("continue-on-error", "false") == "false"


@pytest.mark.parametrize("mode,expected", [("strict", 0), ("advisory", 0), ("bogus", 1), ("Advisory", 1), ("", 1)])
def test_enforcement_mode_validation_executes_before_setup(tmp_path, mode, expected):
    step = _job()["steps"][0]
    assert step["name"] == "Validate enforcement mode"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={**os.environ, "ENFORCEMENT_MODE": mode},
        capture_output=True,
    )
    assert result.returncode == expected


@pytest.mark.parametrize("mode", ["strict", "advisory"])
@pytest.mark.parametrize("exit_code", [1, 125, 126, 127, 137])
def test_scan_shell_preserves_container_failures(tmp_path, mode, exit_code):
    binary = tmp_path / "docker"
    binary.write_text(f"#!/bin/sh\nexit {exit_code}\n")
    binary.chmod(0o755)
    scan = next(step for step in _job()["steps"] if step.get("name") == "Evaluate selected skills")
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", scan["run"]],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "GITHUB_WORKSPACE": str(tmp_path / "checkout"),
            "IMAGE": "ghcr.io/example/scanner@sha256:" + "a" * 64,
            "REPORT_BASE_URL": "",
            "LLM_ENABLED": "false",
            "ENFORCEMENT_MODE": mode,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code
    assert (tmp_path / "output").read_text() == f"exit-code={exit_code}\n"


@pytest.mark.parametrize("mode", ["strict", "advisory"])
@pytest.mark.parametrize("llm", ["true", "false"])
def test_real_workflow_shell_preserves_container_boundary_and_secret_selection(tmp_path, llm, mode):
    scan = next(step for step in _job()["steps"] if step.get("name") == "Evaluate selected skills")
    binary = tmp_path / "docker"
    # Test double captures argv of the actual workflow shell, not a reimplementation.
    binary.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ['CAPTURE'],'w').write(json.dumps(sys.argv[1:]))\n"
    )
    binary.chmod(0o755)
    captured = tmp_path / "argv.json"
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "CAPTURE": str(captured),
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_WORKSPACE": str(tmp_path / "checkout"),
        "IMAGE": "ghcr.io/example/scanner@sha256:" + "a" * 64,
        "REPORT_BASE_URL": "",
        "LLM_ENABLED": llm,
        "ENFORCEMENT_MODE": mode,
        "LLM_API_KEY": "dummy-do-not-pass-in-static-mode",
    }
    subprocess.run(["bash", "-e", "-o", "pipefail", "-c", scan["run"]], env=env, check=True, capture_output=True)
    args = json.loads(captured.read_text())
    image_index = args.index(env["IMAGE"])
    assert args[image_index + 1 :] == (["--enforcement-mode", "advisory"] if mode == "advisory" else [])
    passed_environment = [args[index + 1] for index, arg in enumerate(args) if arg == "--env"]
    assert {
        "SCAN_MODE",
        "DIFF_BASE_SHA",
        "DIFF_HEAD_SHA",
        "MIN_SCORE",
        "SCAN_WORKERS",
        "POLICY_SOURCE_SHA",
        "SECURITY_BASELINES",
    }.issubset(passed_environment)
    assert ("LLM_API_KEY" in args) is (llm == "true")
    for flag in ("--read-only", "--user", "--cap-drop", "--security-opt"):
        assert flag in args
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert args[args.index("--security-opt") + 1] == "no-new-privileges"
    assert any("dst=/workspace,readonly" in arg for arg in args)
    assert args[args.index("SCAN_WORKERS") - 1] == "--env"
    mounts = [args[index + 1] for index, arg in enumerate(args) if arg == "--tmpfs"]
    assert len(mounts) == 2 and all("noexec" in mount for mount in mounts)
    harness = (ROOT / "ci/smoke_image.py").read_text()
    for mount in mounts:
        assert mount in harness


def test_smoke_container_executes_with_the_same_hardening(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("ci_smoke_contract", ROOT / "ci/smoke_image.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    calls = []

    def capture(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(smoke.subprocess, "run", capture)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert smoke._container("example:local", workspace, tmp_path / "case", "changed-one", "a" * 40, "b" * 40) == 0
    args = calls[0]
    assert "--read-only" in args
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert args[args.index("--security-opt") + 1] == "no-new-privileges"
    environment = [args[index + 1] for index, arg in enumerate(args) if arg == "--env"]
    assert {entry.split("=", 1)[0] for entry in environment}.issuperset({"SCAN_MODE", "DIFF_BASE_SHA", "DIFF_HEAD_SHA"})


def test_quality_threshold_is_wired_into_the_container():
    workflow = yaml.load((ROOT / ".github/workflows/evaluate-skills.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["on"]["workflow_call"]["inputs"]["min-score"]["default"] == "70"
    scan = next(step for step in workflow["jobs"]["tier1"]["steps"] if step.get("name") == "Evaluate selected skills")
    assert scan["env"]["MIN_SCORE"] == "${{ inputs.min-score }}"
    assert "--env MIN_SCORE" in scan["run"]


def test_image_validation_does_not_import_consumer_python_modules(tmp_path):
    step = next(step for step in _job()["steps"] if step.get("name") == "Validate the immutable image reference")
    (tmp_path / "re.py").write_text("raise RuntimeError('consumer module executed on the runner')\n")
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={**os.environ, "IMAGE": "ghcr.io/example/scanner@sha256:" + "a" * 64},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_basic_scan_does_not_require_identity_credentials(tmp_path):
    workflow = yaml.load((ROOT / ".github/workflows/evaluate-skills.yml").read_text(), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["tier1"]
    assert "permissions" not in workflow and "permissions" not in job
    step = next(step for step in job["steps"] if step.get("id") == "provenance")
    assert step["id"] == "provenance"
    # Run the actual trusted step from a consumer checkout containing shadow modules.
    for module in ("re", "json", "tempfile"):
        (tmp_path / f"{module}.py").write_text("raise RuntimeError('consumer module executed')\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith("ACTIONS_ID_TOKEN_REQUEST_")}
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "scan artifacts remain available" in result.stdout


def test_signed_receipt_is_separate_and_retained_after_a_failed_scan():
    steps = _job()["steps"]
    capture = next(step for step in steps if step.get("id") == "provenance")
    assert capture["continue-on-error"] == "true"
    upload = next(step for step in steps if step.get("name") == "Retain the separate hosted-report execution receipt")
    assert "always()" in upload["if"]
    assert upload["with"]["name"] == "${{ inputs.artifact-name }}-provenance-attempt-${{ github.run_attempt }}"
    assert upload["with"]["path"] == "${{ steps.provenance.outputs.directory }}/execution.jwt"
    scan = next(step for step in steps if step.get("name") == "Evaluate selected skills")
    assert "provenance" not in scan["run"]
    assert "ID_TOKEN" not in scan["run"]


def test_scan_worker_input_defaults_to_four_parallel_scans():
    workers = _workflow()["on"]["workflow_call"]["inputs"]["scan-workers"]
    assert (workers["type"], workers["default"]) == ("number", "4")


def _sarif_step():
    return next(step for step in _job()["steps"] if step.get("id") == "sarif")


def _sarif_document(rule):
    return {
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "SkillEvaluator", "rules": [{"id": rule}]}},
            "results": [{"ruleId": rule, "message": {"text": "finding"}}],
            "invocations": [{"executionSuccessful": True}],
        }],
    }


def _sarif_evidence(tmp_path, count=2):
    reports = tmp_path / "reports"
    reports.mkdir()
    rows = []
    for index in range(1, count + 1):
        name = f"skill-{index:03d}"
        directory = reports / name
        directory.mkdir()
        (directory / "skillevaluator-tier1.sarif.json").write_text(json.dumps(_sarif_document(f"rule-{index}")))
        rows.append({
            "path": f"skills/{index}", "report_dir": name, "overall_status": "passed",
            "incomplete_scans": [], "exit_code": 0,
        })
    summary = {
        "total": count, "skills": rows,
        "selection": {"mode": "all", "discovered": count, "selected": [row["path"] for row in rows]},
    }
    (reports / "summary.json").write_text(json.dumps(summary))
    output = tmp_path / "output"
    output.mkdir()
    return reports, output, summary


def _prepare_sarif(reports, output):
    script = _sarif_step()["run"].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    return subprocess.run(
        [sys.executable, "-c", script, str(reports), str(output)],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True,
    )


def _run_sarif_step_with_local_docker(tmp_path):
    binary = tmp_path / "docker"
    binary.write_text(
        '#!/bin/sh\nexec "$TEST_PYTHON" - "$RUNNER_TEMP/skill-reports" "$RUNNER_TEMP/skill-sarif"\n'
    )
    binary.chmod(0o755)
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _sarif_step()["run"]],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(ROOT / "src"),
            "TEST_PYTHON": sys.executable,
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(tmp_path / "github-output"),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "step-summary"),
            "IMAGE": "test-image",
            "SCAN_MODE": "all",
            "SCAN_EXIT_CODE": "0",
            "SARIF_CATEGORY": "catalog",
            "EVENT_NAME": "push",
            "FORK_PR": "false",
        },
        capture_output=True,
        text=True,
    )


def test_sarif_upload_is_opt_in_and_failures_are_not_suppressed():
    inputs = _workflow()["on"]["workflow_call"]["inputs"]
    assert inputs["upload-sarif"]["default"] == "false"
    assert inputs["sarif-category"]["default"] == "skillevaluator"
    prepare = _sarif_step()
    assert "always()" in prepare["if"] and "inputs.upload-sarif" in prepare["if"]
    scan = next(step for step in _job()["steps"] if step.get("name") == "Evaluate selected skills")
    assert scan["id"] == "scan"
    assert prepare["env"]["SCAN_EXIT_CODE"] == "${{ steps.scan.outputs.exit-code }}"
    upload = next(step for step in _job()["steps"] if step.get("name") == "Upload SARIF to GitHub Code Scanning")
    assert "always()" in upload["if"] and "steps.sarif.outputs.ready == 'true'" in upload["if"]
    assert upload.get("continue-on-error", "false") == "false"
    assert upload["with"]["category"] == "${{ inputs.sarif-category }}"
    assert "refs/pull/{0}/head" in upload["with"]["ref"]
    assert upload["with"]["sha"] == "${{ github.event.pull_request.head.sha || github.sha }}"
    assert "permissions" not in _job()


def test_complete_reports_are_merged_even_with_a_failed_verdict(tmp_path):
    reports, output, summary = _sarif_evidence(tmp_path)
    summary["skills"][0].update(overall_status="failed", exit_code=1)
    (reports / "summary.json").write_text(json.dumps(summary))
    result = _prepare_sarif(reports, output)
    assert result.returncode == 0, result.stderr
    document = json.loads((output / "catalog.sarif.json").read_text())
    assert len(document["runs"]) == 1
    run = document["runs"][0]
    assert "automationDetails" not in run
    assert [item["ruleId"] for item in run["results"]] == ["rule-1", "rule-2"]
    assert [item["id"] for item in run["tool"]["driver"]["rules"]] == ["rule-1", "rule-2"]


@pytest.mark.parametrize(
    "result_count,warning",
    [
        (5_000, None),
        (5_001, "GitHub displays only the 5,000 highest-priority results"),
        (25_000, "GitHub displays only the 5,000 highest-priority results"),
        (25_001, "GitHub rejects SARIF runs over 25,000 results"),
    ],
)
def test_large_sarif_catalog_warns_but_remains_uploadable(tmp_path, result_count, warning):
    reports, output, _ = _sarif_evidence(tmp_path, 1)
    child = reports / "skill-001/skillevaluator-tier1.sarif.json"
    document = json.loads(child.read_text())
    finding = document["runs"][0]["results"][0]
    document["runs"][0]["results"] = [finding.copy() for _ in range(result_count)]
    child.write_text(json.dumps(document))

    result = _prepare_sarif(reports, output)

    assert result.returncode == 0, result.stderr
    if warning is None:
        assert "::warning::" not in result.stdout
    else:
        assert warning in result.stdout
    merged = json.loads((output / "catalog.sarif.json").read_text())
    assert len(merged["runs"][0]["results"]) == result_count


@pytest.mark.parametrize(
    "uri,expected",
    [("file:///workspace/skills/1/SKILL.md", "skills/1/SKILL.md"), ("SKILL.md", "skills/1/SKILL.md")],
)
def test_sarif_locations_resolve_to_the_scanned_workspace(tmp_path, uri, expected):
    reports, output, _ = _sarif_evidence(tmp_path, 1)
    child = reports / "skill-001/skillevaluator-tier1.sarif.json"
    document = json.loads(child.read_text())
    document["runs"][0]["results"][0]["locations"] = [{
        "physicalLocation": {"artifactLocation": {"uri": uri}},
    }]
    child.write_text(json.dumps(document))
    result = _prepare_sarif(reports, output)
    assert result.returncode == 0, result.stderr
    merged = json.loads((output / "catalog.sarif.json").read_text())
    location = merged["runs"][0]["results"][0]["locations"][0]
    assert location["physicalLocation"]["artifactLocation"]["uri"] == expected


def test_relative_sarif_location_resolves_under_skill_path(tmp_path):
    reports, output, _ = _sarif_evidence(tmp_path, 1)
    child = reports / "skill-001/skillevaluator-tier1.sarif.json"
    document = json.loads(child.read_text())
    document["runs"][0]["results"][0]["locations"] = [{
        "physicalLocation": {"artifactLocation": {"uri": "README.md"}},
    }]
    child.write_text(json.dumps(document))
    result = _prepare_sarif(reports, output)
    assert result.returncode == 0, result.stderr
    merged = json.loads((output / "catalog.sarif.json").read_text())
    assert merged["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == (
        "skills/1/README.md"
    )


def test_skill_relative_path_starting_with_skill_name_stays_relative_to_skill(tmp_path):
    reports, output, _ = _sarif_evidence(tmp_path, 1)
    finding = Finding(
        category="SCHEMA", severity=Severity.LOW, check_name="fixture",
        message="Example finding", file_path="skills/1/README.md",
    )
    validation = ValidationResult(validator_name="Schema", validator_description="Fixture validation")
    validation.add_finding(finding)
    (reports / "skill-001/skillevaluator-tier1.sarif.json").write_text(SARIFReporter().render_all([validation]))

    result = _prepare_sarif(reports, output)

    assert result.returncode == 0, result.stderr
    merged = json.loads((output / "catalog.sarif.json").read_text())
    uri = merged["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    assert uri == "skills/1/skills/1/README.md"


@pytest.mark.parametrize(
    "source,expected",
    [("README.md", "skills/1/README.md"), ("/workspace/skills/1/SKILL.md", "skills/1/SKILL.md")],
)
def test_preparation_accepts_real_redacted_reporter_output(tmp_path, source, expected):
    reports, output, _ = _sarif_evidence(tmp_path, 1)
    finding = Finding(
        category="SCHEMA", severity=Severity.LOW, check_name="fixture",
        message="Example finding", file_path=source,
    )
    validation = ValidationResult(validator_name="Schema", validator_description="Fixture validation")
    validation.add_finding(finding)
    child = reports / "skill-001/skillevaluator-tier1.sarif.json"
    report = SARIFReporter().render_all([validation])
    raw_document = json.loads(report)
    raw_run = raw_document["runs"][0]
    assert raw_run["invocations"][0]["executionSuccessful"] is True
    raw_uri = raw_run["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    if source.startswith("/workspace/"):
        assert raw_uri == f"file://{source}"
    child.write_text(report)
    result = _prepare_sarif(reports, output)
    assert result.returncode == 0, result.stderr
    merged = json.loads((output / "catalog.sarif.json").read_text())
    run = merged["runs"][0]
    assert run["invocations"][0]["executionSuccessful"] is True
    assert run["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == expected


@pytest.mark.parametrize("uri", ["../outside", "file:///workspace/../outside", "https://example.com/file"])
def test_unsafe_sarif_locations_do_not_publish(tmp_path, uri):
    reports, output, _ = _sarif_evidence(tmp_path, 1)
    child = reports / "skill-001/skillevaluator-tier1.sarif.json"
    document = json.loads(child.read_text())
    document["runs"][0]["results"][0]["locations"] = [{
        "physicalLocation": {"artifactLocation": {"uri": uri}},
    }]
    child.write_text(json.dumps(document))
    result = _prepare_sarif(reports, output)
    assert result.returncode != 0
    assert not (output / "catalog.sarif.json").exists()


def test_external_sarif_location_cannot_be_attributed_to_the_workspace(tmp_path):
    reports, output, _ = _sarif_evidence(tmp_path, 1)
    child = reports / "skill-001/skillevaluator-tier1.sarif.json"
    document = json.loads(child.read_text())
    document["runs"][0]["results"][0]["locations"] = [{
        "physicalLocation": {"artifactLocation": {"uri": "file:///tmp/other"}},
    }]
    child.write_text(json.dumps(document))
    result = _prepare_sarif(reports, output)
    assert result.returncode != 0
    assert "outside the workspace" in result.stderr


@pytest.mark.parametrize(
    "locations",
    [
        ["not-an-object"],
        [{"physicalLocation": "not-an-object"}],
        [{"physicalLocation": None}],
        [{"physicalLocation": {"artifactLocation": "not-an-object"}}],
        [{"physicalLocation": {"artifactLocation": {"uri": 42}}}],
        {"not": "a-list"},
    ],
)
def test_malformed_sarif_locations_fail_with_clear_error(tmp_path, locations):
    reports, output, _ = _sarif_evidence(tmp_path, 1)
    child = reports / "skill-001/skillevaluator-tier1.sarif.json"
    document = json.loads(child.read_text())
    document["runs"][0]["results"][0]["locations"] = locations
    child.write_text(json.dumps(document))

    result = _prepare_sarif(reports, output)

    assert result.returncode != 0
    assert "Code Scanning upload skipped: missing, malformed, or incomplete scan evidence" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (output / "catalog.sarif.json").exists()


def test_completed_preparation_sets_ready_for_upload(tmp_path):
    reports, _, _ = _sarif_evidence(tmp_path, 1)
    reports.rename(tmp_path / "skill-reports")

    result = _run_sarif_step_with_local_docker(tmp_path)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "github-output").read_text() == "ready=true\n"
    assert (tmp_path / "skill-sarif/catalog.sarif.json").is_file()


def test_failed_preparation_does_not_set_ready_for_upload(tmp_path):
    reports, _, _ = _sarif_evidence(tmp_path, 1)
    (reports / "skill-001/skillevaluator-tier1.sarif.json").write_text("{")
    reports.rename(tmp_path / "skill-reports")

    result = _run_sarif_step_with_local_docker(tmp_path)

    assert result.returncode != 0
    assert "Code Scanning upload skipped" in result.stderr
    assert not (tmp_path / "github-output").exists()
    assert not (tmp_path / "skill-sarif/catalog.sarif.json").exists()


def test_catalog_at_display_limit_does_not_add_warning_to_step_summary(tmp_path):
    reports, _, _ = _sarif_evidence(tmp_path, 1)
    child = reports / "skill-001/skillevaluator-tier1.sarif.json"
    document = json.loads(child.read_text())
    finding = document["runs"][0]["results"][0]
    document["runs"][0]["results"] = [finding.copy() for _ in range(5_000)]
    child.write_text(json.dumps(document))
    reports.rename(tmp_path / "skill-reports")

    result = _run_sarif_step_with_local_docker(tmp_path)

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "step-summary").exists()
    assert (tmp_path / "github-output").read_text() == "ready=true\n"


@pytest.mark.parametrize("count", [5_001, 25_001])
def test_large_catalog_warning_is_in_step_summary_without_skipping_upload(tmp_path, count):
    reports, _, _ = _sarif_evidence(tmp_path, 1)
    child = reports / "skill-001/skillevaluator-tier1.sarif.json"
    document = json.loads(child.read_text())
    finding = document["runs"][0]["results"][0]
    document["runs"][0]["results"] = [finding.copy() for _ in range(count)]
    child.write_text(json.dumps(document))
    reports.rename(tmp_path / "skill-reports")

    result = _run_sarif_step_with_local_docker(tmp_path)

    assert result.returncode == 0, result.stderr
    summary = (tmp_path / "step-summary").read_text()
    assert f"{count} results" in summary
    assert "5,000 highest-priority results" in summary
    if count > 25_000:
        assert "rejects SARIF runs over 25,000 results" in summary
    assert (tmp_path / "github-output").read_text() == "ready=true\n"
    merged = json.loads((tmp_path / "skill-sarif/catalog.sarif.json").read_text())
    assert len(merged["runs"][0]["results"]) == count


@pytest.mark.parametrize(
    "case",
    ["missing", "malformed", "bad-shape", "incomplete", "error", "partial", "empty",
     "changed-selection", "symlink", "unsuccessful-invocation"],
)
def test_incomplete_or_invalid_full_scans_do_not_publish(tmp_path, case):
    reports, output, summary = _sarif_evidence(tmp_path)
    child = reports / "skill-002/skillevaluator-tier1.sarif.json"
    if case == "missing":
        child.unlink()
    elif case == "malformed":
        child.write_text("{")
    elif case == "bad-shape":
        child.write_text(json.dumps({"version": "2.1.0", "runs": []}))
    elif case == "incomplete":
        summary["skills"][1]["incomplete_scans"] = ["semgrep"]
    elif case == "error":
        summary["skills"][1]["overall_status"] = "error"
    elif case == "partial":
        summary["selection"]["discovered"] = 3
    elif case == "changed-selection":
        summary["selection"]["mode"] = "changed"
    elif case == "symlink":
        child.unlink()
        child.symlink_to(reports / "skill-001/skillevaluator-tier1.sarif.json")
    elif case == "unsuccessful-invocation":
        document = json.loads(child.read_text())
        document["runs"][0]["invocations"][0]["executionSuccessful"] = False
        child.write_text(json.dumps(document))
    else:
        summary.update(total=0, skills=[])
        summary["selection"].update(discovered=0, selected=[])
    (reports / "summary.json").write_text(json.dumps(summary))
    result = _prepare_sarif(reports, output)
    assert result.returncode != 0
    assert "Code Scanning upload skipped" in result.stderr
    assert not (output / "catalog.sarif.json").exists()


@pytest.mark.parametrize("exit_code", ["0", "1", ""])
def test_changed_only_scans_skip_upload_including_no_changes(tmp_path, exit_code):
    step = _sarif_step()
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        env={**os.environ, "SCAN_MODE": "changed", "SCAN_EXIT_CODE": exit_code, "SARIF_CATEGORY": "catalog"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "changed-only scans cannot replace a complete baseline" in result.stdout


@pytest.mark.parametrize("exit_code", ["", "125", "137"])
def test_aborted_full_scan_does_not_attempt_upload(tmp_path, exit_code):
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _sarif_step()["run"]],
        env={
            **os.environ, "SCAN_MODE": "all", "SCAN_EXIT_CODE": exit_code, "SARIF_CATEGORY": "catalog",
            "FORK_PR": "false",
        },
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "did not finish normally" in result.stdout


@pytest.mark.parametrize("exit_code", ["0", "1"])
def test_fork_pr_skips_upload_with_warning_without_overriding_scan_verdict(tmp_path, exit_code):
    output = tmp_path / "github-output"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _sarif_step()["run"]],
        env={
            **os.environ, "SCAN_MODE": "all", "SCAN_EXIT_CODE": exit_code, "SARIF_CATEGORY": "catalog",
            "FORK_PR": "true", "GITHUB_OUTPUT": str(output), "RUNNER_TEMP": str(tmp_path),
        },
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "::warning::" in result.stdout
    assert "cannot target the scanned head commit" in result.stdout
    assert not output.exists()
    assert not (tmp_path / "skill-sarif").exists()
    scan = next(step for step in _job()["steps"] if step.get("name") == "Evaluate selected skills")
    assert scan.get("continue-on-error", "false") == "false"


def test_pull_request_target_cannot_publish_head_results_as_base_results():
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _sarif_step()["run"]],
        env={
            **os.environ, "SCAN_MODE": "all", "SCAN_EXIT_CODE": "0", "SARIF_CATEGORY": "catalog",
            "FORK_PR": "false", "EVENT_NAME": "pull_request_target",
        },
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "not supported on pull_request_target" in result.stdout
