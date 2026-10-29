# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the CI adapter's boundaries; actual scanner coverage is tested by live CI."""

import importlib.util
import io
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from skillevaluator.models import Finding, ValidationResult
from skillevaluator.reporting.json_reporter import JSONReporter

MODULE_PATH = Path(__file__).parents[2] / "ci" / "run_tier1.py"
SPEC = importlib.util.spec_from_file_location("ci_run_tier1", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def skill(workspace, path):
    directory = workspace / path
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: example\n---\n", encoding="utf-8")
    return directory


def payload(passed=True, status="passed", incomplete=None):
    results = [
        ValidationResult(
            validator_name=f"Tier 1 fixture validator {index + 1}",
            validator_description="Controlled CI process result fixture",
        )
        for index in range(12)
    ]
    for result in results:
        result.metadata["gating"] = {"tier": 1, "blocking": True}
    if not passed:
        results[0].add_finding(
            Finding(
                category="SCHEMA",
                severity="high",
                check_name="fixture_failure",
                message="Fixture validation failed",
                file_path="SKILL.md",
            )
        )
    if status == "incomplete":
        results[0].metadata["incomplete_scans"] = incomplete or ["semgrep"]
    return json.loads(JSONReporter(include_timestamp=False).render_all(results))


def fake_cli(responses, calls):
    def invoke(argv, **kwargs):
        calls.append((argv, kwargs))
        exit_code, report = responses[len(calls) - 1]
        kwargs["stdout"].write(b"::error::untrusted scanner text\n")
        output = Path(argv[argv.index("-o") + 1])
        if report is not None:
            (output / "skillevaluator-tier1.json").write_text(json.dumps(report), encoding="utf-8")
        return CompletedProcess(argv, exit_code)

    return invoke


def test_selection_deduplicates_overlapping_roots(tmp_path):
    first = skill(tmp_path, "skills/a")
    second = skill(tmp_path, "skills/b")
    assert runner.discover_skills(tmp_path, "skills\nskills/a\nskills") == [first, second]


def test_contradictory_passing_report_with_incomplete_scanners_has_publishable_error_row(tmp_path):
    import ci.publish_reports as publisher

    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    report = payload()
    report["incomplete_scans"] = ["semgrep"]
    reports = tmp_path / "reports"
    exit_code = runner.run(
        workspace,
        reports,
        {"SKILL_PATHS": "skills", "LLM_ENABLED": "false"},
        execute=fake_cli([(0, report)], []),
    )
    summary = json.loads((reports / "summary.json").read_text())
    assert exit_code == 1
    assert summary["skills"][0]["overall_status"] == "error"
    publisher.validate_summary(summary)
    artifact = io.BytesIO()
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.write(reports / "summary.json", "summary.json")
        archive.write(reports / "skill-001/skillevaluator-tier1.json", "skill-001/skillevaluator-tier1.json")
    retained, children = publisher.read_artifact(artifact.getvalue())
    assert retained["skills"][0]["overall_status"] == "error"
    assert children == {}


@pytest.mark.parametrize(
    "exit_code,status,changes",
    [
        (1, "failed", {"incomplete_scans": ["semgrep"]}),
        (0, "failed", {}),
        (1, "incomplete", {"incomplete_scans": []}),
        (1, "incomplete", {"incomplete_scans": ["trivy"]}),
        (0, "incomplete", {}),
        (0, "passed", {"total_errors": 1}),
    ],
)
def test_any_contradictory_report_becomes_error_row_without_losing_other_skills(tmp_path, exit_code, status, changes):
    import ci.publish_reports as publisher

    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    skill(workspace, "skills/b")
    report = {**payload(status == "passed", status), **changes}
    reports = tmp_path / "reports"
    runner.run(
        workspace,
        reports,
        {"SKILL_PATHS": "skills", "LLM_ENABLED": "false"},
        execute=fake_cli([(exit_code, report), (0, payload())], []),
    )
    summary = json.loads((reports / "summary.json").read_text())
    assert [row["overall_status"] for row in summary["skills"]] == ["error", "passed"]
    assert summary["skills"][0]["incomplete_scans"] == []
    assert summary["skills"][0]["passed"] is False
    artifact = io.BytesIO()
    with zipfile.ZipFile(artifact, "w") as archive:
        for path in sorted(reports.rglob("*.json")):
            archive.write(path, path.relative_to(reports).as_posix())
    retained, children = publisher.read_artifact(artifact.getvalue())
    assert [row["overall_status"] for row in retained["skills"]] == ["error", "passed"]
    assert list(children) == ["skill-002"]
    assert "Report unavailable" in (reports / "skill-001/skillevaluator-tier1.html").read_text()


@pytest.mark.parametrize(
    "root", ["/etc", "../skills", "skills/../other", "C:\\skills", "skills\\a", "sk\tills", "skills/m\x7fskill"]
)
def test_rejects_unsafe_roots(tmp_path, root):
    skill(tmp_path, "sk\tills/a")
    skill(tmp_path, "skills/m\x7fskill")
    with pytest.raises(runner.ConfigurationError, match="without parent traversal"):
        runner.discover_skills(tmp_path, root)


def test_empty_selection_and_missing_root_fail(tmp_path):
    for roots in ("", "skills"):
        with pytest.raises(runner.ConfigurationError):
            runner.discover_skills(tmp_path, roots)
    (tmp_path / "skills").mkdir()
    with pytest.raises(runner.ConfigurationError):
        runner.discover_skills(tmp_path, "skills")


@pytest.mark.parametrize("link_kind", ["root", "ancestor", "file", "directory"])
def test_rejects_symlinks_anywhere_in_tree(tmp_path, link_kind):
    target = skill(tmp_path, "skills/a")
    if link_kind == "root":
        (tmp_path / "alias").symlink_to(target, target_is_directory=True)
        configured = "alias"
    elif link_kind == "ancestor":
        (tmp_path / "alias").symlink_to(tmp_path / "skills", target_is_directory=True)
        configured = "alias/a"
    else:
        (target / "linked").symlink_to(target if link_kind == "directory" else target / "SKILL.md")
        configured = "skills"
    with pytest.raises(runner.ConfigurationError):
        runner.discover_skills(tmp_path, configured)


def test_target_limit_is_global(tmp_path):
    for index in range(101):
        skill(tmp_path, f"skills/a{index:03d}")
    with pytest.raises(runner.ConfigurationError):
        runner.discover_skills(tmp_path, "skills")


def test_all_skills_run_after_failure_and_scanner_stdout_is_not_printed(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    skill(workspace, "skills/b")
    calls = []
    reports = tmp_path / "reports"
    exit_code = runner.run(
        workspace,
        reports,
        {"SKILL_PATHS": "skills", "LLM_ENABLED": "false"},
        execute=fake_cli([(1, payload(False, "failed")), (0, payload())], calls),
    )
    summary = json.loads((reports / "summary.json").read_text())
    assert exit_code == 1
    assert len(calls) == 2
    assert (summary["total"], summary["passed"], summary["failed"]) == (2, 1, 1)
    assert summary["total_validators"] == 24
    assert "::error::" not in capsys.readouterr().out
    assert "::error::" in (reports / "skill-001" / "stdout.log").read_text()
    assert "--no-llm" in calls[0][0]
    assert calls[0][1]["env"]["HOME"] == "/tmp/evaluator-home"


def test_structured_failure_logs_and_annotations_are_escaped(capsys):
    row = {"path": "skills/example", "overall_status": "failed"}
    report = {
        "quality_summary": [{"overall_score": 71.8, "grade": "C"}],
        "results": [
            {
                "validator": "Schema,metadata",
                "status": "failed",
                "passed": False,
                "findings": [
                    {
                        "severity": "high",
                        "message": "Bad name\n::notice::injected",
                        "suggestion": "Use kebab-case",
                        "file_path": "/workspace/skills/example/SKILL.md",
                        "line_number": None,
                    }
                ],
            }
        ],
    }
    runner.log_result(row, report, annotations=True, budget={"errors": 0})
    output = capsys.readouterr().out
    assert "71.8/100" in output
    assert "Suggested fix:" in output
    assert "\n::notice::injected" not in output
    commands = [line for line in output.splitlines() if line.startswith("::")]
    assert len(commands) == 1
    assert "title=Schema%2Cmetadata,file=skills/example/SKILL.md" in commands[0]
    assert "Bad name%0A::notice::injected" in commands[0]
    assert ",line=" not in commands[0]


def test_annotation_budget_does_not_hide_log_details(capsys):
    row = {"path": "skills/example", "overall_status": "failed"}
    report = {
        "results": [
            {
                "validator": "Schema",
                "status": "failed",
                "passed": False,
                "findings": [{"severity": "high", "message": "Bad metadata"}],
            }
        ]
    }
    runner.log_result(row, report, annotations=True, budget={"errors": 10})
    output = capsys.readouterr().out
    assert "Bad metadata" in output
    assert "::error" not in output


@pytest.mark.parametrize(
    "exit_code,report",
    [
        (0, None),
        (0, {}),
        (1, payload()),
        (0, payload(False, "failed")),
        (0, payload(True, "incomplete", ["semgrep"])),
        (0, {**payload(), "results": [{"passed": False, "status": "failed"}]}),
        (0, {**payload(), "overall_passed": "true"}),
        (0, {**payload(), "total_validators": 100}),
        (0, {**payload(), "results": [{"passed": True, "status": "incomplete"}] * 12}),
    ],
)
def test_exit_or_report_disagreement_never_passes(tmp_path, exit_code, report):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    reports = tmp_path / "reports"
    result = runner.run(workspace, reports, {"SKILL_PATHS": "skills"}, execute=fake_cli([(exit_code, report)], []))
    assert result == 1
    assert json.loads((reports / "summary.json").read_text())["overall_passed"] is False


def test_llm_maps_only_explicit_provider_and_credentials(tmp_path):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    calls = []
    environment = {
        "SKILL_PATHS": "skills",
        "LLM_ENABLED": "true",
        "LLM_PROVIDER": "nv_build",
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "test-model",
        "OPENAI_API_KEY": "ambient-key",
        "SKILLSPECTOR_PROVIDER": "untrusted-ambient-provider",
        "SKILL_EVAL_LLM_BASE_URL": "https://unexpected.invalid",
    }
    assert runner.run(workspace, tmp_path / "reports", environment, execute=fake_cli([(0, payload())], calls)) == 0
    argv, kwargs = calls[0]
    assert "--llm" in argv
    assert kwargs["env"]["NVIDIA_API_KEY"] == "test-key"
    assert kwargs["env"]["SKILL_EVAL_LLM_PROVIDER"] == "nv_build"
    assert kwargs["env"]["SKILL_EVAL_LLM_MODEL"] == "test-model"
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert "SKILLSPECTOR_PROVIDER" not in kwargs["env"]
    assert "SKILL_EVAL_LLM_BASE_URL" not in kwargs["env"]


@pytest.mark.parametrize(
    "extra",
    [
        {"LLM_ENABLED": "true"},
        {"LLM_ENABLED": "true", "LLM_PROVIDER": "arbitrary", "LLM_API_KEY": "key"},
        {"LLM_ENABLED": "yes"},
    ],
)
def test_invalid_llm_configuration_is_reported_without_running(tmp_path, extra):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    reports = tmp_path / "reports"
    calls = []
    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills", **extra}, execute=fake_cli([], calls)) == 1
    assert not calls
    assert json.loads((reports / "summary.json").read_text())["overall_status"] == "error"


def test_markdown_paths_cannot_inject_html_or_table_rows(tmp_path):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/<script>|[x]`danger`")
    reports = tmp_path / "reports"
    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills"}, execute=fake_cli([(0, payload())], [])) == 0
    markdown = (reports / "details.md").read_text()
    assert "<script>" not in markdown
    assert "&lt;script&gt;" in markdown
    assert "\\|" in markdown


def test_successful_rollup_preserves_all_counts(tmp_path):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    reports = tmp_path / "reports"
    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills"}, execute=fake_cli([(0, payload())], [])) == 0
    summary = json.loads((reports / "summary.json").read_text())
    assert summary["overall_passed"] is True
    assert summary["overall_status"] == "passed"
    assert summary["incomplete"] == 0
    assert summary["skills"][0]["path"] == "skills/a"


def test_invalid_json_is_reported_and_later_skills_still_run(tmp_path):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    skill(workspace, "skills/b")
    reports = tmp_path / "reports"
    calls = []
    original = fake_cli([(0, payload()), (0, payload())], calls)

    def invoke(argv, **kwargs):
        completed = original(argv, **kwargs)
        if len(calls) == 1:
            output = Path(argv[argv.index("-o") + 1])
            (output / "skillevaluator-tier1.json").write_text("not json", encoding="utf-8")
        return completed

    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills"}, execute=invoke) == 1
    summary = json.loads((reports / "summary.json").read_text())
    assert summary["incomplete"] == 1
    assert summary["passed"] == 1
    assert len(calls) == 2


def test_process_start_error_does_not_skip_later_skills(tmp_path):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    skill(workspace, "skills/b")
    reports = tmp_path / "reports"
    calls = []

    def invoke(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            raise OSError("untrusted failure text")
        output = Path(argv[argv.index("-o") + 1])
        (output / "skillevaluator-tier1.json").write_text(json.dumps(payload()), encoding="utf-8")
        return CompletedProcess(argv, 0)

    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills"}, execute=invoke) == 1
    assert len(calls) == 2


def test_static_run_does_not_receive_ambient_model_credentials():
    env, enabled = runner.child_environment(
        {
            "LLM_ENABLED": "false",
            "LLM_API_KEY": "secret",
            "OPENAI_API_KEY": "ambient",
            "SKILL_EVAL_LLM_PROVIDER": "openai",
            "SKILLSPECTOR_PROVIDER": "anthropic",
        }
    )
    assert enabled is False
    assert not any("KEY" in name or "PROVIDER" in name for name in env)
    assert env["PYTHON_DOTENV_DISABLED"] == "1"
    assert env["SEMGREP_SEND_METRICS"] == "off"
    assert env["SEMGREP_ENABLE_VERSION_CHECK"] == "0"


def test_incomplete_runtime_rollup_is_not_mislabeled_as_inconsistent_evidence(tmp_path):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    reports = tmp_path / "reports"
    assert (
        runner.run(
            workspace,
            reports,
            {"SKILL_PATHS": "skills"},
            execute=fake_cli([(1, payload(False, "incomplete", ["semgrep"]))], []),
        )
        == 1
    )
    summary = json.loads((reports / "summary.json").read_text())
    assert summary["skills"][0]["overall_status"] == "incomplete"
    assert (summary["failed"], summary["incomplete"]) == (1, 1)
    compact = (reports / "summary.md").read_text()
    assert "**Tier 1 scan result: INCOMPLETE**" in compact
    assert "1 skills · 0 passed · 0 failed · 1 incomplete" in compact
    assert "Report evidence is missing or inconsistent" not in compact


def repository_git(workspace, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(workspace), "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgSign=false", *arguments],
        text=True,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
    ).strip()


def repository_commit(workspace):
    repository_git(workspace, "add", "-A")
    repository_git(workspace, "commit", "--allow-empty", "-m", "fixture")
    return repository_git(workspace, "rev-parse", "HEAD")


def selection_checkout(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository_git(workspace, "init", "-q")
    repository_git(workspace, "config", "user.name", "Runner Test")
    repository_git(workspace, "config", "user.email", "runner@example.invalid")
    skill(workspace, "skills/a")
    skill(workspace, "skills/b")
    return workspace, repository_commit(workspace)


def test_changed_mode_runs_only_changed_skill_from_real_git_history(tmp_path):
    workspace, base = selection_checkout(tmp_path)
    (workspace / "skills/a/SKILL.md").write_text("# Changed skill\n")
    head = repository_commit(workspace)
    reports, calls = tmp_path / "reports", []
    assert (
        runner.run(
            workspace,
            reports,
            {
                "SKILL_PATHS": "skills\nskills/a",
                "SCAN_MODE": "changed",
                "DIFF_BASE_SHA": base,
                "DIFF_HEAD_SHA": head,
            },
            execute=fake_cli([(0, payload())], calls),
        )
        == 0
    )
    summary = json.loads((reports / "summary.json").read_text())
    assert len(calls) == 1 and str(workspace / "skills/a") in calls[0][0]
    assert summary["selection"]["selected"] == ["skills/a"]
    assert summary["selection"]["discovered"] == 2


def test_unchanged_skill_pr_is_successful_neutral_noop_with_reports(tmp_path, capsys):
    workspace, base = selection_checkout(tmp_path)
    (workspace / "README.md").write_text("Unrelated documentation\n")
    head = repository_commit(workspace)
    reports, calls = tmp_path / "reports", []
    assert (
        runner.run(
            workspace,
            reports,
            {
                "SKILL_PATHS": "skills",
                "SCAN_MODE": "changed",
                "DIFF_BASE_SHA": base,
                "DIFF_HEAD_SHA": head,
            },
            execute=fake_cli([], calls),
        )
        == 0
    )
    summary = json.loads((reports / "summary.json").read_text())
    assert not calls and summary["overall_status"] == "skipped" and summary["overall_passed"] is False
    assert (summary["total"], summary["passed"], summary["failed"], summary["incomplete"]) == (0, 0, 0, 0)
    assert summary["skills"] == [] and summary["selection"]["selected"] == []
    assert "NO SKILLS CHANGED" in capsys.readouterr().out
    assert "NO SKILLS CHANGED" in (reports / "summary.md").read_text()
    assert "Tier 1: Not run" in (reports / "summary.md").read_text()
    assert (reports / "index.html").exists() and (reports / "details.md").exists()
    assert not list(reports.glob("skill-*"))


def test_completely_deleted_configured_skill_is_disclosed_not_evaluated(tmp_path):
    workspace, base = selection_checkout(tmp_path)
    shutil.rmtree(workspace / "skills/a")
    head = repository_commit(workspace)
    reports, calls = tmp_path / "reports", []
    assert (
        runner.run(
            workspace,
            reports,
            {
                "SKILL_PATHS": "skills/a",
                "SCAN_MODE": "changed",
                "DIFF_BASE_SHA": base,
                "DIFF_HEAD_SHA": head,
            },
            execute=fake_cli([], calls),
        )
        == 0
    )
    summary = json.loads((reports / "summary.json").read_text())
    assert not calls and summary["selection"]["deleted"] == ["skills/a"]
    assert "skills/a" in (reports / "summary.md").read_text()
    assert "Deleted skill directories" in (reports / "summary.md").read_text()
    assert summary["overall_passed"] is False


def test_deleted_manifest_with_remaining_content_is_error_before_evaluation(tmp_path):
    workspace, _ = selection_checkout(tmp_path)
    (workspace / "skills/a/helper.py").write_text("print('fixture')\n")
    base = repository_commit(workspace)
    (workspace / "skills/a/SKILL.md").unlink()
    head = repository_commit(workspace)
    reports, calls = tmp_path / "reports", []
    assert (
        runner.run(
            workspace,
            reports,
            {
                "SKILL_PATHS": "skills",
                "SCAN_MODE": "changed",
                "DIFF_BASE_SHA": base,
                "DIFF_HEAD_SHA": head,
            },
            execute=fake_cli([], calls),
        )
        == 1
    )
    summary = json.loads((reports / "summary.json").read_text())
    assert not calls and summary["overall_status"] == "error"
    assert "Deleted SKILL.md left tracked content" in summary["error"]
    assert "NO SKILLS CHANGED" not in (reports / "summary.md").read_text()


@pytest.mark.parametrize(
    "selection",
    [
        {"SCAN_MODE": "changed"},
        {"SCAN_MODE": "changed", "DIFF_BASE_SHA": "0" * 40, "DIFF_HEAD_SHA": "0" * 40},
        {"SCAN_MODE": "unknown"},
    ],
)
def test_invalid_selection_is_error_without_full_scan_fallback(tmp_path, selection):
    workspace, _ = selection_checkout(tmp_path)
    reports, calls = tmp_path / "reports", []
    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills", **selection}, execute=fake_cli([], calls)) == 1
    summary = json.loads((reports / "summary.json").read_text())
    assert not calls and summary["overall_status"] == "error" and summary["overall_passed"] is False


@pytest.mark.parametrize("variable", ["REPORT_URL", "REPORT_RUN_URL", "REPORT_BASE_URL"])
@pytest.mark.parametrize(
    "url", ["http://example.com", "https://user:secret@example.com", "https://example.com/\nextra"]
)
def test_invalid_report_urls_fail_before_scanning_and_keep_diagnostics(tmp_path, variable, url):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    reports, calls = tmp_path / "reports", []
    assert runner.run(workspace, reports, {variable: url}, execute=fake_cli([], calls)) == 1
    assert not calls
    assert "HTTPS" in json.loads((reports / "summary.json").read_text())["error"]
    assert "HTTPS" in (reports / "summary.md").read_text()
    assert (reports / "index.html").is_file()
    assert "user:secret" not in (reports / "pr-comment.md").read_text()


def test_completed_results_survive_timeout_and_unstarted_skills_stay_incomplete(tmp_path):
    workspace = tmp_path / "workspace"
    for name in ("a", "b", "c"):
        skill(workspace, f"skills/{name}")
    reports, calls = tmp_path / "reports", []
    first = fake_cli([(0, payload())], calls)

    def execute(argv, **kwargs):
        snapshot = json.loads((reports / "summary.json").read_text())
        assert snapshot["total"] == 3 and snapshot["overall_passed"] is False
        assert kwargs["timeout"] <= runner.SCAN_TIMEOUT_SECONDS
        if not calls:
            assert snapshot["incomplete"] == 3
            return first(argv, **kwargs)
        assert snapshot["passed"] == 1 and snapshot["incomplete"] == 2
        assert "INCOMPLETE" in (reports / "summary.md").read_text()
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    assert runner.run(workspace, reports, {}, execute=execute) == 1
    summary = json.loads((reports / "summary.json").read_text())
    assert summary["passed"] == 1 and summary["incomplete"] == 2
    assert "time limit" in summary["skills"][1]["error"]
    assert "Not evaluated" in summary["skills"][2]["error"]
    assert (reports / "index.html").is_file()


def test_unrelated_root_symlink_does_not_break_full_scan(tmp_path):
    target = skill(tmp_path, "skills/a")
    (tmp_path / "skills/README.md").symlink_to(tmp_path / "other.md")
    assert runner.discover_skills(tmp_path, "skills") == [target]


def test_control_character_skill_path_fails_before_scanning(tmp_path):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a\tbad")
    calls = []
    assert runner.run(workspace, tmp_path / "reports", {}, execute=fake_cli([], calls)) == 1
    assert not calls


@pytest.mark.parametrize(
    "path,expected", [("scripts/check.py", "skills/example/scripts/check.py"), ("../secret", None)]
)
def test_relative_annotation_location_is_confined_to_skill(capsys, path, expected):
    report = {
        "results": [
            {
                "validator": "Security",
                "passed": False,
                "status": "failed",
                "findings": [{"severity": "high", "message": "Unsafe operation", "file_path": path, "line_number": 7}],
            }
        ]
    }
    runner.log_result(
        {"path": "skills/example", "overall_status": "failed"}, report, annotations=True, budget={"errors": 0}
    )
    command = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("::error"))
    if expected:
        assert f"file={expected},line=7" in command
    else:
        assert "file=" not in command and "line=" not in command
