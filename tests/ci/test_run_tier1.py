# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the CI adapter's boundaries; actual scanner coverage is tested by live CI."""

import contextlib
import errno
import gc
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import types
import weakref
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
SELECTOR_SPEC = importlib.util.spec_from_file_location(
    "ci_selection", MODULE_PATH.with_name("select_changed_skills.py")
)
selector = importlib.util.module_from_spec(SELECTOR_SPEC)
SELECTOR_SPEC.loader.exec_module(selector)
# Upper bound for cross-thread waits; generous for loaded CI runners.
WAIT = 60


def skill(workspace, path):
    directory = workspace / path
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: example\n---\n", encoding="utf-8")
    return directory


def payload(passed=True, status="passed", incomplete=None, findings=1):
    results = [
        ValidationResult(
            validator_name=f"Tier 1 fixture validator {index + 1}",
            validator_description="Controlled CI process result fixture",
        )
        for index in range(12)
    ]
    for result in results:
        result.metadata["gating"] = {"tier": 1, "blocking": True}
    for number in range(0 if passed else findings):
        results[0].add_finding(
            Finding(
                category="SCHEMA",
                severity="high",
                check_name="fixture_failure",
                message="Fixture validation failed" + (f" ({number + 1})" if number else ""),
                file_path="SKILL.md",
            )
        )
    if status == "incomplete":
        results[0].metadata["incomplete_scans"] = incomplete or ["semgrep"]
    return json.loads(JSONReporter(include_timestamp=False).render_all(results))


def report_dir(output: Path) -> str:
    """Name the skill report directory behind the private scanner directory the runner passes."""
    match = re.fullmatch(r"skillevaluator-private-(skill-[0-9]{3})-.+", output.name)
    assert match, output.name
    return match.group(1)


def fake_cli(responses, calls):
    # Responses follow skill order, not call order: skills may run in parallel.
    def invoke(argv, **kwargs):
        calls.append((argv, kwargs))
        output = Path(argv[argv.index("-o") + 1])
        exit_code, report = responses[int(report_dir(output).removeprefix("skill-")) - 1]
        kwargs["stdout"].write(b"::error::untrusted scanner text\n")
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


LONG_NAME = "x" * 170


@pytest.mark.parametrize(
    "roots",
    [
        "skills/tab\tname",
        "skills/escape\x1bname",
        "\n".join(["skills"] * 101),
        "\n".join([f"skills/{LONG_NAME}"] * 100),
    ],
)
def test_full_scan_and_changed_selection_share_skill_root_rules(tmp_path, roots):
    for name in ("tab\tname", "escape\x1bname", LONG_NAME):
        skill(tmp_path, f"skills/{name}")
    with pytest.raises(runner.ConfigurationError):
        runner.discover_skills(tmp_path, roots)
    with pytest.raises(selector.SelectionError):
        selector.skill_roots(tmp_path, roots)


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
    assert calls[0][1]["env"]["HOME"].startswith("/tmp/evaluator-home/worker-")
    # The CLI writes no HTML; the native renderer supplies each skill page.
    assert calls[0][0][calls[0][0].index("-r") + 1] == "json,markdown,sarif"
    for child in ("skill-001", "skill-002"):
        page = (reports / child / "skillevaluator-tier1.html").read_text()
        assert "Content-Security-Policy" in page


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
        output = Path(argv[argv.index("-o") + 1])
        if report_dir(output) == "skill-001":
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
        output = Path(argv[argv.index("-o") + 1])
        if report_dir(output) == "skill-001":
            raise OSError("untrusted failure text")
        (output / "skillevaluator-tier1.json").write_text(json.dumps(payload()), encoding="utf-8")
        return CompletedProcess(argv, 0)

    assert runner.run(workspace, reports, {"SKILL_PATHS": "skills"}, execute=invoke) == 1
    assert len(calls) == 2
    summary = json.loads((reports / "summary.json").read_text())
    assert [row["passed"] for row in summary["skills"]] == [False, True]


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
    # One worker pins the sequential order: each result is saved before the next skill starts.
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

    assert runner.run(workspace, reports, {"SCAN_WORKERS": "1"}, execute=execute) == 1
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


def parallel_cli(responses, calls, *, before=None):
    """Fake CLI for concurrent runs; ``before(name)`` runs inside the call."""
    lock = threading.Lock()

    def invoke(argv, **kwargs):
        output = Path(argv[argv.index("-o") + 1])
        name = report_dir(output)
        with lock:
            calls.append((name, dict(kwargs["env"])))
        if before:
            before(name)
        exit_code, report = responses[name]
        kwargs["stdout"].write(b"::error::untrusted scanner text\n")
        (output / "skillevaluator-tier1.json").write_text(json.dumps(report), encoding="utf-8")
        return CompletedProcess(argv, exit_code)

    return invoke


def scan_outputs(reports):
    names = ["summary.json", "summary.md", "details.md", "pr-comment.md", "index.html"]
    names += sorted(path.relative_to(reports).as_posix() for path in reports.glob("skill-*/skillevaluator-tier1.html"))
    return {name: (reports / name).read_text() for name in names}


def test_parallel_scan_matches_sequential_rows_logs_annotations_and_reports(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    names = [f"skills/s{index}" for index in range(1, 7)]
    for name in names:
        skill(workspace, name)
    failing = {"skill-002", "skill-003", "skill-005"}
    # Three failing skills with four findings each exceed the 10-annotation budget.
    responses = {
        f"skill-{index:03d}": (1, payload(False, "failed", findings=4))
        if f"skill-{index:03d}" in failing
        else (0, payload())
        for index in range(1, 7)
    }
    environment = {"SKILL_PATHS": "skills", "GITHUB_ACTIONS": "true"}
    outputs, logs, finished = {}, {}, {}
    for workers in ("1", "4"):
        order = finished[workers] = []
        done = {name: threading.Event() for name in responses}

        def before(name, workers=workers, order=order, done=done):
            if workers != "1" and name == "skill-001":
                # Finish skill 1 last so completion order differs from skill order.
                assert done["skill-002"].wait(WAIT)
            order.append(name)
            done[name].set()

        reports = tmp_path / f"reports-{workers}"
        result = runner.run(
            workspace,
            reports,
            {**environment, "SCAN_WORKERS": workers},
            execute=parallel_cli(responses, [], before=before),
        )
        assert result == 1
        outputs[workers] = scan_outputs(reports)
        logs[workers] = [line for line in capsys.readouterr().out.splitlines() if "at a time" not in line]
    assert finished["1"] == sorted(responses)
    assert finished["4"].index("skill-002") < finished["4"].index("skill-001")
    assert logs["1"] == logs["4"]
    assert outputs["1"] == outputs["4"]
    annotations = [line for line in logs["4"] if line.startswith("::error")]
    assert len(annotations) == 10
    assert all("file=skills/s2/SKILL.md" in line or "file=skills/s3/SKILL.md" in line for line in annotations[:8])
    headers = [line for line in logs["4"] if line.startswith("SkillEvaluator: evaluating skill")]
    assert headers == [
        f"SkillEvaluator: evaluating skill {index} of 6: {json.dumps(name)}" for index, name in enumerate(names, 1)
    ]


def test_default_scan_runs_four_skills_at_once(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    for index in range(5):
        skill(workspace, f"skills/s{index}")
    responses = {f"skill-{index:03d}": (0, payload()) for index in range(1, 6)}
    barrier = threading.Barrier(4, timeout=WAIT)
    lock = threading.Lock()
    state = {"active": 0, "peak": 0, "calls": 0}

    def before(name):
        with lock:
            state["calls"] += 1
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            first_wave = state["calls"] <= 4
        if first_wave:
            barrier.wait()
        with lock:
            state["active"] -= 1

    calls = []
    assert runner.run(workspace, tmp_path / "reports", {}, execute=parallel_cli(responses, calls, before=before)) == 0
    assert state["peak"] == 4 and len(calls) == 5
    assert "SkillEvaluator: scanning 5 skills, up to 4 at a time." in capsys.readouterr().out


def test_one_worker_runs_skills_strictly_one_at_a_time(tmp_path):
    workspace = tmp_path / "workspace"
    for index in range(3):
        skill(workspace, f"skills/s{index}")
    reports = tmp_path / "reports"
    responses = {f"skill-{index:03d}": (0, payload()) for index in range(1, 4)}
    seen = []

    def before(name):
        # The previous skill's result is saved before this one starts.
        seen.append(json.loads((reports / "summary.json").read_text())["passed"])

    assert (
        runner.run(workspace, reports, {"SCAN_WORKERS": "1"}, execute=parallel_cli(responses, [], before=before)) == 0
    )
    assert seen == [0, 1, 2]


def test_each_worker_gets_its_own_home_and_temporary_directory(tmp_path):
    workspace = tmp_path / "workspace"
    for index in range(4):
        skill(workspace, f"skills/s{index}")
    audit = tmp_path / "audit-tmp"
    audit.mkdir()
    responses = {f"skill-{index:03d}": (0, payload()) for index in range(1, 5)}
    barrier = threading.Barrier(2, timeout=WAIT)
    pairs = []

    def before(name):
        barrier.wait()
        pairs.append(name)

    calls = []
    environment = {"SCAN_WORKERS": "2", "TMPDIR": str(audit)}
    assert (
        runner.run(workspace, tmp_path / "reports", environment, execute=parallel_cli(responses, calls, before=before))
        == 0
    )
    assert len(calls) == 4
    homes = {env["HOME"] for _, env in calls}
    assert homes == {"/tmp/evaluator-home/worker-1", "/tmp/evaluator-home/worker-2"}
    for _, env in calls:
        worker = env["HOME"].rsplit("-", 1)[1]
        assert env["TMPDIR"] == str(audit / f"skillevaluator-worker-{worker}")
        assert Path(env["HOME"]).is_dir() and Path(env["TMPDIR"]).is_dir()
    # Calls 1-2 and 3-4 ran together, each pair on two different workers.
    assert calls[0][1]["HOME"] != calls[1][1]["HOME"] and calls[2][1]["HOME"] != calls[3][1]["HOME"]


def test_concurrent_timeout_keeps_finished_results_and_starts_no_more_skills(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    for name in "abcde":
        skill(workspace, f"skills/{name}")
    reports = tmp_path / "reports"
    now = [0.0]
    second_started, first_timed_out = threading.Event(), threading.Event()
    calls = []

    def execute(argv, **kwargs):
        output = Path(argv[argv.index("-o") + 1])
        calls.append(report_dir(output))
        if report_dir(output) == "skill-001":
            assert second_started.wait(WAIT)
            now[0] = runner.SCAN_TIMEOUT_SECONDS + 1
            first_timed_out.set()
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        second_started.set()
        # Skill 2 was already running when the budget ran out; its result counts.
        assert first_timed_out.wait(WAIT)
        (output / "skillevaluator-tier1.json").write_text(json.dumps(payload()), encoding="utf-8")
        return CompletedProcess(argv, 0)

    environment = {"SCAN_WORKERS": "2"}
    assert runner.run(workspace, reports, environment, execute=execute, clock=lambda: now[0]) == 1
    assert sorted(calls) == ["skill-001", "skill-002"]
    summary = json.loads((reports / "summary.json").read_text())
    assert (summary["passed"], summary["incomplete"]) == (1, 4)
    rows = summary["skills"]
    assert "time limit" in rows[0]["error"] and rows[1]["passed"] is True
    assert all("Not evaluated" in row["error"] for row in rows[2:])
    output = capsys.readouterr().out
    assert 'evaluating skill 2 of 5: "skills/b"' in output and "evaluating skill 3" not in output
    assert "Scan time limit reached; this skill did not finish." in output
    assert "3 skills were not started" in output


def test_checkpoint_saves_a_later_skill_before_an_earlier_one_finishes(tmp_path):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    skill(workspace, "skills/b")
    reports = tmp_path / "reports"
    responses = {"skill-001": (0, payload()), "skill-002": (0, payload())}
    snapshots = []

    def before(name):
        if name != "skill-001":
            return
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            rows = json.loads((reports / "summary.json").read_text())["skills"]
            if rows[1]["passed"]:
                snapshots.append(rows)
                return
            time.sleep(0.01)

    assert (
        runner.run(workspace, reports, {"SCAN_WORKERS": "2"}, execute=parallel_cli(responses, [], before=before)) == 0
    )
    assert len(snapshots) == 1 and snapshots[0][0]["passed"] is False


@pytest.mark.parametrize("value", ["0", "17", "four", "-1", "2.5", "٣"])
def test_invalid_scan_worker_count_fails_before_scanning(tmp_path, value):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    reports, calls = tmp_path / "reports", []
    assert runner.run(workspace, reports, {"SCAN_WORKERS": value}, execute=fake_cli([], calls)) == 1
    assert not calls
    assert "SCAN_WORKERS" in json.loads((reports / "summary.json").read_text())["error"]


posix_only = pytest.mark.skipif(os.name != "posix", reason="uses a shell script as the evaluator")


def scan_shim(tmp_path, *, fast=(), ignore_term=False):
    """Put a fake ``skillevaluator`` on a PATH and return that PATH.

    A scan records its PID and its background tool's PID in ``scan.pids``,
    then runs for five minutes. Skills named in ``fast`` pass at once instead.
    """
    tool = tmp_path / "bin"
    tool.mkdir()
    report = tmp_path / "passing-report.json"
    report.write_text(json.dumps(payload()), encoding="utf-8")
    cases = "".join(f'*/{name}) cp "{report}" "$out/skillevaluator-tier1.json"; exit 0 ;;\n' for name in fast)
    script = tool / "skillevaluator"
    script.write_text(
        "#!/bin/sh\n"
        + ("trap '' TERM\n" if ignore_term else "")
        + "for out; do :; done\n"
        + f'case "$2" in\n{cases}*) ;;\nesac\n'
        + "sleep 300 &\n"
        + 'echo "$$ $!" > "$out/scan.tmp" && mv "$out/scan.tmp" "$out/scan.pids"\n'
        + "exec sleep 300\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return f"{tool}{os.pathsep}/usr/bin{os.pathsep}/bin"


def scan_pids(output, deadline):
    pids = output / "scan.pids"
    while not pids.exists():
        assert time.monotonic() < deadline, "the scan did not start"
        time.sleep(0.02)
    return [int(pid) for pid in pids.read_text().split()]


def private_scan_pids(temporary, name, deadline):
    """Read the PIDs of the scan for report directory ``name`` from its private directory under TMPDIR."""
    while not (found := sorted(temporary.rglob(f"skillevaluator-private-{name}-*/scan.pids"))):
        assert time.monotonic() < deadline, "the scan did not start"
        time.sleep(0.02)
    return scan_pids(found[0].parent, deadline)


def still_running(pids):
    """Wait up to WAIT for the processes to end; return those still running."""
    deadline = time.monotonic() + WAIT
    running = [pid for pid in pids if not ended(pid)]
    while running and time.monotonic() < deadline:
        time.sleep(0.05)
        running = [pid for pid in running if not ended(pid)]
    return running


def ended(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def kill_quietly(pids):
    # Only for processes not seen to end: a PID that ended may be reused.
    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


@posix_only
def test_scan_timeout_also_ends_the_scanner_tools(tmp_path, monkeypatch):
    path = scan_shim(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    clock = types.SimpleNamespace(offset=0.0)
    clock.monotonic = lambda: time.monotonic() + clock.offset
    monkeypatch.setattr(runner, "time", clock)
    outcome = {}

    def scan():
        try:
            runner.run_scan(
                ["skillevaluator", "tier1", "skills/a", "-o", str(output)],
                abort=threading.Event(),
                timeout=WAIT,
                env={"PATH": path},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=scan, daemon=True)
    thread.start()
    pids = survivors = scan_pids(output, time.monotonic() + WAIT)
    try:
        clock.offset = WAIT
        thread.join(WAIT)
        assert not thread.is_alive() and "error" in outcome
        survivors = still_running(pids)
        assert not survivors
    finally:
        kill_quietly(survivors)


@posix_only
def test_aborted_scan_is_killed_after_the_grace_period(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "STOP_GRACE_SECONDS", 0.5)
    path = scan_shim(tmp_path, ignore_term=True)
    output = tmp_path / "out"
    output.mkdir()
    abort, outcome = threading.Event(), {}

    def scan():
        outcome["process"] = runner.run_scan(
            ["skillevaluator", "tier1", "skills/a", "-o", str(output)],
            abort=abort,
            timeout=WAIT,
            env={"PATH": path},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    thread = threading.Thread(target=scan, daemon=True)
    thread.start()
    pids = survivors = scan_pids(output, time.monotonic() + WAIT)
    try:
        abort.set()
        thread.join(WAIT)
        assert not thread.is_alive()
        # The scan and its tool ignore SIGTERM, so only SIGKILL ends them.
        assert outcome["process"].returncode == -signal.SIGKILL
        survivors = still_running(pids)
        assert not survivors
    finally:
        kill_quietly(survivors)


@posix_only
def test_checkpoint_error_ends_running_scans_at_once(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/fast")
    skill(workspace, "skills/slow")
    audit = tmp_path / "audit-tmp"
    audit.mkdir()
    reports = tmp_path / "reports"
    environment = {"PATH": scan_shim(tmp_path, fast=["fast"]), "TMPDIR": str(audit)}
    render, pids, outcome = runner.render_reports, [], {}
    survivors = pids

    def render_reports(*args):
        if (reports / "skill-001" / "skillevaluator-tier1.json").exists():
            # The slow scan is running when saving the fast result fails.
            pids.extend(private_scan_pids(audit, "skill-002", time.monotonic() + WAIT))
            raise OSError(errno.ENOSPC, "No space left on device")
        render(*args)

    monkeypatch.setattr(runner, "render_reports", render_reports)

    def scan():
        try:
            runner.run(workspace, reports, environment)
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=scan, daemon=True)
    thread.start()
    try:
        thread.join(WAIT)
        assert not thread.is_alive(), "the run waited for a running scan"
        assert isinstance(outcome["error"], OSError)
        survivors = still_running(pids)
        assert pids and not survivors
    finally:
        kill_quietly(survivors)


def start_runner_with_endless_scans(tmp_path, *, ignore_term=False):
    """Start the runner alone in its session, as Docker runs it, and wait for two endless scans."""
    workspace = tmp_path / "workspace"
    for name in ("a", "b"):
        skill(workspace, f"skills/{name}")
    audit = tmp_path / "audit-tmp"
    audit.mkdir()
    reports = tmp_path / "reports"
    environment = {"PATH": scan_shim(tmp_path, ignore_term=ignore_term), "TMPDIR": str(audit)}
    environment.update(
        (name, os.environ[name]) for name in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE") if name in os.environ
    )
    with (tmp_path / "runner.log").open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, str(MODULE_PATH), "--workspace", str(workspace), "--reports", str(reports)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pids = []
    try:
        deadline = time.monotonic() + WAIT
        for name in ("skill-001", "skill-002"):
            pids += private_scan_pids(audit, name, deadline)
    except BaseException:
        process.kill()
        process.wait()
        kill_quietly(pids)
        raise
    return process, pids


@posix_only
def test_ctrl_c_ends_running_scans_and_their_tools(tmp_path):
    process, pids = start_runner_with_endless_scans(tmp_path)
    survivors = pids
    try:
        # Docker sends Ctrl-C to the runner (PID 1) only, not to its scans.
        os.kill(process.pid, signal.SIGINT)
        assert process.wait(WAIT) == -signal.SIGINT
        survivors = still_running(pids)
        assert not survivors
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        kill_quietly(survivors)


@posix_only
def test_second_ctrl_c_does_not_leave_the_runner_waiting(tmp_path):
    # The scans ignore SIGTERM, so the second Ctrl-C arrives while the
    # runner is still waiting for them to end.
    process, pids = start_runner_with_endless_scans(tmp_path, ignore_term=True)
    survivors = pids
    try:
        os.kill(process.pid, signal.SIGINT)
        time.sleep(1)
        os.kill(process.pid, signal.SIGINT)
        assert process.wait(WAIT) == -signal.SIGINT
        survivors = still_running(pids)
        assert not survivors
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        kill_quietly(survivors)


def test_each_report_is_released_once_its_result_is_logged(tmp_path, monkeypatch):
    # Reports can be 32 MiB each; 100 of them must not stay in memory.
    workspace = tmp_path / "workspace"
    for index in range(5):
        skill(workspace, f"skills/s{index}")
    responses = {f"skill-{index:03d}": (0, payload()) for index in range(1, 6)}
    load, reports, alive = runner._load_report, [], []

    class Report(dict):
        pass

    def tracked(path):
        report = Report(load(path))
        reports.append(weakref.ref(report))
        return report

    def before(name):
        gc.collect()
        alive.append(sum(report() is not None for report in reports))

    monkeypatch.setattr(runner, "_load_report", tracked)
    environment = {"SCAN_WORKERS": "1"}
    assert (
        runner.run(workspace, tmp_path / "reports", environment, execute=parallel_cli(responses, [], before=before))
        == 0
    )
    # Only the latest result may still be referenced when the next scan starts.
    assert alive == [0, 1, 1, 1, 1]


def worker_directory(env, name):
    return Path(env[name]) / ("worker-1" if name == "HOME" else "skillevaluator-worker-1")


@pytest.mark.parametrize("name", ["HOME", "TMPDIR"])
def test_symlinked_worker_directory_is_rejected(tmp_path, name):
    env = {"HOME": str(tmp_path / "home"), "TMPDIR": str(tmp_path / "tmp")}
    directory = worker_directory(env, name)
    directory.parent.mkdir()
    (tmp_path / "elsewhere").mkdir(mode=0o700)
    directory.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(runner.ConfigurationError):
        runner.worker_environment(env, 1)


@posix_only
@pytest.mark.parametrize("name", ["HOME", "TMPDIR"])
def test_existing_worker_directory_open_to_others_is_rejected(tmp_path, name):
    env = {"HOME": str(tmp_path / "home"), "TMPDIR": str(tmp_path / "tmp")}
    directory = worker_directory(env, name)
    directory.mkdir(parents=True)
    directory.chmod(0o777)
    with pytest.raises(runner.ConfigurationError):
        runner.worker_environment(env, 1)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o777


@posix_only
def test_worker_directory_owned_by_another_user_is_rejected(tmp_path, monkeypatch):
    env = {"HOME": str(tmp_path / "home"), "TMPDIR": str(tmp_path / "tmp")}
    uid = os.getuid()
    monkeypatch.setattr(runner.os, "getuid", lambda: uid + 1)
    with pytest.raises(runner.ConfigurationError):
        runner.worker_environment(env, 1)


@posix_only
def test_existing_private_worker_directories_are_reused(tmp_path):
    env = {"HOME": str(tmp_path / "home"), "TMPDIR": str(tmp_path / "tmp")}
    for name in ("HOME", "TMPDIR"):
        worker_directory(env, name).mkdir(mode=0o700, parents=True)
        (worker_directory(env, name) / "kept").write_text("", encoding="utf-8")
    isolated = runner.worker_environment(env, 1)
    for name in ("HOME", "TMPDIR"):
        assert isolated[name] == str(worker_directory(env, name))
        assert (worker_directory(env, name) / "kept").is_file()


def test_symlinked_worker_directory_fails_before_scanning(tmp_path):
    workspace = tmp_path / "workspace"
    skill(workspace, "skills/a")
    audit = tmp_path / "audit-tmp"
    audit.mkdir()
    (tmp_path / "elsewhere").mkdir(mode=0o700)
    (audit / "skillevaluator-worker-1").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    reports, calls = tmp_path / "reports", []
    assert runner.run(workspace, reports, {"TMPDIR": str(audit)}, execute=fake_cli([], calls)) == 1
    assert not calls
    summary = json.loads((reports / "summary.json").read_text())
    assert summary["overall_status"] == "error"
    assert summary["error"] == "Scan worker directories must be private directories owned by this user."


@pytest.mark.parametrize("workers", ["1", "4"])
def test_unexpected_worker_error_is_a_row_error_and_later_skills_still_run(tmp_path, capsys, workers):
    workspace = tmp_path / "workspace"
    for name in ("a", "b", "c"):
        skill(workspace, f"skills/{name}")
    reports, calls = tmp_path / "reports", []
    passing = parallel_cli({"skill-002": (0, payload()), "skill-003": (0, payload())}, calls)

    def execute(argv, **kwargs):
        if report_dir(Path(argv[argv.index("-o") + 1])) == "skill-001":
            raise RuntimeError("untrusted worker text")
        return passing(argv, **kwargs)

    assert runner.run(workspace, reports, {"SCAN_WORKERS": workers}, execute=execute) == 1
    assert sorted(name for name, _ in calls) == ["skill-002", "skill-003"]
    rows = json.loads((reports / "summary.json").read_text())["skills"]
    assert rows[0]["error"] == "Evaluator execution failed unexpectedly; no result is claimed."
    assert [row["passed"] for row in rows] == [False, True, True]
    captured = capsys.readouterr()
    for name in ("a", "b", "c"):
        assert f'Skill result: "skills/{name}"' in captured.out
    assert "untrusted worker text" not in captured.out + captured.err
    assert all("untrusted worker text" not in text for text in scan_outputs(reports).values())
