# SPDX-License-Identifier: Apache-2.0
"""Retain useful startup diagnostics without exporting unrelated worker state."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ci import diagnostics


def test_retains_compose_stderr_and_redacts_literals_and_credential_shapes(tmp_path):
    raw = tmp_path / "raw"
    trial = raw / "harbor/job/trial"
    trial.mkdir(parents=True)
    supplied = "fixture-model-secret-12345678"
    shaped = "nvapi-" + "x" * 48
    (trial / "exception.txt").write_text(
        "compose command "
        + "arg " * 400
        + "\nstderr: image pull denied\n"
        + supplied
        + "\n"
        + shaped
        + "\nAuthorization: Bearer fixture-unknown-token\npassword=fixture-password\n"
    )
    (trial / "failure.json").write_text(json.dumps({"error": {"message": supplied}, "api_key": "fixture-key"}))
    output = tmp_path / "reports/skill-001"
    result = diagnostics.retain_diagnostics(raw, output, {supplied})
    assert result["index"] == "diagnostics/index.json"
    assert json.loads((output / result["index"]).read_text()) == result
    text = (output / "diagnostics/harbor/job/trial/exception.txt").read_text()
    assert "stderr: image pull denied" in text
    assert all(value not in text for value in (supplied, shaped, "fixture-unknown-token", "fixture-password"))
    assert "[REDACTED]" in text
    failure = json.loads((output / "diagnostics/harbor/job/trial/failure.json").read_text())
    assert failure["error"]["message"] == "[REDACTED]" and failure["api_key"] == "[REDACTED]"
    assert str(raw) not in json.dumps(result)


def test_whitelist_excludes_agent_state_configs_payloads_and_binary_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for relative in (
        ".env",
        "config.json",
        "trajectory.json",
        "agent/stdout.log",
        "artifacts/exception.txt",
        "workspace/stderr.log",
    ):
        path = raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unrelated private state")
    (raw / "stderr.log").write_bytes(b"binary\x00payload")
    output = tmp_path / "output"
    result = diagnostics.retain_diagnostics(raw, output, set())
    assert result["retained"] == []
    assert {item["reason"] for item in result["omitted"]} >= {"not_allowed", "binary_or_encoding"}
    assert not list((output / "diagnostics").rglob("*.log"))


def test_symlink_and_special_file_are_rejected_without_following_them(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    external = tmp_path / "outside"
    external.mkdir()
    (external / "exception.txt").write_text("outside evidence")
    (raw / "linked-directory").symlink_to(external, target_is_directory=True)
    (raw / "exception.txt").symlink_to(external / "exception.txt")
    os.mkfifo(raw / "trial.log")
    output = tmp_path / "output"
    result = diagnostics.retain_diagnostics(raw, output, set())
    assert result["retained"] == []
    assert {item["reason"] for item in result["errors"]} >= {"unsafe_link", "special_file"}
    assert not (output / "diagnostics/linked-directory").exists()
    with pytest.raises(ValueError):
        diagnostics.retain_diagnostics(raw / "linked-directory", tmp_path / "another", set())


def test_large_text_is_bounded_and_preserves_terminal_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "MAX_FILE_BYTES", 256)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "exception.txt").write_text("begin\n" + "long-command " * 100 + "\nstderr: daemon unavailable\n")
    output = tmp_path / "output"
    result = diagnostics.retain_diagnostics(raw, output, set())
    target = output / "diagnostics/exception.txt"
    assert target.stat().st_size <= 256
    assert "stderr: daemon unavailable" in target.read_text()
    assert result["truncated"] == ["exception.txt"]


def test_file_count_and_total_bytes_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "MAX_TRIALS_PER_JOB", 10)
    monkeypatch.setattr(diagnostics, "MAX_TRIALS", 10)
    monkeypatch.setattr(diagnostics, "MAX_FILES", 2)
    monkeypatch.setattr(diagnostics, "MAX_TOTAL_BYTES", 70)
    raw = tmp_path / "raw"
    for n in range(5):
        path = raw / str(n) / "trial.log"
        path.parent.mkdir(parents=True)
        path.write_text("x" * 30)
    output = tmp_path / "output"
    result = diagnostics.retain_diagnostics(raw, output, set())
    assert len(result["retained"]) <= 2
    assert sum(item["bytes"] for item in result["retained"]) <= 70
    assert result["omitted"]


def test_optional_command_log_is_explicit_and_destination_cannot_escape(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    command = tmp_path / "worker.log"
    command.write_text("worker startup failed\n")
    output = tmp_path / "output"
    result = diagnostics.retain_diagnostics(raw, output, set(), command_log=command)
    assert (output / "diagnostics/command.log").read_text() == command.read_text()
    assert result["retained"][0]["path"] == "command.log"
    with pytest.raises(ValueError):
        diagnostics.retain_diagnostics(raw, raw / "nested", set())
    linked = tmp_path / "linked-output"
    linked.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError):
        diagnostics.retain_diagnostics(raw, linked, set())


def test_only_known_harbor_install_setup_logs_are_retained_from_agent_directory(tmp_path):
    raw = tmp_path / "raw"
    setup = raw / "trial/agent/setup"
    setup.mkdir(parents=True)
    (setup / "install-agent-runtime.attempt-2.failed.log").write_text("stderr: install failed\n")
    (setup / "arbitrary-session.log").write_text("agent state")
    (setup.parent / "stdout.log").write_text("agent output")
    output = tmp_path / "output"
    result = diagnostics.retain_diagnostics(raw, output, set())
    assert [item["path"] for item in result["retained"]] == [
        "trial/agent/setup/install-agent-runtime.attempt-2.failed.log"
    ]
    assert not (output / "diagnostics/trial/agent/stdout.log").exists()


@pytest.mark.parametrize(
    "credential",
    [
        "Authorization: Basic dXNlcjpzZWNyZXQ=",
        'password="secret with spaces"',
        "password='secret with spaces'",
    ],
)
def test_text_credentials_are_redacted_as_complete_values(tmp_path, credential):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "exception.txt").write_text(credential + "\nstderr: startup failed\n")
    output = tmp_path / "output"
    diagnostics.retain_diagnostics(raw, output, set())
    retained = (output / "diagnostics/exception.txt").read_text()
    assert "[REDACTED]" in retained
    assert "dXNlcjpzZWNyZXQ=" not in retained
    assert "secret" not in retained and "with spaces" not in retained
    assert "stderr: startup failed" in retained


def test_agent_setup_uses_the_shared_discovery_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "MAX_ENTRIES", 8)
    raw = tmp_path / "raw"
    for trial in ("trial-a", "trial-b"):
        setup = raw / trial / "agent/setup"
        setup.mkdir(parents=True)
        for attempt in range(1, 6):
            (setup / f"install-agent-runtime.attempt-{attempt}.log").write_text("startup failed\n")
    (raw / "stderr.log").write_text("worker failed\n")
    output = tmp_path / "output"
    result = diagnostics.retain_diagnostics(raw, output, set())
    # Two trial directories, the root log and the first agent/setup directories
    # consume five entries, leaving room for three setup logs across the tree.
    assert len(result["retained"]) == 4
    assert len(list((output / "diagnostics").rglob("*.log"))) == 4
    assert any(item["reason"] == "discovery_limit" for item in result["omitted"])


def test_long_token_free_log_lines_finish_without_quadratic_key_scanning():
    program = """
import time
from ci.diagnostics import _mask_text
from skillevaluator.utils.secret_redaction import Redactor
started = time.monotonic()
redactor = Redactor(set())
for text in ('a' * 100_000, 'a-' * 50_000):
    assert _mask_text(text, redactor) == text
elapsed = time.monotonic() - started
assert elapsed < 5, f"Credential masking exceeded its 5-second budget: {elapsed:.3f}s"
print('unchanged')
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        # Imports can be slow on loaded CI hosts; the child still enforces the
        # original processing budget, while this watchdog bounds startup/hangs.
        timeout=30,
        check=True,
    )
    assert completed.stdout == "unchanged\n"


def test_only_the_first_failing_trial_of_each_job_and_four_per_skill_are_kept(tmp_path):
    jobs = tmp_path / "raw/harbor/run/_harbor-jobs"
    trials = {
        "skill-claude-with/case-1": True,
        "skill-claude-without/case-1": True,
        "skill-codex-with/case-1": False,
        "skill-codex-with/case-2": True,
        "skill-codex-with/case-3": True,
        "skill-codex-without/case-1": False,
        "skill-codex-without/case-2": False,
        "skill-opencode-with/case-1": True,
    }
    for name, failed in trials.items():
        trial = jobs / name
        (trial / "agent/setup").mkdir(parents=True)
        (trial / "trial.log").write_text(f"{name} started\n")
        (trial / "agent/setup/install-agent-runtime.failed.log").write_text("install failed\n")
        if failed:
            (trial / "exception.txt").write_text(f"{name} failed\n")
    output = tmp_path / "output"
    result = diagnostics.retain_diagnostics(tmp_path / "raw", output, set())
    kept = {item["path"].split("/")[3] + "/" + item["path"].split("/")[4] for item in result["retained"]}
    # Failing trials of every job come before the passing-only job that sorts earlier.
    assert kept == {
        "skill-claude-with/case-1",
        "skill-claude-without/case-1",
        "skill-codex-with/case-2",
        "skill-opencode-with/case-1",
    }
    assert len(result["retained"]) == 4 * 3  # trial.log, setup log and exception.txt each
    limited = {item["path"] for item in result["omitted"] if item["reason"] == "trial_limit"}
    assert limited == {
        "harbor/run/_harbor-jobs/skill-codex-with/case-1",
        "harbor/run/_harbor-jobs/skill-codex-with/case-3",
        "harbor/run/_harbor-jobs/skill-codex-without/case-1",
        "harbor/run/_harbor-jobs/skill-codex-without/case-2",
    }


def test_failing_jobs_that_sort_last_keep_their_trials_before_passing_jobs(tmp_path):
    jobs = tmp_path / "raw/harbor/run/_harbor-jobs"
    # Three agents with two arms each; only the agent that sorts last failed.
    for agent in ("claude-code", "codex", "opencode"):
        for arm in ("with", "without"):
            trial = jobs / f"skill-{agent}-{arm}/case-1"
            trial.mkdir(parents=True)
            (trial / "trial.log").write_text(f"{agent} {arm} started\n")
            if agent == "opencode":
                (trial / "exception.txt").write_text("opencode: agent installation failed\n")
    command = tmp_path / "skill-001.log"
    command.write_text("worker output\n")
    result = diagnostics.retain_diagnostics(tmp_path / "raw", tmp_path / "output", set(), command_log=command)
    assert [item["path"] for item in result["retained"]] == [
        "command.log",
        "harbor/run/_harbor-jobs/skill-claude-code-with/case-1/trial.log",
        "harbor/run/_harbor-jobs/skill-claude-code-without/case-1/trial.log",
        "harbor/run/_harbor-jobs/skill-opencode-with/case-1/exception.txt",
        "harbor/run/_harbor-jobs/skill-opencode-with/case-1/trial.log",
        "harbor/run/_harbor-jobs/skill-opencode-without/case-1/exception.txt",
        "harbor/run/_harbor-jobs/skill-opencode-without/case-1/trial.log",
    ]
    assert [item["path"] for item in result["omitted"] if item["reason"] == "trial_limit"] == [
        "harbor/run/_harbor-jobs/skill-codex-with/case-1",
        "harbor/run/_harbor-jobs/skill-codex-without/case-1",
    ]


def test_tier_budget_is_shared_across_skills_and_keeps_command_logs_first(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "MAX_TIER_FILES", 3)
    budget = diagnostics.TierBudget()
    results = []
    for index in (1, 2):
        raw = tmp_path / f"raw-{index}"
        trial = raw / "harbor/job/trial"
        trial.mkdir(parents=True)
        (trial / "trial.log").write_text("trial started\n")
        (trial / "exception.txt").write_text("startup failed\n")
        command = tmp_path / f"skill-{index}.log"
        command.write_text("worker failed\n")
        results.append(
            diagnostics.retain_diagnostics(raw, tmp_path / f"out-{index}", set(), command_log=command, budget=budget)
        )
    assert results[0]["retained"][0]["path"] == "command.log"
    assert len(results[0]["retained"]) == 3 and budget.files == 0
    assert results[1]["retained"] == []
    assert {item["reason"] for item in results[1]["omitted"]} == {"tier_limit"}
    assert (tmp_path / "out-2/diagnostics/index.json").is_file()

    monkeypatch.setattr(diagnostics, "MAX_TIER_BYTES", 20)
    budget = diagnostics.TierBudget()
    raw = tmp_path / "raw-3"
    raw.mkdir()
    (raw / "stderr.log").write_text("x" * 15 + "\n")
    (raw / "stdout.log").write_text("y" * 15 + "\n")
    result = diagnostics.retain_diagnostics(raw, tmp_path / "out-3", set(), budget=budget)
    assert [item["path"] for item in result["retained"]] == ["stderr.log"]
    assert result["omitted"] == [{"path": "stdout.log", "reason": "tier_limit"}]


def test_tier_caps_stay_within_what_the_feedback_reader_accepts():
    from ci import feedback

    # Each tier has at most one index.json per skill plus the Tier 2 catalog root.
    roots = feedback.MAX_SKILLS + 1
    files = diagnostics.MAX_TIER_FILES + roots
    # A GitLab archive also lists every directory of a retained path: diagnostics/,
    # up to MAX_DEPTH walked levels, and an agent/setup pair below the deepest one.
    members = 2 * files * (1 + 1 + diagnostics.MAX_DEPTH + 2)
    assert members <= feedback.MAX_DIAGNOSTIC_MEMBERS
    # Logs share the tier byte budget; an index.json stays below one file's size.
    assert 2 * (diagnostics.MAX_TIER_BYTES + roots * diagnostics.MAX_FILE_BYTES) <= feedback.MAX_DIAGNOSTIC_BYTES
    # Even incompressible logs from both native tiers leave half of the archive for reports.
    assert 2 * diagnostics.MAX_TIER_BYTES <= feedback.MAX_BYTES // 2
