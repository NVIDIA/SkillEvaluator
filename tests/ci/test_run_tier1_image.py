# SPDX-License-Identifier: Apache-2.0
"""Exercise the trusted image boundary without Docker or paid model calls."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

IMAGE = "ghcr.io/nvidia/skillevaluator-tier1@sha256:" + "a" * 64


def test_isolated_cli_help_uses_trusted_helpers_and_ignores_consumer_imports(tmp_path):
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    marker = tmp_path / "consumer-code-ran"
    trap = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\nraise RuntimeError('consumer import trap')\n"
    for filename in ("model_config.py", "select_changed_skills.py", "sitecustomize.py"):
        (consumer / filename).write_text(trap)
    (consumer / "skillevaluator").mkdir()
    (consumer / "skillevaluator" / "__init__.py").write_text(trap)
    launcher = Path(__file__).resolve().parents[2] / "ci" / "run_tier1_image.py"
    completed = subprocess.run(
        [sys.executable, "-I", str(launcher), "--help"],
        cwd=consumer,
        env={**os.environ, "PYTHONPATH": str(consumer)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--workspace" in completed.stdout
    assert "--in-image" in completed.stdout
    assert not marker.exists()


@pytest.fixture
def launch(tmp_path, monkeypatch):
    runner = importlib.import_module("ci.run_tier1_image")
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    native_run = subprocess.run

    def git(*args):
        return native_run(
            ["git", "-C", str(workspace), *args], text=True, capture_output=True, check=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    (workspace / "skills").mkdir()
    (workspace / "skills" / "SKILL.md").write_text("fixture")
    git("add", ".")
    git("commit", "-qm", "fixture")
    head = git("rev-parse", "HEAD")
    env = {
        "PATH": os.environ["PATH"],
        "REPORT_COMMIT": head,
        "DIFF_HEAD_SHA": head,
        "SKILL_EVAL_TIER1_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "selected-secret",
        "OPENAI_API_KEY": "unselected-secret",
        "GH_TOKEN": "publisher-secret",
        "CI_JOB_TOKEN": "gitlab-secret",
        "SKILL_EVAL_TIER1_MODEL": "claude-fixture",
        "PYTHONPATH": str(workspace),
        "LD_PRELOAD": "candidate.so",
    }
    monkeypatch.setattr(os, "environ", env)
    calls = []

    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    reports = tmp_path / "reports"
    args = ["--workspace", str(workspace), "--reports", str(reports), "--image", IMAGE]
    return runner, args, calls, env, workspace, reports, git


def test_docker_launch_is_hardened_and_maps_only_selected_key(launch):
    runner, args, calls, env, workspace, reports, _ = launch
    env.update(SKILL_EVAL_PATHS="skills", SKILL_EVAL_MIN_SCORE="81", SKILL_EVAL_SCAN_WORKERS="2")
    assert runner.main(args) == 0
    pull, run = calls
    assert pull[0][-2:] == ["pull", IMAGE]
    config = Path(pull[0][pull[0].index("--config") + 1])
    assert config != Path.home() / ".docker"
    assert not config.exists()
    assert "selected-secret" not in str(run[0])
    command, kwargs = run
    for flag in ("--read-only", "--cap-drop", "--security-opt", "--user"):
        assert flag in command
    assert "ALL" in command and "no-new-privileges" in command
    assert "/tmp:rw,nosuid,noexec,size=1g" in command
    assert "/audit-tmp:rw,nosuid,noexec,mode=1777,size=1g" in command
    assert f"type=bind,src={workspace},dst=/workspace,readonly" in command
    assert f"type=bind,src={reports},dst=/reports" in command
    assert command[-1] == IMAGE
    child = kwargs["env"]
    assert child["LLM_API_KEY"] == "selected-secret"
    assert child["LLM_PROVIDER"] == "anthropic"
    assert child["LLM_MODEL"] == "claude-fixture"
    assert child["LLM_ENABLED"] == "true"
    assert child["SCAN_MODE"] == "all"
    assert child["MIN_SCORE"] == "81" and child["SCAN_WORKERS"] == "2"
    assert (
        not {"GH_TOKEN", "CI_JOB_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PYTHONPATH", "LD_PRELOAD"}
        & child.keys()
    )
    assert "LLM_API_KEY" not in pull[1]["env"]


@pytest.mark.parametrize("mutation", ["missing-key", "wrong-head", "conflicting-head", "gateway", "base-url", "static"])
def test_invalid_llm_request_never_launches_or_exposes_keys(launch, mutation, capsys):
    runner, args, calls, env, *_ = launch
    if mutation == "missing-key":
        del env["ANTHROPIC_API_KEY"]
    elif mutation == "wrong-head":
        env["REPORT_COMMIT"] = env["DIFF_HEAD_SHA"] = "b" * 40
    elif mutation == "conflicting-head":
        env["DIFF_HEAD_SHA"] = "b" * 40
    elif mutation == "gateway":
        env.update(
            SKILL_EVAL_TIER1_PROVIDER="openai-compatible",
            SKILL_EVAL_TIER1_BASE_URL="https://example.invalid/v1",
            SKILL_EVAL_LLM_API_KEY="gateway-secret",
        )
    elif mutation == "base-url":
        env["SKILL_EVAL_TIER1_BASE_URL"] = "https://example.invalid/v1"
    else:
        env["SKILL_EVAL_TIER1_PROVIDER"] = "static"
    assert runner.main(args) == 1
    assert not calls
    output = capsys.readouterr()
    assert "secret" not in output.err + output.out


@pytest.mark.parametrize("unsafe", ["inside", "parent", "symlink", "comma", "workspace-link"])
def test_unsafe_mount_paths_fail_before_launch(launch, unsafe):
    runner, args, calls, _, workspace, reports, _ = launch
    if unsafe == "inside":
        args[3] = str(workspace / "reports")
    elif unsafe == "parent":
        args[3] = str(workspace.parent)
    elif unsafe == "symlink":
        reports.symlink_to(workspace.parent, target_is_directory=True)
    elif unsafe == "comma":
        args[3] = str(reports.with_name("report,unsafe"))
    else:
        link = workspace.with_name("linked")
        link.symlink_to(workspace, target_is_directory=True)
        args[1] = str(link)
    assert runner.main(args) == 1
    assert not calls


def test_in_image_invokes_installed_runner_outside_candidate(launch):
    runner, args, calls, env, workspace, reports, _ = launch
    env["PATH"] = str(workspace) + os.pathsep + env["PATH"]
    assert runner.main([*args, "--in-image"]) == 0
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == [
        "/opt/skillevaluator/.venv/bin/python",
        "-E",
        "-s",
        "/opt/skillevaluator/ci/run_tier1.py",
        "--workspace",
        str(workspace),
        "--reports",
        str(reports),
    ]
    assert kwargs["cwd"] == "/opt/skillevaluator"
    assert kwargs["env"]["LLM_API_KEY"] == "selected-secret"
    assert "PYTHONPATH" not in kwargs["env"]
    assert str(workspace) not in kwargs["env"]["PATH"]


def test_gateway_rejection_explains_immutable_image_limitation_without_endpoint(launch, capsys):
    runner, args, calls, env, *_ = launch
    env["SKILL_EVAL_TIER1_PROVIDER"] = "openai-compatible"
    assert runner.main(args) == 1
    assert not calls
    assert "default endpoints only" in capsys.readouterr().out


@pytest.mark.parametrize("provider,key", [("openai", "OPENAI_API_KEY"), ("nv_build", "NVIDIA_API_KEY")])
def test_supported_provider_selects_its_own_credential(launch, provider, key):
    runner, args, calls, env, *_ = launch
    env.update(SKILL_EVAL_TIER1_PROVIDER=provider)
    env[key] = "provider-credential"
    assert runner.main(args) == 0
    assert calls[-1][1]["env"]["LLM_PROVIDER"] == provider
    assert calls[-1][1]["env"]["LLM_API_KEY"] == "provider-credential"


def test_native_tier1_launch_does_not_require_the_tier2_gateway_route(launch):
    runner, args, calls, env, *_ = launch
    # The GitHub Tier 1 job intentionally does not receive Tier 2's gateway URL.
    env.update(
        SKILL_EVAL_PROVIDER="openai-compatible",
        SKILL_EVAL_TIER1_PROVIDER="openai",
        SKILL_EVAL_TIER1_MODEL="",
        OPENAI_API_KEY="selected-tier1-secret",
    )
    assert runner.main(args) == 0
    child = calls[-1][1]["env"]
    assert child["LLM_PROVIDER"] == "openai"
    assert child["LLM_API_KEY"] == "selected-tier1-secret"


@pytest.mark.parametrize("mutation", ["dirty", "missing-base", "untrusted-image"])
def test_unverified_inputs_fail_before_credential_launch(launch, mutation):
    runner, args, calls, env, workspace, *_ = launch
    if mutation == "dirty":
        (workspace / "skills" / "SKILL.md").write_text("modified")
    elif mutation == "missing-base":
        env["GITHUB_EVENT_NAME"] = "pull_request"
    else:
        args[-1] = "ghcr.io/attacker/skillevaluator-tier1@sha256:" + "a" * 64
    assert runner.main(args) == 1
    assert not calls


@pytest.mark.parametrize(
    "event", [{"GITHUB_EVENT_NAME": "pull_request"}, {"CI_PIPELINE_SOURCE": "merge_request_event"}]
)
def test_auto_changes_on_pr_and_mr_but_setup_changes_force_all(launch, event):
    runner, args, calls, env, workspace, _, git = launch
    base = env["REPORT_COMMIT"]
    (workspace / "skills" / "SKILL.md").write_text("changed")
    git("commit", "-qam", "skill")
    env.update(
        event, DIFF_BASE_SHA=base, REPORT_COMMIT=git("rev-parse", "HEAD"), DIFF_HEAD_SHA=git("rev-parse", "HEAD")
    )
    assert runner.main(args) == 0
    assert calls[-1][1]["env"]["SCAN_MODE"] == "changed"
    (workspace / ".github").mkdir()
    (workspace / ".github" / "config.yml").write_text("changed")
    git("add", ".")
    git("commit", "-qm", "setup")
    env.update(REPORT_COMMIT=git("rev-parse", "HEAD"), DIFF_HEAD_SHA=git("rev-parse", "HEAD"))
    assert runner.main(args) == 0
    assert calls[-1][1]["env"]["SCAN_MODE"] == "all"


def test_runtime_exceptions_are_fixed_diagnostics(launch, monkeypatch, capsys):
    runner, args, *_ = launch

    def fail(*args, **kwargs):
        raise RuntimeError("selected-secret and consumer-controlled text")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    assert runner.main(args) == 1
    assert "selected-secret" not in str(capsys.readouterr())


def test_reviewed_baselines_reach_the_enriched_scan_only_when_opted_in(launch):
    """The enriched evidence replaces the keyless gate, so it applies the same baselines."""
    runner, args, calls, env, *_ = launch
    env.update(SECURITY_BASELINES="true", POLICY_SOURCE_SHA="c" * 40)
    assert runner.main(args) == 0
    command, kwargs = calls[-1]
    assert kwargs["env"]["SECURITY_BASELINES"] == "true"
    assert kwargs["env"]["POLICY_SOURCE_SHA"] == "c" * 40
    assert {"SECURITY_BASELINES", "POLICY_SOURCE_SHA"} <= {
        command[index + 1] for index, arg in enumerate(command) if arg == "--env"
    }
    calls.clear()
    env.update(SECURITY_BASELINES="false")
    assert runner.main(args) == 0
    child = calls[-1][1]["env"]
    assert child["SECURITY_BASELINES"] == "false"
    assert "POLICY_SOURCE_SHA" not in child  # An inherited policy commit is never forwarded.
    calls.clear()
    del env["SECURITY_BASELINES"]
    assert runner.main(args) == 0
    assert calls[-1][1]["env"]["SECURITY_BASELINES"] == "false"


@pytest.mark.parametrize(
    "baselines",
    [
        {"SECURITY_BASELINES": "TRUE"},
        {"SECURITY_BASELINES": "true"},
        {"SECURITY_BASELINES": "true", "POLICY_SOURCE_SHA": "main"},
    ],
)
def test_invalid_baseline_opt_in_fails_before_credential_launch(launch, baselines, capsys):
    runner, args, calls, env, *_ = launch
    env.pop("POLICY_SOURCE_SHA", None)
    env.update(baselines)
    assert runner.main(args) == 1
    assert not calls
    output = capsys.readouterr()
    assert "secret" not in output.err + output.out


def _setup_only_change(env, workspace, git):
    base = env["REPORT_COMMIT"]
    (workspace / ".github").mkdir()
    (workspace / ".github" / "config.yml").write_text("changed")
    (workspace / "README.md").write_text("changed")
    git("add", ".")
    git("commit", "-qm", "setup only")
    head = git("rev-parse", "HEAD")
    env.update(DIFF_BASE_SHA=base, REPORT_COMMIT=head, DIFF_HEAD_SHA=head)


@pytest.mark.parametrize(
    "event", [{"GITHUB_EVENT_NAME": "pull_request"}, {"CI_PIPELINE_SOURCE": "merge_request_event"}]
)
def test_pr_without_a_skill_change_skips_before_credentials(launch, event, capsys):
    """Setup-only or unrelated changes leave the keyless scan as the gate."""
    runner, args, calls, env, workspace, reports, git = launch
    _setup_only_change(env, workspace, git)
    del env["ANTHROPIC_API_KEY"]  # A skip never needs, reads or exposes a key.
    env.update(event)
    assert runner.main(args) == 0
    assert not calls
    assert not reports.exists()  # No evidence or marker that could be mistaken for a result.
    assert "keyless Tier 1 scan remains the gate" in capsys.readouterr().out


def test_explicit_full_scan_does_not_skip_without_a_skill_change(launch):
    runner, args, calls, env, workspace, _, git = launch
    _setup_only_change(env, workspace, git)
    env.update(GITHUB_EVENT_NAME="pull_request", SKILL_EVAL_SCAN_MODE="all")
    assert runner.main(args) == 0
    assert calls[-1][1]["env"]["SCAN_MODE"] == "all"


def test_skills_changed_cli_matches_the_launcher_decision(launch, capsys):
    """The GitLab aggregate and paid tiers recompute the skip with this exact command."""
    _, _, calls, env, workspace, _, git = launch
    selector = importlib.import_module("ci.select_changed_skills")
    cli = ["--skills-changed", "--workspace", str(workspace)]
    assert selector.main(cli) == 0
    assert capsys.readouterr().out == "true\n"  # Default-branch and manual runs keep the enriched scan.
    _setup_only_change(env, workspace, git)
    env["CI_PIPELINE_SOURCE"] = "merge_request_event"
    assert selector.main(cli) == 0
    assert capsys.readouterr().out == "false\n"
    env["SKILL_EVAL_SCAN_MODE"] = "all"
    assert selector.main(cli) == 0
    assert capsys.readouterr().out == "true\n"
    del env["SKILL_EVAL_SCAN_MODE"]
    (workspace / "skills" / "SKILL.md").write_text("changed again")
    git("commit", "-qam", "skill")
    env.update(REPORT_COMMIT=git("rev-parse", "HEAD"), DIFF_HEAD_SHA=git("rev-parse", "HEAD"))
    assert selector.main(cli) == 0
    assert capsys.readouterr().out == "true\n"
    env["DIFF_BASE_SHA"] = "not-a-commit"
    assert selector.main(cli) == 2
    assert not calls


def test_committed_submodule_ignore_cannot_hide_a_skill_change(launch, capsys):
    """A gitlink update under a skill root still needs the enriched scan."""
    _, _, _, env, workspace, _, git = launch
    selector = importlib.import_module("ci.select_changed_skills")
    base = env["REPORT_COMMIT"]
    (workspace / ".gitmodules").write_text(
        '[submodule "vendor"]\n\tpath = skills/vendor\n\turl = ./vendor\n\tignore = all\n'
    )
    git("add", ".gitmodules")
    git("update-index", "--add", "--cacheinfo", f"160000,{'b' * 40},skills/vendor")
    git("commit", "-qm", "vendor skill")
    head = git("rev-parse", "HEAD")
    env.update(CI_PIPELINE_SOURCE="merge_request_event", DIFF_BASE_SHA=base, REPORT_COMMIT=head, DIFF_HEAD_SHA=head)
    assert selector.main(["--skills-changed", "--workspace", str(workspace)]) == 0
    assert capsys.readouterr().out == "true\n"
