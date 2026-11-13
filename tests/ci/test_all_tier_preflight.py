# SPDX-License-Identifier: Apache-2.0
"""Exercise native bootstrap and GitLab admission without paid execution."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "ci" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_docker_bootstrap_uses_selected_mode_without_host_agent_install(monkeypatch):
    bootstrap = module("bootstrap_tiers")
    calls = []
    monkeypatch.setenv("SKILL_EVAL_ENV_MODE", "docker")
    monkeypatch.setenv("SKILL_EVAL_PROVIDER", "openai")
    monkeypatch.setenv("SKILL_EVAL_AGENTS", "codex")
    monkeypatch.setattr(
        bootstrap.subprocess, "run", lambda args, **_kwargs: calls.append(args) or subprocess.CompletedProcess(args, 0)
    )
    assert bootstrap.main() == 0
    assert calls == [
        [
            sys.executable,
            "-I",
            "-m",
            "skillevaluator.cli",
            "doctor",
            "--tier",
            "3",
            "--verify-models",
            "--env-mode",
            "docker",
            "--agents",
            "codex",
        ]
    ]


def test_bootstrap_rejects_cloud_before_install(monkeypatch):
    bootstrap = module("bootstrap_tiers")
    monkeypatch.setenv("SKILL_EVAL_ENV_MODE", "gke")
    monkeypatch.setattr(
        bootstrap.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("cloud mode must not install or run")
    )
    assert bootstrap.main() == 2


def test_local_bootstrap_installs_only_the_selected_native_runtime(monkeypatch):
    bootstrap = module("bootstrap_tiers")
    calls = []
    monkeypatch.setenv("SKILL_EVAL_ENV_MODE", "local")
    monkeypatch.setenv("SKILL_EVAL_PROVIDER", "anthropic")
    monkeypatch.delenv("SKILL_EVAL_AGENTS", raising=False)
    monkeypatch.setattr(bootstrap, "install_agent", calls.append)
    monkeypatch.setattr(
        bootstrap.subprocess, "run", lambda args, **_kwargs: calls.append(args) or subprocess.CompletedProcess(args, 0)
    )
    assert bootstrap.main() == 0
    assert calls == [
        "claude-code",
        [
            sys.executable,
            "-I",
            "-m",
            "skillevaluator.cli",
            "doctor",
            "--tier",
            "3",
            "--verify-models",
            "--env-mode",
            "local",
            "--agents",
            "claude-code",
        ],
    ]


def test_local_bootstrap_installs_the_opencode_release_docker_runs_pin(monkeypatch):
    from skillevaluator.tier3.harbor import runner

    bootstrap = module("bootstrap_tiers")
    calls = []
    monkeypatch.setattr(
        bootstrap.subprocess, "run", lambda args, **_kwargs: calls.append(args) or subprocess.CompletedProcess(args, 0)
    )
    bootstrap.install_agent("opencode")
    assert calls[0][-1] == f"opencode-ai@{runner.OPENCODE_VERSION}"
    assert runner._harbor_agent_kwargs("opencode", None, "docker") == {"version": runner.OPENCODE_VERSION}


def test_gitlab_admission_rejects_forks_and_stale_heads():
    gate = module("eligible_gitlab")
    env = {
        "CI_PROJECT_ID": "42",
        "CI_COMMIT_SHA": "a" * 40,
        "CI_PIPELINE_SOURCE": "merge_request_event",
        "CI_MERGE_REQUEST_IID": "1",
        "CI_MERGE_REQUEST_SOURCE_PROJECT_ID": "42",
    }
    current = {"source_project_id": 42, "target_project_id": 42, "state": "opened", "sha": "a" * 40}
    assert gate.eligible(env, current)
    assert not gate.eligible(env, {**current, "sha": "b" * 40})
    assert not gate.eligible(env, {**current, "source_project_id": 43})
    assert not gate.eligible(env, {**current, "state": "closed"})


def test_gitlab_admission_allows_only_protected_default_branch_without_mr():
    gate = module("eligible_gitlab")
    env = {
        "CI_PIPELINE_SOURCE": "web",
        "CI_COMMIT_BRANCH": "main",
        "CI_DEFAULT_BRANCH": "main",
        "CI_COMMIT_REF_PROTECTED": "true",
        "CI_COMMIT_SHA": "a" * 40,
    }
    current = {"name": "main", "protected": True, "commit": {"id": "a" * 40}}
    assert gate.eligible(env, current)
    assert not gate.eligible({**env, "CI_COMMIT_REF_PROTECTED": "false"}, current)


def test_actual_workflow_bootstrap_ignores_consumer_pythonpath(tmp_path):
    import os
    import shlex

    import yaml

    package = tmp_path / "skillevaluator"
    package.mkdir()
    marker = tmp_path / "consumer-imported"
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\nraise SystemExit(91)\n"
    )
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(tmp_path),
        "SKILL_EVAL_ENV_MODE": "gke",
    }
    workflow = yaml.load((ROOT / ".github/workflows/evaluate-all-tiers.yml").read_text(), Loader=yaml.BaseLoader)
    step = next(s for s in workflow["jobs"]["paid-tiers"]["steps"] if s.get("name", "").startswith("Bootstrap native"))
    argv = shlex.split(step["run"])
    argv[0] = sys.executable
    argv[-1] = str(ROOT / argv[-1].removeprefix("evaluator/"))
    # Establish that this consumer package exploits the old unisolated command.
    vulnerable = [arg for arg in argv if arg != "-I"]
    before = subprocess.run(vulnerable, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert before.returncode == 91
    assert marker.is_file()
    marker.unlink()
    after = subprocess.run(argv, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert after.returncode == 2  # Unsupported mode is handled by trusted code.
    assert "supports docker and local" in after.stdout
    assert not marker.exists()


def test_unsupported_bootstrap_mode_rejects_before_optional_runtime_imports(tmp_path):
    import os

    probe = """
import importlib.abc
import runpy
import sys

class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "skillevaluator.tier3" or fullname.startswith("skillevaluator.tier3."):
            raise RuntimeError("unsupported mode imported optional runtime: " + fullname)

sys.meta_path.insert(0, RejectRuntime())
runpy.run_path(sys.argv[1], run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(ROOT / "ci/bootstrap_tiers.py")],
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", ""), "SKILL_EVAL_ENV_MODE": "gke"},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2, result.stderr
    assert "supports docker and local" in result.stdout
    assert result.stderr == ""


def test_bootstrap_doctor_receives_the_configured_ci_provider_and_model(monkeypatch):
    bootstrap = module("bootstrap_tiers")
    calls = []
    monkeypatch.setenv("SKILL_EVAL_ENV_MODE", "docker")
    monkeypatch.setenv("SKILL_EVAL_PROVIDER", "nv_build")
    monkeypatch.setenv("SKILL_EVAL_MODEL", "nvidia/example-model")

    def capture(args, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", capture)
    assert bootstrap.main() == 0
    assert calls[0]["env"]["SKILL_EVAL_LLM_PROVIDER"] == "nv_build"
    assert calls[0]["env"]["SKILL_EVAL_LLM_MODEL"] == "nvidia/example-model"
    assert calls[0]["env"]["PYTHON_DOTENV_DISABLED"] == "1"


def test_failed_docker_doctor_prevents_native_provider_execution(tmp_path, monkeypatch):
    import json
    import os

    from ci import run_tiers

    workspace = tmp_path / "candidate"
    skill = workspace / "skills/example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: example\ndescription: Use for test examples.\n---\nInstructions.\n")
    subprocess.run(["git", "init", "-b", "main", str(workspace)], check=True, capture_output=True)
    git = ["git", "-C", str(workspace), "-c", "user.name=CI tests", "-c", "user.email=ci@example.com"]
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-m", "fixture"], check=True, capture_output=True)
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    commands = []

    def unavailable_doctor(args, **_kwargs):
        commands.append(args)
        assert "doctor" in args, "No provider or native evaluation may run after failed Docker readiness"
        return 1

    monkeypatch.setattr(run_tiers, "dataset_preflight", lambda _skill: {})
    monkeypatch.setattr(run_tiers, "run_command", unavailable_doctor)
    reports = tmp_path / "reports"
    code = run_tiers.evaluate(
        3,
        workspace,
        reports,
        {
            "PATH": os.environ["PATH"],
            "SKILL_EVAL_PATHS": "skills",
            "SKILL_EVAL_ENV_MODE": "docker",
            "SKILL_EVAL_PROVIDER": "nv_build",
            "NVIDIA_API_KEY": "test-key-value-123456789",
            "REPORT_COMMIT": commit,
            "REPORT_REPOSITORY": "example/consumer",
            "GITHUB_ACTIONS": "true",
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
        },
    )
    assert code == 0  # Advisory failures retain evidence, while spending no provider calls.
    assert len(commands) == 1
    assert commands[0][commands[0].index("--env-mode") + 1] == "docker"
    result = json.loads((reports / "tier3/ci-tier.json").read_text())
    assert result["tiers"][0]["status"] == "incomplete"


def test_provider_child_excludes_both_gitlab_read_and_feedback_credentials():
    from ci import run_tiers

    child = run_tiers.child_environment(
        {
            "SKILL_EVAL_PROVIDER": "nv_build",
            "SKILLEVALUATOR_GITLAB_READ_TOKEN": "read-only-test-token",
            "SKILLEVALUATOR_GITLAB_TOKEN": "feedback-test-token",
            "CI_JOB_TOKEN": "job-test-token",
        }
    )
    assert not any(
        key in child for key in ("SKILLEVALUATOR_GITLAB_READ_TOKEN", "SKILLEVALUATOR_GITLAB_TOKEN", "CI_JOB_TOKEN")
    )


def test_bootstrap_doctor_uses_isolated_tier3_gateway_and_native_model_choices(monkeypatch):
    bootstrap = module("bootstrap_tiers")
    captured = []
    monkeypatch.setenv("SKILL_EVAL_ENV_MODE", "docker")
    monkeypatch.setenv("SKILL_EVAL_PROVIDER", "nv_build")
    monkeypatch.setenv("SKILL_EVAL_TIER3_PROVIDER", "openai-compatible")
    monkeypatch.setenv("SKILL_EVAL_TIER3_MODEL", "catalog/judge")
    monkeypatch.setenv("SKILL_EVAL_TIER3_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", "gateway-secret")
    monkeypatch.setenv("SKILL_EVAL_AGENTS", "claude-code,codex")
    monkeypatch.setenv("SKILL_EVAL_AGENT_MODELS", "claude-code=aws/anthropic/opus,codex=openai/openai/gpt")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    monkeypatch.setenv("HOME", "/home/runner")

    def doctor(args, **kwargs):
        captured.append((args, kwargs["env"]))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(bootstrap.subprocess, "run", doctor)
    assert bootstrap.main() == 0
    args, child = captured[0]
    assert args[-4:] == ["--agent-model", "claude-code=aws/anthropic/opus", "--agent-model", "codex=openai/openai/gpt"]
    assert child["SKILL_EVAL_LLM_PROVIDER"] == "openai-compatible"
    assert child["SKILL_EVAL_LLM_MODEL"] == "catalog/judge"
    assert child["SKILL_EVAL_LLM_API_KEY"] == "gateway-secret"
    assert child["HOME"] == "/home/runner"
    assert not set(child) & {"GITHUB_TOKEN", "ANTHROPIC_API_KEY", "SKILL_EVAL_AGENT_MODELS"}
