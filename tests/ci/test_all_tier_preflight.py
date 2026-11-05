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
        [sys.executable, "-I", "-m", "skillevaluator.cli", "doctor", "--env-mode", "docker", "--agents", "codex"]
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
        [sys.executable, "-I", "-m", "skillevaluator.cli", "doctor", "--env-mode", "local", "--agents", "claude-code"],
    ]


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
    }
    assert gate.eligible(env, None)
    assert not gate.eligible({**env, "CI_COMMIT_REF_PROTECTED": "false"}, None)


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
