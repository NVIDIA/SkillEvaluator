# SPDX-License-Identifier: Apache-2.0
"""Check that GitHub Tier 2 can use immutable CI without host runtime dependencies."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "ghcr.io/nvidia/skillevaluator-tier2@sha256:" + "a" * 64
REVISION = "b" * 40


def test_isolated_launcher_uses_only_the_standard_library(tmp_path):
    marker = tmp_path / "untrusted-import"
    (tmp_path / "model_config.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(ROOT / "ci/run_tier2_image.py"), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--image" in result.stdout
    assert not marker.exists()


@pytest.fixture
def launch(tmp_path, monkeypatch):
    runner = importlib.import_module("ci.run_tier2_image")
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    reports = tmp_path / "reports"
    git = ["git", "-C", str(workspace), "-c", "user.name=CI", "-c", "user.email=ci@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "--allow-empty", "-qm", "fixture"], check=True)
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    env = {
        "PATH": os.environ["PATH"],
        "REPORT_COMMIT": head,
        "DIFF_HEAD_SHA": head,
        "DIFF_BASE_SHA": "d" * 40,
        "REPORT_REPOSITORY": "nvidia/test-skills",
        "GITHUB_RUN_ID": "1234",
        "GITHUB_RUN_ATTEMPT": "2",
        "SKILLEVALUATOR_TOOLS_REF": REVISION,
        "SKILL_EVAL_PROVIDER": "anthropic",
        "SKILL_EVAL_MODEL": "chat-fixture",
        "SKILL_EVAL_EMBEDDING_PROVIDER": "",
        "SKILL_EVAL_EMBEDDING_MODEL": "embedding-fixture",
        "SKILL_EVAL_PATHS": "skills\nextra-skills",
        "SKILL_EVAL_SCAN_MODE": "auto",
        "SKILL_EVAL_TIMEOUT_SECONDS": "50",
        "SKILL_EVAL_TIER3_MODEL": "tier3-fixture",
        "ANTHROPIC_API_KEY": "chat-secret",
        "OPENAI_API_KEY": "embedding-secret",
        "NVIDIA_API_KEY": "unselected-secret",
        "SKILL_EVAL_LLM_API_KEY": "gateway-secret",
        "GH_TOKEN": "repository-secret",
        "DOCKER_HOST": "tcp://untrusted.invalid:2375",
        "PYTHONPATH": str(workspace),
    }
    monkeypatch.setattr(os, "environ", env)
    calls = []

    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        stdout = f"{REVISION}\n" if argv[3:5] == ["image", "inspect"] else None
        return subprocess.CompletedProcess(argv, 0, stdout=stdout)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    args = ["--workspace", str(workspace), "--reports", str(reports), "--image", IMAGE]
    return runner, args, calls, env, workspace, reports


def test_verified_image_gets_only_selected_credentials_and_report_context(launch):
    runner, args, calls, source, workspace, reports = launch
    assert runner.main(args) == 0
    pull, inspect, run = calls
    assert pull[0][-2:] == ["pull", IMAGE]
    assert inspect[0][3:5] == ["image", "inspect"]
    assert inspect[0][-1] == IMAGE
    assert pull[1]["env"] == inspect[1]["env"] == {"PATH": source["PATH"]}
    assert not Path(pull[0][2]).exists()
    command, kwargs = run
    assert command[-5:] == [IMAGE, "--workspace", "/workspace", "--reports", "/reports"]
    assert "--user" in command and f"{os.getuid()}:{os.getgid()}" in command
    assert "--pull=never" in command
    assert "--read-only" in command and "--cap-drop" in command and "ALL" in command
    assert "no-new-privileges" in command and any("noexec" in arg for arg in command)
    assert f"type=bind,src={workspace},dst=/workspace,readonly" in command
    assert f"type=bind,src={reports},dst=/reports" in command
    child = kwargs["env"]
    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "REPORT_COMMIT",
        "DIFF_BASE_SHA",
        "REPORT_REPOSITORY",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "SKILLEVALUATOR_TOOLS_REF",
        "SKILL_EVAL_MODEL",
        "SKILL_EVAL_EMBEDDING_MODEL",
        "SKILL_EVAL_PATHS",
        "SKILL_EVAL_SCAN_MODE",
        "SKILL_EVAL_TIMEOUT_SECONDS",
    ):
        assert key in command and child[key] == source[key]
    assert not set(child) & {
        "NVIDIA_API_KEY",
        "SKILL_EVAL_LLM_API_KEY",
        "SKILL_EVAL_TIER3_MODEL",
        "GH_TOKEN",
        "DOCKER_HOST",
        "PYTHONPATH",
    }
    assert not any("secret" in arg for arg in command)


def test_compatible_embeddings_can_receive_gateway_fallback_without_gateway_chat_key(launch):
    runner, args, calls, source, *_ = launch
    source.update(SKILL_EVAL_EMBEDDING_PROVIDER="openai-compatible", SKILL_EVAL_EMBEDDING_BASE_URL="https://x.test/v1")
    assert runner.main(args) == 0
    child = calls[-1][1]["env"]
    assert child["SKILL_EVAL_EMBEDDING_API_KEY"] == "gateway-secret"
    assert "SKILL_EVAL_LLM_API_KEY" not in child and "OPENAI_API_KEY" not in child


def test_inference_proxy_settings_reach_only_the_container_run(launch):
    runner, args, calls, source, *_ = launch
    proxy_settings = {
        "HTTP_PROXY": "http://proxy.example:3128",
        "HTTPS_PROXY": "http://proxy.example:3128",
        "NO_PROXY": "127.0.0.1,internal.example",
        "http_proxy": "http://proxy.example:3128",
        "https_proxy": "http://proxy.example:3128",
        "no_proxy": "127.0.0.1,internal.example",
    }
    source.update(proxy_settings)
    assert runner.main(args) == 0
    pull, inspect, (command, run) = calls
    for key, value in proxy_settings.items():
        assert key not in pull[1]["env"] and key not in inspect[1]["env"]
        assert key in command and value not in command
        assert run["env"][key] == value


def test_unmatched_oci_revision_never_runs_the_image(launch, monkeypatch, capsys):
    runner, args, calls, _, *_ = launch
    execute = runner.subprocess.run

    def mismatched(argv, **kwargs):
        result = execute(argv, **kwargs)
        if argv[3:5] == ["image", "inspect"]:
            result.stdout = "f" * 40 + "\n"
        return result

    monkeypatch.setattr(runner.subprocess, "run", mismatched)
    assert runner.main(args) == 1
    assert len(calls) == 2
    assert "tools-ref" in capsys.readouterr().out


@pytest.mark.parametrize("invalid", ["tag", "registry", "head", "reports-in-checkout", "mount-delimiter"])
def test_invalid_inputs_never_pull_an_image(launch, invalid):
    runner, args, calls, source, workspace, reports = launch
    if invalid == "tag":
        args[-1] = IMAGE.split("@", 1)[0] + ":latest"
    elif invalid == "registry":
        args[-1] = IMAGE.replace("nvidia/", "untrusted/")
    elif invalid == "head":
        source["REPORT_COMMIT"] = source["DIFF_HEAD_SHA"] = "f" * 40
    elif invalid == "reports-in-checkout":
        args[3] = str(workspace / "reports")
    elif invalid == "mount-delimiter":
        args[3] = str(reports.with_name('reports,"other"'))
    assert runner.main(args) == 1
    assert not calls


@pytest.mark.parametrize(("tier", "image"), [("2", IMAGE), ("2", ""), ("3", "")])
def test_github_workflow_selects_the_runner_inside_the_protected_tier_path(tier, image):
    workflow = yaml.load((ROOT / ".github/workflows/evaluate-all-tiers.yml").read_text(), Loader=yaml.BaseLoader)
    assert workflow["on"]["workflow_call"]["inputs"]["tier2-image"]["default"] == ""
    jobs = workflow["jobs"]
    paid = jobs["paid-tiers"]
    steps = paid["steps"]
    assert paid["continue-on-error"] == "true"
    assert "needs.eligibility.outputs.allowed == 'true'" in paid["if"]
    assert "needs.eligibility.outputs.skills-changed == 'true'" in paid["if"]
    assert "contains(format(',{0},', inputs.tiers), ',1,')" in jobs["tier1"]["if"]
    candidate = next(s for s in steps if s.get("with", {}).get("path") == "candidate")
    assert candidate["with"]["fetch-depth"] == "0"
    assert "matrix.tier != 2 || inputs.tier2-image == ''" in candidate["with"]["filter"]
    install = next(s for s in steps if s.get("name", "").startswith("Install only"))
    assert "matrix.tier != 2 || inputs.tier2-image == ''" in install["if"]
    current_head = next(s for s in steps if s.get("id") == "current-head")
    evaluate = next(s for s in steps if "run_tier2_image.py" in s.get("run", ""))
    assert steps.index(evaluate) > steps.index(current_head)
    assert "steps.current-head.outputs.allowed == 'true'" in evaluate["if"]
    assert "contains(format(',{0},', inputs.tiers), format(',{0},', matrix.tier))" in evaluate["if"]
    assert evaluate["env"]["TIER2_IMAGE"] == "${{ matrix.tier == 2 && inputs.tier2-image || '' }}"
    upload = next(s for s in steps if s.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["with"]["name"] == "skill-evaluator-tier${{ matrix.tier }}-attempt-${{ github.run_attempt }}"
    prefix = 'python() { printf "%s\\n" "$SKILL_EVAL_TIMEOUT_SECONDS" "$@"; }\n'
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", prefix + evaluate["run"]],
        env={
            "PATH": os.defpath,
            "TIER": tier,
            "TIER2_IMAGE": image,
            "SKILL_EVAL_TIMEOUT_MINUTES": "60",
            "GITHUB_WORKSPACE": "/path with spaces/workspace",
            "RUNNER_TEMP": "/path with spaces/temp",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    launcher = (
        ["evaluator/ci/run_tier2_image.py", "--image", image]
        if image
        else ["evaluator/ci/run_tiers.py", "evaluate", "--tier", tier]
    )
    assert result.stdout.splitlines() == [
        "2700",
        "-I",
        *launcher,
        "--workspace",
        "/path with spaces/workspace/candidate",
        "--reports",
        "/path with spaces/temp/skill-reports",
    ]
