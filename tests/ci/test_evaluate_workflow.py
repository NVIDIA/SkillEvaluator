# SPDX-License-Identifier: Apache-2.0
"""Guard selection wiring and execute the consumer shell's Docker contract."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]


def _job():
    return yaml.load((ROOT / ".github/workflows/evaluate-skills.yml").read_text(), Loader=yaml.BaseLoader)["jobs"][
        "tier1"
    ]


def test_pr_history_mode_and_attempt_artifact_wiring():
    job = _job()
    checkout = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    scan = next(step for step in job["steps"] if step.get("name") == "Evaluate selected skills")
    env = scan["env"]
    assert checkout["with"]["fetch-depth"] == "0"
    assert checkout["with"]["ref"] == env["DIFF_HEAD_SHA"] == "${{ github.event.pull_request.head.sha || github.sha }}"
    assert env["DIFF_BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    assert (
        env["SCAN_MODE"]
        == "${{ inputs.scan-mode == 'auto' && (github.event_name == 'pull_request' && 'changed' || 'all') || inputs.scan-mode }}"
    )
    assert env["LLM_API_KEY"] == "${{ inputs.llm && secrets.LLM_API_KEY || '' }}"
    assert env["CALLER_WORKFLOW_SHA"] == "${{ github.workflow_sha }}"
    upload = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["with"]["name"] == "${{ inputs.artifact-name }}-attempt-${{ github.run_attempt }}"
    assert int(job["timeout-minutes"]) > 50


@pytest.mark.parametrize("llm", ["true", "false"])
def test_real_workflow_shell_preserves_container_boundary_and_secret_selection(tmp_path, llm):
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
        "GITHUB_WORKSPACE": str(tmp_path / "checkout"),
        "IMAGE": "ghcr.io/example/scanner@sha256:" + "a" * 64,
        "REPORT_BASE_URL": "",
        "LLM_ENABLED": llm,
        "LLM_API_KEY": "dummy-do-not-pass-in-static-mode",
    }
    subprocess.run(["bash", "-e", "-o", "pipefail", "-c", scan["run"]], env=env, check=True, capture_output=True)
    args = json.loads(captured.read_text())
    passed_environment = [args[index + 1] for index, arg in enumerate(args) if arg == "--env"]
    assert {"SCAN_MODE", "DIFF_BASE_SHA", "DIFF_HEAD_SHA"}.issubset(passed_environment)
    assert ("LLM_API_KEY" in args) is (llm == "true")
    for flag in ("--read-only", "--user", "--cap-drop", "--security-opt"):
        assert flag in args
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert args[args.index("--security-opt") + 1] == "no-new-privileges"
    assert any("dst=/workspace,readonly" in arg for arg in args)
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
