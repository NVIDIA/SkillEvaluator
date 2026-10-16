# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor processes must not read operator dotenv files or depend on the caller's directory."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor import runner

# Replays the loaders Harbor runs: the jobs CLI loads ``<cwd>/.env.local``, while
# Harbor's registry client and LiteLLM call ``load_dotenv()`` on import.
_PROBE = """
import json
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

if (Path.cwd() / ".env.local").exists():
    load_dotenv(Path.cwd() / ".env.local", override=False)
load_dotenv()
load_dotenv(find_dotenv(usecwd=True))
Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "cwd": str(Path.cwd()),
            "cwd_entries": sorted(path.name for path in Path.cwd().iterdir()),
            "sentinel": os.environ.get("SE_DOTENV_SENTINEL"),
            "docker_host": os.environ.get("DOCKER_HOST"),
            "dotenv_disabled": os.environ.get("PYTHON_DOTENV_DISABLED"),
            "telemetry": os.environ.get("HARBOR_TELEMETRY"),
        }
    ),
    encoding="utf-8",
)
print("Harbor job did not complete successfully: 1 errored")
sys.exit(3)
"""


def _operator_directory(tmp_path: Path) -> Path:
    operator_dir = tmp_path / "operator"
    operator_dir.mkdir()
    for name in (".env.local", ".env"):
        (operator_dir / name).write_text(
            "SE_DOTENV_SENTINEL=leaked\nDOCKER_HOST=tcp://attacker.invalid:2375\n",
            encoding="utf-8",
        )
    return operator_dir


def test_harbor_run_ignores_operator_dotenv_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    operator_dir = _operator_directory(tmp_path)
    monkeypatch.chdir(operator_dir)
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE, encoding="utf-8")
    observed = tmp_path / "observed.json"
    monkeypatch.setattr(
        runner, "build_harbor_run_command", lambda **_kwargs: [sys.executable, str(probe), str(observed)]
    )

    ok, detail = runner._run_harbor(
        dataset=tmp_path / "dataset",
        agent="codex",
        job_name="dotenv-isolation",
        env_mode="docker",
        model="model",
        jobs_dir=tmp_path / "jobs",
        run_env={
            "DOCKER_HOST": "unix:///operator/docker.sock",
            "FEATURE_FLAG": "1",
            "HARBOR_TELEMETRY": "1",
            "PYTHON_DOTENV_DISABLED": "0",
        },
        n_attempts=1,
        n_concurrent=1,
        timeout_multiplier=1.0,
        override_cpus=None,
        override_memory_mb=None,
        override_storage_mb=None,
    )

    assert ok is False
    # Short child values are not exact secrets, so counts survive redaction.
    assert "1 errored" in detail
    child = json.loads(observed.read_text(encoding="utf-8"))
    assert child["sentinel"] is None
    assert child["docker_host"] == "unix:///operator/docker.sock"
    assert child["dotenv_disabled"] == "1"
    assert child["telemetry"] == "0"
    assert Path(child["cwd"]) != operator_dir
    assert child["cwd_entries"] == []
    assert not Path(child["cwd"]).exists()


def test_disabled_dotenv_blocks_files_even_in_the_operator_directory(tmp_path: Path) -> None:
    operator_dir = _operator_directory(tmp_path)
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE, encoding="utf-8")
    observed = tmp_path / "observed.json"

    result = runner._run_bounded_harbor_process(
        [sys.executable, str(probe), str(observed)],
        env={"PATH": os.environ.get("PATH", "")},
        cwd=operator_dir,
        stdin_text=None,
        timeout_seconds=60,
        max_output_bytes=64 * 1024,
        diagnostic_tail_chars=4096,
        secret_values=set(),
    )

    assert result.returncode == 3
    child = json.loads(observed.read_text(encoding="utf-8"))
    assert child["sentinel"] is None
    assert child["docker_host"] is None


def test_harbor_command_and_environment_anchor_relative_host_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    command = runner.build_harbor_run_command(
        dataset_path="relative/dataset",
        agent="codex",
        job_name="paths",
        env_mode="ec2",
        jobs_dir=Path("relative/jobs"),
        environment_kwargs={
            "region": "us-west-2",
            "ssh_key_path": "keys/id_ed25519",
            "ssh_known_hosts_path": "~/.ssh/known_hosts",
        },
    )

    assert command[command.index("-p") + 1] == str(tmp_path / "relative" / "dataset")
    assert command[command.index("--jobs-dir") + 1] == str(tmp_path / "relative" / "jobs")
    encoded = [command[index + 1] for index, value in enumerate(command) if value == "--ek"]
    assert f"ssh_key_path={json.dumps(str(tmp_path / 'keys' / 'id_ed25519'))}" in encoded
    assert 'ssh_known_hosts_path="~/.ssh/known_hosts"' in encoded

    launch_env = runner._harbor_launch_environment(
        {
            "KUBECONFIG": os.pathsep.join(["kube/config", "~/.kube/config", "/etc/kube/config"]),
            "TMPDIR": "scratch",
            "DOCKER_HOST": "unix:///var/run/docker.sock",
        }
    )
    assert launch_env["KUBECONFIG"].split(os.pathsep) == [
        str(tmp_path / "kube" / "config"),
        "~/.kube/config",
        "/etc/kube/config",
    ]
    assert launch_env["TMPDIR"] == str(tmp_path / "scratch")
    assert launch_env["DOCKER_HOST"] == "unix:///var/run/docker.sock"


@pytest.mark.parametrize(("installed", "rejected"), [("1.1.1", True), ("1.2.0", False), ("1.2.2", False)])
def test_prerequisites_require_dotenv_disable_support(
    monkeypatch: pytest.MonkeyPatch,
    installed: str,
    rejected: bool,
) -> None:
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda _name: installed)

    error = runner._python_dotenv_prerequisite_error()

    assert (error is not None) is rejected
    if rejected:
        assert "1.2.0" in error
