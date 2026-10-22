# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A separately installed custom execution backend, exercised from fresh processes.

The registry unit tests patch ``entry_points``; this test builds a real wheel for
an independent package, installs it into its own directory, and checks
discovery, ``doctor``, and Harbor's own import-path dispatch in new interpreters.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

_DIST = "skillevaluator-demo-backend"
_PACKAGE = "skillevaluator_demo_backend"
_VERSION = "0.1.0"
_MODULES = {
    f"{_PACKAGE}/__init__.py": '"""Demo custom execution backend."""\n',
    f"{_PACKAGE}/spec.py": textwrap.dedent(
        """
        from skillevaluator.tier3_environments import EnvironmentBackendSpec

        SPEC = EnvironmentBackendSpec(
            import_path="skillevaluator_demo_backend.environment:DemoEnvironment",
            environment_kwargs=frozenset({"endpoint", "lease_seconds"}),
            required_environment_kwargs=("endpoint",),
            host_env_vars=frozenset({"DEMO_BACKEND_URL"}),
            prerequisites="skillevaluator_demo_backend.checks:prerequisites",
            install_hint="Install skillevaluator-demo-backend.",
        )
        """
    ),
    f"{_PACKAGE}/checks.py": textwrap.dedent(
        """
        def prerequisites(environment_kwargs):
            if environment_kwargs.get("endpoint") == "https://down.example":
                return ["demo backend endpoint is down"]
            return []
        """
    ),
    f"{_PACKAGE}/environment.py": textwrap.dedent(
        """
        from harbor.environments.base import BaseEnvironment


        class DemoEnvironment(BaseEnvironment):
            def __init__(self, *args, endpoint, lease_seconds=600, **kwargs):
                self.endpoint = endpoint
                self.lease_seconds = lease_seconds
                super().__init__(*args, **kwargs)

            @staticmethod
            def type():
                return "demo-backend"

            @classmethod
            def preflight(cls):
                return None

            def _validate_definition(self):
                return None

            async def start(self, force_build):
                raise NotImplementedError

            async def stop(self, delete):
                raise NotImplementedError

            async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
                raise NotImplementedError

            async def upload_file(self, source_path, target_path):
                raise NotImplementedError

            async def upload_dir(self, source_dir, target_dir):
                raise NotImplementedError

            async def download_file(self, source_path, target_path):
                raise NotImplementedError

            async def download_dir(self, source_dir, target_dir):
                raise NotImplementedError
        """
    ),
}


def _record_line(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def _build_wheel(destination: Path) -> Path:
    """Write a standards-conforming pure-Python wheel for the demo backend."""
    info = f"{_PACKAGE}-{_VERSION}.dist-info"
    files = {name: content.encode() for name, content in _MODULES.items()}
    files[f"{info}/METADATA"] = f"Metadata-Version: 2.1\nName: {_DIST}\nVersion: {_VERSION}\n".encode()
    files[f"{info}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: skillevaluator-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    files[f"{info}/entry_points.txt"] = f"[skillevaluator.env_modes]\ndemo-backend = {_PACKAGE}.spec:SPEC\n".encode()
    record = [_record_line(name, data) for name, data in files.items()] + [f"{info}/RECORD,,"]
    files[f"{info}/RECORD"] = ("\n".join(record) + "\n").encode()
    wheel = destination / f"{_PACKAGE}-{_VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return wheel


@pytest.fixture(scope="module")
def installed_backend(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("custom-backend")
    wheel = _build_wheel(root)
    target = root / "site"
    uv = shutil.which("uv")
    if uv is not None:
        subprocess.run(
            [uv, "pip", "install", "--offline", "--no-deps", "--target", str(target), str(wheel)],
            check=True,
            capture_output=True,
            timeout=120,
        )
    else:
        # Installing a pure wheel is unpacking it into a purelib directory.
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(target)
    assert (target / f"{_PACKAGE}-{_VERSION}.dist-info" / "entry_points.txt").is_file()
    return target


def _fresh_python(
    site: Path, code: str, *args: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("SKILLEVALUATOR_DISABLE")}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(site), env.get("PYTHONPATH", "")]))
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=300, env=env, check=False
    )


def test_installed_package_is_discovered_without_importing_it(installed_backend: Path) -> None:
    result = _fresh_python(
        installed_backend,
        "import json, sys; from skillevaluator.cli import ENV_MODE_CHOICE; "
        "print(json.dumps([ENV_MODE_CHOICE.choices[-1], 'skillevaluator_demo_backend' in sys.modules]))",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["demo-backend", False]


def test_doctor_runs_the_installed_backend_checks(installed_backend: Path) -> None:
    cli = "from skillevaluator.cli import cli; cli()"
    ready = _fresh_python(
        installed_backend,
        cli,
        "doctor",
        "--env-mode",
        "demo-backend",
        "-a",
        "opencode",
        "--ek",
        "endpoint=https://up.example",
    )
    down = _fresh_python(
        installed_backend,
        cli,
        "doctor",
        "--env-mode",
        "demo-backend",
        "-a",
        "opencode",
        "--ek",
        "endpoint=https://down.example",
    )
    missing = _fresh_python(installed_backend, cli, "doctor", "--env-mode", "demo-backend", "-a", "opencode")

    ready_row = next(line for line in ready.stdout.splitlines() if "demo-backend prerequisite" in line)
    assert "pass" in ready_row and "ready" in ready_row
    assert "demo backend endpoint is down" in down.stdout
    assert "requires non-empty string --environment-kwarg for: endpoint" in " ".join(missing.stdout.split())


def test_harbor_dispatches_the_installed_environment_class(installed_backend: Path, tmp_path: Path) -> None:
    code = textwrap.dedent(
        """
        import json, sys
        from pathlib import Path
        from harbor.environments.factory import EnvironmentFactory
        from harbor.models.task.config import EnvironmentConfig
        from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
        from harbor.models.trial.paths import TrialPaths
        from skillevaluator.tier3.harbor.runner import build_harbor_run_command

        root = Path(sys.argv[1])
        command = build_harbor_run_command(
            dataset_path=root / "dataset", agent="opencode", job_name="demo", env_mode="demo-backend",
            environment_kwargs={"endpoint": "https://up.example", "lease_seconds": 900},
        )
        import_path = command[command.index("--env") + 1]
        kwargs = {}
        for index, value in enumerate(command):
            if value == "--ek":
                name, raw = command[index + 1].split("=", 1)
                kwargs[name] = json.loads(raw)
        (root / "env").mkdir()
        environment = EnvironmentFactory.create_environment_from_config(
            TrialEnvironmentConfig(import_path=import_path, kwargs=kwargs),
            environment_dir=root / "env", environment_name="demo", session_id="session",
            trial_paths=TrialPaths(trial_dir=root / "trial"), task_env_config=EnvironmentConfig(),
        )
        print(json.dumps([import_path, type(environment).__module__, environment.type(),
                          environment.endpoint, environment.lease_seconds]))
        """
    )
    result = _fresh_python(installed_backend, code, str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == [
        "skillevaluator_demo_backend.environment:DemoEnvironment",
        "skillevaluator_demo_backend.environment",
        "demo-backend",
        "https://up.example",
        900,
    ]
