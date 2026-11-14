# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed scanner provisioning: verified SkillSpector wheel, optional Semgrep, pinned Gitleaks."""

import hashlib
import io
import os
import tarfile
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from skillevaluator.utils import scanner_install, tool_runner
from skillevaluator.utils.tool_runner import ExternalTool, Tools
from skillevaluator.validators.code_risk import REQUIRE_SEMGREP_ENV, CodeRiskValidator


def _resolver(found: dict[str, str]):
    return lambda command, **_kw: found.get(command)


@pytest.fixture
def runtime_dir(tmp_path, monkeypatch):
    root = tmp_path / "runtimes"
    monkeypatch.setenv("SKILLEVALUATOR_RUNTIME_DIR", str(root))
    return root


def _gitleaks_tarball(content: bytes = b"#!/bin/sh\necho gitleaks\n") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        info = tarfile.TarInfo("gitleaks")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    return buf.getvalue()


class TestSkillSpectorWheel:
    @pytest.mark.usefixtures("runtime_dir")
    def test_prepare_downloads_wheel_and_builds_uv_environment(self, monkeypatch):
        payload = b"wheel"
        monkeypatch.setattr(scanner_install, "SKILLSPECTOR_WHEEL_SHA256", hashlib.sha256(payload).hexdigest())
        with (
            patch("urllib.request.urlopen", return_value=io.BytesIO(payload)),
            patch("subprocess.run", return_value=CompletedProcess([], 0, "SkillSpector v2.13.4", "")) as run,
        ):
            assert scanner_install.prepare_skillspector("/opt/uv") is None
        wheel = scanner_install.managed_skillspector_wheel()
        assert wheel.read_bytes() == payload
        assert run.call_args.args[0][-2:] == ["skillspector", "--version"]
        assert str(wheel) in run.call_args.args[0]

    @pytest.mark.usefixtures("runtime_dir")
    def test_prepare_reports_environment_failure(self, monkeypatch):
        monkeypatch.setattr(scanner_install, "ensure_skillspector_wheel", lambda: None)
        with patch(
            "subprocess.run",
            return_value=CompletedProcess([], 2, "", "error: resolution failed\n  Caused by: Network is unreachable\n"),
        ):
            error = scanner_install.prepare_skillspector("/opt/uv")
        assert error == f"couldn't set up SkillSpector {scanner_install.SKILLSPECTOR_VERSION} (network is unreachable)"

    def test_missing_skillspector_runs_verified_wheel_through_uv(self, runtime_dir, monkeypatch):
        payload = b"wheel"
        monkeypatch.setattr(scanner_install, "SKILLSPECTOR_WHEEL_SHA256", hashlib.sha256(payload).hexdigest())
        with patch.object(tool_runner, "resolve_tool_path", _resolver({"uv": "/opt/uv"})):
            tool = ExternalTool("skillspector", "skillspector", managed=True)
        assert tool.is_available
        assert tool.resolution_source == "uv tool run"
        wheel = runtime_dir / "skillspector" / scanner_install.SKILLSPECTOR_VERSION / scanner_install.SKILLSPECTOR_WHEEL
        with (
            patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as urlopen,
            patch("subprocess.run", return_value=CompletedProcess([], 0, "{}", "")) as run,
        ):
            tool.run(["scan", "x"])
            tool.run(["scan", "y"])
        urlopen.assert_called_once()
        assert urlopen.call_args.args[0] == scanner_install.SKILLSPECTOR_WHEEL_URL
        assert wheel.read_bytes() == payload
        assert run.call_args.args[0] == [
            "/opt/uv",
            "tool",
            "run",
            "--quiet",
            "--from",
            str(wheel),
            "skillspector",
            "scan",
            "y",
        ]

    @pytest.mark.usefixtures("runtime_dir")
    def test_checksum_mismatch_never_runs(self):
        with patch.object(tool_runner, "resolve_tool_path", _resolver({"uv": "/opt/uv"})):
            tool = ExternalTool("skillspector", "skillspector", managed=True)
        with (
            patch("urllib.request.urlopen", return_value=io.BytesIO(b"tampered")),
            patch("subprocess.run") as run,
        ):
            result = tool.run(["scan", "x"])
        assert result.error_message == "SkillSpector 2.13.4 download failed checksum verification"
        run.assert_not_called()

    def test_pinned_wheel_url_matches_version(self):
        assert scanner_install.SKILLSPECTOR_WHEEL_URL.endswith(
            f"/v{scanner_install.SKILLSPECTOR_VERSION}/{scanner_install.SKILLSPECTOR_WHEEL}"
        )
        assert len(scanner_install.SKILLSPECTOR_WHEEL_SHA256) == 64

    def test_installed_scanner_wins_over_fallback(self):
        found = {"uv": "/opt/uv", "skillspector": "/usr/bin/skillspector"}
        with patch.object(tool_runner, "resolve_tool_path", _resolver(found)):
            tool = ExternalTool("skillspector", "skillspector", managed=True)
        with patch("subprocess.run", return_value=CompletedProcess([], 0, "", "")) as run:
            tool.run(["--version"])
        assert run.call_args.args[0] == ["/usr/bin/skillspector", "--version"]

    @pytest.mark.usefixtures("runtime_dir")
    def test_patched_path_does_not_inherit_uv_arguments(self):
        with patch.object(tool_runner, "resolve_tool_path", _resolver({"uv": "/opt/uv"})):
            tool = ExternalTool("skillspector", "skillspector", managed=True)
        with (
            patch.object(tool, "_path", "/usr/bin/skillspector"),
            patch("subprocess.run", return_value=CompletedProcess([], 0, "", "")) as run,
        ):
            tool.run(["--version"])
        assert run.call_args.args[0] == ["/usr/bin/skillspector", "--version"]

    @pytest.mark.parametrize("managed", [True, False])
    def test_unavailable_without_uv_or_when_unmanaged(self, managed):
        found = {} if managed else {"uv": "/opt/uv"}
        with (
            patch.object(tool_runner, "resolve_tool_path", _resolver(found)),
            patch.object(tool_runner, "_packaged_uv", return_value=None),
        ):
            assert not ExternalTool("skillspector", "skillspector", managed=managed).is_available

    def test_invalid_override_never_falls_back(self, monkeypatch):
        monkeypatch.setenv("SKILLEVALUATOR_SKILLSPECTOR_PATH", "relative/skillspector")
        with patch.object(tool_runner, "resolve_tool_path", _resolver({"uv": "/opt/uv"})):
            tool = ExternalTool(
                "skillspector", "skillspector", override_env="SKILLEVALUATOR_SKILLSPECTOR_PATH", managed=True
            )
        assert not tool.is_available
        assert "SKILLEVALUATOR_SKILLSPECTOR_PATH" in tool.get_install_hint()


class TestOptionalSemgrep:
    @pytest.fixture
    def skill_dir(self, tmp_path):
        (tmp_path / "SKILL.md").write_text("---\nname: s\n---\n")
        return tmp_path

    def test_missing_semgrep_is_a_note_by_default(self, skill_dir, monkeypatch):
        monkeypatch.delenv(REQUIRE_SEMGREP_ENV, raising=False)
        with patch.object(Tools.semgrep, "_path", None):
            result = CodeRiskValidator()._run_semgrep(skill_dir)
        assert "incomplete_scans" not in result.metadata
        assert not result.warnings
        assert "Semgrep not installed; skipped" in result.messages[0]

    def test_missing_semgrep_is_incomplete_when_required(self, skill_dir, monkeypatch):
        monkeypatch.setenv(REQUIRE_SEMGREP_ENV, "1")
        with patch.object(Tools.semgrep, "_path", None):
            result = CodeRiskValidator()._run_semgrep(skill_dir)
        assert result.metadata["incomplete_scans"] == ["semgrep"]


class TestManagedGitleaks:
    def test_missing_binary_installs_on_first_run_and_is_reused(self, runtime_dir, monkeypatch):
        payload = _gitleaks_tarball()
        key = ("linux", "x64")
        monkeypatch.setattr(scanner_install, "gitleaks_platform", lambda: key)
        monkeypatch.setattr(tool_runner, "gitleaks_platform", lambda: key)
        monkeypatch.setattr(scanner_install, "_GITLEAKS_SHA256", {key: hashlib.sha256(payload).hexdigest()})
        binary = runtime_dir / "gitleaks" / scanner_install.GITLEAKS_VERSION / "gitleaks"
        with patch.object(tool_runner, "resolve_tool_path", _resolver({})):
            tool = ExternalTool("Gitleaks", "gitleaks", managed=True)
        assert tool.is_available
        assert tool.path == str(binary)
        assert tool.resolution_source == "managed"
        assert not binary.exists()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as urlopen:
            result = tool.run(["version"])
            tool.run(["version"])
        assert result.success
        assert urlopen.call_count == 1
        assert urlopen.call_args.args[0] == scanner_install.gitleaks_url(key)
        assert os.access(binary, os.X_OK)

    @pytest.mark.usefixtures("runtime_dir")
    def test_failed_first_run_install_is_reported_not_launched(self, monkeypatch):
        monkeypatch.setattr(scanner_install, "gitleaks_platform", lambda: ("linux", "x64"))
        monkeypatch.setattr(tool_runner, "gitleaks_platform", lambda: ("linux", "x64"))
        with patch.object(tool_runner, "resolve_tool_path", _resolver({})):
            tool = ExternalTool("Gitleaks", "gitleaks", managed=True)
        with (
            patch("urllib.request.urlopen", side_effect=OSError("network unreachable")),
            patch("subprocess.run") as run,
        ):
            result = tool.run(["version"])
        assert not result.success
        assert result.error_message == "couldn't download Gitleaks 8.30.0 (network unreachable)"
        run.assert_not_called()

    @pytest.mark.usefixtures("runtime_dir")
    def test_unsupported_platform_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(tool_runner, "gitleaks_platform", lambda: None)
        with patch.object(tool_runner, "resolve_tool_path", _resolver({})):
            assert not ExternalTool("Gitleaks", "gitleaks", managed=True).is_available

    def test_checksum_mismatch_installs_nothing(self, runtime_dir, monkeypatch):
        monkeypatch.setattr(scanner_install, "gitleaks_platform", lambda: ("linux", "x64"))
        with patch("urllib.request.urlopen", return_value=io.BytesIO(_gitleaks_tarball())):
            error = scanner_install.install_gitleaks()
        assert error == "Gitleaks 8.30.0 download failed checksum verification"
        assert not runtime_dir.exists()

    @pytest.mark.usefixtures("runtime_dir")
    def test_download_error_is_reported(self, monkeypatch):
        monkeypatch.setattr(scanner_install, "gitleaks_platform", lambda: ("linux", "x64"))
        with patch("urllib.request.urlopen", side_effect=OSError("network unreachable")):
            assert scanner_install.install_gitleaks() == "couldn't download Gitleaks 8.30.0 (network unreachable)"

    @pytest.mark.usefixtures("runtime_dir")
    def test_unsupported_platform(self, monkeypatch):
        monkeypatch.setattr(scanner_install, "gitleaks_platform", lambda: None)
        assert "no pinned Gitleaks" in scanner_install.install_gitleaks()
