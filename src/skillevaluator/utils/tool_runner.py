# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External tool execution utilities.

Provides reusable patterns for running external CLI tools, parsing their
JSON output, and handling errors consistently.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from skillevaluator.logging_config import get_logger
from skillevaluator.models.result import Severity
from skillevaluator.utils.scanner_install import (
    GITLEAKS_VERSION,
    SKILLSPECTOR_VERSION,
    SKILLSPECTOR_WHEEL_URL,
    ensure_gitleaks,
    ensure_skillspector_wheel,
    gitleaks_platform,
    managed_gitleaks_path,
    managed_skillspector_wheel,
)

logger = get_logger(__name__)

# CVSSv3 score thresholds for CVE severity mapping
CVSS_THRESHOLDS: dict[Severity, float] = {
    Severity.CRITICAL: 9.0,
    Severity.HIGH: 7.0,
    Severity.MEDIUM: 4.0,
    Severity.LOW: 0.1,
}

# Gitleaks normally uses exit 1 for both findings and operational errors.
# Every invocation overrides the findings code so those states are distinct.
GITLEAKS_FINDINGS_EXIT_CODE = 10


def cvss_to_severity(score: float) -> Severity:
    """Convert CVSSv3 score to severity level."""
    if score >= CVSS_THRESHOLDS[Severity.CRITICAL]:
        return Severity.CRITICAL
    if score >= CVSS_THRESHOLDS[Severity.HIGH]:
        return Severity.HIGH
    if score >= CVSS_THRESHOLDS[Severity.MEDIUM]:
        return Severity.MEDIUM
    return Severity.LOW


def resolve_tool_path(command: str, *, prefer_interpreter_sibling: bool = False) -> str | None:
    """Resolve a tool binary from PATH and the interpreter's own bin dir.

    Isolated installs (``uv tool install``, ``pipx``) place the console
    scripts of bundled extras next to the interpreter, in a directory
    that is not on the user's PATH. Bundled scanners prefer that sibling
    location so an unrelated global executable cannot silently replace the
    version installed with SkillEvaluator. External tools such as Gitleaks
    retain normal PATH precedence.
    """
    if prefer_interpreter_sibling:
        if sys.executable:
            sibling = shutil.which(command, path=str(Path(sys.executable).parent))
            if sibling is not None:
                return sibling
        return shutil.which(command)

    path = shutil.which(command)
    if path is not None:
        return path
    if not sys.executable:
        return None
    return shutil.which(command, path=str(Path(sys.executable).parent))


@dataclass
class ToolResult:
    """Result from running an external tool.

    ``success`` only means the process launched and ran to completion
    (no missing binary, timeout, or OS error) — NOT that the tool
    reported a clean scan. Callers must branch on ``exit_code`` to
    distinguish clean (usually 0) from findings (usually 1) from tool
    errors (anything else); treating an unexpected exit code as a clean
    result makes a crashed scanner indistinguishable from a passing one.
    """

    success: bool
    stdout: str
    stderr: str
    exit_code: int
    error_message: str | None = None


def _captured_text(value: str | bytes | None) -> str:
    """Normalize partial subprocess output without dropping undecodable bytes."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


class ExternalTool:
    """Wrapper for external CLI tools with consistent error handling.

    Provides automatic PATH lookup, installation hints, timeout handling,
    and structured result objects.
    """

    def __init__(
        self,
        name: str,
        command: str | None = None,
        *,
        prefer_interpreter_sibling: bool = False,
        override_env: str | None = None,
        managed: bool = False,
    ):
        """Initialize tool wrapper.

        Args:
            name: Human-readable tool name for error messages
            command: Command name to look up in PATH (defaults to name)
            prefer_interpreter_sibling: Prefer the executable installed next
                to SkillEvaluator's interpreter over a same-named PATH tool.
            override_env: Optional environment variable containing an explicit
                absolute executable path. Invalid overrides fail closed rather
                than silently falling back to another executable.
            managed: When the tool is not installed, fall back to SkillEvaluator's
                pinned provisioning: the verified SkillSpector wheel through
                ``uv tool run``, or the downloaded Gitleaks binary.
        """
        self.name = name
        self.command = command or name
        self.override_env = override_env
        self._prefer_interpreter_sibling = prefer_interpreter_sibling
        self._managed = managed
        self.refresh()

    def refresh(self) -> None:
        """Resolve the executable again, e.g. after a managed install."""
        self.resolution_source = "automatic"
        self._configuration_error: str | None = None
        self._version: str | None = None
        self._uv_run: tuple[str, tuple[str, ...]] | None = None
        self._ensure: tuple[str, Callable[[], str | None]] | None = None
        override_env = self.override_env
        name = self.name

        override = os.environ.get(override_env, "").strip() if override_env else ""
        if override:
            candidate = Path(override)
            if not candidate.is_absolute() or not candidate.is_file() or not os.access(candidate, os.X_OK):
                self._path = None
                self._configuration_error = f"{override_env} must name an absolute executable file; got {override!r}"
            else:
                self._path = str(candidate.resolve())
                self.resolution_source = f"override:{override_env}"
                logger.warning("Using explicit %s executable override from %s: %s", name, override_env, self._path)
        else:
            self._path = resolve_tool_path(
                self.command,
                prefer_interpreter_sibling=self._prefer_interpreter_sibling,
            )
            if self._path is None and self._managed:
                self._resolve_managed()

    def _resolve_managed(self) -> None:
        if self.command == "skillspector":
            uv = resolve_tool_path("uv")
            wheel = managed_skillspector_wheel()
            if uv is not None and wheel is not None:
                self._path = uv
                self._uv_run = (uv, ("tool", "run", "--quiet", "--from", str(wheel), "skillspector"))
                self._ensure = (uv, ensure_skillspector_wheel)
                self.resolution_source = "uv tool run"
        elif self.command == "gitleaks":
            binary = managed_gitleaks_path()
            if binary is not None and gitleaks_platform() is not None:
                self._path = str(binary)
                self._ensure = (self._path, ensure_gitleaks)
                self.resolution_source = "managed"

    def _launch_args(self) -> tuple[str, ...]:
        if self._uv_run is not None and self._path == self._uv_run[0]:
            return self._uv_run[1]
        return ()

    @property
    def is_available(self) -> bool:
        """Check if tool is available in PATH."""
        return self._path is not None

    @property
    def path(self) -> str | None:
        """Get full path to tool binary."""
        return self._path

    @property
    def version(self) -> str | None:
        """The pinned release for managed installs, otherwise what the tool's version command reports."""
        if self.resolution_source == "uv tool run":
            return SKILLSPECTOR_VERSION
        if self.resolution_source == "managed":
            return GITLEAKS_VERSION
        if self._path is None:
            return None
        if self._version is None:
            argv = [self._path, "version" if self.command == "gitleaks" else "--version"]
            try:
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
            except (OSError, subprocess.SubprocessError):
                proc = None
            match = proc and re.search(r"\d+\.\d+(?:\.\d+)?", proc.stdout or proc.stderr or "")
            self._version = match.group(0) if match else ""
        return self._version or None

    @property
    def configuration_error(self) -> str | None:
        """Return why an explicit executable override was rejected, if it was."""
        return self._configuration_error

    def get_install_hint(self) -> str:
        """Get installation hint for this tool."""
        if self._configuration_error:
            return self._configuration_error
        bundled_scanner_hint = (
            "Bundled with SkillEvaluator — reinstall with: "
            'uv tool install --reinstall "skillevaluator @ git+https://github.com/NVIDIA/SkillEvaluator.git"'
        )
        security_extra_hint = (
            "Bundled with the security extra — reinstall with: "
            'uv tool install "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git"'
        )
        hints = {
            "gitleaks": (
                "Install with: brew install gitleaks (macOS) or go install github.com/gitleaks/gitleaks/v8@latest"
            ),
            "pip-audit": security_extra_hint,
            "safety": "Optional secondary scanner; install separately with: uv tool install safety",
            "bandit": bundled_scanner_hint,
            "semgrep": "Optional; install separately with: brew install semgrep (macOS) or uv tool install semgrep",
            "skillspector": f"Install with: uv tool install {SKILLSPECTOR_WHEEL_URL}",
            "skillevaluator": (
                'Install with: uv tool install "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git"'
            ),
        }
        return hints.get(self.command, f"Install {self.name} and ensure it's in PATH")

    def run(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 120,
        log_command: bool = True,
        env: Mapping[str, str] | None = None,
        replace_env: bool = False,
    ) -> ToolResult:
        """Execute tool with given arguments.

        Args:
            args: Command arguments (tool path prepended automatically)
            cwd: Working directory for command
            timeout: Timeout in seconds
            log_command: Whether to log the command being run
            env: Extra environment variables merged over the parent
                environment for this invocation only
            replace_env: Use ``env`` as the complete child environment instead
                of merging it over the parent. Use this for credential-isolated
                subprocesses.

        Returns:
            ToolResult with stdout, stderr, and error info
        """
        if not self.is_available:
            return ToolResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                error_message=f"{self.name} not installed. {self.get_install_hint()}",
            )

        launch_args = self._launch_args()
        if self._ensure is not None and self._path == self._ensure[0] and (error := self._ensure[1]()):
            return ToolResult(success=False, stdout="", stderr="", exit_code=-1, error_message=error)

        cmd = [self._path, *launch_args, *args]

        if log_command:
            logger.info(f"Running {self.name}: {' '.join(cmd[:5])}...")

        try:
            child_env = dict(env or {}) if replace_env else ({**os.environ, **env} if env else None)
            # Text mode decodes with Python's selected codec (the locale codec,
            # or UTF-8 in UTF-8 mode), which is what Python scanners write. A
            # strict decode of bytes outside that codec (a Windows traceback
            # with a CJK path, for example) raises UnicodeDecodeError in the
            # reader thread and leaves proc.stdout/stderr as None, so replace
            # undecodable bytes instead of failing.
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                cwd=cwd,
                env=child_env,
            )
            return ToolResult(
                success=True,
                stdout=_captured_text(proc.stdout),
                stderr=_captured_text(proc.stderr),
                exit_code=proc.returncode,
            )

        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                success=False,
                stdout=_captured_text(exc.stdout),
                stderr=_captured_text(exc.stderr),
                exit_code=-1,
                error_message=f"{self.name} timed out after {timeout} seconds",
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                error_message=f"{self.name} binary not found in PATH",
            )
        except OSError as e:
            return ToolResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                error_message=f"{self.name} execution failed: {e}",
            )


def parse_json_output(output: str, on_error: str | None = None) -> dict | list | None:
    """Parse JSON output from external tool.

    Args:
        output: Raw stdout from tool
        on_error: Optional message to check for in non-JSON output

    Returns:
        Parsed JSON data or None if parsing fails
    """
    if not output or not output.strip():
        return None

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        if on_error and on_error in output:
            return {}
        return None


def create_temp_config(content: str, suffix: str = ".toml") -> Path:
    """Create a temporary configuration file for external tool configs.

    Caller is responsible for cleanup; consider using contextlib for scoped lifetime.

    Args:
        content: Configuration file content
        suffix: File extension (default: .toml for Gitleaks)

    Returns:
        Path to temporary file
    """
    fd, path_str = tempfile.mkstemp(suffix=suffix, prefix="skillevaluator_")
    os.close(fd)
    config_path = Path(path_str)
    config_path.write_text(content)
    return config_path


class Tools:
    """Registry of external tools.

    Provides pre-configured ExternalTool instances for security scanning
    tools and evaluation tools used by SkillEvaluator validators.
    """

    gitleaks = ExternalTool("Gitleaks", "gitleaks", managed=True)
    pip_audit = ExternalTool("pip-audit", "pip-audit", prefer_interpreter_sibling=True)
    safety = ExternalTool("Safety", "safety")
    bandit = ExternalTool(
        "Bandit",
        "bandit",
        prefer_interpreter_sibling=True,
        override_env="SKILLEVALUATOR_BANDIT_PATH",
    )
    semgrep = ExternalTool(
        "Semgrep",
        "semgrep",
        prefer_interpreter_sibling=True,
        override_env="SKILLEVALUATOR_SEMGREP_PATH",
    )
    skillspector = ExternalTool(
        "skillspector",
        "skillspector",
        prefer_interpreter_sibling=True,
        override_env="SKILLEVALUATOR_SKILLSPECTOR_PATH",
        managed=True,
    )
