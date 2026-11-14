# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned, checksum-verified provisioning of the external Tier 1 scanners.

SkillSpector is not on PyPI, so when it is not installed its pinned release
wheel is downloaded once into SkillEvaluator's runtime directory and run with
``uv tool run`` in uv's own cached environment. Gitleaks is a native binary,
downloaded once from its official release into the same directory.
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from types import MappingProxyType

SKILLSPECTOR_VERSION = "2.13.4"
SKILLSPECTOR_WHEEL = f"skillspector-{SKILLSPECTOR_VERSION}-py3-none-any.whl"
SKILLSPECTOR_WHEEL_URL = (
    f"https://github.com/NVIDIA/SkillSpector/releases/download/v{SKILLSPECTOR_VERSION}/{SKILLSPECTOR_WHEEL}"
)
SKILLSPECTOR_WHEEL_SHA256 = "c5e7f290897c65018013a6e37022fabdd2ba1aeeb4994b65df2de8b0d536bea0"
GITLEAKS_VERSION = "8.30.0"

_GITLEAKS_SHA256 = MappingProxyType(
    {
        ("linux", "x64"): "79a3ab579b53f71efd634f3aaf7e04a0fa0cf206b7ed434638d1547a2470a66e",
        ("linux", "arm64"): "b4cbbb6ddf7d1b2a603088cd03a4e3f7ce48ee7fd449b51f7de6ee2906f5fa2f",
        ("darwin", "x64"): "ca221d012d247080c2f6f61f4b7a83bffa2453806b0c195c795bbe9a8c775ed5",
        ("darwin", "arm64"): "b251ab2bcd4cd8ba9e56ff37698c033ebf38582b477d21ebd86586d927cf87e7",
    }
)
_ARCH_ALIASES = MappingProxyType({"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"})
DOWNLOAD_TIMEOUT_SECONDS = 120
_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024


class _InstallError(Exception):
    pass


def _runtime_path(*parts: str) -> Path | None:
    from skillevaluator.tier3.harbor.local_runtime import default_runtime_root

    try:
        return default_runtime_root().joinpath(*parts)
    except ValueError:
        return None


def _download_verified(url: str, sha256: str, name: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            payload = response.read(_MAX_DOWNLOAD_BYTES + 1)
    except OSError as exc:
        reason = getattr(exc, "reason", exc)
        detail = getattr(reason, "strerror", None) or str(reason)
        raise _InstallError(f"couldn't download {name} ({detail.lower()})") from exc
    if len(payload) > _MAX_DOWNLOAD_BYTES:
        raise _InstallError(f"{name} download exceeded the size limit")
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise _InstallError(f"{name} download failed checksum verification")
    return payload


def _write_atomic(target: Path, data: bytes, mode: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}-")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        Path(tmp).chmod(mode)
        Path(tmp).replace(target)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise


def managed_skillspector_wheel() -> Path | None:
    """Where the pinned SkillSpector wheel lives; None when the runtime root is invalid."""
    return _runtime_path("skillspector", SKILLSPECTOR_VERSION, SKILLSPECTOR_WHEEL)


def ensure_skillspector_wheel() -> str | None:
    """Make sure the verified SkillSpector wheel is present; return an error or None."""
    target = managed_skillspector_wheel()
    if target is None:
        return "SkillEvaluator runtime directory is invalid"
    try:
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == SKILLSPECTOR_WHEEL_SHA256:
            return None
        payload = _download_verified(
            SKILLSPECTOR_WHEEL_URL, SKILLSPECTOR_WHEEL_SHA256, f"SkillSpector {SKILLSPECTOR_VERSION}"
        )
        _write_atomic(target, payload, 0o644)
    except _InstallError as exc:
        return str(exc)
    except OSError as exc:
        return f"SkillSpector install failed: {exc}"
    return None


def prepare_skillspector(uv: str) -> str | None:
    """Download the SkillSpector wheel and build its uv environment; return an error or None."""
    error = ensure_skillspector_wheel()
    if error:
        return error
    try:
        proc = subprocess.run(
            [uv, "tool", "run", "--quiet", "--from", str(managed_skillspector_wheel()), "skillspector", "--version"],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"couldn't set up SkillSpector {SKILLSPECTOR_VERSION} ({exc})"
    if proc.returncode != 0:
        lines = [line.strip() for line in (proc.stderr or proc.stdout).splitlines() if line.strip()]
        causes = [line.split("Caused by:", 1)[1] for line in lines if "Caused by:" in line]
        reason = (causes[-1] if causes else lines[0] if lines else f"exit {proc.returncode}").strip()
        reason = reason.removeprefix("error:").lstrip("\N{MULTIPLICATION SIGN} ").rstrip(":").strip()
        return f"couldn't set up SkillSpector {SKILLSPECTOR_VERSION} ({reason[:1].lower()}{reason[1:]})"
    return None


def gitleaks_platform() -> tuple[str, str] | None:
    """Return the release (os, arch) for this host, or None when no pinned binary exists."""
    key = (sys.platform, _ARCH_ALIASES.get(platform.machine().lower(), ""))
    return key if key in _GITLEAKS_SHA256 else None


def gitleaks_url(key: tuple[str, str]) -> str:
    os_name, arch = key
    return (
        f"https://github.com/gitleaks/gitleaks/releases/download/v{GITLEAKS_VERSION}/"
        f"gitleaks_{GITLEAKS_VERSION}_{os_name}_{arch}.tar.gz"
    )


def managed_gitleaks_path() -> Path | None:
    """Where the managed Gitleaks binary lives; None when the runtime root is invalid."""
    return _runtime_path("gitleaks", GITLEAKS_VERSION, "gitleaks")


def ensure_gitleaks() -> str | None:
    """Make sure the verified Gitleaks binary is installed; return an error or None."""
    target = managed_gitleaks_path()
    if target is not None and target.is_file() and os.access(target, os.X_OK):
        return None
    return install_gitleaks()


def install_gitleaks() -> str | None:
    """Download, verify and install the pinned Gitleaks binary; return an error or None."""
    key = gitleaks_platform()
    target = managed_gitleaks_path()
    if key is None or target is None:
        return f"no pinned Gitleaks {GITLEAKS_VERSION} binary for this platform"
    try:
        payload = _download_verified(gitleaks_url(key), _GITLEAKS_SHA256[key], f"Gitleaks {GITLEAKS_VERSION}")
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            member = archive.getmember("gitleaks")
            stream = archive.extractfile(member) if member.isfile() else None
            if stream is None:
                return "Gitleaks archive has no gitleaks binary"
            binary = stream.read()
        _write_atomic(target, binary, 0o755)
    except _InstallError as exc:
        return str(exc)
    except (OSError, KeyError, tarfile.TarError) as exc:
        return f"Gitleaks install failed: {exc}"
    return None
