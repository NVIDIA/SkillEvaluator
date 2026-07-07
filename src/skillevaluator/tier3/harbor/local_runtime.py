# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local runtime discovery for Harbor local mode.

Local mode runs the vendor agent CLIs (claude-code, codex, opencode) that are
already installed on the host or under ``SKILLEVALUATOR_RUNTIME_DIR``. This
module discovers them on ``PATH``; it never downloads or installs anything.
When a CLI is missing, it returns the vendor's own install command as a hint.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from errno import ELOOP
from pathlib import Path
from subprocess import SubprocessError, TimeoutExpired
from typing import Any

from skillevaluator.tier3.harbor.local_sandbox import SandboxUnavailable

LOCAL_RUNTIME_AGENTS: tuple[str, ...] = ("claude-code", "codex", "opencode")
DEFAULT_RUNTIME_ROOT = Path("~/.local/share/skillevaluator/runtimes")
RUNTIME_VERSION_TIMEOUT_SECONDS = 10


class RuntimePathResolutionError(RuntimeError):
    """A runtime executable or dependency path could not be canonicalized."""


_RUNTIME_PROBE_HOST_ENV = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "SHELL",
    "USER",
    "LOGNAME",
    "SYSTEMROOT",
    "WINDIR",
)

_AGENT_COMMANDS: dict[str, tuple[str, ...]] = {
    "claude-code": ("claude", "claude-code"),
    "codex": ("codex",),
    "opencode": ("opencode",),
}

# Vendor install hints shown when a CLI is missing. Local mode never runs these
# for the user — it only prints them — so there is no download-and-exec surface.
_AGENT_INSTALL_HINTS: dict[str, str] = {
    "claude-code": "npm install -g @anthropic-ai/claude-code  (see https://docs.claude.com/en/docs/claude-code)",
    "codex": "npm install -g @openai/codex  (or: brew install codex)",
    "opencode": "npm install -g opencode-ai  (or: brew install sst/tap/opencode; see https://opencode.ai)",
}


def validate_runtime_root(root: Path | str) -> Path:
    """Return a canonical dedicated runtime root without exposing host home."""
    resolved = Path(root).expanduser().resolve(strict=False)
    try:
        home = Path.home().resolve(strict=False)
    except (RuntimeError, OSError):
        home = None
    contains_home = home is not None and (resolved == home or home.is_relative_to(resolved))
    if resolved == Path(resolved.anchor) or contains_home:
        raise ValueError(
            "SKILLEVALUATOR_RUNTIME_DIR must be a dedicated subdirectory and must not be the host home "
            "or one of its parents"
        )
    return resolved


def default_runtime_root() -> Path:
    """Return the validated runtime root, honoring its environment override."""
    raw = os.environ.get("SKILLEVALUATOR_RUNTIME_DIR")
    return validate_runtime_root(raw or DEFAULT_RUNTIME_ROOT)


def runtime_bin_dirs(
    runtime_root: Path | None = None,
    *,
    agents: Sequence[str] | None = None,
) -> list[Path]:
    """Return managed bin directories for only the requested local agents."""
    root = validate_runtime_root(runtime_root or default_runtime_root())
    candidates: list[Path] = []
    selected_agents = LOCAL_RUNTIME_AGENTS if agents is None else agents
    for agent in selected_agents:
        if agent not in _AGENT_COMMANDS:
            continue
        agent_root = root / agent
        candidates.extend(
            [
                agent_root / "bin",
                agent_root / "current" / "bin",
            ]
        )
        if agent_root.is_dir():
            candidates.extend(sorted(p / "bin" for p in agent_root.iterdir() if p.is_dir()))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path in seen or not path.is_dir():
            continue
        seen.add(path)
        out.append(path)
    return out


def _resolved_existing_path(path: str | Path) -> Path | None:
    try:
        return Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def runtime_path(
    runtime_root: Path | None = None,
    base_path: str | None = None,
    *,
    agents: Sequence[str] | None = None,
) -> str:
    """Return PATH with any bring-your-own runtime bins prepended."""
    selected_bins = list(
        dict.fromkeys(
            str(resolved)
            for path in runtime_bin_dirs(runtime_root, agents=agents)
            if (resolved := _resolved_existing_path(path)) is not None
        )
    )
    managed_bins = {
        resolved
        for path in runtime_bin_dirs(runtime_root, agents=LOCAL_RUNTIME_AGENTS)
        if (resolved := _resolved_existing_path(path)) is not None
    }
    inherited = base_path if base_path is not None else os.environ.get("PATH", "")
    inherited_pieces: list[str] = []
    seen_pieces = set(selected_bins)
    for piece in inherited.split(os.pathsep):
        if not piece:
            continue
        try:
            expanded = Path(piece).expanduser()
        except RuntimeError:
            continue
        if not expanded.is_absolute():
            continue
        resolved = _resolved_existing_path(expanded)
        if resolved is not None and not resolved.is_dir():
            continue
        if resolved in managed_bins:
            continue
        normalized = str(resolved) if resolved is not None else str(expanded)
        if normalized in seen_pieces:
            continue
        seen_pieces.add(normalized)
        inherited_pieces.append(normalized)
    pieces = [*selected_bins, *inherited_pieces]
    return os.pathsep.join(pieces)


def local_subprocess_env(
    *,
    runtime_root: Path | None = None,
    runtime_agents: Sequence[str] | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a host env for Harbor local subprocesses with runtime bins first."""
    env = dict(base_env or {})
    root = validate_runtime_root(runtime_root or default_runtime_root())
    env["SKILLEVALUATOR_RUNTIME_DIR"] = str(root)
    env["PATH"] = runtime_path(
        root,
        env.get("PATH") or os.environ.get("PATH", ""),
        agents=runtime_agents,
    )
    return env


def validate_local_agents(agents: list[str] | tuple[str, ...]) -> list[str]:
    """Return agents that are not supported by Harbor local mode."""
    supported = set(LOCAL_RUNTIME_AGENTS)
    return sorted({agent for agent in agents if agent not in supported})


def find_runtime_command(agent: str, *, runtime_root: Path | None = None) -> str | None:
    """Find the executable for a local-mode agent after prepending runtime bins."""
    commands = _AGENT_COMMANDS.get(agent, ())
    path = runtime_path(runtime_root, agents=[agent])
    for command in commands:
        found = shutil.which(command, path=path)
        if found:
            return str(_visible_command_path(found))
    return None


def _append_unique(paths: list[Path], candidate: Path) -> None:
    if candidate not in paths:
        paths.append(candidate)


def _resolve_runtime_path(path: Path, *, strict: bool = True) -> Path:
    try:
        return path.resolve(strict=strict)
    except OSError as exc:
        if exc.errno != ELOOP:
            raise
        raise RuntimePathResolutionError(f"could not resolve runtime path {path}: symlink cycle ({exc})") from exc
    except RuntimeError as exc:
        raise RuntimePathResolutionError(f"could not resolve runtime path {path}: symlink cycle ({exc})") from exc


def _visible_command_path(command: str | Path) -> Path:
    path = Path(command).expanduser().absolute()
    return _resolve_runtime_path(path.parent) / path.name


def _node_package_root(path: Path) -> Path | None:
    """Return a concrete npm package root when ``path`` is below node_modules."""
    parts = path.parts
    indices = [index for index, part in enumerate(parts) if part == "node_modules"]
    if not indices:
        return None
    index = indices[-1] + 1
    if index >= len(parts):
        return None
    if parts[index].startswith("@"):
        index += 1
        if index >= len(parts):
            return None
    # ``node_modules/.bin`` is a shared shim directory, not an npm package.
    if parts[index] == ".bin":
        return None
    return Path(*parts[: index + 1])


def _homebrew_shipped_config_paths(prefix: Path, keg: Path) -> list[Path]:
    """Return installed config files that were shipped by one selected keg."""
    bottled_etc = keg / ".bottle" / "etc"
    installed_etc = prefix / "etc"
    try:
        canonical_prefix = _resolve_runtime_path(prefix)
        canonical_keg = _resolve_runtime_path(keg)
        canonical_bottled_etc = _resolve_runtime_path(bottled_etc)
        canonical_installed_etc = _resolve_runtime_path(installed_etc)
    except (OSError, RuntimePathResolutionError, ValueError):
        return []
    if (
        canonical_prefix != prefix.absolute()
        or canonical_keg != keg.absolute()
        or canonical_bottled_etc != bottled_etc.absolute()
        or canonical_installed_etc != installed_etc.absolute()
        or not canonical_bottled_etc.is_dir()
        or not canonical_installed_etc.is_dir()
    ):
        return []
    paths: list[Path] = []
    for shipped in canonical_bottled_etc.rglob("*"):
        if shipped.is_symlink() or not shipped.is_file():
            continue
        installed = canonical_installed_etc / shipped.relative_to(canonical_bottled_etc)
        if installed.is_symlink() or not installed.is_file():
            continue
        try:
            shipped_resolved = _resolve_runtime_path(shipped)
            resolved = _resolve_runtime_path(installed)
        except (OSError, RuntimePathResolutionError, ValueError):
            continue
        if shipped_resolved != shipped.absolute() or resolved != installed.absolute():
            continue
        _append_unique(paths, resolved)
    return paths


def _homebrew_runtime_roots(path: Path) -> list[Path]:
    """Return the selected Homebrew keg, receipt dependencies, and shipped config.

    Homebrew binaries can load dylibs from sibling/transitive kegs and config
    copied into ``$HOMEBREW_PREFIX/etc``. Strict Seatbelt must allow those
    selected runtime files without exposing the whole Homebrew prefix.
    """
    try:
        resolved = _resolve_runtime_path(path)
    except OSError:
        return []
    parts = resolved.parts
    cellar_indices = [index for index, part in enumerate(parts) if part == "Cellar"]
    if not cellar_indices:
        return []
    index = cellar_indices[-1]
    if len(parts) <= index + 2:
        return []
    prefix = Path(*parts[:index])
    cellar = prefix / "Cellar"
    formula_root = cellar / parts[index + 1]
    keg = Path(*parts[: index + 3])
    try:
        canonical_prefix = _resolve_runtime_path(prefix)
        canonical_cellar = _resolve_runtime_path(cellar)
        canonical_formula_root = _resolve_runtime_path(formula_root)
        canonical_keg = _resolve_runtime_path(keg)
    except (OSError, RuntimePathResolutionError, ValueError):
        return []
    if (
        canonical_prefix != prefix.absolute()
        or canonical_cellar != cellar.absolute()
        or canonical_cellar.parent != canonical_prefix
        or canonical_formula_root != formula_root.absolute()
        or canonical_formula_root.parent != canonical_cellar
        or canonical_keg != keg.absolute()
        or canonical_keg.parent != canonical_formula_root
    ):
        return []
    prefix = canonical_prefix
    cellar = canonical_cellar
    keg = canonical_keg
    receipt = keg / "INSTALL_RECEIPT.json"
    try:
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(metadata, dict):
        return []

    roots = [keg.resolve()]
    for config_path in _homebrew_shipped_config_paths(prefix, keg):
        _append_unique(roots, config_path)

    dependencies = metadata.get("runtime_dependencies", [])
    if not isinstance(dependencies, list):
        return roots
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue
        full_name = dependency.get("full_name")
        if not isinstance(full_name, str):
            continue
        formula = full_name.rsplit("/", 1)[-1]
        if not formula or "\x00" in formula or formula in {".", ".."} or Path(formula).name != formula:
            continue

        formula_root = cellar / formula
        try:
            canonical_formula_root = _resolve_runtime_path(formula_root)
        except (OSError, RuntimePathResolutionError, ValueError):
            continue
        if canonical_formula_root != formula_root.absolute() or canonical_formula_root.parent != cellar:
            continue

        visible = prefix / "opt" / formula
        try:
            dependency_keg = _resolve_runtime_path(visible)
        except (OSError, RuntimePathResolutionError, ValueError):
            package_version = dependency.get("pkg_version")
            if (
                not isinstance(package_version, str)
                or not package_version
                or "\x00" in package_version
                or package_version in {".", ".."}
                or Path(package_version).name != package_version
            ):
                continue
            dependency_keg = prefix / "Cellar" / formula / package_version
            try:
                dependency_keg = _resolve_runtime_path(dependency_keg)
            except (OSError, RuntimePathResolutionError, ValueError):
                continue
        if dependency_keg.parent != canonical_formula_root:
            continue
        if visible.exists():
            _append_unique(roots, visible.absolute())
        _append_unique(roots, dependency_keg)
        for config_path in _homebrew_shipped_config_paths(prefix, dependency_keg):
            _append_unique(roots, config_path)
    return roots


def _shebang_interpreters(path: Path, *, search_path: str, seen: set[Path] | None = None) -> list[Path]:
    """Resolve a bounded shebang chain, including ``/usr/bin/env helper``."""
    visited = set() if seen is None else seen
    try:
        canonical = _resolve_runtime_path(path)
    except OSError:
        return []
    if canonical in visited or len(visited) >= 4:
        return []
    visited.add(canonical)
    try:
        with canonical.open("rb") as executable:
            first_line = executable.readline(4096)
    except OSError:
        return []
    if not first_line.startswith(b"#!"):
        return []
    try:
        tokens = shlex.split(first_line[2:].decode("utf-8", errors="strict").strip())
    except (UnicodeDecodeError, ValueError):
        return []
    if not tokens:
        return []

    interpreter = Path(tokens[0])
    paths: list[Path] = []
    if interpreter.name == "env":
        if interpreter.is_absolute():
            try:
                visible_interpreter = _visible_command_path(interpreter)
                resolved_interpreter = _resolve_runtime_path(visible_interpreter)
            except OSError:
                pass
            else:
                _append_unique(paths, visible_interpreter)
                _append_unique(paths, resolved_interpreter)
        arguments = tokens[1:]
        if arguments[:1] == ["-S"]:
            arguments = arguments[1:]
        helper_name = next(
            (
                argument
                for argument in arguments
                if not argument.startswith("-") and "=" not in argument and argument.strip()
            ),
            None,
        )
        helper = shutil.which(helper_name, path=search_path) if helper_name else None
        if helper:
            visible = _visible_command_path(helper)
            _append_unique(paths, visible)
            resolved = _resolve_runtime_path(visible)
            _append_unique(paths, resolved)
            for dependency in _shebang_interpreters(resolved, search_path=search_path, seen=visited):
                _append_unique(paths, dependency)
        return paths

    if interpreter.is_absolute():
        try:
            visible_interpreter = _visible_command_path(interpreter)
            resolved = _resolve_runtime_path(visible_interpreter)
        except OSError:
            return paths
        _append_unique(paths, visible_interpreter)
        _append_unique(paths, resolved)
        for dependency in _shebang_interpreters(resolved, search_path=search_path, seen=visited):
            _append_unique(paths, dependency)
    return paths


def runtime_command_roots(
    agents: Sequence[str],
    *,
    runtime_root: Path | None = None,
) -> list[Path]:
    """Return exact executable/dependency reads and proven package roots."""
    roots: list[Path] = []
    search_path = runtime_path(runtime_root, agents=agents)

    for agent in agents:
        found = find_runtime_command(agent, runtime_root=runtime_root)
        if not found:
            continue
        command = _visible_command_path(found)
        resolved = _resolve_runtime_path(command)
        _append_unique(roots, command)
        package_root = _node_package_root(resolved)
        if package_root is not None:
            _append_unique(roots, _resolve_runtime_path(package_root))
        else:
            _append_unique(roots, resolved)
        for dependency in _homebrew_runtime_roots(resolved):
            _append_unique(roots, dependency)
        for dependency in _shebang_interpreters(resolved, search_path=search_path):
            package_root = _node_package_root(dependency)
            _append_unique(roots, _resolve_runtime_path(package_root) if package_root else dependency)
            for brew_root in _homebrew_runtime_roots(dependency):
                _append_unique(roots, brew_root)
    return roots


def missing_local_runtimes(
    agents: list[str] | tuple[str, ...],
    *,
    runtime_root: Path | None = None,
) -> dict[str, str]:
    """Return missing runtime commands keyed by agent."""
    missing: dict[str, str] = {}
    for agent in agents:
        if agent not in _AGENT_COMMANDS:
            continue
        if find_runtime_command(agent, runtime_root=runtime_root):
            continue
        missing[agent] = " or ".join(_AGENT_COMMANDS[agent])
    return missing


def local_runtime_install_command(agents: Sequence[str], **_ignored: object) -> str:
    """Return the vendor install hint(s) for the given agent(s)."""
    hints = [_AGENT_INSTALL_HINTS.get(agent, f"install the '{agent}' CLI and put it on PATH") for agent in agents]
    return "\n  ".join(hints)


def local_runtime_error_message(agent: str, command: str) -> str:
    """Return a concise, actionable install hint for a missing local runtime."""
    hint = _AGENT_INSTALL_HINTS.get(agent, f"install the '{agent}' CLI and put it on PATH")
    return (
        f"Local mode agent '{agent}' CLI not found ({command}).\n"
        "Install it (or set SKILLEVALUATOR_RUNTIME_DIR to a dir containing it):\n"
        f"  {hint}"
    )


def _runtime_probe_error_message(agent: str, command: str, detail: str) -> str:
    """Return an actionable diagnostic for an installed but unusable CLI."""
    hint = _AGENT_INSTALL_HINTS.get(agent, f"reinstall or update the '{agent}' CLI")
    return (
        f"Local mode agent '{agent}' CLI was found at {command}, but {detail}.\n"
        "Reinstall or update it, then confirm its version command succeeds:\n"
        f"  {hint}"
    )


def _runtime_probe_sandbox_error_message(agent: str, command: str, exc: Exception) -> str:
    detail = str(exc).strip()
    cause = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return (
        f"Local mode agent '{agent}' CLI was found at {command}, but its sandboxed --version check "
        f"could not be prepared ({cause}).\n"
        "Fix the runtime installation or local sandbox configuration, or use --env-mode docker."
    )


def _probe_runtime_version(
    agent: str,
    command: str,
    *,
    runtime_root: Path | None,
    environ: Mapping[str, str] | None,
    sandbox: Any | None,
    strict_reads: bool,
) -> str | None:
    """Run a selected CLI's version command in a credential-free temp HOME."""
    supplied = environ or {}
    safe_host_env: dict[str, str] = {}
    for key in _RUNTIME_PROBE_HOST_ENV:
        value = supplied.get(key) or os.environ.get(key)
        if value:
            safe_host_env[key] = value
    safe_host_env.setdefault("PATH", os.defpath)
    probe_env = local_subprocess_env(
        runtime_root=runtime_root,
        runtime_agents=[agent],
        base_env=safe_host_env,
    )

    with tempfile.TemporaryDirectory(prefix="skillevaluator-runtime-probe-") as temp_dir:
        root = Path(temp_dir)
        home = root / "home"
        tmp = root / "tmp"
        xdg_config = home / ".config"
        xdg_cache = home / ".cache"
        xdg_data = home / ".local" / "share"
        for path in (home, tmp, xdg_config, xdg_cache, xdg_data):
            path.mkdir(parents=True, exist_ok=True)
        probe_env.update(
            {
                "HOME": str(home),
                "TMPDIR": str(tmp),
                "XDG_CONFIG_HOME": str(xdg_config),
                "XDG_CACHE_HOME": str(xdg_cache),
                "XDG_DATA_HOME": str(xdg_data),
            }
        )
        argv = [command, "--version"]
        if sandbox is not None:
            try:
                read_roots = runtime_command_roots([agent], runtime_root=runtime_root)
            except (OSError, RuntimePathResolutionError) as exc:
                return _runtime_probe_sandbox_error_message(agent, command, exc)
            try:
                argv = sandbox.wrap(
                    argv,
                    workdir=home,
                    write_roots=[root],
                    home=home,
                    tmp=tmp,
                    allow_net=False,
                    extra_ro=read_roots,
                    strict_reads=strict_reads,
                )
            except SandboxUnavailable as exc:
                return _runtime_probe_sandbox_error_message(agent, command, exc)
        try:
            result = subprocess.run(
                argv,
                cwd=str(home),
                env=probe_env,
                capture_output=True,
                text=True,
                timeout=RUNTIME_VERSION_TIMEOUT_SECONDS,
                check=False,
            )
        except TimeoutExpired:
            return _runtime_probe_error_message(
                agent,
                command,
                f"its --version check timed out after {RUNTIME_VERSION_TIMEOUT_SECONDS} seconds",
            )
        except (OSError, SubprocessError) as exc:
            return _runtime_probe_error_message(
                agent,
                command,
                f"its --version check could not run ({type(exc).__name__})",
            )
    if result.returncode != 0:
        return _runtime_probe_error_message(
            agent,
            command,
            f"its --version check failed (exit {result.returncode})",
        )
    return None


def ensure_local_runtimes(
    agents: Sequence[str],
    *,
    runtime_root: Path | None = None,
    reporter: object = None,
    env: Mapping[str, str] | None = None,
    sandbox: Any | None = None,
    strict_reads: bool = False,
) -> list[str]:
    """Check requested local runtimes are installed and can report a version.

    Purely diagnostic — never installs. Probes run with a bounded timeout and a
    disposable credential-free HOME. Missing or unusable agents yield vendor
    install guidance.
    """
    _ = reporter
    errors: list[str] = []
    for agent in agents:
        if agent not in _AGENT_COMMANDS:
            continue
        command = find_runtime_command(agent, runtime_root=runtime_root)
        if command is None:
            errors.append(local_runtime_error_message(agent, " or ".join(_AGENT_COMMANDS[agent])))
            continue
        if error := _probe_runtime_version(
            agent,
            command,
            runtime_root=runtime_root,
            environ=env,
            sandbox=sandbox,
            strict_reads=strict_reads,
        ):
            errors.append(error)
    return errors
