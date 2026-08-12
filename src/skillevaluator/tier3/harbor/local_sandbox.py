# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OS-level confinement for Harbor local mode commands.

Strategy layer around kernel sandbox launchers. Bubblewrap on Linux provides
namespace isolation, a read-only system view, and run-dir-only writes; network
is isolated when airgapped and shared for model egress. macOS Seatbelt provides
a labelled semi-trusted backend: it confines reads, writes, network, process
metadata, and signals, but has no PID namespace and cannot guarantee cleanup of
detached descendants.
``Sandbox.wrap()`` turns a command argv into a confined argv.

This module deliberately has no skillevaluator imports so it stays portable to
the public SkillEvaluator release.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - native Windows
    resource = None

try:
    import pwd
except ImportError:  # pragma: no cover - native Windows
    pwd = None

SANDBOX_MODE_ENV = "SKILLEVALUATOR_LOCAL_SANDBOX"
ALLOW_NET_ENV = "SKILLEVALUATOR_LOCAL_ALLOW_NET"
INHERIT_AGENT_KEYS_ENV = "SKILLEVALUATOR_LOCAL_INHERIT_AGENT_KEYS"
STRICT_READS_ENV = "SKILLEVALUATOR_LOCAL_STRICT_READS"

#: require = fail closed when no kernel backend is usable (default);
#: prefer = degrade to advisory-only guardrails with a loud warning;
#: off = trusted mode on supported hosts, skip backend probing entirely.
SANDBOX_MODES = ("require", "prefer", "off")
SUPPORTED_LOCAL_SYSTEMS = frozenset({"Darwin", "Linux"})

RLIMIT_CPU_SECONDS = 900
RLIMIT_NOFILE = 4096
# RLIMIT_AS is deliberately not set: modern Node/V8 (the managed agent CLIs)
# reserves multi-GiB virtual address ranges at startup and would crash under
# any address-space cap small enough to be useful.
# RLIMIT_NPROC is deliberately NOT set: it is enforced per real-UID across ALL
# of the user's processes, not per sandbox subtree, so a fixed cap denies the
# agent's forks on a busy/shared host (EAGAIN) while providing no real
# per-trial fork-bomb protection. Process-count limits belong to cgroups /
# containers (the Docker tier), not a preexec_fn rlimit.

_TRUTHY = {"1", "true", "yes", "on"}

# Read-only view of the host system so interpreters and tools can run. Bound
# with --ro-bind-try, so entries missing on a given distro are skipped.
_SYSTEM_RO_PATHS = (
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib32",
    "/lib64",
    "/opt",
    "/etc/alternatives",
    "/etc/ssl",
    "/etc/pki",
    "/etc/resolv.conf",
    "/etc/passwd",
    "/etc/group",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/localtime",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
)
_STRICT_SYSTEM_RO_PATHS = (
    "/usr/bin",
    "/usr/sbin",
    "/usr/lib",
    "/usr/lib32",
    "/usr/lib64",
    "/usr/libexec",
    "/usr/share",
    *(path for path in _SYSTEM_RO_PATHS if path not in {"/usr", "/opt"}),
)

# Exact roots that are too broad for an explicit strict-mode exception.  A
# selected executable, venv, keg, or package below any of these roots remains
# valid; only mounting the entire shared tree is filtered.  Include both macOS
# temp spellings because /tmp and /var are normally symlink/firmlink aliases.
_STRICT_SHARED_READ_ROOTS = (
    Path("/"),
    Path("/usr"),
    Path("/usr/local"),
    Path("/opt"),
    Path("/tmp"),
    Path("/private/tmp"),
    Path("/var/tmp"),
    Path("/var"),
    Path("/private"),
    Path("/private/var"),
    Path("/etc"),
    Path("/home"),
    Path("/Users"),
    Path("/root"),
)

# macOS Seatbelt denies the entire host HOME except for the canonical paths
# required by the current run and selected runtime. Deny rules cannot be
# overridden by allow rules, so the exceptions must be part of the deny filter
# itself. Under bubblewrap the protection remains structural: host HOME is not
# mounted at all.
_SEATBELT_ABSOLUTE_LITERAL_DENY = (
    "/private/etc/master.passwd",
    "/etc/master.passwd",
)
_SEATBELT_SYSTEM_READ_ROOTS = (
    "/System",
    "/usr/bin",
    "/usr/lib",
    "/usr/libexec",
    "/usr/sbin",
    "/usr/share",
    "/bin",
    "/sbin",
    "/Library/Apple",
    "/Library/Keychains",
    "/Library/Developer",  # Command Line Tools — git/clang shims resolve the real binaries here
    "/etc",  # visible firmlink alias used by git's system configuration lookup
    "/private/etc",
    "/private/var/db",
    "/private/var/run",
    "/private/var/select",  # xcode-select's active developer-dir pointer (git/clang shims)
    "/var/select",  # macOS's visible firmlink alias used by xcode-select
    "/dev",
)


class SandboxUnavailable(RuntimeError):
    """No OS-level sandbox backend is usable on this host."""


@dataclass(frozen=True)
class SandboxPlan:
    backend: str  # "bubblewrap" | "seatbelt" | "none"
    strength: str  # "kernel" | "kernel-macos" | "advisory-only"
    reason: str  # human-readable, surfaced in run summaries


def require_supported_platform() -> str:
    """Return the supported host system or reject local mode before overrides."""
    system = platform.system()
    if system in SUPPORTED_LOCAL_SYSTEMS:
        return system
    if system == "Windows":
        raise SandboxUnavailable(
            "Native Windows local mode is unsupported, including with "
            f"{SANDBOX_MODE_ENV}=prefer or off. Use WSL2 for Linux local mode or --env-mode docker."
        )
    raise SandboxUnavailable(
        f"Local mode is unsupported on platform {system!r}. Use Linux, macOS, WSL2, or --env-mode docker."
    )


class Sandbox:
    """Wraps a command argv in an OS-level confinement launcher."""

    def __init__(self, plan: SandboxPlan):
        self.plan = plan

    def wrap(
        self,
        argv: list[str],
        *,
        workdir: Path,
        write_roots: list[Path] | tuple[Path, ...],
        home: Path,
        tmp: Path,
        allow_net: bool,
        extra_ro: list[Path],
        strict_reads: bool = False,
        deny_reads: Iterable[Path] = (),
    ) -> list[str]:
        if self.plan.backend == "bubblewrap":
            return _bwrap_argv(
                argv,
                workdir=workdir,
                write_roots=write_roots,
                home=home,
                tmp=tmp,
                allow_net=allow_net,
                extra_ro=extra_ro,
                strict_reads=strict_reads,
                deny_reads=deny_reads,
            )
        if self.plan.backend == "seatbelt":
            return _seatbelt_argv(
                argv,
                write_roots=write_roots,
                tmp=tmp,
                home=home,
                extra_ro=extra_ro,
                allow_net=allow_net,
                strict_reads=strict_reads,
                deny_reads=deny_reads,
            )
        return list(argv)


def resolve_mode(value: str | None, environ: Mapping[str, str] | None = None) -> str:
    """Resolve the sandbox mode: explicit value, then env var, then ``require``."""
    env = os.environ if environ is None else environ
    raw = value if value is not None else env.get(SANDBOX_MODE_ENV, "")
    mode = str(raw).strip().lower() or "require"
    if mode not in SANDBOX_MODES:
        raise ValueError(f"invalid local sandbox mode {raw!r}; expected one of {', '.join(SANDBOX_MODES)}")
    return mode


def coerce_flag(
    value: str | bool | None,
    *,
    env_var: str,
    environ: Mapping[str, str] | None = None,
    default: bool = False,
) -> bool:
    """Parse a boolean from a Harbor ``--ek`` string, falling back to an env var.

    ``default`` is returned only when neither an explicit value nor the env var
    is set, so an explicit ``allow_net=false`` (or ``SKILLEVALUATOR_LOCAL_ALLOW_NET=0``)
    still turns a default-True flag off.
    """
    env = os.environ if environ is None else environ
    if value is None:
        if env_var not in env:
            return default
        value = env.get(env_var, "")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUTHY


def detect(mode: str = "require") -> Sandbox:
    """Pick the strongest usable backend for this host.

    ``require`` raises :class:`SandboxUnavailable` when only advisory-only
    confinement is possible; ``prefer`` degrades with the reason recorded in
    the plan; ``off`` skips backend probing and returns advisory-only. Native
    Windows is unsupported and raises before any mode-specific escape hatch.
    """
    if mode not in SANDBOX_MODES:
        raise ValueError(f"invalid local sandbox mode {mode!r}; expected one of {', '.join(SANDBOX_MODES)}")
    system = require_supported_platform()
    if mode == "off":
        return Sandbox(SandboxPlan("none", "advisory-only", "sandbox disabled by configuration (trusted local mode)"))

    if system == "Linux":
        bwrap = shutil.which("bwrap")
        if bwrap and _userns_enabled() and _bwrap_smoke_test(bwrap):
            return Sandbox(SandboxPlan("bubblewrap", "kernel", f"bubblewrap at {bwrap}"))
        if not bwrap:
            reason = "bubblewrap not installed (install the bubblewrap package)"
        elif not _userns_enabled():
            reason = (
                "unprivileged user namespaces disabled "
                "(sysctl kernel.unprivileged_userns_clone / user.max_user_namespaces)"
            )
        else:
            reason = f"bubblewrap at {bwrap} failed its smoke test"
    elif system == "Darwin":
        if _seatbelt_available() and _seatbelt_smoke_test():
            # Seatbelt confines the filesystem (reads/writes) and network at the
            # kernel level. It is weaker than Linux bubblewrap — it does not give
            # full process isolation (a fully detached `setsid` descendant can
            # outlive cleanup) — so it is labelled semi-trusted, not untrusted-
            # safe. Use --env-mode docker for arbitrary untrusted skills.
            return Sandbox(
                SandboxPlan(
                    "seatbelt",
                    "kernel-macos",
                    "macOS Seatbelt (sandbox-exec): semi-trusted — filesystem + network confined, "
                    "weaker than Linux bubblewrap (no full process isolation); use docker for untrusted skills",
                )
            )
        reason = "sandbox-exec is unavailable; macOS local mode needs it (or use --env-mode docker)"
    else:
        reason = f"no OS sandbox backend for platform {system!r} (Windows: use WSL2 or --env-mode docker)"

    if mode == "require":
        raise SandboxUnavailable(
            f"Harbor local mode requires an OS-level sandbox, but none is usable: {reason}. "
            f"Fix the host, use --env-mode docker, or set {SANDBOX_MODE_ENV}=off to run "
            "without a sandbox for skills you fully trust."
        )
    return Sandbox(SandboxPlan("none", "advisory-only", reason))


def apply_rlimits() -> None:
    """Bound the sandboxed process subtree; used as a ``preexec_fn``.

    RLIMITs inherit across exec, so they apply to the launcher (bwrap or
    sandbox-exec) and every descendant on both Linux and macOS. Only
    async-signal-safe work is allowed here (setrlimit only).
    """
    if resource is None:
        return
    for name, desired in (
        ("RLIMIT_CPU", RLIMIT_CPU_SECONDS),
        ("RLIMIT_NOFILE", RLIMIT_NOFILE),
    ):
        res = getattr(resource, name, None)
        if res is None:
            continue
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(res, _clamped(desired, resource.getrlimit(res)))


def _clamped(desired: int, limits: tuple[int, int]) -> tuple[int, int]:
    """Clamp a desired limit to the current hard limit (soft == hard)."""
    _, hard = limits
    if resource is None:
        return (desired, desired)
    value = desired if hard == resource.RLIM_INFINITY else min(desired, hard)
    return (value, value)


def _is_within(path: Path, root: Path) -> bool:
    """True if ``path`` is ``root`` or a descendant of it (both pre-resolved)."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_shared_strict_read_root(visible: Path, resolved: Path) -> bool:
    """Return whether a strict exception is exactly one shared host tree."""
    for shared in _STRICT_SHARED_READ_ROOTS:
        shared_visible = shared.absolute()
        shared_resolved = shared_visible.resolve()
        if visible in {shared_visible, shared_resolved} or resolved in {shared_visible, shared_resolved}:
            return True
    return False


def _validated_strict_read_roots(extra_ro: Iterable[Path]) -> list[Path]:
    """Filter shared trees and reject HOME-covering strict exceptions."""
    host_homes = _safe_host_homes()
    if host_homes is None:
        raise SandboxUnavailable("strict sandbox cannot determine existing host HOME roots")
    roots: list[Path] = []
    for path in extra_ro:
        raw = Path(path).expanduser().absolute()
        visible = raw.parent.resolve() / raw.name
        resolved = visible.resolve()
        if _is_shared_strict_read_root(visible, resolved):
            continue
        contains_host_home = any(_is_within(host_home, resolved) for host_home in host_homes)
        if contains_host_home:
            raise ValueError(f"strict read root {resolved} contains the host home and is not allowed")
        roots.append(raw)
    return roots


def _strict_read_root_variants(extra_ro: Iterable[Path]) -> list[Path]:
    """Return canonical and visible aliases for strict Seatbelt exceptions.

    macOS exposes many Homebrew/system paths through firmlinks and symlinks
    (for example ``/opt/homebrew/bin/opencode``).  Seatbelt matches the path
    spelling used by the process, so allowing only the resolved target makes a
    valid runtime fail with ``EPERM``.  Keep both spellings, while applying the
    same host-home safety validation to each root.
    """
    candidates = _validated_strict_read_roots(extra_ro)
    roots: list[Path] = []
    seen: set[Path] = set()
    host_homes = _safe_host_homes()
    if host_homes is None:
        raise SandboxUnavailable("strict sandbox cannot determine existing host HOME roots")
    for candidate in candidates:
        raw = Path(candidate).expanduser().absolute()
        try:
            visible = raw.parent.resolve(strict=True) / raw.name
            resolved = visible.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SandboxUnavailable(f"macOS local sandbox cannot resolve runtime exception: {candidate}") from exc
        if _is_shared_strict_read_root(visible, resolved):
            continue
        for root in (visible, resolved):
            root = root.resolve() if not root.is_symlink() else root
            if any(_is_within(home, root) for home in host_homes):
                raise ValueError(f"strict read root {root} contains the host home and is not allowed")
            if root not in seen:
                seen.add(root)
                roots.append(root)
    return roots


def _userns_enabled() -> bool:
    # Debian and some hardened kernels ship unprivileged user namespaces off.
    for path, disabled in (
        ("/proc/sys/kernel/unprivileged_userns_clone", "0"),
        ("/proc/sys/user/max_user_namespaces", "0"),
    ):
        try:
            if Path(path).read_text(encoding="ascii").strip() == disabled:
                return False
        except OSError:
            pass  # sysctl absent => not the blocking factor
    return True


def _bwrap_smoke_argv(bwrap: str) -> list[str]:
    """Argv proving bwrap can create a namespace AND exec a real binary.

    Binds the exec-critical system dirs, not just /usr: ``/usr/bin/true`` is
    dynamically linked, so its ELF interpreter under ``/lib``/``/lib64`` must be
    present in the namespace. On usr-merged distros ``/lib`` is a root symlink
    that is absent inside the mount namespace unless bound, so ``--ro-bind /usr``
    alone makes exec fail with ENOENT and the smoke test would falsely report
    bwrap unusable on every real Linux host.
    """
    argv = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for path in ("/bin", "/lib", "/lib64", "/lib32"):
        argv += ["--ro-bind-try", path, path]
    argv += ["--", "/usr/bin/true"]
    return argv


def _bwrap_smoke_test(bwrap: str) -> bool:
    # Some hosts have bwrap installed but user namespaces refused at runtime
    # (containers, seccomp policies); prove the launcher actually works.
    try:
        result = subprocess.run(_bwrap_smoke_argv(bwrap), capture_output=True, timeout=5)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _seatbelt_available() -> bool:
    return Path("/usr/bin/sandbox-exec").exists()


def _seatbelt_smoke_test() -> bool:
    try:
        result = subprocess.run(
            ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _bwrap_argv(
    argv: list[str],
    *,
    workdir: Path,
    write_roots: list[Path] | tuple[Path, ...],
    home: Path,
    tmp: Path,
    allow_net: bool,
    extra_ro: list[Path],
    strict_reads: bool = False,
    deny_reads: Iterable[Path] = (),
) -> list[str]:
    """Build the bubblewrap launcher argv.

    Read-only system view, the run directories as the only writable binds, no
    inherited privileges, loopback-only network unless ``allow_net``. The env
    is intentionally not cleared: the caller already passes a curated env to
    the subprocess, and ``--clearenv`` would discard it.
    """
    wrapper = [
        "bwrap",
        "--unshare-all",  # user, ipc, pid, uts, cgroup, AND net (loopback only)
        "--die-with-parent",
        "--new-session",  # blocks TIOCSTI terminal-injection escapes
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    if allow_net:
        wrapper.append("--share-net")  # opt-in only

    system_ro_paths = _STRICT_SYSTEM_RO_PATHS if strict_reads else _SYSTEM_RO_PATHS
    for path in system_ro_paths:
        wrapper += ["--ro-bind-try", path, path]

    created_bind_dirs: set[Path] = set()
    # Skip exact binds and parent creation only below roots actually mounted by
    # the selected policy. Strict mode deliberately omits broad /usr and /opt.
    system_mount_roots = tuple(Path(path) for path in system_ro_paths)
    mounted_subtree_roots = (*system_mount_roots, Path("/proc"), Path("/dev"))
    directory_anchors = (Path("/tmp"),)

    def ensure_bind_parents(path: Path) -> None:
        if any(_is_within(path, root) for root in mounted_subtree_roots):
            return
        anchor = next((root for root in directory_anchors if _is_within(path, root)), None)
        if anchor is None:
            current = Path(path.anchor)
            parts = path.parent.parts[1:]
        else:
            current = anchor
            parts = path.parent.relative_to(anchor).parts
        for part in parts:
            current /= part
            if current in created_bind_dirs:
                continue
            wrapper.extend(["--dir", str(current)])
            created_bind_dirs.add(current)

    # Parents first so nested roots layer correctly; the run dirs are bound at
    # their real host paths, so existing path-resolution logic keeps working.
    resolved_write_roots = sorted({Path(root).resolve() for root in write_roots}, key=lambda p: len(p.parts))
    for root in resolved_write_roots:
        ensure_bind_parents(root)
        wrapper += ["--bind", str(root), str(root)]

    read_roots = _validated_strict_read_roots(extra_ro) if strict_reads else extra_ro
    published_read_roots: list[Path] = []
    for ro_path in read_roots:
        raw = Path(ro_path).expanduser().absolute()
        try:
            visible = raw.parent.resolve(strict=True) / raw.name
            resolved = visible.resolve(strict=True)
        except OSError:
            continue
        # The broad system mount already publishes this visible entry. In
        # particular, bwrap accepts an existing symlink only when its raw target
        # exactly matches; replacing a relative host link with our canonical
        # target would fail. A separately supplied canonical target is still
        # processed below. If the system link traverses an outside, unpublished
        # intermediate alias it can remain dangling, but the sandboxed version
        # preflight fails closed and reports that unusable runtime.
        if any(_is_within(visible, root) for root in system_mount_roots):
            continue
        # Skip an extra_ro path that lives inside a writable run root: a trailing
        # --ro-bind would re-mount that subtree read-only inside the writable
        # tree, so a skill writing there would get EROFS.
        if resolved.exists() and not any(_is_within(resolved, root) for root in resolved_write_roots):
            published_read_roots.extend((visible, resolved))
            ensure_bind_parents(visible)
            if visible.is_symlink():
                # Intermediate aliases are not published, so point the final
                # link straight at the separately bound canonical target.
                wrapper += ["--symlink", str(resolved), str(visible)]
            else:
                wrapper += ["--ro-bind", str(resolved), str(visible)]

    published_roots = (*system_mount_roots, *resolved_write_roots, *published_read_roots)
    for denied_path in deny_reads:
        try:
            raw_denied = Path(denied_path).expanduser().absolute()
            denied_variants = {raw_denied, raw_denied.resolve(strict=True)}
        except (OSError, RuntimeError):
            continue
        for denied in denied_variants:
            if not any(_is_within(denied, root) for root in published_roots):
                continue
            ensure_bind_parents(denied)
            wrapper += ["--ro-bind", "/dev/null", str(denied)]

    wrapper += [
        "--setenv",
        "HOME",
        str(home),
        "--setenv",
        "TMPDIR",
        str(tmp),
        "--chdir",
        str(workdir),
        "--",
    ]
    return wrapper + list(argv)


def _sbpl_quote(path: Path | str) -> str:
    """Escape a path for a double-quoted SBPL string literal.

    Backslash first, then double-quote, so a run/working dir containing either
    can't produce a malformed profile (parse abort) or widen the allow-list.
    """
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _safe_host_homes() -> tuple[Path, ...] | None:
    """Return canonical environment and passwd homes, or fail closed."""
    env_home = os.environ.get("HOME")
    candidates: list[Path] = []
    if env_home:
        candidates.append(Path(env_home))
    if pwd is not None:
        try:
            passwd_home = pwd.getpwuid(os.getuid()).pw_dir
        except (KeyError, OSError):
            return None
        if not passwd_home:
            return None
        candidates.append(Path(passwd_home))
    elif platform.system() == "Darwin":
        return None

    homes: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve(strict=True)
        except (RuntimeError, OSError):
            return None
        if not resolved.is_dir():
            return None
        if resolved not in homes:
            homes.append(resolved)
    return tuple(homes) if homes else None


def _seatbelt_path_filter(path: Path) -> str:
    """Return the narrow SBPL filter for one existing canonical path."""
    if path.is_file():
        operation = "literal"
    elif path.is_dir():
        operation = "subpath"
    else:
        raise SandboxUnavailable(f"macOS local sandbox exception is not a file or directory: {path}")
    return f'        ({operation} "{_sbpl_quote(path)}")'


def _minimal_roots(paths: Iterable[Path]) -> list[Path]:
    """Drop paths already covered by an earlier directory exception."""
    minimal: list[Path] = []
    for path in sorted(set(paths), key=lambda candidate: len(candidate.parts)):
        if any(parent.is_dir() and _is_within(path, parent) for parent in minimal):
            continue
        minimal.append(path)
    return minimal


def _seatbelt_read_deny_rule(
    operation: str,
    scope: Path,
    read_roots: Iterable[Path],
    *,
    exact_traversal_paths: Iterable[Path] = (),
) -> str:
    """Deny one read operation outside full roots and exact traversal nodes."""
    full_roots = _minimal_roots(read_roots)
    exact_ancestors = sorted(set(exact_traversal_paths).difference(full_roots), key=lambda path: len(path.parts))
    filters = [f'        ({"literal" if path.is_file() else "subpath"} "{_sbpl_quote(path)}")' for path in full_roots]
    filters.extend(f'        (literal "{_sbpl_quote(path)}")' for path in exact_ancestors)
    if not filters:
        return f'(deny {operation} (subpath "{_sbpl_quote(scope)}"))\n'
    return (
        f"(deny {operation}\n"
        "  (require-all\n"
        f'    (subpath "{_sbpl_quote(scope)}")\n'
        "    (require-not\n"
        "      (require-any\n"
        f"{'\n'.join(filters)}))))\n"
    )


def _seatbelt_metadata_ancestors(scope: Path, read_roots: Iterable[Path]) -> set[Path]:
    """Return exact existing ancestors needed to traverse to approved roots."""
    ancestors: set[Path] = set()
    for root in read_roots:
        for ancestor in root.parents:
            if not _is_within(ancestor, scope):
                break
            ancestors.add(ancestor)
            if ancestor == scope:
                break
    return ancestors


def _seatbelt_home_read_rule(host_home: Path, read_roots: Iterable[Path]) -> str:
    """Deny host-HOME reads while allowing metadata-only path traversal."""
    try:
        home = host_home.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SandboxUnavailable(f"macOS local sandbox cannot resolve host HOME: {host_home}") from exc
    if not home.is_dir():
        raise SandboxUnavailable(f"macOS local sandbox host HOME is not a directory: {home}")

    exceptions: list[Path] = []
    for candidate in read_roots:
        try:
            raw = Path(candidate).expanduser().absolute()
            visible = raw.parent.resolve(strict=True) / raw.name
            resolved = visible.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SandboxUnavailable(f"macOS local sandbox cannot resolve runtime exception: {candidate}") from exc
        variants = [visible, resolved] if visible.is_symlink() else [resolved]
        for exception in variants:
            if _is_within(home, exception):
                raise SandboxUnavailable(f"macOS local sandbox runtime exception would expose host HOME: {exception}")
            if _is_within(exception, home):
                exceptions.append(exception)

    full_roots = _minimal_roots(exceptions)
    metadata_ancestors = _seatbelt_metadata_ancestors(home, full_roots)
    operation_rules = "".join(
        (
            _seatbelt_read_deny_rule("file-read-data", home, full_roots),
            _seatbelt_read_deny_rule("file-read-xattr", home, full_roots),
            _seatbelt_read_deny_rule(
                "file-read-metadata",
                home,
                full_roots,
                exact_traversal_paths=metadata_ancestors,
            ),
        )
    )
    filters = "\n".join(_seatbelt_path_filter(path) for path in full_roots)
    if filters:
        broad_read_rule = (
            "(deny file-read*\n"
            "  (require-all\n"
            f'    (subpath "{_sbpl_quote(home)}")\n'
            "    (require-not\n"
            "      (require-any\n"
            f"{filters}))))\n"
        )
    else:
        broad_read_rule = f'(deny file-read* (subpath "{_sbpl_quote(home)}"))\n'

    traversal_allows = "\n".join(
        f'(allow file-read-metadata (literal "{_sbpl_quote(path)}"))'
        for path in sorted(metadata_ancestors, key=lambda path: len(path.parts))
    )
    return f"{operation_rules}{broad_read_rule}{traversal_allows}\n"


def _seatbelt_argv(
    argv: list[str],
    *,
    write_roots: list[Path] | tuple[Path, ...],
    tmp: Path,
    home: Path,
    extra_ro: list[Path],
    allow_net: bool,
    strict_reads: bool = False,
    deny_reads: Iterable[Path] = (),
) -> list[str]:
    """Build the macOS Seatbelt launcher argv.

    Reads: deny host HOME except canonical run/runtime paths. Writes: confined
    to the run dirs. Network: per ``allow_net``. Coarser than bubblewrap's
    deny-by-construction and therefore best-effort for semi-trusted skills;
    Linux/bwrap is the strong path and Docker is the untrusted tier.
    """
    write_roots_sorted = sorted({Path(root).resolve() for root in (*write_roots, tmp)}, key=lambda p: len(p.parts))
    write_allows = "\n".join(f'  (subpath "{_sbpl_quote(root)}")' for root in write_roots_sorted)

    if strict_reads:
        extra_ro = _strict_read_root_variants(extra_ro)
        strict_extra_roots = {
            Path(root) if Path(root).is_symlink() else Path(root).resolve() for root in extra_ro if Path(root).exists()
        }
        explicit_read_roots = {
            *(Path(root).resolve() for root in write_roots_sorted),
            Path(home).resolve(),
            *strict_extra_roots,
            *(
                variant
                for root in _SEATBELT_SYSTEM_READ_ROOTS
                if Path(root).exists()
                for variant in (Path(root), Path(root).resolve())
            ),
        }
        root_scope = Path("/")
        metadata_traversal_roots = _seatbelt_metadata_ancestors(root_scope, explicit_read_roots)
        operation_rules = "".join(
            (
                # sandbox-exec aborts before launching even an approved binary
                # unless it can read the exact filesystem root node. Keep that
                # one fixed bootstrap literal; all derived ancestors remain
                # metadata-only so their directory entries cannot be listed.
                _seatbelt_read_deny_rule(
                    "file-read-data",
                    root_scope,
                    explicit_read_roots,
                    exact_traversal_paths=(root_scope,),
                ),
                _seatbelt_read_deny_rule("file-read-xattr", root_scope, explicit_read_roots),
                _seatbelt_read_deny_rule(
                    "file-read-metadata",
                    root_scope,
                    explicit_read_roots,
                    exact_traversal_paths=metadata_traversal_roots,
                ),
            )
        )
        traversal_roots = {ancestor for root in explicit_read_roots for ancestor in root.parents}
        exception_filters = "\n".join(
            [
                *(
                    f'    (require-not (literal "{_sbpl_quote(path)}"))'
                    for path in sorted(traversal_roots, key=lambda item: len(item.parts))
                ),
                *(
                    f'    (require-not ({"subpath" if path.is_dir() else "literal"} "{_sbpl_quote(path)}"))'
                    for path in sorted(explicit_read_roots, key=lambda item: len(item.parts))
                ),
            ]
        )
        broad_read_rule = f'(deny file-read*\n  (require-all\n    (subpath "/")\n{exception_filters}\n  ))\n'
        read_policy = f"{operation_rules}{broad_read_rule}"
    else:
        host_homes = _safe_host_homes()
        if host_homes is None:
            raise SandboxUnavailable("macOS local sandbox cannot determine existing host HOME roots")
        home_read_rule = "".join(
            _seatbelt_home_read_rule(
                host_home,
                [*write_roots, tmp, home, *extra_ro],
            )
            for host_home in host_homes
        )
        absolute_read_denies = "\n".join(
            f'(deny file-read* (literal "{_sbpl_quote(path)}"))' for path in _SEATBELT_ABSOLUTE_LITERAL_DENY
        )
        read_policy = f"{home_read_rule}{absolute_read_denies}\n"

    explicit_read_denies = "\n".join(
        f'(deny file-read* (literal "{_sbpl_quote(path)}"))'
        for candidate in deny_reads
        for path in {Path(candidate).expanduser().absolute(), Path(candidate).expanduser().resolve(strict=False)}
    )
    if explicit_read_denies:
        read_policy = f"{read_policy}{explicit_read_denies}\n"

    if allow_net:
        # Keep the host IPC boundary while permitting model-serving traffic.
        # macOS DNS uses this one system Unix socket; all other Unix sockets,
        # listeners, and explicit binds remain denied.
        net_rule = """(deny network-inbound)
(deny network-bind)
(deny network-outbound
  (require-all
    (remote unix-socket)
    (require-not (remote unix-socket (path "/private/var/run/mDNSResponder")))))"""
    else:
        net_rule = "(deny network*)"
    # Deny cross-process introspection (pgrep/ps reading another same-user
    # process's argv/metadata) while still allowing the skill to inspect its own
    # process tree. Closes the Seatbelt process-metadata leak.
    process_info_rule = "(deny process-info*)\n(allow process-info* (target self))\n"
    # Deny signalling unrelated same-user processes. The skill may still manage
    # its own process tree: exec() runs the sandbox in a fresh session/process
    # group, so `self` + `pgrp` covers the agent and its children, while other
    # processes (a different process group) cannot be signalled.
    signal_rule = "(deny signal)\n(allow signal (target self))\n(allow signal (target pgrp))\n"
    profile = f"""(version 1)
(allow default)
{read_policy}{process_info_rule}{signal_rule}(deny file-write*)
(allow file-write*
{write_allows}
  (literal "/dev/null")
  (literal "/dev/stdout")
  (literal "/dev/stderr")
  (literal "/dev/dtracehelper")
  (literal "/dev/tty")
  (literal "/dev/urandom"))
{net_rule}
"""
    return ["/usr/bin/sandbox-exec", "-p", profile, *argv]
