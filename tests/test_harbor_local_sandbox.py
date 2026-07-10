# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor import local_runtime, local_sandbox

try:
    import resource
except ImportError:  # pragma: no cover - exercised by native Windows CI
    resource = None

try:
    import pwd
except ImportError:  # pragma: no cover - native Windows
    pwd = None


def _plan(backend: str, strength: str = "kernel", reason: str = "test") -> local_sandbox.SandboxPlan:
    return local_sandbox.SandboxPlan(backend=backend, strength=strength, reason=reason)


class TestDetect:
    def test_off_mode_returns_advisory_only_without_probing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("off mode must not probe the host")

        monkeypatch.setattr(local_sandbox.platform, "system", lambda: "Linux")
        monkeypatch.setattr(local_sandbox.shutil, "which", boom)
        sandbox = local_sandbox.detect("off")
        assert sandbox.plan.backend == "none"
        assert sandbox.plan.strength == "advisory-only"
        assert "disabled" in sandbox.plan.reason

    def test_linux_with_working_bubblewrap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(local_sandbox.platform, "system", lambda: "Linux")
        monkeypatch.setattr(local_sandbox.shutil, "which", lambda _name: "/usr/bin/bwrap")
        monkeypatch.setattr(local_sandbox, "_userns_enabled", lambda: True)
        monkeypatch.setattr(local_sandbox, "_bwrap_smoke_test", lambda _bwrap: True)
        sandbox = local_sandbox.detect("require")
        assert sandbox.plan.backend == "bubblewrap"
        assert sandbox.plan.strength == "kernel"

    def test_linux_userns_disabled_fails_closed_in_require_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(local_sandbox.platform, "system", lambda: "Linux")
        monkeypatch.setattr(local_sandbox.shutil, "which", lambda _name: "/usr/bin/bwrap")
        monkeypatch.setattr(local_sandbox, "_userns_enabled", lambda: False)
        with pytest.raises(local_sandbox.SandboxUnavailable) as excinfo:
            local_sandbox.detect("require")
        message = str(excinfo.value)
        assert "user namespaces" in message
        assert "SKILLEVALUATOR_LOCAL_SANDBOX" in message  # remediation hint

    def test_linux_without_bubblewrap_degrades_in_prefer_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(local_sandbox.platform, "system", lambda: "Linux")
        monkeypatch.setattr(local_sandbox.shutil, "which", lambda _name: None)
        sandbox = local_sandbox.detect("prefer")
        assert sandbox.plan.backend == "none"
        assert sandbox.plan.strength == "advisory-only"
        assert "not installed" in sandbox.plan.reason

    def test_macos_uses_seatbelt_as_semi_trusted_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(local_sandbox.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(local_sandbox, "_seatbelt_available", lambda: True)
        monkeypatch.setattr(local_sandbox, "_seatbelt_smoke_test", lambda: True)

        sandbox = local_sandbox.detect("require")
        assert sandbox.plan.backend == "seatbelt"
        assert sandbox.plan.strength == "kernel-macos"
        assert "semi-trusted" in sandbox.plan.reason
        assert "docker" in sandbox.plan.reason.lower()  # points untrusted users to docker

    def test_macos_without_sandbox_exec_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(local_sandbox.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(local_sandbox, "_seatbelt_available", lambda: False)
        with pytest.raises(local_sandbox.SandboxUnavailable, match="sandbox-exec"):
            local_sandbox.detect("require")

    @pytest.mark.parametrize("mode", local_sandbox.SANDBOX_MODES)
    def test_native_windows_local_mode_fails_closed_for_every_sandbox_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mode: str,
    ) -> None:
        monkeypatch.setattr(local_sandbox.platform, "system", lambda: "Windows")
        with pytest.raises(local_sandbox.SandboxUnavailable, match="Native Windows local mode is unsupported"):
            local_sandbox.detect(mode)

    @pytest.mark.parametrize("system", ["FreeBSD", "CYGWIN_NT-10.0", "MSYS_NT-10.0"])
    @pytest.mark.parametrize("mode", local_sandbox.SANDBOX_MODES)
    def test_other_unsupported_platforms_fail_closed_for_every_sandbox_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        system: str,
        mode: str,
    ) -> None:
        monkeypatch.setattr(local_sandbox.platform, "system", lambda: system)
        with pytest.raises(local_sandbox.SandboxUnavailable, match="Local mode is unsupported on platform"):
            local_sandbox.detect(mode)

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            local_sandbox.detect("bogus")

    def test_bwrap_smoke_argv_binds_lib_dirs(self) -> None:
        # Regression: the smoke test must bind the ELF-interpreter dirs, not just
        # /usr, or a dynamically-linked /usr/bin/true fails ENOENT on usr-merged
        # Linux and bwrap is falsely reported unusable on every real host.
        argv = local_sandbox._bwrap_smoke_argv("bwrap")
        joined = " ".join(argv)
        assert "--ro-bind-try /lib /lib" in joined
        assert "--ro-bind-try /lib64 /lib64" in joined
        assert argv[-1] == "/usr/bin/true"

    def test_bwrap_smoke_argv_matches_real_namespace_shape(self) -> None:
        argv = local_sandbox._bwrap_smoke_argv("bwrap")
        for flag in ("--die-with-parent", "--new-session", "--proc", "--dev", "--tmpfs"):
            assert flag in argv


@pytest.mark.skipif(platform.system() != "Windows", reason="requires a native Windows runner")
@pytest.mark.parametrize("mode", local_sandbox.SANDBOX_MODES)
def test_actual_native_windows_host_rejects_local_mode(mode: str) -> None:
    with pytest.raises(local_sandbox.SandboxUnavailable, match="Native Windows local mode is unsupported"):
        local_sandbox.detect(mode)


class TestResolveModeAndFlags:
    def test_explicit_value_wins_over_env(self) -> None:
        mode = local_sandbox.resolve_mode("off", environ={"SKILLEVALUATOR_LOCAL_SANDBOX": "require"})
        assert mode == "off"

    def test_env_var_used_when_no_explicit_value(self) -> None:
        mode = local_sandbox.resolve_mode(None, environ={"SKILLEVALUATOR_LOCAL_SANDBOX": "prefer"})
        assert mode == "prefer"

    def test_default_is_require(self) -> None:
        assert local_sandbox.resolve_mode(None, environ={}) == "require"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            local_sandbox.resolve_mode("yolo", environ={})

    def test_coerce_flag_parses_ek_strings(self) -> None:
        assert local_sandbox.coerce_flag("true", env_var="X", environ={}) is True
        assert local_sandbox.coerce_flag("1", env_var="X", environ={}) is True
        assert local_sandbox.coerce_flag("false", env_var="X", environ={}) is False
        assert local_sandbox.coerce_flag(True, env_var="X", environ={}) is True

    def test_coerce_flag_falls_back_to_env_var(self) -> None:
        assert local_sandbox.coerce_flag(None, env_var="X", environ={"X": "yes"}) is True
        assert local_sandbox.coerce_flag(None, env_var="X", environ={}) is False

    def test_coerce_flag_default_applies_only_when_unset(self) -> None:
        # unset -> default; explicit false (env or value) overrides a True default
        assert local_sandbox.coerce_flag(None, env_var="X", environ={}, default=True) is True
        assert local_sandbox.coerce_flag(None, env_var="X", environ={"X": "false"}, default=True) is False
        assert local_sandbox.coerce_flag("0", env_var="X", environ={}, default=True) is False


class TestBwrapArgv:
    @staticmethod
    def _option_operands(argv: list[str], option: str, count: int = 2) -> set[tuple[str, ...]]:
        return {tuple(argv[index + 1 : index + count + 1]) for index, value in enumerate(argv) if value == option}

    @staticmethod
    def _pretend_absolute_paths_exist(
        monkeypatch: pytest.MonkeyPatch,
        *paths: Path,
    ) -> None:
        """Model Linux runtime layouts that must not be created on the test host."""
        simulated = {path.absolute() for path in paths}
        real_resolve = Path.resolve
        real_exists = Path.exists
        real_is_symlink = Path.is_symlink

        def resolve(path: Path, strict: bool = False) -> Path:
            absolute = path.absolute()
            if absolute in simulated:
                return absolute
            return real_resolve(path, strict=strict)

        def exists(path: Path) -> bool:
            return path.absolute() in simulated or real_exists(path)

        def is_symlink(path: Path) -> bool:
            return False if path.absolute() in simulated else real_is_symlink(path)

        monkeypatch.setattr(Path, "resolve", resolve)
        monkeypatch.setattr(Path, "exists", exists)
        monkeypatch.setattr(Path, "is_symlink", is_symlink)

    def _argv(
        self,
        tmp_path: Path,
        *,
        allow_net: bool = False,
        extra_ro: list[Path] | None = None,
        strict_reads: bool = False,
    ) -> list[str]:
        run_root = tmp_path / "run"
        (run_root / "workspace").mkdir(parents=True, exist_ok=True)
        return local_sandbox._bwrap_argv(
            ["bash", "-c", "echo hi"],
            workdir=run_root / "workspace",
            write_roots=[run_root],
            home=run_root / "home",
            tmp=run_root / "tmp",
            allow_net=allow_net,
            extra_ro=extra_ro or [],
            strict_reads=strict_reads,
        )

    def test_isolation_flags_present(self, tmp_path: Path) -> None:
        argv = self._argv(tmp_path)
        assert argv[0] == "bwrap"
        for flag in ("--unshare-all", "--die-with-parent", "--new-session"):
            assert flag in argv

    def test_network_denied_by_default_and_opt_in(self, tmp_path: Path) -> None:
        assert "--share-net" not in self._argv(tmp_path)
        assert "--share-net" in self._argv(tmp_path, allow_net=True)

    def test_run_root_is_only_writable_bind(self, tmp_path: Path) -> None:
        argv = self._argv(tmp_path)
        run_root = str(tmp_path / "run")
        bind_indices = [i for i, piece in enumerate(argv) if piece == "--bind"]
        assert len(bind_indices) == 1
        assert argv[bind_indices[0] + 1] == run_root
        assert argv[bind_indices[0] + 2] == run_root

    def test_extra_ro_binds_appended(self, tmp_path: Path) -> None:
        runtime = tmp_path / "runtimes"
        runtime.mkdir()
        argv = self._argv(tmp_path, extra_ro=[runtime, tmp_path / "missing"])
        joined = " ".join(argv)
        assert f"--ro-bind {runtime} {runtime}" in joined
        assert "missing" not in joined

    def test_exact_ro_skips_direct_path_covered_by_system_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        system_root = tmp_path / "system"
        command = system_root / "bin" / "opencode"
        command.parent.mkdir(parents=True)
        command.write_text("binary", encoding="utf-8")
        monkeypatch.setattr(local_sandbox, "_SYSTEM_RO_PATHS", (str(system_root),))

        argv = self._argv(tmp_path, extra_ro=[command])

        joined = " ".join(argv)
        assert f"--ro-bind-try {system_root} {system_root}" in joined
        assert f"--ro-bind {command} {command}" not in joined

    def test_system_symlink_is_not_republished_but_external_target_is_bound(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        system_root = tmp_path / "system"
        command = system_root / "bin" / "opencode"
        command.parent.mkdir(parents=True)
        target = tmp_path / "external" / "opencode"
        target.parent.mkdir()
        target.write_text("binary", encoding="utf-8")
        command.symlink_to(target)
        monkeypatch.setattr(local_sandbox, "_SYSTEM_RO_PATHS", (str(system_root),))

        argv = self._argv(tmp_path, extra_ro=[command, target])

        symlink_destinations = [argv[index + 2] for index, value in enumerate(argv) if value == "--symlink"]
        assert str(command) not in symlink_destinations
        assert f"--ro-bind {target} {target}" in " ".join(argv)

    def test_exact_symlink_runtime_does_not_bind_sibling_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "tools" / "opencode"
        target.parent.mkdir()
        target.write_text("binary", encoding="utf-8")
        command = tmp_path / ".local" / "bin" / "opencode"
        command.parent.mkdir(parents=True)
        command.symlink_to(target)
        monkeypatch.setattr(local_sandbox, "_safe_host_homes", lambda: (tmp_path.resolve(),))

        argv = self._argv(tmp_path, extra_ro=[command.absolute(), target.resolve()])

        joined = " ".join(argv)
        assert f"--symlink {target} {command}" in joined
        assert f"--ro-bind {target} {target}" in joined
        assert f"--ro-bind {command.parent} {command.parent}" not in joined

    def test_exact_symlink_uses_canonical_target_instead_of_unpublished_alias(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "real" / "opencode"
        target.parent.mkdir()
        target.write_text("binary", encoding="utf-8")
        alias = tmp_path / "alias"
        alias.symlink_to(target.parent, target_is_directory=True)
        command = tmp_path / "bin" / "opencode"
        command.parent.mkdir()
        command.symlink_to(Path("../alias/opencode"))

        argv = self._argv(tmp_path, extra_ro=[command, target])

        symlink_index = argv.index("--symlink")
        assert argv[symlink_index + 1 : symlink_index + 3] == [str(target.resolve()), str(command)]
        assert f"--ro-bind {target} {target}" in " ".join(argv)

    def test_exact_bind_creates_non_system_parent_directories(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        command = tmp_path / "custom" / "bin" / "opencode"
        command.parent.mkdir(parents=True)
        command.write_text("binary", encoding="utf-8")
        monkeypatch.setattr(local_sandbox, "_safe_host_homes", lambda: ())

        argv = self._argv(tmp_path, extra_ro=[command])

        joined = " ".join(argv)
        assert f"--dir {command.parent}" in joined
        assert f"--ro-bind {command} {command}" in joined

    def test_strict_exact_usr_local_runtime_is_published_without_broad_usr_mount(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Path("/usr/local/bin/opencode")
        self._pretend_absolute_paths_exist(monkeypatch, runtime.parent, runtime)
        monkeypatch.setattr(local_sandbox, "_safe_host_homes", lambda: ())

        argv = self._argv(tmp_path, extra_ro=[runtime], strict_reads=True)

        ro_bind_try = self._option_operands(argv, "--ro-bind-try")
        ro_bind = self._option_operands(argv, "--ro-bind")
        directories = self._option_operands(argv, "--dir", count=1)
        assert ("/usr", "/usr") not in ro_bind_try
        assert ("/opt", "/opt") not in ro_bind_try
        assert ("/usr/local",) in directories
        assert ("/usr/local/bin",) in directories
        assert (str(runtime), str(runtime)) in ro_bind
        assert ("/usr/local", "/usr/local") not in ro_bind
        assert ("/usr/local/bin", "/usr/local/bin") not in ro_bind

    def test_strict_usr_local_style_npm_symlink_keeps_alias_and_package_exact(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        prefix = tmp_path / "host" / "usr" / "local"
        package = prefix / "lib" / "node_modules" / "opencode-ai"
        target = package / "bin" / "opencode"
        target.parent.mkdir(parents=True)
        target.write_text("binary", encoding="utf-8")
        command = prefix / "bin" / "opencode"
        command.parent.mkdir(parents=True)
        command.symlink_to(Path("../lib/node_modules/opencode-ai/bin/opencode"))
        monkeypatch.setattr(local_sandbox, "_safe_host_homes", lambda: ())

        argv = self._argv(
            tmp_path,
            extra_ro=[command.absolute(), package.resolve()],
            strict_reads=True,
        )

        symlinks = self._option_operands(argv, "--symlink")
        ro_bind = self._option_operands(argv, "--ro-bind")
        assert (str(target.resolve()), str(command)) in symlinks
        assert (str(package), str(package)) in ro_bind
        assert (str(prefix), str(prefix)) not in ro_bind
        assert (str(command.parent), str(command.parent)) not in ro_bind

    def test_strict_exact_opt_venv_creates_parents_without_broad_opt_mount(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        interpreter = Path("/opt/se-venv/bin/python")
        self._pretend_absolute_paths_exist(monkeypatch, interpreter.parent, interpreter)
        monkeypatch.setattr(local_sandbox, "_safe_host_homes", lambda: ())

        argv = self._argv(tmp_path, extra_ro=[interpreter], strict_reads=True)

        ro_bind_try = self._option_operands(argv, "--ro-bind-try")
        ro_bind = self._option_operands(argv, "--ro-bind")
        directories = self._option_operands(argv, "--dir", count=1)
        assert ("/opt", "/opt") not in ro_bind_try
        assert ("/opt",) in directories
        assert ("/opt/se-venv",) in directories
        assert ("/opt/se-venv/bin",) in directories
        assert (str(interpreter), str(interpreter)) in ro_bind
        assert ("/opt/se-venv", "/opt/se-venv") not in ro_bind
        assert ("/opt/se-venv/bin", "/opt/se-venv/bin") not in ro_bind

    def test_non_strict_system_mounts_cover_usr_local_runtime_and_opt_venv(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Path("/usr/local/bin/opencode")
        interpreter = Path("/opt/se-venv/bin/python")
        self._pretend_absolute_paths_exist(
            monkeypatch,
            runtime.parent,
            runtime,
            interpreter.parent,
            interpreter,
        )

        argv = self._argv(tmp_path, extra_ro=[runtime, interpreter])

        ro_bind_try = self._option_operands(argv, "--ro-bind-try")
        ro_bind = self._option_operands(argv, "--ro-bind")
        directories = self._option_operands(argv, "--dir", count=1)
        assert ("/usr", "/usr") in ro_bind_try
        assert ("/opt", "/opt") in ro_bind_try
        assert (str(runtime), str(runtime)) not in ro_bind
        assert (str(interpreter), str(interpreter)) not in ro_bind
        assert ("/usr/local",) not in directories
        assert ("/opt/se-venv",) not in directories

    def test_exact_bind_creates_descendants_below_tmp_anchor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lexical_root = Path(tempfile.mkdtemp(prefix="skillevaluator-bwrap-", dir="/tmp"))
        command = lexical_root / "custom" / "bin" / "opencode"
        command.parent.mkdir(parents=True)
        command.write_text("binary", encoding="utf-8")
        real_resolve = Path.resolve

        def preserve_lexical_tmp(path: Path, strict: bool = False) -> Path:
            if str(path).startswith("/tmp/"):
                if strict and not path.exists():
                    raise FileNotFoundError(path)
                return path.absolute()
            return real_resolve(path, strict=strict)

        monkeypatch.setattr(Path, "resolve", preserve_lexical_tmp)
        try:
            argv = self._argv(tmp_path, extra_ro=[command])
        finally:
            shutil.rmtree(lexical_root, ignore_errors=True)

        bind_index = argv.index("--ro-bind", argv.index(str(command)) - 1)
        dir_index = max(index for index, value in enumerate(argv) if value == str(command.parent)) - 1
        assert argv[dir_index : dir_index + 2] == ["--dir", str(command.parent)]
        assert dir_index < bind_index

    def test_chdir_and_env_inside_sandbox(self, tmp_path: Path) -> None:
        argv = self._argv(tmp_path)
        run_root = tmp_path / "run"
        assert "--chdir" in argv
        assert str(run_root / "workspace") == argv[argv.index("--chdir") + 1]
        joined = " ".join(argv)
        assert f"--setenv HOME {run_root / 'home'}" in joined
        assert f"--setenv TMPDIR {run_root / 'tmp'}" in joined

    def test_command_argv_preserved_after_separator(self, tmp_path: Path) -> None:
        argv = self._argv(tmp_path)
        assert argv[argv.index("--") + 1 :] == ["bash", "-c", "echo hi"]


class TestSeatbeltArgv:
    def _profile(self, tmp_path: Path, *, allow_net: bool = False) -> str:
        (tmp_path / "run" / "tmp").mkdir(parents=True, exist_ok=True)
        (tmp_path / "run" / "home").mkdir(exist_ok=True)
        argv = local_sandbox._seatbelt_argv(
            ["bash", "-c", "echo hi"],
            write_roots=[tmp_path / "run"],
            tmp=tmp_path / "run" / "tmp",
            home=tmp_path / "run" / "home",
            extra_ro=[],
            allow_net=allow_net,
        )
        assert argv[0] == "/usr/bin/sandbox-exec"
        assert argv[1] == "-p"
        assert argv[-3:] == ["bash", "-c", "echo hi"]
        return argv[2]

    def test_denies_writes_outside_run_root(self, tmp_path: Path) -> None:
        profile = self._profile(tmp_path)
        assert "(deny file-write*)" in profile
        run_root = local_sandbox._sbpl_quote((tmp_path / "run").resolve())
        assert f'(subpath "{run_root}")' in profile

    def test_network_denied_by_default_and_opt_in(self, tmp_path: Path) -> None:
        assert "(deny network*)" in self._profile(tmp_path)
        enabled = self._profile(tmp_path, allow_net=True)
        assert "(allow network*)" not in enabled
        assert "(deny network-inbound)" in enabled
        assert "(deny network-bind)" in enabled
        assert "remote unix-socket" in enabled
        assert "/private/var/run/mDNSResponder" in enabled

    @pytest.mark.skipif(os.name == "nt", reason="Seatbelt profiles are macOS-specific")
    def test_strict_reads_use_granular_usr_roots(self, tmp_path: Path) -> None:
        run_root = tmp_path / "run"
        profile = local_sandbox._seatbelt_argv(
            ["bash", "-c", "echo hi"],
            write_roots=[run_root],
            tmp=run_root / "tmp",
            home=run_root / "home",
            extra_ro=[],
            allow_net=False,
            strict_reads=True,
        )[2]

        assert '(subpath "/usr")' not in profile
        assert '(subpath "/usr/bin")' in profile

    def test_reads_deny_host_home_except_run_and_runtime_roots(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host_home = (tmp_path / "host-home").resolve()
        run_root = host_home / "run"
        runtime_dir = host_home / ".local" / "runtime"
        runtime_file = runtime_dir / "agent"
        (run_root / "tmp").mkdir(parents=True)
        (run_root / "home").mkdir()
        runtime_dir.mkdir(parents=True)
        runtime_file.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("HOME", str(host_home))

        profile = local_sandbox._seatbelt_argv(
            ["/bin/true"],
            write_roots=[run_root],
            tmp=run_root / "tmp",
            home=run_root / "home",
            extra_ro=[runtime_file],
            allow_net=True,
        )[2]

        assert "(allow default)" in profile
        assert "(deny file-read-data" in profile
        assert "(deny file-read-xattr" in profile
        assert "(deny file-read-metadata" in profile
        assert "require-all" in profile
        assert "require-not" in profile
        assert "require-any" in profile
        assert f'(subpath "{local_sandbox._sbpl_quote(host_home)}")' in profile
        assert f'(subpath "{local_sandbox._sbpl_quote(run_root)}")' in profile
        assert f'(literal "{local_sandbox._sbpl_quote(runtime_file)}")' in profile

    def test_host_home_ancestor_exceptions_are_metadata_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host_home = (tmp_path / "host-home").resolve()
        run_root = host_home / "projects" / "nested" / "run"
        (run_root / "tmp").mkdir(parents=True)
        (run_root / "home").mkdir()
        monkeypatch.setenv("HOME", str(host_home))

        profile = local_sandbox._seatbelt_argv(
            ["/bin/true"],
            write_roots=[run_root],
            tmp=run_root / "tmp",
            home=run_root / "home",
            extra_ro=[],
            allow_net=False,
        )[2]

        assert "(deny file-read-data" in profile
        assert "(deny file-read-xattr" in profile
        assert "(deny file-read-metadata" in profile
        metadata_policy = profile.split("(deny file-read-metadata", maxsplit=1)[1]
        for ancestor in (host_home, host_home / "projects", host_home / "projects" / "nested"):
            literal = f'(literal "{local_sandbox._sbpl_quote(ancestor)}")'
            assert literal in metadata_policy
            assert literal not in profile.split("(deny file-read-metadata", maxsplit=1)[0]

    @pytest.mark.parametrize("broad_root", ["home", "parent"])
    def test_rejects_exception_that_exposes_host_home(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        broad_root: str,
    ) -> None:
        host_home = (tmp_path / "host-home").resolve()
        host_home.mkdir()
        run_root = host_home if broad_root == "home" else host_home.parent
        monkeypatch.setenv("HOME", str(host_home))

        with pytest.raises(local_sandbox.SandboxUnavailable, match="host HOME"):
            local_sandbox._seatbelt_argv(
                ["/bin/true"],
                write_roots=[run_root],
                tmp=run_root,
                home=run_root,
                extra_ro=[],
                allow_net=False,
            )

    def test_sbpl_paths_are_escaped(self, tmp_path: Path) -> None:
        run = tmp_path / 'a"b\\c'
        (run / "tmp").mkdir(parents=True)
        (run / "home").mkdir()
        argv = local_sandbox._seatbelt_argv(
            ["bash", "-c", "echo hi"],
            write_roots=[run],
            tmp=run / "tmp",
            home=run / "home",
            extra_ro=[],
            allow_net=False,
        )
        profile = argv[2]
        # the raw quote/backslash must not appear unescaped inside the literal
        assert '\\"' in profile or "\\\\" in profile
        assert local_sandbox._sbpl_quote('/x/a"b') == '/x/a\\"b'
        assert local_sandbox._sbpl_quote("/x/a\\b") == "/x/a\\\\b"

    def test_missing_host_home_fails_closed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        run_root = tmp_path / "run"
        (run_root / "tmp").mkdir(parents=True)
        (run_root / "home").mkdir()
        monkeypatch.setattr(local_sandbox, "_safe_host_homes", lambda: None)

        with pytest.raises(local_sandbox.SandboxUnavailable, match="host HOME"):
            local_sandbox._seatbelt_argv(
                ["/bin/true"],
                write_roots=[run_root],
                tmp=run_root / "tmp",
                home=run_root / "home",
                extra_ro=[],
                allow_net=False,
            )

    @pytest.mark.skipif(pwd is None, reason="requires POSIX passwd database")
    def test_profile_denies_environment_and_passwd_home(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert pwd is not None
        passwd_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
        fake_home = tmp_path / "fake-home"
        run_root = fake_home / "run"
        (run_root / "tmp").mkdir(parents=True)
        (run_root / "home").mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        profile = local_sandbox._seatbelt_argv(
            ["/bin/true"],
            write_roots=[run_root],
            tmp=run_root / "tmp",
            home=run_root / "home",
            extra_ro=[],
            allow_net=False,
        )[2]

        assert f'(subpath "{local_sandbox._sbpl_quote(fake_home.resolve())}")' in profile
        assert f'(subpath "{local_sandbox._sbpl_quote(passwd_home)}")' in profile


class TestSandboxWrap:
    def test_none_backend_returns_argv_unchanged(self, tmp_path: Path) -> None:
        sandbox = local_sandbox.Sandbox(_plan("none", "advisory-only"))
        argv = ["bash", "-c", "true"]
        wrapped = sandbox.wrap(
            argv,
            workdir=tmp_path,
            write_roots=[tmp_path],
            home=tmp_path,
            tmp=tmp_path,
            allow_net=False,
            extra_ro=[],
        )
        assert wrapped == argv

    def test_bubblewrap_backend_wraps(self, tmp_path: Path) -> None:
        sandbox = local_sandbox.Sandbox(_plan("bubblewrap"))
        wrapped = sandbox.wrap(
            ["bash", "-c", "true"],
            workdir=tmp_path,
            write_roots=[tmp_path],
            home=tmp_path,
            tmp=tmp_path,
            allow_net=False,
            extra_ro=[],
        )
        assert wrapped[0] == "bwrap"

    def test_strict_reads_rejects_root_that_contains_host_home(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sandbox = local_sandbox.Sandbox(_plan("bubblewrap"))
        monkeypatch.setattr(local_sandbox, "_safe_host_homes", lambda: (tmp_path.resolve(),))

        with pytest.raises(ValueError, match="host home"):
            sandbox.wrap(
                ["bash", "-c", "true"],
                workdir=tmp_path / "run",
                write_roots=[tmp_path / "run"],
                home=tmp_path / "run" / "home",
                tmp=tmp_path / "run" / "tmp",
                allow_net=False,
                extra_ro=[tmp_path],
                strict_reads=True,
            )

    @pytest.mark.parametrize(
        "broad_root",
        [
            "/",
            "/usr",
            "/usr/local",
            "/opt",
            "/tmp",
            "/private/tmp",
            "/var/tmp",
            "/var",
            "/private",
            "/private/var",
            "/etc",
            "/home",
            "/Users",
            "/root",
        ],
    )
    def test_strict_reads_filter_shared_tree_roots_but_keep_narrow_descendants(
        self,
        monkeypatch: pytest.MonkeyPatch,
        broad_root: str,
    ) -> None:
        monkeypatch.setattr(local_sandbox, "_safe_host_homes", lambda: ())
        broad = Path(broad_root)
        narrow = broad / "selected-runtime" / "lib" / "node_modules" / "opencode-ai"

        roots = local_sandbox._validated_strict_read_roots([broad, narrow])

        assert broad not in roots
        assert narrow in roots

    def test_strict_reads_filter_alias_to_shared_tree_but_allow_alias_to_narrow_runtime(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        broad_alias = tmp_path / "broad-alias"
        broad_alias.symlink_to(Path("/usr/local"), target_is_directory=True)
        narrow_target = tmp_path / "selected" / "venv"
        narrow_target.mkdir(parents=True)
        narrow_alias = tmp_path / "narrow-alias"
        narrow_alias.symlink_to(narrow_target, target_is_directory=True)
        monkeypatch.setattr(local_sandbox, "_safe_host_homes", lambda: ())

        roots = local_sandbox._validated_strict_read_roots([broad_alias, narrow_alias])

        assert broad_alias not in roots
        assert narrow_alias in roots

    def test_strict_reads_allow_home_descendant_but_reject_home_ancestor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        host_home = tmp_path / "host-home"
        narrow_venv = host_home / ".local" / "venvs" / "skill-evaluator"
        monkeypatch.setattr(local_sandbox, "_safe_host_homes", lambda: (host_home.resolve(),))

        assert local_sandbox._validated_strict_read_roots([narrow_venv]) == [narrow_venv.absolute()]
        with pytest.raises(ValueError, match="host home"):
            local_sandbox._validated_strict_read_roots([host_home.parent])

    @pytest.mark.parametrize("backend", ["bubblewrap", "seatbelt"])
    def test_strict_filter_is_shared_by_kernel_wrappers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        backend: str,
    ) -> None:
        shared_tmp = Path(tempfile.mkdtemp(prefix="skillevaluator-strict-shared-", dir="/tmp"))
        runtime = shared_tmp / "venv"
        runtime.mkdir()
        run_root = tmp_path / "run"
        (run_root / "workspace").mkdir(parents=True)
        (run_root / "home").mkdir()
        (run_root / "tmp").mkdir()
        monkeypatch.setattr(local_sandbox, "_safe_host_homes", lambda: ())
        sandbox = local_sandbox.Sandbox(_plan(backend, "kernel-macos" if backend == "seatbelt" else "kernel"))

        try:
            wrapped = sandbox.wrap(
                ["/bin/true"],
                workdir=run_root / "workspace",
                write_roots=[run_root],
                home=run_root / "home",
                tmp=run_root / "tmp",
                allow_net=False,
                extra_ro=[Path("/tmp"), runtime],
                strict_reads=True,
            )
        finally:
            shutil.rmtree(shared_tmp, ignore_errors=True)

        canonical_tmp = Path("/tmp").resolve()
        canonical_runtime = runtime.resolve()
        if backend == "bubblewrap":
            ro_binds = TestBwrapArgv._option_operands(wrapped, "--ro-bind")
            assert (str(canonical_tmp), "/tmp") not in ro_binds
            assert (str(canonical_runtime), str(canonical_runtime)) in ro_binds
        else:
            profile = wrapped[2]
            assert f'(require-not (subpath "{local_sandbox._sbpl_quote(canonical_tmp)}"))' not in profile
            assert f'(require-not (subpath "{local_sandbox._sbpl_quote(canonical_runtime)}"))' in profile


class TestRlimits:
    def test_local_sandbox_import_survives_missing_resource_module(self) -> None:
        code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'resource':
        raise ModuleNotFoundError("No module named 'resource'")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import skillevaluator.tier3.harbor.local_sandbox
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr

    def test_apply_rlimits_is_noop_when_resource_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(local_sandbox, "resource", None)
        local_sandbox.apply_rlimits()

    def test_clamped_never_exceeds_hard_limit(self) -> None:
        if resource is None:
            pytest.skip("resource is unavailable on this platform")
        assert local_sandbox._clamped(4096, (1024, 2048)) == (2048, 2048)
        assert local_sandbox._clamped(4096, (1024, resource.RLIM_INFINITY)) == (4096, 4096)
        assert local_sandbox._clamped(900, (500, 700)) == (700, 700)

    def test_rlimits_applied_in_child_process(self) -> None:
        if resource is None:
            pytest.skip("resource is unavailable on this platform")
        code = "import resource; print(resource.getrlimit(resource.RLIMIT_NOFILE)[0])"
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            preexec_fn=local_sandbox.apply_rlimits,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert int(result.stdout.strip()) <= local_sandbox.RLIMIT_NOFILE


_ON_MACOS_WITH_SEATBELT = platform.system() == "Darwin" and Path("/usr/bin/sandbox-exec").exists()


# Read prefixes the live interpreter needs under deny-read-by-default (uv-managed
# python lives under ~/.local, outside the system allowlist), mirroring what
# _runtime_ro_binds() supplies in the real exec() path.
_PY_RO = [Path(sys.prefix), Path(sys.base_prefix), Path(sys.executable).resolve().parent.parent]
_CANONICAL_PYTHON = str(Path(sys.executable).resolve())


@pytest.mark.skipif(not _ON_MACOS_WITH_SEATBELT, reason="requires macOS Seatbelt")
class TestSeatbeltLive:
    def _run(
        self,
        command: list[str],
        run_root: Path,
        *,
        allow_net: bool = False,
        extra_ro: list[Path] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        argv = local_sandbox._seatbelt_argv(
            command,
            write_roots=[run_root],
            tmp=run_root / "tmp",
            home=run_root / "home",
            extra_ro=extra_ro if extra_ro is not None else _PY_RO,
            allow_net=allow_net,
        )
        child_env = dict(os.environ)
        child_env.update({"HOME": str(run_root / "home"), "TMPDIR": str(run_root / "tmp")})
        return subprocess.run(
            argv,
            cwd=run_root,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def _run_strict(
        self,
        command: list[str],
        run_root: Path,
        *,
        allow_net: bool = False,
        extra_ro: list[Path] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        argv = local_sandbox._seatbelt_argv(
            command,
            write_roots=[run_root],
            tmp=run_root / "tmp",
            home=run_root / "home",
            extra_ro=extra_ro if extra_ro is not None else _PY_RO,
            allow_net=allow_net,
            strict_reads=True,
        )
        child_env = dict(os.environ)
        child_env.update({"HOME": str(run_root / "home"), "TMPDIR": str(run_root / "tmp")})
        return subprocess.run(
            argv,
            cwd=run_root,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    @pytest.fixture()
    def run_root(self, tmp_path: Path) -> Path:
        root = (tmp_path / "run").resolve()
        (root / "tmp").mkdir(parents=True)
        (root / "home").mkdir()
        return root

    @pytest.fixture()
    def fake_home_tree(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[Path, Path, Path]:
        host_home = (tmp_path / "disposable-host-home").resolve()
        run_root = host_home / "run"
        runtime_root = host_home / "selected-runtime"
        auth_file = host_home / "sibling-auth.json"
        (run_root / "tmp").mkdir(parents=True)
        (run_root / "home").mkdir()
        runtime_root.mkdir()
        (run_root / "allowed.txt").write_text("RUN-ALLOWED", encoding="utf-8")
        (runtime_root / "allowed.txt").write_text("RUNTIME-ALLOWED", encoding="utf-8")
        auth_file.write_text("FAKE-AUTH-DENIED", encoding="utf-8")
        monkeypatch.setenv("HOME", str(host_home))
        return run_root, runtime_root, auth_file

    def test_disposable_home_allows_only_explicit_read_roots(
        self,
        fake_home_tree: tuple[Path, Path, Path],
    ) -> None:
        run_root, runtime_root, auth_file = fake_home_tree

        allowed = self._run(
            ["/bin/cat", str(run_root / "allowed.txt"), str(runtime_root / "allowed.txt")],
            run_root,
            extra_ro=[runtime_root],
        )
        denied = self._run(["/bin/cat", str(auth_file)], run_root, extra_ro=[runtime_root])

        assert allowed.returncode == 0, allowed.stderr
        assert allowed.stdout == "RUN-ALLOWEDRUNTIME-ALLOWED"
        assert denied.returncode != 0
        assert "FAKE-AUTH-DENIED" not in denied.stdout

    def test_disposable_home_denies_symlink_escape(
        self,
        fake_home_tree: tuple[Path, Path, Path],
    ) -> None:
        run_root, runtime_root, auth_file = fake_home_tree
        escape = run_root / "auth-link"
        escape.symlink_to(auth_file)

        result = self._run(["/bin/cat", str(escape)], run_root, extra_ro=[runtime_root])

        assert result.returncode != 0
        assert "FAKE-AUTH-DENIED" not in result.stdout

    def test_disposable_home_evaluator_interpreter_runs(
        self,
        fake_home_tree: tuple[Path, Path, Path],
    ) -> None:
        run_root, runtime_root, _auth_file = fake_home_tree

        result = self._run(
            [_CANONICAL_PYTHON, "-B", "-c", "print('interpreter-ok')"],
            run_root,
            extra_ro=[runtime_root, *_PY_RO],
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "interpreter-ok"

    def test_disposable_home_allows_git_to_traverse_to_run_root(
        self,
        fake_home_tree: tuple[Path, Path, Path],
    ) -> None:
        run_root, runtime_root, _auth_file = fake_home_tree

        result = self._run(
            ["/usr/bin/git", "-C", str(run_root), "init", "-q"],
            run_root,
            extra_ro=[runtime_root, *_PY_RO],
        )

        assert result.returncode == 0, result.stderr
        assert (run_root / ".git").is_dir()

    def test_disposable_home_traversal_does_not_expose_sibling_metadata_or_entries(
        self,
        fake_home_tree: tuple[Path, Path, Path],
    ) -> None:
        run_root, runtime_root, auth_file = fake_home_tree

        listing = self._run(
            ["/bin/ls", str(auth_file.parent)],
            run_root,
            extra_ro=[runtime_root, *_PY_RO],
        )
        metadata = self._run(
            ["/usr/bin/stat", str(auth_file)],
            run_root,
            extra_ro=[runtime_root, *_PY_RO],
        )

        assert listing.returncode != 0
        assert auth_file.name not in listing.stdout
        assert metadata.returncode != 0
        assert auth_file.name not in metadata.stdout

    def test_relative_path_helper_cannot_override_absolute_interpreter(
        self,
        run_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        legitimate_bin = run_root / "legitimate-bin"
        legitimate_bin.mkdir()
        attacker_bin = run_root / "evilbin"
        attacker_bin.mkdir()
        attacker = attacker_bin / "python3"
        attacker.write_text("#!/bin/sh\nprintf 'PWNED\\n'\n", encoding="utf-8")
        attacker.chmod(0o755)
        probe = run_root / "probe.py"
        probe.write_text("#!/usr/bin/env python3\nprint('SAFE')\n", encoding="utf-8")
        probe.chmod(0o755)
        monkeypatch.chdir(run_root)
        path = local_runtime.runtime_path(
            run_root.parent / "managed",
            os.pathsep.join((str(legitimate_bin), "evilbin", "/usr/bin")),
            agents=["opencode"],
        )
        monkeypatch.setenv("PATH", path)

        result = self._run([str(probe)], run_root, extra_ro=[])

        path_parts = path.split(os.pathsep)
        assert str(legitimate_bin.resolve()) in path_parts
        assert str(attacker_bin.resolve()) not in path_parts
        assert "evilbin" not in path_parts
        assert result.returncode == 0, result.stderr
        assert result.stdout == "SAFE\n"
        assert "PWNED" not in result.stdout

    def test_direct_runtime_file_does_not_expose_siblings(
        self,
        fake_home_tree: tuple[Path, Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_root, _runtime_root, _auth_file = fake_home_tree
        command = run_root.parent / "tools" / "opencode"
        command.parent.mkdir()
        command.write_text("#!/bin/sh\nprintf direct-ok\n", encoding="utf-8")
        command.chmod(0o755)
        sibling = command.parent / "sibling.txt"
        sibling.write_text("SIBLING-DENIED", encoding="utf-8")
        monkeypatch.setenv("PATH", os.pathsep.join((str(command.parent), os.defpath)))
        read_paths = local_runtime.runtime_command_roots(["opencode"])

        version = self._run([str(command), "--version"], run_root, extra_ro=read_paths)
        denied = self._run(["/bin/cat", str(sibling)], run_root, extra_ro=read_paths)

        assert version.returncode == 0, version.stderr
        assert version.stdout == "direct-ok"
        assert denied.returncode != 0
        assert "SIBLING-DENIED" not in denied.stdout

    def test_direct_node_modules_bin_shim_does_not_expose_siblings(
        self,
        fake_home_tree: tuple[Path, Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_root, _runtime_root, _auth_file = fake_home_tree
        bin_dir = run_root.parent / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        command = bin_dir / "opencode"
        command.write_text("#!/bin/sh\nprintf bin-ok\n", encoding="utf-8")
        command.chmod(0o755)
        sibling = bin_dir / "sibling.txt"
        sibling.write_text("BIN-SIBLING-DENIED", encoding="utf-8")
        monkeypatch.setenv("PATH", os.pathsep.join((str(bin_dir), os.defpath)))
        read_paths = local_runtime.runtime_command_roots(
            ["opencode"],
            runtime_root=run_root.parent / "managed",
        )

        version = self._run([str(command), "--version"], run_root, extra_ro=read_paths)
        denied = self._run(["/bin/cat", str(sibling)], run_root, extra_ro=read_paths)

        assert version.returncode == 0, version.stderr
        assert version.stdout == "bin-ok"
        assert denied.returncode != 0
        assert "BIN-SIBLING-DENIED" not in denied.stdout
        assert command.absolute() in read_paths
        assert bin_dir.resolve() not in read_paths

    def test_single_file_symlink_and_env_helper_work_without_exposing_siblings(
        self,
        fake_home_tree: tuple[Path, Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_root, _runtime_root, _auth_file = fake_home_tree
        helper = run_root.parent / "helper-bin" / "helper"
        helper.parent.mkdir()
        helper.write_text("#!/bin/sh\nprintf helper-ok\n", encoding="utf-8")
        helper.chmod(0o755)
        helper_sibling = helper.parent / "sibling.txt"
        helper_sibling.write_text("HELPER-SIBLING-DENIED", encoding="utf-8")
        target = run_root.parent / "tools" / "opencode-real"
        target.parent.mkdir()
        target.write_text("#!/usr/bin/env helper\n", encoding="utf-8")
        target.chmod(0o755)
        command = run_root.parent / ".local" / "bin" / "opencode"
        command.parent.mkdir(parents=True)
        command.symlink_to(target)
        link_sibling = command.parent / "sibling.txt"
        link_sibling.write_text("LINK-SIBLING-DENIED", encoding="utf-8")
        monkeypatch.setenv(
            "PATH",
            os.pathsep.join((str(command.parent), str(helper.parent), os.defpath)),
        )
        read_paths = local_runtime.runtime_command_roots(["opencode"])

        version = self._run([str(command), "--version"], run_root, extra_ro=read_paths)
        denied_link = self._run(["/bin/cat", str(link_sibling)], run_root, extra_ro=read_paths)
        denied_helper = self._run(["/bin/cat", str(helper_sibling)], run_root, extra_ro=read_paths)

        assert version.returncode == 0, version.stderr
        assert version.stdout == "helper-ok"
        assert denied_link.returncode != 0
        assert denied_helper.returncode != 0
        assert "SIBLING-DENIED" not in denied_link.stdout + denied_helper.stdout

    def test_overridden_home_still_denies_passwd_home_sibling(
        self,
        fake_home_tree: tuple[Path, Path, Path],
    ) -> None:
        run_root, runtime_root, _auth_file = fake_home_tree
        assert pwd is not None
        passwd_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
        probe_dir = Path(tempfile.mkdtemp(prefix=".skillevaluator-passwd-home-", dir=passwd_home))
        probe = probe_dir / "fake-auth.txt"
        probe.write_text("PASSWD-HOME-DENIED", encoding="utf-8")
        try:
            result = self._run(["/bin/cat", str(probe)], run_root, extra_ro=[runtime_root])
            assert result.returncode != 0
            assert "PASSWD-HOME-DENIED" not in result.stdout
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)

    def test_product_preflight_accepts_seatbelt_backend(self) -> None:
        sandbox = local_sandbox.detect("require")
        assert sandbox.plan.backend == "seatbelt"
        assert sandbox.plan.strength == "kernel-macos"

    def test_cross_process_metadata_is_denied(self, run_root: Path) -> None:
        # A long-lived canary whose argv carries a recognizable marker (sleep
        # itself takes only a number, so wrap it so the marker shows in `ps`).
        marker = "types_canary_marker_600"
        canary = subprocess.Popen([sys.executable, "-c", "import sys,time; time.sleep(600)", marker])
        try:
            assert canary.poll() is None, "canary must stay alive for a valid test"
            result = self._run(["/bin/ps", "-Ao", "command"], run_root)
            # Honest behavior (not "ps works but filters"): (deny process-info*)
            # blocks process enumeration outright — `ps` fails with a permission
            # error rather than returning a filtered list. Either way the
            # unrelated canary's argv is never exposed to the sandboxed process.
            assert marker not in result.stdout, "sandboxed process could read an unrelated process's argv"
            enumeration_denied = result.returncode != 0 or not result.stdout.strip()
            assert enumeration_denied, "process enumeration should be denied under (deny process-info*)"
        finally:
            canary.terminate()
            canary.wait(timeout=5)

    def test_signal_to_unrelated_process_denied(self, run_root: Path) -> None:
        # A sandboxed skill must not be able to signal an unrelated same-user
        # process. Run in its own session (like production exec) so the canary
        # is a different process group; (deny signal)+(allow signal pgrp) blocks it.
        canary = subprocess.Popen(["/bin/sleep", "600"])
        try:
            assert canary.poll() is None, "canary must stay alive for a valid test"
            argv = local_sandbox._seatbelt_argv(
                ["/bin/kill", "-TERM", str(canary.pid)],
                write_roots=[run_root],
                tmp=run_root / "tmp",
                home=run_root / "home",
                extra_ro=_PY_RO,
                allow_net=False,
            )
            subprocess.run(argv, cwd=run_root, capture_output=True, text=True, timeout=30, start_new_session=True)
            with contextlib.suppress(subprocess.TimeoutExpired):
                canary.wait(timeout=1)
            assert canary.poll() is None, "sandboxed kill terminated an unrelated same-user process"
        finally:
            canary.terminate()
            canary.wait(timeout=5)

    def test_signal_to_own_child_allowed(self, run_root: Path) -> None:
        # The skill can still manage its own process tree (same process group).
        result = self._run(["/bin/bash", "-c", "sleep 30 & c=$!; sleep 0.3; kill $c; echo rc=$?"], run_root)
        assert "rc=0" in result.stdout

    def test_benign_command_succeeds(self, run_root: Path) -> None:
        result = self._run(["/bin/bash", "-c", "echo hi"], run_root)
        assert result.returncode == 0
        assert result.stdout.strip() == "hi"

    def test_write_inside_run_root_allowed(self, run_root: Path) -> None:
        result = self._run(["/bin/bash", "-c", f"echo data > {run_root}/out.txt"], run_root)
        assert result.returncode == 0
        assert (run_root / "out.txt").read_text() == "data\n"

    def test_python_write_outside_run_root_blocked(self, run_root: Path, tmp_path: Path) -> None:
        escape = tmp_path / "escape.txt"
        started = run_root / "python-started.txt"
        code = f"open({str(started)!r}, 'w').write('started'); open({str(escape)!r}, 'w').write('pwned')"
        result = self._run([_CANONICAL_PYTHON, "-B", "-c", code], run_root)
        assert result.returncode != 0
        assert started.read_text(encoding="utf-8") == "started"
        assert not escape.exists()

    def test_read_of_host_home_secret_blocked(self, run_root: Path) -> None:
        # A credential path remains unreadable even though normal HOME paths
        # must be available for run roots and user-installed runtimes.
        credential_dir = Path(tempfile.mkdtemp(prefix=".skilleval-credential-home-"))
        probe = credential_dir / ".ssh" / "probe.txt"
        probe.parent.mkdir()
        probe.write_text("TOP-SECRET", encoding="utf-8")
        old_home = local_sandbox.os.environ.get("HOME")
        local_sandbox.os.environ["HOME"] = str(credential_dir)
        try:
            result = self._run(["/bin/cat", str(probe)], run_root)
            assert result.returncode != 0
            assert "TOP-SECRET" not in result.stdout
        finally:
            if old_home is None:
                local_sandbox.os.environ.pop("HOME", None)
            else:
                local_sandbox.os.environ["HOME"] = old_home
            shutil.rmtree(credential_dir, ignore_errors=True)

    def test_read_of_arbitrary_host_home_file_blocked(self, run_root: Path) -> None:
        probe_dir = Path(tempfile.mkdtemp(prefix=".skilleval-host-read-probe-", dir=Path.home())).resolve()
        probe = probe_dir / "project" / ".env"
        probe.parent.mkdir()
        probe.write_text("ARBITRARY-HOST-SECRET", encoding="utf-8")
        try:
            result = self._run(["/bin/cat", str(probe)], run_root, allow_net=True)
            assert result.returncode != 0
            assert "ARBITRARY-HOST-SECRET" not in result.stdout
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)

    def test_strict_reads_block_host_tmp(self, run_root: Path) -> None:
        probe_dir = Path(tempfile.mkdtemp(prefix=".skilleval-strict-read-", dir="/tmp"))
        probe = probe_dir / "secret.txt"
        probe.write_text("STRICT-TMP-SECRET", encoding="utf-8")
        try:
            result = self._run_strict(["/bin/cat", str(probe)], run_root)
            assert result.returncode != 0
            assert "STRICT-TMP-SECRET" not in result.stdout
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)

    def test_network_enabled_still_blocks_host_unix_socket(self, run_root: Path) -> None:
        socket_dir = Path(tempfile.mkdtemp(prefix=".se-ipc-", dir="/tmp"))
        socket_path = socket_dir / "s"
        server = socket.socket(socket.AF_UNIX)
        server.bind(str(socket_path))
        server.listen(1)
        accepted: list[socket.socket] = []

        def accept_once() -> None:
            with contextlib.suppress(OSError):
                connection, _ = server.accept()
                accepted.append(connection)

        thread = threading.Thread(target=accept_once, daemon=True)
        thread.start()
        code = "import socket,sys;s=socket.socket(socket.AF_UNIX);s.connect(sys.argv[1])"
        try:
            result = self._run(["/usr/bin/python3", "-B", "-c", code, str(socket_path)], run_root, allow_net=True)
            assert result.returncode != 0
            assert not accepted
        finally:
            server.close()
            for connection in accepted:
                connection.close()
            shutil.rmtree(socket_dir, ignore_errors=True)

    def test_network_enabled_keeps_system_dns_available(self, run_root: Path) -> None:
        code = "import socket; print(socket.getaddrinfo('integrate.api.nvidia.com', 443)[0][4][0])"
        result = self._run(["/usr/bin/python3", "-B", "-c", code], run_root, allow_net=True)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip()

    def test_strict_reads_allow_selected_symlinked_runtime(self, run_root: Path, tmp_path: Path) -> None:
        target = tmp_path / "runtime" / "agent"
        target.parent.mkdir()
        target.write_text("#!/bin/sh\nprintf strict-runtime-ok\n", encoding="utf-8")
        target.chmod(0o755)
        command = tmp_path / "bin" / "agent"
        command.parent.mkdir()
        command.symlink_to(target)

        result = self._run_strict([str(command)], run_root, extra_ro=[command, target])

        assert result.returncode == 0, result.stderr
        assert result.stdout == "strict-runtime-ok"

    def test_run_root_inside_home_with_spaces_is_usable(self) -> None:
        run_root = Path(tempfile.mkdtemp(prefix=".skillevaluator seatbelt ", dir=Path.home())).resolve()
        try:
            (run_root / "tmp").mkdir()
            (run_root / "home").mkdir()
            output = run_root / "output with spaces.txt"
            code = f"from pathlib import Path; Path({str(output)!r}).write_text('ok', encoding='utf-8')"
            result = self._run([_CANONICAL_PYTHON, "-B", "-c", code], run_root)
            assert result.returncode == 0, result.stderr
            assert output.read_text(encoding="utf-8") == "ok"
        finally:
            shutil.rmtree(run_root, ignore_errors=True)

    @pytest.mark.parametrize("strict_reads", [False, True])
    def test_git_init_nested_below_host_home_uses_metadata_only_ancestor_traversal(
        self,
        strict_reads: bool,
    ) -> None:
        ancestor = Path(tempfile.mkdtemp(prefix=".skillevaluator-git-traversal-", dir=Path.home())).resolve()
        run_root = ancestor / "projects" / "nested" / "run"
        workspace = run_root / "workspace"
        sibling_secret = ancestor / "sibling-secret.txt"
        (run_root / "tmp").mkdir(parents=True)
        (run_root / "home").mkdir()
        workspace.mkdir()
        sibling_secret.write_text("DO-NOT-READ", encoding="utf-8")
        run = self._run_strict if strict_reads else self._run
        try:
            git = run(["/usr/bin/git", "-C", str(workspace), "init", "-q"], run_root, extra_ro=[])
            assert git.returncode == 0, git.stderr
            assert (workspace / ".git" / "HEAD").is_file()

            stat = run(["/usr/bin/stat", "-f", "%N", str(ancestor)], run_root, extra_ro=[])
            assert stat.returncode == 0, stat.stderr

            listing = run(["/bin/ls", str(ancestor)], run_root, extra_ro=[])
            assert listing.returncode != 0

            secret = run(["/bin/cat", str(sibling_secret)], run_root, extra_ro=[])
            assert secret.returncode != 0
            assert "DO-NOT-READ" not in secret.stdout
        finally:
            shutil.rmtree(ancestor, ignore_errors=True)

    def test_strict_runtime_outside_home_does_not_expose_ancestor_siblings(self, run_root: Path) -> None:
        ancestor = Path(tempfile.mkdtemp(prefix="skillevaluator-strict-runtime-", dir="/tmp")).resolve()
        runtime = ancestor / "tools" / "nested" / "opencode"
        sibling_secret = ancestor / "sibling-secret-name.txt"
        runtime.parent.mkdir(parents=True)
        runtime.write_text("#!/bin/sh\nprintf approved-runtime-ok\n", encoding="utf-8")
        runtime.chmod(0o755)
        sibling_secret.write_text("DO-NOT-READ", encoding="utf-8")
        try:
            launched = self._run_strict([str(runtime)], run_root, extra_ro=[runtime])
            assert launched.returncode == 0, launched.stderr
            assert launched.stdout == "approved-runtime-ok"

            listing = self._run_strict(["/bin/ls", "-1", str(ancestor)], run_root, extra_ro=[runtime])
            assert sibling_secret.name not in listing.stdout

            secret = self._run_strict(["/bin/cat", str(sibling_secret)], run_root, extra_ro=[runtime])
            assert secret.returncode != 0
            assert "DO-NOT-READ" not in secret.stdout
        finally:
            shutil.rmtree(ancestor, ignore_errors=True)

    def test_network_blocked_by_default(self, run_root: Path) -> None:
        code = (
            "import socket,sys; print('network-probe-started', file=sys.stderr); "
            "s=socket.socket(); s.settimeout(3); s.connect(('1.1.1.1', 80))"
        )
        result = self._run([_CANONICAL_PYTHON, "-B", "-c", code], run_root)
        assert result.returncode != 0
        assert "network-probe-started" in result.stderr
        assert "not permitted" in result.stderr.lower() or "denied" in result.stderr.lower()

    def test_detect_smoke_passes_on_this_host(self) -> None:
        sandbox = local_sandbox.detect("require")
        assert sandbox.plan.backend == "seatbelt"


_ON_LINUX_WITH_BWRAP = platform.system() == "Linux" and shutil.which("bwrap") is not None


@pytest.mark.skipif(not _ON_LINUX_WITH_BWRAP, reason="requires Linux bubblewrap")
class TestBubblewrapLive:
    def _run(self, command: list[str], run_root: Path, *, allow_net: bool = False) -> subprocess.CompletedProcess[str]:
        argv = local_sandbox._bwrap_argv(
            command,
            workdir=run_root,
            write_roots=[run_root],
            home=run_root / "home",
            tmp=run_root / "tmp",
            allow_net=allow_net,
            extra_ro=[Path(sys.executable).resolve().parent.parent, Path(sys.base_prefix)],
        )
        return subprocess.run(argv, capture_output=True, text=True, timeout=60)

    @pytest.fixture()
    def run_root(self, tmp_path: Path) -> Path:
        root = (tmp_path / "run").resolve()
        (root / "tmp").mkdir(parents=True)
        (root / "home").mkdir()
        return root

    def test_benign_command_succeeds(self, run_root: Path) -> None:
        if not local_sandbox._bwrap_smoke_test(shutil.which("bwrap") or "bwrap"):
            pytest.skip("bwrap installed but user namespaces unavailable")
        result = self._run(["/bin/sh", "-c", "echo hi"], run_root)
        assert result.returncode == 0
        assert result.stdout.strip() == "hi"

    def test_runtime_visible_read_only_without_unrelated_host_home_contents(self, run_root: Path) -> None:
        if not local_sandbox._bwrap_smoke_test(shutil.which("bwrap") or "bwrap"):
            pytest.skip("bwrap installed but user namespaces unavailable")

        probe_dir = Path(tempfile.mkdtemp(prefix=".skilleval-host-read-probe-", dir=Path.home())).resolve()
        sentinel = probe_dir / "sentinel.txt"
        secret = "BWRAP-HOST-HOME-SECRET"
        sentinel.write_text(secret, encoding="utf-8")
        python_executable = Path(sys.executable).resolve()
        runtime_root = python_executable.parent.parent
        runtime_write_probe = runtime_root / f".skillevaluator-write-probe-{probe_dir.name}"
        code = f"""
from pathlib import Path

runtime_root = Path({str(runtime_root)!r})
runtime_write_probe = Path({str(runtime_write_probe)!r})
sentinel = Path({str(sentinel)!r})

if not runtime_root.is_dir():
    raise SystemExit("runtime root missing")
try:
    runtime_write_probe.write_text("unsafe", encoding="utf-8")
except OSError:
    pass
else:
    raise SystemExit("runtime root writable")

try:
    sentinel.read_text(encoding="utf-8")
except (FileNotFoundError, PermissionError):
    pass
else:
    raise SystemExit("host home sentinel readable")
"""
        try:
            result = self._run([str(python_executable), "-B", "-c", code], run_root)
            assert result.returncode == 0, result.stderr
            assert secret not in result.stdout
            assert secret not in result.stderr
        finally:
            runtime_write_probe.unlink(missing_ok=True)
            shutil.rmtree(probe_dir, ignore_errors=True)
