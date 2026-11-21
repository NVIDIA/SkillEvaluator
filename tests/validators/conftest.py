# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for validator tests."""

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture(params=["windows-junction", "simulated-junction"])
def make_junction(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Callable[[Path, Path], None]:
    """Return ``make(link, target)`` that turns ``link`` into a directory junction to ``target``.

    ``windows-junction`` runs ``mklink /J`` and only runs on Windows.
    ``simulated-junction`` runs everywhere else: it creates a directory symlink,
    then hides the symlink and reports a junction instead, the way Windows does.
    ``os.walk`` therefore sees a plain directory it would descend into, while
    ``os.path.isjunction`` is true and the entry still resolves to ``target``.
    """
    if request.param == "windows-junction":
        if os.name != "nt":
            pytest.skip("requires native Windows junction APIs")

        def make_native(link: Path, target: Path) -> None:
            created = subprocess.run(
                ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                text=True,
                check=False,
            )
            if created.returncode != 0:
                pytest.skip(f"Windows junction creation unavailable: {created.stderr or created.stdout}")

        return make_native

    if os.name == "nt":
        pytest.skip("Windows runs the native junction case instead")

    def make_simulated(link: Path, target: Path) -> None:
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
        real_islink = os.path.islink
        real_isjunction = os.path.isjunction
        real_is_symlink = Path.is_symlink

        def is_link(path: str | os.PathLike[str]) -> bool:
            return False if Path(path) == link else real_islink(path)

        def is_junction(path: str | os.PathLike[str]) -> bool:
            return True if Path(path) == link else real_isjunction(path)

        def path_is_symlink(self: Path) -> bool:
            return False if self == link else real_is_symlink(self)

        monkeypatch.setattr(os.path, "islink", is_link)
        monkeypatch.setattr(os.path, "isjunction", is_junction)
        monkeypatch.setattr(Path, "is_symlink", path_is_symlink)

    return make_simulated
