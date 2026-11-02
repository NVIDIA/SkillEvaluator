# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native filesystem flavours must not change portable catalog identities."""

from __future__ import annotations

import dataclasses
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from skillevaluator.deduplication.plugin import profile
from skillevaluator.deduplication.plugin.catalog_checks import _SelfSkillIdentity
from skillevaluator.plugin_skills import discover_openclaw_skill_manifests


class StrictWindowsPath(PureWindowsPath):
    """Exercise Python3.13's no-cross-flavour comparison even on Python3.12."""

    def is_relative_to(self, *others):
        if any(not isinstance(other, PureWindowsPath) for other in others):
            return False
        return super().is_relative_to(*others)


@pytest.mark.parametrize("flavour", [PureWindowsPath, StrictWindowsPath])
@pytest.mark.parametrize(
    ("relative", "roots", "expected"),
    [
        ("skills/alpha", ["skills"], ("alpha",)),
        ("portable/team/alpha", ["portable"], ("team/alpha",)),
        ("portable/alpha", ["portable/alpha"], (".",)),
        (".", ["."], (".",)),
        ("skills/alpha", [".", "skills", "other"], ("alpha", "skills/alpha")),
    ],
)
def test_native_windows_folders_keep_declared_root_catalog_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flavour,
    relative: str,
    roots: list[str],
    expected: tuple[str, ...],
) -> None:
    folder = tmp_path / relative
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Portable catalog guidance.\n---\n# Alpha\nRead guidance.\n",
        encoding="utf-8",
    )
    [native] = discover_openclaw_skill_manifests(tmp_path, {"skills": [relative]})
    windows = dataclasses.replace(native, relative_path=flavour(native.relative_path.as_posix()))
    extract = profile.extract_skill_manifest

    def read_host_file(root, manifest, **kwargs):
        # Retain a real anchored content read on this host; only the discovered
        # relative metadata uses the Windows flavour under test.
        assert manifest is windows
        return extract(root, native, **kwargs)

    monkeypatch.setattr(profile, "extract_skill_manifest", read_host_file)
    [skill] = profile._bundled_skills(
        tmp_path,
        max_skills=32,
        budget=None,
        skill_manifests=[windows],
        root_relative_prefix=".",
        catalog_roots=tuple(PurePosixPath(root) for root in roots),
    )
    assert skill.entry is not None
    assert skill.catalog_relative_paths == expected
    identity = _SelfSkillIdentity.build(skill, catalog_has_plugins=False, self_prefixes=())
    assert set(expected).issubset(identity.paths)
