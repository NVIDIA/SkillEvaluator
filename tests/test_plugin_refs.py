# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical ids of plugin dependency refs, however the ref is written."""

from __future__ import annotations

from pathlib import Path

import pytest

from skillevaluator.plugin_dependencies import RepositoryIdentity, classify_plugin_dependencies
from skillevaluator.plugin_refs import find_duplicate_refs, normalize_ref

REPO = "org/repo"


@pytest.mark.parametrize(
    ("refs", "canonical"),
    [
        (
            [f"github::{REPO}::skills::shared", f"github::{REPO}::skills::shared/"],
            f"github::{REPO}::skills::shared",
        ),
        (
            [f"github::{REPO}::team-skills::t/shared", f"github::{REPO}::team-skills::t//shared"],
            f"github::{REPO}::team-skills::t/shared",
        ),
        (
            [f"github::{REPO}::skills::shared/", {"source": "github", "repo": REPO, "path": "skills/shared/"}],
            f"github::{REPO}::skills::shared",
        ),
        (
            [
                "github::Org/Repo.GIT::skills::shared",
                {"source": "github", "repo": "Org/Repo.GIT", "path": "skills/shared"},
            ],
            f"github::{REPO}::skills::shared",
        ),
    ],
)
def test_string_and_selector_refs_share_one_canonical_id(refs: list, canonical: str) -> None:
    assert {normalize_ref(ref) for ref in refs} == {canonical}
    [group] = find_duplicate_refs(refs)
    assert group.canonical_id == canonical
    assert group.occurrences == refs


def test_a_name_of_only_slashes_is_left_as_written() -> None:
    assert normalize_ref(f"github::{REPO}::skills::/") == f"github::{REPO}::skills::/"


def test_refs_to_one_folder_share_one_classified_row(tmp_path: Path) -> None:
    (tmp_path / "skills" / "shared").mkdir(parents=True)
    (tmp_path / "skills" / "shared" / "SKILL.md").write_text("# Shared\n", encoding="utf-8")
    identity = RepositoryIdentity(clone_root=tmp_path, local_slug=REPO)
    refs = [
        f"github::{REPO}::skills::shared",
        f"github::{REPO}::skills::shared/",
        f"github::{REPO}::skills::absent",
        f"github::{REPO}::skills::absent/",
    ]

    resolution = classify_plugin_dependencies({"skills": {"refs": refs}}, tmp_path / "plugins" / "p", identity)

    assert [(row.ref, row.state, row.declared) for row in resolution.skills] == [
        (f"github::{REPO}::skills::shared", "referenced", 2),
        (f"github::{REPO}::skills::absent", "missing", 2),
    ]
    assert resolution.status_counts()["missing"] == 1
