# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for embedding tests."""

from pathlib import Path

import pytest


@pytest.fixture
def write_skill():
    """Factory fixture for creating test skill directories with SKILL.md."""

    def _create(root: Path, name: str, description: str) -> Path:
        skill_dir = root / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n")
        return skill_dir

    return _create
