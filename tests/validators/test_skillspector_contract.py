# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end contract tests for the public SkillSpector pin."""

from pathlib import Path

from packaging.version import Version

from skillevaluator.utils.tool_runner import Tools
from skillevaluator.validators.security import SecurityValidator

FIXTURE = Path(__file__).parents[1] / "fixtures" / "skills" / "skillspector-contract"
MINIMUM_FIXED_VERSION = Version("2.4.1")


def test_skillspector_public_pin_preserves_safe_build_idioms() -> None:
    """Safe native build idioms stay clean without hiding unsafe controls."""
    assert Tools.skillspector.is_available, "SkillSpector is a required security-extra dependency"

    result = SecurityValidator(use_llm=False).validate_security_only(FIXTURE)
    scanner_version = Version(str(result.metadata["skillspector_version"]))
    assert scanner_version >= MINIMUM_FIXED_VERSION

    rule_paths = {
        (finding.check_name.rsplit("(", 1)[-1].rstrip(")"), Path(finding.file_path).name) for finding in result.findings
    }
    assert ("TM1", "unsafe-cleanup.sh") in rule_paths
    assert ("PE3", "unsafe-passwd.sh") in rule_paths
    assert (
        not {
            ("TM1", "vector_similarity.cpp"),
            ("PE3", "build.sh"),
        }
        & rule_paths
    )
