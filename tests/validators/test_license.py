# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for license compliance validator."""

from pathlib import Path

import pytest

from skillevaluator.validators.license import LicenseDetection, LicenseValidator

GPL_V3_LICENSE_TEXT = """
                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""

APACHE_2_LICENSE_TEXT = """
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.
"""

MIT_LICENSE_TEXT = """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def skill_with_apache_license(tmp_path: Path) -> Path:
    """Create a skill directory with Apache 2.0 license."""
    skill_dir = tmp_path / "apache-skill"
    skill_dir.mkdir()

    # Create SKILL.md with license reference
    (skill_dir / "SKILL.md").write_text("""---
name: apache-skill
description: A skill with Apache 2.0 license
license: See LICENSE.txt
---

# Apache Licensed Skill
""")

    # Create Apache 2.0 LICENSE file
    (skill_dir / "LICENSE.txt").write_text("""
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.
""")

    return skill_dir


@pytest.fixture
def skill_with_mit_license(tmp_path: Path) -> Path:
    """Create a skill directory with MIT license."""
    skill_dir = tmp_path / "mit-skill"
    skill_dir.mkdir()

    # Create SKILL.md with direct license declaration
    (skill_dir / "SKILL.md").write_text("""---
name: mit-skill
description: A skill with MIT license
license: MIT
---

# MIT Licensed Skill
""")

    (skill_dir / "LICENSE").write_text(MIT_LICENSE_TEXT)

    return skill_dir


@pytest.fixture
def skill_with_gpl_license(tmp_path: Path) -> Path:
    """Create a skill directory with GPL v3 license (blocked)."""
    skill_dir = tmp_path / "gpl-skill"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text("""---
name: gpl-skill
description: A skill with GPL license
---

# GPL Licensed Skill
""")

    (skill_dir / "LICENSE").write_text(GPL_V3_LICENSE_TEXT)

    return skill_dir


@pytest.fixture
def skill_with_proprietary_license(tmp_path: Path) -> Path:
    """Create a skill directory with proprietary license."""
    skill_dir = tmp_path / "proprietary-skill"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text("""---
name: proprietary-skill
description: A skill with proprietary license
---

# Proprietary Skill
""")

    (skill_dir / "LICENSE").write_text("""
PROPRIETARY SOFTWARE LICENSE

All rights reserved.
This software is confidential and proprietary.
Not for distribution.
""")

    return skill_dir


@pytest.fixture
def skill_with_spdx_headers(tmp_path: Path) -> Path:
    """Create a skill directory with SPDX headers in source files."""
    skill_dir = tmp_path / "spdx-skill"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text("""---
name: spdx-skill
description: A skill with SPDX headers in source files
---

# SPDX Header Skill
""")

    # Create Python file with SPDX header
    (skill_dir / "script.py").write_text("""#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 Example Organization

def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")

    # Create another file with same license
    (skill_dir / "utils.py").write_text("""# SPDX-License-Identifier: Apache-2.0
# Utility functions

def helper():
    return True
""")

    return skill_dir


@pytest.fixture
def skill_with_no_license(tmp_path: Path) -> Path:
    """Create a skill directory with no license information."""
    skill_dir = tmp_path / "no-license-skill"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text("""---
name: no-license-skill
description: A skill with no license information
---

# No License Skill
""")

    return skill_dir


@pytest.fixture
def workflow_with_license(tmp_path: Path) -> Path:
    """Create a workflow directory with license in frontmatter."""
    workflow_dir = tmp_path / "licensed-workflow"
    workflow_dir.mkdir()

    (workflow_dir / "workflow-rules.mdc").write_text("""---
alwaysApply: false
title: Licensed Workflow
description: A workflow with explicit license
license: BSD-3-Clause
metadata:
  author: Test <test@example.com>
---

# Licensed Workflow

This workflow is BSD licensed.
""")

    return workflow_dir


def create_skill_with_license_file_only(tmp_path: Path, name: str) -> Path:
    """Create a skill whose LICENSE file is valid but SKILL.md has no license field."""
    skill_dir = tmp_path / name
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: Skill with a LICENSE file but no frontmatter license declaration
---

# License File Only
""")

    (skill_dir / "LICENSE").write_text("""MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.
""")

    return skill_dir


def create_skill_with_declared_license_and_file(
    tmp_path: Path, name: str, declared_license: str, license_text: str
) -> Path:
    """Create a skill with both SKILL.md license and LICENSE file content."""
    skill_dir = tmp_path / name
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: Skill with both frontmatter and file license declarations
license: {declared_license}
---

# Declared License Skill
""")

    (skill_dir / "LICENSE").write_text(license_text)

    return skill_dir


# =============================================================================
# BASIC VALIDATION TESTS
# =============================================================================


class TestLicenseValidator:
    """Test cases for LicenseValidator."""

    def test_validator_properties(self):
        """Test validator name and description."""
        validator = LicenseValidator()
        assert validator.name == "License Compliance"
        assert "license" in validator.description.lower()

    def test_config_loading(self):
        """Test that license config loads correctly."""
        validator = LicenseValidator()
        config = validator.config

        assert "allowed_licenses" in config
        assert "blocked_licenses" in config
        assert "license_patterns" in config
        assert "Apache-2.0" in config["allowed_licenses"]
        assert "GPL-3.0" in config["blocked_licenses"]

    def test_strict_mode_initialization(self):
        """Test strict mode flag."""
        validator_default = LicenseValidator()
        validator_strict = LicenseValidator(strict_mode=True)

        assert validator_default.strict_mode is False
        assert validator_strict.strict_mode is True


# =============================================================================
# TIER 1: FRONTMATTER DETECTION TESTS
# =============================================================================


class TestFrontmatterDetection:
    """Test Tier 1: Frontmatter license detection."""

    def test_detects_direct_license_in_frontmatter(self, skill_with_mit_license: Path):
        """Test detection of direct license declaration in frontmatter."""
        validator = LicenseValidator()
        result = validator.validate(skill_with_mit_license)

        assert result.passed
        assert result.metadata.get("license") == "MIT"
        assert result.metadata.get("license_status") == "allowed"
        assert all(f.check_name != "frontmatter_license_missing" for f in result.findings)

    def test_detects_file_reference_in_frontmatter(self, skill_with_apache_license: Path):
        """Test detection of license file reference in frontmatter."""
        validator = LicenseValidator()
        result = validator.validate(skill_with_apache_license)

        assert result.passed
        assert "Apache-2.0" in result.metadata.get("license", "")
        assert any("LICENSE" in m for m in result.messages)
        assert all(f.check_name != "frontmatter_license_mismatch" for f in result.findings)

    def test_direct_frontmatter_license_matches_license_file(self, skill_with_mit_license: Path):
        """Matching SKILL.md and LICENSE declarations should pass."""
        result = LicenseValidator().validate(skill_with_mit_license)

        assert result.passed
        assert result.metadata.get("license_status") == "allowed"
        assert all(f.check_name != "frontmatter_license_mismatch" for f in result.findings)

    def test_mit_frontmatter_does_not_hide_gpl_license_file(self, tmp_path: Path):
        """MIT in frontmatter must not hide a GPL LICENSE file (#85)."""
        skill_dir = create_skill_with_declared_license_and_file(
            tmp_path,
            "mit-claims-gpl-body",
            "MIT",
            GPL_V3_LICENSE_TEXT,
        )

        result = LicenseValidator().validate(skill_dir)

        assert not result.passed
        error_text = " ".join(result.errors).lower()
        assert any(token in error_text for token in ("gpl", "copyleft", "blocked", "conflict", "mismatch"))
        assert not any("License: MIT (ALLOWED" in message for message in result.messages)
        assert result.metadata.get("license_status") != "allowed"

    def test_mit_frontmatter_without_license_file_is_allowed(self, tmp_path: Path):
        """Frontmatter-only MIT remains allowed when no LICENSE file exists."""
        skill_dir = tmp_path / "mit-frontmatter-only"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: mit-frontmatter-only
description: MIT in frontmatter with no LICENSE file
license: MIT
---

# MIT Frontmatter Only
""")

        result = LicenseValidator().validate(skill_dir)

        assert result.passed
        assert result.metadata.get("license") == "MIT"
        assert result.metadata.get("license_status") == "allowed"

    def test_gpl_license_file_without_frontmatter_is_blocked(self, skill_with_gpl_license: Path):
        """LICENSE-only GPL remains blocked when frontmatter has no license field."""
        result = LicenseValidator().validate(skill_with_gpl_license)

        assert not result.passed
        assert any(f.check_name == "blocked_license" for f in result.findings)
        error_text = " ".join(result.errors).lower()
        assert "gpl" in error_text or "copyleft" in error_text

    def test_gpl_frontmatter_is_not_hidden_by_mit_license_file(self, tmp_path: Path):
        """Blocked frontmatter must still fail even if the LICENSE file is allowed."""
        skill_dir = create_skill_with_declared_license_and_file(
            tmp_path,
            "gpl-claims-mit-body",
            "GPL-3.0",
            MIT_LICENSE_TEXT,
        )

        result = LicenseValidator().validate(skill_dir)

        assert not result.passed
        assert any(f.check_name == "blocked_license" for f in result.findings)
        assert result.metadata.get("license_status") != "allowed"

    def test_conflicting_allowed_licenses_fail_closed(self, tmp_path: Path):
        """MIT frontmatter vs Apache LICENSE must not silently pick either side."""
        skill_dir = create_skill_with_declared_license_and_file(
            tmp_path,
            "mit-vs-apache-body",
            "MIT",
            APACHE_2_LICENSE_TEXT,
        )

        result = LicenseValidator().validate(skill_dir)

        assert not result.passed
        error_text = " ".join(result.errors).lower()
        assert any(token in error_text for token in ("conflict", "mismatch", "differ"))
        assert any(f.check_name == "frontmatter_license_mismatch" for f in result.findings)
        assert result.metadata.get("license_status") == "conflict"
        assert not any("ALLOWED - permissive" in message for message in result.messages)

    def test_copying_gpl_is_not_hidden_by_mit_license_file(self, tmp_path: Path):
        """MIT LICENSE plus GPL COPYING must fail closed even with MIT frontmatter."""
        skill_dir = create_skill_with_declared_license_and_file(
            tmp_path,
            "mit-license-gpl-copying",
            "MIT",
            MIT_LICENSE_TEXT,
        )
        (skill_dir / "COPYING").write_text(GPL_V3_LICENSE_TEXT)

        result = LicenseValidator(strict_mode=True).validate(skill_dir)

        assert not result.passed
        assert any(f.check_name == "blocked_license" for f in result.findings)
        assert result.metadata.get("license_status") == "blocked"
        assert not any("License: MIT (ALLOWED" in message for message in result.messages)

    def test_single_file_mit_and_gpl_signatures_fail_closed(self, tmp_path: Path):
        """One LICENSE file matching both MIT and GPL patterns must not report MIT allowed."""
        skill_dir = tmp_path / "dual-signature-license"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: dual-signature-license
description: LICENSE file contains both MIT and GPL signatures
license: MIT
---

# Dual signature
""")
        (skill_dir / "LICENSE").write_text(MIT_LICENSE_TEXT + "\n" + GPL_V3_LICENSE_TEXT)

        result = LicenseValidator(strict_mode=True).validate(skill_dir)

        assert not result.passed
        assert any(f.check_name == "blocked_license" for f in result.findings)
        assert result.metadata.get("license_status") == "blocked"

    def test_unrecognized_lgpl21_license_file_fails_closed(self, tmp_path: Path):
        """A present LICENSE file without a configured signature cannot be treated as absent."""
        skill_dir = create_skill_with_declared_license_and_file(
            tmp_path,
            "mit-frontmatter-lgpl21-file",
            "MIT",
            """GNU LESSER GENERAL PUBLIC LICENSE
Version 2.1, February 1999

Copyright (C) 1991, 1999 Free Software Foundation, Inc.
""",
        )

        result = LicenseValidator(strict_mode=True).validate(skill_dir)

        assert not result.passed
        assert any(f.check_name == "unrecognized_license_file" for f in result.findings)
        assert result.metadata.get("license_status") == "unrecognized"
        assert result.metadata.get("license_status") != "allowed"

    def test_notice_apache_attribution_does_not_conflict_with_mit(self, tmp_path: Path):
        """NOTICE stays informational; Apache text there must not mismatch MIT LICENSE."""
        skill_dir = create_skill_with_declared_license_and_file(
            tmp_path,
            "mit-with-apache-notice",
            "MIT",
            MIT_LICENSE_TEXT,
        )
        (skill_dir / "NOTICE").write_text(APACHE_2_LICENSE_TEXT)

        result = LicenseValidator(strict_mode=True).validate(skill_dir)

        assert result.passed
        assert all(f.check_name != "frontmatter_license_mismatch" for f in result.findings)
        assert result.metadata.get("license_status") == "allowed"

    def test_strict_dual_license_expression_with_unknown_component_fails(self, tmp_path: Path):
        """Strict mode should fail when any expression component is unknown."""
        skill_dir = create_skill_with_declared_license_and_file(
            tmp_path,
            "dual-license-unknown",
            "MIT OR FOO-1.0",
            """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.
""",
        )

        result = LicenseValidator(strict_mode=True).validate(skill_dir)

        assert not result.passed
        assert result.metadata.get("license_status") == "unknown"
        assert any(f.check_name == "unknown_license" for f in result.findings)
        assert all(f.check_name != "frontmatter_license_mismatch" for f in result.findings)

    def test_workflow_frontmatter_detection(self, workflow_with_license: Path):
        """Test detection of license in workflow frontmatter."""
        validator = LicenseValidator()
        result = validator.validate(workflow_with_license)

        assert result.passed
        assert "BSD" in result.metadata.get("license", "")


# =============================================================================
# TIER 2: LICENSE FILE DETECTION TESTS
# =============================================================================


class TestLicenseFileDetection:
    """Test Tier 2: LICENSE file detection."""

    def test_detects_apache_license_file(self, skill_with_apache_license: Path):
        """Test detection of Apache 2.0 from LICENSE file."""
        validator = LicenseValidator()
        result = validator.validate(skill_with_apache_license)

        assert result.passed
        assert "Apache-2.0" in result.metadata.get("license", "")

    def test_detects_mit_license_file(self, tmp_path: Path):
        """Test detection of MIT from LICENSE file only."""
        skill_dir = tmp_path / "mit-only"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: mit-only
description: Skill with MIT LICENSE file but no frontmatter declaration
---

# MIT Only
""")

        (skill_dir / "LICENSE").write_text("""MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.
""")

        validator = LicenseValidator()
        result = validator.validate(skill_dir)

        assert result.passed
        assert "MIT" in result.metadata.get("license", "")

    def test_detects_bsd_license_file(self, tmp_path: Path):
        """Test detection of BSD-3-Clause from LICENSE file."""
        skill_dir = tmp_path / "bsd-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: bsd-skill
description: BSD licensed skill
---

# BSD Skill
""")

        (skill_dir / "LICENSE.txt").write_text("""BSD 3-Clause License

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.
""")

        validator = LicenseValidator()
        result = validator.validate(skill_dir)

        assert result.passed
        assert "BSD-3-Clause" in result.metadata.get("license", "")


# =============================================================================
# TIER 3: SPDX HEADER DETECTION TESTS
# =============================================================================


class TestSPDXHeaderDetection:
    """Test Tier 3: SPDX header detection in source files."""

    def test_detects_spdx_header_in_python(self, skill_with_spdx_headers: Path):
        """Test detection of SPDX headers in Python files."""
        validator = LicenseValidator()
        result = validator.validate(skill_with_spdx_headers)

        assert result.passed
        assert "Apache-2.0" in result.metadata.get("license", "")
        assert any("SPDX" in m for m in result.messages)

    def test_multiple_files_same_license(self, tmp_path: Path):
        """Test detection when multiple files have same SPDX header."""
        skill_dir = tmp_path / "multi-spdx"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: multi-spdx
description: Multiple files with SPDX
---

# Multi SPDX
""")

        for i in range(3):
            (skill_dir / f"file{i}.py").write_text(f"""# SPDX-License-Identifier: MIT
# File {i}
""")

        validator = LicenseValidator()
        result = validator.validate(skill_dir)

        assert result.passed
        assert "MIT" in result.metadata.get("license", "")

    def test_spdx_or_expression_with_gpl_is_blocked(self, tmp_path: Path):
        """MIT OR GPL-3.0 in an SPDX header must not be truncated to allowed MIT (#86)."""
        skill_dir = tmp_path / "spdx-or-gpl"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: spdx-or-gpl
description: Skill whose only license signal is an SPDX OR expression
---

# SPDX OR
""")
        (skill_dir / "script.py").write_text("""# SPDX-License-Identifier: MIT OR GPL-3.0
print("hi")
""")

        result = LicenseValidator().validate(skill_dir)

        assert not result.passed
        assert result.metadata.get("license_status") == "blocked"
        assert "GPL-3.0" in (result.metadata.get("license") or "")
        assert any(f.check_name == "blocked_license" for f in result.findings)
        assert not any("License: MIT (ALLOWED" in message for message in result.messages)

    def test_spdx_or_expression_in_block_comment_is_blocked(self, tmp_path: Path):
        """`*/` after MIT OR GPL-3.0 must not turn the GPL half into an unknown id."""
        skill_dir = tmp_path / "spdx-js-block"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: spdx-js-block
description: Skill whose SPDX header is inside a JavaScript block comment
---

# SPDX JS
""")
        (skill_dir / "script.js").write_text("/* SPDX-License-Identifier: MIT OR GPL-3.0 */\n")

        result = LicenseValidator().validate(skill_dir)

        assert not result.passed
        assert result.metadata.get("license_status") == "blocked"
        assert "GPL-3.0" in (result.metadata.get("license") or "")
        assert "*/" not in (result.metadata.get("license") or "")
        assert any(f.check_name == "blocked_license" for f in result.findings)

    def test_spdx_inline_block_comment_with_code_suffix_is_blocked(self, tmp_path: Path):
        """Inline `*/` plus trailing code must not leave GPL-3.0 */ attached to the symbol."""
        skill_dir = tmp_path / "spdx-js-inline"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: spdx-js-inline
description: Skill whose SPDX header shares a line with trailing code
---

# SPDX inline
""")
        (skill_dir / "script.js").write_text("/* SPDX-License-Identifier: MIT OR GPL-3.0 */ const x = 1;\n")

        result = LicenseValidator().validate(skill_dir)

        assert not result.passed
        assert result.metadata.get("license_status") == "blocked"
        assert "GPL-3.0" in (result.metadata.get("license") or "")
        assert "*/" not in (result.metadata.get("license") or "")
        assert any(f.check_name == "blocked_license" for f in result.findings)

    def test_spdx_or_expression_in_html_comment_is_blocked(self, tmp_path: Path):
        """`-->` after MIT OR GPL-3.0 must not hide the GPL half."""
        skill_dir = tmp_path / "spdx-html-comment"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: spdx-html-comment
description: Skill whose SPDX header is inside an HTML comment
---

# SPDX HTML
""")
        (skill_dir / "notes.md").write_text("<!-- SPDX-License-Identifier: MIT OR GPL-3.0 -->\n")

        result = LicenseValidator().validate(skill_dir)

        assert not result.passed
        assert result.metadata.get("license_status") == "blocked"
        assert "GPL-3.0" in (result.metadata.get("license") or "")
        assert "-->" not in (result.metadata.get("license") or "")
        assert any(f.check_name == "blocked_license" for f in result.findings)

    def test_spdx_and_expression_with_gpl_is_blocked(self, tmp_path: Path):
        """Apache-2.0 AND GPL-3.0 must evaluate the copyleft half, not only Apache-2.0."""
        skill_dir = tmp_path / "spdx-and-gpl"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: spdx-and-gpl
description: Skill whose only license signal is an SPDX AND expression
---

# SPDX AND
""")
        (skill_dir / "script.py").write_text("# SPDX-License-Identifier: Apache-2.0 AND GPL-3.0\n")

        result = LicenseValidator().validate(skill_dir)

        assert not result.passed
        assert any(f.check_name == "blocked_license" for f in result.findings)
        assert result.metadata.get("license_status") != "allowed"

    def test_spdx_or_expression_of_allowed_licenses_passes(self, tmp_path: Path):
        """Compound expressions stay allowed when every symbol is permissive."""
        skill_dir = tmp_path / "spdx-or-mit"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("""---
name: spdx-or-mit
description: Skill whose SPDX header is MIT OR MIT-0
---

# SPDX OR allowed
""")
        (skill_dir / "script.py").write_text("# SPDX-License-Identifier: MIT OR MIT-0\n")

        result = LicenseValidator().validate(skill_dir)

        assert result.passed
        assert result.metadata.get("license_status") == "allowed"
        assert "MIT OR MIT-0" in (result.metadata.get("license") or "")


# =============================================================================
# NO LICENSE / UNKNOWN LICENSE TESTS
# =============================================================================


class TestNoLicenseHandling:
    """Test handling of assets with no license information."""

    def test_no_license_strict_mode(self, skill_with_no_license: Path):
        """Test that missing license fails in strict mode."""
        validator = LicenseValidator(strict_mode=True)
        result = validator.validate(skill_with_no_license)

        assert not result.passed
        assert any("no license" in e.lower() for e in result.errors)

    def test_unknown_license_strict_mode(self, tmp_path: Path):
        """Test that unknown license fails in strict mode."""
        skill_dir = tmp_path / "unknown-strict"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: unknown-strict
description: Skill with unknown license
license: Custom-Weird-License-1.0
---

# Unknown License
""")

        validator = LicenseValidator(strict_mode=True)
        result = validator.validate(skill_dir)

        assert not result.passed
        assert any("not in" in e.lower() and "allowlist" in e.lower() for e in result.errors)


# =============================================================================
# FOLDER VALIDATION TESTS
# =============================================================================


class TestFolderValidation:
    """Test validation of folders containing multiple assets."""

    def test_validates_multiple_skills(self, tmp_path: Path):
        """Test validation of folder with multiple skills."""
        skills_folder = tmp_path / "skills"
        skills_folder.mkdir()

        # Create skill 1 with MIT license
        skill1 = skills_folder / "skill-one"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text("""---
name: skill-one
description: First skill
license: MIT
---

# Skill One
""")

        # Create skill 2 with Apache license
        skill2 = skills_folder / "skill-two"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text("""---
name: skill-two
description: Second skill
license: Apache-2.0
---

# Skill Two
""")

        validator = LicenseValidator()
        result = validator.validate(skills_folder)

        assert result.passed
        assert any("2 skill(s)" in m for m in result.messages)


# =============================================================================
# LICENSE NORMALIZATION TESTS
# =============================================================================


class TestLicenseNormalization:
    """Test license ID normalization for comparison."""

    def test_case_insensitive_matching(self, tmp_path: Path):
        """Test that license matching is case-insensitive."""
        skill_dir = tmp_path / "case-test"
        skill_dir.mkdir()

        # Use lowercase 'mit' instead of 'MIT'
        (skill_dir / "SKILL.md").write_text("""---
name: case-test
description: Test case sensitivity
license: mit
---

# Case Test
""")

        validator = LicenseValidator()
        result = validator.validate(skill_dir)

        assert result.passed

    def test_handles_license_suffix(self, tmp_path: Path):
        """Test handling of 'License' suffix in license names."""
        skill_dir = tmp_path / "suffix-test"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: suffix-test
description: Test license suffix
license: MIT License
---

# Suffix Test
""")

        validator = LicenseValidator()
        result = validator.validate(skill_dir)

        # Should normalize "MIT License" to "MIT" and match
        assert result.passed or "MIT" in result.metadata.get("license", "")


# =============================================================================
# DETECTION HELPER TESTS
# =============================================================================


class TestLicenseDetection:
    """Test the LicenseDetection dataclass and helpers."""

    def test_detection_dataclass(self):
        """Test LicenseDetection dataclass creation."""
        detection = LicenseDetection(
            license_id="MIT",
            source="frontmatter",
            confidence="high",
            file_path="SKILL.md",
            details="Found in license field",
        )

        assert detection.license_id == "MIT"
        assert detection.source == "frontmatter"
        assert detection.confidence == "high"
        assert detection.file_path == "SKILL.md"
        assert detection.details == "Found in license field"

    def test_is_file_reference(self):
        """Test file reference detection."""
        validator = LicenseValidator()

        assert validator._is_file_reference("See LICENSE.txt")
        assert validator._is_file_reference("Refer to COPYING")
        assert validator._is_file_reference("see license.md")
        assert not validator._is_file_reference("MIT")
        assert not validator._is_file_reference("Apache-2.0")


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_license_file(self, tmp_path: Path):
        """Test handling of empty LICENSE file."""
        skill_dir = tmp_path / "empty-license"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: empty-license
description: Skill with empty license file
---

# Empty License
""")

        (skill_dir / "LICENSE").write_text("")

        validator = LicenseValidator()
        result = validator.validate(skill_dir)

        # Should handle gracefully - likely warn about unknown license
        assert len(result.warnings) > 0 or len(result.errors) > 0

    def test_binary_file_handling(self, tmp_path: Path):
        """Test that binary files don't cause crashes during SPDX scanning."""
        skill_dir = tmp_path / "binary-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: binary-skill
description: Skill with binary files
---

# Binary Skill
""")

        # Create a binary file
        (skill_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        validator = LicenseValidator()
        result = validator.validate(skill_dir)

        # Should not crash
        assert result is not None

    def test_nonexistent_path(self, tmp_path: Path):
        """Test handling of non-existent path."""
        validator = LicenseValidator()
        nonexistent = tmp_path / "does-not-exist"

        # Should handle gracefully
        result = validator.validate(nonexistent)
        assert not result.passed or len(result.warnings) > 0

    def test_malformed_frontmatter(self, tmp_path: Path):
        """Test handling of malformed YAML frontmatter."""
        skill_dir = tmp_path / "malformed"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("""---
name: malformed
description: [invalid: yaml: here
license: MIT
---

# Malformed
""")

        validator = LicenseValidator()
        result = validator.validate(skill_dir)

        # Should handle parse error gracefully
        assert result is not None


# =============================================================================
# INTEGRATION WITH EXISTING VALIDATORS
# =============================================================================


class TestValidatorIntegration:
    """Test integration with existing validator patterns."""

    def test_metadata_population(self, skill_with_mit_license: Path):
        """Test that validation populates metadata correctly."""
        validator = LicenseValidator()
        result = validator.validate(skill_with_mit_license)

        assert "license" in result.metadata
        assert "license_status" in result.metadata
        assert "license_source" in result.metadata
