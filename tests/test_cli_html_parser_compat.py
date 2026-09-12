# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public CLI coverage for supported HTMLParser patch-version boundaries."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.models.result import ValidationResult
from skillevaluator.validators.code_risk import CodeRiskValidator
from skillevaluator.validators.secrets import SecretsValidator


def test_validate_code_integrity_handles_raw_html_on_supported_python_patches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = tmp_path / "html-parser-compatibility"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: html-parser-compatibility\n"
        "description: Validate raw HTML link parsing on every supported Python patch.\n"
        "---\n"
        "\n"
        "# HTML parser compatibility\n"
        "\n"
        "See the [supporting guide](guide.md).\n",
        encoding="utf-8",
    )
    (skill / "guide.md").write_text(
        "<script>const hidden = '<a href=\"missing-script.md\">';</script>\n"
        '<textarea/><a href="missing-textarea.md">hidden</a></textarea>\n'
        "See the [visible guide](visible.md).\n",
        encoding="utf-8",
    )
    (skill / "visible.md").write_text("# Visible guide\n", encoding="utf-8")

    # Exercise the public CLI and real Hygiene validator without invoking the
    # independent scanner integrations.
    monkeypatch.setattr(CodeRiskValidator, "validate", lambda _self, _path: ValidationResult())
    monkeypatch.setattr(SecretsValidator, "validate", lambda _self, _path: ValidationResult())

    result = CliRunner().invoke(
        cli,
        ["validate", str(skill), "--checks", "code-integrity", "--no-dedup", "-r", "cli"],
    )

    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "Traceback" not in result.output
