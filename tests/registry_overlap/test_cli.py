# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from click.testing import CliRunner

from skillevaluator.registry_overlap.__main__ import cli


def test_help_does_not_require_database_or_credentials():
    result = CliRunner().invoke(cli, ["--help"], env={"SKILL_EVAL_REGISTRY_DSN": None})
    assert result.exit_code == 0
    for name in ("init", "sync", "search", "reindex", "activate", "status", "retire"):
        assert name in result.output


def test_status_fails_explicitly_without_dsn():
    result = CliRunner().invoke(cli, ["status"], env={"SKILL_EVAL_REGISTRY_DSN": None})
    assert result.exit_code == 1
    assert "SKILL_EVAL_REGISTRY_DSN" in result.output


def test_invalid_sync_does_not_open_database(tmp_path):
    batch = tmp_path / "batch.json"
    batch.write_text("{}", encoding="utf-8")
    result = CliRunner().invoke(cli, ["sync", "--batch", str(batch)], env={"SKILL_EVAL_REGISTRY_DSN": None})
    assert result.exit_code == 2
    assert "--batch and --sha256" in result.output
