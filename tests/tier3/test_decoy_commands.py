# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI authoring boundary for the opt-in decoy selection workflow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli


def _arguments(tmp_path: Path) -> list[str]:
    target = tmp_path / "skills" / "target"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("# Target", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    return [
        "tier3",
        "select-decoys",
        str(target),
        "--catalog",
        str(catalog),
        "--catalog-root",
        str(target.parent),
        "--count",
        "2",
        "--exclude",
        "helper",
        "--model",
        "example-model",
    ]


def test_selection_command_passes_explicit_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import decoys

    manifest = {"selected": [{"path": "neighbor-a"}, {"path": "neighbor-b"}]}
    select = Mock(return_value=manifest)
    write = Mock()
    monkeypatch.setattr(decoys, "select_decoys", select)
    monkeypatch.setattr(decoys, "write_decoy_manifest", write)
    output = tmp_path / "selection.json"
    result = CliRunner().invoke(cli, [*_arguments(tmp_path), "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert "Saved 2 decoys" in result.output
    assert select.call_args.kwargs["count"] == 2
    assert select.call_args.kwargs["exclude"] == ("helper",)
    assert select.call_args.kwargs["client"]._model == "example-model"
    write.assert_called_once_with(manifest, output)


@pytest.mark.parametrize("destination", ["catalog.json", "skills/target/SKILL.md", "skills/neighbor/new.json"])
def test_output_cannot_replace_or_mutate_catalog_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination: str
) -> None:
    from skillevaluator.tier3 import decoys

    select = Mock()
    monkeypatch.setattr(decoys, "select_decoys", select)
    result = CliRunner().invoke(cli, [*_arguments(tmp_path), "--output", str(tmp_path / destination)])

    assert result.exit_code == 1
    select.assert_not_called()
    assert (tmp_path / "skills" / "target" / "SKILL.md").read_text(encoding="utf-8") == "# Target"


def test_target_evaluator_directory_can_hold_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import decoys

    monkeypatch.setattr(decoys, "select_decoys", Mock(return_value={"selected": []}))
    write = Mock()
    monkeypatch.setattr(decoys, "write_decoy_manifest", write)
    result = CliRunner().invoke(
        cli,
        [*_arguments(tmp_path), "--output", str(tmp_path / "skills" / "target" / "evals" / "decoys.json")],
    )

    assert result.exit_code == 0, result.output
    write.assert_called_once()


def test_selection_failure_does_not_write_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import decoys

    monkeypatch.setattr(decoys, "select_decoys", Mock(side_effect=ValueError("Not enough eligible decoys")))
    output = tmp_path / "selection.json"
    result = CliRunner().invoke(cli, [*_arguments(tmp_path), "--output", str(output)])

    assert result.exit_code == 1
    assert "Not enough eligible decoys" in result.output
    assert not output.exists()


def test_selection_help_does_not_require_llm_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("NVIDIA_API_KEY", "OPENAI_API_KEY", "SKILL_EVAL_LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    result = CliRunner().invoke(cli, ["tier3", "select-decoys", "--help"])

    assert result.exit_code == 0
    assert "--catalog-root" in result.output
