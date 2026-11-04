# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI authoring boundary for the opt-in decoy selection workflow."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli


def _manifest() -> dict:
    from skillevaluator.tier3.decoy_identity import DECOY_SOURCE_DIGEST_ALGORITHM

    return {
        "schema_version": 1,
        "strategy": "description-similarity",
        "catalog_root": ".",
        "catalog": {
            "sha256": "0" * 64,
            "provider": "nv_build",
            "model": "test-model",
            "mode": "description",
            "endpoint_fingerprint": "sha256:" + "1" * 64,
        },
        "target": {"name": "target", "skill_name": "target", "manifest_sha256": "2" * 64},
        "count": 1,
        "exclude": [],
        "selected": [
            {"path": "neighbor", "name": "neighbor", "skill_name": "neighbor", "score": 0.5, "source_digest": "3" * 64}
        ],
        "source_digest_algorithm": DECOY_SOURCE_DIGEST_ALGORITHM,
    }


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


@pytest.mark.parametrize("filename", ["evals.json", "evals.jsonl", "config.yml", "config.yaml"])
@pytest.mark.parametrize("existing", [False, True])
def test_manifest_output_cannot_claim_evaluation_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str, existing: bool
) -> None:
    from skillevaluator.tier3 import decoys

    arguments = _arguments(tmp_path)
    output = tmp_path / "skills" / "target" / "evals" / filename
    output.parent.mkdir()
    if existing:
        output.write_bytes(b"evaluation input that must survive\n")
    select = Mock(return_value=_manifest())
    monkeypatch.setattr(decoys, "select_decoys", select)

    result = CliRunner().invoke(cli, [*arguments, "--output", str(output)])

    assert result.exit_code == 1
    assert "evaluation dataset or run policy" in result.output
    select.assert_not_called()
    if existing:
        assert output.read_bytes() == b"evaluation input that must survive\n"
    else:
        assert not output.exists()


@pytest.mark.parametrize(
    ("destination", "contents"),
    [
        ("skills/target/evals/oracle.txt", "private evaluation answers\n"),
        ("skills/target/evals/custom.json", '[{"id": "case-001"}]\n'),
        ("skills/target/evals/selection.json", '{"selected": []}\n'),
        ("report.txt", "existing report\n"),
    ],
)
def test_existing_nonmanifest_output_survives_real_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination: str, contents: str
) -> None:
    from skillevaluator.tier3 import decoys

    arguments = _arguments(tmp_path)
    output = tmp_path / destination
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(contents, encoding="utf-8")
    select = Mock(return_value=_manifest())
    monkeypatch.setattr(decoys, "select_decoys", select)

    result = CliRunner().invoke(cli, [*arguments, "--output", str(output)])

    assert result.exit_code == 1
    assert "not a decoy manifest" in result.output
    select.assert_not_called()
    assert output.read_text(encoding="utf-8") == contents


def test_existing_stale_manifest_can_be_regenerated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import decoys

    arguments = _arguments(tmp_path)
    output = tmp_path / "skills" / "target" / "evals" / "decoys.json"
    previous = _manifest()
    # Regeneration must work after a target/source edit or a digest recipe change.
    previous["source_digest_algorithm"] = "skill-evaluator-source-tree/2"
    previous["target"]["manifest_sha256"] = "a" * 64
    decoys.write_decoy_manifest(previous, output)
    replacement = copy.deepcopy(previous)
    replacement["target"]["manifest_sha256"] = "b" * 64
    monkeypatch.setattr(decoys, "select_decoys", Mock(return_value=replacement))

    result = CliRunner().invoke(cli, [*arguments, "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8")) == replacement


@pytest.mark.parametrize("destination", ["CATALOG.json", "SKILLS/neighbor/SKILL.md", "SKILLS/new/decoys.json"])
def test_case_insensitive_filesystem_identity_protects_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination: str
) -> None:
    from skillevaluator.tier3 import decoys

    arguments = _arguments(tmp_path)
    protected = tmp_path / "skills" / "neighbor" / "SKILL.md"
    protected.parent.mkdir()
    protected.write_text("neighbor instructions\n", encoding="utf-8")
    real_samefile = Path.samefile
    aliases = {tmp_path / "SKILLS": tmp_path / "skills", tmp_path / "CATALOG.json": tmp_path / "catalog.json"}

    def canonical(path: Path) -> Path:
        for alias, actual in aliases.items():
            if path.is_relative_to(alias):
                return actual / path.relative_to(alias)
        return path

    def samefile(path: Path, other: Path) -> bool:
        # Model only identity lookup on a case-insensitive filesystem; retain
        # ordinary missing-file behavior for not-yet-created output ancestors.
        return real_samefile(canonical(path), canonical(other))

    monkeypatch.setattr(Path, "samefile", samefile)
    select = Mock(return_value=_manifest())
    monkeypatch.setattr(decoys, "select_decoys", select)

    result = CliRunner().invoke(cli, [*arguments, "--output", str(tmp_path / destination)])

    assert result.exit_code == 1
    select.assert_not_called()
    assert protected.read_text(encoding="utf-8") == "neighbor instructions\n"
    assert (tmp_path / "catalog.json").read_text(encoding="utf-8") == "{}"
    if destination == "CATALOG.json":
        assert "must not replace the embedding catalog" in result.output
    else:
        assert "outside the skill collection" in result.output


@pytest.mark.parametrize("destination", ["CATALOG.json", "SKILLS/neighbor/SKILL.md"])
def test_native_case_alias_cannot_overwrite_an_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination: str
) -> None:
    from skillevaluator.tier3 import decoys

    arguments = _arguments(tmp_path)
    neighbor = tmp_path / "skills" / "neighbor" / "SKILL.md"
    neighbor.parent.mkdir()
    neighbor.write_text("neighbor instructions\n", encoding="utf-8")
    output = tmp_path / destination
    if not output.exists():
        pytest.skip("requires a case-insensitive filesystem")
    before = output.read_bytes()
    select = Mock(return_value=_manifest())
    monkeypatch.setattr(decoys, "select_decoys", select)

    result = CliRunner().invoke(cli, [*arguments, "--output", str(output)])

    assert result.exit_code == 1
    select.assert_not_called()
    assert output.read_bytes() == before


def test_distinct_case_sensitive_directory_can_hold_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier3 import decoys

    arguments = _arguments(tmp_path)
    separate = tmp_path / "SKILLS"
    separate.mkdir(exist_ok=True)
    if separate.samefile(tmp_path / "skills"):
        pytest.skip("requires distinct case-sensitive directory names")
    output = separate / "decoys.json"
    manifest = _manifest()
    monkeypatch.setattr(decoys, "select_decoys", Mock(return_value=manifest))

    result = CliRunner().invoke(cli, [*arguments, "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
