# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct ``tier2 PLUGIN`` workflow with an optional local catalog."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.embedding.client import EmbeddingClient
from skillevaluator.models.result import ValidationResult
from skillevaluator.reporting.cli import CLIReporter
from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.validators.similarity import SimilarityValidator

_VOCAB = ("deploy", "kubernetes", "cluster", "docs", "markdown", "confluence")


def _vector(text: str) -> list[float]:
    words = text.lower().replace(":", " ").split()
    return [float(sum(1 for word in words if word.startswith(term))) for term in _VOCAB] + [0.05]


@pytest.fixture(autouse=True)
def fake_providers(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    for name in list(os.environ):
        if name.startswith(("SKILL_EVAL_", "SKILLSPECTOR_")) or name in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}:
            monkeypatch.delenv(name)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-credential")
    calls: list[list[str]] = []

    def embed(_self: EmbeddingClient, texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [_vector(text) for text in texts]

    monkeypatch.setattr(EmbeddingClient, "embed", embed)
    monkeypatch.setattr(EmbeddingClient, "embed_single", lambda _self, text: embed(_self, [text])[0])
    return calls


@pytest.fixture(autouse=True)
def offline_context_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier2 import commands

    def context(_plugin_root: Path, **_kwargs: object) -> list[ValidationResult]:
        result = ValidationResult(validator_name="Context Deduplication")
        result.add_success("context_dedup", "No redundant content detected")
        result.metadata["advisory_tier2"] = True
        return [result]

    monkeypatch.setattr(commands, "run_plugin_skill_context_dedup", context)


def _plugin(root: Path, name: str, description: str, skills: dict[str, str]) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "agent_plugin.yaml").write_text(
        f"name: {name}\ndescription: {description}\nauthor: {{email: dev@example.com}}\n",
        encoding="utf-8",
    )
    for skill_name, skill_description in skills.items():
        skill_dir = directory / "skills" / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill_name}\ndescription: {skill_description}\n---\n", encoding="utf-8"
        )
    return directory


@pytest.fixture
def plugins(tmp_path: Path) -> Path:
    root = tmp_path / "plugins"
    _plugin(root, "alpha", "Deploy kubernetes cluster workloads", {"deploy-app": "Deploy apps to a kubernetes cluster"})
    _plugin(
        root, "beta", "Deploy kubernetes cluster services", {"deploy-svc": "Deploy services to a kubernetes cluster"}
    )
    _plugin(root, "gamma", "Convert confluence docs to markdown", {"conf2md": "Convert confluence docs to markdown"})
    return root


def _report(output_dir: Path) -> dict:
    [path] = list(output_dir.glob("*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def test_save_catalog_cli_accepts_plugin_type(plugins: Path, tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    result = CliRunner().invoke(
        cli,
        ["tier2", "similarity-check", str(plugins), "--type", "plugin", "--save-catalog", str(catalog), "-r", "cli"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(catalog.read_text(encoding="utf-8"))["schema_version"] == 2


def test_tier2_plugin_compares_with_local_catalog(plugins: Path, tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    assert SimilarityValidator(content_type="plugin", save_catalog_path=catalog).validate(plugins).passed
    reports = tmp_path / "reports"

    result = CliRunner().invoke(
        cli,
        ["tier2", str(plugins / "alpha"), "--catalog", str(catalog), "-r", "cli", "-r", "json", "-o", str(reports)],
    )

    assert result.exit_code == 0, result.output
    flattened = " ".join(result.output.replace("│", " ").split())
    assert "Compared with 3 catalog" in flattened
    payload = _report(reports)
    by_name = {item["validator"]: item for item in payload["results"]}
    inter_skill = by_name["Inter-Skill Deduplication"]["plugin"]["catalog_skill_similarity"]
    inter_plugin = by_name["Inter-Plugin Deduplication"]["plugin"]["inter_plugin_similarity"]
    assert inter_skill["status"] == "compared"
    assert inter_skill["matches"] == [{"skill": "skills/deploy-app", "match": "deploy-svc", "similarity": 1.0}]
    assert inter_plugin["status"] == "compared"
    assert [match["name"] for match in inter_plugin["matches"]] == ["beta"]
    assert inter_plugin["matches"][0]["verdict"] is None
    stages = payload["workflow"]["stages"]
    assert set(stages) == {"plugin_references", "intra_skill", "catalog_skills", "catalog_plugins"}
    assert stages["catalog_plugins"]["status"] == "passed"
    assert payload["overall_passed"] is True
    assert str(tmp_path) not in json.dumps(payload)


def test_tier2_plugin_without_catalog_records_skipped_catalog_checks(plugins: Path, tmp_path: Path) -> None:
    reports = tmp_path / "reports"

    result = CliRunner().invoke(cli, ["tier2", str(plugins / "alpha"), "-r", "json", "-o", str(reports)])

    assert result.exit_code == 0, result.output
    assert "catalog plugins skipped" in result.output
    stages = _report(reports)["workflow"]["stages"]
    assert stages["catalog_skills"]["status"] == "skipped"
    assert "--catalog" in stages["catalog_plugins"]["reason"]


def test_tier2_plugin_invalid_catalog_fails_before_model_work(
    plugins: Path, tmp_path: Path, fake_providers: list[list[str]]
) -> None:
    catalog = tmp_path / "bad.json"
    catalog.write_text("[]", encoding="utf-8")

    result = CliRunner().invoke(cli, ["tier2", str(plugins / "alpha"), "--catalog", str(catalog)])

    assert result.exit_code != 0
    assert "Catalog" in result.output
    assert not fake_providers


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--llm"], "--llm requires --catalog"),
        (["--full-body"], "--full-body is not supported for plugins"),
    ],
)
def test_tier2_plugin_rejects_invalid_scope(plugins: Path, args: list[str], message: str) -> None:
    result = CliRunner().invoke(cli, ["tier2", str(plugins / "alpha"), *args])

    assert result.exit_code == 2
    assert message in result.output


def test_tier2_skill_rejects_plugin_only_llm_flag(plugins: Path) -> None:
    result = CliRunner().invoke(cli, ["tier2", str(plugins / "alpha" / "skills" / "deploy-app"), "--llm"])

    assert result.exit_code == 2
    assert "--llm applies to plugin catalog comparison only" in result.output


def test_tier2_plugin_llm_flag_enables_verdicts(plugins: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator.tier2 import commands

    catalog = tmp_path / "catalog.json"
    assert SimilarityValidator(content_type="plugin", save_catalog_path=catalog).validate(plugins).passed
    captured: dict[str, object] = {}

    def scan(_plugin_root: Path, **kwargs: object) -> list[ValidationResult]:
        captured.update(kwargs)
        return [ValidationResult(validator_name="Plugin Dependency Deduplication")]

    monkeypatch.setattr(commands, "run_plugin_dedup_scan", scan)

    result = CliRunner().invoke(cli, ["tier2", str(plugins / "alpha"), "--catalog", str(catalog), "--llm", "-r", "cli"])

    assert result.exit_code == 0, result.output
    assert captured["llm_verdict"] is True
    assert captured["catalog"] == catalog


def test_reporters_render_plugin_catalog_similarity() -> None:
    result = ValidationResult(validator_name="Inter-Plugin Deduplication")
    result.metadata["plugin"] = {
        "inter_plugin_similarity": {"status": "skipped", "catalog_entries": 0, "matches": [], "reason": "No [catalog]"},
        "root": "/host/path/must/not/leak",
    }

    data = json.loads(JSONReporter(include_timestamp=False).render(result))
    rendered = CLIReporter().render_all([result])

    assert data["plugin"] == {
        "inter_plugin_similarity": {"status": "skipped", "catalog_entries": 0, "matches": [], "reason": "No [catalog]"}
    }
    assert "Skipped: No [catalog]" in rendered
