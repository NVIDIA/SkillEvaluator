# SPDX-License-Identifier: Apache-2.0
"""Public discovery and static dispatch use the bounded component implementation."""

import json

import pytest

from skillevaluator.cli_core import detect_content_type, resolve_plugin_path
from skillevaluator.models.result import ValidationResult
from skillevaluator.tier1.commands import run_validation


@pytest.mark.parametrize(
    "manifest,payload",
    [
        (
            "agent_plugin.yaml",
            "name: demo\nauthor:\n  email: dev@example.com\nskills:\n  refs: [github::org/repo::skills::demo]\n",
        ),
        (".claude-plugin/plugin.json", '{"name":"demo"}'),
        (".codex-plugin/plugin.json", '{"name":"demo","version":"1.0.0","description":"Demo"}'),
        (".cursor-plugin/plugin.json", '{"name":"demo","version":"1.0.0","description":"Demo"}'),
        (
            "plugin.json",
            '{"$schema":"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json","name":"demo","version":"1.0.0","description":"Demo"}',
        ),
    ],
)
def test_all_profiles_detect_from_directory_or_manifest(tmp_path, manifest, payload):
    path = tmp_path / manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    assert detect_content_type(tmp_path) == "plugin"
    assert detect_content_type(path) == "plugin"
    assert resolve_plugin_path(path) == tmp_path
    assert run_validation(path, checks="schema", content_type="plugin")


def test_unrelated_plugin_json_does_not_change_skill_detection(tmp_path):
    (tmp_path / "plugin.json").write_text('{"name":"unrelated"}')
    (tmp_path / "SKILL.md").write_text("---\nname: ordinary\ndescription: Ordinary skill.\n---\nBody.")
    assert detect_content_type(tmp_path) == "skill"


def test_bundled_quality_receives_public_options_once(tmp_path, monkeypatch):
    from skillevaluator.tier1 import commands

    manifest = tmp_path / ".claude-plugin/plugin.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"name": "demo"}))
    member = tmp_path / "skills/a/SKILL.md"
    member.parent.mkdir(parents=True)
    member.write_text("---\nname: a\ndescription: A skill.\n---\nBody.")
    seen = []

    class Quality:
        name = "Quality check"
        description = "Fixture quality"

        def __init__(self, **options):
            seen.append(options)

        def validate(self, path):
            assert (path / "SKILL.md").is_file()
            return ValidationResult()

    monkeypatch.setattr(commands, "QualityScoreValidator", Quality)
    config = tmp_path / "quality.yml"
    results = run_validation(tmp_path, checks="quality", content_type="plugin", min_score=91, quality_config=config)
    assert seen == [{"min_score": 91, "quality_config": config}]
    assert any(result.metadata.get("plugin_member") == "skills/a" for result in results)


def test_schema_failure_stops_before_scanner_construction_when_fail_fast(tmp_path, monkeypatch):
    from skillevaluator.tier1 import commands

    manifest = tmp_path / ".claude-plugin/plugin.json"
    manifest.parent.mkdir()
    manifest.write_text("{broken")

    def forbidden(*args, **kwargs):
        pytest.fail("A scanner was constructed after a schema failure")

    monkeypatch.setattr(commands, "SecurityValidator", forbidden)
    results = run_validation(tmp_path, content_type="plugin", fail_fast=True)
    assert any(not result.passed for result in results)


def test_selected_skipped_checks_stay_explicit(tmp_path):
    manifest = tmp_path / ".claude-plugin/plugin.json"
    manifest.parent.mkdir()
    manifest.write_text('{"name":"demo"}')
    results = run_validation(tmp_path, content_type="plugin", checks="schema,quality", skip="quality")
    skipped = [result for result in results if result.metadata.get("skipped_by_user")]
    assert len(skipped) == 1
    assert "Quality" in skipped[0].validator_name
