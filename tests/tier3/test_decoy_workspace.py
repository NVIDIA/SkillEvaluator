# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen decoy selections preserve the paired workspace contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from skillevaluator.provider_config import ProviderConfig
from skillevaluator.publication_identity import publication_source_digest
from skillevaluator.tier3.evals_config import EvalsConfigError, load_evals_config
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks


def _skill(path: Path, *, skill_name: str | None = None) -> Path:
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {skill_name or path.name}\ndescription: Use this {path.name} workflow.\n---\n"
        f"# {path.name}\nFollow the {path.name} procedure.\n",
        encoding="utf-8",
    )
    return path


def _manifest(selected: list[Path]) -> dict:
    return {
        "selected": [
            {"path": path.name, "skill_name": path.name, "source_digest": publication_source_digest(path)}
            for path in selected
        ]
    }


def _config(skill: Path, workspace: str) -> None:
    (skill / "evals").mkdir(exist_ok=True)
    (skill / "evals" / "config.yml").write_text(
        "schema_version: 1\nskill_workspace:\n" + workspace,
        encoding="utf-8",
    )


@pytest.mark.parametrize("value", ["null", "[]", "{}", "3", "true", "''", "'  '"])
def test_decoy_manifest_config_requires_nonempty_path(tmp_path: Path, value: str) -> None:
    target = _skill(tmp_path / "target")
    _config(target, f"  mode: group\n  decoy_manifest: {value}\n")

    with pytest.raises(EvalsConfigError, match=r"skill_workspace\.decoy_manifest"):
        load_evals_config(target)


def test_decoy_config_defers_mode_validation_until_cli_override(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.tier3 import decoys

    target = _skill(tmp_path / "target")
    decoy = _skill(tmp_path / "decoy")
    _config(target, "  mode: isolated\n  decoy_manifest: frozen.json\n")
    config, _ = load_evals_config(target)
    seen: list[Path] = []

    def load(_target, path):
        seen.append(path)
        return [decoy], _manifest([decoy])

    monkeypatch.setattr(decoys, "load_decoy_manifest", load)
    with pytest.raises(ValueError, match=r"requires skill_workspace\.mode=group"):
        runner._decoy_workspace_skills(target, config["skill_workspace"], "isolated")
    assert seen == []

    manual, selected, manifest = runner._decoy_workspace_skills(target, config["skill_workspace"], "group")
    assert manual == []
    assert selected == [decoy]
    assert manifest is not None
    assert seen == [target.parent / "frozen.json"]


@pytest.mark.parametrize("collision", ["basename", "metadata", "reference"])
def test_decoy_names_cannot_be_silently_shadowed(tmp_path: Path, monkeypatch, collision: str) -> None:
    from skillevaluator.tier3 import decoys

    target = _skill(tmp_path / "target")
    decoy = _skill(tmp_path / "catalog" / "decoy")
    helper_name = "DECOY" if collision == "basename" else "helper"
    helper = _skill(tmp_path / "helpers" / helper_name, skill_name="decoy")
    references = tmp_path / "helpers" if collision == "reference" else None
    config = {"mode": "group", "decoy_manifest": "frozen.json"}
    if references is None:
        config["include"] = [str(helper)]
    monkeypatch.setattr(decoys, "load_decoy_manifest", lambda *_: ([decoy], _manifest([decoy])))

    with pytest.raises(ValueError, match="collides with another workspace skill name"):
        runner._decoy_workspace_skills(target, config, "group", reference_skills_dir=references)


def test_invalid_decoy_manifest_fails_before_provider_resolution(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.tier3 import decoys

    target = _skill(tmp_path / "target")
    _config(target, "  mode: group\n  decoy_manifest: frozen.json\n")

    def invalid(*_args):
        raise ValueError("Frozen decoy source changed")

    def unexpected():
        pytest.fail("Provider resolution must follow decoy validation")

    monkeypatch.setattr(decoys, "load_decoy_manifest", invalid)
    monkeypatch.setattr(runner, "resolve_llm_provider", unexpected)

    result = runner._run_harbor_eval_impl(target, ["codex"])

    assert result == {"error": ["Frozen decoy source changed"]}


def test_decoy_snapshot_rejects_source_changed_since_selection(tmp_path: Path) -> None:
    decoy = _skill(tmp_path / "decoy")
    manifest = _manifest([decoy])
    (decoy / "SKILL.md").write_text("changed", encoding="utf-8")

    with (
        pytest.raises(ValueError, match="changed before snapshot"),
        runner._private_decoy_workspace_snapshot([decoy], manifest),
    ):
        pytest.fail("Changed source must never be exposed")


def test_frozen_decoy_stages_identically_after_source_changes(tmp_path: Path) -> None:
    target = _skill(tmp_path / "repo" / "target")
    subprocess.run(["git", "init", "--quiet", str(target.parent)], check=True, capture_output=True)
    (target / "evals").mkdir()
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Use the target procedure."}]), encoding="utf-8"
    )
    decoy = _skill(tmp_path / "decoy")
    (decoy / "evals").mkdir()
    (decoy / "evals" / "oracle.txt").write_text("private evaluator data", encoding="utf-8")
    manifest = _manifest([decoy])
    expected = (decoy / "SKILL.md").read_bytes()

    with runner._private_decoy_workspace_snapshot([decoy], manifest) as frozen:
        (decoy / "SKILL.md").write_text("changed after snapshot", encoding="utf-8")
        tasks = [
            generate_harbor_tasks(
                target,
                tmp_path / arm,
                with_skill=with_skill,
                workspace_skill_paths=frozen,
                workspace_mode="group",
            )[0]
            for arm, with_skill in (("with-skill", True), ("baseline", False))
        ]
        for task in tasks:
            staged = task / "environment" / "skills" / decoy.name
            assert (staged / "SKILL.md").read_bytes() == expected
            assert not (staged / "evals").exists()
        assert (tasks[0] / "environment" / "skills" / target.name / "SKILL.md").exists()
        assert not (tasks[1] / "environment" / "skills" / target.name).exists()
        snapshot_root = frozen[0].parent
    assert not snapshot_root.exists()


def test_runner_reuses_frozen_selection_and_records_original_provenance(tmp_path: Path, monkeypatch) -> None:
    from skillevaluator.tier3 import decoys
    from skillevaluator.tier3.harbor import runtime_preflight

    target = _skill(tmp_path / "repo" / "target")
    subprocess.run(["git", "init", "--quiet", str(target.parent)], check=True, capture_output=True)
    _config(target, "  mode: group\n  decoy_manifest: frozen.json\n")
    decoy = _skill(tmp_path / "catalog" / "decoy")
    manifest = _manifest([decoy])
    expected = (decoy / "SKILL.md").read_bytes()
    emitted: list[tuple[bool, list[Path]]] = []
    provenance_sources: list[Path] = []
    monkeypatch.setattr(decoys, "load_decoy_manifest", lambda *_: ([decoy], manifest))
    monkeypatch.setattr(
        runner,
        "resolve_llm_provider",
        lambda: ProviderConfig(
            provider="nv_build",
            model="test-model",
            api_key="provider-key",
            base_url="https://provider.example/v1",
            litellm_model="openai/test-model",
        ),
    )
    monkeypatch.setattr(runner, "find_evals_file", lambda _: target / "evals" / "evals.json")
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_: [])
    monkeypatch.setattr(runner, "_run_agent_pair", lambda **_: [])
    monkeypatch.setattr(
        runner,
        "collect_harbor_results",
        lambda **_: {"execution_status": "succeeded", "execution_errors": [], "metrics": [], "agents": {}},
    )
    monkeypatch.setattr(runner, "render_agent_eval_html_report", lambda *_args, **_kw: tmp_path / "report.html")
    monkeypatch.setattr(
        runtime_preflight,
        "probe_model",
        lambda provider: runtime_preflight.ModelProbeResult(True, provider.provider, provider.model, "available"),
    )

    def emit(_skill, output, **kwargs):
        sources = kwargs["workspace_skill_paths"]
        assert (sources[0] / "SKILL.md").read_bytes() == expected
        emitted.append((kwargs["with_skill"], sources))
        (decoy / "SKILL.md").write_text("changed after first arm", encoding="utf-8")
        task = output / "case-001"
        task.mkdir(parents=True)
        return [task]

    def publication_inputs(_skill, _tasks, **kwargs):
        provenance_sources.extend(kwargs["workspace_skill_paths"])
        return False

    monkeypatch.setattr(runner, "generate_harbor_tasks", emit)
    monkeypatch.setattr(runner, "_runtime_context_is_bound_to_publication_target", publication_inputs)

    result = runner.run_harbor_eval(target, ["opencode"], output_dir=tmp_path / "results")

    assert "error" not in result
    assert [with_skill for with_skill, _ in emitted] == [True, False]
    assert emitted[0][1] == emitted[1][1]
    assert emitted[0][1][0] != decoy
    assert provenance_sources == [decoy]
    assert result["run_config"]["skill_workspace"]["decoy_manifest"] == manifest
