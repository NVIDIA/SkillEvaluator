# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.cli import _plugin_lift_mode_for_evidence
from skillevaluator.constants import CONTENT_DEDUP_MAX_FILE_BYTES
from skillevaluator.plugin_manifest import locate_plugin_manifest
from skillevaluator.tier3 import plugin_eval as plugin_eval_module
from skillevaluator.tier3.plugin_eval import (
    PluginEvalPackage,
    prepare_plugin_eval_package,
    write_plugin_provenance,
)
from skillevaluator.utils.secure_fs import SecureRoot


def _skill(root: Path, name: str = "demo") -> Path:
    skill = root / "skills" / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Public test skill\n---\n# {name}\n\nUse this skill.\n",
        encoding="utf-8",
    )
    evals = skill / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(
        json.dumps({"skill_name": name, "evals": [{"id": "case-1", "prompt": "Do it", "expected_output": "Done"}]}),
        encoding="utf-8",
    )
    return skill


def test_contained_plugin_stages_member_skill_and_combined_dataset(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "public-plugin", "skills": "./skills"}), encoding="utf-8")
    member = _skill(plugin)

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert not package.skipped
    assert package.include_skills == (member.resolve(),)
    assert package.package_path is not None
    assert (package.package_path / "SKILL.md").is_file()
    assert not (package.package_path / "plugin-eval-metadata.json").exists()
    assert str(tmp_path) not in (package.package_path / "SKILL.md").read_text(encoding="utf-8")
    entries = json.loads((package.package_path / "evals" / "evals.json").read_text(encoding="utf-8"))
    assert entries[0]["id"] == "demo-case-1"
    assert entries[0]["plugin_eval_source_skill"] == "demo"


def test_plugin_integration_evidence_is_counted_from_dataset(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "public-plugin", "skills": "./skills"}), encoding="utf-8")
    _skill(plugin, "alpha")
    _skill(plugin, "beta")
    evals = plugin / "evals"
    evals.mkdir()
    evals.joinpath("evals.json").write_text(
        json.dumps(
            [
                {
                    "id": "composition",
                    "prompt": "Use both skills.",
                    "expected_skills": ["alpha", "beta"],
                    "cross_component": True,
                },
                {
                    "id": "single",
                    "prompt": "Use alpha.",
                    "expected_skills": ["alpha"],
                    "cross_component": False,
                },
            ]
        ),
        encoding="utf-8",
    )

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")

    assert package.dataset_case_count == 2
    assert package.cross_component_case_count == 1
    assert package.integration_evidence_error() is None
    assert package.provenance()["integration_evidence_ready"] is True


def test_both_lift_falls_back_without_composition_evidence() -> None:
    package = PluginEvalPackage(
        plugin_name="public-plugin",
        package_path=Path("/unused"),
        include_skills=(),
        unresolved_mcp_servers=(),
        runnable_mcp_servers=(),
        rule_refs=(),
        dataset_case_count=1,
        cross_component_case_count=0,
    )

    effective, reason = _plugin_lift_mode_for_evidence(package, "both")

    assert effective == "effectiveness"
    assert reason is not None
    assert "cross_component=true" in reason
    assert _plugin_lift_mode_for_evidence(package, "integration") == ("integration", reason)


def test_prepare_rejects_out_of_root_manifest_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text('{"name": "outside"}', encoding="utf-8")
    plugin = tmp_path / "plugin"
    manifest_dir = plugin / ".claude-plugin"
    manifest_dir.mkdir(parents=True)
    try:
        (manifest_dir / "plugin.json").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match=r"symlink|reparse"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")


def test_symlinked_standalone_plugin_directory_is_rejected(tmp_path: Path) -> None:
    real_plugin = tmp_path / "real-plugin"
    manifest = real_plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "linked-plugin", "skills": "./skills"}), encoding="utf-8")
    _skill(real_plugin)
    linked_plugin = tmp_path / "linked-plugin"
    try:
        linked_plugin.symlink_to(real_plugin, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match=r"symlink|junction|reparse"):
        locate_plugin_manifest(linked_plugin)
    with pytest.raises(ValueError, match=r"symlink|junction|reparse"):
        prepare_plugin_eval_package(linked_plugin, stage_root=tmp_path / "stage")


def test_remote_only_public_bundle_is_honestly_skipped(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "agent_plugin.yaml").write_text(
        """
name: remote-only
author: {email: dev@example.com}
skills:
  refs: [github::other/repository::skills::remote]
""",
        encoding="utf-8",
    )
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")
    assert package.skipped
    assert package.package_path is None
    assert package.unresolved_skill_refs == ("github::other/repository::skills::remote",)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/NVIDIA/SkillEvaluator.git",
        "git@github.com:NVIDIA/SkillEvaluator.git",
    ],
)
def test_remote_slug_matches_normalized_canonical_repo(url: str) -> None:
    assert plugin_eval_module._slug_from_remote_url(url) == "nvidia/skillevaluator"


def test_same_repo_public_ref_resolves_without_remote_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    plugin = repo / "plugins" / "bundle"
    plugin.mkdir(parents=True)
    member = _skill(repo, "local")
    (plugin / "agent_plugin.yaml").write_text(
        """
name: bundle
author: {email: dev@example.com}
skills:
  refs: [github::example/repo::skills::local]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(plugin_eval_module, "_local_repo_slug", lambda _root: "example/repo")
    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", repo_root=repo)
    assert len(package.include_skills) == 1
    assert package.include_skills[0].name == member.name
    assert package.include_skills[0] != member.resolve()
    assert package.include_skills[0].is_relative_to(tmp_path / "stage")
    assert (package.include_skills[0] / "SKILL.md").read_text(encoding="utf-8") == (
        member / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert package.unresolved_skill_refs == ()


def test_same_repo_public_ref_stays_unresolved_without_origin_slug(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    plugin = repo / "plugins" / "bundle"
    plugin.mkdir(parents=True)
    _skill(repo, "local")
    (plugin / "agent_plugin.yaml").write_text(
        """
name: bundle
author: {email: dev@example.com}
skills:
  refs: [github::example/repo::skills::local]
""",
        encoding="utf-8",
    )

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", repo_root=repo)

    assert package.skipped
    assert package.unresolved_skill_refs == ("github::example/repo::skills::local",)


def test_same_repo_rule_ref_is_loaded_from_private_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    plugin = repo / "plugins" / "bundle"
    plugin.mkdir(parents=True)
    rules = repo / "rules"
    rules.mkdir()
    (rules / "policy.md").write_text("Keep data local.\n", encoding="utf-8")
    evals = plugin / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text("[]", encoding="utf-8")
    (plugin / "agent_plugin.yaml").write_text(
        """
name: bundle
rules:
  refs: [github::example/repo::rules::policy.md]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(plugin_eval_module, "_local_repo_slug", lambda _root: "example/repo")

    package = prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", repo_root=repo)

    assert not package.skipped
    assert package.staged_rules == ("policy.md",)
    assert package.package_path is not None
    assert "Keep data local." in (package.package_path / "SKILL.md").read_text(encoding="utf-8")
    assert package.unresolved_rule_refs == ()


@pytest.mark.parametrize(
    ("kind", "name"),
    [("skills", "local"), ("rules", "policy.md")],
)
def test_canonical_ref_rejects_symlinked_content_root(
    kind: str,
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    plugin = repo / "plugins" / "bundle"
    plugin.mkdir(parents=True)
    outside = tmp_path / "outside" / kind
    outside.mkdir(parents=True)
    if kind == "skills":
        member = outside / name
        member.mkdir()
        (member / "SKILL.md").write_text("host-only skill\n", encoding="utf-8")
    else:
        (outside / name).write_text("host-only rule\n", encoding="utf-8")
    try:
        (repo / kind).symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    (plugin / "agent_plugin.yaml").write_text(
        f"""
name: bundle
author: {{email: dev@example.com}}
{kind}:
  refs: [github::example/repo::{kind}::{name}]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(plugin_eval_module, "_local_repo_slug", lambda _root: "example/repo")

    with pytest.raises(ValueError, match=r"symlink|reparse|unsafe"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", repo_root=repo)


def test_canonical_ref_rejects_prepopulated_snapshot_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    plugin = repo / "plugins" / "bundle"
    plugin.mkdir(parents=True)
    _skill(repo, "local")
    (plugin / "agent_plugin.yaml").write_text(
        """
name: bundle
author: {email: dev@example.com}
skills:
  refs: [github::example/repo::skills::local]
""",
        encoding="utf-8",
    )
    evals = plugin / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text("[]", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("POISON\n", encoding="utf-8")
    snapshot = tmp_path / "stage" / ".intra-repo-snapshots" / "skills" / "skills" / "local"
    snapshot.parent.mkdir(parents=True)
    try:
        snapshot.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    monkeypatch.setattr(plugin_eval_module, "_local_repo_slug", lambda _root: "example/repo")

    with pytest.raises(ValueError, match=r"symlink|reparse|unsafe"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage", repo_root=repo)


def test_contained_mcp_secret_is_rejected_before_toml_write(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "name": "unsafe",
                "mcpServers": {"server": {"command": "server", "env": {"API_KEY": "literal-secret"}}},
            }
        ),
        encoding="utf-8",
    )
    evals = plugin / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(
        json.dumps({"evals": [{"id": "case", "prompt": "Use server", "expected_output": "done"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"static (?:safety )?validation"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")


def test_contained_mcp_shell_command_is_rejected_before_execution(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"name": "unsafe", "mcpServers": {"server": {"command": "sh", "args": ["-c", "run"]}}}),
        encoding="utf-8",
    )
    evals = plugin / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="mcp_command_dangerous_form"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")


def test_prepare_rejects_nonstring_mcp_args_before_staging(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "agent_plugin.yaml").write_text(
        """
name: unsafe-args
description: Direct Tier 3 input must preserve scalar argument boundaries.
author: {email: dev@example.com}
mcp:
  - name: runner
    command: runner
    args:
      - [--unsafe]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"args\[0\].*string|args.*strings"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")


@pytest.mark.parametrize("field", ["name", "description"])
def test_prepare_rejects_oversized_manifest_text(field: str, tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "plugin", field: "x" * 20_000}), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{field}|character limit|bounded string"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")


def test_plugin_evals_symlink_escape_is_rejected(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "p", "mcpServers": {"x": {"command": "server"}}}), encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evals.json").write_text("[]", encoding="utf-8")
    try:
        (plugin / "evals").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match=r"linked directory|reparse"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")


@pytest.mark.parametrize("relative", [Path("evals/evals.json"), Path("eval/dataset.json")])
def test_member_dataset_symlink_is_rejected(relative: Path, tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "agent_plugin.yaml").write_text("name: p\n", encoding="utf-8")
    member = tmp_path / "member"
    member.mkdir()
    (member / "SKILL.md").write_text("---\nname: demo\ndescription: demo\n---\n", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text('[{"id": "host-only"}]', encoding="utf-8")
    dataset = member / relative
    dataset.parent.mkdir()
    try:
        dataset.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match=r"linked member dataset|symlink|reparse|unsafe"):
        prepare_plugin_eval_package(
            plugin,
            stage_root=tmp_path / "stage",
            include_skills=(member,),
        )


def test_plugin_provenance_does_not_follow_destination_symlink(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("unchanged", encoding="utf-8")
    try:
        (run_dir / "plugin_provenance.json").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    assert write_plugin_provenance(run_dir, {"plugin_name": "demo"}) is None
    assert outside.read_text(encoding="utf-8") == "unchanged"


def test_plugin_provenance_does_not_follow_run_directory_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    run_dir = tmp_path / "run"
    try:
        run_dir.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    assert write_plugin_provenance(run_dir, {"plugin_name": "demo"}) is None
    assert not (outside / "plugin_provenance.json").exists()


def test_symlinked_member_skill_outside_plugin_is_rejected(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "p", "skills": "./skills"}), encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("---\nname: outside\ndescription: outside\n---\n", encoding="utf-8")
    skills = plugin / "skills"
    skills.mkdir()
    try:
        (skills / "outside").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match=r"linked directory|reparse"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")


def test_contained_rule_replacement_after_discovery_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "p", "rules": "./rules"}), encoding="utf-8")
    rules = plugin / "rules"
    rules.mkdir()
    rule = rules / "policy.md"
    rule.write_text("Keep data local.\n", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("host-only content\n", encoding="utf-8")
    original_read = SecureRoot.read_file_text

    def replace_before_read(self: SecureRoot, file, max_bytes: int) -> str:
        rule.unlink()
        rule.symlink_to(outside)
        return original_read(self, file, max_bytes)

    monkeypatch.setattr(SecureRoot, "read_file_text", replace_before_read)

    with pytest.raises(ValueError, match=r"unsafe|changed|symlink"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")


def test_oversized_contained_rule_is_rejected(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin"
    manifest = plugin / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"name": "p", "rules": "./rules"}), encoding="utf-8")
    rules = plugin / "rules"
    rules.mkdir()
    (rules / "policy.md").write_bytes(b"x" * (CONTENT_DEDUP_MAX_FILE_BYTES + 1))

    with pytest.raises(ValueError, match=r"limit|unbounded|exceed"):
        prepare_plugin_eval_package(plugin, stage_root=tmp_path / "stage")
