# SPDX-License-Identifier: Apache-2.0
"""Parity reads the emitted private tasks, including file modes."""

import json
from dataclasses import replace

import pytest

from skillevaluator.tier3 import loose_parts
from skillevaluator.tier3.plugin_native import NativePluginSource


def source(root):
    (root / ".claude-plugin").mkdir(parents=True)
    manifest = {"name": "demo", "description": "A fixture plugin"}
    (root / ".claude-plugin/plugin.json").write_text(json.dumps(manifest))
    member = root / "skills/a"
    member.mkdir(parents=True)
    (member / "SKILL.md").write_text("---\nname: a\ndescription: A skill\n---\nUse a.\n")
    (root / "run.js").write_text("console.log('ok');\n")
    (root / "run.js").chmod(0o755)
    return NativePluginSource(
        "demo",
        "A fixture plugin",
        True,
        root,
        manifest,
        ".claude-plugin/plugin.json",
        member_skills=(member,),
        rules=(("policy", "# Policy\nUse the skill.\n"),),
        plugin_file_mcp_servers=({"name": "tool", "command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/run.js"]},),
    )


def test_missing_expected_arm_sets_cannot_claim_equality():
    result = loose_parts.compare_content_hashes({"with_skill": {"skill:a": "same"}})
    assert result["equal_across_arms"] is not True


def test_mode_only_script_change_is_a_parity_mismatch(tmp_path):
    parts = loose_parts.build_loose_parts(source(tmp_path / "plugin"))
    assert parts.error is None
    tasks = {}
    for arm in ("with_skill", "sum_of_parts", "loose_parts"):
        task = tmp_path / arm
        parts.stage_arm(arm, task / "environment")
        tasks[arm] = [task]
    assert parts.content_report_for_tasks(tasks)["equal_across_arms"] is True
    script = tasks["loose_parts"][0] / "environment/skilleval" / loose_parts.LOOSE_BUNDLE_DIR / "root/run.js"
    script.chmod(0o644)
    result = parts.content_report_for_tasks(tasks)
    assert result["equal_across_arms"] is False
    assert "script:run.js" in result["mismatches"]


@pytest.mark.parametrize("kind,mode,suffix", [("mcp", 0o400, ""), ("skill", 0o700, ""), ("skill", 0o700, "nested")])
def test_structured_config_and_directory_permissions_are_part_of_capability_parity(tmp_path, kind, mode, suffix):
    prepared = source(tmp_path / "plugin")
    nested = prepared.plugin_root / "skills/a/nested"
    nested.mkdir()
    (nested / "guide.md").write_text("Nested fixture guidance")
    parts = loose_parts.build_loose_parts(prepared)
    tasks = {}
    for arm in ("with_skill", "sum_of_parts", "loose_parts"):
        task = tmp_path / arm
        parts.stage_arm(arm, task / "environment")
        tasks[arm] = [task]
    assert parts.content_report_for_tasks(tasks)["equal_across_arms"] is True
    placement = next(item for item in parts._loose if item.kind == kind)
    (tasks["loose_parts"][0] / "environment" / placement.path / suffix).chmod(mode)
    result = parts.content_report_for_tasks(tasks)
    assert result["equal_across_arms"] is False
    assert placement.id in result["mismatches"]


def test_hook_json_permissions_are_part_of_capability_parity(tmp_path):
    from tests.tier3.private_loose_fixtures import make_plugin, native_source

    parts = loose_parts.build_loose_parts(
        native_source(make_plugin(tmp_path / "plugins"), tmp_path / "stage").native_source
    )
    tasks = {}
    for arm in ("with_skill", "sum_of_parts", "loose_parts"):
        task = tmp_path / arm
        parts.stage_arm(arm, task / "environment")
        tasks[arm] = [task]
    assert parts.content_report_for_tasks(tasks)["equal_across_arms"] is True
    placement = next(item for item in parts._loose if item.kind == "hook")
    (tasks["loose_parts"][0] / "environment" / placement.path).chmod(0o400)
    result = parts.content_report_for_tasks(tasks)
    assert result["equal_across_arms"] is False
    assert placement.id in result["mismatches"]


def test_empty_emitted_task_set_stays_unknown(tmp_path):
    parts = loose_parts.build_loose_parts(source(tmp_path / "plugin"))
    assert parts.content_report_for_tasks({})["equal_across_arms"] is not True


def test_census_normalization_cannot_hide_changed_emitted_frontmatter(tmp_path):
    from tests.tier3.private_loose_fixtures import make_plugin, native_source, write

    plugin = make_plugin(tmp_path / "plugins")
    write(
        plugin / "skills/version-bump/SKILL.md",
        "---\nname: version-bump\ndescription: Bump.\nhooks:\n  PreToolUse:\n    - matcher: Bash\n      hooks:\n        - type: command\n          command: echo bump\n---\nBump.\n",
    )
    parts = loose_parts.build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    tasks = {}
    for arm in ("with_skill", "sum_of_parts", "loose_parts"):
        task = tmp_path / arm
        parts.stage_arm(arm, task / "environment")
        tasks[arm] = [task]
    assert parts.content_report_for_tasks(tasks)["equal_across_arms"] is True
    placement = next(item for item in parts._full if item.id == "skill:version-bump")
    staged = tasks["with_skill"][0] / "environment" / placement.overrides[0][0]
    staged.write_text(staged.read_text() + "\nChanged runtime instructions.\n")
    result = parts.content_report_for_tasks(tasks)
    assert result["equal_across_arms"] is not True


def test_unexpected_capability_root_is_not_normalized_as_equal(tmp_path):
    parts = loose_parts.build_loose_parts(source(tmp_path / "plugin"))
    tasks = {}
    for arm in ("with_skill", "sum_of_parts", "loose_parts"):
        task = tmp_path / arm
        parts.stage_arm(arm, task / "environment")
        tasks[arm] = [task]
    placement = next(item for item in parts._loose if item.kind == "mcp")
    config = tasks["loose_parts"][0] / "environment" / placement.path
    payload = json.loads(config.read_text())
    payload["mcpServers"][placement.key]["args"] = ["/unexpected/root/run.js"]
    config.write_text(json.dumps(payload))
    result = parts.content_report_for_tasks(tasks)
    assert result["equal_across_arms"] is False
    assert "mcp:tool" in result["mismatches"]


def test_whole_root_normalization_never_hides_extra_emitted_metadata_content(tmp_path):
    prepared = source(tmp_path / "plugin")
    (prepared.plugin_root / ".claude-plugin/authored.txt").write_text("Not staged by either arm")
    prepared = replace(
        prepared,
        plugin_file_mcp_servers=(
            {"name": "tool", "command": "uv", "args": ["run", "--directory", "${CLAUDE_PLUGIN_ROOT}", "srv"]},
        ),
    )
    parts = loose_parts.build_loose_parts(prepared)
    tasks = {}
    for arm in ("with_skill", "sum_of_parts", "loose_parts"):
        task = tmp_path / arm
        parts.stage_arm(arm, task / "environment")
        tasks[arm] = [task]
    assert parts.content_report_for_tasks(tasks)["equal_across_arms"] is True
    placement = next(item for item in parts._full if item.id == "script:.")
    metadata = tasks["with_skill"][0] / "environment" / placement.path / ".claude-plugin"
    assert {path.name for path in metadata.iterdir()} == {"plugin.json"}
    loose_placement = next(item for item in parts._loose if item.id == "script:.")
    assert not (tasks["loose_parts"][0] / "environment" / loose_placement.path / ".claude-plugin").exists()
    (metadata / "injected.txt").write_text("Unexpected emitted capability")
    result = parts.content_report_for_tasks(tasks)
    assert result["equal_across_arms"] is False
    assert "script:." in result["mismatches"]


def test_piece_missing_from_every_emitted_arm_cannot_disappear_from_parity(tmp_path):
    parts = loose_parts.build_loose_parts(source(tmp_path / "plugin"))
    tasks = {}
    for arm in ("with_skill", "sum_of_parts", "loose_parts"):
        task = tmp_path / arm
        parts.stage_arm(arm, task / "environment")
        tasks[arm] = [task]
        placement = next(item for item in parts._placements(arm) if item.id == "skill:a")
        (task / "environment" / placement.path / "SKILL.md").unlink()
    result = parts.content_report_for_tasks(tasks)
    assert result["equal_across_arms"] is not True


def test_parity_rejects_symlink_content_even_when_all_arms_match(tmp_path):
    parts = loose_parts.build_loose_parts(source(tmp_path / "plugin"))
    tasks = {}
    target = tmp_path / "outside.md"
    target.write_text("Unverified external instruction")
    for arm in ("with_skill", "sum_of_parts", "loose_parts"):
        task = tmp_path / arm
        parts.stage_arm(arm, task / "environment")
        tasks[arm] = [task]
        placement = next(item for item in parts._placements(arm) if item.id == "skill:a")
        staged = task / "environment" / placement.path / "SKILL.md"
        staged.unlink()
        staged.symlink_to(target)
    # Direct secure reads refuse redirects; the private plan catches this
    # ValueError and records unavailable parity rather than an equal claim.
    with pytest.raises(ValueError, match="symlink"):
        parts.content_report_for_tasks(tasks)


def test_same_emitted_task_is_not_read_twice_in_one_parity_check(tmp_path, monkeypatch):
    parts = loose_parts.build_loose_parts(source(tmp_path / "plugin"))
    tasks = {}
    for arm in ("with_skill", "sum_of_parts", "loose_parts"):
        task = tmp_path / arm
        parts.stage_arm(arm, task / "environment")
        tasks[arm] = [task]
    calls = []
    original = loose_parts.SecureRoot.read_bytes

    def read(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(loose_parts.SecureRoot, "read_bytes", read)
    assert parts.content_report_for_tasks(tasks)["equal_across_arms"] is True
    first = len(calls)
    calls.clear()
    tasks["with_skill"] *= 2
    assert parts.content_report_for_tasks(tasks)["equal_across_arms"] is True
    assert len(calls) == first


@pytest.mark.parametrize("limit,value", [("MAX_PARITY_BYTES", 1), ("MAX_PARITY_FILES", 0)])
def test_parity_read_budget_rejects_before_an_extra_secure_read(tmp_path, monkeypatch, limit, value):
    target = tmp_path / "content.txt"
    target.write_bytes(b"bounded")
    monkeypatch.setattr(loose_parts, limit, value)
    calls = []
    monkeypatch.setattr(loose_parts.SecureRoot, "read_bytes", lambda *_args, **_kwargs: calls.append(True))
    with pytest.raises(ValueError, match="budget"):
        loose_parts._ParityReads().read(target)
    assert calls == []


@pytest.mark.parametrize(
    "route,expected",
    [
        ("PrivateClaudeCode", "LoosePartsClaudeCode"),
        ("PrivateContainerClaudeCode", "LoosePartsContainerClaudeCode"),
        ("PrivateNvidiaBuildClaudeCode", "LoosePartsNvidiaBuildClaudeCode"),
        ("NativeNvidiaBuildClaudeCode", "LoosePartsNvidiaBuildClaudeCode"),
    ],
)
def test_private_and_native_route_classes_keep_the_same_loose_backend(route, expected):
    result = loose_parts.loose_parts_agent_import_path(
        "claude-code", "skillevaluator.tier3.harbor.native_agents:" + route
    )
    assert result.endswith(":" + expected)
    from skillevaluator.tier3.harbor.local_agents import SetupNetworkErrorAgent
    from skillevaluator.tier3.harbor.loose_agents import _LoosePartsClaudeCodeMixin

    assert issubclass(_LoosePartsClaudeCodeMixin, SetupNetworkErrorAgent)
