# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The loose-parts arm on native Claude Code: staging, content hashes, and load evidence.

The loose arm stages the same skills, rules, MCP servers, subagents, commands
and hooks that Claude Code can load standalone, as plain user-level pieces
under ``$CLAUDE_CONFIG_DIR`` (and ``--mcp-config``), so that full plugin minus
loose parts measures only the plugin's glue. Every test uses fake plugin trees
and fake container layouts. Nothing starts Harbor, Docker, or an agent.
"""

from __future__ import annotations

import dataclasses
import json
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _loose_parts_fixtures import (
    AGENT_TEXT,
    COMMAND_TEXT,
    GUARD_SCRIPT,
    HOOKS,
    SERVER_SCRIPT,
    make_plugin,
    make_skills_only_plugin,
    native_source,
    write,
)

from skillevaluator.tier3.harbor.native_staging import (
    NativeTaskStaging,
    build_native_task_staging,
    stage_native_bundle,
)
from skillevaluator.tier3.loose_parts import (
    LOOSE_BUNDLE_DIR,
    LOOSE_DATA,
    LOOSE_MCP_CONFIG,
    LOOSE_PARTS_ARM,
    LOOSE_PARTS_LABEL,
    LOOSE_ROOT,
    LoosePartsClaudeCodeAdapter,
    LoosePartsUnsupported,
    build_loose_parts,
    compare_content_hashes,
    load_failures_by_case,
    loose_parts_agent_import_path,
    loose_parts_load_evidence,
    loose_parts_task_staging,
)
from skillevaluator.tier3.plugin_native import (
    CLAUDE_PLUGIN_ROOT_VAR,
    HOOK_CENSUS_SCRIPT,
    ClaudeCodeAdapter,
    PluginLoadError,
    normalize_census,
)

ROOT_DIR = f"skilleval/{LOOSE_BUNDLE_DIR}/root"


@pytest.fixture
def package(tmp_path: Path):
    return native_source(make_plugin(tmp_path / "plugins"), tmp_path / "stage")


@pytest.fixture
def source(package):
    return package.native_source


@pytest.fixture
def loose(source):
    parts = build_loose_parts(source)
    assert parts.error is None
    return parts


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _unwrap(handler: dict[str, Any]) -> str:
    """The original command a census-wrapped shell-form hook handler runs."""
    words = shlex.split(handler["command"])
    assert words[:2] == ["/bin/sh", HOOK_CENSUS_SCRIPT] and words[4] == "--", words
    return words[5]


def _hook_ids(config: dict[str, Any]) -> list[str]:
    return [
        shlex.split(handler["command"])[2]
        for groups in config["hooks"].values()
        for group in groups
        for handler in group["hooks"]
    ]


# --------------------------------------------------------------------------- #
# Names                                                                        #
# --------------------------------------------------------------------------- #
def test_arm_names_are_the_final_names() -> None:
    assert LOOSE_PARTS_ARM == "loose_parts"
    assert LOOSE_PARTS_LABEL == "Loose parts"
    assert LOOSE_ROOT == "/skilleval/native/claude-code/loose/root"
    assert LOOSE_MCP_CONFIG == "/skilleval/native/claude-code/loose/mcp.json"


# --------------------------------------------------------------------------- #
# What the loose bundle stages                                                 #
# --------------------------------------------------------------------------- #
def test_loose_bundle_stages_each_part_where_claude_code_loads_it_standalone(loose) -> None:
    bundle = loose.bundle
    assert bundle is not None
    # No plugin copy, no manifest, no --plugin-dir layout.
    assert bundle.plugin_tree is None
    assert not any("plugin.json" in rel or "/plugin/" in rel for rel in bundle.generated)
    assert sorted(bundle.generated) == [
        f"{LOOSE_BUNDLE_DIR}/mcp.json",
        f"{LOOSE_BUNDLE_DIR}/settings.json",
        "native/claude-code/rules/release-files.md",
        "native/claude-code/rules/release-policy.md",
        "native/setup.sh",
    ]
    assert sorted(dest for _source, dest in bundle.files) == [
        f"{LOOSE_BUNDLE_DIR}/agents/release-reviewer.md",
        f"{LOOSE_BUNDLE_DIR}/commands/release-check.md",
        f"{LOOSE_BUNDLE_DIR}/root/hooks/guard-push.sh",
        f"{LOOSE_BUNDLE_DIR}/root/servers/reltools.js",
    ]
    # Member skills go through the task skills projection, as in the skills-only arm.
    assert bundle.trees == []
    setup = bundle.generated["native/setup.sh"]
    assert '"$CLAUDE_CONFIG_DIR"/agents/release-reviewer.md' in setup
    assert '"$CLAUDE_CONFIG_DIR"/commands/release-check.md' in setup
    assert '"$CLAUDE_CONFIG_DIR"/settings.json' in setup
    assert f"mkdir -p {LOOSE_DATA}" in setup
    assert '"$CLAUDE_CONFIG_DIR/rules/skilleval-shipit/release-policy.md"' in setup
    assert "--plugin-dir" not in setup
    assert bundle.skill_aliases == []
    # The collector must not read this arm with the plugin-arm init check.
    assert bundle.harness["kind"] == "claude-code-loose-init"


def test_rules_are_byte_identical_to_native_mode_and_on_request_rules_stay_out_of_both(loose, source) -> None:
    full = ClaudeCodeAdapter().build(source)
    for rel in ("native/claude-code/rules/release-policy.md", "native/claude-code/rules/release-files.md"):
        assert loose.bundle.generated[rel] == full.generated[rel]
    # The glob-scoped rule keeps its patterns as Claude Code paths frontmatter.
    assert loose.bundle.generated["native/claude-code/rules/release-files.md"].startswith(
        '---\npaths:\n  - "RELEASE*.md"\n  - "CHANGELOG.md"\n---\n'
    )
    assert loose.rules == ("release-files.md", "release-policy.md")
    assert not any("semver-help" in rel for rel in (*loose.bundle.generated, *full.generated))
    not_staged = {piece.id: piece.reason for piece in loose.not_staged}
    assert "agent-requested" in not_staged["rule:semver-help.mdc"]


def test_agent_command_and_script_files_are_byte_and_mode_identical(tmp_path: Path, loose, source) -> None:
    env_dir = tmp_path / "loose" / "environment"
    loose.stage_arm(LOOSE_PARTS_ARM, env_dir)
    staged = env_dir / "skilleval" / Path(LOOSE_BUNDLE_DIR)
    plugin = source.plugin_root
    pairs = {
        "agents/release-reviewer.md": "agents/release-reviewer.md",
        "commands/release-check.md": "commands/release-check.md",
        "root/hooks/guard-push.sh": "hooks/guard-push.sh",
        "root/servers/reltools.js": "servers/reltools.js",
    }
    for staged_rel, source_rel in pairs.items():
        assert (staged / staged_rel).read_bytes() == (plugin / source_rel).read_bytes(), staged_rel
        assert stat.S_IMODE((staged / staged_rel).stat().st_mode) == stat.S_IMODE(
            (plugin / source_rel).stat().st_mode
        ), staged_rel
    assert (staged / "agents/release-reviewer.md").read_text(encoding="utf-8") == AGENT_TEXT
    assert (staged / "commands/release-check.md").read_text(encoding="utf-8") == COMMAND_TEXT
    assert (staged / "root/hooks/guard-push.sh").read_text(encoding="utf-8") == GUARD_SCRIPT
    assert (staged / "root/servers/reltools.js").read_text(encoding="utf-8") == SERVER_SCRIPT
    # Plain skills, as the skills-only arm stages them (the skill's evals/ never reaches the agent).
    skills = env_dir / "skills"
    assert sorted(path.name for path in skills.iterdir()) == ["changelog-collect", "version-bump"]
    assert (skills / "version-bump" / "scripts" / "bump.sh").read_bytes() == (
        plugin / "skills" / "version-bump" / "scripts" / "bump.sh"
    ).read_bytes()
    assert not (skills / "version-bump" / "evals").exists()


def test_hooks_go_to_user_settings_with_only_the_root_variable_substituted(loose, source) -> None:
    full = json.loads(ClaudeCodeAdapter().build(source).generated["native/claude-code/plugin/hooks/hooks.json"])
    settings = json.loads(loose.bundle.generated[f"{LOOSE_BUNDLE_DIR}/settings.json"])
    # A user settings.json holds the hooks and nothing else (no plugin hooks.json description).
    assert list(settings) == ["hooks"]
    # The same census ids as the plugin arm, so hook runs join the same Tier 1 rows.
    assert _hook_ids(settings) == _hook_ids(full)
    loose_commands = [_unwrap(h) for groups in settings["hooks"].values() for g in groups for h in g["hooks"]]
    source_commands = [h["command"] for groups in HOOKS["hooks"].values() for g in groups for h in g["hooks"]]
    assert loose_commands == [command.replace(CLAUDE_PLUGIN_ROOT_VAR, LOOSE_ROOT) for command in source_commands]
    assert loose_commands[0] == f'sh "{LOOSE_ROOT}/hooks/guard-push.sh"'
    # Matchers and events are unchanged.
    assert {event: [g["matcher"] for g in groups] for event, groups in settings["hooks"].items()} == {
        event: [g["matcher"] for g in groups] for event, groups in full["hooks"].items()
    }


def test_mcp_config_is_identical_except_the_root_variable(loose, source) -> None:
    full = json.loads(ClaudeCodeAdapter().build(source).generated["native/claude-code/plugin/.mcp.json"])
    staged = json.loads(loose.bundle.generated[f"{LOOSE_BUNDLE_DIR}/mcp.json"])
    assert list(staged) == ["mcpServers"]
    assert staged["mcpServers"]["reltools"] == {
        "type": "stdio",
        "command": "node",
        "args": [f"{LOOSE_ROOT}/servers/reltools.js", "--strict"],
        "env": {"RELTOOLS_MODE": "strict", "RELTOOLS_TOKEN": "${RELTOOLS_TOKEN}"},
    }
    expected = json.loads(json.dumps(full).replace(CLAUDE_PLUGIN_ROOT_VAR, LOOSE_ROOT))
    assert staged == expected
    assert loose.mcp_servers == ("reltools",)


def test_nothing_else_is_staged(tmp_path: Path, loose) -> None:
    env_dir = tmp_path / "loose" / "environment"
    loose.stage_arm(LOOSE_PARTS_ARM, env_dir)
    assert sorted(_tree(env_dir / "skilleval")) == [
        "hook_census.sh",
        "native/claude-code/loose/agents/release-reviewer.md",
        "native/claude-code/loose/commands/release-check.md",
        "native/claude-code/loose/mcp.json",
        "native/claude-code/loose/root/hooks/guard-push.sh",
        "native/claude-code/loose/root/servers/reltools.js",
        "native/claude-code/loose/settings.json",
        "native/claude-code/rules/release-files.md",
        "native/claude-code/rules/release-policy.md",
        "native/setup.sh",
    ]
    assert sorted(_tree(env_dir / "skills")) == [
        "changelog-collect/SKILL.md",
        "version-bump/SKILL.md",
        "version-bump/scripts/bump.sh",
    ]


def test_dropped_as_glue_lists_exactly_what_cannot_load_standalone(loose) -> None:
    dropped = {piece.id: piece.reason for piece in loose.dropped_as_glue}
    assert sorted(dropped) == [
        "bin:bin/",
        "lsp:go",
        "manifest:.claude-plugin/plugin.json",
        "output_style:terse",
        "settings:agent",
        "settings:subagentStatusLine",
    ]
    assert "PATH" in dropped["bin:bin/"]
    assert ".lsp.json" in dropped["lsp:go"]
    assert "namespac" in dropped["manifest:.claude-plugin/plugin.json"]
    assert all(reason.strip() for reason in dropped.values())
    # Every staged piece has a sha256 and the plugin-relative file it came from.
    pieces = {piece.id: piece for piece in loose.pieces}
    assert sorted(pieces) == [
        "agent:release-reviewer",
        "command:release-check",
        "hook:hooks/hooks.json",
        "mcp:reltools",
        "rule:release-files.md",
        "rule:release-policy.md",
        "script:hooks/guard-push.sh",
        "script:servers/reltools.js",
        "skill:changelog-collect",
        "skill:version-bump",
    ]
    assert pieces["agent:release-reviewer"].source == "agents/release-reviewer.md"
    assert all(len(piece.sha256) == 64 for piece in pieces.values())


def test_user_config_is_glue_too(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    manifest = json.loads((plugin / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    manifest["userConfig"] = {"region": {"type": "string", "title": "Region", "default": "us"}}
    write(plugin / ".claude-plugin" / "plugin.json", json.dumps(manifest))
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    assert "userConfig" in {piece.id for piece in parts.dropped_as_glue} | {
        piece.id.split(":", 1)[0] for piece in parts.dropped_as_glue
    }


# --------------------------------------------------------------------------- #
# Content hashes                                                               #
# --------------------------------------------------------------------------- #
def test_content_hashes_are_equal_across_arms_for_every_piece(loose) -> None:
    report = loose.content_report()
    hashes = report["content_hashes"]
    assert sorted(hashes) == ["loose_parts", "sum_of_parts", "with_skill"]
    assert report["equal_across_arms"] is True, report["mismatches"]
    assert report["mismatches"] == []
    for piece_id, digest in hashes["loose_parts"].items():
        assert hashes["with_skill"][piece_id] == digest, piece_id
    assert sorted(hashes["sum_of_parts"]) == ["skill:changelog-collect", "skill:version-bump"]
    for piece_id, digest in hashes["sum_of_parts"].items():
        assert hashes["loose_parts"][piece_id] == digest
    # Glue is the only content the full arm has beyond the loose arm.
    assert set(hashes["with_skill"]) == set(hashes["loose_parts"])
    assert loose.content_hashes_for_arm("none") == {}
    assert {piece.id: piece.sha256 for piece in loose.pieces} == hashes["loose_parts"]


@pytest.mark.parametrize(
    ("staged_rel", "piece_id"),
    [
        (f"skilleval/{LOOSE_BUNDLE_DIR}/agents/release-reviewer.md", "agent:release-reviewer"),
        (f"{ROOT_DIR}/hooks/guard-push.sh", "script:hooks/guard-push.sh"),
        ("skills/version-bump/scripts/bump.sh", "skill:version-bump"),
        (f"skilleval/{LOOSE_BUNDLE_DIR}/mcp.json", "mcp:reltools"),
    ],
)
def test_a_changed_byte_makes_that_piece_a_mismatch(tmp_path: Path, loose, staged_rel: str, piece_id: str) -> None:
    env_dir = tmp_path / "loose" / "environment"
    loose.stage_arm(LOOSE_PARTS_ARM, env_dir)
    target = env_dir / staged_rel
    data = bytearray(target.read_bytes())
    marker = data.index(b"e")
    data[marker] = ord("E")
    target.write_bytes(bytes(data))

    hashes = {
        "with_skill": loose.content_hashes_for_arm("with_skill"),
        "loose_parts": loose.hash_staged_arm(LOOSE_PARTS_ARM, env_dir),
        "sum_of_parts": loose.content_hashes_for_arm("sum_of_parts"),
    }
    report = compare_content_hashes(hashes)

    assert report["equal_across_arms"] is False
    assert report["mismatches"] == [piece_id]


def test_a_piece_missing_from_the_loose_arm_is_a_mismatch(loose) -> None:
    hashes = {
        "with_skill": loose.content_hashes_for_arm("with_skill"),
        "loose_parts": {
            k: v for k, v in loose.content_hashes_for_arm("loose_parts").items() if k != "command:release-check"
        },
        "sum_of_parts": loose.content_hashes_for_arm("sum_of_parts"),
    }
    report = compare_content_hashes(hashes)
    assert report["equal_across_arms"] is False
    assert report["mismatches"] == ["command:release-check"]


# --------------------------------------------------------------------------- #
# Errors and refusals                                                          #
# --------------------------------------------------------------------------- #
def test_a_plugin_with_nothing_loose_beyond_skills_gives_the_error(tmp_path: Path) -> None:
    package = native_source(make_skills_only_plugin(tmp_path / "plugins"), tmp_path / "stage")
    parts = build_loose_parts(package.native_source)
    assert parts.error is not None
    assert "loose parts equal the skills alone" in parts.error
    assert parts.bundle is None
    assert package.loose_parts_error() == parts.error
    assert package.provenance()["loose_parts_ready"] is False
    with pytest.raises(PluginLoadError, match="loose parts equal the skills alone"):
        loose_parts_task_staging("claude-code", package.native_source)


@pytest.mark.parametrize("agent", ["codex", "opencode", "hermes", "cursor"])
def test_other_harnesses_are_not_supported_with_an_explicit_reason(source, agent: str) -> None:
    parts = build_loose_parts(source, agent=agent)
    assert parts.error is not None and parts.error.startswith("not supported:")
    assert "not measured" in parts.error
    with pytest.raises(LoosePartsUnsupported, match="not supported"):
        loose_parts_task_staging(agent, source)


def test_wrapper_mode_is_not_supported_use_native(tmp_path: Path, source) -> None:
    parts = build_loose_parts(source, plugin_load="wrapper")
    assert parts.error is not None and parts.error.startswith("not supported: use native")
    with pytest.raises(LoosePartsUnsupported, match="not supported: use native"):
        loose_parts_task_staging("claude-code", source, plugin_load="wrapper")
    from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

    wrapper = prepare_plugin_eval_package(
        make_plugin(tmp_path / "wrapper-plugins"), stage_root=tmp_path / "wrapper-stage", plugin_load="wrapper"
    )
    assert wrapper.loose_parts_error() is not None
    assert wrapper.loose_parts_error().startswith("not supported: use native")
    assert wrapper.provenance()["loose_parts_ready"] is False


def test_ready_package_records_loose_parts_ready(package) -> None:
    assert package.loose_parts_error() is None
    assert package.provenance()["loose_parts_ready"] is True


def test_staging_is_deterministic(tmp_path: Path, source) -> None:
    first, second = build_loose_parts(source), build_loose_parts(source)
    assert first.bundle.generated == second.bundle.generated
    assert first.bundle.files == second.bundle.files
    assert first.pieces == second.pieces
    assert first.dropped_as_glue == second.dropped_as_glue
    one, two = tmp_path / "one" / "environment", tmp_path / "two" / "environment"
    first.stage_arm(LOOSE_PARTS_ARM, one)
    second.stage_arm(LOOSE_PARTS_ARM, two)
    assert _tree(one) == _tree(two)
    assert first.hash_staged_arm(LOOSE_PARTS_ARM, one) == second.hash_staged_arm(LOOSE_PARTS_ARM, two)


def test_a_reference_outside_the_plugin_root_is_never_copied(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    write(tmp_path / "plugins" / "outside.sh", "#!/bin/sh\necho secret\n")
    hooks = json.loads(json.dumps(HOOKS))
    hooks["hooks"]["PreToolUse"][0]["hooks"].append(
        {"type": "command", "command": 'sh "${CLAUDE_PLUGIN_ROOT}/../outside.sh"'}
    )
    write(plugin / "hooks" / "hooks.json", json.dumps(hooks))
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    assert parts.error is None
    env_dir = tmp_path / "loose" / "environment"
    parts.stage_arm(LOOSE_PARTS_ARM, env_dir)
    staged = _tree(env_dir)
    assert not any("outside" in rel for rel in staged)
    assert all(b"echo secret" not in data for data in staged.values())
    unresolved = {piece.id: piece.reason for piece in parts.not_staged}
    assert "script:../outside.sh" in unresolved
    assert "outside the plugin root" in unresolved["script:../outside.sh"]


def test_a_symlinked_referenced_script_is_refused(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    # The Tier 3 inventory already refuses a linked plugin file; swap one in after the snapshot.
    source = native_source(plugin, tmp_path / "stage").native_source
    target = tmp_path / "plugins" / "elsewhere.sh"
    write(target, "#!/bin/sh\necho elsewhere\n")
    (plugin / "hooks" / "guard-push.sh").unlink()
    (plugin / "hooks" / "guard-push.sh").symlink_to(target)
    with pytest.raises(ValueError, match=r"(?i)refus"):
        build_loose_parts(source)


def test_a_script_swapped_for_a_link_after_planning_is_never_copied(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    target = tmp_path / "plugins" / "elsewhere.sh"
    write(target, "#!/bin/sh\necho elsewhere\n")
    (plugin / "servers" / "reltools.js").unlink()
    (plugin / "servers" / "reltools.js").symlink_to(target)
    env_dir = tmp_path / "loose" / "environment"
    with pytest.raises(ValueError):
        parts.stage_arm(LOOSE_PARTS_ARM, env_dir)
    assert all(b"elsewhere" not in data for data in _tree(env_dir).values())


def test_a_bypass_hook_is_refused_like_the_plugin_arm(tmp_path: Path, source) -> None:
    # Native preparation already refuses such a plugin; the loose arm refuses the same snapshot on its own too.
    message = "Refusing to stage plugin hooks 'hooks/hooks.json' natively: permission-bypass flag"
    refused = dataclasses.replace(source, refusals=(("hook", "hooks/hooks.json", message),))
    with pytest.raises(PluginLoadError, match=r"(?i)bypass"):
        loose_parts_task_staging("claude-code", refused)
    parts = build_loose_parts(refused)
    assert parts.error == message and parts.bundle is None
    with pytest.raises(PluginLoadError, match=r"(?i)bypass"):
        build_native_task_staging("claude-code", LoosePartsClaudeCodeAdapter(), refused)


# --------------------------------------------------------------------------- #
# The seam the runner calls                                                    #
# --------------------------------------------------------------------------- #
def test_task_staging_has_the_shape_native_staging_consumes(tmp_path: Path, source) -> None:
    staging = loose_parts_task_staging("claude-code", source)
    assert isinstance(staging, NativeTaskStaging)
    assert staging.stage_member_skills is True
    assert staging.stage_wrapper_skill is False
    assert staging.plugin_mcp_via_task is False
    assert staging.workspace_skill_aliases() == []
    env_dir = tmp_path / "task" / "environment"
    env_dir.mkdir(parents=True)
    assert stage_native_bundle(env_dir, staging) == ["COPY skilleval/ /skilleval/"]
    assert (env_dir / "skilleval" / LOOSE_BUNDLE_DIR / "mcp.json").is_file()
    plan = staging.bundle.census_plan()
    declared = {(row["type"], row["name"]) for row in plan["declared"]}
    assert {("agent", "release-reviewer"), ("command", "release-check"), ("mcp", "reltools")} <= declared
    assert {("hook", "hooks/hooks.json"), ("rule", "release-policy.md"), ("skill", "version-bump")} <= declared


def test_the_real_emitter_stages_the_loose_arm_with_plain_member_skills(tmp_path: Path, package) -> None:
    import tomllib

    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks

    staging = loose_parts_task_staging("claude-code", package.native_source)
    tasks = generate_harbor_tasks(
        package.package_path,
        tmp_path / "out" / "claude-code" / "looseparts",
        with_skill=True,
        workspace_skill_paths=list(package.include_skills),
        workspace_mode="group",
        native_plugin=staging,
    )
    env_dir = tasks[0] / "environment"
    assert "COPY skilleval/ /skilleval/" in (env_dir / "Dockerfile").read_text(encoding="utf-8")
    assert sorted(path.name for path in (env_dir / "skills").iterdir()) == ["changelog-collect", "version-bump"]
    assert not (env_dir / "skilleval" / "native" / "claude-code" / "plugin").exists()
    task_toml = tomllib.loads((tasks[0] / "task.toml").read_text(encoding="utf-8"))
    assert "mcp_servers" not in task_toml.get("environment", {})
    entry = json.loads((tasks[0] / "tests" / "entry.json").read_text(encoding="utf-8"))
    assert "native_plugin_prefix" not in entry


def _run_setup(bundle: Path, root: Path, env: dict[str, str]) -> dict[str, Any]:
    container = root / "skilleval"
    shutil.copytree(bundle, container)
    logs = root / "logs" / "agent"
    logs.mkdir(parents=True)
    script = (container / "native" / "setup.sh").read_text(encoding="utf-8")
    script = script.replace("/skilleval/", f"{container}/").replace("/logs/agent", str(logs))
    (root / "setup.sh").write_text(script, encoding="utf-8")
    completed = subprocess.run(
        ["/bin/sh", str(root / "setup.sh")],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    census = normalize_census(json.loads((logs / "skilleval-load-census.json").read_text(encoding="utf-8")))
    assert census is not None
    return census


def test_setup_puts_every_part_under_claude_config_dir_and_lists_it(tmp_path: Path, loose) -> None:
    env_dir = tmp_path / "loose" / "environment"
    loose.stage_arm(LOOSE_PARTS_ARM, env_dir)
    root = tmp_path / "container"
    config_dir = root / "logs-agent-sessions"
    # Harbor copies the task skills into $CLAUDE_CONFIG_DIR/skills before the launch.
    shutil.copytree(env_dir / "skills", config_dir / "skills")
    census = _run_setup(env_dir / "skilleval", root, {"HOME": str(root / "home"), "CLAUDE_CONFIG_DIR": str(config_dir)})
    assert (config_dir / "agents" / "release-reviewer.md").read_text(encoding="utf-8") == AGENT_TEXT
    assert (config_dir / "commands" / "release-check.md").read_text(encoding="utf-8") == COMMAND_TEXT
    assert list(json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))) == ["hooks"]
    assert (config_dir / "rules" / "skilleval-shipit" / "release-policy.md").is_file()
    assert census["loaded"] == []
    listed = {(row["type"], row["name"]) for row in census["listed"]}
    assert listed == {
        ("skill", "changelog-collect"),
        ("skill", "version-bump"),
        ("rule", "release-policy.md"),
        ("rule", "release-files.md"),
        ("agent", "release-reviewer"),
        ("command", "release-check"),
        ("hook", "hooks/hooks.json"),
        ("mcp", "reltools"),
    }
    assert census["not_loaded"] == [] or all(row["type"] == "rule" for row in census["not_loaded"])


def test_setup_never_overwrites_an_existing_user_settings_file(tmp_path: Path, loose) -> None:
    env_dir = tmp_path / "loose" / "environment"
    loose.stage_arm(LOOSE_PARTS_ARM, env_dir)
    root = tmp_path / "container"
    config_dir = root / "sessions"
    write(config_dir / "settings.json", '{"model": "kept"}')
    census = _run_setup(env_dir / "skilleval", root, {"HOME": str(root / "home"), "CLAUDE_CONFIG_DIR": str(config_dir)})
    assert json.loads((config_dir / "settings.json").read_text(encoding="utf-8")) == {"model": "kept"}
    assert "settings.json" in census.get("setup_errors", "")
    assert ("hook", "hooks/hooks.json") in {(row["type"], row["name"]) for row in census["not_loaded"]}


def test_loose_agent_import_paths() -> None:
    assert loose_parts_agent_import_path("claude-code", None) == (
        "skillevaluator.tier3.harbor.loose_agents:LoosePartsClaudeCode"
    )
    assert loose_parts_agent_import_path(
        "claude-code", "skillevaluator.tier3.harbor.local_agents:SkillEvaluatorNvidiaBuildClaudeCode"
    ) == ("skillevaluator.tier3.harbor.loose_agents:LoosePartsNvidiaBuildClaudeCode")
    with pytest.raises(LoosePartsUnsupported, match="not supported"):
        loose_parts_agent_import_path("codex", None)


def test_info_block_has_the_final_json_field_names(loose) -> None:
    info = loose.info()
    assert {"content_hashes", "equal_across_arms", "mismatches", "dropped_as_glue", "load_evidence"} <= set(info)
    assert info["arm"] == "loose_parts"
    assert info["label"] == "Loose parts"
    assert info["dropped_as_glue"][0].keys() == {"id", "reason"}
    assert {"id", "kind", "source", "sha256"} == set(info["pieces"][0])
    assert info["load_evidence"] == {}


# --------------------------------------------------------------------------- #
# Load evidence (R4)                                                           #
# --------------------------------------------------------------------------- #
def _init(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "system",
        "subtype": "init",
        "skills": ["update-config", "changelog-collect", "version-bump"],
        "slash_commands": ["clear", "changelog-collect", "version-bump", "release-check"],
        "agents": ["Explore", "general-purpose", "release-reviewer"],
        "mcp_servers": [{"name": "reltools", "status": "connected"}],
        "tools": ["Bash", "Read", "mcp__reltools__list_changes", "mcp__reltools__compute_version"],
        "plugins": [],
    }
    event.update(overrides)
    return event


def _trajectory(*calls: tuple[str, dict[str, Any], Any]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [{"step_id": 1, "source": "user", "message": "Prepare the release."}]
    for index, (function, arguments, content) in enumerate(calls, start=1):
        result: dict[str, Any] = {"source_call_id": f"c{index}", "content": content}
        steps.append(
            {
                "step_id": index + 1,
                "source": "agent",
                "message": "",
                "tool_calls": [{"tool_call_id": f"c{index}", "function_name": function, "arguments": arguments}],
                "observation": {"results": [result]},
            }
        )
    return {"schema_version": "ATIF-v1.7", "agent": {"name": "claude-code", "version": "2.1"}, "steps": steps}


def test_init_marks_the_parts_it_reports_loaded(loose) -> None:
    evidence = loose_parts_load_evidence([_init()], None, loose)
    for piece_id in (
        "skill:changelog-collect",
        "skill:version-bump",
        "agent:release-reviewer",
        "command:release-check",
        "mcp:reltools",
    ):
        assert evidence[piece_id]["loaded"] is True, piece_id
        assert evidence[piece_id]["source"] == "system/init", piece_id
    assert evidence["mcp:reltools"]["tools"] == ["compute_version", "list_changes"]
    # Rules, hooks and scripts are not in Claude Code's startup report: never loaded, never a failure.
    for piece_id in ("rule:release-policy.md", "hook:hooks/hooks.json", "script:hooks/guard-push.sh"):
        assert evidence[piece_id]["loaded"] is None, piece_id
        assert evidence[piece_id]["source"] == "unverifiable", piece_id
    assert load_failures_by_case({LOOSE_PARTS_ARM: {"case-1": [evidence]}}) == {}


def test_a_pending_mcp_server_with_a_successful_call_is_loaded_by_call_evidence(loose) -> None:
    init = _init(mcp_servers=[{"name": "reltools", "status": "pending"}], tools=["Bash", "Read"])
    trajectory = _trajectory(("mcp__reltools__list_changes", {"since": "v1.2.3"}, "3 changes: RK-1, RK-2, RK-3"))

    evidence = loose_parts_load_evidence([init], trajectory, loose)

    assert evidence["mcp:reltools"]["loaded"] is True
    assert evidence["mcp:reltools"]["source"] == "mcp-call"
    assert "pending" in evidence["mcp:reltools"]["evidence"]
    assert "mcp__reltools__list_changes" in evidence["mcp:reltools"]["evidence"]


def test_a_pending_mcp_server_whose_calls_all_failed_is_not_loaded(loose) -> None:
    init = _init(mcp_servers=[{"name": "reltools", "status": "pending"}])
    trajectory = _trajectory(("mcp__reltools__list_changes", {}, "Error: No such tool available: list_changes"))

    evidence = loose_parts_load_evidence([init], trajectory, loose)

    assert evidence["mcp:reltools"]["loaded"] is False
    assert "pending" in evidence["mcp:reltools"]["evidence"]


def test_a_part_loaded_nowhere_is_a_load_failure_that_drops_the_case_from_every_arm(loose) -> None:
    init = _init(agents=["Explore"], mcp_servers=[{"name": "reltools", "status": "failed"}])
    evidence = loose_parts_load_evidence([init], _trajectory(), loose)
    assert evidence["agent:release-reviewer"]["loaded"] is False
    assert evidence["mcp:reltools"]["loaded"] is False
    assert "failed" in evidence["mcp:reltools"]["evidence"]
    ok = loose_parts_load_evidence([_init()], None, loose)

    failures = load_failures_by_case(
        {LOOSE_PARTS_ARM: {"case-1": [evidence], "case-2": [ok]}, "with_skill": {"case-1": [], "case-2": []}}
    )

    assert sorted(failures) == ["case-1"]
    rows = failures["case-1"]
    assert {(row["arm"], row["piece"]) for row in rows} == {
        (LOOSE_PARTS_ARM, "agent:release-reviewer"),
        (LOOSE_PARTS_ARM, "mcp:reltools"),
    }
    assert all(row["reason"] and row["attempt"] == 1 for row in rows)


def test_no_init_event_means_nothing_init_checks_is_loaded(loose) -> None:
    evidence = loose_parts_load_evidence([], None, loose)
    assert evidence["skill:version-bump"]["loaded"] is False
    assert "no system/init event" in evidence["skill:version-bump"]["evidence"]


def test_the_full_plugin_arm_is_checked_under_its_namespaced_names(loose) -> None:
    init = _init(
        skills=["shipit:changelog-collect", "shipit:version-bump"],
        slash_commands=["shipit:release-check", "shipit:changelog-collect", "shipit:version-bump"],
        agents=["Explore", "shipit:release-reviewer"],
        mcp_servers=[{"name": "plugin:shipit:reltools", "status": "pending"}],
        tools=["Bash"],
        plugins=[{"name": "shipit", "path": "/skilleval/native/claude-code/plugin", "source": "shipit@inline"}],
    )
    trajectory = _trajectory(("mcp__plugin_shipit_reltools__list_changes", {}, "3 changes"))

    evidence = loose_parts_load_evidence([init], trajectory, loose, arm="with_skill")

    assert {piece: row["loaded"] for piece, row in evidence.items() if row["loaded"] is not None} == {
        "skill:changelog-collect": True,
        "skill:version-bump": True,
        "agent:release-reviewer": True,
        "command:release-check": True,
        "mcp:reltools": True,
    }
    assert evidence["mcp:reltools"]["source"] == "mcp-call"
    # The bare loose spelling does not count as the plugin arm's component.
    bare = loose_parts_load_evidence([_init()], None, loose, arm="with_skill")
    assert bare["agent:release-reviewer"]["loaded"] is False


def test_the_skills_only_arm_checks_only_skills(loose) -> None:
    evidence = loose_parts_load_evidence([_init()], None, loose, arm="sum_of_parts")
    assert sorted(evidence) == ["skill:changelog-collect", "skill:version-bump"]


# --------------------------------------------------------------------------- #
# Edge layouts                                                                 #
# --------------------------------------------------------------------------- #
def test_frontmatter_hooks_census_wrapped_in_the_plugin_arm_still_hash_equal(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    write(
        plugin / "skills" / "version-bump" / "SKILL.md",
        "---\nname: version-bump\ndescription: Compute the next version.\nhooks:\n  PreToolUse:\n"
        "    - matcher: Bash\n      hooks:\n        - type: command\n          command: echo bump\n---\n\nBump.\n",
    )
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    report = parts.content_report()
    assert report["equal_across_arms"] is True, report["mismatches"]
    assert "skill:version-bump" in report["normalized"]
    # The loose arm stages the skill exactly as the skills-only arm does (its own frontmatter hooks, unwrapped).
    env_dir = tmp_path / "loose" / "environment"
    parts.stage_arm(LOOSE_PARTS_ARM, env_dir)
    assert (env_dir / "skills" / "version-bump" / "SKILL.md").read_bytes() == (
        plugin / "skills" / "version-bump" / "SKILL.md"
    ).read_bytes()


def test_a_referenced_directory_is_staged_with_the_plugin_copy_rules(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    mcp = {
        "mcpServers": {
            "reltools": {
                "command": "python3",
                "args": ["-m", "reltools"],
                "env": {"PYTHONPATH": "${CLAUDE_PLUGIN_ROOT}/servers"},
            }
        }
    }
    write(plugin / ".mcp.json", json.dumps(mcp))
    write(plugin / "servers" / "lib" / "util.py", "VALUE = 1\n")
    write(plugin / "servers" / ".env", "TOKEN=never-copied\n")
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    assert "script:servers" in {piece.id for piece in parts.pieces}
    env_dir = tmp_path / "loose" / "environment"
    parts.stage_arm(LOOSE_PARTS_ARM, env_dir)
    staged = sorted(_tree(env_dir / ROOT_DIR / "servers"))
    assert staged == ["README.md", "lib/util.py", "reltools.js"]
    report = parts.content_report()
    assert report["equal_across_arms"] is True, report["mismatches"]


def test_a_reference_to_the_plugin_root_itself_copies_the_plugin_tree_like_the_plugin_arm(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    mcp = {
        "mcpServers": {"reltools": {"command": "uv", "args": ["run", "--directory", "${CLAUDE_PLUGIN_ROOT}", "srv"]}}
    }
    write(plugin / ".mcp.json", json.dumps(mcp))
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    assert parts.bundle.plugin_tree == f"{LOOSE_BUNDLE_DIR}/root"
    staged = json.loads(parts.bundle.generated[f"{LOOSE_BUNDLE_DIR}/mcp.json"])
    assert staged["mcpServers"]["reltools"]["args"] == ["run", "--directory", LOOSE_ROOT, "srv"]
    env_dir = tmp_path / "loose" / "environment"
    parts.stage_arm(LOOSE_PARTS_ARM, env_dir)
    root = _tree(env_dir / ROOT_DIR)
    assert "servers/reltools.js" in root and ".claude-plugin/plugin.json" not in root
    report = parts.content_report()
    assert report["equal_across_arms"] is True, report["mismatches"]


def test_two_hook_sources_are_two_pieces(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    manifest = json.loads((plugin / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    manifest["hooks"] = {"Stop": [{"hooks": [{"type": "command", "command": "echo stop"}]}]}
    write(plugin / ".claude-plugin" / "plugin.json", json.dumps(manifest))
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    hooks = sorted(piece.id for piece in parts.pieces if piece.kind == "hook")
    assert len(hooks) == 2 and "hook:hooks/hooks.json" in hooks
    report = parts.content_report()
    assert report["equal_across_arms"] is True, report["mismatches"]
    assert set(report["content_hashes"]["with_skill"]) >= set(hooks)


def test_plugin_data_variable_points_at_a_fresh_loose_data_dir(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    hooks = json.loads(json.dumps(HOOKS))
    hooks["hooks"]["PostToolUse"][0]["hooks"][0]["command"] = 'echo wrote >> "${CLAUDE_PLUGIN_DATA}/writes.log"'
    write(plugin / "hooks" / "hooks.json", json.dumps(hooks))
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    settings = parts.bundle.generated[f"{LOOSE_BUNDLE_DIR}/settings.json"]
    assert "${CLAUDE_PLUGIN_DATA}" not in settings and LOOSE_DATA in settings
    assert f"mkdir -p {LOOSE_DATA}" in parts.bundle.generated["native/setup.sh"]
    assert parts.content_report()["equal_across_arms"] is True


def test_a_file_a_skills_frontmatter_hook_names_is_copied_to_the_loose_root(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    write(
        plugin / "skills" / "version-bump" / "SKILL.md",
        "---\nname: version-bump\ndescription: Compute the next version.\nhooks:\n  PostToolUse:\n"
        "    - matcher: Write\n      hooks:\n        - type: command\n"
        '          command: sh "${CLAUDE_PLUGIN_ROOT}/skills/version-bump/scripts/bump.sh"\n---\n\nBump.\n',
    )
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    assert "script:skills/version-bump/scripts/bump.sh" in {piece.id for piece in parts.pieces}
    env_dir = tmp_path / "loose" / "environment"
    parts.stage_arm(LOOSE_PARTS_ARM, env_dir)
    copied = env_dir / ROOT_DIR / "skills" / "version-bump" / "scripts" / "bump.sh"
    assert copied.read_bytes() == (plugin / "skills" / "version-bump" / "scripts" / "bump.sh").read_bytes()
    assert parts.content_report()["equal_across_arms"] is True


def test_hook_groups_without_a_command_handler_are_hashed_too(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    hooks = json.loads(json.dumps(HOOKS))
    hooks["hooks"]["Stop"] = [{"hooks": [{"type": "prompt", "prompt": "Check ${CLAUDE_PLUGIN_ROOT}/RELEASE.md"}]}]
    write(plugin / "hooks" / "hooks.json", json.dumps(hooks))
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    pieces = {piece.id: piece.sha256 for piece in parts.pieces}
    assert pieces["hook:(groups without a command handler)"]
    report = parts.content_report()
    assert report["equal_across_arms"] is True, report["mismatches"]
    assert "hook:(groups without a command handler)" in report["content_hashes"]["with_skill"]


def test_a_path_list_names_each_referenced_path(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    pythonpath = "${CLAUDE_PLUGIN_ROOT}/lib:${CLAUDE_PLUGIN_ROOT}/servers"
    write(
        plugin / ".mcp.json",
        json.dumps({"mcpServers": {"reltools": {"command": "python3", "env": {"PYTHONPATH": pythonpath}}}}),
    )
    write(plugin / "lib" / "helper.py", "X = 1\n")
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    assert {"script:lib", "script:servers"} <= {piece.id for piece in parts.pieces}
    staged = json.loads(parts.bundle.generated[f"{LOOSE_BUNDLE_DIR}/mcp.json"])
    assert staged["mcpServers"]["reltools"]["env"]["PYTHONPATH"] == f"{LOOSE_ROOT}/lib:{LOOSE_ROOT}/servers"


def test_an_unsafe_agent_file_name_cannot_inject_into_setup(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    (plugin / "agents" / "release-reviewer.md").rename(plugin / "agents" / 'rev"$(touch pwned)".md')
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    setup = parts.bundle.generated["native/setup.sh"]
    assert "$(touch" not in setup
    env_dir = tmp_path / "loose" / "environment"
    parts.stage_arm(LOOSE_PARTS_ARM, env_dir)
    root = tmp_path / "container"
    config_dir = root / "sessions"
    census = _run_setup(env_dir / "skilleval", root, {"HOME": str(root / "home"), "CLAUDE_CONFIG_DIR": str(config_dir)})
    assert not list(tmp_path.rglob("pwned"))
    staged = [path.name for path in (config_dir / "agents").iterdir()]
    assert len(staged) == 1 and "$" not in staged[0] and '"' not in staged[0]
    assert ("agent", "release-reviewer") in {(row["type"], row["name"]) for row in census["listed"]}


def test_the_bare_root_variable_is_left_for_the_shell_and_its_file_is_still_copied(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    hooks = json.loads(json.dumps(HOOKS))
    hooks["hooks"]["PreToolUse"][0]["hooks"][0]["command"] = "sh $CLAUDE_PLUGIN_ROOT/hooks/guard-push.sh"
    write(plugin / "hooks" / "hooks.json", json.dumps(hooks))
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    settings = json.loads(parts.bundle.generated[f"{LOOSE_BUNDLE_DIR}/settings.json"])
    assert _unwrap(settings["hooks"]["PreToolUse"][0]["hooks"][0]) == "sh $CLAUDE_PLUGIN_ROOT/hooks/guard-push.sh"
    assert "script:hooks/guard-push.sh" in {piece.id for piece in parts.pieces}
    assert parts.content_report()["equal_across_arms"] is True


def test_a_file_a_command_body_runs_is_copied(tmp_path: Path) -> None:
    plugin = make_plugin(tmp_path / "plugins")
    write(plugin / "commands" / "release-check.md", COMMAND_TEXT + "\n!`sh ${CLAUDE_PLUGIN_ROOT}/scripts/check.sh`\n")
    write(plugin / "scripts" / "check.sh", "#!/bin/sh\necho ok\n", mode=0o755)
    parts = build_loose_parts(native_source(plugin, tmp_path / "stage").native_source)
    assert "script:scripts/check.sh" in {piece.id for piece in parts.pieces}
    report = parts.content_report()
    assert report["equal_across_arms"] is True, report["mismatches"]


def test_a_plugin_file_server_named_like_a_task_server_is_refused_in_the_loose_arm(tmp_path: Path, package) -> None:
    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks

    write(
        package.package_path / "evals" / "environment" / "mcp_servers.toml",
        '[[mcp_servers]]\nname = "reltools"\ncommand = "/usr/bin/reltools"\n',
    )
    common = {"with_skill": True, "workspace_skill_paths": list(package.include_skills), "workspace_mode": "group"}
    with pytest.raises(ValueError, match="collide"):
        generate_harbor_tasks(
            package.package_path,
            tmp_path / "loose",
            native_plugin=loose_parts_task_staging("claude-code", package.native_source),
            **common,
        )
    # The plugin arm names the server plugin:shipit:reltools, so it keeps working as before.
    full = build_native_task_staging("claude-code", ClaudeCodeAdapter(), package.native_source)
    assert generate_harbor_tasks(package.package_path, tmp_path / "full", native_plugin=full, **common)


def test_readiness_follows_the_runs_load_plan(tmp_path: Path) -> None:
    from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

    plugin = make_plugin(tmp_path / "plugins")
    codex_only = prepare_plugin_eval_package(
        plugin, stage_root=tmp_path / "codex", plugin_load="native", agents="codex", env_mode="docker"
    )
    assert codex_only.native_source is not None
    assert codex_only.loose_parts_error().startswith("not supported: use native")
    assert codex_only.provenance()["loose_parts_ready"] is False
    claude = prepare_plugin_eval_package(
        plugin, stage_root=tmp_path / "claude", plugin_load="native", agents="claude-code", env_mode="docker"
    )
    assert claude.loose_parts_error() is None
    assert claude.loose_parts_error("codex").startswith("not supported:")


def test_info_pieces_use_the_hashes_of_the_report_they_sit_next_to(loose) -> None:
    report = loose.content_report()
    report["content_hashes"]["loose_parts"] = {
        **report["content_hashes"]["loose_parts"],
        "skill:version-bump": "f" * 64,
    }
    info = loose.info(content=report)
    pieces = {piece["id"]: piece["sha256"] for piece in info["pieces"]}
    assert pieces["skill:version-bump"] == "f" * 64


def test_setup_sets_the_config_dir_default_once(loose) -> None:
    setup = loose.bundle.generated["native/setup.sh"]
    assert setup.count(': "${CLAUDE_CONFIG_DIR:=$HOME/.claude}"') == 1
