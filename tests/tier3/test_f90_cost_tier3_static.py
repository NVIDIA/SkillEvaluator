# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Tier 3 static context-cost estimate describes the run's own harness and load mode (check 25).

check-25 tier3-09: the provenance printed one estimate, with the note "Tier 3
wrapper: rules ... load on demand", for native runs too. Each test prepares a
small plugin the way ``tier3 evaluate-plugin`` does; nothing starts Harbor,
Docker, or a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package

# --------------------------------------------------------------------------- #
# M33 (Tier 3 half): the static estimate follows the run's load mode          #
# --------------------------------------------------------------------------- #
_FILES: dict[str, Any] = {
    ".claude-plugin/plugin.json": {"name": "demo", "version": "1.0.0", "description": "Demo plugin"},
    "skills/alpha/SKILL.md": "---\nname: alpha\ndescription: Demo skill alpha\n---\n# alpha\nUse it.\n",
    "rules/style.md": "Write short sentences and name the owner of every dataset. " * 20,
    "agents/reviewer.md": "---\nname: reviewer\ndescription: Reviews code\n---\nReview the code.\n",
    "evals/evals.json": [{"id": "c1", "prompt": "Use the plugin.", "expected_output": "Done."}],
}


def _provenance_cost(tmp_path: Path, plugin_load: str, agents: str) -> dict[str, Any]:
    root = tmp_path / "demo"
    for rel, content in _FILES.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    package = prepare_plugin_eval_package(
        root, stage_root=tmp_path / "stage", plugin_load=plugin_load, agents=agents, env_mode="docker"
    )
    return package.provenance()["context_cost"]


@pytest.mark.parametrize(
    ("plugin_load", "agents", "expected", "rule_always_on"),
    [
        ("native", "claude-code", ("claude-code", "native"), True),
        ("native", "codex", ("codex", "native"), True),
        ("wrapper", "codex", ("codex", "wrapper"), False),
        # No model for OpenCode: the plugin's own harness, with the run's load mode.
        ("wrapper", "opencode", ("claude-code", "wrapper"), False),
        ("native", "opencode", ("claude-code", "native"), True),
    ],
)
def test_m33_tier3_estimate_describes_the_load_mode_of_the_run(
    tmp_path: Path, plugin_load: str, agents: str, expected: tuple[str, str], rule_always_on: bool
) -> None:
    """tier3-09: the provenance said "Tier 3 wrapper: rules load on demand" in native runs too."""
    cost = _provenance_cost(tmp_path, plugin_load, agents)
    rule = next(row for row in cost["by_component"] if row["type"] == "rule")

    assert (cost["harness"], cost["load_mode"]) == expected
    assert (rule["always_on_tokens"] > 0) is rule_always_on
    wrapper_note = any("Tier 3 wrapper" in note for note in cost["notes"])
    assert wrapper_note is (expected[1] == "wrapper")


def test_l29_external_member_skill_counts_cjk_and_hidden_flags(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    for rel, content in _FILES.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    chinese = "把逗号分隔的数据文件整理成带类型的结构化数据" * 4
    external = tmp_path / "outside" / "cjk-tidy"
    external.mkdir(parents=True)
    (external / "SKILL.md").write_text(f"---\nname: cjk-tidy\ndescription: {chinese}\n---\nBody\n", encoding="utf-8")
    hidden = tmp_path / "outside" / "manual-only"
    hidden.mkdir(parents=True)
    (hidden / "SKILL.md").write_text(
        "---\nname: manual-only\ndescription: Only when asked\ndisable-model-invocation: true\n---\nBody\n",
        encoding="utf-8",
    )

    package = prepare_plugin_eval_package(
        root, stage_root=tmp_path / "stage", include_skills=(external, hidden), agents="claude-code", env_mode="docker"
    )
    rows = {row["name"]: row for row in package.provenance()["context_cost"]["by_component"]}

    assert rows["cjk-tidy"]["always_on_tokens"] >= len(chinese)
    assert rows["manual-only"]["always_on_tokens"] == 0


# --------------------------------------------------------------------------- #
# M33 (follow-up): the static rule view matches what native staging loads     #
# --------------------------------------------------------------------------- #
_RULE = "Write short sentences and name the owner of every dataset. " * 18
#: Rule shapes the native adapters split three ways: always on, a paths rule, or not staged.
_RULE_SHAPES: dict[str, str] = {
    "plain.md": _RULE,
    "plain.mdc": _RULE,
    "always.mdc": f"---\nalwaysApply: true\n---\n{_RULE}",
    "always-string.md": f'---\nalwaysApply: "true"\n---\n{_RULE}',
    "always-yes.md": f"---\nalwaysApply: yes\n---\n{_RULE}",
    "quoted-yes.md": f'---\nalwaysApply: "yes"\n---\n{_RULE}',
    "one.md": f"---\nalwaysApply: 1\n---\n{_RULE}",
    "off.md": f"---\nalwaysApply: false\n---\n{_RULE}",
    "requested.mdc": f"---\ndescription: Data rules\n---\n{_RULE}",
    "described.md": f"---\ndescription: Data rules\n---\n{_RULE}",
    "globs.mdc": f"---\nglobs: src/**/*.py\n---\n{_RULE}",
    "globs-off.mdc": f"---\nglobs: '*.ts'\nalwaysApply: false\n---\n{_RULE}",
    "paths.md": f"---\npaths: ['**/*.py']\n---\n{_RULE}",
    "empty-paths.md": f"---\npaths: ['']\n---\n{_RULE}",
    "empty-paths.mdc": f"---\npaths: ['']\n---\n{_RULE}",
    "leading-blank.md": f"\n\n---\nalwaysApply: false\n---\n{_RULE}",
    "bad-yaml.mdc": f"---\n: [\n---\n{_RULE}",
    "bad-yaml.md": f"---\n: [\n---\n{_RULE}",
}


def _adapter_staging(harness: str, name: str, content: str) -> str:
    """What the native adapter does with one rule file (Tier 3 stages the stripped file)."""
    from skillevaluator.tier3.plugin_native import _always_on_rule, _claude_user_rule

    content = content.strip()
    if harness == "codex":
        return "always_on" if _always_on_rule(name, content)[0] is not None else "not_staged"
    staged = _claude_user_rule(name, content)[0]
    if staged is None:
        return "not_staged"
    return "on_demand" if staged.startswith("---\npaths:") else "always_on"


@pytest.mark.parametrize("harness", ["claude-code", "codex"])
def test_m33_static_rule_view_matches_native_staging(tmp_path: Path, harness: str) -> None:
    """Verifier repro: `_rule_view` counted agent-requested, manual, and (for Codex) scoped rules always-on."""
    from skillevaluator.plugin_components import build_plugin_inventory

    root = tmp_path / "demo"
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "demo"}), encoding="utf-8")
    (root / "rules").mkdir()
    for name, content in _RULE_SHAPES.items():
        (root / "rules" / name).write_text(content, encoding="utf-8")
    inventory = build_plugin_inventory(
        root, {"name": "demo"}, contained=True, manifest_rel=".claude-plugin/plugin.json"
    )

    cost = inventory.context_cost(harness=harness, load_mode="native")
    static: dict[str, str] = {}
    for row in cost["by_component"]:
        if row["type"] == "rule":
            static[row["name"]] = (
                "always_on" if row["always_on_tokens"] else "on_demand" if row["on_demand_tokens"] else "not_staged"
            )

    expected = {name: _adapter_staging(harness, name, content) for name, content in _RULE_SHAPES.items()}
    assert static == expected
    assert "not_staged" in expected.values()


@pytest.mark.parametrize("agents", ["claude-code", "codex"])
def test_m33_native_provenance_leaves_out_unstaged_rules(tmp_path: Path, agents: str) -> None:
    """Verifier repro: Tier 3 native provenance put 550 always-on for rules that stage about 280."""
    files = {
        **_FILES,
        "rules/style.md": _RULE,
        "rules/requested.mdc": f"---\ndescription: Data rules\nalwaysApply: false\n---\n{_RULE}",
        "rules/scoped.md": f"---\npaths: ['**/*.py']\n---\n{_RULE}",
    }
    root = tmp_path / "demo"
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")

    package = prepare_plugin_eval_package(
        root, stage_root=tmp_path / "stage", plugin_load="native", agents=agents, env_mode="docker"
    )
    cost = package.provenance()["context_cost"]
    rules = {row["name"]: row for row in cost["by_component"] if row["type"] == "rule"}

    assert (cost["harness"], cost["load_mode"]) == (agents, "native")
    assert rules["style.md"]["always_on_tokens"] > 0
    assert rules["requested.mdc"]["always_on_tokens"] == 0
    assert rules["scoped.md"]["always_on_tokens"] == 0
    assert sum(row["always_on_tokens"] for row in rules.values()) == rules["style.md"]["always_on_tokens"]
