# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool efficiency counts real work, wastes root searches and setup noise, and ignores bookkeeping."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core import checks

_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "src/skillevaluator/tier3/harbor/templates/eval.py"
_SPEC = importlib.util.spec_from_file_location("tool_productivity_template", _TEMPLATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
template = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(template)

_MODULES = [pytest.param(checks, id="host"), pytest.param(template, id="bundled")]
_SKILL = "release-review"


def _call(action: str, **args) -> dict:
    return {"action": action, "action_input": args}


def _classify(module, call: dict, expected_script: str | None = None) -> str:
    """Classify one call through check_tool_efficiency, which both copies expose."""
    result = module.check_tool_efficiency([call], _SKILL, expected_script)
    if result["reason"].startswith("No scored tool calls"):
        kind = "neutral"
    else:
        kind = {1.0: "productive", 0.0: "wasted"}[result["score"]]
    if module is checks:
        assert checks._tool_call_productivity(call, _SKILL, expected_script) == kind
    return kind


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    "call",
    [
        _call("Read", file_path="src/app.py"),
        _call("Write", file_path="report.md", content="x"),
        _call("Edit", file_path="src/app.py"),
        _call("MultiEdit", file_path="src/app.py"),
        _call("NotebookEdit", notebook_path="a.ipynb"),
        _call("Grep", pattern="TODO", path="src"),
        _call("Glob", pattern="**/*.py"),
        _call("LS", path="src"),
        _call("Bash", command="pytest -q"),
        _call("Skill", skill=_SKILL),
        _call("apply_patch", input="*** Begin Patch\n*** End Patch"),
        _call("exec_command", cmd="make test"),
        _call("shell", command=["bash", "-lc", "ls src"]),
        _call("list_dir", dir_path="src"),
        _call("grep_files", pattern="TODO"),
        _call("mcp__github__get_issue", issue=1),
    ],
    ids=lambda call: call["action"],
)
def test_real_work_is_productive(module, call):
    assert _classify(module, call) == "productive"


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    "action",
    [
        "TodoWrite",
        "TodoRead",
        "update_plan",
        "write_stdin",
        "BashOutput",
        "KillShell",
        "KillBash",
        "kill_shell",
        "TaskStop",
        "TaskOutput",
        "ToolSearch",
        "Task",
        "Agent",
        "TaskCreate",
        "TaskGet",
        "TaskList",
        "TaskUpdate",
        "EnterPlanMode",
        "ExitPlanMode",
        "spawn_agent",
        "wait",
        "wait_agent",
        "close_agent",
        "send_input",
        "resume_agent",
    ],
)
def test_bookkeeping_is_neutral(module, action):
    assert _classify(module, _call(action)) == "neutral"


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    ("call", "kind"),
    [
        (_call("write_stdin", session_id=1), "neutral"),
        (_call("write_stdin", session_id=1, chars=""), "neutral"),
        (_call("write_stdin", session_id=1, chars="\n"), "neutral"),
        (_call("write_stdin", session_id=1, chars="\x03"), "neutral"),
        (_call("write_stdin", session_id=1, chars="python review.py\n"), "productive"),
        (_call("write_stdin", session_id=1, chars="pytest -q\n"), "productive"),
        (_call("write_stdin", session_id=1, chars="find / -name review.py\n"), "wasted"),
        (_call("write_stdin", session_id=1, chars="pip install requests\n"), "wasted"),
        (_call("Monitor", description="deploy events", ws={"url": "wss://events.example.test"}), "neutral"),
        (
            _call("Monitor", description="errors", command="tail -f build.log | grep --line-buffered ERROR"),
            "productive",
        ),
        (_call("Monitor", description="skills", command="find / -name SKILL.md"), "wasted"),
    ],
    ids=lambda value: value if isinstance(value, str) else f"{value['action']}:{value['action_input']}",
)
def test_session_input_is_classified_by_the_text_it_sends(module, call, kind):
    # Polling a session or watching a socket is bookkeeping. Typing a command
    # into a shell, or starting a watch script, runs that command.
    assert _classify(module, call, "review.py") == kind


@pytest.mark.parametrize("module", _MODULES)
def test_root_search_typed_into_an_interactive_shell_is_wasted(module):
    calls = [
        _call("exec_command", cmd="bash", tty=True),
        _call("write_stdin", session_id=1, chars="find / -name review.py\n"),
        _call("write_stdin", session_id=1, chars=""),
    ]

    result = module.check_tool_efficiency(calls, _SKILL, "review.py")

    assert result["score"] == 0.5
    assert result["reason"] == "1/2 productive calls (50%); 1 planning or bookkeeping call(s) not counted"


def test_wasted_session_input_is_labelled_with_its_text():
    calls = [_call("write_stdin", session_id=1, chars="find / -name review.py\n")]

    result = checks.check_tool_efficiency(calls, _SKILL)

    assert result["details"]["wasted_calls"] == ["write_stdin(find / -name review.py\n)"]


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    ("call", "kind"),
    [
        (_call("mcp__filesystem__search_files", path="/", pattern="SKILL.md"), "wasted"),
        (_call("mcp__filesystem__list_directory", path="/"), "wasted"),
        (_call("mcp__filesystem__directory_tree", path="//"), "wasted"),
        (_call("mcp__workspace__list", dir_path="/"), "wasted"),
        (_call("mcp__my_filesystem__stat", path="/"), "wasted"),
        (_call("mcp__filesystem__search_files", path="/workspace", pattern="SKILL.md"), "productive"),
        (_call("mcp__filesystem__list_directory", path="/workspace/src"), "productive"),
        (_call("mcp__github__search_code", query="path:/ release"), "productive"),
    ],
    ids=lambda value: value if isinstance(value, str) else f"{value['action']}:{value['action_input']}",
)
def test_mcp_calls_rooted_at_filesystem_root_are_wasted(module, call, kind):
    assert _classify(module, call) == kind


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    "call",
    [
        _call("mcp__github__get_file_contents", owner="o", repo="r", path="/"),
        _call("mcp__github__get_file_contents", owner="o", repo="r", path="src/"),
        _call("mcp__docs__get_page", path="/"),
        _call("mcp__wiki__read_page", dir_path="/"),
    ],
    ids=lambda call: f"{call['action']}:{call['action_input']}",
)
def test_mcp_path_root_of_other_servers_is_productive(module, call):
    # Other servers use path for something other than the local filesystem:
    # path "/" on the GitHub server lists a repository root, not the disk.
    assert _classify(module, call) == "productive"


@pytest.mark.parametrize("module", _MODULES)
def test_claude_code_task_list_calls_are_left_out_of_the_score(module):
    calls = [
        _call("Read", file_path=f"/workspace/skills/{_SKILL}/SKILL.md"),
        *[_call("TaskCreate", subject=f"step {index}") for index in range(3)],
        _call("Edit", file_path="src/a.py"),
        _call("Edit", file_path="src/b.py"),
        *[_call("TaskUpdate", taskId=str(index), status="completed") for index in range(3)],
        _call("Bash", command="pytest -q"),
    ]

    result = module.check_tool_efficiency(calls, _SKILL)

    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["reason"] == "4/4 productive calls (100%); 6 planning or bookkeeping call(s) not counted"


@pytest.mark.parametrize("module", _MODULES)
def test_codex_wait_is_left_out_like_wait_agent(module):
    for waiter in ("wait", "wait_agent"):
        calls = [_call("spawn_agent", message="review"), _call(waiter), _call("exec_command", cmd="pytest -q")]
        result = module.check_tool_efficiency(calls, _SKILL)
        assert result["score"] == 1.0, (waiter, result)
        assert result["reason"] == "1/1 productive calls (100%); 2 planning or bookkeeping call(s) not counted"


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    ("call", "kind"),
    [
        (_call("functions.apply_patch", input="*** Begin Patch\n*** End Patch"), "productive"),
        (_call("functions.list_dir", dir_path="src"), "productive"),
        (_call("functions.grep_files", pattern="TODO", path="src"), "productive"),
        (_call("functions.exec_command", cmd="pytest -q"), "productive"),
        (_call("functions.update_plan", plan=[]), "neutral"),
        (_call("functions.spawn_agent", message="review"), "neutral"),
        (_call("functions.list_dir", dir_path="/"), "wasted"),
        (_call("functions.grep_files", pattern="release", path="/"), "wasted"),
        (_call("functions.exec_command", cmd="find / -name review.py"), "wasted"),
    ],
    ids=lambda value: value if isinstance(value, str) else f"{value['action']}:{value['action_input']}",
)
def test_namespaced_codex_tools_are_classified_by_their_name(module, call, kind):
    assert _classify(module, call, "review.py") == kind


@pytest.mark.parametrize("module", _MODULES)
def test_namespaced_codex_patch_run_keeps_full_tool_efficiency(module):
    calls = [
        _call("functions.exec_command", cmd="pytest -q"),
        _call("functions.apply_patch", input="*** Begin Patch\n*** End Patch"),
        _call("functions.apply_patch", input="*** Begin Patch\n*** End Patch"),
        _call("functions.update_plan", plan=[]),
    ]

    result = module.check_tool_efficiency(calls, _SKILL)

    assert result["score"] == 1.0
    assert result["reason"] == "3/3 productive calls (100%); 1 planning or bookkeeping call(s) not counted"


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize("action", ["mcp__tasks__task", "mcp__planner__update_plan", "mcp__agents__wait"])
def test_mcp_tools_named_like_bookkeeping_tools_stay_mcp_calls(module, action):
    assert _classify(module, _call(action, id="1")) == "productive"


@pytest.mark.parametrize("module", _MODULES)
def test_background_shell_polling_gets_no_execution_credit(module):
    # BashOutput and KillShell contain "bash" and "shell", which used to make
    # them execution calls. Polling a background shell is not new work.
    result = module.check_tool_efficiency(
        [_call("BashOutput", bash_id="1"), _call("KillShell", shell_id="1"), _call("Bash", command="grep -r x /")],
        _SKILL,
    )
    assert result["score"] == 0.0
    assert result["reason"] == "0/1 productive calls (0%); 2 planning or bookkeeping call(s) not counted"


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    "command",
    [
        "grep -r SKILL.md /",
        "grep -rl 'RELEASE|' / 2>/dev/null",
        "find / -name policy.json",
        "rg policy.json / | head",
        "du -sh /",
        "pip install requests",
        "python --version",
        "which jq",
    ],
)
def test_filesystem_scans_and_setup_noise_are_wasted(module, command):
    assert _classify(module, _call("Bash", command=command)) == "wasted"


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    "call",
    [
        _call("Bash", command="find / -name review.py"),
        _call("Bash", command="grep -r review.py /"),
        _call("exec_command", cmd="find / -name review.py 2>/dev/null"),
        _call("shell", command=["bash", "-lc", "find / -name review.py"]),
        _call("shell", command=["find", "/", "-name", "review.py"]),
        _call("shell", cmd=["sh", "-c", "grep -r review.py /"]),
    ],
    ids=lambda call: f"{call['action']}:{call['action_input']}",
)
def test_root_searches_naming_the_expected_script_are_wasted(module, call):
    assert _classify(module, call, "review.py") == "wasted"


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    "call",
    [
        _call("Bash", command="python review.py"),
        _call("Bash", command="cd /work && python scripts/review.py --input /work/pr.diff"),
        _call("shell", command=["python", "review.py"]),
        _call("shell", command=["bash", "-lc", "python review.py"]),
    ],
    ids=lambda call: f"{call['action']}:{call['action_input']}",
)
def test_running_the_expected_script_stays_productive(module, call):
    assert _classify(module, call, "review.py") == "productive"


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    "search",
    [
        _call("Bash", command="find / -name review.py"),
        _call("shell", command=["bash", "-lc", "find / -name review.py"]),
        _call("shell", command=["find", "/", "-name", "review.py"]),
    ],
    ids=["string", "argv-shell", "argv-find"],
)
def test_root_search_for_the_expected_script_scores_like_any_root_search(module, search):
    run = _call("Bash", command="python review.py")
    other = _call("Bash", command="find / -name other.py")

    named_result = module.check_tool_efficiency([search, run], _SKILL, "review.py")
    other_result = module.check_tool_efficiency([other, run], _SKILL, "review.py")
    only_search = module.check_tool_efficiency([search], _SKILL, "review.py")

    assert named_result["score"] == other_result["score"] == 0.5
    assert only_search["score"] == 0.0


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    "call",
    [
        _call("Glob", pattern="**/SKILL.md", path="/"),
        _call("Glob", pattern="/**/SKILL.md"),
        _call("Glob", pattern="/*/policy.json"),
        _call("Grep", pattern="release", path="/"),
        _call("Grep", pattern="release", path="//"),
        _call("LS", path="/"),
        _call("grep_files", pattern="release", path="/"),
        _call("list_dir", dir_path="/", depth=5),
    ],
    ids=lambda call: f"{call['action']}:{call['action_input']}",
)
def test_native_searches_rooted_at_filesystem_root_are_wasted(module, call):
    assert _classify(module, call) == "wasted"


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    "call",
    [
        _call("Grep", pattern="release", path="/work"),
        _call("Glob", pattern="/work/**/*.yaml"),
        _call("Glob", pattern="**/*.py", path="/workspace"),
        _call("LS", path="/tmp"),
        _call("grep_files", pattern="release", path="/work"),
        _call("list_dir", dir_path="/work", depth=2),
    ],
    ids=lambda call: f"{call['action']}:{call['action_input']}",
)
def test_scoped_native_searches_stay_productive(module, call):
    assert _classify(module, call) == "productive"


@pytest.mark.parametrize("module", _MODULES)
def test_native_and_shell_root_scans_score_the_same(module):
    read = _call("Read", file_path="/work/manifest.yaml")
    native = [_call("Glob", pattern="**/SKILL.md", path="/"), _call("Grep", pattern="release", path="/"), read]
    shell = [_call("Bash", command="find / -name SKILL.md"), _call("Bash", command="grep -r release /"), read]

    native_result = module.check_tool_efficiency(native, _SKILL)
    shell_result = module.check_tool_efficiency(shell, _SKILL)

    assert native_result["score"] == shell_result["score"] == round(1 / 3, 4)
    assert native_result["passed"] is shell_result["passed"] is False


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize("command", ["grep -r TODO src/", "find . -name '*.py'", "ls /tmp", "grep -n x /etc/hosts"])
def test_scoped_searches_stay_productive(module, command):
    assert _classify(module, _call("Bash", command=command)) == "productive"


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    "argv",
    [
        ["pip", "install", "requests"],
        ["pip", "install", "-e", "."],
        ["brew", "install", "jq"],
        ["npm", "install"],
        ["which", "jq"],
        ["python", "--version"],
    ],
    ids=" ".join,
)
def test_setup_noise_in_an_argv_command_is_wasted(module, argv):
    # Help, version and install commands recorded as an argument list are
    # waste, like the same command written as text.
    assert _classify(module, _call("shell", command=argv)) == "wasted"
    assert _classify(module, _call("Bash", command=" ".join(argv))) == "wasted"


_README_PATCH = (
    "*** Begin Patch\n*** Update File: README.md\n@@\n-Install it.\n"
    "+Run `pip install -e .`, then pick the parser which matches the input.\n"
    "+See `--help` to adapt the defaults, or `apt-get` and `brew ` for system packages.\n"
    "*** End Patch"
)


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize("action", ["apply_patch", "functions.apply_patch", "applypatch"])
def test_patch_text_is_not_read_as_a_command(module, action):
    # The converter records a non-object tool input under "raw". A patch that
    # mentions an install or help command is still a patch, so it scores the
    # same under every patch argument.
    for key in ("input", "raw", "patch", "patchText"):
        assert _classify(module, _call(action, **{key: _README_PATCH})) == "productive", key


@pytest.mark.parametrize("module", _MODULES)
@pytest.mark.parametrize(
    "call",
    [
        _call("Edit", raw=_README_PATCH),
        _call("Write", file_path="docs/which version.md", content="pip install -e ."),
        _call("Read", file_path="docs/brew install.md"),
        _call("mcp__docs__search", query="pip install --help"),
    ],
    ids=lambda call: f"{call['action']}:{sorted(call['action_input'])}",
)
def test_install_words_in_file_text_are_not_commands(module, call):
    assert _classify(module, call) == "productive"


@pytest.mark.parametrize("module", _MODULES)
def test_a_command_recorded_under_raw_is_still_checked(module):
    assert _classify(module, _call("exec_command", raw="pip install requests")) == "wasted"
    assert _classify(module, _call("exec_command", raw="pytest -q")) == "productive"


@pytest.mark.parametrize("module", _MODULES)
def test_neutral_calls_are_left_out_of_the_score(module):
    calls = [
        _call("TodoWrite"),
        _call("Read", file_path="SKILL.md"),
        _call("Bash", command="grep -r x /"),
        _call("TodoWrite"),
        _call("Write", file_path="out.md"),
    ]

    result = module.check_tool_efficiency(calls, _SKILL)

    assert result["score"] == round(2 / 3, 4)
    assert result["passed"] is True
    assert result["reason"] == "2/3 productive calls (67%); 2 planning or bookkeeping call(s) not counted"


def test_details_count_neutral_calls_and_label_wasted_argv_calls():
    calls = [
        _call("update_plan"),
        _call("shell", command=["find", "/", "-name", "review.py"]),
        _call("apply_patch", input="*** Begin Patch\n*** End Patch"),
    ]

    result = checks.check_tool_efficiency(calls, _SKILL, "review.py")

    assert result["details"] == {
        "productive": 1,
        "wasted": 1,
        "neutral": 1,
        "total": 2,
        "wasted_calls": ["shell(['find', '/', '-name', 'review.py'])"],
    }


@pytest.mark.parametrize("module", _MODULES)
def test_codex_sub_agent_calls_are_left_out_of_the_score(module):
    actions = ["update_plan", "spawn_agent", "spawn_agent", "spawn_agent", "exec_command", "wait_agent"]
    calls = [_call(action, cmd="make test") if action == "exec_command" else _call(action) for action in actions]
    calls += [_call("update_plan"), _call("update_plan"), _call("exec_command", cmd="pytest -q"), _call("update_plan")]

    result = module.check_tool_efficiency(calls, "review-pr")

    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["reason"] == "2/2 productive calls (100%); 8 planning or bookkeeping call(s) not counted"


@pytest.mark.parametrize("module", _MODULES)
def test_only_bookkeeping_calls_score_zero(module):
    # A trial that only plans did no work. It must not outscore a trial with
    # no tool calls, whose skill_efficiency is 0.0.
    result = module.check_tool_efficiency([_call("TodoWrite"), _call("update_plan")], _SKILL)

    assert result["passed"] is False
    assert result["score"] == 0.0
    assert result["reason"] == "No scored tool calls; 2 planning or bookkeeping call(s) not counted"


def test_only_bookkeeping_calls_record_zero_scored_calls():
    result = checks.check_tool_efficiency([_call("TodoWrite"), _call("update_plan")], _SKILL)

    assert result["details"] == {"productive": 0, "wasted": 0, "neutral": 2, "total": 0, "wasted_calls": []}


def _tool_step(index: int, name: str, arguments: dict) -> dict:
    call_id = f"call-{index}"
    return {
        "source": "agent",
        "tool_calls": [{"tool_call_id": call_id, "function_name": name, "arguments": arguments}],
        "observation": {"results": [{"source_call_id": call_id, "content": "ok"}]},
    }


def _run_verifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, dict]]) -> dict:
    """Run the bundled verifier's main() on a no-reference with-skill case."""
    trajectory = {
        "steps": [
            {"source": "user", "message": "Review the release notes and fix the modules."},
            *[_tool_step(index, name, arguments) for index, (name, arguments) in enumerate(calls)],
            {"source": "agent", "message": "Fixed the modules and the tests pass."},
        ]
    }
    agent_dir = tmp_path / "logs" / "agent"
    verifier_dir = tmp_path / "logs" / "verifier"
    agent_dir.mkdir(parents=True)
    verifier_dir.mkdir(parents=True)
    atif = agent_dir / "trajectory.json"
    atif.write_text(json.dumps(trajectory), encoding="utf-8")
    entry = tmp_path / "entry.json"
    entry.write_text(
        json.dumps({"id": "case-1", "question": "Fix the modules", "has_skill": True, "expected_skill": _SKILL}),
        encoding="utf-8",
    )
    monkeypatch.setattr(template, "ATIF_PATH", atif)
    monkeypatch.setattr(template, "ENTRY_PATH", entry)
    monkeypatch.setattr(template, "VERIFIER_DIR", verifier_dir)
    monkeypatch.setattr(template, "REWARD_JSON", verifier_dir / "reward.json")
    monkeypatch.setattr(template, "REWARD_TXT", verifier_dir / "reward.txt")
    monkeypatch.setattr(template, "SKILL_EVALUATOR_REWARD_JSON", verifier_dir / "skill_evaluator_reward.json")
    monkeypatch.setattr(template, "call_public_llm", lambda *_a, **_k: pytest.fail("No-reference case called an LLM"))

    template.main()

    return json.loads((verifier_dir / "skill_evaluator_reward.json").read_text(encoding="utf-8"))


def test_edit_heavy_run_keeps_full_tool_efficiency_in_skill_efficiency(tmp_path, monkeypatch):
    calls = [
        ("Read", {"file_path": f"/workspace/skills/{_SKILL}/SKILL.md"}),
        ("TodoWrite", {"todos": []}),
        *[("Edit", {"file_path": f"/workspace/src/module_{index}.py"}) for index in range(4)],
        ("Bash", {"command": "pytest -q"}),
    ]

    rich = _run_verifier(tmp_path, monkeypatch, calls)

    efficiency = rich["details"]["skill_efficiency"]
    assert efficiency["tool_efficiency"] == {
        "passed": True,
        "score": 1.0,
        "reason": "6/6 productive calls (100%); 1 planning or bookkeeping call(s) not counted",
    }
    routing = efficiency["routing"]["score"]
    assert rich["skill_efficiency"] == round((routing + 1.0) / 2, 4)


@pytest.mark.parametrize(
    "calls",
    [
        [],
        [("TodoWrite", {"todos": []})],
        [("update_plan", {"plan": []})],
        [("Task", {"prompt": "Fix the modules"})],
    ],
    ids=["no-calls", "TodoWrite", "update_plan", "Task"],
)
def test_planning_only_run_scores_skill_efficiency_like_no_tool_calls(tmp_path, monkeypatch, calls):
    rich = _run_verifier(tmp_path, monkeypatch, calls)

    assert rich["skill_efficiency"] == 0.0
    host = checks.score_skill_efficiency([_call(name, **arguments) for name, arguments in calls], _SKILL)
    assert host["score"] == 0.0
