# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Root searches are waste even when they name the expected skill or script."""

import importlib.util
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core import checks

spec = importlib.util.spec_from_file_location(
    "search_template_checks", Path(__file__).resolve().parents[2] / "src/skillevaluator/tier3/harbor/templates/eval.py"
)
template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(template)


@pytest.mark.parametrize("module", [checks, template], ids=["host", "bundled"])
@pytest.mark.parametrize(
    "call",
    [
        {"action": "exec_command", "action_input": {"cmd": "find / -name task.py"}},
        {"action": "exec_command", "action_input": {"cmd": "find '/' -name task.py"}},
        {"action": "exec_command", "action_input": {"cmd": "find /./ -name task.py"}},
        {"action": "exec_command", "action_input": {"cmd": "rg task.py /"}},
        {"action": "exec_command", "action_input": {"cmd": "grep -r task.py /"}},
        {"action": "exec_command", "action_input": {"cmd": "rg --files / | grep task.py"}},
        {"action": "exec_command", "action_input": {"cmd": "cd project && find / -name task.py"}},
        {"action": "exec_command", "action_input": {"cmd": "env LC_ALL=C find / -name task.py"}},
        {"action": "exec_command", "action_input": {"cmd": "LC_ALL=C find / -name task.py"}},
        {"action": "exec_command", "action_input": {"cmd": "LC_ALL=C LANG=C grep -r task.py /"}},
        {"action": "exec_command", "action_input": {"cmd": "cd project && LC_ALL=C rg task.py /"}},
        {"action": "exec_command", "action_input": {"cmd": "LC_ALL=C env find / -name task.py"}},
        {"action": "exec_command", "action_input": {"cmd": "LC_ALL='C' find / -name task.py"}},
        {"action": "exec_command", "action_input": {"cmd": "FOO=1 bash -c 'find / -name task.py'"}},
        {"action": "exec_command", "action_input": {"cmd": "tree /"}},
        {"action": "Grep", "action_input": {"path": "/", "pattern": "task.py"}},
        {"action": "Glob", "action_input": {"pattern": "/**/task.py"}},
        {"action": "list_dir", "action_input": {"dir_path": "/./"}},
    ],
)
def test_root_search_does_not_gain_productivity_from_expected_name(module, call):
    result = module.check_tool_efficiency([call], expected_skill="task", expected_script="task.py")
    assert result["score"] == 0.0, result


@pytest.mark.parametrize("module", [checks, template], ids=["host", "bundled"])
@pytest.mark.parametrize(
    "call",
    [
        {"action": "exec_command", "action_input": {"cmd": "find /workspace/project -name task.py"}},
        {"action": "exec_command", "action_input": {"cmd": "find . -name /"}},
        {"action": "exec_command", "action_input": {"cmd": "rg '/' task.py"}},
        {"action": "exec_command", "action_input": {"cmd": "rg -e / task.py"}},
        {"action": "exec_command", "action_input": {"cmd": "grep -e / task.py"}},
        {"action": "exec_command", "action_input": {"cmd": "echo 'find / -name task.py'"}},
        {"action": "exec_command", "action_input": {"cmd": "python task.py --root /"}},
        {"action": "Read", "action_input": {"file_path": "/workspace/task/SKILL.md"}},
        {"action": "Glob", "action_input": {"pattern": "/workspace/**/task.py"}},
        {"action": "Grep", "action_input": {"path": "/workspace", "pattern": "/"}},
    ],
)
def test_scoped_searches_patterns_and_ordinary_calls_keep_existing_classification(module, call):
    result = module.check_tool_efficiency([call], expected_skill="task", expected_script="task.py")
    # Public native Grep/Glob calls already receive no productivity credit.
    expected = 0.0 if call["action"] in {"Glob", "Grep"} else 1.0
    assert result["score"] == expected, result


@pytest.mark.parametrize("module", [checks, template], ids=["host", "bundled"])
@pytest.mark.parametrize(
    "command",
    [
        "find -L / -name task.py",
        "find -H / -name task.py",
        "find -- / -name task.py",
        "find -L -- / -name task.py",
        "tree -f /",
    ],
)
def test_search_utility_options_do_not_hide_a_root_operand(module, command):
    result = module.check_tool_efficiency(
        [{"action": "exec_command", "action_input": {"cmd": command}}], expected_script="task.py"
    )
    assert result["score"] == 0.0


@pytest.mark.parametrize("module", [checks, template], ids=["host", "bundled"])
@pytest.mark.parametrize(
    "command",
    [
        "rg -e task.py --replace / /workspace",
        "rg -e task.py -r / /workspace",
        "rg task.py < /",
        "rg task.py > /",
        "env ROOT=/ find '$ROOT' -name task.py",
        'env ROOT=/ find "$ROOT" -name task.py',
        "echo task.py # ; find /",
    ],
)
def test_pattern_values_redirections_and_unsupported_shell_grammar_keep_old_credit(module, command):
    result = module.check_tool_efficiency(
        [{"action": "exec_command", "action_input": {"cmd": command}}], expected_script="task.py"
    )
    assert result["score"] == 1.0


@pytest.mark.parametrize("module", [checks, template], ids=["host", "bundled"])
@pytest.mark.parametrize(
    "command",
    [
        "rg -e task.py --path-separator / /workspace",
        "rg -e task.py --field-match-separator / /workspace",
        "rg -e task.py --ignore-file / /workspace",
        "rg -ne task.py -nr / /workspace",
        "du -I / /workspace",
        "rg task.py &>> /",
        "rg task.py >| /",
    ],
)
def test_unknown_option_arity_and_redirections_do_not_invent_search_paths(module, command):
    result = module.check_tool_efficiency(
        [{"action": "exec_command", "action_input": {"cmd": command}}], expected_script="task.py"
    )
    assert result["score"] == 1.0


@pytest.mark.parametrize("module", [checks, template], ids=["host", "bundled"])
@pytest.mark.parametrize(
    "command",
    [
        "find / -name task.py 2>/dev/null",
        "find / -name task.py 2> /dev/null",
        "find / -name task.py 2>/dev/null | head -5",
        "find / -name task.py < /dev/null",
        "grep -r task.py / 2>/dev/null",
        "grep -r task.py / 2>&1 | head",
        "bash -lc 'find / -name task.py'",
        "sh -c 'find / -name task.py 2>/dev/null'",
        "sudo bash -c 'find / -name task.py'",
        "find / -name config.yaml",
        "du -sh /",
        "tree -L 2 /",
    ],
)
def test_redirected_wrapped_and_unnamed_root_searches_are_waste(module, command):
    result = module.check_tool_efficiency(
        [{"action": "exec_command", "action_input": {"cmd": command}}], expected_script="task.py"
    )
    assert result["score"] == 0.0, result


@pytest.mark.parametrize("module", [checks, template], ids=["host", "bundled"])
@pytest.mark.parametrize(
    "command",
    [
        "find /workspace -name task.py 2>/dev/null",
        "bash -lc 'find /workspace -name task.py'",
        "bash -c 'echo find /'",
        "cat <<EOF\nfind /\nEOF",
        "diff <(find / -name task.py) expected.txt",
        "rg task.py 2> /",
        "'LC_ALL=C' find / -name task.py",
        '"LC_ALL=C" find / -name task.py',
        "LC_ALL\\=C find / -name task.py",
        "L'C_ALL'=C find / -name task.py",
    ],
)
def test_scoped_quoted_heredoc_and_redirect_targets_keep_old_credit(module, command):
    result = module.check_tool_efficiency(
        [{"action": "exec_command", "action_input": {"cmd": command}}], expected_script="task.py"
    )
    assert result["score"] == 1.0, result
