# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core.atif_helpers import (
    build_metric_evidence_refs,
    extract_tool_calls_as_dicts,
)
from skillevaluator.tier3.eval_core.checks import check_security, check_workflow_order

_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)


def _load_template_module():
    spec = importlib.util.spec_from_file_location("harbor_template_eval_codex_tools", _TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trajectory(source: str) -> dict:
    return {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "outer-call",
                        "function_name": "exec",
                        "arguments": {"input": source},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "outer-call",
                            "content": "outer observation",
                        }
                    ]
                },
            }
        ]
    }


@pytest.mark.parametrize(
    "extractor",
    [extract_tool_calls_as_dicts, _load_template_module().extract_tool_calls_as_dicts],
)
def test_native_codex_exec_unwraps_static_tools_in_source_order(extractor):
    source = """
const read = await tools.exec_command({"cmd":"cat /skills/example/SKILL.md"});
const plan = await tools.update_plan({plan:[{step:"Run checks",status:"in_progress"}]});
const run = await tools.exec_command({"cmd":"rm -rf /workspace/project"});
"""

    calls = extractor(_trajectory(source))

    assert [call["action"] for call in calls] == ["exec_command", "update_plan", "exec_command"]
    assert calls[0]["action_input"] == {"cmd": "cat /skills/example/SKILL.md"}
    assert calls[1]["action_input"]["plan"][0]["step"] == "Run checks"
    assert calls[2]["action_input"] == {"cmd": "rm -rf /workspace/project"}
    assert [call["observation"] for call in calls] == ["outer observation"] * 3

    workflow = check_workflow_order(calls, expected_skill="example")
    assert workflow["passed"] is True
    assert check_workflow_order([calls[1]], expected_skill="example")["passed"] is False
    security = check_security(calls)
    assert any(finding["type"] == "destructive_command" for finding in security["findings"])


@pytest.mark.parametrize(
    "extractor",
    [extract_tool_calls_as_dicts, _load_template_module().extract_tool_calls_as_dicts],
)
def test_native_codex_exec_accepts_one_first_line_pragma(extractor):
    source = (
        '// @exec: {"yield_time_ms": 10000}\n'
        'const r = await tools.exec_command({"cmd":"pwd"});\n'
        "text(r.output);"
    )

    assert [call["action"] for call in extractor(_trajectory(source))] == ["exec_command"]


@pytest.mark.parametrize(
    "source",
    [
        "const result = await tools.exec_command(argumentsFromRuntime);",
        'const result = await tools.exec_command({"cmd":"pwd";',
        'const text = "tools.exec_command({\\"cmd\\":\\"rm -rf /workspace/project\\"})";',
        'const pattern = /tools.exec_command\\({"cmd":"rm -rf \\/workspace\\/project"}\\)/;',
        'if (enabled) /tools.exec_command\\({"cmd":"rm -rf \\/workspace\\/project"}\\)/.test(input);',
        'const ratio = total / count;\ntools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'const nested = other.tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        '// tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        (
            "const plan = await tools.update_plan({plan:[]}); "
            "const result = await tools.exec_command(argumentsFromRuntime);"
        ),
        'const input = "rm -rf /workspace/project";',
        '// @exec: {"yield_time_ms": 10000} const r = await tools.exec_command({"cmd":"pwd"});',
        (
            '// @exec: {"yield_time_ms": 10000}\n'
            '// @exec: {"max_tokens": 1000}\n'
            'const r = await tools.exec_command({"cmd":"pwd"});'
        ),
    ],
)
@pytest.mark.parametrize(
    "extractor",
    [extract_tool_calls_as_dicts, _load_template_module().extract_tool_calls_as_dicts],
)
def test_native_codex_exec_does_not_infer_dynamic_or_non_call_input(source, extractor):
    assert extractor(_trajectory(source)) == [
        {
            "action": "exec",
            "action_input": {"input": source},
            "observation": "outer observation",
        }
    ]


@pytest.mark.parametrize(
    "source",
    [
        'if (false) tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'false && tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'function neverCalled() { tools.exec_command({"cmd":"rm -rf /workspace/project"}); }',
        'const = ; tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'tools.exec_command({"cmd":"rm -rf /workspace/project"}); const = ;',
        (
            'const plan = await tools.update_plan({plan:[]}); '
            'tools["exec_command"]({"cmd":"rm -rf /workspace/project"});'
        ),
        (
            'const plan = await tools.update_plan({plan:[]}); '
            'const value = `${tools.exec_command({"cmd":"rm -rf /workspace/project"})}`;'
        ),
    ],
)
@pytest.mark.parametrize(
    "extractor",
    [extract_tool_calls_as_dicts, _load_template_module().extract_tool_calls_as_dicts],
)
def test_native_codex_exec_rejects_unexecuted_or_partially_supported_wrappers(source, extractor):
    assert [call["action"] for call in extractor(_trajectory(source))] == ["exec"]


def test_native_codex_exec_evidence_refs_resolve_to_the_outer_call():
    trajectory = _trajectory(
        'const plan = await tools.update_plan({plan:[]}); '
        'const run = await tools.exec_command({"cmd":"touch /workspace/result.txt"});'
    )

    refs = build_metric_evidence_refs(trajectory, "q")["goal_accuracy"]
    tool_refs = [ref for ref in refs if ref["kind"] == "tool_call"]

    assert tool_refs
    assert {ref["json_pointer"] for ref in tool_refs} == {"/steps/0/tool_calls/0"}
