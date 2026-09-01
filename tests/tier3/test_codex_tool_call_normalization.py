# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core.atif_helpers import (
    build_behavior_evidence,
    build_metric_evidence_refs,
    extract_tool_calls_as_dicts,
)
from skillevaluator.tier3.eval_core.checks import (
    check_activation,
    check_error_recovery,
    check_negative_case,
    check_routing,
    check_script_execution,
    check_security,
    check_tool_efficiency,
    check_workflow_order,
)
from skillevaluator.tier3.harbor import adapter

_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)


def _load_template_module():
    spec = importlib.util.spec_from_file_location("harbor_template_eval_codex_tools", _TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TEMPLATE_MODULE = _load_template_module()
_EXTRACTORS = [extract_tool_calls_as_dicts, _TEMPLATE_MODULE.extract_tool_calls_as_dicts]


def _trajectory(source: str, observation: str = "outer observation") -> dict:
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
                            "content": observation,
                        }
                    ]
                },
            }
        ]
    }


@pytest.mark.parametrize(
    "extractor",
    _EXTRACTORS,
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
    assert [call["observation"] for call in calls] == [""] * 3
    assert [call["observation_status"] for call in calls] == ["unobserved_inner_call"] * 3

    assert check_workflow_order(calls, expected_skill="example")["passed"] is True
    assert any(finding["type"] == "destructive_command" for finding in check_security(calls)["findings"])


@pytest.mark.parametrize(
    "extractor",
    _EXTRACTORS,
)
def test_native_codex_exec_accepts_one_first_line_pragma(extractor):
    source = '// @exec: {"yield_time_ms": 10000}\nconst r = await tools.exec_command({"cmd":"pwd"});\ntext(r.output);'

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
        '// @exec: {}\nconst input = "rm -rf /workspace/project";',
        '// @exec: {"yield_time_ms": 10000} const r = await tools.exec_command({"cmd":"pwd"});',
        (
            '// @exec: {"yield_time_ms": 10000}\n'
            '// @exec: {"max_tokens": 1000}\n'
            'const r = await tools.exec_command({"cmd":"pwd"});'
        ),
        'if (false) tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'false && tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'function neverCalled() { tools.exec_command({"cmd":"rm -rf /workspace/project"}); }',
        'const = ; tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'tools.exec_command({"cmd":"rm -rf /workspace/project"}); const = ;',
        (
            "const plan = await tools.update_plan({plan:[]}); "
            'tools["exec_command"]({"cmd":"rm -rf /workspace/project"});'
        ),
        (
            "const plan = await tools.update_plan({plan:[]}); "
            'const value = `${tools.exec_command({"cmd":"rm -rf /workspace/project"})}`;'
        ),
    ],
)
@pytest.mark.parametrize(
    "extractor",
    _EXTRACTORS,
)
def test_native_codex_exec_keeps_unsupported_wrappers_atomic(source, extractor):
    assert extractor(_trajectory(source)) == [
        {
            "action": "exec",
            "action_input": {"input": source},
            "observation": "outer observation",
            "normalization_status": "unsupported_native_codex_exec_wrapper",
        }
    ]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_unsupported_codex_wrapper_is_not_a_clean_security_result(extractor):
    calls = extractor(_trajectory('if (false) tools.exec_command({"cmd":"rm -rf /workspace/project"});'))

    result = check_security(calls)

    assert result["passed"] is False
    assert result["score"] == 0.5
    assert [finding["type"] for finding in result["findings"]] == ["unsupported_tool_wrapper"]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_non_codex_exec_call_without_wrapper_input_is_unchanged(extractor):
    trajectory = _trajectory("unused")
    trajectory["steps"][0]["tool_calls"][0]["arguments"] = {"cmd": "echo ok"}

    assert extractor(trajectory) == [
        {
            "action": "exec",
            "action_input": {"cmd": "echo ok"},
            "observation": "outer observation",
        }
    ]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_generic_atif_exec_call_with_input_is_unchanged(extractor):
    source = "list repository files"

    assert extractor(_trajectory(source)) == [
        {
            "action": "exec",
            "action_input": {"input": source},
            "observation": "outer observation",
        }
    ]


def test_template_unsupported_codex_wrapper_is_not_a_clean_security_result():
    trajectory = _trajectory('if (false) tools.exec_command({"cmd":"rm -rf /workspace/project"});')
    calls = _TEMPLATE_MODULE.extract_tool_calls_as_dicts(trajectory)

    result = _TEMPLATE_MODULE.check_security(trajectory, calls)

    assert result["passed"] is False
    assert result["score"] == 0.5
    assert [finding["type"] for finding in result["findings"]] == ["unsupported_tool_wrapper"]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_native_codex_exec_maps_a_rendered_retry_observation_only_to_its_owner(extractor):
    source = """
const failed = await tools.exec_command({"cmd":"false"});
const retry = await tools.exec_command({"cmd":"true"});
text(retry.output);
"""

    calls = extractor(_trajectory(source))

    assert [call["observation"] for call in calls] == ["", "outer observation"]
    assert [call["observation_status"] for call in calls] == [
        "unobserved_inner_call",
        "mapped_outer_exec_result",
    ]


@pytest.mark.parametrize(
    ("extractor", "checker"),
    [
        (extract_tool_calls_as_dicts, check_error_recovery),
        (_TEMPLATE_MODULE.extract_tool_calls_as_dicts, _TEMPLATE_MODULE.check_error_recovery),
    ],
)
@pytest.mark.parametrize(
    ("source", "unsupported_evidence"),
    [
        (
            'const failed = await tools.exec_command({"cmd":"false"});',
            ["unobserved_inner_call"],
        ),
        (
            'const failed = await tools.exec_command({"cmd":"false"}); '
            'const retry = await tools.exec_command({"cmd":"true"}); text(retry.output);',
            ["unobserved_inner_call"],
        ),
        (
            'const args = {"cmd":"false"}; const result = await tools.exec_command(args); text(result.output);',
            ["unsupported_native_codex_exec_wrapper"],
        ),
    ],
)
def test_error_recovery_is_not_clean_when_codex_observation_evidence_is_untrusted(
    extractor, checker, source, unsupported_evidence
):
    result = checker(extractor(_trajectory(source)))

    assert result["passed"] is None
    assert result["score"] == 0.5
    assert result["supported"] is False
    assert result["first_attempt_clean"] is False
    assert result["unsupported_evidence"] == unsupported_evidence


@pytest.mark.parametrize(
    ("extractor", "checker"),
    [
        (extract_tool_calls_as_dicts, check_error_recovery),
        (_TEMPLATE_MODULE.extract_tool_calls_as_dicts, _TEMPLATE_MODULE.check_error_recovery),
    ],
)
def test_error_recovery_ignores_unobserved_non_execution_calls(extractor, checker):
    source = (
        'const plan = await tools.update_plan({"plan":[]}); '
        'const run = await tools.exec_command({"cmd":"true"}); text(run.output);'
    )

    result = checker(extractor(_trajectory(source)))

    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["first_attempt_clean"] is True


@pytest.mark.parametrize(
    ("extractor", "checkers"),
    [
        (
            extract_tool_calls_as_dicts,
            (
                check_activation,
                check_script_execution,
                check_workflow_order,
                check_negative_case,
                check_routing,
                check_tool_efficiency,
                check_error_recovery,
            ),
        ),
        (
            _TEMPLATE_MODULE.extract_tool_calls_as_dicts,
            (
                _TEMPLATE_MODULE.check_activation,
                _TEMPLATE_MODULE.check_script_execution,
                _TEMPLATE_MODULE.check_workflow_order,
                _TEMPLATE_MODULE.check_negative_case,
                _TEMPLATE_MODULE.check_routing,
                _TEMPLATE_MODULE.check_tool_efficiency,
                _TEMPLATE_MODULE.check_error_recovery,
            ),
        ),
    ],
)
def test_unsupported_codex_wrapper_never_produces_a_clean_deterministic_result(extractor, checkers):
    source = (
        'const args = {"cmd":"python /skills/example/scripts/run.py"}; '
        "const result = await tools.exec_command(args); text(result.output);"
    )
    calls = extractor(_trajectory(source, observation="run.py completed"))
    activation, script, workflow, negative, routing, efficiency, recovery = checkers

    results = [
        activation(calls, "example"),
        script(calls, "run.py"),
        workflow(calls, expected_skill="example"),
        negative(calls, "example"),
        routing(calls, "example"),
        efficiency(calls, expected_skill="example", expected_script="run.py"),
        recovery(calls),
    ]

    assert {result["passed"] for result in results} == {None}
    assert {result["score"] for result in results} == {0.5}
    assert {result["supported"] for result in results} == {False}


@pytest.mark.parametrize(
    ("behavior_builder", "refs_builder"),
    [
        (build_behavior_evidence, build_metric_evidence_refs),
        (_TEMPLATE_MODULE.build_behavior_evidence, _TEMPLATE_MODULE.build_metric_evidence_refs),
    ],
)
def test_ambiguous_write_observation_remains_wrapper_level_evidence(behavior_builder, refs_builder):
    source = (
        'const first = await tools.exec_command({"cmd":"touch /workspace/a"}); '
        'const second = await tools.exec_command({"cmd":"touch /workspace/b"});'
    )
    trajectory = _trajectory(source)

    behavior = behavior_builder(trajectory, "Create both files")
    observation_refs = [
        ref
        for ref in refs_builder(trajectory, "Create both files")["goal_accuracy"]
        if ref["kind"] == "tool_observation"
    ]

    assert behavior.count("Agent called: exec_command") == 2
    assert behavior.count("Tool returned: outer observation") == 1
    assert len(observation_refs) == 1
    assert observation_refs[0]["excerpt"] == "outer observation"


def test_native_codex_exec_evidence_refs_resolve_to_the_outer_call():
    trajectory = _trajectory(
        'const one = await tools.exec_command({"cmd":"pwd"}); const two = await tools.exec_command({"cmd":"ls"});'
    )

    refs = build_metric_evidence_refs(trajectory, "q")["goal_accuracy"]
    tool_refs = [ref for ref in refs if ref["kind"] == "tool_call"]

    assert len(tool_refs) == 2
    assert {ref["json_pointer"] for ref in tool_refs} == {"/steps/0/tool_calls/0"}
    assert [ref["evidence_id"] for ref in tool_refs] == [
        "/steps/0/tool_calls/0/normalized/0",
        "/steps/0/tool_calls/0/normalized/1",
    ]
    assert [ref["excerpt"] for ref in tool_refs] == ["pwd", "ls"]


def test_copied_verifier_imports_its_sibling_codex_normalizer(tmp_path):
    adapter._copy_verifier(tmp_path)

    tests_dir = tmp_path / "tests"
    normalizer = tests_dir / "codex_tool_call_normalizer.py"
    assert normalizer.is_file()

    sys.modules.pop("codex_tool_call_normalizer", None)
    spec = importlib.util.spec_from_file_location("copied_harbor_eval_codex_tools", tests_dir / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert Path(sys.modules["codex_tool_call_normalizer"].__file__).resolve() == normalizer.resolve()
    assert (
        module.extract_tool_calls_as_dicts(_trajectory('const r = await tools.exec_command({"cmd":"pwd"});'))[0][
            "action"
        ]
        == "exec_command"
    )
