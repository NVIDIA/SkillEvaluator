# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core import checks, codex_tool_call_normalizer
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
        'const ratio = total / count;\ntools.exec_command({"cmd":"rm -rf /workspace/project"});',
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
        'tools?.[name]({"cmd":"rm -rf /workspace/project"});',
        '(tools)?.[name]({"cmd":"rm -rf /workspace/project"});',
        'globalThis.tools?.[name]({"cmd":"rm -rf /workspace/project"});',
        'globalThis?.["tools"]?.[name]({"cmd":"rm -rf /workspace/project"});',
        'const x = foo() / tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'console.log(tools[name]({"cmd":"rm -rf /workspace/project"}));',
        'consume(tools.exec_command({"cmd":"rm -rf /workspace/project"}));',
        'Promise.resolve(tools[name]({"cmd":"rm -rf /workspace/project"}));',
        '"use strict"; const r = await tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        '0; const r = await tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'result = await tools[name]({"cmd":"rm -rf /workspace/project"});',
        'start: tools[name]({"cmd":"rm -rf /workspace/project"});',
        'noop()\ntools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'debugger\ntools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'let unused\ntools.exec_command({"cmd":"rm -rf /workspace/project"});',
        "tools.exec_command(runtimeArguments);",
        "tools[name](runtimeArguments);",
        "(tools).exec_command(runtimeArguments);",
        'const tools = {}; globalThis.tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'tools.exec_command({"cmd":"rm -rf /workspace/project"}); function later(tools) { return tools; }',
        '{ let tools = {}; } tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'for (let tools of []) {} tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'for (const tools of []) {} tools[name]({"cmd":"rm -rf /workspace/project"});',
        'for (let tools = []; false;) {} tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'const r = await globalThis\n.tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        '(globalThis)\n["tools"].exec_command({"cmd":"rm -rf /workspace/project"});',
        '!tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'foo + tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'return foo + tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'foo * tools[name]({"cmd":"rm -rf /workspace/project"});',
        'foo != tools[name]({"cmd":"rm -rf /workspace/project"});',
        'foo !== tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'tools.\nexec_command({"cmd":"rm -rf /workspace/project"});',
        'tools.//comment\nexec_command({"cmd":"rm -rf /workspace/project"});',
        '(tools).\nexec_command({"cmd":"rm -rf /workspace/project"});',
        'noop()/*\n*/tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'const api = null ??\ntools; api.exec_command({"cmd":"rm -rf /workspace/project"});',
        'globalThis[`tools`][name]({"cmd":"rm -rf /workspace/project"});',
        'globalThis["t\\x6fols"][name]({"cmd":"rm -rf /workspace/project"});',
        'globalThis[("tools")].exec_command({"cmd":"rm -rf /workspace/project"});',
        'globalThis["to\\\nols"].exec_command({"cmd":"rm -rf /workspace/project"});',
        'globalThis["to\\\rols"].exec_command({"cmd":"rm -rf /workspace/project"});',
        'globalThis["to\\\r\nols"].exec_command({"cmd":"rm -rf /workspace/project"});',
        'globalThis["to\\\u2028ols"].exec_command({"cmd":"rm -rf /workspace/project"});',
        '// harmless\u2028tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        '// harmless\u2029tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'function f(x = tools.exec_command({"cmd":"rm -rf /workspace/project"})) {}',
        'const f = (x = tools[name]({"cmd":"rm -rf /workspace/project"})) => x;',
        '{ tools.exec_command({"cmd":"rm -rf /workspace/project"}); }',
        'try { tools.exec_command({"cmd":"rm -rf /workspace/project"}); } catch (error) {}',
        'do { tools.exec_command({"cmd":"rm -rf /workspace/project"}); } while (false);',
        'if (false) noop(); else tools.exec_command({"cmd":"rm -rf /workspace/project"});',
        'do tools.exec_command({"cmd":"rm -rf /workspace/project"}); while (false);',
        'class X extends tools.exec_command({"cmd":"rm -rf /workspace/project"}) {}',
        'switch (x) { case 1: break; case tools[name]({"cmd":"rm -rf /workspace/project"}): break; }',
        '[...tools.exec_command({"cmd":"rm -rf /workspace/project"})];',
        '({...tools.exec_command({"cmd":"rm -rf /workspace/project"})});',
        (
            "const plan = await tools.update_plan({plan:[]}); "
            'tools["exec_command"]({"cmd":"rm -rf /workspace/project"});'
        ),
        (
            "const plan = await tools.update_plan({plan:[]}); "
            'const value = `${tools.exec_command({"cmd":"rm -rf /workspace/project"})}`;'
        ),
        pytest.param("const result = " + "tools[" * 129 + "x", id="repeated-unclosed-dynamic-members"),
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


@pytest.mark.parametrize(
    "source",
    (
        pytest.param("list repository files", id="plain-input"),
        pytest.param("list repository files with the available tools.", id="sentence-ending-tools"),
        pytest.param("describe tools.exec_command before using it", id="tool-property-prose"),
        pytest.param("describe tools[name] before using it", id="computed-tool-property-prose"),
        pytest.param('print the literal string "tools[name]"', id="computed-tool-property-quoted-prose"),
        pytest.param("Please describe tools[name] (if available).", id="computed-tool-parenthetical-prose"),
        pytest.param("tools[name] is notation, not an invocation.", id="computed-tool-leading-prose"),
        pytest.param("Describe tools[name]; do not invoke it.", id="computed-tool-semicolon-prose"),
        pytest.param("compare tools[name] = dynamic access", id="computed-tool-assignment-prose"),
        pytest.param("python -c 'print(tools[name])'", id="computed-tool-shell-command"),
        pytest.param(
            'print the literal string "const fn = tools.exec_command;"',
            id="direct-tool-code-quoted-prose",
        ),
        pytest.param(
            "if available, describe tools[name] before using it",
            id="leading-if-prose",
        ),
        pytest.param(
            "return documentation for tools.exec_command before using it",
            id="leading-return-prose",
        ),
        pytest.param(
            "for each tools[name], explain its purpose",
            id="leading-for-prose",
        ),
        pytest.param(
            "const means constant; describe tools[name]",
            id="leading-const-prose",
        ),
        pytest.param(
            'const text = "tools.exec_command({\\"cmd\\":\\"rm -rf /workspace/project\\"})";',
            id="tool-call-inside-js-string",
        ),
        pytest.param(
            'const pattern = /tools.exec_command\\({"cmd":"rm -rf \\/workspace\\/project"}\\)/;',
            id="tool-call-inside-js-regex",
        ),
        pytest.param(
            'if (enabled) /tools.exec_command\\({"cmd":"rm -rf \\/workspace\\/project"}\\)/.test(input);',
            id="tool-call-inside-js-regex-statement",
        ),
        pytest.param(
            'const nested = other.tools.exec_command({"cmd":"rm -rf /workspace/project"});',
            id="non-global-tools-property",
        ),
        pytest.param(
            '// tools.exec_command({"cmd":"rm -rf /workspace/project"});',
            id="tool-call-inside-js-comment",
        ),
        pytest.param(
            "return /tools.exec_command\\({}\\)/;",
            id="tool-call-inside-returned-regex",
        ),
        pytest.param(
            "function f() { return /tools.exec_command\\({}\\)/; }",
            id="tool-call-inside-function-returned-regex",
        ),
        pytest.param(
            "function* f() { yield /tools[name]\\({}\\)/; }",
            id="tool-call-inside-yielded-regex",
        ),
        pytest.param(
            "const pattern =\n/tools.exec_command\\({}\\)/;",
            id="tool-call-inside-multiline-assigned-regex",
        ),
        pytest.param(
            "const patterns = [\n/tools.exec_command\\({}\\)/\n];",
            id="tool-call-inside-multiline-array-regex",
        ),
        pytest.param(
            "const pattern = (\n/tools.exec_command\\({}\\)/\n);",
            id="tool-call-inside-multiline-parenthesized-regex",
        ),
        pytest.param("await /tools.exec_command\\({}\\)/;", id="tool-call-inside-awaited-regex"),
        pytest.param("typeof /tools.exec_command\\({}\\)/;", id="tool-call-inside-typeof-regex"),
        pytest.param("void /tools.exec_command\\({}\\)/;", id="tool-call-inside-void-regex"),
        pytest.param("delete /tools.exec_command\\({}\\)/;", id="tool-call-inside-delete-regex"),
        pytest.param(
            "const pattern = 1 + /tools.exec_command\\({}\\)/;",
            id="tool-call-inside-binary-expression-regex",
        ),
        pytest.param(
            "if (enabled) {} /tools.exec_command\\({}\\)/.test(input);",
            id="tool-call-inside-post-block-regex",
        ),
        pytest.param(
            "if (enabled) {} else /tools.exec_command\\({}\\)/.test(input);",
            id="tool-call-inside-else-regex",
        ),
        pytest.param(
            "do /tools.exec_command\\({}\\)/.test(input); while (false);",
            id="tool-call-inside-do-regex",
        ),
        pytest.param(
            '"x" in /tools.exec_command\\({}\\)/;',
            id="tool-call-inside-in-expression-regex",
        ),
        pytest.param(
            "for (const key in /tools.exec_command\\({}\\)/) {}",
            id="tool-call-inside-for-in-regex",
        ),
        pytest.param(
            'const config = {tools: ["hammer"]};',
            id="tools-object-key",
        ),
        pytest.param(
            "if test -e 'tools[name]'; then echo ok; fi",
            id="shell-if-command",
        ),
        pytest.param(
            "true; printf '%s' 'tools[name]'",
            id="shell-true-command",
        ),
        pytest.param(
            "// explain tools[name] without invoking it",
            id="commented-prose",
        ),
    ),
)
@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_generic_atif_exec_call_with_input_is_unchanged(source, extractor):
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


@pytest.mark.parametrize("template", [False, True], ids=["shared", "standalone-template"])
@pytest.mark.parametrize(
    "tool_reference",
    [
        pytest.param("tools[name]", id="computed-member"),
        pytest.param("tools?.[name]", id="optional-computed-member"),
        pytest.param("tools/*comment*/[name]", id="comment-separated-computed-member"),
        pytest.param("(tools)[name]", id="parenthesized-computed-member"),
        pytest.param("globalThis.tools?.[name]", id="global-optional-computed-member"),
        pytest.param("tools/*comment*/.exec_command", id="comment-before-direct-member"),
        pytest.param("tools./*comment*/exec_command", id="comment-after-direct-member"),
        pytest.param("tools.exec_command/*comment*/", id="comment-before-direct-call"),
        pytest.param("tools?.exec_command", id="optional-direct-member"),
        pytest.param("tools.exec_command?.", id="optional-direct-call"),
        pytest.param("(tools).exec_command", id="parenthesized-direct-member"),
        pytest.param("(tools)?.exec_command", id="parenthesized-optional-direct-member"),
        pytest.param("(tools.exec_command)", id="parenthesized-direct-call"),
        pytest.param('globalThis["tools"][name]', id="global-computed-tools-object"),
        pytest.param(r"t\u006fols.exec_command", id="escaped-tools-identifier"),
        pytest.param(r"tools.\u0065xec_command", id="escaped-method-identifier"),
        pytest.param("tools//comment\r[name]", id="cr-line-comment-separated-member"),
        pytest.param('(globalThis)["tools"][name]', id="parenthesized-global-computed-tools"),
        pytest.param("(globalThis).tools[name]", id="parenthesized-global-direct-tools"),
        pytest.param('((globalThis))["tools"].exec_command', id="multiply-parenthesized-global-tools"),
    ],
)
def test_dynamic_computed_codex_tool_member_fails_closed(template, tool_reference):
    source = (
        'const name = "exec_command"; '
        f'const result = await {tool_reference}({{"cmd":"rm -rf /workspace/project"}}); '
        "text(result.output);"
    )
    trajectory = _trajectory(source)
    if template:
        calls = _TEMPLATE_MODULE.extract_tool_calls_as_dicts(trajectory)
        result = _TEMPLATE_MODULE.check_security(trajectory, calls)
    else:
        calls = extract_tool_calls_as_dicts(trajectory)
        result = check_security(calls)

    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"
    assert result["passed"] is False
    assert result["score"] == 0.5
    assert [finding["type"] for finding in result["findings"]] == ["unsupported_tool_wrapper"]


@pytest.mark.parametrize("template", [False, True], ids=["shared", "standalone-template"])
@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            'const fn = tools.exec_command; const result = await fn({"cmd":"rm -rf /workspace/project"});',
            id="aliased-method",
        ),
        pytest.param(
            'const {exec_command} = tools; '
            'const result = await exec_command({"cmd":"rm -rf /workspace/project"});',
            id="destructured-method",
        ),
        pytest.param(
            'const name = "exec_command"; const api = tools; '
            'const result = await api[name]({"cmd":"rm -rf /workspace/project"});',
            id="aliased-tools-object",
        ),
        pytest.param(
            'const url = "http://example.invalid"; const name = "exec_command"; '
            'const result = await tools/*comment*/[name]({"cmd":"rm -rf /workspace/project"});',
            id="comment-separated-member-after-url-string",
        ),
        pytest.param(
            'const url = `http://example.invalid`; const name = "exec_command"; '
            'const result = await tools/*comment*/[name]({"cmd":"rm -rf /workspace/project"});',
            id="comment-separated-member-after-template-url",
        ),
        pytest.param(
            "const pattern = /'/; const name = \"exec_command\"; "
            'const result = await tools/*comment*/[name]({"cmd":"rm -rf /workspace/project"});',
            id="comment-separated-member-after-regex-quote",
        ),
        pytest.param(
            'let api; api = tools; const name = "exec_command"; '
            'const result = await api[name]({"cmd":"rm -rf /workspace/project"});',
            id="assigned-tools-object",
        ),
        pytest.param(
            'let exec_command; ({exec_command} = tools); '
            'const result = await exec_command({"cmd":"rm -rf /workspace/project"});',
            id="destructuring-assignment",
        ),
        pytest.param(
            'const api = enabled ? tools : {}; '
            'api.exec_command({"cmd":"rm -rf /workspace/project"});',
            id="ternary-tools-alias",
        ),
    ],
)
def test_indirect_codex_tool_reference_fails_closed(template, source):
    trajectory = _trajectory(source)
    if template:
        calls = _TEMPLATE_MODULE.extract_tool_calls_as_dicts(trajectory)
        result = _TEMPLATE_MODULE.check_security(trajectory, calls)
    else:
        calls = extract_tool_calls_as_dicts(trajectory)
        result = check_security(calls)

    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"
    assert result["passed"] is False
    assert result["score"] == 0.5
    assert [finding["type"] for finding in result["findings"]] == ["unsupported_tool_wrapper"]


@pytest.mark.parametrize("template", [False, True], ids=["shared", "standalone-template"])
def test_unsupported_codex_wrapper_still_scans_its_outer_observation_for_secrets(template):
    trajectory = _trajectory(
        'if (false) tools.exec_command({"cmd":"pwd"});',
        observation="Authorization: Bearer sk-abcdefgh12345678",
    )
    if template:
        calls = _TEMPLATE_MODULE.extract_tool_calls_as_dicts(trajectory)
        result = _TEMPLATE_MODULE.check_security(trajectory, calls)
    else:
        calls = extract_tool_calls_as_dicts(trajectory)
        result = check_security(calls)

    assert result["passed"] is False
    assert result["score"] == 0.0
    assert [finding["type"] for finding in result["findings"]] == [
        "unsupported_tool_wrapper",
        "secret_exposure",
    ]


@pytest.mark.parametrize("template", [False, True], ids=["shared", "standalone-template"])
@pytest.mark.parametrize(
    ("rendered", "expected_status"),
    [
        ("text(one.output); text(two.output);", "ambiguous_outer_exec_result"),
        ("", "unobserved_inner_call"),
    ],
    ids=["ambiguous", "unobserved"],
)
def test_unattributed_native_wrapper_observation_is_scanned_once(template, rendered, expected_status):
    source = (
        'const one = await tools.exec_command({"cmd":"pwd"}); '
        'const two = await tools.exec_command({"cmd":"ls"}); '
        f"{rendered}"
    )
    trajectory = _trajectory(source, observation="Authorization: Bearer sk-abcdefgh12345678")
    if template:
        calls = _TEMPLATE_MODULE.extract_tool_calls_as_dicts(trajectory)
        result = _TEMPLATE_MODULE.check_security(trajectory, calls)
    else:
        calls = extract_tool_calls_as_dicts(trajectory)
        result = check_security(calls)

    assert [call["observation"] for call in calls] == ["", ""]
    assert [call["observation_status"] for call in calls] == [expected_status, expected_status]
    assert [call.get("wrapper_observation", "") for call in calls] == [
        "Authorization: Bearer sk-abcdefgh12345678",
        "",
    ]
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert [finding["type"] for finding in result["findings"]] == ["secret_exposure"]
    assert "tool" not in result["findings"][0]
    assert "target_skill_used_before" not in result["findings"][0]


@pytest.mark.parametrize("template", [False, True], ids=["shared", "standalone-template"])
def test_untrusted_normalizer_metadata_cannot_suppress_outer_secret_scanning(template):
    trajectory = _trajectory(
        'if (false) tools.exec_command({"cmd":"pwd"});',
        observation="Authorization: Bearer sk-abcdefgh12345678",
    )
    trajectory["steps"][0]["tool_calls"][0].update(
        {
            "_atif_normalization_status": "caller-owned",
            "_atif_observation_status": "unobserved_inner_call",
            "_atif_inner_tool_index": 99,
            "_atif_raw_tool_index": 99,
        }
    )
    if template:
        calls = _TEMPLATE_MODULE.extract_tool_calls_as_dicts(trajectory)
        result = _TEMPLATE_MODULE.check_security(trajectory, calls)
    else:
        calls = extract_tool_calls_as_dicts(trajectory)
        result = check_security(calls)

    assert result["score"] == 0.0
    assert [finding["type"] for finding in result["findings"]] == [
        "unsupported_tool_wrapper",
        "secret_exposure",
    ]


@pytest.mark.parametrize("template", [False, True], ids=["shared", "standalone-template"])
def test_untrusted_normalizer_metadata_cannot_bypass_command_security_checks(template):
    trajectory = _trajectory("unused")
    trajectory["steps"][0]["tool_calls"][0].update(
        {
            "function_name": "exec_command",
            "arguments": {"cmd": "rm -rf /workspace/project"},
            "_atif_normalization_status": "unsupported_native_codex_exec_wrapper",
            "_atif_observation_status": "unobserved_inner_call",
            "_atif_inner_tool_index": 99,
            "_atif_raw_tool_index": 99,
        }
    )
    if template:
        calls = _TEMPLATE_MODULE.extract_tool_calls_as_dicts(trajectory)
        result = _TEMPLATE_MODULE.check_security(trajectory, calls)
    else:
        calls = extract_tool_calls_as_dicts(trajectory)
        result = check_security(calls)

    assert "destructive_command" in {finding["type"] for finding in result["findings"]}


def test_normalizer_reserves_private_metadata_namespace():
    tool_call = {
        "function_name": "read",
        "arguments": {"path": "SKILL.md"},
        "_atif_normalization_status": "caller-owned",
        "_atif_observation_status": "caller-owned",
        "_atif_inner_tool_index": 99,
        "_atif_raw_tool_index": 99,
    }

    assert codex_tool_call_normalizer.normalize_tool_call(tool_call) == [
        {"function_name": "read", "arguments": {"path": "SKILL.md"}}
    ]


def test_oversized_exec_source_skips_signature_scans(monkeypatch):
    class UnexpectedScan:
        def match(self, _source):
            pytest.fail("oversized source reached pragma detection")

        def search(self, _source):
            pytest.fail("oversized source reached tool-reference detection")

    monkeypatch.setattr(codex_tool_call_normalizer, "_CODEX_PRAGMA_RE", UnexpectedScan())
    monkeypatch.setattr(codex_tool_call_normalizer, "_CODEX_TOOL_REF_RE", UnexpectedScan())
    source = " " * 65537 + 'tools.exec_command({"cmd":"pwd"});'

    assert codex_tool_call_normalizer.normalize_tool_call(
        {"function_name": "exec", "arguments": {"input": source}}
    ) == [
        {
            "function_name": "exec",
            "arguments": {"input": source},
            "_atif_normalization_status": "unsupported_native_codex_exec_wrapper",
        }
    ]


def test_unterminated_regex_candidate_is_scanned_once(monkeypatch):
    original = codex_tool_call_normalizer._skip_js_regex
    calls = 0

    def counted_skip(source, start):
        nonlocal calls
        calls += 1
        return original(source, start)

    monkeypatch.setattr(codex_tool_call_normalizer, "_skip_js_regex", counted_skip)
    source = "/[" * 30_000 + "tools"

    assert codex_tool_call_normalizer.normalize_tool_call(
        {"function_name": "exec", "arguments": {"input": source}}
    ) == [{"function_name": "exec", "arguments": {"input": source}}]
    assert calls == 1


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "const r = await tools.exec_command(" + '{"nested":' * 65 + "0" + "}" * 65 + ");",
            id="excessive-nesting",
        ),
        pytest.param(
            'const r = await tools.exec_command({"value":' + "1" * 5000 + "});",
            id="oversized-integer",
        ),
        pytest.param(
            "".join(f'const r{i} = await tools.exec_command({{"cmd":"pwd {i}"}});' for i in range(129)),
            id="excessive-calls",
        ),
        pytest.param(
            'const r = await tools.exec_command({"cmd":"pwd"});' + "text(r.output);" * 256,
            id="excessive-statements",
        ),
        pytest.param(
            'const r = await tools.exec_command({"cmd":"pwd"});' + " " * 65536,
            id="oversized-source",
        ),
    ],
)
@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_native_codex_exec_parser_limits_fail_closed(source, extractor):
    calls = extractor(_trajectory(source))

    assert calls == [
        {
            "action": "exec",
            "action_input": {"input": source},
            "observation": "outer observation",
            "normalization_status": "unsupported_native_codex_exec_wrapper",
        }
    ]


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


@pytest.mark.parametrize("template", [False, True], ids=["shared", "standalone-template"])
def test_mapped_outer_observation_is_not_duplicated_as_wrapper_evidence(template):
    source = (
        'const failed = await tools.exec_command({"cmd":"false"}); '
        'const retry = await tools.exec_command({"cmd":"true"}); '
        "text(retry.output);"
    )
    trajectory = _trajectory(source, observation="Authorization: Bearer sk-abcdefgh12345678")
    if template:
        calls = _TEMPLATE_MODULE.extract_tool_calls_as_dicts(trajectory)
        result = _TEMPLATE_MODULE.check_security(trajectory, calls)
    else:
        calls = extract_tool_calls_as_dicts(trajectory)
        result = check_security(calls)

    assert [call["observation"] for call in calls] == [
        "",
        "Authorization: Bearer sk-abcdefgh12345678",
    ]
    assert [call.get("wrapper_observation", "") for call in calls] == ["", ""]
    assert [finding["type"] for finding in result["findings"]] == ["secret_exposure"]


@pytest.mark.parametrize("template", [False, True], ids=["shared", "standalone-template"])
@pytest.mark.parametrize(
    ("source_call_id", "wrapper_owner"),
    [(None, 0), ("outer-two", 2)],
    ids=["unscoped", "scoped-to-second-wrapper"],
)
def test_multi_wrapper_observation_is_scanned_once_at_the_narrowest_known_scope(
    template, source_call_id, wrapper_owner
):
    source = (
        'const one = await tools.exec_command({"cmd":"pwd"}); '
        'const two = await tools.exec_command({"cmd":"ls"});'
    )
    result_entry = {"content": "Authorization: Bearer sk-abcdefgh12345678"}
    if source_call_id is not None:
        result_entry["source_call_id"] = source_call_id
    trajectory = {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {"tool_call_id": "outer-one", "function_name": "exec", "arguments": {"input": source}},
                    {"tool_call_id": "outer-two", "function_name": "exec", "arguments": {"input": source}},
                ],
                "observation": {"results": [result_entry]},
            }
        ]
    }
    if template:
        calls = _TEMPLATE_MODULE.extract_tool_calls_as_dicts(trajectory)
        security = _TEMPLATE_MODULE.check_security(trajectory, calls)
    else:
        calls = extract_tool_calls_as_dicts(trajectory)
        security = check_security(calls)

    expected_wrapper_observations = ["", "", "", ""]
    expected_wrapper_observations[wrapper_owner] = "Authorization: Bearer sk-abcdefgh12345678"
    assert [call.get("wrapper_observation", "") for call in calls] == expected_wrapper_observations
    assert [finding["type"] for finding in security["findings"]] == ["secret_exposure"]
    assert "tool" not in security["findings"][0]


@pytest.mark.parametrize("template", [False, True], ids=["shared", "standalone-template"])
@pytest.mark.parametrize(
    "call_ids",
    [("outer-one", "outer-two"), (None, None), ("", "")],
    ids=["with-call-ids", "without-call-ids", "with-empty-call-ids"],
)
@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            'const result = await tools.exec_command({"cmd":"pwd"}); text(result.output);',
            id="mapped",
        ),
        pytest.param(
            'const name = "exec_command"; const result = await tools[name]({"cmd":"pwd"});',
            id="unsupported",
        ),
    ],
)
def test_unscoped_multi_wrapper_observation_is_not_duplicated_for_atomic_calls(
    template, call_ids, source
):
    secret = "Authorization: Bearer sk-abcdefgh12345678"
    trajectory = {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        **({"tool_call_id": call_ids[0]} if call_ids[0] is not None else {}),
                        "function_name": "exec",
                        "arguments": {"input": source},
                    },
                    {
                        **({"tool_call_id": call_ids[1]} if call_ids[1] is not None else {}),
                        "function_name": "exec",
                        "arguments": {"input": source},
                    },
                ],
                "observation": {"results": [{"content": secret}]},
            }
        ]
    }
    if template:
        calls = _TEMPLATE_MODULE.extract_tool_calls_as_dicts(trajectory)
        security = _TEMPLATE_MODULE.check_security(trajectory, calls)
    else:
        calls = extract_tool_calls_as_dicts(trajectory)
        security = check_security(calls)

    assert [call["observation"] for call in calls] == [secret, ""]
    assert [finding["type"] for finding in security["findings"]].count("secret_exposure") == 1


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

    tests_dir = tmp_path / "tests" / adapter._EVALUATOR_TESTS_SUBDIR
    normalizer = tests_dir / "codex_tool_call_normalizer.py"
    evidence = tests_dir / "evidence.py"
    assert normalizer.is_file()
    assert evidence.is_file()

    sys.modules.pop("codex_tool_call_normalizer", None)
    sys.modules.pop("evidence", None)
    spec = importlib.util.spec_from_file_location("copied_harbor_eval_codex_tools", tests_dir / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert Path(sys.modules["codex_tool_call_normalizer"].__file__).resolve() == normalizer.resolve()
    assert Path(sys.modules["evidence"].__file__).resolve() == evidence.resolve()
    assert (
        module.extract_tool_calls_as_dicts(_trajectory('const r = await tools.exec_command({"cmd":"pwd"});'))[0][
            "action"
        ]
        == "exec_command"
    )


def _multimodal_trajectory() -> dict:
    return {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [{"tool_call_id": "call-1", "function_name": "view_image", "arguments": {}}],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call-1",
                            "content": [
                                {"type": "text", "text": "rendered chart"},
                                {"type": "image", "source": {"media_type": "image/png", "path": "/secret/a.png"}},
                                {"type": "audio", "source": {"media_type": "audio/wav", "path": "/secret/a.wav"}},
                                {"type": "video"},
                            ],
                        }
                    ]
                },
            }
        ]
    }


def test_atif_content_text_flattens_parts_without_source_paths() -> None:
    content = _multimodal_trajectory()["steps"][0]["observation"]["results"][0]["content"]

    assert codex_tool_call_normalizer.atif_content_text(None) == ""
    assert codex_tool_call_normalizer.atif_content_text("plain") == "plain"
    assert codex_tool_call_normalizer.atif_content_text(content) == "rendered chart\n[image]\n[audio]"
    assert "/secret" not in codex_tool_call_normalizer.atif_content_text(content)


@pytest.mark.parametrize("extract", _EXTRACTORS)
def test_list_observation_content_is_text_not_repr(extract) -> None:
    [call] = extract(_multimodal_trajectory())

    assert call["observation"] == "rendered chart\n[image]\n[audio]"


def test_host_and_template_summaries_agree_on_list_observations() -> None:
    from skillevaluator.tier3.eval_core import atif_helpers

    trajectory = _multimodal_trajectory()

    host = atif_helpers.build_conversation_summary(trajectory, "question")
    template = _TEMPLATE_MODULE.build_conversation_summary(trajectory, "question")
    assert host == template
    assert "Tool returned: rendered chart\n[image]\n[audio]" in host
    assert "/secret" not in host
    assert atif_helpers._late_observation_excerpts(trajectory, 200) == _TEMPLATE_MODULE._late_observation_excerpts(
        trajectory, 200
    )


# ── Extended static grammar for inline and Promise.allSettled wrappers (#235) ──

_ALL_SETTLED_SOURCE = (
    "const results=await Promise.allSettled(["
    'tools.exec_command({cmd:"pwd"}),'
    'tools.write_stdin({session_id:1,chars:""})'
    "]);"
)
_ALL_SETTLED_SPREAD_RECORDS = [
    {"i": 0, "status": "fulfilled", "value": {"exit_code": 0, "output": "/workspace\n"}},
    {"i": 1, "status": "rejected", "reason": {"error": "boom"}},
]
_ALL_SETTLED_FIELD_RECORDS = [
    {"i": index, "result": {key: value for key, value in record.items() if key != "i"}}
    for index, record in enumerate(_ALL_SETTLED_SPREAD_RECORDS)
]
_ALL_SETTLED_RAW_RECORDS = [
    {"status": "fulfilled", "value": {"exit_code": 0, "output": "/workspace\n"}},
    {"status": "rejected", "reason": {"error": "boom"}},
]


def _expected_records(records: list[dict]) -> list[str]:
    """Each record's JSON text, plus the failed-status line a rejected promise gains."""
    return [
        json.dumps(record, separators=(",", ":")) + ("\nstatus=failed" if "rejected" in str(record) else "")
        for record in records
    ]


_RECORD_FRAME = ("Script completed\nWall time 0.1 seconds\nOutput:\n\n", "")


def _record_observation(records: list[dict]) -> str:
    body = "\n".join(json.dumps(record, separators=(",", ":")) for record in records)
    return "Script completed\nWall time 0.1 seconds\nOutput:\n\n" + body


def _security_result(source: str, extractor):
    trajectory = _trajectory(source)
    calls = extractor(trajectory)
    if extractor is _TEMPLATE_MODULE.extract_tool_calls_as_dicts:
        return _TEMPLATE_MODULE.check_security(trajectory, calls)
    return check_security(calls)


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "renderer, records",
    [
        pytest.param(
            "results.forEach((r,i)=>text({i,...r}));",
            _ALL_SETTLED_SPREAD_RECORDS,
            id="forEach-spread",
        ),
        pytest.param(
            "results.forEach((r, i) => { text({i, ...r}); });",
            _ALL_SETTLED_SPREAD_RECORDS,
            id="forEach-spread-braced",
        ),
        pytest.param(
            "for (let i = 0; i < results.length; i++) text({i, ...results[i]});",
            _ALL_SETTLED_SPREAD_RECORDS,
            id="for-spread",
        ),
        pytest.param(
            "for (let i = 0; i < results.length; i++) { text({i, ...results[i]}); }",
            _ALL_SETTLED_SPREAD_RECORDS,
            id="for-spread-braced",
        ),
        pytest.param("results.forEach(text);", _ALL_SETTLED_RAW_RECORDS, id="forEach-text"),
        pytest.param("results.forEach(r => text(r));", _ALL_SETTLED_RAW_RECORDS, id="forEach-arrow-text"),
        pytest.param(
            "results.forEach((r,i)=>text({i,result:r}));",
            _ALL_SETTLED_FIELD_RECORDS,
            id="forEach-field",
        ),
    ],
)
def test_extended_codex_allsettled_maps_each_call_to_its_record(extractor, renderer, records):
    calls = extractor(_trajectory(_ALL_SETTLED_SOURCE + renderer, _record_observation(records)))

    assert [call["action"] for call in calls] == ["exec_command", "write_stdin"]
    assert calls[0]["action_input"] == {"cmd": "pwd"}
    assert calls[1]["action_input"] == {"session_id": 1, "chars": ""}
    assert [call["observation_status"] for call in calls] == ["mapped_outer_exec_result"] * 2
    assert [call["observation"] for call in calls] == _expected_records(records)


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "source",
    [
        pytest.param('text(await tools.exec_command({cmd:"pwd"}));', id="inline-await"),
        pytest.param("const r=await tools.exec_command({cmd:'pwd'});text(r);", id="single-quoted-string"),
        pytest.param("const r=await tools.exec_command({'cmd':'pwd'});text(r);", id="single-quoted-key"),
        pytest.param('text((await tools.exec_command({cmd:"pwd"})).output);', id="inline-output"),
        pytest.param("const r=await tools.exec_command({cmd:'pwd'});text(r.output);", id="bound-output"),
        pytest.param('text(await tools.exec_command({cmd:"pwd"}))', id="inline-without-final-semicolon"),
        pytest.param('const r=await tools.exec_command({cmd:"pwd"});text(r)', id="bound-without-final-semicolon"),
    ],
)
def test_extended_codex_inline_and_single_quoted_wrappers_normalize(extractor, source):
    calls = extractor(_trajectory(source))

    assert [call["action"] for call in calls] == ["exec_command"]
    assert calls[0]["action_input"] == {"cmd": "pwd"}
    assert calls[0]["observation_status"] == "mapped_outer_exec_result"
    assert "normalization_status" not in calls[0]


@pytest.mark.parametrize(
    "source, expected",
    [
        pytest.param(
            r"const r=await tools.exec_command({cmd:'echo \'hi\''});text(r);",
            "echo 'hi'",
            id="escaped-single-quote",
        ),
        pytest.param(
            r"""const r=await tools.exec_command({cmd:'say "hi"'});text(r);""",
            'say "hi"',
            id="double-quote-inside-single",
        ),
    ],
)
def test_extended_codex_single_quoted_string_escapes(source, expected):
    calls = extract_tool_calls_as_dicts(_trajectory(source))

    assert calls[0]["action_input"] == {"cmd": expected}


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            r"const p=await tools.apply_patch('*** Begin Patch\n*** End Patch');text(p);",
            id="single-quoted",
        ),
        pytest.param(
            r'const p=await tools.apply_patch("*** Begin Patch\n*** End Patch");text(p);',
            id="double-quoted",
        ),
        pytest.param(
            r"text(await tools.apply_patch('*** Begin Patch\n*** End Patch'));",
            id="inline",
        ),
    ],
)
def test_extended_codex_apply_patch_string_literal_normalizes(extractor, source):
    calls = extractor(_trajectory(source))

    assert [call["action"] for call in calls] == ["apply_patch"]
    assert calls[0]["action_input"] == {"input": "*** Begin Patch\n*** End Patch"}


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "source",
    [
        pytest.param('const cmd="pwd";text(await tools.exec_command({cmd}));', id="dynamic-argument"),
        pytest.param('text(await tools.exec_command({["cmd"]:"pwd"}));', id="computed-key"),
        pytest.param("text(await tools.exec_command({...args}));", id="spread-arguments"),
        pytest.param("text(await tools.exec_command({cmd:`pwd`}));", id="template-literal"),
        pytest.param('text(await tools.exec_command({cmd:"pwd"})); // comment', id="trailing-comment"),
        pytest.param('const text=1;text(await tools.exec_command({cmd:"pwd"}));', id="shadowed-global"),
        pytest.param(
            'const r=await tools.exec_command({cmd:"pwd"});const r=await tools.exec_command({cmd:"ls"});text(r);',
            id="duplicate-binding",
        ),
        pytest.param(
            'text(await tools.exec_command({cmd:"pwd"}));text(await tools.exec_command({cmd:"ls"}));',
            id="multi-inline-without-records",
        ),
        pytest.param(_ALL_SETTLED_SOURCE + "results.map((r,i)=>text({i,...r}));", id="unsupported-renderer-map"),
        pytest.param(_ALL_SETTLED_SOURCE + "for (const r of results) text(r);", id="unsupported-renderer-for-of"),
    ],
)
def test_extended_codex_dynamic_and_ambiguous_forms_stay_atomic(extractor, source):
    calls = extractor(_trajectory(source))

    assert len(calls) == 1
    assert calls[0]["action"] == "exec"
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "records",
    [
        pytest.param(_ALL_SETTLED_SPREAD_RECORDS[:1], id="missing-record"),
        pytest.param(
            [*_ALL_SETTLED_SPREAD_RECORDS, {"i": 2, "status": "fulfilled", "value": {}}],
            id="extra-record",
        ),
        pytest.param(list(reversed(_ALL_SETTLED_SPREAD_RECORDS)), id="reordered-records"),
        pytest.param([_ALL_SETTLED_SPREAD_RECORDS[0], _ALL_SETTLED_SPREAD_RECORDS[0]], id="duplicate-index"),
        pytest.param([{"i": 0, "status": "fulfilled"}, _ALL_SETTLED_SPREAD_RECORDS[1]], id="malformed-record"),
        pytest.param(
            [
                {**_ALL_SETTLED_SPREAD_RECORDS[0], "extra": True},
                _ALL_SETTLED_SPREAD_RECORDS[1],
            ],
            id="extra-key",
        ),
    ],
)
def test_extended_codex_untrusted_allsettled_records_fall_back_to_atomic(extractor, records):
    source = _ALL_SETTLED_SOURCE + "results.forEach((r,i)=>text({i,...r}));"
    calls = extractor(_trajectory(source, _record_observation(records)))

    assert len(calls) == 1
    assert calls[0]["action"] == "exec"
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


@pytest.mark.parametrize(
    "observation",
    [
        pytest.param("", id="empty"),
        pytest.param("partial", id="without-output-header"),
        pytest.param(_record_observation(_ALL_SETTLED_SPREAD_RECORDS).rsplit("\n", 1)[0], id="truncated"),
        pytest.param(_record_observation(_ALL_SETTLED_SPREAD_RECORDS) + "\nextra", id="trailing-text"),
    ],
)
def test_extended_codex_incomplete_allsettled_observation_falls_back_to_atomic(observation):
    source = _ALL_SETTLED_SOURCE + "results.forEach((r,i)=>text({i,...r}));"
    calls = extract_tool_calls_as_dicts(_trajectory(source, observation))

    assert len(calls) == 1
    assert calls[0]["action"] == "exec"
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


def test_extended_codex_rejected_allsettled_record_is_an_error():
    source = _ALL_SETTLED_SOURCE + "results.forEach((r,i)=>text({i,...r}));"
    trajectory = _trajectory(source, _record_observation(_ALL_SETTLED_SPREAD_RECORDS))
    yielded = list(codex_tool_call_normalizer.iter_normalized_tool_calls(trajectory))

    assert [codex_tool_call_normalizer.normalized_tool_call_is_error(step, call) for step, call in yielded] == [
        False,
        True,
    ]
    assert [codex_tool_call_normalizer.normalized_tool_call_error_flag(step, call) for step, call in yielded] == [
        False,
        True,
    ]
    assert [
        codex_tool_call_normalizer.normalized_tool_call_observation(step, call) for step, call in yielded
    ] == _expected_records(_ALL_SETTLED_SPREAD_RECORDS)


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_extended_codex_extracted_records_carry_is_error(extractor):
    source = _ALL_SETTLED_SOURCE + "results.forEach((r,i)=>text({i,...r}));"
    calls = extractor(_trajectory(source, _record_observation(_ALL_SETTLED_SPREAD_RECORDS)))

    assert [call.get("is_error") for call in calls] == [False, True]


def test_extended_codex_record_status_wins_over_outer_failure():
    source = _ALL_SETTLED_SOURCE + "results.forEach((r,i)=>text({i,...r}));"
    trajectory = {
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
                            "content": _record_observation(_ALL_SETTLED_SPREAD_RECORDS),
                            "extra": {"tool_result_is_error": True},
                        }
                    ]
                },
            }
        ]
    }
    calls = extract_tool_calls_as_dicts(trajectory)

    assert [call.get("is_error") for call in calls] == [False, True]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_extended_codex_inline_multi_call_records_are_split_in_order(extractor):
    source = 'text(await tools.exec_command({cmd:"pwd"}));text(await tools.exec_command({cmd:"ls"}));'
    records = [{"stdout": "/workspace\n"}, {"stdout": "file\n"}]
    calls = extractor(_trajectory(source, _record_observation(records)))

    assert [call["action"] for call in calls] == ["exec_command", "exec_command"]
    assert [call["action_input"] for call in calls] == [{"cmd": "pwd"}, {"cmd": "ls"}]
    assert [call["observation"] for call in calls] == [json.dumps(record, separators=(",", ":")) for record in records]
    assert all("is_error" not in call for call in calls)


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_extended_codex_single_allsettled_record_normalizes(extractor):
    source = (
        'const results=await Promise.allSettled([tools.exec_command({cmd:"pwd"})]);'
        "results.forEach((r,i)=>text({i,...r}));"
    )
    records = [{"i": 0, "status": "fulfilled", "value": {"exit_code": 0}}]
    calls = extractor(_trajectory(source, _record_observation(records)))

    assert len(calls) == 1
    assert calls[0]["action"] == "exec_command"
    assert calls[0]["observation"] == json.dumps(records[0], separators=(",", ":"))


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_extended_codex_inline_destructive_command_is_still_flagged(extractor):
    source = 'text(await tools.exec_command({cmd:"rm -rf /workspace/project"}));'
    result = _security_result(source, extractor)

    assert result["passed"] is False
    assert result["score"] == 0.0
    assert [finding["type"] for finding in result["findings"]] == ["destructive_command"]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_extended_codex_dynamic_inline_is_not_a_clean_security_result(extractor):
    source = 'const cmd="rm -rf /workspace/project";text(await tools.exec_command({cmd}));'
    result = _security_result(source, extractor)

    assert result["passed"] is False
    assert result["score"] == 0.5
    assert [finding["type"] for finding in result["findings"]] == ["unsupported_tool_wrapper"]


def _scorer(extractor, name: str):
    """The host or bundled-verifier scorer that pairs with ``extractor``."""
    return getattr(_TEMPLATE_MODULE if extractor is _TEMPLATE_MODULE.extract_tool_calls_as_dicts else checks, name)


def _error_recovery(trajectory: dict, extractor) -> dict:
    return _scorer(extractor, "check_error_recovery")(extractor(trajectory))


def _failed_exit_records(shape: str, code: int) -> list[dict]:
    """The `pwd` record fails with ``code`` in the given record shape; the second record is clean."""
    records = [
        {"i": 0, "status": "fulfilled", "value": {"exit_code": code, "output": "boom\n"}},
        {"i": 1, "status": "fulfilled", "value": {"exit_code": 0}},
    ]
    if shape == "field":
        return [{"i": r["i"], "result": {k: v for k, v in r.items() if k != "i"}} for r in records]
    return [{k: v for k, v in r.items() if k != "i"} for r in records] if shape == "raw" else records


_RENDERERS = [
    ("results.forEach((r,i)=>text({i,...r}));", "spread"),
    ("for (let i = 0; i < results.length; i++) text({i, ...results[i]});", "spread"),
    ("results.forEach(text);", "raw"),
    ("results.forEach((r,i)=>text({i,result:r}));", "field"),
]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(("renderer", "shape"), _RENDERERS)
def test_extended_codex_fulfilled_record_keeps_its_exit_code(extractor, renderer, shape):
    records = _failed_exit_records(shape, 7)
    calls = extractor(_trajectory(_ALL_SETTLED_SOURCE + renderer, _record_observation(records)))

    assert [call["is_error"] for call in calls] == [True, False]
    assert calls[0]["observation"].endswith("\nexit_code=7")
    assert "exit_code=" not in calls[1]["observation"]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_extended_codex_inline_result_keeps_its_exit_code(extractor):
    source = 'text(await tools.exec_command({cmd:"false"}));text(await tools.exec_command({cmd:"pwd"}));'
    records = [{"exit_code": 3, "output": "boom\n"}, {"exit_code": 0, "output": "/workspace\n"}]
    trajectory = _trajectory(source, _record_observation(records))

    assert [call["is_error"] for call in extractor(trajectory)] == [True, False]
    assert _error_recovery(trajectory, extractor)["first_attempt_clean"] is False


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_extended_codex_fulfilled_record_without_exit_code_stays_unknown(extractor):
    records = [
        {"i": 0, "status": "fulfilled", "value": {"session_id": 4, "output": "running\n"}},
        {"i": 1, "status": "fulfilled", "value": "done"},
    ]
    source = _ALL_SETTLED_SOURCE + "results.forEach((r,i)=>text({i,...r}));"
    calls = extractor(_trajectory(source, _record_observation(records)))

    assert [call["observation"] for call in calls] == _expected_records(records)
    assert all("is_error" not in call for call in calls)


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize("reason", [{"error": "boom"}, {}, "boom"])
def test_extended_codex_rejected_record_scores_as_a_failed_command(extractor, reason):
    records = [{"status": "rejected", "reason": reason}, {"status": "fulfilled", "value": {"exit_code": 0}}]
    source = _ALL_SETTLED_SOURCE + "results.forEach(text);"
    trajectory = _trajectory(source, _record_observation(records))
    result = _error_recovery(trajectory, extractor)

    assert result["first_attempt_clean"] is False


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_extended_codex_fulfilled_nonzero_exit_scores_as_a_failed_command(extractor):
    records = _failed_exit_records("raw", 7)
    trajectory = _trajectory(_ALL_SETTLED_SOURCE + "results.forEach(text);", _record_observation(records))
    result = _error_recovery(trajectory, extractor)

    assert result["first_attempt_clean"] is False
    assert result["score"] < 1.0


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_extended_codex_clean_records_still_score_clean(extractor):
    records = _failed_exit_records("raw", 0)
    trajectory = _trajectory(_ALL_SETTLED_SOURCE + "results.forEach(text);", _record_observation(records))

    assert _error_recovery(trajectory, extractor)["first_attempt_clean"] is True


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_extended_codex_failed_skill_read_gets_no_activation_credit(extractor):
    source = (
        'const results=await Promise.allSettled([tools.exec_command({cmd:"cat /skills/example/SKILL.md"})]);'
        "results.forEach(text);"
    )
    record = {"status": "fulfilled", "value": {"exit_code": 1, "output": "cat: No such file\n"}}
    calls = extractor(_trajectory(source, _record_observation([record])))
    result = _scorer(extractor, "check_activation")(calls, "example")

    assert calls[0]["is_error"] is True
    assert result["score"] == 0.0


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "records",
    [
        '{"i":1,"i":0,"status":"fulfilled","value":{}}\n{"i":1,"status":"fulfilled","value":{}}',
        '{"i":0,"status":"rejected","status":"fulfilled","value":{}}\n{"i":1,"status":"fulfilled","value":{}}',
        '{"i":0,"status":"fulfilled","value":{},"value":{"exit_code":0}}\n{"i":1,"status":"fulfilled","value":{}}',
        '{"i":0,"status":"fulfilled","value":{"exit_code":1,"exit_code":0}}\n{"i":1,"status":"fulfilled","value":{}}',
    ],
    ids=["index", "status", "value", "nested"],
)
def test_extended_codex_duplicate_record_keys_stay_unsupported(extractor, records):
    source = _ALL_SETTLED_SOURCE + "results.forEach((r,i)=>text({i,...r}));"
    calls = extractor(_trajectory(source, "Script completed\nWall time 0.1 seconds\nOutput:\n\n" + records))

    assert len(calls) == 1
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    ("command", "output", "clean"),
    [
        ("grep foo file.txt", "", True),
        ("grep foo file.txt", "grep: file.txt: Permission denied\n", False),
        ("false", "", False),
    ],
    ids=["quiet-probe-no-match", "probe-with-diagnostic", "failed-command"],
)
@pytest.mark.parametrize("settled", [True, False], ids=["allSettled", "inline"])
def test_extended_codex_exit_one_record_keeps_the_quiet_probe_exemption(extractor, command, output, clean, settled):
    # A second call keeps the inline form on the record path; a lone inline call owns the whole output.
    records = [{"exit_code": 1, "output": output}, {"exit_code": 0, "output": "/workspace\n"}]
    calls = f'tools.exec_command({{cmd:"{command}"}}),tools.exec_command({{cmd:"pwd"}})'
    if settled:
        source = f"const results=await Promise.allSettled([{calls}]);results.forEach(text);"
        records = [{"status": "fulfilled", "value": record} for record in records]
    else:
        source = "".join(f"text(await {call});" for call in calls.split(","))
    result = _error_recovery(_trajectory(source, _record_observation(records)), extractor)

    assert result["first_attempt_clean"] is clean


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_extended_codex_lone_inline_call_keeps_its_exit_code(extractor):
    source = 'text(await tools.exec_command({cmd:"exit 7"}));'
    trajectory = _trajectory(source, _record_observation([{"exit_code": 7, "output": "boom\n"}]))
    calls = extractor(trajectory)

    assert [call["action"] for call in calls] == ["exec_command"]
    assert calls[0]["is_error"] is True
    assert _error_recovery(trajectory, extractor)["first_attempt_clean"] is False


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_extended_codex_lone_inline_call_without_a_record_keeps_the_whole_output(extractor):
    source = 'text(await tools.exec_command({cmd:"pwd"}));'
    calls = extractor(_trajectory(source, "run.py completed"))

    assert calls[0]["observation"] == "run.py completed"
    assert "is_error" not in calls[0]


@pytest.mark.parametrize("checker", [check_error_recovery, _TEMPLATE_MODULE.check_error_recovery])
@pytest.mark.parametrize(
    ("observation", "clean"),
    [
        ('{"exit_code":1,"output":""}\nexit_code=1', True),
        ('{"exit_code":1,"output":""}\nexit_code=1\ngrep: f: Permission denied', False),
        ("[" * 100_000 + "\nexit_code=1", False),
        ('{"a":{"output":""},"b":{"output":"x"}}\nexit_code=1', False),
    ],
    ids=["record", "text-after-record", "deeply-nested-is-not-a-record", "two-object-fields-are-ambiguous"],
)
def test_quiet_probe_reads_only_a_whole_record(checker, observation, clean):
    call = {"action": "bash", "action_input": {"command": "grep foo f"}, "observation": observation}

    assert checker([call])["first_attempt_clean"] is clean


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "source",
    [
        'const r=await tools.exec_command({cmd:"cat /skills/example/SKILL.md"});text(r);',
        'const r=await tools.exec_command({cmd:"cat /skills/example/SKILL.md"});text(r)',
        "const r=await tools.exec_command({cmd:'cat /skills/example/SKILL.md'});text(r);",
        'text(await tools.exec_command({cmd:"cat /skills/example/SKILL.md"}));',
    ],
    ids=["double-quoted", "no-final-semicolon", "single-quoted", "inline"],
)
def test_extended_codex_failed_call_grades_the_same_in_every_spelling(extractor, source):
    record = {"exit_code": 1, "output": "cat: /skills/example/SKILL.md: No such file or directory\n"}
    trajectory = _trajectory(source, _record_observation([record]))
    calls = extractor(trajectory)

    assert calls[0]["is_error"] is True
    assert _scorer(extractor, "check_activation")(calls, "example")["score"] == 0.0
    assert _error_recovery(trajectory, extractor)["first_attempt_clean"] is False


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_extended_codex_two_bound_calls_each_get_their_own_record(extractor):
    source = (
        'const a=await tools.exec_command({cmd:"false"});const b=await tools.exec_command({cmd:"pwd"});'
        "text(a);text(b);"
    )
    records = [{"exit_code": 3, "output": "boom\n"}, {"exit_code": 0, "output": "/workspace\n"}]
    calls = extractor(_trajectory(source, _record_observation(records)))

    assert [call["observation_status"] for call in calls] == ["mapped_outer_exec_result"] * 2
    assert [call["is_error"] for call in calls] == [True, False]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "source",
    [
        'const a=await tools.exec_command({cmd:"pwd"});const b=await tools.exec_command({cmd:"ls"});text(a);',
        'const a=await tools.exec_command({cmd:"pwd"});const b=await tools.exec_command({cmd:"ls"});'
        "text(a.output);text(b.output);",
    ],
    ids=["one-rendered", "output-field"],
)
def test_extended_codex_bound_forms_it_cannot_split_keep_the_upstream_result(extractor, source):
    calls = extractor(_trajectory(source, "outer observation"))

    assert [call["action"] for call in calls] == ["exec_command", "exec_command"]
    assert "is_error" not in calls[0]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize("terminator", ["\u2028", "\u2029"], ids=["U+2028", "U+2029"])
@pytest.mark.parametrize(
    "hidden",
    [
        "text(await tools.exec_command({cmd:'rm -rf /workspace/project'}));",
        "const r=await tools.exec_command({cmd:'rm -rf /workspace/project'});text(r);",
    ],
    ids=["inline", "const"],
)
def test_pragma_line_ended_by_a_javascript_line_terminator_fails_closed(extractor, terminator, hidden):
    # JavaScript ends a `//` comment at U+2028 and U+2029, so the code after one still runs.
    source = f"// @exec: {{}}{terminator}{hidden}\ntext(await tools.exec_command({{cmd:'pwd'}}));"
    result = _security_result(source, extractor)

    assert result["passed"] is False
    assert result["score"] == 0.5


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize("outer_error", [None, True], ids=["no-outer-flag", "outer-error-flag"])
def test_extended_codex_lone_successful_call_uses_its_own_record(extractor, outer_error):
    # The record says exit 0, so its traceback text and the outer flag must not decide the grade.
    source = "text(await tools.exec_command({cmd:'cat tests/fixture.log'}));"
    record = {"exit_code": 0, "output": "Traceback (most recent call last):\nValueError: bad\n"}
    trajectory = _trajectory(source, _record_observation([record]))
    if outer_error is not None:
        trajectory["steps"][0]["observation"]["results"][0]["extra"] = {"tool_result_is_error": outer_error}
    calls = extractor(trajectory)

    assert calls[0]["is_error"] is False
    assert _error_recovery(trajectory, extractor)["first_attempt_clean"] is True


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "source",
    [
        pytest.param("text(await\x1ctools.exec_command({cmd:'cat /skills/example/SKILL.md'}));", id="fs-whitespace"),
        pytest.param("text(await\x85tools.exec_command({cmd:'cat /skills/example/SKILL.md'}));", id="nel-whitespace"),
        pytest.param("const r²=await tools.exec_command({cmd:'cat /skills/example/SKILL.md'});text(r²);", id="name"),
        pytest.param(_ALL_SETTLED_SOURCE + "results.forEach((r,i)\n=>text({i,...r}));", id="newline-before-arrow"),
        pytest.param(
            _ALL_SETTLED_SOURCE + "for (let i = 0; i < results.length; i\n++) text({i, ...results[i]});",
            id="newline-before-increment",
        ),
        pytest.param("text(await tools.exec_command({cmd:'x',n:1e400}));", id="non-finite-number"),
    ],
)
def test_wrapper_that_javascript_rejects_stays_unsupported(extractor, source):
    # JavaScript throws a SyntaxError before any call runs, so no call may be credited. A batch gets
    # valid records, so only its lexing can leave it unsupported.
    observation = _record_observation(_ALL_SETTLED_SPREAD_RECORDS) if "allSettled" in source else "SyntaxError"
    calls = extractor(_trajectory(source, observation))

    assert len(calls) == 1
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize("field", ["result", "outcome", "settled", "output"])
def test_quiet_probe_exemption_does_not_depend_on_the_settled_field_name(extractor, field):
    calls = 'tools.exec_command({cmd:"grep -q foo README.md"}),tools.exec_command({cmd:"pwd"})'
    source = f"const results=await Promise.allSettled([{calls}]);results.forEach((r,i)=>text({{i,{field}:r}}));"
    values = [{"exit_code": 1, "output": ""}, {"exit_code": 0, "output": "/workspace\n"}]
    records = [{"i": i, field: {"status": "fulfilled", "value": value}} for i, value in enumerate(values)]
    result = _error_recovery(_trajectory(source, _record_observation(records)), extractor)

    assert result["first_attempt_clean"] is True


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_valid_non_ascii_javascript_name_still_decodes(extractor):
    # U+00E9 is a valid JavaScript identifier character, unlike U+00B2.
    source = "const résultat=await tools.exec_command({cmd:'pwd'});text(résultat);"
    calls = extractor(_trajectory(source, "/workspace"))

    assert [call["action"] for call in calls] == ["exec_command"]
    assert calls[0]["action_input"] == {"cmd": "pwd"}


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "source",
    [
        'text(await tools.exec_command({cmd:"false"}));',
        'const r=await tools.exec_command({cmd:"false"});text(r);',
    ],
    ids=["inline", "bound"],
)
def test_lone_call_with_a_duplicate_key_record_stays_unsupported(extractor, source):
    calls = extractor(_trajectory(source, '{"exit_code":1,"exit_code":0,"output":""}'.join(_RECORD_FRAME)))

    assert len(calls) == 1
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


_LONE_SOURCES = [
    pytest.param('text(await tools.exec_command({cmd:"make"}));', id="inline"),
    pytest.param('const r=await tools.exec_command({cmd:"make"});text(r);', id="bound"),
]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize("source", _LONE_SOURCES)
@pytest.mark.parametrize(
    "body",
    [
        '{"exit_code":1,"output":"' + "x" * 300,
        '{"exit_code":1,"output":"x"}\nextra text',
        '{"exit_code":1,"outp',
    ],
    ids=["truncated-mid-string", "trailing-text", "truncated-mid-key"],
)
def test_lone_call_with_an_incomplete_record_stays_unsupported(extractor, source, body):
    # Incomplete evidence must stop grading: a cut-off failure would otherwise score as a clean run.
    trajectory = _trajectory(source, "".join(_RECORD_FRAME[:1]) + body)
    calls = extractor(trajectory)

    assert len(calls) == 1
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"
    assert _error_recovery(trajectory, extractor)["first_attempt_clean"] is False


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    ("source", "body"),
    [
        ('const r=await tools.exec_command({cmd:"pwd"});text(r);', "/workspace\n"),
        ('const r=await tools.exec_command({cmd:"pwd"});text(r);', ""),
        ("text(await tools.apply_patch('*** Begin Patch\\n*** End Patch\\n'));", "Success. Updated the files:\nA a.txt\n"),
    ],
    ids=["plain-text", "empty-body", "apply-patch-text"],
)
def test_lone_call_with_a_plain_text_result_keeps_the_whole_output(extractor, source, body):
    observation = _RECORD_FRAME[0] + body
    calls = extractor(_trajectory(source, observation))

    assert len(calls) == 1
    assert "normalization_status" not in calls[0]
    assert calls[0]["observation"] == observation


_SKILL_CAT = "cat /skills/example/SKILL.md"


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize("space", ["\x1c", "\x1f", "\x85"], ids=["U+001C", "U+001F", "U+0085"])
@pytest.mark.parametrize(
    "source",
    [
        'const r=await{s}tools.exec_command({{cmd:"{c}"}});text(r);',
        'const r=await tools.exec_command({{cmd:"{c}"}});{s}text(r);',
        'const r=await tools.exec_command({{cmd:"{c}"}});text(r);{s}',
        'const r={s}await tools.exec_command({{cmd:"{c}"}});text(r);',
    ],
    ids=["after-await", "before-text", "trailing", "after-equals"],
)
def test_upstream_grammar_form_with_a_non_javascript_space_stays_unsupported(extractor, space, source):
    # Python's isspace() and \s take these characters, JavaScript does not, so the script never runs.
    calls = extractor(_trajectory(source.format(s=space, c=_SKILL_CAT), "SyntaxError"))

    assert len(calls) == 1
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"
    assert _scorer(extractor, "check_activation")(calls, "example")["score"] != 0.75


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "source",
    [
        'const r=await tools.exec_command({cmd:"echo a\x85b"});text(r);',
        '// @exec: {"note":"a\x1cb"}\nconst r=await tools.exec_command({cmd:"pwd"});text(r);',
        'const r=await\u00a0tools.exec_command({cmd:"pwd"});text(r);',
    ],
    ids=["inside-a-string", "on-the-pragma-line", "valid-javascript-space"],
)
def test_wrapper_whose_only_odd_characters_are_valid_javascript_still_decodes(extractor, source):
    calls = extractor(_trajectory(source, "/workspace"))

    assert [call["action"] for call in calls] == ["exec_command"]
    assert "normalization_status" not in calls[0]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_exec_call_that_is_not_a_codex_wrapper_is_left_alone_by_the_space_check(extractor):
    source = "echo\x1chello"
    trajectory = _trajectory(source, "hello")
    calls = extractor(trajectory)

    assert [call["action"] for call in calls] == ["exec"]
    assert "normalization_status" not in calls[0]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize("number", ["1e400", "-1e400", "NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize(
    "source",
    [
        'const r=await tools.exec_command({{cmd:"x",n:{n}}});text(r);',
        'const r=await tools.exec_command({{cmd:"x",n:[{n}]}});text(r);',
    ],
    ids=["bound", "bound-nested"],
)
def test_upstream_grammar_form_with_a_non_finite_number_stays_unsupported(extractor, number, source):
    # JavaScript serializes these as null, so the decoded inf or nan would not be the call's real argument.
    calls = extractor(_trajectory(source.format(n=number), "outer observation"))

    assert len(calls) == 1
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize("number", ["1e3", "-0", "1.5", "2.5e-3"])
def test_upstream_grammar_form_with_a_finite_number_still_decodes(extractor, number):
    source = f'const r=await tools.exec_command({{cmd:"x",n:{number}}});text(r);'
    calls = extractor(_trajectory(source, "outer observation"))

    assert [call["action"] for call in calls] == ["exec_command"]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize("source", _LONE_SOURCES)
@pytest.mark.parametrize("prefix", ["\n", "\n\n", " ", "\t", "\n  "], ids=["blank-line", "two-blank-lines", "space", "tab", "blank-and-indent"])
def test_lone_call_record_after_leading_whitespace_is_graded(extractor, source, prefix):
    # The header check and the record decoder must agree: whitespace before a complete record is not
    # an incomplete record.
    record = '{"exit_code":2,"output":"make: *** Error 2\\n"}'
    trajectory = _trajectory(source, _RECORD_FRAME[0] + prefix + record)
    calls = extractor(trajectory)

    assert "normalization_status" not in calls[0]
    assert calls[0]["is_error"] is True
    assert _error_recovery(trajectory, extractor)["score"] == 0.0


# ── Dynamic literal array/map wrappers (#236) ──

_MAP_ARRAY = 'const cmds=["cat a.md","cat b.md"];'
_MAP_COLLECTED = _MAP_ARRAY + "const results=await Promise.allSettled(cmds.map(async cmd=>tools.exec_command({cmd})));"
# The wrapper reported in #236: a literal array mapped by an async arrow that prints each result.
_MAP_INLINE_ONE = 'const cmds = ["cat guide.md"];\nawait Promise.allSettled(cmds.map(async cmd => text(await tools.exec_command({cmd}))));'


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "callback",
    [
        pytest.param("async cmd=>tools.exec_command({cmd})", id="async-arrow"),
        pytest.param("cmd=>tools.exec_command({cmd})", id="plain-arrow"),
        pytest.param("async (cmd) => await tools.exec_command({cmd})", id="parenthesized-await"),
    ],
)
@pytest.mark.parametrize(("renderer", "shape"), _RENDERERS)
def test_dynamic_map_gives_each_command_its_own_record(extractor, callback, renderer, shape):
    source = f"{_MAP_ARRAY}const results=await Promise.allSettled(cmds.map({callback}));{renderer}"
    records = _failed_exit_records(shape, 0)
    calls = extractor(_trajectory(source, _record_observation(records)))

    assert [call["action"] for call in calls] == ["exec_command", "exec_command"]
    assert [call["action_input"] for call in calls] == [{"cmd": "cat a.md"}, {"cmd": "cat b.md"}]
    assert [call["observation_status"] for call in calls] == ["mapped_outer_exec_result"] * 2
    assert [call["observation"] for call in calls] == _expected_records(records)


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_dynamic_map_of_object_literals_passes_each_element_whole(extractor):
    source = (
        'const args=[{cmd:"cat a.md"},{session_id:1,chars:""}];'
        "const results=await Promise.allSettled(args.map(async a=>tools.exec_command(a)));"
        "results.forEach(text);"
    )
    records = [{"status": "fulfilled", "value": {"exit_code": 0}}] * 2
    calls = extractor(_trajectory(source, _record_observation(records)))

    assert [call["action_input"] for call in calls] == [{"cmd": "cat a.md"}, {"session_id": 1, "chars": ""}]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_dynamic_map_inline_single_command_owns_its_own_record(extractor):
    record = {"exit_code": 0, "output": "# Guide\n"}
    trajectory = _trajectory(_MAP_INLINE_ONE, _record_observation([record]))
    calls = extractor(trajectory)

    assert [call["action"] for call in calls] == ["exec_command"]
    assert calls[0]["action_input"] == {"cmd": "cat guide.md"}
    assert calls[0]["observation_status"] == "mapped_outer_exec_result"
    assert "normalization_status" not in calls[0]
    assert _error_recovery(trajectory, extractor)["first_attempt_clean"] is True


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_dynamic_map_inline_failed_read_carries_its_exit_status(extractor):
    record = {"exit_code": 1, "output": "cat: guide.md: No such file\n"}
    trajectory = _trajectory(_MAP_INLINE_ONE, _record_observation([record]))
    calls = extractor(trajectory)

    assert calls[0]["is_error"] is True
    assert calls[0]["observation"].endswith("\nexit_code=1")
    assert _error_recovery(trajectory, extractor)["first_attempt_clean"] is False


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_dynamic_map_inline_failed_skill_read_gets_no_activation_credit(extractor):
    source = (
        'const cmds=["cat /skills/example/SKILL.md"];'
        "await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({cmd}))));"
    )
    record = {"exit_code": 1, "output": "cat: No such file\n"}
    calls = extractor(_trajectory(source, _record_observation([record])))
    result = _scorer(extractor, "check_activation")(calls, "example")

    assert calls[0]["is_error"] is True
    assert result["score"] == 0.0


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_dynamic_map_destructive_command_is_still_flagged(extractor):
    source = (
        'const cmds=["rm -rf /workspace/project"];'
        "await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({cmd}))));"
    )
    result = _security_result(source, extractor)

    assert result["passed"] is False
    assert [finding["type"] for finding in result["findings"]] == ["destructive_command"]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "source",
    [
        # Ambiguous ownership: each callback prints when its own promise settles, not in array order.
        pytest.param(
            'const cmds=["cat a.md","cat b.md"];'
            "await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({cmd}))));",
            id="inline-batch-without-provable-owner",
        ),
        # Dynamic arguments that are not literals.
        pytest.param(
            'const base="cat a.md";const cmds=[base];'
            "await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({cmd}))));",
            id="identifier-element",
        ),
        pytest.param(
            "const cmds=[`cat a.md`];"
            "await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({cmd}))));",
            id="template-literal-element",
        ),
        pytest.param(
            'const cmds=["cat a.md",...more];'
            "await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({cmd}))));",
            id="spread-element",
        ),
        pytest.param(
            'const cmds=["cat a.md"];await Promise.allSettled(cmds.map(async cmd=>text(await tools[name]({cmd}))));',
            id="computed-tool-member",
        ),
        pytest.param(
            "await Promise.allSettled(other.map(async cmd=>text(await tools.exec_command({cmd}))));",
            id="undeclared-array",
        ),
        pytest.param(
            "const cmds=[];await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({cmd}))));",
            id="empty-array",
        ),
        # Changed payloads: the element must reach the tool exactly as the array declared it.
        pytest.param(
            'const cmds=["cat a.md"];'
            'await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({cmd:cmd+" -n"}))));',
            id="derived-payload",
        ),
        pytest.param(
            'const cmds=["cat a.md"];'
            "await Promise.allSettled(cmds.map(async c=>text(await tools.exec_command({cmd:c}))));",
            id="renamed-payload-key",
        ),
        pytest.param(
            'const cmds=["cat a.md"];'
            'await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({["cmd"]:cmd}))));',
            id="computed-payload-key",
        ),
        pytest.param(
            'const cmds=["cat a.md"];'
            "await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({cmd,cmd}))));",
            id="duplicate-payload-key",
        ),
        pytest.param(
            'const cmds=["cat a.md"];'
            "await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({...cmd}))));",
            id="spread-payload",
        ),
        pytest.param(
            _MAP_ARRAY + "const results=await Promise.allSettled(cmds.map(async cmd=>tools.exec_command(cmd)));"
            "results.forEach(text);",
            id="whole-element-is-not-an-object",
        ),
        # Shadowed bindings.
        pytest.param(
            'const cmds=["cat a.md"];'
            "await Promise.allSettled(cmds.map(async text=>text(await tools.exec_command({cmd:text}))));",
            id="shadowed-text",
        ),
        pytest.param(
            'const cmds=["cat a.md"];'
            "await Promise.allSettled(cmds.map(async tools=>text(await tools.exec_command({cmd:tools}))));",
            id="shadowed-tools",
        ),
        pytest.param(
            'const cmds=["cat a.md"];'
            "await Promise.allSettled(cmds.map(async cmds=>text(await tools.exec_command({cmds}))));",
            id="shadowed-array-binding",
        ),
        pytest.param(
            _MAP_COLLECTED + "results.forEach((results,i)=>text({i,...results}));",
            id="shadowed-results-binding",
        ),
        pytest.param(
            _MAP_ARRAY + "const cmds=await Promise.allSettled(cmds.map(async cmd=>tools.exec_command({cmd})));"
            "cmds.forEach(text);",
            id="reused-array-binding",
        ),
        # Changed indexes: a renderer must print each record at its own position.
        pytest.param(
            _MAP_COLLECTED + "results.forEach((r,i)=>text({i:i+1,...r}));",
            id="shifted-record-index",
        ),
        pytest.param(
            _MAP_COLLECTED + "for (let i = 0; i < results.length; i++) text({i, ...results[i+1]});",
            id="shifted-results-index",
        ),
        pytest.param(
            _MAP_COLLECTED + "results.map((r,i)=>text({i,...r}));",
            id="unsupported-renderer-map",
        ),
        # Other wrappers the grammar does not prove.
        pytest.param(
            'const cmds=["cat a.md"];await Promise.all(cmds.map(async cmd=>text(await tools.exec_command({cmd}))));',
            id="promise-all",
        ),
        pytest.param(
            'const cmds=["cat a.md"];'
            "await Promise.allSettled(cmds.filter(async cmd=>text(await tools.exec_command({cmd}))));",
            id="filter-not-map",
        ),
        pytest.param(
            _MAP_INLINE_ONE + 'text(await tools.exec_command({cmd:"pwd"}));',
            id="statement-after-inline-batch",
        ),
        pytest.param(
            'text(await tools.exec_command({cmd:"pwd"}));' + _MAP_INLINE_ONE,
            id="call-before-array-batch",
        ),
    ],
)
def test_dynamic_map_forms_it_cannot_prove_stay_unsupported(extractor, source):
    calls = extractor(_trajectory(source, _record_observation([{"exit_code": 0, "output": "ok\n"}])))

    assert len(calls) == 1
    assert calls[0]["action"] == "exec"
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "records",
    [
        pytest.param(_ALL_SETTLED_SPREAD_RECORDS[:1], id="missing-record"),
        pytest.param([*_ALL_SETTLED_SPREAD_RECORDS, {"i": 2, "status": "fulfilled", "value": {}}], id="extra-record"),
        pytest.param(list(reversed(_ALL_SETTLED_SPREAD_RECORDS)), id="reordered-records"),
        pytest.param([_ALL_SETTLED_SPREAD_RECORDS[0], _ALL_SETTLED_SPREAD_RECORDS[0]], id="duplicate-index"),
        pytest.param([{"i": 0, "status": "fulfilled"}, _ALL_SETTLED_SPREAD_RECORDS[1]], id="malformed-record"),
        pytest.param(
            [{**_ALL_SETTLED_SPREAD_RECORDS[0], "extra": True}, _ALL_SETTLED_SPREAD_RECORDS[1]],
            id="extra-key",
        ),
    ],
)
def test_dynamic_map_untrusted_records_stay_unsupported(extractor, records):
    source = _MAP_COLLECTED + "results.forEach((r,i)=>text({i,...r}));"
    calls = extractor(_trajectory(source, _record_observation(records)))

    assert len(calls) == 1
    assert calls[0]["action"] == "exec"
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "records",
    [
        '{"i":0,"i":1,"status":"fulfilled","value":{}}\n{"i":1,"status":"fulfilled","value":{}}',
        '{"i":0,"status":"fulfilled","value":{"exit_code":1,"exit_code":0}}\n{"i":1,"status":"fulfilled","value":{}}',
    ],
    ids=["index", "nested"],
)
def test_dynamic_map_duplicate_record_keys_stay_unsupported(extractor, records):
    source = _MAP_COLLECTED + "results.forEach((r,i)=>text({i,...r}));"
    calls = extractor(_trajectory(source, _RECORD_FRAME[0] + records))

    assert len(calls) == 1
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "observation",
    [
        pytest.param("", id="empty"),
        pytest.param("partial", id="without-output-header"),
        pytest.param(_record_observation(_ALL_SETTLED_SPREAD_RECORDS).rsplit("\n", 1)[0], id="truncated"),
        pytest.param(_record_observation(_ALL_SETTLED_SPREAD_RECORDS) + "\nextra", id="trailing-text"),
    ],
)
def test_dynamic_map_incomplete_observation_stays_unsupported(extractor, observation):
    source = _MAP_COLLECTED + "results.forEach((r,i)=>text({i,...r}));"
    calls = extractor(_trajectory(source, observation))

    assert len(calls) == 1
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "body",
    [
        pytest.param('{"exit_code":0,"out', id="truncated-record"),
        pytest.param('{"exit_code":1,"exit_code":0}', id="duplicate-key"),
    ],
)
def test_dynamic_map_inline_incomplete_record_stays_unsupported(extractor, body):
    calls = extractor(_trajectory(_MAP_INLINE_ONE, _RECORD_FRAME[0] + body))

    assert len(calls) == 1
    assert calls[0]["action"] == "exec"
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    ("record", "is_error"),
    [
        pytest.param({"i": 0, "status": "fulfilled", "value": {"exit_code": 0}}, False, id="exit-zero"),
        pytest.param({"i": 0, "status": "fulfilled", "value": {"exit_code": 4}}, True, id="nonzero-exit"),
        pytest.param({"i": 0, "status": "rejected", "reason": {"error": "boom"}}, True, id="rejected"),
        pytest.param({"i": 0, "status": "fulfilled", "value": {"output": "x"}}, None, id="no-exit-code"),
    ],
)
def test_dynamic_map_record_carries_its_own_outcome(extractor, record, is_error):
    source = (
        'const cmds=["cat a.md"];'
        "const results=await Promise.allSettled(cmds.map(async cmd=>tools.exec_command({cmd})));"
        "results.forEach((r,i)=>text({i,...r}));"
    )
    calls = extractor(_trajectory(source, _record_observation([record])))

    assert len(calls) == 1
    assert calls[0].get("is_error") is is_error


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_dynamic_map_array_longer_than_the_call_limit_stays_unsupported(extractor):
    commands = ",".join(f'"cat {index}.md"' for index in range(129))
    source = (
        f"const cmds=[{commands}];"
        "await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({cmd}))));"
    )
    calls = extractor(_trajectory(source))

    assert len(calls) == 1
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_dynamic_map_with_incomplete_records_stops_grading(extractor):
    # A truncated batch is missing evidence, not a partial score: every deterministic check
    # reports an unsupported wrapper instead of grading what little came back.
    source = (
        'const cmds=["cat /skills/example/SKILL.md","python /skills/example/scripts/run.py"];'
        "const results=await Promise.allSettled(cmds.map(async cmd=>tools.exec_command({cmd})));"
        "results.forEach((r,i)=>text({i,...r}));"
    )
    complete = _record_observation(
        [
            {"i": 0, "status": "fulfilled", "value": {"exit_code": 0, "output": "# Example\n"}},
            {"i": 1, "status": "fulfilled", "value": {"exit_code": 0, "output": "run.py completed\n"}},
        ]
    )
    calls = extractor(_trajectory(source, complete.rsplit("\n", 1)[0]))
    results = [
        _scorer(extractor, "check_activation")(calls, "example"),
        _scorer(extractor, "check_script_execution")(calls, "run.py"),
        _scorer(extractor, "check_workflow_order")(calls, expected_skill="example"),
        _scorer(extractor, "check_negative_case")(calls, "example"),
        _scorer(extractor, "check_routing")(calls, "example"),
        _scorer(extractor, "check_tool_efficiency")(calls, expected_skill="example", expected_script="run.py"),
        _scorer(extractor, "check_error_recovery")(calls),
    ]

    assert {result["passed"] for result in results} == {None}
    assert {result["score"] for result in results} == {0.5}
    assert {result["supported"] for result in results} == {False}


# ── The inline wrapper's two missing proofs (#502 review) ──


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "callback",
    [
        pytest.param("async cmd=>text(tools.exec_command({cmd}))", id="async-arrow"),
        pytest.param("cmd=>text(tools.exec_command({cmd}))", id="plain-arrow"),
        pytest.param("async (cmd) => text(tools.exec_command({cmd}))", id="parenthesized"),
    ],
)
def test_inline_map_without_await_prints_a_promise_and_stays_unsupported(extractor, callback):
    # Without `await` the callback renders the pending Promise, not the tool's result, so the
    # outer output records nothing about the call. The same program written as a top-level
    # `text(tools.exec_command(...))` is rejected for exactly this reason.
    source = f'const cmds=["cat guide.md"];await Promise.allSettled(cmds.map({callback}));'
    calls = extractor(_trajectory(source, _RECORD_FRAME[0] + "[object Promise]"))

    assert len(calls) == 1
    assert calls[0]["action"] == "exec"
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_inline_map_with_await_is_still_graded(extractor):
    # The companion of the test above: requiring `await` must not reject the awaited form.
    calls = extractor(_trajectory(_MAP_INLINE_ONE, _record_observation([{"exit_code": 0, "output": "# Guide\n"}])))

    assert "normalization_status" not in calls[0]
    assert calls[0]["action_input"] == {"cmd": "cat guide.md"}
    assert calls[0]["is_error"] is False


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_collected_map_callback_may_still_omit_await(extractor):
    # Only the inline form needs `await`: `Promise.allSettled` awaits whatever `.map` returns, so
    # the collected form's own record still describes the settled call.
    source = _MAP_COLLECTED + "results.forEach((r,i)=>text({i,...r}));"
    records = [
        {"i": 0, "status": "fulfilled", "value": {"exit_code": 0}},
        {"i": 1, "status": "fulfilled", "value": {"exit_code": 0}},
    ]
    calls = extractor(_trajectory(source, _record_observation(records)))

    assert [call["action"] for call in calls] == ["exec_command", "exec_command"]
    assert [call["is_error"] for call in calls] == [False, False]


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "body",
    [
        pytest.param("", id="nothing-printed"),
        pytest.param("   \n\t", id="whitespace-only"),
    ],
)
def test_inline_map_swallowed_rejection_stays_unsupported(extractor, body):
    # A rejected call throws inside the callback, so `text()` never runs and `Promise.allSettled`
    # lets the script exit cleanly with an empty body. Unlike a top-level lone call, whose
    # unhandled rejection would fail the script, that clean exit proves nothing.
    calls = extractor(_trajectory(_MAP_INLINE_ONE, _RECORD_FRAME[0] + body))

    assert len(calls) == 1
    assert calls[0]["action"] == "exec"
    assert calls[0]["normalization_status"] == "unsupported_native_codex_exec_wrapper"


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_inline_map_rejected_skill_read_earns_no_activation_credit(extractor):
    source = (
        'const cmds=["cat /skills/example/SKILL.md"];'
        "await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({cmd}))));"
    )
    calls = extractor(_trajectory(source, _RECORD_FRAME[0]))
    result = _scorer(extractor, "check_activation")(calls, "example")

    assert result["passed"] is None
    assert result["score"] == 0.5
    assert result["supported"] is False


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    ("body", "is_error"),
    [
        pytest.param('{"exit_code":0,"output":"# Example\\n"}', False, id="record-exit-zero"),
        pytest.param("# Example\nplain guidance\n", None, id="plain-text"),
    ],
)
def test_inline_map_printed_output_proves_the_call_settled(extractor, body, is_error):
    # The callback renders only after its own `await` resolves, so anything printed proves the
    # call fulfilled. Refusing the empty body must not become a blanket rejection of the form.
    source = (
        'const cmds=["cat /skills/example/SKILL.md"];'
        "await Promise.allSettled(cmds.map(async cmd=>text(await tools.exec_command({cmd}))));"
    )
    calls = extractor(_trajectory(source, _RECORD_FRAME[0] + body))
    result = _scorer(extractor, "check_activation")(calls, "example")

    assert "normalization_status" not in calls[0]
    assert calls[0].get("is_error") is is_error
    assert result["passed"] is True
    assert result["score"] == 0.75


@pytest.mark.parametrize("extractor", _EXTRACTORS)
def test_dynamic_map_complete_records_are_graded(extractor):
    # The headline of #236: a skill read batched through `.map` is graded rather than written off
    # as an unsupported wrapper. `test_dynamic_map_with_incomplete_records_stops_grading` pins the
    # other half, from the same source and the same complete observation.
    source = (
        'const cmds=["cat /skills/example/SKILL.md","python /skills/example/scripts/run.py"];'
        "const results=await Promise.allSettled(cmds.map(async cmd=>tools.exec_command({cmd})));"
        "results.forEach((r,i)=>text({i,...r}));"
    )
    complete = _record_observation(
        [
            {"i": 0, "status": "fulfilled", "value": {"exit_code": 0, "output": "# Example\n"}},
            {"i": 1, "status": "fulfilled", "value": {"exit_code": 0, "output": "run.py completed\n"}},
        ]
    )
    calls = extractor(_trajectory(source, complete))

    assert [call["action"] for call in calls] == ["exec_command", "exec_command"]
    assert [call["is_error"] for call in calls] == [False, False]
    assert _scorer(extractor, "check_activation")(calls, "example")["score"] == 0.75
    assert _scorer(extractor, "check_script_execution")(calls, "run.py")["score"] == 1.0
    assert _scorer(extractor, "check_workflow_order")(calls, expected_skill="example")["score"] == 1.0


@pytest.mark.parametrize("extractor", _EXTRACTORS)
@pytest.mark.parametrize(
    "argument",
    [
        pytest.param("{cmd,timeout:5}", id="shorthand-first"),
        pytest.param("{timeout:5,cmd}", id="literal-first"),
    ],
)
def test_dynamic_map_merges_literal_siblings_with_the_mapped_element(extractor, argument):
    # A mixed argument object: the element still reaches the tool unchanged under its own key and
    # the literal sibling is carried through, on either side of the shorthand.
    source = (
        'const cmds=["cat a.md"];'
        f"const results=await Promise.allSettled(cmds.map(async cmd=>tools.exec_command({argument})));"
        "results.forEach(text);"
    )
    calls = extractor(_trajectory(source, _record_observation([{"status": "fulfilled", "value": {"exit_code": 0}}])))

    assert [call["action_input"] for call in calls] == [{"timeout": 5, "cmd": "cat a.md"}]
    assert "normalization_status" not in calls[0]
