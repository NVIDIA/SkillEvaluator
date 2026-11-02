# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host and staged verifier use one canonical evidence implementation."""

from __future__ import annotations

import json
import subprocess
import sys

from skillevaluator.tier3.eval_core import atif_helpers
from skillevaluator.tier3.harbor.adapter import _copy_verifier


def trajectory():
    return {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "read",
                        "function_name": "exec_command",
                        "arguments": {"cmd": "cat skills/example/SKILL.md"},
                    },
                    {
                        "tool_call_id": "write",
                        "function_name": "Write",
                        "arguments": {"file_path": "/workspace/result.txt", "content": "done"},
                    },
                ],
                "observation": {
                    "results": [
                        {"source_call_id": "read", "content": "Guide"},
                        {"source_call_id": "write", "content": "Written", "extra": {"tool_result_is_error": False}},
                    ]
                },
            },
            {"source": "agent", "message": [{"type": "text", "text": "Done"}]},
        ]
    }


def test_template_uses_current_command_aliases(harbor_eval_template):
    expected = ["cat skills/example/SKILL.md"]
    assert atif_helpers.get_read_calls(trajectory()) == expected
    assert harbor_eval_template.get_read_calls(trajectory()) == expected
    assert harbor_eval_template.get_bash_commands(trajectory()) == expected


def test_staged_helpers_import_without_installed_package(tmp_path):
    _copy_verifier(tmp_path)
    staged = tmp_path / "tests/skill_evaluator"
    payload = trajectory()
    script = """
import json
import sys
sys.path.insert(0, sys.argv[1])
class NoInstalledPackage:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'skillevaluator' or fullname.startswith('skillevaluator.'):
            raise ModuleNotFoundError('Installed package access is forbidden in staged verifier test')
sys.meta_path.insert(0, NoInstalledPackage())
import atif_helpers
import importlib.util
spec = importlib.util.spec_from_file_location('staged_eval', sys.argv[1] + '/eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
trajectory = json.loads(sys.argv[2])
print(json.dumps({'calls': module.extract_tool_calls_as_dicts(trajectory),
                  'read': module.get_read_calls(trajectory),
                  'bundles': module.build_metric_evidence_bundles(trajectory, 'Write result'),
                  'shared': atif_helpers.build_metric_evidence_bundles(trajectory, 'Write result')}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(staged), json.dumps(payload)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["read"] == ["cat skills/example/SKILL.md"]
    assert output["calls"] == atif_helpers.extract_tool_calls_as_dicts(payload)
    assert output["bundles"] == output["shared"] == atif_helpers.build_metric_evidence_bundles(payload, "Write result")


def test_staged_evidence_runs_on_python39_with_normalized_child_calls(tmp_path):
    import shutil
    from pathlib import Path

    import idna
    import pytest

    interpreter = Path("/usr/bin/python3")
    if not interpreter.is_file():
        pytest.skip("System Python 3.9 is unavailable")
    version = subprocess.run(
        [str(interpreter), "-I", "-c", "import sys; print(sys.version_info[:2])"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    if version != "(3, 9)":
        pytest.skip("System interpreter is not Python 3.9")
    _copy_verifier(tmp_path)
    staged = tmp_path / "tests/skill_evaluator"
    # idna is an existing verifier dependency. Make its installed pure-Python
    # files available without granting the subprocess access to the host package.
    shutil.copytree(Path(idna.__file__).parent, staged / "idna")
    payload = {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "outer",
                        "function_name": "exec",
                        "arguments": {
                            "input": 'const read = await tools.exec_command({"cmd":"cat /skills/example/SKILL.md"});\n'
                            'const run = await tools.exec_command({"cmd":"printf completed"});'
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "outer", "content": "ambiguous outer observation"}]},
            }
        ]
    }
    script = """
import importlib.util
import json
import sys
sys.path.insert(0, sys.argv[1])
class NoInstalledPackage:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'skillevaluator' or fullname.startswith('skillevaluator.'):
            raise ModuleNotFoundError('Installed package access is forbidden')
sys.meta_path.insert(0, NoInstalledPackage())
import atif_helpers
spec = importlib.util.spec_from_file_location('staged_eval39', sys.argv[1] + '/eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
trajectory = json.loads(sys.argv[2])
print(json.dumps({'reads': module.get_read_calls(trajectory),
                  'calls': module.extract_tool_calls_as_dicts(trajectory),
                  'evidence': module.build_metric_evidence_bundles(trajectory, 'Inspect skill'),
                  'shared': atif_helpers.build_metric_evidence_bundles(trajectory, 'Inspect skill')}))
"""
    result = subprocess.run(
        [str(interpreter), "-I", "-c", script, str(staged), json.dumps(payload)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["reads"] == ["cat /skills/example/SKILL.md"]
    assert [call["action"] for call in output["calls"]] == ["exec_command", "exec_command"]
    assert [call["observation_status"] for call in output["calls"]] == ["unobserved_inner_call"] * 2
    assert (
        output["evidence"] == output["shared"] == atif_helpers.build_metric_evidence_bundles(payload, "Inspect skill")
    )
