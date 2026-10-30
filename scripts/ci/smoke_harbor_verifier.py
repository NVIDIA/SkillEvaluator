#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage the installed verifier, then exercise it with a separate task Python.

The host needs the tier3 extra. The task interpreter needs only idna and must
not have skillevaluator installed. Use an actual Python 3.9 interpreter in CI:
newer Python parsers do not reproduce every older-interpreter syntax failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path


def check_staged(directory: Path) -> None:
    """Run inside the independent task interpreter, without host dependencies."""
    assert importlib.util.find_spec("skillevaluator") is None, "host package leaked into task interpreter"
    directory = directory.resolve()
    for source in directory.glob("*.py"):
        compile(source.read_text(encoding="utf-8"), str(source), "exec")
    verifier = runpy.run_path(str(directory / "eval.py"))
    shared_checks = verifier["_shared_checks"]
    assert Path(shared_checks.__file__).resolve() == directory / "checks.py"
    for name in ("shell_lexing", "script_invocation", "secret_redaction"):
        assert Path(sys.modules[name].__file__).resolve() == directory / (name + ".py")
    assert verifier["_shell_tokens"] is sys.modules["shell_lexing"]._shell_tokens
    assert verifier["check_error_recovery"] is shared_checks.check_error_recovery
    assert verifier["_shell_tokens"]("cat 2>/dev/null") == ["cat", "2>", "/dev/null"]
    for command, expected in [
        ("python scripts/calc.py", True),
        ("cat scripts/calc.py", False),
        ("sh -c 'python scripts/calc.py'", True),
    ]:
        assert verifier["_cmd_executes_script"](command, "scripts/calc.py") is expected
    calls = [
        {"action": "read", "action_input": {"path": "/skills/calculator/SKILL.md"}, "observation": "calculator"},
        {"action": "bash", "action_input": {"command": "python scripts/calc.py"}, "observation": "2"},
    ]
    results = {
        "activation": verifier["check_activation"](calls, "calculator"),
        "execution": verifier["check_script_execution"](calls, "scripts/calc.py"),
        "recovery": verifier["check_error_recovery"](calls, "scripts/calc.py"),
        "efficiency": verifier["check_tool_efficiency"](calls, "calculator", "scripts/calc.py"),
        "security": verifier["check_security"]({"steps": []}, calls, "calculator"),
    }
    assert all(result["passed"] for result in results.values()), results
    assert "details" not in results["efficiency"]
    # Exercise the recent root-search and cleanup rules on the task interpreter,
    # including both sides of each classification boundary.
    for command, score in [("find / -name calc.py", 0.0), ("find /workspace -name calc.py", 1.0)]:
        call = {"action": "exec_command", "action_input": {"cmd": command}}
        assert verifier["check_tool_efficiency"]([call], "calculator", "calc.py")["score"] == score
    for command, passed in [
        ("rm -rf /tmp/cache", True),
        ("rm -rf /tmp/cache /workspace/data", False),
        ("printf '%s\\n' /workspace/data | xargs rm -rf /tmp/cache", False),
        ("ln -s /workspace /tmp/cache; rm -rf /tmp/cache/data", False),
    ]:
        call = {"action": "bash", "action_input": {"command": command}}
        assert verifier["check_security"]({"steps": []}, [call])["passed"] is passed, command
    # HTTP-date Retry-After values need a timezone that exists on Python 3.9.
    assert verifier["_parse_retry_after"]("Thu, 01 Jan 1970 00:00:00 GMT", 7.0) == 0.0
    assert verifier["_parse_retry_after"]("Fri, 31 Dec 9999 23:59:59 GMT", 7.0) > 7.0
    reward = run_verifier_main(directory)
    print(
        json.dumps({"python": sys.version.split()[0], "verifier": str(directory), "results": results, "reward": reward})
    )


def run_verifier_main(directory: Path) -> dict[str, float]:
    """Score a small trajectory the way test.sh runs eval.py, without judge credentials."""
    entry = {
        "id": "smoke",
        "expected_skill": "calculator",
        "evaluated_skill": "calculator",
        "expected_script": "scripts/calc.py",
        "question": "What is 1 + 1?",
    }
    tool_calls = [
        {"tool_call_id": "c1", "function_name": "Read", "arguments": {"file_path": "/skills/calculator/SKILL.md"}},
        {"tool_call_id": "c2", "function_name": "Bash", "arguments": {"command": "python scripts/calc.py 1 + 1"}},
    ]
    results = [{"source_call_id": "c1", "content": "calculator"}, {"source_call_id": "c2", "content": "2"}]
    trajectory = {
        "steps": [
            {"source": "user", "message": entry["question"]},
            {"source": "agent", "message": "", "tool_calls": tool_calls, "observation": {"results": results}},
            {"source": "agent", "message": "1 + 1 = 2"},
        ]
    }
    with tempfile.TemporaryDirectory(prefix="harbor-verifier-main-") as temporary:
        root = Path(temporary)
        logs, tests, rewards = root / "logs", root / "tests", root / "logs" / "verifier"
        (logs / "agent").mkdir(parents=True)
        tests.mkdir()
        (tests / "entry.json").write_text(json.dumps(entry), encoding="utf-8")
        (logs / "agent" / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
        environment = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HARBOR_LOGS_DIR": str(logs),
            "HARBOR_TESTS_DIR": str(tests),
        }
        completed = subprocess.run(
            [sys.executable, "-I", str(directory / "eval.py")],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
        reward = json.loads((rewards / "reward.json").read_text(encoding="utf-8"))
        rich = json.loads((rewards / "skill_evaluator_reward.json").read_text(encoding="utf-8"))
        overall = float((rewards / "reward.txt").read_text(encoding="utf-8"))
    dimensions = ("security", "skill_execution", "skill_efficiency", "accuracy", "goal_accuracy", "behavior_check")
    assert reward == dict.fromkeys((*dimensions, "overall"), 1.0), reward
    assert overall == 1.0, overall
    assert rich["trajectory_source"] == "trajectory.json", rich
    execution = rich["details"]["skill_execution"]
    assert execution["activation"]["details"]["matched_skill"] == "calculator", execution
    assert execution["script_execution"]["passed"] is True, execution
    return reward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--python", type=Path, help="Separate task interpreter (Python 3.9 with idna)")
    mode.add_argument("--check-staged", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.check_staged:
        check_staged(arguments.check_staged)
        return

    from skillevaluator.tier3.harbor.adapter import _copy_verifier

    with tempfile.TemporaryDirectory(prefix="harbor-verifier-smoke-") as temporary:
        task = Path(temporary)
        _copy_verifier(task)
        subprocess.run(
            [
                str(arguments.python.absolute()),
                "-I",
                str(Path(__file__).resolve()),
                "--check-staged",
                str(task / "tests" / "skill_evaluator"),
            ],
            cwd=task,
            check=True,
        )


if __name__ == "__main__":
    main()
