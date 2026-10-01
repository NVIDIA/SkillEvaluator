# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core import atif_helpers
from skillevaluator.tier3.eval_core.atif_helpers import (
    build_behavior_evidence,
    build_conversation_summary,
)


def _trajectory_with_late_write() -> dict:
    steps = [
        {
            "source": "user",
            "message": "Update this Polars LazyFrame test suite for GPU execution.",
        }
    ]

    for idx in range(12):
        tool_id = f"read-{idx}"
        steps.append(
            {
                "source": "agent",
                "message": f"Executed Read {tool_id}",
                "tool_calls": [
                    {
                        "tool_call_id": tool_id,
                        "function_name": "Read",
                        "arguments": {"file_path": f"/workspace/input/file_{idx}.py"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": tool_id,
                            "content": "early exploration output " + ("x" * 700),
                        }
                    ]
                },
            }
        )

    steps.append(
        {
            "source": "agent",
            "message": "Executed Write write-1",
            "tool_calls": [
                {
                    "tool_call_id": "write-1",
                    "function_name": "Write",
                    "arguments": {
                        "file_path": "/workspace/output/test_gpu_engine_selection.py",
                        "content": (
                            "import polars as pl\n"
                            "def test_gpu_engine_strict(lazy_query):\n"
                            "    engine = pl.GPUEngine(raise_on_fail=True)\n"
                            '    lazy_query.collect(engine="gpu")\n'
                        ),
                    },
                }
            ],
            "observation": {
                "results": [
                    {
                        "source_call_id": "write-1",
                        "content": ("File created successfully at: /workspace/output/test_gpu_engine_selection.py"),
                    }
                ]
            },
        }
    )
    steps.append(
        {
            "source": "agent",
            "message": "Wrote /workspace/output/test_gpu_engine_selection.py.",
        }
    )
    return {"steps": steps}


def test_behavior_evidence_prioritizes_late_write_tool_calls() -> None:
    traj = _trajectory_with_late_write()
    old_summary = build_conversation_summary(traj, "question")

    assert "Agent called: Write" not in old_summary[:4000]

    evidence = build_behavior_evidence(traj, "question")

    assert len(evidence) <= 4000
    assert "FILE CHANGES" in evidence
    assert "Agent called: Write" in evidence
    assert evidence.find("Agent called: Read") == -1 or (
        evidence.index("Agent called: Write") < evidence.index("Agent called: Read")
    )
    assert "/workspace/output/test_gpu_engine_selection.py" in evidence
    assert "pl.GPUEngine(raise_on_fail=True)" in evidence
    assert 'collect(engine="gpu")' in evidence


def _load_harbor_template_module():
    template_path = Path(__file__).parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    spec = importlib.util.spec_from_file_location("harbor_eval_template", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harbor_template_behavior_evidence_matches_shared_helper() -> None:
    module = _load_harbor_template_module()

    evidence = module.build_behavior_evidence(_trajectory_with_late_write(), "question")

    assert len(evidence) <= 4000
    assert "FILE CHANGES" in evidence
    assert "Agent called: Write" in evidence
    assert "pl.GPUEngine(raise_on_fail=True)" in evidence


def test_metric_evidence_refs_link_judges_to_trajectory_and_expected_artifacts() -> None:
    traj = _trajectory_with_metric_refs()

    refs = atif_helpers.build_metric_evidence_refs(
        traj,
        "Run the evaluator and save /logs/agent/string-check-job-results.json.",
        ground_truth="The agent should complete the job and save /logs/agent/string-check-job-results.json.",
        expected_behavior=[
            "Run /app/.venv/bin/nemo evaluator info",
            "Save /logs/agent/string-check-job-results.json",
        ],
    )

    assert set(refs) == {"accuracy", "goal_accuracy", "behavior_check"}
    assert any(
        ref["source"] == "trajectory.json" and ref["json_pointer"] == "/steps/4" and ref["kind"] == "final_response"
        for ref in refs["accuracy"]
    )
    assert any(
        ref["source"] == "trajectory.json"
        and ref["json_pointer"] == "/steps/2/tool_calls/0"
        and ref["kind"] == "tool_call"
        for ref in refs["goal_accuracy"]
    )
    assert any(
        ref["source"] == "trajectory.json"
        and ref["json_pointer"] == "/steps/2/observation/results/0"
        and ref["kind"] == "tool_observation"
        for ref in refs["goal_accuracy"]
    )
    assert any(
        ref["source"] == "trajectory.json"
        and ref["json_pointer"] == "/steps/3/tool_calls/0"
        and ref["kind"] == "file_change"
        for ref in refs["behavior_check"]
    )
    assert any(
        ref["source"] == "evals.json"
        and ref["kind"] == "expected_artifact"
        and ref["path"] == "/logs/agent/string-check-job-results.json"
        for ref in refs["goal_accuracy"]
    )
    assert not any(
        ref["source"] == "evals.json"
        and ref["kind"] == "expected_artifact"
        and ref.get("path") == "/app/.venv/bin/nemo"
        for metric_refs in refs.values()
        for ref in metric_refs
    )
    assert all(len(str(ref.get("excerpt", ""))) <= 300 for metric_refs in refs.values() for ref in metric_refs)


def test_harbor_template_metric_evidence_refs_match_shared_helper() -> None:
    module = _load_harbor_template_module()

    traj = _trajectory_with_metric_refs()
    shared_refs = atif_helpers.build_metric_evidence_refs(
        traj,
        "Run the evaluator and save /logs/agent/string-check-job-results.json.",
        ground_truth="The agent should complete the job and save /logs/agent/string-check-job-results.json.",
        expected_behavior=[
            "Run /app/.venv/bin/nemo evaluator info",
            "Save /logs/agent/string-check-job-results.json",
        ],
    )
    template_refs = module.build_metric_evidence_refs(
        traj,
        "Run the evaluator and save /logs/agent/string-check-job-results.json.",
        ground_truth="The agent should complete the job and save /logs/agent/string-check-job-results.json.",
        expected_behavior=[
            "Run /app/.venv/bin/nemo evaluator info",
            "Save /logs/agent/string-check-job-results.json",
        ],
    )

    assert template_refs == shared_refs


def test_attach_metric_evidence_refs_preserves_existing_judge_details() -> None:
    refs = {
        "accuracy": [{"source": "trajectory.json", "json_pointer": "/steps/4", "kind": "final_response"}],
        "goal_accuracy": [{"source": "trajectory.json", "json_pointer": "/steps/2/tool_calls/0", "kind": "tool_call"}],
        "behavior_check": [
            {"source": "evals.json", "json_pointer": "/expected_behavior/0", "kind": "expected_behavior"}
        ],
    }
    details = {
        "accuracy": {"score": 1.0, "reason": "ok"},
        "goal_accuracy": {"score": 0.5, "reason": "partial", "method": "custom"},
        "behavior_check": {"score": 1.0, "results": [{"passed": True}]},
        "security": {"score": 1.0},
    }

    atif_helpers.attach_metric_evidence_refs(details, refs)

    assert details["accuracy"]["score"] == 1.0
    assert details["goal_accuracy"]["method"] == "custom"
    assert details["behavior_check"]["results"] == [{"passed": True}]
    assert "evidence_refs" not in details["security"]
    assert details["accuracy"]["evidence_refs"] == refs["accuracy"]
    assert details["goal_accuracy"]["evidence_refs"] == refs["goal_accuracy"]
    assert details["behavior_check"]["evidence_refs"] == refs["behavior_check"]


def test_metric_evidence_refs_redact_secret_like_values_in_all_fields() -> None:
    secret_path = "/logs/agent/nvapi-AbCdEfGh12345678.json"
    sha256_token = "sha256~abcdefghijklmnop"
    cursor_token = "crsr_deadbeefcafebabe"
    jwt_token = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJhZG1pbiI6dHJ1ZX0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    traj = {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "write-secret-path",
                        "function_name": "Write",
                        "arguments": {
                            "file_path": secret_path,
                            "content": (f"tokens sk-AbCdEfGh12345678 {sha256_token} {cursor_token} {jwt_token}"),
                        },
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "write-secret-path",
                            "content": f"wrote {secret_path} with {sha256_token} and {cursor_token}",
                        }
                    ]
                },
            },
            {"source": "agent", "message": f"Saved token sk-AbCdEfGh12345678 and {jwt_token}"},
        ]
    }

    refs = atif_helpers.build_metric_evidence_refs(
        traj,
        "Save the result.",
        ground_truth=f"Save {secret_path} after login {sha256_token}",
        expected_behavior=[f"Save {secret_path} using {cursor_token} and {jwt_token}"],
    )

    rendered = json.dumps(refs)
    for raw_secret in (
        "nvapi-AbCdEfGh12345678",
        "sk-AbCdEfGh12345678",
        sha256_token,
        cursor_token,
        jwt_token,
    ):
        assert raw_secret not in rendered
    assert "nvapi-<redacted>" in rendered
    assert "sk-<redacted>" in rendered
    assert "sha256~<redacted>" in rendered
    assert "crsr_<redacted>" in rendered
    assert "jwt-<redacted>" in rendered


def test_harbor_template_metric_evidence_refs_redact_harbor_secret_shapes() -> None:
    module = _load_harbor_template_module()
    sha256_token = "sha256~abcdefghijklmnop"
    cursor_token = "crsr_deadbeefcafebabe"
    jwt_token = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJhZG1pbiI6dHJ1ZX0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    refs = module.build_metric_evidence_refs(
        {
            "steps": [
                {
                    "source": "agent",
                    "tool_calls": [
                        {
                            "tool_call_id": "bash-secret",
                            "function_name": "Bash",
                            "arguments": {"command": f"nemo auth --token {sha256_token} --cursor {cursor_token}"},
                        }
                    ],
                    "observation": {"results": [{"source_call_id": "bash-secret", "content": jwt_token}]},
                }
            ]
        },
        "Run auth.",
        ground_truth=f"Do not leak {sha256_token}",
        expected_behavior=[f"Do not leak {cursor_token} or {jwt_token}"],
    )

    rendered = json.dumps(refs)
    assert sha256_token not in rendered
    assert cursor_token not in rendered
    assert jwt_token not in rendered
    assert "sha256~<redacted>" in rendered
    assert "crsr_<redacted>" in rendered
    assert "jwt-<redacted>" in rendered


def _trajectory_with_leaked_runtime_key(*key_values: str) -> dict:
    """Trajectory whose tool call, tool output, and final response leak *key_values*."""
    joined = " ".join(key_values)
    return {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "bash-runtime-key",
                        "function_name": "Bash",
                        "arguments": {"command": f"curl -H 'Authorization: Bearer {joined}' https://api.test"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "bash-runtime-key",
                            "content": f"request authorized with key {joined}",
                        }
                    ]
                },
            },
            {"source": "agent", "message": f"Done; authenticated using {joined}."},
        ]
    }


@pytest.mark.parametrize("env_var", ["NVIDIA_API_KEY"])
def test_metric_evidence_refs_redact_runtime_api_key_values(monkeypatch, env_var) -> None:
    # Runtime key VALUES need not match sk-/nvapi- shapes, so pattern-based
    # redaction alone would let them survive into persisted evidence_refs.
    key_value = "zzz-not-pattern-shaped-1234567890"
    for var in ("NVIDIA_API_KEY",):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(env_var, key_value)

    refs = atif_helpers.build_metric_evidence_refs(
        _trajectory_with_leaked_runtime_key(key_value),
        "Authenticate against the API.",
        ground_truth=f"Authenticate without echoing {key_value}",
        expected_behavior=[f"Never print {key_value} to the terminal"],
    )

    rendered = json.dumps(refs)
    assert key_value not in rendered
    assert "<redacted>" in rendered


def test_harbor_template_and_shared_redact_runtime_api_key_values_identically(monkeypatch) -> None:
    module = _load_harbor_template_module()

    api_key = "qqq-other-runtime-secret-0987654321"
    monkeypatch.setenv("NVIDIA_API_KEY", api_key)

    traj = _trajectory_with_leaked_runtime_key(api_key)
    kwargs = {
        "ground_truth": f"Authenticate without echoing {api_key}",
        "expected_behavior": [f"Never print {api_key} to the terminal"],
    }
    shared_refs = atif_helpers.build_metric_evidence_refs(traj, "Authenticate against the API.", **kwargs)
    template_refs = module.build_metric_evidence_refs(traj, "Authenticate against the API.", **kwargs)

    assert template_refs == shared_refs
    rendered = json.dumps(shared_refs)
    assert api_key not in rendered
    assert "<redacted>" in rendered


def test_harbor_template_main_persists_evidence_refs_in_reward_json(monkeypatch, tmp_path) -> None:
    module = _load_harbor_template_module()

    logs_dir = tmp_path / "logs"
    agent_dir = logs_dir / "agent"
    verifier_dir = logs_dir / "verifier"
    tests_dir = tmp_path / "tests"
    agent_dir.mkdir(parents=True)
    verifier_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    trajectory_path = agent_dir / "trajectory.json"
    entry_path = tests_dir / "entry.json"
    reward_json = verifier_dir / "reward.json"
    reward_txt = verifier_dir / "reward.txt"

    trajectory_path.write_text(json.dumps(_trajectory_with_metric_refs()), encoding="utf-8")
    entry_path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "question": "Run the evaluator and save /logs/agent/string-check-job-results.json.",
                "ground_truth": "The agent should complete the job and save /logs/agent/string-check-job-results.json.",
                "expected_behavior": ["Save /logs/agent/string-check-job-results.json"],
                "should_trigger": False,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "ATIF_PATH", trajectory_path)
    monkeypatch.setattr(module, "ENTRY_PATH", entry_path)
    monkeypatch.setattr(module, "REWARD_JSON", reward_json)
    monkeypatch.setattr(module, "REWARD_TXT", reward_txt)

    def fake_ragas(*args, **kwargs):
        raise RuntimeError("force custom judge")

    def fake_call_public_llm(prompt, *args, **kwargs):
        if "For each criterion" in prompt:
            return json.dumps(
                {
                    "criteria": {
                        "SKILL_IDENTIFIED": True,
                        "ACTION_CORRECT": True,
                        "FACTUALLY_ACCURATE": True,
                        "TASK_ADDRESSED": True,
                        "ACTIONABLE": True,
                    },
                    "score": 1.0,
                    "reason": "ok",
                }
            ), None
        if "Did the agent achieve the expected goal?" in prompt:
            return json.dumps({"achieved": True, "score": 1.0, "reason": "ok"}), None
        return json.dumps(
            {
                "results": [{"step": 1, "passed": True, "reason": "file saved"}],
                "score": 1.0,
                "summary": "ok",
            }
        ), None

    monkeypatch.setattr(module, "_judge_goal_accuracy_ragas", fake_ragas)
    monkeypatch.setattr(module, "call_public_llm", fake_call_public_llm)
    monkeypatch.setattr(
        module,
        "_call_public_llm_with_provenance",
        lambda prompt, *args, **kwargs: (*fake_call_public_llm(prompt, *args, **kwargs), {}),
    )

    module.main()

    reward = json.loads(reward_json.read_text(encoding="utf-8"))
    assert "details" not in reward
    details = json.loads((verifier_dir / "skill_evaluator_reward.json").read_text(encoding="utf-8"))["details"]
    assert details["accuracy"]["evidence_refs"]
    assert details["goal_accuracy"]["evidence_refs"]
    assert details["behavior_check"]["evidence_refs"]
    assert any(
        ref["source"] == "trajectory.json" and ref["json_pointer"] == "/steps/2/tool_calls/0"
        for ref in details["goal_accuracy"]["evidence_refs"]
    )
    assert any(
        ref["source"] == "evals.json" and ref.get("path") == "/logs/agent/string-check-job-results.json"
        for ref in details["behavior_check"]["evidence_refs"]
    )


def _trajectory_with_metric_refs() -> dict:
    return {
        "steps": [
            {
                "source": "user",
                "message": "Run the evaluator and save /logs/agent/string-check-job-results.json.",
            },
            {
                "source": "agent",
                "message": "I will submit the evaluation job and persist the result JSON.",
            },
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "submit-1",
                        "function_name": "Bash",
                        "arguments": {"command": "nemo evaluator evaluate submit --config /tmp/string-check.yaml"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "submit-1",
                            "content": "Evaluation job completed with artifact URL https://example.test/job/1",
                        }
                    ]
                },
            },
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "write-1",
                        "function_name": "Write",
                        "arguments": {
                            "file_path": "/logs/agent/string-check-job-results.json",
                            "content": '{"accuracy": 1.0}',
                        },
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "write-1",
                            "content": "File created at /logs/agent/string-check-job-results.json",
                        }
                    ]
                },
            },
            {
                "source": "agent",
                "message": "Done: saved /logs/agent/string-check-job-results.json.",
            },
        ]
    }


def test_behavior_evidence_keeps_write_when_prompt_is_long() -> None:
    traj = _trajectory_with_late_write()
    long_question = "Update the suite.\n" + ("existing test code\n" * 500)

    evidence = build_behavior_evidence(traj, long_question)

    assert len(evidence) <= 4000
    assert evidence.startswith("FILE CHANGES")
    assert "Agent called: Write" in evidence
    assert "/workspace/output/test_gpu_engine_selection.py" in evidence


def test_behavior_evidence_does_not_treat_read_like_tool_names_as_writes() -> None:
    traj = {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "read-edited",
                        "function_name": "read_edited_file",
                        "arguments": {
                            "file_path": "/workspace/output/noise.py",
                            "content": "this should not count",
                        },
                    },
                    {
                        "tool_call_id": "overwrite-guard",
                        "function_name": "overwrite_guard",
                        "arguments": {
                            "file_path": "/workspace/output/noise2.py",
                            "content": "this should not count",
                        },
                    },
                    {
                        "tool_call_id": "preview-write",
                        "function_name": "preview_write",
                        "arguments": {
                            "file_path": "/workspace/output/noise3.py",
                            "content": "this should not count",
                        },
                    },
                    {
                        "tool_call_id": "real-write",
                        "function_name": "tools.write_file",
                        "arguments": {
                            "file_path": "/workspace/output/real.py",
                            "content": "print('real')",
                        },
                    },
                ],
                "observation": {
                    "results": [
                        {"source_call_id": "read-edited", "content": "read ok"},
                        {"source_call_id": "overwrite-guard", "content": "guard ok"},
                        {"source_call_id": "preview-write", "content": "preview ok"},
                        {"source_call_id": "real-write", "content": "wrote file"},
                    ]
                },
            }
        ]
    }

    evidence = build_behavior_evidence(traj, "question")
    file_changes = evidence.split("FINAL RESPONSE", 1)[0].split("COMPACT TOOL HISTORY", 1)[0]

    assert "/workspace/output/real.py" in file_changes
    assert "/workspace/output/noise.py" not in file_changes
    assert "/workspace/output/noise2.py" not in file_changes
    assert "/workspace/output/noise3.py" not in file_changes


def test_harbor_template_behavior_judge_keeps_late_tail_evidence(monkeypatch) -> None:
    module = _load_harbor_template_module()
    prompts: list[str] = []

    def fake_call_public_llm(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(
            {
                "results": [{"step": 1, "passed": True, "reason": "write observed"}],
                "summary": "ok",
            }
        ), None

    monkeypatch.setattr(module, "call_public_llm", fake_call_public_llm)
    conversation = ("early exploration\n" * 400) + ('Agent called: Write({"file_path": "/workspace/output/test.py"})\n')

    result = module.judge_behavior_check(conversation, ["writes the output file"])

    assert result["score"] == 1.0
    assert "Agent called: Write" in prompts[0]


def test_behavior_evidence_ignores_non_write_shell_gt() -> None:
    traj = {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "bash-1",
                        "function_name": "Bash",
                        "arguments": {"command": "python3 -c 'print(3 > 2)' 2>&1"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "bash-1", "content": "True"}]},
            },
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "bash-2",
                        "function_name": "Bash",
                        "arguments": {"command": "cat <<'EOF' > /workspace/output/result.txt\nok\nEOF"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "bash-2", "content": "wrote file"}]},
            },
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "bash-3",
                        "function_name": "Bash",
                        "arguments": {
                            "command": (
                                "python3 - <<'PY'\n"
                                "from pathlib import Path\n"
                                "Path('/workspace/output/from_python.txt').write_text('ok')\n"
                                "PY"
                            )
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "bash-3", "content": "wrote python file"}]},
            },
        ]
    }

    evidence = build_behavior_evidence(traj, "question")
    file_changes = evidence.split("FINAL RESPONSE", 1)[0].split("COMPACT TOOL HISTORY", 1)[0]

    assert "print(3 > 2)" not in file_changes
    assert "/workspace/output/result.txt" in file_changes
    assert "/workspace/output/from_python.txt" in file_changes


def _trajectory_with_long_final_response() -> dict:
    """Build a synthetic trajectory with a long agent response."""
    body = "START_MARKER_" + ("A" * 1700) + "_LATE_MARKER_" + ("B" * 500) + "_TAIL"
    return {
        "steps": [
            {
                "source": "user",
                "message": "Verify onboarding setup",
            },
            {
                "source": "agent",
                "message": body,
            },
        ]
    }


# Cut text carries a "...[N chars truncated]..." marker; the tool history marks
# its own (shorter) copy of the final answer too, so look at FINAL RESPONSE.
_CUT_MARKER = "chars truncated]..."


def _final_response_section(evidence: str) -> str:
    section = evidence.split("FINAL RESPONSE\n", 1)[1]
    return section.split("\n\nUSER REQUEST\n", 1)[0]


def test_behavior_evidence_final_response_default_and_env_override(monkeypatch) -> None:
    traj = _trajectory_with_long_final_response()

    # By default, final response is capped at 800 chars and truncated
    default_evidence = build_behavior_evidence(traj, "question")
    assert "START_MARKER_" in default_evidence
    assert "_LATE_MARKER_" not in default_evidence
    assert _CUT_MARKER in _final_response_section(default_evidence)

    # Overriding via environment variable expands the capture limit
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", "3000")
    expanded_evidence = build_behavior_evidence(traj, "question", max_chars=8000)
    assert "START_MARKER_" in expanded_evidence
    assert "_LATE_MARKER_" in expanded_evidence
    assert _CUT_MARKER not in _final_response_section(expanded_evidence)


def test_behavior_evidence_final_response_explicit_arg() -> None:
    traj = _trajectory_with_long_final_response()

    evidence = build_behavior_evidence(traj, "question", max_chars=8000, final_response_limit=3000)
    assert "START_MARKER_" in evidence
    assert "_LATE_MARKER_" in evidence
    assert _CUT_MARKER not in _final_response_section(evidence)


def test_behavior_evidence_standalone_reconciles_max_chars(monkeypatch) -> None:
    traj = _trajectory_with_long_final_response()

    # When final_response_limit is explicitly passed without max_chars, max_chars expands
    evidence = build_behavior_evidence(traj, "question", final_response_limit=3000)
    assert "START_MARKER_" in evidence
    assert "_LATE_MARKER_" in evidence
    assert _CUT_MARKER not in _final_response_section(evidence)

    # When SKILL_EVAL_BEHAVIOR_CHECK_BUDGET is set in env, standalone call uses it
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", "12000")
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", "3000")
    env_evidence = build_behavior_evidence(traj, "question")
    assert "START_MARKER_" in env_evidence
    assert "_LATE_MARKER_" in env_evidence


def test_behavior_check_budget_and_compactor_env_override(monkeypatch) -> None:
    from skillevaluator.tier3.eval_core.llm_judge import _compact_behavior_conversation

    sample_conversation = "A" * 9000

    # Default compactor threshold is 8000
    compacted = _compact_behavior_conversation(sample_conversation)
    assert "...[middle truncated for behavior check]..." in compacted
    assert len(compacted) <= 8000

    # Overriding SKILL_EVAL_BEHAVIOR_CHECK_BUDGET raises compactor threshold
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", "12000")
    uncompacted = _compact_behavior_conversation(sample_conversation)
    assert "...[middle truncated for behavior check]..." not in uncompacted
    assert len(uncompacted) == 9000


def test_behavior_check_budget_auto_reconciles_tool_history_headroom(monkeypatch) -> None:
    from skillevaluator.tier3.eval_core import atif_helpers
    from skillevaluator.tier3.eval_core.llm_judge import _behavior_check_budget as judge_budget

    # If only final response limit is set to a large value, budget expands with headroom (4000)
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", "8000")
    assert atif_helpers._behavior_check_budget() == 12000
    assert judge_budget() == 12000

    # Equal explicit budget (8000) and exceeded explicit budget (6000) both reserve tool-history headroom (4000)
    for explicit_budget in ("8000", "6000"):
        monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", explicit_budget)
        assert atif_helpers._behavior_check_budget() == 12000
        assert judge_budget() == 12000


def _trajectory_with_failed_verification_and_long_claim(*, include_file_changes: bool = False) -> dict:
    """Build a trajectory with a failed command (and optional large file writes) before a 9,000-char success claim."""
    steps: list[dict] = [
        {
            "source": "user",
            "message": "Run the verification suite and confirm all checks pass.",
        }
    ]
    if include_file_changes:
        for idx in range(2):
            steps.append(
                {
                    "source": "agent",
                    "tool_calls": [
                        {
                            "tool_call_id": f"write-{idx}",
                            "function_name": "Write",
                            "arguments": {
                                "file_path": f"/workspace/output/module_{idx}.py",
                                "content": f"# generated module {idx}\n" + ("x = 1\n" * 400),
                            },
                        }
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": f"write-{idx}",
                                "content": f"Wrote /workspace/output/module_{idx}.py",
                            }
                        ]
                    },
                }
            )
    steps.append(
        {
            "source": "agent",
            "tool_calls": [
                {
                    "tool_call_id": "verify-1",
                    "function_name": "Bash",
                    "arguments": {"command": "pytest -q tests/verify_suite.py"},
                }
            ],
            "observation": {
                "results": [
                    {
                        "source_call_id": "verify-1",
                        "content": "FAILED tests/verify_suite.py::test_core - AssertionError: verification failed (exit 1)",
                    }
                ]
            },
        }
    )
    steps.append(
        {
            "source": "agent",
            "message": "CLAIM_START: All verification checks succeeded!\n" + ("S" * 9000) + "\nCLAIM_END",
        }
    )
    return {"steps": steps}


@pytest.mark.parametrize("budget_env", ["8000", "6000"])
@pytest.mark.parametrize("include_file_changes", [False, True], ids=["no_file_changes", "with_file_changes"])
def test_explicit_budget_preserves_tool_history_with_long_final_response(
    monkeypatch,
    budget_env: str,
    include_file_changes: bool,
) -> None:
    template_module = _load_harbor_template_module()
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", budget_env)
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", "8000")

    traj = _trajectory_with_failed_verification_and_long_claim(include_file_changes=include_file_changes)
    question = "Run the verification suite and confirm all checks pass."

    shared_bundles = atif_helpers.build_metric_evidence_bundles(
        traj,
        question,
        expected_behavior=["Run pytest verification and report failures accurately"],
    )
    template_bundles = template_module.build_metric_evidence_bundles(
        traj,
        question,
        expected_behavior=["Run pytest verification and report failures accurately"],
    )

    for impl_name, bundle in (
        ("shared", shared_bundles["behavior_check"]["prompt_evidence"]),
        ("template", template_bundles["behavior_check"]["prompt_evidence"]),
    ):
        assert "FINAL RESPONSE" in bundle, impl_name
        assert "CLAIM_START:" in bundle, impl_name
        assert "USER REQUEST" in bundle, impl_name
        assert "COMPACT TOOL HISTORY" in bundle, impl_name
        assert "pytest -q tests/verify_suite.py" in bundle, impl_name
        assert "AssertionError: verification failed" in bundle, impl_name

    assert shared_bundles["behavior_check"]["prompt_evidence"] == template_bundles["behavior_check"]["prompt_evidence"]


@pytest.mark.parametrize("max_chars", [8000, 6000])
def test_direct_max_chars_argument_reserves_tool_history_headroom(max_chars: int) -> None:
    template_module = _load_harbor_template_module()
    traj = _trajectory_with_failed_verification_and_long_claim(include_file_changes=False)
    question = "Run the verification suite and confirm all checks pass."

    shared_ev = build_behavior_evidence(traj, question, max_chars=max_chars, final_response_limit=8000)
    template_ev = template_module.build_behavior_evidence(
        traj, question, max_chars=max_chars, final_response_limit=8000
    )

    assert shared_ev == template_ev
    assert len(shared_ev) <= max_chars
    assert "FINAL RESPONSE" in shared_ev
    assert "COMPACT TOOL HISTORY" in shared_ev
    assert "AssertionError: verification failed" in shared_ev


@pytest.mark.parametrize("tiny_limit", [10, 19, 20, 43, 44, 60])
def test_behavior_evidence_and_compactor_handle_tiny_limits(monkeypatch, tiny_limit: int) -> None:
    from skillevaluator.tier3.eval_core.llm_judge import _compact_behavior_conversation

    template_module = _load_harbor_template_module()
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", "8000")
    traj = _trajectory_with_failed_verification_and_long_claim(include_file_changes=False)

    shared_ev = build_behavior_evidence(traj, "question", max_chars=tiny_limit, final_response_limit=8000)
    template_ev = template_module.build_behavior_evidence(
        traj, "question", max_chars=tiny_limit, final_response_limit=8000
    )
    assert len(shared_ev) <= tiny_limit
    assert shared_ev == template_ev

    long_text = "HEAD_CONTEXT_" + ("M" * 5000) + "_TAIL_OUTCOME"
    shared_compacted = _compact_behavior_conversation(long_text, limit=tiny_limit)
    template_compacted = template_module._compact_behavior_conversation(long_text, limit=tiny_limit)
    assert len(shared_compacted) <= tiny_limit
    assert shared_compacted == template_compacted


def test_seam_bundle_to_judge_behavior_check_retains_failed_command_and_claim(monkeypatch) -> None:
    from skillevaluator.tier3.eval_core import llm_judge

    template_module = _load_harbor_template_module()
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", "8000")
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", "8000")

    traj = _trajectory_with_failed_verification_and_long_claim(include_file_changes=False)
    question = "Run the verification suite and confirm all checks pass."
    behaviors = ["Run pytest verification and only claim success if tests pass"]

    shared_prompts: list[str] = []
    template_prompts: list[str] = []

    def fake_shared_llm(prompt: str, **kwargs):
        shared_prompts.append(prompt)
        return json.dumps({"results": [{"step": 1, "passed": False, "reason": "test failed"}], "summary": "fail"}), None

    def fake_template_llm(prompt: str, **kwargs):
        template_prompts.append(prompt)
        return json.dumps({"results": [{"step": 1, "passed": False, "reason": "test failed"}], "summary": "fail"}), None

    monkeypatch.setattr(llm_judge, "call_public_llm", fake_shared_llm)
    monkeypatch.setattr(template_module, "call_public_llm", fake_template_llm)

    shared_bundle = atif_helpers.build_metric_evidence_bundles(
        traj, question, expected_behavior=behaviors
    )["behavior_check"]["prompt_evidence"]
    template_bundle = template_module.build_metric_evidence_bundles(
        traj, question, expected_behavior=behaviors
    )["behavior_check"]["prompt_evidence"]

    shared_res = llm_judge.judge_behavior_check(shared_bundle, behaviors)
    template_res = template_module.judge_behavior_check(template_bundle, behaviors)

    assert shared_res["score"] == template_res["score"] == 0.0
    for prompt in (shared_prompts[0], template_prompts[0]):
        assert "CLAIM_START:" in prompt
        assert "AssertionError: verification failed" in prompt


def test_harbor_template_behavior_evidence_respects_custom_limits(monkeypatch) -> None:
    module = _load_harbor_template_module()
    traj = _trajectory_with_long_final_response()

    # Template default caps at 800
    evidence_default = module.build_behavior_evidence(traj, "question")
    assert "START_MARKER_" in evidence_default
    assert "_LATE_MARKER_" not in evidence_default
    assert _CUT_MARKER in _final_response_section(evidence_default)

    # Template override respects SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", "3000")
    evidence_custom = module.build_behavior_evidence(traj, "question", max_chars=8000)
    assert "START_MARKER_" in evidence_custom
    assert "_LATE_MARKER_" in evidence_custom
    assert _CUT_MARKER not in _final_response_section(evidence_custom)

    # Template helpers match shared atif_helpers logic
    assert module._behavior_final_response_limit() == atif_helpers._behavior_final_response_limit()
    assert module._behavior_check_budget() == atif_helpers._behavior_check_budget()


def test_verified_facts_preserve_final_response_in_behavior_judge_prompt(monkeypatch) -> None:
    from skillevaluator.tier3.eval_core import llm_judge

    template_module = _load_harbor_template_module()
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", "8000")
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", "8000")

    steps: list[dict] = [
        {
            "source": "user",
            "message": "Run all twelve verification checks and summarize the outcome.",
        },
        {
            "source": "agent",
            "tool_calls": [
                {
                    "tool_call_id": "write-0",
                    "function_name": "Write",
                    "arguments": {
                        "file_path": "/workspace/output/report.py",
                        "content": "# generated report\n" + ("value = 1\n" * 170),
                    },
                }
            ],
            "observation": {
                "results": [
                    {
                        "source_call_id": "write-0",
                        "content": "Wrote /workspace/output/report.py",
                    }
                ]
            },
        },
    ]
    behaviors: list[str] = []
    for idx in range(12):
        token = f"pytest -q tests/check_module_{idx:02d}.py --maxfail=1 --tb=short"
        behaviors.append(f"Execute `{token}` and confirm the final assertion")
        steps.append(
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": f"cmd-{idx}",
                        "function_name": "Bash",
                        "arguments": {"command": f"{token} && echo status_{idx:02d}_ok_" + ("x" * 90)},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": f"cmd-{idx}",
                            "content": f"passed check {idx:02d} " + ("y" * 80),
                        }
                    ]
                },
            }
        )

    final_claim = "FINAL_CLAIM_ASSERTION: All twelve verification modules passed without errors."
    steps.append(
        {
            "source": "agent",
            "message": final_claim + "\n" + ("P" * 6100) + "\nFINAL_TAIL_MARKER",
        }
    )
    traj = {"steps": steps}
    question = "Run all twelve verification checks and summarize the outcome."

    def make_judge_stub(captured_prompts: list[str], count: int):
        def _stub(prompt: str, **kwargs):
            captured_prompts.append(prompt)
            passed = final_claim in prompt
            score = 1.0 if passed else 0.0
            results = [{"step": i + 1, "passed": passed, "reason": "checked"} for i in range(count)]
            return json.dumps({"results": results, "score": score, "summary": "done"}), None

        return _stub

    shared_prompts_no_facts: list[str] = []
    monkeypatch.setattr(llm_judge, "call_public_llm", make_judge_stub(shared_prompts_no_facts, 1))
    no_facts_bundle = atif_helpers.build_metric_evidence_bundles(
        traj, question, expected_behavior=["Confirm the final assertion in the summary"]
    )["behavior_check"]
    no_facts_res = llm_judge.judge_behavior_check(
        no_facts_bundle["prompt_evidence"], ["Confirm the final assertion in the summary"]
    )
    assert no_facts_res["score"] == 1.0

    shared_prompts: list[str] = []
    template_prompts: list[str] = []
    monkeypatch.setattr(llm_judge, "call_public_llm", make_judge_stub(shared_prompts, len(behaviors)))
    monkeypatch.setattr(template_module, "call_public_llm", make_judge_stub(template_prompts, len(behaviors)))

    shared_bundle = atif_helpers.build_metric_evidence_bundles(
        traj, question, expected_behavior=behaviors
    )["behavior_check"]
    template_bundle = template_module.build_metric_evidence_bundles(
        traj, question, expected_behavior=behaviors
    )["behavior_check"]

    assert shared_bundle == template_bundle
    assert len(shared_bundle["verified"]) == 12
    assert len(shared_bundle["prompt_evidence"]) <= 12000
    assert shared_bundle["omitted"]["truncated"] is True
    assert shared_bundle["omitted"]["count"] == 1

    shared_res = llm_judge.judge_behavior_check(shared_bundle["prompt_evidence"], behaviors)
    template_res = template_module.judge_behavior_check(template_bundle["prompt_evidence"], behaviors)

    assert shared_res["score"] == 1.0
    assert template_res["score"] == 1.0
    for prompt in (shared_prompts[0], template_prompts[0]):
        assert "VERIFIED FACTS (deterministic):" in prompt
        assert final_claim in prompt
        assert "FINAL_TAIL_MARKER" in prompt


def test_compact_behavior_conversation_preserves_final_response_section(monkeypatch) -> None:
    from skillevaluator.tier3.eval_core import llm_judge

    template_module = _load_harbor_template_module()
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_CHECK_BUDGET", "8000")
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", "8000")

    facts_block = "VERIFIED FACTS (deterministic):\n" + "\n".join(
        f"- [OBSERVED step {i}] claim_{i:02d} :: " + ("f" * 140) for i in range(1, 13)
    )
    file_changes_block = "FILE CHANGES\nAgent called: Write\nPath: /workspace/output/out.py\n" + ("c" * 1850)
    final_claim = "FINAL_RESPONSE_CLAIM: The verification report is complete and accurate."
    final_block = f"FINAL RESPONSE\n{final_claim}\n" + ("R" * 6000) + "\nFINAL_RESPONSE_END"
    user_block = "USER REQUEST\nVerify the output and report."
    history_block = "COMPACT TOOL HISTORY\n" + ("H" * 2100) + "\nLAST_TOOL_OBSERVATION"

    oversized_bundle = f"{facts_block}\n\n{file_changes_block}\n\n{final_block}\n\n{user_block}\n\n{history_block}"
    assert len(oversized_bundle) > 12000

    shared_compacted = llm_judge._compact_behavior_conversation(oversized_bundle)
    template_compacted = template_module._compact_behavior_conversation(oversized_bundle)

    assert shared_compacted == template_compacted
    assert len(shared_compacted) <= 12000
    assert final_claim in shared_compacted
    assert "FINAL_RESPONSE_END" in shared_compacted
    assert "LAST_TOOL_OBSERVATION" in shared_compacted


@pytest.mark.parametrize(
    "case_name,limit,final_limit_env,text_builder,expected_substrings",
    [
        (
            "no_prefix_long_suffix",
            1200,
            "600",
            lambda: (
                "FINAL RESPONSE\nSTART_FINAL_CLAIM_OK\n"
                + ("F" * 300)
                + "\nEND_FINAL_CLAIM_OK\n\nUSER REQUEST\nRun check.\n\nCOMPACT TOOL HISTORY\n"
                + ("H" * 1500)
                + "\nTAIL_HISTORY_MARKER"
            ),
            ["START_FINAL_CLAIM_OK", "END_FINAL_CLAIM_OK", "TAIL_HISTORY_MARKER"],
        ),
        (
            "no_prefix_tiny_suffix_budget",
            420,
            "400",
            lambda: (
                "FINAL RESPONSE\n"
                + ("F" * 370)
                + "\nEND_FINAL\n\nCOMPACT TOOL HISTORY\n"
                + ("H" * 500)
                + "\nTINY_TAIL"
            ),
            ["FINAL RESPONSE", "END_FINAL"],
        ),
        (
            "long_prefix_no_suffix",
            1200,
            "500",
            lambda: (
                "FILE CHANGES\nHEAD_FILE_MARKER\n"
                + ("C" * 1500)
                + "\nLATE_FILE_MARKER\n\nFINAL RESPONSE\nFINAL_ONLY_SUFFIX_OK"
            ),
            ["HEAD_FILE_MARKER", "LATE_FILE_MARKER", "FINAL_ONLY_SUFFIX_OK"],
        ),
        (
            "long_prefix_no_suffix_tiny_rem",
            420,
            "400",
            lambda: (
                "FILE CHANGES\nHEAD_TINY_PRE\n"
                + ("C" * 600)
                + "\n\nFINAL RESPONSE\n"
                + ("F" * 370)
                + "\nFINAL_END_TINY_REM"
            ),
            ["FINAL RESPONSE", "FINAL_END_TINY_REM"],
        ),
        (
            "oversized_final_with_small_prefix_and_suffix",
            1500,
            "800",
            lambda: (
                "FILE CHANGES\nSHORT_PREFIX\n\nFINAL RESPONSE\nFINAL_HEAD_KEEP\n"
                + ("F" * 2000)
                + "\nFINAL_TAIL_KEEP\n\nUSER REQUEST\nSHORT_SUFFIX"
            ),
            ["SHORT_PREFIX", "FINAL_HEAD_KEEP", "FINAL_TAIL_KEEP", "SHORT_SUFFIX"],
        ),
        (
            "both_prefix_and_suffix_exceed_half_budgets",
            2000,
            "600",
            lambda: (
                "FILE CHANGES\nPRE_START_MARK\n"
                + ("P" * 1500)
                + "\nPRE_END_MARK\n\nFINAL RESPONSE\nCORE_FINAL_CLAIM\n\nCOMPACT TOOL HISTORY\nSUF_START_MARK\n"
                + ("S" * 1500)
                + "\nSUF_END_MARK"
            ),
            ["PRE_START_MARK", "PRE_END_MARK", "CORE_FINAL_CLAIM", "SUF_START_MARK", "SUF_END_MARK"],
        ),
        (
            "both_prefix_and_suffix_with_tiny_remaining_budget",
            440,
            "400",
            lambda: (
                "FILE CHANGES\n"
                + ("P" * 300)
                + "\n\nFINAL RESPONSE\n"
                + ("F" * 375)
                + "\nTINY_BOTH_FINAL\n\nCOMPACT TOOL HISTORY\n"
                + ("S" * 300)
            ),
            ["FINAL RESPONSE", "TINY_BOTH_FINAL"],
        ),
    ],
)
def test_compact_behavior_conversation_section_layout_permutations(
    monkeypatch,
    case_name: str,
    limit: int,
    final_limit_env: str,
    text_builder,
    expected_substrings: list[str],
) -> None:
    from skillevaluator.tier3.eval_core import llm_judge

    template_module = _load_harbor_template_module()
    monkeypatch.setenv("SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT", final_limit_env)

    raw_text = text_builder()
    assert len(raw_text) > limit, case_name

    shared_out = llm_judge._compact_behavior_conversation(raw_text, limit=limit)
    template_out = template_module._compact_behavior_conversation(raw_text, limit=limit)

    assert shared_out == template_out, case_name
    assert len(shared_out) <= limit, case_name
    for substr in expected_substrings:
        assert substr in shared_out, f"{case_name}: missing {substr!r}"

