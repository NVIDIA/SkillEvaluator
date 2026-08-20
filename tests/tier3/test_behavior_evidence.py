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


def test_harbor_template_behavior_evidence_matches_shared_helper() -> None:
    template_path = Path(__file__).parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    spec = importlib.util.spec_from_file_location("harbor_eval_template", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
    template_path = Path(__file__).parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    spec = importlib.util.spec_from_file_location("harbor_eval_template_refs", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
    template_path = Path(__file__).parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    spec = importlib.util.spec_from_file_location("harbor_eval_template_secret_refs", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    template_path = Path(__file__).parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    spec = importlib.util.spec_from_file_location("harbor_eval_template_env_keys", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
    template_path = Path(__file__).parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    spec = importlib.util.spec_from_file_location("harbor_eval_template_main_refs", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
    template_path = Path(__file__).parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
    spec = importlib.util.spec_from_file_location("harbor_eval_template_for_judge", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
