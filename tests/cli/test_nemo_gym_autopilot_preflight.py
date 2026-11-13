# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym dataset preparation must use its own compatibility and runtime checks."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.evaluation import EvaluationService


@pytest.mark.parametrize("command", [["evaluate", "--autopilot"], ["tier3"]])
def test_keyed_gym_generates_with_its_endpoint_and_one_compatibility_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> None:
    """Harbor credential or Docker checks must not block a compatible Gym worker."""
    from skillevaluator.tier3.harbor import runner

    for name in os.environ:
        if name.startswith("SKILL_EVAL_") or name in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_JUDGE_MODEL"}:
            monkeypatch.delenv(name)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-generation-key")
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_MODEL_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_MODEL_API_KEY", "EMPTY")
    skill = tmp_path / "sample"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: sample\ndescription: Add numbers.\n---\n# Add\n")
    # A config without a dataset takes the real private source snapshot path.
    (skill / "evals").mkdir()
    (skill / "evals" / "config.yml").write_text("schema_version: 1\nharbor:\n  n_attempts: 1\n")
    probe_log = tmp_path / "probe-count"
    gym = tmp_path / "gym"
    gym.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "assert sys.argv[1:] == ['eval', 'run', '--help']\n"
        f"with pathlib.Path({str(probe_log)!r}).open('a') as log: log.write('probe\\n')\n"
        "print('usage: gym eval run TARGET --sandbox --agent --model-type --model --model-url --output --concurrency')\n"
        "print('\\npositional arguments:')\n"
        "print('  TARGET  Harbor task folder')\n"
    )
    gym.chmod(0o755)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_BIN", str(gym))
    monkeypatch.setattr(
        runner, "_check_prerequisites", lambda **_kwargs: pytest.fail("Gym must not check Harbor prerequisites")
    )

    def generate(_self, path, *, use_llm):
        assert use_llm is True
        (path / "evals" / "evals.json").write_text(
            json.dumps(
                {
                    "skill_name": "sample",
                    "evals": [{"id": "sum", "prompt": "Add 2 and 3.", "expected_output": "5"}],
                }
            )
        )

    monkeypatch.setattr(EvaluationService, "create_autopilot_dataset", generate)
    evaluated = []
    monkeypatch.setattr(
        EvaluationService,
        "evaluate",
        lambda _self, options, **_kwargs: evaluated.append(options) or {"execution_status": "succeeded"},
    )

    result = CliRunner().invoke(
        cli, [*command, str(skill), "--backend", "nemo-gym", "--agents", "claude-code", "--progress", "off"]
    )

    assert result.exit_code == 0, result.output
    assert (skill / "evals" / "evals.json").is_file()
    assert len(evaluated) == 1
    assert evaluated[0].backend == "nemo-gym"
    assert probe_log.read_text().splitlines() == ["probe"]


def test_keyed_incompatible_gym_does_not_generate_a_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-generation-key")
    skill = tmp_path / "sample"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: sample\ndescription: Add numbers.\n---\n# Add\n")
    gym = tmp_path / "gym"
    gym.write_text("#!/bin/sh\necho 'usage: gym eval run --input INPUT'\n")
    gym.chmod(0o755)
    monkeypatch.setenv("SKILL_EVAL_NEMO_GYM_BIN", str(gym))
    monkeypatch.setattr(
        EvaluationService, "create_autopilot_dataset", lambda *_args, **_kwargs: pytest.fail("must not generate")
    )

    result = CliRunner().invoke(
        cli, ["evaluate", str(skill), "--backend", "nemo-gym", "--autopilot", "--progress", "off"]
    )

    assert result.exit_code == 2, result.output
    assert "incompatible" in result.output
    assert not (skill / "evals").exists()
