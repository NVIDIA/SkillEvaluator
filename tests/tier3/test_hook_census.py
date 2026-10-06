# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hook execution census: the POSIX sh logger and its host-side collection."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.eval_core.plugin_signals import build_plugin_signals_context
from skillevaluator.tier3.eval_core.runtime_evidence import (
    HOOK_CENSUS_FILENAME,
    parse_hook_census,
    read_hook_census,
    summarize_hook_census,
)
from skillevaluator.tier3.harbor.collector import collect_harbor_results

SCRIPT = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "hook_census.sh"
)
SH = shutil.which("sh") if os.name == "posix" else None
needs_sh = pytest.mark.skipif(SH is None or not Path("/bin/sh").exists(), reason="requires a POSIX /bin/sh")


def _script(tmp_path: Path, census: Path) -> Path:
    """Copy the logger with the census path pointed at a temporary file."""
    text = SCRIPT.read_text(encoding="utf-8")
    default = "census_file=/logs/agent/skilleval-hook-census.jsonl"
    assert default in text
    copy = tmp_path / "hook_census.sh"
    copy.write_text(text.replace(default, f"census_file={census}"), encoding="utf-8")
    return copy


def _run(script: Path, *args: str, stdin: bytes = b"") -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["/bin/sh", str(script), *args], input=stdin, capture_output=True, check=False, timeout=30)


def _census(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_script_is_posix_sh_and_writes_the_contract_path() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n")
    assert "census_file=/logs/agent/skilleval-hook-census.jsonl" in text
    # No bashisms that /bin/sh (dash, BusyBox ash) would reject.
    # ANSI-C quoting is assembled from parts so the OSS boundary scanner does not
    # parse this test's own source as a shell string.
    ansi_c_quote = chr(36) + chr(39)
    for bashism in ("[[", ansi_c_quote, "local ", "function ", "<<<", "${!"):
        assert bashism not in text


@needs_sh
@pytest.mark.parametrize("code", [0, 1, 2, 7, 127])
def test_exit_code_passes_through_exactly(tmp_path: Path, code: int) -> None:
    census = tmp_path / "census.jsonl"
    result = _run(_script(tmp_path, census), "hook-a", "PreToolUse", "--", f"exit {code}")

    assert result.returncode == code
    [record] = _census(census)
    assert record["exit_code"] == code
    assert record["hook_id"] == "hook-a"
    assert record["event"] == "PreToolUse"
    assert isinstance(record["duration_ms"], int) and record["duration_ms"] >= 0
    assert record["started_at"].endswith("Z")


@needs_sh
def test_stdin_stdout_and_stderr_pass_through_byte_for_byte(tmp_path: Path) -> None:
    census = tmp_path / "census.jsonl"
    payload = b'{"tool_name":"Bash","tool_input":{"command":"ls"}}\n\x00binary\xff tail'
    script = _script(tmp_path, census)

    result = _run(script, "h", "PreToolUse", "--", "cat; printf 'to-stderr' >&2; exit 2", stdin=payload)

    assert result.stdout == payload
    assert result.stderr == b"to-stderr"
    assert result.returncode == 2


@needs_sh
def test_argv_form_keeps_arguments_intact(tmp_path: Path) -> None:
    census = tmp_path / "census.jsonl"
    result = _run(_script(tmp_path, census), "h", "Stop", "--", "printf", "%s|", "a b", "$HOME", "*")

    assert result.stdout == b"a b|$HOME|*|"
    assert result.returncode == 0


@needs_sh
def test_logger_never_writes_output_even_when_the_census_is_unwritable(tmp_path: Path) -> None:
    census = tmp_path / "missing-dir" / "census.jsonl"
    result = _run(_script(tmp_path, census), "h", "PostToolUse", "--", "exit 3")

    assert result.stdout == b""
    assert result.stderr == b""
    assert result.returncode == 3
    assert not census.exists()


@needs_sh
@pytest.mark.parametrize(
    "hook_id",
    [
        'quote"id',
        "back\\slash",
        "tab\there",
        "new\nline",
        "percent %s %n",
        "unicode ✓ café",
        "$(touch pwned)",
        "x" * 400,
    ],
)
def test_odd_hook_ids_are_json_escaped(tmp_path: Path, hook_id: str) -> None:
    census = tmp_path / "census.jsonl"
    result = _run(_script(tmp_path, census), hook_id, 'Pre"Tool', "--", "true")

    assert result.returncode == 0
    assert not (tmp_path / "pwned").exists()
    [record] = _census(census)
    expected = "".join(ch for ch in hook_id if ord(ch) >= 0x20 and ord(ch) != 0x7F).encode()[:256]
    assert record["hook_id"].encode() == expected
    assert record["event"] == 'Pre"Tool'


@needs_sh
def test_each_run_appends_one_line(tmp_path: Path) -> None:
    census = tmp_path / "census.jsonl"
    script = _script(tmp_path, census)
    for code in (0, 1, 0):
        _run(script, "h", "PreToolUse", "--", f"exit {code}")

    parsed = parse_hook_census(census.read_text(encoding="utf-8"))
    assert parsed["hooks"] == [
        {
            "hook_id": "h",
            "event": "PreToolUse",
            "runs": 3,
            "failures": 1,
            "blocked": 0,
            "not_started": 0,
            "total_duration_ms": parsed["hooks"][0]["total_duration_ms"],
        }
    ]
    assert parsed["total_runs"] == 3


@needs_sh
def test_usage_error_without_separator(tmp_path: Path) -> None:
    result = _run(_script(tmp_path, tmp_path / "c.jsonl"), "h", "PreToolUse", "true")

    assert result.returncode == 64
    assert result.stdout == b""
    assert b"usage" in result.stderr


def test_parse_hook_census_aggregates_and_counts_bad_lines() -> None:
    text = "\n".join(
        [
            json.dumps({"hook_id": "a", "event": "PreToolUse", "exit_code": 0, "duration_ms": 5}),
            json.dumps({"hook_id": "a", "event": "PreToolUse", "exit_code": 2, "duration_ms": 7}),
            json.dumps({"hook_id": "b", "event": "Stop", "exit_code": 0, "duration_ms": None}),
            "not json",
            json.dumps({"hook_id": "c", "event": "Stop"}),
            json.dumps(["list"]),
            "x" * 5000,
            "",
        ]
    )

    parsed = parse_hook_census(text)

    assert parsed["status"] == "recorded"
    # Exit 2 is a hook's deny decision: blocked, not a failure.
    assert parsed["hooks"] == [
        {
            "hook_id": "a",
            "event": "PreToolUse",
            "runs": 2,
            "failures": 0,
            "blocked": 1,
            "not_started": 0,
            "total_duration_ms": 12,
        },
        {
            "hook_id": "b",
            "event": "Stop",
            "runs": 1,
            "failures": 0,
            "blocked": 0,
            "not_started": 0,
            "total_duration_ms": 0,
        },
    ]
    assert parsed["total_runs"] == 3
    assert parsed["total_failures"] == 0
    assert parsed["total_blocked"] == 1
    assert parsed["invalid_lines"] == 4


def test_parse_hook_census_redacts_and_bounds_ids() -> None:
    secret = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    parsed = parse_hook_census(json.dumps({"hook_id": f"{secret} " + "y" * 600, "event": "E", "exit_code": 0}))

    [hook] = parsed["hooks"]
    assert secret not in hook["hook_id"]
    assert len(hook["hook_id"]) <= 256


# How far a label's raw text is read before redaction: plugin_signals._safe_text(value, 256).
_LABEL_WINDOW = 256 * 4 + 256


@pytest.mark.parametrize(
    ("token", "inside"),
    [
        ("AKIA" + "ABCDEFGHIJKLMNOP", 19),
        ("ghp_" + "Z9" * 18, 20),
        (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJyb290In0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            60,
        ),
    ],
    ids=["aws-key", "github-token", "jwt"],
)
def test_parse_hook_census_redacts_a_token_cut_by_the_label_window(token: str, inside: int) -> None:
    """A forged census line: a long JWT redacts to a short marker, which pulls the text after it into the
    256-character label, and the token there starts *inside* characters before the window ends."""
    lead = "eyJ" + "A" * 575 + ".eyJ" + "B" * 575 + "." + "C" * 40 + " "
    hook_id = lead + "x" * (_LABEL_WINDOW - len(lead) - inside - 1) + " " + token
    parsed = parse_hook_census(json.dumps({"hook_id": hook_id, "event": "PreToolUse", "exit_code": 0}))

    [hook] = parsed["hooks"]
    assert token[:12] not in hook["hook_id"]
    assert len(hook["hook_id"]) <= 256


def test_parse_hook_census_keeps_the_staged_hook_id_spelling() -> None:
    """Census rows join the staged hook ids by exact string, so spaces in a source path are kept."""
    hook_id = "hooks/my  hooks.json#PreToolUse[0].hooks[0]"
    parsed = parse_hook_census(json.dumps({"hook_id": hook_id, "event": "PreToolUse", "exit_code": 0}))

    assert [hook["hook_id"] for hook in parsed["hooks"]] == [hook_id]


def test_read_hook_census_absent_present_and_unsafe(tmp_path: Path) -> None:
    assert read_hook_census(tmp_path)["status"] == "absent"
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / HOOK_CENSUS_FILENAME).write_text(
        json.dumps({"hook_id": "h", "event": "E", "exit_code": 1}) + "\n", encoding="utf-8"
    )
    assert read_hook_census(tmp_path)["total_failures"] == 1

    if os.name == "posix":
        other = tmp_path / "elsewhere.jsonl"
        other.write_text("{}", encoding="utf-8")
        (agent / HOOK_CENSUS_FILENAME).unlink()
        (agent / HOOK_CENSUS_FILENAME).symlink_to(other)
        assert read_hook_census(tmp_path)["status"] == "unreadable"


def test_summarize_hook_census_over_trials() -> None:
    first = parse_hook_census(json.dumps({"hook_id": "a", "event": "E", "exit_code": 0}))
    second = parse_hook_census(
        "\n".join(json.dumps({"hook_id": "a", "event": "E", "exit_code": code}) for code in (0, 1))
    )

    summary = summarize_hook_census([first, second, {"status": "absent"}, {"status": "unreadable"}])

    assert summary["n_trials"] == 4
    assert summary["n_trials_with_census"] == 2
    assert summary["n_trials_unreadable"] == 1
    assert summary["hooks"] == [
        {"hook_id": "a", "event": "E", "runs": 3, "failures": 1, "blocked": 0, "not_started": 0, "trials": 2}
    ]
    assert summary["total_runs"] == 3


def test_census_caps_hooks_and_lines_and_marks_the_result_truncated() -> None:
    def line(index: int, code: int = 0) -> str:
        return json.dumps({"hook_id": f"h{index}", "event": "E", "exit_code": code})

    # One hook more than the cap, then a known hook again: the extra hook is dropped, the known one still counts.
    parsed = parse_hook_census("\n".join([*(line(index) for index in range(257)), line(0, 127)]))
    assert parsed["truncated"] is True
    assert len(parsed["hooks"]) == 256
    assert parsed["hooks"][0]["runs"] == 2
    assert (parsed["total_runs"], parsed["total_failures"], parsed["total_not_started"]) == (257, 1, 1)

    long_run = parse_hook_census("\n".join(line(0) for _ in range(20_001)))
    assert long_run["truncated"] is True
    assert long_run["total_runs"] == 20_000

    first = parse_hook_census("\n".join(line(index) for index in range(200)))
    second = parse_hook_census("\n".join(line(index) for index in range(150, 350)))
    summary = summarize_hook_census([first, second])
    assert summary["truncated"] is True
    assert len(summary["hooks"]) == 256
    assert summary["total_runs"] == 200 + (256 - 150)


def _write_job(jobs_dir: Path, variant: str, census: str | None) -> None:
    job_dir = jobs_dir / f"demo-claude-code-{variant}"
    trial = job_dir / "case-1__AbCd123"
    (trial / "verifier").mkdir(parents=True)
    (trial / "verifier" / "reward.json").write_text(
        json.dumps(
            dict.fromkeys(
                ("security", "skill_execution", "skill_efficiency", "accuracy", "goal_accuracy", "behavior_check"),
                0.8,
            )
        ),
        encoding="utf-8",
    )
    (trial / "agent").mkdir()
    (trial / "agent" / "trajectory.json").write_text(
        json.dumps({"steps": [{"step_id": 1, "source": "user", "message": "hi"}]}), encoding="utf-8"
    )
    if census is not None:
        (trial / "agent" / HOOK_CENSUS_FILENAME).write_text(census, encoding="utf-8")
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_trials": 1,
                    "n_errors": 0,
                    "evals": {"x": {"n_trials": 1, "n_errors": 0, "reward_stats": {"reward": {"0.8": [trial.name]}}}},
                },
            }
        ),
        encoding="utf-8",
    )


def test_collector_attaches_per_trial_census_and_per_arm_summary(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    _write_job(
        jobs, "with", json.dumps({"hook_id": "hooks/hooks.json#PreToolUse[0].hooks[0]", "event": "PreToolUse", "exit_code": 0})
    )
    _write_job(jobs, "without", None)

    results = collect_harbor_results(
        skill_name="demo",
        agents=["claude-code"],
        output_dir=tmp_path / "out",
        jobs_dir=jobs,
        expected_cases=1,
        expected_case_ids=["case-1"],
        expected_trials=1,
        plugin_signals=build_plugin_signals_context(member_skills=["alpha"], wrapper_skills=["demo"]),
    )

    reward = json.loads(
        (tmp_path / "out" / "claude-code" / "with-skill" / "trials" / "case-1__AbCd123" / "reward.json").read_text(
            encoding="utf-8"
        )
    )
    assert reward["plugin_signals"]["hook_census"]["hooks"][0]["hook_id"] == "hooks/hooks.json#PreToolUse[0].hooks[0]"
    summary = results["agents"]["claude-code"]["plugin_signals_summary"]["with_skill"]["hook_census"]
    assert summary["total_runs"] == 1
    assert summary["n_trials_with_census"] == 1
