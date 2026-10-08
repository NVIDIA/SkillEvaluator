# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LLM judges see what the agent wrote.

Write calls used to reach the judges cut to their first 200 characters, and an
OpenCode ``apply_patch`` (argument ``patchText``) was missing from the
file-change evidence entirely. A correct release note then lost its last line
("-- reviewed by acme") and was scored as wrong. These tests pin the fix on the
host helper and on the Harbor verifier copy, and check the two copies agree.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core import atif_helpers

_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)


def _load_template_module():
    spec = importlib.util.spec_from_file_location("harbor_template_eval_write_evidence", _TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_template = _load_template_module()

COPIES = pytest.mark.parametrize("copy", [atif_helpers, eval_template], ids=["host", "template"])
_MARKER_RE = re.compile(r"\.\.\.\[(\d+) chars truncated\]\.\.\.")

_RELEASE_NOTE_PATCH = (
    "*** Begin Patch\n"
    "*** Add File: /workspace/RELEASE_NOTE.md\n"
    "+# Falcon 2.1 release note\n"
    "+\n"
    "+Codename: FALCON-7E43DF\n"
    "+\n"
    "+## Changes\n"
    "+- Faster sync\n"
    "+- Fix for the login crash\n"
    "+\n"
    "+-- reviewed by acme\n"
    "*** End Patch"
)


def _fixture_secret(*parts: str) -> str:
    """Build a key-shaped value from pieces so static scanners do not flag the file."""
    return "".join(parts)


def _fake_key() -> str:
    return "nvapi-" + _fixture_secret("Ab12Cd34Ef56Gh78", "Ij90Kl12Mn34Op56")


def _call(call_id: str, name: str, arguments: object) -> dict:
    return {"tool_call_id": call_id, "function_name": name, "arguments": arguments}


def _step(*calls: dict, message: str = "", results: tuple[str, ...] = ()) -> dict:
    step: dict = {"source": "agent", "message": message, "tool_calls": list(calls)}
    if results:
        step["observation"] = {
            "results": [
                {"source_call_id": call["tool_call_id"], "content": content}
                for call, content in zip(calls, results, strict=True)
            ]
        }
    return step


def _opencode_release_note_trajectory() -> dict:
    """The OpenCode trial that wrote the note right and was judged wrong."""
    return {
        "steps": [
            {"source": "user", "message": "Write the release note file for Falcon 2.1."},
            _step(
                _call("c1", "acme_acme_codename", {"project": "Falcon"}),
                message="Looking up the codename first.",
                results=("Official codename for Falcon: FALCON-7E43DF",),
            ),
            _step(
                _call("c2", "apply_patch", {"patchText": _RELEASE_NOTE_PATCH}),
                message="Writing the note now.",
                results=("Success. Updated the following files:\nA workspace/RELEASE_NOTE.md",),
            ),
            {"source": "agent", "message": "Wrote `RELEASE_NOTE.md` with the codename `FALCON-7E43DF`."},
        ]
    }


def _six_kb_write_trajectory() -> dict:
    lines = [f"line {index:04d} " + ("x" * 50) for index in range(100)]
    content = "HEAD_LINE\n" + "\n".join(lines) + "\nLAST_LINE_OF_THE_FILE"
    assert len(content) > 6000
    return {
        "steps": [
            _step(_call("w1", "Write", {"file_path": "/workspace/big.md", "content": content}), results=("ok",)),
            {"source": "agent", "message": "Done."},
        ]
    }


def _history_lines(summary: str, prefix: str) -> list[str]:
    return [line for line in summary.split("\n") if line.startswith(prefix)]


@COPIES
def test_short_apply_patch_keeps_its_last_line_in_the_history(copy):
    summary = copy.build_conversation_summary(_opencode_release_note_trajectory(), "Write the note.")

    assert "+-- reviewed by acme" in summary
    assert "*** End Patch" in summary
    assert "truncated]" not in summary


@COPIES
def test_opencode_apply_patch_reaches_every_judge(copy):
    bundles = copy.build_metric_evidence_bundles(
        _opencode_release_note_trajectory(),
        "Write the note.",
        ground_truth="RELEASE_NOTE.md ends with -- reviewed by acme",
        expected_behavior=["The note ends with `-- reviewed by acme`"],
    )

    for metric in ("accuracy", "goal_accuracy", "behavior_check"):
        evidence = bundles[metric]["prompt_evidence"]
        assert "+-- reviewed by acme" in evidence, metric
        assert "Path: /workspace/RELEASE_NOTE.md" in evidence, metric


@COPIES
def test_six_kb_write_keeps_head_and_tail_with_a_sized_marker(copy):
    summary = copy.build_conversation_summary(_six_kb_write_trajectory(), "Write a big file.")

    assert "HEAD_LINE" in summary
    assert "LAST_LINE_OF_THE_FILE" in summary
    marker = _MARKER_RE.search(summary)
    assert marker is not None
    assert int(marker.group(1)) > 3000
    write_entry = summary[summary.index("Agent called: Write") : summary.index("Tool returned: ok")]
    assert len(write_entry) < 2200


@COPIES
def test_long_non_write_argument_is_marked_not_silently_cut(copy):
    query = "START_OF_QUERY " + ("q" * 600) + " END_OF_QUERY"
    traj = {
        "steps": [
            _step(_call("g1", "Grep", {"pattern": query}), results=("no matches",)),
            {"source": "agent", "message": "Nothing found."},
        ]
    }

    summary = copy.build_conversation_summary(traj, "Search.")
    call_entry = summary[summary.index("Agent called: Grep") : summary.index("Tool returned:")]

    assert "START_OF_QUERY" in call_entry
    assert "END_OF_QUERY" in call_entry
    assert _MARKER_RE.search(call_entry)
    assert len(call_entry) < 300


@COPIES
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        (
            "functions.apply_patch",
            {"input": "*** Begin Patch\n*** Add File: a.md\n" + "+x\n" * 120 + "+TAIL\n*** End Patch"},
        ),
        ("mcp__filesystem__write_file", {"path": "/workspace/a.md", "content": "y\n" * 300 + "TAIL"}),
        ("tools.Write", {"file_path": "/workspace/a.md", "content": "z\n" * 300 + "TAIL"}),
        ("write", {"filePath": "/workspace/a.md", "content": "w\n" * 300 + "TAIL"}),
        ("edit", {"filePath": "/workspace/a.md", "oldString": "old", "newString": "v\n" * 300 + "TAIL"}),
        (
            "patch",
            {"mode": "replace", "path": "/workspace/a.md", "old_string": "old", "new_string": "p\n" * 300 + "TAIL"},
        ),
        (
            "patch",
            {
                "mode": "patch",
                "patch": "*** Begin Patch\n*** Update File: a.md\n@@\n" + "+u\n" * 200 + "+TAIL\n*** End Patch",
            },
        ),
    ],
    ids=[
        "codex-apply-patch",
        "mcp-write-file",
        "dotted-write",
        "opencode-write",
        "opencode-edit",
        "hermes-patch-replace",
        "hermes-patch-v4a",
    ],
)
def test_namespaced_and_harness_write_tools_get_the_write_budget(copy, name, arguments):
    traj = {"steps": [_step(_call("w1", name, arguments), results=("ok",)), {"source": "agent", "message": "Done."}]}

    summary = copy.build_conversation_summary(traj, "Write a.md.")
    evidence = copy.build_behavior_evidence(traj, "Write a.md.")

    assert "TAIL" in summary
    assert "TAIL" in evidence.split("FINAL RESPONSE", 1)[0]
    assert "a.md" in evidence.split("FINAL RESPONSE", 1)[0]


_ROWS = "\n".join(f"row {index}" for index in range(80))


@COPIES
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("Bash", {"command": f"cat <<'EOF' > /workspace/output/report.txt\nFIRST_ROW\n{_ROWS}\nFINAL_ROW\nEOF"}),
        ("bash", {"command": f"tee /workspace/output/report.txt <<'EOF'\nFIRST_ROW\n{_ROWS}\nFINAL_ROW\nEOF"}),
        (
            "exec_command",
            {
                "cmd": "python3 - <<'PY'\nFIRST_ROW = 1\n"
                f"open('/workspace/output/report.txt', 'w').write({_ROWS!r})\nFINAL_ROW = 2\nPY"
            },
        ),
        (
            "exec_command",
            {
                "cmd": "cd /workspace && apply_patch <<'EOF'\n*** Begin Patch\n"
                f"*** Add File: output/report.txt\n+FIRST_ROW\n+{_ROWS}\n+FINAL_ROW\n*** End Patch\nEOF",
                "workdir": "/workspace",
            },
        ),
        ("terminal", {"command": f"cat > /workspace/output/report.txt <<'EOF'\nFIRST_ROW\n{_ROWS}\nFINAL_ROW\nEOF"}),
        (
            "execute_code",
            {"code": f"FIRST_ROW = 1\nopen('/workspace/output/report.txt', 'w').write({_ROWS!r})\nFINAL_ROW = 2\n"},
        ),
    ],
    ids=["heredoc-redirect", "tee", "python-write", "shell-apply-patch", "hermes-terminal", "hermes-execute-code"],
)
def test_shell_writes_get_the_write_budget(copy, name, arguments):
    assert len(next(iter(arguments.values()))) > 600
    traj = {
        "steps": [
            _step(_call("b1", name, arguments), results=("",)),
            {"source": "agent", "message": "Saved the report."},
        ]
    }

    summary = copy.build_conversation_summary(traj, "Save a report.")
    file_changes = copy.build_behavior_evidence(traj, "Save a report.").split("FINAL RESPONSE", 1)[0]

    assert "FIRST_ROW" in summary
    assert "FINAL_ROW" in summary
    assert "report.txt" in summary
    assert "FINAL_ROW" in file_changes


def _single_call_trajectory(name: str, arguments: dict) -> dict:
    return {"steps": [_step(_call("c1", name, arguments), results=("",)), {"source": "agent", "message": "Done."}]}


@COPIES
@pytest.mark.parametrize(
    ("name", "arguments", "paths"),
    [
        ("Bash", {"command": "echo hi>Out.txt"}, "Out.txt"),
        ("Bash", {"command": "cat notes.txt &> Log.txt"}, "Log.txt"),
        ("Bash", {"command": "echo x >| Force.txt"}, "Force.txt"),
        ("Bash", {"command": "sed -i 's/a/b/' Conf.ini"}, "Conf.ini"),
        ("Bash", {"command": "sed -i -e 's/a/b/' -e 's/c/d/' A.cfg B.cfg"}, "A.cfg, B.cfg"),
        ("Bash", {"command": "make 2>&1 | tee -a Build.log Copy.log"}, "Build.log, Copy.log"),
        ("Bash", {"command": 'echo hi > "My Notes.txt"'}, "My Notes.txt"),
        ("functions.exec_command", {"cmd": "echo hi > Out.txt"}, "Out.txt"),
        ("mcp__shell__bash", {"command": "echo hi > Out.txt"}, "Out.txt"),
        ("Bash", {"command": "make 2>&1 | tee >(grep -i error >&2) Build.log"}, "Build.log"),
        ("Bash", {"command": "echo x | tee -a >(cat) Out.txt"}, "Out.txt"),
        ("Bash", {"command": "echo state > /dev/shm/Session.json"}, "/dev/shm/Session.json"),
    ],
    ids=[
        "glued-redirect",
        "stdout-and-stderr",
        "noclobber-override",
        "sed-in-place",
        "sed-in-place-scripts",
        "every-tee-operand",
        "quoted-path",
        "codex-namespaced-exec",
        "mcp-shell-tool",
        "tee-after-process-substitution",
        "tee-option-and-process-substitution",
        "shared-memory-file",
    ],
)
def test_shell_writes_are_read_like_the_security_extractor(copy, name, arguments, paths):
    traj = _single_call_trajectory(name, arguments)

    file_changes = copy.build_behavior_evidence(traj, "q").split("FINAL RESPONSE", 1)[0]
    refs = copy.build_metric_evidence_refs(traj, "q", expected_behavior=["x"])["behavior_check"]

    # Paths keep their case.
    assert f"Path: {paths}\n" in file_changes
    [ref] = [ref for ref in refs if ref["kind"] == "file_change"]
    assert ref["excerpt"] == next(iter(arguments.values()))


@COPIES
@pytest.mark.parametrize(
    "command",
    [
        "pytest -q > /dev/null 2>&1",
        "make 2>&1 | tee",
        "echo the committee met",
        "sed 's/a/b/' notes.txt",
        "awk 'NR>1{print $2}' data.csv",
        "grep -o '<title>[^<]*' page.html",
        "echo '<b>total</b>'",
        "python3 -c 'def f(x) -> int: return x'",
        "node -e '[1].map(x => x)'",
        "curl -s https://example.com/p1 | grep -o '<title>[^<]*' | sed 's/<title>//'",
        'echo "a > b" \\> c # > d',
        "make 2>&1 | tee >(grep -i error >&2) > /dev/stderr",
    ],
    ids=[
        "discarded-output",
        "tee-to-stdout",
        "tee-inside-a-word",
        "sed-without-in-place",
        "awk-comparison",
        "html-tag-pattern",
        "html-tag-text",
        "python-return-annotation",
        "javascript-arrow",
        "pipeline-of-patterns",
        "quoted-escaped-and-commented",
        "process-substitution-only",
    ],
)
def test_commands_that_write_no_file_are_not_file_changes(copy, command):
    traj = _single_call_trajectory("Bash", {"command": command})

    evidence = copy.build_behavior_evidence(traj, "q")
    refs = copy.build_metric_evidence_refs(traj, "q", expected_behavior=["x"])["behavior_check"]

    assert "FILE CHANGES" not in evidence
    assert all(ref["kind"] != "file_change" for ref in refs)


@COPIES
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("mcp__db__execute", {"code": "SELECT * FROM t WHERE a>b"}),
        ("execute_code", {"code": "rows = load()\nif len(rows) > 3:\n    print(rows[0] >> 1)"}),
    ],
    ids=["sql-comparison", "python-comparison"],
)
def test_program_text_is_not_read_as_shell_redirections(copy, name, arguments):
    traj = _single_call_trajectory(name, arguments)

    evidence = copy.build_behavior_evidence(traj, "q")
    refs = copy.build_metric_evidence_refs(traj, "q", expected_behavior=["x"])["behavior_check"]

    assert "FILE CHANGES" not in evidence
    assert all(ref["kind"] != "file_change" for ref in refs)


@COPIES
def test_program_text_that_writes_a_file_is_still_a_file_change(copy):
    traj = _single_call_trajectory("execute_code", {"code": "open('report.txt', 'w').write('done')"})

    evidence = copy.build_behavior_evidence(traj, "q")

    assert "FILE CHANGES" in evidence


@COPIES
def test_read_only_pipelines_do_not_push_a_write_out_of_the_file_change_refs(copy):
    """Fifteen read-only pipelines used to fill all twelve file_change refs before the real write."""
    steps = [
        _step(
            _call(f"c{index}", "Bash", {"command": f"curl -s https://example.com/p{index} | grep -o '<title>[^<]*'"}),
            results=("<title>x",),
        )
        for index in range(15)
    ]
    steps.append(
        _step(_call("w1", "Write", {"file_path": "/workspace/output/titles.md", "content": "x"}), results=("ok",))
    )
    traj = {"steps": [*steps, {"source": "agent", "message": "Done."}]}

    file_changes = copy.build_behavior_evidence(traj, "q").split("FINAL RESPONSE", 1)[0]
    refs = copy.build_metric_evidence_refs(traj, "q", expected_behavior=["x"])["behavior_check"]

    assert file_changes.count("Agent called: ") == 1
    assert [ref["json_pointer"] for ref in refs if ref["kind"] == "file_change"] == ["/steps/15/tool_calls/0"]


@COPIES
def test_secret_in_a_write_body_is_redacted_for_the_judge(copy):
    key = _fake_key()
    traj = {
        "steps": [
            _step(
                _call("w1", "Write", {"file_path": "/workspace/.env", "content": f"API_KEY={key}\nMODE=prod\n"}),
                results=(f"wrote API_KEY={key}",),
            ),
            {"source": "agent", "message": "Saved the config."},
        ]
    }

    summary = copy.build_conversation_summary(traj, "Save the config.")
    bundles = copy.build_metric_evidence_bundles(
        traj, "Save the config.", ground_truth="config saved", expected_behavior=["saves the config"]
    )

    assert key not in summary
    assert "nvapi-<redacted>" in summary
    assert "MODE=prod" in summary
    for metric, bundle in bundles.items():
        assert key not in bundle["prompt_evidence"], metric


def _very_long_trajectory() -> dict:
    steps: list[dict] = [{"source": "user", "message": "Fix the build."}]
    steps.append(
        _step(
            _call("w0", "Write", {"file_path": "/workspace/out.md", "content": "OLD_DRAFT " + ("o" * 900)}),
            results=("ok",),
        )
    )
    for index in range(80):
        steps.append(
            _step(
                _call(f"r{index}", "Bash", {"command": f"ls /workspace/dir{index} " + ("a" * 300)}),
                message=f"Exploring {index}.",
                results=("listing " + ("b" * 800),),
            )
        )
    steps.append(
        _step(
            _call(
                "w1",
                "Write",
                {"file_path": "/workspace/out.md", "content": "NEW_DRAFT " + ("n" * 900) + "\nLATEST_LAST_LINE"},
            ),
            results=("ok",),
        )
    )
    for index in range(20):
        steps.append(
            _step(_call(f"t{index}", "Bash", {"command": f"pytest -k case{index}"}), results=("passed " * 60,))
        )
    steps.append(
        _step(
            _call("final-check", "Bash", {"command": "git status"}),
            message="FINAL_ANSWER_START The build is fixed and out.md is written. FINAL_ANSWER_END",
            results=("clean",),
        )
    )
    return {"steps": steps}


@COPIES
def test_very_long_history_keeps_the_final_answer_and_the_latest_write(copy):
    traj = _very_long_trajectory()

    full = copy.build_conversation_summary(traj, "Fix the build.")
    fitted = copy.build_conversation_summary(traj, "Fix the build.", max_chars=3500)

    assert len(full) > 30_000
    assert len(fitted) <= 3500
    assert "FINAL_ANSWER_START" in fitted and "FINAL_ANSWER_END" in fitted
    assert "LATEST_LAST_LINE" in fitted
    assert "history entries omitted" in fitted
    assert fitted.rstrip().endswith("FINAL_ANSWER_END")


@COPIES
def test_shrunk_entry_marker_counts_every_cut_character(copy):
    output = "OUT_START " + ("r" * 5000) + " OUT_END"
    traj = {
        "steps": [
            _step(_call("b1", "Bash", {"command": "make"}), results=(output,)),
            {"source": "agent", "message": "Built."},
        ]
    }

    fitted = copy.build_conversation_summary(traj, "Build.", max_chars=300)
    result = next(line for line in fitted.split("\nAgent: ")[0].split("\n") if line.startswith("Tool returned:"))
    counts = [int(count) for count in _MARKER_RE.findall(fitted)]

    assert result.startswith("Tool returned: OUT_START")
    assert "OUT_END" in fitted
    assert len(fitted) <= 300
    assert counts and max(counts) > len(output) - 200


@COPIES
def test_behavior_evidence_keeps_the_final_answer_and_the_latest_write(copy):
    evidence = copy.build_behavior_evidence(_very_long_trajectory(), "Fix the build.", max_chars=8000)
    file_changes, history = evidence.split("COMPACT TOOL HISTORY", 1)

    assert len(evidence) <= 8000
    assert "LATEST_LAST_LINE" in file_changes.split("FINAL RESPONSE", 1)[0]
    assert "Agent final answer: FINAL_ANSWER_START" in history
    assert "history entries omitted" in history
    # FILE CHANGES has the bodies, so the history spends its room on other calls.
    assert "NEW_DRAFT " + "n" * 300 not in history
    assert "pytest -k case19" in history


@COPIES
def test_many_writes_do_not_push_file_changes_out_of_accuracy(copy):
    steps: list[dict] = []
    for index in range(6):
        content = f"FILE_{index}_HEAD " + ("c" * 2500) + f" FILE_{index}_TAIL"
        steps.append(
            _step(
                _call(f"w{index}", "Write", {"file_path": f"/workspace/f{index}.md", "content": content}),
                results=("ok",),
            )
        )
    steps.append({"source": "agent", "message": "Wrote six files."})

    bundle = copy.build_metric_evidence_bundles({"steps": steps}, "q", ground_truth="six files")["accuracy"]

    assert "PRODUCED FILES / WRITES" in bundle["prompt_evidence"]
    assert "FILE_5_TAIL" in bundle["prompt_evidence"]
    assert bundle["omitted"]["truncated"] is True


@COPIES
def test_behavior_bundle_with_facts_is_not_cut_again_by_the_judge(copy):
    traj = _very_long_trajectory()
    bundle = copy.build_metric_evidence_bundles(
        traj,
        "Fix the build.",
        ground_truth="out.md written",
        expected_behavior=[f"Runs `pytest -k case{index}`" for index in range(12)],
    )["behavior_check"]

    assert bundle["prompt_evidence"].startswith("VERIFIED FACTS")
    assert len(bundle["prompt_evidence"]) <= 8000
    assert eval_template._compact_behavior_conversation(bundle["prompt_evidence"]) == bundle["prompt_evidence"]


def _skill_run_with_big_writes(writes: int = 3, size: int = 2500, middle_calls: int = 0) -> dict:
    steps: list[dict] = [
        {"source": "user", "message": "Add the feature and its tests."},
        _step(_call("s", "Skill", {"skill": "release-notes"}), results=("Launching skill: release-notes",)),
        _step(
            _call("r", "Read", {"file_path": "/workspace/.claude/skills/release-notes/SKILL.md"}),
            results=("# Release notes skill\n" + "doc " * 300,),
        ),
    ]
    for index in range(writes):
        content = f"# module {index}\n" + "x = 1\n" * (size // 6) + f"MODULE_{index}_TAIL"
        steps.append(
            _step(
                _call(f"w{index}", "Write", {"file_path": f"/workspace/src/mod{index}.py", "content": content}),
                results=("File created",),
            )
        )
    for index in range(middle_calls):
        steps.append(_step(_call(f"e{index}", "Bash", {"command": f"echo step{index}"}), results=("o" * 300,)))
    steps.append(_step(_call("t", "Bash", {"command": "pytest -q tests/test_mods.py"}), results=("TESTS: 42 passed",)))
    steps.append(_step(_call("l", "Bash", {"command": "ruff check src"}), results=("All checks passed!",)))
    steps.append({"source": "agent", "message": "Added the modules; tests pass."})
    return {"steps": steps}


@COPIES
@pytest.mark.parametrize("middle_calls", [0, 8], ids=["plain", "with-middle-calls"])
def test_behavior_history_keeps_skill_call_and_test_run_next_to_big_writes(copy, middle_calls):
    bundle = copy.build_metric_evidence_bundles(
        _skill_run_with_big_writes(middle_calls=middle_calls),
        "Add the feature and its tests.",
        ground_truth="modules added",
        expected_behavior=["The agent uses the release-notes skill", "The agent runs the tests"],
    )["behavior_check"]
    file_changes, history = bundle["prompt_evidence"].split("COMPACT TOOL HISTORY", 1)

    assert len(bundle["prompt_evidence"]) <= 8000
    assert 'Agent called: Skill({"skill": "release-notes"})' in history
    assert "pytest -q tests/test_mods.py" in history
    assert "TESTS: 42 passed" in history
    for index in range(3):
        assert f"MODULE_{index}_TAIL" in file_changes
        assert f"/workspace/src/mod{index}.py" in history
    assert "[written content is under FILE CHANGES]" in history


@COPIES
def test_history_alone_still_shows_write_bodies(copy):
    summary = copy.build_conversation_summary(_skill_run_with_big_writes(), "Add the feature.")

    assert "MODULE_2_TAIL" in summary
    assert "FILE CHANGES" not in summary


def _single_write_trajectory(content: str, path: str = "/workspace/REPORT.md") -> dict:
    return {
        "steps": [
            _step(_call("w", "Write", {"file_path": path, "content": content}), results=("File created",)),
            {"source": "agent", "message": "Wrote the report."},
        ]
    }


@COPIES
def test_four_kb_write_reaches_every_judge_whole(copy):
    lines = [f"Section {index}: some text about the release, nothing special here." for index in range(70)]
    lines[35] = "Rollback plan: run `falconctl rollback --to 2.0.3` within 24 hours."
    content = "\n".join(lines)
    assert len(content) > 4000

    bundles = copy.build_metric_evidence_bundles(
        _single_write_trajectory(content), "q", ground_truth="has the rollback command", expected_behavior=["x"]
    )

    for metric in ("accuracy", "goal_accuracy", "behavior_check"):
        assert "falconctl rollback --to 2.0.3" in bundles[metric]["prompt_evidence"], metric
        assert bundles[metric]["omitted"]["truncated"] is False, metric
    behavior_file_changes = bundles["behavior_check"]["prompt_evidence"].split("FINAL RESPONSE", 1)[0]
    for written in (
        bundles["accuracy"]["prompt_evidence"],
        bundles["goal_accuracy"]["prompt_evidence"],
        behavior_file_changes,
    ):
        assert content in written


@COPIES
def test_a_cut_write_body_marks_the_bundle_truncated(copy):
    content = "BODY_START\n" + "line of the report\n" * 1200 + "BODY_END"
    assert len(content) > 20_000

    bundles = copy.build_metric_evidence_bundles(
        _single_write_trajectory(content), "q", ground_truth="report written", expected_behavior=["x"]
    )

    for metric in ("accuracy", "goal_accuracy", "behavior_check"):
        evidence = bundles[metric]["prompt_evidence"]
        assert "BODY_START" in evidence and "BODY_END" in evidence, metric
        assert _MARKER_RE.search(evidence), metric
        assert bundles[metric]["omitted"]["truncated"] is True, metric
    # The bigger budget shows more of the same write.
    assert len(bundles["goal_accuracy"]["prompt_evidence"]) > len(bundles["accuracy"]["prompt_evidence"]) + 3000


@COPIES
@pytest.mark.parametrize("writes", [6, 8])
def test_big_writes_do_not_push_the_newest_test_result_out(copy, writes):
    bundles = copy.build_metric_evidence_bundles(
        _skill_run_with_big_writes(writes=writes), "q", ground_truth="tests pass", expected_behavior=["x"]
    )

    for metric in ("accuracy", "goal_accuracy"):
        evidence = bundles[metric]["prompt_evidence"]
        assert "TESTS: 42 passed" in evidence, metric
        assert f"MODULE_{writes - 1}_TAIL" in evidence, metric
        assert len(evidence) <= copy._bundle_budgets()[metric], metric


@COPIES
def test_big_write_keeps_small_recent_results_in_view(copy):
    steps: list[dict] = [
        _step(
            _call("w", "Write", {"file_path": "/workspace/big.md", "content": "BIG_START\n" + "z" * 30_000}),
            results=("File created",),
        )
    ]
    for index in range(5):
        steps.append(_step(_call(f"c{index}", "Bash", {"command": f"check {index}"}), results=(f"CHECK_{index}_OK",)))
    steps.append({"source": "agent", "message": "Done."})

    evidence = copy.build_metric_evidence_bundles({"steps": steps}, "q", ground_truth="done")["accuracy"][
        "prompt_evidence"
    ]

    assert all(f"CHECK_{index}_OK" in evidence for index in range(5))
    assert "BIG_START" in evidence
    assert len(evidence) <= copy._bundle_budgets()["accuracy"]


@COPIES
def test_argv_list_shell_write_reads_as_a_command(copy):
    script = f"cat > /workspace/output/report.txt <<'EOF'\nFIRST_ROW\n{_ROWS}\nFINAL_ROW\nEOF"
    traj = {
        "steps": [
            _step(_call("b1", "shell", {"command": ["bash", "-lc", script]}), results=("",)),
            {"source": "agent", "message": "Saved the report."},
        ]
    }

    file_changes = copy.build_behavior_evidence(traj, "Save a report.").split("FINAL RESPONSE", 1)[0]

    assert "Path: /workspace/output/report.txt" in file_changes
    assert "bash -lc cat > /workspace/output/report.txt <<'EOF'\nFIRST_ROW\n" in file_changes
    assert "\nFINAL_ROW\n" in file_changes
    assert "['bash'" not in file_changes


def _marker_text(digits: str) -> str:
    return "start\n" + "x" * 3000 + f"\n...[{digits} chars truncated]...\n" + "y" * 3000 + "\nend"


@COPIES
def test_marker_like_tool_output_cannot_crash_or_inflate_the_cut_count(copy):
    for digits in ("9" * 5000, "999999999"):
        output = _marker_text(digits)
        traj = {
            "steps": [
                _step(_call("c", "Bash", {"command": "cat log"}), results=(output,)),
                {"source": "agent", "message": "Done."},
            ]
        }

        bundles = copy.build_metric_evidence_bundles(traj, "q", ground_truth="x", expected_behavior=["y"])
        summary = copy.build_conversation_summary(traj, "q", max_chars=300)

        for text in [summary, *(bundle["prompt_evidence"] for bundle in bundles.values())]:
            counts = [int(count) for count in _MARKER_RE.findall(text) if len(count) < 20]
            assert counts and max(counts) < len(output), digits[:12]


@COPIES
def test_placeholder_keys_reach_the_judge_as_written(copy, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    key = _fake_key()
    config = f"OPENAI_API_KEY=sk-your-key-here\nNVIDIA_API_KEY=nvapi-REPLACE_ME_PLEASE\nREAL={key}\n"

    evidence = copy.build_metric_evidence_bundles(
        _single_write_trajectory(config, path="/workspace/.env.example"),
        "q",
        ground_truth="`.env.example` has OPENAI_API_KEY=sk-your-key-here",
        expected_behavior=["x"],
    )["accuracy"]["prompt_evidence"]

    assert "OPENAI_API_KEY=sk-your-key-here" in evidence
    assert "NVIDIA_API_KEY=nvapi-REPLACE_ME_PLEASE" in evidence
    assert key not in evidence
    assert "REAL=nvapi-<redacted>" in evidence


@COPIES
@pytest.mark.parametrize(
    "placeholder",
    [
        "SLACK_BOT_TOKEN=xoxb-your-bot-token",
        "SLACK_APP_TOKEN=xoxp-REPLACE-WITH-YOUR-TOKEN",
        "GITLAB_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx",
        "GITHUB_TOKEN=ghp_your_token_here",
        "GH_PAT=github_pat_YOUR_TOKEN",
        "HF_TOKEN=hf_your_hugging_face_token_goes_here",
    ],
)
def test_placeholder_tokens_of_every_service_reach_the_judge_as_written(copy, monkeypatch, placeholder):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    real = "glpat-" + _fixture_secret("aB3dE6gH9j", "K2mN5pQ8sT")

    redacted = copy._redact_evidence_text(f"{placeholder}\nREAL={real}")

    assert redacted == f"{placeholder}\nREAL=glpat-<redacted>"


@COPIES
def test_runtime_key_is_redacted_even_when_it_looks_like_a_placeholder(copy, monkeypatch):
    runtime_key = _fixture_secret("nvapi-", "local-test-key")
    monkeypatch.setenv("NVIDIA_API_KEY", runtime_key)

    redacted = copy._redact_evidence_text(f"key={runtime_key} example=sk-your-key-here")

    assert runtime_key not in redacted
    assert "sk-your-key-here" not in redacted


@COPIES
def test_configured_credentials_never_reach_the_judges(copy, monkeypatch):
    # A credential value need not look like a key: its exact value is redacted wherever it shows.
    credential = _fixture_secret("opaque", "-anthropic-", "credential")
    monkeypatch.setenv("ANTHROPIC_API_KEY", credential)
    traj = {
        "steps": [
            _step(
                _call("w1", "Write", {"file_path": "/workspace/.env", "content": f"KEY={credential}\n"}),
                _call("b1", "Bash", {"command": f"echo {credential}"}),
                results=("ok", f"{credential}\n"),
            ),
            {"source": "agent", "message": f"The key is {credential}."},
        ]
    }

    bundles = copy.build_metric_evidence_bundles(traj, f"Use {credential}.", ground_truth="x", expected_behavior=["y"])

    for metric, bundle in bundles.items():
        assert credential not in json.dumps(bundle), metric
        assert "<redacted>" in bundle["prompt_evidence"], metric


def _parity_corpus() -> list[dict]:
    final_on_tool_step = {
        "steps": [
            _step(_call("a", "Read", {"file_path": "/workspace/in.txt"}), results=("hello",)),
            _step(
                _call("b", "Bash", {"command": "echo done > /workspace/out.txt"}),
                message="All done: wrote out.txt.",
                results=("",),
            ),
        ]
    }
    reasoning_and_edits = {
        "steps": [
            {
                "source": "agent",
                "reasoning_content": "Think " * 100,
                "tool_calls": [
                    _call(
                        "m", "MultiEdit", {"file_path": "/workspace/a.py", "edits": [{"new_string": "x = 1\n" * 200}]}
                    ),
                    _call(
                        "n", "NotebookEdit", {"notebook_path": "/workspace/n.ipynb", "new_source": "print(1)\n" * 50}
                    ),
                ],
                "observation": {"results": [{"source_call_id": "m", "content": "edited"}]},
            },
            {"source": "agent", "message": "Edited."},
        ]
    }
    note = "# Note\n" + "".join(f"line {index} filler\n" for index in range(200)) + "-- reviewed by acme\n"
    hermes = {
        "steps": [
            _step(
                _call(
                    "p",
                    "patch",
                    {"mode": "replace", "path": "/workspace/NOTE.md", "old_string": "x", "new_string": note},
                ),
                _call("t", "terminal", {"command": f"cat > /workspace/OTHER.md <<'EOF'\n{note}EOF"}),
                _call("a", "shell", {"command": ["bash", "-lc", f"cat > /workspace/THIRD.md <<'EOF'\n{note}EOF"]}),
                results=("Success", "", ""),
            ),
            {"source": "agent", "message": "Wrote the notes."},
        ]
    }
    return [
        _opencode_release_note_trajectory(),
        _six_kb_write_trajectory(),
        _very_long_trajectory(),
        final_on_tool_step,
        reasoning_and_edits,
        {"steps": []},
        hermes,
        _skill_run_with_big_writes(middle_calls=8),
        _skill_run_with_big_writes(writes=8),
        _single_write_trajectory("BODY_START\n" + "report line\n" * 2000 + "BODY_END"),
    ]


@pytest.mark.parametrize(
    "traj",
    _parity_corpus(),
    ids=[
        "opencode",
        "six-kb",
        "long",
        "final-on-tool",
        "edits",
        "empty",
        "hermes-and-argv",
        "skill-run",
        "many-writes",
        "big-write",
    ],
)
def test_host_and_template_build_identical_judge_evidence(traj):
    question = "Do the task."
    args = {"ground_truth": "Done with `out.md`", "expected_behavior": ["writes `/workspace/out.md`"]}

    for max_chars in (None, 2500):
        assert atif_helpers.build_conversation_summary(
            traj, question, max_chars=max_chars
        ) == eval_template.build_conversation_summary(traj, question, max_chars=max_chars)
    assert atif_helpers.build_behavior_evidence(traj, question) == eval_template.build_behavior_evidence(traj, question)
    host = atif_helpers.build_metric_evidence_bundles(traj, question, **args)
    template = eval_template.build_metric_evidence_bundles(traj, question, **args)
    for metric in ("accuracy", "goal_accuracy", "behavior_check"):
        assert host[metric]["prompt_evidence"] == template[metric]["prompt_evidence"], metric
        assert host[metric]["omitted"] == template[metric]["omitted"], metric


def test_final_answer_on_a_tool_step_is_listed_by_both_copies():
    traj = _parity_corpus()[3]

    for copy in (atif_helpers, eval_template):
        summary = copy.build_conversation_summary(traj, "q")
        assert summary.endswith("Agent final answer: All done: wrote out.txt.")


# Write evidence under configured judge budgets (SKILL_EVAL_*_BUDGET and
# SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT): the knobs size what the judges see
# of each write, and write recognition, cut markers, priority fitting,
# redaction, and host/template parity hold at any size.

_BUDGET_VARS = (
    "SKILL_EVAL_ACCURACY_BUDGET",
    "SKILL_EVAL_GOAL_ACCURACY_BUDGET",
    "SKILL_EVAL_BEHAVIOR_CHECK_BUDGET",
    "SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT",
)
_SMALL_BUDGETS = {
    "SKILL_EVAL_ACCURACY_BUDGET": "3000",
    "SKILL_EVAL_GOAL_ACCURACY_BUDGET": "4000",
    "SKILL_EVAL_BEHAVIOR_CHECK_BUDGET": "4000",
}
_LARGE_BUDGETS = {
    "SKILL_EVAL_ACCURACY_BUDGET": "30000",
    "SKILL_EVAL_GOAL_ACCURACY_BUDGET": "40000",
    "SKILL_EVAL_BEHAVIOR_CHECK_BUDGET": "30000",
}
_LONG_FINAL = {"SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT": "6000"}


def _set_budgets(monkeypatch, budgets: dict[str, str]) -> None:
    for name in _BUDGET_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in budgets.items():
        monkeypatch.setenv(name, value)


def _judge_compactor(copy):
    """The behavior judge's prompt compactor for this copy."""
    if copy is atif_helpers:
        from skillevaluator.tier3.eval_core import llm_judge

        return llm_judge._compact_behavior_conversation
    return copy._compact_behavior_conversation


@COPIES
def test_small_configured_budgets_keep_the_latest_write_and_the_test_run(copy, monkeypatch):
    _set_budgets(monkeypatch, _SMALL_BUDGETS)
    budgets = copy._bundle_budgets()
    # The behavior budget keeps 4,000 chars of tool-history room past the 800-char final response.
    assert budgets == {"accuracy": 3000, "goal_accuracy": 4000, "behavior_check": 4800}

    bundles = copy.build_metric_evidence_bundles(
        _skill_run_with_big_writes(writes=6),
        "Add the feature and its tests.",
        ground_truth="tests pass",
        expected_behavior=["The agent uses the release-notes skill", "The agent runs the tests"],
    )

    for metric in ("accuracy", "goal_accuracy"):
        evidence = bundles[metric]["prompt_evidence"]
        assert len(evidence) <= budgets[metric], metric
        assert "TESTS: 42 passed" in evidence, metric
        assert "MODULE_5_TAIL" in evidence, metric
        assert _MARKER_RE.search(evidence), metric
        assert bundles[metric]["omitted"]["truncated"] is True, metric
    behavior = bundles["behavior_check"]["prompt_evidence"]
    file_changes, history = behavior.split("COMPACT TOOL HISTORY", 1)
    assert len(behavior) <= budgets["behavior_check"]
    assert "MODULE_5_TAIL" in file_changes
    assert 'Agent called: Skill({"skill": "release-notes"})' in history
    assert "pytest -q tests/test_mods.py" in history
    assert _judge_compactor(copy)(behavior) == behavior


def _skill_run_with_a_long_answer() -> dict:
    traj = _skill_run_with_big_writes(writes=3, size=2500)
    traj["steps"][-1]["message"] = "ANSWER_START\n" + "Detail line about the change.\n" * 100 + "ANSWER_END"
    return traj


@COPIES
@pytest.mark.parametrize("budget", ["4800", "6000"])
def test_small_behavior_budget_with_a_long_answer_keeps_the_skill_call_and_the_test_run(copy, monkeypatch, budget):
    # The history kept its own 1,500-char copy of the answer at top rank, so a small
    # budget dropped every tool call first and the history held only the answer.
    _set_budgets(monkeypatch, {"SKILL_EVAL_BEHAVIOR_CHECK_BUDGET": budget})

    evidence = copy.build_metric_evidence_bundles(
        _skill_run_with_a_long_answer(),
        "Add the feature and its tests.",
        ground_truth="modules added",
        expected_behavior=["The agent uses the release-notes skill", "The agent runs the tests"],
    )["behavior_check"]["prompt_evidence"]
    head, history = evidence.split("COMPACT TOOL HISTORY", 1)

    assert len(evidence) <= int(budget)
    assert "FINAL RESPONSE\nANSWER_START" in head
    assert 'Agent called: Skill({"skill": "release-notes"})' in history
    assert "pytest -q tests/test_mods.py" in history
    assert "TESTS: 42 passed" in history
    assert "Agent: ANSWER_START" in history
    assert _judge_compactor(copy)(evidence) == evidence


@COPIES
def test_history_keeps_the_answer_first_when_final_response_cannot_show_it(copy, monkeypatch):
    # A final response limit too small for the section title leaves out FINAL
    # RESPONSE, so the history's copy is the only one and keeps its top rank.
    _set_budgets(
        monkeypatch, {"SKILL_EVAL_BEHAVIOR_CHECK_BUDGET": "4000", "SKILL_EVAL_BEHAVIOR_FINAL_RESPONSE_LIMIT": "10"}
    )

    evidence = copy.build_metric_evidence_bundles(
        _skill_run_with_a_long_answer(),
        "Add the feature and its tests.",
        ground_truth="modules added",
        expected_behavior=["The agent runs the tests"],
    )["behavior_check"]["prompt_evidence"]
    history = evidence.split("COMPACT TOOL HISTORY", 1)[1]

    assert "FINAL RESPONSE" not in evidence
    assert "Agent: ANSWER_START" in history
    assert history.rstrip().endswith("ANSWER_END")
    # More than the 160-char short line a shown answer gets.
    assert history.count("Detail line about the change.") > 20


@COPIES
def test_history_shrinks_an_entry_only_as_far_as_the_budget_needs(copy):
    # A history 50 chars over budget cut a 1,000-char tool result down to a
    # 160-char short line and left the rest of the room empty.
    entries = [
        copy._Entry("User: Fix the build.", copy._RANK_MESSAGE),
        copy._Entry("Tool returned: " + "a" * 1000, copy._RANK_LOW),
        copy._Entry("Agent: Fixed it.", copy._RANK_KEEP),
    ]
    full = copy._fit_history(entries, None)

    fitted = copy._fit_history(entries, len(full) - 50)

    # The cut marker reserves room for any count, so a few chars stay unused.
    assert len(full) - 100 <= len(fitted) <= len(full) - 50
    assert fitted.count("a") > 900
    assert _MARKER_RE.search(fitted)
    assert fitted.startswith("User: Fix the build.") and fitted.endswith("Agent: Fixed it.")


@COPIES
def test_default_budget_with_a_long_answer_uses_the_whole_behavior_budget(copy, monkeypatch):
    # Once the history's copy of a shown answer could shrink, it shrank to a short
    # line even when a little less room was all the budget needed, so a 3 KB
    # answer left about 700 chars of the 8,000-char budget unused.
    _set_budgets(monkeypatch, {})

    evidence = copy.build_metric_evidence_bundles(
        _skill_run_with_a_long_answer(),
        "Add the feature and its tests.",
        ground_truth="modules added",
        expected_behavior=["The agent uses the release-notes skill", "The agent runs the tests"],
    )["behavior_check"]["prompt_evidence"]
    history = evidence.split("COMPACT TOOL HISTORY", 1)[1]

    assert 7900 <= len(evidence) <= 8000
    assert 'Agent called: Skill({"skill": "release-notes"})' in history
    assert "TESTS: 42 passed" in history
    assert history.rstrip().endswith("ANSWER_END")
    # More than the 160-char short line.
    assert history.count("Detail line about the change.") > 20


@COPIES
def test_large_configured_budgets_show_a_long_write_whole(copy, monkeypatch):
    key = _fake_key()
    lines = [f"Section {index}: some text about the release." for index in range(500)]
    lines[250] = f"API_KEY={key}"
    content = "\n".join(lines)
    # Longer than the 12,000 chars a write body gets with the default budgets.
    assert 15_000 < len(content) < 30_000
    traj = _single_write_trajectory(content)
    args = {"ground_truth": "report written", "expected_behavior": ["x"]}

    _set_budgets(monkeypatch, {})
    default = copy.build_metric_evidence_bundles(traj, "q", **args)["goal_accuracy"]
    assert _MARKER_RE.search(default["prompt_evidence"])
    assert default["omitted"]["truncated"] is True

    _set_budgets(monkeypatch, _LARGE_BUDGETS)
    bundles = copy.build_metric_evidence_bundles(traj, "q", **args)
    redacted = content.replace(key, "nvapi-<redacted>")
    for metric in ("accuracy", "goal_accuracy", "behavior_check"):
        evidence = bundles[metric]["prompt_evidence"]
        written = evidence.split("COMPACT TOOL HISTORY", 1)[0]
        assert key not in evidence, metric
        assert redacted in written, metric
        assert not _MARKER_RE.search(written), metric
        assert bundles[metric]["omitted"]["truncated"] is False, metric


@COPIES
def test_large_final_response_limit_keeps_writes_and_the_tool_history(copy, monkeypatch):
    _set_budgets(monkeypatch, _LONG_FINAL)
    budget = copy._bundle_budgets()["behavior_check"]
    assert budget == 10000
    traj = _skill_run_with_big_writes()
    traj["steps"][-1]["message"] = "ANSWER_START\n" + "summary line\n" * 600 + "ANSWER_END"

    evidence = copy.build_metric_evidence_bundles(
        traj,
        "Add the feature and its tests.",
        ground_truth="modules added",
        expected_behavior=["The agent uses the release-notes skill", "The agent runs the tests"],
    )["behavior_check"]["prompt_evidence"]
    file_changes, history = evidence.split("COMPACT TOOL HISTORY", 1)
    final = file_changes.split("FINAL RESPONSE\n", 1)[1].split("\n\nUSER REQUEST\n", 1)[0]

    assert len(evidence) <= budget
    # The configured limit, not the default 800 chars, sizes the answer; the cut keeps both ends.
    assert 4000 < len(final) <= 6000
    assert final.startswith("ANSWER_START") and final.endswith("ANSWER_END")
    assert _MARKER_RE.search(final)
    assert "MODULE_2_TAIL" in file_changes.split("FINAL RESPONSE", 1)[0]
    assert 'Agent called: Skill({"skill": "release-notes"})' in history
    assert "pytest -q tests/test_mods.py" in history
    assert "TESTS: 42 passed" in history
    # FINAL RESPONSE shows the answer, so the history lists it as a short line.
    assert history.count("summary line") < 20
    assert _judge_compactor(copy)(evidence) == evidence


@pytest.mark.parametrize("budgets", [_SMALL_BUDGETS, _LARGE_BUDGETS, _LONG_FINAL], ids=["small", "large", "long-final"])
def test_host_and_template_agree_under_configured_budgets(monkeypatch, budgets):
    _set_budgets(monkeypatch, budgets)
    question = "Do the task."
    args = {"ground_truth": "Done with `out.md`", "expected_behavior": ["writes `/workspace/out.md`"]}
    assert atif_helpers._bundle_budgets() == eval_template._bundle_budgets()

    for traj in _parity_corpus():
        assert atif_helpers.build_behavior_evidence(traj, question) == eval_template.build_behavior_evidence(
            traj, question
        )
        host = atif_helpers.build_metric_evidence_bundles(traj, question, **args)
        template = eval_template.build_metric_evidence_bundles(traj, question, **args)
        for metric in ("accuracy", "goal_accuracy", "behavior_check"):
            assert host[metric]["prompt_evidence"] == template[metric]["prompt_evidence"], metric
            assert host[metric]["omitted"] == template[metric]["omitted"], metric
        behavior = host["behavior_check"]["prompt_evidence"]
        assert len(behavior) <= atif_helpers._bundle_budgets()["behavior_check"]
