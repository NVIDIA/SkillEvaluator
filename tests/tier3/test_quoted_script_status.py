# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A quoted script path must retain its recorded command exit status."""

import shlex
import shutil
import subprocess
import sys
from functools import cache

import pytest

from skillevaluator.tier3.eval_core.declarative_checks import evaluate_declarative_checks


@cache
def _bash_supports_stderr_pipeline():
    executable = shutil.which("bash")
    if sys.platform == "win32" or executable is None:
        return False
    completed = subprocess.run(
        [executable, "-c", 'printf %s "$BASH_VERSINFO"'],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return int(completed.stdout) >= 4


@pytest.mark.parametrize(
    "command",
    ["python 'scripts/my task.py'", 'python "scripts/my task.py"'],
)
@pytest.mark.parametrize("status", [0, 2])
def test_single_quoted_script_command_uses_its_recorded_status(command: str, status: int) -> None:
    (result,) = evaluate_declarative_checks(
        [{"type": "command_succeeded", "script": "scripts/my task.py"}],
        tool_calls=[
            {
                "action": "Bash",
                "action_input": {"command": command},
                "observation": f"Process exited with code {status}",
            }
        ],
        final_response="",
        workspace_roots=[],
    )

    assert result["passed"] is (status == 0)
    assert f"exited {status}" in result["reason"]


@pytest.mark.parametrize("tail", [" | tail -1", " |& tail -1", " || true", "; echo done"])
def test_quoted_script_still_does_not_claim_a_shell_tail_status(tail: str) -> None:
    (result,) = evaluate_declarative_checks(
        [{"type": "command_succeeded", "script": "scripts/my task.py"}],
        tool_calls=[
            {
                "action": "Bash",
                "action_input": {"command": "python 'scripts/my task.py'" + tail},
                "observation": "Process exited with code 0",
            }
        ],
        final_response="",
        workspace_roots=[],
    )

    assert result["passed"] is None


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="Requires POSIX bash to record actual negated shell statuses",
)
@pytest.mark.parametrize(
    ("script_operand", "relative_script"),
    [
        ("'scripts/my task.py'", "scripts/my task.py"),
        ('"scripts/my task.py"', "scripts/my task.py"),
        (r"scripts/my\ folder/task.py", "scripts/my folder/task.py"),
    ],
)
@pytest.mark.parametrize("script_status", [0, 2])
@pytest.mark.parametrize("negation", ["!", "time !", "time -p !"])
def test_negated_quoted_script_does_not_claim_the_shell_status(
    tmp_path, script_operand, relative_script, script_status, negation
):
    script = tmp_path / relative_script
    script.parent.mkdir(parents=True)
    script.write_text("import sys\nprint('script-ran')\nraise SystemExit(int(sys.argv[1]))\n", encoding="utf-8")
    command = f"{negation} {shlex.quote(sys.executable)} {script_operand} {script_status}"
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == int(script_status == 0), completed.stderr
    assert completed.stdout == "script-ran\n"
    if negation == "!":
        assert completed.stderr == ""
    else:
        assert "real" in completed.stderr
        assert "user" in completed.stderr
        assert "sys" in completed.stderr

    invoked, succeeded = evaluate_declarative_checks(
        [
            {"type": "script_invoked", "script": relative_script},
            {"type": "command_succeeded", "script": relative_script},
        ],
        tool_calls=[
            {
                "action": "Bash",
                "action_input": {"command": command},
                "observation": f"Process exited with code {completed.returncode}",
            }
        ],
        final_response="",
        workspace_roots=[],
    )

    assert invoked["passed"] is True
    assert succeeded["passed"] is None
    assert "could not be attributed" in succeeded["reason"]


@pytest.mark.skipif(
    not _bash_supports_stderr_pipeline(),
    reason="Requires Bash 4 or newer for stderr pipelines",
)
@pytest.mark.parametrize(
    ("script_operand", "relative_script"),
    [("scripts/run.py", "scripts/run.py"), ("'scripts/my task.py'", "scripts/my task.py")],
)
def test_stderr_pipeline_does_not_claim_the_last_process_status(tmp_path, script_operand, relative_script):
    script = tmp_path / relative_script
    script.parent.mkdir(parents=True)
    script.write_text("import sys\nprint('script-ran', file=sys.stderr)\nraise SystemExit(2)\n", encoding="utf-8")
    command = f"{shlex.quote(sys.executable)} {script_operand} |& tail -1"
    completed = subprocess.run(
        ["bash", "-c", command], cwd=tmp_path, capture_output=True, text=True, check=False, timeout=10
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "script-ran\n"
    assert completed.stderr == ""

    invoked, succeeded = evaluate_declarative_checks(
        [
            {"type": "script_invoked", "script": relative_script},
            {"type": "command_succeeded", "script": relative_script},
        ],
        tool_calls=[
            {
                "action": "Bash",
                "action_input": {"command": command},
                "observation": f"Process exited with code {completed.returncode}",
            }
        ],
        final_response="",
        workspace_roots=[],
    )

    assert invoked["passed"] is True
    assert succeeded["passed"] is None
    assert "could not be attributed" in succeeded["reason"]


@pytest.mark.parametrize("status", [0, 2])
def test_literal_exclamation_argument_does_not_hide_script_status(status):
    (result,) = evaluate_declarative_checks(
        [{"type": "command_succeeded", "script": "scripts/my task.py"}],
        tool_calls=[
            {
                "action": "Bash",
                "action_input": {"command": "python 'scripts/my task.py' '!'"},
                "observation": f"Process exited with code {status}",
            }
        ],
        final_response="",
        workspace_roots=[],
    )

    assert result["passed"] is (status == 0)


def _run_and_grade_comment_command(tmp_path, shell, operand, relative_script, status, suffix, prefix=""):
    executable = shutil.which(shell)
    if sys.platform == "win32" or executable is None:
        pytest.skip(f"Requires POSIX {shell} to record actual shell statuses")
    script = tmp_path / relative_script
    script.parent.mkdir(parents=True)
    script.write_text("import sys\nprint('script-ran')\nraise SystemExit(int(sys.argv[1]))\n", encoding="utf-8")
    command = f"{prefix}{shlex.quote(sys.executable)} {operand} {status}{suffix}"
    if shell == "bash" and "|&" in command and not _bash_supports_stderr_pipeline():
        pytest.skip("Requires Bash 4 or newer for stderr pipelines")
    completed = subprocess.run(
        [executable, "-c", command], cwd=tmp_path, capture_output=True, text=True, check=False, timeout=10
    )
    assert "script-ran\n" in completed.stdout, completed.stderr
    results = evaluate_declarative_checks(
        [
            {"type": "script_invoked", "script": relative_script},
            {"type": "command_succeeded", "script": relative_script},
        ],
        tool_calls=[
            {
                "action": "exec_command",
                "action_input": {"command": command, "shell": executable},
                "observation": f"Process exited with code {completed.returncode}",
            }
        ],
        final_response="",
        workspace_roots=[],
    )
    assert results[0]["passed"] is True
    return completed, results[1]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize(
    ("operand", "relative_script"),
    [("scripts/plain.py", "scripts/plain.py"), ("'scripts/my task.py'", "scripts/my task.py")],
)
@pytest.mark.parametrize("status", [0, 2])
def test_trailing_comment_does_not_hide_recorded_script_status(tmp_path, shell, operand, relative_script, status):
    completed, result = _run_and_grade_comment_command(
        tmp_path, shell, operand, relative_script, status, " # note |& ignored"
    )
    assert completed.returncode == status
    assert result["passed"] is (status == 0)
    assert f"exited {status}" in result["reason"]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("hash_word", ["'#'", '"#"', r"\#", "word#", r"word\ #", "''#", "word\u00a0#", "\v#"])
def test_literal_hash_does_not_hide_a_real_stderr_pipeline(tmp_path, shell, hash_word):
    completed, result = _run_and_grade_comment_command(
        tmp_path, shell, "scripts/plain.py", "scripts/plain.py", 2, f" {hash_word} |& cat"
    )
    assert completed.returncode == 0
    assert result["passed"] is None
    assert "could not be attributed" in result["reason"]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("comment_start", [" # note |& ignored", " \\\n# note |& ignored"])
def test_a_comment_does_not_hide_the_next_physical_line(tmp_path, shell, comment_start):
    completed, result = _run_and_grade_comment_command(
        tmp_path, shell, "scripts/plain.py", "scripts/plain.py", 2, f"{comment_start}\nprintf 'later\\n'"
    )
    assert completed.returncode == 0
    assert completed.stdout == "script-ran\nlater\n"
    assert result["passed"] is None


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("status", [0, 2])
@pytest.mark.parametrize("prefix", ["# note\n", "printf 'before\\n'\n# note\n"])
def test_comment_only_lines_keep_the_existing_command_boundaries(tmp_path, shell, status, prefix):
    completed, result = _run_and_grade_comment_command(
        tmp_path, shell, "scripts/plain.py", "scripts/plain.py", status, "", prefix
    )
    assert completed.returncode == status
    assert result["passed"] is (status == 0)
    assert f"exited {status}" in result["reason"]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize(
    ("operand", "relative_script"),
    [("scripts/plain.py", "scripts/plain.py"), ("'scripts/my task.py'", "scripts/my task.py")],
)
@pytest.mark.parametrize("argument", ['"$(printf " # hi")"', '"`printf " # hi"`"', '"${0:+" # hi"}"'])
@pytest.mark.parametrize(("status", "tail", "shell_status"), [(2, " |& cat", 0), (0, " |& cat; false", 1)])
def test_nested_expansion_quotes_cannot_hide_a_real_pipeline(
    tmp_path, shell, operand, relative_script, argument, status, tail, shell_status
):
    completed, result = _run_and_grade_comment_command(
        tmp_path, shell, operand, relative_script, status, f" {argument}{tail}"
    )
    assert completed.returncode == shell_status
    assert completed.returncode != status
    assert result["passed"] is None
    assert "could not be attributed" in result["reason"]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("status", [0, 2])
@pytest.mark.parametrize(
    "argument",
    [
        "'$(printf \" # hi\")'",
        "'`printf \" # hi\"`'",
        "'${0:+\" # hi\"}'",
        "$'$(printf \" # hi\")'",
        "$'`printf \" # hi\"`'",
        "$'${0:+\" # hi\"}'",
        r'"\$(printf \" # hi\")"',
        r'"\`printf \" # hi\"\`"',
        r'"\${0:+\" # hi\"}"',
    ],
)
def test_literal_substitution_text_keeps_the_script_status(tmp_path, shell, status, argument):
    completed, result = _run_and_grade_comment_command(
        tmp_path, shell, "scripts/plain.py", "scripts/plain.py", status, f" {argument} # note |& ignored"
    )
    assert completed.returncode == status
    assert result["passed"] is (status == 0)
    assert f"exited {status}" in result["reason"]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("status", [0, 2])
@pytest.mark.parametrize(
    "comment",
    [' # "$(printf " # hi")" |& ignored', ' # "`printf " # hi"`" |& ignored', ' # "${0:+" # hi"}" |& ignored'],
)
def test_substitution_text_inside_a_comment_keeps_the_script_status(tmp_path, shell, status, comment):
    completed, result = _run_and_grade_comment_command(
        tmp_path, shell, "scripts/plain.py", "scripts/plain.py", status, comment
    )
    assert completed.returncode == status
    assert result["passed"] is (status == 0)
    assert f"exited {status}" in result["reason"]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize(
    ("operand", "relative_script"),
    [("scripts/plain.py", "scripts/plain.py"), ("'scripts/my task.py'", "scripts/my task.py")],
)
@pytest.mark.parametrize("continuations", [1, 2])
@pytest.mark.parametrize("expansion", ['(printf " # hi")', '{0:+" # hi"}'])
@pytest.mark.parametrize(("status", "tail", "bash_status"), [(2, " |& cat", 0), (0, " |& cat; false", 1)])
def test_continued_expansion_openers_respect_the_recorded_shell(
    tmp_path, shell, operand, relative_script, continuations, expansion, status, tail, bash_status
):
    argument = '"$' + "\\\n" * continuations + expansion + '"'
    completed, result = _run_and_grade_comment_command(
        tmp_path, shell, operand, relative_script, status, f" {argument}{tail}"
    )
    if shell == "bash":
        assert completed.returncode == bash_status
        assert completed.returncode != status
        assert result["passed"] is None
        assert "could not be attributed" in result["reason"]
    else:
        # Zsh does not form an expansion across this continuation: the inner
        # quote ends the argument, making the hash a real trailing comment.
        assert completed.returncode == status
        assert result["passed"] is (status == 0)
        assert f"exited {status}" in result["reason"]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize(
    ("operand", "relative_script"),
    [("scripts/plain.py", "scripts/plain.py"), ("'scripts/my task.py'", "scripts/my task.py")],
)
@pytest.mark.parametrize("status", [0, 2])
@pytest.mark.parametrize(
    "argument",
    ['"${NAME}"', '"${_NAME2}"', '"$' + "\\\n" + '{NAME}"', '"${NA' + "\\\n" + 'ME}"'],
)
def test_plain_braced_identifiers_keep_the_script_status(
    tmp_path, monkeypatch, shell, operand, relative_script, status, argument
):
    # Expansion results are argument data, not another shell parsing pass.
    value = '# " |& false $(false)'
    monkeypatch.setenv("NAME", value)
    monkeypatch.setenv("_NAME2", value)
    completed, result = _run_and_grade_comment_command(
        tmp_path, shell, operand, relative_script, status, f" {argument} # note |& ignored"
    )
    assert completed.returncode == status
    assert result["passed"] is (status == 0)
    assert f"exited {status}" in result["reason"]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("status", [0, 2])
@pytest.mark.parametrize("quoting", ["single", "ansi-c", "escaped"])
def test_literal_continued_expansion_text_keeps_the_script_status(tmp_path, shell, status, quoting):
    literal = "$" + "\\\n" + '(printf " # hi")'
    if quoting == "escaped":
        argument = '"\\$' + "\\\n" + '(printf \\" # hi\\")"'
    else:
        argument = ("$" if quoting == "ansi-c" else "") + shlex.quote(literal)
    completed, result = _run_and_grade_comment_command(
        tmp_path, shell, "scripts/plain.py", "scripts/plain.py", status, f" {argument} # note |& ignored"
    )
    assert completed.returncode == status
    assert result["passed"] is (status == 0)
    assert f"exited {status}" in result["reason"]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize(
    ("operand", "relative_script"),
    [("scripts/plain.py", "scripts/plain.py"), ("'scripts/my task.py'", "scripts/my task.py")],
)
@pytest.mark.parametrize("status", [0, 2])
@pytest.mark.parametrize(
    "argument",
    [
        ' "$(pwd)/out.json"',
        ' "$(pwd)/out#data.json"',
        ' "$(pwd)" "#out.json"',
        " `printf x`",
        ' "`pwd`/out#data.json"',
        ' "${MODEL:-gpt}"',
        ' "${MODEL:-gpt}" "#out.json"',
    ],
)
def test_expansions_without_comments_keep_the_script_status(
    tmp_path, shell, operand, relative_script, status, argument
):
    completed, result = _run_and_grade_comment_command(tmp_path, shell, operand, relative_script, status, argument)
    assert completed.returncode == status
    assert result["passed"] is (status == 0)
    assert f"exited {status}" in result["reason"]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("argument", ['"$(pwd)/out#data.json"', '"$(pwd)" "#out.json"', '"`pwd`/out#data.json"'])
def test_quoted_hash_with_substitution_does_not_hide_a_real_pipeline(tmp_path, shell, argument):
    completed, result = _run_and_grade_comment_command(
        tmp_path, shell, "scripts/plain.py", "scripts/plain.py", 2, f" {argument} | cat"
    )
    assert completed.returncode == 0
    assert result["passed"] is None
    assert "could not be attributed" in result["reason"]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("status", [0, 2])
def test_commentless_cd_expansion_keeps_plain_script_status(tmp_path, shell, status):
    completed, result = _run_and_grade_comment_command(
        tmp_path, shell, "scripts/plain.py", "scripts/plain.py", status, "", 'cd "$(pwd)" && '
    )
    assert completed.returncode == status
    assert result["passed"] is (status == 0)
    assert f"exited {status}" in result["reason"]


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("status", [0, 2])
def test_split_braced_identifier_does_not_hide_a_real_pipeline(tmp_path, shell, status):
    argument = '"${NA' + "\\\n" + 'ME}"'
    completed, result = _run_and_grade_comment_command(
        tmp_path, shell, "scripts/plain.py", "scripts/plain.py", status, f" {argument} |& cat # comment"
    )
    assert completed.returncode == 0
    assert result["passed"] is None
    assert "could not be attributed" in result["reason"]
