# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script execution credit requires evidence of an invocation.

``check_script_execution`` previously asked only whether the expected script
name appeared anywhere in an execution command, so reading, printing or
searching the script earned the same full credit as running it. These cases
cover both the host checker and the bundled Harbor verifier template.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from skillevaluator.tier3.eval_core import checks as shared_checks


def _load_template():
    template_path = Path(__file__).parents[2] / "src/skillevaluator/tier3/harbor/templates/eval.py"
    spec = importlib.util.spec_from_file_location("harbor_eval_script_invocation", template_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEMPLATE = _load_template()
IMPLEMENTATIONS = [
    pytest.param(shared_checks.check_script_execution, id="host"),
    pytest.param(TEMPLATE.check_script_execution, id="harbor-verifier"),
]
EXPECTED_SCRIPT = "run.py"
SCRIPT_SOURCE = "#!/usr/bin/env python3\n# run.py writes report.txt\nprint('done')\n"


def _bash(command: str, observation: str = "Exit code 0") -> list[dict[str, object]]:
    return [{"action": "Bash", "action_input": {"command": command}, "observation": observation}]


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "cat /workspace/skills/demo/run.py",
        "head -20 run.py",
        "tail -5 ./run.py",
        "sed -n '1,10p' run.py",
        "less run.py",
        "wc -l run.py",
        "ls -la run.py",
        "cp run.py /tmp/backup.py",
    ],
)
def test_reading_a_script_is_not_executing_it(check, command) -> None:
    result = check(_bash(command, SCRIPT_SOURCE), EXPECTED_SCRIPT)
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert "Executed" not in result["reason"]


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "echo run.py",
        "printf 'run.py\\n'",
        "grep -n threshold run.py",
    ],
)
def test_naming_a_script_is_not_executing_it(check, command) -> None:
    result = check(_bash(command, "run.py"), EXPECTED_SCRIPT)
    assert result["score"] == 0.0
    assert result["passed"] is False


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize("command", ["python run.py.bak", "python rerun.py", "python run.pyc"])
def test_a_similar_filename_is_a_different_script(check, command) -> None:
    result = check(_bash(command), EXPECTED_SCRIPT)
    assert result["score"] == 0.0
    assert result["passed"] is False


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
def test_reading_a_script_inside_a_nested_shell_is_not_executing_it(check) -> None:
    result = check(_bash("bash -c 'cat run.py'", SCRIPT_SOURCE), EXPECTED_SCRIPT)
    assert result["score"] == 0.0
    assert result["passed"] is False


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
def test_script_source_in_the_observation_does_not_rescue_a_read(check) -> None:
    """The output of ``cat run.py`` is the script's own text, not evidence."""
    result = check(_bash("cat run.py", "print('run.py finished')"), EXPECTED_SCRIPT)
    assert result["score"] == 0.0
    assert result["passed"] is False


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
def test_file_read_tool_observation_does_not_earn_execution_credit(check) -> None:
    calls = [
        {
            "action": "Read",
            "action_input": {"file_path": "/workspace/skills/demo/run.py"},
            "observation": SCRIPT_SOURCE,
        }
    ]
    result = check(calls, EXPECTED_SCRIPT)
    assert result["score"] == 0.0
    assert result["passed"] is False


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "python /workspace/skills/demo/run.py",
        "python3 run.py",
        "python3.12 run.py --verbose",
        "python -u run.py",
        "./run.py",
        "/workspace/skills/demo/run.py --out report.txt",
        "bash -c 'python run.py'",
        "cd /workspace/skills/demo && python run.py",
        "timeout 30 python run.py",
        "uv run python run.py",
        "env PYTHONPATH=/x python run.py",
        "source run.py",
        "cat notes.txt && python run.py",
        "python run.py && cat report.txt",
        "/venv/bin/python skills/demo/run.py && cat totals.csv",
    ],
)
def test_genuine_invocations_keep_full_credit(check, command) -> None:
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["reason"] == f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
def test_unrecognised_command_shape_is_weaker_evidence_not_full_credit(check) -> None:
    result = check(_bash("xargs -I {} {} run.py < runners.txt"), EXPECTED_SCRIPT)
    assert result["score"] == 0.75
    assert result["reason"] != f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
def test_no_expected_script_is_unchanged(check) -> None:
    result = check(_bash("cat run.py"), None)
    assert result["score"] == 1.0
    assert result["passed"] is True


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
def test_unrelated_execution_still_reports_the_expected_script(check) -> None:
    result = check(_bash("ls -la"), EXPECTED_SCRIPT)
    assert result["score"] == 0.0
    assert EXPECTED_SCRIPT in result["reason"]


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
def test_skill_tool_observation_fallback_is_preserved(check) -> None:
    """A non-read tool reporting the script ran keeps the 0.75 fallback."""
    calls = [
        {
            "action": "Skill",
            "action_input": {"name": "demo"},
            "observation": "Ran run.py and wrote report.txt",
        }
    ]
    result = check(calls, EXPECTED_SCRIPT)
    assert result["score"] == 0.75
    assert result["passed"] is True


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
def test_reading_then_running_is_credited(check) -> None:
    calls = _bash("cat run.py", SCRIPT_SOURCE) + _bash("python run.py", "report written")
    result = check(calls, EXPECTED_SCRIPT)
    assert result["score"] == 1.0
    assert result["reason"] == f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "python other.py run.py",
        "python3 wrapper.py --input run.py",
        "bash other.sh run.py",
    ],
)
def test_a_script_named_after_the_interpreter_target_is_only_argv(check, command) -> None:
    """Only the first non-option argument is the script the interpreter runs."""
    result = check(_bash(command, "other only"), EXPECTED_SCRIPT)
    assert result["score"] == 0.0
    assert result["passed"] is False


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    ["python -W ignore run.py", "python -B -u run.py", "python -- run.py", "python run.py extra.py"],
)
def test_interpreter_options_do_not_hide_the_script(check, command) -> None:
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0
    assert result["reason"] == f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "xargs -I{} python3 {} <<< run.py",
        "python3 $(echo run.py)",
        "find skills -name run.py -exec python3 {} \\;",
        "python3 `echo run.py`",
    ],
)
def test_run_time_substitution_is_undecidable_not_a_failure(check, command) -> None:
    """A path the shell fills in at run time cannot be compared statically."""
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 0.75
    assert result["reason"] != f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "python -Wignore run.py",
        "python -Xdev run.py",
        "python -W ignore::DeprecationWarning run.py",
        "python -X dev run.py",
        "python -OO run.py",
        "perl -I lib run.py",
        "ruby -I lib run.py",
        "node -r ./setup.js run.py",
    ],
)
def test_interpreter_options_with_values_do_not_hide_the_script(check, command) -> None:
    """An option value, attached or separate, is not the script argument."""
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0
    assert result["reason"] == f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    ["rm run.py", "touch run.py", "mv run.py backup.py", "git status run.py", "tar cf out.tar run.py"],
)
def test_commands_that_are_not_invocations_score_zero(check, command) -> None:
    """Only a recognised way of running a script earns credit, so no list of
    non-executing commands has to be maintained."""
    result = check(_bash(command), EXPECTED_SCRIPT)
    assert result["score"] == 0.0
    assert result["passed"] is False


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
def test_short_circuited_chain_is_reported_as_an_invocation(check) -> None:
    """Known limitation: the walk is static and does not model exit status.

    ``false && python run.py`` never reaches the interpreter at run time, but
    the command as written is an invocation, so it is credited. Deciding this
    would need the chain's runtime outcome, which the tool-call text does not
    carry.
    """
    result = check(_bash("false && python run.py", "Exit code 1"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        'echo "python run.py"',
        "cat <<EOF\npython run.py\nEOF",
        'git commit -m "run python run.py first"',
        'grep -r "python run.py" notes.txt',
    ],
)
def test_an_invocation_quoted_as_data_is_not_an_invocation(check, command) -> None:
    """Command text that only describes running the script is not evidence.

    A heredoc body in particular is the operand's data, not further commands,
    even though its newlines look like command separators.
    """
    result = check(_bash(command, "written"), EXPECTED_SCRIPT)
    assert result["score"] != 1.0
    assert result["reason"] != f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize("command", ["{ python run.py; }", "( python run.py )", "! python run.py"])
def test_grouping_tokens_do_not_hide_the_invocation(check, command) -> None:
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0
    assert result["reason"] == f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
def test_a_path_piped_to_xargs_is_undecidable(check) -> None:
    """xargs builds its arguments from standard input, which the text lacks."""
    result = check(_bash("echo run.py | xargs python", "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 0.75


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "script"),
    [
        ("python --help run.py", "run.py"),
        ("python --version run.py", "run.py"),
        ("python -h run.py", "run.py"),
        ("python -V run.py", "run.py"),
        ("perl -c run.pl", "run.pl"),
        ("perl -v run.pl", "run.pl"),
        ("ruby -c run.rb", "run.rb"),
        ("ruby --version run.rb", "run.rb"),
        ("node -c run.js", "run.js"),
        ("node --check run.js", "run.js"),
        ("node -v run.js", "run.js"),
        ("bash -n run.sh", "run.sh"),
        ("sh -n run.sh", "run.sh"),
    ],
)
def test_options_that_print_or_only_check_never_run_the_script(check, command, script) -> None:
    """Help, version and syntax-check modes exit before running the script."""
    result = check(_bash(command, "usage: python [option] ..."), script)
    assert result["score"] == 0.0
    assert result["passed"] is False


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "script"),
    [("python -v run.py", "run.py"), ("ruby -v run.rb", "run.rb"), ("bash -x run.sh", "run.sh")],
)
def test_verbose_options_still_run_the_script(check, command, script) -> None:
    """``ruby -v`` prints its version and runs the script; ``ruby --version`` does not."""
    result = check(_bash(command, "report written"), script)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
def test_unrecognised_long_option_is_undecidable(check) -> None:
    """An option this table does not describe may carry the script away."""
    result = check(_bash("python --unknown-option run.py", "written"), EXPECTED_SCRIPT)
    assert result["score"] == 0.75


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "script"),
    [
        ("python --require lib run.py", "run.py"),
        ("python --loader loader run.py", "run.py"),
        ("bash --require lib run.py", "run.py"),
        ("python -Q old run.py", "run.py"),
        ("perl -M strict run.pl", "run.pl"),
        ("perl -Mstrict run.pl", "run.pl"),
        ("bash -Mstrict run.sh", "run.sh"),
        ("bash --check run.sh", "run.sh"),
    ],
)
def test_options_outside_the_interpreter_grammar_are_undecidable(check, command, script) -> None:
    """An option belonging to another interpreter, or one whose argument is
    conditionally attached, cannot be resolved, so the command is not credited.
    """
    result = check(_bash(command, "written"), script)
    assert result["score"] == 0.75
    assert result["reason"] != f"Executed {script}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
def test_a_quoted_heredoc_operator_is_not_a_heredoc(check) -> None:
    result = check(_bash("echo '<<' && python run.py", "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
def test_a_here_string_consumes_only_its_operand(check) -> None:
    """Commands after a here-string are still commands."""
    result = check(_bash("cat <<< data; python run.py", "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize("command", ["sh -c 'python run.py'", "bash -xc 'python run.py'"])
def test_inline_code_payload_is_walked(check, command) -> None:
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(("command", "script"), [("perl -i run.pl", "run.pl")])
def test_measured_behaviour_beats_intuition(check, command, script) -> None:
    """This runs the script, which is why the grammar is derived by execution:
    perl's in-place flag takes no separate argument, so the file after it is
    still the script rather than the flag's value.
    """
    result = check(_bash(command, "report written"), script)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "env --help python run.py",
        "timeout --help python run.py",
        "sudo --help python run.py",
        "stdbuf --help python run.py",
        "setsid --help python run.py",
        "xargs --help python run.py",
        "uv --help run python run.py",
        "nice --version python run.py",
    ],
)
def test_a_wrapper_asked_for_help_never_runs_the_wrapped_command(check, command) -> None:
    """A wrapper given --help prints and exits, so nothing after it runs."""
    result = check(_bash(command, "Usage: ...\nExit code 0"), EXPECTED_SCRIPT)
    assert result["score"] == 0.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "env -i python run.py",
        "timeout 30 python run.py",
        "timeout -s KILL 30 python run.py",
        "nice -n 5 python run.py",
        "nice -5 python run.py",
        "stdbuf -o0 python run.py",
        "setsid -f python run.py",
        "sudo -u build python run.py",
        "timeout --help python run.py; python run.py",
    ],
)
def test_a_wrapper_given_its_own_options_still_runs_the_command(check, command) -> None:
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "env --chunk-size=2 python run.py",
        "timeout --kill-on-idle python run.py",
        "stdbuf --unknown python run.py",
    ],
)
def test_a_wrapper_option_outside_its_grammar_is_undecidable(check, command) -> None:
    """An unlisted option may consume the command, so neither answer holds."""
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 0.75
    assert result["reason"] != f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "cat <<EOF\nnotes\nEOF\npython run.py",
        "cat <<'EOF'\nnotes\nEOF\npython run.py",
        'cat <<"EOF"\nnotes\nEOF\npython run.py',
        "cat <<-EOF\n\tnotes\n\tEOF\npython run.py",
        "cat <<EOF > notes.txt\nnotes\nEOF\npython run.py",
        "cat <<A\none\nA\ncat <<B\ntwo\nB\npython run.py",
    ],
)
def test_commands_after_a_heredoc_terminator_are_still_commands(check, command) -> None:
    """A heredoc body ends at its delimiter line, not at the end of the text."""
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    ["cat <<EOF\npython run.py\nEOF", "cat <<EOF\npython run.py\nEOF\ncat notes.txt"],
)
def test_an_invocation_inside_a_heredoc_body_is_still_data(check, command) -> None:
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 0.75
    assert result["reason"] != f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "echo $((1 << 2)); python run.py",
        "echo $(( 1 << 2 )) && python run.py",
        "shift=$((1 << 2)); python run.py",
    ],
)
def test_an_arithmetic_shift_is_not_a_heredoc(check, command) -> None:
    """``<<`` inside an arithmetic expansion shifts, so no body follows it."""
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "FOO=1; python run.py",
        "FOO=1 && python run.py",
        "A=1 B=2 && python run.py",
        "(cd skills && python run.py); echo done",
        "SCRIPT=run.py; python $SCRIPT",
    ],
)
def test_a_separator_grouped_with_punctuation_still_ends_a_command(check, command) -> None:
    """The tokenizer returns ``));`` whole, which must not swallow what follows."""
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize("command", ["FOO=1", "A=1 B=2", "(cat run.py); echo done"])
def test_a_command_that_only_assigns_or_reads_runs_nothing(check, command) -> None:
    result = check(_bash(command, "Exit code 0"), EXPECTED_SCRIPT)
    assert result["score"] == 0.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "printf '' | xargs -r python run.py",
        "printf '' | xargs -p python run.py",
        "printf '' | xargs python run.py",
        "echo x | xargs -r python run.py",
        "xargs -a /dev/null python run.py",
        "printf '' | parallel -r python run.py",
    ],
)
def test_a_stdin_driven_tool_is_never_a_full_credit_invocation(check, command) -> None:
    """Whether xargs runs anything depends on input the command text lacks.

    ``printf '' | xargs -r python run.py`` runs nothing at all, and ``xargs -p``
    runs nothing with no terminal to confirm at, so neither answer is supported
    by the text and full credit would be a claim the command cannot back.
    """
    result = check(_bash(command, "Exit code 0"), EXPECTED_SCRIPT)
    assert result["score"] == 0.75
    assert result["reason"] != f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "env -- python run.py",
        "env -i -- python run.py",
        "env -- FOO=1 python run.py",
        "timeout -- 30 python run.py",
        "timeout -s TERM -- 30 python run.py",
        "nice -- python run.py",
        "nice -n 3 -- python run.py",
        "stdbuf -o0 -- python run.py",
        "setsid -- python run.py",
        "uv run -- python run.py",
    ],
)
def test_end_of_options_does_not_hide_the_invocation(check, command) -> None:
    """``--`` ends a wrapper's options, not its positional arguments.

    ``timeout -- 30 python run.py`` still reads 30 as the duration, so the
    command after it is what runs.
    """
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        'cat <<< "a; b"; python run.py',
        "cat <<< 'a | b'; python run.py",
        'cat <<< "a && b" && python run.py',
    ],
)
def test_a_separator_inside_a_here_string_operand_does_not_end_it(check, command) -> None:
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    ["uv run --no-project -- python run.py", "uv run -q --no-project python run.py"],
)
def test_an_unmodelled_runner_option_is_unresolved_not_a_failure(check, command) -> None:
    """These do run the script. The walk does not model uv's own options, so it
    reports that it could not resolve the command rather than scoring it zero.
    """
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 0.75
    assert result["reason"] != f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "python -c \"print('run.py')\"",
        "python -c 'print(123)' run.py",
        "python -c 'import runpy; runpy.run_path(\"run.py\")'",
        "python -m run.py",
        "python -m runpy run.py",
        "eval 'python run.py'",
        "python <run.py",
    ],
)
def test_inline_code_naming_the_script_is_unresolved(check, command) -> None:
    """Inline code, a module, eval and standard input are text this walk does
    not read, and they can run the script or merely mention it. Two of these
    run it and two only print or read it, and the command text does not say
    which, so none of them is scored as either.
    """
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 0.75
    assert result["reason"] != f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    ["python -c 'print(123)'", "python -m json.tool", "eval 'echo hello'", "python <other.py"],
)
def test_inline_code_that_never_names_the_script_is_still_zero(check, command) -> None:
    result = check(_bash(command, "Exit code 0"), EXPECTED_SCRIPT)
    assert result["score"] == 0.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "flock /tmp/build.lock python run.py",
        "taskset -c 0 python run.py",
        "ionice -c3 python run.py",
        "chrt -o 0 python run.py",
        "strace -f -o trace.log python run.py",
        "watch -n1 -t python run.py",
    ],
)
def test_an_invocation_inside_an_unmodelled_command_is_unresolved(check, command) -> None:
    """The set of commands that run another command is open, so it is not listed.

    A command with no grammar of its own that carries an interpreter running
    the script is reported as unresolved rather than as running nothing, which
    is what keeps a wrapper nobody anticipated from becoming a wrong answer.
    """
    result = check(_bash(command, "report written"), EXPECTED_SCRIPT)
    assert result["score"] == 0.75
    assert result["reason"] != f"Executed {EXPECTED_SCRIPT}"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    ["flock /tmp/build.lock cat run.py", "strace -f -o trace.log cat run.py", "nsenter -t 1 -m cat run.py"],
)
def test_an_unmodelled_command_that_only_reads_the_script_is_still_zero(check, command) -> None:
    """Carrying the script as an argument is not carrying an invocation of it."""
    result = check(_bash(command, SCRIPT_SOURCE), EXPECTED_SCRIPT)
    assert result["score"] == 0.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "xxd run.py",
        "hexdump -C run.py",
        "jq . run.py",
        "base64 run.py",
        "strings run.py",
        "od -c run.py",
        "shasum run.py",
        "iconv -f utf8 run.py",
        "column -t run.py",
        "rev run.py",
    ],
)
def test_reading_a_script_needs_no_list_of_reading_commands(check, command) -> None:
    """None of these verbs is named anywhere in this module, and all score zero.

    That is the point of recognising invocations rather than listing the
    commands that are not one: a new way to read a file costs nothing, while a
    list of reading verbs would have to grow forever to keep them out of full
    credit.
    """
    result = check(_bash(command, SCRIPT_SOURCE), EXPECTED_SCRIPT)
    assert result["score"] == 0.0


# The cases below came from review of the first revision, each a command the
# shell runs differently from how the walk read it.


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "cd",
        "cd && true",
        "cd -",
        "timeout --frobnicate 5 python other.py",
        "python -Q other.py",
        "env -Z python other.py",
        "for f in other.py; do python $f; done",
        "case $x in a) python other.py;; esac",
    ],
)
def test_partial_credit_requires_the_script_to_be_named(check, command) -> None:
    """A walk that cannot resolve a command which never names the script has
    learned nothing about that script. A bare ``cd`` loses track of the
    directory and an unknown option may have consumed anything, but neither is
    evidence about run.py, so they score as they did before invocation
    evidence was required: zero, not partial credit.
    """
    result = check(_bash(command, ""), EXPECTED_SCRIPT)
    assert result["score"] == 0.0
    assert result["passed"] is False


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    ["cd && cat run.py && python $f", "timeout --frobnicate 5 python run.py", "cd && python -Q run.py"],
)
def test_naming_the_script_keeps_partial_credit_for_an_unresolved_command(check, command) -> None:
    result = check(_bash(command, ""), EXPECTED_SCRIPT)
    assert result["score"] == 0.75


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "bash run.sh -c 'echo done'",
        "bash -- run.sh -c 'echo done'",
        "sh run.sh --help",
        "bash -x run.sh -c x",
        "zsh run.sh -n",
    ],
)
def test_options_after_the_script_operand_belong_to_the_script(check, command) -> None:
    """``bash run.sh -c 'echo done'`` runs run.sh and hands it ``-c``; only the
    options before the first operand are the shell's own.
    """
    result = check(_bash(command, "done"), "run.sh")
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("bash -c 'echo done' run.sh", 0.0),
        ("bash -c 'bash run.sh' run.sh", 1.0),
        ("bash -xc 'bash run.sh'", 1.0),
    ],
)
def test_inline_code_still_decides_when_the_option_precedes_the_operand(check, command, expected) -> None:
    """With ``-c`` before any operand the next word is code and run.sh is only
    its ``$0``; the payload alone decides whether the script ran.
    """
    result = check(_bash(command, "done"), "run.sh")
    assert result["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "python3 < /dev/null run.py",
        "python3 </dev/null run.py",
        "python3 0< /dev/null run.py",
        "python3 <&0 run.py",
        "python3 2>/dev/null run.py",
        "python3 2>&1 run.py",
        "python3 1>>log.txt run.py",
        "python3 < /dev/null -u run.py",
        "python3 -u < /dev/null run.py",
        "python3 run.py < /dev/null",
    ],
)
def test_redirections_before_the_script_are_not_its_operand(check, command) -> None:
    """A redirection belongs to the shell, wherever it stands in the command."""
    result = check(_bash(command, "done"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize("command", ["python3 <run.py", "python3 < run.py", "wc -l < run.py"])
def test_a_script_fed_through_standard_input_is_still_unresolved(check, command) -> None:
    result = check(_bash(command, "done"), EXPECTED_SCRIPT)
    assert result["score"] == 0.75


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "if true; then ./run.py; fi",
        "for i in one; do ./run.py; done",
        "if false; then true; else python run.py; fi",
        "if false; then true; elif true; then python run.py; fi",
        "while read -r line; do python run.py; done < names.txt",
        "until ./run.py; do sleep 1; done",
        "if ./run.py; then echo ok; fi",
        "then FOO=1 python run.py",
        "if [ -f run.py ]; then python run.py; fi",
        "for i in 1 2; do timeout 5 python run.py; done",
        "for f in run.py; do python $f; done",
        'for f in run.py; do python "$f"; done',
        "for f in ./run.py; do $f; done",
        "for f in run.py; do echo $f; python $f; done",
    ],
)
def test_invocations_inside_control_structures_are_credited(check, command) -> None:
    """``then`` and ``do`` stand before a command rather than being one, and a
    loop header with a single value binds its variable for the body.
    """
    result = check(_bash(command, "done"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "if [ -f run.py ]; then cat run.py; fi",
        "for i in one; do cat run.py; done",
        "while true; do wc -l run.py; break; done",
        "if grep -q main run.py; then echo yes; fi",
        "for f in run.py; do cat $f; done",
        'for f in run.py; do wc -l "$f"; done',
    ],
)
def test_reading_the_script_inside_a_control_structure_is_still_zero(check, command) -> None:
    result = check(_bash(command, SCRIPT_SOURCE), EXPECTED_SCRIPT)
    assert result["score"] == 0.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "for f in run.py other.py; do python $f; done",
        "for f in run.py other.py; do cat $f; done",
        "for f in $(ls run.py); do python $f; done",
        "case $x in run.py) python run.py;; esac",
        "case x in x) ./run.py;; esac",
    ],
)
def test_unmodelled_control_syntax_naming_the_script_is_unresolved(check, command) -> None:
    """A header with several values settles nothing about its variable, and
    ``case`` bodies are not modelled at all, so a script named inside either
    is neither credited nor settled as unrun.
    """
    result = check(_bash(command, "done"), EXPECTED_SCRIPT)
    assert result["score"] == 0.75


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "script"),
    [
        ("/usr/bin/perl5.34.0 run.pl", "run.pl"),
        ("/usr/bin/perl5.38.2 run.pl", "run.pl"),
        ("perl5.38 -w run.pl", "run.pl"),
        ("python3.13 run.py", "run.py"),
        ("ruby3.2 run.rb", "run.rb"),
        ("node22 run.js", "run.js"),
        ("bash5 run.sh", "run.sh"),
    ],
)
def test_versioned_interpreter_names_resolve_their_script(check, command, script) -> None:
    result = check(_bash(command, "done"), script)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "script", "expected"),
    [
        ("perl5.38.2 -c run.pl", "run.pl", 0.0),
        ("perl5.38.2 --version run.pl", "run.pl", 0.0),
        ("python3.13 -m json.tool run.py", "run.py", 0.75),
        ("bash5.2 -c 'bash run.sh'", "run.sh", 1.0),
    ],
)
def test_a_versioned_interpreter_keeps_its_grammar(check, command, script, expected) -> None:
    """The version suffix changes the name, not the options."""
    result = check(_bash(command, "done"), script)
    assert result["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "python3 2 > numeric.out run.py",
        "python3 2 >numeric.out run.py",
        "python3 1 >> log.txt run.py",
        "python3 0 < notes.txt run.py",
    ],
)
def test_a_digit_standing_apart_from_its_operator_is_an_operand(check, command) -> None:
    """``2 > numeric.out`` is an argument ``2`` and a redirection of standard
    output, not a redirection of standard error: the shell reads a descriptor
    only when the digits are written flush against the operator. The
    interpreter therefore runs a script named ``2`` and hands it ``run.py``.
    """
    result = check(_bash(command, "done"), EXPECTED_SCRIPT)
    assert result["score"] == 0.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "python3 2>/dev/null run.py",
        "python3 2> /dev/null run.py",
        "python3 2>&1 run.py",
        "python3 1>>log.txt run.py",
        "python3 1> log.txt run.py",
        "python3 0</dev/null run.py",
        "python3 0< /dev/null run.py",
        "python3 2>/dev/null -u run.py",
        "echo '2>' ; python3 run.py",
        'echo "2 > x" ; python3 run.py',
    ],
)
def test_an_attached_descriptor_before_the_script_is_a_redirection(check, command) -> None:
    """The descriptor and its operator stay together through tokenizing, so a
    redirection written before the script never stands where the script is
    looked for, and a ``2>`` inside quotes is data.
    """
    result = check(_bash(command, "done"), EXPECTED_SCRIPT)
    assert result["score"] == 1.0


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("bash -c 2>/dev/null './run.sh'", 1.0),
        ("bash -c 2> /dev/null './run.sh'", 1.0),
        ("bash 2>/dev/null -c './run.sh'", 1.0),
        ("bash -c 2>/dev/null 'bash run.sh'", 1.0),
        ("bash -c 2>/dev/null -- './run.sh'", 1.0),
        ("bash -c 2>/dev/null 'cat ./run.sh'", 0.0),
        ("bash -c 2 './run.sh'", 0.0),
    ],
)
def test_the_shell_payload_is_read_from_the_same_stream_as_its_option(check, command, expected) -> None:
    """A redirection standing between ``-c`` and its payload belongs to the
    shell, so the payload is the next operand after it, not the redirection.
    ``bash -c 2 './run.sh'`` runs the command ``2`` with ``./run.sh`` as its
    ``$0``, and nothing runs the script.
    """
    result = check(_bash(command, "done"), "run.sh")
    assert result["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('for f in run.py; do cat "$f"; done; for f in other.py another.py; do python3 "$f"; done', 0.0),
        ("for f in run.py; do cat $f; done; for f; do python3 $f; done", 0.0),
        ("for f in run.py; do cat $f; done; for f in; do python3 $f; done", 0.0),
        ('for f in run.py; do cat "$f"; done; for f in run.py other.py; do python3 "$f"; done', 0.75),
        ("for f in other.py; do cat $f; done; for f in run.py; do python3 $f; done", 1.0),
        ("for f in run.py; do cat $f; done; for g in other.py another.py; do python3 $f; done", 1.0),
    ],
)
def test_a_loop_variable_rebound_without_a_unique_value_is_forgotten(check, command, expected) -> None:
    """A single-value header binds its variable for its own body. A later
    header that gives the same variable several values, or none, settles
    nothing about it, so the earlier value is dropped rather than read into
    the new body: the second loop above runs two other scripts, not ``run.py``.
    A loop variable keeps its last value after its loop ends, so a later loop
    over a different variable that runs ``python3 $f`` does run ``run.py``.
    """
    result = check(_bash(command, "done"), EXPECTED_SCRIPT)
    assert result["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("printf '' | for f in run.py; do cat $f; done; for g in x; do python3 $f; done", 0.0),
        ("for f in run.py; do cat $f; done | true; python3 $f", 0.0),
        ("if true; then for f in run.py; do cat $f; done; fi | true; python3 $f", 0.0),
        ("echo x | if true; then f=run.py; fi; python3 $f", 0.0),
        ("FOO=run.py | true; python3 $FOO", 0.0),
        ("printf '' | for f in run.py; do python3 $f; done", 1.0),
        ("echo x | while read -r l; do for f in run.py; do python3 $f; done; done", 1.0),
        ("for f in run.py; do cat $f; done; python3 $f", 1.0),
        ("if true; then f=run.py; fi; python3 $f", 1.0),
        ("f=run.py; echo x | python3 $f", 1.0),
    ],
)
def test_a_binding_made_inside_a_pipeline_does_not_outlive_it(check, command, expected) -> None:
    """Each command of a pipeline runs in a subshell, a compound command
    included, so a loop variable or assignment made there is gone once the
    pipeline ends; the body of a loop piped into still sees its own header.
    Outside a pipeline a loop variable keeps its last value, and a value
    bound before the pipeline is read by a command inside it.
    """
    result = check(_bash(command, "done"), EXPECTED_SCRIPT)
    assert result["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("attached", "spaced", "expected"),
    [
        ("python3 0<run.py", "python3 0< run.py", 0.75),
        ("python3 <run.py", "python3 < run.py", 0.75),
        ("python3 0<other.py run.py", "python3 0< other.py run.py", 1.0),
        ("python3 0</dev/null run.py", "python3 0< /dev/null run.py", 1.0),
        ("python3 2>err.txt run.py", "python3 2> err.txt run.py", 1.0),
        ("python3 2>>err.txt run.py", "python3 2>> err.txt run.py", 1.0),
        ("python3 3<>notes.txt run.py", "python3 3<> notes.txt run.py", 1.0),
        ("python3 2>&1 run.py", "python3 2>&1 run.py", 1.0),
        ("python3 2>&- run.py", "python3 2>&- run.py", 1.0),
        ("python3 2>1.log run.py", "python3 2> 1.log run.py", 1.0),
        ("python3 1>2.txt run.py", "python3 1> 2.txt run.py", 1.0),
        ("python3 run.py 2>err.txt", "python3 run.py 2> err.txt", 1.0),
        ("cat 0<run.py", "cat 0< run.py", 0.75),
    ],
)
def test_an_operand_written_flush_against_its_descriptor_tokenizes_as_the_spaced_form(
    check, attached, spaced, expected
) -> None:
    """``0<run.py`` and ``0< run.py`` are the same redirection, so they must
    score the same. The descriptor stays with its operator, and the operand
    becomes its own word rather than concatenating onto the quoted operator.
    A descriptor duplication (``2>&1``) or close (``2>&-``) has no operand;
    ``2>1.log`` has one, and its whole name is kept. A script fed to standard
    input is unresolved, whichever
    command reads it, as ``python3 < run.py`` already was.
    """
    assert check(_bash(attached, "done"), EXPECTED_SCRIPT)["score"] == expected
    assert check(_bash(spaced, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize(
    "reads_skill_md",
    [
        pytest.param(shared_checks._cmd_reads_skill_md, id="host"),
        pytest.param(TEMPLATE._cmd_reads_skill_md, id="harbor-verifier"),
    ],
)
def test_a_skill_md_read_through_an_attached_input_redirection_is_a_read(reads_skill_md) -> None:
    """``cat 0<SKILL.md`` reads the file just as ``cat 0< SKILL.md`` does, and
    ``cat 2>SKILL.md`` overwrites it with standard error and reads nothing.
    """
    assert reads_skill_md("cat 0<SKILL.md") is True
    assert reads_skill_md("cat 0< SKILL.md") is True
    assert reads_skill_md("cat <SKILL.md") is True
    assert reads_skill_md("cat 2>SKILL.md") is False
    assert reads_skill_md("cat 2> SKILL.md") is False


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('f=run.py; for f in; do :; done; python3 "$f"', 1.0),
        ('f=run.py; select f in; do :; done; python3 "$f"', 1.0),
        ('for f in run.py; do :; done; for f in; do :; done; python3 "$f"', 1.0),
        ("for f in; do :; done; python3 run.py", 1.0),
        ('for f in ""; do python3 run.py; done', 1.0),
        ("for f in; do python3 run.py; done", 0.0),
        ('f=run.py; for f in; do python3 "$f"; done', 0.0),
        ('f=run.py; for f in; do :; done; cat "$f"', 0.0),
        ("for f in run.py; do cat $f; done; for f in; do python3 $f; done", 0.0),
        ('f=run.py; printf "" | for f in; do :; done; python3 "$f"', 1.0),
    ],
)
def test_a_loop_over_an_empty_list_keeps_the_binding_and_runs_no_body(check, command, expected) -> None:
    """``for f in; do ...; done`` runs zero iterations and never assigns ``f``,
    so a value it held before still stands for the command after the loop.
    Every shell tested (bash, dash, zsh, ksh, mksh) agrees, for ``select`` too.
    The body never runs, so an invocation written inside it is no evidence,
    while ``for f in ""`` iterates once over the empty string and its body
    does run. ``for f; do`` (no ``in``) is different: it iterates the
    positional parameters, which the text does not carry.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('zsh -c \'printf "" | for f in run.py; do cat "$f"; done; python3 "$f"\'', 1.0),
        ('ksh -c \'printf "" | for f in run.py; do cat "$f"; done; python3 "$f"\'', 1.0),
        ('bash -c \'printf "" | for f in run.py; do cat "$f"; done; python3 "$f"\'', 0.0),
        ('sh -c \'printf "" | for f in run.py; do cat "$f"; done; python3 "$f"\'', 0.0),
        ('dash -c \'printf "" | for f in run.py; do cat "$f"; done; python3 "$f"\'', 0.0),
        ('mksh -c \'printf "" | for f in run.py; do cat "$f"; done; python3 "$f"\'', 0.0),
        ('printf "" | for f in run.py; do cat "$f"; done; python3 "$f"', 0.0),
        ('zsh -c \'for f in run.py; do cat "$f"; done | cat; python3 "$f"\'', 0.0),
        ('zsh -c \'printf "" | f=run.py; python3 "$f"\'', 1.0),
        ('bash -c \'printf "" | f=run.py; python3 "$f"\'', 0.0),
        ("zsh -c 'echo x | if true; then f=run.py; fi; python3 \"$f\"'", 1.0),
        ('zsh -c \'printf "" | { f=run.py; }; python3 "$f"\'', 1.0),
        ('zsh -c \'printf "" | for f in run.py; do python3 "$f"; done\'', 1.0),
        ('bash -c \'zsh -c "printf \\"\\" | for f in run.py; do :; done; python3 \\$f"\'', 1.0),
        ('bash -c \'shopt -s lastpipe; printf "" | for f in run.py; do :; done; python3 "$f"\'', 0.75),
        ('bash -O lastpipe -c \'printf "" | for f in run.py; do :; done; python3 "$f"\'', 0.75),
    ],
)
def test_the_last_stage_of_a_pipeline_keeps_its_binding_where_the_shell_does(check, command, expected) -> None:
    """zsh and ksh run the last command of a pipeline in the current shell, so
    a loop variable or assignment made there outlives the pipeline; bash, sh,
    dash and mksh fork it like every other stage, so it is gone. A command
    piped into another stage is in a subshell in every shell. The ``-c``
    payload carries its shell into the walk, nested shells included. The tool
    call's own shell is read as bash. bash with ``lastpipe`` named keeps the
    binding only if that option is in force when the pipeline runs, which the
    text does not settle, so the walk runs under both readings and reports
    their disagreement as unresolved rather than as "did not run".
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('(f=run.py); python3 "$f"', 0.0),
        ("(for f in run.py; do :; done); python3 $f", 0.0),
        ('(printf "" | for f in run.py; do cat "$f"; done; python3 "$f")', 0.0),
        ('zsh -c \'printf "" | (f=run.py); python3 "$f"\'', 0.0),
        ("(f=run.py) | cat; python3 $f", 0.0),
        ("(cd x); (f=run.py); python3 $f", 0.0),
        ('(f=run.py; python3 "$f")', 1.0),
        ('{ f=run.py; }; python3 "$f"', 1.0),
        ("f=run.py; (g=x); python3 $f", 1.0),
        ("f=run.py; (python3 $f)", 1.0),
        ('(printf "" | for f in run.py; do python3 "$f"; done)', 1.0),
        ("((f=1)); f=run.py; python3 $f", 1.0),
    ],
)
def test_a_binding_made_inside_a_subshell_group_never_escapes_it(check, command, expected) -> None:
    """``( ... )`` runs in a subshell in every shell, so what is bound inside
    the parentheses is gone after them, whether the binding was an assignment,
    a loop variable, or made inside a pipeline within the group. A ``{ ... }``
    group runs in the current shell and its bindings stay. Commands inside the
    group still read what was bound before it, and still run the script.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("ksh run.sh", 1.0),
        ("mksh run.sh", 1.0),
        ("ash run.sh", 1.0),
        ("ksh -c './run.sh'", 1.0),
        ("mksh -c 'sh run.sh'", 1.0),
        ("ksh -c 'cat run.sh'", 0.0),
        ("ksh -n run.sh", 0.0),
        ("ksh -x run.sh", 1.0),
    ],
)
def test_ksh_mksh_and_ash_are_shells(check, command, expected) -> None:
    """They run a script named as their operand and carry a ``-c`` payload
    into the walk like sh, with sh's options: ``-n`` only parses, ``-x`` traces
    while running.
    """
    assert check(_bash(command, "done"), "run.sh")["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("for f in; do :; 'done'; python3 run.py; done", 0.0),
        ("for f in; do d'on'e; python3 run.py; done", 0.0),
        ("for f in; do :; d\\one; python3 run.py; done", 0.0),
        ("for f in; do \\done; python3 run.py; done", 0.0),
        ("for f in; do 'for'; done; python3 run.py", 1.0),
        ('f=run.py; for g in; do printf done; done; python3 "$f"', 1.0),
        ('f=run.py; for g in; do for f in other.py; do :; done; done; python3 "$f"', 1.0),
        ('f=run.py; printf "("; f=other.py; printf ")"; python3 "$f"', 0.0),
        ('f=run.py; printf \\(; f=other.py; printf \\); python3 "$f"', 0.0),
        ('f=run.py; printf "{"; f=other.py; printf "}"; python3 "$f"', 0.0),
        ("python3 '|' run.py", 0.0),
        ("python3 ';' run.py", 0.0),
        ("python3 '>' run.py", 0.0),
        ("python3 '0<run.py'", 0.0),
        ("'python3' run.py", 1.0),
        ('"python3" ./run.py', 1.0),
        ("python3 run.py '|' cat", 1.0),
        ("printf '(' ; python3 run.py", 1.0),
    ],
)
def test_a_quoted_or_escaped_word_is_not_shell_syntax(check, command, expected) -> None:
    """``'done'`` is a command named done, ``printf "("`` prints a parenthesis
    and ``python3 '|' run.py`` runs a script named ``|``. The tokenizer drops
    the quotes, so a word that would read as syntax once unquoted is marked
    before tokenizing and the walk reads it as the ordinary word it is: the
    empty loop's real ``done`` still ends the skipped body, a quoted
    parenthesis opens no subshell, and a quoted separator splits no command.
    Quoting a command name changes nothing: ``'python3' run.py`` runs it.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        'zsh -c \'emulate sh; printf "" | for f in run.py; do :; done; python3 "$f"\'',
        'zsh -c \'setopt shwordsplit; printf "" | for f in run.py; do :; done; python3 "$f"\'',
        'bash -c \'shopt -s lastpipe; printf "" | for f in run.py; do :; done; python3 "$f"\'',
    ],
)
def test_a_shell_option_change_leaves_the_pipeline_rule_unsettled(check, command) -> None:
    """``emulate sh`` makes zsh fork the last stage (measured), and
    ``shopt -s lastpipe`` makes bash keep it. Which options are in force when
    the pipeline runs is not something the text settles, so a command that
    names one is walked under both readings and their disagreement is
    unresolved.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == 0.75


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('f=run.py; for f in; do :; done; for f; do :; done; python3 "$f"', 1.0),
        ("f=run.py; for f; do :; done; python3 $f", 1.0),
        ("for f; do python3 run.py; done", 0.0),
        ("for f in run.py; do cat $f; done; for f; do python3 $f; done", 0.0),
        ("set -- run.py; for f; do python3 $f; done", 0.75),
        ("f=run.py; shift; for f; do :; done; python3 $f", 0.75),
        ("bash -c 'for f; do python3 $f; done' _ run.py", 0.75),
        ("bash -c 'for f; do python3 $f; done' run.py", 0.0),
        ("bash -c 'python3 $1' _ run.py", 0.75),
        ("bash -c 'python3 \"$@\"' _ run.py", 0.75),
        ("bash -c '\"$0\"' ./run.py", 0.75),
        ("bash -c 'python3 other.py' _ run.py", 0.0),
        ("bash -c 'echo done' run.py", 0.0),
    ],
)
def test_the_positional_parameters_are_empty_unless_the_text_gives_some(check, command, expected) -> None:
    """A tool call runs with no positional parameters, so ``for f; do`` at the
    top level runs zero times and keeps the variable's earlier value, like
    ``for f in;``. Where the text can give it some (``set --``, ``shift``, or
    operands after a ``-c`` payload beyond its ``$0``) a later read of the
    variable is unresolved rather than a settled miss, and an operand that
    names the script reaches the payload only through ``$1``, ``$@``, ``$0``
    or ``for f; do`` written in it.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cat run.py | python3", 0.75),
        ("cat 'run.py' | python3", 0.75),
        ("cat run.py | python3 -", 0.75),
        ("cat run.py | python3 -u", 0.75),
        ("python3 - < run.py", 0.75),
        ("cat run.py | python3 -c 'print(1)'", 0.0),
        ("cat run.py | python3 --version", 0.0),
        ("cat run.py | python3 other.py", 0.0),
        ("cat run.py | wc -l", 0.0),
        ("cat run.py | grep x | python3", 0.75),
        ("cat other.py | python3; cat run.py", 0.0),
    ],
)
def test_an_interpreter_reading_its_program_from_a_pipe_is_unresolved(check, command, expected) -> None:
    """``cat run.py | python3`` runs the script and ``cat run.py | wc -l`` does
    not; an interpreter with no script, no inline code and no terminal option
    reads its program from standard input, the same shape as
    ``python3 < run.py``, so when an earlier stage of its pipeline names the
    script the command is unresolved. ``-`` names standard input.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("printf '%s' ';|' python3 run.py", 0.75),
        ("printf '%s' \\;\\| python3 run.py", 0.75),
        ("python3 ';;' run.py", 0.0),
        ("python3 '|&' run.py", 0.0),
        ("python3 '&&' run.py", 0.0),
        ("python3 '2>' run.py", 0.0),
        ("python3 '<<-' run.py", 0.0),
        ("python3 run.py ';|'", 1.0),
        ("python3 run.py '&&' cat", 1.0),
        ("python3 '\\ue000run.py'", 0.0),
        ("python3 \\ue000run.py", 0.0),
        ("python3 '\\ue000\\ue000run.py'", 0.0),
        ("python3 run.py '\\ue000'", 1.0),
    ],
)
def test_a_quoted_run_of_metacharacters_is_one_word_and_a_mark_character_is_data(check, command, expected) -> None:
    """A quoted word made of metacharacters (``';|'``, ``'2>'``) would be split
    into operators after tokenizing, so it is marked like a quoted reserved
    word; ``printf '%s' ';|' python3 run.py`` is one printf, which names the
    interpreter and the script and is unresolved, not a separate python3
    command. A file whose name carries the mark character itself is a
    different file: the character is doubled before marking, so ``\\ue000run.py``
    never reads as ``run.py``.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('f=other.py; { f=run.py; :; } | cat; python3 "$f"', 0.0),
        ('f=other.py; { :; f=run.py; } | cat | cat; python3 "$f"', 0.0),
        ('f=other.py; printf "" | { :; f=run.py; } | cat; python3 "$f"', 0.0),
        ('f=other.py; printf "" | { :; f=run.py; }; python3 "$f"', 0.0),
        ('zsh -c \'f=other.py; printf "" | { f=run.py; :; } | cat; python3 "$f"\'', 0.0),
        ('zsh -c \'f=other.py; printf "" | { :; f=run.py; }; python3 "$f"\'', 1.0),
        ('bash -c \'zsh -c "printf \\\\"\\\\" | { f=run.py; :; } | cat; python3 \\$f"\'', 0.0),
        ('f=other.py; { f=run.py; :; }; python3 "$f"', 1.0),
        ('f=other.py; printf "" | (f=run.py; python3 "$f"); python3 "$f"', 1.0),
        ('f=other.py; printf "" | (f=run.py; python3 "$f") | cat; python3 "$f"', 1.0),
        ('printf "" | (for f in run.py; do :; done; python3 $f)', 1.0),
        ('printf "" | (f=run.py); python3 "$f"', 0.0),
    ],
)
def test_a_brace_group_is_one_pipeline_stage_and_a_subshell_keeps_its_own_bindings(check, command, expected) -> None:
    """``{ ...; } | cat`` runs the whole group as one stage of the pipeline, so
    a binding made inside it is gone afterwards in bash, and stays only when
    the group is the last stage under zsh. A ``( ... )`` group in a pipeline
    is its own scope: a binding made inside it reaches the group's later
    commands, and the pipe before ``(`` belongs to the group, not to the
    first command inside it.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('f=run.py; (f=other.py; for g in; do :; done); python3 "$f"', 1.0),
        ('f=run.py; (f=other.py; for g in; do :; done) | cat; python3 "$f"', 1.0),
        ('f=run.py; (f=other.py; for g in; do :; done); (:); python3 "$f"', 1.0),
        ('f=run.py; (f=other.py; for g in; do for h in; do :; done; done); python3 "$f"', 1.0),
        ('f=other.py; (f=run.py; for g in; do :; done); python3 "$f"', 0.0),
        ('f=other.py; (f=run.py; for g in; do for h in; do :; done; done) | cat; python3 "$f"', 0.0),
        ("bash -c 'bash -c '\"'\"'f=run.py; (f=other.py; for g in; do :; done); python3 \"$f\"'\"'\"''", 1.0),
        ("zsh -c 'zsh -c '\"'\"'f=run.py; (f=other.py; for g in; do :; done); python3 \"$f\"'\"'\"''", 1.0),
        ('f=run.py; (for g in; do :; done; f=other.py); python3 "$f"', 1.0),
        ('f=other.py; for g in; do (f=run.py); done; python3 "$f"', 0.0),
    ],
)
def test_an_empty_loop_skipped_inside_a_subshell_still_closes_the_subshell(check, command, expected) -> None:
    """Skipping the body of an empty loop must not skip the ``)`` that ends the
    group it sits in: the group's bindings stay inside the group, and the
    command after it reads the outer value. A parenthesis inside the skipped
    body counts too.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('(f=run.py; f=other.py | cat; python3 "$f")', 1.0),
        ('(f=other.py; f=run.py | cat; python3 "$f")', 0.0),
        ('(f=run.py; printf "" | f=other.py; python3 "$f")', 1.0),
        ('(f=other.py; printf "" | f=run.py; python3 "$f")', 0.0),
        ('(f=run.py; { f=other.py; } | cat; python3 "$f")', 1.0),
        ('(f=other.py; { f=run.py; } | cat; python3 "$f")', 0.0),
        ('((f=other.py; f=run.py | cat); python3 "$f")', 0.0),
        ('f=run.py; ((f=other.py) | cat; python3 "$f")', 1.0),
        ('(f=run.py; (f=other.py | cat); python3 "$f")', 1.0),
        ('{ f=run.py; f=other.py | cat; python3 "$f"; }', 1.0),
        ('(f=other.py; for g in run.py; do :; done | cat; python3 "$g")', 0.0),
        ('zsh -c \'(f=other.py; printf "" | f=run.py; python3 "$f")\'', 1.0),
        ("zsh -c '(f=other.py; f=run.py | cat; python3 \"$f\")'", 0.0),
        ('zsh -c \'(f=run.py; printf "" | f=other.py; python3 "$f")\'', 0.0),
        ('printf "" | (f=run.py; python3 "$f")', 1.0),
        ('printf "" | (f=run.py; python3 "$f") | cat', 1.0),
    ],
)
def test_a_pipeline_inside_a_subshell_still_isolates_its_own_stages(check, command, expected) -> None:
    """The group's scope is shared by the commands inside it, but a pipeline
    written inside the group still runs each stage in its own subshell:
    ``(f=run.py; f=other.py | cat; python3 "$f")`` runs run.py under bash,
    and under zsh the last stage's binding survives inside the group as it
    would outside. Only the pipe before ``(`` belongs to the group itself.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('bash -c \'printf "" | { f=run.py; (f=other.py); python3 "$f"; }\'', 1.0),
        ('bash -c \'printf "" | { f=other.py; (f=run.py); python3 "$f"; }\'', 0.0),
        ('zsh -c \'printf "" | { f=run.py; (f=other.py); python3 "$f"; }\'', 1.0),
        ('zsh -c \'printf "" | { f=other.py; (f=run.py); python3 "$f"; }\'', 0.0),
        ('printf "" | for g in 1; do f=run.py; (f=other.py); python3 "$f"; done', 1.0),
        ('if true; then f=other.py; (f=run.py); python3 "$f"; fi | cat', 0.0),
        ('f=run.py; { { f=other.py; } | cat; }; python3 "$f"', 1.0),
        ('f=other.py; { if true; then f=run.py; fi | cat; }; python3 "$f"', 0.0),
        ('f=run.py; printf "" | { printf "" | { f=other.py; }; }; python3 "$f"', 1.0),
        (
            'zsh -c \'f=other.py; printf "x\\n" | while read -r l; do { printf "" | f=run.py; }; done; python3 "$f"\'',
            1.0,
        ),
        ('zsh -c \'f=other.py; { printf "" | f=run.py; }; python3 "$f"\'', 1.0),
        ('f=other.py; { printf "" | f=run.py; }; python3 "$f"', 0.0),
        ('f=run.py; { for g in; do :; done; f=other.py | cat; }; python3 "$f"', 1.0),
    ],
)
def test_every_scope_nests_in_either_order_and_each_compound_has_its_own_pipe(check, command, expected) -> None:
    """Groups and compound pipeline stages are one stack of scopes, innermost
    last: a ``( ... )`` inside a compound stage copies the stage's bindings and
    drops its own at ``)``, so the stage's value stands after it. Each compound
    a segment opens is tested for its own pipe: in ``{ if ...; fi | cat; }``
    only the ``if`` is a pipeline stage, and in ``{ printf "" | f=x; }`` the
    pipe belongs to the command inside the braces, not to the braces.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('f=run.py; ((f=other.py)); python3 "$f"', 0.75),
        ("ksh -c 'f=run.py; ((f=other.py)); python3 \"$f\"'", 0.75),
        ("f=run.py; ((f++)); python3 $f", 0.75),
        ("f=run.py; ((g=1)); python3 $f", 1.0),
        ('f=other.py; ((f=run.py)); python3 "$f"', 0.0),
        ("((f=1)); python3 $f", 0.0),
        ("f=run.py; ((n > 0)); python3 $f", 1.0),
        ('f=run.py; ((f=other.py; g=1); python3 "$f")', 1.0),
        ('((f=other.py; f=run.py | cat); python3 "$f")', 0.0),
        ("echo $((2 << 1)); python3 run.py", 1.0),
        ("for ((i=0; i<1; i++)); do python3 run.py; done", 1.0),
    ],
)
def test_an_arithmetic_command_is_not_a_pair_of_subshells(check, command, expected) -> None:
    """``((f=x))`` at command position is arithmetic and runs in the current
    shell: the value it gives a variable is a number, never a script path; on
    an error bash, zsh and mksh leave the variable as it was, and ksh aborts
    the rest of the text. So a variable that held the script is unsettled
    afterwards, and any other stays not the script. ``((f=x; g=y); z)``, closed by a single ``)``, is two nested
    subshells, as every shell reads it; ``$((`` and ``for ((`` are unchanged.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('f=other.py; ((f=run.py; python3 "$f")); python3 "$f"', 0.0),
        ('zsh -c \'f=other.py; ((printf "" | for g in 1; do f=run.py; python3 "$f"; done)); python3 "$f"\'', 0.0),
        ('zsh -c \'f=other.py; (({ f=run.py; }; python3 "$f")); python3 "$f"\'', 0.0),
        ('mksh -c \'f=other.py; ((f=run.py; python3 "$f")); python3 "$f"\'', 0.0),
        ('dash -c \'f=other.py; ((f=run.py; python3 "$f")); python3 "$f"\'', 1.0),
        ('sh -c \'f=other.py; ((f=run.py; python3 "$f")); python3 "$f"\'', 0.75),
        ('f=other.py; ((f=run.py; python3 "$f") ); python3 "$f"', 1.0),
        ('f=other.py; ((f=run.py) ; python3 "$f"); python3 "$f"', 0.0),
        ("f=run.py; ((x=(1+2)*3)); python3 $f", 1.0),
        ("f=run.py; ((f=(1+2))); python3 $f", 0.75),
        ("f=run.py; ((a*(b+c))) 2>/dev/null; python3 $f", 1.0),
    ],
)
def test_double_parentheses_are_read_by_how_they_close_and_by_shell(check, command, expected) -> None:
    """bash, zsh, ksh and mksh read ``((`` as arithmetic when its two halves
    close together as ``))``, whatever is written between: ``((f=x; cmd))`` is
    an arithmetic expression that fails and runs nothing. ``((cmd) )`` and
    ``((f=x) ; cmd)`` close apart and are nested subshells. dash has no
    arithmetic command and always reads subshells; ``sh`` is dash on some
    systems and not others, so there the reading is unresolved. Parentheses
    inside the arithmetic body are counted and stay in it.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('export f=run.py; python3 "$f"', 1.0),
        ('readonly f=run.py; python3 "$f"', 1.0),
        ("export f=run.py; python3 ${f}", 1.0),
        ('declare f=run.py; python3 "$f"', 1.0),
        ('typeset -x f=run.py; python3 "$f"', 1.0),
        ('export -- f=run.py g=x; python3 "$f"', 1.0),
        ('f=run.py; export f; python3 "$f"', 1.0),
        ('export f=run.py; cat "$f"', 0.0),
        ('readonly f=run.py; head -5 "$f"', 0.0),
        ("dash -c 'declare f=run.py; python3 \"$f\"'", 0.0),
        ("zsh -c 'declare f=run.py; python3 \"$f\"'", 1.0),
        ("ksh -c 'typeset f=run.py; python3 \"$f\"'", 1.0),
        ("sh -c 'declare f=run.py; python3 \"$f\"'", 0.75),
        ('local f=run.py; python3 "$f"', 0.75),
        ('declare -u f=run.py; python3 "$f"', 0.75),
        ('f=other.py; echo run.py | { read f; python3 "$f"; }', 0.75),
        ('printf -v f run.py; python3 "$f"', 0.75),
        ('readonly f=run.py; f=other.py; python3 "$f"', 0.75),
        ('f=run.py; unset f; python3 "$f"', 0.0),
        ('f=run.py; unset -f f; python3 "$f"', 1.0),
    ],
)
def test_builtins_that_bind_a_variable_are_read_as_bindings(check, command, expected) -> None:
    """``export f=run.py`` and ``readonly f=run.py`` bind as ``f=run.py`` does
    in every shell; ``declare`` binds in bash and zsh and ``typeset`` also in
    ksh and mksh, and where a shell lacks one the command fails and binds
    nothing, so ``sh`` (dash on some systems, bash on others) is unresolved.
    ``unset f`` leaves ``$f`` empty. A value the text cannot settle leaves the
    variable unsettled rather than a settled miss: ``local`` outside a function
    (it binds in zsh and mksh only), an option that changes the value
    (``declare -u``), ``read`` and ``printf -v``, and an assignment to a
    read-only name. Reading the script stays uncredited.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('f=run.py python3 "$f"', 0.0),
        ('f=run.py true; python3 "$f"', 0.0),
        ('env f=run.py python3 "$f"', 0.0),
        ('f=run.py :; python3 "$f"', 0.0),
        ("bash -c 'f=run.py :; python3 \"$f\"'", 0.0),
        ("dash -c 'f=run.py :; python3 \"$f\"'", 1.0),
        ('f=run.py FOO=1; python3 "$f"', 1.0),
        ("export f=run.py; bash -c 'python3 \"$f\"'", 1.0),
        ("f=run.py; bash -c 'python3 \"$f\"'", 0.0),
        ("f=run.py bash -c 'python3 \"$f\"'", 1.0),
        ("env f=run.py bash -c 'python3 \"$f\"'", 1.0),
        ("f=run.py; env f=run.py bash -c 'python3 \"$f\"'", 1.0),
        ("env f=run.py bash -c 'cat \"$f\"'", 0.0),
        ('f=run.py; bash -c "python3 $f"', 1.0),
        ("set -a; f=run.py; bash -c 'python3 \"$f\"'", 1.0),
        ("set -o allexport; f=run.py; bash -c 'python3 \"$f\"'", 1.0),
        ("set -a; f=run.py; set +a; bash -c 'python3 \"$f\"'", 1.0),
        ("f=run.py; set -a; bash -c 'python3 \"$f\"'", 0.0),
        ("set -a; set +a; f=run.py; bash -c 'python3 \"$f\"'", 0.0),
        ('f=run.py; bash -c "python3 \\$f"', 0.0),
    ],
)
def test_an_assignment_before_a_command_is_that_commands_environment(check, command, expected) -> None:
    """``f=run.py python3 "$f"`` expands ``$f`` before the assignment applies,
    and the assignment is the command's environment only, so nothing after it
    sees ``f`` either; ``env f=run.py`` is the same. Before a special builtin
    (``:``) the assignment outlives the command in dash and the other POSIX
    shells, not in bash or zsh. A ``-c`` payload's shell sees what it inherits:
    the exported names (``export``, ``declare -x``, and anything bound while
    ``set -a`` is on) and the prefix of the command that started it, and a
    ``$f`` quoted or escaped from this shell is expanded there, not here.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("python3 <<A <<B run.py\nx\nA\ny\nB", 1.0),
        ("python3 <<-A <<B run.py\n\tx\n\tA\ny\nB", 1.0),
        ("python3 <<A <<B <<C run.py\nx\nA\ny\nB\nz\nC", 1.0),
        ("cat <<A; python3 <<B run.py\nx\nA\ny\nB", 1.0),
        ("cat <<A <<A\nx\nA\ny\nA\npython3 run.py", 1.0),
        ("cat <<A <<B run.py\nx\nA\ny\nB", 0.0),
        ("python3 <<A <<B other.py\nx\nA\ny\nB", 0.0),
        ("cat <<A <<B\nx\nA\npython3 run.py\nB", 0.75),
    ],
)
def test_every_heredoc_declared_on_a_line_takes_its_body_in_order(check, command, expected) -> None:
    """Every heredoc declared on one line, across commands joined on that line
    included, reads its body after the line ends, in the order declared, as
    bash, dash, zsh, ksh, mksh and busybox ash all do up to the number each
    accepts on a line (the limit is tested separately). The second declaration
    is not left in the command's words, and a body that is data (a line inside
    B's body) is not walked as a command.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('f=other.py; (((for f in run.py; do :; done; python3 "$f")) | cat); python3 "$f"', 0.0),
        ('f=other.py; printf "" | (((for f in run.py; do :; done; python3 "$f")) | cat); python3 "$f"', 0.0),
        ('zsh -c \'f=other.py; (((for f in run.py; do :; done; python3 "$f")) | cat); python3 "$f"\'', 0.0),
        ('ksh -c \'f=other.py; (((for f in run.py; do :; done; python3 "$f")) | cat); python3 "$f"\'', 1.0),
        ('dash -c \'f=other.py; (((for f in run.py; do :; done; python3 "$f")) | cat); python3 "$f"\'', 1.0),
    ],
)
def test_three_opening_parentheses_are_a_subshell_around_double_parentheses(check, command, expected) -> None:
    """The tokenizer hands ``(((`` over as one token. bash, zsh and mksh read it
    as ``( ((``, so the inner ``((...))`` is an arithmetic command that runs
    nothing; ksh and dash read nested subshells, where the loop's binding
    reaches ``python3 "$f"`` and the script runs.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("export f=run.py; env -i bash -c 'python3 \"$f\"'", 0.0),
        ("export f=run.py; env -u f bash -c 'python3 \"$f\"'", 0.0),
        ("export f=run.py; env --ignore-environment bash -c 'python3 \"$f\"'", 0.0),
        ("export f=run.py; env --unset=f bash -c 'python3 \"$f\"'", 0.0),
        ("export f=run.py; env -uf bash -c 'python3 \"$f\"'", 0.0),
        ('export f=run.py; env -i PATH="$PATH" bash -c \'python3 "$f"\'', 0.0),
        ("export f=run.py; env -i env bash -c 'python3 \"$f\"'", 0.0),
        ("f=run.py env -i bash -c 'python3 \"$f\"'", 0.0),
        ("set -a; f=run.py; env -u f sh -c 'python3 \"$f\"'", 0.0),
        ('export f=run.py; bash -c \'env -u f bash -c "python3 \\"\\$f\\""\'', 0.0),
        ("export f=run.py; env -u g bash -c 'python3 \"$f\"'", 1.0),
        ("export f=other.py; env -i f=run.py bash -c 'python3 \"$f\"'", 1.0),
        ("export f=run.py; env -u f f=run.py bash -c 'python3 \"$f\"'", 1.0),
        ("export f=run.py; env -u f env f=run.py bash -c 'python3 \"$f\"'", 1.0),
        ("export f=run.py; env -S '-u f' bash -c 'python3 \"$f\"'", 0.75),
        ("export f=run.py; env -i bash -c 'cat \"$f\"'", 0.0),
    ],
)
def test_env_removes_and_adds_names_before_the_child_starts(check, command, expected) -> None:
    """``env -i``, ``-u NAME`` and their long forms and clusters take names out of
    what the child inherits, in order, before and after its assignments: so
    ``env -i f=run.py`` gives the child ``f`` again, ``env -u g`` leaves ``f``
    alone, and a nested shell inherits only what reached its parent. ``env -S``
    splits a string the walk does not read, so the child's view is unresolved.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('f=other.py; export g=$f; f=run.py; python3 "$g"', 0.0),
        ('f=run.py; export g=$f; f=other.py; python3 "$g"', 1.0),
        ('f=other.py; readonly g=$f; f=run.py; python3 "$g"', 0.0),
        ('f=other.py; declare g=$f; f=run.py; python3 "$g"', 0.0),
        ('f=other.py; typeset g=$f; f=run.py; python3 "$g"', 0.0),
        ('f=other.py; g=$f; f=run.py; python3 "$g"', 0.0),
        ('f=run.py; g=$f; f=other.py; python3 "$g"', 1.0),
        ('f=run.py; g="${f}"; f=other.py; python3 "$g"', 1.0),
        ('export f=other.py g=$f; python3 "$g"', 0.0),
        ('f=run.py; export f=other.py g=$f; python3 "$g"', 1.0),
        ("ksh -c 'f=run.py; typeset f=other.py g=$f; python3 \"$g\"'", 0.0),
        ("ksh -c 'f=other.py; typeset f=run.py g=$f; python3 \"$g\"'", 1.0),
        ("f=run.py; export g='$f'; python3 \"$g\"", 0.0),
        ('f=run.py; export g=$f; unset f; python3 "$g"', 1.0),
        ("f=other.py; export g=$f; f=run.py; bash -c 'python3 \"$g\"'", 0.0),
        ("f=run.py; export g=$f; f=other.py; bash -c 'python3 \"$g\"'", 1.0),
        ('for f in run.py; do g=$f; done; f=other.py; python3 "$g"', 1.0),
    ],
)
def test_a_declaration_copies_the_value_when_it_runs(check, command, expected) -> None:
    """``export g=$f`` copies what ``f`` holds when it runs, as ``g=$f`` does;
    a later ``f=...`` does not reach ``g``. In one declaration of several names
    bash expands every value before assigning any, and ksh assigns in order
    (measured). A ``$`` quoted from this shell is kept literally.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("export f=run.py; declare +x f; bash -c 'python3 \"$f\"'", 0.0),
        ("export f=run.py; typeset +x f; bash -c 'python3 \"$f\"'", 0.0),
        ("export f=run.py; declare -x +x f; bash -c 'python3 \"$f\"'", 0.0),
        ("export f=run.py; export -n f; bash -c 'python3 \"$f\"'", 0.0),
        ("export f=run.py; command export -n f; bash -c 'python3 \"$f\"'", 0.0),
        ("export f=run.py; builtin export -n f; bash -c 'python3 \"$f\"'", 0.0),
        ("export f=run.py; command -p export -n f; bash -c 'python3 \"$f\"'", 0.0),
        ("export f=run.py; command command export -n f; bash -c 'python3 \"$f\"'", 0.0),
        ("export f=run.py; builtin declare +x f; bash -c 'python3 \"$f\"'", 0.0),
        ("export f=run.py; eval export -n f; bash -c 'python3 \"$f\"'", 0.75),
        ('dash -c \'export f=run.py; export -n f; bash -c "python3 \\"\\$f\\""\'', 0.75),
        ('zsh -c \'export f=run.py; command export -n f; bash -c "python3 \\"\\$f\\""\'', 1.0),
        ('export f=run.py; declare +x f; python3 "$f"', 1.0),
        ("export f=run.py; export -f f; bash -c 'python3 \"$f\"'", 1.0),
        ("export f=run.py; export -nf f; bash -c 'python3 \"$f\"'", 1.0),
        ("export f=run.py; declare +x f; export f; bash -c 'python3 \"$f\"'", 1.0),
        ("f=run.py; command export f; bash -c 'python3 \"$f\"'", 1.0),
        ("f=run.py; builtin export f; bash -c 'python3 \"$f\"'", 1.0),
    ],
)
def test_removing_the_export_attribute_is_honoured_however_the_builtin_is_reached(check, command, expected) -> None:
    """``declare +x``, ``typeset +x`` and ``export -n`` take the export away,
    and so do they after ``command``, ``command -p`` or ``builtin``, in any
    number, where the shell lets those reach the builtin (not zsh's
    ``command``). The value stays in this shell. ``export -f`` and ``-nf``
    touch functions only. ``dash`` rejects ``export -n``, and what ``eval``
    runs is not read exactly, so both leave the value unsettled.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


_OTHER_THEN_LISTED = 'export f=run.py; export -p f=other.py; python3 "$f"'
_LISTED_ALONE = 'export -p f=run.py; python3 "$f"'


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "shell", "expected"),
    [
        (_OTHER_THEN_LISTED, None, 0.0),
        (_OTHER_THEN_LISTED, "/bin/bash", 0.0),
        (_OTHER_THEN_LISTED, "/bin/ksh", 0.0),
        (_OTHER_THEN_LISTED, "/bin/mksh", 0.0),
        (_OTHER_THEN_LISTED, "/bin/zsh", 1.0),
        (_OTHER_THEN_LISTED, "/bin/dash", 1.0),
        (_OTHER_THEN_LISTED, "/bin/sh", 0.75),
        (_LISTED_ALONE, None, 1.0),
        (_LISTED_ALONE, "/bin/ksh", 1.0),
        (_LISTED_ALONE, "/bin/mksh", 1.0),
        (_LISTED_ALONE, "/bin/zsh", 0.0),
        (_LISTED_ALONE, "/bin/dash", 0.0),
        (_LISTED_ALONE, "/bin/sh", 0.75),
        ('readonly -p f=run.py; python3 "$f"', None, 1.0),
        ('readonly -p f=run.py; python3 "$f"', "/bin/zsh", 0.0),
        ('f=other.py; readonly -p f=run.py; python3 "$f"', None, 1.0),
        ('f=other.py; readonly -p f=run.py; python3 "$f"', "/bin/dash", 0.0),
        ('export f=run.py; readonly -p f=other.py; python3 "$f"', None, 0.0),
        ('export f=run.py; readonly -p f=other.py; python3 "$f"', "/bin/zsh", 1.0),
        ("export -p f=run.py; bash -c 'python3 \"$f\"'", None, 1.0),
        ("f=run.py; export -p f; bash -c 'python3 \"$f\"'", None, 1.0),
        ("f=run.py; export -p f; bash -c 'python3 \"$f\"'", "/bin/zsh", 0.0),
        ('f=run.py; readonly -p f; f=other.py; python3 "$f"', None, 0.75),
        ('f=run.py; readonly -p f; f=other.py; python3 "$f"', "/bin/dash", 0.0),
        ('f=run.py; export -p >/dev/null; python3 "$f"', None, 1.0),
        ('f=run.py; export -p >/dev/null; python3 "$f"', "/bin/zsh", 1.0),
        ('export -pn f=run.py; python3 "$f"', None, 1.0),
        ("f=run.py; export f; export -pn f; bash -c 'python3 \"$f\"'", None, 0.0),
        ("f=run.py; export f; export -pn f; bash -c 'python3 \"$f\"'", "/bin/zsh", 1.0),
        ('command export -p f=run.py; python3 "$f"', None, 1.0),
        ('command export -p f=run.py; python3 "$f"', "/bin/zsh", 0.0),
        ("bash -c 'export f=run.py; export -p f=other.py; python3 \"$f\"'", None, 0.0),
        ("zsh -c 'export -p f=run.py; python3 \"$f\"'", None, 0.0),
        ('declare -p f=run.py; python3 "$f"', None, 0.0),
        ('f=other.py; declare -px f=run.py; python3 "$f"', None, 0.0),
        ('f=other.py; typeset -p f=run.py; python3 "$f"', "/bin/zsh", 0.0),
        ('f=other.py; typeset -p f=run.py; python3 "$f"', "/bin/ksh", 0.75),
        ('f=other.py; typeset -px f=run.py; python3 "$f"', "/bin/mksh", 0.75),
        ('export f=run.py; export -p > f=other.py; python3 "$f"', None, 1.0),
        ('export f=run.py; export -p >f=other.py; python3 "$f"', "/bin/ksh", 1.0),
        ('export f=run.py; readonly -p 2> f=other.py; python3 "$f"', None, 1.0),
        ('f=other.py; export -p > f=run.py; python3 "$f"', None, 0.0),
        ("f=run.py; export -p > f; bash -c 'python3 \"$f\"'", None, 0.0),
        ('export f=run.py; export > f=other.py; python3 "$f"', None, 1.0),
        ("typeset -pF f=run.py; python3 run.py", "/bin/zsh", 1.0),
    ],
)
def test_a_listing_option_given_names_acts_on_them_where_the_shell_does(check, command, shell, expected) -> None:
    """``export -p`` and ``readonly -p`` given names act on them as they do
    without ``-p`` in bash, bash --posix, ksh, mksh and busybox ash, which list
    only when no name is given: ``export f=run.py; export -p f=other.py``
    runs other.py there, and ``export -p f=run.py`` binds and exports run.py.
    dash and zsh list the names and change nothing. ``declare -p`` and
    ``typeset -p`` list in bash and zsh, whatever else is given with them;
    ksh's ``typeset -p`` and mksh's ``typeset -px`` assign, which is read as
    unsettled. A redirection and its operand are not names: ``export -p >
    f=other.py`` writes the listing to a file and binds nothing. Measured
    with a marker in each shell.
    """
    assert check(_native(command, shell), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "shell", "expected"),
    [
        ('f=other.py; typeset -f f=run.py; python3 "$f"', None, 0.0),
        ('f=other.py; typeset -f f=run.py; python3 "$f"', "/bin/zsh", 0.0),
        ('f=other.py; typeset -f f=run.py; python3 "$f"', "/bin/mksh", 0.0),
        ('f=other.py; typeset -f f=run.py; python3 "$f"', "/bin/ksh", 0.75),
        ('f=run.py; declare -F f; python3 "$f"', None, 1.0),
        ('f=run.py; declare -F f; python3 "$f"', "/bin/zsh", 0.75),
        ("typeset -F f=run.py; python3 run.py", None, 1.0),
        ("typeset -F f=run.py; python3 run.py", "/bin/zsh", 0.75),
        ("typeset -F f=run.py; python3 run.py", "/bin/ksh", 0.75),
        ("typeset -F f=run.py; python3 run.py", "/bin/mksh", 1.0),
        ("typeset -E f=run.py; python3 run.py", "/bin/zsh", 0.75),
        ("typeset -F f=2; python3 run.py", "/bin/zsh", 1.0),
        ("f=1.5; typeset -F f; python3 run.py", "/bin/ksh", 1.0),
        ("typeset -F f; python3 run.py", "/bin/zsh", 1.0),
        ("g=run.py; typeset -F f=g; python3 run.py", "/bin/zsh", 0.75),
        ("f=run.py; (typeset -F f); python3 run.py", "/bin/zsh", 1.0),
        ("f=run.py; local -F f; python3 run.py", "/bin/zsh", 0.75),
        ("f=run.py; typeset -i f; python3 run.py", None, 1.0),
        ("f=run.py; typeset -i f; python3 run.py", "/bin/zsh", 0.75),
        ("f=run.py; typeset -i f; python3 run.py", "/bin/ksh", 0.75),
        ("f=run.py; typeset -i f; python3 run.py", "/bin/mksh", 1.0),
        ("f=2.5; typeset -i f; python3 run.py", "/bin/zsh", 1.0),
        ("typeset -i f=run.py; python3 run.py", "/bin/ksh", 0.75),
        ('f=run.py; typeset -L3 f; python3 "$f"', None, 1.0),
        ('f=run.py; typeset -L3 f; python3 "$f"', "/bin/zsh", 0.75),
        ('f=run.py; typeset -Z3 f; python3 "$f"', "/bin/ksh", 0.75),
        ('f=run.py; typeset -R8 f; python3 "$f"', "/bin/mksh", 0.75),
        ('f=run.py; typeset -L f; python3 "$f"', "/bin/zsh", 1.0),
        ('f=run.py; typeset -L 3 f; python3 "$f"', "/bin/zsh", 0.75),
        ('f=run.py; typeset -x -Z 3 f; python3 "$f"', "/bin/ksh", 0.75),
        ("typeset -i f=0x10; python3 run.py", "/bin/ksh", 1.0),
        ("typeset -i f=0x10; python3 run.py", "/bin/zsh", 1.0),
        ("typeset -i f=16#ff; python3 run.py", "/bin/ksh", 1.0),
        ("typeset -i f=2#101; python3 run.py", "/bin/zsh", 1.0),
        ("typeset -i f=99#1; python3 run.py", "/bin/zsh", 0.75),
        ("typeset -i f=2#9; python3 run.py", "/bin/zsh", 0.75),
        ('f=run.py; typeset -Lx 3 f; python3 "$f"', "/bin/zsh", 0.75),
        ("f=0x10; typeset -i f; python3 run.py", "/bin/zsh", 1.0),
        ("typeset -F f=0x10; python3 run.py", "/bin/ksh", 1.0),
        ("typeset -i f=g+1; python3 run.py", "/bin/zsh", 0.75),
        ("f=run.py; typeset -i +i f; python3 run.py", "/bin/ksh", 0.75),
        ('f=run.py; typeset -A m; python3 "$f"', "/bin/zsh", 1.0),
    ],
)
def test_function_float_and_width_options_follow_each_shell(check, command, shell, expected) -> None:
    """``-f`` and ``-F`` name functions and change no variable in bash, but not
    everywhere: ksh's ``typeset -f f=run.py`` assigns, and ``-F`` and ``-E``
    are float attributes in zsh and ksh. A numeric attribute (``-i`` too)
    given to a name whose value is not a number, given or held, stops zsh and
    ksh; a decimal, ``0x10``, ``16#ff`` or nothing is a number, and an
    expression is not evaluated. A width given with ``-L``, ``-R`` or ``-Z``,
    in the word or as the next one, cuts or pads the value at once in zsh, ksh
    and mksh, so the name is unsettled. Measured with a marker in each shell.
    """
    assert check(_native(command, shell), EXPECTED_SCRIPT)["score"] == expected


_ZEROS = "0" * 4400


@pytest.mark.parametrize("module", [shared_checks, TEMPLATE], ids=["host", "harbor-verifier"])
@pytest.mark.parametrize(
    ("command", "shell", "expected"),
    [
        ("typeset -F f=" + "9" * 4400 + "#1; python3 run.py", "/bin/zsh", 0.75),
        ("typeset -i f=" + "9" * 4301 + "#1; python3 run.py", "/bin/ksh", 0.75),
        ("typeset -i f=" + _ZEROS + "16#ff; python3 run.py", "/bin/zsh", 1.0),
        ("typeset -F f=" + _ZEROS + "36#z; python3 run.py", "/bin/ksh", 1.0),
        ("typeset -i f=" + _ZEROS + "2#9; python3 run.py", "/bin/zsh", 0.75),
        ("typeset -i f=" + _ZEROS + "37#1; python3 run.py", "/bin/zsh", 0.75),
        ("typeset -i f=016#ff; python3 run.py", "/bin/zsh", 1.0),
        ("typeset -i f=36#z; python3 run.py", "/bin/zsh", 1.0),
        ("typeset -i f=37#1; python3 run.py", "/bin/zsh", 0.75),
        ("typeset -i f=1#1; python3 run.py", "/bin/zsh", 0.75),
    ],
)
def test_a_numeric_base_of_any_length_is_scored(module, command, shell, expected) -> None:
    """zsh and ksh read a ``base#digits`` base as a decimal and drop its leading
    zeros, so ``016#ff`` and a base written after thousands of zeros are 16.
    A base past 36 is not a number the walk accepts, however many digits it
    has, and the command still gets a score from both public scorers: int()
    refuses a string of more than 4,300 digits, so the base is not converted
    until it is known to be short. Measured with a marker in zsh and ksh.
    """
    calls = _native(command, shell)
    assert module.check_script_execution(calls, EXPECTED_SCRIPT)["score"] == expected
    aggregate = module.score_skill_execution(calls, "demo", EXPECTED_SCRIPT)
    assert aggregate["details"]["script_execution"]["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "shell", "expected"),
    [
        ("export g='>' f=run.py; python3 \"$f\"", None, 1.0),
        ("export g='>' f=run.py; python3 \"$f\"", "/bin/zsh", 1.0),
        ("export g='>' f=run.py; python3 \"$f\"", "/bin/dash", 1.0),
        ("export g='>' f=run.py; python3 \"$f\"", "/bin/ksh", 1.0),
        ("f=run.py; export g='>' f=other.py; python3 \"$f\"", None, 0.0),
        ("f=run.py; export g='>' f=other.py; python3 \"$f\"", "/bin/zsh", 0.0),
        ("f=run.py; export g='>' f=other.py; python3 \"$f\"", "/bin/dash", 0.0),
        ("f=run.py; export g='>' f=other.py; python3 \"$f\"", "/bin/ksh", 0.0),
        ("export g='>>' f=run.py; python3 \"$f\"", None, 1.0),
        ("export g='>' f=run.py >output; python3 \"$f\"", None, 1.0),
        ('export g=\\> f=run.py; python3 "$f"', None, 1.0),
        ("export g='2>' f=run.py; python3 \"$f\"", None, 1.0),
        ("readonly g='>' f=run.py; python3 \"$f\"", "/bin/dash", 1.0),
        ("typeset g='>' f=run.py; python3 \"$f\"", "/bin/ksh", 1.0),
        ("f=run.py; typeset g='>' f=other.py; python3 \"$f\"", "/bin/zsh", 0.0),
        ("declare g='>' f=run.py; python3 \"$f\"", None, 1.0),
        ('export f=run.py > out; python3 "$f"', None, 1.0),
        ('export >out f=run.py; python3 "$f"', "/bin/dash", 1.0),
        ('export 2>err f=run.py; python3 "$f"', None, 1.0),
        ('export 2> err f=run.py; python3 "$f"', "/bin/ksh", 1.0),
        ('export f=run.py 2>&1; python3 "$f"', "/bin/zsh", 1.0),
        ('export f=run.py 0<run.py; python3 "$f"', None, 1.0),
        ('export g=2> f=run.py; python3 "$f"', None, 0.0),
        ('export 2 > f=run.py; python3 "$f"', None, 0.0),
        ('f=run.py; export -p > f=other.py; python3 "$f"', None, 1.0),
        ('f=other.py; export >>| f=run.py; python3 "$f"', "/bin/zsh", 0.0),
        ('f=other.py; export &>| f=run.py; python3 "$f"', "/bin/zsh", 0.0),
    ],
)
def test_an_assignment_ending_in_an_operator_is_not_a_redirection(check, command, shell, expected) -> None:
    """An unquoted redirection operator is a word of its own, so only such a
    word, with a descriptor written flush against it (``2>``), takes the next
    word as its operand. ``export g='>' f=run.py`` binds both names in every
    shell: ``g=>`` is an assignment that ends in ``>``, and f is bound. A real
    redirection, separate or attached, still takes its operand, zsh's ``>>|``
    and ``&>|`` included, and ``export g=2> f=run.py`` writes to a file named
    f=run.py. Measured with a marker in each shell.
    """
    assert check(_native(command, shell), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "shell", "expected"),
    [
        ('f=run.py; unset > f; python3 "$f"', "/bin/ksh", 0.75),
        ('f=run.py; unset; python3 "$f"', "/bin/ksh", 0.75),
        ('f=run.py; unset -v; python3 "$f"', "/bin/ksh", 0.75),
        ('f=run.py; command unset; python3 "$f"', "/bin/ksh", 1.0),
        ('f=run.py; (unset); python3 "$f"', "/bin/ksh", 1.0),
        ('f=run.py; unset > f; python3 "$f"', None, 1.0),
        ('f=run.py; unset > f; python3 "$f"', "/bin/mksh", 1.0),
        ('f=run.py; unset; python3 "$f"', "/bin/dash", 1.0),
    ],
)
def test_ksh_stops_on_unset_given_no_name(check, command, shell, expected) -> None:
    """``unset`` given no name prints its usage and stops ksh, unless ``command``
    runs it or a subshell holds it; the other shells go on. The target of a
    redirection is not a name, so ``unset > f`` gives none. Measured with a
    marker in each shell.
    """
    assert check(_native(command, shell), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("module", [shared_checks, TEMPLATE], ids=["host", "harbor-verifier"])
@pytest.mark.parametrize(
    ("text", "second_unit", "comment"),
    [
        ("echo $'it\\'s # kept'\npython3 run.py", None, None),
        ("echo $'a\n# still quoted'\ntrue", "true", None),
        ("echo 'a\\'\npython3 run.py # gone", "python3 run.py # gone", "# gone"),
    ],
)
def test_an_ansi_c_quote_honours_its_escapes_when_finding_where_commands_end(
    module, text, second_unit, comment
) -> None:
    """``$'...'`` closes on ``'`` as ``'...'`` does, but a backslash escapes the
    character after it: in ``$'it\\'s # kept'`` the ``#`` is quoted and starts
    no comment, and in ``$'a\n# still quoted'`` the newline ends nothing. In
    ``'a\\'`` the backslash is literal, so the quote closes and a later
    ``#`` starts a comment. Where the text does not settle a newline (the
    escaped quote in the first), the lines are read as one unit.
    """
    starts, blanked = module._parse_unit_starts(text, "bash", False)
    assert starts == ([0] if second_unit is None else [0, text.index(second_unit)])
    if comment is None:
        assert blanked == text
    else:
        assert comment not in blanked and blanked.replace(" ", "") == text.replace(comment, "").replace(" ", "")


def _heredocs(count: int, before: str, after: str = "", body: str = "x", quote: str = "") -> str:
    """``before``, then ``count`` heredocs declared on one line, ``after``, and each body in order."""
    names = [f"D{index}" for index in range(count)]
    declared = " ".join(f"<<{quote}{name}{quote}" for name in names)
    return f"{before} {declared}{' ' + after if after else ''}\n" + "".join(f"{body}\n{name}\n" for name in names)


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (_heredocs(9, "cat", body="python3 run.py"), 0.75),
        (_heredocs(9, "cat", body="python3 run.py", quote="'"), 0.75),
        (_heredocs(12, "cat", body="python3 run.py"), 0.75),
        (_heredocs(16, "cat", body="python3 run.py"), 0.75),
        (_heredocs(17, "cat", body="python3 run.py"), 0.75),
        (_heredocs(16, "python3", "run.py"), 1.0),
        (_heredocs(17, "python3", "run.py"), 0.75),
        (_heredocs(9, "cat", body="python3 run.py") + "python3 run.py", 1.0),
        (_heredocs(16, "cat", body="python3 run.py") + "python3 run.py", 1.0),
        (_heredocs(17, "cat", body="python3 run.py") + "python3 run.py", 0.75),
        ("cat <<A\nx\nA\n" + _heredocs(17, "python3", "run.py"), 0.75),
        ("echo a <<< x\n" + _heredocs(16, "python3", "run.py"), 1.0),
        ("mksh -c '" + _heredocs(10, "python3", "run.py") + "'", 1.0),
        ("mksh -c '" + _heredocs(11, "python3", "run.py") + "'", 0.75),
        ("dash -c '" + _heredocs(17, "python3", "run.py") + "'", 1.0),
    ],
)
def test_a_line_with_more_heredocs_than_the_shell_accepts_is_data(check, command, expected) -> None:
    """Past the heredocs a shell accepts on one line, the line and everything
    after it are data: bash exits with "maximum here-document count exceeded"
    at 17 and mksh stops at 11 before running anything on the line (measured),
    while dash reads more. A script named there is unresolved, never credited;
    up to the limit every body is read in order.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("f=run.py eval 'python3 \"$f\"'", 0.75),
        ("f=run.py; eval 'python3 \"$f\"'", 0.75),
        ("export f=run.py; eval 'python3 \"$f\"'", 0.75),
        ("f=run.py; command eval 'python3 \"$f\"'", 0.75),
        ("f=other.py eval 'python3 \"$f\"'", 0.0),
        ('f=run.py; eval f=other.py; python3 "$f"', 0.75),
        ("f=run.py; eval 'unset f'; python3 \"$f\"", 0.75),
        ("f=run.py; eval 'read f <<< x'; python3 \"$f\"", 0.75),
        ('f=run.py; eval \'echo "$f"\'; python3 "$f"', 1.0),
        ("f=run.py python3 -c 'import os; os.system(\"python3 $f\")'", 0.75),
        ("env f=run.py python3 -c 'import os; os.system(\"python3 $f\")'", 0.75),
        ('f=run.py; bash <<EOF\npython3 "$f"\nEOF', 0.75),
        ("export f=run.py; bash <<'EOF'\npython3 \"$f\"\nEOF", 0.75),
    ],
)
def test_eval_and_inline_code_may_expand_a_quoted_variable(check, command, expected) -> None:
    """``eval`` reads its words again in this shell, with the command's prefix in
    effect, so a ``$f`` quoted from the first reading names the script there;
    a name ``eval`` may bind is unsettled after it. Inline code and a shell
    reading a heredoc or here-string receive the command's environment, and an
    unquoted heredoc body is expanded here, so a variable that holds the
    script leaves them unresolved rather than a settled miss.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("readonly f=run.py; f=other.py; python3 run.py", 0.75),
        ("echo x; readonly f=run.py; f=other.py; env -i f=run.py bash -c 'python3 \"$f\"'", 0.75),
        ("readonly f=run.py; export f=other.py; python3 run.py", 1.0),
        ("readonly f=run.py; f=other.py true; python3 run.py", 1.0),
        ("readonly f=run.py; unset f; python3 run.py", 1.0),
        ("dash -c 'readonly f=run.py; export f=other.py; python3 run.py'", 0.75),
        ("zsh -c 'readonly f=run.py; unset f; python3 run.py'", 0.75),
        ("ksh -c 'readonly f=run.py; f=other.py true; python3 run.py'", 1.0),
        ("sh -c 'readonly f=run.py; f=other.py :; python3 run.py'", 0.75),
        ("(readonly f=run.py; f=other.py); python3 run.py", 1.0),
        ("{ readonly f=run.py; f=other.py; } | cat; python3 run.py", 1.0),
        ("readonly f=run.py; (f=other.py; python3 run.py)", 0.75),
        ('readonly f=run.py; for f in other.py; do python3 "$f"; done', 0.75),
    ],
)
def test_nothing_after_a_refused_read_only_assignment_is_credited(check, command, expected) -> None:
    """An assignment to a read-only name fails. A bare one stops every shell
    (bash abandons the rest of the line); bash goes on after every other form
    and ksh after a prefix, while dash, zsh, mksh and ash stop (measured).
    Where the shell stops, a script named after the failure is unresolved; a
    subshell, a pipeline stage included, stops alone.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("export f=run.py; code='f=other.py'; eval \"$code\"; bash -c 'python3 \"$f\"'", 0.75),
        ("export f=run.py; code='export -n f'; eval \"$code\"; bash -c 'python3 \"$f\"'", 0.75),
        ("export f=run.py; code='unset f'; eval \"$code\"; bash -c 'python3 \"$f\"'", 0.75),
        ("export f=run.py; code='declare +x f'; eval \"$code\"; bash -c 'python3 \"$f\"'", 0.75),
        ("export f=run.py; code='export f=other.py'; eval \"$code\"; bash -c 'python3 \"$f\"'", 0.75),
        ("export f=run.py; code='f=other.py'; eval eval \"$code\"; bash -c 'python3 \"$f\"'", 0.75),
        ("export f=run.py; code='f=other.py'; command eval \"$code\"; bash -c 'python3 \"$f\"'", 0.75),
        ("export f=run.py; code='export -n f'; eval '$code'; bash -c 'python3 \"$f\"'", 0.75),
        ("export f=run.py; code='f=other.py'; eval '$code'; bash -c 'python3 \"$f\"'", 1.0),
        ("export f=other.py; code='f=run.py'; eval \"$code\"; bash -c 'python3 \"$f\"'", 0.75),
        ("export f=run.py; code='echo hi'; eval \"$code\"; bash -c 'python3 \"$f\"'", 1.0),
        ('export f=run.py; eval "$(echo f=other.py)"; bash -c \'python3 "$f"\'', 0.75),
        ('f=run.py; code=\'g=1\'; eval "$code"; python3 "$f"', 1.0),
        ("code='python3 run.py'; eval \"$code\"", 0.75),
    ],
)
def test_eval_program_held_in_a_variable_is_read_after_this_shell_expands_it(check, command, expected) -> None:
    """This shell expands eval's words before eval reads them, so ``eval "$code"``
    runs what ``code`` holds: a name it reassigns, unsets or unexports is
    unsettled afterwards, through nested ``eval`` and ``command eval`` too.
    ``eval '$code'`` expands the text itself, where a word read from a
    variable is a command, never an assignment. Text this walk cannot read
    (a command substitution) leaves every bound name unsettled.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('f=run.py; declare -i f; f=run.py; python3 "$f"', 0.75),
        ('f=other.py; declare -i f; f=run.py; python3 "$f"', 0.75),
        ("export f=run.py; declare -a f; f=run.py; bash -c 'python3 \"$f\"'", 0.0),
        ('declare -a f; f=run.py; python3 "$f"', 1.0),
        ('declare -l f; f=RUN.PY; python3 "$f"', 1.0),
        ('declare -l f; f=run.py; python3 "$f"', 1.0),
        ('declare -u f; f=run.py; python3 "$f"', 0.75),
        ('declare -i f; unset f; f=run.py; python3 "$f"', 1.0),
        ('declare -i f; declare +i f; f=run.py; python3 "$f"', 1.0),
        ('declare -i f; for f in run.py; do python3 "$f"; done', 0.75),
    ],
)
def test_attributes_that_change_a_later_assignment_are_kept_with_the_name(check, command, expected) -> None:
    """``declare -i``, ``-u`` and ``-n`` change what a later assignment stores, so
    the value is unsettled until ``unset`` or ``+i`` removes the attribute;
    ``-l`` lowercases it; an array (``-a``) keeps its value in this shell and
    is never exported to a child.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('g=run.py; declare -n f=g; f=other.py; python3 "$g"', 0.75),
        ('g=other.py; declare -n f=g; f=run.py; python3 "$g"', 0.75),
        ('declare -n f=g; f=run.py; python3 "$g"', 0.75),
        ('g=run.py; declare -n f=g; unset f; python3 "$g"', 0.75),
        ('g=run.py; declare -n f=g; export f=other.py; python3 "$g"', 0.75),
        ('g=run.py; declare -n f=g; read f <<< other.py; python3 "$g"', 0.75),
        ('g=run.py; declare -n f=g; ((f=1)); python3 "$g"', 0.75),
        ('g=run.py; f=g; declare -n f; f=other.py; python3 "$g"', 0.75),
        ("export g=run.py; declare -n f=g; export -n f; bash -c 'python3 \"$g\"'", 0.75),
        ("ksh -c 'g=run.py; nameref f=g; f=other.py; python3 \"$g\"'", 0.75),
        ("mksh -c 'g=run.py; nameref f=g; f=other.py; python3 \"$g\"'", 0.75),
        ("bash -c 'g=run.py; nameref f=g; f=other.py; python3 \"$g\"'", 1.0),
        ('g=run.py; declare -n f=g; h=other.py; python3 "$g"', 1.0),
        ('g=run.py; declare -n f=g; declare +n f; f=other.py; python3 "$g"', 1.0),
        ('g=run.py; declare -n f=g; unset -n f; f=other.py; python3 "$g"', 1.0),
        ('g=run.py; declare -n f=g; declare -n f=h; f=other.py; python3 "$g"', 1.0),
        ('g=other.py; declare -n f=g; declare -n f=h; f=run.py; python3 "$g"', 0.0),
        ('g=run.py; declare -n f; f=g; python3 "$g"', 1.0),
        ('g=run.py; declare -n f; f=g; f=other.py; python3 "$g"', 0.75),
        ('h=run.py; declare -n f; f=g; python3 "$h"', 1.0),
        ('g=run.py; declare -n f="$(echo g)"; f=other.py; python3 "$g"', 0.75),
    ],
)
def test_a_reference_passes_what_is_done_to_it_on_to_the_name_it_refers_to(check, command, expected) -> None:
    """After ``declare -n f=g`` (``nameref`` in ksh and mksh), assigning,
    exporting, unexporting, reading into or unsetting ``f`` acts on ``g``,
    which is unsettled afterwards, bound or not. ``+n``, ``unset -n`` and a
    second ``-n`` free or re-point ``f`` and leave ``g`` alone, and an
    unrelated name leaves it settled. A reference that names nothing yet takes
    the first value assigned as its name; where the text does not settle the
    name, every bound name is unsettled.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("declare -i f; f=other.py; python3 run.py", 0.75),
        ("declare -i f; export f=run.py; python3 run.py", 0.75),
        ("declare -i f; for f in run.py; do :; done; python3 run.py", 0.75),
        ("declare -i f; f=5; python3 run.py", 1.0),
        ("declare -i f; f=x; python3 run.py", 1.0),
        ("declare -i f; f=other.py true; python3 run.py", 1.0),
        ("ksh -c 'typeset -i f; typeset f=other.py; python3 run.py'", 1.0),
        ("mksh -c 'typeset -i f; typeset f=other.py; python3 run.py'", 0.75),
        ("dash -c 'export f=1; export -n f; python3 run.py'", 0.75),
        ("dash -c 'export f=1; command export -n f; python3 run.py'", 1.0),
        ("bash -c 'export -Q f; python3 run.py'", 1.0),
        ("mksh -c 'unset -n f; python3 run.py'", 0.75),
        ("bash -c 'f=run.py; unset -n f; python3 \"$f\"'", 1.0),
        ("ksh -c 'f=run.py; unset -n f; python3 \"$f\"'", 0.0),
        ("zsh -c 'f=run.py; unset -Q f; python3 \"$f\"'", 1.0),
        ('g=run.py; declare -n f; f=other.py; python3 "$g"', 0.75),
        ('f=other.py; declare -n f; f=run.py; python3 "$f"', 1.0),
        ("ksh -c 'f=other.py; typeset -n f; python3 run.py'", 0.75),
    ],
)
def test_an_assignment_or_builtin_that_fails_where_the_shell_stops_ends_the_credit(check, command, expected) -> None:
    """A value that is not a number, given to a ``-i`` name, is an error, and
    bash exits there, as every shell with ``typeset`` does for most forms
    (bash goes on after a prefix, ksh after ``typeset``). A special builtin
    given an option the shell rejects stops a POSIX shell unless ``command``
    runs it; bash and zsh go on. A reference that names nothing given a value
    that is not a name fails as a read-only assignment does, and a declaration
    that refers to such a name fails, where ksh stops. Where the shell stops,
    a script named after the failure is unresolved (measured).
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("dash -c 'export g=run.py; read f <<< x; for f in x; do :; done; python3 \"$g\"'", 0.75),
        ("sh -c 'export g=run.py; command unset -Q f; read f <<< 7; python3 \"$g\"'", 0.75),
        ("dash -c 'cat <<< x; python3 run.py'", 0.75),
        ("dash -c 'export g=run.py; read f < /dev/null; python3 \"$g\"'", 1.0),
        ("bash -c 'cat <<< x; python3 run.py'", 1.0),
        ("zsh -c 'cat <<< x; python3 run.py'", 1.0),
        ("ksh -c 'cat <<< x; python3 run.py'", 1.0),
        ("mksh -c 'cat <<< x; python3 run.py'", 1.0),
        ("dash -c \"echo '<<<'; python3 run.py\"", 1.0),
        ("dash -c 'cat <<< x; cat run.py'", 0.0),
        ("cat <<< x; python3 run.py", 1.0),
    ],
)
def test_a_here_string_the_shell_cannot_parse_is_never_full_credit(check, command, expected) -> None:
    """dash and busybox ash have no here-string: ``<<<`` is a syntax error that
    stops the line holding it, and a compound command spanning lines around
    it, before anything on it runs (measured). Under those readings nothing
    from that line on is credited as run; ``sh`` disagrees across its
    readings and stays unresolved. Shells that read here-strings, and a
    quoted ``<<<``, are unchanged.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("bash -c 'cat <<EOF\npython3 run.py\nEOF'", 0.75),
        ("dash -c 'cat <<EOF\npython3 run.py\nEOF'", 0.75),
        ('bash -c "cat <<EOF\npython3 run.py\nEOF"', 0.75),
        ("bash -c 'cat <<EOF\nx\nEOF\npython3 run.py'", 1.0),
        ("bash -c 'echo a\npython3 run.py'", 1.0),
        ("bash -c 'python3 run.py <<EOF\nignored\nEOF'", 1.0),
        ("dash -c 'cat <<EOF\na <<< b\nEOF\npython3 run.py'", 1.0),
        ('bash -c "f=run.py; python3 \\"$f\\""', 0.0),
        ('bash -c "f=run.py\npython3 $f"', 0.0),
        ('f=run.py; bash -c "python3 \\"$f\\""', 1.0),
        ('bash -c "f=run.py; python3 \\"\\$f\\""', 1.0),
        ("bash -c 'f=run.py; python3 \"$f\"'", 1.0),
    ],
)
def test_a_heredoc_inside_a_c_payload_keeps_its_body_as_data(check, command, expected) -> None:
    """A newline inside the quotes of a ``-c`` payload is kept for the payload's
    shell, so a heredoc there takes its body after its line, as at the top
    level: a script named only in the body is unresolved, and a command after
    the terminator is walked as a command. This shell expands a ``$`` it did
    not leave quoted before the child reads the payload, a name it never
    bound to nothing, so ``bash -c "f=run.py; python3 $f"`` runs ``python3``
    with no script.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('f=run.py; let f=1; python3 "$f"', 0.75),
        ('f=run.py; let "f=1"; python3 "$f"', 0.75),
        ('f=run.py; : $((f=1)); python3 "$f"', 0.75),
        ('f=run.py; x=$((f+=1)); python3 "$f"', 0.75),
        ('f=run.py; : $[f=1]; python3 "$f"', 0.75),
        ('f=run.py; eval "let f=1"; python3 "$f"', 0.75),
        ('f=run.py; let g=1; python3 "$f"', 1.0),
        ('f=run.py; : $((g=1)); python3 "$f"', 1.0),
        ("f=run.py; echo '$((f=1))'; python3 \"$f\"", 1.0),
        ('f=run.py; (: $((f=1))); python3 "$f"', 1.0),
        ('f=run.py; echo $((f=1)) | cat; python3 "$f"', 1.0),
        ('f=run.py; : $( (f=1) ); python3 "$f"', 1.0),
        ('f=run.py; python3 "$f" $((f=1))', 1.0),
        ('f=other.py; let f=1; python3 "$f"', 0.0),
    ],
)
def test_arithmetic_outside_an_arithmetic_command_binds_too(check, command, expected) -> None:
    """``let``, ``$((...))`` and ``$[...]`` assign as ``((...))`` does: a
    variable that held the script ends as a number, or unchanged after an
    error, so a later read of it is unresolved. Quoted text, a subshell, a
    pipeline stage and a command substitution keep this shell's value, and a
    word expanded before the assignment reads the value it had.
    """
    assert check(_bash(command, "done"), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("LANG=C for g in; do python3 run.py; done", 0.75),
        ("f=x for g in; do echo; done; python3 run.py", 0.75),
        ("A=1 for g in x; do python3 run.py; done", 0.75),
        ("A=1 B=2 for g in x; do python3 run.py; done", 0.75),
        ("A=1 select g in x; do python3 run.py; break; done", 0.75),
        ("A=1 while true; do python3 run.py; break; done", 0.75),
        ("A=1 until false; do python3 run.py; break; done", 0.75),
        ("A=1 if true; then python3 run.py; fi", 0.75),
        ("A=1 case x in x) python3 run.py;; esac", 0.75),
        ("A=1 { python3 run.py; }", 0.75),
        ("A=1 ( python3 run.py )", 0.75),
        ("A=1 [[ -n x ]] && python3 run.py", 0.75),
        ("A=1 (( 1 )) && python3 run.py", 0.75),
        ("python3 run.py; A=1 for g in x; do :; done", 0.75),
        ("sh -c 'A=1 for g in x; do python3 run.py; done'", 0.75),
        ("A=1 for x; python3 run.py", 0.75),
        ("python3 run.py\nA=1 for g in x; do :; done", 1.0),
        (r'f=run.py\; for "f"', 0.0),
        ("A=1 for g in x; do cat run.py; done", 0.0),
        ("A=1 if true; then cat run.py; fi", 0.0),
        ("A=(x y); python3 run.py", 1.0),
        ("A='for' python3 run.py", 1.0),
        ("for g in x; do A=1 python3 run.py; done", 1.0),
        ("if true; then A=1 python3 run.py; fi", 1.0),
        ("A=1; for g in x; do python3 run.py; done", 1.0),
        ('f=run.py; for f in; do :; done; python3 "$f"', 1.0),
    ],
)
def test_a_compound_word_after_an_assignment_is_refused_text(check, command, expected) -> None:
    """A compound's opening word is reserved only where a command starts.
    After an assignment, bash, bash --posix, dash, zsh, ksh, mksh and busybox
    ash all refuse it with the compound's own syntax, so the line runs
    nothing: an invocation on it is partial, never full credit, and never an
    exception. A command completed on a line before it has already run (see
    the next test). ``A=(`` is an array and ``A='for'`` a value, and a loop
    word where a command starts is a loop.
    """
    assert check(_bash(command), EXPECTED_SCRIPT)["score"] == expected


_REFUSED = "A=1 for g in x; do :; done"


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("python3 run.py\nA=1 for g in x; do :; done", 1.0),
        ("python3 run.py # comment\nA=1 for g in x; do :; done", 1.0),
        ("# first\n\npython3 run.py\nA=1 ( : )", 1.0),
        ("python3 run.py;\nA=1 if true; then :; fi", 1.0),
        ("python3 run.py | cat\nA=1 for g in x; do :; done", 1.0),
        ("python3 run.py > out 2>&1 &\nA=1 { :; }", 1.0),
        ("python3 run.py <<EOF\nA=1 for g in x; do :; done\nEOF\nA=1 ( : )", 1.0),
        ("python3 run.py\necho a\nA=1 for g in x; do :; done", 1.0),
        ("python3 run.py\ncat <<< x", 1.0),
        ("python3 run.py; A=1 for g in x; do :; done", 0.75),
        ("if true; then\npython3 run.py\nA=1 for g in x; do :; done\nfi", 0.75),
        ("python3 run.py &&\nA=1 for g in x; do :; done", 0.75),
        ("python3 run.py |\nA=1 ( : )", 0.75),
        ("python3 run.py \\\n; A=1 for g in x; do :; done", 0.75),
        ("A=1 for g in x; do :; done\npython3 run.py", 0.75),
        ('export f=run.py\nA=1 for g in x; do :; done\npython3 "$f"', 0.75),
        ("cat run.py\nA=1 for g in x; do :; done\npython3 run.py", 0.75),
        ("cat run.py\nA=1 for g in x; do :; done", 0.0),
    ],
)
def test_a_command_completed_before_a_refused_line_keeps_its_credit(check, command, expected) -> None:
    """bash, bash --posix, dash, ksh, mksh and busybox ash parse and run a
    text one complete command at a time, so the first command, completed on
    a line before the one the shell refuses, has already run (measured). It
    keeps its credit when it is a plain pipeline: blank lines and comments
    may come before it, a heredoc body may follow it, and what comes after
    it does not matter. A line joined to the refused one by ``&&``, ``|`` or
    a trailing backslash, and a compound spanning lines around it, run
    nothing, and nothing after it runs.
    """
    assert check(_bash(command), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "command",
    [
        "echo a\npython3 run.py\nA=1 if true; then :; fi",
        "for x in 1; do\npython3 run.py\ndone\nA=1 for g in x; do :; done",
        "{ python3 run.py\n}\nA=1 ( : )",
        "f() {\npython3 run.py\n}\nf\nA=1 for g in x; do :; done",
        "x=$(echo a\n)\npython3 run.py\nA=1 ( : )",
        "a=(1\n2)\npython3 run.py\nA=1 { :; }",
        "case x in\nx) :;;\nesac\npython3 run.py\nA=1 for g in x; do :; done",
        "echo 'a\nb'; python3 run.py\nA=1 for g in x; do :; done",
        "cat <<EOF\nx\nEOF\npython3 run.py\nA=1 for g in x; do :; done",
        'export f=run.py\npython3 "$f"\nA=1 for g in x; do :; done',
        "exit\npython3 run.py\nA=1 for g in x; do :; done",
        "exec true\npython3 run.py\nA=1 for g in x; do :; done",
        "set -n\npython3 run.py\nA=1 for g in x; do :; done",
        "kill $$\npython3 run.py\nA=1 for g in x; do :; done",
        "set -e; false\npython3 run.py\nA=1 for g in x; do :; done",
        "false && python3 run.py\nA=1 for g in x; do :; done",
        "true || python3 run.py\nA=1 for g in x; do :; done",
        "PATH= python3 run.py\nA=1 for g in x; do :; done",
        "f() {\npython3 run.py\n}\nA=1 for g in x; do :; done",
        "f() {\npython3 run.py\n}\nf; A=1 for g in x; do :; done",
        "bash -c 'python3 run.py'\nA=1 for g in x; do :; done",
        "bash -c 'python3 run.py >'\nA=1 for g in x; do :; done",
        "sh -c 'python3 run.py; fi'\nA=1 for g in x; do :; done",
    ],
)
def test_only_a_plain_first_command_is_credited_before_a_refused_line(check, command) -> None:
    """The walk does not model every command that stops the ones after it
    (``exit``, ``exec``, ``set -n``, ``kill $$``, a failed ``set -e`` step,
    an emptied ``PATH``) or that decides whether they run (``&&``, ``||``,
    a function called or not), and a ``-c`` payload's own text may be refused
    for a reason the walk does not detect. So when a later line is refused,
    only the first command is credited, and only a plain pipeline with no
    payload: nothing ran before it that could have stopped it. Everything
    else stays partial, as at 4594693, whether or not it ran. Measured: the
    first ten and the plain payload run in bash, bash --posix, ksh and mksh,
    and in dash and busybox ash but for the array; the others run nothing.
    """
    assert check(_bash(command), EXPECTED_SCRIPT)["score"] == 0.75


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # The internal review's two cases, and the variants it compared.
        (f"python3 run.py >\n{_REFUSED}", 0.75),
        (f"python3 run.py\r\n{_REFUSED}", 0.75),
        (f"python3 run.py <\n{_REFUSED}", 0.75),
        (f"python3 run.py 2>\n{_REFUSED}", 0.75),
        (f"python3 run.py |&\r\n{_REFUSED}", 0.75),
        (f"time -p python3 run.py\r\n{_REFUSED}", 0.75),
        (f"f() {{\npython3 run.py\n}}\nf\r\n{_REFUSED}", 0.75),
        (f"f() {{\r\npython3 run.py\r\n}}\r\nf\r\n{_REFUSED}", 0.75),
        (f"python3 run.py; function f\r\n{_REFUSED}", 0.75),
        (f"python3 run.py; f()\r\n{_REFUSED}", 0.75),
        (f"python3 run.py;\r\n{_REFUSED}", 0.75),
        # Every redirection operator without its operand refuses its line.
        *(
            (f"python3 run.py {op}\n{_REFUSED}", 0.75)
            for op in (">>", ">|", "&>", "&>>", "<>", ">&", "<&", "2>&", "1>")
        ),
        (f"python3 run.py >;\n{_REFUSED}", 0.75),
        (f"python3 run.py > # out\n{_REFUSED}", 0.75),
        (f"python3 run.py >#out\n{_REFUSED}", 0.75),
        (f"python3 run.py 2>#out\n{_REFUSED}", 0.75),
        (f"python3 run.py >\nout\n{_REFUSED}", 0.75),
        (f"python3 run.py <<<\necho a\n{_REFUSED}", 0.75),
        (f"python3 run.py 2>&1\r\n{_REFUSED}", 0.75),
        # Other lines every modelled shell refuses on their own.
        (f"python3 run.py; ;\n{_REFUSED}", 0.75),
        (f"python3 run.py ;;\n{_REFUSED}", 0.75),
        (f"; python3 run.py\n{_REFUSED}", 0.75),
        (f"python3 run.py & ;\n{_REFUSED}", 0.75),
        (f"python3 run.py; then\n{_REFUSED}", 0.75),
        (f"if python3 run.py; fi\n{_REFUSED}", 0.75),
        (f"python3 run.py (x)\n{_REFUSED}", 0.75),
        (f"python3 run.py\\\necho a\n{_REFUSED}", 0.75),
        (f'echo "$(echo "a\npython3 run.py\nb")"\n{_REFUSED}', 0.75),
        # The same commands complete: they run, and keep their credit.
        (f"python3 run.py '>'\n{_REFUSED}", 1.0),
        (f"python3 run.py \\>\n{_REFUSED}", 1.0),
        (f"python3 run.py 2>&1\n{_REFUSED}", 1.0),
        (f"python3 run.py > out\n{_REFUSED}", 1.0),
        (f"python3 run.py >&-\n{_REFUSED}", 1.0),
        (f"python3 run.py >out #c\n{_REFUSED}", 1.0),
        (f"python3 run.py\n{_REFUSED}\r", 1.0),
    ],
)
def test_a_line_refused_on_its_own_is_not_credited_before_a_refused_line(check, command, expected) -> None:
    """A redirection operator without its operand (at the line's end, before
    ``;``, or before a comment, which a ``#`` starts after an operator too),
    an empty command, a reserved word out of place or a parenthesis after a
    word is refused on its own line by bash, bash --posix, dash, mksh and
    busybox ash, which then run nothing (measured); ksh accepts an empty
    command and ``2>#out``. A carriage return before the newline is part of
    the word before it: ``python3 run.py\\r`` opens ``run.py\\r``, and an
    escaped newline joins the next line to the word. None of these is a
    plain pipeline, so none is credited and the text stays partial, as at
    4594693, including the forms some shells run (``python3 run.py;\\r``).
    A CR on the refused line itself does not touch the command before it.
    """
    assert check(_bash(command), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("bash -c 'python3 run.py\nA=1 for g in x; do :; done'", 0.75),
        ("ksh -c 'python3 run.py\nA=1 ( : )'", 0.75),
        ("dash -c 'python3 run.py\ncat <<< x'", 0.75),
        ("sh -c 'python3 run.py\ncat <<< x'", 0.75),
        ("zsh -c 'python3 run.py\nA=1 for g in x; do :; done'", 0.75),
        ("dash -c 'cat <<< x\npython3 run.py'", 0.75),
        ("dash -c 'python3 run.py; cat <<< x'", 0.75),
        ("bash -c 'python3 run.py >\nA=1 for g in x; do :; done'", 0.75),
        ("bash -c 'python3 run.py\r\nA=1 for g in x; do :; done'", 0.75),
        ("bash -c 'echo a\npython3 run.py\nA=1 for g in x; do :; done'", 0.75),
    ],
)
def test_a_payload_with_a_line_its_shell_refuses_stays_partial(check, command, expected) -> None:
    """A ``-c`` payload is refused by its own shell: a here-string under dash
    and busybox ash, a compound's opening word after an assignment under
    each. bash, dash, ksh, mksh and ash run the commands before the refused
    line and zsh runs none, but the payload is read from what the tokenizer
    left of it, not from the text its shell reads (a carriage return there
    becomes a line's end), so no command in it is credited on its own. The
    payload stays partial, as at 4594693.
    """
    assert check(_bash(command), EXPECTED_SCRIPT)["score"] == expected


def _native(command: str, shell: object = None) -> list[dict[str, object]]:
    arguments: dict[str, object] = {"cmd": command}
    if shell is not None:
        arguments["shell"] = shell
    return [{"action": "exec_command", "action_input": arguments, "observation": "done"}]


_LAST_STAGE_BINDING = 'printf "" | for f in run.py; do :; done; python3 "$f"'


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "shell", "expected"),
    [
        (_LAST_STAGE_BINDING, "/bin/zsh", 1.0),
        (_LAST_STAGE_BINDING, "zsh", 1.0),
        (_LAST_STAGE_BINDING, "/usr/local/bin/zsh5.9", 1.0),
        (_LAST_STAGE_BINDING, "/usr/bin/ksh", 1.0),
        (_LAST_STAGE_BINDING, "/bin/bash", 0.0),
        (_LAST_STAGE_BINDING, "C:\\msys64\\usr\\bin\\bash.exe", 0.0),
        (_LAST_STAGE_BINDING, "/bin/mksh", 0.0),
        (_LAST_STAGE_BINDING, "/bin/sh", 0.0),
        (_LAST_STAGE_BINDING, "", 0.0),
        (_LAST_STAGE_BINDING, None, 0.0),
        ('printf "" | { f=run.py; }; python3 "$f"', "/bin/zsh", 1.0),
        ('printf "" | { f=run.py; }; python3 "$f"', "/bin/bash", 0.0),
        ("python3 run.py\nA=1 for g in x; do :; done", "/bin/zsh", 0.75),
        ("python3 run.py\nA=1 for g in x; do :; done", "/bin/bash", 1.0),
        ("python3 run.py\nA=1 for g in x; do :; done", "/bin/dash", 1.0),
        ("cat <<< x; python3 run.py", "/bin/dash", 0.75),
        ("cat <<< x; python3 run.py", "/bin/bash", 1.0),
        ("python3 run.py\ncat <<< x", "/bin/dash", 1.0),
        ("python3 run.py && true\ncat <<< x", "/bin/dash", 0.75),
        ("python3 run.py >\nA=1 for g in x; do :; done", "/bin/bash", 0.75),
        ("python3 run.py >\nA=1 for g in x; do :; done", "/bin/zsh", 0.75),
        ("python3 run.py\r\nA=1 for g in x; do :; done", "/bin/bash", 0.75),
        ("python3 run.py\r\nA=1 for g in x; do :; done", "/bin/zsh", 0.75),
        ('declare f=run.py; python3 "$f"', "/bin/dash", 0.0),
        ('typeset f=run.py; python3 "$f"', "/bin/mksh", 1.0),
        ('f=run.py; ((f=1)); python3 "$f"', "/bin/dash", 1.0),
        ('f=run.py; ((f=1)); python3 "$f"', "/bin/zsh", 0.75),
    ],
)
def test_a_native_tool_call_is_read_under_the_shell_it_names(check, command, shell, expected) -> None:
    """A native call can name the shell that ran its command (Codex's
    ``exec_command`` keeps ``shell`` in its arguments). The rules that differ
    between shells follow it: zsh and ksh keep a binding made in a
    pipeline's last stage, dash has no here-string, no ``declare`` and no
    arithmetic command, and zsh parses a whole text before running it. A
    call that names no shell is read as bash, as before; ``sh`` is walked
    under each shell it may be, and disagreement is unresolved.
    """
    assert check(_native(command, shell), EXPECTED_SCRIPT)["score"] == expected


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("command", "shell", "expected"),
    [
        (_LAST_STAGE_BINDING, "/usr/bin/fish", 0.75),
        ("python3 run.py", "/usr/bin/fish", 0.75),
        ("cat run.py", "pwsh.exe", 0.75),
        ("echo done", "/usr/bin/fish", 0.0),
        (_LAST_STAGE_BINDING, ["/bin/zsh"], 0.75),
    ],
)
def test_a_shell_this_walk_does_not_model_leaves_the_command_unresolved(check, command, shell, expected) -> None:
    """A shell outside the modelled ones reads the text by rules this walk does
    not know, so no definite score is given: a command that names the script
    is unresolved, and one that never names it is not evidence about it.
    """
    result = check(_native(command, shell), EXPECTED_SCRIPT)
    assert result["score"] == expected


_DAMAGE_BASES = [
    "python3 run.py",
    'f=run.py; for f in; do :; done; python3 "$f"',
    "for f in run.py; do python3 $f; done | cat",
    "LANG=C python3 -u run.py 2>&1 > out.txt",
    "if [ -f run.py ]; then python3 run.py; elif true; then :; else cat run.py; fi",
    "case x in x) python3 run.py;; *) cat run.py;; esac",
    "while read -r l; do python3 run.py; done < in.txt",
    "(cd x && python3 run.py); { cat run.py; }",
    "cat <<EOF\npython3 run.py\nEOF\npython3 run.py",
    "python3 <<A <<B run.py\nx\nA\ny\nB",
    'cat <<< "a; b"; python3 run.py',
    "bash -c 'f=run.py; python3 \"$f\"'",
    "export f=run.py; env -u f bash -c 'python3 \"$f\"'",
    'readonly f=run.py; declare -n g=f; python3 "$g"',
    'f=run.py; let f=1; : $((f=2)); python3 "$f"',
    'f=run.py; eval "g=$f"; python3 "$g"',
    '(((for f in run.py; do :; done; python3 "$f")) | cat)',
    "timeout -- 30 env --help python3 run.py && xargs -r python3 run.py",
    "sh -c 'cat <<< x; python3 run.py'",
    "A=1 for g in x; do python3 run.py; done",
    "echo a # it's\npython3 run.py\nA=1 for g in x; do :; done",
    "cat <<EOF\nx\nEOF\ncase x in\nx) :;;\nesac\npython3 run.py\nA=1 ( : )",
]


def _damaged(command: str) -> list[str]:
    """Every prefix and every single-character deletion of a command: the
    shapes a cut-off or mistyped tool call takes."""
    return sorted(
        {command[:end] for end in range(len(command))}
        | {command[:index] + command[index + 1 :] for index in range(len(command))}
    )


@pytest.mark.parametrize("check", IMPLEMENTATIONS)
@pytest.mark.parametrize("command", _DAMAGE_BASES)
def test_a_damaged_command_is_scored_never_raised(check, command) -> None:
    """A malformed command gets a score like any other. Deleting one ``;``
    from the second base gives ``f=run.py for f in; do :; ...``, which raised
    ``IndexError`` before the walk read a loop word after an assignment as
    the ordinary word it is.
    """
    for damaged in _damaged(command):
        result = check(_bash(damaged), EXPECTED_SCRIPT)
        assert result["score"] in {0.0, 0.75, 1.0}, damaged
