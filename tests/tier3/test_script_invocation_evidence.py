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
