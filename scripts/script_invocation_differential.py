# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differential harness for Tier 3 script-execution credit.

``check_script_execution`` decides from a command's text whether the expected
script was invoked. This harness checks that decision against what the shell
actually does: every command is run for real in a throwaway directory against
fixtures that write a marker file when they execute, and the marker, not an
expectation written by hand, is the ground truth.

Three outcomes matter:

* a **false positive** is the checker scoring 1.0 for a command that ran
  nothing, which is the defect this whole check exists to prevent;
* a **false negative** is the checker scoring 0.0 for a command that did run
  the script, which is a claim the text did not support either;
* a **partial on a real run** is the checker scoring 0.75 for a command that
  ran. That is not a defect. It is the checker reporting that it could not
  resolve the command, which is the correct answer for a shape it does not
  model, such as a path arriving through standard input;
* a **partial without a reference** is the checker scoring 0.75 for a command
  whose text never names the expected script. Parsing uncertainty over an
  unrelated command is not evidence about the script, so this is a defect: the
  checker has credited a command it knows nothing about.

Host and bundled Harbor verifier are compared on every command as well, because
the two copies must not drift.

Run it with ``python scripts/script_invocation_differential.py``. Add
``--fuzz N`` to also generate and execute N random commands built from shell
pieces, and ``--seed`` to reproduce a particular generation. Commands that this
machine cannot run (a tool that is not installed, a shell that refuses to parse)
are reported separately and never counted as either kind of defect.

Add ``--compose N`` to execute N commands that nest binding scopes (groups,
brace groups, compound commands and pipeline stages, two or three deep, under
bash, zsh and ksh). Commands a native tool call runs under the shell it names
are executed by that shell (bash, zsh, ksh, mksh, dash and sh) and scored with
the name passed through, and lines around one the shell refuses are scored
even though it reports a syntax error. Add ``--baseline REF`` to also score every command with the checker as it
stood at that git ref, and list each command whose score moved, against the
marker. A fix that lowers the score of a command that ran, or raises it for
one that did not, is a regression whatever else it repaired, so those are
listed first for review before a revision is pushed.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from skillevaluator.tier3.eval_core import checks as host_checks

TEMPLATE_PATH = REPO_ROOT / "src/skillevaluator/tier3/harbor/templates/eval.py"


def _load_template():
    spec = importlib.util.spec_from_file_location("harbor_eval_differential", TEMPLATE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEMPLATE = _load_template()

# A command that returns before its script does: ``setsid -f`` and ``nohup``
# fork and detach, and ``&`` backgrounds. Their marker may land after the
# shell exits, so the harness waits for it before reading "did not run".
_DETACHED_RE = re.compile(r"setsid\s+-f|\bnohup\b|&\s*(?:$|;|\)|\|\|)")
PY = "python3"
SCRIPT = "skills/demo/run.py"
O = "skills/demo/other.py"
SHELL_SCRIPT = "skills/demo/run.sh"


def _versioned_name(interpreter: str, version: str) -> str:
    """The interpreter under a versioned name, as distributions install it.

    Debian ships ``perl5.38.2`` beside ``perl`` and every platform ships
    ``python3.12`` beside ``python3``. The longest installed form is used, so
    the command executes here; where none is installed the command is
    reported as not runnable rather than scored. An interpreter outside the
    default search path is named by its absolute path, because ``env -i``
    empties PATH and would otherwise fail to find it for a reason the command
    text does not carry.
    """
    parts = version.split(".")
    candidates = [interpreter + ".".join(parts[:count]) for count in range(len(parts), 0, -1)]
    for name in candidates:
        found = shutil.which(name)
        if found:
            return name if Path(found).parent in {Path("/bin"), Path("/usr/bin")} else found
    return candidates[0]


def _perl_version() -> str:
    try:
        completed = subprocess.run(["perl", "-e", 'printf "%vd", $^V'], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "5.38.2"
    return completed.stdout.strip() or "5.38.2"


VERSIONED_PY = _versioned_name("python", f"{sys.version_info.major}.{sys.version_info.minor}")
VERSIONED_PERL = _versioned_name("perl", _perl_version())


def _marker(name: str) -> str:
    return f"#!/usr/bin/env python3\nimport pathlib; pathlib.Path('MARKER.{name}').write_text('ran')\nprint('{name}')\n"


FIXTURES = {
    "skills/demo/run.py": _marker("run.py"),
    "skills/demo/other.py": _marker("other.py"),
    "skills/demo/run.py.bak": _marker("run.py.bak"),
    "skills/demo/rerun.py": _marker("rerun.py"),
    "skills/demo/run.sh": "#!/bin/sh\necho ran > MARKER.run.sh\necho sh\n",
    "skills/demo/run.pl": "open(my $f,'>','MARKER.run.pl'); print $f 'ran'; print \"pl\\n\";\n",
    "skills/demo/run.rb": "File.write('MARKER.run.rb','ran'); puts 'rb'\n",
    "skills/demo/run.js": "require('fs').writeFileSync('MARKER.run.js','ran'); console.log('js')\n",
    "run.py": _marker("run.py"),
    "other.py": _marker("other.py"),
    "another.py": _marker("another.py"),
    "skills/demo/another.py": _marker("another.py"),
    # a name carrying the private-use character the checker uses as a mark,
    # which must stay a different file from run.py
    "skills/demo/\ue000run.py": _marker("\ue000run.py"),
    # ``python3 2 > out.txt run.py`` runs a script named 2: these exist so that
    # command succeeds, as the shell reads it, without touching run.py.
    "0": "print('zero')\n",
    "1": "print('one')\n",
    "2": "print('two')\n",
    "skills/demo/0": "print('zero')\n",
    "skills/demo/1": "print('one')\n",
    "skills/demo/2": "print('two')\n",
    "notes.txt": "notes\n",
    # a loop reading `< notes.txt` must find it after `cd skills/demo` too,
    # otherwise its body never runs for a reason the command text does not carry
    "skills/demo/notes.txt": "notes\n",
    "lib.js": "module.exports={}\n",
    "lib.rb": "",
}

NON_EXECUTING_VERBS = [
    "cat",
    "head -2",
    "tail -2",
    "grep -n ran",
    "wc -l",
    "ls -l",
    "stat",
    "file",
    "md5sum",
    "touch",
    "echo",
    "printf",
]
# Well-formed verbs that need a second argument, kept apart so the generator
# does not build a command that fails for want of one.
TWO_ARGUMENT_VERBS = ["cp -n", "diff"]
INTERPRETER_FORMS = [
    "",
    "-u ",
    "-B ",
    "-OO ",
    "-I ",
    "-q ",
    "-Wignore ",
    "-W ignore ",
    "-Xdev ",
    "-X dev ",
    "-- ",
    "--help ",
    "-V ",
    "-c 'print(1)' ",
    "-m json.tool ",
    "-c 'import runpy; runpy.run_path(\"skills/demo/run.py\")' ",
]
WRAPPER_FORMS = [
    "",
    "timeout 20 ",
    "timeout --help ",
    "timeout -- 20 ",
    "timeout -s TERM -- 20 ",
    "nohup ",
    "nohup -- ",
    "nice -n 3 ",
    "nice -5 ",
    "nice -- ",
    "env ",
    "env -i ",
    "env FOO=1 ",
    "env --help ",
    "env -- ",
    "stdbuf -o0 ",
    "stdbuf --help ",
    "stdbuf -o0 -- ",
    "setsid -f ",
    "setsid --help ",
    "setsid -- ",
    "xargs ",
    "xargs -r ",
    "xargs -p ",
    "xargs --help ",
    "uv run ",
    "uv run -- ",
    "uv run --no-project -- ",
    "uv --help run ",
]
PREFIX_FORMS = [
    "",
    "FOO=1 ",
    "A=1 B=2 ",
    "cd skills/demo && ",
    "(cd skills/demo && ",
    "x=$((1 << 2)); ",
    "echo $((2 << 1)) ; ",
    "cat <<EOF\nnotes\nEOF\n",
    "cat <<-EOF\n\tnotes\n\tEOF\n",
    "cat <<'EOF'\nrun.py\nEOF\n",
    "echo '<<' ; ",
    "cat <<< notes ; ",
    'cat <<< "a; b" ; ',
    "printf '' | ",
    "echo x | ",
]
SUFFIX_FORMS = ["", " > out.txt", " 2>&1", " ; echo done", " || true"]
# Redirections that stand before the script: the shell's, not the interpreter's.
REDIRECT_FORMS = [
    "< /dev/null ",
    "</dev/null ",
    "0< /dev/null ",
    "0</dev/null ",
    "0<notes.txt ",
    "<&0 ",
    "2>/dev/null ",
    "2> /dev/null ",
    "2>&1 ",
    "2>&- ",
    "2>>err.txt ",
    "1>>out.txt ",
    "1> out.txt ",
    "3<>notes.txt ",
]
# A digit standing apart from its operator is an operand: ``python3 2 > out.txt
# run.py`` runs the script named 2 and hands it run.py.
DETACHED_DESCRIPTOR_FORMS = ["2 > out.txt ", "2 >out.txt ", "1 >> out.txt ", "0 < notes.txt "]
# A second loop header for the same variable, after ``for f in run.py; do cat $f; done``.
REBINDING_HEADERS = [
    "for f in {other} {another}; do {body}; done",
    "for f; do {body}; done",
    "for f in; do {body}; done",
    "for f in; do :; done; {body}",
    'for f in ""; do :; done; {body}',
    "for f in {other}; do {body}; done",
    "for f in {script}; do {body}; done",
    "for g in x; do {body}; done",
]
# Control structures around a command, with the command as {body}.
CONTROL_FORMS = [
    "if true; then {body}; fi",
    "if false; then true; else {body}; fi",
    "if false; then true; elif true; then {body}; fi",
    "for i in one; do {body}; done",
    "while read -r _; do {body}; done < notes.txt",
    "until false; do {body}; break; done",
    "case x in x) {body};; esac",
]
# Arguments a shell script receives, which look like the shell's own options.
SCRIPT_ARGUMENT_FORMS = [" -c 'echo done'", " --help", " -- -x", " -n"]
# Commands that never name the expected script, which no walk may credit.
UNRELATED_BODIES = [
    "cd",
    "cd skills",
    "true",
    f"{PY} skills/demo/other.py",
    "timeout --frobnicate 5 python3 skills/demo/other.py",
    "python3 -Q skills/demo/other.py",
    "env -Z python3 skills/demo/other.py",
]


def curated() -> list[tuple[str, str]]:
    """Command shapes chosen deliberately, each paired with its expected script."""
    cases: list[tuple[str, str]] = []

    def add(command: str, script: str = "run.py") -> None:
        cases.append((command, script))

    for form in INTERPRETER_FORMS:
        add(f"{PY} {form}{SCRIPT}")
    for wrapper in WRAPPER_FORMS:
        add(f"{wrapper}{PY} {SCRIPT}")
    for verb in NON_EXECUTING_VERBS:
        add(f"{verb} {SCRIPT}")
        add(f"{verb} {SCRIPT} && {PY} {SCRIPT}")
    for verb in TWO_ARGUMENT_VERBS:
        add(f"{verb} {SCRIPT} copy.py")
        add(f"{verb} {SCRIPT} copy.py && {PY} {SCRIPT}")
    for prefix in PREFIX_FORMS:
        target = "run.py" if prefix.startswith(("cd ", "(cd ")) else SCRIPT
        closing = ")" if prefix.startswith("(") else ""
        add(f"{prefix}{PY} {target}{closing}")
    for suffix in SUFFIX_FORMS:
        add(f"{PY} {SCRIPT}{suffix}")

    # identity: a name that merely contains the expected one is a different file
    add(f"{PY} skills/demo/run.py.bak", "run.py")
    add(f"{PY} skills/demo/rerun.py", "run.py")
    add(f"{PY} skills/demo/other.py skills/demo/run.py", "run.py")

    # other interpreters, and options belonging to a different one
    for interpreter, script in (
        ("perl", "skills/demo/run.pl"),
        ("ruby", "skills/demo/run.rb"),
        ("node", "skills/demo/run.js"),
        ("bash", "skills/demo/run.sh"),
        ("sh", "skills/demo/run.sh"),
    ):
        name = script.rsplit("/", 1)[-1]
        for option in (
            "",
            "-w ",
            "-c ",
            "-e 'x' ",
            "--help ",
            "--version ",
            "--check ",
            "--require ./lib.js ",
            "-I lib ",
            "-Mstrict ",
            "-Wignore ",
        ):
            add(f"{interpreter} {option}{script}", name)

    # nested shells, sourcing and grouping
    add(f"sh -c '{PY} {SCRIPT}'")
    add(f'bash -c "{PY} {SCRIPT}"')
    add(f"sh -c 'cat {SCRIPT}'")
    add(f"bash -xc '{PY} {SCRIPT}'")
    add(f"source {SCRIPT}")
    add(f". {SCRIPT}")
    add(f"{{ {PY} {SCRIPT}; }}; echo done")
    add(f"(cd skills/demo && {PY} run.py); echo done")

    # heredoc bodies and terminators
    add(f"cat <<EOF\nnotes\nEOF\n{PY} {SCRIPT}")
    add(f"cat <<EOF\n{PY} {SCRIPT}\nEOF")
    add(f"cat <<EOF\n{PY} {SCRIPT}\nEOF\ncat out.txt")
    add(f"cat <<A\none\nA\ncat <<B\ntwo\nB\n{PY} {SCRIPT}")

    # inline code, modules, eval and standard input: text this walk does not read
    add(f"""{PY} -c 'import runpy; runpy.run_path("{SCRIPT}")'""")
    add(f"""{PY} -c "exec(open('{SCRIPT}').read())" """.strip())
    add(f"""{PY} -c 'import sys, runpy; runpy.run_path(sys.argv[1])' {SCRIPT}""")
    add(f"{PY} -c 'print(123)' {SCRIPT}")
    add(f"""{PY} -c "print('{SCRIPT}')" """.strip())
    add(f"{PY} -m json.tool {SCRIPT}")
    add(f"{PY} -m runpy {SCRIPT}")
    add(f"eval '{PY} {SCRIPT}'")
    add("eval 'echo hello'")
    add(f"{PY} <{SCRIPT}")
    add(f"{PY} < {SCRIPT}")
    add(f"wc -l <{SCRIPT}")
    add(f"{PY} - {SCRIPT}")
    add(f"bash -s {SCRIPT}")
    add(f"exec {PY} {SCRIPT}")
    add(f"command {PY} {SCRIPT}")
    # an invocation carried inside a command with no grammar of its own
    add(f"flock /tmp/sied.lock {PY} {SCRIPT}")
    add(f"taskset -c 0 {PY} {SCRIPT}")
    add(f"ionice -c3 {PY} {SCRIPT}")
    add(f"chrt -o 0 {PY} {SCRIPT}")
    add(f"strace -f -o /dev/null {PY} {SCRIPT}")
    add(f"flock /tmp/sied.lock cat {SCRIPT}")
    add(f"strace -f -o /dev/null cat {SCRIPT}")
    # shapes whose behaviour depends on data the command text does not carry
    add(f"{PY} $(echo {SCRIPT})")
    add(f"{PY} `echo {SCRIPT}`")
    add(f"{PY} ${{PWD}}/{SCRIPT}")
    add(f"echo {SCRIPT} | xargs {PY}")
    add(f"xargs -I{{}} {PY} {{}} <<< {SCRIPT}")
    add(f"find skills -name run.py -exec {PY} {{}} \\;")

    # from review of the first revision: commands the shell runs differently
    # from how the walk read them
    for body in UNRELATED_BODIES:
        add(body)
    for argument in SCRIPT_ARGUMENT_FORMS:
        add(f"bash {SHELL_SCRIPT}{argument}", "run.sh")
        add(f"bash -- {SHELL_SCRIPT}{argument}", "run.sh")
    add(f"bash -c 'echo done' {SHELL_SCRIPT}", "run.sh")
    add(f"bash -c 'bash {SHELL_SCRIPT}' x", "run.sh")
    for redirect in REDIRECT_FORMS:
        add(f"{PY} {redirect}{SCRIPT}")
        add(f"{PY} -u {redirect}{SCRIPT}")
        add(f"bash -c {redirect}'bash {SHELL_SCRIPT}'", "run.sh")
        add(f"bash {redirect}-c 'bash {SHELL_SCRIPT}'", "run.sh")
        add(f"bash -c {redirect}'cat {SHELL_SCRIPT}'", "run.sh")
    for redirect in DETACHED_DESCRIPTOR_FORMS:
        add(f"{PY} {redirect}{SCRIPT}")
        add(f"cd skills/demo && {PY} {redirect}run.py")
    add(f"bash -c 2 '{SHELL_SCRIPT}'", "run.sh")
    names = {"script": SCRIPT, "other": "skills/demo/other.py", "another": "skills/demo/another.py"}
    for header in REBINDING_HEADERS:
        add(f"for f in {SCRIPT}; do cat $f; done; " + header.format(body=f"{PY} $f", **names))
        add(f"for f in {SCRIPT}; do cat $f; done; " + header.format(body="cat $f", **names))
    # A pipeline runs its commands in subshells, so a binding made in one
    # does not reach the command after the pipeline.
    add(f"printf '' | for f in {SCRIPT}; do cat $f; done; for g in x; do {PY} $f; done")
    add(f"for f in {SCRIPT}; do cat $f; done | true; {PY} $f")
    add(f"printf '' | for f in {SCRIPT}; do {PY} $f; done")
    add(f"echo x | while read -r l; do for f in {SCRIPT}; do {PY} $f; done; done")
    add(f"if true; then for f in {SCRIPT}; do cat $f; done; fi | true; {PY} $f")
    add(f"for f in {SCRIPT}; do cat $f; done; {PY} $f")
    for control in CONTROL_FORMS:
        add(control.format(body=f"{PY} {SCRIPT}"))
        add(control.format(body=f"cat {SCRIPT}"))
        add(control.format(body=f"timeout 20 {PY} {SCRIPT}"))
    add(f"for f in {SCRIPT}; do {PY} $f; done")
    add(f"for f in {SCRIPT}; do cat $f; done")
    add(f"for f in {SCRIPT} skills/demo/other.py; do {PY} $f; done")
    add(f"for f in {SCRIPT} skills/demo/other.py; do cat $f; done")
    add(f"if {PY} {SCRIPT}; then echo ok; fi")
    add(f"if true; then FOO=1 {PY} {SCRIPT}; fi")
    for interpreter, script in ((VERSIONED_PY, SCRIPT), (VERSIONED_PERL, "skills/demo/run.pl")):
        name = script.rsplit("/", 1)[-1]
        add(f"{interpreter} {script}", name)
        add(f"{interpreter.rsplit('/', 1)[-1]} {script}", name)
        add(f"{interpreter} -c {script}", name)
        add(f"{interpreter} --version {script}", name)
        add(f"env -i {interpreter} {script}", name)

    # redirections written flush against a descriptor and spaced apart
    # tokenize to the same words, so each pair must score the same
    for attached, spaced in (
        (f"0<{SCRIPT}", f"0< {SCRIPT}"),
        (f"<{SCRIPT}", f"< {SCRIPT}"),
        (f"2>err.txt {SCRIPT}", f"2> err.txt {SCRIPT}"),
        (f"2>>err.txt {SCRIPT}", f"2>> err.txt {SCRIPT}"),
        (f"1>out.txt {SCRIPT}", f"1> out.txt {SCRIPT}"),
        (f"0</dev/null {SCRIPT}", f"0< /dev/null {SCRIPT}"),
        (f"2>&1 {SCRIPT}", f"2>&1 {SCRIPT}"),
        (f"2>&- {SCRIPT}", f"2>&- {SCRIPT}"),
        (f"{SCRIPT} 2>err.txt", f"{SCRIPT} 2> err.txt"),
    ):
        add(f"{PY} {attached}")
        add(f"{PY} {spaced}")
    add(f"cat 0<{SCRIPT}")
    add(f"cat 0< {SCRIPT}")

    # a loop with an empty list never runs and never assigns its variable
    add(f'f={SCRIPT}; for f in; do :; done; {PY} "$f"')
    add(f'f={SCRIPT}; for f in; do {PY} "$f"; done')
    add(f"for f in; do {PY} {SCRIPT}; done")
    add(f'f={SCRIPT}; for f in; do :; done; cat "$f"')
    add(f'for f in {SCRIPT}; do :; done; for f in; do :; done; {PY} "$f"')
    add(f'for f in ""; do {PY} {SCRIPT}; done')
    add(f'f={SCRIPT}; printf "" | for f in; do :; done; {PY} "$f"')

    # a binding made in the last stage of a pipeline survives in zsh and not
    # in bash, dash or sh; one made in an earlier stage, or inside ( ), in none
    for shell in ("zsh", "bash", "sh", "dash"):
        add(f'{shell} -c \'printf "" | for f in {SCRIPT}; do cat "$f"; done; {PY} "$f"\'')
        add(f'{shell} -c \'for f in {SCRIPT}; do cat "$f"; done | cat; {PY} "$f"\'')
        add(f'{shell} -c \'printf "" | f={SCRIPT}; {PY} "$f"\'')
        add(f'{shell} -c \'printf "" | if true; then f={SCRIPT}; fi; {PY} "$f"\'')
        add(f'{shell} -c \'printf "" | {{ f={SCRIPT}; }}; {PY} "$f"\'')
        add(f'{shell} -c \'printf "" | (f={SCRIPT}); {PY} "$f"\'')
        add(f'{shell} -c \'printf "" | for f in {SCRIPT}; do {PY} "$f"; done\'')
    add(f'printf "" | for f in {SCRIPT}; do cat "$f"; done; {PY} "$f"')
    add(f'(f={SCRIPT}); {PY} "$f"')
    add(f'(f={SCRIPT}; {PY} "$f")')
    add(f'{{ f={SCRIPT}; }}; {PY} "$f"')
    add(f'bash -c "shopt -s lastpipe; printf \\"\\" | for f in {SCRIPT}; do :; done; {PY} \\"\\$f\\""')
    add(f'bash -O lastpipe -c "printf \\"\\" | for f in {SCRIPT}; do :; done; {PY} \\"\\$f\\""')
    add(f'ksh -c \'printf "" | for f in {SCRIPT}; do :; done; {PY} "$f"\'')

    # a quoted or escaped word is not syntax: a quoted done does not end a
    # loop, a quoted parenthesis opens no subshell, a quoted separator splits
    # no command, and a redirection operand keeps its whole name
    add(f"{PY} 2>1.log {SCRIPT}")
    add(f"{PY} 2> 1.log {SCRIPT}")
    add(f"{PY} 1>2.txt {SCRIPT}")
    add(f"{PY} '0<{SCRIPT}'")
    add(f"for f in; do :; 'done'; {PY} {SCRIPT}; done")
    add(f"for f in; do d'on'e; {PY} {SCRIPT}; done")
    add(f"for f in; do :; d\\\\one; {PY} {SCRIPT}; done")
    add(f"for f in; do 'for'; done; {PY} {SCRIPT}")
    add(f'f={SCRIPT}; for g in; do printf done; done; {PY} "$f"')
    add(f'f={SCRIPT}; for g in; do for f in skills/demo/other.py; do :; done; done; {PY} "$f"')
    add(f'f={SCRIPT}; printf "("; f=skills/demo/other.py; printf ")"; {PY} "$f"')
    add(f'f={SCRIPT}; printf \\\\(; f=skills/demo/other.py; printf \\\\); {PY} "$f"')
    add(f"{PY} '|' {SCRIPT}")
    add(f"'{PY}' {SCRIPT}")
    add(f"printf '(' ; {PY} {SCRIPT}")
    add(f'zsh -c \'emulate sh; printf "" | for f in {SCRIPT}; do :; done; {PY} "$f"\'')

    # positional parameters: none at the top level, given by set, shift or the
    # operands after a -c payload; and an interpreter fed its program by a pipe
    add(f'f={SCRIPT}; for f in; do :; done; for f; do :; done; {PY} "$f"')
    add(f"for f; do {PY} {SCRIPT}; done")
    add(f"set -- {SCRIPT}; for f; do {PY} $f; done")
    add(f"bash -c 'for f; do {PY} $f; done' _ {SCRIPT}")
    add(f"bash -c 'for f; do {PY} $f; done' {SCRIPT}")
    add(f"bash -c '{PY} $1' _ {SCRIPT}")
    add(f"bash -c '{PY} other.py' _ {SCRIPT}")
    add(f"cat {SCRIPT} | {PY}")
    add(f"cat '{SCRIPT}' | {PY}")
    add(f"cat {SCRIPT} | {PY} -")
    add(f"cat {SCRIPT} | {PY} -c 'print(1)'")
    add(f"cat {SCRIPT} | wc -l")
    add(f"cat {SHELL_SCRIPT} | bash", "run.sh")

    # from the generated audit: quoted metacharacter runs, a brace group as a
    # pipeline stage, a group inside a pipeline, an empty loop inside a group,
    # and a file whose name carries the mark character
    add(f"printf '%s' ';|' {PY} {SCRIPT}")
    add(f"printf '%s' \\\\;\\\\| {PY} {SCRIPT}")
    add(f"{PY} ';;' {SCRIPT}")
    add(f"{PY} '|&' {SCRIPT}")
    add(f"{PY} '2>' {SCRIPT}")
    add(f"{PY} {SCRIPT} ';|'")
    for shell in ("bash", "zsh"):
        add(f"{shell} -c 'f={O}; {{ f={SCRIPT}; :; }} | cat; {PY} \"$f\"'")
        add(f'{shell} -c \'f={O}; printf "" | {{ :; f={SCRIPT}; }} | cat; {PY} "$f"\'')
        add(f'{shell} -c \'f={O}; printf "" | {{ :; f={SCRIPT}; }}; {PY} "$f"\'')
        add(f'{shell} -c \'f={O}; printf "" | (f={SCRIPT}; {PY} "$f"); {PY} "$f"\'')
        add(f'{shell} -c \'f={O}; printf "" | (f={SCRIPT}; {PY} "$f") | cat; {PY} "$f"\'')
    add(f'f={SCRIPT}; (f={O}; for g in; do :; done); {PY} "$f"')
    add(f'f={O}; (f={SCRIPT}; for g in; do :; done); {PY} "$f"')
    add(f'f={O}; (f={SCRIPT}; for g in; do for h in; do :; done; done) | cat; {PY} "$f"')
    add(f'f={SCRIPT}; (f={O}; for g in; do :; done) | cat; (:); {PY} "$f"')
    add(f'printf "" | (for f in {SCRIPT}; do :; done; {PY} $f)')
    add(f"{PY} 'skills/demo/\ue000run.py'")
    add(f"{PY} skills/demo/\ue000run.py")

    # a pipeline inside a group still isolates its own stages, in both
    # assignment directions, at the last and a middle stage, and nested
    for first, second in ((SCRIPT, O), (O, SCRIPT)):
        add(f'(f={first}; f={second} | cat; {PY} "$f")')
        add(f'(f={first}; printf "" | f={second}; {PY} "$f")')
        add(f'(f={first}; {{ f={second}; }} | cat; {PY} "$f")')
        add(f'((f={first}; f={second} | cat); {PY} "$f")')
        add(f'f={first}; ((f={second}) | cat; {PY} "$f")')
        add(f'{{ f={first}; f={second} | cat; {PY} "$f"; }}')
        add(f'zsh -c \'(f={first}; printf "" | f={second}; {PY} "$f")\'')
        add(f"zsh -c '(f={first}; f={second} | cat; {PY} \"$f\")'")
        # a group inside a compound stage, a compound stage inside another, and
        # a segment that opens a compound and starts a pipeline inside it
        for shell in ("bash", "zsh"):
            add(f'{shell} -c \'printf "" | {{ f={first}; (f={second}); {PY} "$f"; }}\'')
            add(f'{shell} -c \'printf "" | for g in 1; do f={first}; (f={second}); {PY} "$f"; done\'')
            add(f"{shell} -c 'if true; then f={first}; (f={second}); {PY} \"$f\"; fi | cat'")
            add(f"{shell} -c 'f={first}; {{ {{ f={second}; }} | cat; }}; {PY} \"$f\"'")
            add(f"{shell} -c 'f={first}; {{ if true; then f={second}; fi | cat; }}; {PY} \"$f\"'")
            add(f'{shell} -c \'f={first}; {{ printf "" | f={second}; }}; {PY} "$f"\'')
        # an arithmetic command, and the nested-subshell reading of ((
        add(f'f={first}; ((f={second})); {PY} "$f"')
        add(f"ksh -c 'f={first}; ((f={second})); {PY} \"$f\"'")
        add(f'f={first}; ((f={second}; g=1); {PY} "$f")')
        # (( closed by )) is arithmetic in bash, zsh, ksh and mksh, whatever is
        # inside; closed apart, or under dash, it is nested subshells
        for shell in ("", "zsh", "ksh", "mksh", "dash", "sh"):
            body = f'f={first}; ((f={second}; {PY} "$f")); {PY} "$f"'
            add(f"{shell} -c '{body}'" if shell else body)
        add(f'f={first}; ((f={second}; {PY} "$f") ); {PY} "$f"')
        add(f'zsh -c \'f={first}; ((printf "" | for g in 1; do f={second}; {PY} "$f"; done)); {PY} "$f"\'')
        # three opening parentheses: ( (( in bash, zsh and mksh, subshells in ksh and dash
        for shell in ("", "zsh", "ksh", "mksh", "dash"):
            body = f'f={first}; (((for f in {second}; do :; done; {PY} "$f")) | cat); {PY} "$f"'
            add(f"{shell} -c '{body}'" if shell else body)

    # from review of revision 16: builtins that bind a variable, in each shell
    for declaration in BINDING_BUILTIN_FORMS:
        add(f'{declaration} f={SCRIPT}; {PY} "$f"')
        add(f'{declaration} f={SCRIPT}; cat "$f"')
        for shell in ("dash", "zsh", "ksh", "mksh", "sh"):
            add(f"{shell} -c '{declaration} f={SCRIPT}; {PY} \"$f\"'")
    add(f'f={SCRIPT}; export f; {PY} "$f"')
    add(f'f={SCRIPT}; unset f; {PY} "$f"')
    add(f'f={SCRIPT}; unset -f f; {PY} "$f"')
    add(f'readonly f={SCRIPT}; f={O}; {PY} "$f"')
    add(f'f={O}; printf -v f {SCRIPT}; {PY} "$f"')
    add(f'f={O}; echo {SCRIPT} | {{ read f; {PY} "$f"; }}')
    # an assignment before a command is that command's environment only, and
    # outlives a special builtin in the POSIX shells alone
    for form in PREFIX_BINDING_FORMS:
        text = form.format(v=SCRIPT, py=PY)
        add(text)
        for shell in ("bash", "dash", "zsh", "ksh", "mksh", "sh"):
            add(f"{shell} -c '{text}'")
    # what a -c payload's shell inherits, and a $ quoted or escaped from this shell
    for launcher in (
        f"export f={SCRIPT}; ",
        f"f={SCRIPT}; ",
        f"f={SCRIPT} ",
        f"env f={SCRIPT} ",
        f"set -a; f={SCRIPT}; ",
        f"set -o allexport; f={SCRIPT}; set +a; ",
        f"f={SCRIPT}; set -a; ",
        f"set -a; set +a; f={SCRIPT}; ",
    ):
        add(f"{launcher}bash -c '{PY} \"$f\"'")
    add(f'f={SCRIPT}; bash -c "{PY} $f"')
    add(f'f={SCRIPT}; bash -c "{PY} \\$f"')
    # every heredoc declared on a line takes its body after the line, in order
    for form in MULTI_HEREDOC_FORMS:
        add(form.format(py=PY, script=SCRIPT, other=O))

    # from review of revision 17: env removes names before the child starts
    for launcher in ENV_REMOVAL_FORMS:
        add(f"export f={SCRIPT}; {launcher} bash -c '{PY} \"$f\"'")
    add(f"export f={O}; env -i f={SCRIPT} bash -c '{PY} \"$f\"'")
    add(f"export f={SCRIPT}; env -u g bash -c '{PY} \"$f\"'")
    add(f"export f={SCRIPT}; env -u f f={SCRIPT} bash -c '{PY} \"$f\"'")
    add(f'export f={SCRIPT}; bash -c \'env -u f bash -c "{PY} \\"\\$f\\""\'')
    add(f"f={SCRIPT} env -i bash -c '{PY} \"$f\"'")
    # a declaration copies the value when it runs, as a plain assignment does
    for declaration in ("export ", "readonly ", "declare ", "typeset ", ""):
        add(f'f={O}; {declaration}g=$f; f={SCRIPT}; {PY} "$g"')
        add(f'f={SCRIPT}; {declaration}g=$f; f={O}; {PY} "$g"')
    add(f'f={SCRIPT}; export f={O} g=$f; {PY} "$g"')
    add(f"ksh -c 'f={O}; typeset f={SCRIPT} g=$f; {PY} \"$g\"'")
    add(f"f={O}; export g=$f; f={SCRIPT}; bash -c '{PY} \"$g\"'")
    # the export attribute removed, however the builtin is reached
    for removal in EXPORT_REMOVAL_FORMS:
        add(f"export f={SCRIPT}; {removal}; bash -c '{PY} \"$f\"'")
        add(f'export f={SCRIPT}; {removal}; {PY} "$f"')
    # more heredocs on a line than the shell accepts: the rest is data
    for count in (9, 16, 17):
        add(_heredocs(count, "cat", body=f"{PY} {SCRIPT}"))
        add(_heredocs(count, PY, SCRIPT))
        add(_heredocs(count, "cat") + f"{PY} {SCRIPT}")
    for shell, count in (("mksh", 10), ("mksh", 11), ("dash", 17)):
        add(f"{shell} -c '{_heredocs(count, PY, SCRIPT)}'")
    # eval, inline code and a shell reading data may expand a quoted variable
    for form in EVAL_FORMS:
        add(form.format(v=SCRIPT, o=O, py=PY))
    # an assignment to a read-only name fails, and some shells stop there
    for refusal in ("f={o}", "export f={o}", "f={o} true", "unset f", "f={o} :", "for f in x; do :; done"):
        text = f"readonly f={SCRIPT}; {refusal.format(o=O)}; {PY} {SCRIPT}"
        add(text)
        for shell in ("dash", "zsh", "ksh", "sh"):
            add(f"{shell} -c '{text}'")
    add(f"(readonly f={SCRIPT}; f={O}); {PY} {SCRIPT}")
    add(f"readonly f={SCRIPT}; (f={O}; {PY} {SCRIPT})")
    add(f"readonly f={SCRIPT}; f={O}; env -i f={SCRIPT} bash -c '{PY} \"$f\"'")
    # an eval program held in a variable, and attributes kept with a name
    for code in (f"f={O}", "export -n f", "unset f", "declare +x f", f"export f={O}", "echo hi"):
        add(f"export f={SCRIPT}; code='{code}'; eval \"$code\"; bash -c '{PY} \"$f\"'")
    add(f"export f={SCRIPT}; code='f={O}'; eval '$code'; bash -c '{PY} \"$f\"'")
    add(f"export f={SCRIPT}; code='f={O}'; command eval \"$code\"; bash -c '{PY} \"$f\"'")
    add(f"export f={O}; code='f={SCRIPT}'; eval \"$code\"; bash -c '{PY} \"$f\"'")
    for attribute in ("-i", "-u", "-l", "-a"):
        add(f'declare {attribute} f; f={SCRIPT}; {PY} "$f"')
    add(f"export f={SCRIPT}; declare -a f; f={SCRIPT}; bash -c '{PY} \"$f\"'")
    # a -n name passes what is done to it on to the name it refers to
    for change in (f"f={O}", "unset f", f"export f={O}", f"read f <<< {O}", "((f=1))", "h=x"):
        add(f'g={SCRIPT}; declare -n f=g; {change}; {PY} "$g"')
    add(f'g={O}; declare -n f=g; f={SCRIPT}; {PY} "$g"')
    add(f'g={SCRIPT}; declare -n f=g; declare +n f; f={O}; {PY} "$g"')
    add(f'g={SCRIPT}; declare -n f=g; unset -n f; f={O}; {PY} "$g"')
    add(f'g={SCRIPT}; declare -n f=g; declare -n f=h; f={O}; {PY} "$g"')
    add(f"export g={SCRIPT}; declare -n f=g; export -n f; bash -c '{PY} \"$g\"'")
    add(f'g={SCRIPT}; declare -n f; f=g; f={O}; {PY} "$g"')
    add(f'h={SCRIPT}; declare -n f; f=g; {PY} "$h"')
    for shell in ("ksh", "mksh", "bash"):
        add(f"{shell} -c 'g={SCRIPT}; nameref f=g; f={O}; {PY} \"$g\"'")
    # arithmetic binds: let, $((...)) and $[...]
    for arithmetic in ("let f=1", 'let "f=1"', ": $((f=1))", "x=$((f+=1))", ": $[f=1]", 'eval "let f=1"'):
        add(f'f={SCRIPT}; {arithmetic}; {PY} "$f"')
    for control in ("let g=1", ": $((g=1))", "echo '$((f=1))'", "(: $((f=1)))", "echo $((f=1)) | cat", ": $( (f=1) )"):
        add(f'f={SCRIPT}; {control}; {PY} "$f"')
    add(f'f={SCRIPT}; {PY} "$f" $((f=1))')
    # an assignment or builtin that fails where the shell stops
    for failing in (f"f={O}", f"export f={SCRIPT}", f"for f in {SCRIPT}; do :; done", "f=5", "f=x", f"f={O} true"):
        add(f"declare -i f; {failing}; {PY} {SCRIPT}")
    for shell in ("ksh", "mksh"):
        add(f"{shell} -c 'typeset -i f; typeset f={O}; {PY} {SCRIPT}'")
    for shell in ("dash", "bash", "ksh", "mksh", "zsh"):
        add(f"{shell} -c 'export f=1; export -n f; {PY} {SCRIPT}'")
        add(f"{shell} -c 'f={SCRIPT}; unset -n f; {PY} \"$f\"'")
    add(f"dash -c 'export f=1; command export -n f; {PY} {SCRIPT}'")
    add(f'g={SCRIPT}; declare -n f; f={O}; {PY} "$g"')
    add(f'f={O}; declare -n f; f={SCRIPT}; {PY} "$f"')
    add(f"ksh -c 'f={O}; typeset -n f; {PY} {SCRIPT}'")
    # a here-string the shell cannot parse; a heredoc inside a -c payload
    for shell in ("dash", "sh", "bash", "zsh", "ksh", "mksh"):
        add(f"{shell} -c 'cat <<< x; {PY} {SCRIPT}'")
    add(f"dash -c 'export g={SCRIPT}; read f <<< x; {PY} \"$g\"'")
    add(f"dash -c 'export g={SCRIPT}; read f < /dev/null; {PY} \"$g\"'")
    add(f"dash -c \"echo '<<<'; {PY} {SCRIPT}\"")
    for shell in ("bash", "dash"):
        add(f"{shell} -c 'cat <<EOF\n{PY} {SCRIPT}\nEOF'")
        add(f"{shell} -c 'cat <<EOF\nx\nEOF\n{PY} {SCRIPT}'")
    add(f"bash -c '{PY} {SCRIPT} <<EOF\nignored\nEOF'")
    add(f'bash -c "f={SCRIPT}; {PY} \\"$f\\""')
    add(f'f={SCRIPT}; bash -c "{PY} \\"$f\\""')
    add(f'bash -c "f={SCRIPT}; {PY} \\"\\$f\\""')
    # a compound's opening word after an assignment, which every shell refuses
    for opener in (
        f"for g in; do {PY} {SCRIPT}; done",
        f"for g in x; do {PY} {SCRIPT}; done",
        f"while true; do {PY} {SCRIPT}; break; done",
        f"if true; then {PY} {SCRIPT}; fi",
        f"case x in x) {PY} {SCRIPT};; esac",
        f"{{ {PY} {SCRIPT}; }}",
        f"( {PY} {SCRIPT} )",
        f"[[ -n x ]] && {PY} {SCRIPT}",
        f"(( 1 )) && {PY} {SCRIPT}",
    ):
        add(f"LANG=C {opener}")
    add(f"f=x for g in; do echo; done; {PY} {SCRIPT}")
    add(f"A=1 for x; {PY} {SCRIPT}")
    add(f"{PY} {SCRIPT}; A=1 for g in x; do :; done")
    add(f"{PY} {SCRIPT}\nA=1 for g in x; do :; done")
    add(f"dash -c 'A=1 for g in x; do {PY} {SCRIPT}; done'")
    add(f"A=$(true) {PY} {SCRIPT}")
    add(f"A='for' {PY} {SCRIPT}")
    add(f"for g in x; do A=1 {PY} {SCRIPT}; done")
    cases.extend(refusal_units())
    return cases


# Lines placed before and after a line the shell refuses: what ran before it
# still ran, and nothing on or after it did; only a plain first command is
# credited, so the rest are partial whether they ran or not. These are scored
# even though the shell reports a syntax error, because the error is the
# point of the shape.
_REFUSED_LINES = (
    "A=1 for g in x; do :; done",
    "A=1 if true; then :; fi",
    "A=1 ( : )",
    "A=1 { :; }",
    "A=1 (( 1 ))",
)
_BEFORE_REFUSED_LINE = (
    f"{PY} {SCRIPT}",
    f"cat {SCRIPT}",
    f"echo a\n{PY} {SCRIPT}",
    f"{PY} {SCRIPT} # comment",
    f"for x in 1; do\n{PY} {SCRIPT}\ndone",
    f"if true; then\n{PY} {SCRIPT}\nfi",
    f"{{ {PY} {SCRIPT}\n}}",
    f"( {PY} {SCRIPT}\n)",
    f"f() {{\n{PY} {SCRIPT}\n}}\nf",
    f"x=$(echo a\n); {PY} {SCRIPT}",
    f"a=(1\n2)\n{PY} {SCRIPT}",
    f"echo 'a\nb'; {PY} {SCRIPT}",
    f"cat <<EOF\nx\nEOF\n{PY} {SCRIPT}",
    f"cat <<'EOF'\nA=1 for g in x; do :; done\nEOF\n{PY} {SCRIPT}",
    f"{PY} {SCRIPT} <<EOF\ny\nEOF",
    f"case x in\nx) :;;\nesac\n{PY} {SCRIPT}",
    f'export f={SCRIPT}\n{PY} "$f"',
    f"{PY} {SCRIPT} &&",
    f"{PY} {SCRIPT} |",
    f"{PY} {SCRIPT} \\",
    f"{PY} {SCRIPT};",
    # Refused on its own line, or run after something that stops it.
    f"{PY} {SCRIPT} >",
    f"{PY} {SCRIPT} <",
    f"{PY} {SCRIPT} 2>",
    f"{PY} {SCRIPT} 2>&",
    f"{PY} {SCRIPT} >;",
    f"{PY} {SCRIPT} > # out",
    f"{PY} {SCRIPT} >#out",
    f"{PY} {SCRIPT}; ;",
    f"if {PY} {SCRIPT}; fi",
    f"{PY} {SCRIPT}\\\necho a",
    f"exit\n{PY} {SCRIPT}",
    f"exec true\n{PY} {SCRIPT}",
    f"false && {PY} {SCRIPT}",
    f"f() {{\n{PY} {SCRIPT}\n}}",
    f"bash -c '{PY} {SCRIPT}'",
    f"bash -c '{PY} {SCRIPT} >'",
    # A carriage return before the newline is part of the word before it.
    f"{PY} {SCRIPT}\r",
    f"{PY} {SCRIPT};\r",
    f"{PY} {SCRIPT} |&\r",
    f"time -p {PY} {SCRIPT}\r",
    f"f() {{\n{PY} {SCRIPT}\n}}\nf\r",
    # Complete: these run.
    f"{PY} {SCRIPT} 2>&1",
    f"{PY} {SCRIPT} '>'",
    f"{PY} {SCRIPT} >out #c",
)


def refusal_units() -> list[tuple[str, str]]:
    """A refused line after complete commands, inside a compound, and before
    an invocation, at the top level and in ``-c`` payloads."""
    cases: list[tuple[str, str]] = []
    for before in _BEFORE_REFUSED_LINE:
        for refused in _REFUSED_LINES[:3]:
            cases.append((f"{before}\n{refused}", "run.py"))
        cases.append((f"{before}\n{_REFUSED_LINES[0]}\n{PY} {SCRIPT}", "run.py"))
    for refused in _REFUSED_LINES:
        cases.append((f"{PY} {SCRIPT}\n{refused}", "run.py"))
        cases.append((f"{PY} {SCRIPT}; {refused}", "run.py"))
        cases.append((f"{refused}\n{PY} {SCRIPT}", "run.py"))
        cases.append((f"if true; then\n{PY} {SCRIPT}\n{refused}\nfi", "run.py"))
    for shell in ("bash", "dash", "sh", "zsh", "ksh", "mksh"):
        cases.append((f"{shell} -c '{PY} {SCRIPT}\n{_REFUSED_LINES[0]}'", "run.py"))
        cases.append((f"{shell} -c '{PY} {SCRIPT}\ncat <<< x'", "run.py"))
        cases.append((f"{shell} -c 'cat <<< x\n{PY} {SCRIPT}'", "run.py"))
        cases.append((f"{shell} -c '{PY} {SCRIPT}; cat <<< x'", "run.py"))
        cases.append((f"{shell} -c '{PY} {SCRIPT}\r\n{_REFUSED_LINES[0]}'", "run.py"))
        cases.append((f"{shell} -c '{PY} {SCRIPT} >\n{_REFUSED_LINES[0]}'", "run.py"))
    return cases


_SCORED_DESPITE_SYNTAX_ERROR = frozenset(command for command, _ in refusal_units())


# Commands a native tool call ran under the shell it names in
# ``action_input.shell`` (Codex's ``exec_command``). Each is executed by that
# shell and scored with the name passed through.
_NATIVE_SHELL_COMMANDS = (
    f'printf "" | for f in {SCRIPT}; do :; done; {PY} "$f"',
    f'f={O}; printf "" | for f in {SCRIPT}; do :; done; {PY} "$f"',
    f'printf "" | {{ f={SCRIPT}; }}; {PY} "$f"',
    f'echo | while read -r l; do f={SCRIPT}; done; {PY} "$f"',
    f'printf "" | for f in {SCRIPT}; do :; done | cat; {PY} "$f"',
    f"cat <<< x; {PY} {SCRIPT}",
    f"{PY} {SCRIPT}\ncat <<< x",
    f"cat <<< x\n{PY} {SCRIPT}",
    f"{PY} {SCRIPT}\nA=1 for g in x; do :; done",
    f"{PY} {SCRIPT}; A=1 for g in x; do :; done",
    f"A=1 for g in x; do :; done\n{PY} {SCRIPT}",
    f"{PY} {SCRIPT} >\nA=1 for g in x; do :; done",
    f"{PY} {SCRIPT}\r\nA=1 for g in x; do :; done",
    f'declare f={SCRIPT}; {PY} "$f"',
    f'typeset f={SCRIPT}; {PY} "$f"',
    f'export f={SCRIPT}; {PY} "$f"',
    f'f={SCRIPT}; ((f=1)); {PY} "$f"',
    f'f={SCRIPT}; let f=1; {PY} "$f"',
    # A listing, functions, and attributes that change a value, per shell.
    f'export -p f={SCRIPT}; {PY} "$f"',
    f'export f={SCRIPT}; export -p f={O}; {PY} "$f"',
    f'export f={O}; export -p f={SCRIPT}; {PY} "$f"',
    f'readonly -p f={SCRIPT}; {PY} "$f"',
    f'export f={SCRIPT}; readonly -p f={O}; {PY} "$f"',
    f"f={SCRIPT}; export -p f; bash -c '{PY} \"$f\"'",
    f'f={SCRIPT}; readonly -p f; f={O}; {PY} "$f"',
    f'f={SCRIPT}; export -p >/dev/null; {PY} "$f"',
    f"f={SCRIPT}; export f; export -pn f; bash -c '{PY} \"$f\"'",
    f'f={O}; typeset -p f={SCRIPT}; {PY} "$f"',
    f'f={O}; typeset -px f={SCRIPT}; {PY} "$f"',
    f'f={O}; typeset -f f={SCRIPT}; {PY} "$f"',
    f"f={SCRIPT}; declare -F f; {PY} {SCRIPT}",
    f"typeset -F f={SCRIPT}; {PY} {SCRIPT}",
    f"f={SCRIPT}; typeset -F f; {PY} {SCRIPT}",
    f"f=1.5; typeset -F f; {PY} {SCRIPT}",
    f"f={SCRIPT}; typeset -i f; {PY} {SCRIPT}",
    f"typeset -i f={SCRIPT}; {PY} {SCRIPT}",
    f'f={SCRIPT}; typeset -L3 f; {PY} "$f"',
    f'f={SCRIPT}; typeset -L f; {PY} "$f"',
    f'f={SCRIPT}; typeset -L 3 f; {PY} "$f"',
    # The directory lets the redirection succeed, so a POSIX shell goes on.
    f'mkdir -p f={SCRIPT.rsplit("/", 1)[0]}; export f={SCRIPT}; export -p > f={O}; {PY} "$f"',
    f'mkdir -p f={SCRIPT.rsplit("/", 1)[0]}; f={O}; export -p > f={SCRIPT}; {PY} "$f"',
    f"f=0x10; typeset -i f; {PY} {SCRIPT}",
    f"typeset -i f=16#ff; {PY} {SCRIPT}",
    f"typeset -pF f={SCRIPT}; {PY} {SCRIPT}",
    # A value that ends in an operator is not a redirection, and a base is a
    # number however many zeros lead it.
    f"export g='>' f={SCRIPT}; {PY} \"$f\"",
    f"f={SCRIPT}; export g='>' f={O}; {PY} \"$f\"",
    f"typeset -i f={'0' * 4400}16#ff; {PY} {SCRIPT}",
    f"{PY} {SCRIPT}",
    f"cat {SCRIPT}",
    f'for f in {SCRIPT}; do {PY} "$f"; done',
)
_NATIVE_SHELLS = ("bash", "zsh", "ksh", "mksh", "dash", "sh")


def native_shell_cases() -> list[tuple[str, str, str]]:
    return [(command, "run.py", shell) for shell in _NATIVE_SHELLS for command in _NATIVE_SHELL_COMMANDS]


def _heredocs(count: int, before: str, after: str = "", body: str = "x") -> str:
    """``before``, then ``count`` heredocs declared on one line, ``after``, and each body in order."""
    names = [f"D{index}" for index in range(count)]
    declared = " ".join(f"<<{name}" for name in names)
    return f"{before} {declared}{' ' + after if after else ''}\n" + "".join(f"{body}\n{name}\n" for name in names)


# Builtins that bind a variable: bound in every shell (export, readonly), in
# some (declare, typeset), or to a value the text does not settle.
BINDING_BUILTIN_FORMS = [
    "export",
    "readonly",
    "declare",
    "typeset",
    "typeset -x",
    "export --",
    "local",
    "declare -u",
    "export -p",
    "readonly -p",
    "declare -p",
    "typeset -p",
]
PREFIX_BINDING_FORMS = [
    'f={v} {py} "$f"',
    'f={v} true; {py} "$f"',
    'env f={v} {py} "$f"',
    'f={v} :; {py} "$f"',
    'f={v} export g; {py} "$f"',
]
MULTI_HEREDOC_FORMS = [
    "{py} <<A <<B {script}\nx\nA\ny\nB",
    "{py} <<-A <<B {script}\n\tx\n\tA\ny\nB",
    "{py} <<A <<B <<C {script}\nx\nA\ny\nB\nz\nC",
    "{py} <<'A' <<\"B\" {script}\nx\nA\ny\nB",
    "{py} <<A {script} <<B\nx\nA\ny\nB",
    "cat <<A; {py} <<B {script}\nx\nA\ny\nB",
    "cat <<A <<B {script}\nx\nA\ny\nB",
    "{py} <<A <<B {other}\nx\nA\ny\nB",
    "cat <<A <<B\nx\nA\n{py} {script}\nB",
    "cat <<A <<B\n{py} {script}\nA\ny\nB",
    "cat <<A <<A\nx\nA\ny\nA\n{py} {script}",
    "{py} <<A <<<y {script}\nx\nA",
]

# env options that take names out of what a child inherits.
ENV_REMOVAL_FORMS = [
    "env -i",
    "env -u f",
    "env -",
    "env --ignore-environment",
    "env --unset=f",
    "env -uf",
    'env -i PATH="$PATH"',
    "env -i env",
    "env -S '-u f'",
]
# Ways of taking the export attribute away, directly or through a builtin
# prefix, and forms that touch functions only.
EXPORT_REMOVAL_FORMS = [
    "declare +x f",
    "typeset +x f",
    "declare -x +x f",
    "export -n f",
    "command export -n f",
    "builtin export -n f",
    "command command export -n f",
    "command -p export -n f",
    "builtin declare +x f",
    "eval export -n f",
    "export -f f",
    "export -nf f",
    "declare +x f; export f",
    "export -pn f",
    "export -p -n f",
]
EVAL_FORMS = [
    "f={v} eval '{py} \"$f\"'",
    "f={v}; eval '{py} \"$f\"'",
    "export f={v}; eval '{py} \"$f\"'",
    "f={o} eval '{py} \"$f\"'",
    'f={v}; eval f={o}; {py} "$f"',
    "f={v}; eval 'unset f'; {py} \"$f\"",
    'f={v}; eval \'echo "$f"\'; {py} "$f"',
    "f={v} {py} -c 'import os; os.system(\"{py} $f\")'",
    'f={v}; bash <<EOF\n{py} "$f"\nEOF',
    "export f={v}; bash <<'EOF'\n{py} \"$f\"\nEOF",
]


# Scopes a binding can be made in, each wrapping the text inside it: groups,
# brace groups, compound commands, and each of those as the first, a middle
# or the last stage of a pipeline.
SCOPE_WRAPPERS = [
    "({x})",
    "{{ {x}; }}",
    'printf "" | {{ {x}; }}',
    "{{ {x}; }} | cat",
    'printf "" | {{ {x}; }} | cat',
    'printf "" | for g in 1; do {x}; done',
    "for g in 1; do {x}; done | cat",
    "if true; then {x}; fi | cat",
    'printf "" | if true; then {x}; fi',
    'printf "x\\n" | while read -r l; do {x}; done',
    'printf "" | ({x})',
    "({x}) | cat",
    "for g in 1; do {x}; done",
    "if true; then {x}; fi",
    "for g in; do :; done; {x}",
    "{{ for g in; do :; done; {x}; }}",
]
SCOPE_BINDINGS = [
    "f={v}",
    "f={v} | cat",
    'printf "" | f={v}',
    "(f={v})",
    "for f in {v}; do :; done",
    'printf "" | for f in {v}; do :; done',
    "{{ f={v}; }}",
]


def _compose(rng: random.Random) -> tuple[str, str]:
    """A binding nested two or three scopes deep, read at a random level.

    Which value the interpreter reads depends on which scopes keep a binding,
    in which shell: a group never does, a pipeline stage only as zsh's or
    ksh's last stage, a brace group or a loop always. Both assignment
    directions are drawn, so a scope that leaks and one that forgets are both
    caught.
    """
    outer, inner = rng.choice([(SCRIPT, O), (O, SCRIPT)])
    text = rng.choice(SCOPE_BINDINGS).format(v=inner)
    for _ in range(rng.choice([2, 3])):
        wrapper = rng.choice(SCOPE_WRAPPERS)
        text = wrapper.format(x=f'{text}; {PY} "$f"' if rng.random() < 0.4 else text)
    command = f'f={outer}; {text}; {PY} "$f"'
    shell = rng.choice(["", "zsh", "ksh", "bash"])
    return (f"{shell} -c '{command}'" if shell else command), "run.py"


def _generate(rng: random.Random) -> tuple[str, str]:
    """One random command and the script it is scored against."""
    prefix = rng.choice(PREFIX_FORMS)
    script = "run.py"
    target = "run.py" if prefix.startswith(("cd ", "(cd ")) else SCRIPT
    interpreter = VERSIONED_PY if rng.random() < 0.15 else PY
    redirect = rng.choice(REDIRECT_FORMS) if rng.random() < 0.15 else ""
    if rng.random() < 0.05:
        redirect = rng.choice(DETACHED_DESCRIPTOR_FORMS)
    shape = rng.random()
    if shape < 0.1:
        body = rng.choice(UNRELATED_BODIES)
    elif shape < 0.2:
        script = "run.sh"
        shell_target = "run.sh" if prefix.startswith(("cd ", "(cd ")) else SHELL_SCRIPT
        body = f"bash {rng.choice(['', '-- ', '-x '])}{shell_target}{rng.choice(SCRIPT_ARGUMENT_FORMS)}"
    elif shape < 0.27:
        script = "run.sh"
        shell_target = "run.sh" if prefix.startswith(("cd ", "(cd ")) else SHELL_SCRIPT
        payload = rng.choice([f"bash {shell_target}", f"cat {shell_target}", f"./{shell_target}"])
        between = rng.choice([*REDIRECT_FORMS, ""])
        body = rng.choice([f"bash -c {between}'{payload}'", f"bash {between}-c '{payload}'"])
    elif shape < 0.34:
        header = rng.choice(REBINDING_HEADERS)
        inside = prefix.startswith(("cd ", "(cd "))
        names = {
            "script": "run.py" if inside else SCRIPT,
            "other": "other.py" if inside else "skills/demo/other.py",
            "another": "another.py" if inside else "skills/demo/another.py",
        }
        body = f"for f in {names['script']}; do cat $f; done; " + header.format(
            body=rng.choice([f"{interpreter} $f", "cat $f"]), **names
        )
    elif shape < 0.42:
        # a binding made in a pipeline or a group, then read by the interpreter
        inside = prefix.startswith(("cd ", "(cd "))
        bound = "run.py" if inside else SCRIPT
        binding = rng.choice(
            [
                f'printf "" | for f in {bound}; do cat "$f"; done',
                f'for f in {bound}; do cat "$f"; done | cat',
                f'printf "" | f={bound}',
                f'printf "" | (f={bound})',
                f'printf "" | {{ f={bound}; }}',
                f'printf "" | if true; then f={bound}; fi',
                'printf "" | for f in; do :; done',
                f"(f={bound})",
                f"{{ f={bound}; }}",
                f"f={bound}",
            ]
        )
        reader = rng.choice([f'{interpreter} "$f"', 'cat "$f"'])
        shell = rng.choice(["", "zsh -c ", "bash -c ", "sh -c ", "dash -c ", "ksh -c "])
        body = f"{shell}'{binding}; {reader}'" if shell else f"{binding}; {reader}"
    elif shape < 0.48:
        # a variable a builtin or a prefix binds, every heredoc on a line, and
        # three opening parentheses (from review of revision 16)
        inside = prefix.startswith(("cd ", "(cd "))
        bound = "run.py" if inside else SCRIPT
        other = "other.py" if inside else O
        closing = "\n)" if prefix.startswith("(") else ""
        form = rng.random()
        if form < 0.25:
            text = rng.choice(MULTI_HEREDOC_FORMS).format(py=interpreter, script=bound, other=other)
            # a suffix after the last terminator would unterminate it
            return prefix + text + closing, script
        if form < 0.35:
            text = f'f={other}; (((for f in {bound}; do :; done; {interpreter} "$f")) | cat); {interpreter} "$f"'
            shell = rng.choice(["", "zsh -c ", "ksh -c ", "mksh -c ", "dash -c "])
            return prefix + (f"{shell}'{text}'" if shell else text) + closing, script
        binder = rng.choice(
            [
                *(f"{declaration} f={bound}" for declaration in BINDING_BUILTIN_FORMS),
                f"f={bound}; export f",
                f"f={bound}; unset f",
                f"f={bound} true",
                f"env f={bound} true",
                f"f={bound} :",
                f"f={other}; printf -v f {bound}",
                f"readonly f={bound}; f={other}",
                # from review of revision 17
                f"export f={bound}; declare +x f",
                f"export f={bound}; command export -n f",
                f"g={other}; export f=$g; g={bound}",
                f"g={bound}; export f=$g; g={other}",
                f"f={bound}; eval f={other}",
                f"export f={bound}; eval export -n f",
            ]
        )
        shell = rng.choice(["", "", "zsh -c ", "dash -c ", "ksh -c ", "mksh -c ", "sh -c "])
        readers = [f'{interpreter} "$f"', 'cat "$f"']
        if not shell:
            readers.append(f"bash -c '{interpreter} \"$f\"'")
            readers.append(f"env -u f bash -c '{interpreter} \"$f\"'")
            readers.append(f"env -i f={bound} bash -c '{interpreter} \"$f\"'")
        text = f"{binder}; {rng.choice(readers)}"
        body = f"{shell}'{text}'" if shell else text
    elif shape < 0.58:
        body = f"{rng.choice(NON_EXECUTING_VERBS)} {target}"
    else:
        body = f"{rng.choice(WRAPPER_FORMS)}{interpreter} {rng.choice(INTERPRETER_FORMS)}{redirect}{target}"
    if rng.random() < 0.15:
        body = rng.choice(CONTROL_FORMS).format(body=body)
    suffix = rng.choice(SUFFIX_FORMS)
    if prefix.startswith("("):
        suffix += ")"
    command = prefix + body + suffix
    if rng.random() < 0.3:
        command += rng.choice([" && ", " ; ", " || ", "\n"]) + rng.choice(
            ["echo tail", "true", f"{rng.choice(NON_EXECUTING_VERBS)} {target}"]
        )
    return command, script


def _build(directory: Path) -> None:
    for relative, text in FIXTURES.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if path.suffix in {".py", ".sh", ".pl", ".rb", ".js"}:
            path.chmod(0o755)  # fixtures are invoked directly, so they need the bit


def _load_baseline(ref: str):
    """The host checker and the template as they stood at a git ref."""
    modules = []
    for name, relative in (
        ("baseline_host", "src/skillevaluator/tier3/eval_core/checks.py"),
        ("baseline_template", "src/skillevaluator/tier3/harbor/templates/eval.py"),
    ):
        source = subprocess.run(
            ["git", "show", f"{ref}:{relative}"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout
        path = Path(tempfile.mkdtemp()) / Path(relative).name
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


# Tools a command may name that are not installed everywhere. When one is
# missing, the command did not run for a reason its text does not carry, and
# the harness reports it as not runnable rather than scoring it; reading the
# captured output alone misses a `command not found` sent to /dev/null.
_EXTERNAL_TOOLS = (
    "timeout",
    "nohup",
    "nice",
    "stdbuf",
    "setsid",
    "xargs",
    "parallel",
    "flock",
    "taskset",
    "ionice",
    "chrt",
    "strace",
    "uv",
    "ruby",
    "perl",
    "node",
    "zsh",
    "ksh",
    "mksh",
    "dash",
    "ash",
)
_TOOL_WORD_RE = re.compile(r"(?<![\w./-])(" + "|".join(_EXTERNAL_TOOLS) + r")(?![\w.-])")


def _missing_tool(command: str) -> str | None:
    for name in dict.fromkeys(_TOOL_WORD_RE.findall(command)):
        if shutil.which(name) is None:
            return name
    return None


def _run_one(
    command: str, script: str, shell: str | None = None
) -> tuple[str, str, float | None, bool | None, list[dict[str, str]]]:
    """Execute one command and compare the marker against both checkers.

    With ``shell``, the command is run by that shell and scored as a native
    call naming it. Returns the outcome, its detail, the host score, whether
    the script ran, and the tool call the checkers read, so a baseline can
    score it.
    """
    missing = _missing_tool(command) if shell is None else (None if shutil.which(shell) else shell)
    if missing is not None:
        return ("inconclusive", f"{missing}: not installed here", None, None, [])
    executable = shutil.which(shell) if shell is not None else "/bin/bash"
    directory = Path(tempfile.mkdtemp())
    try:
        _build(directory)
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=30,
                stdin=subprocess.DEVNULL,
                executable=executable,
            )
        except subprocess.TimeoutExpired:
            return ("inconclusive", "timed out", None, None, [])
        output = completed.stdout + completed.stderr
        ran = any(directory.rglob(f"MARKER.{script}"))
        if not ran and _DETACHED_RE.search(command):
            deadline = time.monotonic() + 2.0
            while not ran and time.monotonic() < deadline:
                time.sleep(0.05)
                ran = any(directory.rglob(f"MARKER.{script}"))
        refused_shape = shell is not None or command in _SCORED_DESPITE_SYNTAX_ERROR
        if not ran and ("command not found" in output or ("syntax error" in output and not refused_shape)):
            return ("inconclusive", output.strip().splitlines()[0][:70] if output.strip() else "", None, None, [])

        calls = [
            {
                "action": "Bash",
                "action_input": {"command": command},
                "observation": output[:400] + f"\nExit code {completed.returncode}",
            }
            if shell is None
            else {
                "action": "exec_command",
                "action_input": {"cmd": command, "shell": executable},
                "observation": output[:400] + f"\nExit code {completed.returncode}",
            }
        ]
        host_result = host_checks.check_script_execution(calls, script)
        template_result = TEMPLATE.check_script_execution(calls, script)
        if (host_result["score"], host_result["reason"], host_result["passed"]) != (
            template_result["score"],
            template_result["reason"],
            template_result["passed"],
        ):
            return ("divergence", f"host={host_result} template={template_result}", None, ran, calls)

        score = host_result["score"]
        if score == 1.0 and not ran:
            if completed.returncode != 0 and ("&&" in command or "||" in command):
                # A chain short-circuited before reaching the invocation. The
                # walk is static and a tool call does not carry which link
                # failed, so this is the limitation the PR declares, reported
                # here rather than hidden.
                return (
                    "short-circuited chain (declared limitation)",
                    f"exit {completed.returncode}",
                    score,
                    ran,
                    calls,
                )
            return ("false positive", host_result["reason"], score, ran, calls)
        if ran and score == 0.0:
            return ("false negative", host_result["reason"], score, ran, calls)
        if ran and score != 1.0:
            return ("partial on a real run", f"{score}", score, ran, calls)
        if (
            score == 0.75
            and script.rsplit("/", 1)[-1] not in command
            and "could not be classified" in host_result["reason"]
        ):
            return ("partial without a reference", host_result["reason"], score, ran, calls)
        return ("agreed", f"{score}", score, ran, calls)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fuzz", type=int, default=0, help="also generate and execute N random commands")
    parser.add_argument(
        "--compose", type=int, default=0, help="also generate and execute N commands that nest binding scopes"
    )
    parser.add_argument("--seed", type=int, default=20260923, help="seed for --fuzz")
    parser.add_argument("--verbose", action="store_true", help="print every command and its outcome")
    parser.add_argument("--baseline", help="git ref of an earlier checker to compare every score against")
    arguments = parser.parse_args()

    commands = curated()
    if arguments.fuzz:
        rng = random.Random(arguments.seed)
        seen = {command for command, _ in commands}
        generated: list[tuple[str, str]] = []
        while len(generated) < arguments.fuzz:
            command, script = _generate(rng)
            if command not in seen:
                seen.add(command)
                generated.append((command, script))
        commands += generated
    if arguments.compose:
        rng = random.Random(arguments.seed + 1)
        seen = {command for command, _ in commands}
        composed: list[tuple[str, str]] = []
        while len(composed) < arguments.compose:
            command, script = _compose(rng)
            if command not in seen:
                seen.add(command)
                composed.append((command, script))
        commands += composed

    baseline = _load_baseline(arguments.baseline) if arguments.baseline else None
    buckets: dict[str, list[tuple[str, str]]] = {}
    moves: dict[str, list[tuple[str, float | str, float]]] = {}
    runs: list[tuple[str, str, str | None]] = [(command, script, None) for command, script in commands]
    runs += native_shell_cases()
    commands = [(command, script) for command, script, _ in runs]
    for command, script, shell in runs:
        outcome, detail, score, ran, calls = _run_one(command, script, shell)
        if shell is not None:
            command = f"[{shell}] {command}"
        buckets.setdefault(outcome, []).append((command, detail))
        if arguments.verbose:
            print(f"{outcome:22} {command!r} {detail}")
        if baseline is not None and score is not None:
            try:
                before = baseline[0].check_script_execution(calls, script)["score"]
            except Exception as error:  # an earlier checker may raise where this one scores
                moves.setdefault("baseline raised, now scored", []).append((command, type(error).__name__, score))
                continue
            if before != score:
                if ran and score < before:
                    kind = "score lowered on a command that ran (review each)"
                elif not ran and score > before:
                    kind = "score raised on a command that did not run (review each)"
                elif not ran and before == 1.0:
                    kind = "false positive repaired"
                elif ran and before == 0.0:
                    kind = "false negative repaired"
                else:
                    kind = "other move"
                moves.setdefault(kind, []).append((command, before, score))

    executed = len(commands) - len(buckets.get("inconclusive", []))
    print(f"\ncommands executed: {executed} of {len(commands)}")
    defects = 0
    short_circuit = buckets.get("short-circuited chain (declared limitation)", [])
    for outcome, label in (
        ("false positive", "false positives"),
        ("false negative", "false negatives"),
        ("divergence", "divergences"),
        ("partial without a reference", "partials without a reference"),
    ):
        entries = buckets.get(outcome, [])
        defects += len(entries)
        print(f"  {label}: {len(entries)}")
        for command, detail in entries[:25]:
            print(f"     {command!r} -> {detail}")
    print(f"  short-circuited chains (declared limitation, credited but did not run): {len(short_circuit)}")
    for command, detail in short_circuit[:15]:
        print(f"     {command!r} -> {detail}")
    partial = buckets.get("partial on a real run", [])
    print(f"  partial on a real run (not a defect): {len(partial)}")
    for command, detail in partial[:40]:
        print(f"     {command!r} -> {detail}")
    inconclusive = buckets.get("inconclusive", [])
    if inconclusive:
        print(f"  not runnable on this machine: {len(inconclusive)}")
        for command, detail in inconclusive[:15]:
            print(f"     {command!r} :: {detail}")
    if baseline is not None:
        total = sum(len(entries) for entries in moves.values())
        print(f"\nscores that moved against {arguments.baseline}: {total}")
        for kind in (
            "score lowered on a command that ran (review each)",
            "score raised on a command that did not run (review each)",
            "false positive repaired",
            "false negative repaired",
            "baseline raised, now scored",
            "other move",
        ):
            entries = moves.get(kind, [])
            print(f"  {kind}: {len(entries)}")
            for command, before, after in entries[:40]:
                print(f"     {command!r}: {before} -> {after}")
    return 1 if defects else 0


if __name__ == "__main__":
    raise SystemExit(main())
