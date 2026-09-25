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
bash, zsh and ksh). Add ``--baseline REF`` to also score every command with the checker as it
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
    return cases


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
    elif shape < 0.55:
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


def _run_one(command: str, script: str) -> tuple[str, str, float | None, bool | None, list[dict[str, str]]]:
    """Execute one command and compare the marker against both checkers.

    Returns the outcome, its detail, the host score, whether the script
    ran, and the tool call the checkers read, so a baseline can score it.
    """
    missing = _missing_tool(command)
    if missing is not None:
        return ("inconclusive", f"{missing}: not installed here", None, None, [])
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
                executable="/bin/bash",
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
        if not ran and ("command not found" in output or "syntax error" in output):
            return ("inconclusive", output.strip().splitlines()[0][:70] if output.strip() else "", None, None, [])

        calls = [
            {
                "action": "Bash",
                "action_input": {"command": command},
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
    moves: dict[str, list[tuple[str, float, float]]] = {}
    for command, script in commands:
        outcome, detail, score, ran, calls = _run_one(command, script)
        buckets.setdefault(outcome, []).append((command, detail))
        if arguments.verbose:
            print(f"{outcome:22} {command!r} {detail}")
        if baseline is not None and score is not None:
            before = baseline[0].check_script_execution(calls, script)["score"]
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
            "other move",
        ):
            entries = moves.get(kind, [])
            print(f"  {kind}: {len(entries)}")
            for command, before, after in entries[:40]:
                print(f"     {command!r}: {before} -> {after}")
    return 1 if defects else 0


if __name__ == "__main__":
    raise SystemExit(main())
