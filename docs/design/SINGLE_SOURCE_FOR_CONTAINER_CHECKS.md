<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# One source for the container checks

Status: proposal, not implemented. Not a published docs page (not in `fern/docs.yml`).

## The problem

The Tier 3 security and grading checks live twice: `tier3/eval_core/checks.py` (host,
~9,500 lines) and `tier3/harbor/templates/eval.py` (the verifier that runs alone in
the task container, ~13,500 lines). The same bug has had to be fixed in both copies
three times in a row (network exfiltration, rm outside `/tmp`, curl inside a compound
command).

Measured on this branch with `tests/tier3/_tools/measure_checks_template_drift.py`:

- 248 top-level functions/classes and 340 module constants exist in both files.
- 117 of the shared functions sit in two blocks a test keeps byte-identical
  (canary exfiltration, runtime security). The other 131, including every `check_*`
  entry point, had no guard.
- 74 shared functions had different ASTs: 54 only in type annotations, 17 in
  cosmetic ways (renamed locals, keyword-only markers, an equivalent rewrite),
  1 by design (`check_security` takes a trajectory in the container), and 2 for
  real (`check_error_recovery` keywords, `check_tool_efficiency` details). This
  branch fixes those two and adds `tests/tier3/test_checks_template_parity.py`,
  which runs both copies on the same corpus of about 1,900 commands and paths.
- `eval.py` also repeats 66 functions from `atif_helpers.py` and 34 from
  `llm_judge.py`. That is the same problem, but outside the scope of this note.

The parity test catches drift after the fact. It does not stop the double edit.

## How eval.py reaches the container

- `harbor/adapter.py::_copy_verifier` copies four files into each staged task's
  `tests/skill_evaluator/` folder: `templates/eval.py`, `eval_core/log_converters.py`,
  `eval_core/codex_tool_call_normalizer.py` and `evidence.py`. Each staging
  rebuilds that folder from scratch. There is no build or generation step: the
  files are package data, copied as they are.
- `test.sh` runs `python3 -I "${evaluator_dir}/eval.py"` with the task image's
  own `python3`, which can be as old as 3.9. Only the standard library and
  `idna` are there.
- `eval.py` puts its own folder on `sys.path` and imports each helper with
  `try: from <sibling> import ...` / `except ImportError: from skillevaluator... import ...`.
- Guards: CI compiles the staged files on 3.9 and 3.11 and runs
  `scripts/ci/smoke_harbor_verifier.py`. `test_pr176_followups.py` fails if
  `_copy_verifier` stages a file that the CI compile step does not list.
- Other readers of `eval.py`: about 31 test files (they load it with
  `importlib`), `scripts/script_invocation_differential.py`, and the smoke script.

## The seam

That sibling import is already a real seam with two adapters: the copied file
inside the container, and the package import in the source tree. To give the
checks one source, put them behind the same seam. That means one more entry in
`_copy_verifier`, not a new mechanism.

## Options

A. **Keep two copies plus the parity test** (this branch). No migration cost.
   Drift on the corpus is caught. Every fix is still written twice.

B. **Copy `checks.py` into the container.** Stage `eval_core/checks.py` and
   `eval_core/secret_redaction.py` next to `eval.py`. `eval.py` imports the shared
   names and keeps only what is specific to the container: `main`, the
   `check_security` adapter that reads the trajectory, the judges, the
   reward-file writing, and the environment paths. The one source is
   `eval_core/checks.py`.
   Evidence that this works today:
   - `checks.py` compiles on Python 3.9.6.
   - It imports and runs under `/usr/bin/python3` 3.9.6 with only its two helper
     modules (`codex_tool_call_normalizer.py`, `secret_redaction.py`) in a stub
     `skillevaluator/tier3/eval_core/` package. Copied flat beside `eval.py` it
     does not import yet: its two imports name the package, so they need the
     sibling-first fallback `eval.py` already uses.
   - 1,448 `security_scan` / `score_skill_execution` probes over the corpus give
     byte-identical results on 3.9.6 and 3.12.12.
   - No test monkeypatches a shared name on the template. The 37 template
     monkeypatches hit judge, subagent and reward code, which stays in `eval.py`.
   Work needed:
   - Make `checks.py`'s two package imports sibling-first.
   - Add both files to `_copy_verifier` and the CI compile list (the guard test
     enforces this).
   - Delete about 7,000 duplicated lines from `eval.py`.
   - Retire the byte-identical block tests and shrink the parity test to an
     identity check.

C. **Build `eval.py` by concatenating modules.** Either at staging time, or
   committed with a test that checks it is up to date. This needs a generator the
   repo does not have, shifts traceback line numbers, and still gives tests a
   generated file to load. It is more machinery than B for the same result.

D. **Install the `skillevaluator` package in the task image.** Rejected. Task
   images are user images, and this adds a dependency plus version skew between
   host and container.

## Risks of B

- `checks.py` must stay safe to run on Python 3.9. Host authors lose 3.10+
  runtime features in that file (`zip(strict=)`, `match`, `isinstance(x, A | B)`).
  The CI compile step and smoke script cover this. The smoke script should also
  call a few checks from the copied `checks.py`.
- `checks.py` must not start importing host-only modules. Add a guard test:
  `checks.py` imports only the standard library and staged siblings.
- If a task folder is staged without `checks.py`, `eval.py` fails with
  `ImportError`. That is the safe failure, and every staging rebuilds the folder
  from the installed package.
- The diff is large and lands in files other branches are editing. Land it when
  the branch is quiet.

## Recommendation

Do B in two steps.

1. Move the two byte-guarded shared blocks (117 functions, already identical)
   into a new standard-library-only module, `eval_core/runtime_security.py`, and
   stage it next to `eval.py`. Both `checks.py` and `eval.py` import it. This is
   the smallest deepening: about 3,600 duplicated lines go away, and two
   byte-compare tests become one import. Effort: about half a day to a day.
2. Stage `checks.py` itself for the remaining 131 shared check helpers. Effort:
   about one day, mostly mechanical deletion and test fallout.

After step 2, the checks are one deep module. Its interface is `security_scan`,
the `check_*` functions and `score_skill_execution`, and `eval.py` is a thin
adapter from trajectory and reward files to that interface. A fix lands once
(locality), and every test exercises the one implementation (leverage).

This branch does not do either step. Each is a multi-thousand-line move in files
other worktrees are changing right now, so it is not the small, fully tested
change this task allows. The parity test is the guard until step 2 lands.
