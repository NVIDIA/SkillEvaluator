# Public terminal starter

Run from the repository root with [uv](https://docs.astral.sh/uv/getting-started/installation/)
and Git installed. The CLI is pinned to the public 0.8.0 release wheel; the demo files
come from this checkout. No API key, scanner, GPU or Docker is needed.

```bash
uv run --isolated --no-project --python 3.13 --with "skillevaluator @ https://github.com/NVIDIA/SkillEvaluator/releases/download/v0.8.0/skillevaluator-0.8.0-py3-none-any.whl" python scripts/demo_starter.py --output reports/starter
```

The runner copies the installed `text-analyzer` reference skill into the output
directory, adds this folder's task and input fixture, then executes the real
text-analysis script and CLI. It compares their results to `expected.json`:

- The fixture has 4 words, 2 lines, 27 characters and 2 sentences.
- Original documentation scores 86.2/100 (B) and passes the explicit 80 threshold.
- A separate, deliberately shortened copy scores 72.8/100 (C) and fails that same
  threshold. The runner expects that command's exit code 1, then exits 0 itself.
- The passing report still has `publication_status: incomplete`: static quality
  is not security coverage or evidence that an agent benefits from the skill.

These are deterministic quality checks, **not** agent-impact scores or a live
benchmark. No saved or historical report is replayed as new evidence. All JSON
reports and terminal text are freshly generated. `receipt.json` records the
installed version and timestamp only after all checks pass.

The copied skill is editable and includes `evals/evals.json` and
`evals/files/input.txt`. The demo directly checks the script, not an agent's
ability to execute the task. Live evaluation requires additional setup.

Use a new output directory for every run; existing directories are never
overwritten. Reports stay outside the skill. Do not update expected scores just
to silence a failure: inspect the report findings and the evaluator version.

See the [complete walkthrough](../../docs/starter-demo.mdx) for installation,
manual replay, report interpretation, live evaluation, recovery and adaptation.
