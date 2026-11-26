# Guided Tier 3 dataset generation

Standalone `create-eval-dataset` defaults to `--grounding auto`: prefer author
guidance in `--prompt` / `<skill-dir>/evals/EVAL.md` with real content beyond
headings, then usable matching retained trajectories from this skill's completed
runs. Write task requests and observable outcomes in the guidance file first:

```bash
skillevaluator create-eval-dataset <skill-dir> --full --prompt <skill-dir>/evals/EVAL.md
```

If neither guidance nor matching retained traces exists, generation fails before
provider access or writes. `--refine` and `--no-llm` do not bypass that preflight.
For the prior skill-only starter workflow, explicitly opt in:

```bash
skillevaluator create-eval-dataset <skill-dir> --full --no-llm --grounding skill
```

Auto leaves an existing dataset unchanged unless you pass `--force` or
`--refine`, even when matching traces exist. To refine from completed results:

```bash
skillevaluator create-eval-dataset <skill-dir> --refine --from-results <run-dir>
```

Auto trace refinement preserves existing authored case IDs and expectations;
unmatched cases and observed answer proposals need review. When auto selects
traces, proposals that describe skill use leave `expected_output` empty for an
independent authored outcome. Existing authored expectations are preserved. Explicit
`--grounding trace` still opts into in-place refinement without `--refine`.
Generation provenance records the effective `intent` or `trace` grounding and input hashes. Explicit
`intent`, `trace`, and `skill` modes remain available. Unattended autopilot retains
its existing skill-grounded one-case starter contract.
