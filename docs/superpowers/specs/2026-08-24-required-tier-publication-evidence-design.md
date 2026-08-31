# Required-Tier Publication Evidence Design

## Problem

`BenchmarkReporter` can recommend publication when its Tier Status table says
that Tier 1 or Tier 2 was not run. The overall verdict checks the results it was
given and has a special completeness rule for Tier 3, but it does not model the
absence of other publication-required tiers. The public benchmark linter repeats
the gap by validating only Tier 3 consistency.

## Publication contract

- Tier 1 is always required. A card without non-skipped Tier 1 evidence is
  `INCOMPLETE`.
- Tier 2 is required by default. Only an explicitly persisted
  `benchmark_policy.tier2_required = false` makes absent or cleanly skipped Tier
  2 evidence optional.
- Tier 3 keeps its current default-required policy and
  `benchmark_policy.tier3_required = false` escape hatch.
- Run-selection flags such as `--no-dedup` and `--tiers 1,3` are not publication
  waivers. Split-tier CI uses the same selectors, so inferring a waiver from
  them would preserve the original bug.
- `--no-block-on-dedup` changes the process exit gate only. It does not make
  Tier 2 optional for publication.
- Invalid policy values cannot waive required evidence. Each policy key is
  resolved independently by source precedence: the duplicated agent-evaluation
  payload and summary claims first, then peer result metadata. The duplicated
  claims must agree when both are valid booleans; a conflict resolves to
  required evidence. Invalid entries do not count as values; a level with no
  valid boolean falls through to the next lower-precedence source, and no valid
  value anywhere means required. Peer results share one precedence level, so
  conflicting booleans resolve to required evidence independent of aggregation
  order.
- Agent-evaluation policy claims are target-bound. A payload or summary for a
  different skill cannot waive evidence for the current report, including when
  that foreign Tier 3 result is a clean advisory skip.
- Every publication-contributing result is bound to one exact source snapshot
  through a versioned `publication_target`. The digest covers normalized
  author-owned paths, node kinds, executable bits, and file contents, including
  authored `evals/` inputs. The versioned recipe omits only its enumerated
  generated paths: any-depth `.git`, `.venv`, `__pycache__`, and `node_modules`;
  root generated state/result/version directories; `evals/results`; and the
  three generated root publication files. Filesystem aliases follow actual
  filesystem identity, and a same-named nested authored file remains covered.
  Producer scans and Tier 3 agent-visible projections share this exclusion
  contract. A missing, malformed, or different digest makes
  publication `INCOMPLETE` even when the visible target names match.
- Completed Tier 3 evidence carries its run-owned `run_id` in both the payload
  and summary. The runner persists the target identity at the execution
  boundary; reporters never backfill it from the current live path.
- Generic per-result `optional` metadata cannot waive a publication-required
  validator. Only the resolved `benchmark_policy` makes Tier 2 or Tier 3
  optional; Tier 1 is always required.
- Optional evidence that is present but malformed or incomplete cannot certify
  publication.

## Components and data flow

1. A shared publication assessment resolves `tier2_required` and
   `tier3_required`, validates Tier 1/Tier 2 execution evidence, classifies Tier
   3 completeness, and keeps publication status separate from process status.
   `BenchmarkReporter` renders that assessment and both requirements in
   Evaluation Metadata.
2. Tier classification uses one shared validator-name classifier so the
   benchmark and HTML reporters agree that names such as `Similarity Check` and
   `Context Optimization` belong to Tier 2.
3. `JSONReporter` preserves the resolved publication policy plus explicit
   `publication_status` and structured `publication` details so an external
   aggregator does not mistake a successful process gate for publication
   completeness. Per-result source identities remain available for split-job
   aggregation.
4. `check_public_benchmarks.py` parses all three tier rows. For PASS cards it
   requires completed Tier 1, completed Tier 2 unless explicitly optional, and
   completed Tier 3 unless explicitly optional. Missing Tier 2 policy metadata
   defaults to required so older contradictory cards fail closed.
5. Tier 1 and Tier 2 producers compare source snapshots before and after their
   checks. Tier 3 persists its source identity and run ID in `result.json` and
   propagates them through payload, summary, JSON, HTML, and BENCHMARK metadata.
   A source-change conflict propagates as a bounded generic marker instead of
   leaking raw path/digest details or disappearing during rerender.
6. Documentation and the changelog explain that execution selection, CLI exit
   gating, and publication completeness are separate contracts.

The downstream CI aggregator that produced the public cards is outside this
repository. This change gives that aggregator a fail-closed source/run binding;
deployment-specific artifact transport remains a separate integration proof.

## Alternatives considered

### Always require Tier 2 with no override

This is simple but removes the existing pattern of persisted publication-policy
exceptions and gives catalog owners no explicit migration path.

### Infer Tier 2 optionality from CLI selection

This preserves current partial-run PASS cards, but it also treats a split-tier
job as publication-complete. That is the failure described by issue #73.

### Explicit persisted required-tier policy

This is the selected design. It separates what ran, what affects the process
exit code, and what evidence publication requires.

## Error handling and compatibility

- Existing programmatic Tier 1 + Tier 3 calls change from PASS to INCOMPLETE
  unless they explicitly persist `tier2_required = false`.
- Existing cards without a Tier 2 policy line are interpreted as requiring Tier
  2. Good cards with completed Tier 2 remain valid; contradictory PASS cards do
  not.
- Existing `tier3_required` policy behavior remains unchanged.
- Legacy results without source identity and completed Tier 3 results without a
  run ID remain readable but cannot certify publication.
- A first-class CLI or YAML publication-policy option is intentionally out of
  scope. Metadata injection remains the orchestration escape hatch.

## Verification

- Prove the reporter regression red before production changes and green after.
- Cover missing Tier 1, required and optional Tier 2, skipped/incomplete Tier 2,
  invalid policy values, and Tier 2 validator-name classification.
- Cover the publication linter against required, optional, spoofed, and legacy
  cards.
- Exercise `validate` with `--no-dedup`, `--tiers 1,3`, and
  `--no-block-on-dedup` to show that CLI execution and exit semantics are
  preserved while the publication card fails closed.
- Verify generated BENCHMARK, JSON, and HTML outputs from the same result set.
- Merge split-tier results from one unchanged source as a positive control, then
  change an author-owned file between jobs and verify publication becomes
  `INCOMPLETE`.
- Run the focused suites, committed-card gate, Ruff, package build, and the full
  default test suite.
