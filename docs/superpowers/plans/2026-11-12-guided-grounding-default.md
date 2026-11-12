# Guided Grounding Default Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this approved scoped change.

**Goal:** Default standalone generation to author intent or matching retained evidence and make skill-only generation explicit.

**Architecture:** Preflight auto selection before generation. Reuse existing generation/refinement and provenance, preserve explicit modes and unattended autopilot.

**Tech Stack:** Python, argparse, Click, pytest, Fern MDX.

**Spec:** `docs/superpowers/specs/2026-11-12-guided-grounding-default-design.md`

## Global Constraints

- Base exactly on PR456 head `62b0cf92cc919499b5881996fcb9c0f717712cae` in this owned scratch clone.
- No provider calls, external writes, scoring defaults, or schema changes.
- Preserve authored cases and require review for observed answers and unmatched cases.

## Review Focus

- Empty guidance and missing/foreign traces must fail before writes or provider access.
- Retained trace IDs must match preserved authored cases or deterministic seeds.
- Auto plus refine must not collect its own missing preflight evidence.
- CLI, service, and standalone generator defaults must agree.
- Explicit skill generation and unattended autopilot must remain available.

### Task 1: Automatic grounding selection and compatibility

**Files:** generator, CLI, evaluation options, tier3 commands; grounding/service/CLI tests.

- [x] Add failing tests for intent priority, usable matching retained trace fallback, missing/empty/foreign/unmatched evidence, no side effects before failure, refine interactions, effective provenance, explicit skill, and service/CLI parity.
- [x] Run those tests against the base and record expected failures.
- [x] Implement preflight and aligned auto defaults; reuse existing refinement with deterministic seeds and review flags for unmatched cases.
- [x] Run focused generator, fairness, service, and CLI regression tests; update legacy expectations with explicit skill opt-in where appropriate.

### Task 2: Migration documentation and verification

**Files:** CLI reference, eval datasets, README, skill reference, CHANGELOG, CLI golden surface.

- [x] Document default precedence, no-input recovery, scope, explicit compatibility invocation, refine/overwrite interactions, and effective provenance.
- [x] Regenerate CLI golden surface and inspect its targeted diff.
- [x] Run Ruff, focused and default pytest suite, and pinned Fern check; report exact outcomes and limitations.
- [x] Commit only owned files with DCO sign-off and return SHA/report to parent without pushing.

Verification and limits are recorded in `/tmp/se-guided-package-20261112/GUIDED_GROUNDING_REPORT.md`. Runtime source was frozen before final installed-wheel gates; complete default-suite success is not claimed. Parent manages independent review and external publication.
