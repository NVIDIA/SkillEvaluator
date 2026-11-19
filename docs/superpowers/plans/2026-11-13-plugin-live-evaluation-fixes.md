# Plugin Live Evaluation Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct evaluator defects reproduced while running real plugin evaluations through PR 394.

**Architecture:** Keep scanner provisioning and report projection separate. Automatic scanner discovery must honor the pinned release; explicit operator overrides remain explicit. Report partial execution without converting missing evidence into a content failure or a missing-install claim.

**Tech Stack:** Python, pytest, existing scanner wrappers and Rich/report projections.

**Spec:** User request to run real three-tier plugin evaluations, retain HTML, summary, CLI logs and benchmark artifacts, and fix encountered evaluator bugs in PR 394.

## Global Constraints

- Preserve original plugin content, authored datasets and genuine findings.
- Never replace an explicit scanner executable override silently.
- Keep incomplete analysis fail-closed and distinguish it from missing installation.
- Keep private evaluation sources, credentials and raw artifacts outside the public repository.
- Run pytest with a dedicated temporary directory to avoid unrelated historical cleanup.

## Review Focus

- An older or unverifiable automatically discovered scanner cannot silently bypass the pin.
- A matching installed scanner does not require provisioning or network access.
- Explicit executable overrides remain usable and auditable.
- Partial scanner execution is reported with its actual cause and retained findings.
- Known content failures and independent score gates remain visible beside missing evidence.

### Task 1: Honor the automatic SkillSpector release selection

**Files:** `src/skillevaluator/utils/tool_runner.py`, `tests/utils/test_tool_runner.py`, `tests/utils/test_scanner_install.py`.

- [x] Reproduce automatic discovery selecting an older scanner instead of the pinned managed release.
- [x] Add failing tests for old/unverifiable discovery, matching installation, explicit override and unavailable provisioning.
- [x] Implement the smallest selection/diagnostic correction without changing unrelated scanners.
- [x] Run the focused tool and installer regressions and inspect the diff.

### Task 2: Preserve incomplete execution in reports

**Files:** `src/skillevaluator/reporting/console_ui.py`, `src/skillevaluator/reporting/plugin_static.py`, relevant reporting tests.

- [x] Reproduce an executed partial scanner being described as not installed.
- [x] Add failing tests for partial analysis, actual missing/install failure, incomplete-only plugin projection and genuine blocking findings.
- [x] Correct diagnostics and status projection while retaining fail-closed gates.
- [x] Run relevant console, plugin projection and report regressions.

### Task 3: Reject unsupported CI modes before optional runtime imports

**Files:** `ci/bootstrap_tiers.py`, `tests/ci/test_all_tier_preflight.py`.

- [x] Reproduce the unnecessary optional-import path with an unsupported CI environment.
- [x] Add a failing poisoned-import regression without increasing the existing subprocess timeout.
- [x] Move only the mode-dependent imports after validation and run the CI bootstrap regressions.

### Task 4: Preserve measured provider usage across protocol translation

**Files:** `src/skillevaluator/tier3/harbor/nvidia_build_bridge.py`, focused bridge tests.

- [x] Reproduce provider-reported counts being discarded by the Anthropic and Responses translators.
- [x] Propagate valid reported counts without estimating unavailable measurements or cost.
- [x] Verify protocol schema requirements, malformed accounting and missing-usage behavior.
- [ ] Keep the current trials' original artifacts unchanged and visibly qualify unavailable usage in delivered reports.

### Task 5: Preserve local downloads on macOS sources with extended attributes

**Files:** `src/skillevaluator/tier3/harbor/local_environment.py`, `tests/test_harbor_local_environment.py`.

- [x] Reproduce generated AppleDouble tar entries being mistaken for real source files.
- [x] Add failing native macOS regressions for both tar-based transfer paths, including a real user-authored `._` file.
- [x] Disable synthesized metadata in the tar environment without filtering source names or weakening containment.
- [x] Verify the remaining local environment transfer and containment contracts.

### Task 6: Reconcile optional catalog comparison states

- [x] Diagnose no-catalog comparisons counted as passing by the CLI but missing execution evidence in the benchmark.
- [x] Preserve honest not-run states without inventing comparison success or weakening publication requirements.
- [x] Add focused regression coverage and retain the original live-run evidence.
- [x] Correct false-passing aggregate context work-limit stops and audit the other context early returns without relaxing their bounds.

### Task 7: Integrate the updated base without losing applicable checks

- [x] Merge the updated base and preserve both changelog sections.
- [x] Confirm locked Click generation leaves the intentional CLI snapshot unchanged.
- [x] Share the optional catalog policy with rendered benchmark tier tables.
- [x] Record explicit native-plugin harness applicability without inventing native compatibility evidence.
- [x] Retain real standalone harness execution evidence after completed profile evaluation, including clean results.

### Task 8: Verify and retain real evaluation evidence

- [ ] Run both samples through the public three-tier workflow with immutable source provenance and explicit runtime settings.
- [ ] Keep original reference bundles and any contained materializations distinct.
- [ ] Retain `report.html`, a results summary, CLI logs and `BENCHMARK.md` for each evaluated subject.
- [ ] Run lint, appropriate regression checks and the default suite with isolated temporary storage; record all failures or environmental limits.
- [ ] Review, sign off and push verified evaluator fixes to the existing PR branch.
