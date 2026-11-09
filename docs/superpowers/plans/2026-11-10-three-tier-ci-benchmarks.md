# Three-tier CI gateways and benchmarks implementation plan

> **For agentic workers:** Use test-driven changes and independent review. Check off completed tasks; keep provider credentials out of reports and source control.

**Goal:** Provide thin consumer CI that runs LLM-enriched Tier 1, Tier 2 deduplication, paired Claude Code/Codex Tier 3 through a configurable gateway, and publishes source-bound benchmark cards alongside combined reports.

**Architecture:** Extend the shared orchestration and existing native reporting APIs. Keep native agent models separate from evaluator/judge models. Generate benchmark cards from retained typed evidence without model calls or executing consumer code in the publisher.

**Tech stack:** Python, GitHub Actions, GitLab CI, existing SkillEvaluator reporters and provenance validation.

**Spec:** User-approved three-tier onboarding plan in this chat, with the subsequent requirements for Tier 1 `--llm`, gateway-backed Claude/GPT agents, the reference pipeline's separate evaluation/report/publication stages, and `BENCHMARK.md` publication.

## Global constraints

- Only Tier 1 blocks CI. Tier 2/3 remain advisory in every report and adapter.
- Tier 3 supports Docker and local, with complete treatment/baseline coverage for every selected agent/case.
- Preserve two thin consumer entry files; no copied scripts, bundled wheels, or consumer-built images.
- Keep generic gateway settings in OSS; no internal endpoint, model entitlement, or credential defaults.
- Keep keyless scans for forks and unauthorized/stale commits. Authorize and recheck provider-backed work before exposing credentials.
- Retain reports for 14 days, including failed/incomplete evidence. Never derive publication eligibility from the advisory CI gate.
- Pin evaluator/runtime/workflow/tool revisions. Validate candidate orchestration separately from claims about released/latest-main runtime support.
- Never send a Slack message or manually rewrite a bot result comment.

## Review focus

- Missing or contradictory Tier 1 required-check declarations must survive JSON round trips and remain incomplete.
- Missing tier/agent/baseline evidence must never become a passing benchmark.
- Multiple skills and reordered tier rows must compose by exact skill identity, not ordinal.
- Generic gateway keys/base URLs must reach only the selected provider and never inherit CI feedback credentials.
- Benchmark publication must reject stale/fork/foreign evidence and must not cause an evaluated source PR to update itself in a loop.

### Task 1: Preserve publication evidence through retained JSON

Files: `src/skillevaluator/reporting/json_reporter.py`, `src/skillevaluator/reporting/retained.py`, reporting/publication regression tests.

- [ ] Reproduce loss of `tier1_required_checks` and `tier1_unscanned_paths` through JSON -> retained results.
- [ ] Preserve typed bounded string lists; reject malformed declarations.
- [ ] Verify in-memory and retained publication outcomes agree for complete, omitted-check and unscanned-content cases.

### Task 2: Compose combined skill reports and benchmark cards

Files: new `ci/benchmark_reports.py`, new `tests/ci/test_benchmark_reports.py`; root integrates `ci/run_tiers.py` and retained index/feedback contracts.

Interface: `write_benchmark_reports(reports: Path, manifest: dict) -> list[dict]`; output rows identify exact skill path, combined JSON, and `BENCHMARK.md` paths.

- [ ] Test real retained Tier 1/2/3 composition, reordered/multiple skills, missing tiers, source conflicts, and advisory failures.
- [ ] Reuse `retained_results`, `JSONReporter` and `BenchmarkReporter`; never read skill source or call a model while reporting.
- [ ] Preserve source/policy/run provenance. Keep required tiers publication-required regardless of CI gate policy.
- [ ] Retain honest non-PASS cards for incomplete evidence, and expose cards beside the existing combined report.

### Task 3: Support configurable gateway routing and per-tier models

Files: new `ci/model_config.py`, `ci/run_tiers.py` environment plumbing, `ci/bootstrap_tiers.py`, dedicated model-config/bootstrap tests.

Interface: `model_environment(env: dict[str, str], tier: int) -> dict[str, str]` provides only safe runtime settings and selected credentials. `child_environment(env, tier=2)` retains its existing API with a compatible optional tier argument.

- [ ] Add generic `openai-compatible` chat/embedding support and explicit API root/key settings.
- [ ] Add Tier 3 provider/model/base-URL overrides while preserving existing provider defaults and separate native agent model defaults.
- [ ] Use the same sanitized selection for doctor and evaluation; preserve selected embedding credentials independently.
- [ ] Test absent keys, malformed URLs, provider isolation, empty/inherited overrides, and absence of CI/publisher credentials.

### Task 4: Wire shared workflows, onboarding and Tier 1 LLM

Files: shared GitHub/GitLab workflows, CLI/template/descriptor helpers, workflow/onboarding tests.

- [ ] Expose opt-in Tier 1 LLM and generic gateway/per-tier inputs without invalidating old pinned callers.
- [ ] Reuse the hardened Tier 1 image for LLM/SkillSpector execution with protected admission and a current-head recheck; retain keyless baseline for denied admission.
- [ ] Keep native Tier 2/3 independent and advisory, preserve failure diagnostics and source-attempt reuse.
- [ ] Integrate benchmark generation in credential-free aggregation and trusted feedback links.
- [ ] Implement the chosen benchmark publication surface; only a trusted publisher may propose source-file updates from validated evidence.

### Task 5: Documentation, review and live validation

Files: existing onboarding, GitHub/GitLab, reports and benchmark guides; changelog.

- [ ] Explain local setup -> prerequisites -> merge setup -> test PR -> reports/cards, including separate chat/judge/agent settings and gateway APIs.
- [ ] Run focused tests, Ruff, formatting, benchmark lints and Fern checks.
- [ ] Review each task and the final diff; fix findings before opening the implementation PR.
- [ ] Exercise candidate configuration with real provider credentials in dedicated fixtures, verify native completion and exact model routing, and report any remaining release/runtime dependency honestly.

