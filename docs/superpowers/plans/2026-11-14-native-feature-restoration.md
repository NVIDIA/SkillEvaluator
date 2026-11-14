# Native report feature restoration

> **For agentic workers:** use superpowers:subagent-driven-development; record task evidence in the ignored execution ledger.

**Goal:** address the user's PR #560 follow-up by restoring visible native GitLab report features, then update the same branch and PR.

**Design:** the pinned 1.13.1 template remains the presentation reference. Keep its header scale, version identification, consolidated human-readable dimension/baseline comparison, partial comparison annotations, clickable agent scorecards and Harbor actions. Retain the earlier concise Insights and closed routine provenance; reducing clutter must not remove useful native views or actions. The user's explicit follow-up supersedes the compact header and removed scorecards in the previous revision.

**Architecture:** existing assessed display data feeds the single HTML template. Add only bounded, typed display projections for unavailable configured dimensions and custom metric comparisons. Repair Harbor fallback in the existing URL-normalization helper. Canonical payloads, scores, gates, supported backends and CLI behavior stay unchanged.

**Reference:** presentation commit `75c5a82e3bb778edfed37012f603479347f3d205`; public adaptation manifest `docs/report-presentation-port.json`.

## Constraints and review focus

- Restore the native consolidated comparison and Overall row, using recorded public score/lift authorities. Never recompute scores or substitute the internal scoring formula.
- Incomplete runs show configured standard dimensions with unavailable values. Custom-only evaluations show their recorded metrics, never a fabricated five-dimension score grid.
- Preserve missing/zero/not-applicable, partial comparison explanations, failed-agent scopes, source rejection, activation limits and baseline eligibility.
- Show the renderer's SkillEvaluator version even without a timestamp, and only recorded SkillSpector versions. Keep model IDs, paths and routine provenance quiet.
- Harbor actions use existing validated URLs, preserve all recorded jobs/run analysis, work by keyboard inside tier cards and never infer private viewer URLs.
- Keep native CSS declarations intact; remove compact overrides and retain only necessary responsive/accessibility additions. Allow the larger header without hiding report content behind sticky chrome.
- Keep escaping/redaction, finite numbers, URL validation, bounds, exact retained downloads, offline assets, no-JS, print, search and deep links intact.
- No fresh providers, evaluations or rescoring; private source reports and screenshots remain ignored.

## Implementation tasks

- [ ] Add failing display-adapter tests, then restore unavailable standard rows, typed partial comparisons and recorded custom metric comparison rows in `reporting/html.py`.
- [ ] Add failing Harbor helper tests, then repair empty/unsafe primary viewer masking a usable recorded summary viewer; preserve valid primary precedence and canonical immutability.
- [ ] Add presentation tests, then restore native header/version labels, tier-card and Overview Harbor actions, consolidated dimension/Overall/other-agent table and native scorecards. Restore generic Dataset/plugin labels. Preserve the old consolidated-view deep link.
- [ ] Audit the remaining native capabilities across every tier and Tier 3 tab; fix supported presentation regressions and document contract-specific adaptations.
- [ ] Extend installed-wheel Chromium acceptance for actual visibility, baseline cells, partial annotations, native header scale, versions, nested link keyboard behavior and mobile/print geometry. Inspect the existing PR CI failure and fix any branch regression.
- [ ] Run relevant tests, lint, package/source equality and OSS checks; replay retained inputs offline; regenerate both supplied reports and comparable screenshots in a new ignored audit directory.
- [ ] Independently review the final diff, commit with sign-offs, push the same branch and update PR #560's structured BEFORE/AFTER description and evidence. Do not merge.

## Verification

Use synthetic fixtures for tests. Exercise successful two-agent, incomplete, mixed, custom-only, missing-baseline, missing-telemetry and skipped shapes; malformed comparison numbers and viewer URLs; single/multiple/summary-only viewer links. Run every pytest invocation with a fresh explicit basetemp. Verify installed wheel outside the checkout. Preserve the two supplied coverage totals of 28/36 and 12/20 and exact canonical downloads.
