# Requirement ledger: Tier 1

References checked against `main` at `2ec6364f`. Live observations are retained November 2026 runs from `f83c2edc`, not reruns at `2ec6364f`.

| # | Requirement | Code | Docs | Tests | Evidence | Status |
|---|---|---|---|---|---|---|
| T1 | Without an LLM key the semantic analyzers read as failed and the scan as incomplete; with `--no-llm` they read as disabled | `validators/security.py`, `skillspector_provenance.py`, `tier_workflows.py` | `tier1-validation.mdx` | `tests/validators/test_security.py`, `tests/validators/test_security_reference_caveats.py`, `tests/reporting/test_reporters.py`, `tests/cli/test_tier_workflows.py` | Live `tier1` on `main`: with no provider, `llm skipped: No LLM provider configured; set SKILL_EVAL_LLM_PROVIDER and its API key to include LLM checks.`; with `--no-llm`, `llm skipped: LLM checks disabled by --no-llm.` In both cases `Tier 1: 11 of 11 checks passed; 0 failed, 0 incomplete, 0 skipped, 2 N/A` and `Publication eligibility: Incomplete` | Partly met: the two causes are told apart in the reason text, but the Security Scan row reads PASS in both, and neither run counts as an incomplete check |
