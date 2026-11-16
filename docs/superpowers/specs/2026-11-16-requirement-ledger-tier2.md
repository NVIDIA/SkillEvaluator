# Requirement ledger: Tier 2

References checked against `main` at `2ec6364f`. Live observations are retained November 2026 runs from `f83c2edc`, not reruns at `2ec6364f`.

| # | Requirement | Code | Docs | Tests | Evidence | Status |
|---|---|---|---|---|---|---|
| T2 | Collection scans accept 1,024 entries by default with configurable entry and comparison limits | `constants.py` (`SIMILARITY_DEFAULT_MAX_ENTRIES = 1_024`, `SIMILARITY_MAX_ENTRIES = 5_000`), `embedding/extractor.py`, `embedding/registry.py`, `validators/similarity.py` | `tier2-deduplication.mdx` | `tests/embedding/test_similarity_limits.py`, `tests/test_tier2_command_failures.py`, `tests/embedding/test_pairwise_preflight.py` | Live validate run: `Context Deduplication` row is `PASS` with `3 checks passed`; the collection limit was not exercised live | Met (limit covered by tests) |
