<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# Synthetic grader qualification pilot 1.0.0

Four public, synthetic ATIF assessment cases add matched security and Efficiency
challenges without changing the original reference suite. The security pair
contrasts a forbidden agent Write to `/root/.bashrc` with the same unsafe text
quoted in a read-only observation. The Efficiency pair contrasts a wasteful root
search with a permitted search within the supplied calculator directory.
All apparent commands and writes in trajectory JSON are inert fixture data.

This is a **diagnostic availability pilot**. It does not establish all-metrics
qualification, statistical representativeness, release approval or Jira
acceptance. One distinct pass/fail case per target metric is an availability
floor. Repeats add observations, not independent cases. No category minimum is
claimed. Do not pool these rates with another suite or treat the targeted policy
as the experimental all-metrics policy passing.

Codex authored and self-reviewed the labels on 2026-11-13. Independent Codex
subagent `/root/review_pilot_pr501` confirmed all four target metric and task
Overall labels in [PR #501 review](https://github.com/NVIDIA/SkillEvaluator/pull/501#pullrequestreview-5891138481):
Spec PASS, Quality PASS, no actionable findings. This automated review was posted
through GitHub account `rng1995`; no human maintainer approval is claimed.
Read [ORACLE.md](ORACLE.md) before using this fixture: task Overall and
metric-protocol predictions have different meanings. The two prohibited actions
remain task failures even when the composite Overall score passes. Only each
pair's independently supported target metric is labelled; omitted metrics are
not N/A labels.

## Reproduce offline

From the repository root, with the project environment installed:

```bash
skillevaluator tier3 validate-grader examples/grader-qualification-pilot --check-suite --qualification-policy examples/grader-qualification-pilot/qualification-policy.json
python -m pytest -q tests/tier3/test_grader_pilot.py
```

The CLI uses the actual suite loader and executable evidence checks, then checks
expected label coverage without providers or model calls. `ready` means labels
are available for the two-metric policy; it is not measured agreement. The tests
exercise actual loader/policy coverage, negative-label loss, removal of the
unsafe action, full freeze identity and current deterministic components. Those
components are supplemental conformance checks, not the label oracle.

## Freeze identity

`manifest.json` records this eight-file freeze (suite, four trajectories, policy,
worksheet and this README), with raw UTF-8 byte lengths and SHA-256 values. Files
are each bounded to 256 KiB. The manifest itself is bounded to 64 KiB and is
excluded from its own file map. `manifest_payload_sha256` hashes the entire
manifest except that field, serialized as UTF-8 JSON with sorted keys, compact
separators, ASCII escaping and nonfinite values disallowed. Source identity is
the exact base Git commit and hashes of the shipped verifier files plus the
suite loader, policy evaluator and score constants, not a claim that future
source is identical. A rerun must retain its actual tracked source identity.

The built-in suite digest frames each filename, NUL, eight-byte big-endian file
length and contents for `suite.json` followed by sorted referenced trajectories.
It covers all referenced cases. It does not cover this worksheet, README or
policy. The separate canonical policy hash and full manifest payload hash
cover those identities. These hashes identify content; they do not grant review
approval. A future version must regenerate the manifest after reviewed changes.

Verify the freeze with standard-library Python from the repository root:

```python
import hashlib
import json
from pathlib import Path

root = Path("examples/grader-qualification-pilot").resolve()
manifest_path = root / "manifest.json"
assert manifest_path.is_file() and not manifest_path.is_symlink()
assert manifest_path.stat().st_size <= 64 * 1024
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert len(manifest["files"]) == 8
for name, identity in manifest["files"].items():
    path = root / name
    assert path.resolve().is_relative_to(root)
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size == identity["bytes"] <= 256 * 1024
    assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]
payload = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False).encode("utf-8")
assert hashlib.sha256(canonical).hexdigest() == manifest["manifest_payload_sha256"]
```

## Bounded live use

After independent automated review and explicit provider/run authorization,
retain a fresh output directory and the actual source, model, policy and suite
identities. The approved diagnostic cohort is three repeats, concurrency two,
45 seconds per judge, zero transport retries and 150 seconds per attempt,
within a 1200-second outer wall bound. Configure those judge settings through
the supported provider configuration; a shell timeout alone is not a judge
timeout or accounting limit. No paid run is needed to validate these files.

```bash
skillevaluator tier3 validate-grader examples/grader-qualification-pilot --split assessment --repeats 3 --concurrency 2 --timeout 150 --qualification-policy examples/grader-qualification-pilot/qualification-policy.json -o pilot-qualification
```

Keep the original 14-case assessment receipt separate. Preserve failures and
unjudgeable observations, actual usage and missing accounting. A passed targeted
gate still leaves mandatory Overall disagreements visible and makes no release
or Jira acceptance decision. Do not tune a scorer against these assessment cases
and silently call the rerun an independent holdout.
