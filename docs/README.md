<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# Documentation

Long-form documentation for SkillEvaluator. Pages are authored in MDX and
published to <https://docs.nvidia.com/skills/skillevaluator>. This page is the human
quickstart; [`AGENTS.md`](AGENTS.md) holds the detailed authoring rules (MDX
gotchas, link conventions, page-adding checklist) for both AI agents and
contributors making non-trivial edits.

## Layout

- `index.mdx` — landing page and three-tier overview
- `quickstart.mdx` — install, one provider and key, and first runs
- `installation.mdx` — extras, uv, pip, Docker, requirements
- `configuration.mdx` — providers, embeddings, and Tier 3 credential roles
- `tier1-validation.mdx` — checks, profiles, scanner setup, exit codes
- `tier2-deduplication.mdx` — dedup commands, catalogs, thresholds, limits
- `tier3-live-evaluation.mdx` — skill evaluation with live agents
- `eval-datasets.mdx` — the `evals/` contract, dataset generation, run policy
- `agents-and-sandboxes.mdx` — agents, environment backends, local mode, readiness
- `custom-graders.mdx` — bring-your-own graders and native Harbor tasks
- `reports.mdx` — report formats, scores, and the results-on-disk contract
- `benchmark-rollout.mdx` — regenerating and promoting `BENCHMARK.md` cards
- `ci-integration.mdx` — exit codes, GitHub Actions recipe, progressive adoption
- `cli-reference.mdx` — every command and flag
- `environment-variables.mdx` — variable-by-variable inventory
- `developer-guide.mdx` — contributor setup, tests, and docs workflow

Navigation order and slugs are defined in [`../fern/docs.yml`](../fern/docs.yml).

## Build Locally

You only need to build locally if you are editing the documentation. The site is
published automatically from `main` via the Fern GitHub integration.

Prerequisites: Node.js 22+ and npm 10+ (the versions the Fern CLI requires).

```bash
# Install the Fern CLI
npm install -g fern-api

# From the repo root, preview the site with live reload
fern docs dev

# Validate the docs configuration and links
fern check
```

`fern docs dev` serves the site at <http://localhost:3000> and reloads on changes
to `.mdx` files or `fern/docs.yml`.

## Authoring Notes

- New pages must be added to the `navigation` section of
  [`../fern/docs.yml`](../fern/docs.yml); otherwise they will not appear in the
  site.
- Use relative links between MDX pages (`./configuration.mdx`) and relative repo
  links (`../README.md`) for files outside `docs/`.
- The Fern version is pinned in
  [`../fern/fern.config.json`](../fern/fern.config.json); bump it intentionally
  rather than as a side effect.
