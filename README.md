# SkillEvaluator

![SkillEvaluator wordmark](docs/assets/skillevaluator-wordmark.svg)

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/Documentation-docs.nvidia.com-blue.svg)](https://docs.nvidia.com/skills/skillevaluator/)
[![Paper](https://img.shields.io/badge/arXiv-2608.20614-b31b1b.svg)](https://arxiv.org/abs/2608.20614)
[![NVIDIA Developer Blog](https://img.shields.io/badge/Blog-NVIDIA%20Developer-76B900.svg)](https://developer.nvidia.com/blog/evaluating-ai-agent-skill-performance-with-nvidia-skillevaluator/)

SkillEvaluator is an open-source, multi-tier framework for evaluating AI agent
artifacts, starting with agent skills: deterministic quality gates, semantic
overlap detection, synthetic eval dataset generation, and live agent evaluation.

Agent skills extend AI agents with instructions and supporting files, as
defined by the [Agent Skills specification](https://agentskills.io/).
SkillEvaluator is part of the
[NVIDIA Verified Skills pipeline](https://github.com/NVIDIA/skills).
Skills that pass are published to the
[NVIDIA skills catalog](https://github.com/NVIDIA/skills).

> **Research foundation.** Tier 3 implements methods from
> [*Evaluating Skills, Not Just Agents: Agentic Continuous Evaluation of Skills*](https://arxiv.org/abs/2608.20614):
> paired with/without-skill trials and Skill Lift; SkillEvaluator provides
> validation, deduplication, dataset authoring, and reporting.

## Three-tier overview

![SkillEvaluator three-tier pipeline: Skill → Tier 1 Validation → Tier 2 Deduplication → Tier 3 Live Evaluation → Reports](docs/assets/three-tier-overview.svg)

Tiers are independent entry points; nothing requires running earlier ones first.

| Tier | Purpose | Run one tier | Requires |
| --- | --- | --- | --- |
| Tier 1: Validation | Safe & well-formed? | `skillevaluator tier1 ./my-skill` | External scanners for full coverage; a provider enables rubric and LLM security checks |
| Tier 2: Deduplication | Repeated or overlapping guidance? | `skillevaluator tier2 ./my-skill` | Embeddings and chat; add `--catalog ./skill-catalog.json` for inter-skill comparison |
| Tier 3: Live Evaluation | Does it help the agent? | `skillevaluator tier3 ./my-skill` | Provider access, a supported agent, and a running sandbox (Docker by default) |

[SkillSpector](https://github.com/NVIDIA/SkillSpector) provides Tier 1 security
scanning. [Harbor](https://github.com/harbor-framework/harbor), the open-source agent evaluation framework, powers Tier 3
sandboxed agent runs. Full tier guides live in the
[documentation](https://docs.nvidia.com/skills/skillevaluator/).

## Quickstart

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install --python 3.13 "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git"
```

**To configure NVIDIA Build for all three tiers, set just two variables: the
provider and its API key.** Get a key from [build.nvidia.com](https://build.nvidia.com):

```bash
export SKILL_EVAL_LLM_PROVIDER=nv_build
export NVIDIA_API_KEY='nvapi-...'
```

The same key covers chat, Tier 2 embeddings, and the supported Tier 3 agents in
Docker or local mode. Models and embedding providers have configured defaults.

Run all three tiers. `./my-skill` is a directory containing a `SKILL.md`:

```bash
skillevaluator validate ./my-skill
```

Tier 3 uses autopilot to prepare the dataset and evaluate with and without
the skill. Runtime requirements are below.

To run one tier independently:

```bash
skillevaluator tier1 ./my-skill
skillevaluator tier2 ./my-skill
skillevaluator tier3 ./my-skill
```

Each command runs its tier directly:

- **Tier 1** runs static checks, including dependency checks, plus rubric,
  LLM security analysis, and finding verification when a provider is configured.
  Without a provider it runs static checks and explicitly reports the LLM stages
  as skipped. Use `--no-llm` to opt out of LLM checks.
- **Tier 2** checks for repeated content inside the skill. Add
  `--catalog ./skill-catalog.json` to compare against other skills too; without
  a catalog, the output explicitly says inter-skill comparison did not run.
- **Tier 3** creates one starter evaluation case if no dataset or task source
  exists, then runs the provider-native agent in Docker with and without the
  skill. Valid existing sources are reused unchanged; invalid ones produce an
  error. A starter case is a first run, not comprehensive evaluation coverage.

A tier without a path shows help.
Expert commands such as `tier1 security-scan`, `tier2 similarity-check`, and
`tier3 create-eval-dataset` remain available.

### Runtime requirements

`[all]` installs Python extras. Also install Semgrep, SkillSpector, and Gitleaks
for full Tier 1 coverage, and an agent runtime and sandbox for Tier 3. Follow the
[installation guide](https://docs.nvidia.com/skills/skillevaluator/installation),
then check Tier 3 readiness:

```bash
skillevaluator doctor --env-mode docker
```

For a quick deterministic check before installing external scanners, run
`skillevaluator quality-check ./my-skill`; no API key is needed. If the shell
cannot find the command, run `uv tool update-shell` and open a new terminal.

### Other providers and optional model overrides

OpenAI also starts with two variables and uses the default Codex agent for Tier 3:

```bash
export SKILL_EVAL_LLM_PROVIDER=openai
export OPENAI_API_KEY='sk-...'
```

NVIDIA Build defaults to `nvidia/nemotron-3-super-120b-a12b` for evaluator chat,
Tier 3 agent execution, and judging. Its embedding default is
`nvidia/nemotron-3-embed-1b`; OpenAI defaults to `gpt-5.6-sol` and
`text-embedding-3-small`. These are configured defaults, not automatically
updated selections of the latest models. To select a different evaluator model:

```bash
export SKILL_EVAL_LLM_MODEL='your-provider-model-id'
```

The standard judge inherits that model unless overridden. Tier 3 defaults to
OpenCode for NVIDIA Build, Codex for OpenAI, and Claude Code for Anthropic.
Use `--agents` and `--agent-model` to override the agent and its model separately.

Anthropic and Bedrock need a separate embedding provider for Tier 2. Custom
OpenAI-compatible endpoints also require explicit endpoint and model settings.
See [Providers & Credentials](https://docs.nvidia.com/skills/skillevaluator/configuration)
for these setups and advanced overrides.

## Run the combined pipeline

`validate` runs Tier 1, Tier 2, and Tier 3 with autopilot by default for skills:

```bash
skillevaluator validate ./my-skill
```

`--full` remains supported but is unnecessary.
Use `--tiers 1,2` or `--no-tier3` to skip live evaluation; `--tiers 1` runs
only the static suite. `--no-autopilot` requires an existing evaluation source.
Rules, workflows, and plugins retain their applicable validation stages without
automatically enabling skill evaluation.

To prepare broader coverage before a run, generate and review a four-bucket dataset:

```bash
skillevaluator tier3 create-eval-dataset ./my-skill --full
```

Tier 1 always gates `validate`. Tier 2 gates by default;
`--no-block-on-dedup` makes its findings advisory. Tier 3 is advisory by default;
`--block-on-agent-eval` makes its findings gate too. Live model calls and managed
sandboxes can incur charges; start with one agent and a small dataset. See the
[Tier 3 guide](https://docs.nvidia.com/skills/skillevaluator/tier3-live-evaluation#plan-for-cost)
for runtime and cost details.

## Run from a source checkout

```bash
git clone https://github.com/NVIDIA/SkillEvaluator.git
cd SkillEvaluator
uv sync --python 3.13 --extra all
uv run --extra all skillevaluator tier1 ./my-skill
```

Use the provider exports above. Prefix other commands with
`uv run --extra all` to select this checkout.

## Research and citation

Resources: [paper](https://arxiv.org/abs/2608.20614),
[PDF](https://arxiv.org/pdf/2608.20614),
[DOI](https://doi.org/10.48550/arXiv.2608.20614),
[BibTeX](https://arxiv.org/bibtex/2608.20614), and the
[NVIDIA Developer blog](https://developer.nvidia.com/blog/evaluating-ai-agent-skill-performance-with-nvidia-skillevaluator/).
Cite the paper for methodology and results. For reproducibility, record the
SkillEvaluator release or commit. See [CITATION.cff](CITATION.cff).

## Documentation

See [the documentation](https://docs.nvidia.com/skills/skillevaluator/) for
setup, tier guides, reports, CI integration, and the CLI reference.

## Installation and third-party software

Follow the [installation guide](https://docs.nvidia.com/skills/skillevaluator/installation)
to choose the full installation or a smaller per-tier setup.

This project will download and install additional third-party open source
software projects. Review the license terms of these open source projects before
use.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), include tests
for behavior changes, and run the checks before opening a pull request:

```bash
make lint && make test && make build
```

Project governance is described in [GOVERNANCE.md](GOVERNANCE.md). Participation
is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Support

Support level: **Experimental**. SkillEvaluator is community-supported on a
best-effort basis with no SLA or NVIDIA enterprise support entitlement. Report
reproducible bugs and feature requests through
[GitHub Issues](https://github.com/NVIDIA/SkillEvaluator/issues); see
[SUPPORT.md](SUPPORT.md) for details.

## Security

Report suspected vulnerabilities using the private process in
[SECURITY.md](SECURITY.md). Do not disclose security issues in a public GitHub
issue.

## Releases

Release changes are recorded in [CHANGELOG.md](CHANGELOG.md) and
[GitHub Releases](https://github.com/NVIDIA/SkillEvaluator/releases).

## License

Apache License 2.0 — see [LICENSE](LICENSE), [NOTICE](NOTICE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
