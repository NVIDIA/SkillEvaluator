# SkillEvaluator

![SkillEvaluator wordmark](docs/assets/skillevaluator-wordmark.svg)

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)

SkillEvaluator is an open-source, multi-tier framework for evaluating AI agent
artifacts, starting with agent skills: deterministic quality gates, semantic
overlap detection, synthetic eval dataset generation, and live agent evaluation.

Agent skills are folders of instructions and supporting files that extend AI
agents, as defined by the [Agent Skills specification](https://agentskills.io/).
SkillEvaluator is part of the
[NVIDIA Verified Skills pipeline](https://docs.nvidia.com/skills/), with
[SkillSpector](https://github.com/NVIDIA/SkillSpector) providing the specialized
security-scanning capability used by Tier 1 and
[Harbor](https://github.com/harbor-framework/harbor) powering Tier 3 sandboxed
agent evaluation.

## Quickstart

Install all SkillEvaluator evaluation extras with
[uv](https://docs.astral.sh/uv/), then run a deterministic quality check. This
first result needs no API key, Docker daemon, or repository clone:

```bash
uv tool install --python 3.13 "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git"
skillevaluator quality-check ./my-skill
```

`./my-skill` is any directory containing a `SKILL.md`. The command returns a
0–100 quality score and an A–F grade; the default passing score is 70. If your
shell cannot find the command after installation, run `uv tool update-shell`
and open a new terminal.

## LLM provider setup

LLM-backed security analysis, rubric judging, Tier 2 deduplication, dataset
generation, and Tier 3 grading need a configured provider. For NVIDIA Build,
one API Catalog key covers both chat and embeddings:

```bash
export SKILL_EVAL_LLM_PROVIDER=nv_build
export NVIDIA_API_KEY='nvapi-...'
skillevaluator models --limit 10
```

Other supported provider setups are:

- OpenAI: `SKILL_EVAL_LLM_PROVIDER=openai` and `OPENAI_API_KEY`.
- Anthropic: `SKILL_EVAL_LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`.
- Amazon Bedrock: `SKILL_EVAL_LLM_PROVIDER=bedrock` plus the standard AWS
  credential chain and region.
- Local or hosted OpenAI-compatible endpoint: set
  `SKILL_EVAL_LLM_PROVIDER=openai-compatible`,
  `SKILL_EVAL_LLM_BASE_URL`, `SKILL_EVAL_LLM_MODEL`, and
  `SKILL_EVAL_LLM_API_KEY`.

When exactly one of `NVIDIA_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`
is present, SkillEvaluator can auto-select that provider. Anthropic and Bedrock
do not provide embeddings, so Tier 2 also needs a separate OpenAI, NVIDIA
Build, or OpenAI-compatible embedding provider. See
[Providers & Credentials](https://docs.nvidia.com/skills/skillevaluator/configuration)
for model defaults, endpoint overrides, and fully local setup.

## Run deeper evaluations

With a chat and embeddings provider configured, check one skill for repeated
guidance or compare a collection for semantic overlap:

```bash
skillevaluator context-optimization-check ./my-skill
skillevaluator similarity-check ./skills
```

For a live Tier 3 comparison, create or review an evaluation dataset, verify
the selected agent runtime, and run the with-skill and without-skill arms:

```bash
skillevaluator create-eval-dataset ./my-skill --full
skillevaluator doctor --agents codex --env-mode docker
skillevaluator tier3 evaluate ./my-skill --agents codex --env-mode docker \
  --n-attempts 1
```

Tier 3 requires the evaluator provider, the selected agent's credential, and a
Docker, local, or cloud sandbox. Live model calls and managed sandboxes can
incur charges; local mode avoids managed sandbox charges but not necessarily
hosted model charges. Start with one agent, a small dataset, and one attempt.
See the [Tier 3 guide](https://docs.nvidia.com/skills/skillevaluator/tier3-live-evaluation#plan-for-cost)
before scaling a run.

## Documentation

Read the complete documentation at
[docs.nvidia.com/skills/skillevaluator](https://docs.nvidia.com/skills/skillevaluator/)
for installation, the quickstart, provider configuration, tier guides, results
and CI integration, the CLI reference, and contributor guidance.

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
