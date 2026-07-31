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
