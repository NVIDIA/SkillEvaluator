# Installation

The quickest install is the one-liner from the
[README quickstart](../README.md#quickstart) — everything on this page is for
tailoring it: smaller installs, plain pip, and Docker.

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) for the
  recommended install paths (plain pip works too — see below).
- Python 3.12 or 3.13. `uv tool install` and `uv sync` provision a supported
  interpreter automatically when passed `--python 3.13`; only the plain-pip
  path needs one on PATH.
- Git (source installs and the pinned
  [SkillSpector](https://github.com/NVIDIA/SkillSpector) dependency are
  fetched from GitHub).
- Docker, only for the container install and Tier 3 Docker evaluation.
- No API key is needed for deterministic Tier 1 checks. Full default scanner
  coverage requires the `security` extra plus Gitleaks. LLM-backed checks and
  Tiers 2–3 use a configured provider — see
  [CONFIGURATION.md](CONFIGURATION.md).

## Choosing extras

The base package runs schema, PII, license, quality, Unicode, script, and
hygiene checks. Scanner-backed default checks require an extra or external
binary:

| Extra | Capability |
| --- | --- |
| `llm` | Shared LLM and embedding clients |
| `tier2` | Intra-skill deduplication and local-catalog inter-skill similarity; includes `llm` |
| `tier3` | Docker and cloud live-agent evaluation; includes `llm` |
| `telemetry` | Optional OpenTelemetry export |
| `security` | Bandit, Semgrep, pip-audit, and the pinned SkillSpector Git dependency |
| `all` | `tier2`, `tier3`, `telemetry`, and `security` |
| `dev` | Build, test, coverage, and formatting tools |

Swap the extras in the install one-liner:

```bash
uv tool install --python 3.13 "skillevaluator @ git+https://github.com/NVIDIA/SkillEvaluator.git"            # base checks; scanner checks report INCOMPLETE
uv tool install --python 3.13 "skillevaluator[security] @ git+https://github.com/NVIDIA/SkillEvaluator.git"  # + security scanners
uv tool install --python 3.13 "skillevaluator[tier2,tier3] @ git+https://github.com/NVIDIA/SkillEvaluator.git"
```

The `security` extra installs SkillSpector directly from GitHub at the
immutable revision pinned in `pyproject.toml`. It is substantially larger than
the base install because Semgrep and SkillSpector bring their own dependency
stacks; use `all` only when Tier 2/Tier 3 and telemetry are also needed.

## Install from source

For development or an editable environment, clone the repository and let uv
create the environment from the committed lockfile:

```bash
git clone https://github.com/NVIDIA/SkillEvaluator.git
cd SkillEvaluator
uv sync --python 3.13 --all-extras
```

Or with plain pip (requires Python 3.12 or 3.13 on PATH — if your system
`python3` is newer, use `python3.12 -m venv .venv` instead, or prefer the
`uv sync` path above, which provisions a supported Python automatically):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
```

Editable extras work the same way as the one-liner:

```bash
python -m pip install -e .
python -m pip install -e ".[security]"
python -m pip install -e ".[all,dev]"
```

## Run with Docker

Build the image locally, inspect the CLI, and run a deterministic validation
against the included sample skill:

```bash
docker build -t skillevaluator:local .
docker run --rm skillevaluator:local --help
docker run --rm \
  -v "$PWD/tests/fixtures/skills/simple:/workspace/skills/simple:ro" \
  skillevaluator:local validate /workspace/skills/simple --no-llm --no-dedup
```

## Verify the installation

```bash
skillevaluator --version
skillevaluator --help
```

The `gitleaks` secrets scanner is a required external Go binary that pip cannot
install. When `code-integrity` runs without it, validation is `INCOMPLETE` and
returns non-zero. Install it separately: `brew install gitleaks` (macOS) or
download a binary from the official
[Gitleaks releases](https://github.com/gitleaks/gitleaks/releases).
