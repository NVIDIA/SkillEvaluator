# Configuration

## What needs credentials, and what runs offline

Tiers describe evaluation depth; credentials are a separate axis. This is
the complete map — everything not listed under "needs a provider" runs fully
offline:

| Runs with no credential at all | Needs a provider key |
| --- | --- |
| `validate --no-dedup` (all deterministic checks: schema, security scanners, PII, license, secrets, code-integrity, unicode, quality, lint) | `rubric-eval` — LLM-as-judge scoring (chat LLM) |
| `quality-check`, `security-scan`, `pii-scan`, `lint-scripts` | `validate --llm` / `security-scan --llm` — deeper LLM security analysis |
| `doctor`, `health-check` (offline readiness probes; missing credentials are reported as not ready) | `--llm-verify` on `validate`, `security-scan`, `pii-scan` — LLM false-positive suppression |
| | All Tier 2 commands — embeddings API; `context-optimization-check` and `dedup-scan` also use a chat LLM (`similarity-check` is embeddings-only). A local OpenAI-compatible server works too — see [below](#fully-local-tier-2-no-external-calls) |
| Tier 3 scaffolding and inspection: `create-eval-dataset --no-llm`, `init-custom-grader`, `init-harbor-task`, `tier3 validate`, `view`, and `compare` | Tier 3 LLM dataset generation needs a provider key; `evaluate` needs a provider key plus the selected live agent's own credential |

`validate` without `--no-dedup` stays usable keyless: the Tier 2 dedup pass
skips gracefully when no embedding provider is configured.

## Public Providers

Set `SKILL_EVAL_LLM_PROVIDER` to `openai`, `anthropic`, `nv_build`, `bedrock`,
or `openai-compatible`. The selected provider determines the credential, and
each ships a default model that `SKILL_EVAL_LLM_MODEL` overrides:

| Provider (`SKILL_EVAL_LLM_PROVIDER`) | Credential | Endpoint | Default model |
| --- | --- | --- | --- |
| `nv_build` | `NVIDIA_API_KEY` | integrate.api.nvidia.com | `meta/llama-3.1-8b-instruct` |
| `anthropic` | `ANTHROPIC_API_KEY` | api.anthropic.com | `claude-sonnet-4-5` |
| `openai` | `OPENAI_API_KEY` | api.openai.com | `gpt-4.1-mini` |
| `bedrock` | Standard AWS credential chain; `AWS_REGION` (defaults to `us-west-2`) | AWS Bedrock Runtime | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `openai-compatible` | `SKILL_EVAL_LLM_API_KEY` + `SKILL_EVAL_LLM_BASE_URL` + `SKILL_EVAL_LLM_MODEL` | Any OpenAI-compatible URL | _(explicit)_ |

The quickest start is a free NVIDIA API Catalog key from
[build.nvidia.com](https://build.nvidia.com/), exported as `NVIDIA_API_KEY` —
it covers LLM judging and Tier 2 embeddings with one credential. For a custom
OpenAI-compatible endpoint, `SKILL_EVAL_LLM_BASE_URL` takes precedence over
the provider default.

The `--llm` security analysis runs
[SkillSpector](https://github.com/NVIDIA/SkillSpector), which has its own
provider environment (`SKILLSPECTOR_PROVIDER` plus per-provider credential
variables). SkillEvaluator bridges your configured provider automatically for
that invocation, so the one key above is all you need. Setting
`SKILLSPECTOR_PROVIDER` yourself disables the bridge and your own
SkillSpector configuration wins.

## Embeddings

Tier 2 uses an OpenAI-compatible embeddings API. Set
`SKILL_EVAL_EMBEDDING_PROVIDER` to `openai` (`text-embedding-3-small`),
`nv_build` (`nvidia/nv-embed-v1`), or `openai-compatible`. Use
`SKILL_EVAL_EMBEDDING_MODEL` and `SKILL_EVAL_EMBEDDING_BASE_URL` to override
defaults.

Anthropic and Bedrock do not provide embeddings, so when one of those is the
LLM provider, configure an embedding-capable provider separately for Tier 2:

```bash
export SKILL_EVAL_EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY='sk-...'
```

### Fully local Tier 2 (no external calls)

The `openai-compatible` provider accepts any local OpenAI-compatible server
(Ollama, vLLM, llama.cpp, NVIDIA NIM). Verified recipe using Ollama in
Docker:

```bash
docker run -d --name ollama -p 11434:11434 ollama/ollama
docker exec ollama ollama pull nomic-embed-text   # embeddings
docker exec ollama ollama pull qwen2.5:0.5b       # chat LLM for dedup analysis

export SKILL_EVAL_EMBEDDING_PROVIDER=openai-compatible
export SKILL_EVAL_EMBEDDING_BASE_URL=http://localhost:11434/v1
export SKILL_EVAL_EMBEDDING_MODEL=nomic-embed-text
export SKILL_EVAL_EMBEDDING_API_KEY=local-no-key  # must be set; local servers ignore the value

# context-optimization-check and dedup-scan also use a chat LLM:
export SKILL_EVAL_LLM_PROVIDER=openai-compatible
export SKILL_EVAL_LLM_BASE_URL=http://localhost:11434/v1
export SKILL_EVAL_LLM_MODEL=qwen2.5:0.5b
export SKILL_EVAL_LLM_API_KEY=local-no-key

skillevaluator similarity-check ./skills
skillevaluator context-optimization-check ./my-skill
```

`similarity-check` compares skills directly or through a local catalog and
needs only the embedding variables. `context-optimization-check` (also
available as its `dedup-scan` alias) additionally uses the chat model. Analysis
quality tracks that model — the tiny model above proves the plumbing; pick a
stronger local model for verdicts you intend to act on.

## Live Agent Credentials

The evaluator provider is used for dataset generation and verifier-side
judging. It is not automatically an agent credential. Configure credentials
for the agent you select separately.

For example, Codex requires an OpenAI credential compatible with the Responses
API. An NVIDIA Build key is intentionally verifier-only and cannot be used as
the Codex `OPENAI_API_KEY`. For a Docker task, reference the agent credential
from the skill's `evals/config.yml` rather than committing a literal value:

```yaml
schema_version: 1
harbor:
  runtime_env:
    OPENAI_API_KEY: ${OPENAI_API_KEY}
    OPENAI_BASE_URL: ${OPENAI_BASE_URL}
  agents:
    codex:
      model: gpt-4.1-mini
```

This lets an NVIDIA Build provider supply evaluator-owned chat, embeddings, and
verifier calls while Codex uses its own OpenAI credential and model. You can
also supply that model at invocation time with `--agent-model codex=MODEL`.

Do not commit credentials. On the Docker backend, Harbor task environment
values can be visible to users able to inspect host process arguments. Use
scoped, short-lived credentials and a non-shared host, or use an environment
with platform-managed secret injection.

`runtime_env` is for task and agent values, not host-process configuration.
Names that control the launcher or dynamic runtime, including `PATH`,
`PYTHON*`, `LD_*`, `DYLD_*`, `DOCKER_*`, `COMPOSE_*`, proxy variables, and
telemetry exporters, are rejected. Configure the selected Harbor backend in
the host environment instead.

## Telemetry

Telemetry is disabled by default. To opt in, install
`skillevaluator[telemetry]`, set `SKILLEVALUATOR_TELEMETRY_ENABLED=true`, and
configure a standard `OTEL_EXPORTER_OTLP_ENDPOINT`. Identity export is
controlled by `SKILLEVALUATOR_TELEMETRY_IDENTITY_MODE`: the default
`team_only` exports no user identity, `hashed` exports only a hash, and any
other value exports the raw login — leave it unset unless you need
attribution.
