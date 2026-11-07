# Providers and credentials

Set exactly one provider. If `SKILL_EVAL_LLM_PROVIDER` is unset and exactly
one of `NVIDIA_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` is exported,
it is auto-selected; with more than one, LLM commands fail until the
provider is set.

| Provider | Variables | Notes |
| --- | --- | --- |
| `nv_build` | `NVIDIA_API_KEY` | One key covers Tier 1 LLM, Tier 2 embeddings, and Tier 3 agents in `docker`, `kata`, `local`; default agent `opencode` |
| `openai` | `OPENAI_API_KEY` | Default agent `codex` |
| `anthropic` | `ANTHROPIC_API_KEY` | Default agent `claude-code`; Tier 2 needs a separate embedding provider |
| `bedrock` | AWS credential chain, `AWS_REGION` | Tier 2 needs a separate embedding provider |
| `openai-compatible` | `SKILL_EVAL_LLM_BASE_URL`, `SKILL_EVAL_LLM_API_KEY` | Any gateway; override models whose catalog IDs differ |

Overrides: `SKILL_EVAL_LLM_MODEL` (evaluator and judge),
`LLM_JUDGE_MODEL` (judge only), `SKILL_EVAL_EMBEDDING_MODEL`,
`--agent-model AGENT=MODEL` (Tier 3 agent). OpenCode on a gateway takes
`--agent-model opencode=openai/MODEL`. Codex needs a Responses-API model,
OpenCode Chat Completions, Claude Code Anthropic Messages, all with tool
calls.

Rules:

- Credentials come only from the host environment (or masked CI secrets).
  `harbor.runtime_env` in `<skill-dir>/evals/config.yml` rejects provider keys and base
  URLs with a hard error.
- `skillevaluator models` and `doctor --verify-models` check catalog
  reachability; a public catalog does not prove the inference key works.
- Judge failures make a result unscored (`NO SCORE`), not `0.0`.
