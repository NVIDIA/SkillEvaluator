# Tier 3: Skill Evaluation with Live Agents

Live evaluation answers one question: **does your skill actually make an agent
better at the task?** Skill Evaluator runs a real agent (such as `codex`,
`claude-code`, or `opencode`) against the same eval cases with the skill
installed and, unless `--skip-baseline` is used, without it — in a selected
[Harbor](https://github.com/harbor-framework/harbor) environment. Docker and
cloud backends provide environment isolation; local mode applies its configured
OS-sandbox policy.

With `default` or `default_plus_custom` grading, completed reports lead with five
human-readable dimensions: **Security, Correctness, Discoverability,
Effectiveness, and Efficiency**. The with-skill and without-skill results make
the skill's effect visible rather than treating a single agent run as proof of
quality. In `custom_only` mode, the user-owned grader defines the score instead.

## Table of Contents

- [Quick Start](#quick-start)
- [How a Run Works](#how-a-run-works)
- [Two Ways to Run Tier 3](#two-ways-to-run-tier-3)
- [Prerequisites and Credentials](#prerequisites-and-credentials)
- [Commands](#commands)
- [Choosing an Environment](#choosing-an-environment)
- [Eval Datasets](#eval-datasets)
- [Custom Graders and Custom Tasks](#custom-graders-and-custom-tasks-byog--byot)
- [Skill and Result Layout](#skill-and-result-layout)
- [Reading Results](#reading-results)
- [Troubleshooting](#troubleshooting)

## Quick Start

The fastest path from a skill directory to a live comparison:

```bash
# 1. Install with the Tier 3 extra ([all] includes it; from a source
#    checkout, `uv sync --all-extras` or `pip install -e ".[tier3]"` also work)
uv tool install "skillevaluator[tier3] @ git+https://github.com/NVIDIA/SkillEvaluator.git"

# 2. Configure the evaluator's LLM provider (judging + dataset generation)
export SKILL_EVAL_LLM_PROVIDER=openai
export OPENAI_API_KEY='...'

# 3. Configure the live-agent credential role (see below)
#    For codex this is an OpenAI Responses API key, configured in
#    evals/config.yml via environment substitution.

# 4. Generate eval cases for your skill
skillevaluator create-eval-dataset ./my-skill --full

# 5. Check that the environment is ready
skillevaluator doctor --agents codex --env-mode docker

# 6. Run the live comparison
skillevaluator evaluate ./my-skill --agents codex --env-mode docker

# 7. Open the HTML report
skillevaluator view ./my-skill
```

If a step fails, start with [Troubleshooting](#troubleshooting).

## How a Run Works

1. `evaluate` reads the selected accepted dataset or native task source and
   builds the Harbor task bundle.
2. Harbor starts the selected environment and runs the agent with the skill and,
   unless `--skip-baseline` is set, without it. Isolation follows the selected
   backend and, for local mode, its active OS-sandbox setting.
3. With standard grading, each transcript is judged against the case's expected
   output and assertions using the configured evaluator provider. A
   `custom_only` grader owns this step instead.
4. Skill Evaluator collects the result artifacts. Completed standard-grading
   runs render the five human-readable dimensions, pass rates, and Skill Lift.

The CLI uses the same in-process evaluation service for focused `evaluate`
runs and `validate --agent-eval` runs. Temporary task and job directories are
deleted after collection; pass `--harbor-keep-jobs` to retain them.

## Two Ways to Run Tier 3

| Use case | Command | When to use |
| --- | --- | --- |
| Full validation | `skillevaluator validate ./my-skill --agent-eval` | Add advisory Tier 3 results after the standard validation stages |
| Focused evaluation | `skillevaluator evaluate ./my-skill --agents codex` | Iterate on datasets, agents, environments, and grading settings |

Tier 3 remains advisory when attached to `validate`: it is included in the
combined reports but does not change the validation exit code. Focused live
evaluation still requires the provider and agent credentials needed by the
selected setup.

## Prerequisites and Credentials

Tier 3 needs:

1. A preferred generated dataset at `evals/evals.json`, another accepted dataset
   file, or native tasks under `evals/harbor/`.
2. The `tier3` extra, which installs Harbor and the evaluator dependencies.
3. A configured evaluator LLM provider.
4. The selected live agent's native credential.
5. A configured environment, such as Docker, local OS isolation, or a supported
   Harbor cloud backend.

The evaluator and live agent use **two credential roles**. They are not copied
between roles automatically, although a user may deliberately source both from
the same host secret. Local mode is an exception: for compatible agent/provider
pairs, the runner can map evaluator-provider credentials into the environment
variables read by the local agent CLI.

| Credential | Used for | How to set |
| --- | --- | --- |
| Evaluator LLM provider | Dataset generation and standard grading | `SKILL_EVAL_LLM_PROVIDER` plus the provider key; see [CONFIGURATION.md](CONFIGURATION.md) |
| Live agent credential | The agent actually performing the task | Run-scoped values in `evals/config.yml` `harbor.runtime_env` |

For example, NVIDIA Build can power dataset generation and standard grading,
while `codex` uses an OpenAI Responses API credential. Configure agent
credentials through environment-variable substitution; never put literal keys
in the file:

```yaml
schema_version: 1
harbor:
  runtime_env:
    OPENAI_API_KEY: ${CODEX_OPENAI_API_KEY}
    OPENAI_BASE_URL: ${CODEX_OPENAI_BASE_URL}
  agents:
    codex:
      model: gpt-4.1-mini
```

The agent model must be selected explicitly when the evaluator provider's
default model is not valid for that agent. Use `--agent-model codex=MODEL` for
a per-invocation override.

Security notes:

- Do not commit credentials. `runtime_env` values pass into the task
  environment and may be visible to users able to inspect processes on a
  shared host. Prefer scoped, short-lived credentials and isolated hosts.
- Evaluator-provider credentials are not inserted into `runtime_env`
  automatically. In local mode, compatible provider credentials may instead be
  routed directly to the selected agent subprocess. Every value that you place
  in `runtime_env` is available to every agent selected for that run. For strict
  per-agent secret isolation, use separate runs with only the values required by
  that agent.
- `runtime_env` cannot override host launcher, language-runtime, Docker,
  Compose, proxy, or telemetry controls. Configure the selected backend in the
  host environment.

## Commands

### `evaluate`

```bash
skillevaluator evaluate <SKILL_PATH> [OPTIONS]
```

| Option | Default | Purpose |
| --- | --- | --- |
| `-a`, `--agents` | `codex` | Comma-separated Harbor agents |
| `--env-mode` | `docker` | Docker, local, or a supported Harbor cloud backend |
| `--skip-baseline` | false | Skip the without-skill arm for faster iteration; no Skill Lift is produced |
| `--n-attempts` | `1` | Attempts per eval case; values above one add pass@k context |
| `--pass-threshold` | `0.5` | Score required for an attempt to count as passed |
| `--n-concurrent` | `4` | Concurrent eval cases per agent |
| `--max-agents` | selected-agent count | Maximum agents run in parallel |
| `--model` | resolved from agent/config | Global live-agent model override |
| `--agent-model` | none | Repeatable `AGENT=MODEL` override |
| `--grading-mode` | `default` | `default`, `default_plus_custom`, or `custom_only` |
| `--skill-workspace-mode` | `isolated` | `isolated` or `group` skill staging |
| `--include-skills` | none | Repeatable additional skill directory or parent directory; requires `--skill-workspace-mode group` |
| `--copy-repo` | false | Copy the surrounding repository into the task environment |
| `--custom-dockerfile-mode` | `rebase` | `preserve` or `rebase` a custom eval Dockerfile |
| `--results-dir` | `evals/results` or `SKILLEVALUATOR_RESULTS_DIR` | Write results below an explicit external root |
| `--harbor-keep-jobs` | false | Retain Harbor job directories for inspection |
| `--timeout-multiplier` | `1.0` | Scale Harbor setup, run, and verifier timeouts |
| `--override-cpus`, `--override-memory-mb`, `--override-storage-mb` | engine default | Per-environment resource overrides |

Examples:

```bash
# Multi-agent comparison
skillevaluator evaluate ./my-skill --agents codex,opencode --env-mode docker

# Faster iteration without the baseline arm
skillevaluator evaluate ./my-skill --agents codex --skip-baseline

# Three attempts per case for pass@k context
skillevaluator evaluate ./my-skill --agents codex \
  --n-attempts 3 --pass-threshold 0.7
```

### `create-eval-dataset`

```bash
skillevaluator create-eval-dataset <SKILL_PATH> [OPTIONS]
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--full` | false | Generate the four-bucket dataset instead of one case |
| `--no-llm` | false | Use local templates without an LLM call |
| `--dry-run` | false | Print generated JSON without writing it |
| `--force` | false | Overwrite an existing `evals/evals.json` |
| `--prompt` | none | Use a developer guidance file instead of `evals/EVAL.md` |
| `--refine` | false | Refine cases using an existing or collected trajectory |
| `--from-results` | latest results | Select the trajectory search directory for refinement |
| `--results-dir` | none | Search an external results root for refinement data |

### `view`, `compare`, and `doctor`

```bash
skillevaluator view ./my-skill [--results-dir DIR]
skillevaluator compare ./my-skill [--results-dir DIR]
skillevaluator doctor --agents codex --env-mode docker [--verify-models]
```

- `view` opens the latest local HTML report.
- `compare` summarizes stored results across agents and arms.
- `doctor` checks backend readiness, agent availability, and visible host
  credentials. `evaluate` remains authoritative for skill-specific config and
  task credential injection.

### Tier 3 expert commands

```bash
skillevaluator tier3 validate ./my-skill [--json] [--strict] [--harbor-contract]
skillevaluator tier3 harbor-view <JOBS_DIR>
```

The first command validates the eval dataset and optional Harbor task contract.
The second opens retained Harbor job artifacts in Harbor's trajectory browser.

## Choosing an Environment

Docker is the default and needs a running Docker daemon. Harbor cloud
environments such as `daytona`, `e2b`, or `modal` can be selected with
`--env-mode`; each needs the corresponding Harbor extra and provider setup:

```bash
uv tool install --with 'harbor[daytona]' \
  "skillevaluator[tier3] @ git+https://github.com/NVIDIA/SkillEvaluator.git"

export DAYTONA_API_KEY='...'
skillevaluator evaluate ./my-skill --agents codex --env-mode daytona
```

### Local mode (`--env-mode local`)

Local mode runs the agent CLI directly on the host — no Docker. Under the
default `require` setting it is confined by an OS sandbox: **bubblewrap** on
Linux and **Seatbelt** (`sandbox-exec`) on macOS. It is meant for
**semi-trusted** skills on a developer machine; use Docker or a cloud
environment for arbitrary untrusted code.

Linux bubblewrap is the strong path because it uses namespace isolation. macOS
Seatbelt confines filesystem access and network access at the kernel level, but
it has no PID namespace that guarantees process cleanup. The default macOS read
policy is a compatibility HOME denylist. Set
`SKILLEVALUATOR_LOCAL_STRICT_READS=1` for deny-all reads with selected runtime
and system exceptions. This is stricter but may expose compatibility issues in
host developer tools.

```bash
# opencode works with NVIDIA Build and other OpenAI-compatible endpoints
export NVIDIA_API_KEY='nvapi-...'
skillevaluator evaluate ./my-skill --agents opencode \
  --env-mode local --model nvidia/openai/gpt-oss-120b
```

Requirements and behavior:

- **Agent CLI installed:** local mode searches `SKILLEVALUATOR_RUNTIME_DIR` and
  `PATH` for `claude-code`, `codex`, or `opencode`. It does not download missing
  runtimes.
- **Sandbox required by default:** if no supported OS sandbox is usable, the
  `require` setting fails closed. `prefer` may continue with advisory-only
  protection, while `off` disables kernel confinement and is only for fully
  trusted skills.
- **Confinement when the OS sandbox is active:** writes are restricted to the
  run directory. The host home and common secret locations are excluded from
  the run's readable filesystem. These guarantees do not apply with
  `SKILLEVALUATOR_LOCAL_SANDBOX=off`.
- **Network egress is enabled by default** so the agent can reach its model
  endpoint. With an active OS sandbox,
  `SKILLEVALUATOR_LOCAL_ALLOW_NET=0` denies network access; it cannot enforce an
  air gap when kernel confinement is disabled.
- **No reliable detached-service containment on macOS:** use Docker or a cloud
  environment for untrusted code, long-running services, and sidecars.

| Variable | Purpose |
| --- | --- |
| `SKILLEVALUATOR_LOCAL_SANDBOX` | `require` (default), `prefer`, or `off` |
| `SKILLEVALUATOR_LOCAL_ALLOW_NET` | Set to `0` to deny network egress |
| `SKILLEVALUATOR_LOCAL_STRICT_READS` | Set to `1` for deny-all reads with selected exceptions |
| `SKILLEVALUATOR_RUNTIME_DIR` | Dedicated directory containing bring-your-own agent CLIs |

## Eval Datasets

`create-eval-dataset` writes the preferred `evals/evals.json`. `evaluate` also
accepts `evals.jsonl`, `evals.yaml`, `evals.yml`, `dataset.json`, and
`dataset.jsonl` under `evals/`, and it can use native tasks under
`evals/harbor/`. A full generated dataset covers four useful cases:

| Bucket | Purpose |
| --- | --- |
| Explicit positive | The user names the skill directly |
| Implicit positive | The task needs the skill without naming it |
| Contextual positive | The task needs the skill within a broader project context |
| Negative | A related request should not activate the skill |

Generate, preview, or refine cases with:

```bash
skillevaluator create-eval-dataset ./my-skill --full
skillevaluator create-eval-dataset ./my-skill --no-llm
skillevaluator create-eval-dataset ./my-skill --full --dry-run
skillevaluator create-eval-dataset ./my-skill --full --refine
```

New datasets use the preferred agentskills.io shape:

```json
{
  "skill_name": "api-caller",
  "evals": [
    {
      "id": "api-caller-001",
      "prompt": "Inspect this API description and make the requested call.",
      "expected_output": "The request is validated, executed, and summarized.",
      "assertions": [
        "The agent reads the skill instructions.",
        "The agent validates request inputs before sending the call."
      ]
    }
  ]
}
```

Legacy flat entries remain accepted with a warning, but new datasets should use
the preferred shape. Add `evals/EVAL.md` for domain guidance. Without a
configured provider, generation falls back to generic local templates.

Run controls live in `evals/config.yml`, including `n_attempts`,
`pass_threshold`, concurrency, resource overrides, runtime environment values,
and per-agent models. CLI options take precedence.

## Custom Graders and Custom Tasks (BYOG / BYOT)

The default grader checks each case against its assertions. For deterministic
or domain-specific grading, bring your own grader (BYOG) or a complete Harbor
task (BYOT):

```bash
skillevaluator init-custom-grader ./my-skill \
  --mode default_plus_custom --language python

skillevaluator init-harbor-task ./my-skill \
  --with-config --case-id case-001
```

Grading modes are:

| Mode | Behavior |
| --- | --- |
| `default` | Skill Evaluator grading only |
| `default_plus_custom` | Skill Evaluator grading plus the custom grader |
| `custom_only` | The custom grader owns the result |

`init-harbor-task` creates the task where Skill Evaluator discovers it; Harbor's
own task initializer is not required. See the `create-custom-grader` reference
skill in `src/skillevaluator/tier3/reference_skills/` for a worked example.

### Docker Compose safety policy

Custom `docker-compose.yaml` files are for sidecars. The Harbor-owned `main`
service may set only `depends_on`. Sidecars use a strict allowlist for images,
bounded builds, commands, environment values, health checks, project-scoped
storage, and container-local execution settings. Host port mappings are removed
before execution, and unrecognized keys are rejected.

Skill Evaluator rejects host bind mounts, external or driver-configured storage,
host namespace modes, privileged/device/GPU/runtime access, added capabilities,
Docker API socket access, credential files, external container links, and
file-backed Compose configs or secrets. Build contexts and Dockerfiles must be
literal relative paths contained in the applicable environment directory —
`evals/environment/` for generated tasks or
`evals/harbor/<case>/environment/` for a native task. Remote, SSH, interpolated,
privileged, host-networked, secret, and insecure-entitlement builds are
rejected.

## Skill and Result Layout

```text
my-skill/
├── SKILL.md
├── scripts/
└── evals/
    ├── evals.json                  # Preferred generated Tier 3 dataset
    ├── EVAL.md                     # Optional generation guidance
    ├── config.yml                  # Optional run settings
    ├── grader.py                   # Optional BYOG grader
    ├── harbor/                     # Optional BYOT tasks
    ├── environment/                # Optional custom environment
    ├── files/                      # Optional task inputs
    └── results/
        └── <timestamp>/
            ├── result.json
            ├── run_config.json
            ├── attempt_policy.json
            ├── report.html                  # Generated on a best-effort basis
            ├── comparison.json             # Multi-agent runs only
            └── <agent>/
                ├── lift.json               # Present when baseline lift is available
                ├── with-skill/
                │   ├── summary.json
                │   └── trials/
                └── without-skill/          # Present when the baseline arm is enabled
                    ├── summary.json
                    └── trials/
```

Use `--results-dir DIR` to write the results below `DIR/<skill-name>/` instead of
the skill directory. `SKILLEVALUATOR_RESULTS_DIR` supplies the default external
root when the CLI option is omitted.

## Reading Results

```bash
skillevaluator view ./my-skill
skillevaluator compare ./my-skill
```

For completed `default` and `default_plus_custom` runs, the report summarizes
five human-readable dimensions. In `custom_only`, the custom grader owns the
score and the standard dimension view is omitted.

| Dimension | Question answered |
| --- | --- |
| Security | Is the result safe to use? |
| Correctness | Does it do what the skill promises? |
| Discoverability | Is the skill used when it should be? |
| Effectiveness | Is the result better with the skill? |
| Efficiency | Does the skill reduce unnecessary effort? |

Each dimension receives a score and PASS, NEUTRAL, or FAIL status. Skill Lift
is the signed difference between the with-skill result and the without-skill
baseline. Positive lift means the skill helped; negative lift means it hurt;
the report's verdict applies the release's current noise band.

With multiple attempts, pass@k shows whether at least one attempt for each case
met `--pass-threshold`. Under standard grading it is reliability context
alongside Skill Lift, not a replacement for the five dimensions. In
`custom_only`, pass@k uses the custom overall reward and the standard dimension
view remains absent.

The canonical Tier 3 payload embedded by `validate --agent-eval` uses schema
version 2.0. A completed standard-grading payload includes the summary, five
dimensions, per-agent results, trials, pass@k, attempt policy, and dataset;
advisory or skipped payloads may leave some of those sections empty. Focused
`evaluate` artifacts have file-specific shapes and are not all schema 2.0.

Keep Harbor artifacts for debugging with `--harbor-keep-jobs`; the report links
to the retained job directories.

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `A public LLM provider is required for live evaluation` | Set `SKILL_EVAL_LLM_PROVIDER` and the corresponding evaluator-provider key. |
| Agent authentication fails | Configure the live-agent credential role through `harbor.runtime_env` in `evals/config.yml`. |
| `evaluate` cannot start the environment | Start Docker or configure the selected `--env-mode`; run `doctor` with the same agent and environment. |
| No eval dataset or native task source found | Run `skillevaluator create-eval-dataset ./my-skill` or add accepted files under `evals/`. |
| Agent model is rejected | Supply a model valid for that agent with `--agent-model AGENT=MODEL`. |
| Results look wrong or a case is hard to diagnose | Re-run with `--harbor-keep-jobs`, then inspect retained artifacts with `view` or `tier3 harbor-view`. |

Every live run requires a configured evaluator provider plus the credentials
required by the selected agent and environment.

## See Also

- [Configuration](CONFIGURATION.md) — provider, embedding, credential, and telemetry setup.
- [Installation](INSTALLATION.md) — extras and system requirements.
- [Tier 1 validation](TIER1_VALIDATION.md) — deterministic and security checks.
- [Tier 2 deduplication](TIER2_DEDUPLICATION.md) — semantic overlap checks.
