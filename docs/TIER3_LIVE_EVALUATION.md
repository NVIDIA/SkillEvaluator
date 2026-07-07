# Tier 3: Skill Evaluation with Live Agents

Live evaluation answers one question: **does your skill actually make an agent
better at the task?** Skill Evaluator runs a real agent (such as `codex`,
`claude-code`, or `opencode`) against your eval cases twice — once with the
skill installed and once without — inside an isolated
[Harbor](https://github.com/harbor-framework/harbor) environment, then judges
and compares the results.

## Quick Start

The fastest path from a skill directory to a live comparison:

```bash
# 1. Install with the Tier 3 extra ([all] includes it; from a source
#    checkout, `uv sync --all-extras` or `pip install -e ".[tier3]"` also work)
uv tool install "skillevaluator[tier3] @ git+https://github.com/NVIDIA/SkillEvaluator.git"

# 2. Configure the evaluator's LLM provider (judging + dataset generation)
export SKILL_EVAL_LLM_PROVIDER=openai
export OPENAI_API_KEY='...'

# 3. Configure the agent's own credential (separate from step 2 — see below)
#    For codex this is an OpenAI Responses API key, configured in
#    evals/config.yml via environment substitution.

# 4. Generate eval cases for your skill
skillevaluator create-eval-dataset ./my-skill --full

# 5. Check that your environment is ready (Docker, agent, credentials)
skillevaluator doctor --agents codex --env-mode docker

# 6. Run the live comparison
skillevaluator evaluate ./my-skill --agents codex --env-mode docker

# 7. Open the HTML report
skillevaluator view ./my-skill
```

Each step is explained in detail below. If anything fails, start with
[Troubleshooting](#troubleshooting).

## How a Run Works

1. `evaluate` reads `evals/evals.json` in your skill directory and builds one
   Harbor task bundle per eval case.
2. Harbor starts an isolated environment (Docker by default) and runs the
   selected agent on each case — in two arms, with and without your skill
   installed.
3. A verifier judges each transcript against the case's assertions using your
   configured public LLM provider.
4. Reward artifacts are collected into a local report showing pass rates for
   both arms, so you can see the skill's effect directly.

Temporary task and job directories are deleted after collection. Pass
`--harbor-keep-jobs` to retain them for inspection.

## Credentials: Two Separate Things

Live evaluation always involves **two independent credentials**. Mixing them
up is the most common setup mistake.

| Credential | Used for | How to set |
| --- | --- | --- |
| Evaluator LLM provider | Dataset generation, verifier judging | `SKILL_EVAL_LLM_PROVIDER` + provider key (see [CONFIGURATION.md](CONFIGURATION.md)) |
| Live agent's native credential | The agent actually running the task | Per agent, via `evals/config.yml` `harbor.runtime_env` |

For example, NVIDIA Build can power dataset generation and judging, but
`codex` still requires its own OpenAI Responses API credential — the two are
never interchangeable. Configure agent credentials through environment-variable
substitution; never put literal keys in the file:

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

The agent's model must also be chosen explicitly when the evaluator's provider
default is not valid for that agent (NVIDIA Build's evaluator model is not a
Codex model). Use `--agent-model codex=MODEL` to override per invocation.

Security notes:

- Do not commit credentials. `runtime_env` values pass into the task
  environment; on a shared host they can be visible to users able to inspect
  process arguments. Use scoped, short-lived credentials, a non-shared host,
  or an environment with platform-managed secret injection.
- The evaluator's provider key is not shared with the agent, and agent
  credentials are not sent to the evaluator's provider.
- `runtime_env` cannot override host launcher, language-runtime, Docker,
  Compose, proxy, or telemetry control variables. Configure the selected
  Harbor backend directly in the host environment.

## Choosing an Environment

Tier 3 uses Harbor's native environments. Docker is the default and needs a
running Docker daemon. Cloud environments such as `daytona`, `e2b`, or `modal`
can be selected with `--env-mode`; each one requires installing the matching
Harbor extra and configuring that provider's credentials per Harbor's
documentation:

```bash
# uv tool install: add the Harbor extra to the tool's environment
uv tool install --with 'harbor[daytona]' \
  "skillevaluator[tier3] @ git+https://github.com/NVIDIA/SkillEvaluator.git"
# (source checkout: pip install 'harbor[daytona]' into the project venv instead)

export DAYTONA_API_KEY='...'
skillevaluator evaluate ./my-skill --agents codex --env-mode daytona
```

### Local mode (`--env-mode local`)

Local mode runs the agent CLI directly on the host — no Docker — confined by an
OS sandbox: **bubblewrap** on Linux and **Seatbelt** (`sandbox-exec`) on macOS.
It is meant for **semi-trusted** skills on a developer machine; use Docker or a
cloud environment for arbitrary untrusted code.

Linux bubblewrap is the strong path (namespace isolation). macOS Seatbelt is
**semi-trusted**: it confines the filesystem (writes restricted to the run
directory, host home and other secrets unreadable) and network at the kernel
level, and denies cross-process metadata (a skill can't read another process's
command line). The default macOS read policy is a compatibility HOME-denylist.
Set `SKILLEVALUATOR_LOCAL_STRICT_READS=1` to enable the deny-all-read profile
with only the selected runtime/system exceptions; this mode is stricter but
may expose compatibility issues in host developer tools. Common direct and
nested shell spellings of `setsid`, `nohup`, `daemon`, and `disown` are rejected
as defense in depth. This is not process containment: a script or native
program can still detach through system APIs, and Seatbelt has no PID namespace
that guarantees cleanup. For arbitrary downloaded/untrusted skills on macOS,
use `--env-mode docker`.

```bash
# opencode is the agent that works with NVIDIA Build (OpenAI-compatible)
export NVIDIA_API_KEY='nvapi-...'
skillevaluator evaluate ./my-skill --agents opencode \
  --env-mode local --model nvidia/openai/gpt-oss-120b
```

Requirements and behavior:

- **Agent CLI installed on `PATH`** (or under `SKILLEVALUATOR_RUNTIME_DIR`):
  `claude-code`, `codex`, or `opencode`. Local mode never downloads runtimes; if
  one is missing it prints the vendor install command (e.g.
  `npm install -g opencode-ai`). The runtime directory must be a dedicated
  subdirectory; the host home directory and its parents are rejected.
- **Sandbox required by default (fail-closed).** If no OS sandbox is usable
  (e.g. bubblewrap missing or user namespaces disabled on Linux), the run
  refuses to start. Override for skills you fully trust with
  `SKILLEVALUATOR_LOCAL_SANDBOX=off`, or use `--env-mode docker`.
- **Confinement:** writes are restricted to the run directory on both
  platforms. Reads of the user's home are blocked except for the isolated run
  tree and explicit read-only agent/runtime roots — on Linux the host home is
  never mounted; on macOS Seatbelt denies reads of the home subtree — so
  secrets such as `~/.ssh`, `~/.aws`, and project `.env` files stay unreadable.
  macOS additionally denies cross-process metadata (no reading another
  process's command line). Set `SKILLEVALUATOR_LOCAL_STRICT_READS=1` when a
  deny-all-read policy is required.
- **Network egress is ON by default** so the agent can reach the model
  endpoint; set `SKILLEVALUATOR_LOCAL_ALLOW_NET=0` to airgap a skill that must
  not touch the network.
- **Agent choice:** `opencode` works with NVIDIA Build / any OpenAI-compatible
  endpoint; with NVIDIA Build its default model is normalized to
  `nvidia/<provider-model>`. `codex` needs an independent OpenAI Responses API
  credential and both `OPENAI_API_KEY` and `OPENAI_BASE_URL`; `claude-code`
  needs an independent Anthropic-native credential and an explicit model when
  NVIDIA Build is the evaluator provider.
- **No detached services:** background commands and common shell launcher
  spellings are rejected, but script/native detachment cannot be structurally
  contained on macOS. Use Docker or a cloud environment for untrusted code,
  `pre_agent_setup` services, and sidecars.

| Variable | Purpose |
| --- | --- |
| `SKILLEVALUATOR_LOCAL_SANDBOX` | `require` (default, fail closed), `prefer` (advisory-only with a warning), or `off` (trusted, no sandbox) |
| `SKILLEVALUATOR_LOCAL_ALLOW_NET` | Set to `0` to deny network egress inside the sandbox (default on) |
| `SKILLEVALUATOR_LOCAL_STRICT_READS` | Set to `1` to use deny-all reads with only runtime/system exceptions (default `0` for macOS compatibility) |
| `SKILLEVALUATOR_RUNTIME_DIR` | Dedicated subdirectory holding bring-your-own agent CLIs, searched before `PATH`; it cannot be the host home or a parent of it |

```bash
skillevaluator doctor --agents codex --env-mode docker
```

`doctor` verifies the selected environment, agent availability, and available
host credentials. It has no skill path, so `evaluate` remains authoritative for
`evals/config.yml`, per-agent model selection, and task credential injection.

## Eval Datasets

`evaluate` expects `evals/evals.json` inside the skill. Generate it with:

```bash
skillevaluator create-eval-dataset ./my-skill            # 1 case
skillevaluator create-eval-dataset ./my-skill --full     # 4-bucket dataset
skillevaluator create-eval-dataset ./my-skill --no-llm   # templates, no key needed
skillevaluator create-eval-dataset ./my-skill --full --refine  # refine with a real trajectory
```

Without a configured LLM provider, generation falls back to template-based
cases; they are runnable but generic. For higher-quality cases, configure a
provider, add developer guidance in `evals/EVAL.md` (`## Questions`,
`## Behaviors`, `## Notes` sections), or use `--refine` to ground cases in a
real agent trajectory.

Run controls live in `evals/config.yml`: `n_attempts`, `pass_threshold`,
`n_concurrent`, resource overrides, runtime environment variables, and
per-agent model overrides. CLI options take precedence over the file.

## Custom Graders and Custom Tasks (BYOG / BYOT)

The default verifier judges transcripts against each case's assertions. For
deterministic or domain-specific grading, bring your own grader (BYOG) or a
complete Harbor task (BYOT):

```bash
# Scaffold evals/grader.py wired to Skill Evaluator's grading contract
skillevaluator init-custom-grader ./my-skill

# Scaffold a Harbor-format task under evals/harbor/<case-id>/
skillevaluator init-harbor-task ./my-skill --case-id case-001
```

`init-harbor-task` wraps a Harbor-format task in the exact location and shape
Skill Evaluator discovers — you do not need Harbor's own `harbor tasks init`
for this. See the `create-custom-grader` reference skill in
`src/skillevaluator/tier3/reference_skills/` for a worked example.

### Docker Compose safety policy

Custom `docker-compose.yaml` files are for sidecars. The Harbor-owned `main`
service may set only `depends_on`. Sidecars use a strict allowlist: `image`,
bounded `build`, `command`, `entrypoint`, `environment`, `expose`,
`healthcheck`, `depends_on`, `networks`, project-scoped `volumes`, `tmpfs`, and
a small set of container-local execution settings. Host port mappings are
removed before execution. Unrecognized service, build, network, volume, and
top-level keys are rejected rather than passed through to Docker.

Skill Evaluator rejects host bind mounts, external or driver-configured named
volumes, interpolated volume specifications, external/named/custom-driver
networks, host namespace modes, privileged/device/GPU/runtime access, added
capabilities, security and cgroup overrides, Docker API socket access,
`env_file`, `label_file`, `extends`, external container links, credential files,
and file-backed Compose configs or secrets. Lifecycle hooks, deployment/device
reservations, development sync, host logging drivers, build cache imports or
exports, and host-level image tags are also rejected. Sidecar build contexts
and Dockerfiles must be literal relative paths contained under
`evals/environment/`; remote/SSH or interpolated paths and privileged,
host-networked, SSH-forwarded, secret, or insecure-entitlement builds are
rejected. In non-path values, Compose `${NAME}` interpolation is allowed only
for names explicitly declared in `harbor.runtime_env`; nested substitutions and
bare host-value passthrough in service environment entries and build arguments
follow the same rule.

## Reading Results

```bash
skillevaluator view ./my-skill        # open the latest local HTML report
skillevaluator compare ./my-skill     # compare with/without-skill results
```

Keep raw Harbor artifacts for debugging with `--harbor-keep-jobs`; the report
links to retained job directories.

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `A public LLM provider is required for live evaluation` | Set `SKILL_EVAL_LLM_PROVIDER` and its key (evaluator side). |
| Agent fails immediately or authentication errors in the transcript | The agent's own credential is missing — set it via `harbor.runtime_env` in `evals/config.yml`. |
| `evaluate` cannot start the environment | Docker daemon not running, or the selected `--env-mode` is unconfigured. Run `skillevaluator doctor` with the same flags. |
| No `evals/evals.json` found | Run `skillevaluator create-eval-dataset ./my-skill` first. |
| Results look wrong and you need the raw transcripts | Re-run with `--harbor-keep-jobs` and inspect the retained job directories via `skillevaluator view`. |

Every live run requires a configured public LLM provider and any credentials
required by the selected Harbor agent and environment.
