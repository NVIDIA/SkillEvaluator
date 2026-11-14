# Advanced CI setup and maintainer reference

This is the explicitly enabled, unreleased all-tier integration. For standard
no-key Tier 1 onboarding, use the [template catalog](../README.md),
[GitHub guide](../../../docs/github-actions.mdx) or
[GitLab guide](../../../docs/gitlab-ci.mdx).

Released 0.8.0 bundles and plain `skillevaluator ci init` remain Tier 1 only.
A successful setup command is configuration preparation, not a completed CI
run. Inspect real provider-backed outputs before claiming all tiers passed.

## Candidate templates

The unreleased all-tier setup uses two thin consumer files per platform:

| Platform | Evaluation | Trusted feedback |
| --- | --- | --- |
| GitHub | [skill-checks-all-tiers.yml](skill-checks-all-tiers.yml) | [feedback-all-tiers.yml](feedback-all-tiers.yml) |
| GitLab | [gitlab-all-tiers.yml](gitlab-all-tiers.yml) | [gitlab-feedback-all-tiers.yml](gitlab-feedback-all-tiers.yml) |

These files contain placeholders. Use a reviewed schema-version 2 release
with the platform's all-tier capability and immutable workflow, tools and
image pins. Released 0.8.0 downloads remain Tier 1 only. Install a reviewed
candidate build and use `ci init --tiers 1,2,3` to select its bundled
`ci-release-all-tiers.json`; use `--release` to review an alternate descriptor.
The all-tier CLI renders these references and approves the exact GitHub caller blob. See
[the onboarding guide](#opt-in-to-three-tier-ci).
Merge the trusted feedback entrypoint onto the default branch before the
first demo PR/MR. On GitLab, merging into an existing root `.gitlab-ci.yml`
means replacing the Tier 1 include with exactly the two all-tier includes,
keeping `test` in any `stages` list, and starting every other job's rules with
`- if: '$SKILL_EVAL_FEEDBACK == "true"'` / `when: never`, because trusted
feedback runs as a default-branch pipeline. Merge those guards to the default
branch before the setup MR; see the
[GitLab guide](#merge-into-an-existing-root-configuration).
Only Tier 1 gates CI; Tier 2 and Tier 3 remain advisory, including failed or
incomplete outcomes, with their native evidence retained.
The CLI's `--tier3-required`, GitHub's `tier3-required` input and GitLab's
`SKILL_EVAL_TIER3_REQUIRED` variable are deprecated and ignored; they cannot
make Tier 3 blocking. Reports expire after 14 days, Docker is the default, and
native local mode is also supported.

Maintainers must refresh the candidate descriptor's `workflow_ref`,
`feedback_ref`, `gitlab_ref` and `tools_ref` together after workflow, runtime or
feedback fixes, so new consumers receive the reviewed implementation.
`evaluator_version` and `runtime_lock_sha256` must match the package version and
`uv.lock` at `tools_ref`. Keep the image digest independently pinned to the
reviewed Tier 1 image.

## Opt in to three-tier CI

The new all-tier path supports **GitHub and GitLab** with two small consumer
files. Tier 1 is the only required tier; Tier 2 and Tier 3 are advisory, including
failed or incomplete results. `--tier3-required` is deprecated and ignored; it
cannot make Tier 3 block CI. Docker is the default. `--env-mode local` runs native
agent CLIs on the selected CI runner. This setup supports these two modes only.

This is an unreleased capability. Released 0.8.0 bundles and the default
`ci init` setup remain Tier 1 only. Install a reviewed candidate build that
includes the all-tier CLI and its bundled `ci-release-all-tiers.json`, then
opt in with `--tiers 1,2,3`. The CLI selects that descriptor automatically.
It must declare schema version 2 and `all-tiers-github` or `all-tiers-gitlab`,
with immutable workflow, feedback, tools and image references. A Tier 1
descriptor cannot enable all-tier setup.

```bash
skillevaluator ci init --tiers 1,2,3 \
  --platform github --provider openai --paths skills --mode preview --no-input --show-files
skillevaluator ci init --tiers 1,2,3 \
  --platform github --provider openai --paths skills --mode pr --no-input
```

To review an alternate candidate descriptor, add
`--release ./reviewed-ci-release.json` to either command. Review its immutable
references before using it; the option does not install the candidate CLI.

For GitLab 17.0 or later, select `--platform gitlab`. GitHub uses your local
`gh` login; GitLab uses only a process-scoped `SKILLEVALUATOR_GITLAB_TOKEN`
for the repository's host, never a generic `GITLAB_TOKEN`. Origins other than
github.com need `--tiers 1,2,3` or `--platform gitlab` before any API request.
GitLab setup does not require `glab`.
`--mode preview` lists the files it would add or change without writing
anything; add `--show-files` to print their contents.
`--mode save` writes the two local configuration files; `--mode pr` opens the
setup review through the platform API. These commands prepare CI configuration;
they do not run the evaluators on your laptop.

| Local command | Result | Evidence still needed |
| --- | --- | --- |
| `ci init` | Prepare the evaluation and trusted feedback entrypoints | Merge the setup and configure the CI runner and secrets |
| `ci status --tiers 1,2,3` | Read default-branch configuration and latest pipeline status | Inspect the actual CI reports and gates |
| `ci demo --tiers 1,2,3` | Open a small skill-change PR or MR | Inspect the Tier 1 gate, advisory Tier 2/3 results, comment, and retained reports |

Before the first demo, commit native datasets under each skill's `evals/`
directory; see [Evaluation Datasets](../../../docs/eval-datasets.mdx). CI reuses those cases
with full positive and negative paired coverage. It does not generate cases or
use autopilot. Missing datasets or credentials produce actionable incomplete
results; they do not count as a passing tier. A passing Tier 1 CI gate does not
mean the advisory tiers completed or passed. Central dataset mappings are not
supported by this initial workflow.

The trusted PR or MR comment shows one summary table with a row per skill:
Static and Security Validation, Deduplication, and Live Agent Evaluation each
get a one-line status, issue count and main finding or blocker. Counts remain
unknown for incomplete or missing evidence. Each skill has one **Report** link
to the run's combined artifact; normal repository authentication and artifact
expiry apply. Download and extract it, then open `index.html` for the detailed
aggregate and native reports. Hosted HTML remains optional. See
[Reports & Results](#retained-three-tier-ci-evidence) for an example.

Create the `skillevaluator` execution environment and store provider secrets
there (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `NVIDIA_API_KEY`). GitHub also
supports the generated explicit mappings from repository CI secrets; scoped
environment secrets are recommended. On GitHub,
provider-backed jobs require the initiating actor to have repository write,
maintain, or admin permission, as verified through GitHub's API. They also
require the exact current same-repository PR head or current default-branch
commit. These checks run again after entering the execution environment.
Environment reviewers are optional: configure them if your repository wants
manual approval; automatic checks do not require a paid reviewer-protection
feature. Restrict repository and CI workflow write access to trusted maintainers.
GitLab's protected execution and feedback prerequisites are described below.
Choose matching chat and embedding configuration with `--provider`,
`--embedding-provider`, `--model`, and `--embedding-model`; the default embedding
provider follows OpenAI or NVIDIA Build; Anthropic defaults to OpenAI embeddings
and needs that provider's key too. The shared `--model` configures chat/judging
and the native agent provider default; agent-specific models in committed eval
configuration keep their native override semantics. `--agents` overrides the
native provider default: OpenAI uses Codex, Anthropic uses Claude Code, and
NVIDIA Build uses OpenCode. CI installs Tier 2 or Tier 3 extras from the reviewed
tools commit. Tier 3 checks the selected environment with `doctor`; local mode
also installs pinned native CLIs into an isolated runtime directory.

For a reviewed candidate with `ci-model-routing-v1`, independently configure
Tier 1 LLM checks and Tier 3 gateway agents. These options run locally during
setup; CI later reads the committed configuration:

```bash
skillevaluator ci init --tiers 1,2,3 --mode preview --no-input \
  --paths skills --provider nv_build --embedding-provider nv_build \
  --model "$DEDUP_MODEL" \
  --tier1-llm --tier1-provider nv_build --tier1-model "$SECURITY_MODEL" \
  --tier3-provider openai-compatible --tier3-base-url "$GATEWAY_API_ROOT" \
  --tier3-model "$JUDGE_MODEL" --gateway-secret MY_INFERENCE_KEY \
  --agents claude-code,codex \
  --agent-model "claude-code=$CLAUDE_MODEL" --agent-model "codex=$CODEX_MODEL"
```

Set those model variables to exact accessible catalog IDs and the gateway
variable to its API root, commonly ending in `/v1`. `MY_INFERENCE_KEY` is the
name of a CI secret, not its value. Configure that repository Actions secret
and the Build secret `NVIDIA_API_KEY`; never pass a key value to `ci init`.
Use `--mode pr` after reviewing the preview. `--base-url` and
`--embedding-base-url` configure shared chat and embedding endpoints when
needed. Tier 1's immutable image uses native provider endpoints, so choose a
supported `--tier1-provider` when other tiers use a custom gateway.

The separate `--tier3-model` selects judging. Repeatable `--agent-model`
selects each native agent's model without replacing it with the judge's model.
`--tier1-model` also reaches SkillSpector; an empty value keeps the image's
provider defaults. Confirm that the pinned image supports the selected model.

Every combined report retains benchmark cards. On GitHub, add
`--benchmark-pr` with a `ci-benchmark-pr-v1` candidate to propose eligible cards
beside their skills through a dedicated update PR after default-branch runs.
The setup still contains two entry files. GitLab retains cards in its aggregate
artifact; automated benchmark-update MRs are not provided by this option.
Older descriptors reject these new options before writing unsupported CI files.

`--timeout-minutes` sets the native job duration (10 to 360, 300 by default).
The evaluator's runtime budget ends 20 minutes, or a quarter of a shorter
duration, earlier so setup and retained evidence fit. If setup outlasts that
reserve and GitLab stops the native job at its timeout, the job uploads no
native evidence and the aggregate reports that tier's evidence as missing.
GitLab receives a literal duration
so the configuration also works on GitLab 17.8. `--runner` chooses a GitHub
runner label or GitLab runner tag. GitHub defaults
to `ubuntu-24.04`; GitLab defaults to a configured `docker` tag. Docker mode
requires a working daemon, Docker CLI, Compose v2 and Buildx; GitLab's included Docker-in-Docker service requires
a privileged runner that shares `/certs/client` for the daemon's TLS client
certificates. Local mode needs a compatible native runner with Node/npm;
the generated GitLab local configuration removes the Docker service. Cloud
sandbox backends are outside this onboarding path.

`--runner` selects the native Tier 2/3 runner. Static and LLM-enriched Tier 1
keep their separate image-capable runner: GitHub uses `ubuntu-24.04`, and
GitLab uses an untagged image runner. Choosing a local Tier 3 runner therefore
does not add a Docker prerequisite to that native runner.

Merge both generated files onto the default branch **before opening the demo**.
The first setup PR/MR can show evaluation checks, but automatic comments need
the trusted feedback entrypoint to exist on the default branch. GitLab also
needs a protected default branch and a project access token with `api` scope
and the Maintainer role, stored as a scoped, protected
`SKILLEVALUATOR_GITLAB_TOKEN` limited to the `skillevaluator-feedback`
environment. GitLab before 18.4 also needs a separate project Reporter/read_api
`SKILLEVALUATOR_GITLAB_READ_TOKEN` for current-MR admission, scoped to the protected
execution environment (`skillevaluator` by default). The same read token is needed
on any GitLab version when the current protected default-branch API check rejects
`CI_JOB_TOKEN`, for example with HTTP 403. Configure or verify the read token's
project access rather than using the feedback publisher token. The read credential
is removed before provider work. Newer servers use `CI_JOB_TOKEN` by default where
the endpoint permits it. Online init/status report the server
version prerequisite; offline preview/save leave admission readiness unverified.
The MR jobs receive no comment token. GitLab's feedback request uses its
short-lived job token; the initiating user needs permission to create a pipeline
on the protected default branch, pass pipeline variables and run the protected
feedback environment. Without it, the request is an allowed failure: no comment
is posted and the Tier 1 gate is unchanged. GitLab's provider-backed and
feedback jobs record no deployments. Do not add deployment approval rules to
their environments: depending on the GitLab version, approvals either do not
gate these jobs or turn them into manual jobs. Restrict them with the protected
environments' deploy access.
See the platform pages for
[GitHub](#github-checks-and-feedback) and
[GitLab](#gitlab-checks-and-feedback) details.

### Upgrade a Tier 1-only consumer

Keep the existing workflow while previewing the explicit all-tier setup with a
compatible descriptor. Review the new evaluator and feedback pins, provider,
runner, protected environment, paths and Tier 1-only gate. Remove or disable
the old Tier 1 caller in the setup review once the new all-tier caller covers
the same roots, so a PR does not run two independent scans. Merge the new
feedback entrypoint and then open a fresh demo. Existing released bundles and
Tier 1-only callers keep their current behavior.

On GitLab, the Tier 1 and all-tier configurations share the root
`.gitlab-ci.yml`, which the CLI never replaces. `--mode preview` prints the
rendered files and the merge steps; `--mode save` and `--mode pr` stop with an
error while a different root file exists. First guard your other jobs with the
`SKILL_EVAL_FEEDBACK` rule and merge those guards to the default branch: the
setup MR's own feedback request runs the default branch's current
configuration. Then, in your setup MR, replace the Tier 1 `skill-evaluation.yml`
include with the two rendered includes, add the rendered variables, workflow
rules and job overrides, keep `test` in any `stages` list, and add
`.gitlab/skill-feedback.yml`. See
[Merge into an existing root configuration](#merge-into-an-existing-root-configuration).

## GitHub checks and feedback

The opt-in [CLI setup](#opt-in-to-three-tier-ci) creates
`.github/workflows/skill-checks-all-tiers.yml` and
`.github/workflows/feedback-all-tiers.yml`. This unreleased path needs a reviewed
candidate build and its bundled capability-bearing descriptor; the 0.8.0
[released downloads](../../bundles/0.8.0/README.md) remain Tier 1 only. `--tiers 1,2,3` selects the candidate's
bundled descriptor; `--release` can override it for an alternate reviewed candidate.
The [candidate source templates](https://github.com/NVIDIA/SkillEvaluator/tree/main/ci/templates)
contain placeholders and must be filled with reviewed immutable references.

The shared `evaluate-all-tiers.yml` and `feedback-all-tiers.yml` files define
reusable `workflow_call` workflows. Adding them to SkillEvaluator does not
enable checks or comments on SkillEvaluator's own pull requests. Each repository,
including SkillEvaluator itself, needs the generated caller pair committed
there, with the trusted feedback caller on its default branch.

The generated GitHub all-tier caller (`skillevaluator ci init --tiers 1,2,3`)
also carries `paths` filters. A pull request that changes nothing under the
selected skill folders or the two caller workflows starts no run and receives
no comment.

SkillEvaluator evaluates its own `skills/` collection with
`.github/workflows/skill-checks-all-tiers.yml`. Its consumer feedback listener is
`.github/workflows/skill-feedback-all-tiers.yml`, preserving the public reusable
publisher at `.github/workflows/feedback-all-tiers.yml`. The CLI selects this
alternate filename when the conventional feedback file already defines a
reusable workflow, and `ci status --tiers 1,2,3` recognizes it.

The repository callers pin reviewed evaluator workflows and tools, and the
verified released Tier 1 image. They use Docker, one attempt and one concurrent
task, and the committed native datasets. Tier 1 enables `--llm` and SkillSpector
with its own NVIDIA Build model. Tier 2 uses a separate NVIDIA Build chat model
and embeddings. Tier 3 selects both Codex and Claude Code through a configured
gateway, with explicit agent models and a separate judge model.

For this repository's profile, `NVIDIA_API_KEY` is a repository Actions
**secret**. Set the repository Actions **variable** `SKILL_EVAL_TIER3_BASE_URL`
to the gateway's versioned API root, and `SKILL_EVAL_GATEWAY_SECRET` to the name
of the existing repository Actions secret holding the gateway key. The selector
contains only a secret name, never the key itself. The caller resolves
`secrets[vars.SKILL_EVAL_GATEWAY_SECRET]` into the reusable workflow's generic
`LLM_API_KEY`; other repositories can select their own gateway and secret names
during local onboarding. Missing
prerequisites remain visible in the reports of runs that start the provider-backed
jobs (see below for which PRs skip them). A passing Tier 1 gate is not an
all-tier evaluation pass. Forks retain keyless Tier 1 checks. On PRs, ordinary software changes
select no skills, skill changes select the affected skills, and CI configuration
changes select the collection. Pushes to `main` and manual runs scan the collection.
With `scan-mode: auto`, the paid jobs (LLM-enriched Tier 1, Tier 2 and Tier 3)
run on a PR only when it changes a file under a configured skill root. A
same-repository configuration-only PR runs the keyless Tier 1 scan of every
skill, and its report lists Tier 2 and Tier 3 as skipped because no skill
changed. Fork PRs never run the paid tiers, so their reports keep the existing
Tier 2 and Tier 3 reason.

Because the generated caller only starts on skill or caller changes, the
same-repository PRs that now skip the paid jobs are the setup PRs: onboarding,
pin bumps, and model, secret or gateway changes to the caller. Such a PR no
longer shows a wrong model or a missing key. The first provider-backed run is the push to the
default branch after it merges, a manual run there, or the next PR that changes
a skill. To exercise a caller change with providers before merging, set
`scan-mode: all` in the caller on that PR and read the results in the run's
artifact; that PR gets no automatic comment, because the listener approves only
the default branch's caller. Before merging, restore `auto` and then refresh
`trusted-caller-blobs` for the final caller. `ci demo` cannot exercise an unmerged
setup: it requires the all-tier scan caller on the default branch and rejects a
setup that has not merged, and its skill-change PR branches from the default
branch. Use `ci demo` after the setup merges.

If a provider returns HTTP 404, verify that the selected model accepts chat
requests with your account. Presence in the public model catalog alone does not
prove serving availability. Change the caller's `model` input to an accessible
model and refresh `trusted-caller-blobs` before retrying; do not change global
provider defaults to troubleshoot one repository. A PR that only changes the
caller skips the provider-backed jobs, so the new model is first exercised by the
push after it merges, unless that PR temporarily uses `scan-mode: all` as described above.

The setup PR runs the keyless Tier 1 scan of every skill, but its first automatic comment requires
the consumer feedback listener on `main` and a subsequent PR run. Repository
maintainers configure branch protection separately if they want the Tier 1
aggregate check to block merges.

The evaluation workflow calls the existing hardened Tier 1 container workflow
at the same evaluator revision. Tier 2 and Tier 3 run independently with
`fail-fast: false`, Python 3.12, and only their tier's installed extras. Native
evaluator tools come from the pinned tools commit, while the candidate is
checked out separately at the exact PR head. Provider-backed jobs require the
current same-repository PR head or current default-branch commit, plus proven
write, maintain, or admin repository permission for the initiating actor.
Both actor permission and commit freshness are checked again before secrets
are used. Environment-scoped provider secrets in `skillevaluator` are
recommended; explicit mappings from repository CI secrets are also supported.
Reviewers are optional: repositories can add an approval rule when they want one, while
automatic execution does not depend on that GitHub plan feature. Protect
repository and CI workflow write access as part of maintaining this boundary.
Fork PRs retain keyless Tier 1 evidence. Provider-backed Tier 2/3 execution needs
an eligible same-repository run; unavailable evidence remains advisory and visible.

The aggregate job always runs, and only Tier 1 determines its required result.
Tier 2 and Tier 3 remain advisory when they fail or are incomplete; their native
statuses and reports remain visible. The deprecated `tier3-required` input is
ignored, including when set to `true`; it cannot make Tier 3 block CI. Keep the
aggregate check required in branch protection to enforce Tier 1, and review its
actual check name in a completed run. Reports are retained for 14
days in `skill-evaluator-results-attempt-N`, including failed and incomplete
results, with each tier's native JSON and HTML. **Re-run failed jobs** reuses
the latest evidence from tier jobs that were not rerun, and records each tier's
producing attempt as `source_attempt`. Artifact download links require
normal repository access; they do not create public report hosting.

The feedback workflow starts from `workflow_run` on the default branch. It
checks out only pinned trusted tools, validates the producer and current PR
head, and updates one comment with a single summary table. Each skill has
one-line results for Static and Security Validation, Deduplication, and Live
Agent Evaluation, plus one **Report** link to the exact run's combined private
artifact. The lines show status, recorded issue count and the main finding or
blocker; incomplete or missing evidence shows an unknown count. Download and
extract the artifact, then open `index.html` for the detailed aggregate and
native reports. See the
[comment and report contract](#retained-three-tier-ci-evidence).
The workflow never checks out the candidate with a write token. Fork PR runs and
producer attempts superseded by a re-run are skipped without a comment rather than
failed. The workflow defers only to a report comment whose run marker names a
newer same-repository producer run or attempt of this caller for the same pull
request head. Otherwise it updates one existing report comment in place,
preferring one whose marker verifies, and leaves any other report comments
untouched. Report comments with unverifiable markers are never deferred to.
Merge this feedback file onto the default branch before expecting the first
automatic comment. Changes to the caller must refresh its approved Git blob in
`trusted-caller-blobs` in both trusted jobs. An upgrade that changes the approved
caller or producer revision starts automatic feedback only after the updated
listener reaches the default branch; earlier candidate runs retain their artifacts.

A missing provider key, unavailable runtime, or missing committed native dataset
must remain visible as an incomplete or needs-action result. Local `ci init`,
`ci status`, and `ci demo` are setup and inspection commands; a successful setup
is not evidence that the three tiers passed. Optional Pages hosting remains a
separate opt-in; artifact feedback works without it.

### LLM security, gateway agents and benchmark cards

A candidate descriptor advertising `ci-model-routing-v1` supports
`--tier1-llm`, independent Tier 1 provider/model inputs, and separate Tier 3
gateway and judge settings. The protected Tier 1 job reuses the hardened image
with LLM security, SkillSpector semantic findings, verification and rubric
checks. An unconditional keyless baseline remains available for forks and
denied admission. For an authorized LLM run, missing or failed enriched evidence
fails the required Tier 1 gate; it is never replaced with a passing static scan.
With the default `scan-mode: auto`, a PR runs the enriched job only when it
changes a file under a configured skill root. CI configuration or unrelated
changes skip it, and the keyless scan, which covers every skill for a setup
change, remains the Tier 1 gate. Pushes to the default branch, manual runs and
`scan-mode: all` keep the enriched job.
The image supports OpenAI, Anthropic and NVIDIA Build at their native endpoints.

Tier 2 chat/embeddings can use one provider while Tier 3 uses an
`openai-compatible` gateway. Set `tier3-provider`, `tier3-base-url` and
`tier3-model` for the judge, then `agents` and `agent-models` for the actual
Claude Code/Codex models. A gateway must serve Chat Completions for judging,
Responses for Codex, and Anthropic Messages for Claude Code, including tool
calls. A model catalog listing alone does not verify these APIs or your key's
model access. The generic `LLM_API_KEY` reusable-workflow secret maps to the
repository secret selected by local `ci init --gateway-secret`.

The combined artifact includes `benchmarks/skill-NNN/BENCHMARK.md` and its
combined native JSON. The index links to both. Failed or incomplete evidence
retains a card with its actual publication status. Publication still requires
complete required-tier evidence even when Tier 2/3 are advisory CI jobs.
Older Tier 1 images that omit required-check scope cannot certify a passing
card; use a compatible image rather than inferring coverage from a green gate.

With capability `ci-benchmark-pr-v1`, `ci init --benchmark-pr` adds an optional
job to the same feedback file. After a verified current default-branch run, a
separate trusted publisher regenerates eligible cards from native JSON and
opens a benchmark-update PR. It never changes the evaluated source PR or pushes
to the default branch. This job needs `contents: write` and
`pull-requests: write`; the existing comment job keeps read-only contents
permission. GitHub must allow Actions to create PRs. If that setting is
disabled, the job reports its prepared branch and a manual PR action.

## GitLab checks and feedback

This path requires **GitLab 17.0 or later** for validated include inputs.
The unreleased [all-tier CLI path](#opt-in-to-three-tier-ci)
creates two thin files: `.gitlab-ci.yml` includes the pinned shared evaluation
configuration, and `.gitlab/skill-feedback.yml` includes pinned trusted feedback.
Install a reviewed candidate build; `--tiers 1,2,3` selects its bundled
schema-version 2 descriptor declaring `all-tiers-gitlab`. Use `--release` only
to override it with an alternate reviewed descriptor. Released 0.8.0
[released configurations](../../bundles/0.8.0/README.md) remain Tier 1 only. The CLI never replaces an existing,
different `.gitlab-ci.yml`, such as a released Tier 1 setup: `--mode preview`
prints the rendered files and the merge steps, and `--mode save` and `--mode pr`
stop with an error until you merge by hand.

### Merge into an existing root configuration

The trusted feedback job validates the root `.gitlab-ci.yml` of the MR
pipeline, so a merged file must keep this shape:

1. Make `include` exactly the two rendered entries: the pinned `all-tiers.yml`
   remote and the `.gitlab/skill-feedback.yml` local include with its rule.
   Remove the Tier 1 `skill-evaluation.yml` include, which defines the same
   `skill-evaluation` job, and any other include. Feedback rejects other
   includes and reads the root as plain YAML, so do not use `!reference` tags.
2. Add the rendered `variables`, replacing earlier `SKILL_EVAL_*` values.
   If the root defines `stages`, keep `test` in the list: every SkillEvaluator
   job uses stage `test`, and GitLab rejects a job whose stage is not listed.
3. Use the rendered `workflow: rules`. Keep your own workflow rules after the
   two `SKILL_EVAL_FEEDBACK` rules and before the final `when: never`, and
   only if they still allow merge-request, default-branch and web pipelines.
4. Replace any `skill-evaluation` override with the rendered
   `skill-evaluation` and `skill-evaluation-native` entries.
5. Guard every other job. Trusted feedback runs as a pipeline on the protected
   default branch, so an unguarded job, such as a deployment gated on the
   default branch, would run again for every MR. Start each job's rules with
   the guard. A job that has neither `rules` nor `only`/`except` gets the guard
   followed by `- when: on_success`:

   ```yaml
   deploy:
     script: ./deploy.sh
     rules:
       - if: '$SKILL_EVAL_FEEDBACK == "true"'
         when: never
       - if: '$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH'
   lint:
     script: make lint
     rules:
       - if: '$SKILL_EVAL_FEEDBACK == "true"'
         when: never
       - when: on_success
   ```

   Merge these guards to the default branch before the setup MR. Its own
   feedback request starts a pipeline from the current default-branch
   configuration, and the guard is inert until `SKILL_EVAL_FEEDBACK` is set.
   Write these guards as rules: `.skill-evaluation-not-feedback` is defined by
   the all-tiers include, so until the setup MR merges,
   `extends: .skill-evaluation-not-feedback` fails validation with
   `unknown keys in extends`. After that, a job added later with neither
   `rules` nor `only`/`except` can use that `extends` instead.
6. Add `.gitlab/skill-feedback.yml` as rendered and merge both files through
   a setup MR. After it merges, `skillevaluator ci status --tiers 1,2,3`
   reports whether the root is recognized and warns about unguarded root jobs.

The shared configuration runs the pinned Tier 1 image plus parallel native
Tier 2 and Tier 3 jobs. Tier 2 runs without a Docker service. Only Tier 1 is
required; Tier 2 and Tier 3 are advisory,
including failed or incomplete outcomes. The deprecated
`SKILL_EVAL_TIER3_REQUIRED` variable is ignored, including when set to `'true'`;
it cannot make Tier 3 block CI. Native execution jobs retain failed outcomes
without stopping their parallel sibling. The aggregate is the required merge
gate and follows Tier 1 only: failed, incomplete or missing Tier 1 evidence
fails it, while advisory outcomes remain visible in retained reports.
The always-running aggregate job
retains `skill-evaluator-results/ci-results.json`, `index.html`, `summary.md`,
and each tier's native reports for 14 days. Reports stay behind your normal
GitLab artifact access controls.

A reviewed `ci-model-routing-v1` candidate can opt into
`SKILL_EVAL_TIER1_LLM: "true"`, with independent
`SKILL_EVAL_TIER1_PROVIDER` and `SKILL_EVAL_TIER1_MODEL` settings. The additional
Tier 1 image job runs in the protected execution environment, rechecks current
MR/default-branch eligibility, and uses no Docker-in-Docker service. Its
enriched evidence determines the Tier 1 gate for an eligible run; missing
evidence cannot fall back to a passing static baseline. Default-branch enrichment
is limited to `push` and `web` pipelines, matching provider admission. Scheduled,
API, trigger and downstream default-branch pipelines retain the keyless Tier 1
baseline, as do forks. With `SKILL_EVAL_SCAN_MODE: auto`, an MR that changes no
file under a configured skill root skips the paid jobs (enriched Tier 1, Tier 2
and Tier 3) before reading a provider key. The aggregate recomputes that decision
from the checkout with the pinned tools, never from an artifact. It then keeps
the keyless report as the Tier 1 gate and records Tier 2 and Tier 3 as skipped.

Use `SKILL_EVAL_TIER3_PROVIDER`, `SKILL_EVAL_TIER3_BASE_URL` and
`SKILL_EVAL_TIER3_MODEL` for a separate live-agent/judge route, and
`SKILL_EVAL_AGENT_MODELS` for newline-separated `AGENT=MODEL` selections.
Generic gateways use environment-scoped `SKILL_EVAL_LLM_API_KEY`; chat and
embedding settings can remain independent. These variables are also generated
by the [local onboarding options](#opt-in-to-three-tier-ci).

The aggregate retains source-bound `BENCHMARK.md` cards and combined JSON under
`benchmarks/skill-NNN/`. Publication requirements remain separate from advisory
CI policy. GitLab currently publishes these cards as authenticated artifacts;
the optional `--benchmark-pr` source-update feature is GitHub-only.

Trusted MR feedback uses one summary table with a row per skill. Each row has
one-line results for Static and Security Validation, Deduplication, and Live
Agent Evaluation, with status, recorded issue count and the main finding or
blocker. Incomplete or missing evidence shows an unknown count. One **Report**
link per skill downloads the combined aggregate artifact; extract it and open
`index.html` for the detailed aggregate and native reports. Separate tier
sections and per-tier links stay out of the comment. See the
[comment and report contract](#retained-three-tier-ci-evidence).

Configure a native-job runner matching `SKILL_EVAL_RUNNER` (default `docker`).
Tier 1, aggregation and feedback jobs have no runner tags and also need a
compatible runner that accepts untagged jobs and supports their container
images. This remains necessary when a Linux or macOS shell runner runs the
native jobs in local mode. Configure the protected
`SKILL_EVAL_ENVIRONMENT` (default `skillevaluator`) and provider secrets scoped
to that execution environment. The native jobs and the trusted feedback job
use `environment: action: prepare`: each receives its environment's scoped
variables and access protection but records no deployment, so GitLab's
**Prevent outdated deployment jobs** setting cannot drop an older MR's jobs.
Do not add deployment approval rules to these environments. Depending on the
GitLab version and its `prevent_blocking_non_deployment_jobs` feature flag,
approval rules either do not gate these jobs or turn them into manual jobs
that wait to be run. Restrict who can run them with the protected
environments' deploy access. The Tier 3 Docker
service needs a privileged runner with Docker-in-Docker enabled. The daemon accepts only TLS clients on
port 2376 (`DOCKER_TLS_CERTDIR: /certs`), so agent containers cannot use the
Docker API. Configure the runner with `volumes = ["/certs/client"]` so the job
can read the generated client certificates. Docker mode also requires the Docker CLI,
Compose v2 and Buildx; the shared job installs these clients in supported root
Linux containers and checks readiness before evaluation. For local mode the generated caller
removes that service; use a Linux or macOS native runner with Python 3.12,
Git and Node/npm. Missing native tools must be installed by the runner
administrator; system-package bootstrap is limited to root Linux containers. The
trusted tools are installed from `SKILL_EVAL_TOOLS_REF`, an immutable commit.
Local agent CLIs are pinned and bootstrapped before selected-mode readiness
checks. The current same-project MR head is checked before model execution;
fork MRs do not receive provider-backed jobs.

GitLab 17.0–18.3 requires a separate project access token with **Reporter** role
and **read_api** scope for current-MR admission. Store it as masked
`SKILLEVALUATOR_GITLAB_READ_TOKEN`, scoped only to the configured execution
environment (`skillevaluator` by default). Protect that environment to approved CI maintainers. For ordinary
unprotected MR branches, the variable must be available on those refs; enabling
GitLab's protected-variable restriction would withhold it. The admission helper
uses this read credential once, refuses HTTP redirects, and the job removes it
before runtime installation, doctor or provider execution.

GitLab 18.4 and later support the default short-lived `CI_JOB_TOKEN` path for
this MR endpoint. Admission also verifies the current protected default-branch
head through the repository branches API. If a job token cannot read either
endpoint, such as an HTTP 403 response, configure or verify the same separate
`SKILLEVALUATOR_GITLAB_READ_TOKEN` with project Reporter/read_api access. This
default-branch prerequisite can apply on newer GitLab versions too; use the
execution environment's read credential, never the feedback publisher token.
`ci init --mode pr` and `ci status` check the server version through the local
authenticated API.
Offline preview/save keep the version unverified. These local commands report
the prerequisite and do not claim to validate a stored CI secret's privileges.
The setup still consists of two consumer files.

Protect the default branch and create a project access token with `api` scope
and the Maintainer role. The publisher reads source-pipeline variables to verify
provenance as well as reading artifacts and updating MR notes; GitLab 17.8
rejects the variables request for a Developer token. Store the token as masked, protected
`SKILLEVALUATOR_GITLAB_TOKEN`, scoped only to the protected
`skillevaluator-feedback` environment. The MR pipeline never receives that
token. After aggregation, a small job uses its short-lived `CI_JOB_TOKEN` with
GitLab's pipeline trigger API to request a separate pipeline on the protected
default branch. It sends only `SKILL_EVAL_FEEDBACK=true` and the source pipeline
ID, without forwarding repository variables or caller-supplied pipeline overrides.
GitLab also adds its own `TRIGGER_PAYLOAD` file variable. The publisher accepts
that bounded file entry alongside the two selectors and treats its contents as
opaque; additional caller variables remain disallowed.
The API supports same-project pipelines; GitLab's declarative multi-project
bridge rejects a target in the same project. That trusted pipeline uses validated include inputs for fixed evaluator pins,
API host and project, so forwarded MR variables cannot replace its privileged
bootstrap. It fetches only pinned evaluator tools and authenticates the server's
upstream pipeline and source job relationship, then validates the source
configuration, aggregate artifact, MR and current commit before updating one
comment. The publisher needs access to the instance's REST and GraphQL APIs;
the pipeline's persisted MR association supports GitLab 17.x as well.
The initiating pipeline user must be allowed to create a pipeline on
the protected default branch, pass the two pipeline variables and run the
protected feedback environment. Configure these permissions for authorized CI
maintainers; a job token does not bypass protected-branch or protected-environment permissions.
The feedback request is advisory. When GitLab refuses it, for example because
the MR pipeline user cannot run pipelines on the protected default branch or
may not pass pipeline variables, the request job fails as an allowed failure:
the pipeline shows a warning, no comment is posted or updated, and the aggregate
continues to enforce Tier 1. Server and network errors fail the job the same
way; its log says whether to fix permissions or retry. Before the setup MR
installs feedback on the default branch, no comment is posted and the aggregate
still enforces Tier 1.

For a rejected publisher, open its job log. The check location refers to the
pinned `ci/feedback.py` revision, and the category distinguishes validation,
authentication and transport failures.

Install both configuration files on the default branch before the first demo
MR. GitLab must permit the downstream pipeline and the protected feedback job;
a setup MR alone cannot establish trusted automatic feedback. On instances
that restrict protected variables in MR pipelines, approve an eligible protected
execution path according to your project policy; missing provider credentials
remain actionable evidence rather than a pass. Review the pipeline's actual
aggregate check before enabling merge restrictions; it gates Tier 1 only.

The CLI's `--timeout-minutes` option writes a literal native-job duration and
a shorter evaluator budget. The default is 300 minutes. The budget ends 20
minutes, or a quarter of a shorter duration, before the job timeout, leaving
time for setup and for retaining evidence: 300 minutes gives 16800 seconds.
On GitLab 17.8, CI variables are not expanded in a job's `timeout` field. For
a manual consumer, set the literal directly:

```yaml
variables:
  SKILL_EVAL_TIMEOUT_SECONDS: '2040'
skill-evaluation-native:
  timeout: 45 minutes
```

Timeout and runner-tag overrides on `skill-evaluation-native` apply to both
native tiers. The
`skill-evaluation-native-tier2` job extends that public job and clears its
Docker service and TLS client variables; a caller's `services` override applies
to Tier 3. The generated local-mode caller sets `services: []` on
`skill-evaluation-native` to disable Tier 3's Docker service too.

Keep a manual tier selection quoted, for example `SKILL_EVAL_TIERS: '1,2,3'`.
GitLab's YAML parser reads the unquoted value as the number `123`. The CLI
quotes generated variable values automatically.

## Managed three-tier CI

For two small consumer files and automatic artifact-backed PR/MR feedback, use
the opt-in [three-tier CLI onboarding](#opt-in-to-three-tier-ci).
This unreleased capability gates only on Tier 1 and runs advisory Tier 2 and
paired Tier 3, defaults to Docker, and supports native local mode. Failed or
incomplete advisory results retain their native evidence without blocking CI.
The deprecated Tier 3 required settings are ignored. It requires an
explicit compatible release descriptor and committed native datasets. The
released Tier 1-only workflows and bundles continue to work unchanged.

The shared evaluator retains native tier reports and an aggregate manifest,
including incomplete outcomes. A separate trusted feedback entrypoint on the
default branch verifies the current head and updates one comment linking the
exact private CI artifact. Local setup and demo commands prepare the remote
run; inspect that run's gates and reports before considering onboarding verified.
The [hand-written CI recipes](../../../docs/ci-integration.mdx) retain their own CLI flags and policy choices.

## Retained three-tier CI evidence

The opt-in all-tier workflows retain one aggregate artifact for 14 days. GitHub
names it `skill-evaluator-results-attempt-N`; GitLab retains the aggregate job's
`skill-evaluator-results/` directory. The trusted feedback comment links the
actual artifact download for that run. Authentication and artifact expiry still
apply; this link does not imply publicly hosted HTML.

Current candidate aggregates also retain `benchmarks/skill-NNN/BENCHMARK.md`
and `benchmarks/skill-NNN/skillevaluator.json` for each selected skill. The
index links these source-bound cards and combined JSON alongside native tier
reports. Cards retain failed and incomplete publication states. A passing
Tier 1 CI gate does not certify benchmark eligibility; inspect the combined
JSON's `publication.status` and `publication.eligible`.

The PR or MR comment presents one summary table with a row for each skill.
Each row shows a one-line status, issue count and the most important available
finding or blocker for **Tier 1 — Static and Security Validation**,
**Tier 2 — Deduplication** and **Tier 3 — Live Agent Evaluation**. A single
**Report** link per skill opens the combined report artifact for that run;
download and extract it, then open `index.html` to inspect the detailed aggregate
and native reports. The comment keeps tier details in those reports rather
than adding separate tier sections or report links.

This schematic shows the information in a row; real results supply the statuses,
counts, findings and authenticated **Report** link:

| Skill | Tier 1 — Static and Security Validation | Tier 2 — Deduplication | Tier 3 — Live Agent Evaluation | Combined report |
| --- | --- | --- | --- | --- |
| `skills/example` | Passed · 0 issues | Failed · 1 issue · duplicate content found | Incomplete · issues unknown · paired evidence missing | Report |

Issue counts summarize unique actionable findings plus independent validation
errors and warnings from retained native evidence; informational notes and
configuration-only scope notices are excluded. Incomplete or missing evidence
shows an unknown count; it cannot imply zero findings or a completed check.
The Tier 1 gate and each advisory tier's status stay separate, so this example
can pass the required CI gate while still needing review of Tier 2 and Tier 3.
Every tier that runs selects the same skills: the changed skills, or every
skill when the pull request changes CI setup under `.github/`, `.gitlab/`,
`.skillevaluator/ci` or `.gitlab-ci.yml` (on GitHub, also when it changes more
files than the API can list). A cell for a skill that its tier did not select
reads `Skipped · issues unknown · Not selected for this tier`. With
`scan-mode: auto`, a merge request or same-repository pull request that changes
no file under a configured skill root skips Tier 2 and Tier 3. Only Tier 1 then
scans, covering every skill for a setup change, and the Tier 2 and Tier 3 cells read
`Skipped · issues unknown · No file under a configured skill root changed; this paid tier did not run.`

```text
ci-results.json
index.html
source-attempts.json   # GitHub: run attempt whose artifact supplied each tier
summary.md
tier1/                 # hardened scanner reports, including summary.json
tier2/ci-tier.json
tier2/skill-001/       # native retained JSON and HTML
tier3/ci-tier.json
tier3/skill-001/       # native retained JSON and HTML
  diagnostics/        # redacted worker logs when execution failed or is incomplete
```

`ci-results.json` uses report contract `skillevaluator-ci-v1` and records the
repository, exact commit, platform, run ID, attempt, environment, and each
tier's required/advisory policy: Tier 1 has `required: true`, while Tier 2 and
Tier 3 have `required: false`. On GitHub, each tier with retained evidence
records `source_attempt`, the run attempt that produced it; after **Re-run
failed jobs** it can be earlier than the aggregate's own attempt. A tier can be
`passed`, `failed`, `incomplete`, `skipped`, `not_run`, `needs_action`, or
`neutral`. `neutral` means Tier 3 paired execution completed with a neutral
dimension-based verdict; Skill Lift has its own independent diagnostic band.
Like every Tier 3 status, it is advisory. Failed,
incomplete or missing Tier 1 evidence fails the aggregate. Tier 2/3 failures and
incomplete results remain advisory and retain their native status; a passing CI
gate does not mean those tiers passed. A check that never executed cannot become an empty success.
The deprecated Tier 3 required settings are ignored in CI. Standalone validation
and benchmark publication retain their separate policies.

Tier 2 reports retain deduplication and catalog-similarity findings;
Tier 3 reports retain paired scores and coverage, Skill Lift, attempts,
latency/resource metrics and available trial links. Download and extract the
artifact, then open `index.html` to inspect the aggregate and native reports.
Detailed files remain available even when a tier fails or needs configuration.
When a native skill's execution fails or is incomplete, or its worker exits
abnormally, the skill retains bounded, redacted worker, Docker and
agent-installation diagnostics under `diagnostics/`, even if no canonical report
was produced or its verdict is failed. A complete run with a failed or neutral
verdict retains none. The worker's `command.log` comes first, then logs from
one trial of each Harbor job (a failing trial when there is one), at most four
trials per skill, with failing trials chosen before passing ones. All
skills of one tier share a limit of 1000 diagnostic files and 16 MiB. Each
`index.json` records retained, omitted and truncated files, including those
omitted at a limit.
Catalog failures retain diagnostics under `tier2/catalog/diagnostics/`.
Agent trajectories, credentials and runtime configuration are excluded.
The publisher renders validated retained JSON; it does not execute artifacts
or trust candidate-provided HTML in its privileged feedback job. Optional
report hosting is independent from these private artifact links.
