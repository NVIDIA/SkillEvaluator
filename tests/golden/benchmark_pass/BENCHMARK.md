# Skill Benchmark: demo-skill

> ✅ **Overall verdict: PASS — Recommended for publication**

### Why This Verdict

- Decision: **PASS** — every publication-required tier passed and the Tier 3 dimension gate passed.
- Tier 3 evidence: **PASS** (publication-complete; required for publication). Dimension gate: an agent passes when every dimension is at or above 50% and fails when any dimension is below 40%.

| Agent | Dimension gate | Evidence the gate read |
|---|---|---|
| Claude Code | pass | every dimension at or above the 50% PASS boundary (lowest: Security 92.0%) |
| Codex | pass | every dimension at or above the 50% PASS boundary (lowest: Security 88.0%) |

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `demo-skill`
- Source digest: `sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` (skill-evaluator-source-tree/2)
- Evaluation date: 2026-07-24
- Evaluator version: `0.8.2`
- Evaluated source: `NVIDIA/demo_skills`
- Evaluated source revision: `2263a2ebdab903e87f7e7c0a001d22c3a926a9cf`
- Evaluator container revision: `ghcr.io/nvidia/skillevaluator@sha256:0117bc2e3e54da4244a656466526c5b1b5a559ea0117bc2e3e54da4244a65646`
- Agents: Claude Code (`claude-sonnet`), Codex (`gpt-codex`)
- Tasks: 8 evaluation tasks (6 positive, 2 negative)
- Dataset digest: `sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef` (skill-evaluator-dataset-snapshot/1)
- Tier 3 run ID: `fixture-demo-skill-run`
- Attempts per task: 3
- Environment: `Isolated sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 47.0% → 92.0% (+45.0 points) | 55.0% → 88.0% (+33.0 points) |
| Required checks passed | N/A | N/A |
| Goals met | N/A | N/A |
| Security clean | N/A | N/A |
| Security | 47.0% → 92.0% (+45.0 points) | 55.0% → 88.0% (+33.0 points) |
| Correctness | 47.0% → 92.0% (+45.0 points) | 55.0% → 88.0% (+33.0 points) |
| Discoverability | 47.0% → 92.0% (+45.0 points) | 55.0% → 88.0% (+33.0 points) |
| Effectiveness | 47.0% → 92.0% (+45.0 points) | 55.0% → 88.0% (+33.0 points) |
| Efficiency | 47.0% → 92.0% (+45.0 points) | 55.0% → 88.0% (+33.0 points) |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is `skill score - baseline score`, computed from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

**Not scored** means a signal the dimension needs produced no score, and **Not applicable** means no eval case gave its signals anything to judge; neither is a score of zero. Scoring Methodology below explains each dimension and where its evidence comes from.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 1 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 1 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **LOW** INTER\_SKILL/partial\_overlap: Partial overlap with another skill (`SKILL.md`)

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | What it measures | Scored signals | Evidence source | Baseline in this run |
|---|---|---|---|---|
| Security | Is it safe to use? Avoids unsafe operations, secret leakage, and unauthorized access. | `security` (100%) | Deterministic checks of the agent's commands and outputs | Yes, a no-skill baseline was scored |
| Correctness | Is the answer correct? Produces factually accurate output. | `accuracy` (100%) | An LLM judge comparing the final answer with the reference answer | Yes, a no-skill baseline was scored |
| Discoverability | Was the right skill loaded when needed? Activates the right skill when relevant; ignores unrelated skills. | `skill_execution` (100%) | Deterministic checks of the agent's trajectory | Yes, a no-skill baseline was scored |
| Effectiveness | Did the skill help complete the task? Reaches the user's goal and follows the expected behavior. | `goal_accuracy` (50%) + `behavior_check` (50%) | An LLM judge of whether the user's goal was met; an LLM judge of the expected workflow | Yes, a no-skill baseline was scored |
| Efficiency | Did it avoid wasted tool or skill usage? Uses the skill and tools productively without wasted work. | `skill_efficiency` (100%) | Deterministic checks of the agent's trajectory | Yes, a no-skill baseline was scored |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Token efficiency is a separate report-only signal. It does not change a dimension score or the overall verdict.
- The baseline is the same eval cases run by the same agent without the skill. Every dimension is compared with it when the baseline arm ran and scored that dimension.

Which number answers which question:

- **Overall verdict** (top of this card): whether the evidence supports publishing the skill. It needs every required tier, not only Tier 3 scores.
- **Dimension score**: how well the agent did on one question above when it had the skill, from 0% to 100%.
- **Overall** row: the equal-weight mean of the five dimension scores with the skill.
- **Uplift**: the displayed skill score minus the displayed no-skill baseline score. Published Skill Lift compares Overall only on the metrics and cases both arms scored, so it can differ. Uplift does not set the verdict.
- **pass@k** (HTML report and `pass_at_k_lift.json`): the share of eval cases with at least one passing attempt out of k. It measures how repeatable success is, separately from the dimension scores.
- **Quality Score** (Tier 1, out of 100): a static review of the skill's files. It is not an input to any Tier 3 score or uplift.

Signals present in this run:

- `accuracy` (Accuracy): final-answer correctness against the reference answer.

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
