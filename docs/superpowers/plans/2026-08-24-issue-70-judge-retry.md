# Issue 70 Judge Retry Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Tier 3 accuracy and custom goal-accuracy judges recover from one malformed (including empty) or schema-invalid response without weakening fail-closed trial and arm scoring.

**Architecture:** Add one internal call-parse-validate-retry helper to both the shared judge module and the standalone Harbor verifier template. Accuracy and goal accuracy provide strict payload validators, default to exactly 4096 output tokens, and retry only parse/schema failures. The helper accepts pair-returning and provenance-returning call functions so the template goal judge keeps the provider/model from the decisive attempt; behavior-check salvage remains isolated. SkillEvaluator-managed Harbor verifier configs reserve 600 seconds for six sequential direct 90-second provider attempts plus artifact-write overhead, while authored native verifier timeouts remain unchanged. Provider SDK work outside the direct retry helper, including OpenAI RAGAS and Bedrock SDK retries, can require an operator-supplied Harbor timeout multiplier.

**Tech Stack:** Python 3.12/3.13, pytest, Harbor 0.13.2, Ruff, Docker, NVIDIA Build for live verification.

**Scope boundary:** Do not persist raw model responses. They can echo arbitrary task or credential material into portable reports, and provider transports do not consistently expose finish reason or usage. Safe response telemetry and rescore support require separate designs; issue #70's primary retry and token-budget defect is handled here.

---

### Task 1: Test-drive retry, budget, provenance, and fail-closed contracts

**Files:**
- Modify: `tests/tier3/test_judge_result_semantics.py`
- Modify: `tests/tier3/test_judge_failure_artifacts.py`
- Verify: `tests/tier3/test_judge_parse_robustness.py`
- Verify: `tests/test_issue55_collector_truth.py`

- [ ] **Step 1: Add failing unit tests for malformed and schema-invalid recovery**

In `tests/tier3/test_judge_result_semantics.py`, define valid retry payloads:

```python
VALID_ACCURACY_RESPONSE = json.dumps(
    {"criteria": _CRITERIA, "score": 0.6, "reason": "valid retry"}
)
VALID_GOAL_RESPONSE = json.dumps(
    {
        "user_goal": "complete the task",
        "end_state": "task completed",
        "achieved": True,
        "score": 1.0,
        "reason": "valid retry",
    }
)
```

For both values of the existing `judge_module` fixture, add accuracy tests covering:

- malformed text → valid response;
- schema-invalid complete criteria (`ACTIONABLE: "true"`) → valid response;
- malformed → malformed returns a structured error after exactly two calls;
- first provider error returns a structured error after exactly one call;
- malformed → retry provider error returns a bounded, redacted structured error after two calls;
- default `max_tokens` is exactly `4096` on both attempts;
- attempt two contains `previous reply could not be parsed or validated`.

Use a real scripted function at the judge transport boundary and assert returned judge behavior, not the mock itself:

```python
def test_accuracy_retries_schema_invalid_response_once_and_recovers(judge_module, monkeypatch):
    invalid = json.dumps({"criteria": {**_CRITERIA, "ACTIONABLE": "true"}, "score": 0.6})
    calls = []

    def fake_call(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return (invalid, None) if len(calls) == 1 else (VALID_ACCURACY_RESPONSE, None)

    monkeypatch.setattr(judge_module, "call_public_llm", fake_call)
    result = judge_module.judge_accuracy("question", "ground truth", "agent response")

    assert result["score"] == 0.6
    assert [kwargs["max_tokens"] for _, kwargs in calls] == [4096, 4096]
    assert "previous reply could not be parsed or validated" in calls[1][0]
```

For goal accuracy, cover both shared and template surfaces explicitly. Shared tests patch `call_public_llm` with pair responses. Template tests disable RAGAS and patch `_call_public_llm_with_provenance` with triples. Cover:

- malformed → valid;
- schema-invalid `{"achieved": "true"}` → valid;
- invalid → invalid exactly twice;
- initial provider error exactly once;
- invalid → retry provider error twice with redaction;
- exactly `4096` on both attempts;
- template success and final error use the second attempt's `provider` and `model`.

Add shared explicit-override tests for accuracy and goal:

```python
@pytest.mark.parametrize("judge_name", ["judge_accuracy", "judge_goal_accuracy"])
def test_shared_structured_judge_keeps_explicit_max_tokens_override(monkeypatch, judge_name):
    calls = []
    response = VALID_ACCURACY_RESPONSE if judge_name == "judge_accuracy" else VALID_GOAL_RESPONSE

    def fake_call(prompt, **kwargs):
        calls.append(kwargs)
        return response, None

    monkeypatch.setattr(llm_judge, "call_public_llm", fake_call)
    judge = getattr(llm_judge, judge_name)
    judge("question", "ground truth", "agent response", max_tokens=2048)

    assert calls == [{"max_tokens": 2048}]
```

- [ ] **Step 2: Add failing real-verifier recovery and terminal-failure tests**

In `tests/tier3/test_judge_failure_artifacts.py`, use `_load_verifier` so the actual standalone `main()` writes real rich and numeric reward files.

Recovery test transport sequence:

```python
pair_responses = [
    ("not-json", None),
    (
        json.dumps(
            {
                "criteria": {
                    "SKILL_IDENTIFIED": True,
                    "ACTION_CORRECT": True,
                    "FACTUALLY_ACCURATE": True,
                    "TASK_ADDRESSED": True,
                    "ACTIONABLE": True,
                },
                "score": 1.0,
                "reason": "accuracy recovered",
            }
        ),
        None,
    ),
    (json.dumps({"results": [{"step": 1, "passed": True}], "score": 1.0}), None),
]
goal_responses = [
    ("not-json", None, {"provider": "nv_build", "model": "first-model"}),
    (
        json.dumps({"achieved": True, "score": 1.0, "reason": "goal recovered"}),
        None,
        {"provider": "nv_build", "model": "retry-model"},
    ),
]
```

Patch only `call_public_llm` and `_call_public_llm_with_provenance`, disable RAGAS, run `verifier.main()`, then assert:

```python
assert "evaluation_status" not in rich
assert "evaluation_errors" not in rich
assert rich["details"]["accuracy"]["score"] == 1.0
assert rich["details"]["goal_accuracy"]["model"] == "retry-model"
assert numeric["accuracy"] == numeric["goal_accuracy"] == numeric["behavior_check"] == 1.0
assert [kwargs["max_tokens"] for _, kwargs in pair_calls] == [4096, 4096, 4096]
assert [kwargs["max_tokens"] for _, kwargs in goal_calls] == [4096, 4096]
```

Add a separate terminal-failure test where accuracy returns invalid text twice, goal and behavior return valid payloads, and the invalid text contains a configured synthetic credential. Assert:

```python
with pytest.raises(SystemExit) as exc_info:
    verifier.main()
assert exc_info.value.code == 1
assert len(accuracy_attempts) == 2
assert rich["evaluation_status"] == "failed"
assert rich["accuracy"] is None
assert rich["details"]["accuracy"]["status"] == "error"
assert len(rich["evaluation_errors"]["accuracy"]) <= 512
assert synthetic_credential not in json.dumps(rich)
assert "accuracy" not in numeric
```

This proves two invalid attempts still produce the deliberately incomplete sentinel consumed by the existing collector fail-closed test.

- [ ] **Step 3: Run the new tests and verify RED**

```bash
uv run python -m pytest -q \
  tests/tier3/test_judge_result_semantics.py \
  tests/tier3/test_judge_failure_artifacts.py \
  -k 'retry or max_tokens or malformed_accuracy_and_goal'
```

Expected: recovery/default-budget assertions fail because accuracy and goal currently call once at 1024 tokens. Confirm failures are behavioral assertions, not setup errors.

---

### Task 2: Implement one validated-JSON retry for accuracy and goal

**Files:**
- Modify: `src/skillevaluator/tier3/eval_core/llm_judge.py:565-730`
- Modify: `src/skillevaluator/tier3/harbor/templates/eval.py:3280-3470`

- [ ] **Step 1: Add shared constants and payload validators in both modules**

```python
STRUCTURED_JUDGE_MAX_TOKENS = 4096

_JUDGE_RETRY_REMINDER = (
    "\n\nIMPORTANT: Your previous reply could not be parsed or validated. Respond with ONLY the "
    "minified JSON object on a single line -- no markdown fences, no prose, and keep explanations brief."
)


def _accuracy_payload_error(parsed):
    if not isinstance(parsed, dict):
        return "Judge response was not a valid JSON object"
    criteria = parsed.get("criteria")
    criteria_valid = _valid_accuracy_criteria(criteria)
    if "criteria" in parsed and not criteria_valid:
        return "Judge response contained invalid accuracy criteria"
    if _finite_score(parsed.get("score")) is None and not criteria_valid:
        return "Judge response contained no valid accuracy score or complete criteria"
    return None


def _goal_payload_error(parsed):
    if not isinstance(parsed, dict):
        return "Judge response was not a valid JSON object"
    if not isinstance(parsed.get("achieved"), bool):
        return "Judge response contained an invalid achieved value"
    if "score" in parsed and _finite_score(parsed["score"]) is None:
        return "Judge response contained an invalid goal score"
    return None
```

Use existing module-style types in `llm_judge.py` and dependency-free Python in the standalone template.

- [ ] **Step 2: Add the internal call-parse-validate-retry helper**

Implement this contract in both modules, passing `_extract_json` in the shared module and `extract_json` in the template:

```python
def _call_validated_json_judge(prompt, validate, call, extract, **call_kwargs):
    call_kwargs.setdefault("max_tokens", STRUCTURED_JUDGE_MAX_TOKENS)

    def invoke(call_prompt):
        content, error, *metadata = call(call_prompt, **call_kwargs)
        provenance = metadata[0] if metadata and isinstance(metadata[0], dict) else {}
        parsed = extract(content) if content else None
        validation_error = validate(parsed) if not error else None
        return parsed, error, provenance, validation_error

    parsed, error, provenance, validation_error = invoke(prompt)
    if error:
        return None, f"LLM judge error: {error}", provenance
    if validation_error is None:
        return parsed, None, provenance

    parsed, error, provenance, validation_error = invoke(prompt + _JUDGE_RETRY_REMINDER)
    if error:
        return None, f"LLM judge retry error: {error}", provenance
    if validation_error is not None:
        return None, f"{validation_error} after retry", provenance
    return parsed, None, provenance
```

Do not include raw response content in error text or metadata. `_judge_error` remains the bounded/redacted terminal constructor.

- [ ] **Step 3: Route judges through the helper without changing scoring**

- Shared and template accuracy call the helper with `call_public_llm`, then compute the already-validated criteria/score exactly as before.
- Shared goal calls the helper with `call_public_llm`.
- Template custom goal calls the helper with `_call_public_llm_with_provenance` and includes returned provenance in both success and error results.
- Classify a successful provider response with empty model content separately from transport/configuration failures so the shared path retries it as malformed, matching the standalone template. Keep all genuine provider errors one-call.
- Initial provider errors return immediately; only parse/schema validation failures retry.
- Exactly one retry is permitted.
- Define `BEHAVIOR_JUDGE_MAX_TOKENS = STRUCTURED_JUDGE_MAX_TOKENS`; do not refactor behavior parsing or salvage into the generic helper.
- Raise only SkillEvaluator-managed generated/scaffolded/injected Harbor verifier defaults from 180 to 600 seconds. Preserve explicit native-task timeout overrides; existing native 180-second values must be updated explicitly when additional provider headroom is required.

- [ ] **Step 4: Run focused GREEN verification**

```bash
uv run python -m pytest -q \
  tests/tier3/test_judge_result_semantics.py \
  tests/tier3/test_judge_parse_robustness.py \
  tests/tier3/test_judge_failure_artifacts.py \
  tests/test_issue55_collector_truth.py::test_mixed_failed_condition_suppresses_published_quality_and_paired_artifacts
```

Expected: all pass. Recovery writes complete rewards; terminal retry exhaustion remains unscoreable and the collector still suppresses incomplete arm aggregates.

- [ ] **Step 5: Run touched-file lint and commit**

```bash
uv run ruff check \
  src/skillevaluator/tier3/eval_core/llm_judge.py \
  src/skillevaluator/tier3/harbor/templates/eval.py \
  tests/tier3/test_judge_result_semantics.py \
  tests/tier3/test_judge_parse_robustness.py \
  tests/tier3/test_judge_failure_artifacts.py
git add \
  src/skillevaluator/tier3/eval_core/llm_judge.py \
  src/skillevaluator/tier3/harbor/templates/eval.py \
  tests/tier3/test_judge_result_semantics.py \
  tests/tier3/test_judge_parse_robustness.py \
  tests/tier3/test_judge_failure_artifacts.py \
  docs/superpowers/plans/2026-08-24-issue-70-judge-retry.md
git commit -m "fix(tier3): retry malformed structured judge responses"
```

---

### Task 3: Independent review and end-to-end verification

**Files:**
- Review: `origin/main..HEAD`
- Live fixture: temporary one-case copy of `src/skillevaluator/tier3/reference_skills/calculator`

- [ ] **Step 1: Complete two-stage and whole-change review**

Dispatch spec compliance first, then code quality. Fix and re-review every Critical or Important issue. After task reviews pass, dispatch a fresh whole-change reviewer focused on retry count, schema-invalid paths, max-token overrides, provenance, redaction, shared/template drift, fail-closed artifacts, collector/report correctness, and regressions.

- [ ] **Step 2: Run full automated verification**

```bash
uv run python -m pytest -q -n auto
uv run ruff check src tests
uv build
```

Record exact pass/skip counts and build artifacts. Any failure must be diagnosed before proceeding.

- [ ] **Step 3: Run controlled live recovery with NVIDIA Build**

Load only `NVIDIA_API_KEY` from `/Users/christopherk/Work/.env` without sourcing or printing the file. Set:

```bash
export NVIDIA_API_KEY="$(sed -n 's/^NVIDIA_API_KEY=//p' /Users/christopherk/Work/.env | tail -n 1)"
export SKILL_EVAL_LLM_PROVIDER=nv_build
export SKILL_EVAL_LLM_MODEL=nvidia/nemotron-3-nano-30b-a3b
```

The explicit catalog model is required because the standalone verifier's fallback model is the OpenAI default rather than the provider-specific NVIDIA Build default.

For shared accuracy and goal, wrap the real `call_public_llm` so attempt one returns the judge-specific schema-invalid payload and attempt two delegates to the original real NVIDIA Build call.

Load a fresh standalone template module for each template judge. For template accuracy, wrap its real `call_public_llm` so attempt one returns schema-invalid accuracy JSON and attempt two delegates to the original real NVIDIA Build call. For template custom goal, wrap `_call_public_llm_with_provenance` the same way: attempt one returns invalid `achieved` with `{"provider": "controlled", "model": "invalid-first"}`, and attempt two delegates to the original real provider call. Assert:

- exactly two calls per judge;
- both attempts receive `max_tokens == 4096`;
- attempt two returns a finite score without `status: error`;
- template accuracy recovers through its standalone pair-returning transport;
- the template goal result's provider/model are the real second-attempt provenance;
- template goal's second-attempt model is `nvidia/nemotron-3-nano-30b-a3b`.

Then call real behavior-check normally and assert a finite score. This combines a controlled triggering first attempt with a real provider response on the recovery path; it does not claim that NVIDIA Build naturally emitted malformed output.

- [ ] **Step 4: Run paired Docker Tier 3 E2E**

```bash
ISSUE70_E2E_DIR="$(mktemp -d /tmp/skillevaluator-issue70-e2e.XXXXXX)"
cp -R src/skillevaluator/tier3/reference_skills/calculator "$ISSUE70_E2E_DIR/calculator"
uv run python -c 'import json, pathlib, sys; path = pathlib.Path(sys.argv[1]); entries = json.loads(path.read_text()); path.write_text(json.dumps([entries[0]], indent=2) + "\n")' \
  "$ISSUE70_E2E_DIR/calculator/evals/evals.json"
uv run skillevaluator tier3 evaluate "$ISSUE70_E2E_DIR/calculator" \
  --agents opencode \
  --env-mode docker \
  --n-attempts 1 \
  --harbor-keep-jobs
```

Keep the baseline arm enabled. Inspect `result.json`, both condition summaries, retained rich reward files, and rendered report output. Require full scored-attempt coverage and non-empty aggregates for both arms. If practical, repeat with `--agents claude-code`; otherwise report the exact untested boundary.

- [ ] **Step 5: Verify final branch state**

```bash
git status --short --branch
git log --oneline --decorate origin/main..HEAD
git diff --check origin/main..HEAD
```

Expected: clean branch, intentional commits only, and no whitespace errors.
