# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

from skillevaluator.tier3.eval_core import atif_helpers


def _traj():
    return {"steps": [
        {"source": "user", "message": "Submit the evaluator job and save results."},
        {"source": "agent", "message": "running",
         "tool_calls": [{"tool_call_id": "t1", "function_name": "bash",
                          "arguments": {"command": "/app/.venv/bin/nemo evaluator submit --spec-file spec.yml"}}],
         "observation": {"results": [{"source_call_id": "t1", "content": "submitted job nemo-evaluator-abc"}]}},
        {"source": "agent", "message": "Done. Submitted the job."},
    ]}


def test_observed_and_not_observed_facts():
    facts = atif_helpers.build_verified_facts(
        _traj(),
        ["Run `nemo evaluator submit` for the spec",
         "Save results to /logs/agent/string-check-job-results.json"],
        "",
    )
    by_claim = {f["claim"]: f for f in facts}
    sub = by_claim["nemo evaluator submit"]
    assert sub["observed"] is True and sub["step_id"] == 1 and "evaluator submit" in sub["evidence"]
    missing = by_claim["/logs/agent/string-check-job-results.json"]
    assert missing["observed"] is False and missing["step_id"] is None


def test_facts_injected_at_top_of_all_bundles_and_verified_field():
    bundles = atif_helpers.build_metric_evidence_bundles(
        _traj(), "q", ground_truth="Results saved to /logs/agent/string-check-job-results.json",
        expected_behavior=["Run `nemo evaluator submit`"])
    for metric in ("accuracy", "goal_accuracy", "behavior_check"):
        b = bundles[metric]
        assert b["prompt_evidence"].startswith("VERIFIED FACTS (deterministic):"), metric
        assert "[OBSERVED step 1] `nemo evaluator submit`" in b["prompt_evidence"] or "[OBSERVED step 1] nemo evaluator submit" in b["prompt_evidence"], metric
        assert "[NOT OBSERVED] /logs/agent/string-check-job-results.json" in b["prompt_evidence"], metric
        assert isinstance(b["verified"], list) and len(b["verified"]) == 2


def test_no_tokens_means_no_facts_section():
    bundles = atif_helpers.build_metric_evidence_bundles(
        _traj(), "q", ground_truth="The agent should answer politely.", expected_behavior=["Be helpful."])
    for metric in ("accuracy", "goal_accuracy", "behavior_check"):
        assert "VERIFIED FACTS" not in bundles[metric]["prompt_evidence"]
        assert bundles[metric]["verified"] == []


def test_template_verified_facts_match_shared():
    template_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "skillevaluator"
        / "tier3"
        / "harbor"
        / "templates"
        / "eval.py"
    )
    spec = importlib.util.spec_from_file_location("harbor_eval_template_vf", template_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = (["Run `nemo evaluator submit`"], "Results in /logs/agent/out.json")
    shared = atif_helpers.build_verified_facts(_traj(), *args)
    templated = module.build_verified_facts(_traj(), *args)
    assert shared == templated
