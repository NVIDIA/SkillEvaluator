#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Harbor custom metric -- averages SkillEvaluator skill eval scores across all tasks.

Harbor calls this with:
  python metric.py -i rewards.jsonl -o metrics.json

Input: JSONL where each line is one task's reward.json content.
Output: JSON with six averaged evaluator scores and their five-dimension overall.
"""

import argparse
import json
import math
from pathlib import Path

DEFAULT_METRICS = [
    "security",
    "skill_execution",
    "skill_efficiency",
    "accuracy",
    "goal_accuracy",
    "behavior_check",
]
LEGACY_METRICS = [
    "skill_execution",
    "skill_efficiency",
    "accuracy",
    "goal_accuracy",
    "behavior_check",
]
DEFAULT_SCORE_POLICY = "skill-evaluator-dimension-mean-v1"
LEGACY_SCORE_POLICY = "skill-evaluator-metric-mean-v1"


def _metrics_for_rewards(rewards: list[dict]) -> list[str]:
    declared = {reward.get("metric_set") for reward in rewards if reward.get("metric_set")}
    if len(declared) > 1:
        raise ValueError("Cannot aggregate rewards with different metric sets")
    if declared == {"skill-evaluator-default-v2"}:
        return DEFAULT_METRICS
    if declared == {"skill-evaluator-default-v1"}:
        return LEGACY_METRICS
    if declared:
        raise ValueError("Unsupported reward metric set")
    if any("security" in reward for reward in rewards):
        return DEFAULT_METRICS
    if any(any(m in reward for m in LEGACY_METRICS) for reward in rewards):
        return LEGACY_METRICS
    raise ValueError("No supported SkillEvaluator metrics in rewards")


def _score_policy_for_rewards(rewards: list[dict], metrics: list[str]) -> str:
    declared = [reward.get("score_policy") for reward in rewards]
    if any(policy is not None for policy in declared):
        if any(not isinstance(policy, str) or not policy.strip() for policy in declared):
            raise ValueError("Cannot aggregate rewards with missing score policies")
        policies = {policy.strip() for policy in declared}
        if len(policies) != 1:
            raise ValueError("Cannot aggregate rewards with different score policies")
        policy = policies.pop()
    else:
        policy = DEFAULT_SCORE_POLICY if metrics == DEFAULT_METRICS else LEGACY_SCORE_POLICY
    if policy not in {DEFAULT_SCORE_POLICY, LEGACY_SCORE_POLICY}:
        raise ValueError("Unsupported reward score policy")
    if policy == DEFAULT_SCORE_POLICY and metrics != DEFAULT_METRICS:
        raise ValueError("Current score policy requires the default v2 metric set")
    return policy


def _unit_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    return numeric if math.isfinite(numeric) and 0.0 <= numeric <= 1.0 else None


def main(input_path: Path, output_path: Path) -> None:
    rewards: list[dict] = []

    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        reward = json.loads(line)
        if reward is None:
            continue
        if not isinstance(reward, dict):
            raise ValueError("Expected one reward object per JSONL row")
        rewards.append(reward)

    if not rewards:
        raise ValueError("No rewards to aggregate")
    metrics = _metrics_for_rewards(rewards)
    score_policy = _score_policy_for_rewards(rewards, metrics)
    sums: dict[str, float] = dict.fromkeys(metrics, 0.0)
    counts: dict[str, int] = dict.fromkeys(metrics, 0)

    for reward in rewards:
        for metric in metrics:
            val = reward.get(metric)
            if val is None and isinstance(reward.get("metrics"), dict):
                m_data = reward["metrics"].get(metric)
                val = m_data.get("score") if isinstance(m_data, dict) else m_data
            score = _unit_score(val)
            if score is None:
                raise ValueError(f"Incomplete or invalid {metric} reward")
            sums[metric] += score
            counts[metric] += 1

    result: dict[str, float | str] = {}
    for metric in metrics:
        c = counts[metric]
        result[metric] = round(sums[metric] / c, 4)

    if score_policy == DEFAULT_SCORE_POLICY:
        dimensions = (
            result["security"],
            result["accuracy"],
            result["skill_execution"],
            round((result["goal_accuracy"] + result["behavior_check"]) / 2, 4),
            result["skill_efficiency"],
        )
        result["overall"] = round(sum(dimensions) / len(dimensions), 4)
    else:
        result["overall"] = round(sum(float(result[metric]) for metric in metrics) / len(metrics), 4)
    result["metric_set"] = "skill-evaluator-default-v2" if "security" in metrics else "skill-evaluator-default-v1"
    result["score_policy"] = score_policy

    output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=Path, required=True)
    parser.add_argument("-o", "--output-path", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_path, args.output_path)
