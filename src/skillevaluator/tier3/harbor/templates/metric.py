#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Harbor custom metric -- averages SkillEvaluator skill eval scores across all tasks.

Harbor calls this with:
  python metric.py -i rewards.jsonl -o metrics.json

Input: JSONL where each line is one task's reward.json content.
Output: JSON with averaged 5-eval scores.
"""

import argparse
import json
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


def _metrics_for_rewards(rewards: list[dict]) -> list[str]:
    if any(isinstance(reward.get("security"), int | float) for reward in rewards):
        return DEFAULT_METRICS
    if any(any(isinstance(reward.get(m), int | float) for m in LEGACY_METRICS) for reward in rewards):
        return LEGACY_METRICS
    return DEFAULT_METRICS


def main(input_path: Path, output_path: Path) -> None:
    rewards: list[dict] = []

    for line in input_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        reward = json.loads(line)
        if reward is None:
            continue
        rewards.append(reward)

    metrics = _metrics_for_rewards(rewards)
    sums: dict[str, float] = dict.fromkeys(metrics, 0.0)
    counts: dict[str, int] = dict.fromkeys(metrics, 0)

    for reward in rewards:
        for metric in metrics:
            val = reward.get(metric)
            if val is None and isinstance(reward.get("metrics"), dict):
                m_data = reward["metrics"].get(metric)
                val = m_data.get("score") if isinstance(m_data, dict) else m_data
            if isinstance(val, (int, float)):
                sums[metric] += float(val)
                counts[metric] += 1

    result: dict[str, float] = {}
    for metric in metrics:
        c = counts[metric]
        result[metric] = round(sums[metric] / c, 4) if c > 0 else 0.0

    result["overall"] = round(sum(result.values()) / len(result), 4) if result else 0.0
    result["metric_set"] = "skill-evaluator-default-v2" if "security" in metrics else "skill-evaluator-default-v1"

    output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=Path, required=True)
    parser.add_argument("-o", "--output-path", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_path, args.output_path)
