#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and normalize user custom graders inside Harbor verifier containers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _env_path(name, default):
    return Path(os.environ.get(name, str(default)))


LOGS_DIR = _env_path("HARBOR_LOGS_DIR", "/logs")
VERIFIER_DIR = _env_path("HARBOR_VERIFIER_DIR", LOGS_DIR / "verifier")
TESTS_DIR = _env_path("HARBOR_TESTS_DIR", "/tests")

REWARD_JSON = _env_path("HARBOR_REWARD_JSON", VERIFIER_DIR / "reward.json")
REWARD_TXT = _env_path("HARBOR_REWARD_TXT", VERIFIER_DIR / "reward.txt")
SKILL_EVALUATOR_REWARD_JSON = _env_path(
    "HARBOR_SKILL_EVALUATOR_REWARD_JSON", VERIFIER_DIR / "skill_evaluator_reward.json"
)
CUSTOM_REWARD_JSON = _env_path("HARBOR_CUSTOM_REWARD_JSON", VERIFIER_DIR / "custom_reward.json")
GRADER = _env_path("HARBOR_GRADER", TESTS_DIR / "grader.py")
GRADER_SH = _env_path("HARBOR_GRADER_SH", TESTS_DIR / "grader.sh")

RESERVED = {
    "security",
    "skill_execution",
    "skill_efficiency",
    "accuracy",
    "goal_accuracy",
    "behavior_check",
    "details",
    "entry_id",
    "evaluation_errors",
    "evaluation_status",
    "has_skill",
    "metric_set",
    "metrics",
    "overall",
    "trajectory_source",
}
DEFAULT_METRICS = {
    "security",
    "skill_execution",
    "skill_efficiency",
    "accuracy",
    "goal_accuracy",
    "behavior_check",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Invalid or missing {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return data


def _numeric(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _score_from_reward(reward: dict[str, Any]) -> float | None:
    score = _numeric(reward.get("overall"))
    if score is not None:
        return max(0.0, min(1.0, score))
    return None


def _score_from_text(text: str) -> float | None:
    try:
        return max(0.0, min(1.0, float(text.strip())))
    except ValueError:
        return None


def _score_from_txt() -> float | None:
    try:
        text = REWARD_TXT.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return _score_from_text(text)


def _extract_custom_metrics(reward: dict[str, Any]) -> dict[str, float]:
    custom: dict[str, float] = {}
    for key in DEFAULT_METRICS:
        if key in reward:
            raise RuntimeError(f"Custom grader cannot overwrite reserved SkillEvaluator metric '{key}'")
    explicit = reward.get("custom_metrics")
    if isinstance(explicit, dict):
        source = explicit
    else:
        source = {key: value for key, value in reward.items() if key not in RESERVED and not str(key).startswith("_")}

    for key, value in source.items():
        if key in RESERVED:
            raise RuntimeError(f"Custom metric '{key}' collides with reserved SkillEvaluator metric names")
        if isinstance(value, dict):
            value = value.get("score")
        score = _numeric(value)
        if score is not None:
            custom[str(key)] = max(0.0, min(1.0, score))
    return custom


def _numeric_reward_payload(reward: dict[str, Any], *, overall: float | None = None) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in reward.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            payload[str(key)] = float(value)
    if overall is not None:
        payload["overall"] = float(overall)
    return payload


def _run_grader() -> None:
    if GRADER.exists():
        subprocess.run([sys.executable, str(GRADER)], check=True)
        return
    if GRADER_SH.exists():
        subprocess.run(["bash", str(GRADER_SH)], check=True)
        return
    raise RuntimeError("/tests/grader.py or /tests/grader.sh is required for custom grading modes")


def _run_default_plus_custom() -> None:
    if SKILL_EVALUATOR_REWARD_JSON.exists():
        skill_evaluator_reward = _load_json(SKILL_EVALUATOR_REWARD_JSON)
    elif REWARD_JSON.exists():
        skill_evaluator_reward = _load_json(REWARD_JSON)
        SKILL_EVALUATOR_REWARD_JSON.write_text(json.dumps(skill_evaluator_reward, indent=2), encoding="utf-8")
    else:
        raise RuntimeError(
            "SkillEvaluator skill_evaluator_reward.json or reward.json must exist before default_plus_custom merge"
        )
    skill_evaluator_reward_txt = REWARD_TXT.read_text(encoding="utf-8") if REWARD_TXT.exists() else None
    skill_evaluator_overall = (
        _score_from_text(skill_evaluator_reward_txt)
        if skill_evaluator_reward_txt is not None
        else _score_from_reward(skill_evaluator_reward)
    )

    _run_grader()
    custom_reward = _load_json(REWARD_JSON)
    CUSTOM_REWARD_JSON.write_text(json.dumps(custom_reward, indent=2), encoding="utf-8")

    skill_evaluator_reward["custom_metrics"] = _extract_custom_metrics(custom_reward)
    if skill_evaluator_overall is not None:
        skill_evaluator_reward["overall"] = skill_evaluator_overall
    custom_details = custom_reward.get("details")
    if isinstance(custom_details, dict):
        skill_evaluator_reward["custom_details"] = custom_details
    SKILL_EVALUATOR_REWARD_JSON.write_text(json.dumps(skill_evaluator_reward, indent=2), encoding="utf-8")
    harbor_reward = _numeric_reward_payload(skill_evaluator_reward, overall=skill_evaluator_overall)
    harbor_reward.update(skill_evaluator_reward["custom_metrics"])
    REWARD_JSON.write_text(json.dumps(harbor_reward, indent=2), encoding="utf-8")

    # Keep SkillEvaluator default overall/pass@k authoritative for default_plus_custom.
    if skill_evaluator_reward_txt is not None:
        REWARD_TXT.write_text(skill_evaluator_reward_txt, encoding="utf-8")


def _run_custom_only() -> None:
    _run_grader()
    reward = _load_json(REWARD_JSON)
    score = _score_from_reward(reward)
    if score is None:
        score = _score_from_txt()
    if score is None:
        raise RuntimeError("custom_only requires numeric `overall` in reward.json or numeric reward.txt")
    if not 0.0 <= score <= 1.0:
        raise RuntimeError("custom_only overall score must be between 0.0 and 1.0")
    reward["overall"] = score
    CUSTOM_REWARD_JSON.write_text(json.dumps(reward, indent=2), encoding="utf-8")
    harbor_reward = _numeric_reward_payload(reward, overall=score)
    harbor_reward.update(_extract_custom_metrics(reward))
    REWARD_JSON.write_text(json.dumps(harbor_reward, indent=2), encoding="utf-8")
    REWARD_TXT.write_text(str(score), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["default_plus_custom", "custom_only"], required=True)
    args = parser.parse_args()

    try:
        if args.mode == "default_plus_custom":
            _run_default_plus_custom()
        else:
            _run_custom_only()
    except Exception as exc:
        REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
        CUSTOM_REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
        CUSTOM_REWARD_JSON.write_text(
            json.dumps({"overall": 0.0, "error": str(exc)}, indent=2),
            encoding="utf-8",
        )
        REWARD_JSON.write_text(json.dumps({"overall": 0.0}, indent=2), encoding="utf-8")
        REWARD_TXT.write_text("0.0", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
