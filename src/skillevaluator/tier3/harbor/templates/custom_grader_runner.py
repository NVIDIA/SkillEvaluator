#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and normalize user custom graders inside Harbor verifier containers."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
    "custom_details",
    "custom_metrics",
    "details",
    "entry_id",
    "error",
    "evaluation_errors",
    "evaluation_status",
    "has_skill",
    "metric_set",
    "metric_set_version",
    "metrics",
    "overall",
    "trajectory_detail",
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
MAX_CUSTOM_METRICS = 128
MAX_CUSTOM_METRIC_NAME_BYTES = 256
_SAFE_SENSITIVE_METRIC_PREFIXES = {"auth", "secret", "token"}
_SAFE_SENSITIVE_METRIC_SUFFIXES = {
    "accuracy",
    "compliance",
    "count",
    "coverage",
    "efficiency",
    "handling",
    "leakage",
    "precision",
    "quality",
    "rate",
    "ratio",
    "recall",
    "safety",
    "score",
    "usage",
}
_SENSITIVE_KEY_PARTS = {
    "auth",
    "authorization",
    "bearer",
    "credential",
    "credentials",
    "key",
    "password",
    "private",
    "secret",
    "token",
}
_PLURAL_SENSITIVE_KEY_PARTS = {
    "auths",
    "authorizations",
    "bearers",
    "credentials",
    "passwords",
    "secrets",
    "tokens",
}
_TOKEN_COUNT_KEYS = {
    "cached_tokens",
    "completion_tokens",
    "expected_max_tokens",
    "frontmatter_tokens",
    "input_tokens",
    "instructions_tokens",
    "n_input_tokens",
    "n_cache_tokens",
    "n_output_tokens",
    "output_tokens",
    "prompt_tokens",
    "reasoning_output_tokens",
    "last_token_usage",
    "max_completion_tokens",
    "max_output_tokens",
    "max_tokens",
    "recommended_max_tokens",
    "token_count",
    "tokens",
    "total_cached_tokens",
    "total_completion_tokens",
    "total_prompt_tokens",
    "total_tokens",
}
_EMBEDDED_CREDENTIAL_NAME_RE = re.compile(
    r"(?:sk-|nvapi-)[a-zA-Z0-9_-]{8,}"
    r"|crsr_[a-f0-9]{16,}"
    r"|(?:AKIA|ASIA)[A-Z0-9]{16}"
    r"|eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
    r"|sha256~[A-Za-z0-9._~-]+"
    r"|(?i:gh[pour]_[a-z0-9]{36})"
    r"|(?i:ghs_[a-z0-9.\-_]{36,})"
    r"|(?i:github_pat_[a-z0-9_]{20,})"
    r"|(?i:xox[baprs]-[a-z0-9-]{10,})"
    r"|AIza[A-Za-z0-9_-]{20,}"
    r"|(?i:glpat-[a-z0-9_-]{20,})"
)
_CREDENTIAL_URI_NAME_RE = re.compile(r"(?i)[a-z][a-z0-9+.-]{0,31}://[^\s/?#]*@")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Invalid or missing {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return data


def _numeric(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (OverflowError, ValueError):
        return None
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


def _normalized_metric_name_parts(name: str) -> tuple[str, ...]:
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", camel_split).strip("_").lower()
    return tuple(part for part in normalized.split("_") if part)


def _metric_name_is_publishable(name: object) -> bool:
    text = str(name)
    try:
        encoded = text.encode("utf-8")
    except UnicodeError:
        return False
    if not text or text != text.strip() or not text.isprintable() or len(encoded) > MAX_CUSTOM_METRIC_NAME_BYTES:
        return False
    return not _metric_name_contains_sensitive_data(text)


def _metric_name_contains_sensitive_data(text: str) -> bool:
    if _EMBEDDED_CREDENTIAL_NAME_RE.search(text) or _CREDENTIAL_URI_NAME_RE.search(text):
        return True
    parts = _normalized_metric_name_parts(text)
    if "_".join(parts) in _TOKEN_COUNT_KEYS:
        return False
    part_set = set(parts)
    compact = "".join(parts)
    sensitive = bool(part_set & (_SENSITIVE_KEY_PARTS | _PLURAL_SENSITIVE_KEY_PARTS))
    sensitive = sensitive or "apikey" in compact or "accesskey" in compact
    sensitive = sensitive or "privatekey" in compact or "sessiontoken" in compact
    explicitly_safe = (
        len(parts) == 2 and parts[0] in _SAFE_SENSITIVE_METRIC_PREFIXES and parts[1] in _SAFE_SENSITIVE_METRIC_SUFFIXES
    )
    return sensitive and not explicitly_safe


def _metric_name_shape_is_valid(name: object) -> bool:
    text = str(name)
    try:
        encoded = text.encode("utf-8")
    except UnicodeError:
        return False
    return bool(text) and text == text.strip() and text.isprintable() and len(encoded) <= MAX_CUSTOM_METRIC_NAME_BYTES


def _score_from_reward(reward: dict[str, Any]) -> float | None:
    return _numeric(reward.get("overall"))


def _score_from_text(text: str) -> float | None:
    try:
        return _numeric(float(text.strip()))
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
    if "custom_metrics" in reward and not isinstance(explicit, dict):
        raise RuntimeError("Custom metrics container must be a JSON object")

    sources: list[tuple[dict[Any, Any], bool]] = []
    if isinstance(explicit, dict):
        sources.append((explicit, True))
    metrics = reward.get("metrics")
    if isinstance(metrics, dict):
        sources.append((metrics, False))
    sources.append(
        (
            {key: value for key, value in reward.items() if str(key) not in RESERVED and not str(key).startswith("_")},
            False,
        )
    )

    for source, reject_reserved in sources:
        for key, value in source.items():
            name = str(key)
            if name in RESERVED:
                if reject_reserved:
                    raise RuntimeError(f"Custom metric '{key}' collides with reserved SkillEvaluator metric names")
                continue
            if isinstance(value, dict):
                value = value.get("score")
            score = _numeric(value)
            if score is None:
                continue
            if _metric_name_contains_sensitive_data(name):
                continue
            if not _metric_name_shape_is_valid(name):
                raise RuntimeError("Custom metric name exceeds the bounded publication contract")
            if name not in custom and len(custom) >= MAX_CUSTOM_METRICS:
                raise RuntimeError("Custom metric count exceeds the per reward publication limit")
            custom[name] = max(0.0, min(1.0, score))
    return custom


def _sanitized_custom_reward(reward: dict[str, Any], custom_metrics: dict[str, float]) -> dict[str, Any]:
    """Keep custom evidence only for validated metric names."""
    sanitized = dict(reward)
    explicit = reward.get("custom_metrics")
    if isinstance(explicit, dict):
        sanitized["custom_metrics"] = {
            str(raw_name): custom_metrics[str(raw_name)] for raw_name in explicit if str(raw_name) in custom_metrics
        }
    metrics = reward.get("metrics")
    if isinstance(metrics, dict):
        sanitized["metrics"] = {
            str(raw_name): value for raw_name, value in metrics.items() if str(raw_name) in custom_metrics
        }
    for raw_name, value in list(reward.items()):
        name = str(raw_name)
        if name in RESERVED or name.startswith("_"):
            continue
        if not _metric_name_is_publishable(name):
            sanitized.pop(raw_name, None)
            continue
        candidate = value.get("score") if isinstance(value, dict) else value
        if _numeric(candidate) is None:
            continue
        if name not in custom_metrics:
            sanitized.pop(raw_name, None)
    for field in ("details", "custom_details"):
        details = reward.get(field)
        if not isinstance(details, dict):
            continue
        safe_details = {
            str(raw_name): detail for raw_name, detail in details.items() if str(raw_name) in custom_metrics
        }
        if safe_details:
            sanitized[field] = safe_details
        else:
            sanitized.pop(field, None)
    return sanitized


def _numeric_reward_payload(reward: dict[str, Any], *, overall: float | None = None) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in reward.items():
        numeric = _numeric(value)
        if numeric is not None:
            payload[str(key)] = numeric
    numeric_overall = _numeric(overall)
    if numeric_overall is not None:
        payload["overall"] = numeric_overall
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
    custom_metrics = _extract_custom_metrics(custom_reward)
    safe_custom_reward = _sanitized_custom_reward(custom_reward, custom_metrics)
    CUSTOM_REWARD_JSON.write_text(json.dumps(safe_custom_reward, indent=2), encoding="utf-8")

    skill_evaluator_reward["custom_metrics"] = custom_metrics
    if skill_evaluator_overall is not None:
        skill_evaluator_reward["overall"] = skill_evaluator_overall
    custom_details = safe_custom_reward.get("details")
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
        raise RuntimeError("custom_only requires numeric `overall` between 0.0 and 1.0 in reward.json or reward.txt")
    reward["overall"] = score
    custom_metrics = _extract_custom_metrics(reward)
    safe_reward = _sanitized_custom_reward(reward, custom_metrics)
    CUSTOM_REWARD_JSON.write_text(json.dumps(safe_reward, indent=2), encoding="utf-8")
    harbor_reward = _numeric_reward_payload(safe_reward, overall=score)
    harbor_reward.update(custom_metrics)
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
