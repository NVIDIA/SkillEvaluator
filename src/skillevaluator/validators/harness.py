# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advisory Tier 1 check: how each agent harness loads the skill, from versioned profiles."""

from __future__ import annotations

from pathlib import Path

from skillevaluator.harness.overlay import evaluate_profile, facts_for, load_profiles
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.validators.base import ValidatorBase

HARNESS_METADATA_KEY = "harness_compatibility"
# Below the default gate (HIGH), so the check never fails a run unless a policy raises it.
_SEVERITY = {"not_loaded": Severity.MEDIUM, "degraded": Severity.LOW, "advisory": Severity.INFO}


class HarnessValidator(ValidatorBase):
    """Evaluate each skill against the selected harness profiles."""

    def __init__(
        self, harnesses: str | None = None, profiles_dir: Path | None = None, config: dict | None = None
    ) -> None:
        self.harnesses = harnesses or "all"
        self.profiles_dir = profiles_dir
        self.config = config

    @property
    def name(self) -> str:
        return "Harness Compatibility"

    @property
    def description(self) -> str:
        return "Advisory: whether each agent harness loads, degrades or drops the skill"

    def validate(self, skill_path: Path) -> ValidationResult:
        skill_path = skill_path.parent if skill_path.is_file() else skill_path
        profiles = load_profiles(self.harnesses, self.profiles_dir)
        skills = [skill_path] if self._is_skill_directory(skill_path) else self._find_all_skills(skill_path)
        result = ValidationResult(validator_name=self.name, validator_description=self.description)
        block: dict[str, list[dict]] = {}
        for skill, facts in facts_for(skills).items():
            rows = [evaluate_profile(profile, facts, self.config) for profile in profiles]
            block[str(skill)] = [row.to_dict() for row in rows]
            for row in rows:
                for finding in row.findings:
                    result.add_finding(
                        Finding(
                            category="HARNESS",
                            severity=_SEVERITY[finding["effect"]],
                            check_name=finding["id"],
                            message=f"{row.harness}: {finding['message']}",
                            suggestion=finding.get("fix"),
                            file_path=str(skill / "SKILL.md"),
                            metadata={
                                "harness": row.harness,
                                "effect": finding["effect"],
                                "verified_version": row.version,
                                "verified_date": row.verified_date,
                            },
                        )
                    )
        result.metadata[HARNESS_METADATA_KEY] = block
        return result
