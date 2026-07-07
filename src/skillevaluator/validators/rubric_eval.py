# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-as-Judge Rubric Evaluator for skill quality.

Ported from SkillEvaluator QualitativeSkillEvaluator. Sends the full SKILL.md
content to an LLM judge that scores it against 9 qualitative criteria
covering description clarity, instruction quality, trigger simulation,
workflow completeness, and error handling.

Uses the shared public-provider LLM client. Synchronous, consistent with
other SkillEvaluator validators.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Any

from skillevaluator.constants import RUBRIC_CRITERIA, RUBRIC_MAX_TOKENS, RUBRIC_MIN_SCORE
from skillevaluator.inference.client import LLMClient
from skillevaluator.logging_config import get_logger
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.validators.base import ValidatorBase

logger = get_logger(__name__)

_IMPORTANCE_WEIGHTS = {"high": 3, "medium": 2, "low": 1}
_CRITERION_PASS_SCORE = 7
_MAX_SKILL_CONTENT_BYTES = 1024 * 1024
_MAX_SUPPLEMENTARY_FILE_BYTES = 3000
_MAX_SUPPLEMENTARY_TOTAL_BYTES = 30000


class _UnsafeRubricInputError(ValueError):
    """Raised when rubric input is unsafe or cannot be read within its limit."""


def _read_bounded_rubric_file(
    path: Path,
    *,
    skill_root: Path,
    max_bytes: int,
    truncate: bool,
) -> tuple[str, int]:
    """Read a contained regular file without following its final symlink."""
    try:
        resolved_root = skill_root.resolve(strict=True)
        if path.is_symlink():
            raise _UnsafeRubricInputError(f"symlinked rubric input is not allowed: {path}")
        path_mode = path.lstat().st_mode
        resolved_path = path.resolve(strict=True)
    except _UnsafeRubricInputError:
        raise
    except OSError as exc:
        raise _UnsafeRubricInputError(f"could not inspect rubric input {path}: {exc}") from exc

    if not stat.S_ISREG(path_mode):
        raise _UnsafeRubricInputError(f"rubric input must be a regular file: {path}")
    if not resolved_path.is_relative_to(resolved_root):
        raise _UnsafeRubricInputError(f"rubric input resolves outside the skill root: {path}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise _UnsafeRubricInputError(f"rubric input must be a regular file: {path}")
            payload = handle.read(max_bytes + 1)
    except _UnsafeRubricInputError:
        raise
    except OSError as exc:
        raise _UnsafeRubricInputError(f"could not read rubric input {path}: {exc}") from exc

    was_truncated = len(payload) > max_bytes
    if was_truncated and not truncate:
        raise _UnsafeRubricInputError(f"rubric input exceeds the maximum size of {max_bytes} bytes: {path}")
    bounded = payload[:max_bytes]
    try:
        text = bounded.decode("utf-8")
    except UnicodeDecodeError as exc:
        if not (was_truncated and exc.end == len(bounded)):
            raise _UnsafeRubricInputError(f"rubric input is not valid UTF-8: {path}") from exc
        text = bounded[: exc.start].decode("utf-8")
    if was_truncated:
        text += "\n... [truncated]"
    return text, len(bounded)


def _normalize_rubric_checks(checks: Any) -> tuple[list[dict], str | None]:
    """Validate criterion evidence and return it in rubric-schema order.

    The judge is authoritative only for each criterion's bounded score, pass
    flag, and notes. Criterion identity/order and description come from the
    local rubric schema so duplicate, missing, or invented criteria cannot
    influence the aggregate.
    """
    if not isinstance(checks, list):
        return [], "LLM judge returned invalid rubric evidence: checks must be a list"
    if not checks:
        return (
            [],
            f"LLM judge returned incomplete rubric evidence: expected {len(RUBRIC_CRITERIA)} criteria, received 0",
        )

    returned_ids: list[str] = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            return [], f"LLM judge returned invalid rubric evidence: criterion at index {index} must be an object"
        criterion_id = check.get("id")
        if not isinstance(criterion_id, str) or not criterion_id:
            return [], f"LLM judge returned invalid rubric evidence: criterion at index {index} must have a string id"
        returned_ids.append(criterion_id)

    expected_ids = [criterion["id"] for criterion in RUBRIC_CRITERIA]
    expected_id_set = set(expected_ids)
    id_counts = Counter(returned_ids)
    duplicates = sorted(criterion_id for criterion_id, count in id_counts.items() if count > 1)
    missing = [criterion_id for criterion_id in expected_ids if criterion_id not in id_counts]
    unexpected = sorted(criterion_id for criterion_id in id_counts if criterion_id not in expected_id_set)
    identity_errors: list[str] = []
    if missing:
        identity_errors.append(f"missing criterion ids: {', '.join(missing)}")
    if duplicates:
        identity_errors.append(f"duplicate criterion ids: {', '.join(duplicates)}")
    if unexpected:
        identity_errors.append(f"unexpected criterion ids: {', '.join(unexpected)}")
    if identity_errors:
        return [], f"LLM judge returned invalid rubric evidence: {'; '.join(identity_errors)}"

    checks_by_id = {check["id"]: check for check in checks}
    normalized: list[dict] = []
    for criterion in RUBRIC_CRITERIA:
        criterion_id = criterion["id"]
        check = checks_by_id[criterion_id]
        score = check.get("score")
        if (
            not isinstance(score, int | float)
            or isinstance(score, bool)
            or (isinstance(score, float) and not math.isfinite(score))
            or not 0 <= score <= 10
        ):
            return (
                [],
                f"LLM judge returned invalid rubric evidence: criterion {criterion_id} score must be a finite number between 0 and 10",
            )
        judge_pass = check.get("pass")
        if not isinstance(judge_pass, bool):
            return (
                [],
                f"LLM judge returned invalid rubric evidence: criterion {criterion_id} pass must be a boolean",
            )
        notes = check.get("notes", "")
        normalized.append(
            {
                "id": criterion_id,
                "criterion": criterion["criterion"],
                "importance": criterion["importance"],
                "pass": score >= _CRITERION_PASS_SCORE,
                "judge_pass": judge_pass,
                "score": score,
                "notes": notes if isinstance(notes, str) else str(notes),
            }
        )

    return normalized, None


def _aggregate_rubric_score(checks: list[dict]) -> float:
    """Return a local 0-100 score using schema importance weights.

    High, medium, and low importance criteria carry weights 3, 2, and 1.
    The result is rounded to one decimal only after the weighted mean is
    scaled from the criterion 0-10 range to the public 0-100 range.
    """
    weighted_score = sum(check["score"] * _IMPORTANCE_WEIGHTS[check["importance"]] for check in checks)
    total_weight = sum(_IMPORTANCE_WEIGHTS[check["importance"]] for check in checks)
    return round(weighted_score / total_weight * 10, 1)


class _UnavailableRubricReport(dict):
    """Internal marker distinguishing transport fallback from model evidence."""


class RubricJudge(LLMClient):
    """LLM judge that scores a skill against qualitative criteria."""

    default_max_tokens: int | None = RUBRIC_MAX_TOKENS

    _SYSTEM_PROMPT = (
        "You are evaluating an AI Agent Skill documentation file (SKILL.md).\n"
        "Score each criterion 0-10 (0=fails completely, 10=exceeds expectations).\n\n"
        "IMPORTANT: Respond with ONLY a JSON object. No markdown, no explanation, "
        "no preamble. Keep 'notes' values concise (1-2 sentences max). "
        "Output nothing except valid JSON.\n\n"
        "Response format:\n"
        "{\n"
        '  "overall_pass": <boolean>,\n'
        '  "score": <integer 0-100>,\n'
        '  "summary": "<1-2 sentence assessment>",\n'
        '  "checks": [\n'
        "    {\n"
        '      "id": "<criterion id>",\n'
        '      "criterion": "<description>",\n'
        f'      "pass": <boolean; true exactly when score >= {_CRITERION_PASS_SCORE}>,\n'
        '      "score": <integer 0-10>,\n'
        '      "notes": "<1-2 sentence observation>"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    def get_system_prompt(self) -> str:
        return self._SYSTEM_PROMPT

    def create_user_prompt(self, **kwargs: Any) -> str:
        skill_name: str = kwargs["skill_name"]
        skill_content: str = kwargs["skill_content"]
        criteria: list[dict] = kwargs.get("criteria", RUBRIC_CRITERIA)
        supplementary: dict | None = kwargs.get("supplementary_content")

        criteria_text = "\n".join(
            f"  {i + 1}. [{c['id']}] {c['criterion']} (Importance: {c['importance']})" for i, c in enumerate(criteria)
        )

        extra = ""
        if supplementary:
            scripts = supplementary.get("scripts", {})
            references = supplementary.get("references", {})
            if scripts or references:
                extra += "Supplementary Skill Content:\n"
                if scripts:
                    extra += "\nScripts (from scripts/ directory):\n"
                    for name, content in scripts.items():
                        extra += f"--- {name} ---\n{content}\n\n"
                if references:
                    extra += "\nReferences (from references/ directory):\n"
                    for name, content in references.items():
                        extra += f"--- {name} ---\n{content}\n\n"

        return (
            f"Skill Name: {skill_name}\n\n"
            f"SKILL.md Content:\n---\n{skill_content}\n---\n\n"
            f"{extra}"
            f"Evaluation Criteria:\n{criteria_text}\n\n"
            f"For each criterion, evaluate whether the skill meets the requirement.\n"
            f"Score each criterion 0-10 (0=fails completely, 10=exceeds expectations).\n"
            f"Set each criterion pass flag to true exactly when its score is at least {_CRITERION_PASS_SCORE}.\n"
        )

    def parse_response(self, response_text: str, **_kwargs: Any) -> dict:
        return _extract_json(response_text)

    def get_fallback_response(self, **_kwargs: Any) -> dict:
        return _UnavailableRubricReport(
            {
                "overall_pass": False,
                "score": 0,
                "summary": "LLM judge unavailable — rubric evaluation skipped",
                "checks": [],
            }
        )


def _extract_json(content: str) -> dict:
    """Extract JSON from LLM response, handling code fences and truncation."""
    content = content.strip().lstrip("\ufeff")
    if not content:
        msg = "Empty response"
        raise json.JSONDecodeError(msg, content, 0)

    json_match = re.search(r"```(?:json)?\s*(\{.+\})\s*```", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(content[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    msg = f"Could not extract valid JSON from LLM response ({len(content)} chars)"
    raise json.JSONDecodeError(msg, content[:200], 0)


def _collect_supplementary_content(
    skill_path: Path,
    max_bytes_per_file: int = _MAX_SUPPLEMENTARY_FILE_BYTES,
    max_total_bytes: int = _MAX_SUPPLEMENTARY_TOTAL_BYTES,
) -> dict[str, dict[str, str]]:
    """Collect contained regular supplementary files using bounded reads."""
    result: dict[str, dict[str, str]] = {"scripts": {}, "references": {}}
    total = 0
    try:
        resolved_root = skill_path.resolve(strict=True)
    except OSError as exc:
        raise _UnsafeRubricInputError(f"could not resolve skill root {skill_path}: {exc}") from exc

    for subdir, exts in [("scripts", ("*.py", "*.sh")), ("references", ("*.md",))]:
        d = skill_path / subdir
        if not os.path.lexists(d):
            continue
        try:
            if d.is_symlink():
                raise _UnsafeRubricInputError(f"symlinked supplementary directory is not allowed: {d}")
            if not stat.S_ISDIR(d.lstat().st_mode):
                raise _UnsafeRubricInputError(f"supplementary path must be a directory: {d}")
            if not d.resolve(strict=True).is_relative_to(resolved_root):
                raise _UnsafeRubricInputError(f"supplementary directory resolves outside the skill root: {d}")
        except _UnsafeRubricInputError:
            raise
        except OSError as exc:
            raise _UnsafeRubricInputError(f"could not inspect supplementary directory {d}: {exc}") from exc
        for ext in exts:
            for f in sorted(d.glob(ext)):
                if total >= max_total_bytes:
                    break
                read_limit = min(max_bytes_per_file, max_total_bytes - total)
                text, bytes_read = _read_bounded_rubric_file(
                    f,
                    skill_root=skill_path,
                    max_bytes=read_limit,
                    truncate=True,
                )
                result[subdir][f.name] = text
                total += bytes_read

    return result


class RubricEvalValidator(ValidatorBase):
    """LLM-as-Judge rubric evaluation for skill quality.

    Sends the full SKILL.md to an LLM that scores it against 9 qualitative
    criteria. Produces a 0-100 score with per-criterion breakdown.

    Requires a configured public LLM provider.
    """

    def __init__(self, min_score: int = RUBRIC_MIN_SCORE, model: str | None = None) -> None:
        self.min_score = min_score
        self._judge = RubricJudge(model=model) if model else RubricJudge()

    @property
    def name(self) -> str:
        return "LLM Rubric Evaluation"

    @property
    def description(self) -> str:
        return (
            "LLM-as-Judge qualitative evaluation across 9 criteria "
            "(description clarity, instruction quality, trigger simulation, etc.)"
        )

    def validate(self, skill_path: Path) -> ValidationResult:
        if self._is_skill_directory(skill_path):
            return self._validate_single_skill(skill_path)
        return self._validate_folder(skill_path)

    def _validate_folder(self, root: Path) -> ValidationResult:
        skill_dirs = self._find_all_skills(root)
        if not skill_dirs:
            result = ValidationResult(
                validator_name="RUBRIC_EVAL",
                validator_description=self.description,
            )
            result.add_finding(
                Finding(
                    category="RUBRIC_EVAL",
                    severity=Severity.HIGH,
                    check_name="skill_discovery",
                    message="No skills found in target directory",
                    file_path=str(root),
                )
            )
            result.metadata["rubric_eval"] = {
                "execution_status": "failed",
                "overall_score": None,
                "overall_pass": False,
                "summary": "No skills found in target directory",
                "checks": [],
            }
            return result

        result = ValidationResult(
            validator_name="RUBRIC_EVAL",
            validator_description=self.description,
        )
        all_rubric: list[dict] = []
        for skill_dir in skill_dirs:
            sub = self._validate_single_skill(skill_dir)
            result.merge_with_prefix(sub, skill_dir.name)
            if sub.metadata.get("rubric_eval", {}).get("execution_status") == "succeeded":
                all_rubric.append(sub.metadata["rubric_eval"])

        if len(all_rubric) == len(skill_dirs):
            avg_score = sum(r["overall_score"] for r in all_rubric) / len(all_rubric)
            all_pass = all(r["overall_pass"] for r in all_rubric)
            all_checks: list[dict] = [check for rubric in all_rubric for check in rubric.get("checks", [])]

            result.metadata["rubric_eval"] = {
                "execution_status": "succeeded",
                "overall_score": round(avg_score, 1),
                "overall_pass": all_pass and result.passed,
                "summary": f"Average across {len(all_rubric)} skills",
                "skill_count": len(all_rubric),
                "checks": all_checks,
            }
        else:
            failed_count = len(skill_dirs) - len(all_rubric)
            result.metadata["rubric_eval"] = {
                "execution_status": "failed",
                "overall_score": None,
                "overall_pass": False,
                "summary": f"Rubric judging failed for {failed_count} of {len(skill_dirs)} skills",
                "skill_count": len(skill_dirs),
                "successful_skill_count": len(all_rubric),
                "checks": [],
            }
        result.metadata["rubric_eval_all"] = all_rubric

        return result

    @staticmethod
    def _record_judge_failure(
        result: ValidationResult,
        *,
        report: Any,
        skill_path: Path,
        manifest: Path,
        message: str,
        check_name: str,
        suggestion: str,
    ) -> ValidationResult:
        judge_report = report if isinstance(report, dict) else {}
        result.metadata["rubric_eval"] = {
            "execution_status": "failed",
            "overall_score": None,
            "overall_pass": False,
            "summary": message,
            "judge_summary": judge_report.get("summary"),
            "judge_score": judge_report.get("score"),
            "judge_overall_pass": judge_report.get("overall_pass"),
            "skill_name": skill_path.name,
            "checks": [],
        }
        result.add_finding(
            Finding(
                category="RUBRIC_EVAL",
                severity=Severity.HIGH,
                check_name=check_name,
                message=message,
                file_path=str(manifest),
                suggestion=suggestion,
            )
        )
        return result

    def _validate_single_skill(self, skill_path: Path) -> ValidationResult:
        result = ValidationResult(
            validator_name="RUBRIC_EVAL",
            validator_description=self.description,
        )

        manifest = self._find_skill_manifest(skill_path)
        if not manifest:
            result.add_finding(
                Finding(
                    category="RUBRIC_EVAL",
                    severity=Severity.HIGH,
                    check_name="skill_manifest",
                    message="SKILL.md not found",
                    file_path=str(skill_path),
                )
            )
            result.metadata["rubric_eval"] = {
                "execution_status": "failed",
                "overall_score": None,
                "overall_pass": False,
                "summary": "SKILL.md not found",
                "skill_name": skill_path.name,
                "checks": [],
            }
            return result

        try:
            content, _ = _read_bounded_rubric_file(
                manifest,
                skill_root=skill_path,
                max_bytes=_MAX_SKILL_CONTENT_BYTES,
                truncate=False,
            )
            supplementary = _collect_supplementary_content(skill_path)
        except _UnsafeRubricInputError as exc:
            return self._record_judge_failure(
                result,
                report=None,
                skill_path=skill_path,
                manifest=manifest,
                message=f"Unsafe rubric input: {exc}",
                check_name="unsafe_skill_input",
                suggestion="Replace symlinks and special files with contained regular UTF-8 files",
            )

        from rich.console import Console
        from rich.status import Status

        with Status(
            f"[bold cyan]Evaluating {skill_path.name} with LLM judge...[/bold cyan]",
            console=Console(),
            spinner="dots",
        ):
            report = self._judge.process(
                skill_name=skill_path.name,
                skill_content=content,
                supplementary_content=supplementary,
            )

        if not isinstance(report, dict):
            return self._record_judge_failure(
                result,
                report=report,
                skill_path=skill_path,
                manifest=manifest,
                message="LLM judge returned invalid rubric evidence: response must be an object",
                check_name="llm_invalid_response",
                suggestion="Retry rubric evaluation; the judge response was malformed",
            )

        if isinstance(report, _UnavailableRubricReport):
            return self._record_judge_failure(
                result,
                report=report,
                skill_path=skill_path,
                manifest=manifest,
                message="LLM judge unavailable; rubric evaluation did not run",
                check_name="llm_unavailable",
                suggestion="Configure a public LLM provider to enable LLM rubric evaluation",
            )

        checks, validation_error = _normalize_rubric_checks(report.get("checks"))
        if validation_error:
            return self._record_judge_failure(
                result,
                report=report,
                skill_path=skill_path,
                manifest=manifest,
                message=validation_error,
                check_name="llm_invalid_response",
                suggestion="Retry rubric evaluation; the judge response did not contain valid complete criterion evidence",
            )

        overall_score = _aggregate_rubric_score(checks)
        overall_pass = overall_score >= self.min_score and all(check["pass"] for check in checks)

        for check in checks:
            check_score = check["score"]
            check_pass = check["pass"]
            severity = Severity.LOW if check_pass else Severity.MEDIUM

            result.add_finding(
                Finding(
                    category="RUBRIC_EVAL",
                    severity=severity,
                    check_name=f"rubric_{check.get('id', 'unknown')}",
                    message=f"[{check_score}/10] {check.get('criterion', '')}",
                    file_path=str(manifest),
                    suggestion=check.get("notes", ""),
                    metadata={
                        "score": check_score,
                        "pass": check_pass,
                        "judge_pass": check["judge_pass"],
                        "id": check.get("id", ""),
                    },
                )
            )

        failed_criteria = [check["id"] for check in checks if not check["pass"]]
        if not overall_pass:
            reasons: list[str] = []
            if overall_score < self.min_score:
                reasons.append(f"score {overall_score}/100 is below the required {self.min_score}/100")
            if failed_criteria:
                reasons.append(f"criteria below {_CRITERION_PASS_SCORE}/10: {', '.join(failed_criteria)}")
            result.add_finding(
                Finding(
                    category="RUBRIC_EVAL",
                    severity=Severity.HIGH,
                    check_name="rubric_overall",
                    message=f"Rubric evaluation failed: {'; '.join(reasons)}",
                    file_path=str(manifest),
                    suggestion="Address the failed rubric criteria and rerun the evaluation",
                    metadata={
                        "overall_score": overall_score,
                        "min_score": self.min_score,
                        "failed_criteria": failed_criteria,
                    },
                )
            )

        result.metadata["rubric_eval"] = {
            "execution_status": "succeeded",
            "overall_score": overall_score,
            "overall_pass": overall_pass,
            "summary": report.get("summary", ""),
            "judge_score": report.get("score"),
            "judge_overall_pass": report.get("overall_pass"),
            "skill_name": skill_path.name,
            "checks": checks,
            "aggregation": {
                "method": "importance_weighted_mean",
                "importance_weights": dict(_IMPORTANCE_WEIGHTS),
                "min_score": self.min_score,
                "criterion_pass_score": _CRITERION_PASS_SCORE,
                "requires_all_criteria_pass": True,
            },
        }

        result.add_success(
            check_name="rubric_eval",
            message=f"LLM Rubric Score: {overall_score}/100",
            overall_score=overall_score,
            overall_pass=overall_pass,
        )

        return result
