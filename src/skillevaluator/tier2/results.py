# SPDX-License-Identifier: Apache-2.0
"""Provider-independent result handling for public Tier 2 entrypoints."""

from pathlib import Path

from skillevaluator.models.result import ValidationResult
from skillevaluator.publication_evidence import stamp_publication_evidence
from skillevaluator.publication_identity import finalize_publication_target, publication_target_from_path


def guarded_results(title: str, target_path: Path, callback, *, check_id: str) -> list[ValidationResult]:
    initial_target = publication_target_from_path(target_path)
    try:
        results = callback()
    except Exception as exc:
        result = ValidationResult(validator_name=title, validator_description="Tier 2 check")
        result.mark_scan_incomplete(title)
        result.add_error(
            f"{title} could not complete because of an unexpected error ({type(exc).__name__}). "
            "Check the provider configuration and connectivity, then rerun Tier 2. "
            "If the problem persists, report this error type to the maintainers."
        )
        results = [result]
    for result in results:
        if result.metadata.get("security_failure"):
            result.metadata["advisory_tier2"] = False
        if not result.validator_name:
            result.validator_name = title
        if not result.validator_description:
            result.validator_description = f"Tier 2 check for {target_path}"
        stamp_publication_evidence([result], tier=2, check_id=check_id)
    finalize_publication_target(results, target_path, initial_target)
    return results
