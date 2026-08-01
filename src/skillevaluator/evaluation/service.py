# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process Tier 3 evaluation service.

The single boundary CLI and API share for live agent evaluation. It delegates to
the native in-process Harbor engine in :mod:`skillevaluator.tier3.commands`
without starting a separate CLI process. Heavy Tier 3 imports stay lazy so
importing this module on a base install does not pull Harbor/Kubernetes.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

# These lightweight imports intentionally remain available at runtime so API
# frameworks and callers can resolve the public method annotations.
from skillevaluator.evaluation.options import DatasetOptions, EvaluationOptions  # noqa: TC001
from skillevaluator.evaluation.results import DatasetGenerationResult  # noqa: TC001

if TYPE_CHECKING:
    from skillevaluator.tier3.harbor.progress import ProgressReporter


class EvaluationService:
    """Facade over the native Tier 3 engine used by both the CLI and the API."""

    def evaluate(
        self,
        options: EvaluationOptions,
        *,
        progress_reporter: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        """Run a live agent evaluation and return the engine results mapping."""
        from skillevaluator.tier3.commands import evaluate as _evaluate
        from skillevaluator.tier3.harbor.progress import NullProgressReporter

        reporter = progress_reporter or NullProgressReporter()
        return _evaluate(options.skill_path, progress_reporter=reporter, **options.engine_kwargs())

    @staticmethod
    def failure_reason(result: object) -> str | None:
        """Return why a direct evaluation result is not a completed run.

        The standalone command is a gating surface: an empty mapping, a
        legacy result with no execution status, or a result carrying errors
        must not be reported as a successful command merely because no Python
        exception escaped the engine.
        """
        if not isinstance(result, Mapping):
            return "Tier 3 evaluation returned a non-mapping result"
        if not result:
            return "Tier 3 evaluation returned an empty result"

        raw_errors = result.get("error") or result.get("execution_errors")
        if isinstance(raw_errors, str) and raw_errors.strip():
            return raw_errors.strip()
        if isinstance(raw_errors, list):
            errors = [str(error).strip() for error in raw_errors if str(error).strip()]
            if errors:
                return "; ".join(errors)
        elif raw_errors:
            return str(raw_errors)

        status = result.get("execution_status")
        if status != "succeeded":
            label = str(status or "unknown")
            return f"Tier 3 evaluation execution status is {label}"
        return None

    def create_dataset(self, options: DatasetOptions) -> DatasetGenerationResult:
        """Create or preview a synthetic dataset and return its outcome."""
        from skillevaluator.tier3.commands import create_dataset as _create_dataset

        return _create_dataset(options.skill_path, **options.engine_kwargs())

    def create_autopilot_dataset(self, skill_path: Path, *, use_llm: bool) -> Path:
        """Create one missing eval case for the CLI autopilot workflow."""
        from skillevaluator.tier3.generate_dataset import generate_one_case

        return generate_one_case(skill_path, use_llm=use_llm)

    def discover_latest_results(self, skill_path: Path, results_dir: Path | None = None) -> Path | None:
        """Return the latest results directory for a skill, or None if absent.

        Handles timestamped result directories and the ``latest`` symlink via the
        shared results-location resolver, so API report discovery does not depend
        on a fixed report filename.
        """
        from skillevaluator.tier3.results_location import resolve_latest_results

        latest = resolve_latest_results(skill_path, results_dir)
        if not latest.exists():
            return None
        return latest.resolve() if latest.is_symlink() else latest

    def discover_latest_report(self, skill_path: Path, results_dir: Path | None = None) -> Path | None:
        """Return the latest HTML report path for a skill, or None if absent.

        Looks for ``report.html`` in the latest results directory, then falls
        back to a glob for any timestamped ``report*.html``.
        """
        latest = self.discover_latest_results(skill_path, results_dir)
        if latest is None:
            return None
        report = latest / "report.html"
        if report.exists():
            return report
        candidates = sorted(latest.glob("report*.html"))
        return candidates[-1] if candidates else None
