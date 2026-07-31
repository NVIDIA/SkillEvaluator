# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reporting module for SkillEvaluator validation results.

This module provides reporters that render ValidationResult objects
in various formats for different use cases:

- CLIReporter: Terminal output with Rich formatting
- JSONReporter: Machine-readable JSON for CI/CD pipelines
- HTMLReporter: Standalone HTML reports for archiving/sharing
- MarkdownReporter: Markdown for PR comments and documentation

All reporters consume the same ValidationResult data structure from
skillevaluator.models, ensuring consistent information across all output formats.

Example usage:
    from skillevaluator.models import ValidationResult
    from skillevaluator.reporting import CLIReporter, JSONReporter

    result = ValidationResult(validator_name="SCHEMA")
    # ... populate result ...

    # CLI output
    cli = CLIReporter()
    cli.print(result)

    # JSON output
    json_reporter = JSONReporter()
    json_output = json_reporter.render(result)
"""

from skillevaluator.reporting.base import ReporterBase
from skillevaluator.reporting.benchmark import BenchmarkReporter
from skillevaluator.reporting.cli import CLIReporter
from skillevaluator.reporting.html import HTMLReporter
from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.reporting.markdown import MarkdownReporter

__all__ = [
    "BenchmarkReporter",
    "CLIReporter",
    "HTMLReporter",
    "JSONReporter",
    "MarkdownReporter",
    "ReporterBase",
]
