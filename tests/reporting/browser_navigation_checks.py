# SPDX-License-Identifier: Apache-2.0
"""Convenience entry point for the collected offline Chromium report suite."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest


def run(reports: Path | None, output: Path, browser_path: str | None = None) -> int:
    """Run the same checks as CI, retaining screenshots and JUnit results."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    values = {
        "SKILLEVALUATOR_BROWSER_TESTS": "1",
        "SKILLEVALUATOR_BROWSER_SCREENSHOTS": str(output / "screenshots"),
    }
    if browser_path:
        values["SKILLEVALUATOR_TEST_BROWSER"] = browser_path
    if reports is not None:
        values["SKILLEVALUATOR_BROWSER_REPORTS"] = str(reports.resolve())
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        return int(
            pytest.main(
                [
                    "-q",
                    str(Path(__file__).with_name("test_report_browser.py")),
                    "--basetemp=" + str(output / "pytest"),
                    "--junitxml=" + str(output / "browser-results.xml"),
                ]
            )
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run offline HTML report browser checks.")
    parser.add_argument("output", type=Path, help="directory for screenshots and test results")
    parser.add_argument("--reports", type=Path, help="optional existing full.html/stress.html/absent.html fixtures")
    parser.add_argument("--browser", default=os.environ.get("SKILLEVALUATOR_TEST_BROWSER"), help="Chromium executable")
    args = parser.parse_args()
    raise SystemExit(run(args.reports, args.output, args.browser))
