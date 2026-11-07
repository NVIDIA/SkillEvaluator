#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the public starter through the installed CLI and check retained output."""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from importlib.resources import as_file, files
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "starter"
REGRESSED_SKILL = """---
name: text-analyzer
description: Text statistics.
---

# Text Analyzer

Run scripts/analyze_text.py.
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(command: list[str], log: Path, expected_exit: int) -> None:
    print(f"\n$ {shlex.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    print(result.stdout + result.stderr, end="", flush=True)
    require(result.returncode == expected_exit, f"Expected exit {expected_exit}, got {result.returncode}; see {log}")


def check_quality(report: dict, expected: dict) -> None:
    require(report["overall_status"] == expected["status"], "Unexpected quality status")
    require(report["overall_passed"] == (expected["exit_code"] == 0), "Unexpected quality verdict")
    quality = report["quality_summary"][0]
    require(quality["overall_score"] == expected["score"], "Quality score changed; review findings before updating expectations")
    require(quality["grade"] == expected["grade"], "Unexpected quality grade")
    require(report["publication_status"] == expected["publication"], "Unexpected publication status")
    require(not report["publication"]["eligible"] and bool(report["publication"]["reasons"]), "Missing publication explanation")


def demo(output: Path) -> dict:
    expected = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    skill = output / "text-analyzer"
    reference = files("skillevaluator.tier3.reference_skills").joinpath("text-analyzer")
    with as_file(reference) as source:
        skill.mkdir()
        shutil.copy2(source / "SKILL.md", skill / "SKILL.md")
        shutil.copytree(source / "scripts", skill / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    (skill / "evals" / "files").mkdir(parents=True)
    shutil.copy2(FIXTURES / "evals.json", skill / "evals" / "evals.json")
    shutil.copy2(FIXTURES / "input.txt", skill / "evals" / "files" / "input.txt")

    run(
        [sys.executable, str(skill / "scripts" / "analyze_text.py"), "--file", str(skill / "evals" / "files" / "input.txt"), "--json"],
        output / "analysis.json",
        0,
    )
    actual = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    require(actual == expected["analysis"], "Text-analysis output differs from the reviewed fixture")

    regressed = output / "regressed-text-analyzer"
    shutil.copytree(skill, regressed)
    (regressed / "SKILL.md").write_text(REGRESSED_SKILL, encoding="utf-8")
    for label, target in (("quality", skill), ("regressed", regressed)):
        run(
            [sys.executable, "-m", "skillevaluator.cli", "quality-check", str(target), "--min-score", "80", "-r", "cli", "-r", "json", "-o", str(output / label)],
            output / f"{label}.txt",
            expected[label]["exit_code"],
        )
        report = json.loads((output / label / "skillevaluator-quality.json").read_text(encoding="utf-8"))
        check_quality(report, expected[label])

    receipt = {
        "generated_at": datetime.now(UTC).isoformat(),
        "skillevaluator_version": version("skillevaluator"),
        "python_version": sys.version.split()[0],
        "checks": ["text output matches fixture", "quality passes at 80", "seeded documentation regression fails at 80", "publication remains incomplete"],
        "live_agent_evaluation": "not run",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("\nPASS: starter output and quality regression checked; publication INCOMPLETE (no live agent evaluation).")
    print(f"Reports and editable starter: {output}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/starter"), help="New directory; existing paths are never overwritten")
    args = parser.parse_args()
    try:
        demo(args.output.resolve())
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f"Demo failed: {exc}. Keep the output for diagnosis and retry with a new --output directory.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
