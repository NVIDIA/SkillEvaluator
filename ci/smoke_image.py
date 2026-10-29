# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201 -- standalone image verification CLI.
"""Exercise the real container, Git selection and retained reports without model credentials.

Requires Docker, Git and an already built/pulled image. Results and raw container
logs are retained under --output. Only this script's temporary Git repository
and its own named containers are created/removed; no remote repository is used.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

# Public synthetic text-only fixtures adapted verbatim from the personal demo.
# No provider calls, dependency downloads or consumer-repository checkout needed.
PUBLIC_SKILLS = {
    "glossary-builder": """---
name: glossary-builder
description: Builds a small glossary from supplied prose. Use when a reader needs definitions grounded in the text and ambiguous terms identified.
license: MIT
metadata:
  author: SkillEvaluator CI Demo Authors <ci-demo@example.com>
  version: 1.0.0
  tags:
    - demonstration
    - text-assistance
---

# Glossary Builder

## Purpose
Explain specialized terms using only definitions supported by supplied prose.

## Prerequisites
Provide readable source text and, optionally, the intended audience.
No external lookup, software dependencies, or credentials are required.

## Instructions
1. Read the source and identify terms that need explanation for the stated audience.
2. Capture explicit definitions and the sentence that supports each definition.
3. Expand abbreviations only when the expansion appears in the source.
4. Mark an undefined term as “Definition not supplied.”
5. Keep distinct meanings separate when the same term is used in different contexts.
6. Sort terms alphabetically without changing their original spelling.
7. Validate each definition against its supporting sentence.

## Output
Return a Markdown table with Term, Definition, and Source wording columns.
Place ambiguous or undefined terms in a short Follow-up questions list.
If there are no specialized terms, state that no glossary entries are needed.

## Examples
Input:
```text
A batch is a set of requests processed together. Each batch receives a tracking code.
```
Output:
```text
Term: Batch
Definition: A set of requests processed together.
Source wording: A batch is a set of requests processed together.
```
“Tracking code” remains undefined unless the source explains its meaning.

## Limitations
This skill is a reading aid rather than an authoritative dictionary.
It does not introduce external definitions or resolve domain disagreements by guessing.

## Troubleshooting
If the source contains conflicting definitions, include each meaning with its own wording.
If text is empty or invalid, identify the input problem and request a readable excerpt.
""",
    "evidence-table": """---
name: evidence-table
description: Organizes supplied claims and observations into an evidence table. Use when a review must distinguish support, uncertainty, and missing proof.
license: MIT
metadata:
  author: SkillEvaluator CI Demo Authors <ci-demo@example.com>
  version: 1.0.0
  tags:
    - demonstration
    - text-assistance
---

# Evidence Table

## Purpose
Make the relationship between a claim and its supplied evidence easy to inspect.

## Prerequisites
Provide the claims and the observations or source excerpts available for review.
No external search, accounts, credentials, or file modifications are required.

## Instructions
1. Separate each factual claim from recommendations and opinions.
2. Match the claim to the exact observation that supports it.
3. Classify support as Supported, Partially supported, Contradicted, or Unverified.
4. Use Supported only when the evidence covers the entire claim.
5. Record scope limits such as a single example or a simulated run.
6. State the smallest additional observation needed to close each proof gap.
7. Validate that no row converts an assumption into an observed fact.

## Output
Return a Markdown table with Claim, Evidence, Support, and Remaining gap columns.
Follow it with at most three questions needed to resolve important gaps.
If no evidence is supplied, label all factual claims Unverified.

## Examples
Input:
```text
Claim: Every export format works.
Observation: A CSV export opened successfully. No other format was checked.
```
Output:
```text
Claim: Every export format works.
Evidence: One CSV export opened successfully.
Support: Partially supported.
Remaining gap: Verify the other promised formats and relevant options.
```
The successful CSV observation does not establish behavior of other formats.

## Limitations
This skill organizes supplied material and does not independently authenticate sources.
Absence of evidence does not prove a claim false.

## Troubleshooting
If evidence conflicts, record both observations and explain the unresolved difference.
If a claim is too broad to validate, request a narrower, measurable statement.
""",
}


class SmokeError(RuntimeError):
    """The real runtime did not provide the required verification evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "-C", str(repository), *arguments],
        capture_output=True,
        check=True,
        timeout=60,
    )
    return result.stdout.decode("utf-8").strip()


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", "--all")
    _git(
        repository,
        "-c",
        "user.name=SkillEvaluator CI",
        "-c",
        "user.email=ci@example.com",
        "commit",
        "--quiet",
        "-m",
        message,
    )
    return _git(repository, "rev-parse", "HEAD")


def _load(path: Path) -> dict:
    _require(path.is_file() and not path.is_symlink(), f"Required report missing: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Report must be an object: {path.name}")
    return value


def verify_case(directory: Path, case: str, exit_code: int) -> dict:
    """Assert real report semantics; never infer correctness from exit alone."""
    summary = _load(directory / "summary.json")
    for name in ("summary.md", "details.md", "index.html", "pr-comment.md"):
        _require((directory / name).is_file(), f"Required report missing: {name}")
    markdown = (directory / "summary.md").read_text(encoding="utf-8")
    _require(0 < len(markdown.encode()) <= 20 * 1024, "Actions summary must be present and bounded")
    # CI Markdown deliberately escapes identifiers and may uppercase verdict labels.
    visible = html.unescape(re.sub(r"\\([\\`*_{}\[\]()#+.!|~-])", r"\1", markdown))
    rows = summary.get("skills")
    _require(isinstance(rows, list), "Summary lacks skill rows")
    _require(summary.get("total") == len(rows), "Total count contradicts skill rows")
    _require(
        summary.get("execution", {}).get("llm_enabled") is (case == "missing-llm-key"),
        "Recorded LLM scope contradicts the tested configuration",
    )
    expected_total = {"all-pass": 2, "invalid-name": 2, "changed-one": 1, "no-change": 0, "missing-llm-key": 0}[case]
    _require(len(rows) == expected_total, f"Unexpected selected skill count for {case}")
    if case == "no-change":
        _require(exit_code == 0, "No-change selection must finish successfully")
        _require(
            summary.get("overall_status") == "skipped" and summary.get("overall_passed") is False,
            "No-change selection must be neutral, not a scanner pass",
        )
        _require(
            all(summary.get(key) == 0 for key in ("passed", "failed", "incomplete", "total_validators")),
            "No-change selection must not claim executed validators",
        )
        selection = summary.get("selection", {})
        _require(
            selection.get("mode") == "changed" and selection.get("discovered") == 2 and selection.get("selected") == [],
            "No-change selection evidence is missing",
        )
        _require("no skills changed" in visible.casefold(), "No-change summary must explain that no skills changed")
    elif case == "missing-llm-key":
        _require(
            exit_code == 1 and summary.get("overall_status") == "error" and summary.get("overall_passed") is False,
            "Absent LLM credentials must fail explicitly",
        )
        _require(
            "LLM_API_KEY" in str(summary.get("error")) and "LLM_API_KEY" in visible,
            "Missing-key configuration diagnostic was not retained",
        )
        _require(summary.get("total_validators") == 0, "Missing-key rejection must precede scanning")
    else:
        failing = case == "invalid-name"
        _require(exit_code == int(failing), f"Unexpected process result for {case}")
        _require(
            summary.get("overall_status") == ("failed" if failing else "passed")
            and summary.get("overall_passed") is (not failing),
            "Process and summary verdicts disagree",
        )
        _require(
            summary.get("passed") == expected_total - int(failing)
            and summary.get("failed") == int(failing)
            and summary.get("incomplete") == 0,
            "Unexpected passing/failing/incomplete counts",
        )
        if case == "changed-one":
            selection = summary.get("selection", {})
            _require(
                selection.get("mode") == "changed"
                and selection.get("discovered") == 2
                and selection.get("selected") == ["skills/glossary-builder"],
                "Changed selection must evaluate exactly one of the two skills",
            )
        expected_paths = (
            {"skills/glossary-builder"}
            if case == "changed-one"
            else {
                "skills/glossary-builder",
                "skills/evidence-table",
            }
        )
        _require({row.get("path") for row in rows} == expected_paths, "Evaluated skill identities differ")
        for index, row in enumerate(rows, 1):
            _require(row.get("report_dir") == f"skill-{index:03d}", "Unexpected child report location")
            child = directory / row["report_dir"]
            data = _load(child / "skillevaluator-tier1.json")
            expected_pass = not (failing and row["path"] == "skills/glossary-builder")
            _require(
                row.get("passed") is expected_pass and data.get("overall_passed") is expected_pass,
                "Child verdict contradicts the intended fixture",
            )
            _require(
                row.get("exit_code") == int(not expected_pass)
                and row.get("overall_status") == ("passed" if expected_pass else "failed")
                and data.get("overall_status") == row["overall_status"],
                "Child process and report statuses disagree",
            )
            _require(
                not row.get("incomplete_scans") and not data.get("incomplete_scans"),
                "Required scanner execution was incomplete",
            )
            results = data.get("results", [])
            _require(isinstance(results, list) and len(results) >= 10, "Full Tier 1 validator evidence missing")
            _require(
                all(not result.get("incomplete_scans") and result.get("status") != "incomplete" for result in results),
                "A validator scan did not complete",
            )
            for name in (
                "skillevaluator-tier1.html",
                "skillevaluator-tier1.md",
                "skillevaluator-tier1.sarif.json",
                "stdout.log",
                "stderr.log",
            ):
                _require((child / name).is_file(), f"Child report/log missing: {name}")
            if not expected_pass:
                _require(
                    any(
                        finding.get("category") == "SCHEMA" and "INVALID NAME" in finding.get("message", "")
                        for result in results
                        for finding in result.get("findings", [])
                    ),
                    "Expected invalid-name schema finding was not recorded",
                )
    return {
        "case": case,
        "verified": True,
        "exit_code": exit_code,
        "overall_status": summary["overall_status"],
        "total": summary["total"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "incomplete": summary["incomplete"],
        "summary_bytes": len(markdown.encode()),
        "summary_json_sha256": hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest(),
    }


def _container(image: str, workspace: Path, directory: Path, case: str, base: str, head: str) -> int:
    directory.mkdir()
    reports = directory / "reports"
    reports.mkdir()
    name = "skillevaluator-smoke-" + uuid.uuid4().hex[:16]
    environment = {
        "SKILL_PATHS": "skills",
        "LLM_ENABLED": "true" if case == "missing-llm-key" else "false",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "",
        "REPORT_IMAGE": image,
        "SCAN_MODE": "changed" if case in {"changed-one", "no-change"} else "all",
        "DIFF_BASE_SHA": base,
        "DIFF_HEAD_SHA": head,
        "REPORT_COMMIT": head,
    }
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,noexec,size=1g",
        "--tmpfs",
        "/audit-tmp:rw,nosuid,noexec,mode=1777,size=1g",
        "--env",
        "TMPDIR=/audit-tmp",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--mount",
        f"type=bind,src={workspace},dst=/workspace,readonly",
        "--mount",
        f"type=bind,src={reports},dst=/reports",
    ]
    for key, value in environment.items():
        command.extend(["--env", f"{key}={value}"])
    command.append(image)
    try:
        with (
            (directory / "container.stdout.log").open("wb") as stdout,
            (directory / "container.stderr.log").open("wb") as stderr,
        ):
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False, timeout=900)
        return completed.returncode
    finally:
        # A killed Docker CLI may leave its container alive; remove only our own unique name.
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, check=False, timeout=30)


def smoke(image: str, output: Path) -> dict:
    _require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,511}", image)), "Invalid local image reference")
    _require(not output.is_symlink(), "Evidence output must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve(strict=True)
    _require(not any(output.iterdir()), "Evidence output must be empty to prevent stale results")
    _require("," not in str(output), "Evidence output cannot contain a comma in a Docker mount path")
    receipt = {"image": image, "all_verified": False, "cases": []}
    try:
        metadata = subprocess.run(["docker", "image", "inspect", image], capture_output=True, check=True, timeout=60)
        details = json.loads(metadata.stdout)[0]
        receipt["runtime"] = {
            "id": details.get("Id"),
            "digests": details.get("RepoDigests"),
            "os": details.get("Os"),
            "architecture": details.get("Architecture"),
            "source_revision": (details.get("Config", {}).get("Labels") or {}).get("org.opencontainers.image.revision"),
        }
        with tempfile.TemporaryDirectory(prefix="skillevaluator-smoke-") as temporary:
            workspace = Path(temporary).resolve()
            _git(workspace, "init", "--quiet", "--template=")
            for name, content in PUBLIC_SKILLS.items():
                path = workspace / "skills" / name / "SKILL.md"
                path.parent.mkdir(parents=True)
                path.write_text(content, encoding="utf-8")
            base = _commit(workspace, "Add public smoke fixtures")
            for case in ("all-pass", "invalid-name", "changed-one", "no-change", "missing-llm-key"):
                _git(workspace, "reset", "--hard", base)
                if case == "invalid-name":
                    path = workspace / "skills/glossary-builder/SKILL.md"
                    path.write_text(path.read_text().replace("name: glossary-builder", "name: INVALID NAME", 1))
                elif case == "changed-one":
                    path = workspace / "skills/glossary-builder/SKILL.md"
                    path.write_text(path.read_text() + "\nAdditional note: Preserve the supplied wording.\n")
                elif case == "no-change":
                    (workspace / "README.md").write_text("Public smoke fixture repository.\n")
                head = base if case in {"all-pass", "missing-llm-key"} else _commit(workspace, "Prepare " + case)
                directory = output / case
                exit_code = None
                try:
                    exit_code = _container(image, workspace, directory, case, base, head)
                    (directory / "exit-code.json").write_text(
                        json.dumps({"exit_code": exit_code}) + "\n", encoding="utf-8"
                    )
                    verified = verify_case(directory / "reports", case, exit_code)
                    if case in {"changed-one", "no-change"}:
                        selection = _load(directory / "reports" / "summary.json").get("selection", {})
                        _require(
                            selection.get("base") == base and selection.get("head") == head,
                            "Recorded Git selection does not match the requested commits",
                        )
                    verified.update(base_sha=base, head_sha=head)
                    receipt["cases"].append(verified)
                    print(f"SkillEvaluator image smoke: {case} verified.", flush=True)
                except (OSError, ValueError, KeyError, TypeError, SmokeError, subprocess.SubprocessError) as error:
                    receipt["cases"].append(
                        {
                            "case": case,
                            "verified": False,
                            "exit_code": exit_code,
                            "error": type(error).__name__ + ": " + str(error)[:1000],
                        }
                    )
                    print(f"SkillEvaluator image smoke: {case} FAILED; inspect retained reports.", flush=True)
        receipt["all_verified"] = len(receipt["cases"]) == 5 and all(case["verified"] for case in receipt["cases"])
    finally:
        (output / "smoke-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Already built or pulled local Docker image")
    parser.add_argument("--output", required=True, type=Path, help="New/empty evidence directory")
    args = parser.parse_args()
    try:
        return 0 if smoke(args.image, args.output)["all_verified"] else 1
    except (OSError, ValueError, SmokeError, subprocess.SubprocessError):
        print("SkillEvaluator image smoke failed; verify Docker, Git and the evidence output path.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
