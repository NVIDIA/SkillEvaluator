# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, T201 -- standalone CI entrypoint with escaped diagnostic output.

"""Run complete Tier 1 checks on configured repository skills and retain evidence.

Raw scanner output belongs in artifacts. Structured findings are escaped before
being shown in logs or GitHub annotations.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from skillevaluator.reporting.html import _validated_ci_context
from skillevaluator.utils.secret_redaction import Redactor

try:
    from secret_redaction import collect_sensitive_values, sanitized_outputs
except ModuleNotFoundError:
    from ci.secret_redaction import collect_sensitive_values, sanitized_outputs

try:
    from native_reports import render_native_reports
except ModuleNotFoundError:
    from ci.native_reports import render_native_reports

try:
    from publish_reports import PublicationError, validate_report
except ModuleNotFoundError:
    from ci.publish_reports import PublicationError, validate_report

try:
    from report_views import finding_repository_path, render_reports
except ModuleNotFoundError:
    from ci.report_views import finding_repository_path, render_reports

try:
    from select_changed_skills import SelectionError, select_changed_skills
except ModuleNotFoundError:
    from ci.select_changed_skills import SelectionError, select_changed_skills

MAX_SKILLS = 100
MAX_REPORT_BYTES = 32 * 1024 * 1024
SCAN_TIMEOUT_SECONDS = 3000
PROVIDER_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "nv_build": "NVIDIA_API_KEY"}
PROCESS_ENV = {
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}


class ConfigurationError(ValueError):
    """An invalid request, without echoing untrusted paths or credentials."""


def discover_skills(workspace: Path, roots: str) -> list[Path]:
    """Discover a bounded unique set; any linked scan content is rejected."""
    workspace = workspace.resolve(strict=True)
    configured = [line.strip() for line in roots.splitlines() if line.strip()]
    if not configured:
        raise ConfigurationError("SKILL_PATHS must contain at least one relative directory.")
    discovered: set[Path] = set()
    for raw in configured:
        relative = Path(raw)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in raw
            or ":" in raw
            or any(ord(character) < 32 or ord(character) == 127 for character in raw)
        ):
            raise ConfigurationError("Skill roots must be relative directories without parent traversal.")
        current = workspace
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ConfigurationError("Symlinks are not supported in skill roots or scan content.")
        root = workspace / relative
        if not root.is_dir() or not root.resolve(strict=True).is_relative_to(workspace):
            raise ConfigurationError("A configured skill root is missing or outside the workspace.")

        def inaccessible(_error):
            raise ConfigurationError("A skill directory could not be read.")

        for directory, directories, files in os.walk(root, followlinks=False, onerror=inaccessible):
            parent = Path(directory)
            inside_skill = any(parent.is_relative_to(skill) for skill in discovered) or "SKILL.md" in files
            for name in [*directories, *files]:
                child = parent / name
                if any(ord(character) < 32 or ord(character) == 127 for character in name):
                    raise ConfigurationError("Skill paths must not contain control characters.")
                if child.is_symlink() and (inside_skill or name == "SKILL.md"):
                    raise ConfigurationError("Symlinks are not supported in skill roots or scan content.")
                if inside_skill and not child.is_dir() and not child.is_file():
                    raise ConfigurationError("Only regular files and directories are supported in skill trees.")
            if "SKILL.md" in files:
                discovered.add(parent)
                if len(discovered) > MAX_SKILLS:
                    raise ConfigurationError("The CI integration supports at most 100 skills per run.")
    if not discovered:
        raise ConfigurationError("No SKILL.md files were found under the configured skill roots.")
    return sorted(discovered, key=lambda item: item.relative_to(workspace).as_posix())


def child_environment(environ: dict[str, str]) -> tuple[dict[str, str], bool]:
    """Limit provider selection to the explicit CI inputs, including static runs."""
    enabled = environ.get("LLM_ENABLED", "false").strip().lower()
    if enabled not in {"true", "false"}:
        raise ConfigurationError("LLM_ENABLED must be true or false.")
    env = {key: value for key, value in environ.items() if key in PROCESS_ENV}
    env.setdefault("PATH", os.defpath)
    env["HOME"] = "/tmp/evaluator-home"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["SEMGREP_SEND_METRICS"] = "off"
    env["SEMGREP_ENABLE_VERSION_CHECK"] = "0"
    env["NO_COLOR"] = "1"
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    if enabled == "true":
        provider = environ.get("LLM_PROVIDER", "").strip()
        if provider not in PROVIDER_KEYS:
            raise ConfigurationError("LLM_PROVIDER must be openai, anthropic, or nv_build when LLM is enabled.")
        key = environ.get("LLM_API_KEY", "").strip()
        if not key:
            raise ConfigurationError("LLM_API_KEY is required when LLM_ENABLED is true.")
        env["SKILL_EVAL_LLM_PROVIDER"] = provider
        env[PROVIDER_KEYS[provider]] = key
        model = environ.get("LLM_MODEL", "").strip()
        if model:
            if any(ord(character) < 32 for character in model):
                raise ConfigurationError("LLM_MODEL must not contain control characters.")
            env["SKILL_EVAL_LLM_MODEL"] = model
    return env, enabled == "true"


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _load_report(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("The evaluator did not produce a readable JSON report.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The evaluator JSON report is malformed.")
    if type(data.get("overall_passed")) is not bool:
        raise ValueError("The evaluator JSON report lacks a boolean verdict.")
    if data.get("overall_status") not in {"passed", "failed", "incomplete"}:
        raise ValueError("The evaluator JSON report lacks a recognized status.")
    scans = data.get("incomplete_scans")
    if not isinstance(scans, list) or any(not isinstance(scan, str) for scan in scans):
        raise ValueError("The evaluator JSON report lacks scanner completeness evidence.")
    results = data.get("results")
    if not isinstance(results, list) or not results or any(not isinstance(result, dict) for result in results):
        raise ValueError("The evaluator JSON report lacks validator results.")
    for name in ("total_validators", "total_errors", "total_warnings"):
        if type(data.get(name)) is not int or data[name] < 0:
            raise ValueError("The evaluator JSON report has invalid result counts.")
    if data["total_validators"] != len(results):
        raise ValueError("The evaluator JSON report has inconsistent validator counts.")
    counts = data.get("severity_counts")
    if not isinstance(counts, dict) or any(
        type(counts.get(name)) is not int or counts[name] < 0 for name in ("critical", "high", "medium", "low")
    ):
        raise ValueError("The evaluator JSON report has invalid severity counts.")
    from skillevaluator.reporting.retained import retained_results

    retained_results(data)
    return data


def _summary(skills: list[dict], error: str | None = None) -> dict:
    passed = sum(item["passed"] for item in skills)
    incomplete = sum(item["overall_status"] in {"incomplete", "error"} for item in skills)
    payload = {
        "total": len(skills),
        "passed": passed,
        "failed": len(skills) - passed,
        "incomplete": incomplete,
        "overall_passed": bool(skills) and passed == len(skills) and error is None,
        "overall_status": "error"
        if error
        else "incomplete"
        if incomplete
        else "passed"
        if passed == len(skills) and skills
        else "failed",
        "total_validators": sum(item.get("total_validators", 0) for item in skills),
        "total_errors": sum(item.get("total_errors", 0) for item in skills),
        "total_warnings": sum(item.get("total_warnings", 0) for item in skills),
        "severity_counts": {
            name: sum(item.get("severity_counts", {}).get(name, 0) for item in skills)
            for name in ("critical", "high", "medium", "low")
        },
        "skills": skills,
    }
    if error:
        payload["error"] = error
    return payload


def _write_summary(reports: Path, summary: dict) -> None:
    _write_json(reports / "summary.json", summary)
    render_native_reports(reports, summary, summary.get("execution", {}))


def annotation_text(value: str, *, property_value: bool = False) -> str:
    """Escape GitHub command syntax; finding text must never create commands."""
    escaped = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return escaped.replace(":", "%3A").replace(",", "%2C") if property_value else escaped


def log_result(row: dict, data: dict | None, *, annotations: bool, budget: dict[str, int]) -> None:
    """Explain verdicts promptly without replaying arbitrary subprocess logs."""
    print(f"Skill result: {json.dumps(row['path'])} — {row['overall_status'].upper()}", flush=True)
    if not data:
        print(f"  Reason: {json.dumps(row.get('error', 'No complete report was produced.'))}", flush=True)
        return
    for quality in data.get("quality_summary", []):
        print(
            f"  Quality: {json.dumps(quality.get('overall_score', 'unknown'))}/100, grade {json.dumps(quality.get('grade'))}",
            flush=True,
        )
    shown = 0
    for result in data.get("results", []):
        print(f"  Check: {json.dumps(result.get('validator'))} — {json.dumps(result.get('status'))}", flush=True)
        if result.get("incomplete_scans"):
            print(f"    Incomplete scanners: {json.dumps(result['incomplete_scans'])}", flush=True)
        for finding in result.get("findings", []):
            if finding.get("severity") not in {"critical", "high"} or shown >= 20:
                continue
            shown += 1
            message = str(finding.get("message", "Finding details unavailable."))[:4000]
            suggestion = str(finding.get("suggestion") or "Consult the full report.")[:2000]
            location = str(finding.get("file_path") or row["path"])
            print(f"    Failure: {json.dumps(message)}", flush=True)
            print(f"    Location: {json.dumps(location)}", flush=True)
            print(f"    Suggested fix: {json.dumps(suggestion)}", flush=True)
            if annotations and result.get("passed") is False and budget["errors"] < 10:
                budget["errors"] += 1
                properties = "title=" + annotation_text(str(result.get("validator", "Tier 1")), property_value=True)
                relative = finding_repository_path(row["path"], finding.get("file_path"))
                if relative:
                    properties += ",file=" + annotation_text(relative, property_value=True)
                    line = finding.get("line_number")
                    if type(line) is int and line > 0:
                        properties += f",line={line}"
                print(f"::error {properties}::{annotation_text(message + ' Suggested fix: ' + suggestion)}", flush=True)
        if result.get("status") in {"incomplete", "failed"}:
            for message in result.get("legacy", {}).get("errors", [])[:3]:
                print(f"    Scanner error: {json.dumps(str(message)[:4000])}", flush=True)


def execution_context(environ: dict[str, str]) -> dict:
    return {
        "run_url": environ.get("REPORT_RUN_URL", ""),
        "commit": environ.get("REPORT_COMMIT", ""),
        "image": environ.get("REPORT_IMAGE", ""),
        "report_url": environ.get("REPORT_URL", ""),
        "llm_enabled": environ.get("LLM_ENABLED", "false").lower() == "true",
        "workflow_sha": environ.get("CALLER_WORKFLOW_SHA", ""),
    }


def run(
    workspace: Path, reports: Path, environ: dict[str, str], *, execute=subprocess.run, clock=time.monotonic
) -> int:
    """Run all selected skills; return nonzero on every incomplete or failed run."""
    if reports.is_symlink():
        raise ConfigurationError("The report directory must not be a symlink.")
    reports.mkdir(parents=True, exist_ok=True)
    if any(reports.iterdir()):
        raise ConfigurationError("The report directory must be empty to avoid stale evidence.")
    reports = reports.resolve(strict=True)
    rows = []
    context = execution_context(environ)
    annotation_budget = {"errors": 0}
    selection = None
    tokens: set[str] = set()
    try:
        try:
            _validated_ci_context(context)
            if environ.get("REPORT_BASE_URL"):
                _validated_ci_context({"report_url": environ["REPORT_BASE_URL"]})
        except ValueError as exc:
            # Invalid context must not crash rendering of the configuration error.
            context = {"llm_enabled": context["llm_enabled"]}
            raise ConfigurationError(
                "Report links must be bounded HTTPS URLs without credentials or control characters."
            ) from exc
        workspace = workspace.resolve(strict=True)
        if reports.is_relative_to(workspace):
            raise ConfigurationError("The report directory must be outside the read-only workspace.")
        env, llm = child_environment(environ)
        mode = environ.get("SCAN_MODE", "all")
        if mode not in {"all", "changed"}:
            raise ConfigurationError("SCAN_MODE must be all or changed.")
        roots = environ.get("SKILL_PATHS", "skills")
        if mode == "changed":
            selected, selection = select_changed_skills(
                workspace, roots, environ.get("DIFF_BASE_SHA", ""), environ.get("DIFF_HEAD_SHA", "")
            )
            if selected:
                # Validate scan trees without adding nested unchanged targets.
                discover_skills(workspace, "\n".join(path.relative_to(workspace).as_posix() for path in selected))
        else:
            selected = discover_skills(workspace, roots)
            selection = {
                "mode": "all",
                "base": None,
                "head": None,
                "discovered": len(selected),
                "selected": [path.relative_to(workspace).as_posix() for path in selected],
                "deleted": [],
            }
        deadline = clock() + SCAN_TIMEOUT_SECONDS
        try:
            if llm:
                provider_key = environ.get("LLM_API_KEY", "").strip()
                if len(provider_key) < 4:
                    raise ValueError("The provider key is too short for safe redaction.")
                tokens.add(provider_key)
            # A credential found in one selected skill can also be echoed by
            # another. Collect the union before writing or logging any paths.
            for target in selected:
                tokens.update(collect_sensitive_values(target))
                if clock() >= deadline:
                    raise ValueError("Credential preflight exceeded the scan time limit.")
            redactor = Redactor(tokens)
            metadata = json.dumps({"selection": selection, "execution": context}, ensure_ascii=True)
            if redactor.text(metadata) != metadata:
                raise ValueError("A credential is present in report context.")
        except (OSError, ValueError, UnicodeError, RecursionError) as exc:
            # Neither a path nor a context field is safe to echo on failure.
            selection = None
            context = {"llm_enabled": context.get("llm_enabled", False)}
            raise ConfigurationError(
                "Credential preflight could not safely prepare the scan; no results were published."
            ) from exc
    except (ConfigurationError, SelectionError, OSError) as exc:
        message = (
            str(exc)
            if isinstance(exc, (ConfigurationError, SelectionError))
            else "The workspace or runtime configuration could not be read."
        )
        summary = _summary([], error=message)
        if isinstance(exc, SelectionError):
            selection = exc.selection
        if selection is not None:
            summary["selection"] = selection
        summary["execution"] = context
        _write_summary(reports, summary)
        print(f"SkillEvaluator configuration failed: {json.dumps(message)}", flush=True)
        return 1
    if selection["deleted"]:
        print(f"Deleted skill directories skipped (removed completely): {json.dumps(selection['deleted'])}", flush=True)
    if not selected:
        summary = _summary([])
        summary.update(overall_status="skipped", overall_passed=False, selection=selection, execution=context)
        _write_summary(reports, summary)
        print("NO SKILLS CHANGED: no skill was evaluated between the selected Git commits.", flush=True)
        return 0
    rows = [
        {
            "path": target.relative_to(workspace).as_posix(),
            "report_dir": f"skill-{index:03d}",
            "exit_code": None,
            "overall_status": "error",
            "incomplete_scans": [],
            "passed": False,
            "error": "Not evaluated: the scan has not reached this skill.",
        }
        for index, target in enumerate(selected, 1)
    ]

    def checkpoint() -> None:
        summary = _summary(rows)
        summary.update(selection=selection, execution=context)
        _write_json(reports / "summary.json", summary)
        render_reports(reports, summary, context)

    checkpoint()
    for index, (target, row) in enumerate(zip(selected, rows, strict=True), 1):
        remaining = deadline - clock()
        if remaining <= 0:
            break
        print(
            f"SkillEvaluator: evaluating skill {index} of {len(selected)}: {json.dumps(target.relative_to(workspace).as_posix())}",
            flush=True,
        )
        output = reports / f"skill-{index:03d}"
        output.mkdir()
        row["error"] = "Evaluation interrupted before complete evidence was retained."
        data = None
        try:
            remaining = deadline - clock()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("skillevaluator", SCAN_TIMEOUT_SECONDS)
            # Raw scanner files and logs live only in a private temporary
            # directory, never in the upload directory. Docker mounts /audit-tmp
            # as tmpfs, so cancellation cannot upload half-written secrets.
            with tempfile.TemporaryDirectory(prefix="skillevaluator-private-", dir=env.get("TMPDIR")) as temporary:
                raw = Path(temporary).resolve(strict=True)
                if raw.is_relative_to(workspace) or raw.is_relative_to(reports):
                    raise ValueError("A private scanner directory must be outside the workspace and reports.")
                argv = [
                    "skillevaluator",
                    "tier1",
                    str(target),
                    "--llm" if llm else "--no-llm",
                    "--profile",
                    "external",
                    "-r",
                    "json,html,markdown,sarif",
                    "-o",
                    str(raw),
                ]
                with (raw / "stdout.log").open("xb") as stdout, (raw / "stderr.log").open("xb") as stderr:
                    process = execute(
                        argv, cwd=workspace, env=env, stdout=stdout, stderr=stderr, check=False, timeout=remaining
                    )
                safe = sanitized_outputs(raw, tokens)
            for name, content in safe.items():
                (output / name).write_bytes(content)
            row["exit_code"] = process.returncode
            data = _load_report(output / "skillevaluator-tier1.json")
            for key in (
                "overall_status",
                "incomplete_scans",
                "total_validators",
                "total_errors",
                "total_warnings",
                "severity_counts",
            ):
                row[key] = data[key]
            row["passed"] = (
                process.returncode == 0
                and data["overall_passed"] is True
                and data["overall_status"] == "passed"
                and not data["incomplete_scans"]
                and all(
                    result.get("passed") is True
                    and result.get("status") not in {"incomplete", "failed"}
                    and not result.get("incomplete_scans")
                    for result in data["results"]
                )
            )
            try:
                # The publisher rejects the whole artifact for one contradictory
                # report. Check it the same way here so only this skill is lost.
                validate_report(data, row)
                consistent = True
            except PublicationError:
                consistent = False
            if not consistent or (not row["passed"] and row["overall_status"] == "passed"):
                row["overall_status"] = "error"
                row["incomplete_scans"] = []
                row["passed"] = False
                row["error"] = "Process exit or validator results contradict the report verdict."
            else:
                row.pop("error", None)
        except subprocess.TimeoutExpired:
            row["error"] = "Scan time limit reached; this skill did not finish."
            checkpoint()
            break
        except (OSError, ValueError, UnicodeError, RecursionError):
            row["error"] = "Evaluator execution failed or its JSON report was missing, unreadable, or invalid."
        log_result(row, data, annotations=environ.get("GITHUB_ACTIONS") == "true", budget=annotation_budget)
        checkpoint()
    summary = _summary(rows)
    summary["selection"] = selection
    summary["execution"] = context
    _write_summary(reports, summary)
    print(f"SkillEvaluator finished: {summary['passed']} of {summary['total']} skills passed.", flush=True)
    return 0 if summary["overall_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--reports", type=Path, default=Path("/reports"))
    args = parser.parse_args()
    try:
        return run(args.workspace, args.reports, dict(os.environ))
    except (ConfigurationError, OSError):
        print("SkillEvaluator could not safely initialize the report directory.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
