# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code Risk Analysis Validator using Bandit and Semgrep.

Performs static code analysis to detect security vulnerabilities,
code quality issues, and potential bugs via CWE/OWASP patterns.
"""

from __future__ import annotations

import glob
import math
import os
from fnmatch import fnmatch
from pathlib import Path
from time import monotonic

from skillevaluator.config import CONFIG_DIR
from skillevaluator.constants import SCAN_EXCLUDED_DIRS, SCAN_EXCLUDED_FILES
from skillevaluator.utils.path_security import matches_filesystem_name
from skillevaluator.utils.tool_runner import Severity, Tools, parse_json_output
from skillevaluator.validators.base import Finding, ValidationResult, ValidatorBase, iter_scannable_files

_SEMGREP_PARSE_ERROR_TYPES = frozenset(
    {"SyntaxError", "PartialParsing", "LexicalError", "ParseError", "OtherParseError", "AstBuilderError"}
)
_SEMGREP_PARSE_ERROR_PATH_LIMIT = 20


def _semgrep_error_type(error: dict) -> str:
    """Return the Semgrep error type name; JSON reports use a string or ``["Name", ...]``."""
    error_type = error.get("type")
    if isinstance(error_type, list) and error_type:
        error_type = error_type[0]
    return error_type if isinstance(error_type, str) else ""


def _is_semgrep_parse_error(error: dict) -> bool:
    """True for a file-level parse error: Semgrep ran but could not parse one input file."""
    return bool(error.get("path")) and _semgrep_error_type(error) in _SEMGREP_PARSE_ERROR_TYPES


def _semgrep_display_path(path: str, skill_root: Path | None) -> str:
    """Return *path* relative to the scanned skill.

    Semgrep reports absolute paths for the absolute target ``_run_semgrep`` passes, so a
    raw path would put the user's home directory into every report. A path outside the
    skill keeps only its file name.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    if skill_root is not None:
        try:
            return candidate.relative_to(skill_root).as_posix()
        except ValueError:
            pass
    return candidate.name


def _semgrep_error_line(error: dict) -> int | None:
    spans = error.get("spans")
    if isinstance(spans, list) and spans and isinstance(spans[0], dict):
        start = spans[0].get("start")
        if isinstance(start, dict) and isinstance(start.get("line"), int):
            return start["line"]
    return None


def _semgrep_file_excludes(skill_path: Path) -> list[str]:
    """Return generated file patterns using the target filesystem's case semantics."""
    excludes = set(SCAN_EXCLUDED_FILES)
    try:
        for entry in skill_path.iterdir():
            if entry.is_file() and matches_filesystem_name(entry, SCAN_EXCLUDED_FILES):
                excludes.add(entry.name)
    except OSError:
        pass
    return sorted(excludes)


REQUIRE_SEMGREP_ENV = "SKILLEVALUATOR_REQUIRE_SEMGREP"

# Empty Bandit INI passed as ``--ini`` so Bandit never discovers a ``.bandit``
# file inside the scanned skill. Bandit only searches the target tree for
# ``.bandit`` when ``--ini`` is not given.
BANDIT_INI_PATH = (CONFIG_DIR / "bandit.ini").resolve()
_BANDIT_INI_REPORT_LIMIT = 5
# One scan gets this many seconds, however many Bandit runs it takes.
_BANDIT_TIMEOUT = 120
# The most bytes of file names that one Bandit run is given. Windows allows
# 32,767 characters for a command line.
_BANDIT_NAMED_FILES_BYTES = 24_000
# The most scan errors of a scan that are written out.
_BANDIT_ERROR_LIMIT = 3


def _skill_bandit_ini_files(skill_path: Path) -> list[str]:
    """Return skill-relative paths of ``.bandit`` files that Bandit would otherwise apply."""
    found: list[str] = []
    try:
        for root, dirs, filenames in os.walk(skill_path):
            dirs[:] = [d for d in dirs if d not in SCAN_EXCLUDED_DIRS]
            if ".bandit" in filenames and Path(root, ".bandit").is_file():
                found.append(Path(root, ".bandit").relative_to(skill_path).as_posix())
                if len(found) >= _BANDIT_INI_REPORT_LIMIT:
                    break
    except OSError:
        pass
    return sorted(found)


class CodeRiskValidator(ValidatorBase):
    """Static code analysis for security risks.

    Combines Bandit's Python checks with a packaged, high-confidence Semgrep
    baseline for command/dynamic-code execution, Python SQL interpolation,
    and download-to-shell patterns.
    """

    def __init__(
        self,
        use_semgrep: bool = True,
        fail_on_low: bool = False,
        exclude_tests: bool = True,
    ):
        """Initialize code risk validator.

        Args:
            use_semgrep: Run Semgrep for additional coverage
            fail_on_low: Treat LOW severity as errors
            exclude_tests: Skip test files and directories
        """
        self.use_semgrep = use_semgrep
        self.fail_on_low = fail_on_low
        self.exclude_tests = exclude_tests

    @property
    def name(self) -> str:
        return "Code Risk Analysis"

    @property
    def description(self) -> str:
        return "Static code analysis using Bandit and packaged Semgrep rules"

    def validate(self, skill_path: Path) -> ValidationResult:
        """Run code risk analysis on skill(s) at path."""
        return self._validate_folder_or_skill(
            skill_path,
            self._validate_single_skill,
            action_description="Analyzing code risk for",
        )

    def _validate_single_skill(self, skill_path: Path) -> ValidationResult:
        """Run static analysis on a single skill directory."""
        result = ValidationResult()

        file_counts = self._count_code_files(skill_path)
        if not any(file_counts.values()):
            result.add_success(
                check_name="code_file_discovery",
                message="No code files found - code risk analysis is not applicable",
                file_counts=file_counts,
            )
            return result

        result.add_message(
            f"Found {file_counts['py']} Python, {file_counts['sh']} Shell, "
            f"{file_counts['js']} JavaScript/TypeScript files"
        )

        if file_counts["py"]:
            result.merge(self._run_bandit(skill_path))

        if self.use_semgrep:
            result.merge(self._run_semgrep(skill_path))

        return result

    def _count_code_files(self, skill_path: Path) -> dict[str, int]:
        """Count code files by type.

        Files under Tier 1 artifact directories (``evals/``, ``results/``,
        ``versions/`` and dot-prefixed variants) are excluded so that
        evaluation snapshots do not skew the counts that decide whether
        bandit / semgrep should run.
        """
        return {
            "py": len(iter_scannable_files(skill_path, {".py"})),
            "sh": len(iter_scannable_files(skill_path, {".sh"})),
            "js": len(
                iter_scannable_files(
                    skill_path,
                    {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"},
                )
            ),
        }

    def _run_bandit(self, skill_path: Path) -> ValidationResult:
        """Run Bandit security linter on Python code."""
        result = ValidationResult()

        if not Tools.bandit.is_available:
            result.add_warning(f"bandit not installed. {Tools.bandit.get_install_hint()}")
            result.mark_scan_incomplete("bandit")
            return result

        # resolve() so a skill dir named like an option token (e.g.
        # "--exclude=*") cannot be parsed as a scanner flag.
        root = skill_path.resolve()

        for ini_file in _skill_bandit_ini_files(skill_path):
            result.add_warning(
                f"Bandit configuration file {ini_file} is ignored: skill-supplied "
                "Bandit settings and # nosec comments are not applied"
            )

        runs = self._bandit_runs(root)
        issues: list[dict] = []
        errors = 0
        complete = True
        # With nothing left to scan there is no run, and the scan is done.
        recorded = not runs
        if recorded:
            result.record_scanner("bandit", Tools.bandit.version)
        deadline = monotonic() + _BANDIT_TIMEOUT
        for targets, excludes in runs:
            # The runs share the time one run had, so a scan takes no longer
            # than before.
            timeout = math.ceil(deadline - monotonic())
            if timeout <= 0:
                result.add_warning(f"{Tools.bandit.name} timed out after {_BANDIT_TIMEOUT} seconds")
                result.mark_scan_incomplete("bandit")
                complete = False
                break
            args = [*targets, "-f", "json", "-q", "-ll"]
            # This also replaces Bandit's own default list, which is matched as
            # substrings and has ".git" in it.
            args.extend(["--exclude", excludes])
            # Without --ini, Bandit walks the scanned tree for a ``.bandit`` file and
            # applies its ``skips`` / ``tests`` / ``configfile`` settings, so a skill
            # could switch off the checks that would flag it. Point Bandit at a
            # packaged, empty INI instead, and keep ``# nosec`` comments from
            # suppressing findings.
            args.extend(["--ini", str(BANDIT_INI_PATH), "--ignore-nosec"])

            tool_result = Tools.bandit.run(args, timeout=timeout)

            if not tool_result.error_message and tool_result.exit_code in (0, 1):
                # Exit code 0 = clean, 1 = findings; both emit JSON on stdout
                if not recorded:
                    result.record_scanner("bandit", Tools.bandit.version)
                    recorded = True
                found, scan_errors = self._read_bandit_output(
                    tool_result.stdout,
                    result,
                    exit_code=tool_result.exit_code,
                    error_limit=_BANDIT_ERROR_LIMIT - errors,
                )
                issues.extend(found)
                errors += len(scan_errors or ())
                complete = complete and scan_errors == []
                continue
            if tool_result.error_message:
                result.add_warning(tool_result.error_message)
            else:
                error_msg = tool_result.stderr or tool_result.stdout
                result.add_warning(f"Bandit error (exit code {tool_result.exit_code}): {error_msg[:200]}")
            # The scan is incomplete from here, whatever a later run finds.
            result.mark_scan_incomplete("bandit")
            complete = False
            break

        if recorded:
            self._report_bandit_issues(result, issues, complete)

        return result

    def _bandit_runs(self, root: Path) -> list[tuple[list[str], str]]:
        """The targets and the ``--exclude`` value of each Bandit run that the scan of *root* takes.

        One run over the tree is enough when the exclude patterns hide what
        the scan leaves out and nothing else. They can hide more. Bandit's
        ``*`` also matches a path separator, so ``test_*.py`` matches every
        Python file below a directory such as ``test_tools``, and Bandit
        also reads each entry as plain text. A file that the patterns hide
        and the scan covers is therefore passed by name, in a run that
        excludes nothing and opens no other file. Bandit adds its own ``*``
        to an entry that is the path of a directory, and the entry then no
        longer hides what it should. With such an entry every file is passed
        by name, and Bandit does not walk the tree at all.
        """
        excludes = self._bandit_excludes(root)
        files = self._bandit_files(root)
        runs: list[tuple[list[str], str]] = []
        # os.path.isdir is the call Bandit makes. It is False for a path that
        # cannot be read, where Path.is_dir raises.
        if not any(os.path.isdir(entry) for entry in excludes):  # noqa: PTH112
            runs.append((["-r", str(root)], ",".join(excludes)))
            files = [file for file in files if any(fnmatch(file, entry) or entry in file for entry in excludes)]
        named: list[list[str]] = []
        size = 0
        for file in files:
            # The name, a space after it and the quotes it may get. A
            # character takes no more room on a Windows command line than it
            # has bytes here.
            needed = len(os.fsencode(file)) + 3
            if not named or size + needed > _BANDIT_NAMED_FILES_BYTES:
                named.append([])
                size = 0
            named[-1].append(file)
            size += needed
        runs.extend((batch, "") for batch in named)
        return runs

    def _bandit_files(self, root: Path) -> list[str]:
        """The files below *root* that the Bandit scan covers, under the paths Bandit gives them.

        Bandit reads the ``*.py`` and ``*.pyw`` files of a tree and does not
        follow a link to a directory. The scan leaves out the artifact
        directories, and the test directories and test files when tests are
        excluded.
        """
        skipped = self._bandit_directory_names(root)
        files: list[str] = []
        for current, directories, names in os.walk(str(root)):
            directories[:] = [name for name in directories if not any(fnmatch(name, pattern) for pattern in skipped)]
            for name in names:
                if not (fnmatch(name, "*.py") or fnmatch(name, "*.pyw")):
                    continue
                if self.exclude_tests and self._is_test_name(name):
                    continue
                files.append(str(Path(current, name)))
        return sorted(files)

    def _bandit_excludes(self, root: Path) -> list[str]:
        """Patterns for what Bandit skips below *root*.

        Tier 1 artifact dirs (evals/, results/, versions/, ...) hold snapshot
        copies of the live skill, and scanning them produces duplicate
        findings. Bandit drops a file when an ``--exclude`` entry is a
        substring of its whole path, so a bare name such as "test" also hid
        every file of a skill called "webapp-testing". Each directory name
        is therefore written as two patterns anchored at *root*, one for a
        directory of that name right below it and one for any depth, so the
        option has the same size for a small skill and a large tree. When
        tests are excluded, the ``test_*.py`` and ``*_test.py`` files are
        matched by name in the same way, so that Bandit does not open them.
        """
        # Bandit splits the option at every comma, so a comma in the path is
        # written as "?". Every scanned path starts with this same root.
        prefix = glob.escape(str(root)).replace(",", "?")
        patterns: list[str] = []
        for name in self._bandit_directory_names(root):
            for parent in (prefix, f"{prefix}{os.sep}*"):
                patterns.append(f"{parent}{os.sep}{name}{os.sep}*")
        if self.exclude_tests:
            patterns += [f"{prefix}{os.sep}{name}" for name in ("*_test.py", "test_*.py", f"*{os.sep}test_*.py")]
        return sorted(patterns)

    def _bandit_directory_names(self, root: Path) -> list[str]:
        """Patterns for the names of the directories that the scan of *root* leaves out at any depth."""
        names = [glob.escape(name) for name in sorted({"tests", "test"})] if self.exclude_tests else []
        # A differently cased spelling is the same artifact directory where
        # the filesystem ignores case, as for the other Tier 1 scans.
        ignores_case = self._ignores_case(root)
        for name in sorted(SCAN_EXCLUDED_DIRS):
            names.append(
                "".join(f"[{char.lower()}{char.upper()}]" if char.isalpha() else glob.escape(char) for char in name)
                if ignores_case
                else glob.escape(name)
            )
        return names

    @staticmethod
    def _ignores_case(root: Path) -> bool:
        """Whether the filesystem holding *root* treats differently cased names as one entry."""
        swapped = Path(str(root).swapcase())
        try:
            return swapped != root and root.samefile(swapped)
        except OSError:
            return False

    @staticmethod
    def _is_test_name(name: str) -> bool:
        return fnmatch(name, "test_*.py") or fnmatch(name, "*_test.py")

    def _read_bandit_output(
        self,
        output: str,
        result: ValidationResult,
        *,
        exit_code: int | None = None,
        error_limit: int = _BANDIT_ERROR_LIMIT,
    ) -> tuple[list[dict], list | None]:
        """Parse the JSON output of one Bandit run.

        Return its issues and its scan errors, with None in their place for
        a report that cannot be used. No more than *error_limit* of the scan
        errors are written out.
        """
        data = parse_json_output(output)
        if not isinstance(data, dict):
            result.add_error("Bandit did not return valid JSON; scan did not complete")
            result.mark_scan_incomplete("bandit")
            return [], None

        issues = data.get("results")
        if not isinstance(issues, list) or not all(isinstance(issue, dict) for issue in issues):
            result.add_error("Bandit JSON report is missing a valid 'results' list; scan did not complete")
            result.mark_scan_incomplete("bandit")
            return [], None

        errors = data.get("errors", [])
        if not isinstance(errors, list):
            result.add_error("Bandit JSON field 'errors' must be a list; scan did not complete")
            result.mark_scan_incomplete("bandit")
            return [], None
        if errors:
            for error in errors[: max(error_limit, 0)]:
                detail = error.get("reason") or error.get("message") if isinstance(error, dict) else str(error)
                result.add_error(f"Bandit scan error: {str(detail or 'Unknown error')[:200]}")
            result.mark_scan_incomplete("bandit")

        if exit_code == 1 and not issues:
            result.add_error("Bandit findings exit code 1 did not include findings; scan did not complete")
            result.mark_scan_incomplete("bandit")
            return [], errors or None
        if exit_code == 0 and issues:
            result.add_error("Bandit returned findings with clean exit code 0; scanner status is inconsistent")
            result.mark_scan_incomplete("bandit")

        return issues, errors

    def _report_bandit_issues(self, result: ValidationResult, issues: list[dict], complete: bool) -> None:
        """Report the issues of a scan, which is *complete* when each of its Bandit runs ran and read every file."""
        if not issues:
            if complete:
                result.add_success(
                    check_name="bandit",
                    message="Bandit: No security issues found",
                )
            return

        # Summarize by severity
        counts = self._count_by_severity(issues)
        result.add_message(f"Bandit found: {counts['HIGH']} HIGH, {counts['MEDIUM']} MEDIUM, {counts['LOW']} LOW")

        for issue in issues:
            self._report_bandit_issue(result, issue)

    def _count_by_severity(self, issues: list[dict]) -> dict[str, int]:
        """Count issues by severity level."""
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for issue in issues:
            sev = issue.get("issue_severity", "LOW").upper()
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    def _report_bandit_issue(self, result: ValidationResult, issue: dict) -> None:
        """Report a single Bandit finding."""
        severity_str = issue.get("issue_severity", "MEDIUM").upper()
        confidence = issue.get("issue_confidence", "MEDIUM").upper()
        severity = self._map_bandit_severity(severity_str)

        test_id = issue.get("test_id", "B000")
        test_name = issue.get("test_name", "unknown")
        issue_text = issue.get("issue_text", "Security issue detected")
        filename = issue.get("filename", "unknown")
        line = issue.get("line_number", 0)
        cwe_id = issue.get("issue_cwe", {}).get("id", "")

        cwe_ref = f" (CWE-{cwe_id})" if cwe_id else ""
        message = f"{issue_text}{cwe_ref}"

        # HIGH severity or MEDIUM with decent confidence are errors
        is_error = severity_str == "HIGH" or (severity_str == "MEDIUM" and confidence in ("HIGH", "MEDIUM"))

        result.add_structured_finding(
            Finding(
                category="BANDIT",
                severity=severity,
                check_name=f"{test_id}:{test_name}",
                message=message,
                file_path=filename,
                line_number=line,
                suggestion=f"Review {test_name} ({test_id}){cwe_ref}",
                metadata={"cwe": cwe_id} if cwe_id else {},
            ),
            is_error=is_error,
        )

    def _map_bandit_severity(self, severity_str: str) -> Severity:
        """Map Bandit severity string to Severity enum."""
        mapping = {
            "HIGH": Severity.HIGH,
            "MEDIUM": Severity.MEDIUM,
            "LOW": Severity.LOW,
        }
        return mapping.get(severity_str.upper(), Severity.MEDIUM)

    def _run_semgrep(self, skill_path: Path) -> ValidationResult:
        """Run Semgrep security scanner."""
        result = ValidationResult()

        if not Tools.semgrep.is_available:
            if os.environ.get(REQUIRE_SEMGREP_ENV, "").strip().lower() in {"1", "true", "yes"}:
                result.add_warning(f"semgrep not installed. {Tools.semgrep.get_install_hint()}")
                result.mark_scan_incomplete("semgrep")
            else:
                result.add_message(f"Semgrep not installed; skipped. {Tools.semgrep.get_install_hint()}")
            return result

        rules_path = (CONFIG_DIR / "semgrep_rules.yaml").resolve()
        args = [
            "scan",
            "--config",
            str(rules_path),
            "--json",
            "--quiet",
            "--metrics",
            "off",
            "--disable-version-check",
            "--no-rewrite-rule-ids",
            "--error",
        ]
        if self.exclude_tests:
            args.extend(["--exclude", "tests", "--exclude", "test"])
        for d in sorted(SCAN_EXCLUDED_DIRS):
            args.extend(["--exclude", d])
        for f in _semgrep_file_excludes(skill_path):
            args.extend(["--exclude", f])
        args.append(str(skill_path.resolve()))

        tool_result = Tools.semgrep.run(args, timeout=180)

        if tool_result.error_message:
            result.add_error(tool_result.error_message)
            result.mark_scan_incomplete("semgrep")
        elif tool_result.exit_code in (0, 1):
            # Exit code 0 = clean, 1 = findings; both emit JSON on stdout
            result.record_scanner("semgrep", Tools.semgrep.version)
            self._process_semgrep_output(
                tool_result.stdout, result, exit_code=tool_result.exit_code, skill_path=skill_path.resolve()
            )
        else:
            error_msg = tool_result.stderr or tool_result.stdout or "no diagnostic output"
            result.add_error(f"Semgrep error (exit code {tool_result.exit_code}): {error_msg[:200]}")
            result.mark_scan_incomplete("semgrep")

        return result

    def _process_semgrep_output(
        self,
        output: str,
        result: ValidationResult,
        *,
        exit_code: int | None = None,
        skill_path: Path | None = None,
    ) -> None:
        """Parse Semgrep JSON output and report findings.

        *skill_path* is the resolved skill root; it lets parse-error paths be shown relative to it.
        """
        data = parse_json_output(output)
        if data is None:
            result.add_error("Semgrep did not return valid JSON; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return
        if not isinstance(data, dict):
            result.add_error("Semgrep JSON output was not an object; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return

        if "results" not in data:
            result.add_error("Semgrep JSON report is missing required 'results' list; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return
        findings = data["results"]
        if not isinstance(findings, list):
            result.add_error("Semgrep JSON field 'results' must be a list; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return
        if not all(isinstance(finding, dict) for finding in findings):
            result.add_error("Semgrep JSON 'results' entries must be objects; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return

        if "errors" not in data:
            result.add_error("Semgrep JSON report is missing required 'errors' list; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return
        scan_errors = data["errors"]
        if not isinstance(scan_errors, list):
            result.add_error("Semgrep JSON field 'errors' must be a list; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return
        if not all(isinstance(error, dict) for error in scan_errors):
            result.add_error("Semgrep JSON 'errors' entries must be objects; scan did not complete")
            result.mark_scan_incomplete("semgrep")
            return

        # File-level parse errors mean Semgrep ran but skipped part of the input; every
        # other error is reported one by one (limited to avoid noise).
        parse_errors: list[dict] = []
        other_errors: list[dict] = []
        for error in scan_errors:
            (parse_errors if _is_semgrep_parse_error(error) else other_errors).append(error)
        if parse_errors:
            self._report_semgrep_parse_errors(result, parse_errors, skill_path)
        for error in other_errors[:3]:
            message = str(error.get("message") or error.get("type") or "Unknown scan error")
            result.add_error(f"Semgrep scan error: {message[:100]}")
        if scan_errors:
            result.mark_scan_incomplete("semgrep")

        if exit_code == 1 and not findings:
            result.add_error("Semgrep findings exit code 1 did not include findings; scan did not complete")
            result.mark_scan_incomplete("semgrep")
        elif exit_code == 0 and findings:
            result.add_error("Semgrep returned findings with clean exit code 0; scanner status is inconsistent")
            result.mark_scan_incomplete("semgrep")

        if not findings:
            if not scan_errors and not result.is_incomplete:
                result.add_success(
                    check_name="semgrep",
                    message="Semgrep: No security issues found",
                )
            return

        result.add_message(f"Semgrep found {len(findings)} issue(s)")

        for finding in findings:
            self._report_semgrep_finding(result, finding)

    def _report_semgrep_parse_errors(
        self, result: ValidationResult, parse_errors: list[dict], skill_path: Path | None = None
    ) -> None:
        """Report files Semgrep could not fully parse as a partial scan.

        The message names up to ``_SEMGREP_PARSE_ERROR_PATH_LIMIT`` files with the line of
        their first error; ``semgrep_unparsed_files`` keeps every file path, relative to the skill.
        """
        first_line: dict[str, int | None] = {}
        for error in parse_errors:
            path = _semgrep_display_path(str(error.get("path") or "unknown"), skill_path)
            line = _semgrep_error_line(error)
            if first_line.get(path) is None:
                first_line[path] = line
        files = list(first_line)
        locations = [f"{path}:{line}" if line else path for path, line in first_line.items()]
        shown = locations[:_SEMGREP_PARSE_ERROR_PATH_LIMIT]
        extra = len(locations) - len(shown)
        listing = ", ".join(shown) + (f", and {extra} more" if extra > 0 else "")
        result.metadata["semgrep_unparsed_files"] = files
        result.add_error(
            f"Semgrep partial scan: {len(files)} file(s) could not be fully parsed ({listing}). "
            f"Semgrep skips the code it cannot parse, so findings there are missed. Findings for the "
            f"other files are kept. Fix the syntax error, or check whether it comes from Semgrep's parser "
            f"or the rule configuration for that language."
        )

    def _report_semgrep_finding(self, result: ValidationResult, finding: dict) -> None:
        """Report a single Semgrep finding."""
        rule_id = finding.get("check_id", "unknown")
        extra = finding.get("extra", {})

        severity_str = extra.get("severity", "WARNING").upper()
        severity = Severity.HIGH if severity_str == "ERROR" else Severity.MEDIUM

        message = extra.get("message", "Issue detected")[:100]
        path = finding.get("path", "unknown")
        line = finding.get("start", {}).get("line", 0)

        # Build reference string for CWE/OWASP
        metadata = extra.get("metadata", {})
        refs = self._build_reference_string(metadata)

        result.add_structured_finding(
            Finding(
                category="SEMGREP",
                severity=severity,
                check_name=rule_id,
                message=message,
                file_path=path,
                line_number=line,
                suggestion=refs.strip(" ()") if refs else None,
                metadata=metadata,
            ),
            is_error=severity == Severity.HIGH,
        )

    def _build_reference_string(self, metadata: dict) -> str:
        """Build CWE/OWASP reference string from metadata."""
        refs = []
        if cwe := metadata.get("cwe", []):
            refs.append(f"CWE: {', '.join(cwe[:2])}")
        if owasp := metadata.get("owasp", []):
            refs.append(f"OWASP: {', '.join(owasp[:2])}")
        return f" ({'; '.join(refs)})" if refs else ""
