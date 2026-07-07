# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unicode Smuggling Detection Validator.

Detects invisible Unicode characters that may indicate:
- ASCII smuggling via Unicode Tag characters (U+E0020-U+E007E)
- Trojan source attacks via BiDi overrides (CVE-2021-42574)
- Steganographic data hiding via zero-width characters
- Obfuscation via variation selectors and deprecated format controls

Character categories and thresholds are configurable via
skillevaluator/config/unicode_smuggle_patterns.yaml.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from skillevaluator.config import load_unicode_smuggle_patterns
from skillevaluator.constants import BINARY_CHECK_CHUNK_SIZE, UNICODE_SCAN_EXTENSIONS
from skillevaluator.logging_config import get_logger
from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.validators.base import ValidatorBase, iter_scannable_files

logger = get_logger(__name__)

# MIME types that are textual even when not under text/*
_TEXTUAL_MIME_TYPES = frozenset(
    {
        "application/json",
        "application/ld+json",
        "application/x-ndjson",
        "application/xml",
        "application/javascript",
        "application/x-javascript",
        "application/x-sh",
        "application/x-yaml",
        "application/yaml",
        "application/toml",
        "image/svg+xml",
    }
)


class UnicodeSmuggleValidator(ValidatorBase):
    """Detects invisible Unicode characters that may indicate ASCII smuggling,
    hidden data encoding, or trojan source attacks."""

    def __init__(self, include_spaces: bool = False) -> None:
        self._config: dict | None = None
        self.include_spaces = include_spaces
        self._char_lookup: dict[str, tuple[str, str]] | None = None
        self._tag_range: tuple[int, int] | None = None
        self._tag_ascii_range: tuple[int, int] | None = None
        self._vs_basic_range: tuple[int, int] | None = None
        self._vs_supplement_range: tuple[int, int] | None = None
        self._thresholds: dict | None = None

    @property
    def name(self) -> str:
        return "Unicode Smuggling Detection"

    @property
    def description(self) -> str:
        return "Detect invisible Unicode characters and ASCII smuggling"

    @property
    def config(self) -> dict:
        if self._config is None:
            self._config = load_unicode_smuggle_patterns()
            self._build_lookup_tables()
        return self._config

    def _build_lookup_tables(self) -> None:
        """Pre-compute lookup structures from YAML config for fast scanning."""
        cfg = self._config
        categories = cfg.get("categories", {})
        self._char_lookup = {}

        # Explicit character maps
        for cat_name, cat_def in categories.items():
            chars = cat_def.get("characters", {})
            for code_hex, char_name in chars.items():
                code_point = int(str(code_hex), 16) if isinstance(code_hex, str) else int(code_hex)
                self._char_lookup[chr(code_point)] = (cat_name, str(char_name))

        # Range-based categories
        tags = categories.get("unicode_tags", {})
        self._tag_range = (
            self._hex_to_int(tags.get("range_start", 0xE0000)),
            self._hex_to_int(tags.get("range_end", 0xE007F)),
        )
        self._tag_ascii_range = (
            self._hex_to_int(tags.get("ascii_decodable_start", 0xE0020)),
            self._hex_to_int(tags.get("ascii_decodable_end", 0xE007E)),
        )

        vs = categories.get("variation_selectors", {})
        self._vs_basic_range = (
            self._hex_to_int(vs.get("range_basic_start", 0xFE00)),
            self._hex_to_int(vs.get("range_basic_end", 0xFE0F)),
        )
        self._vs_supplement_range = (
            self._hex_to_int(vs.get("range_supplement_start", 0xE0100)),
            self._hex_to_int(vs.get("range_supplement_end", 0xE01EF)),
        )

        # Optional confusable spaces
        if self.include_spaces:
            opt = cfg.get("optional_categories", {}).get("confusable_spaces", {})
            for code_hex, char_name in opt.get("characters", {}).items():
                code_point = int(str(code_hex), 16) if isinstance(code_hex, str) else int(code_hex)
                ch = chr(code_point)
                if ch not in self._char_lookup:
                    self._char_lookup[ch] = ("confusable_spaces", str(char_name))

        self._thresholds = cfg.get("thresholds", {})

    @staticmethod
    def _hex_to_int(value: int | str) -> int:
        if isinstance(value, str):
            return int(value, 16)
        return int(value)

    def validate(self, skill_path: Path) -> ValidationResult:
        """Validate content for invisible Unicode characters."""
        return self._validate_folder_or_skill(
            skill_path,
            self._validate_single,
            action_description="Unicode scanning",
        )

    def _validate_single(self, skill_path: Path) -> ValidationResult:
        """Scan all scannable files in a single skill directory."""
        _ = self.config  # ensure lookup tables are built
        result = ValidationResult()
        files = self._get_scannable_files(skill_path)

        if not files:
            result.add_warning("No scannable files found for Unicode scan")
            return result

        result.summary.files_scanned = len(files)
        smuggle_found = False

        for file_path in files:
            try:
                relative_path = str(file_path.relative_to(skill_path))
            except ValueError:
                relative_path = file_path.name

            file_findings = self._scan_file(file_path, relative_path)
            for finding in file_findings:
                smuggle_found = True
                is_error = finding.severity in (Severity.CRITICAL, Severity.HIGH)
                result.add_structured_finding(finding, is_error=is_error)

        if not smuggle_found:
            result.add_success(
                check_name="unicode_scan",
                message=f"No invisible Unicode characters detected in {len(files)} file(s)",
                files_scanned=len(files),
            )

        return result

    def _scan_file(self, file_path: Path, relative_path: str) -> list[Finding]:
        """Scan a single file for invisible Unicode characters."""
        if self._is_binary(file_path):
            return []

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Could not read %s: %s", file_path, e)
            return []

        findings: list[Finding] = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            line_chars: list[dict] = []

            for col, char in enumerate(line):
                match = self._classify_char(char)
                if match is not None:
                    cat_name, char_name = match
                    line_chars.append(
                        {
                            "col": col,
                            "char": char,
                            "category": cat_name,
                            "name": char_name,
                        }
                    )

            if not line_chars:
                continue

            groups = self._group_consecutive(line_chars)
            for group in groups:
                finding = self._create_finding(
                    group,
                    line_num,
                    line,
                    relative_path,
                )
                if finding is not None:
                    findings.append(finding)

        return findings

    def _classify_char(self, char: str) -> tuple[str, str] | None:
        """Classify a character against the lookup tables.

        Returns (category_name, char_name) or None.
        """
        if char in self._char_lookup:
            return self._char_lookup[char]

        code = ord(char)

        # Unicode Tags range
        if self._tag_range[0] <= code <= self._tag_range[1]:
            return ("unicode_tags", "UNICODE TAG")

        # Variation Selectors basic
        if self._vs_basic_range[0] <= code <= self._vs_basic_range[1]:
            vs_num = code - self._vs_basic_range[0] + 1
            return ("variation_selectors", f"VARIATION SELECTOR-{vs_num}")

        # Variation Selectors supplement
        if self._vs_supplement_range[0] <= code <= self._vs_supplement_range[1]:
            vs_num = code - self._vs_supplement_range[0] + 17
            return ("variation_selectors", f"VARIATION SELECTOR-{vs_num}")

        return None

    @staticmethod
    def _group_consecutive(chars: list[dict]) -> list[list[dict]]:
        """Group consecutive invisible characters by column position."""
        if not chars:
            return []

        sorted_chars = sorted(chars, key=lambda x: x["col"])
        groups: list[list[dict]] = []
        current: list[dict] = [sorted_chars[0]]

        for i in range(1, len(sorted_chars)):
            if sorted_chars[i]["col"] == current[-1]["col"] + 1:
                current.append(sorted_chars[i])
            else:
                groups.append(current)
                current = [sorted_chars[i]]

        groups.append(current)
        return groups

    def _create_finding(
        self,
        group: list[dict],
        line_num: int,
        line: str,
        relative_path: str,
    ) -> Finding | None:
        """Create a Finding from a group of consecutive invisible characters."""
        run_length = len(group)
        categories_in_group = {c["category"] for c in group}
        primary_category = self._primary_category(categories_in_group)

        # BOM exception: U+FEFF at very start of file
        if run_length == 1 and group[0]["char"] == "\ufeff" and line_num == 1 and group[0]["col"] == 0:
            return Finding(
                category="UNICODE",
                severity=Severity.INFO,
                check_name="bom_marker",
                message="UTF-8 BOM marker at file start (benign)",
                file_path=relative_path,
                line_number=line_num,
                suggestion="BOM at file start is harmless. Remove if not needed.",
                metadata={"unicode_category": "zero_width", "char_count": 1},
            )

        # Unicode Tags that decode to ASCII -> CRITICAL
        if "unicode_tags" in categories_in_group:
            tag_chars = [c for c in group if c["category"] == "unicode_tags"]
            decoded = self._decode_tag_payload(tag_chars)
            if decoded:
                return Finding(
                    category="UNICODE",
                    severity=Severity.CRITICAL,
                    check_name="ascii_smuggling_payload",
                    message=f"Hidden ASCII text decoded from Unicode Tags: '{decoded}'",
                    file_path=relative_path,
                    line_number=line_num,
                    line_content=self._safe_line_content(line),
                    suggestion=(
                        "Remove Unicode Tag characters (U+E0020-U+E007E). "
                        "These encode hidden ASCII text invisible to reviewers."
                    ),
                    metadata={
                        "decoded_payload": decoded,
                        "char_count": len(tag_chars),
                        "consecutive_run": run_length,
                        "unicode_category": "unicode_tags",
                    },
                )

        return self._classify_by_run_and_category(
            group,
            run_length,
            primary_category,
            categories_in_group,
            line_num,
            line,
            relative_path,
        )

    def _classify_by_run_and_category(
        self,
        group: list[dict],
        run_length: int,
        primary_category: str,
        categories_in_group: set[str],
        line_num: int,
        line: str,
        relative_path: str,
    ) -> Finding:
        """Classify a finding by consecutive run length and character category."""
        thresholds = self._thresholds or {}
        crit_threshold = thresholds.get("consecutive_run_critical", 40)
        high_threshold = thresholds.get("consecutive_run_high", 10)
        per_line_medium = thresholds.get("per_line_medium", 3)

        if run_length >= crit_threshold:
            char_names = self._summarize_char_names(group)
            return Finding(
                category="UNICODE",
                severity=Severity.CRITICAL,
                check_name="long_invisible_run",
                message=f"Very long consecutive invisible run ({run_length} chars): {char_names}",
                file_path=relative_path,
                line_number=line_num,
                line_content=self._safe_line_content(line),
                suggestion="Long consecutive invisible character runs strongly indicate data hiding. Remove them.",
                metadata={
                    "char_count": run_length,
                    "consecutive_run": run_length,
                    "unicode_category": primary_category,
                },
            )

        # BiDi overrides -> HIGH
        if "bidi_marks" in categories_in_group:
            bidi_chars = [c for c in group if c["category"] == "bidi_marks"]
            char_names = ", ".join(c["name"] for c in bidi_chars[:3])
            if len(bidi_chars) > 3:
                char_names += f" (+{len(bidi_chars) - 3} more)"
            return Finding(
                category="UNICODE",
                severity=Severity.HIGH,
                check_name="bidi_override",
                message=f"BiDi override characters detected (trojan source risk, CVE-2021-42574): {char_names}",
                file_path=relative_path,
                line_number=line_num,
                line_content=self._safe_line_content(line),
                suggestion="Remove directional override characters that can disguise code execution flow.",
                metadata={
                    "char_count": len(bidi_chars),
                    "consecutive_run": run_length,
                    "unicode_category": "bidi_marks",
                },
            )

        if run_length >= high_threshold:
            char_names = self._summarize_char_names(group)
            return Finding(
                category="UNICODE",
                severity=Severity.HIGH,
                check_name="consecutive_invisible_run",
                message=f"Long consecutive invisible run ({run_length} chars): {char_names}",
                file_path=relative_path,
                line_number=line_num,
                line_content=self._safe_line_content(line),
                suggestion="Consecutive invisible character runs suggest intentional encoding. Investigate and remove.",
                metadata={
                    "char_count": run_length,
                    "consecutive_run": run_length,
                    "unicode_category": primary_category,
                },
            )

        # MEDIUM: zero-width chars 3+ per line, deprecated controls, invisible operators
        if run_length >= per_line_medium and primary_category in (
            "zero_width",
            "deprecated_format_controls",
            "invisible_operators",
        ):
            char_names = self._summarize_char_names(group)
            return Finding(
                category="UNICODE",
                severity=Severity.MEDIUM,
                check_name="suspicious_invisible_chars",
                message=f"Multiple invisible characters ({run_length}): {char_names}",
                file_path=relative_path,
                line_number=line_num,
                line_content=self._safe_line_content(line),
                suggestion="Multiple invisible characters in close proximity may indicate steganography.",
                metadata={
                    "char_count": run_length,
                    "consecutive_run": run_length,
                    "unicode_category": primary_category,
                },
            )

        # Deprecated format controls and invisible operators are always at least MEDIUM
        if primary_category in ("deprecated_format_controls", "invisible_operators"):
            char_names = self._summarize_char_names(group)
            return Finding(
                category="UNICODE",
                severity=Severity.MEDIUM,
                check_name="deprecated_format_control",
                message=f"Deprecated/invisible format characters: {char_names}",
                file_path=relative_path,
                line_number=line_num,
                line_content=self._safe_line_content(line),
                suggestion="Deprecated format controls have no legitimate use in modern content. Remove them.",
                metadata={
                    "char_count": run_length,
                    "unicode_category": primary_category,
                },
            )

        # LOW: isolated zero-width, confusable spaces, variation selectors
        char_names = self._summarize_char_names(group)
        return Finding(
            category="UNICODE",
            severity=Severity.LOW,
            check_name="isolated_invisible_char",
            message=f"Isolated invisible character(s) ({run_length}): {char_names}",
            file_path=relative_path,
            line_number=line_num,
            line_content=self._safe_line_content(line),
            suggestion="Likely a copy-paste artifact. Remove if not intentional.",
            metadata={
                "char_count": run_length,
                "unicode_category": primary_category,
            },
        )

    def _decode_tag_payload(self, tag_chars: list[dict]) -> str | None:
        """Decode Unicode Tag characters to their ASCII equivalents."""
        if not tag_chars:
            return None

        decoded_parts: list[str] = []
        ascii_start = self._tag_ascii_range[0]
        ascii_end = self._tag_ascii_range[1]

        for tc in tag_chars:
            code = ord(tc["char"])
            if ascii_start <= code <= ascii_end:
                decoded_parts.append(chr(code - 0xE0000))
            elif code == 0xE0001:
                pass  # TAG_START marker, skip
            elif code == 0xE007F:
                pass  # TAG_END marker, skip

        payload = "".join(decoded_parts).strip()
        return payload or None

    @staticmethod
    def _primary_category(categories: set[str]) -> str:
        """Determine the primary (highest-risk) category from a set."""
        priority = [
            "unicode_tags",
            "bidi_marks",
            "deprecated_format_controls",
            "invisible_operators",
            "zero_width",
            "confusable_spaces",
            "variation_selectors",
        ]
        for cat in priority:
            if cat in categories:
                return cat
        return next(iter(categories)) if categories else "unknown"

    @staticmethod
    def _summarize_char_names(group: list[dict]) -> str:
        """Build a compact summary of character names in a group."""
        names = [c["name"] for c in group[:4]]
        summary = ", ".join(names)
        if len(group) > 4:
            summary += f" (+{len(group) - 4} more)"
        return summary

    @staticmethod
    def _safe_line_content(line: str) -> str:
        """Truncate line content for display, stripping invisible chars."""
        visible = "".join(c if c.isprintable() or c in (" ", "\t") else "\u2423" for c in line)
        stripped = visible.strip()
        if len(stripped) > 120:
            return stripped[:117] + "..."
        return stripped

    @staticmethod
    def _is_binary(file_path: Path) -> bool:
        """Check if a file is likely binary using MIME type and null-byte detection."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if (
            mime_type
            and not mime_type.startswith("text/")
            and mime_type not in _TEXTUAL_MIME_TYPES
            and not mime_type.endswith("+json")
            and not mime_type.endswith("+xml")
        ):
            return True

        try:
            with file_path.open("rb") as f:
                chunk = f.read(BINARY_CHECK_CHUNK_SIZE)
                if b"\x00" in chunk:
                    return True
        except OSError:
            return True

        return False

    @staticmethod
    def _get_scannable_files(skill_path: Path) -> list[Path]:
        """Collect files with scannable extensions from path.

        Skips Tier 1 artifact directories (``evals/``, ``results/``,
        ``versions/`` and the dot-prefixed variants) at any depth so that
        agent transcripts under ``evals/results/`` do not produce false
        invisible-Unicode findings.
        """
        return iter_scannable_files(skill_path, UNICODE_SCAN_EXTENSIONS)
