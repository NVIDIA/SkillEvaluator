# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from skillevaluator.quality_config import (
    QualityConfigError,
    load_quality_config,
    load_quality_config_with_provenance,
    quality_config_digest,
)


def test_packaged_default_quality_config_is_available() -> None:
    config = load_quality_config()

    assert config["expected_headers"] == [
        {
            "heading": "## Prerequisites",
            "alternatives": ["## Requirements"],
            "points": 5,
            "message": "No prerequisites/requirements documented",
            "suggestion": "Document dependencies, API keys, or setup needed",
        },
        {
            "heading": "## Limitations",
            "alternatives": [],
            "points": 5,
            "message": "No limitations documented",
            "suggestion": "Add '## Limitations' section with known issues/constraints",
        },
        {
            "heading": "## Troubleshooting",
            "alternatives": ["## Common Issues", "## FAQ"],
            "points": 5,
            "message": "No troubleshooting section documented",
            "suggestion": "Add '## Troubleshooting' with Error/Cause/Solution patterns",
        },
    ]
    # Upstream deducted the int 5, so reports keep "deduction": 5 rather than 5.0.
    assert all(type(rule["points"]) is int for rule in config["expected_headers"])


def test_quality_config_can_be_overridden(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(
        'expected_headers:\n  - heading: "## Failure Constellations"\n'
        '    alternatives:\n      - "## Known Failures"\n'
        "    message: No failure constellations documented\n"
        "    suggestion: Add '## Failure Constellations'\n"
        '  - heading: "### Security"\n    points: 8.5\n'
        "    message: No security notes documented\n"
        "    suggestion: Add '### Security'\n",
        encoding="utf-8",
    )

    config = load_quality_config(config_path)

    assert config["expected_headers"] == [
        {
            "heading": "## Failure Constellations",
            "alternatives": ["## Known Failures"],
            "points": 5,
            "message": "No failure constellations documented",
            "suggestion": "Add '## Failure Constellations'",
        },
        {
            "heading": "### Security",
            "alternatives": [],
            "points": 8.5,
            "message": "No security notes documented",
            "suggestion": "Add '### Security'",
        },
    ]


@pytest.mark.parametrize(
    "config",
    [
        "expected_headers: not-a-list\n",
        "expected_headers:\n  - heading: Limitations\n",
        'expected_headers:\n  - heading: "##   "\n',
        'expected_headers:\n  - heading: "## Limitations"\n    alternatives: not-a-list\n',
        'expected_headers:\n  - heading: "## Limitations"\n    points: -1\n',
        'expected_headers:\n  - heading: "## Limitations"\n    message: ""\n',
        'expected_headers:\n  - heading: "## Limitations"\n    suggestion:\n',
        'expected_headers:\n  - heading: "## Limitations"\n    message: [Missing]\n',
    ],
)
def test_quality_config_rejects_invalid_expected_headers(tmp_path: Path, config: str) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(QualityConfigError, match="expected_headers"):
        load_quality_config(config_path)


def test_quality_config_generates_message_and_suggestion_when_omitted(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(
        'expected_headers:\n  - heading: "## Security"\n'
        '  - heading: "## Prerequisites"\n    alternatives:\n      - "## Requirements"\n    points: 5\n'
        '  - heading: "## Limitations"\n    message: No limitations documented\n',
        encoding="utf-8",
    )

    config = load_quality_config(config_path)

    assert [(rule["message"], rule["suggestion"]) for rule in config["expected_headers"]] == [
        ("Missing expected heading: '## Security'", "Add a '## Security' section"),
        (
            "Missing expected heading: '## Prerequisites' or '## Requirements'",
            "Add one of these sections: '## Prerequisites' or '## Requirements'",
        ),
        ("No limitations documented", "Add a '## Limitations' section"),
    ]


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            "expected_headers: []\nexpected_header: []\n",
            r"unknown key\(s\) in top level: expected_header",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    point: 10\n'
            "    message: Missing\n    suggestion: Add it\n",
            r"unknown key\(s\) in expected_headers\[0\]: point",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    message: Missing\n    suggestion: Add it\n'
            '  - heading: "## Limits"\n    alternatives:\n      - "##  Limitations"\n'
            "    message: Missing\n    suggestion: Add it\n",
            r"duplicate heading '## Limitations' in expected_headers\[1\] \(already listed in expected_headers\[0\]\)",
        ),
    ],
)
def test_quality_config_rejects_unknown_keys_and_duplicate_headings(
    tmp_path: Path, config: str, match: str
) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(QualityConfigError, match=match):
        load_quality_config(config_path)


def test_quality_config_errors_name_the_file_and_yaml_position(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text("expected_headers: [\n", encoding="utf-8")

    with pytest.raises(QualityConfigError) as exc_info:
        load_quality_config(config_path)

    message = str(exc_info.value)
    assert message.startswith(f"{config_path}: invalid YAML: line 2, column 1: ")
    assert "\n" not in message


def test_quality_config_rejects_non_utf8_file(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text("expected_headers: []\n", encoding="utf-16")

    with pytest.raises(QualityConfigError, match="not valid UTF-8"):
        load_quality_config(config_path)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            'expected_headers:\n  - heading: "## Prerequisites"\nexpected_headers:\n  - heading: "## Security"\n',
            r"invalid YAML: line 3, column 1: duplicate key 'expected_headers'",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    points: 5\n    points: 10\n',
            r"invalid YAML: line 4, column 5: duplicate key 'points'",
        ),
        ("expected_headers: " + "[" * 10000 + "]" * 10000 + "\n", "invalid YAML: nesting is too deep"),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    points: ' + "9" * 400 + "\n",
            r"expected_headers\[0\]\.points must be a non-negative number",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    points: ' + "9" * 5000 + "\n",
            r"invalid YAML: cannot parse a value: Exceeds the limit",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    message: 2026-02-30\n',
            r"invalid YAML: cannot parse a value: day (?:is out of range|\d+ must be in range)",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    points: !!int abc\n',
            r"invalid YAML: cannot parse a value: invalid literal for int\(\)",
        ),
    ],
    ids=[
        "duplicate-top-level-key",
        "duplicate-rule-key",
        "deep-nesting",
        "points-overflow",
        "integer-digit-limit",
        "impossible-date",
        "bad-explicit-tag",
    ],
)
def test_quality_config_rejects_duplicate_keys_and_pathological_values(
    tmp_path: Path, config: str, match: str
) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(QualityConfigError, match=match):
        load_quality_config(config_path)


def test_quality_config_digest_ignores_formatting_but_not_rules(tmp_path: Path) -> None:
    compact = tmp_path / "compact.yaml"
    compact.write_text('expected_headers: [{heading: "## Limitations", points: 5}]\n', encoding="utf-8")
    verbose = tmp_path / "verbose.yaml"
    verbose.write_text(
        '# team rules\nexpected_headers:\n  - heading:   "##   Limitations"\n    points: 5\n',
        encoding="utf-8",
    )
    changed = tmp_path / "changed.yaml"
    changed.write_text('expected_headers:\n  - heading: "## Limitations"\n    points: 6\n', encoding="utf-8")

    digest = quality_config_digest(load_quality_config(compact))

    assert digest.startswith("sha256:") and len(digest) == 71
    assert quality_config_digest(load_quality_config(verbose)) == digest
    assert quality_config_digest(load_quality_config(changed)) != digest


def test_quality_config_provenance_names_custom_file_without_its_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "team-quality.yaml"
    config_path.write_text("expected_headers: []\n", encoding="utf-8")

    assert load_quality_config_with_provenance(config_path)[1] == {
        "source": "custom",
        "file": "team-quality.yaml",
        "digest": quality_config_digest({"expected_headers": []}),
    }
    assert load_quality_config_with_provenance(None)[1]["file"] is None



PACKAGED_ERROR_HANDLING = {
    "mentioned": {
        "points": 10,
        "message": "No mention of error handling or validation",
        "suggestion": "Document expected errors and how to handle them",
        "terms": [
            "error",
            "errors",
            "exception",
            "exceptions",
            "invalid",
            "fail",
            "fails",
            "failed",
            "failure",
            "failures",
            "failing",
            "validate",
            "validates",
            "validated",
            "validating",
            "validation",
            "validations",
        ],
    },
    "scripts": {
        "points": 5,
        "message": "Scripts may lack error handling",
        "suggestion": "Scripts should handle errors explicitly",
        "markers": ["if not ", "if error", "raise", "assert", "ValueError", "FileNotFoundError"],
    },
    "lib": {
        "points": 5,
        "message": "Lib modules may lack error handling",
        "suggestion": "Library code should handle errors explicitly with try/except or raise",
        "markers": ["raise", "assert", "ValueError", "TypeError", "RuntimeError"],
    },
}


def _write_config(tmp_path: Path, body: str) -> Path:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text("expected_headers: []\n" + body, encoding="utf-8")
    return config_path


def test_packaged_error_handling_rules_match_former_hard_coded_values() -> None:
    error_handling = load_quality_config()["error_handling"]

    assert error_handling == PACKAGED_ERROR_HANDLING
    assert all(type(block["points"]) is int for block in error_handling.values())


def test_custom_file_without_error_handling_keeps_packaged_rules(tmp_path: Path) -> None:
    config = load_quality_config(_write_config(tmp_path, ""))

    assert config["expected_headers"] == []
    assert config["error_handling"] == PACKAGED_ERROR_HANDLING


def test_custom_error_handling_overlays_packaged_rules_key_by_key(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "error_handling:\n  scripts:\n    points: 2\n  mentioned:\n    terms: [oops]\n")

    error_handling = load_quality_config(config_path)["error_handling"]

    assert error_handling["scripts"] == {**PACKAGED_ERROR_HANDLING["scripts"], "points": 2}
    assert error_handling["mentioned"] == {**PACKAGED_ERROR_HANDLING["mentioned"], "terms": ["oops"]}
    assert error_handling["lib"] == PACKAGED_ERROR_HANDLING["lib"]


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("error_handling: []\n", r"error_handling must be a mapping"),
        ("error_handling:\n  retries: {}\n", r"unknown key\(s\) in error_handling: retries"),
        (
            "error_handling:\n  mentioned:\n    point: 3\n",
            r"unknown key\(s\) in error_handling\.mentioned: point \(allowed: message, points, suggestion, terms\)",
        ),
        ("error_handling:\n  scripts:\n    terms: [raise]\n", r"unknown key\(s\) in error_handling\.scripts: terms"),
        ("error_handling:\n  lib: [raise]\n", r"error_handling\.lib must be a mapping"),
        ("error_handling:\n  scripts:\n    points: -1\n", r"error_handling\.scripts\.points must be a non-negative"),
        ("error_handling:\n  scripts:\n    points: true\n", r"error_handling\.scripts\.points must be a non-negative"),
        ('error_handling:\n  scripts:\n    points: "5"\n', r"error_handling\.scripts\.points must be a non-negative"),
        ('error_handling:\n  lib:\n    message: " "\n', r"error_handling\.lib\.message must be a non-empty string"),
        ("error_handling:\n  lib:\n    suggestion: 3\n", r"error_handling\.lib\.suggestion must be a non-empty string"),
        ("error_handling:\n  scripts:\n    markers: []\n", r"error_handling\.scripts\.markers must be a non-empty list"),
        ("error_handling:\n  scripts:\n    markers: raise\n", r"error_handling\.scripts\.markers must be a non-empty list"),
        ('error_handling:\n  scripts:\n    markers: [""]\n', r"error_handling\.scripts\.markers\[0\] must be a non-blank"),
        ("error_handling:\n  lib:\n    markers: [raise, 3]\n", r"error_handling\.lib\.markers\[1\] must be a non-blank"),
        ("error_handling:\n  scripts:\n    markers: [raise, raise]\n", r"duplicate entry 'raise' in error_handling\.scripts\.markers"),
        ("error_handling:\n  mentioned:\n    terms: [Error, error]\n", r"duplicate entry 'error' in error_handling\.mentioned\.terms"),
    ],
)
def test_quality_config_rejects_invalid_error_handling(tmp_path: Path, body: str, match: str) -> None:
    config_path = _write_config(tmp_path, body)

    with pytest.raises(QualityConfigError, match=rf"^{tmp_path}.*{match}"):
        load_quality_config(config_path)


def test_error_handling_markers_are_case_sensitive_so_case_variants_are_distinct(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "error_handling:\n  lib:\n    markers: [Raise, raise]\n")

    assert load_quality_config(config_path)["error_handling"]["lib"]["markers"] == ["Raise", "raise"]


def test_quality_config_digest_covers_error_handling(tmp_path: Path) -> None:
    compact = tmp_path / "compact.yaml"
    compact.write_text("expected_headers: []\nerror_handling: {mentioned: {points: 9}}\n", encoding="utf-8")
    verbose = tmp_path / "verbose.yaml"
    verbose.write_text(
        "# team rules\nexpected_headers: []\nerror_handling:\n  mentioned:\n    points:   9\n", encoding="utf-8"
    )
    packaged_headings_only = _write_config(tmp_path, "")

    digest = _digest(compact)

    assert _digest(verbose) == digest
    assert _digest(packaged_headings_only) != digest


def _digest(path: Path | None) -> str:
    return load_quality_config_with_provenance(path)[1]["digest"]


def test_packaged_digest_covers_the_whole_packaged_config() -> None:
    assert _digest(None) == quality_config_digest(load_quality_config())


def test_custom_digest_covers_only_the_rules_the_file_sets(tmp_path: Path) -> None:
    def write(name: str, body: str) -> Path:
        path = tmp_path / name
        path.write_text("expected_headers: []\n" + body, encoding="utf-8")
        return path

    headings_only = write("headings-only.yaml", "")
    empty_section = write("empty-section.yaml", "error_handling: {mentioned: {}}\n")
    one_key = write("one-key.yaml", "error_handling: {mentioned: {points: 9}}\n")

    assert _digest(headings_only) == quality_config_digest({"expected_headers": []})
    assert _digest(empty_section) == _digest(headings_only)
    assert _digest(one_key) == quality_config_digest(
        {"expected_headers": [], "error_handling": {"mentioned": {"points": 9}}}
    )


def test_custom_digest_ignores_packaged_defaults_the_file_omits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator import quality_config

    custom = _write_config(tmp_path, "error_handling: {scripts: {points: 2}}\n")
    before_config, before = load_quality_config_with_provenance(custom)
    changed_default = tmp_path / "changed-default.yaml"
    changed_default.write_text(
        quality_config.DEFAULT_QUALITY_CONFIG.read_text(encoding="utf-8").replace("points: 10", "points: 11"),
        encoding="utf-8",
    )
    monkeypatch.setattr(quality_config, "DEFAULT_QUALITY_CONFIG", changed_default)

    after_config, after = load_quality_config_with_provenance(custom)

    assert after_config["error_handling"]["mentioned"]["points"] == 11  # the new default still applies
    assert before_config["error_handling"]["mentioned"]["points"] == 10
    assert after["digest"] == before["digest"]


def test_custom_file_without_expected_headers_keeps_the_packaged_list(tmp_path: Path) -> None:
    custom = tmp_path / "quality.yaml"
    custom.write_text("error_handling: {mentioned: {points: 9}}\n", encoding="utf-8")

    config = load_quality_config(custom)

    assert config["expected_headers"] == load_quality_config()["expected_headers"]
    assert len(config["expected_headers"]) == 3
    assert config["error_handling"]["mentioned"]["points"] == 9


def test_explicit_expected_headers_replaces_or_empties_the_packaged_list(tmp_path: Path) -> None:
    custom = tmp_path / "quality.yaml"
    custom.write_text("expected_headers: []\n", encoding="utf-8")
    assert load_quality_config(custom)["expected_headers"] == []

    custom.write_text('expected_headers:\n  - heading: "## Examples"\n', encoding="utf-8")
    assert [rule["heading"] for rule in load_quality_config(custom)["expected_headers"]] == ["## Examples"]


@pytest.mark.parametrize("body", ["expected_headers:\n", "expected_headers: {}\n", "expected_headers: x\n"])
def test_present_expected_headers_must_still_be_a_list(tmp_path: Path, body: str) -> None:
    custom = tmp_path / "quality.yaml"
    custom.write_text(body, encoding="utf-8")

    with pytest.raises(QualityConfigError, match="expected_headers must be a list"):
        load_quality_config(custom)


def test_packaged_file_must_still_declare_expected_headers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillevaluator import quality_config

    broken = tmp_path / "broken-default.yaml"
    broken.write_text("error_handling: {}\n", encoding="utf-8")
    monkeypatch.setattr(quality_config, "DEFAULT_QUALITY_CONFIG", broken)

    with pytest.raises(QualityConfigError, match="expected_headers must be a list"):
        load_quality_config()


def test_digest_does_not_declare_packaged_headings_when_the_key_is_omitted(tmp_path: Path) -> None:
    omitted = tmp_path / "omitted.yaml"
    omitted.write_text("error_handling: {mentioned: {points: 9}}\n", encoding="utf-8")
    empty = tmp_path / "empty.yaml"
    empty.write_text("expected_headers: []\nerror_handling: {mentioned: {points: 9}}\n", encoding="utf-8")
    bare = tmp_path / "bare.yaml"
    bare.write_text("{}\n", encoding="utf-8")

    assert _digest(omitted) == quality_config_digest({"error_handling": {"mentioned": {"points": 9}}})
    assert _digest(omitted) != _digest(empty)
    assert _digest(bare) == quality_config_digest({})
    assert _digest(bare) != _digest(None)
