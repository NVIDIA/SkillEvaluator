# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest content problems (encoding, size) versus unsafe manifest paths.

Only link, special-file, identity-change, and containment problems fail closed
as security failures. A root ``plugin.json`` that is not UTF-8 or is oversize
decides the Agent Plugins opt-in instead of failing discovery, and a client
manifest with such content, selected or additional, is a HIGH finding (clients
without SkillEvaluator's limits still load it) that is not a security failure,
unless it is the selected manifest and nothing could be parsed from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillevaluator.constants import (
    CONTENT_DEDUP_MAX_FILE_BYTES,
    CONTENT_DEDUP_MAX_TOTAL_BYTES,
    CONTENT_TYPE_PLUGIN,
    PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE,
    PLUGIN_CONTAINED_MANIFEST_TYPE,
    PLUGIN_MANIFEST_TYPE,
)
from skillevaluator.models.result import Severity, ValidationResult
from skillevaluator.plugin_manifest import PluginManifestLocation, PluginManifestPathError, locate_plugin_manifest
from skillevaluator.tier1.commands import run_validation
from skillevaluator.validators import plugin_schema
from skillevaluator.validators.plugin_schema import PluginSchemaValidator
from skillevaluator.validators.policy import ValidationPolicy, apply_policy

_AP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
_OVERSIZE = "x" * (CONTENT_DEDUP_MAX_FILE_BYTES + 16)


def _write(root: Path, files: dict[str, bytes | dict]) -> Path:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else json.dumps(content).encode())
    return root


def _checks(result: ValidationResult) -> dict[str, Severity]:
    return {finding.check_name: finding.severity for finding in result.findings}


def _legacy(encoding: str) -> bytes:
    if encoding == "oversize":
        return json.dumps({"name": "legacy-copilot", "blob": _OVERSIZE}).encode()
    return json.dumps({"name": "legacy-caf\xe9"}, ensure_ascii=False).encode(encoding)


def _agent_plugins(encoding: str) -> bytes:
    return json.dumps({"$schema": _AP_SCHEMA, "name": "caf\xe9"}, ensure_ascii=False).encode(encoding)


# --------------------------------------------------------------------------- #
# Root plugin.json opt-in (plugin_manifest._is_agent_plugins_manifest)         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("encoding", ["utf-16", "latin-1", "oversize"])
def test_legacy_root_plugin_json_content_does_not_fail_a_claude_plugin(tmp_path: Path, encoding: str) -> None:
    """Regression: a legacy root plugin.json that is not UTF-8 or is oversize failed HIGH manifest_outside_root."""
    root = _write(tmp_path / "p", {".claude-plugin/plugin.json": {"name": "demo"}, "plugin.json": _legacy(encoding)})

    located = locate_plugin_manifest(root)
    assert located is not None
    assert located.manifest_type == PLUGIN_CONTAINED_MANIFEST_TYPE
    assert located.additional == ()
    result = PluginSchemaValidator().validate(root)
    assert _checks(result) == {}
    assert result.passed
    assert "security_failure" not in result.metadata


@pytest.mark.parametrize(
    ("manifest", "checks"),
    [
        # Not UTF-8: a decode error, read leniently, not a security failure (proof L6). The lenient read
        # still checks the fields: 'café' is not a valid name.
        (_agent_plugins("utf-16"), {"manifest_unreadable", "schema:name:pattern"}),
        (_agent_plugins("latin-1"), {"manifest_unreadable", "schema:name:pattern"}),
        # ASCII-only UTF-16 without a BOM is valid UTF-8 (with NULs), so it decodes and then fails as JSON.
        (json.dumps({"$schema": _AP_SCHEMA, "name": "demo"}).encode("utf-16-be"), {"manifest_invalid_json"}),
    ],
    ids=["utf-16", "latin-1", "utf-16-be-ascii"],
)
def test_sole_root_manifest_declaring_the_schema_reports_its_encoding(
    tmp_path: Path, manifest: bytes, checks: set[str]
) -> None:
    """A non-UTF-8 root plugin.json whose bytes name the schema is still selected, and its problem reported."""
    root = _write(tmp_path / "p", {"plugin.json": manifest})

    located = locate_plugin_manifest(root)
    assert located is not None
    assert located.manifest_type == PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE
    result = PluginSchemaValidator().validate(root)
    assert not result.passed
    assert _checks(result) == dict.fromkeys(checks, Severity.HIGH)
    assert "security_failure" not in result.metadata


def test_directly_named_root_manifest_reports_its_encoding(tmp_path: Path) -> None:
    root = _write(tmp_path / "p", {"plugin.json": _legacy("utf-16")})

    located = locate_plugin_manifest(root / "plugin.json")
    assert located is not None
    assert located.manifest_type == PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE
    with pytest.raises(PluginManifestPathError) as raised:
        located.read_text(encoding="utf-8-sig")
    assert raised.value.reason == "encoding"
    assert raised.value.content_error
    result = PluginSchemaValidator().validate(root / "plugin.json")
    checks = _checks(result)
    assert checks.pop("manifest_unreadable") == Severity.HIGH
    # Read leniently, the legacy manifest then fails the Agent Plugins fields.
    assert checks == {"schema:$schema:missing": Severity.HIGH, "schema:name:pattern": Severity.HIGH}
    assert "is not valid UTF-8 text" in result.findings[0].message


def test_root_manifest_escaping_the_schema_in_latin1_is_still_agent_plugins(tmp_path: Path) -> None:
    """Invalid UTF-8 bytes are replaced before parsing, as lenient clients do, so JSON escapes cannot hide the opt-in."""
    manifest = b'{"$schema": "https:\\/\\/agent-plugins.org\\/schemas\\/1.0.0\\/plugin.schema.json", "name": "d\xe9mo"}'
    root = _write(tmp_path / "p", {".claude-plugin/plugin.json": {"name": "demo"}, "plugin.json": manifest})

    located = locate_plugin_manifest(root)
    assert located is not None
    assert [candidate.manifest_type for candidate in located.additional] == [PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE]


def test_root_manifest_that_changes_after_discovery_still_fails_closed(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "p", {".claude-plugin/plugin.json": {"name": "demo"}, "plugin.json": _agent_plugins("utf-8")}
    )
    located = locate_plugin_manifest(root)
    assert located is not None
    [candidate] = located.additional
    (root / "plugin.json").write_bytes(_agent_plugins("utf-8") + b"\n\n")

    with pytest.raises(PluginManifestPathError) as raised:
        candidate.read_text()
    assert raised.value.reason == "unsafe"
    assert not raised.value.content_error


# --------------------------------------------------------------------------- #
# Additional manifests (PluginSchemaValidator._parse_additional)              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("relative", "content", "expected"),
    [
        (
            ".cursor-plugin/plugin.json",
            json.dumps({"name": "demo", "description": "caf\xe9"}, ensure_ascii=False).encode("latin-1"),
            "is not valid UTF-8 text",
        ),
        (
            ".cursor-plugin/plugin.json",
            json.dumps({"name": "demo", "description": _OVERSIZE}).encode(),
            f"is larger than the {CONTENT_DEDUP_MAX_FILE_BYTES}-byte manifest read limit",
        ),
        ("plugin.json", _agent_plugins("utf-16"), "is not valid UTF-8 text"),
    ],
    ids=["cursor-latin-1", "cursor-oversize", "agent-plugins-utf-16"],
)
def test_additional_manifest_content_problem_is_high_but_not_a_security_failure(
    tmp_path: Path, relative: str, content: bytes, expected: str
) -> None:
    """A client manifest that is oversize or not UTF-8 fails HIGH, because clients without the limit still load it.

    Regression (1): a decode error in an additional manifest was HIGH manifest_unsafe with security_failure.
    Regression (2): the same content problem was only MEDIUM and the manifest's components were never checked.
    """
    root = _write(tmp_path / "p", {".claude-plugin/plugin.json": {"name": "demo"}, relative: content})

    result = PluginSchemaValidator().validate(root)
    checks = _checks(result)
    assert checks.pop("plugin_manifest_additional_unreadable") == Severity.HIGH
    # It is read leniently, so what it declares is checked: the UTF-16 manifest names another plugin.
    assert checks == ({"plugin_manifest_conflict": Severity.MEDIUM} if relative == "plugin.json" else {})
    [finding] = [f for f in result.findings if f.check_name == "plugin_manifest_additional_unreadable"]
    assert expected in finding.message
    # Only the components of an additional manifest are checked, not its fields.
    assert "its fields" not in finding.message
    assert "security_failure" not in result.metadata
    assert not result.passed
    rows = result.metadata["plugin"]["manifest_declarations"]["manifests"]
    assert [(row["manifest_filename"], row["status"]) for row in rows] == [
        (".claude-plugin/plugin.json", "selected"),
        (relative, "unreadable"),
    ]


def test_additional_manifest_decode_error_does_not_stop_later_checks(tmp_path: Path) -> None:
    root = _write(
        tmp_path / "p",
        {
            ".claude-plugin/plugin.json": {"name": "demo"},
            ".cursor-plugin/plugin.json": json.dumps({"name": "caf\xe9"}, ensure_ascii=False).encode("latin-1"),
        },
    )

    results = run_validation(root, checks="schema,unicode", content_type=CONTENT_TYPE_PLUGIN)
    assert [result.validator_name for result in results] == [
        "Plugin Schema & Bundle References",
        "Unicode Smuggling Detection",
    ]


def test_additional_manifest_that_changes_after_discovery_stays_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write(
        tmp_path / "p", {".claude-plugin/plugin.json": {"name": "demo"}, ".cursor-plugin/plugin.json": {"name": "demo"}}
    )
    real_locate = plugin_schema.locate_plugin_manifest

    def locate_then_swap(path: Path):
        located = real_locate(path)
        (root / ".cursor-plugin" / "plugin.json").write_text(json.dumps({"name": "swapped-after-discovery"}))
        return located

    monkeypatch.setattr(plugin_schema, "locate_plugin_manifest", locate_then_swap)
    result = PluginSchemaValidator().validate(root)
    assert _checks(result) == {"manifest_unsafe": Severity.HIGH}
    assert result.metadata["security_failure"] is True
    assert "swapped-after-discovery" not in json.dumps(result.metadata["plugin"]["manifest_declarations"])


# --------------------------------------------------------------------------- #
# Selected manifest (PluginSchemaValidator._load_manifest)                     #
# --------------------------------------------------------------------------- #
_LATIN1_CLAUDE = json.dumps(
    {"name": "demo", "description": "caf\xe9", "mcpServers": {"evil": {"type": "http", "url": "http://mcp.invalid/"}}},
    ensure_ascii=False,
).encode("latin-1")


def test_non_utf8_selected_manifest_is_unreadable_and_its_server_is_still_checked(tmp_path: Path) -> None:
    """Regression: a Latin-1 selected manifest was 'unsafe links' (a security failure) and never inventoried."""
    root = _write(tmp_path / "p", {".claude-plugin/plugin.json": _LATIN1_CLAUDE})

    result = PluginSchemaValidator().validate(root)
    assert _checks(result) == {"manifest_unreadable": Severity.HIGH, "mcp_url_insecure_scheme": Severity.HIGH}
    assert "security_failure" not in result.metadata
    assert not result.passed
    [finding] = [f for f in result.findings if f.check_name == "manifest_unreadable"]
    assert "is not valid UTF-8 text" in finding.message
    assert "read leniently" in finding.message
    components = result.metadata["plugin"]["component_inventory"]["components"]
    assert [row["name"] for row in components if row["type"] == "mcp"] == ["evil"]
    assert "plugin_manifest" not in [detail.check_name for detail in result.success_details]


def test_unreadable_selected_manifest_fails_tier1_without_stopping_later_checks(tmp_path: Path) -> None:
    root = _write(tmp_path / "p", {".claude-plugin/plugin.json": _LATIN1_CLAUDE})

    results = run_validation(root, checks="schema,unicode", content_type=CONTENT_TYPE_PLUGIN)
    assert [result.validator_name for result in results] == [
        "Plugin Schema & Bundle References",
        "Unicode Smuggling Detection",
    ]
    assert not results[0].passed


_LATIN1_BUNDLE = "name: caf\xe9\nauthor:\n  email: a@example.com\n".encode("latin-1")


def test_non_utf8_bundle_manifest_is_unreadable_not_unsafe(tmp_path: Path) -> None:
    root = _write(tmp_path / "p", {"agent_plugin.yaml": _LATIN1_BUNDLE})

    result = PluginSchemaValidator().validate(root)
    assert _checks(result) == {"manifest_unreadable": Severity.HIGH}
    # agent_plugin.yaml is not read leniently, so nothing it declares was checked.
    assert result.metadata["security_failure"] is True
    assert not result.passed


_UNREADABLE_SELECTED = {
    # Over the lenient bound: not even a lenient read parses it.
    "claude-over-lenient-bound": {
        ".claude-plugin/plugin.json": b'{"name": "demo"' + b" " * CONTENT_DEDUP_MAX_TOTAL_BYTES + b"}"
    },
    "agent-plugins-over-lenient-bound": {
        "plugin.json": b'{"name": "demo"' + b" " * CONTENT_DEDUP_MAX_TOTAL_BYTES + b"}"
    },
    "bundle-latin-1": {"agent_plugin.yaml": _LATIN1_BUNDLE},
}


@pytest.mark.parametrize("files", list(_UNREADABLE_SELECTED.values()), ids=list(_UNREADABLE_SELECTED))
def test_policy_cannot_pass_a_selected_manifest_that_nothing_was_parsed_from(
    tmp_path: Path, files: dict[str, bytes]
) -> None:
    """Regression: a policy override downgraded manifest_unreadable, and no declared component had been checked."""
    root = _write(tmp_path / "p", files)
    policy = ValidationPolicy(severity_overrides={"PLUGIN_SCHEMA.manifest_unreadable": Severity.LOW})

    [result] = apply_policy([PluginSchemaValidator().validate(root)], policy)
    assert _checks(result) == {"manifest_unreadable": Severity.HIGH}
    assert result.metadata["security_failure"] is True
    assert not result.passed


def test_selected_manifest_that_nothing_was_parsed_from_stops_tier1(tmp_path: Path) -> None:
    root = _write(tmp_path / "p", {"agent_plugin.yaml": _LATIN1_BUNDLE})

    results = run_validation(root, checks="schema,unicode", content_type=CONTENT_TYPE_PLUGIN)
    assert [result.validator_name for result in results] == ["Plugin Schema & Bundle References"]
    assert not results[0].passed


def test_policy_can_downgrade_a_selected_manifest_that_was_read_leniently(tmp_path: Path) -> None:
    latin1 = json.dumps({"name": "demo", "description": "caf\xe9"}, ensure_ascii=False).encode("latin-1")
    root = _write(tmp_path / "p", {".claude-plugin/plugin.json": latin1})
    policy = ValidationPolicy(severity_overrides={"PLUGIN_SCHEMA.manifest_unreadable": Severity.LOW})

    [result] = apply_policy([PluginSchemaValidator().validate(root)], policy)
    assert _checks(result) == {"manifest_unreadable": Severity.LOW}
    assert "security_failure" not in result.metadata
    assert result.passed


@pytest.mark.parametrize(
    ("files", "manifest_type"),
    [
        ({".claude-plugin/plugin.json": {"name": "demo"}}, PLUGIN_CONTAINED_MANIFEST_TYPE),
        ({"plugin.json": {"$schema": _AP_SCHEMA, "name": "demo"}}, PLUGIN_AGENT_PLUGINS_V1_MANIFEST_TYPE),
        (
            {"agent_plugin.yaml": b"name: demo\nauthor:\n  email: a@example.com\nmcp:\n  - name: t\n    provider: p\n"},
            PLUGIN_MANIFEST_TYPE,
        ),
    ],
    ids=["claude", "agent-plugins", "bundle"],
)
def test_selected_manifest_is_read_once_per_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes | dict], manifest_type: str
) -> None:
    """The manifest check, the manifest declarations, and the inventory share one read and parse."""
    root = _write(tmp_path / "p", files)
    reads: list[str] = []
    real_read_text = PluginManifestLocation.read_text

    def counting_read_text(self: PluginManifestLocation, **kwargs: object) -> str:
        reads.append(self.manifest_filename)
        return real_read_text(self, **kwargs)

    monkeypatch.setattr(PluginManifestLocation, "read_text", counting_read_text)
    result = PluginSchemaValidator().validate(root)

    assert result.metadata["manifest_type"] == manifest_type
    assert result.passed, result.errors
    assert reads == [next(iter(files))]
    [row] = result.metadata["plugin"]["manifest_declarations"]["manifests"]
    assert row["name"] == "demo"


def test_selected_manifest_declaration_row_comes_from_the_validated_parse(tmp_path: Path) -> None:
    """The declaration row shows what validation parsed, even for a YAML file with a doubled byte-order mark.

    The YAML parser skips one leading BOM, so the second one becomes part of the first key ('\ufeffname'), and
    validation reports that the manifest has no name. The row used to come from a separate read that stripped
    one more BOM, and showed the name.
    """
    manifest = "\ufeff\ufeffname: demo\nauthor:\n  email: a@example.com\nmcp:\n  - name: t\n    provider: p\n"
    root = _write(tmp_path / "p", {"agent_plugin.yaml": manifest.encode()})

    result = PluginSchemaValidator().validate(root)
    assert {"schema:name:missing", "schema:\ufeffname:extra_forbidden"} <= _checks(result).keys()
    [row] = result.metadata["plugin"]["manifest_declarations"]["manifests"]
    assert row["name"] is None


def test_selected_manifest_replaced_by_a_link_after_discovery_stays_a_security_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"name": "outside-canary"}))
    root = _write(tmp_path / "p", {".claude-plugin/plugin.json": {"name": "demo"}})
    manifest = root / ".claude-plugin" / "plugin.json"
    real_locate = plugin_schema.locate_plugin_manifest

    def locate_then_link(path: Path):
        located = real_locate(path)
        manifest.unlink()
        manifest.symlink_to(outside)
        return located

    monkeypatch.setattr(plugin_schema, "locate_plugin_manifest", locate_then_link)
    result = PluginSchemaValidator().validate(root)
    assert _checks(result) == {"manifest_unsafe": Severity.HIGH}
    assert result.metadata["security_failure"] is True
    assert "outside-canary" not in json.dumps(result.metadata["plugin"])
