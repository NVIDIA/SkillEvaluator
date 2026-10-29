# SPDX-License-Identifier: Apache-2.0
"""Untrusted requirement files cannot invoke pip resolution or build backends."""

import json
from pathlib import Path

import pytest

from skillevaluator.utils.tool_runner import ToolResult, Tools
from skillevaluator.validators.dependencies import DependencySecurityValidator


def _audit(tmp_path, monkeypatch, text, *, returned=None):
    target = tmp_path / "requirements.txt"
    target.write_text(text)
    calls = []
    monkeypatch.setattr(type(Tools.pip_audit), "is_available", property(lambda _self: True))

    def run(args, **kwargs):
        assert "--no-deps" in args and "--disable-pip" in args
        path = Path(args[args.index("-r") + 1])
        assert path != target
        pins = path.read_text().splitlines()
        calls.extend(pins)
        dependencies = (
            returned
            if returned is not None
            else [{"name": pin.split("==")[0], "version": pin.split("==")[1], "vulns": []} for pin in pins]
        )
        return ToolResult(True, json.dumps({"dependencies": dependencies}), "", 0)

    monkeypatch.setattr(Tools.pip_audit, "run", run)
    result = DependencySecurityValidator(use_safety=False)._run_pip_audit_on_file(target)
    return result, calls


@pytest.mark.parametrize(
    "text",
    [
        "payload @ file:///workspace/payload",
        "./payload",
        "-e ./payload",
        "payload @ git+https://example.invalid/payload.git",
        "-r https://example.invalid/private.txt",
        "--index-url https://user:secret@example.invalid",
        "requests>=2",
        "requests==2.*",
        "--find-links ./packages",
        "-c constraints.txt",
        "requests @ ${SECRET_URL}",
    ],
)
def test_unsupported_requirements_are_incomplete_without_invoking_scanner(tmp_path, monkeypatch, text):
    result, calls = _audit(tmp_path, monkeypatch, text)
    assert result.is_incomplete and not result.passed
    assert calls == []
    assert "secret" not in " ".join(result.warnings)


def test_supported_pins_are_audited_even_with_unsupported_declarations(tmp_path, monkeypatch):
    result, calls = _audit(tmp_path, monkeypatch, "idna==3.10\npayload @ file:///workspace/payload\n")
    assert calls == ["idna==3.10"]
    assert result.is_incomplete


def test_hashed_continuations_comments_and_markers_are_normalized(tmp_path, monkeypatch):
    text = (
        "idna==3.10 \\\n    --hash=sha256:"
        + "a" * 64
        + " \\\n    --hash=sha256:"
        + "b" * 64
        + "\n# comment\nrequests==2.32.4; python_version < '2'\n"
    )
    result, calls = _audit(tmp_path, monkeypatch, text)
    assert calls == ["idna==3.10"]
    assert result.passed and not result.is_incomplete


def test_missing_pin_evidence_is_incomplete(tmp_path, monkeypatch):
    result, calls = _audit(tmp_path, monkeypatch, "idna==3.10", returned=[])
    assert calls == ["idna==3.10"]
    assert result.is_incomplete
    assert not result.success_details


def test_empty_file_is_completed_without_installing_anything(tmp_path, monkeypatch):
    result, calls = _audit(tmp_path, monkeypatch, "# no Python dependencies\n")
    assert result.passed and calls == []


def test_symlink_is_not_followed(tmp_path, monkeypatch):
    target = tmp_path / "requirements.txt"
    target.symlink_to(tmp_path / "private")
    (tmp_path / "private").write_text("idna==3.10")
    result = DependencySecurityValidator(use_safety=False)._run_pip_audit_on_file(target)
    assert result.is_incomplete


@pytest.mark.parametrize("marker", ['"foo" in extras', '"foo" in dependency_groups'])
def test_unsupported_marker_context_is_incomplete_instead_of_crashing(tmp_path, monkeypatch, marker):
    result, calls = _audit(tmp_path, monkeypatch, f"demo==1; {marker}")
    assert result.is_incomplete and calls == []
