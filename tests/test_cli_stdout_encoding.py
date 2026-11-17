# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI output remains available without corrupting machine-readable values."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from io import BytesIO, StringIO, TextIOWrapper
from types import SimpleNamespace

import pytest
from rich.console import Console

from skillevaluator import cli as cli_module
from skillevaluator.utils.json_output import json_needs_ascii

_ENCODINGS = ("cp936", "cp1252", "ascii", "utf-8")
_UNICODE = "Résumé 中文 • 😀"
_JSON_CASES = [(encoding, _UNICODE, encoding == "utf-8") for encoding in _ENCODINGS] + [
    ("cp936", "中文", True),
    ("cp1252", "Résumé café", True),
    ("ascii", "plain ASCII", True),
]


@pytest.mark.parametrize("newline", (None, "\n", "\r\n"))
@pytest.mark.parametrize("encoding", _ENCODINGS)
def test_cli_replaces_characters_not_supported_by_stdout_encoding(monkeypatch, encoding, newline) -> None:
    output = BytesIO()
    stdout = TextIOWrapper(output, encoding=encoding, errors="strict", newline=newline)
    monkeypatch.setattr(cli_module.sys, "stdout", stdout)

    with cli_module.cli.make_context("skillevaluator", ["tier3"]):
        Console(file=stdout, force_terminal=False).print(f"  {_UNICODE} Errors: 1")
    stdout.flush()

    assert stdout.errors == "replace"
    assert output.getvalue().decode(encoding).replace("\r\n", "\n") == f"  {_UNICODE} Errors: 1\n".encode(
        encoding, errors="replace"
    ).decode(encoding)


@pytest.mark.parametrize("encoding", _ENCODINGS)
def test_root_help_before_callback_supports_code_pages(encoding) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from skillevaluator.cli import cli; import sys; "
            "cli.main(standalone_mode=False); assert sys.stdout.errors == 'replace'",
            "--help",
        ],
        env={**os.environ, "PYTHONIOENCODING": f"{encoding}:strict", "NO_COLOR": "1"},
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert b"Usage:" in result.stdout


def test_stdout_without_reconfigure_is_supported(monkeypatch) -> None:
    stdout = StringIO()
    monkeypatch.setattr(cli_module.sys, "stdout", stdout)
    cli_module.cli.main(["--help"], standalone_mode=False)
    assert "Usage:" in stdout.getvalue()


@pytest.mark.parametrize("encoding,text,readable", _JSON_CASES)
def test_migration_preview_preserves_unicode(encoding, text, readable, tmp_path) -> None:
    source = tmp_path / "legacy.json"
    source.write_text(
        json.dumps([{"id": 1, "question": text, "ground_truth": text, "expected_skill": "demo"}]),
        encoding="utf-8",
    )
    original = source.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from skillevaluator.cli import cli; cli()",
            "tier3",
            "migrate-dataset",
            str(source),
            "--skill-name",
            "demo",
        ],
        env={**os.environ, "PYTHONIOENCODING": f"{encoding}:strict", "NO_COLOR": "1"},
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rendered = result.stdout.decode(encoding)
    case = json.loads(rendered)["evals"][0]
    assert case["prompt"] == text
    assert case["expected_output"] == text
    if readable:
        assert text in rendered
    assert source.read_bytes() == original


@pytest.mark.parametrize("encoding,text,readable", _JSON_CASES)
def test_validate_json_preserves_unicode(monkeypatch, encoding, text, readable, tmp_path) -> None:
    from skillevaluator.tier3 import commands

    output = BytesIO()
    stdout = TextIOWrapper(output, encoding=encoding, errors="strict")
    monkeypatch.setattr(cli_module.sys, "stdout", stdout)
    monkeypatch.setattr(commands, "console", Console(file=stdout, force_terminal=False, width=120))
    monkeypatch.setattr(
        commands,
        "validate_skillevaluators",
        lambda _: [
            SimpleNamespace(path=text, status="warning", message=text),
        ],
    )
    monkeypatch.setattr(commands, "dataset_coverage_result", lambda _: None)
    monkeypatch.setattr(cli_module, "setup_logging", lambda **_kwargs: None)

    with pytest.raises(SystemExit) as exc:
        cli_module.cli.main(["tier3", "validate", str(tmp_path), "--json"], standalone_mode=False)
    assert exc.value.code == 0
    stdout.flush()
    rendered = output.getvalue().decode(encoding)
    assert json.loads(rendered) == [
        {"path": text, "status": "warning", "message": text},
    ]
    if readable:
        assert text in rendered


def test_json_without_stream_encoding_remains_readable() -> None:
    assert json_needs_ascii({"value": _UNICODE}, None) is False
