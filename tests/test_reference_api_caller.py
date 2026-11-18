# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security checks for packaged API-caller reference scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "skillevaluator"
    / "tier3"
    / "reference_skills"
    / "api-caller"
    / "scripts"
)


def _load_script(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / f"{module_name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
    return module


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        ("plain-id", "plain-id"),
        ("file.txt", "file.txt"),
        ("...", "..."),
        (7, "7"),
        ("red blue", "red%20blue"),
        ("café", "caf%C3%A9"),
        ("a/b?#c%", "a%2Fb%3F%23c%25"),
        ("a%2Fb", "a%252Fb"),
    ],
)
def test_openapi_path_values_are_encoded_as_segments(value: str | int, encoded: str) -> None:
    caller = _load_script("call_api")
    spec = {"servers": [{"url": "https://example.invalid/v1"}]}

    result = caller.build_url_from_spec(spec, "/items/{id}", {"id": value, "limit": 2})

    assert result == f"https://example.invalid/v1/items/{encoded}?limit=2"


@pytest.mark.parametrize("value", [".", ".."])
def test_openapi_path_dot_segments_are_rejected(value: str) -> None:
    caller = _load_script("call_api")
    spec = {"servers": [{"url": "https://example.invalid/v1"}]}

    with pytest.raises(ValueError, match=r"Path parameter 'id' must not be '\.' or '\.\.'"):
        caller.build_url_from_spec(spec, "/items/{id}", {"id": value})


@pytest.mark.parametrize("value", [".", ".."])
def test_openapi_path_dot_segments_report_json_errors(
    value: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    caller = _load_script("call_api")
    spec = {
        "servers": [{"url": "https://example.invalid/v1"}],
        "paths": {"/items/{id}": {"get": {"operationId": "getItem"}}},
    }
    monkeypatch.setattr(caller, "fetch_openapi_spec", lambda _url: spec)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "call_api.py",
            "--spec",
            "https://example.invalid/spec",
            "--operation",
            "getItem",
            "--params",
            json.dumps({"id": value}),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        caller.main()

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "success": False,
        "error": "Invalid input: Path parameter 'id' must not be '.' or '..'",
    }


def test_api_caller_rejects_non_http_urls_without_opening_them() -> None:
    api_caller = _load_script("call_api")

    result = api_caller.make_request("file:///etc/passwd")

    assert result["success"] is False
    assert result["error"] == "Error: Only absolute HTTP or HTTPS URLs are supported"


def test_openapi_parser_rejects_non_http_urls_without_opening_them() -> None:
    parser = _load_script("parse_openapi")

    result = parser.fetch_spec("file:///etc/passwd")

    assert result == {"error": "Failed to fetch spec: Only absolute HTTP or HTTPS URLs are supported"}


@pytest.mark.parametrize("script", ["call_api", "parse_openapi"])
def test_scripts_share_one_url_scheme_check(script: str) -> None:
    module = _load_script(script)

    assert module.validate_http_url.__module__ == "url_safety"
    assert "def validate_http_url" not in (SCRIPTS_DIR / f"{script}.py").read_text(encoding="utf-8")
