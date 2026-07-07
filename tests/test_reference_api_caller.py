# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security checks for packaged API-caller reference scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
    spec.loader.exec_module(module)
    return module


def test_api_caller_rejects_non_http_urls_without_opening_them() -> None:
    api_caller = _load_script("call_api")

    result = api_caller.make_request("file:///etc/passwd")

    assert result["success"] is False
    assert result["error"] == "Error: Only absolute HTTP or HTTPS URLs are supported"


def test_openapi_parser_rejects_non_http_urls_without_opening_them() -> None:
    parser = _load_script("parse_openapi")

    result = parser.fetch_spec("file:///etc/passwd")

    assert result == {"error": "Failed to fetch spec: Only absolute HTTP or HTTPS URLs are supported"}
