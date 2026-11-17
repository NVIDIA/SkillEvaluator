# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security checks for packaged API-caller reference scripts."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
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


def test_inline_path_parameters_are_inherited_and_overridden() -> None:
    parser = _load_script("parse_openapi")
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Inline parameters", "version": "1"},
        "paths": {
            "/items/{id}": {
                "parameters": [
                    {
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "description": "Item ID",
                        "schema": {"type": "string"},
                    },
                    {"name": "limit", "in": "query", "required": True, "schema": {"type": "integer"}},
                    {"name": "id", "in": "header", "schema": {"type": "string"}},
                ],
                "get": {
                    "operationId": "getItem",
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "required": False,
                            "description": "Operation limit",
                            "schema": {"type": "number"},
                        },
                        {"name": "filter", "in": "query", "schema": {"type": "string"}},
                    ],
                    "responses": {"200": {"description": "OK"}},
                },
                "post": {"operationId": "postItem", "responses": {"200": {"description": "OK"}}},
            },
        },
    }
    original_spec = deepcopy(spec)

    get, post = parser.parse_spec(spec)["operations"]

    assert [(p["name"], p["in"]) for p in get["parameters"]] == [
        ("id", "path"),
        ("limit", "query"),
        ("id", "header"),
        ("filter", "query"),
    ]
    assert get["parameters"][0] == {
        "name": "id",
        "in": "path",
        "required": True,
        "type": "string",
        "description": "Item ID",
    }
    assert get["parameters"][1] == {
        "name": "limit",
        "in": "query",
        "required": False,
        "type": "number",
        "description": "Operation limit",
    }
    assert [(p["name"], p["in"]) for p in post["parameters"]] == [
        ("id", "path"),
        ("limit", "query"),
        ("id", "header"),
    ]
    assert post["parameters"][1]["type"] == "integer"
    assert post["parameters"][1]["required"] is True
    assert spec == original_spec


@pytest.mark.parametrize(
    ("path_parameters", "operation_parameters"),
    [
        (
            [{"name": "id", "in": "path"}],
            [{"$ref": "#/components/parameters/Limit"}, {"$ref": "#/components/parameters/Filter"}],
        ),
        ([{"$ref": "#/components/parameters/ItemId"}], [{"name": "limit", "in": "query"}]),
        ([{"in": "path"}], [{"name": "limit", "in": "query"}]),
        ([], [{"name": "limit", "in": "query"}]),
        ([], []),
    ],
    ids=["operation-references", "path-reference", "missing-identity", "operation-only", "empty"],
)
def test_non_inline_parameter_lists_keep_operation_output(
    path_parameters: list[dict], operation_parameters: list[dict]
) -> None:
    parser = _load_script("parse_openapi")
    spec = {
        "paths": {
            "/items/{id}": {
                "parameters": path_parameters,
                "get": {"parameters": operation_parameters, "responses": {"200": {"description": "OK"}}},
            },
        },
    }

    parameters = parser.parse_spec(spec)["operations"][0]["parameters"]

    assert parameters == parser.parse_parameters(operation_parameters)
    assert len(parameters) == len(operation_parameters)
