# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAPI server URL resolution and missing path-parameter checks for api-caller."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

PETSTORE_SPEC_URL = "https://petstore3.swagger.io/api/v3/openapi.json"
PET_BODY = {"id": 1, "name": "doggie", "status": "available"}
PETSTORE_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Petstore", "version": "1.0.0"},
    "servers": [{"url": "/api/v3"}],
    "paths": {
        "/pet/{petId}": {
            "get": {
                "operationId": "getPetById",
                "parameters": [
                    {
                        "name": "petId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
            }
        }
    },
}


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


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script), *args],
        check=False,
        capture_output=True,
        text=True,
    )


SWAGGER2_SPEC = {
    "swagger": "2.0",
    "info": {"title": "Petstore", "version": "1.0.0"},
    "basePath": "/v2",
    "paths": PETSTORE_SPEC["paths"],
}

REDIRECTED_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Catalog", "version": "1.0.0"},
    "servers": [{"url": "v2"}],
    "paths": {
        "/books/{id}": {
            "get": {
                "operationId": "getBook",
                "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
            }
        }
    },
}

_ROUTES = {
    "/api/v3/openapi.json": PETSTORE_SPEC,
    "/api/v3/pet/1": PET_BODY,
    "/v2/swagger.json": SWAGGER2_SPEC,
    "/v2/pet/1": PET_BODY,
    "/current/openapi/catalog.json": REDIRECTED_SPEC,
    "/current/openapi/v2/books/7": {"id": 7},
}
_REDIRECTS = {"/docs/catalog.json": "/current/openapi/catalog.json"}


class _PetstoreHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *args: object) -> None:
        del args

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in _REDIRECTS:
            self.send_response(302)
            self.send_header("Location", _REDIRECTS[path])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        payload = _ROUTES.get(path)
        if payload is None:
            self.send_error(404)
            return
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve_petstore():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PetstoreHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_petstore(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def test_relative_server_is_resolved_against_spec_url() -> None:
    caller = _load_script("call_api")
    spec = {"servers": [{"url": "/api/v3"}]}

    result = caller.build_url_from_spec(spec, "/pet/{petId}", {"petId": 1}, spec_url=PETSTORE_SPEC_URL)

    assert result == "https://petstore3.swagger.io/api/v3/pet/1"


def test_spec_without_servers_uses_spec_origin() -> None:
    caller = _load_script("call_api")

    result = caller.build_url_from_spec({}, "/pet/{petId}", {"petId": 1}, spec_url=PETSTORE_SPEC_URL)

    assert result == "https://petstore3.swagger.io/pet/1"


def test_absolute_server_url_is_unchanged() -> None:
    caller = _load_script("call_api")
    spec = {"servers": [{"url": "https://example.invalid/v1"}]}

    result = caller.build_url_from_spec(spec, "/items/{id}", {"id": 7}, spec_url="https://other.example/openapi.json")

    assert result == "https://example.invalid/v1/items/7"


def test_server_variables_use_defaults() -> None:
    caller = _load_script("call_api")
    spec = {
        "servers": [
            {
                "url": "https://{region}.example.com/v1",
                "variables": {"region": {"default": "us"}},
            }
        ]
    }

    result = caller.build_url_from_spec(spec, "/items/{id}", {"id": 7})

    assert result == "https://us.example.com/v1/items/7"


def test_server_variable_without_default_is_error() -> None:
    caller = _load_script("call_api")
    spec = {
        "servers": [
            {
                "url": "https://{region}.example.com/v1",
                "variables": {"region": {"enum": ["us", "eu"]}},
            }
        ]
    }

    with pytest.raises(ValueError, match=r"server variable 'region' has no default"):
        caller.build_url_from_spec(spec, "/items/{id}", {"id": 7})


def test_non_string_server_url_is_error() -> None:
    caller = _load_script("call_api")
    spec = {"servers": [{"url": 123}]}

    with pytest.raises(ValueError, match=r"server URL must be a string"):
        caller.build_url_from_spec(spec, "/items/{id}", {"id": 7})


@pytest.mark.parametrize("server_url", ["", "   "])
def test_blank_server_url_resolves_to_spec_directory(server_url: str) -> None:
    caller = _load_script("call_api")
    spec = {"servers": [{"url": server_url}]}

    result = caller.build_url_from_spec(spec, "/pet/{petId}", {"petId": 1}, spec_url=PETSTORE_SPEC_URL)

    assert result == "https://petstore3.swagger.io/api/v3/pet/1"


def test_blank_server_url_keeps_directory_spec_url_path() -> None:
    caller = _load_script("call_api")
    spec = {"servers": [{"url": ""}]}

    result = caller.build_url_from_spec(spec, "/pets", {}, spec_url="https://example.invalid/v1/")

    assert result == "https://example.invalid/v1/pets"


def test_trailing_slash_server_does_not_double_slash() -> None:
    caller = _load_script("call_api")
    spec = {"servers": [{"url": "https://example.invalid/v1/"}]}

    result = caller.build_url_from_spec(spec, "/items/{id}", {"id": 7})

    assert result == "https://example.invalid/v1/items/7"


def test_missing_path_parameter_is_error() -> None:
    caller = _load_script("call_api")
    spec = {"servers": [{"url": "https://example.invalid/v1"}]}

    with pytest.raises(ValueError, match=r"missing path parameter 'petId'"):
        caller.build_url_from_spec(spec, "/pet/{petId}", {})


def test_server_variable_without_default_reports_json_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    caller = _load_script("call_api")
    spec = {
        "servers": [
            {
                "url": "https://{region}.example.com/v1",
                "variables": {"region": {"enum": ["us", "eu"]}},
            }
        ],
        "paths": {"/items/{id}": {"get": {"operationId": "getItem"}}},
    }
    monkeypatch.setattr(caller, "fetch_openapi_spec", lambda url: (spec, url))
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
            json.dumps({"id": 7}),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        caller.main()

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "success": False,
        "error": "Invalid input: server variable 'region' has no default",
    }


def test_missing_path_parameter_reports_json_error_and_makes_no_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    caller = _load_script("call_api")
    spec = {
        "servers": [{"url": "https://example.invalid/v1"}],
        "paths": {"/pet/{petId}": {"get": {"operationId": "getPetById"}}},
    }
    requests: list[object] = []
    monkeypatch.setattr(caller, "fetch_openapi_spec", lambda url: (spec, url))
    monkeypatch.setattr(caller, "make_request", lambda *args, **kwargs: requests.append((args, kwargs)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "call_api.py",
            "--spec",
            "https://example.invalid/spec",
            "--operation",
            "getPetById",
            "--params",
            "{}",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        caller.main()

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert captured.out == ""
    assert requests == []
    assert json.loads(captured.err) == {
        "success": False,
        "error": "Invalid input: missing path parameter 'petId'",
    }


def test_call_api_subprocess_resolves_relative_server() -> None:
    server, thread = _serve_petstore()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        result = _run_script(
            "call_api.py",
            "--spec",
            f"{origin}/api/v3/openapi.json",
            "--operation",
            "getPetById",
            "--params",
            json.dumps({"petId": 1}),
        )
    finally:
        _stop_petstore(server, thread)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["data"] == PET_BODY
    assert payload["url"] == f"{origin}/api/v3/pet/1"


def test_parse_openapi_subprocess_reports_absolute_urls() -> None:
    server, thread = _serve_petstore()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        result = _run_script("parse_openapi.py", f"{origin}/api/v3/openapi.json")
    finally:
        _stop_petstore(server, thread)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["base_url"] == f"{origin}/api/v3"
    assert payload["operations"][0]["full_url"] == f"{origin}/api/v3/pet/{{petId}}"
    assert payload["operations"][0]["path"] == "/pet/{petId}"
    assert payload["operations"][0]["operation_id"] == "getPetById"


def test_swagger2_base_url_uses_schemes_host_and_base_path() -> None:
    caller = _load_script("call_api")
    spec = {"swagger": "2.0", "host": "api.example.com", "basePath": "/v2", "schemes": ["http", "https"]}

    result = caller.build_url_from_spec(
        spec, "/pet/{petId}", {"petId": 1}, spec_url="https://docs.example.org/v2/swagger.json"
    )

    assert result == "https://api.example.com/v2/pet/1"


def test_swagger2_defaults_to_spec_url_scheme_and_host() -> None:
    caller = _load_script("call_api")
    spec = {"swagger": "2.0"}

    result = caller.build_url_from_spec(spec, "/pet/{petId}", {"petId": 1}, spec_url="http://127.0.0.1:8080/x/s.json")

    assert result == "http://127.0.0.1:8080/pet/1"


def test_swagger2_without_http_scheme_is_error() -> None:
    caller = _load_script("call_api")

    with pytest.raises(ValueError, match="no HTTP or HTTPS scheme"):
        caller.build_url_from_spec({"swagger": "2.0", "schemes": ["ws"]}, "/pet", {}, spec_url=PETSTORE_SPEC_URL)


def test_swagger2_subprocesses_use_base_path() -> None:
    server, thread = _serve_petstore()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        parsed = _run_script("parse_openapi.py", f"{origin}/v2/swagger.json")
        called = _run_script(
            "call_api.py",
            "--spec",
            f"{origin}/v2/swagger.json",
            "--operation",
            "getPetById",
            "--params",
            json.dumps({"petId": 1}),
        )
    finally:
        _stop_petstore(server, thread)

    assert parsed.returncode == 0, parsed.stderr
    assert json.loads(parsed.stdout)["operations"][0]["full_url"] == f"{origin}/v2/pet/{{petId}}"
    assert called.returncode == 0, called.stderr
    assert json.loads(called.stdout)["url"] == f"{origin}/v2/pet/1"


def test_relative_server_resolves_against_final_spec_url_after_redirect() -> None:
    server, thread = _serve_petstore()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        parsed = _run_script("parse_openapi.py", f"{origin}/docs/catalog.json")
        called = _run_script(
            "call_api.py",
            "--spec",
            f"{origin}/docs/catalog.json",
            "--operation",
            "getBook",
            "--params",
            json.dumps({"id": 7}),
        )
    finally:
        _stop_petstore(server, thread)

    assert parsed.returncode == 0, parsed.stderr
    payload = json.loads(parsed.stdout)
    assert payload["base_url"] == f"{origin}/current/openapi/v2"
    assert payload["operations"][0]["full_url"] == f"{origin}/current/openapi/v2/books/{{id}}"
    assert called.returncode == 0, called.stderr
    result = json.loads(called.stdout)
    assert result["url"] == f"{origin}/current/openapi/v2/books/7"
    assert result["data"] == {"id": 7}
    assert "final_url" not in result


def test_operation_relative_servers_without_document_servers() -> None:
    caller = _load_script("call_api")
    parser = _load_script("parse_openapi")
    spec_url = "http://docs.example/specs/openapi.json"
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Pets", "version": "1.0.0"},
        "paths": {
            "/pets": {
                "get": {
                    "operationId": "listPets",
                    "servers": [{"url": "/v2"}],
                }
            }
        },
    }

    path, _method, operation = caller.find_operation(spec, "listPets")
    built = caller.build_url_from_spec(spec, path, {}, spec_url=spec_url, operation=operation)
    parsed = parser.parse_spec(spec, spec_url=spec_url)

    assert built == "http://docs.example/v2/pets"
    assert parsed["base_url"] == "http://docs.example/"
    assert parsed["operations"][0]["full_url"] == "http://docs.example/v2/pets"


def test_operation_and_path_item_servers_route_away_from_spec_origin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    caller = _load_script("call_api")
    parser = _load_script("parse_openapi")
    spec_url = "http://docs.example/openapi.json"
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Pets", "version": "1.0.0"},
        "paths": {
            "/pets": {
                "get": {
                    "operationId": "listPets",
                    "servers": [{"url": "http://api.example/v2"}],
                }
            },
            "/owners/{ownerId}": {
                "servers": [
                    {
                        "url": "http://{env}.api.example/v2",
                        "variables": {"env": {"default": "prod"}},
                    }
                ],
                "get": {
                    "operationId": "getOwner",
                    "parameters": [{"name": "ownerId", "in": "path", "required": True}],
                },
            },
        },
    }

    pet_path, _method, pet_op = caller.find_operation(spec, "listPets")
    owner_path, _method, owner_op = caller.find_operation(spec, "getOwner")
    assert caller.build_url_from_spec(spec, pet_path, {}, spec_url=spec_url, operation=pet_op) == (
        "http://api.example/v2/pets"
    )
    assert (
        caller.build_url_from_spec(spec, owner_path, {"ownerId": 42}, spec_url=spec_url, operation=owner_op)
        == "http://prod.api.example/v2/owners/42"
    )

    parsed = parser.parse_spec(spec, spec_url=spec_url)
    assert parsed["base_url"] == "http://docs.example/"
    assert {op["operation_id"]: op["full_url"] for op in parsed["operations"]} == {
        "listPets": "http://api.example/v2/pets",
        "getOwner": "http://prod.api.example/v2/owners/{ownerId}",
    }

    sent: list[tuple[str, str, dict | None]] = []

    def fake_make_request(
        url: str, method: str = "GET", data: object = None, headers: dict | None = None, timeout: int = 30
    ) -> dict:
        del data, timeout
        caller.validate_http_url(url)
        sent.append((url, method, headers))
        return {
            "success": True,
            "status_code": 200,
            "headers": {},
            "data": [],
            "error": None,
            "url": url,
            "method": method,
        }

    monkeypatch.setattr(caller, "fetch_openapi_spec", lambda url: (spec, url))
    monkeypatch.setattr(caller, "make_request", fake_make_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "call_api.py",
            "--spec",
            spec_url,
            "--operation",
            "listPets",
            "--headers",
            json.dumps({"Authorization": "Bearer secret-token"}),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        caller.main()

    assert exc.value.code == 0
    capsys.readouterr()
    assert sent == [("http://api.example/v2/pets", "GET", {"Authorization": "Bearer secret-token"})]


def test_server_precedence_operation_overrides_path_item_overrides_document() -> None:
    caller = _load_script("call_api")
    parser = _load_script("parse_openapi")
    spec_url = "http://docs.example/openapi.json"
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "Precedence", "version": "1.0.0"},
        "servers": [{"url": "http://doc.example/v1"}],
        "paths": {
            "/pets": {
                "servers": [{"url": "http://path.example/v2"}],
                "get": {
                    "operationId": "listPets",
                    "servers": [{"url": "http://op.example/v3"}],
                },
                "post": {
                    "operationId": "createPet",
                },
            },
            "/health": {
                "get": {
                    "operationId": "getHealth",
                }
            },
        },
    }

    resolved_calls = {}
    for op_id in ("listPets", "createPet", "getHealth"):
        path, _method, operation = caller.find_operation(spec, op_id)
        resolved_calls[op_id] = caller.build_url_from_spec(spec, path, {}, spec_url=spec_url, operation=operation)

    parsed = parser.parse_spec(spec, spec_url=spec_url)
    expected = {
        "listPets": "http://op.example/v3/pets",
        "createPet": "http://path.example/v2/pets",
        "getHealth": "http://doc.example/v1/health",
    }
    assert resolved_calls == expected
    assert parsed["base_url"] == "http://doc.example/v1"
    assert {op["operation_id"]: op["full_url"] for op in parsed["operations"]} == expected


def test_operation_server_rejects_disallowed_scheme() -> None:
    caller = _load_script("call_api")
    parser = _load_script("parse_openapi")
    spec = {
        "openapi": "3.0.3",
        "paths": {
            "/pets": {
                "get": {
                    "operationId": "listPets",
                    "servers": [{"url": "file:///etc/passwd"}],
                }
            }
        },
    }

    with pytest.raises(ValueError, match="Only absolute HTTP or HTTPS URLs are supported"):
        caller.build_url_from_spec(spec, "/pets", {}, spec_url="http://docs.example/openapi.json")

    with pytest.raises(ValueError, match="Only absolute HTTP or HTTPS URLs are supported"):
        parser.parse_spec(spec, spec_url="http://docs.example/openapi.json")
