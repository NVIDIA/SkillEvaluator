# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-origin, loopback-only HTTP adapter for the local workbench."""

from __future__ import annotations

import asyncio
import hmac
import json
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from skillevaluator.studio.jobs import BusyError, Studio
from skillevaluator.studio.store import MAX_UPLOAD, safe_file

if TYPE_CHECKING:
    from starlette.requests import Request

STATIC = Path(__file__).parent / "static"
MAX_BODY = MAX_UPLOAD * 4 // 3 + 256 * 1024


class LocalBoundary(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "")
        try:
            parsed = urlsplit("http://" + host)
            hostname = parsed.hostname
            port = parsed.port or 80
        except ValueError:
            return JSONResponse({"error": "Use the localhost address printed by the UI command."}, status_code=403)
        server_port = request.scope.get("server", ("127.0.0.1", port))[1]
        if (
            hostname not in {"localhost", "127.0.0.1", "::1"}
            or port != server_port
            or parsed.username
            or parsed.password
        ):
            return JSONResponse({"error": "Use the localhost address printed by the UI command."}, status_code=403)
        origin = request.headers.get("origin")
        if (origin and origin != f"http://{host}") or request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"error": "Requests must come from this local UI."}, status_code=403)
        if request.method not in {"GET", "HEAD"}:
            supplied = request.headers.get("x-studio-token", "")
            if not hmac.compare_digest(supplied, request.app.state.studio.token):
                return JSONResponse({"error": "Refresh this page before making changes."}, status_code=403)
        response = await call_next(request)
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            }
        )
        return response


async def body(request: Request) -> dict:
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
        raise ValueError("Send this request as JSON.")
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BODY:
            raise ValueError("Upload a skill smaller than 12 MB.")
    try:
        result = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("This request contains invalid JSON.") from exc
    if not isinstance(result, dict):
        raise ValueError("This request needs a JSON object.")
    return result


def studio(request: Request) -> Studio:
    return request.app.state.studio


async def state(request: Request):
    return JSONResponse(await run_in_threadpool(studio(request).state))


async def config(request: Request):
    result = studio(request).configure(await body(request))
    return JSONResponse(result)


async def create_skill(request: Request):
    payload = await body(request)
    return JSONResponse(await run_in_threadpool(studio(request).workspace.create, payload), status_code=201)


async def sample(request: Request):
    return JSONResponse(await run_in_threadpool(studio(request).workspace.sample), status_code=201)


async def skill(request: Request):
    workspace = studio(request).workspace
    skill_id = request.path_params["skill_id"]
    if request.method == "PUT":
        payload = await body(request)
        result = await run_in_threadpool(workspace.update, skill_id, payload.get("content"))
    else:
        result = await run_in_threadpool(workspace.describe, skill_id)
    return JSONResponse(result)


async def dataset(request: Request):
    payload = await body(request)
    result = await run_in_threadpool(
        studio(request).workspace.save_cases, request.path_params["skill_id"], payload.get("cases")
    )
    return JSONResponse(result)


async def candidate(request: Request):
    payload = await body(request)
    result = await run_in_threadpool(
        studio(request).workspace.candidate, request.path_params["skill_id"], payload.get("content")
    )
    return JSONResponse(result, status_code=201)


async def export(request: Request):
    workspace = studio(request).workspace
    skill_id = request.path_params["skill_id"]
    data = await run_in_threadpool(workspace.export, skill_id)
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="skill-{skill_id}.zip"'},
    )


async def create_job(request: Request):
    return JSONResponse(await run_in_threadpool(studio(request).start, await body(request)), status_code=202)


async def job(request: Request):
    service = studio(request)
    method = service.cancel if request.method == "DELETE" else service.job
    return JSONResponse(await run_in_threadpool(method, request.path_params["job_id"]))


async def improvement_readiness(request: Request):
    from skillevaluator.studio.improvements import readiness

    service = studio(request)
    result = await run_in_threadpool(
        readiness, service.workspace, request.path_params["job_id"], request.query_params.get("skill_id")
    )
    return JSONResponse(service.scrub(result))


async def download(request: Request):
    service = studio(request)
    root = service._job_path(request.path_params["job_id"])
    relative = request.query_params.get("file", "result.json")
    path = safe_file(root, relative)
    if not Path(relative).parts or relative in {".", "/"}:
        raise ValueError("Choose a report file to download.")
    # Worker source/configs and credentials are not browser artifacts.
    if (
        any(p.startswith(".") for p in Path(relative).parts)
        or Path(relative).parts[0] == "source"
        or path.name in {"request.json", "metadata.json", "config.json", "auth.json", "credentials.json"}
    ):
        raise ValueError("This file is not a downloadable report artifact.")
    allowed = {".json", ".jsonl", ".html", ".md", ".txt", ".log", ".png", ".pdf", ".diff"}
    if not path.is_file() or path.suffix.lower() not in allowed or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("This report artifact is unavailable or too large.")
    if path.suffix.lower() in {".png", ".pdf"}:
        data = path.read_bytes()
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() == ".json":
            try:
                text = json.dumps(service.scrub(json.loads(text)), indent=2, ensure_ascii=False)
            except ValueError:
                text = str(service.scrub(text))
        else:
            text = str(service.scrub(text))
        data = text.encode()
    filename = re_safe_filename(path.name)
    return Response(
        data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def re_safe_filename(name: str) -> str:
    return "".join(c if c.isascii() and (c.isalnum() or c in ".-_") else "_" for c in name)


async def static(request: Request):
    name = request.path_params.get("name") or "index.html"
    if name not in {"index.html", "app.js", "style.css"}:
        return JSONResponse({"error": "Not found."}, status_code=404)
    path = STATIC / name
    content_type = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}[path.suffix]
    return Response(path.read_bytes(), media_type=content_type)


async def invalid(request: Request, exc: Exception):
    return JSONResponse(
        {"error": str(studio(request).scrub(str(exc)))}, status_code=409 if isinstance(exc, BusyError) else 400
    )


def create_app(data_dir: Path | None = None) -> Starlette:
    service = Studio(data_dir or Path.home() / ".local/share/skillevaluator/studio")

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await asyncio.to_thread(service.close)

    app = Starlette(
        routes=[
            Route("/api/state", state),
            Route("/api/config", config, methods=["POST"]),
            Route("/api/sample", sample, methods=["POST"]),
            Route("/api/skills", create_skill, methods=["POST"]),
            Route("/api/skills/{skill_id}", skill, methods=["GET", "PUT"]),
            Route("/api/skills/{skill_id}/dataset", dataset, methods=["PUT"]),
            Route("/api/skills/{skill_id}/candidate", candidate, methods=["POST"]),
            Route("/api/skills/{skill_id}/export", export),
            Route("/api/jobs", create_job, methods=["POST"]),
            Route("/api/jobs/{job_id}", job, methods=["GET", "DELETE"]),
            Route("/api/jobs/{job_id}/download", download),
            Route("/api/jobs/{job_id}/improvement", improvement_readiness),
            Route("/", static),
            Route("/{name}", static),
        ],
        exception_handlers={
            ValueError: invalid,
            OSError: invalid,
            zipfile.BadZipFile: invalid,
            NotImplementedError: invalid,
        },
        lifespan=lifespan,
    )
    app.state.studio = service
    app.add_middleware(LocalBoundary)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    return app
