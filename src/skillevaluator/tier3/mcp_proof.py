# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in public MCP proof for author-supplied URL MCP servers (``--probe-mcp``).

Before a plugin run, :func:`probe_mcp_servers` performs one bounded host probe
per URL server: MCP ``initialize`` followed by ``tools/list`` over streamable
HTTP or SSE, using the ``mcp`` client library with a constrained ``httpx``
client (no redirects, no proxy or ``.netrc`` credentials, strict timeouts, and
a response-size cap). The endpoint policy runs first: cloud metadata endpoints
are never probed, and private, loopback, or link-local endpoints (including
host names that resolve to them) are probed only when
``mcp.allowed_private_hosts`` allowlists them, by address, network, or host
name. Every connection is pinned to the addresses the policy checked, so a
host name that re-resolves elsewhere (DNS rebinding) is never reached. The
only headers sent are the server's declared literal header values.
The plugin chooses both the URL and the header template, so a ``${VAR}`` or
``$VAR`` reference is expanded from the host environment only when the user
names that variable with ``--probe-mcp-env`` (``NAME`` for every server,
``NAME=HOST`` or ``NAME@SERVER`` for one); a header that references any other
variable is not sent, and the variable is never read. Each server's detail
records which variables were sent to which host. The whole probe, DNS lookup
included, runs under one deadline, and a transport error that the ``mcp``
client only logs (an oversized body, an off-origin SSE endpoint) ends the
probe at once.

After the run, :func:`apply_in_agent_mcp_proof` upgrades each server from the
with-plugin arm's ``plugin_signals_summary.mcp_calls.by_server``:
``reachable-in-agent`` when the agent called it, ``used-successfully`` when at
least one call succeeded.

The result is ``provenance["mcp_proof"] = {"<server>": {"status", "tools",
"detail"}}``. It is advisory: a runnable URL server that stays unproven never
changes the ``INCOMPLETE`` rule or any score.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from skillevaluator.tier3.eval_core.plugin_signals import declared_mcp_server_matcher
from skillevaluator.utils.redaction import redact_sensitive_text
from skillevaluator.utils.rich_markup import strip_terminal_controls
from skillevaluator.validators.mcp_static import HostAllowlist, classify_endpoint_host, host_name_is_allowlisted

STATUS_DECLARED = "declared"
STATUS_REACHABLE_HOST = "reachable-host"
STATUS_UNREACHABLE = "unreachable"
STATUS_UNSUPPORTED = "unsupported"
STATUS_REACHABLE_IN_AGENT = "reachable-in-agent"
STATUS_USED_SUCCESSFULLY = "used-successfully"
MCP_PROOF_STATUSES = (
    STATUS_DECLARED,
    STATUS_REACHABLE_HOST,
    STATUS_UNREACHABLE,
    STATUS_UNSUPPORTED,
    STATUS_REACHABLE_IN_AGENT,
    STATUS_USED_SUCCESSFULLY,
)
_STATUS_RANK = {status: rank for rank, status in enumerate(MCP_PROOF_STATUSES)}

MAX_PROBE_TOOLS = 50
MAX_PROBE_SERVERS = 32
MAX_RESOLVED_ADDRESSES = 32
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_DETAIL_CHARS = 300
MAX_TOOL_NAME_CHARS = 128
CONNECT_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 10.0
TOTAL_TIMEOUT_S = 20.0

NOT_REQUESTED_DETAIL = "host probe not requested (pass --probe-mcp to probe URL MCP servers)"
_HEADER_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
# A ``--probe-mcp-env`` value: ``NAME``, ``NAME=HOST`` (only servers whose URL host is HOST) or
# ``NAME@SERVER`` (only that server). NAME has the shape of a header's ``${VAR}`` reference.
_ENV_GRANT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:=(.+)|@(.+))?")
_GRANT_HOST_RE = re.compile(r"[a-z0-9._:-]{1,253}")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
# The ``mcp`` client logs and swallows some transport errors; the probe listens to these loggers.
_MCP_LOGGER_NAMES = ("mcp",)
_STREAMABLE_TRANSPORTS = frozenset({"", "http", "streamable-http", "streamable_http", "streamablehttp"})
_SSE_TRANSPORTS = frozenset({"sse"})


class _ProbeRefused(Exception):
    """The endpoint policy or transport rules forbid a probe."""

    def __init__(self, status: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class _ProbeFailed(Exception):
    """A fatal transport error the ``mcp`` client would otherwise only log (the probe stops at once)."""


@dataclass(frozen=True)
class EnvGrant:
    """One ``--probe-mcp-env`` value: ``NAME`` (any server), ``NAME=HOST`` or ``NAME@SERVER``."""

    name: str
    host: str = ""
    server: str = ""

    def applies_to(self, server: str, host: str) -> bool:
        if self.host:
            return self.host == host
        if self.server:
            return self.server == server
        return True


def parse_env_grant(value: str | EnvGrant) -> EnvGrant:
    """Parse ``NAME``, ``NAME=HOST`` or ``NAME@SERVER``; raise ``ValueError`` when malformed."""
    if isinstance(value, EnvGrant):
        return value
    match = _ENV_GRANT_RE.fullmatch(str(value))
    if match is None:
        raise ValueError(f"{value!r} is not an environment variable name (NAME, NAME=HOST or NAME@SERVER)")
    name, host, server = match.group(1), match.group(2), match.group(3)
    if host is not None:
        host = host.strip("[]").lower()
        if not _GRANT_HOST_RE.fullmatch(host):
            raise ValueError(f"{value!r}: {host!r} is not a host name or address")
        if ":" in host:
            try:
                ipaddress.IPv6Address(host)
            except ValueError as exc:
                raise ValueError(f"{value!r}: give the host without a port") from exc
        return EnvGrant(name, host=host)
    if server is not None:
        if _CONTROL_RE.search(server) or len(server) > 256:
            raise ValueError(f"{value!r}: the server name is not printable")
        return EnvGrant(name, server=server)
    return EnvGrant(name)


def _target_host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _allowed_env(target: Mapping[str, Any], grants: Iterable[EnvGrant]) -> frozenset[str]:
    """Variable names the user opted in for this server (by name, by its URL host, or for every server)."""
    server = str(target.get("name") or "").strip()
    host = _target_host(str(target.get("url") or ""))
    return frozenset(grant.name for grant in grants if grant.applies_to(server, host))


def planned_env_sends(
    targets: Iterable[Mapping[str, Any]], expand_env: Iterable[str | EnvGrant] = ()
) -> list[tuple[str, str, list[str]]]:
    """``(server, host, variables)`` for each server whose declared headers would carry opted-in variables.

    Only names that the user opted in for that server and that its headers
    reference are listed; the values are never read here.
    """
    grants = [parse_env_grant(value) for value in expand_env]
    plan: list[tuple[str, str, list[str]]] = []
    for target in list(targets)[:MAX_PROBE_SERVERS]:
        name = str(target.get("name") or "").strip()
        headers = target.get("headers")
        if not name or not grants or not isinstance(headers, Mapping):
            continue
        allowed = _allowed_env(target, grants)
        referenced: dict[str, None] = {}
        for value in headers.values():
            if isinstance(value, str):
                refs = [first or second for first, second in _HEADER_REF_RE.findall(value)]
                if refs and all(ref in allowed for ref in refs):
                    referenced.update(dict.fromkeys(refs))
        if referenced:
            plan.append((name, _target_host(str(target.get("url") or "")), sorted(referenced)))
    return plan


def _detail(text: Any) -> str:
    """Bounded, printable, redacted text: terminal controls and C0/C1 characters are removed."""
    window = str(text)[: MAX_DETAIL_CHARS * 8]
    return redact_sensitive_text(" ".join(strip_terminal_controls(window).split()))[:MAX_DETAIL_CHARS]


def _entry(status: str, detail: str, tools: Iterable[str] = ()) -> dict[str, Any]:
    return {"status": status, "tools": list(tools)[:MAX_PROBE_TOOLS], "detail": _detail(detail)}


def declared_mcp_proof(targets: Iterable[Mapping[str, Any]], detail: str = NOT_REQUESTED_DETAIL) -> dict[str, Any]:
    """Every URL server as ``declared`` (no host probe was run)."""
    proof: dict[str, Any] = {}
    for target in list(targets)[:MAX_PROBE_SERVERS]:
        name = str(target.get("name") or "").strip()
        if name:
            proof[name] = _entry(STATUS_DECLARED, detail)
    return proof


def _resolve_headers(
    headers: Mapping[str, Any], environ: Mapping[str, str], expand_env: Iterable[str] = ()
) -> tuple[dict[str, str], list[str], list[str], list[str]]:
    """Return ``(headers to send, unset variables, withheld variables, sent variables)``.

    A header whose ``${VAR}``/``$VAR`` references all name variables in
    ``expand_env`` (the names opted in for this server) is sent with them
    resolved from ``environ``; one that references an opted-in but unset
    variable is dropped (``unset``). A header that references any variable not
    opted in for this server is dropped without reading that variable
    (``withheld``). Literal headers are sent as declared.
    """
    allowed = frozenset(expand_env)
    resolved: dict[str, str] = {}
    missing: list[str] = []
    withheld: list[str] = []
    sent: list[str] = []
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        names = [first or second for first, second in _HEADER_REF_RE.findall(value)]
        blocked = [name for name in names if name not in allowed]
        if blocked:
            withheld.extend(blocked)
            continue
        unset = [name for name in names if not environ.get(name)]
        if unset:
            missing.extend(unset)
            continue
        resolved[key] = _HEADER_REF_RE.sub(lambda match: environ[match.group(1) or match.group(2)], value)
        sent.extend(names)
    return resolved, sorted(set(missing)), sorted(set(withheld)), sorted(set(sent))


def _header_note(missing: list[str], withheld: list[str], sent: list[str] | None = None, host: str = "") -> str:
    note = ""
    if sent:
        note += f"; sent {', '.join(sent)} to {host or 'the server'}"
    if withheld:
        note += (
            f"; headers not sent because they reference host variables not passed with --probe-mcp-env "
            f"for this server: {', '.join(withheld)}"
        )
    if missing:
        note += f"; headers skipped because unset: {', '.join(missing)}"
    return note


def _address_policy(address: str, allowlist: HostAllowlist, *, name_allowed: bool, label: str) -> bool:
    """Return True for an allowed private address; raise for a refused one; False when public.

    ``name_allowed`` is True when the URL's host name itself is allowlisted, which
    accepts any private address it resolves to. Metadata addresses are always refused.
    """
    endpoint = classify_endpoint_host(address)
    if endpoint is None:
        return False
    if endpoint.kind == "metadata":
        raise _ProbeRefused(STATUS_DECLARED, f"not probed: {label} is a cloud metadata endpoint (never probed)")
    if name_allowed or allowlist.allows(endpoint):
        return True
    raise _ProbeRefused(
        STATUS_DECLARED,
        f"not probed: {label} is a {endpoint.reason} address; allowlist it with mcp.allowed_private_hosts to probe it",
    )


def _resolve_within(
    resolver: Callable[[str, int], Iterable[str]], host: str, port: int, timeout: float | None
) -> list[str]:
    """Run ``resolver`` in a daemon worker thread and wait at most ``timeout`` seconds.

    ``getaddrinfo`` cannot be interrupted, so a black-holed name would otherwise
    block for the OS resolver timeout. On timeout the lookup is abandoned (the
    daemon thread never holds up interpreter exit) and ``TimeoutError`` is raised.
    """
    if timeout is None:
        return list(resolver(host, port))
    box: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            box["addresses"] = list(resolver(host, port))
        except BaseException as exc:  # handed to the caller below
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=run, name="skilleval-mcp-probe-dns", daemon=True).start()
    if not done.wait(max(timeout, 0.0)):
        raise TimeoutError(f"DNS resolution timed out after {timeout:g}s")
    if "error" in box:
        raise box["error"]
    return box["addresses"]


def _check_endpoint(
    url: str,
    transport: str,
    allowed_private_hosts: HostAllowlist | Iterable[str],
    resolver: Callable[[str, int], Iterable[str]],
    resolve_timeout: float | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Apply the transport and endpoint policy; return ``(transport kind, checked addresses)``.

    A host name is judged by the addresses it resolves to, unless the name itself
    is allowlisted (exact or ``*.suffix``), which accepts its private addresses.
    Plaintext ``http`` is probed only when every address is an allowed private one.
    The probe connects only to the returned addresses and never resolves the host
    name again, so a name that later resolves to a private or metadata address
    (DNS rebinding) cannot redirect the probe or its headers.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = parts.port
    except ValueError as exc:
        raise _ProbeRefused(STATUS_UNSUPPORTED, f"malformed URL: {exc}") from exc
    scheme = (parts.scheme or "").lower()
    kind_raw = (transport or "").strip().lower()
    if kind_raw in _SSE_TRANSPORTS:
        kind = "sse"
    elif kind_raw in _STREAMABLE_TRANSPORTS:
        kind = "streamable-http"
    else:
        raise _ProbeRefused(STATUS_UNSUPPORTED, f"transport {kind_raw!r} is not probed")
    if scheme in {"ws", "wss"}:
        raise _ProbeRefused(STATUS_UNSUPPORTED, "WebSocket MCP transport is not probed")
    if scheme not in {"https", "http"} or not host:
        raise _ProbeRefused(STATUS_UNSUPPORTED, f"scheme {scheme or '(none)'!r} is not probed")
    try:
        ipaddress.ip_address(host.strip("[]"))
        literal = True
    except ValueError:
        literal = False
    allowlist = HostAllowlist.of(allowed_private_hosts)
    label = f"host {host!r}"
    if literal:
        checked = [(host.strip("[]"), label)]
        name_allowed = False
    else:
        endpoint = classify_endpoint_host(host)
        if endpoint is not None and endpoint.kind == "metadata":
            raise _ProbeRefused(STATUS_DECLARED, f"not probed: {label} is a cloud metadata endpoint (never probed)")
        name_allowed = host_name_is_allowlisted(host, allowlist)
        try:
            addresses = _resolve_within(resolver, host, port or (443 if scheme == "https" else 80), resolve_timeout)
            addresses = addresses[:MAX_RESOLVED_ADDRESSES]
        except TimeoutError as exc:
            raise _ProbeRefused(STATUS_UNREACHABLE, str(exc) or "DNS resolution timed out") from exc
        except OSError as exc:
            raise _ProbeRefused(STATUS_UNREACHABLE, f"DNS resolution failed: {type(exc).__name__}") from exc
        if not addresses:
            raise _ProbeRefused(STATUS_UNREACHABLE, "DNS resolution returned no addresses")
        checked = [(str(address), f"{label} (resolved address {address})") for address in addresses]
    private = [
        _address_policy(address, allowlist, name_allowed=name_allowed, label=where) for address, where in checked
    ]
    if scheme == "http" and not all(private):
        raise _ProbeRefused(STATUS_DECLARED, "not probed: plaintext http is probed only for allowlisted private hosts")
    return kind, tuple(address for address, _where in checked)


def _default_resolver(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


def _capped_client(
    headers: dict[str, str],
    *,
    url: str,
    addresses: tuple[str, ...],
    max_bytes: int,
    connect_timeout: float,
    read_timeout: float,
    on_too_large: Callable[[str], None] | None = None,
):
    """An httpx client that never follows redirects, ignores proxy/.netrc settings, and caps bodies.

    Its TCP connections go only to ``addresses`` (the endpoint policy's checked
    addresses) and only for ``url``'s host and port. TLS still verifies the
    certificate against the host name, and requests keep their ``Host`` header.
    A body over ``max_bytes`` raises, and ``on_too_large`` is told first: the
    ``mcp`` client catches that error and would otherwise wait for a reply
    until the total timeout.
    """
    import httpcore
    import httpx

    target = httpx.URL(url)
    pinned_host = target.raw_host.decode("ascii").lower()
    pinned_port = target.port or (443 if target.scheme == "https" else 80)

    class _PinnedBackend(httpcore.AsyncNetworkBackend):
        """Connect only to the checked addresses; the host name is never resolved again."""

        def __init__(self) -> None:
            self._inner = httpcore.AnyIOBackend()

        async def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: float | None = None,
            local_address: str | None = None,
            socket_options: Any = None,
        ) -> Any:
            if host.lower() != pinned_host or port != pinned_port:
                raise httpcore.ConnectError(f"refused: {host}:{port} is not the policy-checked endpoint")
            failure: Exception = httpcore.ConnectError("no policy-checked address")
            for address in addresses:
                try:
                    return await self._inner.connect_tcp(
                        address, port, timeout=timeout, local_address=local_address, socket_options=socket_options
                    )
                except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                    failure = exc
            raise failure

        async def sleep(self, seconds: float) -> None:
            await self._inner.sleep(seconds)

    class _ResponseTooLarge(httpx.HTTPError):
        pass

    class _CappedStream(httpx.AsyncByteStream):
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        async def __aiter__(self):
            total = 0
            async for chunk in self._inner:
                total += len(chunk)
                if total > max_bytes:
                    message = f"response exceeded {max_bytes} bytes"
                    if on_too_large is not None:
                        on_too_large(message)
                    raise _ResponseTooLarge(message)
                yield chunk

        async def aclose(self) -> None:
            await self._inner.aclose()

    class _CappedTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self._inner = httpx.AsyncHTTPTransport(retries=0)
            # httpx exposes no network-backend hook: swap its connection pool for
            # a pinned one, and refuse to run (fail closed) if the layout changed.
            if not isinstance(getattr(self._inner, "_pool", None), httpcore.AsyncConnectionPool):
                raise RuntimeError("cannot pin the probe connection with this httpx version")
            self._inner._pool = httpcore.AsyncConnectionPool(
                ssl_context=httpx.create_ssl_context(),
                retries=0,
                network_backend=_PinnedBackend(),
            )

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            response = await self._inner.handle_async_request(request)
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                stream=_CappedStream(response.stream),
                extensions=response.extensions,
            )

        async def aclose(self) -> None:
            await self._inner.aclose()

    return httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        follow_redirects=False,
        trust_env=False,
        transport=_CappedTransport(),
    )


def _root_cause(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


class _ProbeAbort:
    """The first fatal error of one probe; recording it cancels the probe's scope."""

    def __init__(self) -> None:
        self.reason = ""
        self.scope: Any = None

    def fail(self, reason: str) -> None:
        if not self.reason:
            self.reason = reason or "transport error"
        if self.scope is not None:
            self.scope.cancel()


class _McpErrorCapture(logging.Handler):
    """Turns an error the ``mcp`` client logs (and then swallows) on the probe's thread into a probe failure."""

    def __init__(self, abort: _ProbeAbort) -> None:
        super().__init__(logging.ERROR)
        self._abort = abort
        self._thread = threading.get_ident()

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self._thread:
            return
        exc = record.exc_info[1] if record.exc_info else None
        self._abort.fail(f"{type(exc).__name__}: {exc}" if exc is not None else record.getMessage())


@contextmanager
def _quiet_mcp_logs(abort: _ProbeAbort) -> Iterator[None]:
    """Keep ``mcp`` log records (tracebacks included) off the console during a probe; errors fail it."""
    handler = _McpErrorCapture(abort)
    saved: list[tuple[logging.Logger, int, bool]] = []
    for name in _MCP_LOGGER_NAMES:
        logger = logging.getLogger(name)
        saved.append((logger, logger.level, logger.propagate))
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
    try:
        yield
    finally:
        for logger, level, propagate in saved:
            logger.removeHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate


async def _probe_session(
    url: str, kind: str, headers: dict[str, str], *, addresses: tuple[str, ...], total_timeout: float
) -> list[str]:
    import anyio
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamable_http_client

    abort = _ProbeAbort()

    def factory(headers=None, timeout=None, auth=None):
        return _capped_client(
            dict(headers or {}),
            url=url,
            addresses=addresses,
            max_bytes=MAX_RESPONSE_BYTES,
            connect_timeout=CONNECT_TIMEOUT_S,
            read_timeout=READ_TIMEOUT_S,
            on_too_large=abort.fail,
        )

    listed: Any = None
    with _quiet_mcp_logs(abort), anyio.fail_after(total_timeout):
        try:
            with anyio.CancelScope() as scope:
                abort.scope = scope
                if kind == "sse":
                    context = sse_client(
                        url,
                        headers=headers,
                        timeout=READ_TIMEOUT_S,
                        sse_read_timeout=READ_TIMEOUT_S,
                        httpx_client_factory=factory,
                    )
                    async with context as (read, write), ClientSession(read, write) as session:
                        await session.initialize()
                        listed = await session.list_tools()
                else:
                    async with (
                        factory(headers) as client,
                        streamable_http_client(url, http_client=client) as (read, write, _session_id),
                        ClientSession(read, write) as session,
                    ):
                        await session.initialize()
                        listed = await session.list_tools()
        except BaseException as exc:
            if abort.reason and not isinstance(exc, KeyboardInterrupt | SystemExit):
                raise _ProbeFailed(abort.reason) from None
            raise
        if abort.reason and (scope.cancelled_caught or listed is None):
            raise _ProbeFailed(abort.reason)
    names: list[str] = []
    for tool in getattr(listed, "tools", None) or []:
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name and len(names) < MAX_PROBE_TOOLS:
            names.append(_detail(name)[:MAX_TOOL_NAME_CHARS])
    return names


def probe_mcp_server(
    target: Mapping[str, Any],
    *,
    allowed_private_hosts: HostAllowlist | Iterable[str] = (),
    expand_env: Iterable[str | EnvGrant] = (),
    environ: Mapping[str, str] | None = None,
    resolver: Callable[[str, int], Iterable[str]] | None = None,
    total_timeout: float = TOTAL_TIMEOUT_S,
) -> dict[str, Any]:
    """Probe one URL MCP server; return ``{"status", "tools", "detail"}``.

    ``expand_env`` holds the user's ``--probe-mcp-env`` opt-ins: ``NAME`` (any
    server), ``NAME=HOST`` (servers whose URL host is HOST) or ``NAME@SERVER``
    (that server). Only names opted in for this server are read from
    ``environ`` (the process environment when ``None``) into its declared
    headers, and the detail records which names were sent to which host.
    ``total_timeout`` bounds the whole probe, DNS lookup included.
    """
    url = str(target.get("url") or "")
    started = time.monotonic()
    try:
        kind, addresses = _check_endpoint(
            url,
            str(target.get("transport") or ""),
            allowed_private_hosts,
            resolver or _default_resolver,
            resolve_timeout=total_timeout,
        )
    except _ProbeRefused as refused:
        return _entry(refused.status, refused.detail)
    raw_headers = target.get("headers")
    headers, missing, withheld, sent = _resolve_headers(
        raw_headers if isinstance(raw_headers, Mapping) else {},
        os.environ if environ is None else environ,
        _allowed_env(target, [parse_env_grant(value) for value in expand_env]),
    )
    note = _header_note(missing, withheld, sent, _target_host(url))
    try:
        import anyio
        import httpx  # noqa: F401 -- availability check
        import mcp  # noqa: F401 -- availability check
    except ImportError:
        return _entry(
            STATUS_DECLARED, "not probed: the mcp client library is not installed (install skillevaluator[tier3])"
        )
    remaining = total_timeout - (time.monotonic() - started)
    if remaining <= 0:
        return _entry(STATUS_UNREACHABLE, f"{kind} probe failed: timed out after {total_timeout:g}s")
    try:
        tools = anyio.run(lambda: _probe_session(url, kind, headers, addresses=addresses, total_timeout=remaining))
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt | SystemExit):
            raise
        cause = _root_cause(exc)
        message = str(cause) or type(cause).__name__
        if isinstance(cause, _ProbeFailed):
            return _entry(STATUS_UNREACHABLE, f"{kind} probe failed: {message}{note}")
        if isinstance(cause, TimeoutError):
            message = f"timed out after {total_timeout:g}s"
        return _entry(STATUS_UNREACHABLE, f"{kind} probe failed: {type(cause).__name__}: {message}{note}")
    return _entry(STATUS_REACHABLE_HOST, f"initialize and tools/list succeeded over {kind}{note}", tools)


def probe_mcp_servers(
    targets: Iterable[Mapping[str, Any]],
    *,
    allowed_private_hosts: HostAllowlist | Iterable[str] = (),
    expand_env: Iterable[str | EnvGrant] = (),
    environ: Mapping[str, str] | None = None,
    resolver: Callable[[str, int], Iterable[str]] | None = None,
    total_timeout: float = TOTAL_TIMEOUT_S,
) -> dict[str, Any]:
    """Probe every author-supplied URL MCP server (bounded to 32 servers)."""
    allowed = HostAllowlist.of(allowed_private_hosts)
    opted_in = tuple(parse_env_grant(value) for value in expand_env)
    proof: dict[str, Any] = {}
    for target in list(targets)[:MAX_PROBE_SERVERS]:
        name = str(target.get("name") or "").strip()
        if not name:
            continue
        proof[name] = probe_mcp_server(
            target,
            allowed_private_hosts=allowed,
            expand_env=opted_in,
            environ=environ,
            resolver=resolver,
            total_timeout=total_timeout,
        )
    return proof


def _with_plugin_mcp_servers(
    engine_result: Mapping[str, Any] | None, declared: Iterable[str]
) -> dict[str, dict[str, int]]:
    """Per-declared-server call counts summed over every agent's with-plugin arm.

    Each ``by_server`` key maps to at most one declared server through the
    plugin-signals name rules (exact name first; a harness spelling only when it
    names exactly one declared server). Keys that name no declared server are
    dropped, so one server's calls never credit another.
    """
    totals: dict[str, dict[str, int]] = {}
    agents = engine_result.get("agents") if isinstance(engine_result, Mapping) else None
    if not isinstance(agents, Mapping):
        return totals
    # Index the declared servers once for every agent's lookups.
    match_server = declared_mcp_server_matcher(str(name) for name in declared)
    for agent in agents.values():
        summaries = agent.get("plugin_signals_summary") if isinstance(agent, Mapping) else None
        arm = summaries.get("with_skill") if isinstance(summaries, Mapping) else None
        mcp_calls = arm.get("mcp_calls") if isinstance(arm, Mapping) else None
        by_server = mcp_calls.get("by_server") if isinstance(mcp_calls, Mapping) else None
        if not isinstance(by_server, Mapping):
            continue
        for server, counts in by_server.items():
            if not isinstance(counts, Mapping):
                continue
            target = match_server(str(server))
            if target is None:
                continue
            bucket = totals.setdefault(target, {"total": 0, "succeeded": 0})
            for key in bucket:
                value = counts.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    bucket[key] += value
    return totals


def apply_in_agent_mcp_proof(mcp_proof: Mapping[str, Any], engine_result: Mapping[str, Any] | None) -> dict[str, Any]:
    """Upgrade each server's status from with-plugin MCP call evidence (never downgrades)."""
    calls = _with_plugin_mcp_servers(engine_result, mcp_proof)
    upgraded: dict[str, Any] = {}
    for name, raw in mcp_proof.items():
        entry = dict(raw) if isinstance(raw, Mapping) else _entry(STATUS_DECLARED, "")
        counts = calls.get(str(name))
        status = entry.get("status")
        rank = _STATUS_RANK.get(status, -1) if isinstance(status, str) else -1
        if counts and counts["total"] > 0:
            host_detail = str(entry.get("detail") or "")
            new_status = STATUS_USED_SUCCESSFULLY if counts["succeeded"] > 0 else STATUS_REACHABLE_IN_AGENT
            if _STATUS_RANK[new_status] > rank:
                entry["status"] = new_status
                agent_detail = (
                    f"agent made {counts['total']} call(s), {counts['succeeded']} succeeded, in the with-plugin arm"
                )
                entry["detail"] = _detail(f"{agent_detail}; host probe: {host_detail}" if host_detail else agent_detail)
        upgraded[str(name)] = entry
    return upgraded


__all__ = [
    "MCP_PROOF_STATUSES",
    "NOT_REQUESTED_DETAIL",
    "EnvGrant",
    "apply_in_agent_mcp_proof",
    "declared_mcp_proof",
    "parse_env_grant",
    "planned_env_sends",
    "probe_mcp_server",
    "probe_mcp_servers",
]
