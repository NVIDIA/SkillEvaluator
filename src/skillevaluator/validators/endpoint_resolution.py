# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private, explicitly invoked DNS and single-HEAD endpoint checks.

Imports and construction do no networking. The checker classifies all initial
DNS answers before requests, pins each request to a classified public address,
and classifies one redirect without following it. Private endpoints are never
contacted, including allowlisted endpoints. Missing evidence is incomplete.
"""

from __future__ import annotations

import http.client
import ipaddress
import queue
import socket
import ssl
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from skillevaluator.models.result import Finding, Severity
from skillevaluator.validators.mcp_static import (
    EndpointClass,
    HostAllowlist,
    classify_endpoint_address,
    classify_endpoint_host,
    endpoint_client_host,
)
from skillevaluator.validators.url_policy import DEFAULT_PORTS, safe_url, whatwg_url

MAX_ENDPOINTS = 64
MAX_TARGETS = 4096
MAX_ADDRESSES = 64
DNS_TIMEOUT_SECONDS = 3.0
HEAD_TIMEOUT_SECONDS = 5.0
TOTAL_BUDGET_SECONDS = 60.0
INCOMPLETE_SCAN = "endpoint-resolution"


@dataclass(frozen=True)
class EndpointTarget:
    """A raw URL and the component that declared it; raw URLs are never reported."""

    url: str
    kind: str
    name: str
    file_path: str
    component: tuple[str, str] | None = None
    allowed_hosts: tuple[str, ...] = ()


@dataclass(frozen=True)
class HeadResult:
    status: int | None
    location: str | None
    error: str | None = None


Resolver = Callable[[str, int, float], list[str]]
HeadRequester = Callable[[str, str, int, str, str, float], HeadResult]


def _resolve(host: str, port: int, timeout: float) -> list[str]:
    """Bound even a hung system resolver; the worker cannot delay process exit."""
    if timeout <= 0:
        raise TimeoutError("DNS deadline exhausted")
    answer: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def lookup() -> None:
        try:
            rows = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
            addresses = list(dict.fromkeys(str(row[4][0]).split("%", 1)[0] for row in rows))
            answer.put((True, addresses[: MAX_ADDRESSES + 1]))
        except (OSError, UnicodeError, ValueError) as exc:
            answer.put((False, exc))

    threading.Thread(target=lookup, name="skillevaluator-dns", daemon=True).start()
    try:
        ok, value = answer.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError("DNS deadline exhausted") from None
    if not ok:
        raise value
    return value


def _head(scheme: str, host: str, port: int, address: str, path: str, timeout: float) -> HeadResult:
    """One direct-IP request, with verified TLS and an exchange-wide deadline.

    http.client does not consult proxy environment variables, netrc, cookies or
    ambient authorization. The caller has already removed URL userinfo, query,
    and fragment, and only supplies a classified public numeric address.
    """
    deadline = time.monotonic() + timeout
    sock = socket.create_connection((address, port), timeout=timeout)
    stream = sock
    connection: http.client.HTTPConnection | None = None
    lock = threading.Lock()
    state = {"open": True}
    expired = threading.Event()
    fd, family = sock.fileno(), sock.family

    def expire() -> None:
        with lock:
            if not state["open"]:
                return
            expired.set()
            try:
                with socket.fromfd(fd, family, socket.SOCK_STREAM) as duplicate:
                    duplicate.shutdown(socket.SHUT_RDWR)
            except (OSError, ValueError, AttributeError):
                pass

    timer = threading.Timer(max(0.0, deadline - time.monotonic()), expire)
    timer.daemon = True
    timer.start()
    try:
        if scheme in {"https", "wss"}:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            stream = context.wrap_socket(sock, server_hostname=host, do_handshake_on_connect=False)
            stream.do_handshake()
            connection = http.client.HTTPSConnection(host, port, timeout=timeout, context=context)
        else:
            connection = http.client.HTTPConnection(host, port, timeout=timeout)
        connection.sock = stream
        connection.request(
            "HEAD",
            path or "/",
            headers={"User-Agent": "skillevaluator-endpoint-check", "Accept": "*/*", "Connection": "close"},
        )
        response = connection.getresponse()
        status, location = response.status, response.getheader("Location")
        response.close()
        if expired.is_set() or time.monotonic() > deadline:
            raise TimeoutError("HEAD deadline exhausted")
        # Truncating a Location could change its host. Refuse it instead.
        if location is not None and len(location) > 2048:
            return HeadResult(None, None, "oversize Location header")
        return HeadResult(status, location)
    finally:
        timer.cancel()
        with lock:
            state["open"] = False
            if connection is not None:
                connection.close()
            stream.close()
            sock.close()


def _endpoint(url: str, base: str | None = None) -> tuple[str, str, int, str]:
    parsed = urlsplit(whatwg_url(url, base))
    scheme = parsed.scheme.lower()
    host = endpoint_client_host(parsed.hostname or "")
    port = parsed.port or DEFAULT_PORTS.get(scheme, 0)
    if scheme not in DEFAULT_PORTS or not host or "$" in host or not 1 <= port <= 65535:
        raise ValueError("unsupported or unusable endpoint")
    return scheme, host, port, parsed.path or "/"


def _finding(target: EndpointTarget, check: str, message: str, severity: Severity = Severity.MEDIUM) -> Finding:
    return Finding(
        category="PLUGIN_MCP" if target.kind == "mcp" else "PLUGIN_SCHEMA",
        severity=severity,
        check_name=check,
        message=message,
        file_path=target.file_path,
        suggestion="Use a public endpoint and rerun the explicitly requested endpoint checks.",
        metadata={
            "name": target.name,
            "kind": target.kind,
            **(
                {"component_type": target.component[0], "component_name": target.component[1]}
                if target.component
                else {}
            ),
        },
    )


class EndpointChecker:
    """Network collector used only by an explicit private caller."""

    def __init__(
        self,
        *,
        resolver: Resolver | None = None,
        head: HeadRequester | None = None,
        dns_timeout: float = DNS_TIMEOUT_SECONDS,
        head_timeout: float = HEAD_TIMEOUT_SECONDS,
        budget: float = TOTAL_BUDGET_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.resolver = resolver or _resolve
        self.head = head or _head
        self.dns_timeout, self.head_timeout, self.budget, self.clock = dns_timeout, head_timeout, budget, clock

    def check(self, targets: Iterable[EndpointTarget]) -> tuple[dict[str, Any], list[Finding]]:
        deadline = self.clock() + self.budget
        rows: list[dict[str, Any]] = []
        findings: list[Finding] = []
        reasons: set[str] = set()
        pending = []
        seen = set()

        def classify(target: EndpointTarget, host: str, port: int, *, redirect: bool = False) -> tuple[str, list[str]]:
            prefix = "endpoint_redirect" if redirect else "endpoint_resolves"
            allowed = HostAllowlist.from_entries(target.allowed_hosts)
            static = classify_endpoint_host(host)
            if static is not None:
                addresses = [str(static.address)] if static.address else []
                nonpublic = [static]
            else:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise TimeoutError("time budget exhausted")
                answers = self.resolver(host, port, min(self.dns_timeout, remaining))
                if self.clock() > deadline:
                    raise TimeoutError("DNS exceeded total deadline")
                if not isinstance(answers, (list, tuple)) or not answers or len(answers) > MAX_ADDRESSES:
                    raise ValueError("DNS answer missing or exceeds bound")
                addresses = []
                nonpublic = []
                for answer in answers:
                    if not isinstance(answer, str):
                        raise ValueError("invalid DNS address type")
                    address = ipaddress.ip_address(answer)
                    addresses.append(str(address))
                    found = classify_endpoint_address(address)
                    if found:
                        nonpublic.append(EndpointClass(found[0], found[1], host, address))
            kind = (
                "metadata" if any(ep.kind == "metadata" for ep in nonpublic) else "private" if nonpublic else "public"
            )
            blocked = next((ep for ep in nonpublic if ep.kind == "metadata"), None)
            if blocked is None:
                blocked = next((ep for ep in nonpublic if not allowed.allows(ep)), None)
            if blocked:
                findings.append(
                    _finding(
                        target,
                        prefix + ("_metadata" if blocked.kind == "metadata" else "_private"),
                        f"Declared endpoint {safe_url(target.url)!r} targets a {blocked.kind} address; it was not contacted.",
                        Severity.HIGH if blocked.kind == "metadata" else Severity.MEDIUM,
                    )
                )
            return kind, addresses

        for index, target in enumerate(targets):
            if index >= MAX_TARGETS:
                reasons.add("target enumeration bound exceeded")
                break
            key = (target.kind, target.name, target.file_path, target.url, target.component, target.allowed_hosts)
            if key in seen:
                continue
            seen.add(key)
            row: dict[str, Any] = {
                "kind": target.kind,
                "name": target.name,
                "file_path": target.file_path,
                "url": safe_url(target.url),
            }
            rows.append(row)
            if len(rows) > MAX_ENDPOINTS:
                row.update(status="skipped", reason="endpoint cap")
                reasons.add("endpoint cap exceeded")
                continue
            try:
                scheme, host, port, path = _endpoint(target.url)
                row["host"] = host
                kind, addresses = classify(target, host, port)
                row.update(status=kind, classification=kind, addresses=addresses)
                if kind == "public":
                    pending.append((target, row, scheme, host, port, addresses[0], path))
                else:
                    row["head"] = {"skipped": "non-public endpoint"}
            except (OSError, UnicodeError, ValueError, TypeError) as exc:
                row.update(status="unresolved", reason=type(exc).__name__)
                reasons.add("one or more endpoints could not be classified")
        for target, row, scheme, host, port, address, path in pending:
            remaining = deadline - self.clock()
            if remaining <= 0:
                row.update(status="skipped", reason="time budget exhausted")
                reasons.add("time budget exhausted")
                continue
            try:
                response = self.head(scheme, host, port, address, path, min(self.head_timeout, remaining))
                if (
                    not isinstance(response, HeadResult)
                    or response.error
                    or isinstance(response.status, bool)
                    or not isinstance(response.status, int)
                    or not 100 <= response.status <= 599
                    or self.clock() > deadline
                ):
                    raise ValueError("no trustworthy HEAD evidence")
                row["head"] = {"status": response.status}
                if response.location is not None and not isinstance(response.location, str):
                    raise ValueError("invalid Location header")
                if response.location:
                    if not isinstance(response.location, str) or len(response.location) > 2048:
                        raise ValueError("invalid Location header")
                    absolute = whatwg_url(response.location, whatwg_url(target.url).split("?", 1)[0].split("#", 1)[0])
                    redirect: dict[str, Any] = {"url": safe_url(absolute)}
                    row["redirect"] = redirect
                    new_scheme, new_host, new_port, _ = _endpoint(absolute)
                    if scheme in {"https", "wss"} and new_scheme in {"http", "ws"}:
                        findings.append(
                            _finding(
                                target,
                                "endpoint_redirect_insecure_scheme",
                                "The endpoint redirects from encrypted transport to plaintext.",
                            )
                        )
                    kind, addresses = classify(target, new_host, new_port, redirect=True)
                    redirect.update(classification=kind, addresses=addresses)
            except (OSError, UnicodeError, ValueError, TypeError, http.client.HTTPException) as exc:
                row.update(status="incomplete", reason=type(exc).__name__)
                reasons.add("one or more HEAD or redirect checks produced no evidence")
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        summary: dict[str, Any] = {"enabled": True, "endpoints": rows, "counts": counts}
        if reasons:
            summary.update(incomplete=True, incomplete_reasons=sorted(reasons))
            findings.append(
                Finding(
                    category="PLUGIN_SCHEMA",
                    severity=Severity.MEDIUM,
                    check_name="endpoint_resolution_incomplete",
                    message="Endpoint checks incomplete: " + "; ".join(sorted(reasons)),
                    file_path=rows[0]["file_path"] if rows else "",
                    metadata={"reasons": sorted(reasons)},
                )
            )
        return summary, findings
