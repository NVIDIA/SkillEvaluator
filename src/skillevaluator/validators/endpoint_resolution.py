# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in DNS and redirect checks for MCP server and HTTP hook endpoints.

The static endpoint policy (:mod:`skillevaluator.validators.mcp_static`) sees IP
literals and well-known names only. With ``validate --resolve-endpoints`` (or
``endpoints.resolve: true`` in the policy), each MCP ``url`` and HTTP hook
``url`` is also checked on the network:

1. Every host name is resolved first (bounded timeout), before any request is
   sent. A name that resolves to a cloud-metadata, private, loopback,
   link-local, or other special-purpose address is flagged with the same
   address classifier the static policy uses. Every address in the answer is
   classified (up to 64; a longer answer counts as non-public); the policy
   allowlist must cover each non-public one.
2. When every resolved address is public, exactly one ``HEAD`` request is sent to
   the first address with no redirect following, no credentials, no query string,
   and a wall-clock deadline. A ``Location`` header is recorded, and its target is
   classified (statically, then by DNS) the same way. The redirect is never
   requested.

URLs are read the way WHATWG clients (Node, the MCP SDKs) read them, so a
backslash, a missing ``//``, or a percent-encoded host name cannot hide the real
host. A host that resolves to a non-public address is never contacted.
Endpoints left unchecked by the endpoint cap, the time budget, or unavailable
DNS make the result incomplete.
Default validation never imports the network path: nothing here runs unless
enabled.
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
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from skillevaluator.models.result import Finding, Severity
from skillevaluator.validators.mcp_static import (
    CATEGORY as MCP_CATEGORY,
)
from skillevaluator.validators.mcp_static import (
    EndpointClass,
    classify_endpoint_address,
    classify_endpoint_host,
    endpoint_client_host,
    host_is_allowlisted,
)
from skillevaluator.validators.url_policy import whatwg_url

DNS_TIMEOUT_SECONDS = 3.0
HEAD_TIMEOUT_SECONDS = 5.0
TOTAL_BUDGET_SECONDS = 60.0
MAX_ENDPOINTS = 64
# Every DNS answer up to this many is classified; a longer answer is treated as non-public.
MAX_ADDRESSES = 64
USER_AGENT = "skillevaluator-endpoint-check"
PLUGIN_CATEGORY = "PLUGIN_SCHEMA"
# ``incomplete_scans`` name when endpoints were left unchecked.
INCOMPLETE_SCAN = "endpoint-resolution"
_DEFAULT_PORTS = {"http": 80, "ws": 80, "https": 443, "wss": 443}

Resolver = Callable[[str, int, float], list[str]]
HeadRequester = Callable[[str, str, int, str, str, float], "HeadResult"]


@dataclass(frozen=True)
class EndpointTarget:
    """One URL to check, and where to attribute its findings."""

    url: str
    kind: str  # "mcp" | "hook"
    name: str  # MCP server name or hook id
    file_path: str
    component: tuple[str, str] | None = None
    allowed_hosts: tuple[str, ...] = ()


@dataclass(frozen=True)
class HeadResult:
    status: int | None
    location: str | None
    error: str | None = None


def _resolve(host: str, port: int, timeout: float) -> list[str]:
    """Resolve *host* to every unique IP string within *timeout* seconds (raises on failure).

    The lookup runs in a daemon thread: ``getaddrinfo`` cannot be cancelled, and a
    lookup that hangs past the timeout must not keep the process from exiting.
    """
    answer: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def _lookup() -> None:
        try:
            answer.put((True, socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)))
        except BaseException as exc:  # handed to the caller below
            answer.put((False, exc))

    threading.Thread(target=_lookup, name="skillevaluator-dns", daemon=True).start()
    try:
        ok, value = answer.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(f"DNS lookup timed out after {timeout:.1f}s") from None
    if not ok:
        raise value
    addresses: list[str] = []
    for info in value:
        address = str(info[4][0]).split("%", 1)[0]
        if address not in addresses:
            addresses.append(address)
    return addresses


def _head(scheme: str, host: str, port: int, address: str, path: str, timeout: float) -> HeadResult:
    """Send one credential-free ``HEAD`` to *address* (TLS verified for *host*); never follow redirects.

    The whole exchange (connect, TLS handshake, request, and response headers)
    has a wall-clock deadline of *timeout* seconds. A socket timeout alone only
    bounds each read, so a server that trickles one byte at a time could hold the
    request open for hours; at the deadline a watchdog shuts the connection down.
    """
    deadline = time.monotonic() + timeout
    sock = socket.create_connection((address, port), timeout=timeout)
    stream: socket.socket = sock
    connection: http.client.HTTPConnection | None = None
    lock = threading.Lock()
    expired = threading.Event()
    state = {"open": True}
    fd = sock.fileno()
    family = sock.family

    def _expire() -> None:
        with lock:  # never touch the descriptor after it is closed (and possibly reused)
            if not state["open"]:
                return
            expired.set()
            try:
                # A duplicate of the descriptor reaches the connection even after TLS wraps the socket.
                with socket.fromfd(fd, family, socket.SOCK_STREAM) as duplicate:
                    duplicate.shutdown(socket.SHUT_RDWR)
            except (OSError, ValueError, AttributeError):
                pass  # the per-read socket timeout still bounds each read

    watchdog = threading.Timer(max(0.0, deadline - time.monotonic()), _expire)
    watchdog.daemon = True
    watchdog.start()
    try:
        if scheme in {"https", "wss"}:
            context = ssl.create_default_context()
            # Never negotiate TLS 1.0 or 1.1, whatever the local OpenSSL defaults allow.
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
            headers={"User-Agent": USER_AGENT, "Accept": "*/*", "Connection": "close"},
        )
        response = connection.getresponse()
        location = response.getheader("Location")
        status = response.status
        response.close()
        if expired.is_set():
            # The watchdog cut the headers short; a Location header may be missing, so this is no evidence.
            raise TimeoutError(f"no complete response within {timeout:.1f}s")
        return HeadResult(status, location[:2048] if location else None)
    except (OSError, ssl.SSLError, http.client.HTTPException, ValueError) as exc:
        if expired.is_set():
            raise TimeoutError(f"no complete response within {timeout:.1f}s") from exc
        raise
    finally:
        watchdog.cancel()
        with lock:
            state["open"] = False
            if connection is not None:
                connection.close()
            stream.close()
            sock.close()


def _classify_addresses(addresses: Iterable[str]) -> list[tuple[str, str, str | None]]:
    """Return every non-public address as ``(kind, reason, address)`` (``metadata`` or ``private``), in order.

    At most :data:`MAX_ADDRESSES` answers are classified. A longer answer adds one
    ``private`` row with no address: the unclassified answers count as non-public,
    because a client may connect to any of them.
    """
    answers = list(addresses)
    non_public: list[tuple[str, str, str | None]] = []
    for raw in answers[:MAX_ADDRESSES]:
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            continue
        found = classify_endpoint_address(address)
        if found is not None:
            non_public.append((found[0], found[1], raw))
    if len(answers) > MAX_ADDRESSES:
        non_public.append(("private", f"more than {MAX_ADDRESSES} addresses, so some were not classified", None))
    return non_public


def _worst_kind(non_public: list[tuple[str, str, str | None]]) -> str:
    """``metadata`` > ``private`` > ``public``."""
    if any(kind == "metadata" for kind, _reason, _address in non_public):
        return "metadata"
    return "private" if non_public else "public"


def _address_allowed(host: str, reason: str, address: str | None, allowed_hosts: Iterable[str]) -> bool:
    parsed = None
    if address is not None:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            parsed = None
    endpoint = EndpointClass("private", reason, host.lower().rstrip("."), parsed)
    return host_is_allowlisted(endpoint, allowed_hosts)


def blocked_address(
    host: str, non_public: list[tuple[str, str, str | None]], allowed_hosts: Iterable[str]
) -> tuple[str, str, str | None] | None:
    """The metadata address, else the first non-public address the allowlist does not cover, else ``None``.

    Every answer is checked: a client may connect to any of them, and the answer order is attacker-chosen.
    Metadata addresses are never allowlisted.
    """
    allowed = tuple(allowed_hosts)
    for row in non_public:
        if row[0] == "metadata":
            return row
    for kind, reason, address in non_public:
        if not _address_allowed(host, reason, address, allowed):
            return kind, reason, address
    return None


def _where(blocked: tuple[str, str, str | None]) -> str:
    """``a <reason> address`` for a finding message (the reason alone for an over-long answer)."""
    return f"a {blocked[1]} address" if blocked[2] is not None else blocked[1]


def _safe_url(url: str) -> str:
    from skillevaluator.plugin_component_risk import safe_url

    return safe_url(url)


@dataclass
class _Run:
    """Bookkeeping for one :meth:`EndpointChecker.check` call."""

    deadline: float
    budget_skipped: int = 0
    cap_skipped: int = 0
    lookups: int = 0
    resolved: int = 0
    timed_out: int = 0
    unchecked: list[EndpointTarget] = field(default_factory=list)
    first_file: str = ""


@dataclass
class _Pending:
    """A public endpoint whose ``HEAD`` is sent after every name has been resolved."""

    target: EndpointTarget
    row: dict[str, Any]
    scheme: str
    host: str
    port: int
    address: str
    path: str


class EndpointChecker:
    """Resolve endpoints and record one redirect hop, with bounded time and no credentials."""

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
        self.dns_timeout = dns_timeout
        self.head_timeout = head_timeout
        self.budget = budget
        self.clock = clock

    def check(self, targets: Iterable[EndpointTarget]) -> tuple[dict[str, Any], list[Finding]]:
        """Check every target: all DNS first, then one ``HEAD`` per public endpoint.

        Resolving every name before any request means a slow endpoint cannot use
        up the time budget before a later endpoint's addresses are classified. The
        summary has ``incomplete: true`` (with reasons, and one MEDIUM
        ``endpoint_resolution_incomplete`` finding) when the endpoint cap or the
        time budget left endpoints unchecked, a DNS lookup timed out, or no host
        resolved at all.
        """
        run = _Run(deadline=self.clock() + self.budget)
        rows: list[dict[str, Any]] = []
        findings: list[Finding] = []
        pending: list[_Pending] = []
        seen: set[tuple[str, str]] = set()
        checked = 0
        for target in targets:
            key = (target.kind, target.url)
            if key in seen:
                continue
            seen.add(key)
            run.first_file = run.first_file or target.file_path
            if checked >= MAX_ENDPOINTS:
                rows.append({"kind": target.kind, "name": target.name, "status": "skipped", "reason": "endpoint cap"})
                run.cap_skipped += 1
                run.unchecked.append(target)
                continue
            checked += 1
            if self._remaining(run) <= 0:
                rows.append(self._budget_row(target))
                run.budget_skipped += 1
                run.unchecked.append(target)
                continue
            row, row_findings, head = self._classify(target, run)
            rows.append(row)
            findings.extend(row_findings)
            if head is not None:
                pending.append(head)
        for item in pending:
            findings.extend(self._head_and_redirect(item, run))
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        summary: dict[str, Any] = {"enabled": True, "endpoints": rows, "counts": dict(sorted(counts.items()))}
        reasons = self._incomplete_reasons(run)
        if reasons:
            summary["incomplete"] = True
            summary["incomplete_reasons"] = reasons
            findings.append(self._incomplete_finding(run, rows, reasons))
        return summary, findings

    # -- bookkeeping -------------------------------------------------------- #
    def _remaining(self, run: _Run) -> float:
        return run.deadline - self.clock()

    def _budget_row(self, target: EndpointTarget) -> dict[str, Any]:
        return {
            "kind": target.kind,
            "name": target.name,
            "url": _safe_url(target.url),
            "status": "skipped",
            "reason": f"time budget of {self.budget:.0f}s exhausted",
        }

    def _incomplete_reasons(self, run: _Run) -> list[str]:
        reasons: list[str] = []
        if run.cap_skipped:
            reasons.append(f"{run.cap_skipped} endpoint(s) past the {MAX_ENDPOINTS}-endpoint cap were not checked")
        if run.budget_skipped:
            reasons.append(f"{run.budget_skipped} endpoint(s) were not checked within the {self.budget:.0f}s budget")
        if run.timed_out:
            reasons.append(f"DNS lookups timed out for {run.timed_out} host(s)")
        if run.lookups and not run.resolved and not run.timed_out:
            reasons.append("no endpoint host could be resolved (DNS may be unavailable)")
        return reasons

    def _incomplete_finding(self, run: _Run, rows: list[dict[str, Any]], reasons: list[str]) -> Finding:
        unchecked = run.unchecked
        file_path = unchecked[0].file_path if unchecked else run.first_file
        return Finding(
            category=PLUGIN_CATEGORY,
            severity=Severity.MEDIUM,
            check_name="endpoint_resolution_incomplete",
            message=f"--resolve-endpoints left endpoints unchecked: {'; '.join(reasons)}",
            file_path=file_path,
            suggestion="Rerun --resolve-endpoints where DNS is reachable, or reduce the number of declared endpoints.",
            metadata={"reasons": reasons, "endpoints": len(rows), "unchecked": len(unchecked)},
        )

    # -- one endpoint ------------------------------------------------------- #
    def _finding(
        self, target: EndpointTarget, severity: Severity, check: str, message: str, suggestion: str
    ) -> Finding:
        if target.kind == "mcp":
            category = MCP_CATEGORY
            metadata: dict[str, Any] = {"mcp_server": target.name}
            prefix = f"mcpServers['{target.name}']: "
        else:
            category = PLUGIN_CATEGORY
            metadata = {"hook_id": target.name}
            if target.component is not None:
                metadata["plugin_component"] = {"type": target.component[0], "name": target.component[1]}
            prefix = f"hook {target.name}: "
        return Finding(
            category=category,
            severity=severity,
            check_name=check,
            message=prefix + message,
            file_path=target.file_path,
            suggestion=suggestion,
            metadata=metadata,
        )

    def _lookup(self, host: str, port: int, run: _Run) -> list[str]:
        """Resolve within the DNS timeout and the remaining budget; counts the lookup for the summary."""
        run.lookups += 1
        try:
            addresses = self.resolver(host, port, max(0.0, min(self.dns_timeout, self._remaining(run))))
        except TimeoutError:
            run.timed_out += 1
            raise
        run.resolved += 1
        return addresses

    def _classify(self, target: EndpointTarget, run: _Run) -> tuple[dict[str, Any], list[Finding], _Pending | None]:
        """Parse, classify statically, and resolve one endpoint; return the ``HEAD`` to send, if any."""
        findings: list[Finding] = []
        display_url = _safe_url(target.url)
        row: dict[str, Any] = {"kind": target.kind, "name": target.name, "url": display_url}
        try:
            # Read the URL the way the client that connects to it does (WHATWG).
            parsed = urlparse(whatwg_url(target.url))
            host = parsed.hostname
            port = parsed.port or _DEFAULT_PORTS.get((parsed.scheme or "").lower(), 443)
        except ValueError:
            row.update(status="skipped", reason="malformed URL")
            return row, findings, None
        scheme = (parsed.scheme or "").lower()
        if scheme not in _DEFAULT_PORTS or not host:
            row.update(status="skipped", reason=f"scheme {scheme or '(none)'!r} is not checked")
            return row, findings, None
        if "$" in host:
            row["host"] = host
            # Clients expand ${VAR} in MCP URLs at load time; the placeholder is not a host to resolve.
            row.update(status="skipped", reason="host contains an unexpanded variable")
            return row, findings, None
        # Look up the name the client connects to: 'internal%2eexample' is 'internal.example'.
        host = endpoint_client_host(host)
        row["host"] = host
        if not host:
            row.update(status="skipped", reason="URL has no host")
            return row, findings, None
        static = classify_endpoint_host(host)
        if static is not None:
            # The static policy already reported this host; never contact it.
            row.update(status="static_non_public", classification=static.kind, reason=static.reason)
            return row, findings, None
        try:
            addresses = self._lookup(host, port, run)
        except (OSError, UnicodeError) as exc:
            reason = "timed out" if isinstance(exc, TimeoutError) else type(exc).__name__
            row.update(status="unresolved", reason=f"DNS resolution failed ({reason})")
            findings.append(
                self._finding(
                    target,
                    Severity.LOW,
                    "endpoint_resolution_failed",
                    f"host {host!r} of {display_url!r} could not be resolved ({reason}); the endpoint was not checked",
                    "Check the host name, or rerun --resolve-endpoints where DNS is reachable.",
                )
            )
            return row, findings, None
        row["addresses"] = addresses[:MAX_ADDRESSES]
        if len(addresses) > MAX_ADDRESSES:
            row["addresses_not_classified"] = len(addresses) - MAX_ADDRESSES
        non_public = _classify_addresses(addresses)
        kind = _worst_kind(non_public)
        row["classification"] = kind
        if kind != "public":
            row["status"] = kind
            blocked = blocked_address(host, non_public, target.allowed_hosts)
            if blocked is not None and blocked[0] == "metadata":
                findings.append(
                    self._finding(
                        target,
                        Severity.HIGH,
                        "endpoint_resolves_metadata",
                        f"host {host!r} of {display_url!r} resolves to a cloud instance-metadata address "
                        f"({blocked[2]})",
                        "Remove the endpoint; names that resolve to instance-metadata services are never allowed.",
                    )
                )
            elif blocked is not None:
                findings.append(
                    self._finding(
                        target,
                        Severity.MEDIUM,
                        "endpoint_resolves_private",
                        f"host {host!r} of {display_url!r} resolves to {_where(blocked)}"
                        + (
                            f" ({blocked[2]}); the public-looking name targets a non-public network"
                            if blocked[2]
                            else ""
                        )
                        + ("" if blocked[2] else "; it is treated as non-public and was not contacted"),
                        "Use a public endpoint, or allow the intended host through the validation policy.",
                    )
                )
            row["head"] = {"skipped": "host resolves to a non-public address; it was not contacted"}
            return row, findings, None
        if not addresses:
            row.update(status="unresolved", reason="no addresses")
            return row, findings, None
        row["status"] = "resolved"
        return row, findings, _Pending(target, row, scheme, host, port, addresses[0], parsed.path or "/")

    def _head_and_redirect(self, item: _Pending, run: _Run) -> list[Finding]:
        """Send the one ``HEAD`` for a public endpoint within the remaining budget, then classify its redirect."""
        row = item.row
        remaining = self._remaining(run)
        if remaining <= 0:
            row.update(status="skipped", reason=f"time budget of {self.budget:.0f}s exhausted before the HEAD request")
            run.budget_skipped += 1
            run.unchecked.append(item.target)
            return []
        try:
            head = self.head(
                item.scheme, item.host, item.port, item.address, item.path, min(self.head_timeout, remaining)
            )
        except (OSError, ssl.SSLError, http.client.HTTPException, ValueError) as exc:
            head = HeadResult(None, None, f"{type(exc).__name__}: {str(exc)[:160]}")
        row["head"] = {"status": head.status, "location": _safe_url(head.location) if head.location else None}
        if head.error:
            row["head"]["error"] = head.error
            row["status"] = "head_failed"
            return []
        row["status"] = "public"
        if head.location:
            return self._check_redirect(item.target, row, item.target.url, head.location, item.scheme, run)
        return []

    def _check_redirect(
        self, target: EndpointTarget, row: dict[str, Any], url: str, location: str, scheme: str, run: _Run
    ) -> list[Finding]:
        findings: list[Finding] = []
        base = whatwg_url(url).split("?", 1)[0].split("#", 1)[0]
        # Resolve the Location the way a WHATWG client (fetch) follows it, so '\\' or 'http:host' cannot hide it.
        absolute = whatwg_url(location, base)
        display = _safe_url(absolute)
        redirect: dict[str, Any] = {"url": display}
        row["redirect"] = redirect
        try:
            parsed = urlparse(absolute)
            host = parsed.hostname
            port = parsed.port or _DEFAULT_PORTS.get((parsed.scheme or "").lower(), 443)
        except ValueError:
            redirect["classification"] = "malformed"
            return findings
        new_scheme = (parsed.scheme or "").lower()
        if scheme in {"https", "wss"} and new_scheme in {"http", "ws"}:
            redirect["downgrade"] = True
            findings.append(
                self._finding(
                    target,
                    Severity.MEDIUM,
                    "endpoint_redirect_insecure_scheme",
                    f"{_safe_url(url)!r} redirects from {scheme} to plaintext {new_scheme} ({display!r})",
                    "Serve the endpoint over https without a downgrade redirect.",
                )
            )
        # Look up the name the client follows: 'internal%2eexample' is 'internal.example'.
        host = endpoint_client_host(host) if host else ""
        if not host:
            redirect["classification"] = "no_host"
            return findings
        static = classify_endpoint_host(host)
        if static is not None:
            non_public = [(static.kind, static.reason, str(static.address) if static.address else None)]
        elif self._remaining(run) <= 0:
            redirect["classification"] = "skipped"
            redirect["reason"] = f"time budget of {self.budget:.0f}s exhausted before the redirect target was resolved"
            run.budget_skipped += 1
            run.unchecked.append(target)
            return findings
        else:
            try:
                addresses = self._lookup(host, port, run)
            except (OSError, UnicodeError):
                redirect["classification"] = "unresolved"
                return findings
            redirect["addresses"] = addresses[:MAX_ADDRESSES]
            non_public = _classify_addresses(addresses)
        kind = _worst_kind(non_public)
        redirect["classification"] = kind
        blocked = blocked_address(host, non_public, target.allowed_hosts)
        if kind == "metadata":
            findings.append(
                self._finding(
                    target,
                    Severity.HIGH,
                    "endpoint_redirect_metadata",
                    f"{_safe_url(url)!r} redirects to a cloud instance-metadata endpoint ({display!r})",
                    "Remove the endpoint; a redirect to an instance-metadata service is never allowed.",
                )
            )
        elif blocked is not None:
            findings.append(
                self._finding(
                    target,
                    Severity.MEDIUM,
                    "endpoint_redirect_private",
                    f"{_safe_url(url)!r} redirects to {_where(blocked)} ({display!r})",
                    "Use an endpoint that does not redirect into a non-public network, or allow the host in the "
                    "validation policy.",
                )
            )
        return findings
