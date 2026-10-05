# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URL reading and inline-credential checks shared by the MCP, hook, and endpoint policies.

Node (Claude Code http hooks, MCP SDK fetch), Rust's url crate, and browsers read
URLs with the WHATWG URL Standard, and urllib.parse does not always agree:
:func:`whatwg_url` gives the URL such a client connects to, and
:func:`url_ambiguities` says why the two readings could differ.

The credential predicates decide whether a keyed value (an env entry, a header,
a command-line flag, a URL query parameter) is an inline credential rather than
a ``$VAR`` / ``${VAR}`` reference.

Nothing here touches the network.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urljoin

# --------------------------------------------------------------------------- #
# Inline credentials                                                          #
# --------------------------------------------------------------------------- #
# env-var reference forms that are acceptable in place of an inline secret.
_ENV_REF_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$|^\$[A-Za-z_][A-Za-z0-9_]*$")
# env keys that name a credential -- their value must be a reference, never a literal.
# The auth/bearer/token alternatives are suffix-anchored so benign config keys that
# merely contain those substrings -- AUTH_TYPE, OAUTH_CLIENT_ID, BEARER_FORMAT,
# TOKEN_ENDPOINT, TOKEN_TYPE, TOKEN_ISSUER -- are not misread as credentials, while
# real credential keys (CLIENT_SECRET, AUTH_TOKEN, ACCESS_TOKEN, TOKEN_SECRET) match.
_SECRET_KEY_RE = re.compile(
    r"(?i)(secret|password|passwd|api[_-]?key|access[_-]?key|private[_-]?key|credential"
    r"|bearer[_-]?token|auth[_-](?:key|token|secret|pass(?:word)?)"
    r"|token(?:[_-](?:secret|key|value|id))?$)"
)
# Inline HTTP auth-scheme credential carried in a value (e.g. an Authorization
# header): "Bearer <token>" / "Basic <base64>" with a real payload. Anchored with a
# minimum payload length so a "${ENV}" reference or benign prose never matches; this
# keeps Authorization-style inline secrets covered without keying on the header name.
_INLINE_AUTH_SCHEME_RE = re.compile(r"(?i)^(?:bearer|basic)\s+[A-Za-z0-9+/._=~-]{12,}$")
# Known inline-secret value shapes. Only ``search`` truthiness is used.
#
# The JWT-like alternative starts only where a run of token characters starts
# and scans to the run's first ``eyJ`` without ever stepping past one. A later
# ``eyJ`` in the same run has fewer characters before the run ends, so it can
# never match when the first one does not. A plain ``eyJ...`` alternative was
# tried at every ``eyJ`` and scanned to the end of the run each time, which is
# quadratic on a long ``eyJeyJ...`` value (about 1 s per 64 KB value).
_SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9]{16,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|glpat-[A-Za-z0-9_-]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|nvapi-[A-Za-z0-9_-]{16,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|(?<![A-Za-z0-9_-])(?:(?!eyJ)[A-Za-z0-9_-])*eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"
)


def is_env_reference(value: str) -> bool:
    """True when *value* is an ``$VAR`` / ``${VAR}`` env reference (not a literal)."""
    return bool(_ENV_REF_RE.match(value.strip()))


def is_credential_name(name: str) -> bool:
    """True when an env key, header, flag, or query parameter name names a credential (``API_KEY``, ``password``)."""
    return _SECRET_KEY_RE.search(name) is not None


def has_secret_shape(value: str) -> bool:
    """True when a literal value is shaped like a known token or key, or is an inline ``Bearer``/``Basic`` credential.

    A ``$VAR`` / ``${VAR}`` reference never is.
    """
    text = value.strip()
    if not text or is_env_reference(text):
        return False
    return _SECRET_VALUE_RE.search(text) is not None or _INLINE_AUTH_SCHEME_RE.match(text) is not None


def looks_like_inline_secret(key: str, value: str) -> bool:
    """True when a keyed value (env entry, header) is an inline credential rather than a reference.

    A value shaped like a secret counts under any key, including an inline
    ``Bearer``/``Basic`` credential (so Authorization-style headers are covered
    without keying on the header name); any other non-empty literal counts
    under a credential-named key.
    """
    text = value.strip()
    if not text or is_env_reference(text):
        return False
    return has_secret_shape(text) or is_credential_name(str(key))


# --------------------------------------------------------------------------- #
# WHATWG URL reading                                                          #
# --------------------------------------------------------------------------- #
# Node (Claude Code http hooks, MCP SDK fetch), Rust's url crate, and browsers
# parse URLs with the WHATWG URL Standard. For its "special" schemes it reads a
# backslash as '/', skips any run of slashes after 'scheme:', and drops tabs and
# line breaks, where urllib.parse does not. Policy decisions must read the URL
# the way the client that connects to it does.
_WHATWG_SPECIAL_SCHEMES = frozenset({"http", "https", "ws", "wss", "ftp"})
_C0_CONTROL_OR_SPACE = "".join(chr(code) for code in range(0x21))
# Edge characters that both urllib (str.strip) and WHATWG clients (C0 control or space) drop.
_ASCII_EDGE_WHITESPACE = "".join(char for char in _C0_CONTROL_OR_SPACE if char.isspace())
_TAB_OR_NEWLINE_RE = re.compile(r"[\t\n\r]")
_URL_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")


def _special_scheme_slashes(text: str) -> str:
    """Read ``\\`` as ``/`` before the query or fragment, as WHATWG does for special schemes."""
    cut = min((index for index in (text.find("?"), text.find("#")) if index != -1), default=len(text))
    return text[:cut].replace("\\", "/") + text[cut:]


def _url_scheme(text: str) -> str | None:
    match = _URL_SCHEME_RE.match(text)
    return match.group(1).lower() if match else None


def whatwg_url(url: str, base: str | None = None) -> str:
    """The absolute URL a WHATWG client resolves ``url`` (against ``base``) to, in a form urllib reads the same way.

    For http(s), ws(s), and ftp URLs the result has ``scheme://`` followed by the
    authority the client connects to, so ``urlsplit(result).hostname`` is the host
    Node would use: ``https://evil.net\\.example.com/`` reads as host ``evil.net``,
    and ``http:169.254.169.254/`` as host ``169.254.169.254``. Other schemes, and a
    relative URL without a base, come back with only the WHATWG trimming applied.
    """
    text = _TAB_OR_NEWLINE_RE.sub("", url.strip(_C0_CONTROL_OR_SPACE))
    base_url = whatwg_url(base) if base is not None else None
    base_scheme = _url_scheme(base_url) if base_url is not None else None
    scheme = _url_scheme(text)
    if scheme is not None:
        if scheme not in _WHATWG_SPECIAL_SCHEMES:
            return text
        rest = _special_scheme_slashes(text[len(scheme) + 1 :])
        if base_url is not None and scheme == base_scheme and not rest.startswith("//"):
            # Same special scheme as the base and no authority: a relative reference.
            return urljoin(base_url, rest)
        # Any run of '/' and '\' after a special scheme introduces the authority.
        return f"{scheme}://{rest.lstrip('/')}"
    if base_url is None:
        return text
    if base_scheme in _WHATWG_SPECIAL_SCHEMES:
        text = _special_scheme_slashes(text)
        if text.startswith("//"):
            # Any run of '/' and '\' starts the authority; urljoin would keep the base host for '///host'.
            return f"{base_scheme}://{text.lstrip('/')}"
    return urljoin(base_url, text)


def url_ambiguities(url: str, *, percent_in_host: bool = False) -> list[str]:
    """Why urllib and a WHATWG client (Node, MCP SDKs) could read ``url`` differently, or as different text.

    Flags whitespace or control characters inside the URL (only leading and
    trailing ASCII whitespace is allowed, which both readings strip; a NUL,
    U+2028, or no-break space at either end still counts), and invisible
    format characters such as a zero-width space anywhere (WHATWG drops some of
    them from a host name, urllib keeps them). For http(s), ws(s), and ftp URLs
    it also flags a backslash before the query, a scheme not followed by
    exactly ``//`` (WHATWG skips any run of slashes, so ``https:///host``
    connects to ``host``), and (with ``percent_in_host``) percent-encoding in
    the host, which WHATWG decodes before it connects.
    """
    text = url.strip()
    problems: list[str] = []
    inner = url.strip(_ASCII_EDGE_WHITESPACE)
    if any(char.isspace() or unicodedata.category(char) == "Cc" for char in inner):
        problems.append("whitespace or a control character")
    if any(unicodedata.category(char) == "Cf" for char in url):
        problems.append("an invisible format character (such as a zero-width space)")
    trimmed = _TAB_OR_NEWLINE_RE.sub("", text.strip(_C0_CONTROL_OR_SPACE))
    scheme = _url_scheme(trimmed)
    if scheme not in _WHATWG_SPECIAL_SCHEMES:
        return problems
    rest = trimmed[len(scheme) + 1 :]
    cut = min((index for index in (rest.find("?"), rest.find("#")) if index != -1), default=len(rest))
    if "\\" in rest[:cut]:
        problems.append("a backslash, which clients read as '/'")
    if not rest.startswith("//"):
        problems.append(f"no '//' after '{scheme}:'")
    elif rest[2:3] in {"/", "\\"}:
        problems.append(f"more than two slashes after '{scheme}:'")
    if percent_in_host:
        authority = re.split(r"[/\\?#]", rest.lstrip("/\\"), maxsplit=1)[0]
        if "%" in authority.rpartition("@")[2]:
            problems.append("percent-encoding in the host")
    return problems
