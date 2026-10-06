# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URL reading, credential checks, and report display shared by the MCP, hook, and endpoint policies.

Node (Claude Code http hooks, MCP SDK fetch), Rust's url crate, and browsers read
URLs with the WHATWG URL Standard, and urllib.parse does not always agree:
:func:`whatwg_url` gives the URL such a client connects to, and
:func:`url_ambiguities` says why the two readings could differ.

The credential predicates decide whether a keyed value (an env entry, a header,
a command-line flag, a URL query parameter) is an inline credential rather than
a ``$VAR`` / ``${VAR}`` reference; :func:`url_credentials` applies them to a URL.
:func:`safe_url` and :func:`report_text` are how URLs and other plugin text
appear in findings and reports: bounded, with credentials removed.

Nothing here touches the network.
"""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Literal
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from skillevaluator.utils.redaction import redact_sensitive_text

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
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{22,}"
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


# Where the authority of a URL's raw text ends (a backslash does not end it there).
_RAW_AUTHORITY_END_RE = re.compile(r"[/?#]")


@dataclass(frozen=True)
class UrlCredentials:
    """Where the text of one URL carries credentials; false when it carries none."""

    # The userinfo ('user:password@') carries one.
    userinfo: bool = False
    # Query parameters whose value is one, in the order they appear.
    query_keys: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.userinfo or bool(self.query_keys)


# Which userinfo counts as a credential; see ``url_credentials``.
UserinfoRule = Literal["any", "literal", "secret"]


def url_credentials(url: str, *, userinfo_rule: UserinfoRule) -> UrlCredentials:
    """Credentials written into the text of ``url``: in its userinfo and in its query parameters.

    The URL is read as raw text, so a malformed port or bracket cannot hide a
    credential: the authority runs from ``//`` to the next ``/``, ``?``, or
    ``#``, and the userinfo through its last ``@``.

    * Userinfo, by ``userinfo_rule``:

      - ``"any"``: every user name or password counts, even a ``$VAR`` one. An
        HTTP hook's client sends the userinfo as Basic auth with every request.
      - ``"literal"``: a user name or password counts unless it is a ``$VAR`` /
        ``${VAR}`` reference (MCP server URLs, where references are allowed).
      - ``"secret"``: only a literal password counts, or a user name shaped like
        a token (a URL inside a command line), so
        ``https://x-access-token:${GITHUB_TOKEN}@github.com/...`` does not.
    * Query: a parameter counts when it has a literal value under a credential
      name (``api_key=literal``) or a value shaped like a secret under any name
      (``q=sk-...``). A ``$VAR`` / ``${VAR}`` reference never counts.
    """
    authority = _RAW_AUTHORITY_END_RE.split(url.partition("//")[2], maxsplit=1)[0]
    userinfo, at, _host = authority.rpartition("@")
    user, _colon, password = userinfo.partition(":")
    query = url.partition("?")[2].partition("#")[0]
    return UrlCredentials(
        userinfo=bool(at) and _userinfo_carries_credential(user, password, rule=userinfo_rule),
        query_keys=tuple(
            dict.fromkeys(
                key
                for key, values in parse_qs(query, keep_blank_values=True).items()
                if any(_query_value_is_credential(key, value) for value in values)
            )
        ),
    )


def _userinfo_carries_credential(user: str, password: str, *, rule: UserinfoRule) -> bool:
    if rule == "any":
        return bool(user or password)
    if rule == "literal":
        return any(part and not is_env_reference(part) for part in (user, password))
    decoded = unquote(password)
    literal_password = not password.startswith("$") and bool(decoded.strip()) and not is_env_reference(decoded)
    return literal_password or has_secret_shape(unquote(user))


def _query_value_is_credential(key: str, value: str) -> bool:
    if not value or is_env_reference(value):
        return False
    return is_credential_name(key) or has_secret_shape(value)


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
# Where the authority of a special-scheme URL ends, as WHATWG clients read it.
_AUTHORITY_END_RE = re.compile(r"[/\\?#]")
# The port each scheme an MCP server or HTTP hook uses connects to when the URL names none.
DEFAULT_PORTS: dict[str, int] = {"http": 80, "ws": 80, "https": 443, "wss": 443}


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
        authority = _AUTHORITY_END_RE.split(rest.lstrip("/\\"), maxsplit=1)[0]
        if "%" in authority.rpartition("@")[2]:
            problems.append("percent-encoding in the host")
    return problems


# --------------------------------------------------------------------------- #
# URLs and other text in reports                                              #
# --------------------------------------------------------------------------- #
# Longest text (a URL, a command line) one report field shows.
MAX_REPORT_CHARS = 200
# 'user:password@' in a URL authority (through its last '@'); scrubbed from every text a report shows.
_URL_USERINFO_RE = re.compile(r"//[^/?#\s]*@")
# The characters tokens are written with. A token that the redaction window cuts can
# be too short for its pattern to match, so report_text drops the cut token.
_TOKEN_CHARACTERS = string.ascii_letters + string.digits + "_-.~+/="
# A private-key header that redact_sensitive_text did not read as the start of a key block.
_PRIVATE_KEY_HEADER_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_TRUNCATED = "...<truncated>"


def report_text(value: str, limit: int = MAX_REPORT_CHARS) -> str:
    """Report text: whitespace collapsed, URL ``user:password@`` removed, credentials redacted, length bounded.

    Userinfo is removed from the whole text; only a window of twice the limit is
    redacted (the result keeps at most ``limit`` characters), because the
    redaction patterns can take quadratic time on long unbroken input. A token
    that the window cuts is dropped, and a cut text ends in ``...<truncated>``.
    ``redact_sensitive_text`` runs first, so a private key is redacted whole,
    BEGIN line to END line, and so is a JWT; then every other secret shape the
    inline-credential checks know (``ghp_…``, ``glpat-…``, ``xoxe-…``, ...) is.
    """
    text = _URL_USERINFO_RE.sub("//", " ".join(value.split()))
    window = text[: 2 * limit]
    cut = len(window) < len(text)
    if cut:
        window = window.rstrip(_TOKEN_CHARACTERS)
    shown = redact_sensitive_text(window)
    header = _PRIVATE_KEY_HEADER_RE.search(shown)
    if header is not None:
        # A key whose BEGIN line redact_sensitive_text could not read: withhold the rest of the text.
        shown = f"{shown[: header.start()]}private-key-<redacted>"
    shown = _SECRET_VALUE_RE.sub("<redacted>", shown)
    if cut or len(shown) > limit:
        return shown[: limit - len(_TRUNCATED)] + _TRUNCATED if limit > len(_TRUNCATED) else shown[:limit]
    return shown


def safe_url(url: str) -> str:
    """A URL for reports: no userinfo, query, or fragment; bounded (also for a URL that does not parse).

    An http(s), ws(s), or ftp URL is shown the way a WHATWG client (Node, the MCP
    SDKs) reads it, so the report names the host a client would actually contact.
    A token in the path is redacted like any other report text (:func:`report_text`).
    """
    try:
        parsed = urlparse(whatwg_url(url))
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
    except ValueError:
        return _unparsed_url(url)
    if not parsed.scheme or not host:
        return _unparsed_url(url)
    display_host = f"[{host}]" if ":" in host else host
    return report_text(f"{parsed.scheme}://{display_host}{port}{parsed.path}")


def _unparsed_url(url: str) -> str:
    """A malformed URL for reports: the query, fragment, and userinfo (through the authority's last ``@``) removed."""
    text = url.strip().split("#", 1)[0].split("?", 1)[0]
    prefix, slashes, rest = text.partition("//")
    if not slashes:
        prefix, rest = "", text
    authority, slash, path = rest.partition("/")
    return report_text(f"{prefix}{slashes}{authority.rpartition('@')[2]}{slash}{path}")
